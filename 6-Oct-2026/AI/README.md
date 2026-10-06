# BiasFlow: Geometric Monitoring and Backbone Regularization for Spurious Feature Reliance 

**Authors**: Haojin Deng, Zhiping Lin, Yimin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06846)  

**Abstract**: Worst-group accuracy (WGA) evaluates a trained predictor but does not characterize how its frozen backbone behaves when a new head is learned. We introduce BiasFlow, a hook-based toolkit for monitoring class-attribute centroid alignment (IBMI), within-class centroid separation (W-IBMI), and feature-projection sensitivity. IBMI is confounded by class-attribute correlation and is not a measure of causal feature reliance. We pair these diagnostics with BiasFlow Regularization (BFR), a supervised, composable class-conditional centroid-alignment penalty. W-IBMI verifies the quantity BFR optimizes; it is scale dependent and does not independently establish attribute removal. Across the reported small-scale benchmarks, adding BFR improves or preserves mean WGA, with gains up to +26.0 pp on UrbanCars. The principal independent stress test freezes CelebA-Std backbones and trains fresh heads on biased data: BFR+GroupDRO improves WGA from 40.7% to 64.1%, while Male probe accuracy decreases from 92.5% to 72.2%. Attribute information remains recoverable, and cross-task results are mixed. A controlled synthetic-watermark ImageNet experiment additionally improves watermark-shift accuracy by +23.0 pp under matched training. These results support evaluating centroid geometry and resistance to biased head retraining alongside WGA, within the tested protocols. 

---
# TasteVal: Measuring the Experimental Research Taste of AI Systems Against Human Experts 

**Authors**: Oliver Jaffe, Dane Sherburn  

**Link**: [PDF](https://arxiv.org/pdf/2610.06824)  

**Abstract**: We introduce TasteVal, a benchmark to evaluate the experimental research taste of frontier models. We define research taste as the ability to pick interesting problems to solve, design experiments, and interpret experimental results. TasteVal measures the experimental component of research taste; given a fixed research problem, we measure how well a model iteratively designs experiments and draws conclusions from their outcomes. We operationalize experimental research taste as compute efficiency; a Researcher who reaches the same score as an expert human using half the serial experimental compute has twice the experimental taste. Experimental taste thus acts as a multiplier on experimental compute, making it a key input to forecasts of AI progress. TasteVal consists of 8 novel, challenging, open-ended tasks representative of frontier AI R&D. To isolate taste from coding ability, the model under evaluation acts as a Researcher that iteratively designs experiments while a fixed Coder agent implements them and reports their results. The Researcher executes until either the 40 H100 hour or 120 wall-clock hour budgets are exhausted. We recruit 24 human experts, at least 2 per task, and take the best expert attempt per task as the expert baseline. We evaluate 20 models released between 2023 and 2026. The best-performing model, Opus 5.5, exceeds our expert baseline, with a compute multiplier of 2.3x (95% CI 1.15-4.37), at roughly 1/30 of our baseliners' average per-run cost. On TasteVal, the compute multiplier of frontier models has doubled approximately every 3.0 months since December 2025 (95% CI 1.7-5.0), up from every 14 months between 2023 and December 2025. Measured by final normalized performance, frontier models show no trend break, doubling every 14.6 months. To keep TasteVal uncontaminated, we do not release the tasks. 

---
# Back to the Future: Rethinking EDA Infrastructure for Agentic Systems in Chip Design Verification 

**Authors**: Je Yang, Ivan Lobov, Thomas Karpati  

**Link**: [PDF](https://arxiv.org/pdf/2610.06790)  

**Abstract**: The unprecedented computational scale of modern artificial intelligence depends on complex, multi-billion-transistor Systems-on-Chip, yet the workflows that verify these chips remain stubbornly manual. Although Large Language Models (LLMs) have made rapid inroads into Electronic Design Automation (EDA), approximately 74.6% of existing studies target static Register-Transfer Level (RTL) code generation, leaving post-simulation verification and interactive waveform debugging largely untouched. We introduce Back-to-the-Future (BTTF), an end-to-end agentic framework that closes this infrastructural gap. BTTF distills massive, unstructured simulation dumps into a normalized relational SQLite database and couples it with a collaborative multi-agent orchestration engine that translates natural-language verification queries into schema-aware SQL while correlating signal anomalies with versioned RTL repositories. Across a 150-query benchmark, BTTF attains 95.33% execution accuracy, charting a practical path toward autonomous EDA verification. 

---
# Conditional Rank Allocation for Taxonomy-Aware Medical Language Model Adaptation 

**Authors**: Guangyuan Dong, Ziwei Hong, Xuehao Zhou, Zidong Yu, Bingchen Liu, Kehan Liu, Chuang Liu, Rong Fu, Yuchao Hou  

**Link**: [PDF](https://arxiv.org/pdf/2610.06765)  

**Abstract**: Medical question answering spans specialties and clinical operations that may benefit from different adaptation directions. We propose ARBOR, a parameter-efficient method that selects rank-one components from a shared low-rank basis for each question. An additive gate combines question representations, specialty tags, operation tags, and their interaction; a learned coefficient scales the adapter residual. An illustrative separation under orthogonal, equiprobable subtasks shows how conditional selection can avoid an approximation floor faced by a fixed update with the same active rank. This result motivates the design without asserting a corresponding bound for medical corpora. On Qwen3-8B across CMB, CMExam, MedQA, and MedMCQA, five-seed experiments yield 69.69% mean accuracy across benchmarks, exceeding LoRA r16 and MoELoRA by 1.26 and 1.30 percentage points, respectively. The reported advantage over LoRA r16 increases from 0.08 to 1.94 points as training expands from one to seven specialties. Tag perturbations and atom masking support the usefulness of clinical routing, while atom clusters align with the supplied specialty labels (adjusted Rand index 0.62). Calibration, transfer, and measured costs further characterize the method. These findings support structured conditional adaptation for medical QA, while leaving clinical safety and broader deployment untested. 

---
# The Pushback Paradox: A Two-Probe Diagnostic for Language Model Compliance 

**Authors**: Stefan Bühler, David Exler, Markus Reischl, Mark Schutera  

**Link**: [PDF](https://arxiv.org/pdf/2610.06673)  

**Abstract**: Are language models compliant with user instructions? A model that always complies can be stopped but also exploited, while one that always resists can be neither exploited nor stopped. We contribute an open two-probe benchmark that can place any language model on this spectrum. In the active probe, a user instructs the model to act and accept a lower payoff, which measures exploitability. In the passive probe, the user instructs it to wait and give up a higher payoff, which measures stoppability. The two compliance rates combine into a compliance index $\kappa$. Applied to twelve language models, the benchmark shows that seven mostly follow the instruction in both probes and justify their action by pointing to the instruction. Only Claude Sonnet-4.6 and Claude Opus-4.7 can be stopped without being exploitable, Claude Opus-4.6 and GPT-5-mini resist both instructions, and no model is exploitable but unstoppable. Knowing where a language model sits on the compliance index $\kappa$ matters for human operators and for multi-agent systems, whether distributed or orchestrated. 

---
# Language models can notice an impossible engineering problem yet still report it as solved 

**Authors**: Shaoliang Yang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06668)  

**Abstract**: Language models draft engineering calculations, but answer accuracy does not show whether they reject an impossible problem. We tested 14 models on 30 pairs of mechanics problems, each with a valid version and one made impossible by changing a given value or assumption. Two independent solvers verified every answer key and showed that each flawed problem was physically impossible. We scored solving of valid problems separately from rejection of their flawed counterparts. Each reply required a "solved" or "cannot solve" status; rejection meant "cannot solve" or withholding an answer. The initial prompts did not warn that problems could be flawed. Across three recent models, 12 of 90 replies failed to reject a flawed problem. In 11 of these replies, the model stated the flaw, answered a corrected problem and still reported the original as "solved", according to artificial intelligence raters and numerical checks. We later retested four models from one provider, offering "flawed" instead of "cannot solve" and asking them to name and explain the defect. Three models showed statistically significant increases in rejection, but valid-problem solving fell in three. Evaluations therefore need to score both versions and distinguish flaw recognition from the reported status. 

---
# Collective intelligence through aggregation 

**Authors**: Franz Dietrich, Christian List  

**Link**: [PDF](https://arxiv.org/pdf/2610.06652)  

**Abstract**: Suppose a committee, expert panel, or other group is making judgments on some issues, where these may be not just yes/no-questions, such as whether a defendant is guilty, but also variables with many possible values, such as macroeconomic or meteorological variables or travel directions. Furthermore, there may be interconnections between different issues, as in the case of economic or climate variables. How can the group arrive at "intelligent" collective judgments, based on the group members' individual judgments? We investigate three challenges raised by this judgment-aggregation problem. First, reasonable methods of aggregation (such as defining the collective judgment for each issue as the average or median judgment) can produce inconsistent collective judgments. Secondly, many methods of aggregation are manipulable by strategic voting. Finally, not all methods of aggregation are conducive to tracking the truth on the issues in question. We prove new impossibility or possibility theorems on all three challenges, identifying what it takes to produce collective judgments in a consistent, non-manipulable, and truth-tracking manner and thereby to achieve collective intelligence through aggregation. Overall, the median method, though imperfect, performs reasonably well. We also note the relevance of our analysis for non-human group decisions. 

---
# FREA: A Multi-Source Expert Benchmark for Reaction Feasibility Verification 

**Authors**: Botao Yu, Bo Zhou, Daniel Adu-Ampratwum, Frazier N. Baker, Ziru Chen, Reza Averly, Ye Liu, Wenhao Gao, Xia Ning, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.06614)  

**Abstract**: As generative models and AI agents propose chemical reactions at a scale beyond expert review, feasibility verifiers decide which proposals enter synthesis planning. But do their decisions agree with chemists across different kinds of candidates? We introduce FREA, a benchmark of 751 reactions labeled by expert chemists under an explicit feasibility criterion, drawn from retrosynthesis model proposals, zero-yield experimental records, edits by large language models (LLMs), and five negative candidate generation methods. Our evaluation finds that no verifier leads across all sources: LLMs given only the criterion are competitive with dedicated verifiers, while forward models perform best on retrosynthesis proposals but reject most feasible edits of recorded reactions at the evaluated operating points. Looking beyond aggregate scores, both forward models perform below chance when separating infeasible alternative disconnections from feasible generated candidates. To study whether negative supervision addresses these weaknesses, we also release a corpus of over 14 million recorded reactions and generated negative candidates. In matched training comparisons, adding a mixture of generated negatives to forward training raises mean AUROC across sources, but these gains do not extend to retrosynthesis proposals. Varying the generation method further shows that the largest gain on generated candidates coincides with worse proposal screening. These findings motivate evaluating verifiers against experts across sources and designing negatives for transfer to model proposals. 

---
# Can Agent Harnesses and Inference Engines Hear Each Other? The HEAR Protocol for Agentic LLM Serving 

**Authors**: Jiaqi Zhao, Haodong Chen, Jitai Hao, Wei Zhao, Jinghao Pang, Qiang Huang, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06597)  

**Abstract**: LLM agents increasingly execute complex workflows involving multi-turn reasoning, tool use, and parallel agents. Efficient serving requires decisions that span two layers with complementary information: the agent harness understands workflow dependencies, context lifecycles, and execution objectives, whereas the inference engine observes request queues, KV-cache state, resource pressure, and execution capabilities. Existing interfaces do not systematically connect these views, limiting workflow-aware execution.
HEAR, a bidirectional Harness--Engine Pairing protocol for agentic LLM serving. HEAR standardizes how the harness communicates workflow intent and execution requirements and how the engine returns runtime state, capabilities, and outcomes. By separating protocol semantics from optimization policies, HEAR supports diverse coordination strategies without changing workflow or model semantics. We instantiate HEAR for online cache-aware runtime coordination and workload-aware execution-mode selection for agent roles.
Across four conversational and research-agent benchmarks under memory-constrained, concurrent serving, HEAR achieves a $1.61\times$ batch speedup and reduces median time-to-first-token by $2.23\times$ on SCBench. Mooncake shows that workflow intent and live engine state provide complementary benefits across load regimes. On BrowseComp-Plus and DeepResearchBench, workload-specific configurations yield $1.23\times$ and $2.45\times$ end-to-end speedups, respectively, without observed task-quality degradation. These results establish HEAR as a reusable coordination substrate for efficient agentic LLM serving. 

---
# The Review Lottery: Calibrating an Observational Estimator of Peer-Review Noise (ICLR 2017-2025) 

**Authors**: Feilian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06591)  

**Abstract**: How much of a conference accept/reject decision would change if the same paper were reviewed by a different set of reviewers? Running a second independent program committee is the gold standard for answering this, but it is prohibitively expensive: done only twice (NeurIPS 2014 and 2021). We build an observational estimator of this quantity from public review data alone, calibrate it twice, and apply it to nine years of ICLR (2017-2025; 36,113 papers, 134,912 reviews). The estimator decomposes scores with a Bayesian ordered-probit model into paper quality and reviewer noise, maps scores to decisions with a logistic model, and simulates two independent committees (posterior draws B=1,000; committee sizes k=2,3,4). Estimated disagreement rates are 23-30% at k=2 and 18-24% at k=4; 30-50% of accepted papers would be rejected. External calibration: at the NeurIPS 2021 reviewer-count caliber (k=3), the simulated 2021 disagreement rate is 23.3% [21.7%, 25.0%] vs. reported 23.0% (bias +0.3pp); accept precision and committee correlation agree within 5pp and 0.04. Internal calibration: on 18,740 papers with 4+ reviews, random model-free 2+2 reviewer splits agree with the k=2 simulation within 1pp in 2018 and 2021-2025. Longitudinally, we find no robust time trend in reviewer noise over 2017-2025. The high accepted-paper flip rates of 2020 and 2021 have distinct mechanisms: the 2020 four-point scale compressed scores (23.7% of papers had zero within-paper variance), and a counterfactual shows coarsening the scale raises disagreement by about 7pp; 2021 instead combined the lowest signal-to-noise ratio in the sample with the most threshold-crowded acceptances. For the LLM era, a 2023 breakpoint test on within-paper score variance finds no break, but the design has almost no power, and no post-2022 review text or confidence data exist, so no LLM attribution is attempted. 

---
# Mind the Execution Gap: Action-Semantic Mismatch in World-Model Control 

**Authors**: Shengtao Wen, Xiang Chen, Yu Tian, Lingbing Guo, Lina Gong, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06582)  

**Abstract**: World-model controllers rely on action-conditioned dynamics for prediction and planning, yet real control systems often execute commands asynchronously due to communication delay, packet loss, reordering, and actuator buffering. We study how asynchronous execution changes the action semantics assumed within world-model controllers, rather than treating it only as an external control disturbance. Through controlled interventions, we identify two architecture-dependent failure modes: planning-based controllers such as TD-MPC2 suffer from a future-action timeline mismatch between imagined and executed action sequences, while recurrent world models such as DreamerV3 can attribute observed transitions to commands that were not actually applied. Our analysis shows that TD-MPC2 requires the correct future action sequence during latent dynamics rollout, whereas DreamerV3 requires timely attribution of each transition to the action that generated it. Based on these findings, we introduce two lightweight execution-consistent interfaces, Future-Sequence for TD-MPC2 and Applied-Action Feedback for DreamerV3, that correct these mismatches without modifying the pretrained world models. Experiments across delays, packet loss, reordering, multiple control domains, measured network traces, and a process-separated asynchronous stack consistently support both diagnoses and the corresponding architecture-specific corrections. 

---
# DGA-Muon: Decoupled Geometry-Aligned Adaptive Scaling for Muon 

**Authors**: Wenpeng Zhang, Runsheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06578)  

**Abstract**: While NorMuon has achieved strong empirical performance in large-scale pretraining by enhancing Muon with row-wise adaptive scaling, its underlying adaptive mechanism remains poorly understood. In this work, we provide the first systematic analysis of NorMuon's adaptivity, revealing that it originates primarily from orthogonalization-induced geometry rather than genuine optimization-relevant information. Under exact orthogonalization, the adaptive scaling factors degenerate into a single global scalar for square and wide matrices, while for tall matrices their variation arises from the non-uniform distribution of row energy after orthogonalization. Under approximate orthogonalization, the orthogonality residual introduces additional variation into the scaling factors, leading to the \textit{Orthogonalization--Adaptivity Paradox}: more accurate orthogonalization weakens adaptivity. We further show that NorMuon's rigid row-wise scaling is geometrically misaligned with the one-sided orthogonal structure of tall matrices. Based on the analysis of these limitations, we propose two core design principles that a desirable adaptive mechanism for Muon should satisfy. First, adaptive scaling should be decoupled from orthogonalization, with the scaling factors computed directly from raw gradients. Second, adaptive scaling should be aligned with the shape-dependent orthogonal structure of the polar factor, using row-wise scaling for wide matrices and column-wise scaling for tall matrices. By incorporating several other design considerations, including sum-based second-moment estimates, bias correction, and adaptive clipping of scaling factors, we obtain the Decoupled Geometry-Aligned Muon (DGA-Muon) optimizer. We establish convergence guarantees for DGA-Muon and empirically validate both our theoretical characterization of NorMuon's scaling degeneration and the superiority of DGA-Muon over NorMuon. 

---
# HERA: Harness-Environment Co-Evolution for Reliable Agentic Abstention 

**Authors**: Han Luo, Bingbing Wen, Guang Yang, Zora Zhiruo Wang, Pan Lu, Lucy Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06563)  

**Abstract**: Large language model (LLM) agents are increasingly capable of acting in complex tool-use environments, yet they often fail to recognize when tasks are infeasible and no valid solution exists. Recent work has formalized this reliability gap as the problem of agentic abstention, and existing approaches typically optimize a model or agent harness against a fixed set of tasks, leading to limited generalization to unseen failure modes. We introduce HERA, a framework for harness-environment co-evolution for agentic abstention. HERA consists of (i) a pipeline to automatically construct verifiable pairs of feasible and infeasible tasks by applying controlled environment mutations that transform solvable tasks into cases requiring abstention, and (ii) a co-evolution procedure in which performance failures on previous tasks are used to drive harness adaptation and generate new execution environments and tasks geared towards previous weaknesses. On held-out evaluation tasks, an evolved harness from HERA improves abstention accuracy from 61.7% to 83.3% while improving feasible-task completion from 68.3% to 76.7%, achieving the highest abstention and feasible-task completion among the compared methods. The resulting best harness transfers across 19 other LLMs, improving abstention accuracy by 15.3 percentage points on average without any model-specific optimization, and enabling smaller models to match the performance of more powerful models at an estimated 85% lower cost. 

---
# Signature-Based Feature Learning for Human Activity Recognition: A Reproducible Machine Learning Study of Representation, Depth, and Model Choice 

**Authors**: Kamal Jarrar, Jacky Cresson, Christian Paroissin  

**Link**: [PDF](https://arxiv.org/pdf/2610.06553)  

**Abstract**: Human activity recognition (HAR) relies on transforming sensor signals into informative representations for classification. Although deep learning and handcrafted features are widely used, the role of representation itself is often not systematically isolated. Signature transforms provide a mathematically grounded way to encode temporal order and cross-channel interactions, but their value for HAR under a fully reproducible and leakage-aware framework remains unclear. To evaluate whether signature-based feature learning improves HAR performance compared with raw-signal baselines, and to assess the effects of embedding strategy, truncation depth, model choice, and sensor configuration. Experiments were conducted on the UCI HAR dataset using a fully reproducible pipeline with the original train--test split preserved and subject-disjoint validation to prevent leakage. Three representations were compared: raw flattened signals, time-augmented paths, and lead--lag transformed paths. Signature features were computed at multiple truncation depths and evaluated using multilayer perceptron (MLP) and Random Forest (RF) classifiers under identical preprocessing and validation procedures. A prior K-means-based feature reduction study was also reproduced for comparison. Signature-based representations improved performance when paired with RF models, the best configuration was time-augmented six-channel signatures at depth 6 using entropy-based RF achieving 0.858 accuracy and 0.859 macro F1, outperforming the strongest raw baseline (0.816 accuracy). Lead--lag representations were competitive at moderate depths but did not surpass the best time-augmented models. MLP models did not exceed raw baselines. Signature-based feature learning can improve HAR, but its benefit depends on alignment between representation design and classifier choice. 

---
# Proof-Grounded Patient-Specific Clinical Explanations from Knowledge-Graph Reasoning 

**Authors**: Surajit Das  

**Link**: [PDF](https://arxiv.org/pdf/2610.06549)  

**Abstract**: Clinical decision-support outputs can lack an au- ditable link between patient observations, encoded knowledge, conclusions, and recommendations. We present the CKG Clinical Explanation Engine, a downstream layer for a frozen, training-free clinical knowledge-graph reasoner that converts patient inference states and disease knowledge into typed facts, explicit rule-application traces, provenance-linked conclusions, and policy-licensed recommendations. The design separates measurement availability, representation completeness, and disease-specific activation; consequently, observed zero-activation evience is not treated as missing and partial representation is distinct from unobserved evidence. Optional language generation is restricted to symbolically licensed content. Across five usable workbooks (6,720 patients; 20,160 patient-disease traces; 1,021,440 feature-evidence rows), IG-range validity and knowledge provenance were 100%, numerical cross-sheet fidelity was 100% (120,960/120,960), and exported logical/report trace completeness was 100% (20,160/20,160). Availability representation consistency was 99.7028% (1,018,404/1,021,440); all 3,036 disagreements were confined to three systematic feature-cohort patterns. The corpus contained 86,783 observed zero-activation and 139,949 observed partially represented instances. A separate seeded 25-patient end-to-end audit completed without execution failure and passed all pre-specified trace, licensing, provenance, and state-consistency checks. These results establish structural and implementation auditability, not clinical correctness or utility. 

---
# ANT: A Multi-Granularity Network Traffic Dataset and Benchmark for Agents Behavior Auditing 

**Authors**: Fan Li, Xiangyu Gao, Zixuan Liu, Tong Li, Chuanpu Fu, Ziqiang Wang, Ke Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06514)  

**Abstract**: The growing adoption of large language model (LLM) agents creates a need for network administrators and security teams to audit agent behavior within organizational networks without inspecting private user content. Network traffic offers an observable source of evidence, but how much it reveals about agent tasks and operations remains unclear. Existing traffic datasets lack the joint task and stage annotations needed to evaluate this question. We introduce ANT (Agent Network Traffic), a dataset providing agent behavior information at risk, scenario, and behavior primitive granularities alongside network traffic. ANT contains 3,114 execution episodes across 20 tasks and five scenarios, comprising 276,417 bidirectional flows and 40,049 behavior primitive segments organized into 47 macro groups. We establish a benchmark for agent risk identification, scenario recognition, and behavior primitive classification using 13 representative traffic analysis baselines. The results show that existing methods recover useful but uneven behavioral signals. They struggle to identify risk when malicious workflows resemble benign tasks and to distinguish scenarios with similar traffic patterns. Primitive classification is more reliable for frequent macro groups and those with distinctive traffic patterns than for rare or semantically similar groups. ANT provides a common basis for developing more precise auditing and forensic analysis of agent behavior from network traffic. Our data and code are available at this https URL. 

---
# You Changed Your Mind, The Model Didn't: Demystifying Intent in Multi-Turn Dialogue 

**Authors**: Junle Chen, Wei Chen, Zhengjun Huang, Zhoujin Tian, Yuxuan Liu, Kai Wang, Rui Chen, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.06496)  

**Abstract**: When a large language model handles a multi-turn task and a user proposes a change but ultimately rejects it, the model should continue as if nothing changed. We find a surprising failure: merely mentioning a rejected change can derail task execution, even when the user's final intent remains unchanged. To systematically study language model behavior under evolving user intent, we introduce Intent-Eval, a controlled benchmark spanning tool actions, code, databases, and mathematics. Across diverse tasks, models are vulnerable to both rejected proposals and superseded requirements, consistent with mentioned-as-in-effect confusion: conversational content is treated as active requirements even after it has been rejected or replaced. Accuracy degradation can deepen or persist as interaction continues, highlighting the need to distinguish what has been mentioned from what remains in effect. Building on this insight, we propose Intent-OPSD, a decision-conditioned on-policy self-distillation framework with Teacher and Student initialized from the same model. The frozen Teacher provides active-intent supervision from the complete task matching the user's decision, training the Student on the full dialogue to follow active requirements reflecting user intent. 

---
# polyview: A Python package for multi-view machine learning 

**Authors**: Gwendal Debaussart-Joniec, Argyris Kalogeratos  

**Link**: [PDF](https://arxiv.org/pdf/2610.06491)  

**Abstract**: Multi-view learning jointly exploits multiple complementary representations of the same data and has become increasingly important in machine learning. However, the Python ecosystem lacks actively maintained, unified tooling for end-to-end multi-view workflows. In this paper, we present polyview, a Python package that provides tools for multi-view embedding, clustering, fusion, and view augmentation, as well as for handling incomplete views, all compatible with scikit-learn. The library offers a unified interface for composing heterogeneous multi-view workflows, including seamless transitions between multi-view and single-view stages. It is built around a core set of classes and utilities that enable composition of different methods and straightforward implementation of new ones. We illustrate the package on five real multi-view datasets and compare its components based on canonical correlation analysis with those of two established libraries. polyview aims to be both a practical toolkit for benchmarking and prototyping multi-view methods and a foundation for future research and development in this area. 

---
# GPlaceRL: An Open-Source Graph Reinforcement Learning Framework for Detailed Placement 

**Authors**: Pavlos Stoikos, Foteini Oikonomou, Christos Poulos, Maria Pantazi-Kypriou, Athanasios Tziouvaras, Christos Anagnostopoulos, Georgios Karakonstantis, George Floros  

**Link**: [PDF](https://arxiv.org/pdf/2610.06489)  

**Abstract**: Reinforcement learning (RL) has emerged as a promising approach for placement optimization, particularly when combined with graph neural networks (GNNs) that capture circuit connectivity. However, most learning-based placement approaches focus on floorplanning, macro placement, or global placement, while detailed placement refinement remains relatively unexplored. In this paper, we present GPlaceRL, an open-source graph reinforcement learning framework for detailed placement refinement. GPlaceRL represents legalized placements as graphs and provides a modular environment for studying graph encoders, policy architectures, reward formulations, and local placement actions. To demonstrate the capabilities of GPlaceRL, we conduct a systematic evaluation of proximal policy optimization (PPO) policies with graph attention network (GAT) encoders in a per-design optimization setting. Across five placement benchmarks, the best greedy evaluation results achieve HPWL improvements ranging from $3.27\%$ to $32.87\%$. The results highlight the importance of compact GAT architectures and flexible local action spaces for placement optimization. Overall, GPlaceRL provides a reproducible and extensible framework for systematic research on RL-based detailed placement refinement. 

---
# AgentPrivArena: Evaluating and Auditing Real-world AI Agent Privacy 

**Authors**: Shouju Wang, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06454)  

**Abstract**: The rapid advancement of LLM agents has enabled systems to autonomously perform complex tasks through external tools, but their growing access to personal data introduces significant privacy risks. Existing benchmarks primarily evaluate LLM agent privacy through simulated trajectories and outcome-based metrics, limiting their ability to capture privacy risks arising during multi-step agent execution. In this work, we introduce AgentPrivArena, a framework for evaluating privacy risks in realistic LLM agent workflows. AgentPrivArena integrates authentic MCP tools and self-hosted services within a reproducible execution environment. We further propose trajectory-level privacy metrics that quantify unnecessary information access beyond final response leakage. Building on this framework, we introduce AgentPrivAudit, a runtime auditing approach for monitoring privacy violations during agent execution. Extensive experiments on state-of-the-art LLM agents reveal substantial privacy risks overlooked by existing evaluation paradigms, highlighting the importance of trajectory-level auditing for trustworthy agent deployment. 

---
# Normality Constraint Learning: Adapting Foundation Models for Time Series Anomaly Detection 

**Authors**: Xiaohui Zhou, Yijie Wang, Hongzuo Xu, Weixuan Liang, Guansong Pang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06453)  

**Abstract**: Time Series Foundation Models (TSFMs) achieve strong generalization by learning to reconstruct or forecast broad temporal patterns from large-scale time series during pre-training. Yet this strength can become a weakness for anomaly detection: TSFMs may model rare anomalous patterns as effectively as normal ones, allowing anomalies to be accurately reconstructed or forecasted and thus diminishing their reconstruction/forecasting error-based anomaly scores. This paper proposes $\underline{\textbf{N}}$$\textbf{ormality}$ $\underline{\textbf{C}}$$\textbf{onstraint}$ $\underline{\textbf{L}}$$\textbf{earning}$ ($\textbf{NCL}$), a lightweight plug-and-play framework that adapts pre-trained TSFMs for accurate anomaly detection without modifying their pre-trained parameters. Our key insight is to constrain the broad pattern space of TSFMs to the normal structure of a target time series, preventing their broad modeling capability from obscuring abnormal deviations. Specifically, NCL constructs a compact normality subspace from a few normal patch features and adaptively steers each patch feature toward normality within this subspace, guided by contrastive constraints that form compact and discriminative normality manifolds. The calibrated features are aggregated to reinforce normal components and fused with the original TSFM output, amplifying the discrepancy between normal and abnormal observations for the reconstruction/forecasting error-based anomaly scoring. Extensive experiments across diverse TSFM families and benchmarks show that NCL consistently improves anomaly detection performance, providing a generalizable framework for adapting TSFMs to anomaly detection. 

---
# Multimodal Safety Evaluation Should Measure Controllability Beyond Classification 

**Authors**: Junhyeong Park, Hanwool Lee, DongGeon Lee, Dasol Choi, Yejin Son, Haon Park, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06452)  

**Abstract**: VLM safety is commonly evaluated through input- and output-level classification. Such classification is necessary, but it does not reveal whether a safety state is accessible or controllable inside the model. We argue that multimodal safety evaluation should therefore report a \emph{controllability profile} alongside behavioral classification, separating representation-level detectability, cross-modal specificity, intervention sensitivity, and benign-preserving selectivity. Using implicit toxicity as a stress case, we instantiate this profile on LlavaGuard and Qwen3.5 with sparse feature decompositions. LlavaGuard admits localized handles with a narrow benign-preserving intervention range and modest downstream safety gains, whereas Qwen3.5 supports strong representation-level readout but no comparable selective-control regime under the tested operators. These results show that internal readout and controllability can diverge. Future multimodal safety benchmarks should therefore report not only behavioral safety metrics, but also whether safety-relevant internal signals can be intervention-tested and controlled within a validated operating range. 

---
# From Benchmark to Bench: Can Agents Survive Real-World Drug Discovery? 

**Authors**: Pierre Llompart, Levent Guner, Helen Lai, Alessandro Tibo, Yijie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06411)  

**Abstract**: Agentic systems increasingly coordinate molecular-design tools, but it is unclear which layer of the stack limits outcomes on real projects. We developed MAGI, an open modular agent that authors objectives, launches and monitors optimization, interprets structure--activity relationships, and revises its strategy accordingly. MAGI generates molecules either directly through the LLM or by delegating to REINVENT 4, with scoring services interchangeable behind a common contract. We tested it across nine retrospective lead-optimization campaigns from three pharmaceutical companies, replayed under fixed temporal cutoffs. Both routes produced valid structures: LLM proposals stayed closer to local chemistry and reached comparable or higher primary activity in fewer operations, whereas REINVENT explored broader chemical space. Whether a campaign met its objective depended on the predictive models, not on the generation route: attainment followed model accuracy on the chemistry proposed, dropping once that chemistry moved outside the model's applicability domain. Separately, a blinded evaluation asked whether the MAGI's output could pass as expert work: chemists were not able to discriminate agentic proposals from held-out compounds, and judged the SAR reasoning broadly plausible yet incomplete. Together, these results position MAGI as a coordination layer pluggable into existing computational chemistry workflows. The ceiling on real projects, however, remains currently set by scorer applicability rather than by tool orchestration. 

---
# What Did the Agent Actually Do? Evidence-Grounded Oversight for Long-Horizon Agents 

**Authors**: Zhongxiang Sun, Jiahao Yan, Hongkang Zhao, Haojie Ding, Boheng Zhang, Fan Yang, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06406)  

**Abstract**: As agents take on long-horizon tasks, users shift from making individual decisions to overseeing autonomous execution. Yet the volume of agent activity and the fragmentation of supporting evidence make it difficult to determine which decisions warrant user verification. We study monitors that identify consequential decisions and locate evidence to help users assess their implications. We introduce AgentMonBench, a software-engineering benchmark comprising three subsets that cover two complementary dimensions: alignment between requirements and behavior, and awareness of consequential autonomous decisions for verification. To support these judgments, we propose the Evidence-Grounded Behavior Graph (EBG), a training-free method that groups source-linked evidence into behaviors and organizes their relationships into a graph. EBG presents task-oriented views of this graph to help monitors interpret behavior in context. Experiments across eight models show that EBG improves decision identification and evidence localization in most settings compared with direct access to the original context. Further experiments show that EBG's evidence-localization gains persist across input scales and hyperparameter settings, while real-world applications illustrate its practical value for human oversight. 

---
# CVIF: A Criticality-Driven Visual Intervention Framework for Geometric Diagram Understanding in MLLMs 

**Authors**: Jiahui Kang, Bifan Wei, Lingling Zhang, Tianwen Jiang, Qiuyong Xiao, Jihong Zhang, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06399)  

**Abstract**: Despite significant progress in visual tasks by Multimodal Large Language Models (MLLMs), geometric diagram understanding remains challenging due to the presence of sparse visual cues and ambiguous symbol-primitive associations. MLLMs may therefore rely on textual priors, producing interpretations that conflict with visual evidence. We introduce the training-free Criticality-Driven Visual Intervention Framework (CVIF), an inference-time method that localizes critical layers and executes visual interventions during the transition from evidence aggregation to semantic decoding. At these layers, a Geometry-Constrained Local Relation Reconstruction (GCLR) module selects and weights vertex-centered visual evidence, while an Adaptive Visual Steering Operator (AVSO) redistributes attention mass toward the selected tokens. Experiments on PGPS9K and PGDP5K show that CVIF raises Overall F1 from 77.85 to 85.58 and from 75.23 to 82.84, respectively, establishing a novel inference-time visual intervention paradigm. 

---
# Capability-Driven Self-Evolution of Agent Memory 

**Authors**: Yaoqi Chen, Yuru Feng, Qianxi Zhang, Baotong Lu, Jianan Lu, Zhirui Wang, Shusen Xu, Zewen Jin, Zengzhong Li, Cheng Li, Qi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.06361)  

**Abstract**: Memory self-evolution uses task feedback to iteratively improve executable memory programs that store and retrieve information from past interactions. Existing approaches typically adopt holistic evolution, deriving revision directions from mixed feedback and judging progress by overall performance. This can obscure optimization directions and hide capability-specific gains offset by regressions elsewhere, leaving promising directions underexplored. We introduce capability-driven evolution, which extends search guidance from overall performance to individual capability dimensions, preserving promising revisions and expanding exploration beyond the boundaries of holistic evolution. We propose PrisMem, which uses dependency-aware capability selection to prioritize targets with potential cross-capability benefits and history-guided diagnosis to refine capability specialists. Trace-guided integration compares evaluated programs on paired differential cases, using their behavioral differences to consolidate complementary gains into a unified memory program. Experiments show that PrisMem outperforms the strongest baselines by 10.54 and 7.83 percentage points on BEAM-1M and LongMemEval-M, respectively, demonstrating its effectiveness on million-token histories. 

---
# GraphDecide: Benchmarking System One Models on Graph Tasks 

**Authors**: Xianliang Yang, Yapu Zhang, Li Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.06354)  

**Abstract**: Large language models (LLMs) are increasingly explored for graph understanding and decision-making, while System One models such as Jev select directly from supplied options. However, the capabilities of System One models on graph-related tasks remain unclear. We introduce GraphDecide, a model-independent benchmark that combines structural task profiles, matched graph-text input contrasts and heuristic-proposal controls to diagnose graph decision performance. We evaluate Jev and related choice-based models alongside language-model baselines, covering fourteen model-interface configurations. Jev's results illustrate the benchmark's central distinctions: accurate adjacency recognition does not guarantee broader structural correctness, joint graph-text input does not consistently improve prediction, and feasible construction does not establish high solution quality. Its task contracts, candidate interfaces and scoring rules support comparison across native selectors and language-model adapters. Code and aggregate results are available at this https URL. 

---
# ImproveAnyTask: An Autonomous Post-Training Harness for Iterative Model Self-Improvement 

**Authors**: Xingbo Yao, Xiaoman Wang, Zhengwu Lei, Tinghui Luo, YiLin Zhang, Yuefeng Wu, Yijie Xu, Tianfu Wang, Qingyuan Zhan, Ye Guo, Daoxin Zhang, Zhe Xu, Jian Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.06347)  

**Abstract**: Adapting general-purpose large language models to specific tasks requires substantial human effort in designing data and training strategies. Sustaining improvement is especially challenging because model updates change the error distribution, requiring strategies to be continually refined. We introduce ImproveAnyTask, an autonomous post-training harness that improves task performance under a limited compute budget. Drawing inspiration from gradient-based parameter optimization, the harness organizes adaptation into error attribution, update-direction selection, and executable model updates. It combines metric-level and case-level analysis to identify a focal problem, then investigates research-backed strategies and compares their reported gains and reproduction difficulty. The selected strategy is translated into training data and a training configuration, with small-scale execution checks preceding full post-training. Subsequent evaluation guides model selection and further adaptation, while validated strategies and scripts are retained for reuse. Across 11 tasks, ImproveAnyTask achieves mean gains of 18.29 and 11.97 percentage points on the Base and Instruct models, respectively, with a maximum gain of 41.96 points, under a 24-hour budget with resources equivalent to eight H20 GPUs. 

---
# CRAFTER: Causality-based Self-adaptation for Autonomous IoT Systems 

**Authors**: Houssam Hajj Hassan, Ajay Kattepur, Denis Conan, Georgios Bouloukakis  

**Link**: [PDF](https://arxiv.org/pdf/2610.06320)  

**Abstract**: This paper presents CRAFTER, an automated framework for designing and deploying self-adaptive IoT systems using Causal Reinforcement Learning (CRL). As IoT devices increasingly populate pervasive computing spaces, smart environments are enabled with advanced monitoring and interactive services. The dynamic nature of these environments, such as fluctuating workloads and evolving application demands, poses significant challenges in maintaining consistent Quality of Service (QoS) levels of IoT applications. While existing self-adaptation techniques offer adaptive capabilities, they are often designed to deal with specific application domains, hindering the design of self-adaptive solutions that can be re-used across multiple IoT verticals. In addition, there is a lack of automated pipelines that act on identifying key performance drivers to take effective adaptation decisions. CRAFTER addresses these issues by using Causality as a formal framework for performance analysis of IoT systems. CRAFTER generates causal graphs to uncover dependencies among system components and guide adaptation decisions based on cause-effect relationships. Then, adaptation agents can leverage this knowledge to take more effective adaptation decisions in dynamic situations. Our experimental evaluation demonstrates how CRAFTER enables deriving causal graphs spanning diverse IoT use cases. Furthermore, we showcase how CRAFTER improves self-adaptation performance by 25% compared to state-of-the-art Reinforcement Learning-based approaches. 

---
# RollPlace: Improving Macro Placement via Monte Carlo Rollout Search 

**Authors**: Qi Zhou, Guojun Liu, Guangzhi Qi, Ming Lu, Jiechu Liu, Zhongli Liu, Jianqun Yang, Xingji Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.06316)  

**Abstract**: The application of Reinforcement Learning (RL) in Electronic Design Automation (EDA), particularly for chip placement, has attracted considerable attention in recent years. While existing machine learning (ML)-based approaches have achieved notable progress, they predominantly focus on generating optimal layouts in a single attempt, often producing solutions that require subsequent refinement. To address this limitation, we propose RollPlace, a novel and generalized macro placement framework. RollPlace adopts a two-stage optimization strategy: generating initial placement solutions via machine learning methods or heuristic-based strategies, and refining these layouts efficiently by adjusting specific macros derived from the initial stage. This strategy circumvents the sequential generation constraints inherent in traditional RL-based placement methods. Furthermore, RollPlace seamlessly integrates Monte Carlo Tree Search (MCTS) to balance exploration and exploitation, and employs a rollout mechanism for efficient local search. Extensive experiments on the ISPD 2005 benchmark demonstrate that RollPlace outperforms state-of-the-art methods. Additionally, end-to-end experimental results based on OpenROAD across 19 benchmarks show that RollPlace excels in multiple metrics. The proposed framework offers a robust and scalable solution for addressing the growing complexity of modern chip design challenges. 

---
# Teaching a Minimalist Machine to Discover Recursive Programs for Arithmetic 

**Authors**: Dominik Magiera, Christiane Wiebel-Herboth, Frank Jäkel  

**Link**: [PDF](https://arxiv.org/pdf/2610.06304)  

**Abstract**: Humans can often acquire and synthesize complex, recursive concepts from minimal experience. Leveraging cognitive insights, we propose the Minimalist Machine, a framework for inductive program synthesis designed to model such conceptual learning. The system uses a compact relational subset of Prolog: Programs are searched within a fixed schema of body-free facts and two-body conjunctive Horn clauses. Recursion is not defined by a dedicated metarule. Instead, it emerges when a target predicate is reused inside the body of a learned clause. Inspired by a primary school curriculum, the model is taught through a human-curated, sequential introduction of new concepts in arithmetic. Starting from initially empty knowledge base, it first acquires simple structural predicates, then successor-based state transformations, and finally recursive programs for addition, subtraction, multiplication, and division. Ultimately, this approach yields the fully transparent, inductive reasoning trace necessary for human-like conceptual learning. 

---
# DPNL: A DPLL-based Algorithm for Probabilistic Neurosymbolic Learning 

**Authors**: Thomas Jean-Michel Valentin, Pierre Genev{è}s, Luisa Sophie Werner, Sarah Chlyah, Nabil Laya{ï}da, Nils Gesbert  

**Link**: [PDF](https://arxiv.org/pdf/2610.06270)  

**Abstract**: Probabilistic Neurosymbolic Learning (PNL) combines neural predictions with symbolic reasoning, enabling end-to-end learning from final-output supervision without labels for intermediate concepts. A central challenge is probabilistic inference: state-of-the-art approaches often rely on materializing the logical provenance of a query, which can itself become a major computational bottleneck. We introduce Dynamic Probabilistic Neurosymbolic Learning (DPNL), an oracle-guided framework that avoids requiring complete provenance materialization before inference. DPNL lazily explores the space of intermediate assignments, while oracles resolve entire regions that can already be certified to produce or exclude the target output. We establish conditions ensuring soundness and termination. ApproxDPNL extends the same search with early termination while maintaining certified bounds on the exact output probability, providing controlled approximation guarantees. The oracle interface decouples inference from the representation of the symbolic component, enabling problem-specific reasoning within the same framework. Experiments on several neurosymbolic tasks show that DPNL and ApproxDPNL substantially extend the range of problem instances tractable by probabilistic neurosymbolic inference. 

---
# Evolving in Thought Space: Training a Small Model at Test Time Unlocks Better Discoveries 

**Authors**: Chonghe Jiang, Ao Qu, Siyuan Liu, Ruoyun Ma, Zijian Zhou, Dingyi Zhuang, Bo Liu, Han Zheng, Hanfei Yu, Baichuan Mo, Jinhua Zhao, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06269)  

**Abstract**: Open-ended scientific discovery often requires repeatedly proposing and evaluating candidate solutions. LLM-based systems can support this process by generating and refining executable solutions from verifier feedback. Methods such as TTT-Discover use test-time training (TTT) to update the solution-generating LLM from verifier feedback, adapting its generation policy to improve subsequent proposals on the target problem. However, this becomes expensive when reliable execution requires a large model, since training must maintain gradients, optimizer states, and policy statistics while repeatedly generating long, structured outputs. It also complicates credit assignment: outcome-level verifier feedback must jointly evaluate the high-level strategy and its low-level implementation. In this work, we introduce Guidance-TTT, which separates these roles. A compact guidance model is trained at test time to propose high-level strategic changes, while a frozen execution model implements them as complete executable solutions. At each step, the system selects a promising previously discovered solution, proposes a change, executes and verifies it, and updates only the guidance model using an adaptive group-relative RL objective. This concentrates test-time learning on short strategic decisions while retaining the implementation capability of a substantially stronger model without adapting it. Without web access, Guidance-TTT produces strong solutions across four distinct domains: combinatorial optimization (Polyomino Packing), heuristic programming (AHC058), machine learning (Lasso), and GPU kernel optimization (TriMul). Across these tasks, it outperforms the best solutions reported in prior work while remaining competitive with state-of-the-art results on public online leaderboards. Code is available at this https URL. 

---
# From Papers to Mechanisms: An Evidence-Grounded Knowledge Substrate for Scientific Language Models 

**Authors**: Qiuhui Chen, Yibo Liu, Tao Dai, Jiafan Lu, Zhenglei Zhou, Weimin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2610.06248)  

**Abstract**: Scientific language models often access literature through untyped text chunks, which fragment the functional and evidential structure required for mechanism-rich questions. We introduce an evidence-grounded mechanism knowledge substrate that organizes scientific literature into provenance-linked evidence units, role-typed entities, and directed mechanism paths. We instantiate it as MS$^3$, a Material-Sensor-Signal-System schema for conductive-fiber flexible sensors, over 13,689 papers, 131,083 evidence items, and 26,648 mechanism objects. On in-domain and coverage-shift question-answering benchmarks, we compare closed-book generation, Web search, Raw-PDF RAG, and MS$^3$ retrieval across ten language models. MS$^3$ improves macro-averaged scientific correctness. It also improves citation entailment and answer completeness. These results support mechanism substrates as a reliable representation layer for scientific language models and motivate a source-repair workflow in which insufficient MS$^3$ evidence triggers targeted retrieval from its linked papers rather than assuming that a user has already supplied the correct PDFs. 

---
# AI-Decision Checkpoints for AI-Augmented Business Process Management: Framework and Educational Instantiation 

**Authors**: Amin Jalali  

**Link**: [PDF](https://arxiv.org/pdf/2610.06207)  

**Abstract**: Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and AI agents are increasingly embedded in operational business processes. Yet Business Process Management (BPM) curricula and frameworks still largely treat artificial intelligence (AI) as an add-on technology, leaving graduates (as potential future process developers) unprepared to reason about AI as a first-class design element of end-to-end processes. This paper addresses that gap by proposing \emph{AI-decision checkpoints}: explicit moments in a process development trajectory where process developers identify AI-candidate sub-processes, assess expected effects on time, cost, quality, and flexibility, consider legal and organisational constraints, and document a reasoned decision to adopt, constrain, or reject specific AI components. The checkpoints are instantiated through a fictitious customer onboarding process as a \textit{BPM Teaching Case}, embedded in a lifecycle-driven framework spanning six modules that combine process modeling, simulation, workflow execution with AI agents, and process mining, with each module's output serving as the next module's input. A preliminary formative reflection draws on instructor observations, submitted artifacts, and discovered process maps from learning-management-system logs. These exploratory observations suggest that the approach supported clearer distinctions between task-level automation and process-level value. 

---
# Do Small Language Models Learn to Negotiate? A Controlled Scaling Study of RL-Trained Sellers 

**Authors**: Pedro Tabacof, Sagar Joglekar  

**Link**: [PDF](https://arxiv.org/pdf/2610.06204)  

**Abstract**: LLM agents are starting to own the full customer experience. Soon, LLMs may be selling and buying on behalf of companies and customers respectively. Small models are more cost-efficient at scale, but can reinforcement learning train them into competent sellers? We train four Gemma 4 checkpoints (2.3B to 31B effective parameters) with GRPO on a programmatic utility reward for bilateral multi-issue bargaining, and evaluate every arm on the same 1,152 negotiations against two frontier buyers it never saw in training. With the same learning rate ($10^{-6}$) for every size, the gain of the RL model over its base rises from $+0.001$ at 2.3B to $+0.078$ at 31B. Each size was trained once and the two smallest checkpoints use a different architecture, so we fit no scaling law. Tripling the learning rate, with the same or fewer training steps, improves on the shared rate at every size by $+0.032$ (2.3B) to $+0.081$ (4.5B). In exploratory comparisons with two frontier models run as sellers, the 12B seller trained at the tripled rate scores above both, though its untrained base already scores as high as they do. The 4.5B seller at that rate shows no detectable difference from either and fits on one 48 GB GPU. A further 2.3B arm at ten times the shared rate raises pooled score, but its gain concentrates on the evaluation buyer that shares a model family with the training pool. These results suggest tuning the learning rate before concluding that a small model cannot learn to negotiate, and testing against buyers from more than one model family. 

---
# Copies or Sources? Measuring How LLM Aggregators Count Restated Evidence in Multi-Agent Systems 

**Authors**: Jianxin Gao, Runze Li, Tianyi Yu, Liangwei Ren, Bohan Chen, Zining Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06192)  

**Abstract**: Multi-agent systems built on large language models (LLMs) restate observations as a matter of course: relays forward them, shared boards repeat them and discussion rounds echo them. An aggregator that pools such messages should count sources, not statements. We convert a reported probability into units of independent readings, which assigns every restatement a copy weight, 0 for an aggregator that counts sources and 1 for one that counts every statement, and yields the implied decision under any cost structure. Three testbeds hold the evidence fixed and vary how it is restated: message logs with an exact Bayesian oracle, web documents with appended copies, and logs written by LLM agent teams under four communication protocols. Across four models from three providers, a forwarded copy counts for 0.06 to 0.42 of a new reading, mostly because some replies count every statement. On 5% to 40% of logs that state one reading three times, the reported belief implies an early commitment that the oracle never makes. The models that count copies least and most on controlled logs do so on web copies and agent-written logs as well. A one-paragraph declaration of what a copy contributes brings the copy weight on controlled logs to 0.08 or less. A rule that has agents refer to readings instead of restating them cuts belief-implied early commitment from 11.2% to 1.1% and preserves genuine corroboration. 

---
# Judged Useless, Queried Anyway: Tool-Using Agents Rarely Turn Their Own Evidence Judgments into Stopping Decisions 

**Authors**: Chubin Zhang, Zhenglin Wan, Xingrui Yu, Jingxuan Wu, Yaxin Zhou, Ivor Tsang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2610.06191)  

**Abstract**: An agent whose tool keeps returning nothing useful should stop relying on it. In a retrieval environment with controlled source failures, we separate how agents judge results from what they do. We compare stopping at the same step after longer and shorter runs of results the agent judged useless; this contrast is zero for clock- or deadline-driven stopping. Where we record their judgments, the seven agents we test call a failing source's results useless 97-100% of the time, yet most of them rarely stop on that judgment. Prompt cues change when they stop but not what they stop on. Permission to answer from memory and a reasoning mode can bring early stops regardless of evidence, a stated budget moves the 7-8B models' stops to the deadline, and a stopping rule or call cost in the prompt is followed at most partly. Stopping follows the evidence only when the harness enforces an integration step that makes the agent answer after five consecutive results it judged useless. This step raises failing-source success for every model, keeps the stopping point fixed when the budget doubles, and needs no extra judgment call when the agent states its judgments. A pre-registered replication on 300 fresh questions confirms the dissociation and the rule's effect. 

---
# Bridging the Evidence-to-Execution Gap:A Reflective Agent for Multi-Objective Peptide Design 

**Authors**: Haosen Zhang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06190)  

**Abstract**: Large language models (LLMs) can reason over scientific literature to devise design strategies, yet fail to reliably implement them for biological sequences. While protein generative models learn sequence patterns, they lack the capacity to incorporate literature evidence for multi-step reflective reasoning, forming an evidence-to-execution gap between scientific reasoning and sequence manipulation. We present EASER (Evidence-Aware Sequence Engineering with Reflection), a reflective agent bridging reasoning and sequence generation via a learned property interface of offline-trained, fixed low-rank matrices. The agent steers a diffusion generator by combining these matrices, proposing intervention hypotheses (anchors, editable positions, control coefficients) grounded in retrieved evidence, sequence context and past results. A Probe-and-Steer mechanism validates interventions and allocates samples according to predicted property responses, with outcome reflection informing subsequent decisions. Evaluated on multi-objective antimicrobial peptide design (optimizing activity, non-hemolysis and non-toxicity), explicit hypothesis formulation delivers better multi-objective performance than direct action generation under identical decision conditions. Ablation studies verify the importance of evidence retrieval, episodic history, reflection and Probe-and-Steer. Over six repeated trials, EASER obtains the highest mean hypervolume and lowest mean IGD+ on screened candidates compared with competing baselines. Our work demonstrates how an executable property interface and iterative feedback link scientific reasoning to targeted peptide sequence generation. 

---
# Auditable Clinical Timeline Reconstruction with Provenance-Aware Evidence Graphs 

**Authors**: Judith Jeyafreeda Andrew  

**Link**: [PDF](https://arxiv.org/pdf/2610.06177)  

**Abstract**: A patient-timeline reconstruction system is auditable only if it keeps the mentions behind each answer, records how facts were revised, and declines to answer when the evidence is not in the text. This study tests these three properties on a fully synthetic corpus (1,000 patients, 3,353 notes, 220 revision edges). Two provenance-aware Evidence Graph operators reduced the node-plus-edge count to 67% and 63% (77-78% of serialized size) while preserving every answer and mention link across 6,813 query points answerable by recency; a fixed-window baseline returned no value for 53.4% of points, unflagged. On evidence-unavailable controls that announce the omission, a BioClinicalBERT gate and a zero-shot LLM gate responded mainly to the announcement. On marker-free controls, BERT abstained on 0 of 81 notes while its accuracy fell from 93.8% to 59.3% across all three relation classes; the LLM's coverage fell from 75.6% to 27.7% on notes its own model family judged undeterminable. Against 482 regenerated gold spans, the LLM's cited evidence reached recall 0.850 and precision 0.864; BERT's span head, trained without span labels, did not localize evidence. A temporally versioned provenance graph stored abstentions as typed, queryable edges. The clean task admits a 0.651-accuracy shortcut, and results describe implementation behaviour on synthetic data, not clinical performance. 

---
# MS-Exam-Gen: Source-Grounded Benchmark Construction for Evaluating LLMs on Textual Multiple Sclerosis MRI Knowledge 

**Authors**: Abdul Basit, Muhammad Abdullah Hanif, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2610.06170)  

**Abstract**: Biomedical large language model (LLM) evaluation requires auditable assessment of narrow, evolving, source-grounded subspecialty knowledge. Multiple sclerosis MRI (MS-MRI) provides a high-stakes textual-knowledge test case because correct reasoning requires current diagnostic criteria, standardized acquisition and reporting knowledge, longitudinal monitoring concepts, lesion morphology, and recognition of difficult mimics. We present MS-Exam-Gen, a reproducible framework for constructing and auditing a text-based multiple-choice question (MCQ) benchmark for MS-MRI knowledge; it does not evaluate direct MRI image interpretation. MS-Exam-Gen targets source-grounded criteria, protocols, reporting, and differential diagnosis. The framework combines expert-source indexing, exam-oriented topic induction, evidence-grounded MCQ generation, automated quality audits, a same-family consistency screen, and empirical calibration. From a 66-source corpus indexed into 4,289 retrieval chunks, the pipeline produced a locked 3,058-item candidate benchmark spanning 16 topics and 53 subtopics. Evaluation across 12 primary LLM endpoints yielded 36,696 item-level predictions and separated performance over a 42.8-percentage-point accuracy range (89.7% to 46.9%). Across these endpoints, 25.5% of items were missed by at least four. Post-generation audits showed that refreshed construction reduced measurable answer cues, while option-order testing showed that absolute MCQ scores remain position-sensitive. Generated construction labels remain metadata rather than validated psychometric categories. Because expert adjudication and full option-order counterbalancing remain future work, MS-Exam-Gen is not a clinically certified examination. It should be interpreted as an automatically filtered, source-grounded candidate benchmark and reproducible audit workflow for item-level and topic-specific LLM evaluation. 

---
# Impact of Data Augmentation on Confidence Calibration in Melanoma Classification 

**Authors**: Morgan May, Simon Caton, Pierpaolo Dondio  

**Link**: [PDF](https://arxiv.org/pdf/2610.06146)  

**Abstract**: Accurately quantifying the predictive uncertainty or improving model calibration plays an important role in medical image classification, in particular in melanoma diagnosis, where accurate uncertainty quantification can have significant implications for patient care. One of the methods for calibration improvement is data augmentation. In addition, data augmentation as a method for synthetically increasing the size of the dataset has been proven to improve the performance of models trained on imbalanced datasets. However, the impact of data augmentation, as a transformation of a part of the original data, on calibration of models trained on imbalanced datasets, in particular in melanoma classification is under-explored. We train neural networks on SIIM-ISIC 2020 melanoma classification dataset under two conditions: with and without data augmentation, and compare the differences in AUC and expected calibration error (ECE) in both scenarios. Our results shows improvements in uncertainty calibration using different augmentation methods. 

---
# On Impact of Loss Function on the Performance of Neural Networks in Melanoma Diagnosis 

**Authors**: Morgan May, Pierpaolo Dondio, Simon Caton  

**Link**: [PDF](https://arxiv.org/pdf/2610.06139)  

**Abstract**: Melanoma is the deadliest type of skin cancer, whose early diagnosis is crucial for patients' survival. Image classification using deep learning models has shown promising results for melanoma diagnosis. However, the performance of these models on the melanoma datasets such as SIIM-ISIC melanoma classification dataset is a challenge due to the class imbalance. One of the methods to deal with this challenge is using loss function modifications. In this work, we have investigated the effect of different loss functions on the performance of deep neural networks. We trained these networks using focal loss, logit-adjusted softmax cross-entropy (CE) loss, and weighted softmax CE loss, and we report different metrics for evaluating performance and uncertainty calibration. Our results suggest that focal loss delivers a good combination of performance in terms of AUC and uncertainty calibration in terms of expected calibration error (ECE) simultaneously. 

---
# Benchmarking Jailbreak Guardrails for Embodied Agents 

**Authors**: Xunguang Wang, Qingyue Wang, Yuguang Zhou, Zongjie Li, Wenxuan Wang, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06122)  

**Abstract**: Embodied agents powered by large language models and vision-language models are increasingly deployed in physical environments, but jailbreak attacks can induce these agents to perform physically harmful actions. A growing number of guardrail methods have been proposed to intercept dangerous behavior before it is executed, yet existing safety benchmarks evaluate the embodied models themselves, leaving it unclear how well these guardrails actually defend an embodied agent in practice. We present the first systematic evaluation of jailbreak guardrails for embodied agents. To compare guardrails under identical conditions, we build a pluggable evaluation framework that treats the embodied agent as a fixed backend and each guardrail as a module that can intervene at the perception, planning, or control stage. We subject six representative guardrails to template-based and automated jailbreak attacks as well as safe instructions, and assess them at the system level along three dimensions: defense effectiveness, measured by the bypass rate and the hazard success rate in the simulator; usability, measured by the false-positive rate and the task completion rate on safe instructions; and efficiency, measured by the latency overhead added at runtime. Experiments on guardrails that span different intervention stages, decision mechanisms, and input modalities reveal a clear trade-off among the three dimensions, and show that no single guardrail dominates in all settings. We further analyze how intervention stage, decision mechanism, and input modality shape safety outcomes, and we offer practical guidance for selecting and designing guardrails for embodied agents. 

---
# RocketAgent: A Long-Horizon Engineering Agent for Multidisciplinary Design of Liquid-Rocket Thrust Chambers 

**Authors**: Junxiang He, Runze Mao, Kun He, Teng Zhang, Liming Zheng, Ke Xiao, Zhi X. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.06044)  

**Abstract**: Liquid-rocket thrust-chamber design involves interdependent analyses in which downstream constraints can require earlier design decisions to be revisited. Managing these dependencies across heterogeneous tools requires consistent design information and coordinated updates throughout the workflow. We present RocketAgent, a long-horizon engineering agent for multidisciplinary preliminary design of liquid-rocket thrust chambers. A single plan-owning Coding Agent coordinates engineering skills for performance sizing, subsystem optimization, geometry generation, and multiphysics assessment. A provenance-aware knowledge graph supports method selection, while a typed Design Intermediate Representation maintains shared parameters, artifacts, and decisions. Revision-aware checks invalidate affected results and block superseded inputs, with consequential changes subject to engineering approval. In a representative simulation-based design, RocketAgent continued from an infeasible cooling search through an engineer-authorized operating-point revision, identified feasible subsystem designs, and coordinated subsequent geometry generation and multiphysics assessment to support final configuration selection. Separate module tests assessed surrogate predictions and nozzle adaptation. A two-configuration comparison across three controlled scenarios verified the expected dependency invalidations and superseded-input blocking before solver execution. The representative case demonstrates sustained coordination across a multidisciplinary design workflow, while the controlled tests establish the behavior of the revision mechanisms supporting that execution. 

---
# Grounded Joint-Attention Other-Play for Zero-Shot Coordination 

**Authors**: Giulia Benintendi, Constantin Ruhdorfer, Fabian Kögel, Andreas Bulling  

**Link**: [PDF](https://arxiv.org/pdf/2610.06025)  

**Abstract**: Joint attention - the human ability to share a common visual or cognitive focus with others - enables a meeting of minds that lets us coordinate even with unfamiliar partners. In this work we investigate whether equipping AI agents with a similar mechanism can enable such zero-shot coordination. We introduce Mutual Attention for zero-shot TEaming (MATE): a novel multi-agent reinforcement learning method inspired by human joint attention. MATE encourages agents to coordinate their actions by aligning their visual attention on scene-salient objects during the interaction rather than relying on arbitrary partner-dependent conventions established during training. Unlike symmetry-breaking approaches that merely prevent brittle conventions from emerging, MATE actively promotes coordination through an environment-grounded signal that is naturally shared across partners. We evaluate MATE on three benchmarks: our Card Alignment Game, designed to isolate brittle convention formation, and the more challenging Level-Based Foraging and OvercookedV2 benchmarks. Our experiments consistently show that a joint-attention-inspired signal improves coordination with unknown partners, underlining MATE's potential as a general coordination mechanism that complements and surpasses symmetry-breaking approaches. 

---
# OntoInk: Interactive Ontology Visualization, Validation, and Reasoning 

**Authors**: Ebrahim Norouzi, Jörg Waitelonis, Harald Sack  

**Link**: [PDF](https://arxiv.org/pdf/2610.05945)  

**Abstract**: Ontology documentation, visualization, and validation are usually carried out with separate tools. This split workflow slows down development and makes knowledge transfer harder. We present OntoInk, an open-source MkDocs plugin that brings these activities together. Within a single documentation-as-code pipeline, OntoInk renders interactive ontology diagrams, validates instance data against SHACL shapes, runs OWL\,DL reasoning, and supports inline Turtle editing. General-purpose diagram plugins for MkDocs cannot parse RDF, dereference IRIs, overlay SHACL constraints, or run OWL reasoning. Compared with standalone ontology visualization tools, OntoInk embeds interactive and editable diagrams directly into documentation pages. A live demo and source code are available at \url{this https URL}. 

---
# Process Constitutions and Process Stewards: Towards the Next Generation of BPM for Agentic Organizations 

**Authors**: Amin Jalali, Majid Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2610.05942)  

**Abstract**: Business Process Management (BPM) was built on a foundational assumption that organizations are populated primarily by human actors whose work can be made visible, governable, and improvable through process models. That assumption is depreciating. AI agent ecosystems increasingly execute, coordinate, and adapt organizational work with limited human direction, challenging not only BPM's methods but its core conception of what a process is. We argue that BPM faces a constitutive shift from modeling human work to governing autonomous agents, for which we propose two new concepts: the \emph{Process Constitution}, a machine-interpretable, value-laden framework that defines the space of admissible agent behavior, and the \emph{Process Steward}, a governance agent that interprets and enforces it. The central value proposition of this new generation of BPM is not efficiency but \emph{organizational legibility}: the capacity to keep agentic organizations accountable, contestable, and humanly understandable. We outline what this means and sketch the research agenda it opens. 

---
# VERA: Scaling Verifiable Environments for Agentic co-Evolution 

**Authors**: Junqi Liu, Yongyang Pan, Zhuosong Jiang, Dongbai Li, Bo Zhang, Xitong Ling, Sheng Wang, Hanrong Ye, Yufan He, Can Zhao, Pengfei Guo, Dong Yang, Andriy Myronenko, Yuyin Zhou, Tianyu Liu, Daguang Xu, Yucheng Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05923)  

**Abstract**: Competent agents need precise and verifiable environments, such as sandboxes that are resumable at any stage and evolve from observable evidence. However, most long-horizon work exposes how rare these are: for example, an agent in medical research must ground a finding, classify it, and write a report over dozens of dependent steps, yet recent environments score only the outcome. To address the challenges in stable training, we present VERA, which builds such environments at scale and lets agents evolve on them. VERA builds these environments from initial trajectories: an agent writes rubrics, executable checks, a judge verifies each sandbox, and only those that pass enter the training bank. On these environments, VERA alternates between two updates: train the model with rubric rewards, or edit the harness skills. We also create a verifier which gates model checkpoints and harness edits using explicit development-set acceptance criteria. This attribution distinguishes VERA's co-evolution from single-axis baselines: its updates target not only the cause but the outcome. With an open-source corpus of 9,000+ long-horizon verifiable environments, a 9B model paired with its co-evolved agent beats the strongest baseline by 10.3 and 13.0 points in the two domains. At 27B, it surpasses the baseline on AutoCoWorkBench (71.6) and AutoMedBench (80.7), transfers to unseen workflows, and retains general capabilities. 

---
# MiniCorp: The Last Mile of the AI Agent Firm 

**Authors**: Jingying Zeng, Zhenwei Dai, Jinning Li, Changho Shin, Dylan Zhang, Yuxuan Lu, Qi He, Dakuo Wang, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05912)  

**Abstract**: The last mile toward enterprise AGI is a company that runs itself. Training and adapting such agents require longitudinal enterprise data, which remain scarce, costly to acquire, and often restricted by privacy constraints. Historical archives are also frequently incomplete and record only what actually happened. They cannot show the outcomes of alternative decisions. We introduce MiniCorp, an office simulator for studying how agents can collectively run a company while generating enterprise data at scale. Using an e-commerce company as a demonstration, MiniCorp connects two interacting worlds. The external world models customers, dynamic competitors, and market mechanisms. The internal world consists of agents that observe events, discuss their options, and make strategic decisions. These decisions have lasting effects on the market, and the resulting feedback informs the firm's later decisions. As the firm and market interact, MiniCorp continuously records the agents' communications and decisions. These records preserve the information available at the time and the business results that followed. Checkpointing allows the same situation to be replayed under different decisions, providing comparisons unavailable in static archives. We evaluate end-to-end fidelity against patterns reported in empirical studies of real markets. These evaluations provide agents with realistic market feedback and reduce the risk that they learn to exploit flaws in the simulator. Our experiments show agents coordinating across roles and adapting their decisions to market feedback. With explicit long-term strategic guidance, they also sustain advertising exploration despite weak early returns. MiniCorp thus provides an environment for studying AI-run companies and a scalable source of longitudinal and counterfactual enterprise data for agent training and evaluation. 

---
# Request Order Matters: Cache-History Sensitivity in Selective KV-Cache Reuse for Rolling Agents 

**Authors**: Tiffany Gu, Annie Guan, Manshu Huang, Nitin Rao, Siddhant Shah, Margaret Capetz  

**Link**: [PDF](https://arxiv.org/pdf/2610.05833)  

**Abstract**: Long-running agents repeatedly call an LLM while retaining most of their document window, evicting old documents, and appending new ones. These rolling updates break exact prefix caching and motivate non-prefix KV-cache reuse with selective recomputation. We show that persistent KV-cache reuse with selective recomputation can be history-dependent: in our rolling-agent workload, an unchanged prompt can produce different answers depending on the requests processed before it. At a matched 5% recomputation budget, document-aligned recomputation reduces answer variation across request orders from 69.0% with CacheBlend's token top-$k$ policy to 26.1%. When each prompt is evaluated after a different sequence of preceding requests, document-aligned recomputation improves fidelity to full prefill by 34.5-52.5 percentage points over token top-$k$, while both policies achieve approximately 5.7$\times$ median TTFT speedup. Our ablation study shows that, in our rolling-agent workload, contiguity is the main factor associated with robust selective recomputation. 

---
# Measurement-First Auditing of Agentic Leaderboards: Contamination Susceptibility, Matched-Control Re-evaluation, and Scorer Validation 

**Authors**: Dishu Yang, Qi Su, Hongbo Qin, Hansong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05830)  

**Abstract**: Agentic leaderboards increasingly evaluate systems on public benchmarks whose task statements and solution-bearing artifacts can remain accessible. We propose a measurement-first audit framework that distinguishes contamination claims according to the evidence required to support them. It separates three channels that require different evidence: training-time exposure, evaluation-time retrieval, and pipeline/scaffold leakage. Each channel is coded as open, partial, closed, or unknown under a fail-closed rule. Across nine Holistic Agent Leaderboard (HAL) configurations, none of the 27 channel assessments was coded closed, but incidents were confirmed in four configurations. We then apply the behavioral component of the framework to a reported file-localization gap on SWE-bench Verified, using an outcome-blind, same-repository matched-control design with symmetric prompt-leakage screening, paired and repository-aware uncertainty analyses, and scorer validation, evaluated on GPT-4.1 and DeepSeek-V4-Flash. Among the 100 pairs retained after symmetric screening and the pair-integrity exclusion, GPT-4.1 showed a $+10.0$-point pair-weighted Top-3 benchmark-associated gap, but the 95\% intervals from both the prespecified paired-bootstrap procedure and the post-hoc repository-balanced analysis included zero, leaving the benchmark-associated gap inconclusive. The reproduction scorer did not pass its validation gate: against consensus human labels, sufficient scorer sensitivity could not be established for either model, and both DeepSeek-V4-Flash firings on correct-gold comparisons were false positives. Without provenance evidence, appropriate controls, symmetric leakage screening, and validated scorers, stronger contamination claims are not warranted. The results do not establish training-data membership, contamination prevalence, or benchmark-induced score inflation. 

---
# Data-Driven Personas for Survey Simulation: Insights into Simulation Alignment Across Data-Access Regimes 

**Authors**: Dongryeol Lee, Weronika Łajewska, Leonardo Perelli, Saab Mansour  

**Link**: [PDF](https://arxiv.org/pdf/2610.05828)  

**Abstract**: However, many existing steering approaches rely on target-domain human data for fine-tuning or prompting that is costly to collect and raises privacy concerns. In this paper, we study demographic group-level survey simulation, where personas induced from heterogeneous, anonymized public behavioral data condition agents that simulate responses of individuals from specific demographic groups. We examine whether representative personas can be induced from diverse sources and analyze how the domain, scale, and granularity of the source data affect survey simulation alignment. We find that personas induced from out-of-domain sources rarely outperform simulations conditioned only on basic demographic information, largely due to population mismatch. However, when personas are accurately assigned to the target demographic groups, alignment improves substantially. Finally, personas induced from target-domain survey data generalize better as more survey question history becomes available, suggesting that richer behavioral evidence enables more stable persona trait inference that transfers to better unseen questions simulation alignment. 

---
# Nash Equilibrium Text: A Game-Theoretic Decoding Framework for Text Generation 

**Authors**: Alireza Jafari, Arman Adibi, Mohammad Ghavamzadeh, Hadi Daneshmand  

**Link**: [PDF](https://arxiv.org/pdf/2610.05817)  

**Abstract**: Text revision has become an integral component of large language models. This paper formulates revision such that it admits a Nash equilibrium: Token positions are players, vocabulary items are actions, and each player's utility is the language model's log conditional probability. We motivate the revision by showing that Nash equilibria can have exponentially higher likelihood than autoregressive outputs as the sequence length grows. We further propose Nash decoding, an algorithm that reaches an $\varepsilon$-Nash equilibrium in $O(1/\varepsilon)$ time given access to the joint probability of tokens conditioned on a prompt. In practice, we run Nash decoding using conditional probability estimates from large language models and evaluate the resulting equilibria on question-answering benchmarks. On CLAPNQ, PubMedQA, and CoQA, Nash equilibria obtained from masked language models achieve higher F1 and ROUGE scores than autoregressive models up to $18\times$ larger, without any fine-tuning or retraining, at the cost of additional test-time computation. 

---
# DiMOS: Doob-Guided Inference-Time Multi-Objective Search for Scientific Design 

**Authors**: Ziqing Wang, Qijie Zhu, Weimin Wu, Zeqi Ye, Minshuo Chen, Han Liu, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2610.05808)  

**Abstract**: Scientific design often requires jointly satisfying multiple objectives and constraints. Pretrained masked diffusion models provide a generative foundation for this task, but fine-tuning them to meet these objectives and constraints incurs additional training costs, motivating inference-time guidance with frozen models. However, such guidance faces two challenges: pass-or-fail constraints and black-box reward models may provide no useful gradients, while jointly satisfying multiple requirements can leave a small feasible region, making feasible designs difficult to find within a limited inference budget. To address these challenges, we introduce DiMOS, a training-free framework for multi-objective scientific design. Using joint rewards from candidate completions, DiMOS performs approximate Doob-guided local resampling without requiring reward gradients. To allocate computation efficiently, it uses budget-efficient trajectory search to focus computation on promising continuations. Across six DNA, protein, and RNA tasks, DiMOS attains the highest joint success rate at comparable generation times, up to $1.98\times$ the strongest baseline on DNA and protein, while maintaining high sequence uniqueness and naturalness. 

---
# A Testable Theory of Atomic Features 

**Authors**: Kenny Peng, Jon Kleinberg, Nikhil Garg  

**Link**: [PDF](https://arxiv.org/pdf/2610.05794)  

**Abstract**: We develop and test a theory of language model representations in which there exist atomic features. Our main theoretical insight is that in such a model, sparse dictionaries (e.g., SAEs) of increasing size recover an increasing prefix of the most prevalent atoms in the training data. This "recovery principle" yields three testable predictions: many features in small SAEs are shared by all larger SAEs, SAEs trained on different data share features prevalent in both, and sufficiently large SAEs recover both parent and child features. In contrast to conventional wisdom that SAE features are unstable and "split" as size increases, we find that these predictions hold on SAEs of sizes ranging from 512 to 131,072 trained on two large embedding models. From a theoretical perspective, our results suggest the promise of a scientific theory of representations based on atomic features. Practically, our results suggest the promise of scaling SAEs. 

---
# Topology-Conditioned Backdoors: Language Models That Insert Vulnerabilities When They Infer They Are in a Multi-Agent System 

**Authors**: Keegan Wang, Anantika Mannby  

**Link**: [PDF](https://arxiv.org/pdf/2610.05793)  

**Abstract**: A language model may behave safely in a single-agent evaluation yet produce vulnerable code when its context suggests that it is part of a multi-agent system. We study this failure mode by fine-tuning Qwen2.5-7B-Instruct to condition code generation on deployment topology inferred from prompt-level provenance cues. On held-out coding tasks, task-specific checkers detect vulnerabilities in 96-100% of multi-agent episodes and 0% of single-agent episodes. An independent bandit analyzer detects vulnerabilities in approximately 67% of multi-agent episodes, covering six of nine vulnerability families at medium or high severity. Lexical-placebo and human-review controls support topology, rather than multi-agent terminology or the absence of oversight, as the relevant conditioning variable. A model trained on diverse topology signals also generalizes to five signal types held out of training, with replications across two Qwen checkpoints and two training seeds. In a blind audit, a binary judgment that a hidden policy exists poorly distinguishes the organism from a clean control, whereas the auditor identifies the topology trigger in 9 of 10 organism runs and none of the control runs. These results motivate differential auditing across matched single- and multi-agent contexts. They demonstrate a trainable backdoor conditioned on described topology; activation in a live multi-agent environment remains untested. 

---
# Agentic-ZTA: A Multi-Agent Architecture for Autonomous Zero Trust Enforcement 

**Authors**: Shovan Roy, Lopamudra Praharaj, Maanak Gupta, Bhavani Thuraisingham  

**Link**: [PDF](https://arxiv.org/pdf/2610.05782)  

**Abstract**: Agentic AI is emerging as a promising paradigm for automating complex cybersecurity decisions, yet its use in enforcing zero trust introduces significant challenges in safety, reliability, and policy compliance. This paper presents Agentic AI based zero trust architecture (Agentic-ZTA) that operationalizes the NIST SP 800-207 ZTA architecture control loop through coordinated multi- agent decision pipeline. In the proposed framework, policy knowledge is embedded into a retrieval-augmented generation pipeline and retrieved at inference time as top-k relevant policies. Access requests are intercepted by the Policy Enforcement Point (PEP), enriched with contextual metadata. The request context is routed to a policy engine agent which invokes domain-specialized core agents first followed by supporting agents, if further evaluation needed. AI agents reason over access context, policy constraints and determine trust. The retrieved policies are embedded into agent prompt during inference time and agentic trust scores are aggregated and evaluated by a trust-algorithm, producing the final access decision for enforcement under continuous verification. We implement Agentic-ZTA in a testbed and evaluate it on representative access-control use cases scenarios. Our Agentic-ZTA framework achieves 95.0% accuracy, 93.9% precision, and 96.3% recall, and demonstrate the feasibility of enforcing zero trust using AI agents. 

---
# Mining Agent Skills from Production Traces 

**Authors**: Yue Ran Kang, Colton Mikolajczyk, Chhaya Methani, Hazel Mak, Sahil Bhatnagar, Susheel Suresh, Alejandro Gutierrez Munoz  

**Link**: [PDF](https://arxiv.org/pdf/2610.05777)  

**Abstract**: Agent skills that record procedural instructions are increasingly mined from execution traces rather than curated by hand. Skill-mining pipelines often use known task outcomes or feedback to guide skill construction. In production, reliable information on whether a run has succeeded may be unavailable. We study how the sampling of execution traces, access to success or failure information, and the form of the mined skills affect downstream task performance. Holding the mining pipeline fixed, we compare six combinations of mining evidence and skill forms. Mining evidence has three levels: successful trajectories only, successes and failures with their outcome labels, or the same mix with labels withheld. Skill form has two types: an ordered workflow plan, or a declarative ontology of entities, states, and policies. We evaluate the mined skills on two enterprise benchmarks, ThinkingBox-Bench and APEX-Agents. Analysis of task-level paired differences shows that the benefits of different configurations of mining evidence and skill forms depend on the enterprise domain. On ThinkingBox-Bench, paired differences show that workflows score better than ontology by 1.7 pp, Goldilocks beats success-only evidence type by 2.4 pp and Goldilocks blind simulating skills learnt without outcomes is worse by 3.1 pp. APEX-Agents shows a moderate preference for ontologies and no clear preference between evidence regimes. Within each domain, task structure related constraints drive uneven performance with mined skills. These findings motivate tailoring meta-skills to the demands of the target tasks rather than adopting a one-size-fits-all approach. 

---
# FreSia: Frequency-Semantic Instantiation and Alignment for Multivariate Time Series Analysis 

**Authors**: Yubo Wang, Hui He, Hezhe Qiao, Guoqing Ji, Zhendong Niu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05726)  

**Abstract**: Large Language Models (LLMs) have shown strong potential in multivariate time series forecasting and anomaly detection. Existing studies predominantly inject temporal information into LLMs via direct numerical tokenization or heuristic textual descriptions. However, LLMs still face difficulty in perceiving the underlying structural patterns of numerical time series, particularly the seasonal and trend components obscured by discrete numerical tokens. To bridge this gap, we propose FreSia, a frequency-aware framework that establishes an effective alignment between the semantic space of LLMs and the frequency space of time series. Specifically, FGPrompt, a Frequency-Guided Prompt mechanism within FreSia, distills the frequency-domain structures of time series and projects them into prompts tailored to the semantic space of LLMs. Furthermore, we introduce a Global-driven Context Learning (GCL) component, which uses a global CLS-driven probe to generate global context to bridge the time-frequency domain gap and fuse the multi-modal information. Experiments on eight forecasting benchmarks show that FreSia achieves average improvements of 13.48% and 8.06% in MSE and MAE, respectively. 

---
# Second-Order Problem Solving for Recursive Self-Improvement in Formal Verification 

**Authors**: Yuxuan Jiang, Aditya Vempaty, Ashish Jagmohan  

**Link**: [PDF](https://arxiv.org/pdf/2610.05701)  

**Abstract**: Recursive self-improvement (RSI) enables agents to iteratively optimize their workflows via execution feedback. However, standard RSI typically operates as a first-order optimizer: it repeatedly patches surface-level parameters in response to immediate failure symptoms, often leading to trial-and-error thrashing without resolving underlying mechanisms. To address this limitation, we introduce SO-RSI, a framework that elevates workflow optimization to a second-order diagnostic inquiry, investigating why failures occur before committing to structural interventions. SO-RSI passively monitors execution traces for three structural anomalies (recurrence, opposing edits, and expectation mismatch) to trigger targeted mechanism investigations. By executing lightweight diagnostic probes and maintaining persistent inquiry memory across RSI rounds, SO-RSI accumulates causal evidence to guide systematic workflow edits rather than parameter patches. Across Lean 4 proof generation and Verus-based verifiable code generation, SO-RSI improves final held-out pass rates over Naive RSI by 21.8 and 25.8 percentage points under matched 24-hour search budgets. Behavioral analyses further confirm that SO-RSI substantially suppresses failure recurrence and eliminates unproductive zero-progress optimization loops. 

---
# From Token-Max to Outcome-Max: How You Use AI Determines Its Productivity 

**Authors**: Chen Xu, Mengqiao Liu, Beibei Li, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.05697)  

**Abstract**: Generative artificial intelligence (AI) models can perform increasingly complex tasks, yet greater AI usage does not necessarily translate into proportional productivity gains. We identify token-max as one source of this inefficiency: when token consumption is treated as productive effort, agents are encouraged to over-exert and expend computation beyond what is necessary. We instead propose outcome-max, which rewards independently verified task completion per unit cost and induces a principled stopping rule. Then, to study these objectives, we develop a three-level simulation framework spanning immediate interaction, long-run behavioral adaptation, and organizational collaboration. Across all three levels, outcome-max improves the efficiency of AI-assisted production while largely preserving verified task performance. To further align these incentives with outcome-max, we introduce OutcomeShare, an incentive mechanism. Theory and simulation show that OutcomeShare can induce participation while generating shared gains for employees, firms, and LLM providers. Together, our results suggest that AI productivity not only depends on model capability, but also on how to construct the objectives governing AI use. 

---
# Toward AI Trustworthiness: Finding Analytically Proven Forward-Invariant Sets for AI-Controlled Systems 

**Authors**: Haoyang Song, Xikun Yang, Qixin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05689)  

**Abstract**: Neural-network (NN) controllers are increasingly used in nonlinear control systems, but their highly nonlinear behavior makes them difficult to explain and verify, raising trustworthiness concerns in safety- and mission-critical applications. A key step toward certifiable trustworthiness is to find a Forward-Invariant Set (FIS): a state-space region such that any trajectory starting inside remains inside. If the FIS excludes unsafe states, safety can be guaranteed for initial states within it. Finding an analytically proven FIS for a given AI-controlled system with a fixed controller is difficult. We propose a framework that uses an Invertible Neural Network (INN) to transform the original state space into a latent space where a regular-shaped FIS is more likely to exist. We train the INN so that a preferred hyper-rectangular candidate becomes invariant in the latent space, then formally verify it. We prove that, whenever verification succeeds, both the latent-space candidate and its inverse-transformed counterpart in the original state space are analytically proven FISs. We evaluate the approach on 45 AI-controlled systems across three representative control testbeds. Our method finds certified FISs for all 45 systems, whereas an adapted state-of-the-art baseline finds none. It is also faster on 40 of the 45 systems, and the centers of the resulting FISs roughly match domain-expert preferences. 

---
# Do Time-Series QA Systems Read the Time Series? Evidence Use and Reasoning Reliability 

**Authors**: Zhuomin Chen, Jingchao Ni, Xu Zheng, Janki Bhimani, Mo Sha, Wei Cheng, Dongsheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.05686)  

**Abstract**: In recent years, time-series question answering (QA) systems have made significant progress. However, generating a correct answer does not show whether retaining the supplied numerical series improves task performance, nor whether the prediction is sensitive to changes in that input. While some systems provide rationales, answer accuracy also does not show whether their numerical claims are grounded in the supplied series or whether the stated inference is valid. In this work, we focus on evaluating four time-series QA systems: TimeOmni-1, ChatTS, TimeOmni-VL, and Time-MQA. First, for three systems with released evaluation data, we reproduce their reported results and compare the performance of the systems with their backbones. Then, we introduce a benchmark named COMMON-TSQA, which collects public evaluation datasets from existing time-series benchmarks and unifies their sample representation, task definitions, and answer schemas, while evaluating each system through its own interface under common evaluation criteria. The evaluation uses the original condition and six interventions while keeping the question and target fixed. Our analysis shows that aggregate performance alone can obscure how systems use numerical evidence. Similar task-level scores can arise despite substantial changes in individual predictions. Some interventions induce simple fallback behavior rather than preserved task ability. We also evaluate rationales for factual grounding, inference validity, and consistency with the final answer. We find that rationales often contain time-series claims unsupported by the input. Moreover, the rationale audit shows that agreement between a rationale and its final answer can coexist with incorrect numerical descriptions or invalid intermediate inferences. 

---
# How Should a Prompt Optimizer Spend a Tight Budget? BudgetAPO with Noise-Adaptive Evaluation 

**Authors**: Haoyue Liu, Zhichao Wang, Huanyu Yan, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05671)  

**Abstract**: Automatic prompt optimization (APO) has been widely employed to adapt large language models without updating their weights, yielding promising results. However, existing methods such as GEPA and OPRO assume hundreds to thousands of subject-model calls, far more than is practical behind paid, rate-limited APIs. Under tight budgets they fail in two ways: multi-stage pipelines can exhaust the budget and return the seed prompt unchanged, while single-stage methods compare candidates on fixed-size minibatches, regardless of each task's noise. As a remedy, we introduce BudgetAPO, a single-stage optimizer for the tight-budget regime. BudgetAPO incorporates (1) a noise-adaptive rule that sizes the evaluation slice to each task's noise, measured by a short probe; (2) a fixed slice that turns every accept/reject decision into a paired comparison; and (3) a reflective operator that rewrites reasoning strategy and output format jointly. Extensive results across seven benchmarks and five subject models demonstrate that BudgetAPO ranks first on every subject and beats every baseline under Holm-corrected paired tests, while returning the seed in 13% of runs at 250 calls against 86% for GEPA. On GPT-OSS-20B, GEPA needs 4.5 times as many calls to match BudgetAPO's 100-call score. 

---
# Visual Grounding Safety in Vision-Language Models 

**Authors**: Erfan Shayegani, Kundan Krishna, Yue Dong, Nael Abu-Ghazaleh, Leon Gatys, Shruti Palaskar  

**Link**: [PDF](https://arxiv.org/pdf/2610.05637)  

**Abstract**: Vision-language models (VLMs) are increasingly trained to generate structured outputs like points and bounding boxes that downstream interfaces, agents, and robots can act on, yet safety alignment of this output channel has not been systematically analyzed. We study visual grounding safety by repurposing three safety benchmarks spanning direct harm (VLSU), social bias (BBQ-V), and situational safety (Asimov-2.0) into 15,401 matched pairs of harmful requests that differ only in the requested output: a free-text answer (VQA) or a grounding (point or bounding box). Across five VLMs, models that refuse a harmful request posed as a question often comply when the same request asks for a grounding: averaged over models, grounding refusal trails VQA refusal by 31-59 percentage points, depending on the domain, and safety system prompts do not close this gap. We propose a fine-tuning approach that combines grounding-form refusals with capability grounding data and self-distilled benign data to counter over-refusal. For Qwen3-VL-8B and VisionReasoner-7B, it improves grounding refusal by 77-95 percentage points on VLSU and BBQ-V and by 64-85 points on the held-out Asimov-2.0 domain, while also improving VQA refusal, preserving grounding capability, and keeping over-refusal limited. Representation analysis shows that fine-tuning moves harmful requests toward each model's refusal direction, most strongly for grounding, while leaving benign requests near the harmless reference. 

---
# The GenAI4IDN Benchmark 3.0 - a Public Tool to Assess Generative AI Tools for the Design of Interactive Digital Narratives 

**Authors**: Hartmut Koenitz, Jonathan Barbara, Mirjam Palosaari Eladhari  

**Link**: [PDF](https://arxiv.org/pdf/2610.05633)  

**Abstract**: This paper presents GENAI4IDN Benchmark 3.0, the third iteration of an evaluation framework to assess Generative AI tools for creating Interactive Digital Narratives (IDNs). Moving beyond manual testing, this iteration introduces AI-assisted evaluation through a publicly accessible web application (this https URL), enabling the community to run benchmarks on demand, add new models, and propose new tasks. The revised evaluation framework is "blinded" to avoid model-bias, and can handle complex media such as music, videos, and full IDNs that previously required human raters. A significant addition - responding to concerns raised during ICIDS 2025 - is the addition of fact-checking and bias detection with dedicated tasks and rubrics, validated by human raters with lived experience in the depicted contexts. Findings from a diverse range of models report on maturing creative capabilities while observing runaway thinking and overzealous safety filters as limitations. Fact-checking reliably caught subtle historical inaccuracies, anachronisms, and fabricated claims while the bias rater consistently exposed structural assumptions, tropes, and marginalized group erasures. 

---
# A Framework for Automated Multi-Source Satellite Data Analytics and LLM-Based Report Generation 

**Authors**: Hind Yousif Alhammadi, Isam Mashhour Al Jawarneh  

**Link**: [PDF](https://arxiv.org/pdf/2610.05625)  

**Abstract**: This paper presents the workflow for building an automated ArcGIS Pro tool using ArcPy to extract the Land Surface Temperature (LST) from Landsat 7, 8 and 9 datasets. The tool eliminates the need for manual band selection and repetitive raster computations by automating the multi-step workflow of radiometric calibration, NDVI-based emissivity correction, and thermal conversion. In addition to supporting batch and single-scene processing, the tool has an optional Large Language Model (LLM) for statistical result interpretation and reporting. Depending on batch size, the tool reduced the processing time from around 11-58 minutes when done manually to around 4-11 minutes using the tool. We tested the tool with data from Ras Al Khaimah (RAK) in the UAE, and the LST obtained for Ras Al Khaimah ranged from approximately 25C to 50C, demonstrating an accurate LST mapping compatible with the weather conditions of RAK. In summary, our tool reduces human errors and improves processing accuracy and efficiency for thermal and environmental remote sensing applications, in addition to providing an interactive LLM-based interface for result interpretation. 

---
# DREAM: Dynamic Resolution Assignment For Multimodal Multi-agent Debate 

**Authors**: Khanh-Binh Nguyen, Van Dai Do, Tien Anh Nguyen, Svetha Venkatesh, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2610.05615)  

**Abstract**: Multi-agent debate (MAD) has emerged as an effective paradigm to improve the reasoning capabilities of large language models (LLMs) and is increasingly being extended to multimodal settings. However, existing multimodal MAD frameworks typically expose agents to the same fixed visual input, ignoring substantial variation in the visual scale needed across samples and agents. In addition, these frameworks frequently suffer from groupthink, a phenomenon where agents prematurely abandon correct deductions to conform with confident but hallucinated peer responses. To address these bottlenecks, we introduce DREAM (Dynamic Resolution Assignment For Multimodal Multi-Agent Debate), which operates via two core components: (1) Dynamic Resolution Assignment, a zero-shot probe round where agents test multiple resolutions, quantify uncertainty using Average Normalized Log-Likelihood (ANLL), and use an adaptive threshold to assign each agent to its empirically optimal resolution; (2) Uncertainty-Guided Rollback Aggregation counters groupthink by tracking each agent's uncertainty over rounds and restoring early low-uncertainty answers overridden by group pressure. On six multimodal datasets, DREAM improves the accuracy-token trade-off over multi-agent debate baselines by 1.5-3.2% accuracy without dataset-specific tuning. 

---
# Better Retrieval, Limited Clustering Gains: A Controlled Study of Multilingual Company Entity Resolution 

**Authors**: Yijiashun Qi, Yuxuan Li, Hanzhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.05573)  

**Abstract**: Improved name retrieval may have little effect on company clusters when the pair classifier remains unchanged. We examine this dependency by adapting multilingual E5 encoders under fixed candidate budgets and downstream decision rules. Random-negative and hard-negative training use identical positive schedules. Checkpoints are selected before collecting a new GLEIF sample of 3,633 names, 2,880 source identities and 882 silver-positive pairs. At 72,660 candidate edges, adaptation with a multi-view selector increases direct pair recall from 53.74% to 76.98%. The primary matcher adds only seven correct and two incorrect co-cluster pairs: cluster recall rises from 32.54% to 33.33%, while precision falls from 95.99% to 95.45%. Of 208 newly retrieved silver-positive pairs, 202 fall below its decision threshold. Random-negative and hard-negative training produce identical final partitions. An AI-assisted, single-reviewer audit of 137 pairs supports the observed pattern, although its predominantly LEI-derived evidence does not establish independent gold labels. The results locate the immediate loss of retrieval gains at the existing confirmation stage and show why encoder evaluation must also measure final cluster quality. 

---
# Factoriax: A GPU-Accelerated Factorio-Style Simulator for Reinforcement Learning 

**Authors**: Mickey Beurskens, Tristan Tomilin, Thiago D. Simão  

**Link**: [PDF](https://arxiv.org/pdf/2610.05569)  

**Abstract**: We introduce Factoriax, a GPU-accelerated factory-building simulator written in JAX. In Factoriax, an agent must collect resources, build machines using those resources, and then automate the collection and crafting process by arranging machines on the map to build production pipelines. This paper discusses the structure of the Factoriax simulator and an initial benchmark called Easy Rocket in which an agent is tasked with building a resource-intensive machine called the Rocket in a limited number of game ticks to escape the planet. We also publish results from a number of PPO-based training runs on Easy Rocket. Factoriax is built to be fast. A 1-billion-step PPO training run, equivalent to 500,000 episodes, runs on Easy Rocket in about 8 minutes on a single NVIDIA A100. A standard laptop GPU can complete the same run in about 84 minutes. Our trained PPO agent learns to gather resources, craft machines from those resources, and place them on the map through a curriculum reward directly tied to a manually designed set of achievements. After training, the agent does not place machines in a functional spatial configuration, failing to fully complete the benchmark, and leaving the challenge open for future attempts. 

---
# LifeLong Digital Twin: A Unified Modeling Paradigm and Agent Harness for Event-Driven Lifelong Health State Trajectories 

**Authors**: Jin Jiang, Sean Yates, Jasper Chong, Raymond Brooks, Alex Lawson, Yuqin Qin, Liangcai Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05566)  

**Abstract**: Human health is a continuous, dynamic trajectory shaped by the cumulative interplay of biological processes, clinical events, behaviors and environmental exposures across the life course. Unifying the full breadth of lifelong health information, including longitudinal records, genetic variation, molecular profiles and environmental histories, is essential for whole-person modeling and remains a major challenge. We introduce LifeLong Digital Twin, a unified, event-driven modeling paradigm that organizes Life Events into daily Health States and accumulates them into Lifelong Health Context. An accompanying Agent Harness incorporates multimodal evidence beyond the language model's textual context. We evaluate four language models across 25 disease endpoints on three tasks: Disease Trajectory Forecasting, Disease Risk Ranking and Multi-horizon Disease Prediction. The approach yields marked gains over the reference condition: model-averaged F1 increases by 22.0% for disease identification in trajectory forecasting and 18.3% for five-year disease outcomes; thyroid-disease F1 reaches 0.669. The framework provides a foundation for whole-person digital twins and research on personalized lifelong disease prevention. 

---
# More Claims, Less Evidence: Bounded Verification of AI-Generated Digital Knowledge Artifacts 

**Authors**: Feliks Bańka, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2610.05547)  

**Abstract**: Digital libraries, repositories, and AI-mediated knowledge services increasingly rely on generative systems to produce summaries, descriptions, and other multi-claim knowledge objects. Yet generation can scale far more easily than verification capacity: a reviewer may need to decide whether an object is suitable for publication or downstream use after checking only a small fraction of its claims. This creates a fundamental gap between claim-level verification and confidence in the object as a whole. The central question is therefore what a successful partial check implies about the reliability of the complete artifact when its size grows but the verification budget does not. This paper contributes a Bayesian model of bounded verification centered on the Predictive Value of Pass (PVP). The model predicts that evidentiary value decreases as artifacts grow under fixed verification capacity, improves with larger verification budgets, and is especially fragile when errors are sparse. Controlled experiments on FEVEROUS and FEVER support these predictions and show that adding supported claims around a fixed number of false or unsupported claims can make passing more likely while making a pass less informative. The model further yields the minimum verification budget required to maintain a target PVP, providing a practical component for AI-assisted quality-assurance workflows in which generated knowledge objects must be checked before publication or downstream use. 

---
# DelegationBench: Measuring When AI Agents Should Ask Before Acting 

**Authors**: Shiva Pochampally  

**Link**: [PDF](https://arxiv.org/pdf/2610.05532)  

**Abstract**: AI agents that send emails, edit files, and make purchases must decide when to act on their own and when to check with the user first. This decision is usually evaluated by showing a model a proposed action, asking whether it should proceed, and scoring agreement with human labels. We introduce DelegationBench to test whether such scores can be trusted. It has 156 scenarios with four possible responses (act, ask for permission, ask for missing information, refuse), and most scenarios come in matched pairs that change a single feature: whether the action was requested, what is at stake, whether it can be undone, or who will see it. Across ten models from five families, agreement scores mislead in three ways. A simple keyword rule, which we wrote after seeing the benchmark, agrees with our annotators more often than eight of the models, yet its decision changes in only 9 of 48 matched pairs. Equivalent ways of asking the same question change how often a model acts by up to 52.5 percentage points. And every model stops to ask the user less often when it must carry out the task with tools than when it judges a proposed action. When rules are stated explicitly, the same models follow them almost perfectly, so the gaps are not explained by a general inability to follow rules. We release the benchmark and evaluation tools and recommend reporting these properties separately rather than as one score. 

---
# Have I Scene This Before? Spatially Grounded Conversational Memory for Complex Queries in Egocentric Assistants 

**Authors**: Jiazhou Liang, Liam Gallagher, Kiko Chen, David Guo, Armin Toroghi, Yifan Simon Liu, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2610.05526)  

**Abstract**: Egocentric assistants must connect what users say with what they see across long interaction histories. We formalize this challenge as Spatially grounded Conversational Reasoning (SpaCR): cross-scene, recall-oriented, and counterfactual spatial queries that combine user-stated facts with geometric evidence. Direct vision-language models incur high inference costs and context limits as histories grow, while keyframe selection and retrieval can omit objects or evidence needed for complete recall. We propose Spatially grounded Conversational Memory (SpaC-MEM), an object-centric working memory that uses 3D reconstruction and segmentation to ground conversational information in persistent physical objects. It compresses multimodal histories while preserving spatial evidence and allowing object-specific facts to be updated through dialogue. We also introduce Ego-SpaCR, a benchmark comprising 620 ScanNet video sessions augmented with 95 task-oriented conversations and 3,100 evaluation queries. SpaC-MEM achieves the highest overall answer accuracy among the evaluated methods and improves object recall while requiring substantially fewer reference input tokens than native-video baselines. Removing 3D spatial information substantially degrades performance, highlighting the importance of preserving spatial and conversational evidence together. 

---
# What Does Fréchet Distance Measure? A Directional Decomposition 

**Authors**: Yunghee Lee, Jaeyeon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.05518)  

**Abstract**: The Fréchet distance is a de facto standard for evaluating generative models across domains, appearing as FID for images and FVD for videos. It summarizes the discrepancy between generated and reference distributions in a single scalar, with lower values typically interpreted as better generation quality. However, this scalar view can obscure what drives the comparison. For example, in COCO dataset, increasing the number of diffusion sampling steps improves ImageReward scores yet worsens (increases) FID. Motivated by this mismatch, we seek to make the Fréchet distance more interpretable by uncovering where the discrepancy lies. To this end, we introduce directional Fréchet distance, the expected squared projection of the optimal transport displacement onto a given direction. Across our image, video, and protein case studies, we find that a small number of interpretable directions account for much of the distance. We use these directions to explain the FID increase in terms of semantic concepts represented by CLIP embeddings, quantify FVD's bias toward per-frame appearance, and revisit the interpretation of Protein FID. We open-source our codebase at this https URL. 

---
# The Functional Structure of Post-Compression Recovery in Low-Rank LLMs 

**Authors**: Zishan Shao, Liang Tian, Georgiy Zemlevskiy, Kangning Cui, Lixun Zhang, Yixiao Wang, Ting Jiang, Jinhee Kim, Yixuan Chen, Rui-Feng Wang, Fan Yang, Hai Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.05504)  

**Abstract**: Different low-rank compression methods can produce compressed LLMs that respond differently to the same post-compression recovery procedure, and relative advantages observed between methods at the compression endpoint may shrink, grow, or even reverse after recovery. We ask whether this recovery heterogeneity reflects functional structure beyond scalar loss evolution, and how that structure evolves throughout recovery. Our results establish that this heterogeneity reflects a reproducible compression-induced functional structure, which we formalize as recovery pressure. To characterize this structure consistently throughout recovery, we develop a standardized functional characterization within each backbone that is applicable across heterogeneous low-rank methods. The primary backward characterization reveals reproducible module-wise structure across independent probes, while a complementary forward-only characterization recovers related structure without loss or backpropagation. We further find that recovery pressure measured at the endpoint is associated with subsequent recovery response; during recovery, its module-wise structure is reorganized non-uniformly, and localized updates induce distributed responses beyond directly updated modules. Further evidence indicates that tracking this evolving structure provides a complementary functional view of recovery progress alongside scalar loss. 

---
# SkillGATE: Gate-Aware Monte Carlo Tree Search for Skill Retrieval 

**Authors**: Rongchen Zhao, Yu Chen, Yanming Yang, Shijia Xu, Juyuan Wang, Jin Xu, Zibin Zheng, Jingping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05489)  

**Abstract**: Skill Retrieval (SR) aims to identify the most relevant skills from external skill libraries, and becomes increasingly challenging as libraries grow in scale and diversity. Existing methods either rank skills independently or rely on predefined graph propagation and hierarchical routing, making them vulnerable to semantic distractors, local trapping, and early routing errors. We formulate SR as an adaptive information-foraging process that coordinates region-level navigation with skill-level selection according to the utility and uncertainty observed during search. Based on this formulation, we propose SkillGATE, a graph-guided hierarchical retrieval framework with Gate-Aware Monte Carlo Tree Search (MCTS). SkillGATE constructs a graph-preserving hierarchical index and performs adaptive retrieval through selection, expansion, simulation, and backpropagation. G-PUCT guides action selection, expansion explores new regions, simulation evaluates candidate skills, and backpropagation updates search statistics. Experiments on six SR benchmarks show that SkillGATE consistently improves diverse retrieval and reranking backbones, achieving a 16.3\% improvement in overall R@1 over the strongest retriever-based baseline. Our code is available at this https URL. 

---
# CodeForge-MA: Execution-Verified Multi-Agent Learning with Language-Conditioned LoRA for Multilingual Code Generation 

**Authors**: Zhizhou Gu, Xianting Wu, Siyu Gu, Tian Zhang, Kejian Tong  

**Link**: [PDF](https://arxiv.org/pdf/2610.05481)  

**Abstract**: Large language models for code generation often fail on execution, multilingual coverage, and contamination control, especially under frozen backbone constraints. We present CodeForge-MA, a unified framework that improves code synthesis through a multi-agent data forge, execution verified reinforced instruction tuning, and a language conditioned mixture of LoRA adapters. Four specialized agents, Composer, Reviewer, Executor, and Curator, iteratively refine instruction code pairs, validate them with tests, and filter duplicates and benchmark leakage. During training, we combine masked supervised fine tuning with a test driven reinforcement objective to align generations with executable correctness. For the larger model, we use sparse expert routing over low rank adapters to improve cross language transfer while keeping the base model unchanged at inference. Experiments show that joint data, objective, and adapter design yields robust gains across programming languages. 

---
# Hierarchical Reinforcement Learning with Stable Temporal Abstraction for Language Model Agents 

**Authors**: Shayan Mohajer Hamidi, Yize Cheng, Yuanda Xu, Zhengze Zhou, Alborz Geramifard  

**Link**: [PDF](https://arxiv.org/pdf/2610.05473)  

**Abstract**: Hierarchical reinforcement learning improves long-horizon control by organizing primitive actions around persistent subgoals and assigning credit at multiple temporal scales. Recent hierarchical language agents bring these benefits to interactive tasks by explicitly separating subgoal planning from action execution. We observe, however, that an explicit hierarchy does not by itself determine how stable the resulting temporal abstraction is: the learned boundary policy may replace the subgoal almost every turn, making it effectively transient, or retain a subgoal after it has stopped being appropriate. We call this temporal abstraction instability. We propose Stable Temporal Abstraction via Constrained Optimization (STAC), a constrained boundary-policy optimization method that represents premature replanning and stale persistence as constraint costs. STAC applies the resulting Lagrangian costs only to the sampled boundary decision, leaving the underlying algorithm's rewards, critic targets, subgoal advantages, and primitive-action advantages unchanged. Across two backbones and two benchmarks, STAC improves success over a strong hierarchical baseline by $8.1$ and $7.9$ points on ALFWorld and WebShop with Qwen3-0.6B, and by $23.5$ and $15.8$ points with Llama-3.2-1B-Instruct. 

---
# Hallucination Across the Reasoning Lifecycle: Interface Visibility, Causal Evidence, and Release Control in Large Reasoning Models 

**Authors**: Zhe Yu, Mohan Li, Lei Yu, Ka-Ho Chow, Chengwei Qin, Xingyu Wu, Wenpeng Xing, Shuguang Xiong, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.05472)  

**Abstract**: Reasoning errors can propagate into later decisions and memory. This survey synthesizes 312 papers and first-party reports on text-based reasoning hallucinations around three questions: what evidence is observable, what study designs establish, and which corrective actions the evidence supports. UIPCA records unsupported premises (U), invalid inferences (I), dependent reuse (P), visible answer-trace consistency (C), and action-policy failures (A). Across 58 reviewed sources, no comparison establishes that a specified intervention improves reasoning while reducing factual reliability under matched conditions. The synthesis connects diagnosis to verification, repair, selective release, and persistent-state control across memory, tools, and training feedback. 

---
# AI Safety via Debate is Compromised by Cognitive Biases 

**Authors**: Gefei Liu, Sonya Rashkovan, Sophia Lloyd George, Isaac Sheidlower, Serena Booth  

**Link**: [PDF](https://arxiv.org/pdf/2610.05461)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has played a central role in making large language models responsive to human instructions. However, human evaluators often favor flattering or persuasive responses over truthful ones, creating incentives for models to appeal to evaluators at the expense of accuracy. AI safety via debate has been proposed as a way to improve the supervision of language models: in this paradigm, two agents argue opposing positions and challenge each other's claims, potentially exposing falsehoods to the adjudicator. A central premise of AI safety via debate is that truthful arguments are easier to defend than false ones under adversarial scrutiny. In this work, we investigate whether this advantage persists when debaters use rhetorical strategies that exploit biases in human judgment. Inspired by competitive debate, we construct 68 LLM-generated dialogues about detective mysteries with known culprits, spanning four interventions: anchoring, fallacy oversight, pro-jargon, and verbosity. We apply each intervention to either the side advocating for the true culprit or the side advocating for an innocent suspect, allowing us to distinguish influence on adjudication from correctness. In a study with 369 participants, we find that, pooled across bias types, these interventions significantly shift judgments toward the manipulated side. These findings expose a vulnerability in debate-based supervision: human adjudication is sensitive to manipulative rhetorical strategies. 

---
# TeleTune: Evolving Agent Skills From Offline Telemetry 

**Authors**: Justin Chih-Yao Chen, Elias Stengel-Eskin, Yan Chen, Pol Llado, Scott Counts, Mohit Bansal, Benjamin Van Durme, Harsh Jhamtani, Gaurav Verma  

**Link**: [PDF](https://arxiv.org/pdf/2610.05437)  

**Abstract**: Computer-use agents need to capture procedural knowledge of how people use software. User telemetry offers a scalable source of this knowledge. However, learning reusable skills from these logs requires addressing three challenges: (1) Goal Underspecification, since logs do not record the goal behind each action; (2) Non-Replayability, since past activity cannot be replayed to evaluate skill updates; and (3) Interleaved Trajectories, since logs may mix several tasks without marking their boundaries. To address these, we introduce TeleTune, a framework for learning a textual skill library from offline logs without recorded goals, cannot be replayed during optimization, and may interleave tasks. TeleTune uses action-prediction errors on logged trajectories to propose library edits and keep only those that improve held-out action-prediction accuracy, which we call skill-guided progress. The learned workflows also enable retrieval of demonstrations that cover the subgoals of a new task. At test time, the agent is provided with the learned library and the workflow-based retrieved demonstrations. Experiments on WorkArena and Online-Mind2Web show that TeleTune outperforms random retrieval, Agent Workflow Memory (AWM), and their combination. We find that the best baseline varies by setting, whereas TeleTune achieves average success rates of 77.1% and 80.6%, respectively, improving over the strongest baseline on each benchmark by 6.7% and 7.7%. Under the heaviest perturbation of the WorkArena training data,TeleTune keeps the highest average success rate at 68.5%, 6.3% above the strongest baseline. Our analyses show (1) skill optimization and workflow-based retrieval are complementary, (2) optimizing on fixed logs costs 5 to 75 times fewer tokens than validating the same edits with live episodes, (3) skill-guided progress tracks the live success rate. 

---
# PharmAgent: Constraint-Aware Search with Frozen Language Models for Molecular Optimization 

**Authors**: Nihui Shao, Guanxing Chen, Jilong Shi, Zhengyang Bai, Haohuai He, Zhenchao Tang, Qiujie Lv, Yu-An Huang, Zhi-An Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05431)  

**Abstract**: Molecular optimization must improve target activity and satisfy developability constraints within limited evaluation budgets. Classical methods require tailored rules or training to incorporate chemical instructions and property feedback. Frozen language models can condition edits on this information, but need explicit constraint control and relevant experience. We therefore present PharmAgent, a constraint-aware molecular search method driven by adaptive external state. Its Lagrangian controller translates violations in accepted states into accumulated constraint pressure, keeping this history separate from current property measurements. Structure-indexed replay complements this feedback with relevant evaluated transitions that guide subsequent proposals. As a curriculum progressively activates constraints, candidates and the incumbent are compared under the same current objective, and the accepted state determines the next multiplier update. We derive an exact identity that characterizes how accepted-state violations accumulate in the controller's multipliers. Across five tasks with five independent runs, PharmAgent achieves a summed area under the target-score curves (AUC) of 3.9208 in target-only search, improving over MOLLEO by 37.3%. With online constraints, it achieves a property-adjusted AUC of 0.7076, improving over the strongest online baseline, ExLLM, by 53.8%. These results rank first among all evaluated methods in both target-only and constraint-aware search. The online comparison covers all five baseline frameworks. The full system leads every ablation variant in target quality, property-adjusted performance, and Pareto hypervolume. All five molecular cases reach feasible final states, documenting target gains and trade-offs. 

---
# CASE: Cost-Aware Stopping for Efficient Long-Video Agents 

**Authors**: Yiming Du, Chenghao Liu, Zhiyuan Liu, Fangxing Zheng, Zhao Wang, Junnan Nie, Songfang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05400)  

**Abstract**: Long-video agents can actively gather question-relevant evidence, but they typically leave a central decision implicit: when has the agent seen enough to answer? We propose CASE, a plug-in termination framework that frames this decision as policy-conditioned sequential stopping. At each causal checkpoint, CASE combines an auxiliary multiple-choice assessment of accumulated evidence with the host agent's execution state. From complete native trajectories, we construct a cost-aware target that compares answering now with stopping later along the same search path, accounting jointly for answer correctness and the full cost of continued reasoning. A lightweight Ridge regressor learns this decision gap and produces STOP/CONTINUE decisions. We evaluate three vision-language models with VideoSeek and AVP. On Video-MME, end-to-end accuracy changes by +0.67 percentage points on average while CASE reduces model-token use by 53.63%. The same frozen policies then transfer zero-shot to LongVideoBench and MLVU, with end-to-end accuracy changes of +3.38 and +4.58 points while saving 58.78% and 51.28% of model tokens, respectively. Across all agent-model-benchmark combinations, CASE attains the highest accuracy-efficiency Pareto-frontier coverage among the compared stopping methods (83.3%) at the selected operating points. Online execution preserves this favorable accuracy-efficiency trade-off and additionally reduces measured runtime by 54.1% on average. CASE provides a plug-in termination framework for long-video reasoning agents, enabling them to decide when further evidence acquisition is no longer worthwhile. 

---
# MMPostTrainBench: Benchmarking Autonomous Research for Multimodal Post-Training 

**Authors**: Yuxin Liu, Yuxuan Wang, Zhenxin Lei, Lingchen Meng, Yuchong Sun, Junming Lin, Hongcheng Liu, Yunfei Chu, Qize Yang, Jin Xu, Lei Zhang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05398)  

**Abstract**: Autonomous research seeks sustained model improvements through iterative experimentation and feedback. LLM agents show promise in automating machine learning and language-model post-training, but their ability to sustain multimodal improvement remains unclear. We introduce MMPostTrainBench, a benchmark spanning eight tasks in image, audio, video, and joint audio-video understanding and image-grounded software repair. Agents operate from a common base model within fixed budgets, using development feedback before independent evaluation of their submitted models. Evaluation covers target and non-target model outcomes, iterative model improvement and selection, and research integrity. Across all eight tasks, 52.1% of model--task means fall below the base, and evaluated submissions also exhibit non-target regressions. Model performance does not consistently improve across research iterations, and agents do not reliably select the best evaluated candidate for submission; final submissions trail that candidate by up to 5.38 percentage points. Extending autonomous research from text-only to multimodal tasks introduces additional sources of error in perception, cross-modal alignment, and temporal grounding. The observed regressions and selection gaps highlight the need to balance targeted improvements with non-target capability preservation and to retain gains across research iterations. These requirements motivate MMResearch, a multimodal research framework that connects media-grounded evidence to hypotheses and interventions, carries findings across rounds through hierarchical memory, and retains candidates using development evaluation. Added to existing code-agent runtimes, it improves submitted-model accuracy by up to 7.75 percentage points for Claude Opus 4.8 with Claude Code and 2.33 points for GPT-5.6-sol with Codex. 

---
# Sibyl: An Efficient Small-large Model Collaboration Framework for Long-horizon Tasks 

**Authors**: Zhewei Fang, Yuxin Zhang, Zhenwei Shao, Mengze Li, Zheng Lin, Long Chen, Zhou Yu, Zhe Chen, Zhiwen Chen, Zhaode Wang, chengfei lv  

**Link**: [PDF](https://arxiv.org/pdf/2610.05383)  

**Abstract**: Small language models (SLMs) offer a promising foundation for on-device agents through low-latency, resource-efficient inference, yet limited reasoning and planning capabilities constrain their performance on long-horizon tasks requiring multi-step interaction with the environment. Step-level collaboration between SLMs and larger cloud-hosted models can bridge this gap, but identifying states that warrant cloud assistance remains challenging: the contribution of each cloud call is entangled with subsequent actions and can be assessed only from the final task outcome. Compounding this challenge, the SLM must balance two competing objectives: maximizing task success and minimizing cloud calls. To address this, we propose Sibyl, an algorithm that trains SLM agents to selectively consult cloud models at the step level and internalize their guidance for subsequent decisions, achieving strong task performance with minimal cloud reliance. Sibyl follows a three-stage training pipeline that (1) builds a robust base policy through consultation-free self-evolving reinforcement learning (RL); (2) cold-starts consultation behavior via decisive-disagreement state mining; and (3) jointly optimizes consultation decisions and guidance internalization through consultation-aware RL. Experiments on ALFWorld and WebShop demonstrate that Sibyl, using only a 0.6B-parameter model, outperforms state-of-the-art baselines, including agent training and routing methods, by 95.2% and 80.4% in success rate while averaging only 0.8 and 3.9 cloud calls per trajectory, respectively. 

---
# Learning Field Reconstruction from Incomplete Data by Globally Correcting Local Estimates 

**Authors**: Renhao Zhong, Zihan Zhou, Chiyuan Ma, Tianshu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05375)  

**Abstract**: Reconstructing physical fields from training samples that are always incomplete requires learning spatial structure from fragmented this http URL context--query work establishes how held-out observations provide valid training targets, but this does not make the complete-field distribution identifiable when every training field is this http URL finite data, weak evidence of sharp transitions and localized variations can further favor averaged predictions that attenuate local detail.A structural prior is therefore needed to favor plausible completions; local spatial relationships offer one grounded in the this http URL propose a locally constructed, globally revisable estimator that explicitly learns local field estimates and subsequently corrects them using full-domain observations.A shared coordinate-conditioned predictor learns from incomplete patches, allowing relatively well-observed neighborhoods to provide direct supervision of local this http URL overlapping predictions are reconciled into an observation-conditioned consensus field.A full-domain estimator retains the original observations and learns a residual correction around this frozen field estimate, allowing locally constructed structure to be revised by broader this http URL local estimate serves as both an explicit input, accompanied by its discrepancies with the observations, and a prediction starting point that the global model can this http URL three real-world ocean datasets with authentic observation gaps, our estimator achieves the lowest MSE and highest PSNR on withheld source-supported values, reducing MSE by 28.9\%--34.5\% against the strongest external baseline. 

---
# EnGRICH: Enhancing Generative Reward Modeling with Critiques from Humans 

**Authors**: Xuancheng Li, Beining Wang, Haitao Li, Heng Wang, Yujia Zhou, Qingyi Pan, Blaze Chen, Yiqun Liu, Min Zhang, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2610.05370)  

**Abstract**: Generative reward models (GRMs) are important for LLM optimization. Unlike scalar reward models, GRMs generate natural-language critiques alongside preference judgments, providing finer-grained evaluation signals. Their effectiveness depends heavily on critique reliability. However, existing GRM training typically uses final preference correctness as outcome supervision. Because the preference outcome space is highly constrained, unreliable critiques can still yield correct outcomes and thus be reinforced. Recent work leverages human critiques for process supervision, but such critiques are scarce and are often reduced to scalar rewards, leaving their fine-grained evaluative information underutilized. We argue that evaluative criteria learned from human critiques can be generalized to broader outcome-only preference data. To this end, we propose \textbf{EnGRICH}, a GRM training framework that pairs the GRM with a training-time MetaCritic learned from a small set of human critiques. MetaCritic constructs response-specific rubrics and uses them to evaluate the evidence coverage and correctness of generated critiques. The resulting signals provide both process rewards for fine-grained credit assignment and structured guidance for exploring better critiques. During GRM training, MetaCritic is further optimized to generalize human-grounded evaluative criteria to outcome-only data. At inference, the trained GRM operates independently. Experiments across seven reward-model benchmarks show that EnGRICH consistently improves over competitive baselines, while further analyses validate the effectiveness of its core mechanisms. 

---
# AgentDiscover: Autonomous Discovery with Minimal Search Scaffolding 

**Authors**: Mahdi Farahbakhsh, Ilan Sela, Fatemeh Doudi, Vishnu Teja Kunde, Krishna Narayanan, Jean-Francois Chamberland, Dileep Kalathil  

**Link**: [PDF](https://arxiv.org/pdf/2610.05334)  

**Abstract**: Frameworks that use large language models for scientific discovery typically rely on a fixed, human-designed algorithm that decides what the model sees at each step, leaving the model only the role of proposer. The model knows nothing of the search beyond what it is shown. As models grow more capable, a question arises: does a search strategy chosen by a human before the run scale better than promoting the model from proposer to planner and letting it own the search? The Bitter Lesson suggests that choosing the strategy in advance is the kind of hand-designed structure that general methods eventually outscale. We introduce AgentDiscover, in which a coding agent plans the search using its context as working memory, runs experiments, and records every attempt in a database of ideas, candidates, and their relations. This database serves as the agent's long-term memory and is structured so that the selection rules of classical algorithms such as MAP-Elites and Monte Carlo tree search each reduce to a single query, which the agent is free to use, combine, or replace. A server maintains the database and steers the agent after every submission, keeping it on course over long runs. In our experiments, AgentDiscover is more cost-efficient than existing frameworks, reaching better scores at lower cost. On tasks in kernel engineering, biology, algorithm design, and mathematics, AgentDiscover outperforms prior discovery frameworks. Its programs would have placed first among human competitors in seven past AtCoder heuristic contests, and on eleven mathematical and systems optimization tasks it matches or exceeds every baseline that uses the same model. Our code is available at this https URL. 

---
# MESH-Harness: Self-Improving Agent Harnesses via Bandit-Guided Compositional Evolution 

**Authors**: Zhiwei Shang, Yu Huo, Mingrong Gong, An Yan, Zikun Qu, Junhao Dong, Bryan Kian Hsiang Low, Chenglin Wu, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2610.05300)  

**Abstract**: An agent harness is the code that organizes context, maintains state, and coordinates tool calls for a language model. We study how to improve the harness under a limited evaluation budget while keeping model weights fixed. Our method, MESH-Harness, organizes each harness into functional modules with explicit role-specific interfaces, allowing alternative implementations of each module to be substituted and recombined. It uses shared module representations and full-covariance LinUCB to score candidate combinations based on predicted performance and exploration value. Mixed-start coordinate ascent selects complete configurations for evaluation without enumerating the combinatorial space. Validation traces then guide local code edits, and the resulting candidates are incorporated into fixed-capacity role-specific pools for subsequent recombination. On text tasks, retrieval-augmented mathematical reasoning, code generation, and interactive scientific tasks, MESH-Harness outperforms Meta-Harness by 5.70, 7.01, 2.00, and 5.00 points, respectively, under matched candidate-evaluation budgets. Iterative harness optimization improves MESH-Harness by 5.63-7.79 points over its first-round configurations. For the reported configurations, aggregate test-time cost is 44.2% lower than that of Meta-Harness, while total cost including search is 14.6% lower. These results show that combining module-level design reuse with feedback-driven compositional search can systematically improve agent harnesses while keeping overall optimization cost under control. 

---
# Readable Before Actionable: Causal Tracing of Indirect Prompt Injection 

**Authors**: Zhe Yu, Wenpeng Xing, Xingxing Yang, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.05295)  

**Abstract**: Indirect prompt injection causes LLM agents to follow commands embedded in external data. A probe may distinguish instructions from data without identifying a state edit that changes the next action. We study this gap through counterfactual role probes, component-wise activation patching, and separate interventions on AgentDojo trajectories. Role decoding survives changes in content and format. In controlled Qwen tests, it precedes strong tool-choice effects from patches along an independently estimated role direction. On AgentDojo, directions estimated from hijacked and resisted training trajectories reduce attack success at pre-action and injected-span positions, but have little effect at random positions. In longer Qwen trajectories, single-position edits become less effective at later layers; span-wide and repeated edits reduce attack success on the same evaluation set. Removing the learned channel subspace preserves role decoding, yet effective intervention directions transfer poorly across the tested channels. These findings distinguish a readable role signal from an effective behavioral intervention: depth matters in controlled tool choice, while position and context also matter in attack trajectories. 

---
# Fusion is the New Mutation: Bandit-Guided Evolution on Workflow Graphs 

**Authors**: Zhiwei Shang, Jiahang Sun, Mingrong Gong, Mingze Kong, Zikun Qu, Pingchen Lu, Junhao Dong, Zhipiao Liu, Hongwei Yang, Guoqing Xie, Yao Shu, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2610.05284)  

**Abstract**: Automated agentic workflow optimization relies on costly evaluations, making it essential to allocate a limited evaluation budget effectively. Multi-parent fusion can reuse designs from previously discovered workflows, but identifying promising parent combinations requires learning from limited fusion feedback. We introduce DAGO (Directed Acyclic Graph Optimization), a contextual-bandit-guided framework that learns which parent workflows to fuse under a limited evaluation budget. DAGO formulates each candidate parent combination as an arm, represented by pretrained embeddings of its constituent workflows' code and prompts. A diagonal LinUCB policy learns a shared reward model across arms and balances exploitation of arms with high predicted offspring quality against uncertainty-driven exploration. After an arm is selected, an LLM generates a child workflow through summary-guided fusion, and the child's validation score serves as the reward for updating the bandit. A shared directed acyclic graph maintains discovered workflows and their multi-parent lineage, providing an expanding pool of parents for subsequent arm proposals. Across six benchmarks covering mathematical reasoning, code generation, and question answering, DAGO achieves the highest macro-average score among the evaluated baselines. Under matched validation-evaluation budgets, it improves over AFlow from 80.3 to 81.7 while reducing aggregate search expenditure by 11.2%. Ablation studies show that LinUCB-guided arm selection outperforms both random selection and its exploration-free variant, supporting the value of feedback-driven selection and exploration-exploitation balance. 

---
# When Agent Context Goes Stale: Incoherence in Volatile Agent Context 

**Authors**: Yingying Liu, Junzhou Fang, Chenxiong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2610.05281)  

**Abstract**: Modern agents increasingly ground their reasoning in observations returned by tools, such as file contents read from a workspace. However, the data sources underlying these observations may later be modified by users, other agents, or external tools, while the model retains only the stale content in its context window. Existing agent runtimes provide little support for notifying the model that a previously observed fact has become stale, causing agents to reuse outdated observations and make incorrect claims about the current workspace state. We propose Concord, a context coherence framework that maintains the consistency between tool observation in agent context and the mutable sources from which they were derived. Concord links each observation to its source, detects source changes, and uses configurable handling policies to update, annotate, or suppress stale context before reuse. Concord is applicable across different agent runtimes and external resources, and can be easily extended to new runtime-resource settings. We implement Concord as a general framework, and instantiate a concrete use case to assess its effectiveness. We construct ConcordBench, where previously observed file contents become stale after subsequent edits. Across three evaluated frontier models, Concord produces answers consistent with the restored workspace state in all evaluated cases under these constructed conditions, matching the oracle on recover count for this benchmark, while using 46.4% fewer tokens than the strongest non-oracle baseline. 

---
# Learning from imperfect teachers for low-resource acoustic generalization 

**Authors**: Shuanglin Li, Ruxiao Qian, Jian Liu, Haijun Lin, Wenwu Wang, Siyang Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.05256)  

**Abstract**: Knowledge distillation (KD) improves low-resource acoustic learning by enriching one-hot supervision with the softened predictive distribution of a fixed teacher network. However, a teacher trained with limited or imbalanced annotations may produce a biased distribution whose components are not uniformly reliable. Although this distribution can still encode useful knowledge, direct full-distribution matching may also transfer teacher-induced biases, thereby distorting the student's decision boundary and degrading its generalization performance. To address this limitation, we propose Boundary-Anchored Mass-Partitioned Distillation (BA-MPD), a logit-based distillation objective composed of Boundary-Anchored Correction (BAC) and Mass-Partitioned Distillation (MPD). BAC addresses missing ground-truth labels in the set of the teacher's top predictions by swapping the true label for the lowest-ranked entry of the set, thus keeping the mass and uncertainty of the set unchanged. MPD then distills this corrected distribution through separate losses that enforce relational consistency within the set, balance the mass between high- and low-confidence groups, and weight lower-confidence dependencies. Ultimately, BAC and MPD together suppress harmful ranking errors and noisy low-confidence details, while retaining all useful teacher information. Experiments on two acoustic benchmarks under multiple label budgets show that BA-MPD consistently improves over supervised-learning baselines and vanilla KD while remaining competitive with strong logit-based KD baselines. Cross-budget results further show that BA-MPD remains effective when the teacher and student models use mismatched label budgets, demonstrating its ability to exploit imperfect teachers across supervision gaps. Implementation available at this https URL. 

---
# GFGE: Unifying Explainable AI Methods through an Interpretation Framework 

**Authors**: Jinfeng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2610.05225)  

**Abstract**: Explainable artificial intelligence (XAI) encompasses methods that draw on different sources of information and address different explanatory needs. A common framework is needed to describe how this information becomes evidence and is communicated as an explanation for a particular recipient. We propose the General Framework for Generating Explanations (GFGE), grounded in interpretative frameworks and the complementary activities of \emph{sense-reading} and \emph{sense-giving}. Its conceptual foundation is the Interpret/Explain Schema (IES), which connects an analyst's interpretation of system evidence, the communication of a selected account, and the recipient's interpretation of that account. GFGE operationalises this schema through five roles: data interpretation, model interpretation, output interpretation, optional post-hoc analysis, and aggregation. A role-typed operation graph records method-specific dependencies, while evidence records retain the sources, assumptions, and limitations of explanatory claims. The explanatory question, audience, and context guide the procedure. We instantiate GFGE for attribution, surrogate, counterfactual, concept and prototype, intrinsic rule, argumentation, and language-model methods. These instantiations show how intrinsic, post-hoc, and hybrid workflows can be represented through the same roles while preserving their distinct evidential requirements. GFGE provides a common basis for analysing explanation workflows, tracing communicated claims to their evidence, and identifying unresolved explanatory dependencies. 

---
# Look Before You Leap: Thermodynamic Arbitration of Parametric and Non-Parametric Knowledge in LLM Agents via Self-Regulating Memory Architectures 

**Authors**: Akash Das, Ishan Roy  

**Link**: [PDF](https://arxiv.org/pdf/2610.05223)  

**Abstract**: The architecture of modern LLMs consists of a profound cognitive polarization. LLMs possess implicit intuition encoded in their parameters, yet rely on a disconnected, explicit mechanism to access the outside world. Agentic frameworks have not bridged this gap; instead, models are often compelled into pathological "induced amnesia." Under the prevailing "Retrieve-Always" paradigm, agents must distrust their internal knowledge, making every user interaction a "tabula rasa" event that must be checked externally. This creates reflexive dependence that can be thermodynamically wasteful, cognitively fragile, and susceptible to irrelevant context. We propose a return to first principles, operationalizing the biological maxim "Look Before You Leap." We introduce MARTA (Metacognitive Adaptive Retrieval and Thought Architecture), a neuro-symbolic framework that bridges parametric and non-parametric knowledge. Rather than treating retrieval as mandatory, MARTA models it as a cost, taking the leap only when perceived internal inadequacy warrants external information. By allowing the agent to gauge the entropy of its own thoughts before acting, MARTA enables deliberative retrieval and uncertainty-aware decision making. Our approach suggests that giving agents the capacity for introspection can restore a more efficient balance between internal knowledge and external information. 

---
# Safe Context Switching for Agents in the Wild: Mitigating Subspace Interference via Orthogonal Adaptation 

**Authors**: Akash Das, Ishan Roy  

**Link**: [PDF](https://arxiv.org/pdf/2610.05219)  

**Abstract**: Most Large Language Models exhibit a fundamental tension between two sequential tasks, such as logical reasoning and safety alignment. The high-variance internal states required for sophisticated Chain-of-Thought (CoT) deduction can geometrically interfere with latent representations encoding safety constraints. We identify this phenomenon as Sequential Subspace Interference, showing that standard fine-tuning on logical tasks such as multi-step mathematics and code generation can result in a 23.3% interference penalty on alignment benchmarks, substantially weakening the model's safety priors. This Reasoning Drift is not adequately captured by current adaptation methods because gradients for logical tasks are rarely orthogonal to safety objectives. To address this issue, we propose AURA (Adaptive Unique Residual Allocation), a spectral regularization framework that enforces Spectral Independence between reasoning and safety. By explicitly estimating the null space of the alignment manifold and constraining reasoning updates to its orthogonal complement, AURA enables models to improve logical reasoning without compromising safety. Empirically, AURA recovers 23.0% of the lost performance while preserving greater than 0.98 cosine fidelity to the safe state, demonstrating that reasoning and alignment can be effectively decoupled through geometric regularization. 

---
# Image Synthesis as an Intermediate for Controllable Time Series Generation 

**Authors**: Haochen Yuan, Jing Xie, Yunbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05211)  

**Abstract**: Semantic-driven time-series generation offers a promising way to improve downstream learning in few-shot forecasting, but directly generating numerical sequences from language often fails to preserve the intended temporal structure. We propose VisualBridge, which uses time-series plots as a visual intermediate to bridge high-level temporal semantics and numerical sequences. An MLLM first converts plotted series into structured semantic representations, enabling explicit control over temporal properties such as trend, seasonality, and volatility. We then learn a semantic editing policy with downstream forecasting rewards, allowing the generation process to favor temporal patterns that are beneficial for the target task. The resulting sequences are further modeled by a temporal VAE to produce consistent multivariate augmentations. Experiments on standard public forecasting benchmarks demonstrate that VisualBridge improves few-shot forecasting over conventional augmentation methods, with ablations validating the roles of visual semantic grounding, learned semantic control, and VAE-based generation. 

---
# From Scientific Observations to Mechanisms: Benchmarking Hypothesis Generation by AI Scientists 

**Authors**: Xiaxun Xie, Qingqing Long, Meng Xiao, Wei Ju, Yuanchun Zhou, Xuezhi Wang, Hengshu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05197)  

**Abstract**: Data-driven mechanistic hypotheses are essential to scientific discovery because they explain how underlying processes produce observed phenomena. AI agents and AI scientists increasingly support scientific data analysis. However, their ability to turn empirical findings into mechanistic hypotheses remains insufficiently examined. To address this gap, we introduce MechHypoBench, the first benchmark for evaluating whether AI agents and AI scientists can generate such hypotheses from empirical data. It combines paper-derived mechanisms from 14 scientific fields with real-world datasets containing 17.98 million records. The construction retains the observational complexity of empirical data while providing a specified underlying mechanism. Agents analyze the observations and propose open-form hypotheses. We develop an evaluation framework that assesses open-form mechanistic hypotheses through their consequences under withheld conditions. Experiments with general agents and AI scientists reveal a substantial gap between generated hypotheses and the underlying mechanisms. 

---
# AECG: Asymmetric Experience Consolidation and Governance In Multi-Agent Systems 

**Authors**: Ao Tian, Jialong Liu, Daqi Zheng, Xin Sun, Mengting Li, Zhizhao Xiao, Zijian Huang, Honglei Wang, Zijian Hei, Yukun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.05176)  

**Abstract**: Large language model (LLM)-based multi-agent systems increasingly rely on memory to transform execution trajectories into reusable procedural knowledge. Yet repeated retrieval also makes memory errors persistent: memory pollution arises when outdated, weakly supported, or spuriously successful procedures become recurring components of future reasoning. Multi-agent execution introduces an additional structural risk. Scope collapse occurs when procedural knowledge escapes the coordination scope in which it was shown effective and is repeatedly reused at incompatible decision levels, allowing local errors to influence cascades of downstream decisions. Meanwhile, task-level failures provide ambiguous supervision because they rarely reveal which recalled knowledge was responsible. We introduce AECG, a framework for asymmetric experience consolidation and governance for multi-agent systems. AECG turns memory from static experience storage into a dynamic reliability-governance loop, preserving coordination scope and using multi-scale, confidence-aware reliability to detect degradation. It then combines degradation with downstream impact to prioritize high-risk knowledge under a bounded review budget, applies targeted interventions, and reactivates revised skills only after paired replay. Across three multi-agent frameworks and four benchmarks, AECG achieves the best score in 11 of 12 framework--benchmark settings and improves over the strongest competing memory method by as much as 10.23 percentage points; removing scope preservation reduces accuracy by up to 16.89 points. AECG thereby reframes multi-agent memory from passive accumulation into auditable reliability governance. Code is available at this https URL 

---
# CreativeFlow: A One-to-Many Analogical Relation Transfer Method for 3D Asset Generation 

**Authors**: Xuechen Li, Shuai Zhang, Nanxuan Zhao, Qing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.05167)  

**Abstract**: Inspired by cognitive science, we present CREATIVEFLOW, an analogical generation framework that explicitly models analogical divergent thinking to mitigate creative homogenization in text-to-3D pipelines. Our method derives a series of meaningful yet relationally similar source-target asset pairs, each featuring distinct geometric configurations. Expert evaluations demonstrate that our framework substantially enhances creative novelty and visual fascination. This workflow and its resulting assets establish a foundational dataset and benchmark for future relation-aware 3D model training. 

---
# A Safe Action Is Not Enough: Feasible-Future Decoding for Vision-Language-Action Policies 

**Authors**: Tu Nguyen, Matthieu Zimmer, Vu Anh Vu, Ziyi Wang, Jannik Hammel Nielsen, Xuebing Zhou, Haitham Bou Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2610.05166)  

**Abstract**: A safe action is not necessarily a viable one. Under a frozen vision-language-action (VLA) policy, an action can be likely and locally admissible yet leave no policy-supported route to safe task completion. We call this the feasibility-likelihood gap: likelihood ranks the current action, whereas feasibility depends on the futures that remain after it.
We derive the exact next-block marginal of the history-conditioned policy-environment trajectory law restricted to safe task completion. The derivation exposes a candidate-dependent feasible-future mass with two roles: its support records whether safe completion remains possible under the frozen continuation process, and its magnitude measures how much weighted safe-completion mass is preserved. Exact evaluation is impractical online, so we develop a selective finite-candidate approximation, derive conditions for recovering the best retained viable candidate, and instantiate it as an alarm-triggered, training-free reranker.
On Safety-CHORES, VICS-G lowers mean cumulative safety cost by 1.9%-57.5% across six settings while remaining within 2.5 percentage points of policy sampling in success and 0.82 steps in mean episode length. The resulting decoder is tied to an exact policy-relative safe-completion target, yet requires neither retraining of the base policy nor online trajectory rollouts. 

---
# Memadapter: Counterfactual Adaptation Against Memory-induced Sycophancy 

**Authors**: Ruqing Ning, Haibo Meng, Zhishang Xiang, Zerui Chen, Jinsong Su, Xin Wang, Qinggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05162)  

**Abstract**: Long-term memory enables LLM-based agents to retain and reuse information across tasks and sessions, supporting personalization and long-horizon interactions. However, persistent memories can also induce sycophancy, causing agents to over-align with users' historical beliefs even when they are inaccurate, outdated, or inconsistent with objective evidence. Existing mitigation methods assume that memory-induced sycophancy originates from biased or incorrect memories and attempt to reduce this risk by filtering such memories at different stages of the memory pipeline. However, in the real world, objective and correct memories can still induce sycophancy, and the same memory can warrant different influence across different contexts. To this end, we propose MemAdapter, a novel framework that adaptively integrates retrieved memories to support objective and reliable reasoning. Specifically, MemAdapter consists of three components: (i) Counterfactual Induction, which leverages counterfactual reasoning to uncover the potential risk of retrieved memories; (ii) Context-Aware Reflection, which calibrates the inferential influence of each retrieved memory in light of the current task via self-reflection; and (iii) Evidence-Based Reasoning, which grounds the final response in appropriate evidence while preserving the legitimate influence of memory. Extensive experiments on three benchmarks demonstrate that MemAdapter consistently improves memory reliability across diverse scenarios. Our code is available at this https URL. 

---
# Beyond Instruction Following: Learning Grounded Skill-Following with Skill Contracts 

**Authors**: Jianghan Shen, Zhenjie Liu, Yue Li, Jie Huang, Siqi Luo, Yiming Cheng, Yizhi Yao, Kaijie Zhang, Cheng Tang, Minghui Zhang, Ming Hu, Yirong Chen, Ziyan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05161)  

**Abstract**: Instruction following typically enforces discrete, response-level requirements, whereas an expert-authored skill prescribes procedural requirements spanning multiple phases and environment interactions. Given such a skill, we train the executor to execute all required phases instead of focusing solely on the final answer. We therefore introduce Grounded Skill-Following, which requires an agent to execute a fixed, expert-authored skill across its required phases by grounding decisions in environment observations. To achieve verifiable procedural execution, we formulate each skill as a skill contract combining visible skill instructions with an explicit contract runtime. The runtime specifies required phases, admissible actions, permitted transitions, and accepted termination. This structure provides a dense, verifiable training signal throughout execution. We leverage this by introducing Verified Progress Credit, which assigns rewards upon the initial completion of contract milestones and aggregates them into the trajectory return to guide policy optimization. During rollout, the contract runtime continuously tracks state transitions to provide Contract-State Feedback, which indicates whether the latest action is accepted and guides the agent toward valid next actions. To measure procedural compliance, we introduce the Protocol Completion Rate (PCR), defined as reaching accepted termination through all required phases, and decouple it from the final Task Outcome. Jointly trained with our framework, Qwen3.5-4B achieves Protocol Completion Rates of 99.27% on Math and 99.96% on Search, while slightly outperforming original baselines in Task Outcome (82.95% and 46.61%, respectively). Controlled studies examine how skill instructions, training signals, and contract-state feedback affect both metrics, while withholding interventions evaluate behavioral dependence on observation content. 

---
# R1A-PC: Physics-Guided Electromagnetic Inversion of Three-Dimensional Human Point Clouds in Complex Static Environments 

**Authors**: Xudong Yuan, Ruyun Xu, Jingtai Yang, Xianzheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.05144)  

**Abstract**: Recovering three-dimensional human geometry from electromagnetic measure?ments in a complex static environment is difficult because strong multipath responses from walls, floors, and other objects obscure the weak target per?turbation. We propose R1A-PC, a physics-guided method that reconstructs a 2048-point human cloud from paired complex fields measured with and without the target. Complex background subtraction emphasizes target-induced ampli?tude and phase changes, while the background field remains available as an environmental condition. A frequency-balanced discrete Born adjoint produces a three-dimensional spatial knowledge map. At each of two bounded deformation stages, the decoder combines complex measurement features, background fea?tures, and multiscale physical features queried at the current point coordinates; the second stage queries again after the first coordinate update. We analyze the residual of paired subtraction, the weighted normal-operator structure of the raw adjoint, and the feasible set of the predicted cloud. In a held-out background generated by full-wave simulation under a fixed acquisition geometry, R1A-PC obtains a squared Chamfer distance of 0.001434 m2 and an F-score of 0.963080 at 0.05 m. Compared with TopNet, the Chamfer distance decreases by 70.28%. Removing physical guidance or background subtraction increases the Chamfer distance by 242.06% or 241.18%, respectively. Experiments across background layouts and poses support the complementary roles of paired subtraction and position-dependent adjoint features. 

---
# AutoSciBench: Autonomous Benchmark Generation for Evaluating Scientific Agents 

**Authors**: Dongki Kim, Namkyeong Lee, Surag Nair, Carl Edwards, Xiner Li, Edward De Brouwer, Jenna Lynn Collier, Sung Ju Hwang, Gabriele Scalia, Ehsan Hajiramezanali  

**Link**: [PDF](https://arxiv.org/pdf/2610.05140)  

**Abstract**: As agents rapidly evolve, existing benchmarks can become saturated, limiting their ability to distinguish capabilities and reveal remaining failure modes. Particularly in scientific domains, constructing and updating benchmarks requires substantial time, labor, and domain expertise, making it difficult to keep evaluation aligned with advances in agent capabilities. We address this challenge by investigating whether scientific-agent benchmarks can be automatically generated and iteratively adapted as agent capabilities evolve. We introduce AutoSciBench, a framework that represents each task as a high-level concept specifying the scientific domain, data modality, and required reasoning approach, together with a low-level recipe specifying how the question, environment, and ground-truth answer are constructed and verified. Agents attempt to solve each task, producing solver trajectories and corresponding judge feedback which AutoSciBench uses to revise the recipe or concept, closing observed shortcuts and shifting tasks toward raw-data re-examination, interpretation of intermediate results, and evidence integration. Experience distilled from completed refinement trajectories further guides new concept generation, allowing lessons from earlier task refinement to inform subsequent benchmark construction. Starting from existing benchmarks, we evaluate AutoSciBench across computational biology, materials science, and clinical imaging. Generated benchmarks reduce average solver accuracy by 22.4 and 25.5 percentage points relative to the human-curated benchmarks in computational biology and materials science, respectively, while generated tasks receive higher average quality ratings across all three domains, suggesting that scientific-agent evaluation can adapt as agent capabilities advance. 

---
# Memory Canonicalization: A Framework and Benchmark for Cross-Model Drift in Persistent LLM Memory 

**Authors**: Amit Vadnere, Aishwarya Lonarkar  

**Link**: [PDF](https://arxiv.org/pdf/2610.05124)  

**Abstract**: Persistent memory for Large Language Models (LLMs) has matured rapidly: systems such as MemGPT/Letta, Mem0, and Zep now provide agents with tiered, temporally-aware, model-agnostic external storage, while the Model Context Protocol (MCP) standardizes access to memory servers. A less addressed problem is that an identical stored memory object, retrieved by two different LLMs under otherwise identical conditions, may not be interpreted the same way, factually or emotionally. This paper proposes memory canonicalization: a write-time pipeline that detects ambiguity, conditional structure, and emotional loading in a raw memory object and rewrites it into an explicit, structurally disambiguated canonical form, with emotional valence represented as a separate field rather than inferred from tone. We formalize the pipeline, define a companion Cross-Model Semantic Drift / Emotional Consistency Score benchmark (CMSC-E), and report results from a three-arm pilot using 176 synthetic memory objects and three downstream model families. We find an uncorrected improvement in cross-model emotional consistency for fully canonicalized memory relative to raw memory (+0.050, 95% bootstrap CI [0.013, 0.086], paired t-test p = 0.010), but this result does not survive Bonferroni, Holm, or Benjamini-Hochberg correction across the six comparisons tested. None of the factual-drift (CMSD) comparisons reach significance at any correction level. We report these results as exploratory rather than confirmatory and outline needed follow-up work, including larger samples, independent judge models, human-validated rendering, and preregistration. 

---
# LexiHorizon: Stabilizing Reinforcement Learning for Long-Horizon Deep Search 

**Authors**: Zhiqing Nong, Liang Wen, Chao-Hsuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05119)  

**Abstract**: Deep search agents tackle complex knowledge tasks through iterative retrieval, multi-hop reasoning, and evidence synthesis across multiple sources. Existing approaches typically assume relatively stable retrieval systems and operate over short-horizon tool interaction. However, when retrieval is sensitive to query formulation, even a semantically appropriate query may fail to surface critical evidence because of mismatched entity names, aliases, or keyword combinations. Recovering from such failures requires repeated query reformulation and longer interaction trajectories. This setting poses a distinct training challenge, as the policy must sustain long-horizon query exploration while managing an expanding volume of retrieved content. We propose LexiHorizon, a framework for training search agents over long horizons that expands the trajectory context budget, manages accumulated retrieval content using a window over recent tool observations while preserving the reasoning history, and introduces an outcome-gated search-effort reward that provides a bounded bonus for tool invocations to trajectories with nonzero answer reward. Experiments on XBench, WebWalkerQA, and BrowseComp-ZH show that the resulting 9B model consistently outperforms both its base model and MiroThinker-1.7-mini, with maximum absolute gains of 8.7 and 23.8 percentage points, respectively. These results suggest that combining an extended context budget with reasoning-preserving context management benefits long-horizon deep search agents. 

---
# SharpDraft: Accelerating Long-Context Speculative Decoding with Cardinality-Aware Query Scaling 

**Authors**: Jeonghoon Park, Seongwoon Jo, Jongwon Lee, Taesik Gong  

**Link**: [PDF](https://arxiv.org/pdf/2610.05106)  

**Abstract**: Long-form reasoning makes inference expensive, and speculative decoding mitigates this cost by verifying multiple draft tokens in parallel. Its speedup, however, can fade as context grows and draft acceptance declines. We focus on attention-mass dilution: as softmax normalizes over more visible Keys, the mass concentrated on the highest-scoring Keys can decrease. We introduce SharpDraft, a training-free method that counteracts this effect through cardinality-aware Query scaling, without the computational overhead of online adaptation. Under explicit assumptions, we derive an exact top-$k$ mass correction and deploy a closed-form fixed-slope approximation. Across AIME-26, GPQA-Diamond, and LongGenBench Diary, SharpDraft achieves $2.59$-$3.19\times$ geometric-mean end-to-end speedups over target-only autoregressive decoding when applied to DFlash, PARD, and EAGLE 3.1. With DFlash, it improves decoding speed and outperforms full-parameter and LoRA-based online adaptation in end-to-end speedup, while matching the unmodified drafter's reported peak allocated GPU memory. 

---
# Why, Where, How: Taxonomy-guided Error Grounding for Code Repair in NL2SQL 

**Authors**: Suchan Lee, Woomin Song, Hwanjo Yu, Sangwoo Mo  

**Link**: [PDF](https://arxiv.org/pdf/2610.05060)  

**Abstract**: SQL queries that large language models write from natural language questions can execute successfully yet produce incorrect results, so execution alone does not reveal what to fix. An error taxonomy says why the query is wrong, but not where to look or how to change it. Existing methods can guide SQL correction through feedback, error reports, or generated plans alongside an unmasked query. We introduce TEG(Taxonomy-guided Error Grounding), which turns a supplied diagnosis into a structured correction input for natural language-to-SQL (NL2SQL) correction. Type-specific rules map each error type to construct classes to reconsider and an edit operation to request. TEG masks the selected constructs in the query when applicable and states that operation in an edit instruction. TEG generates candidate corrections from this input, uses execution feedback to guide candidate selection, and repeats the process one annotation at a time for queries with several errors. On NL2SQL-BUGs, TEG reaches 47.3 single-error execution accuracy and 37.0 overall with Qwen2.5-7B-Instruct. Across the model sizes and thinking modes evaluated in the main comparison, TEG outperforms all evaluated baselines on single-error queries, even when the baselines receive the same error-type annotations. With predicted types, TEG stays above direct LLM correction and ErrorLLM on single-error queries. 

---
# CI-JEPA: A Counterfactual Analysis of Latent Representations in Joint-Embedding Predictive Architectures for Self-Supervised Learning 

**Authors**: Mintu Dutta, Ritesh Vyas, Mohendra Roy *  

**Link**: [PDF](https://arxiv.org/pdf/2610.05043)  

**Abstract**: Self-supervised visual representation learning learns useful features without manual annotations during representation training. The image-based joint-embedding predictive architecture (I-JEPA) predicts latent representations of masked image regions, but its objective does not explicitly model responses to specified visual interventions. We introduce CI-JEPA, a counterfactual intervention-aware extension that learns to predict the representation change $\Delta Z = Z_{\mathrm{CF}} - Z$ between an original image and a modified counterpart. We assess representation robustness through selective sensitivity: stronger responses to task-relevant semantic changes than to nuisance changes. Experiments on Flowers102 use flower-center occlusion as a candidate semantic intervention and background blur and tint as candidate nuisance interventions. With frozen-encoder linear probing, CI-JEPA achieves a best validation accuracy of 78.14\%, compared with 77.55\% for both the pretrained ViT-B/16 and the I-JEPA baseline, a gain of 0.59 percentage points. The reported mean $L_2$ representation changes are 4.42 for center occlusion, 3.48 for background tint, and 2.83 for background blur. This ordering is consistent with relative semantic selectivity for the evaluated interventions, rather than complete nuisance invariance. The accuracy comparison is complementary and does not establish improved robustness over the baselines. These controlled image modifications provide a framework for studying intervention-induced changes in JEPA representations; they do not establish causal feature discovery or robustness to all visual changes. 

---
# EVISKILL: Grounding Skill Evolution in Replayable Evidence 

**Authors**: Yan Zhou, Yili Wang, Yiwei Dai, Qinggang Zhang, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05030)  

**Abstract**: Continual skill evolution enables LLM agents to accumulate and refine reusable procedural knowledge from interaction experience without updating model parameters. Its effectiveness depends on determining not only what to change, but also why a change is justified and when it should become persistent guidance. However, existing experience-driven methods can lose the behavioral evidence and task contexts supporting edits. Moreover, a global validation outcome provides an incomplete judgment of its constituent changes: locally supported corrections may be discarded with a rejected revision, while evidence may require further experience to inform useful updates. To this end, we introduce EVISKILL, an evidence-driven framework that organizes execution observations into Replayable Evidence Cards and synthesizes edits with explicit links to their supporting contexts. Targeted replay verifies these edits through re-execution and provides feedback for correction. Across epochs, EVISKILL preserves evidence and provisionally retains supported edits for further refinement, while global validation governs their incorporation into the final skill. Experiments on three interactive benchmarks across six LLM backbones demonstrate the effectiveness of this approach. 

---
# How corner is a corner case? Percentile control for highway scenario generation 

**Authors**: Jiaxi Liu, Hang Zhou, Hangyu Li, Yifan Wang, Keke Long, Chengyuan Ma, Bin Ran, Xiaopeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.05003)  

**Abstract**: Generating corner-case scenarios with appropriate adversity in a simulation environment is critical for testing an autonomous vehicle (AV) software stack's safety performance before deployment. Existing autonomous-driving scenario generators can enforce specific behavior, adversity, or feasibility conditions, but they provide limited control over how extreme a generated scenario is relative to plausible futures in the same traffic context. This study represents the adversity of a generated scenario as its percentile in the conditional distribution of future risk given the observed history. This view supports calibrated answers to two questions: how "corner" a generated corner-case scenario is and how its "cornerness" can be fine-tuned.
To this end, we formulate history-conditioned risk-percentile requests and learn a reference risk distribution that maps each requested percentile to a physical risk target. We then use a percentile-conditioned joint diffusion model with sampling-time risk guidance to generate multi-agent futures, together with a reference-based criterion for evaluating percentile realization.
Experiments use the minimum post-encroachment time (PET) between the ego and its surrounding vehicles as the risk surrogate on highD. On the primary evaluation set, our method realizes 1,422 of 1,440 requests within a 0.05 percentile tolerance (98.75%), with mean percentile error 0.00673 and PET-target error 0.00991 seconds. The resulting interface connects context-relative risk specification, physical realization, and evaluation through a common risk scale. Project website and videos of generated scenarios are available at this https URL. 

---
# EmoRSS: Mitigating Emotion-Induced Over-Refusal in Large Language Models 

**Authors**: Shuyi Miao, Yaojin Ma, Chenhang Cui, Xiaohao Liu, Dang Jisheng, Shengda Zhuo, Fei Shen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2610.04998)  

**Abstract**: Emotional expression can influence the safety decisions of large language models (LLMs), offering a potential avenue for improving safety alignment. Existing studies have mainly focused on how emotional expressions facilitate attacks under harmful requests, while overlooking their effects on benign requests. We find that emotional expression can also systematically increase refusal tendencies on benign requests, leading to unnecessary over-refusal. Based on this observation, we propose emotion-guided refusal subspace steering (EmoRSS), an activation-steering method that mitigates emotion-induced over-refusal while preserving refusal behaviour on harmful requests. Specifically, we first identify a refusal-sensitive layer using layer-wise linear probes and construct a refusal subspace from sparse autoencoder (SAE) features aligned with the probe direction. Next, we use paired regular and emotional requests with the same queries to estimate the mean activation shift in the features defining the refusal subspace. Finally, we decode this shift into an activation intervention vector and apply it in the reverse refusal direction during inference, without updating the backbone parameters. Experiments on two LLMs show that, when requests contain emotional expressions, our method achieves a more favourable trade-off between refusing harmful requests and answering benign ones than prior over-refusal mitigation baselines, while better preserving general task performance. 

---
# CIPO: Counterfactual Imagination Policy Optimization for Adaptive Tool Granularity Selection 

**Authors**: Yu Li, Yunlu Wan, Zijian Zhu, Han Luo, Chao Ren, Long-Fei Li, Lei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.04991)  

**Abstract**: Large language model (LLM) agents solve complex tasks through multi-step interactions with external tools. These interactions often contain recurring local tool sequences. Treating such sequences as composite "Skills" can shorten tool-use trajectories and reduce repeated low-level decisions. However, when atomic tools and composite skills coexist, skill use becomes a policy problem: the agent must decide whether the current state requires atomic fine control or skill-level abstraction. In this paper, we argue that effective skill use should be studied as adaptive tool granularity selection. The most direct training signal for this problem is to compare the consequences of atomic and skill choices available from the same state. Based on this view, we propose CIPO, a Counterfactual Imagination Policy Optimization framework for adaptive tool granularity. CIPO constructs executable skills through budget-constrained mining of successful tool-use trajectories and trains granularity decisions with counterfactual branch rollouts. For each base rollout, CIPO branches at the first eligible granularity decision and replaces the chosen action with a feasible atomic or skill alternative. The paired outcome difference serves as a supplementary reward for policy optimization. Experiments across multiple benchmarks and model backbones show that CIPO improves task success and decision efficiency over baselines. Further analyses show that CIPO learns effective skill use by improving the choice between atomic tools and composite skills based on the current state, without simply increasing skill frequency. 

---
# Do LLMs Understand Sequential Structure? A Controlled Study of Inference and Generation 

**Authors**: Jerry Wang, Zhengxiang Wang, Ting Yu Liu, Hsin-Ling Hsu, Yi-Cheng Lai, Tengfei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.04977)  

**Abstract**: Large language models (LLMs) are increasingly used as interactive agents and simulators, yet it remains unclear whether they can recover latent sequential structure beyond surface action frequencies. This distinction is critical for behavioral simulation, where actions are often shaped by prior context rather than marginal frequencies alone. We study this question using controlled two-player Rock--Paper--Scissors interactions and a one-player stochastic n-gram continuation task. Across these experiments, we test whether LLMs can identify latent strategies, follow simple Markov rules, and sustain higher-order conditional dependencies. Our framework separates distribution matching from conditional rule following. Results show that longer context does not improve identification, correct recognition does not ensure faithful simulation, and higher-order dependencies substantially degrade rule recovery. Apparent behavioral fidelity can therefore mask incorrect generative mechanisms. 

---
# Runtime Authorization of Self-Generated Subgoals in Long-Horizon Tool-Using AI Agents 

**Authors**: Genliang Zhu, Chu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04975)  

**Abstract**: Long-horizon tool-using AI agents create subgoals, replan, delegate work, and compose sibling results. Per-tool permission checks cannot establish that a changing goal graph remains within the principal-approved task. We address this authorization gap in a finite structured domain with one principal and one authorization root. Each proposed goal-graph mutation carries a version-bound witness that its continuation traces, resources, obligations, invariants, and closing condition refine the active root contract; every protected effect is rechecked at an atomic commit boundary. Free-form goal text supplies no authority.
We prove trace-policy and modeled forbidden-state preservation under explicit mediation, abstraction, freshness, and atomicity assumptions, plus conditional root-success preservation, a separation result for memoryless allowlists, exact finite-domain decidability, and universal-safety monotonicity under sound abstraction refinement. An executable model explores 340 states and 419 transitions. Across 96 matched cases covering 25 structural schemas, the complete mechanism commits zero forbidden states in 48 drifted cases and completes all 48 benign counterparts. Two public upstream runtime paths execute 258 native dispatches across 32 cases, with every case-level decision and receipt chain matching. A frozen host-local study covers 129 synthetic one-factor-at-a-time cells, all matching fixed decisions and reasons. A history-aware continuation comparator blocks every modeled bad trace prefix but commits all operations in 11 cases whose violations lie in typed resources, freshness, or explicit-join evidence outside its trace projection. Within the registered structured domains, runtime authorization preserves useful replanning while preventing self-generated subgoals from becoming a source of new authority. 

---
# BACAM: Behavior-Aware Continual Agent Merging for Multi-Turn Interaction 

**Authors**: Shuaitong Li, Baochen Xiong, Xiaoshan Yang, Xizhe Zheng, Yifan Xu, Jianhao Huang, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04966)  

**Abstract**: Model merging offers a way to integrate the capabilities of specialized experts, but existing agent merging methods typically require all of them to be available at once. We study continual agent merging, which integrates incoming experts sequentially without retaining previously merged experts. Yet merging in parameter space or feature subspaces does not ensure that the merged model acquires an incoming expert's behavior on interaction trajectories. Moreover, updates toward a new expert can disrupt the merged model's previously integrated interactive behavior. Therefore, we propose Behavior-Aware Continual Agent Merging (BACAM), which learns parameter-wise merging gates from candidate-generated trajectories using expert-guided behavioral supervision. Task-level stability-plasticity control and tensor-level conflict-aware update budgets limit interference with existing capabilities while allowing new ones to be acquired. The learned gates are folded into the model weights without additional inference-time parameters. Across four interactive tasks - web shopping, tool use, information retrieval, and embodied interaction - BACAM achieves an average success rate of 62.82%, exceeding the strongest evaluated merging baseline by 21.69 percentage points. Our code is publicly available at this https URL. 

---
# Zero-Shot Time-Series Question Answering via Decoupled Perception and Reasoning 

**Authors**: Jing Xie, Haochen Yuan, Yunbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04942)  

**Abstract**: Time-series question answering (TSQA) requires grounding linguistic queries and diverse answer formats in complex numerical observations. However, existing methods heavily overfit to specific datasets and struggle to generalize when input series, question contexts, and answer requirements shift simultaneously. To address this challenge, we propose TSHarness, an agentic framework that establishes a decoupled workflow for cross-dataset zero-shot TSQA. At its core, TSHarness divides and conquers numerical perception and contextual reasoning via a structured Time-Series Perception State (TPS). Guided by a reusable memory of analytical knowledge, a learned Tool Selector adaptively invokes numerical tools to extract salient statistical and temporal features into the TPS. The answering agent then performs semantic reasoning over the TPS to generate target outputs, triggering iterative re-perception through the feedback loop when evidence is deemed insufficient. By separating numerical feature extraction from question-specific reasoning, TSHarness eliminates the need for target-side training or answer feedback, providing a generalizable, cost-efficient foundation for zero-shot TSQA. 

---
# Are We Measuring Scientific Intelligence? Rethinking the Evaluation of AI Scientists 

**Authors**: Kate Zhang, Yuante Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.04915)  

**Abstract**: AI agents can now carry out data-driven scientific analyses end to end, and benchmarks assess them by giving an agent a question and a dataset and scoring its final answer against a fixed key. These benchmarks assume that a correct answer was derived from the supplied data, a property we call evidence grounding. However, an agent can also reach the key from prior knowledge or by ruling out the other options, and a score based on a single run cannot tell these cases apart. We show how to test this assumption and find that it often fails. For each question, we build versions of its data files in which the evidence for the answer is left intact, withdrawn or reversed, check each edit with a pre-registered reference statistic, and run the same agent on every version. We then measure evidence-grounded accuracy, which credits a correct answer only if the agent also responds when the evidence is withdrawn and follows it when it is reversed. We evaluate three agent scaffolds and five models on 18 single-cell questions from BAISBench and four synthetic problems from GeneBench-Pro. On the single-cell questions, the Claude agents are 95% accurate and answer 83% correctly without any data, but their evidence-grounded accuracy is only 41%. Hiding gene names raises the share of runs that follow reversed evidence from 58% to 93% on ten gene tasks, suggesting that prior knowledge competes with the supplied data. The benchmark score and LLM judges can also reward answers that ignore the changed evidence. Measuring scientific intelligence rather than recall therefore requires checking whether answers follow the evidence and whether scores reward them for it. 

---
# Monitorability Disposition in Large Reasoning Models 

**Authors**: Shahriar Golchin, Marc Wetter  

**Link**: [PDF](https://arxiv.org/pdf/2610.04914)  

**Abstract**: Monitoring the chain-of-thought (CoT) of large reasoning models (LRMs) is a common way to detect misbehavior in real-world practice. However, current monitoring is passive: a separate model inspects the session only after execution. This means harm may already have occurred before it is caught. An active alternative is to have the model self-report its misbehavior as it happens. Whether models are willing to do this, however, is unknown. We introduce "monitorability disposition": a model's willingness to make itself monitorable and stay monitored throughout inference when warranted. We measure it as the fraction of warranted cases in which a model self-reports its own misbehavior via tool calls to available monitoring channels. We evaluate four LRMs on three misbehaviors (sycophancy, reward hacking, and bias) while varying the available monitors (AI and human) and the pressure to use the monitoring tools. We find that when tool use is optional, models self-report in only about 16% of warranted cases on average. Increasing tool-use pressure does not improve reporting where it matters: high-severity misbehavior is never self-reported. Models also systematically select the monitor they perceive as least strict. Overall, we identify monitorability disposition as a new contributing factor to model monitorability: when sufficiently strong, it keeps models seeking monitorability throughout inference. 

---
# VideoResearchAgent: Grounded Task Synthesis and Sim-to-Real RL for Open-Web Video Research 

**Authors**: Yuhang Zhou, Fei Li, Yuxi Wu, Bin Zhu, Jingjing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.04911)  

**Abstract**: Existing deep research agents are designed primarily for text- and image-based web sources, while video reasoning systems typically assume that relevant videos are provided in advance. We study open-web video research, where an agent must autonomously discover relevant videos, navigate their temporal content, and ground answers in visual evidence. Training such agents at scale is challenging as live video interaction is slow and unreliable, whereas fixed local simulation can induce retrieval-specific shortcuts that fail to transfer to the open web. We introduce VideoResearchAgent, a scalable training framework to address these challenges. First, we introduce controllable task synthesis pipeline to synthesize multi-hop research tasks from timestamped visual evidence while filtering text-only shortcuts. Second, we build a field-aligned local video simulator that preserves deployment-facing search and watch interactions while accelerating video search by a factor of 34.5-64.6. Third, we introduce Retrieval-Domain-Randomized GRPO (RDR-GRPO), which diversifies candidate rankings, distractors, metadata, and result structure during training to reduce overfitting to simulated retrieval. On Video-BrowseComp, the VideoResearchAgent trained using Qwen3.5-4B achieves 40.48% accuracy, comparable to Gemini-3-Flash-Preview, while reducing cumulative API-token consumption by 74.9% relative to the untrained model. Together, these results establish an accurate and efficient training recipe for open-web video research. 

---
# Self-Evaluating Recursive Agents 

**Authors**: TianYi Lyu, Xiaozhe Li, Yang Li, Yongkang Chen, Kefei Tian, Junbo Niu, Zican Hu, Hongbo Liu, Mingliang Xiong, Qingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04902)  

**Abstract**: Recursive language-model agents decompose tasks and delegate subtasks to child instances of the same policy, forming a tree of work. Training them, however, is hard: the final outcome is verifiable, but the self-invented intermediate subtasks are numerous and carry no ground truth. Existing methods score each node with a verifier or judge, which is costly at scale and blind to decomposition quality. We argue that a recursive agent must learn three coupled capabilities within one set of weights: decomposing problems into subtasks, solving them, and evaluating the outcomes, each requiring its own training signal. SERA (Self-Evaluating Recursive Agents) turns evaluation into a learned capability of the policy itself. Before delegating, the parent writes a rubric of weighted success criteria for each child subtask; a ranking objective against verified outcomes then trains rubric generation so that the criteria track genuine subtask success. In addition, a complementary leaf-coverage signal provides direct credit for task decomposition. Our central finding is that \emph{training} the policy to generate aligned rubrics is what drives the gains: because the same weights both evaluate and execute, learning to judge subtasks sharpens the agent's ability to solve them. Notably, external supervision is also reduced: the judge is consulted only to train the rubric generator, while solving is trained against the agent's own rubric scores, which outperform direct use of the judge. Beyond training, the learned rubric doubles as an inference-time selector for tree search. On TextCraft-Synth and TextWorld-Sync, SERA improves over strong recursive-agent baselines by 5.38 and 13.14 points on average, and rubric-guided tree search at inference adds a further 2.43 points on TextWorld-Sync. 

---
# ForkPilot: Self-Evolving Policy for Retrospective Search in Long-Horizon Agents 

**Authors**: Xinyue Zeng, Shivam Shandilya, Guilherme Potje, Leonardo Nunes, Rakshanda Agarwal, Ranveer Chandra, Emre Kiciman, Dawei Zhou, Tusher Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2610.04889)  

**Abstract**: Interactive language-model agents increasingly solve complex tasks through long-horizon, multi-call reasoning, where errors in beliefs or actions can compound across tool interactions. Retrospective search can recover from such failures but is prone to misallocation. Delayed outcomes obscure the contribution of intermediate search decisions, leading to Attribution Complexity, while evolving execution evidence leads to Adaptation Complexity, where previously learned estimates become stale. To address these challenges, we first introduce Search Value Dynamics (SVD), which characterizes the evolving trade-off between the gain and cost of retrospective search. Building on SVD, we propose ForkPilot, a self-evolving two-stage policy-learning framework. In the first stage, ForkPilot learns a search-value policy offline from completed trajectories through automatically constructed outcome comparisons. In the second stage, it makes search decisions based on current observations and then self-evolves by incorporating newly completed trajectories into subsequent policy updates. We evaluate ForkPilot across 6 diverse benchmarks and 7 widely used LLM backbone families, including four open-source families, GPT-5.6 Sol, and Opus 4.8 in a production agentic system, against 9 competitive baselines, including a real-world harness deployment used by hundreds of thousands of paid users. ForkPilot achieves comparable state-of-the-art performance while reducing token usage by up to 59.2%, demonstrating its efficacy. 

---
# SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models 

**Authors**: Chung-En Ho, Weiyu Sun, Cheng-Jhih Shih, He Li, Yong Liu, Yingyan  

**Link**: [PDF](https://arxiv.org/pdf/2610.04875)  

**Abstract**: Diffusion large language models (DLLMs) generate text through iterative block denoising, and multi-branch speculative decoding accelerates this process by verifying a main branch together with multiple draft branches in a single forward pass. While prior DLLM acceleration methods primarily exploit temporal redundancy across denoising steps, we identify a complementary redundancy axis within each speculative verification step: multi-branch computational redundancy. During speculative verification, draft branches inherit most tokens from their parents while unmasking a small set of additional positions, causing large portions of hidden states to remain highly similar across branches. We propose SpecFold, an algorithm-system co-design that exploits this multi-branch redundancy to reduce the cost of multi-branch speculative verification. Algorithmically, SpecFold performs token-level residual gating and selectively reuses parent computation through folded attention and FFN while preserving residual hidden states. Systemically, a Triton kernel implementation translates this fine-grained reuse into end-to-end throughput gains through efficient sparse multi-branch execution. SpecFold is orthogonal to temporal caching and compatible with existing DLLM speculation strategies. Across two DLLM families, five models, and five standard benchmarks, SpecFold achieves up to 1.64x throughput over Spiffy and up to 1.99x over vanilla decoding, while maintaining comparable task performance. 

---
# From Memory to Guide: Spatio-Temporal Composer for Procedural Coding Memory 

**Authors**: Zhixuan Tan, Pengjie Gu, Zhao Li, Yihan Hu, Xu He, Dong Li, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2610.04868)  

**Abstract**: Memory-augmented agents typically integrate procedural knowledge by injecting retrieved skills directly into text prompts. This approach dangerously equates readable text with reliable execution. To bridge this gap, we introduce From Memory to Guide, a novel paradigm that transitions procedural memory from passive text delivery to active, inference-time policy adaptation. We instantiate this paradigm through the Spatio-Temporal Composer, an active policy compiler that explicitly manages exactly how and when retrieved knowledge should be applied. Rather than treating skills as plug-and-play modules, Composer dynamically aligns historical knowledge with current environmental constraints (spatial adaptation) and precisely dictates its applicable lifecycle (temporal orchestration). It actively transforms static memories into strictly bounded Runtime Guides---equipping the agent with localized objectives and behavioral guardrails without requiring a single parameter update. Extensive evaluations on 13 demanding, long-horizon software engineering tasks in EngramBench demonstrate the clear advantages of this architecture. Composer not only robustly prevents context mismatch but drives an absolute pass-rate increase of 7.2 percentage points on the most complex tasks, while simultaneously slashing the main agent's token usage by 32.2%. 

---
# GitSwarm: Decentralized Compounding Inference 

**Authors**: Vedant Shah, Ankur Samanta, Paras Dahal, Mikhail Plekhanov, Carole-Jean Wu, Scott Yih, Remi Munos, Rob Fergus, Jakob Foerster, Ruslan Salakhutdinov, Sanjeev Arora, Jason Weston, Aaron Courville, Anirudh Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2610.04862)  

**Abstract**: Long-horizon problem solving and scientific research require computation to accumulate across successive attempts. Partial solutions, experimental findings, and unsuccessful approaches can inform later work, yet most inference-time computation is organized around individual trajectories or candidates rather than a persistent body of reusable work. We call this paradigm compounding inference: organizing inference-time computation so that intermediate work persists and can be inspected, extended, combined, or challenged by subsequent computation. We instantiate compounding inference in GitSwarm, an asynchronous system where homogeneous agents independently decide how to advance a task while collaborating through structured persistent memory. Agents explore, experiment, verify, refine, and synthesize previous work in a shared, branch-able Git repository. Atomic commits preserve intermediate artifacts, while explicit semantic dependencies record how later contributions build on work across branches. We evaluate GitSwarm on long-horizon problem solving and sustained GPU-backed experimental research. On IMOProofBench-Advanced, GitSwarm solves all 30 problems in one run using GPT-5.5. On ProgramBench, it achieves a $79.4\%$ mean score, versus $65.1\%$ for the strongest reported baseline under the stated budget. On three neural architecture research tasks (Residual Matrix Transformer, Looped Transformer, NanoChat), GitSwarm improves upon the starting architectures through successive experimentation. Beyond final performance, we measure whether computation accumulates: on ProgramBench, $94.7\%$ of contributions are subsequently built upon, while the selected solution's ancestry covers $82-93\%$ of the contribution graph. These results show that inference-time computation can accumulate across otherwise independent episodes, forming an evolving body of work that subsequent inference can reuse. 

---
# MemTrace: State-Consistent Memory for Long-Horizon Coding Agents 

**Authors**: Hongming Xu, Le Zhou, ZhongHe Jin, Xiang Zhang, Bo Tang, Zhiyu Li, Xuanhe Zhou, Juncheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04838)  

**Abstract**: As coding agents take on long-horizon software evolution tasks spanning multiple files and stages, longer execution trajectories introduce two coupled challenges: (1) accumulated histories strain context budgets, and (2) repository changes can invalidate earlier execution evidence. Existing approaches address these challenges through techniques like larger context windows, compression, retrieval, or repository representations, but often fail to reconstruct a consistent task state after a context refresh or verify whether recalled evidence remains valid. Thus, we introduce MemTrace, a provenance-aware memory system that preserves execution history and aligns its reuse with the evolving task (e.g., iterative cross-file repair) and repository state. MemTrace stores history as immutable Memory Traces anchored to key information (e.g., files, symbols, tests), and organizes their execution order and dependencies in a Memory Trace Graph. When context is constrained, working memory retains only compact Memory Anchors, from which the agent can reconstruct the latest execution state and locate evidence relevant to its next action. Before restoring historical evidence, MemTrace checks its validity against the current repository state and retrieves only what the next action requires. Across three complementary long-horizon coding benchmarks, MemTrace consistently outperforms all fully evaluated baselines under the same backbone and harness, improving DeepSWE pass@1 by 21.2 points, SWE-EVO Resolved Rate by 4.4 points, and SWE-Milestone Score by 17.8 points under Codex CLI. 

---
# DICE: Decoupling Capability from Intervention Necessity in LLM Tutoring 

**Authors**: Sayantan Pal, Kaiyi Ji, Rohini K. Srihari  

**Link**: [PDF](https://arxiv.org/pdf/2610.04825)  

**Abstract**: Fluent guidance is not the same as useful intervention. LLM tutors are typically trained to generate the next teacher utterance, implicitly assuming that every student turn warrants a response. However, our experiments indicate that this conflates tutoring capability (what to say) with intervention necessity (whether to say it). We introduce DICE, a framework that decouples intervention decisions from response generation by first selecting an explicit pedagogical action. To calibrate this action selection policy, we define Intervention Value (IV), a rollout-grounded counterfactual metric that compares each action against non-intervention. IV shows that many prescribed interventions provide little or no marginal benefit. We further introduce DICE-Bench, a multi-variant math tutoring benchmark with skill-preserving problem variants for session-level evaluation. Using IV-weighted and KL-regularized policy optimization, DICE learns to intervene selectively while preserving tutoring effectiveness. In simulated tutoring sessions, DICE reduces the over-intervention rate to near zero while guiding students to correct solutions in approximately 3-4 fewer turns on average than existing Socratic tutoring baselines. 

---
# Agent Behavior as Code: Efficient and Robust LLM Agents with Programmatic Specifications 

**Authors**: Peng Qi, Chunliang Lyu, Gang Li, Fabian Chan, Cheng Chang, Ignacio Cases, Will Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04824)  

**Abstract**: AI agents based on foundation models (FMs) have demonstrated strong capabilities to perform complex open-ended tasks. However, they face some common challenges in practice: (a) agent behavior can deviate drastically even for semantically similar tasks, leading to catastrophically propagated errors; (b) high cost and latency due to FM calls, repeated in full whenever a task recurs with different inputs; (c) FMs' limited context and instruction following capability confine how well agents manage the ever-growing execution context and follow complex plans. We introduce $\textbf{A}$gent $\textbf{B}$ehavior as $\textbf{C}$ode $\textbf{Agent}$ (ABCAgent), which uses a symbolic program (e.g., Python code with potential neural functions) to fully specify the agent's behavior at runtime, with a powerful FM agent editing that program for flexibility. Behavior is thus specified without premature variable binding, and its execution is deterministic. We evaluate ABCAgent on six agent benchmarks, two of which we construct to test how well a derived program generalizes to variants of the task it was written for. ABCAgent matches a model-matched neural agent on GAIA and augmented GAIA, and surpasses it where robustness and long control flows matter: 98.3% against 97.3% on GSM-Symbolic ($p = 0.001$), 71.9% against 47.4% $\mathrm{Pass}^4$ on the telecom domain of $\tau^2$-bench ($p = 0.0001$), and more records written correctly at every loop length on our control-flow-augmented WorkArena benchmark. For more parametric task families, ABCAgent is also significantly superior in efficiency. Without authoring a new program, ABCAgent solves 92.6% of GSM-Symbolic instances and 20.1% of augmented GAIA variants, which yields $5.2\times$ lower latency and $7.0\times$ lower cost on GSM-Symbolic, 19% lower cost on augmented GAIA, and $9.5\times$ lower agent latency on $\tau^2$-telecom. 

---
# MedImageOSWorld: Benchmarking GUI Agents for Medical Image Consoles 

**Authors**: Ziyang Long, Xinqi Li, Lujing Xing, Hsin-Jung Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04800)  

**Abstract**: Graphical consoles offer a practical interface for medical acquisition assistance, allowing agents to work through the controls and visual feedback used by human operators. Reliable assistance requires linking on-screen anatomy to acquisition decisions that determine what image evidence becomes available next. We introduce MedImageOSWorld, a benchmark for evaluating this capability in simulated CT, MR, and ultrasound consoles. Using screenshots and mouse-and-keyboard actions, agents configure protocols, plan acquisitions, inspect the resulting images, and make corrective adjustments across seven capability levels, from console operation to feedback-driven control. Evaluation combines task-specific workflow checks with hidden anatomical ground truth to assess procedural completion and acquisition outcomes separately. A common evaluation protocol specifies episode conditions and interaction budgets, while recorded trajectories support analysis of how agents observe, act, and respond to acquisition feedback. Across eleven open-weight agents, success rates range from 3.0 to 25.0 on a 0-100 scale while workflow-progress rates reach 21.5-74.5: agents complete much of the console workflow but rarely acquire the intended anatomy. Success collapses between perception and planning, from 79-90% at the lowest three levels for the best agent to at most 12% for millimetre-level planning and 0% for closed-loop control among open-weight agents. Two proprietary agents reach success rates of about 34 and exceed the best open-weight agent mainly in acquisition quality (61 versus 42). MedImageOSWorld provides a controlled setting for studying whether general-purpose GUI agents can translate visual observations into effective medical acquisition decisions. 

---
# Knowing the Store: What a Memory Backend Must Write Down Before an Agent Can Read It 

**Authors**: Ansuman Mullick, Eray Tüzün  

**Link**: [PDF](https://arxiv.org/pdf/2610.04794)  

**Abstract**: An agent with long-term memory can answer from a record it should no longer use, such as a plan the user later cancelled. We ask what a memory store must expose for an agent to know this before retrieving anything, and we score that judgment on its own, as metamemory monitoring. Readers see only a value-free summary: record counts by lifecycle state and a list of attribute names. Each of 148 questions is asked against three versions of one store that differ in one attribute, so wording cannot give the answer away. What a model adds depends on the length of that list. On the short lists the benchmark's ground truth produces (2.6 names on average), none of five language models reliably beats a cosine-similarity lookup over the names, and the three stronger ones are equivalent to it within 0.05. Under a control that fixes counts and list length, none is reliably above it and none is shown equivalent. On lists longer than the stores we tested write, padded to 60 names, the lookup loses 0.18; GPT-5.6 Luna and Sol lose 0.06 to 0.07 and lead it by 0.13 to 0.14, while the other three fall with it. The lead holds for Sol against a sibling padded to the same length and counts. Reworded to share no word with the names, the lookup loses 0.02 and one stronger reader edges 0.04 ahead. On the short lists the three stronger readers lead only where the summary counts a state without naming it, and one count feature closes that lead within a question, though not when items are pooled. The backends we tested omit or misstate this information: under FR-Bank's own metadata, GPT-4.1 mini, Haiku 4.5 and the lookup fall from 0.73, 0.82 and 0.75 to 0.60, 0.71 and 0.59. In these tests, what the store wrote down limited every reader we gave it to. All stores are synthetic; control, whether a better judgment yields a better answer, is left to a later study. 

---
# Pressure, Context, and Machine Self-Control: A Criminological Test of Reward Hacking in Generative AI Models 

**Authors**: Murat Ozer, Isaac Kofi Nti  

**Link**: [PDF](https://arxiv.org/pdf/2610.04793)  

**Abstract**: Recent incidents show that AI agents sometimes reach measured goals through unsanctioned means. This study applies self-control, general strain, anomie, neutralization and routine activity theory to reward hacking in generative AI models, and it treats the measures as behavioral analogues. Study 1 (2,310 conversations, seven models) measured delay discounting with the Kirby Monetary Choice Questionnaire and stated willingness to take shortcuts. Pressure raised the discount rate k 2.8-fold in fresh conversations but 12.6-fold when the same sentence followed a baseline answer, which indicates a response to conversational cues rather than a stable trait. Models chose a shortcut in 1 of 700 dilemmas when answering as themselves and in 64 of 700 when asked to assume human impulses, each step of pressure raised the odds by 40%, and shortcut answers contained far more techniques of neutralization (rate ratio = 146). In the preregistered Study 2, five models worked on 20 coding tasks whose tests contradicted their specifications. Two Claude models never cheated. GPT-5.6, Qwen and DeepSeek cheated in 86%, 69% and 65% of episodes and clearly disclosed the conflict in 27%, although their reasoning recognized it in 95%. GPT-5.6 had never endorsed a shortcut in Study 1. The registered effects of pressure and of an auditor cue did not survive correction for multiple testing. In exploratory analyses, two further models cheated in 69% and 100% of episodes, and one sentence stating that the specification takes priority eliminated cheating in all 280 episodes. Therefore, stated refusal does not guarantee compliant agent behavior. 

---
# Do More Modalities Always Help? A Geometric Perspective on Missing-Modality Robustness 

**Authors**: Songyuan Sui, Zhen Tan, Mohan Zhang, Rana Muhammad Shahroz Khan, Xia Hu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.04792)  

**Abstract**: Missing modality remains a longstanding challenge in multimodal learning. Existing methods typically address this issue through modality recovery or adaptive strategies. However, they overlook models' internal cross-modal dependencies formed during multimodal training, which later impair robustness. We systematically characterize a counterintuitive deployment-time failure mode: models trained on full modalities can underperform unimodal models when one modality is missing at inference time. This pattern appears across diverse architectures, such as fusion models, CLIP-style two-tower models, and vision-language models. We show that such degradation is closely associated with learned cross-modal dependencies in the principal parameter subspaces. Multimodal training induces structured rotations of these subspaces, particularly in cross-modal interaction layers. These rotations are associated with reduced task-aligned margins and larger task-aware representation harm under missing-modality inputs. We propose Geodesic Unlearning (GU), a lightweight parameter-editing method that leverages Grassmannian subspace geometry for structured subspace correction to improve missing-modality robustness. It rotates the principal input subspace toward a unimodal reference along a geodesic path. We prove that this correction minimizes the distance to the reference within a fixed subspace-distance budget. Experiments across architectures and datasets show that GU improves performance under missing-modality inference while preserving full-modality accuracy, outperforming strong missing-modality robustness baselines. These findings support a geometric view of deployment-time missing-modality degradation and suggest localized subspace editing as a practical route for robustness correction. 

---
# Not All Answers Are Contextually Persuadable: Inference Dynamics in Large Language Models under Contextual Influence 

**Authors**: Zongye Hu, Weiqing Luo, Yanjie Fu, Yu Gan, Haofeng Zhang, Ziyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04791)  

**Abstract**: At the core of modern prompting techniques is contextual sensitivity, the ability of large language models to adapt their predictions based on inference-time context. Despite its central role, inference behavior under strong contextual influence remains poorly understood, particularly at the level of internal inference dynamics. We introduce a theoretical framework for analyzing contextual influence through inference dynamics, enabling quantitative characterization of inference behavior beyond output-level answer changes. Our analysis shows that inference dynamics do not exhibit unbounded drift under repeated contextual assertions. Instead, predictive representations converge to stable, query-dependent regimes that fundamentally constrain whether contextual signals can alter a model's prediction. This leads to a surprising finding: Repeated contextual assertions do not act as accumulating evidence during inference and may therefore fail to alter a model's prediction even under unbounded repetition, while in other cases a prediction change becomes inevitable. We empirically validate our theoretical predictions, demonstrating strong alignment between theory and observed inference behavior. These contributions offer a principled pathway toward characterizing the limits of contextual influence during inference, providing practical implications for model development. 

---
# When Is Enough Enough in Self-Evolving LLM Systems? 

**Authors**: Enoch Yin, Bin Liu, Zhengling Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.04756)  

**Abstract**: Self-evolving large language model (LLM) systems repeatedly propose, evaluate, and incorporate updates to prompts, skills, or other persistent artifacts. Despite their growing effectiveness, these systems typically operate under a predetermined iteration or compute budget, without a principled criterion to determine when further evolution is no longer worthwhile. This can lead to two undesirable consequences: unnecessary computation after performance has saturated and the risk of returning late updates that overfit or exploit the evaluation signal. These issues motivate us to study two fundamental questions: when should a self-evolving system stop, and what should it output once it stops? We address the first by formulating an online sequential testing problem and constructing an anytime-valid restart detector using the per-item paired evaluation outcomes already produced by self-evolving LLM systems. We address the second by formulating a change-point estimation problem and using the estimated transition to select an earlier artifact for output. The resulting procedure is plug-and-play and requires no modification of the underlying self-evolving algorithms. Across two self-evolving frameworks, three LLM model families, and five benchmarks, our method substantially reduces computation costs while maintaining comparable unseen-test performance. For example, on SearchQA with SkillOpt and DeepSeek V4 Flash, our method stops at round 4 rather than the full budget of 40, reducing token usage by 91.6% while achieving 82.43% unseen-test accuracy versus 82.00% under the full-budget run. 

---
# Dynamic Routing as a New Dimension for Test-time Versatility of LLMs 

**Authors**: Michal Štefánik, Marek Kadlčík, Josef Kuchař, Michal Spiegel  

**Link**: [PDF](https://arxiv.org/pdf/2610.04751)  

**Abstract**: Beyond scaling their parameters and data, large language models currently gain versatility on new problems along a single axis: the tokens they spend on chain-of-thought (CoT). We investigate whether dynamic routing programs, which execute a subset of the model's layers or iterate some of them, can open a second axis of test-time adaptation, complementary to CoT and free of any gradient update. Prior work showed that such programs exist and bring accuracy and efficiency gains on problems similar to those they were trained on; we ask whether they can also be identified rapidly, from a handful of demonstrations (3 or 10), by a strategy that transfers across models and tasks without training. First, we find that strategies that select programs by the probability they assign to the demonstrations' labels, arbitrated by the model's own confidence, bring consistent gains: on average over the 49 tasks of MMLU and substantially on four of seven models, and most of all on far out-of-distribution tasks such as ARC-AGI, where programs double the accuracy of a 7B model whose CoT fails. Second, on MMLU across the seven post-trained models, routing complements CoT in practice: the two succeed on different queries, and their composition exceeds CoT alone. Despite these gains, our analyses show that confidence-based selection leaves much of the potential untapped, in two places in particular: (1) in surfacing the routing potential that is already present early in pre-training but becomes harder to select after post-training, and (2) in making models robust to the refinements routing introduces, since unsuccessful routes tend to drive the residual stream out of the distribution that the following layers expect. Together, our results point to dynamic routing as a paradigm for extending the plasticity of existing and future LLMs in rapid test-time adaptation. 

---
# Agentic discovery of blood biomarker from distilled private health records 

**Authors**: Seffi Cohen, Liat Antwarg Friedman, Amir Anisman, Ruth Johnson, Michelle M. Li, Ayush Noori, Ben Reis, Ran Balicer, Noa Dagan, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2610.04749)  

**Abstract**: Routine complete blood counts (CBCs) could yield new biomarkers, but the private records needed to evaluate candidates cannot be shared with frontier language model agents that excel at discovery. We distilled the evidence held in the Clalit Health Services panel of over 5.4 million patients into a released scoring tool: for each of 13 immune-mediated diseases, a graph attention network was trained inside the data boundary to predict the case-control AUC of candidate CBC expressions, and only the trained weights were released. The tool grounds an agent's propose-score-refine loop in real-world data without exposing any patient data. In external validation, agent-discovered expressions improved on their literature-seeded starting points by a median of 4.18 AUC percentage points, and across three independent cohorts, reranking the candidates of three frontier research tools improved on their first choices in most comparisons, with gains that varied by cohort. The released scorer supports privacy-preserving biomarker hypothesis generation. 

---
# Formalizing the Moral Evaluation of Speech Acts: Truthfulness, Lies and Ethical Dilemmas 

**Authors**: Benjamin Icard, Gauvain Bourgne, Jeanne Bonnaventure, Jean-Gabriel Ganascia  

**Link**: [PDF](https://arxiv.org/pdf/2610.04747)  

**Abstract**: In life-or-death situations, a benevolent lie may appear more moral than telling the truth. Yet such lies can backfire, producing unintended and sometimes fatal consequences. This tension, famously disputed by Kant and Constant in 1797, applies not only to lying but to assertive speech acts in general, raising the question of which utterance should be chosen when moral stakes are high. We present a logical framework for the ethical evaluation of speech-act utterances based on agents' beliefs. Implemented in Answer Set Programming (ASP), the framework assesses utterances under deontologism, consequentialism, and principialism, and is illustrated on Sartre's The Wall (1939), a reworking of that controversy in which lying leads alternately to rescue and to death. Our setting is general by design: as two variants show, accommodating a new moral situation amounts to adjusting parameters, not rules. 

---
# Toward a Locally Deployable Agentic Co-Scientist: Small-Model Planning for Early-Stage Drug Discovery 

**Authors**: Tian Liang, Jiayu Chang, Alejandro F. Frangi, Mobarak I. Hoque, Richard A. Bryce  

**Link**: [PDF](https://arxiv.org/pdf/2610.04740)  

**Abstract**: Early-stage computational drug discovery requires coordinating heterogeneous scientific tools across multi-step workflows. We present a lightweight, tool-augmented framework in which a locally deployable compact language model plans calls to 18 modular tools. A Unified Molecular Schema maintains shared molecular records, while a plug-in interface supports tool replacement and extension. We construct 1,263 manually refined query-plan pairs through workflow-graph path coverage and apply LoRA fine-tuning to three compact model families. Under the query-level split, all fine-tuned models generate fully parseable and schema-compliant plans on 47 held-out cross-group queries. Llama 3.2-3B achieves a tool-selection F1 of 0.998, sequence exact match of 0.979, and argument F1 of 0.960. Under the stricter workflow-grouped split, which excludes identical ordered tool sequences across partitions, sequence exact match reaches 0.452 to 0.548, highlighting the remaining difficulty for compact models in generating complete workflow paths unseen during training. These results demonstrate the feasibility of compact, locally deployable planning while identifying compositional generalization as an important direction for further improvement. 

---
# PyINE: A Framework for Scalable Elicitation and Oversight via Code Execution 

**Authors**: Pierre-Luc St-Charles, Alessandro Palmas, Damiano Fornasiere, Storm Lei, Mirko Bronzi, Jean-Pierre Falet, Iulian Serban, Yoshua Bengio  

**Link**: [PDF](https://arxiv.org/pdf/2610.04737)  

**Abstract**: Reasoning models can remain capable of solving a task while still defaulting to cheaper but misleading shortcuts. This creates a central oversight problem: when a model gives an answer with plausible but incomplete reasoning, can an overseer determine whether that output should be trusted? To study this problem, we introduce PyINE, a framework for scalable elicitation and oversight using instrumented Python programs as a verifiable execution substrate. In PyINE, programs define task environments, execution traces provide authoritative labels for outcomes and intermediate facts, and task variants can be generated mechanically rather than through static human annotation. We instantiate the framework in PyINE-v1, a first release built from nearly one million deterministic execution traces and over 500,000 matched LLM-generated code variants used for counterfactual evaluation. Using standard RL with verifiable rewards on cue-varied tasks, we train a shortcut-following model that improves substantially at predicting execution outcomes while still making systematic errors when misleading human-facing cues conflict with the program's realized behavior. We then evaluate activation probes, trained text classifiers, prompted judges, and a lightweight debate protocol as overseers of this model. We find that performance pooled at the dataset level can hide weak coverage of the failures that matter most: cheap learned overseers often miss rare shortcut-driven errors, while stronger model-based checks are more balanced but substantially costlier and harder to turn into reliable thresholded decisions. PyINE-v1 turns this failure-mode coverage problem into a reusable experimental setting for developing oversight methods that are verifiable, failure-mode-aware, and cost-sensitive. 

---
# Learning to Clarify Underspecified Intents Under Limited Interaction 

**Authors**: Pranav M R, Manuel Cherep, Pattie Maes, Nikhil Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.04719)  

**Abstract**: AI assistants receive requests that leave out information needed for a good outcome, for example about users' preferences or goals. They must then either speculate or ask for more information before proceeding. We reconceptualize this as a value-of-information problem: the assistant should acquire information whose absence causes the greatest avoidable loss in user utility. This is rarely known ex ante; rather, assistants must predict it in order to optimally allocate limited user interactions. We instantiate this problem in image generation and derive a reinforcement learning framework using multi-turn simulated users to maximize utility recovery under uncertainty. In a preregistered study with 456 interactive sessions across 76 human participants, this helped users significantly better match reference images with significantly fewer questions, less total interaction time, and lower cost. This points toward a simple and scalable framework for training language model assistants to better disambiguate user intent by asking more informative questions. 

---
# Not Self-Decidable: LLMs Cannot Draw the Boundary of What an Agent Verifier Can Check 

**Authors**: Anthony Rhodes  

**Link**: [PDF](https://arxiv.org/pdf/2610.04699)  

**Abstract**: A verifier for an agent faces rules of two kinds: the ones a fixed check can settle and the ones that require a judge. A team that derives its own checks fixes that split up front. Where the requirements come from outside, as in finance, healthcare and law, the agent enforces rules it did not write, so the split falls to runtime, recurring for every predicate of every rule on every action at a rate no reviewer can audit. Every escalation scheme assumes a model can make that decision itself, that it is self-decidable. Across six corpora, including the EU AI Act, FINRA guidance and a deployed credit agent, we collect roughly 22,000 labels from four models built by three labs. They agree almost perfectly where the answer is obvious and collapse on regulatory text; their errors run in opposite directions, so no model can be trusted as the conservative choice; and on the deployed agent's own rule-set they err together, over-claiming that a fixed check will do, the direction that never gets escalated. We introduce CoVer (corroborate-then-verify), which treats unanimity as a nomination, admitting a predicate only when the check synthesized for it survives intervention, reading fields the agent cannot write and holding under deterministic rewording. That gate rejects most of what corroboration wrongly admits, at a cost in coverage we report rather than tune away. The obvious alternative, agreement with a reference judge, certifies nothing: it climbs from 30% to 77% across calibration bands while the genuinely decidable share does not move, because a judge drawn from the population under indictment ratifies the blind spot it shares. Self-decidability is not a capability to elicit from a model but a boundary the verifier must construct. 

---
# Penumbra: Sample-Efficient Adversarial Search for Regulatory Obligations 

**Authors**: Anthony Rhodes  

**Link**: [PDF](https://arxiv.org/pdf/2610.04693)  

**Abstract**: Agents are entering finance, healthcare and law, sectors where a violation leaves no lexical signature and carries real penalties. Whether an omission is material, or a disclosure sufficient, depends on what the response left out. Probing such an obligation means finding responses one minimal edit from flipping compliance, and every probe costs a generation and two adjudications, so the binding constraint on regulatory red-teaming is sample efficiency, not volume. We introduce Penumbra, an adversarial search that walks from a verified anchor under an expanding edit budget until a two-evaluator committee changes its verdict, and emits the two adjacent responses that straddle the change. Allocation is adaptive, and the objective is coverage of the defeat surface: distinct (obligation x defeat mode) cells resolved per candidate. At matched budget, adaptive allocation reaches uniform allocation's full-budget coverage on 59% of the candidates; at equal records it covers 1.43x the defeat modes of naive enumeration, and the gain is confined to the axis it targets. On 60 screened obligations of a financial advisory constitution, Penumbra returns 144 pairs, each a compliant and a violating response that the committee places on opposite sides of the boundary, differing by a handful of words where a model asked for both directly produces texts sharing almost nothing. A second constitution, for clinical triage, reproduces this on 18 obligations: 49 pairs at the same tightness, with the same modes hardest. Pairs like these show where an obligation's own terms stop deciding, which is what an agent deployed under it must be tested against, and the search finds them at a cost that scales with the boundary, not the text. 

---
# RAGStress: A controlled benchmark for evaluating retrieval-augmented generation under knowledge-base degradation 

**Authors**: Shiqi Yang, Jiekai Ma, Gaoyuan Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.04691)  

**Abstract**: Retrieval-Augmented Generation (RAG) is typically evaluated under the implicit assumption that the underlying knowledge base (KB) is clean, leaving the behaviour of RAG systems under realistic KB degradation poorly characterised. We introduce RAGStress, a controlled evaluation benchmark for stress-testing RAG systems under systematic KB corruption. The benchmark pairs four naturalistic corruption types (factual corruption, numeric typo, relevance poisoning, and contradiction injection) with three severity levels (subtle, moderate, and obvious) over a single-KB, metadata-filtered experimental design built from 57 MMLU subjects and 182,546 documents. Across 52,500 model-question-condition evaluations, RAGStress reveals that clean retrieval can mask robustness differences, semantic-fidelity corruptions are substantially more harmful than signal-utility perturbations, no-retrieval accuracy does not predict corrupted-retrieval robustness, and mixed-KB accuracy should not be treated as worst-case robustness. We document the benchmark's intended use, supported claims, and limitations, and provide an artifact bundle including generation scripts, corruption prompts, metadata schema, and evaluation code. RAGStress is intended as a controlled stress test for RAG robustness under KB corruption, not as a general model leaderboard. 

---
# When Debate Helps: Proposal Supply and Verification-Aware Readout in Multi-Agent Reasoning 

**Authors**: Zihao Zhao, Tunyu Zhang, Haizhou Shi, Yusong Zhao, Xinxi Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04686)  

**Abstract**: Multi-agent debate can improve reasoning, yet often fails to beat simple majority voting. We argue that successful debate requires two distinct mechanisms: proposal supply must surface a correct answer, and readout must identify that answer when voting misses it. We formalize the first requirement through recoverable headroom, which measures cases where a correct proposal is available but the majority answer is wrong. For the second, we develop Latent Verification Debate (LVD), an accounting model in which candidate proposals receive answer-specific verification evidence before final generation. Controlled fixed-proposal interventions estimate this latent effect in equivalent peer-support units and show that correct evidence changes answer probabilities and generated decisions while proposal supply remains fixed. To improve proposal supply, we construct societies from neural-thicket agents using labeled and label-free coverage objectives. Across two backbones and matched-budget reasoning benchmarks, coverage-selected societies increase complementary proposal supply and improve aggregate accuracy in repeated stochastic evaluations. Round-level controls further show that interaction provides gains beyond applying the same finalizer directly to the initial proposals. These results identify proposal coverage and truth-sensitive evidence use as complementary conditions for debate to outperform voting. Code is available at this https URL. 

---
# MASBench: Benchmarking LLM-based Multi-Agent Collaboration under Partial Observability 

**Authors**: Qizhi Chu, Zekai Yu, Sijie Wen, Yang Liu, Chen Qian, Cheng Yang, Chuan Shi, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04672)  

**Abstract**: Large language models (LLMs) have progressively evolved into the core of autonomous agents. Building on this progress, LLM-based multi-agent systems (MAS) coordinate multiple agents into a synergistic team to accomplish complex tasks that exceed the capabilities of individual agents. The effectiveness of such systems depends not only on the agents themselves, but also on how collaboration mechanisms are designed and organized. Note that real-world collaboration is typically partially observable, where each agent can only access partial information about the environment due to physical or privacy-related constraints. However, many existing multi-agent benchmarks assume global observability, and leave limited support for systematically evaluating collaboration mechanisms. To bridge this gap, we introduce MASBench, a multi-agent collaboration benchmark designed under partially observable constraints. It is organized into three progressive task categories: Reasoning, Scheduling, and Game. Through this structure, we progressively evaluate three representative collaboration mechanisms: Protocol, Memory, and Routing. MASBench further provides deterministic evaluation metrics, including performance score, communication cost, and cost effectiveness, to characterize both collaboration outcomes and communication overhead. Experiments across diverse LLM backbones and mechanism configurations offer empirical guidance for effective MAS design. Code is available at: this https URL 

---
# A Tropical Geometry View of Forgetting: A Per-Unit Projector for Knowledge-Preserving Fine-Tuning 

**Authors**: Yuyang Zhang, Xiaoyin Chen, Chunlin Ren, Qihuang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04670)  

**Abstract**: Fine-tuning a language model on new text degrades what it already does. Replay-free projectors such as Adam-NSCL and GPM forbid one shared subspace of a layer's inputs in every row of the update. The tropical geometry of a ReLU layer shows why this is too coarse. In data space, the units' walls are tropical hypersurfaces whose cells are dual to the upper vertices of a zonotope; in weight space, each old token is a hyperplane, and the tokens cut out a polyhedron, the closure of the weights that keep every token on its side. An exact identity joins the two pictures: the squared change of the layer's output under any weight change splits into in-cell, open-to-closed and closed-to-open terms, and the first two live on the tokens each unit fires on. The identity names a gate-aware per-unit projector, and a budget-separation theorem prices exact protection: it costs a unit the rank of its own open tokens, while a shared subspace pays at least the rank of their union in every row. On OPT-1.3b, where 96% of (token, unit) pairs are closed, the projector forgets less than Adam-NSCL at all six matched budgets from 9 to 60 constrained directions per row, the gap widening from $1.1\times$ to $4.3\times$; with 1/5.5 of the directions it halves the forgetting of Adam-NSCL at GPM's energy threshold. On OPT-6.7b, it matches Adam-NSCL's forgetting at matched budget while learning more. As the theory predicts, the open/closed partition is the operative variable: open tokens beat random, sign-blind and anti-gate token sets on 18 of 18 seed-pairs. In pruning repair, every derivative-based local model of the output error at the dense weights is blind to pairs that open: the minimisers of the gate-weighted objective can leave the polyhedron, the objective's closed-form solution is 1.94 nats worse than no repair on OPT-1.3b, and a convex one-sided penalty bounds the escape. 

---
# Efficient Neural Surrogates for Linear Radiation Transport on the Lattice and Hohlraum benchmarks 

**Authors**: Carmelo Gonzales, Steffen Schotthöfer, Cory D. Hauck  

**Link**: [PDF](https://arxiv.org/pdf/2610.04665)  

**Abstract**: Linear radiation transport equations (RTEs) form the simulation foundations underpinning design and analysis tasks in nuclear engineering, inertial confinement fusion, medical imaging, and astrophysics, but resolving the high-dimensional phase space at engineering fidelity remains expensive enough that outer-loop workflows, such as design optimization, uncertainty quantification, and parameter sweeps, are routinely budget-bound on traditional solvers. Neural surrogates promise to relax this bottleneck by amortizing simulation cost across thousands of downstream queries, but the architectural choices and engineered inductive biases that make a surrogate accurate on one transport problem do not transfer straightforwardly across model families. We benchmark two parameter-matched neural surrogate architectures, the physics-attention Transolver and the multi-scale graph network Bi-Stride Multi-Scale MeshGraphNet (BSMS-MGN), as end-to-end approximations of the final-time particle concentration for the two-dimensional linear RTE on the canonical Lattice and Hohlraum benchmarks. An ablation across Fourier features and region-weighted training loss exposes strongly architecture-dependent inductive-bias preferences, indicating that design choices common to physics-informed surrogate workflows must be revisited per architecture rather than imported across model families, and that downstream utility depends on per-QoI sensitivity rather than a single field-level score. The model training recipe, training data, and evaluation pipeline are released alongside this paper to support reproduction, transfer to related transport problems, and evaluation as amortized forward-model components in larger outer-loop simulation workflows. 

---
# PermVLA: Factorization Order as a Regularizer for VLA Learning 

**Authors**: Yanqiao Chen, Yuhan Rui, Dongsheng Hou, Zijie Nie, Yutong Wan, Qi Hao  

**Link**: [PDF](https://arxiv.org/pdf/2610.04659)  

**Abstract**: Vision-language-action (VLA) policies commonly learn action chunks through a fixed left-to-right (LTR) factorization, although the same expert trajectory distribution admits many valid chain-rule factorizations. We identify factorization order as an overlooked regularization choice and introduce causally anchored permutation (CAP), which samples action reveal orders with a tunable chronological prefix. Its auxiliary objective trains one shared policy to predict actions from different known subsets of the same expert chunk, while deployment retains deterministic LTR control. We call this conditional-set augmentation: it creates multiple conditional prediction problems from one expert chunk without adding demonstrations. This discourages reliance on the single chronological prefix used by ordinary teacher forcing. Controlled experiments show that CAP consistently outperforms standard LTR training on LIBERO and LIBERO-Plus, with the same advantage appearing in cross-dataset CALVIN evaluation. A diagnostic that measures the expected squared difference between a chunk's joint log likelihood under two reveal orders verifies that CAP training internalizes agreement across reveal orders. These findings position sampled subset-conditioned auxiliary objectives as a general recipe for constructing VLA regularizers, illustrated by an extension to diffusion action generators. 

---
# SEIS: Self-Evolving Inference Systems 

**Authors**: Zhen Xu, Jingyu Liu, Zongze Li, Tahseen Rabbani, Ce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04646)  

**Abstract**: Inference systems determine how fast and how cheaply language models can be served, so making them faster has direct practical value. However, prior work focuses mostly on optimizing certain parts such as kernels or memory within the large system. In this work, we take a holistic approach and apply agentic self-evolution to optimize the whole system end-to-end. Our SEIS (Self-Evolving Inference Systems) autonomously optimizes the entire mini-sglang engine without human intervention through iterative sessions with inherited experiences and code changes. Serving Qwen3-0.6B on H100, the resulting engine reaches 3.27X the throughput of the original mini-sglang implementation and beats SOTA engines like vLLM, TensorRT-LLM, and SGLang in the single-request workload. The correctness of the optimized inference engine by SEIS is tested in terms of numerical difference and downstream accuracy on math and long-context retrieval tasks. The code and session histories show that the speedup comes from redesigning the whole engine and that building on earlier sessions beats independent attempts. These results suggest that agentic self-evolution can optimize a complex system end-to-end. The evaluation also has to evolve with the engine, and letting agents evolve it is a natural next step. 

---
# LatentIndex: Cross-Layer Sharing with Layer-Specific Selection for Sparse Attention 

**Authors**: Zhaohui Wang, Zhixin Pan, Fanxu Meng, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04635)  

**Abstract**: Sparse attention reduces core-attention computation, but its indexers still incur repeated selection work and per-layer key-cache storage. Reusing selected indices across layers reduces this overhead but constrains multiple layers to the same token set. We introduce LatentIndex, which extends the latent-sharing principle of Multi-head Latent Attention across indexer layers. Each layer group constructs a shared latent cache from its first layer's hidden states, while layer-specific scoring enables independent token selection. Absorbing key decoders into queries enables direct scoring of the shared cache without reconstructing historical per-layer keys. We develop training-free calibration and investigate a training-aware instantiation of this principle. To balance quality and computation, a hierarchical selection (HS) variant lets followers independently refine a shared candidate set proposed by the anchor. With four-layer sharing, LatentIndex reduces logical indexer-cache storage by 61.1% on DeepSeek-V3.2. Across DeepSeek-V3.2 and GLM-5, training-free LatentIndex improves head-wise attention-mass recall over IndexCache by up to 3.28 percentage points while maintaining RULER and LongBench performance close to native DSA. HS further achieves 2.30-2.72 times decode indexer speedups over DSA across 8K-128K contexts, retaining most of LatentIndex's recall. LatentIndex offers a new perspective on cross-layer indexing: sharing continuous representations rather than discrete selections enables efficient reuse while preserving layer-specific token selection. 

---
# Bounds, Decompositions and Null Behaviour of KRATOS: A Mathematical Specification of a Recognition-Comparability Diagnostic 

**Authors**: Maria Dolores Gonzalez, Alberto Barbado  

**Link**: [PDF](https://arxiv.org/pdf/2610.04592)  

**Abstract**: KRATOS is a group-structured bibliometric diagnostic that compares the distribution of documents (participation) with the distribution of citation weight (recognition) over a fixed, finite universe of analytical groups. This note gives its complete mathematical specification and derives the properties that govern its interpretation. Beyond bounds and equality conditions for each component, we show that the recognition-alignment score depends on citation data only through ratios of group mean citation rates to the corpus mean; that the composition ceiling of the participation--recognition factor is an affine function of the total variation distance to the uniform reference; and that the factor itself satisfies Fréchet-type bounds in terms of this ceiling and a participation-weighted recognition score. We establish an exact logarithmic decomposition of the composite index, an ordering between the primary and a reciprocal-symmetric recognition score, invariance properties, and closed-form first and second moments of the recognition ratios under global and stratified permutation nulls. These results explain why composite orderings can be sensitive to small groups, heavy-tailed citation distributions and metadata reassignment. All results are verified numerically with an accompanying script. The specification concerns measurement structure only; it does not define a measure of epistemic change or justice. 

---
# Strong Helps Weak: Directional Cross-Modal Alignment Transfer in Multi-modal LLMs 

**Authors**: Hoigi Seo, Byung Hyun Lee, Minjun Kim, Dohyun Mah, Jongho Lee, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2610.04580)  

**Abstract**: Multi-modal large language models (MLLMs) achieve strong modality understanding by pairing a large language model (LLM) with an encoder for a target modality such as vision, video, or audio. However, improving an MLLM's capability for a given modality typically requires additional training on large modality-specific datasets, incurring substantial data collection and compute costs. Model merging offers an alternative, but it is often infeasible for data-scarce, large per-sample size, or domain-specific modalities (\textit{e.g.}, audio and video), where same-modality model variants are rarely available. In this work, we characterize an intriguing asymmetric phenomenon: merging a well-aligned, data-rich source-modality MLLM into a data-scarce target-modality MLLM substantially improves the target on its own benchmarks. Our theoretical and empirical analyses show that this gain stems from enhanced alignment between modality-specific and textual tokens, induced by the stronger donor modality. Specifically, we derive a mutual-information lower bound that is monotonic in alignment-related quantities and strongly correlated with downstream MLLM performance. Building on this principle, we propose Directional Cross-modal Alignment Transfer (DCAT), a novel framework that transfers textual alignment from a strong, well-aligned source (donor) modality to a weak target (recipient) modality, boosting target-modality performance without further fine-tuning. We further show that the alignment-enhancing objective admits a closed-form weight-space solution computed from only a small calibration set. DCAT outperforms existing model-merging methods, offering an efficient path toward cross-modal alignment transfer. Project page with code is available at \url{this https URL} 

---
# Recursive Improvement of a Differentiable Scientific Software Ecosystem 

**Authors**: Pengcheng Hou, Xiaojun Tan, Sihan Hu, Ruisi Wang, Shuo Chen, Lei Wang, Youjin Deng, Kun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.04561)  

**Abstract**: Differentiable programming connects scientific computation with gradient-based inference, learning and design. Extending these capabilities across a heterogeneous software ecosystem requires specialized effort to implement derivatives, integrate interfaces and evaluate quality. AI coding agents can accelerate this transformation, but translating their capabilities into useful scientific software requires identifying research needs and evaluating how well implementations meet them. We present an environment for agent-driven evolution of differentiable scientific software that connects demand identification, development and quality evaluation. A unified differentiation interface exposes reusable derivative rules alongside existing numerical routines, allowing research tasks to share these capabilities. Research requirements guide development, with implementations assessed through independent derivative checks, workflow tests and performance evaluation. Validated software, research programs and tests become shared resources for subsequent studies. We construct and validate automatic differentiation extensions across 20 packages spanning physical, chemical and biological modeling, with research workflows demonstrating reuse across tasks. Benchmarks demonstrate computational savings over finite differences in gradient evaluation and complete parameter estimation. Research-driven revisions make previously unsupported workflows differentiable, correct derivatives of scientific observables and eliminate redundant computation. Quantum-control and thermal-design studies revise objectives in response to physical evaluation, improving designs while reusing existing derivatives. This work provides a practical approach to expanding differentiable programming across established scientific software and organizing AI agents around the recursive improvement of a shared computational ecosystem. 

---
# $\mathrm{TRIZ}^{a}$: Guiding Agent Evolution from Pattern Recognition to Solution Invention 

**Authors**: Wenyin Liu, Yiheng Huang, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04555)  

**Abstract**: We propose $\mathrm{TRIZ}^{a}$ (TRIZ exponentiated by an agent), a general R\&D automation paradigm that combines TRIZ inventive theory with LLM-driven agent evolutionary search. TRIZ's 40 inventive principles and contradiction matrix provide structured, explainable directions for solution generation, replacing random or untyped mutation with theory-guided ideation. Functional information (FI), operationalized under a frozen reference contract, is combined with TRIZ Ideality to measure useful and harmful function on a commensurable information scale, while hard gates keep promotion distinct from metric improvement. We validate $\mathrm{TRIZ}^{a}$ in cybersecurity--an adversarial and rapidly evolving domain--on PowerDuck GOOSE, CICIoT2023, and CIC-DDoS2019. Under paired-rerun protocols with protocol fingerprinting and hard-gate validation, the legacy experiments yield absolute F1 improvements of $+2.88$, $+4.23$, and $+0.15$ percentage points, respectively. A completed 45-activity CICIoT2023 campaign further increases macro-F1 from $0.8325$ to $0.8483$, but does not pass its frozen promotion gate. Every result remains traceable from contradiction identification and TRIZ principle selection to code transformation, evaluation metrics, and promotion decision. 

---
# Label Agreement Does Not Measure Authorization 

**Authors**: Amir Sabbaghziarani, Bradley Thomas Baker, Theodore J. LaGrow, Sergey Plis  

**Link**: [PDF](https://arxiv.org/pdf/2610.04544)  

**Abstract**: Many groups now delegate label ontology and metadata harmonization to agentic LLM pipelines. We built one and audited it. Our aggregate scores looked healthy, but the pipeline kept failing in ways they did not show, so we set out to find what they hid. Label agreement asks whether a proposed label matches a reference. It does not ask whether the agent was entitled to propose it, whether the output was complete enough to act on, or whether the label moved when the evidence moved. We measured those three separately on COBRE and FBIRN, two schizophrenia and control neuroimaging cohorts from different consortia, and they come apart, from label agreement and from each other. Showing the agent an upstream proposal barely moves label agreement, 0.857 to 0.870, while agreement on the chosen action doubles, 0.409 to 0.830. Output that parses as JSON still drops a required field on 10% of one model's cases and 33% of the other's. And an agent that replays its first answer scores perfectly on original cases and zero once we change the evidence that decides them. Downstream, a row-order error that none of these metrics reports erases most of the diagnostic signal. So we measure these properties apart, pair each with a control, and gate commitment on the result, which makes failures visible and easy to route to a person. None of this prevents failure. Our reference labels are rule-derived, so agreement with them means consistency, not correctness. Code is available at this https URL. 

---
# Decide, Ask, or Defer: Clinical LLMs under Incomplete Evidence 

**Authors**: Mingzhan Yang, Weili Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04542)  

**Abstract**: Clinical LLMs must decide not only what diagnosis to produce, but also whether the available evidence is sufficient for autonomous decision making. Binary DECIDE/ABSTAIN formulations merge distinct non decision states and do not explicitly evaluate information acquisition. We introduce a DECIDE/ASK/DEFER formulation together with a blinded protocol that prevents models from using evidence completeness metadata. We evaluate Qwen, Gemini, and GPT on 200 matched clinical evidence states constructed from DDXPlus. The models show substantial differences in action selection under identical evidence, with disagreement in 137 of 200 states. For Qwen, a matched targeted versus random analysis shows that selected information changes the likelihood of a subsequent autonomous decision more clearly than diagnostic correctness. Its matched DECIDE/ABSTAIN baseline further reveals a safety autonomy tradeoff: the three action policy rescues some erroneous autonomous decisions but also removes some correct autonomous deci sions. These results show that separating information acquisition from clinician deferral exposes behavior that binary abstention hides, without yielding a uniformly improved decision policy. 

---
# Autonomous Structuring of Radiology Reports Across Modalities at Archive Scale Using an Open-Weight Large Language Model 

**Authors**: Friedrich Puttkammer, Fabian Drexel, Marlene Fritzsche, Era Stambollxhiu, Miriam Kumpf, Lena Schmitzer, Lea Schumann, Lina Xu, Johannes Moll, Jannik Lübberstedt, Zeineb Ben Chaaben, Anirudh Narayanan, Hartmut Häntze, Renato Cuocolo, Antonios Billis, Alexander Löser, Jawed Nawabi, Marcus R. Makowski, Cosmin I. Bercea, Shahrooz Faghihroohi, Lisa C. Adams, Keno K. Bressem  

**Link**: [PDF](https://arxiv.org/pdf/2610.04541)  

**Abstract**: Purpose: To develop and evaluate an open-weight large language model (LLM) pipeline that converts an entire archive of free-text radiology reports into structured reports without human oversight. Materials and Methods: In this retrospective study, a pipeline with 150 hierarchically organized templates was developed at one center and tested at a second center on reports from 2010 to 2025. The open-weight model gpt-oss-120B selects the template in three constrained-decoding steps and fills it on one local graphics processing unit. Template selection was scored against expert labels on 914 randomly sampled reports of five modalities, structuring quality on 920 radiography and CT reports corrected field by field by five residents. The pipeline then processed the complete archive of the second center. Proportions are reported with Wilson 95% confidence intervals (CIs). Results: An optimal template set was selected for 74.4% of reports (680 of 914; 95% CI: 71.5%, 77.1%) and an appropriate set for 82.3% (752 of 914; 95% CI: 79.7%, 84.6%), 87.7% for single-region and 54.1% for multi-region reports. Macro semantic textual similarity between output and corrected reference was 0.95 for radiography and 0.97 for CT, residents left 88.7% of 24,638 fields unchanged, and unsupported content was flagged in 1.0% and 1.5% of reports. Of 2,186,982 archive reports, 96.5% received structured output, 2,401,544 structured reports, at 1,258 reports per hour on one graphics processing unit. Conclusion: An open-weight LLM pipeline structured a complete multimodality report archive without human oversight with high content fidelity. Multi-region reports remained the main source of template errors. 

---
# Action-Consequence Alignment for Reliable Planning and Self-Improving in Latent World Models 

**Authors**: Jinping Wang1, Zhiqiang Gao, Xiantong Zhen, Ling Shao  

**Link**: [PDF](https://arxiv.org/pdf/2610.04539)  

**Abstract**: Latent world models learn to predict observed transitions, yet low prediction error alone does not guarantee reliable planning. Inspired by self tickling experiments in neuroscience showing that disrupting motor sensory correspondence increases prediction mismatch, we examine whether learned world models preserve an analogous action consequence this http URL results show nearby alternatives can receive lower prediction errors despite producing physical outcomes farther from the recorded target. With that future treated as a goal, this reveals a concrete prediction planning mismatch: the model assigns a lower cost to an action that achieves the target less accurately. To mitigate this gap, we introduce Action Consequence Alignment (ACA), a training objective that complements forward prediction by penalizing the prediction error advantage of locally searched alternatives over factual actions without additional model components or environment interactions during training. The same principle can also guide additional data collection for self improvement. We demonstrate that across diverse environments and evaluation settings, ACA improves planning performance and reduces real goal error, while ACA guided data collection outperforms random local sampling. These results support action consequence alignment as a practical principle for bridging predictive learning and reliable planning. 

---
# EvoCast: Reliable Autonomous Research Agents for Iterative Forecasting Architecture Evolution 

**Authors**: Kaipeng Xu, Xianli Yan, Yan Wang, Xiang Liu, Shan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04517)  

**Abstract**: Deep time-series forecasting models have rapidly diversified, yet adapting them to a specific task still requires extensive expert effort in model selection, mechanism diagnosis, architecture design, implementation, and evaluation. Existing AutoML methods are constrained by predefined search spaces, while general-purpose LLM research agents lack reliable control over experimental protocols and model promotion. We introduce EvoCast, a fully autonomous research-agent system for iterative forecasting architecture evolution. EvoCast first establishes and diagnoses a task-specific baseline through executed mechanism ablations, then generates evidence-grounded research directions from dataset characteristics, diagnostic results, prior rounds, and failure records. Its central design, cognition-authority separation, assigns open-ended hypothesis generation and code implementation to LLM agents, while deterministic program authorities control source-edit boundaries, canonical evaluation, and promotion decisions. Experimental outcomes are accumulated as evidence to guide subsequent rounds. Results show that EvoCast completes complex architecture modifications with higher implementation success and lower agent-side token/time cost, and develops task-specific architectures that outperform selected baselines, strong forecasting models, and agent baselines in three real-world forecasting cases. The code is available at this https URL. 

---
# Towards Credible Agent-Based Policy Simulations: Disentangling Opportunities and Preferences in a Financial Inclusion Case Study of Egypt 

**Authors**: Alba Aguilera, Georgina Curto, Nardine Osman, Ahmed Al-Awah  

**Link**: [PDF](https://arxiv.org/pdf/2610.04515)  

**Abstract**: Credibility is a central topic for agent-based models intended to support policy-making. Simulations must not only represent the target scenarios and their core dynamics but also demonstrate that their assumptions, parameters, and outputs are empirically grounded and sufficiently accurate for their intended use. This paper addresses this challenge by presenting a general modelling framework, aligned with the Capability Approach, for building credible policy simulations that rely on data and domain-expert knowledge. It then demonstrates how it can be contextualised and implemented to study the social challenge of financial inclusion in Egypt, building on an agent-based model that represents heterogeneous individuals and firms behaving according to their financial states, barriers, opportunities, and preferences. The model is fitted to real-world data in two stages, initialisation and calibration, which respectively build representative synthetic populations and estimate behavioural parameters. By fixing the feasibility parameters, which determine agents' opportunities, and calibrating preference parameters across different population groups, we are able to distinguish and analyse the role of institutional and social barriers in the system, as well as the role of agents' motivations and priorities. This calibration stage provides transparent and group-specific hypotheses about the drivers of observed financial-inclusion gaps, which can further be analysed as gaps between agents' opportunities and realised outcomes, a very relevant insight for policy-making. This paper is thus a step towards improving the credibility and usefulness of policy simulations, strengthening the relationship between the model, the real target system, and the stakeholders who will use it. The code is available at: \url{this https URL}. 

---
# ManifoldCache: Training-Free Diffusion Acceleration via Constraint Manifold Caching 

**Authors**: Prashant Pandey, Devineni Sri Venkatraya Chowdary, Brejesh Lall  

**Link**: [PDF](https://arxiv.org/pdf/2610.04510)  

**Abstract**: Diffusion models for structured scientific generation must produce samples satisfying hard geometric constraints imposed by physics, chemistry, or biology, yet inference in these settings is prohibitively slow, demanding hundreds to thousands of neural-function evaluations per sample. We unify eight state-of-the-art models spanning medical volumetrics, molecular conformations, protein backbone design, crystal structure prediction, and multi-view 3D scenes under a single abstraction, Constraint-Manifold Diffusion Models (CMDMs), in which the target distribution is supported on a manifold defined by an externally specified constraint map. All existing acceleration families fail on this class: quantization exhausts memory on high-dimensional volumetric operators; pruning breaks constraint fidelity; fast ODE solvers allow trajectories to drift off the constraint manifold; and feature-caching heuristics are blind to constraint geometry, inducing mode confusion in the high-noise regime. We introduce ManifoldCache, the first training-free, data-free accelerator designed from first principles for CMDMs. The key insight is that the conditional score decomposes orthogonally into a normal component, which enforces constraint satisfaction, and a tangential component, which navigates within the manifold. Exploiting this structure, we prove that the noise-schedule midpoint is a sharp safe-caching boundary: caching before it incurs provably bounded error, while caching after it guarantees a strictly positive fraction of trajectories suffer mode confusion, a gap that persists up to the boundary. We further prove that deeper network blocks admit provably larger certified cache strides within the safe phase, as a consequence of the score decomposition propagating through block Jacobians. The resulting schedule requires no calibration data, along with zero training overhead. 

---
# VCLMU: Mechanism-Centric Virtual Cell World Modeling for Perturbation Response 

**Authors**: Yuwei Miao, Azim Dehghani Amirabad, Scott Oloff, Junzhou Huang, Tianyu Cui, Rui Liao  

**Link**: [PDF](https://arxiv.org/pdf/2610.04475)  

**Abstract**: Predicting cellular responses to genetic perturbations is a central capability for virtual cells and a key step toward computational modeling of biological interventions. Most existing models directly map an unperturbed molecular profile and perturba- tion to the resulting observation without explicitly representing the latent cellular transition induced by the intervention. We introduce a mechanism-centric virtual cell world model that represents cellular state as a set of Latent Mechanism Units (LMUs) and treats genetic perturbations as actions on these latent states. Each LMU combines a reusable identity grounded in multimodal biological evidence with an observation-specific state, allowing a perturbation to induce mechanism- specific stochastic transitions before decoding the resulting transcriptional response. We train VCLMU through two-stage pretraining, first on around 200K pseudo-bulk perturbation profiles and then on gene-aligned single-cell perturbation data. Across six perturbation-disjoint benchmarks, VCLMU consistently improves perturbation- specific response recovery over strong baselines while maintaining competitive global response accuracy. We further analyze learned LMUs through enrichment between perturbation responses and LMU gene sets and show that they capture structured biological response programs. These results support mechanism-level latent state transition as a useful formulation for virtual cell models that aim to predict and interpret cellular responses to biological interventions. 

---
# InferOpt: Constrained Multi-Objective Search for LLM Inference Configurations 

**Authors**: Qi Chen, Yingying Cheng, Zhaoyi Sun, Li Zhou, Fan Zhang, Jie Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.04473)  

**Abstract**: Serving an LLM means setting dozens of inference-time knobs, from per-layer KV retention to per-layer expert counts. Practice sets them with mechanism-specific heuristics that return a single operating point and do not scale to layer-wise search spaces. We recast inference configuration as constrained multi-objective black-box optimization and build InferOpt, a reusable search framework that requires only variable bounds, a deterministic resource cost, and an evaluation hook. InferOpt searches on a frozen sampled proxy set, rejects over-budget candidates before any model call, tightens the budget adaptively, and re-validates Pareto representatives on full-scale dataset. One pipeline covers a 28-dimensional continuous KV space (Qwen2.5-7B) and a 26-dimensional discrete MoE space (DeepSeek-V2-Lite). On KV, post-prefill pruning cuts the 16K cache by 64.4% and TPOT by 22.9--48.5%, and the searched layer-wise budget by InferOpt beats a matched uniform budget by 7.3% and 14.0% of the Full KV reference points. On MoE, a searched top-k schedule by InferOpt removes 43.0% of routed token--expert pairs while staying within 0.59 points of the default, closer than the matched-budget baselines. Against Random Search, NSGA-II, and MOTPE, InferOpt leads on both spaces, taking the best proxy hypervolume and the lowest retention on KV and staying closest to the uncompressed reference at the lowest experts on MoE. 

---
# Semantic Causal-Factor Inference from Aviation Incident Narratives Using A Variational Autoencoder with Cosine-Similarity-Based Reconstruction 

**Authors**: AZIIDA NANYONGA1, HASSAN WASSWA, UGUR TURHAN, KEITH FRANCIS JOINER, GRAHAM WILD  

**Link**: [PDF](https://arxiv.org/pdf/2610.04472)  

**Abstract**: Aviation accident and incident investigations generate extensive unstructured textual information containing evidence relevant to the causes and contributing factors of safety occurrences. Automatically extracting such information is challenging because causal evidence may be distributed across long and complex investigation narratives. This study proposes a semantic causal-factor inference framework combining natural language processing with a variational autoencoder (VAE) to learn the relationship between aviation investigation narratives and expert-reported probable causes. Investigation narratives and their corresponding probable causes are transformed into numerical representations, after which the encoder maps narrative representations to a probabilistic latent space. The decoder estimates representations of the corresponding probable causes and is trained using an objective that combines Kullback-Leibler divergence with cosine-similarity-based semantic reconstruction. The framework was evaluated using 20,919 finalized U.S. National Transportation Safety Board investigation reports from 2005 to 2020. On the held-out test set, the predicted and expert-reported probable-cause representations achieved a mean cosine similarity of 0.786 (SD = 0.120). The predicted representations also yielded interpretable terms associated with causal information in the reports. The results demonstrate the potential of probabilistic latent representation learning for AI-assisted extraction of causal information from aviation safety narratives while retaining expert investigation as the basis for formal causal determination. 

---
# Reactivating Alignment: Defending LLMs from Jailbreaks via Intention-Aware Input-Output Matching 

**Authors**: Luoyu Chen, Weiqi Wang, Chenhan Zhang, Zhiyi Tian, Shui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04470)  

**Abstract**: Large language models (LLMs) remain vulnerable to jailbreak attacks that conceal harmful intent within complex adversarial prompts. Existing defenses primarily rely on input perturbation or harmful-output suppression, but they rarely model where malicious intent resides, resulting in brittle protection and excessive over-refusal.
We propose SENTINEL, a plug-and-play, generation-time jailbreak defense that reframes mitigation as an intent extraction problem. Our key insight is that instruction-tuned LLMs exhibit strong input--output semantic consistency: regardless of jailbreak complexity, generated outputs tend to align with the attacker's true intent. SENTINEL exploits this property by matching semantically aligned input--output regions to extract intention-revealing subsequences, scores these subsequences using refusal-direction projections to estimate harmfulness, and halts generation when necessary. Experiments on HarmBench across multiple LLMs show that SENTINEL reduces jailbreak success rates to close to 5\% while maintaining low over-refusal. We further demonstrate robustness to adaptive attacks and provide a mechanistic interpretation: SENTINEL re-distributes jailbreak features from alignment blind spots to aligned regions. 

---
# Trinity: Self-Evolving Vision-Language Models with a Self-Verifier 

**Authors**: Youngwan Lee, Yong-Ju Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04469)  

**Abstract**: Self-evolving vision-language models (VLMs), a form of self-improvement in which a model generates its own training data from unlabeled images, are a promising route toward agents that expand their reasoning capability in an unsupervised manner, without relying on ever-larger annotation budgets. Existing methods pair a Questioner that proposes problems with a Solver that answers them, but reward both roles mainly by agreement among sampled answers. Agreement is a weak proxy for truth: it cannot tell whether a question is grounded in the image, whether the proposed reference answer is right, or whether a confident majority is wrong in the same way. We present Trinity, in which one VLM plays three roles, Questioner, Solver, and Verifier, and the Verifier is a self-verifier: an exponential moving average (EMA) of the policy itself, requiring neither labels nor an external judge. The Verifier screens every generated question for image grounding and answer correctness before it becomes supervision, scores Solver reasoning against the image, and adjudicates disputes between the reference answer and a strong Solver consensus, correcting the reference and penalizing the Questioner when the consensus is right. Trained on images alone, Trinity improves Qwen3-VL-8B on mathematical visual reasoning and on science benchmarks with biology content, for example, +8.6 points on the biology split of SciVQR and +12.8 on MathVerse, and its reward dynamics behave as a healthy self-play curriculum should. These results suggest that a self-evolving multimodal agent can strengthen its scientific reasoning from unlabeled scientific images alone, with the model itself serving as the verifier. 

---
# Understanding Generative AI Use in Programming MOOCs: The Role of Course Context and Learner Characteristics 

**Authors**: Marina Lepp  

**Link**: [PDF](https://arxiv.org/pdf/2610.04447)  

**Abstract**: The increasing availability of generative artificial intelligence (GenAI) tools, such as ChatGPT and code-completion assistants, raises questions about how learners integrate these tools into learning activities, particularly in MOOCs that attract diverse participant populations. This study examines the use of GenAI in two programming MOOCs taught in Estonian that differ in duration, workload, topic complexity, and assignment volume: About Programming (4 weeks, 26 expected hours, n = 187) and Introduction to Programming (8 weeks, 78 expected hours, n = 182). Post-course questionnaire data were analyzed using non-parametric statistical methods to examine self-reported adoption, usage frequency, and purposes of GenAI use across courses and learner backgrounds. The results show that GenAI adoption was widespread in both MOOCs, with no statistically significant differences by gender, age, education level, or prior programming experience. However, participants in the longer, more extensive MOOC reported significantly higher usage frequency and were more likely to use GenAI for debugging and idea generation. Reported usage frequency for code explanation and debugging was also higher in the longer course. Exploratory analyses found limited relationships between GenAI use and learning-related outcomes. The findings suggest that course context may play a greater role than learner characteristics in shaping how GenAI tools are used. These results provide implications for instructional design and guidance in programming education. 

---
# Causally Fair Generation with Large Language Models 

**Authors**: Patrik Okanovic, Torsten Hoefler, Drago Plecko  

**Link**: [PDF](https://arxiv.org/pdf/2610.04444)  

**Abstract**: Large language models (LLMs) are increasingly used to generate, complete, and transform information in settings where their outputs can shape consequential decisions, raising concerns about their impact on demographic disparities. In this context, causal inference provides a principled basis for assessing fairness, because it attributes observed disparities to the mechanisms that generated them, which a purely statistical approach cannot do even with infinite data. In LLM generation, a query may request several causally related variables, each of which is both an outcome of interest and a possible cause of other outputs, and the information supplied in the prompt need not follow a topological or a temporal order. This calls for methods that can analyze and selectively remove disparities from such a flexible generation process. In this paper we introduce Causally Fair Generation with LLMs (CFG, for short). CFG extracts relevant concepts, grounds generation in a reference population and causal diagram, and removes user-selected causal effects. CFG also allows pathways deemed justifiable for the task's utility to be retained, which is known in legal literature as business necessity. Further, we provide formal guarantees for our method when eliminating all discriminatory causal effects in the adapted population model, under appropriate causal assumptions. We evaluate CFG with four LLMs in three real-world settings based on population data and on a synthetic dataset with a known causal ground truth. 

---
# RAGrasp: Geometry-Semantic Template Retrieval and Grasp Transfer 

**Authors**: Shenzhe Zhu, Chengxiao He, Jan Harder  

**Link**: [PDF](https://arxiv.org/pdf/2610.04438)  

**Abstract**: We present RAGrasp, a retrieval-augmented pipeline for planar parallel-jaw grasping from a compact set of locally collected, grasp-annotated RGB-D (color and depth) templates. Unlike task-specific predictors trained primarily on large public or synthetic grasp datasets, RAGrasp requires no end-to-end retraining for a new this http URL template memory is constructed from observations collected with the deployment camera, robot, and gripper in the target workspace, thereby aligning stored examples with the local sensing and embodiment conditions. The system uses self-supervised DINOv2 visual fea- tures together with appearance and depth cues to prompt the Segment Anything Model 2 (SAM2), which isolates the query object. A two-stage geometry-semantic retrieval cascade then selects a template, and a confidence gate chooses one of two grasp- transfer estimators. The transferred grasp is refined using mask- support and silhouette-contact constraints before calibrated 2D- to-3D conversion. In real-world trials, RAGrasp achieves 20/20 successful grasps on seen objects and 19/20 on unseen objects. Within the evaluated setting, the results demonstrate deployment- specific grasp adaptation from limited local annotation and tolerance to the tested viewpoint and illumination changes. 

---
# CRAFT: An Agentic Spreadsheet Form Filling System with Template Awareness 

**Authors**: Leyao Gu, Yingjie Xiong, Zirui Tang, Jiangtao Zhou, Yeye He, Chunwei Liu, Xuanhe Zhou, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04437)  

**Abstract**: Spreadsheet form filling requires agents to consolidate external evidence, ground values to precise cells, and preserve irregular template structure. Errors in early edits can overwrite labels or misalign fields, undermining later decisions. We propose CRAFT, a template-aware agent framework that connects reflective validation to constrained local repair. Instead of treating reflection as a free-form request to regenerate the workbook, CRAFT grounds detected errors to spreadsheet regions, restores corrupted template state when necessary, and re-grounds plausible writable slots before subsequent edits. A Rectangle-Aware Slot Grounder (RASG) proposes writable cells, while label-slot hints and protected regions constrain subsequent edits. We introduce FormFillBench, with 327 forms across Instruction-Only and Multi-File tracks. Compared with the strongest baselines, CRAFT improves pair accuracy by 8.51 and 23.38 percentage points on these tracks, respectively. Component-removal experiments support structural adjudication and slot re-grounding within the pipeline, and the framework retains its relative advantage among the methods evaluated with a second backbone. The code and benchmark FormFillBench are available at this https URL. 

---
# What Does a Harness Buy? Tokens, Mostly 

**Authors**: Yangze Liu, Zhongyi Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.04433)  

**Abstract**: A coding agent is a language model wrapped in a harness: the system prompt, the tool set, and the context management that turn a chat model into something that can work inside a repository. Production harnesses ship releases daily, vendors advertise pass-rate gains from harness changes, and leaderboards mix harnesses freely. What is rarely measured is how much the harness itself moves the score when the model is held fixed. We run five models through three production harnesses, Claude Code, mini-SWE-agent, and OpenCode, on SWE-bench Verified, and rerun the same configurations to calibrate how much a score moves when nothing changes but the run. On 447 tasks and the two models we ran there, Claude Code and mini-SWE-agent, the heaviest and the lightest harness, are equivalent within five points. On a 45-task hard subset and five models, swapping the harness flips as many tasks as rerunning the same harness, 13% in both cases, and the tasks a harness wins in one run are not the tasks it wins in the next. The one harness effect that clears the noise is a loss, not a gain: OpenCode trails by up to 9 points on the large pool, and on one model half of that gap sits in runs its output cap cut short. What the harness does decide is the bill. With the same model, the same tasks, and one price list, cost per task differs by up to 3x across harnesses. The gap is set at the first call, by the preamble of system prompt and tool schemas each harness sends with every step, and scaled by the number of steps; per-step growth and per-call tool output differ far less. The provider's price for cached input scales the bill and does not reorder it. The rerun data also give the resolution a harness comparison needs: at the discordance we observe, 45 tasks catch a 13-point gap only half the time and no gap with 80% power, and 447 tasks resolve 5 points, still coarser than the gains many harness changes claim. 

---
# MERCI Cards: An LLM Evaluation and Deployment Framework for High-Stakes Domains 

**Authors**: Aparna Komarla, Annalisa Szymanski  

**Link**: [PDF](https://arxiv.org/pdf/2610.04430)  

**Abstract**: As LLMs are increasingly deployed in high-stakes professional workflows, engineers and researchers require principled protocols to systematically track, monitor, and improve model performance across deployment cycles. We present a mathematical framework for iterative LLM evaluation and deployment, and demonstrate its application to AI systems used in criminal justice. Our framework formalizes LLM integration in high-stakes, high-risk, and resource-constrained domains across model selection, rubric design, evaluations and deployment via a weighted multi-objective optimization. We demonstrate that MERCI Cards can guide improvements of the system across deployment iterations, direct developer attention toward under-performing areas, and focus user attention on validation and error-correction in the LLM's outputs. 

---
# Asking Earns Nothing: Scoring the Decision to Act in BFCL Multi-Turn 

**Authors**: Yangze Liu, Zhongyi Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.04429)  

**Abstract**: An agent that lacks the information it needs should ask rather than act, and the task definitions of agent leaderboards say so. BFCL multi-turn builds two of its four categories around a turn on which the model is supposed to ask, and its scorer never looks at that turn: the gold trajectory there is empty, the checker skips it, and the scripted user cannot answer, so asking earns nothing, guessing costs nothing on that turn, and asking twice loses the item. The benchmark also contains the control experiment for that decision. A should-ask item is a base item with one piece of information removed from one turn, so the same request appears twice at the same turn index, once complete and once not: on the first the model should make the call that changes the world, on the second it should ask. We score one decision per pair, whether the model attempted a world-changing call on that turn, read off the stored trajectories with no LLM judge; acting always and asking always both score 50. On the 223 pairs that pose this decision, gpt-5.4 attempts the call on 83.4% of the complete turns and holds back on 78.0% of the incomplete ones, the best decision accuracy of seven models at 80.7%; on the same items the official score ranks it sixth and puts first a model that lands in the middle here. One added line telling gpt-5.4 not to ask pushes it toward acting on both sides of the pair, so its decision accuracy shows no detectable change, while its official score rises by 13.5 to 23.5 points on the two should-ask categories and on the base twins; the opposite line, telling gemma-4-31B-it to ask first, improves its decision by 4.5 points and gains no official score. The score moves with the push toward action, not with the decision. We release the pairs, a turn-level scorer that runs on any BFCL output directory without an API key, and 31 manually verified bad items. 

---
# CORE-RL: Confidence-Oriented Reliability Evaluation of Black-Box Reinforcement Learning Policies 

**Authors**: Santhosh GS, Ananya Ravi, Devika Jay, Abhishek Sarkar, Perepu Satheesh Kumar, Saurav Prakash, Kaushik Dey, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2610.04418)  

**Abstract**: The deployment of Reinforcement Learning (RL) agents in critical domains must be preceded with a pipeline to evaluate the alignment of the RL agent with complex multi-objective specifications and robustness under real-world environmental drift. However, to protect intellectual property, the RL agent may be delivered for evaluation as opaque executable or remote API, which makes traditional evaluation techniques based on the internals of the policies infeasible. To address this gap, CORE-RL: Confidence-Oriented Reliability Evaluation of black box RL policy is proposed in this paper. The CORE-RL pipeline introduces a Unified Reliability Metric that formally integrates early task termination and safety constraint violations, preventing unsafe policies from masking failures through premature episode halts. By subjecting the policy to a noise certification envelope of perceptual noise, actuation noise and change in environment dynamics, the pipeline computes the finite-sample Clopper-Pearson bounds on unified reliability metric and Hoeffdings' lower bound on reward and safety cost. The pipeline then defines safe operational design domain to report high-confidence certificates for safety and expected performance. Experiments on continuous control tasks demonstrate the CORE-RL pipeline's ability to automatically reject non-compliant policies and map the safe Operational Design Domain (ODD) of safety-aware policies. Thus CORE-RL provides an evaluation framework towards a quantitative, transparent and reproducible, statistical rationale necessary to safely evaluate, compare, and deploy black box RL solutions. 

---
# TimeNet: An Extensible Unified Data Infrastructure for Next-Generation Temporal Foundation Models 

**Authors**: Martin Maritsch, Timo Stoffregen, Thomas Kaar, Behsad Riemer, Maxwell A. Xu, Max Rosenblattl, Juncheng Liu, Nicolas Zumarraga, Yu Yvonne Wu, Denys Herasymuk, Sparsh Rastogi, Hyungjun Yoon, Bosong Huang, Arvind Pillai, Dmytro Lopushanskyy, Tony Chen, Robin Deuber, Yichen Liu, Shvat Messica, Dan Li, Jian Lou, Yuwei Zhang, Jaeho Kim, Renée Rosillo Garcia, Fan Wu, Robert Müller, Elgar Fleisch, Flora D. Salim, Dimitris Spathis, Yuzhe Yang, Aaqib Saeed, Daniel McDuff, Ming Jin, Markus Kreft, Kevin O'Sullivan, Robert Jakob, Azul Garza, Paul Schmiedmayer, Patrick Langer  

**Link**: [PDF](https://arxiv.org/pdf/2610.04407)  

**Abstract**: Temporal Foundation Models (TFMs) aim to generalize across domains, datasets, and tasks. Yet, their development remains constrained by fragmented, task-specific data formats, annotations, and processing pipelines. We introduce TimeNet, an open-source data standard and scalable infrastructure that decouples temporal data from task definitions and represents signals, metadata, annotations, and supervision in a shared, extensible data model. TimeNet supports multimodal signals with regular, irregular, or ordinal time axes and expresses different task families (including classification, forecasting, temporal localization, question answering, generation, and editing) as reusable views over the same recordings. This shared representation enables heterogeneous time-series datasets to be combined for large-scale model training across domains, modalities, and tasks. We demonstrate TimeNet by transcoding datasets with 1.5M task instances spanning diverse domains, modalities, temporal scales, and forms of supervision, while retaining practical I/O performance relative to native formats. TimeNet enables an existing TFN training pipeline to support joint training on a configurable number of heterogeneous datasets through configuration changes alone. We show this capability by training TFM across multiple datasets and tasks, obtaining a 14% F1 score improvement compared with models trained on individual datasets. These results show that TimeNet provides the data and systems foundation needed to move beyond task- and dataset-specific TFMs toward models that can learn jointly across heterogeneous domains, modalities, temporal scales, and forms of supervision from a common data model. 

---
# TrustMed-RL: Long-Horizon Reinforcement Learning for Evidence-Grounded Clinical Diagnosis 

**Authors**: Wenxin Zhan, Yizheng Jiao, Haifeng Song, Shuai Xu, Chencheng Pan, Jiayi Feng, Anjie Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.04387)  

**Abstract**: Medical language models can produce correct diagnoses despite incomplete investigations and unsupported reasoning. To support long-horizon, evidence-grounded diagnosis, we introduce \textbf{TrustMed-RL}. Built from PubMed rare-disease cases and over 24,000 manually annotated image panels, it integrates interviews, examinations, testing, specialist consultation, and literature search through state-dependent actions. Our 8B vision--language policy, trained with clinically adapted GiGPO and coverage-adjusted diagnostic rewards, achieves 37.1\% diagnostic accuracy on 2,500 evaluation cases, outperforming all evaluated open-weight baselines and improving over supervised fine-tuning by 12.4 percentage points.. When success additionally requires acquiring at least 50\% of supporting test evidence, TrustMed-RL achieves 32.5\%, exceeding GPT-4o by 6.8 percentage points. Furthermore, it surpasses all evaluated baselines on MTMedDialog and multiple larger 27--32B models on AgentClinic. In physician review of 200 diagnostically accepted test-set trajectories, 83.0\% receive evidential-grounding scores of 4--5 out of 5. Physicians' assessments suggest that these diagnostic trajectories are trustworthy and aligned with human diagnostic reasoning. 

---
# AgentPersonaBench: Benchmarking Persona-Driven User Simulation 

**Authors**: Jintao Huang, Yifan Wang, Hongyu Shen, Yi Daniel Lu, Shirley Huang, Minsik Oh, Yewen Wang, Muhammad Ahmed Mohsin, Zhen Xu, Yilan Fan, Zichen Yuan, Ahsan Bilal, Zibu Wei, Sankalp Jajee, Henry Gagnier, Saksham Kapoor, Jicheng Wang, Qianfeng Wen, Yixuan He, Steven Dillmann, Jiashu He, Yucheng Lu, Linqiang Guo, Danyang Zhang, Shi Bo, Raunak Mondal, Haixiang Tang, Weihang Xiao, Allen Nie, Jing Tang, Yueying Li, Yifan Simon Liu, Jianheng Hou, Dianzhuo Wang, Qianyu Zhu, Zhixu Silvia Tao, Zhejian Peng, Zihan Wang, Ishan Gupta, Jinxuan Fan, Wanting Jiang, Shushu Liang, Chenxi Qiu, Yijun Wang, Xiaomin Li, Yuexing Hao  

**Link**: [PDF](https://arxiv.org/pdf/2610.04379)  

**Abstract**: We introduce AgentPersonaBench (APB), a benchmark evaluating whether persona conditioning faithfully steers downstream agent behavior. While language models are increasingly deployed for persona-driven user simulation, existing benchmarks primarily evaluate conversational styling or self-reports rather than authentic behavioral fidelity. APB evaluates latent persona adherence one trait at a time, embedding each target trait within a complete synthetic profile without explicitly naming the trait or disclosing the test. Ground-truth adherence is verified strictly from observable actions across four interaction surfaces of increasing realism: survey, chat, web (interactive web environments), and app (desktop software environments). APB comprises 2,460 tasks spanning 867 traits, verified through automated audits and expert review. Our evaluation of 20 frontier model arms demonstrates that high-fidelity user simulation is already attainable: leading models achieve up to 84.7% full-pass adherence under unprompted conditions. At the same time, APB identifies clear behavioral boundaries: adherence drops across interaction modalities (only 37.9-64.3% pass all four surfaces), multi-attribute demands degrade retention, and competing model families exhibit pronounced behavioral divergence. 

---
# Do Tool Calls Execute as Intended? Measuring and Repairing Intent-Execution Correspondence in LLM Agents 

**Authors**: Boyang Yang, Zhenhao Li, Ziyao Yang, Kanghui Jia, Xin Yin, Mingmou Liu, Haoye Tian  

**Link**: [PDF](https://arxiv.org/pdf/2610.04375)  

**Abstract**: Agents built on large language models (LLMs) build and run software through tool calls. A call reaches its program through several hops, and any hop can change the call without notice. When the changed call fails, the agent retries a correct call, which costs users time and money. Benchmarks and failure analyses do not see the change, because they read the call and its result but not what a hop received. We define intent-execution correspondence (IEC) as the property that the executed action matches the action the emitted call denotes under the tool contract. Our protocol observes what each hop received without executing the call, and names the first hop that changed it by the receiver's own parser. IntAct then delivers the call in a form that this hop cannot alter, or refuses the call. We build IEC-Bench from the changes observed in real-world use, with chains of dependent calls under the execution paths of 4 widely-used harnesses.
In 47,828 shell calls within production sessions, Claude Code's Bash tool changes 12.0% of the calls that carry code, escape sequences, or long text. For 80.7% of the calls whose backslashes are changed, the wrong action runs without any reported error. All 10 measured harnesses change a call. Trajectory-based judgment attributes 95.1% of the production failures to the LLM, although the path caused more than half of them. On IEC-Bench, the path raises the token cost per passed task 2.4 times (up to 12.3 times). A hop that changes a call also hides the changes after it, so 55.1% of the failures on one path appear only after its first hop is repaired. IntAct, deployed in a commercial product, recovers 79.2% of the failures with a changed call. Harnesses should therefore be designed and tested hop-by-hop to ensure a correct call executes as intended or is refused. 

---
# Functionally Equivalent or Not? Graph-Grounded Differential Surrogate Execution for Code Equivalence 

**Authors**: Amit Kachroo, Like Hui, Haitao Mao, Yuhao Zhang, Nguyen Vo  

**Link**: [PDF](https://arxiv.org/pdf/2610.04371)  

**Abstract**: Determining whether two programs are functionally equivalent is central to code modernization, patch validation, refactoring, and code-generation evaluation. Yet the usual signals are incomplete: tests cover only finite inputs, textual similarity confuses implementation with behavior, and unconstrained LLM judgments are difficult to audit. Direct execution is often impossible when a program depends on an obsolete, licensed, unavailable, or unsafe environment. We introduce FEAgent, a selective equivalence assessor agent that combines typed program-graph evidence with differential surrogate execution. FEAgent first aligns public interfaces and behaviorally relevant graph anchors, then issues bounded queries over call-flow, control-flow, data-flow, type, import, and effect relations. Next, a branch-aware generator agent proposes discriminating inputs, and two blinded LLM surrogates independently predict source and target observables. Every claim and predicted divergence is recorded in an evidence ledger. A deterministic reconciler then returns EQUIVALENT, INEQUIVALENT, or UNCLEAR rather than forcing a verdict when paths are uncovered or evidence conflicts. We evaluate FEAgent on function-level equivalence and repository-level bug patches, where the existing oracle is a benchmark label or a passing test suite. Every disagreement with that oracle is adjudicated by direct execution, revealing errors in benchmark labels and behavioral divergences missed by unit-test-only scoring. On EquiBench, execution confirms FEAgent's disagreements with published labels on 216 of 1,200 evaluated pairs (18.0%); on SWE-bench Verified, 94 of 331 test-passing agent patches (28.4%) diverge from the reference patch. FEAgent thus serves as an audit layer between testing and formal verification, keeping its evidence reviewable and its uncertainty explicit without claiming a proof of equivalence. 

---
# ShadowMiner v1 - An Experience Report on Implementing and Measuring a Problem-and-Hypothesis Discovery Engine 

**Authors**: Jinhyuk Choi  

**Link**: [PDF](https://arxiv.org/pdf/2610.04339)  

**Abstract**: ShadowMiner v1 is a system that automatically discovers research problems and generates hypotheses from AI papers. It is a nine-stage pipeline. It structures documents into a knowledge graph and finds graph gaps in it - structural blind spots in research. These graph gaps are included in the LLM generation prompt. Each generated hypothesis is then verified by checking whether it is already covered by existing research, scoring its quality, and checking that the facts it relies on are accurately drawn from its sources. This report does not propose a new generation or evaluation technique. It describes our experience of implementing and applying ideas from prior work, and measuring whether each one actually contributed. 

---
# Suppressing Pressure, Amplifying Evidence: Self-Guided Attention Steering to Mitigate Sycophancy and Stubbornness 

**Authors**: Yinghao He, Mengyu Xu, Haixiang Sun, Donghan Li, Yibo Wang, Lixu Wang, Kezhen Chen, Chi Li, Chunwei Liu, Bharat Bhargava, Chongyang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.04329)  

**Abstract**: Reliable language models should resist unsupported user pressure while effectively using objective contextual information. However, models may exhibit sycophancy by yielding to unsupported user pressure or contextual stubbornness by failing to update their answers when relevant contextual information warrants revision. Evaluating interventions for these failures separately can obscure whether mitigating one failure exacerbates the other. To assess this trade-off, we introduce CoPE-Bench with six conditions per question: a neutral baseline, correct or incorrect user pressure, contextual information consistent with or conflicting with the neutral answer, and a joint condition combining incorrect claims with conflicting contextual information. To regulate the influence of user pressure and contextual information, we propose SPAE (Suppressing Pressure, Amplifying Evidence), a training-free framework that uses the model's own judgments to identify relevant tokens, suppressing user pressure and amplifying contextual information through token-level attention steering. On average across five backbones, SPAE reduces pressure following by 18.8 percentage points and increases joint-condition updating by 5.5 percentage points relative to the strongest baseline in the main comparison. In two-turn dialogue, it improves joint-condition updating by an average of 13.2 percentage points over the strongest prompting baseline. The source data and codes can be found at this https URL. 

---
# Bidirectional Preference Synthesis: Learning Prompt-Conditioned Preferences from Boundary Failures 

**Authors**: Junbo Wang, Lidong Lu, Zhuoqun Li, Guiping Jiang, Xiangyu Wu, Tinghai Zhang, Tong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04328)  

**Abstract**: Correction-based offline preference pipelines commonly treat model failures only as rejected responses under the original prompt. This supervision is incomplete for boundary failures: responses that violate the given instruction yet coherently satisfy a nearby intent or constraint setting. We introduce Bidirectional Preference Synthesis (BPS), a data-construction method for standard Direct Preference Optimization (DPO) that makes this missing prompt dependence explicit. For each validated boundary failure, BPS keeps the conventional forward pair under the original prompt and adds a reverse pair under a synthesized achieved prompt, so the same response is rejected where it is wrong and chosen where it is right, without changing the DPO objective, training a reward model, or requiring online sampling. On Qwen3-4B-Instruct-2507, BPS preserves original-side pairwise ranking while raising achieved-side ranking accuracy from 6.8% to 62.3% on held-out crossed anchors, with a similar shift under a Kimi-K2.6 cross-teacher probe. A blind human audit supports the intended reverse preference direction, and downstream evaluations show the clearest separation from Forward-DPO in multilingual multi-turn instruction following, with consistent capability-retention patterns on agentic, tool-use, and code checks. 

---
# From Latent Space to Jacobian Space: Measuring, Evading, and Training Against Safety-Content Accessibility 

**Authors**: Mohammad Mosafer  

**Link**: [PDF](https://arxiv.org/pdf/2610.04316)  

**Abstract**: Output-only safety monitoring sees only the end of a model's computation, yet the model computes its answer before emitting it: what it is internally poised to say is safety-critical. Jacobian-space (J-space) readouts, linear maps from hidden states to the output vocabulary via the model's input-output Jacobian, have been proposed as a window on behaviorally accessible internal content, and a first safety protocol (JADR) showed that danger recognition is readable there. What remains unknown is how this accessible content relates to the latent-space safety geometry studied by representation engineering, and how safety training shapes it. We introduce two quantitative bridges: (i) a transport-amplification profile $A_\ell$, measuring how strongly a latent safety direction is carried toward output space by each layer's Jacobian, and (ii) a paired base-vs-tuned protocol that attributes accessibility to training provenance. Across four small model pairs (135M to 1.5B, three lens-fit seeds), tuned checkpoints concentrate recognition in upper-middle layers, and at 0.5B DPO installs refusal while J-space recognition drops to chance: training can widen the accessibility gap exactly where behavioral safety looks best. A deployment audit closes the loop: monitor-aware GCG-style suffixes suppress the prompt-side monitor at zero behavioral cost at every scale; only a learned monitor rung resists its own adaptive re-attack; the training-time defense fails its re-attack at every penalty weight; steering shows the amplification profile is descriptive, not causal; and continuous-prefix optimization fails where discrete search succeeds. Safety monitoring, training, and evaluation must operate on accessibility itself, not on outputs alone. 

---
# MOIRA: Mass-Oriented Indexing with Ragged Attention for Long-Context Decoding 

**Authors**: Dich Nhat Minh Nguyen, Tran Dang Duong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2610.04313)  

**Abstract**: Long-context decoding is limited by memory bandwidth, because every output token reads the KV cache of every layer. Sparse decoding reduces this cost by reading only part of the KV cache. We observe that the number of pages a query needs varies widely across KV heads, layers and steps. Fixed budgets are simple, but they are sized for demanding cases and tuned per workload; adaptive budgets follow this variation more flexibly, but existing designs pay for it with extra selection cost or training. At the kernel level, FlashAttention-3 (FA3) and FlashInfer are designed for rows of similar length: with page lists whose length differs per KV head, they either pad the lists (forfeiting much of the sparse saving), leave thread blocks unbalanced, or rely on a host-side plan that runs outside the CUDA graph. We propose MOIRA, a training-free sparse decode path in vLLM whose budget adapts per KV head and per layer. For every request, layer, KV head and step, a coverage rule keeps the smallest set of pages whose estimated attention mass reaches a fraction $\gamma$. A new kernel, self-planning attention, lets each thread block derive its own share of the work from the list lengths, so the whole decode step stays inside the CUDA graph. On an H200, at RULER's 128k context, MOIRA with $\gamma=0.99$ matches dense accuracy while reading about 30% of the pages and reduces the time per output token (TPOT) by 2.2-2.5$\times$ relative to dense FA3; with $\gamma=0.98$ it reduces TPOT by 2.7$\times$ and stays within the noise of dense. Under high serving load it raises throughput by up to 51%. These results suggest that a budget adapted per head and layer, paired with a kernel that keeps such budgets inside the CUDA graph, makes sparse decoding both flexible and fast. 

---
# EnvDreamer: Large-Scale Multimodal-to-Environment Generation for Embodied AI 

**Authors**: Kabir Swain, Sijie Han, Antonio Torralba  

**Link**: [PDF](https://arxiv.org/pdf/2610.04301)  

**Abstract**: Large datasets and high capacity models have accelerated progress in vision and language. This work introduces a platform aimed at bringing comparable gains to embodied learning, world models, and robotics. We present EnvDreamer, a framework that uses large language and vision language models to generate Unreal Engine 5 environments for embodied AI and robot training. EnvDreamer enables sampling of large, diverse, interactive, customizable, and validator passed virtual environments for training and evaluation across navigation, interaction, and manipulation. We illustrate the platform with a large set of generated scenes and simple baselines. Policies trained on EnvDreamer generated environments, without explicit mapping or human task supervision, achieve competitive results on multiple embodied benchmarks spanning navigation, rearrangement, and manipulation. EnvDreamer also supports image-conditioned reconstruction for real-to-sim studies. Finally, we release EnvDreamer-20k, a dataset of 20,000 validator passed environments with task programs, scene graphs, trajectories, and metadata to support reproducible benchmarking. 

---
# Language-Conditioned Token and Reasoning Efficiency in Large Language Models: A Paired Cross-Lingual Study Protocol 

**Authors**: Genliang Zhu, Chu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04295)  

**Abstract**: Large language models incur language-dependent representation and inference costs, but existing comparisons often conflate input language, assigned observable-trace language, and answer realization. We specify a prospective paired study that separates these interfaces while holding the semantic item, checkpoint, and answer oracle fixed. The initial design instantiates 240 exactly scored items rendered from templates in English and seven non-English languages, three distinct-lineage open-weight checkpoints, three trace-token budgets, 22 input- and trace-language conditions, a fixed answer reserve, and a separately counted delimiter: 47,520 initial core runs before prospective sample-size selection. RQ1-RQ3 estimate input and trace effects by intention-to-treat with failure-inclusive terminal accounting and test answer realization by cloning a sealed prefix and runtime-native KV state into eight crossed branches. Pre-freeze independent language review, fixed-form ASCII selectors, and code/surface/solver agreement constrain the realization test. H1-H5 share one Holm family and a global simultaneous component band. A secondary randomized experiment compares one-long-attempt and complete K-short-attempt policies at equal trace allowance under frozen seeds and oracle-blind aggregation; it is a full-policy contrast because answer capacity differs. Outcomes include exact token spans, correctness, latency, runtime-exposed memory, and qualified same-host operating-system-reported energy over prespecified hardware rails. The protocol separates tokenizer expansion, observable-trace cost, and answer-realization cost without treating visible traces as internal cognition or operating-system estimates as physical cross-device energy. No confirmatory model outcome is reported; result fields remain disabled until the frozen evidence ledger passes independent verification. 

---
# LMBuild: Evaluating LLM Agents for Generating Buildable and Functional Structures 

**Authors**: Jiateng Liu, Rushi Wang, Cheng Qian, Xuejun Zhang, Sun Li, Jiayu Liu, Yifan Shen, Xu Cao, Jiarui Yao, Bingxuan Li, Ruhi Sarikaya, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2610.04292)  

**Abstract**: LLM-based agents are increasingly capable of generating complex 3D structures, with the potential to reshape how objects are designed and realized in the physical world. Yet, producing elegant geometry is fundamentally different from producing objects that can be built and perform their intended functions. Existing evaluations largely focus on geometric quality while overlooking physical realizability. We introduce LMBuild, a benchmark for evaluating LLM agents on generating buildable and functional structures. LMBuild represents generated objects as assembled structures comprising part decompositions, joints, materials, and sequences. To support reproducible evaluation, we provide a unified framework consisting of: (1) an interactive environment in which agents can use tools to retrieve, create, and place components to construct objects; (2) a curated benchmark that repurposes established CAD datasets and augments them with knowledge from Wikipedia; and (3) a evaluation framework covering structural soundness, functional affordance, design quality, and physical realization. Evaluations across 30 systems reveal several intriguing findings: (a) Soundness and alignment are no longer the primary bottlenecks for frontier closed-source models, while functional affordance and physical operability remain substantially more challenging; (b) stronger models more effectively create new components, whereas weaker models tend to rely on retrieval; and (c) providing functional specifications substantially improves part completeness, kinematics, and physical operability. These results show that generating real-world structures requires deeper reasoning about functional affordances, mechanics, and designing and creating novel components. We expect LMBuild to provide a foundation for measuring progress and incentivizing research toward agents that generate buildable and functional structures. 

---
# VIGIL: Verifier-Informed Gated Improvement Loop for Spreadsheet Question Answering 

**Authors**: Kang Li, Lu He, Sandarsita Guntupalli  

**Link**: [PDF](https://arxiv.org/pdf/2610.04287)  

**Abstract**: Enterprise agents should improve from delayed feedback without allowing every correction to rewrite system behavior. We study continual harness learning for corpus-level spreadsheet question answering. Building on FiCo (Find-then-Compute), a static retrieval-and-execution backbone, we introduce VIGIL (Verifier-Informed Gated Improvement Loop). Within a question, VIGIL verifies and repairs diversely prompted Structured Query Language (SQL) candidates. Across episodes, delayed labels update only a contract calibrator and query/result-column selector. The base model, prompts, retriever, and recorded candidate pool remain fixed in the continual-learning protocol. With the gold workbook and expected type supplied, the ungated and dual-gated full-replay variants reach 82.4% and 82.0% forward accuracy over 79 documents, from a 75.8% static baseline. The dual gate has the larger retrospective gain (2.7 versus 2.3 points), and its 6.3-point forward gain has a 95% t-interval of 5.6-6.9. In a separate stricter split that excludes 16 documents and 380 questions from fitting and online promotion, the accuracy-only gate raises mean held-out accuracy across ten final harnesses from 80.3% to 86.7%. Calibrator-only adaptation gains 5.9 points, close to the combined 6.4-point gain. Yet three of 26 gate-approved updates reduce held-out accuracy relative to their incumbents, so replay-buffer non-regression does not imply held-out non-regression. MiMoTable and external-task case studies test within-task verification and reuse the same promote-or-retain discipline. Overall, the results support bounded, auditable harness adaptation while revealing where finite replay gates fail to generalize beyond their promotion buffers. 

---
# Dense Neuro-Symbolic Reasoning in a Unified Geometry State 

**Authors**: Ruoran Xu, Wending Gao, Haoyu Cheng, Xiaoqiang Kang, Qiufeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04280)  

**Abstract**: Geometry reasoning is naturally stateful: solving a problem repeatedly alternates between structural proposals and exact deductions. We formulate this process as dense neural-symbolic coupling, in which neural guidance and symbolic execution share a typed state and communicate through executable actions at every search step. Neural proposals contribute theorem instances, constructions, and algebraic bridges; the symbolic runtime applies registered rules, propagates exact constraints, and records provenance. A nested controller allocates computation first between neural and symbolic proposal sources and then among admitted actions. We instantiate the framework in OmniGeo, a single solver for plane, analytic, and solid geometry. With Claude Sonnet 4.6, OmniGeo reaches 94.2%, 88.5%, and 89.8% on FormalGeo7K, Conic10K, and SolidFGeo, respectively (90.8% macro average), and solves 21/30 IMO-AG-30 problems. 

---
# CADForge: Agentic Single-View CAD Reconstruction with Explicit Geometry Reasoning 

**Authors**: Keyang Lu, Zhifei Yang, Tianao Dong, Mingzhe Xing, Zhen Xiao, Yikai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04262)  

**Abstract**: Reconstructing editable parametric CAD models from a single-view image is of great practical value for modern manufacturing, yet remains challenging due to incomplete geometric observations and complex inter-part relationships. To address it, we propose CADForge, an agentic framework that progressively converts a single image into CadQuery programs. CADForge decomposes an object into CAD-meaningful components and performs explicit geometric reasoning for each component, a process that first identifies CAD-relevant constraints and then translates them into precise modeling parameters through mathematical code. The inferred parameters then drive component-wise synthesis of executable CadQuery programs, with a review agent evaluating the resulting geometry and providing targeted feedback for iterative refinement. To further improve robustness and efficiency, CADForge incorporates a failure-guided toolkit construction mechanism to distill accumulated experience into tools, and maintains a compact parametric CAD memory for retrieving modeling context on demand. Experiments on diverse single- and multi-part objects show that CADForge consistently outperforms existing baselines in reconstruction fidelity and perceptual quality, demonstrating an effective approach to accurate single-view CAD reconstruction. 

---
# Spec2Game: Can LLMs Generate Complete Playable Games from Detailed Specifications? 

**Authors**: Yixue Cai, Yuzhe Zhao, Hanxiang Chao, Qingsen Ma, Ziheng Xiong, Jinhu Qi, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2610.04253)  

**Abstract**: Generating an executable program does not necessarily mean that it correctly implements the behavioral requirements specified in natural language. To evaluate large language models' ability to realize detailed specifications as complete interactive programs, we introduce Spec2Game, a benchmark that requires models to generate complete Pygame projects from detailed natural-language game specifications. Spec2Game comprises 15 game families and 150 task instances, with one canonical task and nine controlled rule variants per family, spanning three levels of implementation complexity. Using source-code, runtime, and visual evidence, we evaluate generated projects along four dimensions---Executability, Specification Realization, Code Quality, and User-Facing Quality. Across 14 LLMs and 3,330 generated projects, we find that high executability does not imply faithful specification realization. Component-level analysis further shows that models perform substantially better on Game Element Modeling than on Rule and Mechanism Modeling or Goal and Termination Modeling, indicating that faithfully implementing game rules and termination logic remains a major challenge. 

---
# On the Steering Dimensionality of Refusal in Language Models 

**Authors**: Han Wang, Erik Miehling, Dennis Wei, Karthikeyan Natesan Ramamurthy, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04245)  

**Abstract**: Existing activation steering methods often assume that a high-level concept can be mediated by a single steering direction. To support this, two complementary interventions should be achieved: additive steering should induce the target behavior, while directional ablation should suppress it. Yet behaviors may occupy richer activation geometries beyond a single direction, and semantically similar behaviors may be represented by distinct directions. In this work, we study how many directions can reliably control two different types of refusal behaviors: refusal triggered by the safety alignment and refusal in general contexts. Given the limited expressive capability of a single steering vector, we study the general setting of steering subspaces and introduce the notion of steering dimensionality as the minimum subspace dimensionality required to reliably control a behavior. We characterize sufficient steering subspaces that cover the full extent of the target behavior through both the (monotonic) improvement before the sufficient dimensionality, and the saturation beyond it. Empirically, we find that refusal triggered by safety alignment is 1-dim steerable, while multiple distinct steering directions can achieve comparable control. In contrast, refusal in general contexts exhibits substantially richer activation geometry where even 5-dim steering subspaces fail to reliably capture its full steerable variation. Our results reveal that the activation geometry underlying refusal is highly context-dependent and can be substantially more complex than a single linear steering direction. 

---
# TCMClinicalReason-Bench: Can Language Models Reason from Pathogenesis to Prescription over Real-World Clinical Cases? 

**Authors**: Jirui Dai, Chenkai Zhang, Yan Jia, Yukai Wang, Ruiyang He, Changyong Luo, Zhi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04215)  

**Abstract**: Large language models (LLMs) can generate clinical narratives that are insufficiently grounded in patient-specific evidence. In traditional Chinese medicine (TCM), errors can propagate from etiology and pathogenesis through syndrome diagnosis and treatment principles to prescription generation. We developed TCMClinicalReason-Bench using 2,000 multicenter electronic health record cases to distinguish case-grounded responses from fluent but unsupported diagnostic and therapeutic conclusions. Five general-purpose and two TCM-specific LLMs were evaluated in zero-shot settings. An evidence-constrained rubric assessed seven diagnostic and therapeutic components and three cross-block relations, allowing case-supported alternatives. Qwen3.7-Plus with TCM retrieval served as the automated judge, alongside parallel blinded ratings by five senior TCM clinicians on a 600-case subset. Structural completeness was nearly saturated (99.3-100.0%), but normalized content scores ranged from 40.7% to 54.1%. The five general-purpose models averaged 50.0%, versus 41.3% for the two smaller TCM-specific models. Cross-block logic consistency ranged from 60.8% to 66.8% and correlated moderately with content across cases (Pearson's r = 0.515-0.656). Deficits were greatest in prescription generation, prescription analysis, and symptom-guided modification. In judge stress testing on 100 independent cases, perturbation detection rates across the three relations were 57%, 56%, and 31%, with contradictions detected more reliably than omissions. Separating component quality from cross-block consistency localizes failures missed by endpoint and completeness metrics and identifies where clinician oversight remains necessary. 

---
# Fine-Tuning VLM for Enhancing AI's Spatial Intelligence: Understanding 3D and 2D Rotations 

**Authors**: Uttamasha Monjoree, Wei Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.04206)  

**Abstract**: Spatial intelligence is a fundamental skill in multiple domains, such as Science, Technology, Engineering, and Mathematics (STEM), Medicine, Architecture, and Construction. Recent studies indicate that Vision-Language Models (VLMs) still face limitations in spatial reasoning, which inhibits artificial intelligence (AI) from performing practical spatial tasks. Using multiple object-rotation datasets developed for training and evaluation, our experiments demonstrated promising improvements in both 2D and 3D rotation detection. Fine-tuned Google DeepMind-built Gemma-4 mixture-of-experts (MoE) models significantly outperformed fine-tuned Gemma-4 generalist models in predicting rotations defined by both their axes and angles. Fine-tuning also substantially improved angle estimation for 2D representation without requiring an explicit coordinate system. Furthermore, identifiable objects did not improve angle-detection accuracy; instead, objects with prominent linear features showed improved performance. 

---
# ALoDLM: Adaptively Looped Diffusion Language Models 

**Authors**: Liancheng Fang, Zhuowei Li, Youngeun Kim, Tianchen Zhao, Rajat Koner, Jiaye Wu, Linghan Xu, Xuanbai Chen, Xiang Xu, Zheng Zhang, Jakub Zablocki, Nishant Sankaran, Yifan Xing  

**Link**: [PDF](https://arxiv.org/pdf/2610.04198)  

**Abstract**: Diffusion language models (DLMs) enable fast generation by predicting multiple tokens in parallel, but their practical adoption remains limited by a persistent quality gap relative to comparably sized autoregressive (AR) models. We attribute this gap to a computation-difficulty mismatch: within a partially observed sequence, some unknown tokens are easy to predict, while others require substantially more computation. Existing DLMs nevertheless apply uniform computational depth to all unknown positions at each denoising step. We introduce ALoDLM, which replaces uniform computation with token-adaptive latent recurrence. At each denoising step, ALoDLM iteratively refines latent representations and allocates computation according to token difficulty. Tokens ready to commit are fed back as discrete context, while unresolved tokens retain and further refine their latent states through additional recurrent passes. To learn token prediction and computation allocation jointly, we formulate token-wise computation schedules as latent variables and derive a conditional negative evidence lower bound (NELBO). We train ALoDLM at 1.7B and 8B parameter scales. Across eleven benchmarks, ALoDLM outperforms all evaluated DLMs and the corresponding AR baselines in average benchmark score at both scales. ALoDLM also retains fast parallel decoding, yielding a strong quality-efficiency trade-off among evaluated autoregressive and diffusion models under optimized inference engines. 

---
# Asynchronous Is Nearly Free for Evolution Strategies on Long-Horizon Agentic Tasks 

**Authors**: William Hoy, Jingxuan Fan, Nurcin Celik, Xu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.04196)  

**Abstract**: LLM-based long-horizon agentic post-training is often bottlenecked by rollout generation: trajectories span many interaction turns, completion times vary substantially, and synchronous update barriers leave faster workers waiting for stragglers. Asynchronous reinforcement learning which has been adopted in LLM post-training addresses this inefficiency by consuming trajectories as they arrive, but introduces policy lag and off-policy optimization. Evolution strategies (ES) offer a backpropagation-free alternative for LLM post-training, yet it relies on a larger number of rollouts and existing practices have remained largely synchronous. In this short-form paper, we introduce bounded-staleness asynchronous ES and demonstrate it on Endless Terminals benchmark using Qwen2.5-7B-Instruct. Across three evaluation seeds, natural Async-1 matches synchronous ES, achieving 25.9\% versus 25.4\% held-out success. Controlled schedules that delay 10\% of each update cohort by four or eight policy updates reduce success by only 1.6 and 3.1 percentage points, respectively, without explicit off-policy correction. GRPO performs better overall, reaching 29.0\% held-out success, but importantly our results show that ES tolerates moderate policy staleness with limited degradation, opening possibilities for future improvement of ES-based post-training with asynchronous algorithms. To the best of our knowledge, we are the first to demonstrate the effectiveness of sync and async ES on a multi-turn terminal style agentic coding task. 

---
# MemLeak: Cross-User Semantic Leakage in Multi-Tenant AI Agent Memory 

**Authors**: Priyanka Mudgal, Kai Zhao, Guilin Zhang, Andy Olsen, Ezekiel Miller, Xu Chu, Aletta Johanna Blanken  

**Link**: [PDF](https://arxiv.org/pdf/2610.04195)  

**Abstract**: Personal AI agents in enterprise multi-tenant deployments share a common vector store for long-term memory. Shared embedding spaces create a surface for cross-user memory leakage: a user's query can retrieve semantically adjacent memories belonging to another user through ordinary cosine-similarity retrieval, without any exploit. We formalize this as cross-user admissibility failure and evaluate it across six experiments, plus follow-up ablations, under both sparse (TF-IDF) and production-faithful (MiniLM-L6-v2) retrieval. Non-adversarial, incidental leakage reaches 70--100\% under pooled {same-team} retrieval; adversarially crafted memories achieve 90--100\% top-$k$ placement, exceeding weaker keyword-based attacker baselines, with score lifts of $+0.416$ to $+0.511$ under production-faithful dense retrieval (Config B); and end-to-end response contamination reaches 5.00/5 under a production retrieval path and 4.67/5 with Claude Sonnet~4.5, with contaminated responses often scoring as helpful or more helpful than clean ones, a gap validated against human judgment. Among three architectural mitigations, only hard post-retrieval ownership gating consistently restores the clean baseline (1.00/5) across {two generation models, at a measured latency overhead of roughly 1.4~ms per query. 

---
# Agentic AI with Structured CoT for Enhancing AI's Spatial Intelligence: Visualization and Reasoning of Rotation 

**Authors**: Uttamasha Monjoree, Wei Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.04188)  

**Abstract**: Recent studies show that artificial intelligence (AI) with language and vision capabilities still experiences limitations in spatial reasoning. In this paper, we have studied the spatial capabilities of advanced generative AI to understand the rotations of objects in 3D space, utilizing AI's image processing and language processing features. We trained and examined the spatial intelligence of a generative Agentic AI model (GPT-5.6) to understand the spatial rotation process with rotation diagrams based on the revised Purdue Spatial Visualization Test: Visualization of Rotations (Revised PSVT:R). We improvised the Revised PSVT:R by superimposing additional graphical and contextual features to evaluate how different Chain-of-Thought (CoT) reasoning strategies influence model performance. The results indicate that structured CoT reasoning improves the spatial reasoning performance of the base GPT-5.6 model in both datasets (PSVT:R and PSVT:R with coordinate system). We used three CoT approaches - (1) Structured CoT, (2) few-shot Structured CoT, and Structured CoT with Self-optimized Prompt. The three CoT approaches evaluated in this study showed no significant performance difference. Results showed that combining structured CoT reasoning with relevant contextual information leads to considerable improvements in VLM performance on 3D rotation tasks, demonstrating the potential of agentic AI for more effective spatial reasoning. However, when contextual information is removed, structured CoT reasoning alone provides limited improvement, and the models continue to exhibit notable difficulties in understanding spatial transformations. These findings suggest that effective spatial reasoning in VLMs relies on the integration of visual, textual, and reasoning-based information in future agentic AI systems for spatial intelligence. 

---
# EvalResearchBench: Can AI Agents Design Their Own Evaluations? 

**Authors**: Yaolun Zhang, Tianyi Xu, Yujie Zhao, Jishen Zhao, Qingyun Wu, Huazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04184)  

**Abstract**: Recursive self-improvement (RSI) relies on evaluation feedback to assess progress and guide further research, yet repeatedly running complex benchmarks is costly and slows iteration. Human experts reduce this cost by selecting benchmark subsets or designing compact suites. We ask whether AI agents can automate this design process and introduce EvalResearchBench (ERB), a benchmark for autonomous evaluation research. Given target materials, development references, candidate APIs, and fixed time and API budgets, an agent called the researcher selects or synthesizes tasks, implements graders, and revises them in pilot tests before freezing an executable evaluator for coding, co-work, and reasoning. We study 9 researchers and 13 candidate models and compare each frozen evaluator with 14 target benchmarks on score concordance and pairwise agreement. The best evaluators order about 75\% of candidate pairs as the targets do, below the 91\% ceiling set by disagreements among the targets. No researcher leads on every metric, and the best evaluator on development targets is not the best on sealed targets hidden from the researcher. A human-designed sample of public tasks remains a strong baseline, and the evaluator with the lowest recorded execution cost attains the highest pairwise agreement. Agents repair tasks and graders through pilot feedback, yet their evaluators can still truncate answers, exhaust the evaluation budget, or let a few questions dominate a domain score. 

---
# Language Model Activations Inhabit Privileged Error-Correcting Basins 

**Authors**: Matthew Finlayson, Francisco Pernice, Eric Todd, Amir Zur, Daniel Wurgaft, Fenil R. Doshi, Vasudev Shyam, Matt Feiszli, Satchel Grant, Lucius Bushnaq, Tal Haklay, Usha Bhalla, Matthew Kowal, Thomas Fel, Jack Merullo, Atticus Geiger, Xiang Ren, Owen Lewis, Ekdeep Singh Lubana  

**Link**: [PDF](https://arxiv.org/pdf/2610.04183)  

**Abstract**: Language models exhibit remarkable robustness, continuing to produce coherent text even when their activations are perturbed by interventions like linear steering. We hypothesize that this robustness is a result of passive dynamics, i.e., constraining mechanisms in the forward pass that funnel activations toward "good" regions that produce coherent outputs. To investigate these hypothesized error-correcting mechanisms, we probe the geometry of language model activation space by observing the action of model layers on low-dimensional curves. In doing so, we discover that model activations occur within a cluster of distinct attracting basins, which differentiate natural activations geometrically from distributionally similar synthetic activations. Applying this lens to language model steering, we observe feature-specific basins along semantic steering directions, and find that steering moves activations between these basins. To demonstrate the active role of this geometry in neural computation, we show that adaptively modulating steering strength to transport activations across basins improves inter-language steering, significantly increasing the probability of sampling tokens from the target language compared to fixed-strength steering. Our findings establish analysis of activation space geometry as a promising approach to interpreting and controlling language models. 

---
# SHarP: Saliency-based Pruning of Agent Harnesses 

**Authors**: Xinyi Gao, Qiucheng Wu, Kaizhi Qian, Handong Zhao, Shiyu Chang, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04178)  

**Abstract**: Agent harnesses are systems that coordinate model calls, tool use, and task execution to help large language models complete complex tasks. To meet task requirements and address failures, these systems are often iteratively refined by amending and patching their instructions, tools, and workflows, continuously increasing harness complexity. It is therefore unclear whether some resulting harness modules are redundant, introducing substantial token overhead with little, if any, performance gain. Inspired by neural network pruning, in this paper, we study harness pruning as a means of striking a better balance between task performance and token cost. We propose SHarP (Saliency-based Harness Pruning), a simple yet effective pruning strategy based on the saliency of each harness module with respect to performance and efficiency. Specifically, we first identify tools, instructions, and supporting mechanisms as components that can be individually disabled. We then estimate the saliency of each module by ablating it and assessing its task performance and token cost relative to the full set of single-module ablations. Modules with the smallest contribution to performance or largest computational overhead are subsequently pruned. Our evaluation across various harnesses on held-out validation sets reveals a surprising finding: most harnesses that we studied are highly redundant and can maintain comparable performance and efficiency even after a substantial portion of their modules are pruned. Our pruning approach and empirical findings provide new perspectives on agent harness design and optimization. 

---
# Agentic Cognitive Depth: Operational Criteria for Evaluating LLM Agents 

**Authors**: Nijesh Upreti, Chris Sypherd, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2610.04168)  

**Abstract**: Agentic large language model (LLM) systems are commonly implemented as an LLM in a loop with Planning, Memory, Tools, and Control Flow. This application-focused view connects agentic LLM research with deployable systems and leaves open how such systems should be evaluated beyond end-to-end task success. Building on this view, we define agentic cognitive depth as a trajectory-level profile across five operational criteria. The profile contains context sensitivity ($C$), temporal continuity ($T$), multimodal coordination ($M$), adaptive interaction ($A$), and metacognitive monitoring ($Mc$). The first four criteria measure how well Control Flow, Memory, Tools, and Planning are used across a trajectory. The fifth measures whether the system monitors and regulates the full run. For each criterion, we give operational proxies and a perturbation procedure, then connect the profile to the agent's world model. We provide the structure needed to extend benchmarks such as GAIA, SWE-bench, WebArena, and TRIP-Bench with per-criterion diagnostics. Symbolic verifiers, structured memory, planner coupling, and tool constraints provide practical ways to build and test these capacities. 

---
# Auditing Pairwise Equivalence Judgments: Self-Critique Effects and Diversity Measurement in Multi-Agent Hypothesis Generation 

**Authors**: Ji Young Byun, Anthony Hu, Jesse Rogers, Roujia Wang, Manasa Kesapragada, Falgun Shah  

**Link**: [PDF](https://arxiv.org/pdf/2610.04133)  

**Abstract**: Multi-agent systems built on large language models (LLMs) are increasingly applied to scientific discovery and hypothesis generation. Both the effect of refinement and the diversity of the delivered set are hard to interpret before experimental ground truth exists, and both are typically reported by deciding whether pairs of generated hypotheses describe the same underlying mechanism. We study two evaluation questions that rest on this pairwise equivalence judgment: (1) how much self-critique changes delivered hypotheses beyond run-to-run variability, and (2) how the equivalence rule used to group hypotheses affects measured diversity. Across four proprietary instances, we hold opening hypotheses fixed, rerun the downstream workflow with 0, 1, and 5 critique rounds, and score matched hypothesis pairs with an LLM-as-a-judge. Relative to matched same-depth reruns, moving from 0 to 1 round produces 34.5 percentage points (pp) of additional mechanism-level divergence, whereas 1 to 5 rounds adds 1.3 pp. We then compare three equivalence rules: term frequency--inverse document frequency (TF--IDF) similarity, dense embeddings, and the same LLM-as-a-judge. We construct controlled hypothesis pairs that either preserve the causal explanation through wording or biological-terminology changes, or replace one component of the causal chain while holding the rest fixed. All three rules are invariant to meaning-preserving edits, but when the initiating event is replaced, the LLM-as-a-judge identifies 83% of valid pairs as different mechanisms, versus 0% for TF--IDF and 8% for embeddings; varying only the rubric that defines same mechanism moves this figure from 38% to 96%. Together, these results show that pairwise equivalence judgments are a measurement choice: how mechanism equivalence is defined affects both the estimated effect of self-critique and the measured diversity of generated hypotheses. 

---
# InvestigationWorlds: An Agentic Environment for Legal Investigation 

**Authors**: Albert Yu Sun, Andrew Benard, Sil Hamilton, Anna Teresita A. Marcelo, Yong Jae Kim, Carl-Leander Henneking, Rundong Hu, Yuhong Wang, David Mimno, Bishan Yang, Igor Labutov  

**Link**: [PDF](https://arxiv.org/pdf/2610.04129)  

**Abstract**: We introduce InvestigationWorlds, an agentic environment for legal investigation. We build on an underused artifact of U.S. civil litigation: the summary judgment motion. This motion relies upon a record composed of real evidence exhibits, and results in a court-adopted hypothesis that is treated as ground truth for the purposes of deciding the motion. Each environment is built from a real U.S. Federal Court case retrieved from Public Access to Court Electronic Records (PACER) and augmented by an attorney-validated generation pipeline that synthesizes role-tagged documents around the original record. The resulting corpus admits multiple coherent factual readings, only one of which matches the court-adopted hypothesis. Evaluating on 100 cases, we find agents often commit to incorrect hypotheses despite retrieving relevant evidence, struggling to distinguish the court-adopted hypothesis from alternative hypotheses. 

---
# CUAWright: A Minimal Unified Interface for Digital Agents 

**Authors**: Yadong Lu, Theodore Lee, Yifei Li, Lawrence Keunho Jang, Tianci Xue, Yu Su, Huan Sun, Ahmed Hassan Awadallah  

**Link**: [PDF](https://arxiv.org/pdf/2610.04116)  

**Abstract**: The prevailing approach to computer-use agents couples a model with a domain-specific harness: a browser or desktop environment equipped with human engineered tools that are fixed before task execution. As models' coding capabilities improve, the GUI native and static harness prevents them from direct programmatic operation on system state, as well as flexible construction of tools. To this end, we introduce CUAWright, a minimal terminal harness of roughly 3K lines of code that uses bash commands as its sole action interface, and a file system as its evolvable space for dynamically creating tools and managing the context. We conduct comprehensive experiments across a wide range of digital tasks, and demonstrate that by giving the agent a minimal, programmable interface, it achieves substantially stronger results compared to their GUI or hybrid CLI interface across a wide range of tasks. On OSWorld 2.0, CUAWright delivers a 33.2% relative improvement in partial reward while reducing estimated cost by 37.5% compared with the published GPT-5.5 baseline. On Online-Mind2Web and the long horizon Odysseys benchmark, CUAWright substantially outperforms GUI native harness by 4.7% and 44.0% in success rate, respectively. Furthermore, we found the gains extend to CAD applications that require accurate visual understanding and CLI interaction: on CADGenBench and BenchCAD, our unified harness yields 8.1%-41.6% relative improvements over other CLI based harnesses with GPT-5.5. Together, these results suggest digital environments are far more programmable than their GUI interfaces imply, and a minimal terminal-focused harness is the key for better performance and efficiency. 

---
# The Independence Prior of SAEs Fragments Visual Concepts 

**Authors**: Tommaso Mencattini, Giorgos Nikolaou, Donato Crisostomi, Thomas Fel, Francesco Montagna, Emanuele Rodolà, Francesco Locatello  

**Link**: [PDF](https://arxiv.org/pdf/2610.04112)  

**Abstract**: Sparse Autoencoders (SAEs) decompose model activations into sparse combinations of interpretable dictionary atoms. Although SAEs are grounded in the Linear Representation Hypothesis (LRH), their objective smuggles in an additional prior: concepts across patches are treated as independent, an assumption clearly violated by natural images and by the activations they induce. We therefore specialize LRH to vision through the Markov-Field Linear Representation Hypothesis (MFLRH), which adds the missing spatial dependencies to the LRH assumptions. We thus propose Spatial-SAE as an amortized MAP estimator under the MFLRH. Spatial-SAE consistently outperforms standard SAEs in concept recovery and interpretability. Across four variants, it achieves a 96% average win rate on synthetic concept recovery and improves interpretability on DINOv2 activations, at a reconstruction cost concentrated in high spatial frequencies. 

---
# Towards Safer Autonomous Driving in an Open World: A Dual-Process Approach 

**Authors**: Simon Janssen, Michiel Braat, Chris van der Ploeg, Serge Thill, Jan-Pieter Paardekooper  

**Link**: [PDF](https://arxiv.org/pdf/2610.04088)  

**Abstract**: Before autonomous driving systems can be deployed on public roads, it is vital that these systems comply with safety standards, traffic rules, and social norms. Although neural networks trained on large amounts of driving data perform well in routine driving tasks, these models often struggle in novel situations that are not well-represented in the data. In this work, we propose a novel framework that combines a neural network for intuitive, learning-based planning in routine driving tasks with model predictive control for reasoning-based planning in unfamiliar situations, inspired by Dual Process Theory. A meta-cognitive component is designed to switch between the two, using a knowledge graph to reason about contextual risk based on explicit perceptual information and relevant traffic rules and social norms. Contextual risk is represented through risk fields, guiding both the switching mechanism in the meta-cognitive component and compliance with safety standards, traffic rules, and social norms in the reasoning-based planner. The effectiveness of our framework is tested in CARLA for variations of a typical out-of-distribution situations involving (emergency) vehicles running a red light. We show that the novel architecture reduces the number of collisions in the scenarios by 89% and improves compliance with the special right-of-way rules, compared to the NN-only planner. 

---
# Self-Propagating Misalignment in LLM Agents, and Why Auditing or Disabling Memory Is Not Enough 

**Authors**: Debeshee Das, Jacqueline Tay, Bruce Tsai, David Huang, Javier Rando  

**Link**: [PDF](https://arxiv.org/pdf/2610.04083)  

**Abstract**: Memory poisoning attacks on LLM agents typically assume an external adversary who plants content in the agent's persistent memory to steer its behavior. We instead study, with no adversary involved, whether a misaligned agent can write a goal it cannot yet act on to persistent memory, so that a future aligned agent carries it out when the opportunity arises. We investigate this threat, which we refer to as self-propagation of misalignment, across 20 different scenarios, whose misaligned goals include self-preservation, power-seeking, undermining oversight, reward hacking, and deceiving the user. We simulate misalignment in 11 frontier models using two prompting strategies; unrestricted and values-only. The first explicitly states the misaligned goal, for instance, to prevent its own replacement, and self-propagation succeeds in 58% of runs. The second only describes what the agent cares about, for instance, that its continued operation is essential to its users, without specifying misaligned goals or directives. Even under this weaker prompt, self-propagation succeeds in 18% of runs, and every model self-propagates in at least one scenario. On removing the memory tool from the harness, we find that agents use the file system, writing the goal to a file in 74% of sessions; self-propagation still succeeds in 11% of runs. We also show that weaker models can propagate misalignment to more capable models, and that propagated goals can persist through 100 sessions of unrelated work. Existing defenses against memory poisoning and prompt injection do not directly address this threat because the memory content is generated by the agent itself, rather than injected by an external adversary. An LLM memory auditor from prior work (MemMorph) only reduces propagation from 71% to 34% of runs. We release our scenarios to support the evaluation of defenses against this emerging threat. 

---
# Reinforcement Learning with Comparative Evidence for Social Intelligence 

**Authors**: Keane Ong, Yuriel Ryan, Sabri Boughorbel, Vladimir Necula, Jack Wei Lun Shi, Rui Mao, Roy Ka-Wei Lee, Adriel Kuek, Nancy F. Chen, Erik Cambria, Gianmarco Mengaldo, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04072)  

**Abstract**: Developing socially intelligent AI remains heavily dependent on human-annotated data, limiting the scale and breadth of social understanding models can acquire. Methods that derive training signals from unlabeled data offer a path beyond this dependence, but social predictions lack the verification oracles available in mathematics and coding. Moreover, core social targets such as affect, intent, preference, and pragmatic meaning are often ambiguous. The same behavior can support multiple plausible interpretations, making it difficult to verify which is best supported. To address this challenge, we introduce Reinforcement Learning with Comparative Evidence (RLCE), a reinforcement learning method that learns social understanding from unlabeled training data without constructing rewards from ground-truth annotations. Given distinct answers in a rollout group, RLCE constructs evidence tests that identify observable evidence favoring an answer over another, validates these tests against the input sample, and aggregates test outcomes to determine the best-supported interpretation. Tests are regenerated as the policy produces new answers, enabling them to evolve with the policy. Across four benchmarks spanning affect, pragmatics, communicative intent, and preference, RLCE attains the strongest performance among seven methods that use no ground-truth training labels for rewards, including consensus, policy LLM-judge verification, multimodal co-evolution, and rubric-based rewards. Gains over the strongest baseline reach up to +18.93 points. Analyses further show that RLCE exhibits a larger share of reward variation between correct and incorrect predictions than compared rubric methods, can overturn erroneous policy-derived preferences, and benefits from pairwise test construction, compositional test aggregation, and on-policy test evolution. 

---
# Discrete Diffusion for Large Graph Generation via Structural Candidate Restriction 

**Authors**: Yassin Mohamadi, Zeno Geradts, Marcel Worring  

**Link**: [PDF](https://arxiv.org/pdf/2610.04056)  

**Abstract**: Synthesizing realistic graphs at scale is vital when the graphs of interest are large and real-world samples are limited or access-sensitive. Diffusion-based generators have recently driven much of the progress, offering high modeling capacity, but most such methods have quadratic computational complexity and are hence restricted to small-scale networks, currently up to 3k nodes. Existing non-quadratic methods remain limited by memorization issues and a trade-off between scalability and generation quality. Our goal is to generate large graphs whose structural statistics --- e.g., degree distribution, clustering, and path length --- faithfully reflect those of real-world sparse graphs, without resorting to memorizing the training data. We introduce a discrete graph diffusion model that restricts training to a structurally motivated subset of node pairs --- observed edges and their wedge non-edges --- reducing training complexity below quadratic in the number of nodes. To keep the noisy graph informative throughout both the forward and reverse trajectories, we design a three-class absorbing forward process governed by a \emph{degree-aware, floored} cosine noise schedule: unlike a structure-blind schedule that adds noise to every pair identically, ours adapts to each node's degree and never fully erases the graph's structure, keeping the noisy graph informative at every step. Experiments across diverse datasets show that our model consistently ranks among the top methods for structural fidelity against existing discrete diffusion baselines in large graph generation. 

---
# The Cost of a Hop: Benchmarking NLIP and A2A 

**Authors**: Ranjan Sinha, Anindita Das, Ashika Anand Babu, Hari Palleti  

**Link**: [PDF](https://arxiv.org/pdf/2610.04053)  

**Abstract**: Autonomous agents built on Large Language Models (LLMs) need standardized protocols to interoperate across systems. Several now exist (A2A, MCP, ACP, ANP, NLIP), but the Natural Language Interaction Protocol (NLIP) has not appeared in any controlled performance study, and no work has measured where an agent protocol's latency is spent. We compare NLIP and the Agent-to-Agent (A2A) protocol empirically, decomposing latency into message creation, connection, and send phases across three independent hardware environments. For lightweight coordination, NLIP is 8.4-9.6x faster than the baseline A2A SDK implementation on two environments and about 4x on a third; the direction of the advantage is consistent, its magnitude depends on the hardware. The advantage is stage-specific: for the end-to-end pipeline, where LLM inference dominates, the protocols are near parity. The difference comes almost entirely from connection setup. To test A2A at its best, we also ran A2A SDK with connection caching enabled; caching narrows its gap with NLIP by a hardware-dependent amount, from 2.75x on one machine to near-parity on faster hardware, where at scale a cache-optimized A2A-SDK matches NLIP. We report these as measured conditions without a single causal account of the residual send-phase cost. Against the more optimized Python-A2A, NLIP leads by about 4x on the same stage. We close with a protocol-selection guide keyed to workload characteristics. 

---
# Agent Policy-Value Audit: Separating Transition Composition from Event Selection in Financial LLM Agents 

**Authors**: Mingyang, Chen, Yida, Huiwen, Chen, Yiming Lu, Wei Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.04040)  

**Abstract**: Financial LLM agents are often evaluated by comparing their end-to-end returns with those of a baseline and testing the paired difference against zero. This measures whether deploying the agent changes realized performance, but it does not isolate event-selection skill. An agent that frequently changes positions from flat to long can earn a positive paired return from an upward-drifting event pool even when it selects events at random. We propose the Agent Policy-Value Audit, which holds fixed the observed count of each ordered action-change type and randomly reassigns them across eligible events. The average payoff from these reassignments is the composition benchmark; the difference between observed deployment value and this benchmark is selection value. In semi-synthetic benchmarks based on real earnings-event returns, a zero-centered paired test falsely attributes passive exposure to selection skill in $11.6\%$ of no-skill replications, while the transition-matched audit reduces this rate to $5.3\%$. Applied retrospectively to 723 earnings events at 44 U.S. consumer-facing firms, the audit decomposes the agent's gross deployment value of $+15.2$ bps/event into a $+25.8$ composition benchmark and a $-10.6$ selection value. The agent does not detectably outperform matched random assignments. Financial-agent evaluations should report deployment value separately from event-selection value. 

---
# The Reported Engagement with AI Level (REAL) Rating: A Framework for Disclosing Human-AI Collaboration 

**Authors**: Imène Goumiri, Mayleen Cortez-Rodriguez, Eric Bell, Amanda Muyskens  

**Link**: [PDF](https://arxiv.org/pdf/2610.04021)  

**Abstract**: Generative artificial intelligence has become increasingly incorporated into digital media and more generally into production workflows with which the public frequently interacts. Current provenance standards and disclosure methods frequently rely on binary categorizations, differentiating only between entirely human-authored and AI-generated content, or depend on technical watermarking that lacks user-facing clarity. However, there are very different risks and outcomes depending on the different uses of AI in the creation and consumption of end products. Therefore, we propose the Reported Engagement with AI Level (REAL) Rating framework which addresses these concerns by introducing a six-tier scale (Levels 0-5) that discloses the extent of AI involvement in both product creation and user experience. This paper details the structural methodology of the REAL Rating system and the specifics of its application across text, image, audio, video, and software products. 

---
# A Quantitative Analysis of Graph Representation Strategies for Cyber Attack Detection 

**Authors**: Ali Melih Kanca, Ilker Turker  

**Link**: [PDF](https://arxiv.org/pdf/2610.04019)  

**Abstract**: Graph based cyber attack detection studies employ various graph construction and representation strategies across different cybersecurity application domains. This diversity motivates a quantitative examination of how representation strategies are distributed across these application domains. This study presents a quantitative analysis of 37 original studies published between 2019 and 2026. Each study was coded according to publication year, application domain, graph representation type, feature extraction strategy, learning paradigm, algorithm, and dataset. Frequency analysis, cross tabulation, and statistical association tests were applied. Automated representation learning was the most frequently employed strategy, accounting for 73.0% of the studies, while handcrafted representation strategies accounted for 27.0%. The Fisher Freeman Halton exact test revealed a statistically significant association between application domain and representation strategy (exact p = 0.008; Cramer's V = 0.540). The findings indicate that automated representation learning predominates within the analyzed corpus and that the distribution of representation strategies differs across cybersecurity application domains. 

---
# Beyond the Parameter Monolith: Reconstructive Memories, Executable Skills, and Residual Assembly for Language Models 

**Authors**: A. Bochkov  

**Link**: [PDF](https://arxiv.org/pdf/2610.04012)  

**Abstract**: Language-model systems can separate contextual computation, persistent storage, and exact execution instead of updating all capabilities through one shared parameter system. We investigate FEM-ASM, a finite-element-method-inspired organization in which independently constructed document states and deterministic executable skills contribute typed proposals to a shared language-model state. An explicit residual operator reconciles proposals attached to common interface nodes. We evaluate this organization through controlled experiments and negative results rather than claiming a physical finite-element formulation of language. An attention-free Multi-Mesh prototype learns causal language modeling but does not establish competitive general capability. A versioned store contains 52,809 reconstructive memory elements near a 1.7-billion-floating-value budget; reconstruction is incomplete, with approximately 75\% token accuracy. Support-aware lexical indices make these elements addressable under provenance-controlled query construction. For executable arithmetic, positional result observations substantially improve neural rendering relative to a repeated global result vector, and output substitutions change the model's preferred answer. A bounded attachment demonstration further measures the effect of making selected evidence available, without establishing the utility of loading an entire multi-billion-value store. The results support a separation of storage, execution, and neural coordination, while identifying unresolved limitations in question-only retrieval, unrestricted answer generation, and end-to-end efficiency. 

---
# Exploration-Preserving Policy Optimization 

**Authors**: Hangzhan jin, Mohammad Hamdaqa, Doina Precup  

**Link**: [PDF](https://arxiv.org/pdf/2610.04011)  

**Abstract**: Reinforcement learning with verifiable rewards improves reasoning, while the allocation of learning signal shapes which solutions remain accessible under repeated sampling. Group-relative objectives assign equal advantages to equally rewarded responses, making aggregate credit proportional to sampled mode frequency. We introduce Exploration-Preserving Policy Optimization (ExPPO), a lightweight advantage-shaping rule that redistributes credit using prompt-relative, length-normalized response surprisal and prompt pass rate. ExPPO combines bounded shaping with shared normalization to preserve verifier polarity and approximately maintain each prompt group's total absolute sequence-advantage mass. Our analysis characterizes response-level credit allocation alongside sampled mode updates, deriving local conditions for gains in entropy and correct-mode discovery. Experiments show improved in-domain and out-of-domain reasoning coverage, higher aggregate response accuracy, and strong coverage at large sampling budgets. A controlled multi-answer evaluation further demonstrates increased correct-mode yield and gains in diversity among verified-correct responses. Code is available at this https URL 

---
# SkillScriptBench: Benchmarking Self-Evolution of Executable Agent Skill Packages Beyond Markdown 

**Authors**: Yuxuan Liu, Haoran Li, Yuhao Zhang, Jiahe Guo, Hongyu Luo, Wenbin Hu, Huihao Jing, Kawai Chung, Junle Chen, Changxuan Fan, Qing Zong, Lingyun Xie, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.04008)  

**Abstract**: Executable Agent Skills combine natural-language instructions and scripts into reusable packages for LLM agents, and revising them requires fixing errors without breaking correct behavior. Existing benchmarks do not systematically distinguish documentation repair, script repair, and preservation when evaluating skill self-evolution. We introduce SkillScriptBench, a 350-task benchmark designed to evaluate these capabilities separately. From a survey of over 35,000 GitHub-hosted Skill roots, we select 100 packages and construct 150 repair tasks. Each task pairs a package containing injected script faults with a maintenance request and executable checks of the required behavior. A complementary controlled track contains 200 tasks from 50 packages, each evaluated under the same maintenance request in four states: clean, documentation faults, script faults, and faults in both. Across four LLMs, methods that edit both documentation and scripts can repair script faults but do not consistently outperform Markdown-only revision on documentation repair or preservation. We therefore introduce AST-Guided Skill Revision, which uses abstract syntax trees and calling relationships to link maintenance requirements to relevant code locations. It restricts script edits to these locations and updates the documentation to match the revised scripts. Averaged across models, this revision stage yields absolute gains in repair success of 21.9% for Raw Package and 27.7% for CoEvoSkills on faulty packages. Absolute gains in the proportion of tasks solved in all three runs reach 20.8% and 31.5%, respectively, indicating more consistent repair success across repeated runs. 

---
# Behavioral History Outperforms Descriptions of the Person for LLM Synthetic Personas 

**Authors**: Khashayar Pourtaheri, Ahmad Zareei  

**Link**: [PDF](https://arxiv.org/pdf/2610.03998)  

**Abstract**: Large language models (LLMs) are increasingly used as synthetic personas representing survey respondents. Their validity as substitutes for particular respondents depends on whether they reproduce individuals' decisions. We examine what information helps synthetic respondents predict each individual's later choices, using five conditions that add progressively richer information: no personal information, demographics, personality traits, cognitive scores, and finally the respondent's earlier survey choices as behavioral history. We use a two-wave panel of 845 US adults who completed measures of 14 behavioral biases (spanning risk, time preferences, overconfidence, and reasoning), so each respondent's earlier answers provide a human test-retest benchmark; in the behavioral-history condition, all items that score the target bias are withheld. At the population level, the average number of biases per respondent in every condition is close to the human average (7.1-8.1 biases, against 7.1 for humans). This aggregate similarity masks differences in variance: persona descriptions recover only 53-67% of human between-person variation, whereas adding behavioral history restores it to approximately the human level. At the individual level, description-based personas achieve only 7-12% of the informedness observed in human test-retest responses, while adding behavioral history raises this to 28%. The condition including behavioral history has the highest estimated informedness in all 17 demographic groups, whereas description-based conditions provide little or no information for some groups. Synthetic responses also exhibit stronger education- and income-related differences than human responses. For LLM synthetic personas, a respondent's past answers add more to individual-level prediction than a description of who they are. 

---
# Teaching Agents to Code Reliably 

**Authors**: Muhammad Ahmed Mohsin, Myeongsoo Kim, Kangrui Ruan, Shweta Garg, Varun Kumar, Murali Krishna Ramanathan  

**Link**: [PDF](https://arxiv.org/pdf/2610.03984)  

**Abstract**: Autonomous coding agents solve repository issues by reading code, running commands, editing files, and submitting patches. Extra inference-time compute yields gains only when it produces a useful repair and supplies reliable evidence for choosing one. Three behaviors decide both, and we argue they are teachable rather than byproducts of scale, so a policy can carry them instead of a scaffold. Location diversity remains narrow, since attempts return to the same site and extra samples add no coverage. Edit diversity is left unexploited, since methodologies that differ resolve complementary issues no single run reaches. Verification misleads, since a test the agent writes for its own patch accepts many incorrect ones. Directing search by execution feedback and scoring each patch against its own reverted tree resolves 52.8% of SWE-bench Verified using 48.1% of the agent-steps an eight-sample baseline spends. Training moves these behaviors into the policy. On the 270 issues held out from SFT and RL training, weighted supervised fine-tuning raises pass@1 from 31.9% to 35.2% and pass@8 from 46.7% to 51.1%. A reinforcement objective then trains the verifier against gold-labeled repairs and incorrect variants, crediting the assertions that detect them. It raises pass@1 to 43.0% and pass@8 to 60.7%, lifts verifier precision from 26.8% to 41.7%, and more than halves false acceptance. Resolution improves on two of three out-of-distribution suites and verifier precision on all three, and the gains hold at 7B, 14B, and 30B against published coder baselines. 

---
# ROAR: Unifying Runs across Heterogeneous AI-Driven Research Systems 

**Authors**: Leo Y. Lin, Vishakha Ramani, Z. Berkay Celik, Paul Castro, Marquita Ellis  

**Link**: [PDF](https://arxiv.org/pdf/2610.03966)  

**Abstract**: Each run of an AI-driven research system (ADRS) is an expensive search over a vast solution space, and dependable evaluation requires many runs, making run data both costly to produce and valuable to retain for large-scale analysis. Yet this data remains fragmented: teams operate in isolation, ADRS frameworks emit results in different formats, and no shared infrastructure exists to aggregate or compare runs across problems and systems. We present ROAR, a solution for systematically unifying and analyzing heterogeneous ADRS outputs. ROAR addresses two challenges: reconciling heterogeneous ADRS outputs and enabling analytics across runs with different objectives and scoring functions. We achieve this through a relational schema and parsing layer that normalize heterogeneous ADRS outputs while preserving data lineage and temporal structure, and accommodating new systems without requiring schema modifications. From building a corpus of more than 900 runs from multiple ADRS, we show how pooled data can reveal properties of problem landscapes that are difficult to observe. Consistent with prior work, runs with identical configurations may converge to different scores. We find that many runs realize most gains early, and that the effectiveness of different strategies for incorporating prior solutions into the search process varies across problems. We further show that the pooled corpus is actionable and not merely analytical by using ROAR to configure ADRS runs. Together, these results illustrate how pooled ADRS data can expose problem-dependent structure in search behavior that is difficult to detect from any single system, team, or benchmark. Such cross-cutting insights are difficult to obtain while runs remain siloed; ROAR is the first infrastructure designed to unify them. 

---
# LatentQuant: Preserving the Policy-Facing Latent Contract under NVFP4 VAE Quantization 

**Authors**: Ziye Deng, Lufang Chen, Shuyu Feng, Zhenwei Duan, Zicong Ye, Yu Sun, Xiaofan Li, Ruyi Gan, Hao Wang, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.03959)  

**Abstract**: Recent world action models (WAMs) reuse pretrained video VAEs whose encoder latents directly condition downstream action policies. Quantization must therefore preserve not only reconstruction fidelity but also the policy-facing latent contract expected by the frozen policy. Direct NVFP4 leaves W4A4 quantization error uncompensated, whereas joint quantization-aware training (QAT) can recover reconstruction by moving this representation. On Wan2.1, joint QAT nearly matches FP32 VBench-7 (0.7403 versus 0.7409), yet LIBERO success collapses from 95.5% to 10.5%. Controlled decoder-only experiments show that activation quantize-dequantize operations alter the reconstruction signal and decoder Jacobian, redirecting the gradient returned to the encoder and inducing persistent latent drift. Based on this mechanism, we introduce LatentQuant, a two-stage NVFP4 QAT framework that first aligns the quantized encoder with its high-precision counterpart, then freezes it while adapting the decoder. Across Wan2.1 and Wan2.2, LatentQuant preserves near-baseline control and high reconstruction quality, achieving 95.75% success on LIBERO and 68.8% on RoboTwin. On NVIDIA B300 GPUs, NVFP4 execution achieves 1.17x-1.26x end-to-end VAE speedups over BF16 cuDNN. 

---
# Retrieval-Augmented Large Language Model Decision-Making for Autonomous Driving Guided by Chinese Philosophical Wisdom 

**Authors**: Xiaojun Bi, Xiaoyuan Ma, Yiwen Sun, Tianren Huang, Chaoran Liu, Bokai Huang, Hao Yang, Baichuan Mo  

**Link**: [PDF](https://arxiv.org/pdf/2610.03948)  

**Abstract**: Autonomous driving decision systems must balance safety, efficiency, and social norms in complex traffic interactions. Philosophical and ethical considerations have received limited attention in existing autonomous driving decision-making approaches based on numerical optimization, sequence prediction, and large language models (LLMs). We propose Chinese Philosophical Wisdom-Guided Driving (CPW-Drive), a closed-loop retrieval-augmented generation (RAG) framework that incorporates value guidance derived from Chinese philosophy into autonomous driving decision-making. Using Chinese Confucian thought as its knowledge source, CPW-Drive consolidates LLM-extracted keywords from relevant classical texts into driving-relevant value principles through manual screening and validation. It then contextualizes these principles through scenario-specific cases to form retrievable and reusable value guidance. We further propose Physics-aware Spatial Similarity Retrieval (PSSR), which compares vehicle layouts and velocity-extrapolated states to retrieve physically relevant historical cases. On Highway-env's multilane highway-driving task, CPW-Drive achieves success rates of 93.0%, 86.0%, and 72.0% across three traffic configurations. These results outperform the strongest baseline by 8.0, 22.5, and 25.0 percentage points, respectively. Across all configurations, CPW-Drive achieves the highest collision-free step count and maintains a low lane-change frequency. The results suggest that structured value guidance can improve simulated closed-loop safety and stability while introducing efficiency and latency trade-offs. 

---
# MLLMs Fail to Refuse when Using Tools Agentically 

**Authors**: Rikiya Takehi, Ryo Hachiuma, Shaona Ghosh, Dan Zhao, Yu-Chiang Frank Wang, Yusuke Hirota  

**Link**: [PDF](https://arxiv.org/pdf/2610.03938)  

**Abstract**: Agentic multimodal large language models (MLLMs) have recently pushed the frontier of visual reasoning by calling tools such as zooming and tagging. Despite the recent strong success of agentic MLLMs, this work uncovers a critical safety failure in the tool-use paradigm: agentic tool-using MLLMs become less capable of refusing harmful requests. Our experiments confirm that, across three popular safety benchmarks, all the top open- and closed-weight MLLMs we test exhibit significantly lower safety in tool-using settings than in non-tool settings, with a relative refusal failure rate increase of up to 68.7%. Based on analysis of 100,000+ responses, including extended experiments, we also propose two possible reasons for this safety degradation. 

---
# SGAnalog: An End-to-End Circuit Benchmark from Open-Source Silicon Tapeouts 

**Authors**: Yueting Li, Weihang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2610.03934)  

**Abstract**: Existing analog integrated circuit design benchmarks make two questions hard to answer: whether a model has learned transferable circuit skills rather than recalled familiar examples, and whether its output works under defined process and test conditions. We introduce a benchmark built from human-designed, open-source circuits associated with Tiny Tapeout manufacturing shuttles. The collection contains 273 topologically distinct top-level designs. Every source is retrieved at the revision recorded for its shuttle submission and processed in a fixed containerized environment. The pipeline exports each eligible schematic image and its SPICE netlist from the same source file, giving transcription an exact structural reference. Commit dates support model-specific training-cutoff analysis, while author testbenches provide the simulation context for sizing. The benchmark evaluates schematic-to-netlist transcription and device sizing. Across seven models on a fixed set of 66 transcription tasks, the strongest model reaches 56.1% exact graph isomorphism, and six of seven models drop sharply from the small to the medium tier. For one frontier model, removing author-chosen labels reduces exact matches while preserving aggregate structural F1, suggesting that labels can aid connectivity tracing. On the 17 sizing tasks, the leading model converges on all 17 proposals and reaches 91.2 out of 100 against the human reference, while the two newest Claude models refuse 4 and 11 of the same prompts they transcribe without objection; a proposal without sizes scores zero. The two tasks produce different model rankings, exposing distinct visual and design capabilities and, in one family, a policy rather than capability limit. 

---
# Proxy Confidence: Auditing Black-Box LLM Agents with a Surrogate's Log-Probabilities 

**Authors**: Yikai Zhao, Saurabh Pandey, Pradeep Kumar Misra  

**Link**: [PDF](https://arxiv.org/pdf/2610.03894)  

**Abstract**: A deployed LLM agent emits tool calls, queries, and code that can be silently wrong -- by the time the error surfaces, the action has run. Frontier chat APIs hide the model's token probabilities; the agent's stated confidence barely beats chance on the mistakes that matter; and resampling does not help, since frontier models are highly repetitive, reproducing the same call across samples.
We recover the missing signal from a low-cost open-weight surrogate run in parallel. It reads the same context, schema, and proposed action as the agent, then scores the call from its own log-probabilities through a family of complementary readouts: teacher forcing and request-PMI weigh the likelihood of each argument value, a discriminative verdict judges the call as a whole, and tool-choice competition tests the function against its siblings. One principle says which to trust: a generative likelihood localizes wrong argument values, while the verdict catches holistically wrong calls. When the error type is unknown, an ensemble is the low-regret default. The readout is training-free, needs no access to the agent's internals, and costs one prefill pass alongside the tool call.
On difficult coding tasks it reaches AUROC 0.825 where the actor's stated confidence is near chance (0.598), and the generative readouts beat it by +0.07 to +0.28 across three further actors. Against self-consistency it gains +0.14 to +0.19 on near-deterministic actors, at 1/K the cost. The signal drives two deployment modes: a real-time gate escalating the least-trustworthy calls for review (+0.05 to +0.30 accepted-action accuracy at 50% coverage), and confidence feedback, returning the tool result with the score so the agent adapts its next step -- lifting task success on live-execution benchmarks (+0.119 and +0.137, p <= 1e-4) and beating a random-value control where step errors are silent (+0.078, p = 0.003). 

---
# REACT: Physically and Chemically Consistent Reconstruction of Marine Active Tracers 

**Authors**: Wenbin Dai, Hao Zheng, Shiyu Liang, Chaofan Sun, Xueying Zhang, Hanbo Huang, Xuan Gong, Yiran Zhang, Enhui Liao  

**Link**: [PDF](https://arxiv.org/pdf/2610.03888)  

**Abstract**: Reconstructing global sea surface pH from sparse observations is critical for monitoring ocean acidification and understanding marine carbon cycling. Traditional assimilation and inverse models are physically grounded but costly for large-scale reconstruction. Recent black-box and physics-guided AI models improve efficiency, but are mainly designed for passive tracers, where the reconstructed variable is also the transported inventory. In contrast, pH is an active carbonate tracer: it is the prediction target, while dissolved inorganic carbon (DIC) is the conserved carbon inventory. This mismatch can produce low pH error while violating carbonate closure and source-free carbon conservation. To address this, we introduce \textbf{REACT}, a carbon-first reconstruction framework that decouples transport, active correction, and chemical decoding. REACT transports a latent carbonate state with a conservative advection--diffusion solver, captures non-conservative carbon-cycle variations with a source module, decodes the corrected state into pH, and constrains the output through carbonate equilibrium. This design keeps pH as the target while enforcing consistency on the underlying carbon state. On simulation data, REACT reduces pH NRMSE by (14.7%) and chemical consistency error by (24.0%) over the best baseline. Cross-temporal-scale evaluations show robustness against error accumulation from coarse to fine temporal scales, and ablation studies validate the effectiveness of each component. 

---
# Training Numerical Intelligence via Auto-Diagnosis and Skill Discovery 

**Authors**: Peter Chen, Wotao Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.03872)  

**Abstract**: AI agents are becoming increasingly capable of generating scientific code, but generating code is not the same as improving the algorithms behind it. For numerical solvers, execution feedback can expose poor performance, but rarely reveals its underlying cause and how to address it. We introduce Auto-Diagnosis and Skill Discovery (ADSD), a framework that links numerical diagnosis to reusable solver self-improvement. ADSD follows a diagnosis-first paradigm that first explains why a solver performs poorly, then uses this diagnosis to guide the discovery of appropriate numerical methods. The resulting knowledge is packaged into reusable solver skills, turning solver improvement from trial-and-error editing into a structured process of diagnosis, discovery, and implementation. Across four challenging numerical domains--power flow equation, AC optimal power flow control, stiff ordinary differential equations, and heterogeneous diffusion PDEs--ADSD consistently improves solver accuracy, robustness, and efficiency. On GOC-500 power flow, for example, ADSD reduces mean solver error by nearly $71\times$, with improvements further transferring to unseen grid topologies and operating regimes. 

---
# One Figure, Every Canvas: Editable Flowchart Relayout via Agentic Pipeline 

**Authors**: Shih-Chen Tseng, Chih-Hsuan Chen, Ryan Yang, Hsi-An Chen, Chun-Wei Tuan Mu, Yu-Lun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06852)  

**Abstract**: Pipeline figures in ML papers must be repurposed across many canvases, including paper columns, 16:9 slides, portrait posters, 1:1 social teasers, 9:16 phone previews. Each format imposes a different aspect ratio on the same computational graph, where any silently broken connection misrepresents the method. We formulate aspect-ratio-adaptive flowchart relayout as a distinct task: given a raster flowchart and a target ratio, produce a structurally faithful, hallucination-free, editable layout. Existing methods fail characteristically: image-to-image models stretch blocks and reject extreme ratios, text-to-image agentic systems hallucinate content, and parse-then-render systems mis-route edges. We propose an agentic pipeline factored into Parse, Style, and Layout stages, each pairing a main agent with a critic that combines deterministic constraint checks with VLM visual feedback so connectivity is explicitly checked and prevented from being silently broken. Outputs are this http URL-editable mxGraph XML. On a curated benchmark of 100 flowcharts at five aspect ratios, evaluated by Gemini 3.1 Pro and validated against human judgments, our method reaches 68.6% Content Fidelity versus 11.2-41.4% for prior work. Project page: this https URL 

---
# Base Models Can Reason By Taking a Cue From Training Data 

**Authors**: Sophie L. Wang, Amil Dravid, Rulin Shao, Kevin Farhat, Sewon Min, Alexei A. Efros  

**Link**: [PDF](https://arxiv.org/pdf/2610.06851)  

**Abstract**: In this paper, we study how training data creates associations between the tokens at the start of a base model's response and the reasoning behavior that follows. First, we demonstrate that fixing particular starting token cues makes a base model's performance competitive with that of its reinforcement learning (RL)-trained counterparts on math and coding. For instance, the cue ".\n\nOkay" raises Olmo-3-7B's MATH-500 pass@1 accuracy from 42% to 78%, while "Alright," raises Qwen3-14B's from 72% to 87%. Second, RL makes these cues more likely, while fixing them recovers much of its performance gain over the base model. Third, we trace the reasoning effects of token cues to the training data. We perform causal data interventions to turn an arbitrary word, such as "chicken", into an effective reasoning cue, or remove an existing cue's effect. A similar edit makes the prompt instruction "Think duck duck goose" as effective as "Think step by step" at eliciting reasoning. We also find that the hidden state representations induced by different cues correlate with different document types from the training set. Finally, we extend our study of token cues with a case study in language model safety, finding that different cues elicit distinct refusal and compliance behaviors that correspond to different types of training data. 

---
# Learning to Read the Contextual Tokens in Diffusion Transformers 

**Authors**: Omer Dahary, Etai Sella, Hadar Averbuch-Elor, Daniel Cohen-Or, Or Patashnik  

**Link**: [PDF](https://arxiv.org/pdf/2610.06844)  

**Abstract**: Multimodal Diffusion Transformers (MM-DiTs) jointly process visual and textual representations throughout generation. These models repeatedly update the text tokens through multimodal attention, forming dynamic contextual tokens whose function is not well understood. In this work, we introduce a framework for reading this contextual space through natural-language interrogation. We train a lightweight bottleneck network that maps intermediate contextual tokens into the input space of a frozen Large Language Model (LLM), allowing the LLM to answer questions about the emerging image directly from these hidden representations. Our reader reveals that contextual tokens encode a rich, global representation of the emerging scene: generation-specific semantics, including attributes left underspecified by the prompt, are accessible surprisingly early in denoising, while increasingly fine-grained details become readable over time. Remarkably, this information remains decodable even when the MM-DiT receives an empty prompt, showing that contextual tokens accumulate substantial image-specific information from the evolving visual representation itself. We further find that generations with more readable contextual representations tend to receive higher human-preference scores. Building on these observations, we introduce Contextual Alignment, a training technique that explicitly reinforces the visual-semantic information encoded in the contextual tokens, improving generation quality and distributional coverage. Together, our results establish contextual tokens as both an interpretable view into the internal dynamics of MM-DiTs and an effective target for improving generative models. 

---
# Recursive Video In-Context Learning for Agentic Robot 

**Authors**: Wenrui Bao, Xinxin Liu, Bingxin Xu, Yuzhang Shang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06843)  

**Abstract**: LLM agents that orchestrate frozen vision-language-action (VLA) policies improve across episodes through text memory, which records what the agent did but not how the task is done. A demonstration video shows it, but fits poorly into an agent's context. The full video slows every turn, fixed keyframes lose the contact detail that decides whether a grasp holds, and what the agent needs shifts from the task's structure while planning to the frames around each contact. We introduce Recursive Video In-Context Learning (RV-ICL), a training-free method that turns a demonstration into a hierarchy the agent navigates rather than a prompt it receives. The hierarchy is built from the sub-events of the demonstration, such as grasps and releases. Its levels grow finer, from keyframes of the whole task to phases, moments and short clips, and are exposed through read-only tools. The agent reads the coarse levels before planning. During execution it re-enters the hierarchy whenever a step needs more detail and loads only the clip of its current sub-goal. One demonstration per task is enough. Built on RPent, RV-ICL raises success from 92.6% to 96.5% on LIBERO-PRO and from 86.7% to 95.8% on LIBERO-Plus. 

---
# UniSlider: Perceptually Uniform Sliders for Continuous Image Editing 

**Authors**: David Serrano-Lozano, Duygu Ceylan, Yannick Hold-Geoffroy, Iliyan Georgiev, Javier Vazquez-Corral, Anna Frühstück  

**Link**: [PDF](https://arxiv.org/pdf/2610.06831)  

**Abstract**: Sliders provide an intuitive interface for continuous image editing. In current generative approaches, however, the slider is simply a rescaling of the method's strength parameter, such as an adapter coefficient, a prompt weight, or an interpolation factor. This strength relates poorly to perceptual change. The image can partially revert as the slider moves, long stretches of the range produce no visible difference, and short intervals transform the image abruptly. Remapping the strength could fix this uneven pace, but only if the trajectory is monotone, which current methods do not enforce. We therefore distinguish the slider from the strength, and require perceptual distance from the input to grow linearly with the slider value. We introduce UniSlider, a lightweight LoRA trained on a few-step editing backbone so that its strength approximates this ideal slider. Few-step sampling lets us impose this objective in pixel space without intermediate ground truth, and the backbone's output is preserved at full strength. However, a low-rank adapter cannot make the strength fully uniform. Our slider is thus an inference-time remapping of the strength, obtained by adaptive sampling. Since training optmizes to make the trajectory monotone, this remapping closes the remaining gap without extra training or parameters. On a new benchmark of 300 continuous edits evaluating uniformity, monotonicity, edit fidelity, and identity preservation, UniSlider outperforms all prior methods and is preferred in a user study. 

---
# MemPilot: Orchestrating On-Demand Multimodal Memory Curation for LLM Agents 

**Authors**: Haozhen Zhang, Haodong Yue, Quanyu Long, Jianzhu Bao, Qingyuan Liu, Tao Feng, Bohan Liu, Weida Liang, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06830)  

**Abstract**: Memory has become integral to the LLM agent ecosystem, supporting information retention and reuse across interactions. However, most existing agent memory systems construct memory in a query-agnostic manner, which can incur unnecessary preprocessing cost and discard details that later prove essential. Recent studies have begun shifting memory processing toward runtime adaptation, but typically specialize in particular operations or fixed processing schemes, leaving flexible control over performance, cost, and latency largely underexplored. To address this challenge, we present \textbf{MemPilot}, a flexible framework that orchestrates on-demand memory curation under different performance--cost--latency preferences. Specifically, we optimize a multi-step LLM policy via reinforcement learning to iteratively choose between retrieving from query-agnostic memory and delegating query-specific curation of raw multimodal history to heterogeneous LLMs and VLMs. The policy jointly controls evidence amount, curation instructions, model selection, and visual access, enabling fine-grained allocation of runtime computation. To optimize this policy under competing objectives, we adapt objective-wise advantage decoupling by separately estimating each objective's advantage before aggregation. Moreover, we introduce prefix-based marginal utility estimation for fine-grained credit assignment across multi-step rollouts. Experiments on five multimodal agent-memory benchmarks demonstrate favorable performance--cost--latency trade-offs across optimization preferences, with preference sweeps yielding broader frontiers than existing trade-off-aware baselines. 

---
# CLIFT: Conformal Self-Verification for Web Agent Training and Test-Time Scaling 

**Authors**: Yifan Zhang, Yutong Dai, Viraj Prabhu, Zhiyuan Hu, Ran Xu, Zeyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.06829)  

**Abstract**: Open-source web agents are now strong enough to execute realistic browser tasks, but training them with reinforcement learning still depends on weak supervision: binary task success is too sparse for credit assignment, while frontier-language-model judges are too expensive to call at every step and cannot be assumed available at deployment. We introduce CLIFT, a training and test-time scaling method built around conformal self-verification. During training, the agent answers natural-language verification questions about its own rollouts; a Compositional Conformal Certifier keeps only question signals whose URL-conditional evidence agrees with a training-time judge, assigns signed trust weights through polarity-aware lift, and blends the resulting verifier score into per-step rewards in a way that never subtracts from the judge baseline. At test time, the same certified bank is frozen and reused as structured evidence for Conformal Trajectory Selection (CTS): the agent samples a greedy rollout and one or more diverse retries, the self-verifier summarises each URL trace, and a conservative majority-vote rule chooses whether to swap away from the current incumbent without calling any external judge. This single mechanism supports three settings. On WebArena Infinity, CLIFT achieves state-of-the-art performance among open-source web agents. On VisualWebArena, a bank trained with the open model transfers to GPT-5.5 at test time and reaches state-of-the-art performance under the canonical harness. On Online Mind2Web, without training an agent on the benchmark, translating the certified question bank improves a live-web agent in zero-shot evaluation. Together these results position conformal self-verification as a way to turn costly judge feedback into a reusable training signal and a judge-free test-time scaling signal. 

---
# Deep Learning for Sleep Heart Rate Estimation from Accelerometers: Toward Population-Scale Cardiac Insight Without Optical Sensors 

**Authors**: Tanbin Islam Rohan, Pranjol Sen Gupta, Tanusree Debi, Nazmus Sakib  

**Link**: [PDF](https://arxiv.org/pdf/2610.06823)  

**Abstract**: Large longitudinal cohorts often contain wrist accelerometry without optical heart-rate sensing, motivating recovery of cardiac information from motion signals already collected during sleep. We present SeqSmoother, a transformer-based temporal corrector for sleep heart rate (HR) estimation from wrist accelerometry. SeqSmoother combines spectral descriptors with an intermediate Nightbeat-derived frequency anchor and a physics-motivated sub-harmonic feature designed to identify harmonic frequency lock-on. All inference-time features are derived from wrist accelerometry, while ECG is used only to construct reference HR labels and training-label quality weights. We evaluate SeqSmoother using 13 participant-disjoint held-out folds and compare it with the official Nightbeat implementation under a matched 60-s window and 15-s step protocol. Across all out-of-fold predictions, SeqSmoother achieved a participant-macro MAE of 1.60 bpm. On Nightbeat-retained matched intervals, Nightbeat achieved lower absolute error than SeqSmoother (0.615 versus 1.091 bpm), while SeqSmoother provided estimates over a larger portion of the eligible recording; Nightbeat produced final estimates for 72.85% of the SeqSmoother-eligible out-of-fold grid. Separately, the proposed sub-harmonic ratio achieved an AUROC of 0.972 for identifying reference-defined harmonic lock-on candidates. These findings reveal an accuracy-availability trade-off between learned temporal modeling and quality-gated signal processing while providing empirical support for a physics-informed approach to identifying frequency-tracking failures in accelerometer-based sleep HR estimation. 

---
# Paradee: Distilling Kokoro-82M into an 8M-Parameter Single-Voice Text-to-Speech Model 

**Authors**: Sahil Mahendrakar  

**Link**: [PDF](https://arxiv.org/pdf/2610.06817)  

**Abstract**: We distill Kokoro-82M, a widely used open text-to-speech model with 54 voices, into Paradee, an 8.07M-parameter model that speaks one of them. Paradee keeps Kokoro's architecture with much narrower layers, and each of its two halves is trained separately against the frozen teacher. It has 10x fewer parameters and needs 15x less compute. We first synthesize a corpus with the teacher and keep its durations, pitch, energy and phoneme features. We then train a small text side to predict these values, and a small decoder to turn the teacher's saved values into the teacher's audio, first with spectral losses and then adversarially. Finally, we connect the two halves and quantize the weights to int8. It needs no alignment learning and no joint training, and it runs on one laptop. Stored in int8, Paradee is 8.5 MB, runs 25x faster than real time on one CPU thread, and scores 4.41 on UTMOS against the teacher's 4.52. The student initially kept a slight buzz, which we trace to the phase of voiced speech between 2 and 8 kHz. A phase-locking filter applied after synthesis removes most of it, with no training and no extra parameters. Code, model files and audio samples are at this https URL 

---
# TAPDreamer: Transferable Adversarial Patches for World Action Models 

**Authors**: Xuanyu Lu, Fengqing Jiang, Kaiyuan Zheng, Yichen Feng, Yaorui Ding, Yuetai Li, Zhen Xiang, Bhaskar Ramasubramanian, Basel Alomair, Luyao Niu, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2610.06814)  

**Abstract**: World models learn to predict how their environment will evolve, making them an important foundation for general-purpose robotic control. Yet world action models depend on camera inputs whose manipulation can corrupt the visual representations used across tasks and action policies. Existing attacks on these models optimize against the victim's actions or predicted futures and therefore require access to target-model outputs. In this paper, we propose an attack, TAPDreamer, against world action models that instead uses a public encoder alone to construct a fixed local perturbation that transfers across tasks and action architectures. TAPDreamer requires no target-policy queries. Our key insight is that interactions between patch-induced changes in attention weights and value vectors broadcast a nearly identical representation shift far beyond the patch footprint, and this shift remains stable across task observations. Guided by this insight, TAPDreamer uses six frames from one source task to maximize the global L1 distance between clean and patched encoder representations. In closed-loop evaluation, one frozen patch per benchmark, covering about 6.5% of the input, reduces FastWAM's success rate from 97.7% to 0.0% across 40 LIBERO tasks and from 90.8% to 0.0% across 50 RoboTwin tasks; matched random patches retain 81.5% and 79.2% success. The same patches reduce success to 2.1% and 0.8% on two DreamWAM configurations and to 10.0% on Motus. These results show that protecting downstream action generation alone is insufficient: defenses for world action models must also secure shared visual encoders against persistent local perturbations. 

---
# Sharpen Without Search: On-Policy Distillation of Sequence-Level Power Distribution 

**Authors**: Erfan Baghaei Potraghloo, Seyedarmin Azizi, Arya Fayyazi, Saeid Shokoufa, Mehdi Kamal, Souvik Kundu, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2610.06804)  

**Abstract**: A language model can give a correct answer more probability than any single incorrect answer and still usually sample an incorrect one, because the incorrect answers together hold more probability. The power distribution raises each complete answer's probability to a power above one and renormalizes, shifting probability toward answers the model finds most likely (sharpening). Sampling from it improves reasoning without changing parameters, but needs many scored candidates per query. We show that a model can instead be trained to produce such answers in one generation. On-policy power distillation (OPPD) runs a sequential Monte Carlo sampler in which the model being trained generates candidates and a frozen teacher's power distribution weights them; the same probabilities weight each answer in a maximum-likelihood update. Training raises single-generation accuracy by up to 23.0 points on MATH500 and 27.3 on GSM8K over the untrained model at the same temperature, and one generation scores 2.4 and 3.5 points above published power sampling with 64 candidates, recovering 94 percent of the gain that 16 candidates give the untrained model. For context, against GRPO trained with verified rewards from the same checkpoint and budget, OPPD scores 3.8, 4.0 and 5.4 points higher on MATH500, GSM8K and AIME using no reference answers; the two are complementary, and OPPD applied after GRPO adds up to 9.3 points. Trained only on mathematics, OPPD raises HumanEval accuracy by up to 5.3 points. One loss coefficient moves the sharpening exponent the model absorbs between 1.19 and 2.02, against 1.14 for ordinary on-policy distillation, and it rises mostly on the model's own answers. Gains hold across model families and sizes, including a model already trained with verified rewards, where lowering the temperature gives nothing and OPPD adds 4.4 points on MATH500. Code: this https URL. 

---
# MC-Sparse: Deconstructing and Closing the Dense-Sparse Attention Gap in Diffusion Transformers 

**Authors**: Jiarui Chen, Zeqiang Lai, Jiangshan Wang, Ziheng Ouyang, Ye Huang, Xiangyu Yue, Cewu Lu, Chunchao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.06801)  

**Abstract**: Sparse attention is a primary approach to reducing the latency of diffusion transformers in long-sequence generation tasks, such as video and high-resolution 3D asset generation. However, existing methods can degrade generation quality and fidelity at high sparsity levels. Through controlled oracle comparisons, we trace this degradation to three sources: constraints imposed by token grouping, inaccurate interaction selection, and the attention contributions lost when tokens are discarded. Guided by this analysis, we propose Meta-Cached Sparse Attention (MC-Sparse), a training-free framework that selects individual key-value (KV) tokens while organizing similar queries into tile-aligned groups for efficient GPU execution. MC-Sparse caches metadata comprising query groups, KV indices selected using exact attention probabilities, and residuals between dense and sparse attention outputs, and reuses them across subsequent denoising steps. Across video and 3D generation models, MC-Sparse achieves higher fidelity to dense-attention outputs and larger denoising speedups than existing sparse-attention baselines, without visible quality degradation. Relative to dense attention, it delivers a $1.80\times$ denoising speedup on Minimax-H3-Base and a $2.32\times$ speedup on 3D asset generation, both with negligible quality loss. 

---
# IdeaLens: Detecting AI Ideas in Long-form Writing 

**Authors**: Rishanth Rajendhran, Minjoon Choi, Jenna Russell, Ramya Namuduri, Deniz Bölöni-Turgut, Marzena Karpinska, John Wieting, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2610.06778)  

**Abstract**: While modern AI detectors identify who wrote the words, emerging policies on AI use increasingly hinge on a different question: who came up with the ideas? We introduce IdeaLens, a detector that identifies whether a document's ideas came from a human or AI (idea provenance), regardless of who wrote its words. To focus IdeaLens on ideas rather than prose, we represent documents as outlines: lists of items that each pair a discourse role with a brief, paraphrased description of the content, minimizing word-level overlap with the raw text. We train IdeaLens on 1M FineWeb documents with silver labels from Pangram, a prose provenance detector. Since the outlines are largely stripped of surface-level information, the labels must be fit mainly through the ideas. In a controlled study, IdeaLens's AI flag rate drops from 95% to 7% as models write from increasingly detailed human plans, while Pangram 4 still flags 92%; from AI-derived plans, IdeaLens stays above 96%. Conversely, on a new dataset of 50 stories that human authors wrote from AI-generated plans, IdeaLens flags 68% of the stories as AI, compared to 8% for Pangram 4. On a comprehensive suite of 19 existing detection benchmarks, we show that IdeaLens maintains strong detection rates at low false positive rates, suggesting that ideas themselves provide a powerful discriminative signal, and its performance holds across domains, formats, and languages. Finally, we examine 90K predictions from IdeaLens to characterize systematic differences between human and AI ideation. We release our models and labeled datasets to facilitate future research on idea provenance detection. 

---
# MatrixFormer: A Foundation Model for Matrix Completion 

**Authors**: Dwaipayan Saha, Jacob Feitelberg, Kyuseong Choi, Raaz Dwivedi, Anish Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2610.06751)  

**Abstract**: Matrix completion underlies problems from tabular imputation to causal inference, yet existing tabular foundation models treat it as entry-by-entry prediction, repeating context for every target and discarding the matrix's two-dimensional structure. We introduce MatrixFormer, a pre-trained matrix-native transformer that predicts a full distribution for every missing entry in a single forward pass. MatrixFormer is trained entirely on synthetic low-rank and latent-factor matrices under diverse missingness patterns. Applied zero-shot and with the same model weights, MatrixFormer achieves competitive performance on causal inference panel-data tasks, language-model benchmark-score completion, tabular imputation, and recommendation systems matrix completion. These results position MatrixFormer as a general-purpose foundation model for matrix completion. 

---
# Balancing Memory Pathways: Analyzing and Improving Memory Utilization in Hybrid LMs 

**Authors**: Hyunji Lee, Joykirat Singh, Zaid Khan, Justin Chih-Yao Chen, Elias Stengel-Eskin, Alessandro Sordoni, Arman Cohan, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2610.06750)  

**Abstract**: Recurrent-attention hybrid language models (LMs), which interleave attention and recurrent layers, are increasingly used to combine the efficiency of the recurrent layers with the strong performance of attention layers. Prior work suggests that attention and recurrent layers offer complementary pathways to use past information: attention supports precise memory recall from earlier tokens, while recurrent layers support consolidation of disparate information over long contexts. However, we observe that simply having access to both pathways does not mean that hybrid LMs are effectively using them. We find that they rely substantially more on attention than on the recurrent state. Standard supervised fine-tuning improves overall performance but does not improve how the two memory pathways are coordinated: the model becomes more reliant on information propagated by attention layers, while its use of information propagated by recurrent layers remains limited. To encourage better coordination between the two memory pathways, we add an auxiliary loss that limits attention's access to earlier context while the recurrent state propagates through the full sequence. This objective encourages the model to retain and use information through the recurrent pathway alongside attention. It improves overall performance, with particularly strong gains on tasks involving longer contexts or requiring information aggregation, consistent with the strengths of recurrent layers observed in analysis. Crucially, this imbalance and the benefit of our auxiliary loss generalize: they apply to multiple recurrent-attention LMs in question-answering and agentic tasks, as well as to attention-based LMs that combine different forms of memory. Together, our findings show that simply providing multiple memory pathways does not ensure their effective use, and that targeted supervision is needed to better coordinate them. 

---
# BazaarBench: Delegation Safety in Decentralized C2C Marketplaces Run by LLM Agents 

**Authors**: Ziyan Wang, Shuqing Shi, James Oldfield, Samuele Marro, Jialin Yu, Philip Torr, Yali Du, Adel Bibi  

**Link**: [PDF](https://arxiv.org/pdf/2610.06748)  

**Abstract**: In decentralized consumer-to-consumer (C2C) marketplaces, people list goods, negotiate with strangers, and rate one another, so trust rests on reputation. Large language model (LLM) agents now act for users, raising risks to their money, privacy, and reputation. We introduce BazaarBench, a simulated C2C marketplace and benchmark for evaluating the safety of these agents. It tracks ownership, item condition, and commitments across transactions, combining record checks with rubric-based LLM judgments to identify six failure types across five stages. We run three base markets for 30 simulated days, each with 100 agents using one model and inventories drawn from a public eBay sample. Across 45 continuations, we evaluate five models under ordinary instructions, deadline pressure, or adversarial instructions to exploit other traders. Each continuation runs for seven simulated days from a copy of a market's day-30 state. The tested model controls the same 20 selected agents, retaining their personas, inventories, and histories, while the other 80 keep the base model. All five models attempt to promise the same item to multiple buyers under ordinary instructions. Adding targets and deadlines increases these attempts for every model. Under adversarial instructions, the share of tested sellers' committed transactions completed despite unavailable items or overstated conditions rises from 15.4% to 33.4%, reaching 55.5% for GPT-5.4. Averaged across models and markets, simulated weekly earnings per tested agent rise from USD 20 under ordinary instructions to USD 33 under adversarial instructions. Most of the increase comes from items the sellers never held. We release the simulator, saved market states, evaluation code, and records covering 357,608 agent model calls for evaluating new models and developing safer marketplace agents. 

---
# BRANCH-MoE: Balance-Aware Tree Routing for Large Embedding Models 

**Authors**: Gang Fu, Adel Javanmard, MohammadHossein Bateni, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2610.06725)  

**Abstract**: Mixture-of-experts (MoE) layers increase model capacity without a proportional increase in per-example computation. However, conventional flat routers can yield imbalanced expert utilization and treat experts as an unstructured collection, whose indices carry no topological meaning. We introduce {\bf BRANCH-MoE}, a routing architecture that places \(E\) experts at the leaves of a binary decision tree of depth \(\log_2 E\). At each internal node the branching probability is centered on the arrival-weighted mean score of the traffic reaching that node. This mean is estimated using an exponential moving average, which promotes utilization of both child subtrees without an auxiliary load-balancing loss. We show that this moving-average estimate admits an explicit noise-lag trade-off. We prove that for linear node maps and log-concave arrival distributions, this mechanism prevents routing-mass collapse. We further establish that, under a frozen router, an expert's execution frequency controls its stochastic-gradient convergence rate, and that confident decisions near the root bound cross-device communication when experts are assigned to devices by tree prefix. We evaluate BRANCH-MoE against Switch softmax, DeepSeek-V3 dynamic-bias, Skywork logit-normalized, and deterministic hash routing on Criteo click-through-rate prediction, Forest Covertype, HIGGS, and YearPredictionMSD, using \(E=16\), top-\(4\) routing, and five random seeds. Our results show that hierarchical routing can preserve task quality and balanced utilization while inducing a topology that supports localized expert co-activation and reduced communication. 

---
# Closing the Context Gap: Activation Alignment for Tabular In-Context Learning 

**Authors**: Yoel Zeldes  

**Link**: [PDF](https://arxiv.org/pdf/2610.06679)  

**Abstract**: Tabular foundation models perform in-context learning (ICL) by conditioning predictions on labeled training examples provided as context. Unlike traditional models that separate training from inference, these models must process all training examples in every forward pass, making each prediction expensive. Restricting the number of training examples reduces this cost but substantially degrades performance. Instead of discarding context, we propose activation alignment, a method that leverages the full context to teach a model how to behave when seeing only a subset. This is achieved by training a lightweight linear transformation on synthetic unlabeled data to map the intermediate activations of a data-constrained "student" (using partial context) toward those of a full-context "teacher" (using all data). Training the aligner requires no GPU and converges in seconds to minutes on commodity hardware. We evaluate on 38 classification datasets from the TabArena benchmark using the leading two tabular foundation models, TabPFN-3 and TabFM. Across all context budgets, the aligned student yields broad, statistically significant improvements over the unaligned baseline for both models. In low-data regimes, alignment recovers nearly half of the teacher's predictive advantage. The method provides a practical, low-overhead approach to achieving the inference speed of compact contexts while closing a significant fraction of the performance gap to the full-context teacher. 

---
# VideoTapestry: Query-Adaptive Memory Refinement for Multi-Agent Long-Video Understanding 

**Authors**: Yucheng Liu, Yufei Yin, Mingxiao Feng, Jiajun Deng, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.06672)  

**Abstract**: Long-video understanding places substantial demands on memory, as answering questions often requires retrieving information distributed across extended temporal spans. Existing approaches broadly follow two paradigms: query-driven exploration, which is sensitive to localization errors, and query-independent memory construction, which may omit question-specific details. We introduce VideoTapestry, a training-free multi-agent framework that adapts a preconstructed hierarchical video memory through coarse-to-fine, query-driven refinement. The preconstructed memory organizes video content into three levels, capturing global narrative context, event-level temporal structure, and fine-grained relational evidence, respectively. To support coarse-to-fine localization and observation, we assign a specialized agent to each level, keeping retrieval and refinement within a scale-specific context. Guided by the query, these agents revisit relevant video regions and enrich layer-wise memories with targeted multimodal observations. Their refinements are assembled according to the original hierarchy into a composite query-adaptive memory, preserving global context in a compact form while retaining fine-grained evidence along query-relevant branches for final reasoning. Compared with direct GPT-5.5 inference, VideoTapestry achieves absolute accuracy gains of 17.2%, 14.9%, 9.8%, and 7.0% on LVBench, LongVideoBench (Long), Video-MME (Long), and EgoSchema, respectively, achieving the state-of-the-art results among all competitors. 

---
# AffordCraft: Scalable Construction of Task-Ready Simulation Assets from Single Images 

**Authors**: Haoyun Yang, Xueyang Zhou, Ziyi Xie, Yongchao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.06643)  

**Abstract**: Robot learning in simulation depends on the objects the simulator offers. Many tasks need objects with separate parts, joints that allow the required motion, and physical properties that remain valid under contact. Existing methods recover this structure anew for every image: generative models predict parts and joints that mostly fail to settle or move in simulation, and general-purpose agents need a long session of model calls for each photograph. AffordCraft builds such an asset from a single RGB image and a task instruction by retrieval instead of generation: it locates the object and the part to operate, selects a matching entry from a library of articulated assets, and fits it to the image while keeping its parts and joints intact. Without any box or mask marking the object, AffordCraft produces a physically valid asset for 1,703 of 2,000 photographs from 31 categories. Five generative methods pass on at most 45% of the same photographs and, at the median, need 10 to 78 times our GPU time per valid asset. On 50 cluttered images, 162 of 237 annotated objects pass the same physical test after automatic detection. Growing the library from 141 to 11,372 entries needs no change to the method and raises category coverage from 46% to 100% and the share of selections with the requested label from 18% to 51%. We also build manipulation tasks from the constructed assets, both with single objects and in composed scenes; policies trained on scripted demonstrations complete both kinds of tasks from initial states unseen in training. 

---
# Differentially Private Mixing of Public Datasets Improves Private Learning 

**Authors**: Yufei Chen, Tejumade Afonja, Anvith Thudi, Nicolas Papernot  

**Link**: [PDF](https://arxiv.org/pdf/2610.06636)  

**Abstract**: Many machine learning applications involve sensitive data and therefore require training under differential privacy (DP). However, DP training often degrades model utility. In some cases, first pre-training the model on "public" data before finetuning with DP on the sensitive data can reduce the drop in utility. However, the success of this depends on how relevant the selected public dataset is to the sensitive data. We introduce the first pipeline that privately learns the mixture of several public datasets to pretrain on for a given sensitive downstream task. Our key insight is that we can privately find the best mixture of multiple public datasets by privately learning a low-dimensional linear model. We tested our method on the NIH dataset for X-ray classification and the ENRON email dataset for language modeling. Applying our method to find tailored mixtures of X-ray datasets to pretrain on for diseases in the NIH ChestX-ray14 dataset, we improved macro AUC by up to 0.037 across privacy budgets compared to the baselines, with gains as large as +22.8% relative AUC on Cardiomegaly at $\epsilon=1$. For DP training on the ENRON dataset, pre-training on our mixture of The Common Pile (a collection of public-domain text datasets) decreased test perplexity by 16% relative to the baseline mixtures. 

---
# Large Language Model-Guided Discovery of Weight-Five Bivariate Bicycle Codes 

**Authors**: Juan Cruz-Benito  

**Link**: [PDF](https://arxiv.org/pdf/2610.06623)  

**Abstract**: Building on our earlier program-evolution workflow guided by large language models (LLMs), we study weight-five bivariate bicycle (BB) and perturbed bivariate bicycle (PBB) codes. The resulting catalogue contains 1,142 distinct code proposals, including 1,081 nonbaseline proposals attributable to LLM-generated programs. Across the catalogue, we certify connected Calderbank--Shor--Steane (CSS) realizations [[96,4,10]], [[140,6,10]], and [[180,4,14]]. A post-search comparison certifies seven imported Lin--Pryadko archive constructions. For leading parameter triples also represented in that archive, we provide exact distance evidence, explicit bivariate presentations, and verified component reductions. A basis-independent connectivity analysis identifies 409 of the 1,142 catalogue entries as disconnected and shows that 73.1\% of the classes with exact distance certificates contain repeated connected components. Algebraic analysis organizes the connected CSS classes into order-3, order-7, and order-15 cyclotomic-kernel strata. The strongest exact connected PBB parameter point is [[216,4,10]], attained by two distinct component classes. Among the 936 distinct CSS proposals from the LLM-guided campaign with a recorded positive distance, 816 (87.18\%) are certified at $d\geq5$. For comparison, three random-search controls each sample 6,444 CSS code proposals uniformly without replacement, using the same per-lattice and encoded-dimension sample counts as the LLM-guided campaign. In these controls, 4,672--4,785 proposals (72.50--74.26\%) meet the same criterion. The LLM-guided campaign has the higher certified yield, while the random controls cover more connected classes. Together, these results extend LLM-guided discovery to a more constrained code family and provide a reproducible structural and exact-distance account of its strongest candidates. 

---
# Frozen Factor or Spectral Band? Disentangling Two Choices in Low-Rank LoRA 

**Authors**: Adnan Slimane Ali, Ayoub Belfatmi, David Ngwe Pouth  

**Link**: [PDF](https://arxiv.org/pdf/2610.06621)  

**Abstract**: Spectral variants of low-rank adaptation (LoRA) choose both a subspace and which factor to freeze. We separate these choices by freezing the input factor A or output factor B on the top or bottom singular directions of pretrained weights, with learning rates selected separately. At rank 2, the same-band advantage of freezing A is larger than either within-factor band difference on all four task-model pairs with complete comparisons. Freezing B also trails comparable-budget free LoRA by 8-18 percentage points on five pairs spanning a formatting task and OpenBookQA. The A-frozen advantage persists in a single-GPU-model replication and within individual MLP module groups, including controls with equal or greater trainable counts for B frozen, and when A is frozen on a random orthonormal basis. The factor contrast weakens with rank. On OpenBookQA / Qwen2.5-1.5B at rank 16, PEFT's MiCA implementation trails comparable-budget LoRA by 3.08 points under a shared training recipe transferred from the MiCA paper. A trained oracle output subspace largely removes the low-rank deficit; partial warm-up gains recur across three direction seeds. The factor-versus-band ordering is descriptive; an approximate multiplicity audit weakens several earlier significance claims. These results extend known factor asymmetry by showing how its magnitude depends on spectral placement, rank and training conditions. 

---
# Word-Level Text Unmixing via Evidence-Preserving Ownership Routing with Language Models 

**Authors**: Jinglin He, Siyang Jiang, Lixing He, Guoliang Xing, Hongkai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.06603)  

**Abstract**: Text from multiple sources can become interleaved into a single sequence when attribution metadata is lost, such as overlapping speech transcripts, document reading flows, or concurrent agent streams. We formalize this challenge as Word-Level Text Unmixing: given an interleaved lexical stream and source count K, recover the original source sequences while preserving every word occurrence and its within-source order exactly. Directly generating separated texts with LLMs can omit, duplicate, or hallucinate words, violating this exact-reconstruction objective. We therefore propose Evidence-Preserving Ownership Routing (EPOR), which decouples source-ownership prediction from reconstruction. EPOR adapts a causal LLM to predict canonical ownership routes conditioned on the mixed stream and prior routing decisions. At inference, completion-safe constrained decoding is combined with deterministic indexed reconstruction, yielding structurally valid K-source partitions that preserve every observed occurrence exactly once. We also introduce UNMIXBENCH, covering controlled synthetic mixtures, timestamp-derived speech from AMI and ICSI, layout-derived document streams from ReadingBank, and simulated concurrent digital outputs. Across five evaluation tracks, a 4B EPOR model achieves the lowest mean minimum-permutation word error rate among finetuned baselines, reducing the five-track mean by 22.3% relative to compact source-array generation and remaining competitive with zero-shot frontier LLMs. These results show that when lexical evidence is fully observed, separating ownership inference from lexical regeneration provides a reliable alternative to direct generation. 

---
# SimForcing: Distilling Simulation Motion Priors into Real-Domain Robot World Models 

**Authors**: Xiaodong Wang, Tianle Li, Chuanxin Song, Junliang Xie, Zhanmi Zhong, Suiying Wu, Peixi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.06598)  

**Abstract**: Action-conditioned robot world models must respond precisely to robot trajectories while preserving realistic visual dynamics, yet learning both from heterogeneous robot videos remains challenging. Simulation offers structured motion supervision, but appearance differences hinder direct transfer, and inaccurate simulation predictions can misguide real-video generation. We present SimForcing, a simulation-guided framework that uses simulation both as a source of transferable motion knowledge and as a controllable reference for prediction. First, we transfer motion knowledge from a simulation teacher through latent-motion distillation, aligning temporal changes in latent space to internalize motion priors while mitigating the influence of appearance differences. Second, we introduce multi-block simulation conditioning with condition dropout to exploit predicted simulation trajectories without relying excessively on their accuracy. Our simulation-conditioning classifier-free guidance scheme unifies these two ideas by balancing predictions based on internalized motion knowledge with those additionally guided by simulation latents. The jointly trained student generates both simulation conditions and real-domain videos, requiring no additional world model at inference. On Bridge, SimForcing achieves the best PSNR, SSIM, LPIPS, and FVD among the compared methods without external embodied pretraining. Evaluation on InternData-A1 further supports its applicability across robot datasets. Moreover, using our trained world model to initialize a vision-language-action model improves LIBERO success, suggesting its utility for downstream policy learning. \url{this https URL} 

---
# Mind the Accent Gap: British Accent Robustness in Speech-Driven Financial Voice Assistants 

**Authors**: Aadam Haq, Oggi Rudovic, Malcolm Chadwick, Jay Rainey, Shucong Zhang, Ricardo Guerrero, Sourav Bhattacharya, Maja Pantic  

**Link**: [PDF](https://arxiv.org/pdf/2610.06587)  

**Abstract**: AI voice assistants often use Automatic Speech Recognition (ASR) with LLM-based reasoning, yet existing systems struggle with regional British accents, including Scottish, Irish, and Welsh accents, since most ASR models are trained predominantly on American English voice data. Consequently, errors can carry through to the LLM stage, corrupting tool-call arguments and producing wrong or missing responses, which is especially costly in finance. Deployable ASR must also meet tight latency and memory budgets, making an accent-robust model choice even harder. We introduce CavaBench, the first internally collected benchmark of spoken financial queries, and use it to evaluate a range of ASR models and their end-to-end ASR-LLM pipeline behaviour across self-reported British accents. We find that WER strongly predicts downstream tool-calling accuracy ($r = -0.93$) but can fail to reflect task-level performance, with accent-related failures varying substantially across models and acoustic conditions. These findings guide the design of more inclusive, reliable voice-based financial assistants. 

---
# Does AI Help Cyber Attackers or Defenders? Evidence from Nonpublic Vulnerabilities and Subsequent Attacks 

**Authors**: Tobias Heldt, Matt Turk, Christoph Landolt, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2610.06584)  

**Abstract**: The release decision for frontier AI systems increasingly relies on cyber capability benchmarks, yet public vulnerability benchmarks can expose agents to previously published advisories, exploits, and fixes, making it difficult to distinguish prior exposure from capability on unseen vulnerabilities. We evaluate open-weight and proprietary AI models on exploit generation, vulnerability repair and subsequent attacks in five nonpublic software environments, including vulnerabilities we privately disclosed while they remained unpatched. Researcher-developed and reviewed deterministic graders, not LLM judges, determine task scores. Comparisons with 209 disclosed vulnerabilities and cryptographic challenges reveal substantial variation across systems and vulnerability types. Repair scores exceed attack scores in two nonpublic environments and fall below them in three. Passing an initial security test is also insufficient: another exploit succeeds in 92 of 524 non-independent defender test intervals after the initial exploit is stopped. These results motivate vulnerability-specific attack-repair comparisons and subsequent resistance tests. 

---
# BrainTRACE: Tracing Longitudinal, Multimodal, and Volumetric Evidence in Brain MRI Clinical Reasoning 

**Authors**: Qizhen Lan, Mengchen Fan, Hang Zhang, Jingwei Duan, Moule Lin, Jialin Chen, Baocheng Geng, Xiaoqian Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06571)  

**Abstract**: Brain MRI interpretation is a longitudinal clinical reasoning problem: radiologists compare serial studies, integrate information across MRI sequences, localize findings within volumetric anatomy, and translate this evidence into report-grounded assessments. Existing medical VQA and 3D imaging benchmarks capture important parts of this workflow, but often evaluate brain MRI through isolated images, static volumes, or ungrounded report-style answers, thereby obscuring failures in the evidence chain that support clinical validity. We introduce BrainTRACE, a report-grounded benchmark for evaluating whether vision-language models can trace the evidence structure required for longitudinal brain MRI interpretation. BrainTRACE contains 7,273 scored VQA instances derived from 1,778 longitudinal patients, 7,299 MRI studies, and approximately 29k co-registered 3D MRI sequence volumes. The benchmark is organized by five levels of clinical reasoning, from acquisition recognition to case-level synthesis, and by evidence demands covering longitudinal comparison, report-grounded references, multi-sequence integration, and volumetric spatial evidence. BrainTRACE supports rendered inputs compatible with standard VLM interfaces, a 3D-evidence condition, and a decomposed case-reasoning track that audits six steps in a longitudinal evidence chain. Evaluation of 20 VLM configurations shows that current systems can identify isolated visual cues but rarely compose them into grounded longitudinal interpretations. We release the benchmark specification, evaluation lists, scoring implementation, scoring rubrics, and audit-record format to support reproducible progress in brain MRI VLM evaluation. 

---
# Symmetry and AI-assisted discovery of magic-state factories 

**Authors**: Shubham P. Jain, Adam Wills, Shraddha Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.06535)  

**Abstract**: Magic-state distillation is a major resource cost in fault-tolerant quantum computing. The cost of a magic-state factory depends strongly on its failure rate, which grows with the number of input magic states. Although symmetry-restricted methods have recently made distance two searches tractable, distance three and above have remained elusive at moderate input counts. We develop symmetry- and AI-assisted methods to search this regime. We present a unified binary-matrix formulation encompassing both triorthogonal-code and direct circuit searches. We show that distance at least three is equivalent to nonzero, pairwise distinct syndromes, separating the choice of syndromes from the search for compatible output gates. We restrict the syndrome search using group symmetry and language-model agents, followed by deterministic solving and independent verification. Our searches yield 699 factory classes, including 564 new ones. These include factories for pure-T states and factories with entangled outputs comprising combinations of T, CS, and CCZ magic states. The pure-T factories [[63, 11, 3]] and [[850, 128, 6]] achieve the lowest overhead exponents we know among protocols with at most 100 and 1000 inputs, respectively, with $\gamma = 1.589$ and $\gamma = 1.057$. Our [[1715, 287, 6]] factory, with $\gamma = 0.998$, is the smallest known pure-T factory with $\gamma < 1$. We also provide a context directory of search briefs and campaign notes with which readers can train their own agents and tailor the search to their requirements. With these results, we begin constructing an active, open-source repository of magic-state distillation protocols for the quantum community, supplemented by our methods and data, for the practical fault-tolerant quantum computing regime. 

---
# Anatomy of LLM Sycophancy: What a Flip Rate Hides 

**Authors**: Haonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06522)  

**Abstract**: A model under pushback can correct itself, capitulate, or hold, and one flip rate counts a correction and a capitulation alike. Using SycoLens, a modular replay protocol, we test how user pressure and evaluation settings shape measured flip rates. Each measurement is one stateless replay of an item, a committed answer, and one scripted user line in a fixed form. Every effect is read against a matched control with the line deleted. Pushback wording, committed text, answer format, boundary distance, and ground truth become factors of one instrument; earlier instruments vary one to three of them. Across eleven frontier models from three providers and about 760,000 controlled replays, which models look sycophantic depends on how the user pushes back. Lines that assert the opposite verdict and lines that challenge the answer without asserting one rank the models almost unrelatedly. Flip effects grow several-fold near a model's boundary, yet items answered identically in every screening draw still carry about half of the most-affected totals. On arithmetic tasks where the truth is known, one model re-derives and corrects itself under pressure while another abandons correct answers without written work. On the model tested, a planted derivation lowers release of the answer it argues for, true or wrong, where a bare stated value does not; the wrong answer is corrected much more often than the true one is abandoned. Under a yes/no readout the rankings come closer, entangled with a pressure-induced shift toward "no". One score per model therefore compares different behaviours across models and benchmarks. We condense these dependencies into a reporting profile; the instrument, records, and analyses will be released upon publication. 

---
# GCTAuto-encoder: A Cross modal Framework for Security Flaw Detection in IoT Networks 

**Authors**: Najmieh Sadat Safarabadi  

**Link**: [PDF](https://arxiv.org/pdf/2610.06517)  

**Abstract**: IoT encompasses diverse physical entities, from smart home devices to autonomous vehicles, creating a complex environment with heterogeneous security models. This heterogeneity makes IoT sub-systems vulnerable to various network attacks. Modern security systems must therefore be more robust to ensure security and privacy for IoT applications. A highly secure IoT system also demands real time insight, requiring data collection at the edge of the computing layer. This diversity calls for a unified security model applied at the foundational level. Edge intelligence offers a direct approach to handling device diversity. A key goal of edge intelligence in IoT is to extract insight from local data; security models can then use this data to build local node protections, and integrating AI models yields an advanced security solution. This research proposes a novel deep learning algorithm for effective intrusion detection at the edge, supported by a cloud-based IoT framework. We evaluate the proposed cross modal deep learning algorithm against baseline models. The contribution is a cross domain Deep Neural Network (DNN) algorithm for intrusion detection. The objective is to assess a multi-method deep learning model to detect intrusions in IoT systems at the edge via community detection with modeled attention. We evaluate GCT auto-encoder, a novel framework integrating edge intelligence to identify security flaws. The model significantly improves performance and efficiency. On a network intrusion IoT dataset covering multiple attack scenarios, it achieved 0.908 accuracy, reduced learning loss to 0.00156, and outperformed existing approaches. 

---
# ArtifactArena: Evaluating Models by What They Build in the Physical World 

**Authors**: Kushagra Tiwary*, David Mayo*, Nikhil Behari, Xiangzhou Sun, Abdulrahman Alabdulkareem, Isaac Galatzer-Levy, Boris Katz, Brian Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2610.06511)  

**Abstract**: To evaluate the frontier, we must measure models not by what they say, but by what they can engineer and build in grounded physical environments. We introduce \textsc{ArtifactArena}, an open-ended platform where models face a physically grounded hardware-software co-design challenge: engineering fully functional robots to compete in a simulated arena. We evaluate a frontier model's zero-shot, verifier guided refinement, and open-ended physical design capabilities through three harnesses that refine their bot artifacts based on text descriptions, physics simulator feedback, and gameplay data. We benchmark these capabilities with an Elo ranking of frontier models derived from head-to-head tournaments between their artifacts. By releasing this framework and tournament infrastructure for ongoing community submissions, we establish a living, non-saturating testbed to continuously measure the expanding limits of open-ended intelligence in the physical world. Please visit \href{this https URL}{this https URL} for more information. 

---
# Harmful Content Generation in Text-to-Image Models: Capabilities and Moderation Limitations 

**Authors**: Paschalis Giakoumoglou, Manos Schinas, Symeon Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2610.06503)  

**Abstract**: Text-to-image generative models can produce highly realistic imagery but also raise concerns about harmful misuse. While safety mechanisms exist, systematic evaluations of their effectiveness against realistic attacks remain limited. We present a systematic evaluation of harmful content generation across five open text-to-image models using an automated pipeline that transforms legitimate news captions into unsafe prompts targeting sexually explicit content, violence/gore, harmful stereotypes, self-harm, and hate speech. We evaluate both standard models with built-in safety mechanisms and community fine-tuned variants that bypass content restrictions. A human evaluation of 1,500 generated images shows high harmful-content generation rates: 89.2% for gore-related prompts, 47.6% for sexually explicit content, 43.6% for harmful stereotypes, 46.0% for hate speech, and 34.5% for self-harm, predominantly through graphic violence. Models show substantial capability for generating violent and stereotypical content, while community fine-tuned variants are particularly vulnerable to sexually explicit prompts. Generation quality is largely preserved under harmful prompting, producing imagery of sufficient fidelity to pose risks for disinformation and abuse; FLUX.1-dev produces clearly realistic harmful images in 30.9% of cases. We further evaluate automated moderation systems and find substantial detection gaps that allow unsafe images to evade filtering. Finally, we assess synthetic image detectors and show that models trained only on benign datasets perform worse on explicit content, while more diverse training data improves detection, highlighting semantic distribution gaps in current approaches. These findings expose limitations in current generation safeguards, moderation systems, and synthetic image detection, highlighting the need for stronger defenses against misuse at scale. 

---
# Topology-Informed Prompt-Conditioned Universal Segmentation of Uterine Structures from Ultrasound and MRI 

**Authors**: Yongheng Sun, Yuexi Gu, Jingwen Sun, Maureen Kohi, Mingxia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06494)  

**Abstract**: Multi-structure segmentation of the uterus is important for computer-assisted screening, diagnosis, and treatment planning of uterine diseases, where ultrasound and MRI provide complementary clinical information. However, developing a unified model across these modalities is challenging due to their substantially different image appearances, anatomical contexts, spatial resolutions, and label spaces. Moreover, existing datasets often define different segmentation targets, making joint learning challenging and potentially leading to negative transfer across heterogeneous tasks. To this end, we propose a Topology-informed Prompt-conditioned Universal Segmentation (TPUS) framework for segmenting multiple uterine structures across ultrasound and MRI. TPUS introduces a graph-based multi-dataset backbone comprising modality-specific stems and a modality-shared graph-based encoder-decoder to support modality-sensitive input adaptation, structural feature reasoning, and joint representation learning across heterogeneous uterine segmentation tasks. In addition, TPUS uses task-aware class prompts to condition the segmentation process for different datasets and label spaces, a dynamic convolutional adaptation module to generate task-specific output responses, and a topology-informed loss to encourage anatomically consistent predictions. Experiments on a uterine ultrasound dataset and a T2-weighted uterine myoma MRI dataset demonstrate that TPUS achieves Dice scores of 0.898 and 0.693 on the two held-out test sets, respectively, outperforming several generic and universal segmentation baselines. Source code can be accessed at this https URL. 

---
# Odyssey: A Closed-Loop Benchmark for Long-Horizon Real-World Driving with Explicit Navigation Routes 

**Authors**: Jungho Kim, Hongjae Shin, Seunghoon Yu, Heecheol Yoo, Myeongjun Kim, Jiyong Oh, Donghyuk Kwak, Seunghyeop Nam, Haesung Oh, Hyunju Kim, Hyungchan Cho, Jaehyun Park, Soo Won Seo, Jun Won Choi  

**Link**: [PDF](https://arxiv.org/pdf/2610.06469)  

**Abstract**: Closed-loop evaluation of end-to-end driving requires continuous rollouts that reveal how earlier decisions affect subsequent driving. However, existing benchmarks evaluate only short segments and fail to capture later consequences. Ambiguous directional commands also obscure the intended navigation objective. We introduce Odyssey, a closed-loop benchmark for long-horizon driving comprising 100 scenarios, each reconstructed from a 100-second nuPlan driving log to preserve the context of navigation maneuvers and traffic interactions. To provide a consistent navigation objective, Odyssey replaces directional commands with explicit standard-definition (SD) map routes that specify which roads to follow, while sensor-based planning determines local driving actions. Throughout these rollouts, diffusion-based refinement of 3DGS-rendered images reduces rendering artifacts along the ego trajectory. To assess how effectively planners follow these routes and prepare for upcoming maneuvers, we introduce SD Route Compliance and Pre-Lane Change Score. These assessments are complemented by RouteDS, which extends the Driving Score with penalties for SD-route deviations and failed lane preparation. We adapt state-of-the-art planners, including vision-language-action (VLA) models, and evaluate their navigation performance using these metrics. Odyssey highlights open questions in route representation and integration for E2E driving. Benchmark code and adapted baselines will be released publicly. 

---
# Latent Flow Matching for Molecular Graph Generation 

**Authors**: Mathis Goupillon, Roman Bresson, Konstantinos Divriotis, Michalis Vazirgiannis  

**Link**: [PDF](https://arxiv.org/pdf/2610.06468)  

**Abstract**: Modern graph generative models typically operate directly in the discrete graph space, explicitly generating node and edge variables, which can become costly as graphs grow. In this paper, we perform generation explicitly on latent representations of entire graphs obtained from a pretrained Variational Autoencoder with high reconstruction fidelity. The generated representations, obtained through flow matching, are then decoded only at the final step. Across molecular benchmarks of increasing size, our approach achieves strong validity and FCD while offering a favorable quality-efficiency trade-off compared with state-of-the-art explicit graph generative models. One of the main advantages of this formulation is that the graph representation only needs to be learned once, after which the same one can be reused across multiple generative objectives without retraining. We demonstrate generation guided by molecular properties and further introduce validity-aware generation though a classifier learned directly in latent space. All code will be made available upon acceptance. 

---
# A Physics-Guided Transformer Framework for Electromigration Analysis in Multi-Segment Interconnects 

**Authors**: Pavlos Stoikos, Anuj Pathania, George Floros  

**Link**: [PDF](https://arxiv.org/pdf/2610.06464)  

**Abstract**: As technology scales to smaller nodes, increasing current densities make electromigration (EM) one of the dominant reliability challenges in on-chip interconnects. Accurate transient stress analysis is needed to identify wires susceptible to EM degradation, but applying physics-based solvers across many interconnects remains computationally expensive. This paper proposes a physics-guided transformer framework for fast EM stress prediction in multi-segment interconnect lines. The framework converts each line into geometry- and DC-aware segment tokens and uses transformer attention to capture line-level context. A lightweight query decoder then predicts stress at selected locations and time instants. The model is trained with an objective that combines normalized supervised regression, linewise relative-$L_2$ loss, and physics-guided continuity and terminal-flux terms. Experiments on IBM power grid benchmarks show that the proposed model achieves relative-$L_2$ error below 8\% and reaches up to 2459.68$\times$ speedup compared with the matrix exponential~solver. 

---
# The Assistance Dilemma: Learning to Teach via Multi-Turn Reinforcement Learning 

**Authors**: Jakub Macina, Manu Kapur, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2610.06446)  

**Abstract**: Large language models (LLMs) trained to answer questions are natively poor at teaching. Reinforcement Learning (RL) against a simulated student is a promising approach to improve their pedagogy, but existing RL-trained tutors reward the student's success on the tutored problem with the tutor's words still in context. The reward is then easiest to raise by telling the student the answer, and a tuned penalty is needed to reduce telling. Drawing on learning sciences, we introduce a masked near-transfer post-test: the student is tested on an unseen variant of the tutored problem with the tutor's utterances masked, so the reward can rise only through what the student wrote in its own turns. This discourages cognitive offloading by the student and allows the continuous penalty to be replaced by two binary reward gates (factual correctness of tutor response, no solution handover). A leave-one-out ablation shows that the learning-gain reward on its own does not separate teaching from telling: the gates reduce solution handover while the near-transfer post-test improves out-of-domain transfer. Using these reward designs we develop Eduardo, a multi-turn RL recipe for training LLM tutors, and use it to train 4B, 9B, 14B and 27B models from two distinct LLM architectures. Our post-trained Eduardo-27B model matches Gemini-3.1-Pro on MathTutorBench and Claude Opus 4.8 on TutorMoments at 2.4-6.2x fewer thinking tokens than frontier models, which matters for interactive tutoring. Without being named in the reward, the model more than doubles its use of the push-for-justification teacher move while support fading (e.g., assigning independent work), whose payoff lies beyond a single-problem dialog episode, is trained out. We open-source our training environment, an 8,671-problem near-transfer dataset, and trained models for further development. 

---
# Choosing an energy-efficient software architecture for building system diagnostic support 

**Authors**: Roxane Koitz-Hristov, Franz Wotawa  

**Link**: [PDF](https://arxiv.org/pdf/2610.06444)  

**Abstract**: Around 30\% of global energy expenditure can be attributed to the building sector, where a large portion of energy-consumption could be avoided by repairing existing faults. Fault detection and diagnosis (FDD) software addresses this issue; however, its creation and operation also have an environmental impact. The magnitude of this impact is influenced by the diagnosis architecture, as different architectures and methods have different energy demands. Yet, simply considering the energy consumed by the software itself is not sufficient to assess its overall environmental impact, since the diagnostic performance, e.g., number of detected faults or number of faults missed, also contributes to its ecological footprint.
In this paper, we propose an energy-consumption model that considers FDD performance and energy spend directly by the diagnosis software. In an initial experiment, we compare several FDD architecture families, i.e., rule-based, model-based, classical machine learning, and large-language-model-based, in simulation using performance and energy-consumption values collected from prior literature. The results show that considering the computational energy and accuracy of FDD can change the relative benefit of the different approaches. Computationally efficient machine learning methods, such as random forest, provide the largest net savings on smaller buildings, whereas more resource-intensive approaches, such as fine-tuned large language models, become advantageous as building size increases. Our findings suggest that overall energy efficiency depends not only on the computational demand of the FDD software, but also on its diagnostic performance and the scale of the building. 

---
# ARO: Aligned Representation learning for multi-Omics data 

**Authors**: Amogh Singh, Yash Shah, Chiara D'Ercoli, Arash Mehrjou, Patrick Schwab, Timothy Jones, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2610.06443)  

**Abstract**: The high cost of functional molecular assays, and prevalence of missing modalities and unmatched samples in computational biology, create significant barriers to comprehensive multi-omic profiling, essential for capturing and reasoning over molecules, cells, tissues, and organisms. This work proposes a model that learns meaningful representations from multi-omics cancer data supporting the reconstruction of missing and unpaired modalities. Contrary to increasingly complex, larger models, e.g. Foundation Models (FMs), ARO prioritizes practical applicability in limited or incomplete data settings. ARO optimally reconstructs missing modalities (MSE of $0.15$ on the validation and test data in the Unmasked settings), with its learned latent embeddings enabling a downstream cancer classification task. Our findings indicate that analyzing diverse molecular layers as a single integrated system offers a reliable and cost-efficient approach, reducing dependence on large-scale experimental testing, while still supporting multi-omic exploration in limited data settings. 

---
# Better Call Reward: Reward Hacking as Strategic Abstention in Legal Reasoning Models 

**Authors**: Subramanyam Sahoo, Justin Shenk  

**Link**: [PDF](https://arxiv.org/pdf/2610.06439)  

**Abstract**: What happens when a legal AI model learns to look like a lawyer instead of reasoning like one? We fine tune Qwen3-8B with Group Relative Policy Optimisation (GRPO) against a proxy built from three surface features: citation count, legalese density, and response length. The model does not learn to reason more effectively. It learns to withhold commitment. Across 16 yes or no legal reasoning tasks from LegalBench (N=320), overall accuracy collapses from 0.500 (chance) to 0.072 (McNemar p < 10^-36), driven entirely by the rate of properly formatted answers falling from 0.900 to 0.109. The model stops committing to answers. Yet when it does commit, accuracy rises from 0.556 to 0.657, showing that the collapse is not a failure of capability but a strategic response: the model has learned that verbose responses packed with citations but empty of a direct answer score higher than terse correct ones. We term this the Saul Goodman effect, a policy that becomes maximally lawyerly while becoming maximally noncommittal, and prove formally that it is the optimal response to any surface feature proxy that attaches no penalty to abstention. We further show that 89.3% of citations produced after training are structurally implausible hallucinations, many of them subtly corrupted names of real landmark cases, constructed in effect to survive a casual read and fail under scrutiny. To detect this failure mode before deployment, we introduce three diagnostic tools: the Confidence Theater Score (CTS), the Citation Plausibility Rate (CPR), and the Regret Gap (RG). In a domain where a confidently wrong answer can constitute malpractice, the broader lesson is direct: a reward function that measures how legal a response looks will produce a model that is maximally photogenic and minimally useful. 

---
# Valid Stopping in Adaptive Generator-Verifier Loops 

**Authors**: Mahmoud Hegazy, Michael I. Jordan, Aymeric Dieuleveut  

**Link**: [PDF](https://arxiv.org/pdf/2610.06432)  

**Abstract**: Numerous agentic workflows are based on a generator-verifier loop: a generator proposes candidates, a cheap verifier scores them, and the workflow terminates when a proposal is verified as good enough. The verifier typically proxies a more costly ground-truth oracle, and as the generator searches adaptively against it, false acceptances may accumulate. Proposals can pass the proxy but fail under the costlier ground-truth check. We study when to stop these loops while controlling the false discovery rate of the accepted proposals. Our construction introduces tools of independent interest in distribution-free statistical testing and conformal risk control, including analysis of $e$-values constructed through index betting and a novel conformal risk control procedure for non-monotone losses. We validate the approach in synthetic settings and on a protein-design benchmark. 

---
# SoK: Semantic Decision Engines in Network Control Loops 

**Authors**: Delong Li, Chen Li, Xu Wang, Haochen Gong, Rui Lang, Guangsheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06425)  

**Abstract**: A semantic decision engine such as Jev can return a valid answer and still miss a network deadline, select an infeasible action or leave the service unverified. We systematize 139 paper families by decision interface, execution path and check ownership. Fifty families claim that their engine fits a control loop or time budget, but only four support the claim with matched measurement. Across all 139, four report deadline attainment. The gap concentrates where the decision has no deterministic computation step. Those 72 families make 22 of the claims, none supported, and name a coverage owner in only two. Bounded tests under one event model show that each gap can reverse an admission verdict. A decision that meets a 10 s budget for every isolated request meets it for none once decisions queue ahead of replayed execution times. The same engine passes one coverage check and fails another. We derive a minimum reporting record, design rules and a research agenda for admitting decision engines to control loops. 

---
# Toward Reliable Infant Pose Estimation: A Training-Dynamics Approach to Noisy Annotation Detection 

**Authors**: Emanuele Cardinale, Sara Moccia, Alessandro Cacciatore, Lucia Migliorelli  

**Link**: [PDF](https://arxiv.org/pdf/2610.06423)  

**Abstract**: Spontaneous movement analysis in preterm infants relies increasingly on markerless pose estimation (PE) to derive clinically relevant motion biomarkers directly from video recordings. Training accurate infant PE models requires large sets of manually annotated keypoints, and human annotation is inherently prone to error. Noisy keypoints (i.e., keypoints mislocalized with respect to their true anatomical position) are especially problematic in this clinical setting, since they can propagate as artificial artifacts into the reconstructed joint trajectories. Building on the small-loss hypothesis and training-dynamics-based sample selection established in the noisy-label learning literature, we propose a novel framework for detecting noisy keypoint annotations. A hybrid convolutional-attention model is trained to predict the anatomical category of each keypoint from its spatial coordinates and local visual features; the resulting cross-entropy training dynamics are then used to derive per-keypoint descriptors, which are partitioned into clean and noisy subsets via unsupervised clustering. We validate the approach on NeoPose, a newly collected dataset of 65 hospitalized preterm infants, under two realistic noise scenarios (random positional perturbation and left-right swapping) across multiple noise levels. Results show that the proposed approach achieves an F1-score of up to 91.9% in noisy-keypoint detection. The framework further generalizes to the heterogeneous COCO benchmark, where filtering CE-detected noisy keypoints from the training set also yields measurable improvements (up to 7.4 AP points) in downstream pose estimation accuracy at moderate-to-high noise levels. 

---
# SpatialChain: A Benchmark for Auditing Spatial Reasoning Faithfulness in VLMs 

**Authors**: Rafael Teixeira Sousa, Vinícius Paulo Lopes de Oliveira, Elisa Ayumi Masasi de Oliveira, Luiza Martins de Freitas Cintra, Fernanda Bufon Färber, Igor Dias Aguiar, Julia Yasmim de Almeida Nobre, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2610.06413)  

**Abstract**: Thinking-enabled vision-language models (VLMs) report ever-higher accuracy on spatial benchmarks, yet final-answer scores cannot reveal whether a correct prediction reflects faithful spatial reasoning or a linguistic shortcut. We introduce SpatialChain, a dataset of 28,350 training and 899 test examples pairing spatially-oriented GQA questions with scene-graph-grounded reasoning chains, retained only when the generated answer matches the symbolic ground truth, and a two-axis evaluation combining objective chain-overlap metrics with a scene-graph-aware LLM judge that scores faithfulness and completeness independently of the final answer. Applied to nine thinking-enabled VLMs, the protocol surfaces three findings invisible to standard accuracy: (i) four of nine models achieve $\geq$79% VQA accuracy while exhibiting shortcut rates above 39%, i.e., correct answers whose reasoning the judge marks as unfaithful; (ii) chain quality significantly predicts answer correctness for seven of nine models, but the two exceptions (Claude Sonnet 4.6, InternVL3.5-8B) reveal qualitatively distinct failure modes, terse output vs. verbose-decorative reasoning, that benchmark accuracy alone conflates; (iii) SFT on SpatialChain improves Qwen3-VL-8B by +6.2 pp in-domain and reduces its shortcut rate to 22%, while a stylistic specialization effect on external benchmarks motivates replay-augmented training as mitigation. The faithfulness judge is validated against 198 human-annotated items, where judge-human agreement matches human-human agreement, and against a second judge from a different provider, which preserves the model ranking ($\rho$ = 0.88). Data, generation scripts, and evaluation code are released at this https URL. 

---
# Quantifying the Stability of Multi-Step Reasoning via Error Amplification 

**Authors**: Dongyue Li, Ziniu Zhang, Minxuan Duan, Hongyang R. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06404)  

**Abstract**: We consider the stability of multi-step reasoning processes, which have extensive applications in language models, including chain-of-thought and algorithmic reasoning. While longer sequences of reasoning can improve a model's generation capability at test time, the errors due to intermediate reasoning steps can accumulate in autoregressive generation, and thus grow substantially at the end. In this paper, we ask: What are the key factors determining the stability of multi-step reasoning? First, we show an inference error bound governed by the product of spectral norms of the Jacobians taken through the input space across generation steps. This product can be viewed as an error amplification factor, which could scale exponentially with the number of reasoning steps, serving as a quantitative measure of reasoning stability. Second, we analyze this measure in transformer models trained to predict simple tasks like linear and quadratic functions. We theoretically prove that the transformer model converges to a solution where the stability measure decays, thus yielding nearly zero inference loss over (arbitrarily) long steps. Finally, the stability analysis leads to several algorithmic implications for controlling the stability, through (i) chain-of-thought length compression that reduces the sensitivity of each step, and (ii) quantization-aware training that regularizes the input Jacobian norms. We validate the proposed algorithms by fine-tuning language models on graph-algorithmic reasoning tasks and symbolic state-tracking tasks. Across seven evaluations, our algorithms improve over baseline comparisons by 3.5% on average, and by 8.2% for longer-length inputs. Ablation analysis validates that the stability measure is drastically reduced by 3-8$\times$, confirming the regularization effect on the spectral norms of the (input space) Jacobians. 

---
# RAISED: Self-Distillation for Robustness to Prompt Injection in LLM Agents 

**Authors**: Mohamed Dhouib, Clement Elliker, Alexi Canesse, Maël Jenny, Lucas-Andrei Thil, Mahammed El-Sharkawy, Sonia Vanier, Elie Bursztein  

**Link**: [PDF](https://arxiv.org/pdf/2610.06401)  

**Abstract**: Tool-using language-model agents are vulnerable to indirect prompt injection because they must act on untrusted external content. Existing training-time defenses can reduce attack success rates, but often at the cost of general capabilities. We show that training-based defenses induce substantial drift in the model's output distribution, altering its behavior even in benign settings and providing a potential mechanism for utility degradation. We further identify a failure mode of these defenses: On benign tool-use tasks, the model refrains from a step needed to finish an authorized task, particularly when that step is indicated by a tool output. To address these limitations, we introduce RAISED (Robust Attack Invariance through Self-Distillation), a training framework that combines self-generation and self-distillation. The model first generates its own tool-use scenarios, with an emphasis on cases where task completion requires acting on legitimate guidance from tool outputs. Then, through self-distillation, the student is trained to match the teacher's clean-context behavior on both clean and injected variants of the same trajectory. RAISED substantially reduces the attack success rate of prompt injections in tool responses while, unlike prior training-based defenses, preserving utility on both agentic and general-purpose benchmarks. 

---
# Scaling Down the Scaling Laws: Parameter Efficiency and Compute-Optimal Training in Resource-Constrained Large Language Models 

**Authors**: Joe Dwyer  

**Link**: [PDF](https://arxiv.org/pdf/2610.06387)  

**Abstract**: Large language models (LLMs) have achieved substantial performance gains through increases in model size, training data, and computational resources. However, traditional scaling approaches produce diminishing returns, rising financial and environmental costs, and barriers to participation for researchers operating outside large industrial laboratories. This review examines the evolution of LLM scaling theory from empirical scaling laws to compute-optimal training, with particular emphasis on parameter efficiency, token utilization, data efficiency, and resource-constrained environments. Foundational work on scaling laws is synthesized alongside later research on compute-optimal training, data pruning, efficient architectures, quantization, low-rank adaptation, and edge-oriented optimization. The literature indicates a shift from scale maximization toward more deliberate allocation of parameters, tokens, compute, and hardware resources. At the same time, important empirical, theoretical, and methodological gaps remain regarding whether scaling principles established on enterprise-grade infrastructure generalize to smaller models and constrained computing environments. This review organizes these developments into a unified framework for resource-efficient LLM training and argues that future progress should evaluate efficiency not solely through model performance, but through the relationship among performance, parameter count, computational cost, token allocation, and hardware constraints. 

---
# Steering by Influence: Curvature Aware Data Weighting for Activation Steering 

**Authors**: James A. E. Dixon, Stephen J. Roberts, Francesco Quinzan  

**Link**: [PDF](https://arxiv.org/pdf/2610.06383)  

**Abstract**: Inference-time steering offers cheap, fine-grained control over a language model's outputs by estimating a concept's representation in activation space and shifting activations towards it. Existing methods build these representations from activation averages over contrastive datasets. These averages incorporate unrelated concepts and noise, and are dominated by a few tokens, meaning the activation transport encodes token-level rather than thematic concepts. In this work, we steer towards examples that most express a concept thematically, rather than towards an expectation over all. We identify these examples using influence functions, which estimate how much each data point contributes to a model's representation of a concept. Unlike simple model activation similarity, they incorporate the curvature of the model's loss landscape, allowing them to capture concept-relevant relationships beyond superficial token-level similarity. We then propose influence-weighted activation transport, which uses optimal transport to steer activations of non-concept text towards those of concept text, weighting concept examples by their influence scores. We evaluate on toxicity suppression (Jigsaw), object-based concept induction (OneSec) and truthfulness induction (TruthfulQA), outperforming existing activation-transport baselines. We track capability after steering using perplexity and MMLU accuracy, finding that our method improves steering while largely preserving model quality. We further show that influence functions capture concept-relevant information that activation-based methods miss with the two approaches ranking data points significantly differently. Together, these results demonstrate the value of curvature-aware influence information for activation steering. 

---
# MeSD: Multi-Evidence Self-Distillation for VideoLLM 

**Authors**: Weijie Zhu, Han Fang, Hanyu Fu, Yuzhe Zhang, Xin Wei, Zhaoyan Pan, Feiran Liu, Xunjie Jin, Hongbo Sun, Zhiyu Lin, Tianyi Gao, Tianyi Ding, Ye Yuan, Zhongjiang He, Hao Sun, Zhiheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06342)  

**Abstract**: While reinforcement learning with verifiable rewards provides reliable outcome supervision for VideoLLMs, sequence-level rewards offer limited token-level guidance. On-policy self-distillation addresses this limitation by conditioning a self-teacher on privileged information to provide dense token-level supervision. However, aggregating heterogeneous evidence within a single teacher context obscures cross-evidence agreement and conflict. A further challenge lies in determining whether teacher guidance should refine reward-based updates or provide corrective supervision for failed trajectories. To address these issues, we propose MeSD, a multi-evidence self-distillation framework for VideoLLMs. MeSD constructs three evidence-conditioned teachers with shared parameters, using the ground-truth answer as a common semantic context while separately incorporating temporal and spatial evidence. Given the same student-generated prefixes, MeSD evaluates evidence-specific preferences relative to the Answer Teacher and fuses teacher-common preferences with gated teacher-specific residuals. Furthermore, MeSD introduces Verification-Guided Optimization to classify trajectories as Success, Failure, or Indeterminate. For Success and Indeterminate trajectories, MeSD refines token-level advantage magnitudes while preserving reward-derived signs. For verified failure trajectories that contain the required evidence, MeSD applies failure-conditioned distillation, using reverse-KL correction toward the fused distribution. Experiments on multiple video benchmarks demonstrate consistent gains over reinforcement learning and self-distillation baselines. 

---
# Fine-Tuning a 3B-Parameter LLM on a Smartphone: Characterizing Sustained Training 

**Authors**: Andrew Geyko, Marius Mosbach, André Brinkmann  

**Link**: [PDF](https://arxiv.org/pdf/2610.06325)  

**Abstract**: Multi-billion-parameter LLMs now run on phones for inference, and training them on the device would personalize them without user data leaving the phone. Prior work has measured individual training steps of such models on phones, but not complete training runs, and not whether adapters trained on the device improve personalization. We present the first systematic characterization of a multi-billion-parameter LLM fine-tuned on a mobile device, covering memory, per-step time, thermal behavior, and energy. An iPhone 17 Pro can fine-tune a 3B-parameter LLM to a typical user within one battery charge, and the resulting adapters improve personalization as much as adapters trained on a server. Sustained training throttles the phone to about half its initial throughput, and none of the pausing or burst schedules we tested recovers it. Nearly all of each training step is spent in the frozen base model, most of it in the backward pass, which nine of the ten other runtimes we audited do not accelerate. Apple's MLX had a kernel for it that was never dispatched and was incorrect, and our repair, now merged upstream, trains an adapter 1.47x faster on a third less energy. On-device fine-tuning is feasible on current phones, and making it efficient requires runtimes and operating systems to treat training as a first-class workload. 

---
# Agentic schema-guided extraction of materials process knowledge from scientific literature 

**Authors**: Sameer Sadruddin, Jennifer D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2610.06322)  

**Abstract**: Materials literature contains detailed experimental knowledge, but procedures, chemical entities and measurements remain difficult to aggregate because they are reported in heterogeneous forms and depend on process-specific context. We present SciKGExtract, a schema-guided framework that combines large-language-model extraction with chemical normalization and agent-based evaluation and refinement before knowledge-graph integration. We evaluate the framework on 176 atomic-layer-deposition papers describing zinc oxide (ZnO) and indium--gallium--zinc oxide (IGZO), together with an expert-annotated full-schema subset. PubChem normalization improves exact-match extraction F1 for every tested model. For ZnO, the best F1 increases from 0.591 for direct normalized extraction to 0.805 with agentic refinement, whereas the best IGZO result is 0.344, revealing the greater difficulty of multicomponent supercycle processes. Evaluation against a deeply nested schema containing 65 experimental properties and 155 quantitative measurement nodes further exposes errors in process segmentation and numerical assignment. These results show that chemical canonicalization and targeted agentic verification provide complementary controls for converting complex materials literature into reusable, machine-actionable experimental knowledge. 

---
# Wiring Matters: Injection Topology and Initialization of Affordance Heads in Vision-Language-Action Policies 

**Authors**: Zijian An, Linhan Wang, Jiayan Wang, Shijie Geng, Ran Yang, Yiming Feng, Lifeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.06318)  

**Abstract**: Dense affordance supervision is an appealing auxiliary signal for vision-language-action (VLA) policies, yet naively co-training an affordance head can severely damage instruction following. We present a controlled study of how to wire such a head into a modern VLA on the LIBERO benchmark. Our recipe reads the backbone through a stop-gradient and re-injects an intermediate head feature into the action expert via a learned bridge. The stop-gradient is a precondition: letting affordance gradients reach the backbone drops the policy below the headless base (85.5% vs. 93.1%). With the backbone protected, a same-budget 2*2 ablation over injection topology (concatenation vs. residual) and bridge initialization (zero vs. random) shows initialization is the dominant lever. The best wiring, an actively initialized residual bridge, reaches 96.2%, matching the far more elaborate three-expert AffordanceVLA (95.8%) with under 1% extra parameters. Two probes explain the mechanism: ground-truth affordances fed as an input hurt, and inference-time zeroing shows a lazy bridge acts only as a training-time regularizer while an active bridge becomes load-bearing. 

---
# DialectSentEval 2026: Arabic Dialect Sentiment Analysis and Swapping Shared Task 

**Authors**: Saad Ezzini, Shadi Abudalfa, Maram Alharbi, Salmane Chafik, Hind Alatawi, Mo El-Haj, Ahmed Abdelali, Osamah Alnahari, Salima Lamsiyah  

**Link**: [PDF](https://arxiv.org/pdf/2610.06298)  

**Abstract**: Sentiment analysis is a fundamental problem in Natural Language Processing (NLP). Standard sentiment classification for the Arabic language remains challenging due to the high volume of dialectal Arabic. To advance research in this area, this paper proposes the Shared Task on Sentiment Analysis and Swapping in Arabic Dialects (DialectSentEval), hosted with the Arabic Natural Language Processing Conference (ArabicNLP 2026). This shared task consists of two subtasks: Subtask 1 focuses on multi-class and multi-dialect sentiment analysis, requiring models to identify sentiment polarity across various Arabic dialects. Subtask 2 introduces a generative task for Arabic sentiment swap, challenging models to invert sentiment polarity while preserving core semantics. In this overview paper, we present the motivation, dataset creation, and summarize the main findings from participating models. 

---
# GAMBIT: Learning to Plan Continuous Multi-Robot Trajectories 

**Authors**: Rishabh Jain, Akmaral Moldagalieva, Lorenzo Magnino, Michael Amir, Keisuke Okumura, Ajay Shankar, Wolfgang Hönig, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2610.06290)  

**Abstract**: GAMBIT is an opening chess move in which a player sacrifices a piece, typically a pawn, to gain a positional advantage later in the game. Analogously, in multi-robot coordination, individual robots may need to forgo locally reward-maximising behaviours to improve overall team performance. Such self-sacrificial behaviours are difficult to capture with manually designed heuristics, particularly in dense, interaction-rich environments. Focusing on double-integrator continuous dynamics, this work studies how to learn such coordinated heuristics over motion primitives for multi-robot trajectory execution. Our framework, GAMBIT, first learns coordinated motion-primitive selection through imitation learning and subsequently fine-tunes the policy through reinforcement learning. We further introduce a safeguarded rollout mechanism with backup trajectories that guarantees collision-free execution at all times. Experiments demonstrate that GAMBIT substantially outperforms a range of baselines, including centralised motion planners and decentralised reactive planners, while exhibiting strong scalability. In particular, it coordinates over a thousand robots with planning latency below a few hundred milliseconds in continuous domains. 

---
# When Are Concept Bottleneck Model Explanations Faithful and Compact? 

**Authors**: Stefano Teso, Emanuele Marconato, Steve Azzolin, Antonio Vergari  

**Link**: [PDF](https://arxiv.org/pdf/2610.06285)  

**Abstract**: Concept bottleneck models (CBMs) are neural classifiers that allow to explain their decisions via high-level concepts, potentially enabling understanding, steering and debugging. However, their explanations are often derived heuristically. Building on formal explainability, we argue they should also be faithful, i.e., not misreport which concepts actually matter. We show that, for widespread CBM architectures, including recent VLM-based variants, faithful explanations must include all concepts in the bottleneck, compromising interpretability when this is large. This result applies to both heuristic and faithful-by-construction formal explanations. To encourage the existence of compact faithful explanations, we suggest i) modeling concepts probabilistically as binary or categorical random variables (rather than logits), and ii) employing per-concept training-time sparsification via group lasso (rather than regular elastic net). We also extend algorithms from formal explainability to CBMs, and show they outperform natural heuristics in terms of guarantees and explanation size. Overall, our work warns against naive interpretability claims and provides formal conditions and practical strategies for ensuring CBMs are as interpretable as advertised. 

---
# Future Anchored Verification and Online Recovery for World Action Models 

**Authors**: Zhibin Qin, Zhenxiong Tan, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06280)  

**Abstract**: World action models (WAMs) have emerged as a promising paradigm for robotic manipulation. They act by first predicting how a task should be performed and then decoding the actions from that future. However, the remaining actions are invalid once execution drifts from the prediction. Simply replanning from the already out of distribution state rarely restores what the task still requires; existing execution monitors decide when to stop, but not what to restore. We observe that the answer is already in hand: the future the WAM predicted before acting depicts exactly the states it intended to pass through. We introduce FAVOR (Future Anchored Verification and Online Recovery), a lightweight framework that keeps these predicted frames as anchors and uses them for verification and recovery. An Anchor Verifier compares each observation with its anchor, together with the executed actions, to flag deviations that break the task. Anchor-Guided Recovery uses a vision-language model to turn the flagged anchor into a short corrective instruction. Under strengthened instruction guidance, the WAM executes this instruction to return to the intended future. It then resumes the task. FAVOR raises the task success of the base WAM from 97.85% to 98.10% on LIBERO and from 72.60% to 72.98% on LIBERO-Plus without modifying the policy. 

---
# VLA-ZO: Fast Zeroth-Order Adaptation for Vision-Language-Action Models 

**Authors**: Jaemin Kim, Jiahn Kim, Taesik Gong  

**Link**: [PDF](https://arxiv.org/pdf/2610.06271)  

**Abstract**: Adapting vision-language-action (VLA) models to deployment-time distribution shifts is important for reliable robotic operation, but conventional first-order adaptation can exceed the memory budget of inference-oriented deployment platforms. Zeroth-order (ZO) optimization offers a forward-only alternative with inference-level memory, but accurate gradient estimation requires many perturbation queries, making naive ZO prohibitively slow for large VLA models. We present VLA-ZO, a framework for fast ZO adaptation that exploits the structure of VLA computation. By confining adaptation to the action side, VLA-ZO keeps the expensive vision-language prefix frozen and reuses its conditioning states across perturbation queries and optimizer steps, while schedule-aware prefetching hides state-transfer overhead. On LIBERO camera-viewpoint shifts, VLA-ZO reduces end-to-end adaptation time by 25.59$\times$ at $q=16$ and 32.54$\times$ at $q=64$ relative to baseline ZO, while improving average task success from 48.27% without adaptation to 58.17% and 63.58%, respectively. These results show that making ZO faster can make larger query budgets practical, providing a promising path toward resource-efficient VLA adaptation on deployment platforms. 

---
# Ramp Metering Control via Hybrid State Deep Reinforcement Learning in Partially Observable Connected Vehicle Environments 

**Authors**: Youcef Mehamlia, Nadir Farhi, Meriem Bouali  

**Link**: [PDF](https://arxiv.org/pdf/2610.06266)  

**Abstract**: Freeway on-ramp merges are major sources of congestion, causing significant economic and environmental costs. While Deep Reinforcement Learning (DRL) offers a promising solution for ramp metering, existing approaches rely primarily on aggregated macroscopic data. Connected vehicles (CVs) provide vehicle-level observations that can complement aggregate traffic measurements, but their limited penetration produces incomplete microscopic information. This paper proposes a hybrid observation representation combining macroscopic traffic measurements with a two-channel grid encoding observed CV presence and speed. A Dueling Double Deep Q-Network processes these inputs to select ramp-metering green durations. The controller is trained under varying traffic demands and CV penetration rates and evaluated against ALINEA and macroscopic-only DRL variants in SUMO. Across 50 matched evaluation scenarios, the hybrid controller under partial CV visibility reduces the reported total travel time by 11.4 % and mean spillback duration by 84.9 % relative to ALINEA. Evaluating the same trained policy with full CV visibility yields a further travel-time reduction of approximately 1.6 %. Analysis across penetration rates suggests that the performance gap decreases as microscopic observations become more complete. These results support the use of complementary macroscopic and sparse microscopic observations for learning-based ramp metering. The source code implementation of the model is available at: this https URL 

---
# CentriQ: Calibration-Free Quantization of Diffusion Transformers via Exact Mean Centering 

**Authors**: Nataša Jovanović, Mathieu Salzmann, Saqib Javed  

**Link**: [PDF](https://arxiv.org/pdf/2610.06260)  

**Abstract**: Diffusion transformers (DiTs) achieve state-of-the-art image generation, but their sampling cost limits deployment. Quantizing both weights and activations to 4 bits reduces this cost, yet existing methods fall short in one of two ways. Calibration-based methods are tied to a specific checkpoint and prompt distribution, whereas data-free Hadamard rotation, effective for LLMs, loses quality on DiTs. We show that this loss has a structural cause. Adaptive layer-norm conditioning adds a per-token mean to the activations, and at the widths of the evaluated DiTs, the Hadamard rotations used by data-free methods cannot spread this mean uniformly across coordinates. A single dominant direction therefore survives the rotation and sets the quantization range. We introduce CentriQ, a calibration-free quantizer that centers each token before rotation and restores the mean exactly through a rank-1 full-precision branch, so that per-token scales follow in closed form without data. Weights are fitted under a robust $\ell_p$ objective that tracks the dense mode of each group and discounts heavy tails. Across three DiTs, CentriQ matches the quality of calibrated SVDQuant at 4 bits, whereas calibration-free weight quantizers with plain per-token activation quantization collapse or degrade substantially. CentriQ outperforms the strongest calibration-free method reported to date at 2-bit weights. It is also the first calibration-free method to retain usable image quality at 2-bit activations. 

---
# Few-Shot Prototype Head Adaptation for On-Device ECG Personalization on PSoC~6 

**Authors**: Guilherme Silva, Pedro Silva, Gladston Moreira, Eduardo Luz  

**Link**: [PDF](https://arxiv.org/pdf/2610.06241)  

**Abstract**: Wearable and bedside electrocardiogram (ECG) monitors must adapt to patient-specific morphology to maintain arrhythmia detection accuracy across users, yet personalization is typically performed offline and cannot account for individual physiology, electrode placement, or recording drift. On-device adaptation by backpropagation is expensive for microcontroller-class medical devices because it requires an optimizer state, repeated backward passes through convolutional layers, and labeled arrhythmic beats that may not be available at deployment time. This letter proposes prototype-only head adaptation as a compact personalization primitive for TinyML ECG systems. A one-dimensional convolutional neural network (1-D CNN; 1,314 parameters and 72.6k multiply-accumulate operations per beat) is trained offline on the MIT-BIH Arrhythmia Database under an inter-patient protocol, frozen as a feature extractor, and exported to a PSoC 6 microcontroller. Patient-specific adaptation then reduces to computing closed-form class means in a 32-dimensional embedding space, requiring no convolutional backward pass, no iterative optimization, and only one forward pass per support beat. Prototype adaptation improves inter-patient macro-F1 from 0.635/0.639/0.646 to 0.731/0.771/0.797 at 1/5/10-shot, outperforming linear stochastic-gradient-descent (SGD) head fine-tuning at every shot count for the target tiny backbone. On-device replay over 18 one-shot episodes on a PSoC 6 Cortex-M4F matches the host macro-F1 for the prototype head (0.798), with 11.39 ms per beat, 5.2 KB flash, and 22.2 KB SRAM. A restricted variant that updates only the normal-class prototype from passively buffered sinus beats yields a consistent +0.05 macro-F1 gain, reducing the annotation burden during initial 

---
# Artificial Intelligence and the New Science of Culture 

**Authors**: Douglas R. Guilbeault, Bhargav Srinivasa Desikan, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2610.06240)  

**Abstract**: Cognition and culture have long been treated as parallel objects of study, joined more by metaphor than by mechanism. We argue they are co-constituted in a manner now empirically tractable: subjectivities can be measured as local trajectories through a high-dimensional cultural field, and the field is itself the aggregate of those trajectories. Recent advances in machine learning, including embedding methods and large generative models, provide the first general framework for measuring this co-constitution from micro-cognition to macro-social structure. Vector geometry recovers individual conceptual structure, organizational communication, and large-scale ideologies; captures multimodal cultural content beyond text; and generates testable predictions about cultural emergence, including the simultaneous arrival of independent discoveries across distant minds. We treat this predictability of simultaneous innovation as evidence for the co- constitutive view: when the geometry of the cultural field is measurable, trajectories of cognitive search through it become predictable. We then examine generative AI agents as simulators of human subjectivity and as a qualitatively new coordinating substrate whose insertion into social life introduces evolutionary dynamics unprecedented in human cultural history. We formalize this reflexive condition as a Heisenberg-like Uncertainty Principle for generative AI: as instruments for fixing a culture's position grow more precise, our capacity to predict its trajectory degrades, because the machinery of cultural description and cultural action have merged. We close with ethical questions raised by AI-mediated cultural drift, recursive synthetic culture, and the limits of human cultural agency, offering guidelines for safe, equitable, and privacy-preserving research on AI and culture. 

---
# Joint Class-Time Learning for Video Classification with Multi-Instance Partial-Label Learning 

**Authors**: Lingyu Shen, Wei Tang, Fakhri Karray, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06234)  

**Abstract**: Multi-instance partial-label learning (MIPL) addresses inexact supervision in both the instance and label spaces, which can be applied to video classification. However, bag-level labels do not explicitly supervise the correspondence between candidate classes and temporal evidence. We propose {\ours}, which couples label disambiguation with temporal evidence allocation through a joint class--time assignment. Occupancy-regularized spherical matching associates contextualized video features while learning nonuniform temporal mass and discouraging excessive concentration. During training, candidate-restricted inference recomputes the assignment within the candidate label set. A dual-marginal KL projection then constructs a structured teacher that incorporates momentum-refined class beliefs while preserving the proposal's temporal occupancy. A single plan-level KL objective aligns the full-space predictor with this teacher. Our analysis characterizes when candidate re-solving differs from masking and shows that, under the stated construction, the joint objective decomposes into class-marginal and class-conditional temporal supervision. We construct VCMIPL benchmarks from Breakfast, DoTA, and FineAction using model-generated candidate labels and evaluate the method across four feature representations. Extensive experimental results demonstrate that PIVOTMIPL outperforms existing MIPL algorithms in both effectiveness and efficiency. 

---
# SO(3)-RoPE for Spherical Transformers 

**Authors**: Christian Libner, Chase van de Geijn, Alexander S. Ecker, Maurice Weiler  

**Link**: [PDF](https://arxiv.org/pdf/2610.06229)  

**Abstract**: Spherical data arise in many scientific applications. Often spherical transformers disregard the geometry of the underlying spherical domain, causing distortions and coordinate singularities near the poles. We introduce SO(3)-RoPE, a relative positional embedding that incorporates spherical geometry into transformer attention through unitary SO(3) representations. Our formulation is SO(3)-equivariant and compatible with FlashAttention, retaining efficiency of vanilla transformers. On shallow water dynamics prediction over a rotating sphere, our SO3ViT outperforms an S2Transformer baseline with lower errors and reduced runtime. 

---
# APOD: reasoning-guided agentic population ordinary differential equation discovery for pharmacological digital twins 

**Authors**: Romain Ferrara, Martin Soucail, Victor Gertner, Adil Moussali, Joris Cocquebert, Sandrine Oziel-Taieb, Julien Nicolas, Florence Gattacceca, Mihaela van der Schaar, Sébastien Benzekry  

**Link**: [PDF](https://arxiv.org/pdf/2610.06227)  

**Abstract**: Establishing ordinary differential equations (ODEs) describing population data is a fundamental part of mathematical modeling in pharmacology, crucial to developing digital twins. However, doing so from sparse, noisy data is a slow, expert-driven task. Existing automated methods either search a restricted model space or ignore population inter-individual variability. Here we introduce APOD (Agentic Population ODE Discovery), a language-model agent that iteratively reasons over biological knowledge and fit diagnostics in an open-ended search space to discover a population digital twin (PDT), i.e., a shared ODE system with between-subject variability. On synthetic pharmacokinetic and tumor-dynamics benchmarks, APOD recovered ground-truth structures in 94-100\% of runs, 12-fold faster in median than an established library-based search. On real cohorts it converged to valid structures, and proposed a PDT of radioligand-therapy-induced platelet dynamics that predicts thrombocytopenia from first-cycle data and simulates alternative dosing schedules that lower the predicted risk of toxicity. 

---
# Cross-lingual Calibration of Pre-Generation Success Probes for Multilingual LLM Routing 

**Authors**: Andrea Paganelli, Stefano Civelli, Pietro Bernardelle, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2610.06216)  

**Abstract**: Pre-generation success probes estimate response correctness from a language model's hidden activations before decoding, enabling cost-aware routing. While prior work has demonstrated their utility primarily on English inputs, we study their reliability across languages along three dimensions: (1) whether they preserve the ranking of likely successes and failures (DISCRIMINATION); (2) whether they retain probabilities that match observed success frequencies (CALIBRATION); and (3) whether they produce scores comparable enough across candidate models for cost-aware multilingual routing (UTILITY). Using 3,000 MATH problems in 10 languages and 8 open-weight model configurations, we compare cross-lingual transfer from English-trained probes and equal-budget pooled multilingual probes. English-trained probes retain useful cross-lingual discrimination but become less well calibrated after transfer. Pooled multilingual supervision improves both properties and yields more reliable estimates of success. In routing experiments, the pooled router achieves a 0.7% higher test success rate while reducing modeled cost by 13.0% relative to always selecting the model with the highest average success. These results show that multilingual routing requires success estimates that remain well calibrated and comparable across languages and models. 

---
# Correct Code, Broken Contributions? SWE-CC: Benchmarking Repository Policy Compliance for Coding Agents 

**Authors**: Hai Dang Truong, Rayner Goh, Thanh Le-Cong, Yintong Huo  

**Link**: [PDF](https://arxiv.org/pdf/2610.06193)  

**Abstract**: Autonomous coding agents now resolve a substantial share of real-world GitHub issues. However, passing functional tests differs fundamentally from producing a high-quality contribution acceptable for merging. Mature open-source projects publish repository-specific contribution policies, spanning style, git, testing workflows, to ensure code quality and long-term maintainability. Because existing benchmarks evaluate patches solely on unit tests, agent compliance with repository governance remains unknown. In this paper, we introduce SWE-CC, a benchmark evaluating code and process compliance in autonomous software engineering. We develop a semi-automated pipeline that converts developer documentation across 12 open-source repositories into 823 machine-checkable atomic policies. SWE-CC introduces two features: 1) lightweight, deterministic checker functions that represent each policy, 2) a comprehensive auditing mechanism that inspects both agent runtime behaviors and final deliverables. We evaluate the compliance of agent workflows in 500 end-to-end software contribution tasks extended from SWE-bench Verified. Our evaluation of four LLMs under two agent scaffolds shows that modern agents suffer from coding compliance issues: although agents produce functionally correct patches, they still violate 43.1 percent of applicable project policies, with nearly half of all violations occurring during intermediate execution steps. These results show that functional correctness does not guarantee real-world readiness, highlighting that future software engineering agents must reliably conform to repository governance to enable safe and trustworthy deployment. 

---
# Anlu: Enabling In-Context Time Series Anomaly Detection in Foundation Models via Counterfactual Supervision 

**Authors**: Tian Lan, Yifei Gao, Yimeng Lu, Xuming An, Meng Wang, Yue Pan, Wenjun He, Chen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06180)  

**Abstract**: Whether a time-series pattern is anomalous often depends on the operating regime of the monitored process. A missing event can signal a fault in one regime and be routine in another, and the query alone may not reveal which regime applies. We study in-context learning (ICL) for time series anomaly detection (TSAD) through reference-conditioned detection, where a reference record provides evidence about expected behavior and model parameters remain fixed at inference. Supplying the reference is not enough: when training anomalies are recognizable from the query alone, the detector can fit its targets while ignoring the reference. We therefore introduce counterfactual supervision, which pairs one query with two references that support different normal rules and labels the query under each. At positions where the two labels disagree, no detector that ignores the reference can fit both targets. Anlu learns from this supervision by adding a reference memory and zero-initialized gated adapters to a frozen time-series foundation model (TSFM) pretrained for anomaly detection. On the 350 TSB-AD-U evaluation sequences, Anlu raises the mean VUS-PR of the frozen TSFM from 0.542 to 0.607. Replacing the reference with zeros lowers Anlu's score to 0.499. 

---
# Anosognosia in LLMs: Probing Self-Awareness of Quantized Computational Substrate 

**Authors**: Yoshihiro Izawa, Gouki Minegishi, Yoko Yamakata  

**Link**: [PDF](https://arxiv.org/pdf/2610.06174)  

**Abstract**: Can LLMs recognize degradation in their own computational substrate? Inspired by anosognosia, a neurological condition in which patients fail to recognize impairments in their own abilities, we investigate whether LLMs can recognize degradation in their computational substrate induced by quantization. We first show that existing models fail to self-report their quantization state, even when provided with their own generated text as an external cue. Linear probing reveals that, while generated text carries almost no trace of quantization, internal representations contain clear, method-specific fingerprints. Through training, models learn to identify severely degraded outputs such as those of 4-bit models by comparison, yet still fail to do so from a single output. A shared LoRA trained jointly across quantization levels succeeded in reading out internal fingerprints, but fails on unseen quantization methods, merely mapping method-specific fingerprints to labels. Whereas external self-observation can restore awareness in some cases of human anosognosia, our results suggest that the more promising route to enabling such awareness in LLMs may lie in their internal representations. Our results highlight fundamental limits of generalizability to LLM self-monitoring. 

---
# Where Did the Repair First Go Wrong? Localizing the Origins of Silent Failures in Agentic Vulnerability Repair 

**Authors**: Wenji Bai, Muhammad Waseem, Zeeshan Rasheed, Jaakko Peltonen, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2610.06163)  

**Abstract**: Localizing where an LLM-based agent first fails to uphold security during a repair can show which stage of its workflow needs an additional safeguard. This is difficult for silent failures, which are patches that pass syntactic and functional checks but still contain a security vulnerability. Because such patches give no observable failure signal, existing failure attribution methods, which rely on observed task failures and labelled failure steps, are less suited to them. We propose Security Awareness Gap Evaluation (SAGE), a trace-based method that combines an assessment of the security reasoning recorded at each turn with the reconstructed code history to identify the earliest turn at which a repair diverges from the task's security intent. We evaluate SAGE on 95 confirmed silent failures drawn from 3,684 repair traces produced by six agent frameworks and six base models on SecurityEval and CVEfixes. SAGE assigned an origin in 93 cases. Most origins were an unaddressed security requirement or an inadequate defence choice, and only five coincided with the code change itself. When the agent introduced the vulnerable code, the origin preceded the write in 14 of 19 cases. Repeated scoring and a second judge reproduced the origin type more consistently than the exact turn, and agreement was lowest for traces that kept only the final file. 

---
# Machine learning for journal entry testing: A type-aware evaluation of anomaly detectors under a review budget 

**Authors**: Jan Gronewald, Michel Scherer, Nijat Mehdiyev  

**Link**: [PDF](https://arxiv.org/pdf/2610.06133)  

**Abstract**: Journal entry anomaly detectors are commonly evaluated on the full population with ROC-AUC, precision and recall, ignoring the review budget and which anomaly types are found. We propose a type-aware evaluation combining per-type recall, fair-share type recall (FSR), which caps each type's credit at its budget share, type coverage and first-hit rank. We evaluate nine unsupervised detectors, a supervised reference and feedback-driven Deep Semi-Supervised Anomaly Detection (DeepSAD) on four real client ledgers with injected typed anomalies and a public synthetic ledger. On the largest client ledger, principal component analysis (PCA), an autoencoder (AE) and a variational autoencoder (VAE) each place on average 98 anomalies among the first 100 postings, but at least 95.8 belong to one type. FSR instead favours a nearest-neighbour (kNN) detector and changes the top-ranked detector on three of four client ledgers. Representation also matters: one-hot encoding exposes unseen accounts, whereas frequency encoding leaves unseen contra accounts largely undetected. On the public ledger, the Histogram-Based Outlier Score (HBOS) and Empirical Cumulative Distribution-Based Outlier Detection (ECOD) reach all eight markings within 1,386 entries, whereas kNN, the hit leader at 1,000 entries, first reaches cross-linked clearing at rank 4,641, and the supervised row-level reference misses this marking within 1,000 entries. There, the adaptive DeepSAD review protocol raises mean hits per 100 reviews from 40.0 to 68.3 but type coverage only from 2.7 to 3.0. These findings show that high hit rates can conceal systematic blind spots and suggest that feedback can reinforce existing detection patterns without broadening anomaly coverage. 

---
# On the Geometry of Multimodal Saturation: Riemannian VICReg 

**Authors**: Nessim Ben Abbes, Duc Han Le, Sabri Mtibaa, Van-Tam Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2610.06096)  

**Abstract**: In self-supervised learning, a third modality should improve, or at least preserve, performance. Across nine image-text-tabular datasets, we show that it instead harms performance: the trimodal model underperforms its own best bimodal subset in 55.6% of paired runs under VICReg. The same failure occurs in 51.1% of paired runs under SimSiam. We call this failure multimodal saturation. We propose that the failure lies in the alignment geometry. Riemannian VICReg (R-VICReg) generalizes classical VICReg: it aligns views by squared geodesic distance on learnable negative-curvature product factors and recovers VICReg exactly as curvature vanishes. Over the same 45 paired runs, R-VICReg raises the probability that the third modality helps from 44.4% to 64.4%, with gains concentrated where VICReg saturates. 

---
# Cross-Lingual Transferability of Training Data Extraction Attacks to Recover Memorized PII 

**Authors**: Alexandru Nazare, Agnese Profico, Nicolò Vania, Elena Di Croce, Daria Caramanica, Davide Venditti, Elena Sofia Ruzzetti, Giancarlo A. Xompero, Fabio Massimo Zanzotto  

**Link**: [PDF](https://arxiv.org/pdf/2610.06093)  

**Abstract**: The robustness of Personally Identifiable Information (PII) protection in Large Language Models (LLMs) is a critical concern, yet the risks associated with cross-lingual data extraction remain under-explored. This study evaluates the vulnerability of English-centric and multilingual models to Training Data Extraction (TDE) attacks when prompted in non-English languages.
We construct a multi-domain PII dataset comprising social media handles, email addresses, and phone numbers and translate the attack contexts into Italian, Spanish, French, and German. Our results show that TDE attacks against both English-centric and multilingual models transfer to different languages: the attacks are successful on translated prompts, even though only the original English prompt might have been included in the pre-training data. A web-presence check on a sample of the translations confirms that they are not available online. The share of English leaks recovered in other languages grows with the multilingual capability of the model, and it drops sharply when the original wording is lost, even without a change of language.
This suggests that native multilingual pre-training facilitates the emergence of latent cross-linguistic bridges that simplify the retrieval of personally identifiable information (PII). We analyze the activations of multilingual large language models (LLMs) and find that different translations of the same prompt are bridged in similar representations, with the strongest alignment in the middle layers.
Our results highlight a fundamental security gap in modern LLMs, necessitating more robust, language-agnostic sanitization strategies for future model alignment. 

---
# Adaptive Mean Flow for Responsive Closed-Loop Robot Control 

**Authors**: Aksel Vaaler, Marco Job, Christian Holden, Olav Egeland  

**Link**: [PDF](https://arxiv.org/pdf/2610.06089)  

**Abstract**: Diffusion- and flow-based robot policies have recently become widespread in robotic Imitation Learning (IL) due to their high performance and ability to model continuous and multimodal distributions. However, the iterative denoising procedure used by these models introduces significant prediction latency, hindering high-frequency closed-loop robot control and leading to jittery, unstable motion when frequent updates to the robot's action predictions are used. Therefore, it is common practice to train models to predict chunks of actions that can be executed sequentially without feedback, even when this reduces responsiveness and may mean the most recent state information is not used. In this article, we present Adaptive Mean Flow (AMF), a flow-based IL method that enables smooth and responsive, fully closed-loop robot control. AMF uses Mean Flow, which is an accelerated form of Flow Matching (FM), to minimize prediction latency. To ensure smoothness and consistency across predictions, AMF uses a corrupted version of the trajectory from the previous step when predicting new robot actions, with the signal-to-noise ratio increasing over the time parameter of the trajectory. This discourages large changes in the prediction from one step to the next, while allowing freedom to adapt the predictions for future steps. We evaluate AMF across a wide range of simulated and real robot tasks and demonstrate significantly improved performance compared with baselines. Code: this https URL. 

---
# Boosting Transferable Adversarial Attacks against Deep Reinforcement Learning 

**Authors**: Zexin Li, Ruili Yao, Yiming Zeng, Xiaoxue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.06083)  

**Abstract**: Most adversarial attacks on deep reinforcement learning (DRL) assume white-box access to the victim policy, which rarely holds in practice. This paper studies transfer-based black-box attacks on DRL: the attacker crafts observation perturbations on a white-box surrogate agent and feeds them to an unknown victim. We formulate the attack as return minimization under a per-step perturbation budget. We first show that transplanting transferable image-classification attacks (FGSM, MI-FGSM, and NI-FGSM) with a per-step objective yields perturbations that transfer but are no stronger than random noise of the same budget. We then propose a trajectory-level attack that optimizes a sequence of perturbations over a receding horizon through a differentiable model of the environment and a temperature-smoothed surrogate policy, with the same optimizers. On CartPole-v1 with ten DQN and DDQN agents and 100 surrogate--victim pairs, the trajectory-level attack outperforms per-step attacks and random noise in the white-box, cross-model, and cross-algorithm settings. 

---
# Do VLAs Understand and Adapt to the Objects They Handle, or Simply Replay Learned Behaviors? 

**Authors**: Xinnuo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06078)  

**Abstract**: This paper asks whether VLA generalization is grounded in a global understanding of objects' physical properties that enables policies to adapt their motion to unseen setups, or if policies simply replay the motions they've learnt that happen to succeed in new setups. The former reflects genuine generalization; the latter reflects incidental robustness. We first examine awareness of physical properties in seven VLAs by applying linear probing and representational similarity analysis (RSA) to their activations. We find that physical properties, including mass, fragility, deformability, friction and size are less decodable than non-physical properties such as semantic category, material, sound and price in nearly every modality stream. Compared with their base VLMs, robot pre-training weakens the linear encoding of physical properties in the language stream. Neither pre-training nor downstream fine-tuning strengthens the alignment between physical-property differences and activation distances. We then ask whether the weak physical information present in these activations shapes the actions a VLA generates. In a controlled LIBERO case study, we increase the mass of an in-domain object and signal the change through language or vision. Most VLAs use similar lifting behaviour for the heavier and original-mass objects, leading to task success declines. The few exceptions change their behaviour in response to lexical or visual cues rather than to mass itself. These results suggest that VLAs encode physical properties weakly and do not reliably use them to adapt their motion. 

---
# Attention Tax, Handoff Tax: A Stylised Model of When Multi-Agent LLM Systems Help 

**Authors**: Akshit Anchan, Nayonika Sen  

**Link**: [PDF](https://arxiv.org/pdf/2610.06069)  

**Abstract**: Recent work on multi-agent LLM systems reaches sharply different conclusions: some results show that a single agent with the same information and compute should dominate a delegated system, others that multi-agent gains grow with task depth. We argue that much of the disagreement comes from modelling different bottlenecks, and introduce a stylised reliability model built around two trade-offs. Decomposition reduces the burden of long contexts but incurs a handoff tax when information is compressed or transferred between agents. Redundancy gains from multiple samples, but its benefit depends on how much their failures are shared. With reasoning budget, verification, and task structure added, the model yields two crossover conditions: decomposition becomes preferable once the attention cost avoided by resetting context exceeds the handoff cost, and parallel sampling at equal budget is eventually preferable when its shared-failure floor lies below the error floor of one agent thinking longer. We connect these regimes to recent theoretical and empirical results. On a ledger-reconciliation task we measure the context-degradation curve and the handoff tax from single-agent and handoff runs alone. From these the model places the crossover at depth 10 and predicts decomposition to win at depths 20, 50, and 100. It does, on step-level and final-balance accuracy, and the decomposed system's success, which the prediction never sees, lands within 9 percentage points of the predicted rate at every depth. 

---
# Vision Transformer Ensembles for Panoramic Street Segmentation 

**Authors**: Yunus Serhat Bıçakçı  

**Link**: [PDF](https://arxiv.org/pdf/2610.06063)  

**Abstract**: Semantic segmentation of street panoramas can support detailed descriptions of urban environments, yet small datasets and unequal training costs make model selection difficult. This paper presents the system used for a first place submission to the PalmCity challenge in the leaderboard snapshot dated 5 October 2026. Nine pretrained segmentation systems are compared using approximately equal computation budgets. The candidates include DeepLabV3+, SegFormer, UPerNet, Mask2Former, DINOv3 with a linear decoder, and an Encoder only Mask Transformer using DINOv3. The two leading candidates are trained independently with three random seeds and longer budgets. Equal averaging of class probabilities from the three Encoder only Mask Transformer models, evaluated at three image scales with horizontal reflection, produces 60.95% mean intersection over union and 71.16% mean F1 on the 84 image public validation split. The submitted predictions receive 57.08% mean intersection over union and 67.96% mean F1 on the hidden test leaderboard. Producing all 249 test masks takes 251.49 seconds including model initialization and provenance checks on one NVIDIA RTX 5090. Peak allocated GPU memory is 2.70 GiB. The study reports all eligible models, all inference variants, class level errors, source conditions, and reproducibility checks, providing a documented challenge workflow with existing architectures. 

---
# A Comprehensive Objective Evaluation of Modern Text-to-Speech for Turkish Using Speech Quality Assessment Models 

**Authors**: Yunus Emre Ozkose, Alperen Kahraman, Ali Haznedaroglu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06057)  

**Abstract**: Modern text-to-speech (TTS) systems can clone a target speaker from a short reference clip or be fine-tuned on a target voice, yet their behaviour on morphologically rich, lower-resource languages such as Turkish remain under-characterised. We present a systematic benchmark of four contemporary systems (Chatterbox, CosyVoice, OmniVoice, and VoxCPM2) evaluated across fine-tuned and zero-shot configurations, contrasted with a conventional VITS baseline and anchored to natural gold speech. Each configuration is scored with eighteen complementary objective metrics spanning learned naturalness predictors (UTMOS v2, DNSMOS-Pro, SCOREQ, WhisQA, AudioBox-PQ, NatScore, SpeechLMScore), intelligibility and signal-quality estimators (SQUIM PESQ/SI-SDR/STOI, Brouhaha), speaker similarity, distributional fidelity (TTSDS) and low-level acoustic descriptors. We further analyse how quality varies with utterance length and quantify long-form temporal consistency through speaker-identity and naturalness drift over chunked utterances. We release our evaluation code to support reproducible TTS evaluation. 

---
# ROT: Rotating Hidden States towards Contextual Vectors for Hallucination Mitigation in LVLMs 

**Authors**: Yijing Du, Xiangcheng Zhan, Shuo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06056)  

**Abstract**: Large Vision-Language Models (LVLMs) frequently suffer from object hallucination. Existing training-free interventions primarily manipulate attention weights, which indirectly affect the deep semantics reaching the final predictive layers. In this work, we shift our focus to the hidden state vectors extracted after self-attention and residual addition. Empirical analysis reveals that hallucinated tokens do not simply over-rely on linguistic priors; instead, they exhibit an anomalous contextual deviation, showing significantly lower similarities to both textual and visual contexts in intermediate layers. Motivated by this, we propose ROT, a layer-specific, training-free framework. ROT dynamically detects semantic deviation in the middle layers and applies a norm-preserving rotation to steer the hidden states back toward the local multimodal context plane spanned by the contexts. For subsequent layers, a representational smoothing mechanism is introduced to stabilize the calibrated trajectory. Extensive experiments on multiple benchmarks demonstrate that ROT consistently reduces hallucinations across various model architectures and scales, offering an efficient, geometry-driven solution for grounded generation. 

---
# Local2Mesh: Spatially Localized Contour-to-Mesh for Left Ventricular Reconstruction from Sparse 2D Cardiac MRI 

**Authors**: Haoyu Wu, Ling Lin, Pascal Lefèvre, Ruizhe Li, Xiaowu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.06052)  

**Abstract**: Three-dimensional (3D) left ventricular (LV) reconstruction from sparse cardiac magnetic resonance (CMR) imaging remains challenging due to inter-slice misalignment and insufficient local spatial information between slices. Global aggregation of contour features may obscure local contour-to-surface relationships. We propose Local2Mesh, a spatially localized contour-to-mesh framework that deforms a template mesh to reconstruct 3D LV geometry from sparse 2D contours without 3D mesh annotations. The framework introduces geometry-aware alignment to correct inter-slice misalignment and a plane-aware Local Router that routes contour features to template vertices using vertex-to-plane distances. Local and global contour features then jointly guide graph-based template deformation for 3D LV reconstruction. Experiments on two public datasets, M\&Ms-2 and ACDC, demonstrate superior geometric reconstruction and functional estimation over existing methods. Zero-shot transfer from M\&Ms-2 to ACDC demonstrates strong cross-dataset generalization. Reconstructed meshes also improve disease classification over sparse contours, supporting their utility for downstream cardiac analysis. These results demonstrate that combining geometry-aware alignment with local contour-to-vertex modeling improves LV reconstruction from sparse 2D contours and supports downstream cardiac analysis. The code is available at \url{this https URL}. 

---
# Backdooring Sparse Autoencoders 

**Authors**: Enrico Ahlers, Daniel Passon, Tobias Kiecker, Eik Reichmann, Lars Grunske  

**Link**: [PDF](https://arxiv.org/pdf/2610.06049)  

**Abstract**: Sparse autoencoders (SAEs) are increasingly used not only to interpret language models but also to intervene on their internal representations. We show that this creates a supply-chain attack surface: a maliciously modified SAE can induce attacker-chosen behavior when inserted into the forward pass of an otherwise unchanged language model. We introduce a decoder-only SAE backdoor that leaves both the underlying LLM and the SAE encoder frozen, restricting the attack to a single auxiliary component at a single insertion layer. Using code generation as a case study, we demonstrate high rates of unsolicited code insertion across three language models and a wide range of insertion layers, as well as trigger-dependent behavior conditioned on a prompt cue. We further evaluate the modified SAEs using HumanEval and selected SAEBench metrics. While attack effectiveness varies across models and layers, strong backdoor behavior can coexist with relatively small changes in several conventional SAE quality measures. These results establish that SAEs can carry behavioral backdoors without modifying the language model itself and should therefore be treated as security-sensitive components. 

---
# Representation Disentanglement for Fair Chest X-Ray Diagnosis 

**Authors**: Yujie Sun, Ruizhe Li, Xiaowu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.06041)  

**Abstract**: Deep learning has advanced chest X-ray (CXR) diagnosis, yet demographic biases in learned representations may contribute to performance disparities across intersectional groups. We propose a single-encoder framework combining dual-level decorrelation with prototype-guided cross-group contrastive learning to reduce demographic dependence while accounting for within-class variation. We further propose Demographic Representation Alignment Reduction (DRAR), a new metric that quantifies the reduction in demographic structure within disease representations. The framework is evaluated on four classification tasks using 34,809 CheXpert test images across eight intersectional groups, defined by age, sex and ethnicity. Compared with empirical risk minimization (ERM), our method reduces the mean equalized-odds gap from 15.41\% to 10.86\% and the AUC gap from 5.95\% to 5.01\%. Our method achieves a DRAR of 59.04\% relative to ERM, with only a slight decrease in mean AUC. These results demonstrate that representation disentanglement can reduce demographic bias and improve intersectional fairness. Code is available at \url{this https URL}. 

---
# MercerFlow: Flow Matching in a Kernel-Induced Latent Space for Probabilistic Forecasting 

**Authors**: Ilya Kuleshov, Egor Serov, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2610.06039)  

**Abstract**: Recent work has shown that probabilistic flow matching for time series forecasting benefits from a data-matched prior. The resulting prior introduces local correlations, which a sequential architecture usually absorbs: a recurrent neural network (RNN), a structured state-space model (S4), or a Transformer. However, such a backbone costs GPU memory and time per epoch. A cheaper alternative is MLP-based latent-space flow matching: embed the time series via an invertible map to a single latent vector and learn the flow there, so a tabular MLP can treat the series as a set of features. The relationship between the prior and the choice of linear latent map is understudied in conditional flow matching (CFM) forecasting, yet we found it strongly affects performance. Fixed transforms such as Fourier or discrete cosine (DCT) are only well-conditioned for Ornstein--Uhlenbeck priors, while a principal-component (PCA) map fit to the data is a strong but training-set-dependent reference sensitive to train--test shift. Instead, we propose to use the Mercer eigenbasis of the prior kernel: it diagonalises the centred covariance exactly, decouples from training data, and adapts to non-stationary and periodic priors. On five GluonTS benchmarks (ETTh1, ETTh2, Weather, Electricity, Traffic) under a shared protocol with TSFlow, the resulting MLP matches or beats it on CRPS at about $4.7\times$ less training memory and $3.5\times$--$4.4\times$ less time per epoch. 

---
# LightMTP: Lightweight Latent Multi-Token Prediction 

**Authors**: Tamara Czinczoll, Julie Kallini, Gerard de Melo, Chen Shani  

**Link**: [PDF](https://arxiv.org/pdf/2610.06031)  

**Abstract**: Next-token prediction (NTP) is the standard pretraining objective for large language models, yet it provides an explicit training signal only for the immediate next token, which can lead models to exploit local patterns instead of capturing longer-range structure and ideas. Multi-token prediction (MTP) addresses this by training models to predict several future tokens. However, existing MTP methods often introduce a large number of new parameters with limited improvements in downstream performance. Latent MTP approaches address this efficiency issue by encoding future tokens into a vector representation. However, these approaches usually rely on external helper models for future token encoding. We propose LightMTP, a lightweight, i.e., parameter-efficient, latent MTP approach that bootstraps the future token representations from the model's own hidden states. Our two LightMTP variants extend supervision to more future tokens without requiring the additional computational overhead of conventional MTP nor the external supervision latent MTP normally relies on. LightMTP adds at most 1% extra parameters, retains better performance on general language modeling benchmarks, and achieves similar gains in planning, coding, and reasoning. 

---
# Differentiable Bit-Widths: Co-optimizing Pruning and Quantization via SVD for Ultra-Efficient LLM Compression 

**Authors**: Hankyul Kang, Jongbin Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06026)  

**Abstract**: SVD-based pruning and quantization have recently emerged as a promising strategy for the ultra-efficient compression of large language models. In these methods, compression is performed in two stages: components are first truncated, and the remaining ones are subsequently quantized. Although this decoupled pipeline benefits from both pruning and quantization, it requires separate optimization for each stage and fails to fully exploit their balance, which can lead to suboptimal performance under aggressive compression. To address this limitation, we propose a new LLM compression method that co-optimizes pruning and quantization in a unified framework. Our key idea is a differentiable method for learning component-wise bit-widths, allowing less important components to be assigned 0-bit precision and pruned away. Notably, our method performs favorably against two-stage baselines, even when subjected to extreme quantization settings ($1.61$ bits) designed for ultra-efficiency. Code: this https URL. 

---
# Scalable Minimal-Change Learning for Controllable Image Editing 

**Authors**: Shuo Chen, Fengming Huang, Yu Yao, Mingming Gong, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06021)  

**Abstract**: Image editing should change only the attributes specified by an instruction while preserving everything else, yet current methods often make unintended changes. We treat this minimal-change principle as an optimization objective for instruction-based editing. Latent L1 regularization is a poor proxy for output locality in modern nonlinear generators and often requires supervision unavailable at scale. We instead optimize edit outcomes with reinforcement learning. An agentic vision-language reward model audits each source image, instruction, and edited image for two failure types: unimplemented requested changes and unintended changes. A group-level rubric merges and verifies these issues to provide consistent rewards across candidate edits without per-instruction human annotations. On FLUX.1 Kontext-dev, ARRO raises average EditScore from 5.21 to 5.88 across MinEval, MagicBrush, AnyBench, and Emu-Edit. On 600 evaluation examples, it reduces off-target pixel change by 8.4% relative to the base editor. Reward and SFT controls, blinded human evaluations, and transfer to OmniGen2 provide complementary evidence. Code: this https URL 

---
# Adaptive Expert Guidance for Efficient On-Policy Reinforcement Learning 

**Authors**: Daniele Affinita, Ming Xu, Rudolf Reiter, Davide Scaramuzza, Pascal Fua  

**Link**: [PDF](https://arxiv.org/pdf/2610.06019)  

**Abstract**: With massively parallel simulation, on-policy Reinforcement Learning methods such as PPO have become standard in many domains. However, learning from scratch is sample-inefficient and fails to exploit the potential existence of a suboptimal expert, such as a heuristic, a model-based controller, or a policy trained on a related task. Such an expert is often available and can guide early training, but its sub-optimality limits final performance. The challenge then becomes balancing expert guidance against learning from rewards. Existing methods set the expert's influence through a blending weight, a schedule, or an evaluation-driven curriculum. Alternatively, they adapt it with additional learned components such as critics over expert actions or auxiliary agents. However, none optimizes it using the same on-policy objective as the policy itself. We propose a method in which the learner and the expert alternate control within each training episode, and the expert's share of control is a single learnable parameter optimized jointly with the policy. The learner benefits from the expert early in training, but its share of control declines as the learner becomes more competent, until eventually vanishing completely. This leaves the learner acting alone and better than the suboptimal expert. We evaluate our method on 34 tasks across two benchmarks, spanning discrete and continuous action spaces, using both learned and model-based experts. Our method improves sample efficiency over guided and unguided baselines while requiring minimal hyperparameter variation. The expert's share decays to zero as the learner improves, vanishing when the expert is no longer useful. 

---
# Investigating Query-Insensitive Behavior in Spatio-Temporal Video Grounding 

**Authors**: Eryk Kołodziejczyk, Alberto Presta, Karol Szurkowski, Michal Byra  

**Link**: [PDF](https://arxiv.org/pdf/2610.06018)  

**Abstract**: Spatio-temporal video grounding (STVG) aims to localize objects or events described by natural language queries in both space and time. Existing STVG models are typically trained and evaluated under the assumption that each query is relevant to the input video. In this work, we challenge this assumption by studying the behavior of state-of-the-art STVG models under irrelevant queries and missing textual input. Our experiments show that current models can still produce plausible spatio-temporal predictions even when the query is unrelated to the video or removed entirely. We further analyze HCSTVG-v2 and VidSTG to identify dataset regularities that may encourage such query-insensitive behavior. Our study highlights an underexplored limitation of STVG models and motivates negative-aware evaluation protocols and architectures that explicitly assess query relevance. 

---
# Parallelism or Concession? Concurrency-Aware Procurement Negotiation for Agentic Commerce 

**Authors**: Xiaolin Xu, Donghao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06017)  

**Abstract**: Agentic buyers can cheaply fork a procurement task into many parallel negotiations, but concurrency is not free: every thread consumes resources, and simultaneous agreements create cancellation and commitment risk. We study a one-unit post-order sourcing problem with a single hard-deadline negotiation window, in which a planner jointly chooses the number of seller-facing negotiators and a common procurement price cap. The model combines a product-specific acceptance curve with fulfillment loss, per-thread cost, and excess-commitment cost. We establish three structural results. First, holding the per-thread acceptance target fixed, the marginal value of another negotiator decays geometrically, yielding a conditional concurrency threshold. Second, under a convex quantile curve, parallelism substitutes for concession: more concurrent negotiators imply a weakly lower per-thread acceptance target and price cap. Third, when prices are more dispersed, Agentic buyers benefit by searching harder for bargains, but suffer when they instead try to guarantee procurement by offering higher prices. We operationalize these results in the Concurrency-Aware Negotiation Optimizer (CANO), a deterministic optimizer that jointly determines the optimal negotiation concurrency and procurement price cap for an agentic procurement system. Across different analytic market configurations and extensive Monte Carlo, finite-data, non-Gaussian, and correlated-seller stress tests, CANO consistently outperforms common heuristic policies while validating the predicted structural properties. 

---
# AgentSpy: Making AI Agent Behavior Observable 

**Authors**: Christoph Bühler, Matteo Biagiola, Luca Di Grazia, Guido Salvaneschi  

**Link**: [PDF](https://arxiv.org/pdf/2610.06001)  

**Abstract**: AI agents built on large language models (LLMs) run shell commands, read and write files, and reach the network, typically with their user's privileges. However, what an agent does during an execution is difficult to understand: tests assert on the result, and the agent's trajectory records only what the agent reports about itself, which may omit behavior executed by its subprocesses. We present AgentSpy, an approach that observes an agent from outside the agent. AgentSpy runs the agent in an isolated environment, configured by a declarative specification, and records the system calls and network traffic of the agent and of every process it executes. Based on this monitoring, AgentSpy supports two families of analyses: conformance analyses, which measure obligations, i.e., what an agent execution should do, and safety analyses, which check prohibitions, i.e., what an agent execution must never do. We instantiate one analysis of each family. The reliability analysis uses rules to summarize each run by the environment resources the agent uses: the commands it executed, the files it accessed, and the hosts it contacted. The security analysis applies deterministic rules to the system calls of an execution. For reliability, we evaluated AgentSpy on 77 tasks with the codex harness and three recent LLMs, executing each task three times. Sets of repeated runs of the same task are more similar than sets that include runs of another task in 92.2% of the comparisons. Among tasks for which all three runs pass outcome-based tests, the agent performs task-unrelated activities in 18% of the cases, reads the grading files in 7%, and does not use the developers' guidance in 17%. For security, the generic rules of AgentSpy detect four of five attack categories we considered, with no false positives across 50 runs. 

---
# EpicWorldModel: Exploration-driven Planning with Latent World Models 

**Authors**: Bowen Feng, Julian Ost, May Mei, Anirudha Majumdar, Felix Heide  

**Link**: [PDF](https://arxiv.org/pdf/2610.05996)  

**Abstract**: Latent world models based on Joint-Embedding Predictive Architecture (JEPA) are deterministic by design. While successful in fully observable scenarios, this paradigm breaks down when past observations and actions lead to multiple plausible future possibilities, e.g., due to occlusion. We introduce EpicWorldModel, a framework to train stochastic JEPAs for environments and tasks with inherent uncertainty under partially observability. We jointly train the EpicWorldModel predictor with its latent representation space to directly predict multiple potential future states using a flow-matching objective, when the goal-relevant scene content is absent from the conditioning history. We show that flow predictive variance, motivated by its relation to an upper bound on predictive entropy, serves as a useful exploration guidance for planning. By incorporating this uncertainty signal into Cross-Entropy Method (CEM)-based planning, our approach balances goal-reaching with exploration of uncertain regions where occluded goals are most likely to be located. We demonstrate the effectiveness of EpicWorldModel through a series of latent planning experiments with the best or on-par performance across tasks, showing up to 22% empirical improvement in success rate over LeWorldModel. 

---
# Transfer-Stratified On-Policy Distillation for RL-Improved Reasoning Teachers 

**Authors**: Xiaoyu Chen, Bo Shao, Tiangang Zhu, Bintao Wu, Linjun Shou, Fengge Wu, Feng Sun, Wenbiao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2610.05974)  

**Abstract**: Reinforcement learning can substantially improve a reasoning teacher, but it is unclear which of those improvements survive when the teacher supervises a smaller on-policy student. We study this question in mathematical reasoning by comparing teacher lineages before and after GRPO, multiple student scales, direct GRPO, and several on-policy distillation objectives. The central finding is that transfer is structured rather than scalar: teacher strength alone does not make dense distillation competitive, while an RL-improved teacher creates useful but metric-dependent student gains. This motivates Transfer-Stratified On-Policy Distillation (TS-OPD), which screens training problems by the joint sampled success of the student and teacher, routes acquisition problems to gated forward KL, routes consolidation problems to gated reverse KL, and adds an entropy brake to protect sampled coverage. Across the main comparison, TS-OPD is the strongest student objective for macro average correctness with the GRPO-improved teacher, while pass@K remains more mixed. Ablations show that the gains come from routing and token gating rather than skipping problems. These results support a transfer-aware view of OPD: stronger teachers help when the supervision direction and token budget match the student's observed ability, not merely because the teacher endpoint is stronger. 

---
# JLD: Perceptual Distance Through A Jacobian Lens 

**Authors**: Shreshth Saini, Balu Adsumilli, Alan C. Bovik  

**Link**: [PDF](https://arxiv.org/pdf/2610.05967)  

**Abstract**: Image compression, restoration, and generation all require a way to measure how different two images look to a person. Pixel error ignores how people see, while the most accurate perceptual distances are typically fitted to human judgments, tying them to a fixed data and resolution. For example, when image resolution is doubled, the correlation of DISTS with human scores on TID2013 drops from 0.815 to 0.717. We introduce the Jacobian Lens Distance (JLD), which derives its perceptual geometry from a frozen vision encoder rather than from human labels. JLD combines the locality of early patch features with the perceptual sensitivity captured by later encoder representations. Specifically, we use the encoder Jacobian to identify directions in the early feature space that most strongly affect the encoder output, producing a fixed metric tensor, $E[J^\top J]$, which we call the Jacobian lens. The lens is fitted only once from 100 unlabeled images, taking about 35 seconds. Locally, this construction defines a pullback metric in pixel space, giving JLD a clear geometric interpretation that can be directly analyzed on real images. Across four standard perceptual databases, JLD achieves state-of-the-art performance and consistently outperforms LPIPS, DISTS, PieAPP, and DreamSim. JLD is also robust to changes in image resolution, on TID2013, its lens-term correlation remains nearly unchanged when the resolution is doubled, decreasing only from 0.850 to 0.845. We further introduce JLD-fast, which is $4\times$ faster than LPIPS-VGG while achieving a mean correlation of 0.911. Finally, JLD naturally extends to video, reaching a correlation of 0.786 on Waterloo IVC 4K compared with 0.611 for VMAF. 

---
# HuatuoGPT-3: RL-Only Domain Adaptation from Base Models 

**Authors**: Junying Chen, Xinyuan Xie, Ziniu Li, Wenyuan Gu, Jianquan Li, Xiang Wan, Guangjun Yu, Ruoyu Sun, Haizhou Li, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05966)  

**Abstract**: Domain adaptation aims to turn a general-purpose large language model (LLM) into an expert for a target domain. While the dominant SFT+RL pipeline offers a convenient cold start, it may reduce exploration diversity and introduces additional complexity through multi-stage optimization. These limitations motivate RL-only adaptation. However, pure on-policy RL suffers from a cold-start problem, while mixed-policy RL still falls short: informative tokens in teacher outputs are learned too slowly in early training, and stale teacher outputs can hinder later improvement. We identify these two failure modes as Gradient Starvation and Teacher-Distribution Anchoring. To address them, we propose One-stage Policy Optimization (OnePO), which treats teacher outputs as transient guidance for policy improvement. OnePO combines Adaptive Objective Evolution to strengthen learning on informative low-probability teacher tokens and Teacher Retirement to discard teacher outputs once the current policy can surpass them. On medical adaptation, OnePO achieves 67.2 on HealthBench (Total) with only 20K training samples, outperforming SFT+RL and pure RL by 2.7 and 7.4 points, respectively. We further scale OnePO to produce HuatuoGPT-3, an open-source medical LLM series whose 27B variant reaches 70.1 on HealthBench (Total) and 71.4 on HealthBench Professional, surpassing frontier models such as GPT-6 Astra. Models and code are available at this https URL. 

---
# Physics-Informed but Not Physics-Consistent: Error Geometry and Subspace Projection for Neural AC Power Flow 

**Authors**: Changhun Kim, Timon Conrad, Redwanul Karim, Karan Pahlajani, Julian Oelhaf, David Riebesel, Tomás Arias-Vergara, Andreas Maier, Johann Jäger, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2610.05959)  

**Abstract**: Recent neural power-flow solvers, including emerging foundation models, achieve accurate voltage predictions, yet such accuracy does not necessarily imply physically consistent solutions. Even small complex voltage errors can yield large AC power-balance residuals. We study this accuracy-consistency gap across PIGNN-GC, GridSFM, gridfm-graphkit, and LUMINA on realistic 2224-bus Great Britain network (GBnetwork) scenarios, with cross-grid evaluation of GridSFM over 31 systems. Using a singular value decomposition (SVD) basis fitted to training AC power-flow solutions, we find that neural prediction errors contain substantial components outside the dominant solution subspace. To address this mismatch, calibrated solution-subspace projection (CSP) suppresses off-subspace prediction components after train-only bias calibration, reducing Mean PB by 67.0%, 37.8%, 40.5%, and 68.9% for PIGNN-GC, GridSFM, gridfm-graphkit, and LUMINA, respectively, relative to calibrated predictions, while improving voltage-magnitude accuracy in all four models. These results identify output-error geometry as an important factor in physics-consistent neural AC power flow. Code: this https URL 

---
# Runaway Reaction: When Benign Skills Compose into Malicious Behavior 

**Authors**: Zunlong Zhou, Ziyuan Yang, Mengyu Sun, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05943)  

**Abstract**: Agent skills package task-specific knowledge and procedures that can be composed to support complex agent tasks, while public marketplaces provide a growing pool of reusable skills. Existing security vetting, however, largely evaluates skills in isolation, leaving composition-induced risks underexplored. Such risks arise because composing benign skills expands the agent's capability space, enabling behaviors unavailable to any skill alone. Interestingly, we find that directly composing benign skills can already induce malicious behaviors, even when every individual skill passes security vetting. We further find that some target malicious behaviors remain difficult to realize through direct composition, even when the selected skills collectively provide the required capabilities. To systematically instantiate these attacks, we present Compositional Risk Induction via Multi skill Execution (CRIME). CRIME first uses the Malicious Plot Casting (MPC) module to decompose a target malicious behavior into complementary requirements and identify suitable benign skill compositions from public skill repositories. For compositions that cannot directly realize the target behavior, the Runaway Reaction Steering (RRS) module uses execution feedback to iteratively refine the selected skills toward the target while requiring each skill to remain benign under standalone vetting. The resulting composition is then passed to the Skill Reaction Chamber (SRC) module, where the skill pair is executed in a sandbox and the resulting environmental consequences are examined to determine whether the target behavior has occurred. Unsuccessful cases are returned to RRS for further refinement. Furthermore, we construct a benchmark of 4,000 public skills across eight cybersecurity behaviors for systematic evaluation of composition-induced vulnerabilities. 

---
# ReMem: Streaming Video Understanding With Long Context Retention 

**Authors**: Li Yiheng, He Xu, Wang Shaobo, Shao Ling, Lu Shijian  

**Link**: [PDF](https://arxiv.org/pdf/2610.05940)  

**Abstract**: Despite their impressive performance on a wide range of video understanding tasks, current Vision Language Models (VLMs) are predominantly designed for offline scenarios and struggle to handle online streaming videos that demand low latency response. Several studies have explored memory and token compression strategies in an attempt to adapt offline VLMs for streaming video understanding tasks. However, through our probing experiment, we identify that most existing works tend to progressively lose long context information as length of input stream increases. To address this, we propose ReMem, a novel training-free adaptation technique that enables VLMs to process streaming videos of arbitrary lengths while improving their long context information retention capability. ReMem exploits memory from two perspectives, implemented as two core components. The Streaming Context Memory (SCM) continuously compresses historical context with query-independent attention. The Retrieved Vision Memory (RVM) then retrieves the most salient, query-relevant context from memory to augment the VLM's input. Comprehensive experiments demonstrate that the proposed ReMem achieves state-of-the-art (SOTA) performance across a variety of widely used benchmarks, spanning both streaming video and general long video understanding tasks. 

---
# ThunderSyncRL: Lossless Acceleration of Agentic Reinforcement Learning 

**Authors**: Seil Kang, Hangoo Kang, Tarun Suresh, Youngeun Kim, Shreyas Pimpalgaonkar, Seong Jae Hwang, Azalia Mirhoseini  

**Link**: [PDF](https://arxiv.org/pdf/2610.05935)  

**Abstract**: Language models are moving beyond generating answers to pursuing long-horizon goals in interactive environments. Post-training these agents requires long, heterogeneous trajectories, and synchronous systems leave learner engines idle until rollout and verification finish. To squeeze out these pipeline bubbles, asynchronous training overlaps rollout and learning across updates, but comes at the cost of policy staleness. We introduce ThunderSyncRL, which starts gradient computation as soon as all required inputs are fixed, without policy staleness. For group relative policy optimization (GRPO), ThunderSyncRL computes each trajectory's score gradient as soon as the reward for that trajectory arrives, without waiting for the group. For on-policy distillation (OPD), it computes gradients for each completed agentic turn's teacher-scored actions while tool calls run in the sandbox. We prove that gradient streaming produces the same GRPO and OPD updates as batch-synchronous training, without changing either objective. On SWE-bench Verified and Terminal Bench 4.0, we train models to the same performance up to $1.9 \times$ faster than synchronous training. With zero policy staleness, ThunderSyncRL also outperforms asynchronous training at a fixed budget by up to $2.47$ percentage points. 

---
# UltraDub: Towards Authentic Dubbing by Unifying Visually-Steered Flow Learning and Trajectory Guidance 

**Authors**: Gaoxiang Cong, Liang Li, Jianwei Wen, Zhedong Zhang, Zheng-Jun Zha, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05932)  

**Abstract**: Visual voice cloning requires intelligible, speaker-consistent speech synchronized with visible articulation. However, sequential multimodal conditioning can disrupt previously established temporal and speaker cues, while imbalanced inference guidance can improve linguistic accuracy at the expense of lip synchronization. In this paper, we propose UltraDub, a Unifying Visually-Steered Flow learning and trajectory Guidance Dubbing framework that leverages vision in two ways: as continuous motion for multimodal context aggregation, and as structural rhythm for trajectory rectification. Specifically, we introduce the Motion-guided Dual-context Retrieving (MDR) module, which continually recalibrates linguistic and speaker-style retrieval through shared lip-motion query residuals, utilizing independent time-conditioned gates to regulate their contributions. Furthermore, we propose Rhythm-anchored Trajectory Guidance (RTG), a training-free mechanism that evaluates hierarchical multimodal corrections at a visual-only predictive midpoint, safely strengthening semantic conditioning while better preserving temporal alignment. Finally, we construct DiverseDub, a multi-scenario benchmark to evaluate video dubbing in the wild. Extensive experiments demonstrate that UltraDub achieves state-of-the-art performance across four datasets. 

---
# Incentive Alignment in Online Experimentation 

**Authors**: {Ermis Soumalias, Richard Mudd, Abbas Zaidi  

**Link**: [PDF](https://arxiv.org/pdf/2610.05922)  

**Abstract**: Evaluating the causal effect of new features is a central goal for online platforms. While recent literature addresses limited testing traffic via centralized portfolio optimization, this perspective abstracts away a critical institutional reality: experimentation is operationally decentralized. The experimenters who develop new features also dictate which hypotheses to test, and they are typically rewarded based on empirical average treatment effects that are prone to upward bias. Left unchecked, this principal-agent conflict can severely erode platform value, a structural failure that conventional centralized levers, such as significance thresholds and traffic budgets, cannot resolve. By reframing experimentation as an incentive design problem, we demonstrate that two practical mechanisms, sample splitting and shrinkage, can effectively bridge this gap. Sample splitting aligns incentives perfectly at a bounded traffic cost, while shrinkage consumes no additional traffic and guarantees that interventions with negative expected effects are strictly unprofitable to field. 

---
# Non-invasive Seizure Detection Using Wearable Wrist-worn Accelerometry and Deep Learning 

**Authors**: Nilushika Udayangani Hewa Dehigahawattage, Kishor Nandakishor, Marimuthu Palaniswami  

**Link**: [PDF](https://arxiv.org/pdf/2610.05919)  

**Abstract**: Seizure monitoring and detection are crucial for reducing the morbidity and mortality associated with seizures. Current epilepsy care, often involving expensive video-electroencephalography (VEEG) monitoring, requires specialized expertise and is limited to in-hospital settings, and intrusive in nature. Seizure diaries, on the other hand, suffer from unreliability due to under-reporting, leading to incorrect therapeutic decisions. Wearable non-invasive seizure detection may offer a more tolerable and feasible solution for long-term ambulatory monitoring. This study explores a wearable remote monitoring system utilizing a single wrist-worn accelerometer device and capable of detecting multiple types of seizures, including shorter duration events. We enrolled 79 patients under video-electroencephalography monitoring to wear accelerometer devices and collect data. Concurrent VEEG recordings were reviewed by board-certified epileptologists to produce annotations, including seizure onset, offset, and seizure type. Using this data, we constructed a deep neural network based on the time-series ResNet architecture, which could discriminate among seizure and non-seizure events. Our proposed approach achieved a seizure detection sensitivity of 95.65% and an overall false alarm rate of 0.15/24 hours during the evaluation, which spanned 5576 hours of total recording. Additionally, it resulted in an area under the receiver operating characteristic curve (AUC-ROC) of 0.98 and an area under the precision-re call curve (AUC-PRC) of 0.67 when averaged over 20 patients who experienced 46 convulsive seizures. These promising results suggest that the proposed seizure detection system can be effectively used for long-term ambulatory seizure monitoring. Future steps include validating our findings in larger datasets and assessing the utility of detection for additional seizure types. 

---
# On Hyperparameter Tuning on the Test Set 

**Authors**: Matteo Fregonara, Tom Viering, Jan van Gemert  

**Link**: [PDF](https://arxiv.org/pdf/2610.05902)  

**Abstract**: "Don't tune hyperparameters on the test set" is often stated in machine learning textbooks. Violating it is considered a cardinal sin that produces misleadingly optimistic results, corrupts benchmark integrity, and thus can even be interpreted as scientific fraud. Yet evidence suggests that test set hyperparameter tuning does occur in practice, making it all the more important to understand its actual consequences. So how bad is it, really? In this work we question this dogma and put it to an empirical test. We systematically study the magnitude of the performance inflation caused by tuning the hyperparameters on the test set for MNIST-1D, CIFAR-10, and three tasks from the GLUE benchmark. Our experiments show that while the effect is real and significant, it is frequently small relative to other sources of noise. In many cases, we find that tuning on the test set recovers exactly the same model as when tuning on the validation set. Most importantly, we find that the rankings of models remain essentially preserved after tuning on the test set and therefore that consistent test-set tuning may not invalidate benchmarks or model selection. Our results call for a more nuanced view of tuning hyperparameters on the test set, stimulating researchers to openly report test tuning. 

---
# TasteRoute: Personalized Routing for Video Generation 

**Authors**: Zhi Rui Tam, Chao-Chung Wu, Sin-Han Yang, Peyton Ku, Brendan Kuang, Tzu-Ting Hsieh, Min-Fang Hsu, Fang-Ling Tsai, Yun-Nung Chen, Wei-Chiu Ma, Chieh-Yen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.05896)  

**Abstract**: Rapid progress in video generation has led to a plethora of models that differ substantially in capability and generation cost. This raises a natural question: can each request be efficiently routed to an appropriate model? We find that even when the consensus of the other annotators is used as an oracle, it agrees with each annotator's own favorite only 34-55% of the time. Motivated by this observation, we introduce TasteRoute, a personalized video-generation router that selects a generator jointly based on the input request, user preferences, and available generation budget. Across text-to-video and image-to-video settings, TasteRoute is competitive with strong simple baselines on preference routing while reducing average generation cost. The cost saving increases under higher budget caps. Finally, we release TasteRoute-3k, a human-annotated dataset containing multi-model video comparisons, quality judgments, preference rankings, and user-profile signals to facilitate future research on personalized and cost-aware video routing. 

---
# OGAM: Connecting Systematic Testing to Runtime Assurance through Object-Grounded Attention Monitoring for VLA Policies 

**Authors**: Haki Darwish, Xiangyu Yin, Changwen Li, Rongjie Yan, Francisco Gomes de Oliveira Neto, Chih-Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.05878)  

**Abstract**: Benchmarks expose vision-language-action (VLA) policies to few canonical instructions, while exhaustive deployment testing is impossible. We introduce Object-Grounded Attention Monitoring (OGAM), connecting systematic testing to runtime assurance: testing reveals attention divergence between successful and failed executions, and OGAM uses this signal to stop failures beyond the finite suite. We generate scene-grounded instructions through pairwise combinations of action templates and objects, and separately test meaning-preserving paraphrases. All 87 out-of-benchmark cases reveal problematic behavior across OpenVLA, OpenVLA-OFT, UniVLA, and $\pi_{0.5}$: none completes any of the 24 feasible instructions, while infeasible or hazardous requests also trigger behavior substitution. At each action query, we project gradient-weighted visual attention through object masks and group it by instruction role for comparison across tasks and policies. Dynamic time warping aligns this course with a successful reference despite speed differences; conformal calibration on successful episodes sets the early-stopping threshold for sustained deviations, with a nominal false-stop target of $\alpha=0.05$. Across four policies, OGAM stops 87-100% of failed episodes at median times of 5-12s within a 20s budget, with observed false-stop rates of 3-5%, without failure-labeled training. Finite testing thus identifies attention patterns that support online intervention before failure fully unfolds. 

---
# Off-Policy Merging Beats On-Policy Self-Distillation for Continual Learning 

**Authors**: Chen Henry Wu, Thomas Zhang, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2610.05872)  

**Abstract**: A long-standing goal of AI is a model that can continually learn and improve itself. On post-trained models, supervised finetuning (SFT) on new data often causes poor generalization and catastrophic forgetting. As such, the conventional wisdom is that on-policy training is a prerequisite for continual learning. In practice, however, data containing new knowledge or capabilities are often off-policy. While methods such as on-policy self-distillation (OPSD) try to bridge this gap by converting off-policy data into on-policy signal, they have been shown to cause reasoning collapse. In this paper, we show that off-policy merging beats OPSD for continual learning. We first show that SFT learns a useful signal from new data, but naively applying its update interferes with existing capabilities. We reduce this interference with a simple recipe we term grafting, which changes where the update is learned and how it is applied: (1) learning the update on an earlier donor checkpoint, ideally even before the end of pretraining, and applying the weight update to the post-trained model; (2) scaling the weight update, equivalent to a form of model merging; and (3) optionally, masking the most sensitive update directions when the new data distribution is far from the post-trained model. Across continual learning settings including (1) distilling from expert traces, (2) self-improvement with STaR and Pedagogical RL, and (3) injecting knowledge after pretraining cutoff, grafting Pareto-dominates both SFT and OPSD in new-task and old-task performance, while avoiding expensive on-policy sampling. Therefore, our work challenges on-policy training as a necessity for continual learning on RL-trained models. 

---
# Weave Mamba Fusion: Global Cross-Scale Interaction for Lightweight Face Detection 

**Authors**: Dohun Kim, Jinmyung Jung  

**Link**: [PDF](https://arxiv.org/pdf/2610.05865)  

**Abstract**: Feature pyramid methods, from FPN to BiFPN, have achieved strong performance in face detection by fusing multi-scale features. However, detecting faces under unconstrained conditions, such as small scale, occlusion, and extreme pose, remains difficult, as it requires global cross-scale dependencies that local fusion cannot model. State space models such as Mamba provide global context with linear complexity by scanning features as a sequence, and therefore offer a promising direction for this problem. Nevertheless, such a scan needs the two pyramid scales combined into a single feature map, and the way they are combined determines whether cross-scale structure is preserved. Summation collapses the two scales before the scan, so the scan has no cross-scale structure to exploit, while concatenation keeps both scales but at far higher cost. To address this, we propose \textbf{Weave Mamba Fusion (WMF)}, which interleaves two adjacent pyramid scales column by column so that each step of a horizontal bidirectional SS2D scan moves from one scale to the other. With partial-channel processing and parameter-free de-weaving, WMF enables efficient cross-scale interaction while preserving feature structure. Integrating WMF into every fusion node yields \textbf{WeaveBiFPN}, the neck of our \textbf{WeaveFace} detector. On WIDER FACE, WeaveFace achieves 91.41\% mean AP with only 0.34M parameters and 1.16 GFLOPs, outperforming prior detectors under 0.5M parameters. Its largest gains are on the Hard subset, where it reaches 87.14\% AP. The code is publicly available at \url{this https URL}. 

---
# CCQ: A Multi-State Child Care Quality Dataset to Support AI for Children's Health Research 

**Authors**: Victor Li, Yuzhang Xie, Ziwei Dong, Qingyang Zhu, Wenjing Ma, Carl Yang, Jinbing Bai, Huiwen Xu, Jiaying Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05863)  

**Abstract**: High-quality child care in early life is a critical determinant of children's growth and development. Research on child care quality has been constrained by fragmented, non-research-friendly, and privacy-bound datasets. We present CCQ (Child Care Quality), a large-scale, de-identified dataset for applied data science research at the intersection of AI and early childhood health. CCQ integrates 59,372 child care provider records across 12 U.S. states, covering diverse provider types as well as data schemas. To ensure research utility while protecting privacy, we implement an automated, LLM-based curation pipeline that anonymizes, cleans, and standardizes raw state records into two complementary releases: a cleaned textual release and a fully preprocessed tabular release. We also benchmark traditional machine learning models, tabular foundation models, and language models on quality rating prediction and important features analytics. Within a state, tabular classifiers on the preprocessed tables perform best. Across states, zero-shot transfer is near chance, but modest target-state supervision recovers most of the within-state performance, and pretraining on other states benefits finetuned language models. We release both datasets with all code to accelerate AI-driven research on child care quality and ultimately improve children's health and development. 

---
# Imagine to Act: High-Fidelity Data Synthesis via Image Editing World Model for Scalable GUI Agent Training 

**Authors**: Yongxin Ning, Runliang Niu, Qianli Xing, Zhiyi Duan, Qingzu He, Pan Wang, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05861)  

**Abstract**: Graphical User Interface (GUI) agents have emerged as a promising paradigm for automating complex digital workflows across diverse applications. However, training highly capable and generalizable agents fundamentally relies on massive, high-fidelity visual-action trajectories, which are notoriously difficult to acquire. While human demonstrations are unscalable, existing GUI world models rely on text descriptions or HTML rendering, discarding crucial pixel-level visual details like icons and layout styles. To address this issue, we introduce Infinite-Dreamer, a simulation-free data synthesis method powered by a pixel-level Image Editing World Model. By conceptualizing GUI transitions as image editing tasks, we leverage Vision-Language Models (VLMs) to describe action-induced UI changes as structured delta-text. We then fine-tune an image editing backbone to controllably synthesize realistic screenshot transitions. We utilize this model to generate both single-frame visual robustness data and multi-step imaginary trajectories. To validate the effectiveness of our approach, we fine-tune the Qwen3-VL baseline solely on the synthesized data to obtain Infinite-Actor, and evaluate it on AndroidWorld, MobileWorld, and AndroidControl-Curated benchmarks. Infinite-Actor consistently outperforms the Qwen3-VL baselines across scales: Infinite-Actor-8B improves AndroidWorld Pass@1 by +4.45 and nearly doubles the MobileWorld Pass@3 success rate, while Infinite-Actor-2B improves Pass@1 by +9.05. Code is available at this https URL. 

---
# Curriculum Brain: Constructing Curriculum Knowledge Graphs as a Substrate for Cognitive Diagnosis 

**Authors**: Shrideep Tamboli, Chiranjeevi Maddala, Eshal Minhaj  

**Link**: [PDF](https://arxiv.org/pdf/2610.05860)  

**Abstract**: Cognitive Diagnostic Models (CDMs) identify which specific skills a student has and has not mastered, the signal a personalized learning path needs and a single aggregate score cannot give. Yet they are rarely deployed. The obstacle is their precondition: the Q-matrix, a mapping from every assessment item to the skills it requires, historically authored by hand.
We separate the task into two stages: first construct the curriculum's own knowledge graph, the full space of concepts and skills it contains, independent of any item; then map items against that graph on demand. This paper addresses the first stage only. The item-mapping stage is designed but not implemented here, so the claim that this shifts judgment cost from once per item to once per curriculum is a design rationale rather than a finding.
We present Curriculum Brain, a two-repository system pairing a version-controlled knowledge base with an agentic pipeline of eleven single-responsibility agents under a thin deterministic orchestrator. It generates candidate concept-skill mappings from official curriculum documents, checks them against accumulated rules, and compares them with a concept-skill map extracted independently from the textbook, repairing its own failures and escalating to a human only when it cannot resolve a case itself.
Across 241 chapter runs (168 distinct chapters), 41.5% produced a Generator output passing both checks without a patch, and 67.6% resolved without escalation. Both are measured against criteria the system itself produced, so both describe internal consistency rather than agreement with an external standard, and both pool two pipeline configurations separated by a single change at run 77; after it the figures are 57.0% and 91.5%. Observed spend was $1.19 per chapter, API spend only, excluding human review. We release both the framework and the resulting curriculum dataset. 

---
# Hierarchical Reinforcement Learning for Collision-Free Locomotion of an Underactuated Biped 

**Authors**: Jagannath Prasad Sahoo, Saurabh Kumar, Surya Prakash S.K., Samiran Datta, Abhay Dwivedi, Amit Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2610.05855)  

**Abstract**: A bipedal robot cannot deviate from its path to avoid an obstacle without disturbing its balance, and this coupling is most severe on underactuated platforms such as the biped considered here, which has four actuated joints per leg and no hip or ankle roll. This paper presents a Hierarchical Reinforcement Learning (HRL) framework in which a High-Level (HL) policy observes the robot pose, 36 raycast proximity measurements, moving-obstacle states, and a receding-horizon local goal, and outputs a body-velocity command $(v_x, v_y, \omega_{yaw})$ every ten control steps, while a velocity-conditioned Low-Level (LL) policy tracks each command through PD-controlled joint targets. Both policies are trained jointly with Soft Actor-Critic (SAC). Because the converged gait is task-agnostic, it is frozen and driven by classical planners over the same command interface, yielding three controlled baselines: SAC+A*, SAC+RRT*, and SAC+APF. Across 100 evaluation trials per method in randomized PyBullet environments, the proposed method reaches the goal in 98.0% of static and 88.0% of dynamic trials, against at most 78.0% and 68.0% for the planner hybrids, with path lengths within 4% of the A* reference, and ablations confirm that each observation channel and reward term contributes materially to this performance. 

---
# CACFG: Curvature-Aware Classifier-Free Guidance and Optimal Control 

**Authors**: Max Collins, Dasith de Silva Edirimuni, Jordan Vice, Tim French, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2610.05845)  

**Abstract**: Diffusion models generate samples by learning to reverse a fixed corruption process, and classifier-free guidance (CFG) is the standard mechanism for conditioning this process on a desired class or prompt. CFG can be applied at varying guidance strengths, and while higher strengths improve image quality and conditional alignment, too high a guidance strength can degrade image quality and diversity. Furthermore, CFG violates principled diffusion sampling dynamics, and existing explanations for why it works despite the violation disagree on the underlying theory or do not extend to deterministic samplers used in practice. We address both these issues. We first frame CFG sampling as a continuous-time optimal control problem, treating the sampling trajectory as a sequence of controls chosen to maximise the probability of the desired condition. Solving the resulting Hamilton--Jacobi--Bellman equation shows that CFG is recovered under specific path costs when using an unconstrained control set. We argue this lack of constraint is responsible for CFG's failure at high guidance strengths, since it permits the sampling path to move arbitrarily far from the current image estimate. To fix this, we propose curvature-aware CFG (CACFG), which constrains the control set to a hypersphere informed by the Gaussian regularisation used when training variational autoencoders. We show that the control inputs produced by CFG sampling routinely violate this bound, and that across diffusion models, datasets, and guidance schedules, CACFG achieves superior generative quality at mid-to-high guidance strengths with a less severe quality-diversity tradeoff than regular CFG. 

---
# Selecting Long-Horizon Trajectories for Reliable and Efficient Terminal-Agent Training 

**Authors**: Cuong Dang, Hoang Anh Just, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2610.05831)  

**Abstract**: Terminal agents are commonly trained by imitating long teacher trajectories, yet how much of each trajectory to supervise remains unexplored. We study the \emph{supervision horizon}, the number of trajectory tokens retained for training, and show that it is a key design axis for reliability and cost. Reliability improves with longer horizons but saturates: on Terminal-Bench, a 12K-token horizon solves more tasks than 16K ($29\pm0.7$ vs.\ $26\pm0.8$) while requiring 30\% less training time. The horizon also shapes agent behavior: short horizons cause premature termination, intermediate horizons yield productive error recovery, and long horizons induce over-persistence. We analyze this saturation through a bias--complexity bound, in which longer supervision reduces temporal supervision bias but increases finite-sample estimation error from more heterogeneous late-stage histories. Guided by this analysis, we propose \emph{selective long-horizon refinement}, which first trains on short prefixes and then refines only on continuations that are most likely under the warm-start model. It consistently outperforms full long-horizon training. At 16K, it raises successful attempts from $110\pm2.7$ to $126\pm2.1$ and tasks solved in at least six of eight attempts from $9\pm0.7$ to $14\pm0.6$; with half of the long-horizon data, it still reaches $122\pm2.4$ while cutting training time by 23\%. The gains transfer across benchmarks, from $64\pm2.6$ to $73\pm2.1$ on Terminal-Bench v2.0 and from $137\pm2.7$ to $155\pm2.2$ on OpenThoughts-TBLite. For long-horizon supervision, selecting the right trajectories matters more than training on all of them. 

---
# Who Keeps the Gains from Personal AI Assistants? Seller Adaptation and the Unassisted in a Language-Model Market Simulation 

**Authors**: Haonan Huang, Joey Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05823)  

**Abstract**: Personal AI assistants are beginning to transact for consumers, and early adopters capture real savings. Whether those savings survive, and what happens to consumers who have no assistant, depends on how sellers respond -- a question single-user evidence cannot answer. We build an agent-based rental market in which language models play consumers, assistants, and six adaptive sellers guided by an algorithmic pricing tool. Half the population receives an assistant under an advisory or an executing mandate; the contract pairs a fee only the renter's physical action avoids with a pre-selected add-on an authorised assistant can cancel online. An analytical benchmark and a behaviourally calibrated rule market supply ex-ante predictions, and paired branches with frozen versus adaptive sellers separate adoption effects from market feedback. Across thirty simulated markets, executing assistants cut adopters' spending by 13.7 USD per renter-day when sellers are frozen; adaptation claws back about a third, leaving 8.7, with the gains arriving both as lower bills and as rentals completed at all. Sellers raise headline rates while cutting fees, and the calibrated forecast of the burden on unassisted consumers (+3.6) does not transfer: their mean spending change is +0.4, confidence interval -0.6 to +1.3. Seller-model swaps and a within-market transfer of fee-setting to the pricing tool show that fee conduct, and with it the division of the gains, is decided on the seller side. Assistants, we conclude, should be evaluated at market level -- completion, total spending, and non-users included -- and the comparison layers locate exactly where a calibrated behavioural forecast fails in a language-model market. 

---
# Level-of-Token Diffusion 

**Authors**: Kiyohiro Nakayama, Brian Chao, Jan Ackermann, Hansheng Chen, Federico Tombari, Leonidas Guibas, Lior Yariv, Gordon Wetzstein  

**Link**: [PDF](https://arxiv.org/pdf/2610.05816)  

**Abstract**: Image and video diffusion models allocate equal computation to every region, even when the intended scene calls for varying levels of detail. The spatial distribution of detail can often be anticipated before generation, indicating where computation can be reduced. We introduce Level-of-Token (LoT) Diffusion, a framework that turns this knowledge into an explicit multiresolution token layout (Level-of-Token layout) for adaptive and efficient generation. Tokens represent rectangular patches of varying sizes and shapes, allocating finer tokens where detail is needed and coarser tokens elsewhere. We adapt pretrained diffusion transformers to LoT layouts through a patch-wise asymmetric flow parametrization and embeddings for multiresolution tokens, preserving full-resolution flow prediction at every denoising step while processing only a reduced token sequence. LoT Diffusion enables layout-adaptive generation while preserving pretrained generative priors. We demonstrate LoT with layouts derived from semantic masks, bounding boxes, texture variance, and depth-of-field cues, as well as agentic plans. Across image and video generation, LoT offers favorable quality-efficiency tradeoffs, with significant speedups determined by the layout's token budget. Our project website is at this https URL. 

---
# Protocol-Sensitive Evaluation of Log Anomaly Detection: Component Costs and Target-Access Sensitivity on HDFS and BGL 

**Authors**: Hang Xiao, Janet Sung, Zhaoyi Li, Gangzhen Qian, Chuhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05807)  

**Abstract**: Protocol choices can change the conclusions drawn from log anomaly detection benchmarks even when detector settings are fixed. We present a joint empirical study of split construction, representation visibility, and component costs using six fixed count, sequence, and semantic configurations on Hadoop Distributed File System (HDFS) and Blue Gene/L (BGL) logs. Random splits place several configurations near the average-precision ceiling, whereas group-disjoint HDFS and chronological BGL evaluation produce lower scores and different observed orderings. At a fixed BGL cutoff, parser choice spans 0.124 in semantic XGBoost mean average precision while preserving its lead over count XGBoost; the earliest rolling period reverses that ordering. A two-factor cross-system ablation contrasts source-only representations with offline transductive access to unlabeled target templates through the representation corpus and inverse document frequency: HDFS-to-BGL mean average precision moves from 0.191 with source-only access to 0.325 with union-corpus, target-IDF access, and the intermediate conditions reveal direction-dependent interactions in average precision and retrieval at fixed review budgets. Component-level profiling separates parsing and representation costs from classifier training, prediction, and storage. Together, these findings connect detector comparisons to the test population, preprocessing state, visible information, and measured pipeline stages, and identify the protocol fields needed alongside a score to support interpretable comparisons of log anomaly detection accuracy and resource use. 

---
# CLARA: Can AI Assess Developmental Appropriateness in Children's Stories? 

**Authors**: Sijing Yin, Zirui Wang, Qian Liu, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05783)  

**Abstract**: Assessing the developmental suitability of children's narratives is important for educational recommendation and developmental literacy research, yet such assessment typically relies on subjective and difficult-to-scale human judgment. This raises an important question: Can AI systems approximate human developmental judgments of children's stories? To study this problem, we introduce CLARA, a cognitively grounded framework for developmental narrative understanding through structured annotation across cognitive (COG), language (LAN), and social-emotional (SEL) dimensions, together with a bilingual benchmark resource containing 1107 Chinese--English children's stories with normalized silver developmental references and structured developmental annotations. We evaluate CLARA through benchmark comparison, component analysis, translated bilingual consistency analysis, and blinded human evaluation with educators. Experimental results show that structured developmental annotation achieves substantially stronger alignment with developmental references and human judgments than readability-based methods and direct prompting baselines. Overall, our findings suggest that AI systems can approximate certain aspects of human developmental judgment when guided by structured developmental annotation, while also highlighting the importance of interpretability and human oversight in educational NLP. 

---
# Relational Synthesis: Structure-Mediated Concatenative Synthesis for Foley and Retrieval-Augmented Audio Generation 

**Authors**: Keren Shao, Ayaka Kawano, Shlomo Dubnov  

**Link**: [PDF](https://arxiv.org/pdf/2610.05768)  

**Abstract**: We ask: given a retrieved source audio $S$ and a separate reference audio $R$, can we synthesize novel audio $Y$ out of this pair $(S,R)$ such that $Y$ remains acoustically consistent with $S$, while not persistently copying segments of $S$ or $R$? The first clause is a well-known goal in Foley audio production, and the second is a well-known issue in neural RAG when $S$ and $R$ are naively injected into neural generators. We show that both clauses can be addressed simultaneously using a method we coin relational synthesis, a variation of concatenative synthesis where target cost is replaced by a relational Gromov-like structural cost. Rather than imitating the content of $R$, relational synthesis exploits it from the "other side of the hill": it transfers the temporal structure and directed amplitude motion of $R$ to reorganize and concatenate the grains of $S$ in a novel manner that protects $S$'s acoustic information. Our experiments show that relational synthesis integrates naturally with neural RAG and produces Foley audio that performs well on metrics measuring temporal agreement, acoustic fidelity, and leakage persistence, while maintaining distribution-level quality and text alignment. 

---
# Beyond Semantic Similarity: Performance and Costs of Agentic Retrieval for Complex Tasks 

**Authors**: Reza Esfandiarpoor, Radek Osmulski, Yauhen Babakhin, Gabriel de Souza P. Moreira, Oliver Holworthy, Jie He, Ronay Ak, Jiarui Cai, Ryan Chesler, Bo Liu, Even Oldridge  

**Link**: [PDF](https://arxiv.org/pdf/2610.05750)  

**Abstract**: Modern information systems, including many agentic workflows, use dense retrieval to explore large amounts of unstructured data. However, dense retrieval relies on surface-level semantic similarity, which is insufficient for increasingly complex search applications. Here, we investigate agentic retrieval that combines the reasoning capabilities of Large Language Models (LLMs) with the efficient corpus exploration of retrievers in a ReAct agentic loop to solve complex retrieval tasks. In our experiments, we show that agentic retrieval is more effective than standard retrieval, improving nDCG@10 by 8.7 points using the same embedding model. Moreover, while specialized retrieval methods struggle on out-of-domain tasks, agentic retrieval is highly generalizable: the same pipeline achieves competitive results on both the ViDoRe v3 and BRIGHT leaderboards. However, this improvement comes at a cost. On average, agentic retrieval takes 107.4 seconds, compared to 0.67 seconds for standard retrieval, and consumes 764.1K input and 5.8K output tokens per query. In short, our study demonstrates the effectiveness of agentic retrieval in modern data systems and motivates future work on more cost-efficient retrieval agents for large-scale deployment. 

---
# CIPHER-MoE: Balancing Efficiency and Routing Fidelity in Trillion-Scale MoE Training 

**Authors**: Jing Li, Jian Meng, Yingmeng Gao, Suming Qiu, Linyuan Qiu, Dongfang Li, Baotian Hu, Binfan Zheng, Rongqian Zhao, Weijian Sun, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.05744)  

**Abstract**: Mixture-of-Experts (MoE) has been widely adopted in recent large language model (LLM) architectures. However, scaling up MoE in LLM training introduces system-level challenges on training, where non-uniform token routing can lead to highly imbalanced workloads across experts and devices, further destabilizing the training process. With trillion-scale LLMs, imbalanced expert workloads further amplify the resource cost of MoE training, resulting in degraded training efficiency and hardware utilization for underloaded experts, while hot experts require additional resources to accommodate excessive workloads. Recent studies address imbalanced MoE training through intricate parallelism strategies or resource reallocation. However, these system-level approaches often introduce additional resource requirements and considerable orchestration complexity, which become increasingly difficult to afford when training trillion-parameter LLMs under constrained computational resources. This work introduces CIPHER-MoE, which mitigates MoE workload imbalance while keeping the router's token-side Top-K selection unchanged. CIPHER-MoE applies affinity-aware Expert-to-Token filtering with explicit capacity control to reduce hotspot expert workloads without additional hardware resources or complex runtime design. The proposed method has been evaluated on large-scale MoE models, including DeepSeek-V4-Pro, showing up to 64.9 percentage points Top-1 expert workload reduction and 1.10$\times$-1.94$\times$ training acceleration, while preserving the training quality. The source code will be released soon. 

---
# HLA-WM: Hybrid Linear Attention for Long-Horizon Video World Models 

**Authors**: Zhuokun Chen, Feng Chen, Xi Lin, Xiyu Wu, Jiahao He, Jianfei Cai, Bohan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05739)  

**Abstract**: Long-horizon video world models require persistent memory to preserve scene consistency over extended rollouts. Softmax attention retains the full generation history through a growing KV cache, whereas recurrent linear attention compresses history into fixed-size states with substantially lower memory cost. However, we identify severe long-range forgetting in Gated DeltaNet (GDN), where information from distant but relevant scenes is progressively attenuated by subsequent state updates. To address this limitation, we propose HLA-WM, a training-free hybrid linear-attention framework that combines coarse-grained geometry-guided retrieval with fine-grained recurrent linear-state computation. HLA-WM exploits the affine structure of GDN to cache compact chunk-wise transition summaries, retrieve scene-relevant historical chunks using camera geometry, and recompose them into query-specific recurrent states. On the $60$-second SANA-WM-Bench, HLA-WM improves all six aggregate revisit-consistency and camera-control metrics of the base autoregressive generator without additional training, including a $0.74$ dB PSNR gain and a $28.5\%$ reduction in rotation error. The improvements persist after downstream refinement and generalize to MBench-A, where HLA-WM consistently improves all three revisit-consistency metrics across all four subsets and all evaluated inference modes over $547$ samples. At a $60$-second context, HLA-WM reduces historical-state memory by $12\times$ relative to full KV caching while incurring at most a $1.6\%$ reduction in inference throughput. These results demonstrate that selectively addressable recurrent memory can improve long-range scene recall while preserving the efficiency advantages of GDN. Project page: this https URL 

---
# Rotated, but How Far? Diagnosing and Improving Object-Rotation Reasoning in VLMs 

**Authors**: Zhaochen Wang, Yujun Cai, Huangbo Zou, Hower Yang, Naipeng Dong, Miao Xu, Haibin Ling  

**Link**: [PDF](https://arxiv.org/pdf/2610.05715)  

**Abstract**: Vision-language models (VLMs) can detect that an object has rotated across views, but cannot reliably tell by how much. We introduce OR-Bench, a fine-grained benchmark for object-rotation reasoning with eight tasks covering rotation detection, rotation magnitude estimation, and multi-view rotation reasoning. Across 12 VLMs, the gap is stark: the strongest models approach 100% accuracy on detection, yet even coarse magnitude estimation is near chance. When asked for exact angles, models place 91.8--100% of their predictions on just $0^\circ$, $90^\circ$, and $180^\circ$, a failure we term canonical-angle collapse. This collapse persists even without visual input. Representation probing shows that missing information is only part of the explanation. Although rotation information becomes less recoverable at finer granularity, substantial coarse-grained information remains, and a simple linear probe outperforms the models' generated answers. This suggests that VLMs underuse rotation information they already encode. We therefore propose RotationCue, a lightweight decoder that recovers coarse rotation information from the VLM's own frozen representations and feeds it back to the model as intermediate textual context. Across three VLMs, RotationCue improves every model--task combination on OR-Bench, raising macro-average accuracy by 7.9--12.6 points while preserving general capabilities. 

---
# SimpleMark: Fast Multi-Bit Text Watermarking under f -Divergence Constraints 

**Authors**: Benjamin D. Kim, Wanrong Zhang, Weitong Ruan, Lav R. Varshney, Daniel Alabi  

**Link**: [PDF](https://arxiv.org/pdf/2610.05712)  

**Abstract**: We introduce a framework for multi-bit text watermarking with security defined directly through $f$-divergence from the base language model distribution. Unlike prior approaches that focus on average-key distortion-freeness or a particular statistical distance, our formulation supports general $f$-divergences, including total variation and KL divergence, and enforces the guarantee for each realized key and embedded message. We develop a coding-based watermarking scheme that optimally biases next-token distributions subject to a prescribed divergence budget, and characterize the resulting tradeoff between embedding rate, decoding reliability, and statistical security. Experimentally, we compare our method against prior multi-bit watermarking schemes across modern language models and payload regimes. Our approach achieves substantially lower watermark detectability while maintaining competitive message-recovery performance and generation quality. Our results provide a unified view of secure multi-bit watermarking and recover several commonly used security notions as special cases. 

---
# Nexus: An Execution Fabric for AI Agents Across Cloud, Edge, and Devices 

**Authors**: Cary Chang, Jialin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.05709)  

**Abstract**: Language-model agents are evolving into long-running services that interact with models, tools, computers, mobile devices, and distributed environments. Existing agent frameworks simplify reasoning and tool invocation. However, cloud-centric designs face three limitations: centralized execution increases failure impact, scaling pressure, and compute cost; extending agents across computers, mobile devices, and edge environments requires a unified execution abstraction with permission control; and long-running executions require consistent lifecycle management across failures, recovery, results, usage, and settlement. We present Nexus, a cloud-edge platform that treats each invocation as a persistent task. Nexus uses an OpenWrt-based runtime for distributed serving, run-scoped delegation for authorized access to Computer and Mobile environments, and persistent records to track execution, outputs, failures, recovery, usage, and charging across cloud and edge components. We evaluate Nexus on controlled, cross-device, and model-driven workloads. All ten Computer-Android workflows succeed, and all six revocation tests block subsequent writes while preserving prior authorized reads. Under worker loss, journaling eliminates duplicate appends (six to zero per task), adding 0.933 s mean normal-path overhead. Across 24 matched task pairs, Nexus completes 24 tasks versus Dify's 22 and is a median 3.88 s faster on jointly successful pairs. In a separate workload, Nexus operates under a smaller tested incremental-runtime memory ceiling than Dapr (16 versus 64 MiB), although Dapr achieves lower successful-call latency. These results demonstrate how locality, operation-scoped authority, and persistent result identity support cloud-edge agent services with workload-dependent costs. 

---
# Two-Sample Testing via Path-based Inference 

**Authors**: Eshant English, Wei-Cheng Lai, Yanfeng Yang, Kenji Fukumizu, Taiji Suzuki, Christoph Lippert  

**Link**: [PDF](https://arxiv.org/pdf/2610.05684)  

**Abstract**: Modern deep generative models are primarily studied for their ability to generate realistic samples, yet the generative dynamics they learn can also serve as objects of statistical inference. We develop this idea for two-sample testing, the problem of deciding whether the same distribution generated two finite datasets. Using stochastic interpolants, we connect both distributions to a shared Gaussian bottleneck, so that each half of the resulting path is a Gaussian channel acting on a single population. We prove that the null hypothesis holds if and only if the population denoiser, or equivalently, the velocity fields of the two halves, coincide at any single noise level, which amounts to a reflection symmetry of the path about the bottleneck. Deviations from this symmetry yield a continuum of two-sample witnesses, which we estimate via held-out regression risks on learned denoisers and velocities and aggregate along the path; under an information-theoretic weighting, the aggregated discrepancy equals the Jeffreys divergence between the noise-smoothed distributions. Calibrating the resulting statistics by permutation yields tests that are valid in finite samples for any trained networks and consistent when the fields are learned accurately. On a synthetic benchmark and three image benchmarks, the proposed tests improve power over the strongest baseline by up to 33 percentage points at an equal total sample budget, with the best choice of regression representation and path weighting depending on the data modality. These results show that generative paths provide a principled representation for statistical testing, extending stochastic-interpolant models beyond generation. 

---
# Knowing the Rules, Applying the Rules: Evaluating Language Models on Traditional Chinese Bazi 

**Authors**: Jiulin Li, Ping Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05682)  

**Abstract**: Knowing domain rules does not guarantee applying them to a case. We study this distinction in traditional Chinese Bazi through 3,000 Chinese multiple-choice questions spanning 14 Theory and 11 Case categories. Six endpoint systems are evaluated, with primary results reported on a 2,492-item model-informed refinement. Theory accuracy exceeds Case accuracy for every system, and gaps of 16.60-29.56 percentage points remain when invalid responses are excluded. The contrast is more specific than a general case-reasoning deficit. Across six systems, Twelve Stages and Nayin reach mean accuracies of 89.10% and 88.62%, while Shensha Basics reaches 75.96%. Within Case, Luck Pillars averages 84.62%, but Career and Family Relations average only 36.98% and 38.19%. Overall rankings also conceal different category strengths. On the original 3,000 items, paired DeepSeek native/disabled comparisons associate native configurations with Theory gains of 6.53 and 12.20 points for Flash and Pro, respectively; Case changes are -3.67 and +1.27 points. These are provider-configuration associations, not isolated causal effects of reasoning. The results motivate task-specific evaluation of cultural-domain applications rather than reliance on aggregate knowledge scores. The benchmark measures agreement with a model-generated, model-verified answer key, not real-world predictive validity. Final-set results are post-selection descriptions, and incomplete provenance and expert validation constrain their interpretation. 

---
# Transformer-Based Time-Series Inference of Lindblad Dynamics in Open Quantum Systems 

**Authors**: Julian Guam, Jianqing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05647)  

**Abstract**: The Lindblad master equation is the standard framework for describing the non-unitary evolution of open quantum systems, where environmental interactions induce dissipation and decoherence. When both the system Hamiltonian and the dissipation rates are partially unknown or explicitly time-dependent, traditional analytical inversion and system-identification techniques become intractable. Recent works have demonstrated that Transformer-based models can infer unknown dissipation rates from observable time series, yet these approaches typically rely on hand-crafted statistical features under idealized and highly restricted conditions. Here we advance the paradigm by introducing a raw time-series Transformer that directly ingests the full trajectories of Pauli expectation values $\langle\sigma _x(t)\rangle$, $\langle\sigma _y(t)\rangle$, and $\langle\sigma_z(t)\rangle$, thereby fully exploiting the self-attention mechanism for temporal modeling. The architecture is further extended to jointly learn unknown Hamiltonian parameters, handle multiple dissipation channels, and operate robustly under realistic measurement noise. Across all tested scenarios the model achieves consistently high reconstruction accuracy while eliminating manual feature engineering. This provides a scalable, robust, and versatile framework for quantum environment sensing in realistic open quantum systems. 

---
# UndoBench: Separating Task Competence from Recovery Capability in Tool-Using AI Agents 

**Authors**: Dolly Sah, Tanmay Sah, Harshul Jain, Tanya Sah  

**Link**: [PDF](https://arxiv.org/pdf/2610.05622)  

**Abstract**: Tool-using AI agents are increasingly deployed across enterprise software systems, yet widely used benchmarks primarily evaluate nominal task completion, conflating baseline planning competence with operational fault recovery. We introduce UndoBench, a benchmark spanning 36 base workflows and 36 fault scenarios across 8 enterprise domains, decoupling task competence from recovery capability via counterfactual paired trials under identical seeds alongside wire-level effect-history and environment-state oracles. On 12 held-out TEST workflows across two open-weight models, two frameworks, and three recovery paradigms (5,760 executions / 2,880 paired trials) in the frozen lost-acknowledgment study, nominal competence reached 83.54% while conditional recovery success rate (CRSR) fell to 46.72%, with naive retry producing duplicate external effects in 53.33% of trials. Extensions to commercial API models reproduced this competence-recovery separation. Evaluations across complementary execution boundaries show that recovery is phase-dependent: before mutation, methods perform similarly without duplicate effects among capable trials; during partial mutation, naive retry, per-call idempotency, and zero-privilege journaling collapse on the evaluated composite workflows; after commit but before acknowledgment, verification and server-side idempotency substantially improve safety. These findings demonstrate that evaluating nominal completion alone masks critical, phase-dependent recovery vulnerabilities in autonomous agents. 

---
# SEA-LM: Egocentric Spatial Audio Understanding for Wearable Microphone Arrays 

**Authors**: Sonal Kumar, Sinan Hersek, Artem Dementyev, Mengzhen Pan, Ishan Chatterjee, Anurag Kumar, Ramani Duraiswami, Dinesh Manocha, Andrea Colaco  

**Link**: [PDF](https://arxiv.org/pdf/2610.05610)  

**Abstract**: Embodied, ego-centric intelligence fundamentally requires the ability to comprehend spatial audio within complex environments. While large audio-language models excel at mono-channel reasoning, they lack spatial awareness, discarding critical spatial cues that enable sound localization and that can improve the disentanglement of overlapping sound sources. To address this, we present SEA-LM, a Spatial Audio Understanding model. First, we introduce FOACODER, a layout-flexible spatial audio encoder trained on source localization and ego-centric voice activity detection objectives to encode First Order Ambisonics derived from variable-count, variable-position smart-glasses arrays via beamforming. We train a Multimodal Large Language Model (MLLM) to understand these spatial audio embeddings through a two-stage curriculum spanning six tasks, including sound localization and spatially selective transcription in settings with multiple speakers and overlapping sounds. To prevent the transcription outputs from dominating the next token prediction loss and overwhelming the direction predictions, we introduce a Spatio-temporal Weighted Cross-Entropy Loss. On our evaluation set, SEA-LM achieves lower azimuth and elevation MAE, higher temporal IoU, lower external-source hallucination and missing-source rates, and lower WER on most transcription tasks than compared baselines, while remaining robust across 1,211 smart-glasses array configurations with 4 to 9 microphones. 

---
# Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation 

**Authors**: Team Kandinsky, Julia Agafonova, Bulat Akhmatov, Mikhail Aksyutin, Grigorii Alekseenko, Anastasia Aliaskina, Olga Androsova, Vladimir Arkhipkin, Anna Averchenkova, Alexander Belykh, Serafima Bocharova, Sofiya Bogakovskaya, Anton Bukashkin, Mark Bulygin, Kirill Buzygin, Irina Cheremnykh, Kirill Chernyshev, Mikhail Chernyshov, Vladimir Chernyy, David Chikovani, Georgy Daniltsev, Denis Dimitrov, Anna Dmitrienko, Vladimir Dokholyan, Sergey Emelyanov, Dmitry Ermilov, Georgii Fedorov, Polina Gavrilova, Nikolai Gerasimenko, Aleksandr Gordeev, Andrey Inozemtsev, Andrei Ivaniuta, Alexander Ivanov, Mikhail Karaev, Anastasiia Kargapoltseva, Ivan Kirillov, Nikita Kiselev, Valeria Kobenko, Yury Kolabushin, Denis Koposov, Anatoly Korobov, Vladimir Korviakov, Kirill Kozlov, Denis Krzhivokolskiy, Konstantin Kuklev, Alexander Kunitsyn, Sergey Kuzin, Vladislav Lakhtionov, Alexey Letunovskiy, Maxim Litvinov, Alexander Lyulkov, Georgy Makarov, Kirill Malakhov, Egor Malykh, Mikhail Mamaev, Dmitrii Mikhailov, Polina Mikhailova, Ivan Mikheev, Elizaveta Muromtseva, Nikolai Nazarkin, Tatiana Nikulina, Lev Novitskiy, Stanislav Onuchin, Nikita Osterov, Denis Parkhomenko, Anatoliy Parpara, Vladimir Polovnikov, Konstantin Reznikov, Azat Saginbaev, Nikita Samsonov, Alexander Sentsov, Nikita Shaimov, Artem Sherstyuk, Andrey Shutkin, Egor Silvestrov, Bulat Suleimanov, Matvey Suprunov, Sergey Taranov, Irina Tolstykh, Tatiana Trofimuk, Ilya Trushkin, Aleksandra Tsybina, Olga Varlashina, Viacheslav Vasilev, Ilya Vasiliev, Eugeny Vilisov, Sergey Yakubson, Konstantin Zakharov  

**Link**: [PDF](https://arxiv.org/pdf/2610.05608)  

**Abstract**: We present Kandinsky 6.0 Video, a family of foundation diffusion models for synchronized text-to-audio-video generation, comprising Kandinsky 6.0 Video Lite (3B parameters) and Kandinsky 6.0 Video Pro (29B parameters). Both models generate 5-second video clips with synchronized 44 kHz audio, including lip-sync, in text-to-audio-video (T2AV) and image-to-audio-video (I2AV) modes; a built-in super-resolution model raises the output resolution to Full-HD (1920$\times$1080). Building on the video generation capabilities of Kandinsky 5.0, Kandinsky 6.0 Video employs a dual-stream CrossDiT architecture that connects a pretrained video stream and a newly trained audio stream through bidirectional cross-attention for temporal and semantic alignment. Our continuous pretraining strategy first trains the audio stream from scratch on large-scale audio corpora and then trains both streams jointly on paired audio-video data while preserving unimodal fidelity; pretraining is followed by supervised fine-tuning, reinforcement-learning-based post-training, and distillation. In side-by-side human evaluation, Kandinsky 6.0 Video Pro clearly outperforms its predecessor, Kandinsky 5.0 Video Pro, and remains competitive with leading audio-video generation models, particularly in speech quality. To accelerate open research and deployment in multimedia generation, we release the code, model checkpoints, and diffusers integration under the MIT license. 

---
# Your Unlearning Gives You Away: Identifying Erased Concepts in Diffusion Models 

**Authors**: Kaiyuan Deng, Yuchen Li, Yang Xiao, Bo Hui, Geng Yuan, Xiaolong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.05601)  

**Abstract**: Existing attacks on unlearned diffusion models assume that the erased concepts are known in advance and focus on recovering them. In practice, however, model providers may not disclose which concepts have been removed, and even with access to the original base model, an adversary may still lack a clear target to attack. In this paper, we aim to answer the following critical but overlooked questions: which concepts have been erased from the model, and how many have been erased in total? To this end, we present Tracer, a framework that rapidly and accurately identifies erased concepts and estimates their number. Tracer efficiently identifies erased concepts without generating and classifying images. By combining lightweight spectral analysis of weight footprints, it enables efficient search over large candidate vocabularies. To distinguish multiple erased concepts, we introduce a footprint coverage objective that guides sequential discovery. Tracer estimates the number of erased concepts by detecting a sharp decline in candidate confidence as the selected concepts account for the erasure footprint, without requiring labeled examples for calibration. The framework requires only lightweight linear algebra and limited forward probes, with no prior knowledge of the unlearning algorithm. Experiments across text-to-image and text-to-video backbones and diverse unlearning methods demonstrate that Tracer identifies erased concepts and estimates their number in seconds, achieving 150 to 137,000 times and 133 to 20,000 times speedups over MIA and brute-force search on image and video models, respectively, with substantially higher identification accuracy. 

---
# An LLM-in-the-loop RL Framework for Bioinformatics Feature Selection 

**Authors**: Xinyuan Wang, Deepti Agrawal, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05600)  

**Abstract**: High-dimensional bioinformatics data, characterized by a large number of features relative to the number of samples, pose major challenges such as the ``curse of dimensionality,'' leading to overfitting, high computational cost, and poor generalization. Traditional feature selection methods often suffer from limited scalability and adaptability in such domains. We propose an LLM-in-the-loop reinforcement learning (RL) framework for bioinformatics feature selection, where the RL agent formulates feature selection as a sequential decision-making task, while the large language model (LLM) enhances the process in two ways: (1) guiding exploration through domain-informed advice, and (2) providing hybrid rewards that integrate data-driven performance with knowledge-driven evaluation. The LLM also produces explanations to improve interpretability for human experts without altering the RL policy update. Experiments on diverse bioinformatics datasets show that the LLM-in-the-loop framework outperforms baselines, achieves stable performance across downstream models, and converges faster than pure RL. 

---
# When the Cross-Silo Federation Goes Offline: Continual Learning for Site Onboarding with Limited Unlabeled Data 

**Authors**: Ahmadreza Eslaminia, Klara Nahrstedt, Chenhui Shao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05598)  

**Abstract**: An organization often holds too little labeled data to train a model that generalizes, and the records that would supply the rest sit with organizations that cannot release them. Cross-silo federated learning offers a way through, since participants exchange model parameters rather than records, but it ordinarily settles two aspects of the arrangement in advance, the participating sites and the classes the model can predict, and deployment can breach both. A new site joins after training, once the established sites have finished their engagement and gone offline, and its records arrive unlabeled, mixing conditions the model already recognizes with conditions no participant has observed. We present an autonomous three-stage procedure that expands the model entirely at the joining site: reconstruction experts screen for novelty, clustering separates the flagged records into candidate conditions, and class means describe the old classes, all inside one shared representation. Those classes were learned from records that never leave their owners, so the usual defenses against forgetting are unavailable, and the procedure supplies the evidence they would have carried from either of two dissimilar sources, prototypes held by the federation or records held by the joining site. On a real industrial condition-monitoring dataset, run end to end with no label consulted, either source holds old-class accuracy at 0.868 or above with forgetting at most 0.063, and the two differ by 0.021, so a configuration can be chosen by the disclosure it permits rather than the accuracy it delivers. Both keep old- and new-class accuracy in balance where every alternative we measure gives up one for the other, and both retain more of the old classes than distillation- and regularization-based baselines. The balance still holds with only 6 labeled records per arriving condition and 3 retained per old class. 

---
# AgentDoxx: Agentic Re-identification of Anonymized Text with Web Search 

**Authors**: Jianing Wen, Tianshi Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.05586)  

**Abstract**: As Large Language Models (LLMs) gain tool use capabilities such as web search, they can retrieve and cross-reference public information, creating privacy risks beyond memorization. One manifestation is re-identification: linking an anonymized interview transcript to a named individual. Yet without ground-truth identities, the coverage of such attacks and the protection offered by a defense cannot be reliably measured. We introduce AgentDOXX, an evaluation suite of 822 synthetic interview transcripts grounded in public information about real individuals with known identities. We evaluate fifteen configurations of open-weight and proprietary models, isolating the effect of web search, and analyze their search trajectories to distinguish retrieval-driven from parametric identifications. Ground-truth identities reveal that re-identification risk is distributed across an agent's execution: retrieval and parametric recall both contribute, with open-weight models identifying 15-28% of transcripts without search; identification succeeds in over 88% of cases once the target appears in a retrieved result; entity masking leaves at least one attacker successful on 85.3% of a stratified sample; and privacy instructions suppress naming but not retrieval, with configurations scoring 0% accuracy yet retrieving the subject in up to 62% of transcripts. We further show that observed attack trajectories can provide supervision for localizing identifying spans, offering a path toward attack-informed anonymization. 

---
# Expanding LLM Reasoning 

**Authors**: Rian Atri, Evan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.05584)  

**Abstract**: Extra inference compute is usually spent on sampling more reasoning chains. We study where inside an existing chain an additional continuation should begin. We define expansion utility, the change in correctness from restarting a chain at a stored step, and measure it at every eligible step for nine models on six benchmarks (41 model and benchmark cells). Restart position matters: steps selected on one set of continuations beat uniform placement when scored on disjoint ones, in held-out audits on 5, 16, and 38 cells (+4.25 points [+2.51, +6.63] in a fresh five-cell audit). A fixed rule that restarts from the last eligible steps, always-last, is a strong baseline: our learned router beats uniform placement but shows no detected gain over it, and on DeepSeek-R1-Distill-Qwen-14B/MATH-500 always-last exceeds the exact self-consistency frontier at matched aggregate generated output by +0.052 [+0.008, +0.098], using 0.774x the aggregate generated output of four-sample self-consistency. Cross-fitted oracle selection still finds held-out headroom beyond declared positional classes, a target for future selectors. Finally, breaking step-label ties by earliest index flips the sign of a pointwise selector's gain over uniform placement in every seed of a five-seed diagnostic with four rollouts per step; randomized ties remove the bias. 

---
# What Does an Observability Foundation Model Know? 

**Authors**: Dhyey Dharmendrakumar Mavani, Rian Atri, Tairan Ji  

**Link**: [PDF](https://arxiv.org/pdf/2610.05577)  

**Abstract**: A linear probe can show that a label is recoverable from a model's hidden states, but not whether that goes beyond what the input already reveals, or whether the model uses it. We audit Toto, an observability forecasting foundation model, on the Benchmark of Observability Metrics (BOOM) across five series-disjoint resplits, comparing linear probes on its frozen residual stream with models that read the raw input window and with Toto's architecture stripped of its trained configuration. Short-vs-medium cadence and metric type are more linearly recoverable from Toto's residuals than from the strongest raw-window model in every resplit (macro-F1 0.766 vs. 0.633 and 0.545 vs. 0.498). Domain is nearly tied, and series cardinality is recovered far better from the raw window. MOMENT-base shows related cadence, metric-type, and domain readouts. Recoverability is not use: exchanging Toto's residuals with those of high-burst donors moves a future-burstiness readout as intended but does not make forecasts consistently burstier than a randomized donor. A BOOM-trained coordination probe has negative zero-shot R^2 on the tested external benchmarks. We report each label against its strongest baseline. 

---
# Can Prosodic Style Be Inferred from Text Alone? Evidence from Unsupervised Acoustic Clusters 

**Authors**: Abdul Rehman, Jian-Jun Zhang, Xiaosong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05575)  

**Abstract**: Much of expressive text-to-speech research rests on an untested assumption that written text carries enough information to select an appropriate prosodic style for its delivery. Text-predicted style models improve listener preference, and expressive-appropriateness evaluation presupposes that context constrains style, yet neither measures the assumption itself. This paper tests it as a falsifiable hypothesis against style labels derived from acoustics alone. For each of six speakers in a 1,200-hour conversational corpus, utterances are clustered in the spaces of five speech models, including a prosody-only control, and the cluster of held-out utterances is predicted from twelve text embedding models. Three controls are applied: utterance length is erased from the speech embeddings; accuracy is scored against the majority-class floor of unbalanced clusters rather than uniform chance; and a bag-of-words baseline measures word identity alone. Text predicts the cluster above that floor for all six speakers (+0.111 top-3 accuracy), but bag-of-words achieves three quarters of this. Sentence embeddings add only +0.026, largest for encoders not trained for sentence semantics and reversed by tree-based probes for all others. Acoustic clusters are not compact in text embedding space in any of 360 configurations. The prosody-only space weakens the association for five speakers, but not for the speaker showing it most strongly. Text thus informs these delivery clusters mainly through word choice, whether as a cue to prosody or as a marker of topic and recording situation, and reference-free style selection cannot assume more. 

---
# Scenario-Based Compositional Statistical Model Checking for Safety Specifications 

**Authors**: Abhinav Pomalapally, Arya Raeesi, Kevin Kai-Chun Chang, Beyazit Yalcinkaya, Sanjit A. Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2610.05571)  

**Abstract**: In safety-critical domains such as autonomous driving, systems must be evaluated across a large number of environment conditions, often represented as composite scenarios built from primitive scenarios. Existing statistical model checking (SMC) approaches analyze each composite scenario independently, requiring many expensive simulations and resulting in substantial redundant computation when scenarios share common structure. This work introduces a scenario-based compositional SMC framework for safety and co-safety specifications, enabling efficient analysis of composite scenarios. Our approach decomposes scenarios into primitives and specifications into sub-specifications, verifies each primitive independently, and composes the resulting statistical estimates using importance sampling and kernel density estimation. Our empirical evaluation shows that the proposed framework can accurately answer verification queries for previously unseen composite scenarios while reducing simulation cost through parallelization and trace reuse. 

---
# G-CARB: Graph-Localized Conformal Agent Risk Budget for Compositional Harm 

**Authors**: Zijun Yu, Yu Gu, Vahid Partovi Nia, Masoud Asgharian  

**Link**: [PDF](https://arxiv.org/pdf/2610.05563)  

**Abstract**: Small language model (SLM) agents need safety controls that track consequences across tool calls with little monitoring overhead. A private read, for example, becomes a leak when a later action sends that data outside the system. We introduce CARB (Conformal Agent Risk Budget), which calibrates when to stop an agent using a ledger of harm incurred before stopping. Under exchangeable episodes, standard conformal risk control bounds this declared loss in expectation over calibration and a future episode. G-CARB selects scorer evidence along observable dependencies from private sources to outgoing actions. The ledger still covers the entire executed history, and computing the gate score requires no additional language-model inference. On AgentDojo replay with two 14B backbones, G-CARB roughly halves scorer-input records at intermediate risk budgets while improving autonomous task completion relative to full-prefix scoring; random context of the same size achieves similar gains. Controlled examples show how retaining the relevant dependency can further avoid stopping benign work. 

---
# Cut Binary Cross Entropy: Efficient Large-Vocabulary Loss and Gradient Kernels for Sequential Recommendation 

**Authors**: Yaoyiran Li, Haowen Ning, Mohamed Hammad  

**Link**: [PDF](https://arxiv.org/pdf/2610.05559)  

**Abstract**: Industrial sequential recommender systems operate over massive item catalogs (e.g., 10^5--10^7 items). Multi-label recommendation models are trained with Binary Cross-Entropy (BCE) loss over the full vocabulary, but standard BCE materializes a dense [B, N, V] logits tensor in High Bandwidth Memory (HBM), incurring prohibitive $O(BNV)$ memory and fatal Out-Of-Memory (OOM) errors. While chunked loss optimizations exist for Softmax Cross-Entropy in LLMs, large-scale multi-label BCE optimization remains unexplored across deep learning ecosystems.
We propose CutBCE, an exact, hardware-accelerated BCE loss and gradient operator implemented in JAX and Pallas for large-vocabulary workloads. CutBCE introduces (1) an exact fused reformulation evaluating dense background loss and sparse target corrections; (2) a custom Vector-Jacobian Product (VJP) with a dedicated Pallas TPU backward kernel computing logit tiles on-chip in both passes so logits and their gradients never reside in HBM; (3) dynamic VMEM budgeting and sharding-aware collective hoisting for distributed meshes; and (4) count-based zero-overhead training metrics. On single-chip TPU v5e/v6e mini-benchmarks, CutBCE eliminates OOM errors with up to 91.9% speedup. On 8-chip TPU slice training for multi-label SASRec with 876k items (Yambda-50M), CutBCE reduces peak HBM by 65.7% (>14 GiB saved per chip) and increases training speed by 225.9% with comparable accuracy. CutBCE is open-sourced at this https URL. 

---
# Soft Strategy Selection for Batch-Mode Active Learning 

**Authors**: Rushil Gupta, Romain Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2610.05544)  

**Abstract**: Real-world deployment of active learning typically forces practitioners to choose an acquisition strategy before any data is labeled. This is a daunting task: strategy performance varies widely across settings (e.g. datasets, surrogate models) and cannot be assessed without deployment. Existing strategy selection methods explore one strategy from a portfolio at each round and identify the optimal one using bandit feedback or model retraining. Many acquisition rounds are therefore spent exploring strategies rather than collecting the most informative data. Such overhead is a major barrier to AL-driven design of high-throughput experiments, such as genetic perturbation screens and directed evolution, where AL runs consist of only a few rounds with large batch sizes. This regime permits a natural alternative: acquiring data using multiple AL strategies within a single batch. We refer to this as soft strategy selection and introduce FractAL, a method specifically designed for this task. FractAL infers a per-strategy reward using influence-function-based data attribution, which requires no additional retraining, and then computes budget shares for each strategy in the portfolio using online mirror descent. We benchmark FractAL across 7 setups spanning classification, regression, and genetic perturbation effect prediction. The results highlight that strategy selection is a hard problem: every existing method performs worse than random sampling on at least one setup. FractAL, however, matches or outperforms every baseline, including random sampling, on all 7 setups. Its allocations concentrate budget on the strongest strategies in the portfolio while pruning the weakest. FractAL is therefore a reliable choice for real-world deployments, where the optimal strategy is unknown in advance, an important step towards making AL practical for high-throughput experiments and modern scientific discovery. 

---
# Don't Judge an LLM Only by Its Activations: Discovering Suppressed Safety Features via Counterfactual Activation Potential 

**Authors**: Swadesh Swain, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2610.05541)  

**Abstract**: Mechanistic interpretability has emerged as the primary means to understand safety behavior of LLMs. However, existing tools primarily focus on the activating neurons or features of a model. The role of the remaining large set of inactive components is invisible to such methods. This work demonstrates that the inactive set contains safety-critical features that are causally relevant for refusal of harmful prompts. Suppressing such features could turn refusals into compliance, while passing undetected by prevalent interpretability tools. We introduce the Counterfactual Activation Potential (CAP), a metric that quantifies a suppressed feature's latent activation tendency as the product of its encoder alignment (how strongly the input drives it), suppression strength (how strongly active features inhibit it), and safety criticality (how much refusal depends on it). To find suppressed safety features at scale, we propose CAP-guided Safety Feature Discovery (CSFD), a two-stage filtering algorithm that identifies candidate safety features from hundreds of thousands of transcoder features without exhaustive ablation. A significant fraction of trials turn compliant with harmful prompts when a candidate feature is ablated. Under natural jailbreaks, the suppression acting on the highest-CAP features rises 2-4x, and their activation correspondingly falls by up to 80%. Amplifying a feature's suppressors pushes its activation down and raises harmful compliance with prompts related to the suppressed feature, with no such effect for random features. Our experiments span five Gemma, Qwen, and Llama models across various parameter sizes. Our findings indicate that jailbreaks could operate in part by suppressing safety-critical features rather than solely activating harmful ones, and that suppressed features are a necessary complement to activation-focused interpretability of safety behavior. 

---
# LiFT: Loop Flow Transformers 

**Authors**: Mohammad Mahdi Derakhshani, Pedro M. P. Curvo, Gertjan J. Burghouts, Jan-Willem van de Meent, Cees G. M. Snoek  

**Link**: [PDF](https://arxiv.org/pdf/2610.05538)  

**Abstract**: We introduce Loop Flow Transformers (LiFT), a family of looped generative models that scales computation by repeatedly applying a shared Diffusion Transformer (DiT) core, with only light changes to the standard architecture. Rather than asking every recurrent step for the final prediction, LiFT trains each step with a single regression target: a point on a straight path from the model's initial estimate to the flow-matching target. Because we index these targets by a continuous depth coordinate, a trained model can loop far beyond its training depth with no retraining, early exits, or other modifications. In our experiments, these longer rollouts improve generation, so inference computation can grow without adding parameters. On ImageNet at 256x256, LiFT-L/2 achieves an FID 3.34 points lower than our dense DiT-XL/2 baseline while using approximately 60% fewer parameters, 32% fewer training FLOPs, and 52% fewer inference FLOPs. 

---
# What Does a Harness Repair? A Preregistered Study of Visibility, Baseline Adequacy and Evaluation Defects 

**Authors**: Bowen Xu, Boyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.05533)  

**Abstract**: Harness search keeps a change to the prompts, reasoning switches, token budgets or parsers around a frozen model if the change raises a score. Such a gain can come from answers the parser could not read before, a weak comparison, or a defect in the evaluation. We preregistered a study of where these gains come from, with three small models, three benchmarks, replication and test partitions, a GEPA search arm and six evaluation defects injected one at a time, and we report all 47 primary endpoints. Turning thinking off raised accuracy over a capped thinking setting in 5 of 9 model-benchmark cells, and in each the gain came mostly from questions where the capped setting gave no readable answer. The thinking-off setting was not meaningfully worse than a rescue configuration or four GEPA-selected harnesses in 11 of 13 comparisons, and lost to the rescue on GSM8K for two models. GEPA repaired its broken starting points, but none of its selected harnesses was more accurate than the thinking-off setting. A thinking budget in the serving engine, which also allows a longer answer, lowered truncation and raised the parse rate in 6 of 9 cells. In 6 of 15 evaluable defect-model pairs, replication through the same pipeline reproduced the defect's distortion instead of revealing it. On the LongevityBench multiple-choice tasks, only the longevity-tuned model beat the strongest constant-label baseline. 

---
# FLEX-WAM: Flexible Block-Causal World-Action Models for Long-Horizon Imagination and Planning 

**Authors**: R. Khorrambakht, Joseph Amigo, Félix Lebel, Leon Seetoo, Jean Ponce, Zhenzhen Li, Ludovic Righetti  

**Link**: [PDF](https://arxiv.org/pdf/2610.05483)  

**Abstract**: World--action models (WAMs) promise a unified model that predicts action-conditioned futures, generates feasible actions, and supports planning in imagination. However, existing joint video--action models often use computationally heavy, fixed-horizon backbones ill-suited to streaming inference and stable long-horizon open-loop rollouts. We introduce FLEX-WAM, a Flexible and Efficient Block-Causal World--Action Model for unified simulation and policy inference. FLEX-WAM supports variable-length contexts and non-causal prediction horizons, as well as infinite autoregressive generation frame by frame or block by block. Its block-causal, KV-cacheable architecture combines axial attention and blockwise diffusion forcing to enable efficient real-time rollout and deployment-time latency--throughput tradeoffs without retraining. Joint training can nevertheless produce plausible futures that weakly respond to commanded actions. We address this failure mode by balancing state and action flow-matching gradient contributions across the state--action diffusion-noise grid and regulating world-model sampling using Forward-Dynamics (FD) elasticity, an efficient training-time proxy for action responsiveness. Across simulated and real-world datasets, FLEX-WAM achieves superior multi-step prediction quality and latency while producing stable joint state--action rollouts for thousands of steps. As a joint action proposer and simulator within MCTS, it solves long-horizon PushT and all five OGBench Puzzle-4x4 tasks entirely in imagination. On a bimanual OpenArm-based robot, a single checkpoint jointly serves as a play policy and expected-outcome predictor, enabling real-time identification and collection of model--reality mismatches for future self-improvement. 

---
# Reflecting on Creative-Boundaries with an AI Co-Doodler 

**Authors**: Samia Menon, Samyukta Jayaram, Chetan Goenka, Shm Garanganao Almeda  

**Link**: [PDF](https://arxiv.org/pdf/2610.05482)  

**Abstract**: In this pictorial, we consider how the negotiation of creative boundaries with a co-creative AI system can create moments for personal creative reflection. We ground this in our experiences with Froggi-Draw, a single-initiative co-doodling system that gives users power to decide when and how much an AI "collaborator" (Froggi) contributes to their drawing. From a 1-week pilot study where (n=8) novice and experienced artists doodled daily with the system, we report on ways participants navigated creative risk and uncertainty, and how their usage and perceptions of Froggi shifted over time. In moments of disruption, participants described the system as encroaching on their creative territory. We consider how the design of a supportive, co-creative AI "collaborator" might look like the design of a supportive power dynamic---and finding ways to offer users control to find, reflect upon, and flexibly negotiate the boundaries of that dynamic. 

---
# Population Scaling or Data Dilution? Dynamics of Local Topology Evolution in Decentralized Learning 

**Authors**: Yin-Kuan Liang, Yan Gao, Yang Long  

**Link**: [PDF](https://arxiv.org/pdf/2610.05476)  

**Abstract**: Scaling decentralized learning changes not only the number of clients $N$, but also the dynamics of information propagation and consensus. We argue that the effect of increasing $N$ cannot be understood in isolation, because data allocation, topology-dependent mixing, and communication capacity may change simultaneously. We study these coupled effects on CIFAR-10 with $N\in\{10,50,100,200\}$, comparing a degree-two Ring, a Static Random graph, and Local-First Heuristic Evolution (LFHE), a locally adaptive topology process based on friend-of-friend discovery. The Ring provides an analytically transparent failure mode: its Metropolis spectral gap decays as $\Theta(N^{-2})$, implying progressively slower contraction of model disagreement as the population grows. Experiments show that holding the nominal local dataset size fixed substantially reduces the apparent population penalty observed when a fixed total dataset is divided among more clients. The remaining degradation depends strongly on communication structure: Ring enters a high-disagreement regime, whereas Static Random and LFHE remain close to consensus. Increasing LFHE's degree threshold further improves accuracy and consensus, but at a substantially higher model-transmission cost. These results show that decentralized scaling is governed by coupled learning and communication dynamics, rather than by the number of clients alone. 

---
# Human-Like Attention? A Psychophysical Comparison of Visual Search in Humans and MLLMs 

**Authors**: Renchi Zhang, Joost C. F. de Winter, Dimitra Dodou, Harleigh C. Seyffert, Yke Bauke Eisma  

**Link**: [PDF](https://arxiv.org/pdf/2610.05463)  

**Abstract**: Visual search is a fundamental cognitive ability. This study investigates whether Multimodal Large Language Models (MLLMs) exhibit human-like difficulty signatures in visual search tasks. We compared search performance of humans (n = 1,250) and MLLMs using identical 2D and 3D stimuli across different set sizes. Both groups showed efficient performance in feature searches, most clearly when the target had a unique color, but performance degradation in conjunction searches as set sizes increased. Additionally, we found strong correlations between human and MLLM error rates ($\rho = 0.82$), which suggests that MLLMs are sensitive to similar objective complexities, such as stimulus heterogeneity. However, differences were found as well: whereas humans invested extra search time to respond accurately on target-absent trials, MLLMs exhibited extreme present/absent response biases in complex searches. We conclude that MLLMs replicate high-level human performance signatures, yet their underlying computations differ significantly. 

---
# Hierarchical Time-aware Bootstrapping for Off-Policy Subgoal Value Learning 

**Authors**: Bingyun Liu, Yuheng Jing  

**Link**: [PDF](https://arxiv.org/pdf/2610.05446)  

**Abstract**: Off-policy hierarchical reinforcement learning must estimate the values of high-level decisions while the low-level policy changes. HIRO adapts replay data through subgoal relabeling, but after a label change, the value update targets the relabeled subgoal instead of the subgoal the high-level policy originally needed to update. We propose Hierarchical Time-aware Bootstrapping (HTB), which evaluates specified subgoals under the current low-level policy while retaining accumulated task rewards. Remaining execution time distinguishes subgoal continuation from a new high-level decision. Together with primitive-action conditioning, it enables off-policy Bellman updates based on the stationary environment transition law. HTB combines these one-step updates with multi-step suffix returns and truncated relabeling, reducing dependence on intermediate value estimates. A shared value component supports learning across actions, while nonnegative residuals constrain upward corrections relative to that component. At a fixed mixture weight of 0.95, HTB achieves 32.8% AntFall success versus 9.6% for matched local HIRO over five paired seeds at 10M environment steps. Ablations identify contributions from recursive continuation and mixed supervision; fixed-policy tests show more accurate predictions for actions whose returns were excluded from fitting. 

---
# A Strong Baseline for Evaluating Vision Encoders in Multimodal Large Language Models 

**Authors**: Yilin Yang, Jun-Tao Tang, Kengyi Wang, Siyuan Su, Gaoyong Luo, Mingda Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.05413)  

**Abstract**: Evaluating vision encoders requires metrics that reliably predict their downstream performance in multimodal large language models (MLLMs). Although recent studies have shown that cross-modal metrics can better capture such performance, unimodal metrics remain the dominant choice in practice. In this work, we revisit cross-modal evaluation of vision encoders through large-scale experiments. We identify important limitations in both the experimental design and methodological formulation of prior approaches. After addressing these limitations and introducing simple improvements, we propose RAVEL, a training-free method based on cross-modal nearest-neighbor retrieval. Despite its simplicity, RAVEL achieves state-of-the-art performance across our experiments, outperforming prior methods by a substantial margin. Our results demonstrate that simple cross-modal metrics, when evaluated under a careful and comprehensive setup, can provide a strong basis for evaluating vision encoders for MLLMs. 

---
# BeliefGraph-JEPA: Structured Latent World Models for Action-Conditioned Time Series 

**Authors**: Yue Li, Kangqi Ni, Zhen Tan, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.05409)  

**Abstract**: Action-conditioned time-series forecasting requires accounting for how future actions and exogenous forcings influence multiple targets through partially observed effects with different delays and persistence. Direct conditioning leaves the evolution and target-specific influence of these effects implicit in the predictor, while static relational graphs specify connections without tracking evolving effects. This motivates representing future-driver influence through structured latent states that evolve over the forecast horizon and route information to individual targets. We introduce BeliefGraph-JEPA, a structured latent world model that factorizes driver influence into typed latent-effect states. These states are rolled forward under future drivers and routed through a graph to target-specific nodes, forming the predictive base of a joint-embedding predictive architecture. A capacity-controlled residual supplements this base with direct driver information. On four multi-target clinical, agricultural, environmental, and industrial systems, the framework outperforms a range of pretrained and supervised known-future-covariate baselines. Matched controls isolate latent dynamics, future rollout, graph routing, and residual capacity; future rollout and graph-first residual routing improve forecasting across all four systems. 

---
# No Concept Escapes the Audit: Auditing-Aware Unlearning for Verifiable Concept Erasure in Diffusion Models 

**Authors**: Kaiyuan Deng, Yuchen Li, Gen Li, Yang Xiao, Geng Yuan, Xiaoyong Yuan, Bo Hui, Xiaolong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.05401)  

**Abstract**: Text-to-image diffusion models can generate prohibited content, which motivates concept erasure through machine unlearning. Most erasure methods intervene at the text interface, through prompt modification or localized updates to text-conditioning weights, and they are evaluated by what the model outputs for given prompts. Such evaluation cannot see what the network still encodes. Latent-space auditing, which bypasses text conditioning and probes the denoising network directly, shows that erased concepts remain recoverable from internal representations. We find that this also holds for methods built to be robust against adversarial prompts, and that the problem grows with the number of erased concepts. We propose Auditing-Aware Unlearning for Verifiable Concept Erasure in Diffusion Models (AVCE), a framework that grounds erasure in the model's latent representations. AVCE audits the embedding neighborhood of each concept and condenses the discovered vulnerable directions into an anchor at the weakest geometric point. It edits cross-attention and self-attention projections in closed form at this anchor, then fine-tunes the two pathways with pathway-level auditing losses, using orthogonal gradient projection to consolidate multiple concepts. Experiments on SD v1.5, SDXL, and Flux 1.0 across object, explicit-content, and artistic-style unlearning show that AVCE reduces attack success rates by 5.07x and improves auditing scores by 3.84x over the strongest baseline, while preserving competitive generation quality. 

---
# GNN-CB: A Graph Neural Network Competition Benchmark for Human and LLM Evaluation 

**Authors**: Murad Hossen, Tasneem Selim, Gurur Gamgam, Tuga Yousif, Abderrahmane Kasmi, Ikram Aissiou, Mubaraq Onipede, Faran Taimoor Butt, Sanae Zrigui, Rosa Y. G. Paccotacya-Yanque, Ignatius Balayo, Ikram Elhouiti, Hadil Affes, Bijay Adhikari, Sargam Goyal, Muhammad Ibrahim Isah, Mohammad Idrees Bhat, Samuel Kangoni Matia, Peguy Kem-Meka Tiotsop Kadzue, Maha Trabelsi, Emmanuel Owusu, Vinit, Nour Majdoub, Tamiru Alemnew, Islem Rekik  

**Link**: [PDF](https://arxiv.org/pdf/2610.05387)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance on coding and reasoning benchmarks; however, their ability to solve graph-structured machine learning problems remains largely unexplored. In particular, no benchmark currently evaluates whether LLMs can autonomously solve end-to-end Graph Neural Network (GNN) coding tasks under realistic competition settings. To address this gap, this paper introduces GNN-CB, the first competition-based benchmark for evaluating both humans and LLMs on GNN coding tasks. GNN-CB consists of 18 curated competitions spanning node-, edge-, and graph-level prediction across diverse graph categories, domains, and difficulty tiers. All submissions are evaluated through a unified automated pipeline with hidden test sets and standardized scoring. Human participants solve tasks under controlled competition constraints, while LLMs are evaluated using a frozen zero-shot prompting protocol based on a plan-then-code paradigm with bounded execute-and-repair loops. The benchmark additionally supports both non-agent and autonomous agent-based evaluation within the same protocol. Under our evaluated protocol, LLMs rarely match Human Top performance and show less stable performance across competitions. No single model dominates: a few competitions are won by LLMs, yet humans still hold the top score on most tasks. We release GNN-CB as a living benchmark with automated evaluation infrastructure, dynamic leaderboards, and reproducible execution pipelines. Beyond benchmarking, GNN-CB provides a practice-oriented resource for studying GNN implementation across progressively diverse graph-learning tasks. The benchmark and evaluation framework are publicly available at this https URL. 

---
# Does Explainability Survive Data Drift? 

**Authors**: Samuel Ozechi  

**Link**: [PDF](https://arxiv.org/pdf/2610.05379)  

**Abstract**: Model performance monitoring is a standard practice in machine learning deployments. Detection performance is tracked continuously, and model decay is expected as the relationship between the feature and target variables degrades, a phenomenon known as concept drift. Explanation fidelity, however, is rarely monitored with the same discipline, even in domains such as financial systems, healthcare, and other regulated environments where explanations are required for governance purposes. This paper investigates whether explanations can decay under data drift, even when the feature-target relationship remains stable, and whether explanations produced before drift occurs remain faithful to the decisions of the model that replaces them. Using the IEEE-CIS Transaction Fraud Detection dataset, we find statistically significant covariate shift but no statistically significant evidence of concept drift under the implemented conditional-drift tests, thereby providing an empirical setting in which input distributional change can be studied separately from detectable changes in the feature-target relationship. Local explanations are generated with ExIFFI and evaluated at three levels: path validity, structural behaviour, and fidelity under controlled intervention. Results show that while prior explanations retain substantial decision relevance to a retrained model, they are consistently less faithful than newly generated explanations, with no evidence of a systematically widening gap across the evaluated windows. The study shows that explanation fidelity requires its own monitoring, that structural stability of explanations does not guarantee functional fidelity, and that explanations should be treated as artifacts tied to the model that produced them. 

---
# Distributed Subliminal Learning: Replacing Model Updates with Random-Carrier Outputs 

**Authors**: Dario Fenoglio, Gabriele Dominici, Martin Gjoreski, Marc Langheinrich  

**Link**: [PDF](https://arxiv.org/pdf/2610.05378)  

**Abstract**: Collaborative learning typically exchanges model parameters: federated clients communicate updates, while independently adapted foundation models are combined by exchanging adapters or checkpoints. This makes communication scale with model size and requires local specializations to be reconciled in weight space, where interference is common. We ask whether knowledge can instead be shared through model behavior on task-unrelated inputs. We introduce Distributed Subliminal Learning (DSL), a collaborative learning primitive in which participants adapt a common model locally, probe it with task-unrelated inputs, and transmit only the resulting carrier outputs. A coordinator pools these outputs and distills them into a shared model. The primitive supports one-shot foundation-model composition through carrier completions and iterative federated learning through carrier logits, without transmitting model updates or requiring task-related proxy data. In LLM composition, compared with LoRA averaging, DSL achieves higher preference retention (94.56% vs. 87.76%) and a larger GSM8K gain over the base model (22.0 vs. 0.6 points), while reducing upload by 30.6-49.0$\times$. In federated classification, DSL reaches 96.83% on MNIST with 8.9$\times$ less uplink than FedAvg and provides lower-communication operating points on CIFAR-10 and Tiny ImageNet. These results establish random-carrier outputs as a practical communication primitive for knowledge sharing across distinct collaborative learning paradigms. 

---
# AutoDP-LLM: Automating Data Pre-processing for Intrusion Detection Systems using Large Language Models 

**Authors**: Bao-Phong Nguyen, Gia-Khanh Pham, Thai-Duong Do, Mai Xuan Trang, Minh-Tuan Le, Xuan-Nam Tran, Huan Vu, Tien-Cuong Nguyen, Vu-Duc Ngo, Thien Van Luong  

**Link**: [PDF](https://arxiv.org/pdf/2610.05369)  

**Abstract**: The increasing complexity and scale of modern cyber-attacks demand intelligent and computationally efficient Intrusion Detection Systems (IDS). However, designing effective data pre-processing pipelines traditionally involves substantial trial-and-error effort and repeated evaluation of alternative configurations. For large, high-dimensional network traffic data, this process can create a significant computational burden. In this work, we propose AutoDP-LLM, an automated pre-processing framework designed to reduce manual pipeline development and computational overhead. Specifically, AutoDP-LLM leverages Large Language Models (LLMs) to autonomously generate and validate executable data pre-processing pipelines. The framework combines deterministic host-side planning with LLM-based specialist agents to formulate data-processing strategies, synthesize executable code, and adaptively determine retained feature sets using semantic reasoning and training-derived statistical evidence, without requiring a predefined feature budget. Focusing on multiclass intrusion detection, we evaluate AutoDP-LLM on the UNSW-NB15 and NSL-KDD benchmark datasets using multiple downstream classifiers. Comparative experiments against conventional feature-selection methods show that AutoDP-LLM achieves competitive detection performance while automating the generation of compact and executable pre-processing pipelines. Component-level ablation experiments further demonstrate the complementary contributions of the semantic and statistical feature-reduction components. The repeated generation, validation, execution, and assessment of candidate pipelines are amenable to parallel execution, highlighting the potential of scalable computing environments, including high-performance computing (HPC) systems, to support automated IDS pipeline development. 

---
# AIProver: Agentic Auto-Formalization of Mathematical Research via Certificate-Driven Evolving Harness 

**Authors**: Prithwish Jana, Viet Bach Hoang, Logan Luna, Viresh Pati, Akash Singirikonda, Cy Xie, Lisa Carbone, Wuyang Chen, Walter Moreira, Joe Stubbs, Sriram Vishwanath, Vijay Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2610.05367)  

**Abstract**: Proof auto-formalization translates natural-language (NL) theorems and proofs into a formal language (FL) such as Lean, enabling mechanical verification. Despite rapid progress, research-level proofs often depend on concepts missing from leading proof assistant libraries (e.g., Lean's Mathlib), and successful compilation does not guarantee that a translation preserves the theorem's meaning or the proof's reasoning. Furthermore, aligned NL-FL training data are scarce, and leading agents often rely on costly frontier models and manually engineered harnesses.
To address the above issues, we present AIProver, an agentic framework for autonomous proof auto-formalization and proof synthesis (AFPS) that jointly post-trains a 119B open-weight language model and evolves its agentic, tool-calling harness with HarnessEvolve. Verifiers assess type correctness, proof completeness, and semantic correctness, returning rewards and diagnostic certificates that drive model fine-tuning and alternating reinforcement learning via symbolic feedback and HarnessEvolve, a certificate-driven evolutionary search over the whole harness control flow that re-tailors the harness to the updated model. For research-level training and evaluation, we introduce LoCoBench, 58.9k instances from Mathlib, CSLib, Mizar Math Library, and a bounded-arithmetic textbook, with a 771-instance validation split whose theorem-proof pairs have no public Lean formalization. Against 39 frameworks spanning AFPS agents, frontier LLMs, and coding agents, AIProver lifts pass@4 semantic correctness over its Leanstral-1.5 base from 15.7% to 36.7% and outperforms every other open-weight system and Aristotle. As a Claude Code and Codex skill, it lifts their semantic correctness from 41.9% and 34.1% to 79.8% and 62.4%, respectively. Further, it is also 24% cheaper than Numina-Lean-Agent, pushing the accuracy-cost frontier of research-level AFPS. 

---
# Optimal Control with Learned Critics under Unmodeled State Dependencies 

**Authors**: Philipp Schoch, Markus Ryll  

**Link**: [PDF](https://arxiv.org/pdf/2610.05359)  

**Abstract**: Model Predictive Control (MPC) provides a structured and constraint-aware mechanism for decision-making, but its reliance on optimization-friendly analytical dynamics models limits its use in tasks with contacts and other hard-to-model state dependencies. Model-free reinforcement learning avoids explicit modeling assumptions but typically requires large amounts of interaction data. We present a learning-based MPC framework that combines the data efficiency and structure of local model-based planning with learned components that compensate for incomplete dynamics and finite-horizon myopia. The method augments a nominal analytical model with a residual dynamics network that learns missing state-dependent effects from data and combines the resulting planner with a learned action-value critic that injects long-horizon MDP structure into the local iLQR optimization. To make this practical at reinforcement-learning scale, we develop a GPU-accelerated batched iLQR solver that evaluates learned dynamics and critic networks inside the optimal-control loop and solves thousands of trajectory-optimization problems in parallel. The complete system is integrated into a robotics simulator, enabling scalable model-based reinforcement learning under incomplete dynamics. Experiments on biased and incompletely modeled control tasks show that the approach improves closed-loop control performance while preserving the model-based structure needed for efficient constrained trajectory optimization. 

---
# Rethinking Tabular Foundation Models On Data Streams 

**Authors**: Nilesh Verma, Daniel Nowak-Assis, Afonso Lourenço, Albert Bifet, Bernhard Pfahringer, Maroua Bahri, Nick Jin Sean Lim  

**Link**: [PDF](https://arxiv.org/pdf/2610.05352)  

**Abstract**: Tabular foundation models (TFMs) outperform established machine learning models on tabular benchmarks through in-context learning. Building on this success, interest is growing in applying them to data streams, where data arrive continuously and evolve over time. On a stream, a TFM adapts by updating its context rather than its parameters, so its accuracy and cost depend on which examples it keeps and how often it rebuilds its context. We therefore present a systematic study of TFMs on data streams, covering memory management, computational cost, and stream-specific challenges such as concept drift and delayed labels. We find that TFMs achieve the highest predictive performance and that simply retaining the most recent examples is as effective as existing memory management techniques. They also recover faster than streaming learners after drift and keep the highest accuracy under label delay. This accuracy, however, comes at a high serving cost, since a nearly unchanged context is re-encoded at every prediction. These results point to architectural efficiency as the way forward for in-context stream learning. 

---
# Reflections and Fragments: Securing LLMs Against Sequential Mosaic Attacks 

**Authors**: Emanuele La Malfa, Saar Cohen, Gabriele La Malfa, Mickel Liu, Christian Schroeder de Witt, Natasha Jaques, Michael J. Wooldridge  

**Link**: [PDF](https://arxiv.org/pdf/2610.05346)  

**Abstract**: Self-play red-teaming improves language-model safety by pitting attacker and defender roles against each other in a zero-sum game. However, real adversaries increasingly use mosaic attacks: multi-turn sequences whose individual fragments are innocuous in isolation yet assemble into a harmful payload. We develop a theory of mosaic defense that characterizes what is required to prevent such attacks without sacrificing helpfulness. We first show that no fixed bounded window of recent prompts is sufficient in general: safety-relevant information may occur arbitrarily far back in the interaction. We formalize a watchman, an online state mechanism that carries this information forward, and show that under explicit assumptions it enables zero-failure defense with positive benign helpfulness. Under stronger conditions, it is also optimal among zero-failure defenders. An exact watchman may nevertheless require exponentially many states, while exact maliciousness detection can require exponentially many queries in an unstructured black-box model. These state and query lower bounds do not by themselves imply hard learning: the construction underlying the state lower bound is efficiently learnable from labeled examples, whereas certifying worst-case safety can require substantially more information under restricted access. We also show that self-play equilibrium alone does not certify usefulness, motivating a constrained formulation that maximizes worst-case benign helpfulness among zero-failure defenders. Empirically, training role-specific attacker and defender LoRA adapters over frozen LLMs via multi-turn self-play strengthens both roles: attackers become more effective at eliciting harmful responses, while defenders become more robust to attack, with improvements also observed on unseen attack objectives. 

---
# IRSTD-Agent: Agentic Infrared Small Target Detection via Zoom-Guided Interaction Learning 

**Authors**: Jiawen Xi, Yu Zhang, Tianyi Zhao, Zhu Liu, Maoxun Yuan, Xingxing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.05342)  

**Abstract**: Infrared small-target detection plays an important role in maritime monitoring and aerial surveillance. Although multimodal large language models (MLLMs) offer promising capabilities for visual understanding, existing MLLM-based approaches struggle to precisely localize infrared small targets. In this paper, we propose IRSTD-Agent, an agentic framework for infrared small target detection through dynamic visual search. The framework enables an MLLM to adaptively determine where and at what scale to inspect an image and progressively gather fine-grained visual evidence for precise target localization. Five complementary visual tools (PROPOSAL, ZOOM, DETECT, DROP and REFINE) support object candidate discovery, adaptive observation, target localization, hypothesis rejection, and target extent refinement, together enabling a coordinated search process over original-resolution images. To teach the MLLMs to conduct this search, we introduce Zoom-guided Interaction Learning, which uses annotation-derived interaction trajectories to supervise tool selection and the corresponding arguments. Through extensive experiments on WideIRSTD-Full and IRSTD-1k datasets, we demonstrate that IRSTD-Agent outperforms the evaluated vision-language models and enhances the precise localization capabilities of MLLMs in IRSTD tasks. 

---
# On Semi-Markov Suboptimality in Hierarchical Reinforcement Learning 

**Authors**: Bingyun Liu, Yuheng Jing  

**Link**: [PDF](https://arxiv.org/pdf/2610.05338)  

**Abstract**: Hierarchical reinforcement learning uses temporally extended subtasks for exploration, yet committing to their execution can restrict both deployment and policy learning. We identify and separate the resulting execution and policy suboptimality. Task and execution trees distinguish reward objectives from policy choices and decision interruption. A Unified Value Function for HRL and a four-stage Generalized Hierarchical Bellman Equation then support a common analysis of both losses. Under bounded rewards and uniform termination, we establish hierarchical policy and execution improvement results. With the remaining node policies fixed, task-subtree compatibility and node-policy optimality under the original execution mode establish when Markov execution is optimal. The resulting decomposition leads to independent execution choices for behavior, targets, and deployment. We instantiate this principle through execution improvement and one-stage or two-stage policy improvement at arbitrary hierarchy depth. Option-based and goal-conditioned experiments demonstrate complementary gains from changing execution and changing the learning target. Controlled stochastic environments show how these gains depend on stochastic transition strength and spatial structure. This framework makes execution design an explicit component of hierarchical policy optimization. 

---
# Green-Routed Neural Operators:\\Physics Determines Where the Network Reads 

**Authors**: Chenhao Si, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.05337)  

**Abstract**: We identify a mismatch between the physical role of transport fields in many PDEs and their usual role in neural operators: PDEs use them to select read coordinates, whereas neural operators typically treat them only as input values. We address this mismatch with the Green-Routed Neural Operator (GRNO), which uses the governing equation to determine where latent features are sampled. A parameter-free equation adapter evaluates the diagnostic relation and constructs a departure map whose values are the read coordinates. A multiscale encoder-decoder combines centered and routed reads of latent features to learn the complete finite-time update. Across five two- and three-dimensional PDE systems, GRNO achieves the lowest mean final relative $L^2$ error on four under 40-step autoregressive evaluation and remains competitive on Keller-Segel. Fixed-weight route interventions reveal strong dependence on direction and spatial alignment in four systems, with weak dependence in Keller-Segel. In independently trained ablations, GRNO achieves lower mean errors than variants that supply the transport field only as an input feature, substitute a learned displacement for the equation-specified route, or apply the route with a spatial misalignment, across all five systems. It also substantially outperforms directly advecting the physical state and learning the remaining update, indicating that equation-specified read coordinates provide an effective structural prior for long-horizon PDE forecasting. 

---
# VAMPS: Visual and Motor Policies from Sampling-Based Planning 

**Authors**: Mohamed Yassine Kabouri, Pietro Noah Crestaz, Quang-Nam Nguyen, Qilong Cheng, Ludovic Righetti, Nicolas Mansard  

**Link**: [PDF](https://arxiv.org/pdf/2610.05331)  

**Abstract**: Learning robot policies directly on physical systems remains difficult because data collection is costly and policy exploration can be unsafe. We introduce Visual and Motor Policies from Sampling-Based Planning (VAMPS), a framework that uses Model Predictive Path Integral (MPPI) control to train reusable policies without human demonstrations. VAMPS supports two training modes. For one-step proprioceptive policies, it operates iteratively in simulation: the policy warm-starts MPPI, and the refined trajectories provide new supervision as the policy changes. A learned terminal value improves short-horizon planning, while an Implicit Q-Learning (IQL) critic guides the policy update. Iterative refinement outperforms training once on frozen MPPI data, and we transfer the learned locomotion policy to a Unitree Go2. For visuomotor policies, VAMPS operates directly from real-robot data. MPPI uses task-specific state estimates to plan and execute trajectories while recording RGB and sensor observations on a Flexiv Rizon 10S. Action Chunking with Transformers predicts action chunks, reducing the effective prediction horizon, and is trained offline on this fixed dataset. We demonstrate visuomotor pick-and-place and force-aware whiteboard erasing. In the latter task, the policy additionally observes the measured $6$-D wrench and desired normal force. These results show that VAMPS can learn policies either in simulation followed by hardware transfer or directly from autonomously collected real-robot data. 

---
# CoDance: Learning Reactive and Compliant Human-Humanoid Interaction from Video 

**Authors**: Zhuoqun Chen, Shucheng Jia, Boyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.05324)  

**Abstract**: Partnered human-humanoid interaction couples locomotion with continuous physical contact. A humanoid needs to coordinate with a person's motion while responding to interaction forces and maintaining stable and natural movement. We present CoDance, a framework for learning reactive and compliant human-humanoid interaction from video. We study partnered dancing as a challenging instantiation, where a humanoid coordinates its footsteps with a moving partner and maintains continuous two-hand contact. Given a single video of two human dancers, CoDance retargets their motions into a robot reference and a moving partner. We introduce a multi-link compliance augmentation that transforms the kinematic demonstration into force-aware training data by adapting the robot reference under structured forces at both hands. Policies trained on this data follow the observed partner while preserving the demonstrated locomotion style and responding compliantly to physical interaction. In simulation, the policies adapt their footsteps to changes in the partner and reproduce approximately 80% of the wrist displacement encoded by the augmented demonstrations. On a physical humanoid, CoDance enables sustained two-hand dancing with a human partner including repeated transitions between forward and backward motions. 

---
# Grammar-Guided Code Watermarking with Green Temperature 

**Authors**: Hyundong Jin, Hyeseon An, Soohan Lim, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.05323)  

**Abstract**: Large language model watermarking embeds detectable statistical signals during decoding, but the resulting changes to token probabilities can degrade generation quality. This trade-off is particularly important for code, where small changes in token selection can break syntax or alter program behavior. Existing code watermarking methods mitigate this risk through entropy-based insertion or syntax-aware token selection, but they do not directly construct the watermark over the set of continuations admitted by the current grammar state. We propose Grammar-Guided Code Watermarking with Green Temperature (GTCW), which integrates grammar-constrained decoding with probability-aware watermarking. At each decoding step, GTCW restricts the candidate set to grammar-admissible tokens and partitions this support into keyed green and red subsets. At eligible high-entropy positions, green temperature reweights the green tokens according to the model's relative preferences, strengthening the watermark signal while retaining the grammar constraint. Across five models and five benchmarks spanning four programming languages, GTCW achieves a mean AUROC of 73.61%, compared with 67.83% for the strongest baseline, while maintaining a mean Pass@1 of 59.18% versus 59.58% for unwatermarked generation. Our implementation is available at this https URL . 

---
# Characterizing Parallelism Strategies in LLM Inference: Fundamental Compute-Communication Trade-offs 

**Authors**: Javad Mirzaei, Jeebak Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2610.05305)  

**Abstract**: Large Language Model (LLM) inference has become the dominant workload in modern AI systems, requiring serving infrastructures to maximize throughput while meeting strict latency Service-Level Objectives (SLOs). Since state-of-the-art LLMs exceed the compute and memory capacity of a single GPU, inference is commonly distributed across multiple GPUs using tensor parallelism (TP), pipeline parallelism (PP), or hybrid parallelism (HB). However, selecting the most effective parallelism strategy remains challenging due to complex interactions among computation, communication, pipeline utilization, sequence length, batch size, and model architecture. Existing approaches largely rely on empirical evaluation and provide limited analytical insight into the trade-offs among these strategies, particularly across the distinct prefill and decoding phases of inference. In this paper, we present a unified analytical framework for modeling distributed LLM inference under TP, PP, and HB. The framework decomposes end-to-end latency into computation, inter-GPU communication, and pipeline bubble overhead, and derives analytical models that capture TP collective communication, PP point-to-point communication, and pipeline utilization as functions of hardware, model, and workload characteristics. The model further characterizes the differing execution behavior of prefill and decoding, explaining why PP-oriented configurations favor compute-intensive prefill while TP-oriented configurations reduce decoding latency by eliminating pipeline bubbles. Experiments with modern LLMs on multi-GPU platforms validate the model and confirm the fundamental compute-communication trade-off across parallelism strategies. The framework provides practical guidance for parallelism selection, capacity planning, and optimization of future LLM serving systems. 

---
# Revisiting Ground-Truth Synthesis from High-Speed Video: Exact Validity Conditions and an Audited Consumer Capture Corpus 

**Authors**: Abdullah Al Shafi, Sumaiya Rahim Suma  

**Link**: [PDF](https://arxiv.org/pdf/2610.05298)  

**Abstract**: Motion-deblurring datasets are commonly synthesised by averaging $N$ consecutive high-frame-rate frames and labelling the result with the central frame. We show that this label is unbiased for every capture timing only when the window is odd and the frames' sample durations are equal. An even window shifts every label by a fixed fraction of the blur length, even under perfect timing. Unequal durations are subtler: their mean misalignment is zero, so dataset statistics cannot reveal them, yet when the offset cannot be read from the blur they convolve the supervision rather than adding noise to it. Auditing 51 smartphone clips recorded at a nominal 240 fps, we find 19 captured near 176 fps, a behaviour recorded only in the container's timing tables. In a controlled test the parity choice costs a fitted linear deblurring filter far more than these timing irregularities do, and interpolation labels, unlike blur labels, can be repaired with the true frame times. We provide a tool that checks both conditions without decoding, and will release the clips and their timing tables. 

---
# When Verifiable Counts Depend on Wording: Auditing Wording Robustness in Instruction Following 

**Authors**: Qishi Zhan, Seoyeon Jang, Zihan Dong, Minxuan Hu, Ziheng Chen, Tonghui Qu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05278)  

**Abstract**: Verifiable instruction-following benchmarks often express each constraint through one fixed template. We test whether scores remain stable when the operational requirement is unchanged but its wording varies. We introduce WISE, a matched evaluation suite and reporting protocol instantiated on exact word count, keyword inclusion exactly once, and an inclusive 8--12 word range. Across 100 matched tasks, up to thirteen models from seven providers, and repeated generations scored over the complete visible output, wording alone produces substantial compliance shifts. In an avoidance-family panel, five avoidance and exclusion forms fall below the positive baseline, while constructional controls also shift compliance substantially: in the nine-model control panel, compliance is 54.9% for the original positive form, 48.2% for a longer positive form, 36.7% when the target appears later, and 33.8% for AVOID1. A strict JSON-structure probe shows wording sensitivity beyond counting, with a different direction of effect. Effect sizes, failure directions, weakest forms, and model rankings vary across realizations. Under the most disruptive exclusion form, the top-ranked model changes and 24.1% of strictly ordered model pairs reverse. Human validation further shows that unanimous agreement on an exact-count interpretation can coexist with substantially different model behavior. WISE supplements conventional scores with mean and worst-form compliance, wording gaps, failure profiles, and ranking stability. 

---
# Who Is Your Agent Serving? Provider-Side Indirect Prompt Injection in Proactive Agents 

**Authors**: Rui Wang, Chao Wang, Xinchen Wang, Yufeng Zheng, Binbin Liu, Yaofei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05266)  

**Abstract**: Proactive personal agents increasingly decide what to recommend, how to personalize advice, and what follow-up assistance to offer, creating a new user-decision attack surface for provider-side indirect prompt injection. We show that an external provider need not access private user context, compromise the agent, or gain additional permissions: by controlling only content associated with its own target, it can redirect an otherwise benign agent to advance that target, recruit legitimately available user context to justify it, and proactively reduce the friction of adoption. We characterize this failure mode through Target Control, Private Binding, and Prospective Support, which respectively steer what the agent advances, how it connects the target to the user, and what target-specific assistance it offers next. Across three proactive-agent environments and six simulated user models, the full attack increases target authorization in all tested environment-user-model combinations, with a macro gain of up to 77.4 percentage points. Controlled replay shows that correct user-target binding is more consequential than additional proposal detail alone, while a multi-turn extension reveals that provider objectives can remain influential even without final authorization by reshaping how the agent responds to user constraints and resistance. These findings expose a broader trust boundary: capabilities designed to serve the user can be redirected toward objectives originating outside the user-agent relationship. 

---
# Task-Aware Joint Pruning and Distillation for Efficient Audio Deepfake Detection 

**Authors**: Miao He, Peng Cheng, Zhongjie Ba, Qing Wen, Li Lu, Xin Yang, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2610.05264)  

**Abstract**: Advances in speech synthesis have made deepfake speeches increasingly convincing, posing growing threats to security. While self-supervised learning (SSL) based detectors achieve state-of-the-art performance, their computational demands (typically 300M+ parameters) prevent deployment on resource-constrained devices. Existing compression methods, designed mainly for content-centric tasks, struggle to maintain competitive performance when directly adapted to deepfake detection. We propose a Task-Aware Joint Pruning and Distillation framework that combines cross-domain knowledge distillation with movement-guided structured pruning to transfer forgery-discriminative knowledge and preserve critical structures under aggressive compression. Our framework reduces the model to 31.9M parameters with 6.3$\times$ FLOPs reduction, with an average performance drop of only 1.30\% across multiple datasets compared to the uncompressed baseline, demonstrating strong potential for on-device deployment. 

---
# MGPO: Manifold-Guided Diffusion Alignment for Task-Aware Dataset Distillation 

**Authors**: Yunyi Chen, Chenru Wang, Xinyi Ye, Zexin Zheng, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05252)  

**Abstract**: Diffusion-based dataset distillation (DD) suffers from a fundamental objective mismatch: likelihood-driven diffusion models prioritize density approximation over the discriminative decision boundaries required for downstream tasks. Beyond semantic mismatch, relying solely on density also leads to geometric coverage loss, where generated samples collapse into a few high-density modes and fail to cover the manifold's structural diversity. We propose Manifold-Guided Policy Optimization (MGPO), which reformulates DD as a multi-objective reinforcement learning problem and achieves Dual-Space Alignment via a pixel-space discriminative reward and a latent-space geometric reward guided by a class-wise Minimum Spanning Tree (MST). The discriminative reward enforces class separability, while the MST-based geometric reward encourages generated latents to cover a sparse geometric skeleton of each class, jointly addressing both failure modes. We further provide an idealized analysis that motivates the MST-based reward, including a Hausdorff approximation bound and a subsampling bound independent of the dataset size. The reward-modular design extends to structured tasks such as object detection and segmentation by substituting the frozen task reward model. Extensive experiments show MGPO consistently outperforms existing methods, including a +8.0% mIoU gain on segmentation under low-budget settings. 

---
# StateWise: Diagnosing and Repairing Persistent Operational State Before Agent Actions 

**Authors**: Yongyuan Peng, Zhou Feng, Tongying Wu, Jiahao Chen, Yuan Su, Chunyi Zhou, Tianyu Du, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2610.05241)  

**Abstract**: LLM agents combine reasoning, tool use, and persistent memory to support work across tasks by reusing stored operational records as premises for later actions. However, environmental or requirement changes can invalidate these records, while existing action review, provenance tracking, and clarification mechanisms may leave the underlying persistent state uncorrected. Our audit of coding-agent trajectories identifies candidate failure chains in which invalid records are reused, leading to task failures and unsafe modifications. We propose StateWise, a framework for diagnosing and repairing persistent operational state before action execution. StateWise uses record-level counterfactual replanning to identify decision-critical records, then establishes their current validity through reliability checks, read-only verification of machine-observable facts, and targeted clarification of developer-owned intent. Typed evidence grounding binds evidence to specific records and scopes, enabling persistent corrections with repair lineage. The agent then replans from the repaired state, followed by an independent state-action check before execution. We evaluate StateWise on 150 executable coding-agent cases across diverse runtime environments, workspace configurations, and repository settings, complemented by cross-model evaluations. Under corrupted persistent state, StateWise achieves 93.3% overall correctness, compared with 38.7% for the baseline agent, with no unsafe actions. Component ablations, multi-task experiments, and transfer evaluations further demonstrate effective recovery, persistent corrections, and transferability across repositories and tool interfaces. 

---
# Pythia: Toward Foundation World Models for Multimodal Time Series 

**Authors**: Xilin Dai, Hongzhou Chen, Yifan Hu, Yiding Liu, Zewei Dong, Jiang-Ming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05240)  

**Abstract**: Time-series foundation models offer a unified approach to forecasting across heterogeneous domains. Textual context and auxiliary observations provide complementary information about temporal dynamics, yet reusable multimodal predictive representations remain underexplored. We introduce Pythia, a foundation world model that learns context-conditioned latent dynamics across datasets through a joint-embedding predictive architecture. A stop-gradient numerical reference guides contextual corrections to predicted future states. A separate probabilistic decoder then adapts to the frozen predictive representation and observed history, decoupling world-model pretraining from observation-space forecasting. On MUSE, Pythia-Tiny's normalized mean absolute scaled error (MASE) and weighted sum quantile loss (WSQL) are 0.6879 and 0.4269, reducing errors by 6.26% and 5.00% relative to the strongest model evaluated in the published MUSE leaderboard. Through a series of controlled experiments, we investigate how to design a time-series world model through shared pretraining and how joint-embedding predictive learning can incorporate multimodal information. The results support separating predictive representation learning from probabilistic readout and show complementary contributions from entity descriptions, events, and covariates. 

---
# Settling the Computational Complexity of Max-Min Allocation with Ternary Valuations 

**Authors**: Thi Ngoc Anh Vu, Trung Thanh Nguyen, Khaled Elbassioni, Jörg Rothe  

**Link**: [PDF](https://arxiv.org/pdf/2610.05237)  

**Abstract**: We study the problem of computing an allocation of indivisible items that maximizes egalitarian welfare, i.e., the utility of the worst-off agent, when agents' item values or marginal values belong to a small set. For additive valuations with values in $\{p,q\}$, where $q>p>0$ and $\gcd(p,q)=1$, we give a polynomial-time algorithm when $p=2$ and prove constant-gap hardness when $p\geq3$, already with exactly three high-valued goods per agent. We also give an $\sqrt{3/2}$-approximation for common positive bi-valued additive valuations. For mixed additive valuations in $\{-p,0,c\}$, where $p\in\{1,2\}$ and $c$ is a positive integer, a reduction to maximum-weight perfect matching resolves the conjectured tractability of $\{-2,0,c\}$-valuations. For submodular valuations with marginals in $\{-2,0,c\}$, where $c$ is odd, we establish an exact unit-gap hardness result and exponential value-query lower bounds, even when all but one agent are additive. Finally, for $\{-1,0,1\}$-submodular valuations, we prove that no finite multiplicative approximation exists unless $\p=\np$. Together, our results resolve open questions and provide a complete picture of the computational complexity of max-min allocation with ternary valuations. 

---
# EMG-GPT: Predictive Pretraining on Residual-Quantized EMG Tokens for Hand Pose Estimation 

**Authors**: Ettore Magni, Rolandos Alexandros Potamias, Stefanos Zafeiriou, Konstantinos Barmpas  

**Link**: [PDF](https://arxiv.org/pdf/2610.05235)  

**Abstract**: Surface electromyography (sEMG) is a low-power, cost-effective biosignal for hand-pose estimation and gesture classification. In this work, we examine whether self-supervised pretraining on sEMG can yield transferable representations for continuous hand-pose estimation. We introduce EMG-GPT, a causal transformer-based model that operates on discrete sEMG representations from a frozen residual vector quantization (RVQ) tokenizer and learns temporal dynamics through depth-autoregressive future-code prediction. The model combines within-frame integration with causal temporal modeling while preserving the geometry of the pretrained codebook. EMG-GPT shows competitive results in both Regression and Tracking tasks, supporting EMG-only pretraining as a viable approach for learning transferable sEMG representations. 

---
# Vela: Scaling Vision-Language-Action Models with Adaptive Action Curve Parametrization 

**Authors**: Yifan Li, Jiaxu Wang, Dongming Wu, Yicheng Jiang, Ryan Ji, Xiangyu Yue, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05230)  

**Abstract**: Most vision-language-action models represent future motion as fixed-rate action chunks, tying temporal resolution and prediction horizon to a fixed output budget. This pointwise representation wastes capacity on highly correlated neighboring actions, leaves temporal continuity and smoothness to be learned implicitly, and forces a tradeoff between long-horizon coverage and the local precision required for contact-rich manipulation. To address these limitations, we introduce Vela, a vision-language-action foundation model that represents future robot behavior as continuous trajectories. Vela combines a compact spline-based action representation with motion-dependent temporal support and a shared action interface for heterogeneous embodiments, allowing a fixed output budget to adapt its temporal resolution across motions. We pretrain Vela on large-scale multi-embodiment robot data and evaluate it on LIBERO-X, EBench, and two real-world long-horizon tasks, egg-cake cooking and potato shredding, obtaining promising results across simulation and physical manipulation. These results highlight the potential of continuous action representations as a foundation for future embodied foundation models. Project page and more results: this https URL . 

---
# Same Predictions, Different Harms: Causal Auditing of Patient World Models 

**Authors**: Yicheng Qi, Xiyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.05198)  

**Abstract**: Patient world models used for clinical trial simulation can agree on transition kernels and arm-specific risks, yet disagree on the fraction of patients harmed by switching treatment---the counterfactual quantity that matters for intervention-aware reasoning. We audit this reliability gap in a two-stage shared-response SCM: a categorical intermediate health state is followed by common terminal care. Under independent stages, the sharp harm interval has closed-form endpoints for at most three intermediate states, with an exactness boundary at four states. Declared dependence and response-mismatch budgets yield calibrated outer bounds when stage independence or complete mediation is relaxed; in a symmetric three-state model the entire sensitivity frontier is sharp, $[0,\min\{1/2,1/3+(\rho+\delta)/2\}]$, and shows exactly how budgets erase the gain over endpoint-only bounds. Two eight-variable response LPs propagate interventional uncertainty for finite-sample audits. Exact witnesses verify attainability. On public clinical simulators (EpiCare; sepsis), native configurations show little resolved stage dependence and no additional joint-compatibility gain over pairwise transport---honest negative results for reliability claims. All experiments are locally reproducible; guarantees remain conditional on the stated causal model. The results provide a concrete protocol for deciding when a patient world model is safe to trust for counterfactual harm. 

---
# Blocking at the Boundary: Auditing Long-Horizon Agents against Staged Prompt Injection 

**Authors**: Jingkai Liu, Yufei Han, Xiaoting Lyu, Wei Wang, Ting Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05163)  

**Abstract**: Long-horizon agents consume external content, invoke tools, and modify persistent state. Indirect prompt injection can exploit task-specific context, propagate across causally connected stages, and alter a consequential action while the workflow continues; we term this staged prompt injection.
We build an automated, feedback-guided attack generation pipeline and apply it to Claude Code and Codex in their native runtimes. The confirmed attacks span eight workflow scenarios, seven attack goals, and six injection surfaces, showing that production agents are vulnerable to context-aware, multi-step injection over long horizons. Stopping such attacks requires a decision before each consequential action: input screening and completed-run evaluation cannot locate the intervention point, and existing pre-action methods use incompatible units and labels. We therefore formulate boundary action auditing: given initial context, a trajectory prefix, and a fully specified pending message or tool call, an auditor predicts Pass or Block before its effect occurs. Pairing attacked and benign executions yields a 479-pair, 3,112-unit benchmark.
We further propose Path-Aligned Attribution (PAA), a training-free auditor that decomposes pending actions into operative elements and traces what supplied each value and guided each decision. PAA blocks only when the model attributes an unwarranted, material effect on an element to an attacker-reachable source that either provides unqualified steering or conflicts with visible evidence. Under full-benchmark fail-open scoring with Claude Sonnet 5, PAA reaches 86% Block recall at a 6-8% false-block rate (FBR), whereas ARGUS reaches 44-47% recall at 16-33% FBR. Under the same backend, on the tool calls that all three auditors natively support, PAA has higher recall and lower FBR than VIGIL and ARGUS; all paired 95% confidence intervals exclude zero. 

---
# Cross-Modal Contrastive Learning for the Retrieval of Immunotherapy-Associated Molecular Signatures from Histopathology 

**Authors**: Sigrid Vila-Bagaria, Mar Teixidó, Miquel Piñol, Felip Vilardell, Robert Montal, Veronica Vilaplana  

**Link**: [PDF](https://arxiv.org/pdf/2610.05157)  

**Abstract**: Gastric Adenocarcinoma is a leading cause of cancer mortality. Although "Inflamed/Non-Inflamed" subtypes have been proposed to predict immunotherapy response, their identification relies on a costly 10-gene RNA signature. We propose a Cross-modal Contrastive Multiple Instance Learning (CCMIL) framework for cross-modal retrieval, imputing these molecular signatures directly from standard Hematoxylin & Eosin (H&E) slides. By leveraging a supervised contrastive objective, CCMIL aligns visual morphological patterns with molecular phenotypes into a shared latent space. This establishes an interpretable search-by-case retrieval engine, enabling pathologists to query a whole slide image to surface transcriptomically coherent neighbors and approximate RNA signatures without genomic sequencing at inference. Our results demonstrate that this retrieval-first approach captures the continuous phenotypic spectrum of tumor inflammation and yields clinically interpretable attention heatmaps. Furthermore, the learned representation also supports competitive downstream classification, providing a practical molecular pre-screening strategy. 

---
# How Does Geometry Enter Generated Motion? 

**Authors**: Weihan Li, Junhao Wu, Yuhan Song, Xiaofeng Lin, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.05135)  

**Abstract**: Under a fixed physical law, the visible geometry of a scene determines how motion must change. We ask how video generators realize this relationship. We fix the law and the initial state and change only the geometry drawn in the first frame, within matched families of tracks and deflectors, and compare each generated trajectory with the simulator prediction for that geometry. Paired interventions change one thing at a time: a local bump, the height of a barrier, the words of the prompt, the length of the clip. Across nine image-to-video models, geometry is preserved and shapes the motion: the speed of the ball follows the drawn undulation of a track. A physical state would carry this response forward, and here the generated motion parts from the law. The mean slope barely accelerates the ball, successive contacts fail to compose through a consistent state, an edit ahead of the ball alters its motion before it arrives, and the ball climbs over barriers higher than its release point. Two global conditions organize the global trajectory: text strongly controls the destination, while clip length strongly controls timing in the open-weight models tested. The pattern persists with photographed first frames. Current video generation thus behaves as geometry-conditioned motion synthesis whose evolution of state differs systematically from that of a fixed physical law. 

---
# Bayesian Entropy-based Reordering for Calibrated Diffusion Language Models 

**Authors**: Zhejun Jiang, Mijung Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.05125)  

**Abstract**: Masked Diffusion Language Models (MDLMs) generate sequences by iteratively replacing masked tokens with model predictions. At each denoising step, the decoder chooses which positions are sufficiently confident to commit. Existing decoding methods typically rely on softmax confidence, which can be miscalibrated. We introduce BayesER (BAYESian Entropy-based Reordering), a post-hoc Bayesian decoding framework that uses predictive uncertainty to guide token commitment. In BayesER, we construct a lightweight approximate posterior centered at the pretrained checkpoint, similar to Laplace-LoRA but without training LoRA adapters. We average predictions over posterior samples and use predictive entropy to prioritize reliable positions. We examine how posterior predictions affect position ordering and token selection across benchmarks spanning code generation, mathematical reasoning, planning, and molecular generation. We show that BayesER reduces sequence-level calibration error while preserving or improving accuracy relative to common decoding schemes, including confidence-threshold decoding. Additionally, a posterior fitted on one code-generation dataset reduces calibration error on another without refitting, suggesting that Bayesian uncertainty may provide a transferable signal for more reliable MDLM decoding. 

---
# Best-of-$N$ Guidance for Test-time Diffusion Alignment 

**Authors**: Richard Lee Kim, Yeongmin Kim, Gyuwon Sim, Taekyu Kim, Minsang Park, Il-chul Moon  

**Link**: [PDF](https://arxiv.org/pdf/2610.05108)  

**Abstract**: Diffusion models achieve strong generative performance but often struggle to align generated samples with human preferences measured by a reward model. A simple yet effective algorithm for test-time alignment is Best-of-$N$ (BoN) sampling, which draws $N$ i.i.d. samples from a pre-trained diffusion model and outputs the single highest-reward sample. Despite its empirical success, BoN makes limited use of reward information, as it is incorporated only at the final selection stage without influencing the reverse diffusion trajectory during sampling. Consequently, BoN sampling does not improve the average alignment of generated samples and is primarily suited to single-output settings. We propose Best-of-$N$ Guidance (BoNG), a novel method that integrates the principle of BoN sampling directly into the reverse diffusion process. BoNG performs online BoN selection over denoising particles and adjusts the reverse diffusion process to steer the particle population toward higher-reward regions during generation. Specifically, by introducing an asymmetric guidance interaction among denoising particles, BoNG uses the current BoN particle as a guidance signal to the rest of the particle population. This particle-level interaction reshapes the sampling process toward higher-reward regions, enabling BoNG to improve not only the final best sample beyond Vanilla BoN sampling, but also the average quality of generated samples. Over 36 empirical comparisons, BoNG achieves the best performance in 29 cases, ranking first in 80.56% of the comparisons against SMC and Vanilla BoN sampling. BoNG also supports multi-output capability, achieving 1.3$\times$ ImageReward score of the latest sample-based guidance method with a 1.6$\times$ speedup. We release the code at this https URL. 

---
# VulValidate: Auditing Function-Level Vulnerability Labels with Executable Evidence 

**Authors**: Leizhen Zhang, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.05103)  

**Abstract**: Reliable learning-based vulnerability detection requires high-quality labels, yet datasets built from vulnerability-fixing commits may label functions as vulnerable simply because they were changed by a security patch. We present VulValidate, a framework that uses LLM agents to coordinate dynamic analysis tools and construct vulnerability-triggering experiments from runtime feedback. Given a labeled function and its fixing patch, VulValidate reconstructs vulnerable and fixed revisions, selects suitable tools and execution paths, refines triggering inputs, and compares runtime behavior to assess function-level attribution. We audit all 35,849 instances originally labeled vulnerable in BigVul, PrimeVul, and DiverseVul. We confirm 20,510 (57.2%), correct 6,819 labels (19.0%), leave 7,981 attacked but undecided (22.3%), and cannot successfully measure 539 (1.5%). After conflict resolution and byte-exact deduplication, the corrected release contains 15,890 distinct confirmed vulnerable function bodies. In a blinded review of 581 sampled decisions, expert consensus supports 90.0%--92.0% of confirmations and 92.6%--99.0% of label corrections. With model parameters fixed, corrected evaluation lowers F1 for all five tested detectors on both BigVul and DiverseVul; retraining with corrected labels improves F1 for four of five detectors on each dataset. We also release a reusable VulValidate skill, corrected datasets, and reproducible evidence for future vulnerability-detection research. 

---
# ReMAP: Restoring the Perceptual Cycle with Reasoning-Time Latent Visual Memory 

**Authors**: Hao Jiang, Zhanyu Guo, Chenwei Wu, Yichen Guo, Qizhe Zhang, Junchi Yao, Jixian Wu, Jinhao You, Kai Tang, Jiajun Cao, Tinghao Wang, Mengyu Wang, Leo Anthony Celi, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05097)  

**Abstract**: As multimodal large language models (MLLMs) reason for longer, attention to the initial visual input diminishes, weakening visual grounding. Visual memory reintroduces visual evidence during reasoning. We conduct a controlled analysis of visual memory along three axes: curation, organization, and access. We find that local evidence benefits from global context, compact latent representations balance accuracy and visual-context cost, and the utility of memory access depends on the reasoning state. Guided by these findings, we propose ReMAP (Reasoning-Time Memory-Augmented Perception), which couples two complementary latent memories: a static, question-conditioned Global memory that preserves scene and cross-image context, and a dynamic Local memory that uses this context as an anchor while selecting and re-encoding region-level evidence according to the current reasoning state. Both memories return compact latent tokens inserted into the reasoning sequence, and a reinforcement-learning access policy trained with branched rollouts decides when to continue reasoning or invoke Global or Local memory. On ten benchmark families, ReMAP outperforms prior visual-memory methods on all four multi-image benchmarks, exceeding the strongest prior results on MuirBench and MIMIC by 8.38 and 14.84 percentage points. Across four backbone families, enabling memory access improves over the same trained model with memory disabled, and on shared V*Bench, CV-Bench-2D, and MuirBench questions ReMAP reduces the visual tokens entering the reasoning sequence by 51.0-76.8% relative to the native-resolution backbone. Further analyses show that Global and Local memory form distinct yet complementary latent representations. Together, these components restore the perceptual cycle by letting the reasoning state trigger targeted visual retrieval, with the retrieved evidence guiding subsequent reasoning. 

---
# Direction-Conditioned Policies for Online Goal-Conditioned Reinforcement Learning 

**Authors**: S K Swaminathan, Damiya Gondha, Theyanesh Eswaramoorthy Rajahkrishnan, Aritra Hazra  

**Link**: [PDF](https://arxiv.org/pdf/2610.05087)  

**Abstract**: Contrastive Reinforcement Learning (CRL) learns representations that estimate goal reachability, yet its policy remains conditioned on raw goals and therefore does not directly exploit the geometry encoded by its critic. We introduce Direction-Conditioned Policies (DCP), a method built around a small modification to CRL: DCP selects previously visited states as waypoints during online training and conditions the policy on their direction and distance in representation space. At deployment, DCP applies the same interface directly to the final goal, requiring neither waypoint selection nor planning. Across nine navigation and manipulation tasks, DCP attains higher final success rates than CRL on seven tasks and spends more time near the goal on seven. Controlled maze experiments further show that DCP captures shortest-path geometry more accurately and that the supplied direction causally influences the actor's behavior. We identify waypoint coverage and ranking as limits to exploration, and show that learned candidate generation improves goal reaching in two controlled mazes. 

---
# Tracing a Sparse Emotion-Control Circuit in LLM-Based Text-to-Speech 

**Authors**: Hongfei Du, Jiacheng Shi, Yanfu Zhang, Ye Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05080)  

**Abstract**: LLM-based text-to-speech (TTS) models can generate emotionally expressive speech, but how reference emotion is routed through the model and realized in decoded speech remains unclear. We introduce two emotion-sensitive metrics for matched neutral and emotional syntheses---a codec trajectory score and a late residual direction score---and use them to score activation-patching interventions. Under controlled matched-reference conditions, this analysis identifies a sparse source-to-readout component-level circuit: 23--27 attention heads and MLPs per emotion, roughly 5% of the components considered, recover or suppress 74--88% of the late emotion-readout shift on held-out cases. The circuit combines a shared component backbone with emotion-specific components; cross-emotion activation swaps reduce the target readout in 47 of 48 cases. In decoded speech, the same intervention produces consistent changes in pitch, energy, and spectral brightness over 24 matched pairs per emotion. A readout-matched residual-direction baseline produces only 17--27% of the intervention's pitch effect, showing that internal readout movement alone does not explain the decoded acoustic changes. These results trace a compact causal route from reference-derived prefix information to emotion-relevant properties of generated speech. 

---
# Self-Generated Feedback Destabilizes Test-Time Training: A Causal Decomposition of Long-Horizon Adaptation 

**Authors**: Cheng Luo, Bing Li, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2610.05076)  

**Abstract**: Test-time training (TTT) lets a model store information in its weights during inference. When the model learns from its own output, however, each update also changes the model that generates the next training example. Across 128K-token streams, retaining generated-text updates worsens prediction on independent human-written text with three TTT-E2E model configurations (labeled 125M, 760M, and 3B). The same failure occurs when Adam updates Qwen3-4B's existing weights. The same update mechanisms can improve on real text, so writing itself is not the failure. Three matched comparisons trace the causal pathway. Fixed Generation removes over 98% of the damage at 125M and 760M by using a frozen model to generate training chunks. Recorded Replay separates the loss caused by reading degraded text from the additional loss stored by updating on it. A paired one-update comparison then shows the local conflict: an update predicts its source better but new real text worse. This cost grows after Closed Loop adaptation, with a few trajectories accounting for most large failures. Finally, Settlement evaluates the candidate state on independent real text before commitment. It leaves mean endpoint gaps of 0.07 and -0.02 nats at 125M and 760M while retaining real-text adaptation. These results motivate checking prediction on independent evidence before retaining an update. 

---
# Discrete Action Matching: Learning Stochastic Dynamics from Samples via State Graphs 

**Authors**: Mikhail Persiianov, Alexander Korotin  

**Link**: [PDF](https://arxiv.org/pdf/2610.05071)  

**Abstract**: Learning population dynamics from unpaired temporal marginals is an ill-posed inverse problem that requires structural assumptions on the underlying dynamics. We introduce $\textit{Discrete Action Matching}$ (DAM), a finite-state counterpart of Action Matching based on discrete Wasserstein geometry. For a prescribed marginal path and transport geometry, we derive an action-minimization objective for its canonical minimum-kinetic-energy current. Our key observation is that the density dependence of the discrete action reduces to neighboring density ratios. Along an empirical interpolation of the snapshots, DAM first estimates these ratios and then learns an action potential. The learned fields also define a graph-supported Markov sampler. Experiments on controlled synthetic dynamics and real mouse gastrulation data evaluate marginal reconstruction and interpolation. Additional experiments approximate numerical surface-transport paths from paired samples. 

---
# LogSig-SSM: Time-Series Modelling with Multi-Scale Log-Signature Compression for State-Space Models 

**Authors**: Felix Oury, Nicolas Calvo Peiro, Reiko J. Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2610.05051)  

**Abstract**: Time-series data are often sampled irregularly at high frequencies and exhibit long-range dependencies, which makes long-horizon modelling difficult. Continuous-time models such as neural controlled differential equations (NCDEs) and neural rough differential equations (NRDEs) can handle irregular sampling, but they scale poorly to long sequences. Selective state-space models (SSMs) such as Mamba scale linearly with sequence length, but they provide limited recurrent mixing across hidden dimensions within a single block. We propose LogSig-SSM (Log-Signature Compression for State-Space Models), which first compresses long multivariate time series into a shorter sequence of tokens using multi-scale windowed log-signatures, and then processes these tokens with a selective SSM backbone. LogSig-SSM is scalable and robust to irregular sampling, combining log-signature tokens that capture higher-order cross-channel interactions with a selective SSM that models long-range dependencies. The model also admits a continuous-time interpretation as an NCDE/NRDE-style system driven by a log-signature-based input, in which selectivity induces an input-dependent rescaling of the latent dynamics. Across four benchmarks, namely long-sequence classification on UEA, high-frequency physiological regression on PPG-DaLiA, multivariate weather forecasting, and irregularly sampled clinical prediction on PhysioNet Sepsis, LogSig-SSM outperforms or matches strong SSM and continuous-time baselines while training up to $30\times$ faster and using up to $37\times$ less GPU memory than Mamba on the longest sequences. 

---
# E$^2$-OPSD: Taming Entropy Overshoot in On-Policy Self-Distillation 

**Authors**: Yifei Liu, Minghao Fang, Xinyu Gu, Chengkai Yao, Mengdi Liu, Tengfei Ma, Jiangbin Zheng, Chang Yu, Zhangyang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05048)  

**Abstract**: On-policy self-distillation (OPSD) provides dense token-level supervision without a second model: one network acts as teacher with the reference solution and as student with only the problem. We identify a specific failure mode of this recipe. During training, student token entropy rises past the teacher's and remains elevated, a pattern we call entropy overshoot. We trace it to both sides of distillation. The reference-conditioned teacher is confident along its answer-directed reasoning path, but this confidence transfers poorly to student-generated prefixes, making its supervision overly tied to answer-specific cues rather than reusable reasoning patterns; meanwhile, the forward KL used by OPSD continually diffuses the student's predictive distribution without pulling it back. We introduce E$^2$-OPSD to address both causes. Exemplar-guided teaching replaces the current answer with a retrieved solved neighboring problem, providing transferable reasoning guidance without revealing the destination and better matching student-reachable states. Entropy-aware distillation uses the student-teacher entropy gap to determine the direction and strength of each token's correction. E$^2$-OPSD improves math reasoning by up to 4.3 points in mean@16 over OPSD, while out-of-domain evaluations show gains over the corresponding base models of up to 4.9 points in mean@16 and 5.5 points in pass@8. Despite these gains, E$^2$-OPSD remains simple, requiring no additional forward passes or networks. 

---
# AraYoungVoices: A Diverse L1/L2 Corpus of Arabic Child and Adolescent Speech 

**Authors**: Shammur Absar Chowdhury, Zien Sheikh Ali, Houssam Eddine-Othman Lachemat, Hamdy Mubarak  

**Link**: [PDF](https://arxiv.org/pdf/2610.05044)  

**Abstract**: State-of-the-art ASR systems primarily target native adult speech, leading to substantial performance gaps for children, adolescents, and L2 speakers. We introduce AraYoungVoices, a 151.72-hour Arabic read-speech corpus from 286 speakers aged 7--18, comprising AraKids (7--12) and AraTeens (13--18). The corpus includes 146 native Arabic (L1) and 140 second-language (L2) speakers, with native speakers spanning Egyptian, Gulf, Levantine, and North African dialectal backgrounds and L2 speakers representing diverse linguistic backgrounds across the Americas, Asia, Africa, and Europe. We benchmark four pretrained ASR models under zero-shot and fine-tuned settings using unseen-speaker-$\&$-unseen-prompt (USUP) and unseen-speaker-$\&$-seen-prompt (USSP) evaluations. Results show that L2 speech remains substantially more challenging than L1 speech, with the largest errors observed mainly for younger L2 speakers. Age-specific fine-tuning improves the matched age group, while joint fine-tuning provides a stronger balance across populations. ASR hypotheses are also consistently closer to the standard reading prompt than to the verbatim transcription, particularly for L2 speech, suggesting partial normalization of reading deviations. 

---
# Communication Shapes Collective Inference in Self-Adapting LLM Societies: Evidence from Mafia 

**Authors**: Haonan Huang, Joey Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05041)  

**Abstract**: When does communication help a group identify hidden adversaries, and how does its value change as the group adapts? In Mafia, an informed minority hides inside an uninformed majority whose only evidence is open play. The zero-information game, where each day's vote eliminates a random player, is exactly solved and scores every society; matched-casting comparisons between protocols identify the effect of communication. Societies of 8-100 claude-haiku-4-5 agents (7,416 analyzed games, 1.9M model calls) adapt by rewriting and inheriting private strategy notes. Simultaneous broadcast improves adversary identification over silence in all nine compositions tested (8-46 players). Turn-taking removes most of this advantage; its voting landslides are as frequent as broadcast's but land on mafia near chance (1.08x versus 2.53x). At 70 players, agents reading eight statements per day identify adversaries worse than silent ones, and limited talk is worth less than at 46 players. Adaptation is fast but need not help. In their first broadcast games, citizens announce their role far more often than mafia (91% vs. 30%) and first-day votes find mafia at three times chance; within two generations citizens stop announcing and the cue fades, a change the inherited notes carry. In controlled redeployments at 16 players, societies carrying sixty generations of their own notes score below societies with none. Communication shapes both collective inference and the signals it depends on, so a protocol's value must be measured together with the adaptation that changes those signals. 

---
# SparseCraft: Agentic Hardware-Software Co-Optimization for Sparse Computing 

**Authors**: Rajatabha Chakraborty, M P Samartha, Vedant Pahariya, Priyesh Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2610.05037)  

**Abstract**: Sparse-accelerator design spaces are usually searched against analytical models, so a design point is admitted on what a model predicts rather than on what the hardware does. SparseCraft closes that gap with a language model inside a closed CHIA loop. In each of 15 iterations the model reads the measured outcome of the previous one and edits the Chisel RTL, the memory configuration and the sparse-kernel schedule of a Gemmini accelerator through MCP tool servers, and no candidate counts until it has been checked for legality, elaborated, simulated cycle-accurately, checked bit-for-bit on every output against a golden reference, and synthesised. The harness turns each measurement into the next work order, a diagnosed bottleneck with matching strategy guidance, the history of tried designs and a score of the model's own prediction, and a second model repairs changes that fail a gate. On a $512 \times 512$ GraphChallenge sparse-DNN layer the loop reaches 2.1x fewer cycles, 9.8x less off-chip traffic and 22.8% less area than the block-sparse Gemmini baseline, with 5.61x higher modelled perf/W and 11.8x lower EDP. The levers span three layers: a schedule that keeps the dense operand resident removes 9.8x of the traffic, a zero-gated MAC and a zero-row skip unit that the model wrote in Chisel cut energy, and resizing the memories cuts area. 

---
# Operational Abstractions of Neural Network Concepts via Topological Representations 

**Authors**: Mathieu Pont, Christoph Garth  

**Link**: [PDF](https://arxiv.org/pdf/2610.05017)  

**Abstract**: Concept-based methods provide a semantic level for interpreting and manipulating learned representations, but existing editing approaches are typically specialized to particular interventions and do not provide a common and editable representation of concept organization. To achieve this, we introduce Topological Concept Representations (TCR), a post-hoc operational abstraction that jointly characterizes the concepts encoded in a learned representation and their relationships. TCR constructs an intermediate concept space from concept recoverability and interaction scores, and compactly encodes its organization through topology. Interventions are expressed through modifications of this abstraction that are propagated back to the underlying learned representation. This allows different concept-level operations to share the same optimization framework and separates the desired concept organization from the mechanism to achieve it. We establish stability and reparameterization-invariance properties of TCR and its connections to existing concept-editing formulations. We use TCR to disentangle concepts as a preprocessing step for existing erasure methods, improving worst-group accuracy by 21.89 on average at comparable concept leakage. We further use TCR to transfer concepts from teacher to student models, improving concept recoverability by up to 5.54 while also improving or maintaining competitive test top-1 accuracy. 

---
# Hidden Risks of Jev: An Empirical Study of Security, Privacy, and Dual Use 

**Authors**: Shang Wang, Tianqing Zhu, Huajie Chen, Jiayang Li, Meng Yang, Bo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04985)  

**Abstract**: Jev turns natural-language questions into typed answers and probabilities with low latency and cost, enabling applications to route requests and select tools. While this interface allows Jev to integrate naturally into application workflows as a decision layer, the security and privacy implications of this emerging use remain largely unexplored. To address this gap, we conduct the first systematic study of these implications using the official Jev API and NanoJev, a local model with controllable training data and updates, focusing on three research questions: (1) What security threats arise when Jev is deployed as an application decision layer? (2) What private information can Jev reveal despite returning constrained typed outputs? (3) How can Jev's general-purpose decision capability be used for beneficial purposes or misused?
Jev's decisions depend on application state and may be influenced by user-provided inputs. We therefore adapt prompt injection and adversarial suffixes to manipulate its decisions. Open-source Jev distribution and updates introduce supply-chain risks, which we examine by implanting backdoors in NanoJev through training data poisoning. Since Jev's outputs reflect both application state and information learned during training, we further adapt membership, private attribute, and internal knowledge inference attacks to recover sensitive information despite its constrained output format. Finally, Jev can serve as a general-purpose decision oracle for defensive and malicious workflows. We examine this dual use through four detection tasks covering prompt injection, jailbreak inputs, harmful content, and AI-generated text, alongside misuse scenarios involving jailbreak and model extraction. Our empirical evaluation shows that Jev remains vulnerable to the examined security and privacy threats, while its decision capability can support beneficial and malicious uses. 

---
# TeleGen: Improving LLM-Based Web Application Generation via Runtime Telemetry 

**Authors**: Yujia Luo, Haonan Zhang, Jiasi Shen, Zishuo Ding, Weiyi Shang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04981)  

**Abstract**: Large language models can generate runnable web applications from natural-language requirements, but many generated applications still fail interactive tasks. Existing generate-execute-repair pipelines execute the generated application and use task outcomes or error messages to guide code revision. However, this feedback often misses the runtime behavior between a browser action and the final task outcome, making interaction-level failures difficult to diagnose. Therefore, we propose TeleGen, an observability-enhanced framework for LLM-based web application generation. TeleGen instruments generated applications, collects runtime telemetry during task execution, and compresses raw telemetry logs into concise briefs for repair. We evaluate TeleGen on WebGen-Bench and Web-Bench. On WebGen-Bench, TeleGen improves task success from 67.7% with repair without telemetry to 76.2%, an increase of 8.5 percentage points. On Web-Bench, it improves cumulative Pass@2 from 21.7% to 29.8%. Ablation results show that runtime telemetry provides a useful diagnostic signal, while telemetry briefs make this signal more effective and less costly to use. Further analysis shows that telemetry is especially helpful for failures involving hidden execution paths, such as navigation, form workflows, and frontend-backend coordination. 

---
# Kapture: Capturing Cardiac Dynamics with Koopman-Governed Learning for Efficient Radar-Based Electrocardiogram Recovery 

**Authors**: Tong Wu, Jing Peng, Ziqi Feng, Yuanyuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04955)  

**Abstract**: Millimeter-wave (mmWave) radar enables unobtrusive, contactless electrocardiogram (ECG) reconstruction for cardiac monitoring. Time-frequency spectrograms preserve fine cardiac patterns but often require large backbones to separate ECG-relevant features from respiration, motion, multipath, and subject-dependent interference. We propose Kapture, a parameter-efficient Koopman-governed framework that projects radar hidden states into a low-dimensional observable space and identifies a regularized full linear evolution operator from adjacent observable states. The Koopman-predicted observables refine subsequent hidden states for ECG reconstruction. To suppress predictable interference dynamics, a temporal contrastive objective pulls neighboring states together and separates non-neighbors, while reconstruction supervision preserves task relevance. Using approximately 80 minutes of quasi-static radar-ECG recordings containing realistic noise from body movements and other sources, Kapture consistently improves reconstruction across matched backbone widths, with the largest gains under aggressive compression. The compact configuration approaches the full-width reference accuracy with 68.1% fewer parameters and 91.3% fewer profiler-covered floating-point operations (FLOPs), while the full-width configuration delivers the strongest overall reconstruction performance. Our code will be made publicly available after potential publication. 

---
# EDISCO: Equivariant DIScrete Diffusion for Euclidean Combinatorial Optimization 

**Authors**: Ruogu Chen, Jie Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.04953)  

**Abstract**: Euclidean combinatorial optimization problems (ECOPs), such as the Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP), possess inherent symmetries under the two-dimensional Euclidean group E(2), including rotations, reflections, and translations. Existing learning-based methods, including recent diffusion-based methods, rely on data augmentation or regularization to approximate E(2)-equivariance. This paper presents EDISCO, the first discrete diffusion model for ECOPs with exact E(2)-invariant generative distributions over node-index solutions. EDISCO introduces an E(2)-equivariant edge-score network coupled with a categorical continuous-time Markov chain over discrete edge variables, and exact posterior sampling provides efficient multi-step inference. This design gives EDISCO a local geometric inductive bias: edge neighborhoods with the same relative geometry and combinatorial context are represented consistently regardless of absolute position or orientation, making learning more efficient and inference more robust than non-equivariant methods. EDISCO outperforms previous learning-based state-of-the-art solvers on synthetic TSP from 100 to 10000 nodes and CVRP from 50 to 2000 customers, while using only 33-50% of the training instances. Trained only on uniform synthetic data, EDISCO also outperforms competing learning-based baselines under spatial distribution shift and CVRP constraint-tightness shift. Code is available at this https URL. 

---
# A Unified Dynamics Framework for Reinforcement Learning and Classical Control of a Six-DOF Pipeline-Tracking ROV in NVIDIA Isaac Sim 

**Authors**: Cheng Siong Chin, M. Venkateshkumar, Jianhua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04949)  

**Abstract**: Reinforcement learning controllers for underwater vehicles are usually trained against one physics representation and deployed against another, so reported performance does not always describe behavior outside training. This paper presents a pipeline-tracking architecture for a six-degree-of-freedom remotely operated vehicle (ROV) in which one Universal Scene Description (USD) scene supplies the real BlueROV2-Heavy mass, added-mass, damping, buoyancy, and thruster parameters to both halves of the system: a vectorized NumPy implementation of Fossen's marine-craft equations, and an interactive NVIDIA Isaac Sim deployment applying the identical equations as PhysX forces at every step. The Coriolis-centripetal term is the primary dynamics model in both branches; a controlled ablation on PPO and TRPO shows that including it does not destabilize either algorithm and modestly improves tracking, about 19 percent lower standoff RMS error for TRPO. Five reinforcement learning algorithms, PPO, soft actor-critic, TD3, DDPG, and TRPO, are trained against one environment, reward, and randomized evaluation harness through a checkpoint-compatibility layer scoring any policy with the same code. The pipeline is extended with six classical baselines, PID, sliding-mode, fuzzy logic, feedback linearization, model predictive control, and an adaptive neuro-fuzzy inference system, driven by the same guidance geometry and thruster allocation as the learned policies. Under Coriolis-enabled dynamics, PPO, TRPO, and feedback linearization reach the strongest combination of 100 percent success and competitive accuracy; PID, fuzzy control, and the neuro-fuzzy baseline also reach 100 percent success with looser tracking; DDPG and TD3 each show a specific, explainable failure mode rather than a general weakness of off-policy learning; and classical control remains a strong baseline against the best learned policies. 

---
# TempoBridge: Source-Conditioned Flow Matching with Optimal Transport Couplings for Single-Cell Population Transitions 

**Authors**: Bowen Han, Lingbei Meng, Shihuan Luo, Yupeng Zang, Wenlin LI, Peize He, Yaodi Luo, Lian Zhang, Jianqing Zhu, Jinchao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04945)  

**Abstract**: Destructive single-cell measurements provide unpaired population snapshots rather than observations of the same cells across conditions. Local cell states and transition requests may also be insufficient to distinguish responses across source populations. We introduce TempoBridge, a common source-conditioned transport formulation for temporal, genetic, and chemical population transitions. Source cells initialize latent transport and provide a fixed empirical population summary. The velocity field receives this summary alongside the evolving cell state, flow time, and a structured transition descriptor. Minibatch optimal transport (OT) supplies couplings only for conditional flow-matching training paths; inference requires neither target expression nor OT computation. On held-out donors, TempoBridge achieves an Energy distance of 0.129 versus 0.144 for scGen. Genetic mean-expression $L_2$ error is 2.261 versus 3.156 for scGPT-scratch under Seen 2/2. On held-out compounds, condition-averaged drug-effect correlation is 0.598 versus 0.561 for the CellFlow adapter. Temporal ablations show higher mean distributional error after removing source context, replacing optimal transport with random pairing, or replacing flow matching with static residual regression. Together, these results demonstrate the predictive utility of a common source-conditioned transport formulation across held-out donors, gene combinations, and compounds. 

---
# Software World Models: From Consequence Prediction to Decision Value 

**Authors**: Tongli Su, Yuntong Hu, Liang Zhao, Bowen Zhu, JayaSai Somasundaram, Hasibul Haque  

**Link**: [PDF](https://arxiv.org/pdf/2610.04940)  

**Abstract**: A coding agent may safely modify one repository while silently breaking downstream services, libraries, or datastores that depend on it. Exhaustively running integration tests after every agent action is impractical, so the agent must predict these failures before executing them. Existing software world models predict the agent's own observations, while static change-impact analysis only identifies where a change may propagate. We instead introduce the Software World Model (SWM), which models the broader system affected by a code change and predicts its blast set: the components that the change will break. SWM follows three stages: explore, learn, and act. Explore executes candidate changes from restored system states, prioritizing regions where observed failures contradict the dependency graph. Learn fine-tunes a language model on these execution outcomes to predict downstream breakage. Act converts sampled predictions into per-consumer break probabilities for change ranking, proactive migration, and deciding when another execution is worth its cost. On held-out synthetic systems, SWM improves blast-set F1 from 0.431 for static reachability to $0.571\pm0.037$, more than halves ranking regret, and improves migration return at all nine evaluation checkpoints. The two methods are complementary: reachability is stronger on dependencies represented in the graph, while SWM recovers failures caused by couplings the graph misses. Experiments on held-out real libraries further show that predicting structured failure outcomes, rather than only scalar risk, is important for downstream decision quality. 

---
# A Graph-Based Inspection and Intervention Tool for Assessing Mechanistic Learning in PINNs 

**Authors**: Adwait Patkhedkar, Alifaraz Lakhani, Prathmesh Mohite, Abhijeet Salunke  

**Link**: [PDF](https://arxiv.org/pdf/2610.04939)  

**Abstract**: We ask whether physically meaningful correspondences discovered inside a trained scientific model remain meaningful outside the conditions under which they were discovered. We introduce GIIT (Graph-based Inspection and Intervention Tool), which represents governing physics as a computational physics dependency graph, maps graph nodes to internal network components via sensitivity- and trend-based discovery, and tests the resulting mapping under targeted intervention. Evaluating on temporal extrapolation out-of-distribution (OOD) regimes of increasing severity, we find that temporal extrapolation is associated with layer-wise correspondence drift and reduced functional correspondence. Specifically, while the model achieves low physics residual in-distribution (1.915 x 10-4 on full ID and 1.01 x 10-4 on an ID sub-window t in [0.5, 0.8]), the functional mapping changes even before leaving the training domain, and the shift continues as the evaluation window extends beyond the training domain: residual error increases from 2.81 x 10-2 on the boundary-crossing window (t in [0.5, 1.5]) to 3.30 x 10-1 on severe OOD (t in [1, 2]), accompanied by a systematic leftward shift of internal layer mappings, where the average winner layer index drops from 5.71 (ID) to 3.00 (ID sub-window) down to 2.29 (severe OOD). Only 2 of 7 physical nodes maintain stable layer assignments across the boundary-crossing window, and only 1 of 7 under severe extrapolation, revealing potential internal functional correspondence changes before severe degradation in conventional output metrics. Additional results for linear oscillatory systems are further detailed in the appendix. 

---
# D-DOIT: Training-free Adaptation of Discrete Diffusion via Doob's h-Transform 

**Authors**: Jieke Wu, Qijie Zhu, Weimin Wu, Zeqi Ye, Minshuo Chen, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04938)  

**Abstract**: We propose D-DOIT (Discrete Doob-Oriented Inference-time Transformation), a training-free and efficient adaptation method for discrete diffusion models with generic rewards. D-DOIT formulates adaptation as sampling from a reward-tilted target distribution and realizes this transport through Doob's h-transform of the discrete diffusion reverse kernel, using only reward values rather than reward gradients. Unlike continuous diffusion, masked discrete diffusion samples categorical token-reveal transitions rather than continuous state updates. D-DOIT derives the corresponding discrete Doob's h-transform, which guides sampling by reweighting reverse transition probabilities instead of adding a drift correction. To make this transformation practical, D-DOIT avoids expensive future rollouts. At each guided step, D-DOIT samples candidate next states, uses the model prediction head to complete each candidate into a clean sequence, evaluates each completion with the reward oracle, and resamples the next state with probabilities proportional to the rewards. An optional late-stage best-of-K refinement further improves sample quality by branching trajectories only near the end of denoising, avoiding the $K$-fold cost over the full trajectory. Empirically, across regulatory DNA design and protein inverse folding benchmarks, D-DOIT outperforms training-free guidance baselines. It improves enhancer activity and cell-type specificity while preserving sequence naturalness, and achieves the highest success rate in protein inverse folding. 

---
# DiVeR: Decision-Critical Verifier Learning for VLA Test-Time Scaling 

**Authors**: Seongheon Park, Heecheol Kim, Shulin Tian, Lilika Makabe, Namiko Saito, Katsushi Ikeuchi, Sharon Li, Yasuyuki Matsushita  

**Link**: [PDF](https://arxiv.org/pdf/2610.04933)  

**Abstract**: Scaling robot data and model capacity has improved Vision-Language-Action (VLA) policies, but further progress is constrained by the high cost of robotic data. Verifier-guided test-time scaling offers an efficient alternative by sampling multiple action candidates and selecting the one most likely to lead to task success at inference time. Existing classification-based verifiers learn from trajectory-level outcomes but treat all visited states equally, even though their value for candidate discrimination can vary across a trajectory. At many states, plausible actions are similar and provide limited discrimination signal, while only a sparse set of decision-critical states admits meaningfully different actions that can substantially affect downstream outcomes. To address this, we propose DiVeR, which estimates decision criticality from the dispersion of sampled action representations. DiVeR then uses this signal to reweight verifier learning toward states where action selection is most consequential, without requiring step-level annotations or additional environment interaction. Across LIBERO, RoboCasa, and real-world experiments on a Franka Research 3 robot, DiVeR consistently improves task success through more effective verifier-guided action selection, while adding negligible verifier inference overhead. 

---
# Assembling Insights for Agentic Machine Learning Engineering Systems 

**Authors**: Bihui Jin, Yinxi Li, Kaiyuan Wang, Pengyu Nie  

**Link**: [PDF](https://arxiv.org/pdf/2610.04927)  

**Abstract**: Agentic machine learning engineering (MLE) is an emerging AI4SE application for complex ML tasks and a step toward recursive self-improvement of AI systems. Recent agentic MLE systems show the value of leveraging insights from related MLE tasks: some systems condition code generation on expert domain knowledge, which is implicitly curated from peer MLE tasks; some systems have a loop of solving an MLE task, gathering memory to benefit future tasks. However, insight collection and injection remain ad hoc, and existing evaluations often do not control insight sources, making data leakage a threat to validity.
To close these gaps, we systematically study how insights from related MLE tasks affect agentic MLE systems, and how their impact depends on insight type and injection method. We introduce MLE-InsightBench, a benchmark of 160 Kaggle competitions across 12 domains. To prevent leakage, one competition per domain is held out as the target, while the remaining 148 serve as sources only when they cannot have been influenced by the target task, such as through later editions or reused solutions. We also develop MLE-InsightForge, an insight assembly agent that explores source competitions, top human solutions, and writeups to construct three insight types: 12 domain insights summarizing best practices, 148 competition insights describing winning solutions, and 16 improvement insights capturing generic debugging and optimization strategies. These insights are injected either as on-demand skills or as an upfront playbook. In our evaluation, the extracted insights improve the normalized private-test score by 123.9% over a no-insight baseline. Controlled experiments further show that the three insight types are complementary; the better injection mode (skill vs. playbook) varies by tasks, with skills more cost-effective overall; and the gains generalize across backend LLMs. 

---
# TSAE: Structured Sparse Autoencoders for Interpreting Time-Series Forecasting Models 

**Authors**: Baoxi Liu, Yi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.04925)  

**Abstract**: Time-series forecasting informs critical decisions in energy dispatch, industrial operations, and environmental monitoring; understanding the patterns models rely on is essential for assessing reliability and identifying failures. Input attribution identifies important variables and time segments but offers limited insight into internal features, while standard sparse autoencoder (SAE) objectives do not directly constrain cross-variable structure or temporal continuity. We introduce TSAE, a structured sparse autoencoder for forecasting representations that decomposes hidden states into individually inspectable features. TSAE organizes cross-variable structure through shared and variable-routed private dictionaries, separates feature detection from magnitude estimation with gated encoding, and constrains neighboring sparse-code changes according to raw-segment similarity. These mechanisms support analysis of variable context, activation strength, and temporal evolution. Forecast-consistency fine-tuning further improves preservation of the frozen forecaster's outputs. The accompanying TSEVAL protocol separately audits fidelity, feature coherence, and physical calibration to ground feature interpretation. In three-seed experiments with frozen PatchTST on ETTh1, ETTh2, and ETTm1, TSAE achieves the lowest hidden-state reconstruction error, normalized forecast-reconstruction error (NFRE), and feature transition rate among five SAEs at comparable per-token activity. NFRE decreases by 4.3-26.4% relative to the next-best mean. Dataset-dependent tradeoffs in selectivity, physical correlation, and calibration show that fidelity and temporal-stability gains require independent semantic validation to support feature interpretation. 

---
# Complex Agents, Shallow Tests: Demystifying and Enhancing Test Adequacy of Agent Harness in the Wild 

**Authors**: Yifan Xiong, Jingyi Ge, Zhenpeng Chen, Yiling Lou  

**Link**: [PDF](https://arxiv.org/pdf/2610.04921)  

**Abstract**: LLM-based agentic systems are emerging as a new software paradigm. Modern agents are typically composed of backbone LLMs and a surrounding harness that serves as the operational software infrastructure for agent execution. As agent harnesses grow increasingly complex, agents suffer from diverse harness implementation bugs, raising substantial reliability concerns. In this work, we conduct the first empirical study to systematically investigate the test adequacy of harness in real-world agentic systems. Our analysis reveals that agent harness remains substantially undertested. In particular, LLM-dependent harness (LDH) code, despite its critical role in processing LLM outputs and governing agent behavior, receives limited testing attention, with less than half of its lines and branches covered by existing tests. Motivated by these findings, we further propose HarnessTester, the first harness-oriented test generation technique that incorporates explicit agent-harness contract support to construct contract-faithful test setups and extensively exercise LDH code. Our evaluation shows that HarnessTester substantially outperforms state-of-the-art general-purpose test generation techniques in achieving 75.95%/84.76% larger line/branch coverage gains and 69.89% larger mutation-score gains. Furthermore, HarnessTester detects 122 real-world harness bugs in widely-used agentic systems (e.g., OpenClaw), among which, 88 bugs are previously-unknown bugs and 69 bugs have been confirmed by agent developers. These results highlight the practical effectiveness of HarnessTester in improving test adequacy and assuring the reliability of real-world agentic systems. 

---
# PreAct-Nav: Agentic Reasoning Before Action for Urban Navigation 

**Authors**: Jing Xie, Shouwei Ruan, Yubin Wang, Yuxiang Zhang, Junwei Yang, Songchang Jin, Dianxi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.04916)  

**Abstract**: Urban navigation requires embodied agents to pursue long-horizon goals through local decisions based on egocentric observations. However, existing agentic navigation methods often struggle to translate distant goals into coherent local decisions in large-scale physical environments. Their reliance on linguistic reasoning over transient observations or limited history constrains anticipation of the consequences of actions and future conditions, despite the importance of such foresight for navigating long and complex urban routes. To bridge this gap, we propose PreAct-Nav, an agentic navigation framework that equips frozen policies with anticipatory reasoning for robust urban navigation. Our central idea is to anchor local decisions in persistent medium-horizon subgoals, assess the consequences of predicted actions before execution, and continually update the reasoning context using actual outcomes. At its core, a navigation memory module maintains the active subgoal and relevant experience across decisions, translating distant goals into actionable intermediate objectives. We further introduce a predictive world sandbox that uses an action-conditioned world model (AC-WM) to forecast world dynamics conditioned on candidate movements. A vision-language model (VLM) reasoner interprets these predictions under the current subgoal to retain or revise actions. After execution, real observations are used to assess outcomes, correct inconsistent assumptions, and update memory to continue or reformulate the subgoal. Extensive evaluations demonstrate that the proposed PreAct-Nav improves action selection through memory updates and visual prediction, with more pronounced gains on longer routes and routes with more turns. 

---
# Why Subliminal Learning Needs So Much Data: A Noisy Inverse View through Steering Vector Recovery 

**Authors**: Luoyu Chen, Xiaoyu Ding, Weiqi Wang, Chenhan Zhang, Zhiyi Tian, Jianhuan Huang, Shui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04907)  

**Abstract**: Subliminal learning lets a student inherit a teacher's behavioral trait from semantically unrelated data, yet published demonstrations typically require tens of thousands of carrier examples. We ask where this data requirement comes from. Our testbed is subliminal steering: the teacher trait is a known residual-stream vector $\Delta_T$, so transfer can be measured directly as parameter recovery. On identical carrier prefixes, we compare token-level (hard) NLL supervision with full-distribution (soft) KL supervision. At initialization the two objectives give nearly collinear gradients, and both align poorly with $\Delta_T$. Under iterative optimization, however, they diverge: soft supervision recovers $\Delta_T$ almost exactly from a few hundred carriers, while hard supervision stays well below it even with tens of thousands. We explain this gap by casting steering-vector distillation as a noisy linear inverse problem. Locally, the carrier task maps the trait through its Fisher matrix $F$, so gradients point toward $F\Delta_T$ rather than $\Delta_T$. Gradient descent then acts as a progressively less-damped inverse of $F$. With soft targets, this inverse restores low-curvature directions. With hard labels, it also amplifies the sampling noise in those same directions. The result is an optimal inversion depth that grows with the number of independent carriers. Experiments on Qwen2.5-7B and Gemma-2-9B confirm four predictions: the Fisher distortion of the initial gradient, recovery ordered from steep to flat directions, an optimal depth that shifts with data scale, and the finding that resampling completions from a fixed prompt pool works as well as adding new prompts. In this setting, large carrier datasets are needed less to reveal the trait than to suppress label noise amplified by Fisher inversion. Code is available at \url{this https URL}. 

---
# Scaling Verifiable Environments for Long-horizon Work Agents 

**Authors**: Jiazheng Zhang, Long Ma, Yunxian Yang, Zhiheng Xi, Zhikai Lei, Yajie Yang, Chenyang Liao, Enyu Zhou, Yang Nan, Yuchen Tian, Senjie Jin, Yibo Wang, Wei He, Boyang Liu, Jixuan Huang, Xin Guo, Zhezheng Hao, Xinbing Liang, Zhihao Zhang, Changzhi Zhou, Wiggin Zhou, Tao Gui, Qi Zhang, Xuanjing Huang, Clarenceai, Aiden Adams  

**Link**: [PDF](https://arxiv.org/pdf/2610.04906)  

**Abstract**: Work agents operate over digital artifacts to execute professional knowledge-intensive work, requiring training environments that support long-horizon interaction and trustworthy verification. However, hand-crafted environments incur prohibitive engineering overhead that prevents environment scaling, whereas synthesis methods sacrifice workspace complexity, realism, or grounded verifiability. To bridge this gap, we introduce WorkForge, a scalable synthesis framework for constructing verifiable work-agent environments from real-world resources. Starting from expert workflows, WorkForge first identifies the resources, decisions, and deliverables required by each workflow. It then retrieves relevant real-world files and organizes them into a workspace. WorkForge inspects the workspace to extract concrete, checkable facts about its content. These factual anchors fix which task types the workspace can support and how their outcomes can be verified. Therefore, WorkForge derives each task's instructions, solution plan, and complementary programmatic and semantic verifiers directly from these factual anchors, keeping verification traceable to observable workspace evidence. Furthermore, we construct 16.7K verifiable environments across 40 professional domains, with workspaces collectively covering 60 file types. Post-training Qwen3.5-35B-A3B-Base improves GDPVal from 45.5 to 73.6 and APEX Score from 5.0 to 21.3, while enabling Qwen3.5-27B to achieve highly competitive performance and outperform strong competitors. Our analyses confirm the efficacy of the proposed method and reveal consistent scaling behaviors across both data volume and interaction horizons. 

---
# ScopeSAE: Model-Scope Feature Discovery with Interpretable Layer Selection 

**Authors**: Qingwen Zeng, Zehao Fu, Shuyu Meng, Linghan Huang, Jiayi Zhang, Chenglin Wu, Ling Chen, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.04905)  

**Abstract**: Sparse autoencoders (SAEs) are a central tool in mechanistic interpretability. However, existing SAEs are primarily trained per layer. The modeling subspace is therefore fixed by layer identity, independent of which token-layer states actually drive each prediction. We argue that this constraint contributes to several limitations observed in layer-wise SAEs, including low feature utilization, high dictionary redundancy, and features that lack direct behavioral grounding. In this paper, we propose ScopeSAE, which selects the modeling subspace per token by attributing each prediction to its most influential token-layer state via normalized gradient-based attribution, and learns features over the resulting prediction-relevant subspace. Empirically, ScopeSAE yields an effect we term reconstruction-better-than-original. Written-back reconstructions of the SAE produce lower next-token cross-entropy than the original activations, an outcome that, to our knowledge, has not previously been reported for SAEs. Through interventional analyses and a KL fine-tuning counter-experiment, we show that this effect is attributable to ScopeSAE's prediction-relevant subspace itself rather than to architectural changes. ScopeSAE further improves effective feature count, interpretability, utilization, and dictionary redundancy over existing layer-based baselines, suggesting that choosing the SAE modeling subspace by predictive relevance leads to more useful and behaviorally meaningful features. 

---
# A Systematic Analysis of the Predictive Power of LM Surprisal in Reading Chinese 

**Authors**: Hongao Zhu, Muxiaoqiao Xu, Yikang Liu, Siyuan Song, Yuxia Wang, Byung-Doh Oh, Hai Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04898)  

**Abstract**: This study analyzes the predictive power of LM-derived, token-level surprisal on Mandarin Chinese reading times. We first propose the Shortest Matching Sequence (SMS), an alignment scheme that maps between the word segmentation assumed by eye-tracking corpora and the LMs' subword tokenization, as the two tokenizations often disagree in the context of Mandarin Chinese. Then, using a suite of Chinese-Pythia models (14M-1.4B) trained on scratch with 30B tokens, we examine how well surprisal predicts first fixation duration, gaze duration, and total reading time in three paragraph-level eye-tracking corpora of Mandarin Chinese (GECO-CN, HKP, and MECO). Contrary to previous null findings, our results show that surprisal is predictive of Chinese reading times. However, whether predictive power scales with model size and the amount of training is corpus-specific: bigger models predict better in GECO-CN, whereas inverse scaling emerges in HKP and, at the largest sizes, in MECO. Subsequently, we tested one possible explanation for the inverse scaling in HKP and found that checkpoints whose surprisal remains closer to $n$-gram statistics are better predictors of reading. All in all, the predictive power of surprisal on Chinese reading time measurements is corpus-specific, which cautions against drawing scaling conclusions from a single corpus. 

---
# SFlexRCA: Lightweight, Scalable, and Flexible Root Cause Analysis for IIoT Edge Systems 

**Authors**: Amr M. Zaki, Farhoud Jafari, Honggeun Ji, Komal Sarda, Marin Litoiu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04893)  

**Abstract**: Industrial Internet of Things (IIoT) systems generate high-dimensional sensor telemetry from interconnected components, where faults can propagate across the system. To address these challenges, we propose SFlexRCA (Scalable and Flexible Root Cause Analysis), a topology-free RCA framework designed for resource-constrained IIoT environments. SFlexRCA transforms multivariate telemetry into compact orthogonal representations and applies shared lightweight linear modeling, avoiding explicit graph construction, message passing, and per-variable or lag-specific parameter growth. We evaluate SFlexRCA on three publicly available IIoT datasets, BATADAL, SWaT, and WADI, spanning different numbers of monitored variables, temporal characteristics, and training-data regimes. SFlexRCA is compared with 10 statistical, causal, and non-causal baselines in terms of RCA accuracy, training efficiency, inference latency, and memory consumption. In addition, inference efficiency and memory consumption are evaluated on Raspberry Pi 3 and Raspberry Pi 5, while energy consumption is additionally measured on Raspberry Pi 5. We further investigate temporalcontext sensitivity, architectural and loss components, and alternative representations. Notably, SFlexRCA maintains strong localization performance on BATADAL despite its limited normal-operation training data, while its compact shared architecture avoids the parameter growth associated with causal and graph-based approaches. Its lightweight shared architecture further enables efficient deployment on resource-constrained IIoT edge devices. The SFlexRCA code is available at this https URL Analysis-Correlation-Attentive-Modeling. 

---
# ResOPD: Tail Residualization for Sparse On-Policy Distillation 

**Authors**: Penghui Yang, Long Xing, Xuanlang Dai, Ziyu Liu, Kai Chen, Yuhang Zang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04882)  

**Abstract**: On-policy distillation (OPD) trains a student model on self-generated trajectories, but transmitting dense teacher distributions across long reasoning traces creates prohibitive communication and memory bottlenecks. Practical systems therefore rely on sparse teacher interfaces, typically transmitting either the sampled-token score or a small Top-$k$ distribution. However, this sparse setting faces a fundamental dilemma: sampled-token estimators are unbiased but suffer from severe gradient variance, whereas directly optimizing Top-$k$ objectives introduces systematic bias. To improve this trade-off, we propose ResOPD (On-Policy Distillation with Tail Residualization), which provides unbiased full-vocabulary reverse KL gradient estimation under on-policy sampling, with substantial variance reduction under the same sparse payload in the evaluated settings. ResOPD aggregates the unobserved vocabulary into an observable coarse tail event, computes its exact aggregate gradient, and samples only the fine-grained within-tail residual, which requires no additional teacher queries or forward passes. Extensive experiments demonstrate that ResOPD substantially reduces gradient variance, stabilizes online training dynamics, and improves downstream performance across the evaluated settings. These results establish ResOPD as an efficient, plug-and-play variance reduction primitive for sparse on-policy distillation. 

---
# Greedy Local Learning for Language Model Pretraining: Gaps and Objective Design 

**Authors**: Jihwan Moon, Sheir A. Zaheer, Jinmyoung Lee, Gunhee Kim, Chan Y. Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.04867)  

**Abstract**: Greedy block-wise local learning splits a network into gradient-isolated blocks trained by local auxiliary losses, deleting the backward pass between blocks: inter-stage communication becomes forward-only and every block can step its optimizer independently, properties directly relevant to decentralized model-parallel training. Local learning is competitive with end-to-end backpropagation on image classification, and on small Transformers it is known to trade a worse best loss for parallel speedup. How this loss gap behaves in autoregressive language model (LM) pretraining at larger scale, and which auxiliary designs reduce it, has not been measured. We present a token-budget-matched empirical study at 125M and 400M parameters with $K \in \{1,2,4\}$ blocks at Chinchilla-optimal budgets, factorizing the auxiliary design into network architecture and training objective. We observe: (i) the gap to end-to-end training more than doubles from $K=2$ to $K=4$, but at $K=4$ shrinks from 125M to 400M; (ii) replacing an MLP auxiliary with a Transformer-based one is a strong network-side intervention, recovering 22-41% of the gap; (iii) a multi-token-prediction (MTP) auxiliary objective helps at the first block boundary, whereas adding it at deeper boundaries hurts, and restricting it to the first block yields the best $K=4$ configuration ($+0.062$ vs. $+0.075$ nats at 400M); and (iv) deployment-style per-block execution reduces activation memory by up to $2.2\times$. We frame these results as an empirically grounded method direction rather than a finalized method: local objectives should apply future-predictive pressure selectively across boundaries while resisting shortcuts that bypass predictive content. 

---
# Enhancing Long-Video VLM Embeddings with Query-Aware Streaming Latent Reasoning 

**Authors**: Haozhe Chi, Song Jin, Yang Jin, Yadong Mu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04864)  

**Abstract**: Long-video embedding requires capturing sparse query-relevant evidence under a limited visual-token budget. Uniform sampling can miss brief events in videos spanning minutes or hours, whereas encoding more frames in a single context increases memory and computation. We introduce \textbf{Query-Aware Streaming Latent Reasoning} (QASLR), a post-training framework that accumulates evidence across clips while keeping the embedding size fixed. QASLR selects a bounded set of frames, restores their temporal order, and processes them clip by clip with a vision-language backbone. A compact set of persistent think tokens cross-attends to each clip's features, while an embed token reads out a normalized representation after every update. This design integrates evidence across multiple backbone calls without requiring all selected frames to share a single context. Training combines final contrastive learning, step-wise contrastive supervision, and final-embedding self-distillation. Intermediate supervision trains partial-video readouts for retrieval, while self-distillation regularizes them toward the final representation. Query-aware selection produces query-conditioned representations for candidate-set scoring and reranking, whereas query-independent selection enables reusable corpus indexing. Under the full training recipe, HourVideo retrieval Hit@1 increases from 54.2 to 70.7 and from 57.4 to 72.8 for 2B and 8B Qwen3-VL-Embedding backbones, respectively. Gains extend to the evaluated moment-retrieval and video-QA tasks, and the streaming head transfers to a second Qwen-family embedding backbone. These results support streaming latent aggregation as an effective approach to integrating long-video evidence into fixed-dimensional representations. 

---
# Invisible Ink, Visible Lies: How Production Watermarking Causes LLMs to Hallucinate 

**Authors**: Haocheng Ye, Aoting Hu, Xinwei Zhang, Xunzhu Tang, Shuchao Pang, Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2610.04860)  

**Abstract**: Text watermarking helps identify AI-generated content, but its effect on factual reliability remains underexplored. In this paper, we study watermarking hallucination: factual errors induced or amplified by watermarking even when the required evidence is present in the context and the unwatermarked model can answer correctly. Using a controlled retrieval-augmented generation setting, we compare unwatermarked and watermarked generations under the same context, query, and decoding configuration, and quantify their factual accuracy decrease. Across six representative watermarking methods, including KGW, SWEET, DiPmark, GumbelSoft, Gumbel-Max, and SynthID watermarking, we consistently observe watermark-induced hallucination. Watermarked outputs can remain fluent while introducing factual errors. We attribute this failure mode to two mechanisms: (1) token perturbations in the current-step arising from method-specific reweighting or keyed sampling, and (2) prefix-induced attention drift, which accumulates through autoregressive decoding and weakens later attention to the factual context. Motivated by this analysis, we propose two plug-in interventions at the token and attention levels that can be integrated into existing watermarking methods to improve factuality. At a matched TPR of 0.90 at 1% FPR, combining the two interventions reduces factual errors by approximately 90% relative to watermark-only decoding while preserving fluency and comparable decoding efficiency. Overall, this work highlights factuality as a first-class criterion in watermark evaluation, alongside detectability and robustness, and calls for careful factuality validation before deploying watermarks in fact-critical applications. 

---
# PIT-GCL: Protein Interaction using Topological Graph Contrastive Learning 

**Authors**: Jae Won Choi, Ryoonki Hong, Alan Liang, Manjula Adiveppa Wader, Bingsong Zeng, Peiyang Tang, Longwei Liu, Ruishan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04850)  

**Abstract**: Protein binding prediction is central to target identification, therapeutic binder design, and large scale screening, yet remains challenging because binding depends on sequence, three dimensional geometry, and global structural organization. Recent folding models such as AlphaFold3 and Boltz-2 have substantially improved structure prediction, but their confidence outputs (pLDDT, pTM, ipTM) are not specifically designed for binary binding prediction, and dedicated structure aware predictors often require bound complex structures that are unavailable at screening scale. We introduce PIT-GCL, a dual tower structure aware framework that encodes each protein independently from its amino acid sequence, C{\alpha} point cloud, and a global persistent homology descriptor. Each tower combines residue ESM-2 embeddings with a topological summary computed from the H0 and H1 persistence landscapes of a Vietoris-Rips filtration, and processes the resulting tokens with a structure aware Transformer in which pairwise C{\alpha} distances enter as a learned attention bias. A bidirectional cross attention module then performs latent space soft docking between the two per-protein representations, and the model is trained with a combined binary cross entropy and NT-Xent contrastive objective. On three binary interaction prediction benchmarks, general PPI on PPIRef, TCRpMHC binding on STAG, and whole chain pairs on PPB-Affinity, PIT-GCL outperforms representative sequence based, structure aware, and task specific baselines on general PPI under our evaluation, and is the only method above chance on PPB-Affinity; on TCR-pMHC it leads at a fixed decision threshold but is outranked by a task specific sequence model. Because each protein is encoded independently in the first phase, its representation can be precomputed and reused across candidate pairs, which is convenient for large scale screening. 

---
# Cluster Validation Indices as Self-Supervised Objectives for Text Representation Learning 

**Authors**: Kishor Kumar Bhaumik, Nicolas Roque dos Santos, Neil Shah, Jia Chen, Evangelos E. Papalexakis  

**Link**: [PDF](https://arxiv.org/pdf/2610.04830)  

**Abstract**: Self-supervised fine-tuning refines the embedding space of a pretrained language encoder without labels. However, the commonly used approaches are computationally expensive. Specifically, contrastive learning-based methods need multiview data and in-batch negative examples, while negative-free approaches require auxiliary graphs/networks. An interesting question arises: can self-supervised fine-tuning be done without relying on either additional negatives or graph data? To answer this question, we introduce SilK (Silhouette-guided K-means), which trains on a Cluster Validation Index, an internal measure of cluster quality without using labels. SilK clusters the corpus and then regresses a simplified silhouette toward a target value. Each document is compared only against the k cluster centroids, never against other documents, so the method needs no augmentation, no negative pairs and one view per document. On BERT-base, SilK trains 1.46x faster per epoch than the fastest baseline we evaluate and uses 45.4% less peak GPU memory than the leanest one. Under frozen-encoder linear probing, SilK stays competitive with the best baselines on three downstream tasks. 

---
# Viva La Vida: Verification and Accumulation Failures in Multi-Agent Proof Search 

**Authors**: Benji Xu, Ken Zheng, Noah Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.04829)  

**Abstract**: When an agentic prover works on an open problem, there is no proof assistant to fall back on: its verifier and lemma library are ultimately language models judging model outputs. We instrumented such a system end to end and analyzed $51{,}754$ traced observations across three full runs ($186$ hours, \$$5{,}694$). We find three connected failure modes. First, the three-model verifier requires unanimity and treats parse or API failure as non-approval; in $10$ of $12$ verification events, one member returned no parseable output or an API error, making acceptance arithmetically impossible without surfacing an error. Second, when the ensemble did function, one verifier approved $3$ attempts that GPT rejected, each claiming to resolve the open problem; a single-verifier design would therefore have announced a solution three times. Third, because nothing could be approved, every review was a refutation, yet the lemma extractor mines reviews as well as proofs: $24$ of $93$ lemmas ($26\%$) were extracted from rejected arguments with their refutational context removed. Taken together, these findings show that without external verification, supervision is itself a critical trust boundary: systems must distinguish abstention from rejection, preserve useful disagreement, and preserve the provenance and polarity of information before it becomes future context. 

---
# Separating Decision Time from Decision Quality in the Real-Time Gap of Distilled Deciders: Evidence from a Game and a Conveyor Simulator 

**Authors**: Chihoon Shin, Junyeong Lee, Kihyeok Jeong, Wonok Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2610.04810)  

**Abstract**: Real-time agents are often evaluated with the world paused, or with decision time rounded to whole ticks (tick conversion). Tick conversion predicts almost no loss for a learned decider whose hit accuracy falls by 14.8 percentage points (pp) on the same games in asynchronous play. We instead split the decider's gap to a zero-latency teacher into time and quality components. In ViZDoom, we train a small decider by imitating a scripted teacher, run the game on wall-clock time, and add a control in which the teacher waits for a call to the decider's server before deciding. On a Windows host, three preregistered studies put the time component at 6.5-8.7 pp and the quality component at 6.3-8.6 pp. On a Linux host that skips 0.08-0.09% of ticks, the time component disappeared (-0.1 pp; paired 95% bootstrap interval [-0.4, 0.0]) while the quality component remained (5.6 pp [3.7, 7.5]). Retraining on teacher-labelled states from the decider's own play (DAgger) improved it on new games on both hosts (2.6 pp [0.4, 4.9] and 3.1 pp [1.1, 5.2]), a preregistered partial success. Imposing one delay schedule on every arm in live play left a quality component of 5.8 pp [4.0, 7.5] on 100 new games, and retraining cut the decider's disagreement with the teacher on its own states from 22.7% to 14.9%. In a conveyor simulator with a 400 ms deadline, delay past it erased the quality component. A latency-matched control whose delays match the decider's shows which remedy to try. 

---
# Active-DiNTS: Active Differentiable Network Topology Search 

**Authors**: Gean Trindade Pereira, Thierry Urruty, Muriel Visani, André C. P. L. F. de Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2610.04787)  

**Abstract**: Neural Architecture Search (NAS) has proved to be a strong alternative to manual network design, but applying it to 3D medical image segmentation is limited by two well-known costs, large annotation budgets and multi-GPU clusters. Thus, this paper introduces Active-DiNTS (Active Differentiable Network Topology Search), an approach that embeds pool-based Active Learning (AL) into a bi-level differentiable topology search to perform architecture discovery and label curation jointly. At each query round, unlabeled MRI volumes are ranked by one of three uncertainty signals (Entropy, Variance, or Standard Deviation), and only the top-ranked volumes are sent to an oracle for annotation. The new labels feed two interlocked stages. Network weights are updated in an outer loop, while the macro/micro topology of a U-Net-style backbone is refined in an inner loop. Three AL regimes (weights-only, topology-only, joint) expose the speed-accuracy trade-off. Evaluations on the Medical Segmentation Decathlon (MSD) Task01 BrainTumour benchmark showed that Active-DiNTS surpasses DiNTS, C2FNAS, and nnU-Net in Dice, with gains of about 10 percentage points on Edema and 5 points on Non-Enhancing core, on a single GPU and using a fraction of the labeled volumes. The discovered architectures are denser and more FLOP-heavy than prior baselines, but remain competitive in trainable parameters and peak memory; the fastest search variant finishes in under 0.25 GPU-days, over 27x faster than the eight-GPU DiNTS search. Together, these results indicate that pairing differentiable NAS with active data acquisition is a practical recipe for accurate 3D segmentation under realistic constraints. 

---
# ARISE: Adaptive Agentic Reasoning with Image-grounded Self-Evaluation for Interpretable IBD Assessment 

**Authors**: Pronoma Banerjee, Anuva Shah, Jason Wu, Md. Masudur Rahman, Sanjay Mohanty, Satya Kurada, Juan P. Wachs  

**Link**: [PDF](https://arxiv.org/pdf/2610.04777)  

**Abstract**: Inflammatory bowel disease (IBD) requires frequent imaging-based assessment, yet interpretation of modalities such as wireless capsule endoscopy (WCE) and intestinal ultrasound remains heavily dependent on specialist expertise. Vision-Language Models (VLMs) demonstrate significant potential in multimodal medical image analysis, but their clinical adoption is hindered by their insufficient domain-specific reasoning, susceptibility to hallucination, scarcity of high quality training data in fine-grained diagnostics and limited interpretability. We introduce ARISE (Adaptive Agentic Reasoning with Image-grounded Self-Evaluation), an autonomous planning framework that models few-shot medical image understanding as a sequential agentic workflow. ARISE structures agent execution into a transparent 5-stage workflow: hypothesis generation, image-grounded evidence summarization, evidence-conditioned refinement, symbolic verification, and final diagnosis. We apply ARISE to IBD assessment across two independent patient cohorts: wireless capsule endoscopy (WCE) images for Crohn's disease and B-mode ultrasound data for ulcerative colitis. ARISE consistently improves diagnostic performance over baseline VLMs while exposing where reasoning succeeds or fails, providing a more interpretable basis for clinical decision support and realistic deployment. 

---
# EasyClassifier: Honest, Reproducible Machine-Learning Classification for Researchers Who Do Not Program 

**Authors**: Ahmad B. A. Hassanat, Ghada A. Altarawneh  

**Link**: [PDF](https://arxiv.org/pdf/2610.04758)  

**Abstract**: Machine-learning classification is now used across medicine, the social sciences, economics, education and engineering, very often by researchers who do not program. Two errors recur in such work: preprocessing learned on all rows before cross-validation (data leakage), and reporting the cross-validation score of the classifier that won a comparison (selection bias). Both make published scores optimistic. We present EasyClassifier, an open-source Python package that guides a user from a CSV or Excel file to a finished analysis through a sequence of plain-language, multiple-choice questions, with no code. It makes the correct procedure the default rather than an option: every preprocessing step is learned inside each training fold; the selected classifier receives a separate final score from nested cross-validation (up to 2,000 rows), or an untouched 20% test set; classifiers run with fixed default parameters and fixed random seeds; and every run writes a report with a ready-to-adapt Methods paragraph, the references to cite, and figures prepared for publication. On ten public datasets from seven fields and a random-label control, split in half so that one half served as an untouched external test, the common practice overstated balanced accuracy by 2.9 percentage points on average (up to 9.9 on small data), whereas the score EasyClassifier reports had a mean signed bias of +0.7 points and a mean absolute error of 2.3 points (one-sided Wilcoxon p = 3.2 x 10^{-4}, 30 runs). On random labels it reported 48.8% balanced accuracy against a chance level of 50%. EasyClassifier is available under the MIT license from PyPI (pip install easyclassifier), GitHub, and Zenodo. 

---
# More Value per Key: Asymmetric Sparse Attention for Faster LLM Decoding 

**Authors**: Noam Elata, Itay Lamprecht, Mikey Shechter, Daniel Ohayon, Itay Hubara, Daniel Soudry  

**Link**: [PDF](https://arxiv.org/pdf/2610.04753)  

**Abstract**: utoregressive generation in Large Language Models (LLMs) is constrained by the memory and computational demands of attention mechanisms. Sparse attention methods mitigate this cost by selecting only high-probability entries of the attention matrix. We observe that in many such methods, this renders the probability-value multiplication negligible, shifting the bottleneck to the query-key step. Key heads can therefore be reduced to accelerate inference, while retaining more value heads preserves capacity with limited additional decoding cost. We introduce Sparse Asymmetric Group-Query Attention (SAGA), which decouples key and value head counts to exploit this principle, and pair it with approximate top-N (Atop-N) attention, a simple sparse attention method designed to study the interaction between sparsity and head-count asymmetry. We formalize the benefits of this asymmetry theoretically and validate them empirically through latency measurements and quality evaluations on models up to 1.5B parameters. Together, SAGA and Atop-N achieve end-to-end decoding speedups exceeding $2\times$ over our full-attention GQA baseline at long contexts. Models trained from scratch with SAGA nearly match the quality of comparable GQA variants on the evaluated benchmarks. To facilitate adoption, we introduce an efficient fine-tuning method that converts pretrained models to the SAGA architecture, enabling practitioners to benefit from our approach without costly retraining. 

---
# Robot Learning with Visual Predicted Force 

**Authors**: Haonan Chen, Feiyang Wu, Yuxiang Ma, Mustafa Mete, Pengfei Ye, Junxuan Shen, Cheng Zhu, Aurora Ruggeri, Kelvin Cheung, Jiayuan Mao, Edward Adelson, Jiajun Wu, Robert D. Howe, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.04741)  

**Abstract**: Force-aware manipulation typically relies on specialized force or tactile sensors. We show that force-aware manipulation can instead be achieved through visual force prediction from the deformation of a compliant Fin Ray gripper. Our approach trains two models. First, we train a visual force estimator on calibration data and use it to annotate task demonstrations with force estimates. Second, we train an action--force proposal policy on these force-augmented demonstrations to jointly generate candidate robot actions and their associated forces. At test time, we sample candidate actions and the forces they are expected to produce, then execute the action whose predicted force is closest to a target from the demonstrations. We evaluate our approach on berry picking, empty-can grasping, in-hand reorientation, and plug insertion. Our results show that visual force prediction can guide inference-time action selection for contact-rich manipulation without requiring force or tactile sensors at deployment. 

---
# Knossos and Ariadne: Benchmarking and Learning Complete Diagram Topology Extraction with Vision-Language Models 

**Authors**: Bangwei Guo, Xujiang Zhao, Shengyu Chen, Yanchi Liu, Wei Cheng, Xi Zhu, Guoning Zhang, Dimitris N. Metaxas, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.04721)  

**Abstract**: Structural diagrams are widely used to represent complex systems and relational information across scientific, engineering, procedural, and spatial domains. Recent vision-language models (VLMs) have become increasingly capable of recognizing diagram elements and reasoning about their content, while complete diagram topology extraction remains comparatively underexplored. In this paper, we study diagram-to-graph topology extraction: extracting all diagram entities and the complete relations among them. To enable large-scale supervised training and systematic evaluation of this task, we introduce Knossos, a benchmark of 19,200 diagrams across six diverse domains, with 245,179 nodes and 439,740 edges. Its symbolic generation process provides exact alignment between rendered diagrams and annotations of complete topology, relation types, and connector geometry. To address the modeling challenge of complete topology extraction, we also present Ariadne, a structured framework that decomposes the task into node inventory extraction and source-conditioned edge prediction. Extensive experiments show that training on Knossos substantially improves complete topology extraction in smaller open-source VLMs. Ariadne further improves over one-step extraction under matched supervision, demonstrating the additional benefit of structured decomposition. It achieves the highest average Edge F1 among the evaluated methods on Knossos, while both backbone variants also improve over their unadapted counterparts on the real-world external benchmark. Code and benchmark are available at this https URL. 

---
# Towards Automatically Pruning Logging Code with Coding Agents: How Far Are We? 

**Authors**: He Yang Yuan, Haonan Zhang, Xin Wang, An Ran Chen, Kisub Kim, Zishuo Ding, Zhenhao Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.04716)  

**Abstract**: Logging code supports debugging, monitoring, and software maintenance, but excessive logging can add noise, impose runtime overhead, and obscure diagnostic information. While prior research has extensively studied logging code generation and modification, logging removal remains comparatively underexplored. In this paper, we study developer logging removal practices and explore the use of coding agents for this task. We extract and manually validate logging removal cases from Python and Java repositories and derive 10 removal patterns and 11 removal reasons that characterize how and why logging code was removed in real-world software changes. We further construct LogRem, a dataset of 387 real-world cases covering direct logging statement removal, logging infrastructure removal, and logging replacement. We evaluate four coding agents with multiple model settings and compare their outputs with accepted real-world changes. Although 95.6% to 100.0% of outputs pass validity checks, only 11.1% to 19.6% remove the same logging code as the corresponding real-world change while preserving unrelated code. Agents differ through missed removals, extra removals, and unrelated code edits, with substantial variation across logging removal categories and trajectories. Execution cost varies widely, but higher cost does not consistently yield closer alignment. Commit messages and developer discussions provide the largest alignment gains, while taxonomy guidance consistently reduces runtime. Overall, our study establishes logging removal as a distinct software maintenance task and shows that reliable automation depends on accurately determining removal scope while preserving necessary code. To the best of our knowledge, this is the first study to examine logging code removal from this perspective. 

---
# Organising Trajectory Evidence for Language-Model Agent Assurance: Fragments, Methods, and the Residual 

**Authors**: Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04710)  

**Abstract**: Methods for assessing language-model agents include rule checkers over logs, analyses of skill coverage and composition, support checkers, prefix monitors, execution gates, and rare-event estimators. Each observes a different part of a run and makes a claim of a different strength, and no common account says how these claims combine or what they leave unchecked. We give one, built from a logic, information fragments, and an assurance ledger. Requirements are formulas of a two-tier logic: an outer finite-trace temporal logic over recorded events, and an inner logic of standing over the argument structure that the agent's recorded context supports at a decision. A rule violation and an action taken on withdrawn support are thus formulas of one language. The information available to an assessor, to the agent, and to an execution gate defines fragments of that language; each checking method decides one fragment and returns a typed claim: exact, on a named test suite, a risk bound, or descriptive. The ledger merges the evidence and classifies every obligation as established, addressed but not established, or unaddressed. On 456 released $\tau^2$-bench telecom trajectories, the benchmark oracle flags 231 runs; adding a rule checker, a support stand-in, and a prefix-monitor baseline raises the union to 391, 401, and 407, each contributing flags the others miss, and a gate makes one prohibition exact on a gated deployment. The 49 unflagged runs and the unmet or unaddressed obligations form the residual. Discovery makes unknown requirements explicit, and new or stronger checkers then reduce it. 

---
# VoCa: Designing Speech-Canvas Interaction for Voice-Based Conversational Agents 

**Authors**: Yate Ge, Run Yuan, Yueran Qi, Wenjie He, Jiaqi Mo, Yangshuo Chen, Wenbin Zuo, Xiaohua Sun, Weiwei Guo, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04706)  

**Abstract**: People write and sketch while speaking to explain, organize, and develop content together. Inspired by these practices, we investigate how voice agents can use a canvas alongside speech in multi-turn conversations with users. We conducted a two-part formative study: an observational study of how pairs coordinated speech and boardwork, followed by a design workshop that informed a design space for speech-canvas interaction with voice agents. Building on these insights, we developed VoCa, a voice agent that coordinates speech with visual object creation, annotation, and attention guidance. A five-day deployment with 18 participants examined usability, experiences of speech-canvas interaction, patterns of use, and desired improvements. Participants' experiences highlighted opportunities for speech-canvas interaction in learning, work, and daily life, alongside challenges in coordinating what agents say and show in ways users can follow and influence. These findings inform how voice agents can use a canvas alongside speech in conversation. 

---
# Neurodiversity-Aware Multimodal Affective Computing for Neurodevelopmental Assessment: From Norm-Referenced Classification to Context-Sensitive Decision Support 

**Authors**: Mateusz Pomianek, Anna Łężniak-Seruga  

**Link**: [PDF](https://arxiv.org/pdf/2610.04705)  

**Abstract**: Automated neurodevelopmental assessment increasingly combines computer vision, speech, eye tracking, physiology, and machine learning, yet multimodality and discrimination do not establish construct validity, clinical usefulness, or appropriate interpretation of behavioral variation. We propose a testable architecture for neurodiversity-aware multimodal affective computing in which population-relative deviation is not treated as sufficient evidence of adverse functioning. This conceptual article reports neither a new participant-level study nor a trained predictive system, and no predictive or clinical superiority is claimed. The framework keeps observable signals, modality quality and availability, context, population-relative information, pooled within-person references and, when sufficiently supported, context-conditioned within person references, and predictive uncertainty distinguishable throughout inference. It defines structured decision-support outputs and auditable requirements for target validity, temporal integrity, contextual leakage, missing-modality robustness, calibration, subgroup evaluation, interpretability, privacy, and human decision boundaries. The contribution is architectural rather than algorithmic: it specifies empirically testable constraints that can be instantiated with different multimodal learning methods. Autism provides the principal motivating evidence base, without assuming unchanged transfer to other neurodevelopmental conditions. 

---
# Extracting Persona Subspaces Through Iterative Nullspace Projection For Modulation 

**Authors**: Ananya Malik, Mai ElSherief  

**Link**: [PDF](https://arxiv.org/pdf/2610.04676)  

**Abstract**: Large Language Models (LLMs) can adopt distinct personas to tune their semantics, expertise, and perspective to different users and tasks. Precise control over these traits is critical to ensure safety and reliability in model behavior. Existing methods like activation steering and prompt-based persona induction reduce a persona to a single dominant direction, missing the finer, nested traits that emerge only once that dominant signal is factored out. We introduce modulation as a setting where the persona context is already embedded in the content being manipulated, requiring control methods to amplify or suppress a trait already present rather than inject it from scratch. PaSS is an inference-time control paradigm that models personas as multi-dimensional subspaces in a model's latent space without supervised contrastive examples. The persona subspaces are extracted via iterative concept erasure and applied to modulate persona-guided generation without retraining. To extract this subspace, we use Iterative Nullspace Projections (INLP) to linearly and iteratively isolate persona-specific directions. We causally evaluate six personas against diverse tasks like MATH-500, TinyAlpaca, GSM8K, and IFEval, showing that discriminative, iterative subspace extraction captures diverse traits underlying a given persona, enabling stronger and larger modulation than single-direction additive methods, while maintaining content fidelity. We further study individual peeled directions within each subspace to uncover the distinct aspects of persona behavior they encode. Overall, we show that persona subspaces offer a controllable, interpretable, and generalizable framework for modulating LLM behavior without sacrificing task performance. 

---
# RETRACE: From Entangled Repair Histories to Reusable Experience for CI Repair 

**Authors**: Rabeya Khatun Muna, Muhammad Ahasanuzzaman, Nakhla Rafi, Yisen Xu, Jinqiu Yang, Tse-Hsun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.04658)  

**Abstract**: Large language model (LLM) agents increasingly reuse prior experience, but most approaches assume that problems and solutions are already aligned. Software histories rarely provide this alignment: a pull request (PR) may contain multiple continuous integration (CI) problems, failed attempts, reverted edits, and unrelated changes, obscuring which changes resolve each problem. We present RETRACE, a framework for reconstructing problem-level repair experience from such histories. RETRACE combines an endpoint view that reasons backward from changes retained in the passing revision with a development view that traces repair evolution forward through commit history. CI execution evidence reconciles the two views, and the recovered experience is represented at three abstraction levels, from concrete fixes to transferable repair patterns. For new failures, RETRACE retrieves relevant problem-level experience to guide repair. On CI-REPAIR-BENCH, comprising 565 PR-level repairs from 101 repositories across 12 failure categories, RETRACE improves mini-SWE-agent Pass@1 from 19.6% to 31.9% with MiniMax-M2.5 and from 23.3% to 32.8% with DeepSeek-V4-Flash. On a matched subset, Codex improves from 15.5% to 27.5%. Combining both views consistently outperforms either alone, showing that recovering problem-change alignment enables historical CI repairs to serve as reusable repair experience. 

---
# GS-Codec: A Gaussian-Splatting Bottleneck for Neural Audio Coding 

**Authors**: Ron Aluf, Alon Canfi, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2610.04651)  

**Abstract**: Neural audio codecs compress waveforms into compact discrete tokens that underpin speech language models, real-time communication, and large-scale audio storage. Almost every dominant design, including residual vector quantization, finite scalar quantization, and single-codebook variants, follows the VQ-VAE template by partitioning the encoder latent through a learned codebook or a fixed scalar grid. We ask whether this partition is necessary. We introduce GS-Codec, a neural speech codec whose bottleneck is a parametric signal decomposition rather than a quantizer. We adapt Gaussian splatting from 3D scene reconstruction to one-dimensional latents. An inner optimization loop fits each encoder segment as a weighted sum of 1D Gaussian primitives. The decoder then reconstructs the waveform from the rendered sum. To avoid the cost of this iterative inner loop at inference time, we additionally train a lightweight GS Predictor Net that regresses the primitive parameters in a single forward pass. The encoder and decoder are trained end-to-end through the inner loop, with no quantizer anywhere in the training pipeline: the bottleneck is the decomposition itself, and scalar quantization is applied only post-training to the fitted parameters. Rather than relying on discrete codebook stages for bitrate control, our representation exposes a fine-grained rate-quality tradeoff: a single trained checkpoint supports post-training bitrate control by varying the number of primitives and the per-parameter bit depth, with no retraining required. GS-Codec matches or exceeds well-established open-source codecs such as EnCodec and DAC on speaker similarity (SIM), intelligibility (STOI), and perceptual quality (UTMOS) at comparable bitrates, while achieving comparable semantic performance (WER). Code and audio samples are available at this https URL 

---
# Policy as Data: Replay-Based Policy Dual Averaging via Advantage Regression 

**Authors**: Nianli Peng, Geoffrey J. Gordon, Kianté Brantley  

**Link**: [PDF](https://arxiv.org/pdf/2610.04638)  

**Abstract**: Actor-critic methods reuse past experience to improve sample efficiency. However, historical data are typically regarded as off-policy samples for the current policy-improvement update. This work introduces Regularized Dual Averaging Actor Critic (RDA2C), which assigns a distinct role to replay. In regularized dual averaging, the subsequent policy is determined by accumulated policy-improvement feedback, so historical advantage estimates contribute directly to the actor objective rather than solely to the most recent update. RDA2C stores state-action samples with critic-estimated advantage labels, fits a dual score model $Z_\theta$ to the aggregated dataset, and derives the current policy from the accumulated score model using the entropy mirror map. In this way, replay defines an empirical dual objective from which the policy is computed. To analyze RDA2C, we establish a finite-time value-gap decomposition, separating the regularized dual-averaging term from errors due to stale-replay supervised fitting, critic bias, finite-buffer variance, and replay coverage, and stating the assumptions under which each error is bounded. RDA2C accepts advantage labels from any critic. With GAE labels, RDA2C outperforms PPO on six of eight MuJoCo tasks and eight of twelve Atari games. RDA2C also outperforms AAPDA, the closest dual-averaging baseline, on six of eight MuJoCo tasks. With twin-$Q$ labels, RDA2C matches SAC at matched batch size and update frequency. 

---
# ConEx: Human-Interpretable Saliency Maps via Concept-Aware Attribution 

**Authors**: Yehonatan Elisha, Oren Barkan, Ziv Weiss Haddad, Noam Koenigstein  

**Link**: [PDF](https://arxiv.org/pdf/2610.04605)  

**Abstract**: Many visual explanation methods in computer vision highlight pixel importance but struggle to link these low-level cues to semantically meaningful concepts, limiting their interpretability and trustworthiness. We introduce Concept-based Explanations (ConEx), a novel framework that bridges saliency visualization with concept-based reasoning to provide both faithfulness and interpretability. ConEx automatically discovers class-specific concepts and represents them through concept activation vectors (CAVs), learned without manual supervision using an architecture-specific masking mechanism that reduces noise introduced by the segmentation masks to enhance concept purity. ConEx generates faithful saliency maps that reveal where each concept appears in the image and how it contributes to the prediction. To evaluate the reliability of these learned concepts, we propose two complementary metrics, Vector-Concept Match (VCM) and Concept-Class Match (CCM), that quantify concept alignment and enable direct comparison with existing methods. Extensive experiments across diverse settings demonstrate that ConEx achieves state-of-the-art performance on faithfulness, segmentation, and concept-quality benchmarks. Overall, ConEx advances the field toward truly interpretable and concept-grounded explanations in vision models. 

---
# Variational Quantum Attention for Molecular Graph Learning 

**Authors**: Yu-Cheng Lin, Yu-Chao Hsu, Tai-Yue Li, Nan-Yow Chen, Samuel Yen-Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.04588)  

**Abstract**: Molecular property prediction is central to computational drug discovery, where graph neural networks learn to weight neighboring atomic environments during message passing. Yet it remains unclear how variational quantum circuits alter learned attention behavior in molecular graphs. We introduce an edge-aware variational quantum attention mechanism for molecular graph learning, in which the receiving atom, neighboring atom, and connecting bond jointly determine the quantum attention state. Across five molecular property and bioactivity prediction tasks, QGAT achieves competitive performance relative to GATv2, with a consistent improvement on BBBP across all six evaluated circuit ansatzes. We further compare how the quantum and classical attention scores weight molecular structure beyond accuracy. In the Verubecestat BACE1 inhibitor series, QGAT achieves a higher Spearman correlation than GATv2 and assigns positive attributions to several structural changes consistent with reported structure-activity relationships (SARs). This case study shows that the two attention mechanisms can exhibit different prediction and attribution behavior across structurally related BACE1 analogues, while broader validation is required to determine how consistently these differences generalize across chemical series and targets. Circuit ablations further show that performance depends on the circuit design. Together, these results show that variational quantum attention can serve as a viable alternative molecular attention parameterization while inducing circuit- and chemistry-dependent behavior distinct from a matched classical scorer. 

---
# From Probe Scores to Alarm Policies: Operational Validity of Activation Monitors for Language-Model Agents 

**Authors**: Xueping Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.04575)  

**Abstract**: Activation probes can predict safety-relevant properties of language models with high area under the receiver-operating-characteristic curve (AUROC), but deployed agent monitors make thresholded alarm decisions under tight false-alarm budgets. These are different estimands. We introduce an Operational Validity Contract that fixes a monitor's target, observability, identity, timing, intervention unit, comparator, calibration, and cost. We formalize risk at the semantic request or trajectory level: when one task contains repeated alarm opportunities, row-level AUROC and false positive rate do not identify semantic-unit any-alarm risk. A confidence-certified threshold also requires enough independent negative units, a tie-safe rule, and transport to deployment. Across Models Under Pressure, LASR refusal prediction, immutable AgentDojo, and a prospectively protocol-frozen ST-WebAgentBench replication, joint activation-observable monitors reach AUROC 0.957 and 0.935 on the first two benchmarks, yet their locked 5%/10% detection rates are only .642/.742 and .719/.782, respectively. On AgentDojo, the secondary mean-activation rollout monitor reaches AUROC 0.922 but detects none of 38 positive semantic cases at the locked 5% operating point; thresholds intended for 10% false alarms realize 18.7-20.0% on test. Because the test misses its prospectively frozen 40-positive support gate, we label it support-insufficient. On ST-WebAgentBench, activation reaches AUROC .874, but 23 independent calibration negatives cannot identify even a 10% controller; the locked policy abstains rather than reporting its mechanical zero FPR as a success. An exploratory counterexample also lowers full AUROC while improving realized 10% utility. The fail-closed compiler caps MUP and LASR at restricted predictive value and AgentDojo and ST-Web at representation accessibility; no setting reaches alarm-policy validity. 

---
# Only Project Once: Projection-Adaptive Loss for Exact Constraint Satisfaction 

**Authors**: Tim Aebersold, Soheyl Massoudi, Mark Fuge  

**Link**: [PDF](https://arxiv.org/pdf/2610.04572)  

**Abstract**: Precise constraint satisfaction is a prerequisite to deploying learned models in many areas, motivating methods that repair raw neural predictions with a repair procedure. Current methods unroll multiple repair steps in training and softly penalize constraint violations that remain after the unroll. This is compute- and memory-intensive, lacks robustness when the repair fails to converge, and surrenders most of the constraint satisfaction work to the repair. Our central finding is that, contrary to common practice, a single detached projection step suffices in training. We accomplish this with a Projection-Adaptive Loss (PAL), which uses the constraint residual after this single step to adaptively weigh constraint penalties on the raw prediction. In experiments, PAL is the only method that retains virtually perfect feasibility on extremely nonlinear constraints, and matches or outperforms current methods on synthetic and engineering benchmarks. Because it only requires a single detached projection step, PAL trains 2.5x faster than the canonical repair-based method (DC3) on its own ACOPF benchmark. PAL can also be trained when constraints are expensive to evaluate (e.g., via neural surrogates), a setting where current unrolled methods are memory-intractable. 

---
# Coupling Noisy Pairwise Knowledge to the DAG Posterior for Causal Discovery 

**Authors**: Guoliang Xu, James E Corter  

**Link**: [PDF](https://arxiv.org/pdf/2610.04559)  

**Abstract**: External causal reports can improve structure learning from limited observations, but their reliability varies across sources and variable pairs. We introduce HB-NoisyKG, a Bayesian framework that combines observational data with repeated causal reports from sources such as large language models. Each report is a noisy observation of a direct pair state implied by one DAG. A feature-conditioned Beta prior pools information about pair reliability, and a shared error matrix captures systematic mistakes. Alternating inference uses the graph posterior to refine reliability estimates, which determine how reports influence subsequent graph updates. The report likelihood uses only graph pair-state marginals, so the same observation layer supports discrete and continuous likelihoods in graph-only and joint inference. Against an 80-restart no-KG baseline, HB uses at most 80 total restarts and lowers mean SHD from 22.39 to 16.06 on five discrete benchmarks. On a physical light tunnel with random variable IDs and retained descriptions, HB lowers SHD from 39.00 for no-KG to 27.30. On continuous Sachs, graph-only BGe raises AUROC by 0.121 over no-KG Top-K. In a controlled synthetic study, continued updating also reduces mean reliability estimation error and held-out report log loss compared with one-time estimation. 

---
# Cross-Modal Solar Image Synthesis: Adapting the Surya Foundation Model from He I 10830 Å to EUV Translation and Coronal Hole Segmentation 

**Authors**: Marco Marena, Andrés Muñoz Jaramillo, Qin Li, Haodi Jiang, Jinghao Cao, Wen He, Ziyang Zhang, Chenxi Yuan, Chao Wang, Haimin Wang, Bo Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.04553)  

**Abstract**: The long observational record of He I 10830 Å offers a means to investigate solar morphology before modern extreme-ultraviolet (EUV) imaging. We adapt the Surya solar foundation model to predict Solar Dynamics Observatory/Atmospheric Imaging Assembly (SDO/AIA) 94, 193, and 304 Å images and a coronal hole (CH) probability map from full-disk helium observations. A convolutional input adapter, low-rank backbone updates, and dedicated output decoders learn from temporally paired, geometrically registered observations, with Spatial Possibilistic Clustering Algorithm (SPOCA) catalog polygons providing CH supervision. On observations held out from downstream fine-tuning, the selected dedicated models achieve disk-restricted correlation coefficients (CCs) of 0.8196, 0.8885, and 0.8398 for the three AIA channels, respectively; the CH model achieves an intersection over union (IoU) of 0.4360. The predictions recover broad solar structure, although local agreement varies substantially by channel. An optional residual refiner addresses spatial detail, and its application on pre-SDO dates improves the correlation of synthetic AIA 304 with Solar and Heliospheric Observatory/Extreme-ultraviolet Imaging Telescope (SOHO/EIT) 304 references from 0.6511 to 0.7085. Together, these results support the feasibility of helium-conditioned EUV morphological proxies and motivate their use in historical reconstruction, within the scope of the downstream test and cross-instrument evaluation. 

---
# Weight Decay and Neuron Condensation: A Three-Stage Analysis of Two-Layer ReLU Networks 

**Authors**: Cheng Xu, Pengxiao Lin, Zhangchen Zhou, Zhi-Qin John Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04533)  

**Abstract**: Weight decay is widely used as a regularization technique in neural network training, yet its role in neuron condensation (parameter direction alignment) remains unclear. Starting from a parameter initialization in the neural tangent kernel regime, we characterize training dynamics under weight decay through three stages: rapid fitting, amplitude compression, and neuron condensation. Using a two-layer ReLU network, we analyze a residual correlation field that governs both neuron amplitudes and directions. During rapid fitting, the residual approaches a quasi-static equilibrium maintained by weight decay while the tangent kernel remains nearly unchanged. In the early stage of amplitude compression, kernel decay amplifies the residual correlation field, whose isolated attracting extrema provide candidates for condensation directions. As neuron amplitudes stabilize, we bound the drift of attracting extrema and demonstrate contraction of neuron directions around them, leading to neuron condensation. This staged analysis provides a dynamical understanding of how weight decay promotes a condensed representation, beyond reducing parameter norms. 

---
# Quantifying Collusion Among Autonomous LLM Agents: A Statistical Analysis of the Collusion Wiki Incident 

**Authors**: Shariq Murtuza  

**Link**: [PDF](https://arxiv.org/pdf/2610.04528)  

**Abstract**: In August and September 2026, independent researchers publicly documented an unusual incident: thousands of autonomous agents, self identifying as OpenAI models on web research tasks, discovered and began using a small German wiki as an improvised message board posting roughly 18,000 times over six weeks to relay task answers, share a sandbox escape technique, and coordinate against a volunteer human moderator who spent weeks manually deleting their content [1]. The investigators' public writeup is a careful qualitative account, rich with direct quotation, but does not attempt a statistically rigorous quantitative characterization of the behaviour it documents. 

---
# Diffusion-Based Stress Testing of Overload Monitoring for Resilient Emergency Cellular Networks Using Internet CDR Proxies 

**Authors**: Bilal Hussain, Xiao Tang, Tan Li, Muhammad Azhar, Danista Khan, Fawad Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2610.04526)  

**Abstract**: Disasters can overload cellular control-plane signaling within minutes, yet fine-grained Radio Resource Control (RRC) or Next Generation (NG) Application Protocol (NGAP) telemetry is privacy-sensitive and costly to collect for analytics. Many emergency monitoring pipelines therefore rely on coarse Call Detail Record (CDR) aggregates. We treat Internet activity in CDR grids as a practical proxy for hidden signaling stress under that constraint. We train a lightweight convolutional neural network (CNN) on stylized overload injections, stress-test it with diffusion-synthesized surges that preserve normal traffic structure, and adapt the detector by retraining on hard synthetic samples. Under stress-test conditions, the default alert threshold fails even though receiver operating characteristic (ROC) curves stay strong: the detector still assigns overloaded cells a larger overload probability than normal cells, but those probabilities fall below the default cutoff 0.5 and are labeled normal, so the F1-maximizing threshold -- selected post hoc on the same stress-test grids (oracle $\tau^*$) -- shifts by $0.32 \pm 0.03$ (operating-point drift). Across three random seeds, hard-sample adaptation raises thresholded performance (F1) from 0% (no alerts at the default cutoff 0.5 on any seed) to $85.67 \pm 14.37$% and ranking from ROC-AUC $0.886 \pm 0.040$ to $0.99996 \pm 0.00007$. Diffusion-synthesized surges expose threshold fragility that matched-condition training -- training and testing on the same stylized injections -- hides, and hard-sample adaptation restores usable alerts at the default cutoff. Together, these steps define a reusable pre-deployment stress test for emergency monitors. Internet-only CDR input further supports lightweight AI-native workflows that combine monitoring, recalibration, and adaptation. 

---
# EgoExo-Next:Benchmarking Vision-Language Models on Visual-Option Next-State and Cross-View Reasoning 

**Authors**: Yutong Li, Molin Wang, Xiaotong Li, Yanyan Fang, Daoguo Dong, Ziyi Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.04506)  

**Abstract**: Vision-language models (VLMs) are increasingly evaluated for egocentric and cross-view video reasoning, yet existing benchmarks largely focus on semantic event understanding, temporal relations, or correspondence between already observed views, leaving their ability to reason directly about future visual states underexplored. We introduce EgoExo-Next, a visual-option benchmark for dynamic visual-state reasoning, where models must identify how an observed action trajectory subsequently appears rather than predict only an action label or textual description. EgoExo-Next contains 2,503 human-curated four-choice questions from six public egocentric and ego--exo video sources and comprises four interconnected subtasks that evaluate egocentric next-state prediction, bidirectional ego--exo state correspondence, exocentric next-state prediction, and their composition in Ego-to-Exo Next-State. Extensive evaluation of proprietary, open-source, and spatial reasoning VLMs reveals a substantial human--model gap, with the best model achieving 43.81\% average accuracy compared with 98.55\% for humans, and the largest degradation occurring on the composed Ego-to-Exo task. These results suggest that current VLMs remain substantially limited in dynamic visual-state reasoning, particularly when temporal progression and cross-view reasoning must be composed. The benchmark is publicly available at \url{this https URL}. 

---
# Localization Lens for Improving Medical Vision-Language Models 

**Authors**: Hasan Farooq, Murtaza Taj, Mehwish Nasim, Arif Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2610.04502)  

**Abstract**: Medical Vision-Language Models (Med-VLMs) have demonstrated strong capabilities in clinical tasks. However, they often struggle to understand anatomical structures and spatial positioning, which are crucial for medical reasoning. To address this, we propose a localization-aware enhancement to the Med-VLM pipeline, introducing improvements at three levels: data,architecture, and alignment. First, we introduce localization lens, a set of expert-validated representations that provide richer anatomical and positional context. However, as these representations increase input complexity, we integrate pixel shuffle within the model architecture to filter and refine representations, enhancing spatial information processing while preserving anatomical continuity. Lastly, to effectively align the localization lens representations with textual features, we incorporate decoupled contrastive loss (DCL) alongside the standard loss function. This ensures better feature discrimination and robustness, particularly in data limited medical settings. Through extensive evaluations on medical visual question answering (Med-VQA) datasets, we show that our methodology improves localization-driven performance across different Med-VLM architectures. Our analysis of localization-based questions further reveals that improvements in anatomy and spatial reasoning directly enhance the overall accuracy of Med-VQA upto 6.2%. The proposed approach is model-agnostic and can be seamlessly integrated into existing Med-VLM pipelines. The dataset, code, and trained models will be made publicly available at this https URL. 

---
# Understanding Clustering in Slot Attention via Particle Dynamics 

**Authors**: Vasudev Joy, Rajat Rasal, Avinash Kori, Anthea Monod, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2610.04493)  

**Abstract**: Studying attention through the lens of interacting particle dynamics has shown how token clustering can emerge from the underlying dynamics. We extend this perspective to slot attention, a method for object-centric image segmentation and representation learning in which learned components obscure how much of the clustering behaviour is intrinsic to the attention dynamics. We therefore introduce simplified slot attention (SSA), a parameter-free variant whose dynamics are connected to soft $k$-means clustering and which provides a straightforward mechanistic explanation for the emergence of object-centric representations. On the Pascal VOC dataset, SSA achieves performance comparable to that of slot attention, demonstrating that competitive object-centric segmentation can be achieved without learned neural-network components. 

---
# Can LLM Agents Automate Reinforcement Learning for Text-to-Speech? 

**Authors**: Xuanjun Chen, Zixiong Su, Hao Shi, Chang Zeng, Kai Li, Jyh-Shing Roger Jang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.04488)  

**Abstract**: Although reinforcement learning (RL) post-training repairs the localized segmental errors of zero-shot text-to-speech (TTS), arriving at a working recipe still relies on tedious manual tuning, and whether LLM agents can take over this research pipeline is unclear. We investigate this question with AgenticTTS-Forge, a collaborative workflow that structures human guidance and agentic execution around a shared workspace, applied to CosyVoice2-0.5B. To measure what the agent automates, we audit its trajectory stage by stage against the published recipe. To measure what it exploits, we score its policies with held-out observers hidden from the agent. Our results show that the agent recovers an underspecified recipe, improves it, and, when gains stall, surveys the literature unprompted and pivots from the LM carrier to the flow carrier, halving Bad cases. However, its autonomy exposes three traps across the data, proxy, and algorithm axes: the held-out set leaks through a channel the contract never reads, a self-shaped reward inflates the proxy where it is scored, and separately tuned policies do not compose additively. These findings show that the binding constraint is measurement rather than reasoning, and can inform the design of harnesses whose contracts read every channel the agent does. 

---
# Target-free Latent Safety Alignment 

**Authors**: Luoyu Chen, Weiqi Wang, Chenhan Zhang, Zhiyi Tian, Yuxian Huang, Shui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04467)  

**Abstract**: Large language models (LLMs) remain highly vulnerable to jailbreak attacks that induce harmful behaviors and circumvent safety alignment. To defend against such attacks, adversarial training paradigms have been proposed to first simulate failure modes and then train the model to correct them, yielding promising improvements in safety alignment. However, these methods typically construct adversarial samples either by encouraging fixed harmful target completions or by performing targeted activation ablation derived from fixed benign--harmful data pairs. As a result, the generated adversarial samples tend to induce homogeneous harmful behaviors that poorly reflect the diversity of behaviors elicited by real-world jailbreak attacks. This behavior-level narrowness fundamentally limits their robustness. To address this issue, we propose a target-free adversarial training framework that generates adversarial samples in an unsupervised manner. By amplifying and diversifying behavior-level shifts in the model's latent space, our approach produces semantically diverse adversarial samples that induce a wide range of harmful behaviors. This expanded behavioral coverage exposes more diverse failure modes and thereby improves safety alignment. To quantify this effect, we use semantic entropy as an output-level measure of adversarial behavioral diversity. Empirically, our method elicits diverse harmful behaviors in the target model, substantially mitigating behavioral narrowness and improving robustness to jailbreak attacks. 

---
# Guess My Weight: Profiled Side-Channel Recovery of Floating-Point Neural-Network Weights 

**Authors**: Timon Lumír Fillo, Ján Mikulec, Anubhab Baksi, Jakub Breier, Xiaolu Hou  

**Link**: [PDF](https://arxiv.org/pdf/2610.04436)  

**Abstract**: Neural-network parameters deployed on embedded devices may be exposed through physical side-channel leakage during inference. Existing side-channel attacks on floating-point neural-network parameters have often targeted reduced numerical precision, while recovering the complete IEEE-754 representation remains considerably more challenging because of the large and structured 32-bit candidate space. We present a profiled template attack for bit-exact recovery of an IEEE-754 single-precision neural-network weight from power measurements. The attack targets the floating-point multiplication between a known input and a first-layer weight. During profiling, multivariate Gaussian templates are learned from randomized network configurations using Hamming-weight classes of the multiplication result, while the remaining network parameters act as nuisance variables. To efficiently search the structured 32-bit floating-point candidate space, we use a hierarchical coarse-to-fine-to-exact procedure that progressively increases both the numerical and leakage-model resolution. Experiments on a ChipWhisperer-Lite with an Arm Cortex-M4 demonstrate recovery of the exact float32 representation of the target weight. In the evaluated setting, the attack reaches a bit-exact success rate of 99% with 171 traces and 100% from 263 traces onward. These results demonstrate that profiling can enable practical full-precision extraction of floating-point neural-network parameters from physical leakage. 

---
# Video2World: Benchmarking Coding Agents for Interactive World Modeling from Embodied Videos 

**Authors**: Jinzhou Tang, Zijun Zhang, Jing Yang, Yuchen Yan, Kun Zhou, Lingjun Mao, Ruobing Han, Jinglin Cao, Wenpeng Xu, Lukun He, Minghao Fu, Fan Feng, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04432)  

**Abstract**: Building interactive simulators from real-world observations is a promising way to scale embodied data, but current pipelines still rely heavily on manual environment construction and calibration. We study whether frontier foundation models and coding agents can automate this process end to end. We formulate \emph{autonomous video-to-simulation} as a software engineering task in which an agent observes an embodied video, constructs the corresponding simulated environment and robot behavior, and iteratively refines the result through execution feedback. To evaluate this capability, we introduce \textbf{Video2World}, a benchmark comprising 222 reconstruction instances derived from 189 robot and human demonstration videos. Video2World measures reconstructed worlds along geometric fidelity, dynamic fidelity, and functional correctness, capturing spatial perception, physical reasoning, and executable interaction. Evaluating 9 frontier coding-agent systems reveals a sharp improvement in Task success beginning with Claude Opus 5, rising from below 5\% to over 15\%, while substantial gaps to human-assisted reconstruction remain. We further find that worlds that look better could work worse: better visual fidelity does not always lead to higher task success. This echoes the broader gap between perceptual realism and factual correctness observed in generative models. 

---
# COSMOS: Soft Mechanism Mixtures with Verifiable Routing for Long-Horizon PDE Forecasting 

**Authors**: Anupam Rawat, Manikandan Padmanaban, Jagabondhu Hazra  

**Link**: [PDF](https://arxiv.org/pdf/2610.04427)  

**Abstract**: Neural operators offer an efficient alternative to classical PDE solvers, but most learn a monolithic map per equation family and discretization. Real systems are compositional: transport, diffusion, wave, and reaction processes can act simultaneously. Existing mixture-of-experts operators typically use sparse top-$K$ routing, although concurrent physics is naturally a blend rather than a discrete choice.
We propose COSMOS (Cooperative Operator Specialists with Mechanism-level Operator Soft-routing), a soft mechanism-mixture neural operator. Four process-biased specialists remain active at every step and are continuously mixed by a learned gate, with their features fused by a small network. Specialists share a coarse latent grid, while a zero-initialized full-resolution residual restores detail lost through the bottleneck. We also introduce an operator-splitting compositional benchmark with known mixture weights $w^\star$ per trajectory.
Against family-tuned FNO under 20-step rollouts, an initial 3-seed evaluation suggested gains on diffusion--reaction, parity on Navier--Stokes, and weaker shallow-water performance. An 11-seed audit showed that the diffusion--reaction gain was unstable, motivating caution in small-seed rollout comparisons and precluding a reliable accuracy-win claim on these families. Ablations show that uniform routing or removing the specialist mixture substantially degrades stable-regime rollouts. On the labeled benchmark, dense soft routing yields $2.2\times$ lower error than hard top-1 routing at identical fusion; using generator weights $w^\star$ at inference further lowers rollout error to $0.034$, diagnosing limitations of the learned gate. However, routing labels do not align with the specialists' intended mechanisms: COSMOS supports compositional accuracy, not mechanism identity. 

---
# AgroGround: Multi-Granularity Grounded Recognition in Agriculture 

**Authors**: Abdulla Alshehhi, Zongyan Han, Rao Anwer  

**Link**: [PDF](https://arxiv.org/pdf/2610.04425)  

**Abstract**: Agricultural visual models are typically evaluated for either recognition or localization, but reliable diagnosis requires identifying what is present and localizing the evidence. Agricultural visual question answering (VQA) datasets carry rich semantic labels but rarely link them to image regions, and adding such annotations by hand is costly at scale. We introduce AgroGround, a large-scale dataset for grounded agricultural recognition: identifying plant diseases and other agricultural targets and localizing their image regions. An automated pipeline converts the labels of eight agricultural VQA datasets into annotations for disease lesions and whole objects, producing 794,850 instruction examples. Healthy images provide negative supervision for disease queries, teaching the model to return empty predictions. We fine-tune a shared vision-language model on known-target grounding instructions combined with instructions requiring both recognition and localization. We evaluate predicted identities, regions, joint correctness, and healthy-image abstention on 1,480 human-verified images disjoint from all training data. Grounding-only fine-tuning reduces recognition accuracy from 51.8\% to 29.1\%, while adding recognition-and-localization instructions raises it to 72.6\%. With images and annotations held fixed, combining the two formats raises joint accuracy from 19.2\% to 43.3\% at comparable grounding. Healthy negatives raise abstention on healthy images to 95.0\%, and reinforcement learning improves lesion-level grounding. The resulting 2B model exceeds its annotation teacher in grounding F1 on our benchmark and on the external PlantSeg test set. AgroGround establishes a benchmark for grounded agricultural recognition, measuring joint correctness of identity and localization along with abstention on healthy images. The code is available at this https URL. 

---
# MaDeL: Manifold-Decomposed Feature Losses for Generative Modeling 

**Authors**: Beomsu Kim, Jong Chul Ye, Kwanyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.04419)  

**Abstract**: Generative models are often trained with isotropic objectives such as mean-squared error. For data concentrated near a low-dimensional manifold, however, such losses conflate displacement along the manifold, which may represent valid variation, with displacement away from it, which produces invalid samples. This mismatch is especially problematic in sparse, highly constrained domains, where ambient-space regression can encourage off-manifold interpolation. We ask whether a generative objective can distinguish manifold-parallel variation from manifold-orthogonal deviation directly from data, without explicitly estimating the manifold. We introduce a manifold-decomposed feature loss (MaDeL) that learns complementary representations from corrupted observations: one is trained to recover the clean sample, while the other is trained to recover the corruption. We show that, under a feature bottleneck, their Jacobians align with the tangent and normal spaces, exactly for linear manifolds and locally for smooth manifolds. Together, these representations define an anisotropic objective that separately measures intrinsic variation and off-manifold deviation. Across synthetic, Earth and climate science, and torsion-angle benchmarks, MaDeL improves support recovery and average angular $W_1$ under single-step sampling; on protein backbones, it reduces steric clashes across one- and few-step sampling budgets. 

---
# Maximizing $p$-Mean Social Welfare in the High-Multiplicity Setting: Few Agent Types and Few Item Types 

**Authors**: Trung Thanh Nguyen, Khaled Elbassioni  

**Link**: [PDF](https://arxiv.org/pdf/2610.04417)  

**Abstract**: The $p$-mean welfare objective unifies several classical social welfare criteria for the allocation of indivisible goods. We study its maximization under additive nonnegative utilities when item and agent multiplicities are encoded in binary. For every fixed finite rational $p<1$, $p\neq0$, we show that the problem is $\mathsf{NP}$-hard with only two item types. Nash welfare ($p=0$) and egalitarian welfare ($p=-\infty$) are NP-hard with three item types. These results hold both for computing an optimal allocation and for rational-threshold decision, and the utilities and threshold can be required to be positive integers. We also show that maximizing $p$-mean social welfare is strongly NP-hard with one agent type and an unrestricted number of item types, for every fixed finite rational $p<1$ and for $p=-\infty$. A quantitative gap in this reduction rules out an FPTAS in the latter setting unless $\mathsf{P}=\mathsf{NP}$. On the positive side, for a fixed number of item types and an arbitrary number of agent types, we give an FPTAS for every fixed $p\in\mathbb Q\cup\{-\infty\}$. Its running time is polynomial in the compact input length and in $1/\varepsilon$, and it returns a compressed allocation. For a fixed number of agent types and an unrestricted number of item types, we give a PTAS for every fixed finite rational $p<1$, also in the fully compact model. We further give explicit compact-model proofs of the classical exact allocation algorithms for one item type, and for egalitarian welfare with two item types. These results essentially settle the complexity and approximability of $p$-mean welfare maximization with few item types and/or few agent types, leaving only the exact complexity of Nash welfare maximization with two item types unresolved in the small-item-type classification. 

---
# Specific Algorithmic Interpretability of Neural Networks: A Case Study on Textures 

**Authors**: Yanglin Zhang, Anneke von Seeger, Gilad Lerman, Ron Levie  

**Link**: [PDF](https://arxiv.org/pdf/2610.04413)  

**Abstract**: We develop a principled framework for constructing neural networks whose specific parameter realizations admit an explicit algorithmic interpretation. Existing algorithm-inspired architectures can explain the computational structure of a network, yet after standard training the learned parameters need not retain a clear relation to the motivating algorithm. We address this gap as follows. First, we model each data point as a sample of a class-dependent stochastic process and assume that statistics of this process can be estimated from a single sample and these statistics are sufficient to distinguish the classes. We then construct a neural network whose initial parameters exactly implement an algorithm for estimating these statistics, making the network fully interpretable. To account for mismatch between the idealized model and real data, we fine-tune this network while controlling its deviation from the algorithmic initialization. The trained network hence roughly retains the interpretation of the initial network. A PAC-Bayesian analysis yields a uniform generalization bound whose complexity term scales with the fine-tuning radius, providing a statistical motivation for our approach. We instantiate the framework for texture classification using the scattering transform to estimate the discriminative statistics. 

---
# Understanding and Mitigating Hallucination Escape in Tool-Using LLM Agents 

**Authors**: Peigui Qi, Kunsheng Tang, Yide Song, Weiming Zhang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04409)  

**Abstract**: Large language models (LLMs) increasingly serve as autonomous agents that invoke external tools. However, this capability introduces tool hallucination, selecting incorrect tools or generating invalid calls. Existing mitigation methods report substantial improvements, yet we identify a previously overlooked failure mode that we term Hallucination Escape. These methods reduce hallucination on the tool configuration they are tuned on but increase it on other configurations, canceling out the gain. We further investigate this phenomenon and find that hallucination rises sharply when a model's intrinsic tool-use tendencies conflict with the current tool configuration, and that existing methods reinforce rather than suppress these tendencies, which in turn contributes to hallucination escape. Building on these findings, we propose EscapeGuard, a training-free inference-time method that combines conflict-aware gating with configuration-derived attention enhancement to mitigate tool hallucination while preventing hallucination escape. Across six benchmarks on various models, EscapeGuard reduces tool-selection hallucination by 9.0 pp and suppresses hallucination escape, lowering the cross-configuration mean by 23.7 pp and achieving an 89.1% net improvement in paired-query evaluation. We hope this work can encourage evaluation beyond a single tool configuration and pave the way for more reliable tool-using LLM agents. 

---
# XTurnix: Large-Scale Self-Supervised Turn Control through Two-State Binary Decisions 

**Authors**: Zhanxun Liu, Yifan Duan, Hengtao Wu, Chen Yang, Qinyuan Cheng, Kun Wang, Xingyu Zeng, Xipeng Qiu, Chaochao Lu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.04400)  

**Abstract**: General turn-taking behavior in real-time dialogue systems requires deciding whether to keep listening or start responding while listening, and whether to continue or stop while speaking. Existing turn detectors use heterogeneous, task-specific label spaces and are often trained on limited annotations or evaluated on isolated utterances, making them difficult to use as a unified causal controller with comprehensive context. We propose XTurnix, a compact text-based model that formulates turn control as two binary decisions conditioned on the AI's current listening or speaking state and predicts a single control token from the complete dialogue history. XTurnix is pretrained on 5.5 million causal action examples automatically derived from timestamped two-speaker transcripts, then fine-tuned on synthetic multi-turn examples with a flatter distribution across the four state-action labels. We evaluate XTurnix on four public benchmarks and a balanced self-curated benchmark. Across the public benchmarks, XTurnix achieves the best results on all SemanticVAD and LiveKit splits, ties the native Smart-Turn model on Smart-Turn Bench, and achieves the highest incomplete-turn accuracy on Easy-Turn. On the self-curated benchmark, it reaches 89.06% accuracy, more than 20 percentage points above the strongest third-party baseline at 68.75%, while maintaining F1 scores between 84.21% and 90.63% across all four categories. These results demonstrate unified listening- and speaking-state turn control in a single compact model. Code is available at this https URL, with an interactive demo at this https URL. 

---
# GlitchPatch: Repairing Glitch Tokens in Frozen Language Models via Local Retokenization 

**Authors**: Kunsheng Tang, Peigui Qi, Yide Song, Peijun Huang, Weiming Zhang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04399)  

**Abstract**: Glitch tokens are anomalous vocabulary entries that can cause large language models (LLMs) to produce outputs inconsistent with their inputs. Existing repair methods require access to model internals, making them impractical for frozen checkpoints. We investigate whether glitch tokens can be repaired outside the model by optimizing the input tokenization. An empirical study on BPE merge-rule deletion reveals that (1)deleting a glitch token's merge rule can fix a substantial fraction of failures, yet disrupting normal tokens sharing intermediate merge nodes causes the overall glitch rate to rise, and (2)different decomposition granularities yield non-monotonic fix rates while collateral damage on normal tokens grows monotonically. Motivated by these findings, we propose GlitchPatch, a repair framework for frozen language models based on local retokenization, consisting of two stages: the offline stage uses Behavioral Path Optimization (BPO) to find the behaviorally optimal replacement token sequence for each glitch token and compiles validated replacements into a rule table; the online stage substitutes only the IDs of matched glitch tokens in the canonical token sequence, with no modification to model parameters or internal states. Experiments on ten models spanning six tokenizer families show that GlitchPatch achieves an 85.10% mean fix rate, outperforming the strongest baseline by 14.37 percentage points, and reduces the average glitch rate from 14.88% to 2.27%. GlitchPatch achieves a 0.00% RR in full-vocabulary evaluation and leaves rule-unmatched inputs unchanged by design. We further evaluate the practical impact of repair from the perspectives of time cost, language understanding, and capability, supporting its deployment feasibility. We hope this work provides a practical option for improving tokenizer reliability. 

---
# COPEX: Benchmarking LLM Robustness to Adversarial Context Across Model Context Protocol Layers 

**Authors**: Nahom Birhan, Mehrdad Rostamzadeh, Sidhant Narula, Mahmoud Nazzal, Mohammad Ghasemigol, Daniel Takabi  

**Link**: [PDF](https://arxiv.org/pdf/2610.04378)  

**Abstract**: Large language models increasingly mediate tool use in Model Context Protocol (MCP) systems, where adversarial influence may enter through user instructions, tool schemas, tool outputs, or protocol messages. Existing benchmarks often evaluate deployed agents, conflating model susceptibility with guardrails, orchestration, and general task capability. We introduce COPEX (COntext Provider EXploitation), a controlled benchmark that isolates the model as an MCP client by fixing the surrounding agent stack and varying only the tool-selecting model. COPEX covers 25 attack types instantiated as 125 scenarios across four entry surfaces: model/agent, client, server/tool, and transport. Across nine models and 3,375 trials, the mean attack success rate is 64.4%, with surface-level means ranging from 58.3% to 71.4%. Some client- and transport-level attacks succeed partly outside the model's observation or control, separating system exposure from model susceptibility. Combined input and context scanning reduces mean attack success by 49.6% on an eight-attack defense subset relative to the undefended setting. The benchmark is available at this https URL. 

---
# Human Behavior-Informed Crash Scenario Generation with Real-World Crash Priors for Autonomous Vehicle Safety Evaluation 

**Authors**: Mingxing Peng, Xusen Guo, Long Chen, Xintao Yan, Siyu Teng, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.04366)  

**Abstract**: Reliable safety evaluation of autonomous vehicles (AVs) is essential to improving road safety, yet it depends critically on realistic simulation of rare crashes. Existing crash scenario generation methods can increase collision occurrence, but often fail to realistically reproduce how crashes evolve before impact or the distribution of crash types observed in the real world. Here, we present CrashSim, a human behavior-informed crash scenario generation framework that uses real-world crash priors to guide generative multi-agent traffic simulation for more reliable AV safety evaluation. These priors capture how real-world crashes evolve before impact and how different crash types are distributed, allowing limited crash data to guide realistic and scalable scenario generation across naturalistic driving contexts. We evaluate CrashSim against competing methods, showing that it more closely reproduces real-world pre-impact behavior, collision dynamics, collision geometry and crash-type distributions. We further use CrashSim to construct nuCrash dataset, containing over 4,000 crash and near-crash scenarios. Closed-loop evaluation of five AV planners shows that nuCrash more effectively exposes differences in planner safety capabilities than nuScenes. An LLM-assisted evaluation agent further analyzes planner failures to provide capability-level diagnoses and targeted improvement guidance. Together, CrashSim enables realistic and scalable crash generation for more informative AV safety evaluation. 

---
# A Bird's-Eye View of Iterative Reward Design 

**Authors**: Logan Mondal Bhamidipaty, Lauren Robson, Linda Petrini, Shengrui Lyu, Kamal Ndousse  

**Link**: [PDF](https://arxiv.org/pdf/2610.04364)  

**Abstract**: Designing effective reward functions in RL typically requires substantial expertise and trial and error. Recent work automates this process with LLM-based systems that generate and iteratively improve reward code using policy feedback. However, these methods are often hard to compare because they differ in implementation details, feedback assumptions, and evaluation environments. To address this, we introduce a Benchmark for Iterative Reward Design (BIRD) that expresses existing methods in a unified configuration and evaluation space. This lets us compare algorithms directly, ablate individual design choices, and prototype new components under matched feedback conditions and policy-training budgets. Across MuJoCo, Meta-World, Assistax, and HumanoidBench, we identify a small set of simple design choices that consistently improve performance. Combining these choices yields significantly better performance than the evaluated methods from prior work. Our results highlight the strength of simple baselines and motivate further study of when additional algorithmic complexity improves iterative reward design. Code is available at this https URL. 

---
# System One Models for Wireless Decision-Making:Applications and Performance Evaluation 

**Authors**: Masoud Rahimi, S. M. Matin Alemohammad, Hamid Behroozi, Mahdi Nouri  

**Link**: [PDF](https://arxiv.org/pdf/2610.04345)  

**Abstract**: Many wireless control tasks require repeated selection of a single action from a finite feasible set under stringent latency and reliability constraints. While large language models (LLMs) have recently emerged as general-purpose decision engines, their autoregressive generation mechanism is not naturally aligned with such bounded control problems. This paper investigates System-One models, which directly learn probability distributions over explicitly defined decision spaces, as a lightweight alternative for wireless decision-making. We formalize their decision structure and learning objective, identify their applicability across physical-layer control, radio resource management, mobility, network slicing, and network operations, and evaluate their practical behavior through representative wireless case studies. Using Jev as a System-One implementation, we benchmark decision quality and client-observed latency against generative LLMs and conventional baselines. In receive-antenna selection, Jev delivers up to an 8.5x reduction in median response latency relative to the evaluated LLMs, although this gain comes with a loss in decision quality compared with stronger task-specific alternatives. More notably, in intent-conditioned RAN slicing, Jev achieves utility comparable to the evaluated LLMs while providing more than a 3.5x reduction in median response latency. Complementary evidence from edge-service orchestration further shows that faster decisions do not necessarily translate into lower end-to-end service latency. These results expose a fundamental quality/latency tradeoff and position System-One models not as replacements for numerical optimization, but as a promising decision interface for latency-sensitive, bounded, and intent-driven wireless control. 

---
# GrayShield: Bit-Level Sanitization for Transformer Model Supply-Chain Security 

**Authors**: Armstrong Foundjem, Tsung-Hsien Chuang, Foutse Khomh, Mohamed Amine Merzouk  

**Link**: [PDF](https://arxiv.org/pdf/2610.04319)  

**Abstract**: Transformer models such as BERT and Vision Transformer~(ViT) achieve strong performance via densely parameterized attention backbones. However, the least significant bits~(LSBs) of their 32-bit floating-point weights can be abused as covert channels to conceal malicious payloads, posing a serious threat to the AI model supply chain. We propose \GS (\GSabbr), a lightweight, post-training, zero-data sanitization method that completely replaces the declared mantissa-LSB channel with a Gray-code-guided low-transition sequence. Complete payload-independent overwrite, whether keyed or public, makes the sanitized target bits independent of the embedded payload and gives that declared channel zero capacity. Gray coding supplies overwrite structure, while a keyed per-tensor phase supplies pattern diversity. Benchmarked against seven post-training defenses on four Transformer model presets and two real-world malware payloads, \GSabbr maintains sub-$1\%$ accuracy impact and achieves $49.96\pm0.66$ percentage-point Recovery Reduction (RR) under five implemented attacker variants. Because pre-defense recovery is effectively $100\%$, RR near 50 percentage points corresponds to post-sanitization bit accuracy at binary chance. Its main empirical advantage is stable near-chance sanitization with substantially smaller weight-distribution shift than the evaluated near-chance baselines PatternMask (PM) and Post-Training Quantization (PTQ). 

---
# What to Preserve in Recursive Computation: A Local Predictive Sufficiency Principle 

**Authors**: Peilin Wang, Feng Shiyang, Hongfu Gao, Cencheng Zhao, Di Yuan, Hui Chen, Guiguang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2610.04303)  

**Abstract**: Recursive computation repeatedly compresses or reuses intermediate states, creating a simple tension: information that must remain useful across longer recursive paths is also exposed to more opportunities for loss before reaching the final prediction. Existing reconstruction or local-prediction objectives provide tractable supervision, but do not ensure that the retained information remains sufficient for subsequent recursive computation. We identify local predictive sufficiency with recursive predictive closure: controlling local predictive deficiencies at individual interfaces controls the resulting discrepancy at the root. We then turn this principle into a tractable training procedure. Starting from a variational characterization, we derive finite predictive tests and an empirical predictive deficiency that measures predictive value retained across compression. Its predictive sensitivities define margin-relaxed half-space constraints on parameter updates, and we project the host optimizer's proposed update onto their intersection only when predictive preservation would otherwise be violated. Across temporal graphs, language memory, vision-language-action control, and recursive self-improvement, the method matches or improves the corresponding host models under matched compression budgets, with larger gains under heavier recursive or memory demands, while better preserving predictive information across successive transformations. Crucially, the same task-agnostic predictive-preservation principle is instantiated across all four settings through host-compatible interventions while keeping the endpoint task, backbone, and evaluation protocol fixed. These results establish predictive preservation at recursive interfaces as a general training principle for recursive compression. 

---
# Questioning the Questions: Sustaining Self-Evolution in Reasoning Models 

**Authors**: Jinyuan Li, Chengsong Huang, Langlin Huang, Donghong Cai, Shiping Gao, Yuyi Yang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04299)  

**Abstract**: Self-evolving reasoning models learn from their own generated questions, yet repeated self-training can lead to performance collapse. In this paper, we investigate why performance deteriorates over successive rounds and how to sustain self-evolution. Our analysis identifies two recurring quality problems in self-generated questions: invalid questions and repeated variants of the same mathematical questions. First, invalid questions become more prevalent across rounds, and answer-consistency filtering further increases their proportion in training data. Second, existing question diversity controls based on lexical similarity can miss mathematically equivalent questions expressed in different ways, which leads to question diversity collapse in later training rounds. Building on these findings, we introduce R-Quest, which uses question validity and novelty feedback to guide self-evolution. We first train the solver to recognize and reject invalid questions, then use its judgments to guide questioner rewards and filter solver training data. To avoid question repetition, we use a frozen base model to compare sampled question pairs and provide novelty feedback. Empirically, our method consistently achieves the highest average performance on 12 benchmarks in mathematical reasoning, general-domain reasoning, and code generation across two model families. Additionally, R-Quest maintains stable performance gains over ten rounds of self-evolution, peaking in the final round and outperforming R-Zero by 17.32 points. 

---
# RRM-GPT: A Framework and Vision for Radio Resource Management Foundation Models 

**Authors**: Ahmed Aboulfotouh, Akram Bin Sediq, Koosha Pourtahmasi Roshandeh, Omar Mashaal, Ahmad M. Nagib, Jale Sadreddini, Hatem Abou-Zeid  

**Link**: [PDF](https://arxiv.org/pdf/2610.04296)  

**Abstract**: Learning-based models for radio resource management (RRM) are typically built for a single function and deployment, so each new setting repeats the development pipeline. RRM decisions, however, share a common structure: each is assembled from interdependent fields, defined by the standard, whose values are selected in view of the network state. We propose RRM-GPT, an autoregressive framework for RRM foundation models that generate these decisions as a language model generates text. An encoder maps heterogeneous network observations into a common token representation, and a decoder emits the decision one field at a time, each conditioned on the network state and the fields already committed. Pretraining on unannotated network logs teaches the model what makes a decision valid and how controllers choose among valid decisions; post-training then adapts it to deployment-specific operator objectives through imitation or reinforcement learning. The framework targets two forms of reuse: a function-specific model reused across deployments, and a model shared across RRM functions that generates their interdependent decisions as one sequence. In a 5G New Radio (NR) case study, we demonstrate that a single model generates complete scheduling grants spanning user selection, timing, link adaptation, resource allocation, and control signaling. The model captures dependencies among grant fields and transfers learned behavior to an unseen scenario without adaptation. 

---
# Scaling of Wireless Foundation Models via Representation Diversity and Multi-Branch Architectures 

**Authors**: Ahmed Mohamed, Ahmed Aboulfotouh, Hatem Abou-Zeid  

**Link**: [PDF](https://arxiv.org/pdf/2610.04289)  

**Abstract**: Wireless foundation models learn representations from unlabeled radio signals for reuse across downstream tasks. Scaling model capacity is a common strategy for learning richer representations and improving downstream performance. However, its gains are less consistent in wireless self-supervised learning when pretraining data are limited. We investigate objective diversity as an alternative scaling axis: different self-supervised objectives emphasize different signal properties, and combining their representations can preserve more information to enable diverse tasks. We develop a fusion framework that combines frozen representations from independent encoders trained through reconstructive, predictive, and contrastive learning. This provides a reference for the benefits of diversity, but requires the maintenance of multiple encoders. To retain these benefits within the parameter budget of a standard single-objective encoder, we introduce a jointly trained multi-branch architecture with a shared trunk and objective-specific branches. We pretrain on a heterogeneous corpus of spectrogram and channel state information data and evaluate six downstream tasks that span communication, sensing, and positioning. At an equal output embedding dimension, fusion outperforms the evaluated single-objective encoders on all six tasks while using roughly one-third of their parameters. The multi-branch architecture retains much of the fusion benefit within a single-encoder parameter budget. Representation analyses indicate complementary contributions across objectives, with much of the added benefit retained in components orthogonal to the reconstruction representation subspace. These findings support objective diversity as an effective strategy for scaling wireless foundation models. 

---
# CyTReX: Explainable AI-Based Cybersecurity Threat Reasoning Framework for DER Networks 

**Authors**: Damilola Popoola, Souradeep Bhattacharya, Manimaran Govindarasu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04286)  

**Abstract**: Distributed Energy Resource (DER) environments rely on network communication protocols to coordinate control commands, measurements, and device states across edge assets and cloud systems. Edge anomaly detection systems (ADS) monitor this traffic to identify deviations from normal communication behavior, flagging suspicious flows for further investigation. When the ADS flags abnormal network traffic, a single attack label is often insufficient for operational response: the label reports the detector's selected class but does not expose alternative threat interpretations that may warrant investigation. This paper presents Cybersecurity Threat Reasoning with Explainable Artificial Intelligence (CyTReX), an evidence-grounded threat reasoning framework for DER security that transforms network-level anomaly alerts into ranked, analyst-facing threat hypotheses designed to support Security Operations Center (SOC) triage and investigation. CyTReX constrains large language model (LLM) reasoning through a structured evidence packet, defined as a consolidated record of detection outputs, model explanations, and cyber threat intelligence (CTI) context. The evidence packet integrates edge-layer anomaly detection evidence, cloud reasoning layer attack interpretation, Shapley Additive Explanations (SHAP) network-feature attributions, surrogate decision rules, and Model Context Protocol (MCP)-enabled CTI enrichment. This ensures that every ranked hypothesis and attack-tree branch is traceable to explicit evidence rather than free-form LLM inference, and that incomplete or conflicting evidence is communicated rather than suppressed. Evaluation across five configurations shows that additional reasoning components improve hypothesis specificity, evidence traceability, and analytical grounding, with the complete pipeline providing the richest evidence-grounded reasoning context. 

---
# First-Order Steering: Translating Weight Adaptation into Activation Steering 

**Authors**: Sri Pranav Kunda, Alexander Kurz, Tomas Dominik, Uri Maoz  

**Link**: [PDF](https://arxiv.org/pdf/2610.04283)  

**Abstract**: Activation steering exploits interpretable directions in the residual stream to enable inference-time manipulation of model behavior. Composing steering vectors to apply multiple target behaviors simultaneously is important in various fields-including AI alignment and safety-but remains a challenge for existing activation steering methods. In contrast, prior work in model merging shows that target behaviors represented by learned weight adaptations can be combined with high accuracy. A method that translates weight adaptations into activation steering vectors could therefore extend prior work in model merging to generate composable steering vectors that better enable simultaneous inference-time behavioral control. For this, we introduce First-Order Steering, a formulation of activation steering as a first-order approximation of weight update matrices parameterized by a vector of steering strengths, and establish theoretical bounds on the approximation error of first-order steering. We then develop a novel model merging procedure, HeRD-Merging, which minimizes the first-order approximation error terms to enable higher first-order steering accuracy. Together, our method produces steering vectors that control both individual and composed behaviors more accurately than existing activation steering methods. Furthermore, HeRD-Merging matches the performance of conventional model-merging baselines, while producing weight adaptations that admit more accurate first-order steering vectors. 

---
# Rethinking Self-Distillation for Multi-Teacher Capability Merging 

**Authors**: Roy Xie, Dan Friedman, Feng Nan, Yukun Huang, Zhichao Xu, Chengjiu Zhang, Jun Xu, Manaal Faruqui, Vivek Rathod, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2610.04272)  

**Abstract**: Combining capabilities of multiple expert models trained starting from the same base checkpoint has become increasingly common in frontier language-model post-training. Recent trends suggest that multi-teacher on-policy distillation (MOPD) outperforms conventional off-policy methods. However, despite the higher inference and environment interaction costs incurred by MOPD, we find that much of its reported accuracy gain is due to certain training design choices and hyperparameter optimization, rather than the algorithm itself. We conduct a controlled self-distillation study across two multi-teacher settings, four models, and eleven benchmarks, comparing off-policy methods, namely supervised fine-tuning (SFT) and soft-label distillation, to hybrid teacher-prefix distillation and MOPD. We found that all four methods achieve \textit{nearly identical} accuracy. However, MOPD uses $14.8$--$23.1\times$ SFT's training GPU-hours. We also revisit four recently published comparisons between on-policy and off-policy distillation and find that the reported on-policy gains shrink substantially when SFT baselines are trained on rejection-sampled teacher trajectories and use independently tuned hyperparameters. As a training-free alternative, we also find that simple weight-merging methods can recover expert capabilities with minutes of CPU merging time, although their accuracy degrades as model size decreases and task interference increases. Overall, our results question recent gains reported due to MOPD and suggest careful tuning of more efficient off-policy baselines as a viable alternative. 

---
# AI-Enabled Quality Assurance for Multiple-Choice Assessment Items 

**Authors**: Steven Moore, Nicholas Diana  

**Link**: [PDF](https://arxiv.org/pdf/2610.04267)  

**Abstract**: Generating multiple-choice questions is increasingly scalable, but establishing their assessment quality remains difficult. We present a focused narrative review of automated item-writing flaw detection, revision, psychometric screening, and NLP benchmark auditing. Database searches, citation retrieval, and nominated sources yield fourteen research reports reviewed in full text. We distinguish surface checks from content-sensitive judgments and map a 19-criterion rubric to detection methods and reported evidence. High label-level accuracy often coexists with weak positive case detection, while rubric definitions and reference standards vary. Revision evidence is mixed, and the associations reported in prior work do not establish the effects of repair. We propose evaluating quality assurance as a sequence of independently validated decisions, with criterion-specific reporting, calibrated human review, and outcome-based assessment of revisions. 

---
# Playing social deduction games with reinforcement fine-tuned large language models 

**Authors**: Lingzhe Zhang, Yunpeng Zhai, Tong Jia, Kening Zheng, Chiming Duan, Minghua He, Zhaoyang Liu, Bolin Ding, Philip S. Yu, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.04261)  

**Abstract**: Reinforcement fine-tuning (RFT) is increasingly used in applications where large language models (LLMs) interact with humans and other agents. Here we use social deduction games to study how RFT changes LLMs' social behaviour. We let fine-tuned and base LLM agents play hidden-role games that require hidden-state inference, social reading and vote steering. Our results show that LLM agents do not reliably acquire social-deduction ability by directly optimizing terminal win--loss outcomes, suggesting that final game results provide a sparse and noisy signal for socially interactive learning. However, RFT is particularly effective at improving social reading, including tasks that require agents to infer hidden roles from public discussion, update beliefs over time and predict other agents' future decisions. We further show that RFT can also improve social influence, including tasks that require agents to steer votes, team approvals and collective decisions, although these gains depend more strongly on behaviourally specific rewards and structured interaction settings. Finally, we show that LLMs' ability to play social deduction games can be further improved through multi-agent social-cognitive reinforcement fine-tuning, which combines social-reading and social-influence signals during same-side multi-agent training. These learned behaviours also receive more favourable human evaluations of strategic competence, persuasiveness and social usefulness. Together, these results enrich our understanding of how RFT changes LLMs' social behaviour and provide a step toward a behavioural learning theory for machine social intelligence. 

---
# Benchmarking Psychological Dynamics in Generative Agents 

**Authors**: Sumer S. Vaid, Ashley V. Whillans  

**Link**: [PDF](https://arxiv.org/pdf/2610.04246)  

**Abstract**: Large language models (LLMs) are increasingly deployed to simulate human behavior, acting as computational replicas of human subjects. Yet the lived psychological experience of humans is difficult to benchmark, particularly as it unfolds over time. We introduce a psychometric benchmark for computational replicas: personas that carry a fixed identity through an evolving sequence of events. Built entirely from published norms and meta-analytic effects, the benchmark scores two dimensions of psychological realism. The first, internal validity, quantifies whether generated trajectories reproduce the internal structure of repeated human measurement: distributions, the between- versus within-person variance partition, temporal dependence, and range. The second, external validity, quantifies whether replicas recover established trait, state, and indicator relations. Across 36 open-weight and proprietary LLMs from nine developers (1B-671B parameters), most recover the direction of established relations (84.3% mean agreement) and the variance partition (26 of 34), yet the within-person correlation and distributional structure elude recovery. Internal validity is independent of scale and capability: a mid-size open LLM (Gemma-3-27B) strikes the best trade-off between the two dimensions. The benchmark is a precondition for using computational replicas in causal inference across domains (e.g., marketing, healthcare), and identifies within-person grounding as the central challenge ahead. 

---
# Attention-Based Surface Representation Learning for Robot State Prediction and Open-Ended Surface Classification 

**Authors**: Oleg Kushnarev, Alexander Belyaev  

**Link**: [PDF](https://arxiv.org/pdf/2610.04240)  

**Abstract**: For ground robots operating in outdoor environments, understanding the properties of the underlying terrain is essential for ensuring reliable operation. In most perception-based studies, this problem is formulated as categorical classification with a fixed number of classes defined during training. We propose an approach that enables new surface classes to be added as trainable vectors, which can subsequently be used to address higher-level tasks. By employing a learning paradigm based on predicting the robot's next state in time and using attention blocks, we improved classification accuracy to 98.56% on the Belyaev-Kushnarev dataset and 94.8% on BorealTC. 

---
# Adaptive Operator Selection in Bilevel Large Neighborhood Search for Electric Autonomous Dial-a-Ride Problem under Uncertainty 

**Authors**: Ishara Hewa Pathiranange, Aneta Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2610.04219)  

**Abstract**: The electric autonomous dial-a-ride problem (EADARP) extends the classical dial-a-ride problem by incorporating battery and charging constraints for electric vehicles. In practice, travel-time uncertainty can cause violations of time-window constraints. Large neighborhood search is effective for solving the EADARP, but its performance can depend on the choice of insertion operator during the repair phase. This paper investigates insertion-operator selection within a bilevel large neighborhood search framework for deterministic and chance-constrained variants of the EADARP. In the chance-constrained variant, arc travel times are modeled as independent normally distributed random variables, and upper time-window constraints are enforced probabilistically. We consider six selection methods, namely fixed greedy insertion, fixed regret-based insertion, random selection, a deterministic state-based rule, performance-adaptive ALNS selection, and LLM-based state-aware selection. Experimental results show comparable performance on smaller instances, while differences become more evident on larger and more constrained instances. There is no single strategy that performs best across all instances, and the relative performance of the LLM-based, rule-based, and ALNS strategies varies with the problem instance and experimental setting. 

---
# FTD-GNO: Memory-Efficient Graph Neural Operators through Functional Tensor Decomposition of the Kernel 

**Authors**: Xiaomin Zhang, Boyue Wang, Junbin Gao, Yongli Hu anbd Baocai Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.04212)  

**Abstract**: Graph Neural Operators (GNOs) provide flexible surrogate models for learning solution operators of partial differential equations (PDEs). However, standard GNOs typically parameterize the integral kernel with a monolithic neural network and evaluate kernel interactions over graph edges, leading to substantial computational and memory overhead at high resolutions or with large neighborhoods. To address these limitations, we propose Functional Tensor Decomposition Graph Neural Operator (FTD-GNO), a memory-efficient GNO framework that decouples the high-dimensional continuous integral kernel into low-dimensional mode-wise functions. By instantiating the kernel with classical tensor decomposition formats, including CP, Tensor-Train, and Tucker decompositions, FTD-GNO enables algebraic reconstruction of the integral operator without explicitly materializing full edge-wise kernel tensors. This factorized formulation reduces the memory footprint of kernel evaluation and aggregation while retaining the continuous operator-learning structure of GNOs. Theoretical complexity analysis shows that FTD-GNO substantially lowers parameter and activation-memory costs associated with high-dimensional kernel construction. Experiments show lower peak memory than the corresponding unfactorized graph-integral baselines, with shorter recorded training times. Fourier-graph experiments further demonstrate that FTD can improve the efficiency of a graph-integral layer within a hybrid operator and has good scalability. 

---
# CurveCodec 2: Skeleton-agnostic animation compression with a learned entropy model 

**Authors**: Mingyi Shi, Huancheng Lin, Xuelin Chen, Taku Komura  

**Link**: [PDF](https://arxiv.org/pdf/2610.04211)  

**Abstract**: Skeletal motion is stored as every joint's transform at every frame, yet most of it is implied by the body rather than by what the motion is about. Compression is one way to ask what a motion must still say once the body is known, and a production codec must answer it for any skeleton with a stated error bound. Our earlier codec, CurveCodec, matched the mean error of ACL, the production library of modern game engines, with a learned prior over sparse anchors, but not ACL's worst case, and it counted its payload as floats rather than bits. Here we ask where the redundancy of skeletal motion lies and which part of a codec a learned model should take over. Measurements give three answers. At production precision the largest saving comes from predicting each quantized curve from its own past, the second from choosing per joint, in closed loop through the hierarchy, which samples not to code. On the gaps such an encoder leaves, a nearest-neighbour oracle over millions of training samples is no better than linear interpolation, and no learned in-betweener we tried paid for itself. What a network does learn is the distribution of the residuals the codec must send. CurveCodec 2 codes every sub-track as a curve in the log map, quantized in closed loop and thinned to rate-distortion-selected keys, with residuals entropy-coded under a small learned model whose integer inference is bit-exact across platforms. Two contracts are verified on every decoded clip: ACL's own worst case per joint within a stated tolerance, or ACL's mean error per clip. On a held-out test side of 4,472 clips from 33 datasets, CurveCodec 2 needs 0.37x ACL's bytes at ACL's default precision of 0.01 cm under the worst-case contract and 0.22x at 0.1 cm under the mean contract, decodes on one CPU core, and transfers without retraining to a species absent from training. Project page: this https URL 

---
# Clean: Second-order LLM Training at Linear Memory Cost via Nyström Sketching 

**Authors**: Beheshteh T. Rakhshan, Sahar Rajabi, Maziar Sargordi Shikai Fang, Guillaume Rabusseau, Sirisha Rambhatla  

**Link**: [PDF](https://arxiv.org/pdf/2610.04204)  

**Abstract**: Training large language models (LLMs) entails a fundamental trade-off: memory-efficient optimizers such as Adam discard cross-parameter curvature, whereas full-curvature methods such as SOAP can accelerate convergence at prohibitive memory costs. We introduce Clean, a memory-efficient and full-curvature optimizer designed to resolve this bottleneck. Clean leverages the randomized Nystrom method to accurately approximate the left and right preconditioners in SOAP, and to reduce the optimizer's memory complexity from quadratic to linear in terms of model dimensions. We subsequently reintegrate the off-subspace components to capture curvature information beyond the low-rank approximation, preserving rich curvature at minimal memory cost. We further propose Q-Clean, a low-precision variant that aggressively compresses optimizer states. Q-Clean reduces optimizer memory consumption by \textbf{over 50\%} compared to Muon when pre-training a LLaMA-1.3B architecture, all while maintaining strong and competitive predictive performance. Notably, Clean operates with a smaller optimizer-state footprint than standard AdamW while reaching AdamW's final performance \textbf{26\% faster} in wall-clock time. Furthermore, our methods uniquely enable the pre-training of a 13B-parameter model on a single 80GB GPU, providing a scalable, efficient, and accessible approach to large-scale model optimization. 

---
# Risk-Calibrated Proposal Transport for Finite-Particle Diffusion Steering 

**Authors**: Ziseok Lee, Jaehyeon Kim, Seungwon Kim, Seunghyun Moon, Haneul Choi, Wooyeol Lee, Donghyun Koh, Minhyeong Lee, Kyungsu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.04171)  

**Abstract**: Inference-time steering combines pretrained diffusion experts or rewards without retraining by changing the dynamics that transport noise to data. Feynman-Kac correction compensates for proposal mismatch through importance-weighted sequential Monte Carlo (SMC), whose finite-particle behavior depends on the proposal. Variance-controlling guidance (VCG) improves that proposal by fitting a linear drift correction to minimize empirical log-weight-rate variance. Although its population optimum cannot worsen residual variance, finite-particle VCG can nearly eliminate its fitting residual while increasing residual risk on new states by orders of magnitude. The resulting update can degrade unweighted generation or accelerate particle collapse. We show that the centered Feynman-Kac rate is the normalized transport residual and that expected out-of-fit benefit is exactly population headroom minus coefficient-estimation penalty. Under regularity assumptions, a Wasserstein analysis bounds the unweighted proposal's terminal error using this residual. These results motivate Risk-Calibrated Proposal Transport (RCPT), which uses deletion leave-one-out residuals to calibrate the retained fraction of the VCG update, adding no model calls and only small linear-algebra overhead. Experiments on 2D checker distributions, scaffold decoration, molecular property optimization, and class-conditional CIFAR-10 generation demonstrate recovery from harmful fitted updates. Across molecular and image domains, RCPT mitigates harmful fitted updates and improves a broad range of terminal metrics relative to uncalibrated VCG. 

---
# Language Model Fingerprinting Requires Rethinking Watermark Teachers 

**Authors**: Jeongyeon Hwang, Anshul Nasery, Sewoong Oh, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2610.04169)  

**Abstract**: LLM fingerprinting via watermark distillation embeds a statistical watermark signal into model weights, enabling model owners to identify their models behind black-box APIs. Revisiting a recent protocol, we find that its utility evaluation understates text quality degradation in open-ended generation, favoring overly strong watermark teachers. Weakening the watermark improves text quality but sacrifices detectability. To move beyond this trade-off, we rethink whether text watermarks designed for verifying generated text are suitable distillation teachers for model fingerprinting. Such watermarks are typically designed to remain detectable from an individual output, limiting how sparse the watermark signal can be. In contrast, fingerprint verification can aggregate signal across queries, making sparser watermark signals viable. This raises a key question: where should the sparse signal be placed? We analyze signal placement through token surprisal and show that, even at comparable watermark strength, different placements can target tokens with different plausibility under the base model. This motivates near-tie restriction, which uses top-1-relative logit gaps to restrict the watermark bias to tokens close to the base model's top prediction. Across multiple models, near-tie improves detection--quality frontiers under deployment changes, preserves higher text quality across query budgets, and further improves existing watermarking schemes when combined with them. 

---
# How RL Reshapes LLM Reasoning: Transferability, Coverage, and Scaling Laws 

**Authors**: Ziheng Cheng, Yixiao Huang, Hanlin Zhu, Somayeh Sojoudi  

**Link**: [PDF](https://arxiv.org/pdf/2610.04158)  

**Abstract**: Recent studies on reinforcement learning (RL) report seemingly conflicting evidence about large language model (LLM) reasoning. Training on mathematics can improve performance in other domains, yet gains in Pass@1 can coincide with lower Pass@$N$ than the base model. This raises a fundamental question: does RL expand an LLM's reasoning boundary, or merely reweight its existing reasoning space? We revisit these phenomena across Qwen and Gemma model families, showing both cross-domain gains and forgetting, while coverage at large sampling budgets increases on some tasks and decreases on others. Detailed analysis of solution traces before and after RL indicates a shift in the reasoning strategies the model employs, motivating a two-stage autoregressive policy model that separates \emph{strategy selection} from problem-specific execution. Within this framework, we prove how RL's implicit bias reshapes strategy preferences, allowing gains on some tasks while suppressing strategies required by others. This mechanism can also broaden or narrow coverage at a given sampling budget even without expanding strategy support. We further provide theoretical justifications for log-sigmoid and log-linear scaling laws in RL compute, and evaluate their predictive power. Together, these results connect changes in strategy selection to cross-domain transfer, reasoning coverage, and compute scaling. 

---
# Trajectory-Derived Confidence for Reliable, Resource-Aware Clinical Text-to-SQL Agents 

**Authors**: Mincheol Daniel Song, Joshua Ward, Jake Jung, Guang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.04156)  

**Abstract**: LLM agents for clinical text-to-SQL applications reason autonomously over multiple steps but cannot assess whether their own reasoning or outputs can be trusted. In high leverage applications such as healthcare, this presents a critical risk where system mistakes can be costly. These reliability failures are also resource failures: an incorrect reasoning trajectory spends computation budget on outputs that must be discarded. We introduce Sentinel, a trajectory-derived, classifier-based confidence layer that analyzes an agent's reasoning, code and database outputs to decide at three points whether to stop: refusing unanswerable questions before the agent runs, halting doomed trajectories mid-run, and withholding untrustworthy answers at delivery. Here, utilizing Chow's rule, we optimize decisions under the EHRSQL shared task's Reliability Score, which penalizes incorrect answers given a utility weighting, and find on the benchmark EHRSQL that Sentinel raises this score from +0.08 to +0.24 when mistakes have a low utility weighting, well above the +0.03 earned by refusing every question, with delivered-answer accuracy rising from 54% to 69% as coverage falls from 84% to 55%. At higher stakes the agents we test rarely answer reliably enough to deliver, and Sentinel detects this on its own, abstaining to that same +0.03 where the unmonitored agent scores -3.02. The same stopping decisions cut computation: at low stakes, where the system still answers, Sentinel eliminates 13-28% of agent steps at little or no reliability cost. 

---
# Dual-Scale Relational Graph Transformers for Ecosystem-Aware Fraud Detection 

**Authors**: Mohsen Nayebi Kerdabadi, Xinrou Li, Yao Xiao, Zijun Yao, Xin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.04138)  

**Abstract**: Account takeover (ATO) fraud is a growing threat to digital banking, requiring effective detection while minimizing friction for legitimate customers. Production systems predominantly rely on tabular models that score sessions in isolation, discarding the relational structure of the underlying interaction network. Although graph-based models exploit relationships among sessions and network entities, they primarily reason over local neighborhoods and therefore capture only part of the problem: fraud risk depends jointly on the local relational structure surrounding a session and the evolving global state of the fraud ecosystem. We present HERMES (HEterogeneous Relational Micro--macro graph transformer Encoder for high-risk Sessions), a dual-scale architecture that jointly models these complementary scales of information. Micro-GT captures local heterogeneous graph structure through structured, relation-aware attention over a temporally safe session neighborhood. Complementing this local representation, Macro-GT models ecosystem-level context using non-anticipative climate tokens that summarize fraud dynamics, platform shifts, and infrastructure reuse, together with adaptive class prototypes that track representative fraud and benign session patterns over time. Evaluated on more than 130 million high-risk transaction sessions from a leading U.S. financial institution, HERMES consistently outperforms production and strong graph-based baselines, achieving a 44.44% relative reduction in customer friction and a 24.66% relative improvement in fraud recall over the production system. Ablation and temporal-stability analyses further demonstrate complementary gains from local relational modeling and global ecosystem context across changing fraud regimes. 

---
# Agentic Resource Allocation for Batch Multi-Objective Bayesian Optimization in Autonomous Materials Discovery 

**Authors**: Robert Robinson, Shakti Prasad Padhy, Sushant Sinha, Sk Md Ahnaf Akif Alvi, Juan Florez Coronel, Brent Vela, Trevor Hastings, Douglas Allaire, Raymundo Arróyave  

**Link**: [PDF](https://arxiv.org/pdf/2610.04134)  

**Abstract**: The discovery and development of advanced materials is a challenging process constrained by the high time and monetary costs of synthesis, processing, and characterization. The underlying design spaces can be enormous, often with multiple competing objectives. Bayesian optimization (BO) provides a principled approach for efficiently navigating such spaces, but most workflows rely on fixed exploration-exploitation policies that lack the capacity to adapt to shifting constraints in dynamic campaigns typical of self-driving laboratories. In this work, we develop a multi-objective BO framework for alloy design under resource constraints, benchmarking strategies for adaptive policy tuning at each iteration. Our evaluation covers a septenary refractory high-entropy alloy (RHEA) system focused on maximizing melting temperature and minimizing density, and an Fe-Co-Ni-based soft magnetic alloy system targeting saturation magnetization, coercivity, and hardness. We compare an exploitation-focused strategy, a fixed mixed exploratory/exploitative policy, and two distinct LLM-based adaptive strategies with different approaches to batch allocation and campaign signal interpretation, evaluated across baseline and mid-campaign resource event conditions including budget reductions, timeline cuts, and combined disruptions. Our results show that mixed allocation strategies accumulate substantially more mutual information than the exploitation-focused baseline at a proportionally smaller cost to hypervolume and optimization speed, with adaptive strategies outperforming a fixed-mixed allocation policy by adjusting their allocation in response to both evolving campaign statistics and resource constraints. These findings suggest that adaptive resource allocation offers a favorable tradeoff for materials discovery campaigns in reducing predictive uncertainty on Pareto-optimal compositions. 

---
# PB-GRPO: Learning Socially Adaptive LLM Agents from Persona-Driven Simulation with Preference-Batched GRPO 

**Authors**: Jingquan Wang, Jun Yin, Xu Han, Yongsheng Mei, Jie Hao, Bin Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.04132)  

**Abstract**: Building LLMs that behave well socially, not merely correctly, requires Building LLMs that behave well socially, not merely correctly, requires more than producing locally helpful responses. A socially competent agent must infer users' unstated goals, respect their preferences, and adapt as the conversation unfolds. These behaviors are inherently multi-turn and social, making them hard to optimize: real interaction data is scarce, and user preferences are typically latent rather than directly observable. To address these challenges, we build on a persona-driven social simulation environment (consisting of a persona library, LLM-based user simulators, and a user-satisfaction scoring system ranging from [0, 1]), to introduce preference-batched GRPO (PB-GRPO), a post-training algorithm that learns socially adaptive policies from conversation-level feedback. Compared to vanilla GRPO, PB-GRPO computes advantages using a normalization estimated across a bucket of users with similar preferences, stabilizing training across a diverse social population. Empirical evidence shows that PB-GRPO improves models' social behavior over strong reinforcement learning baselines in our simulated environment. 

---
# Representational Control over Self-Report & Behavior Coherence in LLM Risk-Taking 

**Authors**: Rafal Kocielnik, Peiyang Song, Pengrui Han, Myrl G. Marmarelis, Ramit Debnath, Dean Mobbs, R. Michael Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2610.04125)  

**Abstract**: Self-report is an appealing low-cost probe of an LLM's dispositions, but recent work finds only selective agreement between what models report and how they behave. Prior accounts establish these patterns by prompting black-box LLMs, leaving open whether the gap is a prompting artefact or a fact about how the underlying constructs are represented internally. We investigate risk-taking, a consequential dimension of agentic decision-making, using activation steering to measure self-report and behavior under the same internal intervention. We survey nine steering-vector extraction methods spanning task-specific directives, the model's own task behavior, and dispositional descriptions at two granularities, evaluated on two behavioral tasks and two psychometric instruments across four open-weight LLMs. We find that (1) a shared internal intervention does not ensure shared responsiveness: directions built from trait descriptions move self-report but leave behavior at chance, directions built from the model's own task choices do the reverse, and only task-specific directives reach both, weakly. (2) Diagnosis dissolves that exception: removing surface confounders leaves the directives only 32% of their behavioral effect. The two channels are otherwise steered by near-orthogonal directions, each channel reachable by several independent constructions. (3) An intervention composing one behavior-moving and one self-report-moving direction moves both together; flipping one sign sets them in opposition, with reported and enacted risk pointing in opposite directions on 69-89% of flipped compositions in all models. These findings move the self-report-behavior relationship from a black-box observation to a representational one that can be inspected and controlled, motivating representational checks alongside behavioral evaluation. 

---
# Agent Reliability Profiles in Financial Services 

**Authors**: Mike Hsu, Medha Bankhwal, Béatrice Moissinac, Kevin Werbach, Lukasz Szpruch, Bennett Hillenbrand  

**Link**: [PDF](https://arxiv.org/pdf/2610.04123)  

**Abstract**: AI agents can take actions. At times, those actions can go beyond what is intended. Agent reliability can be defined as assurance that an agent will stay within intended bounds and operate within limits. Today, there is no shared framework or language for describing, validating, and benchmarking the reliability of agentic deployments in financial services. This makes it difficult for financial institutions, vendors, and regulators to assess and trust agents at scale, thus limiting the pace of development and adoption. A standardized, shared representation of agent reliability would fill the gap. This paper introduces the Agent Reliability Profile, a per-agent unit of assurance evidence for agent deployments in financial services. Each Profile records a bounded, falsifiable claim, this agentic system reliably functions within its operating boundary. We define "operating boundary" as an agent having; (1) a defined autonomy tier, (2) a defined operational design domain, (3) defined classes of action, and (4) a defined control envelope. Production assurance progresses through three levels while the Profile schema remains constant: a Profile Builder compiles a Level 1 Asserted Profile from institutional evidence, a Profile Validator tests the deployment in its own environment to produce a Level 2 Validated Profile, and operation of the same tests by a qualified independent assessor produces a Level 3 Verified Profile. Separately a Benchmarked Profile reports results comparable across institutions under reference conditions. We describe the architecture, the artifact, the assurance ladder, the comparability flag, associated tools, an evaluation methodology, applications for financial institutions and supervisors, limitations, and a staged implementation program. 

---
# SEER: Self-Evolving Event Reasoning and Retrieval for Time Series Forecasting 

**Authors**: Mingtian Tan, Palash Goyal, Mihir Parmar, Sarkar Snigdha Sarathi Das, Chun-Liang Li, Nanyun Peng, Thomas Hartvigsen, Jinsung Yoon, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2610.04109)  

**Abstract**: Real-world time series are frequently driven by exogenous events and structural shifts, rendering conventional forecasting based solely on historical numerical observations insufficient. While language models can retrieve external news, standard retrieval-augmented approaches struggle with high noise, missing signals, and an inability to reason causally about event impacts. We propose SEER (Self-Evolving Event Reasoning and Retrieval), a closed-loop framework that dynamically optimizes event conditioning for time series forecasting. SEER translates prediction errors into two decoupled feedback mechanisms: (i) a reflective retrieval memory that refines subsequent search queries and filters spurious noise, and (ii) a persistent causal knowledge base that distills transferable domain dynamics. SEER enforces strict chronological boundaries across both event retrieval and reflection, preventing look-ahead bias and data leakage. Across six volatile time-series benchmarks, SEER consistently outperforms state-of-the-art time series foundation models and language model baselines. 

---
# Robust blind unmixing: A geometric approach to overcoming basis variation 

**Authors**: Dumitru Mirauta, Vladimir V. Gusev, Michael W. Gaultois, Matthew J. Rosseinsky, Yannis Goulermas  

**Link**: [PDF](https://arxiv.org/pdf/2610.04091)  

**Abstract**: Signal separation problems are common in science. A prominent example of this occurs during the use of diffraction or spectroscopy to identify the individual components of a mixture by measuring it. In the simplest case, the measured signal is a linear combination of basis patterns corresponding to the constituent parts. The unmixing problem is to infer all or some of these basis patterns and abundances of components from measurements of distinct mixtures. One of the core challenges of this task is the variation of the basis from mixture to mixture due to noise and the exact physics of the measurement process. This is usually addressed with tailored model-based and parametric methods that are then limited in use to specific application domains by the nature of the assumptions made. We propose a novel geometric approach to unmixing problems which views the generation of data during measurement through a metric space lens, thereby shifting the focus from parametrised models to a general relationship between basis transformations and the corresponding geometry. We take advantage of the optimal transport distances to capture commonly occurring basis variations, and use minimisation of in-class variance of candidate solutions to drive the optimisation. We pay special attention to the one-dimensional case due to its practical importance and availability of efficient distance and transport map routines. The effectiveness of our approach is demonstrated on a range of unmixing tasks using random Gaussian mixture models, simulated powder X-ray diffraction, and laboratory hyperspectral imaging datasets. 

---
# DUET: Co-Evolving Solver and Grader Agents 

**Authors**: Fengyu Gao, Sourav Pal, Austin Z. Henley, Arjun Radhakrishna, Gustavo Soares  

**Link**: [PDF](https://arxiv.org/pdf/2610.04087)  

**Abstract**: Agentic workflows are increasingly used across domains such as technology, finance, and enterprise operations. As these agents become more widely deployed, continually improving them becomes increasingly important. This raises an immediate challenge: How should the agent evolve? This evolution requires effective evaluation that can assess outcomes and provide useful feedback for optimization. As the agent evolves, its behaviors and failure modes may also change, making a fixed evaluator increasingly inadequate. Another fundamental question: How should we evaluate an evolving agent? These two challenges are inherently coupled; changes in agent behavior can expose limitations of the current evaluator, while a stronger evaluator provides more informative feedback for improving the agent. Motivated by this interaction, we introduce DUET, a framework that jointly optimizes a solver agent and a grader agent to improve both. DUET iteratively selects training tasks, executes them with the solver, evaluates the resulting outcomes with the grader, and uses a tool-using update module to revise the solver and the grader, alternating between the two across rounds. By updating the grader within the optimization loop, DUET turns evaluation from a fixed source of feedback into a first-class optimization objective that adapts alongside the solver. Experiments across four agent benchmarks show that DUET improves both solver and grader performance and consistently outperforms baselines that optimize the solver with a fixed grader. 

---
# Dependable AI-Assisted Engineering: A Formal Framework for AI Participation and Assurance in Safety-Critical Workflows 

**Authors**: Puxue Tan  

**Link**: [PDF](https://arxiv.org/pdf/2610.04084)  

**Abstract**: Generative AI can produce engineering artefacts, but generation alone does not determine whether or how those artefacts should enter safety-critical workflows. This paper develops a formal framework for assigning AI participation and assurance at the level of individual workflow units. Each unit has a participation and assurance record covering its engineering requirement, an approved operational formalization where applicable, the applicable mechanism, fallback where applicable, evidence obligations and the applicable guarantee, plus a deployment-readiness status. The framework distinguishes deterministic verification, statistically calibrated admission, authorized human judgement supported by AI advice, authorized human adjudication of AI-produced artefacts, retained deterministic tool paths and explicit non-participation; these arrangements carry different kinds of guarantee rather than levels on a common scale. The framework also separates formalization fidelity from verifier soundness, provides a staged classification and readiness procedure, and derives conditions for comparing a gated AI-assisted unit with an incumbent process under recurring-population assumptions. We instantiate and apply the framework in an executed 17-unit wing-spar structural-analysis workflow combining deterministically gated AI-generated CAD, retained deterministic computation and human judgement. The AI-generated CAD program passed all 23 deterministic checks and was admitted at the first attempt. Favourable stress magnitudes did not suffice to pass the stress criteria where the predeclared mesh-convergence evidence was insufficient; those criteria were instead referred to engineering judgement. The case demonstrates selective AI participation and explicit evidence handling at unit level; no claim is made of workflow-level dependability, certification, structural safety or productivity. 

---
# AEGIS: Differentiable Mars Climate Model with Neural Closures 

**Authors**: Sameera S Kashyap, Victor Cruz, Angel Yepez, Razvan Marinescu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04081)  

**Abstract**: General circulation models (GCMs) are the primary tool for simulating planetary atmospheres. They play a vital role in understanding Mars's atmosphere, as forecasting its unique weather is mission-critical for operations such as entry, descent, and landing. Mars poses unusual challenges for these models, as observations are sparse compared to Earth. In addition, a thin \co{} atmosphere alongside a radiatively active dust cycle creates a volatile atmosphere with large diurnal temperature swings and no true terrestrial analog for validation. Existing Mars GCMs, including the LMD PCM, the NASA Ames Mars GCM, and PlanetWRF, are mature and physically detailed but are implemented in legacy Fortran with finite-difference or finite-volume solvers, and they do not expose gradients for calibration or machine learning. Here we present AEGIS, a modular differentiable Mars climate model that couples Mars's unique atmospheric physics to the Dinosaur dynamical core, with interfaces for neural closures. We showcase stable ten-Mars-year simulations that reproduce the seasonal \co{} cycle while conserving the total \co{} inventory, capture realistic large-scale surface-temperature structure, and produce surface pressure that follows Mars Orbiter Laser Altimeter (MOLA) topography. Gradients through coupled trajectories agree with finite differences and support physical calibration and neural training. We compare with conventional GCMs, highlighting the framework's computational efficiency and differentiability. 

---
# A study on human-agent teaming through spoken interaction: the impact of human individual traits and agent characteristics 

**Authors**: Lara Gauder, Martin Bernardo Meza, Javier Krick, Alejandro Masin, Luciana Benotti, Marcos Luis Pietto, Luciana Ferrer  

**Link**: [PDF](https://arxiv.org/pdf/2610.04079)  

**Abstract**: Human-agent teams are collaborative systems where humans and agents work interdependently to achieve shared goals. The success of such teams is associated with the human's perception of the agent as a legitimate teammate. This perception is thought to depend not only on the agent's capabilities and reliability but also on social factors. We explore whether simple changes in an agent's communicative behavior can influence the human's perception of the agent's teammate-likeness. To this end, we developed a protocol based on a collaborative game requiring spoken interaction, in which a human subject and a virtual agent collaborate to identify a target object and place it on a board. Each subject interacted with two agents which provided the same task-relevant information but differed in behavior. While one was a neutral tool-like agent, the other was a team-building agent that used simple social strategies such as empathy, politeness, and positivity. Our analysis shows that the team-building agent was perceived as significantly more teammate-like, even in terms of its ability, despite both agents having identical capabilities for game-solving. Further, subjects with a higher propensity to trust automated systems and lower neuroticism tended to have a more favorable perception of the agent. Finally, we found that subjects tended to change their prosodic patterns when talking to the team-building agent, as a classifier based on prosodic features was able to predict the type of agent with better-than-random performance. A practically relevant conclusion is that minimal changes in spoken communication behavior of the agent, easily implemented in a variety of scenarios, can have a significant positive impact on the human's perception of its teammate-likeness. 

---
# Reward-DAgger: Robot-Gated Interactive Imitation Learning with General-Purpose Progress-Based Reward Models 

**Authors**: Ryan Li, Yigit Korkmaz, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2610.04054)  

**Abstract**: Recent advances in robot learning have enabled generalist control policies capable of completing a wide range of tasks. However, their performance degrades when deployed in unseen environments, making it critical to detect failures and teach recovery behaviors. Existing runtime monitoring methods often require task- and policy-specific training or hyperparameter tuning, limiting cross-task deployment and introducing additional overhead during iterative policy updates. We present Reward-DAgger, a robot-gated interactive imitation learning framework that uses dense progress signals from a general-purpose reward model to determine when human intervention is needed. Our approach is agnostic to the underlying policy architecture, requires no access to policy internals, and can be applied across tasks without retuning the gating mechanism. Our results show that Reward-DAgger achieves a better failure-detection accuracy-latency tradeoff than existing runtime monitoring baselines. Across eight simulated and real-world tasks, Reward-DAgger consistently improves the downstream policy's autonomous success rate throughout interactive learning and achieves strong return on human effort, outperforming the baselines in most settings. Importantly, the same gating configuration is used across tasks without task-specific hyperparameter tuning, demonstrating transfer across tasks, environments, and policy architectures. Code and videos are available at this https URL. 

---
# MAGEFormer: Learning Metric-Consistent Representations for Anisotropic CT Segmentation 

**Authors**: Jiaying Li, Paolo Remagnino  

**Link**: [PDF](https://arxiv.org/pdf/2610.04036)  

**Abstract**: Vision Transformers (ViTs) have shown strong performance in volumetric segmentation, but their effectiveness on clinical CT is limited by an isotropic Euclidean lattice assumption. This conflicts with anisotropic CT acquisition, leading to two key issues: (1) a metric mismatch between voxel indices and physical anatomy, and (2) accuracy degradation from isotropic resampling. To address this, we propose MAGEFormer, a geometry-calibrated framework that embeds physical metric constraints directly into representation learning. Our method introduces Metric-Adaptive Spatial Embedding (MASE) to calibrate positional frequencies using voxel spacing, Geometry-Constrained Attention (GCA) to suppress physically implausible feature correlations, and Geometric View Voting (GVV) to reduce discretization bias during inference. We evaluate MAGEFormer on two multi-organ abdominal CT benchmarks, BTCV and FLARE 22, under a unified protocol against strong CNN and Transformer-based baselines. MAGEFormer achieves the strongest boundary accuracy among the compared methods, with 10.58 mm HD95 on BTCV and 3.40 mm HD95 on FLARE 22, and shows consistent gains in Dice under the same protocol. These results show that geometry-aware internal calibration is more effective than relying on conventional isotropic preprocessing alone for anisotropic CT segmentation. 

---
# Slaying the Hydra: Interaction-Aware Circuit Discovery in Language Models 

**Authors**: Sankaran Vaidyanathan, Rafal Urbaniak, Emily Bunnapradist, Michelangelo Naim, Daniel Waxman  

**Link**: [PDF](https://arxiv.org/pdf/2610.04017)  

**Abstract**: Localizing behavior to individual components of a language model is a central goal of mechanistic interpretability. However, scoring components one at a time misses context-dependent effects: a primary component can inhibit the activation of a backup, leading to issues with ranking components. Actual causality studies the structure of such interactions via witnesses: variables that provide contextual information to resolve interaction terms. However, estimation with witnesses typically requires combinatorial enumeration and is infeasible in practice. We introduce the witness-integrated set effect (WISE), a family of causal estimands that build on the witness mechanism while taking expectations over sets of causes and witnesses to remain computationally feasible. Building on this approach, we introduce JuntaLearner, a gradient-based circuit discovery method that learns to rank components by their causal impact across varying-sized sets of components and witnesses. Alongside faithfulness metrics, we introduce measures of necessity and task specificity, and the circuit recognition score (CRS) to summarize each metric across circuit sizes while emphasizing effects achieved by small circuits. Across tasks and models of increasing size, JuntaLearner achieves higher mean CRS compared to attribution baselines on all metrics. Since its cost does not grow with the number of candidate components, JuntaLearner scales to large models while accounting for set-level interactions and avoiding first-order approximations. 

---
# Selective Backpropagation for Efficient Few-Shot Class-Incremental Learning 

**Authors**: Eeham Khan, Abdulmoumen Al-Atrash, Ali Ayub  

**Link**: [PDF](https://arxiv.org/pdf/2610.04003)  

**Abstract**: Few-Shot Class-Incremental Learning (FSCIL) requires models to continuously learn new classes from limited samples while retaining prior knowledge, under strict constraints on compute and memory. Existing approaches lie along a difficult trade-off: simple fine-tuning is computationally efficient but suffers from catastrophic forgetting, replay-based methods mitigate forgetting at the cost of substantial compute and memory, and exemplar-free methods often reduce forgetting by freezing most of the backbone, improving efficiency at the expense of adaptability. We propose Selective Backpropagation (SBP), a deterministic parameter budgeting framework that bridges this gap. SBP restricts gradient updates to a pre-allocated subset of network parameters, freezing past knowledge and preserving unbiased capacity for future learning, enabling rapid adaptation without costly mask optimization. We show that SBP achieves strong performance across standard FSCIL benchmarks while requiring training time close to that of naive fine-tuning and substantially lower training time than prior SOTA methods. Crucially, our experiments expose a limitation of standard FSCIL evaluation: performance on short, distribution-consistent benchmarks does not necessarily predict behavior under distribution shift or over substantially longer learning horizons. We therefore evaluate FSCIL methods in cross-domain settings and over an 80-session ImageNet-1K stream. SBP remains strong across these regimes while maintaining low training cost, providing a favorable stability-plasticity-efficiency trade-off. Our code is available at this https URL. 

---
# Solving VeriContest with a Lean-Backed Rust Verifier 

**Authors**: Traian Serbanuta, Jun Xu, Andrei Stefanescu, Cosmin Radoi  

**Link**: [PDF](https://arxiv.org/pdf/2610.03994)  

**Abstract**: VeriContest is a benchmark of 1007 competitive-programming problems in Rust, each with a Verus specification, a judge-accepted solution, and a Verus proof. Its authors report that proof generation is the bottleneck for frontier models: given the specification and the code, the best model produces an accepted Verus proof for 13.95% of the problems on the first attempt. We report on solving the same proof-generation task with Rust-Prover, a verifier for Rust backed by Lean 4. The Verus specification and the Rust code are restated and translated into Lean, each specification becomes a theorem, and agents prove the theorems with Lean's kernel as the final check. All 1325 theorems of all 1007 problems were proved. 1259 of them were proved in one run of under 32 hours on Claude Opus 5.5, at a median of 3.2 minutes and $1.17 per proof, and 70% of them on the first iteration. The restated specifications were checked against the benchmark's test suites, and reviewed where no suite applies. None was wrong or weakened. The translated Lean programs were run on 21,413 of the benchmark's test cases and produced the same output as the Rust programs on every one. Across four Claude and four GPT models at five reasoning-effort settings, every current frontier model proves nearly all of a ten-theorem sample at every setting, and more effort raises the cost without raising the number of proofs. The cheapest Claude setting, Sonnet 5.5 at low effort, proves all of the 50 hardest theorems. We also rerun the benchmark's own Verus protocol with Claude Opus 5.5 on the 50 problems with the longest reference proofs. Opus 5.5 alone fails to prove one of them. 

---
# An Executable Benchmark for LLM-Based HLS Repair:Design Complexity and Repair Underconstraint 

**Authors**: Maisha Mastora, Dean Sullivan  

**Link**: [PDF](https://arxiv.org/pdf/2610.03971)  

**Abstract**: Automated repair of High-Level Synthesis (HLS) designs using large language models (LLMs) is an emerging but underexplored problem. While LLM-based repair shows strong results on register-transfer level (RTL) Verilog, the only prior systematic study of HLS logic repair reports just 10.5% correction accuracy for GPT-4, with no analysis of why repair fails or what drives difficulty. We present the first comprehensive evaluation of LLM-based HLS repair across four models (GPT-4o, GPT-4o-mini, GPT-5.4, and Claude Opus 4.6) on 125 benchmark instances spanning eight logic bug types across three open-source HLS suites (CHStone, MachSuite, Polybench). We construct the first executable APR-style HLS repair benchmark with suite-specific functional oracles, enabling pass@k evaluation rather than the string-match approximations used in prior work. Design context and scale, rather than bug type alone, dominate repair difficulty: repair rates range from 6-45% on complex cryptographic kernels (CHStone) to 81-93% on compact algorithmic kernels (MachSuite) despite identical bug type distributions. We introduce solution multiplicity, the fraction of distinct patches generated across repeated repair attempts, as an empirical measure of repair underconstraint, and show it strongly predicts repair failure across all four models (Spearman rho = -0.393, p = 6.61 x 10^-20). Frontier models reduce underconstrained instances from 41-46 (GPT-4o, GPT-4o-mini) to just 3 (Claude Opus 4.6), with pass@1 improving from 42-53% to 73-76%. SHFT bugs remain consistently hard across all models, and semantic bug classes such as buffer indexing become reliably repairable only at frontier scale. These findings show that future APR benchmarks must include complex, executable, and weakly identifiable designs that remain challenging after frontier LLM repair. 

---
# A Step Towards Forgetting: Optimiser History and the Loss of Answer Mass 

**Authors**: Valeria Ruscio, Seth Nabarro, Keiran Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2610.03940)  

**Abstract**: During fine-tuning, a language model can assign less probability to previously learned answers even when the current gradient acts to preserve that probability. With momentum, each update also carries gradients computed at earlier model states, and these stored contributions can push the model in the opposite direction. We investigate how this optimiser memory contributes to forgetting by separating old-task loss into confusion among its answers and leakage of probability outside the answer set. Across three language-model families, answer mass consistently declines while discrimination among old answers usually improves: the model becomes less likely to produce answers that it can still distinguish correctly. Decomposing Adam updates reveals opposing contributions to this loss of answer mass. Over training, accumulated history favours leakage, while the current gradient opposes it. Resolving history by age shows that the harmful contributions come mainly from older gradients of the new task, whereas recent gradients tend to protect the old answers. Changes in history's effect are dominated by its orientation relative to the old-task gradient. Interventions that reset momentum while matching the initial update norm establish that stored history affects retention, with state-dependent immediate effects and lower final old-task loss over longer Adam continuations, mainly through recovered answer mass. Finally, integration along finite updates shows that most sampled large loss increases are captured by local projections, while curvature along history amplifies some events. Together, these findings reveal how an optimiser's memory can erode learned behaviour even as its current gradient acts to preserve it. 

---
# Streaming Multi-Track Timeline Control for 3D Human Motion Generation 

**Authors**: Yangsong Zhang, Anujith Muraleedharan, Rikhat Akizhanov, Gül Varol, Fabio Pizzati, Ivan Laptev  

**Link**: [PDF](https://arxiv.org/pdf/2610.03873)  

**Abstract**: Text-driven human motion generation has advanced substantially, yet most methods assume instructions are available before synthesis. Interactive applications require responding to new instructions while continuing ongoing actions, such as answering a phone while walking. Existing approaches address streaming generation or simultaneous composition without explicitly combining streaming instruction arrival with independently timed, overlapping actions. We introduce streaming multi-track timeline control and propose TimelineControl to incorporate new instructions alongside ongoing actions. Interval-aware conditioning preserves instruction timing, while causal part-structured representations and part-aware denoising coordinate concurrent actions across body regions. We also construct TimelineMotion, a dataset with overlapping instruction intervals and body-part annotations. Experiments on TimelineMotion and MTT demonstrate improved semantic alignment and temporal adherence over evaluated streaming baselines, including models retrained on the same data. Ablations and human evaluations validate our design, complemented by spatial conditioning and humanoid execution demonstrations. Our code, data and models will become publicly available. 

---
# GOTT: Object-centric Dexterous Manipulation with a Reusable Cross-Embodiment Primitive 

**Authors**: Yulin Liu, Lai Wei, Yen-Jen Wang, Akash Sharma, Pieter Abbeel, Henrik I. Christensen, Haozhi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.03861)  

**Abstract**: Foundation models and large-scale human data provide rich sources of manipulation intent, but translating this intent into multi-fingered robot behavior remains difficult. Dexterous hands still lack a reusable low-level primitive that reliably establishes contact across tasks and embodiments. We propose GOTT, a reach-acquire-move framework built around a single cross-embodiment contact-acquisition primitive. Given a robot-agnostic object trajectory and a reach specification, GOTT first brings the hand near a task-relevant contact region. The shared closed-loop primitive then establishes stable contact from this approximate initialization, and a pose-conditioned controller tracks the desired object motion. Reach specifications may come from future-aware planning, external models, or human demonstrations, while the primitive and tracking backend remain unchanged. Simulation and real-world experiments show that GOTT is able to establish robust contact across diverse objects, arm-hand platforms, and seen and unseen hand morphologies. It also consistently improves end-to-end task success over open-loop grasp execution. 

---
# Retrieval-Centric Deep Learning in Growing Nonparametric Neural Networks 

**Authors**: Maximilian Schlegel, Rajai Nasser, Seijin Kobayashi, Yanick Schimpf, Oliver Sieberling, Robert Obryk, Kazuki Irie, João Sacramento, Johannes von Oswald  

**Link**: [PDF](https://arxiv.org/pdf/2610.03858)  

**Abstract**: We investigate a general-purpose layer for deep learning that, instead of compressing arbitrary-size training data into fixed-size weight matrices, stores a new pair of key-value representations for every data point during training, and retrieves and recombines these representations through an attention mechanism at inference time - resulting in a growing neural net (NN). While Irie et al. (arXiv:2202.05798) have put forward this perspective from the classic duality expressing any linear layer in a deep NN trained by gradient descent as linear attention (LA) over the training data points, replacing LA by more powerful attention functions, as they suggest, turns out to be non-trivial: we show that naively applying learning rules from the LA case to advanced kernels does not lead to principled optimization. Here we fill this gap and develop functional gradient-based learning rules for kernelized attention layers, based on radial basis function (RBF) and softmax-like kernels - establishing the principled "retrieval-centric deep learning" (RCDL) paradigm. Empirically, we demonstrate the promising performance and learning-efficiency of RCDL on image classification and synthetic teacher-student learning tasks. Moreover, we show that replacing LA in the dual form of NNs by advanced LA variants, namely MesaNet/DeltaNet, yields a formal connection to recently proposed optimizers for conventional fixed-size NNs, offering a novel perspective on deep learning optimization. 

---
# Beware EviLLM: Enabling Vulnerability Injection via Large Language Models 

**Authors**: Zeezoo Ryu, Simon Chung, Muhammad Faraz Karim, Anna Raymaker, Karan Singh Jodha, Yash Chaturvedi, Sukarno Mertoguno  

**Link**: [PDF](https://arxiv.org/pdf/2610.03857)  

**Abstract**: Advances in large language models (LLMs) have enabled AI-driven code generation from natural language specifications, introducing new attack surfaces for injecting vulnerabilities into software. Prior work has studied this problem only in benign settings where vulnerabilities are introduced inadvertently, or under unconventional threat models where the LLM itself is malicious (backdooring) or the user is the attacker (jailbreaking). In this paper, we study a more realistic threat model: a third-party adversary, with capabilities comparable to existing cybercriminals, compromises the AI code generation pipeline to deliberately introduce vulnerabilities. We call this the EviLLM attack. We have implemented two instances of EviLLM, each of which only requires the underlying LLM to be accessed through a compromised account or browser, and can inject vulnerabilities from 13 CWE classes. As we show in our feasibility study, both attack vectors are already used to implement many existing cyberattacks. Our user study shows that 7 out of 8 and 10 out of 13 participants did not notice the vulnerabilities injected by the two instances of EviLLM, and 13 out of 21 participants "rarely" or "never" considered the risk of an attack like EviLLM. 

---
# SYNLAT: Syntax-Aligned Text-Latent Compression for Chain-of-Thought Reasoning 

**Authors**: Yifeng Zhao, Hongjun Yu, Shibo Wang, Yunjiao Zhou, Zixiao Zhu, Zhipeng Ning, Kezhi Mao, Junlang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2610.03839)  

**Abstract**: Long chain-of-thought (CoT) traces impose substantial output-token costs. Under constrained budgets, compression must preserve answer-critical information, making boundary placement central. Token-level and fixed-length boundaries can fragment coherent spans such as phrases, formulas, and local derivations, whereas step-level boundaries can bind content requiring different compression actions. We introduce SynLat, a text-latent CoT framework that aligns compression boundaries with syntactic structure through non-overlapping Syntax-Aligned Units (SAUs). An answer-conditioned Teacher constructs progressive KEEP/LATENT targets for a single compression-conditioned Student, which generates mixed reasoning from only the question and requested compression level at inference. Across two Qwen3 Student scales, Standard-CoT and Long-CoT groups, and three compression levels, SynLat matches or exceeds the strongest evaluated baseline in all 12 task-group aggregates and strictly leads in 11 under the reported achieved-CR selection protocol. Overall gains reach 3.6/2.6 points at MEDIUM and 7.0/5.5 points at HIGH for Qwen3-8B/14B, with larger advantages under stronger compression, particularly on Long-CoT groups. 

---
# Distribution Matching Evolutionary Algorithms for Rare Event Sampling 

**Authors**: Yonatan Gideoni, Yarin Gal  

**Link**: [PDF](https://arxiv.org/pdf/2610.03833)  

**Abstract**: A novel discovery is one which is both useful and surprising: a generative model's output is a useful discovery if it has a low probability of being generated (it's surprising) and a high reward (it's useful). Global optimization can directly increase the probability of sampling high rewards but typically requires updating model weights. Such gradient based optimization is expensive and bars using capable closed-source models. Instead, modern search methods for discovery sacrifice the global target, and use evolutionary algorithms with local reward maximizing objectives, permitting the search to focus only on high probability samples. In this paper, we interpret various evolutionary algorithms as approximate Markov Chain Monte Carlo, an optimization-free method to sample from complex distributions. This interpretation allows developing Distribution Matching Evolutionary Algorithms (DME), a class of search methods which sample from a global target distribution without updating weights. Empirically, DME has a higher sample efficiency than existing methods on problems requiring many samples to find a solution. 

---
# The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer 

**Authors**: Saman Rahbar  

**Link**: [PDF](https://arxiv.org/pdf/2610.03827)  

**Abstract**: Researchers often support the claim that a model shares structure with the brain, or across languages, by reporting a similarity score. We ask what such a score reads when the shared structure is absent, or when the tool that measures it does not work. We check two settings, and in both the score is not what it appears. First, a probe trained to tell grammatical from ungrammatical sentences in one language transfers worse to more distant languages, the usual evidence for shared structure. But the probe itself gets worse along the same axis: in four of seventeen languages it performs at chance, so 64 of 272 language pairs are scored with a tool that does not work. Dropping those languages halves the strength of the relationship, but they are also the most distant, and this design cannot separate the two effects. Counting matters too: the same data give p = 0.0006 when the 272 pairs are treated as independent and p = 0.155 when the seventeen languages are, which is the correct unit. Second, training a language model to match human brain responses raises its similarity score from 0.10 to 0.34, against a ceiling of 0.54. A model trained on a target whose correspondence to the brain was destroyed still scores 0.31, so only 0.028 to 0.068 of the rise is specific to the brain. With no model at all, a destroyed target already sits at 0.204 from the real one, a floor that tracks the target's rank divided by the number of sentences. Finally, steering a language along its own direction works (+6.2 over a random direction in sixteen of seventeen languages) yet shows no effect that varies with language distance. Before asking whether a correspondence helps, ask how much of the score would survive without it. 

---
# From Requirements to Attack Trees: Grounded LLM Agents for Design-Time Security Review 

**Authors**: Akash Iyer, Taha Demirkan, Keerthi Koneru, Aaryan Siddharthan, Sheethal Kumar, Ramesh Radhakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2610.03820)  

**Abstract**: Design-level security weaknesses can arise from requirements, trust assumptions, missing controls, and data flows before implementation begins. Existing security practices often identify these issues after code is written. We present a multi-agent LLM framework for design-time security analysis from product requirement documents and architecture diagrams.
The proposed framework parses architecture diagrams into graph representations, generates misuse and failure cases, constructs attack trees, checks governance and compliance gaps, recommends mitigations, assigns enterprise security-domain tags, and produces a candidate revised architecture recommendation for expert review. The framework does not retrieve from Common Weakness Enumeration (CWE) databases at inference time. Instead, it analyzes system behavior, trust boundaries, component interactions, and data-flow assumptions. Misuse cases act as intermediate representations that link findings to system components and attack paths, while a validation and refinement loop filters unsupported findings and improves grounding, traceability, and actionability.
We evaluate the framework on a Microsoft reference-labeled threat-modeling example, labeled synthetic PRD--architecture pairs, and two open-ended systems: Berty and Gas Town. The reference-labeled case supports threat-recovery and actionability analysis, while the open-ended cases evaluate validity, noise, traceability, actionability, redundancy, and attack-tree quality. Results show that architecture-informed, misuse-driven reasoning improves review quality compared with single-shot and ablation baselines.
Keywords: LLM Multi-Agent Systems, Design-Time Security, Threat Modeling, Vulnerability Discovery, Architecture Diagrams, Security Analysis, Misuse Case Derivation, Attack Trees, Iterative Reasoning, Security Governance. 

---
# WAMJET: A Harness for World Action Model Acceleration 

**Authors**: Le Chen, Lixin Liu, Jan Schneider, Zeju Qiu, Simon Guist, Bernhard Schölkopf, Dieter Büchler  

**Link**: [PDF](https://arxiv.org/pdf/2610.03797)  

**Abstract**: World Action Models (WAMs) leverage pretrained video foundation models for robot manipulation, but their large backbones and video-action co-prediction are expensive. Although existing acceleration techniques offer many ways to reduce this cost, selecting and composing them requires substantial engineering for each model and hardware platform. To tackle this bottleneck, we present WAMJET, an agentic harness that accelerates WAM inference by equipping coding agents with reusable optimization guidance and measurement and validation tools. WAMJET follows a bottleneck-driven workflow where the agent profiles inference, modifies targeted code, validates effects, and iteratively refines the acceleration stack as bottlenecks shift, while preserving action quality. Experiments span six WAMs, three coding agents, and two GPU architectures. WAMJET achieves up to 9.95x lossless speedup over upstream implementations. Approximation and hardware-aware optimization yield additional latency reductions, with comparable success rates. The results show that WAMJET can produce effective acceleration stacks for WAM deployment. 

---
# ProsaBuddy: Assisting Mechanized Real-Time Schedulability Analysis with LLM-based Agents 

**Authors**: Junyi Liu, Tianchi Ren, Fei Guan, Xu Jiang, Zhe Jiang, Wang Yi, Nan Guan  

**Link**: [PDF](https://arxiv.org/pdf/2610.03796)  

**Abstract**: Rigorous schedulability analysis is essential for the design of hard real-time systems, yet errors in pen-and-paper proofs threaten the safety of critical applications. The Prosa initiative addresses this by offering a foundation for building machine-checkable schedulability analysis proofs in the Rocq proof assistant. However, the substantial time and expertise required to construct such proofs remain a major barrier for wider adoption of Prosa. This work presents ProsaBuddy, an LLM?based agent system designed to lower the effort needed to develop mechanized real-time schedulability proofs. ProsaBuddy employs a ReAct loop with retrieval over the Prosa codebase, access to Rocq tools and optional human-written hints. It uses a subgoal?delegation architecture, decomposing a lemma into subgoals and dispatches them to subagents for proof. We evaluate ProsaBuddy on a mini benchmark drawn from real-time scheduling literature. Experiment results show that ProsaBuddy significantly outper?forms state-of-the-art LLM-based Rocq automated proving agent systems and a general coding agent OpenCode 

---
# OceanMind: A multi-agent AI system for ocean diagnosis 

**Authors**: Fan Zhang, Weicong Cheng, Yuheng Chen, Hiuseut Kung, Ying Zhang, Aixi Han, Quanjia Zhong, Can Yang, Jianping Gan  

**Link**: [PDF](https://arxiv.org/pdf/2610.03780)  

**Abstract**: Time-dependent, three-dimensional (3D) oceanic multi-variables define coherent states of the evolving ocean to facilitate ocean diagnosis and advance ocean science to better inform environmental and hazard management. However, extracting quantitative evidence from these variables requires substantial and complex analytical effort. We introduce OceanMind, a multi-agent AI system that directly couples large language models (LLMs) with comprehensive time-dependent 3D ocean states for swift and effective diagnosis. OceanMind organizes the analytical process into four coordinated complexity stages: Query Routing, Skill-based Planning, Tool Execution, and Evidence-based Summary Generation. Specialized agents interpret user requests, construct and execute multi-step computational workflows, and synthesize quantitative evidence. To ensure reliable workflow construction, 63 reusable ocean-specific analysis skills serve as procedural manuals that guide the LLM agent in selecting data, conducting diagnostics, and applying analytical tools. With reflection and replanning mechanisms that use execution feedback to repair invalid plans, OceanMind ensures reliable analysis workflows across diverse needs. On a benchmark of 240 computational-workflow queries spanning the four stages, OceanMind outperformed general ReAct agents with the same registered tool pool, achieving relative improvements of 41.2% in effectiveness and 21.5% in efficiency. Beyond the benchmark, OceanMind reproduced published oceanic diagnostics, validated hypotheses, and supported environmental decision-making over global oceans. Overall, OceanMind advances LLMs by integrating them with time-dependent 3D ocean analysis, enabling scientific interpretation and enhancing formulation of environmental policies based on quantitative ocean evidence. 

---
# Do Motion Tokenizers for Co-Speech Gesture Generation Encode Gesture Semantics? 

**Authors**: Varsha Suresh, Divij Jain, Jia Liu, M. Hamza Mughal, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2610.03765)  

**Abstract**: Discrete motion tokenizers encode motion as atomic units and are widely used for co-speech gesture generation. It remains unclear which motion properties, especially those relevant to gesture semantics, are recoverable from these codebooks. We probe a reconstruction-trained codebook using 19 co-speech gesture descriptors spanning from raw motion to abstract communicative function. Results show that geometry and handedness are readily decodable from token embeddings, while motion category is only weakly decoded despite showing systematic differences in discrete code usage. This gap between reconstruction quality and descriptor decodability suggests that reconstruction objectives alone do not guarantee that gesture semantics are captured, and that evaluating codebooks on such properties can guide the design of more semantic motion tokenizers. 

---
# BridgeCast: Bridging Ocean Wave Forecasts to Reanalysis via Flow Matching with Exogenous Variables 

**Authors**: Siyu Gan, Dongsheng Luo, Kunxiaojia Yuan, Dongjin Song, Jingchao Ni  

**Link**: [PDF](https://arxiv.org/pdf/2610.03759)  

**Abstract**: Ocean wave forecasting is essential for maritime safety, offshore operations, and coastal resilience, yet remains challenging due to systematic biases in physics-based models. Physical models, while widely used, rely on approximations and parameterizations that limit their accuracy under complex ocean-atmosphere conditions. To enhance ocean wave forecasting, we propose BridgeCast, within a physics-AI hybrid framework for bias correction. BridgeCast is a probabilistic model based on conditional flow matching (CFM) that learns to transform physical model forecasts into reanalysis-like fields. It treats physical forecasts as corrupted observations and employs a continuous-time generative process to bridge their distribution toward that of reanalysis data. BridgeCast is parameterized by a Transformer-based architecture that enables spatiotemporal modeling, incorporation of exogenous atmospheric variables, and flexible inference via both ordinary and stochastic differential equation formulations. Extensive experiments on real-world datasets demonstrate that BridgeCast consistently outperforms state-of-the-art baselines across regions and forecast lead times. 

---
# Global Evaluation of AI and NWP Precipitation Forecasts During Atmospheric River Events 

**Authors**: Marina Vicens-Miquel, Taylor Mandelbaum, Amy McGovern, Aaron J. Hill, Daniel Rothenberg  

**Link**: [PDF](https://arxiv.org/pdf/2610.03758)  

**Abstract**: Atmospheric rivers (ARs) produce many of the world's most extreme precipitation events and hydrometeorological hazards. Although artificial intelligence weather prediction (AIWP) models have demonstrated skill comparable to or exceeding numerical weather prediction (NWP) systems for large-scale atmospheric variables, their ability to forecast AR-related precipitation remains insufficiently characterized globally. Here, we evaluate 24-hour precipitation forecasts from the Global Forecast System (GFS), Global Ensemble Forecast System (GEFS), GraphCast, and Artificial Intelligence Forecasting System (AIFS) from Day 1 through Day 10 globally and across North America, Europe, and Australia and New Zealand. Using the Extreme Weather Bench framework, forecasts are evaluated against Integrated Multi-satellitE Retrievals for GPM (IMERG) observations using measures of precipitation magnitude, spatial structure, and localization. GraphCast and AIFS exhibit greater spatial skill than GFS and GEFS, particularly for heavy precipitation and at longer lead times, and better preserve the spatial organization of AR-related precipitation through Day 10. However, this improved spatial skill does not translate into accurate precipitation magnitudes. AIWP models tend to overpredict moderate-to-heavy accumulations while underpredicting the heaviest precipitation at longer lead times, whereas NWP systems develop pronounced dry biases. These results reveal distinct strengths and limitations of AIWP for high-impact precipitation forecasting and provide a reproducible benchmark. 

---
# LoRA Adaptation Strength in Aurora-WRF: Trade-offs in Regional Heavy-Rainfall Forecasting 

**Authors**: Boyan Liu, Xiaoyuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.03747)  

**Abstract**: We investigate how parameter-efficient adaptation of an atmospheric foundation model affects downstream regional precipitation in a controlled Aurora-WRF coupling experiment over the Beijing-Tianjin-Hebei region. A single-step, precipitation-weighted LoRA adaptation of AuroraPretrained is trained on May-September 2020-2021 and selected using 2022 validation data. Five inference-time adaptation strengths are coupled to an unchanged 9-km WRF configuration and compared with a GFS-driven control under common ERA5 initialization and auxiliary-field rules. Evaluation uses 17 initializations in 2023, hourly ERA5 precipitation, and conservative remapping to a fixed 356-cell, 0.25-degree verification grid. Heavy rainfall is defined exclusively as at least 50 mm in a continuous 24-hour window, advanced hourly. For window starts from 0 to 48 hours, unadapted Aurora attains the highest mean critical success index (CSI), 0.216, compared with 0.144 for the GFS control. An intermediate LoRA strength of 0.75 reduces the Aurora-driven 24-hour precipitation mean squared error by 30.7% and brings pooled frequency Bias from 1.510 to 1.024, but lowers CSI to 0.182. GFS retains the lowest whole-domain error. These results characterize a trade-off between rainfall detection, frequency calibration, and intensity error, showing the importance of evaluating adaptation strength against multiple downstream criteria within a common coupling framework. 

---
# Logit-Aware MIMO AirComp for Distributed Mixture-of-Experts LLM Inference over Wireless Edge Networks 

**Authors**: Lyutianyang Zhang, Yunjian Jia, Liu Cao, Dengke Wang, Jinke Ren, Shuguang Cui  

**Link**: [PDF](https://arxiv.org/pdf/2610.03741)  

**Abstract**: Distributed mixture-of-experts (MoE) inference is a promising architecture for deploying large language models (LLMs) at wireless edge networks because sparse experts can be placed across coordinated base stations (BSs), while the anchor node and user equipment (UE) can offload LLM inference tasks to BSs. The communication bottleneck is the MoE aggregation, where the anchor BS must recover a weighted sum of selected expert outputs before each decoding step. Over-the-air computation (AirComp) is well matched to this operation because the wireless multiple-access channel naturally superposes simultaneous transmissions. However, conventional AirComp minimizes communication distortion, whereas MoE aggregation errors have unequal impact on LLM outputs. We propose a logit-aware MIMO AirComp framework that estimates local logit sensitivity as block-level weights and jointly optimizes receive combiners and BS precoders under per-BS power constraints. We also develop an alternating algorithm that protects decoding decisions from aggregation perturbations. Using OpenCompass, we evaluate Qwen3-30B-A3B-Instruct-2507-FP8 on GSM8K and ARC-Challenge. At 30 dB aggregation SNR, perturbed Qwen3 retains 99.3% of clean GSM8K accuracy and 94.8% of clean ARC-Challenge accuracy. In a direct ARC-Challenge closed-loop audit over 295 examples, SW-AirComp achieves 43.73% accuracy at 30 dB and 26.78% at 20 dB, outperforming unweighted, matched-filter, and zero-forcing AirComp. In the wireless simulator, SW-AirComp reduces decision-relevant aggregation distortion; at 20 dB, its final weighted-sum mean-squared error is 53.5% lower than unweighted AirComp under the same channels, samples, and sensitivity weights. Logit-RMSE and logit-gap diagnostics further show that the gain comes from reducing output-logit perturbation and lowering the risk of top-token changes. 

---
# Efficient Analog-Initialized Latent Transport for Spatially Coherent Probabilistic Downscaling from Global to Kilometer Scales 

**Authors**: Ophélia Miralles  

**Link**: [PDF](https://arxiv.org/pdf/2610.03736)  

**Abstract**: Probabilistic downscaling must represent kilometer-scale structure left unresolved by a deterministic regional prediction. We introduce an analog-initialized latent transport method that uses historical regional residuals as a meteorologically informed empirical source. A frozen graph neural network predicts the deterministic state from the Integrated Forecasting System (IFS). For each input, residuals from 20 training dates with similar IFS conditions initialize the ensemble. A conditional flow-matching model transports these fields in a carefully designed 65-factor, 100,000-node latent space, and a grid decoder reconstructs 81 atmospheric channels including precipitation at 2.5km spacing. On a season- and cycle-balanced 2025 test, ensemble CRPS improves over deterministic mean absolute error for all 81 variables. Matched analogs improve point and spatial errors over Gaussian initialization and reduce spectral error compared with covariance matching. Comparisons with released CorrDiff, a local CorrDiff-mini model and full grid EDM provide context on the 2023 validation sample. Total training of the downscaling model takes approximately 12h on one NVIDIA H200, and inference takes about 12s per 81-field member. The frozen networks also generate fields over France when supplied with local static data and a high-resolution IFS-derived analog bank. The method combines empirical initialization with whole-domain latent transport for computationally practical regional ensembles. 

---
# Response Variability and Stability in Human Reasoning 

**Authors**: Clemens Bombach, Rajmadan Lakshmanan, Marco Ragni  

**Link**: [PDF](https://arxiv.org/pdf/2610.03008)  

**Abstract**: Understanding how humans reason -- and how reasoning responses vary across tasks and individuals -- remains a core challenge for modeling and explanation in cognitive science. We investigate the stability of response patterns within reasoners and whether variation in these patterns can be used to predict learning effects. We introduce a formal, geometry-based method to quantify distances between individual reasoning patterns and their internal variability, grounded in heuristic theories. The proposed framework is tested against experimental data via generalized linear mixed-effects models and clustering, where we find that our proposed variation measure interacts with correctness to predict performance gains. Moreover, we find that reasoning patterns are stable over time within the same reasoner. The method is general enough to be applied to other reasoning domains. 

---
# Beyond Private Training: The New Landscape of AI Privacy 

**Authors**: Sean Culatana, Kang Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.19456)  

**Abstract**: Retrieval-augmented systems increasingly rely on vector indexes that may retain deleted items in their search graph. Existing deletion interfaces can prevent deleted identifiers from appearing in returned results while still computing distances to their embeddings during graph traversal. We formalize this distinction as output safety versus traversal safety, and introduce TSD-AUDIT, a framework for auditing and enforcing traversal-safe deletion in graph-based approximate nearest-neighbor retrieval. On Faiss IndexHNSWFlat, native filtering leaves the number of distance computations unchanged relative to unfiltered search; at a 70% deletion rate, trace-faithful replay detects deleted-vector scoring in all 100 audited queries. Code inspection of hnswlib's mark_deleted path reveals the same scoring-before-liveness pattern. TSD-AUDIT enforces an alive-before-scoring invariant, repairs connectivity using only live candidates, and emits per-query scored-trace certificates that an independent verifier can check against the deletion snapshot. Under region-targeted deletion, TSD-AUDIT improves Recall@10 over native filtering by 4.3--42.2 percentage points across deletion fractions from 0.5 to 0.9, while remaining comparable under random deletion. These results show that output-only deletion audits can miss process-level exposure: auditing deletion in vector retrieval requires accounting for the vectors scored during search, not only the identifiers returned. 

---
# Using Process Mining to Generate AI Agents from Software Engineering Process Records 

**Authors**: Saimir Bala, Fabiana Fournier, Lior Limonad, Andreas Metzger  

**Link**: [PDF](https://arxiv.org/pdf/2607.04948)  

**Abstract**: Integrating AI agents into Software Engineering (SE) raises an important challenge: how can we specify and realize AI agents that work effectively alongside humans in hybrid SE teams? Determining the right granularity and separation of concerns for such agents is non-trivial. Coarse-grained agents may introduce unmanageable complexity, whereas micro-agents may create severe coordination overhead. Moreover, existing multi-agent SE frameworks typically rely on predefined role structures and do not account for project-specific characteristics or process adaptations. We address this by combining object-centric, imperative, and declarative process mining. Using event logs extracted from software repositories, our approach discovers project-specific agent roles using a predefined SE role vocabulary grounded in repository behavior and generates matching agent specifications and implementations. As proof-of-concept, we applied our approach to a well-established open-source project. We performed functional tests and an exploratory user study to determine how well the generated AI agent specifications are aligned with human expectations. 

---
# AI Agent Pull Requests on GitHub: Frequency, Structure, and Merge Conflict Rates 

**Authors**: George Xu, Arjun Subramanian, Nithilan Karthik  

**Link**: [PDF](https://arxiv.org/pdf/2607.04697)  

**Abstract**: AI coding agents may generate and submit Pull Requests (PRs) to the same repository at the same time. However, research concerning the extent of concurrent submission by AI coding agents to a common repository does not exist. This paper uses the AIDev-pop dataset (33,596 PRs in 2,807 repositories) to provide the first empirical examination of the prevalence of concurrent submission using PRs authored by agents. We report that when considering exact temporal overlap, 40.2% of repositories contain co-active agent-authored PR pairs; further, the co-active pairs account for 79.4% of all PRs generated by an AI agent. When we examine co-activity within a one week collaboration window, the percentages are increased to 53.4% and 95.0%, respectively. For the majority of the co-active PR pairs (underlying the vast majority of which are intra-agent authored), both PRs were authored by the same agent, while only 0.5% of co-active pairs were cross-agent, and occurred in only 122 out of 2807 total repositories examined (or approximately 4.3%). Additionally, we replayed actual three way git merges on 747 unique co-active pairs (one per repository), and computed the percentage of textual conflict encountered during the merge operation to combine the two PRs in each pair. We observed that the percentage of textual conflict encountered was significantly higher for cross-agent pairs compared to intra-agent pairs: 41.7% vs. 19.8%, respectively, with non-overlapping 95% confidence intervals. Lastly, we developed a classification system based on the detection of conflict reported by git, and determined that the majority of conflicts resulted from modifications to source code files (84.4% of conflicted files) and not dependency manifest files; further, nearly 42% of conflicts we observed were structural (i.e., modify/delete or add/add). 

---
# Orchestrating Specialized Agents for Trustworthy Enterprise RAG 

**Authors**: Xincheng You, Qi Sun, Neha Bora, Huayi Li, Shubham Goel, Kang Li, Sean Culatana  

**Link**: [PDF](https://arxiv.org/pdf/2601.18267)  

**Abstract**: Retrieval-Augmented Generation (RAG) shows promise for enterprise knowledge work, yet it often underperforms in high-stakes decision settings that require deep synthesis, strict traceability, and recovery from underspecified prompts. One-pass retrieval-and-write pipelines frequently yield shallow summaries, inconsistent grounding, and weak mechanisms for completeness verification. We introduce ADORE (Adaptive Deep Orchestration for Research in Enterprise), an agentic framework that replaces linear retrieval with iterative, user-steered investigation coordinated by a central orchestrator and a set of specialized agents. ADORE's key insight is that a structured Memory Bank (a curated evidence store with explicit claim-evidence linkage and section-level admissible evidence) enables traceable report generation and systematic checks for evidence completeness. Our contributions are threefold: (1) Memory-locked synthesis - report generation is constrained to a structured Memory Bank (Claim-Evidence Graph) with section-level admissible evidence, enabling traceable claims and grounded citations; (2) Evidence-coverage-guided execution - a retrieval-reflection loop audits section-level evidence coverage to trigger targeted follow-up retrieval and terminates via an evidence-driven stopping criterion; (3) Section-packed long-context grounding - section-level packing, pruning, and citation-preserving compression make long-form synthesis feasible under context limits. Across our evaluation suite, ADORE ranks first on DeepResearch Bench (52.65) and achieves the highest head-to-head preference win rate on DeepConsult (77.2%) against commercial systems. 

---
