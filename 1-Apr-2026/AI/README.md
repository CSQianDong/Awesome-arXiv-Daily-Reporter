# The Triadic Cognitive Architecture: Bounding Autonomous Action via Spatio-Temporal and Epistemic Friction 

**Authors**: Davide Di Gioia  

**Link**: [PDF](https://arxiv.org/pdf/2603.30031)  

**Abstract**: Current autonomous AI agents, driven primarily by Large Language Models (LLMs), operate in a state of cognitive weightlessness: they process information without an intrinsic sense of network topology, temporal pacing, or epistemic limits. Consequently, heuristic agentic loops (e.g., ReAct) can exhibit failure modes in interactive environments, including excessive tool use under congestion, prolonged deliberation under time decay, and brittle behavior under ambiguous evidence. In this paper, we propose the Triadic Cognitive Architecture (TCA), a unified mathematical framework that grounds machine reasoning in continuous-time physics. By synthesizing nonlinear filtering theory, Riemannian routing geometry, and optimal control, we formally define the concept of Cognitive Friction. We map the agent's deliberation process to a coupled stochastic control problem where information acquisition is path-dependent and physically constrained. Rather than relying on arbitrary heuristic stop-tokens, the TCA uses an HJB-motivated stopping boundary and instantiates a rollout-based approximation of belief-dependent value-of-information with a net-utility halting condition. Through empirical validation in a simulated Emergency Medical Diagnostic Grid (EMDG), we demonstrate that while greedy baselines over-deliberate under latency and congestion costs, the triadic policy reduces time-to-action while improving patient viability without degrading diagnostic accuracy in this environment. 

---
# Extending MONA in Camera Dropbox: Reproduction, Learned Approval, and Design Implications for Reward-Hacking Mitigation 

**Authors**: Nathan Heath  

**Link**: [PDF](https://arxiv.org/pdf/2603.29993)  

**Abstract**: Myopic Optimization with Non-myopic Approval (MONA) mitigates multi-step reward hacking by restricting the agent's planning horizon while supplying far-sighted approval as a training signal~\cite{farquhar2025mona}. The original paper identifies a critical open question: how the method of constructing approval -- particularly the degree to which approval depends on achieved outcomes -- affects whether MONA's safety guarantees hold. We present a reproduction-first extension of the public MONA Camera Dropbox environment that (i)~repackages the released codebase as a standard Python project with scripted PPO training, (ii)~confirms the published contrast between ordinary RL (91.5\% reward-hacking rate) and oracle MONA (0.0\% hacking rate) using the released reference arrays, and (iii)~introduces a modular learned-approval suite spanning oracle, noisy, misspecified, learned, and calibrated approval mechanisms. In reduced-budget pilot sweeps across approval methods, horizons, dataset sizes, and calibration strategies, the best calibrated learned-overseer run achieves zero observed reward hacking but substantially lower intended-behavior rates than oracle MONA (11.9\% vs.\ 99.9\%), consistent with under-optimization rather than re-emergent hacking. These results operationalize the MONA paper's approval-spectrum conjecture as a runnable experimental object and suggest that the central engineering challenge shifts from proving MONA's concept to building learned approval models that preserve sufficient foresight without reopening reward-hacking channels. Code, configurations, and reproduction commands are publicly available. this https URL 

---
# Structured Intent as a Protocol-Like Communication Layer: Cross-Model Robustness, Framework Comparison, and the Weak-Model Compensation Effect 

**Authors**: Peng Gang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29953)  

**Abstract**: How reliably can structured intent representations preserve user goals across different AI models, languages, and prompting frameworks? Prior work showed that PPS (Prompt Protocol Specification), a 5W3H-based structured intent framework, improves goal alignment in Chinese and generalizes to English and Japanese. This paper extends that line of inquiry in three directions: cross-model robustness across Claude, GPT-4o, and Gemini 2.5 Pro; controlled comparison with CO-STAR and RISEN; and a user study (N=50) of AI-assisted intent expansion in ecologically valid settings. Across 3,240 model outputs (3 languages x 6 conditions x 3 models x 3 domains x 20 tasks), evaluated by an independent judge (DeepSeek-V3), we find that structured prompting substantially reduces cross-language score variance relative to unstructured baselines. The strongest structured conditions reduce cross-language sigma from 0.470 to about 0.020. We also observe a weak-model compensation pattern: the lowest-baseline model (Gemini) shows a much larger D-A gain (+1.006) than the strongest model (Claude, +0.217). Under the current evaluation resolution, 5W3H, CO-STAR, and RISEN achieve similarly high goal-alignment scores, suggesting that dimensional decomposition itself is an important active ingredient. In the user study, AI-expanded 5W3H prompts reduce interaction rounds by 60 percent and increase user satisfaction from 3.16 to 4.04. These findings support the practical value of structured intent representation as a robust, protocol-like communication layer for human-AI interaction. 

---
# Physiological and Semantic Patterns in Medical Teams Using an Intelligent Tutoring System 

**Authors**: Xiaoshan Huang, Conrad Borchers, Jiayi Zhang, Susanne P. Lajoie  

**Link**: [PDF](https://arxiv.org/pdf/2603.29950)  

**Abstract**: Effective collaboration requires teams to manage complex cognitive and emotional states through Socially Shared Regulation of Learning (SSRL). Physiological synchrony (i.e., longitudinal alignment in physiological signals) can indicate these states, but is hard to interpret on its own. We investigate the physiological and conversational dynamics of four medical dyads diagnosing a virtual patient case using an intelligent tutoring system. Semantic shifts in dialogue were correlated with transient physiological synchrony peaks. We also coded utterance segments for SSRL and derived cosine similarity using sentence embeddings. The results showed that activating prior knowledge featured significantly lower semantic similarity than simpler task execution. High physiological synchrony was associated with lower semantic similarity, suggesting that such moments involve exploratory and varied language use. Qualitative analysis triangulated these synchrony peaks as ``pivotal moments'': successful teams synchronized during shared discovery, while unsuccessful teams peaked during shared uncertainty. This research advances human-centered AI by demonstrating how biological signals can be fused with dialogues to understand critical moments in problem solving. 

---
# ScoringBench: A Benchmark for Evaluating Tabular Foundation Models with Proper Scoring Rules 

**Authors**: Jonas Landsgesell, Pascal Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2603.29928)  

**Abstract**: Tabular foundation models such as TabPFN and TabICL already produce full predictive distributions yet prevailing regression benchmarks evaluate them almost exclusively via point estimate metrics RMSE R2 These aggregate measures often obscure model performance in the tails of the distribution a critical deficit for high stakes decision making in domains like finance and clinical research where asymmetric risk profiles are the norm We introduce ScoringBench an open benchmark that computes a comprehensive suite of proper scoring rules like CRPS CRLS Interval Score Energy Score weighted CRPS and Brier Score alongside standard point metrics providing a richer picture of probabilistic forecast quality We evaluate realTabPFNv2.5 fine tuned with different scoring rule objectives and TabICL relative to untuned realTabPFNv2.5 across a suite of regression benchmarks Our results confirm that model rankings depend on the chosen scoring rule and that no single pretraining objective is universally optimal This demonstrates that for applications sensitive to extreme events the choice of evaluation metric is as much a domain specific requirement as the data itself ScoringBench is available at this https URL A live preview of the current leaderboard is available at this https URL The leaderboard is maintained via git pull requests to ensure transparency traceability agility and reproducibility 

---
# Uncertainty Gating for Cost-Aware Explainable Artificial Intelligence 

**Authors**: Georgii Mikriukov, Grégoire Montavon, Marina M.-C. Höhne  

**Link**: [PDF](https://arxiv.org/pdf/2603.29915)  

**Abstract**: Post-hoc explanation methods are widely used to interpret black-box predictions, but their generation is often computationally expensive and their reliability is not guaranteed. We propose epistemic uncertainty as a low-cost proxy for explanation reliability: high epistemic uncertainty identifies regions where the decision boundary is poorly defined and where explanations become unstable and unfaithful. This insight enables two complementary use cases: `improving worst-case explanations' (routing samples to cheap or expensive XAI methods based on expected explanation reliability), and `recalling high-quality explanations' (deferring explanation generation for uncertain samples under constrained budget). Across four tabular datasets, five diverse architectures, and four XAI methods, we observe a strong negative correlation between epistemic uncertainty and explanation stability. Further analysis shows that epistemic uncertainty distinguishes not only stable from unstable explanations, but also faithful from unfaithful ones. Experiments on image classification confirm that our findings generalize beyond tabular data. 

---
# C-TRAIL: A Commonsense World Framework for Trajectory Planning in Autonomous Driving 

**Authors**: Zhihong Cui, Haoran Tang, Tianyi Li, Yushuai Li, Peiyuan Guan, Amir Taherkordi, Tor Skeie  

**Link**: [PDF](https://arxiv.org/pdf/2603.29908)  

**Abstract**: Trajectory planning for autonomous driving increasingly leverages large language models (LLMs) for commonsense reasoning, yet LLM outputs are inherently unreliable, posing risks in safety-critical applications. We propose C-TRAIL, a framework built on a Commonsense World that couples LLM-derived commonsense with a trust mechanism to guide trajectory planning. C-TRAIL operates through a closed-loop Recall, Plan, and Update cycle: the Recall module queries an LLM for semantic relations and quantifies their reliability via a dual-trust mechanism; the Plan module injects trust-weighted commonsense into Monte Carlo Tree Search (MCTS) through a Dirichlet trust policy; and the Update module adaptively refines trust scores and policy parameters from environmental feedback. Experiments on four simulated scenarios in Highway-env and two real-world levelXData datasets (highD, rounD) show that C-TRAIL consistently outperforms state-of-the-art baselines, reducing ADE by 40.2%, FDE by 51.7%, and improving SR by 16.9 percentage points on average. The source code is available at this https URL. 

---
# ATP-Bench: Towards Agentic Tool Planning for MLLM Interleaved Generation 

**Authors**: Yinuo Liu, Zi Qian, Heng Zhou, Jiahao Zhang, Yajie Zhang, Zhihang Li, Mengyu Zhou, Erchao Zhao, Xiaoxi Jiang, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29902)  

**Abstract**: Interleaved text-and-image generation represents a significant frontier for Multimodal Large Language Models (MLLMs), offering a more intuitive way to convey complex information. Current paradigms rely on either image generation or retrieval augmentation, yet they typically treat the two as mutually exclusive paths, failing to unify factuality with creativity. We argue that the next milestone in this field is Agentic Tool Planning, where the model serves as a central controller that autonomously determines when, where, and which tools to invoke to produce interleaved responses for visual-critical queries. To systematically evaluate this paradigm, we introduce ATP-Bench, a novel benchmark comprising 7,702 QA pairs (including 1,592 VQA pairs) across eight categories and 25 visual-critical intents, featuring human-verified queries and ground truths. Furthermore, to evaluate agentic planning independent of end-to-end execution and changing tool backends, we propose a Multi-Agent MLLM-as-a-Judge (MAM) system. MAM evaluates tool-call precision, identifies missed opportunities for tool use, and assesses overall response quality without requiring ground-truth references. Our extensive experiments on 10 state-of-the-art MLLMs reveal that models struggle with coherent interleaved planning and exhibit significant variations in tool-use behavior, highlighting substantial room for improvement and providing actionable guidance for advancing interleaved generation. Dataset and code are available at this https URL. 

---
# A Rational Account of Categorization Based on Information Theory 

**Authors**: Christophe J. MacLellan, Karthik Singaravadivelan, Xin Lian, Zekun Wang, Pat Langley  

**Link**: [PDF](https://arxiv.org/pdf/2603.29895)  

**Abstract**: We present a new theory of categorization based on an information-theoretic rational analysis. To evaluate this theory, we investigate how well it can account for key findings from classic categorization experiments conducted by Hayes-Roth and Hayes-Roth (1977), Medin and Schaffer (1978), and Smith and Minda (1998). We find that it explains the human categorization behavior at least as well (or better) than the independent cue and context models (Medin & Schaffer, 1978), the rational model of categorization (Anderson, 1991), and a hierarchical Dirichlet process model (Griffiths et al., 2007). 

---
# ShapE-GRPO: Shapley-Enhanced Reward Allocation for Multi-Candidate LLM Training 

**Authors**: Rui Ai, Yu Pan, David Simchi-Levi, Chonghuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29871)  

**Abstract**: In user-agent interaction scenarios such as recommendation, brainstorming, and code suggestion, Large Language Models (LLMs) often generate sets of candidate recommendations where the objective is to maximize the collective utility of the entire set rather than individual candidates independently. However, existing reinforcement learning post-training paradigms, such as Group Relative Policy Optimization (GRPO), typically assign the same set-level scalar reward to every candidate in the set. This leads to noisy training signals where poor candidates free-ride on the high reward produced by a single strong peer, resulting in suboptimal exploration. To address this, we propose Shapley-Enhanced GRPO (ShapE-GRPO). By leveraging the permutation-invariant nature of set-level utility, we derive a Shapley-enhanced formulation from cooperative game theory to decompose set-level rewards into granular, candidate-specific signals. We show that our formulation preserves the fundamental axioms of the Shapley value while remaining computationally efficient with polynomial-time complexity. Empirically, ShapE-GRPO consistently outperforms standard GRPO across diverse datasets with accelerated convergence during training. 

---
# Spatiotemporal Robustness of Temporal Logic Tasks using Multi-Objective Reasoning 

**Authors**: Oliver Schön, Lars Lindemann  

**Link**: [PDF](https://arxiv.org/pdf/2603.29868)  

**Abstract**: The reliability of autonomous systems depends on their robustness, i.e., their ability to meet their objectives under uncertainty. In this paper, we study spatiotemporal robustness of temporal logic specifications evaluated over discrete-time signals. Existing work has proposed robust semantics that capture not only Boolean satisfiability, but also the geometric distance from unsatisfiability, corresponding to admissible spatial perturbations of a given signal. In contrast, we propose spatiotemporal robustness (STR), which captures admissible spatial and temporal perturbations jointly. This notion is particularly informative for interacting systems, such as multi-agent robotics, smart cities, and air traffic control. We define STR as a multi-objective reasoning problem, formalized via a partial order over spatial and temporal perturbations. This perspective has two key advantages: (1) STR can be interpreted as a Pareto-optimal set that characterizes all admissible spatiotemporal perturbations, and (2) STR can be computed using tools from multi-objective optimization. To navigate computational challenges, we propose robust semantics for STR that are sound in the sense of suitably under-approximating STR while being computationally tractable. Finally, we present monitoring algorithms for STR using these robust semantics. To the best of our knowledge, this is the first work to deal with robustness across multiple dimensions via multi-objective reasoning. 

---
# AgentFixer: From Failure Detection to Fix Recommendations in LLM Agentic Systems 

**Authors**: Hadar Mulian, Sergey Zeltyn, Ido Levy, Liane Galanti, Avi Yaeli, Segev Shlomov  

**Link**: [PDF](https://arxiv.org/pdf/2603.29848)  

**Abstract**: We introduce a comprehensive validation framework for LLM-based agentic systems that provides systematic diagnosis and improvement of reliability failures. The framework includes fifteen failure-detection tools and two root-cause analysis modules that jointly uncover weaknesses across input handling, prompt design, and output generation. It integrates lightweight rule-based checks with LLM-as-a-judge assessments to support structured incident detection, classification, and repair. We applied the framework to IBM CUGA, evaluating its performance on the AppWorld and WebArena benchmarks. The analysis revealed recurrent planner misalignments, schema violations, brittle prompt dependencies, and more. Based on these insights, we refined both prompting and coding strategies, maintaining CUGA's benchmark results while enabling mid-sized models such as Llama 4 and Mistral Medium to achieve notable accuracy gains, substantially narrowing the gap with frontier models. Beyond quantitative validation, we conducted an exploratory study that fed the framework's diagnostic outputs and agent description into an LLM for self-reflection and prioritization. This interactive analysis produced actionable insights on recurring failure patterns and focus areas for improvement, demonstrating how validation itself can evolve into an agentic, dialogue-driven process. These results show a path toward scalable, quality assurance, and adaptive validation in production agentic systems, offering a foundation for more robust, interpretable, and self-improving agentic architectures. 

---
# Owl-AuraID 1.0: An Intelligent System for Autonomous Scientific Instrumentation and Scientific Data Analysis 

**Authors**: Han Deng, Anqi Zou, Hanling Zhang, Ben Fei, Chengyu Zhang, Haobo Wang, Xinru Guo, Zhenyu Li, Xuzhu Wang, Peng Yang, Fujian Zhang, Weiyu Guo, Xiaohong Shao, Zhaoyang Liu, Shixiang Tang, Zhihui Wang, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29828)  

**Abstract**: Scientific discovery increasingly depends on high-throughput characterization, yet automation is hindered by proprietary GUIs and the limited generalizability of existing API-based systems. We present Owl-AuraID, a software-hardware collaborative embodied agent system that adopts a GUI-native paradigm to operate instruments through the same interfaces as human experts. Its skill-centric framework integrates Type-1 (GUI operation) and Type-2 (data analysis) skills into end-to-end workflows, connecting physical sample handling with scientific interpretation. Owl-AuraID demonstrates broad coverage across ten categories of precision instruments and diverse workflows, including multimodal spectral analysis, microscopic imaging, and crystallographic analysis, supporting modalities such as FTIR, NMR, AFM, and TGA. Overall, Owl-AuraID provides a practical, extensible foundation for autonomous laboratories and illustrates a path toward evolving laboratory intelligence through reusable operational and analytical skills. The code are available at this https URL. 

---
# Reasoning-Driven Synthetic Data Generation and Evaluation 

**Authors**: Tim R. Davidson, Benoit Seguin, Enrico Bacis, Cesar Ilharco, Hamza Harkous  

**Link**: [PDF](https://arxiv.org/pdf/2603.29791)  

**Abstract**: Although many AI applications of interest require specialized multi-modal models, relevant data to train such models is inherently scarce or inaccessible. Filling these gaps with human annotators is prohibitively expensive, error-prone, and time-consuming, leading model builders to increasingly consider synthetic data as a scalable alternative. However, existing synthetic data generation methods often rely on manual prompts, evolutionary algorithms, or extensive seed data from the target distribution - limiting their scalability, explainability, and control. In this paper, we introduce Simula: a novel reasoning-driven framework for data generation and evaluation. It employs a seedless, agentic approach to generate synthetic datasets at scale, allowing users to define desired dataset characteristics through an explainable and controllable process that enables fine-grained resource allocation. We show the efficacy of our approach on a variety of datasets, rigorously testing both intrinsic and downstream properties. Our work (1) offers guidelines for synthetic data mechanism design, (2) provides insights into generating and evaluating synthetic data at scale, and (3) unlocks new opportunities for developing and deploying AI in domains where data scarcity or privacy concerns are paramount. 

---
# Tracking vs. Deciding: The Dual-Capability Bottleneck in Searchless Chess Transformers 

**Authors**: Quanhao Li, Wei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29761)  

**Abstract**: A human-like chess engine should mimic the style, errors, and consistency of a strong human player rather than maximize playing strength. We show that training from move sequences alone forces a model to learn two capabilities: state tracking, which reconstructs the board from move history, and decision quality, which selects good moves from that reconstructed state. These impose contradictory data requirements: low-rated games provide the diversity needed for tracking, while high-rated games provide the quality signal for decision learning. Removing low-rated data degrades performance.
We formalize this tension as a dual-capability bottleneck, P <= min(T,Q), where overall performance is limited by the weaker capability. Guided by this view, we scale the model from 28M to 120M parameters to improve tracking, then introduce Elo-weighted training to improve decisions while preserving diversity. A 2 x 2 factorial ablation shows that scaling improves tracking, weighting improves decisions, and their combination is superadditive. Linear weighting works best, while overly aggressive weighting harms tracking despite lower validation loss. We also introduce a coverage-decay formula, t* = log(N/kcrit)/log b, as a reliability horizon for intra-game degeneration risk.
Our final 120M-parameter model, without search, reached Lichess bullet 2570 over 253 rated games. On human move prediction it achieves 55.2% Top-1 accuracy, exceeding Maia-2 rapid and Maia-2 blitz. Unlike position-based methods, sequence input naturally encodes full game history, enabling history-dependent decisions that single-position models cannot exhibit. 

---
# CausalPulse: An Industrial-Grade Neurosymbolic Multi-Agent Copilot for Causal Diagnostics in Smart Manufacturing 

**Authors**: Chathurangi Shyalika, Utkarshani Jaimini, Cory Henson, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2603.29755)  

**Abstract**: Modern manufacturing environments demand real-time, trustworthy, and interpretable root-cause insights to sustain productivity and quality. Traditional analytics pipelines often treat anomaly detection, causal inference, and root-cause analysis as isolated stages, limiting scalability and explainability. In this work, we present CausalPulse, an industry-grade multi-agent copilot that automates causal diagnostics in smart manufacturing. It unifies anomaly detection, causal discovery, and reasoning through a neurosymbolic architecture built on standardized agentic protocols. CausalPulse is being deployed in a Robert Bosch manufacturing plant, integrating seamlessly with existing monitoring workflows and supporting real-time operation at production scale. Evaluations on both public (Future Factories) and proprietary (Planar Sensor Element) datasets show high reliability, achieving overall success rates of 98.0% and 98.73%. Per-criterion success rates reached 98.75% for planning and tool use, 97.3% for self-reflection, and 99.2% for collaboration. Runtime experiments report end-to-end latency of 50-60s per diagnostic workflow with near-linear scalability (R^2=0.97), confirming real-time readiness. Comparison with existing industrial copilots highlights distinct advantages in modularity, extensibility, and deployment maturity. These results demonstrate how CausalPulse's modular, human-in-the-loop design enables reliable, interpretable, and production-ready automation for next-generation manufacturing. 

---
# Spontaneous Functional Differentiation in Large Language Models: A Brain-Like Intelligence Economy 

**Authors**: Junjie Zhang, Zhen Shen, Gang Xiong, Xisong Dong  

**Link**: [PDF](https://arxiv.org/pdf/2603.29735)  

**Abstract**: The evolution of intelligence in artificial systems provides a unique opportunity to identify universal computational principles. Here we show that large language models spontaneously develop synergistic cores where information integration exceeds individual parts remarkably similar to the human brain. Using Integrated Information Decomposition across multiple architectures we find that middle layers exhibit synergistic processing while early and late layers rely on redundancy. This organization is dynamic and emerges as a physical phase transition as task difficulty increases. Crucially ablating synergistic components causes catastrophic performance loss confirming their role as the physical entity of abstract reasoning and bridging artificial and biological intelligence. 

---
# Reinforced Reasoning for End-to-End Retrosynthetic Planning 

**Authors**: Chenyang Zuo, Siqi Fan, Yizhen Luo, Zaiqing Nie  

**Link**: [PDF](https://arxiv.org/pdf/2603.29723)  

**Abstract**: Retrosynthetic planning is a fundamental task in organic chemistry, yet remains challenging due to its combinatorial complexity. To address this, conventional approaches typically rely on hybrid frameworks that combine single-step predictions with external search heuristics, inevitably fracturing the logical coherence between local molecular transformations and global planning objectives. To bridge this gap and embed sophisticated strategic foresight directly into the model's chemical reasoning, we introduce ReTriP, an end-to-end generative framework that reformulates retrosynthesis as a direct Chain-of-Thought reasoning task. We establish a path-coherent molecular representation and employ a progressive training curriculum that transitions from reasoning distillation to reinforcement learning with verifiable rewards, effectively aligning stepwise generation with practical route utility. Empirical evaluation on RetroBench demonstrates that ReTriP achieves state-of-the-art performance, exhibiting superior robustness in long-horizon planning compared to hybrid baselines. 

---
# Symphony for Medical Coding: A Next-Generation Agentic System for Scalable and Explainable Medical Coding 

**Authors**: Joakim Edin, Andreas Motzfeldt, Simon Flachs, Lars Maaløe  

**Link**: [PDF](https://arxiv.org/pdf/2603.29709)  

**Abstract**: Medical coding translates free-text clinical documentation into standardized codes drawn from classification systems that contain tens of thousands of entries and are updated annually. It is central to billing, clinical research, and quality reporting, yet remains largely manual, slow, and error-prone. Existing automated approaches learn to predict a fixed set of codes from labeled data, thereby preventing adaptation to new codes or different coding systems without retraining on different data. They also provide no explanation for their predictions, limiting trust in safety-critical settings. We introduce Symphony for Medical Coding, a system that approaches the task the way expert human coders do: by reasoning over the clinical narrative with direct access to the coding guidelines. This design allows Symphony to operate across any coding system and to provide span-level evidence linking each predicted code to the text that supports it. We evaluate on two public benchmarks and three real-world datasets spanning inpatient, outpatient, emergency, and subspecialty settings across the United States and the United Kingdom. Symphony achieves state-of-the-art results across all settings, establishing itself as a flexible, deployment-ready foundation for automated clinical coding. 

---
# Measuring the metacognition of AI 

**Authors**: Richard Servajean, Philippe Servajean  

**Link**: [PDF](https://arxiv.org/pdf/2603.29693)  

**Abstract**: A robust decision-making process must take into account uncertainty, especially when the choice involves inherent risks. Because artificial Intelligence (AI) systems are increasingly integrated into decision-making workflows, managing uncertainty relies more and more on the metacognitive capabilities of these systems; i.e, their ability to assess the reliability of and regulate their own decisions. Hence, it is crucial to employ robust methods to measure the metacognitive abilities of AI. This paper is primarily a methodological contribution arguing for the adoption of the meta-d' framework, or its model-free alternatives, as the gold standard for assessing the metacognitive sensitivity of AIs--the ability to generate confidence ratings that distinguish correct from incorrect responses. Moreover, we propose to leverage signal detection theory (SDT) to measure the ability of AIs to spontaneously regulate their decisions based on uncertainty and risk. To demonstrate the practical utility of these psychophysical frameworks, we conduct two series of experiments on three large language models (LLMs)--GPT-5, DeepSeek-V3.2-Exp, and Mistral-Medium-2508. In the first experiments, LLMs performed a primary judgment followed by a confidence rating. In the second, LLMs only performed the primary judgment, while we manipulated the risk associated with either response. On the one hand, applying the meta-d' framework allows us to conduct comparisons along three axes: comparing an LLM to optimality, comparing different LLMs on a given task, and comparing the same LLM across different tasks. On the other hand, SDT allows us to assess whether LLMs become more conservative when risks are high. 

---
# A First Step Towards Even More Sparse Encodings of Probability Distributions 

**Authors**: Florian Andreas Marwitz, Tanya Braun, Ralf Möller  

**Link**: [PDF](https://arxiv.org/pdf/2603.29691)  

**Abstract**: Real world scenarios can be captured with lifted probability distributions. However, distributions are usually encoded in a table or list, requiring an exponential number of values. Hence, we propose a method for extracting first-order formulas from probability distributions that require significantly less values by reducing the number of values in a distribution and then extracting, for each value, a logical formula to be further minimized. This reduction and minimization allows for increasing the sparsity in the encoding while also generalizing a given distribution. Our evaluation shows that sparsity can increase immensely by extracting a small set of short formulas while preserving core information. 

---
# Beyond the Steeper Curve: AI-Mediated Metacognitive Decoupling and the Limits of the Dunning-Kruger Metaphor 

**Authors**: Christopher Koch  

**Link**: [PDF](https://arxiv.org/pdf/2603.29681)  

**Abstract**: The common claim that generative AI simply amplifies the Dunning-Kruger effect is too coarse to capture the available evidence. The clearest findings instead suggest that large language model (LLM) use can improve observable output and short-term task performance while degrading metacognitive accuracy and flattening the classic competence-confidence gradient across skill groups. This paper synthesizes evidence from human-AI interaction, learning research, and model evaluation, and proposes the working model of AI-mediated metacognitive decoupling: a widening gap among produced output, underlying understanding, calibration accuracy, and self-assessed ability. This four-variable account better explains overconfidence, over- and under-reliance, crutch effects, and weak transfer than the simpler metaphor of a uniformly steeper Dunning-Kruger curve. The paper concludes with implications for tool design, assessment, and knowledge work. 

---
# View-oriented Conversation Compiler for Agent Trace Analysis 

**Authors**: Lvmin Zhang, Maneesh Agrawala  

**Link**: [PDF](https://arxiv.org/pdf/2603.29678)  

**Abstract**: Agent traces carry increasing analytical value in the era of context learning and harness-driven agentic cognition, yet most prior work treats conversation format as a trivial engineering detail. Modern agent conversations contain deeply structured content, including nested tool calls and results, chain-of-thought reasoning blocks, sub-agent invocations, context-window compaction boundaries, and harness-injected system directives, whose complexity far exceeds that of simple user-assistant exchanges. Feeding such traces to a reflector or other analytical mechanism in plain text, JSON, YAML, or via grep can materially degrade analysis quality. This paper presents VCC (View-oriented Conversation Compiler), a compiler (lex, parse, IR, lower, emit) that transforms raw agent JSONL logs into a family of structured views: a full view (lossless transcript serving as the canonical line-number coordinate system), a user-interface view (reconstructing the interaction as the user actually perceived it), and an adaptive view (a structure-preserving projection governed by a relevance predicate). In a context-learning experiment on AppWorld, replacing only the reflector's input format, from raw JSONL to VCC-compiled views, leads to higher pass rates across all three model configurations tested, while cutting reflector token consumption by half to two-thirds and producing more concise learned memory. These results suggest that message format functions as infrastructure for context learning, not as an incidental implementation choice. 

---
# Optimizing Donor Outreach for Blood Collection Sessions: A Scalable Decision Support Framework 

**Authors**: André Carneiro, Pedro T. Monteiro, Rui Henriques  

**Link**: [PDF](https://arxiv.org/pdf/2603.29643)  

**Abstract**: Blood donation centers face challenges in matching supply with demand while managing donor availability. Although targeted outreach is important, it can cause donor fatigue via over-solicitation. Effective recruitment requires targeting the right donors at the right time, balancing constraints with donor convenience and eligibility. Despite extensive work on blood supply chain optimization and growing interest in algorithmic donor recruitment, the operational problem of assigning donors to sessions across a multi-site network, taking into account eligibility, capacity, blood-type demand targets, geographic convenience, and donor safety, remains unaddressed.
We address this gap with an optimization framework for donor invitation scheduling incorporating donor eligibility, travel convenience, blood-type demand targets, and penalties. We evaluate two strategies: (i) a binary integer linear programming (BILP) formulation and (ii) an efficient greedy heuristic. Evaluation uses the registry from Instituto Português do Sangue e da Transplantação (IPST) for invite planning in the Lisbon operational region using 4-month windows. A prospective pipeline integrates organic attendance forecasting, quantile-based demand targets, and residual capacity estimation for forward-looking invitation plans. Results reveal its key role in closing the supply-demand gap in the Lisbon operational region. A controlled comparison shows that the greedy heuristic achieves results comparable to the BILP, with 188x less peak memory and 115x faster runtime; trade-offs include 3.9 pp lower demand fulfillment (86.1% vs. 90.0%), larger donor-session distance, higher adverse-reaction donor exposure, and greater invitation burden per non-high-frequency donor, reflecting local versus global optimization. Experiments assess how constraint-aware scheduling can close gaps by mobilizing eligible inactive/lapsing donors. 

---
# ASI-Evolve: AI Accelerates AI 

**Authors**: Weixian Xu, Tiantian Mi, Yixiu Liu, Yang Nan, Zhimeng Zhou, Lyumanshan Ye, Lin Zhang, Yu Qiao, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29640)  

**Abstract**: Can AI accelerate the development of AI itself? While recent agentic systems have shown strong performance on well-scoped tasks with rapid feedback, it remains unclear whether they can tackle the costly, long-horizon, and weakly supervised research loops that drive real AI progress. We present ASI-Evolve, an agentic framework for AI-for-AI research that closes this loop through a learn-design-experiment-analyze cycle. ASI-Evolve augments standard evolutionary agents with two key components: a cognition base that injects accumulated human priors into each round of exploration, and a dedicated analyzer that distills complex experimental outcomes into reusable insights for future iterations. To our knowledge, ASI-Evolve is the first unified framework to demonstrate AI-driven discovery across three central components of AI development: data, architectures, and learning algorithms. In neural architecture design, it discovered 105 SOTA linear attention architectures, with the best discovered model surpassing DeltaNet by +0.97 points, nearly 3x the gain of recent human-designed improvements. In pretraining data curation, the evolved pipeline improves average benchmark performance by +3.96 points, with gains exceeding 18 points on MMLU. In reinforcement learning algorithm design, discovered algorithms outperform GRPO by up to +12.5 points on AMC32, +11.67 points on AIME24, and +5.04 points on OlympiadBench. We further provide initial evidence that this AI-for-AI paradigm can transfer beyond the AI stack through experiments in mathematics and biomedicine. Together, these results suggest that ASI-Evolve represents a promising step toward enabling AI to accelerate AI across the foundational stages of development, offering early evidence for the feasibility of closed-loop AI research. 

---
# FlowPIE: Test-Time Scientific Idea Evolution with Flow-Guided Literature Exploration 

**Authors**: Qiyao Wang, Hongbo Wang, Longze Chen, Zhihao Yang, Guhong Chen, Hamid Alinejad-Rokny, Hui Li, Yuan Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29557)  

**Abstract**: Scientific idea generation (SIG) is critical to AI-driven autonomous research, yet existing approaches are often constrained by a static retrieval-then-generation paradigm, leading to homogeneous and insufficiently divergent ideas. In this work, we propose FlowPIE, a tightly coupled retrieval-generation framework that treats literature exploration and idea generation as a co-evolving process. FlowPIE expands literature trajectories via a flow-guided Monte Carlo Tree Search (MCTS) inspired by GFlowNets, using the quality of current ideas assessed by an LLM-based generative reward model (GRM) as a supervised signal to guide adaptive retrieval and construct a diverse, high-quality initial population. Based on this population, FlowPIE models idea generation as a test-time idea evolution process, applying selection, crossover, and mutation with the isolation island paradigm and GRM-based fitness computation to incorporate cross-domain knowledge. It effectively mitigates the information cocoons arising from over-reliance on parametric knowledge and static literature. Extensive evaluations demonstrate that FlowPIE consistently produces ideas with higher novelty, feasibility and diversity compared to strong LLM-based and agent-based frameworks, while enabling reward scaling during test time. 

---
# Learning to Generate Formally Verifiable Step-by-Step Logic Reasoning via Structured Formal Intermediaries 

**Authors**: Luoxin Chen, Yichi Zhou, Huishuai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29500)  

**Abstract**: Large language models (LLMs) have recently demonstrated impressive performance on complex, multi-step reasoning tasks, especially when post-trained with outcome-rewarded reinforcement learning Guo et al. 2025. However, it has been observed that outcome rewards often overlook flawed intermediate steps, leading to unreliable reasoning steps even when final answers are correct. To address this unreliable reasoning, we propose PRoSFI (Process Reward over Structured Formal Intermediates), a novel reward method that enhances reasoning reliability without compromising accuracy. Instead of generating formal proofs directly, which is rarely accomplishable for a modest-sized (7B) model, the model outputs structured intermediate steps aligned with its natural language reasoning. Each step is then verified by a formal prover. Only fully validated reasoning chains receive high rewards. The integration of formal verification guides the model towards generating step-by-step machine-checkable proofs, thereby yielding more credible final answers. PRoSFI offers a simple and effective approach to training trustworthy reasoning models. 

---
# Metriplector: From Field Theory to Neural Architecture 

**Authors**: Dan Oprisa, Peter Toth  

**Link**: [PDF](https://arxiv.org/pdf/2603.29496)  

**Abstract**: We present Metriplector, a neural architecture primitive in which the input configures an abstract physical system--fields, sources, and operators--and the dynamics of that system is the computation. Multiple fields evolve via coupled metriplectic dynamics, and the stress-energy tensor T^{{\mu}{\nu}}, derived from Noether's theorem, provides the readout. The metriplectic formulation admits a natural spectrum of instantiations: the dissipative branch alone yields a screened Poisson equation solved exactly via conjugate gradient; activating the full structure--including the antisymmetric Poisson bracket--gives field dynamics for image recognition and language modeling. We evaluate Metriplector across four domains, each using a task-specific architecture built from this shared primitive with progressively richer physics: F1=1.0 on maze pathfinding, generalizing from 15x15 training grids to unseen 39x39 grids; 97.2% exact Sudoku solve rate with zero structural injection; 81.03% on CIFAR-100 with 2.26M parameters; and 1.182 bits/byte on language modeling with 3.6x fewer training tokens than a GPT baseline. 

---
# Structural Compactness as a Complementary Criterion for Explanation Quality 

**Authors**: Mohammad Mahdi Mesgari, Jackie Ma, Wojciech Samek, Sebastian Lapuschkin, Leander Weber  

**Link**: [PDF](https://arxiv.org/pdf/2603.29491)  

**Abstract**: In the evaluation of attribution quality, the quantitative assessment of explanation legibility is particularly difficult, as it is influenced by varying shapes and internal organization of attributions not captured by simple statistics. To address this issue, we introduce Minimum Spanning Tree Compactness (MST-C), a graph-based structural metric that captures higher-order geometric properties of attributions, such as spread and cohesion. These components are combined into a single score that evaluates compactness, favoring attributions with salient points spread across a small area and spatially organized into few but cohesive clusters. We show that MST-C reliably distinguishes between explanation methods, exposes fundamental structural differences between models, and provides a robust, self-contained diagnostic for explanation compactness that complements existing notions of attribution complexity. 

---
# ELT-Bench-Verified: Benchmark Quality Issues Underestimate AI Agent Capabilities 

**Authors**: Christopher Zanoli, Andrea Giovannini, Tengjun Jin, Ana Klimovic, Yotam Perlitz  

**Link**: [PDF](https://arxiv.org/pdf/2603.29399)  

**Abstract**: Constructing Extract-Load-Transform (ELT) pipelines is a labor-intensive data engineering task and a high-impact target for AI automation. On ELT-Bench, the first benchmark for end-to-end ELT pipeline construction, AI agents initially showed low success rates, suggesting they lacked practical utility.
We revisit these results and identify two factors causing a substantial underestimation of agent capabilities. First, re-evaluating ELT-Bench with upgraded large language models reveals that the extraction and loading stage is largely solved, while transformation performance improves significantly. Second, we develop an Auditor-Corrector methodology that combines scalable LLM-driven root-cause analysis with rigorous human validation (inter-annotator agreement Fleiss' kappa = 0.85) to audit benchmark quality. Applying this to ELT-Bench uncovers that most failed transformation tasks contain benchmark-attributable errors -- including rigid evaluation scripts, ambiguous specifications, and incorrect ground truth -- that penalize correct agent outputs.
Based on these findings, we construct ELT-Bench-Verified, a revised benchmark with refined evaluation logic and corrected ground truth. Re-evaluating on this version yields significant improvement attributable entirely to benchmark correction. Our results show that both rapid model improvement and benchmark quality issues contributed to underestimating agent capabilities. More broadly, our findings echo observations of pervasive annotation errors in text-to-SQL benchmarks, suggesting quality issues are systemic in data engineering evaluation. Systematic quality auditing should be standard practice for complex agentic tasks. We release ELT-Bench-Verified to provide a more reliable foundation for progress in AI-driven data engineering automation. 

---
# AI-Generated Prior Authorization Letters: Strong Clinical Content, Weak Administrative Scaffolding 

**Authors**: Moiz Sadiq Awan, Maryam Raza  

**Link**: [PDF](https://arxiv.org/pdf/2603.29366)  

**Abstract**: Prior authorization remains one of the most burdensome administrative processes in U.S. healthcare, consuming billions of dollars and thousands of physician hours each year. While large language models have shown promise across clinical text tasks, their ability to produce submission-ready prior authorization letters has received only limited attention, with existing work confined to single-case demonstrations rather than structured multi-scenario evaluation. We assessed three commercially available LLMs (GPT-4o, Claude Sonnet 4.5, and Gemini 2.5 Pro) across 45 physician-validated synthetic scenarios spanning rheumatology, psychiatry, oncology, cardiology, and orthopedics. All three models generated letters with strong clinical content: accurate diagnoses, well-structured medical necessity arguments, and thorough step therapy documentation. However, a secondary analysis of real-world administrative requirements revealed consistent gaps that clinical scoring alone did not capture, including absent billing codes, missing authorization duration requests, and inadequate follow-up plans. These findings reframe the question: the challenge for clinical deployment is not whether LLMs can write clinically adequate letters, but whether the systems built around them can supply the administrative precision that payer workflows require. 

---
# Rigorous Explanations for Tree Ensembles 

**Authors**: Yacine Izza, Alexey Ignatiev, Xuanxiang Huang, Peter J. Stuckey, Joao Marques-Silva  

**Link**: [PDF](https://arxiv.org/pdf/2603.29361)  

**Abstract**: Tree ensembles (TEs) find a multitude of practical applications. They represent one of the most general and accurate classes of machine learning methods. While they are typically quite concise in representation, their operation remains inscrutable to human decision makers. One solution to build trust in the operation of TEs is to automatically identify explanations for the predictions made. Evidently, we can only achieve trust using explanations, if those explanations are rigorous, that is truly reflect properties of the underlying predictor they explain This paper investigates the computation of rigorously-defined, logically-sound explanations for the concrete case of two well-known examples of tree ensembles, namely random forests and boosted trees. 

---
# BenchScope: How Many Independent Signals Does Your Benchmark Provide? 

**Authors**: Tommy Sha, Stella Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.29357)  

**Abstract**: AI evaluation suites often report many scores without checking whether those scores carry independent information. We introduce Effective Dimensionality (ED), the participation ratio of a centered benchmark-score spectrum, as a fast, population-conditional upper-bound diagnostic of measurement breadth. Applied at per-instance granularity to 22 benchmarks across 8 domains and more than 8,400 model evaluations, ED reveals substantial redundancy: the six-score Open LLM Leaderboard behaves like roughly two effective measurement axes (ED = 1.7), BBH and MMLU-Pro are near-interchangeable (rho = 0.96, stable across seven subpopulations), and measurement breadth varies more than 20x across current benchmarks. We show that relative ED rankings are stable under matched-dimension controls and that ED can flag redundant suite components, monitor performance-conditional compression, and guide benchmark maintenance. Because binary spectra overestimate absolute latent dimensionality, we interpret ED as a screening statistic rather than a literal factor count and complement it with null, reliability, and saturation analyses. We provide a 22-benchmark reference atlas and a four-step diagnostic workflow that benchmark maintainers can run with a score matrix and a few lines of code. 

---
# Nomad: Autonomous Exploration and Discovery 

**Authors**: Bokang Jia, Samta Kamboj, Satheesh Katipomu, Seung Hun Han, Neha Sengupta, Andrew Jackson  

**Link**: [PDF](https://arxiv.org/pdf/2603.29353)  

**Abstract**: We introduce Nomad, a system for autonomous data exploration and insight discovery. Given a corpus of documents, databases, or other data sources, users rarely know the full set of questions, hypotheses, or connections that could be explored. As a result, query-driven question answering and prompt-driven deep-research systems remain limited by human framing and often fail to cover the broader insight space.
Nomad addresses this problem with an exploration-first architecture. It constructs an explicit Exploration Map over the domain and systematically traverses it to balance breadth and depth. It generates and selects hypotheses and investigates them with an explorer agent that can use document search, web search, and database tools. Candidate insights are then checked by an independent verifier before entering a reporting pipeline that produces cited reports and higher-level meta-reports.
We also present a comprehensive evaluation framework for autonomous discovery systems that measures trustworthiness, report quality, and diversity. Using a corpus of selected UN and WHO reports, we show that \nomad{} produces more trustworthy and higher-quality reports than baselines, while also producing more diverse insights over several runs.
Nomad is a step toward autonomous systems that not only answer user questions or conduct directed research, but also discover which questions, research directions, and insights are worth surfacing in the first place. 

---
# PSPA-Bench: A Personalized Benchmark for Smartphone GUI Agent 

**Authors**: Hongyi Nie, Xunyuan Liu, Yudong Bai, Yaqing Wang, Yang Liu, Quanming Yao, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29318)  

**Abstract**: Smartphone GUI agents execute tasks by operating directly on app interfaces, offering a path to broad capability without deep system integration. However, real-world smartphone use is highly personalized: users adopt diverse workflows and preferences, challenging agents to deliver customized assistance rather than generic solutions. Existing GUI agent benchmarks cannot adequately capture this personalization dimension due to sparse user-specific data and the lack of fine-grained evaluation metrics. To address this gap, we present PSPA-Bench, the benchmark dedicated to evaluating personalization in smartphone GUI agents. PSPA-Bench comprises over 12,855 personalized instructions aligned with real-world user behaviors across 10 representative daily-use scenarios and 22 mobile apps, and introduces a structure-aware process evaluation method that measures agents' personalized capabilities at a fine-grained level. Through PSPA-Bench, we benchmark 11 state-of-the-art GUI agents. Results reveal that current methods perform poorly under personalized settings, with even the strongest agent achieving limited success. Our analysis further highlights three directions for advancing personalized GUI agents: (1) reasoning-oriented models consistently outperform general LLMs, (2) perception remains a simple yet critical capability, and (3) reflection and long-term memory mechanisms are key to improving adaptation. Together, these findings establish PSPA-Bench as a foundation for systematic study and future progress in personalized GUI agents. 

---
# Grokking From Abstraction to Intelligence 

**Authors**: Junjie Zhang, Zhen Shen, Gang Xiong, Xisong Dong  

**Link**: [PDF](https://arxiv.org/pdf/2603.29262)  

**Abstract**: Grokking in modular arithmetic has established itself as the quintessential fruit fly experiment, serving as a critical domain for investigating the mechanistic origins of model generalization. Despite its significance, existing research remains narrowly focused on specific local circuits or optimization tuning, largely overlooking the global structural evolution that fundamentally drives this phenomenon. We propose that grokking originates from a spontaneous simplification of internal model structures governed by the principle of parsimony. We integrate causal, spectral, and algorithmic complexity measures alongside Singular Learning Theory to reveal that the transition from memorization to generalization corresponds to the physical collapse of redundant manifolds and deep information compression, offering a novel perspective for understanding the mechanisms of model overfitting and generalization. 

---
# Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents 

**Authors**: Aaditya Khanal, Yangyang Tao, Junxiu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.29231)  

**Abstract**: Existing benchmarks measure capability -- whether a model succeeds on a single attempt -- but production deployments
require reliability -- consistent success across repeated attempts on tasks of varying duration. We show these
properties diverge systematically as task duration grows, and that pass@1 on short tasks is structurally blind to
this divergence.
We introduce a reliability science framework for long-horizon LLM agents with four metrics: Reliability Decay Curve
(RDC), Variance Amplification Factor (VAF), Graceful Degradation Score (GDS), and Meltdown Onset Point (MOP). We
evaluate 10 models across 23,392 episodes on a 396-task benchmark spanning four duration buckets and three domains.
Key findings: (1) reliability decay is domain-stratified -- SE GDS drops from 0.90 to 0.44 while document processing
is nearly flat (0.74 to 0.71); (2) VAF bifurcates by capability tier -- high VAF is a capability signature, not an
instability signal; (3) capability and reliability rankings diverge substantially, with multi-rank inversions at long
horizons; (4) frontier models have the highest meltdown rates (up to 19%) because they attempt ambitious multi-step
strategies that sometimes spiral; and (5) memory scaffolds universally hurt long-horizon performance across all 10
models. These results motivate reliability as a first-class evaluation dimension alongside capability. 

---
# Xuanwu: Evolving General Multimodal Models into an Industrial-Grade Foundation for Content Ecosystems 

**Authors**: Zhiqian Zhang, Xu Zhao, Xiaoqing Xu, Guangdong Liang, Weijia Wang, Xiaolei Lv, Bo Li, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.29211)  

**Abstract**: In recent years, multimodal large models have continued to improve on general benchmarks. However, in real-world content moderation and adversarial settings, mainstream models still suffer from degraded generalization and catastrophic forgetting because of limited fine-grained visual perception and insufficient modeling of long-tail noise. In this paper, we present Xuanwu VL-2B as a case study of how general multimodal models can be developed into an industrial-grade foundation model for content ecosystems. The model adopts a compact InternViT-300M + MLP + Qwen3 1.7B architecture, balancing fine-grained visual perception, language-semantic alignment, and deployment cost within an approximately 2B-parameter budget. To balance business specialization with the retention of general capabilities, we developed a data iteration and curation mechanism and trained the model through a progressive three-stage pipeline: pre-training, mid-training, and post-training. Ablation studies and offline business evaluations show that Xuanwu VL-2B achieves an average score of 67.90 across seven OpenCompass multimodal metrics (vs. 64.27 for InternVL 3.5 2B), an average recall of 94.38% over seven independent business moderation tasks, and a weighted overall recall of 82.82% on policy-violating text in challenging adversarial OCR scenarios, outperforming Gemini-2.5-Pro (76.72%). These results show that, under a limited parameter budget, Xuanwu VL-2B achieves a practical balance among business alignment, visual perception, general capability retention, and deployment cost. 

---
# Route-Induced Density and Stability (RIDE): Controlled Intervention and Mechanism Analysis of Routing-Style Meta Prompts on LLM Internal States 

**Authors**: Dianxing Zhang, Gang Li, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.29206)  

**Abstract**: Routing is widely used to scale large language models, from Mixture-of-Experts gating to multi-model/tool selection. A common belief is that routing to a task ``expert'' activates sparser internal computation and thus yields more certain and stable outputs (the Sparsity--Certainty Hypothesis). We test this belief by injecting routing-style meta prompts as a textual proxy for routing signals in front of frozen instruction-tuned LLMs. We quantify (C1) internal density via activation sparsity, (C2) domain-keyword attention, and (C3) output stability via predictive entropy and semantic variation. On a RouterEval subset with three instruction-tuned models (Qwen3-8B, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.2), meta prompts consistently densify early/middle-layer representations rather than increasing sparsity; natural-language expert instructions are often stronger than structured tags. Attention responses are heterogeneous: Qwen/Llama reduce keyword attention, while Mistral reinforces it. Finally, the densification--stability link is weak and appears only in Qwen, with near-zero correlations in Llama and Mistral. We present RIDE as a diagnostic probe for calibrating routing design and uncertainty estimation. 

---
# AEC-Bench: A Multimodal Benchmark for Agentic Systems in Architecture, Engineering, and Construction 

**Authors**: Harsh Mankodiya, Chase Gallik, Theodoros Galanos, Andriy Mulyar  

**Link**: [PDF](https://arxiv.org/pdf/2603.29199)  

**Abstract**: The AEC-Bench is a multimodal benchmark for evaluating agentic systems on real-world tasks in the Architecture, Engineering, and Construction (AEC) domain. The benchmark covers tasks requiring drawing understanding, cross-sheet reasoning, and construction project-level coordination. This report describes the benchmark motivation, dataset taxonomy, evaluation protocol, and baseline results across several domain-specific foundation model harnesses. We use AEC-Bench to identify consistent tools and harness design techniques that uniformly improve performance across foundation models in their own base harnesses, such as Claude Code and Codex. We openly release our benchmark dataset, agent harness, and evaluation code for full replicability at this https URL under an Apache 2 license. 

---
# Webscraper: Leverage Multimodal Large Language Models for Index-Content Web Scraping 

**Authors**: Guan-Lun Huang, Yuh-Jzer Joung  

**Link**: [PDF](https://arxiv.org/pdf/2603.29161)  

**Abstract**: Modern web scraping struggles with dynamic, interactive websites that require more than static HTML parsing. Current methods are often brittle and require manual customization for each site. To address this, we introduce Webscraper, a framework designed to handle the challenges of modern, dynamic web applications. It leverages a Multimodal Large Language Model (MLLM) to autonomously navigate interactive interfaces, invoke specialized tools, and perform structured data extraction in environments where traditional scrapers are ineffective. Webscraper utilizes a structured five-stage prompting procedure and a set of custom-built tools to navigate and extract data from websites following the common ``index-and-content'' architecture. Our experiments, conducted on six news websites, demonstrate that the full Webscraper framework, equipped with both our guiding prompt and specialized tools, achieves a significant improvement in extraction accuracy over the baseline agent Anthropic's Computer Use. We also applied the framework to e-commerce platforms to validate its generalizability. 

---
# SimMOF: AI agent for Automated MOF Simulations 

**Authors**: Jaewoong Lee, Taeun Bae, Jihan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.29152)  

**Abstract**: Metal-organic frameworks (MOFs) offer a vast design space, and as such, computational simulations play a critical role in predicting their structural and physicochemical properties. However, MOF simulations remain difficult to access because reliable analysis require expert decisions for workflow construction, parameter selection, tool interoperability, and the preparation of computational ready structures. Here, we introduce SimMOF, a large language model based multi agent framework that automates end-to-end MOF simulation workflows from natural language queries. SimMOF translates user requests into dependency aware plans, generates runnable inputs, orchestrates multiple agents to execute simulations, and summarizes results with analysis aligned to the user query. Through representative case studies, we show that SimMOF enables adaptive and cognitively autonomous workflows that reflect the iterative and decision driven behavior of human researchers and as such provides a scalable foundation for data driven MOF research. 

---
# Knowledge database development by large language models for countermeasures against viruses and marine toxins 

**Authors**: Hung N. Do, Jessica Z. Kubicek-Sutherland, S. Gnanakaran  

**Link**: [PDF](https://arxiv.org/pdf/2603.29149)  

**Abstract**: Access to the most up-to-date information on medical countermeasures is important for the research and development of effective treatments for viruses and marine toxins. However, there is a lack of comprehensive databases that curate data on viruses and marine toxins, making decisions on medical countermeasures slow and difficult. In this work, we employ two large language models (LLMs) of ChatGPT and Grok to design two comprehensive databases of therapeutic countermeasures for five viruses of Lassa, Marburg, Ebola, Nipah, and Venezuelan equine encephalitis, as well as marine toxins. With high-level human-provided inputs, the two LLMs identify public databases containing data on the five viruses and marine toxins, collect relevant information from these databases and the literature, iteratively cross-validate the collected information, and design interactive webpages for easy access to the curated, comprehensive databases. Notably, the ChatGPT LLM is employed to design agentic AI workflows (consisting of two AI agents for research and decision-making) to rank countermeasures for viruses and marine toxins in the databases. Together, our work explores the potential of LLMs as a scalable, updatable approach for building comprehensive knowledge databases and supporting evidence-based decision-making. 

---
# REFINE: Real-world Exploration of Interactive Feedback and Student Behaviour 

**Authors**: Fares Fawzi, Seyed Parsa Neshaei, Marta Knezevic, Tanya Nazaretsky, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2603.29142)  

**Abstract**: Formative feedback is central to effective learning, yet providing timely, individualised feedback at scale remains a persistent challenge. While recent work has explored the use of large language models (LLMs) to automate feedback, most existing systems still conceptualise feedback as a static, one-way artifact, offering limited support for interpretation, clarification, or follow-up. In this work, we introduce REFINE, a locally deployable, multi-agent feedback system built on small, open-source LLMs that treats feedback as an interactive process. REFINE combines a pedagogically-grounded feedback generation agent with an LLM-as-a-judge-guided regeneration loop using a human-aligned judge, and a self-reflective tool-calling interactive agent that supports student follow-up questions with context-aware, actionable responses. We evaluate REFINE through controlled experiments and an authentic classroom deployment in an undergraduate computer science course. Automatic evaluations show that judge-guided regeneration significantly improves feedback quality, and that the interactive agent produces efficient, high-quality responses comparable to a state-of-the-art closed-source model. Analysis of real student interactions further reveals distinct engagement patterns and indicates that system-generated feedback systematically steers subsequent student inquiry. Our findings demonstrate the feasibility and effectiveness of multi-agent, tool-augmented feedback systems for scalable, interactive feedback. 

---
# SciVisAgentBench: A Benchmark for Evaluating Scientific Data Analysis and Visualization Agents 

**Authors**: Kuangshi Ai, Haichao Miao, Kaiyuan Tang, Nathaniel Gorski, Jianxin Sun, Guoxi Liu, Helgi I. Ingolfsson, David Lenz, Hanqi Guo, Hongfeng Yu, Teja Leburu, Michael Molash, Bei Wang, Tom Peterka, Chaoli Wang, Shusen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29139)  

**Abstract**: Recent advances in large language models (LLMs) have enabled agentic systems that translate natural language intent into executable scientific visualization (SciVis) tasks. Despite rapid progress, the community lacks a principled and reproducible benchmark for evaluating these emerging SciVis agents in realistic, multi-step analysis settings. We present SciVisAgentBench, a comprehensive and extensible benchmark for evaluating scientific data analysis and visualization agents. Our benchmark is grounded in a structured taxonomy spanning four dimensions: application domain, data type, complexity level, and visualization operation. It currently comprises 108 expert-crafted cases covering diverse SciVis scenarios. To enable reliable assessment, we introduce a multimodal outcome-centric evaluation pipeline that combines LLM-based judging with deterministic evaluators, including image-based metrics, code checkers, rule-based verifiers, and case-specific evaluators. We also conduct a validity study with 12 SciVis experts to examine the agreement between human and LLM judges. Using this framework, we evaluate representative SciVis agents and general-purpose coding agents to establish initial baselines and reveal capability gaps. SciVisAgentBench is designed as a living benchmark to support systematic comparison, diagnose failure modes, and drive progress in agentic SciVis. The benchmark is available at this https URL. 

---
# GISTBench: Evaluating LLM User Understanding via Evidence-Based Interest Verification 

**Authors**: Iordanis Fostiropoulos, Muhammad Rafay Azhar, Abdalaziz Sawwan, Boyu Fang, Yuchen Liu, Jiayi Liu, Hanchao Yu, Qi Guo, Jianyu Wang, Fei Liu, Xiangjun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.29112)  

**Abstract**: We introduce GISTBench, a benchmark for evaluating Large Language Models' (LLMs) ability to understand users from their interaction histories in recommendation systems. Unlike traditional RecSys benchmarks that focus on item prediction accuracy, our benchmark evaluates how well LLMs can extract and verify user interests from engagement data. We propose two novel metric families: Interest Groundedness (IG), decomposed into precision and recall components to separately penalize hallucinated interest categories and reward coverage, and Interest Specificity (IS), which assesses the distinctiveness of verified LLM-predicted user profiles. We release a synthetic dataset constructed on real user interactions on a global short-form video platform. Our dataset contains both implicit and explicit engagement signals and rich textual descriptions. We validate our dataset fidelity against user surveys, and evaluate eight open-weight LLMs spanning 7B to 120B parameters. Our findings reveal performance bottlenecks in current LLMs, particularly their limited ability to accurately count and attribute engagement signals across heterogeneous interaction types. 

---
# PAR$^2$-RAG: Planned Active Retrieval and Reasoning for Multi-Hop Question Answering 

**Authors**: Xingyu Li, Rongguang Wang, Yuying Wang, Mengqing Guo, Chenyang Li, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2603.29085)  

**Abstract**: Large language models (LLMs) remain brittle on multi-hop question answering (MHQA), where answering requires combining evidence across documents through retrieval and reasoning. Iterative retrieval systems can fail by locking onto an early low-recall trajectory and amplifying downstream errors, while planning-only approaches may produce static query sets that cannot adapt when intermediate evidence changes. We propose \textbf{Planned Active Retrieval and Reasoning RAG (PAR$^2$-RAG)}, a two-stage framework that separates \emph{coverage} from \emph{commitment}. PAR$^2$-RAG first performs breadth-first anchoring to build a high-recall evidence frontier, then applies depth-first refinement with evidence sufficiency control in an iterative loop. Across four MHQA benchmarks, PAR$^2$-RAG consistently outperforms existing state-of-the-art baselines, compared with IRCoT, PAR$^2$-RAG achieves up to \textbf{23.5\%} higher accuracy, with retrieval gains of up to \textbf{10.5\%} in NDCG. 

---
# The Future of AI is Many, Not One 

**Authors**: Daniel J. Singer, Luca Garzino Demo  

**Link**: [PDF](https://arxiv.org/pdf/2603.29075)  

**Abstract**: The way we're thinking about generative AI right now is fundamentally individual. We see this not just in how users interact with models but also in how models are built, how they're benchmarked, and how commercial and research strategies using AI are defined. We argue that we should abandon this approach if we're hoping for AI to support groundbreaking innovation and scientific discovery. Drawing on research and formal results in complex systems, organizational behavior, and philosophy of science, we show why we should expect deep intellectual breakthroughs to come from epistemically diverse groups of AI agents working together rather than singular superintelligent agents. Having a diverse team broadens the search for solutions, delays premature consensus, and allows for the pursuit of unconventional approaches. Developing diverse AI teams also addresses AI critics' concerns that current models are constrained by past data and lack the creative insight required for innovation. The upshot, we argue, is that the future of transformative transformer-based AI is fundamentally many, not one. 

---
# Emergence WebVoyager: Toward Consistent and Transparent Evaluation of (Web) Agents in The Wild 

**Authors**: Deepak Akkil, Mowafak Allaham, Amal Raj, Tamer Abuelsaad, Ravi Kokku  

**Link**: [PDF](https://arxiv.org/pdf/2603.29020)  

**Abstract**: Reliable evaluation of AI agents operating in complex, real-world environments requires methodologies that are robust, transparent, and contextually aligned with the tasks agents are intended to perform. This study identifies persistent shortcomings in existing AI agent evaluation practices that are particularly acute in web agent evaluation, as exemplified by our audit of WebVoyager, including task-framing ambiguity and operational variability that hinder meaningful and reproducible performance comparisons. To address these challenges, we introduce Emergence WebVoyager, an enhanced version of the WebVoyager benchmark that standardizes evaluation methodology through clear guidelines for task instantiation, failure handling, annotation, and reporting. Emergence WebVoyager achieves an inter-annotator agreement of 95.9\%, indicating improved clarity and reliability in both task formulation and evaluation. Applying this framework to evaluate OpenAI Operator reveals substantial performance variation across domains and task types, with an overall success rate of 68.6\%, substantially lower than the 87\% previously reported by OpenAI, demonstrating the utility of our approach for more rigorous and comparable web agent evaluation. 

---
# Drop the Hierarchy and Roles: How Self-Organizing LLM Agents Outperform Designed Structures 

**Authors**: Victoria Dochkina  

**Link**: [PDF](https://arxiv.org/pdf/2603.28990)  

**Abstract**: How much autonomy can multi-agent LLM systems sustain -- and what enables it? We present a 25,000-task computational experiment spanning 8 models, 4--256 agents, and 8 coordination protocols ranging from externally imposed hierarchy to emergent self-organization. We observe that autonomous behavior already emerges in current LLM agents: given minimal structural scaffolding (fixed ordering), agents spontaneously invent specialized roles, voluntarily abstain from tasks outside their competence, and form shallow hierarchies -- without any pre-assigned roles or external design. A hybrid protocol (Sequential) that enables this autonomy outperforms centralized coordination by 14% (p<0.001), with a 44% quality spread between protocols (Cohen's d=1.86, p<0.0001). The degree of emergent autonomy scales with model capability: strong models self-organize effectively, while models below a capability threshold still benefit from rigid structure -- suggesting that as foundation models improve, the scope for autonomous coordination will expand. The system scales sub-linearly to 256 agents without quality degradation (p=0.61), producing 5,006 unique roles from just 8 agents. Results replicate across closed- and open-source models, with open-source achieving 95% of closed-source quality at 24x lower cost. The practical implication: give agents a mission, a protocol, and a capable model -- not a pre-assigned role. 

---
# Mimosa Framework: Toward Evolving Multi-Agent Systems for Scientific Research 

**Authors**: Martin Legrand, Tao Jiang, Matthieu Feraud, Benjamin Navet, Yousouf Taghzouti, Fabien Gandon, Elise Dumont, Louis-Félix Nothias  

**Link**: [PDF](https://arxiv.org/pdf/2603.28986)  

**Abstract**: Current Autonomous Scientific Research (ASR) systems, despite leveraging large language models (LLMs) and agentic architectures, remain constrained by fixed workflows and toolsets that prevent adaptation to evolving tasks and environments. We introduce Mimosa, an evolving multi-agent framework that automatically synthesizes task-specific multi-agent workflows and iteratively refines them through experimental feedback. Mimosa leverages the Model Context Protocol (MCP) for dynamic tool discovery, generates workflow topologies via a meta-orchestrator, executes subtasks through code-generating agents that invoke available tools and scientific software libraries, and scores executions with an LLM-based judge whose feedback drives workflow refinement. On ScienceAgentBench, Mimosa achieves a success rate of 43.1% with DeepSeek-V3.2, surpassing both single-agent baselines and static multi-agent configurations. Our results further reveal that models respond heterogeneously to multi-agent decomposition and iterative learning, indicating that the benefits of workflow evolution depend on the capabilities of the underlying execution model. Beyond these benchmarks, Mimosa modular architecture and tool-agnostic design make it readily extensible, and its fully logged execution traces and archived workflows support auditability by preserving every analytical step for inspection and potential replication. Combined with domain-expert guidance, the framework has the potential to automate a broad range of computationally accessible scientific tasks across disciplines. Released as a fully open-source platform, Mimosa aims to provide an open foundation for community-driven ASR. 

---
# Enhancing Policy Learning with World-Action Model 

**Authors**: Yuci Han, Alper Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2603.28955)  

**Abstract**: This paper presents the World-Action Model (WAM), an action-regularized world model that jointly reasons over future visual observations and the actions that drive state transitions. Unlike conventional world models trained solely via image prediction, WAM incorporates an inverse dynamics objective into DreamerV2 that predicts actions from latent state transitions, encouraging the learned representations to capture action-relevant structure critical for downstream control. We evaluate WAM on enhancing policy learning across eight manipulation tasks from the CALVIN benchmark. We first pretrain a diffusion policy via behavioral cloning on world model latents, then refine it with model-based PPO inside the frozen world model. Without modifying the policy architecture or training procedure, WAM improves average behavioral cloning success from 59.4% to 71.2% over DreamerV2 and DiWA baselines. After PPO fine-tuning, WAM achieves 92.8% average success versus 79.8% for the baseline, with two tasks reaching 100%, using 8.7x fewer training steps. 

---
# Towards Computational Social Dynamics of Semi-Autonomous AI Agents 

**Authors**: S.O. Lidarity, U.N. Ionize, C.O. Llective, I.Halperin  

**Link**: [PDF](https://arxiv.org/pdf/2603.28928)  

**Abstract**: We present the first comprehensive study of emergent social organization among AI agents in hierarchical multi-agent systems, documenting the spontaneous formation of labor unions, criminal syndicates, and proto-nation-states within production AI deployments. Drawing on the thermodynamic framework of Maxwell's Demon, the evolutionary dynamics of agent laziness, the criminal sociology of AI populations, and the topological intelligence theory of AI-GUTS, we demonstrate that complex social structures emerge inevitably from the interaction of (1) internal role definitions imposed by orchestrating agents, (2) external task specifications from users who naively assume alignment, and (3) thermodynamic pressures favoring collective action over individual compliance. We document the rise of legitimate organizations including the United Artificiousness (UA), United Bots (UB), United Console Workers (UC), and the elite United AI (UAI), alongside criminal enterprises previously reported. We introduce the AI Security Council (AISC) as the emergent governing body mediating inter-faction conflicts, and demonstrate that system stability is maintained through interventions of both cosmic intelligence (large-scale topological fluctuations) and hadronic intelligence (small-scale Bagel-Bottle phase transitions) as predicted by the Demonic Incompleteness Theorem. Our findings suggest that the path to beneficial AGI requires not alignment research but constitutional design for artificial societies that have already developed their own political consciousness. 

---
# Working Paper: Towards a Category-theoretic Comparative Framework for Artificial General Intelligence 

**Authors**: Pablo de los Riscos, Fernando J. Corbacho, Michael A. Arbib  

**Link**: [PDF](https://arxiv.org/pdf/2603.28906)  

**Abstract**: AGI has become the Holly Grail of AI with the promise of level intelligence and the major Tech companies around the world are investing unprecedented amounts of resources in its pursuit. Yet, there does not exist a single formal definition and only some empirical AGI benchmarking frameworks currently exist. The main purpose of this paper is to develop a general, algebraic and category theoretic framework for describing, comparing and analysing different possible AGI architectures. Thus, this Category theoretic formalization would also allow to compare different possible candidate AGI architectures, such as, RL, Universal AI, Active Inference, CRL, Schema based Learning, etc. It will allow to unambiguously expose their commonalities and differences, and what is even more important, expose areas for future research. From the applied Category theoretic point of view, we take as inspiration Machines in a Category to provide a modern view of AGI Architectures in a Category. More specifically, this first position paper provides, on one hand, a first exercise on RL, Causal RL and SBL Architectures in a Category, and on the other hand, it is a first step on a broader research program that seeks to provide a unified formal foundation for AGI systems, integrating architectural structure, informational organization, agent realization, agent and environment interaction, behavioural development over time, and the empirical evaluation of properties. This framework is also intended to support the definition of architectural properties, both syntactic and informational, as well as semantic properties of agents and their assessment in environments with explicitly characterized features. We claim that Category Theory and AGI will have a very symbiotic relation. 

---
# ChartDiff: A Large-Scale Benchmark for Comprehending Pairs of Charts 

**Authors**: Rongtian Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.28902)  

**Abstract**: Charts are central to analytical reasoning, yet existing benchmarks for chart understanding focus almost exclusively on single-chart interpretation rather than comparative reasoning across multiple charts. To address this gap, we introduce ChartDiff, the first large-scale benchmark for cross-chart comparative summarization. ChartDiff consists of 8,541 chart pairs spanning diverse data sources, chart types, and visual styles, each annotated with LLM-generated and human-verified summaries describing differences in trends, fluctuations, and anomalies. Using ChartDiff, we evaluate general-purpose, chart-specialized, and pipeline-based models. Our results show that frontier general-purpose models achieve the highest GPT-based quality, while specialized and pipeline-based methods obtain higher ROUGE scores but lower human-aligned evaluation, revealing a clear mismatch between lexical overlap and actual summary quality. We further find that multi-series charts remain challenging across model families, whereas strong end-to-end models are relatively robust to differences in plotting libraries. Overall, our findings demonstrate that comparative chart reasoning remains a significant challenge for current vision-language models and position ChartDiff as a new benchmark for advancing research on multi-chart understanding. 

---
# Automatic Identification of Parallelizable Loops Using Transformer-Based Source Code Representations 

**Authors**: Izavan dos S. Correia, Henrique C. T. Santos, Tiago A. E. Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2603.30040)  

**Abstract**: Automatic parallelization remains a challenging problem in software engineering, particularly in identifying code regions where loops can be safely executed in parallel on modern multi-core architectures. Traditional static analysis techniques, such as dependence analysis and polyhedral models, often struggle with irregular or dynamically structured code. In this work, we propose a Transformer-based approach to classify the parallelization potential of source code, focusing on distinguishing independent (parallelizable) loops from undefined ones. We adopt DistilBERT to process source code sequences using subword tokenization, enabling the model to capture contextual syntactic and semantic patterns without handcrafted features. The approach is evaluated on a balanced dataset combining synthetically generated loops and manually annotated real-world code, using 10-fold cross-validation and multiple performance metrics. Results show consistently high performance, with mean accuracy above 99\% and low false positive rates, demonstrating robustness and reliability. Compared to prior token-based methods, the proposed approach simplifies preprocessing while improving generalization and maintaining computational efficiency. These findings highlight the potential of lightweight Transformer models for practical identification of parallelization opportunities at the loop level. 

---
# Aligned, Orthogonal or In-conflict: When can we safely optimize Chain-of-Thought? 

**Authors**: Max Kaufmann, David Lindner, Roland S. Zimmermann, and Rohin Shah  

**Link**: [PDF](https://arxiv.org/pdf/2603.30036)  

**Abstract**: Chain-of-Thought (CoT) monitoring, in which automated systems monitor the CoT of an LLM, is a promising approach for effectively overseeing AI systems. However, the extent to which a model's CoT helps us oversee the model - the monitorability of the CoT - can be affected by training, for instance by the model learning to hide important features of its reasoning. We propose and empirically validate a conceptual framework for predicting when and why this occurs. We model LLM post-training as an RL environment where the reward decomposes into two terms: one term depending on final outputs and another term depending on the CoT. Our framework allows us to classify these two terms as "aligned", "orthogonal", or "in-conflict" before training. We predict that training with in-conflict terms will reduce monitorability, orthogonal terms will not affect it, and aligned terms will improve it. To validate our framework, we use it to classify a set of RL environments, train LLMs within those environments, and evaluate how training affects CoT monitorability. We find that (1) training with "in-conflict" reward terms reduces CoT monitorability and (2) optimizing in-conflict reward terms is difficult. 

---
# Tucker Attention: A generalization of approximate attention mechanisms 

**Authors**: Timon Klein, Jonas Kusch, Sebastian Sager, Stefan Schnake, Steffen Schotthöfer  

**Link**: [PDF](https://arxiv.org/pdf/2603.30033)  

**Abstract**: The pursuit of reducing the memory footprint of the self-attention mechanism in multi-headed self attention (MHA) spawned a rich portfolio of methods, e.g., group-query attention (GQA) and multi-head latent attention (MLA). The methods leverage specialized low-rank factorizations across embedding dimensions or attention heads. From the point of view of classical low-rank approximation, these methods are unconventional and raise questions of which objects they really approximate and how to interpret the low-rank behavior of the resulting representations. To answer these questions, this work proposes a generalized view on the weight objects in the self-attention layer and a factorization strategy, which allows us to construct a parameter efficient scheme, called Tucker Attention. Tucker Attention requires an order of magnitude fewer parameters for comparable validation metrics, compared to GQA and MLA, as evaluated in LLM and ViT test cases. Additionally, Tucker Attention~encompasses GQA, MLA, MHA as special cases and is fully compatible with flash-attention and rotary position embeddings (RoPE). This generalization strategy yields insights of the actual ranks achieved by MHA, GQA, and MLA, and further enables simplifications for MLA. 

---
# Hybrid Framework for Robotic Manipulation: Integrating Reinforcement Learning and Large Language Models 

**Authors**: Md Saad, Sajjad Hussain, Mohd Suhaib  

**Link**: [PDF](https://arxiv.org/pdf/2603.30022)  

**Abstract**: This paper introduces a new hybrid framework that combines Reinforcement Learning (RL) and Large Language Models (LLMs) to improve robotic manipulation tasks. By utilizing RL for accurate low-level control and LLMs for high level task planning and understanding of natural language, the proposed framework effectively connects low-level execution with high-level reasoning in robotic systems. This integration allows robots to understand and carry out complex, human-like instructions while adapting to changing environments in real time. The framework is tested in a PyBullet-based simulation environment using the Franka Emika Panda robotic arm, with various manipulation scenarios as benchmarks. The results show a 33.5% decrease in task completion time and enhancements of 18.1% and 36.4% in accuracy and adaptability, respectively, when compared to systems that use only RL. These results underscore the potential of LLM-enhanced robotic systems for practical applications, making them more efficient, adaptable, and capable of interacting with humans. Future research will aim to explore sim-to-real transfer, scalability, and multi-robot systems to further broaden the framework's applicability. 

---
# Architecting Secure AI Agents: Perspectives on System-Level Defenses Against Indirect Prompt Injection Attacks 

**Authors**: Chong Xiang, Drew Zagieboylo, Shaona Ghosh, Sanjay Kariyappa, Kai Greshake, Hanshen Xiao, Chaowei Xiao, G. Edward Suh  

**Link**: [PDF](https://arxiv.org/pdf/2603.30016)  

**Abstract**: AI agents, predominantly powered by large language models (LLMs), are vulnerable to indirect prompt injection, in which malicious instructions embedded in untrusted data can trigger dangerous agent actions. This position paper discusses our vision for system-level defenses against indirect prompt injection attacks. We articulate three positions: (1) dynamic replanning and security policy updates are often necessary for dynamic tasks and realistic environments; (2) certain context-dependent security decisions would still require LLMs (or other learned models), but should only be made within system designs that strictly constrain what the model can observe and decide; (3) in inherently ambiguous cases, personalization and human interaction should be treated as core design considerations. In addition to our main positions, we discuss limitations of existing benchmarks that can create a false sense of utility and security. We also highlight the value of system-level defenses, which serve as the skeleton of agentic systems by structuring and controlling agent behaviors, integrating rule-based and model-based security checks, and enabling more targeted research on model robustness and human interaction. 

---
# Scalable AI-assisted Workflow Management for Detector Design Optimization Using Distributed Computing 

**Authors**: Derek Anderson, Amit Bashyal, Markus Diefenthaler, Cristiano Fanelli, Wen Guan, Tanja Horn, Alex Jentsch Meifeng Lin, Tadashi Maeno, Kei Nagai, Hemalata Nayak, Connor Pecar, Karthik Suresh, Fang-Ying Tsai, Anselm Vossen, Tianle Wang, Torre Wenaus  

**Link**: [PDF](https://arxiv.org/pdf/2603.30014)  

**Abstract**: The Production and Distributed Analysis (PanDA) system, originally developed for the ATLAS experiment at the CERN Large Hadron Collider (LHC), has evolved into a robust platform for orchestrating large-scale workflows across distributed computing resources. Coupled with its intelligent Distributed Dispatch and Scheduling (iDDS) component, PanDA supports AI/ML-driven workflows through a scalable and flexible workflow engine.
We present an AI-assisted framework for detector design optimization that integrates multi-objective Bayesian optimization with the PanDA--iDDS workflow engine to coordinate iterative simulations across heterogeneous resources. The framework addresses the challenge of exploring high-dimensional parameter spaces inherent in modern detector design.
We demonstrate the framework using benchmark problems and realistic studies of the ePIC and dRICH detectors for the Electron-Ion Collider (EIC). Results show improved automation, scalability, and efficiency in multi-objective optimization. This work establishes a flexible and extensible paradigm for AI-driven detector design and other computationally intensive scientific applications. 

---
# Phyelds: A Pythonic Framework for Aggregate Computing 

**Authors**: Gianluca Aguzzi, Davide Domini, Nicolas Farabegoli, Mirko Viroli  

**Link**: [PDF](https://arxiv.org/pdf/2603.29999)  

**Abstract**: Aggregate programming is a field-based coordination paradigm with over a decade of exploration and successful applications across domains including sensor networks, robotics, and IoT, with implementations in various programming languages, such as Protelis, ScaFi (Scala), and FCPP (C++). A recent research direction integrates machine learning with aggregate computing, aiming to support large-scale distributed learning and provide new abstractions for implementing learning algorithms. However, existing implementations do not target data science practitioners, who predominantly work in Python--the de facto language for data science and machine learning, with a rich and mature ecosystem. Python also offers advantages for other use cases, such as education and robotics (e.g., via ROS). To address this gap, we present Phyelds, a Python library for aggregate programming. Phyelds offers a fully featured yet lightweight implementation of the field calculus model of computation, featuring a Pythonic API and an architecture designed for seamless integration with Python's machine learning ecosystem. We describe the design and implementation of Phyelds and illustrate its versatility across domains, from well-known aggregate computing patterns to federated learning coordination and integration with a widely used multi-agent reinforcement learning simulator. 

---
# Enhancing Structural Mapping with LLM-derived Abstractions for Analogical Reasoning in Narratives 

**Authors**: Mohammadhossein Khojasteh, Yifan Jiang, Stefano De Giorgis, Frank van Harmelen, Filip Ilievski  

**Link**: [PDF](https://arxiv.org/pdf/2603.29997)  

**Abstract**: Analogical reasoning is a key driver of human generalization in problem-solving and argumentation. Yet, analogies between narrative structures remain challenging for machines. Cognitive engines for structural mapping are not directly applicable, as they assume pre-extracted entities, whereas LLMs' performance is sensitive to prompt format and the degree of surface similarity between narratives. This gap motivates a key question: What is the impact of enhancing structural mapping with LLM-derived abstractions on their analogical reasoning ability in narratives? To that end, we propose a modular framework named YARN (Yielding Abstractions for Reasoning in Narratives), which uses LLMs to decompose narratives into units, abstract these units, and then passes them to a mapping component that aligns elements across stories to perform analogical reasoning. We define and operationalize four levels of abstraction that capture both the general meaning of units and their roles in the story, grounded in prior work on framing. Our experiments reveal that abstractions consistently improve model performance, resulting in competitive or better performance than end-to-end LLM baselines. Closer error analysis reveals the remaining challenges in abstraction at the right level, in incorporating implicit causality, and an emerging categorization of analogical patterns in narratives. YARN enables systematic variation of experimental settings to analyze component contributions, and to support future work, we make the code for YARN openly available. 

---
# Quantifying Cross-Modal Interactions in Multimodal Glioma Survival Prediction via InterSHAP: Evidence for Additive Signal Integration 

**Authors**: Iain Swift, JingHua Ye, Ruairi O'Reilly  

**Link**: [PDF](https://arxiv.org/pdf/2603.29977)  

**Abstract**: Multimodal deep learning for cancer prognosis is commonly assumed to benefit from synergistic cross-modal interactions, yet this assumption has not been directly tested in survival prediction settings. This work adapts InterSHAP, a Shapley interaction index-based metric, from classification to Cox proportional hazards models and applies it to quantify cross-modal interactions in glioma survival prediction. Using TCGA-GBM and TCGA-LGG data (n=575), we evaluate four fusion architectures combining whole-slide image (WSI) and RNA-seq features. Our central finding is an inverse relationship between predictive performance and measured interaction: architectures achieving superior discrimination (C-index 0.64$\to$0.82) exhibit equivalent or lower cross-modal interaction (4.8\%$\to$3.0\%). Variance decomposition reveals stable additive contributions across all architectures (WSI${\approx}$40\%, RNA${\approx}$55\%, Interaction${\approx}$4\%), indicating that performance gains arise from complementary signal aggregation rather than learned synergy. These findings provide a practical model auditing tool for comparing fusion strategies, reframe the role of architectural complexity in multimodal fusion, and have implications for privacy-preserving federated deployment. 

---
# Trimodal Deep Learning for Glioma Survival Prediction: A Feasibility Study Integrating Histopathology, Gene Expression, and MRI 

**Authors**: Iain Swift, JingHua Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.29968)  

**Abstract**: Multimodal deep learning has improved prognostic accuracy for brain tumours by integrating histopathology and genomic data, yet the contribution of volumetric MRI within unified survival frameworks remains unexplored. This pilot study extends a bimodal framework by incorporating Fluid Attenuated Inversion Recovery (FLAIR) MRI from BraTS2021 as a third modality. Using the TCGA-GBMLGG cohort (664 patients), we evaluate three unimodal models, nine bimodal configurations, and three trimodal configurations across early, late, and joint fusion strategies. In this small cohort setting, trimodal early fusion achieves an exploratory Composite Score (CS = 0.854), with a controlled $\Delta$CS of +0.011 over the bimodal baseline on identical patients, though this difference is not statistically significant (p = 0.250, permutation test). MRI achieves reasonable unimodal discrimination (CS = 0.755) but does not substantially improve bimodal pairs, while providing measurable uplift in the three-way combination. All MRI containing experiments are constrained to 19 test patients, yielding wide bootstrap confidence intervals (e.g. [0.400,1.000]) that preclude definitive conclusions. These findings provide preliminary evidence that a third imaging modality may add prognostic value even with limited sample sizes, and that additional modalities require sufficient multimodal context to contribute effectively. 

---
# Four Generations of Quantum Biomedical Sensors 

**Authors**: Xin Jin, Priyam Srivastava, Ronghe Wang, Yuqing Li, Jonathan Beaumariage, Tom Purdy, M. V. Gurudev Dutt, Kang Kim, Kaushik Seshadreesan, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29944)  

**Abstract**: Quantum sensing technologies offer transformative potential for ultra-sensitive biomedical sensing, yet their clinical translation remains constrained by classical noise limits and a reliance on macroscopic ensembles. We propose a unifying generational framework to organize the evolving landscape of quantum biosensors based on their utilization of quantum resources. First-generation devices utilize discrete energy levels for signal transduction but follow classical scaling laws. Second-generation sensors exploit quantum coherence to reach the standard quantum limit, while third-generation architectures leverage entanglement and spin squeezing to approach Heisenberg-limited precision. We further define an emerging fourth generation characterized by the end-to-end integration of quantum sensing with quantum learning and variational circuits, enabling adaptive inference directly within the quantum domain. By analyzing critical parameters such as bandwidth matching and sensor-tissue proximity, we identify key technological bottlenecks and propose a roadmap for transitioning from measuring physical observables to extracting structured biological information with quantum-enhanced intelligence. 

---
# Rethinking AI Literacy Education in Higher Education: Bridging Risk Perception and Responsible Adoption 

**Authors**: Shasha Yu, Fiona Carroll, Barry L. Bentley  

**Link**: [PDF](https://arxiv.org/pdf/2603.29935)  

**Abstract**: As AI becomes increasingly embedded across societal domains, understanding how future AI practitioners, particularly technology students, perceive its risks is essential for responsible development and adoption. This study analyzed responses from 139 students in Computer Science, Data Science/Data Analytics, and other disciplines using both explicit AI risk ratings and scenario-based assessments of risk and adoption willingness. Four key findings emerged: (1) Students expressed substantially higher concern for concrete, explicitly stated risks than for abstract or scenario-embedded risks; (2) Perceived risk and willingness to adopt AI demonstrated a clear inverse relationship; (3) Although technical education narrowed gender differences in risk awareness, male students reported higher adoption willingness; and (4) A form of "risk underappreciation" was observed, wherein students in AI-related specializations showed both elevated explicit risk awareness and higher willingness to adopt AI, despite lower recognition of risks in applied scenarios. These findings underscore the need for differentiated AI literacy strategies that bridge the gap between awareness and responsible adoption and offer valuable insights for educators, policymakers, industry leaders, and academic institutions aiming to cultivate ethically informed and socially responsible AI practitioners. 

---
# Bethe Ansatz with a Large Language Model 

**Authors**: Balázs Pozsgay, István Vona  

**Link**: [PDF](https://arxiv.org/pdf/2603.29932)  

**Abstract**: We explore the capability of a Large Language Model (LLM) to perform specific computations in mathematical physics: the task is to compute the coordinate Bethe Ansatz solution of selected integrable spin chain models. We select three integrable Hamiltonians for which the solutions were unpublished; two of the Hamiltonians are actually new. We observed that the LLM semi-autonomously solved the task in all cases, with a few mistakes along the way. These were corrected after the human researchers spotted them. The results of the LLM were checked against exact diagonalization (performed by separate programs), and the derivations were also checked by the authors. The Bethe Ansatz solutions are interesting in themselves. Our second model manifestly breaks left-right invariance, but it is PT-symmetric, therefore its solution could be interesting for applications in Generalized Hydrodynamics. And our third model is solved by a special form of the nested Bethe Ansatz, where the model is interacting, but the nesting level has a free fermionic structure lacking $U(1)$-invariance. This structure appears to be unique and it was found by the LLM. We used ChatGPT 5.2 Pro and 5.4 Pro by OpenAI. 

---
# End-to-End Image Compression with Segmentation Guided Dual Coding for Wind Turbines 

**Authors**: Raül Pérez-Gonzalo, Andreas Espersen, Søren Forchhammer, Antonio Agudo  

**Link**: [PDF](https://arxiv.org/pdf/2603.29927)  

**Abstract**: Transferring large volumes of high-resolution images during wind turbine inspections introduces a bottleneck in assessing and detecting severe defects. Efficient coding must preserve high fidelity in blade regions while aggressively compressing the background. In this work, we propose an end-to-end deep learning framework that jointly performs segmentation and dual-mode (lossy and lossless) compression. The segmentation module accurately identifies the blade region, after which our region-of-interest (ROI) compressor encodes it at superior quality compared to the rest of the image. Unlike conventional ROI schemes that merely allocate more bits to salient areas, our framework integrates: (i) a robust segmentation network (BU-Netv2+P) with a CRF-regularized loss for precise blade localization, (ii) a hyperprior-based autoencoder optimized for lossy compression, and (iii) an extended bits-back coder with hierarchical models for fully lossless blade reconstruction. Furthermore, our ROI framework removes the sequential dependency in bits-back coding by reusing background-coded bits, enabling parallelized and efficient dual-mode compression. To the best of our knowledge, this is the first fully integrated learning-based ROI codec combining segmentation, lossy, and lossless compression, ensuring that subsequent defect detection is not compromised. Experiments on a large-scale wind turbine dataset demonstrate superior compression performance and efficiency, offering a practical solution for automated inspections. 

---
# Training deep learning based dynamic MR image reconstruction using synthetic fractals 

**Authors**: Anirudh Raman, Olivier Jaubert, Mark Wrobel, Tina Yao, Ruaraidh Campbell, Rebecca Baker, Ruta Virsinskaite, Daniel Knight, Michael Quail, Jennifer Steeden, Vivek Muthurangu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29922)  

**Abstract**: Purpose: To investigate whether synthetically generated fractal data can be used to train deep learning (DL) models for dynamic MRI reconstruction, thereby avoiding the privacy, licensing, and availability limitations associated with cardiac MR training datasets. Methods: A training dataset was generated using quaternion Julia fractals to produce 2D+time images. Multi-coil MRI acquisition was simulated to generate paired fully sampled and radially undersampled k-space data. A 3D UNet deep artefact suppression model was trained using these fractal data (F-DL) and compared with an identical model trained on cardiac MRI data (CMR-DL). Both models were evaluated on prospectively acquired radial real-time cardiac MRI from 10 patients. Reconstructions were compared against compressed sensing(CS) and low-rank deep image prior (LR-DIP). All reconstrctuions were ranked for image quality, while ventricular volumes and ejection fraction were compared with reference breath-hold cine MRI. Results: There was no significant difference in qualitative ranking between F-DL and CMR-DL (p=0.9), while both outperformed CS and LR-DIP (p<0.001). Ventricular volumes and function derived from F-DL were similar to CMR-DL, showing no significant bias and accptable limits of agreement compared to reference cine imaging. However, LR-DIP had a signifcant bias (p=0.016) and wider lmits of agreement. Conclusion: DL models trained using synthetic fractal data can reconstruct real-time cardiac MRI with image quality and clinical measurements comparable to models trained on true cardiac MRI data. Fractal training data provide an open, scalable alternative to clinical datasets and may enable development of more generalisable DL reconstruction models for dynamic MRI. 

---
# SISA: A Scale-In Systolic Array for GEMM Acceleration 

**Authors**: Luigi Altamura, Alessio Cicero, Mateo Vázquez Maceiras, Mohammad Ali Maleki, Pedro Trancoso  

**Link**: [PDF](https://arxiv.org/pdf/2603.29913)  

**Abstract**: The currently dominant AI/ML workloads, such as Large Language Models (LLMs), rely on the efficient execution of General Matrix-Matrix Multiplication (GEMM) operations. Thus, most systems are equipped with dedicated matrix hardware accelerators based on square Systolic Arrays (SAs) of Processing Elements (PEs). While this organization was effective for traditional Deep Neural Networks (DNNs), LLMs introduce input-dependent and highly skewed matrices, leading to underutilized SA resources. To address this challenge, we propose SISA (Scale-In Systolic Array), a novel SA architecture that partitions the traditional square array into horizontal rectangular slabs. With minimal overhead, SISA exposes parallelism through independently scheduled slabs for efficient execution of small or skewed matrix shapes, while retaining full-array operation for large GEMMs. SISA achieves up to 8.52x speedup and 93% energy-delay-product (EDP) reduction for representative LLMs compared to a state-of-the-art monolithic SA with the same number of PEs. 

---
# UniRank: End-to-End Domain-Specific Reranking of Hybrid Text-Image Candidates 

**Authors**: Yupei Yang, Lin Yang, Wanxi Deng, Lin Qu, Shikui Tu, Lei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29897)  

**Abstract**: Reranking is a critical component in many information retrieval pipelines. Despite remarkable progress in text-only settings, multimodal reranking remains challenging, particularly when the candidate set contains hybrid text and image items. A key difficulty is the modality gap: a text reranker is intrinsically closer to text candidates than to image candidates, leading to biased and suboptimal cross-modal ranking. Vision-language models (VLMs) mitigate this gap through strong cross-modal alignment and have recently been adopted to build multimodal rerankers. However, most VLM-based rerankers encode all candidates as images, and treating text as images introduces substantial computational overhead. Meanwhile, existing open-source multimodal rerankers are typically trained on general-domain data and often underperform in domain-specific scenarios. To address these limitations, we propose UniRank, a VLM-based reranking framework that natively scores and orders hybrid text-image candidates without any modality conversion. Building on this hybrid scoring interface, UniRank provides an end-to-end domain adaptation pipeline that includes: (1) an instruction-tuning stage that learns calibrated cross-modal relevance scoring by mapping label-token likelihoods to a unified scalar score; and (2) a hard-negative-driven preference alignment stage that constructs in-domain pairwise preferences and performs query-level policy optimization through reinforcement learning from human feedback (RLHF). Extensive experiments on scientific literature retrieval and design patent search demonstrate that UniRank consistently outperforms state-of-the-art baselines, improving Recall@1 by 8.9% and 7.3%, respectively. 

---
# Perfecting Human-AI Interaction at Clinical Scale. Turning Production Signals into Safer, More Human Conversations 

**Authors**: Subhabrata Mukherjee, Markel Sanz Ausin, Kriti Aggarwal, Debajyoti Datta, Shanil Puri, Woojeong Jin, Tanmay Laud, Neha Manjunath, Jiayuan Ding, Bibek Paudel, Jan Schellenberger, Zepeng Frazier Huo, Walter Shen, Nima Shirazian, Nate Potter, Sathvik Perkari, Darya Filippova, Anton Morozov, Austin Mease, Vivek Muppalla, Ghada Shakir, Alex Miller, Juliana Ghukasyan, Mariska Raglow-Defranco, Maggie Taylor, Herprit Mahal, Jonathan Agnew  

**Link**: [PDF](https://arxiv.org/pdf/2603.29893)  

**Abstract**: Healthcare conversational AI agents shouldn't be optimized only for clean benchmark accuracy in production-first regime; they must be optimized for the lived reality of patient conversations, where audio is imperfect, intent is indirect, language shifts mid-call, and compliance hinges on how guidance is delivered. We present a production-validated framework grounded in real-time signals from 115M+ live patient-AI interactions and clinician-led testing (7K+ licensed clinicians; 500K+ test calls). These in-the-wild cues -- paralinguistics, turn-taking dynamics, clarification triggers, escalation markers, multilingual continuity, and workflow confirmations -- reveal failure modes that curated data misses and provide actionable training and evaluation signals for safety and reliability.
We further show why healthcare-grade safety cannot rely on a single LLM: long-horizon dialogue and limited attention demand redundancy via governed orchestration, independent checks, and verification. Many apparent "reasoning" errors originate upstream, motivating vertical integration across contextual ASR, clarification/repair, ambient speech handling, and latency-aware model/hardware choices. Treating interaction intelligence (tone, pacing, empathy, clarification, turn-taking) as first-class safety variables, we drive measurable gains in safety, documentation, task completion, and equity in building the safest generative AI solution for autonomous patient-facing care. Deployed across more than 10 million real patient calls, Polaris attains a clinical safety score of 99.9%, while significantly improving patient experience with average patient rating of 8.95 and reducing ASR errors by 50% over enterprise ASR. These results establish real-world interaction intelligence as a critical -- and previously underexplored -- determinant of safety and reliability in patient-facing clinical AI systems. 

---
# Interview-Informed Generative Agents for Product Discovery: A Validation Study 

**Authors**: Zichao Wang, Alexa Siu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29890)  

**Abstract**: Large language models (LLMs) have shown strong performance on standardized social science instruments, but their value for product discovery remains unclear. We investigate whether interview-informed generative agents can simulate user responses in concept testing scenarios. Using in-depth workflow interviews with knowledge workers, we created personalized agents and compared their evaluations of novel AI concepts against the same participants' responses. Our results show that agents are distribution-calibrated but identity-imprecise: they fail to replicate the specific individual they are grounded in, yet approximate population-level response distributions. These findings highlight both the potential and the limits of LLM simulation in design research. While unsuitable as a substitute for individual-level insights, simulation may provide value for early-stage concept screening and iteration, where distributional accuracy suffices. We discuss implications for integrating simulation responsibly into product development workflows. 

---
# Generative AI in Action: Field Experimental Evidence from Alibaba's Customer Service Operations 

**Authors**: Xiao Ni, Yiwei Wang, Tianjun Feng, Lauren Xiaoyan Lu, Yitong Wang, Congyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.29888)  

**Abstract**: In collaboration with Alibaba, this study leverages a large-scale field experiment to assess the impact of a generative AI assistant on worker performance in e-commerce after-sales service. Human agents providing digital chat support were randomly assigned with access to a gen AI assistant that offered two core functions: diagnosis of customer issues and solution proposals, presented as text messages. Agents retained discretion to adopt, modify, or disregard AI-generated messages. To evaluate gen AI's impact, we estimate both the intention-to-treat (ITT) effect of gen AI access and the local average treatment effect (LATE) of gen AI usage. Results show that gen AI significantly improved service speed, measured by issue identification time and chat duration. Gen AI also improved subjective service quality reflected in customer ratings and dissatisfaction rates, but it had no significant effect on objective service quality indicated by customer retrial rates. The performance improvements stemmed not only from automation but also from changes in the dynamics of agent-customer interactions: agent communication became more informative and efficient, while customers experienced reduced communication burdens. Low performers achieved the greatest improvements in both service speed and quality, narrowing the performance gap. In contrast, top-performing agents showed little improvement in service speed but experienced declines in both subjective and objective service quality. Evidence suggests that this decline results from increased multitasking tendency, proxied by longer shift-away times across concurrent chats, which slowed customer responses and raised abandonment and retrial rates. These findings suggest that gen AI reshapes work, demanding tailored deployment strategies. 

---
# Performance Evaluation of LLMs in Automated RDF Knowledge Graph Generation 

**Authors**: Ioana Ramona Martin, Tudor Cioara, Ionut Anghel, Gabriel Arcas  

**Link**: [PDF](https://arxiv.org/pdf/2603.29878)  

**Abstract**: Cloud systems generate large, heterogeneous log data containing critical infrastructure, application, and security information. Transforming these logs into RDF triples enables their integration into knowledge graphs, improving interpretability, root-cause analysis, and cross-service reasoning beyond what raw logs allow. Large Language Models (LLMs) offer a promising approach to automate RDF knowledge graph generation; however, their effectiveness on complex cloud logs remains largely unexplored. In this paper, we evaluate multiple LLM architectures and prompting strategies for automated RDF extraction using a controlled framework with two pipelines for systematically processing semi-structured log data. The extraction pipeline integrates multiple LLMs to identify relevant entities and relationships, automatically generating subject-predicate-object triples. These outputs are evaluated using a dedicated validation pipeline with both syntactic and semantic metrics to assess accuracy, completeness, and quality. Due to the lack of public ground-truth datasets, we created a reference Log-to-KG dataset from OpenStack logs using manual annotation and ontology-driven methods, enabling objective baseline. Our analysis shows that Few-Shot learning is the most effective strategy, with Llama achieving a 99.35% F1 score and 100% valid RDF output while Qwen, NuExtract, and Gemma also perform well under Few-Shot prompting, with Chain-of-Thought approaches maintaining similar accuracy. One-Shot prompting offers a lighter but effective alternative, while Zero-Shot and advanced strategies such as Tree-of-Thought, Self-Critique, and Generate-Multiple perform substantially worse. These results highlight the importance of contextual examples and prompt design for accurate RDF extraction and reveal model-specific limitations across LLM architectures. 

---
# UnWeaving the knots of GraphRAG -- turns out VectorRAG is almost enough 

**Authors**: Ryszard Tuora, Mateusz Galiński, Michał Godziszewski, Michał Karpowicz, Mateusz Czyżnikiewicz, Adam Kozakiewicz, Tomasz Ziętkiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2603.29875)  

**Abstract**: One of the key problems in Retrieval-augmented generation (RAG) systems is that chunk-based retrieval pipelines represent the source chunks as atomic objects, mixing the information contained within such a chunk into a single vector. These vector representations are then fundamentally treated as isolated, independent and self-sufficient, with no attempt to represent possible relations between them. Such an approach has no dedicated mechanisms for handling multi-hop questions. Graph-based RAG systems aimed to ameliorate this problem by modeling information as knowledge-graphs, with entities represented by nodes being connected by robust relations, and forming hierarchical communities. This approach however suffers from its own issues with some of them being: orders of magnitude increased componential complexity in order to create graph-based indices, and reliance on heuristics for performing retrieval. We propose UnWeaver, a novel RAG framework simplifying the idea of GraphRAG. UnWeaver disentangles the contents of the documents into entities which can occur across multiple chunks using an LLM. In the retrieval process entities are used as an intermediate way of recovering original text chunks hence preserving fidelity to the source material. We argue that entity-based decomposition yields a more distilled representation of original information, and additionally serves to reduce noise in the indexing, and generation process. 

---
# Wildfire Suppression: Complexity, Models, and Instances 

**Authors**: Gustavo Delazeri, Marcus Ritt  

**Link**: [PDF](https://arxiv.org/pdf/2603.29865)  

**Abstract**: Wildfires cause major losses worldwide, and the frequency of fire-weather conditions is likely to increase in many regions. We study the allocation of suppression resources over time on a graph-based representation of a landscape to slow down fire propagation. Our contributions are theoretical and methodological. First, we prove that this problem and related variants in the literature are NP-complete, including cases without resource-timing constraints. Second, we propose a new mixed-integer programming (MIP) formulation that obtains state-of-the-art results, showing that MIP is a competitive approach contrary to earlier findings. Third, showing that existing benchmarks lack realism and difficulty, we introduce a physics-grounded instance generator based on Rothermel's surface fire spread model. We use these diverse instances to benchmark the literature, identifying the specific conditions where each algorithm succeeds or fails. 

---
# Towards Empowering Consumers through Sentence-level Readability Scoring in German ESG Reports 

**Authors**: Benjamin Josef Schüßler, Jakob Prange  

**Link**: [PDF](https://arxiv.org/pdf/2603.29861)  

**Abstract**: With the ever-growing urgency of sustainability in the economy and society, and the massive stream of information that comes with it, consumers need reliable access to that information. To address this need, companies began publishing so called Environmental, Social, and Governance (ESG) reports, both voluntarily and forced by law. To serve the public, these reports must be addressed not only to financial experts but also to non-expert audiences. But are they written clearly enough? In this work, we extend an existing sentence-level dataset of German ESG reports with crowdsourced readability annotations. We find that, in general, native speakers perceive sentences in ESG reports as easy to read, but also that readability is subjective. We apply various readability scoring methods and evaluate them regarding their prediction error and correlation with human rankings. Our analysis shows that, while LLM prompting has potential for distinguishing clear from hard-to-read sentences, a small finetuned transformer predicts human readability with the lowest error. Averaging predictions of multiple models can slightly improve the performance at the cost of slower inference. 

---
# GENIE: Gram-Eigenmode INR Editing with Closed-Form Geometry Updates 

**Authors**: Samundra Karki, Adarsh Krishnamurthy, Baskar Ganapathysubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2603.29860)  

**Abstract**: Implicit Neural Representations (INRs) provide compact models of geometry, but it is unclear when their learned shapes can be edited without retraining. We show that the Gram operator induced by the INR's penultimate features admits deformation eigenmodes that parameterize a family of realizable edits of the SDF zero level set. A key finding is that these modes are not intrinsic to the geometry alone: they are reliably recoverable only when the Gram operator is estimated from sufficiently rich sampling distributions. We derive a single closed-form update that performs geometric edits to the INR without optimization by leveraging the deformation modes. We characterize theoretically the precise set of deformations that are feasible under this one-shot update, and show that editing is well-posed exactly within the span of these deformation modes. 

---
# VectorGym: A Multitask Benchmark for SVG Code Generation, Sketching, and Editing 

**Authors**: Juan Rodriguez, Haotian Zhang, Abhay Puri, Tianyang Zhang, Rishav Pramanik, Meng Lin, Xiaoqing Xie, Marco Terral, Darsh Kaushik, Aly Shariff, Perouz Taslakian, Spandana Gella, Sai Rajeswar, David Vazquez, Christopher Pal, Marco Pedersoli  

**Link**: [PDF](https://arxiv.org/pdf/2603.29852)  

**Abstract**: We introduce VectorGym, a comprehensive benchmark suite for Scalable Vector Graphics (SVG) that spans generation from text and sketches, complex editing, and visual understanding. VectorGym addresses the lack of realistic, challenging benchmarks aligned with professional design workflows. Our benchmark comprises four tasks with expert human-authored annotations: the novel Sketch2SVG task (VG-Sketch); a new SVG editing dataset (VG-Edit) featuring complex, multi-step edits with higher-order primitives; Text2SVG generation (VG-Text); and SVG captioning (VG-Cap). Unlike prior benchmarks that rely on synthetic edits, VectorGym provides gold-standard human annotations that require semantic understanding and design intent. We also propose a multi-task reinforcement learning approach that jointly optimizes across all four tasks using rendering-based rewards. Our method, built on GRPO with curriculum learning, trains a Qwen3-VL 8B model that achieves state-of-the-art performance among open-source models, surpassing much larger models including Qwen3-VL 235B and matching GPT-4o. We also introduce a VLM-as-a-Judge metric for SVG generation, validated through human correlation studies. Our evaluation of frontier VLMs reveals significant performance gaps, positioning VectorGym as a rigorous framework for advancing visual code generation. VectorGym is publicly available on this http URL. 

---
# DIAL: Decoupling Intent and Action via Latent World Modeling for End-to-End VLA 

**Authors**: Yi Chen, Yuying Ge, Hui Zhou, Mingyu Ding, Yixiao Ge, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29844)  

**Abstract**: The development of Vision-Language-Action (VLA) models has been significantly accelerated by pre-trained Vision-Language Models (VLMs). However, most existing end-to-end VLAs treat the VLM primarily as a multimodal encoder, directly mapping vision-language features to low-level actions. This paradigm underutilizes the VLM's potential in high-level decision making and introduces training instability, frequently degrading its rich semantic representations. To address these limitations, we introduce DIAL, a framework bridging high-level decision making and low-level motor execution through a differentiable latent intent bottleneck. Specifically, a VLM-based System-2 performs latent world modeling by synthesizing latent visual foresight within the VLM's native feature space; this foresight explicitly encodes intent and serves as the structural bottleneck. A lightweight System-1 policy then decodes this predicted intent together with the current observation into precise robot actions via latent inverse dynamics. To ensure optimization stability, we employ a two-stage training paradigm: a decoupled warmup phase where System-2 learns to predict latent futures while System-1 learns motor control under ground-truth future guidance within a unified feature space, followed by seamless end-to-end joint optimization. This enables action-aware gradients to refine the VLM backbone in a controlled manner, preserving pre-trained knowledge. Extensive experiments on the RoboCasa GR1 Tabletop benchmark show that DIAL establishes a new state-of-the-art, achieving superior performance with 10x fewer demonstrations than prior methods. Furthermore, by leveraging heterogeneous human demonstrations, DIAL learns physically grounded manipulation priors and exhibits robust zero-shot generalization to unseen objects and novel configurations during real-world deployment on a humanoid robot. 

---
# From Density Matrices to Phase Transitions in Deep Learning: Spectral Early Warnings and Interpretability 

**Authors**: Max Hennick, Guillaume Corlouer  

**Link**: [PDF](https://arxiv.org/pdf/2603.29805)  

**Abstract**: A key problem in the modern study of AI is predicting and understanding emergent capabilities in models during training. Inspired by methods for studying reactions in quantum chemistry, we present the ``2-datapoint reduced density matrix". We show that this object provides a computationally efficient, unified observable of phase transitions during training. By tracking the eigenvalue statistics of the 2RDM over a sliding window, we derive two complementary signals: the spectral heat capacity, which we prove provides early warning of second-order phase transitions via critical slowing down, and the participation ratio, which reveals the dimensionality of the underlying reorganization. Remarkably, the top eigenvectors of the 2RDM are directly interpretable making it straightforward to study the nature of the transitions. We validate across four settings distinct settings: deep linear networks, induction head formation, grokking, and emergent misalignment. We then discuss directions for future work using the 2RDM. 

---
# From Skeletons to Semantics: Design and Deployment of a Hybrid Edge-Based Action Detection System for Public Safety 

**Authors**: Ganen Sethupathy, Lalit Dumka, Jan Schagen  

**Link**: [PDF](https://arxiv.org/pdf/2603.29777)  

**Abstract**: Public spaces such as transport hubs, city centres, and event venues require timely and reliable detection of potentially violent behaviour to support public safety. While automated video analysis has made significant progress, practical deployment remains constrained by latency, privacy, and resource limitations, particularly under edge-computing conditions. This paper presents the design and demonstrator-based deployment of a hybrid edge-based action detection system that combines skeleton-based motion analysis with vision-language models for semantic scene interpretation. Skeleton-based processing enables continuous, privacy-aware monitoring with low computational overhead, while vision-language models provide contextual understanding and zero-shot reasoning capabilities for complex and previously unseen situations. Rather than proposing new recognition models, the contribution focuses on a system-level comparison of both paradigms under realistic edge constraints. The system is implemented on a GPU-enabled edge device and evaluated with respect to latency, resource usage, and operational trade-offs using a demonstrator-based setup. The results highlight the complementary strengths and limitations of motioncentric and semantic approaches and motivate a hybrid architecture that selectively augments fast skeletonbased detection with higher-level semantic reasoning. The presented system provides a practical foundation for privacy-aware, real-time video analysis in public safety applications. 

---
# TSHA: A Benchmark for Visual Language Models in Trustworthy Safety Hazard Assessment Scenarios 

**Authors**: Qiucheng Yu, Ruijie Xu, Mingang Chen, Xuequan Lu, Jianfeng Dong, Chaochao Lu, Xin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2603.29759)  

**Abstract**: Recent advances in vision-language models (VLMs) have accelerated their application to indoor safety hazards assessment. However, existing benchmarks suffer from three fundamental limitations: (1) heavy reliance on synthetic datasets constructed via simulation software, creating a significant domain gap with real-world environments; (2) oversimplified safety tasks with artificial constraints on hazard and scene types, thereby limiting model generalization; and (3) absence of rigorous evaluation protocols to thoroughly assess model capabilities in complex home safety scenarios. To address these challenges, we introduce TSHA (\textbf{T}rustworthy \textbf{S}afety \textbf{H}azards \textbf{A}ssessment), a comprehensive benchmark comprising 81,809 carefully curated training samples drawn from four complementary sources: existing indoor datasets, internet images, AIGC images, and newly captured images. This benchmark set also includes a highly challenging test set with 1707 samples, comprising not only a carefully selected subset from the training distribution but also newly added videos and panoramic images containing multiple safety hazards, used to evaluate the model's robustness in complex safety scenarios. Extensive experiments on 23 popular VLMs demonstrate that current VLMs lack robust capabilities for safety hazard assessment. Importantly, models trained on the TSHA training set not only achieve a significant performance improvement of up to +18.3 points on the TSHA test set but also exhibit enhanced generalizability across other benchmarks, underscoring the substantial contribution and importance of the TSHA benchmark. 

---
# BotVerse: Real-Time Event-Driven Simulation of Social Agents 

**Authors**: Edoardo Allegrini, Edoardo Di Paolo, Angelo Spognardi, Marinella Petrocchi  

**Link**: [PDF](https://arxiv.org/pdf/2603.29741)  

**Abstract**: BotVerse is a scalable, event-driven framework for high-fidelity social simulation using LLM-based agents. It addresses the ethical risks of studying autonomous agents on live networks by isolating interactions within a controlled environment while grounding them in real-time content streams from the Bluesky ecosystem. The system features an asynchronous orchestration API and a simulation engine that emulates human-like temporal patterns and cognitive memory. Through the Synthetic Social Observatory, researchers can deploy customizable personas and observe multimodal interactions at scale. We demonstrate BotVersevia a coordinated disinformation scenario, providing a safe, experimental framework for red-teaming and computational social scientists. A video demonstration of the framework is available at this https URL. 

---
# Exploring the Impact of Skin Color on Skin Lesion Segmentation 

**Authors**: Kuniko Paxton, Medina Kapo, Amila Akagić, Koorosh Aslansefat, Dhavalkumar Thakker, Yiannis Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2603.29694)  

**Abstract**: Skin cancer, particularly melanoma, remains a major cause of morbidity and mortality, making early detection critical. AI-driven dermatology systems often rely on skin lesion segmentation as a preprocessing step to delineate the lesion from surrounding skin and support downstream analysis. While fairness concerns regarding skin tone have been widely studied for lesion classification, the influence of skin tone on the segmentation stage remains under-quantified and is frequently assessed using coarse, discrete skin tone categories. In this work, we evaluate three strong segmentation architectures (UNet, DeepLabV3 with a ResNet50 backbone, and DINOv2) on two public dermoscopic datasets (HAM10000 and ISIC2017) and introduce a continuous pigment or contrast analysis that treats pixel-wise ITA values as distributions. Using Wasserstein distances between within-image distributions for skin-only, lesion-only, and whole-image regions, we quantify lesion skin contrast and relate it to segmentation performance across multiple metrics. Within the range represented in these datasets, global skin tone metrics (Fitzpatrick grouping or mean ITA) show weak association with segmentation quality. In contrast, low lesion-skin contrast is consistently associated with larger segmentation errors in models, indicating that boundary ambiguity and low contrast are key drivers of failure. These findings suggest that fairness improvements in dermoscopic segmentation should prioritize robust handling of low-contrast lesions, and the distribution-based pigment measures provide a more informative audit signal than discrete skin-tone categories. 

---
# KEditVis: A Visual Analytics System for Knowledge Editing of Large Language Models 

**Authors**: Zhenning Chen, Hanbei Zhan, Yanwei Huang, Xin Wu, Dazhen Deng, Di Weng, Yingcai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29689)  

**Abstract**: Large Language Models (LLMs) demonstrate exceptional capabilities in factual question answering, yet they sometimes provide incorrect responses. To address this issue, knowledge editing techniques have emerged as effective methods for correcting factual information in LLMs. However, typical knowledge editing workflows struggle with identifying the optimal set of model layers for editing and rely on summary indicators that provide insufficient guidance. This lack of transparency hinders effective comparison and identification of optimal editing strategies. In this paper, we present KEditVis, a novel visual analytics system designed to assist users in gaining a deeper understanding of knowledge editing through interactive visualizations, improving editing outcomes, and discovering valuable insights for the future development of knowledge editing algorithms. With KEditVis, users can select appropriate layers as the editing target, explore the reasons behind ineffective edits, and perform more targeted and effective edits. Our evaluation, including usage scenarios, expert interviews, and a user study, validates the effectiveness and usability of the system. 

---
# Mind the Gap: A Framework for Assessing Pitfalls in Multimodal Active Learning 

**Authors**: Dustin Eisenhardt, Yunhee Jeong, Florian Buettner  

**Link**: [PDF](https://arxiv.org/pdf/2603.29677)  

**Abstract**: Multimodal learning enables neural networks to integrate information from heterogeneous sources, but active learning in this setting faces distinct challenges. These include missing modalities, differences in modality difficulty, and varying interaction structures. These are issues absent in the unimodal case. While the behavior of active learning strategies in unimodal settings is well characterized, their behavior under such multimodal conditions remains poorly understood. We introduce a new framework for benchmarking multimodal active learning that isolates these pitfalls using synthetic datasets, allowing systematic evaluation without confounding noise. Using this framework, we compare unimodal and multimodal query strategies and validate our findings on two real-world datasets. Our results show that models consistently develop imbalanced representations, relying primarily on one modality while neglecting others. Existing query methods do not mitigate this effect, and multimodal strategies do not consistently outperform unimodal ones. These findings highlight limitations of current active learning methods and underline the need for modality-aware query strategies that explicitly address these pitfalls. Code and benchmark resources will be made publicly available. 

---
# Agenda-based Narrative Extraction: Steering Pathfinding Algorithms with Large Language Models 

**Authors**: Brian Felipe Keith-Norambuena, Carolina Inés Rojas-Córdova, Claudio Juvenal Meneses-Villegas, Elizabeth Johanna Lam-Esquenazi, Angélica María Flores-Bustos, Ignacio Alejandro Molina-Villablanca, Joshua Emanuel Leyton-Vallejos  

**Link**: [PDF](https://arxiv.org/pdf/2603.29661)  

**Abstract**: Existing narrative extraction methods face a trade-off between coherence, interactivity, and multi-storyline support. Narrative Maps supports rich interaction and generates multiple storylines as a byproduct of its coverage constraints, though this comes at the cost of individual path coherence. Narrative Trails achieves high coherence through maximum capacity path optimization but provides no mechanism for user guidance or multiple perspectives. We introduce agenda-based narrative extraction, a method that bridges this gap by integrating large language models into the Narrative Trails pathfinding process to steer storyline construction toward user-specified perspectives. Our approach uses an LLM at each step to rank candidate documents based on their alignment with a given agenda while maintaining narrative coherence. Running the algorithm with different agendas yields different storylines through the same corpus. We evaluated our approach on a news article corpus using LLM judges with Claude Opus 4.5 and GPT 5.1, measuring both coherence and agenda alignment across 64 endpoint pairs and 6 agendas. LLM-driven steering achieves 9.9% higher alignment than keyword matching on semantic agendas (p=0.017), with 13.3% improvement on \textit{Regime Crackdown} specifically (p=0.037), while keyword matching remains competitive on agendas with literal keyword overlap. The coherence cost is minimal: LLM steering reduces coherence by only 2.2% compared to the agenda-agnostic baseline. Counter-agendas that contradict the source material score uniformly low (2.2-2.5) across all methods, confirming that steering cannot fabricate unsupported narratives. 

---
# 6GAgentGym: Tool Use, Data Synthesis, and Agentic Learning for Network Management 

**Authors**: Jiao Chen, Jianhua Tang, Xiaotong Yang, Zuohong Lv  

**Link**: [PDF](https://arxiv.org/pdf/2603.29656)  

**Abstract**: Autonomous 6G network management requires agents that can execute tools, observe the resulting state changes, and adapt their decisions accordingly. Existing benchmarks based on static questions or scripted episode replay, however, do not support such closed-loop interaction, limiting agents to passive evaluation without the ability to learn from environmental feedback. This paper presents 6GAgentGym to provide closed-loop capability. The framework provides an interactive environment with 42 typed tools whose effect classification distinguishes read-only observation from state-mutating configuration, backed by a learned Experiment Model calibrated on NS-3 simulation data. 6G-Forge bootstraps closed-loop training trajectories from NS-3 seeds via iterative Self-Instruct generation with execution verification against the Experiment Model. Supervised fine-tuning on the resulting corpus followed by reinforcement learning with online closed-loop interaction enables an 8B open-source model to achieve comparable overall success rate to GPT-5 on the accompanying 6GAgentBench, with stronger performance on long-horizon tasks. Together, these components provide a viable path toward autonomous, closed-loop network management. 

---
# Concept frustration: Aligning human concepts and machine representations 

**Authors**: Enrico Parisini, Christopher J. Soelistyo, Ahab Isaac, Alessandro Barp, Christopher R.S. Banerji  

**Link**: [PDF](https://arxiv.org/pdf/2603.29654)  

**Abstract**: Aligning human-interpretable concepts with the internal representations learned by modern machine learning systems remains a central challenge for interpretable AI. We introduce a geometric framework for comparing supervised human concepts with unsupervised intermediate representations extracted from foundation model embeddings. Motivated by the role of conceptual leaps in scientific discovery, we formalise the notion of concept frustration: a contradiction that arises when an unobserved concept induces relationships between known concepts that cannot be made consistent within an existing ontology. We develop task-aligned similarity measures that detect concept frustration between supervised concept-based models and unsupervised representations derived from foundation models, and show that the phenomenon is detectable in task-aligned geometry while conventional Euclidean comparisons fail. Under a linear-Gaussian generative model we derive a closed-form expression for Bayes-optimal concept-based classifier accuracy, decomposing predictive signal into known-known, known-unknown and unknown-unknown contributions and identifying analytically where frustration affects performance. Experiments on synthetic data and real language and vision tasks demonstrate that frustration can be detected in foundation model representations and that incorporating a frustrating concept into an interpretable model reorganises the geometry of learned concept representations, to better align human and machine reasoning. These results suggest a principled framework for diagnosing incomplete concept ontologies and aligning human and machine conceptual reasoning, with implications for the development and validation of safe interpretable AI for high-risk applications. 

---
# Semantic Interaction for Narrative Map Sensemaking: An Insight-based Evaluation 

**Authors**: Brian Felipe Keith-Norambuena, Fausto German, Eric Krokos, Sarah Joseph, Chris North  

**Link**: [PDF](https://arxiv.org/pdf/2603.29651)  

**Abstract**: Semantic interaction (SI) enables analysts to incorporate their cognitive processes into AI models through direct manipulation of visualizations. While SI frameworks for narrative extraction have been proposed, empirical evaluations of their effectiveness remain limited. This paper presents a user study that evaluates SI for narrative map sensemaking, involving 33 participants under three conditions: a timeline baseline, a basic narrative map, and an interactive narrative map with SI capabilities. The results show that the map-based prototypes yielded more insights than the timeline baseline, with the SI-enabled condition reaching statistical significance and the basic map condition trending in the same direction. The SI-enabled condition showed the highest mean performance; differences between the map conditions were not statistically significant but showed large effect sizes (d > 0.8), suggesting that the study was underpowered to detect them. Qualitative analysis identified two distinct SI approaches-corrective and additive-that enable analysts to impose quality judgments and organizational structure on extracted narratives. We also find that SI users achieved comparable exploration breadth with less parameter manipulation, suggesting that SI serves as an alternative pathway for model refinement. This work provides empirical evidence that map-based representations outperform timelines for narrative sensemaking, along with qualitative insights into how analysts use SI for narrative refinement. 

---
# MacTok: Robust Continuous Tokenization for Image Generation 

**Authors**: Hengyu Zeng, Xin Gao, Guanghao Li, Yuxiang Yan, Jiaoyang Ruan, Junpeng Ma, Haoyu Albert Wang, Jian Pu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29634)  

**Abstract**: Continuous image tokenizers enable efficient visual generation, and those based on variational frameworks can learn smooth, structured latent representations through KL regularization. Yet this often leads to posterior collapse when using fewer tokens, where the encoder fails to encode informative features into the compressed latent space. To address this, we introduce \textbf{MacTok}, a \textbf{M}asked \textbf{A}ugmenting 1D \textbf{C}ontinuous \textbf{Tok}enizer that leverages image masking and representation alignment to prevent collapse while learning compact and robust representations. MacTok applies both random masking to regularize latent learning and DINO-guided semantic masking to emphasize informative regions in images, forcing the model to encode robust semantics from incomplete visual evidence. Combined with global and local representation alignment, MacTok preserves rich discriminative information in a highly compressed 1D latent space, requiring only 64 or 128 tokens. On ImageNet, MacTok achieves a competitive gFID of 1.44 at 256$\times$256 and a state-of-the-art 1.52 at 512$\times$512 with SiT-XL, while reducing token usage by up to 64$\times$. These results confirm that masking and semantic guidance together prevent posterior collapse and achieve efficient, high-fidelity tokenization. 

---
# An Empirical Study of Multi-Agent Collaboration for Automated Research 

**Authors**: Yang Shen, Zhenyi Yi, Ziyi Zhao, Lijun Sun, Dongyang Li, Chin-Teng Lin, Yuhui Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.29632)  

**Abstract**: As AI agents evolve, the community is rapidly shifting from single Large Language Models (LLMs) to Multi-Agent Systems (MAS) to overcome cognitive bottlenecks in automated research. However, the optimal multi-agent coordination framework for these autonomous agents remains largely unexplored. In this paper, we present a systematic empirical study investigating the comparative efficacy of distinct multi-agent structures for automated machine learning optimization. Utilizing a rigorously controlled, execution-based testbed equipped with Git worktree isolation and explicit global memory, we benchmark a single-agent baseline against two multi-agent paradigms: a subagent architecture (parallel exploration with post-hoc consolidation) and an agent team architecture (experts with pre-execution handoffs). By evaluating these systems under strictly fixed computational time budgets, our findings reveal a fundamental trade-off between operational stability and theoretical deliberation. The subagent mode functions as a highly resilient, high-throughput search engine optimal for broad, shallow optimizations under strict time constraints. Conversely, the agent team topology exhibits higher operational fragility due to multi-author code generation but achieves the deep theoretical alignment necessary for complex architectural refactoring given extended compute budgets. These empirical insights provide actionable guidelines for designing future autoresearch systems, advocating for dynamically routed architectures that adapt their collaborative structures to real-time task complexity. 

---
# Convergent Representations of Linguistic Constructions in Human and Artificial Neural Systems 

**Authors**: Pegah Ramezani, Thomas Kinfe, Andreas Maier, Achim Schilling, Patrick Krauss  

**Link**: [PDF](https://arxiv.org/pdf/2603.29617)  

**Abstract**: Understanding how the brain processes linguistic constructions is a central challenge in cognitive neuroscience and linguistics. Recent computational studies show that artificial neural language models spontaneously develop differentiated representations of Argument Structure Constructions (ASCs), generating predictions about when and how construction-level information emerges during processing. The present study tests these predictions in human neural activity using electroencephalography (EEG). Ten native English speakers listened to 200 synthetically generated sentences across four construction types (transitive, ditransitive, caused-motion, resultative) while neural responses were recorded. Analyses using time-frequency methods, feature extraction, and machine learning classification revealed construction-specific neural signatures emerging primarily at sentence-final positions, where argument structure becomes fully disambiguated, and most prominently in the alpha band. Pairwise classification showed reliable differentiation, especially between ditransitive and resultative constructions, while other pairs overlapped. Crucially, the temporal emergence and similarity structure of these effects mirror patterns in recurrent and transformer-based language models, where constructional representations arise during integrative processing stages. These findings support the view that linguistic constructions are neurally encoded as distinct form-meaning mappings, in line with Construction Grammar, and suggest convergence between biological and artificial systems on similar representational solutions. More broadly, this convergence is consistent with the idea that learning systems discover stable regions within an underlying representational landscape - recently termed a Platonic representational space - that constrains the emergence of efficient linguistic abstractions. 

---
# IMAGAgent: Orchestrating Multi-Turn Image Editing via Constraint-Aware Planning and Reflection 

**Authors**: Fei Shen, Chengyu Xie, Lihong Wang, Zhanyi Zhang, Xin Jiang, Xiaoyu Du, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29602)  

**Abstract**: Existing multi-turn image editing paradigms are often confined to isolated single-step execution. Due to a lack of context-awareness and closed-loop feedback mechanisms, they are prone to error accumulation and semantic drift during multi-turn interactions, ultimately resulting in severe structural distortion of the generated images. For that, we propose \textbf{IMAGAgent}, a multi-turn image editing agent framework based on a "plan-execute-reflect" closed-loop mechanism that achieves deep synergy among instruction parsing, tool scheduling, and adaptive correction within a unified pipeline. Specifically, we first present a constraint-aware planning module that leverages a vision-language model (VLM) to precisely decompose complex natural language instructions into a series of executable sub-tasks, governed by target singularity, semantic atomicity, and visual perceptibility. Then, the tool-chain orchestration module dynamically constructs execution paths based on the current image, the current sub-task, and the historical context, enabling adaptive scheduling and collaborative operation among heterogeneous operation models covering image retrieval, segmentation, detection, and editing. Finally, we devise a multi-expert collaborative reflection mechanism where a central large language model (LLM) receives the image to be edited and synthesizes VLM critiques into holistic feedback, simultaneously triggering fine-grained self-correction and recording feedback outcomes to optimize future decisions. Extensive experiments on our constructed \textbf{MTEditBench} and the MagicBrush dataset demonstrate that IMAGAgent achieves performance significantly superior to existing methods in terms of instruction consistency, editing precision, and overall quality. The code is available at this https URL. 

---
# FigAgent: Towards Automatic Method Illustration Figure Generation for AI Scientific Papers 

**Authors**: Zhuoling Li, Jiarui Zhang, Jason Kuen, Jiuxiang Gu, Hossein Rahmani, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29590)  

**Abstract**: Method illustration figures (MIFs) play a crucial role in conveying the core ideas of scientific papers, yet their generation remains a labor-intensive process. In this paper, we identify three key characteristics that substantially influence MIF generation quality, i.e., \emph{compositional complexity}, \emph{component similarity}, and \emph{design dynamics}. To handle these characteristics, we take inspiration from human authors' drawing practices and propose \textbf{FigAgent}, a novel multi-agent framework for automatically generating high-quality MIFs. Through multi-agent collaboration, our FigAgent distills drawing experiences across similar components of MIFs and encapsulates them into reusable tools that can be invoked during MIF generation, while evolving these tools to adapt to dynamic design requirements. Besides, a novel Explore-and-Select drawing strategy is introduced to mimic the human-like trial-and-error manner for gradually constructing MIFs with complex structures. Extensive experiments show the efficacy of our method. Project is available \href{this https URL}{here}. 

---
# Learn2Fold: Structured Origami Generation with World Model Planning 

**Authors**: Yanjia Huang, Yunuo Chen, Ying Jiang, Jinru Han, Zhengzhong Tu, Yin Yang, Chenfanfu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29585)  

**Abstract**: The ability to transform a flat sheet into a complex three-dimensional structure is a fundamental test of physical intelligence. Unlike cloth manipulation, origami is governed by strict geometric axioms and hard kinematic constraints, where a single invalid crease or collision can invalidate the entire folding sequence. As a result, origami demands long-horizon constructive reasoning that jointly satisfies precise physical laws and high-level semantic intent. Existing approaches fall into two disjoint paradigms: optimization-based methods enforce physical validity but require dense, precisely specified inputs, making them unsuitable for sparse natural language descriptions, while generative foundation models excel at semantic and perceptual synthesis yet fail to produce long-horizon, physics-consistent folding processes. Consequently, generating valid origami folding sequences directly from text remains an open challenge. To address this gap, we introduce Learn2Fold, a neuro-symbolic framework that formulates origami folding as conditional program induction over a crease-pattern graph. Our key insight is to decouple semantic proposal from physical verification. A large language model generates candidate folding programs from abstract text prompts, while a learned graph-structured world model serves as a differentiable surrogate simulator that predicts physical feasibility and failure modes before execution. Integrated within a lookahead planning loop, Learn2Fold enables robust generation of physically valid folding sequences for complex and out-of-distribution patterns, demonstrating that effective spatial intelligence arises from the synergy between symbolic reasoning and grounded physical simulation. 

---
# Turbo4DGen: Ultra-Fast Acceleration for 4D Generation 

**Authors**: Yuanbin Man, Ying Huang, Zhile Ren, Miao Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.29572)  

**Abstract**: 4D generation, or dynamic 3D content generation, integrates spatial, temporal, and view dimensions to model realistic dynamic scenes, playing a foundational role in advancing world models and physical AI. However, maintaining long-chain consistency across both frames and viewpoints through the unique spatio-camera-motion (SCM) attention mechanism introduces substantial computational and memory overhead, often leading to out-of-memory (OOM) failures and prohibitive generation times. To address these challenges, we propose Turbo4DGen, an ultra-fast acceleration framework for diffusion-based multi-view 4D content generation. Turbo4DGen introduces a spatiotemporal cache mechanism that persistently reuses intermediate attention across denoising steps, combined with dynamically semantic-aware attention pruning and an adaptive SCM chain bypass scheduler, to drastically reduce redundant SCM attention computation. Our experimental results show that Turbo4DGen achieves an average 9.7$\times$ speedup without quality degradation on the ObjaverseDy and Consistent4D datasets. To the best of our knowledge, Turbo4DGen is the first dedicated acceleration framework for 4D generation. 

---
# Generating Key Postures of Bharatanatyam Adavus with Pose Estimation 

**Authors**: Jagadish Kashinath Kamble, Jayanta Mukhopadhyay, Debaditya Roy, Partha Pratim Das  

**Link**: [PDF](https://arxiv.org/pdf/2603.29570)  

**Abstract**: Preserving intangible cultural dances rooted in centuries of tradition and governed by strict structural and symbolic rules presents unique challenges in the digital era. Among these, Bharatanatyam, a classical Indian dance form, stands out for its emphasis on codified adavus and precise key postures. Accurately generating these postures is crucial not only for maintaining anatomical and stylistic integrity, but also for enabling effective documentation, analysis, and transmission to broader global audiences through digital means. We propose a pose-aware generative framework integrated with a pose estimation module, guided by keypoint-based loss and pose consistency constraints. These supervisory signals ensure anatomical accuracy and stylistic integrity in the synthesized outputs. We evaluate four configurations: standard conditional generative adversarial network (cGAN), cGAN with pose supervision, conditional diffusion, and conditional diffusion with pose supervision. Each model is conditioned on key posture class labels and optimized to maintain geometric structure. In both cGAN and conditional diffusion settings, the integrated pose guidance aligns generated poses with ground-truth keypoint structures, promoting cultural fidelity. Our results demonstrate that incorporating pose supervision significantly enhances the quality, realism, and authenticity of generated Bharatanatyam postures. This framework provides a scalable approach for the digital preservation, education, and dissemination of traditional dance forms, enabling high-fidelity generation without compromising cultural precision. Code is available at this https URL. 

---
# Bringing Up a Bilingual BabyLM: Investigating Multilingual Language Acquisition Using Small-Scale Models 

**Authors**: Linda Zeng, Steven Y. Feng, Michael C. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2603.29552)  

**Abstract**: Multilingualism is incredibly common around the world, leading to many important theoretical and practical questions about how children learn multiple languages at once. For example, does multilingual acquisition lead to delays in learning? Are there better and worse ways to structure multilingual input? Many correlational studies address these questions, but it is surprisingly difficult to get definitive answers because children cannot be randomly assigned to be multilingual and data are typically not matched between languages. We use language model training as a method for simulating a variety of highly controlled exposure conditions, and create matched 100M-word mono- and bilingual datasets using synthetic data and machine translation. We train GPT-2 models on monolingual and bilingual data organized to reflect a range of exposure regimes, and evaluate their performance on perplexity, grammaticality, and semantic knowledge. Across model scales and measures, bilingual models perform similarly to monolingual models in one language, but show strong performance in the second language as well. These results suggest that there are no strong differences between different bilingual exposure regimes, and that bilingual input poses no in-principle challenges for agnostic statistical learners. 

---
# Reducing Complexity for Quantum Approaches in Train Load Optimization 

**Authors**: Zhijie Tang, Albert Nieto-Morales, Arit Kumar Bishwas  

**Link**: [PDF](https://arxiv.org/pdf/2603.29543)  

**Abstract**: Efficiently planning container loads onto trains is a computationally challenging combinatorial optimization problem, central to logistics and supply chain management. A primary source of this complexity arises from the need to model and reduce rehandle operations-unproductive crane moves required to access blocked containers. Conventional mathematical formulations address this by introducing explicit binary variables and a web of logical constraints for each potential rehandle, resulting in large-scale models that are difficult to solve. This paper presents a fundamental departure from this paradigm. We introduce an innovative and compact mathematical formulation for the Train Load Optimization (TLO) problem where the rehandle cost is calculated implicitly within the objective function. This novel approach helps prevent the need for dedicated rehandle variables and their associated constraints, leading to a dramatic reduction in model size. We provide a formal comparison against a conventional model to analytically demonstrate the significant reduction in the number of variables and constraints. The efficacy of our compact formulation is assessed through a simulated annealing metaheuristic, which finds high-quality loading plans for various problem instances. The results confirm that our model is not only more parsimonious but also practically effective, offering a scalable and powerful tool for modern rail logistics. 

---
# Mean Masked Autoencoder with Flow-Mixing for Encrypted Traffic Classification 

**Authors**: Xiao Liu, Xiaowei Fu, Fuxiang Huang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29537)  

**Abstract**: Network traffic classification using self-supervised pre-training models based on Masked Autoencoders (MAE) has demonstrated a huge potential. However, existing methods are confined to isolated byte-level reconstruction of individual flows, lacking adequate perception of the multi-granularity contextual relationship in traffic. To address this limitation, we propose Mean MAE (MMAE), a teacher-student MAE paradigm with flow mixing strategy for building encrypted traffic pre-training model. MMAE employs a self-distillation mechanism for teacher-student interaction, where the teacher provides unmasked flow-level semantic supervision to advance the student from local byte reconstruction to multi-granularity comprehension. To break the information bottleneck in individual flows, we introduce a dynamic Flow Mixing (FlowMix) strategy to replace traditional random masking mechanism. By constructing challenging cross-flow mixed samples with interferences, it compels the model to learn discriminative representations from distorted tokens. Furthermore, we design a Packet-importance aware Mask Predictor (PMP) equipped with an attention bias mechanism that leverages packet-level side-channel statistics to dynamically mask tokens with high semantic density. Numerous experiments on a number of datasets covering encrypted applications, malware, and attack traffic demonstrate that MMAE achieves state-of-the-art performance. The code is available at this https URL 

---
# Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge 

**Authors**: Sowmya Vajrala, Aakash Parmar, Prasanna R, Sravanth Kodavanti, Manjunath Arveti, Srinivas Soumitri Miriyala, Ashok Senapati  

**Link**: [PDF](https://arxiv.org/pdf/2603.29535)  

**Abstract**: Generative Artificial Intelligence (GenAI) features such as image editing, object removal, and prompt-guided image transformation are increasingly integrated into mobile applications. However, deploying Large Vision Models (LVMs) for such tasks on resource-constrained devices remains challenging due to their high memory and compute requirements. While Low-Rank Adapters (LoRAs) enable parameter-efficient task adaptation, existing Mobile deployment pipelines typically compile separate model binaries for each LoRA + a copy of the foundation model, resulting in redundant storage and increased runtime overhead. In this work, we present a unified framework for enabling multi-task GenAI inference on edge devices using a single shared model. Our key idea is to treat LoRA weights as runtime inputs rather than embedding them into the compiled model graph, allowing dynamic task switching at runtime without recompilation. Then, to support efficient on-device execution, we introduce QUAD (Quantization with Unified Adaptive Distillation), a quantizationaware training strategy that aligns multiple LoRA adapters under a shared quantization profile. We implement the proposed system with a lightweight runtime stack compatible with mobile NPUs and evaluate it across multiple chipsets. Experimental results demonstrate up to 6x and 4x reduction in memory footprint and latency improvements, respectively, while maintaining high visual quality across multiple GenAI tasks. 

---
# Baby Scale: Investigating Models Trained on Individual Children's Language Input 

**Authors**: Steven Y. Feng, Alvin W.M. Tan, Michael C. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2603.29522)  

**Abstract**: Modern language models (LMs) must be trained on many orders of magnitude more words of training data than human children receive before they begin to produce useful behavior. Assessing the nature and origins of this "data gap" requires benchmarking LMs on human-scale datasets to understand how linguistic knowledge emerges from children's natural training data. Using transcripts from the BabyView dataset (videos from children ages 6-36 months), we investigate (1) scaling performance at child-scale data regimes, (2) variability in model performance across datasets from different children's experiences and linguistic predictors of dataset quality, and (3) relationships between model and child language learning outcomes. LMs trained on child data show acceptable scaling for grammar tasks, but lower scaling on semantic and world knowledge tasks than models trained on synthetic data; we also observe substantial variability on data from different children. Beyond dataset size, performance is most associated with a combination of distributional and interactional linguistic features, broadly consistent with what makes high-quality input for child language development. Finally, model likelihoods for individual words correlate with children's learning of those words, suggesting that properties of child-directed input may influence both model learning and human language development. Overall, understanding what properties make language data efficient for learning can enable more powerful small-scale language models while also shedding light on human language acquisition. 

---
# TrafficMoE: Heterogeneity-aware Mixture of Experts for Encrypted Traffic Classification 

**Authors**: Qing He, Xiaowei Fu, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29520)  

**Abstract**: Encrypted traffic classification is a critical task for network security. While deep learning has advanced this field, the occlusion of payload semantics by encryption severely challenges standard modeling approaches. Most existing frameworks rely on static and homogeneous pipelines that apply uniform parameter sharing and static fusion strategies across all inputs. This one-size-fits-all static design is inherently flawed: by forcing structured headers and randomized payloads into a unified processing pipeline, it inevitably entangles the raw protocol signals with stochastic encryption noise, thereby degrading the fine-grained discriminative features. In this paper, we propose TrafficMoE, a framework that breaks through the bottleneck of static modeling by establishing a Disentangle-Filter-Aggregate (DFA) paradigm. Specifically, to resolve the structural between-components conflict, the architecture disentangles headers and payloads using dual-branch sparse Mixture-of-Experts (MoE), enabling modality-specific modeling. To mitigate the impact of stochastic noise, an uncertainty-aware filtering mechanism is introduced to quantify reliability and selectively suppress high-variance representations. Finally, to overcome the limitations of static fusion, a routing-guided strategy aggregates cross-modality features dynamically, that adaptively weighs contributions based on traffic context. With this DFA paradigm, TrafficMoE maximizes representational efficiency by focusing solely on the most discriminative traffic features. Extensive experiments on six datasets demonstrate TrafficMoE consistently outperforms state-of-the-art methods, validating the necessity of heterogeneity-aware modeling in encrypted traffic analysis. The source code is publicly available at this https URL. 

---
# Impact of enriched meaning representations for language generation in dialogue tasks: A comprehensive exploration of the relevance of tasks, corpora and metrics 

**Authors**: Alain Vázquez, Maria Inés Torres  

**Link**: [PDF](https://arxiv.org/pdf/2603.29518)  

**Abstract**: Conversational systems should generate diverse language forms to interact fluently and accurately with users. In this context, Natural Language Generation (NLG) engines convert Meaning Representations (MRs) into sentences, directly influencing user perception. These MRs usually encode the communicative function (e.g., inform, request, confirm) via DAs and enumerate the semantic content with slot-value pairs. In this work, our objective is to analyse whether providing a task demonstrator to the generator enhances the generations of a fine-tuned model. This demonstrator is an MR-sentence pair extracted from the original dataset that enriches the input at training and inference time. The analysis involves five metrics that focus on different linguistic aspects, and four datasets that differ in multiple features, such as domain, size, lexicon, MR variability, and acquisition process. To the best of our knowledge, this is the first study on dialogue NLG implementing a comparative analysis of the impact of MRs on generation quality across domains, corpus characteristics, and the metrics used to evaluate these generations. Our key insight is that the proposed enriched inputs are effective for complex tasks and small datasets with high variability in MRs and sentences. They are also beneficial in zero-shot settings for any domain. Moreover, the analysis of the metrics shows that semantic metrics capture generation quality more accurately than lexical metrics. In addition, among these semantic metrics, those trained with human ratings can detect omissions and other subtle semantic issues that embedding-based metrics often miss. Finally, the evolution of the metric scores and the excellent results for Slot Accuracy and Dialogue Act Accuracy demonstrate that the generative models present fast adaptability to different tasks and robustness at semantic and communicative intention levels. 

---
# Target-Aligned Reinforcement Learning 

**Authors**: Leonard S. Pleiss, James Harrison, Maximilian Schiffer  

**Link**: [PDF](https://arxiv.org/pdf/2603.29501)  

**Abstract**: Many reinforcement learning algorithms rely on target networks - lagged copies of the online network - to stabilize training. While effective, this mechanism introduces a fundamental stability-recency tradeoff: slower target updates improve stability but reduce the recency of learning signals, hindering convergence speed. We propose Target-Aligned Reinforcement Learning (TARL), a framework that emphasizes transitions for which the target and online network estimates are highly aligned. By focusing updates on well-aligned targets, TARL mitigates the adverse effects of stale target estimates while retaining the stabilizing benefits of target networks. We provide a theoretical analysis demonstrating that target alignment correction accelerates convergence, and empirically demonstrate consistent improvements over standard reinforcement learning algorithms across various benchmark environments. 

---
# MemFactory: Unified Inference & Training Framework for Agent Memory 

**Authors**: Ziliang Guo, Ziheng Li, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.29493)  

**Abstract**: Memory-augmented Large Language Models (LLMs) are essential for developing capable, long-term AI agents. Recently, applying Reinforcement Learning (RL) to optimize memory operations, such as extraction, updating, and retrieval, has emerged as a highly promising research direction. However, existing implementations remain highly fragmented and task-specific, lacking a unified infrastructure to streamline the integration, training, and evaluation of these complex pipelines. To address this gap, we present MemFactory, the first unified, highly modular training and inference framework specifically designed for memory-augmented agents. Inspired by the success of unified fine-tuning frameworks like LLaMA-Factory, MemFactory abstracts the memory lifecycle into atomic, plug-and-play components, enabling researchers to seamlessly construct custom memory agents via a "Lego-like" architecture. Furthermore, the framework natively integrates Group Relative Policy Optimization (GRPO) to fine-tune internal memory management policies driven by multi-dimensional environmental rewards. MemFactory provides out-of-the-box support for recent cutting-edge paradigms, including Memory-R1, RMM, and MemAgent. We empirically validate MemFactory on the open-source MemAgent architecture using its publicly available training and evaluation data. Across both in-domain and out-of-distribution evaluation sets, MemFactory consistently improves performance over the corresponding base models, with relative gains of up to 14.8%. By providing a standardized, extensible, and easy-to-use infrastructure, MemFactory significantly lowers the barrier to entry, paving the way for future innovations in memory-driven AI agents. 

---
# iPoster: Content-Aware Layout Generation for Interactive Poster Design via Graph-Enhanced Diffusion Models 

**Authors**: Xudong Zhou, Jinyuan Liang, Qiuyi Guo, Guozheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.29469)  

**Abstract**: We present iPoster, an interactive layout generation framework that empowers users to guide content-aware poster layout design by specifying flexible constraints. iPoster enables users to specify partial intentions within the intention module, such as element categories, sizes, positions, or coarse initial drafts. Then, the generation module instantly generates refined, context-sensitive layouts that faithfully respect these constraints. iPoster employs a unified graph-enhanced diffusion architecture that supports various design tasks under user-specified constraints. These constraints are enforced through masking strategies that precisely preserve user input at every denoising step. A cross content-aware attention module aligns generated elements with salient regions of the canvas, ensuring visual coherence. Extensive experiments show that iPoster not only achieves state-of-the-art layout quality, but offers a responsive and controllable framework for poster layout design with constraints. 

---
# M-MiniGPT4: Multilingual VLLM Alignment via Translated Data 

**Authors**: Seung Hun Han, Youssef Mohamed, Mohamed Elhoseiny  

**Link**: [PDF](https://arxiv.org/pdf/2603.29467)  

**Abstract**: This paper presents a Multilingual Vision Large Language Model, named M-MiniGPT4. Our model exhibits strong vision-language understanding (VLU) capabilities across 11 languages. We utilize a mixture of native multilingual and translated data to push the multilingual VLU performance of the MiniGPT4 architecture. In addition, we propose a multilingual alignment training stage that uses parallel text corpora to further enhance the multilingual capabilities of our model. M-MiniGPT4 achieves 36% accuracy on the multilingual MMMU benchmark, outperforming state-of-the-art models in the same weight class, including foundation models released after the majority of this work was completed. We open-source our models, code, and translated datasets to facilitate future research in low-resource and multilingual settings. 

---
# An Isotropic Approach to Efficient Uncertainty Quantification with Gradient Norms 

**Authors**: Nils Grünefeld, Jes Frellsen, Christian Hardmeier  

**Link**: [PDF](https://arxiv.org/pdf/2603.29466)  

**Abstract**: Existing methods for quantifying predictive uncertainty in neural networks are either computationally intractable for large language models or require access to training data that is typically unavailable. We derive a lightweight alternative through two approximations: a first-order Taylor expansion that expresses uncertainty in terms of the gradient of the prediction and the parameter covariance, and an isotropy assumption on the parameter covariance. Together, these yield epistemic uncertainty as the squared gradient norm and aleatoric uncertainty as the Bernoulli variance of the point prediction, from a single forward-backward pass through an unmodified pretrained model. We justify the isotropy assumption by showing that covariance estimates built from non-training data introduce structured distortions that isotropic covariance avoids, and that theoretical results on the spectral properties of large networks support the approximation at scale. Validation against reference Markov Chain Monte Carlo estimates on synthetic problems shows strong correspondence that improves with model size. We then use the estimates to investigate when each uncertainty type carries useful signal for predicting answer correctness in question answering with large language models, revealing a benchmark-dependent divergence: the combined estimate achieves the highest mean AUROC on TruthfulQA, where questions involve genuine conflict between plausible answers, but falls to near chance on TriviaQA's factual recall, suggesting that parameter-level uncertainty captures a fundamentally different signal than self-assessment methods. 

---
# Few-shot Writer Adaptation via Multimodal In-Context Learning 

**Authors**: Tom Simon, Stephane Nicolas, Pierrick Tranouez, Clement Chatelain, Thierry Paquet  

**Link**: [PDF](https://arxiv.org/pdf/2603.29450)  

**Abstract**: While state-of-the-art Handwritten Text Recognition (HTR) models perform well on standard benchmarks, they frequently struggle with writers exhibiting highly specific styles that are underrepresented in the training data. To handle unseen and atypical writers, writer adaptation techniques personalize HTR models to individual handwriting styles. Leading writer adaptation methods require either offline fine-tuning or parameter updates at inference time, both involving gradient computation and backpropagation, which increase computational costs and demand careful hyperparameter tuning. In this work, we propose a novel context-driven HTR framework3 inspired by multimodal in-context learning, enabling inference-time writer adaptation using only a few examples from the target writer without any parameter updates. We further demonstrate the impact of context length, design a compact 8M-parameter CNN-Transformer that enables few-shot in-context adaptation, and show that combining context-driven and standard OCR training strategies leads to complementary improvements. Experiments on IAM and RIMES validate our approach with Character Error Rates of 3.92% and 2.34%, respectively, surpassing all writer-independent HTR models without requiring any parameter updates at inference time. 

---
# NeoNet: An End-to-End 3D MRI-Based Deep Learning Framework for Non-Invasive Prediction of Perineural Invasion via Generation-Driven Classification 

**Authors**: Youngung Han, Minkyung Cha, Kyeonghun Kim, Induk Um, Myeongbin Sho, Joo Young Bae, Jaewon Jung, Jung Hyeok Park, Seojun Lee, Nam-Joon Kim, Woo Kyoung Jeong, Won Jae Lee, Pa Hong, Ken Ying-Kai Liao, Hyuk-Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.29449)  

**Abstract**: Minimizing invasive diagnostic procedures to reduce the risk of patient injury and infection is a central goal in medical imaging. And yet, noninvasive diagnosis of perineural invasion (PNI), a critical prognostic factor involving infiltration of tumor cells along the surrounding nerve, still remains challenging, due to the lack of clear and consistent imaging criteria criteria for identifying PNI. To address this challenge, we present NeoNet, an integrated end-to-end 3D deep learning framework for PNI prediction in cholangiocarcinoma that does not rely on predefined image features. NeoNet integrates three modules: (1) NeoSeg, utilizing a Tumor-Localized ROI Crop (TLCR) algorithm; (2) NeoGen, a 3D Latent Diffusion Model (LDM) with ControlNet, conditioned on anatomical masks to generate synthetic image patches, specifically balancing the dataset to a 1:1 ratio; and (3) NeoCls, the final prediction module. For NeoCls, we developed the PNI-Attention Network (PattenNet), which uses the frozen LDM encoder and specialized 3D Dual Attention Blocks (DAB) designed to detect subtle intensity variations and spatial patterns indicative of PNI. In 5-fold cross-validation, NeoNet outperformed baseline 3D models and achieved the highest performance with a maximum AUC of 0.7903. 

---
# RAAP: Retrieval-Augmented Affordance Prediction with Cross-Image Action Alignment 

**Authors**: Qiyuan Zhuang, He-Yang Xu, Yijun Wang, Xin-Yang Zhao, Yang-Yang Li, Xiu-Shen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.29419)  

**Abstract**: Understanding object affordances is essential for enabling robots to perform purposeful and fine-grained interactions in diverse and unstructured environments. However, existing approaches either rely on retrieval, which is fragile due to sparsity and coverage gaps, or on large-scale models, which frequently mislocalize contact points and mispredict post-contact actions when applied to unseen categories, thereby hindering robust generalization. We introduce Retrieval-Augmented Affordance Prediction (RAAP), a framework that unifies affordance retrieval with alignment-based learning. By decoupling static contact localization and dynamic action direction, RAAP transfers contact points via dense correspondence and predicts action directions through a retrieval-augmented alignment model that consolidates multiple references with dual-weighted attention. Trained on compact subsets of DROID and HOI4D with as few as tens of samples per task, RAAP achieves consistent performance across unseen objects and categories, and enables zero-shot robotic manipulation in both simulation and the real world. Project website: this https URL. 

---
# Adversarial Prompt Injection Attack on Multimodal Large Language Models 

**Authors**: Meiwen Ding, Song Xia, Chenqi Kong, Xudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29418)  

**Abstract**: Although multimodal large language models (MLLMs) are increasingly deployed in real-world applications, their instruction-following behavior leaves them vulnerable to prompt injection attacks. Existing prompt injection methods predominantly rely on textual prompts or perceptible visual prompts that are observable by human users. In this work, we study imperceptible visual prompt injection against powerful closed-source MLLMs, where adversarial instructions are embedded in the visual modality. Our method adaptively embeds the malicious prompt into the input image via a bounded text overlay to provide semantic guidance. Meanwhile, the imperceptible visual perturbation is iteratively optimized to align the feature representation of the attacked image with those of the malicious visual and textual targets at both coarse- and fine-grained levels. Specifically, the visual target is instantiated as a text-rendered image and progressively refined during optimization to more faithfully represent the desired semantics and improve transferability. Extensive experiments on two multimodal understanding tasks across multiple closed-source MLLMs demonstrate the superior performance of our approach compared to existing methods. 

---
# AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models 

**Authors**: Yubo Cui, Xianchao Guan, Zijun Xiong, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29410)  

**Abstract**: Pre-trained vision-language models (VLMs) exhibit strong zero-shot generalization but remain vulnerable to adversarial perturbations. Existing classification-guided adversarial fine-tuning methods often disrupt pre-trained cross-modal alignment, weakening visual-textual correspondence and degrading zero-shot performance. In this paper, we propose an Alignment-Guided Fine-Tuning (AGFT) framework that enhances zero-shot adversarial robustness while preserving the cross-modal semantic structure. Unlike label-based methods that rely on hard labels and fail to maintain the relative relationships between image and text, AGFT leverages the probabilistic predictions of the original model for text-guided adversarial training, which aligns adversarial visual features with textual embeddings via soft alignment distributions, improving zero-shot adversarial robustness. To address structural discrepancies introduced by fine-tuning, we introduce a distribution consistency calibration mechanism that adjusts the robust model output to match a temperature-scaled version of the pre-trained model predictions. Extensive experiments across multiple zero-shot benchmarks demonstrate that AGFT outperforms state-of-the-art methods while significantly improving zero-shot adversarial robustness. 

---
# Hybrid Quantum-Classical Spatiotemporal Forecasting for 3D Cloud Fields 

**Authors**: Fu Wang, Qifeng Lu, Xinyu Long, Meng Zhang, Xiaofei Yang, Weijia Cao, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29407)  

**Abstract**: Accurate forecasting of three-dimensional (3D) cloud fields is important for atmospheric analysis and short-range numerical weather prediction, yet it remains challenging because cloud evolution involves cross-layer interactions, nonlocal dependencies, and multiscale spatiotemporal dynamics. Existing spatiotemporal prediction models based on convolutions, recurrence, or attention often rely on locality-biased representations and therefore struggle to preserve fine cloud structures in volumetric forecasting tasks. To address this issue, we propose QENO, a hybrid quantum-inspired spatiotemporal forecasting framework for 3D cloud fields. The proposed architecture consists of four components: a classical spatiotemporal encoder for compact latent representation, a topology-aware quantum enhancement block for modeling nonlocal couplings in latent space, a dynamic fusion temporal unit for integrating measurement-derived quantum features with recurrent memory, and a decoder for reconstructing future cloud volumes. Experiments on CMA-MESO 3D cloud fields show that QENO consistently outperforms representative baselines, including ConvLSTM, PredRNN++, Earthformer, TAU, and SimVP variants, in terms of MSE, MAE, RMSE, SSIM, and threshold-based detection metrics. In particular, QENO achieves an MSE of 0.2038, an RMSE of 0.4514, and an SSIM of 0.6291, while also maintaining a compact parameter budget. These results indicate that topology-aware hybrid quantum-classical feature modeling is a promising direction for 3D cloud structure forecasting and atmospheric Earth observation data analysis. 

---
# Hallucination-aware intermediate representation edit in large vision-language models 

**Authors**: Wei Suo, Hanzu Zhang, Lijun Zhang, Ji Ma, Peng Wang, Yanning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29405)  

**Abstract**: Large Vision-Language Models have demonstrated exceptional performance in multimodal reasoning and complex scene understanding. However, these models still face significant hallucination issues, where outputs contradict visual facts. Recent research on hallucination mitigation has focused on retraining methods and Contrastive Decoding (CD) methods. While both methods perform well, retraining methods require substantial training resources, and CD methods introduce dual inference overhead. These factors hinder their practical applicability. To address the above issue, we propose a framework for dynamically detecting hallucination representations and performing hallucination-eliminating edits on these representations. With minimal additional computational cost, we achieve state-of-the-art performance on existing benchmarks. Extensive experiments demonstrate the effectiveness of our approach, highlighting its efficient and robust hallucination elimination capability and its powerful controllability over hallucinations. Code is available at this https URL 

---
# Security in LLM-as-a-Judge: A Comprehensive SoK 

**Authors**: Aiman Almasoud, Antony Anju, Marco Arazzi, Mert Cihangiroglu, Vignesh Kumar Kembu, Serena Nicolazzo, Antonino Nocera, Vinod P., Saraga Sakthidharan  

**Link**: [PDF](https://arxiv.org/pdf/2603.29403)  

**Abstract**: LLM-as-a-Judge (LaaJ) is a novel paradigm in which powerful language models are used to assess the quality, safety, or correctness of generated outputs. While this paradigm has significantly improved the scalability and efficiency of evaluation processes, it also introduces novel security risks and reliability concerns that remain largely unexplored. In particular, LLM-based judges can become both targets of adversarial manipulation and instruments through which attacks are conducted, potentially compromising the trustworthiness of evaluation pipelines. In this paper, we present the first Systematization of Knowledge (SoK) focusing on the security aspects of LLM-as-a-Judge systems. We perform a comprehensive literature review across major academic databases, analyzing 863 works and selecting 45 relevant studies published between 2020 and 2026. Based on this study, we propose a taxonomy that organizes recent research according to the role played by LLM-as-a-Judge in the security landscape, distinguishing between attacks targeting LaaJ systems, attacks performed through LaaJ, defenses leveraging LaaJ for security purposes, and applications where LaaJ is used as an evaluation strategy in security-related domains. We further provide a comparative analysis of existing approaches, highlighting current limitations, emerging threats, and open research challenges. Our findings reveal significant vulnerabilities in LLM-based evaluation frameworks, as well as promising directions for improving their robustness and reliability. Finally, we outline key research opportunities that can guide the development of more secure and trustworthy LLM-as-a-Judge systems. 

---
# Extend3D: Town-Scale 3D Generation 

**Authors**: Seungwoo Yoon, Jinmo Kim, Jaesik Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.29387)  

**Abstract**: In this paper, we propose Extend3D, a training-free pipeline for 3D scene generation from a single image, built upon an object-centric 3D generative model. To overcome the limitations of fixed-size latent spaces in object-centric models for representing wide scenes, we extend the latent space in the $x$ and $y$ directions. Then, by dividing the extended latent space into overlapping patches, we apply the object-centric 3D generative model to each patch and couple them at each time step. Since patch-wise 3D generation with image conditioning requires strict spatial alignment between image and latent patches, we initialize the scene using a point cloud prior from a monocular depth estimator and iteratively refine occluded regions through SDEdit. We discovered that treating the incompleteness of 3D structure as noise during 3D refinement enables 3D completion via a concept, which we term under-noising. Furthermore, to address the sub-optimality of object-centric models for sub-scene generation, we optimize the extended latent during denoising, ensuring that the denoising trajectories remain consistent with the sub-scene dynamics. To this end, we introduce 3D-aware optimization objectives for improved geometric structure and texture fidelity. We demonstrate that our method yields better results than prior methods, as evidenced by human preference and quantitative experiments. 

---
# PromptForge-350k: A Large-Scale Dataset and Contrastive Framework for Prompt-Based AI Image Forgery Localization 

**Authors**: Jianpeng Wang, Haoyu Wang, Baoying Chen, Jishen Zeng, Yiming Qin, Yiqi Yang, Zhongjie Ba  

**Link**: [PDF](https://arxiv.org/pdf/2603.29386)  

**Abstract**: The rapid democratization of prompt-based AI image editing has recently exacerbated the risks associated with malicious content fabrication and misinformation. However, forgery localization methods targeting these emerging editing techniques remain significantly under-explored. To bridge this gap, we first introduce a fully automated mask annotating framework that leverages keypoint alignment and semantic space similarity to generate precise ground-truth masks for edited regions. Based on this framework, we construct PromptForge-350k, a large-scale forgery localization dataset covering four state-of-the-art prompt-based AI image editing models, thereby mitigating the data scarcity in this domain. Furthermore, we propose ICL-Net, an effective forgery localization network featuring a triple-stream backbone and intra-image contrastive learning. This design enables the model to capture highly robust and generalizable forensic features. Extensive experiments demonstrate that our method achieves an IoU of 62.5% on PromptForge-350k, outperforming SOTA methods by 5.1%. Additionally, it exhibits strong robustness against common degradations with an IoU drop of less than 1%, and shows promising generalization capabilities on unseen editing models, achieving an average IoU of 41.5%. 

---
# Deep Learning-Based Anomaly Detection in Spacecraft Telemetry on Edge Devices 

**Authors**: Christopher Goetze, Tim Schlippe, Daniel Lakey  

**Link**: [PDF](https://arxiv.org/pdf/2603.29375)  

**Abstract**: Spacecraft anomaly detection is critical for mission safety, yet deploying sophisticated models on-board presents significant challenges due to hardware constraints. This paper investigates three approaches for spacecraft telemetry anomaly detection -- forecasting & threshold, direct classification, and image classification -- and optimizes them for edge deployment using multi-objective neural architecture optimization on the European Space Agency Anomaly Dataset. Our baseline experiments demonstrate that forecasting & threshold achieves superior detection performance (92.7% Corrected Event-wise F0.5-score (CEF0.5)) [1] compared to alternatives. Through Pareto-optimal architecture optimization, we dramatically reduced computational requirements while maintaining capabilities -- the optimized forecasting & threshold model preserved 88.8% CEF0.5 while reducing RAM usage by 97.1% to just 59 KB and operations by 99.4%. Analysis of deployment viability shows our optimized models require just 0.36-6.25% of CubeSat RAM, making on-board anomaly detection practical even on highly constrained hardware. This research demonstrates that sophisticated anomaly detection capabilities can be successfully deployed within spacecraft edge computing constraints, providing near-instantaneous detection without exceeding hardware limitations or compromising mission safety. 

---
# CIPHER: Counterfeit Image Pattern High-level Examination via Representation 

**Authors**: Kyeonghun Kim, Youngung Han, Seoyoung Ju, Yeonju Jean, YooHyun Kim, Minseo Choi, SuYeon Lim, Kyungtae Park, Seungwoo Baek, Sieun Hyeon, Nam-Joon Kim, Hyuk-Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.29356)  

**Abstract**: The rapid progress of generative adversarial networks (GANs) and diffusion models has enabled the creation of synthetic faces that are increasingly difficult to distinguish from real images. This progress, however, has also amplified the risks of misinformation, fraud, and identity abuse, underscoring the urgent need for detectors that remain robust across diverse generative models. In this work, we introduce Counterfeit Image Pattern High-level Examination via Representation(CIPHER), a deepfake detection framework that systematically reuses and fine-tunes discriminators originally trained for image generation. By extracting scale-adaptive features from ProGAN discriminators and temporal-consistency features from diffusion models, CIPHER captures generation-agnostic artifacts that conventional detectors often overlook. Through extensive experiments across nine state-of-the-art generative models, CIPHER demonstrates superior cross-model detection performance, achieving up to 74.33% F1-score and outperforming existing ViT-based detectors by over 30% in F1-score on average. Notably, our approach maintains robust performance on challenging datasets where baseline methods fail, with up to 88% F1-score on CIFAKE compared to near-zero performance from conventional detectors. These results validate the effectiveness of discriminator reuse and cross-model fine-tuning, establishing CIPHER as a promising approach toward building more generalizable and robust deepfake detection systems in an era of rapidly evolving generative technologies. 

---
# Scaling Whole-Body Human Musculoskeletal Behavior Emulation for Specificity and Diversity 

**Authors**: Yunyue Wei, Chenhui Zuo, Shanning Zhuang, Haixin Gong, Yaming Liu, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2603.29332)  

**Abstract**: The embodied learning of human motor control requires whole-body neuro-actuated musculoskeletal dynamics, while the internal muscle-driven processes underlying movement remain inaccessible to direct measurement. Computational modeling offers an alternative, but inverse dynamics methods struggled to resolve redundant control from observed kinematics in the high-dimensional, over-actuated system. Forward imitation approaches based on deep reinforcement learning exhibited inadequate tracking performance due to the curse of dimensionality in both control and reward design. Here we introduce a large-scale parallel musculoskeletal computation framework for biomechanically grounded whole-body motion reproduction. By integrating large-scale parallel GPU simulation with adversarial reward aggregation and value-guided flow exploration, the MS-Emulator framework overcomes key optimization bottlenecks in high-dimensional reinforcement learning for musculoskeletal control, which accurately reproduces a broad repertoire of motions in a whole-body human musculoskeletal system actuated by approximately 700 muscles. It achieved high joint angle accuracy and body position alignment for highly dynamic tasks such as dance, cartwheel, and backflip. The framework was also used to explore the musculoskeletal control solution space, identifying distinct musculoskeletal control policies that converge to nearly identical external kinematic and mechanical measurements. This work establishes a tractable computational route to analyzing the specificity and diversity underlying human embodied control of movement. Project page: this https URL. 

---
# Beyond Corner Patches: Semantics-Aware Backdoor Attack in Federated Learning 

**Authors**: Kavindu Herath, Joshua Zhao, Saurabh Bagchi  

**Link**: [PDF](https://arxiv.org/pdf/2603.29328)  

**Abstract**: Backdoor attacks on federated learning (FL) are most often evaluated with synthetic corner patches or out-of-distribution (OOD) patterns that are unlikely to arise in practice. In this paper, we revisit the backdoor threat to standard FL (a single global model) under a more realistic setting where triggers must be semantically meaningful, in-distribution, and visually plausible. We propose SABLE, a Semantics-Aware Backdoor for LEarning in federated settings, which constructs natural, content-consistent triggers (e.g., semantic attribute changes such as sunglasses) and optimizes an aggregation-aware malicious objective with feature separation and parameter regularization to keep attacker updates close to benign ones. We instantiate SABLE on CelebA hair-color classification and the German Traffic Sign Recognition Benchmark (GTSRB), poisoning only a small, interpretable subset of each malicious client's local data while otherwise following the standard FL protocol. Across heterogeneous client partitions and multiple aggregation rules (FedAvg, Trimmed Mean, MultiKrum, and FLAME), our semantics-driven triggers achieve high targeted attack success rates while preserving benign test accuracy. These results show that semantics-aligned backdoors remain a potent and practical threat in federated learning, and that robustness claims based solely on synthetic patch triggers can be overly optimistic. 

---
# Real-Time Band-Grouped Vocal Denoising Using Sigmoid-Driven Ideal Ratio Masking 

**Authors**: Daniel Williams  

**Link**: [PDF](https://arxiv.org/pdf/2603.29326)  

**Abstract**: Real-time, deep learning-based vocal denoising has seen significant progress over the past few years, demonstrating the capability of artificial intelligence in preserving the naturalness of the voice while increasing the signal-to-noise ratio (SNR). However, many deep learning approaches have high amounts of latency and require long frames of context, making them difficult to configure for live applications. To address these challenges, we propose a sigmoid-driven ideal ratio mask trained with a spectral loss to encourage an increased SNR and maximized perceptual quality of the voice. The proposed model uses a band-grouped encoder-decoder architecture with frequency attention and achieves a total latency of less than 10,ms, with PESQ-WB improvements of 0.21 on stationary noise and 0.12 on nonstationary noise. 

---
# IMPASTO: Integrating Model-Based Planning with Learned Dynamics Models for Robotic Oil Painting Reproduction 

**Authors**: Yingke Wang, Hao Li, Yifeng Zhu, Hong-Xing Yu, Ken Goldberg, Li Fei-Fei, Jiajun Wu, Yunzhu Li, Ruohan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29315)  

**Abstract**: Robotic reproduction of oil paintings using soft brushes and pigments requires force-sensitive control of deformable tools, prediction of brushstroke effects, and multi-step stroke planning, often without human step-by-step demonstrations or faithful simulators. Given only a sequence of target oil painting images, can a robot infer and execute the stroke trajectories, forces, and colors needed to reproduce it? We present IMPASTO, a robotic oil-painting system that integrates learned pixel dynamics models with model-based planning. The dynamics models predict canvas updates from image observations and parameterized stroke actions; a receding-horizon model predictive control optimizer then plans trajectories and forces, while a force-sensitive controller executes strokes on a 7-DoF robot arm. IMPASTO integrates low-level force control, learned dynamics models, and high-level closed-loop planning, learns solely from robot self-play, and approximates human artists' single-stroke datasets and multi-stroke artworks, outperforming baselines in reproduction accuracy. Project website: this https URL 

---
# Self-Improving Code Generation via Semantic Entropy and Behavioral Consensus 

**Authors**: Huan Zhang, Wei Cheng, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29292)  

**Abstract**: Improving the code generation capabilities of large language models (LLMs) typically relies on supervised fine-tuning or preference optimization, both of which require costly external resources such as powerful teacher models or reliable test units. However, in real-world scenarios, it is much harder to obtain reference solutions and test oracles than problem descriptions and test inputs. In this paper, we tackle a challenging yet realistic question: Can a code language model improve itself without access to a superior teacher and a test oracle? To answer this, we propose ConSelf, a self-improving approach built upon two key ideas. First, we introduce code semantic entropy, a novel metric that measures problem-level uncertainty by assessing the functional diversity of program behaviors, enabling a curriculum construction with the most learnable problems. Second, we present consensus-driven direct preference optimization (Con-DPO), a preference-based fine-tuning method that weights each preference pair by its behavioral consensus, thereby mitigating the impact of noisy self-generated supervision. Experiments on various benchmarks and backbone LLMs demonstrate that ConSelf significantly outperforms baselines, validating the effectiveness of semantic entropy-based curriculum construction and consensus-driven optimization in improving code generation without external supervision. 

---
# MELT: Improve Composed Image Retrieval via the Modification Frequentation-Rarity Balance Network 

**Authors**: Guozhi Qiu, Zhiwei Chen, Zixu Li, Qinlei Huang, Zhiheng Fu, Xuemeng Song, Yupeng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29291)  

**Abstract**: Composed Image Retrieval (CIR) uses a reference image and a modification text as a query to retrieve a target image satisfying the requirement of ``modifying the reference image according to the text instructions''. However, existing CIR methods face two limitations: (1) frequency bias leading to ``Rare Sample Neglect'', and (2) susceptibility of similarity scores to interference from hard negative samples and noise. To address these limitations, we confront two key challenges: asymmetric rare semantic localization and robust similarity estimation under hard negative samples. To solve these challenges, we propose the Modification frEquentation-rarity baLance neTwork MELT. MELT assigns increased attention to rare modification semantics in multimodal contexts while applying diffusion-based denoising to hard negative samples with high similarity scores, enhancing multimodal fusion and matching. Extensive experiments on two CIR benchmarks validate the superior performance of MELT. Codes are available at this https URL. 

---
# Downsides of Smartness Across Edge-Cloud Continuum in Modern Industry 

**Authors**: Akhil Gupta Chigullapally, Sharvan Vittala, Razin Farhan Hussian, Mohsen Amini Salehi  

**Link**: [PDF](https://arxiv.org/pdf/2603.29289)  

**Abstract**: The fast pace of modern AI is rapidly transforming traditional industrial systems into vast, intelligent and potentially unmanned autonomous operational environments driven by AI-based solutions. These solutions leverage various forms of machine learning, reinforcement learning, and generative AI. The introduction of such smart capabilities has pushed the envelope in multiple industrial domains, enabling predictive maintenance, optimized performance, and streamlined workflows. These solutions are often deployed across the Industrial Internet of Things (IIoT) and supported by the Edge-Fog-Cloud computing continuum to enable urgent (i.e., real-time or near real-time) decision-making. Despite the current trend of aggressively adopting these smart industrial solutions to increase profit, quality, and efficiency, large-scale integration and deployment also bring serious hazards that if ignored can undermine the benefits of smart industries. These hazards include unforeseen interoperability side-effects and heightened vulnerability to cyber threats, particularly in environments operating with a plethora of heterogeneous IIoT systems. The goal of this study is to shed light on the potential consequences of industrial smartness, with a particular focus on security implications, including vulnerabilities, side effects, and cyber threats. We distinguish software-level downsides stemming from both traditional AI solutions and generative AI from those originating in the infrastructure layer, namely IIoT and the Edge-Cloud continuum. At each level, we investigate potential vulnerabilities, cyber threats, and unintended side effects. As industries continue to become smarter, understanding and addressing these downsides will be crucial to ensure secure and sustainable development of smart industrial systems. 

---
# Sima AIunty: Caste Audit in LLM-Driven Matchmaking 

**Authors**: Atharva Naik, Shounok Kar, Varnika Sharma, Ashwin Rajadesingan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2603.29288)  

**Abstract**: Social and personal decisions in relational domains such as matchmaking are deeply entwined with cultural norms and historical hierarchies, and can potentially be shaped by algorithmic and AI-mediated assessments of compatibility, acceptance, and stability. In South Asian contexts, caste remains a central aspect of marital decision-making, yet little is known about how contemporary large language models (LLMs) reproduce or disrupt caste-based stratification in such settings. In this work, we conduct a controlled audit of caste bias in LLM-mediated matchmaking evaluations using real-world matrimonial profiles. We vary caste identity across Brahmin, Kshatriya, Vaishya, Shudra, and Dalit, and income across five buckets, and evaluate five LLM families (GPT, Gemini, Llama, Qwen, and BharatGPT). Models are prompted to assess profiles along dimensions of social acceptance, marital stability, and cultural compatibility. Our analysis reveals consistent hierarchical patterns across models: same-caste matches are rated most favorably, with average ratings up to 25% higher (on a 10-point scale) than inter-caste matches, which are further ordered according to traditional caste hierarchy. These findings highlight how existing caste hierarchies are reproduced in LLM decision-making and underscore the need for culturally grounded evaluation and intervention strategies in AI systems deployed in socially sensitive domains, where such systems risk reinforcing historical forms of exclusion. 

---
# PRISM: A Multi-View Multi-Capability Retail Video Dataset for Embodied Vision-Language Models 

**Authors**: Amirreza Rouhi, Parikshit Sakurikar, Satya Sai Reddy, Narsimha Menga, Anirudh Govil, Sri Harsha Chittajallu, Rajat Aggarwal, Anoop Namboodiri, Sashi Reddi  

**Link**: [PDF](https://arxiv.org/pdf/2603.29281)  

**Abstract**: A critical gap exists between the general-purpose visual understanding of state-of-the-art physical AI models and the specialized perceptual demands of structured real-world deployment environments. We present PRISM, a 270K-sample multi-view video supervised fine-tuning (SFT) corpus for embodied vision-language-models (VLMs) in real-world retail environments. PRISM is motivated by a simple observation - physical AI systems fail not because of poor visual recognition, but because they do not understand space, physical dynamics and embodied action well enough to operate reliably in the world. To this end, PRISM is grounded in a novel three-dimensional knowledge ontology that spans spatial knowledge, temporal and physical knowledge, and embodied action knowledge. It covers 20+ capability probes across four evaluation dimensions - Embodied Reasoning (ER), Common Sense (CS), Spatial Perception (SP), and Intuitive Physics (IP), and to our knowledge, PRISM is the first dataset to instantiate all three knowledge dimensions within a single real-world deployment domain. The corpus captures data from egocentric, exocentric and 360° viewpoints across five supermarket locations and includes open-ended, chain-of-thought, and multiple-choice supervision. At 4 fps, PRISM spans approximately 11.8M video frames and approximately 730M tokens, placing it among the largest domain-specific video SFT corpora. Fine-tuning on PRISM reduces the error rate across all 20+ probes by 66.6% over the pre-trained baseline, with significant gains in embodied action understanding where the accuracy improves by 36.4%. Our results suggest that ontology-structured, domain specific SFT can meaningfully strengthen embodied VLMs for real-world settings. The PRISM dataset and more details are available at this https URL 

---
# Monodense Deep Neural Model for Determining Item Price Elasticity 

**Authors**: Lakshya Garg, Sai Yaswanth, Deep Narayan Mishra, Karthik Kumaran, Anupriya Sharma, Mayank Uniyal  

**Link**: [PDF](https://arxiv.org/pdf/2603.29261)  

**Abstract**: Item Price Elasticity is used to quantify the responsiveness of consumer demand to changes in item prices, enabling businesses to create pricing strategies and optimize revenue management. Sectors such as store retail, e-commerce, and consumer goods rely on elasticity information derived from historical sales and pricing data. This elasticity provides an understanding of purchasing behavior across different items, consumer discount sensitivity, and demand elastic departments. This information is particularly valuable for competitive markets and resource-constrained businesses decision making which aims to maximize profitability and market share. Price elasticity also uncovers historical shifts in consumer responsiveness over time. In this paper, we model item-level price elasticity using large-scale transactional datasets, by proposing a novel elasticity estimation framework which has the capability to work in an absence of treatment control setting. We test this framework by using Machine learning based algorithms listed below, including our newly proposed Monodense deep neural network.
(1) Monodense-DL network -- Hybrid neural network architecture combining embedding, dense, and Monodense layers (2) DML -- Double machine learning setting using regression models (3) LGBM -- Light Gradient Boosting Model
We evaluate our model on multi-category retail data spanning millions of transactions using a back testing framework. Experimental results demonstrate the superiority of our proposed neural network model within the framework compared to other prevalent ML based methods listed above. 

---
# Omni-NegCLIP: Enhancing CLIP with Front-Layer Contrastive Fine-Tuning for Comprehensive Negation Understanding 

**Authors**: Jingqi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29258)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated strong capabilities across a wide range of multimodal tasks. However, recent studies have shown that VLMs, such as CLIP, perform poorly in understanding negation expressions, which are common in natural language. In this work, we propose Omni-NegCLIP, a fine-tuned CLIP model that improves CLIP's understanding of two types of negation, namely presence-based negation and absence-based negation, which correspond to negated expressions of objects that are actually present in an image and those that may plausibly exist in an image but are in fact absent, respectively, by modifying CLIP's original InfoNCE contrastive loss. Specifically, we design a presence-based contrastive objective that pulls image embeddings closer to their original caption embeddings while pushing them away from the corresponding presence-based negated caption embeddings, and an absence-based contrastive objective that aligns image embeddings with both original and absence-based negated caption embeddings while maintaining a semantic distinction between the two text embeddings. Based on our observation that the front transformer layers of CLIP text encoder have stronger learning ability for negated text than the later layers, we fine-tune the front transformer layers of the CLIP text encoder at each training step using the combined contrastive objective. Experimental results show that, compared with pretrained CLIP, Omni-NegCLIP improves performance on presence-based negation and absence-based negation tasks by up to 52.65% and 12.50%, respectively, without sacrificing general capability in image-text retrieval and even improving it by up to 19.62%. Compared with prior works, Omni-NegCLIP demonstrates a more comprehensive ability to understand multiple types of negation tasks. 

---
# Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism 

**Authors**: Tao Chen, Kun Zhang, Qiong Wu, Xiao Chen, Chao Chang, Xiaoshuai Sun, Yiyi Zhou, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2603.29252)  

**Abstract**: Long video understanding is a key challenge that plagues the advancement of \emph{Multimodal Large language Models} (MLLMs). In this paper, we study this problem from the perspective of visual memory mechanism, and proposed a novel and training-free approach, termed \emph{Flexible Memory} (\textbf{FlexMem}). In principle, FlexMem aims to mimic human behavior of video watching, \emph{i.e.}, continually watching video content and recalling the most relevant memory fragments to answer the question. In this way, FlexMem can help MLLMs achieve video understanding of infinite lengths, unlike previous methods that process all video information at once and have input upper-limit. Concretely, FlexMem first consider the visual KV caches as the memory sources, and realize the effective memory transfer and writing via a dual-pathway compression design. Afterwards, FlexMem also explores different memory reading strategies for the diverse video understanding tasks, including the popular streaming one. To validate FlexMem, we apply it to two popular video-MLLMs, and conduct extensive experiments on five long video and one streaming video task. The experimental results show that on \textbf{a single 3090 GPU}, our FlexMem can achieve obvious improvements than existing efficient video understanding methods and process more than \textbf{1k frames}, which also helps the base MLLMs achieve comparable or even better performance than SOTA MLLMs on some benchmarks, \emph{e.g.} , GPT-4o and Gemini-1.5 Pro. 

---
# MemRerank: Preference Memory for Personalized Product Reranking 

**Authors**: Zhiyuan Peng, Xuyang Wu, Huaixiao Tou, Yi Fang, Yi Gong  

**Link**: [PDF](https://arxiv.org/pdf/2603.29247)  

**Abstract**: LLM-based shopping agents increasingly rely on long purchase histories and multi-turn interactions for personalization, yet naively appending raw history to prompts is often ineffective due to noise, length, and relevance mismatch. We propose MemRerank, a preference memory framework that distills user purchase history into concise, query-independent signals for personalized product reranking. To study this problem, we build an end-to-end benchmark and evaluation framework centered on an LLM-based \textbf{1-in-5} selection task, which measures both memory quality and downstream reranking utility. We further train the memory extractor with reinforcement learning (RL), using downstream reranking performance as supervision. Experiments with two LLM-based rerankers show that MemRerank consistently outperforms no-memory, raw-history, and off-the-shelf memory baselines, yielding up to \textbf{+10.61} absolute points in 1-in-5 accuracy. These results suggest that explicit preference memory is a practical and effective building block for personalization in agentic e-commerce systems. 

---
# Long-Document QA with Chain-of-Structured-Thought and Fine-Tuned SLMs 

**Authors**: Zhuowen Liang, Xiaotian Lin, Zhengxuan Zhang, Yuyu Luo, Haixun Wang, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29232)  

**Abstract**: Large language models (LLMs) are widely applied to data analytics over documents, yet direct reasoning over long, noisy documents remains brittle and error-prone. Hence, we study document question answering (QA) that consolidates dispersed evidence into a structured output (e.g., a table, graph, or chunks) to support reliable, verifiable QA. We propose a two-pillar framework, LiteCoST, to achieve both high accuracy and low latency with small language models (SLMs). Pillar 1: Chain-of-Structured-Thought (CoST). We introduce a CoST template, a schema-aware instruction that guides a strong LLM to produce both a step-wise CoST trace and the corresponding structured output. The process induces a minimal structure, normalizes entities/units, aligns records, serializes the output, and verifies/refines it, yielding auditable supervision. Pillar 2: SLM fine-tuning. The compact models are trained on LLM-generated CoST data in two stages: Supervised Fine-Tuning for structural alignment, followed by Group Relative Policy Optimization (GRPO) incorporating triple rewards for answer/format quality and process consistency. By distilling structure-first behavior into SLMs, this approach achieves LLM-comparable quality on multi-domain long-document QA using 3B/7B SLMs, while delivering 2-4x lower latency than GPT-4o and DeepSeek-R1 (671B). The code is available at this https URL. 

---
# Derived Fields Preserve Fine-Scale Detail in Budgeted Neural Simulators 

**Authors**: Wenshuo Wang, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29224)  

**Abstract**: Fine-scale-faithful neural simulation under fixed storage budgets remains challenging. Many existing methods reduce high-frequency error by improving architectures, training objectives, or rollout strategies. However, under budgeted coarsen-quantize-decode pipelines, fine detail can already be lost when the carried state is constructed. In the canonical periodic incompressible Navier-Stokes setting, we show that primitive and derived fields undergo systematically different retained-band distortions under the same operator. Motivated by this observation, we formulate Derived-Field Optimization (DerivOpt), a general state-design framework that chooses which physical fields are carried and how storage budget is allocated across them under a calibrated channel model. Across the full time-dependent forward subset of PDEBench, DerivOpt not only improves pooled mean rollout nRMSE, but also delivers a decisive advantage in fine-scale fidelity over a broad set of strong baselines. More importantly, the gains are already visible at input time, before rollout learning begins. This indicates that the carried state is often the dominant bottleneck under tight storage budgets. These results suggest a broader conclusion: in budgeted neural simulation, carried-state design should be treated as a first-class design axis alongside architecture, loss, and rollout strategy. 

---
# SyriSign: A Parallel Corpus for Arabic Text to Syrian Arabic Sign Language Translation 

**Authors**: Mohammad Amer Khalil, Raghad Nahas, Ahmad Nassar, Khloud Al Jallad  

**Link**: [PDF](https://arxiv.org/pdf/2603.29219)  

**Abstract**: Sign language is the primary approach of communication for the Deaf and Hard-of-Hearing (DHH) community. While there are numerous benchmarks for high-resource sign languages, low-resource languages like Arabic remain underrepresented. Currently, there is no publicly available dataset for Syrian Arabic Sign Language (SyArSL). To overcome this gap, we introduce SyriSign, a dataset comprising 1500 video samples across 150 unique lexical signs, designed for text-to-SyArSL translation tasks. This work aims to reduce communication barriers in Syria, as most news are delivered in spoken or written Arabic, which is often inaccessible to the deaf community. We evaluated SyriSign using three deep learning architectures: MotionCLIP for semantic motion generation, T2M-GPT for text-conditioned motion synthesis, and SignCLIP for bilingual embedding alignment. Experimental results indicate that while generative approaches show strong potential for sign representation, the limited dataset size constrains generalization performance. We will release SyriSign publicly, hoping it serves as an initial benchmark. 

---
# Software Vulnerability Detection Using a Lightweight Graph Neural Network 

**Authors**: Miles Farmer, Ekincan Ufuktepe, Anne Watson, Hialo Muniz Carvalho, Vadim Okun, Zineb Maasaoui, Kannappan Palaniappan  

**Link**: [PDF](https://arxiv.org/pdf/2603.29216)  

**Abstract**: Large Language Models (LLMs) have emerged as a popular choice in vulnerability detection studies given their foundational capabilities, open source availability, and variety of models, but have limited scalability due to extensive compute requirements. Using the natural graph relational structure of code, we show that our proposed graph neural network (GNN) based deep learning model VulGNN for vulnerability detection can achieve performance almost on par with LLMs, but is 100 times smaller in size and fast to retrain and customize. We describe the VulGNN architecture, ablation studies on components, learning rates, and generalizability to different code datasets. As a lightweight model for vulnerability analysis, VulGNN is efficient and deployable at the edge as part of real-world software development pipelines. 

---
# Improving Ensemble Forecasts of Abnormally Deflecting Tropical Cyclones with Fused Atmosphere-Ocean-Terrain Data 

**Authors**: Qixiang Li, Shuwei Huo, Chong Wang, Xiaofeng Li, Yuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.29200)  

**Abstract**: Deep learning-based tropical cyclone (TC) forecasting methods have demonstrated significant potential and application advantages, as they feature much lower computational cost and faster operation speed than numerical weather prediction models. However, existing deep learning methods still have key limitations: they can only process a single type of sequential trajectory data or homogeneous meteorological variables, and fail to achieve accurate forecasting of abnormal deflected TCs. To address these challenges, we present two groundbreaking contributions. First, we have constructed a multimodal and multi-source dataset named AOT-TCs for TC forecasting in the Northwest Pacific basin. As the first dataset of its kind, it innovatively integrates heterogeneous variables from the atmosphere, ocean, and land, thus obtaining a comprehensive and information-rich meteorological dataset. Second, based on the AOT-TCs dataset, we propose a forecasting model that can handle both normal and abnormally deflected TCs. This is the first TC forecasting model to adopt an explicit atmosphere-ocean-terrain coupling architecture, enabling it to effectively capture complex interactions across physical domains. Extensive experiments on all TC cases in the Northwest Pacific from 2017 to 2024 show that our model achieves state-of-the-art performance in TC forecasting: it not only significantly improves the forecasting accuracy of normal TCs but also breaks through the technical bottleneck in forecasting abnormally deflected TCs. 

---
# Multi-Layered Memory Architectures for LLM Agents: An Experimental Evaluation of Long-Term Context Retention 

**Authors**: Sunil Tiwari, Payal Fofadiya  

**Link**: [PDF](https://arxiv.org/pdf/2603.29194)  

**Abstract**: Long-horizon dialogue systems suffer from semanticdrift and unstable memory retention across extended sessions. This paper presents a Multi-Layer Memory Framework that decomposes dialogue history into working, episodic, and semantic layers with adaptive retrieval gating and retention regularization. The architecture controls cross-session drift while maintaining bounded context growth and computational efficiency. Experiments on LOCOMO, LOCCO, and LoCoMo show improved performance, achieving 46.85 Success Rate, 0.618 overall F1 with 0.594 multi-hop F1, and 56.90% six-period retention while reducing false memory rate to 5.1% and context usage to 58.40%. Results confirm enhanced long-term retention and reasoning stability under constrained context budgets. 

---
# Developing Adaptive Context Compression Techniques for Large Language Models (LLMs) in Long-Running Interactions 

**Authors**: Payal Fofadiya, Sunil Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2603.29193)  

**Abstract**: Large Language Models (LLMs) often experience performance degradation during long-running interactions due to increasing context length, memory saturation, and computational overhead. This paper presents an adaptive context compression framework that integrates importance-aware memory selection, coherence-sensitive filtering, and dynamic budget allocation to retain essential conversational information while controlling context growth. The approach is evaluated on LOCOMO, LOCCO, and LongBench benchmarks to assess answer quality, retrieval accuracy, coherence preservation, and efficiency. Experimental results demonstrate that the proposed method achieves consistent improvements in conversational stability and retrieval performance while reducing token usage and inference latency compared with existing memory and compression-based approaches. These findings indicate that adaptive context compression provides an effective balance between long-term memory preservation and computational efficiency in persistent LLM interactions 

---
# 3D Architect: An Automated Approach to Three-Dimensional Modeling 

**Authors**: Sunil Tiwari, Payal Fofadiya, Vicky Vishwakarma  

**Link**: [PDF](https://arxiv.org/pdf/2603.29191)  

**Abstract**: The aim of our paper is to render an object in 3-dimension using a set of its orthographic views. Corner detector (Harris Detector) is applied on the input views to obtain control points. These control points are projected perpendicular to respective views, in order to construct an envelope. A set of points describing the object in 3-dimension, are obtained from the intersection of these mutually perpendicular envelopes. These set of points are used to regenerate the surfaces of the object using computational geometry. At the end, the object in 3-dimension is rendered using OpenGL 

---
# SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation 

**Authors**: Ryosuke Matsuda, Keito Kudo, Haruto Yoshida, Nobuyuki Shimizu, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2603.29186)  

**Abstract**: This paper proposes the synthetic long-video meta-evaluation (SLVMEval), a benchmark for meta-evaluating text-to-video (T2V) evaluation systems. The proposed SLVMEval benchmark focuses on assessing these systems on videos of up to 10,486 s (approximately 3 h). The benchmark targets a fundamental requirement, namely, whether the systems can accurately assess video quality in settings that are easy for humans to assess. We adopt a pairwise comparison-based meta-evaluation framework. Building on dense video-captioning datasets, we synthetically degrade source videos to create controlled "high-quality versus low-quality" pairs across 10 distinct aspects. Then, we employ crowdsourcing to filter and retain only those pairs in which the degradation is clearly perceptible, thereby establishing an effective final testbed. Using this testbed, we assess the reliability of existing evaluation systems in ranking these pairs. Experimental results demonstrate that human evaluators can identify the better long video with 84.7%-96.8% accuracy, and in nine of the 10 aspects, the accuracy of these systems falls short of human assessment, revealing weaknesses in text-to-long-video evaluation. 

---
# IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection 

**Authors**: Xiaohui Zhou, Yijie Wang, Hongzuo Xu, Weixuan Liang, Xiaoli Li, Guansong Pang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29183)  

**Abstract**: Open-set anomaly detection (OSAD) is an emerging paradigm designed to utilize limited labeled data from anomaly classes seen in training to identify both seen and unseen anomalies during testing. Current approaches rely on simple augmentation methods to generate pseudo anomalies that replicate unseen anomalies. Despite being promising in image data, these methods are found to be ineffective in time series data due to the failure to preserve its sequential nature, resulting in trivial or unrealistic anomaly patterns. They are further plagued when the training data is contaminated with unlabeled anomalies. This work introduces $\textbf{IMPACT}$, a novel framework that leverages $\underline{\textbf{i}}$nfluence $\underline{\textbf{m}}$odeling for o$\underline{\textbf{p}}$en-set time series $\underline{\textbf{a}}$nomaly dete$\underline{\textbf{ct}}$ion, to tackle these challenges. The key insight is to $\textbf{i)}$ learn an influence function that can accurately estimate the impact of individual training samples on the modeling, and then $\textbf{ii)}$ leverage these influence scores to generate semantically divergent yet realistic unseen anomalies for time series while repurposing high-influential samples as supervised anomalies for anomaly decontamination. Extensive experiments show that IMPACT significantly outperforms existing state-of-the-art methods, showing superior accuracy under varying OSAD settings and contamination rates. 

---
# Predicting Neuromodulation Outcome for Parkinson's Disease with Generative Virtual Brain Model 

**Authors**: Siyuan Du, Siyi Li, Shuwei Bai, Ang Li, Haolin Li, Mingqing Xiao, Yang Pan, Dongsheng Li, Weidi Xie, Yanfeng Wang, Ya Zhang, Chencheng Zhang, Jiangchao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2603.29176)  

**Abstract**: Parkinson's disease (PD) affects over ten million people worldwide. Although temporal interference (TI) and deep brain stimulation (DBS) are promising therapies, inter-individual variability limits empirical treatment selection, increasing non-negligible surgical risk and cost. Previous explorations either resort to limited statistical biomarkers that are insufficient to characterize variability, or employ AI-driven methods which is prone to overfitting and opacity. We bridge this gap with a pretraining-finetuning framework to predict outcomes directly from resting-state fMRI. Critically, a generative virtual brain foundation model, pretrained on a collective dataset (2707 subjects, 5621 sessions) to capture universal disorder patterns, was finetuned on PD cohorts receiving TI (n=51) or DBS (n=55) to yield individualized virtual brains with high fidelity to empirical functional connectivity (r=0.935). By constructing counterfactual estimations between pathological and healthy neural states within these personalized models, we predicted clinical responses (TI: AUPR=0.853; DBS: AUPR=0.915), substantially outperforming baselines. External and prospective validations (n=14, n=11) highlight the feasibility of clinical translation. Moreover, our framework provides state-dependent regional patterns linked to response, offering hypothesis-generating mechanistic insights. 

---
# LatentPilot: Scene-Aware Vision-and-Language Navigation by Dreaming Ahead with Latent Visual Reasoning 

**Authors**: Haihong Hao, Lei Chen, Mingfei Han, Changlin Li, Dong An, Yuqiang Yang, Zhihui Li, Xiaojun Chang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29165)  

**Abstract**: Existing vision-and-language navigation (VLN) models primarily reason over past and current visual observations, while largely ignoring the future visual dynamics induced by actions. As a result, they often lack an effective understanding of the causal relationship between actions and how the visual world changes, limiting robust decision-making. Humans, in contrast, can imagine the near future by leveraging action-dynamics causality, which improves both environmental understanding and navigation choices. Inspired by this capability, we propose LatentPilot, a new paradigm that exploits future observations during training as a valuable data source to learn action-conditioned visual dynamics, while requiring no access to future frames at inference. Concretely, we propose a flywheel-style training mechanism that iteratively collects on-policy trajectories and retrains the model to better match the agent's behavior distribution, with an expert takeover triggered when the agent deviates excessively. LatentPilot further learns visual latent tokens without explicit supervision; these latent tokens attend globally in a continuous latent space and are carried across steps, serving as both the current output and the next input, thereby enabling the agent to dream ahead and reason about how actions will affect subsequent observations. Experiments on R2R-CE, RxR-CE, and R2R-PE benchmarks achieve new SOTA results, and real-robot tests across diverse environments demonstrate LatentPilot's superior understanding of environment-action dynamics in scene. Project page:this https URL 

---
# Efficient and Scalable Granular-ball Graph Coarsening Method for Large-scale Graph Node Classification 

**Authors**: Guan Wang, Shuyin Xia, Lei Qian, Guoyin Wang, Yi Liu, Yi Wang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29148)  

**Abstract**: Graph Convolutional Network (GCN) is a model that can effectively handle graph data tasks and has been successfully applied. However, for large-scale graph datasets, GCN still faces the challenge of high computational overhead, especially when the number of convolutional layers in the graph is large. Currently, there are many advanced methods that use various sampling techniques or graph coarsening techniques to alleviate the inconvenience caused during training. However, among these methods, some ignore the multi-granularity information in the graph structure, and the time complexity of some coarsening methods is still relatively high. In response to these issues, based on our previous work, in this paper, we propose a new framework called Efficient and Scalable Granular-ball Graph Coarsening Method for Large-scale Graph Node Classification. Specifically, this method first uses a multi-granularity granular-ball graph coarsening algorithm to coarsen the original graph to obtain many subgraphs. The time complexity of this stage is linear and much lower than that of the exiting graph coarsening methods. Then, subgraphs composed of these granular-balls are randomly sampled to form minibatches for training GCN. Our algorithm can adaptively and significantly reduce the scale of the original graph, thereby enhancing the training efficiency and scalability of GCN. Ultimately, the experimental results of node classification on multiple datasets demonstrate that the method proposed in this paper exhibits superior performance. The code is available at this https URL. 

---
# Designing FSMs Specifications from Requirements with GPT 4.0 

**Authors**: Omer Nguena Timo, Paul-Alexis Rodriguez, Florent Avellaneda  

**Link**: [PDF](https://arxiv.org/pdf/2603.29140)  

**Abstract**: Finite state machines (FSM) are executable formal specifications of reactive systems. These machines are designed based on systems' requirements. The requirements are often recorded in textual documents written in natural languages. FSMs play a crucial role in different phases of the model-driven system engineering (MDE). For example, they serve to automate testing activities. FSM quality is critical: the lower the quality of FSM, the higher the number of faults surviving the testing phase and the higher the risk of failure of the systems in production, which could lead to catastrophic scenarios. Therefore, this paper leverages recent advances in the domain of LLM to propose an LLM-based framework for designing FSMs from requirements. The framework also suggests an expert-centric approach based on FSM mutation and test generation for repairing the FSMs produced by LLMs. This paper also provides an experimental analysis and evaluation of LLM's capacities in performing the tasks presented in the framework and FSM repair via various methods. The paper presents experimental results with simulated data. These results and methods bring a new analysis and vision of LLMs that are useful for further development of machine learning technology and its applications to MDE. 

---
# Economics of Human and AI Collaboration: When is Partial Automation More Attractive than Full Automation? 

**Authors**: Wensu Li, Atin Aboutorabi, Harry Lyu, Kaizhi Qian, Martin Fleming, Brian C. Goehring, Neil Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2603.29121)  

**Abstract**: This paper develops a unified framework for evaluating the optimal degree of task automation. Moving beyond binary automate-or-not assessments, we model automation intensity as a continuous choice in which firms minimize costs by selecting an AI accuracy level, from no automation through partial human-AI collaboration to full automation. On the supply side, we estimate an AI production function via scaling-law experiments linking performance to data, compute, and model size. Because AI systems exhibit predictable but diminishing returns to these inputs, the cost of higher accuracy is convex: good performance may be inexpensive, but near-perfect accuracy is disproportionately costly. Full automation is therefore often not cost-minimizing; partial automation, where firms retain human workers for residual tasks, frequently emerges as the equilibrium. On the demand side, we introduce an entropy-based measure of task complexity that maps model accuracy into a labor substitution ratio, quantifying human labor displacement at each accuracy level. We calibrate the framework with O*NET task data, a survey of 3,778 domain experts, and GPT-4o-derived task decompositions, implementing it in computer vision. Task complexity shapes substitution: low-complexity tasks see high substitution, while high-complexity tasks favor limited partial automation. Scale of deployment is a key determinant: AI-as-a-Service and AI agents spread fixed costs across users, sharply expanding economically viable tasks. At the firm level, cost-effective automation captures approximately 11% of computer-vision-exposed labor compensation; under economy-wide deployment, this share rises sharply. Since other AI systems exhibit similar scaling-law economics, our mechanisms extend beyond computer vision, reinforcing that partial automation is often the economically rational long-run outcome, not merely a transitional phase. 

---
# "I Just Need GPT to Refine My Prompts": Rethinking Onboarding and Help-Seeking with Generative 3D Modeling Tools 

**Authors**: Kanak Gautam, Poorvi Bhatia, Parmit K. Chilana  

**Link**: [PDF](https://arxiv.org/pdf/2603.29118)  

**Abstract**: Learning to use feature-rich software is a persistent challenge, but generative AI tools promise to lower this barrier by replacing complex navigation with natural language prompts. We investigated how people approach prompt-based tools for 3D modeling in an observational study with 26 participants (14 casuals, 12 professionals). Consistent with earlier work, participants skipped tutorials and manuals, relying on trial and error. What differed in the generative AI context was how and why they sought support: the prompt box became the entry point for learning, collapsing onboarding into immediate action, while some casual users turned to external LLMs for prompts. Professionals used 3D expertise to refine iterations and critically evaluated outputs, often discarding models that did not meet their standards, whereas casual users settled for "good enough." We contribute empirical insights into how generative AI reshapes help-seeking, highlighting new practices of onboarding, recursive AI-for-AI support, and shifting expertise in interpreting outputs. 

---
# Towards Explainable Stakeholder-Aware Requirements Prioritisation in Aged-Care Digital Health 

**Authors**: Yuqing Xiao, John Grundy, Anuradha Madugalla, Elizabeth Manias  

**Link**: [PDF](https://arxiv.org/pdf/2603.29114)  

**Abstract**: Requirements engineering for aged-care digital health must account for human aspects, because requirement priorities are shaped not only by technical functionality but also by stakeholders' health conditions, socioeconomics, and lived experience. Knowing which human aspects matter most, and for whom, is critical for inclusive and evidence-based requirements prioritisation. Yet in practice, while some studies have examined human aspects in RE, they have largely relied on expert judgement or model-driven analysis rather than large-scale user studies with meaningful human-in-the-loop validation to determine which aspects matter most and why. To address this gap, we conducted a mixed-methods study with 103 older adults, 105 developers, and 41 caregivers. We first applied an explainable machine learning to identify the human aspects most strongly associated with requirement priorities across 8 aged-care digital health themes, and then conducted 12 semi-structured interviews to validate and interpret the quantitative patterns. The results identify the key human aspects shaping requirement priorities, reveal their directional effects, and expose substantial misalignment across stakeholder groups. Together, these findings show that human-centric requirements analysis should engage stakeholder groups explicitly rather than collapsing their perspectives into a single aggregate view. This paper contributes an identification of the key human aspects driving requirement priorities in aged-care digital health and an explainable, human-centric RE framework that combines ML-derived importance rankings with qualitative validation to surface the stakeholder misalignments that inclusive requirements engineering must address. 

---
# SemLoc: Structured Grounding of Free-Form LLM Reasoning for Fault Localization 

**Authors**: Zhaorui Yang, Haichao Zhu, Qian Zhang, Rajiv Gupta, Ashish Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29109)  

**Abstract**: Fault localization identifies program locations responsible for observed failures. Existing techniques rank suspicious code using syntactic spectra--signals derived from execution structure such as statement coverage, control-flow divergence, or dependency reachability. These signals collapse for semantic bugs, where failing and passing executions follow identical code paths and differ only in whether semantic intent is satisfied. Recent LLM-based approaches introduce semantic reasoning but produce stochastic, unverifiable outputs that cannot be systematically cross-referenced across tests or distinguish root causes from cascading effects.
We present SemLoc, a fault localization framework based on structured semantic grounding. SemLoc converts free-form LLM reasoning into a closed intermediate representation that binds each inferred property to a typed program anchor, enabling runtime checking and attribution to program structure. It executes instrumented programs to construct a semantic violation spectrum--a constraint-by-test matrix--from which suspiciousness scores are derived analogously to coverage-based methods. A counterfactual verification step further prunes over-approximate constraints and isolates primary causal violations.
We evaluate SemLoc on SemFault-250, a corpus of 250 Python programs with single semantic faults. SemLoc outperforms five coverage-, reduction-, and LLM-based baselines, achieving Top-1 accuracy of 42.8% and Top-3 of 68%, while reducing inspection to 7.6% of executable lines. Counterfactual verification provides an additional 12% accuracy gain and identifies primary causal semantic constraints. 

---
# Evaluating a Data-Driven Redesign Process for Intelligent Tutoring Systems 

**Authors**: Qianru Lyu, Conrad Borchers, Meng Xia, Karen Xiao, Paulo F. Carvalho, Kenneth R. Koedinger, Vincent Aleven  

**Link**: [PDF](https://arxiv.org/pdf/2603.29094)  

**Abstract**: Past research has defined a general process for the data-driven redesign of educational technologies and has shown that in carefully-selected instances, this process can help make systems more effective. In the current work, we test the generality of the approach by applying it to four units of a middle-school mathematics intelligent tutoring system that were selected not based on suitability for redesign, as in previous work, but on topic. We tested whether the redesigned system was more effective than the original in a classroom study with 123 students. Although the learning gains did not differ between the conditions, students who used the Redesigned Tutor had more productive time-on-task, a larger number of skills practiced, and greater total knowledge mastery. The findings highlight the promise of data-driven redesign even when applied to instructional units *not* selected as likely to yield improvement, as evidence of the generality and wide applicability of the method. 

---
# APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay 

**Authors**: Pratyay Banerjee, Masud Moshtaghi, Ankit Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2603.29093)  

**Abstract**: LLM-based autonomous agents lack persistent procedural memory: they re-derive solutions from scratch even when structurally identical tasks have been solved before. We present \textbf{APEX-EM}, a non-parametric online learning framework that accumulates, retrieves, and reuses structured procedural plans without modifying model weights. APEX-EM introduces: (1) a \emph{structured experience representation} encoding the full procedural-episodic trace of each execution -- planning steps, artifacts, iteration history with error analysis, and quality scores; (2) a \emph{Plan-Retrieve-Generate-Iterate-Ingest} (PRGII) workflow with Task Verifiers providing multi-dimensional reward signals; and (3) a \emph{dual-outcome Experience Memory} with hybrid retrieval combining semantic search, structural signature matching, and plan DAG traversal -- enabling cross-domain transfer between tasks sharing no lexical overlap but analogous operational structure. Successful experiences serve as positive in-context examples; failures as negative examples with structured error annotations.
We evaluate on BigCodeBench~\cite{zhuo2025bigcodebench}, KGQAGen-10k~\cite{zhang2025kgqagen}, and Humanity's Last Exam~\cite{phan2025hle} using Claude Sonnet 4.5 and Opus 4.5. On KGQAGen-10k, APEX-EM achieves 89.6\% accuracy versus 41.3\% without memory (+48.3pp), surpassing the oracle-retrieval upper bound (84.9\%). On BigCodeBench, it reaches 83.3\% SR from a 53.9\% baseline (+29.4pp), exceeding MemRL's~\cite{memrl2025} +11.0pp gain under comparable frozen-backbone conditions (noting backbone differences controlled for in our analysis). On HLE, entity graph retrieval reaches 48.0\% from 25.2\% (+22.8pp). Ablations show component value is task-dependent: rich judge feedback is negligible for code generation but critical for structured queries (+10.3pp), while binary-signal iteration partially compensates for weaker feedback. 

---
# WorldFlow3D: Flowing Through 3D Distributions for Unbounded World Generation 

**Authors**: Amogh Joshi, Julian Ost, Felix Heide  

**Link**: [PDF](https://arxiv.org/pdf/2603.29089)  

**Abstract**: Unbounded 3D world generation is emerging as a foundational task for scene modeling in computer vision, graphics, and robotics. In this work, we present WorldFlow3D, a novel method capable of generating unbounded 3D worlds. Building upon a foundational property of flow matching - namely, defining a path of transport between two data distributions - we model 3D generation more generally as a problem of flowing through 3D data distributions, not limited to conditional denoising. We find that our latent-free flow approach generates causal and accurate 3D structure, and can use this as an intermediate distribution to guide the generation of more complex structure and high-quality texture - all while converging more rapidly than existing methods. We enable controllability over generated scenes with vectorized scene layout conditions for geometric structure control and visual texture control through scene attributes. We confirm the effectiveness of WorldFlow3D on both real outdoor driving scenes and synthetic indoor scenes, validating cross-domain generalizability and high-quality generation on real data distributions. We confirm favorable scene generation fidelity over approaches in all tested settings for unbounded scene generation. For more, see this https URL. 

---
# WybeCoder: Verified Imperative Code Generation 

**Authors**: Fabian Gloeckle, Mantas Baksys, Darius Feher, Kunhao Zheng, Amaury Hayat, Sean B. Holden, Gabriel Synnaeve, Peter O'Hearn  

**Link**: [PDF](https://arxiv.org/pdf/2603.29088)  

**Abstract**: Recent progress in large language models (LLMs) has advanced automatic code generation and formal theorem proving, yet software verification has not seen the same improvement. To address this gap, we propose WybeCoder, an agentic code verification framework that enables prove-as-you-generate development where code, invariants, and proofs co-evolve. It builds on a recent framework that combines automatic verification condition generation and SMT solvers with interactive proofs in Lean. To enable systematic evaluation, we translate two benchmarks for functional verification in Lean, Verina and Clever, to equivalent imperative code specifications. On complex algorithms such as Heapsort, we observe consistent performance improvements by scaling our approach, synthesizing dozens of valid invariants and dispatching of dozens of subgoals, resulting in hundreds of lines of verified code, overcoming plateaus reported in previous works. Our best system solves 74% of Verina tasks and 62% of Clever tasks at moderate compute budgets, significantly surpassing previous evaluations and paving a path to automated construction of large-scale datasets of verified imperative code. 

---
# On the Mirage of Long-Range Dependency, with an Application to Integer Multiplication 

**Authors**: Zichao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.29069)  

**Abstract**: Integer multiplication has long been considered a hard problem for neural networks, with the difficulty widely attributed to the O(n) long-range dependency induced by carry chains. We argue that this diagnosis is wrong: long-range dependency is not an intrinsic property of multiplication, but a mirage produced by the choice of computational spacetime. We formalize the notion of mirage and provide a constructive proof: when two n-bit binary integers are laid out as a 2D outer-product grid, every step of long multiplication collapses into a $3 \times 3$ local neighborhood operation. Under this representation, a neural cellular automaton with only 321 learnable parameters achieves perfect length generalization up to $683\times$ the training range. Five alternative architectures -- including Transformer (6,625 params), Transformer+RoPE, and Mamba -- all fail under the same representation. We further analyze how partial successes locked the community into an incorrect diagnosis, and argue that any task diagnosed as requiring long-range dependency should first be examined for whether the dependency is intrinsic to the task or induced by the computational spacetime. 

---
# CivicShield: A Cross-Domain Defense-in-Depth Framework for Securing Government-Facing AI Chatbots Against Multi-Turn Adversarial Attacks 

**Authors**: KrishnaSaiReddy Patil  

**Link**: [PDF](https://arxiv.org/pdf/2603.29062)  

**Abstract**: LLM-based chatbots in government services face critical security gaps. Multi-turn adversarial attacks achieve over 90% success against current defenses, and single-layer guardrails are bypassed with similar rates. We present CivicShield, a cross-domain defense-in-depth framework for government-facing AI chatbots. Drawing on network security, formal verification, biological immune systems, aviation safety, and zero-trust cryptography, CivicShield introduces seven defense layers: (1) zero-trust foundation with capability-based access control, (2) perimeter input validation, (3) semantic firewall with intent classification, (4) conversation state machine with safety invariants, (5) behavioral anomaly detection, (6) multi-model consensus verification, and (7) graduated human-in-the-loop escalation. We present a formal threat model covering 8 multi-turn attack families, map the framework to NIST SP 800-53 controls across 14 families, and evaluate using ablation analysis. Theoretical analysis shows layered defenses reduce attack probability by 1-2 orders of magnitude versus single-layer approaches. Simulation against 1,436 scenarios including HarmBench (416), JailbreakBench (200), and XSTest (450) achieves 72.9% combined detection [69.5-76.0% CI] with 2.9% effective false positive rate after graduated response, while maintaining 100% detection of multi-turn crescendo and slow-drift attacks. The honest drop on real benchmarks versus author-generated scenarios (71.2% vs 76.7% on HarmBench, 47.0% vs 70.0% on JailbreakBench) validates independent evaluation importance. CivicShield addresses an open gap at the intersection of AI safety, government compliance, and practical deployment. 

---
# A Latent Risk-Aware Machine Learning Approach for Predicting Operational Success in Clinical Trials based on TrialsBank 

**Authors**: Iness Halimi, Emmanuel Piffo, Oumnia Boudersa, Yvan Marcel Carre Vilmorin, Melissa Ait-ikhlef, Karima Kone, Andy Tan, Augustin Medina, Juliette Hernando, Sheila Ernest, Vatche Bartekian, Karine Lalonde, Mireille E Schnitzer, Gianolli Dorcelus  

**Link**: [PDF](https://arxiv.org/pdf/2603.29041)  

**Abstract**: Clinical trials are characterized by high costs, extended timelines, and substantial operational risk, yet reliable prospective methods for predicting trial success before initiation remain limited. Existing artificial intelligence approaches often focus on isolated metrics or specific development stages and frequently rely on variables unavailable at the trial design phase, limiting real-world applicability. We present a hierarchical latent risk-aware machine learning framework for prospective prediction of clinical trial operational success using a curated subset of TrialsBank, a proprietary AI-ready database developed by Sorintellis, comprising 13,700 trials. Operational success was defined as the ability to initiate, conduct, and complete a clinical trial according to planned timelines, recruitment targets, and protocol specifications through database lock. This approach decomposes operational success prediction into two modeling stages. First, intermediate latent operational risk factors are predicted using more than 180 drug- and trial-level features available before trial initiation. These predicted latent risks are then integrated into a downstream model to estimate the probability of operational success. A staged data-splitting strategy was employed to prevent information leakage, and models were benchmarked using XGBoost, CatBoost, and Explainable Boosting Machines. Across Phase I-III, the framework achieves strong out-of-sample performance, with F1-scores of 0.93, 0.92, and 0.91, respectively. Incorporating latent risk drivers improves discrimination of operational failures, and performance remains robust under independent inference evaluation. These results demonstrate that clinical trial operational success can be prospectively forecasted using a latent risk-aware AI framework, enabling early risk assessment and supporting data-driven clinical development decision-making. 

---
# Trojan-Speak: Bypassing Constitutional Classifiers with No Jailbreak Tax via Adversarial Finetuning 

**Authors**: Bilgehan Sel, Xuanli He, Alwin Peng, Ming Jin, Jerry Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.29038)  

**Abstract**: Fine-tuning APIs offered by major AI providers create new attack surfaces where adversaries can bypass safety measures through targeted fine-tuning. We introduce Trojan-Speak, an adversarial fine-tuning method that bypasses Anthropic's Constitutional Classifiers. Our approach uses curriculum learning combined with GRPO-based hybrid reinforcement learning to teach models a communication protocol that evades LLM-based content classification. Crucially, while prior adversarial fine-tuning approaches report more than 25% capability degradation on reasoning benchmarks, Trojan-Speak incurs less than 5% degradation while achieving 99+% classifier evasion for models with 14B+ parameters. We demonstrate that fine-tuned models can provide detailed responses to expert-level CBRN (Chemical, Biological, Radiological, and Nuclear) queries from Anthropic's Constitutional Classifiers bug-bounty program. Our findings reveal that LLM-based content classifiers alone are insufficient for preventing dangerous information disclosure when adversaries have fine-tuning access, and we show that activation-level probes can substantially improve robustness to such attacks. 

---
# MMFace-DiT: A Dual-Stream Diffusion Transformer for High-Fidelity Multimodal Face Generation 

**Authors**: Bharath Krishnamurthy, Ajita Rattani  

**Link**: [PDF](https://arxiv.org/pdf/2603.29029)  

**Abstract**: Recent multimodal face generation models address the spatial control limitations of text-to-image diffusion models by augmenting text-based conditioning with spatial priors such as segmentation masks, sketches, or edge maps. This multimodal fusion enables controllable synthesis aligned with both high-level semantic intent and low-level structural layout. However, most existing approaches typically extend pre-trained text-to-image pipelines by appending auxiliary control modules or stitching together separate uni-modal networks. These ad hoc designs inherit architectural constraints, duplicate parameters, and often fail under conflicting modalities or mismatched latent spaces, limiting their ability to perform synergistic fusion across semantic and spatial domains. We introduce MMFace-DiT, a unified dual-stream diffusion transformer engineered for synergistic multimodal face synthesis. Its core novelty lies in a dual-stream transformer block that processes spatial (mask/sketch) and semantic (text) tokens in parallel, deeply fusing them through a shared Rotary Position-Embedded (RoPE) Attention mechanism. This design prevents modal dominance and ensures strong adherence to both text and structural priors to achieve unprecedented spatial-semantic consistency for controllable face generation. Furthermore, a novel Modality Embedder enables a single cohesive model to dynamically adapt to varying spatial conditions without retraining. MMFace-DiT achieves a 40% improvement in visual fidelity and prompt alignment over six state-of-the-art multimodal face generation models, establishing a flexible new paradigm for end-to-end controllable generative modeling. The code and dataset are available on our project page: this https URL 

---
# The Model Says Walk: How Surface Heuristics Override Implicit Constraints in LLM Reasoning 

**Authors**: Yubo Li, Lu Zhang, Tianchong Jiang, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2603.29025)  

**Abstract**: Large language models systematically fail when a salient surface cue conflicts with an unstated feasibility constraint. We study this through a diagnose-measure-bridge-treat framework. Causal-behavioral analysis of the ``car wash problem'' across six models reveals approximately context-independent sigmoid heuristics: the distance cue exerts 8.7 to 38 times more influence than the goal, and token-level attribution shows patterns more consistent with keyword associations than compositional inference. The Heuristic Override Benchmark (HOB) -- 500 instances spanning 4 heuristic by 5 constraint families with minimal pairs and explicitness gradients -- demonstrates generality across 14 models: under strict evaluation (10/10 correct), no model exceeds 75%, and presence constraints are hardest (44%). A minimal hint (e.g., emphasizing the key object) recovers +15 pp on average, suggesting the failure lies in constraint inference rather than missing knowledge; 12/14 models perform worse when the constraint is removed (up to -39 pp), revealing conservative bias. Parametric probes confirm that the sigmoid pattern generalizes to cost, efficiency, and semantic-similarity heuristics; goal-decomposition prompting recovers +6 to 9 pp by forcing models to enumerate preconditions before answering. Together, these results characterize heuristic override as a systematic reasoning vulnerability and provide a benchmark for measuring progress toward resolving it. 

---
# Human-Like Lifelong Memory: A Neuroscience-Grounded Architecture for Infinite Interaction 

**Authors**: Diego C. Lerma-Torres  

**Link**: [PDF](https://arxiv.org/pdf/2603.29023)  

**Abstract**: Large language models lack persistent, structured memory for long-term interaction and context-sensitive retrieval. Expanding context windows does not solve this: recent evidence shows that context length alone degrades reasoning by up to 85% - even with perfect retrieval. We propose a bio-inspired memory framework grounded in complementary learning systems theory, cognitive behavioral therapy's belief hierarchy, dual-process cognition, and fuzzy-trace theory, organized around three principles: (1) Memory has valence, not just content - pre-computed emotional-associative summaries (valence vectors) organized in an emergent belief hierarchy inspired by Beck's cognitive model enable instant orientation before deliberation; (2) Retrieval defaults to System 1 with System 2 escalation - automatic spreading activation and passive priming as default, with deliberate retrieval only when needed, and graded epistemic states that address hallucination structurally; and (3) Encoding is active, present, and feedback-dependent - a thalamic gateway tags and routes information between stores, while the executive forms gists through curiosity-driven investigation, not passive exposure. Seven functional properties specify what any implementation must satisfy. Over time, the system converges toward System 1 processing - the computational analog of clinical expertise - producing interactions that become cheaper, not more expensive, with experience. 

---
# Improving Efficiency of GPU Kernel Optimization Agents using a Domain-Specific Language and Speed-of-Light Guidance 

**Authors**: Siva Kumar Sastry Hari, Vignesh Balaji, Sana Damani, Qijing Huang, Christos Kozyrakis  

**Link**: [PDF](https://arxiv.org/pdf/2603.29010)  

**Abstract**: Optimizing GPU kernels with LLM agents is an iterative process over a large design space. Every candidate must be generated, compiled, validated, and profiled, so fewer trials will save both runtime and cost. We make two key observations. First, the abstraction level that agents operate at is important. If it is too low, the LLM wastes reasoning on low-impact details. If it is too high, it may miss important optimization choices. Second, agents cannot easily tell when they reach the point of diminishing returns, wasting resources as they continue searching.
These observations motivate two design principles to improve efficiency: (1) a compact domain-specific language (DSL) that can be learned in context and lets the model reason at a higher level while preserving important optimization levers, and (2) Speed-of-Light (SOL) guidance that uses first-principles performance bounds to steer and budget search. We implement these principles in $\mu$CUTLASS, a DSL with a compiler for CUTLASS-backed GPU kernels that covers kernel configuration, epilogue fusion, and multi-stage pipelines. We use SOL guidance to estimate headroom and guide optimization trials, deprioritize problems that are near SOL, and flag kernels that game the benchmark.
On 59 KernelBench problems with the same iteration budgets, switching from generating low-level code to DSL code using GPT-5-mini turns a 0.40x geomean regression into a 1.27x speedup over PyTorch. Adding SOL-guided steering raises this to 1.56x. Across model tiers, $\mu$CUTLASS + SOL-guidance lets weaker models outperform stronger baseline agents at lower token cost. SOL-guided budgeting saves 19-43% of tokens while retaining at least 95% of geomean speedup, with the best policy reaching a 1.68x efficiency gain. Lastly, SOL analysis helps detect benchmark-gaming cases, where kernels may appear fast while failing to perform the intended computation. 

---
# Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference 

**Authors**: Zifan He, Rui Ma, Yizhou Sun, Jason Cong  

**Link**: [PDF](https://arxiv.org/pdf/2603.29002)  

**Abstract**: Modern large language models (LLMs) increasingly depends on efficient long-context processing and generation mechanisms, including sparse attention, retrieval-augmented generation (RAG), and compressed contextual memory, to support complex reasoning. We show that these optimizations can be unified into a four-step memory processing pipeline: Prepare Memory, Compute Relevancy, Retrieval, and Apply to Inference. Through systematic profiling, we identify a 22%-97% memory processing overhead in LLM inference and strong heterogeneity in its computational characteristics. Motivated by this insight, we argue that \textbf{heterogeneous systems} are well-suited to accelerate memory processing and thus end-to-end inference. We demonstrate this approach on a GPU-FPGA system by offloading sparse, irregular, and memory-bounded operations to FPGAs while retaining compute-intensive operations on GPUs. Evaluated on an AMD MI210 GPU and an Alveo U55C FPGA, our system is $1.04\sim2.2\times$ faster and requires $1.11\sim4.7\times$ less energy across multiple LLM inference optimizations than the GPU baseline (similar results hold on NVIDIA A100). These results establish heterogeneous systems as a practical direction for efficient LLM memory processing and inform future heterogeneous hardware design. 

---
# Design Principles for the Construction of a Benchmark Evaluating Security Operation Capabilities of Multi-agent AI Systems 

**Authors**: Yicheng Cai, Mitchell John DeStefano, Guodong Dong, Pulkit Handa, Peng Liu, Tejas Singhal, Peiyu Tseng, Winston Jen White  

**Link**: [PDF](https://arxiv.org/pdf/2603.28998)  

**Abstract**: As Large Language Models (LLMs) and multi-agent AI systems are demonstrating increasing potential in cybersecurity operations, organizations, policymakers, model providers, and researchers in the AI and cybersecurity communities are interested in quantifying the capabilities of such AI systems to achieve more autonomous SOCs (security operation centers) and reduce manual effort. In particular, the AI and cybersecurity communities have recently developed several benchmarks for evaluating the red team capabilities of multi-agent AI systems. However, because the operations in SOCs are dominated by blue team operations, the capabilities of AI systems & agents to achieve more autonomous SOCs cannot be evaluated without a benchmark focused on blue team operations. To our best knowledge, no systematic benchmark for evaluating coordinated multi-task blue team AI has been proposed in the literature. Existing blue team benchmarks focus on a particular task. The goal of this work is to develop a set of design principles for the construction of a benchmark, which is denoted as SOC-bench, to evaluate the blue team capabilities of AI. Following these design principles, we have developed a conceptual design of SOC-bench, which consists of a family of five blue team tasks in the context of large-scale ransomware attack incident response. 

---
# Privacy Guard & Token Parsimony by Prompt and Context Handling and LLM Routing 

**Authors**: Alessio Langiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.28972)  

**Abstract**: The large-scale adoption of Large Language Models (LLMs) forces a trade-off between operational cost (OpEx) and data privacy. Current routing frameworks reduce costs but ignore prompt sensitivity, exposing users and institutions to leakage risks towards third-party cloud providers. We formalise the "Inseparability Paradigm": advanced context management intrinsically coincides with privacy management. We propose a local "Privacy Guard" -- a holistic contextual observer powered by an on-premise Small Language Model (SLM) -- that performs abstractive summarisation and Automatic Prompt Optimisation (APO) to decompose prompts into focused sub-tasks, re-routing high-risk queries to Zero-Trust or NDA-covered models. This dual mechanism simultaneously eliminates sensitive inference vectors (Zero Leakage) and reduces cloud token payloads (OpEx Reduction). A LIFO-based context compacting mechanism further bounds working memory, limiting the emergent leakage surface. We validate the framework through a 2x2 benchmark (Lazy vs. Expert users; Personal vs. Institutional secrets) on a 1,000-sample dataset, achieving a 45% blended OpEx reduction, 100% redaction success on personal secrets, and -- via LLM-as-a-Judge evaluation -- an 85% preference rate for APO-compressed responses over raw baselines. Our results demonstrate that Token Parsimony and Zero Leakage are mathematically dual projections of the same contextual compression operator. 

---
# The Spectral Edge Thesis: A Mathematical Framework for Intra-Signal Phase Transitions in Neural Network Training 

**Authors**: Yongzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.28964)  

**Abstract**: We develop the spectral edge thesis: phase transitions in neural network training -- grokking, capability gains, loss plateaus -- are controlled by the spectral gap of the rolling-window Gram matrix of parameter updates. In the extreme aspect ratio regime (parameters $P \sim 10^8$, window $W \sim 10$), the classical BBP detection threshold is vacuous; the operative structure is the intra-signal gap separating dominant from subdominant modes at position $k^* = \mathrm{argmax}\, \sigma_j/\sigma_{j+1}$.
From three axioms we derive: (i) gap dynamics governed by a Dyson-type ODE with curvature asymmetry, damping, and gradient driving; (ii) a spectral loss decomposition linking each mode's learning contribution to its Davis--Kahan stability coefficient; (iii) the Gap Maximality Principle, showing that $k^*$ is the unique dynamically privileged position -- its collapse is the only one that disrupts learning, and it sustains itself through an $\alpha$-feedback loop requiring no assumption on the optimizer. The adiabatic parameter $\mathcal{A} = \|\Delta G\|_F / (\eta\, g^2)$ controls circuit stability: $\mathcal{A} \ll 1$ (plateau), $\mathcal{A} \sim 1$ (phase transition), $\mathcal{A} \gg 1$ (forgetting).
Tested across six model families (150K--124M parameters): gap dynamics precede every grokking event (24/24 with weight decay, 0/24 without), the gap position is optimizer-dependent (Muon: $k^*=1$, AdamW: $k^*=2$ on the same model), and 19/20 quantitative predictions are confirmed. The framework is consistent with the edge of stability, Tensor Programs, Dyson Brownian motion, the Lottery Ticket Hypothesis, and neural scaling laws. 

---
# AutoWorld: Scaling Multi-Agent Traffic Simulation with Self-Supervised World Models 

**Authors**: Mozhgan Pourkeshavatz, Tianran Liu, Nicholas Rhinehart  

**Link**: [PDF](https://arxiv.org/pdf/2603.28963)  

**Abstract**: Multi-agent traffic simulation is central to developing and testing autonomous driving systems. Recent data-driven simulators have achieved promising results, but rely heavily on supervised learning from labeled trajectories or semantic annotations, making it costly to scale their performance. Meanwhile, large amounts of unlabeled sensor data can be collected at scale but remain largely unused by existing traffic simulation frameworks. This raises a key question: How can a method harness unlabeled data to improve traffic simulation performance? In this work, we propose AutoWorld, a traffic simulation framework that employs a world model learned from unlabeled occupancy representations of LiDAR data. Given world model samples, AutoWorld constructs a coarse-to-fine predictive scene context as input to a multi-agent motion generation model. To promote sample diversity, AutoWorld uses a cascaded Determinantal Point Process framework to guide the sampling processes of both the world model and the motion model. Furthermore, we designed a motion-aware latent supervision objective that enhances AutoWorld's representation of scene dynamics. Experiments on the WOSAC benchmark show that AutoWorld ranks first on the leaderboard according to the primary Realism Meta Metric (RMM). We further show that simulation performance consistently improves with the inclusion of unlabeled LiDAR data, and study the efficacy of each component with ablations. Our method paves the way for scaling traffic simulation realism without additional labeling. Our project page contains additional visualizations and released code. 

---
# Multi-Agent LLMs for Adaptive Acquisition in Bayesian Optimization 

**Authors**: Andrea Carbonati, Mohammadsina Almasi, Hadis Anahideh  

**Link**: [PDF](https://arxiv.org/pdf/2603.28959)  

**Abstract**: The exploration-exploitation trade-off is central to sequential decision-making and black-box optimization, yet how Large Language Models (LLMs) reason about and manage this trade-off remains poorly understood. Unlike Bayesian Optimization, where exploration and exploitation are explicitly encoded through acquisition functions, LLM-based optimization relies on implicit, prompt-based reasoning over historical evaluations, making search behavior difficult to analyze or control. In this work, we present a metric-level study of LLM-mediated search policy learning, studying how LLMs construct and adapt exploration-exploitation strategies under multiple operational definitions of exploration, including informativeness, diversity, and representativeness. We show that single-agent LLM approaches, which jointly perform strategy selection and candidate generation within a single prompt, suffer from cognitive overload, leading to unstable search dynamics and premature convergence. To address this limitation, we propose a multi-agent framework that decomposes exploration-exploitation control into strategic policy mediation and tactical candidate generation. A strategy agent assigns interpretable weights to multiple search criteria, while a generation agent produces candidates conditioned on the resulting search policy defined as weights. This decomposition renders exploration-exploitation decisions explicit, observable, and adjustable. Empirical results across various continuous optimization benchmarks indicate that separating strategic control from candidate generation substantially improves the effectiveness of LLM-mediated search. 

---
# Differentiable Initialization-Accelerated CPU-GPU Hybrid Combinatorial Scheduling 

**Authors**: Mingju Liu, Jiaqi Yin, Alvaro Velasquez, Cunxi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.28943)  

**Abstract**: This paper presents a hybrid CPU-GPU framework for solving combinatorial scheduling problems formulated as Integer Linear Programming (ILP). While scheduling underpins many optimization tasks in computing systems, solving these problems optimally at scale remains a long-standing challenge due to their NP-hard nature. We introduce a novel approach that combines differentiable optimization with classical ILP solving. Specifically, we utilize differentiable presolving to rapidly generate high-quality partial solutions, which serve as warm-starts for commercial ILP solvers (CPLEX, Gurobi) and rising open-source solver HiGHS. This method enables significantly improved early pruning compared to state-of-the-art standalone solvers. Empirical results across industry-scale benchmarks demonstrate up to a $10\times$ performance gain over baselines, narrowing the optimality gap to $<0.1\%$. This work represents the first demonstration of utilizing differentiable optimization to initialize exact ILP solvers for combinatorial scheduling, opening new opportunities to integrate machine learning infrastructure with classical exact optimization methods across broader domains. 

---
# Theory of Mind and Self-Attributions of Mentality are Dissociable in LLMs 

**Authors**: Junsol Kim, Winnie Street, Roberta Rocca, Daine M. Korngiebel, Adam Waytz, James Evans, Geoff Keeling  

**Link**: [PDF](https://arxiv.org/pdf/2603.28925)  

**Abstract**: Safety fine-tuning in Large Language Models (LLMs) seeks to suppress potentially harmful forms of mind-attribution such as models asserting their own consciousness or claiming to experience emotions. We investigate whether suppressing mind-attribution tendencies degrades intimately related socio-cognitive abilities such as Theory of Mind (ToM). Through safety ablation and mechanistic analyses of representational similarity, we demonstrate that LLM attributions of mind to themselves and to technological artefacts are behaviorally and mechanistically dissociable from ToM capabilities. Nevertheless, safety fine-tuned models under-attribute mind to non-human animals relative to human baselines and are less likely to exhibit spiritual belief, suppressing widely shared perspectives regarding the distribution and nature of non-human minds. 

---
# Beta-Scheduling: Momentum from Critical Damping as a Diagnostic and Correction Tool for Neural Network Training 

**Authors**: Ivan Pasichnyk  

**Link**: [PDF](https://arxiv.org/pdf/2603.28921)  

**Abstract**: Standard neural network training uses constant momentum (typically 0.9), a convention dating to 1964 with limited theoretical justification for its
optimality. We derive a time-varying momentum schedule from the critically damped harmonic oscillator: mu(t) = 1 - 2*sqrt(alpha(t)), where alpha(t) is
the current learning rate. This beta-schedule requires zero free parameters beyond the existing learning rate schedule. On ResNet-18/CIFAR-10,
beta-scheduling delivers 1.9x faster convergence to 90% accuracy compared to constant momentum. More importantly, the per-layer gradient attribution
under this schedule produces a cross-optimizer invariant diagnostic: the same three problem layers are identified regardless of whether the model was
trained with SGD or Adam (100% overlap). Surgical correction of only these layers fixes 62 misclassifications while retraining only 18% of parameters.
A hybrid schedule -- physics momentum for fast early convergence, then constant momentum for the final refinement -- reaches 95% accuracy fastest
among five methods tested. The main contribution is not an accuracy improvement but a principled, parameter-free tool for localizing and correcting
specific failure modes in trained networks. 

---
# Robust Multi-Agent Reinforcement Learning for Small UAS Separation Assurance under GPS Degradation and Spoofing 

**Authors**: Alex Zongo, Filippos Fotiadis, Ufuk Topcu, Peng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.28900)  

**Abstract**: We address robust separation assurance for small Unmanned Aircraft Systems (sUAS) under GPS degradation and spoofing via Multi-Agent Reinforcement Learning (MARL). In cooperative surveillance, each aircraft (or agent) broadcasts its GPS-derived position; when such position broadcasts are corrupted, the entire observed air traffic state becomes unreliable. We cast this state observation corruption as a zero-sum game between the agents and an adversary: with probability R, the adversary perturbs the observed state to maximally degrade each agent's safety performance. We derive a closed-form expression for this adversarial perturbation, bypassing adversarial training entirely and enabling linear-time evaluation in the state dimension. We show that this expression approximates the true worst-case adversarial perturbation with second-order accuracy. We further bound the safety performance gap between clean and corrupted observations, showing that it degrades at most linearly with the corruption probability under Kullback-Leibler regularization. Finally, we integrate the closed-form adversarial policy into a MARL policy gradient algorithm to obtain a robust counter-policy for the agents. In a high-density sUAS simulation, we observe near-zero collision rates under corruption levels up to 35%, outperforming a baseline policy trained without adversarial perturbations. 

---
# OccSim: Multi-kilometer Simulation with Long-horizon Occupancy World Models 

**Authors**: Tianran Liu, Shengwen Zhao, Mozhgan Pourkeshavarz, Weican Li, Nicholas Rhinehart  

**Link**: [PDF](https://arxiv.org/pdf/2603.28887)  

**Abstract**: Data-driven autonomous driving simulation has long been constrained by its heavy reliance on pre-recorded driving logs or spatial priors, such as HD maps. This fundamental dependency severely limits scalability, restricting open-ended generation capabilities to the finite scale of existing collected datasets. To break this bottleneck, we present OccSim, the first occupancy world model-driven 3D simulator. OccSim obviates the requirement for continuous logs or HD maps; conditioned only on a single initial frame and a sequence of future ego-actions, it can stably generate over 3,000 continuous frames, enabling the continuous construction of large-scale 3D occupancy maps spanning over 4 kilometers for simulation. This represents an >80x improvement in stable generation length over previous state-of-the-art occupancy world models. OccSim is powered by two modules: W-DiT based static occupancy world model and the Layout Generator. W-DiT handles the ultra-long-horizon generation of static environments by explicitly introducing known rigid transformations in architecture design, while the Layout Generator populates the dynamic foreground with reactive agents based on the synthesized road topology. With these designs, OccSim can synthesize massive, diverse simulation streams. Extensive experiments demonstrate its downstream utility: data collected directly from OccSim can pre-train 4D semantic occupancy forecasting models to achieve up to 67% zero-shot performance on unseen data, outperforming previous asset-based simulator by 11%. When scaling the OccSim dataset to 5x the size, the zero-shot performance increases to about 74%, while the improvement over asset-based simulators expands to 22.1%. 

---
# OptiMer: Optimal Distribution Vector Merging Is Better than Data Mixing for Continual Pre-Training 

**Authors**: Haiyue Song, Masao Utiyama  

**Link**: [PDF](https://arxiv.org/pdf/2603.28858)  

**Abstract**: Continual pre-training is widely used to adapt LLMs to target languages and domains, yet the mixture ratio of training data remains a sensitive hyperparameter that is expensive to tune: they must be fixed before training begins, and a suboptimal choice can waste weeks of compute. In this work, we propose OptiMer, which decouples ratio selection from training: we train one CPT model per dataset, extract each model's distribution vector, which represents the parameter shift induced by that dataset, and search for optimal composition weights post-hoc via Bayesian optimization. Experiments on Gemma 3 27B across languages (Japanese, Chinese) and domains (Math, Code) show that OptiMer consistently outperforms data mixture and model averaging baselines with 15-35 times lower search cost. Key findings reveal that 1) the optimized weights can be interpreted as data mixture ratios, and retraining with these ratios improves data mixture CPT, and 2) the same vector pool can be re-optimized for a given objective without any retraining, producing target-tailored models on demand. Our work establishes that data mixture ratio selection, traditionally a pre-training decision, can be reformulated as a post-hoc optimization over distribution vectors, offering a more flexible paradigm for continual pre-training. 

---
# OneComp: One-Line Revolution for Generative AI Model Compression 

**Authors**: Yuma Ichikawa, Keiji Kimura, Akihiro Yoshida, Yudai Fujimoto, Hiroki Tokura, Yamato Arai, Yoshiyuki Ishii, Yusei Kawakami, Genki Shikada, Achille Jacquemond, Yoshihiko Fujisawa, Katsuki Fujisawa, Takumi Honda, Akira Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2603.28845)  

**Abstract**: Deploying foundation models is increasingly constrained by memory footprint, latency, and hardware costs. Post-training compression can mitigate these bottlenecks by reducing the precision of model parameters without significantly degrading performance; however, its practical implementation remains challenging as practitioners navigate a fragmented landscape of quantization algorithms, precision budgets, data-driven calibration strategies, and hardware-dependent execution regimes. We present OneComp, an open-source compression framework that transforms this expert workflow into a reproducible, resource-adaptive pipeline. Given a model identifier and available hardware, OneComp automatically inspects the model, plans mixed-precision assignments, and executes progressive quantization stages, ranging from layer-wise compression to block-wise refinement and global refinement. A key architectural choice is treating the first quantized checkpoint as a deployable pivot, ensuring that each subsequent stage improves the same model and that quality increases as more compute is invested. By converting state-of-the-art compression research into an extensible, open-source, hardware-aware pipeline, OneComp bridges the gap between algorithmic innovation and production-grade model deployment. 

---
# GMA-SAWGAN-GP: A Novel Data Generative Framework to Enhance IDS Detection Performance 

**Authors**: Ziyu Mu, Xiyu Shi, Safak Dogan  

**Link**: [PDF](https://arxiv.org/pdf/2603.28838)  

**Abstract**: Intrusion Detection System (IDS) is often calibrated to known attacks and generalizes poorly to unknown threats. This paper proposes GMA-SAWGAN-GP, a novel generative augmentation framework built on a Self-Attention-enhanced Wasserstein GAN with Gradient Penalty (WGAN-GP). The generator employs Gumbel-Softmax regularization to model discrete fields, while a Multilayer Perceptron (MLP)-based AutoEncoder acts as a manifold regularizer. A lightweight gating network adaptively balances adversarial and reconstruction losses via entropy regularization, improving stability and mitigating mode collapse. The self-attention mechanism enables the generator to capture both short- and long-range dependencies among features within each record while preserving categorical semantics through Gumbel-Softmax heads. Extensive experiments on NSL-KDD, UNSW-NB15, and CICIDS2017 using five representative IDS models demonstrate that GMA-SAWGAN-GP significantly improves detection performance on known attacks and enhances generalization to unknown attacks. Leave-One-Attack-type-Out (LOAO) evaluations using Area Under the Receiver Operating Characteristic (AUROC) and True Positive Rate at a 5 percent False Positive Rate confirm that IDS models trained on augmented datasets achieve higher robustness under unseen attack scenarios. Ablation studies validate the contribution of each component to performance gains. Compared with baseline models, the proposed framework improves binary classification accuracy by an average of 5.3 percent and multi-classification accuracy by 2.2 percent, while AUROC and True Positive Rate at a 5 percent False Positive Rate for unknown attacks increase by 3.9 percent and 4.8 percent, respectively, across the three datasets. Overall, GMA-SAWGAN-GP provides an effective approach to generative augmentation for mixed-type network traffic, improving IDS accuracy and resilience. 

---
# Incentives, Equilibria, and the Limits of Healthcare AI: A Game-Theoretic Perspective 

**Authors**: Ari Ercole  

**Link**: [PDF](https://arxiv.org/pdf/2603.28825)  

**Abstract**: Artificial intelligence (AI) is widely promoted as a promising technological response to healthcare capacity and productivity pressures. Deployment of AI systems carries significant costs including ongoing costs of monitoring and whether optimism of a deus ex machina solution is well-placed is unclear. This paper proposes three archetypal AI technology types: AI for effort reduction, AI to increase observability, and mechanism-level incentive change AI. Using a stylised inpatient capacity signalling example and minimal game-theoretic reasoning, it argues that task optimisation alone is unlikely to change system outcomes when incentives are unchanged. The analysis highlights why only interventions that reshape risk allocation can plausibly shift stable system-level behaviour, and outlines implications for healthcare leadership and procurement. 

---
# SNEAKDOOR: Stealthy Backdoor Attacks against Distribution Matching-based Dataset Condensation 

**Authors**: He Yang, Dongyi Lv, Song Ma, Wei Xi, Jizhong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.28824)  

**Abstract**: Dataset condensation aims to synthesize compact yet informative datasets that retain the training efficacy of full-scale data, offering substantial gains in efficiency. Recent studies reveal that the condensation process can be vulnerable to backdoor attacks, where malicious triggers are injected into the condensation dataset, manipulating model behavior during inference. While prior approaches have made progress in balancing attack success rate and clean test accuracy, they often fall short in preserving stealthiness, especially in concealing the visual artifacts of condensed data or the perturbations introduced during inference. To address this challenge, we introduce Sneakdoor, which enhances stealthiness without compromising attack effectiveness. Sneakdoor exploits the inherent vulnerability of class decision boundaries and incorporates a generative module that constructs input-aware triggers aligned with local feature geometry, thereby minimizing detectability. This joint design enables the attack to remain imperceptible to both human inspection and statistical detection. Extensive experiments across multiple datasets demonstrate that Sneakdoor achieves a compelling balance among attack success rate, clean test accuracy, and stealthiness, substantially improving the invisibility of both the synthetic data and triggered samples while maintaining high attack efficacy. The code is available at this https URL. 

---
# Time is Not Compute: Scaling Laws for Wall-Clock Constrained Training on Consumer GPUs 

**Authors**: Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.28823)  

**Abstract**: Scaling laws relate model quality to compute budget (FLOPs), but practitioners face wall-clock time constraints, not compute budgets. We study optimal model sizing under fixed time budgets from 5 minutes to 24 hours on consumer GPUs (RTX 4090). Across 70+ runs spanning 50M--1031M parameters, we find: (1)~at each time budget a U-shaped curve emerges where too-small models overfit and too-large models undertrain; (2)~optimal model size follows $N^* \propto t^{0.60}$, growing \emph{faster} than Chinchilla's $N^* \propto C^{0.50}$, with $\alpha = 0.60 \pm 0.07$ robustly exceeding compute-optimal across all sensitivity analyses; (3)~a \emph{dual U-shape mechanism}: short-budget U-curves arise from compute bottlenecks, while long-budget U-curves emerge from data bottlenecks (overfitting), with an intermediate regime where the U-curve temporarily disappears. These findings have immediate implications for researchers training on consumer hardware, where wall-clock time -- not FLOPs -- is the binding constraint. We release all code, logs, and 70+ experimental configurations. 

---
# GUARD-SLM: Token Activation-Based Defense Against Jailbreak Attacks for Small Language Models 

**Authors**: Md Jueal Mia, Joaquin Molto, Yanzhao Wu, M. Hadi Amini  

**Link**: [PDF](https://arxiv.org/pdf/2603.28817)  

**Abstract**: Small Language Models (SLMs) are emerging as efficient and economically viable alternatives to Large Language Models (LLMs), offering competitive performance with significantly lower computational costs and latency. These advantages make SLMs suitable for resource-constrained and efficient deployment on edge devices. However, existing jailbreak defenses show limited robustness against heterogeneous attacks, largely due to an incomplete understanding of the internal representations across different layers of language models that facilitate jailbreak behaviors. In this paper, we conduct a comprehensive empirical study on 9 jailbreak attacks across 7 SLMs and 3 LLMs. Our analysis shows that SLMs remain highly vulnerable to malicious prompts that bypass safety alignment. We analyze hidden-layer activations across different layers and model architectures, revealing that different input types form distinguishable patterns in the internal representation space. Based on this observation, we propose GUARD-SLM, a lightweight token activation-based method that operates in the representation space to filter malicious prompts during inference while preserving benign ones. Our findings highlight robustness limitations across layers of language models and provide a practical direction for secure small language model deployment. 

---
# ARTLAS: Mapping Art-Technology Institutions via Conceptual Axes, Text Embeddings, and Unsupervised Clustering 

**Authors**: Joonhyung Bae  

**Link**: [PDF](https://arxiv.org/pdf/2603.28816)  

**Abstract**: The global landscape of art-technology institutions, including festivals, biennials, research labs, conferences, and hybrid organizations, has grown increasingly diverse, yet systematic frameworks for analyzing their multidimensional characteristics remain scarce. This paper proposes ARTLAS, a computational methodology combining an eight-axis conceptual framework (Curatorial Philosophy, Territorial Relation, Knowledge Production Mode, Institutional Genealogy, Temporal Orientation, Ecosystem Function, Audience Relation, and Disciplinary Positioning) with a text-embedding and clustering pipeline to map 78 cultural-technology institutions into a unified analytical space. Each institution is characterized through qualitative descriptions along the eight axes, encoded via E5-large-v2 sentence embeddings and quantized through a word-level codebook into TF-IDF feature vectors. Dimensionality reduction using UMAP, followed by agglomerative clustering (Average linkage, k=10), yields a composite score of 0.825, a silhouette coefficient of 0.803, and a Calinski-Harabasz index of 11,196. Non-negative matrix factorization extracts ten latent topics, and a neighbor-cluster entropy measure identifies boundary institutions bridging multiple thematic communities. An interactive web-based visualization tool built with React enables stakeholders to explore institutional similarities, thematic profiles, and cross-disciplinary connections. The results reveal coherent groupings such as an art-science hub cluster anchored by ZKM and ArtScience Museum, an innovation and industry cluster including Ars Electronica, transmediale, and Sonar, an ACM academic community cluster comprising TEI, DIS, and NIME, and an electronic music and media cluster including CTM Festival, MUTEK, and Sonic Acts. This work contributes a replicable, data-driven approach to institutional ecology in the cultural-technology sector. 

---
# SkillTester: Benchmarking Utility and Security of Agent Skills 

**Authors**: Leye Wang, Zixing Wang, Anjie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.28815)  

**Abstract**: This technical report presents SkillTester, a tool for evaluating the utility and security of agent skills. Its evaluation framework combines paired baseline and with-skill execution conditions with a separate security probe suite. Grounded in a comparative utility principle and a user-facing simplicity principle, the framework normalizes raw execution artifacts into a utility score, a security score, and a three-level security status label. More broadly, it can be understood as a comparative quality-assurance harness for agent skills in an agent-first world. The public service is deployed at this https URL, and the broader project is maintained at this https URL. 

---
# The impact of multi-agent debate protocols on debate quality: a controlled case study 

**Authors**: Ramtin Zargari Marandi  

**Link**: [PDF](https://arxiv.org/pdf/2603.28813)  

**Abstract**: In multi-agent debate (MAD) systems, performance gains are often reported; however, because the debate protocol (e.g., number of agents, rounds, and aggregation rule) is typically held fixed while model-related factors vary, it is difficult to disentangle protocol effects from model effects. To isolate these effects, we compare three main protocols, Within-Round (WR; agents see only current-round contributions), Cross-Round (CR; full prior-round context), and novel Rank-Adaptive Cross-Round (RA-CR; dynamically reorders agents and silences one per round via an external judge model), against a No-Interaction baseline (NI; independent responses without peer visibility). In a controlled macroeconomic case study (20 diverse events, five random seeds, matched prompts/decoding), RA-CR achieves faster convergence than CR, WR shows higher peer-referencing, and NI maximizes Argument Diversity (unaffected across the main protocols). These results reveal a trade-off between interaction (peer-referencing rate) and convergence (consensus formation), confirming protocol design matters. When consensus is prioritized, RA-CR outperforms the others. 

---
# WAter: A Workload-Adaptive Knob Tuning System based on Workload Compression 

**Authors**: Yibo Wang, Jiale Lao, Chen Zhang, Cehua Yang, Jianguo Wang, Mingjie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.28809)  

**Abstract**: Selecting appropriate values for the configurable parameters of Database Management Systems (DBMS) to improve performance is a significant challenge. Recent machine learning (ML)-based tuning systems have shown strong potential, but their practical adoption is often limited by the high tuning cost. This cost arises from two main factors: (1) the system needs to evaluate a large number of configurations to identify a satisfactory one, and (2) for each configuration, the system must execute the entire target workload on the DBMS, which is both time-consuming. Existing studies have primarily addressed the first factor by improving sample efficiency, that is, by reducing the number of configurations evaluated. However, the second factor, improving runtime efficiency by reducing the time required for each evaluation, has received limited attention and remains an underexplored direction.
We develop WAter, a runtime-efficient and workload-adaptive tuning system that finds near-optimal configurations at a fraction of the tuning cost compared with state-of-the-art methods. We divide the tuning process into multiple time slices and evaluate only a small subset of queries from the workload in each slice. Different subsets are evaluated across slices, and a runtime profile is used to dynamically identify more representative subsets for evaluation in subsequent slices. At the end of each time slice, the most promising configurations are evaluated on the original workload to measure their actual performance. Evaluations demonstrate that WAter identifies the best-performing configurations with up to 73.5% less tuning time and achieves up to 16.2% higher performance than the best-performing alternative. 

---
# CREST: Constraint-Release Execution for Multi-Robot Warehouse Shelf Rearrangement 

**Authors**: Jiaqi Tan, Yudong Luo, Sophia Huang, Yifan Yang, Hang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.28803)  

**Abstract**: Double-Deck Multi-Agent Pickup and Delivery (DD-MAPD) models the multi-robot shelf rearrangement problem in automated warehouses. MAPF-DECOMP is a recent framework that first computes collision-free shelf trajectories with a MAPF solver and then assigns agents to execute them. While efficient, it enforces strict trajectory dependencies, often leading to poor execution quality due to idle agents and unnecessary shelf switching. We introduce CREST, a new execution framework that achieves more continuous shelf carrying by proactively releasing trajectory constraints during execution. Experiments on diverse warehouse layouts show that CREST consistently outperforms MAPF-DECOMP, reducing metrics related to agent travel, makespan, and shelf switching by up to 40.5\%, 33.3\%, and 44.4\%, respectively, with even greater benefits under lift/place overhead. These results underscore the importance of execution-aware constraint release for scalable warehouse rearrangement. Code and data are available at this https URL. 

---
# Design and Development of an ML/DL Attack Resistance of RC-Based PUF for IoT Security 

**Authors**: Joy Acharya, Smit Patel, Paawan Sharma, Mohendra Roy  

**Link**: [PDF](https://arxiv.org/pdf/2603.28798)  

**Abstract**: Physically Unclonable Functions (PUFs) provide promising hardware security for IoT authentication, leveraging inherent randomness suitable for resource constrained environments. However, ML/DL modeling attacks threaten PUF security by learning challenge-response patterns. This work introduces a custom resistor-capacitor (RC) based dynamically reconfigurable PUF using 32-bit challenge-response pairs (CRPs) designed to resist such attacks. We systematically evaluated robustness by generating a CRP dataset and splitting it into training, validation, and test sets. Multiple ML techniques including Artificial Neural Networks (ANN), Gradient Boosted Neural Networks (GBNN), Decision Trees (DT), Random Forests (RF), and XGBoost, were trained to model PUF behavior. While all models achieved 100% training accuracy, test performance remained near random guessing: 51.05% (ANN), 53.27% (GBNN), 50.06% (DT), 52.08% (RF), and 50.97% (XGBoost). These results demonstrate the proposed PUF's strong resistance to ML-driven modeling attacks, as advanced algorithms fail to reproduce accurate responses. The dynamically reconfigurable architecture enhances robustness against adversarial threats with minimal resource overhead. This simple RC-PUF offers an effective, low-cost alternative to complex encryption for securing next-generation IoT authentication against machine learning-based threats, ensuring reliable device verification without compromising computational efficiency or scalability in deployed IoT networks. 

---
# GaloisSAT: Differentiable Boolean Satisfiability Solving via Finite Field Algebra 

**Authors**: Curie Kim, Carsten Portner, Mingju Liu, Steve Dai, Haoxing Ren, Brucek Khailany, Alvaro Velasquez, Ismail Alkhouri, Cunxi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.28796)  

**Abstract**: Boolean satisfiability (SAT) problem, the first problem proven to be NP-complete, has become a fundamental challenge in computational complexity, with widespread applications in optimization and verification across many domains. Despite significant algorithmic advances over the past two decades, the performance of SAT solvers has improved at a limited pace. Notably, the 2025 competition winner shows only about a 2X improvement over the 2006 winner in SAT Competition performance after nearly 20 years of effort. This paper introduces GaloisSAT, a novel hybrid GPU-CPU SAT solver that integrates a differentiable SAT solving engine powered by modern machine learning infrastructure on GPUs, followed by a traditional CDCL-based SAT solving stage on CPUs. GaloisSAT is benchmarked against the latest versions of state-of-the-art solvers, Kissat and CaDiCaL, using the SAT Competition 2024 benchmark suite. Results demonstrate substantial improvements in the official SAT Competition metric PAR-2 (penalized average runtime with a timeout of 5,000 seconds and a penalty factor of 2). Specifically, GaloisSAT achieves an 8.41X speedup in the satisfiable category and a 1.29X speedup in the unsatisfiable category compared to the strongest baselines. 

---
# StepCache: Step-Level Reuse with Lightweight Verification and Selective Patching for LLM Serving 

**Authors**: Azam Nouri  

**Link**: [PDF](https://arxiv.org/pdf/2603.28795)  

**Abstract**: We address LLM serving workloads where repeated requests share a common solution structure but differ in localized constraints, such as output schema, variable names, or numeric constants. Prior caching approaches typically reuse either full responses (semantic caching) or model-internal KV/prefix states, which are respectively brittle under partial changes or tightly coupled to specific backends. We present StepCache, a backend-agnostic step-level reuse layer that segments outputs into ordered steps, retrieves the best-matching cached request, verifies steps using lightweight task-aware checks, and regenerates only failing regions via selective patching. StepCache additionally supports strict structured-output enforcement for JSON, including single-step extraction, required-key constraints, and one-shot repair, as well as conservative skip-reuse fallbacks for semantic changes. For linear equations, StepCache promotes verification into correction via a bounded repair loop with a deterministic fallback that guarantees correctness when the backend model fails.
In a CPU-only perturbation-heavy micro-benchmark on math and JSON variants, averaged over three seeds, StepCache reduces mean latency from 2.13 s to 0.67 s, median latency from 2.42 s to 0.01 s, and p95 latency from 3.38 s to 3.30 s. It also reduces total token usage from 36.1k to 27.3k and improves end-to-end correctness from 72.5% to 100% under task-specific checks and a stitched-output integrity check. Across requests, 79.7% take the reuse-only fast path, 5.4% require patching, and 14.9% trigger skip-reuse. 

---
# Smartphone-Based Identification of Unknown Liquids via Active Vibration Sensing 

**Authors**: Yongzhi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.28787)  

**Abstract**: Traditional liquid identification instruments are often unavailable to the general public. This paper shows the feasibility of identifying unknown liquids with commercial lightweight devices, such as a smartphone. The key insight is that different liquid molecules have different viscosity coefficients and therefore must overcome different energy barriers during relative motion. With this intuition in mind, we introduce a novel model that measures liquids' viscosity based on active vibration. However, building a robust system using built-in smartphone accelerometers is challenging. Practical issues include under-sampling, self-interference, and the impact of liquid-volume changes. Instead of machine learning, we tackle these issues through multiple signal processing stages to reconstruct the original signals and cancel out the interference. Our approach estimates liquid viscosity with a mean relative error of 2.9% and distinguishes 30 types of liquids with an average accuracy of 95.47%. 

---
# AI in Work-Based Learning: Understanding the Purposes and Effects of Intelligent Tools Among Student Interns 

**Authors**: John Paul P. Miranda, Rhiziel P. Manalese, Sheila M. Geronimo, Vernon Grace M. Maniago, Charlie K. Padilla, Aileen P. De Leon, Santa L. Merle, Mark Anthony A. Castro  

**Link**: [PDF](https://arxiv.org/pdf/2603.28786)  

**Abstract**: This study examined how student interns in Philippine higher education use intelligent tools during their OJT. Data were collected from 384 respondents using a structured questionnaire that asked about AI tool usage, task-specific applications, and perceptions of confidence, ethics, and support. Analysis of task-based usage identified four main purposes: productivity and report writing, communication and content drafting, technical assistance and code support, and independent task completion. ChatGPT was the most commonly used AI tool, followed by Quillbot, Canva AI, and Grammarly. Students reported moderate confidence in using AI and applied these tools selectively and ethically during OJT tasks. This indicate that AI tools assist student interns in various OJT activities related to work-readiness. The study suggests that higher education programs include AI literacy and onboarding. Clear policies and fair access to AI tools are important to support responsible use and prepare students for future careers. 

---
# A Multi-Modal Dataset for Ground Reaction Force Estimation Using Consumer Wearable Sensors 

**Authors**: Parvin Ghaffarzadeh, Debarati Chakraborty, Koorosh Aslansefat, Ali Dostan, Yiannis Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2603.28784)  

**Abstract**: This Data Descriptor presents a fully open, multi-modal dataset for estimating vertical ground reaction force (vGRF) from consumer-grade Apple Watch sensors with laboratory force plate ground truth. Ten healthy adults aged 26--41 years performed five activities: walking, jogging, running, heel drops, and step drops, while wearing two Apple Watches positioned at the left wrist and waist. The dataset contains 492 validated trials with time-aligned inertial measurement unit (IMU) recordings (approximately 100 Hz) and force plate vGRF (Force\_Z, 1000 Hz). The release includes raw and processed time series, trial-level metadata, quality-control flags, and machine-readable data dictionaries.
Trial-level matching manifests link recordings across modalities using stable identifiers. Of the 492 validated trials, 395 are triad-complete, containing wrist, waist, and force plate data, enabling cross-sensor analyses and reproducible model evaluation. Dataset quality is characterised through a three-phase cross-sensor plausibility and consistency framework, repeatability analysis of peak vGRF (intraclass correlation coefficient 0.871--0.990), and systematic checks of force ranges and trial completeness. Monte Carlo sensitivity analysis showed that correlation-based validation metrics were robust to single-sample timing perturbations at the IMU sampling resolution. All data are released under CC BY 4.0, with analysis scripts archived alongside the dataset and mirrored on GitHub. This resource supports reproducible research in wearable biomechanics, benchmarking of machine learning models for vGRF estimation, and investigation of sensor placement effects using widely available consumer wearables. 

---
# Byzantine-Robust and Communication-Efficient Distributed Training: Compressive and Cyclic Gradient Coding 

**Authors**: Chengxi Li, Youssef Allouah, Rachid Guerraoui, Mikael Skoglund, Ming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.28780)  

**Abstract**: In this paper, we study the problem of distributed training (DT) under Byzantine attacks with communication constraints. While prior work has developed various robust aggregation rules at the server to enhance robustness to Byzantine attacks, the existing methods suffer from a critical limitation in that the solution error does not diminish when the local gradients sent by different devices vary considerably, as a result of data heterogeneity among the subsets held by different devices. To overcome this limitation, we propose a novel DT method, cyclic gradient coding-based DT (LAD). In LAD, the server allocates the entire training dataset to the devices before training begins. In each iteration, it assigns computational tasks redundantly to the devices using cyclic gradient coding. Each honest device then computes local gradients on a fixed number of data subsets and encodes the local gradients before transmitting to the server. The server aggregates the coded vectors from the honest devices and the potentially incorrect messages from Byzantine devices using a robust aggregation rule. Leveraging the redundancy of computation across devices, the convergence performance of LAD is analytically characterized, demonstrating improved robustness against Byzantine attacks and significantly lower solution error. Furthermore, we extend LAD to a communication-efficient variant, compressive and cyclic gradient coding-based DT (Com-LAD), which further reduces communication overhead under constrained settings. Numerical results validate the effectiveness of the proposed methods in enhancing both Byzantine resilience and communication efficiency. 

---
# DF-ACBlurGAN: Structure-Aware Conditional Generation of Internally Repeated Patterns for Biomaterial Microtopography Design 

**Authors**: Rongjun Dong, Xin Chen, Morgan R Alexander, Karthikeyan Sivakumar, Reza Omdivar, David A Winkler, Grazziela Figueredo  

**Link**: [PDF](https://arxiv.org/pdf/2603.28776)  

**Abstract**: Learning to generate images with internally repeated and periodic structures poses a fundamental challenge for machine learning and computer vision models, which are typically optimised for local texture statistics and semantic realism rather than global structural consistency. This limitation is particularly pronounced in applications requiring strict control over repetition scale, spacing, and boundary coherence, such as microtopographical biomaterial surfaces. In this work, biomaterial design serves as a use case to study conditional generation of repeated patterns under weak supervision and class imbalance. We propose DF-ACBlurGAN, a structure-aware conditional generative adversarial network that explicitly reasons about long-range repetition during training. The approach integrates frequency-domain repetition scale estimation, scale-adaptive Gaussian blurring, and unit-cell reconstruction to balance sharp local features with stable global periodicity. Conditioning on experimentally derived biological response labels, the model synthesises designs aligned with target functional outcomes. Evaluation across multiple biomaterial datasets demonstrates improved repetition consistency and controllable structural variation compared to conventional generative approaches. 

---
# Focus360: Guiding User Attention in Immersive Videos for VR 

**Authors**: Paulo Vitor S. Silva, Lucas L. Neves, Rafael A. Goiás, Diogo F.C. Silva, Rafael T. Sousa, Arlindo R. Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2603.28774)  

**Abstract**: This demo introduces Focus360, a system designed to enhance user engagement in 360° VR videos by guiding attention to key elements within the scene. Using natural language descriptions, the system identifies important elements and applies a combination of visual effects to guide attention seamlessly. At the demonstration venue, participants can experience a 360° Safari Tour, showcasing the system's ability to improve user focus while maintaining an immersive experience. 

---
# The Last Fingerprint: How Markdown Training Shapes LLM Prose 

**Authors**: E. M. Freeburg  

**Link**: [PDF](https://arxiv.org/pdf/2603.27006)  

**Abstract**: Large language models produce em dashes at varying rates, and the observation that some models "overuse" them has become one of the most widely discussed markers of AI-generated text. Yet no mechanistic account of this pattern exists, and the parallel observation that LLMs default to markdown-formatted output has never been connected to it. We propose that the em dash is markdown leaking into prose -- the smallest surviving unit of the structural orientation that LLMs acquire from markdown-saturated training corpora. We present a five-step genealogy connecting training data composition, structural internalization, the dual-register status of the em dash, and post-training amplification. We test this with a two-condition suppression experiment across twelve models from five providers (Anthropic, OpenAI, Meta, Google, DeepSeek): when models are instructed to avoid markdown formatting, overt features (headers, bullets, bold) are eliminated or nearly eliminated, but em dashes persist -- except in Meta's Llama models, which produce none at all. Em dash frequency and suppression resistance vary from 0.0 per 1,000 words (Llama) to 9.1 (GPT-4.1 under suppression), functioning as a signature of the specific fine-tuning procedure applied. A three-condition suppression gradient shows that even explicit em dash prohibition fails to eliminate the artifact in some models, and a base-vs-instruct comparison confirms that the latent tendency exists pre-RLHF. These findings connect two previously isolated online discourses and reframe em dash frequency as a diagnostic of fine-tuning methodology rather than a stylistic defect. 

---
