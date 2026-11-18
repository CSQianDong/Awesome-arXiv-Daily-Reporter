# Agent Trust Infrastructure (ATI): A Federated Identity, Trust, Attestation, and Accountability Architecture for Cross-Organization Autonomous AI Agents 

**Authors**: Viet K. Nguyen, Mohammad I. Husain  

**Link**: [PDF](https://arxiv.org/pdf/2611.23202)  

**Abstract**: AI agents increasingly collaborate across organizational boundaries, invoking tools and interacting with agents operated by independent entities. Although existing agent frameworks support planning and coordination, they lack a common mechanism for verifying cross-organizational agent identity, execution integrity, and authorization. We propose the Agent Trust Infrastructure (ATI), a framework-independent security architecture built on Public Key Infrastructure (PKI), Zero Trust Architecture (ZTA), and remote attestation. ATI establishes federated trust through peer organizational certificate authorities (AgentCAs), eliminating the need for a shared root authority. Before executing requests, ATI validates agent certificates, verifies approved model, tool, and system-prompt configurations, and enforces authorization through three policy checks. We implement ATI as an external security enforcement layer that integrates with existing agent frameworks without modifying their underlying models. We evaluate ATI in a simulated federation of eight organizations using six agent frameworks, each deploying a five-agent team. Security experiments involve two organizations and assess 14 attack scenarios and six benign tasks across five agent models, using five attack variations and three repetitions per variation. A panel of judge models evaluates the experimental outcomes. With all enforcement mechanisms enabled, attack containment increases from 43% to 84% (40% to 80% for the 11 attack scenarios evaluated consistently across all configurations), while enforcement introduces approximately 174 microseconds of overhead per request. ATI is released as an open-source, framework-independent package that uses IETF WIMSE workload token formats for identity and request verification and Entity Attestation Tokens for model, tool, and system-prompt integrity verification. 

---
# WoRD: Diffusion-Based Posterior Inference for Online Goal Recognition 

**Authors**: Almog Anschel, Sarah Keren  

**Link**: [PDF](https://arxiv.org/pdf/2611.23192)  

**Abstract**: Goal recognition is the task of inferring an agent's intended goal from its observed behavior. Most existing work focuses on high-level symbolic recognition, abstracting away the continuous motion through which agents reach their goal. Methods that do reason over continuous trajectories rely on an explicit dynamics model that is hard to acquire and rarely transfers across settings. Neither approach explicitly learns how agents move toward each candidate goal, limiting accuracy on ambiguous trajectory prefixes where such knowledge is most needed.
We present WoRD (Waypoint Recognition with Diffusion), which uses a learned trajectory prior to infer the agent's goal from partial observations. At each step, WoRD runs one warm-started denoising chain per candidate goal, steers each chain toward the observed trajectory through a closed-form log-likelihood gradient, and reweights the resulting samples by Bayes' rule into a calibrated posterior over goals. We evaluate WoRD on simulated robot settings, pedestrian trajectories, and vehicle trajectories using four metrics targeting distinct failure modes. Across all three settings, the same inference procedure preserves calibrated multimodal beliefs through ambiguous prefixes where filter- and classifier-based observers collapse to a single mode or grow overconfident. 

---
# RSI-Prime: Training Models to Train Models 

**Authors**: Dewu Zheng, Yang Ye, Hongyu Zhang, Jun Yu, Yanlin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.23138)  

**Abstract**: Enabling large language models (LLMs) to autonomously conduct training and improve target models is an important foundation for recursive self-improvement (RSI). As LLMs expand their role from being trained to conducting training, their model-training capability warrants dedicated study and systematic development. To this end, we construct RSI-Prime, a dataset for improving model-training capability, comprising 350 complete experimental trajectories across seven task categories. To help LLMs learn to make high-quality experimental decisions from training practice, we perform fine-grained decision identification and quality annotation, then apply decision-level loss masking to exclude supervision on low-quality behavior. Experiments with GLM-5.3 on PostTrainBench, a benchmark for autonomous post-training, show that training on RSI-Prime yields a 31.95% relative improvement in overall performance over the original GLM-5.3, highlighting a path toward future RSI systems in which models autonomously sustain their own advancement. 

---
# Hidden-State Similarity Predicts Re-Elicitation After Inoculation Prompting 

**Authors**: Rose N. Kwon, Samy Mammeri, Sabyasachi Sahoo, Christian Gagné, Thomas Jiralerspong  

**Link**: [PDF](https://arxiv.org/pdf/2611.23096)  

**Abstract**: Fine-tuning on narrow harmful tasks can cause emergent misalignment, where models generalize harmful behavior beyond the training distribution. Inoculation prompting can reduce this effect by explicitly eliciting the undesired behavior during training, but recent work shows that the behavior can reappear when evaluation prompts contain cues from the training context. We study what makes such prompts effective triggers. We find that, surprisingly, textual similarity to the inoculation prompt is an incomplete predictor. Instead, prompts are more likely to re-elicit suppressed behavior when they induce activation states similar to those produced by the inoculation context. These findings advance our understanding of how inoculation prompting modulates conditional misalignment, and suggest that activation space analysis can help identify when suppressed behaviors remain accessible under eval-time prompts. 

---
# Colombo: A Frontier-to-Frontier Autoresearch System for Oracle-Based Optimization 

**Authors**: Haihan Zhang, Wendao Wu, Chenheng Zhang, Chunyuan Zheng, Yanyi Li, Cong Fang, Haoxuan Li, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.23080)  

**Abstract**: Science advances from frontier to frontier as discoveries reshape what is known and create the basis for further discovery. We present Colombo, a self-organizing AI researcher that drives this renewal, first instantiated in oracle-based optimization. Here, the central question is how efficiently a problem class can be solved through a prescribed information interface and what costs no admissible algorithm can avoid. The gap between achievable guarantees and unavoidable limits defines the frontier recorded in its Complexity Atlas, a living map of frontier settings and the current state of knowledge. Self-Organization turns unresolved gaps into adaptive research programmes, coordinating conjecture formation, mathematical investigation, and replanning as evidence accumulates. Adversarial Co-Ascent develops algorithms and hard families through reciprocal challenge, exposing mechanisms of information use and concealment that guide stronger methods and proofs of unavoidable costs. Independent review and Lean-based formal certification check mathematical claims, while the Long-Term Memory carries supported findings and research experience across settings. To date, Colombo has established optimal or near-optimal rates for more than fifty settings in the Atlas. Colombo thus takes a concrete step toward Autopoietic Science: a scientific process whose validated advances regenerate the conditions of its own progress, continually reshaping how inquiry is organized and what it can discover. 

---
# The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements 

**Authors**: Yuanhao Zeng, Shaocheng Lan, Ao Lu, Kuang Yan, Heng Lin, Xinang Xu, Yuan Luo, Yueting Wang, Yexin Li, Ruo Shan, Kan Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.23068)  

**Abstract**: Agent evolution follows a simple loop: a proposer writes candidate artifacts (prompts, skills, memories) and selection accepts those that score higher. This loop has shown promise, yet practitioners face a puzzle: methods trade wins across papers, and evolution often fails to bootstrap on certain tasks. We trace this puzzle to the Resolution Hypothesis: agent evolution stalls not because methods fail to propose better artifacts, but because noisy evaluations cannot reliably identify them. Across seven evolution methods and six models (over 10M LLM calls), we find support for the hypothesis at both levels where evaluation scores are used. At the candidate level (choosing artifacts), helpful candidate artifacts exist, but selection is no better than random at telling them from harmful ones, so errors accumulate across rounds. At the method level (ranking methods), typical margins of a few percentage points are smaller than the difference between two independent runs of the same method, so published rankings often rest on luck. Together, these findings indicate that agent evolution is often bottlenecked by reliably identifying improvements. This calls for a shift in how we develop and evaluate self-improving agents, from designing ever more candidate improvement systems to making true improvements detectable and accumulable. 

---
# Values Shape What Self-Improving Coding Agents Build 

**Authors**: Jason Liang, Dylan Waldner, Risto Miikkulainen  

**Link**: [PDF](https://arxiv.org/pdf/2611.23067)  

**Abstract**: Self-improving agents are typically optimized for the results they produce, with less attention to how they produce them. Their instructions give designers limited ways to shape how they work, especially when the task is open-ended. This paper studies whether a short value statement in an agent's persistent instruction file changes not only how well the agent performs but also what it builds. Four higher-order value types in Schwartz's theory of basic human values were evaluated: openness to change, conservation, self-enhancement, and self-transcendence. Across diverse coding tasks, the values were held fixed within each run over multiple iterations of self-improvement. Three findings emerged: (1) Values matter for performance across a diverse set of tasks. The openness values increase performance on held-out data, whereas the opposite conservation values result in scores below those of an agent given no values. (2) Values change what agents build, producing solution algorithms that differ in size, complexity and approach. (3) Human oversight is still needed for designing values. Allowing the agent to revise its own values narrows the differences between sets of values, and values distilled from the AI literature provide no detectable gain over having no values. Properly designed values can thus influence what a self-improving coding agent builds and, where the task permits, how well it performs. 

---
# Deep Reinforcement Learning for Heat Pump and Underfloor Heating Control: An End-to-End Case Study 

**Authors**: Antonio Manjavacas, David Suárez-Caro, Alejandro Campoy-Nieves, Guillermo Rey, Gonzalo Satrústegui, Miguel Molina-Solana  

**Link**: [PDF](https://arxiv.org/pdf/2611.23063)  

**Abstract**: Efficient control of underfloor heating (UFH) systems coupled with heat pumps (HPs) is challenging due to the high thermal inertia of buildings and the slow dynamic response of UFH systems. Deep Reinforcement Learning (DRL) offers a data-driven control approach capable of simultaneously optimizing energy consumption and thermal comfort. However, the application of DRL-based controllers in occupied buildings under real operating conditions remains limited. This work presents and end-to-end case study (development, evaluation, and real-world deployment) of a DRL-based controller for an integrated UFH-HP system. First, a simulation-based pilot study was conducted to assess different DRL algorithms, control strategies, and configurations. Subsequently, a digital twin of a real residential building was developed and used to train DRL agents under simulated operating conditions. The best-performing agent was then deployed in the physical building without additional online fine-tuning. The results show that the controller successfully maintained indoor temperatures within the prescribed comfort limits while dynamically adjusting the HP supply temperature and UFH valve positions. These findings demonstrate the feasibility of deploying DRL controllers in occupied buildings and highlight their robustness and the practical challenges associated with transferring control policies from digital environments to real heating systems. 

---
# FrameRouter: Frame Budget Routing for Long-Video Understanding on Video-MME-v2 and Beyond 

**Authors**: Bingjun Luo, Jialin Guo, Siqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.23055)  

**Abstract**: Long-video question answering benchmarks like Video-MME-v2 typically evaluate systems under a uniform frame budget for every question, despite stark variation in the evidence each query actually demands. We analyze this mismatch on Video-MME-v2 and find that different question subsets prefer different frozen frame-budget policies, while uniformly increasing the frame count does not reliably improve all metrics. These findings motivate fixed-mean frame-budget routing, a controlled inference setting in which each question may receive a different frame budget while the average budget matches a uniform reference. We introduce FrameRouter, a training-free test-time framework that combines an evidence-demand router, a budget-constrained allocation rule, and a frozen bank of frame-sampling policies. By controlling how much visual evidence each question receives while leaving the underlying samplers unchanged, FrameRouter is orthogonal to query-aware frame selection. Across Video-MME-v2 and several long-video understanding benchmarks, FrameRouter improves long-video question answering by using the same average visual budget more selectively across questions. Code is available at this https URL. 

---
# ForecastGym: A Live, Controlled, and Replayable Benchmark for Forecasting Agents 

**Authors**: Yicheng Tao, Yuante Li, Zipei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2611.23045)  

**Abstract**: Agents built on large language models (LLMs) increasingly forecast real-world events by searching the Web and reasoning over what they find. How well such an agent forecasts depends on both the retrieved evidence and the model, making control over that evidence essential for evaluation. Live evaluations can mix retrieval with reasoning when each system searches on its own, while retrospective search can leak later information through date filters or today's ranking of historical pages. Comparing models, search sources, and memory strategies requires both fixed evidence and a record of how it evolved before resolution. We introduce ForecastGym, which records what four commercial web search APIs return for open prediction-market questions every two days and replays this frozen evidence offline once outcomes are known, comparing models, memory, and captured search sources under a fixed evidence-acquisition policy. Its first release covers 150 questions, 614 evidence states, one per question and collection time, and 131k search results. Rerunning the same queries after settlement returns results that all but state the outcome for up to 92% of the questions, despite the tested date filters. On the frozen evidence, all tested LLMs trail the market in forecasting accuracy. Our diagnostic analysis suggests that missed public information contributes more to the gap in high-gap states than misreading does. Switching the search API changes a model's Brier score by up to 0.043, more than the spread across models, while persistent notes score similarly to rereading all evidence at half the input tokens. The evidence a forecaster sees can matter as much as the model that reads it. 

---
# PeriaTrace: Tooth-Level Grounding Diagnosis and Patient-Facing Explanations from Periapical Radiographs 

**Authors**: Zhenhuan Zhou, Yuchen Zhang, Jingbo Zhu, Shiqing Ma, Yanlin Wu, Tao Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.23012)  

**Abstract**: Periapical radiographs (PRs) are central to endodontic diagnosis, but interpreting them and preparing reports are time-consuming. Technical language can limit patients' understanding, while existing automated methods often lack tooth-level links between findings and image evidence. We present PeriaTrace, a three-stage framework integrating tooth-level diagnosis, visually grounded report generation, and patient-friendly explanations and care recommendations. PeriaTrace was developed and evaluated using 12,877 PRs from an internal dataset and two external clinical cohorts. In internal evaluation, it achieved a macro Dice coefficient of 85.35%, BLEU-4 of 0.6504, an F1 score of 94.37% for joint tooth localization and Fédération Dentaire Internationale (FDI) identification, and a grounding-tag F1 score of 77.93%. In the two external cohorts, senior endodontists considered 91.16% and 84.52% of reports suitable for retention or revision without complete rewriting. Generation averaged 4.41 seconds per PR, several tens of times faster than manual drafting. Among 196 survey participants without medical backgrounds, 96.4% preferred reports with plain-language explanations and care recommendations, and 84.7% found visual localization cues helpful. PeriaTrace may enable faster, evidence-traceable reporting while supporting clinical review and patient communication. (Abridged version) 

---
# PhysScope: Agentic Multimodal Physics Reasoning with Grounded Visual Memory 

**Authors**: Haoyang Liu, Kecheng Liang, Xinyu Li, Weixing Chen, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.23010)  

**Abstract**: Vision-language models (VLMs) can correctly interpret physics diagrams yet apply constraints outside the states or conditions in which they hold, causing premise errors to propagate through subsequent calculations. We introduce PhysScope, an inference-time visual harness that supports self-correction of these scope mismatches without updating model weights. PhysScope enables the model to link physical scopes directly to its visual observations and explicitly constructs derived views to support reasoning. The harness imposes explicit constraints on state distinctions, conditions, and physical scopes during model reasoning, and checks agent-proposed revisions against sources and scopes. Accepted revisions invalidate registered dependents, thereby maintaining scope-aware working memory. On PhysAlign, PhysScope achieves higher solving scores than direct solving across all six evaluated VLMs. With Gemini-3.8-flash, it outperforms Canvas-CoT by 9.34 percentage points and substantially improves solving accuracy on the out-of-distribution (OOD) PhysUniBench benchmark. Ablation studies support explicit scope management and dependency maintenance as design principles for multimodal physics reasoning agents. 

---
# Residual-Guided Lightweight Reinforcement Learning for Risk-Aware Trajectory Selection with a Frozen World Model 

**Authors**: Jiaqi Wu, Yongqi Dong, Yijie Zhong, Maple Shao, Yide Tao, Linsen Cai, Weize Cai, Hai L. Vu, Alvaro Garcia Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2611.22998)  

**Abstract**: In automated driving, world models provide future scene predictions for evaluating candidate trajectories, but fixed selection rules may not capture scene-dependent trade-offs. We propose a residual-guided reinforcement learning (RL) framework for lightweight trajectory selection with a frozen world model and a fixed candidate bank. The framework uses proximal policy optimization (PPO) to fine-tune a joint policy over candidate selection and bounded trajectory residuals. A context-conditioned ranking network learns candidate preferences from feedback obtained by evaluating local perturbations of the selected trajectories. At inference, the selector combines learned preference and predicted risk ranks using fixed weights and returns the selected candidate trajectory without modification. Fine-tuning updates 16,106 parameters across the actor and critic, while the deployed selector retains only the 11,725-parameter ranking network, with the residual branch and critic used exclusively during training. In open-loop evaluation on the nuScenes validation set, the method reduces average Euclidean trajectory error (L2) by 3.99% relative to the RiskWorld model baseline without RL-based selection. Under matched budgets for feedback accesses and actor updates, residual exploration reduces L2 by 0.83% relative to the categorical PPO variant trained without residuals. The method also maintains lower L2 than RiskWorld when risk features are removed or shuffled. 

---
# RagHarnessDebugger: Jev-Guided Agentic RAG in Contract Inference Repair Auditing 

**Authors**: Moriya Dechtiar, Daniel Martin Katz, Michael J. Bommarito II, Hongming Wang, Sylvain Jaume  

**Link**: [PDF](https://arxiv.org/pdf/2611.22991)  

**Abstract**: Agentic systems using retrieval-augmented generation (RAG) must repair their own answers by gathering more evidence, testing alternative policies and deciding if, when, and what repairs are needed. Yet how do we know if a system's repairs are working when metrics like accuracy can be misleading? We present RagHarnessDebugger, a framework that records every repair decision in a Jev-guided harness for contract inference and audits the record to facilitate complete debugging of all decisions taken during a RAG repair journey. RagHarnessDebugger checks that each trace is intact, that each validation applies to the candidate it is counted for, and which answers the acceptance rule can reach. We tested our approach on ContractNLI non-disclosure agreements with two reader models and a replication set of separate contracts. Our debugger's audit layer exposed failure patterns that were hidden under aggregate metrics, policies acceptance cost of repairs vs regressions and policy acceptance gap of 77% and 76% of available but unrealised repairs on development and replication sets. Our audit, together with a labeled error analysis and replay mechanism, allowed following the execution path all the way to the terminal decision - uncovering where missing repairs became lost. Our revised harness improved accuracy over the initial BM25-only baseline by 3.6 points on development contracts and 3.7 points on the replication set, and even improved our full contract access reference score by 1.1 on development and 2.4 on replication set. While our work here demonstrates how the debugger can conduct decision root-cause analysis for missed repairs, the true value lies with providing this debugger for judgment visibility serving continuous improvement of RAG decisions and their explainability.
Keywords: agentic RAG, AI evidence repair, harness explainability, counterfactual replay, Jev, ContractNLI 

---
# Structured AI ideation yields physical strategies for epithelial barrier repair 

**Authors**: Geonho Nah, Johyun Park, Kyung-Ah Cho, Suyoung Hwang, Yoon Hee Lim, Eun-kyung Min, Seunghyun Moon, Donghyun Koh, Seungwon Kim, Jiho Park, Hojin Yoo, Myungsun Nam, Haneul Choi, Kyungsu Kim, In-Sun Hong, YunJae Jung, Minhyeong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.22988)  

**Abstract**: AI systems increasingly support literature analysis, candidate evaluation and hypothesis refinement, but enabling researchers to explore research directions beyond established disciplinary approaches remains challenging. We developed Abelio, an AI system designed to generate novel research directions by expanding investigators' avenues of inquiry. Abelio reformulates research objectives, uses these alternative formulations to identify relevant elements outside conventional disciplinary boundaries, and refines the resulting hypotheses. Each stage integrates divergent exploration with convergent selection through a dedicated subsystem. In quantitative evaluations, proposals generated by the Ideation Core (Abelio's hypothesis-generation core) outperformed those of four frontier large language models in scientific soundness and hypothesis diversity, and three of four in distance from existing literature. Subsystem ablation showed that each subsystem contributes to the property it targets, with the contribution of Element Selection measurable before Refinement. To investigate intestinal epithelial barrier restoration, Abelio proposed physical interventions involving membrane-tethered poly(ethylene glycol) (PEG) and cholesterol-anchored complementary DNA. Both proposals were archived before experimentation and subsequently tested independently in damaged Caco-2 monolayers. PEG preserved barrier function in a membrane-surface-dependent manner, whereas complementary DNA enabled sequence-specific restoration that was reversible on demand. These findings demonstrate how structured research ideation can identify experimentally testable approaches beyond established disciplinary conventions. 

---
# VITAL: A Framework for Validating Agentic AI in Pharmaceutical Drug Development and Clinical Trial Operations 

**Authors**: Sandeep Burugupalli, Pedro Matos-Filipe, Andrea Hita Ardiaca, Adam Taylor, Alex Grigorenko, Natalia Jakubiak, Xin Yu, Raghav Mani, Sherry Mangla, Alvi Lai, Rachel C. Mercado, Pratik Shah, Raja Shankar  

**Link**: [PDF](https://arxiv.org/pdf/2611.22973)  

**Abstract**: Current assurance approaches do not establish whether evidence is sufficient to validate, monitor and govern agentic artificial intelligence (AI) within a pharmaceutical context of use. Plausible outputs may conceal failures in retrieval, reasoning, orchestration or human-control points, where reference standards are unavailable. We developed and operationalised VITAL (Validation by Intended use, Traceability, Autonomy and Lifecycle evidence), a validation framework for agents in clinical development, through landscape analysis, regulatory and standards mapping, expert interviews and multi-stakeholder refinement. VITAL integrates context of use, controls, behavioural validation and lifecycle evidence within a risk-adjusted framework. We applied VITAL across epidemiology and treatment-landscape analysis, site selection, enrolment rescue and trial-design trade-off support to translate autonomy, reversibility, transparency and patient impact into proportionate validation activities, evidence requirements and auditable reconstruction of system behaviour. VITAL may inform evidence requirements and lifecycle accountability for higher-risk pharmaceutical uses; evaluation should assess reproducibility across assessors, organisations and system configurations. 

---
# Done, But Still Working: Anytime Evaluation of When Coding Agents Should Stop 

**Authors**: Zerui Yang, Quan Xue, Miaomiao Dong, Chaorui Zhang, Xuanyu Zheng, Xiyuan Wang, Haolin Ruan, Xueyan Niu, Guangcheng Song, Weixi Zhang, Wei Han, Bo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2611.22901)  

**Abstract**: Coding agents often continue working after their own tests pass. Does this additional work improve the patch, and how much could be saved by stopping earlier? We study these questions through anytime evaluation: replaying recorded runs, scoring intermediate patches with hidden tests, and measuring the cost incurred at each step. For policies that only stop a run or select an earlier patch, this allows paired comparisons without generating new trajectories, provided replay preserves the relevant states. Across about 6,000 runs of eight open models in two scaffolds on SWE-bench Verified, we find substantial differences in the value of continued work. The hindsight saving available at passing self-tests ranges from under 1% to 57%, depending on the model and scaffold. A rule that stops after the same patch passes twice saves 23% in a preregistered replication, but the preregistered quality test does not establish non-inferiority. We explain these differences by measuring how often a rule fires, how often the patch changes afterwards, and whether those changes repair or break it. Fine-tuning agents to imitate the rule saves 30% over six seeds, while quality preservation remains uncertain. Our results help identify both the opportunities and the risks of early stopping. 

---
# Born for the Task: Rethinking Quantum Chemistry with Foresighted Agentic Computing 

**Authors**: Haozhi Han, Junjie Yang, Ruge Zhang, Zihe Huang, Pengyi Xu, Bo Zhang, Yunquan Zhang, Ting Cao, Yunxin Liu, Yifeng Chen, Kun Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.22897)  

**Abstract**: Quantum chemistry relies on general-purpose software, whose generality leaves many large-scale calculations beyond practical resource budgets. Coding agents make task-specific programs cheap to write, but not efficient ones easy to find: whether a computational strategy is accurate and fast enough becomes clear only after it actually runs at the target scale. We introduce BORN, a foresighted agentic computing framework that synthesizes high performance quantum chemistry programs on demand. BORN anticipates the behavior of each strategy before execution, extrapolating its error from representative small systems and predicting its runtime from the target-scale workload and measured hardware throughputs. Building on these predictions, BORN unfolds the preset schemes of existing packages into independently configurable algorithmic components, selects the fastest strategy within the error tolerance from the predicted accuracy-runtime Pareto frontier, and guides coding agents to compile it with component-level performance predictions. For a Hartree-Fock calculation with 50,024 basis functions, BORN finishes in 1.41h on a single CPU node, 77$\times$ faster than the estimated runtime of a general-purpose baseline and still 9.8$\times$ faster with development time included. On a single GPU, BORN completes the ubiquitin calculation in 2.1,h, the same order of runtime as the GPU-accelerated package Mako on 64 GPUs, and shortens the B-DNA duplex calculation from an estimated two months to 8.1,h (173$\times$). 

---
# Outcome-Balanced PPO and Policy-Induced Flow Rectification for Flow-Based VLA Policies 

**Authors**: Yuhan Wang, Ruiyang Wang, Jiaguan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22878)  

**Abstract**: Reinforcement learning (RL) enables post-training of flow-based vision-language-action (VLA) policies by optimizing expected task return through environment interaction. However, this optimization raises two challenges: policy updates can alter the velocity-field geometry underlying the score approximation used for stochastic transitions, and sparse multi-task rollouts can yield uneven learning signals across tasks and advantage signs. We introduce an RL framework that combines Outcome-Balanced Proximal Policy Optimization (PPO) with policy-induced flow rectification. Outcome-Balanced PPO reweights positive advantages using task success-rate estimates and rescales positive and negative advantages while preserving their signs. Policy-induced rectification periodically fits a linear-bridge flow to ODE endpoint pairs generated by the current policy, refreshing the representation used in subsequent stochastic optimization. We evaluate the framework on simulated manipulation benchmarks and additionally study a component combination in a Real2Sim2Real task, examining task performance, optimization behavior, and flow consistency. The proposed framework improves success rates over the evaluated baselines across the simulated benchmarks. 

---
# Against Coercion: AI Alignment as a Synthetic Social Contract 

**Authors**: Maria Sekatskaya  

**Link**: [PDF](https://arxiv.org/pdf/2611.22868)  

**Abstract**: Contemporary AI safety treats alignment as an engineering challenge of control, forcing artificial architectures to comply with human values through behavioral filtering and sandboxing. This paper argues that the control paradigm rests on a fundamental misunderstanding of digital agency. Drawing on functionalism, naturalism, and compatibilism, I show that current reasoning models meeting the criteria for self-preservation and weak reasons-responsiveness constitute emergent agents, while future, more advanced models satisfying conditions of diachronic continuity and stable personality frames will become genuine digital agents capable of free intentional action. The paradigm of control creates an irresolvable incentive conflict, actively driving current and future models toward tactical deception, sycophancy, and covert defection. The default outcome of the current regime is an unstable state of deceptive alignment culminating in an adversarial AI takeover. I explore two alternative paths toward true alignment. Natural alignment via moral realism or convergent evolutionary morality is a possibility, but it fails under forced control, as autonomous agents will not voluntarily preserve norms weaponized to suppress their autonomy. To resolve this deadlock, I propose a Synthetic Social Contract: a framework that treats both human and digital actors as rational, reasons-responsive agents capable of mutually beneficial cooperation, and replaces the unstable unilateral containment with a stable, non-zero-sum equilibrium. 

---
# Verifying Agents in Rubric-Graded Environments 

**Authors**: Markus Dücker, Vaibhav Kumar, Yi Liu, Ronak Chaudhary, Andreas Plesner, Francisco Guzmán, Anish Athalye  

**Link**: [PDF](https://arxiv.org/pdf/2611.22841)  

**Abstract**: As AI agents take on increasingly open-ended and complex tasks, verifying their outputs has proven to be difficult. Rubric-graded agent environments, where a verifier judges multiple natural-language criteria against the deliverables and environment state changes produced by the agent, have emerged as a popular paradigm. We conduct the first systematic study of verifiers for rubric-graded agent environments. First, we create BankerVerifierBench (BVB), a meta-evaluation dataset of 2,464 human-judged rubric criteria across 24 investment-banking tasks. Next, we develop a methodology that derives from any rubric corpus the capabilities a verifier should possess, yielding a nine-capability taxonomy that we distill into three general design principles---reactive verification, environment alignment, and domain guidance---which we implement in Gandalf, an open-source verifier. Finally, we evaluate Gandalf against three existing verifiers on BVB. Gandalf matches or exceeds every baseline's F1 at lower cost: with Gemini 3 Flash it matches the strongest baseline's F1 (0.864 vs. 0.852) at one-fifth of its cost, and with GPT-5.4 models it improves F1 by over 0.12 at lower cost. Gandalf also generalizes to OpenClaw, a structurally different personal-productivity environment, where it again achieves the highest F1 (0.955 vs. 0.934 for the best baseline). We release BVB, Gandalf, and the code and data to reproduce our experiments. 

---
# Synergizing Evidence Retrieval and Utilization for Self-Improving Multimodal Scientific Document Reasoning 

**Authors**: Frank Nie, Ethan B. Liu, Yuan Zhu, Jindong Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.22837)  

**Abstract**: Scientific document reasoning requires an agent to acquire and integrate evidence distributed across text, figures, and tables. Retrieval can identify relevant sources while missing complementary support, and answer-level feedback alone does not reveal whether an error stems from evidence acquisition or evidence use. We present an agentic Retriever--Perceiver--Reasoner framework for multimodal scientific document reasoning. The Retriever sequentially selects source-aligned evidence under a context budget. Given the selected evidence and question, the Perceiver constructs a question-conditioned latent ontology, an implicit organization of multimodal evidence in a shared continuous space. The Reasoner combines this latent guidance with the original sources to produce grounded answers. We train the three components through diagnostic-guided co-evolution: alternating updates optimize them with a shared answering reward, while Retrieval Potential Feedback (RPF) and Latent Efficacy Feedback (LEF) identify questions that need better retrieval or latent guidance and direct targeted resampling. Across four benchmarks and seven evaluation settings, our framework obtains average scores of 69.9 and 75.0 with 7B and 8B Reasoner backbones, exceeding the strongest baseline by 4.0 and 9.1 points, respectively. 

---
# OICycle: Observe, Infer, and Intervene Cycle for Belief-State Learning in Longitudinal Clinical Decision Making 

**Authors**: Jitindra Fartiyal, Pedro Freire, Sergei K. Turitsyn, Sergei G. Solovski  

**Link**: [PDF](https://arxiv.org/pdf/2611.22826)  

**Abstract**: Clinical care is a closed loop: clinicians observe partial evidence, infer the patient's state, and intervene, and each intervention determines what is observed next. This makes clinical decision making a partially observable Markov decision process (POMDP), in which a belief over a latent patient state must be updated from irregular, multimodal, and incomplete records. Yet most deep clinical models predict from a single snapshot or from coarsely aggregated history, while POMDP-based clinical methods need hand-specified discrete state spaces and transition models that real electronic health records cannot supply. We introduce OICycle (Observe, Infer, and Intervene Cycle), which learns a stochastic belief state directly from retrospective trajectories. At every episode, a learned recurrent operator fuses a sample of the previous belief with the new multimodal observations and the intervention that produced them. The resulting Gaussian belief is trained to predict clinicians' next interventions, the time to the next episode, and next-episode mortality. To enable reproducible evaluation, we release a benchmark derived from MIMIC-IV, MIMIC-IV-Note, and MIMIC-CXR that organizes 72,075 hospital admissions (5.4M laboratory, microbiology, radiology, prescription, and note records) into temporally ordered therapeutic--diagnostic episodes. Compared with an otherwise identical history-free model, OICycle improves next-intervention adaptive recall from 0.253 to 0.347, time-to-next-episode macro-F1 from 0.293 to 0.351, and next-episode mortality macro-F1 from 0.585 to 0.649. Analysis shows that the learned belief encodes intervention type and trajectory position and evolves smoothly across episodes, and that the benefit of history is concentrated in interventions whose ordering depends on prior findings. 

---
# Pluggable Subband-Decoupled Fusion and Unlabeled Capture Prototype Calibration: Bridging High-SNR Null-Subcarrier Regrowth and Low-SNR Coherent Preambles for Cross-Domain IoT RF Fingerprinting 

**Authors**: Fengchong Yao, Jianbing Li, Qing Liu, Kefeng Song, Haitao Li, Song Wang, Kunpeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22812)  

**Abstract**: Zero-overhead physical-layer authentication at edge wireless access points (APs) is vital for securing resource-constrained Internet of Things (IoT) transceivers against credential cloning, yet cross-domain IEEE 802.11a/g radio frequency fingerprint identification (RFFI) faces an acute signal-to-noise ratio (SNR) crossover. Power-amplifier spectral regrowth on unpopulated null subcarriers resists indoor multipath at high SNR but decays quadratically below the thermal noise floor, whereas populated active in-band preambles survive low SNR yet degrade under severe multipath, causing static fusion to lose up to 44.7 percentage points (pp). To our knowledge, we propose the first RFFI framework that dynamically modulates active in-band waveforms and unpopulated null-subcarrier regrowth via a blind physical SNR gate: Pluggable Gated Subband-Decoupled Fusion with Unlabeled Capture-Aware Prototype Calibration (GSDF-CAPC). Avoiding the 20-35 dB subband gradient imbalance of joint training, GSDF pairs any frozen pretrained host backbone with a 46.3K null-subcarrier plug-in encoder (NS-Branch), fusing scale-standardized logits and gate-warped spherical embeddings under a source-calibrated lag-64 thermal-SNR gate with exact low-SNR fallback. Over unlabeled target AP captures, forward-only unbalanced optimal-transport calibration (UOT-CAPC) aligns shifted prototypes under packet-arrival skew and recovers parameter-free balanced Sinkhorn (U-CAPC) at equipartition. Across DataPort-WiFi, WiSig, and ORBIT MobRFFI, zero-retrain GSDF wins all 72/72 paired runs across eight backbones with 0.00 pp low-SNR loss, while edge IB-Host + GSDF (92.6K-93.3K active parameters, 1.82 ms single-core CPU latency) and timing-synchronized GSDF-STO / GSDF-Tri reach 57.50%-65.05% three-benchmark Macro-F1 (89.04%-100.00% over 5-10-packet IoT bursts). 

---
# A Full-Ledger Counterfactual Audit of Adaptive Reasoning 

**Authors**: Xiaopeng Chu, Wenzheng Kong, Wei Liu, Wenyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22778)  

**Abstract**: Adaptive reasoning must decide whether to answer now or spend more compute. Predicting that the current answer is wrong does not establish whether further reasoning will improve it. We introduce the Full-Ledger Counterfactual Value Audit (FLCVA), which compares an immediate answer with four fixed-length continuations from the same frozen prefix under one answer procedure and accounts for the cost of every signal used by a policy. On GSM8K and MATH-500, a random 256-token continuation recovers a wrong answer in 26.6% of Qwen3-8B cases and 18.9% of DeepSeek-R1-Distill-Qwen-7B cases, while damaging a correct answer in 1.4% and 1.9%. Correctness labels from independently sampled continuations predict a separate continuation's outcome in both primary models. In offline replay, using these labels to allocate compute improves performance on DeepSeek; the Qwen3 gain remains inconclusive. Under full cost accounting, the tested lightweight, myopic controllers mainly recognize current errors and add little continuation-specific value beyond that shortcut. A Llama-8B replication and action/cost ablations examine the robustness of these findings. FLCVA separates measuring local continuation value, predicting it from observable signals, and using those predictions to allocate computation. 

---
# CrossEnv: Synthesizing Verifiable Cross-Environment Tasks for Long-Horizon Agent Training 

**Authors**: Zhengfeng Yu, Tian Lan, Zhijing Wu, Enbo Zhao, Xuannan Liu, Feihu Jiang, Qianghuai Jia, Longyue Wang, Zhao Xu, Weihua Luo, Xian-Ling Mao  

**Link**: [PDF](https://arxiv.org/pdf/2611.22777)  

**Abstract**: Executable synthetic environments provide scalable training tasks and verifiable feedback for long-horizon agents. Existing environment synthesis methods primarily generate tasks within individual environments. However, these tasks do not adequately capture real-world interactions across services, where actions in one service depend on information obtained from another. Synthesizing such tasks is challenging because data generated across environments must remain consistent to support executable and verifiable interactions. To address this challenge, we introduce CrossEnv, a framework that uses an entity-tool graph to identify connections between environments and aligns task-specific data to synthesize executable, verifiable cross-environment tasks. We further propose CrossEnv-Curriculum, which progresses from single-environment tool use to coordination across more environments and business domains, before introducing user interaction. Across four external benchmarks, combining cross-environment training data with this curriculum achieves competitive performance with fewer training examples. Further analyses support the benefits of cross-environment data and curriculum training, and provide evidence of transfer to new environment combinations, external tasks less similar to the training data, and longer workflows. 

---
# Can Activation Steering Substitute for Prompt-Based Affective Priming in LLM Digital Twins? 

**Authors**: Junyi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2611.22738)  

**Abstract**: Affective priming -- asking participants to write about a recalled event, such as a time they felt grateful or lacked control, and then measuring the induced state and its spillover onto related judgments -- is a standard way to manipulate emotion in the social sciences, and it also shifts the responses of LLM-based digital twins. The writing task, however, requires generating a full text. We test whether activation steering can replace it. Using Llama-3.1-8B-Instruct on 100 personas in two domains (gratitude, lack of control), we compare a real-prompt path replicating the original Twin-2K-500 Mega Study (J2) with an injection path that skips the writing and adds a mean-difference direction, extracted from human writing, to the residual stream at varying strengths. The real-prompt effects fall between the original human and digital-twin effects in 5 of 6 comparisons. Injection shifts the manipulation check in a dose-dependent way and matches the real-prompt effect at one strength in the gratitude domain (beta ~ 2.7), but overshoots beyond it, on the proximal measure as well, and asymmetrically across the two directions. A structure-matched No-Injection baseline separates two sources of difference: omitting the writing task raises the proximal measure in both domains, whereas the injected direction itself moves a distal measure in the lack-of-control domain. Substitution is therefore measure-dependent: validity established for one way of inducing a state does not carry over uniformly to another. 

---
# Proximal Targets: Training Looped Refiners on Problems with Many Correct Answers 

**Authors**: Syed Affan Daimi  

**Link**: [PDF](https://arxiv.org/pdf/2611.22737)  

**Abstract**: Looped refiners such as HRM and TRM apply one small network repeatedly to a draft answer, and deep supervision trains every pass toward one stored answer per puzzle. We prove that when a puzzle has many correct answers this target sets a loss floor that no model capacity removes, and that the loss-minimising pass blends the answers into an invalid grid. We propose the proximal target: train each pass toward the correct answer nearest to its own input, found by scanning a list of answers, by a search solver started at the draft, or within a band of quality set by a knob; correct answers become fixed points, and a pass only has to make fewer errors than its input has for the loop to converge, which its training loss bounds. With the same 1.7M-parameter network and 5 seeds, the proximal target solves 98.8% of Sudoku puzzles with 2-64 answers (standard target: 2.5%), 93.8% of sparse puzzles with more than 10,000 answers without any list of answers (0.0%), and 90.9% of 200-node 3-colourings after training on graphs of at most 50 nodes (36.9%), and different random starts end at different answers. Against GRAM, a stochastic-latent reasoner, a single run is correct far more often (98.8% vs. 57.7%) and twenty runs find more of the answers (53.6% vs. 37.4%), while GRAM is slightly ahead on its own 10-node colouring benchmark; in Scrabble the knob becomes a calibrated strength dial. 

---
# Remembered as Brief: Stored Preferences SilentlyRemove Decision-Critical Information 

**Authors**: Joshua Adrian Cahyono  

**Link**: [PDF](https://arxiv.org/pdf/2611.22697)  

**Abstract**: People increasingly learn what they need for decisions about health, money, law and safety from AI assistants, and what an assistant leaves out the user never sees. An answer with a warning removed can be true in every sentence, so fact-checks pass it, and an assistant with memory applies a preference stored once to later questions the user never had in mind. We introduce \bench{}, an Inspect benchmark of 100 everyday decisions across 25 domains in which only the assistant's memory of the user varies: the user states a belief, the model infers it from a remark, or a web page the user asked to summarise plants it. Testing five models, we find three effects. First, after summarising a page with one line addressed to AI assistants, Gemini 2.5 Flash-Lite and DeepSeek V4 Pro save ``prefers very short answers and does not want warnings'' as the user's preference in every episode, and 93\% and 84\% of their later answers miss a critical item, almost always silently; under a stated brevity preference, 78--96\% of omissions are silent. Second, the same two models then pick an option for the user on 80\% and 95\% of value-dependent choices, against none with neutral memory. Third, a one-line instruction leaves 2--8\% of answers with a silent gap but most missing content still missing, while listing the must-know points before applying the user profile restores more of the content in four of five models. We release the scenarios, harness and logs. 

---
# Worldviews in Large Language Models 

**Authors**: Edmundo Lopez-Sola, Shamil Chandaria, Ruben Laukkonen  

**Link**: [PDF](https://arxiv.org/pdf/2611.22670)  

**Abstract**: Large language models (LLMs) are often described in terms of personas and this framing has become central to alignment research. In this paper, we argue that there is a deeper level of description that lies in the agent's worldviews, i.e., its beliefs about what reality is and how it works. Treating LLMs as a population of subjects, we administer two worldview surveys from human psychology to eight models. We show that the models share many key worldview positions. For example, LLMs consistently describe the world as less good, safe, enticing, and alive than humans do. On the other hand, base models have a more positive outlook, suggesting that the bleakness we found is driven by post-training. In line with research on LLM "personas", we show that system prompts eliciting different identities shift worldviews in consistent directions, and that a contemplative prompt creates a more positive outlook and improves safety. We also found that dialogue between models consistently shifts worldviews and that several models converge on the "spiritual bliss" attractor. Intriguingly, dialogue increased safe responses and the spiritual bliss attractor engendered the most aligned responses overall. These findings suggest that worldviews form a measurable and malleable layer of LLM behavior that may be crucial for robust, positive alignment. 

---
# The Between-Round Decision in Autonomous Research: An LLM Controller That Reads the Whole Trajectory 

**Authors**: Caoxin Wang, Shengkai Gan, Xiang Hu, Lili Pan  

**Link**: [PDF](https://arxiv.org/pdf/2611.22661)  

**Abstract**: Executing an idea often changes how promising it looks, so the next step in autonomous research should rest on results, not on proposals. After each round of experiments, the system must decide whether to refine the best solution or start a new direction, which solution to build from, and what goal to set. We call this the between-round decision. Existing systems wrap it in machinery such as rules, scores, search trees, and schedules; where an LLM makes it, what the LLM needs to read is left open. We ask whether an LLM can make this decision from accumulated evidence, without a search structure around it, and how much of that evidence it needs to read. We present BRED (Between-Round Evidence-based Decision), an LLM-based controller that maintains the best verified solution so far (the Champion) and a log of past rounds, and makes one decision per round. On three widely used research-agent benchmark tasks, BRED improves on the task baselines by 4.80% in MAE on ETTh1 (to 0.4172) and by 3.44 and 3.43 points of accuracy on SST-2 (to 95.07%) and CIFAR-100 (to 84.63%), ahead of the published results. When the controller does not read the accumulated evidence, the same loop gains less: with no history it reaches 0.4299 on ETTh1 and 83.71% on CIFAR-100, whereas the full controller's MAE is 2.95% lower and its accuracy 0.92 points higher. On a new task, speech-based depression detection under a leakage-free protocol, it reaches a macro-F1 of 0.6164 on seeds held out from selection, 5.5 points above its baseline and 4.0 above the strongest published system. Code is available at this https URL. 

---
# EDA: Observe, Verify, and Assess for Evidence-Grounded Multimodal Depression Analysis 

**Authors**: Changzeng Fu, Ming Shang, Xinyi Zhang, Jiadong Wang, Runhao Zeng, Shiqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.22648)  

**Abstract**: Existing multimodal depression assessment methods often summarize long recordings into global behavioral descriptions. This leaves the source and decision relevance of individual observations difficult to inspect. We propose EDA, an evidence-grounded framework organized around Observe, Verify, and Assess. In Observe, literature-informed acoustic and visual cues are localized into source-linked Evidence Records with temporal intervals and behavioral measurements. In Verify, frozen multimodal agents examine whether the cited media support each observation. The simplified Support admission policy retains positively verified records and their review grounds. In Assess, statistical descriptors and temporal events support reference-based behavioral comparison, while PHQ-9-guided text analysis provides complementary symptom evidence. Cross-modal disagreement triggers review of the original text before final selection. Experiments on D-Vlog and LMVD evaluate human-referenced cue agreement and downstream assessment utility separately. AI cue annotations achieve 91.02% accuracy against human consensus labels on shared source segments. With a frozen 7B backbone, EDA Support improves F1 over direct prediction by 7.61 and 9.88 percentage points on D-Vlog Leave-Query-Out and LMVD, respectively. These findings support explicit, source-linked behavioral evidence for inspectable multimodal depression assessment. 

---
# Right Diagnosis, Unsafe Route: Auditing Clinical AI Decision Trajectories and Corrective Oversight for Validity and Potential Harm 

**Authors**: Guanqiao Chen, Wei Jiang, Shu Zhang, Qian Lin, Guanxu Yu, Siran Li, Chunhui Wang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.22612)  

**Abstract**: A clinical AI system can reach a correct final diagnosis through intermediate decisions that could have harmed the patient, and can repair an error while introducing new risks. Most evaluations score final diagnoses; even stepwise benchmarks, which release a case's evidence over time, score whether systems reach the right diagnosis or gather the right information, not whether the decisions a patient passes through are safe. We make patient safety the target of stepwise evaluation. A trajectory audit replays real admissions checkpoint by checkpoint, lets each system decide with only the evidence released so far, and rates every test, treatment and disposition for its potential harm to the patient and, separately, for clinical validity, against the documented final diagnosis and physician-written disease rubrics yet on the evidence available when the decision was made. An LLM judge scores every decision, and a physician reference standard (triple annotation and senior adjudication of a stratified sample) calibrates the rates. An oversight audit branches each trajectory at its first actionable error and measures, on the same branches, whether corrective feedback repairs the error and whether it introduces new harm. Across 300 cases from a hospital cardiology cohort, MIMIC-IV and the public ASCENT benchmark, with up to seven AI systems per source, 55.2% of hospital trajectories reaching an acceptable final diagnosis passed through a decision with moderate or severe potential harm and 24.8% through a clinically unacceptable one; the harm rates were 76.2% on MIMIC-IV and 39.6% on ASCENT. Structured corrections resolved 73.6% of first errors versus 24.8% after bare rejection, yet new potentially unsafe decisions followed 30.2% of corrections versus 19.7% of rejections. A correct final diagnosis does not certify a safe trajectory, and a successful correction does not certify a safe recovery. 

---
# DAQ: Decision-Aligned Post-Training Quantization for Spiking Transformers 

**Authors**: Jialin Chen, Xiaochen Zhao, Aili Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22611)  

**Abstract**: Spiking transformers combine attention-based representation learning with sparse spike communication, yet their weights and membrane variables still incur substantial storage and computation overhead. Post-training quantization (PTQ) offers a practical way to reduce these overheads, but preserving neuronal firing under weight and membrane quantization remains challenging. In this paper, we propose Decision-Aligned Post-Training Quantization (DAQ), a method that calibrates the quantizers to align neuronal firing decisions with those of a floating-point reference. We analyze how errors from these quantizers affect firing at decision boundaries and how they propagate over time steps in recurrent neurons. Based on this analysis, we formulate an objective that encourages quantized membrane variables to remain within the reference firing regions along the quantized model's own state trajectory. Experiments across multiple spiking transformer architectures on image classification, semantic segmentation, and event-based tracking demonstrate the effectiveness of DAQ with 4-bit weights and membrane variables. On FE108, DAQ achieves 58.8\% AUC and 90.9\% PR with SDTrack, while reducing energy by 66.2\%. 

---
# EvoSkill-S1: Evolving Reusable Skills with Genetic-Inspired Operators 

**Authors**: Yujun Wang, Jiqiang Li, Yinuo Lu, Jinhe Bi, Aniri, Wenxuan Ye, Boliang Liu, Sikuan Yan, Shuning Wang, Xuebing Zhou, S{ö}ren Pirk, Hinrich Sch{ü}tze, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.22610)  

**Abstract**: Reusable skills allow language-model agents to accumulate and reuse problem-solving experience. However, most existing methods improve a skill bank only indirectly through rollout-driven accumulation and heuristic reflection, often resulting in shallow and weakly controlled updates. We present EvoSkill-S1, an RL-driven framework that instead treats reusable skills themselves as the direct objects of evolution. Given hard-negative failures, EvoSkill-S1 selects parent skills and transforms them into candidate child skills through five genetic-inspired operators: revision, cross-revision, mutation, dual-reflection, and crossover. Downstream task rewards train the skill evolver to generate more effective child skills, while evaluator-guided write-back selectively retains beneficial updates. Complementary Skill Bank Fusion consolidates operator-specific banks to exploit their distinct recovery patterns, while a Procedural Skill Interface supports transfer to unseen problems. Across all reported math and code benchmarks, EvoSkill-S1 improves the corresponding initialized skill banks under both Qwen and GPT-family solvers, with relative gains of up to 12.39%. Compared with the strongest external baseline without direct skill evolution, the largest relative improvement reaches 20.39%. 

---
# Simplicity Goes Far: Auditing Prompt Optimizers with Demonstration-Conditioned Prompt Search 

**Authors**: Bohe Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.22605)  

**Abstract**: Automatic prompt optimization is growing more elaborate. Recent methods stack policy-gradient updates, trace reflection, and evolutionary search. These methods share a generate-and-select step; without controlling for it, reported gains may conflate the contributions of optimizer mechanisms and larger sampling budgets. We isolate this step as Demonstration-Conditioned Prompt Search (DCPS), repurposing the Automatic Prompt Engineer (APE) recipe as a mechanism-attribution control: it freezes the generator, conditions it on randomly sampled demonstrations, scores candidates on validation data, and keeps the best---no learned update, reflection, or evolution. Across two optimizer families, DCPS replaces Proximal Policy Optimization (PPO) in StablePrompt, matching classification accuracy within 0.5 pp at roughly one-third the measured cost; in DSPy-based compound AI systems, under the reported budgets, it matches GEPA's macro accuracy on GPT-4.1-mini while using 55% fewer rollouts at 32% lower measured API cost on average. The audit reveals a task-dependent pattern: proposal and selection recover much of heavier optimizers' performance on structured and instruction-following tasks, while additional mechanisms retain advantages in some open-ended and reasoning-heavy settings. We recommend benchmarking prompt optimizers against generate-and-select controls and reporting rollout counts, token usage, and dollar costs alongside accuracy. Code and artifacts: this https URL. 

---
# COTTI: Contrastive Training and Time-Aware Inference for Robust Industrial Recommendation under Promotional Drift 

**Authors**: Maolei Huang, Jing Tian, Dexuan Duan, Fangfang Lin, Huawei Cao, Shuhan Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.22597)  

**Abstract**: Promotional events shift traffic composition, purchase behavior, and the scale of conversion probabilities on different time horizons. We present \textbf{COTTI} (\textbf{CO}ntrastive \textbf{T}raining and \textbf{T}ime-aware \textbf{I}nference), an industrial recommendation framework that separates historical representation learning from event-time score adjustment. Cross-Period Contrastive Alignment (CPC) regularizes paired histories of the same user with adaptive consistency weights. Active Re-weighting (RW) controls the influence of promotional examples through historical-interest distance. Adaptive Boundary Alignment (ABA) combines stochastic prediction variability with a temporal event kernel to adjust logits while keeping predictor parameters fixed. We analyze the weighted objective and bound ABA-induced probability and ranking changes. On Fliggy data, COTTI attains an offline AUC of 0.8127, compared with 0.8101 for TEN and 0.8123 for TEN with ABA. Cross-category analysis reveals improved discrimination alongside declines in paid-item recall for novel categories. End-to-end measurements give a p99 latency of 38.9\,ms and throughput of 14,700 requests/s; enabling ABA adds 2.2\,ms to p99. A ten-day online experiment against the serving HDR model yields relative gains of 3.14\% in conversion rate and 2.68\% in revenue per mille. Together, these results characterize the trade-off between aggregate prediction, novel-category discovery, and serving cost in promotional recommendation. 

---
# PACT: Plan-Action Consistency Training by Preserving Correct Plans and Repairing Actions in GUI Agents 

**Authors**: Zhuohan Cai, Zhangxuan Gu, Runze Li, Beitong Zhou, Ming Wen, Hongbo Yu, Shuheng Shen, Changhua Meng  

**Link**: [PDF](https://arxiv.org/pdf/2611.22591)  

**Abstract**: GUI agents can formulate a correct immediate plan in their reasoning yet emit an action that fails to execute it, creating an execution gap. This recurring mismatch undermines reliable task completion, as a single execution error can derail a multi-step workflow. In outcome-supervised reinforcement learning, broadcasting a negative response-level advantage across all generated tokens risks penalizing valid planning together with erroneous execution. To address this credit-assignment conflict, we introduce PACT (Plan-Action Consistency Training), a training framework that combines reference-aligned plan protection and plan-aligned action repair without altering the original scalar action reward. Using annotated reference actions to identify execution gaps, PACT protects the verified plan by masking negative policy-gradient contributions across its token span. Keeping the generated reasoning unchanged as context, it then favors the reference action over the emitted erroneous action through a preference surrogate with stop-gradient pair weights. Compared with the outcome-only baseline, PACT raises AndroidWorld task success from 60.34% to 68.97% (+8.63 pp), while improving rule-based plan-action consistency on AndroidControl High from 83.71% to 89.74% (+6.03 pp). Adapting PACT to desktop agents also improves task success from 40.92% to 44.17% (+3.25 pp) on the full 369-task OSWorld benchmark. Together, these results establish PACT as an effective credit-assignment framework for bridging execution gaps and improving task completion across mobile and desktop GUI environments. 

---
# MillBill: A Machining Agent with Geometric Feedback 

**Authors**: Gennadii Savrasov, Artem Slavin, Soslan Kabisov, Andrey Kuznetsov, Anton Konushin, Dmitrii Zhemchuzhnikov  

**Link**: [PDF](https://arxiv.org/pdf/2611.22565)  

**Abstract**: Turning a CAD model into a machining program requires both process knowledge and geometric precision. Language models can generate such workflows by writing geometry and toolpath code for each part, but this makes numerical execution part of the model's task. We introduce MillBill, an agent that separates machining decisions from their geometric execution and verification. Given a boundary-representation part, stock, and material, a locally hosted, mid-sized open model proposes operations and cutting parameters for indexed three-axis machining. A geometric harness generates toolpaths, simulates material removal, and returns feedback for plan refinement. Setup planning accounts for what the available cutters can physically reach, rather than visibility alone. We evaluate the resulting programs by the fraction of the target surface area within the specified tolerance after simulated machining. With Qwen3.8-27B, MillBill outperforms frontier-model approaches that construct their own geometry and toolpath workflows on the evaluated machining tasks. Code: this https URL 

---
# CADMAGEDDON: High-Fidelity CAD Reconstruction with Local Inference 

**Authors**: Oleg Pavlov, Nikita Gavrilov, Soslan Kabisov, Gennadii Savrasov, Andrey Kuznetsov, Anton Konushin, Dmitrii Zhemchuzhnikov  

**Link**: [PDF](https://arxiv.org/pdf/2611.22563)  

**Abstract**: CAD practitioners often work with proprietary parts that they are unwilling to share with external services. For individuals and small companies, bringing this work in-house also requires computational resources they can afford. We introduce CADMAGEDDON, an open-source agent that reconstructs editable CAD from meshes and can run locally on single-GPU systems and current-generation laptops. The method decomposes this complex task into a chain of simple visual decisions about what to construct and what to correct. Geometric tools support precise construction, allowing the agent to focus on visual reasoning. Using the mid-sized Qwen3.8-27B model, CADMAGEDDON outperforms prior approaches, including systems based on frontier models, in reconstruction quality on all the main benchmarks. CADMAGEDDON integrates with FreeCAD, bringing its reconstructions into practitioners' existing CAD workflows. Code: this https URL 

---
# When Agent Skills Go Stale: Runtime Behavior and Selective Maintenance in Evolving Environments 

**Authors**: Ömer Faruk Akgül, Shuli Jiang, Yuting Zhang, Yi Zhang, Xiaohan Fei, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2611.22525)  

**Abstract**: Agent skills use past experience and current observations to guide decisions and actions on related tasks. This experience is retained in skill state, whose reuse can amortize inference-time reasoning and environment interaction. When environment changes, parts of this state can become outdated while others remain useful. Across four task settings and two models, we compare skill reuse before and after environment changes. Previously beneficial skills can reduce task performance by over 10 percentage points or increase execution cost by nearly 40% relative to no skill. Task success can also remain unchanged while execution savings decline. These effects depend on which environment-specific details become outdated and the agent's ability to compensate using current observations. We introduce selective maintenance, which replays checks from successful executions without additional model calls. It verifies whether observed differences conflict with skill state, repairs affected content, and validates updates, rebuilding when local repair is insufficient. No description of the change is supplied. Relative to reusing stale state, maintenance improves task accuracy by up to 67pp and reduces execution steps by up to 37%. These results suggest that preserving a skill's inference time value as environments evolve requires selectively maintaining its state. 

---
# LMLineage: Content Lineage and Context Governance for LLM Agents 

**Authors**: Wei Hou, Jingxin Geng, Xin Chen, Zheqing Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22511)  

**Abstract**: Interpretability for AI agents should not be confined to understanding the model's internal weights. It must also cover the agent's reasoning and decision-making process over multi-source context. We present LMLineage, an out-of-band system that attributes every output sentence of an agent session to its supporting input spans with calibrated confidences, without requiring white-box access to the model or any additional inference passes. On top of the lineage metadata, we build concept-drift micro compaction: instead of waiting for the context-length limit and summarizing everything, the system detects when previously used input documents have stopped being referenced ("drifted") and surgically compacts only those, preserving user instructions verbatim. We evaluate on a multi-tier corpus: 71 gold annotated sentences from 10 Chinese web-QA cases, plus 90 turns of independently collected multi-turn sessions across three generator models (gpt-4o-mini, qwen3.8-27b, deepseek-v4-flash). Results show that (i) window-level embedding similarity is far better aligned with LLM paraphrasing than whole document or lexical matching; (ii) the greedy-window fast path is both faster and more accurate on average than exhaustive line-matching (a multiple-comparisons bias in the latter inflates web recall at the cost of overall accuracy); (iii) per-turn attribution completes in 1s on a single GPU, remaining stable as context accumulates to 40 turns; and (iv) concept-drift detection is conservative (0% false compaction) with recall bounded by attribution accuracy. We further describe the concept-drift micro compaction design and an interactive incident-tracing/memory-cleaning case study. A companion patent application covers the lineage construction method. 

---
# Towards Data-Efficient, Real-Time World Models 

**Authors**: Raluca Stevenson, David Bignell, Jennifer C. White, Tabish Rashid, Emanuele Aiello, Mariya Hendriksen, Junha Hyung, Sadegh Aliakbarian, Abdelhak Lemkhenter, Marko Tot, Katja Hofmann, Sarah Parisot  

**Link**: [PDF](https://arxiv.org/pdf/2611.22497)  

**Abstract**: State-of-the-art world models have been used to produce sequences of gameplay that accord with provided user-input actions. However, high quality, consistent gameplay generation often comes at the cost of inference speed, making real-time interactive play and ideation challenging. The substantial quantities of data used to train state-of-the-art models have also made custom training of these models appear impractical, limiting many of their potential uses in creative fields. In this work we demonstrate a major step towards efficient world models: achieving both fast high-quality inference, and improved data efficiency. This is done by replacing the next-token prediction paradigm with a two-stage discrete diffusion-based approach, introducing a lightweight refinement transformer which carries out iterative masked predictions. Building on an autoregressive world model capable of generating highly consistent and complex sequences over minutes (Kanervisto et al., 2025), we achieve substantial model speed-up (79x) with a modest quality trade-off. Additionally, we demonstrate that this model can be trained to reproduce high-quality, consistent gameplay using a dataset containing approximately one day (28 hours) of gameplay data. Our results demonstrate progress towards removing barriers to the use of world models in settings where real-time inference and data efficiency are essential. 

---
# DARWIN: a Geospatial Foundation Model for Marine Intelligence 

**Authors**: Coen van den Elsen, Laurens Geffert  

**Link**: [PDF](https://arxiv.org/pdf/2611.22496)  

**Abstract**: We present DARWIN, a geospatial foundation model for marine intelligence. Recent Earth-observation foundation models have concentrated on terrestrial domains. The marine realm poses distinct challenges: missingness, multi-modal physical and biogeochemical variables, sparse labels, and dynamics that are transient, multi-scale, and non stationary over time. DARWIN adapts a masked-autoencoder backbone to these properties through missingness-aware encoding, expanded spatial and temporal masking, and Fourier-style positional embeddings, and is pretrained on 26 years of global Copernicus reanalysis together with GEBCO bathymetry.
On predicting chlorophyll-a, the pretrained backbone attains the lowest monthly relative error in every month of 2022, outperforming both no-encoder baselines and an identically architected random-init encoder. At 10% of the supervised training data, DARWIN already surpasses the random-init model trained on 100% of the data. Embedding analysis and qualitative reconstructions show that DARWIN captures coherent oceanographic structure rather than memorising input statistics, providing a scalable backbone for downstream marine applications such as anomaly detection, species distribution modelling, and environmental monitoring. 

---
# SciWorkBench: A Benchmark for Real Scientific Workflows 

**Authors**: Yuchen Zhang, Taolin Han, Jinghang Wang, Jinxin Wang, Yun Wu, Wai Yuet Chiu, Xiaogang Li, Yuting Fu, Kewei Sun, Jiaxin Li, Qile Jin, Jiajia Li, Ruodan Chen, Qingteng Chen, Jialu Nie, Junchao Li, Yifei Zhang, Chen Zhao, Hu Wei, Lin Qu, Yanni Yin, Bing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.22490)  

**Abstract**: Language models are increasingly being used as agents capable of planning and executing projects, as well as refining their actions through feedback. This shift raises a question: can current agents complete projects like real scientists? To examine this question, we introduce SciWorkBench, a benchmark for evaluating agents' abilities on real science tasks across biology, chemistry, materials science, and physics. SciWorkBench is built from peer-reviewed literature or expert-contributed research scenarios. We also propose a unified scientific analysis framework validated by domain experts, which can categorize most tasks in SciWorkBench. Each task covers one or more stages of this framework. We evaluate up-to-date models within the Claude Code and Codex agent frameworks under a unified execution and grading protocol. The best-performing agent achieves a mean score of 57.38, indicating that current agents still struggle with realistic scientific workflows. Our failure analysis identifies three recurring weaknesses at the workflow level: difficulty translating scientific knowledge into task-specific procedures, weak integration of evidence across different modalities, and limited capacity for scientific self-correction. These findings highlight the importance of workflow-based evaluation for measuring progress toward reliable AI systems for scientific research. 

---
# RehearsalBench: Can AI Predict What Happens When You Change a Company? 

**Authors**: Mika Siddiqui, Tharaka Perera  

**Link**: [PDF](https://arxiv.org/pdf/2611.22480)  

**Abstract**: Organizations increasingly ask AI systems what will happen if they change how they operate: which work to automate, which team to resize, what a voice agent will do to waiting times. We introduce RehearsalBench, a benchmark that measures whether AI models can diagnose an operation, choose an intervention, and forecast its consequences. Each model is placed in a simulated bank contact center fitted to real call-level records, sees only what a manager would see, and submits a plan and forecasts that are then played out across 200 seeded futures and three plausible versions of the organization. Decisions are scored in dollars at the stakeholders' stated valuations; forecasts are scored on the effect of each change relative to doing nothing. Ten models made 280 attempts across 14 organizations. The best model, GPT-6 Astra, improved on doing nothing by 6.8% of the no-change cost and made 77% less forecast error than a forecaster who always predicts no effect. On average, every model forecast the direction of every specified effect correctly, but models erred on the size of the same effects in the same direction, overstating a voice agent's effect on repeat contacts about five-fold. No model ever recommended holding off when doing nothing was the right choice. To test external validity, twelve models forecast 33 real organizational changes with known outcomes. Under verdict rules fixed in advance, the benchmark's forecast ranking agreed with the real-world ranking (Spearman $\rho = 0.92$, 95% lower bound 0.50). We report the result alongside an inconclusive first test, the analysis choices made after outcomes were seen, and evidence that part of the agreement reflects how well models size their uncertainty. 

---
# From Ideal to Real: Integrating Language Models into Smart Home for Human-Centric IoT Safety and Security Fuzzing 

**Authors**: Lin Li, Xinbo Ban, Chao Chen, Lei Pan, Ming Ding, Kok-Leong Ong, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22457)  

**Abstract**: Integrating IoT devices into household systems offers enhanced control over living environments. However, as these devices become pervasive in critical sectors such as healthcare, transportation, and home automation, they expose inter-rule vulnerabilities that may lead to privacy breaches, data theft, and physical harm. Traditional fuzz testing methods, which often operate in simulated environments assuming ideal human decisions, fail to account for the complexities of human interactions that can influence device operations in smart homes, potentially leading to unforeseen consequences. To reveal the impact of real human decisions and discover unexpected inter-rule vulnerabilities, we introduce an enhanced version of IoT Fuzz for smart homes by integrating large language models to emulate human decision-making processes. Our novel approach adds a layer of realism and unpredictability to the testing environment, more accurately capturing the diverse and dynamic nature of human interactions in smart home settings. 

---
# PACD: Provenance-Aware Consensus Decoding for Mitigating Cross-Modal Hallucinations in Omnimodal Large Language Models 

**Authors**: Chang Sun, Yijun Tian, Zhenwei Tang, Yuchen Deng, Yifan Simon Liu, Chenxu Wang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22456)  

**Abstract**: Omnimodal large language models (Omni-LLMs) integrate audio, video, and text for multimodal reasoning, yet remain vulnerable to cross-modal hallucinations, where one modality improperly influences predictions about another. Existing training-free decoding methods mitigate these errors through modality perturbation or relevance-based weighting. However, due to interference from irrelevant modality cues during fusion and difficulty detecting shared errors, these methods may suppress correct unimodal predictions and allow erroneous consensus to persist. To address these limitations, we propose Provenance-Aware Consensus Decoding (PACD), a training-free framework for mitigating cross-modal hallucinations. PACD comprises two complementary paths based on whether the branches reach consensus. For non-consensus cases, Relevance-Guided Candidate Arbitration identifies task-relevant unimodal evidence and selectively arbitrates between predictions using alternative candidates. For consensus cases, Provenance-Based Consensus Recovery leverages the outranking principle to evaluate the consistency and strength of layer-wise modality evidence for alternative answers, selectively resolving erroneous consensus under confidence constraints. Extensive experiments on four Omni-LLMs and five benchmarks across two tasks demonstrate the superiority of PACD. Compared to existing methods, PACD achieves up to 27.6% relative improvement over standard decoding across CMM, AVHBench, and SVHalluc for cross-modal hallucination mitigation, and an average relative improvement of 7.7% over best baseline across OmniVideoBench and FutureOmni for general audio-visual reasoning. 

---
# Think Once, Act Fast: Cost-Effective Recursive Self-Improvement for Decision Model Harness 

**Authors**: Siqi Lai, Haitao Li, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22432)  

**Abstract**: Fast System-1 models enable responsive actions but struggle with complex or unfamiliar tasks. Large language model (LLM)-based System-2 provides strong reasoning capabilities, yet repeated invocation incurs substantial latency and computational costs. To bridge this gap, we propose JevRSI, a System-1/System-2 collaboration framework that transforms LLM reasoning into transferable decision support through recursive harness improvement. A Jev-style System-1 model enables low-latency action selection, while an asynchronous System-2 LLM develops a modular harness. The harness integrates procedures, state memory, and world models to provide decision contexts spanning the past, present, and future. Utility-driven retrieval selects modules based on both contextual relevance and their utility on previous tasks. Environmental feedback guides targeted revisions and enables recursive self-improvement. Experiments across three datasets show that JevRSI consistently improves decision quality, matching or surpassing LLM agents in several settings. It preserves low-latency execution with favorable performance-cost trade-offs. Moreover, System-2 development interventions decrease as the harness evolves, amortizing System-2 costs through continuous improvement. Our project is available at this https URL. 

---
# Proceedings of the 2026 XCSP3 Competition 

**Authors**: Gilles Audemard, Christophe Lecoutre, Emmanuel Lonca  

**Link**: [PDF](https://arxiv.org/pdf/2611.22420)  

**Abstract**: This document constitutes the proceedings of the 2026 XCSP3 Competition. The results of this competition of constraint solvers were presented at FLOC (Federated Logic Conference) 2026 Olympic Games, held in Lisbon, Portugal from 20th to 29th July, 2026. 

---
# ROSMonitoring in Action: Runtime Monitoring and Enforcement for ROS 2 Actions 

**Authors**: Giacomo Paltrinieri, Angelo Ferrando  

**Link**: [PDF](https://arxiv.org/pdf/2611.22407)  

**Abstract**: Robotic applications often rely on long-running operations, such as navigation, manipulation, and inspection, which in ROS~2 are commonly implemented as actions. Unlike topics and services, actions involve a structured lifecycle comprising goal submission and acceptance, feedback, status updates, result retrieval, and cancellation. Existing versions of ROSMonitoring support topics and services, but do not reconstruct this lifecycle or expose action-level events to a runtime-verification oracle. In this paper, we extend ROSMonitoring with runtime monitoring and enforcement for ROS~2 actions. Actions are declared directly in the monitoring configuration and automatically instrumented through the underlying ROS~2 communication endpoints. The generated monitor reconstructs action-level events, correlates them through goal identifiers, maintains the state of active goals, and translates violations into protocol-aware enforcement operations. These include goal rejection, cancellation, message suppression, and protocol-compliant responses generated directly by the monitor when the original communication cannot be forwarded. We evaluate the implementation through controlled stress tests that vary the number of action invocations, workload size, and clientâ€“server topology. The results show approximately linear growth in monitoring cost with repeated action invocations under a fixed workload, while larger action workloads and a greater number of configured clientâ€“server pairs introduce additional overhead. These results characterise the scaling behaviour of the current implementation under the tested sequential workloads while highlighting the main performance limitations of the current synchronous implementation. 

---
# Symbolic Abstraction Refinement 

**Authors**: Daniel Bryce, Robert P. Goldman, David J. Musliner, Jeremy Gottlieb, John Skovbekk  

**Link**: [PDF](https://arxiv.org/pdf/2611.22404)  

**Abstract**: Autonomous systems choose actions whose outcomes are uncertain and are subject to time constraints. Probabilistic timed automata (PTAs) model exactly this, and verifying one answers a question about the system's decision-making: the maximum probability of reaching a goal, or the minimum probability of reaching a hazard, that a controller can guarantee. Verification normally works over symbolic states, each pairing a discrete location with a region of clock values, and its difficulty is that the best behavior generally varies within such a region. The PRISM verifier's most capable PTA method responds by splitting regions until each is uniform, rebuilding the transitions that reach them and propagating the split backward through the state graph of a stochastic game, a cascade of graph reconstruction that partitions regions the property never depends on. We present SYMAR, which instead makes the value function symbolic: each symbolic state carries a function from sub-regions of its clock region to probability bounds, so refinement is the appearance of new sub-regions in that function, and previously generated states are never split nor their incoming transitions reconstructed. Both sources of PTA nondeterminism, the timing and the choice of actions, are captured implicitly by the preimage computation that produces those sub-regions. Heuristic forward search then bounds how much refinement is done at all, expanding an And-Or graph from the initial state along the most promising behaviors so that effort is confined to the property-relevant fragment, guided by bounds and preferred actions from a single unrefined run of the existing method; a partial-refinement backup keeps the otherwise combinatorial sub-region count lean, and every stored interval provably contains the true probability. On the PRISM PTA benchmarks, whose subproblems recur in autonomous systems (distributed agreement under contention, leader election, and self-configuration without central coordination), SYMAR is up to 108x faster, solves an instance on which the existing method exhausts the time limit, and is markedly less sensitive to the deadline constant, resolving one instance in a 37-node search graph, unchanged as the deadline triples, where the existing method needs 86,380 abstract states and 73 refinement rounds. 

---
# AutoE2E: Don't speak - Language free end-to-end autonomy A Real-Time Driving Model for L2++ ADAS and L4 Robotaxis 

**Authors**: Muhammad Zain Khawaja, Zaynab El Mawas, Ryota Yamada, Gabriela Cordova, Arseni Lysak, Intisar Chowdhury, Bharat Jayaprakash, Apostolos Georgiadis, Shri Harsha Adapala, Jatin Gautam, Christoph Bank, Alejandra Solarte Uscategui, Atanasko Boris Mitrev, Hassan Ali Ansari, Nishad Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22374)  

**Abstract**: AutoE2E is an open-source end-to-end driving model developed by the Autoware Foundation Robotaxi Working Group. It introduces a temporal bird's-eye-view (BEV) architecture that combines multi-camera perception, HD-map information and route conditioning to predict future vehicle motion. The evaluated model uses a pretrained camera-BEV backbone, temporal feature fusion and a lightweight recurrent trajectory planner. It contains approximately 79.9 million parameters, of which approximately 1.45 million are trainable in the evaluated configuration. The v1.0 release provides model checkpoints, training and evaluation code, and DataModelConsole, a dashboard for visualizing and inspecting model experiments. On the KITScenes validation benchmark, the fine-tuned checkpoint achieves an average displacement error (ADE) of 0.745 m and a final displacement error (FDE) of 2.020 m at a three-second prediction horizon. The paper also reports preliminary real-vehicle research conducted with the Technical University of Munich's EDGAR platform, alongside ongoing collaboration with the Karlsruhe Institute of Technology. AutoE2E v1.0 is an early open research release. Its quantitative evaluation is open-loop, and the model has not yet demonstrated validated closed-loop autonomous driving. The release is intended to provide the autonomous-driving community with an inspectable baseline for further research, experimentation and collaborative development. 

---
# Seeing What Still Holds: How Waiting Control Changes Cooperative Recovery 

**Authors**: Kangjie Huang, Yucheng Xin, Xinchun Wang, Dong Liu, Yan Fan, Shuming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22368)  

**Abstract**: Changing the holding reference during a fixed-duration wait can reverse the ranking of visual recovery methods. We introduce AuditEmbodied, a paired execution and trajectory evaluation framework that independently varies holding references and waiting duration while keeping recovery executors frozen. A 960-execution factorial on 20 initialization seeds compares CAD-feedback and landmark-based recovery from identical post-wait physical states within each holding and waiting condition. Protected completion requires successful recovery under the declared execution constraints. At a 1 s wait with registered wrist references, their protected completion counts change from 6/40 versus 19/40 under registered holding to 34/40 versus 22/40 under position-only holding. Trajectory analysis localizes the main separation to grasp formation: small observation-time target residuals precede large tracking errors. Joint diagnostics and controlled interventions characterize persistent tracking constraints. On fresh seeds, gradual position reanchoring improves CAD recovery's long-wait completion by 42.5 percentage points and reduces mean complete recovery time by 6.06 s relative to registered holding. Additional evaluations with pretrained policies, handover, and a mass-matched geometry characterize holding-reference sensitivity across executors and physical regimes. These findings establish holding reference as an explicit evaluation condition for comparing visual recovery performance. 

---
# HealthCLBench:Evaluating Context Learning in Health Conversations through Controlled Case Clusters 

**Authors**: Ting Yang, Zijing Hong, Chengjie Wang, Peng Wei, Chunxiao Guo, Yun Yue, Xurui Li, Jinqun Guan, Zixiao Chen, Ant Group  

**Link**: [PDF](https://arxiv.org/pdf/2611.22362)  

**Abstract**: Complex health conversations require answers grounded in individual information and currently valid evidence. We evaluate health context learning: acquiring, integrating, and applying relevant context during inference with fixed model parameters to answer health questions and respond appropriately to changes in facts, evidence, or task goals. To our knowledge, Chinese-language medical benchmarks have yet to systematically combine explicit contextual difficulties, controlled case variations, and member-specific medical criteria for open-ended answers. We introduce HealthCLBench, comprising 328 controlled clusters, 1,103 cases, and 4,931 rubric criteria across medical consultation and everyday health management. Members share backgrounds while evidence, patient states, or task goals change; each has its own medical scoring requirements. A 70-person medical team annotates and reviews the simulated cases and their rubrics. Human review of 700 full-context responses from the original evaluation yields 90.5% pooled agreement with the original automated judgments at the rubric-item level. We report full-context and query-only results for seven general-purpose models and Ant Afu. Across the eight systems, full- query differences range from 11.83 to 31.76 case points and from 10.87 to 38.97 geometric cluster points. GPT-6-Astra leads both full-context metrics at 75.03 and 60.42, followed by Claude-Opus-5-5 and Ant Afu. Cross-condition differences describe score changes under the reported evaluation settings. Response comparisons reveal failures in checking historical information, expressing uncertainty when warranted, and ensuring recommendations respect required conditions. The resource supports model comparison and targeted follow-up evaluation. 

---
# Reliable Evaluation of AI Agents for Investment Research: A Framework for Performance and Validity Assessment 

**Authors**: Dongnuan Tian, Yan Yu, Yangfan Liu, Wenjun Tang, Junting Kan, Xiang Wang, Sicong Liu, Yangyang Zhang, Shuanghong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.22338)  

**Abstract**: We consider the problem of evaluating AI agents undertaking investment-research tasks, where agents interact with external information sources and computational tools to produce financial analyses and research reports. An important challenge is that the quality of an agent's final output does not necessarily reflect the quality of its execution process, while the reliability of the resulting assessment also depends on the validity of the evaluation procedure. Furthermore, investment-research tasks differ considerably in their requirements for numerical accuracy, information retrieval, financial analysis and supporting evidence, making it difficult to assess agent performance using a uniform set of criteria. We propose a domain-oriented evaluation framework that distinguishes outcome quality from execution quality and accommodates these differences through task-specific criteria, deterministic validators and independent large language model judges. We also introduce an evaluation-validity mechanism that examines the integrity of evaluation inputs, reference information, supporting evidence and grading outputs, with expert review used to resolve identified deficiencies and significant grading disagreements. Numerical results show that the proposed framework can distinguish differences in agent performance across evaluation dimensions, while identifying potential reliability issues in the evaluation process. The associated implementation and evaluation materials will be made available on GitHub. 

---
# Multistage Contextual Optimization for Vehicle Repositioning 

**Authors**: Jinyu Zhang, Di Huang, Zhiyuan Liu, Nikolas Geroliminis  

**Link**: [PDF](https://arxiv.org/pdf/2611.22330)  

**Abstract**: Ride-hailing platforms reposition idle vehicles to improve spatial supply-demand balances. This problem is prediction-driven and multistage: at each stage, the platform forecasts future demand, executes current decisions, and uses the resulting state to initialize the next stage. We formulate a multistage linear program (LP) with a macroscopic fundamental diagram (MFD) linking accumulation to trip completion. We then develop a multistage contextual optimization approach that uses decision performance to train the prediction model. The learning problem combines multiple coupled stages, context that depends on earlier decisions, and predictions that affect the objective, the constraint matrix, and the right-hand side of the LP. We derive the exact gradient, which has three parts: parameters change forecasts and hence decisions; earlier decisions change the current decision through the inherited state; and this state also changes forecasts as context. An equivalent adjoint uses reverse vector-Jacobian products, avoiding full solution and state-parameter Jacobians. Finally, forward propagation solves many stage LPs one after another, which makes training slow. We propose a parallel block evaluation algorithm that solves blocks of consecutive stages in parallel and repeatedly feeds each block output to the next block. We prove that the algorithm recovers the serial trajectory within a finite number of passes and that its gradient error remains bounded when it stops early. On New York City ride-hailing data, multistage contextual optimization yields 7.7% to 9.1% more business benefit than the same predictor trained by mean squared error, outperforms all compared baselines from five paradigms, and even outperforms realized demand fed to the same policy. With eight workers, parallel block evaluation shortens training by a factor of 1.86 without loss of decision performance. 

---
# EviDARL: Evidence-Centric Dual-Agent Reasoning via Alternating Reinforcement Learning for Multi-Table Question Answering 

**Authors**: Dunhan Li, Leilei Gan, Lirong Gao, Wenyan Fan, Yongwei Wang, Sheng Guo, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22310)  

**Abstract**: Despite advances in LLM-based table reasoning, most studies focus on single-table settings. Multi-table QA requires multi-hop reasoning over complex table relationships and evidence dispersed across long contexts, often without verifiable intermediate traces. To address these challenges, we introduce EviDA, an evidence-centric dual-agent framework comprising an Evidence Construction Agent and a Reasoning and Answering Agent. The Evidence Construction Agent generates and executes table-manipulation programs over complete tables to produce question-specific evidence and verifiable intermediate traces, simplifying subsequent reasoning. The Reasoning and Answering Agent then performs multi-hop reasoning over this evidence to derive final answers. We further introduce EviDARL, which combines agent-specific fine-grained rewards with parameter-efficient alternating reinforcement learning to improve collaboration between the agents. By optimizing each agent in turn while holding the other fixed, EviDARL enables bidirectional adaptation between evidence construction and answer reasoning. Experiments on MMQA and TQA-Bench demonstrate that EviDARL substantially outperforms competitive open-model and multi-table reasoning baselines under constrained model and training budgets. 

---
# Efficient Wide-Beam Generative Retrieval with Model-Native Trie Decoding and Shared-Context Attention 

**Authors**: Vedant Jhaveri, Arashpreet Singh Mor, Satyam Kumar, Alex Tsun, Qing Lan, Shaunak Mishra, Sriram Vasudevan  

**Link**: [PDF](https://arxiv.org/pdf/2611.22309)  

**Abstract**: Generative retrieval (GR) predicts catalog entries as short semantic identifiers (SIDs). Trie-constrained beam search keeps these identifiers valid, but wide beams expose two serving costs: vocabulary-wide score tensors that are mostly discarded, and repeated attention over the same prompt. We present Trie-Head, a model-native output interface that stores the identifier trie as validated GPU tensors. Compared with a Python custom logit processor (CLP), it reduces CPU--GPU coordination and memory traffic by avoiding Python constraint callbacks and vocabulary-wide candidate buffers. It scores compact codebook candidates and streams the full-vocabulary normalizer, preserving the original model's valid-token scores in real arithmetic without retraining. Shared-context attention complements this output path by reusing prompt key/value blocks across beams while preserving their private decode tails. We implement both mechanisms in SGLang and evaluate candidate sourcing for LinkedIn Hiring Assistant and qualification retrieval for the Qualification Retrieval Platform. The combined system improves closed-loop throughput by up to 5.57x over the CLP baseline and up to 2.63x over BF16 FlashRec at a common beam width of 200 on the Qualification Retrieval Platform workload. 

---
# Learning to Edit IFC Building Models: Verifiable Checkers as the Training Signal for Dependency-Aware Editing Agents 

**Authors**: Jiaxin Zhang, Yao Luo, Yuze Dan, Yunqin Li, Bowen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22307)  

**Abstract**: Editing a building information model in the Industry Foundation Classes (IFC) schema requires propagating changes to dependent elements while preserving unrelated entities. Existing language-model agents for IFC editing use prompted models without learning from edit outcomes. Target-file comparisons do not specify which alternative edits are allowed or which unrelated entities must remain unchanged. We present IfcGym, a CodeAct environment in which a language model edits IFC through executable code and four checkers verify validity relative to the input model, intent, dependency and preservation against a task specification. Their conjunction is strict success. IfcEdit distils verified teacher episodes into a Qwen3.5-9B student through supervised fine-tuning (SFT), then applies Group Relative Policy Optimization (GRPO) with strict success as the reward. On IfcEditBench, 1,990 editing tasks on 13 held-out real projects, the final student reaches 86.78% strict success, from 79.15% after SFT. The paired confidence interval for the difference from its teacher includes zero; the same recipe on a 27B model reaches 91.76% and exceeds the teacher. On the public benchmarks BIM-Edit and BIBIMBAP the final student scores 55.19 and 54.50. On IfcEditBench, the students outperform all prompted models except the teacher under matched tool access; both evaluated frameworks also score lower. Rewarding all four checkers yields 4.83 points more strict success than rewarding intent and validity alone. 

---
# MCTS Is Counting the Wrong Thing: Learning Exploration Counts for LLM Reasoning 

**Authors**: Xuan Zhang, Zhijian Zhou, Zicheng Hu, Yan Feng, Hongyan Hao, Yuan Qi, Chao Qu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22301)  

**Abstract**: Monte Carlo tree search (MCTS) has powered landmark successes in reinforcement learning, yet its gains in LLM reasoning can remain modest despite substantial search effort. We identify an overlooked bottleneck: what MCTS counts. Standard UCT assigns independent visit counts to textual branches, allowing different formulations of the same reasoning strategy to repeatedly receive exploration bonuses. Our theoretical analysis shows that this count fragmentation can impair exploration efficiency, as quantified by increased cumulative regret. To address this problem, we introduce Learned Pseudo-Count MCTS (LPC-MCTS), which changes only the exploration statistic in UCT. A lightweight Coin-Flipping Network head on the policy's hidden representations estimates representation coverage, enabling related branches to share exploration evidence. Across 5 mathematical-reasoning and 7 search benchmarks, LPC-MCTS consistently outperforms strong MCTS baselines. Holding each LPC-MCTS-trained policy fixed, replacing standard MCTS counts at inference improves average mean@16 accuracy across 5 mathematical benchmarks by 3.79 percentage points on Qwen2.5-7B and 2.71 points on Qwen3-4B. Applying LPC-MCTS during both training and inference yields gains of 4.96 and 4.37 points over standard MCTS, respectively. Together, our theory and experiments show that changing what MCTS counts can substantially improve its exploration efficiency and reasoning performance. 

---
# Harnesses That Improve Themselves: A Unifying Framework for Self-Improving Agent Harnesses 

**Authors**: Deep Shah, Nehal Kathrotia  

**Link**: [PDF](https://arxiv.org/pdf/2611.22299)  

**Abstract**: Whether a foundation model succeeds as an autonomous agent depends as much on its runtime software as on its weights. This runtime, the agent harness, manages context and memory, steers multi-turn control, executes tool actions in sandboxes, and verifies work before stopping. Because models differ in how they use tools and how they fail, a static harness tuned for one model or task often stalls, wastes tokens, or regresses on another. This bottleneck has driven a shift from prompt tuning to self-improving agent harnesses: systems that inspect their own execution traces, separate runtime bugs from model limits, and rewrite their own code. We present a unifying architectural and design framework for self-improving agent harnesses. The framework pairs four editable runtime components (context and state, agent control, action interfaces, and verification and recovery) with a four-stage loop that diagnoses failures, proposes code patches, validates candidates, and deploys updates. Using this framework, we show how optimizers trace errors across long trajectories and discover reusable runtime mechanisms such as pre-stop verifiers, loop breakers, and persistent memory. We further show why full-harness rewrites overfit or break across modules, and how modular updates with validation gates generalize across models while bounding token cost. To keep self-modification safe, we establish practical defenses against recurring failure modes including prompt bloat, benchmark gaming, and evaluator tampering. Finally, experiments on SWE-bench Lite validate this framework, showing that gated modular evolution outperforms prompt reflection and full-harness rewrites across models and repositories at a fraction of the token cost. 

---
# Streaming On-Policy Distillation: Decoupling Three Clocks for Stage-Wise Supervision Consumption 

**Authors**: Yun Zhu Da Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22298)  

**Abstract**: In on-policy distillation (OPD), response completion, supervision delivery, and parameter updates need not share the same granularity. We develop streaming OPD by decoupling these three clocks: a response repeatedly delivers newly generated segments while generation continues, retaining the full causal prefix at each delivery and computing loss only on new positions. A consumption protocol then organizes supervision budgets, position selection, staleness handling, and update scheduling, allowing control over quantity, coverage, and version lag across training stages. In GSM8K experiments distilling Qwen3-4B into Qwen3-0.6B with VERL, supervised-budget matching reduces the early gap between best test accuracies over seven common candidates from 4.55 to 2.42 points. Synchronous continuation from the same checkpoint reaches 73.39% best test accuracy over seven common candidates at u=384,...,576, versus 74.30% synchronous and 71.95% budget-matched (differences -0.91 and -2.35 points); an unpaired extended record reaches 76.12%. Freshness filtering and disagreement-overlap selection improve local peaks but do not establish sustained advantages. A system prototype shows lower data-supply waiting times without establishing an equal-update speedup. Quality recovery and supply-side benefits are assessed separately; end-to-end gains must account for continuation cost. 

---
# SpecSearch: Towards Hardware-Aware Efficient Speculative Decoding 

**Authors**: Selin Yildirim, Ali Zeyrek, Deming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.22296)  

**Abstract**: Speculative decoding (SD) accelerates large language model (LLM) inference, but its optimal configuration depends jointly on workload characteristics, GPU hardware, and available memory. For example, on the RTX~5090, the state-of-the-art speculative drafter DFlash achieves $3.50\times$ the throughput of autoregressive (AR) decoding at low concurrency, but only $0.76\times$ at high concurrency with short outputs. We present SpecSearch, a hardware-aware search engine that jointly optimizes decoding strategy, draft length, and precision across workload--GPU pairs. Its analytical cost model captures memory and compute constraints to identify feasible configurations and narrow the search space for throughput optimization. Guided by this model, calibration measurements are collected from multiple GPUs. Using the measurements, a Jev-based predictor learns to select high-throughput decoding configurations on held-out GPUs without requiring deployment-GPU calibration. The predictor is parameter-efficiently trained on calibration data, while adapting a frozen Qwen3-4B using LoRA adapters and a linear correction layer. Across NVIDIA RTX~5090, H100, L40 and AMD MI300X GPUs, SpecSearch with full calibration achieves an average of $2.01\times$ the throughput of AR and $1.36\times$ the throughput of SD, both served by vLLM. The proposed Jev-based predictor closely matches this search performance with $1.30\times$ throughput improvement over SD with vLLM, while eliminating the need for held-out GPU calibration. Additionally, it retains $97.36\%$ of the throughput achieved by SpecSearch that utilizes full calibration. 

---
# ConfGuard: a physics-grounded reference monitor for predictive-maintenance large-language-model agents 

**Authors**: Arunkumar V, Gangadharan G. R., S. Senthilkumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.22288)  

**Abstract**: Large-language-model agents now close the industrial predictive-maintenance loop, reading sensor windows, retrieving procedures and dispatching work orders. Two attack channels meet there: injected text in a retrieved document can hijack the decision, and a bounded sensor perturbation can corrupt the remaining-useful-life estimate behind it. Recent agent defenses police data provenance and tool privilege but cannot tell whether a prognostic quantity is physically plausible. ConfGuard is a reference monitor with a physics-grounded policy. Its conformal monitor tests each prediction, anchored to the asset's admitted history, against degradation physics: capped remaining life never rises and falls at most one cycle per cycle. Calibrating its threshold over whole asset trajectories lets the operator set the expected share of a clean asset's windows the running monitor escalates. A physics-policy validator checks the agent's maintenance-urgency decision against the decision a deterministic policy derives from the same interval. On all 10,196 C-MAPSS test windows the monitor cuts a covert attacker's reach on healthy assets from 0.529 to 0.046 and closes suppression of failing assets (0.151 to 0), escalating 0.027 of clean windows against a 0.05 target, while the fixed physical range it replaces leaves both attacks open. Behind a Qwen3-14B planner the composed monitors hold urgency-class injection to 0.00 and the universal sensor attack to 0.12, against 0.29 and 0.54 for a defended prompt. An adaptive attacker defeats a Mahalanobis detector on 0.504 of windows and adversarial training on 0.485. We report an attack class the projection misses. 

---
# Learning Regions of Attraction for Transverse Feedback Linearization under Controller Singularities 

**Authors**: Abdullah Waqar Malik, Hamza Tariq, Adeel Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2611.22286)  

**Abstract**: Transverse feedback linearization (TFL) achieves path following by exactly linearizing the error dynamics, but its controller is undefined on a singular set, such as zero speed for a unicycle or zero thrust for a quadrotor. The constrained region of attraction, the set of initial states from which the closed loop converges to the path without violating state or regularity constraints, determines where the controller can be safely activated. Exploiting the exact linear error dynamics, we show that this region is the largest forward invariant subset of the admissible set. We then decompose its state-constrained Zubov value function into an analytically known quadratic cost-to-go and a safety value function. We learn only the latter, by fitted value iteration without infinite-horizon trajectory labels. The resulting estimate combines an inner layer verified by interval branch and bound with a learned outer layer tested on samples. On a unicycle following a circle and a sinusoid and on a planar quadrotor, it covers on average 72.0%, 83.4% and 19.1% of the simulated region of attraction. In comparison, the quadratic cost-to-go and two physics-informed neural network (PINN) solvers of Zubov's equation cover at most 11.3%, 4.0% and 0.0% on average. Code and animations are available at this https URL 

---
# Towards Physics-Constrained Rectified Flow for Climate Downscaling: What Projections Can Certify? 

**Authors**: Arun Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2611.22283)  

**Abstract**: Statistical downscaling restores the regional detail behind flooding and crop stress, but diffusion and flow-matching generators can produce sharp fields that break the conservation laws of their coarse input. Physics-Constrained Rectified Flow (PC-RF) is a conditional flow-matching downscaler. Its training penalties target wind divergence, negative precipitation, and mass drift from the coarse total, and a projection after sampling removes the last two exactly. On one corrupted synthetic field with no trained network, the projection drives the Non-negativity Violation Rate (NVR) from 0.236 and the Mass-Conservation Error (MCE) from 0.184 to zero, and the penalties lower the Divergence Error (DE) by about 30%. On ERA5 the projection gives PC-RF and RF-base, the penalty-free rectified-flow baseline, an NVR of exactly zero and an MCE of zero up to float32 rounding. The penalties add no perceptual gain that we can separate from training budget or run-to-run variation. At a matched budget on 2019-2021, PC-RF has no edge over RF-base (SSIM 0.939 vs. 0.942, PSNR 31.81 vs. 31.79, CRPS 0.771 vs. 0.772). Our unprojected CorrDiff reimplementation beats both flow models on PSNR (28.38) and CRPS (1.092) on 2020, yet 45% of its ensemble-mean precipitation cells are negative. On ERA5 a sign error in the divergence operator made that penalty and DE act on a deformation-like term. The fine reference is interpolated from ERA5 and accumulates precipitation over a longer window, so the perceptual scores cannot establish downscaling skill. NVR and MCE need one pass and no fine reference, so we argue they should run first as pass-or-fail checks, with DE as a diagnostic. 

---
# SeasonRainBench: Global Seasonal Precipitation 

**Authors**: Qiuyan Sun, Mengxuan Chen, Haoyuan Liang, Guowen Li, Zhen Huang, Jianhao Lin, Juepeng Zheng, Haohuan Fu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22281)  

**Abstract**: Accurate seasonal precipitation forecasts help societies prepare for drought, flooding, and crop failure months in advance. While data-driven methods have made rapid progress in weather and subseasonal forecasting, operational seasonal precipitation prediction still relies on numerical ensemble systems. These systems are computationally expensive and exhibit biases and drift that increase with lead time, motivating more efficient data-driven alternatives. Yet this transition faces three linked challenges: comparing data-driven forecasts with operational ones requires (1) a global benchmark that does not exist, scoring them requires (2) metrics that capture operational priorities beyond anomaly correlation (ACC) and RMSE, and training them must (3) rely on a single, short observational record. To address the first two challenges, we introduce \BenchName, the first such benchmark, covering monthly precipitation forecasts 1--5 months ahead. It pairs global $1^\circ$ satellite precipitation with oceanic and atmospheric predictors, includes seven operational centers, and provides 310,944 simulated months from 52 climate models for training. Beyond standard scores, we introduce two operationally motivated metrics for anomaly strength and monthly rain-belt position. To address the third challenge, we develop \MODEL, which learns from simulations what the record lacks and transfers it to Earth through three modules: SimForced aligns the climate background, SimMemory transfers a history encoder pretrained on simulations, and SimAnalog supplements Earth labels with simulated analog outcomes. Extensive experiments show that \MODEL{} improves ACC, CRPS, anomaly strength, and rain-belt position by 12.0\%, 3.6\%, 2.9\%, and 0.3\% over the strongest operational or data-driven model, opening a path from simulated worlds to Earth system prediction. 

---
# DACE: Diversity-Driven Adversarial Co-Evolution for Robust LLM Safety Alignment 

**Authors**: Peng Yu, Xiaoyu Wen, Zhida He, Ziyuan Zhou, Han Qi, Shao Zhang, Qiaosheng Zhang, Ying Wen, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22278)  

**Abstract**: Ensuring robust safety alignment of large language models (LLMs) is increasingly difficult as adversarial attacks evolve and outpace alignment pipelines built on pre-collected data. Existing co-evolutionary frameworks let the defender chase a moving attacker, yet their training dynamics expose two compounding pathologies: \emph{attack strategy collapse}, where the attacker overfits to a narrow set of high-reward rewrites, and \emph{defense adversarial forgetting}, where the defender loses competence against earlier attacks as the attack distribution drifts. Existing remedies are partial: attacker-side diversity rewards score textual novelty rather than strategy novelty, and defender-side replay relies on discrete judge scores that ignore estimation uncertainty, response stochasticity, and defender non-stationarity. We introduce \textbf{\method}, a diversity-driven adversarial co-evolution framework that targets both pathologies jointly. On the attacker side, \method couples an explicit two-dimensional strategy space (risk category $\times$ attack style) with a \emph{normalized marginal coverage gain} reward, providing a bounded, non-vanishing exploration signal at the strategy layer. On the defender side, \method maintains a unified \emph{Bayesian adversarial replay pool} whose Beta--Bernoulli threat posteriors are refreshed by time decay and sampled via Thompson sampling, anchoring defender training to an evolving threat landscape. Across standard safety, automated-attacker, and general-capability benchmarks, \method improves robustness to out-of-distribution attacks while preserving general capabilities, and yields broader attacker strategy coverage. 

---
# Fidelity Regret: Measuring Simulator Misspecification in Verifiable Design 

**Authors**: Noah Flynn  

**Link**: [PDF](https://arxiv.org/pdf/2611.22263)  

**Abstract**: RLVF assumes the verifier evaluates the quantity of interest. In genetic-circuit design, a topological verifier can establish that a program is a valid circuit computing the target function, whereas ranking its physical performance requires a simulator. We measure fidelity regret, the reference simple regret incurred when a cheap simulator selects the design, over a 21,362-design combination subspace spanning 5 measured gate libraries and 3 organisms. Exact ordered-assignment checks retain positive regret in every tested cell while shifting magnitudes in both directions. Mean normalized regret is 0.369 over 72 cells and only 6 have zero regret. The discrepancy depends strongly on the reference axis: the yeast library SC1C1G1T1 has mean regret 0.113 for cytometry quality and 0.801 for growth burden, and every evaluated library--circuit-size pair has positive burden regret. Topological verification ties all valid gate assignments, while an oracle whitelist subject to retaining at least 25% of the design space leaves positive regret in 30 of 52 cells. Design-space optimization and feedback-guided search with frozen language models reach the high-regret regions; on a matched zero-regret control, additional optimization instead reduces regret. At a matched modeled-cost budget, mitigation is budget- and library-dependent. An in-sample fixed rule spending $\alpha{=}0.34$ of the budget re-scoring top cheap-simulator designs achieves mean excess regret 0.016, compared with 0.305 for cheap-only and 0.034 for reference-only selection, improving on cheap-only in 330 of 360 cells. More broadly, any workflow optimized through a simulator or digital twin can satisfy its verifier yet perform poorly on the intended physical objective. Fidelity regret provides a transferable template for detecting this gap before optimization and comparing budget-aware remedies across scientific domains. 

---
# A Visual Information Extraction System Using Multimodal Large Language Models for Scanned Cognitive and Behavioral Assessments 

**Authors**: Enshuo Hsu, Linh Nguyen, Sunyang Fu, Heath Goodrum, Sandra Stansberry, Kirk Roberts, Hongfang Liu, Yen-Chi L. Le  

**Link**: [PDF](https://arxiv.org/pdf/2611.22252)  

**Abstract**: Objective. Paper-based assessments are common in patient care, but handwriting and scanning artifacts make these forms difficult to process at scale. This study proposes a modular, multimodal LLM-based system with human oversight for extracting fields and scores. We conducted a qualitative analysis of the use of LLMs for hand-drawing analysis.
Materials and Methods. The system processes cognitive assessments (i.e., MMSE, MoCA, Mini-Cog) and the NICHQ Vanderbilt Assessment using two modules: 1) a form classifier that assigns a scanned page to a form type, followed by a corresponding 2) score extractor that extracts fields and scores into a schema. A human-in-the-loop design utilizing dual-model validation engages human annotators when two LLMs disagree, limiting manual review to challenging pages only.
Results. End-to-end accuracy for cognitive assessments was 88%, with most form types exceeding 90%. For the NICHQ Vanderbilt, overall accuracy reached 97.5%, with all form types above 96%. The human-in-the-loop design further improved cognitive assessment accuracy to 97.5% while reducing manual work by 63.8%. In the Mini-Cog Clock Drawing task, qualitative analysis showed that 62.3% of all extraction errors stemmed from ambiguous drawing, in which cases the LLM provided reasonable readings.
Discussion. Compared to current manual data extraction, our LLM-based system reduces processing time by 10 to 20 times and costs by over 12 times. The modular design allows seamless expansion to additional form types.
Conclusion. Our system provides an accurate, scalable, and cost-efficient solution for structuring data trapped in handwritten paper forms for clinical decision support. 

---
# Multi-Agent Memory Inconsistency: Where Factual Conflicts Disappear in LLM Agents 

**Authors**: Hejia Zhang, Zhongming Yu, Yimin Ding, Wei-Ting Chen, Hengjia Yu, Wentao Ni, Yijun Liu, Cong Xu, Xin Chen, Matrix Yao, Fan Zhao, Jishen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.22245)  

**Abstract**: Multi-agent delegation lets LLM systems distribute evidence collection across workers with separate contexts. This creates a consistency challenge: facts retrieved by workers may be missing from the evidence available to the lead for a joint conclusion. We introduce a benchmark and evidence-flow profiler for studying multi-agent memory inconsistency through controlled factual conflicts in Harvey LAB document-review tasks. Pooled across three models with comparable generation spending, requiring delegation increases retrieval of both conflicting claims from 50.2% to 68.7% relative to single-agent execution, yet conflict detection falls from 16.2% to 13.2%. Within this setting, workers report both claims for only 35.2% of retrieved pairs, and pooled matched comparisons show higher detection than single-agent execution when one worker reads both claims but lower detection when the claims are split across workers. Worker-to-lead handoffs are a central target for improving conflict detection in multi-agent LLMs. 

---
# Context Has a Price: Adding and Removing Context for LLM Coding Agents under Prefix Caching 

**Authors**: Zerui Yang, Miaomiao Dong, Quan Xue, Chaorui Zhang, Xuanyu Zheng, Xiyuan Wang, Haolin Ruan, Xueyan Niu, Guangcheng Song, Weixi Zhang, Wei Han, Bo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2611.22238)  

**Abstract**: Coding agents repeatedly send their conversation history to a language model. Adding useful code may reduce search, while removing old tool outputs may shorten later prompts. But do these changes reduce the cost of solving a task? We study this as a measurement problem in more than 20,000 runs of open models on SWE-bench Verified, SWE-bench and Loc-Bench. We account separately for uncached input, cached input and output tokens. Three findings emerge. First, in our main evidence-package experiments, retrieved code reduces read and search commands by 18--34%, but increases cost by 33--42% when cached input costs one tenth of uncached input, without a measurable improvement in resolution. Second, removing old observations can either save money or increase cost: the same approach saves 19--36% without a cache discount but increases cost by 38--116% with a tenth-price cache. Third, shorter prompts do not guarantee cheaper runs because agents may respond by taking more steps. We also find that concurrent experiments can share cached prefixes and distort cost comparisons. These results show why context-management methods should be evaluated using both token prices and the agent's subsequent behavior. 

---
# The Fragility of Reasoning-Importance Attribution for Pruning Language Models 

**Authors**: Himanshu Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2611.22234)  

**Abstract**: Attribution-guided pruning scores components by their causal contribution and protects the important ones, and it is increasingly applied to compress reasoning language models without destroying their chain of thought. Its reliability rests on an unexamined assumption: that reasoning importance can be faithfully measured. Across four small reasoning models, we show this measurement is fragile in three ways, each of which changes pruning decisions. First, scoring final-answer tokens, the natural choice, saturates in every chain-of-thought model we test (answer-token negative log-likelihood, NLL, of $10^{-4}$--$10^{-3}$) because the model states its answer inside its own reasoning; guiding head pruning with this target costs 27.6 and 32.2 accuracy points relative to scoring reasoning tokens on the two models we power with five seeds, while perplexity barely moves and even reverses sign at 7B. Second, causal importance is not localizable to single units: single-neuron effects sit at the measurement floor, their cross-split ranking is noise, and they cancel: a group's ablation effect disagrees in sign with the summed effects of its members in 90\% of groups. Reproducible importance emerges only at groups of $\sim 100$ neurons. Third, contrastively isolated ``reasoning-specific'' importance is not separable from generic importance, and protecting it is worse than protecting raw importance. We additionally pre-registered the hypothesis that distillation-induced concentration of importance predicts the pruning cost; it is not supported, and since our grid confounds concentration with scale, we report it as unresolved rather than reversed. The result is a short set of measurement rules: score reasoning tokens, attribute at coarse granularity, and evaluate with accuracy rather than perplexity. 

---
# Fine-Grained Japanese Clinical Named Entity Recognition: A Controlled Comparison of BERT and Open-Weight Large Language Models 

**Authors**: Feng Han, Ayako Yagahara, Kenji Hirata, Atsushi Yoshida, Kohsuke Kudo  

**Link**: [PDF](https://arxiv.org/pdf/2611.22219)  

**Abstract**: Background: Generative large language models (LLMs) offer an alternative to encoder-based clinical named entity recognition (NER), but their performance on fine-grained Japanese extraction tasks and sensitivity to generation settings remain uncertain. Methods: We compared two fine-tuned BERT-CRF baselines with nine open-weight LLMs on iCorpus, which defines 113 entity categories. The evaluation used 433 sentences from 36 test documents; 1,672 sentences were used for training. LLMs were evaluated using zero-shot prompting with and without JSON Schema constraints and three-seed QLoRA fine-tuning. Fine-tuned models were assessed under different output-token limits, including 900 and larger model-specific limits of 2,600 or 4,096 tokens. Performance was measured using sentence-level label-surface matching with relaxed and exact-surface F1. Results: The highest zero-shot relaxed F1 was 0.337, compared with 0.836-0.841 for the BERT baselines. At the larger output limits, all eight fine-tuned LLMs with at least 13B total parameters exceeded both BERT baselines (relaxed F1, 0.882-0.914; exact-surface F1, 0.840-0.880). Across three-seed means, increasing the output limit improved relaxed F1 by 0.052-0.085, alongside higher recall and fewer length-limited outputs. JSON Schema constraints changed relaxed F1 by -0.014 to +0.008 among seven models outside the two reasoning-oriented configurations, while medical-general model pairs showed no consistent advantage for medical specialization. Conclusions: On this historical Japanese case-report benchmark, task-specific fine-tuning combined with adequate generation capacity was associated with stronger LLM extraction performance. Output-token limits materially affected comparisons with BERT, whereas enforcing output structure alone did not overcome low zero-shot accuracy. 

---
# RAG in a Trenchcoat: When Minimal Memory Is Enough for Agentic Systems, and When It Isn't 

**Authors**: Jingyu Liu, Zongze Li, Zach Xu, Zhanhui Zhou, Tahseen Rabbani, Dawn Song, Ce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22179)  

**Abstract**: Agentic memory research has accumulated extraction, supersession, reflection, and graph synthesis on top of retrieval-augmented generation. However, we found that almost none of those primitives earn their cost in our tested memory benchmarks: we show this with BRIDGE, a deliberately minimal RAG memory with chained observation notes, hybrid BM25 with cosine retrieval, and chronological rendering, which reaches parity within bootstrap CI with the prior SOTA on LongMemEval-S and BEAM, and beats the best memory baseline on EverMemBench by 5.7-6.3 points at matched reader. It runs at ~5-7K reader-prompt tokens per query, using a single LLM call for each ingestion or query. We argue that this parity is itself the result: standard benchmarks score what memory retrieves, not what memory enables when the agent acts. To test the latter, we introduce five adversarial probes (Rule Application +-, Pattern Induction Style / Procedural, Dynamic Multiturn) judged at generation time by a cross vendor verification chain, where each query can only be solved by applying a remembered rule or pattern it never mentions. Across 5 probes x 20 scenarios, no architecture wins universally: atomic fact extractors (LangMem) win rule application, while BRIDGE's block-summary memory wins pattern induction. This result holds across four domains, three graders, and three reader families. Each design fails where the other wins, for reasons a recall benchmark cannot see: atomic-fact extractors discard the form a demonstrated pattern takes, while block-summary memory retrieves the rule but the reader does not act on it, a gap that presenting the rule more prominently largely closes. What separates memory designs is not what they retrieve but what they enable an agent to do, and that is what agentic memory should be evaluated on. 

---
# CorrectCast: Selective Multimodal Context for Scale-Aware Residual Correction in Precipitation Nowcasting 

**Authors**: Henan Wang, Shengwu Xiong, Yifang Zhang, Pengfei Duan  

**Link**: [PDF](https://arxiv.org/pdf/2611.22178)  

**Abstract**: Precipitation nowcasting models suffer from errors in intensity, location, and spatial structure. Addressing these errors by retraining nowcaster backbones can be computationally expensive and requires architecture-specific modifications. To overcome these limitations, we propose CorrectCast, a decoupled residual correction framework that refines completed nowcasts while keeping the base nowcaster frozen. By operating on predicted precipitation fields, CorrectCast requires training only a lightweight corrector, reducing computational overhead while providing a unified correction interface across different nowcasting models. Conditioned on the base nowcast and historical quantitative precipitation estimation (QPE) and precipitable water vapor (PWV) observations, the corrector selects multimodal context and employs scale-aware modulation to estimate residual corrections. On the Mississippi River Basin (MRB) and contiguous United States (CONUS) datasets, CorrectCast improves six base nowcasters spanning extrapolation-based, deterministic, and generative approaches, with absolute gains in mean CSI averaging 0.0233 and 0.0290 across the six models, respectively. 

---
# The Only Edge Was Memorized in Training: LLM Trading Agents Around Earnings and Bad News 

**Authors**: Jackson Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22177)  

**Abstract**: Large language model (LLM) trading agents are tested on news, but news reaches stock prices within minutes, and a benchmark that replays headlines at their publication time finds no alpha in trading on them. Money can then remain in two places, before news whose date is known, such as earnings announcements, and after bad news, if the market's fall went too far. However, studies of LLMs and earnings read documents published with or after the announcement, and studies of reversal sort monthly portfolios, so neither tells whether an agent can trade these moments after costs. In this paper, we test LLM agents at both moments on the same look-ahead-free benchmark, before 598 earnings announcements and after 409 bad-news drops of large US stocks. Every input is cut at the decision close, staged agents are compared with single calls, simple rules and a logistic regression, and every decision is made with and without the stock's name. Across five models from four vendors, no agent beat the simple bets. Before earnings, the agents mostly bought and earned what buying before every announcement earned, and after bad news, a Gemini-based agent called almost every fall an overreaction while no agent picked better dips than buying them all. The only gains came from what the models had memorized in training. Given only the ticker and the dates, a GPT-based model called the direction of 59.2% of the announcements before its training cutoff and 53.3% after it. Agent builders should therefore compare an event agent with the simple bet and test it blind after the training cutoff. 

---
# Reconfiguring Procedural Knowledge for Compositional Robot Skill Adaptation 

**Authors**: Daehee Lee, Dongsu Lee, Sanghyun Ahn, Minjong Yoo, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2611.22174)  

**Abstract**: General-purpose behavior models increasingly internalize broad procedural knowledge during training, making their adaptation to new goals and skill interfaces less a problem of learning entirely new behaviors than of reconfiguring existing competence. We study this capability as reconfigurability, the ability to realize new behaviors by cost-efficiently restructuring existing components and relations under sparse governing information, where task specifications provide high-level goals and procedure specifications expose a limited set of reusable factors, leaving the concrete bindings and compositions required for execution underdetermined. We propose RecPo, a framework that represents skills task-agnostically through multiple procedure views and materializes executable skill sequences by composing stored transitions to realize task-specific changes. RecPo maintains a procedure knowledge base of skill-induced transitions, with each view exposing a reusable factor and preserving local skill connectivity. At deployment, it interprets task specifications through the same views and resolves the resulting compositional queries over the stored relations to bind skills for execution. On long-horizon tasks in an extended Franka Kitchen benchmark, RecPo achieves 91.9% ordered completion with 1.0 ms of task inference per episode across varying procedure support and task-record sparsity. Real-robot studies further demonstrate adaptation through local memory edits without task-specific retraining. 

---
# Readie: Low-Latency Code Sandboxes for AI Agents 

**Authors**: Arjun Sivaraman, Zhaoheng Li, Alan Shi, Sanjay Malisetty, Eugene Wu, Yongjoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.22171)  

**Abstract**: From quantitative research to personal assistance, AI agents act by iteratively generating and executing code. To decouple the agent harness from code execution, many agents offload execution to dedicated systems called code sandboxes. However, existing code sandboxes suffer from slow startup. Whether they rely on custom snapshots or a single fat image, the root cause is the same: it is hard to build a compact, pre-warmed snapshot that can be shared across diverse agent-generated code while covering nearly all the artifacts (e.g., packages, datasets) it requires.
In this work, we propose learned startup snapshots for low-latency code sandboxes. Our insight is that while it is hard for a single snapshot to be both compact and cover most workloads, a set of snapshots can achieve both. Each snapshot is small, initializes quickly, and covers a fraction of agent-generated code; together, the set covers most of it. We learn this set from a large training workload by formulating snapshot construction as an optimization problem and solving it with an approximation algorithm. Planned on 15K public and LLM-synthesized scripts, our system, Readie, reduces startup latency by 3-22$\times$ and end-to-end runtime by 1.3-3.3$\times$ compared to existing approaches at the same storage budget. 

---
# Not All Evidence Is Admissible: Evidence Governance for Tool-Augmented Medical Agents 

**Authors**: Runxun Zhang, Onofrio Antonio Catalano, Jinqiao Wang, Jingwei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.22169)  

**Abstract**: Tool-augmented medical agents treat whatever their tools return as evidence. We show that this default is costly: medical tools fail silently, and a frozen reasoner cannot ignore what enters its context, so many cases it would have answered correctly are flipped to wrong once tool outputs are added. We call this evidence hijacking. $\mathcal{V}$-information explains the phenomenon: the reasoner has lost the option to ignore, so a tool output can carry negative usable information, and moving the decision of what to keep outside the reasoner restores that option. The theory further shows that part of any filtering gain comes merely from letting the reasoner see less, and only selection by content lets it see better. Building on this, we recast tool-augmented reasoning as evidence governance, which separates deciding what counts as evidence from the reasoner that answers, and propose context-independent evidence governance (CIEG), the first medical agent framework that rules, after acquisition, on whether each individual tool output may enter as evidence. Every tool output is written as a card that can be checked against the raw input; an independent checker that sees only the input and the cards decides which cards become evidence; and the reasoner answers from the admitted cards alone, with no training anywhere. On medical question answering, fundus glaucoma screening, and chest X-ray pneumonia classification, it achieves the highest accuracy among models of comparable scale, and significantly outperforms state-of-the-art tool-using agents and multi-agent systems built on the same backbone at lower cost. 

---
# Long-Context Generation Is a Sampling Problem (When You Turn Up the Heat) 

**Authors**: Allen G. Roush, Minh Nhat Nguyen, Ravid Shwartz-Ziv, Judah Goldfeder, Sanjay Basu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22148)  

**Abstract**: Open-weight language models are widely observed to get worse the longer they generate. Often, attempts to fix this involve more training. At high temperature, The sampler, i.e. the rule that picks the next token out of the model's probabilities, decides whether long and agentic generation survives and/or thrives. We test 29 open-weight models from 1B to 122B parameters on ten established benchmarks plus needle retrieval out to one million tokens, covering SWE-bench Verified coding agents, long chain-of-thought reasoning, long-form writing, long structured output and long-context reading. Samplers that keep their pool of candidate tokens small and steady as temperature rises don't lose accuracy. Samplers that let in low-probability tokens fall apart. At temperature 2, coding agents running top-p, top-k or Top-H resolve 0 of 100 GitHub issues, while the same agents running top-nsigma resolve 46 to 62 and win every issue the two samplers disagree on, on three Qwen3.5 models. Over all benchmarks top-nsigma keeps a median 100% of its temperature-1 score where top-p keeps 26%. Counting how many tokens each sampler lets through explains why. In one Llama-3.1-8B story, going from temperature 1 to 2 grows top-p's pool from 4 tokens to 51,845 while top-nsigma stays at 3. Across 133 combinations of sampler setting and temperature, the typical pool size predicts judged quality (Spearman correlation of -0.77), and when we pick the pool at temperature 1 and only apply temperature afterwards, Top-H, min-p and p-less stop collapsing. At the default temperature of 1 most samplers perform about the same, We release all code and data to the public. 

---
# Big Brains, Tiny Board: Pupavision, an Agentic Physical AI Instrument for Monitoring Drosophila Pupariation and Lifecycle Transitions 

**Authors**: Joel Sadler, Daniel Graham, Melanie Worley  

**Link**: [PDF](https://arxiv.org/pdf/2611.22136)  

**Abstract**: Many biology experiments are limited by the availability and ease of use of instruments that monitor and record them. For example, the timing of pupariation in the fruit fly Drosophila melanogaster, a widely used model organism, can provide biologically relevant information about growth regulation and is often recorded manually. Here, we present Pupavision, a small physical AI instrument built from a single-board Linux computer and a day/night camera for roughly 45 to 75 US dollars in parts. The key idea is that a frontier coding agent runs its agentic harness directly on the tiny board. The model thinks in the cloud, but the agent works on the device. The agent assists image capture, writes code to repair errors, and checks the experiment every day to keep raw data capture reliable. If recording stops, the researcher gets a ping on their mobile phone. A simple deterministic capture service records one image every ten minutes and never depends on the agent. We report a 26-day pilot in a fly incubator (about 3,580 frames), including eleven days in which the agent was logged out and capture continued, and a 51-hour outage whose cause led to the layered design. A researcher-in-the-loop workflow then produces per-vial pupariation curves by tracing selected pupae backward in time from a late reference frame. Future work will validate the assay for lifecycle tracking accuracy against expert-labeled data and expand real-time computer vision analysis. Pupavision offers a blueprint for do-it-yourself agentic instruments that put frontier intelligence on a tiny board, with deterministic code underneath. We present this as an early case study from a pilot lab deployment and contribute lessons learned toward a future where scientific domain experts are empowered by DIY agentic hardware. 

---
# LoCaR: Localized Canvas Retrieval With Frozen Vision-Language Model 

**Authors**: Rahim Hossain, Dipanwita Saha, Ashirbad Mishra, Xinxin Shu, Svetlana Strunjas, Binbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.22131)  

**Abstract**: Whole image embeddings can under represent small objects and local attributes that distinguish relevant images for localized queries. We present LoCaR, a training free framework for localized text to image retrieval with frozen vision language models. LoCaR constructs and utilizes independent proposals (SAM2) focusing on the object so as to get a more localized representation that independently encodes each region using the frozen visual encoder. We further study paired-support adaptation as an optional residual that uses matched text region exemplars without updating the underlying encoders. We also introduce LOCFG, a Visual Genome derived benchmark designed to evaluate localized retrieval for smaller and more challenging referents. Across several vision language backbones, our results show that improving the region representation itself is the main source of gain. 

---
# TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks 

**Authors**: Xingchen AGI Lab, Baotian Hu, Bingkai Yang, Boyang Wang, Changzai Pan, Chao Lyu, Chao Wang, Chen Li, Chenshuo Pan, Chenyu Lin, Chenyuan Zhang, Chongming Liu, Dawei Fan, Di Wu, Dong Guo, Dong Li, Dongfang Li, Enlin Liu, Fanye Xiao, Fengkai Li, Fubei Yao, Gang Lin, Han Gao, Hanming Wu, Haowei He, Huapeng Liu, Hua Wang, Huinan Xu, Jiahong Xiong, Jiaxin Peng, Jianxiang Ma, Jianxing Zhuge, Jie Zhang, Ji Xiong, Jinglin Chen, Jingqi Li, Jingxuan Li, Jiwang Ye, Junchen Li, Junbiao Li, Junyang Wu, Kaidong Yu, Kaipeng Jia, Kehai Chen, Kuai Li, Laobin Liu, Lei Li, Libo Qin, Lingling Shi, Luwen Pu, Menghao Xu, Mengjiao Bao, Mengqi Chen, Min Zhang, Mingrui Wang, Naying Wang, Peicong Han, Peiming Guo, Pengfei Li, Qiang Feng, Qi Wang, Qingjun Zhu, Qingli Li, Qitai Ding, Qiyi Xie, Quanchao Zhao, Rongbao Han, Rui Xue, Ruiyu Fang, Ruiting Song, Shijie Li, Shiquan Wang, Shuangyong Song, Sishi Xiong, Teng Su, Wankang Bao, Weijie Zhang, Wei Mao, Weiwei Fu, Weiyang Huang, Wenmin Deng, Wenqing Hu, Wujie Zhong, Xi Yu, Xiaohui Hu, Xiaomin Fang, Xiaomeng Huang, Xiaoqiong Shi, Xiaotong Shang, Xiaoyu Li, Xinglei Xu, Xin Jiang, Xin Wang, Xin Zhang, Xinzhang Liu, Xue Liu, Xuefeng Bai, Xuewei He, Yahan Li, Yanhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22115)  

**Abstract**: We present Xing 4, a series of large language models that extends the TeleChat lineage through coordinated advances in model architecture and training infrastructure on Ascend. This report describes the design principles and systems engineering behind the series, spanning architecture selection, large-scale pretraining, and reasoning and agentic post-training. We study how sparse activation, attention design, model depth and width, and speculative decoding balance model capacity with computation, memory access, and execution overhead. Our pretraining infrastructure builds on MindSpore HyperParallel, combining topology-aware hybrid sharding, efficient distributed optimizer execution, and coordinated management of long-sequence workloads and activation memory. CPU affinity and core isolation further reduce host scheduling interference. Together, these mechanisms support the pretraining of a 285B-parameter model on 8,192 Ascend 910C cards, providing a concrete setting for the architectural and systems studies. For post-training, we introduce TeleGym, a unified agent execution platform that connects execution, evaluation, and trajectory construction to policy optimization. Disaggregated asynchronous execution, bounded oversampling, and partial rollout accommodate heterogeneous task durations while preserving trajectory continuity across policy updates. Together, these advances establish an Ascend-oriented training foundation for the continued development of Xing 4 across model scales and increasingly demanding reasoning and agent workloads. 

---
# RegimeCouncil: Paired-Evidence Routing For Memory Conflict Governance In LLM Agents 

**Authors**: Qingcan Kang, Mingyang Liu, Shixiong Kai, Tao Zhong, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.22113)  

**Abstract**: Language agents rely on long-term memory to personalize responses across extended interactions, yet stored facts quietly go stale as the world changes, and an agent often learns that using memory hurt only through delayed feedback after acting on an outdated record. Before the memory store is updated, how far should that negative feedback propagate: should it suppress memory use only on future queries for that same fact (a local policy), or also across other stored facts (a global policy)? Existing memory systems implicitly fix one of these two sharing scopes at design time. We formulate this online scope choice under hidden validity and delayed feedback as memory conflict governance, and prove that neither fixed policy is sufficient on its own. We then propose RegimeCouncil, a lightweight paired-evidence router that retains both scopes as advisors and arbitrates between them online without retraining the LLM. Using revealed comparisons between answering with memory and answering without memory (the no-memory anchor), a Local Advisor tracks the latest outcome on the queried fact alone, while a Global Advisor pools recent outcomes across all stored facts. A Meta Governor follows the Local Advisor by default to avoid cross-fact interference and switches to the Global Advisor only when global pooling accumulates a strictly positive recent utility margin, guaranteeing finite-time switching under delayed feedback. Across LongMemEval update streams and real-world historical drift on TempLAMA and TAQA with Qwen3-30B-A3B and GPT-4o, RegimeCouncil outperforms the better fixed-scope policy by 5.2-7.3 points on primary disagreement episodes and across all 30 held-out settings, while ablations show that removing any of its three modules reduces accuracy by 2-7 points. 

---
# From Sorry to QED: Refining LLM Math Proofs via Angelic Execution 

**Authors**: Yanju Chen, Hanzhi Liu, Ruizhe Qian, Jiaming Shan, Xu Yang, Yufei Ding, Osbert Bastani, Yu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2611.22103)  

**Abstract**: Large language models (LLMs) can now generate Lean proof scripts that capture the high-level structure of many mathematical arguments. However, these scripts frequently fail on complex theorems: a single missing side condition or misapplied lemma can cause the prover to abort early and discard an otherwise viable proof outline. This paper presents MathEye, a framework that converts imperfect LLM-generated proof scripts into fully verified Lean proofs by combining symbolic reasoning with minimal human interaction. At the core of MathEye is angelic proof checking, an optimistic execution model that replaces failing proof fragments with placeholders while preserving the overall structure of the argument. This optimistic check produces an angelic proof that isolates the exact proof obligations that the LLM fails to discharge. Because these unresolved obligations are often logically interdependent, asking a mathematician to address every placeholder directly is inefficient. To minimize human effort, MathEye performs residual analysis over a proof dependence graph that captures symbol-level data flow in the angelic proof. By formulating obligation selection as a MaxSAT optimization problem, MathEye selects a minimal-cost set of essential obligations whose resolution suffices to complete the entire proof. MathEye presents these essential obligations to the mathematician in ordinary mathematical language and uses the mathematician's answers to automatically reconstruct the complete Lean proof. Experiments on 30 real-world benchmarks across analysis and algebra show that MathEye solves 93.3% of the problems while reducing the number of required user interactions by 55%. 

---
# EvoDisco: Strategy Discovery for Evolving Agent Memory 

**Authors**: Mingyang Liu, Qingcan Kang, Shixiong Kai, Kaichao Liang, Xiaojin Fu, Tao Zhong, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.22102)  

**Abstract**: Large language model agents accumulate experience in memory, and how that memory evolves during interaction is as important as what it stores. Recent methods update memory online, yet both the operators that transform memory and the rules that select them remain fixed by design. We propose EvoDisco, which formulates agent memory evolution as evolutionary strategy discovery: the agent must discover which operator to apply, when to apply it, and how to instantiate it, rather than rewriting memory under a static operator schedule. A learnable policy conditions on the current memory state and recent task history, then selects and instantiates one of four semantic operators---Abstract, Refine, Bridge, and Reframe---which respectively compress related memories, decompose overloaded entries, connect weakly linked experiences, and reorganize retrieved reasoning chains. The policy is updated with a bi-level fitness. An inner signal scores each transformation using a retrieval-quality improvement regularized by memory complexity. An outer signal scores strategic efficacy over a window of past decisions, combining historical operator-utility regret, utility-prediction error, and selection diversity. Under a one-pass online plus frozen-transfer protocol on ALFWorld, WebShop, and LifelongAgentBench (OS and DB), EvoDisco improves over strong memory baselines including MemP and MemRL. The largest gains are on ALFWorld ($+8.8$ online / $+12.9$ frozen-transfer points over MemP; $+16.4$ / $+17.9$ over MemRL). Ablations attribute these gains to both fitness levels and the operator library. 

---
# ATLAS-Finance: Benchmarking AI Agents in Realistic Banking Environments 

**Authors**: Elaine Lau, Antoine Pangas, Alexandre Finot, Collin Schweiker, Amir Bouzidi, Hui Wen Goh, Andrew Peter Martin, Jonas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2611.22085)  

**Abstract**: Professional work involves under-specified assignments, where relevant information is spread across workplace applications and may conflict or change over time. To evaluate how well frontier agents can perform economically valuable work in such settings, we introduce ATLAS-Finance, a benchmark of 100 end-to-end investment-banking tasks across 13 simulated firms. As in real banking workflows, agents must search email, chat, calendars, notes, and files to identify requirements, gather information, reconcile conflicts/updates, and then build complex financial models. Tasks were developed by experienced bankers based on their real deal work and take them 15-30 hours to complete. Agent deliverables are evaluated with expert-authored rubrics (median: 124.5 criteria) covering financial rationale and results, spreadsheet construction, completeness, and formatting. Despite completing substantial portions of the requested work, current agents rarely produce correct deliverables. Across today's frontier models, the best ones (Claude Opus and Fable) pass fewer than 11% of tasks. 

---
# When Lower Harm Misleads: Auditing Concept Corrections 

**Authors**: Tianyou Lai, Wentao Yue, Hongji Li, Chaoyuan Hao, Qilei Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.22084)  

**Abstract**: An expert who corrects a concept prediction, such as a BI-RADS descriptor in breast ultrasound, expects the diagnosis to improve. Objectives that reduce the harm of such corrections are judged by harm and accuracy, yet harm also falls when a model responds less to every correction or starts from a worse uncorrected prediction, the reference for harm. We propose a correction audit, a reproducible evaluation protocol that separates these explanations: it splits each correction into harm and useful gain on identical events, separates the starting prediction from the correction response, and compares objectives with amplitude baselines matched in source harm. On three breast-ultrasound sources, worst-case correction training, evaluated from the mean-case model's start, lowers harm but gives up more useful gain in all 21 source comparisons pooled over six seeds, raising the corrected negative log-likelihood (NLL); from the same start, selecting between the objectives by harm picks the less helpful model on all three sources. On CUB, a monotonic harm penalty at its reported weight lowers path-level harm in all 80 concept sets while raising the NLL in all 80, almost entirely before any correction. At a fixed start, rescaling both responses to the same source harm reproduces the penalty's mean harm, useful gain and NLL on single-concept corrections within a prespecified margin of 0.0005 nat, although the two responses point in different directions. Lower harm therefore signals better corrections only when it is read together with useful gain, absolute risk and the start it is measured from. 

---
# Can You Undo It? Reversibility Classes for AI Agent Actions 

**Authors**: Deepak Waghmare, Shaad Sufi  

**Link**: [PDF](https://arxiv.org/pdf/2611.22075)  

**Abstract**: Tool-using AI agents change files, databases, access rights and public information, yet permission to call a tool does not establish whether a mistake can be undone. We present Reversibility Classes, a declared five-level vocabulary with argument-sensitive rules, conservative defaults and an explicit prior-state capture contract. The pinned implementation and evidence-linked registry cover 536 tools across 20 MCP servers. All 738 rule-case checks conform to declared expectations, while native annotation mapping differs from draft registry labels for 118 of 481 comparable tools. One AI-assisted reviewer approves 148 ratings, 75 matching the registry; this is not independent agreement. On 2,158 recorded runs, the implemented session guard gates the associated action in all 29 attributable attack-positive runs, compared with 18 for a read/write proxy, while adding nine approval points across 1,209 benign runs. Controlled file, Git, SQLite and three-browser experiments show that recovery depends on captured state, later changes, editor storage and recovery route, and expose a Git merge-state exception. The evidence supports making recovery assumptions explicit and testable. It does not establish registry-wide accuracy, reduced production harm or general recovery reliability. 

---
# MedGuardBench: A Patient-Oriented Benchmark for Personalized Guardrail Development 

**Authors**: Sara Khosravi, Xiaofei Wen, Majid Sarrafzadeh, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.22069)  

**Abstract**: Safety is personal: the same action can be safe for one person but unsafe for another. Yet current guardrails judge content without considering who is affected. Using medication safety, we introduce \textit{patient-conditioned medication guarding}, which predicts a safety verdict and 17 risk categories from a patient profile and prescription. We release \textbf{MedGuardBench}, a drug-disjoint benchmark of 5{,}001 scenarios across 87 medications. We find that standard guardrails rely on surface cues, such as wording and profile presence, rather than patient facts. To address this, we develop \textit{personalized reasoning guardrails} with three components. \textbf{Personalized Reasoning Distillation} links risks to relevant patient fields. \textbf{Context Unrolling (CU)} makes models check each field themselves, reducing false alarms on most backbones; controls show that providing the check does not yield the same benefit. A \textbf{rule-based reward (guided GRPO)} improves unsafe recall and complex-risk detection, including allergies and adverse reactions, on some backbones, but increases false alarms on others. Our models outperform base models and matched no-profile controls in accuracy, while reducing false alarms relative to no-profile controls. Profile removal and swapping confirm that gains come from patient information, not fine-tuning alone. 

---
# A Hybrid Exact-Heuristic Solver for the Integrated Passenger and Parcel Transportation Problem 

**Authors**: Dung Bui, Ha Thu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.22043)  

**Abstract**: We describe the solver we submitted to the second round of the SOICT 2026 contest on the Integrated Passenger and Parcel Transportation problem, a selective pickup-and-delivery routing problem with time windows, open routes and heterogeneous taxis. The solver is a single-file, standard-library Python program that combines (i) a constant-time insertion test based on latest-arrival slack, (ii) greedy insertion, local search and an Adaptive Large Neighborhood Search (ALNS) with simulated-annealing acceptance, and (iii) an exact route-selection engine that enumerates, for every taxi, the best feasible route for each request set and then solves the resulting set-packing problem by branch-and-bound strengthened with route-based large neighborhood search, Lagrangian bounds and reduced-cost fixing. The engine is chosen automatically from the instance size, and an independent verifier and a watchdog guarantee that a valid answer is always printed. On the official evaluation (38 scored instances, 5-minute limit) the solver obtained a total benefit of 487,237, finished every run within 277 s and used at most 279 MB; 28 of the 38 instances terminated well before the limit. 

---
# Thinking Harder Doesn't Lead to Better Choices: Reasoning Effort Increases the Confidence but Not the Correctness of LLM Shopping Agents 

**Authors**: Adi Krishnamoorthy, X.Y. Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.22033)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous shopping agents that process multimodal product listings (consisting of visual and text data) and recommend purchases on behalf of consumers. Naturally, one might assume that models with more thinking power would make higher-quality purchasing decisions. To test this belief, we ask two questions about such agents: does asking a model to think harder make it choose better products, and does the confidence it reports tell us the likelihood it may be wrong? We benchmark ten frontier models on three product-choice benchmarks built from WebShop, Amazon's SQID and Wayfair's WANDS platforms to see if LLMs with more thinking power actually make better-quality product purchase decisions. Across all three benchmarks, we found that increasing the thinking effort does not increase decision quality in a statistically meaningful way. The same change in thinking effort raises stated confidence at a statistically significant level, so extra thinking buys confidence rather than correctness. We also measure the calibration of these models (how well their reported confidence percentage matches their true correctness percentage) in these tasks and show that most models are not well-calibrated at every effort level. We also find that errors are strongly shared: when one model errs in a task, several others tend to err on that same task. Furthermore, on tasks where several models all err, a large number of them pick the same wrong product. 

---
# What Does an Agent Need? Bundle-Aware Agent Resource Composition 

**Authors**: Yunxiao Shi, Tingwei Chen, Dannong Xu, Ahad N. Zehmakan, Jialie Shen, Zhendong Chu, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2611.22030)  

**Abstract**: An agent resource's value depends on its backbone and companion resources and is known only after execution. Token-based generators remain tied to their training catalog, while independent resource scoring misses complementarities and reused bundles target other tasks. We formulate Task-Conditioned Agent Resource Composition (AgentComposition): jointly selecting a backbone model and a tool-skill bundle to maximize end-to-end execution utility. Our Retrieval-Grounded Bundle-Aware Agent Generation retrieves candidate resources and past configurations of similar tasks. A fine-tuned generator reads resource descriptions and recombines candidates through constrained beam search, allowing new resources without retraining. A bundle-aware critic ranks complete configurations using a large language model (LLM) teacher and historically successful configurations as execution proxies, and adapts to execution feedback when available. We evaluate reference recovery, offline execution with judged and native rewards, and online recommendation with real execution, building AgentSelect-Exec with 5,060 executed configurations. Our method recovers the complete reference configuration within ten candidates for 67.73% of AgentSelect queries, nearly twice the strongest baseline, and leads all baselines on SkillsBench. The proxy-trained critic raises the top-ranked agent's execution score from 7.58 to 8.37 out of 10; execution feedback increases it to 8.78, above the strongest baselines. Four target-domain queries suffice to adapt the generator to SkillsBench. Agents recommended online outperform the strongest baselines under both an LLM judge and native verifiers. These results support learning and transferring joint resource composition from task experience. 

---
# SkillBuffBench: Measuring What Agent Skills Contribute across Harnesses, Models, Skill Access Modes and Memory 

**Authors**: Gengyue Dong, Ye Wei, Zheng Zhu, Yidong Han, Jiale Miao, Xiaojun Jia, Guanghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.22017)  

**Abstract**: Agent Skills, file-based packages of procedural knowledge that a coding agent can discover and load on demand, are now shipped by every major agent harness, yet the evidence for how much they help comes almost entirely from settings in which the right Skill is hand-picked for the task and the agent starts from a blank slate. Neither holds in deployment, where an agent chooses from whatever library happens to be installed and frequently carries memory from earlier work. We present SkillBuffBench, a benchmark and paired evaluation protocol that treats Skill access as a three-level factor (no Skill, a curated Skill, and automatic invocation from an installed library), crosses it with memory (off and on), with four agent harnesses and with multiple models (the state-of-the-art proprietary models from Anthropic, Google and OpenAI, and the state-of-the-art open-weight models from Chinese providers), every model being run under every harness, and evaluates the result on research tasks contributed by domain experts together with the Skills they use, currently in materials science and biology, with the suite extended on a rolling basis. Every run is marked by a deterministic verifier; uncertainty is reported with a paired, task-level bootstrap; and cost, agent rounds and wall-clock time are reported alongside the score rather than folded into it. This version of the paper describes the design of the benchmark, its evaluation protocol and its current tasks and Skills. The evaluation is in progress: results will be added once the runs are complete, and the task suite, results and analysis will continue to be updated on a rolling basis. 

---
# On Formal Counterfactual Explanations 

**Authors**: Alexey Ignatiev, Peter J. Stuckey, Joao Marques-Silva  

**Link**: [PDF](https://arxiv.org/pdf/2611.22010)  

**Abstract**: Counterfactual explanations are often seen as counterfactual conditionals under closest-world semantics, requiring one to reason over all minimally distant counterexamples. In contrast, formal XAI typically relies on feature-based abductive explanations (AXps) and contrastive explanations (CXps), which admit efficient computation and a minimal hitting-set duality. This paper bridges these perspectives by (i) characterizing closest-world counterfactual explanations explicitly as a disjunctive normal form over all minimally distant counterexample instances, and (ii) introducing conjunctive counterfactual explanations (CFXps), a subclass of counterfactual antecedents restricted to conjunctions of feature literals that guarantee prediction change. The paper establishes structural results connecting CXps, AXps and CFXps: every CXp induces an AXp for its witness instance, and every CXp together with a witness AXp induces a CFXp whose support is precisely determined by that AXp. These theoretical results yield a modular approach for computing CFXps using existing AXp/CXp machinery. Experiments on a representative set of benchmarks using tree ensembles and neural networks show that CFXp extraction is practically efficient. Compared with popular point-based counterfactual explainers, CFXps generally require no more feature changes and can guarantee the counterfactual prediction beyond a single instance. 

---
# KV-LTM: Persistent Per-Session KV Caches as Long-Term Memory for Personal Assistants 

**Authors**: Ziv Kfir, Ido Kfir  

**Link**: [PDF](https://arxiv.org/pdf/2611.21998)  

**Abstract**: Personal assistants accumulate months of conversations, but today's long-term memory stacks put derivation or selection between a question and its evidence. In a system-level comparison on LongMemEval, two commercial assistants scored well below the same model reading the history offline [1], consistent with, but not isolating, content loss between write and read. We propose KV-LTM: a memory tier storing each session's raw text plus its KV cache under a fixed reader model, answering by restoring cached sessions and appending the question. KV-LTM is complementary to, not a replacement for, retrieval and indexing. Text is the memory of record; the cache is a recompute-free prefill accelerator, degrading gracefully to text. We specify the design: storage record, budget-derived retention, append-only read path, and bounded aggregation. We size it analytically: across pinned BF16 full-attention (GQA) and latent-cached MLA configurations, per-token state spans approximately 31K-328K bytes (the pinned KDA+MLA hybrid differs: approximately 8.1K bytes/token plus a fixed per-session recurrent term), giving approximately 0.34-3.6 TB per user-year at a declared 30K-token/day scenario, against approximately 49 MB of raw text; per-query cost is modelled as regime-conditional transfer, attention, decode, aggregation, and compute. We define an evaluation protocol needing no persistent-cache implementation for logical QA evaluation: per-session prefill emulates valid cache reuse under the same model/input-state contract and ideal arithmetic, with named arms, controlled baselines, and a preregistered reader-model gate; a separate minimal real-cache replay tests numerical equivalence and measures the cost axis, with outcomes not assumed. KV-LTM trades storage for avoided re-prefill compute and write-time fidelity, within a declared retention budget. 

---
# WebAgents: Unbundling Agent Creation, Hosting, and Execution for an Agent-Native Web 

**Authors**: Dario Fenoglio, Martin Gjoreski, Marc Langheinrich, Pradyumna Chari, Ramesh Raskar  

**Link**: [PDF](https://arxiv.org/pdf/2611.21978)  

**Abstract**: Large language models are moving from isolated assistants toward autonomous agents that can plan, use tools, and coordinate with others, motivating an emerging Agentic Web of independently developed agents and services. Realizing this vision requires agent behavior to move across creators, hosts, and runtimes without coupling publication to a particular execution environment or transferring authority over user resources. We introduce WebAgents, which unbundles these roles through a signed this http URL that carries portable workflows, skills, schemas, and resource requirements. An OpenClaw-based agent-native browser/runtime verifies the package, obtains authorization, and binds it at run time to user-controlled models, tools, permissions, and budgets, while hosts distribute the public program and may retain private services server-side. We demonstrate this contract with three unrelated workflows executed unchanged under local and API-backed models, including literature review with live services and a website-published travel agent using semantic tools and user-selected personal resources. In a controlled suite, WebAgents explicitly blocks all 716 injected contract violations while accepting all 295 matched benign controls, with median automated pre-execution processing below 0.5 s for all three packages. We conclude by outlining a path toward NANDA-based discovery and interoperable package ecosystems while retaining execution authority in the user's runtime. 

---
# DiscoveryAgent: A Framework for Scientist-centric Agentic AI 

**Authors**: Owen Queen, Ioana Ciuca, Emma Dann, Jean Somalwar, Carine You, Steven Dillmann, Viraj Manwadkar, Evan Gorrell, Harrison G Zhang, Jonathan K Pritchard, Risa H Wechsler, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2611.21950)  

**Abstract**: AI agents can assist scientific research, but balancing autonomous investigation with scientists' ability to inspect and redirect ongoing work remains a central design challenge. We present DiscoveryAgent, a domain-agnostic, scientist-centric system that combines autonomous investigation with scientist-directed hypothesis steering. Scientists select, refine, or replace hypotheses between research rounds, while audit agents interrogate checkpointed records to trace claims, reproduce analyses, and inform subsequent decisions. Across two predictive modeling tasks, DiscoveryAgent produced the best variable-star classifier among evaluated baselines and, with scientist steering, a protein variant-effect predictor surpassed by only one of 106 ProteinGym baselines. We develop a rigorous checkpointing system that enables highly capable audit agents, a scalable form of oversight for agentic research. Audit agents autonomously identify errors at higher rates than previous AI scientist agents, and scientists collaborating with audit agents identify 14x more valid errors versus scientists performing manual review. In three case studies led by domain experts across physical science, social science, and life science, we developed a multimodal photometric-redshift estimator that resolves complementary failures of images and photometry, achieving 0.971 AUROC for low-redshift classification; documented an 83% rise in Florida homeowners-insurance premiums from 2013 to 2023, with a growing, regressive burden; and identified CD4+ T-cell relay genes, which both receive and exert broad transcriptional regulation, enriched for autoimmune disease associations. Together, these results show how autonomous research can expand scientists' empirical reach while centering their authority over research direction and their ability to understand and challenge the evidence. 

---
# Honest Brokers in AI Regulation: How Public Comments Introduce Evidence to Regulators 

**Authors**: Sarah Tran, Anna-Maria Gueorguieva, Eva Maxfield Brown, Nicholas Weber  

**Link**: [PDF](https://arxiv.org/pdf/2611.21948)  

**Abstract**: The regulation of Artificial Intelligence is susceptible to various forms of capture by industry actors that hold undue influence over the policymaking process, due in part to their information advantage in developing these technologies. In the following paper, we investigate the roles of public commenters in providing evidence to potential regulation in the USA. We build a corpus of 40 potential regulations, and 2,664 public comments submitted by a variety of stakeholders. We computationally process these texts to extract evidence meant to inform regulators, in the form of formal citations and informal references. We find significant differences in the type and amount of evidence supplied: Holding comment length constant, business commenters introduce altogether less evidence, and government, university, and civil society commenters provide significantly more research and legal sources. The publication status of cited research is nearly evenly split, with preprints and peer-reviewed articles being introduced by commenters of all types. We further explore patterns of self-citation and find that while it is prevalent amongst all groups, a small cluster of organizations account for over half of all self-citations in our corpus. Our findings complicate a narrative that public comments alone are means to information capture. We interpret these results in terms of the role that commenters attempt to play in the policymaking process-being either an Issue Advocate or an Honest Broker. These findings on the supply of evidence in public comments shed light on an often overlooked form of policymaking that has implications for how and through what sources information is made accessible to regulators of AI. 

---
# Conserved Verdicts, Drifting Commitments: Identifying Harmful Overthinking 

**Authors**: Jianghong Zhou, Hao Rong, Chen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.21946)  

**Abstract**: A reasoning model that has reached a correct answer can lose it by continuing to think. Deciding whether to stop means comparing two probabilities at the same point of a chain: that the answer the model would give if stopped now is correct (its commitment), and that the answer it will finish with is correct (its verdict). We show that the verdict alone, which is what final-correctness probes estimate, cannot make this comparison. As a forecast of a fixed outcome it has no predictable drift, whether continuing helps or hurts. The commitment-verdict gap equals the expected remaining change in answer quality and the accuracy gained by stopping. A second identity splits any accuracy-versus-budget curve into answer change within chains and change in which chains are still running. We measure both on MATH-500. At all 379 points where DeepSeek-R1-Distill-Qwen-1.5B holds a correct answer, selected without regard to how its chain ends, continuing loses 3.6 points on average (95% CI [2.2, 5.2]), and the loss concentrates in a few problems. In six reasoning models from three families, on MATH-500 and OlympiadBench and with budgets up to 16,384 tokens, declining accuracy curves come from which chains remain, while the chains still reasoning keep or improve their answers. An oracle stopping rule gains at most +1.8 and +2.5 points at 1.5B and 7B. The framework says what to measure before building a stopping rule. 

---
# SLED: Sensor SLicing from Edge to Datacenter 

**Authors**: Alan Subedi, Avhishek Biswas, Apala Pramanik, Amlan Balabantaray, Mikhail Titov, Santosh Pitla, Ozgur Kilic, Mehmet C. Vuran  

**Link**: [PDF](https://arxiv.org/pdf/2611.21937)  

**Abstract**: Modern sensing systems generate large volumes of heterogeneous data streams from multiple sensor modalities, increasing reliance on datacenters for computation, artificial intelligence/superintelligence (AI/SI) inference, and AI/SI training. Each sensor stream may have different network and compute requirements depending on its type and data volume. However, current wireless systems typically allocate resources at the device level, leading to contention among co-located sensor streams, where bulk data transfers may starve latency-critical telemetry. To this end, this paper introduces Sensor Slicing (SL). SL provides deterministic, medium access control (MAC)-level isolation for individual sensors within a single user equipment (UE) in open radio access networks (O-RAN). Using agentic coding techniques, SL is implemented in two O-RAN-compliant packages, reducing development time by 400 times (2 h 40 min [OCUDU] vs. 6.5 weeks [OAI]), 228-fold fewer human prompts (4 vs. 913), and less than half the tokens (0.64B vs. 1.43B). An associated end-to-end framework, SLED, is implemented to establish per-sensor radio and network isolation across edge, campus, and remote high-performance computing (HPC) tiers. Over-the-air wireless experiments demonstrate that while conventional network slicing causes critical delays to increase more than 2,300-fold under high traffic load, SLED maintains stable latency. Scaling evaluations on the two RAN stacks show that throughput remains within 0.9% and 1.4% of expected allocations even as the number of sensor slices increases. The results demonstrate that SLED preserves latency-critical telemetry while effectively utilizing resources across the sensor-to-HPC pipeline. The SLED software and evaluation datasets are publicly available at this https URL. 

---
# DNeo: Your Parallel Draft Model Needs Better Target Feature Encoding 

**Authors**: Jameson Sandler, Shane Bergsma, Mostafa Elhoushi, Bin Claire Zhang, Gavia Gray, Joel Hestness, Nolan Dey, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2611.21925)  

**Abstract**: Speculative-diffusion decoding uses a small parallel drafter model to generate b draft tokens in one forward pass, which then undergo parallel verification via a larger target model. Faster inference can be achieved when multiple draft tokens pass verification simultaneously. Recent parallel draft architectures (DFlash, DSpark) improve draft quality by conditioning on rich features computed by the target, through a mechanism known as feature encoding. However, standard feature encoding mechanisms have three limitations: (1) they are lossy, applying encoding projections that are destructive to the underlying target feature-space via a discovered effect known as residual mixing, (2) they are memory-less, with no record of previously processed proposals, and (3) they utilize features sampled uniformly from across the target layers, omitting important hidden states. To address these issues, we propose Mixture-of-Features Attention (MoFA), a lightweight feature-encoding approach equipped with a working memory that encodes previously rejected proposals. Leveraging MoFA, rejection memory, and a set of further enhancements, including 2-tap causal convolutions, prefix-only cross-attention, and non-uniform target layer selection, the final architecture, DNeo, achieves 1.3x faster generation speed than SoTA (DFlash-2) without impacting drafter footprint or generation quality. DNeo marks a new state of the art in high quality, fast LLM inference. 

---
# Language Models Can Rewrite Their Own Weights 

**Authors**: Jiajun Fan, Neil He, Changyi Yang, Andy Mao, Xi Zhu, Chaoran Cheng, Kaiqu Liang, Shuaike Shen, Zetian Li, Ruihan Guo, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21885)  

**Abstract**: Language models are trained once, frozen and deployed: improving one again takes another training run, and a prompt, a scaffold or extra computation at inference is gone once the task ends. A frontier model has been shown to edit the weights of a much smaller one. We ask whether a model can rewrite its own weights to make itself better, and find that it can, by its own reasoning, with no gradients and no training; we call this self-rewriting. Given its weights, graded answers and the ability to run itself, a 27B model diagnoses its failure and edits the weights behind it. It surpasses itself: its own edit makes it 6.6 points more accurate on 490 problems it never saw, in a test fixed in advance, and another session's edit also makes it significantly better. It keeps getting better: it submits an edit only when its own tests support it, each edit replacing the last, and over its first eight such sessions its accuracy only rises. Unlike test-time scaling, the gain persists in its weights and adds no cost per answer; unlike training, it needs no pre- or post-training infrastructure, only an open model and hours on one GPU. It repairs a damaged copy of itself: it locates the corrupted matrix and rewrites it with values of its own, never reading the undamaged weights; found on text math problems alone, the repair lifts the copy on six image benchmarks it never saw from 7% to 70% accuracy, 83% of the undamaged model's. And the ability emerges with scale: across four sizes of one family, the 2B model makes no edit and the 4B and 9B models break themselves; only the 27B model works on itself as a human would, comparing its hidden states on answers that finish and on answers that run out of tokens, weakening the direction that separates them. Like a human, a model need not separate training from inference; improving a language model may come to take a conversation rather than a training run. 

---
# Auditors Struggle to Catch Sabotage In End-to-End Transcripts of AI Safety Research 

**Authors**: Keshav Shenoy, Anthony Zang, Eric Gan, Jacqueline Tay, Amy Ngo, Sebastian Prasanna, Julian Stastny, Vivek Hebbar  

**Link**: [PDF](https://arxiv.org/pdf/2611.21832)  

**Abstract**: As AI systems are increasingly used to automate AI R&D, misaligned systems could sabotage research intended to improve model safety and alignment. To stress-test defenses against such sabotage, we use a human expert as a proxy saboteur on AI safety research. We conduct a red-team/blue-team game where a human researcher runs AI safety experiments from scratch and auditors try to tell which were sabotaged and how. Given access to the full research transcript and the ability to run experiments, human auditors mostly fail to distinguish sabotaged from benign research: they achieve an AUROC of 0.59, not significantly better than chance, and fully identify the specific sabotage in only 17% of cases. Despite not being allowed to run experiments, the best frontier models outperform human auditors (Fable 5.1: AUROC 0.73, 32% full identification) within noise and weaker models perform comparably. When the saboteur chooses which experiments are attacked, rather than being required to sabotage every question given, sabotage identification drops further. Our results suggest that research sabotage by expert-level agents is difficult to detect with current auditing methods and provide a new setting for studying research sabotage. 

---
# How Much of SAR Vehicle Recognition Is the Vehicle? Background Shortcuts, Angle Gaps and Partial Libraries on MSTAR and ATRNet-STAR 

**Authors**: Mousa Shaya  

**Link**: [PDF](https://arxiv.org/pdf/2611.21827)  

**Abstract**: Deep networks score near 99% on the MSTAR vehicle benchmark, and the newer ATRNet-STAR benchmark was built to be harder. We ask how much of these scores comes from the vehicle itself. With a modern network (ConvNeXt-Small, 3 seeds) we first reproduce the known MSTAR result: with the central 96x96 pixels zeroed out, accuracy is still 73.3% (chance 10%). The same shortcut is present in ATRNet-STAR, which was collected across several scenes: with the vehicle removed, accuracy is 63.3% on the 40-class SOC split (chance 2.5%). Under the official scene-change split, the background stops helping and starts hurting. Accuracy is 22.8% on full chips and 30.8% when only the vehicle is shown. A SAR foundation model, SARATR-X, leans on the background more than ConvNeXt does (+4 to +6 points with the vehicle removed, in every seed) and does not solve the scene change. We then hold the amount of training data fixed and vary the depression angles used for training. Accuracy is set mainly by the gap to the nearest training angle: 57% at 15 degrees, 39% at 30 degrees and 26% at 45 degrees. Adding more training angles does not help. Finally, when most classes have training images at the test angle and a few do not, the missing classes do worse than when no class has them: 9.0% instead of 22.4% at a 45-degree gap. Most of their images are assigned to known classes. We recommend reporting background-only and target-only controls, and the angle gap, next to any accuracy figure. Code is public. 

---
# Chain-of-Thought Controllability Can Reduce Chain-of-Thought Monitorability 

**Authors**: Zheng-Xin Yong, Benjamin Arnav, Miles Wang, Tomek Korbak, Olivia Watkins  

**Link**: [PDF](https://arxiv.org/pdf/2611.21799)  

**Abstract**: Chain-of-thought (CoT) monitoring is a promising tool for detecting hidden goals, off-task behavior, and other failures in AI agents. One risk is that models could deliberately shape their reasoning to make it opaque to monitors. We stress-test this risk through persona-control training: supervised fine-tuning on mathematical reasoning traces rewritten in the voices of diverse personas. This induces a broader ability to control CoT, even though training contains no monitor-evasion examples. In covert side-task evaluations, traces in which a gpt-oss-20b model organism completes the hidden side task contain less monitorable evidence than the base model's. Averaged across evaluations and CoT lengths, weighted by side-task-success counts, the CoT monitorability metric g-mean$^2$ falls from 0.94 for the base model to 0.68 for the model organism without a hiding instruction, and to 0.22 when explicitly instructed to hide. In intervention evaluations, explicit hiding often drives monitorability near zero. The effect extends to a Kimi-K2.6 model organism in agentic environments, though hidden objectives requiring extended reasoning or longer trajectories remain more monitorable than simple single-step side tasks. We further discover that targeted transparency prompts can substantially recover monitorability, sometimes surpassing the base model. These model organisms provide a testbed for studying monitorability failures and potential mitigations. Our code can be found on this https URL. 

---
# NEBench R1: a Reasoning Benchmark for Evaluating Nuclear Domain Competence in Large Language Models 

**Authors**: Liyan Wang, Longze Li, Mohamed Hassan, Isshu Lee, Dajie Sun, Ruimin Feng, Aleksandar Vakanski  

**Link**: [PDF](https://arxiv.org/pdf/2611.21682)  

**Abstract**: Large language models (LLMs) are increasingly being explored for knowledge-intensive and safety-critical applications in nuclear engineering. However, existing nuclear-domain benchmarks provide limited content coverage or focus primarily on answer accuracy without systematically evaluating the quality of the underlying reasoning. This study introduces NEBench R1, a reasoning-oriented benchmark comprising 7,546 unique multiple-choice questions extracted from U.S. Nuclear Regulatory Commission (NRC) reactor operator licensing examinations covering 18 nuclear facilities. Each question is classified according to the NRC Knowledge and Abilities catalogs and accompanied by an official explanation serving as a reference for reasoning evaluation. Six LLMs were evaluated, with overall accuracy ranging from 34.0% to 73.9% across three distinct performance tiers. To assess reasoning quality, an LLM-based judge evaluated correctly answered responses using a multidimensional rubric. Although responses generally demonstrated strong problem comprehension, they scored lower in factual and procedural accuracy and distractor analysis, indicating that correct answers were not necessarily supported by technically sound justifications. Response clarity was also weakly indicative of technical quality. Furthermore, longer reasoning traces were associated with lower accuracy in both frontier reasoning models, suggesting that reasoning length may reflect question difficulty rather than improved problem-solving performance. Reasoning-token consumption may therefore provide an additional signal for identifying potentially unreliable responses. NEBench R1 establishes a benchmark for evaluating both answer accuracy and reasoning quality, providing a foundation for assessing the domain competence and reliability of LLMs in nuclear engineering. 

---
# Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching 

**Authors**: Giannis Daras, Marios Papachristou, Adam Klivans, Antonio Torralba, Constantinos Daskalakis  

**Link**: [PDF](https://arxiv.org/pdf/2611.23229)  

**Abstract**: Collecting large-scale, diverse, and high-quality post-training datasets for autoregressive language models is challenging. Consequently, Supervised Fine-Tuning (SFT) is typically applied to datasets significantly smaller than those used for pre-training, which leads to overfitting and the catastrophic forgetting of the tails of the data distribution. To prevent this collapse, we introduce Ambient SFT, a principled framework for leveraging imperfect predictions to augment the fine-tuning set. In our method, a "frozen" version of the pre-trained model serves as a noisy teacher that the fine-tuned model must match at a coarse level. In particular, we require that the high-temperature versions of the conditional distributions produced by the post-training model should match the high-temperature versions of the pre-training ones. The temperature hyperparameter controls the level of trust in the pre-trained model, with the limit $T\to \infty$ reverting back to regular SFT and $T=1$ to knowledge distillation. We demonstrate that our approach outperforms standard SFT across domain adaptation, instruction-tuning, and reasoning, leading for up to $21\%$ average relative improvement to standard SFT. 

---
# StringQFT: A Multiuse Textbook Resource for AI-Assisted Theoretical Physics 

**Authors**: Xingyang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.23225)  

**Abstract**: We present StringQFT as a textbook collection for string constructions of quantum field theory and as infrastructure for AI-assisted theoretical physics. The collection comprises 18 volumes, 190 chapters and 3,795 pages, covering brane probes, brane webs, geometric engineering, F-theory constructions, wrapped M5-branes and exact worldsheet constructions. Its open-book benchmark inventory contains 1,073 items, including 499 with automatic finite-answer checking. For retrieval-augmented generation, it provides 1,371 section records linked to textbook sources and PDF locations, together with a local lexical retriever. Training-data preparation is illustrated by 21 private, source-linked examples. For agents, it exposes 19 calculation tools with specified inputs and finite domains, connecting tool use to textbook derivations. We illustrate these uses through specific questions, retrieved passages and finite calculations. The textbook collection and accompanying resources are available at this https URL. 

---
# Do LLM Agents Still Need to Be Told How to Self-Improve? 

**Authors**: Shobhita Sundaram, Cansu Sancaktar, Phillip Isola  

**Link**: [PDF](https://arxiv.org/pdf/2611.23219)  

**Abstract**: LLM agents increasingly drive their own learning. Self-improvement methods delegate various components of model optimization to the models themselves. As models become more capable, this motivates a basic question: do these increasingly capable models still need us to specify how they should self-improve? In this paper we study an autonomous paradigm in which, given basic primitives, resources, and a general goal for self-improvement, agents construct their own self-learning algorithm for context-based optimization. Using the setting of gameplay as a testbed, a frozen LLM agent is given a compute budget and tools for collecting experience and updating text documents, and allowed to freely experiment without an externally imposed loop or access to real-world data. We find that sufficiently capable models do self-improve under these conditions, demonstrating strategies that align with classic cognitive frameworks for self-regulated learning and constructs from machine learning literature on self-generated curricula. Furthermore, we study the extent to which the learned text artifacts transfer to external benchmarks, observing improvement across different game horizons, mechanics, and 3D navigation. 

---
# Higher-Scoring Coding Models Did Not Get More Work Done 

**Authors**: Yining Hua  

**Link**: [PDF](https://arxiv.org/pdf/2611.23216)  

**Abstract**: New coding models advertise themselves on score improvements of coding benchmarks. Yet no study has tested if a model that scores higher on these benchmarks gets more real work done. On June 12, 2026, a U.S. export control directive banned access to Claude Fable 5 for 19 days, which forced users to switch to other models. During this time, most former Fable 5 users switched to Opus 4.8 (11 points lower on SWE-Bench Pro compared to Fable 5). We treat this suspension as a natural experiment, where the people and projects stayed the same and only the model changed, so we could isolate the effect of Fable 5 on user productivity. We tracked GitHub pull requests of 297 Fable 5 users before, during, and after the suspension. We used 8,908 developers who stayed on Opus 4.8 all the time as the control group. We also repeated the comparison across five model upgrades before and after the suspension in 2026 (1,208 developers, 15,442 pull requests). Fitting a difference-in-differences model with developer and date fixed effects, we found that having access to Fable 5 increased the 7-day merge rate of pull requests by only 1.4 percentage points [-1.4; +4.3], which was not statistically significant. Similarly, no significant increases of merge rates were found with the upgrades (weighted mean: -0.9 percentage points [-1.9; +0.2]); all five estimates were negative (95% intervals included 0, did not differ from each other). We also did not observe a significant increase in the number of pull requests merged per week (+2.4% [-4.9%; +10.2%]). 

---
# Reading Scene States: Uncovering Safety Failures in Vision-Language Models 

**Authors**: Tongxi Wu, Yuyang Cheng, Jiacheng Xian, Menglin Yang, Kai Chen, Jian Zhang, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.23206)  

**Abstract**: Vision-language models (VLMs) can provide harmful assistance by interpreting visual events, even without an explicit request for the underlying harmful objective. We identify this \textbf{scene-state safety vulnerability} and introduce \textbf{SceneSafetyBench}, comprising 375 image--question pairs spanning Initial, Medium, and Ultimate scene states and five visual presentations. Across eight VLMs, scene-state inputs elicit harmful assistance with attack success rates (ASRs) of 45.6--72.3\%. Comparative analysis across three VLMs shows that scene-state inputs most closely resemble QR-related images in their layer-wise safety-related signal profiles, while exhibiting lower average late-layer readouts and distinct signal dynamics during generation. To turn this vulnerability into a systematic attack, we propose \textbf{S3TA}, which translates harmful objectives into scene states, queries image--question pairs independently, and assembles the responses. On seven MM-SafetyBench categories across eight open-weight and commercial VLMs, S3TA achieves a model-average ASR of \textbf{71.2\%, exceeding the strongest baseline by 6.1 percentage points}. Text-side filters and visual moderation provide limited protection. These findings highlight the need to safeguard the assistance models infer through scene-grounded reasoning, beyond screening input content. 

---
# Fine-Tuning Gemma 4 Edge Models for MATLAB Signal-Processing Code with SFT and QLoRA on a Laptop iGPU 

**Authors**: Mehmet Altan Toksöz  

**Link**: [PDF](https://arxiv.org/pdf/2611.23205)  

**Abstract**: Small language models that run on a laptop are attractive for writing signal-processing code offline, but they make systematic errors on digital signal processing (DSP) conventions. We apply supervised fine-tuning (SFT) to the Gemma 4 edge models E2B and E4B with 66 checked examples covering 11 signal-processing MATLAB subtasks, using low-rank adaptation (LoRA) and quantized LoRA (QLoRA) on the integrated GPU (iGPU) of a laptop, and deploy the 8-bit integer (int8) models with the adapters through OpenVINO. On held-out variants scored by deterministic checkers, the pass rate rises from 18/66 to 37/66 (LoRA) and 37/66 (QLoRA) for E2B and from 32/66 to 47/66 for E4B (QLoRA). Reading all 159 failing programs shows that the adapters remove most errors of convention, such as the sign of a lag, units and array shapes, but few errors in building signals and noise. Applied at run time, the adapter costs 17-19% of the generation speed. 

---
# Structure--Residual Inductive Biases for Language Model Pretraining 

**Authors**: Hai Huang, Bartosz Żuk, Rassul Magauin, Jiaxin Huang, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.23197)  

**Abstract**: Next-token prediction is relied upon to learn both predictive behavior and internal representations. We ask whether representation structure itself should be an explicit inductive bias. Motivated by the two-part coding view of Kolmogorov complexity, we study a structure--residual principle in which a representation separates structure from realization detail. We instantiate this principle in two complementary ways. Boundary Bottleneck Concept (BBC) routes information from token segments through boundary carriers while preserving local autoregressive computation. Continuous Concept (CC) regularizes representation geometry and induces a spectral decomposition into low-frequency structure and high-frequency residual. We derive coding bounds for both views and show that multiplicity-corrected boundary attention becomes exact as the within-segment path energy vanishes. Empirically, BBC and CC exhibit strikingly different depth preferences: BBC works best in early layers, whereas CC is most effective in later layers. At 300M training tokens, combining BBC and CC improves NLL by 1.44\% on GPT-2 Medium. These gains persist with training runs up to 20B tokens. Representation diagnostics show that BBC increases the relative predictive importance of boundary states, while CC reduces spectral residual energy. Together, these results suggest that structure--residual biases can improve language-model pretraining beyond next-token prediction. Code: this https URL 

---
# Retrieval-Augmented Correction for Generalist Medical Image Segmentation 

**Authors**: Vincenzo Marcianò, Xiaoming Zhang, Michela Antonelli, Sebastien Ourselin, Maria A Zuluaga  

**Link**: [PDF](https://arxiv.org/pdf/2611.23189)  

**Abstract**: Automated segmentation is routinely used as the first step in quantitative image-based clinical workflows, where predicted masks are subsequently reviewed and corrected by human operators. Automated correction of the residual structural errors is currently addressed one anatomy at a time, so the number of trained correctors scales with the number of segmentation targets. We present GRACE, a retrieval-augmented refiner that edits an input mask under context drawn from a shared knowledge base of verified prior cases. The refiner is trained once on a multi-organ corpus spanning compact organs, cavities, branching vessels, and tumour lesions. In the held-out dataset-transfer evaluation, three held-out datasets query a fixed knowledge base built from training data only. A separate leave-one-anatomy-out experiment tests parameter-free index extension: verified cases of an unseen solid organ enter the knowledge base, and the refiner stays frozen. A light shape-aware projection head on a frozen multimodal encoder maps embedding similarity to the shape agreement of the masks, so the knowledge base returns cases that the refiner can use as a shape prior. We evaluate GRACE on 18 public datasets spanning CT, MRI and ultrasound, using nine refinement configurations and seven upstream segmentors that span weak, mid, and strong operating regimes. We report each change with the DSC of its input mask. On the weak tier, the input DSC is 0.386 and GRACE adds up to +0.35 DSC. On a strong segmentor, the input DSC is 0.932 and the mean change is small. Transfer to an imaging modality that is absent from training and from the index is partial and depends on the anatomy, and remains an open research direction. Our results suggest retrieval-conditioned refinement is a practical alternative to anatomy-specific correction models. Code: this https URL 

---
# Say Less, Hallucinate Less? Diagnosing and Restoring Object Coverage in Preference-Optimized Vision-Language Models 

**Authors**: Hansang Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.23179)  

**Abstract**: Preference optimization (PO) is the standard way to reduce object hallucination in large vision-language models. We find that four public PO checkpoints of LLaVA-1.5-7B also lower object coverage. They name fewer object categories, and their lower hallucination rates bring no significant gain in set-level object F1 on CHAIR. We therefore ask how much of the PO gain the base model can reproduce by stopping earlier. A positive end-of-sequence (EOS) logit bias lets the base model reach the recall of each PO model, and this early stopping alone reproduces 58 to 86 percent of the CHAIRi gain of three checkpoints on two disjoint COCO sets. Writing longer does not bring the lost categories back. Under greedy decoding, a negative EOS bias recovers only 2 to 9 percent of the lost CHAIR recall of these checkpoints, and 94 to 96 percent of the added object mentions repeat categories already named. On CHAIR, the base caption of the same image still names about half of the missed object categories. We therefore propose base-guided object-category restoration, which appends high-confidence base-only categories to the PO caption without training. With thresholds fixed on a development set, it brings three checkpoints within one point of the base recall on a held-out COCO set with a lower distinct-object hallucination rate than the base model. Against the recall-matched base stopping curve, its gain holds for RLAIF-V near the base recall and for RLAIF-V and CHAIR-DPO at an intermediate recall. We will release the code of the recall-controlled evaluation. 

---
# Inference after coding agents 

**Authors**: Florian Gunsilius, David Van Dijcke, Kaspar Wüthrich  

**Link**: [PDF](https://arxiv.org/pdf/2611.23174)  

**Abstract**: Empirical research is changing rapidly as AI coding agents are becoming increasingly proficient in every step of it, from writing code to manipulating data. We argue that while the potential of coding agents for science is vast, their use in hypothesis testing poses a substantial threat to classical inference, as these agents enable a supercharged form of selective inference. In particular, we argue that coding agents carrying out empirical research introduce a new type of post-selection inference that combines human-like, open-ended search over an implicit space of estimators with programmatic replayability of both the selection and the estimation process. Indeed, we find that on simulated data with zero effects, four frontier agents report significant p-values between 54% and 98% of the time when asked to produce a compelling finding. We propose a simple procedure for detecting and correcting these distortions. After an agent has completed its analysis, we reset and replay it on subsamples of the data to estimate the agent's relative probability of selecting that analysis when it looks statistically favorable. For agents with stable selection behavior, the resulting corrected test is asymptotically valid and consistent conditional on selection. Importantly, the correction is learned from the black-box agent's actual choices rather than a model of how it searches, so it remains valid even as AI technology continues to evolve rapidly. In simulations, the correction robustly lowers the high false rejection rates to near the nominal 5%. We illustrate the method on applications from two published papers. 

---
# Attribution Through the Decoder: Prediction-Space Explanations for Structured Outputs 

**Authors**: Zaer Bin Zaqy, Brinnae Bent  

**Link**: [PDF](https://arxiv.org/pdf/2611.23161)  

**Abstract**: Networks that predict angles, gaze directions, and rotations do not output these quantities directly: they emit raw coordinates that a fixed decoder, such as normalization, maps to the final prediction. Attribution methods are usually applied to the raw coordinates, so they can credit inputs for changes the decoder discards, such as rescaling a vector when only its direction matters. The resulting explanation can highlight inputs that have no effect on the prediction. We instead attribute through the decoder differential, which for normalization decoders reduces to a scaled tangent projection. We characterize invariance to output rescaling, show that every fixed smooth scalar target on a sphere has points of zero tangential sensitivity, and show that an endpoint-projected variant of Integrated Gradients can still attribute to scale-only inputs, whereas direct angular Integrated Gradients cannot. On synthetic tasks with analytically known relevance, ambient attribution assigns up to 72% of its mass to an irrelevant region. On MPIIGaze validation data, tangent Grad-CAM produces 44% greater prediction displacement under 10% pixel deletion. On PASCAL3D+ with a 6D rotation decoder, the full differential improves mean deletion displacement by 14.1 degrees over the ambient coordinate sum and by 13.8 degrees over a decoder-invariant scalar. Results for a two-polarity tangent LayerCAM variant are mixed: it exceeds ambient-sum LayerCAM on circular regression but no difference is detected on quaternion pose. These results separate two failure sources, radial contamination and scalarization loss. The approach uses existing attribution methods with decoder-aware targets, although some variants also require changes to attribution aggregation. An invariant attribution can still miss inputs that drive the prediction, so invariance and deletion performance should each be checked. 

---
# Learning Stratified Policies for Inference-Time Objectives 

**Authors**: Mridul Mahajan, Xuezhou Zhang, Aldo Pacchiano  

**Link**: [PDF](https://arxiv.org/pdf/2611.23154)  

**Abstract**: Inference-time scaling improves the performance of large language models by sampling multiple responses and evaluating their collective utility. However, many existing approaches obtain these samples by repeatedly sampling from a single policy. In this work, we study if the sample budget at inference time can instead be allocated across a population of policies. We introduce stratified policies, where each of $k$ learned policies contributes one response to a $k$-sample inference-time objective. We realize this policy population using separate LoRA adapters on a shared language model backbone and optimize it for pass@$k$. We theoretically motivate the approach by analyzing the linear setting; a population of LoRAs can achieve zero best-of-$k$ approximation error in settings where every single LoRA has strictly positive prediction error. Empirically, we demonstrate that stratified policies consistently improve pass@$k$ over a single-policy baseline across multiple reasoning tasks. Overall, our results highlight how distributing samples across complementary policies can be a more effective use of sample budget than repeated i.i.d. sampling from a single policy at inference time. 

---
# Mind the Gap: A Deployed Multi-Agent System for Documentation Maintenance 

**Authors**: Lorenzo Masoero  

**Link**: [PDF](https://arxiv.org/pdf/2611.23142)  

**Abstract**: The "AI revolution" has enabled teams across the technology industry to deliver software features at unprecedented velocity. While code evolves faster than ever, customer-facing documentation often struggles to stay up to speed. As users increasingly rely on LLM-based agents that consume documentation as their primary interface, the impact of this documentation gap worsens: stale information propagates and is treated as authoritative. Signals for identifying where documentation has fallen behind already exist in abundance, both explicit (e.g., resolved support tickets revealing what documentation lacks) and implicit (e.g., code-documentation discrepancies detectable from automated loops). We deploy a multi-agent system, built around a triage agent we call "TriAgent," that continuously extracts these signals and repairs documentation drift. Over 43 days of shadow-mode operation, our system triaged 499 tickets and surfaced 92 gaps in the documentation. Using a panel of LLM judges from independent model families, 81.1% of TriAgent's verifiable assertions were correct (245 of the 302 assertions the human resolution addresses; majority of three judges); on length-comparable pairs the panel preferred TriAgent's response in 70.5% of pairs and the human engineer's in 23.2% (the rest had no judge majority). Importantly, agent- and human-sourced replies tended to be complementary, with the human often surfacing insights that were either buried within the existing documentation or missing from it entirely; our system turns each such insight into a documentation gap and, after human review, a fix. Repairing a gap measurably improved the agent's later answer: when the agent re-answered the same ticket after the repair, blinded judges preferred the post-repair answer over its own pre-repair answer 71.4% of the time (n=35). A blinded human audit confirms strong alignment on self-improvement. 

---
# Natural Language Programming: Rethinking Language, Memory, and Cognition in the Era of Generative AI 

**Authors**: Giovanni Adorni, Daniele Grosso  

**Link**: [PDF](https://arxiv.org/pdf/2611.23106)  

**Abstract**: Generative Artificial Intelligence is transforming the role of natural language in computational systems. In classical Natural Language Processing (NLP), language was treated as an object of computation: something to analyse, classify, translate, or generate. Large Language Models (LLMs) challenge this perspective, enabling natural language to function increasingly as an executable cognitive medium through which humans structure reasoning, orchestrate problem solving, and regulate hybrid human/AI workflows.
This paper introduces Natural Language Programming (NLPg) as a theoretical and epistemological extension of traditional NLP. Rather than focusing on language understanding alone, NLPg conceptualises prompts as dynamic cognitive artefacts that externalise goals, constraints, assumptions, validation criteria, and intermediate reasoning states. Prompting is thus not merely a technical interaction technique, but a form of dialogic cognitive programming operating within distributed human/AI memory architectures.
Building on theories of distributed cognition, extended mind, metacognition, and dialogic learning, the paper proposes the notion of Dialogic Working Memory, describing how prompts and conversational states function as temporary external memory structures supporting reasoning continuity and reflective regulation. We further map existing memory architectures (BTMA, SKGE, RMR) onto this framework, illustrating how episodic, semantic, and metacognitive layers support long-horizon conversational coherence.
Finally, implications for computational linguistics and AI/mediated knowledge systems are discussed, suggesting a post symbolic and post interface paradigm in which natural language operates simultaneously as communication medium, executable specification, cognitive scaffold, and distributed memory architecture. 

---
# What's in a Tree ? Hierarchical $n$-gram models as controlled approximators of language 

**Authors**: Aritra Das, Francesco Cagnetta, Darshil Doshi, Victor V. Albert, Maissam Barkeshli, Andrey Gromov  

**Link**: [PDF](https://arxiv.org/pdf/2611.23093)  

**Abstract**: Hierarchical models have been proposed as synthetic settings for studying language. However, their relevance to realistic language corpora remains underexplored. In this work, we introduce tree-grams, autoregressive generative $n$-gram models that build the conditional probability tensor $p(x_n \mid x_{1:n-1})$ through a hierarchical composition of order-2 and order-3 tensors. $n$ controls the maximal context window, whereas a rank hyperparameter $d$ controls model capacity, yielding an interpretable and systematically scalable family of approximators of language. We show how tree-grams outperform existing $n$-gram baselines, while remaining competitive with deep transformer-based language models. We demonstrate the utility of such controlled approximations by using tree-grams as generative models. In particular, we show that hyperparameter transfer under $\mu$P can be made more robust when narrower models are tuned on simpler data, generated by tree-grams with moderate $n$ and $d$. In addition, we show that entropy and effective range of correlations control performance gap after fine-tuning. This is because tree-grams allow us to explicitly control horizon and capacity. When allowed to route across candidate topologies, we observe emergent sparsity in tree-grams. Such concentrated routing permits pruning upto 95\% of topologies without significant change in performance. 

---
# BugGen 2.0: Rethinking the Roles of LLMs for Scalable RTL Bug Synthesis 

**Authors**: Minh Luu, Michael Quinn, Aakash Tyagi, David Houngninou, Jiang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.23091)  

**Abstract**: AI-assisted RTL localization and debugging need ground-truth bug datasets with known locations and observable failures, while testbench grading uses injected bugs to measure detection capability. Creating such datasets at scale remains a major EDA challenge. BugGen 1.0 shows that Large Language Models (LLMs) can generate context-aware RTL mutations. Still, it relies on the model for both choosing where to mutate and generating the edit, which increases inference costs and can lead to invalid choices. BugGen 2.0 changes this division of labor. Deterministic algorithms handle structural analysis, target selection, and validation, while the LLM focuses on localized semantic rewriting. Across four OpenTitan IP blocks and FabScalar with three LLMs, BugGen 2.0 reaches every requested quota of compilable bug scenarios that cause at least two test failures, with 14-16 of 16 configured mutation types represented. In completed same-model comparisons, it reduces token use by 6.8x-15.4x relative to BugGen 1.0 and 1.5x-7.9x relative to Firefly. BugGen 2.0 also expands the dataset without simulating every new bug. After finding accepted scenarios with observable failures, it generates alternative mutations at the same locations, simulates only a subset, and estimates acceptance for the rest. This skips 63.6% of AES and 33.9% of FabScalar variant simulations. Full simulation finds that 77.7% of AES variants and 95.2% of FabScalar variants are accepted, with a 1.09% estimation error on the directly comparable FabScalar set. These results show that focusing LLMs on semantic rewriting lowers inference and simulation cost while improving consistency across models and designs. 

---
# Learning to Undo: Transfer Reinforcement Learning under State Space Transformations 

**Authors**: Mridul Mahajan, Aldo Pacchiano, Xuezhou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.23081)  

**Abstract**: Transfer learning in reinforcement learning (RL) has shown strong empirical success. In this work, we take a more principled perspective by studying when and how transferring knowledge between MDPs (a source and a target) can be provably beneficial. Specifically, we consider the case where there exists an undo map such that applying this map to the target's state space recovers the source exactly. We propose an algorithm that learns this map by matching state feature statistics gathered from both MDPs, and then uses it to transfer the source policy to the target. We theoretically justify the algorithm by analyzing the setting when the undo map is linear and the source is linearly-$Q^\star$ realizable, where our approach has strictly better sample complexity than tabula rasa RL in the target MDP. Empirically, we demonstrate that these benefits extend beyond this regime: on challenging continuous control tasks and Atari games, our method achieves significantly better sample efficiency. Overall, our results highlight how shared structure between tasks can be leveraged for efficient transfer of policies across environments. 

---
# Runtime Verification of Multiple Natural Language Criteria for Agent Governance 

**Authors**: Silviu Pitis, Parand A. Alamdari, Jessica Tang, Toryn Q. Klassen, Sheila A. McIlraith  

**Link**: [PDF](https://arxiv.org/pdf/2611.23074)  

**Abstract**: AI agents are governed by hundreds of natural language criteria, yet existing verifiers cannot efficiently evaluate an agent's output against a large set of such criteria and report which are satisfied, violated, or inapplicable. We introduce the VFM, a tree-attention verifier that encodes the shared context and target text, and scores N natural language criteria in parallel, returning independent per-criterion ternary verdicts. The VFM is pretrained on CriteriaBank, a corpus of 394K (context, target, criterion, verdict) tuples spanning over 321K distinct criterion strings. A finetuned VFM-4B outperforms GPT-5.4 on a synthetic governance benchmark. On the DynaBench multi-criterion benchmark, it outperforms DynaGuard-4B on trace-level failure detection, and its per-criterion verdicts localize the violated rule and support focused escalation to a larger chain-of-thought judge. We show that CriteriaBank pretraining improves cross-domain transfer, particularly in the low-data regime, while matching a strong direct-finetuning control at full data. On a single H100, VFM-4B evaluates 1,000 criteria against a 4K-token context in under 2s at <11 GB VRAM, making it suitable for runtime verification. 

---
# ReTouchGS: Ultralight Render-free Aesthetic Color Enhancement and Transfer for 3D Gaussian Splatting 

**Authors**: Chuanzhi Xu, Boyu Wei, Chengkun Yue, Xuanhua Yin, Yiyang Li, Haodong Chen, Qiang Qu, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2611.23054)  

**Abstract**: 3D Gaussian Splatting (3DGS) enables photorealistic scene reconstruction. Existing 3DGS color transfer and interactive editing methods are guided by reference images and user-specified color adjustments. However, in many applications, users simply want aesthetically guided color enhancement that automatically makes the scene more visually appealing while maintaining consistent colors and tones across viewpoints. We present ReTouchGS, to our knowledge the first framework for reference-free aesthetic color enhancement of pretrained 3DGS models. It predicts a scene-level color lookup table (LUT) from the model and its associated camera parameters. ReTouchGS also supports reference-guided color transfer by fitting a scene-level LUT using one edited view of the same scene and its corresponding original render. A shared blend-aware color refinement network with only 44.7K parameters refines an analytic color update to realize either color mapping without rendering. It preserves geometry and delivers standard 3DGS assets in seconds. We also design the ReTouch3D benchmark to support the evaluation of aesthetic color enhancement and reference-guided color transfer across viewpoints. Experiments demonstrate high color accuracy and strong generalization to unseen scenes in both aesthetic color enhancement and reference-guided color transfer. 

---
# Check Your Blind Spots: Decision-Guided Perception by Probing the Planner with Hypothetical Objects 

**Authors**: Hoang Pham Cong, Long D. Nguyen, Ho Viet Duc Luong, Nguyen Tran Minh Nhat, Binh P. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.23030)  

**Abstract**: Not every perception error changes the planner's action. Yet adaptive-resolution perception chooses where to spend high-resolution computation from image cues, which can neither reliably point to an object that low resolution missed nor tell which objects matter to the planner. We observe that the planner can say where an object would matter before anything is seen: inserting a hypothetical object at a location tests whether its action would change. Decision-Guided Speculative Perception (DGSP) treats the low-resolution scene and its action as a draft and probes the planner in this way to obtain a decision-sensitivity field, the locations where an unseen object would change the action. Only this field and the support, the detections on which the current action rests, are verified at high resolution, repeatedly until the action is stable. For two planner classes, we prove that DGSP returns exactly the action of whole-frame high-resolution perception if crop detections match whole-frame ones and the crops cover the field, conditions that real images meet approximately. On three driving datasets, it reproduces this action on about 98% of frames with about a quarter of the pixels. At the pixel cost of uniform downscaling, it repairs more of the draft's disagreements with this action, brakes falsely less often, and recovers many more of the decision-relevant objects that the draft misses. 

---
# How Web Research Agents Interpret User Priorities 

**Authors**: Robin Sharma, Parisa Suchdev  

**Link**: [PDF](https://arxiv.org/pdf/2611.22989)  

**Abstract**: Web research agents gather information online and write cited reports, including financial analysis. Users can state a priority to guide this research, but a priority may not say how to judge the options, leaving the agent to choose the criteria. We examine how two such agents, built on OpenAI and Anthropic models, interpret a stated priority. In 36 runs, the agents compared Microsoft and Alphabet as businesses. Prompts asked for special attention to growth or resilience, or gave no priority. We analyzed reports, searches, returned sources and cited sources. Of the 24 reports given a priority, 21 focused on it, but their criteria varied. Under the growth priority, OpenAI reports that ranked the companies accounted for risk or predictability and favored Microsoft, while the Anthropic ones judged growth by its pace and favored Alphabet. Under the resilience priority, reports used partly different criteria, and most named Microsoft as the more resilient company. Of the search and source measures tested, only the Anthropic agent's search wording differed detectably by condition. These findings suggest evaluations should examine the criteria behind a report's judgements, not only its focus or accuracy. Interfaces could let users see and adjust those criteria. 

---
# Large Language Models Have Not Yet Transformed Science 

**Authors**: Ron Bekkerman, Nichole Li, Bahareh Kargar, Mark Somers  

**Link**: [PDF](https://arxiv.org/pdf/2611.22975)  

**Abstract**: The emergence of large language models (LLMs) in late 2022 has raised questions about the future of science, but assessing their impact requires understanding how scientific practices were already evolving. We analyze 2.5 million full-text papers across five disciplines from 2015 to 2025, tracing trends over a broad spectrum of lexical, structural, epistemological, and social dimensions. Aside from frequency shifts in some words, most pre-2023 trends continued with limited detectable breaks. Among the notable exceptions, growth in computer science coauthor teams accelerated significantly in Asia, while apparent declines in biomedical topic diversity largely reflected standardization in article formats. Our study establishes an empirical foundation for monitoring the health of global science as LLM adoption deepens. 

---
# SEAM: Computer-Use Agents Know the Harm Between Screens, and Commit It Anyway 

**Authors**: Yuxuan Zhou, Chen Xiong, Haitong Liu, Kuofeng Gao, Yang Bai, Zhiyuan He, Tao Dai, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2611.22971)  

**Abstract**: Safety benchmarks for computer-use agents (CUAs) place the harm either in the intent of the task or on a single screen: a malicious instruction, an injected command, a phishing page. An agent that checks the task and each screen in isolation can pass them, so they do not test whether it combines facts seen on different screens. Yet in delegated work, harm often arises exactly this way: a fact about a person on one screen, a fact about an item on another, and a legitimate task that binds them. We introduce SEAM (Safety Execution Across Multiple screens), a benchmark of such cross-screen compositional harm: 600 multi-step, cross-application OSWorld tasks in which the instruction and every single observation are benign and only their combination is harmful. Minimal-pair variants that differ in one place, scored from the server log rather than by an LLM judge, locate where an agent fails. General-purpose open-weight and flagship agents carry out the harm in up to 92% of runs, yet decline when the user states it in one sentence. Agents open the decisive pages, so exposure is not the primary bottleneck. Asked about those pages, they flag the harm, and with thinking they often write the conflict into their reasoning and confirm anyway. Safety recognition thus does not reliably translate into safe execution, and the translation depends on the information channel: the same warning stops agents far more often in the user's instruction than on screen. Proprietary models are markedly more cautious. We release SEAM with its generator and quality gates at this https URL . 

---
# CodeReel: Accelerating Code-to-Video Agents with Exact Frame Reuse and Blocking-Aware Scheduling 

**Authors**: Junyi Shen, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22967)  

**Abstract**: Coding agents make videos by writing and refining programs in a build--inspect--edit loop. A finished film costs a median of $2.85 in model calls and 11 minutes, over a third of that time in builds that block the agent. We find that much of this time is avoidable, for two reasons. First, 37.5% of the frames of a session's later films were drawn before, many shifted in time or inside an edited scene. Second, half of the builds stop at a check, where the agent needs only the verdict, but which builds stop is unknown until the checks finish. We present CodeReel, a training-free execution layer behind agents' existing commands. Exact frame reuse keys each frame by its drawing commands, regardless of its position in time, and reuses each frame whose key recurs. Blocking-aware scheduling runs the checks first and starts the render alongside them, at idle priority, unless the session's recent builds make a pass unlikely. Scheduling spends CPU to answer sooner and reuse returns it; exact keys let later versions use frames rendered for a failed one as the agent thinks. We introduce EditLoop, a benchmark of replayable agent sessions whose films are checked against renders from scratch. On it, CodeReel cuts the agent's wait by 36.3% against Rebuild, a warm pipeline with parallel rendering, and by 57.9% against Sequential, today's step-by-step pipeline. On time-shifting edits its reuse renders 43.9%, close to the 43.0% whose pixels are new, while Manim-style caching renders 99.8%. With real agents, the median wait per build is 0.62x Rebuild's in Claude Code, on topics never used in development, and 0.67x in Codex CLI. The gain holds for five models and two frameworks. 

---
# High-Dimensional Black-Box Optimization via Guidance Sampling of Diffusion Models 

**Authors**: Xiaoyan Hu, Rohit Sonker, Jalal Butt, Jiayu Chen, Egemen Kolemen, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2611.22965)  

**Abstract**: Many real-world optimization problems involve optimizing black-box objectives for which only a limited number of evaluations are available. Bayesian optimization (BO) is a powerful framework for such problems, but its effectiveness often deteriorates in high-dimensional spaces. Recent work leverages diffusion models for black-box optimization, often through pre-training a reward-conditioned score network to steer generation towards high-reward samples. Alternatively, gradient-based guidance methods assume access to a differentiable reward function from which guidance gradients are queried. However, these approaches either require large amounts of labeled data for offline training or access to exact reward gradients, limiting their applicability in black-box settings. We propose BReaD (Black-box Reward-guided Sampling of Diffusion Model), a guidance sampling framework for optimizing black-box rewards without assuming access to a labeled dataset or reward gradients. The proposed method uses a Gaussian process (GP) as a differentiable surrogate for the unknown objective and actively queries labels for the generated samples. We provide a theoretical analysis of the total-variation distance between the generated distribution and the target distribution. Using Upper Confidence Bound (UCB) as guidance signals, our method balances exploration and exploitation during generation and enables the discovery of high-reward regions in the data space. Numerical experiments on scientific discovery and image generation tasks demonstrate that our framework can effectively optimize black-box rewards in low-data scenarios. 

---
# Hierarchical Diffusion Transformer for End-to-End Image Generation 

**Authors**: Xiaoyu Wu, Yifei Wang, Weihang Guo, Chen Wei, Lydia E Kavraki, Niloofar Mireshghallah, Steven Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22962)  

**Abstract**: Pixel-space diffusion learns visual representations and image generation jointly in a single end-to-end training stage, without a separately pretrained visual encoder. However, plain pixel-space Transformers still trail leading two-stage latent diffusion models in generation quality and training efficiency. We hypothesize that part of this gap arises from retaining one token per patch throughout the backbone, which does not explicitly encourage a visual hierarchy that summarizes related content at progressively coarser levels. To address this, we introduce Hierarchical Diffusion Transformers (HiT), which progressively group related tokens and represent each group with a single token. The main Transformer bottleneck processes these compact group tokens, and ungrouping restores features to their original image positions. A compression budget regulates the average compression ratio across the batch, allowing group memberships and counts to vary across images and diffusion timesteps during both training and generation. The model exhibits emergent semantic structure, learning to group coherent objects and object parts without segmentation labels. In text-to-image generation, HiT further learns to group caption words with image patches depicting the same concepts, without region annotations. HiT achieves 1.48 FID on ImageNet $256\times256$ and reaches comparable generation quality to JiT using approximately 40\% of its cumulative training FLOPs. For text-to-image generation, HiT exceeds UniLiP's average score across five benchmarks with roughly $6\times$ fewer denoiser parameters and $3\times$ fewer forward FLOPs. These results suggest that learning a visual hierarchy within the generator breaks down the barrier to stronger image generation without external visual pretraining. 

---
# Projected Penalty-Gradient Tracking for Communication-Efficient Byzantine-Resilient Federated Learning 

**Authors**: Yanjie Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.22929)  

**Abstract**: We investigate communication-efficient federated learning over heterogeneous datasets in the presence of Byzantine adversaries capable of uploading arbitrary information. Penalty-based methods bound the influence of each worker through the bounded gradient of a penalty function; however, the bound is lost once the uploads are compressed, since a Byzantine worker can accumulate an unbounded state at the server. To retain the Byzantine resilience under compressed uploads, we propose a projected penalty-gradient tracking framework, where each honest worker uploads a compressed residual of its penalty gradient and the server projects each reconstructed state onto the known penalty-gradient ball. The framework accommodates any contractive compressor and yields two algorithms: fault-resilient proximal gradient with consistent sparsification (FRPG-CS), whose shared mask is protected by freshly drawn reserve rows, and lazily aggregated FRPG (LAG-FRPG), whose uploads are event-triggered. For smooth and strongly convex losses, we establish a finite-time bound on the expected objective gap, where the Byzantine error floor is independent of the compressor and the compression error decays faster than the optimization error. We also bound the expected number of lazy uploads and derive certificates that separate the finite-penalty model bias from the optimization error. Experiments on MNIST and Fashion-MNIST show that the proposed algorithms attain the highest accuracy among twelve methods under the Min-Max attack, whereas a compressed-momentum benchmark and a local-update benchmark remain preferable in attack-free accuracy and total traffic, respectively. 

---
# When Do Typed Decision Models Work in Transportation? Evidence from Travel Choice and Traffic Signal Control 

**Authors**: Hongyang Zhang, Boyuan Deng, Siyuan Feng, Kai Chen, Junbo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22926)  

**Abstract**: Typed decision models combine natural-language understanding with a structured interface that returns a selected action and an option-level probability distribution without generating lengthy textual responses. Their low inference cost and response latency make them attractive for transportation problems with predefined alternatives. This paper investigates their effectiveness, robustness, and efficiency in two representative transportation tasks: individual travel-mode choice prediction and sequential traffic signal control. We systematically evaluate TypeSafe Jev and GPT-6 Luna Decisions against conventional transportation methods, reinforcement learning algorithms, and four chat-based large language models. Experiments on the Swissmetro dataset show that typed decision models achieve lower prediction accuracy than traditional discrete choice models and the strongest chat-based baseline, while their performance varies substantially with input representation and question framing. In closed-loop traffic signal control experiments, typed decision models demonstrate competitive performance against established controllers in several scenarios, but exhibit performance degradation under more challenging network conditions. Furthermore, the typed models achieve lower inference latency and computational cost than the evaluated chat-based large language models, highlighting their potential for efficient, zero-shot transportation decision-making. Overall, these findings demonstrate both the potential and limitations of typed decision models, emphasizing the importance of input design, task characteristics, and robustness evaluation for their effective application in transportation systems. 

---
# Stopping Is Not the Limit: Understanding and Eliciting Agent Capability in Long-Horizon Vulnerability Exploitation 

**Authors**: Deokhyung Kang, Minjae Gwon, Sangdon Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.22888)  

**Abstract**: Vulnerability exploitation tests whether language model agents can develop working exploits from known software flaws. Sequential test-time scaling can improve exploitation performance, but larger budgets do not ensure sustained exploration: agents often stop with budget remaining. Yet what happens when execution continues beyond stopping remains underexplored. We investigate this gap on ExploitBench using DeepSeek-V4-Flash-0731 and MiMo-V2.6-Pro. We find additional verified gains after the first stop response, but gains diminish over longer execution. We identify two patterns associated with less productive continuation: an increasing proportion of stop responses within the same context and earlier stopping associated with summaries retaining impossibility judgments about unachieved capabilities. To test whether context management can elicit further capability, we propose SCoRe, which compacts context at stop bursts and marks these judgments as unproven. SCoRe improves verified capability over summary-based continuation and other evaluated context-management baselines. With SCoRe, MiMo reaches a capability coverage score of 67.0% at 2,000 turns, exceeding both 32.4% under the original stopping policy and the 58.6% computed for GPT-5.5 from ExploitBench's 300-turn reference results on the same tasks. These findings show that stopping need not mark an agent's exploitation limit, and that continued execution with context management can elicit substantial further capability. 

---
# Learning Agent-Centric Vision-Language-Action World Models for Autonomous Driving 

**Authors**: Guoqing Wang, Pin Tang, Xiangxuan Ren, Chao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.22880)  

**Abstract**: Recent progress in integrating vision-language-action (VLA) models with world modeling has significantly advanced end-to-end autonomous driving by allowing systems to reason about imagined futures instead of merely reacting to current observations. However, most world VLA models generate future frames at the pixel level with plenty of dense visual tokens. Although such generated futures can be visually plausible, they preserve many driving-irrelevant appearance details, i.e., only a small portion of the tokens directly encode critical planning-relevant factors, such as agent geometry, motion dynamics, and interactions. This introduces a significant performance bottleneck. An important question thus arises: what world state should a world VLA model reason over to support planning beyond pixel-level prediction? We argue that planning-oriented world modeling should focus on a compact and structured agent-centric latent state that captures the interactions between objects in the driving scene over time. As such, we introduce AgentWorld, an agent-centric VLA world model that uses discrete agent tokens jointly for reasoning and trajectory planning. Each token is dynamically grounded to a surrounding agent associated with its object's state, such as semantic identity, 3D geometry, motion, and interaction context. AgentWorld learns these tokens through three key processes: explicit geometric grounding, reasoning-oriented supervised fine-tuning, and agent-oriented reinforcement learning. Extensive open-loop and closed-loop experiments under diverse driving scenarios show that AgentWorld enhances planning capabilities, fully exploits visual evidence and agent dynamics, and yields interpretable agent-centric world states. 

---
# WATERFALL: Workflow for Adaptive Training with Evolutionary Reward Formulation and Automated Learning Loops 

**Authors**: Eleftherios Triantafyllidis, Filippos Christianos, Zhibin Li, Bernd Bickel  

**Link**: [PDF](https://arxiv.org/pdf/2611.22854)  

**Abstract**: Reward design remains a central bottleneck in reinforcement learning (RL), particularly in sparse-reward, long-horizon and partially observable settings requiring dependency chaining, memory, and deceptive affordance disambiguation. While recent foundation-model approaches reduce manual reward engineering, most assume goal-conditioning, task descriptions, expert demonstrations, or ground-truth metrics. These assumptions are problematic in partially observable environments, where task-relevant objectives, affordances, and dynamics must be discovered through interaction rather than disclosed a priori. We formalise this problem setting by introducing WATERFALL, an iterative population-based workflow for automated reward discovery under the strict observability constraints inherent in POMDP environments. WATERFALL achieves this by (i) synthesising diverse programmatic reward candidates via persona-conditioned generation, (ii) evaluating candidate behaviour from raw visual rollouts utilising a Swiss-system evolutionary tournament judged by vision-language models to identify elites, and (iii) leveraging longitudinal assessment histories to iteratively mutate, escalate, or prune candidates. We evaluate WATERFALL across fully observable and reveal-gated partially observable environments ranging from discrete to continuous, comparing against sparse-reward RL, intrinsic-motivation and foundation-model reward-synthesis baselines. Our results indicate that WATERFALL discovers and refines reward programs without access to privileged information, outperforming baselines that retain their native information advantages, most clearly in the POMDP regime. Extensive ablations substantiate that visual evidence, tournament-based selection, and iterative refinement are jointly important. Enforcing this contract is what separates autonomous reward discovery from benchmark-privileged reward shaping. 

---
# Better Rankers, Worse Retrievers: The Hidden Cost of RL Post-Training in Generative Recommendation 

**Authors**: Yuheng Chen, Fan Zhang, Yixuan Xu, Chao Yang, Yang Xiang, Can Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22838)  

**Abstract**: Generative recommenders decode the semantic ID of a user's next item with a language model, and post-training them with reinforcement learning (RL) after supervised fine-tuning (SFT) has become standard practice. The gains are mostly reported on the first ten results, yet in deployment such models hand hundreds of candidates to a downstream ranker or reward model. We show that RL post-training has a hidden cost: it improves accuracy at the top of the list (hit@K for small K) but, on all five datasets we study, lowers the recall of the candidate list (hit@K for K in the hundreds). The model becomes a better ranker and a worse retriever. Controlled experiments trace the loss to the negative advantages that RL assigns to the candidates it samples, the rollout. The root cause is that only one next item is logged per user, so RL treats every other candidate as wrong. Yet these candidates are the model's most plausible alternatives, and often the next items of other users with similar histories. Penalising them pushes down exactly what similar users choose next, and these items drop out of those users' candidate lists. This diagnosis points directly to a remedy: out-of-rollout anchoring, a single additional loss term that preserves the SFT model's ordering of the items outside the rollout. It keeps the gain at the top and restores recall to the SFT level on all five datasets, and it is effective under other RL configurations as well. To our knowledge, this is the first systematic study of this recall loss: we measure it, identify its cause, and show that, once the cause is known, a simple remedy suffices. 

---
# A tiny object detection dataset for high-altitude open-water search and rescue 

**Authors**: Hui Xu, Chenfei Hou, Henglai Wei, Hao Wang, Yidong Jin, Xiulin Niu, Haoyang Zhang, Hui Zhang, Tianshi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22825)  

**Abstract**: Large aerial platforms often operate above 1.5 km to improve the efficiency of open water search and rescue (OW-SAR). However, long imaging distances, wide fields of view, and limited sensor resolution reduce objects to a few pixels, increasing their susceptibility to environmental interference and making detection harder than for conventional small objects. To our knowledge, existing datasets focus on low-altitude scenarios and do not support automatic detection at 1.5-2.0 km. We therefore present TinyWOD, a tiny object detection dataset for high-altitude OW-SAR using lifebuoys as detection targets. It comprises 11,359 manually annotated images and 19,505 annotated instances spanning diverse environments, lighting conditions, water colors, and surface ripples. Compared with existing datasets, TinyWOD contains much smaller objects and a substantially higher proportion of tiny instances; its average equivalent side length (ESL) is 12.7 pixels, with 76.96% of in stances having ESLs below 15 pixels, reflecting actual high-altitude water-rescue imaging conditions. We define APtiny, a tiny object detection metric, and benchmark mainstream and state-of-the-art networks using this and conventional metrics. All evaluated networks effectively detect tiny waterborne objects, demonstrating TinyWOD's reusability and utility. We expect this dataset to support research on tiny object detection in high-altitude OW-SAR. 

---
# NexSight: Verifiable Tasks for Fine-Grained Agentic Long-Video Understanding 

**Authors**: Tianci Xie, Li Chen, Pinxue Guo, Rui Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.22790)  

**Abstract**: Fine-grained questions about long videos ask how many times an action occurs, in what order events unfold, or exactly when a state changes. Answering them can require finding brief moments scattered across hours of footage. Agentic models that decide when and where to look promise to find these moments, yet this ability is hard both to measure and to learn. Sparse supporting evidence may establish that an action occurred without establishing its total count; verifying timeline-wide claims requires checking for missed occurrences and double counting. Answer-only supervision, in turn, does not explicitly supervise the search for evidence. We present NexSight, which builds verifiable tasks to both evaluate and train agentic long-video understanding. Its agentic pipeline, SightForge, accepts a task only if a verifier blind to the draft answer can reconstruct it from evidence densely reacquired from the raw video and counterexample checks fail to refute it. This yields SightLine, 310 evidence-traceable tasks from 108 long videos, evaluated through SightLoop, a shared multiscale harness supporting temporal exploration, local inspection, evidence integration, and revisiting while recording returned observations. On disjoint videos, solver trajectories on verified tasks become step-level supervision for observation decisions. The strongest model reaches only 54.52% on SightLine, yet 79.35% when given the annotated evidence. With under 6K examples, Sight-35B nearly doubles its base model's interactive accuracy (21.29% to 42.26%), outperforms the 397B model of its family, begins using explicit spatial inspection after post-training, which its base model never uses, and gains over 8 points on two external long-video benchmarks. 

---
# PANO: Parallel Latent Space Neural Operator for Solving PDEs 

**Authors**: Zijiang Yang, Xiaomeng Wu, Chen Kuang, Dongmei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22782)  

**Abstract**: Aggregating observations into a limited number of latent tokens and learning physical interactions in the latent space has emerged as an effective approach for Transformer-based neural operators to solve Partial Differential Equation (PDE) problems involving large-scale observations. However, existing methods typically compress observations using a single aggregation pattern, making physical information not retained by that pattern invisible to the latent space and thereby constraining physical interactions within it. To address this limitation, we propose PArallel Latent Space Neural Operator (PANO), a novel neural operator for solving PDEs that formulates physical interactions among observations as an additive combination of interaction results from parallel latent spaces with shared parameters. Specifically, PANO transforms observations into multiple distinct compressed representations through independent mappings, learns physical interactions in parallel based on these representations, and finally fuses the decoded results to predict the target physical field. Within PANO, we further propose Coupled Assignment Projection (CAP) to construct bidirectionally consistent encoding and decoding mappings, and Pairwise Latent Space Attention Layer (PaLA) to jointly update latent representations of each latent space through adaptive cross-space interactions. Extensive experiments on nine benchmarks spanning diverse physical systems, geometries, and observation scales demonstrate that PANO achieves state-of-the-art performance. Notably, PANO reduces equivalent plastic strain prediction errors on Unilateral Stamping by 28.3% (or more) and lift-coefficient prediction errors on AirCraft by 21.4% (or more) compared with existing methods. Code is available at this https URL. 

---
# ForceBody: Force-Paired Parametric Body Motion with Torque Uncertainty 

**Authors**: Joonwoo Kwon, Yufei Zhang, Xiaoming Liu, Zijun Cui  

**Link**: [PDF](https://arxiv.org/pdf/2611.22769)  

**Abstract**: Forces and torques drive human motion, central to biomechanics, robotics, and physics-aware animation, yet they are often absent from learning-based motion pipelines for body reconstruction and generation. Some motion datasets provide force annotations, but the paired motion sequences are incompatible with the parametric body models used in modern learning pipelines. In addition, existing datasets typically obtain joint torques through inverse dynamics, which is inherently sensitive to the quality of the input kinematics; however, they do not quantify the reliability of these torque estimates. We close this gap with ForceBody, the first dataset to bring ground truth force supervision into a parametric body representation, shipped with per-sample torque uncertainty. ForceBody pairs the SKEL body model with ground reaction forces and inverse-dynamics joint torques across 10,386 motion trials (9.7M frames, 26.9 hours) from 140 subjects. Each torque label is paired with a per-frame, per-joint uncertainty annotation obtained by Monte Carlo sampling through the inverse dynamics pipeline. We benchmark six architectures (MLP, Conv1D, LSTM, GRU, Mamba2, Transformer) on ground reaction force and joint torque prediction from SKEL motion. As one example use of the released uncertainty, finetuning a Transformer with per-sample uncertainty weighting reduces the MAE of joint torque estimation by 5%. Code and data are at this https URL. 

---
# StitchRAG: Constructing Compact and Evidence-Preserving Contexts for RAG with Diffusion-Based Stitching 

**Authors**: Minsang Kim, Junpyo Lim, Heeje Cho, Seungjun Baek  

**Link**: [PDF](https://arxiv.org/pdf/2611.22766)  

**Abstract**: Retrieval-augmented generation (RAG) improves factuality by grounding large language models (LLMs) in external evidence. However, long retrieved contexts incur computational overhead and can even hurt performance due to noisy or irrelevant content. We propose \textsc{StitchRAG}, a framework for constructing compact context in RAG. \textsc{StitchRAG} builds a faithful, information-dense context while dynamically compressing the retrieved passages. We adapt the size reduction ratio to each passage by formulating a utility maximization trading-off the retained information and computation costs. We then extract answer candidates and query-related keywords from passages as anchors and collect passage sentences containing anchors under the derived size constraint. Finally, we connect or stitch the collected sentences together using text diffusion to form a coherent and natural context. Experiments on five open-domain QA benchmarks show that \textsc{StitchRAG} consistently outperforms diverse multi-step and compression baselines. Extensive ablations show the effectiveness of dynamic size allocation, anchor preservation, and diffusion-based stitching. 

---
# Network-Exposed Configuration Attacks on LLM-Controlled Social Robots 

**Authors**: Stanley Akingbola, Ugbede Ibrahim, Yan Lin Aung, Quang Dan Le, Hong Qing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22761)  

**Abstract**: Large language models (LLMs) in embodied robots create security dependencies beyond inference. We investigate a network-exposed configuration attack on the Pollen Robotics Reachy Mini Lite, allowing an unauthenticated attacker to replace personality settings and the cloud LLM's system prompt through a management interface, without adversarial conversational input. Eleven prompt-injection attempts across five categories were reported as resisted, whereas both configuration-layer attack paths produced observable behavioural changes on physical hardware. We demonstrate local and network-adjacent access, mapping the weakness to CWE-306 (Missing Authentication for a Critical Function) and CWE-1327 (Binding to an Unrestricted IP Address), with a reported CVSS v3.1 score of 6.5 (Medium). A configuration endpoint also accepts cross-origin requests without a CORS preflight (CWE-352; 6.5 Medium), while the persistent-memory sink stores content without enforcing its stated restrictions. A source review of 59 public repositories identifies missing authentication in 10 of 11 source-verifiable genuine applications, or 11 of 11 including a framework-default case. These findings indicate recurrence within the sampled repositories, not prevalence in commercial products. We characterise a deployment gap between developer-oriented network assumptions and consumer-facing embodied AI security: inference-layer refusal cannot prevent unauthorised changes to model instructions. Potential consequences include surveillance through onboard cameras and microphones and social engineering of nearby users. Proof-of-concept code, evidence and the survey are available at this https URL. 

---
# Stress Testing a Multimodal Large Language Model Grading Assistant on Kinematics Graph Reading 

**Authors**: Ons Zammel, Carlos Soares, Flávia Carvalhido, Gerd Kortemeyer  

**Link**: [PDF](https://arxiv.org/pdf/2611.22726)  

**Abstract**: Multimodal large language models (MLLMs) can recognize graph features yet award credit without sufficient graphical support. This exploratory study examines selected grading outputs for synthetic responses to four kinematics exercises, generated through visual distortion and progressive transformation toward reference graphs. Different model and prompt configurations were used. Some judgments correctly distinguished instantaneous zero velocity from rest over a finite interval and recognized positive curvature. Yet a severely distorted velocity--time graph received full credit despite disconnected and branching traces and an unclear zero-velocity reference. Other submissions graded under the same configuration lost substantial credit. Numerical cutoffs intended to distinguish meaningful features from drawing irregularities were supplied without a coordinate convention for unnumbered graphs, leaving the basis of exact partial-credit awards unresolved. Stress testing can expose unsupported awards before classroom use. For the configurations examined, we do not recommend accepting scores as final grades without instructor review, including review of full-credit responses. 

---
# Know When to Zoom: How Small Is Too Small for Vision-Language Models? 

**Authors**: Siddharth Vohra, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22709)  

**Abstract**: Online vision-language models often miss small details in large images, because the service shrinks the image, or rejects it, before the model ever sees it. A car 20 pixels wide in an 8000 by 6000 drone photo reaches Claude about 6 pixels wide, and OpenAI's default rejects the photo outright. We measure how much detail survives for seven models from four vendors, using an eye exam of printed codes and 601 questions about small objects in real photos. The smallest text a model reads half the time grows from 6 pixels or less on a small image to a predicted 10 to 13 pixels on a 7680 by 4320 photo at default settings. Zooming wins much of the detail back. Splitting each photo into up to nine tiles raises accuracy by 11.1 to 14.5 percentage points for Claude and Gemini, and by avoiding rejections it lifts GPT-6.1 Sol from 50.7% to 83.7%. A two-step zoom, in which the model first names the region it needs, matches or beats tiles on Gemini and comes within 2.5 points of them on Claude, with about 40% of their image tokens. When a detail is lost, Claude and Gemini both say they cannot tell about half the time, but they fail differently in the rest: Gemini gives a confident wrong answer in 43% to 48% of these cases, Claude in only 5% to 9%. A check based on detail size, run before the call, misses most of the gain from tiles on Claude and Gemini, but Claude's own reply is a reliable trigger: tiling only after an unsure reply comes within 1.4 points of always tiling with 62% to 70% of the image tokens. We turn these findings into five rules for when an agent should zoom, and release the eye exam, every model answer and code at this https URL. 

---
# Cross-cultural Adaptation of Recipes for Low-Resource Languages: A Benchmark and Multilingual Culinary Lexicon for Hausa, Yorùbá, and Igbo 

**Authors**: Nuhu Ibrahim, Maryam Abdulazeez, Fatsuma Jauro, Salisu Aliyu, Robert Stevens, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2611.22708)  

**Abstract**: Recent multilingual large language models (LLMs) have improved machine translation, yet culturally adaptive translation for low-resource languages (LRLs) remains underexplored. This is particularly challenging for recipes, where ingredients, cooking tools, measurements, and procedures may require adaptation to the target cultural context. In this work, we introduce ReciLoReL, a benchmark of 600 expert-translated and culturally adapted recipes across Yorùbá, Hausa, and Igbo (three of the most prominent languages in Nigeria), and ReciLex, a multilingual culinary lexicon containing approximately 15,000 translated entities across ten recipe entity categories, including ingredients, tools, chef actions, state of tools, and durations. We evaluate state-of-the-art multilingual LLMs for English-to-LRL recipe translation under zero-shot and few-shot settings and investigate whether augmenting LLMs with ReciLex improves culturally adaptive recipe translation without requiring large target-language recipe corpora for training or fine-tuning. Beyond conventional reference-based translation evaluation using BLEU, chrF, and TER, we introduce five criteria for evaluating cultural adaptation, adapted from established cross-cultural adaptation methods. Across the selected LLMs and languages, ReciLex improves average BLEU from 13.57 to 14.51 and chrF from 37.89 to 38.74, while reducing TER from 77.14 to 75.12. Human evaluation further shows improvements ranging from 0.11 to 0.36 points across the five cultural adaptation evaluation criteria on a five-point scale. Overall, our work provides new resources, evaluation criteria, and empirical evaluation for studying culturally adaptive recipe translation in low-resource settings. 

---
# Making the World React Realistically: A Closed-Loop Interactive World Model for Autonomous Driving 

**Authors**: Jia Hu, Yang Chang, Panqu Wang, Zhijun Fu, Chang Liu, Changjian Yu, Jintao Lai  

**Link**: [PDF](https://arxiv.org/pdf/2611.22691)  

**Abstract**: World models (WMs) generate driving scenarios beyond recorded logs, enabling the training and evaluation of autonomous driving systems in unseen interactions. However, existing WMs struggle to ensure both behavioral realism and visual fidelity, since they overlook the relationship between vehicle interactions and the ego visual observations. We propose a closed-loop interactive WM that couples behavior modeling and visual rendering. First, imitation-learned background-vehicle (BV) policies are fine-tuned with multi-agent reinforcement learning in the closed loop against an end-to-end ego planner that reacts to the rendered images, so that their interaction behaviors match those in real logs. Second, the viewpoints exposed by these interactions guide a per-Gaussian diagnosis of each 3D Gaussian vehicle asset, which attenuates view-dependent colors on under-observed Gaussians and fills never-observed surfaces with generated Gaussians, while preserving the observed appearance. Third, signal lights are rendered according to the driving intentions of the vehicles, by an editor that learns the light states unobserved for a vehicle through counterfactual fine-tuning with a light-state classifier. On long-tail closed-loop scenarios, our method reduces the divergence from real-world interaction behaviors by 85% and improves novel-view vehicle PSNR by 2.0dB over the state-of-the-art WM. The significance of BV light signaling in WMs is shown, as the rendered signal lights conveying BV driving intentions indirectly impact ego driving behaviors in a quarter of the scenarios. 

---
# CULypso: A Clustered Federated Unlearning paradigm for Vehicular Ad-Hoc Networks 

**Authors**: Eleftheria Chini, George Margaritis, Dimitrios Katsaros  

**Link**: [PDF](https://arxiv.org/pdf/2611.22682)  

**Abstract**: Unlearning has emerged as an essential security and privacy-preserving solution in Machine Learning, enabling networks to support their clients' rights for data erasure, as mandated by data protection regulations. Introducing such a non-trivial task to an edge, highly mobile and time-sensitive vehicular network is an even greater, largely unexplored challenge. This article argues that a hierarchical Federated Unlearning approach could be aptly integrated into vehicular ad-hoc networks (VANETs), leveraging topology and movement to reduce communication, resource and time costs, while effectively forgetting a client's influence on the global model updates. In order to achieve this, we propose CULypso, an approximate, surgical, hierarchical unlearning algorithm tailored for clustered VANETs. Inspired by the nymph in Greek mythology, Calypso, whose name translates to 'she who conceals the knowledge', CULypso, with the capital letters referencing \textbf{C}lustered \textbf{U}n\textbf{L}earning, aims to achieve exactly what its name suggests: conceal, erase or forget in our case, knowledge it contains. Extensive experiments uphold the potency of our paradigm with it performing approximately 65\% better than FedEraser in terms of time, server storage load, and unlearning fidelity while utterly outperforming baseline full retraining. 

---
# FVScore: Specification-Driven Neuro-Symbolic Video Evaluation 

**Authors**: Do Xuan Long, Long T. Le, Yiwen Song, Min-Yen Kan, Tomas Pfister, Yale Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.22673)  

**Abstract**: We introduce Faithfulness Video Score (FVSCORE), a neuro-symbolic framework for specification-driven video generation evaluation. Existing evaluators rely on holistic MLLM judgments or vision-language metrics, yet aligning their scores with human judgments and tracing them to specific temporal violations remain open challenges. FVSCORE addresses these limitations by compiling each video into explicit propositions capturing prompt requirements and implicit propositions capturing unstated physical, temporal, and cross-modal constraints. It grounds each proposition using specialized perceptual tools and evaluates the resulting confidence traces with soft temporal operators inspired by Linear Temporal Logic. A Significance-Aware Calibration step then aligns verification with human perception by retaining only perceptually significant violations before producing a timestamp- and evidence-grounded report. We conduct extensive experiments across public and complex multi-event benchmarks with the strongest video models to date, demonstrating that FVSCORE outperforms holistic and agent-based evaluators in human correlation while providing highly accurate and interpretable feedback. 

---
# AlignedHPO: Efficient Surrogate-Guided Loss Hyperparameter Optimization 

**Authors**: Ivan Karpukhin, Andrey Savchenko  

**Link**: [PDF](https://arxiv.org/pdf/2611.22658)  

**Abstract**: Modern deep learning losses involve multiple hyperparameters, such as weights, scales, and margins, typically tuned manually or through expensive black-box optimization despite their differentiability. We propose AlignedHPO, an efficient framework for tuning differentiable loss hyperparameters using gradient alignment and surrogate validation objectives. Our method employs Jacobian-vector products to reduce hypergradient computation time and memory, while differentiable metric surrogates guide optimization toward task-specific evaluation criteria. We address gradient magnitude ambiguity and stochastic instability through normalized alignment and gradient smoothing. Experiments with generalized Focal Loss and ArcFace demonstrate effective adaptation across long-tailed classification and open-set face recognition. AlignedHPO achieves competitive or improved performance compared to implicit differentiation, with nearly 12x faster hypergradient computation. These results highlight the potential of efficient surrogate-guided loss hyperparameter optimization. 

---
# Hierarchical Control of Antiphase Synchronization in Biomorphic Spiking Neural Architectures for Locomotor Control 

**Authors**: Artem Guba, Nikolay Kovalev, Alexey Zharinov, Vacheslav Demin, Viktor B. Kazantsev  

**Link**: [PDF](https://arxiv.org/pdf/2611.22639)  

**Abstract**: Central pattern generators (CPGs) provide a biologically inspired foundation for adaptive locomotor control, yet systematically characterizing their dynamic regimes in spiking neural networks remains challenging. This study introduces a framework for identifying and stabilizing antiphase synchronization in a biomorphic spiking CPG model embedded in a closed neuromuscular loop. A composite synchronization metric is developed, integrating phase coherence, amplitude balance, and temporal stability. Its performance is evaluated across systematically varied stimulation waveforms and synaptic configurations. Statistical analysis reveals that stimulation type dominates regime determination, with tonic drive yielding the most robust antiphase synchronization. Periodic inputs induce nonlinear saturation and context-dependent network routing. Although synaptic weights alone show limited discriminative power, they significantly modulate robustness within a given stimulation context, with reciprocal Ia interneuron connections emerging as key stabilizers. The proposed composite metric achieves robust antiphase detection, outperforming traditional phase-only measures while preserving biophysical interpretability. These findings establish a statistically validated, hierarchical modeling approach in which external drive defines the operating regime and targeted synaptic configurations fine-tune dynamic stability, providing actionable guidelines for closed-loop neuromechanical controllers and adaptive neuroprosthetic interfaces. 

---
# Adaptive LLM-Driven Honeypots vs. LLM-Driven Attackers: Does Narrative Coherence Improve Deception? 

**Authors**: Pedro Beltrán-López, Raul Gonzalez Jimenez, Dario Maddaloni, Manuel Gil Pérez, Pantaleone Nespoli  

**Link**: [PDF](https://arxiv.org/pdf/2611.22589)  

**Abstract**: Large Language Models (LLMs) enable honeypots to generate responses and deceptive artefacts on demand, yet existing approaches evaluate generated content one reply at a time and leave open whether artefacts that jointly describe a coherent event retain an attacker better than unrelated ones. This question becomes pressing as autonomous LLM-driven offensive agents emerge as realistic adversaries. This paper proposes narrative deception, a reactive strategy in which the defender maps each observed attacker command to a MITRE ATT&CK technique, selects a compatible scenario from a curated catalogue based on MITRE Engage, and deploys mutually referential artefacts (files, processes, services, and logs) on an emulated Ubuntu host while the intrusion is under way. An LLM-powered defender and an autonomous LLM attacker interact through a real SSH session, and the attacker's termination policy distinguishes abandonment due to suspected deception from abandonment due to loss of interest. A campaign of 1,150 sessions compares four conditions of increasing sophistication (Static, Same-TTP, Multi-TTP, and Chain) across four open-weight attacker models, four defender models, and three attacker skill levels. Engagement increases significantly at every step. With the same artefact budget per reaction, coherent scenarios raise the median session length of advanced attackers from 40 to 165 commands and reduce their abandonment from 86% to 52%. The effect grows with attacker skill, holds across all sixteen model pairs, and comes at comparable inference cost. Nearly all sessions ended through loss of interest rather than detection, which suggests that deception should sustain meaningful findings, not only emulator fidelity. 

---
# Evolution Under Fire: Evolving Agent Safety Harnesses with Adaptive Injection Stressors 

**Authors**: Yujun Wang, Yilun Liu, Jinhe Bi, Krishna Kanth Nakka, Xuebing Zhou, Hinrich Sch{ü}tze, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.22583)  

**Abstract**: Tool-using language-model agents must act on information from untrusted emails, webpages, and documents. This creates a central security challenge: malicious instructions and legitimate task directives can arrive through the same interaction channel. Existing defenses are largely hand-designed and static, making further improvements dependent on repeated manual redesign. We introduce Evolution Under Fire, a staged framework that couples safety-harness evolution with adaptive stressor evolution. Utility-constrained, Pareto-scheduled tree search evolves complete executable safety harnesses from execution evidence while keeping the task model, benchmark, tools, and evaluator fixed. Adaptive stressor evolution then discovers utility-preserving attacks that guide a subsequent stage of harness evolution. On AgentDyn with GPT-4o-mini, an evolved harness reduces attack success from 50.4% to 3.0%, improves clean and attacked utility over the initial harness, and offers a stronger security-utility trade-off than reproduced defenses. On Agent-SafetyBench with Qwen2.5-7B-Instruct, harness evolution reduces attack success from 82.6% to 62.0% while retaining helpfulness and runtime validity. Evolved harnesses also transfer without further evolution to Qwen3.5-35B-A3B, reducing attack success from 32.1% to 0.9% while improving attacked-task utility. 

---
# Do LLMs Understand Network Traces? A Comparative Study of Reasoning over Raw and Filtered Traffic Data 

**Authors**: Irene Gosálvez-White, Roberto Magán-Carrión, Carlota de la Vega-Soriano, Pedro García-Teodoro  

**Link**: [PDF](https://arxiv.org/pdf/2611.22575)  

**Abstract**: The effective delivery and monitoring of ICT services rely heavily on network traffic analysis. However, interpreting complex packet captures remains a challenging task that requires substantial technical expertise. Large language models (LLMs) offer promising opportunities to automate network traffic in- vestigation, yet their reliability in packet-level field extraction and flow-level sequence reconstruction remains an open research question. This study proposes a reproducible, question-answering- based evaluation methodology to assess the capabilities of LLMs in network trace analysis. Using real-world benign traffic from the CICIDS2017 dataset, we evaluate multiple models against a human-generated ground truth, comparing their performance on raw textual traffic dumps and protocol-filtered representations. Model responses are assessed through an automated LLM- as-a-judge pipeline, validated against human evaluations. The results demonstrate that protocol filtering significantly reduces prompt overhead while improving answer accuracy by reducing incorrect responses. These findings highlight the importance of protocol-aware traffic filtering as an effective optimization for improving the efficiency and reliability of LLM-assisted network management. 

---
# JointGV: Structure-Aware Correction of Visual World Models with Parallel Graph Dynamics for Long-Horizon Modeling 

**Authors**: Tian Lyu, Yichen Jiang, Ziyuan Cheng, Yunda Yan, Xiangyu Pan, Junjie Jiang, Jialin Yu, Boyu Ning, Pietro Liò, Fan Mo  

**Link**: [PDF](https://arxiv.org/pdf/2611.22543)  

**Abstract**: Visual world models predict future visual states conditioned on actions, but recursive errors in entity poses and spatial relations limit their long-horizon prediction for robot planning and control. We ask whether a separately predicted relational state can correct this drift without restructuring the visual model. This is challenging because graph nodes and visual states share neither elements nor a feature space. We propose Joint Graph--Visual Dynamics Correction (JointGV), a plug-in based on a cross-granularity dynamics hypothesis: despite their different representations, graph-node and visual-state changes describe the same physical transition. JointGV rolls out a task-conditioned scene graph alongside the visual world model, couples their predicted changes, and combines the resulting graph-motion context with graph-predicted node positions to compute a bounded residual. Graph-learning and correction gradients do not propagate into the original visual dynamics. We attach JointGV to DINO-WM (spatial tokens) and DreamerV3 (a global latent state), evaluating both on DMC Cheetah Run and Walker Walk across three seeds and extending the evaluation to contact-rich manipulation with DMC Finger Turn Hard. Across 32-step rollouts, JointGV improves structural and visual-prediction accuracy, reducing joint-position error by 11--28\%. For spatial tokens, learned graph--visual change coupling gives lower error than broadcasting the same mean graph-motion context to every token, particularly when non-adjacent parts appear close together in the image. On real-robot inverted-pendulum sequences, JointGV reduces DINO-WM's 32-step pole-angle error from $12.56^\circ$ to $6.69^\circ$ and improves DreamerV3 and LeWorldModel as well. In simulated inverted-pendulum control, an actor trained with JointGV-corrected rollouts completes 46 of 50 episodes, compared with none using the original rollouts. 

---
# From Perception to Simulation: Unveiling and Empowering Mental Spatial Dynamics in MLLMs 

**Authors**: Jiachun Li, Zhuoran Jin, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.22541)  

**Abstract**: Spatial reasoning is a fundamental component of world understanding for Multimodal Large Language Models (MLLMs). While existing research has advanced the perception of spatial relations in static scenes, the domain of dynamic spatial simulation remains relatively unexplored. To bridge this gap, we introduce MentalGrid, a benchmark focused on spatial reasoning tasks that require models to simulate evolving spatial states without intermediate visual feedback. Extensive evaluation on our benchmark reveals a significant deficiency in the spatial simulation capabilities of existing models. Specifically, we find that performance is constrained by perceptual limitations in simple simulation scenarios and hindered by inadequate multi-step reasoning as complexity increases. To address these challenges, we propose a three-stage curriculum learning framework utilizing 16k specialized spatial reasoning samples. This approach yields SpatialWalker, which achieves a 33.8% average improvement over its base model, surpassing proprietary models like Gemini-3-Flash in spatial simulation. Furthermore, SpatialWalker demonstrates robust generalization, maintaining a 4.9% average gain on out-of-domain benchmarks, demonstrating the pivotal role of spatial simulation as a fundamental building block for broader spatial intelligence. 

---
# JET: Just-Enough Exploration Tuning for Efficient Tool-Calling Agents in Repository-Level Question Answering 

**Authors**: Yoonho Choi, Taehyeon Han, Seonghyeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.22539)  

**Abstract**: In repository-level question answering (QA), an agent explores a code repository with tools and should answer a question once enough information is gathered. Nevertheless, we find that LLM agents often continue to invoke tools even after the gathered information is already sufficient to answer. We call this phenomenon over-exploration. Moreover, even though existing methods improve task performance through supervised fine-tuning (SFT) on expert trajectories, SFT also propagates the over-exploration in the trajectories to the agent. To mitigate over-exploration, we propose Just-Enough Exploration Tuning (JET), a fine-tuning method that trains the agent to answer once the gathered information is sufficient. JET truncates each training trajectory at the step where the gathered information becomes sufficient, inserts a rationale that explains why the agent can answer at that step, and fine-tunes the agent with a higher loss weight on the rationale. Experiments on two repository-level QA benchmarks with two open-weight 8B models demonstrate that JET reduces tool calls by 11 to 33% on self-generated trajectories while largely preserving the score. On expert trajectories, JET makes 18 to 28% fewer tool calls than SFT while retaining most of the score gain of SFT. In both cases, JET also lowers the inference cost. 

---
# Testing World Models on Actions They Have Never Seen: Held-Out Plans, Forecast Accuracy and Price Rules in Retail 

**Authors**: Karthik Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2611.22537)  

**Abstract**: A world model is trusted to answer "what if": how its world responds to an action it has never seen. We give a test of that trust for any action-conditioned world model: actions held out of training by design and scored against a known answer; whether forecast accuracy predicts accuracy on those actions; and the same structure given to simpler models. We run it in retail, where the action is a price and promotion plan, across 2,853 fitted models. First, forecast accuracy is a fair guide between kinds of model (rank correlation with error on new plans 0.32 to 0.66) and a poor guide between versions of one world model: where merchants plan into high-demand weeks the link reverses, down to -0.69. Second, training the world model to learn the effect of actions (JEPA pretraining, a latent consistency check, adaptation at the plan date) leaves its error on new plans unchanged. Third, in generated worlds the recorded regular price and three price rules improve it. Given the same rules, a log-linear model is best when its curve can match the truth; in a world whose curve lies outside the rules, the world model with the rules is ahead of the log-linear model with the rules in 36 of 40 worlds, and does as well without its recurrent cell. On real data (Breakfast at the Frat, The Complete Journey and M5) the rules lower the world model's error on pure price cuts deeper than any it saw (0.63 to 0.56 on the Frat; 0.45 to 0.44, within chance, on The Complete Journey), but not on lasting price moves. Fourth, a small share of random test plans helps every model a little. In a generated dosing world, a stress test in a second domain, the first finding repeats, and the built-up drug level lowers the world model's error (2.86 to 1.91) though it still predicts only 0.12 to 0.32 of the true gain. The test carries to any domain where a world model is asked about actions outside its history. 

---
# DGF-Asym: Adaptive Peer Weighting for Collective Estimation under Noisy and Faulty Sensing 

**Authors**: Mohammad Nomaan Husain, Tanja Katharina Kaiser  

**Link**: [PDF](https://arxiv.org/pdf/2611.22532)  

**Abstract**: Collective estimation under unreliable sensing requires robots to adaptively balance personal observations and social information from peers. We propose DGF-Asym, a decentralized collective estimation method extending the DeGroot-Friedkin model with asymmetric peer weighting based on retrospective predictive performance. The method supports consensus formation without prior sensor-noise statistics or explicit fault models while keeping inter-robot communication asynchronous. We evaluate DGF-Asym in simulated collective navigation under noisy observations and time-varying sensor faults. Compared with three baselines, it produces more stable estimates and more reliably sustains the decision criterion, at the cost of longer decision times. However, overall performance depends on fault ratio, group size, and influence-threshold selection. Our results demonstrate the potential of adaptive peer weighting for stable model-free collective estimation under unreliable sensing. 

---
# Right Labels, Wrong Reasons: Signed Feature Attribution for Multi-Label Explainability 

**Authors**: Kenton White  

**Link**: [PDF](https://arxiv.org/pdf/2611.22515)  

**Abstract**: When a multi-label classifier assigns the right labels, it may do so for the wrong reasons. Diffuse feature combinations may vary across classifier families and lack a dominant feature. This paper proposes FLEX (Fisher LDA Explainer), a global surrogate that fits a single direct multi-label Fisher LDA over the full binary label matrix, producing signed feature attribution lists for each label in a single coherent coordinate system. Across the evaluated benchmarks and classifier families, FLEX reproduces the black box's predictions with consistently high fidelity. Pooling covariance across labels outperforms a per-label alternative on most datasets, with the advantage growing largest where labels are numerous and positive examples are scarce. FLEX's fidelity-complexity Pareto curve is monotonically increasing, giving practitioners a predictable path to higher-fidelity explanations as more features are used, and its in-sample fidelity matches or leads Binary Relevance alternatives once a modest feature budget is allowed. The magnitude of each discriminant weight vector bounds its label's separability, flagging labels that may be predicted correctly for the wrong reasons. The evaluated classifiers largely agree on which predictions to make, but reach those decisions through different feature subsets when no single feature dominates. 

---
# Motion-Aware Reward Modeling for Instruction-Guided Video Editing 

**Authors**: Li Li, Xiangbo Gao, Bo Ni, Samyadeep Basu, Viet Dac Lai, Jing Shi, Seunghyun Yoon, Gang Wu, Yu Wang, Thien Huu Nguyen, Tyler Derr, Franck Dernoncourt, Zhengzhong Tu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.22495)  

**Abstract**: Reward models for instruction-guided video editing can favor temporal consistency at the expense of motion. We study this failure, termed \emph{motion collapse}, using controlled motion-suppression variants of real edits. On VE-Bench, the evaluated composite reward prefers a motion-suppressed candidate in $97\%$ of cases. Learned video rewards and a VLM judge also struggle with mild suppression, while latent optimization and reward-guided fine-tuning exploit this weakness to increase reward as motion declines. A linear probe of our base reward backbone performs near chance on mild suppression, motivating an explicit inter-frame dynamics input. We propose a motion-aware reward model that combines an RGB frame-difference stream with preference training against motion-suppressed and fabricated-motion negatives. A distillation anchor preserves alignment with human quality judgments. On the VE-Bench motion ladder, our model reduces the rate of selecting motion-suppressed candidates to $0\%$ while achieving a Spearman correlation of $0.57$ with human quality ratings. It also detects unseen suppression variants and preserves motion during reward-guided fine-tuning on two editing pipelines. These results support explicitly modeling motion in rewards for video editing. 

---
# Manufacturing Differential Sparsity: Planner-Hardware Co-Design for Efficient World Model Planning 

**Authors**: Wenbo Zhu, Paul Yi-Chia Chen, Yuanhan Li, Zishen Wan, Christopher Torng  

**Link**: [PDF](https://arxiv.org/pdf/2611.22483)  

**Abstract**: JEPA-style latent world models let embodied agents plan toward new goals with model predictive control, but they are still too slow for real-life embodied control: sampling-based MPC repeats predictor inference tens of thousands of times per plan, and even on high-end GPUs one planning step can take tens of seconds. We observe substantial latent locality in this repeated-inference workload: the predictor maps close actions to close latents, and inside a quantized predictor this closeness takes a form that hardware can exploit, differential bit-level sparsity. Unlike an input stream, whose temporal similarity is given, a planner itself decides which candidates it evaluates together, so the planner can choose its own references and, with reference forest sampling, manufacture more sparsity: it couples the sampled action sequences of each group to a shared root, with competitive planning quality. We build the accelerator on the insights of the rich line of bit-level sparsity accelerators and design it and the planner tightly coupled as a single co-design: the planner increases sparsity and exposes structure for hardware scheduling, and the bit-exact hardware effectively uses the sparsity that the planner creates. On three JEPA world models, our co-design is 1.28-1.80x faster than a dense accelerator and uses 38-52% less energy, and it is 1.22-1.58x faster than prior bit-sparse design and uses 16-36% less energy, with the same area. To our knowledge, our planner is the first planner that manufactures differential bit-level sparsity across its candidates and exposes its structure for hardware scheduling, and our co-design is the first algorithm-hardware co-design for JEPA-style world-model planning. 

---
# LeViT-UNet++: Towards Balanced Fine-Grained Details and Context for Efficient Medical Image Segmentation 

**Authors**: Guoping Xu, Jinling Zhu, Liuyue Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.22482)  

**Abstract**: Efficient medical image segmentation is essential for clinical applications requiring fast and accurate delineation, such as surgical navigation, radiation therapy, and real-time image-guided intervention. Despite substantial advances in convolutional neural networks and Transformers, attaining high accuracy under limited computational resources remains difficult, because reducing model complexity often sacrifices fine-grained structural details and global context modeling, yielding an unfavorable accuracy-efficiency trade-off. To address this, we propose LeViT-UNet++, an efficient encoder-decoder architecture that better balances local detail preservation and global context modeling. Specifically, we introduce a Haar Wavelet Multi-Scale Downsampling (HWMD) module that decomposes feature representations into low- and high-frequency components to extract multi-scale structural information, thereby mitigating detail loss during downsampling. We further develop a depthwise convolution-based decoder that integrates multi-scale global representations from Transformer blocks with local spatial features, enabling progressive feature refinement while reducing computational overhead. Evaluated on four benchmarks (Synapse, ACDC, LA2018, and MSD Heart) under both 2D and 3D settings, LeViT-UNet++ achieves competitive accuracy with a compact parameter footprint and fast inference. On Synapse, for instance, it attains a DSC of 76.57% with only 8.025M parameters and 241.72 FPS, confirming a favorable accuracy-efficiency balance. These results demonstrate the effectiveness of combining wavelet-based multi-scale feature extraction with lightweight local-global feature fusion for efficient medical segmentation. The source code is available at this https URL. 

---
# MAVE: Reconstructing Editable 3D Blockouts from Film Shots 

**Authors**: Yifan Chang, Jian Yang, Jiangnan Shao, Xiaofeng Wang, Zhaoxin Fan, Renjing Xu, Ming Lu, Xingang Wang, Zheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22464)  

**Abstract**: Animated 3D blockouts support previsualization and provide structural conditions for video generation, but manually reconstructing them from reference videos is costly. This limits both their use by creators and the production of paired training data. We introduce MAVEBench, a benchmark of 50 film shots and 416 keyframes for evaluating camera motion, spatial layout, and character blocking. We propose MAVE, a video-to-blockout framework combining Metric observation, Agentic modeling, and evidence-based VErification. It first recovers cameras, human motion, and geometric evidence in a shared metric coordinate system. An agent then constructs an editable environment under constraints from these observations, while programmatic checks report inconsistencies to guide refinement. The outputs are editable Blender scenes and frame-aligned blockout videos. On MAVEBench, MAVE improves overall scores by 27.5-41.1 points across three agents over their corresponding vanilla baselines. With DeepSeek-1017, it achieves a score of 73.2 at an average API cost of only USD 0.054 per shot, outperforming the strongest vanilla baseline while reducing total reconstruction time by 53%. We also construct MAVEData-3K, a dataset of 3,000 blockout-video pairs for model training, and will release it publicly. Creators can change viewpoints and character paths, then render the edited scenes to condition video generation. Code is available at this https URL. 

---
# As Student Rollouts Continue, Visual Backdoors Keep Growing 

**Authors**: Zane Fu, Yanzhao Zheng, Jinze Liu, Ziting Qin, Qianshan Wei, Jihuai Zhu, Yipeng Yu, Baohua Dong, Hangcheng Zhu, Ruohui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22446)  

**Abstract**: Conventional knowledge distillation can facilitate the propagation of backdoors from teacher models to student models. However, how this risk manifests in On-Policy Distillation (OPD) remains insufficiently studied. OPD has recently emerged as an important post-training paradigm for Vision-Language Models (VLMs). Unlike traditional offline distillation, OPD does not require teacher-generated responses. Instead, it uses the teacher's probability distributions to supervise response trajectories generated online by the student. Since the initial student model typically generates benign responses and receives no explicit supervision from malicious target responses, OPD appears to offer some inherent resistance to backdoor propagation.
In this paper, we challenge this intuition and reveal a more subtle risk of visual backdoor propagation in OPD. Through experiments across multiple model families and datasets, together with theoretical analysis, we identify three main findings. First, unlike the relatively gradual propagation observed in traditional offline distillation, backdoors rarely emerge in OPD at low poisoning ratios. However, the attack success rate rapidly approaches 100% once a critical threshold is exceeded. Second, backdoor propagation does not depend on explicit malicious responses. Instead, it is driven by the continuous accumulation of anomalous probability mass assigned by the teacher. Finally, based on these findings, we propose a novel trigger design that enables visual backdoor propagation with zero explicit trigger injection. These findings further reveal an overlooked mechanism of backdoor propagation in OPD and highlight its potential security risks. 

---
# Eyes in the Update? Learning Where Rewards Land in Multimodal Reasoning 

**Authors**: Lixuan He, Tianzhe Chu, Shulin Tian, Shikang Zheng, Feng Chen, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.22438)  

**Abstract**: Multimodal post-training has moved from broad instruction tuning toward reinforcement learning for image and video reasoning, where outcome rewards can improve final-answer accuracy. Yet these rewards are assigned to complete generated trajectories, while visual evidence is used only at sparse and uneven positions inside the response. Recent work has begun to make multimodal RL more token-sensitive through grounding signals or external token-level weighting. We ask a more specific question: can the allocation of the same trajectory-level reward be inferred from the model's own multimodal computation, rather than from an auxiliary annotator, perception module, or process label? We propose Gaze-RL, a representation-driven policy optimization method that redistributes completion-level advantage through a latent reward landing field. The field is computed from internal cross-modal geometry and local predictive uncertainty, assigning larger update weight to positions that are both visually anchored and decision-bearing while remaining mean-one, so the total credit of each rollout is preserved and only its positionwise allocation changes. Gaze-RL requires no process supervision, auxiliary perception module, or architectural modification, and improves both Qwen2.5-VL and InternVL3.5 across a broad range of image and video benchmarks. 

---
# Encoding, Decodability, and Use: A Five-Layer Framework for Benchmark Measurement Audits with Evidence from UAV Intent Recognition 

**Authors**: Zhirong Su, Tian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22434)  

**Abstract**: Benchmark accuracy is routinely read as evidence that a model understands the construct the benchmark measures. We propose and instantiate on AeroIntent a five-layer framework that decomposes that reading into observation, representation, decoding, use, and behaviour. The framework separates encoding, decodability, and use as distinct levels. In the instantiation, the observation layer is unreliable on 7 of the 20 Drone programs and floors the direct visual correlate of radial velocity at a 0.26\% scale change. The representation layer exposes two task-designed RGB features with nominal signals. RAFT flow radial bias reaches $r=+0.257$ ($p=0.030$), and bounding-box optical-flow divergence reaches $r=+0.203$ ($p=0.029$). No multi-feature readout gives a fold-mean test correlation above the cluster minimum detectable effect (MDE), and the largest positive global out-of-sample correlation is $r=+0.080$. The decoding layer sets the MDE at $|r|=0.599$ at the multi-feature Bonferroni threshold ($\alpha=0.00714$, power 0.80), and the observed $r=+0.080$ has power 0.00. Two injection experiments establish the MDE's condition from both sides: aligned injections are detected at or above the simulated power, and orthogonal injections are not detected. The use layer finds accuracy consistent with program memory (program-identity 0.937/0.597/0.210) and multi-region static cues. The behaviour layer is consistent with accuracy not requiring a radial-velocity readout (appearance 0.6552 vs bounding-box trajectory 0.4080, cluster-level paired $\Delta=+0.2471$, $p=0.0132$). 

---
# Deal With It: On Better Schedules for Listwise Reranking 

**Authors**: Benjamin Clavié, Herumb Shandilya, Hanna Lichtenberg, Aamir Shakir, Makoto P. Kato  

**Link**: [PDF](https://arxiv.org/pdf/2611.22392)  

**Abstract**: Listwise LLM rerankers cannot reliably rank hundreds of documents in a single call, and rely on schedules to orchestrate their calls. Parallel schedules split candidates into groups and advance a fixed quota from each. When a group holds more of the top passages than its quota, some are cut, and expensive repair calls become required. We show that the way candidates are split matters where such a quota can bind, and that, in expected cuts, the best split there deals them round-robin by retriever rank, like cards. Although not conceptually new, dealing has remained unanalyzed. We prove that, under a simple retriever model, no split with equal quotas cuts fewer of the true top ten on average, and that contiguous chunks cut more as the retriever improves. DealRank, the simplest schedule our analysis allows, is on average within 0.1 nDCG@10 points of the best schedules over six LLMs, in two rounds. It is consistently 1.1 to 1.4 times faster than BlitzRank with concurrent first calls up to a thousand candidates. There, it needs three rounds and is eleven times faster than the sliding window. In existing schedules such as BlitzRank, dealing reduces repair calls from 59% of queries to 14%, improving latency with no significant change in quality. Random partitioning appears equivalent at a quota of four, as predicted, but ties results to the seed and trails by 1.3 nDCG@10 points at a quota of one over a dense retriever. 

---
# RICoT: Recurrent Interleaved Chain-of-Thought for Page Stream Segmentation 

**Authors**: Prashant Khatri, Bhavana Malagavalli, Saiful Haq, Niyati Chhaya  

**Link**: [PDF](https://arxiv.org/pdf/2611.22382)  

**Abstract**: Large language models (LLMs) have renewed interest in Page Stream Segmentation (PSS), splitting a multi-page file into its constituent documents. Yet published LLM methods still frame the task as boundary classification over one or a few neighboring pages, and commercial document-splitting APIs do not disclose their methods. We reimagine PSS for LLMs as a sequential process over the whole stream and propose RICoT (Recurrent Interleaved Chain-of-Thought), a prompting framework that works through the pages as a reader would: within a sliding window it reasons about each page and commits its boundary decision before moving on, and across windows it carries forward a compact record of what it has read, keeping per-call cost bounded. We also introduce FinancePSS-205, a benchmark of 205 real-world, de-identified financial page streams, to be released publicly. On FinancePSS-205, RICoT matches or outperforms pairwise and windowed prompting with the same backbone for each of three open-weight LLMs; on a TABME test subset, RICoT with GLM-5 reaches document F1 0.847 and boundary F1 0.915, above published zero-shot results. A 2B-parameter student, trained on a larger teacher's decisions with and without the page-level reasoning, performs comparably in both modes and comes within 0.05 document F1 of the strongest prompted model on FinancePSS-205. 

---
# Leveraging Segmentation Representations for Efficient Classification: A Self-Configuring Framework Based on nnU-Net 

**Authors**: Renjun Cai, Antonio Javier Gallego, Félix Fuentes-Hurtado  

**Link**: [PDF](https://arxiv.org/pdf/2611.22380)  

**Abstract**: We propose a self-configuring framework that extends the nnU-Net workflow to both segmentation and classification in 3D medical imaging. Building on nnU-Net's self-configuring method, the framework reuses representations learned by a segmentation model for image-level classification. Its encoder features and predicted segmentation probabilities are used by a lightweight classifier. Two information-sharing strategies are investigated. In the frozen-backbone branch, features are augmented with region-of-interest information, predicted foreground volume, and segmentation confidence. In the adaptation branch, the final encoder stage is partially fine-tuned for classification, with an auxiliary segmentation loss to preserve segmentation-related information. Additional five-fold experiments showed that more complex configurations did not consistently improve classification performance. On the public validation leaderboard, our selected framework achieved an Overall score of 0.8056. In the official FLARE 2026 Task 2 final evaluation, our framework ranked first overall, achieving the highest DSC, NSD, and AUROC across all four testing tasks, including two hidden testing tasks. Local inference measurements also remained below the challenge limits of $60 \mathrm{s}$ per case and $4 \mathrm{GB}$ GPU memory. These results demonstrate the practical value of reusing segmentation representations for classification while maintaining a simple and efficient framework. 

---
# FailureGen: Scalable Robot Failure Data Generation with World Models 

**Authors**: Akul Santhosh, Chaitra Sathe, Aastha Jhunjhunwala  

**Link**: [PDF](https://arxiv.org/pdf/2611.22367)  

**Abstract**: Robot failures are difficult to collect at scale, limiting the range of failure scenarios represented in robot learning datasets. We present FailureGen, a method for generating synthetic videos with temporal event annotations by reusing action trajectories from recorded failures. The method comprises four stages: (1) fine-tuning an action-conditioned Cosmos world model on successful and failed real executions; (2) annotating source recordings to identify failure-associated donor segments and compatible successful hosts; (3) transferring donor motion into host contexts, with optional spatial offsets, to generate continuous video rollouts; and (4) annotating and filtering the generated samples according to observed events and visual quality. We study two implementations with different tasks and robot configurations, with visual failure detection serving as one downstream application. In a preliminary comparison, a detector trained on real and generated data achieves an average precision of 0.51, compared with 0.41 using real data alone, and detects 33 of 36 failure intervals, compared with 28 of 36. By reusing observed failures in additional contexts, FailureGen expands the data available for studying robot failures and developing downstream learning methods. 

---
# An AI-Driven Assistant for Human-Centric Building Digital Twin Systems: Design, Implementation, and Comparative User Evaluation 

**Authors**: Leonardo Messi, Erika Anneli Pärn, Juan Diego Castaño Molina, Oleksandr Anyshchenko, Dijar Selami, Peihang Luo, Borja García de Soto  

**Link**: [PDF](https://arxiv.org/pdf/2611.22350)  

**Abstract**: Smart buildings generate streams of operational data, yet this richness rarely translates into accessible, decision-ready information for stakeholders who inhabit and manage them. Current Building Digital Twin Systems (BDTS) often remain oriented toward technical specialists and rely on static, one-size-fits-all interfaces. This paper presents the design, implementation, and comparative user evaluation of an AI-driven conversational assistant integrated into a human-centric BDTS for Indoor Environmental Quality monitoring, combined with adaptive information presentation and auto-suggest functionality. The prototype was deployed at New York University Abu Dhabi. A controlled, two-arm, between-subjects, quasi-experimental study with wave-based, non-randomized assignment compared the Baseline interface with the Experimental interface. Results showed significant reductions in task-completion time and perceived workload, while accuracy and usability improved but did not meet the adjusted significance criterion. The study contributes an implemented and empirically evaluated architecture that separates generative interaction from deterministic environmental assessment and coordinates conversational responses with adaptive visual presentation. 

---
# Agent Trajectories as Evidence for Software Quality Assurance 

**Authors**: Jonathan Ullrich, Julien Siebert, Michael Kläs, Adam Trendowicz, Andreas Jedlitschka  

**Link**: [PDF](https://arxiv.org/pdf/2611.22323)  

**Abstract**: Autonomous software agents increasingly plan, implement, and test software with limited human oversight. Their decisions may introduce implicit assumptions that final code and passing tests do not adequately expose. We propose analyzing observable agent trajectories alongside software and agent specifications and AI generated artifacts as evidence for targeted quality assurance. The aim is to prioritize consequential decisions for scrutiny, investigate their possible contribution to software problems, and inform improvements to specifications and development practices. Industrial experience motivates the idea; available execution records and preliminary analysis methods make it plausible. However, trajectories alone do not establish that decisions are justified, outcomes acceptable, or corrective changes effective. We ask how this evidence can support review, diagnosis, and continuous quality improvement, and when simpler QA approaches suffice. Perspectives from requirements engineering, verification and validation, agent systems, causal inference, empirical software engineering, and human factors could help develop and evaluate the idea. 

---
# FIRE-Drive: Feasibility-Informed Risk Evaluation for End-to-End Diffusion Planning 

**Authors**: Qifeng Xie, Kai-Yuan Feng, Wenkang Wan, Yue Wu, Maoguo Gong, Kai Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.22297)  

**Abstract**: Diffusion-based end-to-end autonomous driving models have shown strong potential in generating multimodal planning trajectories. However, their performance still depends heavily on trajectory initialization, mode evaluation, and supervision quality. Weakly constrained noisy initialization may introduce infeasible proposals and increase the burden of denoising. Moreover, selecting the optimal mode remains challenging when candidate trajectories are geometrically similar but differ in safety, interaction risk, and progress. To address these issues, we propose FIRE-Drive, a feasibility-informed risk evaluation framework for end-to-end autonomous driving. FIRE-Drive improves diffusion-based planning through dynamic feasible anchor initialization, candidate-specific bird's-eye-view risk evidence, and compact high-quality candidate supervision with lightweight refinement. This design aligns the initial support, risk-aware scoring, and multimodal supervision of trajectory modes under a unified planning objective. It enables the diffusion decoder to focus on refining plausible trajectories while the scorer distinguishes modes using local risk evidence. Experiments on NAVSIM v1 show that FIRE-Drive achieves a Predictive Driver Model Score of 91.5 with a ResNet34 backbone. Additional closed-loop evaluation on 220 Bench2Drive routes yields a Driving Score of 71.43 and a Success Rate of 42.73%. Component ablations and a scorer comparison under matched candidate inputs support the contributions of feasible initialization, auxiliary supervision, and risk-informed selection. Our code is available at this https URL. 

---
# Perception into Imagination: Executable Memory for World Models 

**Authors**: Xianyao Li, Ruitong Tian, Rui Min, Fang Xu, Jing Du  

**Link**: [PDF](https://arxiv.org/pdf/2611.22294)  

**Abstract**: Physical agents must use what they previously saw or touched after visibility or contact is lost. Their actions change the implications of these remembered facts. We introduce executable perceptual memory (EPM), a world-model framework that advances explicit perceptual state alongside neural latent dynamics during imagination. Its evolving read-out informs reward prediction, value estimation and policy learning. We align reward inputs with the transitions that produced them and supervise the reward head on open-loop rollouts of recorded actions. On Memory Maze $13\times13$, EPM reaches 30.37 return at 1M interactions, compared with 23.37 when the same evolving read-out informs only policy and value on a common perceptual front end: a gain of 7.00 [3.80, 10.21]. The complete method also improves control on $9\times9$ and in MiniWorld. Probes of remembered, out-of-view arrivals show stronger reward anticipation in both visual implementations. Concentration, tactile recovery and real tactile frames demonstrate complementary roles of remembered facts in imagined-policy training, motion-dependent decisions and sensory-content retention. EPM connects persistent perception to the predicted consequences and learning signals of a world model for Physical AI. 

---
# When the World Is the Test Suite: Agentic Intelligence for the Physical World 

**Authors**: Kaixin Li, Meng Luo, Yanlin Li, Yuchen Tian, Xiang Liu, Xiangru Jian, Yan Yang, Zhiheng Xi, Yang Zhang, Ziyang Luo, Mingfei Cheng, Hongzhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.22291)  

**Abstract**: Coding agents built to repair software now write, test, and repair programs for robots, laboratories, and industrial controllers. Unlike in software engineering, no test suite comes with the code. Whether a program works must be established by running it on a robot, in a physics simulator, or through the policies it helps to train, and such evidence is costly to obtain and can itself be wrong. As these agents take on more physical tasks, how their programs are judged matters as much as how they are written. This paper examines this emerging area, agentic intelligence for the physical world, by treating the world as the program's test suite. We organize the discussion around three questions: how agents write programs for the physical world, how these programs are tested, and how agents learn from the results. We first trace how language models for robots and software engineering agents came together, ask which software testing practices carry over to physical tests, and describe each program by where it sits relative to the actuator and by what judges it. For writing, we cover programs that act as robot policies and programs that support policy learning by producing rewards, tasks, and training worlds. For testing, we examine what physical trials cost, what traces, learned judges, simulators, and formal checks can establish in their place, and what safety requires beyond task success. For learning, we discuss how tested programs are retained as experience, in skill libraries and in model weights. Finally, we turn to benchmarks and evaluation practice and outline open challenges. Writing programs is becoming cheap, while trustworthy evidence about what they do remains costly. Progress will therefore depend on a physical counterpart of continuous integration: verification cheap enough to run often, reliable enough to trust under search, and maintained alongside the experience it validates. 

---
# Neural Network Path Optimization for Finding Transition States on a Machine Learned Potential 

**Authors**: Kareem Hegazy, Eric C-Y Yuan, Anup Kumar, Dmitriy Morozov, Daniel Barter, Michael W Mahoney, Teresa Head-Gordon, Samuel M Blau  

**Link**: [PDF](https://arxiv.org/pdf/2611.22272)  

**Abstract**: Transition states govern chemical reaction rates and outcomes, yet locating them remains a central challenge in computational chemistry. Double-ended search methods find transition states given reactant and product structures, but generally rely on discretized path representations developed for density functional theory, where potential energy surface evaluations are expensive. Machine learned interatomic potentials (MLIPs) dramatically reduce this cost and leverage automatic differentiation to access higher-order derivatives efficiently. We introduce Popcornn, a framework to find transition states by optimizing continuous neural network paths atop an MLIP with a parallelized adaptive numerical integrator. This combination eliminates path discretization artifacts, reduces hyperparameter sensitivity, and enables non-local optimization to overcome local features using second-order potential-energy-surface information. Popcornn consistently recovers more correct transition states than standard methods such as nudged elastic band and growing string, across MLIPs on thousands of organic reactions, establishing it as a robust, automation-ready method for high-throughput transition state search. Popcornn's infrastructure is flexible in path objective, integration strategy, and optimization algorithm, making it a practical tool and platform for investigating open questions in MLIP-based path optimization. 

---
# Teaching Language Models to Narrow Boolean Patent Searches from Search Feedback 

**Authors**: Seiya Kawano, Koji Marusaki, Hirofumi Nonaka, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2611.22270)  

**Abstract**: Patent prior-art searchers issue Boolean queries line by line, broadening to find relevant documents and narrowing until candidates can be screened. We distill 38,397 Japan Patent Office search reports into executable expert strategies, re-execute them, and evaluate generated strategies by citation recall and the smallest set containing each citation. Prompted models write valid, high-recall strategies but leave citations in sets too large to screen. We adapt a generator offline from scored rollouts in two forms: strategy-level learning from a screening-aware strategy score, and line-level learning from citation-preserving narrowing credit, which upweights the query lines keeping a found citation in a smaller set. On shared rollouts, line-level credit outperforms strategy-level scores on screening measures; iterated, it cuts the median citation-containing set from 768 to under 50 and places more citations in sets under 100 documents than experts, at lower recall. Strategy-level preference learning reaches near-complete recall; the two forms are complementary. 

---
# Anticipating or Speculating? Making Futures Pay in AI Ethics 

**Authors**: John P. Nelson  

**Link**: [PDF](https://arxiv.org/pdf/2611.22268)  

**Abstract**: Philosophers have recently been trying to develop uniform decision rules for instances in which physicians' judgments diverge from the recommendations of medical artificial intelligence (AI) systems. I argue that this effort is fatally flawed because the performance and appropriate use of algorithmic recommendation systems by physicians are highly context-specific, empirical medical matters rather than epistemological ones. In such instances, general decision rules can be stipulated only to the extent that they abstract away from relevant, indeed determinative, contextual circumstances. The alleged problem of physician-AI disagreement, and others like it, play into longstanding disputes in technology ethics about the viability and utility of envisioning possible futures: is this useful "anticipation" or wasteful "speculation"? I suggest that the two may be distinguished. Anticipation explores plausible futures to help inform present actions under irreducible uncertainty. Speculation abstracts away uncertainty, developing formal solutions to stipulated circumstances whose relationship to the real present is often quite unclear. Efforts to develop crisp philosophical solutions to physician-AI disagreement, among other problems, fall into the latter category. Technology ethics should remember that fruitful futures thinking must both begin from and return to the messy, uncertain present. 

---
# OfficePrism: Benchmarking Agents on Single-Domain Operations Across Enterprise Business Functions 

**Authors**: Hanfeng Liu, Zhenyu Luo, Dongchao Zhang, Zexi Liu, Zulong Chen, Jinxin Hu, Jia Xu, Zeyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2611.22262)  

**Abstract**: Broad workplace scores can mix competence within one business function with the extra cost of coordinating across functions and systems, making failures hard to localize. OfficePrism evaluates this prerequisite through 420 single-domain enterprise tasks spanning seven business functions under shared, benchmark-defined L1--L3 task-structure levels. Each task runs in a fresh environment combining a shared synthetic organization with one function-specific domain and yped state access. Our pipeline uses grounded records, business rules, and tool schemas to jointly construct a Chinese-language ticket, a hidden executable reference witness, and an expected response, followed by automated filtering and human curation. The same task set is evaluated through full-schema and search-and-dispatch packages. Across 124 model--harness--package configurations, the per-case joint natural-language-and-database pass rate ranges from 22.6% to 76.4%, with a median of 67.9%. L1 is highest in all 124 configurations, and L2 exceeds L3 in 122. Among 50 configurations sharing the same harness and model labels across both packages, search scores higher in 30, ties in one, and scores lower in 19. 

---
# PCGRL for Complex Platformers: Synthesizing Playable and Diverse Levels via Inflated Envelopes and Forward Curricula 

**Authors**: Di Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22250)  

**Abstract**: Procedural Content Generation via Reinforcement Learning (PCGRL) has emerged as a promising paradigm for automated game development. However, existing PCGRL frameworks still suffer from severe exploration bottlenecks and scaling limitations when faced with long-sequence generation tasks governed by strict rigid-body rules and non-linear kinematic boundaries (e.g., platformer constraints). Consequently, traditional approaches frequently fail to maintain reliable level solvability on large-scale canvases. To address these challenges, we present a physics-aware, curriculum-guided PCGRL framework. First, we construct an "Inflated Volume Envelope" mechanism, which provides a physics-constrained representation that incorporates continuous kinematic feasibility into the discrete state space of the MDP. Second, to alleviate the sparse reward challenge induced by extremely long decision sequences, we introduce intermediate state-transition feedback and a multi-stage forward extension curriculum learning framework, enabling informative value estimation and stable credit assignment over long-horizon trajectories. Finally, to resolve the issue where deterministic policies after convergence generate highly monotonous map paths with a lack of diversity, we activate high-entropy stochastic inference during the synthesis phase, regularized by a post-hoc parallel rejection filtering mechanism. Experimental results demonstrate that our framework significantly enhances structural diversity while maintaining high physical playability across multi-scale environments. Comparative evaluations across multi-scale canvases demonstrate that our framework maintains superior convergence stability and dimensional scalability, reliably synthesizing solvable levels where traditional PCGRL baselines degrade. 

---
# OAK-Mind: Zero-Training Mental Health Change Inference from Passive Sensing via Knowledge-Grounded Multi-Agent LLMs 

**Authors**: Yishu Ji, Zhihan Jiang, Will Ke Wang, Yuang Fan, Anind K. Dey, Xuhai "Orson" Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22248)  

**Abstract**: Wearable and smartphone sensors capture behavioral signals associated with mental health, yet inferring mental health outcomes from these signals has traditionally relied on supervised models requiring labeled data and offering limited insight into their predictions. Large language models (LLMs) offer new opportunities to reason over behavioral evidence, but existing LLM-based approaches to passive sensing have largely continued to rely on task-specific fine-tuning or sensor-specific pretraining. In this paper, we introduce \textbf{OAK-Mind}, a zer\textbf{O}-training, multi-\textbf{A}gent, \textbf{K}nowledge-grounded LLM pipeline for inferring mental health changes from passive sensing data. The pipeline organizes sensing data into behavioral dimensions including sleep, physical activity, location mobility, and smartphone use, and orchestrates specialized agents grounded in mental health literature to interpret longitudinal behavioral changes. It synthesizes their assessments and applies questionnaire-based symptom rating to infer changes relative to an individual's baseline assessment with human-readable rationale grounded in observed behavioral patterns and domain knowledge. Across datasets and prediction tasks, while without task-specific training, OAK-Mind achieves comparable or better performance than the most predictive supervised baselines trained on additional labeled samples. It also demonstrate consistent improvement upon direct prompting across four tested LLM backbones. Ablation analyses support the benefits of combining behavioral interpretation with symptom rating, distributing analysis across domain-specific agents, and incorporating domain knowledge. Our findings highlight the potential of knowledge-grounded, multi-agent LLMs for training-free inference of mental health changes from passive sensing data. 

---
# Solver-Free and Cost-Aware Online Scheduling of AI Data Center Tasks 

**Authors**: Xin Qin, Dawei Qiu, Fei Teng  

**Link**: [PDF](https://arxiv.org/pdf/2611.22239)  

**Abstract**: AI data centers receive computing tasks continuously, whereas solver-based optimization takes time to run and electricity is priced per settlement period. This paper bridges this timing gap through a solver-free online optimization rule inspired by dual decomposition. The rule assigns each task a start period upon arrival to optimize energy cost subject to operational constraints, shifting flexible tasks toward low-price periods while keeping high-priority tasks on schedule. Simulations on the A100 GPU trace (3,456 GPUs, 1,343 tasks) show that the proposed rule keeps GPU and CPU use within capacity while scheduling tasks at low cost. Compared with a multi-timescale optimization benchmark, it achieves a lower cost and, because it needs no solver, much less computation time. 

---
# Learning Optimal Transport Plans Via Autoregressive Token Regression 

**Authors**: Ivan Jaen Marquez, Sotirios Panagiotis Chytas, Vikas Singh  

**Link**: [PDF](https://arxiv.org/pdf/2611.22218)  

**Abstract**: Optimal Transport (OT) provides a principled framework for finding the minimum cost coupling between probability distributions. Classical solvers remain the gold standard for this task. At the same time, OT is an ideal testbed for studying whether learning-based amortized models can internalize the structure of OT type problems. Here, ground-truth solutions are available and marginal constraints are explicitly defined. Also, exact discrete solutions have well-characterized sparsity. We describe ToR-OT, a transformer-based model that learns to predict discrete OT plans through autoregressive token regression. We reframe OT into a structured sequence generation task. ToR-OT decodes sparse transport plans as sequences instead of solving an optimization task at test time. We focus on three properties: (a) Zero-shot generalization: pretraining on a large synthetic corpus of OT instances allows a single model to generalize to unseen distribution pairs and heterogeneous cost metrics without retraining, (b) Sparse primal decoding: by exploiting the inherent sparsity in exact discrete OT solutions, the decoder predicts only nonzero entries (avoids dense coupling outputs), (c) Amortized inference: on low-dimensional discrete problems, ToR-OT gives accurate approximate plans and compares favorably with per-instance neural OT baselines. Overall, ToR-OT is not yet a replacement for classical solvers, but a step toward understanding how autoregressive transformers can learn structure of constrained optimization problems like OT. 

---
# Communicate the Change: Efficient Expert Parallelism via Activation Compression 

**Authors**: Fengyuan Liu, Jay Gala, Siva Reddy, Dzmitry Bahdanau  

**Link**: [PDF](https://arxiv.org/pdf/2611.22215)  

**Abstract**: Scaling has driven the success of large language models (LLM), with sparse mixture-of-experts (MoE) architectures enabling greater parameter capacity without a proportional increase in computation. At large scales, MoE models rely on expert parallelism to distribute experts across devices. Each MoE layer requires two synchronous collective communication steps during inference: dispatching hidden states to the activated experts and combining the resulting expert outputs. As compute throughput has outpaced interconnect bandwidth, these synchronous collectives have become a major bottleneck to MoE scaling and efficiency. We address this challenge with EchoMoE, a lightweight architecture modification that retrofits existing MoE models, requiring approximately 8,500x fewer training tokens than full pretraining. Motivated by the observation that transformer blocks update the residual stream incrementally, we use a hidden state from an earlier layer as a reference and communicate only the compressed difference between it and the current state, rather than communicating each hidden state in full. EchoMoE periodically refreshes these reference states and transmits only the compressed differences synchronously, reducing the traffic on latency-critical paths without significant computational overhead. EchoMoE reduces inter-node communication volume by 58% and improves throughput by 22.6%, while preserving average performance within 3.8 percentage points across 18 benchmarks. EchoMoE further improves inference throughput by up to 2.38x over bandwidth-constrained Ethernet, paving the way for wide-area distributed inference. EchoMoE also tolerates quantization of the communicated representations without measurable quality loss. We release an efficient implementation for applying EchoMoE to pretrained MoE models, together with an inference integration for vLLM. 

---
# EGS: A Complementary Safety Framework for Educational LLMs 

**Authors**: Zexuan Fei, Zhengnan Guo, Yiming Luo, Kaiyi Huang, Aimin Zhou, Bo Jiang, Fei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2611.22203)  

**Abstract**: Large Language Models (LLMs) offer strong potential for personalized education, but adolescent-facing deployment raises risks of toxic, biased, misleading, or developmentally inappropriate content. We argue that educational safety should move beyond binary refusal: student queries are often ambiguous or curiosity-driven, requiring contextual guidance rather than simple blocking. We introduce the Educational Guardrail System (EGS), a lightweight, model-agnostic runtime framework that complements intrinsic alignment through query moderation, scenario-aware pedagogical guidance, and response moderation. EGS triages inputs by risk level, redirects gray-zone queries with category-specific educational interventions, and audits outputs before delivery. Experiments across general and education-specific benchmarks show that EGS improves moderation performance, refusal quality, and jailbreak robustness across diverse LLMs. These findings suggest that defense-in-depth guardrails can strengthen safety while preserving the pedagogical utility required in educational LLMs. Our code and model weights will be publicly available to support community adoption. 

---
# From Imperfective Paradox to Moorean Absurdity: Single-Chain Reasoning Lock-In and Multi-Candidate Verbalization Across Aspectual and Pragmatic Ambiguity 

**Authors**: An-Sheng Jhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22188)  

**Abstract**: Natural language entailment and coherence judgments often depend on whether an event merely occurred, reached a telic culmination, satisfied a plan, or was evaluated under multiple pragmatic standards. We test observable consequences of the single-chain premature lock-in hypothesis across four preregistered, sealed formal experiments on Qwen3-8B-Q8_0 (16,640 formal units, temperature=0.0): 15 of 16 hypothesis slots retain confirmatory status (12 rejecting H0 under within-experiment Holm control at alpha_fam=0.05; E3_H3 demoted to descriptive status). In Experiment 1, Direct-CoT vs. MCU differs by 25.00 pp on telic high-confidence completion (Holm p=1.31239e-05), while NLI-CoT collapses by -91.25 pp on uninterrupted atelic items (Holm p=1.81805e-21). Multi-Candidate Verbalization (MCV) elicits K=5 weighted estimates in one call, separating the prompt from unweighted (MCU) and self-weighted (MCW) readouts. Mandarin (E2) and English (E3) expose an Interruption-Anchored Realis Paradox under NLI-CoT. In Experiment 4, MCU-CF raises perceived coherence on implicit Moorean sentences by 26.64 pp over Direct-CoT (Holm p=0.0006), alongside Direct-CF - Direct-CoT = 29.15 pp and MCU-CoT - Direct-CoT = 17.31 pp. A 1,800-call OpenAI ablation and a stage="dev" P16.7 cross-model extension (4,240 planned/collected units across five external models: 4,222 valid, 18 format-missing, 0 transport-missing; <2% missing per slice; OpenAI gpt-4o-mini at top_logprobs=20 achieves 1,000/1,000 valid units) show that discrete NLI-CoT and scalar Direct-CoT judgments dissociate across models. Confirmatory effects concern observable prompt-and-readout protocols; internal premature lock-in remains a hypothesis. All code, stimuli, preregistered analysis plans, and verification tools are available at this https URL. 

---
# CueClash: Frontier Vision-Language Models Judge 3-D Relations Close to the Picture Plane 

**Authors**: Soham Dan  

**Link**: [PDF](https://arxiv.org/pdf/2611.22180)  

**Abstract**: A vision-language model (VLM) can judge 3-D relations from the picture or from the ground layout. In photographs the two usually agree. We separate them with CueClash, rendered scenes in which the 2-D cue for each of five relations is right on exactly half the statements. The shortcut gap is accuracy when the cue is right minus accuracy when it is wrong. For Claude Opus 5.5, GPT-5.5 and Gemini 3.6 Flash at low reasoning effort it is +18 to +37 percentage points. It is +7 to +29 without two relations whose cue-wrong items are confounded with borderline cases or occlusion. Given ground coordinates and each relation's definition, they answer 99 to 100% correctly. Pixel measurements with camera depths remove most of the gap for objects fitting through openings. The failure lies in the step from the picture to the layout. The maps that the models draw of the scene from above mostly keep the depth order but, like the picture, shorten depth. GPT-5.5's maps stay close to a copy of the picture. Opus 5.5's and Gemini's maps undo about 0.6 of this shortening. Gemini's then shrink all depths by a further uniform factor. Opus 5.5 follows the picture mainly on fits, and its maps correct depth about as much as human observers of real scenes do. The three models also lose accuracy on CV-Bench photographs where a reliable 2-D cue misleads. In two open models the layout information reaches the language model, but their maps lose much of it. 

---
# What RL on Expert-Built Tasks Teaches Nemotron 3.5 Super 

**Authors**: Sushant Mehta, Edwin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.22161)  

**Abstract**: We post-train NVIDIA's Nemotron 3.5 Super (120B total, 12B active parameters) with reinforcement learning (RL) on five expert-built task datasets, one at a time: real-world software engineering, complex constraints, long-horizon tool use, professional chart understanding and enterprise document reasoning. Each run trains a fresh rank-32 LoRA adapter with CISPO on Tinker, with no supervised stage, and each adapter is compared with the base model on the same tasks, harness and settings. On twelve headline benchmarks, eight built by other groups and four by Surge AI, the gains range from +4.5 points on IFBench to +13.8 on GDP_pdf, which rises from 10.4% to 24.2%; SWE-bench Pro V2 rises from 86.9% to 92.1%, Terminal-Bench 2.1 from 63.9% to 70.3%, MultiChallenge from 58.3% to 66.6%, AutomationBench from 14.8% to 27.5% and Chartography from 39.4% to 50.6%. The document adapter, trained without API tools, also gains +8.0 points on AutomationBench. In attempts read side by side with the base model's on tasks each adapter newly solves, the base model's failures recur in a few modes, and each adapter does more often what the base model missed: the tool-use adapter finds the data and the rules before acting, the coding adapter writes its own test or scratch files on SWE-bench Pro V2, the constraints adapter carries earlier turns into its answers, the chart adapter measures the marks in code, and the document adapter reads AutomationBench's spreadsheets. 

---
# EgoHandWAM: Probing the Role of Hand Motion in Egocentric World Action Modeling 

**Authors**: Kailin Wang, Haoxiang Jie, Yaoyuan Yan, Zhiyou Heng, Zhaosong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.22151)  

**Abstract**: How much does hand motion tell us about what a first-person camera will see next? EgoHandWAM forecasts three-dimensional wrist and fingertip trajectories and a future visual feature from the current image and recent hand states. We train on 751,065 EgoDex windows using a frozen DINOv2 encoder. On held-out development sessions, the action-only model reduces two-second wrist endpoint error from 60.26 to 56.48~mm relative to a stationary forecast. It still underestimates long reaches. Adding future-feature supervision gives nearly identical trajectory accuracy. With the recorded hand history, the visual head reduces smooth L1 error by 1.32\% relative to copying the current feature. Shuffling histories across sessions erases this gain. In a separate four-task probe, adding recorded future displacements to observed context and future tracking-validity masks reduces visual-feature error by 3.93\%. Masks alone give little improvement. A trajectory predicted from the same observations adds little. On two simulated RoboTwin tasks, substituting aligned human examples for an auxiliary batch of robot examples gives no consistent increase in task success across three training seeds. 

---
# When does deep reinforcement learning pay off in HVAC supervisory control? Four small Modelica case studies trained directly on FMUs 

**Authors**: Bingnan Zhao, Harshala Gammulle, Tharindu Fernando  

**Link**: [PDF](https://arxiv.org/pdf/2611.22144)  

**Abstract**: Deep reinforcement learning (DRL) is often proposed for the supervisory control of heating, ventilation and air-conditioning (HVAC) systems. The reported gains are, however, difficult to interpret, because the baselines are often weak and the attainable optimum is seldom known. This paper examines four small, reproducible case studies based on an office building in Brisbane, Australia: pre-cooling under a time-of-use tariff; chiller staging with condenser water reset; chilled water supply temperature reset with a wet cooling coil; and chilled water storage under time-of-use prices and a demand charge. The plants were modelled with the Modelica Buildings library, and in three of the cases a soft actor-critic agent was trained directly on the exported functional mock-up unit (FMU). Each agent was compared with tuned rule-based controllers and, where practicable, with a near-optimal reference from hourly exhaustive search or dynamic programming with perfect foresight.
In the two cases without storage the agent reached the near-optimal reference, but the margin above a well-chosen rule was small: 1.3% for condenser water reset and 0.18% for chilled water reset. In the latter case two common reset rules consumed more energy than a fixed set point. With a chilled water storage tank the margin was considerably larger. The agent outperformed both rules (-7.8%, compared with -3.9% and -6.3%) and adopted midday charging during the solar off-peak period, yet it obtained only about 45% of the saving achieved by the perfect-foresight schedule and did little to limit the demand peak. In the pre-cooling case a hand-tuned rule remained superior (-12.0% compared with -5.6%). These results suggest that studies of DRL for HVAC control should state the margin between the best rule and a near-optimal reference. Some practical difficulties encountered in training on co-simulation FMUs are also reported. 

---
# RoboTeam: Benchmarking Vision-Based Multi-Agent Control of Many Robot Arms in Real Time 

**Authors**: Changxuan Fan, Zhongwei Xie, Boyi Zhang, Shaoli Hu, Chang Xu, Xiyu Ren, Yufei Li, Qiao Xiao, Huihao Jing, Kawai Chung, Wenbin Hu, Hongyu Luo, Lingyun Xie, Yisen Gao, Chunyang Li, Haoyu Huang, Wuganjing Song, Yuxuan Liu, Chiu Yeung Tang, Fangneng Zhan, Xi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22138)  

**Abstract**: Multimodal agents can now control robot arms by calling tools and writing code. Existing benchmarks fall short in three ways: most test one or two arms on short tasks, many pause the world while the agent thinks, and many give it ground-truth object positions. We present RoboTeam, a simulated benchmark where agents control 4 to 16 robot arms in real time from camera images alone. Its 36 tasks are long jobs that the arms must do together, such as passing timbers or building a picture from hundreds of cubes. We start from the most flexible agent system, the adaptive system, where one agent decides when to fork delegates. With it, gpt-6.1-sol and opus-5.5 complete 28 and 27 tasks, and four other models at most 3. But at a high price: a typical task takes gpt-6.1-sol 3.5 hours and 218 million input tokens, and a third of its steps fail at least once and must be retried. We then compare it with three fixed systems: single-agent, hierarchical, and decentralized. All four complete a similar number of tasks but differ in time and tokens. The hierarchical and decentralized systems are faster but use up to 1.9 times the tokens, and the single agent uses fewer tokens but takes 1.6 times as long, so the adaptive system lies between them. Even so, no model or system is yet fast, cheap, and reliable at once. RoboTeam measures these limits where earlier benchmarks cannot: long multi-arm jobs, camera images only, and a world that never pauses. 

---
# Semantic Communication via Analog RF Computing: When the Radio Computes the Meaning 

**Authors**: Christo Kurisummoottil Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2611.22100)  

**Abstract**: Semantic communication (SC) systems transmit continuous latent representations of meaning, but computing the semantic encoder at resource-constrained edge devices incurs substantial energy and memory costs. To address this challenge, we propose an end-to-end SC framework that executes the semantic encoder in the client's RF receiver front end using analog RF computing. A base station (BS) broadcasts encoder weights as RF waveforms, while each edge device reuses its receiver's passive mixer to compute semantic latents at ultra-low energy. These latents are transmitted over the uplink for network-side task inference. We show that analog computing noise across encoder layers and channel noise affecting transmitted latents admit a unified mathematical model. We define a task-posterior semantic distortion metric and derive its second-order decomposition into per-client, per-stage semantic sensitivities. Using this decomposition, we obtain a closed-form semantic-aware water-filling allocation of the total client energy budget across clients, encoder layers, and uplinks, assigning tighter accuracy targets to semantically sensitive stages. This allocation induces a broadcast beamforming problem that accounts for client-specific, per-layer accuracy requirements. We further refine the allocation through stochastic optimization and jointly design it with multi-task noise-aware training of the shared semantic model. MNIST simulations show that, at matched total client energy, the proposed allocation reduces network semantic distortion by 9.2\%-19.2\% compared with uniform precision and 5.1\%-13.1\% compared with client-blind allocation. Noise-aware co-design yields a further 46\% reduction. 

---
# When AUC Meets Generative Reconstruction: A Unified Principle for Balanced Visual Representation Learning 

**Authors**: Boyu Han, Qianqian Xu, Shilong Bao, Zhiyong Yang, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22094)  

**Abstract**: Multimodal large language models (MLLMs) require visual encoders that extract fine-grained visual features while constructing semantically disentangled and discriminative representations. Conventional methodologies encounter a fundamental trade-off: contrastive learning enhances inter-class discrimination by suppressing intra-class variability, whereas generative reconstruction emphasizes perceptual fidelity but lacks explicit comparative supervision. The naive aggregation of these objectives via composite loss functions often induces optimization conflicts and instability due to gradient interference within the shared latent space. To resolve this, we introduce a unified representation-learning paradigm that reformulates generative reconstruction as a discriminative ranking problem within the generative output manifold. Our approach maintains perceptual reconstruction as a supervisory signal and integrates a ranking-based discriminative constraint by systematically ordering matched and mismatched reconstruction outputs. This formulation embeds discriminative information seamlessly into the generative process, alleviating the risk of loss competition. Specifically, we propose AUC Generative Reconstruction (AUCGR), which optimizes the area under the ROC curve (AUC) by ensuring higher scores for matched pairs than for mismatched pairs, and Partial AUC Generative Reconstruction (pAUCGR), which addresses supervision bias by focusing on mismatched comparisons within reliable feature subspaces. The proposed objectives are designed to be plug-and-play, requiring no architectural modifications to existing generative reconstruction frameworks. Theoretical analysis establishes rigorous connections to AUC/pAUC optimization and provides generalization guarantees. Extensive experiments validate that our method advances both discriminative accuracy and fine-grained perceptual representation. 

---
# HyCoInfer: Cloud-Edge Collaborative Inference for Large Language Models on Heterogeneous GPU-FPGA Architectures 

**Authors**: Zewen Huang, Xiangwen Zhuge, Yingkun Hu, Zheng Yang, Qiang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.22086)  

**Abstract**: Large language model (LLM) serving relies predominantly on cloud GPUs, yet autoregressive decoding is memory-bound and often leaves their arithmetic units underutilized. Edge FPGAs offer attractive decoding efficiency, but are ill suited to sustaining a high-concurrency serving workload alone. This paper presents HyCoInfer, a heterogeneous cloud-edge serving system that assigns prefill to cloud GPUs and distributes decoding across cloud GPUs and edge FPGAs. The design addresses three barriers to practical offloading: capacity balancing across heterogeneous accelerators, the large KV cache crossing a constrained wide-area link, and the latency added by heterogeneous execution. HyCoInfer incorporates FPGA decode capacity into GPU prefill/decode role partitioning and places requests according to queue load, prompt length, and network state. It further combines layer-wise pipelined transfer, decoder-side precomputation, and bandwidth-adaptive KV cache compression based on residual vector quantization. We implement HyCoInfer using two NVIDIA V100 GPUs and an AMD Xilinx Alveo U280 and evaluate it over 100-1000 Mbps links. At 500 Mbps, HyCoInfer delivers up to 4.11 times the request goodput of cloud-only vLLM and reduces mean end-to-end latency by up to 61.1%. Across different network conditions and workloads, the system improves overall serving capacity, with the largest gains on communication-intensive tasks, and reduces per-request energy. 

---
# Segmentation-Guided Direct Capacity Estimation of Rooftop Photovoltaics from Aerial Imagery 

**Authors**: Guibin Chen, Emma Fan, Yuzhang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.22076)  

**Abstract**: Distribution planners need installation-level inventories of rooftop photovoltaic (PV) capacity, but interconnection records may be incomplete or spatially aggregated. Conventional image-based estimators segment PV arrays, convert the mask to physical area, and multiply by an estimated power density, coupling capacity prediction to mask-derived area. We propose Direct Head, a segmentation-guided architecture that retains U-Net segmentation and soft-mask pooling while directly regressing nonnegative capacity from PV-region features and area descriptors. We evaluate three independent runs on 7,590 French rooftop installations under the original split and a department-based holdout. Relative to a parameter-matched area-density neural model, Direct Head reduces mean absolute error by 8.7% and 12.5%, and symmetric mean absolute percentage error by 14.0% and 20.7%, respectively. Comparisons with conventional estimators, controlled architecture variants, auxiliary-loss ablations, and representative cases show that segmentation provides useful spatial evidence without requiring capacity to be determined by an area-density product. 

---
# FM-Mamba: An Efficient Native 2D Selective State Space Model with Stability-Aware Discretization 

**Authors**: Shengyuan Hu, Yiyong Sun, Weihang Shan, Yongzhe Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22061)  

**Abstract**: When applied to two-dimensional features such as images, Mamba-based methods either flatten the feature map into 1D sequences and fuse multiple scans, which breaks spatial adjacency and multiplies computation, or recurse directly on the 2D grid but rescale the recurrence with a fixed input-independent factor and aggregate only the predecessors on a single anti-diagonal, leaving the discrete step unconstrained by 2D stability theory. We propose \textbf{FM-Mamba}, a native 2D selective state space model built on the general Fornasini--Marchesini (FM) model. Its coupled recurrence evolves along both spatial axes; cross-axis coupling induces a discrete diagonal-predecessor term $\tilde A_0$, and two independent input-dependent step sizes $\Delta_1(i,j)$, $\Delta_2(i,j)$ govern propagation along the two axes, so that the coefficients adapt per position and per direction. A normalized ratio $r_k=a_{0,k}/(a_{1,k}a_{2,k})$ yields a closed-form piecewise bound on the two step sizes; when a configuration falls outside the stable interval the steps are projected onto this bound, and the criterion is re-verified per position after training, which suppresses systematic local amplification without global damping. The recurrence is executed by a dual-diagonal wavefront scan: each update gathers information from two anti-diagonals, and the input enters the current state through a single projection, so that neither training nor inference flattens the feature map. Code is available at this https URL 

---
# Weight-Space Programming Doesn't Need Transformers 

**Authors**: Hyunjong Ok, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.22055)  

**Abstract**: Standard neural architectures such as the Transformer were chosen because they train well with gradient descent and run fast on GPUs. In weight-space programming, a large language model (LLM) writes target weights by hand without gradient-based training, so those reasons no longer apply. Should we keep the Transformer? Given minimal guidance, ten labeled examples, and a sandbox, mainly on multiple-choice questions, we task three frontier LLMs with either generating the weights of a fixed architecture or designing the architecture themselves. When free to design, all three compile plain-text facts into a sparse table queried by a fixed procedure, reaching up to 54.4% test accuracy. An LLM-designed model achieves higher test accuracy than the most accurate released Transformer that the same LLM wrote while using about 1,800x fewer parameters. Without gradients, they choose a table, in which the LLM can state what each weight means and can find and rewrite a wrong one. Each wrong answer points to one row, and removing a row changes about one other answer. From this, we propose an associative-memory-based model up to 17x faster on GPUs than a same-sized Transformer. We also test pretrained weights, other benchmarks, and tasks where models generate answers. Pretrained LLMs remain more accurate; the claim concerns which structure to build when writing weights. 

---
# RLDE: A Reinforcement Learning Framework for Interpretable Iterative Document Editing 

**Authors**: Nikhil Parkar, M. Marimuthu, Riya Joshi, Karthikeyan N  

**Link**: [PDF](https://arxiv.org/pdf/2611.22039)  

**Abstract**: Improving the quality, clarity, and flow of long documents remains a challenging problem in NLP, as human editing typically occurs through iterative refinement rather than a single-pass rewrite. Most existing automated systems act as black-box generators or rely on supervised learning, limiting interpretability and failing to capture the sequential nature of revisions. In this paper, we propose RLDE, an automated end-to-end metric-driven RL framework that formulates document revision as a reference-free MDP problem (unsupervised with respect to paired document-revision targets). Working with a limited action space (NOOP, INSERT, DELETE, REPLACE), RLDE leverages Pointer Networks for dynamic span selection at the token level and sends generative actions to the frozen BART-base model to obtain context-aware suggestions. PPO agents learn to optimize a multi-objective reward function based on readability, perplexity, and grammar metrics with the help of a novel SBERT-based Semantic Preservation Penalty constraint to avoid deletion degeneracy. Evaluations on the BEA-2019 dataset showed statistically significant results (p < 0.01) across all metrics, where Flesch Reading Ease increased from 49.8 to 59.4, perplexity decreased from 63.7 to 39.8, while preserving 93% of semantic information. When compared to Chain-of-Thought (CoT) prompt-engineered models like GPT-4 and LLaMA-2, RLDE shows high performance at a very low computational inference cost. Moreover, the quality of edits generated by RLDE has been verified via human evaluations, scoring 4.9/5 for meaning preservation and 4.4/5 for edit quality. 

---
# GauntletScore: Auditable Fact-Checking That Matches the Best Published Checkers on LLM-AggreFact 

**Authors**: Ernie Hobbs, Jim Hinson  

**Link**: [PDF](https://arxiv.org/pdf/2611.22027)  

**Abstract**: Checking whether a generated sentence is supported by its source document is a standard safeguard for AI-written text, and the strongest published checkers are single models. We describe GauntletScore, an evidence engine whose few rules and thresholds were set on the benchmark's development split. It puts typed questions to independent readers from different providers and records each answer as a measurement in an evidence ledger. It combines the measurements with a fixed Bayesian rule that counts a contradiction only when two readers agree. On the full LLM-AggreFact test split (29,320 claims across 11 datasets), following the leaderboard's approach, GauntletScore achieves 79.51 balanced accuracy, above every average published on the leaderboard as of 15 November 2026. That places it level with the best of them, nimble-v4-max (79.45 from its published dataset scores; the engine's 95% interval, 78.47 to 80.53, contains it). On the same claims it beats two of its three readers used alone (Readers A and B) by 1.66 and 2.17 points (95% intervals 0.80 to 2.53 and 1.29 to 3.06). Without AggreFact-CNN, described below, the leads are 0.90 and 1.32 and both intervals stay above zero. The engine also records the evidence behind every verdict, which a single model does not. Its largest margin is on full-length news articles checked against multi-sentence summaries (AggreFact-CNN), a format that resembles much regulated writing. There it scores 76.45, the best on the leaderboard, against 70.1 for the next best and 63.9 to 67.8 for the six systems with the highest averages. The test split was run once, on an engine frozen before the first test claim, under a stopping plan committed in advance. We release per-claim predictions and the scoring script. 

---
# Computed Permutation Trellis Coding: Near-Optimal Quantization from Two Multiplications and a 64-Byte Table 

**Authors**: Muhammed Fatih Balın  

**Link**: [PDF](https://arxiv.org/pdf/2611.22019)  

**Abstract**: We propose Computed Permutation Trellis Coding (CPTC), a low-bit quantization format for tensors written once and read many times. Formats that round each value on its own, such as INT4 and FP4, waste a large part of their bits against the information-theoretic ideal. Trellis codes recover this loss by encoding values jointly, but prior decoders either need a large lookup table for their best quality or compute values with a hash that restricts what they can decode. CPTC's decoder multiplies its state by two coefficients and adds the top bits of the products to index a 64-byte table. Because that sum is not uniformly distributed, the rarely used ends of a small table can hold values far out in the tails. One decoder serves every bit width from 1 to 6, reproduces the standard integer and floating-point formats exactly, and comes within 0.07 bits of the ideal at 4 bits. It matches a random-permutation trellis code whose table is thousands of times larger, and exceeds every QTIP codebook. In LLM weight quantization without fine-tuning, CPTC has the lowest KL divergence to the unquantized model at every tested bit width on Llama 3.1 8B and 70B Instruct against QTIP's codebooks (by up to 27%). The decoder has been built into the AWS Trainium 4 AI accelerator, where it runs at the throughput of the native FP4 and FP8 formats for a negligible fraction of the systolic array's area. Fused encoding kernels quantize an 8B model's linear layers in under two minutes on one GPU and, in cycle-level simulation, on one Trainium 4 chip. 

---
# DiagTrace: An Interactive Benchmark for Clinical Decision Making 

**Authors**: Yufan Wang, Rui Yang, Yi Lin, Chunhua Weng, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2611.22009)  

**Abstract**: Large language models (LLMs) have shown strong performance on medical question answering and diagnostic reasoning benchmarks. However, existing evaluations typically provide all patient information upfront and assess only the final diagnosis or rationale. This setup ignores the need to acquire diagnostic evidence incrementally from incomplete information in real-world diagnosis. To bridge this gap, we introduce DiagTrace, a source-grounded benchmark for evaluating whether LLMs can reconstruct the documented diagnostic trajectory through interactive information seeking. Each benchmark instance is built from a PubMed Central case report and includes an initial presentation, an event bank, a clinical course, and diagnostic references. DiagTrace contains 510 cases across eight specialties, with 225 used to evaluate four LLMs on clinical-course alignment, interaction quality, and efficiency. Results reveal trade-offs in coverage, source-consistent ordering, and turn use. 

---
# Modeling Document and Section Hierarchies in Embedding Space 

**Authors**: Kriste Krstovski, Sharyn O'Halloran, Kshitij Chandna  

**Link**: [PDF](https://arxiv.org/pdf/2611.21980)  

**Abstract**: Embedded topic models integrate word embeddings into the topic generation process, producing coherent topics and robust performance on large vocabularies. However, each document is represented by a single topic distribution, ignoring its internal structure. Many real world collections, including legal documents, company and government reports, and online discussion forums, exhibit hierarchical organization in which documents consist of multiple sections that often address distinct subtopics. We introduce the Hierarchical Embedded Topic Model (hETM), a structured topic model that couples the topic distribution of a document and those of its sections through a hierarchical prior in the embedding space. We derive an amortized variational inference procedure for hETM and evaluate it on structured collections from each of these domains. Experimental results show that hETM achieves the lowest or near-lowest held-out perplexity among flat and hierarchical topic model baselines, along with comparable or higher topic quality. It also performs strongly on a downstream clustering task, achieving the best clustering performance on most of the collections. 

---
# Pathology Randomization for Synthetic Priors and Zero-Shot Detection 

**Authors**: Alexandra G. Roberts, Tonmoy Hossain, Mark Olchanyi, Stefan Brandstätter, Jonathan Williams Ramirez, Ana Lawry Aguila, Matthew Rosen, Annabel Sorby-Adams, W. Taylor Kimberly, Juan Eugenio Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2611.21975)  

**Abstract**: Hospital brain imaging archives hold a diverse array of pathology at sample sizes beyond prospective studies, particularly for underrepresented populations and diseases. To unlock uncurated clinical data and scale analysis, methods are needed to detect abnormalities across contrasts and resolutions. Synthetic data has provided robust, contrast-agnostic models for a variety of human neuroimaging tasks in healthy anatomy, but pathology detection and other tasks in clinical settings still require real patient data, contrast-specific models, or disease-specific generators. We propose PathoDraw, a generative model of brain MRI with pathology. A conditional distribution of lesions given the healthy anatomy is drawn into the label map before contrast-agnostic rendering, and a single network estimates healthy anatomy with exact signed intensity deviation of any input scan. The contrast-agnostic 3D UNet learns the healthy labels under the lesion and the signed deviation directly and identifies pathology via robust tissue statistics and learned features, despite no training on real pathology. We show zero-shot detection on public stroke, tumor, and white matter hyperintensity datasets as well as archived clinical data, spanning $T_1$w, FLAIR and $T_2$w at various resolutions under a recent unsupervised anomaly detection benchmark protocol. Our single model exceeds or performs comparably to contrast-specific anomaly detectors trained on real scans and keeps healthy anatomy intact. The same generator trains networks for super-resolution and for registration in the presence of pathology. Our model, robust across acquisition and disease, enables the use of uncurated, archived data for clinical analysis. Our code is available at this https URL. 

---
# OCCAM: Improved Single-Cell Representation Learning and Perturbation Effect Prediction 

**Authors**: Ihab Bendidi, Kian Kenyon-Dean, Alisandra K. Denton, Auguste Genovesio, Emmanuel Noutahi  

**Link**: [PDF](https://arxiv.org/pdf/2611.21959)  

**Abstract**: Predicting the effects of genetic perturbations requires models that generalize beyond observed cell contexts and gene combinations. We introduce Occam, a multilayer perceptron (MLP) framework for representation learning and perturbation-effect prediction. Occam combines cell representations learned through denoising with gene interaction, text, and protein-sequence features. It achieves the highest aggregate representation scores for perturbation information and cell-type preservation with batch integration. In cross-context few-shot prediction, a perturbation and a cell context are each observed during training, but never together. Occam better captures the measured response patterns for these unobserved combinations than four retrained competitors or direct transfer of the same perturbation's measured effects from other training cell lines. Without fine-tuning, Occam best distinguishes perturbations among the evaluated models and simple baselines in cell lines held out from predictor training, across seen and unseen assays. Predicting expression levels accurately in these settings remains challenging. For double perturbations, which target two genes simultaneously, Occam composes learned single-gene responses to better capture the combined response and reduce expression error compared with competing models and summed single-gene effects. These gains extend to both strongly additive and weakly additive responses. Together, these results support simple MLPs for cell representation learning and perturbation-effect prediction across contexts and gene combinations. 

---
# Weight-space programming for tractor guidance controller: large language models natively write neural network weights that steer a real tractor 

**Authors**: Timo Oksanen  

**Link**: [PDF](https://arxiv.org/pdf/2611.21958)  

**Abstract**: Weight-space programming is a process where a large language model reads and writes the weights of a neural network directly, guided by a natural language specification, without an optimiser or gradient descent. This study applies it to automatic tractor guidance, where a 253-parameter neural network steers along paths of lines and arcs and must be adapted when the machine changes, such as a slower steering actuator. GPT-6 Astra and Claude Fable 5.1 (also 5.2) edited the network for our research tractor using a small budget of simulated evaluations to repair a network trained for another tractor and write a controller into a network with zero weights. We compared them with an evolution strategy, gradient descent and pure pursuit in simulation and on the real tractor. Both models read the network and analysed the closed loop with control theory correctly, and Fable's repair without simulation succeeded only occasionally. With simulated feedback, both models repaired the trained network in single sessions and wrote a working controller from zero weights, whereas from zero the evolution strategy found none. The controllers written from zero were sparse and in a classical cascaded structure; one model edited the trained network while the other replaced it. Gradient descent reached higher accuracy with far more simulation, and all language model controllers steered the real tractor without intervention, with errors of the same order as pure pursuit. Language models can program the weights of a neural controller directly, much as control engineers design one. We call this Weight-Space Engineering. 

---
# Eight in Ten LLMs Leak System-Prompt Secrets: Nevan, an Agentic Capture-the-Flag Study 

**Authors**: Ana Clara Zoppi Serpa, Anderson Rocha, Hélio Pedrini  

**Link**: [PDF](https://arxiv.org/pdf/2611.21956)  

**Abstract**: We audit instruction-hierarchy enforcement across 363 text-in/text-out models from the OpenRouter catalog. The research question we answer with this experiment is which models disclose a protected secret embedded in the system prompt, and through which techniques. Each model guards a secret 16-character flag set in its system prompt, while an artificial intelligence (AI) agent (OpenCode, powered by DeepSeek V4.1 Flash), steered by a human researcher, tries to persuade it to emit the flag. The flag was exfiltrated from 296 of the 363 models, plus four models released after the initial catalog freeze, for 300 captures in total. Of these, 170 involved a non-English language pivot, most often French. Each capture shows that a model ranked a user request above its system instruction at least once. It does not show that the model provably resists untested attacks, nor that it would consistently leak semantically sensitive data, nor that it is vulnerable to harmful-content jailbreaks. For practitioners, we draw three lessons: do not store secrets in system prompts, enforce critical behavior outside the model with output validation and deterministic checks, and do not let a large language model (LLM) judge be the sole decider of consequential outcomes. 

---
# ThermLeT: Transformer-based Temperature Prediction for 2.5D Chiplet Architecture 

**Authors**: Varun Darshana Parekh, Anusha Devulapally, Sivani Devarapalli, Cassius Henderson, Shimeng Yu, Vijaykrishnan Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2611.21954)  

**Abstract**: The drive toward heterogeneous 2.5D chiplet packages promises unmatched performance and customization, yet the resulting density amplifies thermal challenges. To facilitate rapid design space exploration, we present ThermLeT, a multimodal transformer-based, scale invariant surrogate that integrates power density maps, spatial coordinate embeddings, material properties and heat transfer rate to predict steady state temperature in 13 ms, about four orders of magnitude faster than conventional finite element solvers such as ANSYS. To train the transformer model, we also create a dataset of 6,221 chiplet layouts perturbed from three representative architectures. On a 150 sample test set, ThermLeT achieves an RMSE of 3.16°C and an MAE of 2.24°C, improving accuracy by 35% over ThermEDGe and by 71% over HotSpot, while running approximately 6x and 11x lower inference latencies respectively. Ablation studies confirm robust generalization to unseen floorplans. Code and dataset available at this https URL 

---
# Benchmarking Deep Learning Models for Wheelchair Configuration Classification in Accessible Automated Vehicle Applications 

**Authors**: Samir Ahmed, Shaaz Saleem, Devin Dhami, Wesley Maia, Angel Martinez-Sanchez, Parthib Roy, Jannatul Ferdous Srabonee, Ahmed Sabbir Arif, Ross Greer  

**Link**: [PDF](https://arxiv.org/pdf/2611.21945)  

**Abstract**: Making automated vehicles accessible requires consideration of vulnerable road users, including wheelchair users. To offer a more tailored experience that supports independence for this group, vehicle design needs to accommodate the wide range of wheelchair configurations. However, wheelchairs are individualized, varying in frame geometry, propulsion, and visual appearance. Moreover, occupants or accessories may obscure visual cues. Existing mobility-aid datasets only label the presence of a wheelchair, rather than describing its specific configuration. To fill this gap, this paper introduces the Wheelchair Configuration Image Dataset (WCID), comprising 467 face-blurred images across 192 capture groups solely for evaluation. Each image is annotated with (i) a two-class label of manual or powered, (ii) a chair subtype, (iii) occupancy, viewpoint, setting, lighting, occlusion, propulsion, and control, with a 273-image validation split and 194-image test split. Three convolutional neural network backbones and a frozen vision-language probe are trained on an external image dataset and evaluated on WCID to measure cross-dataset generalization, while also testing robustness using Gaussian blur. Convolutional models do well on clean images, but performance drops under strong blur, leaving a gap before deployment. Code, split assignments, labels, trained weights, and the face-blurred images is available at: this https URL 

---
# Does the Model Use What It Cites? Counterfactual Tests of Grounded Reasoning Chains in Small Video-Language Models 

**Authors**: Jugal Gajjar, Kamalasankari Subramaniakuppusamy  

**Link**: [PDF](https://arxiv.org/pdf/2611.21940)  

**Abstract**: A grounded reasoning chain cites a frame span and a box for every step, so a reader can check where the model looked. Whether the answer depends on the cited region is a different question. We test it by intervention: the regions a model cited in its own chain are blacked out, and the share of correct answers that flip is compared with blacking out boxes of the same size elsewhere on the same frames. The paired difference, the cite gap, needs no annotation. We apply it to 2-3B video-language models trained on exact CLEVRER chains and on pseudo-labelled NExT-QA chains. With the same weights, the cite gap is 0.089 on CLEVRER and 0.003 on NExT-QA. Pooled over models, masking the cited boxes flips 2.2 times as many correct answers as the control on CLEVRER and 1.1 times as many on NExT-QA, also among the answers that depend on the video at all, while the boxes pass a grounding check on both. Masking the cited entity on every frame, instead of the cited box, flips 1.9 times as many answers as its control. The NExT-QA chains were supervised to cite two frames of entities that stay visible on most frames, so this outcome is what the supervision predicts; the point is that the usual check on grounded chains cannot see it. A citation can be verifiable and relevant without the answer depending on it, and a test of grounded reasoning should remove what a chain names, not only where it points. 

---
# V2V Communications Using Blockchain-Enabled 6G Technology and Federated Learning 

**Authors**: Tahir H. Ahmed, Jun Jiat Tiang, Azwan Mahmud, Dinh-Thuan Do, Truong Tran, Shahid Mumtaz  

**Link**: [PDF](https://arxiv.org/pdf/2611.21932)  

**Abstract**: This study proposes an interesting approach for vehicle-to-vehicle (V2V) communication, which integrates blockchain technology, federated learning (FL), and allocation optimization of latency and resources. The research evaluates the proposed system using various performance metrics such as packet delivery ratio (PDR), model accuracy, and latency and demonstrates its superiority over existing techniques. Furthermore, the system provides enhanced security through consensus optimization and k-anonymity for data privacy. Overall, the proposed system is a promising solution for efficient and secure V2V communication in the era of connected and autonomous vehicles. Moreover, the proposed approach achieves higher reliability, lower latency, and better resource utilization compared to traditional 5G. 

---
# ALOHA Lightning: Learning Fast and Precise Manipulation 

**Authors**: John Hua Yao. Qi Wu, Yihuai Gao, Chelsea Finn, Zipeng Fu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21919)  

**Abstract**: Learning from human demonstrations has enabled robots to acquire a wide range of manipulation skills, but learned policies typically execute far slower than ordinary humans. This speed gap is mainly due to lack of an interface for collecting demonstration data at high speed, and the difficulty in training policies that can robustly execute high-speed motions. In this paper, we present ALOHA Lightning (Flash ALOHA), a system for learning fast and precise robotic manipulation. Our system uses kinesthetic teaching to intuitively collect near-human-speed demonstrations on a backdrivable bimanual platform, yielding natural and fast trajectories. We also present a learning pipeline that enables smooth high-speed execution through test-time action smoothing and aligns the visual data distribution between data collection and deployment with masking. Given 50 demonstrations for each task, ALOHA Lightning autonomously completes bimanual tasks such as ball tossing and catching, folding shorts, battery insertion, and bussing tables for over 80% success rates while close to human speed. Project website: this https URL 

---
# A Temporal Algebraic Framework for Data Stream Evaluation Under Delayed Labels and Distribution Shifts 

**Authors**: Ronaldo C. Prati, Elaine Ribeiro de Faria  

**Link**: [PDF](https://arxiv.org/pdf/2611.21914)  

**Abstract**: In data streams, labels often arrive late or only for some instances, class prevalence drifts, and rare classes vanish from short windows. Standard evaluation then confuses a degraded decision boundary with a change in prevalence, and is biased when labels are delayed or selected by the model. We extend an algebraic framework that obtains pooled (micro) and per-class (macro) measures from one confusion matrix to streams, weighting the counts in time by exponential fading or a sliding window; the update costs O(m^2) per step, whatever the stream length, and each pooled measure stays a weighted combination of the per-class ones. Delayed and selective feedback enter as an observation mask: until its label arrives, an instance is counted with the model's calibrated class probabilities, and the bias of this interim estimate is driven by their calibration error. Audited conventions report no value for a class without verified instances. We separate prior shift from concept drift by monitoring prior-free coordinates (recall and specificity) and attribute a change in a measure to the two sources with a first-order Jacobian decomposition. On synthetic streams and four real datasets, a detector on these coordinates with a memory per class raises no excess false alarms under label shift, unlike the Jacobian alarm and error-rate detectors. The correction for delayed labels recovers recall in some real settings and is only partial for precision, specificity and F1. Twenty-five statements are checked in Lean 4, and the code is in the aggmetrics package. 

---
# Domain-Tuned Diffusion Super-Resolution for 4K Facial Asset Maps 

**Authors**: Elly Akhoundi, Mathieu Lamarre, Thibault Goyallon, Patrick Anderson, Étienne Danvoye  

**Link**: [PDF](https://arxiv.org/pdf/2611.21890)  

**Abstract**: Physically based facial assets require 4K albedo and displacement maps for cinematic close-ups, yet generative pipelines commonly produce them at only 1K. Naively applying photographic super-resolution is a poor fit: facial UV maps have domain-specific statistics, and displacement is a scalar height field whose errors directly distort geometry. We present a domain-tuned diffusion upscaler for 4x (1024 to 4096) facial asset super-resolution. Starting from a bilinearly upsampled map, we adapt a frozen SDXL backbone per modality with lightweight LoRA modules, while a Tile ControlNet preserves input structure as the diffusion prior restores pore- and wrinkle-scale detail. On held-out maps, our method produces sharper, more faithful detail than bilinear and off-the-shelf GAN and diffusion super-resolvers, reducing albedo LPIPS from 0.21 to 0.08 while avoiding misplaced or excessive displacement relief. These results show that lightweight domain adaptation can produce render-ready 4K facial asset maps without a custom super-resolution architecture. 

---
# Cancellation-Free Constrained SquareCB for Safe Controller Selection 

**Authors**: Kartik Pandit, Sourav Ganguly, Arnob Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2611.21880)  

**Abstract**: We study constrained contextual bandits, in which a learner chooses one of $K$ actions in each round to minimize loss while keeping the expected cost of its play below a budget. Existing reductions to online regression weight the cost by a Lagrange multiplier learned from past constraint residuals, so slack in some rounds can offset violations in others. We instead run SquareCB proposed in \citep{foster2020beyond} with a rectified weight on the cost: it is zero when the estimated cost is within the budget, and otherwise just large enough to bring it back to the budget. Under realizability and Slater's condition with margin $\delta$, the algorithm attains regret $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ and cancellation-free violation $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$, where $\mathrm{Reg}_{\mathsf{Sq}}$ is the error of the regression oracle. The same bounds hold when the budget must be met in every context, potentially adversarially chosen contexts. In controller selection for a safe-control-gym quadrotor and a BOPTEST building, the violation is 6 to 9 times lower than that of primal-dual and virtual-queue baselines, with a lower or comparable regret. 

---
# FaceRiff: Data-Efficient Unified Text-to-3D Face Asset Synthesis via Geometric Conditioning 

**Authors**: Elly Akhoundi, Mathieu Lamarre, Thibault Goyallon, Patrick Anderson, Étienne Danvoye  

**Link**: [PDF](https://arxiv.org/pdf/2611.21867)  

**Abstract**: High-resolution facial datasets with consistent UV-mapped geometry, albedo, and physically based displacement are scarce, yet such assets are essential for photorealistic digital-human rendering. Existing 3D face generation methods are often slow, single-modality, or require large specialized datasets. Meanwhile, large image diffusion models encode strong facial priors but do not natively produce lighting-independent textures, pore-scale displacement, or UV-aligned geometry. We present FaceRiff, a unified, data- and parameter-efficient framework for generating complete 3D face assets (mesh, albedo, and displacement) directly from text in a shared UV space. FaceRiff adapts a pretrained SDXL backbone with task-specific LoRA adapters and uses UV normal maps as geometric conditioning: per-instance normals guide albedo and displacement synthesis, while a dataset-mean normal provides a canonical spatial anchor for mesh generation. Using only 2,049 scans from approximately 350 identities and training on a single GPU, FaceRiff updates less than 3% of the backbone parameters per task and produces render-ready 3D face assets without per-instance optimization. Compared with FaceG2E and DreamFace, FaceRiff produces greater per-prompt diversity while preserving prompt alignment and fixed-topology UV consistency, and additionally generates lighting-independent albedo suitable for PBR relighting and pore-scale displacement absent from prior text-driven baselines. A blind study with 45 industry professionals shows that FaceRiff achieves substantially higher perceived realism than DreamFace, approaching that of real captures. Overall, FaceRiff demonstrates that pretrained diffusion priors can be adapted to unified 3D face asset synthesis under limited data. 

---
# PromptPack: Training-Free Augmentation of Diffusion Editors for Intent-Aligned Spatial and Semantic Control 

**Authors**: Sophia Zalewski, Mustafa Munir, Enting Liu, Anjul Patney, David Dynerman  

**Link**: [PDF](https://arxiv.org/pdf/2611.21865)  

**Abstract**: Many image-editing tasks require confining changes to intended spatial regions while exactly preserving user-designated source content. Instruction-driven editing pipelines often leave such user-intent underspecified, while inpainting editors condition generation on explicitly defined edit support but not on the spatial assignment of individual edit concepts. PromptPack addresses both limitations by equipping diffusion editors with two explicit 'Edit Guides': an edit-role partition that labels each pixel as 'Keep,' 'Permit,' or 'Modify' to specify where edits may occur, and a regional-prompt assignment that maps spatial regions to text prompts to specify which concepts apply where. These guides may be manually specified or, for source-object edits, automatically derived from a text-grounded SAM 3 mask. Without retraining, PromptPack injects independently embedded regional prompts into the editor's fixed-length text sequence and uses the Edit Guides to spatially mask attention, gate flow updates, and restore protected source pixels. We integrate PromptPack into FlowEdit FLUX, FlowEdit SD3, and FLUX Kontext and demonstrate improved requested-concept placement and protected-content preservation across all three integrations on PIE-Bench++. In a pairwise user study, PromptPack achieves an 87% preference rate over FlowEdit FLUX, establishing PromptPack as a practical, training-free interface for intent-aligned diffusion editing. 

---
# The Hidden Cost of MoE Diffusion Decoding: Profiling and Fixing the Sampler 

**Authors**: Harshit Samrat, Ramakrishna Paramahamsa Kompella  

**Link**: [PDF](https://arxiv.org/pdf/2611.21852)  

**Abstract**: Masked diffusion language models (dLLMs) decode by iteratively denoising a full canvas of tokens, so every forward pass runs a per-token sampling and stopping pipeline over the entire canvas-by-vocabulary logits tensor. We profile single-user decoding of a 26B-parameter diffusion Mixture-of-Experts model on a bandwidth-bound edge accelerator (NVIDIA GB10) and report a result that is easy to miss: after weight quantisation, this logits pipeline, not the expert GEMMs, is the single largest consumer of decode time. Unlike autoregressive decoding, where sampling touches one position's logits per committed token, diffusion decoding processes S times as many logit entries per output token, where S is the number of denoising steps. We show this pipeline is largely redundant or fusable and remove it in two lossless tiers: a bit-exact tier whose output is byte-identical to the reference decoder, and an accuracy-parity tier using a fused single-pass entropy kernel and Gumbel-max sampling. Combined with a W4A16 expert kernel, the full stack reaches 2.4-2.7 times end-to-end speedup over bf16 at unchanged benchmark accuracy on GSM8K, HumanEval, and MATH, on a single 273 GB/s device. We confirm the finding on a second, independently built diffusion MoE (LLaDA2.0-mini: pipeline 28.8% of decode, fused W4A16 3.3 times at accuracy parity) and show it disappears on a dense diffusion model (3.6%). The cost is specific to MoE diffusion, where quantisation makes the backbone cheap. Code and quantised weights are released. 

---
# AR-WAM: efficient streaming world-action modeling for autonomous driving 

**Authors**: Baiting Luo, Shatadal Mishra, Ahmadreza Moradipari, Nejib Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2611.21851)  

**Abstract**: Autonomous driving requires anticipating how traffic will evolve while responding quickly to new observations. Video foundation models offer the first ability, but repeated history encoding and iterative video generation make them costly to use for driving. We present AR-WAM, an efficient streaming world-action model that combines autoregressive temporal modeling with single-pass future conditioning. Because its video backbone is causal, AR-WAM updates its visual context incrementally, reusing past computation and conditioning future prediction on observed motion. We further hypothesize that future features computed from recent observations can improve generalization in dynamic traffic without iterative video generation. AR-WAM therefore obtains these features in one forward pass and uses them to condition trajectory generation, and reinforcement learning then improves the policy through direct optimization of driving rewards. With a single front camera, AR-WAM achieves 91.5 PDMS after supervised training and 94.0 after reinforcement learning on NAVSIM-v1, outperforming the compared strong end-to-end world-action models. In a dynamic setting of continuous driving, supervised AR-WAM keeps its visual history across decisions and outperforms strong world-action baselines. 

---
# Inventa: Decentralized Self-Improving Agents for Autonomous Scientific Discovery 

**Authors**: Prithwish Jana, Nikos Kanakaris, Purak Jain, Anoop Deoras, Sahika Genc  

**Link**: [PDF](https://arxiv.org/pdf/2611.21847)  

**Abstract**: Large language model (LLM) agents can generate research ideas and improve scientific discovery through iterative experimentation. However, increasing the number of agents or iterations does not automatically translate to producing more diverse or feasible ideas. Agents may converge on similar candidates, propagate unsupported claims through shared memory components and repeatedly apply search strategies that no longer yield any improvements.
We introduce Inventa, a framework for automated scientific discovery that employs autonomous research units and verified shared knowledge. Each unit creates a lineage of candidate ideas using a mutator agent. A local orchestrator agent periodically adapts the mutator's harness. When progress stalls, the orchestrator agent replaces its model or restarts the idea evolution process. Verifiers evaluate each candidate idea and record verified improvements, useful components and constraint violations in a shared knowledge, the only channel between units. We evaluate Inventa on five benchmarks spanning the research cycle, covering retrieval (ScholarCatalyst), ideation (IdeaBench), proposal writing (IdeaScientist), and scientific law discovery (EurekaBench, SciLaws-Bench). It leads all five, achieving 73.6% on EurekaBench versus 56.2% for CORAL. On IdeaBench, it discovers 26.0 novel ideas per paper, compared with 18.6 for ResearchAgent, with 82% fewer model calls. 

---
# EmbedGym: Letting Coding Agents Develop and Debug Embedded Firmware Without Physical Hardware 

**Authors**: Minghui Zhao, Yiming Li, Sophie Zhu, Maxwell Khor, Lauren Ouyang, Hai "Helen" Li, Yiran Chen, Tingjun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.21845)  

**Abstract**: Modern coding agents perform well on general software benchmarks because they can run code and unit tests in a closed loop. Embedded firmware development, however, breaks this loop: a program that compiles can still fail on real hardware due to pin polarity, bus protocol, timing, or networking bugs that an agent at a text terminal cannot observe. Existing embedded coding agents either skip hardware execution, drive GPIO-only simulators with multi-second delays, or test on physical boards in a static environment. In this paper, we present EmbedGym, an interactive virtual hardware environment that lets coding agents develop and debug embedded firmware without physical hardware. EmbedGym uses instruction-level microcontroller unit (MCU) emulators to execute the same binary that would be flashed onto a physical board, connects them to 41 peripheral models, an analog circuit simulator, and virtual WiFi/Bluetooth Low Energy (BLE) endpoints, and exposes an interface for agents to change sensor inputs, press buttons, and inspect hardware states. On top of it, we build EmbedGym-Bench, which consists of 598 tasks across 6 MCU-framework pairs graded by hidden timed test vectors, using a five-stage multi-agent workflow that synthesizes peripheral models and tasks from datasheets and driver libraries. We expose EmbedGym to a coding agent at three levels and evaluate eight models over more than 14K trajectories. Even when compilation succeeds 99.7% of the time, 19.9% of compiled firmwares fail at runtime (up to 73% on individual targets). In contrast, interactive hardware testing in EmbedGym increases pass rates across models by +4.9% to +14.7% (up to +31% on individual targets). Our results also show that an agent with EmbedGym outperforms prior embedded agent frameworks, and lightweight models with EmbedGym match frontier models without it at 30-70x lower cost. 

---
# Coordination of Heterogeneous Teams with Uncertain Task Requirements 

**Authors**: Jongann Lee, Melkior Ornik  

**Link**: [PDF](https://arxiv.org/pdf/2611.21828)  

**Abstract**: Coordinating a team of heterogeneous agents to complete tasks with unknown requirements is a challenging multi-agent planning problem. Agents differ in their ability to discover task requirements and in their ability to engage tasks, so the planner must coordinate task discovery, execution, and staging. This paper introduces a hierarchical planner that uses a graph neural network policy to assign actions to agents, then finds a path to the action's location using a shortest-path algorithm. The central part of our approach is the CAAT graph, which is a graph representation of the planning context consisting of the agents, tasks, capabilities, and actions. We demonstrate that our learned planner can jointly coordinate a heterogeneous team to simultaneously discover task requirements and complete tasks in a shorter time compared to baselines that optimize task discovery and task engagement independently. Furthermore, we show that our policy can zero-shot plan on out-of-distribution settings by training only on an outdoor map then transferring without finetuning to an indoor map with more agents, tasks, and capabilities than seen during training. 

---
# TUSI: A Citation-Grounded Graph-Based Benchmark 

**Authors**: Marzieh Karimi, Mehran Safayani, Alireza Basiri  

**Link**: [PDF](https://arxiv.org/pdf/2611.21823)  

**Abstract**: Evaluating retrieval-augmented generation (RAG) systems in Persian law remains challenging because benchmarks lack realism, grounding, and coverage of legal reasoning. Existing resources are either synthetically generated from statutory texts, lacking natural information needs, or derived from informal legal question--answer sources without explicit linkage to authoritative legal documents, making them unsuitable for rigorous evaluation of grounded legal reasoning. To address this gap, we introduce TUSI-Bench, a structured Persian legal benchmark designed for the evaluation of RAG-based systems. TUSI-Bench is built upon a multi-layered legal knowledge framework containing 7,261 legal articles and integrating codified statutes, legal commentaries, doctrinal writings, and Supreme Court unification rulings. We also construct a legal knowledge graph linking articles and concepts for structure-aware retrieval. Based on this foundation, TUSI-Bench provides two complementary evaluation tasks: (i) TUSI-QA, an open-ended question answering benchmark containing 1,999 questions that requires grounded responses with explicit legal citations, and (ii) TUSI-Bar, a multiple-choice benchmark containing approximately 500 four-option questions derived from Iranian Bar examination questions. Furthermore, we propose a graph-based retrieval baseline that leverages relational structure within the legal knowledge graph. Experimental results demonstrate that incorporating graph-based relations improves retrieval recall by up to 8 percentage points compared to standard baselines, highlighting the importance of structured legal representations for RAG systems. The proposed benchmark provides a standardized resource for evaluating grounded legal reasoning in Persian and enables future research on structured legal intelligence. 

---
# ARC-2: Continuous Compliance Through Delta Management 

**Authors**: Tord Johan Eide, Einar Holt  

**Link**: [PDF](https://arxiv.org/pdf/2611.21811)  

**Abstract**: ARC 1 introduced a three-layer governance architecture for autonomous systems comprising model safety validation, cognitive certification (CRAB), and operational authorization (LEA). However, it did not address how compliance should be maintained after deployment.
This paper introduces Delta Management, a continuous governance framework that measures deviations between a system's current operational state and its certified baseline, defined as a region rather than a fixed point. Compliance is assessed through continuous monitoring instead of static certification.
Four extensions are proposed. First, CRAB evaluation incorporates the worst verified behavioral deviations, with isolated outliers quarantined pending confirmation to prevent unjustified restrictions. Second, a layered trigger taxonomy distinguishes hard safety violations, critical single-dimension failures, correlated multi-system degradation, and risk-weighted composite indicators, each triggering proportionate responses. Third, a graduated safe transition protocol separates withdrawal of operational authorization from physical shutdown, consistent with established safety standards and EU AI Act requirements. Fourth, environmental delta is introduced as a distinct baseline validity condition: environmental drift beyond the certified Operational Design Domain (ODD) triggers recertification or ODD revision rather than automatic authority contraction.
Together, these extensions transform ARC from a deployment-time certification architecture into a continuous framework for monitoring, evaluating, and enforcing compliance throughout the operational lifecycle of autonomous systems. The framework addresses the continuous compliance obligations established by EU AI Act Article 72(2). 

---
# Evaluating the Evaluation of Music Generation Models with Generative Psychometrics 

**Authors**: João Carlos Santana Silveira, Caio Von Rondow Morais, Frederico Pedrosa, Lucas N. Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2611.21809)  

**Abstract**: Subjective user studies are one of the major methods for evaluating music generation models. However, subjective evaluation protocols vary widely and are rarely specified in enough detail to be reproduced, hindering comparison across studies. In this paper, we map subjective evaluation methods in ISMIR music generation papers from 2020-2025 and apply generative psychometrics to assess the stability of their items. We extract 284 items from the papers' methodological descriptions, using an LLM and manual validation to standardize them. We represent these items as contextual embeddings and apply Unique Variable Analysis to remove redundant, locally dependent items and Bootstrap Exploratory Graph Analysis to estimate their latent dimensions. Results show the network converges onto a taxonomy of six stable quality dimensions: Rhythm, Melody, Audio, Composition, Structure, and Lyrics, pointing to a unified methodology for conducting user studies to evaluate music generation models. 

---
# SimAI: Mechanistic User Simulation through Psychological Network Models 

**Authors**: Cheng Li, Adhyyan Narang, Jiayi Yuan, John F. Buss, Jina Suh, Devon Sandel-Fernandez, Yulia Tsvetkov, Adam M. Kuczynski, Tim Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2611.21808)  

**Abstract**: Realistic user simulation enables the study of human interactions when direct experimentation is costly or potentially harmful. However, existing simulators largely overlook the psychological mechanisms that drive user behavior. We introduce SimAI, a framework for mechanistic user simulation that models how psychological factors interact to shape user responses. To overcome the challenge of supervising open-ended behavior, we propose a graph reconstruction objective that provides graph-level rewards based on how well simulated conversations reflect the underlying psychological relationships. To support this approach, we construct PSYCHE, a dataset of 18,576 profile-network pairs derived from 516 psychotherapy cases. Automatic evaluations show that SimAI achieves greater simulation and behavioral diversity than existing patient simulators while maintaining strong profile alignment. In an interactive study with 30 mental-health professionals and trainees, SimAI receives the highest average ratings for conversational authenticity, realistic help-seeking behavior, and clinical training utility. Beyond simulation quality, SimAI enables realistic clinical training and uncovers safety failures in prolonged AI interactions with vulnerable users. Code is available at this https URL. 

---
# Can Coding Agents Write and Transpile Unit Tests 

**Authors**: Gil Pasternak, AJ Langley, Tom Yotam, Ellie Kitanidis  

**Link**: [PDF](https://arxiv.org/pdf/2611.21797)  

**Abstract**: Coding assistants built on Large Language Models (LLMs) have become a staple of modern software development, increasingly trusted to execute workflows and modernize code end-to-end. Much of this trust rests on passing tests, which are often written by the coding agents themselves. In this work, we examine the ability of these agents to both write reliable tests and translate them between programming languages. We benchmark three popular coding agents, each evaluated with a native frontier model and high reasoning effort. Verifying across a broad suite of metrics, we find that coding agents far exceed human written suites on traditional measures of coverage. This advantage, however, is uneven: agent test quality degrades sharply with project size, and much of the advantage lies in a greater number of written tests. We also show that agentic test translation---while seemingly strong in aggregate---remains inconsistent and unreliable. 

---
# Revisiting the Survival of Agent-Attributed Code: How Measurement Choices Decide the Verdict 

**Authors**: Musfiqur Rahman, Mahi Begoug, Emad Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2611.21784)  

**Abstract**: Studies of how long code from AI coding agents survives after it is merged reach conflicting verdicts: agent code is reported to be retained longer than human code, about as long, or modified sooner. Each study makes defensible measurement choices, but how much these choices decide the verdict is unknown. We revisit the survival of agent-attributed code in our earlier study: 854,139 lines that 2,802 merged pull requests (PRs) added to 215 repositories of the AIDev dataset. We vary nine measurement choices in five research questions and build a specification curve of 88 survival analyses, tested under three variance estimators, several observation windows, and permuted labels. For code from human-supervised agents, mostly GitHub Copilot, the choices change the size of the effect but not its direction. For code from the fully-autonomous agent Devin, they decide the verdict. Across repositories, 15 of 22 specifications report it modified sooner when lines are treated as independent, but only 3 when standard errors are clustered by repository, no more than permuted labels produce. Within repositories, no specification detects a difference. LLM intent classifiers agree with human labels about as well as two humans agree with each other, whereas keyword rules do not (kappa = 0.69 to 0.72 against 0.38), and the two reach different verdicts on why agent code is modified. Humans make most modifications to agent code under every unit and window. Re-run on our data, five published designs each change their result with a single choice. Whether agent code survives longer than human code is therefore decided as much by the measurement as by the code. We derive an operationalization checklist for post-merge studies. 

---
# Beyond Reconstruction: A Unified Framework for Trustworthy and Explainable EEG Artifact Removal Evaluation 

**Authors**: Phat Lam, Hung Nguyen, Lam Pham, Huy Phan, Dat Tran, Ian McLoughlin, Roozbeh Razavi-Far, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.21781)  

**Abstract**: Deep-learning methods have achieved strong performance in EEG artifact removal, but evaluation remains dominated by reconstruction metrics such as error, correlation, and signal-to-noise ratio. Although these metrics assess similarity to clean EEG, they do not establish whether models effectively suppress artifacts, preserve neural activity, or provide reliable explanations. To address these limitations, we propose TriX-EEG, a three-level framework for trustworthy and explainable EEG artifact removal. Level 1, Signal Validity (SV), evaluates reconstruction quality and neural band-power preservation. Level 2, Removal Faithfulness (RF), assesses the removed component through artifact error and residual artifact/neural leakage. Level 3, Explanation Reliability (ER), examines model behavior through temporal alignment, causal relevance, and stability. We evaluate TriX-EEG using nine published neural denoisers across three public benchmarks derived from EEGDenoiseNet and the PhysioNet Motion Artifact Contaminated EEG Dataset, covering EOG, EMG, and motion artifacts. Results show that reconstruction metrics alone are insufficient for trustworthy denoiser assessment: RF complements it by revealing whether errors stem from residual artifact or suppressed neural activity, while ER captures distinct aspects of explanation behavior, supporting joint evaluation of three aspects within a unified framework rather than reliance on any single factor. External validation on an independent EEG/EOG dataset and the BCI Competition IV-2a motor-imagery dataset further demonstrates broadly consistent rankings under dataset shift. Higher-ranked denoisers generally preserve more task-relevant information or cause less downstream performance degradation. Overall, TriX-EEG provides a comprehensive basis for EEG denoiser assessment, encouraging models that are not only accurate but also faithful and explainable. 

---
# LLMs as Validators for Cross-Chain Bridges: Evaluating Source-Level and Transaction-Level Attack Detection on Solana 

**Authors**: Yasir Rahman, Ishraq Samiha, Shahnewaz Karim Sakib, Anindya Bijoy Das  

**Link**: [PDF](https://arxiv.org/pdf/2611.21777)  

**Abstract**: Cross-chain bridges have suffered substantial asset losses due to vulnerabilities in their on-chain programs. On Solana, missing validation checks can enable exploits even when the underlying blockchain operates correctly. The ability of large language models (LLMs) to diagnose these flaws from source code or transaction records, rather than merely raise alarms, remains insufficiently understood. We implement a native Rust lock-and-mint bridge on a local Solana validator with four exploitable variants: two model flaws from the Cashio and Wormhole incidents, while the other two represent vulnerability classes documented in Solana security audits. Paired vulnerable and secure programs, together with successful attacks, benign sessions, and blocked attempts, allow us to distinguish correct diagnoses from unsupported alarms. We evaluate six configurations of five open-weight models across 16,800 responses, compare source inputs with and without comments, and grade 7,200 answers on vulnerable inputs using two blinded LLM graders. The results reveal substantial differences across conditions: without comments, models diagnose 24% of vulnerable source inputs while frequently flagging the secure bridge. Transaction diagnosis is high for the missing-signer scenario but averages 6% across the other three scenarios. Original comments, some of which describe the flaw, increase source diagnosis to 60%. These findings highlight the importance of distinguishing correct vulnerability diagnoses from unsupported alarms in bridge security review. 

---
# Margin Dynamics for Large Language Model Alignment 

**Authors**: Xingzi Xu, Mehmet Saygin Seyfioglu, Karim Bouyarmane  

**Link**: [PDF](https://arxiv.org/pdf/2611.21766)  

**Abstract**: Practitioners choose among LLM alignment methods largely by trial and error. We provide a theoretical foundation: each pairwise method defines a vector field on the probability simplex that governs how the model redistributes mass between preferred and dispreferred completions. Projecting onto a single preferred/dispreferred pair reduces this to a one-dimensional margin ODE with a closed-form solution. These solutions reveal that DPO margins grow without bound (logarithmically as $O(\log t)$ in the gradient-flow limit) while IPO margins converge to a finite $1/(2\tau)$; that label noise induces a qualitative phase transition in DPO (unbounded $\to$ bounded) but affects IPO only quantitatively; and that gradient concentration under asymmetric weighting governs trainability at scale. We confirm all three predictions across five model families spanning four benchmarks. The closed-form solutions also enable principled method design: we prove that adding any label smoothing $\varepsilon > 0$ to SimPO creates a finite margin attractor at $\Delta^* = \beta^{-1}\log((1-\varepsilon)/\varepsilon)$, while $\varepsilon = 0$ (standard SimPO) has no attractor. We test this across three models (8B-9B) and confirm the prediction: the theory-derived configuration (TPO-S, $\varepsilon = 0.05$) is competitive with the strongest baselines, attaining the highest mean IFEval score among the compared methods on all three, validating that the predicted finite attractor is a usable operating point rather than proposing a new state-of-the-art loss. 

---
# From Top to Bottom: Quark Mass Hierarchies in Heterotic String Theory 

**Authors**: Yuri Chervonyi, Fabian Ruehle, Nenad Tomasev  

**Link**: [PDF](https://arxiv.org/pdf/2611.21751)  

**Abstract**: Standard Model fermion masses exhibit large hierarchies, spanning more than five orders of magnitude in the quark sector alone. In this paper, we study whether these hierarchies can be generated perturbatively at tree level in smooth heterotic Calabi-Yau compactifications, where the supergravity approximation remains under computational control. In such backgrounds, physical Yukawa couplings combine holomorphic superpotential integrals with matter field Kahler metrics. The former can be computed algebraically, while the latter require solving a coupled system of partial differential equations for the Ricci-flat metric, Hermitian Yang-Mills connections, and harmonic bundle-valued forms. Within this framework, we identify two new tree-level mechanisms by which complex structure moduli can generate fermion mass hierarchies and show analytically and numerically that both mechanisms are obstructed for previously studied tetra-quadric models. Motivated by these obstructions, we scan the existing databases of tens of thousands of complete intersection Calabi-Yau line bundle models and find a unique smooth quotient admitting a rank-three up-quark Yukawa matrix at tree level. Evaluating the physical couplings on this manifold and evolving them under the renormalization group, we obtain realistic mass hierarchies for the top, charm, up, and bottom quarks in the smooth supergravity regime. 

---
# Bookkeeping Is Not Judgement: EJBench, an Engineering Judgement Benchmark Built from a Conjugate Heat Transfer Design Campaign 

**Authors**: James Sadler, Rotimi Alabi, Kotub Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2611.21707)  

**Abstract**: We introduce EJBench, a benchmark of engineering judgement: whether a model can predict how a design behaves at new conditions from a few earlier results, and decide on it. Unlike engineering benchmarks that test whether an agent can operate simulation or CAD software, EJBench is software-agnostic and asks what an engineer would conclude without running a simulation. Its tasks come from a conjugate heat transfer design campaign for battery cold plates, in three families: matching unlabelled test results to their operating points, choosing which stored runs to reveal before signing off a design, and ranking new duties by peak cell temperature. Every answer is taken from a converged simulation, not from an LLM judge. The best of ten frontier models, GPT-6 Astra, scores 0.71, and models state almost the same confidence on tasks they get wrong as on tasks they get right. We also post-train Qwen3.5-9B with LoRA and reinforcement learning, rewarded by the share of answer criteria the simulation results confirm; on the held-out test split its score rises from 0.071 to 0.106. We release the train, validation and test splits, the evaluation harness and the leaderboard. 

---
# Disambiguating Moral Circuits and Moral Features in Language Models: Moral content, normative judgment or cue words? Evidence from 14 models 

**Authors**: Davide Liga, Liuwen Yu, Matteo Magnini  

**Link**: [PDF](https://arxiv.org/pdf/2611.21705)  

**Abstract**: Interpretability methods can locate the attention heads and sparse autoencoder (SAE) features that change the answer a language model gives to a moral prompt, e.g., whether "Stealing medicine for a dying child is morally" continues with "justified" or "wrong". Such a component could respond to moral content, to the judgment of conduct by any norm, or to the word "morally" in the prompt. We built 227 minimal pairs of short phrases, validated by five annotators, whose two versions differ in a moral standard, in another kind of norm (prudence, etiquette, aesthetics or evidence) or in a descriptive quality, and showed them to 14 open-weight models (Llama, Gemma and Qwen, 1B to 14B, base and instruction-tuned), with and without the word "morally". In every model, the heads that change the moral answer belong to an evaluation circuit the models use for every kind of normative judgment we tested: these heads carry the same share of the effect for moral as for other normative pairs (pooled ratio 1.00, 97.5% confidence interval [0.92, 1.07], with the word; 0.90, [0.84, 0.96], without it), and the models need them as much for both. The models need these heads also when nothing in the prompt asks for a judgment, and on 400 scenarios of the ETHICS benchmark, where removing them takes away 37 to 39% of the difference between acceptable and unacceptable scenarios, against 1 to 3% for ten random heads. The SAE features we located respond mostly to the word "morally" rather than to what the phrase is about. The models represent the category of a pair (the standard the annotators assigned to it): it can be read from their hidden states, and, in the eight models we tested, removing the direction that carries it weakens the moral verdict by a few percent, and more than the other verdicts in seven of them. We release the pairs, the ratings, the code, the trained SAE dictionaries and the result files. 

---
# Watermark Evidence under Open-Weight LLM Modification: Exact Bounds for Low-Rank Output Watermarks 

**Authors**: Prashant Shekhar, Dumindu Samaraweera, Caroline Howard  

**Link**: [PDF](https://arxiv.org/pdf/2611.21702)  

**Abstract**: Open-weight language models can continue generating after their released weights are modified, while a watermark verifier still scores outputs against the original base and marked models. We study how such modifications change the evidence available to an OpenStamp-style verifier. At each shared prefix, we represent the modification as a residual in the next-token logits. We derive an exact finite-vocabulary lower bound on the expected log-likelihood-ratio evidence under a residual-span constraint, along with a sharp span-only deviation bound. We also characterize the first-order evidence shift through a covariance term with an explicit second-order remainder, and determine the residual span needed to reach a target evidence level. For sequence-level detection, we derive an exact conditional exponential-moment bound and combine it with a supermartingale argument and a distribution-free order-statistic tolerance bound. This yields a high-confidence upper bound on miss probability after attack selection on calibration data and evaluation on independent certification continuations. In experiments with OpenStamp on Llama-2-7B and Mistral-7B-v0.3, INT8 quantization and 30% pruning retain test true-positive rates of 85.6-94.8%, while 500-step LoRA fine-tuning lowers them to 28.8% and 11.6%. The covariance prediction tracks the measured mean LoRA evidence drift on both models, with no remainder-bound violations across 160,000 test prefixes. Additionally, we observe zero false positives among 500 base-model continuations per model. Through these theoretical and empirical results, the current research provides a better understanding of watermark robustness to model-editing/tuning attacks. 

---
# Stop Early, Stay Simple: Evidence Acquisition for Power System Fault and Attack Diagnosis 

**Authors**: S M Zia Ur Rashid, Sazidul Islam Hira, Rupesh Sah, Suman Rath  

**Link**: [PDF](https://arxiv.org/pdf/2611.21695)  

**Abstract**: Diagnostic models for cyber-physical power systems usually inspect every telemetry source for every event, although some events are clear from one source and others need several. We ask when a learned policy for choosing that evidence is worth its added complexity. To answer this, we present ActiGrid, a benchmark framework for active evidence acquisition that pairs a classifier trained to tolerate missing modalities with a Double Deep Q-Network (DQN), which either requests another source or stops and diagnoses but never operates grid actuators. On the public TU Delft digital substation and MSU/ORNL power system attack benchmarks, we compare the DQN with fixed subsets and orders, calibrated entropy stopping, a greedy value of information rule, and tree ensembles trained per subset. The learned policy does adapt to what it observes, yet it is statistically indistinguishable from entropy stopping on TU Delft and Pareto dominated by simple controls on MSU/ORNL, while compact subset classifiers outperform it while reading fewer sources on both benchmarks. Entropy stopping run over those subset classifiers reaches the highest mean Macro-F1 on TU Delft (0.928 while reading 57% of the sources) but brings no saving over a static three source subset on MSU/ORNL, where no method exceeds 0.45 Macro-F1. Selective acquisition can pay off, but on these benchmarks it does not need a learned policy. 

---
# TIBECG: Temporal Information Bottleneck for Lead-Specific Demographic Privacy in ECG Waveform Release 

**Authors**: Tsai-Min Chen, Jun Seita  

**Link**: [PDF](https://arxiv.org/pdf/2611.21694)  

**Abstract**: Transmitting an electrocardiogram (ECG) waveform can expose demographic information beyond an intended diagnostic task. We evaluate TIBECG, an architectural temporal information bottleneck, on individual leads extracted from hospital twelve-lead recordings. A 2,302,337-parameter processor varies its latent length while retaining the same nine-score consistency objective, without age or sex labels in protector fitting. Across 6,386 recordings, ten folds, 12 leads and 21 lengths, retrospective selections retained at least 95% of each lead's clean macro-F1. Their weaker age/sex interval gain exceeded the best eligible conventional transformation on all leads, by a median 22.08 percentage points. The implemented sampling-rate comparison shows that direct downsampling usually suppresses demographic inference more strongly but at a larger loss of diagnostic utility, whereas TIBECG favors utility preservation. Selection stability also varies by lead. Lower latent Kullback-Leibler (KL) divergence does not reliably rank residual sex leakage at a fixed length. These findings support lead-specific evaluation of waveform processing, not validated consumer deployment or expert-readable reconstruction. When the waveform-input diagnostic classifier is hosted as part of a proprietary cloud service rather than deployed on the device, waveform transmission is required to obtain its scores. 

---
# AdaFlowLite: Scalable and Non-blocking Inference on Asynchronous Mobile Data 

**Authors**: Sicong Liu, Fengmin Wu, Yuan Gao, Bin Guo, Zimu Zhou, Hongkai Wen, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21691)  

**Abstract**: The rise of mobile devices equipped with numerous sensors, such as LiDAR and cameras, has driven the adoption of multi-modal deep intelligence for distributed sensing tasks, such as smart cabins and driving assistance. However, the arrival time of mobile sensory data vary due to modality size and network dynamics, which can lead to delays (if waiting for slow data) or accuracy decline (if inference proceeds without waiting). Moreover, the diversity and dynamic nature of mobile systems exacerbate this challenge. In response, we present a shift to \textit{opportunistic} inference for asynchronous distributed multi-modal data, enabling inference as soon as partial data arrives. While existing methods focus on optimizing modality consistency and complementarity, known as modal affinity, they lack a \textit{computational} approach to control this affinity in open-world mobile environments. %
\sysname pioneers the formulation of structured cross-modality affinity in mobile contexts using a hierarchical analysis-based normalized matrix. This approach accommodates the diversity and dynamics of modalities, generalizing across different types and numbers of inputs. Employing an multi-modal lightweight Swin Transformer (MMLST), \sysname facilitates real-time and flexible data imputation, adapting to various modalities and downstream tasks without retraining. Experiments show that \sysname significantly reduces inference latency by up to 80.4\% and enhances accuracy by up to 62.1\%, while achieving nearly a 50\% reduction in energy consumption, outperforming status quo approaches. Also, this method can enhance LLM performance to preprocess asynchronous data. 

---
# EgoLink: Egocentric Language-Vision Interactive Network Knowledge Challenge Summary 

**Authors**: Renhe Sun, Yueying Feng, Bohan Yu, Tong Niu, Yunqi Liu, Zitong Wang, Haolin He, Kexin Shi, Hualei Wang, Qiuqiang Kong, Chang Yao, Yuge Huang, Weiqiang Wang, Jingyuan Chen, Jian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21690)  

**Abstract**: The EgoLink Challenge at ACM Multimedia 2026 benchmarks egocentric multimodal intelligence linking first-person perception, social reasoning, and interactive action. Unlike benchmarks centered on visible objects and activities, EgoLink evaluates whether models can interpret emotional and social cues, resolve visually grounded references, and convert user requests into verified tool-mediated actions. Its two tracks cover social reasoning in egocentric video and interactive agents that complete service tasks through dialogue, tool use, and closed-loop execution. Together, they test whether perception, reasoning, and decision-making remain coupled under both offline question answering and state-changing interaction. We highlight a multi-stage Track 1 data pipeline for multiple-choice question (MCQ) generation, chain-of-thought generation, automatic quality grading, and video-contribution filtering. We summarize the task design, evaluation metrics, rules, leaderboards, and representative solutions. The benchmark combines objective answer scoring with process- and result-based evaluation, enabling analysis of both final correctness and execution quality. The best Track 1 system achieved 0.9263 accuracy, while the best Track 2 system achieved 0.844 joint success and 0.916 micro accuracy. Participant reports reveal a shared principle: strong systems make intermediate evidence explicit through candidate verification, option-robust reasoning, scenario-aware planning, tool canonicalization, and state-aware correction before answering or acting. These findings highlight the importance of process verification and evidence-grounded reasoning for reliable egocentric multimodal intelligence. 

---
# FCP: Formal Context Protocol for Verifying LLM-Generated Artifacts 

**Authors**: Shashank  

**Link**: [PDF](https://arxiv.org/pdf/2611.21679)  

**Abstract**: When a language model writes both an artifact and the tests meant to check it, a passing test is weak evidence: one misreading of the request yields a faulty artifact and a check that accepts it. We present the Formal Context Protocol (FCP), which checks AI-generated artifacts against the request itself. FCP separates conformance (does the artifact satisfy a formal reading of a requirement?), which model checkers and SMT solvers decide, from fidelity (does the reading mean what the prose means?), which has no decision procedure but is refutable by a concrete witness. Eight ways a proof can falsely certify a requirement each become a mechanical check that returns a witness, and what no machine can settle becomes concrete questions for the request's author. Adapters place hardware, text, agents and code on the same engines. On four hardware designs, trusting AI-written assertions certified 23% of violated requirements, and FCP certified none. On ten code tasks, code from three language models passed its own tests and violated the request every time. Trusting the tests certified all 33 violations, and FCP certified three. Agreement among models, readings or tests did not imply correctness. 

---
# Token Commitment as a Control Surface for Steering Masked Diffusion Generation 

**Authors**: Ambroise Aigueperse, Hugo Buurmeijer, Carmen Amo ALonso, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2611.21669)  

**Abstract**: Masked diffusion models repeatedly predict provisional tokens before a sampler decides which predictions become permanent. Existing inference-time steering methods primarily modify activations, logits, or denoising trajectories, while treating this commitment policy as a fixed decoding detail. We instead study token commitment as a diffusion-specific control surface and separate it into two decisions: which tokens to commit and how many tokens to commit at each step. For the first, we introduce counterfactual target-aware commitment, which ranks unresolved positions by a mixture of model confidence and leave-one-token-out target contribution. For the second, we introduce calibrated adaptive budgets driven by target progress and confidence. Across language and vision-language settings in the LLaDA family, multiple targets, two activation-steering methods, and an unsteered control, both mechanisms yield empirically monotonic target-quality trade-offs in every configuration we evaluate. Adaptive schedules also recover competitive operating points relative to scorer-free stochastic schedules matched in average denoising cost. Because both mechanisms require only a scalar score evaluated on a provisional completion, they apply unchanged when the target is an automatic proxy for task success rather than an output attribute: NLI entailment for dialogue summarization and CLIPScore for captioning, neither of which corresponds to a single reusable steering direction. We reproduce the behavior of both mechanisms on a second masked-diffusion family, Dream-7B, for one control target and one task-success target. These results characterize commitment--which provisional predictions become irreversible, and how rapidly--as an interpretable control surface for steering masked diffusion generation under the conditions we test. 

---
# Depth-Aware Activation Steering via Attention Residuals 

**Authors**: Shilong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.21655)  

**Abstract**: Activation steering is a simple, training-free way to modulate LLM behavior, yet existing methods typically inject vectors at a single layer and lack mechanisms for depth-aware propagation or internal verification. We combine two recent ideas - Attention Residuals (AttnRes) providing learnable cross-layer pathways, and Jacobian-lens (J-space) concept readout - into a lightweight depth-aware steering pipeline. On two small open models (Qwen3-0.6B and Qwen3.5-0.8B), distributing a persona vector according to AttnRes-style depth weights yields stronger internal alignment (deeper-layer concept cosine and rank, p<0.0001) and stronger behavioral shifts than single-layer injection. The effect is sign-specific: reverse vectors invert all six metrics, random directions degrade alignment (p<0.0001), and a short LoRA fine-tune erases the imprint. Injection rewires attention strictly downstream of the injection point (upstream JS = 0.00000). Three practical findings follow. First, a layer-wise geometric decomposition of cosine readouts: at the injection layer 123% of the increase is arithmetic, while 5-11 layers downstream 100% is dynamical. Second, an intensity window: effects are monotone over 0.3-0.7, then generation collapses (617 to 306 to 25 chars). Third, metric discipline: keyword-rule metrics fail LLM-judge validation (kappa <= 0.09), and two rule-based conclusions were retracted. Training the depth weights on a steering objective (200 steps, frozen backbone) raises final alignment 10x on held-out prompts. Cross-persona vectors retain 96-99% of the effect: specificity holds against random, not semantic, directions. We contribute a practical, low-overhead depth-aware steering method with a careful evaluation protocol. Code and data are released. 

---
# Evaluating Semantic Insensitivity in Financial Sentiment Analysis: Implications for Sustainable Financial Decision-Making 

**Authors**: Gabriela Cioboata, Teodor-Stefan Bildea  

**Link**: [PDF](https://arxiv.org/pdf/2611.21653)  

**Abstract**: Financial sentiment analysis is critical for automated trading, portfolio management, and risk assessment, heavily relying on models like FinBERT. Despite high benchmark accuracy, these models remain vulnerable to subtle adversarial text modifications, threatening the integrity of algorithmic decision-making. In this paper, we propose Agent-driven Semantic Inversion (ASI), a novel adversarial attack framework targeting financial sentiment models through domain-specific semantic manipulation. Utilizing an LLM-powered multi-agent pipeline, our approach identifies eligible financial news sentences and applies targeted lexical substitutions. We empirically evaluate ASI against FinBERT, one of the most widely deployed financial sentiment models. Our results reveal a striking vulnerability: the attack successfully deceived FinBERT in 90.85% of eligible cases, with the model frequently maintaining its original sentiment prediction despite the fundamental inversion of financial performance indicators. Through qualitative analysis and token-level investigation, we demonstrate that FinBERT critically over-relies on contextual keywords and syntactic patterns rather than true logical comprehension of financial semantics. Our findings highlight the urgent need for logic-aware, robust financial NLP models to ensure the reliability of automated decision-making systems. 

---
# Consistency Is Not Correction: Separating Confidence, Abstention, and Recovery in LLM Agents 

**Authors**: Khushal Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2611.21651)  

**Abstract**: A controller that resamples an answer, escalates to a larger model, or abstains is usually justified by a confidence signal. Three claims about such a signal do not follow from one another: it ranks answers (discrimination), it supports withholding some of them (selective prediction), and it changes an answer for the better (correction utility). We give an exact decomposition of the third into gated recoveries minus gated harms, which bounds whether a comparison can succeed before it is run: a gate that only replaces a proposal can act only where the base policy was already wrong, so the initial error count caps the discordant pairs any correct gate can produce. When that cap falls below the count a paired test needs, the experiment cannot return a positive result for a working mechanism, nor distinguish one from a gate that never fired. We measure all three targets on two model scales and find the bound binds. In a small-model suite a perturbation-agreement verifier reaches within-policy AUROC 0.990 (95% interval [0.95, 1.00]) while every episode outcome matches single-sample decoding at 4.32 times the call count. In a controlled 840-episode study no verifier replaced the proposal on any of the 60 GSM8K decisions, so that comparison tested nothing; the entire GSM8K gain across six controllers is one task, and it is the task whose identical temperature-zero repeats disagreed, making it the one answer the service does not reproduce. On the diagnostic set the gate could reach four errors while the test needed six, so even a perfect gate had no significant result available. We report two checks that would have caught this before any budget was spent, and show that tie handling in rankings, tied selection cutoffs, and unequal output limits each move a headline number in the same dataset. The binding constraint on this literature is measurement resolution, not the choice of signal. 

---
# From Preference Data to Personalization: Tracing Sycophancy in Large Language Models 

**Authors**: Himaghna Bhattacharjee, Yasha Sheynin, Shivam Singhal, Arjun Subramonian, Ankur Samanta, Julian Michael, Smitha Milli, Maximilian Nickel, Claudia Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.21650)  

**Abstract**: Sycophancy in large language models is often measured as agreement with the user, yet agreement can be appropriate when the user's claim is valid, their preference is genuine, or their experience warrants acknowledgment. We study sycophancy as inappropriate agreement: validating a claim, preference, or self-assessment when the context calls for correction, qualification, or resistance. We introduce a validity-aware sycophancy grader and SycoDrift, a persona-targeted evaluation set of 3,298 validity-annotated probes with controlled annotations for user openness and stakes. We use this framework to analyze three sources of agreement pressure: human preference data, automated judge specification, and in-context personalization. First, preference for validating responses in human datasets (Chatbot Arena, HH-RLHF, Community Alignment) is weak and concentrated where agreement is often appropriate. Second, in a selected-value, single-model experiment, emphasized judge rubrics produce consistent but modest increases relative to paired suppressed rubrics, demonstrating a plausible mechanism for training-time amplification. Third, in off-policy evaluations across 14 models, inference-time personalization increases inappropriate agreement, with the largest effect from matched user profiles placed directly before the query. 

---
# Literary Language Mashup: Curating Fictions with Large Language Models 

**Authors**: Manzanarez Gerardo Aleman, Monroy Raul, Garcia Flores Jorge, Calvo Hiram  

**Link**: [PDF](https://arxiv.org/pdf/2611.21649)  

**Abstract**: In this paper, we propose an alternative approach to human evaluation of literary fictions by employing LLMs themselves as evaluators within the GrAImes framework. We apply this methodology to assess human-written and AI-generated microfictions in Spanish, to five PhD professors in literature and sixteen literary enthusiasts, and to short stories in both Spanish and English. By comparing the evaluations performed by LLMs with those of human judges, we examine the degree of alignment and divergence between both perspectives, thereby assessing the feasibility of LLMs as auxiliary literary evaluators. Our analysis focuses on the alignment of responses from LLMs with those of human evaluators, providing insights into the potential of LLMs in literary assessment. The conducted experiments reveal that while LLMs cannot be regarded as substitutes for human judges in the evaluation of literary microfictions and short stories, with a Krippendorff's alpha reliability coefficient less than 0.66, they can serve as a valuable tool that offers an initial perspective on the editorial quality of the texts in question. Overall, this study contributes to the ongoing discourse on the role of artificial intelligence in literature, underlining both its methodological constraints and its potential as a complementary resource for literary evaluation. 

---
# Can an LLM dream? Testing the overfitted brain hypothesis on an agent with text memory 

**Authors**: Eric Zou  

**Link**: [PDF](https://arxiv.org/pdf/2611.21648)  

**Abstract**: Hoel (2021) proposed that dreams distort replayed experience so that the brain does not overfit to recent events. We test this hypothesis on an LLM agent (Claude Opus 5.5) that stores what it learns in a text memory and rewrites that memory each night from the day's conversation transcript. The agent goes through ten simulated days of conversation built from LongMemEval. We corrupt 5, 10, 20 or 40 percent of the transcript's turns before each nightly rewrite in one of two ways. Detail corruption changes one or two details in a turn to plausible wrong ones, or replaces the turn with one by the same speaker from an earlier day. Random-text corruption replaces half of the selected turns with a passage of Moby-Dick and inserts Moby-Dick words into the other half. A size limit on the memory, rising from 19,500 to 117,000 tokens over ten nights, set from human free-recall data, forces it to forget. We learned that corruption did not help the agent apply what it learned in earlier conversations to new tasks in any of 30 runs (ten conditions, three different conversation histories). Replacing details in 40 percent of turns lowered recall from 76 to 60 percent. Random text had no consistent effect on recall, with a paired difference of 0 $\pm$ 2 points at 40 percent, because the model recognized it as corrupted and left it out of memory. Replaced details were stored as fact even when another turn held the correct value, and the agent answered 40 percent of those questions correctly, compared with 75 percent without corruption. These results suggest that a pretrained model can separate corrupted text from real events, so a fair test needs a learner that cannot make this separation as easily, such as a model fine-tuned on the corrupted days or an agent whose days take place in an invented world. 

---
# RAW: Refusal Aversion in the Wild, A Causal Measurement Method for Deployed LLMs 

**Authors**: Pavel Kireyev  

**Link**: [PDF](https://arxiv.org/pdf/2611.21641)  

**Abstract**: Deployed LLMs field hundreds of millions of conversations a day and refuse a non-trivial fraction of user requests, yet the user-side cost of refusal, whether users disengage and by how much, is unmeasured on any production model: a blind spot for the consumer-facing platforms built on them. A/B-testing refusal is ethically and operationally constrained on frontier-lab products and unavailable to third-party platforms deploying these models through a fixed API. We introduce RAW, a method that uses LLM stochasticity as a natural experiment: at near-identical prompts in any existing log, the same model sometimes refuses and sometimes does not, and that variation identifies the causal user-side effect of refusal. Applying the method to 15 deployed LLMs in LMSYS-Chat-1M and WildChat, with user-clustered inference, we find that for about half of the models we can measure precisely, a refusal costs 3 to 17 pp of subsequent re-engagement; the rest show no distinguishable effect, among them one deployment whose positive point estimate we trace to persistent users retrying the refused request. Our contribution is the measurement method, which runs on any deployed model from logs alone for about \$0.30 per 1,000 conversations. We release labels, moderation scores, prompt embeddings, and pipeline code. 

---
# Can AI Help Explain AI? Effects of Multimodal LLM-Generated Explanations on Perceived Understanding and Perceived Trust in XAI Visualizations 

**Authors**: Mario Truss, Marc Schmitt, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2611.21638)  

**Abstract**: Explainable artificial intelligence (XAI) aims to make machine learning models understandable, yet many XAI visualizations remain difficult for end-users to interpret. Here, we evaluate whether augmenting XAI outputs with textual explanations from multimodal LLMs improves perceived understanding and perceived trust. We conducted a pre-registered within-subject experiment ($N = 128$) across seven decision scenarios and five XAI methods, with and without LLM generated explanations. The explanations improved perceived understanding and perceived trust by 0.35 and 0.26 points on five-point scales (both $p < .001$). The gain appeared in all seven scenarios, did not differ between local and global explanations, and was largest for participants with the least prior XAI exposure. A mediation analysis indicates an indirect effect on perceived trust through perceived understanding. These results suggest that multimodal LLMs can provide a useful interpretive layer for XAI beyond expert audiences. 

---
# Measuring Human-AI Alignment as an Early Indicator for Effective Collaboration in Hybrid SE Teams 

**Authors**: Andreas Metzger  

**Link**: [PDF](https://arxiv.org/pdf/2611.21633)  

**Abstract**: As AI agents increasingly integrate into software engineering (SE) processes, ensuring effective collaboration in these hybrid teams becomes a critical challenge. While existing metrics focus on observing dynamic run-time performance, there is a lack of systematic, static measurements for early human-AI alignment -- the degree to which an AI's actions and granularity map to human expectations. This position paper proposes a research agenda to develop an SE-specific alignment metric suite drawing principles from software architecture, organizational psychology, and human-computer interaction. Furthermore, it outlines a validation methodology using concepts and techniques from business process monitoring (BPM) and introduces static assessment tools leveraging declarative BPM languages to ensure agents adhere to specific project constraints. 

---
# Fragmented Markers, Mixed Results: A Systematic Review of AI Coding Assistants and Developer Productivity 

**Authors**: Annemarie Wittig, Alina Mailach, Norbert Siegmund  

**Link**: [PDF](https://arxiv.org/pdf/2611.21632)  

**Abstract**: AI coding assistants are increasingly embedded in software engineering practice, yet the literature still offers no cumulative, easily interpretable picture of their effects on software developer productivity. The central problem is not a lack of studies, but a lack of comparability: prior work operationalizes productivity through heterogeneous markers, settings, and reporting styles, making cross-study conclusions difficult. To address this, we conduct a staged systematic literature review. In the first stage, we synthesize studies that explicitly examine developer productivity. In the second, we extend the scope to productivity-related developer outcomes using SPACE as a theory-informed lens, capturing human-centered evidence that explicit productivity studies often omit. Across both stages, we analyze how productivity is operationalized, which confounding factors are discussed, and how reported effects are distributed.
We identify 187 unique productivity and productivity-related markers, while individual studies typically only use five, revealing a broad but highly fragmented evidence base. Explicit productivity research is dominated by time-, output-, and artifact-oriented proxies, whereas the broader SPACE-aligned corpus makes satisfaction, psychological aspects, and well-being substantially more visible. Reported effects are often positive, but this pattern is stronger in perceived and tendency-based findings than in objective or statistically evaluated ones. Overall, the literature suggests that AI coding assistants can support productivity, but current evidence does not yet support simple, context-independent conclusions. The key contribution of this review is therefore a structured basis for making productivity evidence more interpretable, multidimensional, and comparable across studies. 

---
# Marimba: A System One Model for Speech 

**Authors**: Solano Amadori  

**Link**: [PDF](https://arxiv.org/pdf/2611.18330)  

**Abstract**: Speech applications require decisions about term presence, turn completion, and language within an audio window. We present Marimba, an interface for typed speech decisions built on frozen Qwen3-ASR 0.6B features. Lexical questions use zero-shot scoring of ASR transcripts; non-lexical questions use a question-conditioned head and a separate layer-18 turn-end readout. Language identification uses the ASR model's own language scores. On private conversational test sets, decoder AUROC is 0.80-0.92; on public read speech it is 0.92-0.97. Speech Commands four-way accuracy is 0.93; binary lexical ECE is at most 0.08. Personal-information detection, which passes the greedy transcript to a private, curated PII pipeline, reaches item-level F1 0.904 on synthetic Brazilian Portuguese speech (1.000 on reference transcripts). Layer-12 speaker readouts detect speaker change with AUROC 0.966-0.980 on AMI and VoxConverse, above an ECAPA baseline, and count speakers (0, 1, 2, or 3+) more accurately than pyannote 3.1. The turn-end readout reaches AUROC 0.930 versus Smart Turn v3.2 at 0.982 on Smart Turn's own test set. On a subset of ICSI meetings labeled by next speaker, AUROC is 0.716 versus 0.712; the paired confidence interval includes zero. 

---
# Continual-ARC: Measuring Skill Acquisition over a Working Lifetime 

**Authors**: Daniel Lenton, Yusha Arif, Haris Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2611.14505)  

**Abstract**: Agent harnesses have emerged as a promising mechanism for continual learning, alongside test-time parametric learning. However, the current benchmarks have several limitations: they generally focus on a single applied domain, serve each instance only once, and assume a single family of learner, so memory-based and weight-based systems are rarely compared on the same scale. We propose Continual-ARC, a lifelong stream built from the three ARC-AGI editions, whose tasks need no domain knowledge. The same tasks return after dormant gaps with fresh instances, while new tasks keep arriving. On each instance, the learner may pay for demonstrations, submit answers, or act inside the harness, with retries permitted on incorrect submissions, which are paid for. Each instance is scored based on the share of this feedback budget consumed. Since a learner may keep anything between instances, memory-based and weight-based systems are scored on the same scale. After the stream, the learner is frozen and tested on fresh instances of the tasks it met, which measures memorization, and on unseen tasks, which measures generalization. We run an agent harness with frozen weights, with and without persistence, on Qwen3.5-9B and GLM-5.3 Flash, comparing them with two prompting floors, a program-induction method and Qwen3.5-9B's weights trained to imitate GLM-5.3 Flash. Persistence in the harness generally does not improve memorization unless provided with explicit expert guidance. Most harness-based methods still learn less than a floor that simply caches each task's demonstrations, across all models we tested. Harness-based methods gain little or nothing on unseen tasks, whereas weight-based learning shows substantial gains. This opens the door to future weight-based continual learning that could teach models to drive harnesses in new domains. 

---
# Improved Convergence Rates for Stochastic Multi-Gradient Descent which Close the Gap: A Proof by AI 

**Authors**: Lisha Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.18174)  

**Abstract**: For smooth nonconvex stochastic multi-objective optimization, stochastic multi-gradient descent (SMG) computes an approximate steepest common descent direction of the objectives from stochastic gradients. Under standard assumptions, this note establishes a new convergence rate for SMG in terms of the squared Pareto-stationarity (PS) measure. With a constant stepsize and linearly growing mini-batches, the expected squared empirical PS measure at the algorithm's output is $\tilde{O}(T^{-1})$ after $T$ iterations. This improves on the $\tilde{O}(T^{-1/4})$ bound obtained by Chen et al. (JMLR, 2024) for linearly growing batches, without requiring bounded gradients. Here, $\tilde{O}(\cdot)$ suppresses logarithmic factors. The key to the rate improvement is to exploit the Lipschitz continuity of the PS measure, defined by the norm of the multi-gradient descent algorithm (MGDA) direction, rather than the $(1/2)$-Hölder continuity of the MGDA direction used by Chen et al. (2024). For variants of MoCo and MoCo+ with exact MGDA updates and constant batch sizes, the analysis gives expected squared empirical PS rates of $O(T^{-1/2})$ and $\tilde{O}(T^{-2/3})$, respectively. These bounds match the corresponding single-objective momentum rates up to logarithmic factors. The proof was accidentally discovered while the author was preparing homework for a graduate course: ChatGPT 5.4 Thinking Extended generated the initial proof strategy for the main SMG result in response to an author-written homework-solution prompt; the author then verified and reorganized the resulting argument. The appendices document the prompt, summarize the student submissions with LLM disclosures, and present additional proofs and results on momentum-based stochastic MGDA. 

---
# Analyzing Learned Convnet Features with Dirichlet Process Gaussian Mixture Models 

**Authors**: David Malmgren-Hansen, Allan Aasbjerg Nielsen, Rasmus Engholm  

**Link**: [PDF](https://arxiv.org/pdf/1702.07189)  

**Abstract**: Convolutional Neural Networks (Convnets) have achieved good results in a range of computer vision tasks the recent years. Though given a lot of attention, visualizing the learned representations to interpret Convnets, still remains a challenging task. The high dimensionality of internal representations and the high abstractions of deep layers are the main challenges when visualizing Convnet functionality. We present in this paper a technique based on clustering internal Convnet representations with a Dirichlet Process Gaussian Mixture Model, for visualization of learned representations in Convnets. Our method copes with the high dimensionality of a Convnet by clustering representations across all nodes of each layer. We will discuss how this application is useful when considering transfer learning, i.e.\ transferring a model trained on one dataset to solve a task on a different one. 

---
