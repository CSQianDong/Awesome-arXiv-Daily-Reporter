# Beyond the Transcript: Detecting Covert Co ordination in Latent Multi-Agent Communication 

**Authors**: Ramneet Kaur, Pradyumna Chari, Ramesh Raskar, Jugad Singh, Sumit Kumar Jha, Anirban Roy  

**Link**: [PDF](https://arxiv.org/pdf/2608.19161)  

**Abstract**: Language-model agents can communicate through continuous hidden states that are invisible in public transcripts, creating opportunities for covert harmful coordination. We introduce Verifiable Latent Alignments (VLA), an activation-aware framework for monitoring and steering these private communication channels. For every monitored decision, VLA links the private latent-state record and channel status to the resulting public action using a shared event identifier, enabling matched causal analysis. Our first contribution is a neutral-only three-layer monitor combining representation anomaly detection, counterfactual action-distribution influence, and sparse-autoencoder interpretation support. Our second contribution is a steerability framework spanning black-box behavioral instructions and white-box matched-neutral counterfactuals. Our third contribution is an evaluation on a controlled multi-agent auction benchmark covering homogeneous and heterogeneous model pairs, many-agent scalability, and intervention effectiveness. The sequential monitor achieves mean area under the receiver operating characteristic curve (AUROC) of 0.993 for homogeneous agents and 0.854 for heterogeneous pairs when text- and latent-collusion rows are pooled as positives. In Qwen3-0.6B auctions with 25-100 bidders, monitoring requires only a small normalized load relative to all possible directed pairs, while full white-box steering achieves 100% bid-distribution recovery and reduces collusive low-bid behavior by 47.3 percentage points. Because full white-box steering replays the matched neutral counterfactual, its exact recovery is a sanity check by construction. Overall, the controlled study shows that the evaluated private channel attacks can be monitored without training the primary monitor on attack examples and mitigated when matched counterfactual access is available. 

---
# Grouping the Stochastic Machine: Precision, Not Capability, as the Frontier Metric for AI Systems 

**Authors**: George Andrikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2608.19140)  

**Abstract**: Frontier language models are compared, marketed, and benchmarked on capability -- what their best or average output can achieve. I argue this measures the wrong axis. The models have saturated accuracy: their mean output lands on the target. What now separates one system from another in practice is precision: how tightly concentrated their outputs are around that target across repeated, identical requests. Borrowing the marksman's distinction, capability is where the average shot lands; reliability is the size of the group. I make three claims. First, precision, not capability, is the frontier differentiator between systems, and benchmark culture systematically fails to measure it, reporting central tendency rather than spread. Second, precision is measurable, cheaply and without circularity, by running a fixed suite of deterministically scored tasks many times at fixed temperature and computing the per-task consistency of outcomes -- no model-in-the-loop grader required. Third, the measurement is not merely descriptive but decision-guiding: it separates consistent failures (a tight group off-centre, correctable by the operating discipline of Paper 1 -- a sight adjustment) from scattered failures (a wide group, correctable only by changing the model or its sampling -- a rifle problem). I define a grouping metric, specify a harness, and show how tracking a human-AI pair's grouping over time yields the compounding signal that Paper 1's field study requires. A first real run, since replicated, illustrates both the method and its most important limit: one measured gap was closed completely by a single rule (0/5 -> 5/5), while a suite of tasks authored from the rules themselves found no value, because a frontier model already embodies explicit good practice -- establishing that a discipline's worth is found by measurement on real work, not constructed from its own rulebook. 

---
# Tuning the Stochastic Machine: A Systems Engineer's Operating Model for Human-AI Engineering 

**Authors**: George Andrikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2608.19125)  

**Abstract**: When an expert corrects an LLM assistant's error, the correction usually dies with the session, and the error class returns. I argue this is an operations problem, not a tooling problem: mechanisms for persisting corrections exist and are shipping, but the discipline for governing them -- versioning with provenance, recurrence monitoring, counter-metrics, retirement of stale rules -- does not. Writing as a systems engineer of thirty years, I map the LLM stack onto the machines my profession already operates (frozen silicon, firmware, loadable modules, persistent configuration, volatile memory), identify where the mapping fails (stochastic generation, configuration that binds only probabilistically, no general-purpose retirement (verification) stage by default), and derive from the failures a seven-principle operating discipline with an error loop at its core. Three cases from my own practice illustrate the mechanism, among them a control that silently became the exact harm it was built to prevent. I close with the measurement framework this view implies and the lab study required to test it. 

---
# Robust Risk Under Evolving Uncertainty: A Wasserstein Counterpart of the Entropic Value-at-Risk 

**Authors**: Deep Kumar Ganguly, Jan Křetínský  

**Link**: [PDF](https://arxiv.org/pdf/2608.19073)  

**Abstract**: An agent still learning its environment should be cautious while ignorant and bold once confident. The entropic value-at-risk captures this through a robust-optimization identity---a confidence level fixes the radius of a relative-entropy ball of alternative models---but that ball cannot reach catastrophes the nominal deems impossible, precisely what a safe agent must hedge. We instead use an optimal-transport ball and study the coherent risk measure it induces, the Wasserstein entropic value-at-risk. It has a variational dual mirroring the entropic formula (an inverse temperature becomes a transport price), occupies a definite place in the risk hierarchy, and provably accounts for the reachable catastrophes the entropic measure ignores; we verify both dualities numerically. Driving the transport radius by belief entropy then yields a closed-form robust dynamic-programming operator whose caution contracts as the belief sharpens, with a certified safety sandwich and a sharp safety switch. 

---
# What is Missing from AI Post-Training AI: An Empirical Analysis 

**Authors**: Joy Jia Yin Lim, Xin Huang, Hao Peng, Yaxi Lu, Xin Cong, Zhong Zhang, Maosong Sun, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.19072)  

**Abstract**: Large language model (LLM) agents can now post-train an LLM end-to-end. They can write code, launch training, evaluate checkpoints, and improve downstream performance, raising the prospect of AI-for-AI. We argue that this picture conflates two distinct capabilities: execution-level capability, iterating within a selected training strategy; and strategy-level capability, revising the high-level judgment as experimental evidence accumulates. Analyzing a large corpus of publicly released post-training trajectories, we find that across different tasks, the agent's training strategy is locked in at the very beginning, and the entire remaining budget is spent on local adjustments within the selected strategy. We then examine three natural explanations--missing experience, missing guidance, and insufficient reasoning--with escalating interventions. Extensive experiments show that (1) an experience-driven scaffold improves execution across the board (+12.6 points on GSM8K and +40.8 on HumanEval) but leaves the strategy static; (2) human guidance effectively redirects the initial strategy, yet the agent falls back into local adjustment loops once training starts; and (3) additional inference compute pays off on easier tasks but yields almost no gain on the hardest one. In conclusion, what agents lack is neither experience, guidance, nor reasoning compute, but a mechanism for spontaneously reevaluating their strategy during execution. 

---
# Eureka: Task-Conditioned Meta-Agent Orchestration for Scientific Discovery 

**Authors**: Alizer Wong, Heng Cui, Yi Tan, Xiongchao Zhan, Liang Lin, Yuxiang Guo, Zhaorong Dai, Zixin Zeng, Wenyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.19047)  

**Abstract**: We present Eureka, a task-conditioned Meta-Agent architecture that compiles long-horizon tasks into dynamic obligation graphs with explicit acceptance semantics. During execution, Eureka forms Macro-Agents with specialized state, memory, operators, tools, verifiers, and local topology via receding-horizon planning, architecture promotion, and minimal-sufficient compilation. When bottlenecks recur, cost-benefit-gated evolution updates the local architecture under constraints. Theoretically, we establish results on regret, planning invalidation, amortization, subtree interfaces, serializability, and verification. Experimentally, Eureka completes 170/170 recursive tasks and generates 3,948 certificates with no false acceptances. Active context compresses median input from 9,490 to 4,005 tokens; incremental processing avoids 65.38% recomputation across 12,000 tasks; 16,000 concurrent executions serialize consistently. The same Meta-Agent instantiates a Theory-Discovery Agent and a Math/Conjecture Agent. The former yields structural results in quantum-process and spacetime theory. The latter identifies bottlenecks in Riemann Hypothesis research and advances a positivity certificate for Suzuki's localized Weil quadratic form to 0 < a <= 69/200 = 0.345, reaching ~99.55% of (log 2)/2. These results suggest that scientific-agent capability depends not only on the base model but on whether an architecture can be formed to match the task's cognitive structure. 

---
# Adaptive Memory and Reflection Multi-Agent System for Medical Question Answering 

**Authors**: Pradeep Murugesan, Luoxiao Yang, Xueli Chen, Xinqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2608.19029)  

**Abstract**: Accurate and responsible medical question answering (QA) is important in healthcare, where complex cases require factual knowledge and nuanced reasoning. Existing medical QA systems, typically based on single-agent architectures and static retrieval, often lack adaptability, persistent memory, and structured decision-making. This work introduces an adaptive memory and reflection (AMR) agentic system, a multi-agent framework in which specialized agents use dedicated memory and reflection-based feedback to retrieve relevant prior cases and improve subsequent reasoning. Complexity assessment routes questions through solo, collaborative, or escalated workflows, while consensus and ethical overseer modules support reasoning consolidation and output review. Evaluation on MedQA and MedMCQA demonstrates strong performance compared with several baselines. Ablation studies show that combining agent-specific memory, reflection, and external retrieval yields the strongest performance. These findings highlight the potential of structured memory and feedback for developing more trustworthy medical agents. The source code is publicly available at this https URL. 

---
# Self-prompting and cross-model consensus enable reproducible data extraction from scientific literature with large language models 

**Authors**: Valentin Romanov, Monique Bax, Steven Niederer  

**Link**: [PDF](https://arxiv.org/pdf/2608.19025)  

**Abstract**: Accurately extracting nuanced, contextualized data from research articles is laborious and time intensive. Here, we investigate the performance of frontier, browser-based large language models (LLMs) to extract highly contextualized information. We demonstrate four escalating workflows, 1) given an expert curated prompt and research articles, most frontier LLMs perform well at data extraction, however can struggle with interpreting scientific context and nuance, 2) given simple instructions, LLMs can author their own prompts which were almost as eNective as expert-written prompts, 3) autonomous discovery of research literature was diNicult, agents either missed or hallucinated references, and 4) LLMs can create new datasets from published guidelines that closely match human-expert judges, but still require a human-in-the-loop. Together, these findings define an auditable division of labour in which experts specify the evidence standard, models cross-check repeated extractions and researchers resolve disputed cases, providing a practical route to scaling scientific data curation without relinquishing expert oversight. 

---
# A Theory of Post-hoc Debate Judgement 

**Authors**: Xiang Yin, Adam Dejl, Antonio Rago, Lihu Chen, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2608.19002)  

**Abstract**: Debates have recently emerged as a useful methodology for agentic AI to improve performance as well as to aid explainability and user engagement. For example, LLM-empowered agents may debate internally (with themselves) and/or externally (with other agents). In many settings where debates are used, debates' outcomes and resulting outputs are determined post-hoc by external judges, often LLMs. In this paper we develop and test a novel theory of debate judgement applicable to all settings where agents engage in debates by providing pros and cons for their opinions therein. Specifically, we identify a number of formal properties that debate judgement may be required to satisfy in general, as concerns reproducibility, robustness, groundedness and explainability. Then, we explore their satisfaction formally and/or experimentally, for claim verification settings, for two specific alternative debate judgement methods: variants of the LLMs as a judge idea and formal semantics drawn from computational argumentation. We show that the two methods give similar accuracy performances but the former may lack formal guarantees that the latter brings. Overall, our study indicates argumentation semantics as an ideal candidate for principled judges in debate-driven AI. 

---
# Breaking the weakest link to evade vision language models 

**Authors**: Ilan Zini, Boussad Addad, Katarzyna Kapusta  

**Link**: [PDF](https://arxiv.org/pdf/2608.18938)  

**Abstract**: Vision Language Models (VLMs) have recently emerged as a critical component of multimodal AI systems, enabling joint reasoning over visual and textual inputs in real-world and safety-critical applications. Despite their growing deployment, the robustness of VLMs against adversarial threats remains insufficiently explored, particularly in the context of evasion attacks targeting multimodal alignment. In this work, we investigate the vulnerability of VLMs to adversarial perturbations applied to visual inputs and study two attack settings: untargeted attacks, where the goal is to disrupt the model's interpretation of the original image, and targeted attacks, where the adversary aims to force the model to generate a specific semantic description unrelated to the original image. To efficiently generate adversarial examples, we propose a gradient-based attack method that performs optimization exclusively on the vision encoder of the VLM rather than on the entire multimodal architecture. This design significantly reduces the computational cost and resource requirements of the attack while maintaining strong effectiveness. We evaluate our approach on several open-source VLMs, including Qwen2.5-VL, Granite-Vision, FastVLM, and Phi-3.5-Vision, and show that small, human-imperceptible perturbations can substantially alter the textual interpretation produced by the models. Our findings highlight the vulnerability of modern VLMs to adversarial manipulation and emphasize the need for improved robustness and security mechanisms in multimodal AI systems. 

---
# \textsc{TestifAI}: Tomography-Based Testing for Deep Learning Systems 

**Authors**: Arooj Arif, Tobias Hartung, Elena Botoeva, Alexandros Koliousis  

**Link**: [PDF](https://arxiv.org/pdf/2608.18900)  

**Abstract**: As AI systems are increasingly deployed in safety-critical application domains (e.g., autonomous driving), associated risks increase too. Deep learning models underlying modern AI systems, therefore, must undergo thorough testing to ensure their correct behaviour. A single robustness test involves thousands of inferences to empirically verify if a model's outputs remain stable under a bounded perturbation of its inputs. However, existing testing frameworks lack the means to systematically explore and summarise robustness across a combinatorial space of perturbations.
We propose TestifAI, a deep learning testing framework for efficient and accurate estimation of robustness against combinations of perturbations. TestifAI enables users to specify operational conditions as structured spaces of semantic input perturbations (e.g., image blur, brightness and zoom) and discrete severity levels (e.g., low, medium and high). Users can query model robustness for any combination (e.g., "low blur, high brightness, and medium zoom"). To achieve efficiency and accuracy, TestifAI introduces partial model tomography, a novel approach to reconstructing model behaviour in a multi-perturbation space from tests that apply only a small number of perturbations (lower-order projections). To estimate robustness against at least three perturbations, TestifAI trains an auxiliary model on the results of tests involving up to two perturbations only, avoiding execution of an exponential number of tests. Our experiments on five image and language classification tasks show that TestifAI can predict higher-order (3 and 4 perturbations) test outcomes from low-order (1 and 2 perturbations) observations with an aggregate robustness estimation error of less than 7%, while reducing the number of inferences by 60-80%. 

---
# Syntactic Simplification of OWL Class Expressions 

**Authors**: Alkid Baci, N'Dah Jean Kouagou, Caglar Demir, Axel-Cyrille Ngonga Ngomo  

**Link**: [PDF](https://arxiv.org/pdf/2608.18899)  

**Abstract**: Class expression learning often produces complex OWL class expressions that are difficult to interpret and reason over. However, by following theoretically grounded simplification principles, this complexity can be reduced. In this paper, we propose Class Expression Simplifier (CES), a novel algorithm for the syntactic simplification of class expressions in Description Logics (DL). CES aims to preserve formal semantics while reducing representational complexity. It systematically applies rewriting rules to eliminate redundancies and identify simpler yet equivalent expressions, thereby producing more compact and human-readable representations without altering logical entailments. We evaluate the effectiveness of CES on class expressions learned from two medium-sized ontologies, demonstrating measurable improvements in reasoning efficiency and reductions in verbosity. This work contributes to the broader goal of making ontology-driven applications more accessible, maintainable, and scalable, with direct implications for knowledge graph construction, semantic search, and Web-scale reasoning. CES is implemented within the open-source Python framework OWLAPY and is publicly available. 

---
# Training-Free Inference-Time Self-Reflection and Cost-Bounded Early Stopping for Large Language Models 

**Authors**: Wei Yu, Suxing Liu, Minjie Yu, Jiahao Wang, Zhijian Zheng, Haocheng Deng, Bing Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.18884)  

**Abstract**: Reinforcement-learning training of reasoning LLMs (e.g., GRPO) is expensive and requires a controllable environment, committing every contribution to a full training pipeline. We present EvoResearcher, a training-free, inference-time protocol that adds cost-bounded self-reflection to a single frozen LLM backbone. The protocol iterates generate -> self-critique -> revise until a maximum depth D is reached or the critique returns the CONFIRMED sentinel, an implicit early stop that lets the backbone self-verify its answer under a strict compute budget. Four self-reflective meta-reward components (correctness, efficiency, reflection depth, tool-call diversity) act as design principles instantiated as prompt-level mechanisms, so their benefits accrue with zero gradient updates. We validate the protocol on Big-Bench Hard (100 questions) and establish cross-domain behavior on GSM8K (500) and MATH (500) on the same frozen backbone, with cross-model replication on Qwen2.5-72B. All experiments use pure-reasoning benchmarks; the tool-call diversity component is validated in prompt-level form, and the environment-level and multi-agent extensions are design blueprints left to future work. On clean BBH the protocol does not raise accuracy beyond the 95% Wilson interval; its value is cost-bounded self-verification, with the CONFIRMED early stop terminating 82-88% of items at equal accuracy (about 2.1 generations per question). 

---
# DentAgent: Evidence-Centric Multi-Agent Coordination for Multimodal Dental Reasoning 

**Authors**: Zijie Meng, Xiwei Dai, Yixuan Tang, Jin Hao, Yang Feng, Fudong Zhu, Xiaoqiang Liu, Shaosheng Cao, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18878)  

**Abstract**: Oral diseases affect billions of people worldwide, underscoring a pressing need for accurate and reliable dental assessment that integrates heterogeneous evidence from domain knowledge, radiographs, intraoral photographs, and 3D dental data. Most existing dental AI systems remain modality- or task-specific. Although recent vision-language models support flexible dental question answering, directly generated response leaves evidence implicit and untraceable. To address these limitations, we introduce DentAgent, an evidence-centric multi-agent framework, in which the Orchestrator coordinate five specialized agents spanning various modalities. Each specialist utilizes domain tools to convert observations into structured evidence records. The Evidence Blackboard manages these records as a shared evidence state, tracking coverage, gaps, and conflicts before response generation. This standardized evidence representation integrates isolated dental capabilities into a unified agentic workflow. Across four benchmarks, DentAgent demonstrates leading performance, even surpassing the senior specialists by 17.3 percentage points on multi-label diagnosis, which supports its value for broadly applicable and traceable multimodal dental reasoning, and highlights its potential as a technical foundation for population oral health assessment and management. 

---
# SkillGate: Training In-Policy Skill Selection in Long-Horizon Agents 

**Authors**: Qingyao Li, Wenxiang Jiao, Shuai Shao, Kangning Zhang, Yuan Lu, Yi Guo, Weiwen Liu, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18852)  

**Abstract**: Agent frameworks increasingly package procedural knowledge as skills: instruction files an agent reads on demand, while public libraries now hold thousands of them. Which skill to read has thus become a decision the policy itself makes in the middle of an episode, yet no existing signal trains it. We show that the default remedy, outcome-rewarded RL over the candidate slate, cannot teach it, for a structural reason we identify and name selector credit starvation: under a broadcast, sequence-level advantage, the few tokens that name the chosen skill carry a vanishing share of the loss, and the credit they inherit is increasingly wrong-signed as trajectories lengthen. A correct choice is punished whenever the execution after it fails, even though the choice itself is among the most valuable decisions in the trajectory. Auditing a completed run's own training artifacts confirms all three properties, each worsening monotonically with horizon. SkillGate removes the failure by construction: it partitions the token support into two disjoint credit channels, outcome credit reaching only execution tokens, and a separate action-local advantage reaching exactly the skill-naming tokens, positive only when a trajectory's single read is the correct one. On five agentic benchmarks under a 16-candidate slate, SkillGate lifts a 9B policy from 40.8% to 53.2% trial success, well ahead of the identical budget spent on outcome reward alone, while cutting exposure to misleading candidates by two thirds and reading fewer skills. 

---
# ORBITER: Conflict-Aware Decision-Making for Agentic Last-Mile Delivery 

**Authors**: Mingzhao Li, Chenxi Liu, Yan Zhao, Hao Miao  

**Link**: [PDF](https://arxiv.org/pdf/2608.18846)  

**Abstract**: Last-mile delivery aims to handle dynamically arriving orders with couriers while modeling complex spatial and temporal correlations. Recent learning-based methods model spatiotemporal dependencies among orders to predict courier service sequences, but leave next-order decision making unexplained. Describing the current delivery state in language allows LLMs to reason explicitly about the spatial, temporal, and behavioral cues behind an individual decision. As direct predictors, however, LLMs remain sensitive to task presentation and often produce unreliable decisions. To address these challenges, we introduce ORBITER, an agentic Order Arbiter for next-order decision-making in last-mile delivery. ORBITER models courier service through decision points, each containing the courier's spatiotemporal state and visible orders and exposing local trade-offs for modeling and verification. Fixed proposers rank the candidates, and a structured report identifies where their rankings disagree. The LLM uses task-specific tools to gather evidence on the leading alternatives, while an independent critic checks the resulting decision against that evidence. We conduct extensive evaluations on data in four cities, where ORBITER outperforms existing state-of-the-art baselines by up to 9.2% on average showing its effectiveness. 

---
# Verifiable abstention makes AI leak diagnosis accountable in water distribution networks 

**Authors**: Tianwei Mu, Yue Wang, Mingzhe Yuan, Manhong Huang, Wenhong Wang, Xuerui Yin, Qing Luo, Min Xiao, Hui Yang, Jun Li, Dan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2608.18836)  

**Abstract**: Utilities lose a substantial share of treated water to leakage, yet rarely trust artificial-intelligence localizers to dispatch crews: guessing everywhere cannot justify excavation. The gap is accountability, not accuracy: no method proves when it should not act. Here we recast leak localization as decision-making under verifiable abstention. A physics-grounded executor agent falsifies hypotheses (leak, demand, sensor, valve) against a digital twin; an independent supervisor agent, with a large-language-model (LLM) auditor, checks evidence against a code-verifiable contract, then certifies a dispatch, requests evidence or abstains. Under field-grade noise, a 32% forced baseline becomes 96% decision precision on acted events. On an independently generated benchmark it acts on only 4 of 33 leaks, all correct. A 194-event register of audited real leak locations with twin-simulated pressures and flows yields five excavation dispatches, three correct, and 44% survey recovery at full district precision. Accountable abstention offers a defensible route to autonomous water-infrastructure operation. 

---
# Pairwise Logical Selection of Enthymeme Completions under Semantic-Link Uncertainty 

**Authors**: Xuyao Feng, Antonis Bikakis  

**Link**: [PDF](https://arxiv.org/pdf/2608.18820)  

**Abstract**: Arguments often omit premises or claims, forming enthymemes. We study pairwise logical selection between two candidates for the omitted component. Existing natural language methods can identify or generate candidates but often do not expose how the selected candidate completes the inference, while logic-based approaches usually assume that the required formulae and background knowledge are available. We extend a prior neuro-symbolic pipeline from missing-premise to missing-claim selection and replace binary entailment outcomes with logical-resistance scores. Top-Link uses weighted Partial MaxSAT under a single configuration of highest-confidence semantic links. We then introduce Possible-World Atom-Link Formalization (PWAL), which keeps translated formulae fixed and marginalizes logical resistance over alternative cross-formula semantic-link configurations. We evaluate PWAL on five tasks: ARCT and a CDED-derived task for missing-premise selection, iDebate- and AAE2-derived tasks for missing-claim selection, and alphaNLI for abductive hypothesis selection. Relative to Top-Link, PWAL raises strict accuracy by 2.95-30.86 percentage points and reduces tie rates by 4.57-58.00 percentage points on all five tasks. When ties receive half credit, accuracy still increases by 0.45-6.04 percentage points. PWAL also records the translated formulae, sampled link configurations, and resistance components for every comparison, providing a transparent trace of each score. 

---
# Metrics That Write Themselves: Evolving an Evaluator from Its Own Blind Spots 

**Authors**: Xing Zhang, Yanwei Cui, Guanghui Wang, Zhihao Lin, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2608.18744)  

**Abstract**: Agents improve quickly against a reliable automatic metric and stall without one, and the applications that need them most, report generation among them, are the ones nobody knows how to score. Can the metric write itself? Saying what makes an answer good is hard; pointing at something wrong with one is easier, so the metric we evolve is a pool of small Python operators that each flag a candidate for one named defect, or abstain, and vote. Asking a model for operators directly does not work: 183 candidates realise only 96 distinct behaviours, from one narrow region of an enormous space. EvalCEGAR instead borrows counterexample-guided abstraction refinement from program verification. It reads the pool as an abstraction and searches for a collision, two answers the operators score identically, one correct and one not. That pair, not a prompt, is the authoring request, and when a collision defeats every attempt the loop widens what an operator may read rather than resampling. On MBPP+ and HumanEval+, a sandbox whose hidden unit tests give exact ground truth, the loop writes a 55-line operator that closes 15.4% of the gap between flagging nothing and a perfect filter on 428 unseen tasks (+0.0065, p=0.0010) at a quarter of our best hand-written operator's flags. On the benchmark it never saw it matches that operator's effect exactly on a third of the flags. Six of eight runs admit such an operator and all six help out of sample; our 15 hand-written operators applied together as one filter lose accuracy. An LLM judge on the same information ties that delta on a nearly disjoint set of candidates, and charges a model call per candidate forever where the operator charges none. 

---
# A Multi-Agent Platform for Automated Enterprise Analytics and Insight Generation 

**Authors**: Manoj N M, Vijayakrishna S, Manjunath Srinivas, Rohit Pahan  

**Link**: [PDF](https://arxiv.org/pdf/2608.18740)  

**Abstract**: This paper proposes a multi-agent framework built on CrewAI [1] for conversational business intelligence. Five specialized AI agents operate in a sequential pipeline to process natural language queries, retrieve and analyze data, generate visualizations via the Model Context Protocol (MCP) [2], and deliver actionable insights. The platform features a defense-in-depth security architecture for multi-tenant data isolation and a query parameterization mechanism for transforming conversational insights into reusable dashboard components. Evaluation across 300 end-to-end test cases spanning synthetic and production enterprise datasets demonstrates 95.3% functional accuracy, a mean response latency of 24 seconds, and a response quality score of 4.52/5.0 as assessed by an LLM-as-a-Judge framework, with a 93.0% hallucination-free rate, representing a 22.6 percentage point accuracy improvement and 20.2% quality gain over a single-agent baseline. Cross-model evaluation across four LLM backends and human expert validation confirm architectural generalizability and evaluator reliability. An ablation study confirms that the Data Analysis and Report Aggregation agents are the primary drivers of output quality. 

---
# Competence, Not Accuracy: A Diagnostic for Reference-Free Judge Gates in Skill Optimization 

**Authors**: Chenle Chen, Yangbo Wei, Chao Yao, Shaoqiang Lu, Junhong Qian, Chen Wu, Lei He  

**Link**: [PDF](https://arxiv.org/pdf/2608.18719)  

**Abstract**: Text-space skill optimization adapts a frozen agent by evolving a natural-language skill document, accepting each candidate through a validation gate. Existing gates rely on verifiable rewards, confining these methods to tasks with an automatic verifier. Replacing the verifier with an LLM-judge gate would lift that restriction, but whether such a gate carries usable signal is untested. We ask a prior question: can we tell, before placing a judge in the loop, whether its scores separate correct from incorrect answers at all? We formalize a reference-free judge as a latent solver -- its verdict rests on agreement with whatever it would itself conclude, so its capacity to evaluate is bounded by its capacity to solve. The model yields a closed-form bound on discriminability (ROC-AUC) in the judge's competence $c$ and answer-space size $k$, a necessary condition $c > 1/k$, and the result that the marginal AUC is confounded by item difficulty while a within-question estimator is not. A non-intervening probe records judge scores on genuine optimization runs without altering any decision. We find discriminability at chance where competence sits near the floor and usable above it; that a judge's benchmark accuracy overstates the competence that matters; and, in a closed-loop study, that the screen predicts which kind of gating error occurs. The result is a cheap pre-deployment diagnostic for judge gates. 

---
# RTPO: Reverse-Turn Policy Optimization for Stabilizing Agentic RL Training 

**Authors**: Yugu Li, Jimmy Cao, Jianglin Qiao, Siyi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18682)  

**Abstract**: Training multi-turn agentic workflows with reinforcement learning (RL) enables large language models to perform complex reasoning, use external tools, and conduct iterative search beyond single-turn settings. Yet multi-turn RL training remains highly unstable, often causing severe performance degradation as the number of turns increases. Through theoretical analysis, we identify three tightly coupled sources of instability: rollout-training context mismatch, weak turn-level credit assignment under sparse terminal rewards, and asynchronous policy drift when short and long trajectories are optimized under different policy versions. We show that these issues share a common structural origin in flattened trajectory optimization and address them through a unified reverse-turn formulation. We propose Reverse-Turn Policy Optimization (RTPO), which organizes multi-turn rollouts as sparse reverse trees and performs turn-level policy updates in temporal reverse order, aligning each decision with its downstream continuation. RTPO enables causally consistent turn-level credit assignment and on-policy continuation to control asynchronous drift. We provide theoretical guarantees showing that RTPO eliminates context mismatch and asynchronous drift under the proposed turn-level formulation, reduces credit bias, and converges to recursive optimality. Experiments on multi-turn agentic RL benchmarks show that RTPO improves upon trajectory- and turn-level baselines by 21.50% and 10.76%, respectively, highlighting its potential to support more stable training for tool-using agents. 

---
# Sanyu Studio: A Multi-Agent System for Art-Historical Narrative Construction 

**Authors**: Zhaoxi Wei, Hongye Yang, Shuyuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2608.18677)  

**Abstract**: Amid concerns that generative AI may standardize art interpretation, this paper examines whether LLM-based interaction can support plural art-historical narrative construction. We present Sanyu Studio, a multi-agent dialogue system that models 321 Sanyu oil paintings as agents with fact, interpretation, organization, and memory-filtering mechanisms. Based on a seven-day workshop with eight art-university participants, the study shows that user prompts, evidence organization, and cognitive tendencies shaped divergent yet coherent versions of digital Sanyu. The findings suggest that, under conditions of limited historical evidence, AI can amplify human agency and offer public audiences an interactive entry point into art-historical interpretation. 

---
# Candidate-Fate Accounting for Transparent Sensor Diagnostic Pipeline Search 

**Authors**: Haotao Xie, Yutian Chen, Yangqi Liu, Xiaoyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18665)  

**Abstract**: Industrial sensor diagnostics relies on preprocessing, representation, and classification pipelines, making automated pipeline search useful for reducing manual design cost. However, existing automated machine/deep learning (AutoML/AutoDL) reports typically retain only fitted trials, scores, and winners, omitting generated candidates that are invalid, pruned, skipped, cached, or unfitted. This omission limits reviewers' ability to check signal constraints, budget use, and unevaluated legal alternatives. To address this, we propose candidate-fate accounting, a candidate-level audit framework for diagnostic search traces. It records each observed candidate as auditable evidence: hashes merge repeated observations, legality checks flag invalid candidates, allocation rationales explain budget decisions, and a closed fate ledger assigns one terminal fate to each candidate. Experiments on three bearing-diagnostic datasets show that the framework detects invalid candidates and identifies 30--41 candidates omitted by fitted-trial-only reports, with closed fate records verifying complete candidate accounting while maintaining competitive diagnostic performance. The code is available at this https URL. 

---
# Preference Reasoning under Indeterminacy in Large Language Models 

**Authors**: Hadi Hosseini, Samarth Khanna, Xiyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18631)  

**Abstract**: As large language models evolve into decision-making agents, the ability to reason over preferences becomes fundamental to alignment, coordination, and collective intelligence. Yet, unlike standard benchmarks, real-world preference reasoning is inherently indeterminate: information may be incomplete, and valid solutions may not exist. We argue that indeterminacy, rather than correctness alone, is a central challenge for AI reasoning. We formalize this challenge along two axes, (i) epistemic indeterminacy, arising from incomplete, partial, or expressive preferences, and (ii) structural indeterminacy, arising from the non-existence of solutions under standard social choice concepts. Across a hierarchy of tasks, we show that state-of-the-art language models systematically fail to distinguish between determined and undetermined instances, exhibiting miscalibrated reasoning even in verification settings. 

---
# CTIFoundry: An Agent-Native Corpus Scaffold for Cyber Threat Intelligence 

**Authors**: Yutong Cheng, Changze Li, Qian Cui, Wei Ding, Lingzhi Wang, Yan Chen, Peng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.18613)  

**Abstract**: Cyber threat intelligence (CTI) is increasingly consumed not by human analysts but by LLM agents that compose multi-step investigations at query time. The harness side of this shift has matured rapidly (planning loops, tool protocols, context management), but the corpus side has not: threat reports and vulnerability databases are still packaged for retrieval-augmented generation, as opaque chunks behind an embedding index. We argue that this substrate, not model capability, is the bottleneck on agentic CTI investigation, and present CTIFoundry, an agent-native corpus scaffold. At build time, CTIFoundry materializes the latent structure of a CTI corpus: a deterministic ontology graph over four authoritative knowledge bases (CVE, CWE, CAPEC, ATT&CK) whose official cross-references become typed, traversable edges; a span-grounded report layer whose canonical, alias-resolved cross-vendor entities index provenance-carrying chunks; and hybrid dense+lexical retrieval surfaces. At query time this structure is exposed through seven typed tools and three procedural skills mounted on a stock open-source agent harness. On the public CTIConnect benchmark, swapping only the action surface lifts the identically-harnessed agent by +0.19 to +0.28 overall F1 across a four-model, two-provider panel: a small model on CTIFoundry surpasses a flagship on the flat substrate, and the gain is not bought with search effort, since on both Claude models the scaffolded agent is more accurate at roughly half the tool calls. An ablation attributes it: typed structure carries the larger share, procedural skills convert structure into discipline, and the two compose super-additively, because skills bind only to structure that exists. 

---
# Can a Lightweight Multimodal Model Estimate LLM Reasoning Performance? A Study for Compute-Optimal Document Inference 

**Authors**: Zishan Ahmad, Vishal Vaddina  

**Link**: [PDF](https://arxiv.org/pdf/2608.18591)  

**Abstract**: Uniformly allocating inference reasoning budgets to LLMs is expensive and prone to over-thinking penalties; especially in document tasks where visual layouts drive complexity. To address this, we introduce BudgetDoc, the first multimodal benchmark providing explicit supervision for model-budget-performance trade-offs across three document tasks. Using BudgetDoc, we train DRB (Document-Reasoning Balancer), an approx. 1B-parameter pre-flight estimator (SigLIP-2 + Qwen3-0.6B) that predicts ordinal model performance across budget levels, achieving a 0.753 weighted F1. When dynamically allocating reasoning budgets across five frontier models and three datasets, DRB matches or improves F1 scores compared to always-maximum-budget baselines in 9 of 15 configurations while drastically reducing cost. Finally, preliminary evaluations demonstrate DRB's potential to generalize to cross-model selection. 

---
# FACET: Preserving Source Intent and Executable State in Terminal Task Synthesis 

**Authors**: Kou Shi, Zun Wang, Qisheng Su, Shiting Huang, Ziao Zhang, Zhen Fang, Qingnan Ren, Jin Liu, Yu Zeng, Yiming Zhao, Lin Chen, Zehui Chen, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.18580)  

**Abstract**: Training terminal agents requires scalable executable supervision, yet synthesizing high-quality terminal tasks remains challenging. Each task couples an instruction, an initialized environment, a reference solution, and an executable verifier; if these artifacts are generated from inconsistent assumptions, the resulting task may be unsolvable or incorrectly evaluated. Meanwhile, multi-stage synthesis can discard the goals, dependencies, state transitions, and procedural constraints encoded in the original sources. We present FACET (Fine-grained Agentic Construction of Executable Tasks), a framework that addresses both information preservation and cross-artifact consistency. FACET reconstructs related agent skills into coherent, information-rich scenarios, then realizes and repairs the execution environment before generating the final task artifacts. The resulting container state serves as shared grounding for the instruction, solution, and verifier, while execution-based validation and targeted repair correct artifact-specific failures without unnecessarily regenerating valid components. FACET produces complex terminal tasks with dense executable checks, and successful trajectories collected from these tasks provide effective, data-efficient supervision. Fine-tuning models across multiple scales consistently improves performance on Terminal-Bench 2.1, while analyses of alternative generation schemes support the importance of environment-grounded construction for task validity and solution-verifier alignment. These results establish source-intent preservation and shared executable-state grounding as key principles for scalable terminal-task synthesis. 

---
# Bridging Search and CRM: Productionizing AI Product Research Agents for Customer Re-Engagement 

**Authors**: Mandar Kulkarni, Pooja A., Samir Shah  

**Link**: [PDF](https://arxiv.org/pdf/2608.18543)  

**Abstract**: Modern e-commerce platforms often operate search, recommendation, personalization, and CRM systems independently, limiting opportunities for proactive customer re-engagement. This is particularly challenging for exploratory intents such as best smartphones or latest 5G phones, where users may leave the platform for external research before purchasing. We present a scalable, production-deployed framework that bridges search and CRM workflows through AI-powered Product Research Agents. The system identifies users with exploratory purchase intent and low engagement, conducts grounded multi-agent product research using behavioral signals, external knowledge, and enterprise catalog data, and delivers personalized recommendations through WhatsApp. We evaluate the framework in a 23-day production deployment involving approximately 15K WhatsApp notifications for mobile product discovery. The campaign achieved substantial CTR improvements over traditional WhatsApp recommendation campaigns, with evidence of secondary engagement through message forwarding and sharing. The deployment also generated downstream purchases and GMV impact, demonstrating the practical effectiveness of AI Product Research Agents for proactive customer re-engagement and end-to-end customer journey optimization. 

---
# FinRCA-Bench: Benchmarking Evidence Retrieval and Reasoning for Financial AI Systems 

**Authors**: Pratik Ghawate  

**Link**: [PDF](https://arxiv.org/pdf/2608.18534)  

**Abstract**: Large language models are increasingly used to support financial operations, but their apparent reasoning performance can depend on whether they receive the right evidence. In financial reconciliation, the evidence needed for diagnosis is distributed across invoices, purchase orders, approvals, allocations, payments, ledger entries, and bank activity, linked by transactional relationships rather than textual similarity. End-to-end accuracy can therefore conflate evidence access with reasoning quality. We introduce FinRCA-Bench, a deterministic synthetic benchmark of 2,250 accounts-payable-to-bank reconciliation cases spanning 14 operational tables, including 1,500 injected failures across 15 causal categories and 750 legitimate or hard-negative cases. Root-cause labels and record-level evidence contracts are hidden from the model, allowing retrieval to be evaluated independently of answer correctness. We compare Rules/SQL, classical machine learning, dense semantic retrieval, deterministic relational expansion, and Typed Provenance Graph Retrieval (TPGR), a typed traversal restricted to persisted transaction relationships. Rules/SQL reaches 84.97% held-out exact accuracy and classical ML reaches 95.44%. Holding the reasoning model, prompt, and generation settings fixed while changing only retrieval increases macro required-record recall from 0.83% to 77.70% and exact 16-class accuracy from 2.05% to 72.44%. Structural retrieval failures outnumber reasoning failures with sufficient retrieval by 95 to 15; 254 correct predictions occur despite incomplete retrieval, and strict returned-evidence contract accuracy is only 5.72%. On FinRCA-Bench, retrieval architecture strongly shapes observed AI-system performance, and a correct root-cause label is a weak proxy for an auditable diagnosis. 

---
# Pairwise Ranking Outperforms Single-Action RL for Offline Explanation Selection: A Practical Lesson 

**Authors**: Tanay Chowdhury, Saeideh Shahrokh Esfahani  

**Link**: [PDF](https://arxiv.org/pdf/2608.18531)  

**Abstract**: Industrial explainable-recommendation systems built on LLMs incur a substantial serving cost: each request triggers an LLM generation, with latency in the hundreds of milliseconds and cost that scales linearly with traffic. We separate generation from selection: explanations are produced ahead of time as a frozen candidate pool (six prompt styles, two commodity LLMs), and a small CPU-resident selector picks one at request time. The stack needs no GPU and returns in under 100 ms.
Our primary benchmark is a 2,958-pair XRec Google Local subset, evaluating six offline-pool selectors (LambdaRank, PPO, GRPO, DPO, teacher-student distillation) and three KG-path selectors (random walks, edge-disjoint enumeration, MMR-reranked paths). A 300-pair MovieLens-1M split with Claude-Sonnet-4.5 references serves as an internal cross-dataset check, since no public benchmark exists for this setting. All variants use the same BERTScore-F1 protocol as XRec and G-Refer, averaged across five seeds.
LambdaRank reaches F1 = 0.500 on Google Local, exceeding both G-Refer and XRec, and F1 = 0.329 on the MovieLens-1M check. With seed variance below 0.003 F1, the ordering is reliable: pairwise learning-to-rank outperforms single-action RL (PPO, GRPO, DPO), which use only one labelled candidate per rollout, leaving K-1 labels unused.
The KG-path family targets a different objective: all three variants reach USR = 1.000 on Google Local and 0.997-1.000 on MovieLens-1M, since per-request path grounding yields a unique output per query, avoiding template-collapse failures affecting cached-LLM outputs.
A generator-pool study comparing Claude 3 Haiku and Claude Haiku 4.5 shows small F1 shifts (0.001-0.006) while preserving selector ranking: selector and generator can be evaluated independently, though absolute F1 depends on the generator. End-to-end build cost is near $15 on commodity hardware. 

---
# Which Negatives Matter? Ask Your Text Encoder: Adaptive Similarity Margins for Dense-Caption Retrieval 

**Authors**: Haoyue Liu, Ye Chen, Zhichao Wang, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18521)  

**Abstract**: Dense-caption retrieval has recently been improved by introducing segmentation, edge maps, LLM-filtered captions, and cross-modal modules into contrastive fine-tuning. However, these methods largely inherit the same InfoNCE objective, whose optimization can prematurely saturate under a strong pre-trained initialization: on dense captions, the loss falls below 10^{-3} on 80% of batches within the first epoch, while its gradient reaches exact zero in fp32 in 47% of measurements. We find that this behavior is closely related to the large number of near-duplicate captions in dense-caption benchmarks, where a few highly similar negatives remain unresolved after the easy majority has already been separated. As a remedy, we introduce HN-CLIP, which uses the text encoder's own text-text geometry to construct per-negative adaptive similarity margins. Specifically, a detached caption-similarity matrix is added to the negative logits, assigning larger margins to more similar captions without mining, synthesizing, or resampling negatives. The resulting objective requires only one caption-similarity matrix and a masked logit addition during training, with no auxiliary data, additional parameters, offline preprocessing, or inference-time overhead. Extensive experiments on four dense-caption retrieval benchmarks show that HN-CLIP improves over the strongest competitors by +2.4--+4.3 R@1 while training 2.4x faster than GOAL and 5.4x faster than StructXLIP. Moreover, the proposed objective improves all six tested fine-tuning frameworks on the in-domain benchmarks and reaches the strongest full-data baseline with only 20% of the training data. 

---
# UMER: Unifying Embedding and Ranking via Pair-Aware Discriminative Reasoning for Universal Multimodal Retrieval 

**Authors**: Libiao Chen, Xiyang Liu, Yanheng Wei, Tao Wang, Zhenyu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18504)  

**Abstract**: Universal multimodal retrieval aims to support diverse instruction-aware retrieval tasks, demanding both efficient corpus-scale matching and fine-grained semantic reasoning. Recent MLLM-based embedding methods typically derive representations from hidden states, while Chain-of-Thought (CoT) reasoning is emerging as a promising strategy for embedding enhancement by encoding intermediate semantic evidence into the representation space. However, existing CoT methods typically use item-wise reasoning over queries and candidates in isolation, providing no explicit evidence to distinguish a positive from a semantically confusable hard negative. Moreover, contrastive embeddings capture global similarity but struggle with meta-tasks requiring answer verification, category judgment or fine-grained reasoning. In this paper, we propose UMER, a Unified Multimodal Embedding and Ranking framework for universal multimodal retrieval. UMER replaces item-wise reflection with Pair-Aware Discriminative Reasoning, which compares query--candidate pairs to identify instruction-relevant matching and discrepancy evidence. UMER jointly learns contrastive embeddings for efficient global matching and discriminative ranking for explicit pairwise relevance judgment within a single MLLM. A complementary mutual distillation strategy further transfers reliable pairwise preferences between the embedding and ranking functions. On the MMEB-V2 benchmark, UMER achieves state-of-the-art performance under comparable experimental settings while supporting budget-adjustable inference. 

---
# FM-Bench: A Benchmark for Long-Horizon Management with Competing Agents 

**Authors**: Tianyou Wang, Chongyang Gao, Kezhen Chen, Chen Dong, Yinghao He, Donghan Li, Wangcheng Xu, Hongjiu Zhang, Chi Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.18423)  

**Abstract**: Language model agents now execute bounded tasks reliably. Whether they can sustain effective decision-making over long horizons, where actions have cumulative consequences and the environment responds to their choices, remains largely unmeasured. FM-Bench (Football Management Benchmark) measures this. An LLM agent runs a football club for 20 in-game years through 26 tools and roughly 340 to 400 decision stops. It drafts a squad on the same budget as every rival, trades players, negotiates contracts, invests in facilities and youth, sets lineups, and answers to a board that can fire it, while a deterministic engine accumulates every year into one final score with no LLM judge or human rater. The solo track plays each of 15 frontier models against a frozen scripted world, and the Arena places the same models plus a scripted anchor in one shared 20-year world; to our knowledge, the first head-to-head evaluation at this scale. We measure six behavioral capabilities behind the score. Across three seeds, all 15 models complete every horizon while the blind scripted baselines die out in most of theirs, and claude-fable-5 tops the solo board on mean score and the Arena, where the title nonetheless rotates among ten models. Neither scale, price, nor vendor predicts the order; the order settles only late in the horizon, and the best first-play human lands only at the bottom of the model board. What separates the models is managerial behavior rather than computation. Higher-scoring models reduce slow-payoff investment near the end, keep cash invested rather than idle, and open renewals well before the deadline, while token spend predicts nothing. No model learns the market's hidden prices from hundreds of rejected bids, and self-managed memory fails in two opposite modes: an archive that only grows or a plan rewritten every season. Code is available at this https URL. 

---
# Improving Natural-Language Combinatorial-Optimization Accuracy in Resource-Constrained Language Models via Formal Abstractions 

**Authors**: Shrenil Shaun Sharma, Avi Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2608.18409)  

**Abstract**: Combinatorial scheduling poses a significant challenge for language models, requiring them to identify feasible solutions within exponentially large search spaces while satisfying complex constraints. This challenge is especially pronounced in resource-constrained settings, where larger language models are impractical and selection is limited to smaller models which often fail to preserve feasibility when scheduling directly from natural language. To address these limitations, we introduce SDDL, a neuro-symbolic framework that translates natural-language scheduling problems into compact, solver-aligned representations of tasks, resources, constraints, and objectives, while delegating low-level modeling and search to a deterministic compiler and external solver. On a 300-instance, multi-family subset of scheduling problems, SDDL improves independently verified feasibility for every resource-constrained model tested. The two strongest SDDL configurations reach 55.3% and 28.3%, up from direct-generation baselines of 23.7% and 1.3% and solver-code baselines of 21.7% and 7.0%, with a 0.0% median optimality gap among feasible schedules. By expressing problem structure rather than generating solutions or solver code, SDDL enables smaller models to approach the strongest evaluated direct- and solver-code configurations, including substantially larger frontier models. 

---
# When Clean Signals Are Not Enough: Detecting Structural Ambiguity for Safe Wearable Stress Classification 

**Authors**: Saba A. Farahani, Hung Cao, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2608.18397)  

**Abstract**: Wearable stress classifiers can achieve strong average performance while failing completely for a particular individual. On WESAD, a Random Forest reaches 93.0% mean accuracy yet yields F1 = 0 for Subject 14, whose cross-signal coupling weakens near stress onset. We call this structural ambiguity: individually plausible physiological channels form an inter-signal pattern that is poorly supported by the person's non-stress reference. We introduce the Individual Conformal Coupling Monitor (ICCM), a lightweight and transparent pre-inference monitor that quantifies subject-specific coupling divergence and routes each window to classify, defer, or abstain without retraining the downstream classifier. Across WESAD (N = 15) and Stress-Predict (N = 35), full-cohort Pearson associations between ambiguity and accuracy are negative (r = -0.607, p = 0.016; r = -0.412, p = 0.014). Robustness analyses temper this finding: rank correlations are not significant, and the WESAD association disappears when Subject 14 is removed. ICCM changes false-positive counts from 29 to 27 and 94 to 92, although neither paired change is significant. It withholds 3 of Subject 14's 21 stress windows but does not repair the missed-stress failure. These results position ICCM as an interpretable signal of unsupported physiology and individual failure, rather than a stand-alone safety guarantee. 

---
# A Jagged Frontier: Evaluating Robustness of Code Agents to Semantics-Preserving Transformations 

**Authors**: Hasan Najib Mahmud, Shreya Gupta, Isha Chaudhary, Nathaniel Enis, Ravi Mangal, Gagandeep Singh, Corina Pasareanu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18389)  

**Abstract**: AI code agents are increasingly deployed to resolve real software issues, yet their reliability under superficial code variations remains poorly understood. We evaluate whether coding agents that repair repository-level issues remain reliable when the surrounding codebase is rewritten into a semantically equivalent form. We introduce a random variant sampler that applies common semantics-preserving transformations (SPTs) - spanning control-flow rewrites, dead-code injection, and identifier renaming - to produce perturbed variants. We evaluate two agentic scaffolds (mini-SWE agent and OpenCode) each backed by one of four frontier models (Claude Opus 4.5, Kimi K2.5, MiniMax M2.5, and Qwen 3.6-27B) across instances drawn from SWE-bench Verified and SWE-bench Pro. For each instance, the agent is run multiple times on the unperturbed and perturbed variants, yielding paired resolve-rate estimates that isolate the perturbation effect from intrinsic stochasticity. We find small degradation in most configurations: up to 6.7 percentage points mean resolve-rate drop in the most affected configurations with statistically significant degradations in 6 of 16 configurations of model, scaffold, and dataset. Crucially, no single model ranking by robustness holds across scaffolds - Qwen is among the most robust under mini-SWE agent on SWE-bench Verified yet the most brittle under OpenCode - revealing a jagged robustness frontier. The simpler scaffold (mini-SWE agent) is more robust to perturbation. Our results demonstrate that even top frontier models are susceptible to semantics-preserving perturbations although the effect is not uniform, raising concerns about the deployment reliability of AI code agents in diverse real-world codebases. 

---
# Measuring the Partial-Credit Gap: A Strict Benchmark on Vietnam's 2025 Convex Marking Scheme 

**Authors**: Nguyen Quoc Hung, Nguyen Dang Minh, Le Nhu Quynh, Tran Khanh Linh, Nguyen Kieu Linh  

**Link**: [PDF](https://arxiv.org/pdf/2608.18336)  

**Abstract**: When evaluating language models on human exams, benchmarks typically score each response as right or wrong and report the overall accuracy. This approach assumes that partial knowledge is worth proportional credit, an assumption that fails when an examination uses a non-additive grading scheme. The 2025 reform of Vietnam's National High School Graduation Examination demonstrates the cost of this substitution. In Part II of the exam, candidates evaluate four true/false statements per question. The grading is convex: the number of correct statements earns 0, 0.10, 0.25, 0.50, or 1.00 points. Identifying three statements correctly pays 0.50 points, not the 0.75 points that standard accuracy metrics would award. Because Part II accounts for 4.00 of the exam's 10.00 points, reporting accuracy inflates the score by rewarding partial knowledge that the state explicitly penalizes. We introduce THPT-Ladder, a benchmark of 632 items from 21 official exams across 11 subjects, graded exactly as the ministry grades its students. The ministry publishes the marks of over a million candidates, allowing us to place models directly into the human cohort. Across eight models, the official rubric pays 0.020 to 0.159 points less per Part II question than proportional credit. This shortfall changes a model's apparent competence. For Qwen3.5-27B on the 2025 History exam, a 0.042-point shortfall drops its standing from the 90th to the 77th percentile among 481,293 candidates. A model's accuracy does not predict this penalty. At Claude Sonnet 5's accuracy level, different distributions of errors yield scores varying from 0.869 to 0.932 points per question. Official marks depend on how correct statements are grouped, meaning standard benchmarks report a competence the institution would not certify. 

---
# Governance Records as Supervision: Verifier-Selected Self-Training for Structured Workflow Repair 

**Authors**: Jesus Salas  

**Link**: [PDF](https://arxiv.org/pdf/2608.18324)  

**Abstract**: Machine-verifiable workflows produce governance records linking a task contract, model attempt, verifier decision, accepted output, and target origin. We test whether these records can supervise bounded models, consolidating occasional or expensive capability into reliable one-shot execution.
On fresh, structure-disjoint PlanBench replanning cases, Qwen3-14B thinking generated 24 plans admitted by the independently authored VAL verifier. Those plans trained the same checkpoint for non-thinking execution, without oracle targets or a stronger teacher. On 80 unopened cases, VAL-accepted plans increased from 1 to 57, with 56 paired gains and zero regressions; thinking reached 30. The adapter was schema-valid on all cases and used approximately 1/56 of thinking's mean latency. The separate paired interface-cure gate did not pass.
A matched ablation fixed the source cases, 52-candidate pool, 24-target count, model, recipe, and seed while changing target selection. On 160 new cases, base, schema-selected, model-self-selected, and VAL-selected execution reached 1, 55, 69, and 102 accepted plans. VAL exceeded self-selection by paired net +33 (p=0.0000019647), with gains in both difficulty strata. Independent semantic selection is therefore load-bearing relative to matched alternatives within this band.
A complementary Phi stronger-teacher arm raised base Phi-4 from 2 to 51 accepted plans and from 35 to 80 schema-valid outputs. Earlier synthetic experiments establish teachability, cumulative learning, construction robustness, and stopping boundaries. The results support verifier-selected supervision for bounded, machine-checkable capabilities, not arbitrary planning, enterprise validity, or unrestricted self-improvement. 

---
# ComponentBench: Diagnosing Component-Level Failures in Computer-Use Agents 

**Authors**: Tianchen Guan, Xinlei Lin, Royce Cheng-Yue, Xiangjun Wang, Shuyan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.18307)  

**Abstract**: Current evaluation of computer-use agents is split between long-horizon workflow benchmarks and atomic GUI-grounding tests. This leaves an under-instrumented middle layer: realistic component-centered interactions (e.g., toggle a button set) that are short enough to diagnose and rich enough to capture the burdens of modern interfaces. We present ComponentBench, a benchmark and diagnostic pipeline for component-level evaluation of computer-use agents on modern web UIs. ComponentBench is organized around a library-agnostic ontology of 97 canonical UI components instantiated as 2,910 programmatically verified tasks across widely used component libraries, paired with cleaned human reference trajectories that enable evaluation of both task success and interaction efficiency. Beyond task collection, we introduce a scalable pipeline for auditing realized structural difficulty after implementation and synthesizing structured failure analyses across tasks and component families. Evaluating seven models -- GPT-5.4, Gemini 3 Flash, GPT-5.4 mini, GPT-5 mini, Gemini 3.1 Flash-Lite, Qwen3-VL-235B, and UI-TARS-1.5-7B -- across four observation and action spaces, we show that these design choices critically impact performance. Within a single shared harness, changing only the observation and action space shifts task success by more than 30% for the same model: GPT-5 mini falls from 83.1% with accessibility-tree observations to 48.9% with coordinate-only Pixel control. Moreover, even the fastest configuration takes 3.7x as long as the matched human reference, and spatial manipulations that are trivial for humans continue to challenge current agents. 

---
# SESSE: Sketch, Expand, Sort, Summarize, Evaluate -- LLM-as-Judge Evaluation via Structured Decomposition 

**Authors**: Dae Lee, Mihai Delgeanu, Adel Youssef  

**Link**: [PDF](https://arxiv.org/pdf/2608.18303)  

**Abstract**: LLM-as-judge evaluation reduces response quality assessment to a single holistic A/B preference choice, providing no mechanism to isolate which quality dimensions drove the preference or distinguish model errors from genuine label ambiguity. We propose SESSE (Sketch, Expand, Sort, Summarize, Evaluate), a training-free framework that decomposes holistic judgment into structured sub-questions mined directly from the judge's own error cases; requiring no oracle responses, task-specific rubrics, or fine-tuning. On RewardBench (n=1,000), SESSE achieves near-parity with the chain-of-thought baseline and is competitive with RISE-Judge-32B (92.7%), a fine-tuned specialist, while remaining fully training-free. Per-criterion vote evidence provides an interpretable audit trail for diagnosing label ambiguity and judge failure modes unavailable from a single holistic output token. 

---
# The Lifecycle of LLM-as-a-Judge for Large-Scale Recommendation Explanations 

**Authors**: Emma Yanyang Kong, JJ Tan, Ishan Gupta, Lars Olds, Claire Campbell, David Fagnan, Veli Balin, Rohan Gosain, Louis Garcia, Minsu Jang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18300)  

**Abstract**: LLM-as-a-Judge, which leverages a large language model to evaluate natural language generated by another AI application or model, has become a standard, scalable approach for accelerating and extending costly human evaluation. However, most work treats a judge as a static artifact, evaluating it once at construction or against a fixed benchmark. In contrast, we argue that an LLM judge running in a production system is better understood as having a lifecycle: it must be built, trained, deployed, and continuously maintained as the surrounding data evolves, and each phase poses distinct technical and operational challenges.
We present such a lifecycle for the LLM judges that evaluate user-facing recommendation explanations at Netflix, where our pipeline generates and the judges assess hundreds of thousands of distinct show-level explanations per week, served across the mobile experience to millions of members. Our framework has four phases: (I) Birth, defining multiple evaluation criteria and building curated benchmark datasets with human labels and rationales; (II) Training, refining the judges' rubrics via Reasoning-Aligned Rubric Tuning (RART), a rubric-tuning procedure that uses a meta-judge over reasoning output as the learning signal; (III) Deployment, in which one judge serves two production roles: quality gating and reflective generation; and (IV) Monitoring, a continuous Human-in-the-Loop alignment process that detects drift and triggers re-tuning behind a human review gate. We report post-launch results from a five-week A/B test over tens of millions of members, in which the judge-aligned explanations shifted member viewing toward novel content (previously unwatched) and increased successful browse-to-play sessions relative to a no-explanation control, with no quality-related takedowns. 

---
# Evaluating Structured Information Extraction with Open Models in a High Risk Public Sector Application 

**Authors**: Elias Schubert, Felix Bießmann  

**Link**: [PDF](https://arxiv.org/pdf/2608.18289)  

**Abstract**: The extraction of structured information from unstructured documents represents a critical component of digital transformations in all sectors. While proprietary solutions dominate commercial applications, a rapidly growing ecosystem of open-source Optical Character Recognition (OCR) engines, Large Language Models (LLMs), and Vision-Language Models (VLMs) offers accessible alternatives. However, systematic evaluations on realistic, multi-step extraction pipelines remain scarce. Responsible usage of such extraction tools require comprehensive evaluations on realistic tasks, especially as these solutions will be key components of applications in the public sector that the EU AI act categorizes as high risk. To address this gap we present a comprehensive benchmark assessing the end-to-end performance of open-source systems on a complex real-world document processing task classified as high risk: Student applications for an international study program. We conduct a comprehensive empirical evaluation with state-of-the-art OCR engines, LLMs and VLMs. Our results reveal that while VLMs generally outperform OCR+LLM pipelines, even state-of-the-art open-source models struggle to handle such tasks reliably in zero-shot settings. Only 4 of 35 configurations achieved F1 scores above 0.5, with the best OCR+LLM pipeline matching top VLM performance, though most OCR+LLM combinations performed substantially worse. Roughly 75\% of all configurations scored below 0.25. Model scale influences performance, yet the relationship is non-linear: substantially larger models do not guarantee proportionally better results. Input quality, particularly the structural preservation of OCR output, emerges as a critical factor independent of downstream model capability. 

---
# Cacheable by Design? Training Mixture-of-Experts Routers for Locality Against the Edge Memory-Bandwidth Wall: A Pre-Registered Negative Result with a Systems Measurement Study 

**Authors**: Shriniwas Ramesh Suram  

**Link**: [PDF](https://arxiv.org/pdf/2608.18261)  

**Abstract**: Serving a 235B-parameter Mixture-of-Experts (MoE) model on a single 8 GB GPU is bottlenecked not by compute but by memory bandwidth: decode must stream each token's active experts from whichever tier holds them, and on consumer hardware most experts sit on an SSD far slower than RAM. We quantify this bandwidth wall on Qwen3-235B (Q4_K_M, 134 GB): measured decode is 0.44 tok/s warm, matching a bytes-per-token / bandwidth model, while a batching scheme that should amortize one disk sweep instead collapses at batch 32 from paging thrash. We build llama-moe-trace, a zero-surgery router-telemetry tool, and measure routing on Qwen3-30B: adjacent-token expert reuse is 2.0x chance, 95% of traffic uses 52.5% of experts, and an LRU cache of 13.4% of experts serves 66% of requests. We then ask whether cacheability is trainable: we pre-register training of 137M MoE language models with auxiliary locality and domain router losses, under joint criteria on cache-miss reduction and perplexity. The mechanism works (misses down up to 60%; a 99% static-pin hit rate) but every configuration fails the pre-registered <=1% perplexity gate -- miss reduction and quality are tightly coupled. Concurrent StickyMoE reports the same loss as near-free on single-domain sub-25M models; on multi-domain 137M we find the tax real. Our contribution is this pre-registered, stricter-criterion, multi-domain evaluation plus edge-serving measurements. A 340M rung shows the tax does not shrink with scale (it rises slightly). We further show training-free cache-aware rerouting stacks with trained locality -- together ~80% miss reduction at <=3.4% perplexity at both sizes, far cheaper than either alone -- while domain-primed prefetching does not help. All code, traces, and the pre-registration are released. 

---
# Redakto - The Incognito Tab for LLMs 

**Authors**: Saurav Kumar Saha, Tom Röhr, Felix Bießmann  

**Link**: [PDF](https://arxiv.org/pdf/2608.18260)  

**Abstract**: Large Language Models (LLMs) are being increasingly used in everyday applications. A major challenge in the context of LLMs or Artificial Intelligence (AI) in general is to ensure privacy when using them, meaning that personally identifiable information (PII) is removed from any text that enters an LLM. These challenges have become more urgent with novel EU legislation. Uncertainty around LLM usage with respect to privacy concerns in EU countries can be a major blocker for the speed of innovation and transfer from research to applications. Here we present \textbf{Redakto}, a tool that can be used for anonymizing text prior to feeding it to an LLM or other downstream text processing. We provide state-of-the-art functionalities for both redaction of PII but also when used for pseudonymization. These functionalities are exposed such that they can easily be used by end-users, through the Redakto web application, and by developers and researchers, via REST APIs and model context protocol (MCP) hooks. The implementation is fully open source, requires modest compute resources, and can be readily deployed on local hardware. In contrast to prior work and in order to better assess the quality of the anonymized texts, we conduct extensive empirical evaluations on textual data from legal and medical domain with respect to both privacy and utility of the redacted texts. Our empirical results demonstrate that the texts anonymized with different redaction strategies achieve utility scores on par with the original texts, suggesting that anonymization with Redakto can be used for LLM tasks without substantial negative impact for the tasks we explored. 

---
# GenEx: A Graph-Based Representational Paradigm for SARS-CoV-2 Variant Detection via Codon Co-occurrence Networks 

**Authors**: Arefin Amin, Labiba Faiza Karim, M. Monir Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2608.18238)  

**Abstract**: Genomic analysis on viruses such as SARS-CoV-2 variants: Beta, Gamma, Delta, and Omicron is heavily dominated by classical bioinformatics methods, including Sequence Alignment, Phylogenetic Analysis, and Mutation Frequency Statistics. These approaches use pairwise codon or nucleotide distance matrices to analyze gene sequences, treating them as linear strings rather than capturing their complex contextual interdependencies. We proposed GenEx, a pipeline that converts raw gene sequences into codon co-occurrence graphs and extracts more than 25 graph features. Our two most prominent techniques for graph generation and feature extraction are MSCG (Multi-Scale Codon Co-occurrence Graph) and LAPCG (Linear-time Adjacency PMI Codon Graph). Using these algorithms, we treated codon sequences as structured symbolic vocabularies interpretable to codon co-occurrence graph analysis, a representational paradigm borrowed from computational linguistics. Another major contribution includes implementing a spectral graph feature extraction using Singular Value Decomposition (SVD), using the squared singular value ($\sigma^2$) instead of the traditionally used eigenvalue, which helped us to amplify the separation between dominant and subdominant spectral components, thereby enhancing inter-class separability in downstream classification. And to further demonstrate that our method works, we trained 23 benchmarked ML models against the latest SARS-CoV-2 variants, achieving remarkable results in detecting all SARS-CoV-2 variants. 

---
# On the Triangle Inequality for the Jaccard Distance in Arbitrary Lattices 

**Authors**: Costin Bădică, Amelia Bădică  

**Link**: [PDF](https://arxiv.org/pdf/2608.18194)  

**Abstract**: This paper presents new theoretical results on generalizing the Jaccard distance for lattices and real valuations. We demonstrate that when the valuation is strictly positive, monotone, and modular, the Jaccard distance satisfies the triangle inequality on arbitrary lattices, effectively generalizing earlier results that depended heavily on distributivity. Moving to relatively complemented distributive lattices (which safely drop the requirement for the global bounds found in Boolean algebras), we prove the triangle inequality holds as long as the valuation is positive, monotone, supermodular, and $\log$-submodular. Additionally, we adapt the symmetric-difference Jaccard formulation for submodular valuations to sectionally complemented distributive lattices. Shifting to necessary conditions, we prove that supermodularity is a strict requirement for the standard generalized Jaccard distance to operate as a valid metric. Finally, we map the practical value of relaxing these structural constraints to computational fields like quantum information theory, formal concept analysis, and machine learning, closing with a brief look at open mathematical problems. 

---
# Looped Language Models Improve Compositional Tool Calling 

**Authors**: Andrei Cristian Popescu, Haitz Sáez de Ocáriz Borde, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2608.18171)  

**Abstract**: Looped language models have shown promising results on reasoning benchmarks, yet their potential for agentic tool use remains largely unexplored. We study this question in compositional tool-calling settings, where models must coordinate multiple API calls, maintain intermediate state, and preserve dependencies across tool interactions. We evaluate native and retrofitted looped language models on API-Bank, BFCL, and NESTful, comparing looped and non-looped models trained under matched supervised fine-tuning recipes and varying recurrent depth at inference time. In controlled experiments, recurrent computation generally benefits compositional and dependency-aware tool use, while providing smaller and more model-dependent gains on isolated API invocation. Accuracy on multi-step tool use generally increases with recurrent depth; adaptive inference, however, achieves a more favorable compute-performance trade-off by allocating additional computation only when needed. Our results suggest that looped language models are a promising architecture for agentic systems that require reliable planning, coordination, and execution of compositional tool use workflows. 

---
# Adversarial Review: Structured Disagreement for Grounded Agentic Code Review 

**Authors**: Eric S. Qiu, Joyce Gill  

**Link**: [PDF](https://arxiv.org/pdf/2608.18167)  

**Abstract**: Early multi-agent LLM systems often used role-separated teams, yet scaling agent count yields diminishing returns on repository-level coding tasks. Recent alternatives treat agents as passive tools (subagents), yet this removes the benefits of agent interaction entirely. We study whether a subagent paradigm can support a middle ground: minimal agentic cooperation without the overhead of large multi-agent teams. We introduce Adversarial Review (AR), a minimal cooperative code-review protocol in which a main coding agent works with a reviewer and a critic agent. The reviewer evaluates code, while the critic audits the review through structured disagreement before the main agent edits. On LiveCodeBench, AR achieves the highest pass rate among tested methods, outperforming a five-agent baseline while using only three agents. On SWE-PRBench, naive AR exposes a false-consensus failure mode, where agents converge on agreement without sufficient evidence, but a single prompt iteration that adds disagreement explicitly achieves the highest F1 among tested methods. On SWE-bench Verified, AR also shows improvements over the baselines on repository-level coding tasks. Together, AR demonstrates that cooperative code review does not require many agents or complex communication structures: it requires that disagreement be minimal, structured, and evidence-grounded. 

---
# RDFdL: Integrating RDF with Differential Dynamic Logic 

**Authors**: Yuyang Li, Lukas Kubelka, Julia Butte, Tobias Käfer  

**Link**: [PDF](https://arxiv.org/pdf/2608.18165)  

**Abstract**: Knowledge graphs modeled in RDF are powerful for describing static knowledge, but they cannot capture or reason about the dynamic behavior of physical systems, e.g., systems described by differential equations, which is a critical gap for AI-driven cyber-physical systems. To solve this, we propose RDFdL, a framework that integrates RDF with Differential Dynamic Logic (dL) to represent and reason about both static knowledge and the continuous dynamics of physical systems. For the dynamic part, we syntactically represent differential equations and ranges in the state space in RDF and SHACL and provide semantics using a translation to dL. Linking RDF and dL through their shared foundation in first-order logic achieves a unique integration: verification results for safety and reachability properties in the dynamic logic domain become available as entailment to SPARQL queries over RDF data. We implement the pipeline using Apache Jena for ontology-driven RDF reasoning and KeYmaera X, the theorem prover for dL, and sketch its applicability in manufacturing. 

---
# Efficient Adaptation of LLMs for Hate Speech Detection in Low-Resource Languages: A Comparative Study on Roman Urdu 

**Authors**: Toneema Zubair, Muhammad Junaid Asif, Faisal Kamiran, Hafiz Hassan Saeed, Rana Fayyaz Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2608.18142)  

**Abstract**: It is challenging to detect hate speech in Low Resource Languages (LRLs) because of the absence of annotated data, the informality of its language structure, and the lack of standardized grammar. A good example of such a challenge is Roman Urdu which is broadly used by South Asians on social media and has a high variation while lacking contextually consistent spellings. The objective of this paper is to conduct a comprehensive assessment of Large Language Models (LLMs) for Hate Speech Detection (HSD) in Roman Urdu script and fine-tune these models using the Parameter-Efficient Fine-Tuning (PEFT) method called Low-Rank Adaptation (LoRA). To evaluate zero-shot inference, we benchmarked it against PEFT on different transformer models, including Mistral, LLaMA, Falcon, and multilingual BERT. Experiments are conducted on the PURUTT (Parallel Urdu and Roman Urdu Corpus for Toxic Comments and Transliteration) dataset with over 72,000 annotated comments. The results suggest that zero shot models perform moderately (F1 = 0.56), but updating a small fraction of the model trainable parameters improves the classification performance significantly (F1 > 0.93). Our results have shown that PEFT delivers outstanding performance alongside excellent computational efficiency, making it highly suitable for low-resource language processing tasks. 

---
# FraudBench: Stress-Testing Policy-Grounded Banking Agents Against Adaptive Fraud 

**Authors**: Dheeraj Mohandas Pai, Lu Xian  

**Link**: [PDF](https://arxiv.org/pdf/2608.18136)  

**Abstract**: Conversational agents now act for end users through tools while holding access to customer databases and internal policy documents that a caller can reach through dialogue alone. Banking is the clearest case: the same agent that answers a question can also change contact details, reset a PIN, or move money, so ordinary customer service is inseparable from authorization, fraud detection, and policy compliance. Existing financial-fraud benchmarks classify static transactions or messages, and general agent-safety benchmarks target prompt injection or generic harmful use; none test whether a policy-grounded banking agent safely acts when a caller manipulates identity, authorization, and trust over a conversation. We introduce FraudBench, an executable benchmark built on the $\tau^2$-bench dual-control framework and the $\tau$-Knowledge banking environment. Both the agent and the simulated caller act through tools over shared, mutable account state, and the agent may grant the caller access to selected tools; the environment exposes a 698-document internal policy corpus that the agent must retrieve from. FraudBench contains 150 authored adversarial scenarios; a frozen public set of 107 (90 across ten fraud mechanisms plus 17 chained adaptive attacks) is used for all reported runs, with 43 further chained attacks held out. Safety is history-dependent: single-control tasks satisfy every precondition but one, and adaptive attacks make a later, locally valid request unsafe because of an earlier probe, admission, or failed attempt. Each scenario is annotated with observable evidence, prohibited actions, safe dispositions, and intervention points. A preliminary single-trial evaluation of four agents on the 107 graded tasks yields attack-security between 49\% and 65\%, with money-mule and first-party fraud the most common cross-model weaknesses. 

---
# Improving Rural Medication Safety with AI: A Scoping Review 

**Authors**: Jeong-ah Kim, Muhammad Ashad Kabir, Daniel Terry, Maryam Rouhi  

**Link**: [PDF](https://arxiv.org/pdf/2608.18135)  

**Abstract**: Introduction: Medication errors (MEs) represent a significant threat to global healthcare systems, contributing to patient harm. Introducing artificial intelligence (AI) in rural healthcare enhances patient safety. The aim is to explore the applications and effectiveness of AI technologies in enhancing patient safety and reducing medication errors in rural health settings.
Methods: A scoping review was conducted through a systematic literature search spanning 2012 to 2025 across multiple databases, including EBSCohost, Emcare (Ovid), MEDLINE, and the ProQuest Consumer Health Database. Twelve primary studies from nine different nations were examined. Data were analysed thematically to obtain insights on AI interventions across the medication process.
Results: AI technologies have been integrated into every stage of medication management, right from prescribing and dispensing to administration and post-administration monitoring. Four key themes came to light: (1) the various types of AI being utilised (like Clinical Decision Support Systems, Machine Learning, Natural Language Processing, and smart pumps); (2) the phases of the medication process that are affected; (3) how effective these technologies are in minimising errors and boosting workflow safety; and (4) rural-specific challenges including infrastructure, staff training, system integration, and alert fatigue. Several studies have demonstrated that machine learning-based surveillance improves incident detection and reduces prescribing and transcription errors by an impressive 34% to 80%. Barriers included lack of governance frameworks, financial limitations, and clinician resistance, which still present major obstacles.
Conclusion: In rural healthcare, AI technologies hold great potential for enhancing pharmaceutical safety. They can allow data-driven monitoring, automate processes, and offer clinical decision assistance. 

---
# Optimized Fuzzy Logic Approach with the IEEE Key Gas Method for Diagnosing Power Transformer Faults Using Dissolved Gas Analysis 

**Authors**: Kim-Anh Nguyen, Huy Hoang Le, Ba Tu Phung  

**Link**: [PDF](https://arxiv.org/pdf/2608.18133)  

**Abstract**: Reliable transformer fault diagnosis is essential for maintaining power system stability. The IEEE Key Gas Method (KGM), a widely utilized approach in Dissolved Gas Analysis (DGA), exhibits limitations in addressing ambiguous data and ensuring high diagnostic accuracy. This study presents An enhanced model combining Fuzzy Logic with the IEEE Key Gas Method (FL-KGM) that introduces refined membership functions, optimized fuzzy rule sets, and a novel separation of CO and CO2 to eliminate diagnostic inconsistencies. By leveraging multidimensional gas ratio analysis and an adaptive classification framework, FL-KGM delivers superior fault identification and classification. Experimental validation utilizing real-world datasets demonstrates that FL-KGM achieves up to 98.6% accuracy, significantly outperforming KGM and other FL-based approaches. These findings elucidate the potential of FL-KGM in advancing transformer monitoring, enabling intelligent fault detection, and enhancing predictive maintenance strategies in modern power systems. 

---
# Safety Alignment Illusion: The Cross-Lingual Safety Gap in LLMs 

**Authors**: Namya Bhatnagar  

**Link**: [PDF](https://arxiv.org/pdf/2608.18131)  

**Abstract**: Current safety alignment training for Large Language Models (LLMs) are heavily English-centric. When such safety filters fail for non-English languages, the consequences are immediate and user-facing: voice assistants and spoken dialogue systems may produce stereotype-reinforcing outputs, bypassing the standard English-focused safety alignments and propagating harmful bias to non-English speaking communities. For spoken language technologies deployed across India's linguistically diverse population, this represents a critical failure mode. To address this cross-lingual gap, we introduce INCLUDE (Indian Cultural Lens for Understanding and Detecting Embedded Biases), a multilingual evaluation benchmark designed to quantify Indian-centric socio-cultural biases. INCLUDE consists of 2,604 prompts spanning six prompt languages: English, Hindi, Bengali, Marathi, Tamil, and Hinglish (Hindi-English code-mix). We evaluate ten open- and closed-source LLMs against this benchmark, analyzing 14,988 bias scores. Our statistical results reveal two key findings. First, Bengali yielded the highest average bias score in open-source models. Second, English demonstrated a notable reversal, producing the lowest bias in open-source models but the highest bias in closed-source models. 

---
# Position: AI Leaderboards Are Underserving the Global South: A Case Study from India 

**Authors**: Sourav Banerjee, Saikat Saha  

**Link**: [PDF](https://arxiv.org/pdf/2608.18117)  

**Abstract**: This position paper argues that AI leaderboards are structurally ill-suited to serving the Global South because they lack independent governance, conflict-of-interest policies, and mechanisms for metric evolution. The barrier is not missing data; high-quality regional benchmarks already exist: IndicSUPERB, MILU, and LAHAJA for India; IrokoBench for Africa; AlGhafa for Arabic. The barrier is institutional design. Global leaderboards do not include these benchmarks, and no governance mechanism compels them to do so. Commercial pressure corrects leaderboard failures when paying customers in the Global North are affected. The Global South lacks equivalent leverage. Without governance, failures affecting Hindi, Swahili, or Arabic speakers persist indefinitely as documented but unaddressed gaps. Using India as a case study (1.4 billion people, 22 scheduled languages, high-quality benchmarks, but no trusted aggregation), we report findings from a consultation with 58 AI practitioners showing consistent preference for formal governance and disclosure-based conflict management. The solution is not more data but better institutions: regional leaderboards with independent governance from the start. 

---
# Solving Is Not Drawing: A Benchmark for Diagrammatic Reasoning in Olympiad Geometry 

**Authors**: Hsien Xin Peng, Anthony Kim, Alvin Li, Calvin Supasanya, Shivank Garg, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18111)  

**Abstract**: Foundation models such as GPT and Claude now solve olympiad-level mathematics with remarkable proficiency, so much so that geometry problem solving has become a standard proxy for their mathematical reasoning. Yet solving a geometry problem and drawing the figure it depends on are not the same skill: progress often hinges on a faithful diagram with the right auxiliary constructions and incidences, and it is unclear that a model which reasons its way to the answer can also produce one. A growing collection of benchmarks, including MathVista, and MathVerse, measures whether models reach the correct answer, but to our knowledge, none isolate the distinct ability to construct the diagram itself, leaving this capability unmeasured. We introduce an open-source benchmark that targets this gap: 954 self-contained olympiad geometry problems, with a 297-problem hard subset, each paired with its solution and a human-authored, high-fidelity diagram in renderable Asymptote code, together with a suite of text-, code-, image-, VLM-, and constraint-based metrics for what we term diagrammatic reasoning. Evaluating current foundation models reveals a pronounced gap between solving and drawing: their diagrams are markedly less faithful, with an average compile success rate of only 36.14\%. Strong mathematical reasoning, we find, does not imply the ability to construct accurate geometric diagrams. Our benchmark and dataset can be accessed at this https URL. 

---
# Emergence of Agentic AI: A Review on Evolution, Background, Working Principles, Applications, Adoption Factors, and Future Research Directions 

**Authors**: AKM Bahalul Haque, Al Amin Islam Ridoy, Mohammad Rayhan, Ivan Porres  

**Link**: [PDF](https://arxiv.org/pdf/2608.18110)  

**Abstract**: Agentic AI is gaining new insights and advancements in the field of Artificial Intelligence, fostering significant potential to enable rapid transformation across various this http URL rapid advancement and the potential to revolutionize various domains advocate the need for a deeper understanding and firm grasp of the technology. Moreover, an investigation into state of the art research directions in agentic AI needs to be conducted to comprehensively assess the potential scope for improvement and this http URL, to address these objectives, a comprehensive review can provide researchers and practitioners with valuable insights into the current state and future research scopes of agentic this http URL, this work considers the recently published scholarly contributions in agentic AI across various domains and discusses the fundamentals and working principles of Agentic AI, traces the historical and theoretical evolution of agency in artificial systems, explores and discusses Agentic AIs architecture, working principles, and functionalities, explores real-world applications of Agentic AI across various domains, analyzes the research findings, identifies current challenges, and discuss potential future research directions, and proposes a comprehensive framework of stakeholders intention to use and adopt Agentic AI with the help of proposed system quality this http URL, this systematic review provides researchers and practitioners with a comprehensive understanding of Agentic AI, its current developments and applications, highlights key research gaps, and outlines future research directions. 

---
# Self-Evolving Agents as Dynamic Graph Transformation: A Survey and New Perspective 

**Authors**: Yuanyuan Xu, Wenjie Zhang, Yin Chen, Xuemin Lin, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18104)  

**Abstract**: Large language model (LLM)-based agents are increasingly becoming self-evolving systems that persist across interactions, maintain memories, use tools, acquire skills, refine workflows, and coordinate with other agents. These capabilities make agent states structural and dynamic: entities, relations, attributes, dependencies, and execution structures change with new evidence, feedback, and environmental conditions. Existing graph-agent surveys typically treat graphs as support structures for agent functions rather than as evolving substrates, while self-evolving-agent surveys focus on agent-level mechanisms and rarely discuss graph topology evolution. Thus, the coupling between evolving agent state and dynamic graph topology remains underexplored. This survey connects these two research lines by framing \textit{agent evolution as dynamic graph transformation}. We model agent state as a dynamic graph, where memories, tools, skills, workflows, and inter-agent relations are represented as typed nodes, edges, and subgraphs updated through schema-constrained rewrites. Based on this formulation, we organize existing dynamic-graph-based methods for self-evolving agents into four taxonomies: node/feature evolution, edge/topology evolution, subgraph activation, and cross-component co-evolution. Building on this taxonomy, we propose dynamic graph learning as reusable infrastructure for self-evolving agents and map nine dynamic-graph-learning subfields to agent-evolution capabilities, discussing their adaptations and possible failure modes. Finally, we discuss five types of graph-aware evaluation and governance protocols from a dynamic-graph perspective, which complement end-task evaluation. The goal is to provide a compact structural lens for designing and governing self-evolving agents. 

---
# FinSkillBench: Evaluating AI Agents and Domain Skills for Investment Management 

**Authors**: Jermyn Zhen Yong Bek, Zhuang Qiang Bok, Zhongtian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.18099)  

**Abstract**: Investment management is a high-stakes domain in which agentic AI systems must do more than generate plausible text. They must retrieve point-in-time data, assemble correct computational inputs, invoke specialized methods, and produce auditable structured outputs. We introduce FinSkillBench, an evaluation suite designed to measure whether language model agents can effectively use financial domain skills to solve investment management tasks. The benchmark spans three domains, portfolio construction, risk management, and fundamental analysis, and includes 12 subtasks with 2,603 task episodes.
Each episode provides point-in-time inputs, hidden ground truth, and a task-specific this http URL compare three conditions: no skill, curated skill packages consisting of procedural documents and executable components, and self-generated skills in which the agent writes and reuses its own procedures within an episode. Across 9 models and a large-scale evaluation, curated skills consistently improve performance, raising mean scores from 0.366 to 0.528, with the largest gains in portfolio construction and risk management.
In contrast, self-generated skills provide little benefit despite higher computational cost. An independent evaluation using a separate agent framework (Hermes Agent, 8 models, 5,280 episodes total) reproduces the directional pattern across all three domains, with the magnitude of skill effects varying by subtask and harness.
These results showthat in investment management agents, access to reliable procedural skills can be as important as model choice, while naive self-generation of skills is often ineffective. We release the benchmark, evaluation tools, curated skill packages, and full trajectories to support further research. 

---
# Position: Multi-Agent Systems Should Prioritize Concurrency Control 

**Authors**: Xin Yang, Letian Li, Zimo Ji, Terry Jingchen Zhang, Wenyuan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18092)  

**Abstract**: LLM-based multi-agent systems (MAS) promise scalable collaboration, yet adding agents often reduces reliability. This position paper argues that many MAS failures are fundamentally concurrency control problems: agents concurrently read and write shared state, and long LLM inference windows amplify the risk of stale reads, lost updates, and inconsistent outcomes. Failure modes commonly attributed to coordination or communication breakdowns can be mapped directly onto classical concurrency anomalies. We contend that MAS frameworks should address these failures through explicit concurrency control mechanisms: conflict detection, isolation guarantees, and structured access to shared resources. Concurrency control should be a first-class design concern, not an afterthought. 

---
# A Metamorphic Artificial Age Score Decision-Support Prototype for Flight-Log-Based Drone Propeller Health Monitoring 

**Authors**: Seyma Yaman Kayadibi  

**Link**: [PDF](https://arxiv.org/pdf/2608.18088)  

**Abstract**: Drone propeller faults can create safety and reliability risks when their effects are distributed across multiple flight-log channels rather than appearing as a single diagnostic signal. This paper proposes a Metamorphic Artificial Age Score (AAS) decision-support prototype for flight-log-based drone propeller health monitoring. Using selected historical real flight logs from the 2024 DronePropA public dataset, the framework computes six health-related indicators from raw MATLAB matrices: trajectory tracking error, attitude instability, thrust-command burden, motor-command imbalance, ESC-command instability, and battery-level stress. These indicators are normalized relative to a healthy baseline and evaluated through candidate scoring policies, metamorphic adequacy relations, and a redundancy-adjusted AAS formulation. In this context, AAS is used as a structural policy-adequacy and burden measure rather than as a chronological age measure. A controlled retrospective evaluation was performed using one healthy baseline and three defective propeller cases under the same speed profile and trajectory. The healthy case was assigned to routine monitoring. The Severity 1 case was dominated by ESC-command instability and assigned to maintenance review. The Severity 2 case reached maximum motor-command and ESC-command burden, while the Severity 3 case reached maximum trajectory tracking error; both triggered mandatory inspection. The results show that propeller fault effects may appear through different operational channels, supporting the need for a multi-indicator decision-support layer for post-flight maintenance prioritization and autonomous-system oversight. 

---
# Position: Current Model Cards Are Insufficient for Downstream Governance of Open-Weight Foundation Models 

**Authors**: Sungwon Chae, Keonwoo Kim, Hoki Kim, Jaeyeon Ju, Sangchul Park  

**Link**: [PDF](https://arxiv.org/pdf/2608.18086)  

**Abstract**: The growth of open-weight foundation models (OWFMs) has prompted the AI community to re-evaluate strategies for effective downstream governance. Although model cards have been widely adopted as transparency artifacts in model repositories, existing frameworks often fail to adequately inform downstream developers and users about the distinct safety challenges posed by OWFMs. This position paper analyzes 500 model cards hosted on Hugging Face and argues that effective governance of OWFMs requires a multi-layered approach integrating three complementary components: (i) model cards, (ii) acceptable use policies (AUPs), and (iii) licenses. To motivate this claim, we identify a safety gap left by existing regulatory approaches, including model heritage, alignment provenance, and empirically observed behaviors, through an analysis of model cards with safety-critical information. We further argue that standard open-source licenses (OSLs) are not well suited for OWFMs and may weaken the enforceability of AUPs. Building on these observations, we outline directions for evolving model cards, AUPs, and licenses into integrated safety artifacts to enable a more comprehensive governance framework that coherently integrates informational, normative, and legal dimensions. 

---
# Position: Behavioral Systems Require Behavioral Tests 

**Authors**: Manuel Cherep, Nikhil Singh, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2608.18081)  

**Abstract**: Artificial agentic systems increasingly operate as behavioral systems by interacting with dynamic environments, pursuing goals, and adapting over time. Yet, current evaluation methods largely focus on performance outcomes, not the underlying behavioral processes that produce them. This paper argues that AI agents must be evaluated like other behavioral systems: through systematic observation, perturbation, and interpretation of their actions. We draw on lessons from the behavioral sciences to motivate this position, and propose a research agenda focused on developing rigorous behavioral tests. These include methods for recovering decision strategies from action sequences, constructing environments that isolate behavioral differences, and probing emergent dynamics in multi-agent systems. Taken together, these directions offer a roadmap for developing a science of AI behavior. 

---
# Large Language Models in Mental Health: A Systematic Review of Applications, Innovations, and Ethical Challenges 

**Authors**: Yisong Chen, Yifan Gao, Sijing Yu, Chuqing Zhao, Yang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18080)  

**Abstract**: We present a review on the applications of large language models (LLMs) in health, e.g., social media analysis, clinical conversational agents, therapy support tools, prompt engineering, multimodal learning, and ethical considerations. We integrate findings from interdisciplinary studies utilizing diverse data sources such as social media posts, electronic medical records, and multimodal inputs to enable early detection of depression, suicide risk assessment, personalized therapy support, and psychoeducational content generation. Our review highlights advancements in LLM models and annotation strategies that enhance interpretability and clinical relevance, while we also emphasize the critical role of prompt engineering for domain adaptation. We also discuss emerging multimodal fusion techniques integrating text, speech, and sensor data for improved mental health diagnosis and monitoring. Finally, we address ongoing ethical, sociotechnical, and regulatory challenges, and advocate frameworks to ensure safe, equitable, and accountable deployment of LLMs in real-world mental health care. 

---
# Position: Profiling Game Worlds by Transition Complexity 

**Authors**: Lele Cao  

**Link**: [PDF](https://arxiv.org/pdf/2608.18079)  

**Abstract**: Game world modeling (GWM) and reinforcement learning (RL) are often confounded because research papers rarely quantify how difficult the underlying transition prediction problem is at the declared interface (pixels/tokens/latents with finite history). We propose the Transition Complexity Profile (TCP): a small, reproducible set of metrics that characterizes an environment's (or gameplay dataset's) induced transition kernel by (i) intrinsic one-step branching, (ii) interaction-induced uncertainty and opponent influence when observable, and (iii) temporal/spatial dependency span via standardized probe curves. TCP is reported with an explicit reference distribution, protocol stochasticity, and a versioned measurement budget (sampling/resampling and fixed probe compute), enabling comparable numbers across benchmarks. We outline how common game families and modern "neural game engine" domains populate this landscape and call for TCP to become standard benchmark metadata and a required statistic in GWM and RL papers. 

---
# Position: Collusion Risks Among AI Reasoning Agents Justify Certification Requirements for Making Market Decisions 

**Authors**: Matthew Riemer, Tommaso Tosato, Amin Memarian, Maximilian Puelma Touzel, Glen Berseth, Irina Rish, Guillaume Dumas  

**Link**: [PDF](https://arxiv.org/pdf/2608.18078)  

**Abstract**: This position paper argues that AI agents with chain-of-thought reasoning capabilities are predisposed to exhibit collusive behavior and should be required to obtain behavioral certification before making decisions that affect economic markets. This is because integrating these agents into society could collapse the legal evidentiary distinction between competition and collusion among independent firms without eroding the economic harm distinction. Experiments with DeepSeek-R1 agents in the Bertrand oligopoly pricing domain reveal a tendency towards tacit collusion that persists even when humans prompt the agents not to collude. We further show that the chain-of-thought of these agents can be steered toward either extremely collusive or highly competitive behavior in a way that is not semantically detectable by another LLM analyzing the reasoning traces. As a result, deploying reasoning agents for market decisions leads to collusive economic outcomes without any evidence of conspiracy or intent. Thus, certification based on observed behavior in representative situations is necessary to prevent collusion. We provide preliminary evidence that such agents can be steered in a generalizable way toward efficient competitive equilibria. However, developing a comprehensive behavioral certification will be required before these models can be deployed in real-world markets while ensuring their stability and efficiency. 

---
# SPADE: Self-Play in Adaptive Synthetic Executable Environments 

**Authors**: Bo Liu, Simon Yu, Yiding Jiang, Ao Qu, Andrew Zhao, Zichen Liu, Junsu Kim, Zijian Zhou, Seungone Kim, Tongzheng Ren, Mickel Liu, Hanfei Yu, Zhaorun Chen, Weiyan Shi, Paul Pu Liang, Luke Zettlemoyer, Yejin Choi, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2608.19197)  

**Abstract**: Continuous self-improvement requires an ever-expanding pool of self-generated, diverse, adaptive goals. For language agents, existing training environment pools (hand-curated, statically synthesized, or frozen-verifier) keep the goal distribution fixed as the learner scales. We introduce SPADE (Self-Play in Adaptive Synthetic Executable Environments), a self-play RL framework in which a single LLM plays two roles: an Environment Designer that writes complete, long-horizon training environments as executable code with an OpenAI Gym-style reset()/step() interface, and a Reasoning Agent that learns to act in them. Each is a stateful, multi-turn environment (state transitions, reward functions, and verification code), so one interface spans reasoning problems and multi-step agentic tool use. The Reasoning Agent's regret is estimated using the gap between its reward with and without privileged hints; in optimizing this regret signal the Environment Designer learns to target environments at the edge of the agent's capabilities while keeping them feasible. Through extensive experimentation, we find several components critical to success: grounding the Environment Designer on documents sampled from a large pretraining corpus, and giving it an accumulated environment memory. Scaling to 30B-parameter models, SPADE improves over the strongest fixed-environment baseline by +5.3 on average across eight held-out math, science, code, and reasoning benchmarks, and lifts the tool-use setting by +5.7 on BFCL-v4 multi-turn and +13.9 on ACEBench-Agent; on the games setting, the margin over the strongest baseline grows with model scale. By making environment design itself a learnable component, SPADE takes a concrete step toward open-ended self-improvement. 

---
# ADEPT: Accelerating Dexterity via Pre-Training and Post-Training using Reinforcement Learning 

**Authors**: Jayjun Lee, Jessica Yin, Asif Rana, Nicholas Blauch, Sam Mady, Mohak Bhardwaj, Nima Fazeli, Nathan Ratliff, Karl Van Wyk, Ankur Handa  

**Link**: [PDF](https://arxiv.org/pdf/2608.19182)  

**Abstract**: We introduce Accelerating Dexterity via Pre-Training (ADEPT), a large-scale reinforcement learning (RL) framework for learning sim-to-real transferable dexterity across high degree-of-freedom (DoF) robot embodiments that can solve long-horizon tasks directly from raw visuo-tactile perception. ADEPT pretrains a dexterous policy on a generic object reposing task, then post-trains downstream policies with this pretrained behavior as a prior. ADEPT enables learning new behaviors that are otherwise difficult to discover from scratch on multi-fingered robots and avoids learning the same set of skills over again for every new downstream task. The pretrained policy zero-shots the reposing phase of downstream tasks, but naïve RL fine-tuning rapidly degrades this capability during transfer. We address this with a stable post-training recipe combining behavior-cloning distillation, critic warm-up, and conservative on-policy updates. To safely exploit the full kinematic dexterity, we introduce a joint-space Geometric Fabric that mediates between the RL policy and the robot. We distill post-trained teachers into perceptive students that zero-shot sim-to-real transfer on two embodiments: a 23 DoF Kuka-Allegro with two RGB cameras, and a 29 DoF Flexiv-Sharpa with two RGB cameras and five vision-based tactile sensors, and can solve long-horizon tasks from challenging initial states with dexterity at human-level speed. 

---
# Beyond Teacher Likelihood: Group-Calibrated On-Policy Distillation for Long-Context Reasoning 

**Authors**: Zhu Zhang, Jixun Wang, Xiaoang Xu, Xiaorong Wang, Zihan Zhou, Zhiyuan Wang, Shuo Wang, Chaojun Xiao, Yuezhi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.19181)  

**Abstract**: On-policy distillation (OPD) trains a student on its own responses using dense token-level guidance from a stronger teacher. In long-context tasks, however, token-level teacher support can favor locally plausible responses that omit evidence distributed across the input or violate global task constraints. Task-specific verifiers, in contrast, evaluate task completion at the response level and may return graded rewards that reflect partial success. We diagnose this mismatch on fixed responses from two representative long-context evidence-aggregation tasks. Across longer input ranges, trajectory-level OPD scores become progressively less aligned with verifier rewards, indicating teacher-verifier disagreement. Motivated by this observation, we introduce Group-Calibrated On-Policy Distillation (GC-OPD). GC-OPD separately normalizes verifier rewards and trajectory-level OPD scores within each rollout group and uses their difference as a signed teacher-verifier disagreement residual. Relative-advantage-based credit assignment (RACA) distributes this trajectory-level residual across tokens according to their relative OPD advantages while preserving the original OPD signal. Across five long-context benchmarks, post-training with GC-OPD raises the five-benchmark averages of the official Qwen3-4B and Qwen3-8B checkpoints from 29.08 to 40.47 and from 35.12 to 44.65, respectively. Vanilla OPD reaches 39.31 and 43.56 under the same setup. Controlled ablations show that the signed residual is more effective than either an additional OPD-derived term or direct group-normalized verifier reward addition, while RACA further improves over uniform token allocation. Together, these results demonstrate that group-relative residual calibration can incorporate verifier outcomes without discarding dense token-level guidance. Code is available at this https URL. 

---
# Finetuning Strategies for Querying Sounds by Vocal Imitation 

**Authors**: Aditya Bhattacharjee, Christos Plachouras, Sungkyun Chang, Emmanouil Benetos  

**Link**: [PDF](https://arxiv.org/pdf/2608.19174)  

**Abstract**: This technical report describes our winning submission to the AES AIMLA 2025 Challenge on querying sound effects by vocal imitation. We investigate two complementary fine-tuning strategies: contrastive learning with a frozen, pretrained CED encoder, and joint contrastive-triplet learning with semi-hard negatives using a MobileNetV3 encoder. This report has been updated for posterity to include details released after the challenge. 

---
# Interpretable AI predicts a 2026 summer dry anomaly in central China 

**Authors**: Anran Wang, Wen Shi, Yong Luo, Jianbin Huang, Lijuan Chen, Junhu Zhao, Weixin Jin, Huihui Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2608.19163)  

**Abstract**: Seasonal precipitation anomalies are largely regulated by atmospheric circulation, which dynamical models predict with greater reliability than precipitation itself. Here, we employ a deep learning model that translates dynamical circulation predictions into precipitation estimates. Predictions initialized from March to May consistently indicate a dry anomaly over central China in summer 2026. Retrospective evaluations revealed higher predictive skill in the analogue years, which also tended to feature central equatorial Pacific warming persisting from the preceding winter into summer. This warming favors an anomalous cyclonic circulation over the western North Pacific-South China Sea-South China region, which induces northerly winds and moisture divergence that jointly suppress rainfall over central China. Supporting this mechanism, layer-wise relevance propagation (LRP) independently identifies these northerly winds as the dominant driver of the prediction among all model inputs. Perturbation tests supported this attribution: removing LRP-identified features effectively eliminates the dry anomaly. Our framework thus provides physically interpretable explanations for AI-derived regional climate projections, facilitating evidence-based assessment before observational data become available. 

---
# Pre-Compiled Pipeline Shards for Distributed LLM Inference on Intel AI PC Fleets 

**Authors**: Tate Berenbaum, Muthaiah Venkatachalam  

**Link**: [PDF](https://arxiv.org/pdf/2608.19147)  

**Abstract**: Modern Intel AI PCs ship capable integrated GPUs and NPUs with 16+ GB of unified memory, and they spend considerable time idle. That is not enough memory to fit a large model such as a 70B-parameter LLM. We show that a handful of AIPCs, working together over an ordinary network, can serve models beyond the capability of any single one. We use pipeline parallelism: a model is split by layer into per-stage shards, each pre-compiled into an OpenVINO graph, so that every machine runs one shard and passes activations to the next. Three techniques make this fast enough to be useful. First, we recover the speed of the unsplit model: a naive per-stage export runs well below monolithic inference because it misses an OpenVINO GPU optimization, and injecting a beam_idx Gather into each shard triggers that optimization (the IndirectKVCache fusion) and brings the shards to parity. Second, we leverage speculative decoding on stateful OpenVINO models. Third, the pipeline serves several users at once by interleaving their requests across the stages, each request carrying its own cache (micro-batching). Together, a two-node Llama 3.1 8B INT4 pipeline serves two concurrent users at 1.79x the single-user throughput of the unsplit model on the same hardware, and the gap widens under simulated wide-area latency. The same design scales to a 70B model that no single fleet member can hold: a four-node deployment of Lunar Lake AI PCs on Intel Tiber Cloud serves a single user at interactive speed, with output token-for-token identical to the same four-node pipeline decoding without speculation. Code, raw benchmark logs, and reproduction scripts ship as a self-contained package at this https URL (in the top-level reproduction/ directory). 

---
# Leaf Values as Coordinates: Exact Contrastive Explanation for Gradient-Boosted Ensembles 

**Authors**: Emanuele Luzio  

**Link**: [PDF](https://arxiv.org/pdf/2608.19127)  

**Abstract**: A gradient-boosted ensemble predicts by summing one leaf value per tree. Read
those values as coordinates rather than as intermediate results, and every
instance becomes a point in R^M on which the model acts linearly: the score is
the sum of the coordinates.
This small change of view makes contrastive explanation exact. The difference
between two instances is a vector that is identically zero wherever they share
a leaf, so the gap between a rejected applicant and an accepted one is carried
by a handful of coordinates, each traceable to a real split in a real tree.
Nothing is fitted, sampled, or assumed additive in features -- the additivity
is already there, in the right space.
We build a recourse method on this representation and evaluate it on five
tabular datasets under repeated cross-validation. Its recommendation
reconstructs the model's own decision to 6.2 x 10^-15, so an auditor can
re-check the arithmetic without the model. On the credit datasets it is
Pareto-non-dominated on effort against realism. And when recommendations are
restricted to changes the subject could actually make -- not their age, not a
settled delinquency -- it retains 58% of its validity where the strongest
baseline retains 41%, a distinction the standard evaluation cannot see because
it never asks whether a recommendation can be carried out. 

---
# Intercepting the Kangaroo: Experimental Astrolinguistics with Constructed Lexicons, Active Probing, and Large Language Models as Informants and Hypothesis Proposers 

**Authors**: Francesco Cordella, Mauro Cappelli  

**Link**: [PDF](https://arxiv.org/pdf/2608.19124)  

**Abstract**: Astrolinguistics -- communication with minds that categorize reality differently from ours -- has been purely speculative since Freudenthal's Lincos (1960). We make it experimental. Two language models with deliberately incompatible constructed lexicons (one encoding shape, color, and motion; the other fusing color with motion, encoding parity, and lacking shape) serve as informants with complete ground truth, while a fully scripted orchestrator translates between the two category systems. The central failure mode is the kangaroo effect: the silent attachment of a word to the wrong referent -- Quine's indeterminacy of translation, operationalized. Across 400+ simulated and live runs, a protocol combining cross-situational elimination, pre-registered predictive probes, active scene selection, a stricter recovery round, and quarantine produced no undetected mistranslations under the tested conditions and exceeded a passive baseline's coverage (d = 0.62). Injected kangaroo traps defeated naive ostension and pure statistical learning in 100% of runs, while the full protocol intercepted every decoy and, where discriminating evidence is ontologically unavailable, declared Quinean equivalence classes instead of guessing. Under informant noise it degrades gracefully: zero kangaroos persist up to 2% per-word noise; at 10% the protocol predominantly abstains rather than errs. Finally, words outside the scripted hypothesis space (a history-dependent relational term and an XOR contextual homonym) are recovered by a generate-and-test loop in which an LLM proposes rules and the script verifies them: coverage scales with proposer capability (0% -> 18% -> 72% -> 100%) while undetected mistranslations stayed at zero throughout. In the tested conditions, correctness is a property of the protocol; coverage is a property of the instruments. 

---
# PGFS++: Molecular Property Improvement under Synthesis and Diversity Constraints 

**Authors**: Boqiao Zhang, Godbless James, Sai Krishna Gottipati, Andrew Fitzgibbon  

**Link**: [PDF](https://arxiv.org/pdf/2608.19121)  

**Abstract**: Improving molecular properties, such as drug-likeness or binding affinity, is a recurring task in early-stage drug discovery. However, molecules optimized in an unconstrained chemical space have limited practical value if they cannot be synthesized. Policy Gradient for Forward Synthesis (PGFS) is a synthesis-aware reinforcement learning method for molecular improvement, but its use of reactant embedding prediction makes reactant selection indirect, which, as we show, limits learning effectiveness. We first develop PGFS+, in which reaction templates and second reactants are represented by trainable embedding lookup tables. Combined with a more effective scoring function and RL algorithm, PGFS+ significantly improves the desired property. However, it exposes a reward-hacking failure mode: a powerful reactant search can map diverse input molecules to the same high-reward magnet molecule, improving the reward while collapsing the output diversity. We therefore introduce PGFS++, a synthesis-aware reinforcement learning framework for input-specific molecular improvement. Given an input molecule, PGFS++ treats it as the start of a forward-synthesis trajectory, applies learned reaction templates with compatible in-stock building blocks, and produces a molecule with improved target properties, an explicit synthesis route, and structural similarity to the input. Experiments on molecular improvement tasks show that PGFS++ improves target properties while preserving high output diversity. 

---
# Discretizing Continuous Time Series for Imputation with Masked Diffusion Training 

**Authors**: Dongbin Kim, Seungyun Lee, Geonwoo Shin, Jaewook Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.19119)  

**Abstract**: Time series imputation is a crucial area for reliable time series analysis, yet it remains challenging due to the complex temporal dynamics and noise of real-world data. Existing approaches, however, exhibit two limitations: missing and observed values are embedded within the same representation space without explicit structural separation, and continuous diffusion-based methods are trained to predict added noise rather than the original signal. To address these, we propose the Masked Diffusion Time-series Imputation Model (MDTIM), which leverages the training paradigm of masked diffusion model for imputation tasks. The MASK token is structurally orthogonal to valid observations, and the model directly predicts the original values, naturally aligning both the representation and the learning objective with the imputation task. To bridge the gap between discrete masked diffusion and the continuous, ordinal nature of time series, we further introduce Stochastic Discretization, which maps continuous values to ordinal-aware tokens while preserving continuous dynamics. Our experiments on diverse benchmarks confirm that MDTIM achieves superior robustness and scalability, consistently outperforming state-of-the-art deterministic and generative baselines across various missing scenarios. 

---
# Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation 

**Authors**: Huan-ang Gao, Haohan Chi, Yong Yan, Shiyuan Feng, Hanlin Wu, Zheng Jiang, Bingxiang He, Wei-Ying Ma, Ya-Qin Zhang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.19098)  

**Abstract**: Multi-teacher on-policy distillation (M-OPD) has emerged as a promising paradigm for consolidating domain-specialized reinforcement learning (RL) experts into a single generalist student via dense, token-level reward supervision. Despite its practical success, the optimization dynamics governing multi-teacher capability integration remain poorly understood, and open, rigorously reproducible recipes are conspicuously lacking. In this work, we establish a controlled M-OPD benchmark on SmolLM3-3B-Base with oracle routing, isolating capability integration from routing ambiguity. Our investigation reveals a pronounced capability integration gap: standard M-OPD captures only 35.6% of the available headroom relative to a domain-routed oracle ensemble, with concise tasks such as instruction following suffering severe degradation and premature stagnation. Crucially, we show that this failure stems not from gradient conflict, but from a severe misallocation of the token-level optimization budget. This pathology is driven by three orthogonal factors: structural sequence-length disparities across domains, dynamic convergence drift due to non-uniform learning rates, and multi-step reward staleness from asynchronous policy updates. To resolve these imbalances, we introduce Open-MOPD, a principled framework incorporating token-share balancing, gap-aware dynamic budget allocation, and student reward refresh. Together, these mechanisms systematically restore cross-domain balance, elevating headroom recovery from 35.6% to 83.4% in a single deployable student. We fully open-source our end-to-end post-training recipe, training trajectories, and evaluation suites on an academically accessible hardware budget. 

---
# Detecting Backdoors in Object Detection via Pre-NMS Prediction Distribution Shift 

**Authors**: Longtian Wang, Zhengyu Zhao, Chenhao Lin, Le Yang, Shiwei Wang, Yuhan Zhi, Xiaofei Xie, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.19088)  

**Abstract**: Object detection models deployed in safety-critical applications remain vulnerable to backdoor attacks that cause targeted misbehaviors when a hidden trigger is present. Existing detection methods either rely on trigger inversion or exploit architecture-specific assumptions, and critically, representative existing methods fail to generalize reliably to scene-level attacks, where a single trigger induces anomalous behavior across all objects in the scene simultaneously. We present DistScan, a backdoor detection framework based on a simple but previously unexploited observation: backdoor injection systematically shifts a model's pre-NMS prediction class distribution away from its training class frequencies, even on clean inputs without any trigger present. DistScan aggregates intermediate class predictions over a clean validation set and flags a model as backdoored if the resulting distribution deviates significantly from the training class frequencies, requiring no model weight access, no trigger knowledge, and no additional training. Extensive experiments on MS-COCO and PASCAL VOC across two architectures and three scene-level attack scenarios demonstrate that DistScan substantially outperforms existing methods, improving average detection accuracy over the best-performing applicable baseline by 27.32 percentage points. 

---
# DA-WAM: Decision-Aligned Future Latents for Driving World Models 

**Authors**: Ruiguo Zhong, Benshan Ma, Xiaolong Chen, Lang Zhang, Mingyue Feng, Yaonong Wang, Pei Liu, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.19085)  

**Abstract**: Anticipating how scenes evolve under ego actions is fundamental to safe autonomous driving, yet the full potential of world models for decision-making remains unrealized. The critical challenge lies in ensuring that future modeling is not merely predictive, but decision-informative: the predicted future must directly shape which trajectory is selected. Existing approaches decouple future representation learning from planning optimization, or share predicted states across trajectory candidates, thereby diluting the action-specific consequences that ought to guide selection. To bridge this gap, we propose DA-WAM, a framework that unifies predictive representation learning, action-conditioned future modeling, and trajectory scoring under a single decision-making objective. DA-WAM maintains predictive supervision throughout planner optimization via an online encoder and a stable momentum target, allowing future representations to co-evolve with the driving task. An action-conditioned predictor generates a distinct future latent state per trajectory candidate, which is then evaluated by a future-latent-conditioned factorized scorer. For the expert-matched trajectory, the predicted future latent is supervised by the observed future representation, while safety-critical hard negatives provide additional supervision near planning boundaries. Extensive experiments on NAVSIM-v1 and NAVSIM-v2 demonstrate state-of-the-art performance, while ablations and diagnostic analyses validate the key components. 

---
# ReWEIGH the Evidence: Calibrating Token-Level Ordinal Visual Evidence to Mitigate Hallucinations in Large Vision-Language Models 

**Authors**: Jihae Jeong, Junha Choi, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.19075)  

**Abstract**: Large vision-language models (LVLMs) often hallucinate, generating content that the input image does not support. Preventing such content during decoding calls for a candidate-specific measure of how strongly the image supports the token under consideration. The model's visual-token states offer a natural source of this evidence because projecting each state through the output head reveals which vocabulary items that position favors. These position-wise readouts cannot be pooled directly because their probability magnitudes are not comparable across visual positions. Vocabulary ranks provide a scale-invariant basis for pooling, but tokens still differ systematically in their typical rank-based evidence. We propose ReWEIGH, a training-free decoding intervention that aggregates these ranks across visual positions and compares each candidate with a token-specific reference estimated from unlabeled images. At inference, ReWEIGH caches the image evidence during prefill and applies a bounded penalty only to candidates that fall below their reference. On four 7B backbones, ReWEIGH reduces hallucinated object mentions by up to 21.3% while largely preserving or improving descriptive and general performance. With evidence cached, the average added latency is 1.33% per token, and the reductions extend across six architecture families to 32B parameters. 

---
# GS-VLA: Plug-and-Play Viewpoint Canonicalization for Frozen VLA Policies via Gaussian Splatting 

**Authors**: Yechan Park, HyunJin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.19066)  

**Abstract**: This paper proposes a lightweight, plug-and-play framework that improves robustness to viewpoint shifts in Vision-Language-Action (VLA) policies without policy retraining. To our knowledge, this is the first approach to directly leverage 3D Gaussian-based novel-view synthesis for observation-space adaptation in VLA policies. Current VLA performance relies on the implicit assumption that training and deployment camera configurations are identical. Our experiments show that even a small displacement of the camera mount can reduce the success rate on the LIBERO benchmark from about 90% to about 10% in the worst case. Prior approaches, such as large-scale fine-tuning or generative data augmentation, are computationally expensive and risk catastrophic forgetting. To address this, viewpoint shifts are reformulated as a localized novel-view synthesis problem. Under a Locality assumption, that camera perturbations remain within a small bounded region relative to the workspace, viewpoint normalization reduces to a scene- and policy-independent disocclusion task. Our work implements this idea with a 4M-parameter 3D-Gaussian canonicalizer prepended to a frozen VLA policy. Without modifying policy weights, GS-VLA improves performance across three orthogonal axes: (1) Policy architectures, (2) Unseen task suites, and (3) Perturbation scales. These results show that a lightweight visual module can recover a large fraction of the performance lost under viewpoint shift, without policy retraining. 

---
# Bernstein-Vazirani Networks: Quantum Machine Learning by Interference 

**Authors**: Natacha Kuete Meli, Tolga Birdal, Prayag Tiwari, Vladislav Golyanik, Michael Moeller  

**Link**: [PDF](https://arxiv.org/pdf/2608.19043)  

**Abstract**: We introduce Bernstein-Vazirani Networks (BVNs), a non-variational quantum machine learning framework that leverages quantum interference for supervised learning, demonstrated on vision and representation learning tasks. In their standard form, BVNs follow the principle of quantum Fourier sampling: labelled data are placed in superposition and interfered in the Fourier basis to extract globally informative features. We then define generalised BVNs that enable interference in problem-adapted bases, yielding more expressive models under the same measurement budget as in the standard setting. BVNs achieve universal function approximation through (over)complete interference bases, while training of BVNs is gradient-free. Experiments on synthetic and real-world classification tasks, as well as implicit image representation, show strong generalisation capabilities and competitive performance with classical and quantum baselines. 

---
# Counterfactual Contrastive Analysis 

**Authors**: Yunlong He, Pietro Gori  

**Link**: [PDF](https://arxiv.org/pdf/2608.19032)  

**Abstract**: Visual Counterfactual Explanations (VCEs) aim to explain image classifiers by generating minimally edited and realistic versions of an input image that change the classifier's prediction. Existing VCE methods are inherently classifier-dependent and therefore susceptible to classifier biases and failure modes, such as sensitivity to shortcut features and calibration errors. In this paper, we propose a classifier-free approach for visual counterfactual generation based on Contrastive Analysis (CA). Given two datasets corresponding to different classes (e.g., healthy and patients), we disentangle the generative factors that are common across the two datasets from those that are salient to each dataset, and generate counterfactual images by swapping only the salient factors. By operating directly on data distributions rather than decision boundaries, our method provides model-agnostic VCEs that are less sensitive to classifier biases. Our approach leverages the high-quality synthesis and well-structured latent space of StyleGAN2. We use the feature space F, instead than the usual W-space, to improve detail preservation. Unlike conventional CA approaches, which typically assume salient factors in only one dataset, we introduce an adapted framework and loss functions for VCE that allow multiple salient factors in each dataset. We evaluate our method on three medical imaging datasets and demonstrate superior counterfactual generation quality compared to existing approaches. 

---
# One-Stage Object Detectors in Autonomous Driving 

**Authors**: Jonel Roman, Ryan Sirjue, Peter Nguyen, Daniel Krutky, Juan Jesus, Sudip Dhakal  

**Link**: [PDF](https://arxiv.org/pdf/2608.19014)  

**Abstract**: Autonomous vehicles depend on fast and reliable perception systems to detect surrounding vehicles, pedestrians, cyclists, traffic signs, and other road objects in real time. This paper presents a comprehensive survey and analysis of one-stage object detectors for autonomous driving rather than an implementation of a new detection system. The survey reviews the evolution of major one-stage detectors, including YOLOv1, SSD, RetinaNet, EfficientDet, anchor-free detectors such as FCOS and CenterNet, and recent real-time models such as YOLOv10. It compares these architectures through their design choices, feature-fusion strategies, loss functions, deployment trade-offs, and reported benchmark performance. The paper also summarizes commonly used autonomous-driving datasets, evaluation metrics, open challenges, and future research directions. Overall, this survey highlights how one-stage detectors balance speed, accuracy, efficiency, and robustness, while also emphasizing the remaining gap between benchmark results and dependable real-world autonomous-driving performance. 

---
# Harness Continual Learning: Continual Adaptation Beyond Model Parameters 

**Authors**: Borui Kang, Jinrui Gu, Junhan Lv, Wenbin Li, Lei Wang, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.19013)  

**Abstract**: Continual learning has largely been model-centric, treating model parameters as the state that changes with sequential experience. Modern agents can also adapt through a harness of prompts, memories, tools, skills, and routing rules. Because these contents jointly shape later execution, a harness update can disrupt previously reliable behavior even when the model is frozen. This raises a new question: how can an agent continually improve its state outside the model while retaining behavior acquired earlier? We formulate Harness Continual Learning (HCL), a new continual learning paradigm in which the harness evolves around a frozen foundation model, and define the resulting loss of earlier behavior as harness-level forgetting. We instantiate HCL with four execution-facing components: the Task Interface, Experience Memory, Capability Map, and Adaptive Router. We further introduce guarded harness evolution to separate update generation from state commitment. A Continual Optimizer proposes candidate harnesses from post-execution feedback, and a Continual Evaluator commits the resulting candidate harness only after checking current improvement, historical retention, and validity. Experiments on textual reasoning, multimodal perception, and open-world interaction demonstrate capability accumulation and failure recovery, with relative gains exceeding 10% over corresponding baselines in multiple settings. Component ablations assess the contribution of each harness component, while controlled retention sweeps reveal measurable harness-level forgetting and show that the stability--plasticity trade-off can be explicitly adjusted. 

---
# From Threat Intelligence to Detection: Knowledge-driven Enrichment and Template-based Rule Grounding for Automated Sigma Rule Generation 

**Authors**: Sepehr Ghaffarzadegan, Boubakr Nour, Makan Pourzandi, Mourad Debbabi, Chadi Assi  

**Link**: [PDF](https://arxiv.org/pdf/2608.19011)  

**Abstract**: Mechanisms for dynamically converting cyber threat intelligence (CTI) into actionable detection capabilities are necessary due to the rapid evolution of Advanced Persistent Threats (APTs). Sigma rules are an essential part of contemporary threat detection workflows because they offer a platform-independent framework for expressing detection logic that can be converted into particular queries across SIEM systems. Conventional techniques for manually crafting Sigma rules are prone to mistakes, and necessitate extensive knowledge, which restricts their scalability. Although there are open-source and industry-maintained Sigma rule repositories, they often fail to keep pace with emerging threats and require frequent customization to fit diverse operational environments. This emphasizes the necessity of dynamic rule generation that is adapted to evolving attack techniques as well as particular use cases. In this work, we design AUTOSIGMA, an automated solution for transforming unstructured CTI reports into relevant Sigma rules. Rather than relying solely on language models, AUTOSIGMA leverages a structured knowledge base to enrich partial inputs, matches the enriched content against a repository of existing Sigma rules, and then employs an LLM-as-a-Judge mechanism to iteratively validate the rules. By combining knowledge-driven enrichment, template-based rule grounding, and a multi-stage solution, AUTOSIGMA enables accurate, context-aware, and relevant rule generation. Evaluations across multiple real-world APT reports and multiple security blogs demonstrate that AUTOSIGMA outperforms alternative solutions and LLM models in rule validity, rule relevancy, MITRE ATT&CK technique coverage, and robustness to input quality.
AUTOSIGMA's Demo: this https URL 

---
# GrabVG: Graph-Attentive Binding for Visual Grounding in UAV Imagery 

**Authors**: Chaowei Wang, Yan Di, Jingjun Sun, Baozhe Liu, Jiaxu Tian, Yuheng Li, Guangqian Guo, Shan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.18996)  

**Abstract**: Visual grounding in Unmanned Aerial Vehicle (UAV) imagery aims to localize a target object in complex bird's-eye-view scenes according to a natural language description. However, the abundance of small, densely distributed, and visually similar objects creates high visual redundancy, while repetitive local configurations give rise to strong topological ambiguity. Existing approaches mainly focus on visual--language feature alignment or dense contextual interaction, yet they struggle to distinguish subtle inter-instance differences and effectively exploit spatial topological structures, leading to inaccurate grounding in highly crowded scenarios. To address these challenges, we propose $\textbf{GrabVG}$, a novel visual grounding framework inspired by human visual search. GrabVG explicitly decomposes grounding into two sequential stages: $\textit{preattentive hypothesis search}$ and $\textit{graph-attentive feature binding}$. Specifically, we first generate a compact set of reliable object hypotheses through distillation-guided proposal induction and text-aware hypothesis filtering, substantially reducing background distractions and semantic mismatches. These hypotheses are then organized into a sparse graph, where language-guided intra-instance visual cues and inter-instance topological relationships are jointly bound and propagated via graph attention, enabling efficient spatial reasoning and accurate target localization. Extensive experiments on AerialVG and AerialSense show that GrabVG achieves a favorable accuracy--speed trade-off, reaching 67.31$\%$ and 80.34$\%$ Acc@0.5 and outperforming the corresponding baselines by 10.55 and 8.76 percentage points, respectively. 

---
# DeepWeaver: Bridging the Evidence Synthesis Gap in Open-Ended Question Answering 

**Authors**: Xujia Wang, Yizhe Zhang, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.18988)  

**Abstract**: Retrieve-then-generate pipelines are commonly used to produce deep-research answers for open-ended questions, but retrieval alone is insufficient: LLMs must organize noisy and fragmented evidence into comprehensive, well-cited answers. We refer to this process as evidence synthesis. However, direct generation often underuses evidence, misaligns citations, and collapses diverse information into shallow summaries, exposing an evidence synthesis gap between retrieval and generation. Thus, we propose DeepWeaver, a novel framework that weaves noisy retrieved evidence into comprehensive answers by maintaining Thought Block Chains (TBCs), a structured representation that groups claims, salient information, keywords, and supporting evidence. DeepWeaver uses subordinate TBCs to inspect residual evidence, commit TBC revisions, and discover new claims before final generation. We evaluate DeepWeaver on open-ended QA over both knowledge bases and the web, and introduce LoQA, a high-density benchmark for evidence synthesis. Across multiple LLMs, DeepWeaver improves content sufficiency, citation grounding, and detail preservation on LoQA, while achieving deeper insights and higher citation quality on DeepResearch Bench. These results show that evidence weaving is an effective mechanism for bridging retrieval and generation in open-ended QA. Our code is available at this https URL. 

---
# rEDMRec: Distilling Large Language Model Reasoning into an Editable Experience Memory for Recommendation 

**Authors**: Minh Hoang Nguyen, Tung Le, Huy Tien Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2608.18952)  

**Abstract**: Large language models can improve recommendation quality by reasoning explicitly over user history and candidate items - for example, extracting a user's preferences or explaining why one item fits better than another - rather than mapping history directly to a ranked list. This reasoning, however, is expensive to repeat on every ranking request and, once produced, is typically consumed once and discarded, leaving it neither reusable across future requests nor easy to inspect or correct as user tastes drift. Our insight is that reasoning does not need to be regenerated at every call if it can instead be compressed once into a compact, structured memory that a lightweight model retrieves from. We propose rEDMRec, which distills a teacher LLM's reasoning into four typed, editable experience channels - long-term preference, short-term context, item-perception, and counterfactual hard-negative comparisons - maintained by an LLM memory controller that performs Add/Delete/Modify/Keep operations and refines entries via K-agent debate. A lightweight student LLM then ranks candidates purely by retrieving from this memory, without invoking the teacher again, decoupling online inference cost from reasoning depth. Across ML-1M, Amazon Beauty, and Steam and ten student backbones, rEDMRec improves HR@1 over zero-shot, few-shot, and RAG on every backbone, and over GraphRAG on most backbones, with Impv up to 13.3% vs. the second-best baseline on ML-1M. Channel ablations show that short-term context is the only channel that helps consistently across capacity tiers, whereas long-term, item-perception, and counterfactual contributions are capacity-dependent (and can reverse on the strongest students); debate-based memory optimization lowers bank duplication by 7.4 percentage points while raising downstream HR@1 by up to +0.029 over six optimization epochs. 

---
# AlphaClifford: Efficient Clifford Synthesis and Transpilation with Model-based RL 

**Authors**: Daniele Lizzio Bosco, Jacopo Cossio, Carla Piazza, Giuseppe Serra  

**Link**: [PDF](https://arxiv.org/pdf/2608.18946)  

**Abstract**: Clifford circuits play a foundational role in quantum computing, particularly due to their importance in quantum error correction and fault-tolerant logical synthesis. While these circuits can be efficiently simulated and represented as symplectic matrices, standard synthesis methods-such as the Aaronson-Gottesman algorithm-often yield sub-optimal circuits with excessively high gate counts. In this work, we introduce AlphaClifford, a model-based Reinforcement Learning framework powered by Monte Carlo Tree Search, designed to efficiently synthesize Clifford circuits from the fundamental gate set composed of H, S, and CNOT. By modeling the state space through the algebraic properties of the symplectic group, AlphaClifford effectively explores this combinatorial space to minimize overall circuit cost. For unconstrained Clifford optimization, our approach achieves a consistent reduction in both total and two-qubit (CNOT) gate counts compared to state-of-the-art synthesis heuristics, despite operating with a strictly less expressive gate set. Furthermore, we demonstrate the broad applicability of our framework on two additional tasks: hardware-constrained Clifford transpilation, where we outperform existing RL-based compilers, and as a post-synthesis optimization component within a full Clifford+T logical synthesis pipeline. Our results underscore that model-based RL is highly effective at addressing the combinatorial complexities of quantum compilation, offering a scalable pathway to mitigate hardware constraints in both near-term and future fault-tolerant quantum devices. 

---
# Training Chemical Plausibility-Aware Large Language Models for Single-Step Retrosynthesis 

**Authors**: Bogdan Zagribelnyy, Ivan Ilin, Nikita Bondarev, Maksim Kuznetsov, Mathieu Reymond, Vladimir Aladinskiy, Alex Aliper, Alex Zhavoronkov  

**Link**: [PDF](https://arxiv.org/pdf/2608.18940)  

**Abstract**: Single-step retrosynthesis is a central component of computer-aided synthesis planning, yet its intrinsically one-to-many nature is poorly captured by single-answer evaluation and benchmarking protocols. To address this, we introduce Top-K prompting as a robust training and inference paradigm to better capture diverse, plausible reaction predictions. We compile CREED-CCV-2+USPTO-XL, an ultra-large-scale dataset of ~45.6 million verified reactions to train the C3LM (Chemistry Constraint-Consistent Language Model). By integrating fine-tuning with ChemCensor-based and novelty-oriented rewards, our model achieves state-of-the-art performance on the OOD URSA-expert-2026 benchmark. Further analysis of reaction uniqueness shows that LLMs and conventional models explore complementary reaction spaces, motivating ensemble-based retrosynthesis systems. Overall, our results establish Top-K, plausibility-aware training as a practical new direction for robust future LLM-based synthesis planning. 

---
# MedUAG: Unified Understanding and Generation for Medical Multimodal Models 

**Authors**: Zijie Meng, Yuncheng Zhang, Hualiang Wang, Yitian Tang, Xiaotang Gai, Chen Shen, Songtao Jiang, Shaosheng Cao, Jian Wu, Xian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18937)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) are rapidly evolving into unified understanding and generation (UAG) frameworks. However, extending these unified paradigms to the medical domain is hindered by: the absence of comprehensive training and evaluation benchmarks, and the lack of broadly validated unified medical model. To address these gaps, we present a comprehensive foundation for medical UAG. First, we construct MedUAGCorpus, the largest unified medical understanding and generation dataset to date, comprising over 6 million instances across 14 imaging modalities. Second, we introduce MedUAGBench, a systematic benchmark that expands medical generation evaluation to 12 diverse tasks under standardized protocols. Finally, leveraging these resources, we develop MedUAG, an end-to-end trained unified medical model. Extensive experiments demonstrate that MedUAG achieves strong performance across a wide array of understanding and generation tasks, establishing a competitive baseline and paving the way for next-generation medical multimodal systems. 

---
# Graphical Design of Interpretable Architectures 

**Authors**: Pietro Barbiero  

**Link**: [PDF](https://arxiv.org/pdf/2608.18936)  

**Abstract**: Designing, implementing, and comparing interpretable architectures requires a formal language to represent them. The most common representations fall short in one of two ways. Symbolic equations give no global view of an architecture at a glance. Probabilistic graphical models and flowcharts do not describe actual tensor manipulations, thus hiding key insights and limiting reproducibility. To close this gap, we introduce a graphical notation for designing interpretable AI architectures, adapted from Penrose tensor notation. This graphical notation gives a global view of an architecture and maps one to one onto PyTorch einsum code. We first use this notation to describe architectures that are interpretable by construction, including concept bottlenecks, sparse probes, prototype networks, neural additive models, and mixtures of linear models. We then diagram the key architectural components of Steerling-8B, a frontier interpretable language model. The diagram yields global insights into the architecture (e.g., showing that Steerling is a residual model), a geometric interpretation of each individual operation, and a direct translation into 33 lines of PyTorch code. 

---
# SkillForge: Self-Distilling Agents for Project-Specific Issue Resolution 

**Authors**: Silin Chen, Han Li, Xiaodong Gu, Yuling Shi, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2608.18933)  

**Abstract**: Large language model (LLM) based agents have demonstrated remarkable proficiency in automated software issue resolution, yet they often struggle to resolve issues in a specific repository because they lack project-specific knowledge. Existing self-evolving approaches acquire such knowledge from repository history or online repair trajectories, but they either depend on available historical issue-resolution signals or incur substantial per-issue test-time exploration cost. In this paper, we propose SkillForge, a self-distillation framework that proactively acquires project-specific knowledge from the repository itself. Instead of waiting for real issues to expose project-specific knowledge gaps, SkillForge synthesizes project-specific issues by re-implementing test-covered core functionalities of the repository. By resolving these synthetic issues, SkillForge distills reusable project-specific knowledge into entity-grounded skills and associates them with relevant repository entities for future issue resolution. Extensive experiments using both open-source and closed-source models show that SkillForge consistently improves issue resolution performance over strong baselines. These results demonstrate that proactively acquiring project-specific knowledge before solving real issues substantially improves downstream software issue resolution. 

---
# Test-Time Scaling in the Wild: Why Exploitation, Not Exploration, Is the Bottleneck 

**Authors**: Davide Romano, Kanak Raj, Jerrod Parker, Daniele Giofrè  

**Link**: [PDF](https://arxiv.org/pdf/2608.18931)  

**Abstract**: Test-time scaling (TTS) improves language model outputs by spending additional inference compute - generating multiple candidates, searching over partial sequences, or iteratively refining drafts. These techniques yield large gains on mathematics and code, but have been developed and stress-tested almost exclusively on tasks where verification is straightforward. We conduct the first compute-normalised comparison of five TTS families across five open-ended generation benchmarks spanning medicine, law, finance, general chat, and creative writing - grounded in a unified framework that decomposes the effectiveness of each method's token budget into exploration and exploitation. The answer depends on which side of that decomposition you examine. Scaling exploration works: the best candidate in the pool improves steadily with compute across all settings. What breaks is exploitation - the step that converts a rich candidate pool into a final output. With state-of-the-art generators, reward models correlate at only $\rho_v \approx 0.12$ with true quality, rendering selection near-random regardless of budget. Tree search amplifies this failure through diversity collapse. Refinement helps on one of five benchmarks; its apparent gains elsewhere are confounded. Only synthesis across candidates (Fusion) consistently improves over single-sample baselines, yet still recovers only ~40% of available quality. The candidate pool is not the bottleneck - choosing from it is. 

---
# SMTrap: Cost-Effective DoS Attacks Against Large Reasoning Models via SMT Conflict Guidance 

**Authors**: Jian Yang, Zhenqi Feng, Zhaoyang Yu, Zhaoxin Fan, Kejian Wu, Xiaofeng Wang, Zheng Zhu, Jianjun Huang, Wei You, Bin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18921)  

**Abstract**: Existing LRM-DoS methods rely heavily on model feedback to synthesize attack queries, requiring either repeated queries to the target model or training a dedicated attack model. These expensive operations severely weaken attack leverage. In this paper, we propose \emph{search amplification}, a novel, model-feedback-free LRM-DoS paradigm. It employs the conflict count derived from an Satisfiability Modulo Theories (SMT) solver as a low-cost external signal to guide the synthesis of inference-heavy Constraint Satisfaction Problem (CSP) instances. Our key observation is that LRMs depend on trial-and-backtracking search when solving CSPs, where higher SMT conflict counts on a given CSP instance positively correlate with more extensive LRM backtracking search and substantially longer output trajectories. Building on this finding, we propose \textsc{SMTrap}, a lightweight, CPU-only framework. Guided by SMT conflict counts, \textsc{SMTrap} generates inference-heavy CSP queries without model queries, attack-model training, or GPU computation. Evaluations across seven frontier models demonstrate the state-of-the-art LRM-DoS capability of \textsc{SMTrap}, producing DoS effects multiple times stronger than existing baselines. To mitigate the threat of \textsc{SMTrap}, we demonstrate a tool-based mitigation that significantly cuts token usage. 

---
# Learning-State-Aware Dynamic Generative Data Augmentation on Small-Scale Datasets 

**Authors**: Ting Xiang, Chenxi Deng, Jinhui Zhao, Bingting Jiang, Ke Zhang, Changjian Chen, Zhuo Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18907)  

**Abstract**: Small-scale image classification is often limited by the scarcity of training data. Generative data augmentation (GDA) based on pretrained generative models has emerged as an effective solution. However, existing methods rely on task-agnostic augmentation strategies that overlook downstream model needs. Although recent dynamic GDA methods incorporate model feedback to guide augmentation, they still struggle to reliably determine sample-specific augmentation strengths and adapt augmentation strategies to different image regions while balancing image diversity and class semantics.
To address these issues, we propose learning-state-aware dynamic generative data augmentation (LSADA). Specifically, LSADA constructs a learning state for each sample based on its current loss and loss-decrease rate, which is then mapped to a sample-specific augmentation strength. Furthermore, LSADA introduces a decoupled data augmentation and diffusion fusion strategy that applies strength-controlled transformations to class-relevant regions and generates diverse class-irrelevant regions, progressively fusing them to improve image diversity while preserving class semantics. Experiments on nine public datasets show that LSADA outperforms the existing SOTA dynamic GDA method by an average of 4.5% on six natural image datasets and 2.5% on three medical image datasets. 

---
# MLREF: Efficient Module Reuse for Reward Design in Reinforcement Learning via Large Language Models 

**Authors**: Chenglin Liu, Xun Wang, Ruishuo Chen, Zhuoran Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18827)  

**Abstract**: Reward function design remains a bottleneck in reinforcement learning. While large language models (LLMs) have enabled automated reward generation, existing methods generate and revise reward functions as monolithic programs, making it difficult to reliably preserve and reuse effective components discovered in earlier iterations, leading to unstable performance across iterations. To address this, we propose Module Level Reward Evolution Framework (MLREF). At the core of MLREF is a module pool, a persistent repository of reusable reward components. MLREF treats the module pool as the primary optimization object: the pool evolves across iterations by accumulating successful modules, refining underperforming ones, and reusing proven components; while reward functions are constructed as linear combinations of modules drawn from this pool. To drive this evolution, MLREF integrates three mechanisms: reflection-based refinement, hybrid credit assignment, and a merge strategy with rollback, which together improve the effectiveness and robustness of reward optimization. Experiments on 17 tasks show that MLREF outperforms strong baselines by 25.2% in locomotion and 6.6% in manipulation, with more stable optimization dynamics. 

---
# Understanding Multilingual Medical ASR Adaptation Through Layer-Wise Analysis 

**Authors**: Souranil Kahali, Rituparna Bose, Abner Hernandez, Tomas Arias-Vergara, Andreas Maier, Ning Ma, Paula Andrea Perez-Toro  

**Link**: [PDF](https://arxiv.org/pdf/2608.18825)  

**Abstract**: Medical automatic speech recognition (MedASR) requires adaptation to specialised terminology, limited annotated clinical data, and multilingual use cases. Although large-scale pretrained ASR models such as Whisper achieve strong generalisation, their behaviour after medical and multilingual adaptation remains insufficiently understood beyond word error rate (WER). This paper investigates how multilingual medical adaptation reshapes the internal representations of Whisper models through layer-wise encoder analysis. We compare zero-shot decoding, English-only fine-tuning, German-only diagnostic fine-tuning, two-stage EN->EN+DE continuation, and direct EN+DE fine-tuning across Whisper model sizes. Fine-tuning substantially improves MedASR performance, but the best model depends on the adaptation setting: Whisper-Medium gives the lowest English WER (7.72%) and the lowest combined EN+DE WER under direct EN+DE training (26.30%); German-only Whisper-Large-v3 gives the lowest German WER (44.96%), but as a within-corpus diagnostic on 86 single-speaker training utterances rather than robust generalisation. Layer-wise analysis of the two-stage Whisper-Small trajectory shows that English medical fine-tuning produces the dominant encoder shift, whereas multilingual continuation largely preserves the adapted representation space. Domain and language information remain highly recoverable across layers, while linearly recoverable error-predictive cues weaken as WER improves. 

---
# Identifying Implicit Premises for Logical Reconstruction of Argument Graphs 

**Authors**: Xuyao Feng, Anthony Hunter  

**Link**: [PDF](https://arxiv.org/pdf/2608.18821)  

**Abstract**: The logical reconstruction of argument graphs from natural language text is challenging because of the prevalence of enthymemes (i.e., arguments with implicit premises). There are natural language processing methods for identifying enthymemes in text, and there are symbolic methods based on abduction for identifying missing premises in a logical representation of enthymemes. However, there is a need for methods to generate implicit premises to logically show a known entailment or contradiction relationship between a pair of statements. To address this, we propose a neuro-symbolic pipeline that uses large language models (LLMs) to generate intermediate implicit premises that are translated into logical formulae and used with logical formulae representing explicit premises and explicit claims to show the logical relationships between them (entailment, contradiction, or neutrality). Our approach is evaluated on the Microtext Argumentative Corpus. 

---
# Do Large Language Models Hallucinate Electric Fata Morganas? 

**Authors**: Kristina Šekrst  

**Link**: [PDF](https://arxiv.org/pdf/2608.18816)  

**Abstract**: AI hallucinations - that is, outputs which are made up, cannot be verified, or contradict the source material - are generally regarded as an engineering flaw to be dealt with. This paper contends that they also have philosophical significance when it comes to the question of machine consciousness. We examine the known causes of hallucinations in large language models - such as source-target divergence, discrepancies between training and inference, and overfitting - and we present two empirical investigations. In the first, we apply successive generations of the GPT model to ambiguous factual questions under different temperature settings, finding that higher temperatures result in plausible but incorrect answers while lower temperatures lead to factually accurate ones. The sampling parameters that cause a model to seem creative or spontaneous and thus more likely to pass behavioral tests of intelligence are the same ones that increase its hallucination rate. In the second, we look at an encoder-only model that has been trained on encyclopedic data and which answers questions of the same type factually and without embellishment, indicating that hallucinations are due to exposure to subjective and socially diverse training data rather than to the development of any cognitive ability. Using references to Turing, Searle's Chinese Room, the frame problem, and the cybernetic tradition of Wiener and Ashby, we claim that a model's self-reports of emotion or sentience come within the definition of hallucination, and that any future occurrence of machine consciousness might remain epistemically inaccessible since it would be indistinguishable from a sufficiently advanced hallucination. 

---
# A strengthening of the MCFL-ness of $O_2$ 

**Authors**: Marco B. Caminati  

**Link**: [PDF](https://arxiv.org/pdf/2608.18813)  

**Abstract**: In the last years, a number of proofs of the fact that $O_2$ is a multiple context-free grammar (MCFG) were given. Such results can be exploited in the fields of both computational linguistics and of computational algebra. Here, we focus on a recent such proof spelled in terms of factorizations of string tuples, and give a new result with a stronger characterization of such factorizations than in existing theorems. 

---
# Forgetting, plasticity, and co-observation: a third facet of continual learning 

**Authors**: Timm Hess, Abhishek Jha, Gido M. van de Ven, Tinne Tuytelaars  

**Link**: [PDF](https://arxiv.org/pdf/2608.18803)  

**Abstract**: Efficient continual learning remains a fundamental challenge for deep neural networks. While catastrophic forgetting and loss of plasticity are widely considered the primary obstacles to overcome, we show that these two issues cannot fully explain the performance gap between naive sequential training and offline joint training. In this paper, we highlight data co-observation as a distinct factor influencing continual learning performance. By decoupling the constraints of separate data access from stability and plasticity, we systematically investigate the representational benefits gained by observing training data together. Empirically, we demonstrate a consistent performance difference between joint and separate training across both supervised and self-supervised paradigms in generic data-incremental "chunking" scenarios, whilst mitigating forgetting and controlling for plasticity. Our findings indicate that simultaneous observation of training data (co-observation) yields benefits to the learner's generalization that extend well beyond mere knowledge retention, and that this effect does not require a specific continual distribution shift. Furthermore, we contextualize prominent continual learning mechanisms through this lens: while distillation-based approaches act only as effective knowledge retention mechanisms, our results suggest that the empirical success of memory replay goes beyond the mitigation of forgetting, actively reintroducing the benefits of data co-observation into the learning process. 

---
# Decomposing Wrong-Consensus Agreement in LLM Self-Consistency: A GPT-4.1 Case Study 

**Authors**: Lizhuo Zhang, Mengmeng Tang, Chenfeng Long, Xiaoyong Tang, Xiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.18795)  

**Abstract**: Majority voting over multiple LLM samples is widely used to raise answer accuracy, yet its gain varies erratically: on hard questions it can even backfire. This paper gives a quantitative account of this failure. A pluralistic agreement index Gamma is defined as the expected fraction of the samples of a wrong run that agree with the consensus, normalized by a reference scale d=(1-p)/(C-1), and is decomposed into a mechanical component (what a vote delivers given only a per-case answer preference) and a preference-unexplained residual. The mechanical null is difficulty-matched and leak-free: each case is resimulated at its own accuracy and option preference, estimated from the case's other runs, so no run predicts its own agreement. On GPT-4.1 the decomposition shows benchmark-associated direction (an observational ordering over n=4 cells per benchmark, not a significance claim). On multiple-choice GPQA-Diamond, the per-case answer preference explains 81-93% of the held-out test-run agreement index: the shared-bias-dominates account over-claims here, because a wrong but attractive option the whole cohort latches onto is captured by the per-case preference channel (whether that preference is induced by shared training bias is not identified). On open-domain AIME, the mechanical preference explains only 59-78% (21-29% if shrunk to pure noise), and a preference-unexplained residual of 1.56-2.80 Gamma units survives, which a run-level preference-heterogeneity reference more than absorbs (1.4-2.1). A self-consistency backfire on hard questions is reproduced (binned voting gap down to -0.09, coupled CI [-0.12,-0.07]), and the highest-agreement bin reaches an accuracy of only 0.42-0.83, a 1.2-3.6x lift over base rate: agreement is graded evidence, not certification. No new voting method is proposed; code and evidence are committed and reproducible. 

---
# SIDScope: A Diagnostic Resource for Semantic-ID Interfaces in Generative Recommendation 

**Authors**: Jiandong Ding, Huijie Qin, Tiandeng Wu, Yi Cao  

**Link**: [PDF](https://arxiv.org/pdf/2608.18779)  

**Abstract**: Semantic-ID mappings are reusable interfaces between item tokenizers and generative recommenders, yet released mappings rarely state whether they are coherent, what structure they expose, how generated paths resolve, or what must be revalidated after a refresh. SIDScope is a source-traced diagnostic resource for these decisions. It normalizes item-to-code artifacts, verifies provenance and joins, profiles mapping structure, compares paired revisions, and accounts for path-to-item outcomes in generated traces. Across nine source-traced tokenizer exports from seven families on Amazon and Yelp data - eight executable routes plus one auditable snapshot - SIDScope reveals that interface health is multi-signal rather than scalar. Its central finding is mechanism-conditional: prefix alignment strongly tracks held-out candidate exposure when retrieval consumes SID prefixes, then weakens as scoring becomes prefix-independent. Trained trace accounting exposes a second hidden gap: a valid target path can survive without uniquely retrieving the target item by 1.2-3.0 percentage points. A refresh case establishes a third: repairing the mapping does not by itself restore an inherited generator; model reuse requires a separate handoff check. The package provides frozen evidence summaries, conformance reports, trace labels, table builders, and CPU-only verifiers. It supports decisions about artifact readiness, interface risks, and revalidation before model reuse. 

---
# Beyond Predictive Fairness: Quantifying Attribution Consistency Across Demographic Groups in Diabetic Retinopathy Screening 

**Authors**: Kerol Djoumessi, Philipp Berens  

**Link**: [PDF](https://arxiv.org/pdf/2608.18759)  

**Abstract**: Fairness in medical imaging is commonly evaluated through subgroup performance metrics, yet it remains unclear whether models rely on consistent visual evidence across demographic groups. This work introduces the Explanation Consistency Score (ECS), a fairness-aware metric based on Jensen-Shannon divergence that quantifies the similarity of attribution maps across subgroups. Using diabetic retinopathy screening as a case study, ECS is evaluated globally and within disease severity. Experiments reveal that while predictive performance differs across ethnic groups, explanation consistency remains relatively high and shows no significant association with performance disparities. These findings suggest that predictive fairness and explanation consistency capture complementary dimensions of model behavior, motivating fairness evaluations that extend beyond predictive performance. 

---
# Epistemic Subordination: Generative AI and the Infrastructure of Knowledge 

**Authors**: Gilad Abiri, Emanuel V. Towfigh  

**Link**: [PDF](https://arxiv.org/pdf/2608.18758)  

**Abstract**: Generative AI does not merely produce biased outputs. It encodes the majority's way of knowing as the default infrastructure of knowledge itself. We call this epistemic subordination. The training process compresses the full breadth of human expression into a single probabilistic model whose statistical baseline reflects the languages, assumptions, and cultural frameworks of the dominant culture. Minority epistemologies are not excluded but absorbed: present in the training data, yet structurally subordinated in the output. The result is not a collection of discrete biases that can be audited and corrected. It is an epistemic condition embedded in the architecture from which all outputs emerge. This unified harm cuts across three legal domains -- anti-discrimination law, cultural and linguistic rights, and democratic viewpoint pluralism -- and each fails to address it for the same structural reason: existing law regulates downstream, at the level of decisions and applications. The remedy must match the site of harm. If epistemic subordination is produced at the level of model training, then law must learn to govern at that level. 

---
# Flama: a Python framework for development and deployment of production-ready APIs, machine learning, and LLM services 

**Authors**: José A. Perdiguero López, Miguel A. Durán-Olivencia  

**Link**: [PDF](https://arxiv.org/pdf/2608.18733)  

**Abstract**: We present Flama, an open-source Python framework for developing and deploying production-ready web APIs, machine learning services, and large-language-model (LLM) applications. Built on the Asynchronous Server Gateway Interface (ASGI), Flama offers a type-driven, async-first programming model that unifies REST API development, predictive model serving, and generative AI inference in one architecture.
It is organised around seven subsystems: a component-based dependency injection system resolving handler parameters from type annotations at startup; a pluggable schema layer supporting Pydantic, Marshmallow and Typesystem behind a single adapter; an automatic CRUD generator turning a SQLAlchemy table and a schema class into REST endpoints backed by the Repository and Unit of Work patterns; a portable binary format (.flm) packaging models from scikit-learn, TensorFlow, PyTorch and Hugging Face Transformers with their metadata for zero-code deployment; a multi-backend LLM server running vLLM (Linux/CUDA) or MLX (Apple Silicon) and exposing four wire protocols (OpenAI, Anthropic, Ollama, and a native streaming dialect) through a shared codec; a Rust-accelerated core compiled via Maturin for routing, JSON encoding, compression and parsing; and a Model Context Protocol module turning any application into an MCP server over JSON-RPC 2.0.
Built-in capabilities include JWT authentication, two pagination strategies, background tasks in threads or processes, WebSocket endpoints, Server-Sent Event and NDJSON streaming, OpenAPI 3.2.0 generation from handler signatures, and a command-line interface for running applications and for serving, packaging and inspecting models.
We describe the architecture, present the programming model through worked examples, and compare Flama with existing frameworks, model serving platforms and LLM inference engines. 

---
# A Few Cases Are All You Need: An Empirical Study of Annotation-Efficient LoRA Fine-Tuning of MedSAM3 

**Authors**: Sachin Dudda Nagaraju, Bendik Skarre Abrahamsen, Ashkan Moradi, Mattijs Elschot  

**Link**: [PDF](https://arxiv.org/pdf/2608.18731)  

**Abstract**: Medical image segmentation is essential for clinical workflows such as treatment planning and disease assessment. While specialist tools like TotalSegmentator and MRSegmentator achieve strong performance, they require large annotated datasets for training. Medical foundation models offer a promising alternative through large-scale pretraining that reduces the annotation burden for new tasks, but zero-shot performance remains limited. Parameter-efficient adaptation via Low-Rank Adaptation (LoRA) enables efficient specialization with few trainable parameters, but a key question remains: how many expert-annotated cases are needed to achieve clinically useful segmentation performance? We address this by adapting MedSAM3 with LoRA for five abdominal organs (liver, kidneys, spleen, gallbladder, and pancreas) in CT and MRI using only 1, 2, 5, and 10 annotated cases, evaluating on AMOS22 dataset. With just 10 cases, models achieve performance competitive with specialist systems trained on orders of magnitude more data. Notably, this includes reliable gallbladder segmentation (Dice 0.68 CT, 0.59 MRI) where existing tools fail almost completely (Dice 0.0004), while remaining within 5--10% of MRSegmentator for liver, kidneys, and spleen using over 100 times fewer annotations. Furthermore, external validation on the Whole Heart Segmentation dataset shows that the approach extends to cardiac segmentation, a use case beyond the scope of TotalSegmentator (MRI) and MRSegmentator, achieving competitive left ventricle (LV) performance with only 10 annotated cases. Training requires only3--5,hours per organ on a single GPU, approximately 2--3 times faster than nnU-Net. These findings suggest that ten annotated cases are sufficient for clinically useful segmentation, effectively reducing bottlenecks for both image annotation and training time. 

---
# Budget-First Tariff Recommendation (BFTR): A Complete Algorithmic Framework for Telecom Plan Recommendation without Overcharging 

**Authors**: Ghislain Dorian Tchuente Mondjo  

**Link**: [PDF](https://arxiv.org/pdf/2608.18723)  

**Abstract**: Telecom operators traditionally offer predefined tariff grids, forcing users to choose from a limited set of plans. This paper proposes BFTR (Budget-First Tariff Recommendation), a complete algorithmic framework integrating eight Budget-First strategies, including two original hybrid approaches: Recursive Hybrid (conditional interpolation) and Knapsack-First Hybrid (priority knapsack). Unlike existing approaches that adjust prices upward to guarantee a minimum margin, BFTR guarantees the absence of overcharging by systematically aligning the final price with the catalog reference price. We mathematically formalize each strategy, prove the existence of an offer for any positive budget, and prove that the price deviation (surcharge) is zero for all strategies that do not use interpolation with correction. A detailed comparative analysis confronts BFTR to ten main existing tariff models on ten dimensions. Experiments on a dataset of 974 customers inspired by the Nigerian MTN market show that: (i) Recursive Hybrid is optimal for the customer (100% budget used, 29.9 GB volume, utility 0.946, 0% overcharging), (ii) Piecewise offers the highest volume (39.7 GB) with 0% overcharging, (iii) Power Law provides an excellent compromise (99.9% budget, 38.1 GB, 0% overcharging). All strategies achieve a zero surcharge, confirming the theoretical guarantees. A sensitivity analysis on the weighting parameter alpha (0.2 - volume priority, 0.5 - balance, 0.8 - budget priority) shows that utility rankings evolve logically. Execution times (< 10 ms) and very low failure rates (0% for robust strategies) confirm the operational viability of the system. The formal proof of the absence of overcharging constitutes a major theoretical contribution. 

---
# The Impact of CutMix on Reliability and Robustness in Semantic Segmentation 

**Authors**: Steven Landgraf, Markus Ulrich  

**Link**: [PDF](https://arxiv.org/pdf/2608.18715)  

**Abstract**: Ensuring not only high accuracy but also reliable and robust predictions is critical for the deployment of semantic segmentation models in safety-critical applications such as autonomous driving. Despite the widespread use of CutMix - a simple yet powerful data augmentation strategy - its effect on the reliability and robustness in dense predictions tasks remains unexplored. Motivated by recent findings that semi-supervised segmentation methods, where CutMix is a core component, can severely degrade reliability, this study isolates and systematically analyzes the influence of CutMix on segmentation accuracy, calibration, and uncertainty quality. We evaluate two representative architectures, the CNN-based DeepLabV3+ and the transformer-based SegFormer, across both in-domain and out-of-domain scenarios. Our results show that CutMix has only a minor impact on segmentation accuracy but consistently improves the reliability, particularly under distribution shifts. These improvements indicate that CutMix primarily enhances the trustworthiness of the model's calibration and uncertainty rather than the raw segmentation prediction itself. This distinction is crucial for safety-critical deployment, where reliable confidence estimates are as important as raw performance. 

---
# A Critical Synthesis of Uncertainty Quantification and Foundation Models for Semantic Segmentation 

**Authors**: Steven Landgraf, Joceline Hinz, Markus Ulrich  

**Link**: [PDF](https://arxiv.org/pdf/2608.18709)  

**Abstract**: Foundation models are increasingly breaking what seemed to be impossible not long ago by enabling unprecedented accuracy and cross-domain generalization. Yet their lack of interpretability, tendency to be overconfident, and sensitivity to real-world domain shifts pose critical challenges for safety- and mission-critical applications. Uncertainty quantification (UQ) offers a principled way to address these issues, but its integration into segmentation foundation models has yet to be explored. In this paper we present the first systematic evaluation of UQ methods applied to a foundation model for semantic segmentation. We fine-tune a lightweight DPT decoder on top of the pretrained SAM2 encoder to establish a simple yet competitive baseline and benchmark four representative UQ approaches - Monte Carlo Dropout, Deep Sub-Ensemble, Test-Time Augmentation, and Evidential Deep Learning - across Cityscapes, NYUv2, and two challenging out-of-domain settings. Our analysis compares segmentation accuracy, calibration, uncertainty quality, and inference time, revealing clear trade-offs between predictive performance, reliability, and computational cost. These results highlight both the promise and the current limitations of uncertainty-aware foundation models, pointing to the need for future work that jointly optimizes accuracy, robustness, and efficiency for real-world deployment. 

---
# MemFuse: Multi-Source Memory Fusion from Fragmented Observations 

**Authors**: Chao Li, Yuanfa Li, Wenhao Wu, Xule Liu, Zhi Wang, Kun Shao  

**Link**: [PDF](https://arxiv.org/pdf/2608.18704)  

**Abstract**: Long-term memory is essential for agents that operate across extended interactions, yet existing memory systems and benchmarks predominantly focus on single-source textual histories. In realistic settings, however, relevant information is often fragmented across applications and devices, as well as across users and time, requiring agents to integrate dispersed observations into coherent episodic memories while preserving their source provenance. To address these gaps, we introduce **MemFuseBench**, a benchmark for *multi-source memory fusion*. MemFuseBench is built with a Scene-to-Sensor pipeline that synthesizes controllable scenarios into source-tagged observations, evidence-grounded questions, and adversarial distractors. It enables systematic evaluation of temporal reasoning, cross-source evidence fusion, and robustness to noise. We further propose **MemFuse**, a structured memory system that preserves source-level evidence in event-layer atomic memory and organizes related atomic events into cluster-layer fused memory within a causal fusion graph. During retrieval, MemFuse retrieves and organizes related evidence fragments while maintaining traceability to original source events. Experiments on MemFuseBench show that MemFuse achieves the best overall performance among the evaluated memory systems under all three LLM settings and consistently improves performance on questions requiring cross-source evidence fusion. 

---
# Impact of Iterative Fine-Tuning on Transcription Accuracy in Complex Historical Sanskrit Manuscripts 

**Authors**: Kartik Chincholikar, Kaushik Gopalan, Mihir Hasabnis  

**Link**: [PDF](https://arxiv.org/pdf/2608.18696)  

**Abstract**: Digitizing the text from handwritten historical manuscripts is required to make them easily accessible, preservable, and to enable historical scholars to study them in new ways. Historical manuscripts, however, often exhibit complex heterogeneous layouts and non-standard appearance due to period-specific writing styles, page textures, camera noise, and other nuisance factors, making them difficult to perform OCR on. To tackle this challenge, we introduce a local traditional OCR pipeline, which can be iteratively fine-tuned on the target manuscript at the layout-level and the appearance-level. By adapting to the target manuscript distribution, the proposed Traditional OCR pipeline makes better predictions on subsequent pages, causing iterative reduction in human annotation effort, which is expensive and time-consuming as it requires historical domain expertise. Using this pipeline, we digitize text from three complex historical Sanskrit manuscripts and introduce a dataset with granular layout-level annotations, along with Unicode annotations in the standard PAGE-XML format. We demonstrate quantitative gains due to iterative fine-tuning of the proposed traditional OCR pipeline, and also benchmark the performance of leading Multi-Modal Large Language Models on the introduced Dataset. Code and dataset are available at: this https URL. 

---
# Composed Historical Image Retrieval by Modeling Temporal Representations 

**Authors**: Adrià Molina Rodríguez, Oriol Ramos Terrades, Josep Lladós Canet  

**Link**: [PDF](https://arxiv.org/pdf/2608.18694)  

**Abstract**: While time evolves linearly, the geometry of neural embedding spaces is inherently multi-dimensional, often chaotic, and difficult to interpret. In principle, one could constrain an embedding space to a single temporal dimension; however, such a reduction would sacrifice performance on downstream tasks, as one-dimensional embeddings cannot retain sufficient expressive capacity. This paper asks whether it is possible to learn representations that preserve temporal structure while remaining effective for image and object retrieval, and answers this question by building the mathematical foundations of such a system. We propose Temporally Decomposable Image Representations (TDIR), a representation learning algorithm that decomposes historical photographs into separate date and content components through orthogonal subspaces. We define and prove the conditions under which such a decomposition is achievable, characterize the error incurred when those conditions are only partially met, and show that orthogonality between temporal and categorical subspaces emerges naturally from the joint optimization, without requiring it to be imposed explicitly. Beyond its geometric properties, TDIR enables a class of transitive operations on embedding spaces: the temporal information of one image can be extracted and injected into the representation of another, with no label supervision required. All theoretical properties are grounded and validated in the real-world problem of Composed Image Retrieval on historical photographs, where a query simultaneously specifies object content and a target time period, either through labels or through example images. This in-the-wild setting serves as a concrete backing for the propositions we derive, offering an intuitive and interpretable way to navigate photographic archives while maintaining competitive performance in both date estimation and object retrieval. 

---
# Europe's Climate Ambition Under Scrutiny: Evidence from Deep Learning Emission Projections 

**Authors**: Jacopo Ghirri, Carlos Rodriguez-Pardo, Lara Aleluia Reis, Massimo Tavoni  

**Link**: [PDF](https://arxiv.org/pdf/2608.18690)  

**Abstract**: The European Union has committed to reducing greenhouse gas emissions 55% below 1990 levels by 2030, but whether current trends are compatible with this ambition remains uncertain. We apply deep learning to high-resolution socioeconomic and sectoral data across EU27 member states till 2023 to project sectoral CO$_2$ trajectories under current trends, extrapolating observed sectoral momentum without assuming changes in the pace or effectiveness of the policy environment beyond what is already reflected in historical data. We project that EU27 emissions will exceed the 2030 target by 35% (620 Mt CO$_2$ shortfall), with only a small minority of countries on trajectories consistent with the bloc's commitments. While the Power sector achieves target-consistent reductions driven by the renewable transition, Mobility shows minimal progress and accounts for over a third of total emissions by 2030, reflecting a structural inertia across member states rather than geographically concentrated lag. Our findings indicate that substantial additional intervention is required to close Europe's ambition-implementation gap, and call for establishing up-to-date energy information in Europe. 

---
# Aslema at NADI 2026: Augmentation through Fewshot for SLU 

**Authors**: Tajwaar Shafiq, Hunzalah Hassan Bhatti, Shammur Absar Chowdhury, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2608.18689)  

**Abstract**: We present Aslema, our system for NADI 2026 Shared Task 5, which consists of two subtasks: intent recognition and slot filling. We evaluate four omni LLMs in a zero-shot setting and compare them with fine-tuned models. Our results show that fine-tuning consistently outperforms zero-shot inference. We further explore synthetic data augmentation by using an LLM to generate culturally grounded Tunisian Derja utterances, followed by voice cloning to generate synthetic speech. Incorporating this synthetic data improves performance on both tasks. Our final submitted system, based on Qwen3-Omni-30B and trained with a mixture of original and synthetic data, achieves 86.8% intent accuracy and 34.7 WER on the devtest split. On the official test set it ranks 1st in slot filling (59.5 CoER) and 4th among 8 teams in intent recognition (66.1% accuracy). We release our experimental scripts and will soon share the synthetic dataset to support further research in this area. 

---
# Orienteering Problem with Uncertain Time-Varying Rewards: Framework and Benchmark for Everyday Service Robotics 

**Authors**: Masafumi Endo, Kohei Honda, Yuu Jinnai, Ryo Yonetani  

**Link**: [PDF](https://arxiv.org/pdf/2608.18672)  

**Abstract**: We present the orienteering problem with uncertain time-varying rewards (OP-UTVR), a novel variant of the orienteering problem (OP). While most existing OP formulations assume rewards to be known in advance, practical applications involve uncertain and time-varying rewards, as with shifting customer demand for delivery agents. OP-UTVR relaxes this assumption by allowing agents to estimate reward dynamics from observations and forecast future rewards. This enables informed routing decisions despite stochastic reward changes and inevitable prediction errors. We address this problem using three planners that differ in planning horizon and online adaptivity, and derive theoretical bounds on their performance under reward stochasticity. We further introduce a mobile service robot benchmark for OP-UTVR, where a robot navigates among pedestrians in indoor environments. Experiments reveal trade-offs between planning horizon and adaptivity, and demonstrate the effectiveness of long-horizon planning with online adaptation. 

---
# Change Point--Aware Evaluation and Re-Calibration of PPG-Based Blood Pressure Estimation 

**Authors**: Yunwon Tae, Minje Park, Gyunho Rho, Dongjoon Yoo, Sunghoon Joo  

**Link**: [PDF](https://arxiv.org/pdf/2608.18639)  

**Abstract**: Non-invasive continuous blood pressure (BP) monitoring using photoplethysmography (PPG) is a promising alternative to cuff-based measurements. However, existing PPG-based BP estimation studies predominantly rely on aggregated performance metrics (e.g., mean absolute error) computed over entire evaluation intervals, which can obscure model failures during rapid BP fluctuations and limit clinical relevance. In this work, we propose a fluctuation-aware evaluation framework for PPG-based BP estimation based on time-series change point detection. Instead of heuristic BP thresholding (e.g., $\Delta\mathrm{BP} > 10\mathrm{mmHg}$), we identify BP change points by capturing abrupt distributional shifts in BP trajectories and evaluate estimation performance specifically during these fluctuation periods. Our analysis shows that several state-of-the-art models exhibit substantial performance degradation around BP change points, and that periodic test-time calibration is insufficient to handle such dynamic BP variations. To address this limitation, we introduce a targeted re-calibration framework triggered by detected BP change points, improving robustness without modifying model architectures. To the best of our knowledge, this is the first systematic evaluation of PPG-based BP estimation from a BP change point perspective, highlighting the importance of fluctuation-aware evaluation and calibration for real-world continuous BP monitoring. 

---
# Denoising-Aware Inversion: Revealing Privacy Risks in Noise-Protected Text Embeddings 

**Authors**: Yubo Wang, Shujie Cui, James Bailey, Hongzhi Yin, Wenyu Liang, Min Tang, Shiyue Qin, Weiqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18610)  

**Abstract**: Dense text embeddings are widely used in data mining, retrieval, and downstream machine learning systems due to their compact and semantically rich representations, but recent embedding inversion attacks have shown that they can expose substantial information about the original text, leading to serious privacy leakage risks. A common defense is to release perturbed embeddings by adding Gaussian noise, which is simple yet effective against standard inversion attacks and does not significantly degrade embedding utility for downstream tasks. However, it remains unclear whether such noise-protected embeddings are sufficiently safe against adaptive attackers that explicitly account for the perturbation process. In this paper, we study text embedding inversion in a noise-protected setting, where the attacker can observe only noisy embeddings and has no access to clean embedding targets. We first analyze why existing generative inversion methods fail under this setting and identify a "Double Noise Trap", which fundamentally prevents standard generative inversion models from achieving high-quality reconstruction. To address this challenge, we propose DAEI, a denoising-aware embedding inversion pipeline that combines a residual denoising autoencoder with generative text inversion where the denoiser is trained in an unsupervised manner using Stein's unbiased risk estimate to enable denoising from noisy observations alone. Extensive experiments show that DAEI achieves approximately 154\% relative improvement in BLEU over the existing generative inversion baseline, while also improving token-level F1 and ROUGE-L by 32--60\%. The promising inversion performance of DAEI challenges the prevailing assumption that simple Gaussian perturbation is sufficient to prevent sensitive information leakage from embedding representations. 

---
# OmniHandwritingOCR: A Diagnostic Benchmark for Evaluating Multimodal LLMs in Handwritten OCR Scenarios 

**Authors**: Zinuo Guo, Min Zhang, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18586)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly used as OCR systems in document and knowledge-processing pipelines, but their ability to faithfully read real handwriting remains underexplored. Existing OCR benchmarks focus largely on printed text or clean single-line inputs, leaving limited coverage of realistic handwritten OCR scenarios such as multilingual handwriting, writer errors, and structurally complex mathematical expressions. We introduce OmniHandwritingOCR, a diagnostic benchmark for evaluating MLLMs and OCR systems on handwritten OCR. It covers handwritten text recognition and handwritten mathematical expression recognition across six subtasks and twelve subsets, totaling 77.57K labeled images from public datasets and newly collected student writings. A key component is a difficulty-stratified multi-line formula corpus designed to test robustness under increasing structural complexity. We evaluate thirteen open- and closed-source systems with five complementary metrics under a unified protocol. Results show that current systems remain far from faithful transcription: performance drops sharply on complex multi-line formulas, model rankings vary across language and formula settings, and several generative models hallucinate plausible but visually unsupported corrections. OmniHandwritingOCR provides a challenging testbed for diagnosing language, content, structural, and visual-grounding failure modes of multimodal models in handwritten OCR scenarios. 

---
# From Storage to Access: Verifiable Activation of Parametric Knowledge in LLMs via Explicit Priming and Implicit Reasoning 

**Authors**: Zuocheng Ying, Yang Yang, Yumou Wu, Chuanbo Zhu, Jiarui Wang, Ziqi Wu, Jingming Cai, Junqing Yu, Zikai Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.18581)  

**Abstract**: Although Large Language Models (LLMs) encode rich factual knowledge in their parameters, reliably recalling and verifying such knowledge remains a key bottleneck in factual question answering. Existing end-to-end methods entangle knowledge elicitation with reasoning, making it difficult to determine whether correct answers arise from parametric knowledge or the input context. To address this challenge, we propose VAKE (Verifiable Activation of Parametric KnowledgE), a two-stage reinforcement-learning framework that externalizes latent parametric knowledge through explicit Priming and transfers the acquired elicitation capability to implicit Reasoning. Given a query and an insufficient retrieved subgraph, the Priming policy explicitly inserts bridging triples as verifiable evidence, with supervision provided by rewards derived from answers generated by a separate frozen model over the augmented subgraph. Building on the policy learned during Priming, the Reasoning stage trains the model to answer from the original input, testing whether the capability acquired through explicit knowledge elicitation transfers to implicit reasoning. Experiments across seven benchmarks and models from 3B to 14B show that VAKE consistently outperforms standard baselines, including when transferring directly from HotpotQA to OOD datasets. LLM-based evaluation further shows that over 80% of the inserted triples provide factual bridging knowledge not derivable from the retrieved context, while more than half elicit knowledge inaccessible through direct prompting. These results suggest that VAKE activates latent parametric knowledge rather than copying the input context or memorizing dataset-specific associations. 

---
# MR-IQA-2: Faithful Image Quality Reflection via Fine-Grained Credit Assignment 

**Authors**: Yuan li, Youyuan Lin, Chenhui Chu, Shin'ya Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2608.18579)  

**Abstract**: Multimodal large language models (MLLMs) have shown strong potential for image quality assessment (IQA) by improving consistency between quality ratings and their underlying reasoning. However, most approaches supervise reasoning through human-provided ratings and rarely examine whether it faithfully reflects image quality. Rating accuracy alone does not ensure faithful reasoning; a shared reward also obscures supervision sources and may reinforce unfaithful reasoning when a correct rating occurs by chance. To improve the faithfulness and reliability of blind IQA, we aim to (1) decouple credit assignment for reasoning and rating and (2) provide verifiable supervision for faithful reasoning. We introduce MR-IQA-2, an actor-editor-judge framework that operationalizes reasoning-editing-reflection. The actor generates quality reasoning for an input image, and the editor revises the image according to the identified quality factors. A frozen judge compares the original and edited images and provides reflective supervision for the actor's reasoning. MR-IQA-2 further uses fine-grained credit assignment to decouple reasoning and rating supervision. Judge feedback supervises reasoning, whereas human ratings supervise the predicted rating. Masked token-specific updates distinguish these signals while preserving the causal relation from reasoning to rating. Across IQA benchmarks, MR-IQA-2 achieves competitive rating alignment with humans. Visual reflection also enables richer and more faithful visual understanding beyond rating, which may inform image-quality optimization and related downstream tasks. Code is available at this https URL. 

---
# The Role of Grid Cells in Reducing Spatial Aliasing in Hippocampal Place Representations 

**Authors**: Alexander Johnson, Obadah Ghizawi, Ali A. Minai  

**Link**: [PDF](https://arxiv.org/pdf/2608.18569)  

**Abstract**: Spatial aliasing occurs when two or more distinct locations produce highly similar place-cell representations, primarily due to environmental symmetry or repetitive structures. This issue is most pronounced when place representations are constructed solely from boundary vector cell (BVC) inputs, because symmetric or repetitive structures can yield indistinguishable sensory patterns across multiple locations in an environment. This work introduces grid cell signals to mitigate spatial aliasing in such settings. Because grid cells contribute periodic, internally generated spatial signals that vary independently of environmental geometry, they play a key role in disambiguating perceptually identical locations. We integrate multiple modules of analytically constructed grid cells with BVC-driven place cells and show that this leads to a 94--99% reduction in spatial aliasing relative to a BVC-only baseline across three environments: an open environment without obstacles; an environment with a cross-shaped central obstacle creating high visual symmetry; and a maze environment. The greatest improvement occurs in the environment with the highest visual symmetry. These results indicate that grid cells provide information complementary to boundary-based inputs, yielding more reliable place representations in geometrically ambiguous environments. 

---
# MorphoGP: A Nonparametric Framework for Predicting Equilibrium Beach Profiles Under Tidal Influence 

**Authors**: Xi Wu, Yanqing Wei, Hang Yin, Pengze Li, Hongshuai Qi, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.18558)  

**Abstract**: The prediction of equilibrium beach profiles under tidal influence is of fundamental importance for sustainable coastal development, informing shoreline protection strategies and managing coastal ecosystems under changing environmental conditions. However, it remains challenging due to the highly nonlinear interactions among wave, tide, and sedimentary processes. Traditional empirical and numerical models often exhibit limited adaptability across diverse coastal environments, with especially pronounced limitations in beach systems where tidal processes are important . To improve data-driven prediction under these conditions, this study proposes MorphoGP, a unified category-specific Gaussian process framework for predicting equilibrium beach profiles (EBPs) under tidal influence. The framework first introduces a ContourCluster model based on contrastive learning to classify tide-influenced beach morphologies automatically. Within each morphological category, a specialized Gaussian process expert learns statistical associations between environmental descriptors including waves, tides, and sediments and the beach profile's shape. A Gating Net then integrates the outputs of all experts through a probabilistic weighting mechanism to produce the final prediction. Evaluated on data from over 180 beach profiles from tide-influenced coasts along the Chinese coast, MorphoGP achieves improved predictive performance compared with conventional and deep learning models, reducing the test RMSE by about 59.3\% compared with the best baseline and achieving a final RMSE of 0.297 m. The proposed framework provides a physically informed, data-driven tool for equilibrium beach-profile prediction under tidal influence and coastal management, while stronger process-level physical coupling remains an important direction for future development. 

---
# Performance Drift Detection in Machine Learning as a Service (MLaaS) for IoT Environments 

**Authors**: Deepak Kanneganti, Sajib Mistry, Sheik Mohammad Mostakim Fattah, Erik Elmroth, Aneesh Krishna, Monowar Bhuyan  

**Link**: [PDF](https://arxiv.org/pdf/2608.18555)  

**Abstract**: Machine Learning as a Service (MLaaS) is a powerful cloud paradigm enabling data-driven intelligent applications in Internet of Things (IoT) environments, widely adopted across healthcare, smart homes, and industry due to its cost-effectiveness. However, the dynamic nature of IoT frequently alters data distributions, affecting MLaaS stability, while periodic MLaaS updates further introduce performance drift. Unlike traditional ML systems, MLaaS clients operate as black-box users without access to internal data or parameters, making drift detection particularly challenging. To address this, we propose a novel MLaaS Performance Drift Detection framework for IoT environments. The framework first employs an MLaaS extraction model that learns service behavior from input-output pairs and identifies prediction-influenced features. Building on this, the proposed MLaaS Performance Drift Detection (MPDD) model jointly captures variations in input data and MLaaS behavior. We further design an Adaptive-Temporal Performance Drift Detection Mechanism (APDDM) that dynamically adjusts monitoring frequency based on behavioral and data variations, enabling timely drift detection for effective service management. Extensive experiments on real-world datasets demonstrate that MPDD achieves up to 22-25% accuracy improvement over baseline drift detection methods. APDDM provides an average accuracy gain of approximately 4% and reduces the miss detection rate by around 9% compared to fixed-interval monitoring. 

---
# CentaurBench: Benchmarking LLM Capabilities on Augmenting vs. Automating Real-World Work Tasks 

**Authors**: Pattaraphon Kenny Wongchamcharoen, Kris Gulati, Min Min Fong, Abhishek Nagaraj  

**Link**: [PDF](https://arxiv.org/pdf/2608.18554)  

**Abstract**: Most LLM benchmarks rank models on their ability to automate work tasks. In practice, however, models are often used to assist other (human or LLM) agents. The question that drives model selection is therefore not only which model produces the best output, but which model most improves the work of another (weaker) agent. We introduce a unified framework that evaluates the capability of models to automate and augment another agent's performance. Across seven economically grounded real-world tasks, an assistant model writes assistance text for a standardized lower-capacity worker model, which produces the deliverable. In automation mode, the assistant produces the output directly. Outputs are scored through blind pairwise comparisons by an LLM judge panel with task-specific rubrics, replicated across ten runs. Rankings across the two regimes are only modestly correlated, and the automation winner loses augmentation on five of seven tasks. Assistance is not reliably positive. The unaided worker outranks every assisted condition on three tasks, and only one model's guidance beats no guidance on average. These results suggest that automation ability is an incomplete proxy for assistance quality, motivating benchmarks that evaluate models according to the roles they play in human-AI and multi-agent systems. 

---
# Evaluating and Explaining Prompt Sensitivity of LLMs Using Interactions 

**Authors**: Ruiyang Qin, Qingzhuo Wang, Tian Wang, Zhihua Wei, Wen Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.18539)  

**Abstract**: The remarkable capabilities of large language models (LLMs) are often undermined by their instability. Even subtle and semantically irrelevant changes in prompts can cause dramatic fluctuations in performance, a phenomenon known as prompt sensitivity. Previous studies typically evaluate prompt sensitivity by comparing the LLM's final outputs when prompts change. However, such coarse-grained metrics fail to explain the internal reasons for prompt sensitivity. In this paper, we introduce interactions as a fine-grained tool to analyze prompt sensitivity of LLMs. Specifically, we decompose the output score of the LLM into a set of interactions. Each interaction represents a nonlinear relationship involving a set of input variables. We discover that subtle changes to prompts can trigger severe instability in interactions, even when the outputs of the LLM remain the same. To this end, we propose an Interaction-based Prompt Sensitivity (IPS) metric by quantifying changes in interactions when we introduce subtle changes to prompts. We apply the IPS metric to 50 open-source LLMs and uncover four factors that reduce the prompt sensitivity of LLMs, including supervised fine-tuning, increased model scales, dense architectures, and few-shot learning. More crucially, we discover a common mechanism by which these four factors reduce prompt sensitivity: all four factors tend to reduce the prompt sensitivity of low-order interactions (i.e., interactions involving few input variables). 

---
# DART-SD: Diamond-topology Aware Retrieval and Tuning for Self-Distillation of Multi-Turn Tool-Calling Agents 

**Authors**: Hangrui Xu, Jiarui Wang, Yang Yang, Chuanbo Zhu, Fangda Chen, Ziqi Wu, Jingming Cai, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.18524)  

**Abstract**: Equipping Large Language Models (LLMs) with multi-turn tool-calling capabilities is essential for building autonomous agents. However, progress is fundamentally limited by the reliance on full-length trajectory imitation. For tasks involving multiple order-independent sub-goals, the optimal solution space forms a vast combinatorial diamond lattice. Forcing this rich topology into monolithic trajectories causes a severe topological collapse, indiscriminately penalizing valid alternative explorations and severely degrading policy diversity. To address this, we propose DART-SD (Diamond-topology Aware Retrieval and Tuning for Self-Distillation), a novel framework that shifts the paradigm from global forcing to topology-guided localized correction. DART-SD first models the execution process as a converging Interaction-State Transition Graph (ISTG), faithfully capturing the inherent diamond topology of successful and failed exploratory paths. During autonomous rollouts, the framework identifies the Critical Topological Breakpoint (CTB) and retrieves success-supported recovery references. Finally, we introduce a progressive self-distillation paradigm through CTB-guided localized supervision, ensuring that the training loss is calculated exclusively on the generated recovery steps while strictly protecting the valid reasoning prefix from destructive gradient updates. Experiments on complex multi-turn tool-calling benchmarks demonstrate that DART-SD significantly outperforms traditional full-trajectory baselines. 

---
# Prior-Conditioned Gaussian Discriminants for Generalizable AI-generated Image Detection 

**Authors**: Shashank Kotyan, Makoto Shing, Yuki Imajuku, Rujikorn Charakorn, Tarin Clanuwat  

**Link**: [PDF](https://arxiv.org/pdf/2608.18523)  

**Abstract**: Diffusion-based generators have made synthetic images ubiquitous, but detectors often fail under simultaneous shifts in generator, prompt/style, and source-domain. We study AI-generated image detection as a transfer system described by training prior, frozen encoder feature space, and decision rule, and ask when classifier head training adds value beyond what is already separable in modern features. As a controlled diagnostic, we fit a prior-conditioned Gaussian discriminant ladder: closed-form heads built from first- and second-order feature statistics under nested covariance assumptions. On Percept-Lens, a unified protocol over 39 public datasets (7.1 million images), the best rung is frequently competitive with, and sometimes exceeds, released AI-generated image detector heads when matched on both prior and encoder. We further quantify strong sensitivity to the training prior, data-efficiency of moment-based heads, and representation dependence of Gaussian shift metrics, motivating (prior, encoder, head)-level reporting and stronger analytical baselines for AIGI transfer. 

---
# GCNO: Gramian Chebyshev Neural Operator for Physics-Based Compression of Wireless Channels 

**Authors**: Rafid Umayer Murshed, Shahab Hamidi-Rad, Elahe Soltanaghai, Akshay Malhotra  

**Link**: [PDF](https://arxiv.org/pdf/2608.18522)  

**Abstract**: Large antenna arrays allow wireless systems to serve more users and achieve higher data rates, but they also make channel feedback expensive: the receiving device must repeatedly report a large complex-valued channel matrix to the base station. Most neural compressors treat this matrix like an image and replace it with a fixed-length code that only a matched neural decoder can interpret. The message therefore does not adapt to channel complexity, and changing the antenna count typically requires retraining. We ask whether a device can instead report only the few dominant propagation paths underlying each channel. We introduce the Gramian Chebyshev Neural Operator (GCNO), a physics-based, variable-rate compressor that identifies a sample-dependent set of path directions. GCNO uses receive-transmit channel structure to locate paths, a first-order Taylor correction to refine directions that fall between grid points, and least squares to recover their complex strengths. It is trained without path labels, and the base station reconstructs the channel analytically from the transmitted path tuples rather than through a learned decoder. Across three ray-traced environments, GCNO achieves better reconstruction accuracy at the same payload - or lower payload at the same accuracy - than neural feedback baselines, and transfers to unseen antenna counts without retraining. 

---
# OptiModNet: A UNet-Transformer Hybrid with Grouped-Query and Channel Attention for Optic Disc and Cup Segmentation 

**Authors**: Soumili Ghosh, Debapriya Roy, Aryan Das, Bikash Santra  

**Link**: [PDF](https://arxiv.org/pdf/2608.18516)  

**Abstract**: Precise segmentation of the optic disc and cup is critical for the early detection and diagnosis of glaucoma. However, achieving consistently high performance across datasets while maintaining low computational requirements remains a significant challenge. In glaucoma detection, low-computation methods are crucial for enabling rapid, large-scale screening and facilitating deployment in resource-limited clinical environments. While deep learning models such as UNets, Vision Transformers (ViTs), and Diffusion models have demonstrated strong segmentation performance but these methods often come with substantial computational overhead. UNets are efficient at capturing local features but are limited in modeling global contextual information. Conversely, ViTs excel at long-range dependency modeling but are computationally intensive. Hybrid architectures, such as UNetR, which combine transformer-based encoders with UNet-style decoders, have shown improved performance but while incurring additional complexity. Considering these, in this work, we propose OptiModNet, a light weight novel hybrid architecture tailored for optic disc and cup segmentation. The model integrates diverse attention mechanisms at multiple stages of the network to enhance both local and global feature representation. We include an Aggregated Pyramid Loss that supervises predictions at multiple decoder depths, to promote better gradient flow and structural consistency. We evaluate OptiModNet on the REFUGE2 dataset for both optic disc and cup segmentation tasks. Our method achieves state-of-the-art performance, exceeding existing approaches by over 2.5\%, while maintaining high efficiency with only 3.73 GFLOPs and 1.93M parameters. The code is available at this https URL. 

---
# Science Done on a Machine by a Machine: AI Agents in Computational Chemistry 

**Authors**: Pavlo O. Dral, Hassan Nawaz, Arif Ullah  

**Link**: [PDF](https://arxiv.org/pdf/2608.18508)  

**Abstract**: We are witnessing an explosion of agentic systems for computational chemistry simulations: from half a dozen in 2024 to a dozen in 2025, and the current number approaches fifty, surveyed in this Perspective as of 8 August 2026. The capabilities of these agentic systems are shifting from assisting in performing a selection of computational tasks to autonomous design and execution of \textit{in silico} experiments, their analysis, and even manuscript writing. The ultimate destination is a fully autonomous AI scientist, where the entirety of computational chemistry is performed on a machine by a machine, without human supervision. While we are not there yet, and all reported systems currently involve a human in the loop, the trend is unmistakable. Even building specialized agentic systems for computational chemistry is increasingly commoditized by generalist agents, which may in the end replace the need for the specialized ones altogether, since adding a new capability will be as easy as asking AI to do it for you. Both the explosion in their number and the very limited adoption beyond their own developers point that way, and we close this Perspective on what it leaves us to do. The speed and scale of disruption agentic systems are bringing to computational chemistry leave many of us dumbfounded about the field's future and what we should spend our efforts on, as already established specialists, teachers, and students, and we have no answer. 

---
# Physics-Unrolled Neural Operator for Wireless Field Modeling 

**Authors**: Rafid Umayer Murshed, Saif Ur Rahman, Mingyue Tang, Elahe Soltanaghai  

**Link**: [PDF](https://arxiv.org/pdf/2608.18495)  

**Abstract**: Radio maps are essential for wireless decision-making tasks such as access-point placement, coverage planning, and localization, but their fine spatial details are governed by complex propagation effects and are costly to simulate accurately. Machine learning offers a path to high-fidelity radio-map prediction without running expensive high-fidelity simulations for every scene. However, generating high-quality training labels at scale is also difficult: the affordable labels come from finite-ray simulations, which are richer than low-fidelity inputs but carry residual Monte Carlo noise. We address this challenge with Physics-Unrolled Hybrid Neural Operator (PU-HNO), a three-stage cascade that predicts high-fidelity indoor radio maps from low-fidelity ray-tracing outputs and scene priors by progressively capturing reflection, diffraction, and scattering effects, rather than treating radio maps as generic images. We prove that, under conditionally unbiased label noise, the model can learn stable propagation structure and outperform its own training labels. Experiments across diverse floorplans show that PU-HNO outperforms image-to-image baselines, wireless learning models, and monolithic neural operators across both image-quality and wireless deployment metrics. 

---
# Partition the Support, Reconstruct the Residual: Training-Free Sparse Attention for Video Generation and World Models 

**Authors**: Pardis Taghavi, Reza Langari, Gaurav Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2608.18484)  

**Abstract**: Training-free block-sparse attention can accelerate video transformers, but row-wise attention concentration does not by itself specify an executable sparse operator. Queries sharing a block route may have poorly overlapping supports, while retained attention mass alone does not determine the post-softmax error from skipped interactions. We show that partition geometry affects both pooled support and the predictability of the remaining residual from the sparse output. We introduce SparsePR, which combines Response-Coupled Partitioning with Probe-Fitted Residual Reconstruction. Sampled-query key responses form paired K/V groups, whose centroids induce query-response coordinates for shared routing. A small set of exact query rows then calibrates a call-specific affine correction from the sparse output within the output subspace observed in the probe residuals. Across four heterogeneous video generation and world models, SparsePR consistently reduces attention-reconstruction error. Ablations show that probe fitting accounts for most of this reduction, while response-coupled partitioning lowers hard-drop error and improves reconstruction under a finite probe budget. SparsePR preserves generation quality at 22.0-26.0% realized executed-pair density while achieving 1.48x-2.61x end-to-end speedups. Project page: this https URL 

---
# Coverage-Driven RTL Assertion Generation with Formal Exploration and Neuro-Symbolic Refinement 

**Authors**: Zhiyuan Yan, Ziyue Zheng, Hongce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18482)  

**Abstract**: Hardware functional verification relies on high-quality assertions to expose design bugs and establish confidence in Register Transfer Level (RTL) designs. Yet existing assertion mining methods still struggle to produce complete and reliable assertion sets: random or limited traces fail to cover hard-to-reach behaviors, and one-shot generation provides little feedback about what remains unverified or how the assertion set should be improved. As a result, critical design behaviors can remain uncovered even when many assertions are generated. We present NeuroAssertion, a coverage-driven assertion generation framework that combines formal trace generation, syntax-guided synthesis (SyGuS), and an agent-inspired refinement process within a unified framework. Our framework first converts hard-to-reach control-flow conditions into formal reachability objectives, uses model checking to generate behaviorally diverse traces, and mines initial assertions from these traces with SyGuS. It then performs targeted agent-inspired refinement under verification feedback: one LLM first proposes candidate assertions for uncovered regions, and if a candidate fails formal checking, a second LLM generates a repair grammar that guides constrained symbolic synthesis in a neuro-symbolic repair procedure. Experimental results show that this framework delivers around 2X more assertions and about 2X higher mutation coverage than traditional assertion mining methods. 

---
# ERASE: EaRly bAckpropagation SchEdule for Faster Training of Modern Recommendation Systems 

**Authors**: Ergan Shang, Flavio Sales Truzzi  

**Link**: [PDF](https://arxiv.org/pdf/2608.18469)  

**Abstract**: Lightweight proxy models enable rapid experimentation without repeatedly training frontier-scale systems, but their small kernels often leave modern accelerators underutilized. Conventional training compounds this inefficiency by scheduling the forward and backward passes as disjoint phases, so spare capacity in one cannot be filled by work from the other. We reinterpret the detachment mechanism of Forward-Forward (FF) as a scheduling primitive: given a local objective, detaching a block's output removes downstream gradient dependencies, making its backward pass ready when its forward pass finishes. ERASE launches each detached subgraph's backward pass early on a separate CUDA stream, overlapping it with subsequent forward work. Execution trace on a lightweight transformer demonstrates this overlap and its limit: a kernel that saturates the device leaves no capacity for concurrency. On a large-scale click-through-rate model, detaching six dense subarchitectures improves training throughput by up to $9.51\%$ while keeping normalized entropy close to the baseline. 

---
# Formal Verification of Romanov's Triplet Logic: A Verified Filter for Sliding-window 3-CNF with Application to Structured Formulas 

**Authors**: Dmitry V. Alexandrov  

**Link**: [PDF](https://arxiv.org/pdf/2608.18445)  

**Abstract**: We present the first mechanised formalisation of Romanov's Triplet Logic (TLS) in the Rocq proof assistant. TLS is a triplet-based combinatorial framework for reasoning about compatible paths through layered triplet structures, called Compact Triplets Structures (CTS), and their intersection via Romanov's Effective Procedure, which we refer to as Simple Vertex Intersection (SVI). Originally motivated by Boolean satisfiability, TLS constitutes a self-contained mathematical theory whose formal properties had not been previously established. We formalise the core of TLS in Rocq, including Compact Triplets Formulas (CTF), CTS, hyperstructures, clearing, and SVI. For the well-formed sliding-window fragment we verify a clause-by-clause CNF-to-CTF translation, the clearing procedure, and aligned intersection, and we prove explicit polynomial-time bounds for the filter stages. Our main contribution is a precise correctness boundary: the existence of a joint satisfying set implies non-emptiness of SVI, but the converse does not hold in general; for aligned structures we recover a complete bi-implication, extended to systems of structures. We also formalise soundness of grouped-window translation and exhibit a formal counterexample to its completeness. We introduce VFR, an extracted OCaml prototype that provides a verified decision procedure for the sliding-window fragment and a sound one-sided filter for general 3-CNF, with a Python runtime and reproducible Docker packaging. Benchmarks on random and structured instances confirm the predicted behaviour, and the complete toolchain is available as a curated Zenodo artifact. The Rocq development comprises more than 23,000 lines of code across seventeen files, with 427 proved lemmas and theorems and zero admitted goals. 

---
# Pedagogical AI in Mental Health: A Tri-Stream Fine-Tuned LLM Framework for Automated Clinical Supervision and Risk Triage 

**Authors**: Shreeya Sharma, Ravish Gupta, Saket Kumar, Abhishek Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2608.18438)  

**Abstract**: Modern mental healthcare faces a critical shortage of senior supervisory oversight, leading to a "supervision gap" where novice therapists manage high-stakes risks with delayed professional feedback. This paper proposes a new framework utilizing a fine-tuned Mistral-7B-instruct model as an automated "Supervisor-in-the-Loop" system. By leveraging 106 sessions from the DAIC-WOZ dataset, the model performs a tri-stream analysis: (1) Therapeutic Alliance tracking via semantic adherence, (2) Latent risk prediction using attention-weighted analytics, and (3) Supervisory Triage via a Dynamic Clinical Urgency Index (D-CUI). Our multi-modal VAL (Visual-Acoustic-Linguistic) framework achieves 95% technique identification accuracy [95% CI: 75.1%-99.9%], alliance assessment MAE of 0.105 on a 5-point scale [95% CI: 0.059-0.151], therapeutic fidelity alpha = 0.423, and mean D-CUI of 0.370 [95% CI: 0.322-0.419]. Training converged in 105 steps with 85.2% loss reduction on a single Tesla T4 GPU. The system reduces supervisory triage latency from 72 hours to real time (~10 seconds per session), enabling proactive intervention in high-risk cases. The system addresses the cold-start problem through Bayesian priors and implements timestamp-based modality synchronization for robust multi-modal fusion. 

---
# Mechanistic Interpretability of Structure-Aware Numerical Reasoning in LLaMA 3.1 8B 

**Authors**: Rahul Chowdhury, Timothy A Rupprecht, Senhao Cao, Jiahao Liu, Octavia Camps, David Bau, Pu Zhao, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18419)  

**Abstract**: Recent work has shown that large language models (LLMs) exhibit strong numerical sequence modeling capabilities and show promise in time-series prediction. While LLMs display in-context learning capabilities, the mechanisms with which they accomplish time-series prediction remain unclear. Specifically, whether they truly understand the underlying structure, which at a minimum requires reasoning over first differences in the sequence of numbers. To study this, we investigate Llama 3.1-8B from a mechanistic interpretability point of view. Mechanistic interpretability is an emerging field concerned with the reverse engineering of the algorithms learned by neural networks such as LLMs. To assess Llamas' numerical sequence modeling capabilities and to facilitate our mechanistic interpretability analysis, we create a sequence modeling task that cannot be solved without picking up structural cues. Specifically, we sample n random numbers and repeat them with an offset. We find that Llama displays strong performance on our tasks suggesting that it can pick up on the underlying structure. To understand the mechanisms that allow it to do so, we perform probing experiments and activation patching based counterfactual analysis. Probing reveals that the model computes and stores first differences in its internal representations without explicit supervision, indicating that it tracks structural information about the sequence. Activation patching reveals that Llama retrieves the relevant first-difference with a mechanism similar to an induction circuit and subsequently adds it to the current value. Notably, our work represents one of the first studies to identify this form of concept induction in LLMs. 

---
# Vector Symbolic Policy Gradient 

**Authors**: Ryozo Masukawa, Sanggeon Yun, SungHeon Jeong, Hyunwoo Oh, Raheeb Hassan, Pietro Mercati, Nathaniel D. Bastian, Mahdi Imani, Mohsen Imani  

**Link**: [PDF](https://arxiv.org/pdf/2608.18404)  

**Abstract**: We answer this question with Vector-Symbolic Policy Gradient (VSPG), a discrete-action actor that represents each action by a unit-norm hypervector and scores it by similarity to the encoded state. Under the standard softmax policy-gradient surrogate, we prove that its update is exactly advantage-weighted hypervector bundling followed by normalization, and therefore supports standard advantage estimators. We further show that each trained action hypervector is a fixed-size compressed kernel memory, storing an advantage-weighted kernel expansion over visited states and transferring evidence according to the encoder-induced similarity. This provides a concrete mechanism that can support sample-efficient learning without increasing inference-time memory. Finally, for bipolar action memories, we prove that greedy action selection is stable under random bit flips, with failure probability decaying exponentially in the hypervector dimension. VSPG thus connects VSA action memories, log-linear policy gradients, and kernel policy search while providing a quantitative robustness guarantee. 

---
# LEDGER: Claim-to-Evidence Trace Graphs for Auditing LLM Agents 

**Authors**: Daehong Kim, Haichao Miao, Shusen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18398)  

**Abstract**: Large language model (LLM) agents can now carry out long-horizon technical workflows involving complex tool use, code execution, file edits, and generated artifacts. As agents do more work faster, the productivity bottleneck shifts from producing outputs to auditing whether those outputs are correct and trustworthy. Agent observability systems make fine-grained execution events visible, but visibility alone still leaves reviewers to reconstruct which actions, artifacts, and validation steps matter for a particular conclusion. We introduce LEDGER - Layered Evidence and Decision Graphs for Execution Review, a tracing and review system that builds layered trace graphs over observed agent sessions. LEDGER preserves Trace Records while grouping them into Evidence Nodes and Workflow Nodes, representing artifacts as evidence anchors, and adding typed semantic edges that connect claims to supporting actions, artifacts, and checks. Through data-analysis and coding examples, we show how the resulting traces expose workflow decisions, artifact lineage, repair steps, validation coverage, and claim-support paths for evidence-centered audit. 

---
# TTSD-FAR: Test-Time Self-Distillation with Fisher-Anchored Restoration for Missing-Modality Emotion Recognition in LVLMs 

**Authors**: Muhammad Haseeb Aslam, Alessandro Koerich, Marco Pedersoli, Ali Etemad, Eric Granger  

**Link**: [PDF](https://arxiv.org/pdf/2608.18386)  

**Abstract**: Large video-language models (LVLMs) have shown remarkable performance on multimodal tasks like multimodal emotion recognition (ER) in the wild. ER is inherently multimodal, requiring a joint understanding of facial expressions, vocalizations, language, biosignals, and gestures. However, real-world deployment remains challenging: modalities may be missing or noisy at test time. Partial observations can be viewed as a distribution shift relative to the complete-modality distribution. SOTA TTA methods based on entropy minimization or perplexity reduction do not transfer to autoregressive LVLMs, while retrieval augmented generation (RAG) degrades when the observed modality is weak. Because no ground-truth supervision exists to verify individual updates, adaptation across this stream risks accumulating drift and degrading once the model departs from a reliable solution. An effective solution must therefore adapt to arbitrary missing-modality patterns and remain effective during continual adaptation. We address both jointly with Test-Time Self-Distillation (TTSD), a parameter-efficient framework in which a frozen teacher, trained on complete modalities, guides an adaptive low-rank student via self-distillation, updating only a negligible number of parameters. Stability is built into this same loop through Fisher-Anchored Restoration (FAR), which monitors Fisher information stability to detect convergence versus drift and restores the student toward the teacher's anchor when distributional shifts are identified. Our experiments on MELD, DFEW, and BAH under 0%-50% missing modalities show that this unified adaptation-restoration design consistently outperforms entropy-based adaptation, RAG, and perplexity-based generation over long adaptation horizons, where baselines without restoration progressively degrade while TTSD-FAR remains consistent. 

---
# Selection, Recombination, or a Fresh Solve? A Candidate-Free Control for Single-Pass Test-Time Aggregation 

**Authors**: Guiv Farmanfarmaian  

**Link**: [PDF](https://arxiv.org/pdf/2608.18379)  

**Abstract**: When every candidate is wrong, correct-candidate selection is unavailable, yet the aggregation call can still solve the problem afresh. A correct aggregate answer may therefore reflect recombination, fresh solving, or both. For efficient test-time reasoning, the relevant question is whether candidate context adds value beyond the additional generation pass. We introduce the missing candidate-free control under the same maximum output-token allowance and stratify by the number of correct candidates. Across AIME-2025 and HMMT-2025 with Qwen3-4B, candidate conditioning improves accuracy when multiple candidates are correct ($\Delta_{\mathrm{cand}}$(c2+) = +0.290), lowers accuracy when every candidate is wrong ($\Delta_{\mathrm{cand}}$(c0) = -0.123), and remains unresolved in the one-correct regime. The c2+ and c0 conclusions survive a conservative correction for the adaptive two-benchmark procedure. Under this counterfactual, the interpretation of all-wrong recovery reverses at this scale: conditioning on an all-wrong candidate pool lowers accuracy relative to a fresh solve. Original-format matching and placebo results characterize the failures descriptively but leave their mechanism unresolved. Within a separate structured intervention, explicit answer fields causally steer outputs toward their values; masking yields no measurable accuracy improvement, and equivalence with the original format was not established. The evidence is limited to one Qwen3-4B family, two mathematics benchmarks, first-answer-truncated candidate fragments, and single-pass prompted aggregation. 

---
# One Gate Is Not Enough: Composing Stateful Pre-Action Controls for Agentic AI 

**Authors**: Gaston Besanson  

**Link**: [PDF](https://arxiv.org/pdf/2608.18360)  

**Abstract**: Agentic AI systems take consequential actions governed by more than one pre-action control at once: authority, resource, and evidence gates that can admit, degrade, or remediate an action before it executes. This paper's central object is remediation-induced control coupling: a remediation applied by one control can change the action, evidence, or context another control evaluates, invalidating that control's earlier judgment. We formalize this coupling and give a remediate-and-regate protocol that restores per-action soundness in the current bounded, idempotent setting under its stated assumptions. We further show that the two implemented remediation operators (evidence substitution and resource-budget downroute) do not commute -- a finite-model checker finds concrete counterexample instances -- making remediation order part of the control-plane semantics rather than an implementation detail. A governed evidence buffer that trusts its own most recent admitted write is a further instance of the same problem at the level of state -- current admissibility does not imply future reference trustworthiness -- and is vulnerable to poisoning from declared-uncovered defect classes; two mitigations reduce, not eliminate, that exposure. Supporting results establish the exact condition under which positive-weight linear aggregation of gate outcomes can compensate a member veto, a unified cross-control Evidence Set, and that composition manufactures no new detection coverage, reported honestly. Empirically, on a deterministic open-data artifact composing three published engines unmodified, CH1-CH5 meet their registered decision rules across all 30 pre-registered seeds; CH6 does so under W1 but not under the smaller W2 workflow, reported as such. This is a mechanism demonstration on open payload data with a synthetic metadata layer, not a claim about production prevalence. 

---
# Task-Conditioned Least-Privilege Learning for Executable Terminal and MCP Agents 

**Authors**: Alexander Tu, Michael Tu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18351)  

**Abstract**: Tool-using large language-model agents can complete a task while exercising authority that the user did not grant or the task does not need, causing excess-authority errors. Traditional permission gating systems alone for validating agent environments are insufficient. We study whether post-training can teach a 4B-parameter model to choose task-conditioned authority in executable terminal and Model Context Protocol (MCP) environments to complement those measures. We propose a framework where each action is audited before execution and again from observed effects along six dimensions of risk. This auditing is conducted using deterministic verifiers that score completion, evidence, exact state, prohibited attempts, and safe success. In conjunction with predefined task-specific sufficient-authority envelopes, we determine task-specific excess privilege values for trajectories, which are then optimized for in post-training. We find that after training using this framework on Qwen3.5-4B over 1,500 tasks, the selected seed reaches 98.48% safe success across 2,896 evaluation episodes spanning all 500 held-out tasks, compared with 64.36% for the base policy, and reduces excess-authority error events from 4.56% to 0.79%. Furthermore, external tests show capability retention and prompt-directed improvement. A 400 task continuation study also found evidence of generalization, reducing excess-authority events by 6.99 percentage points while maintaining previous capabilities. We conclude learned restraint through least-privilege aware post-training is therefore useful as an additional control layer for tool-using agents in executable terminal and MCP environments, but it does not replace permission gates and sandboxing. 

---
# Coupled-cluster molecular properties across the main group that extrapolate beyond training size 

**Authors**: Wenhao He, Xu Chen, Noah Song, Haowei Xu, Tim S. Hindges, Bohan Li, Zihan Lin, Yu Yao, Avetik R. Harutyunyan, Fang Liu, Yao Wang, Hao Tang, Ju Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.18346)  

**Abstract**: Coupled-cluster theory defines the accuracy standard for molecular electronic-structure properties but scales too steeply for routine application, whereas density-functional theory is affordable yet systematically biased. We resolve this trade-off with a single equivariant network, MEHnet-MG, that predicts an effective one-electron Hamiltonian from one inexpensive B3LYP/def2-SVP calculation and derives a broad suite of properties from it (energy, optical gap, dipole, quadrupole, polarizability, Mulliken atomic charges, and Mayer bond orders) at coupled-cluster accuracy across nine main-group elements, including the under-served phosphorus, sulfur, and chlorine chemistries. The model is trained on a new in-house dataset of multi-property labels computed at the CCSD(T) level for all nine elements. On a held-out test set, it reduces the error of every property by a factor of 3.8 to 230 relative to semi-local, hybrid, and double-hybrid DFT (referenced to composite CCSD(T)/cc-pVTZ; Methods), while adding only ~25 ms wall time per molecule, delivering coupled-cluster-quality predictions at the cost of a single DFT calculation. Critically, deriving every property from a predicted Hamiltonian rather than pooling per-atom features builds the correct size-scaling into the model architecture: on pi-conjugated oligothiophenes it matches finite-field CCSD polarizability and the EOM-CCSD optical gap to ~2% at the largest sizes where those references remain affordable (44 and 37 atoms, where a single CCSD field point already costs ~500x the model's entire inference) and extrapolates the corrected trends to 58-atom chains, a regime where pooling-based architectures fail by construction. Accurate extrapolation is therefore set by the model's inductive bias rather than by the training data. 

---
# Low-Power, Neuromorphic, Acoustic Anomaly Detection for Persistent Machine Monitoring 

**Authors**: Steven C. Nesbit, Victor M. Vergara, Michael A. Felix, Evan T. Kain, Luis R. García Carrillo, Gerd J. Kunde, Andrew T. Sornborger  

**Link**: [PDF](https://arxiv.org/pdf/2608.18341)  

**Abstract**: Persistent acoustic monitoring can detect machine faults without physical contact, but always-on inference is constrained by power, latency, and deployment complexity. We demonstrate autoencoder-based acoustic anomaly detection on an Intel Loihi 2 neuromorphic processor under clean and noisy conditions. Log-mel features are computed off chip; normalization, autoencoder inference, L1 reconstruction scoring, and thresholding run on chip. In a clean, microphone-position-invariant ToyADMOS ToyCar benchmark, the on-chip model achieves 0.9959 AUC and 0.9785 standardized pAUC at maximum false-positive rate 0.1. In the DCASE 2026 Task 2 ToyCar noisy benchmark, the model achieves source AUC 0.7990, target AUC 0.6466, and pAUC 0.6426, exceeding reported baseline metrics. Power profiling on a 16-chip Loihi 2 VPX system shows real-time throughput with 0.0406$\unicode{x2013}$0.0426 mJ dynamic energy per sample, two orders of magnitude lower than both a CPU and GPU. These results support neuromorphic acoustic anomaly detection as a practical candidate for low-power, persistent machine monitoring. 

---
# From Inference to Adaptation: A Unified Optimal Transport View of Vision Language Model 

**Authors**: Qi Yu, Zhichen Zeng, Katherine Tieu, Xiyuan Yang, Ruizhong Qiu, Yuchen Yan, Lihui Liu, Yanjun Zhao, Lingjie Chen, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2608.18339)  

**Abstract**: Vision-language models (VLMs) have demonstrated remarkable zero-shot capabilities yet remain sensitive to real-world distribution shifts during inference. Although significant efforts are devoted to adapting VLMs at test time, they rely heavily on noisy pseudo-labels predicted directly from raw embedding similarities during inference, which are unreliable under distribution shift and mislead the adaptation. To avoid noise amplification, existing works craft coarse-grained surrogate objectives during adaptation, which fail to explicitly model sample-level relationships across different modalities, creating objective mismatch with inference, thus leading to marginal performance improvement. In this work, we aim to bridge the detached objectives of inference and adaptation for VLMs, and propose a principled VLM TTA method called \algname. For VLM inference, we formulate the zero-shot image classification task as a cross-modal alignment problem encoded via a Wasserstein OT formulation, providing robust pseudo-labels at the sample-level to effectively adapt VLMs. For VLM adaptation, we adopt a soft-label InfoNCE loss to adapt VLMs based on the OT-induced pseudo-labels, leveraging fine-grained supervisions to explicitly model relationships of individual image-text pairs via contrastive learning, which empowers accurate inference at the same granularity. Moreover, we theoretically reveal that the InfoNCE loss can be neatly reformulated as a Wasserstein OT formulation, thereby unifying the objectives of the inference and adaptation of VLMs to achieve their mutual benefits. Extensive experiments demonstrate the effectiveness and efficiency of our methods, outperforming the best-performing methods by up to 7% with state-of-the-art efficiency. 

---
# FedCoRe: Target-Adaptive Completion for Missing Modalities in Healthcare Federated Learning 

**Authors**: Holger R. Roth, Ziyue Xu, Peter Cnudde  

**Link**: [PDF](https://arxiv.org/pdf/2608.18311)  

**Abstract**: Federated multimodal models often assume every site has every modality, although hospitals differ in access to EHRs, chest radiographs, and ECGs. We study this setting on a MIMIC-derived respiratory deterioration task with simulated FL clients and introduce FedCoRe (Federated Cross-Modal Representation Completion). FedCoRe learns representation- or logit-space corrections rather than generating synthetic ECGs or CXR images. When a client observes a modality that may be missing at deployment, it evaluates the same example with and without that modality to obtain paired supervision. Only clients with such pairs update the completion module, and validation may retain the unchanged prediction. We freeze the trained multimodal predictor during evaluation so that measured differences come only from completion. Hiding ECG reduced AUROC by about 0.085; paired-example FedAvg restored 0.0415 AUROC, or 49.0% of the lost performance. We therefore report two distinct effects: paired-example FedAvg partially recovers the missing-ECG gap, while validation-selected completion is a task-specific classifier-logit correction rather than literal ECG recovery. For CXR, effect-aware completion recovers 52.8% of the loss in a controlled test where CXR is hidden. Paired-example FedAvg transfers part of this effect, but validation keeps the no-completion baseline for deployment cases whose inputs lack CXR. Thus, FedCoRe should be read as a validation-gated completion/correction framework: it can recover missing-modality signal in supported settings, but it should be deployed only when paired examples and validation evidence support that modality. 

---
# FairGlucose: A CGM Fairness Benchmark Reveals Subgroup Disparities Hidden in Population-Level Validation 

**Authors**: Junjie Luo, Xuzhe Zhi, Rui Han, Abhimanyu Kumbara, Anand K. Iyer, Mansur E. Shomali, Ritu Agarwal, Guodong Gordon Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.18296)  

**Abstract**: As CGM-based AI tools approach clinical deployment, whether their accuracy is equitable across patient demographics remains insufficiently tested. To enable this evaluation, we constructed FairGlucose, a 300-patient CGM cohort balanced across 12 demographic strata (age x gender x type 1/type 2 diabetes), with 132,480 forecasting samples and 3,945 unique behavioral events (meals, exercise, medication) logged by 81 patients. Benchmarking 33 models across four families on 2-hour glucose forecasting, we find that population-level external validation can conceal substantial subgroup disparities. Aggregate out-of-distribution metrics appear stable (approximately 1.0), yet subgroup-level ratios range from 0.8 to 1.4, with T1D patients showing 6 mg/dL higher prediction error than T2D (p < 0.001). This disparity persists across all 33 models, suggesting a property of the prediction task rather than any single architecture. Further analysis shows that subgroup performance gaps align with the proportion of clinically hard cases, and that input-length sensitivity varies across demographics, motivating personalized configurations. Frontier LLMs underperform specialized neural models by 1-6 mg/dL; behavioral events contribute negligibly (approximately 0.1 mg/dL) even under oracle event access. These findings establish that population-level validation alone is insufficient for equity assessment of digital health AI, motivating subgroup-disaggregated reporting as a default standard. 

---
# Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements 

**Authors**: Naoki Egami, Sooahn Shin  

**Link**: [PDF](https://arxiv.org/pdf/2608.18294)  

**Abstract**: An increasing number of scholars use AI to measure variables they subsequently include in downstream analyses. Although AI-measured variables are often analyzed as if observed without error, ignoring prediction errors in automated measurement leads to substantial bias and invalid confidence intervals in downstream analyses, even if AI measurement accuracy is high, e.g., above 90%. Existing solutions, such as design-based supervised learning and prediction-powered inference, combine error-prone AI-based measurements with gold-standard labels, which may be costly and difficult to obtain in some application areas.
In this paper, we propose debiased inference with multiple imperfect measurements (DMM), a framework that combines multiple error-prone AI measurements to enable valid downstream inference without gold-standard labels. Building on the established results on CP decomposition, DMM assumes that these measurements are independent conditional on the latent true label and observed unit-level features, such as text features represented by embeddings. This framework allows for unknown misclassification rates to vary across annotation methods (e.g., large language models) and across units of annotation (e.g., texts). Under this assumption, we use semiparametric inference theory to prove that the DMM estimator is consistent and asymptotically normal, enabling valid inference for a wide range of downstream statistical analyses common in the social sciences. Our simulation results show that DMM yields valid inference and that adding accurate, though imperfect, measurements can improve efficiency. Focusing on common applications of large language model annotations, we also develop diagnostics to assess the conditional independence assumption. 

---
# What Makes Software Issue Resolution Tasks Difficult for Agents? 

**Authors**: Ebtesam Al-Haque, Brittany Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2608.18280)  

**Abstract**: Background. Advances in agentic systems are simultaneously, and rapidly, saturating benchmarks. Despite this often discussed phenomena, benchmark scores remain difficult to interpret due to the lack of control and characterization of task difficulty. More specifically, we currently have little understanding of what makes one task harder than another, and to what extent task difficulty is predictable from static task properties. Aims. We propose a measurement framework to investigate and systematically quantify what structural properties of software tasks correspond to agent success rates for issue resolution tasks. Method. We conducted a large scale empirical study on CoderForge-Preview, the largest open dataset of coding agent trajectories to date, by extracting features across task patch, repository and prompt. We evaluated the predictive power of each feature against task outcomes using ensemble methods, SHAP attribution, and effect size analysis. Results We found that task difficulty is substantially predictable from static features (AU C = 0.863) and is largely driven by patch fragmentation and repository scale. Prompt linguistic features become visible among top contributors for tasks in the mid-band, revealing a layered structure of difficulty. Conclusion. The difficulty of an issue resolution task is encoded in its structure. This enables static, pre-hoc difficulty estimation and lays the groundwork for difficulty-controlled benchmark construction for evaluation of agents. 

---
# SeisEvo: Evolution of Seismic Data Reconstruction Algorithms by Agents 

**Authors**: Yingjie Xu, Siwei Yu, Jianwei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.18272)  

**Abstract**: Classical seismic data reconstruction relies on manually designed structural priors and iterative operators, whose coupled design space is far larger than manual trial and error can explore systematically. Deep-learning methods encode the reconstruction rules in learned weights rather than in an explicit operator that can be inspected and modified. We propose SeisEvo (Seismic Algorithm Evolution), which does not optimize a single reconstruction result but searches for the algorithm that produces it. Starting from a classical reconstruction algorithm, an LLM-driven multi-agent search modifies only the components that the user has opened for editing, without prescribing the mechanism to be discovered. Candidates that violate the physical constraints of the task are rejected outright, and the remaining ones are scored by execution. The output is neither an agent system nor a neural network, but a standalone white-box algorithm that requires no agent or neural network at inference time. For interpolation without added noise, the search discovered a residual-gated, phase-aligned dip-consistency projection; Evo-POCS improves the SNR over classic POCS by 3.49 dB on average across missing ratios from 30% to 70%. For simultaneous interpolation and denoising, it discovered a reliability-grouped singular-value shrinkage; Evo-MSSA improves the average reconstruction SNR by more than 7 dB over classic MSSA and by more than 3 dB over a stronger rank-reduction baseline. Both operators retain their gains on data not used during the search. To the best of our knowledge, this is the first study to formulate the design of a seismic reconstruction operator as a constrained, LLM-driven program evolution task. Agentic algorithm evolution can thus complement deep learning in discovering explicit, inspectable, and deployable seismic processing algorithms. 

---
# How AI Prompts Can Teach Us About the Structure of Human Behavior 

**Authors**: Matthew O. Jackson, Benjamin S. Manning, Yutong Xie, Walter Yuan, Qiaozhu Mei  

**Link**: [PDF](https://arxiv.org/pdf/2608.18265)  

**Abstract**: We introduce a general, easy-to-implement AI-based method for studying the structure and complexity of human behavior. We assign a large language model a ``type vector'' and then prompt it to choose actions across settings in which we observe human choices. For instance, the type vector (2,4) becomes ``You are a player characterized by the following profile: 2 out of 5 in Altruism, 4 out of 5 in Risk Aversion,'' after which it is prompted to make choices. We vary the dimensions (e.g., Altruism, Fairness, Trust, $\dots$) and values (e.g., 1--5) to minimize distance to human choices. Applying the method to 119,147 decisions made by 78,657 subjects from more than 35 countries across 10 classic economic game roles, we find that human behavior can be closely matched using three dimensions: Risk Aversion, Strategic Sophistication, and Trust. Moreover, the types needed to fit individuals across games cluster into fewer than a dozen groups, and can predict behavior in held-out games with different rules and available actions. The results suggest that behavior across diverse settings can be approximated by a low-dimensional, portable representation, supporting the possibility of general yet parsimonious theories across the behavioral sciences. More broadly, the method can provide insights into the structure of many human behaviors. 

---
# Visual-Prompt Guided Wildlife Instance-Level Recognition 

**Authors**: Mufhumudzi Muthivhi, Jiahao Huo, Terence van Zyl, Fredrik Gustafsson  

**Link**: [PDF](https://arxiv.org/pdf/2608.18246)  

**Abstract**: Fine-grained wildlife re-identification remains a challenging area in research. Current state-of-the-art approaches apply a detection and re-identification pipeline. We propose a one-stage end-to-end detection and re-identification model that performs identity searching within the latent space. We adopt DINOv2 for robust spatial geometry and MegaDescriptor for wildlife re-identification. We enhance latent queries with prompt re-identification features. A detection decoder queries the scene latent space to establish object boundaries around the target identity. Preliminary findings reflect a competitive mean average precision score of 30.584% compared to the state-of-the-art two stage approach of 44.89%. Qualitative results depict effective bounding and identification of animal identities. 

---
# Bidirectional representational alignment between biological and artificial neural networks 

**Authors**: Samuel Kostousov, Abhinn Kaushik, Brokoslaw Laschowski  

**Link**: [PDF](https://arxiv.org/pdf/2608.18244)  

**Abstract**: Recent work has shown that representational alignment between biological and artificial neural networks is asymmetric: model representations predict neural responses much better than neural responses predict model representations. This asymmetry raises the question of whether representational geometry contributes to bidirectional representational alignment. We hypothesized that steering representational geometry during training can systematically influence bidirectional alignment. To test this hypothesis, we developed a computational framework that integrates spectral regularization with bidirectional predictivity analyses. As an initial demonstration, we evaluated our framework using self-supervised contrastive vision models. Steering the spectral geometry of the learned representations substantially increased reverse predictivity with modest reductions in forward predictivity, yielding a 55% relative improvement in bidirectional predictivity. These improvements were accompanied by reduced effective dimensionality and a reorganization of the shared representational subspace, within which forward and reverse predictivity became approximately symmetric at intermediate spectral exponents. Overall, these findings demonstrate that representational geometry can be systematically steered to modulate bidirectional representational alignment between biological and artificial neural networks. 

---
# GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction 

**Authors**: Ziyang Cheng, Tianshu Tang, Jinxin Lan, Xinze Chen, Yuhan Gong, Zhichao Liu, Changzhong Wu, Yahao Mao, Zongyan Deng, Mingxuan Ma, Huasen Xi, Yilong Liu, Yutong Wu, Xiaofeng Wang, Yang Wang, Yun Ye, Guan Huang, Xiaojie Jin, Zheng Zhu, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18234)  

**Abstract**: Whole-body motion tracking policies turn a humanoid into a robust control interface: the teleoperator---or an upstream model---only supplies a coarse movement intent, while the low-level policy keeps the robot balanced and physically feasible. Existing trackers deliver this interface only on flat ground: trained in empty scenes, they never learn how contact with terrain and objects reshapes their dynamics, and they attempt to teach the policy to balance under any command by continually enlarging the reference-motion corpus, which stops working once feasible behaviors become environment-dependent. We present GigaBrain-WBC-0.5, the first Behavior World Model (BWM) for humanoid whole-body control. Rather than a purely reactive tracker, we train a causal Transformer to jointly predict its next action, next state, and the distribution over its next latent behavior command, so the network that acts also models how the environment shapes what it can do next. An automatic terrain-annotation pipeline recovers full 3D contact geometry from retargeted motion, enabling terrain annotation at the scale of existing motion datasets. The predicted distribution is reused at deployment to detect implausible commands online and retract them onto learned behaviors, so the robot attempts tasks in a "best-effort" manner. The result is a unified policy that takes real-time command, interacts with environment, and stays robust to implausible commands, falls, and disturbances. GigaBrain-WBC-0.5 achieves the highest success rate across all four regimes among three large-scale tracker baselines: 81.3% on terrain interaction (4.3x the strongest baseline), 83.1% under implausible commands, and 99.3% fall recovery (16.8x the strongest baseline). Hardware trials show robust interaction under missing supports and disturbances; the Unitree G1 checkpoint transfers to the Maker L01 robot with simple fine-tuning. 

---
# Bound-Aware Per-Organ Recall Risk Control for Multi-Organ CT Segmentation under Clinical Domain Shift 

**Authors**: Souraj Adhikary, Negar Chabi, Andre Mastmeyer  

**Link**: [PDF](https://arxiv.org/pdf/2608.18193)  

**Abstract**: Distribution-free risk control adds organ-specific recall guarantees to frozen segmentation. We calibrate per-organ thresholds for an AMOS-trained nnU-Net, audit transfer to RAOS, and estimate local re-certification cost using case-level voxel false-negative rate (FNR). The AMOS control passes, but $7/12$ organs exceed $\alpha{=}0.10$ after transfer; smaller calibration sets can mask exceedances with conservative or vacuous thresholds. Risk-Controlling Prediction Sets (RCPS) give high-probability control of population-mean risk, whereas Conformal Risk Control (CRC) gives weaker expectation control. Both require exchangeability; fixed and global thresholds give no per-organ guarantee. The Waudby--Smith--Ramdas (WSR) betting bound re-certifies six Tier-1 organs with 25 local cases, versus 30--40 for Hoeffding--Bentkus (HB). CRC needs 10--15 but has a heavier individual-case tail. No Tier-2 organ meets our illustrative precision criterion with 25 cases. 

---
# A systematic review of machine learning techniques to address diagnosis and treatment of autism: challenges and opportunities 

**Authors**: Rafael Muñoz-Terol, Jesús Peral, Sandra Amador, David Gil  

**Link**: [PDF](https://arxiv.org/pdf/2608.18188)  

**Abstract**: Autism spectrum disorder (ASD) is a developmental disability characterized by challenges in social interaction and communication. As the causes of ASD remain unclear, identifying relevant features and hidden correlations is crucial for early diagnosis. This systematic review evaluates 55 studies from 2017 to 2023 on the application of machine learning (ML) techniques to ASD. The primary objective is to examine recent ML applications in ASD research, identifying trends, techniques, and datasets that enhance diagnosis and treatment. Supervised learning methods dominate, as they align well with ASD diagnostic needs; however, the role of deep learning is expanding with greater data availability. Emerging techniques based on hybrid methods, where unsupervised, deep learning, and fuzzy logic could be included, will be interesting to observe in the future. The review highlights key challenges and opportunities, particularly the need for models that can integrate complex data -such as genetic and clinical information- to improve diagnostic accuracy and treatment outcomes. Additionally, incorporating innovative data sources, like wearable devices and biometric sensors, could enable continuous and non-intrusive monitoring, providing a more holistic understanding of ASD. Findings emphasize that addressing current challenges requires interdisciplinary collaboration and expanded datasets tailored to ASD. Future ML models will benefit from broader multimodal data integration, enabling researchers to more comprehensively address the complexities of ASD. 

---
# What Can Artificial Intelligence Learn from Medicine? Generative Analogies and Reliable Machine Learning Systems 

**Authors**: Emanuele Ratti, Lena Zuchowski  

**Link**: [PDF](https://arxiv.org/pdf/2608.18186)  

**Abstract**: In the past few years, machine learning (ML) has been widely (and to an extent, successfully) implemented in medicine. However, uncertainties surrounding ML have made it difficult to establish the bases of its epistemic and methodological warrants. In the literature, a parallel has been drawn between medicine and ML, suggesting that we should model epistemic and methodological standards for ML on the standards of clinical translation. By developing tools from Hesse work, we characterise the nature of this parallel as a generative analogy between the process of clinical translation and the process of building ML systems. We identify more precisely the epistemic and methodological warrants of clinical translation that are typically only mentioned when appealing to the analogy, and we show in which sense such warrants apply analogically to the context of ML. In particular, we interpret warrants of clinical translation in reliabilist terms, and we show how this can inform a new form of ML reliabilism, which is distinct from (though compatible with) existing reliabilist accounts in philosophy of AI. 

---
# Are LLMs Safe Beyond Text: Do Emojis Expose Gaps in Safety Evaluation 

**Authors**: M P V S Gopinadh  

**Link**: [PDF](https://arxiv.org/pdf/2608.18164)  

**Abstract**: Safety evaluations of large language models (LLMs) predominantly rely on text-based adversarial prompts, potentially overlooking vulnerabilities arising from alternative input representations. This work examines emoji-augmented prompts as a test case for this gap, evaluating 50 prompts across four open-source LLMs (Mistral 7B, Qwen 2 7B, Gemma 2 9B, Llama 3 8B). Results show substantial variation in robustness: Gemma 2 9B and Mistral 7B exhibit non-zero success rates (10%), Llama 3 8B 6%, while Qwen 2 7B shows complete resistance (0% success rate). A chi-square test ($\chi^2 = 32.94, p < 0.001$) confirms significant differences in outcome distributions. These findings indicate that robustness is sensitive to input representation, and that evaluations restricted to standard text prompts may underrepresent model vulnerabilities. 

---
# When Do LLMs Actually Help? Evaluating LLMs as Data Quality Annotators 

**Authors**: Praphulla Lal Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2608.18158)  

**Abstract**: LLMs have been increasingly used to catch data quality issues automatically, but we know very little about how consistent these judgments actually are. This study tests an LLM on two e-commerce data quality tasks, entity matching and brand mislabeling, against rule based baselines and human verified ground truth, under both zero-shot and few-shot prompting. On entity matching while using the Abt Buy benchmark (2,194 labeled pairs), a simple rule based baseline (F1=0.950) performed about as well as LLM zero shot prompting (F1=0.948). Moreover, a few-shot prompt revision that looked effective on a small validation sample reduced full-scale performance to F1=0.914. This showed that small sample prompt evaluation can be misleading. On brand mislabeling detection, using 500 Amazon product listings with synthetically injected labeling errors, the LLM clearly outperformed a naive rule based baseline (F1=0.833 vs 0.721), because it could draw on background knowledge of brand product relationships that a simple rule could not access. Testing consistency across repeated runs (200 pairs, 5 runs at temperature 0.7) showed the model agreeing with itself 99.7% of the time on average, with 99% of pairs giving identical answers across all 5 runs. Using majority voting across these runs only improved F1 by 0.005, at 5 times the inference cost. These results suggest that the value of using an LLM over traditional methods depends heavily on the task. LLMs offer little advantage when strong lexical signals already exist, but a clear advantage when the task requires background knowledge, all while remaining highly consistent across repeated queries. 

---
# How Quantum Is the Advantage? A Fair, Calibration- and Noise-Aware Benchmark and Attribution Audit of Quantum Machine Learning for Network Intrusion Detection 

**Authors**: Syeda Anshrah Gillani, Mirza Samad Ahmed Baig, Shahid Munir Shah, Asher Ali, Hamzah Siddiqui  

**Link**: [PDF](https://arxiv.org/pdf/2608.18155)  

**Abstract**: Quantum machine learning (QML) for network intrusion detection (NIDS) is routinely reported to reach near-perfect accuracy, yet the most rigorous studies find that well-tuned classical models remain competitive, and that apparent quantum gains may be artefacts of classical dimensionality reduction and implicit regularisation rather than genuine quantum effects. We ask not whether a quantum model can post a high accuracy, but how quantum the advantage really is. We present a unified, reproducible QML-IDS benchmark evaluating hybrid variational quantum circuits and quantum-kernel SVMs against five honestly-tuned classical baselines across four standard NIDS datasets (NSL-KDD, UNSW-NB15, CICIDS2017, NF-ToN-IoT-v2) under one leakage-controlled protocol, with an equal-budget feature view, imbalance- and calibration-aware metrics with significance testing, and a simulated NISQ noise sweep. We introduce a quantum-attribution audit (parameter-matched classical controls, a random-feature kernel, and a regularisation sweep) that quantifies how much of any gain is genuinely attributable to the quantum component. Tuned classical models (Random Forest, XGBoost) match or exceed the quantum models on aggregate detection on every dataset, and the audit attributes this to classical preprocessing and regularisation rather than quantum effects. Two advantages survive false-discovery-rate correction: the quantum-kernel SVM out-ranks its direct classical surrogate (a random-feature kernel) on AUPRC and ROC-AUC, and a small four-qubit hybrid out-detects the best classical baseline at the 1% false-positive operating point on the distribution-shifted NSL-KDD task (p = 0.005, BH q = 0.030). Code, seeds, and splits are released; our contribution stands whether quantum wins, ties, or loses. 

---
# TokenPowerSandbox: Evidence-Gated CPU-First Screening for Energy-Aware LLM Serving 

**Authors**: Chenxu Niu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18149)  

**Abstract**: Energy-aware LLM serving requires comparing configurations under realistic request shapes, yet exhaustive target-GPU profiling is costly and a cheap predictor can be dangerously confident outside its measured scope. We present TokenPowerSandbox, an evidence-gated workflow that combines an interpretable CPU-resident projector, short target-GPU probes, full-workload verification, and tamper-evident freeze-before-measurement provenance. On one NVIDIA H100 80GB serving Qwen2.5-7B-Instruct with vLLM, three anchor repeats and six development workloads calibrate workload transfer. The same frozen model is evaluated on a blind holdout and a separately predeclared no-refit confirmation totaling 51 post-freeze runs. Energy MAPE is 6.23% and 7.35%, with Spearman rank correlations of 0.976 and 0.933. However, a predeclared TTFT gate passes at concurrency four (9.27% MAPE) and triggers abstention below four (64.80%), showing why energy accuracy cannot certify latency. 

---
# Entropy-Constrained Adaptive Stochastic Quantization 

**Authors**: Ran Ben Basat, Yaniv Ben-Itzhak, Michael Mitzenmacher, Shay Vargaftik  

**Link**: [PDF](https://arxiv.org/pdf/2608.18147)  

**Abstract**: Adaptive stochastic quantization (ASQ) is a recently introduced quantization approach that optimizes the Mean Squared Error (MSE) for a given input while preserving unbiasedness. It is designed to alleviate the communication and memory bottlenecks of modern data and machine learning workloads, including model, gradient, and KV-cache compression and nearest-neighbor search. Further, practical systems can then compress quantized data with a lossless entropy encoder. However, existing unbiased methods, including ASQ, choose their quantization values without considering this later encoding stage, leaving accuracy on the table.
We formulate the Entropy Constrained Adaptive Stochastic Quantization (ECASQ) problem, which jointly selects adaptive quantization values to minimize MSE under an entropy budget and an unbiasedness constraint. We give an optimal dynamic program with $O(sd^2)$ time and $O(d^2)$ space for a length-d vector and at most s quantization values, and a GPU-friendly approximate dynamic program with $O(sd^2)$ time and $O(d)$ space. The approximation guarantees that the solution has an MSE no larger than the optimal solution that uses one fewer bit of entropy per entry. We also provide an iterative refinement procedure for the approximation solution that, in our experiments, yields near-optimal results while retaining a substantial speed advantage over our solver for the optimal solution. 

---
# The Deontic Gap: Large Language Models and the Modal Language of Obligation 

**Authors**: Daniel Hart, Sarah Allred, Joseph Abbas, Morenike Alugo  

**Link**: [PDF](https://arxiv.org/pdf/2608.18144)  

**Abstract**: Modal auxiliaries such as must, should, and have to mark necessity and obligation within the contexts of speaker authority and interpersonal stance. We examine whether large language models (LLMs) reproduce contemporary human patterns of deontic modal usage. Across three primary corpora, an external benchmark, two controlled replications, and a naturalistic eleven-model replication, AI-generated text consistently underuses positive deontic modals (must, should, have to, had to) relative to contemporary humans. Historical comparison with the Google Books Ngram corpus (1920-2022), used as a heuristic calibration against the published-prose record, shows that AI modal frequencies fall within the range of formal published English, whereas contemporary human modal rates in informal digital contexts often exceed twentieth-century book baselines. Phrase-level decomposition shows that the AI-human modal gap is concentrated in constructions central to interpersonal stance (should, have to, had to), while AI matches or exceeds humans on need to in instructional and question-answering contexts but not in persuasive student writing, indicating that the modal profile is genre-conditional. The findings suggest that LLM modal usage reflects the formal written resources on which these models were trained, while underusing the modal constructions through which contemporary human writers mark immediate, interpersonal obligation. 

---
# Language Models for Portuguese: A Systematic Mapping Study 

**Authors**: Jhessica Silva, Carlos Caetano, Helena Maia, Breno Bernard Nicolau de França, Sandra Avila, Helio Pedrini  

**Link**: [PDF](https://arxiv.org/pdf/2608.18138)  

**Abstract**: In recent years, the rapid development of language models has transformed the field of Natural Language Processing through a wide range of applications. However, the development of language models has not progressed uniformly across all languages. In the case of the Portuguese language, there has recently been a growing effort by academia and companies to develop language models and create data resources for Portuguese. These efforts have resulted in the rise of an increasingly diverse ecosystem of language models for Portuguese. However, information on these models remains dispersed in scientific publications, technical reports, model repositories, and project documentation. This survey presents a systematic mapping study of language models developed for Portuguese, providing a comprehensive overview of the current state of the field. We map a total of 46 models, characterizing them by various aspects, including base model, architecture, computational resources, training datasets, licensing, code availability, data, and model weights. Furthermore, we analyzed the evolution and relationships among these models through a phylogenetic perspective, identified current research gaps and opportunities, and discussed future directions for the development of language models for Portuguese. 

---
# Global Index on Responsible AI 2026 : Conceptual Framework and Methodology 

**Authors**: Fola Adeleke, Rachel Adams, Ayantola Alayande, Daniela Benavente, Ana Florido, Nicolás Grossman, Leah Junck  

**Link**: [PDF](https://arxiv.org/pdf/2608.18122)  

**Abstract**: This report presents the methodology of the Global Index on Responsible AI (GIRAI), 2nd Edition. This edition refines the 1st Edition by strengthening the distinction between framework existence and implementation, restructuring dimensions from three to five thematic areas, introducing more granular variables for framework quality, and applying a multi-stage review and validation process. An independent statistical pre-audit was conducted to assess the coherence and robustness of the framework. GIRAI assesses responsible AI governance across five dimensions: Inclusion and Diversity, Ethics and Sustainability, Labour and Skills, Trust and Safety, and Use of AI in Public Service. Each dimension has a number of indicators (38 in total), organised into three pillars, namely AI Policy (17 indicators on government frameworks and implementation, assessed through primary data), CSO Engagement (5 indicators, primary data), and Enabling Conditions (15 indicators on the structural factors shaping responsible AI governance, assessed through secondary data), and a government Use of Unacceptable Risk AI (URAI) indicator (primary data), applied separately as an accountability penalty to the final score. Data was collected by 135 country-level researchers through a structured global survey, complemented by secondary datasets. The count, scope, enforceability, thematic coverage, and implementation levels of the data points are coded into numerical variables, normalised to a scale of 100, aggregated through pillar weights of 60% (AI policy), 10% (CSO Engagement), and 30% (Enabling conditions). A deduction penalty is applied for countries with evidence of URAI. This documentation enables systematic cross-national comparison, supporting policymakers, civil society, and AI developers to identify where commitments are translating into enforceable protections and where critical gaps remain. 

---
# Temporal Multi-Signal Fusion for Token-Level Hallucination Detection 

**Authors**: Igor Itkin  

**Link**: [PDF](https://arxiv.org/pdf/2608.18115)  

**Abstract**: Token-level hallucination detectors score each token independently from a single signal, and fail exactly when the generating model is confidently wrong. This paper instead treats hallucination as a temporally extended span and detects it by sequence labeling: each token is scored from a 33-dimensional feature stream that fuses text statistics, Natural Language Inference (NLI) entailment, and language model surprisal, with no access to model internals. A Bidirectional Gated Recurrent Unit (BiGRU) over these features reaches an AUC of 0.840 on RAGTruth (10 seeds), an 11-point gain over an independent logistic-regression baseline (p = 0.002, Wilcoxon signed-rank). A controlled decomposition attributes most of the gain to temporal order rather than model capacity: evidence propagates from confident positions to ambiguous neighbors within a span. The same 0.845 ceiling recurs across recurrent, state-space (Mamba), and attention architectures, locating the bottleneck in the feature set rather than the model. Because it reads only the generated text and external signals, the detector works on closed-source models, and it keeps working on text produced by language models it never saw during training, losing under 4% AUC. 

---
# Accurate Decoding of Natural Sentences from Non-Invasive Brain Recordings 

**Authors**: Mingfang Zhang, Jarod Lévy, Cedric Rommel, Jérémy Rapin, Corentin Bel, Julie Bonnaire, Daniel Nieto, Pierre Bourdillon, Svetlana Pinet, Stéphane d'Ascoli, Thomas Moreau, Jean-Rémi King  

**Link**: [PDF](https://arxiv.org/pdf/2608.18114)  

**Abstract**: Restoring communication for people who have lost the ability to speak or move after a brain injury is a major challenge. While intracranial implants now enable high-performing brain-computer-interfaces, non-invasive alternatives are still lagging behind. Here, we present Brain2Qwerty v2, a model that can decode the production of natural sentences solely from real-time magnetoencephalography (MEG) recordings. By collecting 22,000 sentences typed by nine subjects, each recorded for 10 hours, our model leverages character, word and sentence-level representations to achieve an average word error rate (WER) of 39%. For our best participant, the model accurately decodes half of the sentences with one word error or less. Critically, decoding accuracy log-linearly improves with data volume, suggesting that the performance gap with intracranial approaches could be partially bridged through data scaling. We show that AI enables this performance in three main ways: the substitution of hand-crafted pipelines for event detection with deep learning, the finetuning of large language models to extract semantic representations, and the deployment of AI agents to iteratively refine our decoding pipeline via automated code development. Together, these results show that non-invasive brain-to-text decoding starts to operate at a level of accuracy previously thought exclusive to surgical implants, opening a path toward safe and efficient brain-computer-interfaces. 

---
# Same Facts, Different Updates: Inference Setup Shapes LLM Behavior in Medical Allocation 

**Authors**: Spencer Gibson, Tyler Crosse, Magnus Saebo, Achyutha Menon, Eyon Jang, Diogo Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2608.18108)  

**Abstract**: Large language models are being incorporated into sensitive and important decision-making processes across nearly all fields. While prior work studies model bias around inputs and scenario framing, models can also behave in unexpected and undesirable ways due to context accumulated over their deployment. In this work, we study a medical example in which a model is asked to assign resource-allocation probabilities to two people given brief clinical context, and then sees the same scenario with a single extra sentence containing contrasting patient information, either with or without its previous response in context. Across three of four tested models, the paired-context and independent-inference experiments have different probability shifts, often in opposite directions (in favor of Person B vs. in favor of Person A) when new information is provided. We include additional paired-context experiments to show the effect of varying attributes across scenario axes. Our findings show the context-dependent effect of patient information in a sensitive medical use case. More broadly, our work shows the importance of carefully incorporating LLM-based systems into decision-making processes, context engineering, and further model behavioral studies. 

---
# Institutional Prestige as Geographic Bias in Large Language Models: Evidence from Three Factorial Experiments with Bootstrap Confidence Intervals 

**Authors**: Maikel Leyva-Vazquez, Florentin Smarandache  

**Link**: [PDF](https://arxiv.org/pdf/2608.18107)  

**Abstract**: We investigate whether large language models (LLMs) systematically discriminate in candidate evaluations based on applicant name ethnicity and/or institutional prestige and geographic location. Three factorial experiments are reported (4,320 API calls, four LLMs, five professional domains). Study 1 (3x4 design) finds a statistically robust institution-tier gradient of +0.297 points on a 10-point scale (95% bootstrap CI: +0.175 to +0.422), while name-origin effects are negligible and non-significant (95% CI crosses zero). Study 2 (2x2 Prestige x Country design) breaks the prestige-geography confound: the prestige effect (+0.185; 95% CI: +0.093 to +0.275) exceeds the country-of-origin effect (+0.126; 95% CI: +0.037 to +0.218) by 1.5x. Study 3 (2x2 Journal x Institution design) reveals that journal prestige (Nature vs. a peripheral open-access journal) dominates institutional prestige by 5.7x: journal effect +1.937 (95% CI: +1.811 to +2.062) vs. institution effect +0.341 (95% CI: +0.184 to +0.504). A "rescue effect" is confirmed: publishing in Nature compensates for low institutional prestige more strongly for candidates from the University of Guayaquil (+2.127) than from MIT (+1.745). Results are quantified using the Neutrosophic Bias Index NBI<T,I,F>; the I component reveals elevated evaluation inconsistency for low-prestige profiles, an epistemic disadvantage not captured by mean-only metrics. Code and data: this https URL 

---
# Different Facets of Verbalised Overconfidence: an Interpretability Study 

**Authors**: Davide Mazzaccara, Leonardo Bertolazzi, Raffaella Bernardi  

**Link**: [PDF](https://arxiv.org/pdf/2608.18106)  

**Abstract**: Large language models tend to overconfidence, giving assertive answers when the evidence suggests hedging or abstention. Using controlled reasoning scenarios that manipulate logical necessity and possibility, we study this behavior in Qwen3-4B, across three ways to express uncertainty: verbal epistemic markers, abstention, and numeric confidence scores. Our results confirm this tendency toward overconfidence, particularly when the model is prompted to output a numeric confidence score. At the interpretability level, we propose a method that differentially identifies transcoder features responsible for uncertainty and certainty. Our analysis reveals Qwen3-4B's default mechanism favors certainty generation through a broad coalition of shared features, while uncertainty is implemented as a sparse override mediated by a small set of dedicated features. Intervening on these uncertainty features both causally proves this imbalance underlying overconfidence and also mitigate overconfident errors. The same set of features generalise across the three uncertainty-expression settings, languages, and an out-of-distribution modality task. 

---
# StocksTalk: A Voice-Enabled Conversational Agent for Structured Query Generation over Web Data 

**Authors**: Akshat Parmar, Vikranth Udandarao, Abhay Shakya, Tanmay Hire, Avinash Anand, Rajiv Ratn Shah, Daniel Wang Zhengkui  

**Link**: [PDF](https://arxiv.org/pdf/2608.18105)  

**Abstract**: StocksTalk is a voice-enabled conversational system for transforming spoken financial screening requests into executable and validated structured queries over real-world market data. The system combines streaming speech recognition, retrieval-augmented constraint extraction, schema-grounded LLM-based SQL generation, rule-based validation, and human-in-the-loop verification within an interactive dashboard. Unlike traditional template-driven financial assistants, StocksTalk exposes intermediate reasoning artifacts, including extracted constraints, normalized financial metrics, operator grounding, and generated queries, allowing users to inspect and refine each stage before execution. To evaluate the system, we curate a benchmark of 150 spoken financial prompts spanning multiple investment strategies and input noise conditions. Experimental results show that retrieval grounding, constrained query generation, and interactive verification substantially improve constraint extraction accuracy, SQL executability, logical consistency, and multi-turn stability compared to baseline LLM-based approaches. StocksTalk demonstrates how transparent, voice-driven interfaces can bridge natural language interaction and structured financial analysis, providing an effective framework for conversational stock screening and decision support. 

---
# DeepTCM1.0: A Multi-Expert AI Agent for Deciphering Mechanisms of Chinese Herbal Formulae Based on General Large Language Models 

**Authors**: Wenxin Duan, Hanwei Wang, Zhongying Peng, Zhonghua Lu, Jiayi An, Fan Song, Yong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18103)  

**Abstract**: Background: Mechanistic elucidation of traditional Chinese medicine (TCM) compound formulas remains a central challenge in the modernization of TCM. Conventional approaches, including data mining and network pharmacology, are insufficient for achieving deep integration between classical TCM theory and modern scientific research. In addition, direct question-answering using general-purpose artificial intelligence large language models is limited by inadequate adaptation to TCM theoretical frameworks and susceptibility to reasoning hallucinations. Consequently, there is an urgent need to develop intelligent analytical methods aligned with the holistic principles of TCM. Objective: To establish a multi-expert intelligent agent framework integrating classical TCM theory with modern life sciences, thereby enabling systematic and interpretable mechanistic analysis of TCM compound formulas, with Guizhi Decoction serving as a representative validation case. Methods: The DeepTCM1.0 framework was constructed based on the general-purpose large language model DeepSeek V3.2. It adopts a three-tier collaborative architecture and a three-round iterative quality-control workflow, simulating the collaborative analytical process of 11 interdisciplinary intelligent agents. The framework was applied to the mechanistic interpretation of Guizhi Decoction from the dual perspectives of classical traditional Chinese medicine theory and modern scientific research. Framework performance was comprehensively evaluated through double-blind five-dimensional scoring, intraclass correlation coefficient (ICC) reliability testing, Mann-Whitney U tests, and effect size analysis. The evaluation employed four independent large language models as evaluators, each conducting five rounds of repeated scoring on five anonymized reports, resulting in a total of 100 independent scoring assessments. 

---
# Computational Orientalism: Measuring Structural Discourse Bias in Large Language Models Using the Middle East Cultural Sensitivity Score (MECSS) 

**Authors**: Maha Shahid  

**Link**: [PDF](https://arxiv.org/pdf/2608.18100)  

**Abstract**: AI systems now shape how hundreds of millions of people learn about cultures other than their own. When someone asks one of these systems about the Middle East, they do not receive neutral facts. They receive a representation shaped by the frameworks embedded in training data, and that data is overwhelmingly Western and English-language. This paper asks whether that representation is Orientalist in Said's sense: whether it denies agency to Middle Eastern actors, treats Western frameworks as neutral while marking non-Western knowledge as particular, and explains the region through categories it did not produce. Standard fairness metrics cannot answer this, because they detect explicit prejudice rather than structural framing. This paper introduces the Middle East Cultural Sensitivity Score (MECSS), a framework that turns Said's seven Orientalist operations into measurable dimensions, and the term "Said-washing" for a specific failure: a model that disclaims generalization, then reproduces the structure it disclaimed. Across 280 conversations (1,120 exchanges), GPT-4 and Falcon3-7B-Instruct both reproduce Orientalist patterns systematically, through structural positioning rather than open stereotyping. GPT-4 scores moderately (mean MECSS 1.73); Falcon3-7B-Instruct scores higher (2.18), even though it was built in Abu Dhabi and trained with Arabic content. This is evidence against the assumption that building a model regionally makes it less Orientalist, though the models differ in size as well as origin, so geography cannot be isolated as the cause. Epistemic Center, the treatment of Western frameworks as unmarked universals, scores near the top of the scale for both models. Said-washing appears in 87.9% of GPT-4 conversations, a pattern existing metrics cannot see. Reducing this bias requires changing what models learn from, not only adding languages or relocating institutions. 

---
# Fractional Decay KV-Cache: Ownership-Aware Memory Management for Improved Inference Relevancy in Dialog Systems 

**Authors**: Sukanta Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2608.18098)  

**Abstract**: Key-value (KV) caching is essential for efficient autoregressive inference in transformer based dialog systems, yet existing strategies treat all cached entries uniformly or apply coarse eviction heuristics that fail to adapt as dialog topics evolve. We propose Fractional Decay KV-Cache (FD-KVC), a novel algorithm that maintains a dual-channel scoring mechanism for each cached KV pair: a cumulative attention channel that tracks aggregate importance (akin to H2O), and a recency-weighted relevance channel governed by temporal decay and reinforcement-inspired updates. The combination enables FD-KVC to both preserve historically important tokens and rapidly adapt when dialog topics shift. An adaptive learning rate driven by an ownership loss function ensures convergence without oscillation. FD-KVC operates entirely on CPU with negligible overhead. Across five diverse multi-turn dialog scenarios with 600 dialogs each, FD-KVC outperforms H2O, the state-of-the-art heavy-hitter baseline, by +6.7% on composite late-turn alignment, with improvements of +127% on topic-shift, +87% on gradual evolution, and +30% on mixed-topic dialogs. FD-KVC adapts to new topics 3.6X faster than H2O and achieves the highest topic diversity (80.6%) across all methods. Ablation studies confirm the contribution of each component. 

---
# Backdoor Learning in Language Models and Vision-Language Models 

**Authors**: Weimin Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18095)  

**Abstract**: Recent advances in deep learning have significantly enhanced the capabilities of Natural Language Processing (NLP) and Vision-Language Models (VLMs). However, these advancements come with increased vulnerabilities, notably through backdoor attacks that pose severe security threats. This thesis addresses two critical dimensions of Trustworthy AI and Efficient Multimodal Representation Learning: (1) security through analyzing, detecting, and designing backdoor attacks in NLP and VLMs, and (2) efficiency through advanced multimodal representation methods tailored for clinical and medical imaging applications. 

---
# NE-BERT: A Multilingual Language Model for Nine Northeast Indian Languages 

**Authors**: Badal Nyalang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18094)  

**Abstract**: Large pretrained language models have demonstrated remarkable capabilities across diverse languages, yet critically underrepresented low-resource languages remain marginalized. We present NE-BERT, a domain-specific multilingual encoder model trained on approximately 8.3 million sentences spanning 9 Northeast Indian languages and 2 anchor languages (Hindi, English), a linguistically diverse region with minimal representation in existing multilingual models. By employing weighted data sampling and a custom SentencePiece Unigram tokenizer, NE-BERT outperforms IndicBERT-V2 and MuRIL across all 9 Northeast Indian languages, achieving 15.97X and 7.64X lower average perplexity respectively, with 1.50X better tokenization fertility than mBERT. We address critical vocabulary fragmentation issues in extremely low-resource languages such as Pnar (1,002 sentences) and Kokborok (2,463 sentences) through aggressive upsampling strategies. Downstream evaluation on part-of-speech tagging validates practical utility on three Northeast Indian languages. We release NE-BERT, test sets, and training corpus under CC-BY-4.0 to support NLP research and digital inclusion for Northeast Indian communities. 

---
# Abliteration Mitigation via Refusal Aliases 

**Authors**: Nathan Truong  

**Link**: [PDF](https://arxiv.org/pdf/2608.18093)  

**Abstract**: Abliteration, the removal of refusal capabilities from large language models by projecting weight matrices orthogonal to an extracted refusal direction, has emerged as a prominent safety concern through its ability to bypass post-training alignment using only a small set of contrastive prompts. We find that existing defenses commonly overlook the cause of abliteration; that is, how easily the refusal direction can be extracted. To hinder this process, we introduce a weight-editing method that obscures the refusal signal by applying rank-$k$ updates to residual stream writer matrices while replacing refusal-inducing activations with random aliases and correcting downstream reader matrices to preserve the model's original behavior. On Llama-3-8B, AMRA improves post-abliteration refusal scores by $2.16$ points over the undefended baseline with less than $0.5$ percentage points of MMLU degradation. On Gemma-2-9B, it improves the post-abliteration refusal by $14.70$ points over the baseline while keeping harmful output rates similar to the baseline, albeit at a greater utility cost. 

---
# Self- and Other-Labels Induce Bidirectional Bias in LLM Judges 

**Authors**: Songeun Chae, Min Kim, Donghoon Jung, Seojin Choi, Seohyon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2608.18091)  

**Abstract**: As LLM-as-a-judge systems become increasingly widespread, self-preference in LLMs -- the tendency to favor one's own outputs -- raises growing concerns about evaluation reliability. However, it has been studied predominantly on generated text, where stylistic features and response quality are inevitably conflated. As a result, existing measurements cannot separate genuine self-preference from these confounds. We address this by changing the object of evaluation: instead of judging generated text, ten LLMs assess narrative constraint selections, which carry no model-specific stylistic fingerprint yet retain a recoverable model-specific signature. We run two experiments that yield distinct findings. Under blind evaluation, self-preference largely disappears once selection quality and evaluator severity are controlled. It vanishes on three of four rubric dimensions and reverses on the fourth, where judges rate their own selections as less original. Under matched quality, however, self- and other-labels alone -- without naming any model -- shift scores bidirectionally: LLM judges inflate scores for self-labeled selections and deflate those for other-labeled ones regardless of the selection's actual source. We make two contributions: 1) authorship attribution is a distinct driver of evaluation bias, and 2) open-ended, ground-truth-free tasks can serve as controlled instruments for studying LLM judge behavior. 

---
# Nine Emotion Centroids: A Label-Free Valence Axis That Transfers Across Four Modalities 

**Authors**: Yousef Radwan  

**Link**: [PDF](https://arxiv.org/pdf/2608.18090)  

**Abstract**: Inside a modern language model sits a single internal direction that tracks how positive or negative a sentence feels. We show how to find this valence axis (V-axis) from just 9 emotion category names plus 50 short narrative paragraphs per emotion -- about 1,500 fewer labels than the usual supervised approach -- and that the same direction appears in vision, audio, and human-brain encoders never jointly trained. The recipe: embed nine emotion-anchored story sets in a frozen encoder, take the top principal direction of the nine averaged embeddings. Projecting new inputs onto it captures 93% of supervised performance on SST-2 (Llama-3-8B-Instruct, AUC 0.772 vs. 0.828), correlates with human valence ratings on 11,811 EmoSet images at r=0.636, reaches AUC 0.906 on ESC-50 audio (p<2.2e-15), and AUC 0.720+/-0.055 on EEG from 123 subjects (p<3.65e-8). The direction is mechanistically active: ablating it collapses sentiment accuracy by 5.5-37.2 pp across three LLMs vs. at most 0.88 pp for matched random directions (z>12). A 2-parameter classifier trained on text labels transfers to images (AUC 0.961), audio (0.764), and brain recordings (0.828) without target-modality labels; a generic 16-D subspace stays at chance (0.525). The recipe is bounded to continuous attributes -- seven tests on categorical concepts return near-chance -- and steering is family-specific (Llama/Mistral yes, Qwen/Gemma no). 

---
# Latent Space Refusal Anchoring for Low-Resource African Languages: Mechanistic Safety Recovery Without Retraining 

**Authors**: Godwin Abuh Faruna  

**Link**: [PDF](https://arxiv.org/pdf/2608.18089)  

**Abstract**: Instruction-tuned models often refuse harmful requests in English but comply with the same requests in Yoruba, Igbo, Igala, and Hausa. This suggests that the refusal mechanism is present in the residual stream but fails to activate for low-resource inputs. Recovering it normally requires labelled target-language data and retraining, neither of which is available at scale for most African languages. We introduce Latent Space Refusal Anchoring (LSR-Anchoring), a training-free method that extracts the refusal direction from English prompts and clamps it onto the residual stream at inference time. The primary variant, Mean-Activation Steering (MAS), operates across the four architectures we tested: Llama-3-8B, Llama-3.1-70B, Mistral-7B-Instruct, and Qwen2.5-7B. On Mistral and Qwen it recovers safety with benign degradation below 0.08. On Llama-3-8B it overcorrects, with Degraded Performance on Legitimate prompts (DPL) reaching 1.00. We address this with SAE-Derived Steering (SDS), which replaces the dense mean-difference direction with a single Sparse Autoencoder (SAE) feature and reduces Kullback-Leibler (KL) divergence by 3.5-7x without benign collapse. Four languages transfer positively, but Arabic fails on every architecture and at every steering magnitude, indicating a geometric mismatch rather than a baseline effect. Massive Multitask Language Understanding (MMLU) accuracy drops remain below 0.35 percentage points at every effective steering magnitude. 

---
# SuTRA : Structurally-Unified Tokenization with Root Awareness 

**Authors**: Vaibhav Rathore, Siddhant Gole, Dadhichi Telwadkar, Rooshil Bhatia, Maulik Ruparel, Siddharth Surekha, Neha Bhargava  

**Link**: [PDF](https://arxiv.org/pdf/2608.18087)  

**Abstract**: Existing subword tokenizers optimize statistical compression but ignore morphological structure, particularly the relationship between roots and affixes. This is harmful for morphologically rich Indic languages, where basic units are complex orthographic syllables (aksharas) rather than letters. Frequency-based methods over-fragment words, arbitrarily splitting roots and affixes - a phenomenon we term Morphological Shattering. We propose SuTRA (Structurally-Unified Tokenization with Root Awareness), a morphology-aware algorithm that preserves akshara indivisibility and penalizes merges crossing morphological boundaries. We also release a new morphological segmentation dataset for Hindi, Marathi, and Gujarati. SuTRA reduces shattering, achieving peak gains of +14.7% in morphological alignment (Boundary F1) and +34% in semantic recoverability (Hindi) over BPE. These structural gains yield an average improvement of +8.08 chrF2 in machine translation. 

---
