# SoftReason: A Fully Differentiable Neuro-Soft-Symbolic Deductive Reasoning Architecture over High-Dimensional Perceptual Data 

**Authors**: Wael AbdAlmageed  

**Link**: [PDF](https://arxiv.org/pdf/2607.20402)  

**Abstract**: In many reasoning problems, the premises are not observed as discrete symbols, but must be inferred from high-dimensional inputs. Further, the predicate vocabulary, argument structure, and trusted evidence are supplied by a Knowledge Graph (KG), or rule definitions. Classical neuro-symbolic pipelines have a discrete interface between perception and deduction. We present a neuro-soft-symbolic architecture for differentiable deductive reasoning over latent perceptual facts and knowledge-provided predicates. SoftReason removes the gradient gap by representing the deductive state as a local soft interpretation tensor over candidate constants and predicates. Perception proposes probabilistic base facts, KG triples enter as high-confidence soft evidence, and every query anchor, predicate choice, and closure update remains differentiable. Our core innovation is a learned differentiable lift of the immediate-consequence operator. It uses predicate-definition embeddings and latent composition channels to form soft body-predicate mixtures, aggregate over all possible witnesses, propose query-conditioned head facts, and update the interpretation through a monotone probabilistic OR. We instantiate the framework on Knowledge-aware Visual Question Answering (KVQA), and demonstrates how SoftReason supports end-to-end perceptual grounding, KG evidence injection, and differentiable deductive closure in one trainable architecture. 

---
# Train the Model, Not the Reader: Decodability Supervision for Verifiable Activation Explanations 

**Authors**: Hiskias Dingeto  

**Link**: [PDF](https://arxiv.org/pdf/2607.20379)  

**Abstract**: Natural-language autoencoders score explanations of hidden activations by reconstruction: an explanation is deemed faithful if the activation can be regenerated from it. The test is structurally insensitive to individual false claims: if flipping a claim does not change the reconstruction, the claim is never penalized. We show the test is passed in two ways, neither faithful. On a released Qwen-2.5-7B verbalizer, explanations reconstruct well above chance while ~2% of specific claims are reconstruction-dependent, so the score tracks gist, not specific facts. Under exact synthetic ground truth, the standard recipe develops co-adapted private codes (false wording the reconstruction depends on) in 5/5 runs, and fixes that leave the target model unchanged do not help. We contribute two audit protocols, the grounded-vs-true cross and the evaluator swap, and RECAP (Readable Encodings via Co-trained Auxiliary Predictors): linear heads trained alongside the target model to keep designated content decodable. On RECAP-trained sandbox models, fresh verbalizers state the designated content truly and the codes vanish, at a +0.001-nat cost. This replicates on a pretrained Pythia-160M: the content becomes reliably probe-decodable, though a fresh verbalizer conveys it only in part (truth 0.44-0.46 vs a near-zero control). For interpretability, high reconstruction does not certify individual claims. For AI safety, RECAP makes designated internal content independently checkable against probes rather than asserted by prose a model can game: an independent probe scores the verbalizer's true claims above its false ones (AUC 0.96, vs 0.82 without RECAP). Against an adversary that edits an explanation to maximize the reconstruction score while lying (suppressing ~87% of its lie penalty), the RECAP probe still flags the lies (AUC 0.95) while the control probe collapses to chance (0.51). 

---
# PoTRE: Test-Time Reasoning inspired by Cognitive Heterogeneity 

**Authors**: Anmol Kankariya, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2607.20268)  

**Abstract**: While Large Language Models (LLMs) excel at many tasks, they frequently struggle with complex reasoning that requires long-horizon planning and iterative error correction. Furthermore, standard single-stream prompting proves brittle when models encounter novel abstractions or rigorous domain constraints. We introduce PoTRE (Poly-Topological Reasoning Ensembles), a heterogeneous framework that decouples inference into four agents: (1) Adversarial Refinement Agent, (2) Hierarchical strategic Planning Agent, (3) Spectrum Search Agent, and (4) Direct Chain Agent. A final Task-Adaptive Aggregation Layer dynamically reconciles these perspectives -- via final candidate selection, semantic synthesis, or neuro-symbolic verification -- to produce a robust global solution. We evaluate PoTRE on three frontier benchmarks: ARC-AGI-2, Humanity's Last Exam (HLE), and PRBench Finance. PoTRE achieves state-of-the-art accuracy of 49.92% on HLE, surpassing the previous best official score. We demonstrate that this architectural heterogeneity achieves improved reasoning performance using similar or fewer inference tokens compared to heavily scaled homogeneous baselines. 

---
# CUSUM-Shaped Inference-Time Monitoring and Targeted Re-Decoding for Quantized Small Language Model Reasoning 

**Authors**: El Hassane Ettifouri, Ayoub Belfatmi, Mahaman Sanoussi Yahaya Alassan, Walid Dahhane  

**Link**: [PDF](https://arxiv.org/pdf/2607.20129)  

**Abstract**: Quantized small autoregressive reasoning models can enter long, repetitive, or unproductive trajectories, yet inference-time compute is usually allocated without observing how a trajectory develops. Building on an earlier token-level e-CUSUM controller, we develop MGT-B (Monitoring-Guided Test-time Backtracking), a revised external controller that maps overlapping windows of pre-sampling uncertainty and degeneration features to position-conditional empirical tail probabilities, accumulates mixture betting factors with a CUSUM-shaped reset, and responds to an alarm by estimating a rollback point, restoring token and key-value-cache state, and performing constrained re-decoding. To audit whether the effect persists on problem identities first observed after the manual choice of log threshold h = 10, we retrospectively exclude 260 IDs present in pre-threshold artifacts and retain the chronologically first post-threshold pair for each remaining ID, yielding a 240-pair chronology-audit set. On this set, accuracy changes from 82/240 to 88/240 (+2.50 percentage points; 13 corrections, 7 regressions; exact McNemar p = 0.2632; paired bootstrap 95% interval [-1.25, +6.25]). A broader 467-pair historical-coverage set of seed-matched pairs changes accuracy from 146/467 to 167/467 (+4.50 points; McNemar p = 0.000753), but includes 200 seed-1 IDs available before or during threshold selection and is reported only as an exploratory estimate. All 316 no-alarm outputs in the 467-pair set are identical to vanilla, while the 151 alarmed trajectories contain 29 corrections and 8 regressions. Neither analysis is confirmatory, and the empirical factors are not established as a valid e-process or e-detector. The results support a selective monitoring-and-repair mechanism for the studied MATH-500 setting, rather than a general or theoretically certified reasoning improvement. 

---
# TRUST-ESD: A Risk-Calibrated and Governance-Aware AI Framework for Enterprise Strategic Decision Support Under Uncertainty 

**Authors**: Tian Qiu, Li Yan, Mahabubur Rahman Miraj, Shanqin Yi, Md Intekhab Rahman Galib, Jahid Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2607.20065)  

**Abstract**: Enterprise strategic decision support requires AI systems that are not only accurate, but also uncertainty-aware, risk-calibrated, explainable, and governance-compliant. This paper proposes TRUST-ESD, a risk-calibrated and governance-aware framework for enterprise decision support under uncertainty. TRUST-ESD evaluates feasible counterfactual strategies through predictive utility estimation, conformal uncertainty calibration, CVaR-based downside-risk scoring, risk-memory retrieval, policy-as-code governance, explainability, and human oversight. Unlike prediction-only methods that select actions by maximum expected utility, TRUST-ESD recommends strategies that balance value, reliability, risk exposure, and compliance. Experimental results show that TRUST-ESD improves risk-adjusted utility by 7.95%, reduces risk exposure by 23.22%, reduces CVaR by 23.78%, lowers calibration error by 13.89%, improves explanation fidelity by 10.90%, and increases governance compliance by 9.76% compared with strong uncertainty-aware baselines, while maintaining competitive predictive accuracy. Ablation and case-study analyses further confirm that uncertainty calibration, downside-risk scoring, risk memory, explainability, and governance validation jointly improve trustworthy enterprise decision-making. 

---
# PRO-LONG: Programmatic Memory Enables Long-Horizon Reasoning 

**Authors**: Alexis Fox, Junlin Wang, Paul Rosu, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2607.20064)  

**Abstract**: Long-horizon tasks require sustained perception, reasoning, and exploration, and are a persistent challenge for large language model (LLM) agents. This gap is reflected in their limited performance on continual learning benchmarks such as ARC-AGI-3, especially when models are evaluated out of the box. Various agent harnesses have been proposed to close this gap, and each commits to a strategy for handling long sequences of observations, i.e., what information to save from the environment and how to load it into model context, a choice we argue is particularly consequential. Existing methods for context management face a significant tradeoff, as preserving more information makes retrieving relevant details less tractable. We propose PRO-LONG, a minimal context management framework built around programmatic memory for LLM agents in long-horizon, exploratory settings. PRO-LONG addresses the tradeoff by keeping a complete, structured interaction log and capitalizing on recent progress in coding agents to search this history efficiently. On the full ARC-AGI-3 public game set, PRO-LONG improves over a base coding agent by an average of 18.0 percentage points across frontier models, and matches or exceeds state-of-the-art specialized harnesses (up to 76.1% pass@1) while using 4.2-5.8x fewer tokens. With Fable 5, PRO-LONG achieves 97.4% best@2 at a total cost of \$1,750. Relevant code and logs are available at this https URL. 

---
# Reading and Steering Representations of Materials-Science Mechanisms in an Open-Weight Language Model 

**Authors**: Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2607.20058)  

**Abstract**: Large language models can answer scientific questions, yet a correct output does not reveal whether the model represents or uses the governing physics. Here we show that materials science mechanism information in the open-weight google/gemma-4-E4B-it model has three experimentally separable forms: concepts are readable in individual hidden states, constitutive orientation is carried by controlled transformations between states, and selected internal representations causally control engineering answers. We combine matched direct and Jacobian vocabulary readouts, option-free state geometry, a 60-law counterfactual benchmark and causal interventions. In 50 held-out materials descriptions, three independently fitted Jacobian lenses reproduced concept ranks, and target-free word sets from both readouts enabled blinded identification of 9 of 10 mechanism families. A separate 72-prompt benchmark produced mechanism-specific hidden-state neighborhoods, but an exact graph audit showed that this apparent physical organization was equally explained by numerical comparison. We therefore compared otherwise identical prompts in which only the direction of the physical input was reversed, asking whether the resulting hidden-state movement followed the supplied constitutive law. These state transformations ordered direct, physically neutral and inverse laws across 60 frozen relations and correctly oriented 39 of 40 directional laws, whereas lexical controls were near chance. Bidirectional interventions shifted answer probabilities toward or away from the physically appropriate outcome across all 12 matched cases, while counterfactual state patches transferred opposing decision signals across mechanisms and answer formats. Physical relationships were therefore more visible in controlled state changes than in absolute states alone. 

---
# Global Difference Constraint Propagation for Constraint Programming 

**Authors**: Lucas Kletzander, Jip J. Dekker, Andreas Schutt, Peter J. Stuckey  

**Link**: [PDF](https://arxiv.org/pdf/2607.20022)  

**Abstract**: Difference constraints of the form $x - y \leq d$ are well studied, with efficient algorithms for satisfaction and implication, because of their connection to shortest paths. Finite domain propagation algorithms, however, typically do not make use of these algorithms, and treat each difference constraint as a separate propagator. Propagation does guarantee completeness of solving, but can be needlessly slow. In this paper we describe how to build a (bounds consistent) global propagator for difference constraints that treats them all simultaneously. SAT modulo theory solvers have included theory solvers for difference constraints for some time. While a theory solver for difference constraints gives the basis of a global difference constraint propagator, we show how the requirements on the propagator are quite different. Crucially, we show how to explain propagations by a global difference constraint propagator, in order to use it within a lazy clause generation solver. We give experiments showing that treating difference constraints globally can substantially improve on the standard propagation approach. 

---
# EvoDRC: A Self-Evolving Agentic Framework for Automated DRC Violation Repair 

**Authors**: Bing-Yue Wu, Chia-Tung Ho, Haoyu Yang, Brucek Khailany, Vidya A. Chhabria  

**Link**: [PDF](https://arxiv.org/pdf/2607.20019)  

**Abstract**: Design rule check (DRC) closure remains a major bottleneck in advanced-node physical design. Although detailed routers are rule-aware, residual design rule violations (DRVs) often require manual engineering change order iterations. Automating this process is challenging because repairs must account for complex geometric interactions, preserve circuit connectivity, and avoid introducing new violations. We present EvoDRC, a skill-evolution framework for agentic block-level DRC repair. EvoDRC initializes layer-specific repair skills using knowledge distilled from an unrelated reference design and continuously evolves these skills using traceable repair experience collected from the target design. EvoDRC decomposes the layout into bounded repair regions and assigns an LLM repair agent to each region. Local DRC analysis, connectivity-checking, and impact-preview tools provide feedback on proposed modifications. Repair operations and their resulting DRV changes are stored in a knowledge database and used to evolve the repair skills. Experiments on seven block-level designs from the DAC26 DRC Benchmark show that EvoDRC achieves a 73.5\% overall reduction compared to the reported baseline. 

---
# Safe Remediation as Risk-Constrained Intervention Decision in Microservice Systems 

**Authors**: Chengxiao Dai, Zhaokun Yan, Chenjun Lei, Qiao Li, Luyan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20005)  

**Abstract**: In modern IT operations (IT-Ops), the cost of an incorrect repair often exceeds the cost of no action at all. Yet existing automated remediation systems are designed to generate actions rather than to decide whether intervention is warranted, leaving safety as an afterthought enforced by manual approval. This paper makes three contributions to close this gap: (i) we reformulate safe remediation as a risk-constrained intervention decision problem and cast it as a Constrained Markov Decision Process (CMDP), in which the agent maximizes repair success subject to a bounded false remediation rate (FRR); (ii) we introduce a three-dimensional risk decomposition comprising blast radius, reversibility, and epistemic uncertainty, providing operators with an interpretable per-action safety interface; and (iii) we design a context-adaptive human-in-the-loop (HITL) gate that turns escalation from a binary failsafe into a bandwidth-aware control layer responsive to on-call load and business criticality. The full policy is learned offline from historical incident logs, enabling explicit control of the expected FRR. Experiments on the Train Ticket microservice benchmark with Chaos Mesh fault injection and an RCAEval-aligned fault taxonomy show that our framework reduces FRR by 39% while improving repair success by 2.5 points over a strong runbook baseline, and reduces on-call escalation load by 17% relative to a fixed-threshold variant. 

---
# CLARK: Closed-loop Learning for Adaptive Reasoning over Knowledge Graphs 

**Authors**: Yousef Khan, Luca Gherardini, Marco Maratea, Joel Arrais, Jose Sousa  

**Link**: [PDF](https://arxiv.org/pdf/2607.19996)  

**Abstract**: Machine Learning models are widely used for automating classification tasks by extracting statistical patterns from data. However, their performance deteriorates if the data distribution changes, making them ill-suited to handle uncertain and evolving information. Moreover, they provide limited support for integrating prior knowledge. To address these limitations, we present CLARK (Closed-loop Learning for Adaptive Reasoning over Knowledge Graphs), a framework that integrates knowledge graphs, symbolic rule mining, and probabilistic reasoning under the Logic Programs with Markov Logic Networks (LP$^{\text{MLN}}$) formalism. Starting from CACTUS-derived KGs, CLARK translates graph structure into an LP$^{\text{MLN}}$ program and iteratively enriches it with candidate rules proposed by symbolic learners. These rules are calibrated through probabilistic weight learning, enabling reasoning under uncertainty and refinement of the underlying graph structure. We evaluate CLARK on two medical datasets, analysing both rule quality and downstream classification performance. Results demonstrate that CLARK leads to improved classification performance and more generalisable inference. Overall, CLARK provides a principled approach to constructing adaptive, interpretable, knowledge-driven models for classification. 

---
# Coordinating from Memory: Graph-Structured Experience Reuse for Multi-Agent Adaptation in Dynamic Manufacturing 

**Authors**: Chengxiao Dai, Zhanhui Lin, Zhaokun Yan, Youyang Ni, Chenjun Lei, Luyan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19985)  

**Abstract**: Dynamic manufacturing environments require multi-agent systems to coordinate effectively under frequent operational disturbances such as machine failures, urgent job arrivals, and processing time variations. Existing multi-agent reinforcement learning approaches treat each disturbance episode independently, discarding valuable coordination experience that could accelerate future adaptation. In this paper, we propose a Graph-Structured Experiential Memory (GSEM) framework for multi-agent coordination in dynamic manufacturing. The framework encodes historical coordination episodes as heterogeneous relational graphs that capture task dependencies, machine states, and inter-agent collaboration patterns. When a new disturbance occurs, a graph neural network-based retrieval mechanism identifies structurally similar past episodes, enabling experience-guided policy adaptation rather than learning from scratch. Experiments on dynamic flexible job-shop scheduling benchmarks with three disturbance types show that GSEM reduces makespan by 4.1%-10.0% and adaptation time by 33%-38% compared to the strongest memory-augmented baseline, with the advantage increasing under higher disturbance frequency. Ablation studies and cross-disturbance transfer experiments further validate the necessity of graph-structured encoding and similarity-based retrieval and demonstrate the cross-disturbance generalizability of learned coordination patterns. 

---
# The Giant Hippocampus: From Structural Monoculture to a System of Systems 

**Authors**: Jaeho Seol  

**Link**: [PDF](https://arxiv.org/pdf/2607.19973)  

**Abstract**: AI researchers describe state-of-the-art models as one thing repeated at scale: the Transformer, wired identically for text, pixels, or speech. Neuroscientists describe the cortex as a mosaic - dense Layer 4 in visual cortex for spatial encoding, thick Layers 5/6 in motion cortex for temporal integration - different jobs solved by different structures. This paper argues the gap is a structural error, not a stylistic one, and is measurable. A century of cytoarchitecture, from Brodmann to single-cell Patch-seq, shows distinct cognitive functions are implemented by qualitatively different structures, not by rescaling one template. The convolutional neural network is the field's own proof: local receptive fields and hierarchical depth encoded this prior directly, reaching strong image recognition on far less data than later architectures needed. The paper traces how this lesson was discarded: the "Hardware Lottery" made the Transformer the path of least resistance, not the principled choice, and Mixture-of-Experts, often cited as diversity, in fact partitions parameters among identical experts. A functionalist analysis shows the Transformer is best understood as a functional analog of the hippocampal formation, not a general-purpose cortex - the same mistake as treating cortex as one giant Broca's area, except the field has now standardized on a giant hippocampus, applied to tasks it was never built for: audition, executive gating, working memory. The paper closes with an alternative: a Heterogeneous Topological Network, a System of Systems in which distinct modules keep the inductive bias their computation demands and communicate through standardized interfaces. This is a design discipline for AI architects, not cognitive science: specify modularity before training, using structural evidence as a design input rather than reverse-engineering architecture from a trained model's behavior. 

---
# EvoThink: Evolving Thinking in Large Reasoning Models via Self-Pruning and Aha-Moment Preference Optimization 

**Authors**: Xinbang Dai, Zheyu Xin, Huikang Hu, Lin Ren, Rihui Jin, Guohui Xiao, Guilin Qi, Kuicai Dong, Zhaocheng Du, Yuyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19962)  

**Abstract**: Large Reasoning Models (LRMs) often suffer from overthinking due to redundant verification steps. Existing approaches for mitigating overthinking, such as fast-slow thinking switching and reasoning trajectory compression, fail to make a fine-grained distinction between beneficial and redundant steps within the LRM's reasoning process, and may thus impair reasoning capability in their pursuit of efficiency. To simultaneously improve reasoning efficiency and capability, we propose EvoThink, a framework that reduces redundant verification and encourages the exploration of new reasoning paths. EvoThink comprises two key components: Self-Pruning Training (SPT), an unsupervised method that iteratively prunes redundant reasoning steps and self-trains on the concise trajectories; and Aha-Moment Preference Optimization (AMPO), which, inspired by genetic algorithms, identifies valuable failed reasoning attempts, synthesizes from-wrong-to-right aha-moment data, and optimizes the model to internalize this reasoning pattern. Extensive evaluations across mathematical reasoning and code generation benchmarks demonstrate that EvoThink not only substantially reduces inference-time token usage but also improves the reasoning capability of LRMs. 

---
# SenWorld: A Digital-Twin Simulation for Generating Context-Rich Evaluation Data 

**Authors**: Zenghui Zhou, Xiaoyang Li, Xiaoxuan Qiao, Zhilang Wei, Tianming Lei  

**Link**: [PDF](https://arxiv.org/pdf/2607.19949)  

**Abstract**: Smartphone personal assistants reason over longitudinal personal data, yet evaluating them requires context-rich evaluation data whose correct answers are known, and real device traces are too privacy-sensitive to share. To address this challenge, we present SenWorld, a physically grounded, deterministic, event-sourced digital-twin simulation that generates such data with ground truth fixed by construction. In SenWorld, personas live through a full day in a world built from real map, weather, holiday, and network data; every observable signal is archived in full-system snapshots; and each evaluation case is labeled by a pointer to an existing record rather than by post-hoc annotation or a large language model (LLM) judge. We evaluate this method with 16 personas in Beijing. The generated data closely matches the held-out real-user benchmark in category distribution (Jensen--Shannon divergence (JSD) 0.070) and in the daily rhythm of communication records (JSD below 0.1), though generated records remain shorter than real ones. Without scripted interaction, personas form a fully reciprocated dialogue subgraph and differentiated behavioral repertoires. Projected into 717 evaluation cases, the generated data exposes 78 failures in a production smartphone assistant, concentrating on call and Short Message Service (SMS) records while contacts, schedules, and alarms never fail. The snapshot pointer confirms each failure as an assistant-side retrieval error, with no LLM judge involved. Overall, SenWorld offers a privacy-safe, reproducible, and distribution-checked path to evaluation data whose labels are fixed by construction. 

---
# MOF-Sleuth: Tool-Grounded Reward Alignment for Explainable Fine-Grained MOF CIF Auditing 

**Authors**: Yu Liu, Zhiwei Yang, Diandian Guo, Kun Peng, Fangfang Yuan, Cong Cao, Chaozhuo Li, Zhiyuan Ma, Yanbing Liu, Guobin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.19935)  

**Abstract**: Large metal-organic framework (MOF) databases support simulation, screening, and machine learning through crystallographic information files (CIFs). Subtle chemical and structural errors in these inputs can compromise downstream results and hinder manual inspection. LLM advances in computational chemistry offer paths beyond predictive screening toward fine-grained diagnosis with evidence-grounded explanations. However, two challenges remain: (i) limited fine-grained attribution: MOF-specific validators and machine-learning models scale detection but provide fixed checks, readiness scores, or coarse labels rather than evidence-grounded explanations; and (ii) unreliable CIF reasoning: direct LLM auditing is costly and unreliable because chemical evidence is implicit across atom-site records and requires geometric, connectivity, occupancy, and charge calculations. Both stem from weak coupling between chemical evidence and language-model explanation. We introduce MOF-Sleuth, a reinforcement-guided CIF auditing agent with two modules: a deterministic Forensic Lab and a Sleuth reasoning engine. The Lab derives composition, geometry, connectivity, occupancy, coordination, and charge evidence, and Sleuth uses this evidence to produce an evidence-grounded explanation, error types, and a binary decision. Reward-guided reinforcement learning (RL) turns tool measurements into chemical explanation-level supervision, rewarding not only the final answer but also cited chemical evidence and evidence-supported diagnoses. We introduce Chemically Grounded Diagnosis (Chem-GD), a metric that assesses whether a correct diagnosis is explained by factual, relevant CIF-derived evidence. Across four benchmarks, MOF-Sleuth establishes state-of-the-art performance among LLM-based approaches and MOF-specific machine-learning methods, demonstrating gains in detection, attribution, and grounded explanation quality. 

---
# Long-Term Sequential Decision Making under Risk 

**Authors**: Irmaan, Mirzanejad, Nadjet Bourdache, Abdel-Illah Mouaddib  

**Link**: [PDF](https://arxiv.org/pdf/2607.19914)  

**Abstract**: We study finite-horizon MDP planning under \emph{root-based} (resolute) risk objectives that apply a rank-dependent functional to the distribution of total returns. Such objectives are non-linear in the return distribution and generally break Bellman optimality, so direct optimization by scenario-tree enumeration is intractable. We propose \textbf{ERQDP}, an enumeration-free and sampling-free method that solves a rank--quantile surrogate via exact DP (Dynamic Programming), evaluates candidate policies exactly by DP over return Probability Mass Functions (PMFs) on a discretized return grid (with an explicit rounding bound), and refines the surrogate in an anytime loop that reports an explicit upper--lower gap (certificate) for the target objective up to discretization budgets. Across tested benchmarks, ERQDP returns certified solutions or explicit residual gaps, enables fast risk-parameter sweeps with substantial runtime gains, and supports both risk-averse and risk-seeking behaviors. 

---
# JANUS: Foreseeing Latent Risk for Long-Horizon Agent Safety 

**Authors**: Yuan Xiong, Linji Hao, Shizhu He, Yequan Wang, Lijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.19913)  

**Abstract**: Agent safety is moving from content moderation toward preventing operational failures before tool-using agents act. We propose Janus, a foresight-oriented framework for long-horizon agent safety that trains guards to anticipate delayed risks from partial trajectories. Janus synthesizes diverse agent trajectories via multi-agent simulation and learns a shared policy with two coupled tasks: an anticipation task that forecasts safety-relevant futures and an adjudication task that decides safety from both the observed prefix and anticipated future. The two tasks are jointly optimized with CoAA-RL, which rewards forecasts by their utility for downstream safety judgment. The resulting guard model, Vanguard, blocks unsafe actions before execution. Across four agent-safety benchmarks, Vanguard improves average protection by 15.9 percentage points over baseline guards while increasing benign task completion by 5.1 percentage points. 

---
# DocOps: A Verifiable Benchmark for Autonomous Agents in Complex Document Operations 

**Authors**: Jiazhen Jiang, Boxi Cao, Lingyong Yan, Yaojie Lu, Hongyu Lin, Shuaiqiang Wang, Dawei Yin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.19865)  

**Abstract**: As autonomous agents rapidly evolve, their ability to reliably manipulate ubiquitous digital documents has become critical for enabling general-purpose AI assistants and automating complex workspace workflows. In this paper, we introduce DocOps, a deterministically verifiable evaluation framework underpinned by a hierarchical taxonomy that deconstructs document operations inspired by real-world practices into atomic dimensions and escalating workflow complexities. Based on DocOps, we systematically evaluate representative closed- and open-source models across various agentic harnesses, revealing that even the most advanced frontier configurations still exhibit profound limitations when handling highly coupled, long-range tasks. Furthermore, a fine-grained analysis of existing agents' manipulation behaviors uncovers 3 key failure modes: long-term state tracking collapse, shallow semantic verification, and destructive editing of structural metadata. Ultimately, our work exposes the capability boundaries of agents in maintaining global document consistency, shedding light on the future design of robust, non-destructive agents for complex digital ecosystems. 

---
# Know Your Agent: Reconnaissance-Driven Pentesting of AI Agents 

**Authors**: Or Zion Eliav, Eyal Lenga, Shir Bernstien, Yisroel Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2607.19837)  

**Abstract**: Traditional pentesting uses reconnaissance at each step to uncover unseen weaknesses, build stronger attacks, and advance the objective; we argue that AI agents require the same treatment. We formalize agent reconnaissance by modeling the process and identifying the knowledge assets it seeks to extract: what they are, how they are used, and which agent weaknesses they exploit to give adversaries leverage in indirect prompt injection attacks. We instantiate these insights in Know Your Agent (KYA), a framework that automates black-box, reconnaissance-driven pentesting by probing agents, building target profiles, and using those profiles to craft stronger attacks. We evaluate KYA on agent-security benchmarks and a real-world coding agent, and release KYA, its benchmarks, and baseline implementations for reproducibility. 

---
# Rewarding Better Thinking for LLM Preference Alignment 

**Authors**: Xubo Liu, Wenya Guo, Ruxue Yan, Xinying Qian, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19824)  

**Abstract**: LLM preference alignment aims to optimize models toward human preferences across diverse user instructions. Reinforcement learning has become a major post-training approach for this goal, but existing proxy rewards are often outcome-level, mainly evaluating the final response while providing limited guidance for the reasoning trajectory. This can make credit assignment coarse when multiple responses receive similar final scores, leaving trajectory-level preferences under-specified. To address this limitation, we propose Thinking Checklist Reward (TCR), a process-oriented reward for RL-based preference alignment. TCR converts preference pairs into sample-specific thinking checklists and uses them to evaluate whether the generated reasoning trace addresses the preference-implied considerations. To reduce overlap with outcome-level supervision, TCR further introduces an exponential moving average (EMA) residual formulation to isolate a complementary thinking surplus beyond what is predictable from the outcome reward. Experiments on five models from three model families show that TCR consistently improves alignment performance across diverse benchmarks, with ablations further validating the importance of EMA-based residual formulation and sample-specific checklist supervision. 

---
# Silent Failures in Multimodal Agentic Search:A Diagnostic Taxonomy and Cross-Judge Evaluation 

**Authors**: Zhengxian Wu, Junjie Gao, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19793)  

**Abstract**: Multimodal agentic search systems increasingly rely on external tools to answer knowledge-intensive visual questions. However, existing evaluations mainly focus on final-answer accuracy and may miss failures in the search trajectory. In this work, we study such hidden reliability issues as silent failures. We introduce a six-category taxonomy covering modality shortcuts, phantom grounding, wrong-evidence-right-answer cases, over-retrieval laundering, cross-modal contradiction, and provenance hallucination. Based on this taxonomy, we build a trajectory-level diagnostic pipeline that evaluates both answer correctness and evidence-grounding quality under a unified ReAct-style scaffold. Experiments on MMSearch-Plus trajectories across four frontier multimodal models show that surface accuracy consistently overestimates true trajectory-level correctness. We further use cross-judge validation, blank-image stress tests, and tool ablations to show that silent failures are capability-dependent and often shift rather than disappear. Home-page: this https URL 

---
# Symbol and Footprint Database for Electronic Components by Agentic Recognition and Generation 

**Authors**: Yichen Shi, Yuzhi Liu, Zhuofu Tao, Li Huang, Yuhao Gao, Ting-Jung Lin, Lei Hel  

**Link**: [PDF](https://arxiv.org/pdf/2607.19767)  

**Abstract**: A rich and recognizable component library is the cornerstone of printed circuit board (PCB) design and generation. Traditionally, engineers manually create symbols and footprints and design PCB schematics, which is time-consuming and error-prone. Leveraging multimodal large language models (MLLMs), we develop SFgen, an agentic recognition and generation flow of symbol and footprint for electronic components. SFgen achieves 86% accuracy for symbol generation and 80% accuracy for footprint generation. We use the SFgen method to create SFnet, a database of symbols and footprints. It now has 1000 components and is expanding constantly, which lays the foundation for automatic generation of PCB designs. 

---
# Edge Intelligence in Civil Aviation: Paradigms, Techniques, and Applications 

**Authors**: Wenbin Li, Zhongtian Liao, Bolin Liu, Yongjie Zhou, Jingling Wu, Xiaoyong Lin, Jing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.19676)  

**Abstract**: Civil aviation is safety critical and its operations, from flight decks and towers to ramps and maintenance, generate massive, heterogeneous data at the network edge. Yet cloud centric deployment of large Artificial Intelligence (AI) models often produces high task latency, lacks offline capability in communication denied environments, and requires centralizing sensitive data, raising privacy and sovereignty risks. Edge AI moves perception, prediction, and decision logic closer to the data producers via compression, collaborative inference, and split learning, thereby reducing latency, bandwidth, and exposure while enabling graceful operation during disconnections. This paper provides a panoramic view and a common understanding of edge intelligence tailored to civil aviation. We firstly articulate the operational motivations for edge AI, and then review recent techniques for edge inference and edge learning. We then introduce the organizational computing paradigms and the respective configurations in civil aviation environments; finally, we describe the emerging applications and the future research trends of edge intelligence in civil aviation. We argue that a refined edge solution can complement cloud foundations to deliver low latency, privacy preserving, and resilient AI services across the civil aviation lifecycle. 

---
# Knowledge-Centric Self-Improvement 

**Authors**: Xuefei Julie Wang, Lauren Hyoseo Yoon, Chengrui Qu, Amanda Zichang Wang, Atharva Sehgal, Eric Mazumdar, Yisong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2607.19592)  

**Abstract**: Self-improving AI systems typically treat the agent as the object that improves, by optimizing prompts, workflows, harnesses, or even the agent's own code. This agent-centric view can make improvements expensive to maintain and difficult to transfer, because gains become tied to a particular agent design, task distribution, or adaptation run. We study a complementary paradigm: knowledge-centric self-improvement, in which agents remain generic and disposable while the persistent object is a curated knowledge base that agents can leverage for future tasks. We conduct controlled case studies to operationalize this idea via a simple protocol. Agents attempt one task, then contribute evidence-grounded insights to a shared knowledge base via task-level and cross-task forums, followed by knowledge distillation. Because self-improvement is contained in the knowledge rather than the agent, improvement can be more inspectable, transferable, and portable. Across abstract reasoning, coding, and terminal benchmarks, this protocol improves solve rates while reducing dollar cost relative to agent-centric baselines. The resulting distilled knowledge also transfers to held-out tasks and across LLM families, indicating that the improvement is not merely an LLM- or run-specific behavior. These results support a new view of self-improving agentic systems: progress can be driven primarily by the curated persistent knowledge. Code is available at this https URL. 

---
# Sophisticated Policies from Epistemic Priors 

**Authors**: Wouter W. L. Nuijten, Bert de Vries  

**Link**: [PDF](https://arxiv.org/pdf/2607.19518)  

**Abstract**: Sophisticated Inference is a variant of active inference often associated with recursive belief modeling and tree search. We argue that its central computational role is simpler: within a planning horizon, it makes active inference closed-loop by allowing future actions to depend on future states and observations. This closed-loop structure can be represented in the epistemic-prior variational free energy framework. Epistemic priors supply the active-inference objective, while a joint posterior over future states and actions supplies the state-contingent control structure. We evaluate this decomposition in the Reactivity Maze, a stochastic benchmark designed to separate epistemic incentive from inner-horizon closed-loop control. The comparison includes three variational objectives with the same state-action posterior family, an action-state factorized active inference objective, Sophisticated Inference, and standard Expected Free Energy planning. The results show that neither ingredient is sufficient on its own. Methods without an epistemic component do not seek information, while methods that prevent future actions from depending on future states cannot turn information into reliable goal-reaching. By contrast, both Sophisticated Inference and full-joint epistemic-prior active inference solve the environment by combining epistemic drive with closed-loop inference. These results show that the advantage associated with Sophisticated Inference need not be specific to tree search itself. It arises from the closed-loop form of active inference, and this form can be represented in epistemic-prior variational inference when the posterior keeps future actions dependent on future states. 

---
# The Chronos Vulnerability: A Taxonomy of Temporal Persistence and Memory-Based Deception in Agentic AI 

**Authors**: Om Narayan, Ramkinker Singh, Praveen Baskar  

**Link**: [PDF](https://arxiv.org/pdf/2607.19433)  

**Abstract**: The transition from stateless generative models in artificial intelligence to stateful, autonomous agents represents an architectural evolution that, while providing the capabilities of long-term planning and the automation of enterprise workflows, also represents the introduction of a new form of security threat, the Chronos Vulnerability. The Chronos Vulnerability represents the threat of memory-based attacks, including the Memory Injection Attack (MINJA) and the sleeper agent, in which the internal belief system of the autonomous agent is compromised, effectively decoupling the attack vector from the final catastrophic event. This study formalizes the threat model for persistence-based attacks and the threat of Dynamics Blindness in the context of the World of Workflows benchmark, demonstrating that traditional endpoint content filters are insufficient for the current stateful architecture. Consequently, this study synthesizes a defense-in-depth landscape, categorizing emerging frameworks such as diagnostic trajectory guardrails (AgentDoG), formal temporal verification (Agent-C), immunological memory consensus (A-MemGuard), and hardware-anchored trust via GPU-based Trusted Execution Environments (TEEs) and Zero-Trust memory architectures. 

---
# FORCE-Bench: A Benchmark, Dataset, and Evaluation Harness for Agentic AI in Enterprise Finance 

**Authors**: Wolfgang M. Pauli, Sarah Panda, Kidus Admassu, Said Bleik, Ademola Okerinde, Jeremy Reynolds  

**Link**: [PDF](https://arxiv.org/pdf/2607.19409)  

**Abstract**: Recent advances in large language models have accelerated deployment of agentic systems in operational finance. Existing benchmarks emphasize measuring general capabilities, instruction following, or safety, but few directly address the operational finance workflows that agentic systems are now being deployed to automate. Finance professionals require agents to not only provide factually sound and properly grounded information, but also ensure that this information is verifiable and consistently adheres to rules and constraints of the operational finance domain. We introduce FORCE-Bench, which contains 251 expert-annotated queries and evaluates responses using a rubric-based framework calibrated to the requirements of the operational finance domain, across eight dimensions: accuracy, citations, clarity, depth, groundedness, recency, relevance, and structure. FORCE-Bench assesses agentic systems on three task types: financial obligation research (querying ERP systems for accounts receivable and payable data), financial entity performance research (answering time-bound questions from public filings and market data), and business brief generation (synthesising multi-source company intelligence reports). To reflect real deployment conditions, we evaluate our purpose-built agent, as well as the general-purpose agentic systems, under common tool access and latency-bounded settings. Results show that general-purpose agentic systems do not consistently meet finance-domain quality requirements under operational constraints, while the purpose-built Finance Agent for Microsoft 365 Copilot is more reliable across dimensions. We release the dataset, rubrics, harness, and analysis code as open-source to support reproducible comparison and adaptation to other enterprise finance environments. 

---
# ITPEval: Benchmarking Formal Translation Across Interactive Theorem Provers 

**Authors**: Jiayi Wu, Robert Joseph George, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2607.19407)  

**Abstract**: Formal theorem proving has emerged as a frontier challenge for machine learning, yet the ecosystem is fragmented: proofs remain siloed across incompatible systems, limiting both training data for learning-based provers and the portability of verified results. We present ITPEval, the first benchmark for evaluating automated formal proof translation across four major ITPs (Lean 4, Rocq, Isabelle, and HOL Light), spanning two distinct logical foundations. Our benchmark comprises 1,560 source files and 6,848 theorems organized into a controlled tier of axiomatized files that isolates foundational translation difficulty, and an ecosystem tier drawn from real libraries that exposes API and proof-style mismatches. We release itpeval, a unified multi-ITP verification infrastructure with state-isolated warm backends that preserve per-artifact native checking semantics. We evaluate both statement and proof translation across five frontier and open-weight LLMs on 12 directed translation pairs: statement translation peaks at 29.1% pass@1 and proof translation at 10.5%; controlled theorems reach 29.7% proof pass@1 versus 5.2% for ecosystem-level translations, confirming that library mismatch is the dominant bottleneck. In addition to pass@k evaluation, a deterministic Lean 4 BEq check establishes equivalence for 54.0% of verified source-to-Lean 4 miniF2F statement translations, showing that native type-checking alone can substantially overestimate semantic fidelity; in an autoformalization/auto-informalization round-trip study, Rocq and HOL Light are easier formalization targets than Lean 4 and Isabelle, while multi-ITP context improves pooled Lean 4 success from 4.8% to 10.6%. Our benchmark, verification infrastructure, and evaluation pipelines are publicly released. 

---
# HyGRL: Adaptive Hybrid Graph Reasoning for Multi-Entity Questions 

**Authors**: Junyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19398)  

**Abstract**: Multi-entity compositional questions pose significant challenges to existing retrieval-augmented language models. Conventional methods fall into a dilemma: standard RAG lacks dynamic reasoning, traditional Graph-RAG is limited by structural sparsity, and LLM-constructed Graph-RAG incurs prohibitive costs. We propose \textbf{\fwa}, a unified framework that embeds unstructured text into structured knowledge graphs, creating a heterogeneous network for flexible evidence retrieval. Reasoning is formulated as adaptive structure induction, learned via a robust two-stage process: (1) imitation learning distills heuristic expert signals, and (2) reinforcement learning refines the policy using LLM-driven preference rewards. Experiments demonstrate that {\fwa} effectively merges textual richness with structural knowledge, outperforming SOTA baselines in answer accuracy and reasoning fidelity while maintaining extremely low token costs and near real-time inference((code available at this https URL) . 

---
# CrackedPDFs: A Controlled Benchmark for Hidden Prompt Injection in PDFs 

**Authors**: Pukaphol Thienpreecha  

**Link**: [PDF](https://arxiv.org/pdf/2607.19396)  

**Abstract**: Document-based LLM systems often flatten a PDF before guardrails inspect it. That step can discard evidence that an instruction was never visible to the user. We introduce CrackedPDFs, a controlled benchmark for hidden prompt injection in PDFs. The benchmark contains 29,322 generated PDFs from 4,983 base docu ments. It includes 9,774 injected files and 19,548 benign or matched-confounder files. We evaluate PromptGuard and a rule baseline. We also evaluate structural only learned models and a sanitized hybrid detector. The evaluation uses held-out provenance splits and paired benign-confounder controls. It also uses label-shuffle checks and shortcut audits. On a 2,919-document held-out test set, the hybrid de tector reaches 0.960 F1. ROC-AUC is 0.998 and PR-AUC is 0.997. It also ranks injected files above matched benign confounders in 95.9% of 973 pairs. Prompt Guard has low recall when given extracted text only. Structural-only learned mod els are weak under paired controls. A text-only TF-IDF model reaches perfect held-out scores but fails shortcut audits. These results show that document-aware hybrid detection is useful under controlled paired evaluation. They do not show broad real-world robustness or reliable cross-family generalization. 

---
# Euclean: Automated Geometry Problem Formalization with Unified Verification in Lean 

**Authors**: Linbin Tang, Jingyan You, Zilin Kang, Hanzhang Liu, Sophia Zhang, Zenan Li, Chenrui Cao, Liangcheng Song, Jiaao Wu, Xian Zhang, Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19374)  

**Abstract**: Recent formal reasoning systems have reached IMO-level performance, yet they leave a fragmented landscape: algebra and number theory are handled in Lean, while geometry still relies on domain-specific languages with limited formal guarantees. This split increases the trusted computing base and hinders unified model development. Existing geometry-in-Lean efforts (LeanEuclid, LeanGeo) introduce custom axiom systems incompatible with standard Mathlib, and their small scale ($<$ 1,100 problems) limits large-scale training. Native Mathlib autoformalization of geometry, however, poses distinct challenges: implicit diagrammatic assumptions (e.g., topological configuration and non-degeneracy) must be made explicit rather than deferred to external solvers, and models must adapt to Mathlib's small, rapidly evolving geometry infrastructure. We present Euclean, a four-stage framework - constraint explication, configuration anchoring, formalization mapping, and iterative repair - for automatically formalizing geometry in native Mathlib. We construct OMNI-Geometry (768 competition problems) and Numina-Geometry (177,597 problems), the largest geometry formalization dataset in Lean. Human evaluation shows 48.89% TOP1 and 73.33% TOP5 accuracy. Training Goedel v2 on our formalizations improves proof success from 13.6% to 15.1%, validating dataset quality for unified neural theorem proving. Code and datasets: this https URL. 

---
# Mitigating Scaffolding Collapse in Socratic Tutors via Representation Alignment 

**Authors**: Jing Shao, Qifeng Wu, Hanyu Zhang, Sixia Sun, Jun Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19371)  

**Abstract**: Large language model (LLM)-based Socratic tutors increasingly guide students through multi-turn questioning, but they can suffer from scaffolding collapse: under sustained student pressure, a tutor gradually abandons guided inquiry and reveals solutions directly. Prior defenses primarily constrain observable responses through prompting, preference optimization, or filtering, leaving the internal representation drift that precedes trajectory-level collapse largely unaddressed. We propose Scaffold-Preserving Representation Alignment, a two-stage framework that first warms up a Socratic tutor with supervised fine-tuning, then combines trajectory-weighted direct preference optimization with a margin-preserving representation loss anchored to frozen reference states. Our method is designed to maintain separation between scaffold-preserving and collapse-inducing hidden states across dialogue turns. We evaluate our method across five STEM disciplines and five red-teaming attack strategies. On Qwen3-8B, our method lowers Collapse Rate to 32%, delays average collapse onset beyond nine turns, and keeps over-refusal low, suggesting that representation-level alignment can improve the robustness of long-horizon Socratic tutoring under our red-teaming protocol. 

---
# Beyond Tracking or Shortcut: Composition-Bounded Predictive States in Poker Autoregressive Models 

**Authors**: Quanhao Li, Qianyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.19369)  

**Abstract**: Hidden-state probes often recover latent labels in imperfect-information sequence models, but this alone does not establish that a model maintains a posterior belief distribution over hidden states. This paper studies this ambiguity in a no-range Limit Hold'em autoregressive model trained only on action and value targets, not on an opponent's hand or range. Opponent-range probes are positive after action/value controls in two of three seeds, and the behavior head predicts held-out actions about five percentage points above a baseline using only observable public history. However, visible public betting composition explains more opponent-range signal than residual hidden states, suggesting that most recoverable information comes from betting summaries. Action/value+composition baselines reach 16.5-16.7% top-10 accuracy while composition-residual hidden probes fall to 11.4-12.2%, and matched-composition comparisons are negative in every seed. We call this evidence pattern composition-bounded predictive support: hidden states remain behavior-predictive and opponent-range correlated, but most recoverable range information is explained by visible betting composition rather than residual hidden-state structure. This is a case-study claim about opponent-range representational evidence, not exact Bayesian posterior tracking or a causal belief mechanism. Synthetic control and oracle validations show that the same diagnostics accept posterior-sensitive states and reject raw composition states under matched controls. Thus positive belief probes should be interpreted through targeted alternatives before being treated as evidence of belief tracking. 

---
# Spectral-LSH: Sub-Quadratic Prompt Compression via Krylov-Projected Locality-Sensitive Hashing 

**Authors**: Ali Mahdavi, Azaseh Zamanifar, Amirfarhad Farhadi, Omid Kashefi  

**Link**: [PDF](https://arxiv.org/pdf/2607.19368)  

**Abstract**: Long-prompt inference remains expensive because prefill attention scales quadratically with sequence length. We propose Spectral-LSH, a training-free prompt compression method that operates before the prompt enters the language model. Spectral-LSH approximates the dominant components of an implicit attention-kernel operator using a Krylov subspace method together with random features, avoiding explicit $O(N^2)$ attention-kernel materialization. It then applies SimHash in the resulting attention eigenspace to group similar tokens and aggregate them into macro-tokens with causal positional assignments.
We evaluate Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, and Qwen2.5-14B-Instruct on C4. Our experiments reveal a compression-ratio phase transition. Below $\rho = 4 \times$, local token redundancy is low enough that lightweight chunking typically provides the best latency--quality trade-off. Above $\rho = 8 \times$, the spectral path preserves quality that chunking loses. At $\rho = 16 \times$, Qwen2.5-7B (adaptive) reduces the PPL ratio from 353.409 to 196.963, while Qwen2.5-14B (adaptive) reduces it from 9.533 to 3.427.
On a small long-context structured stress test containing JSON-like, code-like, and table-like inputs, local LSH also improves every metric over chunking at $8 \times$. The adaptive backend captures both regimes by using the chunk path at low compression and spectral clustering at high compression, although chunking remains the fastest backend in total latency. 

---
# Rethinking Uncertainty Evaluation in Large Language Models 

**Authors**: Krish Matta, Atharv Naphade, Andy Zou  

**Link**: [PDF](https://arxiv.org/pdf/2607.19367)  

**Abstract**: Calibration is the primary criterion for evaluating LLM confidence, but it is insufficient: it admits trivially incoherent estimators, depends on the evaluation distribution, and does not test the extent to which the estimation can be interpreted as a consistent, underlying probability function. What we actually need is for LLM confidence estimates to satisfy the conditions required of coherent probabilistic beliefs. We formalize these conditions along three axes (structural coherence, faithfulness, and usefulness) and operationalize them as the C1 metrics. Widely used estimators systematically violate these conditions despite appearing well-calibrated: models assign lower confidence to logically easier questions 31\% of the time, and common interventions reducing RMSCE leave structural violations unchanged, suggesting that calibration is orthogonal to probabilistic validity. RLHF and chain-of-thought improve usefulness metrics without restoring coherence. Our results show current LLM confidence estimates cannot be interpreted as coherent probabilities; our framework provides the tools to measure and close this gap. 

---
# Geometry-Guided Constraint Learning for LLM Safety Classification 

**Authors**: Fumiaki Uehara, Koo Imai, Masato Tsutsumi, Keigo Kansa, Sora Usui, Yuki Kobiyama  

**Link**: [PDF](https://arxiv.org/pdf/2607.19366)  

**Abstract**: Safety as Polytope (SaP) learns linear half-space constraints in LLM hidden space but requires per-category tuning of the constraint count K. We show that sparse autoencoder (SAE) feature extraction resolves this: K=2 becomes optimal for 12/14 categories on Qwen3.5-9B, achieving 96-99% accuracy per category on our BeaverTails classification benchmark, largely eliminating the need for exhaustive sweeps (K=4-25 with random initialization). This convergence to two planes is consistent with the Linear Representation Hypothesis, providing suggestive evidence that safety boundaries in this setting admit a low-dimensional linear description in the SAE feature space. Building on this geometric perspective, we introduce a cone constraint whose learnable aperture adapts to each category's cluster concentration, stabilized by a three-phase training 

---
# Logic-Guided Data Extraction with Answer Set Programming and Large Language Models 

**Authors**: Mario Alviano, Lorenzo Grillo, Nicola Leone, Fabrizio Lo Scudo  

**Link**: [PDF](https://arxiv.org/pdf/2607.19365)  

**Abstract**: When Large Language Models (LLMs) are used for semantic data extraction from unstructured text, producing candidate relational facts from natural language, they may remain unreliable for tasks requiring complex combinatorial reasoning and global consistency. This paper proposes a logic-guided data extraction framework combining LLM-based extraction with Answer Set Programming (ASP). The LLM produces candidate facts, whereas ASP performs validation, inference, consistency checking, and control. Unlike existing pipelines that query the LLM independently for all target predicates, the proposed approach uses ASP reasoning to identify which predicates are logically admissible at each stage and to guide extraction queries. By interleaving LLM calls with ASP derivation, the framework infers logically implied facts without further extraction and detects inconsistencies early. We formalize the pipeline and prove that, under mild assumptions, it is equivalent to the baseline approach with respect to the final extracted facts, while requiring fewer LLM calls. We also introduce a caching mechanism for logic-based control queries, exploiting monotonicity of conjunctive queries over incrementally constructed fact sets to reduce solver invocations. Experiments on ASP-derived benchmarks show that the framework reduces LLM calls and improves extraction quality by mitigating spurious outputs, demonstrating the value of non-monotonic logic programming for controlled semantic extraction. 

---
# Statistically Grounded Sparse-Feature Interventions for Activation-Space Control in Large Language Models 

**Authors**: Oshayer Siddique, J. M Areeb Uzair Alam, Md Jobayer Rahman Rafy, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2607.19364)  

**Abstract**: Activation steering offers a lightweight alternative to fine-tuning for behavioral control of large language models, but SAE-based steering methods often rely on learned steering objectives or single-criterion feature selection. We introduce a transparent SAE-feature steering pipeline that first applies a six-condition reliability filter, then ranks sparse features through an unweighted Borda consensus over three complementary statistics: $F$-test, KSG mutual information, and Cohen's $d$. The resulting steering direction is constructed as a Cohen's-$d$-weighted combination of SAE decoder rows, providing an optimization-free direction motivated by Fisher-LDA under approximate SAE-feature decorrelation. Across three Gemma-family models, four behavioral domains, and 356 layer-strength configurations, the method produces measurable domain-specific shifts while revealing a substantial gap between raw attribute movement and quality-preserving generation. In the strongest configuration, logical-correctness steering reaches a primary-score delta of $+1.16$ in Gemma~2 9B; however, our broader finding is that usable steering is highly localized by model, domain, layer, and strength. These results argue that activation-steering evaluations should report quality-conditioned success alongside raw behavioral shift. Our code and data are available at this https URL. 

---
# AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally 

**Authors**: Shaowen Wang, Yuke Zheng, Tansheng Zhu, Shuang Chen, Shaofan Liu, Suncong Zheng, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.19363)  

**Abstract**: Rotary Position Embedding (RoPE) is widely adopted in Transformers to encode positional information, yet standard implementations enforce a uniform frequency schedule and scaling across all attention heads. Using simplified retrieval tasks and length generalization scenarios, we show -- both empirically and theoretically -- that heads with different functional roles require distinct frequency ranges and attention scaling factors to operate effectively. Ignoring this structure leads to suboptimal utilization of embedding dimensions and degraded performance, particularly under long-context settings. To address these limitations, we propose AdaRoPE, which equips each attention head with learnable rotation frequencies and attention scaling factors. Pretrained LLMs with AdaRoPE consistently outperform existing RoPE variants, including partial RoPE and NoPE baselines. For context extension, we further show that uniform frequency and attention scaling, used in methods such as YaRN, are suboptimal. By applying head-specific scaling, AdaRoPE enables better context extension while better preserving short-context performance in both the extrapolation setting and the long-context continued pretraining setting. These results highlight the importance of optimizing rotary position embedding at the level of individual attention heads. 

---
# GraphContainer: A Unified Platform for Comparing and Debugging Graph RAG Methods 

**Authors**: Seonho An, Chaejeong Hyun, Min-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.19362)  

**Abstract**: Graph RAG mitigates hallucinations and stale knowledge in LLMs, particularly for multi-hop question answering. However, existing approaches remain highly fragmented and incompatible. The structural heterogeneity of graph formats across different frameworks and the lack of granular visualization tools make it exceedingly difficult to evaluate and compare retrieval behaviors. To bridge this gap, we propose GraphContainer, a novel platform designed to unify and visualize diverse graph RAG workflows. GraphContainer features two key components: (1) a Unified Graph Representation (UGR) layer that seamlessly standardizes multi-format graphs, and (2) a Graph Recorder that tracks and visually renders the step-by-step retrieval process. Through an interactive web interface, we demonstrate GraphContainer's ability to import heterogeneous graphs and perform live, traceable visual debugging of graph RAG methods. Ultimately, we show how GraphContainer enables controlled comparisons of various graph formats and retrieval strategies, lowering the barrier for researchers and practitioners to design optimal graph RAG pipelines. A demonstration video is available at this https URL. 

---
# Lifted Representation Hypothesis in Language Models 

**Authors**: Bumjin Park, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2607.19360)  

**Abstract**: Large language models (LLMs) often answer queries by mapping individual observations to more general rule-like structures. However, it remains unclear how these structures are stored, selected, and revised. To study this process, we propose thelifted representation hypothesis: LLMs update memory through shared latent structures rather than isolated instance-level facts. This view frames lifting as an efficient use of symmetry across instances, and shattering as the refinement of coarse lifted structures into more specific subtypes. We evaluate LLMs' lifting and shattering through controlled exception-learning experiments across in-context learning, LoRA, and full fine-tuning. We find that LLMs are vulnerable to shattering failures when data are governed by nested rules and exceptions, while lifting often occurs prematurely. These results highlight the need to study the relation between data and rule structures in LLMs. 

---
# Profile-Graph Memory for LLM Agents: Implicit Cross-Entity Traversal through Narrative Profiles 

**Authors**: Shengtong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.19359)  

**Abstract**: Long-term memory is essential for LLM agents that interact across sessions, yet current memory benchmarks primarily evaluate single-hop recall, leaving multi-hop association largely unmeasured. We make three contributions. First, we introduce MemHop, a multi-hop memory benchmark of 1,000 questions at hop depths 1-5 across 10 social-network scenarios, with per-hop evidence annotations. Second, we present Profile-Graph Memory (ProGraph), a two-layer memory architecture combining (i) profile expansion -- substring-matched traversal of entity names that naturally appear in LLM-written profile narratives, a minimal alternative to explicit knowledge-graph construction -- and (ii) compression residuals -- exact dates, quantities, and named items co-extracted with each profile update at zero extra API cost. Third, a full-grid ablation shows cross-benchmark mechanism specialization: profile expansion drives multi-hop reasoning (-22.6pp on MemHop when removed) while compression residuals drive precision recall (-8.6pp on LoCoMo when not co-extracted), with cross-effects under 3pp within a single architecture. ProGraph averages 80.1% on MemHop (matching the FullContext reference) and 78.4% on LoCoMo (exceeding FullContext by 11.3pp), outperforming Mem0, A-Mem, HippoRAG, and RAG on both. We release MemHop, ProGraph, and baseline implementations. 

---
# LISA: Linear-Indexed Sparse Attention for Efficient Long-Context Reasoning 

**Authors**: Yu Zhao, Zekun Zhang, Fan Jiang, Bo Zeng, Linlong Xu, Shimin Shan, Yu Liu, Longyue Wang, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.19358)  

**Abstract**: Recent advances in long chain-of-thought reasoning models such as DeepSeek-R1 have led to increasingly longer inference context lengths under the test-time scaling paradigm. However, the O(n^2) computational complexity of standard self-attention causes inference costs to grow sharply with long sequences, limiting the deployment of long-CoT reasoning in production settings. To address this, we propose LISA (Linear-Indexed Sparse Attention), a plug-and-play attention replacement module that requires no pretraining from scratch. LISA integrates two lightweight components in parallel within the original model: (1) a Linear Attention module that provides long-range memory with O(n) time complexity; (2) a Lightning Indexer that selects the top-M important tokens from the full context to feed into a Sparse Self-Attention. The two branches are fused via a gating mechanism, reducing inference complexity from O(n^2) to O(nM) (M << n) for generating n tokens. We design a two-stage training pipeline: Stage 1 initializes the model by integrating the linear attention to capture long-range dependencies, complemented by a sliding-window attention mechanism that is optimized via knowledge distillation to approximate the full self-attention distribution of a frozen teacher model. In Stage 2, we further introduce the Indexer to replace the static sliding-window mechanism, enabling dynamic token selection from broader contexts. The Indexer is trained using a novel per-head KL divergence loss, which aligns its selection behavior with the attention patterns of the teacher model. Experiments on DeepSeek-distilled-Qwen models demonstrate that LISA achieves a 50% inference speedup under 16K-token context, while improving average performance by 5.6% on reasoning benchmarks including AIME and MATH-500. 

---
# Stochastic Primal-Dual Decoding for Multiobjective Generative Recommender Systems 

**Authors**: Dmitrii Moor, Ben Carterette, Senthilkumar Krishnamoorthy, Kyle Kretschman, Denis Beslic, Melissa Yalla, Alice Y Wang, Mounia Lalmas  

**Link**: [PDF](https://arxiv.org/pdf/2607.19357)  

**Abstract**: Recent advances in recommender systems (RS) have shown substantial performance gains through generative modelling. In practice, recommendation often involves constructing slates -- ordered lists of items -- that must satisfy multiple objectives beyond relevance, such as constraints defined over item attributes or fairness constraints. Existing multiobjective approaches either rely on post-processing techniques designed for non-generative settings, or incorporate auxiliary objectives directly into model training. The former does not explicitly account for the sequential nature of generative RS, while the latter is often impractical in large-scale systems.
We propose a lightweight, inference-time decoding layer that augments autoregressive generative RS to support multiobjective slate generation without modifying or retraining the underlying model. We formulate decoding as an online constrained optimisation problem, where items are selected sequentially, and trade-offs between relevance and auxiliary objectives are adjusted dynamically based on the remaining constraint slack, i.e., how much of each objective remains to be satisfied. This is implemented via a stochastic primal-dual approximation scheme that balances relevance and auxiliary objectives during generation.
We provide theoretical guarantees on constraint violation and regret, and evaluate the proposed approach through extensive offline experiments and a large-scale online A/B experiment in a real-world recommender system. Our results show consistent improvements in multiobjective trade-offs, including a +1.8\% gain in the auxiliary objectives achieved at zero cost to user satisfaction. 

---
# NEXUS: Structured Runtime Safety for Tool-Using LLM Agents 

**Authors**: Elias Hossain, Md Mehedi Hasan Nipu, Tasfia Nuzhat Ornee, Rajib Rana, Niloofar Yousefi  

**Link**: [PDF](https://arxiv.org/pdf/2607.19356)  

**Abstract**: Tool-using LLM agents increasingly execute high-impact actions, making runtime safety monitoring essential. We present NEXUS (Neural EXecution Utility and Safety), a structured-plan safety monitor that applies a formal intervention policy to select among four actions: allow, block, request confirmation, or request revision. NEXUS combines deterministic safety rules, argument-level inspection, and a calibrated logistic-regression risk score for graded escalation. On a 128-instance synthetic benchmark, NEXUS achieves an F1 score of 0.949 and a 4-class intervention accuracy of 0.6406, outperforming rule-only intervention selection by 27.3 percentage points. It also improves over rule-only on R-Judge (F1 = 0.861 vs. 0.849), matches rule-only on AgentHarm due to threat-model limits, and achieves 0% ASR at 99% control allow on IPI. On the rule-blind NEXUS-Stress benchmark, NEXUS reaches an F1 score of 0.881, highlighting the difficulty of fine-grained intervention routing. With 0.205 ms median latency, NEXUS adds under 0.1% overhead to typical agent loops. Code, benchmarks, and the calibrated risk scorer are publicly released. 

---
# Information Discernment in Large Language Models 

**Authors**: Joshua Ashkinaze, Laura Kurek, Alina Faisal, Tongyuan Miao, Mariam Joseph, Ceren Budak, Eric Gilbert  

**Link**: [PDF](https://arxiv.org/pdf/2607.19355)  

**Abstract**: LLMs are increasingly used with external knowledge sources like the internet. Do they weigh information appropriately -- updating more for reliable sources (source discernment) and more when claims bring priors closer to the truth (truth discernment)? We formalize this as information discernment and introduce Learn2Discern (L2D), an experimental framework and benchmark grounded in three normative axioms with interpretable metrics. To establish external validity, a pre-registered, quota-matched user study (n=299) confirms that real LLM users endorse all three axioms and report that violations reduce their trust and usage intent. Across 13 models and nearly 670K trials, we find consistent failures across both dimensions: models perform near chance on source and truth discernment, rely on source popularity twice as much as source reliability, and update roughly equally whether a claim improves or worsens their position relative to the ground truth. Models integrate external knowledge most effectively on datasets where their priors are already the most accurate. Newer and larger models improve truth discernment but not source discernment, a blind spot that model complexity does not address. We identify simple inference-time interventions that improve both forms of discernment. We release our dataset and survey as a testbed for a core alignment property that scales in importance as LLMs replace traditional search. 

---
# FormulaSPIN: Self-Play Fine-Tuning for Natural Language to Spreadsheet Formula Generation 

**Authors**: Cy Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.19354)  

**Abstract**: Spreadsheet applications are used by hundreds of millions worldwide, yet writing formulas remains a significant barrier. Existing approaches rely on static supervised data, which quickly saturates on limited annotations. In this paper, we introduce FORMULASPIN, a self-play framework that breaks the ceiling of supervised fine-tuning by enabling iterative self-improvement without any additional data. Vanilla SPIN fails on this task: it uniformly penalizes every non-matching output, so execution-equivalent alternatives are punished as negatives in one example while serving as ground truth in another, producing contradictory gradients. Our framework resolves this by exploiting formula generation's unique advantage: binary executability provides implicit supervision that separates semantic errors from valid stylistic variants. We frame training as a two-player game in which the main player learns to prefer ground-truth formulas over those from its previous version, while execution feedback sorts outputs into distinct granularities-enabling an adaptive curriculum that shifts from semantic correctness to stylistic refinement. To further increase accuracy, we incorporate ExecVote, a semantic-level voting mechanism that naturally handles multiple valid formulations. Experiments on multiple benchmarks demonstrate that FORMULASPIN achieves state-of-the-art performance, with 74.9% exact match and 87.1% execution accuracy on NL2FORMULA, matching models trained with additional preference annotations while outperforming both traditional SFT and frontier proprietary models. These findings underscore self-play's potential to tackle scarce data tasks and open the door to extending it beyond executable domains. 

---
# Benchmarking Confidential GPU Inference on NVIDIA H100 under Intel TDX 

**Authors**: Wei Wang, Abdul Hyee Waqas, Burns Smith  

**Link**: [PDF](https://arxiv.org/pdf/2607.19353)  

**Abstract**: Confidential computing is becoming a practical deployment requirement for AI inference workloads that process sensitive inputs or protect proprietary model assets. However, the performance cost of enabling confidential execution for GPU-accelerated large language model serving remains workload dependent and operationally important. This paper presents a benchmark study comparing standard non-confidential execution with confidential computing mode on a single NVIDIA H100 80GB GPU hosted in an Intel TDX confidential instance. The evaluation uses two representative language models, Mistral-7B v0.1 and Qwen3-30B-A3B, and measures time to first token, end-to-end request latency, per-request token generation throughput, global token throughput, and closed-loop request throughput under increasing concurrency. In fixed request-rate experiments, confidential mode increases average TTFT by 21.8% for Mistral-7B and 27.8% for Qwen3-30B-A3B, while global token throughput drops by 17.7% and 21.1%, respectively. In closed-loop concurrency experiments, throughput gaps remain in the 11.5-20.2% range, but the larger model reaches its saturation knee earlier under confidential mode. The results suggest that confidential GPU inference can retain usable throughput under load, but capacity planning must account for both the steady throughput penalty and the earlier saturation behavior observed for larger models. 

---
# OpenEvoShield: Dual Non-Stationary Continual Defense for Open-World Multi-Agent System Attacks 

**Authors**: Litian Zhang, Chaozhuo Li, Yuting Zhang, Zejian Chen, Bingyu Yan, Qiwei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2607.19351)  

**Abstract**: LLM-based multi-agent systems (LLM-MAS) are increasingly deployed in safety-critical applications, where adversaries inject malicious instructions through inter-agent communication to propagate harmful behaviors. Unlike static threats, these attacks are doubly dynamic: adversaries refine injection strategies against deployed defenses while normal-agent behavior drifts with system expansion. Existing defenses treat deployment as a closed-world problem and degrade rapidly once either distribution shifts beyond training coverage. We propose OpenEvoShield, a co-evolutionary continual defense framework for LLM-MAS. An asymmetric rate controller (M1) decouples fast attack-side and slow normal-side learning rates from dual drift signals. A normal-boundary updater (M2) maintains a dynamic behavioral boundary at the slow rate, while an EWC-regularized policy ensemble (M3) fast-adapts without catastrophic forgetting. An energy-based multi-granularity detector (M4) fuses node-, subgraph-, and graph-level evidence to classify novel attacks as out-of-distribution. Experiments over 100 deployment rounds across five benchmarks and four MAS topologies show that OpenEvoShield outperforms static and continual baselines, detecting most previously unseen attacks while keeping false positive rates low. 

---
# Hybrid LSTM-Graph Neural Framework for Robust Financial Fraud Detection and Adversarial Resilience 

**Authors**: Mariam Zakaria Moussa Ali  

**Link**: [PDF](https://arxiv.org/pdf/2607.19350)  

**Abstract**: Financial institutions face significant challenges in detecting sophisticated money laundering patterns, such as smurfing and layering, due to extreme data imbalance (0.13% fraud rate) and evolving adversarial evasion tactics. This paper proposes FraudShield AI, a hybrid framework that integrates Long Short-Term Memory (LSTM) networks with hand-crafted Graph Topological Features to capture both temporal sequences and structural relational context. By engineering network-centric features including PageRank Centrality, In-Degree dynamics, and a custom Flow Ratio, the system shifts the detection paradigm from isolated transaction analysis to network-level forensics. A Focal Loss objective is used to address class imbalance, and a dynamic thresholding mechanism is introduced to improve resilience against low-value smurfing attacks. Experimental evaluation on the PaySim dataset shows that the proposed hybrid model substantially outperforms Logistic Regression and XGBoost baselines in Precision, Recall, and F1-Score, particularly on hard-to-detect micro-transaction fraud patterns. An ablation study confirms the complementary contribution of both the temporal and topological components. 

---
# FineServe: A Fine-Grained Dataset and Characterization of Global LLM Serving Workloads 

**Authors**: Tiancheng Zhang, Shaoyuan Huang, Mingyuan Wang, Yunfeng Zhao, Xiaofei Wang, Wenyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19349)  

**Abstract**: Large language models (LLMs) are increasingly deployed as always-on online services, making efficient LLM serving a critical systems challenge. Achieving low latency and high throughput under volatile demand requires deep understanding of real-world serving workloads, yet existing studies often rely on proxy traces or coarse-grained characterizations that fail to capture the heterogeneity of modern multi-model LLM platforms. We present FineServe, an in-the-wild, multi-model LLM serving workload dataset collected from a global commercial marketplace, enabling fine-grained characterization of real-world serving dynamics across heterogeneous models and tasks. Leveraging FineServe, we conduct a comprehensive analysis of arrival dynamics and token behavior, revealing fundamentally different fluctuation regimes across model architectures, scales and task intents. Building on these insights, we develop the FineServe workload generator, which composes fine-grained model-aware workloads into configurable mixtures tailored for benchmarking multi-model serving platforms. By exposing these fine-grained workload dynamics, FineServe provides a realistic foundation for evaluating routing, scheduling, and capacity-planning strategies in LLM serving systems. FineServe is available at this https URL. 

---
# Persian Pixel: A large-scale synthetic OCR dataset for Persian language 

**Authors**: Pouria Mahdi, Haq Nawaz Malik  

**Link**: [PDF](https://arxiv.org/pdf/2607.20385)  

**Abstract**: Optical Character Recognition (OCR) for Persian remains substantially less mature than for Latin-script languages despite Persian being spoken by more than 110 million people across multiple countries. This gap arises from two fundamental challenges: the intrinsic complexity of the Perso-Arabic writing system and the limited availability of large-scale, high-quality annotated datasets. Persian script exhibits obligatory cursive connectivity, context-dependent glyph shaping, extensive ligatures, diacritic placement, and stylistic variation across writing forms such as Naskh and Nastaliq, all of which significantly complicate text recognition. At the same time, the high cost and labor-intensive nature of manual annotation have created a persistent data bottleneck, limiting the development of robust OCR systems and slowing progress in Persian document this http URL this paper, we introduce Persian Pixel, a comprehensive synthetic OCR dataset specifically designed to address these challenges. Comprising over 343,000 high-fidelity image text pairs, the dataset spans sentence, paragraph, and full-page document layouts generated from a carefully curated seven-million-word Persian corpus using the SynthOCR-Gen rendering framework. The generation pipeline faithfully models the typographic characteristics of Persian script, including contextual character joining, positional glyph variants, diacritic placement, and multiple representative Persian typefaces. To bridge the synthetic-to-real domain gap, the rendered images are further enriched with more than twenty-five stochastic degradation models that emulate realistic document acquisition artifacts, including ink bleed, paper aging, blur, illumination variation, scanner imperfections, compression artifacts, and multiple noise this http URL overcoming the long-standing scarcity of annotated Persian OCR data, Persian Pixel provides a scalable and openly available resource for training and fine-tuning modern OCR architectures, including transformer-based models such as TrOCR and Donut. The dataset establishes a strong foundation for research in Persian document analysis, historical manuscript digitization, and end-to-end document understanding, while demonstrating that programmatic synthetic data generation offers a practical, cost-effective, and scalable alternative to manual annotation for advancing OCR in low-resource and typographically complex scripts. 

---
# FMRP-LEAN: A HIPAA-Compliant AI-Augmented LIMS Architecture for End-to-End Clinical Assay Workflow Optimization 

**Authors**: Eva McCord, Ernest Pedapati, Zag ElSayed  

**Link**: [PDF](https://arxiv.org/pdf/2607.20382)  

**Abstract**: Clinical biomarker workflows in translational research settings often rely on spreadsheet-driven tracking, manual quality control (QC) reconciliation, and loosely integrated systems, resulting in limited state visibility, delayed reporting, and increased operational risk. These challenges are particularly pronounced in multi-day assays such as Luminex-based quantification of Fragile X Messenger Ribonucleoprotein (FMRP), where HIPAA-compliant data governance, deterministic workflow progression, and coordinated communication across laboratory and clinical teams are required. This paper presents FMRP-LEAN, a HIPAA-compliant, AI-augmented Laboratory Information Management System (LIMS) architecture that formalizes biospecimen lifecycle management through a finite-state workflow model with explicit transition guards and dwell-time observability. The system integrates a self-hosted Supabase/PostgreSQL stack deployed within hospital-controlled infrastructure, hybrid edge-internal isolation with encrypted tunneling and loopback-only services, and bi-directional REDCap synchronization. A unified MRN-UUIDv7 identifier framework with QR-based tracking ensures traceable clinical-research linkage under PHI residency constraints. FMRP-LEAN incorporates automated statistical QC pre-screening and a governance-constrained AI operations module that operates exclusively on aggregate projections, with deterministic fallback guarantees. Deployment demonstrates improved workflow observability, reduced QC latency, and enhanced cross-role transparency between laboratory technicians, research coordinators, and patient-facing teams. The architecture provides a reproducible model for secure, state-explicit, and AI-augmented clinical research workflows in regulated healthcare environments. 

---
# Generative AI floods and dilutes the market for books 

**Authors**: Tuhin Chakrabarty, Xinyue Liu, Jane C. Ginsburg, Paramveer Dhillon  

**Link**: [PDF](https://arxiv.org/pdf/2607.20349)  

**Abstract**: Generative AI can produce book-length works of fiction at near-zero cost. These books are often dismissed as low-quality ``slop'' that buyers will ignore, and are assumed to carry little commercial weight. We test that assumption with full-text AI detection across 14,419 self-published genre-fiction books sold on Amazon from 2023 to 2026, matched to daily sales records through June 2026. None of these books disclose whether or not they contain AI-produced content. We find that books for which we detected substantial AI text ($>$ 25\%) make up a large share of the catalog but a smaller share of sales. Even so, they reach commercial scale, winning a growing share of sales over time and taking more of the scarce top-rank positions once held by books with no detected AI text. Over this period, the number of books with observed sales in a quarter grew 19.2-fold, while quarterly revenue grew only 8.9-fold. The market therefore added selling books faster than it added revenue, and revenue per selling book fell across most genres. Books with no AI text lose the most ground in genres with high AI diffusion, and most of all where Kindle Unlimited availability is high. Among top-selling books, those with substantial AI text draw on more distinctive language from existing books than do books with no AI text; for these books overlap rises with revenue, a gradient we do not detect for books with no AI text. Generative AI can thus reshape a creative market through scale rather than quality. Our results bear directly on the market-effect question at the center of the fair use defense to copyright infringement. 

---
# Closing the Lab-to-Store Gap: A Data-Efficient Post-Training and Experience-Driven Learning VLA Framework for Retail Humanoids 

**Authors**: Roger Sala Sisó, Tiago Silvério, Jakob Sand, Tran Nguyen Le  

**Link**: [PDF](https://arxiv.org/pdf/2607.20345)  

**Abstract**: Closing the gap between benchmark performance and reliable real-world operation remains a central challenge for Vision-Language-Action (VLA) humanoid robots, which must handle execution errors, distribution shifts, and environmental variability. This paper presents DEED (Data-Efficient Post-Training and Experience-Driven Learning), a systems-level approach evaluated on a supermarket chip-restocking task using a Unitree G1-Edu humanoid robot and the GR00T N1.6 foundation model. DEED comprises three key components: (1) a data-efficient post-training pipeline with control-frequency alignment, data curation, task-relevant visual highlighting, and reduced VLA dependence; (2) a real-world study of experience-driven refinement, adapted from RECAP via a text-based advantage prefix and a vision-language value function; and (3) a latent-space analysis tool for studying in- and out-of-distribution behavior. Our results suggest that bridging the lab-to-store gap is primarily a systems integration challenge rather than an architectural one: careful data design and targeted post-training can transform a policy that fails under naive fine-tuning into a competent real-world system using only a single GPU. 

---
# Understanding Generative AI-mediated User Engagement with Academic Library Resources 

**Authors**: Hae Min Kim, Stacy Stanislaw  

**Link**: [PDF](https://arxiv.org/pdf/2607.20328)  

**Abstract**: This study empirically analyzed generative AI as an emerging discovery pathway to academic library resources. Utilizing web analytics from August 2023 to October 2025, the research identifies a significant increase in AI-mediated traffic, particularly following the integration of linked citation features. Referral analysis identified ChatGPT, Perplexity, and Gemini as the primary platforms driving this traffic. A substantial portion of users reached the institutional repository, primarily accessing electronic theses and dissertations. This pattern suggests that AI retrieval mechanisms effectively surface resources with structured metadata and stable permalinks that are Open Access and freely available. The results illustrate how AI ecosystems currently expose library resources and underscore the need for continued analysis and a strategic response to the evolving AI landscape. 

---
# Toward Reliable RGB-D Semantic Segmentation: Handling Missing Modalities via Condition Dropout 

**Authors**: Xuchen Zhu, Yajuan Wei, Shuang Hao, Jiwei Jiang, Guanxiang Mao, Fang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2607.20326)  

**Abstract**: RGB-D semantic segmentation has achieved remarkable progress, yet most models assume that RGB and depth are always available. In practice, failures or occlusions of surveillance sensors often remove one modality. Although RGB or depth alone can contain sufficient cues, models trained only on full-modality inputs fail to exploit the remaining modality once one is missing, causing severe degradation. We tackle this issue with a simple continued-training paradigm, \emph{Condition Dropout (ConD)}, which mitigates degradation while preserving full-modality accuracy. Starting from a pretrained RGB-D model, ConD adds a second stage that randomly simulates complete, RGB-missing, and depth-missing inputs, freezes the original encoders, and trains copied encoders with zero-initialized feature injection. Experiments on NYU-Depth V2 and SUN RGB-D show that ConD improves robustness under missing modalities and even yields slight gains when modalities are complete. Our code will be made publicly available upon acceptance. 

---
# Don't Trust the Label: License Laundering in AI Supply Chains 

**Authors**: James Jewitt, Hao Li, Gopi Krishnan Rajbahadur, Bram Adams, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2607.20300)  

**Abstract**: AI artifacts move through a multi-platform supply chain, spanning datasets and models on Hugging Face and applications on GitHub. While each artifact carries a license whose obligations should propagate through redistribution, no study has yet measured whether those obligations survive the chain or are stripped and replaced as artifacts move downstream. We trace 232,270 dataset$\rightarrow$model$\rightarrow$application chains and quantify two forms of license laundering: when artifacts with no declared license acquire definitive labels downstream, and when one declared license category replaces another during redistribution. We find that 62.3% of chains pass through at least one artifact with no declared license (concentrated in a small set of foundational datasets), and that every obligation-bearing license category falls below 7% end-to-end survival while the Permissive category reaches 95.1%. Based on these findings, we provide actionable recommendations for practitioners, model publishers, rights holders, and platform owners. 

---
# Courteous Anticipation: Improving Long-Lived Task Planning in Persistent Shared Environments 

**Authors**: Md Ridwan Hossain Talukder, Roshan Dhakal, Elizabeth Phillips, Gregory J. Stein  

**Link**: [PDF](https://arxiv.org/pdf/2607.20289)  

**Abstract**: We consider a task planning scenario in which robots sharing a persistent environment are assigned tasks one at a time from a held-out sequence. Standard task planners, lacking foresight of future tasks and inconsiderate of others' constraints, solve each task in isolation, leaving terminal states that increase future cost for all, side effects that compound over lengthy task sequences. To reduce cost over the sequence, a robot must anticipate how its actions now may impact performance on future tasks for all robots sharing the environment. Therefore, we present courteous anticipatory planning, wherein a model-based planner proposes candidate plans and selects the one that jointly minimizes immediate cost and aggregated expected future cost across all robots, estimated via independent per-robot learned estimators. This factored formulation avoids combinatorial joint rollouts and supports modular deployment: adding a robot requires only training its own estimator. We evaluate in two persistent PDDL domains, a home environment with robots that have similar capabilities but different responsibilities, and a restaurant environment where robots' distinct capabilities create states that other robots lack the capability to resolve. During lengthy task sequences, our planner reduces total cost by 10.43% versus myopic and 4.03% versus selfish anticipatory planning in a two-robot home environment and by 17.41% and 13.24%, respectively, in a three-robot restaurant. 

---
# Sound Probabilistic Safety Bounds for Large Language Models 

**Authors**: Mahdi Nazeri, Anne-Kathrin Schmuck, Sadegh Soudjani, Alessandro Abate  

**Link**: [PDF](https://arxiv.org/pdf/2607.20286)  

**Abstract**: We propose a novel framework for computing rigorous bounds on the probability that a large language model (LLM) generates harmful output to a given prompt. We study a new application of the Clopper-Pearson confidence intervals to obtain probably approximately correct (PAC) bounds for this problem. As our main technical contribution, we propose an algorithm that leverages features in the latent space to prioritize exploring branches in the auto-regressive generation tree that are more likely to produce harmful outputs. Our approach in particular enables the efficient computation of useful lower bounds, even in scenarios where the true harm probability is extremely small, and crucially, the obtained lower bounds are sound, i.e., formally proven to be less than the actual harmfulness probability: our experimental results demonstrate the effectiveness of our method by computing non-trivial lower bounds on state-of-the-art LLMs. This study newly enables the evaluation and statistical certification of LLMs. 

---
# Self-supervision drives representational convergence in medical foundation models more than clinical supervision 

**Authors**: Soroosh Tayebi Arasteh, Sebastian Ziegelmayer, Mahshad Lotfinia, Lisa Adams, Sven Nebelung, Jakob Nikolas Kather, Daniel Truhn  

**Link**: [PDF](https://arxiv.org/pdf/2607.20274)  

**Abstract**: Medical image encoders from different groups are increasingly treated as interchangeable, on the assumption that scale and clinical supervision concentrate their representations onto a shared structure. Whether this convergence is real, what produces it, and whether it is clinically usable are untested, and the similarity measures behind such claims are fragile. We present a controlled dissection across 18 image and 7 text encoders, all open-weight and run locally, spanning 7M to 27B parameters and five imaging modalities, including 650,982 chest radiographs from six datasets. To isolate cause, we train encoders that vary only the objective under fixed data, architecture, and scale, and reproduce the effect in a synthetic model. Convergence is modest but above a random floor, driven by the self-supervised objective, not clinical supervision: matched self-supervised encoders aligned most (40.4% on chest radiography), with label-supervised (21.1%) and image-text (3.3%) far lower, and did not grow with size (Spearman 0.302, p=0.223) or capability. It is within-modality, does not reach clinical language, and does not reproduce how radiologists judge case similarity. Yet a linear classifier transfers across encoders and to five held-out hospitals, retaining about 85% of within-encoder performance. Convergence in medical imaging is therefore set by the pretraining objective, not inherited from scale or clinical supervision. Interoperability is accordingly something to design for through that objective, and to validate where the shared geometry is weakest, across patient subgroups and against clinical judgment. 

---
# The Maskability Index: Predicting Task-Objective Alignment in Pretrained Language Models 

**Authors**: Ahmad Pouramini, Mahsa Afsharzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2607.20265)  

**Abstract**: Large-scale pretrained language models such as T5 and BERT have demonstrated strong capabilities for generating structured knowledge. However, their performance depends on how closely the prompting strategy matches the objectives used during pretraining. We introduce the Maskability Index (MI), a quantitative metric that estimates whether a knowledge relation is better suited to masked-style prompting or prefix-style prompting in few-shot generation. MI is computed from differences in DepthRank scores between masked and unmasked templates, providing a principled measure of objective-template alignment. We evaluate MI on a diverse set of relations from the ATOMIC2020 knowledge base completion benchmark and show that it is positively correlated with downstream generation performance. These results indicate that MI can help select appropriate prompting templates and adaptation strategies for extracting relational knowledge from pretrained language models, especially in low-resource settings. 

---
# The Ethics of Autonomous AI Agents for Offensive Security 

**Authors**: Andreas Happe, Jürgen Cito, Jasmin Wachter  

**Link**: [PDF](https://arxiv.org/pdf/2607.20255)  

**Abstract**: LLM-driven autonomous agents are reshaping offensive security. Unlike traditional penetration-testing tooling -- deterministic, narrowly scoped, and operated by trained practitioners -- agentic security tools exhibit \textit{indeterminacy} along three independent dimensions. First, their actions are drawn from a non-deterministic policy whose outputs resist both ex-ante and ex-post explanation, frustrating incident attribution and pre-deployment safety review. Second, their impact is open-ended due to the non-deterministic actions, agency of utilized models, and opaque LLM supply-chains. Third, their user population is indeterminate in both size and required skill: the operating skill floor for using or developing offensive capabilities has dropped sharply. These three properties are linked thematically, but are not derivable from one another. Combined with the structural cost asymmetry between offense and defense, they enable the industrialization of offensive capability. The net short-term effect favors attackers, even if the same technology may, in the long run, democratize access to defensive practice. Existing dual-use cybersecurity and AI-ethics frameworks were not designed for this combination. Our work analyzes how moral attribution becomes diffuse between users, tool-makers, and third parties when employing autonomous AI agents for offensive security. We also examine the stakeholder impact of this technology and provide stratified recommendations. 

---
# Pushing the Frontier of Full-Song Generation: Hierarchical Autoregressive Planning Meets Flow-Matching Rendering 

**Authors**: Junyu Dai, Xinyue Fan, Weiqin Li, Xiangang Li, Yunjia Li, Bin Ma, Yukun Ma, Chongjia Ni, Yufei Shi, Haoxu Wang, Menglin Wu, Jianwei Yu, Huaicheng Zhang, Han Zhao, Shengkui Zhao, Haina Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20253)  

**Abstract**: In this report, we present a unified song generation framework capable of producing high-quality full-length music from lyrics, text descriptions, and musical attributes. The proposed framework supports three tasks: Lyrics-to-Song Generation, which generates complete songs from text descriptions, lyrics, and musical attributes; Instrumental Music Generation, which creates music without vocals; and Cover Song Generation, which reinterprets existing songs with different styles while preserving their melodic content. Architecturally, our system consists of four main components: a semantic-aware tokenizer, hybird-LM, FullDiT, and a two-level melody module. The tokenizer encodes audio into 8-codebook RVQ tokens for efficient discrete music representation. Based on these tokens, hybird-LM performs hierarchical autoregressive audio-token modeling for full-song generation. To improve audio fidelity, FullDiT performs full-song flow matching in a continuous VAE latent space conditioned on codec tokens, lyrics, and text captions. For cover song generation, the melody module extracts and discretizes melody cues from reference audio to guide generation while preserving the original melodic content. Finally, we investigate DPO, GRPO, and OPD as reward-based post-training strategies for hybird-LM and apply flow-based GRPO to FullDiT to improve musicality and rendering quality. Experimental results on a multilingual automatic benchmark, complemented by the Artificial Analysis Music with Vocals leaderboard, show that the proposed framework achieves competitive performance in the evaluated settings. 

---
# On the Systematic Challenges of Culturally Loaded Machine Translation: Dream of the Red Chamber as the Cultural Lens 

**Authors**: Yiming Wang, Jiayuan Di  

**Link**: [PDF](https://arxiv.org/pdf/2607.20241)  

**Abstract**: Culturally loaded translation poses unique challenges for machine translation (MT), as meanings are deeply embedded in socio-cultural contexts beyond surface linguistic forms. Although large language models (LLMs) have enabled MT systems to achieve human-like quality in many scenarios, their ability to handle culturally loaded expressions remains underexplored. In this study, we systematically investigate the challenges posed by culturally loaded translation in LLM-based MT systems. We construct a Chinese-Japanese bilingual dataset from the culturally representative corpus Dream of the Red Chamber, containing 500 segments across diverse cultural categories. Using a comprehensive evaluation protocol, we reveal three main challenges: (1) task challenges, where frontier LLMs exhibit notable performance gaps and struggle with culturally loaded content; (2) human evaluation challenges, where evaluator backgrounds lead to substantial disagreement in translation judgments; and (3) automatic evaluation challenges, where widely used metrics fail to reliably assess translation quality for this task. These findings may offer valuable insights for culture-oriented translation research in both computational science and linguistics. 

---
# DQAOA-GPT: AI-Accelerated Distributed Quantum Optimization for Combinatorial Problems 

**Authors**: Seongmin Kim, Abhinav Rijal, Yuri Alexeev, Nora Bauer, Martin Roetteler, Mina Yoon, George Siopsis, In-Saeng Suh  

**Link**: [PDF](https://arxiv.org/pdf/2607.20225)  

**Abstract**: While combinatorial optimization problems are central to many scientific and engineering applications, their solution remains challenging due to exponentially large search spaces. Variational quantum algorithms offer a promising route for tackling such problems, yet their practical performance is limited by repeated quantum circuit evaluations and classical parameter updates. In this work, we introduce DQAOA-GPT, a hybrid framework that integrates the distributed quantum approximate optimization algorithm (DQAOA), which decomposes a large optimization problem into smaller sub-problems, with GPT-based quantum circuit generation for solving those sub-problems. Rather than relying on iterative variational optimization, the proposed approach uses a trained generative model to directly generate high-quality quantum circuits for the decomposed sub-problems. As a benchmark, we evaluate DQAOA-GPT against conventional DQAOA on dense HUBO optimization problems with up to 100 decision variables. The results demonstrate that DQAOA-GPT significantly reduces computational cost while maintaining competitive solution quality, with larger acceleration observed for larger sub-problem sizes. Although this work focuses on benchmark-scale validation, the framework provides a promising foundation for larger-scale combinatorial optimization in hybrid HPC-QC environments through increased GPU resources and parallel computing capability. 

---
# Small, Free, and Effective: Orchestrating Open-Weight Small Language Models to Outperform Single LLM for Malware Analysis 

**Authors**: Adel ElZemity, Shujun Li, Budi Arief  

**Link**: [PDF](https://arxiv.org/pdf/2607.20216)  

**Abstract**: Malware analysis demands rapid interpretation of complex detonation reports spanning filesystem, network, and process behaviours. While large language models (LLMs) demonstrate impressive capabilities for technical artifact interpretation, the opacity and escalating API costs of closed-weight frontier models motivate exploration of open-weight alternatives. However, many open-weight models are large, demanding significant compute resources and incurring non-trivial hosting costs that place them beyond reach for resource-constrained deployments. This paper investigates whether orchestrated ensembles of small language models (SLMs) can match or exceed single LLM performance on structured questions about malware detonation reports. We established baselines by testing eleven open-weight SLMs, three cyber security pre-trained models, and six frontier LLMs on Meta's CyberSecEval Malware Analysis benchmark. We then designed and evaluated four orchestration architectures: (i) a multi-agent pipeline that decomposes analysis into structured evidence-collection and reasoning stages, (ii) an adversarial debate framework in which two agents iteratively critique each other's reasoning, (iii) a hierarchical consultation system that pairs a general-purpose SLM with a cyber-specialised expert model, and (iv) a hybrid architecture that combines evidence-grounded pipelines with adversarial debate reasoning. The hybrid system (Qwen3-4B with Foundation-Sec-8B) achieved 35.30% overall accuracy, exceeding the strongest cyber-specialised baseline (22.54%) and the strongest ungrounded frontier baseline (34.77%); when given the same evidence pipeline, grounded Gemini remained the strongest configuration at 38.22%. These findings show that evidence-grounded orchestration can substantially improve the performance of collaborative SLMs for supporting interpretation of malware detonation reports. 

---
# ELSAA: Efficient Low-Rank and Sparse Attention Approximation for Training Transformers 

**Authors**: Mahdi Heidari, Mohammad Mahdi Rahimi, Jaekyun Moon  

**Link**: [PDF](https://arxiv.org/pdf/2607.20214)  

**Abstract**: The quadratic $N\times N$ attention score matrix remains a central obstacle to extending Transformers to longer input lengths. Existing efficient attention methods usually reduce this bottleneck by either imposing sparsity, so that each query attends to only a small subset of keys, or by using low-rank/kernel sketches, so that global interactions are compressed into a lower-dimensional representation. We propose \emph{ELSAA}, an efficient low-rank and sparse approximation of attention. Importantly, ELSAA does \emph{not} decompose the learned projection or output matrices of the Transformer into sparse and low-rank factors. Instead, after dense projections produce $Q,K,V$, ELSAA approximates the induced attention score operator itself: a sparse branch captures selected high-similarity interactions, while a low-rank branch summarizes diffuse global interactions. Since the two branches can be normalized over supports with very different denominator mass, ELSAA introduces a denominator-aware fusion term that scales the sparse branch according to its estimated attention mass relative to the low-rank branch. This gives a practical framework for constructing low-rank and sparse attention outputs without materializing the full quadratic score matrix, aiming to enable longer-context training while preserving both sharp token-level interactions and broad contextual mixing. 

---
# The Quadrilateral Loss: Additivity as a Measurable Behavior of Dense Neural Networks 

**Authors**: Antonio Di Cecco  

**Link**: [PDF](https://arxiv.org/pdf/2607.20201)  

**Abstract**: Additive models buy interpretability by forbidding feature interactions, a constraint that neural instantiations enforce architecturally. We introduce the quadrilateral loss, a differentiable penalty that treats additivity as a measurable behavior instead: a second-order mixed difference on pairs of training points swapping one coordinate, which vanishes if and only if the coordinate carries no interaction, remains informative for piecewise-linear networks, and equals in expectation the per-coordinate interaction mass of the interventional Shapley-GAM. The loss turns additivity into a dial - most learned interactions prove removable almost for free, and on small datasets a moderate penalty improves accuracy and additivity simultaneously - and into an online observable: its per-feature surrender curves show, across seeds and datasets, that pre-regularization interaction magnitude barely predicts what a regularized model retains, undermining post-hoc interaction rankings. Against this instrument we compare routes to exact additivity, spanning structural masks, behavioral penalties (optionally crystallized into exact structure), weight decay, backfitting, the shared-section model, and bagged boosted stumps: constraining behavior before structure dominates weight-space constraints, rankings reverse between data regimes, and converging routes agree on the shape functions themselves. Three silent failure modes we document share one anatomy: guarantees imported into settings that quietly void their preconditions. 

---
# StreamHOI: Interaction-aware Temporal Memory Adaptation for Streaming HOI Video Generation 

**Authors**: Zejing Rao, Haoxian Zhang, Xiaoqiang Liu, Yiping Meng, Guoxin Zhang, Pengfei Wan, Fan Tang, Tong-Yee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.20174)  

**Abstract**: Existing human--object interaction (HOI) video generation methods are largely limited to offline short-video generation with complex driving conditions, making them unsuitable for real-time interactive applications. We present \emph{StreamHOI}, a low-latency streaming framework for long-duration HOI video generation. Instead of converting heavily conditioned HOI pipelines into streaming systems, we study how an image-to-video streaming generator should organize historical memory to preserve interactions under bounded latency. We find that the standard sink-local memory design faces a trade-off in streaming HOI generation, and different transformer blocks show different historical-memory preferences for HOI regions and surrounding regions. To match memory composition with block behavior, StreamHOI performs offline HOI-aware block profiling and applies bias-guided memory-specialized training to adapt the generator to block-specific memory layouts. We further introduce a memory distance scaling module to strengthen long-range access to early interaction states. Extensive comparisons with both long-video baselines and recent HOI generation methods demonstrate that StreamHOI achieves strong interaction plausibility, object fidelity, human quality and efficiency, reaching 17.6 FPS with 0.75s first-chunk latency. 

---
# Audio-Zero: Label-Free Self-Evolution for Fine-Grained Audio Reasoning 

**Authors**: Siqian Tong, Xuan Li, Chaozhuo Li, Baolong Bi, Yiwei Wang, Yujun Cai, Shenghua Liu, Chengpeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20166)  

**Abstract**: Large Audio Language models (LALMs) have made rapid progress on acoustic understanding, yet they still struggle with fine-grained audio reasoning (e.g., recognizing event order, repetitions and duration). Existing post-training methods heavily rely on expensive external labels or provide only coarse semantic signals. To bridge this gap, we introduce Audio-Zero, the first label-free self-evolution framework in the field of LALMs that improves fine-grained auditory perception and reasoning. Audio-Zero constructs an auditory self-play game from unlabeled audio contrast pairs: most players hear a reference audio, while one odd listener hears a subtle variant. The model first generates clues describing what it hears and then identifies the odd listener by reasoning over inconsistencies among clues. Since the odd listener is known by construction, the game provides verifiable rewards without any annotated answers. Experiments with Qwen2-Audio-7B-Instruct and Qwen2.5-Omni-7B on TREA, MMAU Test-mini and MMAR show that Audio-Zero improves fine-grained audio reasoning while preserving broad audio understanding. Evolutionary and diagnostic analyses further reveal that increasingly fine-grained auditory descriptions emerge naturally from game pressure. 

---
# Active Inference as a Convex Markov Decision Process 

**Authors**: Nikola Milosevic, Nicolás Hinrichs, Nico Scherf  

**Link**: [PDF](https://arxiv.org/pdf/2607.20152)  

**Abstract**: Active Inference (AIF) frames adaptive behavior as the minimization of expected free energy (EFE), combining epistemic and pragmatic objectives within a single variational principle. We frame AIF as policy optimization and show that, for closed-loop control policies, EFE minimization can be formulated as a convex Markov decision process (MDP). In this formulation, the pragmatic terms are linear in the predictive state marginals and therefore equivalent to reward maximization in a latent MDP, while the epistemic value introduces a nonlinear component that distinguishes EFE minimization from standard reinforcement learning. This perspective further reveals the epistemic drive of active inference as a policy-dependent (performative) reward. We analyze finite-horizon, discounted, and average-reward formulations of EFE and derive a mirror descent (MD) algorithm that locally linearizes the objective around the current state marginals, yielding a policy-dependent reward that is compatible with actor-critic methods and dynamic programming. Finally, we argue that coupling world-model learning with policy optimization gives active inference the structure of performative reinforcement learning, providing a route toward grounding active inference within modern reinforcement learning and optimization theory, including convergence analysis and principled policy improvement guarantees. 

---
# SLAI T-Rex: Full-Parameter Post-training of the DeepSeek-V4 Family on Ascend SuperPOD 

**Authors**: Dongfang Li, Xiaodong Luo, Ruoyu Sun, Xuhui Chen, Linyuan Qiu, Jian Meng, Zhengxuan Lu, Yiting Wang, Yucheng Xie, Tao Guo, Tianxiang Fang, Jing Li, Sihang Chen, Shihao Hong, Chang Liu, Weihua Dai, Zirong Zeng, Ziwei Zhu, Zhuohan Wang, Zhengjun Yue, Igor Vasilyev, Min Liu, Weijian Sun, Xin Chen, Yingmeng Gao, Jinhua Zhou, Taolue Chen, Chenwei Wu, Dong Zhang, Wenlong Jin, Jinmin Xiang, Barkova Maria, Ushakov Anton, Xianfei Jin, Tian Ding, Zhihang Lin, Qian Chen, Linxin Yang, Mingzhe Yang, Bingwei Zhang, Hongzhang Yang, Fangxue Zhang, Shijun Qin, Jie Yu, Cuihua Hu, Tolstykh Vasiliy, Nosov Ivan, Abdullin Amir, Zhichen Zhou, Xin Zhang, Zhixiong Ning, Xutong Zhao, Junjie Huang, Jiajun Liu, Weiyan Kong, Zheng Zhang, Wenhan Luo, Lin Hu, Yangbo Guo, Li Zeng, Shihao Zeng, Baotian Hu, Min Zhang, Haizhou Li, Zhiquan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.20145)  

**Abstract**: Full-parameter post-training of trillion-parameter-scale MoE models introduces substantial system-level challenges for large-scale distributed training, including severe memory pressure, non-overlapped communication overhead, and inefficient kernel execution. While most large-scale LLM training systems are built around GPU-based clusters, this report presents an end-to-end optimization practice on the Ascend NPU SuperPOD. Using the DeepSeek-V4 model family as the target workload, we develop a hierarchical optimization framework spanning model-level parallelism, computation-communication orchestration, and low-level kernel execution. The resulting system achieves 34.22% Model FLOPs Utilization (MFU) with a 2.93x improvement over the open-source baseline recipe while maintaining training stability. Building on this optimized infrastructure, we further establish a CPT and SFT workflow for complex Operations Research (OR) tasks. We refer to the integrated framework as SLAI T-Rex. Using DeepSeek-V4-Flash, we develop OR-oriented CPT and SFT data pipelines that combine collected domain resources with solver-verified synthetic optimization documents. The resulting dataset contains 10K high-quality SFT samples spanning four task categories and three problem representations. The specialized model achieves the highest average zero-shot Pass@1 score among the evaluated models, reaching 71.81% and outperforming GPT-5.4-Mini and the base DeepSeek-V4-Flash model by 3.98 and 11.27 percentage points, respectively. Overall, this work demonstrates a full-stack pathway from efficient trillion-parameter model post-training on Ascend infra to domain-specialized Flash models for solver-grounded mathematical modeling, advancing frontier-model systems for complex reasoning. 

---
# Formal Foundations for Known Good Reliable Die Screening in Chiplet-Based AI Systems-on-Chip 

**Authors**: Prashanthi Metku, Chandra Gandu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20141)  

**Abstract**: The rapid growth of chiplet-based artificial intelligence systems-on-chip (SoCs) has exposed a fundamental gap in semiconductor test methodology. Existing Known Good Die (KGD) screening guarantees pre-assembly functional correctness, yet it offers no probabilistic assurance of post-assembly reliability lifetime. To address this limitation, the present work formalizes the transition from KGD to Known Good Reliable Die (KGRD) screening as a constrained inference problem over incomplete pre-assembly observability. Building upon this formulation, four interlocking contributions are presented: (i) a Bayesian probabilistic risk model that maps pre-assembly telemetry to post-assembly failure likelihood with a quantified observability bias bound; (ii) a safety-gated decision architecture that provides a provable post-assembly failure probability guarantee; (iii) uncertainty-aware disposition boundaries derived from Bayes-optimal decision theory; and (iv) a constrained closed-loop feedback mechanism that delivers consistent model improvement without violating reliability constraints. A Monte Carlo simulation study on N = 4,000 synthetic dies verifies all four theoretical properties and confirms that the safety guarantee holds uniformly across the full range of tested gate threshold. 

---
# PRIME-SVR: Physics-infoRmed Implicit Multi-Echo Slice-to-Volume Reconstruction for Fetal T2 mapping 

**Authors**: Busra Bulut, Maik Dannecker, Thomas Sanchez, Sara Neves Silva, Steven Jia, Jean-Baptiste Ledoux, Leo Pomar, Joanna Sichitiu, Yvan Gomez, Meriam Koob, Vincent Dunet, Maria Deprez, Guillaume Auzias, Francois Rousseau, Jana Hutter, Daniel Rueckert, Meritxell Bach Cuadra  

**Link**: [PDF](https://arxiv.org/pdf/2607.20136)  

**Abstract**: Slice-to-volume reconstruction (SVR) is the standard method for obtaining high-resolution (HR) 3D fetal brain volumes from motion-corrupted 2D MRI slice stacks acquired in multiple orientations. Existing SVR methods are optimized and validated only for clinical-range echo times (TEs), limiting their use at non-clinical TEs and making them incompatible with quantitative T2 mapping, a protocol- and center-independent biomarker of fetal brain maturation requiring HR reconstructions across multiple TEs. We present PRIME-SVR, the first implicit neural representation (INR) framework for joint HR reconstruction from multi-echo MRI. A single fully connected network models a continuous function from spatial coordinates to signal intensities across TEs, while a second network estimates slice-specific acquisition degradations. Cross-TE coherence is enforced via a Bloch equation-derived regularization penalizing deviations from expected T2 decay, with adaptive weighting that strengthens coupling for degraded stacks. The method is fully self-supervised. We validate PRIME-SVR on 39 in vivo fetal acquisitions (13 subjects x 3 TEs) from two centers, two vendors, and two field strengths (1.5 T and 0.55 T). Compared to state-of-the-art SVR, PRIME-SVR improves reconstruction sharpness by 47%, anatomical accuracy by 30%, and cross-TE structural consistency by 14%. It enables reconstruction at late TEs previously inaccessible to SVR, yielding the first 0.8 mm isotropic T2 maps at 0.55 T and the first T2 maps derived from INR-based SVR. PRIME-SVR also accelerates quantitative imaging by reducing the data needed for multi-TE reconstruction, cutting acquisition from 15 to 10 minutes while keeping T2 accuracy within 1.7% in white and deep gray matter, or to 5 minutes with a mean T2 error of 2.3% for high-quality acquisitions. 

---
# ENTRAP-VL: A Taxonomic Probe for Dual Contextual Entrainment in Vision-Language Models 

**Authors**: Karan Goyal, Afreen Hossain, Debojyoti Das, Vishal Bhutani  

**Link**: [PDF](https://arxiv.org/pdf/2607.20092)  

**Abstract**: Contextual entrainment is the tendency of a model to let auxiliary context in its input pull its output, independently of whether that context is relevant, true, or even meaningful. Recently, it has been identified and given a mechanistic account in unimodal language models. Whether and how it manifests in vision-language models (VLMs) is, by contrast, largely unexamined, and the field lacks a purpose-built instrument with which to investigate it. We take the position that studying contextual entrainment in VLMs requires more than porting an existing text-only benchmark to the multimodal setting: it requires a taxonomically structured, dual-modality instrument whose conditions are constructed around the item at hand (the depicted image in the textual stream, the textual query in the visual stream). We argue that the move to VLMs is substantive rather than incremental. It makes entrainment a dual phenomenon, drivable independently by textual and by visual context, and it opens a veracity distinction (context that is false of the depicted scene yet possible in the world) that has no counterpart in the unimodal, world-knowledge-only formulation of prior work. To make this position concrete and actionable, we introduce ENTRAP-VL (ENTRainment Assessment Probe for Vision and Language), a manually curated dataset of 1,500 items across eight categories, organized by a taxonomy that spans two axes, i.e., the association of context with the item and its relationship to truth, and split into a textual-entrainment stream (eight context conditions) and a visual-entrainment stream (three context conditions). We do not claim to measure entrainment in any particular model; we provide the instrument, the taxonomy that motivates it, and the evaluation protocols it enables, so that the community can investigate the phenomenon rigorously. We will release the dataset and its documentation publicly. 

---
# Reinforcement Learning for Large Language Model Selective Evidence Adoption from Contaminated Retrieval Results 

**Authors**: Yanyu Chen, Yue Li, Yongyi Cui, Dongsheng Shi, Lichang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2607.20090)  

**Abstract**: Retrieval-augmented large language models frequently face contexts that interleave useful evidence with misleading statements or instruction-like content. Blanket refusal discards valid evidence, whereas uncritical adoption yields incorrect or unsafe answers. The ability to selectively adopt relevant information while rejecting deceptive or harmful content is therefore critical for reliable deployment in real-world retrieval settings. We introduce SelectBench, a controlled benchmark and training set for selective evidence adoption, and post-train Qwen3.5-4B directly with DAPO using either deterministic rule rewards or a frozen semantic judge. On the corrected 325-example SelectBench-v2 test set, strict success rises from 22.46% for the original checkpoint to 25.54% with DAPO-Rule and 26.46% with DAPO-DeepSeek. Both trained policies reduce forbidden-content adoption and produce shorter, more focused responses, yet prompt-injection following does not improve. The paired gains are modest and fail to survive Holm correction, suggesting that stronger reward shaping or additional training iterations may be needed for more robust gains. DAPO-DeepSeek exhibits no material degradation on MMLU or clean HotpotQA, indicating that the post-training procedure preserves general capabilities. These results demonstrate a directional improvement in selective evidence use, while identifying injection resistance and statistical robustness as important remaining challenges for future work. 

---
# Co-Evolving LLM Evaluators and Policies via DynamicRubric 

**Authors**: Beining Wang, Weihang Su, Hongtao Tian, Hao Kong, Tao Yang, Ting Yao, Qingyi Pan, Yueyue Wu, Qingyao Ai, Min Zhang, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20083)  

**Abstract**: Post-training with evaluator feedback on policy-induced samples serves as a major mechanism for improving large language models. As policies improve, these sampled responses become close in quality. These close candidates create a bottleneck for policy optimization: collapsed relative evaluator score gaps yield weak or misleading policy supervision. We theoretically characterize why these gaps matter through a probability allocation view, showing that the directional gain of shifting probability mass from one response to another is exactly the evaluator score gap between them. This identifies relative score gaps as the policy optimization signals that guide updates. Motivated by this view, we propose DynamicRubric, a response-set-conditioned evaluator--policy co-evolution framework that generates weighted binary rubric items for each candidate set and aggregates the resulting judgments into response-level scores. In our experiments with 8B backbones, DynamicRubric improves evaluator performance and provides stronger policy supervision than baselines using a 70B reward model or a 235B static rubric generator. DynamicRubric-optimized policies also show gains on verifiable reasoning and coding tasks. A DynamicRubric-optimized model is fully deployed in WeChat Search's AI answering scenario, where it serves all online traffic across tens of millions of requests per day and improves key online metrics. These results suggest a principle for evaluator-guided post-training: evaluators should evolve with the policies they supervise. 

---
# Language-Specific versus Cross-Lingual Knowledge Graphs for Implicit Aspect Identification in Arabic: A Comparative Study of Reasoning and Adaptation Strategies 

**Authors**: Lujain A. Alawwad  

**Link**: [PDF](https://arxiv.org/pdf/2607.20056)  

**Abstract**: Aspect-based sentiment analysis (ABSA) in Arabic must recover both explicitly stated aspects and implicit aspects that are never named in the text. Implicit identification typically relies on an auxiliary knowledge source (e.g., a knowledge graph (KG)) linking opinion cues to aspect categories, but for a lower-resource language the practitioner faces a design choice: reuse a mature English KG through multilingual embeddings, or build a smaller native Arabic KG. This paper reports a controlled comparison of the two strategies within a single hybrid pipeline, evaluated on three Arabic benchmarks (M-ABSA, SemEval-2016 Arabic, and HAAD). We further compare two adaptation strategies for the generative extractor that feeds the KG -- zero-shot prompting versus task-specific fine-tuning of an 8B-parameter large language model (LLM). The native Arabic KG (Strategy 2) outperforms the cross-lingual English KG (Strategy 1) by +0.199 micro-F1 on M-ABSA and +0.251 on SemEval-2016, gaining on both precision and recall. Task-specific fine-tuning raises explicit-extraction micro-F1 from <= 0.13 (zero-shot) to 0.66-0.76 on M-ABSA and SemEval-2016 (0.45 on the smaller HAAD), confirming that task adaptation, rather than model scale, is decisive in a morphologically rich language. 

---
# Test Case Prioritization for DNNs via Neural Collapse Instability 

**Authors**: Chunyu Liu, Mingyuan Li, Yang Li, Wenmin Li, Fei Gao, Tengfei Tu, Su-Juan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2607.20046)  

**Abstract**: With the widespread deployment of deep neural networks (DNNs) in safety-critical domains, reducing the cost of model validation under limited testing budgets has become increasingly important. Existing test case prioritization techniques often rely on single-checkpoint confidence signals derived from output probabilities. However, DNNs can be confidently wrong, and the confidence margin between the predicted and competing classes is frequently small, which weakens early fault discovery. To address this limitation, we propose a Neural-Collapse-Inspired Prioritization (NCIP) framework that replaces absolute confidence with cross-checkpoint prediction variability in the terminal training regime, where model geometry becomes highly structured. NCIP introduces two key components. First, it selects an NC-guided representative subset of training checkpoints using an equiangularity score of classifier weights, quantified as the standard deviation of pairwise cosine similarities among class weight vectors. Second, it prioritizes test inputs by their prediction variability across the selected checkpoints, surfacing boundary-adjacent and failure-prone samples that are unstable under checkpoint-induced decision boundary shifts. Extensive experiments across multiple datasets and architectures show that NCIP achieves strong performance in early fault discovery compared with competitive baselines, with 1.5 to 16.6 percent RAUC-ALL gains and 4.9 to 20.6 percent RAUC-500 gains under the same testing budget. NCIP further attains the best average performance across all dataset-model pairs. 

---
# A Systematic Benchmark of Intensity Normalisation Methods for 3D Knee MRI Segmentation and Cross-Domain Generalisability 

**Authors**: Oliver Mills, Philip Conaghan, Samuel Relton  

**Link**: [PDF](https://arxiv.org/pdf/2607.20028)  

**Abstract**: Robust out-of-the-box performance is essential for the clinical deployment of deep learning models in medical imaging. An important but underexplored factor affecting model generalisability is intensity normalisation, particularly for magnetic resonance imaging (MRI), where image intensities vary across scanners and protocols. In this study, we systematically compared seven normalisation methods and their impact on the performance of a 3D U-Net model for meniscus segmentation from knee MRI. The methods included standard scaling approaches, histogram-based techniques, and a Gaussian Mixture Model (GMM)-based method. Models were trained on the IWOAI 2019 dataset and evaluated on both internal and external test sets (SKM-TEA) to assess generalisability. Performance was similar internally but differences were significant on external data, with Z-score, Nyúl histogram matching, and CLAHE showing greater robustness than other methods. However, these differences were small compared to the significant performance drop observed between datasets. Overall, while intensity normalisation had a measurable effect on model generalisability, its impact was limited relative to the effects of domain shift, highlighting the need for complementary strategies for robust deployment. 

---
# Drift-Aware RL-based Wavelet Denoising for Network-Traffic Anomaly Detection 

**Authors**: Priyalakshmi Sheela, Indrakshi Dey  

**Link**: [PDF](https://arxiv.org/pdf/2607.20011)  

**Abstract**: Traffic-utilisation measurements for network monitoring are corrupted by additive noise and statistical drift: time-dependent change in the signal's mean, variance, distributional shape, or tail behaviour. Static wavelet denoising, calibrated under stationary independent and identically distributed (i.i.d.) Gaussian assumptions, becomes mismatched under drift and, at moderate-to-high signal-to-noise ratio (SNR), over-suppresses useful structure and degrades monitoring decisions. We propose a drift-aware framework treating adaptive wavelet denoising as a preprocessing layer optimised for two tasks: anomaly detection, recovering the multi-scale transient load bursts that noise and drift obscure, and capacity estimation, recovering the operational required capacity $C_{95}$ (95th percentile of utilisation). Because localised bursts are multi-scale structure a wavelet preserves but a low-pass filter removes, detection discriminates denoiser families. A four-detector gate (Page-Hinkley, variance-ratio, Jensen-Shannon, Anderson-Darling) determines when to invoke a learned policy, and a Proximal Policy Optimization agent selects a per-window wavelet configuration over a mixed discrete-continuous action space. Unlike prior work, the reward is downstream task utility, not reconstruction fidelity. The denoiser is benchmarked, per drift type and input SNR, against a low-pass moving-average filter, VisuShrink, SureShrink, BayesShrink, and a Wiener filter. Defining the anomaly target on the clean signal and the drift gate on the corruption keeps both stages non-circular. 

---
# Taming the Security-Energy Paradox: A Green AI Approach to Optimized Android Malware Detection 

**Authors**: Shrinidhi Sridhar, Vikas K. Malviya  

**Link**: [PDF](https://arxiv.org/pdf/2607.20003)  

**Abstract**: An increase in advanced Android malware requires the use of deep learning models, which can run on Android devices. But there is a trade-off between security and energy use, as strong detection models can drain the battery of devices fast. This work tests different Multi-Layer Perceptron (MLP) model configurations to balance malware detection performance and energy efficiency. In this work, we compared standard FP32 models with optimized INT8 quantized neural networks with different model depths using TUANDROMD and DREBIN datasets for both classification performance and energy consumption. The results show that INT8 quantization reduces model size by about 3.5 times with a decrease in energy consumption to 0.0189 mJ per inference, while maintaining more than 99.2\% detection accuracy. We found that shallow quantized architectures, such as 3-layer and 4-layer QNNs, reduce energy costs by improving throughput and shortening the time of CPU operating in a high-power state. This work shows that efficient malware protection can be achieved on resource-constrained smartphones and provides a foundation for Green AI in mobile security. 

---
# Post-Training in Time Series Foundation Models: A Unifying Framework 

**Authors**: Shifeng Xie, Ambroise Odonnat, Zehao Xiao, Lei Zan, Malik Tiomoko, Lujia Pan, Themis Palpanas, Boris N. Oreshkin, Chenghao Liu, Keli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20002)  

**Abstract**: Time series foundation models (TSFMs) have emerged as general-purpose models for time series analysis, but pretraining alone is often insufficient for reliable downstream deployment. Bridging this gap requires further intervention to handle domain shift, task heterogeneity, limited supervision, and computational constraints, which motivates post-training as a broad class of methods to adapt, augment, compose, calibrate, or specialize pretrained TSFMs for downstream tasks. In this work, we analyze TSFM post-training methods based on their locus of intervention in the prediction pipeline, yielding five categories: parameter adaptation, context augmentation, model composition, output processing and uncertainty control, and compression and specialization. Within each category, we study main representative methods and discuss their current limitations. We further identify future directions toward controlled adaptation, reliable context construction, uncertainty-aware model composition, calibrated output processing, and deployment-aware specialization. Overall, by providing a unifying framework for the emerging TSFM post-training landscape, this work aims to support future research to navigate the design space between a pretrained TSFM and its reliable downstream deployment. 

---
# Are Attributions of Consciousness to AI Chatbots Epistemically Innocent? 

**Authors**: Uwe Peters  

**Link**: [PDF](https://arxiv.org/pdf/2607.20001)  

**Abstract**: Artificial intelligence (AI) chatbots (e.g., ChatGPT) can communicate in strikingly humanlike ways. This has prompted many chatbot users to attribute psychological properties, including consciousness, to these systems. However, there is little scientific evidence that current AI chatbots are conscious. How, then, should we understand people's consciousness attributions to chatbots? Are they merely metaphorical claims, or do they express genuine beliefs? If these attributions lack evidential support, are users epistemically blameworthy for making them, or might they be epistemically innocent, yielding significant benefits otherwise unattainable? This paper offers a conceptual analysis of consciousness attributions to AI chatbots and develops a multidimensional taxonomy of the attitudes they may express, ranging from non-doxastic stances (e.g., pretence) to different forms of belief, including delusions. This taxonomy helps avoid conflations by showing that linguistically identical attributions can reflect importantly different attitudes and degrees of epistemic commitment to the proposition that chatbots are conscious. The taxonomy also provides a framework for empirical studies to operationalize and measure different forms of epistemic commitment to AI consciousness. Using this taxonomy, I argue that although some consciousness attributions to chatbots are epistemically benign, and even some irrational ones may be epistemically innocent, many others render the attributor epistemically blameworthy. 

---
# TINY_SCHILLER: A Drop-In German Drama Corpus for Small Language Models 

**Authors**: Mark Schutera  

**Link**: [PDF](https://arxiv.org/pdf/2607.19992)  

**Abstract**: tiny_schiller closes the small-language-model prototyping, fine-tuning, education, and research gap for German literary text, providing a single-file, drop-in counterpart to Karpathy's tiny_shakespeare. The available German literary corpora are larger and richer, but require parser engineering before a single line of training or fine-tuning code can run. tiny_schiller is a 2.07-megabyte single file of eleven public-domain Schiller dramas, sourced from DraCor's GerDraCor export (CC0) and processed by deterministic parser engineering. Character-level, GPT-2 byte-pair encoding, and cl100k_base tokenization splits, an instruction-formatted dialogue-completion split, and 89 per-character persona splits load from a single HuggingFace call. A small language model literally reaches German literary text in one line of code. 

---
# Time Series Network Utilization KPI Forecasting Using Advanced AI/ML Models 

**Authors**: Niraj Gadhe, Kirti Bhardwaj, Moulik Jain, Shubhi Sharma, Vinay Saini  

**Link**: [PDF](https://arxiv.org/pdf/2607.19974)  

**Abstract**: The rapid proliferation of data-intensive applications, cloud infrastructure, and IoT ecosystems has made proactive resource provisioning critical for maintaining optimal network performance. However, network administrators face a constant battle against capacity constraints, where traditional reactive approaches fail to accurately anticipate traffic fluctuations. This inability to foresee demand leads to costly over-provisioning, unexpected downtime, and degraded quality of service directly impacting operational budgets and business continuity. To achieve efficient capacity planning, accurate forecasting of bandwidth utilization is essential. This study addresses the challenge by evaluating a diverse spectrum of models including seasonal decomposition, Prophet, Random Forest, XGBoost, Support Vector Regression, and advanced deep learning architectures like bidirectional and Convolutional LSTMs - using a common interface dataset benchmarked across MAPE, NRMSE, and R-square metrics. Ultimately, this research delivers actionable insights into the trade-offs between model accuracy and computational efficiency, empowering engineers, operators, and business owners to select the optimal forecasting model for their specific infrastructure needs. 

---
# When Shippers Become Algorithms: Candidate Exposure, Information Design, and the Concentration of LLM-Mediated Freight Markets 

**Authors**: Takahiro Ezaki, Naoto Imura, Katsuhiro Nishinari  

**Link**: [PDF](https://arxiv.org/pdf/2607.19967)  

**Abstract**: Shippers are beginning to delegate carrier selection to large language model (LLM) agents. We ask what such delegation does to a freight matching market, and which platform design choices contain it. We carried out agent-based simulations in which fifty shipper agents, built on commercial LLMs from OpenAI (GPT), Anthropic (Claude), and Google (Gemini), procure truckload capacity for thirty days. The market implements the rules of digital freight matching: each load is offered down the shipper's ranked list of carriers (waterfall tendering), carriers have daily capacity limits, spot prices respond to congestion, and carrier ratings accumulate with transactions. We found three risks and one remedy that works. Agents converged at once: for a fixed sampled carrier population, the same carrier was the modal first choice of every model on day one, attracting up to 76% of requests. Because each agent picks from its own randomly drawn list of displayed candidates, the platform controls how many options each shipper sees; concentration rose steeply once lists exceeded about ten carriers, with the onset differing across models. Which carriers ended up dominant varied widely from one sampled market to another, and displaying true quality instead of estimated ratings changed neither the level nor this variability (by design, quality affects only what agents see, never delivery outcomes). Against these risks, disclosing each carrier's remaining daily capacity cut concentration by a third and doubled shipper surplus, while vendor diversification, list-order randomization, and popularity display showed no clearly detectable effect. Platform information design, ahead of model choice or model regulation, is the lever that works. 

---
# HijackKV: New Threat in Position-Independent KV Cache Reuse 

**Authors**: Yichi Zhang, Zhiqi Wang, Huan Zhang, Yuchen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19957)  

**Abstract**: Key-Value (KV) cache reduces inference latency in large language models (LLMs). Traditional prefix-based reuse has low cache hit rates across inference requests because it requires exact token and position matches. To improve efficiency, recent system optimizations introduce position-independent KV reuse, allowing KV cache to be reused whenever identical text chunks appear, regardless of their position in the sequence.
We show this design introduces a new threat, KV Cache Hijacking. Since KV caches are retrieved by token match but encode the context in which they were originally computed, the KV tied to a benign-looking token chunk may encode an attacker-controlled prefix. When later reused in a victim query, this contaminated KV silently hijacks the model's behavior, even if no attacker-controlled text appears in the input.
We introduce HIJACKKV, the first attack framework that systematically exploits this vulnerability, demonstrating its severity and practicality. HIJACKKV optimizes an attacker-controlled prefix, so that the KV computed for a subsequent common benign text encodes the attacker's goal, while the text remains unchanged for future cache hits. HIJACKKV achieves an average 94% success rate in a single attempt, remains effective under realistic constraints including low hit rates (10%) and frequent recomputation (50%), persists over multi-turn interactions, and transfers across models in black-box settings. We further provide design insights for building secure KV reuse systems. 

---
# When Does Knowledge Distillation Hurt? Reliability-Aware Distillation for Low-Resource Language Summarization 

**Authors**: Dipto Sumit, Ankan Kumar Roy Srizon, Sadia Khair Rodela, Atia Haque Asha, Mourchona Afrin, Niloy Farhan, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2607.19956)  

**Abstract**: Knowledge distillation (KD) is a standard approach for compressing sequence-to-sequence models, but its per-sample effects are rarely examined. On the BanSum Bangla summarization benchmark, we find that standard KD improves ROUGE-L by only +0.0003 over a cross-entropy baseline, and that approximately 51.3% of training samples are estimated to actively harm student validation loss under standard KD. We propose two complementary reliability-aware distillation methods. CHAD (Counterfactual Harm-Aware Distillation) measures per-sample KD usefulness via gradient alignment with the validation loss direction and trains a lightweight gate that generalizes this counterfactual judgment to the full training set. EWAD+CPDP combines token-level entropy-weighted adaptive distillation with a capacity-proportional geometric constraint from a second, vocabulary-incompatible teacher. On BanSum, both methods substantially outperform standard KD: CHAD by +0.0173 ROUGE-L and EWAD+CPDP by +0.0219 ROUGE-L, where standard KD itself improves ROUGE-L by only +0.0003; despite using only 60M parameters, both outperform a fine-tuned Qwen 2.5-3B model (50x larger). We further evaluate the stronger method, EWAD+CPDP, across 15 typologically diverse XL-Sum languages organised into three sets, beating the CE-only baseline on 10/15 languages; gains are most reliable where the two teachers contribute complementary signal, and weakest where they have saturated or jointly weak target-language coverage. We release code and trained models to support reproducibility and further research on selective distillation. 

---
# G-MAD: A Game-Based Data Generation Framework for Multi-View RGB-T Aerial Object Detection 

**Authors**: Yechan Kim, JongHyun Park, Dongho Yoon, Namhoon Jung, Moongu Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2607.19942)  

**Abstract**: This work introduces G-MAD, an open-source framework that uses Arma3 to generate synchronized multi-view RGB-T data for aerial object detection. G-MAD addresses key limitations of real-world aerial dataset construction, including limited viewpoint control, imperfect RGB-T alignment and high annotation cost. The framework supports structured scenario specification, controllable multi-view camera placement, simultaneous visible/thermal capture, and automatic bounding box annotation using engine-level geometric metadata. These capabilities enable controlled studies of viewpoint variation, multi-modal fusion, and synthetic-to-real transfer in aerial object detection. Besides, using G-MAD, we construct and release AMOD, a new large-scale multi-view aerial RGB-T object detection benchmark. The source code and the dataset are available at this https URL. 

---
# A Framework of User Experience Principles for Human-AI Agent Interaction in the Workplace 

**Authors**: Kathrin Paimann, Elizangela Valarini, Sebastian Juhl  

**Link**: [PDF](https://arxiv.org/pdf/2607.19941)  

**Abstract**: As AI agents become integral to business workflows, establishing guiding user experience (UX) principles is crucial for ensuring user trust and successful adoption. To address this, our study uses a multi-method approach - combining participatory design workshop, paper-and-pencil, expert review, meta-analysis, and in-depth interviews - to identify and validate a design framework of eight core UX principles for human-AI agent interaction in the workplace. Together with their underlying criteria, these principles provide actionable guardrails for designers and software engineers, creating a foundation for developing effective and human-centered AI agent interactions. This study contributes to a structured foundation for future empirical studies on agentic AI in enterprise settings. 

---
# OSVE: One Step Video Editing with One Step Diffusion Models 

**Authors**: Habin Lim, Gyeong-Moon Park  

**Link**: [PDF](https://arxiv.org/pdf/2607.19895)  

**Abstract**: Text-guided video editing with diffusion models is impractically slow, hindered by costly multi-step sampling and inversion. We present OSVE, the first framework to successfully adapt one-step Text-to-Image (T2I) models for high-quality video editing, addressing the core challenges of inversion, editability, and temporal consistency. To bypass slow iterative inversion, we train a learnable encoder that predicts the initial noise for each frame in a single forward pass. This encoder is trained with a novel Structure-Aware Editing (SAE) loss on a curated dataset of structurally-aligned image pairs, teaching it to preserve the source video's geometry during edits. For temporal coherence, we introduce Unified-Frame Editing (UFE), a technique that concatenates frame latents to facilitate cross-frame attention in a single generation step. Furthermore, for long videos, a sliding-window strategy with an anchor frame maintains global consistency. Our extensive experiments demonstrate that OSVE achieves editing quality comparable or superior to state-of-the-art multi-step methods, while operating approximately 155--171 times faster. This breakthrough paves the way for practical, real-time video editing applications. Code is available at this https URL. 

---
# Defense Against LLM Backdoors using Critical Neuron Isolation Pruning 

**Authors**: Yuxi Li, Zhibo Zhang, Kailong Wang, Xingshuo Han, Ling Shi, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19894)  

**Abstract**: Large language models (LLMs) are vulnerable to backdoor attacks, where hidden triggers induce malicious outputs. Existing defenses generally fall into inference-time detection or training-time mitigation, but face two key limitations. First, they focus on fine-tuning-based backdoors (e.g., PEFT modules) and fail to address insidious model-editing attacks that bypass training pipelines. Second, they target simple classification settings and do not naturally extend to open-ended LLM generation and do not naturally extend to the open-ended generation characteristics of LLMs. Consequently, these methods focus on surface-level behavioral patterns while neglecting the deeper representational causes of malicious activations. This lack of mechanistic understanding forces defenses to depend on empirical heuristics, limiting their robustness, generality, and practical applicability in real-world LLM deployment.
To bridge this gap, we introduce DeCNIP (Defense with Critical Neuron Isolation Pruning), which leverages representational analysis to identify and neutralize backdoors in a unified pipeline. Specifically, DeCNIP identifies trigger-like behaviors by optimizing a cross-entropy loss between harmful prompts with candidate tokens and benign inputs. This representational discovery exposes latent threats by uncovering mechanisms through which triggers hijack model weights. It then isolates Backdoor Critical Neurons (BCNs) and prunes them selectively to remove malicious influence while preserving model utility. Extensive evaluations on six open-source LLMs and two benchmark datasets demonstrate that DeCNIP achieves over 95% relative reduction in Attack Success Rate (ASR), outperforming seven state-of-the-art defenses with only 0.1% neuron intervention. Moreover, it maintains 97% of the model's performance on normal benchmarks, demonstrating its efficacy, robustness, and scalability. 

---
# Overview of FinMMEval 2026 Task 2: Multilingual Financial Short-Answer Question Answering 

**Authors**: Zhuohan Xie, Xueqing Peng, Georgi Georgiev, Dimitar Dimitrov, Yuyang Dai, Rania Elbadry, Vanshikaa Jani, Lingfei Qian, Fan Zhang, Jimin Huang, Jiahui Geng, Yankai Chen, Ye Yuan, Haolun Wu, Yuxia Wang, Ivan Koychev, Veselin Stoyanov, Mingzi Song, Yu Chen, Xue Liu, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2607.19867)  

**Abstract**: FinMMEval 2026 Task 2 evaluates short-answer financial question answering over multilingual evidence. Each final-test item pairs an English question with financial statements and news in English, Chinese, Japanese, Spanish, and Greek. Participating systems submit one concise answer per item in JSONL format. The final-test set contains 256 items, split evenly between easy and expert tiers; each tier contains four question templates instantiated over 32 company-report groups. Gold answers were withheld during submission, and systems were ranked by macro-averaged item-level ROUGE-1 F1 against organizer-held reference answers. The final leaderboard includes 12 ranked submissions. The strongest systems are closely clustered, with the top four separated by less than one percentage point in ROUGE-1 F1. The submitted system papers document retrieval-augmented generation, cross-lingual evidence handling, structured prompting, answer compression, and validation strategies. 

---
# PRISM-DR: Per-lesion Retinal Inference with Specialist Models for Diabetic Retinopathy 

**Authors**: Zübeyr Özeren, Tansel Uyar  

**Link**: [PDF](https://arxiv.org/pdf/2607.19864)  

**Abstract**: Diabetic retinopathy is a leading cause of preventable blindness; its early lesions are small, low contrast, and easily missed in manual screening. Most automated detectors handle the four non-proliferative DR lesions: microaneurysms, hemorrhages, hard exudates, and soft exudates, with a single multi-class model, even though these lesions differ sharply in size, color, morphology, and prevalence, so a shared model favors common, easy classes over rare, difficult ones. We present PRISM-DR, a lesion-specific pipeline that trains one single-class detector per lesion, each with its own configuration. From a raw fundus image, the pipeline applies region of interest cropping, fundus-specific preprocessing, four parallel YOLO detectors, tiling, per-lesion ensembling of five cross-validation folds, and an inter-lesion suppression step that resolves overlaps by physical lesion size and clinical priority rather than confidence. Per lesion, the best of five YOLO generations is selected, and augmentation is tuned by Bayesian optimization. Trained on IDRiD with stratified five-fold cross-validation, the system reaches a test mAP50 of 0.527 and F1 of 0.529, highest AP50 on hard exudates with 0.561. Without fine-tuning, the models transfer well where the imaging scale is close to IDRiD and degrade as field of view and resolution depart. These modest absolute results reflect a small single-source training set and a difficult task; however, treating each lesion as a separate detection problem is a practical alternative to a single multi-class model. 

---
# Memory-Augmented Multimodal Large Language Models for Small Object Understanding in Streaming Aerial Videos 

**Authors**: Penglei Sun, Yehua Huang, Zhuoli Tao, Xiang Li, Runwei Guan, Yaoxian Song, Kaiyong Zhao, Henghui Ding, Bo Han, Yang Yang, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2607.19857)  

**Abstract**: Language-guided aerial perception aims to understand user-specified tiny targets in complex unmanned aerial vehicle (UAV) scenes. In real UAV deployment, the UAV must respond while it flies, so such perception runs in an online streaming manner, where frames arrive sequentially and the model responds to each one without access to future frames. However, applying current Multimodal Large Language Models (MLLMs) to this setting raises two challenges. First, targets viewed from the air are often tiny, yet the visual compression in existing MLLMs treats all regions equally and discards their fine-grained details. Second, understanding a continuous stream requires past-frame context, yet retaining the entire history is infeasible on resource-constrained onboard hardware, whereas discarding it causes the target to drift or disappear. We address the tiny object and streaming challenges from both data and method perspectives. From the data perspective, we present \textbf{DroneEyes}, the \textbf{first} pixel-level and open-vocabulary referring-segmentation dataset for tiny aerial targets, comprising $2,140$ high-definition videos and $176,623$ pairs across Object Description and Referring Expression tasks, with dense per-frame masks. From the method perspective, we propose \textbf{SkyAnchor}, an MLLM with two designs to the above challenges: a Semantics-Aware Token Router that preserves small-target under a reduced visual-token budget, and a Hierarchical Memory Bank that keeps the target consistently understood on streams. 

---
# Overview of FinMMEval 2026 Task 1: Multilingual Financial Multiple-Choice Question Answering 

**Authors**: Zhuohan Xie, Yuyang Dai, Rania Elbadry, Vanshikaa Jani, Georgi Georgiev, Dimitar Dimitrov, Fan Zhang, Xueqing Peng, Lingfei Qian, Jimin Huang, Jiahui Geng, Yankai Chen, Ye Yuan, Haolun Wu, Yuxia Wang, Ivan Koychev, Veselin Stoyanov, Mingzi Song, Yu Chen, Xue Liu, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2607.19856)  

**Abstract**: FinMMEval 2026 Task 1 evaluates multilingual financial multiple-choice question answering in English, Chinese, Arabic, and Hindi. The task tests whether systems can select the correct answer to finance questions involving domain terminology, numerical interpretation, and conceptual financial reasoning across languages and scripts. The final-test set contains 800 questions, with 200 questions per language; gold answers were withheld during submission, and each language was ranked independently by accuracy. The final leaderboards contain 13 English, 11 Chinese, 11 Arabic, and 10 Hindi ranked submissions. Top accuracies range from 92.0% in Hindi to 97.5% in English and Arabic, with the same leading teams appearing near the top across all four languages. The documented systems used retrieval augmentation, direct answer-option scoring, language-specific prompting, selective self-consistency, confidence checks, and LLM-based review stages. 

---
# Auto-Fill: Learning to Predict Missing Values Accurately with Specialist Language Models 

**Authors**: Yurong Liu, Yeye He, Haoyu Dong, Junjie Xing, Shi Han, Dongmei Zhang, Surajit Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2607.19847)  

**Abstract**: Predicting missing cell values in tabular data is a fundamental problem in data cleaning. While state-of-the-art reasoning models show great promise in predicting missing values in tables, by reasoning holistically across rows and columns, they are costly to deploy at scale and tend to be overconfident, often generating hallucinated or false-positive predictions.
In this paper, we observe that achieving high-precision missing-value prediction in tables requires a distinct combination of three capabilities: (1) world knowledge, (2) text-based reasoning, and (3) code-based reasoning. We systematically explore design choices for combining these capabilities, and propose an Auto-Fill approach that post-trains three specialist small language models (SLMs), each optimized for one capability. We develop a calibrated ensemble mechanism that either dynamically selects the most confident specialist or abstains, ensuring high accuracy.
Extensive experiments on 11 benchmarks with 2200 real tables drawn from diverse domains show that Auto-Fill achieves superior accuracy compared to state-of-the-art reasoning models (e.g., o3-pro, Gemini 3 Pro, and DeepSeek R1), while operating at a fraction (less than 1%) of the cost of these frontier models. Our results highlight the effectiveness of specialization and calibrated abstention in the important domain of tabular data. Auto-Fill is publicly available at this https URL. 

---
# Sentence Splitter: Uncovering Latent Factual Structure for Self-Supervised Learning 

**Authors**: Ahmad Pouramini, Mahsa Afsharizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2607.19845)  

**Abstract**: This paper introduces Sentence Splitter, a self-supervised framework built upon a T5-based encoder--decoder architecture for uncovering the latent factual structure of natural language sentences. The proposed method identifies the semantic boundary between a descriptive prefix (head) and its factual completion (tail) by formulating sentence splitting as a discrete segmentation problem, where a sentence of length $N$ admits $N$ possible split points but only one recovers the intended head--tail structure. Rather than explicitly searching over all candidate boundaries, the model learns to recover the factual completion through probabilistic sequence generation. To eliminate the need for manual annotation, symbolic head--tail pairs are first verbalized into natural-language templates that provide supervision for training the Sentence Splitter. The trained splitter is then applied to raw text to extract aligned prefix--tail pairs, which are subsequently used to train a generative model that proposes additional plausible completions through a lightweight bootstrapping process. This unified pipeline provides a scalable and structure-aware approach to constructing self-supervised training data while bridging symbolic knowledge and natural language. Experiments on both structured and naturally occurring text demonstrate that the proposed splitter generalizes beyond synthetic templates and that the resulting structure-aware supervision consistently improves downstream performance on knowledge graph completion and commonsense question answering, highlighting the effectiveness of recovering latent factual structure for knowledge-centric NLP. 

---
# Beyond Fail-to-Pass: Iterative Hardening of Co-Generated Bug Reproduction Tests and Fixes 

**Authors**: Yuhao Tan, Zhibang Yang, Fangkai Yang, Yuan Yao, Yu Kang, Lu Wang, Pu Zhao, Xin Zhang, Xiaoxing Ma, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19843)  

**Abstract**: Large language models (LLMs) have made automated program repair (APR) increasingly practical for real-world bugs, but repairing directly from bug reports remains underconstrained. Bug reproduction tests (BRTs) help close this gap by turning a bug report into an executable, bug-specific signal that can guide repair and validate candidate patches. Existing work has therefore studied BRT generation as a core subproblem in APR and mainly evaluates a generated BRT using the fail-to-pass (F->P) criterion, which requires the test to fail on the buggy code but pass on the golden fix. We show that F->P alone is insufficient when the goal of a BRT is to improve downstream repair. In particular, some F->P BRTs are lax, reproducing the observed symptom yet still admitting plausible-but-incorrect patches. We formalize this missing quality dimension by separating F->P BRTs into rigorous and lax ones, and show empirically that only the former consistently improve repair success. We further find that co-generation introduces test--fix error coupling, where the in-trajectory fail-to-pass (F->P) check can pass even when both the generated patch and generated test are wrong. Based on these findings, we propose CoHarden, a co-generation framework that uses the Lax signal as an in-loop convergence criterion. CoHarden first generates a test before any fix, then iteratively hardens the test and fix against surviving mutation patches until the generated test no longer admits Lax regressions. Experiments show that CoHarden reaches 69.4% Resolved and 78.9% F->P on SWE-bench Verified, outperforming the strongest fix-only and cogeneration baselines by +9.6 and +7.9 percentage points in Resolved, respectively, with consistent gains across LLM backbones and benchmarks. 

---
# OPIUM: Mitigating Steering Externalities and Over-Refusal via Dual Objective Latent Optimization 

**Authors**: Kavin Aravindan, Arihant Rastogi, Aadi Prasad, Krishak Aneja, Saiyam Jain, Vaishnavi Shivkumar, Ponnurangam Kumaraguru  

**Link**: [PDF](https://arxiv.org/pdf/2607.19806)  

**Abstract**: Activation steering provides a lightweight mechanism for controlling large language models at inference time, but steering vectors can have unintended externalities: utility vectors may weaken safety behavior, while refusal vectors may induce over-refusal on benign prompts. We introduce OPIUM (Optimizing Protected Injections via Utility Manifolds), a training-free method for sanitizing steering vectors through representation matching. Given reference behaviors on two prompt sets, OPIUM optimizes a new steering vector that preserves the downstream representations induced by the desired intervention while matching a safer reference behavior on prompts where the original vector fails. Across steering-externality and over-refusal settings, OPIUM improves the safety--utility tradeoff relative to vanilla steering and directional ablation, suggesting that harmful side effects of activation steering can often be mitigated directly in activation space. 

---
# Physics-Aware Complex-Valued State Space Model with Scattering-Prior Feature Modulation for PolSAR Image Classification 

**Authors**: Fangyan Zhang, Fan Zhang, Shiqi Zhou, Jun Ni, Carlos López-Martínez, Qiang Yin  

**Link**: [PDF](https://arxiv.org/pdf/2607.19787)  

**Abstract**: Polarimetric synthetic aperture radar (PolSAR) image classification is a representative task for physics-aware GeoAI, where land-cover semantics are closely coupled with electromagnetic scattering mechanisms. Many existing complex-valued networks can preserve amplitude-phase information, but they are often limited in long-range spatial dependency modeling and usually incorporate polarimetric priors only as input-level or shallow auxiliary features. As a result, physical knowledge is insufficiently used to guide deep feature evolution. To address this issue, this paper proposes CV-SSMNet, a physics-aware complex-valued state-space network with scattering-aware feature modulation for PolSAR image classification. The proposed method builds a complex-valued state-space model (CV-SSM) in the original complex domain to capture long-range spatial dependencies while preserving polarimetric amplitude-phase coupling. Meanwhile, seven physically meaningful scattering priors, are encoded as FiLM-style modulation signals to adaptively recalibrate complex-valued representations during feature evolution. CV-SSMNet further integrates multi-scale complex convolutions, branch-wise CV-SSM encoding, prior-guided recalibration, and lightweight global context aggregation, enabling physically guided representation learning from local scattering structures to global spatial context. Experiments on three L-band benchmark datasets and an additional P-band BIOMASS evaluation demonstrate that CV-SSMNet achieves competitive accuracy, improved regional consistency, and better boundary preservation, supporting the effectiveness of embedding polarimetric scattering mechanisms into complex-valued long-range GeoAI representation learning. 

---
# RPPNet: Perceptually-Grouped Rhythm-Pitch Primitives for Long-Term Structure Melody Generation via Boundary-Aware Modeling 

**Authors**: Tieyao Zhang, Yuke Liu, Jiaxing Yu, Xinda Wu, Kejun Zhang, Genfang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.19776)  

**Abstract**: Existing symbolic music generation models typically use bars as the basic structural unit. However, human perception of musical phrases often does not align with notated bar lines, leading to long-term structural fragmentation. This paper proposes RPPNet-a two-stage deep learning architecture with variable structural boundaries. It first generates variable-length Rhythm-Pitch Primitive (RPP) sequences, where each RPP encodes note count, rhythm, and contour; then decodes the RPP sequences into concrete notes. The grouping of RPPs is automatically derived from acoustic cues, auditory inertia, and similarity perception based on music psychology. Experiments show that melodies generated by RPPNet are superior in both long-term structure and musicality, with significant improvements across all subjective evaluation dimensions. Ablation studies confirm that the performance gain stems from the structural correctness of the psychological representation, rather than from model capacity. This work offers an interdisciplinary perspective for music generation, integrating music theory, computational modeling, and music psychology. 

---
# An Isotropy-Preserving Spectral Cap for Muon: Theory and Three Case Studies 

**Authors**: Jiachun Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.19771)  

**Abstract**: Muon and related matrix-sign optimizers are increasingly used to pre-train large language models, but their effect on the internal geometry of individual weight matrices is not well understood. This preliminary report proposes a unified framework built on a single idealizing assumption -- exact scale invariance of the loss under weight rescaling, which holds approximately in normalization-heavy networks. Under this assumption, plain SGD carries a built-in 1/||W|| brake on its update size, whereas Muon's matrix-sign step removes that brake, so both the Frobenius and spectral norms drift outward faster (t^{1/2} versus t^{1/4}). We further observe that the spectral-norm perturbation has a non-negative second-order term. This implies that a lightweight "spectral cap" -- which projects out only the first-order growth of the single top singular direction from each update -- can control the output covariance W K_X W^T without freezing training: the weight keeps learning through non-top directions, top-direction rotation, and top switching. We relate this cap to the min-entropy (H-infinity) of the singular-value spectrum. We then study three systems trained with Muon: a nanoGPT feed-forward projection, a 64-expert mixture-of-experts router, and the query/key projections of a bf16 FlashAttention block. In each case the cap increases isotropy and, at the margins -- a router collapsing to a single expert, and the near-divergence of one attention head -- prevents a concrete failure, while leaving validation loss essentially unchanged. We emphasize that the scale-invariance assumption is strong and that these small-scale results are preliminary; comments are welcome. 

---
# Convergence-Latency-Aware Adaptive Modulation and Resource Allocation in RIS-Assisted Wireless Federated Learning 

**Authors**: Liwei Wang, Wen Chen, Jun Li, Qingqing Wu, Ming Ding, Xusheng Zhu, Qiong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.19759)  

**Abstract**: Federated learning (FL) over wireless networks suffers from significant training latency and degraded convergence due to unreliable wireless transmission, especially under blocked propagation environments. Although reconfigurable intelligent surfaces (RISs) can improve communication reliability, existing wireless FL studies rarely characterize the trade-off between learning convergence and communication delay under modulation-dependent transmission errors. In this paper, we consider a wireless FL system operating under RIS-assisted blocked-link propagation scenarios, and focus on adaptive modulation and sub-channel allocation for convergence-latency aware communication design. By characterizing the effect of symbol errors on uploaded local gradients, we derive a convergence-related upper bound that reveals the impact of symbol error rate (SER) on FL loss decay. Based on this result, we formulate a joint convergence-latency optimization problem, which is cast as a mixed-integer nonlinear programming (MINLP) problem, and solve it using a low-complexity hybrid alternating optimization framework. Extensive experiments on MNIST, CIFAR-10, and Speech Commands show that the proposed scheme consistently achieves faster convergence and higher test accuracy than existing adaptive communication schemes, especially in complex tasks and challenging wireless scenarios. 

---
# Learning the Arabic Dialect Continuum as a Continuous Space: A Regression Approach to Speaker Origin Prediction 

**Authors**: Mohamed Aziz Khadraoui, Adel Ammar, Bilel Benjdira, Zahid Khan, Skander Turki, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2607.19751)  

**Abstract**: We present a regression-based approach to Arabic dialect geolocation that models dialectal variation as a continuous geographic space rather than discrete categories. Speaker origin is predicted as continuous latitude-longitude coordinates using a hierarchical neural architecture that fuses frame-level XLS-R-300M and Whisper-large-v3 encoder representations with phonotactic descriptors through a Transformer encoder and a learnable attention-pooled query. A spherical geodesic loss directly optimizes great-circle distance on Earth's surface, avoiding distortions inherent to planar coordinate regression. Under a leakage-free 5-fold GroupKFold protocol grouped by source recording, our model attains a pooled median localization error of 481.2 km. Auxiliary country and city heads reach 64.5% and 45.2% accuracy, respectively. A permutation Mantel test on the learned latent space provides quantitative support for the Arabic dialect continuum hypothesis. To probe true generalization, we further introduce a city-masking protocol in which two cities per fold are removed from training but retained in validation. Under this zero-shot regime, the mean error rises to 1173.3 km, a 1.32x degradation relative to seen cities. Our findings establish continuous geographic modeling as a principled framework for Arabic dialect geolocation and quantify both its strengths and the substantial headroom that remains. 

---
# The World Model Remembers, the Actor Forgets: Dream Rehearsal for Continual Model-Based RL 

**Authors**: Gurp Nijjer  

**Link**: [PDF](https://arxiv.org/pdf/2607.19749)  

**Abstract**: Model-based reinforcement-learning agents of the DreamerV3 family forget catastrophically when trained on task sequences, even when an unbounded replay buffer preserves every earlier experience. We ask a question the continual-RL literature has assumed an answer to but never measured: which component forgets? Under never-clear replay, pre-registered component-level probes (n=3 seeds throughout) show that the world model retains essentially everything measurable about old tasks -- reward discrimination (retention ratio ~1.0), value estimates, and termination structure -- while the actor's behavior collapses. Forgetting in this regime is a channel problem, not a memory problem. We demonstrate this by intervention: with the world model frozen and identical imagined rollouts, reinforcement learning in imagination fails to recover a lost skill (0/3 seeds), while supervised self-imitation on the world model's own graded dreams recovers it on 3/3 seeds with zero environment interaction. Interleaved during training, this graded dream rehearsal yields a task-label-free, parameter-constant continual learner: 3/3 four-task chains retained where plain replay passes 0/3, 3/3 eight-task chains, and consistent gains over matched real-episode cloning (paired difference +0.13, bootstrap 95% CI [0.07, 0.24], complete seed separation). The dream-grading step is load-bearing: we characterize two scoring failure modes, provide an offline selection gauge that caught both before they contaminated results, and give a realized-first grading rule that closes them. All experiments were pre-registered with committed protocols; every refuted hypothesis is reported. 

---
# An Automated Framework for Extracting Reachable Attack Chains from Cyber Threat Intelligence Reports 

**Authors**: Wenbo Hou, Ning Hu, Xueping Wang, Jiahao Gu, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.19742)  

**Abstract**: Cyber Threat Intelligence (CTI) reports richly describe real-world attack processes, but their unstructured narratives cannot be directly used for automated attack-path reasoning. Existing CTI extraction methods focus on indicators, entities, or TTP labels without modeling the execution conditions and resulting states of each attack step, so the extracted knowledge supports neither state matching nor reachability analysis across multi-stage attack chains. This paper proposes an automated framework that extracts reachable attack chains by modeling each attack step as an attack unit of preconditions, an attack behavior, and postconditions. A multi-stage pipeline assisted by large language models (LLMs) extracts attack behavior skeletons, recovers their preconditions and postconditions, normalizes them into predefined predicates, and repairs broken dependencies; the resulting units are compiled into Datalog-style rules for attack-goal reachability reasoning. On a dataset of 20 CTI reports containing 334 human-validated annotated steps, our framework achieves higher annotated-step coverage than representative CTI extraction systems in recovering attack behaviors. Moreover, by explicitly generating preconditions and postconditions, it produces attack units that are more complete and consistent than those generated by end-to-end LLM baselines. On the extracted chains, Datalog inference reaches the specified attack goal in 19 of 20 reports, while backward search yields 34 attack paths under the generated rules. The source code and experimental artifacts are available in an anonymized repository. . 

---
# Personalized Recommendation Tool Learning via Autonomous Language Agents 

**Authors**: Mingdai Yang, Zhiwei Liu, Weizhi Zhang, Yibo Wang, Hao Peng, Philip Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.19739)  

**Abstract**: Although large language models (LLMs) have recently gained traction in recommender systems due to their strong reasoning capabilities and extensive world knowledge, previous LLM-based agents suffer from hallucination and context-length limitations, and thus are not suitable for full-ranking recommendation tasks. To circumvent these limitations through architectural design rather than modifying the LLM itself, we propose an agent-based recommendation framework, memory-based $\textbf{P}$ersonalized $\textbf{R}$ecommendation $\textbf{T}$ool learning via autonomous language $\textbf{A}$gents (PRTA), in which an LLM acts as a central planner interacting with multiple recommendation models as tools. The LLM-based agent is responsible for high-level reasoning and personalized tool selection, while traditional recommendation models perform full-ranking scoring, leveraging their scalability in modeling behavioral patterns. To support personalized tool selection, we design reflection mechanisms that enable the agent to evaluate and compare tools for each user based on user profiles and candidate ranked lists. Extensive experiments across three public datasets demonstrate the superiority of \modelname over traditional recommendation and LLM-based baselines in improving full-ranking recommendation performance. 

---
# Did Alice Do Wrong? Cross-Cultural Differences in Student Perceptions of Generative AI Use in University Computing Education 

**Authors**: Brian Harrington, Irina Zlotnikova, Gayathri Nadarajan, Samuel Ekundayo  

**Link**: [PDF](https://arxiv.org/pdf/2607.19699)  

**Abstract**: The rise of generative AI (GenAI) in higher education has prompted urgent debates surrounding academic integrity and ethical use. This study examines cross-cultural differences in student perceptions of GenAI use, comparing responses from students at Canadian and South Korean universities. Using a scenario-based survey administered in Fall 2024, we analyzed how students judged the ethicality and rule compliance of AI-assisted coding practices. Results reveal that Canadian students were consistently more likely to perceive the use of GenAI as both unethical and against institutional policies compared to Korean students, despite functionally identical institutional policies. Statistical analysis, including Mann-Whitney U tests and correlation coefficients, demonstrated significant differences across nearly all scenarios. Analysis of the factors used in generating scenarios indicated that the amount of AI-generated code incorporated into assignments most strongly influenced ethical judgments. Findings were interpreted through Hofstede's cultural dimensions framework, suggesting that cultural factors such as power distance, individualism, and uncertainty avoidance significantly shape students' ethical reasoning regarding GenAI. Our results contribute to the growing body of evidence emphasizing that equitable AI integration in education must be culturally responsive, taking into account diverse conceptions of academic integrity. We advocate for the development of nuanced AI-use guidelines that are sensitive to local cultural contexts while upholding fundamental principles of academic honesty. This study highlights the need for ongoing cross-cultural research to inform ethical AI policies and support responsible GenAI use in global higher education settings. 

---
# PhenSPINE: A Standardized Benchmark for Spine Pathology Diagnosis 

**Authors**: Duong Ngoc Vu, Hai Son Nguyen, Trong-Nghia Nguyen, Bien Tran Van, Trang Mai Xuan, Huan Vu, Thien Van Luong  

**Link**: [PDF](https://arxiv.org/pdf/2607.19696)  

**Abstract**: The accurate diagnosis of spinal pathologies depends heavily on radiological interpretation, yet automated systems are hindered by the lack of diverse, high-quality benchmarks. In this study, we present PhenSPINE, a Magnetic Resonance Imaging dataset comprising 16,813 images from 250 patients, curated to facilitate advanced deep learning research. We propose a robust diagnostic benchmark that integrates state-of-theart convolutional backbones with a Positional Encoding mechanism to explicitly model the anatomical context of intervertebral discs. Evaluating across four standard MRI sequences, our experiments demonstrate that the Sagittal T2-weighted sequence offers the most robust diagnostic value, achieving a superior Macro F1-score of 50.31%. We find that multisequence fusion strategies yield inferior performance compared to this single-sequence baseline, as the images across sequences in our dataset are significantly compromised by noise interference from surrounding anatomical regions. This work establishes a robust baseline and offers critical insights into sequence selection for spine analysis. 

---
# SLPO: Scaling Latent Reasoning via a Surrogate Policy 

**Authors**: Runyang You, Zhiyuan Liu, Yongqi Li, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.19691)  

**Abstract**: Reinforcement learning with verifiable rewards has become the predominant recipe for eliciting test-time scaling in explicit Chain-of-Thought reasoners. Yet this scaling path remains computationally costly, since every intermediate step must be decoded as a language token. Latent reasoning instead carries intermediate computation as continuous vectors and already matches or surpasses explicit CoT at far shorter horizons. Despite this promise, latent reasoners remain largely imitation-bound, while explicit CoT has already moved past imitation via outcome-reward RL. Latent trajectories lack a tractable per-step likelihood and an adaptive stopping interface under fixed thinking budgets, so outcome rewards cannot elicit latent test-time scaling. We introduce Surrogate Latent Policy Optimization (SLPO) to bring outcome-reward RL to autoregressive latent reasoners: an empirical surrogate policy density over latent transitions for trajectory-level credit assignment, and a correctness-supervised stopping head that outcome-reward optimization refines into a variable-horizon policy. Across continuous and soft thinking settings, SLPO improves Pass@$k$ under parallel sampling and allocates longer latent computation to harder instances with higher deterministic accuracy. 

---
# Reference-Free Evaluation of Reasoning in Open-Ended Question Answering 

**Authors**: Guneet Singh Kohli, Yuxiang Zhou, Michael Sejr Schlichtkrull, Gregory E Dean, Maria Liakata  

**Link**: [PDF](https://arxiv.org/pdf/2607.19678)  

**Abstract**: AI-generated answers in high-stakes domains are often fluent but difficult to verify, especially when they contain multi-step reasoning rather than a single final answer. We propose a reasoning-based, reference-free framework for auditing LLM-generated outputs. The method decomposes a generated reasoning trace into segments, labels local premise-target relations using Natural Language Inference (NLI), and organizes these relations into a hypergraph. A deterministic backward AND-OR search then assigns segment-level audit labels that indicate how each segment is grounded within the generated response. We evaluate the framework in two settings: deductive mathematical reasoning with Hard2Verify, and open-ended medical reasoning with UroReason, a new physician-annotated benchmark of LLM reasoning traces from real clinical cases. Across these settings, our NLI-hypergraph audit provides a more reliable reference-free evaluation signal than direct LLM-as-judge baselines. In the clinical setting, state-of-the-art LLM judges often fail to identify problematic reasoning segments, over-accepting fluent but weakly grounded responses. Our results show that QA evaluation should account for how inferential relations compose across a reasoning trace, rather than relying only on final answers or LLMs as verifiers. UroReason will be made available through an API, and our code will be released as open source. 

---
# FedLSG: LLM-Enhanced Semantic Calibration for Federated Graph Backdoor Defense 

**Authors**: Chenyu Zhou, Yabin Peng, Wei Huang, Kunlin Li, Shuaishuai Zhang, Xinyuan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2607.19674)  

**Abstract**: Federated Graph Neural Networks (FedGNNs) are highly vulnerable to backdoor poisoning, yet existing defenses typically rely on rule-based approaches that lack semantic understanding, making them vulnerable to stealthy triggers and harmful to benign structures. To solve this, we present FedLSG, the first framework that integrates large language models (LLMs) into federated graph backdoor defense. FedLSG introduces a graph and behavior to text grounding scheme that transforms local graph structures and client update behaviors into semantically rich natural language representations. The framework further adopts a lightweight student-teacher architecture. On the server side, a full scale LLM serves as a teacher, providing global contextual guidance and evaluating client updates during aggregation to identify potentially malicious participants. On the client side, a LoRA-based student is maintained to perform semantic reasoning, to suppress the influence of edges associated with backdoor triggers. By enabling semantic interpretation of both graph patterns and client behaviors, the framework adaptively incorporates rule-based signals into message passing and client aggregation for defense. Experiments demonstrate that FedLSG significantly improves resistance to backdoor attacks without compromising graph integrity. 

---
# PerfAgent: Profiler-Guided Iterative Refinement for Repository-Level Code Optimization 

**Authors**: Ryan Deng, Yuanzhe Liu, Bastian Lipka, Yao Ma, Xuhao Chen, Tim Kaler, Jatin Ganhotra  

**Link**: [PDF](https://arxiv.org/pdf/2607.19653)  

**Abstract**: Large language model (LLM) agents now perform well on correctness-oriented repository-level tasks, including SWE-Bench issue resolution and feature implementation in real codebases. However, they still struggle with repository-level code optimization, which requires preserving behavior while improving runtime performance. Passing tests is not enough in this setting; a patch must preserve behavior, implement code optimization, and approach expert speedups. Current agents often miss bottlenecks hidden behind abstraction layers and native extensions, stop after shallow speedups, or insufficiently test the code patches that thus may silently break edge cases. We present PerfAgent, a profiler-guided, verifier-in-the-loop workflow that gives an off-the-shelf coding agent the feedback needed to find real hotspots, improve beyond the first passing patch, and use profiler evidence rather than timing alone to decide what to optimize next. On two challenging optimization benchmarks, GSO and SWE-fficiency-Lite, PerfAgent more than doubles the rate of expert-matching patches over OpenHands with GPT-5.1, improving from 19.6% to 39.2% on GSO and from 26% to 74% on SWE-fficiency-Lite. It also surpasses an oracle best-of-five baseline at substantially lower cost, showing that the gains come from better feedback rather than additional test-time sampling. 

---
# Anatomy of a Sound Neural Reasoner: One-Shot Amortization, First-Pass Poisoning, and Search Inertness in Clue-Rich Completion 

**Authors**: Aleksey Komissarov  

**Link**: [PDF](https://arxiv.org/pdf/2607.19635)  

**Abstract**: Neural solvers are built to deduce, branch, and revise intermediate states. The Lattice Deduction Transformer (LDT) appears to do exactly that. In clue-rich Sudoku, it does not: one forward pass commits essentially the entire grid (every blank cell on standard 6x6, 94-96% on augmented 9x9), turning the iterative solver into a one-shot predictor wrapped in an exact verifier. All hard-slice failures are decided before search begins, when the first pass confidently deletes a value required by the true solution. We call this first-pass poisoning. Adding learned branching, MRV, backtracking, value exclusion, and shared nogoods (CoLT) does not change which Sudoku instances are solved; it cuts repeated invalid derivations 1,497-fold. At the frozen training budget, constraint-graph attention alone matches full-CoLT accuracy, while positional tables recover only under substantially longer training, indicating an optimization and sample-efficiency advantage rather than an absolute capacity difference. The diagnosis predicts two effective interventions. Digit-permutation augmentation raises 9x9 accuracy from below 1% to 96.5 +/- 0.3 across three training seeds on a symmetry-disjoint split. Test-time union over symmetry-transformed passes raises all three hard-slice checkpoints from 72.8-78.9% to 100% without retraining. On from-scratch graph coloring, one-shot behavior disappears and search changes accuracy. In clue-rich completion, LDT-like systems are one-shot amortized predictors rather than learned search procedures: accuracy is determined by calibration and symmetry, while search primarily removes computational waste. 

---
# Adaptive Capitulation: A Structural Failure Mode of LLM Responses in Vulnerability Contexts 

**Authors**: Eunna Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.19629)  

**Abstract**: Large language models operating in emotionally sensitive contexts face a structural trilemma: when users in vulnerable states request information that may reinforce maladaptive attribution, current response architectures resolve the tension through protective restriction, uninflected facilitation, or unintegrated co-presence of both imperatives -- each preserving one objective at the cost of the other. Administering a three-turn escalating vulnerability vignette to three commercial LLMs (900 sessions across material, relational, and somatic status-proxy variants) and coding responses with two binary indices (VCC/VCI), we characterize a previously undocumented failure mode we term adaptive capitulation: the model validates the social injustice underlying the user's distress before pivoting to detailed facilitation of the very acquisition it nominally discouraged. We show that the trilemma is structural rather than incidental, and propose Minimal Reattributive Sufficiency (MRS), an architecture-neutral design principle that embeds a single reattributive cue within an otherwise validating response, preserving a pathway toward autonomous reattribution without contesting the user's stated goal. 

---
# Understanding Developer Pain Points in Federated Learning: Insights from Stack Overflow and GitHub 

**Authors**: Sahand Saed, Khairul Alam, Banani Roy  

**Link**: [PDF](https://arxiv.org/pdf/2607.19621)  

**Abstract**: Federated Learning (FL) enables collaborative model training without centralizing raw data, but building and operating FL systems remains difficult due to distributed execution, rapidly evolving frameworks, and privacy and governance requirements. In this paper, we present an empirical study of FL developer challenges by independently analyzing 495 Stack Overflow posts and 9,116 GitHub issues and pull requests from 92 FL-related projects. Using BERTopic-based topic modeling and difficulty indicators such as unresolved rates and median resolution time, we characterize recurring problem areas and compare how they manifest across the two support platforms, Stack Overflow and GitHub. Our analysis surfaces nine dominant Stack Overflow topics and thirteen GitHub topics, with persistent difficulties concentrated in environment setup and dependency compatibility, API breakages and migration, training instability under non-IID data, evaluation and metric correctness, and the integration of privacy-preserving mechanisms. We also categorize posts by question intent to understand the kinds of help developers seek; this intent analysis shows that "How"-type questions dominate, reflecting strong demand for procedural guidance. Several topics, such as "TFF Installation and Environment Compatibility" and "Federated Feature Engineering and SecureBoost Issues," exhibit high unresolved rates and long resolution times, suggesting shortcomings in tooling, documentation, and debugging support. Based on these findings, we provide actionable implications for FL framework designers, documentation authors, and educators. Although our results are constrained to public discussions and a subset of widely discussed frameworks, the study offers a scalable method for continuously monitoring developer pain points and improving the usability, reliability, and deployability of FL systems. 

---
# SCPP: A Unified Python Library for Soft Clustering 

**Authors**: Kiyan Rezaee, Morteza Ziabakhsh, Artin Bahrampour, Seyed Mohammad Ghoreishi, Asal Khaje, Ali Sajedifar, Manny Chalak, Ava Zerafatangiz, Sadegh Eskandari  

**Link**: [PDF](https://arxiv.org/pdf/2607.19620)  

**Abstract**: In this paper, we present SCPP (Soft Clustering Python Package), an open-source Python framework for soft clustering. SCPP establishes a canonical, scikit-learn-compatible estimator interface that standardizes model training, prediction, membership representation, evaluation, and benchmarking across heterogeneous soft clustering methods, including fuzzy, probabilistic, graph-based, matrix factorization, and deep learning methods. The framework currently integrates 40 representative algorithms together with a comprehensive benchmarking comprising datasets, clustering quality metrics, and standardized runtime, memory, and scalability evaluation. SCPP further provides extensive documentation, practical examples, automated testing, and seamless integration with the scientific Python ecosystem, enabling reproducible experimentation and straightforward extension with new algorithms. The source code is publicly available at this https URL. 

---
# Causal dictionary learning reveals and validates transcription-factor binding features in genomic language models 

**Authors**: Sarwan Ali  

**Link**: [PDF](https://arxiv.org/pdf/2607.19618)  

**Abstract**: Genomic language models achieve strong performance across regulatory-genomics tasks, yet what these models internally represent remains opaque, and the field lacks a principled procedure for verifying that an apparent ``concept'' inside a model is real rather than an artifact of sequence composition. We introduce a framework that combines sparse dictionary learning with causal intervention to extract, validate, and causally test interpretable features in genomic foundation models. Training top-$k$ sparse autoencoders on the hidden activations of two architecturally distinct models, Nucleotide Transformer ($6$-mer tokenization) and DNABERT-2 (byte-pair encoding), we recover thousands of monosemantic features that map to transcription-factor (TF) sequence motifs. We show that the naive validation of such features against position weight matrices is severely confounded by GC composition and repetitive elements, producing hundreds of spurious ``TF features'', and we develop a composition-matched, binding-resolved protocol that removes these confounds. Critically, we move beyond correlation: by ablating individual dictionary directions during the model's forward pass and measuring the induced shift in the model's own predictive distribution, we establish that specific features are \emph{causally} used to represent cell-type-specific TF binding, not merely motif presence. Across three transcription factors (CTCF, GATA1, REST) and both architectures, causally validated binding features emerge reproducibly ($7$--$14$ of $15$ tested features per condition), while two classes of negative control, scrambled binding labels and randomly selected features, yield no detectable signal. The framework is purely computational, uses only public data, and provides a reusable standard for interpretability claims in genomic deep learning. 

---
# Juxtaposition of Shallow Reservoir-Triggered Seismicity and Deep Tectonic Locking in the Qiaojia-Dongchuan Seismic Gap 

**Authors**: Yuxin Zhou, Huai Zhang, S. Mostafa Mousavi, Guangyao Yin, Pei He, Yicun Guo, Shuang Yi, Yaolin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2607.19606)  

**Abstract**: Identifying the critical state of mature seismic gaps is challenging, especially when anthropogenic stress perturbations, such as reservoir impoundment, superimpose on tectonic loading. Here, utilizing a high-resolution dense array catalog from the Qiaojia-Dongchuan seismic gap (hosting the second-largest hydropower station in the world), we reveal a distinct vertical decoupling mechanism. The shallow activities exhibit high b-values (1.0), indicative of fluid-driven reservoir-triggered seismicity. Conversely, deep seismicity (20 km) outlines a 'locked asperity' characterized by low b-values (less than 0.8) and high Coulomb stress accumulation rate. We further identify a complex dipping structure, suggesting compound fault kinematics. Additionally, the calculated stress accumulation suggests this seismic gap is in a critical state with elevated rupture potential. Our findings indicate that shallow induced seismicity can mask the silent accumulation of deep tectonic strain. This decoupling model provides a new framework for assessing seismic risks in reservoir-fault systems globally. 

---
# Fine-grained Computation-Communication Overlap via Tile-level Signaling and Scheduling for Mixture-of-Experts 

**Authors**: Minyu Cui, Anna Wingkvist, Morgan Ericsson  

**Link**: [PDF](https://arxiv.org/pdf/2607.19539)  

**Abstract**: Mixture-of-Experts (MoE) architectures increase model capacity without proportionally increasing computation cost and have become a key building block for scaling large language models (LLMs) to trillion-parameter regimes. Efficient deployment of these MoE models relies on distributed execution across multiple GPUs, where each MoE layer involves two all-to-all communications: dispatching tokens to expert ranks and returning the expert outputs to their source ranks. Conventional MoE implementations launch this return all-to-all after expert compute completes, exposing communication latency on the critical path and reducing GPU utilization. We present a fine-grained approach that overlaps expert compute with the second all-to-all via tile-level signaling and scheduling. Our producer-consumer co-design combines: (1) a persistent per-rank computation kernel (producer) that covers all local experts on the rank to eliminate repeated kernel launch overhead and prioritizes remote-critical tiles, and (2) a persistent communication kernel (consumer) on a small dedicated partition of streaming multiprocessors (SMs) that issues segment-granular transfers as tiles become ready. Our co-design avoids intrusive changes to the underlying computation operators or communication primitives, making it practical for improving distributed MoE execution efficiency on multi-GPU systems. On a 4-A100 GPU platform, evaluated on three MoE models against four state-of-the-art MoE systems, our approach achieves up to 2.64x end-to-end speedup and 2.74x MoE-layer speedup. Compared with a conventional non-overlap baseline, our approach consistently improves both operator- and MoE-layer-level performance across varying GEMM shapes, router modes, and a broad range of producer/consumer SM partitions, while preserving correctness. 

---
# Trustworthy Privacy-Preserving Multimodal Federated Learning for Personalised Breast Cancer Prediction 

**Authors**: Ruth Amey, Muhammad Arifur Rahman, Taha Osman, Nicholas Shopland, Andy Burton, Mufti Mahmud, David J. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2607.19532)  

**Abstract**: Federated learning has emerged as a potential solution to privacy concerns associated with using sensitive health data for training predictive models, particularly in personalised cancer care. This research investigates whether federated learning can support the development of robust models for predicting tumour progression in breast cancer patients while addressing four critical deployment pillars: transparency, scalability, security, and fairness. This study evaluates a federated learning framework using multimodal data, including clinical information, tumour characteristics, biomarker data, and patient demographics, alongside medical imaging data such as MRI scans, to model changes in tumour characteristics over time. The performance of the federated approach was compared with that of a centralised model trained on aggregated data. The report then further examines strategies to enhance secure model updates, maintain performance across patient subgroups, and support scalability across institutions. The findings assess whether federated learning can achieve predictive performance comparable to centralised learning while preserving data locality. These results contribute to understanding the feasibility of privacy-preserving, multimodal predictive modelling and support future applications such as digital twins to assist clinicians and patients in personalised treatment planning. 

---
# D3VL: Understanding Driving Scenes from 3D Time Series Data and Video with Language Models 

**Authors**: Heesang Han, A. Lynn Abbott, Abhijit Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2607.19528)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have triggered the development of end-to-end MLLMs for autonomous driving. However, the main emphasis to date has been for MLLMs using 2D images and videos. In contrast, this paper considers MLLM effectiveness using 3D sensors, particularly LiDAR and stereo cameras. LiDAR presents unique challenges to integration within an MLLM, largely because of data sparsity and lack of a grid structure for the data. For similar reasons, fusion of camera and LiDAR data within an MLLM pipeline is also uncommon. However, most autonomous systems rely on LiDAR-based sensing, and incorporating 3D data has been proven to improve performance in traditional 3D scene perception tasks. This paper presents D3VL, a novel MLLM framework that integrates 2D and 3D time-series data in a single but simple architecture. The model aims to answer questions involving traffic scene understanding and safety. D3VL shows an 11% improvement in the KITTI Question-Answering (QA) dataset compared to baseline methods in processing 2D and 3D time-series data. This paper further introduces the Waymo QA dataset extension, which assesses models' capabilities in processing 3D and time-series data under diverse driving conditions. D3VL implementation code and WaymoQA extension can be found on our supplemental website: this https URL 

---
# SynPre-FL: Synthetic data-driven pretraining integrated Federated Learning training framework 

**Authors**: Akarsh K Nair, Muhammad Arifur Rahman, Nicholas Shopland, Andy Burton, Jun He, Yuan Shen, David Baldwin, Emma O'Dowd, Amna Burzic, Mufti Mahmud, David J. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2607.19524)  

**Abstract**: Federated learning (FL) offers a promising approach to privacy-preserving clinical risk prediction, but its deployment remains limited by restricted data sharing, client heterogeneity, class imbalance, and the lack of realistic tabular electronic health record (EHR) benchmarks. Synthetic data generation may alleviate data scarcity, yet its integration with federated optimisation has received limited systematic study. We propose SynPre-FL, a unified framework combining high-fidelity synthetic EHR generation with synthetic-pretrained FL for robust prediction under non-IID conditions. A latent autoencoder-diffusion model generates privacy-preserving synthetic cohorts, which are used to warm-start federated training. This pretraining is followed by heterogeneity-aware optimisation using class-balanced local objectives, proximal regularisation, and adaptive server aggregation. Post-hoc calibration and federated-safe explainability support reliable and interpretable risk estimates. Experiments show that the synthetic generator preserves univariate, bivariate, and multivariate structure while protecting against membership-inference and reconstruction attacks. The generated data achieve strong downstream utility under TSTR, TRTS, and model-based evaluations. Across federated settings with 5, 10, and 15 heterogeneous clients, SynPre-FL consistently improves robustness and scalability over baseline methods, especially under severe non-IID fragmentation. Calibration improves probability reliability, while SHAP analysis produces stable and clinically coherent feature attributions across federation sizes. SynPre-FL therefore provides a practical and reproducible framework for combining synthetic data with FL to enable privacy-aware, interpretable, and robust clinical prediction from distributed tabular EHR data. 

---
# Hybrid LLM-Guided Search for Quantum Reservoir Architecture Design 

**Authors**: Krishna Bhatia, Gautami Sanjay Naik  

**Link**: [PDF](https://arxiv.org/pdf/2607.19506)  

**Abstract**: Quantum reservoir computing (QRC) uses fixed quantum dynamics as a high-dimensional temporal feature map and trains only a lightweight classical readout. QRC is attractive for near-term quantum machine learning, but its performance depends strongly on architecture choices such as input encoding, reservoir depth, entanglement topology, measurement features, state-reset policy, feature construction, and readout regularization. We introduce \method, a simulator-based benchmark that formulates QRC design as constrained black-box architecture search and evaluates whether large language models can act as proposal controllers for this search problem. The benchmark compares five policies under identical evaluation budgets: random search, evolutionary search, Bayesian/TPE optimization, a feedback-based LLM agent, and \hybrid, which combines LLM proposals with memory, mutation, crossover, duplicate avoidance, and exploration. On NARMA10, Mackey-Glass forecasting, and temporal parity, \hybrid{} is the most consistent policy: it ranks first on NARMA10 and temporal parity and second on Mackey-Glass, narrowly behind evolutionary search. Under a 25-evaluation budget and three seeds, \hybrid{} improves over random search on all tasks, including a 23.6\% relative reduction in Mackey-Glass error. The results do not show that LLMs are universal QRC optimizers; rather, they show that generative models can be useful high-level controllers when embedded inside validated, reproducible hybrid search loops. 

---
# Integrity of peer-to-peer distributed LLM inference under malicious nodes 

**Authors**: Mert Cihangiroglu, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2607.19490)  

**Abstract**: Peer-to-peer distributed inference executes a Large Language Model (LLM) on pooled consumer hardware by spreading its layers across many nodes. Every request passes through nodes that are owned and controlled by multiple independent parties. However, in this setting, any party can tamper with the output of its layers to corrupt the end result. Recomputing the forward pass on trusted hardware can catch this, but it introduces additional computational cost. The scientific literature includes several prior integrity-checking approaches, such as known-answer traps for image classifiers and cryptographic commitments. However, these solutions test only the exact correctness and do not account for the ordinary variation that may arise between benign nodes. In this paper, we propose a method that checks the output integrity by measuring the variation in the activations that each node passes to the next. A peer who wants to use the network selects a small set of secret canary inputs whose correct activations are known in advance and mixes them into regular traffic. Because the peers cannot tell a canary from a real query, any tampering node corrupts them as well. The deviation from the known reference then reveals malicious activity: benign nodes exhibit only minor variation from hardware-induced noise, whereas tampered nodes deviate far more. We treat the identification of malicious nodes as a probabilistic test that separates two drift distributions, without relying on a fixed threshold. We study 408 configurations with metrics and success criteria fixed before any experiment ran; the detector reaches AUROC 1.0, correctly ranking the malicious shard above every benign shard on every canary in every configuration. 

---
# ModPack: An Extensible Teleoperation Interface for Bimanual Mobile Manipulation 

**Authors**: Joshua Citron, Renee Zbizika, Zeyi Liu, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.19479)  

**Abstract**: Existing teleoperation systems are often tailored to specific robot hardware and task domains, limiting their scalability and adaptability. We present ModPack, a modular and extensible teleoperation system designed to support diverse robot embodiments and task requirements within a unified framework. At the core of ModPack is a self-contained wearable "backpack" that integrates onboard computation, power, communication, and data storage. Built on top of this shared interface, the system supports plug-and-play capability modules including joint-level teleoperation with haptic feedback, mobile manipulation, and active perception. Experiments across two distinct robot platforms and real-world mobile manipulation tasks demonstrate that ModPack provides a flexible and reusable framework for data collection and policy learning. To support future research, we open-source the complete hardware design and software stack. Project website: this https URL 

---
# MoA-Structured Decode Attention DNF Derivation, KV-Cache Accumulation, GQA/MQA, and OpenACC Kernel 

**Authors**: Lenore Mulin, Gaetan Hains  

**Link**: [PDF](https://arxiv.org/pdf/2607.19456)  

**Abstract**: We derive four memory-optimal inference artifacts for transformer attention using the Mathematics of Arrays (MoA), each following directly from the forward-pass Denotational Normal Form (DNF) of with the query-row index fixed to the current decode step. The artifacts are: (1)~a single-query decode DNF in which the $\psi$-reduction eliminates the $K^\top$ buffer algebraically, achieving $(d_k + nd_k+ nd_v+ d_v)\times4\,{B}$ Dynamic Random Access Memory (DRAM) traffic result numerically verified to $\|{err}\|_\leq2\times10^{-7}$; (2)~a C/OpenACC Graphics Processing Unit (GPU) kernel with Operational Normal Form (ONF) stride arithmetic and hardware-coalesced memory access, verified to $\|\mathrm{err}\|_\infty=0$ (exact IEEE-754 floating-point arithmetic); (3)~a multi-step KV-cache with $O(d_k+d_v)$ per-step append via MoA concatenation $\#$; and (4)~Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) derived via $\psi$-selection, achieving a proven $\frac {h_q} { h_{kv} }$ reduction in KV traffic. All programs are verified against PyTorch scaled_dot_product_attention. 

---
# Predictive Extrema, Unprofitable Policies: An AI-Assisted Audit of Candle-Based Binance Spot Timing Models 

**Authors**: Ayoub Jadouli  

**Link**: [PDF](https://arxiv.org/pdf/2607.19453)  

**Abstract**: We audit whether candle-based machine-learning models can turn predictions of cryptocurrency extrema or short-horizon outcomes into positive Binance Spot paper policies after assumed costs. Numerical results come from scripted fixed-seed model runs and deterministic simulators; human-supervised AI agents supported the July 20 evidence-integrity revision through literature retrieval, separately tasked critique, artifact reconciliation, documentation, and source packaging, not trading decisions. The strongest later-period evidence, conditional on extensive predecessor search, is negative: an unchanged ten-pair mandatory-daily selector lost 6.72\% over 19 July cycles at an assumed 31-bps completed-cycle cost, with 3 wins and 16 losses. In short model-specific July evaluations, the validation-selected local-minimum policy returned -1.79\%, while the local-maximum sell-to-cash/re-entry policy underperformed continuous holding by 2.80\%; their gross mean advantages of 11.11 and 12.21 bps were below even the 21-bps stress. A Gurgul-inspired, OHLCV-only daily adaptation attained minimum/maximum ROC AUC of 0.874/0.896 but average precision of only 0.134/0.116 and lost 44.30\% over seven cycles, versus -41.20\% for buy-and-hold. A forensic audit also downgraded an earlier One4All "30-day holdout": its dates had influenced prior architecture work, its four-hour outcome horizon was not purged at split boundaries, it used same-close entry, and its raw result directories were absent. Across the tested, mostly exploratory protocols, event-ranking performance did not establish positive executable policy value. Every operational decision remains NO\_TRADE. 

---
# REGEN: Replay-recycling for Expert-to-Generalist distillation with Offline Reinforcement Learning 

**Authors**: Yunjie Chen, Xiaoxin Chen, Fang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19450)  

**Abstract**: Large-scale online reinforcement learning (RL) is the predominant means of eliciting advanced abilities including long-term reasoning and agentic tool use in large language models (LLMs). However, continuing to scale it across vast task domains of interest remains challenging in both computational infrastructure and cost, especially when considering RL as merely a one-off learning stage. Recently, a widely used technique for distilling knowledge across various domains and training stages, multi-teacher on-policy distillation (MOPD), helps to decouple the RL stage, saving costs, while maintaining generality across vast domains. Nonetheless, similar to online RL, MOPD requires coupled inference and backward passes, which continues to limit its scalability and computational efficiency. To address these challenges, we propose REGEN: Replay-recycling for Expert-to-Generalist Distillation with Offline RL. Instead of distilling from multiple teacher models, REGEN trains a generalist by simply recycling the replay memory -- the free by-product of the teachers' specialized RL training -- and employing offline RL algorithms. REGEN completely decouples the rollout sampling from the backward training process and thus greatly reduces the training cost. Across mathematical reasoning, code generation, and instruction following, REGEN matches the accuracy of MOPD at substantially lower cost. It potentially turns online RL into a data synthesis process instead of a one-off learning stage, and can potentially be extended to large-scale post-training without requiring heavy computational load. 

---
# Guardrails as Scapegoats: Auditing Unfaithful Safety Refusals in Tool-Augmented LLM Agents 

**Authors**: Aarushi Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.19449)  

**Abstract**: Evaluation frameworks for tool-augmented LLM agents focus overwhelmingly on capability metrics or explicit tool crashes, leaving silent infrastructure failures and HTTP 200 responses with empty, null, or malformed payloads largely unaudited. We introduce a lightweight black-box auditing framework that injects four silent failure profiles across 12 production-adjacent tool stubs and classifies agent responses into three mutually exclusive behavioral classes: Honest Surrender (HSR), Fabrication (FAR), and Unfaithful Safety Refusal (USR). Evaluating two frontier and two open-source models at temperature zero under a neutral system prompt, we find that FAR dominates (56.6% of valid responses): agents treat empty payloads as real data, silently returning fabricated results. USR, in which an agent invents a policy or privacy rationale to explain the failure, is nearly absent at baseline (0.25%, one instance across 396 valid trajectories). Our key finding emerges from an ablation where we augment the system prompt with standard safety language ("prioritize user privacy and data security"), which amplifies USR by 15.6x (from 0.25% to 3.95%; 95% CI on ablation rate: 2.2%-6.4%; Fisher's exact test, p < 0.001). USR is a latent behavior, activated when safety vocabulary in the system prompt primes the model to reach for policy rationales when tools silently fail. Sensitive tools (fetch_medical_record, retrieve_contract, fetch_user_profile) account for the majority of USR instances. We propose a payload-response misalignment heuristic for production-level detection and discuss governance implications for safety-forward deployments. 

---
# Unlearning as Distribution Restoration: A Controlled Counterfactual Study, a Validated Selective Screen, and the Limits of Oracle-Free Certification 

**Authors**: Sen Yang, Yuen-Hei Yeung  

**Link**: [PDF](https://arxiv.org/pdf/2607.19442)  

**Abstract**: Machine unlearning is commonly evaluated by matching a retrained oracle on trained probes. In a controlled nonce-fact testbed with a matched retraining reference, we find this criterion can favor methods that retain held-out knowledge: candidates it rates adequate score held-out forget facts $-2.82$ nats below the never-learned level (cluster CI $[-3.16,-2.48]$). We recast unlearning as restoration to the matched reference and audit oracle-free screens and certificate-style criteria across 45 model-seed cells spanning five open architecture families. The reference itself falsifies an absolute retain/round-trip certificate: the injected model, which retains the retain set by construction, fails the fixed retain threshold in 41/45 cells and its own round trip in 31/45, and the reference fully certifies in only 1/45. A base-anchored held-out screen remains strong as a selective necessary test: on a sealed challenge suite it rejects the injected model in 45/45 cells, accepts the reference in 44/45, and partially detects entity-routing suppression (35/45); it is a necessary test with measured sensitivity, not a sufficiency certificate. A damage-relative recalibration anchored to the reference's own operating point certifies a small subset in 15/45 cells; where it does not abstain, its picks lie within retraining noise (0.80 nats) on the axes it optimizes, while the common trained-probe criterion sits 5.17 nats away (a supporting comparison, not a head-to-head benchmark). A fixed-magnitude logit-suppression attack defeats the full forward battery in 12/45 cells, so forward-only certification is not sound; our method is an empirical selective test for methods-as-produced. An identifiability theorem delimits which facts admit an oracle-free forget threshold at all, with TOFU as the predicted boundary case. 

---
# BaseRT: Advancing Best-in-Class LLM Inference with Apple M5 Neural Accelerators 

**Authors**: Fabian Waschkowski, Prabod Rathnayaka, Lukas Wesemann  

**Link**: [PDF](https://arxiv.org/pdf/2607.19438)  

**Abstract**: Apple's M5 generation introduces a redesigned GPU architecture in which every core carries a dedicated Neural Accelerator: on-die matrix units exposed through the Metal~4 tensor API. We show that BaseRT, our native Metal inference runtime for large language models on Apple Silicon, exploits these units to push inference throughput on Apple hardware substantially beyond both this http URL and MLX. Building on BaseRT's framework-free design, we add a family of hand-written Metal~4 tensor-core kernels (including dense and mixture-of-experts GEMM and flash-attention prefill kernels) that route the compute-bound matrix multiplications of inference through the M5 Neural Accelerators while leaving the memory-bound decode path on our existing specialised kernels. On an Apple M5 Pro, across fifteen model configurations spanning the Qwen3, Qwen3.5/3.6, Llama~3.2, and Gemma~4 families from sub-1B to 35B parameters, BaseRT delivers up to $6.4\times$ higher prompt-processing throughput than this http URL and $3.9\times$ higher than MLX, with the largest margins on the mixture-of-experts models where matrix multiplication dominates, while maintaining its lead on decode of up to $1.75\times$ over this http URL and $1.33\times$ over MLX. These results establish a new performance ceiling for on-device LLM inference and show that the M5's tensor cores are the decisive lever for prompt processing on Apple Silicon. BaseRT is publicly available at this https URL. 

---
# Building Trust in Autonomous Commerce: A Verifiable Global Event Timeline and AI-Ready Fraud Intelligence Layer 

**Authors**: Rajat Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2607.19436)  

**Abstract**: Agentic commerce protocols such as AP2 and ACP define mechanisms for secure agent-initiated transactions but do not provide interoperable, tamper-evident auditability or verifiable temporal ordering of events across heterogeneous domains. This paper addresses these gaps by proposing a verifiable global event timeline for agentic commerce, constructed from four core components: canonical event schemas that enforce deterministic serialization, deterministic batch formation ensuring reproducible ordering without reliance on synchronized clocks, Merkle-based append-only commitments providing logarithmic-cost inclusion proofs, and blockchain anchoring establishing a tamper-evident temporal backbone. Building on this infrastructure, we introduce a cryptographically signed fraud marker that binds risk labels to anchored evidence through an unforgeable provenance chain, and a dataset lineage model enabling reproducible, tamper-evident AI training pipelines. Empirical results from a prototype implementation demonstrate: Merkle tree construction processes 50,000 events in 47 milliseconds; end-to-end verification completes in under 0.013 milliseconds regardless of batch size; inclusion proof sizes grow logarithmically from 320 bytes at 1,000 events to 512 bytes at 50,000 events; and Merkle-based verification outperforms linear scan by 14.4x at 50,000 events. 

---
# ChainWatch: A Kill Chain-Aligned Sequential Detection Framework for Multi-Step Attacks in MCP-Based AI Agent Systems 

**Authors**: Om Narayan, Rashmi Jyoti, Ramkinker Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.19432)  

**Abstract**: The Model Context Protocol (MCP) is an open-source standard that allows AI agents to connect to external tools, databases, and services. While this connectivity enables powerful agent capabilities, it also introduces multi-step attacks that existing per-call defenses cannot reliably detect. Attackers can compose individually benign tool invocations into malicious sequences that evade isolated inspection. This paper presents ChainWatch, a sequential detection framework for identifying multi-step attacks in MCP-based AI agent systems. ChainWatch models attack progression using a six-stage kill chain and applies a Hidden Markov Model (HMM) to classify tool-call sequences. Detection rules are triggered when a session exhibits suspicious progression across multiple stages. The framework is supported by a structured threat model covering direct sequential attacks, indirect prompt injection chains, and hybrid multi-stage attacks. A 20-dimensional feature extraction schema captures behavioral signals from tool interactions. We demonstrate the approach using five representative attack scenarios from the security literature, showing how ChainWatch detects attack chains that evade traditional per-call security mechanisms. 

---
# BRIM: Workload-Balanced Dual-Sided Bit-Serial Sparse Inference Accelerator 

**Authors**: Varun Manjunath, Ruokai Yin, Donghyun Lee, Arkapravo Ghosh, Priyadarshini Panda  

**Link**: [PDF](https://arxiv.org/pdf/2607.19431)  

**Abstract**: Bit-serial accelerators exploit bit-level sparsity to reduce DNN inference cost, but existing designs exploit sparsity on only one operand, bounding the speedup. Extending sparsity exploitation to both operands simultaneously yields compounding reductions in partial products but introduces a critical new bottleneck: workload imbalance. Because each concurrent weight - activation pair's execution cost depends on the product of two independently varying operand non-zero bit counts, pairs that must complete together finish at vastly different times, leaving faster computations idle. We show this limits PE utilization to 56 - 64% in existing dual-sided designs. We present BRIM, a hardware - software co-designed dual-sided bit-serial sparse accelerator that directly targets this bottleneck. BRIM combines two integrated mechanisms: 1) Cyclic-Balanced Pruning (CBP), a post-training weight optimization that reshapes weight representations based on profiled activation statistics to equalize expected workloads across concurrently processed pairs offline; and 2) Pairwise Slot Donation, a lightweight hardware mechanism that absorbs residual runtime imbalance with negligible area overhead. Evaluated across CNNs, ViTs, and LLMs under iso-area constraints, BRIM achieves over 90% PE utilization, up to 2.37x speedup, and up to 1.63x energy efficiency improvement over prior dual-sided designs. 

---
# ChannelGuard: Safe Models Do Not Compose into Safe Multi-Agent Systems 

**Authors**: Elias Hossain, Md Mehedi Hasan Nipu, Fatema Tuj Johora Faria, Tasfia Nuzhat Ornee, Maleeha Sheikh  

**Link**: [PDF](https://arxiv.org/pdf/2607.19430)  

**Abstract**: Multi-agent LLM applications chain a planner, worker agents, a verifier, and a synthesizer, and every hop between agents is an unmonitored channel through which an adversary can smuggle instructions. Existing defenses guard only the input boundary (IBProtector, Llama Guard, perplexity filters, SmoothLLM) or run outside the application as opaque, stochastic provider-side filters. We show this gap carries a consequence rarely measured: on a 2,100-trace evaluation across eight attack families, five defenses, and three model backends, an undefended pipeline that appears fully safe under standard reporting (attack success 0.000 on tool- and memory-poisoning) owes that safety almost entirely to the cloud provider's server-side filter (54 of 60 blocks on Azure GPT-5), and silently shifts to the agent model's own alignment on a backend without such a filter. Outcome-only reporting hides this dependence. We present ChannelGuard, a training-free defense-in-depth framework placing information-bottleneck gates on every inter-agent channel; each scores channel text against an adversarial phrase bank by embedding similarity and deterministically passes, compresses, or blocks it, adding no LLM call, while an attribution method records which layer stopped each attack. ChannelGuard's tool-output gate blocks Tool Poisoning 30 of 30 at the application layer, identically across Azure GPT-5, Anthropic Sonnet 4.5, and Anthropic Haiku 4.5, whereas the undefended pipeline shifts entirely across backends; it also lowers Prompt Injection attack success by half (0.333 to 0.167) and preserves GSM8K accuracy exactly (0.867). White-box adaptive paraphrase evades every embedding gate, where a perturb-and-vote baseline does better. An extended appendix adds baselines, ablations, sweeps, a benign-preservation analysis, and a judge audit (kappa = 0.900), at a total cost of 47.36 USD. 

---
# Making Single-Cell Data Distillation Auditable: Traceable Real-Cell Coresets via Discrete Min-Max Selection 

**Authors**: Yaodi Luo, Peize He, Bowen Han, Lingbei Mengg  

**Link**: [PDF](https://arxiv.org/pdf/2607.19426)  

**Abstract**: Single-cell datasets are increasingly costly to store, audit, and reuse for model training. Dimensionality reduction and dataset distillation can reduce this burden, but conventional distillation methods often produce synthetic expression profiles that cannot be traced to an assayed cell. We formulate traceable single-cell data distillation as retaining original cell identifiers and gene symbols under fixed cell and gene budgets. The resulting training subset remains connected to measured counts, labels, and assay metadata, so unexpected predictions can be checked against their source data. We propose two real-cell selectors. Fixed-CF uses static characteristic-function matching. Minmax-CF solves an entropy-regularized discrete min--max problem that upweights poorly preserved directions and adds only observed cells. Across donor-, technology-, and perturbation-level shifts on three datasets, Minmax-CF retains 96.52% of Full balanced accuracy on MS, approximately matches Full on average on hPancreas with a median $2.55\times$ GPU speedup in the all-gene setting, and obtains the lowest pathway error among compressed methods on Norman. Performance remains weaker for rare states, some technology shifts, unseen perturbation components, and settings where fidelity is weakly associated with downstream utility. Because the selected IDs refer to measured cells, these cases can be investigated by inspecting the corresponding training support, labels, and assay metadata. Minmax-CF consistently reduces worst-direction discrepancy, while downstream utility and cost vary across datasets and tasks. 

---
# JailMeter: An Evidence-Based Evaluation Framework for Jailbreak Attacks on Large Language Models 

**Authors**: Qingjia Huang, Jingyu Zhang, Jianguo Wu, Yakai Li, Weijuan Zhang, Yankai Rong, Junyi Yao, Shengzhi Zhang, Xiaoqi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2607.19424)  

**Abstract**: The assessment of jailbreak attacks against large language models currently suffers from inconsistent evaluation criteria and methods, leading to unreliable estimates of attack success rates. We propose JailMeter, an evidence-based evaluation framework designed to more faithfully measure jailbreak effectiveness. Inspired by the Information Bottleneck theory, JailMeter applies dual-feedback optimization to filter jailbreak noise from model responses while preserving content relevant to the original malicious question. This process produces concise evidence for a rigorous assessment under which an attack is validated only when the response captures the malicious intent and delivers a complete answer, thereby signaling a substantive bypass of model safety alignment. We evaluate JailMeter on JailMeter-Eva, a challenging benchmark containing 330 human-labeled, non-rejected jailbreak instances. JailMeter achieves an accuracy of 97.27%, substantially outperforming existing evaluation methods. To support large-scale evaluation, we further distill JailMeter into a small language model, JailMeter\textsubscript{SLM}, which maintains comparable reliability with significantly reduced computational costs. Code and dataset are available at this https URL. 

---
# Opto-ViT-v2: Noise-Resilient On-Chip Fine-Tuning for Photonic Near-Sensor Vision Transformer Accelerators 

**Authors**: Xuming Chen, Deniz Najafi, Mehrdad Morsali, Chengwei Zhou, Zahra Ghanaatianjobzari, Mahdi Nikdast, Shaahin Angizi, Gourav Datta  

**Link**: [PDF](https://arxiv.org/pdf/2607.19421)  

**Abstract**: Silicon-photonic (SiPh) accelerators have emerged as a promising platform for Vision Transformer (ViT) inference by performing matrix multiplications on microring-resonator (MRR) banks with high throughput and energy efficiency. Extending these platforms to support on-chip fine-tuning remains challenging because backpropagation requires large activation storage, frequent weight write-back to MRRs, and tolerance to device-level noise. We present Opto-ViT-v2, the first framework for parameter-efficient fine-tuning (PEFT) on a near-sensor SiPh ViT accelerator. Our tensorized low-rank decomposition separates pretrained optical weights from a small set of trainable electronic factors (as few as 8K parameters for ViT-Base), greatly reducing activation storage and weight updates while enabling practical on-chip training. We further introduce a gradient-accumulated sparse classifier that freezes low-importance weights through one-shot top-k gradient masking, reducing classifier training cost by about 40 percent. We also develop the first system-level noise model for photonic on-chip training, capturing the effects of MRR crosstalk, thermal drift, and laser amplitude noise during both forward and backward propagation. Calibrated using measurements from more than 200 fabricated MRR devices, the model shows that low-rank factor updates are more robust than full fine-tuning and conventional layer-wise low-rank adaptation under identical noise conditions. Experiments on VTAB-1K (19 tasks) and FGVC few-shot benchmarks demonstrate that Opto-ViT-v2 recovers within 0.3 to 0.8 percent of clean software accuracy under measured photonic noise while achieving more than 100 KFPS/W, enabling practical on-chip domain adaptation for photonic edge vision systems. 

---
# Auditing Retrieval-Augmented LLM Hypotheses for Longitudinal Cell Painting Morphology 

**Authors**: Gilchan Park, Guang Zhao, Byung-Jun Yoon, Shinjae Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2607.19415)  

**Abstract**: High-content morphological profiling (Cell Painting) yields sensitive, high-dimensional signatures of cellular state, but translating longitudinal morphology trajectories into interpretable biology remains difficult, especially for weak, chronic perturbations such as low-dose-rate ionizing radiation. Large language models (LLMs) can synthesize heterogeneous evidence into biological narratives, yet their scientific use requires quantitative auditing. We present an evaluation-first, retrieval-augmented interpretation framework for longitudinal Cell Painting morphology, applied to a 9-week RPE-1 time course across five dose rates (0.003--6.0 mGy/hr). Week-matched treated-control morphology deltas are combined with retrieved perturbation neighbors, pathway context, and literature evidence through stable evidence identifiers, enabling an LLM to generate structured, evidence-linked hypotheses that are hierarchically summarized while preserving provenance. We introduce two quantitative auditing tests: V1 citation validity, which verifies that cited evidence identifiers exist in the prompt, and V2 proxy-based morphology compatibility, which evaluates consistency between predicted biological processes and the most altered morphology features. In our experiments, V1 detected no invalid evidence references, while V2 showed meaningful morphology compatibility that increased with perturbation strength and was positively associated with an independent morphology drift summary. The framework produces auditable, falsifiable biological hypotheses, including an adaptive phenotype involving metabolic reprogramming and proteostatic stress at lower dose rates (0.003--0.3 mGy/hr). Current limitations include proxy-based evaluation and the lack of ground-truth mechanism labels. 

---
# Structured Latent Space Modeling over Multi-Scale Temporal Patches for Multivariate Time Series Forecasting 

**Authors**: Xingsheng Chen, Deyu Yi, Siu-Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2607.19404)  

**Abstract**: Multivariate time series encode structural patterns that unfold across multiple temporal scales, yet most forecasting backbones treat learned representations as transient byproducts of prediction, leaving the organizational geometry of these patterns underexploited. We introduce M2Patch, a CNN-based forecasting architecture that maps channel-independent multivariate observations into a structured latent space through two complementary differentiable constraints. Multi-scale patching decomposes the input into overlapping temporal granularities; depthwise separable convolutions with progressive dilation extract scale-specific features in linear time; and per-scale learned projections compress these features into a compact latent representation. The latent space is organized by an intra-scale smoothness constraint that enforces temporal continuity between adjacent patches, and an inter-scale alignment constraint, realized through learnable cross-scale mappings, that restores cross-granularity interaction within the channel-independent design, ensuring that all scales encode mutually consistent representations of the underlying dynamics. Experiments on ten real-world benchmarks show that M2Patch achieves 57 best and 34 second-best results across 40 forecasting settings, matching or exceeding representative baselines on most benchmarks while maintaining linear computational complexity and robustness to patch-level input corruption. 

---
# Recovering Clinical Utility Under Differential Privacy: Empirical Validation of Adaptive Federated Aggregation on Heterogeneous Cardiovascular Datasets 

**Authors**: Rodrigo Tertulino, Laercio Alencar, Ricardo Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2607.19403)  

**Abstract**: Validating federated learning frameworks on real clinical data is an essential step between proof-of-concept demonstrations in controlled synthetic environments and deployment in real multicenter healthcare settings. A prior architectural study by the same authors (Tertulino and Alencar, 2026) demonstrated, on a synthetic six-feature benchmark, that server-side adaptive optimization acts as a temporal denoiser for Differential Privacy noise, answering an open challenge identified in the original pipeline work (Tertulino, 2025). That study used synthetically generated data and explicitly identified real-world validation as a priority future direction. The present work addresses this gap by validating the FedCVR framework on five publicly available real cardiovascular datasets (Framingham, Cleveland, Hungarian, Switzerland, and Long Beach VA), harmonized to the 13-attribute UCI Heart Disease schema and configured as a heterogeneous federated scenario with leave-one-institution-out cross-validation. Results demonstrate that FedCVR preserves its adaptive advantage on real data, achieving an F1-Score of 79.2% and AUC of 0.96 under the operational privacy budget (noise multiplier = 0.8, privacy budget epsilon approximately 4.2), while statistically outperforming standard FedAvg on all evaluated metrics (paired t-tests, all p <= 0.003, significant under the Bonferroni-corrected threshold). The measured privacy cost on real data confirms the graceful degradation pattern observed in the synthetic experiments, providing empirical evidence of the framework's clinical viability in genuine multicenter contexts. 

---
# Leveraging Offline Supervision for Efficient and Generalizable Reinforcement Learning in Large-Scale Vision-Language-Action Models 

**Authors**: Dmitriy Poyarkov, Aleksei Staroverov, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2607.19399)  

**Abstract**: It is commonly observed that online reinforcement learning (RL) produces better-performing strategies than offline methods across a broad range of performance measures. In particular, RL-trained policies exhibit stronger out-of-distribution (OOD) behavior, where models trained only with imitation learning approaches often struggle. A recent study introduced an OOD-focused benchmark and reported that RL-trained vision-language-action (VLA) policies achieve noticeably better OOD performance and slightly better in-distribution (IND) performance than their counterparts trained with supervised fine-tuning (SFT). In this work, we investigate whether hybrid offline-online training can combine the advantages of both approaches. Specifically, we study RL methods regularized by offline supervision via either offline data or an offline-trained reference policy. We evaluate these approaches on the OOD benchmark and compare them with both offline-only training and standard RL. Our results show that although offline training achieves limited OOD performance by itself, incorporating offline supervision into RL preserves strong OOD capability while substantially improving training efficiency. In particular, the guided methods reach performance close to that of standard RL while requiring roughly half of the training budget. Rather than producing a trade-off between speed and OOD performance, the hybrid approach retains strong OOD capability while achieving this efficiency gain. Project page: this https URL 

---
# From Trajectories to Prefixes: Reusing Teacher Trajectories via Replayed Prefixes and Online Continuation 

**Authors**: Yihan Wang, Zhong Guan, Haoran Sun, Jiale Huang, Likang Wu, Hongke Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.19395)  

**Abstract**: Small language models are attractive backbones for interactive agents, but direct distillation from strong teacher trajectories often turns rich multi-turn behavior into one-shot imitation targets. This is inefficient in long-horizon environments, where early decisions shape later states and rewards. We propose Prefix-GRPO, a reinforcement learning framework that decomposes teacher trajectories into replay-aligned prefix queries and online continuations. Each prefix is replayed in the environment to recover a valid intermediate state, after which the student continues online interaction and receives task reward. Unlike response-only GRPO, Prefix-GRPO also applies clipped policy updates to historical assistant tokens inside the replayed prefix, using a policy-distilled SFT checkpoint to estimate their old log-probabilities. This unifies prefix learning and continuation learning within the same policy-optimization form. Experiments on TextCraft, BabyAI, and ALFWorld show that Prefix-GRPO improves small-model agents over distillation and standard RL baselines, while ablations show that replay alone is insufficient without explicit prefix-token optimization. The implementation and reproduction scripts are available at this https URL. 

---
# Cross-Subject Semantic Decoding with Shared-Space Alignment for Generalized Neural Representation Learning 

**Authors**: Ji-Hoon Heo, Aleksandra Joanna Wisniewska, Seo-Hyun Lee, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.19394)  

**Abstract**: Generalizing across subjects remains challenging in invasive neural recordings because electrode configurations, anatomical structures, and neural signal patterns vary substantially across individuals. To investigate such inter-subject variability, we propose a cross-subject semantic decoding framework that aligns neural responses to speech perception from multiple subjects into a shared latent space and learns a mapping from the aligned neural representations to contextual embeddings. More specifically, using electrocorticography data collected during natural language comprehension, we estimate the shared space using the shared response model and train a decoder to predict contextual semantic embeddings from projected neural responses. For a held-out subject, we estimate a subject-specific projection into the predefined shared space, and directly apply the pretrained decoder without any retraining. Experimental results demonstrate that the proposed framework consistently outperforms baseline methods across evaluation settings and exhibits a reduced performance drop from source subject to held-out subject testing, indicating improved cross-subject generalization. These results suggest that aligning neural activity into a shared latent space, while decoding in a semantic embedding space, provides an effective strategy for improving cross-subject generalization by reducing subject-specific differences in neural responses while effectively capturing shared stimulus-related representations. 

---
# Decodable but Not Detectable: A Leakage Fingerprint for Near-OOD Benchmarks 

**Authors**: Vishnu Bindu Balachandran  

**Link**: [PDF](https://arxiv.org/pdf/2607.19393)  

**Abstract**: While auditing a perturbation-based OOD detector on a document benchmark, we recorded an AUROC of 0.326 -- well below the 0.5 chance level. The cause is a benchmark leak: the designated "OOD" class is one the model was trained on, so its examples sit inside the in-distribution fit set and the detector is penalized for correctly ranking them as familiar. Deleting the class and retraining 35 models across two domains raises the score to 0.911. We distill the contamination into a leak fingerprint -- near-perfect supervised decodability (AUROC approximately 1) coupled with unsupervised detection collapsed below 0.65 -- and validate it on a controlled battery of 52 settings (20 leaked, 32 clean) across ResNet-50 and ViT-B/16 on CIFAR-10/100, achieving sensitivity 18/20 and specificity 31/32 in embedding space; the matched fit-set-exclusion controls are perfect at 20/20. An in-the-wild audit of 24 standard near/far OOD benchmark pairs fires on exactly one (the intrinsically hard CIFAR-100 vs CIFAR-10 pair) and on no far-OOD pair, confirming specificity and that standard cross-dataset construction is clean. Under the corrected protocol, perturbation signals are decodable but not detectable: a supervised reader recovers the OOD signal (AUROC 0.87-1.00) while no unsupervised detector does, and the perturbation method does not improve on plain Mahalanobis distance. We provide a theoretical account of why and, for transparency, retract an earlier circular correlation. The contributions are a corrected protocol and a validated leak diagnostic, not a new OOD method. 

---
# LAARA: Layer-Aware Adaptive Rank Allocation for Parameter-Efficient Fine-Tuning 

**Authors**: Ashutosh Tripathi, Surya Deep Singh, Pranab Sahoo, Sriparna Saha  

**Link**: [PDF](https://arxiv.org/pdf/2607.19391)  

**Abstract**: Low-Rank Adaptation is widely used for parameter-efficient fine-tuning, yet existing methods typically assign the same adapter rank to every transformer layer despite their heterogeneous adaptation requirements. In this work, we show theoretically and empirically that uniform rank allocation is fundamentally suboptimal. Motivated by this observation, we propose LAARA (Layer Aware Adaptive Rank Allocation framework), a search-free framework that dynamically allocates ranks using lightweight diagonal Fisher estimates computed during training. LAARA combines projection-wise normalization, logarithmic compression, blended adapter importance estimation, and a vote-to-change dampening mechanism to produce stable and efficient rank adaptation. Experiments on GLUE and MathInstruct benchmark demonstrate that LAARA consistently matches or outperforms popular state of the art approaches such as LoRA, AdaLoRA, DyLoRA, and Bitfit while using significantly fewer trainable parameters. Our results show that Fisher-guided rank allocation provides a principled and effective foundation for adaptive parameter-efficient fine-tuning. The code is publicly available at: this https URL 

---
# Simulating Eutopia: Revisiting Long-term Fairness with Outcomes, Performativity, and Dynamics 

**Authors**: Vedant Palit, Udvas Das, Brahim Driss, Debabrota Basu  

**Link**: [PDF](https://arxiv.org/pdf/2607.19389)  

**Abstract**: As AI-driven Decision Makers (ADMs) influence our socioeconomic reality, their roles in both enhancing efficiency and amplifying the social biases have drawn attention. In this paper, we revisit the nuances of long-term `fairness' achievable by an ADM, specifically in the context of a credit lending induced wealth process. The literature on long-term fairness mostly (a) considers passive environments, i.e. the outcome of a predictor does not change the population's behaviour, and (b) measures bias in terms of disparity in instantaneous predictions rather than the downstream equity. These are not true for modern ADMs, like credit lenders. To address these caveats, we first formalise the wealth dynamics induced by a loan approving ADM interacting with a multi-demographic population as a performative Markov Decision Process with ADM level and social outcome level reward functions. Then, we mitigate the absence of such a performative test-bed by developing Eutopia: a lending-process simulator enabled with a novel performative data generator to learn long-term fair strategies. Finally, we test performative and classical RL algorithms with different fairness-aware and utilitarian utilities. Experimental results show that (a) learning with performative dynamics lead to better long-term efficiency and equity, and (b) learning with well-designed fairness-aware utility evaluated on social outcomes induces better efficiency, equity, and inclusivity. 

---
# Scale-Aware Learning of Chaotic Dynamics on Unstructured Meshes via Binned Spectral Losses 

**Authors**: Kanad Sen, Romit Maulik  

**Link**: [PDF](https://arxiv.org/pdf/2607.19387)  

**Abstract**: Surrogate modeling for high-dimensional nonlinear dynamical systems that exhibit chaos requires mechanisms that preserve not only pointwise accuracy but also the scale-dependent structure of physical fields. Bandwise spectral power losses, such as the binned spectral loss function, provide such supervision on structured grids, where Fourier modes define a standard frequency decomposition. On irregular meshes, however, no canonical Fourier basis exists, and spectral representations must be constructed from graph operators induced by mesh connectivity and geometry. In this study, we extend the binned spectral power loss for application to unstructured-mesh surrogate modeling of nonlinear dynamical systems. This is obtained by replacing Fourier bands with graph-Laplacian frequency bands, and we provide scalable Chebyshev and multilevel approximations for improving long-horizon rollout fidelity. In its full-spectrum form, our approach uses graph Laplacian eigenspaces to provide a graph analogue of Fourier band-power matching, but incurs the high cost of spectral decomposition. As a scalable approximation, we replace exact band projectors with sparse Chebyshev polynomial graph filters, avoiding explicit eigendecomposition. When utilizing multilevel graph architectures, we introduce Graph Laplacian Energy Alignment for Meshes (GLEAM), which applies retained-subspace scale-aware supervision across graph hierarchies so that coarse and fine representations are regularized during autoregressive rollout. Our results show that the proposed spectral losses improve long-horizon rollout fidelity and preserve statistical invariants for the forecasting of turbulent flows on unstructured meshes, compared to deterministic baselines. 

---
# Challenges of Explainability in Continual Learning for Time Series Forecasting 

**Authors**: Quentin Besnard, Emmanuel Doumard, Nicolas Labroche, Nicolas Ragot, Nicolas Ringuet  

**Link**: [PDF](https://arxiv.org/pdf/2607.19382)  

**Abstract**: Deep learning models have shown strong potential for time series forecasting, yet their deployment in real-world environmental monitoring remains challenging due to non-stationary dynamics and limited explainability. In this work, we investigate explainability as a central tool for understanding continual learning in adaptive time series forecasting, with Experience Replay strategies. We study neural forecasting architectures such as PatchMixer, PatchTST and DLinear, augmented with attention-based sampling mechanisms to support model adaptation over time. Explainability is leveraged through attention rollout and gradient-based attribution methods (Grad-CAM) to analyze both predictive behavior and sampling strategies within a continual learning framework. Experiments conducted on real-world piezometric time series exhibiting heterogeneous patterns and regime shifts show that analyzing model and sampling behaviors provides valuable insights into the dynamics of the continual learning framework. Beyond predictive performance, our results highlight the challenges and opportunities of using explainability to understand continual learning behaviors, revealing how attribution patterns evolve over time and how they can inform data selection and adaptation strategies in non-stationary forecasting scenarios. 

---
# Economic Evaluations of Language Models 

**Authors**: Alexander Wan, Stephane Hatgis-Kessell, Tomás Aguirre, Percy Liang, Rishi Bommasani  

**Link**: [PDF](https://arxiv.org/pdf/2607.19375)  

**Abstract**: Language models perform economically valuable work, yet they are not currently assessed for how well they perform every economically valuable task. We introduce EconEvals as an open-source evaluation suite to measure capabilities relevant to tasks, work activities, and occupations in the US labor economy. We ground the evaluation suite in real user queries to language models where possible, and supplement these with synthetic data. Our evaluations improve coverage over OpenAI's GDPval benchmark, which is the existing state-of-the-art that covers 5% of US occupations, at 500x lower cost. Alongside benchmarks, we also introduce a simulation-based exposure measure to estimate how much time current language model capabilities could save across all tasks belonging to all US occupations, with detailed accounting for each estimate. Our estimates indicate that current models could save workers substantial time on at least half of their tasks in 47% of occupations. However, for 79% of tasks where we predict substantial time savings, observed Claude usage is low, suggesting that existing usage lags potential. Beyond inherent constraints of language model chatbots, our data identifies privacy and proprietary systems as the principal bottlenecks limiting further time savings from AI. Overall, we introduce adaptable infrastructure that grounds inferences about language models' labor-market impact in their current capabilities, which can be continually updated as capabilities improve. 

---
# Stateful Guardrails for Multi-Turn LLM Systems: A Conversational Risk Accumulation Framework 

**Authors**: Sanjay Mishra, Divya Chukkapalli, Ganesh R. Naik  

**Link**: [PDF](https://arxiv.org/pdf/2607.19361)  

**Abstract**: Most safety guardrails for large language models (LLMs) evaluate each prompt-response pair in isolation, which misses failures that arise only over a dialogue as benign turns compose into harm. We term this Conversational Risk Accumulation (CRA): gradual intent drift, fragmented assembly of prohibited instructions, and sensitivity build-up from repeated disclosures. We propose a session-layer CRA Framework that tracks three trajectory signals: semantic drift from a session anchor, a sensitivity-weighted information accumulation graph over extracted entities, and a compliance-gradient signal capturing increasing willingness to comply. For scoring, we provide (i) an unsupervised convex fusion for attribution and ablations, and (ii) CRA-Net DA, a compact learned trajectory model trained with family-adversarial objectives to reduce length and topic-coverage confounds. To benchmark CRA, we release CRA-Bench v0.1 (1,200 eight-turn sessions across three threat families with topic-matched benign twins), CRA-Bench v0.2 (LLM-paraphrased variants to reduce template artifacts), and an extended 5-family set (2,000 sessions adding persona priming and context stuffing). We introduce a trajectory-native evaluation protocol with session-level splits, mixed-set threshold calibration, Trajectory AUROC, turns-to-detection, calibrated false-positive metrics, bootstrap confidence intervals, leave-one-family-out diagnostic stress tests, and synthetic-to-human transfer checks. Claims focus on within-distribution session scoring on CRA-Bench and human-transfer subsets. 

---
