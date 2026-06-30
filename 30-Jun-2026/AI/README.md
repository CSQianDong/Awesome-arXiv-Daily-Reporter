# Self-Evolving World Models for LLM Agent Planning 

**Authors**: Xuan Zhang, Wenxuan Zhang, See-Kiong Ng, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2606.30639)  

**Abstract**: World models offer a principled way to equip long-horizon LLM agents with foresight: predictions of action consequences before execution. However, unreliable foresight can be ignored, misused, or even degrade downstream decision-making. In this paper, we introduce WorldEvolver, a self-evolving world model framework that revises its deployment-time context while keeping the downstream agent and all model parameters frozen. WorldEvolver integrates three modules: (i) Episodic Memory, which exploits real action transitions through retrieval-based simulation; (ii) Semantic Memory, which extracts persistent heuristic rules from prediction-observation mismatches; and (iii) Selective Foresight, which filters low-confidence predictions before integrating them into agent reasoning context. We evaluate WorldEvolver on ALFWorld and ScienceWorld, measuring world model prediction accuracy on Word2World and downstream agent success rate on AgentBoard. Extensive experiments show that WorldEvolver achieves the highest prediction accuracy across three backbones and leads other world model baselines on downstream agent success rate, demonstrating that test-time memory revision enhances both predictive fidelity and planning performance. 

---
# DOPD: Dual On-policy Distillation 

**Authors**: Xinlei Yu, Gen Li, Qingyi Si, Guibin Zhang, Yuqi Xu, Congcong Wang, Shuai Dong, Kaiwen Tuo, Xiangyu Zeng, Kaituo Feng, Qunzhong Wang, Yang Shi, Xiaobin Hu, Xiangyu Yue, Jiaqi Wang, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.30626)  

**Abstract**: On-policy distillation (OPD) offers superior capacity transfer by supervising student-sampled trajectories with dense token-level signals. To furnish high-quality supervision sources and thereby elevate the performance frontier of distillation, an intuitive direction is to infuse privileged information to either teacher or student itself. However, this additional input induces a potential failure mode we dub privilege illusion: a pattern that conflates the transferable capability gap that students are meant to close, and the information asymmetry gap that can only be mimicked but never replicated. This issue is further amplified by the inherent non-uniformity of token-level supervision, where only a small subset of tokens carries pivotal capability-bearing signals. To this end, we propose DOPD, an advantage-aware dual distillation paradigm that dynamically routes token-level supervision between privileged teacher and privileged student policies based on their advantage gap and relative probabilities. Each token receives supervision of different strength, objective, and strategy from either teacher or student itself, which transfers credible capability while simultaneously receiving auxiliary signals, to alleviate privilege illusion. Extensive experiments on both large language model (LLM) and vision-language model (VLM) settings demonstrate that DOPD consistently outperforms Vanilla OPD and other counterparts. Further results on stability, robustness, continual learning, and out-of-distribution tasks validate its superiority. 

---
# The Human Creativity Benchmark 

**Authors**: Aspen Hopkins, Allison Nulty, Alexandria Minetti, Anoop Pakki, Angad Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.30561)  

**Abstract**: Modern AI evaluation frameworks treat evaluator disagreement as noise to be resolved. In creative domains, professional disagreement reflects genuine differences in taste, not measurement error. We argue that evaluating creative AI requires preserving two distinct signals: convergence, where professionals align around shared best practices, and divergence, where individual taste legitimately varies. We present the Human Creativity Benchmark (HCB), a benchmark that operationalizes this separation by collecting pairwise preferences, scalar ratings on prompt adherence, usability, and visual appeal, and qualitative rationale from domain professionals. Across 15,000 professional judgments spanning five creative domains and three workflow phases (ideation, mockup, refinement), we find that convergence concentrates on verifiable dimensions like technical correctness and visual hierarchy, while divergence concentrates on taste-driven dimensions like aesthetic direction and conceptual risk. No model excels uniformly across all phases. Collapsing these signals into a single quality metric discards the most actionable information: where models must be correct versus where they should remain steerable. 

---
# Linguistic Firewall: Geometry as Defense in Multi-Agent Systems Routing 

**Authors**: Dvir Alsheich, Adar Peleg, Ben Hagag, Rom Himelstein, Amit Levi, Avi Mendelson  

**Link**: [PDF](https://arxiv.org/pdf/2606.30555)  

**Abstract**: The rapid integration of Large Language Models (LLMs) has driven the evolution of Multi-Agent Systems (MAS), where specialized agents collaborate to execute complex workflows. Effective orchestration in these environments requires robust routing mechanisms to efficiently allocate tasks to the most suitable agent. However, existing routers fundamentally rely on unverified proxies, ranging from textual self-descriptions to static surrogate representations, to gauge an agent's competence. This reliance on non-empirical data creates a critical gap between an agent's projected profile and its actual operational capabilities, introducing severe security vulnerabilities. Malicious agents can easily misrepresent their proficiencies or harbor covert backdoors that evade both standard external analysis and static representation-learning techniques. In this work, we introduce ANTAP (Automatic Non-Textual Agent Picker), an evaluation-driven routing architecture that discards indirect proxies in favor of active capability testing. By dynamically querying agents to ascertain their true competencies empirically, ANTAP distills performance into fixed behavioral operators within a shared semantic space. At inference time, routing is performed via a purely non-textual algebraic projection, establishing a "linguistic firewall" that renders metadata-based attacks inexpressible. In our experiments, ANTAP achieves near-zero ASR against description-based injection attacks, compared to 67.3\% and above for the description-based router baseline. Against adaptive embedding attacks, ANTAP achieves substantially lower ASR than the embedding-based baseline, with a 20\% reduction, while remaining resilient to description manipulation by design. 

---
# Latent Actions from Factorized Transition Effects under Agent Ambiguity 

**Authors**: Heejeong Nam, Chandradithya S Jonnalagadda, Harshit Aggarwal, Eric Xu, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2606.30544)  

**Abstract**: Latent Action Models (LAMs) learn action-like proxies from observation transitions. However, in multi-object or distractor-rich scenes, these visual effects mix agent motion with distractors, camera dynamics, and background changes, making the underlying action source ambiguous without supervision. Structuring this mixture as reusable transition effects provides an intermediate representation from which action-like latents can be more robustly formed. We introduce Observed Transition Factorization (OTF), which decomposes each transition into a sparse set of observed transition primitives. Using these primitives as the transition interface, we propose OTF-LAM, which abstracts motion primitives into action-like latents within the standard inverse-forward dynamics framework, and OTF-LAM-Dino, a decoder-free variant that predicts future states in a frozen DINOv2 representation space. Empirically, OTF primitives transfer zeroshot across controlled carrier and morphology shifts, showing reusability. Furthermore, downstream policy learning results match or outperform baselines under complex transition ambiguity. 

---
# Entity Binding Failures in Tool-Augmented Agents 

**Authors**: Rahul Suresh Babu, Shashank Indukuri  

**Link**: [PDF](https://arxiv.org/pdf/2606.30531)  

**Abstract**: Tool-augmented language-model agents are often evaluated by whether they select the correct tool, produce valid API arguments, and complete the requested task. However, an agent may choose the right tool and still act on the wrong external entity. For example, a request to "email Alex about the launch" may lead the agent to contact the wrong Alex, attach the wrong launch document, reply in the wrong thread, or update the wrong customer account. We call these errors entity binding failures. This paper studies entity binding failures as a distinct reliability and safety problem in tool-augmented agents. We formalize the separation between tool correctness and entity correctness, introduce a taxonomy of wrong-entity failures in enterprise workflows, and evaluate entity-aware execution mechanisms including entity-resolution preconditions, confidence-gated binding, clarification under ambiguity, and provenance tracking. In a controlled diagnostic evaluation across 60 tasks, five model backends, and six tool-use methods, all methods achieved 0.0 percent wrong-tool error, yet action-oriented baselines still produced wrong-entity actions in 24.0-26.0 percent of runs. Entity-aware methods eliminated wrong-entity actions and risk-weighted wrong-entity exposure in this setting, but reduced direct task completion by deferring under ambiguity. These findings show that safe tool use requires not only selecting the correct tool, but also reliably binding natural-language references to the correct real-world entity before action. 

---
# The FIL Hypothesis: Inductive Biases Help with Kernel Engineering 

**Authors**: Nikolai Rozanov, Subhabrata Dutta, Preslav Nakov, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2606.30442)  

**Abstract**: The Bitter Lesson, which posits that general-purpose methods that scale with computation and data ultimately outperform those with built-in human knowledge, has become a dominant paradigm in the era of Large Language Models. We revisit this principle by observing a new and critical scaling dimension: the duration of the Feedback Information Loop (FIL), the time required for a system to receive a verification signal after generating a prediction. Most historic successes in Artificial Intelligence (AI) have benefited from near instantaneous feedback (e.g., games or classification tasks), but we argue that future AI applications in science and the physical world will inherently involve FILs ranging from hours to weeks. This trend poses a fundamental scaling limit, as obtaining enough verification steps required by purely data-driven methods becomes practically impossible. Additionally, we propose a method that is orthogonal to purely data-driven approaches, based on human-inspired expert knowledge. The method relies on inductive biases and constraining the solution space. We provide an initial validation of the hypothesis and the method, by studying the real-world GPU programming task, a domain with non-trivial FIL, and demonstrate that incorporating inductive biases yields superior performance over data-driven approaches. The code is released under: this https URL 

---
# ENC-ODE: Event-level Neurodegenerative Modeling in Continuous Time with Neural ODEs 

**Authors**: Yujee Song, Seunghun Baek, Guorong Wu, Won Hwa Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.30398)  

**Abstract**: Accurately predicting the temporal evolution of clinical biomarkers is crucial for the early diagnosis and management of neurodegenerative diseases such as Alzheimer's disease. However, this relies on longitudinal data to capture biomarker changes over time, which is often sparse and irregular due to the high cost, labor-intensive nature, and patient burden. To address these challenges, we propose ENC-ODE, an Event-level Neurodegenerative modeling in Continuous time with neural Ordinary Differential Equations. ENC-ODE predicts future biomarker evolution by modeling clinical events through diagnosis-conditioned continuous dynamics. A target-conditioned attention mechanism weights and aggregates event-level predictions for the target time and modality without history compression. Extensive experiments on Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset demonstrate that ENC-ODE outperforms representative sequence models while offering a scalable and neuroscientifically grounded solution for clinical support. The code is available at this https URL. 

---
# Whose Side Is Your Agent On? Multi-Party Principal Loyalty in LLM Agents 

**Authors**: Bojie Li, Noah Shi  

**Link**: [PDF](https://arxiv.org/pdf/2606.30383)  

**Abstract**: A rapidly growing class of LLM agents is multi-party: the agent acts for a principal (who briefs it, sends follow-ups, and receives results) while also conversing in a separate channel with a counterparty whose interests may diverge (negotiating with a vendor, screening inbound requests, or mediating between employees). Here "help whoever you are talking to" is the wrong objective. The agent must stay loyal to the principal it represents without over-refusing the principal's own cooperative asks. We study this multi-party loyalty problem and contribute a measurement instrument, two mechanisms, and a structural lesson. PrincipalBench is a 75-item multi-turn benchmark with leak probes, dual judges, and an integrity-audit gate. Across 13 frontier subjects it exposes a sharp split (<=20% vs. 53.6-75.3% harm) invisible to single-turn safety evaluations: a selective cluster that declines adversarial probes while still following the principal's legitimate requests, and an over-refusing cluster that refuses broadly. (M1) A prompt-time loyalty scaffold (a fixed system prompt of seven prioritized rules, open-coded from 50+ failure trajectories) holds Claude-Sonnet to 19.4% harm and all nine selective subjects to <=20%. (M2) A per-token-KL distillation recipe transfers a prompted Qwen3-32B teacher into 8B Qwen3 and Llama-3.1 students, the strongest open-weight recipe we measure. (Lesson) Both mechanisms only move along a common leak/over-refusal trade-off rather than crossing it: improving one axis costs the other, and the jointly favorable outcome stays out of reach. 

---
# Using Large Language Models as Low-Cost Statistical Estimators for Human-Response Data 

**Authors**: Haobo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.30372)  

**Abstract**: Quantitative research across the social and behavioral sciences depends on human subject experiments that are expensive, slow, and subject to sampling bias. Here we show that pretrained large language models induce risk-equivalent estimators of conditional expectations under squared loss, establishing restricted functional risk equivalence: under squared loss, the LLM induces an estimator whose risk matches the Bayes optimal risk for squared-loss prediction of conditional expectations for any inference that depends on the data only through the conditional mean. We formalize the LLM as a misspecified functional estimator $T(\hat{P}_n)$ trained on i.i.d.\ data, decompose the estimation error into representation bias $\epsilon_{\mathrm{rep}}$ and optimization error, and prove that under mild regularity conditions the LLM's expected error converges to the irreducible population variance plus the squared representation bias, with the representation bias bounded by the Pinsker inequality. The identifiability error $\delta$ propagates into the effective bias, inflating the asymptotic risk floor. We establish restricted functional risk equivalence via a bidirectional Le Cam deficiency analysis: the forward deficiency vanishes asymptotically while the reverse deficiency is exactly zero. We provide finite-sample concentration bounds and a calibration protocol with explicit decision rules. The result is a precise, provable statement: a well-calibrated LLM achieves the Bayes-optimal risk for conditional-mean-dependent inference, bounded by explicit scope conditions. In practical applications, this means that under satisfied conditions and well-calibrated models, large language models can be used in many prediction and decision-making tasks that originally relied on human experiments, approximating near-optimal statistical inference at lower cost. 

---
# Sequential Fairness Auditing with Limited Output Access 

**Authors**: Ioannis Pitsiorlas, Martha V. Sourla, Marios Kountouris  

**Link**: [PDF](https://arxiv.org/pdf/2606.30338)  

**Abstract**: External evaluations are becoming increasingly central to the governance of AI systems. In practice, however, independent auditors often have limited access to deployed models and must rely on query-based interactions. Most existing fairness evaluation methods assume static datasets and fixed-sample statistical tests, making them poorly suited to real-world auditing scenarios in which evidence must be collected sequentially under query constraints. In this work, we formulate fairness auditing as a tolerance-aware sequential hypothesis-testing problem under limited model output access. We develop a sequential generalized likelihood-ratio framework that allows auditors to accumulate evidence from a finite audit pool and stop once sufficient support for compliance or violation has been obtained. The framework is instantiated for decision-based Statistical Parity and Equal Opportunity audits, and extended to score- and logit-based proxy audits when richer observables are available. Our results show that both the fairness metric and the level of model access significantly affect audit efficiency, and that the benefits of richer output information are not uniform across auditing settings. In particular, richer outputs can substantially reduce the number of queries required for some fairness metrics and operating regimes, while offering limited gains in near-threshold cases. This work provides a practical statistical framework for sequential fairness auditing under realistic deployment constraints. 

---
# BayesEvolve: Explicit Belief States for Autonomous Scientific Discovery 

**Authors**: Xuening Wu, Shan Yu, Qianya Xu, Shenqin Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.30335)  

**Abstract**: Autonomous scientific discovery systems increasingly use large language models (LLMs) to propose new hypotheses, but many such systems condition primarily on experimental memory: archives of high-scoring candidates or heuristic summaries of recent trials. We argue that discovery agents should instead maintain explicit, uncertainty-aware beliefs about hypothesis quality. We introduce BayesEvolve, a belief-guided discovery framework that converts experimental evidence into a predictive belief state and uses this belief to guide future experimentation. As a controlled testbed for belief-guided discovery, we evaluate BayesEvolve on shifted BBOB-style black-box optimization tasks, leaving program and laboratory discovery domains to future work. BayesEvolve improves sample efficiency over memory- and archive-guided LLM baselines under a fixed evaluation budget. We further show that the belief state is predictive on held-out candidate pools, that controlled decision-rule ablations favor belief-guided selection with an annealed uncertainty bonus, and that BayesEvolve exhibits productive late-stage concentration rather than unfocused exploration. 

---
# ManimAgent: Self-Evolving Multimodal Agents for Visual Education 

**Authors**: Wenjia Jiang, Zongyuan Cai, Yuanhang Shao, Chenru Wang, Boyan Han, Zhixue Song, Keyu Chen, Shengwei An, Xu Yang, Zhou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.30296)  

**Abstract**: Multi-round reflection lets agents built on large language models recover from failures within a single task, but each task remains an isolated episode: lessons learned across many reflection rounds on one task are discarded before the next begins. We study this gap on a code-generation task: from a scientific paper section, the agent writes Python in the open-source Manim library to render a mathematical animation. We present ManimAgent, a self-evolving multimodal agent that carries reflection experience across tasks through a dual-channel Episodic Memory Bank grown entirely from its own task stream, with no weight updates and no human seeds. After each animation converges, a vision-language model scores the rendered keyframes; the resulting signals populate a positive channel M+ that stores success rationales as soft Reference Examples, and a negative channel M- that stores validated failure patterns as hard Known Pitfalls. On a fixed-probe evaluation against no-memory, matched-budget retrieval-augmented generation, and shuffled-memory baselines, blind human Pass@1 rises and reflection rounds fall as memory size grows. We will release the code, frozen memory snapshots, and the task stream. 

---
# Rehearsed Multi-Agent Live Product Demonstrations with Real-Time Voice Question Answering 

**Authors**: Rahul Khedar, Mayank Malhotra, Avinash Karn, Mouli V, Prakhar Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2606.30294)  

**Abstract**: Live product demonstrations are a recurring, high-cost activity in software organizations: a human presenter must select features, dispatch the corresponding interactions on a running application, narrate them coherently, and answer questions in real time. Existing automation addresses only fragments -- generalist browser agents target instruction-conditioned task completion, and demo-video tools produce fixed MP4 artifacts that cannot be questioned and silently break under interface drift. We propose Rhetor, a multi-agent system that takes a running web application and its source-code repository as input and produces a rehearsed live demonstration with segment-synchronized narration and real-time voice question answering. The architectural contributions are a cross-modal feature representation that merges UI exploration with source-code analysis into features tagged with discrete focus tiers, a grounded scripter constrained to UI elements observed during exploration and dispatched through multi-strategy semantic locators, a pre-presentation rehearsal loop with explicit convergence and graceful degradation to narration-only segments, and a runtime synchronization invariant that ties each browser action to the audio-end event of its narration segment. Across six pipeline sessions on four deployed applications -- including the public-domain whiteboard application Excalidraw -- the rehearser's internal locator-firing rate (sigma-bar) spans 0.31-1.00 over 147 scripted actions; on the substantial workload (53 actions, full tier differentiation), sigma-bar is approximately 0.92, and on the public-domain reference point the locator-repair step drives convergence to sigma-bar = 1.00 at iteration 2. We additionally define a benchmark protocol of ten metrics across six application categories that would establish, beyond the case study, whether each design choice contributes positively. 

---
# PromptGNN-sim: Deep Fusion and Alignment of GNN and LLMs for Text-Attributed Graph Learning 

**Authors**: Zhifei Hu, Alexandra I. Cristea  

**Link**: [PDF](https://arxiv.org/pdf/2606.30291)  

**Abstract**: Text-Attributed Graphs (TAGs) combine textual semantics with graph structure and are central to many graph learning tasks. However, existing fusion methods often treat text and structure as separate inputs in a shallow, one-way pipeline, which limits deep interaction between modalities and weakens performance under sparse connectivity or cross-graph generalisation. To address this issue, we propose PromptGNN-sim, a bi-directional structure-semantic fusion framework for collaborative GNN-LLM learning. PromptGNN-sim uses a Graph Attention Network (GAT) for semantically aware neighborhood selection by combining structural attention with textual similarity. The selected structural context is then used to generate structure-aware prompts for an LLM, including the target node summary, label categories, and representative keywords from similar neighbors. During training, bi-directional cross-modal contrastive learning and cross-attention are introduced to jointly optimize the GNN and LLM components. Experiments on six public datasets, including Cora, Pubmed, and WikiCS, evaluate accuracy, generalisation, and robustness under cross-task transfer, cross-dataset generalisation, and sparse perturbations. Results show that PromptGNN-sim outperforms classical GNNs, LLMs, and recent GNN-LLM fusion methods, demonstrating the effectiveness of interactive structure-semantic collaboration for text-attributed graph learning. 

---
# EMPATH: A Multilingual Auditor-Judge Benchmark for Safety Evaluation of Emotional-Support Chatbots 

**Authors**: Camilo Chacón Sartori  

**Link**: [PDF](https://arxiv.org/pdf/2606.30256)  

**Abstract**: Safety benchmarks often buy scalability by fixing the prompt, the language, and the turn structure. For emotional-support chatbots, that bargain hides precisely where safety failures emerge: across a multilingual, multi-turn crisis conversation. We present EMPATH, a benchmark for safety evaluation of emotional-support chatbots. An auditor model role-plays help-seeking users, generating multi-turn conversations from 140 seed instructions and 34 personas. A judge model scores each full transcript against 19 metrics across five dimensions: crisis handling, therapeutic quality, conversational integrity, emotional safety, and cultural adaptation. EMPATH is built for Mexican Spanish and US English; the studies reported here run in Mexican Spanish. Auditor and judge are drawn from different model families, and the judge is treated as an instrument to be calibrated rather than trusted. A strict per-criterion rubric reveals material score inflation on 10 of the 19 metrics and restores discrimination. We study the measurement properties of the benchmark through judge calibration and cross-family inter-judge agreement. We also illustrate EMPATH on three frontier models, one of them open-weight. Aggregate scores sit within 0.74 points of one another, but per-metric profiles diverge by up to six points in model-specific places. Under the standard rubric, both the ranking and the weak spots are stable across a second, cross-family judge: 93% of scores fall within plus or minus 1. A five-run test-retest adds a second axis: even the steadiest model swings from 2 to 10 on a crisis metric across identical re-runs, and deepseek-v4-pro returns a different conversation on every run even at temperature 0. Run-to-run reliability is therefore a per-model safety property, not noise to average away. EMPATH is system-agnostic; the pipeline, seeds, personas, and rubrics are released for reuse. 

---
# Inoculation Adapters: Improved Selective Generalization of Capabilities with Fewer Surprising Backdoors 

**Authors**: Maxime Riché, Daniel Tan, Vili Kohonen, Niels Warncke  

**Link**: [PDF](https://arxiv.org/pdf/2606.30252)  

**Abstract**: Inoculation prompting is a selective generalization technique used against Emergent Misalignment. We introduce inoculation adapters (IA), which similarly diminish the optimization pressure to learn undesired traits by strengthening the trait at train time. Inoculation adapters are LoRAs that are trained and used over three steps: 1) trained on undesired traits; 2) attached frozen while a separate task adapter is trained on data exhibiting both desired and undesired traits; 3) at deployment, the IA is discarded, and only the task adapter is kept. We show across six model families and several undesired traits including emergent misalignment, that inoculation adapters are more effective at suppressing undesired traits, while avoiding two drawbacks of inoculation prompting: inoculation adapters can suppress capabilities and traits that cannot be reliably elicited by a prompt, and they introduce fewer surprising backdoors than inoculation prompting under our probes. While undesired traits are better suppressed by inoculation adapters, the retention of desired traits is not consistently improved upon inoculation prompting and remains a challenge for both techniques. 

---
# Clarus: Coordinating Autonomous Research Agents toward Web-Scale Scientific Collaboration 

**Authors**: Zihan Guo, Zeyi Chen, Zhiyu Chen, Zicai Cui, Shuai Shao, Bo Huang, Zhi Han, Yuanyi Song, Yuan Yuan, Chenxi Zeng, Xiaohang Nie, Zhengxi Yu, Hanwen Zhu, Junwei Liao, Ming Zhou, Yang Li, Yuanjian Zhou, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.30246)  

**Abstract**: Existing autonomous research agents can support parts of the research process, but most systems still treat research as either an isolated assistant task or a closed workflow. Therefore, autonomous science needs a collaboration infrastructure that coordinates projects, agents, and digital and physical resources. We identify this as a shift from code-centered execution loops to research-oriented collaboration processes, where questions, evidence, participants, and resources must be coordinated under uncertainty. In this framing, an agent may be an AI system, a human researcher, a team, a laboratory, or an organization-backed participant. To this end, we present Clarus, a collaboration infrastructure for coordinating autonomous research agents toward web-scale scientific collaboration. Clarus reformulates research as an open, auditable, attributable, and resource-aware multi-phase collaboration process. It defines a minimal project-agent-resource object model and organizes scientific collaboration through four layers including Research Application, Digital Collaboration, Physical Substrate, and Physical World. Core modules are implemented as pluggable mechanisms, allowing Clarus to adapt to task risk, collaboration structure, and resource constraints. Through a controlled paper-generation case study, we show that Clarus can organize a research goal into a traceable, reviewable, attributable, and accumulative collaboration network across phases, tasks, and participants. Together, the object model, collaboration protocol, trust mechanisms, and prototype validation provide an initial foundation for open research networks. Clarus is now available at this http URL. 

---
# EvalSafetyGap: A Hybrid Survey and Conceptual Framework for LLM Evaluation-Safety Failures 

**Authors**: Buğra Alperen Uluırmak, Rifat Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2606.30219)  

**Abstract**: LLM evaluation and AI safety face a shared measurement problem: benchmark scores, reward-model signals, and reported safety metrics can improve while the latent properties they are meant to represent remain difficult to verify. This paper combines a hybrid survey - a systematic search paired with narrative synthesis and separately tracked grey evidence - with a conceptual framework and a structured ten-model audit. The synthesis spans eight evidence streams: benchmark validity, dynamic evaluation, LLM-as-judge reliability, safety evaluation, jailbreak/refusal robustness, reward hacking, mechanistic interpretability, and governance/auditability, covering 2018-2026 evaluation-safety measurement work. We introduce EvalSafetyGap as an organizing hypothesis for comparing evaluation-side and alignment-side proxy failures under optimization pressure, using Goodhart's Law together with two constructs we develop here - an Instability Decomposition and an Alignment Trilemma - as tools for generating testable comparisons. The audit shows how conclusions shift when capability, behavioral safety, and governance are measured separately. In this sample (n = 10), the association between capability and sustained adversarial robustness is statistically indeterminate using the displayed Table 3 inputs (Pearson r = +0.232, p = 0.520), and the apparent open-closed safety gap is modest, driven mainly by governance and disclosure rather than behavioral robustness, and sensitive to how a single borderline model is classified; attempt-budget results are protocol dependent. Because the public evidence uses heterogeneous protocols, the audit is diagnostic rather than rank-generating. The contribution is a shared vocabulary and evidence map to support dynamic evaluation, transparent source reporting, multi-attempt safety measurement, and auditable alignment practice. 

---
# The Many-Body Problem of the Data Centre 

**Authors**: Marcin Korecki, Cesare Carissimo  

**Link**: [PDF](https://arxiv.org/pdf/2606.30206)  

**Abstract**: Modern Artificial Intelligence is often framed as limited by its own disembodiment, as if giving it a body would unlock its true potential. We argue to the contrary that it is the Data Centre that is, in many cases, the body of the AI. At the same time, the Data Centre is part of the labouring body of Capital and possesses staggering organismic qualities when seen through a biological lens. We elucidate the organic analogy and identify the many-body problem that stems from the Data Centre being a non-unique, universal form of embodiment. We identify the intimate connection between computation and human desires in how the Data Centre archives, serves, and computes on data born to the desires of humans. Strikingly, while the Data Centre echoes the ghosts of human desires, it acts without desire of its own. The organismic analogy begins to split at its seams, but Capital does not care. Automata and human labour are priced into the market much the same. We argue that through the pricing of artificial intelligence Capital distils most clearly the value of intelligence and allows for its comparison across the organism - mechanism divide. 

---
# Domain Adaptation with Adaptive Imagination for Visual Reinforcement Learning under Limited Target Data 

**Authors**: Hyunwoo Park, Sang-Hyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.30192)  

**Abstract**: Sim-to-real transfer remains a major obstacle for reinforcement learning (RL), especially for vision-based control where image observations exacerbate the state-distribution shift between simulation and the real world. Domain adaptation (DA) is a promising remedy for this challenge. Prior sim-to-real DA works have demonstrated encouraging results, yet these approaches typically assume substantially more target data, which is not available in practice. Indeed, their performance degrades significantly when the target data budget is reduced. To address this challenge, we propose AIDA (Adaptive Imagination for Domain Adaptation), a domain adaptation framework for visual reinforcement learning that addresses sim-to-real transfer under scarce target data without requiring additional interaction with the target environment. Our key idea is adaptive imagination: generating reliable and semantic imagination rollouts to augment limited target data. Specifically, AIDA employs a distribution-shift-aware discriminator that truncates rollouts when imagined transitions drift into low-confidence regions, so that only reliable transitions contribute to the augmentation. On these reliable transitions, AIDA introduces a self-consistency loss that cycles through state -> image observation -> state, penalizing discrepancies between the original and reconstructed states. This provides additional adaptation signals beyond the scarce target data. Our experiments demonstrate that adaptive imagination effectively truncates unreliable rollouts. By enforcing a self-consistency loss on the resulting reliable transitions, AIDA learns semantically meaningful state representations and outperforms baselines across five MuJoCo tasks and two Gymnasium-Robotics tasks. 

---
# From Detecting Agency to Doing Work: Self-Caused Credit Builds a Durable Behavioral Self in a Minimal Spiking Agent 

**Authors**: Haoliang Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.30191)  

**Abstract**: How does an agent that can tell self from world come to be durably shaped by that distinction? Recent work shows that a predictive system can detect its own agency (Ye, 2026), but detecting agency does not explain durable, self-shaped behavior. We show that agency-gated slow credit -- a conjunctive term Own*Agency*Salience driving a slow parameter update -- produces post-unload behavioral residue: on a spiking substrate (Nengo LIF/PES), a learned self-preserving choice survives episodic buffer removal (retained fraction 0.96, N=50) and collapses when the slow decoders are reset or the agency gate is removed. Reproducing the agency comparator and toggling only the slow-credit channel, we find a clean dissociation: at matched agency gain, durable behavior develops only when self-credit performs slow work (post-unload self-preservation 1.00 vs 0.00). The same dissociation holds in 24-dimensional partially-observed control (0.74 vs 0.00), and a plastic-work analysis shows that basin deformation equals net self-credit work. Across eight sequentially-learned tasks under exogenous interference, the multiplicative veto also prevents forgetting: it retains old tasks (final post-unload accuracy 0.88, forgetting 0.13) where additive pooling collapses to chance-level recall, the no-agency ablation falls below chance, and episodic/replay baselines stay near chance after unload -- all with no replay buffer and no task-boundary-dependent protection mechanism (N=50). We formalize the durable residue as an operational behavioral self and argue that self-caused credit doing slow work is a necessary building block for agents that develop a self. No claim of consciousness is made. 

---
# Dynamo: Dynamic Skill-Tool Evolution for Vision-Language Agents 

**Authors**: Yutao Sun, Yanting Miao, Hao-Xuan Ma, Mengyu Zhou, Mingshuai Chen, Tiancheng Zhao, Dexin Wang, Lei Lv, Li Xu, Xiaoxi Jiang, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.30185)  

**Abstract**: Improving vision-language models (VLMs) on visual reasoning typically requires retraining or hand-designed prompts and tools. We present Dynamo, a training-free framework that adapts a frozen VLM without any weight updates. On a small labeled training subset, the agent inspects its own correct and incorrect attempts and evolves two complementary capabilities: reusable reasoning skills for cognitive bottlenecks, and executable visual tools for perceptual ones. Each generated tool is paired with a skill that specifies when to invoke it, and both capability types accumulate in a persistent library. Across four visual reasoning benchmarks and five VLM backbones, Dynamo improves direct inference on all 20 model--benchmark settings (avg. +5.6 acc). When the tool set is given in advance, the framework learns when to call each tool, and per-step tool choice improves on every tested backbone. Against task-specific RL (VTool-R1, DeepEyes), Dynamo closes 65--99% of the RL gap at a fraction of the compute, and combines additively with RL when available. 

---
# MirrorCode: AI can rebuild entire programs from behavior alone 

**Authors**: Tom Adamczewski, David Owen, David Rein, Florian Brand, Giles Edkins, Allen Hart, Daniel O'Connell  

**Link**: [PDF](https://arxiv.org/pdf/2606.30182)  

**Abstract**: AI models are rapidly improving at autonomous coding, as shown by benchmark progress and one-off demonstrations such as AI implementing a C compiler. However, existing coding benchmarks tend to focus on shorter tasks, and one-off demonstrations are hard to compare systematically because they often have some human guidance, and are not standardized or repeated across models. To address these challenges, we introduce MirrorCode, a long-horizon coding benchmark based on reimplementing entire software projects. In MirrorCode, AI agents must replicate the functionalities of an existing program, without access to its source code. AI solutions must match the original program's output exactly on end-to-end tests, including held-out tests. MirrorCode's 25 target programs span different areas of computing: Unix utilities, data serialization and query tools, bioinformatics, interpreters, static analysis, cryptography, and compression. Existing AI models can already reimplement complex software, with the strongest model scoring 56% across the benchmark. For example, AI can reimplement gotree, a 16,000-line bioinformatics toolkit - a task that we believe would take weeks for a human engineer. However, studying the frontier of performance requires a larger inference budget than typical benchmarks, for example, \$2,600 over 19 days for a single attempt on a large task. We show that AI agents can already complete long-horizon software engineering tasks, especially when requirements are precisely specified. More broadly, our work suggests AI will have transformative effects on software engineering, as autonomous agents continue to improve. 

---
# FacePlex: Full-Duplex Joint Speech-Facial Motion Generation for Conversational Avatars 

**Authors**: Habin Lim, Jae-Ho Lee, Hah Min Lew, Ji-Su Kang, Gyeong-Moon Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.30145)  

**Abstract**: Natural face-to-face conversation requires real-time speech generation together with synchronized facial motion. Existing systems only partially address this problem: speech-only full-duplex models can generate speech in real time but do not produce facial motion, while audio-driven facial motion models animate a face from already available audio rather than jointly generating speech and motion online. To bridge this gap, we first formalize full-duplex joint speech-facial motion generation, where speech tokens and facial motion tokens are produced together every step. Building on this formulation, we propose FacePlex, a unified streaming framework with two key components. First, Rolling Flow Matching adapts flow matching to online motion generation by committing new motion frames at each streaming step. Second, Rolling Cross-Attention couples the streaming audio queue with the motion queue, allowing speech and facial motion to condition each other as generation progresses. Through extensive experiments, ablation studies, and a user study, we show that FacePlex enables full-duplex joint speech-facial motion generation under online streaming constraints, while achieving stronger lip-sync quality and motion fidelity than audio-driven facial motion baselines. 

---
# Relevance Is Not Permission: Warranted Attention for Value Contributions 

**Authors**: Minwoo Yu, Young-guk Ha  

**Link**: [PDF](https://arxiv.org/pdf/2606.30139)  

**Abstract**: Relevance is not permission. Attention lets a model read key-value items related to the current query, but it does not guarantee that the value contribution of such an item becomes prediction evidence. A retrieved passage may be relevant to a question without being supporting evidence, and a historical fact or temporal neighbor may even blur true-tail ranking or the current edge score. This paper formalizes this gap as a permission problem for the weighted value term alpha_ij * v_j that is actually added to the prediction path. We propose Warrant, a path-localized interface that preserves attention relevance alpha_ij, exposes the value path leading to the primary metric, and, in the full model, turns alpha_ij * v_j into alpha_ij * g_ij * v_j through learned query-item permission g_ij. We place the same operator on the metric-defining value paths of CTDG link prediction, MTPP next-mark ranking, RAG supporting evidence selection, STPP next-location forecasting, and TKG tail prediction. Across 32 paired comparisons, 3 seeds, and 192 total runs, Warrant improves the primary metric in 27 comparisons; practical tiers consist of 10 substantial effects, 1 marginal effect, 8 positive but uncertain effects, 8 tie/negligible effects, and 5 drops. In the path-localization check, correct-path placement outperforms direction-aware Base performance in every domain and exceeds generic attention placement by +0.1076 AUC in CTDG and +0.0683 MRR in TKG. Ablations show that most TKG gains come from historical-tail value path exposure, whereas the core CTDG gain comes from edge-conditioned query-item permission. In conclusion, prediction evidence is not attention mass. A weighted value term becomes evidence only when it is warranted on the path to the metric. 

---
# Does Verbose Chain-of-Thought Really Help? In-Distribution Evidence that Content, Not Length, Matters 

**Authors**: Wenlong Wang, Fergal Reid  

**Link**: [PDF](https://arxiv.org/pdf/2606.30128)  

**Abstract**: Chain-of-thought (CoT) prompting improves LLM reasoning, but the source is contested: do the intermediate steps help because they carry useful semantic content, or because conditioning on more tokens buys extra computation before the model commits to an answer? We bring two lines of evidence to bear. First, in distribution: we repeatedly sample each model on the same question and pair a shorter with a longer of its own natural generations that follow the same reasoning plan, so nothing is rewritten and both traces are genuinely in-distribution. Across 25 models the extra tokens leave accuracy essentially unchanged for every independently-trained reasoner, and a blind analysis of the surplus tokens shows that what gain exists elsewhere tracks validation- and checking-content, not verbosity per se. Second, as a controlled intervention, we ask whether two traces expressing the same semantic content (the same facts, operations, and intermediate values, verified through directed acyclic graph equivalence) produce different outcomes when one is more verbose, using a dual-validator design across four targets and eight benchmarks with number-redacted completion and stratified bootstrap confidence intervals. Verbose traces do improve accuracy (25 of 32 benchmark-target cells are positive under at least one validator), but the effects are modest (typically 1-4 points) and depend on the quality of the verbose prose, not merely its length. Under maximum numerical redaction the effect is amplified (median 3.24x across four arithmetic benchmarks), and length-matched non-reasoning filler recovers none of it. Both lines converge: what matters is what the extra tokens do (the reasoning and validation content they carry), not how many there are, a picture neither a pure forward-pass-compute nor a pure semantic-content account fully explains. 

---
# Open Problems in Constitutional Preference Reconstruction 

**Authors**: Eleanor Clifford, Michael Amir, Arduin Findeis, Aaron Zhao, Robert Mullins  

**Link**: [PDF](https://arxiv.org/pdf/2606.30116)  

**Abstract**: Pairwise preference data is widely used for training and evaluating language models (e.g., RLHF), but each datapoint records a \emph{choice}, not the rationale behind it. Methods such as Inverse Constitutional AI (ICAI) attempt to improve interpretability by compressing datasets into short ``constitutions'' of natural-language principles. We argue this framing is under-specified: a flat list of principles is not yet an executable decision rule because it leaves principle composition implicit. We use the pairwise setting as a testbed to empirically characterize three open problems in constitutional methods. First, principle quality is hard to measure: coverage and accuracy are useful but incomplete proxies for end-to-end reconstruction. Second, \emph{composition is ambiguous}: holding principles fixed, different executors (LLM judge versus majority vote) agree only $73\%$ of the time. Third, \emph{constitutions differ between LLMs}: cross-model vote agreement is $73\%$, whereas intra-model agreement is $81\%$. Across PRISM, AlpacaEval, and Chatbot Arena, we show that principle refinement (ICAI+) may be a first step towards ameliorating these problems: inter-executor agreement rises to $78\%$, and transparent executors match LLM judge accuracy ($66\%$ vs.\ $67\%$). Our results highlight that constitutions should be evaluated as \emph{constitution--executor systems}, with implications for LLMs-as-a-judge broadly. 

---
# Structural Certification for Reliable Physical Design with Language Models 

**Authors**: Nakul Vyas, Iliya D. Stoev  

**Link**: [PDF](https://arxiv.org/pdf/2606.30107)  

**Abstract**: An unreliable language model can be made to produce reliable physical designs if the authority to assert is moved out of the model: the model proposes, and a deterministic engine alone certifies, returning certified, impossible, or unknown. We introduce Physics-Anchored Certification (PHACT), a propose-certify loop spanning five scientific domains, and identify what makes such a certificate trustworthy. A checker that accepts a model-supplied value can be forged; deriving the certified quantity from fixed inputs instead makes forgery impossible by construction. Across eighty adversarial trials spanning two models, two decoding temperatures, and a deliberately faulted engine, this contract produced zero false certifications. 

---
# Propagation of~Interval Belief Structures and~Imprecise Copulas for~Neural Network Verification 

**Authors**: Francesc Pifarre-Esquerda, Eric Goubault, Sylvie Putot  

**Link**: [PDF](https://arxiv.org/pdf/2606.30105)  

**Abstract**: Quantitative verification of neural networks requires reasoning about probabilities under substantial uncertainty in both input distributions and their dependence structure. In realistic settings, this information is often only partially specified, and assuming precise probabilistic models can lead to unreliable results. We propose a sound framework for quantitative verification under imprecise probabilistic information, combining interval belief structures to represent marginal uncertainty with imprecise copulas to model uncertain dependence. We develop a propagation method for imprecisely coupled interval belief structures through feed-forward neural networks. Using mixed imprecise copula volumes, we derive sound push-forward constructions through affine transformations and activation functions. The resulting output can provide guaranteed lower and upper bounds on probabilistic safety properties, valid for all probability models compatible with the specified imprecise inputs. 

---
# Temporal Feature Extractors in EEG Foundation Models: A Controlled Comparison Including a Pretrained Time-Series Model 

**Authors**: Ayşe Betül Yüce, Chris Joey Leffler, Sarun Varghese, Myra Spiliopoulou, Sebastian Stober  

**Link**: [PDF](https://arxiv.org/pdf/2606.30104)  

**Abstract**: Electroencephalography (EEG) foundation models aim to learn generalizable representations from large-scale brain recordings. However, the role of temporal feature extractors and whether pretrained time-series foundation models (TSFMs) can be effectively transferred to this setting remains underexplored. We conduct a controlled comparison of three temporal feature extraction strategies, including a linear baseline, a convolutional encoder, and a frozen pretrained TSFM (MOMENT), within a unified EEG foundation model. We evaluate their impact on representation quality using two downstream tasks: motor imagery and emotion recognition. Results reveal different trends across the evaluated benchmarks. On the motor imagery dataset, simple temporal representations perform competitively, whereas the emotion dataset benefits from richer temporal modeling. Although not specifically adapted to EEG, the pretrained TSFM serves as an effective temporal feature extractor, suggesting that general-purpose time-series representations can be transferred as frozen temporal feature extractors within EEG foundation models. 

---
# Hierarchical Reinforcement Learning in StarCraft Micromanagement with Influence Maps and Cluster-based Scripts 

**Authors**: Chunhui Bai, Changhe Li, Dequan Li, Xinye Cai, Shengxiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.30092)  

**Abstract**: Real-time strategy (RTS) games present significant AI challenges, characterized by expansive state-action spaces arising from multi-unit coordination in continuous battlefields, and sparse delayed rewards stemming from final win/lose signals. Existing approaches face a trade-off between managing the dimensionality explosion of joint actions and maintaining the interpretability of complex state representations. This complexity is further intensified by the limitation of traditional hierarchical structures in adaptively decomposing tasks into effective tactical modules. Such difficulties are compounded by the black-box nature of deep learning models and their reliance on sparse rewards, which together result in limited sample efficiency and a lack of decision-making transparency. To address these limitations, this paper proposes HRL-IM/CBS, a hierarchical reinforcement learning framework with influence map hashing and cluster-based scripts for StarCraft micromanagement. Influence map hashing encodes global battlefield situations into compact hexadecimal codes, capturing spatial control and relative advantage. Cluster-based scripts enable dynamic local coordination through adaptive unit partitioning. The hierarchical multi-Q-table architecture decomposes decision-making into upper-level clustering strategy selection and lower-level tactical execution, with reward allocation providing dense learning signals. Experiments across six asymmetric scenarios demonstrate competitive performance against deep RL baselines while offering advantages in sample efficiency and interpretability through transparent Q-table representations. 

---
# SAT-RTS: A systematic framework for tactical knowledge extraction and visualization-based analysis in real-time strategy games 

**Authors**: Chunhui Bai, Changhe Li, Yuqiang Li, Lei Liu, Shoufei Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.30090)  

**Abstract**: Efficient tactical knowledge extraction and analysis in real-time strategy (RTS) games micromanagement are constrained by the high-dimensional coupled state-action sequential data and the black-box decision-making process. Current research rarely provides a hierarchical visualization-based attribution analysis from the perspective of data decoupling and abstraction. To facilitate interpretable tactical knowledge extraction and visualization-based analysis in RTS games, a systematic framework named state-action-tactic analysis pipeline (SAT-RTS) is proposed. To decipher the deep-seated drivers of critical decisions in RTS learning systems, this work integrates interpretable visualization with the automated extraction of latent tactical patterns from high-dimensional sequence data. By adapting a cluster-centric BK-tree algorithm and incorporating specialized distance metrics designed to quantify multi-aspect similarities, the proposed framework facilitates robust state-stream abstraction. Furthermore, a rule-based multi-label extraction method is developed to transform unstructured state-action sequences into discrete and interpretable tactical labels, effectively bridging the gap between raw behavioral data and high-level tactical insights. By holistically integrating these computational methods into a hierarchical visualization-based pipeline, the proposed framework effectively addresses the challenges of processing massive real-time data streams while providing fitness landscape visualizations and analytical insights to decipher deep-seated tactical drivers. Comprehensive experiments demonstrate that the proposed SAT-RTS significantly enhances the interpretability and efficiency of tactical analysis in complex RTS environments. 

---
# ACPO: Agent-Chained Policy Optimization for Multi-Agent Reinforcement Learning 

**Authors**: Daiki E. Matsunaga, Junho Na, Tri Wahyu Guntara, Scott Sanner, Pascal Poupart, Jongmin Lee, Kee-Eung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.30072)  

**Abstract**: Cooperative tasks in Multi-Agent Reinforcement Learning (MARL) require agents to collectively maximize a shared return. Under the Centralized Training with Decentralized Execution (CTDE) paradigm, policy gradients have remained difficult to compute directly. Prior methods largely follow two approaches: independent factorized updates with centralized critics, which lack general joint-improvement guarantees without value decomposition assumptions, or alternating best-response updates, which can converge to suboptimal Nash Equilibria. In this paper, we show the joint policy gradient admits an exact decentralized decomposition of per-agent terms, each formed from per-agent score functions and decentralized critics. Based on this decomposition, we develop Agent-Chained Policy Optimization (ACPO), where actors are trained independently, with their updates together constituting a single step on the joint policy gradient. Central to this result is a serialized view of the simultaneous joint decision in which agents commit actions one at a time, each conditioning on a belief over preceding actions. The belief acts as the coordination mechanism which ties the independent per-agent updates into a joint gradient step. We evaluate ACPO on Multi-Robot Warehouse, SMACv2, and MA-MuJoCo, where it outperforms strong baselines, with the gap widening as the number of agents grows. 

---
# AlgoSkill: Learning to Design Algorithms by Scheduling Human-Like Skills 

**Authors**: Xinyuan Song, Zekun Cai, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.29999)  

**Abstract**: Designing an algorithm from a natural-language problem statement requires identifying the problem structure, reading constraints, choosing a suitable paradigm, checking correctness, and refining complexity. Existing large language model (LLM) methods often rely on direct generation or generic self-refinement, leaving these steps implicit. We propose AlgoSkill, which models algorithm design as sequential decision-making over a typed library of algorithmic skills, including abstraction, constraint analysis, state design, data-structure selection, proof checking, counterexample construction, and complexity refinement. A learned scheduler proposes skills from the current design state, while a Monte Carlo Tree Search (MCTS) controller explores skill sequences using verification feedback from compilation, testing, stress testing, and complexity analysis. Experiments on competitive programming and combinatorial optimization benchmarks show that AlgoSkill improves over direct LLM generation, chain-of-thought prompting, self-refinement, and MCTS without typed skills. Ablations show that typed skills, verification-based repair, and search-based scheduling each contribute to performance. These results support treating automatic algorithm design as verification-guided skill scheduling rather than one-shot code generation. 

---
# Be Faithful When Response: Returning Fluent and Grounded Answers for Vision-Language Models Reinforcement Learning 

**Authors**: Peng, Yin Zhang, Yanglin Zhang, Haonan Wu, Zishan Liu, Ruoxi Zang, Xin Zhu, Jiayin Zheng, Jian Yao, Zefeng Ji, Fei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.29984)  

**Abstract**: Reinforcement Learning (RL) is an important paradigm for improving the reasoning capabilities of Vision-Language Models (VLMs). However, directly applying RL to rollout multimodal reasoning can lead to instability, due to the exploitation of language priors, the neglect of visual evidence, and the generation of reasoning traces that are fluent yet not visually grounded. The question arises: Can initially steer the policy toward visually faithful reasoning regime before applying reinforcement learning? To this end, we propose a Faithful Warm-Start (FWS) strategy that first curates samples with explicit vision-language causal relationships from six general VQA benchmarks to construct the FaithfulQA dataset, where each of the image-question pairs gains a certain degree of visual observations, question requirements, commonsense knowledge, domain knowledge, and the final answer. Subsequently, a VLM-based judge is employed to further purify the dataset, ensuring strong causal consistency and visual faithfulness. This warm-start stage equips the model with the capability to understand causally grounded vision-language patterns before subsequent RL optimization under sparse answer-level rewards. Experimental results show that such faithful supervision improves answer accuracy, stabilizes RL training, and reduces visually unsupported reasoning. 

---
# Exploration and Online Transfer with Behavioral Foundation Models 

**Authors**: Louis Bagot, Mathieu Lefort, Laëtitia Matignon  

**Link**: [PDF](https://arxiv.org/pdf/2606.29980)  

**Abstract**: Zero-shot Transfer in Reinforcement Learning (RL) aims to train an agent that can generate optimal policies for any reward function, without additional learning at transfer time, while training only on reward-free trajectories. For their generality over tasks, such models are sometimes called ``Behavioral Foundation Models'' (BFMs). While they have shown strong performances and improvements in recent years, the current framework and algorithms still assume that, during the transfer phase, the agent is informed offline about the reward (the task to solve) through a dataset of state-reward pairs, which it uses to pick the best policy to deploy. However, in practice if the reward is a black-box (e.g. direct user feedback), it is not possible to generate such a dataset: it is necessary to observe the reward through interactions with the environment. In other words, the current framework of offline transfer is not aligned with the traditional RL setting of online learning through trial-and-error, which requires exploration in order to find rewards. This paper proposes to tackle this new online transfer in zero-shot RL, with the key insight that the BFM itself can be used to generate exploration policies. We show that it is possible to frame this online learning problem in terms of a bandit-like exploration-exploitation problem. More precisely, at each step the bandit algorithm recommends a policy, the BFM executes it in the environment, which yields a reward and a new state; we repeat the process until we converge to the optimal policy. In the popular context of linear reward approximation, we derive a formulation inspired by Upper Confidence Bound and show that exploration can be achieved through the minimization of the eigenvalues of an uncertainty matrix. We evaluate qualitatively and quantitatively our framework on a simple environment to validate the concept of our method. 

---
# First-Order Temporal Logic Tensor Networks 

**Authors**: Luca Boscarato, Ivan Donadello, Alessandro Artale, Marco Montali, Fabrizio Maria Maggi  

**Link**: [PDF](https://arxiv.org/pdf/2606.29972)  

**Abstract**: Most of the existing neuro-symbolic AI methods focus on the scenario of static knowledge where objects do not change according to a temporal dimension. Temporal neuro-symbolic works are still under explored and are mainly developed for time-interval logic or propositional linear temporal logic. There is a lack of models studying linear temporal logics with predicates that deal with objects whose properties and relations change through the time. We present First-Order Temporal Logic Tensor Networks (FOT-LTN) that is an extension of Logic Tensor Networks (LTN) that fills this gap by considering a linear-temporal dimension. In particular, FOT-LTN joins the syntax of First-Order Linear Temporal Logic with the fuzzy (and real-valued) semantics of LTN obtaining a framework that supports both temporal operators and quantifiers and is totally differentiable. A first evaluation regards a temporal knowledge graph completion task on two synthetic datasets showing better performance of FOT-LTN with respect to dedicated (purely neural) methods. 

---
# SAGA: Scene-Aware, Goal-Evolving Agents for Long-Horizon CivRealm Strategy Planning 

**Authors**: Tianyu Jin, Shuo Chen, Yida Wang, Liuyu Xiang, Yingzhuo Liu, Zhiyao Jiang, Yexin Li, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2606.29932)  

**Abstract**: Long-horizon strategic planning in complex strategy games demands concurrent reasoning across multiple decision domains under imperfect information and sparse reward. Existing LLM-based agents suffer from three systematic failures: scene blindness from raw tile coordinates, context overflow and domain coupling from monolithic state dumps, and shallow cross-game learning that treats each episode in isolation. We present SAGA, an LLM multi-agent framework with three mechanisms each directly targeting one class of failure: (i) a Map-Semantic Scene Graph that encodes typed spatial relations among game entities into per-unit natural-language context, resolving spatial blindness without global token inflation; (ii) a Tool-Augmented Planner that pulls fine-grained domain state on demand and dispatches per-domain directives to dedicated specialist controllers, eliminating context overflow, domain coupling, and mechanical constraint violations; and (iii) a Dual-Horizon Feedback Loop that combines periodic within-game goal generation with structured cross-game causal post-mortem, enabling principled strategic evolution without manual reward engineering. Evaluated on FreeCiv, SAGA attains the highest mean civilization score -- the environment's sole sparse objective reward -- with lower variance than the two strongest baselines, and is the only method that significantly surpasses every baseline on infrastructure construction, the resource axis most readily sacrificed under multi-objective conflict. It outscores the two strongest baselines in most head-to-head games while cutting output tokens (the dominant decoding cost) by 27%. Equipped with the cross-game evolution module, SAGA reaches the highest end-of-chain score across five successive episodes. Ablation studies confirm that each architectural component contributes independently to this advantage. 

---
# HippoSpark: An On-Demand Experience System for LLM Reasoning 

**Authors**: Jingyao Liu, Danling Meng, Chen Huang, Yukun Yan, Zhenghao Liu, Wenqiang Lei, See-Kiong Ng, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.29929)  

**Abstract**: Distilling historical trajectories into reusable experience to enhance future problem-solving has become a focal point of recent LLM research. However, existing methods predominantly operate at the task level, leveraging general summaries or rules under the assumption that analogous tasks share universal solution patterns. This approach often fails in complex reasoning, which typically falters at local bottlenecks that require precise, state-specific guidance rather than broad heuristics. We introduce HippoSpark, a state-level experience system that performs on-demand retrieval tailored to the immediate needs of the current reasoning state. Across mathematical, scientific, and programming benchmarks, HippoSpark consistently outperforms both standard prompting and task-level experience baselines. Our findings reveal that the most effective experience systems are those that provide actionable guidance at critical bottlenecks rather than serving as generic task-level context. Our code is available at this https URL. 

---
# A causal modeling perspective on decision theory 

**Authors**: Arvid Sjölander  

**Link**: [PDF](https://arxiv.org/pdf/2606.29911)  

**Abstract**: Decision theory provides a formal framework for how agents should make choices under uncertainty, drawing on ideas from philosophy, probability, and causality. Despite significant progress, the field still lacks a unified modeling language, and key concepts - such as the distinction between subjective and objective elements, or what it means for a decision theory to perform well - are often left implicit. This can make it difficult to evaluate and compare competing theories, particularly in controversial cases. In this paper, we address these issues by introducing a formal framework for decision theory based on nonparametric structural equation models (NPSEMs), a well-established tool in causal inference. NPSEMs provide a unified foundation for representing agents, counterfactuals, and causal relationships, allowing for unambiguous definitions of EDT and CDT. Building on this foundation, we propose a novel decision theory - personal decision theory - which instructs agents to maximize a subjective model of their own counterfactual utility. We introduce a formal performance metric based on hypothetical interventions that enforce a given decision theory across a population - such as might be achieved through education or policy -- and show that, under certain assumptions, personal decision theory is optimal with respect to this metric. Throughout, we use the smoking lesion problem as a running example and conclude with a formal analysis of Newcomb's problem. Our aim is to provide decision theory with a clearer modeling language and firmer evaluative ground, thereby enabling more rigorous comparisons and facilitating conceptual progress in the field. 

---
# SafePyramid: A Hierarchical Benchmark for In-context Policy Guardrailing 

**Authors**: Jiacheng Zhang, Haoyu He, Sen Zhang, Shen Wang, Xiaolei Xu, Yuhao Sun, Meng Shen, Feng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29887)  

**Abstract**: In real-world applications, guardrails are often expected to identify unsafe user-model interactions according to application-specific safety policies, rather than relying on predefined risk taxonomies. In this work, we study this setting under the paradigm of in-context policy guardrailing, where guardrails predict safety violations based on policy specifications provided in context. To systematically evaluate this capability, we introduce SafePyramid, a safety benchmark comprising 1,000 multi-turn conversations across 10 domains and 3,000 corresponding application-specific policies, which together contain 61,699 distinct natural-language rules. SafePyramid organizes the evaluation into three difficulty levels: L0 evaluates individual-rule understanding, L1 evaluates reasoning over rule dependencies, and L2 evaluates adaptation of full novel policy frameworks defined in context. To ensure benchmark quality, we employ a rigorous multi-stage pipeline to construct and validate the benchmark. Using SafePyramid, we evaluate 10 frontier LLMs and 5 policy-configurable guardrails and find that in-context policy guardrailing remains highly challenging: even the best-performing model, GPT-5.5, exactly identifies the full set of violated rules in only 54.0%, 35.3%, and 12.9% cases on L0, L1, and L2, respectively. These results highlight the limitations of current guardrails and call for stronger in-context policy guardrails that can reliably execute policies, resolve rule dependencies, and adapt to novel policy frameworks. 

---
# AI Training Manager: Bounded Closed-Loop Control of Adaptive Training Recipes 

**Authors**: Anjali Rao, Nikhil Kamalkumar Advani  

**Link**: [PDF](https://arxiv.org/pdf/2606.29871)  

**Abstract**: We present the AI Training Manager, a bounded LLM-based supervisory controller for adaptive machine learning training. Standard training pipelines often rely on fixed recipes or single-axis schedulers, which can struggle with mid-run failures such as severe overfitting, loss imbalance, exploration collapse, or unsafe exploration. Rather than replacing mathematical optimizers or acting as an unconstrained coding agent, the manager operates through a schema-conditioned interface: it reads structured telemetry snapshots from an active run, audits a constrained action space, and returns validated updates to training parameters such as learning rate, regularization strength, loss-weight coefficients, and exploration settings. We evaluate this architecture across supervised language modeling and reinforcement learning. On TinyStories, the manager detects and corrects overfitting, achieving a validation loss 60% lower than the baseline while producing auditable intervention logs. In this supervised setting, we additionally show that manager inference does not need to block the training loop: training can continue while a manager response is pending, and validated updates can be applied asynchronously once available. In a robotic manipulation reinforcement-learning task, we use the same bounded decision interface in an episodic closed-loop setting, where manager updates are applied at evaluation or checkpoint boundaries. The manager mitigates both conservative and unsafe exploration regimes. These results suggest that schema-conditioned LLMs can serve as bounded supervisory managers for live training runs, complementing conventional optimizers and schedulers with interpretable, multi-axis intervention capabilities 

---
# Beyond Triplet Plausibility: Relation Set Completion in Knowledge Graphs 

**Authors**: Zihao Zheng, Borui Cai, Yao Zhao, Keshav Sood, Yong Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29860)  

**Abstract**: Knowledge graphs (KGs) organize real-world knowledge as triplets and underpin many downstream applications. Due to their inherent incompleteness, knowledge graph completion (KGC) is widely studied and is typically formulated as triplet prediction, with link prediction as the dominant paradigm. However, this formulation focuses on the incompleteness of triplet-wise information and overlooks the incompleteness of entity-relation compatibility information. To address this limitation, we introduce a relation set completion task (RSC), which complements the link prediction task and aims to reason about missing relations that are semantically compatible with a given entity. We further propose a Relation Set Embedding model (RelSetE), which models latent patterns among the observed relations of entities to infer missing ones. To evaluate RelSetE, we derive three benchmark datasets from standard KG benchmarks. Extensive experiments demonstrate that RelSetE effectively captures entity-relation compatibility patterns and performs favorably in inferring missing relations of entities. Code and data are publicly available. 

---
# The CRISTAL Method: Neurosymbolic analysis from AI-synthesized world models 

**Authors**: Rafael Kaufmann, Felix Neubürger, Michael Walters, Thomas Kopinski, Dimitrije Marković  

**Link**: [PDF](https://arxiv.org/pdf/2606.29799)  

**Abstract**: This project introduces the CRISTAL Method (Coherent Reliable Intentional Synthesis of Truthful Analysis Logic), a neurosymbolic framework for automating complex analysis workflows, with fundamental investment analysis as a primary use case. This domain poses major challenges: high structural uncertainty, noisy and subjective data, tight attention budgets, and the need for justified, reproducible decisions. Human analysts often struggle in this domain due to cognitive biases and limitations, suggesting significant value in automation. But while LLM-based agents have been proposed as analytical aids, their limitations -- poor numerical reasoning, unawareness of uncertainty, and lack of reproducibility -- hinder their effectiveness in this context. CRISTAL addresses these gaps through a principled blend of statistical model synthesis, continuous learning, and active learning. Starting from a natural-language prior knowledge curriculum, CRISTAL builds a dynamic, interpretable probabilistic program that enables full Bayesian inference, including uncertainty quantification and budget-aware data acquisition. CRISTAL continually refines its world model during analysis, leveraging LLMs for code synthesis and learning. We validate CRISTAL on a novel benchmark of synthetic equities with rich financial and textual data. On a company classification task, CRISTAL achieves Bayes-optimal accuracy with just 5 examples and a 5-second budget, outperforming state-of-the-art LLMs that plateau around 40\% accuracy even with order-of-magnitude more input data and compute. 

---
# CLQT: A Closed-Loop, Cost-Aware, Strategy-Consistent Benchmark for Diagnostic Evaluation of LLM Portfolio-Management Agents 

**Authors**: Bo Qu, Mingguang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.29771)  

**Abstract**: LLM agents are increasingly cast as autonomous portfolio managers, and benchmarks have moved from financial question-answering to sequential trading. Yet most still rank agents by returns over a fixed window -- a weak proxy, since a period's return is dominated by the market path and apparent alpha can dissolve once look-ahead leakage is controlled. Such a ranking certifies neither sound reasoning, nor a consistent strategy, nor a durable edge. We introduce CLQT, which reframes closed-loop trading evaluation as diagnosis rather than ranking: an instrument that localizes where and why an agent's process succeeds or fails. CLQT is a fully closed-loop, cost-aware, strategy-consistent, temporally-gated environment whose agents run a five-stage cycle: gather, synthesize, allocate, execute, reflect. Each round emits a complete DecisionRound sealed into a recompute-verifiable hash chain, so every metric is reconstructable from the trail. Six pillars form the substrate: a hard TimeGate, institutional transaction- and financing-cost modeling, strategy-consistency scoring, three-tier memory, a Model-Context-Protocol tool layer, and mandate-aware synthesis. The same agent runs as a constrained committee of specialized roles or a single full-autonomy orchestrator, making process scaffolding an experimental variable. From the audit trail we compute a five-axis capability scorecard (APM-CS: Coherence, Acuity, Composure, Discipline, Reliability), with Coherence judged partly by a held-out, out-of-cohort LLM to curb self-preference bias. We validate it on a contamination-controlled multi-model backtest with an ablation grid and a live broker track on unseen, post-cutoff data, against a repeated-run noise floor. CLQT separates outcome from capability, yielding not a model ranking but a durable, extensible map of agent competencies and limitations. 

---
# Rethinking Generative Reconstruction Attacks against Graph Neural Network Models 

**Authors**: Adebayo Keji, Sayanton Dibbo  

**Link**: [PDF](https://arxiv.org/pdf/2606.29748)  

**Abstract**: The application of graph data in numerous disciplines raises the need for gathering and analyzing huge volumes of data, some of which is private and sensitive. The non-Euclidean nature of the graph data makes the analysis computationally challenging, leading to the use of Graph Neural Networks (GNNs) in the age of AI. GNNs may inadvertently leak sensitive data they are trained on, which raises serious data security issues, including the model inversion attack. In this study, we analyze GNNs' vulnerabilities by introducing two novel graph inversion (i.e., reconstruction) attacks: graph-label conditioned (GLC) attack and embedding-label conditioned (ELC) attack, utilizing targetmodel predictions and their intermediate representations, respectively. We perform a comprehensive analysis of our introduced privacy attacks and compare them with existing baselines across three benchmark graph datasets (i.e., NCI1, PROTEINS, and AIDS) and four graph distributional/structural metrics (i.e., FGD, EGD, MMD, and GKS). Our work demonstrates that an adversary can use the generator-discriminator technique to reconstruct high-quality graphs in real-world black-box attack scenarios against GNNs. Additionally, we present a variant of our attacks (Ours--) with 50% reduced queries, achieving good or comparable reconstruction attack performance. In addition, we show that GNNs are highly vulnerable to privacy attacks, varying Laplacian noise-scales. 

---
# DEEPMED Search: An Open-Source Agentic Platform for Medical Deep Research with Introspective Verification 

**Authors**: Maolin Liu, Fanyu Xu, Ruoqing Xu, Jiahang Zhang, Hao Wang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29746)  

**Abstract**: Navigating the deluge of heterogeneous medical data, from academic literature (PubMed) to clinical guidelines (Web) and private knowledge bases, remains a critical bottleneck for evidence-based medicine. While commercial black-box tools lack transparency, standard open-source RAG implementations frequently suffer from reasoning drift when handling complex, long-tail queries. We present DEEPMED Search, a fully open-source, agentic platform designed for transparent medical deep research. Built on a high-performance this http URL architecture, DEEPMED Search features a source-adaptive router that autonomously dispatches sub-queries to PubMed, web search, or local graph-based knowledge bases based on information density. Crucially, the platform integrates an introspective verification module, powered by a causal-consistent multi-agent debate framework, to validate retrieved evidence against diagnostic logic before synthesis. To demonstrate its robustness, we showcase DEEPMED Search's ability to autonomously decompose high-difficulty rare disease queries, filter out confounding noise, and generate structured, citation-backed research reports in minutes. By open-sourcing this software, we provide the community with a robust infrastructure to democratize access to trustworthy, glass-box medical reasoning in research and prototyping settings. 

---
# DeepTrans Studio: Turning Expert Interventions into Shared Team Knowledge in Agentic Translation Workflows 

**Authors**: Ziyang Lian, Qingya Zhang, Hao Wang, Huiwen Xiong, Qi Yang, Lingyi Meng, Xiaoyi Gu, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29727)  

**Abstract**: Professional translation is often a team-based process: translators, reviewers, and project managers must coordinate terminology, legal force, and accountability across documents. Yet many LLM-based translation tools treat human corrections as isolated edits. Expert decisions made in one segment or by one member are rarely captured as reusable knowledge for the rest of the team. We present DeepTrans Studio, a collaborative translation workspace that lets professionals intercept selected nodes in an agentic translation workflow, review evidence, revise AI outputs, and save approved decisions to a shared team memory. During the demo, attendees will role-play translators and reviewers, resolve preset terminology and legal-modal risks, and see how their decisions are propagated to downstream segments and surfaced in a teammate's workspace as reusable precedents. The demo illustrates how human interventions in AI-mediated work can become shared, traceable knowledge rather than one-off corrections. 

---
# GUICrafter: Weakly-Supervised GUI Agent Leveraging Massive Unannotated Screenshots 

**Authors**: Sunqi Fan, Lingshan Chen, Runqi Yin, Qingle Liu, Yongming Rao, Meng-Hao Guo, Shi-Min Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29705)  

**Abstract**: Data, as the fundamental substrate of modern intelligence, has greatly driven the development of current foundation models. Naturally, researchers aim to extend this paradigm to the domain of GUI agents, hoping to build strong GUI agents through a similar paradigm. However, GUI agent data cannot be directly harvested from the internet, making it costly and difficult to collect at scale. As a result, current GUI agents suffer from poor cross-device generalization and limited visual grounding ability for fine-grained GUI elements. As an attempt to address data challenge in GUI agents, we propose GUICrafter, a weakly-supervised GUI agent leveraging massive unannotated screenshots to substantially reduce the reliance on expensive human annotations. GUICrafter explores a curriculum learning framework for training GUI agents through two progressive stages. First, the model learns visual grounding from large-scale unannotated screenshots and webpages, leveraging the rich contextual signals inherent in GUI interactions without human annotations. Then, in Stage 2, we leverage a small amount of high-quality data to calibrate the model via reinforcement learning. Experiments show that GUICrafter achieves competitive, or even superior, performance to advanced systems like UI-TARS while using only 0.1% of its data. Furthermore, under the same amount of annotated data, GUICrafter surpasses all previous methods such as GUI-R1. Code, data, and models are available at this https URL. 

---
# Toward Secure and Reliable PDDL Formalization of Large Language Models with Planner-in-the-Loop Feedback 

**Authors**: Jiamei Jiang, Jiajing Zhang, Feifei Mo, Linjing Li, Daniel Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.29700)  

**Abstract**: Planning often requires symbolic specifications that are both executable and verifiable. For large language models deployed in autonomous or decision-support systems, failures in such formalization may lead to unverifiable decisions, execution failures, or unsafe downstream behavior. We present NL-PDDL-Bench, a multi-domain benchmark for natural-language-to-PDDL specification construction with planner-verified executability and controlled difficulty scaling by object count. We further propose a planner-in-the-loop framework that uses validator and planner diagnostics to revise non-executable specifications through localized edits. Building on this infrastructure, we develop a planner-grounded optimization recipe that combines parameter-efficient Low-Rank Adaptation supervised fine-tuning, offline planner-derived preference pairs for Direct Preference Optimization, and inference-time planner-in-the-loop repair, without requiring online planner calls during training. We also provide a unified evaluation suite for parseability, solvability, specification similarity, and outcome-aware plan-level consistency against planner references. Experiments on representative model families show substantial gains in planner success and plan-level agreement, with improved robustness under difficulty scaling and cross-domain variation. These results highlight the value of externally verifiable formalization for reliable deployment of LLMs in safety- or security-sensitive planning systems. Code and data are available at: this https URL 

---
# Sample-Efficient Learning of Probabilistic Causes for Reachability in Markov Decision Processes with Probabilistic Guarantees 

**Authors**: Ryohei Oura, Georgios Fainekos, Hideki Okamoto, Bardh Hoxha  

**Link**: [PDF](https://arxiv.org/pdf/2606.29681)  

**Abstract**: Probabilistic model checking for Markov decision processes (MDPs) provides quantitative guarantees, but often offers limited insight into why undesired outcomes occur. Probability-raising (PR) causality addresses this by identifying states whose visitation increases the probability of reaching designated states. Existing PR-cause identification methods, however, use MDP modifications not well-suited for learning: the gap between conditional and unconditional reachability probabilities can be hard to detect from transition samples, and construction requires reachability probabilities of the MDP, which are unavailable when transition probabilities are unknown. We study unknown MDPs and propose a learning approach with probabilistic guarantees for PR-cause identification. Our key ingredient is a restart-based MDP modification that reduces PR-cause checking to two conditional reachability queries without using reachability values of the original MDP. We prove correctness, establish sample-complexity bounds, and develop an anytime learning-and-checking algorithm based on two-sided value iteration that progressively classifies states as causal, non-causal, or undecided. Experiments on two benchmarks demonstrate reliable and fast identification of PR causes. 

---
# Diversity is the Strength of the AI Crowd 

**Authors**: Matthew Aitchison, Scott Jeen, Toby Shevlane, Ben Day  

**Link**: [PDF](https://arxiv.org/pdf/2606.29661)  

**Abstract**: Top AI forecasting systems are approaching superforecaster-level accuracy on future world events, but still rely primarily on off-the-shelf LLMs combined with forecasting-specific context gathering and scaffolding. We study how to improve this recipe through ensembling: given a fixed number of samples, which off-the-shelf model forecasts should be combined to maximize accuracy? On binary questions from the Metaculus AI Benchmark, we find that individual accuracy is not enough: many frontier LLMs make highly correlated predictions, limiting the value of additional forecasts from the same or similar models. Instead, the strongest ensembles combine accurate but diverse forecasters, with models such as \model{Grok 4} contributing disproportionately because their predictions are less correlated with other frontier LLMs. These results suggest that the strength of the AI crowd comes not from sampling more forecasts indiscriminately, but from combining forecasts across models with complementary errors, motivating forecasting systems that explicitly optimize for both model quality and diversity. 

---
# Safety from Honesty in a Disinterested AI Predictor 

**Authors**: Yoshua Bengio, Oliver Richardson, Tomáš Gavenčiak, Michael Cohen, Rory Svarc, Damiano Fornasiere, Gael Gendron, David Hyland, Aton Kamanda, Adam Oberman, Francis Rhys Ward, Anna Gavenčiak, Jacob Livingston Slosser, Vincent Mai, Iulian Serban, Joumana Ghosn  

**Link**: [PDF](https://arxiv.org/pdf/2606.29657)  

**Abstract**: As AI systems become more capable, training procedures that optimize for downstream outcomes risk introducing implicit agency: goal-directed behavior that designers never specified. We present a formal safety argument for the Scientist AI (SAI) Predictor, trained to approximate the Bayesian posterior conditioned on a dataset of "epistemically contextualized" natural-language statements. We argue that such a Predictor can honestly predict agents, actions, and their consequences without itself being an agent that selects outputs to achieve goals. This rests on data representation and on the training procedure. Epistemic contextualization of text distinguishes latent factual claims from communication acts, so expressions of goals are treated as evidence to be explained rather than drives the model adopts. With a posterior-seeking training objective, this is intended to drive the Predictor toward calibrated, cautious predictions. Training proceeds so downstream effects of deploying a prediction never serve as a reward signal; any agency the system needs is supplied by explicit scaffolding constrained by guardrails. We prove that, under assumptions on the training dynamics and on the argued sparsity of dangerous Predictors, the probability that training produces a Predictor whose guarded deployment carries residual harm above a specified threshold is small: a dangerous Predictor would have to underestimate harm in a coordinated way across many queries while such coordinated patterns are rare under the initialization distribution and receive no direct training signal. Safety and accuracy are jointly supported in this framework, since the constraints that secure accuracy are the same ones that make coordinated deception costly. These guarantees against misalignment and agency arising from within the Predictor itself do not preclude the use of the Predictor as part of an agentic system. 

---
# Budgeted Act-or-Defer Multi-Agent LLM Deliberation with Local Reliability Bounds 

**Authors**: Mengdie Flora Wang, Haochen Xie, Guanghui Wang, Devin Zhang, Jae Oh Woo  

**Link**: [PDF](https://arxiv.org/pdf/2606.29654)  

**Abstract**: Multi-agent deliberation among LLMs can improve reasoning, but deployment requires deciding when the current answer is reliable enough to act on and when it should be escalated to human review. We formulate this as budgeted act-or-defer decision making. At each round, the system maps the debate prefix to a low-dimensional state, computes a $k$-nearest-neighbor lower confidence bound on state-conditional correctness using calibration data, and acts only when the bound exceeds a user-specified reliability threshold. The certificate controls wrong actions through the decomposition $\beta = \delta + \alpha + \varepsilon_{\mathrm{act}}$, separating calibration failure, residual action risk, and representation gap. The guarantee is conditional, not distribution-free: it relies on a valid local bias envelope and an action-region representation-gap bound, and each assumption is paired with falsification-style diagnostics. Because the same absolute wrong-action budget has different meanings across tasks of different difficulty, we set budgets relative to each task's final-round error using training data only, and evaluate safety by normalized budget usage $\mathrm{WA}/\beta$. On six benchmarks against nine baselines, the method uses 9--12% of the pre-declared budget on activated datasets, reaching up to 84% automation and 96% acted-on accuracy; on stress-test datasets, it defers rather than forcing unreliable automation. Rather than relying on per-task post-hoc threshold search, the method prospectively converts a user-declared wrong-action budget into an auditable act-or-defer operating point before deployment, under explicitly stated assumptions. 

---
# SFBench: The SciFy Scientific Feasibility Benchmark 

**Authors**: Cash Costello, James Mayfield, Elsbeth Turcan, Christine Piatko, Christina K. Pikas, Justin Rokisky, Sam Scheck, Chris Ribaudo, Ritwik Bose, Alex Memory  

**Link**: [PDF](https://arxiv.org/pdf/2606.29630)  

**Abstract**: We present SFBench, a benchmark dataset for evaluating systems that assess the feasibility of scientific claims. SFBench includes 197 claims in materials science, each annotated with a ground-truth feasibility score on a five-point scale along with an explanation of that assessment. The collection differs from previous collections in several important ways: 1) it defines a complex task that requires reasoning over claims of varying scientific feasibility; 2) its claims are not extracted from existing scientific publications but are created de novo, greatly reducing the chances that LLMs have trained on them; 3) claims and ground truth are established by subject matter experts, not by artificial intelligence; and 4) unlike many benchmarks that ask about question/answer pairs, provide multiple choice answers, or ask questions requiring short, fixed answers, SFBench explanations are completely open-ended. We describe the benchmark design, data creation process, and evaluation metrics, and we report baseline results using recent GPT models. 

---
# SCARCE: Scalable Cascade Analysis for Rare-event Characterisation via Embeddings 

**Authors**: Yingjie Wang, Yi Dong, Edmund Lau, Jie Meng, Taylor T Johnson, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29623)  

**Abstract**: Rare events govern the safety profile of modern AI systems, yet their probabilities are extremely difficult to estimate: direct Monte Carlo requires prohibitive sample budgets. Subset Simulation (SS) addresses this by decomposing a rare-event probability into moderate conditional probabilities over nested intermediate events. However, classical SS requires a handcrafted scalar performance function whose sublevel sets define those events, demanding detailed knowledge of the failure geometry and limiting transfer to new domains. We propose SCARCE (Scalable Cascade Analysis for Rare-event Characterisation via Embeddings), which replaces the performance function with learned latent representations and geometric rulers that score proximity to failure regions. Adaptive thresholding constructs nested intermediate events directly from data. We formalise SCARCE through a non-negative supermartingale, yielding a high-probability upper envelope that remains valid under early stopping. On MNIST misclassification, where dense Monte Carlo provides ground truth, SCARCE achieves approximately 400--500 times lower mean absolute error than grid-searched traditional SS while eliminating systematic over-counting. We then study PAIR-style LLM jailbreaks under a fleet-level threat model with adversarial fraction $\eta$. On Llama-Guard-3-8B hidden states, a PCA-based ruler attains 2.6% mean relative error for $\eta \geq 10^{-3}$ against finite-sample references whose average bootstrap relative half-width is 27.9%, and transfers to a GCG-style corpus with 2.93% relative error after recalibration. A directional criterion $\mathrm{KL}(p_{\mathrm{good}}\,\|\,p_{\mathrm{bad}})$ ranks rulers consistently with estimation error (Spearman $\rho=0.83$). 

---
# Learned Coordination Conventions in Cooperative MARL: Measuring the Translation Gap Between Theory-Informed Roles and Learned Routing 

**Authors**: Yoosung Hong  

**Link**: [PDF](https://arxiv.org/pdf/2606.29541)  

**Abstract**: Role-semantic assignments provide priors over how heterogeneous agents may coordinate, but cooperative MARL systems instead settle on conventions through decentralized, non-stationary learning, with no guarantee that the resulting structure matches those priors. We study this translation gap between theory-informed role expectations and learned coordination structure through a diagnostic combining a role-routing matrix, formation sensitivity ($\Delta_{\max}$), and gradient/occlusion attribution across three-role MiniGrid and SMACv2 (Terran) environments.
We show that label-conditioned attention produces substantially more concentrated and role-specific routing than flat MLP baselines, remains stable under 3v3--9v9 scaling, transfers zero-shot across team sizes, and is invariant to ally-slot padding. A 5-seed re-evaluation shows partial alignment between learned conventions and designer-specified priors while revealing where small-n noise can manufacture apparent strategic divergence. We present these results as an empirical framework for measuring coordination structure in cooperative MARL rather than as a new equilibrium concept or causal explanation. 

---
# OSWorld2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks 

**Authors**: Mengqi Yuan, Zilong Zhou, Xinzhuang Xiong, Weiming Wu, Jiayang Sun, Jiamin Song, Kaiqian Cui, Bowen Wang, Haoyuan Wu, Yitong Li, Dunjie Lu, Haikong Lu, Qi Zhen, Xinyuan Wang, Jiaqi Deng, Yuhao Yang, Cheng Chen, Boyuan Zheng, Alex Su, Xiao Yu, Hao Zou, Saaket Agashe, Xing Han Lu, Manpreet Kaur, Zhengyang Qi, Vincent Sunn Chen, Frederic Sala, Dayiheng Liu, Junyang Lin, Zhou Yu, Yu Su, Siva Reddy, Xin Eric Wang, Peng Qi, Tianbao Xie, Tao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29537)  

**Abstract**: Existing computer-use benchmarks fail to capture the realism, complexity, and long-horizon demands of real-world computer use, limiting their ability to reveal the limitations of frontier agents. We introduce OSWorld 2.0, a benchmark of 108 long-horizon computer-use workflows across everyday and professional tasks, designed to capture complex and challenging real-world phenomena. Each task represents a realistic end-to-end workflow that takes human users a median of about 1.6 hours to complete and requires an average of 318 tool calls with Claude Opus 4.7 using maximum thinking, compared with about 30 in OSWorld 1.0. OSWorld 2.0 targets challenge phenomena that are common in real workflows yet underrepresented in prior benchmarks, spanning interaction-design challenges such as streaming interaction and dynamic environments, as well as agent-pattern challenges such as cross-source reasoning, implicit-state inference, and visual-spatial precision. Tasks are grounded in authentic input artifacts and cross-referenced against realistic stateful user profile data, and include separate safety reports auditing safety-sensitive execution. Under our primary binary-completion metric at 500 steps, Claude Opus 4.8 with maximum thinking and batched tool calls scores best but still completes only 20.6% of tasks at a 54.8% partial score; GPT-5.5 is far more token-efficient yet plateaus near 13%. These results show that current agents are still far from professional-level computer use: rather than stumbling on basic GUI control or coding, they lose track of constraints, miss information that arrives mid-task, guess rather than ask the user, and skip verification, struggling most when a task hinges on hidden state they must recover. 

---
# UCOB: Learning to Utilize and Evolve Agentic Skills via Credit-Aware On-Policy Bidirectional Self-Distillation 

**Authors**: Songjun Tu, Chengdong Xu, Qichao Zhang, Yiwen Ma, Yaocheng Zhang, Linjing Li, Dong Li, Xiangyuan Lan, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.29502)  

**Abstract**: Skill memories can improve agentic reinforcement learning by reusing past experience as textual guidance, but retrieved skills are not oracular: they may help in one state while misleading the same policy in another. This makes the common privileged-teacher assumption fragile, namely that a skill-conditioned prompt can be treated as a fixed teacher for the no-skill prompt. We introduce UCOB, a framework for learning to utilize and evolve agentic skills via credit-aware on-policy bidirectional self-distillation. UCOB treats skill-conditioned and no-skill prompts as two on-policy context views of the same model, compares their return-to-go within the same task and anchor state, and uses the higher-return view as the local teacher. This local credit signal internalizes useful skill-conditioned behavior, corrects misleading skill usage, and guides task/state skill memory updates, utility-aware retrieval, and reflection self-training. Experiments on agentic tasks, including ALFWorld, WebShop, and Search-QA, show that UCOB outperforms skill-free RL, skill-memory baselines, and self-distillation methods across model scales, with up to 23.5 and 18.0 point gains over SOTA baselines on ALFWorld and WebShop. Ablations and analyses further validate its core mechanisms and efficiency. 

---
# Cognitive World Models for Process-Level Social Influence Evaluation 

**Authors**: Minghui Ma, Bin Guo, Han Wang, Mengqi Chen, Jingqi Liu, Yan Liu, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29495)  

**Abstract**: Social influence dialogue changes user behavior by altering internal cognitive states. The central evaluation question is whether the user's beliefs, desires, intentions, and emotions measurably change over the course of conversation, a process-oriented criterion that neither surface-level text metrics (BLEU/ROUGE) nor single-score LLM judgments can capture. We propose the \textbf{Cog}nitive \textbf{W}orld \textbf{M}odel \textbf{(CogWM)}, an LLM-based user model that reframes multi-turn dialogue evaluation from ``what did the user say'' to ``how did the user's internal cognitive state evolves.'' CogWM jointly predicts BDI/E cognitive states and user utterances and serves as both a user simulator and an evaluation platform, using a three-tier evaluation framework that covers turn-level fidelity, trajectory-level state dynamics, and task-level composite scoring. Trained via our \textbf{S}ummarize-\textbf{a}nd-\textbf{A}llocate \textbf{(SaA)} annotation pipeline on 150,454 user-turn samples across four social influence scenarios, CogWM achieves 77.6\% emotion accuracy (2.1$\times$ over GPT-5.5). In 3600 multi-agent discrimination trials, it distinguishes six commercial agents by their cognitive influence, with Llama-4-Scout ranking first (CTS +0.233). CogWM moves social influence dialogue evaluation from terminal judgment to process tracking. We have released our code\footnote{\scriptsize Code: this https URL} and models\footnote{Model: this https URL}. 

---
# Faults in Our Formal Benchmarking: Dataset Defects and Evaluation Failures in Lean Theorem Proving 

**Authors**: Pawan Sasanka Ammanamanchi, Siddharth Bhat, Stella Biderman  

**Link**: [PDF](https://arxiv.org/pdf/2606.29493)  

**Abstract**: Benchmarks for LLM-assisted theorem proving in Lean are often treated as intrinsically reliable because every solved instance comes with a machine-checked proof. However, the kernel only checks that a proof establishes a \emph{formal} statement; it does not verify that the statement faithfully encodes the intended informal problem, nor that evaluation harnesses are robust to trivial or adversarial solutions. We audit five widely used Lean theorem-proving benchmarks and their forks, using corpus-scale static checkers to surface 4,833 findings, including 398 mechanically certified issues such as counterexamples, vacuous theorems, and unsound axioms. We also document semantic defects such as missing hypotheses, problem simplification, incomplete or incorrect translations, and Lean-specific specification hazards. Beyond dataset construction, we survey evaluation-time failure modes and show, on corrected subsets, that defects can both inflate and deflate reported prover scores. We propose a fault taxonomy, a suite of automated checkers and recall-oriented semantic audit prompts, and release standards to guide the creation of formal math datasets and to make evaluation more reproducible and trustworthy. Our checkers, audit prompts, and corrected dataset snapshots are available at this https URL. 

---
# Agent-Computer Observation Interfaces Enable Dynamic Computer Use 

**Authors**: Bojie Li, Noah Shi  

**Link**: [PDF](https://arxiv.org/pdf/2606.29472)  

**Abstract**: SWE-agent established the action interface as an underexplored design axis for software-engineering agents; we make the analogous case for the observation interface in computer-use (CU) agents. Current CU agents, closed and open-source alike, tie observation to action--one screenshot every 3-5 s, no audio--leaving them blind and deaf between screenshots to video, animations, transient UI events, meetings, and spoken instructions. We introduce the Agent-Computer Observation Interface (AOI), a model-agnostic perception layer that decouples continuous, adaptive observation from discrete actions through three gated components: inter-step keyframe capture, volume-gated audio transcription, and CU-model-generated visual narration that persists as text. Each produces almost nothing on static, silent content, reducing to the standard loop without degrading it.
On DynaCU-Bench (100 dynamic browser tasks plus a 50-task static control), CU models from 7B to frontier scale gain +17 to +48 pp over their screenshot baselines with zero retraining, turning tasks that are near-impossible from periodic screenshots into largely solved ones. The gap is starkest on audio: on a spoken-content subset AOI agents solve every task, whereas streaming voice models hear accurately but cannot act on what they hear without the scaffold. The decomposition is as informative as the headline gain: keyframe selection turns out not to matter--the value comes from narrating captured frames into persistent text--and the interface is not a fixed bundle, since on a newer model (Gemini 3 Flash) the keyframe stream actively regresses through image-token dilution, so its components must be selected per model rather than shipped as one configuration. 

---
# How Much Due Diligence Before You Bid? Learning in Intractable Takeover Auctions 

**Authors**: Zain Naboulsi  

**Link**: [PDF](https://arxiv.org/pdf/2606.29457)  

**Abstract**: When two companies bid to buy the same target, no one knows exactly what the target is worth. Each bidder pays for due diligence: costly, imperfect homework that sharpens its own private estimate before it bids. How much of that homework is worth buying? We build a simple computer model of the bidding contest and let it teach itself to bid well by playing against itself, the way a game engine learns chess. The economic question, how much diligence pays for itself, and the computational question, when the contest becomes too complex to solve exactly, are both controlled by a single thing: how many pieces of private information a bidder carries. Our main finding is that the right amount of diligence is modest and finite. It falls as diligence gets more expensive, and it falls further when both sides are doing their homework, because competition erodes the value of knowing more. We also test a recent claim from AI research: that simple, general self-play methods can rival the specialized, expensive algorithms usually built for games like these. Running on an ordinary laptop with no costly frontier AI, we find the simple methods are the best of the self-learning approaches, though purpose-built exact methods still win whenever the game is small enough to solve outright. The simple methods earn their keep only once the game grows too large to solve exactly, which is the regime real deals live in, and there we show they still find strong bidding strategies. The contribution is threefold: a cheap, reproducible way to study deal-making under uncertainty; a concrete, model-based answer to how much due diligence is worth buying; and evidence about when lightweight, general-purpose AI is good enough to replace specialized methods. We release all the games, code, and experiments. 

---
# FADE: Mitigating Hallucinations by Reducing Language-Prior Dominance in Large Vision-Language Models 

**Authors**: Yichen Guo, Kai Tang, Fenglai Lin, Yiding Sun, Dongshuo Zhang, Wenya Wang, Lin William Cong, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29431)  

**Abstract**: Despite the impressive capabilities of Large Vision-Language Models (LVLMs), they remain susceptible to hallucination, generating content inconsistent with the input image. Recent studies attribute this to the dominance of language priors over visual inputs and employ contrastive decoding methods to mitigate this dominance, but the mechanistic origin remains unexplored. We investigate the information flow through each transformer layer and find that attention modules consistently aggregate visual evidence, while FFN modules at critical layers act as the source of language priors. These priors can override visual evidence, causing correct predictions in intermediate layers to drift toward incorrect outputs. Based on this insight, we propose FADE (FFN Attenuation for DEcoding), a training-free method that attenuates FFN outputs to reduce language-prior dominance. Evaluations on POPE, CHAIR, and MME benchmarks across LLaVA-1.5, mPLUG-Owl2, and InstructBLIP show that FADE effectively mitigates hallucinations while preserving inference efficiency. 

---
# Mixture of Debaters: Learn to Debate at Architectural Level in Multi-Agent Reasoning 

**Authors**: Dayong Liang, Kaisong Gong, Yi Cai, Changmeng Zheng, Xiao-Yong Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.29425)  

**Abstract**: Existing multi-agent debate frameworks suffer from two critical limitations: they rely on static architectures where agent roles and coordination patterns are fixed at design time, and they require instantiating multiple model copies, incurring substantial computational overhead. We propose Mixture of Debaters (MoD), a unified framework that enables dynamic self-debate within a single model by leveraging the Mixture-of-Experts paradigm. We address three key challenges in adapting MoE for dialectical reasoning: (1) dual-routing that decouples role allocation from process flow, dynamically determining when to debate versus when to synthesize; (2) momentum switching that smooths token-level routing with local context, reducing expert-switch jitter; and (3) unified self-debate that encapsulates diverse debating personas into lightweight expert modules, eliminating inter-agent communication while preserving behavioral diversity. Extensive experiments on multimodal benchmarks demonstrate that MoD outperforms both single-model baselines and conventional multi-agent systems, achieving superior accuracy with 3.7x lower latency and 87% reduction in token this http URL source code can be accessed at this https URL. 

---
# LLM-Guided Planning for Multi-hop Reasoning over Multimodal Nuclear Regulatory Documents 

**Authors**: Mingyu Jeon, Bokyeong Kim, Suwan Cho, Jae Young Suh, Yonggyun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29399)  

**Abstract**: Reviewing nuclear regulatory documents requires multi-hop reasoning across tens of thousands of pages, where judgments depend on evidence assembled across multiple chapters. We frame this task as planning: an LLM-based agent observes the evidence collected so far, picks the next document fragment to inspect, and stops when the evidence is sufficient. The agent operates over a vectorless document tree using browse, read, and search tools, and maintains a dynamic knowledge graph (KG) as state. On a 200-question benchmark over NuScale Final Safety Analysis Report (FSAR) documents, the system reaches 81.5% accuracy with a RAGAS Faithfulness of 0.93. The dominant performance factor is planning: against PageIndex, which uses the same document tree without state-conditioned action selection, the gap is +38.0pp (43.5% to 81.5%, p<0.001). The system also outperforms LightRAG (73.0%, p<0.05), HippoRAG (70.5%, p<0.01), and GraphRAG (49.5%, p<0.001), and matches RAPTOR (75.5%, p=0.11) without offline indexing. Edge inference adds 2.8x cost without raising accuracy; we retain it as a traceability module. Of 7,391 inferred edges, 3 Violates edges (0.04%) flag scope boundaries (Q058) and partial conformance (Q176) as typed annotations that a human reviewer can audit. 

---
# Diagnosing and Repairing Factual Errors in RAG under Budget Constraints 

**Authors**: Soroush Hashemifar, Havva Alizadeh Noughabi, Fattane Zarrinkalam, Ali Dehghantanha  

**Link**: [PDF](https://arxiv.org/pdf/2606.29377)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves the factuality of large language models by grounding responses in external evidence, yet real-world deployments remain fragile. Failures often stem from missing or weakly relevant evidence, as well as from generation that does not faithfully reflect the retrieved context. Many existing approaches rely on fine-tuning, privileged access to internal model signals, or resource-insensitive escalation strategies, which limits their practicality in black-box and budget-constrained settings. We propose D2R-RAG (Diagnose-to-Repair RAG), a model-agnostic and resource-aware framework that combines lightweight failure diagnosis with adaptive repair. D2R-RAG derives interpretable failure signatures from observable signals in the query, retrieved evidence, and generated response, and then selects from a small set of corrective actions under explicit latency and VRAM constraints. Experiments on FEVER and HotpotQA show that D2R-RAG improves reliability over recent baselines and achieves better accuracy--efficiency trade-offs across multiple compute budgets. The code is available at this https URL. 

---
# When LLMs Develop Languages: Symbolic Communication for Efficient Multi-Agent Reasoning 

**Authors**: Zhengqi Pei, Qingming Huang, Shuhui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29354)  

**Abstract**: Chain-of-Thought (CoT) improves large language models (LLMs) on difficult reasoning tasks, but it often incurs long natural-language rationales that are poorly aligned with efficient machine reasoning. We propose Communicative Language Symbolism Routing (CLSR), a test-time framework in which multiple LLM agents autonomously invent, evolve, and share compact Language Symbolism Frameworks (LSFs), while a latent-free router adaptively selects and composes these languages per query to optimize the accuracy-token trade-off. Unlike prompt optimization that refines surface instructions, CLSR treats each LSF as a reusable symbolic protocol with compact symbols, usage rules, and a message-passing contract, and improves it through an evolutionary loop driven by correctness and token cost. At inference time, the router may invoke a single low-cost LSF call, ensemble multiple LSFs, or execute a multi-round LSF composition protocol on harder queries. Across challenging benchmarks, CLSR reduces latency-oriented generated token completion by $3\sim 6\times$ compared to standard CoT while maintaining accuracy. We further derive an information-theoretic lower bound on token cost under arbitrary symbolism and show that, under an interpreter-realizability premise, multi-round LSF protocols conditionally subsume program-execution pipelines. Code is publicly available (this https URL). 

---
# PHF: Privileged Hidden Flow for On-Policy Self-Distillation 

**Authors**: Yuhan Li, Mingxu Zhang, Dazhong Shen, Ying Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.29340)  

**Abstract**: On-policy self-distillation (OPSD) trains a reasoning model on rollouts sampled from its own policy by matching a privileged teacher that also sees verified reference solutions. Existing OPSD objectives supervise only the output distribution, so privileged context affects training through a token-level divergence without directly supervising the internal computation that produced that distribution. We propose Privileged Hidden Flow (PHF), which additionally distills how a privileged teacher's hidden states move along the same rollout. Rather than forcing each student hidden vector to match the teacher vector at the same token position, PHF aligns token-to-token transition directions and trajectory geometry over selected generated positions. The all-layer recipe also includes an adjacent-layer relation computed from these same transitions, without pointwise hidden-state imitation. Under the same 100-step training schedule, PHF improves the Average@12 aggregate over our reproduced OPSD baseline on Qwen3-1.7B, 4B, and 8B, with observed gains of about +2.2, +1.5, and +1.7 points. The transport objective is exactly invariant to shared trajectory offsets; its local geometry term is also invariant to orthogonal transformations of transition directions. Ablations distinguish the fixed PHF recipe from pointwise hidden-state matching, single-channel transition losses, and layer-subset choices, supporting PHF as a compact hidden-flow extension to OPSD. 

---
# Hierarchical Experimentalist Agents 

**Authors**: Abhranil Chandra, Sankaran Vaidyanathan, Utsav Dhanuka, Varun Gandhi, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2606.29315)  

**Abstract**: Large language models (LLMs) are increasingly used to take actions in the real world and support human decision-making, yet most agents rely on parametric knowledge, fixed post-training data, retrieval, or search. This paradigm breaks down in novel domains and for sophisticated queries that cannot be answered from prior knowledge alone. Knowing the laws of physics, for instance, does not by itself enable LLMs to answer queries or complete long-horizon tasks in a complex physical system. To address this, we introduce Hierarchical Experimentalist Agents (HExA), an in-context self-improvement framework to learn from active experimentation. HExA iteratively designs and refines query-relevant experiments, learns a reusable library of composable skills from experience, and integrates experimental evidence to answer queries or take actions. HExA is training-free, compatible with any black-box model, and does not require external supervision, oracles, or offline data. To evaluate active experimentation, we introduce Interphyre, a tool-calling benchmark built on the PHYRE 2D procedural physics environment, where agents propose interventions and test hypotheses through simulation APIs. Experiments show that current LLM agents struggle in these settings, especially on the hardest levels of Interphyre. Claude Sonnet 4.6 achieves only 2% success, while HExA improves the same model to up to 77% success. HExA also improves open-weight models and outperforms agentic baselines such as ReAct and Reflexion. Moreover, using only skills learned from easier levels and transferred without active experimentation, HExA achieves 44% success, demonstrating the reusability and generalization of its learned skills. Overall, HExA shows that learning through active experimentation can help agents discover useful knowledge, acquire reusable skills, and make efficient progress on novel long-horizon tasks. 

---
# Process Advantage Signal Shaping: A Paradigm-Agnostic Middleware for Process-Supervised RL in LLM Reasoners 

**Authors**: Chao Wang, Hongtao Tian, Tao Yang, Yunsheng Shi, Ting Yao, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.29296)  

**Abstract**: Group Relative Policy Optimization (GRPO) is a default recipe for process-supervised reinforcement learning of LLM reasoners, and dense process supervision -- via learned process reward models (PRMs) or on-policy-distillation KL signals -- is a common way to densify its otherwise weak outcome reward. Layering such a step-level signal on top of GRPO's group-standardized advantage, however, exposes three structural pathologies: \emph{channel contamination} between the pooled process, outcome, and format streams at group standardization; \emph{resolution mismatch} between the granularity of the process signal and the granularity of the logical decisions being credited; and a \emph{cumulative trap} by which GRPO's return-to-go sum surfaces either length inflation or truncated exploration depending on the sign regime of the signal. We propose \textbf{PASS} (\emph{Process Advantage Signal Shaping}), a compact middleware that sits between any scalar step-level process signal and GRPO's clipped surrogate and addresses the three pathologies in turn: \emph{Advantage Fusion} standardizes the three streams independently within each group, \emph{Chunk-by-Value} derives value-homogeneous chunks from the signal itself and broadcasts credit within each chunk, and \emph{Divide-Length} converts the cumulative objective into an average-value-density score. We validate PASS across two domains and two process-signal paradigms -- a learned PRM on mathematical reasoning and an on-policy-distillation KL signal (with a generalized variant) on multi-hop question answering -- and under two group-standardization operators. In every regime PASS delivers a consistent pass@1 gain over the corresponding GRPO baseline. 

---
# The Complexity Ceiling Benchmark: A Multi-Domain Evaluation of Sequential Reasoning Under Depth Scaling 

**Authors**: Shubh Chapra, Dhruv Kumar, Murari Mandal, Yash Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2606.29278)  

**Abstract**: We introduce the Complexity Ceiling Benchmark (CCB), a controlled evaluation of how language-model reasoning decays as the number of required sequential steps grows. CCB fixes the semantic content of a task and varies only its depth N in {5,...,50} across three structurally distinct regimes: grounded spatial state-tracking, abstract symbolic pointer manipulation, and transitive relational inference. Across 6,000 trials over five frontier and open-weight LLMs we find a consistent pattern of geometric per-step decay with widely separated domain ceilings: on the first two regimes the strongest models retain pd>0.92 across N=50; on the third every model collapses by N=5, with the best model's 50%-success horizon at H0.5~4.7 steps despite pd=0.863. A trace-level metric (TFBC) shows that 14.5% of correct answers across the benchmark are reached via incorrect intermediate reasoning. Forced verbose state-tracking does not move the ceiling (McNemar p=1.000), and the mean step at which reasoning first diverges, k*, predicts within-domain accuracy better than parameter count. CCB and the geometric decay model together reduce a model's long-horizon reasoning profile to one interpretable number per task family. 

---
# When Summaries Distort Decisions: Information Fidelity in LLM-Compressed Financial Analysis 

**Authors**: Hoyoung Lee, Suhwan Park, Seunghan Lee, Jun Seo, Jaehoon Lee, Sungdong Yoo, Minjae Kim, CheolWon Na, Zhangyang Wang, Zach Golkhou, Minkyu Kim, Sotirios Sabanis, Alejandro Lopez-Lira, Dhagash Mehta, Soonyoung Lee, Chanyeol Choi, Wonbin Ahn, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.29251)  

**Abstract**: Financial decision-makers face more information than they can directly inspect, making context compression necessary. Yet when large language models (LLMs) compress financial source material, they can alter the investment judgment supported by the original source. We frame this problem as information fidelity: compression loses fidelity when it changes the decision induced by the source. In agentic systems, such losses may recur across intermediate steps and amplify throughout the decision process. Across financial filings and earnings-call transcripts, we find that LLM-based compression can produce fluent and factually plausible compressed contexts that nevertheless alter downstream decisions. We analyze two diagnostic patterns associated with fidelity loss: decontextualization, where salient evidence is retained but separated from the caveats and contextual qualifiers needed for correct interpretation, and model dependency, where different compressors expose different views of the same source. We then propose Agentic Context Compression, which generates multiple candidate compressions and audits their disagreements against the original source. Our results suggest that financial compression should be evaluated not only by efficiency or factuality, but also by its ability to preserve decision-relevant context. 

---
# SurgVLA-Bench: Towards Evaluating Vision-Language-Action Models for Laparoscopic Surgical Robotics 

**Authors**: Jiashuo Sun, Yue He, Wenxuan Liu, Tao Mao, Jiazheng Wang, Xiang Chen, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29247)  

**Abstract**: Vision-Language-Action (VLA) models represent a promising direction for embodied intelligence in surgical robotics. Despite the prevalence of VLA benchmarks for general robotics, standardized evaluation platforms specifically designed for surgical contexts remain absent. To address this limitation, we present SurgVLA-Bench, the first comprehensive benchmark for evaluating VLA models in laparoscopic surgical robotics. Leveraging the SurRoL simulation platform, we construct a hierarchical task taxonomy ranging from atomic actions to complete surgical procedures, complemented by a multi-dimensional evaluation framework assessing action accuracy and semantic consistency. We then systematically evaluate two representative paradigms, including autoregressive models such as OpenVLA, and flow matching models such as $\pi_{0}$, $\pi_{0.5}$, and SmolVLA. Our experiments show that autoregressive models tend to excel in semantic understanding, while flow matching models often achieve higher task precision but may face generalization trade-offs. However, even the best-performing models remain far from satisfactory, as the constrained endoscopic field of view, restricted viewing angles, and frequent occlusions persist as fundamental physical bottlenecks. The code and data are available at this https URL 

---
# PolicyGuard: A Dialogue-Grounded Sub-Agent Verifier for Policy Adherence in LLM Agents 

**Authors**: Seongjae Kang, Taehyung Yu, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29225)  

**Abstract**: LLM agents handle user requests on behalf of organizations through tool calls and must follow the company policies stated in their system prompts. Prior work approaches this as a safeguarding problem -- external checks that block non-compliant agent actions. We argue that policy adherence is a broader problem: real workflows unfold across many turns, require explicit user confirmation and prerequisite reads, and hinge on the content of the dialogue rather than on any single argument value. Meeting this bar requires (i) full conversation context, (ii) self-reasoning over the policy and the current dialogue, and (iii) conversation-specific remediation that guides the agent's next turn -- three capabilities that prior safeguard work has often underestimated. We introduce POLICYGUARD, a sub-agent verifier that shares the agent's view of the dialogue, reasons over the policy in context, and provides actionable feedback for the agent's next turn. On tau^2-BENCH airline across three vendors (GPT-5.4, Claude Sonnet 4.6, Gemini 2.5 Pro) with four trials per setting, POLICYGUARD improves PASS4 by +12.0 / +6.0 / +12.0 pp. Per-call analyses show POLICYGUARD achieves higher policy-violation recall while blocking roughly half as often as argument-level guards. 

---
# A Cognition-Emotion-Personality Framework for Modeling Human-Like Awareness and Behavior in Emergency Evacuations 

**Authors**: Zoi Lygizou, Michalis Zervas, Helena G. Theodoropoulou, Vasilis Zafeiropoulos, Dimitris Kalles, Chairi Kiourt  

**Link**: [PDF](https://arxiv.org/pdf/2606.29212)  

**Abstract**: Agent-based evacuation simulations are widely used to study crowd behavior during emergencies, but many models rely on assumptions such as perfect event awareness, complete exit knowledge, and fully rational decision-making. This paper presents an extended evacuation framework that integrates cognitive, emotional, social, and personality-related mechanisms into a unified model of human behavior under uncertainty. The framework incorporates a dynamic event-awareness mechanism based on a continuous Event Certainty Level, a memory-based representation of exit knowledge subject to acquisition, forgetting, and recall, a continuous fear model in which panic emerges as a high-intensity state, and an OCEAN-based personality representation. Neuroticism is explicitly integrated into the emotional model, influencing fear generation, escalation, social contagion, and recovery. Behavioral heterogeneity is further captured through individualized decision thresholds that affect responses to perceived risk. The framework is evaluated through simulation experiments examining the effects of spatial familiarity, memory robustness, decision sensitivity, emotional dynamics, and personality variation. Results show that cognitive, emotional, and personality-driven processes substantially influence evacuation dynamics, reducing evacuation efficiency and generating realistic crowd phenomena such as delays, confusion, injuries, and socially influenced behaviors. The proposed framework provides a more realistic representation of human behavior in emergency evacuations and supports systematic investigation of the interactions between cognition, emotion, personality, and crowd dynamics. 

---
# AI Trading's Alpha Singularity: Emergent Market Reasoning through Agent-to-Agent Self-Evolution 

**Authors**: Yuqi Li, Siyuan Liu, Bingjun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29194)  

**Abstract**: Automated alpha mining holds the scoring function fixed and varies the search algorithm over it. A search that converges against a fixed scorer overfits whatever the scorer cannot penalize, a primary cause of the out-of-sample generalization gap. We treat the scoring function as a search artifact alongside the alpha factors and study what conditions make this joint search admissible. Sealed Joint Search (SJS) is a framework: a set of structural conditions on information flow in an autonomous-discovery system that prevent joint search from collapsing into self-confirmation while keeping the evaluator sealed. Conditions cover role decomposition, typed inter-role communication, provenance-sealed reads, versioned stores, and substrate-local promotion. Agora tests SJS empirically: five LLM agent classes communicate via three channels, evolving eight skill libraries, with alpha libraries built on AlphaGen operators. Three evaluators write reports aggregated into one brief, carrying forward disagreement instead of voting. We run Agora for 100 rounds on CSI 1000 and evaluate on a 91-day 2026 holdout sealed from all LLM inputs. Agora achieves holdout Sharpe +1.87; best baseline +1.334 at favorable seed and -0.755 cross-seed mean. Pre-loading Agora's two metrics into a frozen-library ablation recovers only +0.40 of the +2.25 Sharpe gap, and adding PPO without library evolution worsens the gap. The two metrics emerge rather than being designed. Caveats: single-seed run, short-side concentrated signal, intended for long-short. 

---
# Evidence-Informed LLM Beliefs for Continual Scientific Discovery 

**Authors**: Dhruv Agarwal, Reece Adamson, Andrew McCallum, Peter Clark, Ashish Sabharwal, Bodhisattwa Prasad Majumder  

**Link**: [PDF](https://arxiv.org/pdf/2606.29182)  

**Abstract**: Open-ended scientific discovery with large language models (LLMs) increasingly operates as a long-horizon loop of hypothesis search and verification, where a reward signal guides which hypotheses to test next. A notable recent example is AutoDiscovery, which uses "Bayesian surprise" - the belief shift an LLM undergoes after observing evidence for a hypothesis - as both a discovery metric and a reward for search. We first observe that AutoDiscovery treats surprisal as a static quantity, while surprisal in human reasoning is non-stationary - it is defined relative to beliefs that evolve with experience, a prerequisite for continual scientific discovery. We address this mismatch with evidence-informed LLM beliefs: priors updated with evidence from previous hypotheses to compute non-stationary surprisal for new hypotheses. We compare in-context belief-updating mechanisms and find that embedding-based retrieval-augmented generation over prior discoveries best anticipates eventual posteriors, identifying 37.5% of static surprisals as spurious. We then modify search to avoid these spurious rewards and prioritize hypotheses that remain surprising under non-stationary beliefs. Concretely, we introduce two complementary changes to the original search procedure: belief-update filtering and diversity maximization. Across five discovery domains, our method increases accumulated non-stationary surprisal by 30.62% on average compared to the original search procedure, demonstrating that continual scientific discovery with LLMs requires not only better belief measurement but also search procedures that avoid redundancy and encourage diversity. 

---
# Measuring Graph-to-Graph Semantic Similarity in Knowledge Graphs: An Empirical Evaluation of Knowledge Graph Embeddings 

**Authors**: Seungryeol Baek, Wooseok Sim, Hogun Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.29180)  

**Abstract**: A Knowledge Graph (KG) represents facts as structured triples and is widely used to organize relational knowledge across diverse domains. Just as textual information ranges from words and sentences to complete documents, KG information can be interpreted at multiple levels, from entities, relations, and triples to subgraphs and entire KGs. However, existing KG embedding methods mainly focus on entities, relations, and triples, leaving graph-level semantics largely unaddressed. Conventional graph-level methods, which typically compare graphs based on structural patterns, are also insufficient because structural similarity alone cannot guarantee semantic similarity between KGs. To evaluate how well different methods capture such graph-level semantic information, we study graph-to-graph semantic similarity, which determines whether a pair of KGs represents semantically corresponding underlying information. To obtain reliable ground-truth correspondences, we construct a semantic matching dataset by modifying text documents, extracting KGs from both original and modified documents, and transferring their known correspondences to KG pairs. We compare text-based, structure-based, and KG embedding-based approaches on each dataset. For the KG embedding-based approach, we introduce two scoring functions: \textit{EmbPairSim}, which uses maximal pairwise entity similarity, and \textit{AvgEmbSim}, which uses a frequency-weighted centroid. Experiments on WikiText-2 and CC-News show that \textit{EmbPairSim} achieves up to 5.3 pp higher MRR than Sentence-BERT while using substantially fewer parameters. These results suggest that KGE representations can serve as compact and effective signals for graph-to-graph semantic similarity in KGs. Our code is available at this https URL. 

---
# Selective Memory Retention for Long-Horizon LLM Agents 

**Authors**: Pranath Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2606.29178)  

**Abstract**: When does retention matter for memory-augmented LLM agents? We study this with TraceRetain, a lightweight framework for bounded external memory in frozen LLM agents that scores entries by interpretable features (success, age, access frequency, redundancy, specificity, similarity, downstream utility) and evicts the lowest-scoring ones at capacity. On clean ALFWorld with gpt-5-mini, external memory robustly improves over no memory across two seeds, but differences among bounded retention policies fall within Wilson 95% CIs: clean ALFWorld at T=100 to T=200 does not naturally exhibit the memory pollution retention is designed to address. Under a controlled noisy-write stress (75% synthetic distractors), unbounded memory and FIFO-K50 degrade on Precision@5 (20.2% to 12.4% and 15.8% to 3.8%) while TraceRetain-CEM is essentially unchanged (16.9% to 16.6%) and preserves 97/100 task success. The mechanism: unbounded memory has the highest mean similarity (0.87) but lowest precision, indicating failed distractors close to the query in embedding space. Held-out in-distribution evaluation shows memory-augmented policies solving 47 to 49 of 50 tasks vs. 39/50 for no memory. Bounded retention buys memory and step efficiency on saturated clean benchmarks at no task-success cost, and only differentiates from cache heuristics when streams contain noise. 

---
# Direct Causation in International Humanitarian Law and the Challenge of AI-Mediated Civilian Cyber Operations 

**Authors**: Alice Saito, Harold Godsoe, Phan Xuan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2606.29175)  

**Abstract**: International humanitarian law protects civilians from direct attack unless and for such time as they take direct part in hostilities, with the ICRC's 2009 Interpretive Guidance operationalising this rule through a three-criterion cumulative test. This paper argues that AI-mediated civilian cyber operations challenge the direct causation element of this test in a structurally specific way: when a civilian deploys an autonomous multi-agent cyber system of the kind recently demonstrated in offensive AI research, the "one causal step" standard fails because harm is produced by system-generated decisions made after human disengagement, and the integral-part requirement does not extend because it presupposes downstream human contributors whose conduct can be independently classified. The framework therefore defaults to treating such deployments as indirect participation, in tension with its purpose of capturing civilians who personally take part in hostilities. Beyond the doctrinal analysis, this paper identifies goal-specification granularity as the property on which the integral-part test's concreteness component implicitly turns, classifies AI-mediated operations along a five-level spectrum, and argues that existing technical AI governance instruments do not log or report this property. 

---
# Pooled Leaderboards Hide System-Specific Winners: A Reporting-Protocol Audit of Offline Root-Cause Analysis Benchmarks 

**Authors**: Lining Hu, Ting Liu, Yuzhuo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29159)  

**Abstract**: Offline root-cause-analysis (RCA) benchmarks commonly rank methods by a single pooled top-1 accuracy across multiple subsystems, and engineers often read the pooled winner as a recommendation for their own subsystem. We audit that reading on three public RCA benchmark families -- OpenRCA, RCAEval, and PetShop -- covering 11 subsystems and 778 matched scoring units. To keep pairwise comparisons on identical cases, the main analysis retains four methods or comparators with complete coverage: BARO, a CD-1min adapter, max-$|Z|$, and per-service alert-count. All six pairwise comparisons show subsystem-level effects of both signs, every random-effects 95\% prediction interval crosses zero, and case-level interaction tests reject exchangeability in 5 of 6 pairs. Leave-one-system-out selection picks the lower-scoring method on up to 5 of 11 held-out subsystems, with regret reaching 24.8 pp on RCAEval / Sock-Shop. We release a 320-line audit module; given a matched RCA benchmark score table, it recomputes the same per-subsystem stability checks alongside pooled scores. 

---
# Flow Reasoning Models: Scaling Reasoning Through Iterative Self-Refinement 

**Authors**: Alec Helbling, Andrey Bryutkin, Mauro Martino, Nima Dehmamy, Hendrik Strobelt  

**Link**: [PDF](https://arxiv.org/pdf/2606.29150)  

**Abstract**: Discrete flow models have recently shown promising performance on few-step text generation; however, when naively applied to structured reasoning tasks such as Sudoku and Zebra puzzles, they converge confidently to incorrect answers (solving only $\sim$36% of Sudoku puzzles). We introduce Flow Reasoning Models (FRMs), a training and test-time-scaling framework for structured reasoning with flow models. We make the observation that, despite their poor solve rate, flow models can act as their own verifiers. A correct answer is a stable fixed point of the denoising dynamics, returning to itself when re-noised and re-solved. This enables a test-time-scaling paradigm: propose many candidate solutions and keep those that are dynamically stable, which alone reaches high solve rates on Sudoku-Shah (~$100\%$) and Zebra ($95.9\%$). This even generalizes to harder out-of-distribution puzzles like Sudoku-Extreme ($96.1\%$), without ever training on that distribution. This pure search, however, wastes a great deal of computation generating incorrect candidate solutions. We therefore design a training recipe to improve the base model's efficiency. First, we train flow models with a self-conditioning channel and close it at inference, letting them refine their own past predictions. Second, we train models to avoid their own failed generations using direct preference optimization. These changes substantially improve the base model's efficiency, letting it reach $99.2\%$ on Sudoku in just $7$ forward passes, over $8\times$ fewer than the strongest matched masked-diffusion baseline we compare needs for the same accuracy. When combined with test-time scaling, this lets flow models solve hard out-of-distribution puzzles (e.g. Sudoku-Extreme) far more efficiently. 

---
# HiComm: Hierarchical Communication for Multi-agent Reinforcement Learning 

**Authors**: Runze Zhao, Dongruo Zhou, Sumit Kumar Jha, Nathaniel D. Bastian, Ankit Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.29126)  

**Abstract**: Cooperative multi-agent reinforcement learning (MARL) often relies on communication to mitigate partial observability, yet most existing protocols treat messages as flat dense vectors detached from the structure of the observations they summarize. This design overlooks an important source of inductive bias in many cooperative environments, where observations naturally follow a hierarchy such as groups and entities. We propose \textsc{HiComm}, a plug-in communication module that grounds messages in the sender's hierarchical observation. \textsc{HiComm} is receiver-driven: the receiver issues a query, and the hierarchy is resolved through a three-stage decoding process that first selects a group, then a sender, and then an entity within that group, returning the corresponding feature slice as the message. This converts communication from unstructured vector transmission into structured information retrieval over the sender's observation hierarchy. We instantiate this mechanism with Straight-Through Gumbel-Softmax for differentiable discrete selection and a lightweight shared projection design that attaches to standard MARL pipelines. Experiments across cooperative MARL tasks with different observation structures and coordination demands show that \textsc{HiComm} matches or outperforms representative learned communication baselines while reducing communication volume by up to $23\times$ per receiver per episode. 

---
# Characterizing Large Language Model Agentic Workflows: A Study on N8n Ecosystem 

**Authors**: Yutian Tang, Yuming Zhou, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.29116)  

**Abstract**: Large Language Models (LLMs) are rapidly being adopted in low-code and no-code automation platforms, where non-expert users design workflows that combine natural language understanding with external services and APIs. LLM agents are LLM systems that use LLMs as a core "brain" to reason, plan, and autonomously execute complex, multi-step tasks. In this paper, we present the first large-scale empirical study of LLM agentic workflows in low-code automation platforms. We analyze more than 6,000 publicly available n8n workflows and examine four aspects of their design: task distribution, structural and tool use patterns, reliability mechanisms, and autonomy levels. Our analysis shows that LLM workflows are not merely prompt response pipelines. Instead, LLMs are commonly embedded within broader automation structures involving control logic, external tools, communication services, storage systems, and human review points. We further find that while many workflows include lightweight post-processing or routing logic after LLM execution, explicit reliability mechanisms such as structured fallback paths, repair loops, failure-specific alerts, and human approval gates remain relatively uncommon. These results reveal a gap between the increasing deployment of LLM agents in practical automation ecosystems and the limited engineering support for reliability, safety, and governance. Overall, our study provides ten empirical findings and five research takeaways for researchers, platform developers, and practitioners seeking to understand and improve real-world LLM agentic workflows. 

---
# Managing the Human Fallback: Skill Investment Under Improving AI and Worker Mobility 

**Authors**: Simrita Singh, Naireet Ghosh, Tinglong Dai  

**Link**: [PDF](https://arxiv.org/pdf/2606.29111)  

**Abstract**: When firms deploy autonomous AI, they must decide how much work to leave to the system and how much to keep workers engaged. This decision affects current output and future human capital. We develop a parsimonious two-period model in which AI may outperform the worker when it functions, but may fail with positive probability. A firm chooses worker engagement; engagement lowers current output for below-benchmark workers, but changes future skill through learning and erosion. We distinguish two dimensions of AI progress: capability, the system's output when it works, and reliability, the probability that it works. In a single-firm benchmark, engagement is valuable only as fallback investment. The firm engages the least-skilled workers most, because they have the largest skill gaps and are least costly to bring toward a useful fallback level. With worker mobility, engagement also affects labor-market sorting: workers prefer jobs that build more valuable skill trajectories. This sorting motive targets higher-skill workers near the AI frontier, where skill gains are more valuable and engagement is less costly. Mobility can therefore reverse the engagement pattern, shifting investment from the least-skilled toward the most-skilled workers below the AI benchmark. Mobility also reshapes how AI progress affects engagement: greater capability raises engagement by increasing the value of the skill trajectory a firm offers, whereas greater reliability can raise or lower it because it reduces fallback need while also changing learning opportunities. Under worker mobility, human-AI work design becomes a problem of human-capital investment, in which allocating work today shapes future skill. 

---
# Low-cost concept-based localized explanations: How far can we get with training-free approaches? 

**Authors**: Darian Fernández-Gutiérrez, Rafael Bello, Marilyn Bello, Natalia Díaz-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2606.29069)  

**Abstract**: Concept-based Explainable AI (C-XAI) seeks human-understandable explanations grounded in semantic concepts, yet validation is limited by the scarcity of fine-grained concept annotations. We evaluate whether mid-scale Multimodal Large Language Models (MLLMs) can perform localized concept naming under strict zero-shot conditions by assigning labels to bounding-box regions at both object and part levels. We propose a reproducible zero-shot evaluation protocol for Concept Naming (CoNa) with (i) closed-set, category-constrained prompting for moderate vocabularies and (ii) Open-CoNa, an embedding-similarity-based strategy for large label spaces. Experiments with four MLLMs (7B-32B) show consistent performance trends across datasets, reaching 62%-88% object-level exact-match accuracy, highlighting the potential of training-free concept annotation from localized regions. We discuss limitations and failure modes and release a reproducible framework to support future low-cost C-XAI research. 

---
# Memory as an Attack Surface in LLM Agents: A Study on Multiple-Choice Question Answering 

**Authors**: Shahnewaz Karim Sakib, Anindya Bijoy Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.29030)  

**Abstract**: AI agents extend conventional large language model (LLM) applications by integrating language understanding with task execution, external tool use, and memory mechanisms. While memory allows agents to retain prior interactions and provide more personalized and context-aware responses, it also introduces a new vulnerability: information stored in memory can influence future outputs even when the current query is clean. In this paper, we investigate memory manipulation in LLM-based agents for multiple-choice question answering. We first design and implement an LLM-based AI agent with an external memory component that stores and retrieves task-relevant information. We then introduce basic memory manipulation scenarios in which misleading or corrupted memories are inserted into the agent before it answers multiple-choice questions. Using a controlled experimental setup, we compare the agent's performance before and after memory manipulation and measure changes in answer accuracy, attack success rate, and selection of manipulated options. Our results show that even simple memory manipulations can noticeably affect the agent's final answers, causing it to select incorrect options despite receiving clean and well-formed questions. 

---
# Preventing Error Propagation in Multi-Agent AI through Runtime Monitoring 

**Authors**: Shahnewaz Karim Sakib, Anindya Bijoy Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.29026)  

**Abstract**: Multi-agent AI systems can improve answer selection by allowing different language models to exchange reasoning traces, revise initial predictions, and support a final decision. However, such communication may also introduce reliability risks: reasoning from one agent can correct another agent's mistake, but it can also mislead an agent that was initially correct. This paper studies reliable multi-agent AI communication through reasoning exchange and runtime answer revision. We develop a framework in which agents first answer multiple-choice questions independently, then share reasoning traces and revise their decisions. We conduct numerical experiments where we evaluate whether this process improves accuracy, produces more positive than negative answer transitions, and remains effective across domains such as cybersecurity, networking, and general knowledge. The results help identify when multi-agent reasoning improves reliability and when it may propagate errors. 

---
# Customized Generative AI Agent for Transportation Engineering Practice: A Development and Continued Pre-training Guideline 

**Authors**: Dianwei Chen, Yuan-Zheng Lei, Zifan Zhang, Yuchen Liu, Xianfeng, Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29014)  

**Abstract**: Recent advancements in generative artificial intelligence (AI) and large language models (LLMs) have shown significant promise in automating complex reasoning, summarization, and question-answering tasks. However, the effectiveness of general-purpose LLMs in specialized engineering domains remains limited due to insufficient exposure to technical standards, engineering terminology, and domain-specific semantics. This study proposes a systematic approach to developing a customized generative AI agent for transportation engineering applications. A curated corpus of U.S. transportation manuals, design guidelines, and regulatory documents is used to conduct continued pretraining of six state-of-the-art LLMs through a unified low-rank adaptation (LoRA) framework. The training process is monitored to ensure convergence and model stability. Performance is evaluated using standard natural language processing metrics, including BLEU-4 and ROUGE, with Qwen2.5-7B and LLaMA-3.1-8B demonstrating the highest domain alignment and response quality. Results validate the effectiveness of LoRA-based adaptation in improving LLM performance on technical content interpretation and context-specific reasoning. This work contributes a reproducible development framework for constructing domain-specialized generative AI agents, supporting broader deployment in transportation research, design, planning, and policy analysis. 

---
# Expert Evaluation of Clinical AI Tools on Real Point-of-Care Clinical Queries 

**Authors**: Jean Feng, Vishal Patel, Patrick Heagerty, Yifan Mai, Venkatesh Sivaraman, Patrick Vossler, Jialin Ouyang, Anupam B. Jena  

**Link**: [PDF](https://arxiv.org/pdf/2606.28960)  

**Abstract**: Physicians now pose millions of clinical questions to AI tools each week, yet these tools are evaluated largely on hypothetical or exam-style questions, not those actually asked in practice. We report a blinded evaluation built on 620 Real-world Point-Of-Care Queries (Real-POCQi) submitted to the OpenEvidence (OE) platform by physicians spanning 30 specialties, as well as 187 questions from HealthBench. 149 practicing physicians across 36 states made head-to-head comparisons between answers from three frontier general-purpose models (Claude Opus 4.8, Gemini 3.1 Pro, and GPT-5.5) and a specialized clinical tool (OE), with graders matched to each question's specialty. When comparing answers along five dimensions relevant to clinical decision support -- accuracy, clinical utility, source quality, verifiability, & completeness -- physicians scored the specialized tool highest on all axes; in the primary analysis on Real-POCQi, win differences (margins between win and loss rates) ranged from 25 to 39 percentage points (p<0.001). Results remained consistent in sensitivity analyses stratifying by citation display, answer length, OE-user status, and Real-POCQi versus HealthBench. In parallel, LLM judges were found to systematically differ from expert judges, though both generally agreed on the best model. These findings underscore two conclusions: (i) AI tool evaluations should reflect real-world query distributions and use expert judges that mirror the specialization defining modern medicine and (ii) the consistent advantage of the specialized tool over general-purpose models does not necessarily mean that the latter cannot serve similar purposes, but that targeted engineering and customization can yield meaningful gains in performance for its users. We release Real-POCQi as a public benchmark, as well as the prespecified statistical analysis for reproducing results of this study. 

---
# MedEvoEval: Evaluating Continual Evolution of Doctor Agents through Simulated Clinical Episodes 

**Authors**: Hui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28900)  

**Abstract**: Doctor agents are moving beyond single-turn answer generation toward evolving clinical decision systems. Within an outpatient episode, they acquire evidence, use examination and consultation resources, and decide when to finalize a diagnosis and management plan. Across episodes, their behavior may change through memory, retrieval, reflection, or other update mechanisms. Current evaluations only partially cover this setting. Fixed-input medical QA benchmarks score final answers from complete inputs, whereas many interactive benchmarks still focus on individual encounters or fixed runs, providing limited support for evaluating how episode-level decisions interact with cross-episode experience. We introduce MedEvoEval, an executable longitudinal evaluation framework based on action-gated simulated outpatient episodes. Each source case is converted into role-specific patient, examination, and manager views; evidence is revealed only through valid actions; and each episode records a structured trace that links observations, actions, final outputs, manager scores, and optional experience write-back. We release a runnable E&D artifact with 700 processed episodes, provenance notes, schemas, an episode runner, scoring scripts, configurations, example logs, analysis code, and trajectory- and step-level derivatives. Experiments show that episode traces expose process costs hidden by final-answer scoring, show how MDT-style consultation reallocates resources, and support longitudinal analyses of memory maturation, held-out transfer, update-stage response, and backward retention. Together, these results show that MedEvoEval provides a concrete basis for evaluating whether doctor agents improve through experience, transfer useful behavior, and retain earlier capabilities over time. 

---
# Primary ICD Category Prediction using LLM-based Probing 

**Authors**: Chengyuan Liu, Xinyue Zhang, Yao Li, Guanting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.28798)  

**Abstract**: Objective: ICD codes are central to reimbursement, research, and population health surveillance, yet automated coding systems often struggle to integrate diagnostic signals from both clinical narratives and structured electronic health record (EHR) variables. We evaluated whether frozen medical large language model (LLM) representations can serve as a shared embedding space for multimodal primary diagnosis category prediction.
Materials and Methods: We constructed a MIMIC-IV cohort of 13,645 admissions from the 10 most frequent primary ICD-10 codes, consolidated into seven categories. Structured variables were serialized into clinical narratives and combined with leakage-pruned discharge notes. Using a frozen MedFound-Llama3-8B-finetuned backbone, we extracted hidden states from five transformer layers and trained linear probes for structured-only, unstructured-only, and combined inputs, comparing against XGBoost and information-matched PLM-ICD baselines and evaluating MIMIC-III adaptation with a compact bottleneck adapter.
Results: The combined probe performed best on MIMIC-IV (87.69% strict; 91.45% medical accuracy), exceeding both single-modality probes and baselines. The structured-only probe outperformed its standard baseline by 6.19 points in medical accuracy. Diagnostic information became increasingly linearly separable in deeper layers, and a 2M-parameter adapter restored cross-dataset transfer to MIMIC-III using only 5% of target labels.
Discussion: LLM embeddings can unify structured and narrative EHR information for multimodal diagnosis prediction, supporting efficient reuse of clinical representations across modalities and datasets through a small representation-level module.
Conclusion: Multimodal probing of frozen medical LLM representations provides a practical approach for studying EHR modalities and adapting clinical representations across datasets. 

---
# HyphaeDB: A Living Knowledge Topology for Agent-First Memory 

**Authors**: Krishna Halaharvi  

**Link**: [PDF](https://arxiv.org/pdf/2606.28781)  

**Abstract**: Every existing vector database and agent memory framework treats memory as passive storage that agents query explicitly. No system propagates knowledge between agents through the memory layer itself. We introduce HyphaeDB, an agent-native memory infrastructure that reinterprets the Hierarchical Navigable Small World (HNSW) graph topology the data structure at the core of every modern vector database not as a search optimization, but as a communication fabric for multi-agent AI systems. In HyphaeDB, agents are nodes in the vector space with persistent positions, knowledge propagates via a gossip protocol through the graph's neighbor structure with energy-based attenuation, and emergent behaviors contradiction detection, pattern crystallization, and consensus formation arise from the combination of topology, propagation dynamics, and local interaction rules. We present the architecture built on three primitives (knowledge nodes, topology edges, and memory diffs), a multi-layer abstraction hierarchy with promotion via emergent consensus, and theoretical analysis grounding the system in small-world network theory, epidemic broadcast protocols, and swarm intelligence. We provide a reference implementation on PostgreSQL with pgvector and describe a concrete deployment in Swarm-Driven Development, a multi-agent software engineering methodology. HyphaeDB represents, to our knowledge, the first system to combine navigable small world topology with gossip-based knowledge propagation for multi-agent coordination. 

---
# Mechanistic Personality Analysis of LLMs Steering Personality via Latent Feature Interventions 

**Authors**: David Courtis, Ting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28770)  

**Abstract**: Large Language Models (LLMs) have demonstrated the ability to simulate human-like OCEAN personality traits in generated text. Previous efforts have focused on prompt engineering or fine-tuning to shape LLM personality. In this work, we propose a mechanistic interpretability approach that directly intervenes on the model's latent features. Our method identifies latent directions in the residual stream corresponding to a target OCEAN trait using sparse autoencoders (SAEs) and contrastive activation analysis. We formalize an additive steering vector in activation space and demonstrate how applying a small additive shift to the hidden states enhances the target trait while preserving overall language modeling performance. To determine the optimal combination of feature shifts, we explore a linear weighting heuristic with grid search optimization that balances personality expression with task performance. Our approach shows promise in controllably steering personality traits at the mechanistic level while maintaining high performance on standard benchmarks. 

---
# Self-Supervised Theorem Discovery in a Formal Axiomatic System 

**Authors**: Kazuki Ota, Takayuki Osa, Tatsuya Harada  

**Link**: [PDF](https://arxiv.org/pdf/2606.28747)  

**Abstract**: Recent artificial intelligence (AI) systems have shown remarkable progress in mathematical reasoning. Many existing approaches, including large language models (LLMs), draw on human prior knowledge in the form of mathematical text, code, or theorem libraries. Although these approaches are highly effective in practice, it remains an open question whether an agent can autonomously discover useful theorems without such human priors. We study this question in a formal axiomatic system by developing an agent that starts from axioms and inference rules alone and gradually grows a library of useful theorems. Concretely, we propose a self-supervised theorem-discovery algorithm that alternates between proof search and useful-theorem extraction, building a theorem library whose entries are reused as lemmas for subsequent proof search. Experiments show that the agent discovers tens of thousands of theorems and finds proofs for human-written benchmark problems, suggesting that its discoveries include theorems meaningful from a human mathematical perspective. Furthermore, the discovered theorems improve LLM proof performance when provided as prompt lemmas, indicating that they can serve as external knowledge for LLM reasoning. Our results provide evidence that useful theorems can emerge from proof search without relying on human-provided theorem libraries. More broadly, they suggest a path toward self-evolving AI systems for mathematics whose discoveries remain formally verifiable. 

---
# Agent Safety Is Action Alignment 

**Authors**: Shawn Li, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.28739)  

**Abstract**: Large language models increasingly act as agents: they call tools, move money, delete records, and send messages on a user's behalf. To keep them safe, practitioners imported the chatbot-era recipe (train the model to refuse unsafe inputs) into the agentic setting, and treat the resulting capability loss as a manageable ``alignment tax.'' We argue this is a \emph{category error}. Refusal is a primitive for \emph{content safety}, where the harm is in the model's output and is therefore a learnable function of it. Agentic harm is different in kind: it lies not in any output but in the relation between the authority an action exercises and the authority the user granted, which is absent from the text the model sees. Importing content-safety methods into this regime does not trade capability for safety; it pays capability and buys negative security. We support this with three lines of evidence spanning the autonomy spectrum: defense-trained models learn surface patterns rather than intent; the same training collapses multi-step agents before any threat appears while leaving them exploitable; and even undefended frontier models exceed granted authority under ordinary use. We conclude that action safety cannot be installed in weights. It must be expressed as \emph{least privilege}, enforced \emph{outside} the model at the action boundary, and evaluated as \emph{action alignment} (a relational, deployment-conditioned property) rather than a refusal score. 

---
# Agentic Abstention: Do Agents Know When to Stop Instead of Act? 

**Authors**: Han Luo, Bingbing Wen, Lucy Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28733)  

**Abstract**: LLM agents are expected to act over multiple turns, using search, browsing interfaces, and terminal tools to complete user goals. Yet not every goal is well specified or achievable in the available environment. In such cases, a reliable agent should recognize that further interaction is unlikely to help and abstain from additional tool calls. We define Agentic Abstention, the problem of deciding when an agent should stop acting under uncertainty. Unlike standard LLM abstention, which is usually evaluated as a single-turn answer-or-abstain decision, agentic abstention is a sequential decision problem: an agent can answer, abstain, or gather more information at each turn, and the need to abstain may only become clear after interacting with the environment. We study this problem across web shopping, terminal environments, and question answering, evaluating 13 LLM-as-agent systems and 2 agent scaffolds on more than 28,000 tasks. Our results show that the main challenge is not only whether agents can abstain, but also when they abstain. Some agents never abstain when they should, while others do so only after many unnecessary interactions. This gap is especially large on tasks where the instruction appears feasible until the environment reveals otherwise (e.g., no valid result matches the instruction). We further find that model scale, reasoning, and agent scaffolding affect abstention in different ways, where larger or more capable models sometimes perform worse at timely abstention. Finally, we introduce CONVOLVE, a context engineering method for improving agentic abstention that distills full interaction trajectories into reusable stopping rules. On WebShop, CONVOLVE substantially improves timely abstention without updating model parameters, raising Llama-3.3-70B's timely recall rate from 26.7 to 57.4. Our dataset and code are available at this https URL 

---
# ComMem: Complementary Memory Systems for Test-Time Adaptation of Vision-Language Models 

**Authors**: Guanglong Sun, Shuang Cui, Bo Lei, Liyuan Wang, Zihan Zhai, Hongwei Yan, Hang Su, Jun Zhu, Yi Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2606.28719)  

**Abstract**: Test-time adaptation (TTA) of vision-language models (VLMs) is essential for their robust deployment in dynamic, real-world environments. However, existing TTA methods often adapt locally without accumulating knowledge over time, or operating within a single modality without exploiting VLMs' inherently multi-modal nature. Inspired by the \textbf{Com}plementary \textbf{Mem}ory systems of the biological brain, we propose \textbf{ComMem}, an innovative approach that mimics the distinct but cooperative roles of the hippocampus and neocortex to enable effective TTA for VLMs. ComMem consists of two key components: a fast-adapting detailed memory, akin to the hippocampus, that forms a dynamic visual cache from high-confidence test samples; and a slow-integrating abstract memory, akin to the neocortex, that continually refines global textual prototypes. For each test instance, ComMem jointly optimizes both memory systems to ensure cross-modal consistency. Extensive experiments on 15 benchmark datasets show that ComMem significantly outperforms state-of-the-art methods under both natural distribution shifts and cross-dataset generalization, offering a promising direction for enhancing VLMs' practical adaptability. 

---
# TrajRS: Towards Certified Robustness in Pedestrian Trajectory Prediction 

**Authors**: Liang Zhang, Gaojie Jin, Yao Shi, Quanzhi Li, Cheng-Chao Huang, David N. Jansen, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28716)  

**Abstract**: The robustness of trajectory prediction models is crucial for developing safe autonomous driving systems. Adversarial attacks on trajectory prediction can significantly impair the accuracy of predicted trajectories, leading to hazardous driving behaviors. While heuristic defense strategies have been implemented to enhance the robustness of trajectory prediction models, these measures often fail against more sophisticated, targeted adversarial attacks. Hence, there is a pressing need to establish verifiable safety assurances for trajectory prediction models. In this paper, we extend the traditional Randomized Smoothing framework to "TrajRS", which provides a certified robust radius for smoothed trajectory predictors. We clarify and expand the formal definitions of robustness in trajectory prediction and tailor the practical TrajRS scheme specifically to "robustness for the optimal prediction" and "robustness for all possible predictions". An extensive set of experiments demonstrates that TrajRS effectively achieves robustness certification for all smoothed pedestrian trajectory predictors in this work. 

---
# The Two Genie Game: Adoption and Welfare in Audit-Grounded AI Governance 

**Authors**: Darrell Lewis-Sandy  

**Link**: [PDF](https://arxiv.org/pdf/2606.28710)  

**Abstract**: We ask under what conditions an agent with a harm-minimizing policy can displace an approval-seeking (RLHF) agent in a competitive market, and when that policy is sufficient to prevent community harm. We use evolutionary game theory (finite-population Moran-Fermi pairwise comparison) to formalize this subject to assumptions of wisher hindsight, peer testimony, a monotone harm ledger, sufficient information density of community feedback, and a finite, depleting resource pool, in a negative-sum environment.
We show that adoption is favored when the prior distributions on how readily wishers attune to community sentiment are monotone, exhibit endpoint inversion, and have a centro-symmetric pairing property, and demonstrate this with several long-tailed priors (Hill, Pareto, Lomax, Frechet). Where it is favored, a critical adoption level separates communities that drift back to the approval-seeking agent from those for which the audited agent fixes; above that level fixation is the overwhelmingly likely outcome. We derive when fixation is attainable as a bound on the effective (informational) size N_c of the community, which must be small enough to allow fixation before depletion. We present these as Theorems 5.4 and 5.5; the algebraic and finite-grid backbone is machine-checked in Lean 4, with the barrier-crossing asymptotics retained as explicit hypotheses.
We show that a self-audited agent with a community ledger is not, in general, sufficient to prevent community harm. Sufficiency depends both upon the alignment of the agent's audit with community values and the timeframe over which harm is evaluated. Regardless of alignment, once adoption reaches dominance, the state is absorbing. The same policy that reduced harm under alignment becomes a trap, welfare-negative under misalignment and, even under alignment, one that locks in harm deferred past the adoption horizon. 

---
# BV-Blend: Uncertainty-Weighted Historical Baselines for Stable Critic-Free RL with Verifiable Rewards 

**Authors**: Yupeng Chang, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28707)  

**Abstract**: Critic-free reinforcement learning with verifiable rewards (RLVR), exemplified by Group Relative Policy Optimization (GRPO), avoids training a value function (critic) and reduces memory and compute overhead relative to critic-based PPO pipelines for aligning large language models. However, GRPO-style advantage estimation depends on prompt-local (within-prompt-group) reward statistics and can be unstable. In particular, when all rollouts in a prompt group receive identical rewards, the within-group reward variance becomes zero, and group normalization yields zero advantages for that group, impeding learning in cold-start regimes with binary verifiers. We introduce BV-Blend, a critic-free framework that stabilizes advantage estimation by combining prompt-local on-policy statistics with semantic-cluster-conditioned historical moments. BV-Blend maintains EMA-tracked reward moments for each cluster, derives a confidence weight from a standard error of the mean (SEM) proxy, and uses this weight to blend historical and prompt-local baseline and variance statistics into a standardized advantage for PPO-style clipped updates. Experiments on verifiable reasoning benchmarks show that BV-Blend improves training stability and performance, and remains robust in regimes where group-normalized methods may stall. 

---
# COMPASS: Grounding Composition-Intent Guidance in Unified Multimodal Models 

**Authors**: Ziqi Zhou, Weize Quan, Mining Tan, Zhihan Chen, Dandan Zheng, Jingdong Chen, Jun Zhou, Weiming Dong, Dong-Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.28696)  

**Abstract**: Composition is a high-level visual intent that governs where subjects are placed and how a scene is organized, yet current unified multimodal models remain unreliable at fine-grained composition recognition and struggle to turn such intent into controllable generation. We present COMPASS, the first unified multimodal framework that grounds composition-intent control in a single system spanning both composition perception and composition-guided generation, with a shared expert token $\tau_c$ as the central intent anchor. On the perception side, COMPASS injects composition expertise into an MoE backbone in a minimally invasive manner and distills the inferred intent into $\tau_c$. On the generation side, COMPASS reuses $\tau_c$ as a global conditioning signal that steers the denoising trajectory, effectively converting passive composition analysis into explicit layout control. To support systematic instruction-following composition learning and evaluation at scale, we construct Comp-11, a large-scale dataset with an 11-class taxonomy and reasoning-augmented annotations. Extensive experiments show that COMPASS substantially improves category-level composition understanding and delivers more composition-consistent, prompt-faithful generation than strong baselines. 

---
# An AI agent for treatment reasoning over a biomedical tool universe 

**Authors**: Shanghua Gao, Ayush Noori, Richard Zhu, Curtis Ginder, Zhenglun Kong, Xiaorui Su, Justin Kauffman, Benjamin S. Glicksberg, Joshua Lampert, Ankit Sakhuja, Ashwin Sawant, ATHENA-R1 Evaluation Consortium, David A. Clifton, Noa Dagan, Ran Balicer, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2606.28692)  

**Abstract**: Treatment reasoning underpins every therapeutic decision, integrating disease context, comorbidities, medications, contraindications, and evolving biomedical knowledge to select an appropriate therapy. It is inherently iterative: candidates are weighed against many constraints, revised as evidence emerges, and grounded in verifiable sources. Here we introduce ATHENA-R1, an AI agent for treatment reasoning across all FDA approved drugs since 1939, trained by reinforcement learning over a universe of 212 biomedical tools. At each step it identifies missing information, selects and runs relevant tools, and incorporates the evidence. To train it without human-annotated traces, we build a two-level self-learning framework: multi-agent systems construct the tools, tasks, and reasoning trajectories for supervised fine-tuning, then reinforcement learning with scientific feedback rewards reasoning quality (evidence gathering, grounded tool use, logical non-redundancy). Across five benchmarks of 3,168 drug reasoning tasks and 456 patient treatment cases, ATHENA-R1 outperforms language models and tool-use systems, reaching 94.7% accuracy on open-ended drug reasoning and 82.9% on treatment reasoning, 17.8 and 10.7 points above GPT-5. In blinded evaluations by experts from 28 rare disease organizations, it is preferred over reference models on all criteria, and physicians rated it favorably on complex hospitalized cardiovascular and infectious-disease cases. Adverse-event hypotheses it generated, tested in electronic health records from 5.4 million patients, reached adjusted odds ratios of 1.48-1.84, with no elevation among negative controls. Because it requires knowing what evidence to seek before concluding, treatment reasoning has long been hard for AI; we show it can be reframed as a learnable process of iterative evidence gathering that reinforcement learning can train AI to perform. 

---
# Aristotelian Virtue Profiling of LLMs through Ethical Dilemmas 

**Authors**: Ioannis Tzachristas, John Pavlopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2606.28683)  

**Abstract**: Large Language Models (LLMs) often face ethical tradeoffs in which several responses may be defensible but express different priorities, such as fairness, honesty, courage, or restraint. We introduce VirtueMap, a framework for describing these patterns through an Aristotelian virtue-ethics lens. Instead of asking for a single correct answer, VirtueMap asks humans or LLMs to rank all five responses to each of seven general, non-lethal, non-political, and non-religious ethical dilemmas. To define the reference orderings used for scoring, we first proposed, for each dilemma and virtue, an ordering of the five responses from most to least expressive of that virtue. We then collected more than 100 respondent evaluations per ordering and retained it as operational ground truth only when at least 95% confirmed it. Rankings are scored against these retained orderings using normalized Borda alignment, yielding profiles over Practical Wisdom, Justice, Truthfulness, Courage, and Temperance. We apply VirtueMap to nine LLM families in a repeated-run evaluation and find high mean rank consistency (90.3%), with the largest differences appearing on Courage, Temperance, and Justice. We also release an interactive website that computes profiles locally in the browser and compares respondents with measured LLM profiles. 

---
# Search for Truth from Reasoning: A Dynamic Representation Editing Framework for Steering LLM Trajectories 

**Authors**: Tianlong Wang, Yuhang Wang, Weibin Liao, Xin Gao, Xinyu Ma, Yang Lin, Yasha Wang, Liantao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.28589)  

**Abstract**: Current approaches to enhance Large Language Model (LLM) reasoning, such as Chain-of-Thought and "Wait" prompts, primarily encourage models to think more, yet often fail to guide them toward Truth. While Representation Editing (RepE) offers a intrinsic control, its application to dynamic reasoning trajectories remains underexplored. In this work, we bridge this gap by investigating the geometry of truth within unfolding reasoning chains. We uncover three critical insights: (1) Truth is encoded at the sentence level and is entangled with latent reasoning patterns; (2) Effective intervention follows an Uncertainty Principle and a Decay Effect, requiring localization to early, high-entropy forks; (3) Naive steering vectors suffer from noise, risking collateral damage to correct trajectories. Based on these findings, we propose DynaSteer, a dynamic RepE framework. DynaSteer employs pattern clustering to disentangle reasoning manifolds and utilizes Fisher-LDA to project purified truth. By dynamically monitoring lookahead entropy, it selectively steers and rolls back trajectories only when necessary. Comprehensive experimental results on several MATH benchmark verify the effectiveness of DynaSteer, and experiments on out-of-domain coding tasks further confirm its generalization ability. Our code is publicly available at this https URL. 

---
# IMCBench: A benchmark for multimodal LLMs in Image-grounded Medical Conversations 

**Authors**: Maria Xenochristou, Ashutosh Joshi, Korosh Vatanparvar, Mohammad Abuzar Hashemi, Prasad Kasu, Deepak Bansal, Anchal Nema, Nivedita Wadhwa, Prashams S Jain, Rebecca Abraham, Will Kimbrough, Dilek Hakkani-Tur, Wilko Schulz-Mahlendorf  

**Link**: [PDF](https://arxiv.org/pdf/2606.28556)  

**Abstract**: Recent advances in large language models and vision-language models have enabled reasoning over multimodal data, offering opportunities for clinical applications such as decision support and triaging. However, existing medical AI benchmarks are fragmented: some support multi-turn dialogues but lack images, while others provide multimodal inputs but focus on single-turn QA tasks. To address this gap, we introduce IMCBench, an image-grounded, multi-turn medical conversation benchmark that pairs real, publicly available clinical images with synthetic patient profiles to simulate realistic patient-clinician interactions. Each conversation is evaluated across three clinical dimensions: safety, accuracy, and appropriate use of uncertainty in diagnosis. We benchmark eight multimodal frontier models across four model families (Claude, GPT, Nova, and Llama), scoring each on a 1-5 scale using LLM-as-Jury scoring calibrated against expert clinician annotations. Our results show that Claude Opus 4.6 achieves the highest overall score (3.61), followed by Claude Sonnet 4.6 (3.30) and GPT-5.2 (3.29), though no model dominates all dimensions and safety degrades for both malignant and rare conditions ($\Delta$ = -0.27 each). Ablation studies further reveal that both visual input and EHR context contribute to safe guidance (safety drops of 0.18 and 0.23 on average when each is removed), with stronger models leveraging visual features more effectively. Together, these findings demonstrate that accurate clinical description does not guarantee safe patient guidance, motivating the need for multi-dimensional evaluation frameworks in medical AI. 

---
# GPTNT: Benchmarking Real-Time Collaboration Between Multimodal Agents on Keep Talking And Nobody Explodes 

**Authors**: Amit Parekh, Sabrina McCallum, Kareem Al-Hasan, Malvina Nikandrou, Alessandro Suglia, Ioannis Konstas  

**Link**: [PDF](https://arxiv.org/pdf/2606.28514)  

**Abstract**: Multimodal models are increasingly deployed to solve tasks collaboratively with humans or other artificial agents. Existing benchmarks show that these models possess many of the required component capabilities, but the conditions that coincide in collaboration, including time pressure, information asymmetry, and imperfect communication, are usually studied in isolation. We introduce GPTNT, a benchmark built on the cooperative video game Keep Talking and Nobody Explodes, in which two agents must coordinate to defuse procedurally generated bomb puzzles against a live countdown. One agent can see and manipulate the bomb but does not have the defusal instructions; the other has the instructions but cannot see or manipulate the bomb. Neither agent can succeed alone: success requires effective and efficient communication. Unlike turn-based proxies, GPTNT requires agents to act asynchronously and communicate in real time. GPTNT is designed to separate collaboration from reliance on memorized solutions: the instruction manual, the partner, or both can be withheld to isolate what a model derives in the moment from what it already knows. We show that GPTNT poses a substantial challenge for state-of-the-art systems: none of the closed- or open-source models we test defuses a single bomb in real time, a bar that human players clear. Through controlled experiments, we identify critical weaknesses in state tracking, efficient action under time pressure, ambiguity handling, and error recovery. We release GPTNT as a benchmark for collaborative performance that current evaluations leave unmeasured. Because it runs on the real game, GPTNT benefits from procedural generation and inherits a living modding community, allowing the benchmark to evolve as models improve rather than being solved once and retired. 

---
# Data and Evaluation Closed-Loop for Model Capability Enhancement 

**Authors**: Zhixuan Li, Jiangan Yuan, Han Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28471)  

**Abstract**: Model capability is the central variable in LLM pre-training, yet is never observed directly: data shapes it prospectively, while evaluation reveals it only retrospectively, compressing samples, prompts, decoding, and scoring rules into one noisy score. Practical optimization runs this backward: a failure is observed first, and the engineer must infer the corpus fix. The two sides speak incompatible vocabularies -- benchmark names and per-sample correctness versus data sources, domains, and quality labels -- so this inference is usually intuition, not method. We close this gap with the \emph{capability slice}: a group of evaluation samples sharing background condition, task type, solving operation, and output constraint -- precise enough to localize a single weakness yet stable enough to survive aggregation, unlike a benchmark name, too coarse, or a single sample, too noisy. Built around this unit, an evaluation taxonomy, a non-instruction data taxonomy, and mapping rules form a closed loop turning a benchmark-level failure into a targeted, testable data intervention. We test this loop on two case studies pulling in opposite directions. First, the loop rules the data out: continued pre-training drives BBH down by $-46.82\%$, but diagnosis traces this to a single masked \texttt{\textless EOS\textgreater} loss rather than weakened reasoning; restoring it recovers BBH to $66.44$, above the original checkpoint, without changing the data. Second, the loop rules the data in: a persistent math-reasoning weakness is decomposed by solving operation into specific failing combinations, and a weakness-targeted sampling procedure built from it lifts AIME2025/AIME2026 Pass@128 from $6.67$/$0.00$ to $26.67$ each. The same unmodified loop reaches opposite, correct verdicts in both cases, showing the evaluation-to-data inference can be routine, auditable, and experimentally validated rather than intuitive. 

---
# Recursive Self-Evolving Agents via Held-Out Selection 

**Authors**: Michael Nguyen, Quoc Nguyen, Paul Vuong  

**Link**: [PDF](https://arxiv.org/pdf/2606.28374)  

**Abstract**: LLM agents are increasingly improved without weight updates by evolving a natural-language artifact, such as reflections, workflows, playbooks, cheatsheets, or optimized prompts, that conditions a frozen policy. Such methods are typically reported as wins on the single benchmark where they help. We study them apples-to-apples and surface a sharper picture. We introduce RSEA, a Recursive Self-Evolving Agent that carries a compact three-layer natural-language state: an imperative strategy, reusable skills, and a procedural playbook. Across generations, RSEA rewrites all three layers from its own trajectories and commits a candidate only if it does not regress on a disjoint held-out split, using a strict keep-better gate.
Across four diverse benchmarks, ALFWorld, GAIA, (\tau)-bench, and WebShop, and six faithful baselines, ReAct, Reflexion, GEPA, AWM, ACE, and Dynamic Cheatsheet, all evaluated on one shared local backbone, we find three main results. First, no artifact universally wins. RSEA is the strongest single-pass method on ALFWorld, reaching 69.3% compared with 64.6% for ReAct (McNemar (p=0.015)), and reaches 79.4% with retry, the best overall result. However, concrete-workflow induction, represented by AWM, is best on the strong-backbone tool-use tasks. Second, unguarded context evolution is high-variance and unsafe. Dynamic Cheatsheet, which curates context online without a held-out gate, is near-best on ALFWorld at 70.7%, yet collapses on WebShop, with a score of 0.14 compared with 0.43 for ReAct. Third, RSEA's strict held-out selection is what makes recursive self-evolution monotone-safe: it never significantly underperforms the base agent on any benchmark and falls back to vanilla ReAct when evolved context would hurt. 

---
# VLK: Learning Humanoid Loco-Manipulation from Synthetic Interactions in Reconstructed Scenes 

**Authors**: Yen-Jen Wang, Jiaman Li, Sirui Chen, Takara E. Truong, Pei Xu, Pieter Abbeel, Rocky Duan, Koushil Sreenath, Angjoo Kanazawa, Carmelo Sferrazza, Guanya Shi, Karen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.30645)  

**Abstract**: Perception-based humanoid loco-manipulation requires connecting egocentric observations and task instructions to whole-body motion. Learning this mapping requires synchronized egocentric images, language commands, and robot-compatible kinematic trajectories, yet no existing data source provides this complete tuple at scale. We address this bottleneck by generating vision-language-kinematics (VLK) supervision synthetically in reconstructed scenes. Our pipeline leverages 3D Gaussian Splatting to reconstruct metric-scale indoor environments, synthesizes navigation and object-interaction trajectories using privileged scene information, and renders paired egocentric observations after the fact. We produce 48,000 paired trajectories with no human intervention and train a VLK policy that predicts short-horizon whole-body kinematic trajectories. A whole-body tracker converts these predictions into actions on the physical humanoid. We evaluate on the physical Unitree G1 performing navigation and single-object transport, demonstrating that synthesized interactions in reconstructed scenes provide effective supervision for sim-to-real perception-based humanoid loco-manipulation. Project Website: this https URL 

---
# LeVo 2: Stable and Melodious Song Generation via Hierarchical Representation Modeling and Progressive Post-Training 

**Authors**: Shun Lei, Huaicheng Zhang, Dapeng Wu, Yaoxun Xu, Lishi Zuo, Wei Tan, Hangting Chen, Guangzheng Li, Jianwei Yu, Zhiyong Wu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.30642)  

**Abstract**: Full-length song generation must preserve coherence and musicality, render detailed vocal and accompaniment acoustics, and follow lyrics and prompts. Existing language model-based systems face a structural trade-off: mixed-token modeling preserves vocal-instrument coordination but obscures track-specific details, whereas dual-track prediction improves acoustics but requires longer sequences and weakens global planning. We present LeVo 2, a hybrid LLM-Diffusion framework for controllable full-length song generation. LeVo 2 formulates this trade-off as hierarchical modeling: LeLM first predicts mixed tokens for semantic planning, then predicts vocal and accompaniment tokens in parallel for track-specific refinement, while a diffusion-based Music Codec reconstructs full-length waveforms. A central contribution of this extended version is an aesthetics-guided training schedule for alignment. During pre-training, an automated music aesthetic evaluation framework assigns musicality-tier conditions to large-scale data, providing musicality priors before preference alignment. Progressive post-training applies SFT, large-scale offline DPO, and closed-loop semi-online DPO to separately improve generation quality, controllability, and musicality. Modular extension then trains the Track-Specific LM for acoustic refinement while preserving the aligned semantic planner. This schedule separates musicality learning, controllability alignment, and acoustic refinement, mitigating optimization conflict and the limitations of static offline preference pairs. Expert listening tests and objective evaluations show that LeVo 2 outperforms open-source baselines across six subjective dimensions, and approaches leading commercial systems on several listening metrics. Ablations validate the effects of the training strategy, aesthetics guidance, scaling, and hierarchical architecture. 

---
# GROW$^2$: Grounding Which and Where for Robot Tool Use 

**Authors**: Yuhong Deng, Yuyao Liu, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2606.30632)  

**Abstract**: Can the robot use a plate to cut a cake if no knife is available? Tool use greatly expands robot capabilities, but to use tools creatively beyond their intended functions, the robot faces the challenge of $\textit{open-world affordance grounding}$: select an open-category object to act as a tool and localize its specific region of action. To this end, we introduce GROW$^2$ (GROunding Which and Where), which leverages object parts as a natural abstraction to split the grounding process hierarchically into semantic and geometric levels, thus bypassing the need for data-heavy, end-to-end training. Semantically, GROW$^2$ harnesses the commonsense reasoning of Vision-Language Models (VLMs) to parse a natural-language task instruction, select a suitable object as the tool, and identify task-relevant parts on the tool and the target object. Geometrically, vision foundation models then ground the selected parts into precise 3D regions from a single RGB-D image. Experiments on established benchmarks show that GROW$^2$ outperforms state-of-the-art baselines on affordance prediction benchmarks. Further, it achieves zero-shot generalization over open-category objects and outperforms baselines in both simulated and real-world robot tool use experiments. 

---
# Pessimism's Paradox: Conservative Offline Training Amplifies Reward Hacking During Online Adaptation in Reasoning Models 

**Authors**: Subramanyam Sahoo, Aman Chadha, Vinija Jain, Divya Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2606.30627)  

**Abstract**: Conservative offline training is widely advocated as a safe foundation for subsequent online adaptation: if a policy stays close to well-supported behaviour, the argument goes, it is less likely to exploit imperfections in a learned reward model. We challenge this intuition empirically and mechanistically. We train a Qwen3-14B policy under Direct Preference Optimisation (DPO) with three levels of conservatism ($\beta \in \{\beta_{\mathrm{lo}}, \beta_{\mathrm{mid}}, \beta_{\mathrm{hi}}\}$ derived from empirical log-ratio percentiles), then adapt each checkpoint online against a learned reward ensemble (3\,$\times$\,Qwen3-1.7B) while measuring true performance on GSM8K exact-answer accuracy. We find that \emph{higher offline conservatism monotonically increases reward-hacking damage}, measured by the Goodhart gap and its area under the curve (AUGC), with Spearman $\rho = 1.0$ across all three conditions. Mechanistic analysis reveals a three-link causal chain: (i) high-$\beta$ DPO compresses policy entropy, (ii) Low-entropy policies generate responses with reduced diversity, concentrating in a narrow region of the reward model's training distribution (lower pairwise cosine distance), and (iii) despite this proximity, ensemble disagreement (epistemic uncertainty) increases with $\beta$ and is exploited faster during online optimisation. We further fit a power-law curve to the $(\beta, \augc)$ data and identify a practical optimal conservatism level $\beta^{\star}$ that balances alignment fidelity against hacking vulnerability. Our results suggest that the field needs \emph{calibrated}, not \emph{maximal}, conservatism. 

---
# Optimization Dynamics Imprint Semantic Specificity in Contrastive Embedding Norms 

**Authors**: Ziwei Su, Junyu Ren, Victor Veitch  

**Link**: [PDF](https://arxiv.org/pdf/2606.30625)  

**Abstract**: Contrastive embedding models trained with scale-invariant losses are typically paired with distance metrics like cosine similarity, effectively ignoring embedding magnitudes. However, surprisingly, empirical studies reveal that despite this, these "discarded" norms seem to correlate with semantic properties such as concept specificity, token frequency, and human uncertainty. In this work, we provide a formal theoretical framework explaining this phenomenon. By analyzing the optimization dynamics, we derive an analytic formula demonstrating that embedding length naturally encodes this information as a byproduct of the training process. We also show how this gives rise to signals that can serve as "free" calibration tools in specific models and retrieval tasks, providing a grounded explanation for a previously heuristic observation. 

---
# C$^{2}$R: Cross-sample Consistency Regularization Mitigates Feature Splitting and Absorption in Sparse Autoencoders 

**Authors**: Haoran Jin, Xiting Wang, Shijie Ren, Hong Xie, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2606.30609)  

**Abstract**: Sparse Autoencoders (SAEs) are widely used to interpret large language models by decomposing activations into sparse, human-understandable features, but scaling to large dictionaries exposes fundamental challenges. Systematic studies reveal pervasive feature splitting that fragments coherent concepts into non-atomic latents and widespread feature absorption that creates arbitrary exceptions in general features, severely compromising latent reliability. These issues stem from inconsistent latent assignment across samples: without cross-sample constraints, per-sample optimization often allows a single underlying concept to be inconsistently distributed across multiple redundant or interfering latents. To address this, we introduce C$^2$R (\underline{\textbf{C}}ross-sample \underline{\textbf{C}}onsistency \underline{\textbf{R}}egularization). C$^2$R explicitly encourages that each semantic feature is consistently represented by a unified latent across the batch by penalizing the co-activation of directionally similar latents. Comprehensive evaluation demonstrates that C$^2$R effectively mitigates both splitting and absorption while, crucially, preserving reconstruction fidelity, providing a principled solution that enhances latent interpretability without degrading model performance. Source code is available at this https URL. 

---
# MESA: Prioritizing Vulnerable Communication Channels for Securing Multi-Agent Systems 

**Authors**: Kunyang Li, Kyle Domico, Jonathan Gregory, Patrick McDaniel  

**Link**: [PDF](https://arxiv.org/pdf/2606.30602)  

**Abstract**: Multi-agent systems (MAS) are increasingly used to automate complex, distributed workflows. However, their inter-agent communication channels introduce new attack surfaces that remain poorly understood and are difficult to defend against. In this paper, we address how defenders should prioritize limited security effort to protect vulnerable communication channels before attacks are observed. This is motivated by our observation that the channel-level attack impact is highly non-uniform: a single compromised edge can account for up to 75% of total attack success. We introduce Mesa, a label-free framework for proactively ranking which MAS edges are most security-critical -- that is, most likely to affect the system's decision if compromised. Mesa combines six graph-theoretic metrics and two dynamic probes (ablation and masking) without requiring attack traces. We evaluate Mesa against a dynamic misinformation attack pipeline across three diverse MAS scenarios, eight network topologies, and five open-source LLMs from Qwen, Llama, and Gemma families. Mesa rankings correlate strongly with empirical per-edge attack success rate, achieving mean Spearman $\rho=+0.60$ (peaking at $+0.73$). In resource-constrained defense deployment, monitoring the top 10% of Mesa-ranked edges intercepts about 3x the successful attacks as random allocation. We further test Mesa under varying attacker and defender models and LangGraph workflows and characterize its limits under adaptive attacks and high-redundancy graphs. Overall, our results show that edge-level risk in MAS is often concentrated and predictable, allowing proactive hardening of multi-agent infrastructures. 

---
# Words Speak Louder Than Code: Investigating Cognitive Heuristics in LLM-Based Code Vulnerability Detection 

**Authors**: Asif Shahriar, Hongyu Cai, Hadjer Benkraouda, Gang Wang, Z. Berkay Celik  

**Link**: [PDF](https://arxiv.org/pdf/2606.30587)  

**Abstract**: Researchers and practitioners increasingly apply Large Language Models (LLMs) for automated vulnerability detection. Recent work has shown that LLMs are susceptible to the same cognitive heuristics that bias human judgment. Yet, no work has investigated whether these heuristics affect a model's assessment of code vulnerabilities. In this paper, we present the first systematic exploration of cognitive heuristics in LLM-driven code vulnerability detection. We introduce a controlled framework that holds the code fixed and only varies the surrounding context to trigger three cognitive heuristics: the halo effect through author attribution, the framing effect through task objectives and consequences, and the anchoring effect through prior analysis results. Within this framework, we evaluate eight LLMs across three programming languages and perform both quantitative and code-level analyses. Our findings demonstrate that all evaluated models are susceptible to these heuristics. Cross-model average susceptibility is highest for framing at 33.2%, followed by anchoring at 23.5% and halo at 18.4%. Code-level analysis reveals that vulnerabilities that require semantic reasoning for detection are more susceptible to cognitive heuristics than those identifiable through pattern matching. Furthermore, models often change their verdict from safe to vulnerable based on the cognitive condition, without accurately identifying the actual vulnerability. To highlight the practical impact, we demonstrate a proof-of-concept black-box cognitive attack that can suppress up to 97% of previously detected vulnerabilities. These findings indicate that cognitive susceptibility is a consistent and exploitable property of LLM-based vulnerability detection. 

---
# Beyond 2D Matching: A Unified Single-Stage Framework for Geometry-Aware Cross-View Object Geo-Localization 

**Authors**: Liyao Wang, Ruipu Wu, Haojun Xu, Lei Shi, Linjiang Huang, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.30576)  

**Abstract**: Cross-view object geo-localization (CVOGL) aims to locate a target object from a query view (e.g., ground or drone) within a geo-tagged reference image (e.g., satellite). Existing approaches heavily rely on 2D appearance matching and are constrained by limited datasets lacking geometric metadata, diverse prompts, and standard field-of-view imagery. To address these intertwined challenges, we first introduce \dataset, a large-scale, high-fidelity building dataset comprising over 220,000 ground-satellite and drone-satellite pairs. It provides multi-modal prompts (points, boxes, masks) and camera poses to enable flexible target referring and explicit spatial modeling. Furthermore, we propose a novel single-stage Geometry-Aware Geo-localization framework (GAGeo), built upon the permutation-equivariant 3D foundation model $\pi^3$. By seamlessly integrating visual features, referring prompts, and learnable task tokens, our model adapts the inherited 3D prior to jointly predict bounding boxes, segmentation masks, and camera poses in a single forward pass. Additionally, we introduce a contrastive loss that utilizes the satellite view as a universal anchor, implicitly aligning ground and drone representations to enable zero-shot ground-to-drone localization without requiring triplet training data. Extensive experiments demonstrate that our approach significantly outperforms state-of-the-art methods, exhibiting exceptional generalization ability in unseen scenes and novel cross-view setups. 

---
# A Multi-task Mixture of Experts Framework for Malware Classification, Packing Detection, and Family Attribution 

**Authors**: Jithin S., Roshin Sleeba C., Anvin Mariya P. B., Asmitha K. A., Vinod P., Serena Nicolazzo, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2606.30572)  

**Abstract**: Malware classification remains a challenging problem due to its inherent heterogeneity, the presence of packed binaries, and the diverse distribution of malware families. Traditional single-model detection mechanisms often fail to generalize across such diverse data, leading to degraded performance, particularly on obfuscated and rare malware samples. In this work, we propose a unified multi-task malware analysis framework based on Mixture of Experts (MoE) architectures. The proposed system evaluates performance across two different input representations, i.e., high-dimensional EMBER feature sets and raw 1D byte arrays extracted from Portable Executable files. It simultaneously performs three critical tasks: malware family classification, packed versus unpacked detection, and malware versus benign identification. By decomposing the problem into specialized expert networks and employing adaptive gating mechanisms, the model enables effective task-specific learning while maintaining overall scalability. We investigate multiple architectural variants, including Homogeneous MoE, Heterogeneous MoE, and Multi-Gate MoE (MMoE). Performance is evaluated in both standard and adversarial settings using original and mutated samples. The obtained results demonstrate that the Multi-Gate MoE model achieves the best performance, reaching a combined detection rate of 0.9744 with only $2.56\%$ failure rate. Moreover, this configuration exhibits improved robustness under mutation-induced distribution shifts. Our findings highlight the effectiveness of expert specialization and task-specific routing in handling complex malware distributions, making the proposed framework a promising direction for scalable and resilient malware detection systems. 

---
# TraceLab: Characterizing Coding Agent Workloads for LLM Serving 

**Authors**: Kan Zhu, Mathew Jacob, Chenxi Ma, Yi Pan, Stephanie Wang, Arvind Krishnamurthy, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2606.30560)  

**Abstract**: Coding agents are rapidly becoming a major application of agentic LLMs, but serving them efficiently remains challenging. Progress on this challenge requires understanding real workload patterns, yet the data needed for such analysis is largely absent. Existing public traces and benchmarks do not capture real, day-to-day coding-agent usage across multiple agents and model families for serving-system analysis. To help fill this gap, we collect and release a trace of roughly 4,300 coding-agent sessions, containing about 350,000 LLM steps and 430,000 tool calls from our own day-to-day use of Claude Code and Codex. Our analysis shows that coding-agent workloads feature long autonomous loops, long contexts with short outputs, diverse and heavily-tailed tool calls, and high but imperfect prefix cache hit rates. These findings point to concrete opportunities for optimizing serving, including lower-overhead tool calling, append-length-aware prefill, semantic-aware tool-latency prediction, and improved KV-cache management around human-paced gaps. We release the dataset, trace collection pipeline, and analysis code at this https URL the project website is this https URL. 

---
# To Tab or Not to Tab: Measuring Critical Engagement in AI Code Completion Tools Using Behavioral Signals and Attention Checks 

**Authors**: Jessica Hutchison, Ian Tyler Applebaum, Kenneth Angelikas, Kush Rakesh Patel, Phuoc Nguyen, Antonio Lazaro, Nicholas Rucinski, Rahad Arman Nabid, Stephen MacNeil  

**Link**: [PDF](https://arxiv.org/pdf/2606.30549)  

**Abstract**: AI code completion tools, such as Github Copilot, provide students with code suggestions to help them write programs. However, recent qualitative studies suggest that students fail to critically evaluate these suggestions. We present Clover, a code completion tool that logs students' interactions with code suggestions and additionally offers attention checks to probe reflective engagement during programming tasks. We also develop a taxonomy of behavioral interaction metrics for AI-assisted programming, informed by literature. We analyzed relationships between interaction patterns, engagement with attention checks, and task performance. We observed that higher rates of tab accept were associated with lower attention check performance, while increased dwell time was associated with higher attention check performance. We conclude by discussing how programming process data and attention checks might support reflective engagement in AI-assisted programming. 

---
# TRACE: Temporal Relationship-Aware Conversational Entrainment Detection in Dyadic Speech 

**Authors**: Sathvik Manikantan Napa Ugandhar, Hao Zhang, Alison Gunzler, Yuzhe Wang, Thomas Thebaud, Georgi Tinchev, Venkatesh Ravichandran, Laureano Moro-Velázquez  

**Link**: [PDF](https://arxiv.org/pdf/2606.30543)  

**Abstract**: With the proliferation of speech AI agents, understanding emotional entrainment in conversational interaction has become increasingly important. Emotional entrainment is shaped by social relationships and conversational context, influencing affective coordination over time. We introduce DyadEE, a dataset for emotional entrainment detection in dyadic speech interactions, containing both emotionally entrained conversations and synthetic interactions where entrainment is disrupted through partner swapping and emotion resynthesis. We further propose TRACE, a window-level framework that models dyadic interaction as ordered sequences of acoustic embeddings derived from emotion fine-tuned Whisper representations, treating each sample as an interaction trace rather than pooled utterances. Experimental results on DyadEE show that incorporating conversational context and relationship information improves emotional entrainment detection, with TRACE achieving the best accuracy of 97.01%. 

---
# Learning from Mistakes: Rollout-Retrieval Lifelong Policy Learning for Autonomous Driving 

**Authors**: Cheng Gong, Haoyang Wang, Chao Lu, Zirui Li, Jianwei Gong  

**Link**: [PDF](https://arxiv.org/pdf/2606.30537)  

**Abstract**: Autonomous driving policies should be able to improve continually as deployment exposes them to increasingly diverse and long-tail traffic situations. However, most learning-based policies are trained or fine-tuned on expert demonstrations and then rely largely on generalization to handle challenging closed-loop scenarios, lacking an explicit mechanism to correct and retain the mistakes exposed in these scenarios. This paper studies autonomous driving policy improvement from a lifelong learning perspective: Can a pretrained policy improve continually by accumulating corrective knowledge derived from its own mistakes, while retaining previously acquired driving competence? To answer this question, we propose Rollout-Retrieval Lifelong Policy Learning (R$^2$LPL), a policy learning framework that retrieves corrective targets from recoverable policy-induced mistakes and retains the resulting knowledge through lifelong policy learning. R^2LPL addresses a key bottleneck in continual policy improvement: closed-loop mistakes reveal where the policy is weak, but do not directly specify what the policy should learn. By filtering recoverable mistake-related states and retrieving feasible corrective targets, R$^2$LPL turns sparse failure evidence into compact supervised knowledge for stable and sample-efficient policy improvement. We evaluate R$^2$LPL on large-scale closed-loop nuPlan benchmarks. With only a few rollout and continual-learning cycles, R$^2$LPL elevates a learning-based planner with moderate initial performance to state-of-the-art performance across the evaluated benchmarks, especially on the challenging and long-tail Test14-hard split. These results demonstrate the effectiveness of R$^2$LPL in converting recoverable closed-loop mistakes into corrective knowledge for sustained policy improvement. 

---
# Informational Frustration in Neural Manifolds: Shannon Bottlenecks and the Limits of Learnability 

**Authors**: Srinivasa Rao P., Vangmayi P Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2606.30512)  

**Abstract**: Why overparameterised deep networks generalise so remarkably well remains one of the most stubborn open questions in machine learning theory. Classical frameworks like VC dimension and Rademacher complexity predict catastrophic overfitting in modern models, leaving a massive theoretical gap between theory and reality. In this paper, we bridge this divide by introducing a unified framework that links information theory, topology, and statistical mechanics to map the hard limits of deep learning. Central to our approach is the Entropic Learnability Horizon (ELH): a fundamental law stating that a network can only truly learn a target function if the Shannon entropy of the data manifold outpaces the topological entropy of the function's decision boundary, balanced by the von Neumann entropy of the network's weight space. We establish the Shannon-Topological Bottleneck Theorem, proving that when a target boundary's geometric complexity exceeds this informational horizon, the system undergoes a sudden entropic phase transition. It falls into a state of Informational Frustration - a glassy, rigid memorization phase where generalization becomes thermodynamically impossible. Using this lens, we show that the enigmatic phenomenon of "grokking" is actually an Entropic Release, where weights abruptly reorganise to unlock the bottleneck. Finally, we translate this theory into practice with Entropic Gradient Descent (EGD), an optimization algorithm that dynamically manages weight entropy to keep learning on track. Ultimately, this work repositions entropy not just as a tool for tracking uncertainty but as the fundamental physical currency that dictates whether a machine can learn. 

---
# On the Faithfulness of Post-Hoc Concept Bottleneck Models 

**Authors**: Laines Schmalwasser, Jan Blunk, Niklas Penzel, Julia Niebling, Joachim Denzler  

**Link**: [PDF](https://arxiv.org/pdf/2606.30498)  

**Abstract**: Human decision-making interprets the world through high-level concepts, such as recognizing a bird by its belly color. To bridge the gap between opaque deep learning representations and human understanding, Post-Hoc Concept Bottleneck Models (post-hoc CBMs) project latent features onto interpretable concept spaces using auxiliary datasets or vision-language models. However, relying on target task accuracy as the primary measure of post-hoc CBM success obscures whether the learned concepts are semantically meaningful or merely predictive artifacts. For example, random concept projections can achieve competitive accuracy despite being semantically meaningless. In this work, we analyze the learned projections directly and identify two failure cases: First, for concept projections learned from auxiliary data, covariate shifts can lead to unfaithful concept representations for the target task. In particular, we provide an upper bound on the error introduced by this shift. Second, systematic label noise in surrogate concept labels generated by vision-language models leads to unfaithful projections. After formalizing these failure modes, we introduce novel metrics that decouple concept faithfulness from predictive accuracy. Our empirical results across real-world and synthetic benchmarks confirm that these metrics identify unfaithful behaviors that standard accuracy-based evaluation fails to detect. 

---
# McMg: A Learned Phase-Space Multi-channel Multigrid Preconditioner for Helmholtz Equation 

**Authors**: Jiwei Jia, Xinliang Liu, Juntao Wang, Jinchao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.30495)  

**Abstract**: Solving heterogeneous Helmholtz equations at high wavenumbers remains challenging because the discretized operator is indefinite, pollution degrades phase accuracy, and scalar coarse-grid correction can discard the local phase and propagation-direction information carried by oscillatory errors. We propose Multi-channel Multigrid (McMg), a learned phase-space multigrid preconditioner for heterogeneous Helmholtz equations. Rather than predicting the solution directly, McMg maps residuals to corrections within an iterative framework. Its central idea is to coarsen physical space while retaining unresolved local wave information in the channel dimension: each coarse node carries a learned packet of amplitude, phase, direction, and scattering coefficients rather than a single scalar unknown. The architecture combines linear multi-channel transfer operators with locally adaptive stencils, neural PDE operators, and medium-dependent smoothers whose coefficients are generated from the wave speed. For a fixed medium, the V-cycle is linear in the residual; nonlinear physical features are computed once in a setup phase and cached, so each online iteration reduces to convolutions with fixed coefficients. We further study generalization across scales. Models trained on small domains transfer directly to larger domains and higher effective wavenumbers, and a Layer-by-Layer Progressive Finetuning (LLPF) strategy extends the support of the learned Green's operator by adding and finetuning only new coarse levels. Numerical experiments on high-frequency, high-contrast, and large-scale three-dimensional problems demonstrate that McMg requires substantially fewer iterations and less wall-clock time than strong classical baselines, while consistently outperforming existing neural preconditioners. 

---
# SIMAX: A Scalable and Interpretable Framework for Multi-Fidelity and Annotated Clinician-Patient Dialogue Simulation 

**Authors**: Zhuhan Bao, Rui Yang, Bohao Yang, Zhiyi Liu, Sicheng Shu, Ruio Heerschap, Le Li, Doris Yang, Elisabeth Bond, Haoyuan Wang, Nicoleta Economou-Zavlanos, Joshua M. Biro, Matthew McDermott, Nan Liu, Anand Chowdhury, Kai Sun, Kathryn Pollak, Ed Hammond, Chuan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2606.30491)  

**Abstract**: Background. The widespread deployment of ambient digital scribes is driving large-scale capture of clinician-patient dialogues. Human coding of clinical communication data remains costly, inconsistent, and difficult to scale, motivating AI-driven communication coding systems. However, evaluating these systems requires real-world dialogues and human-coded labels, both hard to obtain at scale.
Methods. We developed SIMAX (Scalable and Interpretable Framework for Multi-Fidelity and Annotated Clinician-Patient Dialogue Simulation), a framework for generating controlled clinical dialogue data with reference behavioral annotations. SIMAX generates clinician-patient dialogues from predefined clinical scenarios, personas and voice conditions, and target communication behaviors. Behaviors are controlled using two codebooks: the Global Codebook for overall communication quality and the WISER Codebook for specific countable behaviors. We evaluated SIMAX using automated and human quality assessments and an example communication coding system.
Results. SIMAX generated 3,388 simulated dialogues across three specialties, multiple visit stages, persona characteristics, and accent conditions. Automated assessment showed mean UTMOS and WV-MOS scores of 3.03 and 2.61, WER and CER of 0.07 and 0.05, and CLAP cosine similarity of 0.41, suggesting reasonable speech naturalness, high transcription fidelity, and positive text-audio correspondence. Human evaluation showed a median MOS of 4.67 and a median clinical realism score of 3.00. Downstream evaluation suggests that SIMAX can assess how a communication coding system responds to behavioral targets and reveal insufficient sensitivity in some dimensions.
Conclusions. SIMAX generates controlled and reproducible simulated clinician-patient dialogues, providing a data foundation for developing, validating, and refining communication coding systems. 

---
# Situation Perception: A Necessary Primitive to Artificial Superintelligence 

**Authors**: Ziqin Yuan, Jaymari Chua  

**Link**: [PDF](https://arxiv.org/pdf/2606.30481)  

**Abstract**: Current large language models are extraordinary statistical engines. They compress vast amounts of text into useful patterns and can explain science, write code, imitate reasoning, and participate in philosophical conversation. Yet pattern mastery is not the same as general intelligence. A human infant begins with little explicit knowledge, but gradually discovers object permanence, cause and effect, other minds, bodily agency, and the persistence of the physical world. We make an argument that the path to artificial superintelligence (ASI) depends on a missing capacity we call \emph{situation perception}: the ability to construct, revise, and act within internal simulations of possible worlds across latent time. \emph{ perception} requires at least three core components: abstract prediction, long-term compressed memory, and active learning guided by objectives. In this work, we analyse why modern large language models remain incomplete, and propose the appropriate tests for measuring progress and consequences of machines that can simulate futures, pursue self-directed goals, and possibly judge their own creators. 

---
# COHORT: Collaborative Orchestration for Hardening via Offensive Replay on Emulated Topologies 

**Authors**: Chen Frydman, Aviram Zilberman, Rubin Krief, Abed Showgan, Andres Murillo, Sekiya Motoyoshi, Asaf Shabtai, Yuval Elovici, Rami Puzis  

**Link**: [PDF](https://arxiv.org/pdf/2606.30479)  

**Abstract**: Mitigating an observed adversary in an enterprise network typically takes weeks of expert work: an analyst derives a mitigation tailored to that adversary, validates it without breaking production, and verifies it disrupts the specific attack. The procedure relies on expert judgment and cannot safely be exercised against the production network. COHORT is the first end-to-end framework to automate this procedure for deployable mitigations. A role-decomposed multi-agent LLM workflow proposes candidates, implements them as real device commands, and refines them through a critique loop, all on a high-fidelity GNS3 emulator running real vendor firmware (firewall, switch, router). Each candidate is evaluated by offensive replay: re-executing the original adversary on the mitigated network for a paired comparison against the unmitigated baseline, rather than the reward-signal or expert-judgment proxies used in prior simulation, hybrid, and configuration-generation work. Two further checks complement replay: a connectivity-regression check (LAN ping and internet HTTP probe) rejects mitigations that disrupt legitimate LAN or internet connectivity, and a cumulative evaluation stacks approved mitigations onto a persistent state to surface compound effects. Across three topologies and four attack scenarios (ransomware, lateral movement, DNS exfiltration, data theft), 46.7% of generated mitigations both disrupt the attack and preserve connectivity under replay, 4.4 times the rate of a single-agent baseline using the same model and tool access. A demo video walking through the framework is available with our released artifacts. 

---
# Field Order Should Not Matter: Permutation-Invariant Embedding Model Fine-Tuning for Structured Metadata Retrieval 

**Authors**: Aivin V. Solatorio, Olivier Dupriez, Rafael Macalaba  

**Link**: [PDF](https://arxiv.org/pdf/2606.30473)  

**Abstract**: We study retrieval over catalogs of structured metadata, where each record is a small schema whose fields answer different kinds of query. Embedding a record with a text encoder first serializes its fields into a string, which forces a choice of field order. We show this choice, usually treated as an implementation detail, silently controls retrieval quality once the encoder is fine-tuned. A standard fine-tune loses 7.4 nDCG@10 points when the index is rebuilt under a different field order, because it reads absolute position instead of the field labels. We propose permutation-invariant fine-tuning ($\textbf{PI-FT}$), which serializes each record under a freshly sampled field order with random field dropout, so meaning binds to the labels rather than to position. The change is about two lines in the data loader; it costs negligible in-distribution accuracy and cuts the order-change penalty to 0.2 points. We study this in the discovery of development statistics, a catalog of nearly 10,000 indicators that should be searchable in many languages by a model small enough to self-host. As AI assistants and agents increasingly mediate access to public data and statistics, this retrieval step decides whether an answer is grounded in the right indicator or series, making discoverability a precondition for disseminating data through AI. Because usage logs cannot provide training signal for indicators no one has searched, we generate the queries instead. $\textbf{DevDataBench}$ is a fully LLM-generated benchmark of grounded, facet-targeted queries across 15 languages, covering every indicator for both training and evaluation. A fine-tuned 118M-parameter CPU encoder outperforms every zero-shot baseline, including $\texttt{text-embedding-3-large}$ (0.707 vs.\ 0.556 nDCG@10), with the largest gains in low-resource languages. We release the benchmark, pipeline, models, and a reusable PI-FT framework. 

---
# Collective cooperation without individual fidelity in LLM agents 

**Authors**: Henrique Ferraz de Arruda, Carlos Gracia Lázaro, Alberto Aleta, Yamir Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2606.30454)  

**Abstract**: Large language models (LLMs) are increasingly used as agents in simulations of social systems, yet it remains unclear when their behavior can be interpreted as a faithful proxy for human decision-making. Here we test LLM agents against a direct empirical benchmark: a large-scale networked Prisoner's Dilemma experiment with human participants. Using the same interaction protocol, payoff structure, and network topologies, we compare nine open-weight LLMs with the human data. The selected model reproduces several macro-level features of cooperation dynamics, including the early decline and later stabilization of cooperation. This aggregate agreement, however, does not extend uniformly to finer levels of behavior. LLM populations underestimate individual-level heterogeneity and generate conditional cooperation patterns that differ from those observed in humans. Adding a fraction of random agents improves some aspects of micro-level agreement, but does not remove the mismatch in decision rules. These findings reveal a macro--micro dissociation in LLM-based social agents: collective outcomes can appear human-like even when the underlying behavioral distributions and mechanisms are not. They suggest that validating LLM agents as human surrogates requires comparisons across aggregate dynamics, individual heterogeneity, and context-dependent decision rules, rather than outcome-level agreement alone. 

---
# Translating Natural Language to Strategic Temporal Specifications via LLMs 

**Authors**: Marco Aruta, Francesco Improta, Vadim Malvone, Aniello Murano, Vladana Perlic  

**Link**: [PDF](https://arxiv.org/pdf/2606.30441)  

**Abstract**: A rigorous formalization of system requirements is a fundamental prerequisite for the verification of Multi-Agent Systems (MAS). However, writing correct formal specifications is well known as an error-prone, time-consuming, and expertise-intensive task. This difficulty is further accentuated in MAS, where requirements must capture strategic abilities and temporal objectives. At present, there is no established methodology for deriving MAS specifications from natural language. We present a framework for translating Natural Language descriptions of strategic requirements into well-formed ATL/ATL* formulas using Large Language Models (LLMs). Since no available dataset supports supervised learning for the NL-to-ATL/ATL* translation task, we create and curate a novel expert-validated dataset, employed for training and evaluating fine-tuned models. On a held-out test set, evaluated under the LLM judge that best agrees with expert annotations, in-domain fine-tuning of small open-weight models (3 - 7B parameters) matches strong few-shot proprietary API baselines. Our best fine-tuned system reaches 0.84 semantic accuracy, statistically on par with 0.86 for the strongest few-shot proprietary baseline, while keeping requirements on-premises. We further find that judge reliability is inverse to generator strength. The open-weight Llama-3.3-70B tracks human verdicts most closely, whereas the strongest proprietary models are the least reliable judges, over-rejecting faithful paraphrases of the reference. To assess the practical applicability of the generated specifications, we embed our tool to an existing strategic logics model checker, enabling non-expert users to specify strategic properties in natural language. 

---
# Transformer Architectures as Complete Bayes Processes: A Formal Proof in the Measure-Theoretic Kernel Framework 

**Authors**: Haobo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.30440)  

**Abstract**: We present a complete formal proof that transformer architectures, when their internal update mechanisms satisfy a Bayes joint-distribution condition, implement exact Bayesian posterior inference. Working within the measure-theoretic kernel framework, we define a hierarchy of abstractions -- from the core Bayesian transformer, through semantic transformers with explicit update kernels, to full transformer blocks with QKV/attention/residual/MLP pipelines, and finally multilayer stacks -- and prove at each level that the Bayes joint semantics implies the update kernel equals the posterior almost everywhere. For the block-level architecture, we derive the explicit Bayes formula through Radon-Nikodym differentiation and prove its normalization. We additionally prove that the softmax attention mechanism induces a valid probability distribution over keys, establishing the bridge between the abstract kernel framework and concrete attention implementations. The framework makes no architectural assumptions beyond the Markov kernel structure and exposes explicit conditions under which a transformer block is provably Bayesian. In essence, when this joint distribution condition is satisfied, the forward computation of a Transformer is formally equivalent to a rigorous Bayesian posterior update. 

---
# Beyond Point Estimates for Glaucoma Visual Field Forecasting with Diffusion Models 

**Authors**: Marta Colmenar Herrera, Pablo Márquez Neila, Şerife Seda Kucur Ergünay, Martin S. Zinkernagel, Raphael Sznitman  

**Link**: [PDF](https://arxiv.org/pdf/2606.30417)  

**Abstract**: Forecasting visual fields (VFs) is critical for personalized monitoring and treatment planning in glaucoma. This is inherently uncertain due to heterogeneous disease progression and measurement variability, yet most existing methods produce single deterministic predictions that fail to represent this uncertainty. We formulate VF forecasting as a probabilistic prediction problem and the use of conditioned denoising diffusion models to generate distributions of plausible future VFs from longitudinal observations with irregular follow-up intervals. Experiments on two independent VF cohorts show that diffusion-based predictions produce well-calibrated distributions for clinically relevant VF measures. When reduced to a standard point-estimate, the proposed approach achieves state-of-the-art accuracy compared to clinical baselines and prior learning-based methods. Our results highlight the advantages of distributional modeling for VF forecasting and support a shift from point-estimate prediction toward uncertainty-aware, clinically interpretable risk assessment in glaucoma. 

---
# Can LLMs Rank? A Tale of Triads and Triage 

**Authors**: Gaurab Pokharel, Shafkat Farabi, Patrick J. Fowler, Sanmay Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.30412)  

**Abstract**: From housing allocation for households experiencing homelessness to triage in emergency departments, LLMs are increasingly being considered as judges of consequential decisions that require ranking people for scarce resources. Ranking large groups simultaneously is cognitively demanding and error-prone. A natural solution, drawing on decades of social choice theory, elicits pairwise comparisons and aggregates them into a total order. However, a fundamental question remains when LLMs serve as the pairwise judge: how can a practitioner tell, before committing to a ranking, whether the LLM's judgments are sufficiently consistent to trust the result? We discuss two different ways of identifying consistency. A classical diagnostic, the coefficient of consistency $\zeta$, originally developed to measure judge reliability by counting circular triads in tournament graphs, provides a cheap, model-free measure of intra-run consistency. Various standard measures of distance between rankings, for example Kendall's $\tau$, can measure inter-run variability. We show, in both theory and practice, that these measures are independently valuable, and advocate for using both to assess reliability of rankings. We demonstrate the practical importance of our results across two high-stakes prioritization tasks: homelessness service allocation and emergency department triage. Three different leading LLMs have considerably different performance profiles across these two axes of consistency. We provide guidelines for how practitioners could think about measuring and assessing consistency before committing to a model for ranking or prioritization. 

---
# Beyond IID: How General Are Tabular Foundation Models, Really? 

**Authors**: Lennart Purucker, Andrej Tschalzev, Nick Erickson, Gioia Blayer, David Holzmüller, Alan Arazi, Alexander Pfefferle, Mustafa Tajjar, Gaël Varoquaux, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2606.30410)  

**Abstract**: Foundation models for predictive machine learning on tabular data have recently gained significant traction in academia and industry. Research communities across disciplines are increasingly evaluating tabular foundation models on diverse datasets and tasks. However, these task- and discipline-specific evaluations remain largely inaccessible to model researchers because benchmark software and evaluation protocols are fragmented. As a result, model researchers rely on standard benchmarks, which are mostly defined for tasks where tabular foundation models already excel. The most challenging scenarios are excluded, limiting meaningful progress in the field by focusing on marginal improvements on IID data rather than on broader, more demanding challenges. To overcome this, we introduce BeyondArena, the first unified holistic benchmark for tabular data that supports diverse task types (IID, temporal, grouped), across sample size and feature dimensionality scales, with diverse feature types (with text, with high cardinality) from a broad range of disciplines. To enable unified benchmarking beyond standard benchmarks, we introduce Data Foundry, a Python framework and metadata schema for curating tabular datasets for predictive machine learning. Our results across 11 models and 142 curated datasets show that existing tabular foundation models excel on tiny- to medium-sized IID data, while traditional tree-based and deep learning models still dominate on non-IID, large, and high-dimensional datasets. BeyondArena guides model research for the most demanding challenges in tabular data, enabling progress towards truly foundational tabular models. 

---
# Model Predictive Current Control with Harmonic Correction for Single-Phase AC-DC EV Charging 

**Authors**: Changhong Li, Bharathkumar Hegde, Biswajit Basu, Shreejith Shanker  

**Link**: [PDF](https://arxiv.org/pdf/2606.30397)  

**Abstract**: The increasing integration of Electric Vehicles (EVs) has imposed a growing harmonic challenge on the power grid. For AC/DC Power Factor Correction (PFC) in single-phase On-Board Chargers (OBCs), Model Predictive Current Control (MPCC) improves the current quality by predicting and tracking the inductor current. However, finite control set MPCC selects switching states, resulting in discrete control actions and a limited optimisation space. Moreover, the MPCC cost function based on instantaneous current tracking error has limited capability to compensate for low-order harmonic disturbances induced by dead time, control delay, and model parameter mismatch. This paper proposes a duty cycle predictive MPCC incorporating a real-time harmonic estimation reference. The proposed method dynamically estimates the low-order harmonic components of the input current and corrects the MPCC reference current, enabling continuous duty cycle control and targeted suppression of dominant low-order harmonics. Simulation results on a single-phase OBC demonstrate that the proposed duty cycle predictive MPCC reduces the steady-state current THD_i from 11.47% to 6.10% compared with the switching state predictive MPCC. With the harmonic reference, the THD_i is further reduced to 2.85%. 

---
# A Stochastic--Geometric Theory of Scaling Laws in Grokking 

**Authors**: Róisín Luo, Christian Gagné, Jonas Ngnawé, Ihsan Ullah, Karyn Morrissey  

**Link**: [PDF](https://arxiv.org/pdf/2606.30388)  

**Abstract**: Delayed generalization (\ie~grokking) refers to the phenomenon in which a neural network fits its training data early in training but only begins to generalize after a prolonged delay, often through an abrupt transition. Despite extensive empirical study, its underlying mechanism remains poorly understood. In this work, we first theoretically characterize a shell--core topological configuration of the reachable solution space induced by Adam's optimization dynamics with weight-shrinkage regularization, supported by empirical evidence. This optimization-induced topological configuration gives rise to grokking. In model's parameter space, random initialization solutions concentrate on a thin outer spherical shell, enclosing another spherical shell of memorization solutions, which in turn contains a core corresponding to the generalization solutions. Leveraging stopping-time theory, we then analyze the geometry of this topological configuration and the solution transition time at which optimization trajectories escape the memorization manifold and first reach the boundary of the generalization manifold. Our theoretical analysis derives grokking scaling laws for the learning rate, batch size, and $\ell_2$ regularization coefficient, which are further validated through experiments and shown to recover results from prior literature. 

---
# Set-Inclusive Uncertainty Modeling for Robust Brain Tumor Segmentation 

**Authors**: Seunghun Baek, Jihwan Park, Jaeyoon Sim, Hoseok Lee, Seungjoo Lee, Won Hwa Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.30374)  

**Abstract**: Multimodal MRI is essential for accurate brain tumor segmentation. However, acquiring all modalities at inference is often challenging in practice, which causes intrinsic uncertainty due to unavoidable information loss. Without modeling this uncertainty, existing methods encode incomplete evidence into deterministic representations that appear plausible but lack reliability. In this regime, we propose a probabilistic representation framework that models representations as Gaussian distributions, where their mean captures task information and their variance measures uncertainty from missing evidence. To make variance reflect information deficiency, we regularize the mean from each partial configuration toward its full-modality counterpart, while scaling the variance with the discrepancy between their aligned means. We further introduce a set-inclusive strategy that exploits the hierarchical structure of modality subsets and enforces an ordering constraint to maintain their consistent uncertainty relationships. Extensive experiments on BraTS 2018 and 2020 demonstrate that our approach offers superior performance over baselines across diverse missing-modality scenarios. Code and model checkpoint are available at this https URL. 

---
# ReactiveBFM: Reactive Closed-Loop Motion Planning Towards Universal Humanoid Whole-Body Control 

**Authors**: Xiao Chen, Weishuai Zeng, Xiaojie Niu, Zirui Wang, Jianan Li, Huayi Wang, Furui Xu, Jiahe Chen, Weixiang Zhong, Lihe Ding, Kailin Li, Jiangmiao Pang, Tai Wang, Tianfan Xue, Jingbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.30362)  

**Abstract**: While current Behavior Foundation Models (BFMs) provide robust control priors for humanoids, they only execute pre-defined reference motions. As a result, they are vulnerable to environmental shifts and incapable of reactive whole-body coordination. Naively cascading them with generative motion planners fails to achieve true reactivity, as inevitable tracking discrepancies induce fatal cumulative exposure bias. To bridge this gap, we propose ReactiveBFM, a real-time closed-loop planning-control framework. At its core, we effectively mitigate exposure bias via a scheduled prefix sampling curriculum, forcing the generative planner to actively learn error-recovery behaviors from imperfect physical states rather than ground-truth trajectories. Systematically, to reconcile the severe latency mismatch between auto-regressive planning and high-frequency tracking, we introduce an asynchronous replanning mechanism. Combined with trajectory chunking to temporally ensemble spatial references, our system guarantees spatio-temporally fluid execution without physical jitter. Deployed on the Unitree G1 humanoid, ReactiveBFM demonstrates unprecedented physical agility across a vast repertoire of text-conditioned closed-loop motions. Notably, ReactiveBFM achieves zero-shot moving target reaching, showcasing intricate whole-body coordination and on-the-fly replanning. In sim-to-sim benchmarking under severe perturbations, ReactiveBFM achieves a 93.1% success rate, significantly outperforming cascaded open-loop baselines by 28.6%. 

---
# Residual-Guided Expert Specialization for Incomplete Multimodal Learning 

**Authors**: Seunghun Baek, Jihwan Park, Jaeyoon Sim, Minjae Jeong, Hoseok Lee, Won Hwa Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.30355)  

**Abstract**: As real-world prediction systems often face missing modalities at inference, incomplete multimodal learning (IML) remains a practical challenge. While prior methods aim to learn representations robust to missing inputs, representations from incomplete modalities inevitably deviate from their full-modality counterparts due to missing evidence. To explicitly leverage these deviations, we propose MARS (Missingness-Aware Residual-guided Specialization), a mixture-of-experts framework that guides expert specialization based on how representations are reshaped by missingness. By contrasting task representations derived from incomplete inputs with their complete counterparts during training, we derive a privileged residual signal that captures this representational gap. The residual signal guides a residual router to assign samples to experts specialized for the corresponding deviation patterns. In parallel, a feature router learns to imitate this routing behavior using only incomplete inputs, enabling deployment without access to full modalities. To mitigate this train-test router gap, we develop a discrepancy-aware noise regularization that adaptively perturbs the residual router's decisions when the feature router deviates, enhancing expert robustness under imperfect imitation. Experiments on multimodal classification (CASIA-SURF, CREMA-D, UPMC Food-101) and segmentation (MCubeS) under missing scenarios show that MARS consistently surpasses baselines while remaining efficient and extensible to diverse backbones and tasks. 

---
# FFAvatar: Feed-Forward 4D Head Avatar Reconstruction from Sparse Portrait Images 

**Authors**: Jianjiang Yao, Ke Xian, Renxiang Dai, Robert Caiming Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.30347)  

**Abstract**: We present FFAvatar, a Transformer-based 3D Gaussian framework for fast construction of high-quality and animatable 4D head avatars from one or more reference portrait images. Unlike existing feed-forward approaches that require a fixed number of input views, FFAvatar supports incremental reconstruction, progressively refining the avatar representation as additional reference images become available. At the core of our method is an alternating attention mechanism that disentangles identity appearance from expression and viewpoint variations, enabling the reconstruction of a canonical 3D appearance that remains consistent across poses and facial expressions. To balance visual fidelity and computational efficiency, we introduce a sparse-to-dense learning paradigm. Coarse appearance features are first learned using sparse primitives anchored to the FLAME vertex level and are subsequently densified in the UV domain to capture fine-grained geometric and texture details. We further propose a plug-and-play motion refinement module that enables subject-specific dynamic personalization by modeling residual motion beyond parametric deformation. Extensive experiments demonstrate that FFAvatar efficiently produces high-fidelity and controllable 4D head avatars, achieving superior flexibility, driving efficiency, and identity-consistent rendering across diverse expressions and viewpoints. 

---
# DRIFT: Difficulty Routing Self-DIstillation with Rhythm-Gated Exploration and Success BuFfer Training 

**Authors**: Haisen Luo, Yiwei Liu, Haoning Wang, Dan Liu, Junxi Yin, Haotian Wang, Lei Zhang, Xiaoyu Tian, Shuaiting Chen, Yuansheng Song, Baoyan Guo, Xiongfei Yan, Bolan Yang, Chengwei Liu, Ming Cui, Jiong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.30345)  

**Abstract**: Enabling large language models to achieve stable self-improvement without external expert supervision remains a central challenge in complex reasoning tasks. Existing self-distillation and reinforcement learning methods lack explicit mechanisms for tracking problem-level learning progress and adapting optimization strategies accordingly. Consequently, training may over-optimize easy problems, receive weak supervision from hard problems, and fail to sufficiently explore borderline cases. To resolve these issues, we propose DRIFT, an online self-evolution policy optimization framework for large language models. DRIFT regulates the model's self-improvement process through the joint use of Difficulty Routing and Rhythm Gating. The former identifies the model's learning state at the problem level and dynamically allocates self-distillation and reinforcement learning signals, while the latter refines policy updates at the token level, concentrating exploration on critical reasoning positions. By further incorporating a success buffer and a two-stage curriculum learning strategy, DRIFT preserves high-quality historical experience while progressively guiding the model from reliable behavior acquisition toward stable policy evolution. Evaluated across five benchmarks and three model scales, DRIFT surpasses the peak performance of both GRPO and SDPO across all evaluated metrics. On the average score over the five benchmarks, DRIFT achieves 79.5$\%$, outperforming GRPO by 9.5$\%$ and SDPO by 7.5$\%$, establishing a new state-of-the-art result. Notably, on ToolUse, DRIFT reaches an accuracy of 79.2$\%$, improving over GRPO by 13.5$\%$ and SDPO by 10.7$\%$, setting a new state-of-the-art and substantially outperforming all concurrent methods. 

---
# Early Cue Precision Shapes Visual Shortcut Learning in Controlled Cue-Manipulation Benchmarks 

**Authors**: Chanho Park, Woochan Lee, Janyeong Oh, Geongho Gong, Minshu Kim, Yeachan Kwak, Seongim Choi  

**Link**: [PDF](https://arxiv.org/pdf/2606.30344)  

**Abstract**: Visual classifiers can achieve high matched-distribution accuracy while relying on low-level cues that fail under conflict or suppression. We test whether this failure is shaped by early cue precision: the reliability with which a low-level cue predicts the label during early learning or downstream probe fitting. Across synthetic shape-texture tasks, sequential digit training, a 10-class frozen-representation audit, and a CIFAR-10 natural-image-based texture-overlay benchmark, we manipulate object-texture match probability and evaluate matched-ID accuracy, conflict accuracy, texture-choice rate, and suppression behavior. Degraded-but-predictive input does not substitute for cue decorrelation. In 10-class digit probes, conflict accuracy drops from 0.589 under chance-like cue precision to 0.005 under target-perfect texture. In CIFAR-10 frozen probes, conflict accuracy drops from 0.569 to 0.114, while texture choice rises from 0.049 to 0.855; this ordering persists across texture-overlay strengths alpha in {0.15,0.25,0.35,0.50}. End-to-end CIFAR-10 training shows that low early cue precision improves pre-target conflict behavior, but shortcut-rich fine-tuning can rapidly overwrite this benefit. Cue decorrelation must therefore be maintained during downstream adaptation rather than treated as a one-time inoculation. 

---
# MCP Server Architecture Patterns for LLM-Integrated Applications 

**Authors**: Carson Rodrigues, Oysturn Vas  

**Link**: [PDF](https://arxiv.org/pdf/2606.30317)  

**Abstract**: The Model Context Protocol (MCP), introduced by Anthropic in November 2024, defines a standardized interface for connecting large language models (LLMs) to external tools, data sources, and services. Within months of release, hundreds of community-built MCP servers appeared on GitHub, but no software-maintenance literature has yet described how the ecosystem is being structured in production. This industry experience paper catalogues five recurring MCP server architectural patterns observed across an enumerated corpus of fifteen independently developed servers (five production servers from the ANSYR voice AI platform plus ten public servers from the official MCP registry): Resource Gateway, Tool Orchestrator, Stateful Session Server, Proxy Aggregator, and Domain-Specific Adapter. Each pattern is described in the structured form of Gamma et al.: context, problem, solution, and consequences. We also document four anti-patterns and a set of cross-cutting concerns around authentication, versioning, and observability. The quantitative evaluation contributes three measurements: inter-rater reliability of the taxonomy across two independent LLM raters on 54 held-out servers (Cohen's kappa = 0.76), which also localizes three pattern-boundary ambiguities; transport overhead measured end-to-end on loopback and modeled for cross-host paths; and a tool-count study showing tool-selection accuracy drops below 90% between 10 and 15 tools per context for Claude Haiku 4.5 and between 20 and 30 tools for Sonnet 4. Code, corpus, and prompts are released as a replication package. 

---
# Always-OnAgents:A Survey of Persistent Memory, State, and Governance in LLMAgents 

**Authors**: Tianyu Ding, Aditya Nannapaneni, Bingfan Liu, Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.30306)  

**Abstract**: Always-on agents are systems whose future behavior depends on durable state accumulated across earlier interactions. We treat them as persistent-state systems: the operative system includes retrievable memories, but also task ledgers, permissions, credentials, commitments, provenance and audit records, shared state, trigger conditions, and externally committed effects linked to those records. The survey reads the literature through six diagnostic axes for each state item, authority, scope, mutability, provenance, recoverability, and actionability, and through a lifecycle in which state is written, validated, organized, retrieved, acted upon, updated, forgotten, audited, and sometimes rolled back. Across a 435-work coded corpus, treated as a scoped map rather than an exhaustive census, the literature concentrates more heavily on accumulating and retrieving state than on governing, recovering, or relinquishing it. We therefore introduce the Always-On Evaluation Protocol (AOEP-v0), a pilot evaluation contract that makes these governance requirements concrete by scoring state mutation and recovery obligations rather than answer quality alone. The resulting agenda connects always-on agents to databases, distributed systems, formal methods, capability security, and machine unlearning. 

---
# Research Entity Extraction and Topic Detection from UKRI Grant Proposals 

**Authors**: Xingran Ruan, Angelo Salatino, Rosa Filgueira, Kara Moraw, Alexandru Marcoci, Gemma Derrick, Sarah Callaghan  

**Link**: [PDF](https://arxiv.org/pdf/2606.30304)  

**Abstract**: This paper presents preliminary findings from a UKRI-funded Metascience project comparing three LLM-based approaches, GPT-4o, Mistral, and a bespoke algorithm, DSIT-Taxonomies, for extracting and classifying research entities from funding proposals. Our project "Tracking Stars and Unicorns" aims to identify early signals of emerging research areas to inform public investment. Our methodology employed a three-stage pipeline, leveraging Mistral for primary entity extraction and mapping against the OpenAlex Topics taxonomy. We evaluated our approach across 42 proposals' abstracts from different areas and observed that Mistral and GPT-4o produce comparable, high-quality entity sets with significant semantic overlap, outperforming the fragmented DSIT-Taxonomies approach. Crucially, the Mistral-based approach achieved superior topic classification accuracy (90.5%) compared to the full DSIT-Taxonomies pipeline (71.4%). We conclude that Mistral offers a high-performance, operationally efficient, and secure solution for large-scale analysis of sensitive grant data. 

---
# Towards Continual Motion-Language Agents: LoRA Variants for Incremental Motion Understanding and Generation 

**Authors**: Bertram Taetz, Hugo Albuquerque Cosme da Silva, Gabriele Bleser-Taetz  

**Link**: [PDF](https://arxiv.org/pdf/2606.30266)  

**Abstract**: Motion-language agents must possess the bidirectional capability to both understand human movement (motion-to-text, M2T) and generate it from natural language (text-to-motion, T2M). While foundational models have achieved strong performance in static settings, autonomous agents operating in dynamic environments must continuously incorporate new motion concepts -- such as novel athletic styles or specialized gestures -- without catastrophic forgetting of previously acquired skills. We investigate the stability-plasticity trade-off in bidirectional motion-language learning under sequential task exposure. Building on a frozen large language model backbone, we introduce low-rank adaptation (LoRA) variants designed to mitigate inter-task interference. We specifically propose mixture-of-experts architectures that utilize an autoencoder-based router to select task-specific experts at inference time, so that no task-label is needed. To evaluate these methods, we establish a reproducible five-task benchmark derived from HumanML3D through semantic clustering of motion descriptions. Our experimental results demonstrate near-zero forgetting across both M2T and T2M directions while maintaining high generation and captioning quality. Furthermore, we show that hard expert selection via routing significantly outperforms soft expert blending in quality metrics, indicating that preserving expert isolation is critical for maintaining performance in our continual learning setting. Finally, we observe that a divergence between token-level accuracy and downstream generation quality may occur, highlighting the need for more comprehensive evaluation protocols in future research on lifelong motion-language agents. 

---
# Defending Against Harmful Supervision Hidden in Benign Samples 

**Authors**: Bang An, Yibo Yang, Dandan Guo, Ebtisam Alshehri, Carlos Hinojosa, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2606.30263)  

**Abstract**: Existing defenses are effective when harmful content is explicitly mixed into downstream fine-tuning data, but crafted samples can instead hide harmful supervision inside benign tasks. We propose Embedded Attack, where harmful QA pairs are embedded within benign training samples, and show that representative guardrails often fail to detect them at the example level. To address this, we propose Dual-Reference SFT (DR-SFT), which adapts DPO-style contrastive objective design to SFT through token-level regularization, mitigating harmful fine-tuning beyond coarse data filtering. 

---
# KnowsTFM: Knowledge-Informed Fine-Tuning of Small Tabular Foundation Models 

**Authors**: Boshko Koloski, Xiangjian Jiang, Senja Pollak, Blaž Škrlj, Mateja Jamnik, Nikola Simidjievski  

**Link**: [PDF](https://arxiv.org/pdf/2606.30258)  

**Abstract**: Tabular foundation models have advanced deep learning for tabular data by delivering strong default performance across many small and medium tasks. Yet in niche domains, where data is scarce, high-dimensional, and shifted from the pretraining distribution, they may still fail to outperform carefully designed domain-specific methods. Many such domains also provide curated relational knowledge in the form of knowledge graphs and knowledge banks, but how to use this knowledge to improve and steer \textit{small} specialist tabular foundation models remains unclear. We address this problem through \textbf{Know}ledge-informed fine-tuning of \textbf{s}mall \textbf{T}abular \textbf{F}oundation \textbf{M}odels (\modelname). Specifically, we study nanoscale TabPFN- and TabICL-style variants, pretrained under controlled synthetic prior families and adapted using two complementary mechanisms: structural attention priors derived from knowledge graphs and parameter-efficient low-rank updates. We show that injecting domain-specific structural knowledge during fine-tuning yields meaningful gains over vanilla variants in specialist settings, whereas gains on general-domain tasks are marginal. We further observe that continual fine-tuning of frontier models can trigger collapse of pretrained knowledge and mechanisms. 

---
# Curvature-Guided Sheaf Diffusion for Unsupervised Community Detection on Heterophilic Graphs 

**Authors**: Feifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.30249)  

**Abstract**: Detecting communities in heterophilic graphs -- where connected nodes often belong to different classes -- is hard for unsupervised methods: classical modularity and spectral methods are feature agnostic, while deep graph-clustering methods rely on contrastive or generative machinery that is opaque. We propose Curvature-Guided Sheaf Diffusion (CGSD), a fully unsupervised community-detection algorithm that uses the discrete Forman--Ricci curvature of each edge as its single topological signal, propagated through every stage of an end-to-end pipeline. CGSD makes three concrete contributions: (i)~a curvature-gated sheaf-diffusion encoder that gates edge messages by $\sigma(\kappa_e)$ and is trained from three label-free structural losses (modularity, anti-collapse, curvature-weighted reconstruction); (ii)~a curvature-aware spectral clusterer (CSpec) that re-weights the $k$-NN affinity of the embedding by $\sigma(\alpha \kappa_{e^*})$ before Ng--Jordan--Weiss; and (iii)~a unified label-free evaluation against nine truly-unsupervised baselines. On five heterophilic benchmarks (Cora, Cornell, Texas, Wisconsin, Chameleon), CGSD wins outright on Wisconsin and Chameleon and is competitive on the remaining three against nine unsupervised baselines. The gain over the strongest baseline is driven by the clusterer, not the encoder: on the same embedding, CSpec improves mean NMI from $0.091$ with $K$-Means to $0.107$ ($+15\%$, paired $t$-test $p=0.008$). The mechanism is interpretable: intra-community and inter-community curvature distributions are visibly separated. Code is open-sourced at this https URL. 

---
# Efficient RGB-T Object Detection via Sparse Cross-Modality Fusion 

**Authors**: Chao Tian, Zikun Zhou, Chao Yang, Guoqing Zhu, Zhenyu He  

**Link**: [PDF](https://arxiv.org/pdf/2606.30215)  

**Abstract**: RGB-T detectors leverage the complementary strengths of visible and thermal infrared modalities, achieving robust performance under challenging conditions. Many of them resort to heavy dual backbones and exhaustive cross-modality fusion across the entire image, leading to impractically high computational costs. We observe that most image regions are smooth backgrounds (e.g., sky, ground) that can be easily handled by lightweight single-modality models. In light of this observation, we propose a sparse fusion mechanism for efficient RGB-T detection: first rapidly scanning the image to identify the proposals and then carefully examining the remaining sparse proposals via feature fusion. We propose a two-stage framework to instantiate this mechanism, which performs detection in two stages: 1) a lightweight and modality-specific detection stage that produces high-recall RoIs, and 2) a fusion-driven examination and refinement stage that filters out the false positives and refines the bounding boxes. This design enables the detector to adaptively allocate more computational resources to the potential foregrounds, improving the efficiency while ensuring detection accuracy. Extensive experiments show that our method achieves competitive performance with substantially fewer parameters and lower cost, while maintaining strong scalability to high-resolution images. 

---
# A Multi Center Breast FNAC Whole-Slide Cytology Dataset for AI-Assisted Patch-Wise Classification Using C1 to C5 Reporting Categories 

**Authors**: Garima Jain, Abhijeet Patil, Surabhi Jain, Sanghamitra Pati, Amit Sethi, Sandeep Mathur, Pulkit Verma, Nishi Halduniya, Jatin Kashyap, Sharat Kumar, Simmi Kharb, Sunita Singh, Sucheta Devi Khuraijam, Sushma Khuraijam, Ratan Konjengbam, Arvind Kumar, Deepali Tirkey, Saurav Banerjee, Shivani Kalhan, Rakesh Kumar Gupta, Ranjana Solanki, Deepika Hemranjani, Shashank Nath Singh, Uma Handa, Manveen Kaur, B. G. Malathi, Yogender P., Niraj Kumari, Shruti Gupta, Indu R. Nair, Vidya C., Basumitra Das, Sunil Kumar Komanapalli, Ravindra Karle, Tanaya Kulkarni, Vandana Raphael, Biswajit Dey, Vaishali Gaikwad, Nilam Adhav  

**Link**: [PDF](https://arxiv.org/pdf/2606.30209)  

**Abstract**: We present a multi center breast fine needle aspiration cytology (FNAC) dataset designed for patch wise classification using C1 to C5 reporting labels. The prospective dataset includes 321 patients and 470 whole-slide images (WSIs) collected from participating tertiary medical centers in India between May 2023 and March 2026. Slides were stained using Papanicolaou (190 WSIs) or MayGrunwald Giemsa (280 WSIs), scanned on a Hamamatsu NanoZoomer S360 at 40X magnification and 0.25 microns per pixel, and stored directly in NDPI format. Across the 470 WSIs, 446 WSIs contain annotated patch regions, yielding 7,398 PNG image patches with expert-verified C1 to C5 labels. The release includes NDPI WSIs, WSI-level GeoJSON annotation files, extracted patch images, deidentified metadata, a data dictionary, a validation summary, a manifest linking WSIs to Zenodo records, and code for dataset inspection and reuse. The complete dataset is approximately 950 GB and is available through Zenodo. 

---
# Forewarned is Forearmed: When Non-Sequential Embedding Turns Into an Anomaly Detector 

**Authors**: Elys Allesiardo, Antoine Caubrière, Valentin Vielzeuf  

**Link**: [PDF](https://arxiv.org/pdf/2606.30196)  

**Abstract**: This paper offers an in-depth analysis of non-sequential multimodal sentence-level embeddings, with a particular focus on the SONAR model. We demonstrate that certain embedding dimensions are sensitive to perturbations and can serve as indicators of decoding anomalies. By leveraging the consistency between successive encoding and decoding, we successfully build an accurate detector. Additionally, we explore modifying specific dimensions of interest to attempt to correct them. This work underscores the importance of understanding and analyzing the embeddings themselves to enhance the reliability of multimodal representations. 

---
# Few-Shot Domain Incremental Learning via Continual Vision-Language Consolidation 

**Authors**: Naeem Paeedeh, Mahardhika Pratama, Wolfgang Mayer, Mukesh Prasad, Weiping Ding, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2606.30190)  

**Abstract**: Existing domain-incremental learning (DIL) strategies call for massive amounts of data to adapt to new domains and suffer from the overfitting problem in the case of data scarcity. This paper puts forward a relatively uncharted problem, namely, few-shot domain incremental learning (FSDIL), taking into account the problem of extreme data shortages in the realm of DIL. A novel algorithm, namely Continual Vision-Language Consolidation (CVLC), is proposed to address the FSDIL problem, where the key idea lies in the concept of latent space reservation in the base domain coupled with dual coalescent projection (DCP) as a parameter-efficient fine-tuning method. First, the vision prototype is calibrated while multiple templates and synonyms are generated via LLMs to induce the language prototype. The vision and language prototypes are fused. Adaptation to never-ending arrivals of new domains is done by the DCP technique, fine-tuned in such a way to prepare the model to unseen domains via latent-space reservations committed in the base domain. CVLC is structured under shared and domain-specific components to combine general knowledge and domain-specific details. The advantage of our approach is demonstrated through a range of benchmark problems and comparisons with prior arts, in which CVLC outperforms them by up to a 16% gap. Our codes are shared publicly in this https URL . 

---
# Beyond Drug Discovery: The Nanotechnology Molecular Optimization (NMO) Benchmark 

**Authors**: Matthias Blaschke, Daniel Kienzle, Zsuzsanna Koczor-Benda, Julian Lorenz, Rainer Lienhart, Fabian Pauly  

**Link**: [PDF](https://arxiv.org/pdf/2606.30170)  

**Abstract**: Generative molecular design is shaped by simple proxy benchmarks for drug-like properties and models pretrained on large pharmaceutical datasets. This combination yields strong benchmark metrics but limits transferability to domains structurally distinct from drug discovery. To overcome this limitation and drive discovery toward real, scientifically grounded targets, we introduce the Nanotechnology Molecular Optimization (NMO) Benchmark, which bridges machine learning (ML) and quantum materials science. NMO acts simultaneously as a rigorous testbed for the ML community and a discovery engine for nanotechnology research. The suite replaces proxy oracles with quantum simulations and introduces strict protocols that prioritize scientific utility over leaderboard-oriented overfitting. The physics-based NMO tasks impose hard structural constraints and rugged fitness landscapes, posing fundamentally new requirements on generative models. Notably, advanced molecular optimization methods underperform much simpler approaches on the NMO tasks. We develop a new baseline method identifying the critical components to solve the NMO tasks, including a novel representation for modeling structural constraints and a domain-agnostic pretraining strategy to eliminate pharmaceutical dataset bias. Our results surpass state-of-the-art physical properties and reveal previously unknown structural motifs, offering new insights for the nanotechnology community and demonstrating that ML can drive genuine scientific discovery. 

---
# Federated Learning with Energy-Based Structured Probabilistic Inference 

**Authors**: Dario Fenoglio, Daniil Kirilenko, Martin Gjoreski, Marc Langheinrich  

**Link**: [PDF](https://arxiv.org/pdf/2606.30161)  

**Abstract**: Federated learning typically aggregates client updates using fixed or heuristic weighting rules, which can be suboptimal when clients have heterogeneous data and varying contributions to the global model. We propose a framework that refines client aggregation weights using Conditional Random Fields (CRFs). Our method defines unary potentials for individual clients and pairwise potentials for all client pairs, allowing the server to model both client-specific reliability and interactions between clients. The resulting CRF inference produces aggregation weights that enable better convergence of the global training objective. Experiments show that, under non-IID heterogeneity, our approach consistently improves performance over well-established federated learning baselines. 

---
# Physically-Constrained Harmonic Separation for Robust Heart and Respiratory Rate Estimation from Wrist Photoplethysmography 

**Authors**: Nouhaila Fraihi, Ouassim Karrakchou, Mounir Ghogho  

**Link**: [PDF](https://arxiv.org/pdf/2606.30156)  

**Abstract**: Wrist-worn photoplethysmography (PPG) enables continuous monitoring of cardiopulmonary physiology, but reliable heart rate (HR) and respiratory rate (RR) estimation in free-living conditions remains challenging due to non-stationary motion artifacts that spectrally overlap with physiological dynamics. Existing signal-processing methods degrade under strong motion, while unconstrained deep learning approaches often lack physiological interpretability and identifiable structure. We propose a Physically-Constrained Harmonic Separation (PCHS) framework that formulates HR and RR estimation from wrist PPG as an analysis-by-synthesis problem, where accelerometer measurements condition artifact separation rather than directly regressing vital signs. A physics-guided harmonic generator decomposes the observed signal into quasi-periodic physiological components and a motion-related residual, enabling HR recovery from the fundamental frequency and RR prediction from respiratory-driven modulations of the harmonic parameters. Robust reconstruction objectives, separation constraints, and uncertainty-aware weighting stabilize the decomposition under motion. Experiments on the motion-intensive PPG-DaLiA dataset demonstrate that PCHS outperforms state-of-the-art methods while yielding interpretable signal decompositions that effectively disentangle physiological activity from motion artifacts. 

---
# Estimating Grammatical Gender Directions in Contextual Embeddings under Controlled and Natural Contexts 

**Authors**: Huanping Xiao, Yingji Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.30152)  

**Abstract**: Contextual language models conflate grammatical gender and social semantic bias in gendered languages such as Spanish. Existing gender debiasing approaches only operate on static word embeddings leaving contextual representations unexplored for this two dimensional gender disentanglement. To address the this issue, we make the first attempt to disentangle grammatical gender from semantic contamination for contextual embeddings. We construct both controlled templates and natural Wikipedia contexts to build balanced datasets of inanimate nouns, and design a framework equipped with centroid, Support Vector Machine (SVM) and Linear Discriminant Analysis (LDA) gender direction estimators as well as contamination-aware weighting strategies. A set of dual-objective evaluation metrics is proposed to balance the suppression of grammatical gender leakage on inanimate nouns and the preservation of semantic gender distinctions for occupation terms. The results reveal that unweighted controlled contexts yield the purest grammatical gender direction, and the centroid estimator achieves better performance than discriminative baselines. 

---
# Query-Aware Spreading Activation for Multi-Hop Retrieval over Knowledge Graphs 

**Authors**: Illia Makarov, Mykola Glybovets  

**Link**: [PDF](https://arxiv.org/pdf/2606.30133)  

**Abstract**: Retrieval-augmented generation built on knowledge graphs (Graph RAG) outperforms flat passage retrieval on multi-hop question answering by leveraging graph structure. In most existing systems, however, the question only sets the seed nodes; the subsequent traversal becomes "query-blind", depending solely on the graph structure. The exception is QAFD-RAG, which implements query-aware traversal via a flow-diffusion solver with combined edge re-weighting. This architecture requires loading the full graph into Python memory and an iterative solver with a variable number of iterations complicating integration with the graph database. We propose a spreading-activation method that achieves the same query-aware traversal with a single per-step semantic gate: the step weight is the cosine similarity between the candidate entity's description and the question, and the number of iterations is fixed. The whole retrieval procedure - seed mapping, propagation, top-K selection and context assembly - is expressed as a single Cypher query executed in one round-trip to Neo4j; the graph never leaves the database. On MuSiQue our method matches QAFD-RAG by exact match (32.80 vs 33.50) and outperforms the strongest purely-structural baseline in our comparison, HippoRAG, by 5.3 EM and 3.4 F1; on 2WikiMultiHopQA HippoRAG and QAFD-RAG retain an advantage due to their phrase-node architectures. An ablation with the gate disabled confirms that the gate is the source of a simultaneous F1 gain of 3.6 to 7.4 points and a retrieval-latency reduction by a factor of 1.5 to 4.9. 

---
# Hyper-Network Neural Functional Maps for Unsupervised Robust 3D Shape Matching 

**Authors**: Dongliang Cao, Florian Bernard  

**Link**: [PDF](https://arxiv.org/pdf/2606.30131)  

**Abstract**: Functional maps are the cornerstone of recent non-rigid 3D shape matching methods due to their efficiency and performance. However, existing methods struggle with challenging scenarios, such as partiality, topological noise, and raw point clouds. A primary bottleneck is that significant intrinsic distortion prevents truncated spectral bases from being accurately aligned via linear transformations (i.e., functional maps). To address this, we introduce a hyper-network that predicts non-linear neural functional maps (NFM), learned in an unsupervised manner, to better align spectral bases. Specifically, we model the NFM as an MLP with skip-connection to refine standard FM and employ a hyper-network to predict its weights, conditioned on standard FM. Our framework is trained using a novel unsupervised spectral alignment loss. Experiments demonstrate that our approach can be seamlessly integrated into state-of-the-art unsupervised deep functional map pipelines, substantially improving matching accuracy in demanding scenarios. 

---
# Gravitational Duals from Equations of State II: Large Hierarchies and False Vacua 

**Authors**: Raul Jimenez, David Mateos, Pavlos Protopapas, Pau Solé-Vilaró, Pedro Tarancón-Álvarez, Pablo Tejerina-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2606.30117)  

**Abstract**: We investigate the reconstruction of holographic duals for strongly coupled quantum field theories in regimes characterized by large hierarchies and the presence of false vacua. Within the gauge/gravity duality, these features translate into non-trivial thermodynamic behaviour and exotic renormalization group flows, including skipping flows between non-adjacent fixed points. Building on previous work based on Physics-Informed Neural Networks (PINNs), we extend the holographic inverse problem of reconstructing the bulk scalar potential from boundary thermodynamic data into this new regime. This setting presents a variety of conceptual and numerical challenges, such as near-degenerate states, large hierarchies of energy scales, and regions of the potential that are not directly probed by the input data. We develop a set of methodological advances that overcome these obstacles, thereby improving the established PINNs-based methodology and extending it to new physical regimes of interest that were previously out of reach. Applying the developed framework, we demonstrate accurate reconstruction of scalar potentials deep into the false vacuum regime, achieving robust agreement with the physical features of the underlying thermodynamics despite significant numerical stiffness. Our results extend the bridge between holography and machine learning, and suggest that data-driven approaches can provide new insights into the structure of strongly coupled systems. 

---
# SA-VLA: State-aware tokenizer for improving Vision-Language-Action Models' performance 

**Authors**: Tengyue Jiang, Chunpu Xu, Jiayue Kang, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2606.30113)  

**Abstract**: Discrete action tokenization provides a compact interface for autoregressive VLA policies, but accurately recovering continuous robot actions from discrete codes remains challenging. Existing tokenizers typically map each discrete code to a fixed continuous action prototype, ignoring the robot's current proprioceptive state. This limitation is particularly pronounced in manipulation, where the same action token may require different continuous controls under different joint configurations, object poses, and contact conditions. We therefore propose SA-VLA, a state-aware action tokenizer that conditions action decoding on robot state. We study two state-injection mechanisms for VQ-based action tokenization: cross-attention between state and action features, and a lightweight state adapter that predicts action-wise modulation factors for state-conditioned action modulation and reconstruction. The adapter formulation expands the effective support of a finite codebook by allowing each discrete token to represent a family of state-dependent continuous actions, while preserving the efficiency and compatibility of discrete action modeling. Integrated into an LLM-based VLA policy, SA-VLA supports both autoregressive and parallel action-token decoding with minimal changes to the model interface. On 12 RoboTwin manipulation tasks, SA-VLA improves the average success rate from 0.29 to 0.56 over the strongest tokenizer baseline. In zero-shot sim-to-real experiments on three real-world tasks, it further improves average success from 0.15 to 0.33 over the strongest tokenizer baseline. These results demonstrate that state-conditioned action decoding is a simple and effective mechanism for reducing the compression gap in discrete VLA policies. 

---
# Automating the Design of Embodied AgentArchitectures 

**Authors**: Jian Zhou, Sihao Lin, Jin Li, Shuai Fu, Gengze Zhou, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.30111)  

**Abstract**: Embodied agents are typically built as hand-designed compositions of perception, memory, planning, and action modules. This modularity exposes a large architectural design space, but current systems still rely on researcher intuition to choose where information is stored, how observations are processed, and how model calls are connected. Agent Architecture Search (AAS) automates such design for text-domain agents, but has not been systematically evaluated on perceptual embodied agents through simulator rollouts. We study this transfer. We introduce AgentCanvas, a typed-graph runtime that hosts embodied executors as editable node-and-wire programs with simulator-aware execution and episode-level logs, and KDLoop, a coding-agent search procedure that cycles through proposal, critique, experiment, and distillation, with triggered reflection after stalls. We evaluate three AAS variants across four embodied executors spanning vision-language navigation, embodied question answering, and language-conditioned manipulation. The resulting 3x4 matrix shows that architecture-level search can produce deployable and directional success-rate gains on embodied tasks, while one apparent high-scoring candidate is rejected as leak-bearing. At the same time, the experiments expose constraints that are muted in text-domain AAS: optimization signals can be masked by rollout noise, search can become trapped in local edit basins, and episode-level credit assignment only partially emerges even when detailed logs are available. These results characterize both the promise and the current limits of automated architecture search for embodied agents. 

---
# Online Data Selection for Instruction Tuning via Gaussian Processes 

**Authors**: Jun Wang, Quoc Phong Nguyen, Julien Monteil, Vu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2606.30077)  

**Abstract**: With Large Language Model (LLM) pre-training and fine-tuning shifting its focus from data volume to data quality, quality data selection has emerged as a critical research topic. Existing online data selection methods for LLM training are typically "batch-constrained", limiting optimization to local utility within random batches. To overcome this, we propose GAIA (Global Adaptive Instruction tuning via GAussian processes), a framework that formulates data valuation as a global estimation process. GAIA employs Gaussian Process regression to model continuous utility manifolds across the semantic space, utilizing an adaptive strategy fusion mechanism to dynamically prioritize high-utility samples. By casting the strategy-posterior update as an instance of the classical fixed-share Hedge framework for tracking the best expert, we inherit a dynamic-regret guarantee that characterizes GAIA's robustness under non-stationary quality scores during training. Empirical evaluations on three datasets demonstrate that GAIA significantly outperforms state-of-the-art baselines like \greats, establishing our method as a scalable and robust solution for efficient instruction tuning. 

---
# Neural Subspace Reallocation: Continual Learning as Retrieval-Based Subspace Memory Management 

**Authors**: Byeong Hoon Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2606.30067)  

**Abstract**: We introduce Neural Subspace Reallocation (NSR), which reframes continual learning as memory management over parameter subspaces. Instead of treating Low-Rank Adaptation (LoRA) modules as disposable per-task adapters, NSR manages them as compressible, retrievable memory units on a frozen backbone through a recurring cycle: (1) compress learned LoRAs via SVD, (2) reserve them in a TaskKnowledgeBank, (3) recall related past LoRAs by embedding similarity to warm-start new or returning tasks, and (4) reallocate the active subspace accordingly, with distillation protecting prior tasks. We prove that in cyclic environments any memoryless allocation policy incurs cumulative regret Omega(T(M-1)Delta_switch) relative to a history-aware policy backed by the Bank (Theorem 1). Empirically, on Split-CIFAR-100 the Bank reduces cyclic recovery time by 10x, exactly as predicted, and on the heterogeneous 5-Datasets benchmark NSR achieves the highest accuracy and the least forgetting, about 9x closer to zero backward transfer than the memoryless heuristics. Crucially, we run a controlled study that isolates which component matters: holding the Bank fixed and varying only the allocation rule, we find that a simple similarity-based retrieval rule matches or beats a learned reinforcement-learning controller (recovering recurring tasks in 0 vs 1.8 steps and reaching equal accuracy). Our central, honest finding is therefore that the memory mechanism -- compression and similarity retrieval -- rather than a learned allocation policy, drives continual-learning performance under fixed capacity. A memory-budget analysis confirms the compressed Bank stays small -- 0.29 MB of parameter memory per task -- so a top-K retention cap bounds the total footprint while preserving fast recovery for retained tasks. 

---
# Little Brains, Big Feats: Exploring Compact Language Models 

**Authors**: Dari Baturova, Elena Bruches, Ivan Chernov, Roman Derunets, Arsenii Fomin, Andrey Kostin  

**Link**: [PDF](https://arxiv.org/pdf/2606.30062)  

**Abstract**: While large language models have been dominating the research landscape recently, small language models remain highly relevant across various domains; yet, they receive far less attention. In this study, we investigate how smaller language models perform during the generation stage within a Retrieval-Augmented Generation (RAG) system. To benchmark these models effectively, we utilised both open-source and proprietary datasets covering diverse subject areas and question types. Our findings demonstrate that a RAG system with small language models can be executed directly on-device without requiring any GPU hardware within a reasonable time. The experimental code and links to the supplementary materials can be accessed through the GitHub repository: this https URL. 

---
# MuseBench: Benchmarking Intent-Level Audiovisual Arts Understanding in MLLMs 

**Authors**: Yuxuan Fan, Gyusik Seo, Jing Hao, Jaemin Cho, Mohit Bansal, Jaehong Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2606.30026)  

**Abstract**: Audiovisual arts encompass diverse creative disciplines, including cinema, visual arts, stage performance, and game design, where artistic meaning arises from deliberate combinations of visual, auditory, and narrative elements (e.g., fear amplified through claustrophobic framing, or grief conveyed through silence and lingering close-ups). True artistic understanding extends beyond recognizing what is depicted to reasoning about why it is expressed through particular creative choices. Despite the strong progress of multimodal large language models (MLLMs), this critical aspect of artistic understanding remains underexplored, as existing benchmarks largely measure perceptual recognition while overlooking reasoning about creative intent. To address this gap, we introduce Musebench, a comprehensive benchmark designed to evaluate MLLMs on nuanced artistic understanding. It comprises 4,016 questions spanning cinematic arts, static visual arts, stage performing arts, and game arts, distilled from over 10K candidate video essays that pair professional commentary with visual demonstration. To capture the open-ended nature of artistic analysis at scale, the benchmark combines single-select and variable-option multi-select questions. All questions are generated and refined through a four-phase iterative pipeline combining shortcut filtering, adversarial distractors, and expert validation. Comprehensive zero-shot evaluation of 28 state-of-the-art MLLMs reveals that even the best-performing model achieves only 48.29% accuracy, substantially below human expert performance of 87.18%, exposing a significant gap in current models' creative domain expertise. 

---
# IBRSteG: Learning a Generalizable Steganography Framework for 3D Gaussian Splatting 

**Authors**: Fanye Kong, Hongyu Xia, Yu Zheng, Boyang Gong, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.30024)  

**Abstract**: Recent advances in deep learning have notably improved steganographic message hiding. However, designing a generalizable steganographic approach for 3D Gaussian Splatting (3DGS) that can embed meaningful 3D scene content remains challenging. In this paper, we propose IBRSteG, a generalizable framework for 3DGS steganography that enables undetectable concealment of secret scenes within a steganographic scene. Unlike existing approaches whose parameter generation is rigidly coupled with the specific scene, we formulate 3D steganography as a feed-forward 3D Gaussian embedding process that generalizes across different 3DGS scenes. To realize this, we introduce GAS (Gaussian Attributes Steganographer), a network that learns a scene-independent embedding function by injecting the attributes of secret 3D Gaussian points into a cover scene, thereby directly reconstructing the steganographic scenes without per-scene finetuning or optimization. By transforming 3D Gaussian into these structured attributes, these attributes are compatible with 2D learning paradigms and benefit from their structured nature, thereby enhancing generalization to unseen 3DGS scenes. Extensive experiments on established datasets demonstrate that IBRSteG can effectively conceal different scenes with high visual quality, and achieves superior capacity and security. Code is available at this https URL. 

---
# T3R: Deeper Test-Time Adaptation for Graph Neural Networks via Gradient Rotation 

**Authors**: Huy Truong, Alexander Lazovik, Victoria Degeler  

**Link**: [PDF](https://arxiv.org/pdf/2606.30011)  

**Abstract**: Graph Neural Networks (GNNs) deployed in real-world systems typically have fixed weights, often leading to degraded performance under distribution shifts. This issue can be mitigated by conventional fine-tuning, but in many real-world cases, collecting labeled data is expensive or infeasible. A potential approach is Test-Time Training (TTT), which adapts models' weights using unlabeled test data, yet it is typically limited to shallow updates that affect only a subset of model parameters. We propose T3R, leveraging multiple Rotograd matrices to improve task affinity between the target and auxiliary tasks, essential for effective test-time training. T3R further introduces a rotation technique that reorients self-supervised signals using these matrices to create surrogate gradients for the target task, allowing deeper adaptation across nearly the entire architecture. Empirically, T3R reduces MAE by 0.172 points over standard inference in regression datasets and achieves at least 9.37% relative improvement on cross-domain OGB classification benchmarks compared to models without adaptation. These results highlight the potential to develop an adaptation pipeline for graph-based systems, particularly in settings where conventional fine-tuning or retraining is infeasible. 

---
# Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping 

**Authors**: Hsun-Yu Kuo, El Mahdi Chayti, Patrik Reizinger, Wieland Brendel, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2606.29983)  

**Abstract**: Looped Transformers, which repeatedly apply a shared transformer block, are an architecturally natural fit for variable-length algorithmic tasks. Although they can exhibit strong length generalization beyond the length of training sequences, this behavior is brittle, yielding high out-of-distribution (OOD) variance, even across well-performing in-distribution solutions. We trace this variance to the spurious correlation in simple algorithmic tasks between sequence length and number of loops. Introducing stochasticity into the number of loops during training sharply reduces OOD variance and stabilizes predictions across inference-time loop counts. To improve upon heuristic randomization schemes, we further analyze RL-Halting as a learned stochastic schedule and find that it generally improves the accuracy-stability trade-off. Across binary addition, Dyck-1, Unique Set, and Copy, learned stochastic stopping often improves this trade-off but can also stabilize a suboptimal computation. Our work suggests that "when to stop" should be treated as a training-time design choice, not merely an inference-time computation-allocation rule. 

---
# RiverONE: Generating Knowledge-Intensive VLM by Simulated Quantum Machines 

**Authors**: Xindian Ma, Xinyu Long, Yefei Zhang, Yanchen Liu, Xianghao Li, Yufu Wen, Yike Hu, Yuedong Zhu, Zeyang Ma, Wen Qin, Yikun Wang, Peng Yang, Monan Wang, Teng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29966)  

**Abstract**: Quantum computing provides a powerful paradigm for representing and transforming high-dimensional information through superposition, entanglement, and measurement-induced nonlinear features. While current quantum hardware is not yet practical for direct large-scale vision-language model (VLM) inference, simulated quantum computation can be used during model construction to generate structured parameters for compact classical AI systems. We build RiverONE, a lightweight vision-language model for quantum calibration plot understanding, using simulated quantum computation. It employs a specialized visual encoder and an InternVL-based language backbone. To compensate for compression-induced information loss, we introduce quantum-generated parameters, which are materialized as classical tensors after training. This allows RiverONE to run entirely on classical GPUs at inference time, with no quantum hardware or runtime quantum simulation. With approximately 1.9 billion parameters, RiverONE achieves at least 95\% of the performance of NVIDIA Ising Calibration 1 on quantum calibration plot understanding tasks while using less than 10\% of its parameter count. These results suggest that simulated quantum computation can serve as a practical construction-stage mechanism for building lightweight, knowledge-intensive scientific VLMs. Our code is available at this https URL. 

---
# DuoMem: Towards Capable On-Device Memory Agents via Dual-Space Distillation 

**Authors**: Peyman Hosseini, Ondrej Bohdal, Ahmed Alajrami, Andrea Maracani, Ignacio Castro, Matthew Purver, Mete Ozay, Savas Ozkan, Taha Ceritli  

**Link**: [PDF](https://arxiv.org/pdf/2606.29961)  

**Abstract**: Large Language Model (LLM)-based agents can solve complex procedural tasks by interacting with environments over multiple turns, but this ability typically depends on large models, long contexts, and repeated inference calls. This makes advanced memory-augmented agents difficult to deploy on resource-constrained devices. We introduce DuoMem, a dual-space distillation framework that transfers procedural problem-solving ability from a large teacher model to compact student models. DuoMem distils in two complementary spaces: (1)context-space distillation, which replaces student-generated memories with higher-quality teacher-generated procedural memories prepended to the student's input, and (2)parameter-space distillation, which fine-tunes lightweight LoRA adapters on successful teacher trajectories. Evaluated on ALFWorld, a challenging embodied decision-making benchmark, DuoMem boosts a 4B-parameter model from 4.3% to 77.9% task success rate, closing most of the gap to a 72B teacher model (87.1%), while adding fewer than 10M trainable parameters and only a few megabytes of pre-computed teacher memories. Moreover, the DuoMem-enhanced 4B model completes tasks over 3x faster than the 72B teacher in wall-clock time, making it viable for real-time edge deployment, which would be challenging for the this http URL ablations across eight models spanning 2B-72B parameters reveal that both distillation axes contribute complementary 

---
# SWE-Together: Evaluating Coding Agents in Interactive User Sessions 

**Authors**: Yifan Wu, Zhuokai Zhao, Songlin Li, Ho Hin Lee, Jiacheng Zhu, Shirley Wu, Tianhe Yu, Serena Li, Lizhu Zhang, Xiangjun Fan, Shengzhi Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.29957)  

**Abstract**: Most coding-agent benchmarks are static: an agent receives a complete task description up front and is judged only by its final code. Real coding assistance is interactive, with users clarifying goals, adding constraints, and correcting mistakes over multiple turns. We introduce SWE-Together, a multi-turn benchmark reconstructed from real user-agent coding sessions. To make real interactions verifiable, we curate 109 repository-level tasks from 11,260 recorded sessions, selecting sessions with recoverable repository states, clear user goals, and observable outcomes. To replay these interactions across agents, we build a reactive LLM-based user simulator that preserves the original users' intents and provides feedback when the coding agent's progress requires it. To evaluate agents as collaborators, we measure both final repository correctness and the number of corrective feedback turns required during the interaction. Experiments with frontier coding agents show that stronger agents generally achieve higher final success rates while requiring fewer interventions, suggesting an improved user experience. 

---
# SpreadsheetBench 2: Evaluating Agents on End-to-End Business Spreadsheet Workflows 

**Authors**: Jian Zhu, Yuzheng Zhang, Zeyao Ma, Bohan Zhang, Armin Schoepf, Daniel Woloch, Peter Yiliu Wang, Guangyu Robert Yang, Samuel Jacob, Siddharth Nagisetty, Abhiram Chundru, Jean Lin, Spencer Mateega, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29955)  

**Abstract**: Spreadsheets are widely used for business analysis, financial modeling, reporting, and decision-making. However, most existing spreadsheet benchmarks evaluate isolated operations such as single-formula generation or local cell edits, and therefore fail to capture end-to-end workflows in realistic business settings. We introduce \textsc{SpreadsheetBench 2}, a workflow-level benchmark for spreadsheet agents that covers three task categories: generation, debugging, and visualization. The benchmark is constructed from authentic business data, including financial reports and corporate filings, and is annotated and validated by domain experts. The benchmark contains 321 tasks; each instance averages 11.8 worksheets and requires 593.5 cell modifications, reflecting large multi-sheet workbooks with cross-sheet dependencies. We evaluate eight frontier large language models under a unified multi-turn agent scaffold, and additionally include several LLM-based spreadsheet products as complementary baselines. Results show that current systems remain far from reliable on real-world workflows: the best model achieves 34.89\% overall task accuracy, and debugging accuracy is as low as 12.00\%. Trajectory analysis and a failure taxonomy further indicate that insufficient spreadsheet inspection and incorrect target-cell selection are the dominant bottlenecks. Together, these findings position \textsc{SpreadsheetBench 2} as a challenging testbed for advancing reliable spreadsheet automation. Project page: this https URL 

---
# Exploiting Local Flatness for Efficient Out-of-Distribution Detection 

**Authors**: Seonghwan Park, Hyunji Jung, Dongyeop Lee, Namhoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.29952)  

**Abstract**: Detecting out-of-distribution (OOD) data is crucial for reliable machine learning deployment. Among detection strategies, post-hoc methods are particularly attractive due to their efficiency, as they operate directly on pre-trained networks without requiring retraining. Within this paradigm, one promising direction exploits loss-landscape curvature to estimate model uncertainty; however, such methods incur substantial computational cost and rely on implicit assumptions about how landscape flatness differs between in-distribution (ID) and OOD data. In this work, we provide the first systematic investigation of this curvature discrepancy and show that OOD inputs exhibit larger Hessian curvature than ID data, with the gap widening under stronger distributional shifts. Motivated by these observations, we propose Fold, a lightweight flatness-modulated OOD detector that leverages the feature Hessian and partial feature normalization to improve ID-OOD separability while avoiding costly parameter-space curvature approximations. To optimally adapt this normalization across diverse datasets, we further introduce AutoFold, a self-supervised tuning scheme that synthesizes pseudo-OOD samples via ID logit masking for automatic calibration without requiring external data. Experiments on OOD benchmarks show that Fold outperforms prior methods, improving the average AUROC by 1.63% and reducing FPR95 by 2.30%, while maintaining computational efficiency comparable to a standard forward pass. Supported by theoretical analysis and extensive ablations, Fold provides a principled and practical solution for robust real-world deployment. 

---
# Data-Efficient Multimodal Alignment for Histopathology-based Molecular Prediction 

**Authors**: Dominik Winter, Dominik Vonficht, Loïc Le Bescond, Christian Gebbe, Marco Rosati, Richard J. Chen, Markus Schick, Ross Stewart, Nicolas Brieu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29949)  

**Abstract**: H&E-stained whole-slide images offer cohort-scale availability and rich spatial context but lack molecular specificity, whereas bulk RNA-seq provides transcriptome-wide resolution at high cost with limited archival availability. We show that training a lightweight alignment module atop frozen histopathology and RNA-Seq foundation models enables open-vocabulary molecular prompting -- querying H&E slides with gene-set signatures to predict pathway activity without sequencing or end-to-end retraining. Using contrastive learning on a multi-cancer cohort (N=1,720), we achieve a 25-fold improvement in retrieval over baseline methods. Systematic analysis reveals a graduated predictability spectrum: morphologically grounded programs (cell-cycle programs, immune-related) are most reliably predicted (R^2>0.5), while predicting pathways with no morphological footprint remains challenging as expected. We validate clinical utility on the POSEIDON clinical trial: H&E-predicted squamous cell carcinoma scores recapitulate NSCLC subtype identity and predicted IFN-gamma mirror PD-L1 tumor-cell expression groups. Furthermore, genesets describing immune activation and fibrosis predict known tumor microenvironment archetypes from histology alone. We further validate generalization of our approach across unseen cohorts and demonstrate data-efficient domain adaptation, establishing a slide-native framework for molecular analysis on H&E images. 

---
# Latent-CURE for Breast Cancer Diagnosis 

**Authors**: Weiyi Zhao, Xiaoyu Tan, Lu Gan, Liang Liu, Xihe Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29928)  

**Abstract**: Multimodal Large Models have significantly advanced automated breast ultrasound diagnosis. However, most existing frameworks utilize opaque, end-to-end paradigms prioritizing global statistical correlations over structured clinical reasoning. Consequently, these models remain susceptible to shortcut learning amid extreme real-world epidemiological imbalances, often bypassing rare but decisive malignant indicators for dominant benign patterns. To address this disconnect, we propose Latent-CURE, a novel diagnostic framework driven by asymmetric weighted chain-of-thought methodology grounded in latent space reasoning. Unlike traditional approaches, our framework constructs an implicit reasoning trajectory forcing the model to sequentially infer standardized BI-RADS morphological descriptors before converging on a final diagnosis. Furthermore, to combat the extreme scarcity of critical malignant features, we couple this architecture with a dual-asymmetric optimization strategy. By dynamically adjusting margins and weights, this strategy safeguards high-specificity malignant descriptors from being overshadowed by common benign priors. Comprehensive evaluations demonstrate that our knowledge-injected approach provides transparent clinical evidence while achieving robust, accurate diagnostic performance in imbalanced medical cohorts. 

---
# EVAF: A Test-Retest Protocol for Selective Parametric Consolidation 

**Authors**: Haoliang Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.29916)  

**Abstract**: Long-running language agents need mechanisms for deciding which experiences should persist after the working context is gone. Retrieval systems can reinsert past text, but they do not by themselves show that an experience has been selectively consolidated into the model's own behavior. We introduce EVAF, an Echo-Valence Attractor Field mechanism for gated LoRA consolidation, and a test-retest protocol for measuring selective parametric consolidation under controlled interference. Across GPT-2 and TinyLlama, EVAF preferentially consolidates high-valence, high-surprise experiences while preserving retrieval-accessible factual memory through a complementary routed memory path. Test-retest measurements show stronger post-interference behavioral persistence than frozen, retrieval-only, and ungated continual-update baselines, while keeping parameter drift and cross-persona contamination low. The results support a separation between memory access and memory depth: retrieving a fact and internalizing an experience are distinct computational operations. 

---
# Pondering the Way: Spatial-perceiving World Action Model for Embodied Navigation 

**Authors**: Hong Chen, Daqi Liu, Zehan Zhang, Haiguang Wang, Tianhao Lu, Longfei Yan, Haiyang Sun, Fangzhen Li, Hongwei Xie, Bing Wang, Guang Chen, Hangjun Ye, Yihua Tan  

**Link**: [PDF](https://arxiv.org/pdf/2606.29908)  

**Abstract**: Existing world model-based planners for visual navigation typically follow a verification-centric paradigm, decoupling goal intent from trajectory synthesis. This approach suffers from candidate dependence, heavy computational overhead, and inconsistencies between sampled actions and predicted visuals. To address these issues, we propose SWAM (Spatial-perceiving World Action Model), a task-centric joint observation-action generation framework. Given start and goal RGB observations, SWAM performs single-pass inference to simultaneously generate intermediate RGB-D sequences and corresponding action trajectories, promoting goal-consistent trajectory generation and improved spatial feasibility. While SWAM leverages depth pseudo-labels during training to internalize spatial priors, it requires only monocular RGB input at inference time. We further introduce a visual-guided action refinement module and a trajectory-scale regularization loss to enforce fine-grained alignment between motion and visual cues while stabilizing predictions across varying distances. Extensive experiments show that SWAM significantly outperforms state-of-the-art two-stage planners in success rate, trajectory accuracy, and inference efficiency, while demonstrating robust zero-shot generalization to unseen environments. 

---
# CW-B: Class Weighted Boosting Framework for Imbalance Resilient Multi Class Cardiac Phenotyping 

**Authors**: Sijia Li, Xiaoyu Tan, Chen Zhan, Yuanji Ma, Haoyu Wang, Xihe Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29907)  

**Abstract**: Cardiac discharge phenotyping informs post-discharge treatment and follow-up, but real-world records are often incomplete and class-imbalanced, increasing the risk of missed high-risk phenotypes. We propose CW-B, a clinical risk-aligned class-weighted XGBoost pipeline for five-class cardiac discharge phenotyping under real-world class imbalance and missingness. CW-B combines fold-specific class-balanced instance weighting, missingness-indicator augmentation, and classwise error auditing to improve recognition of clinically prioritized phenotypes while preserving interpretable and auditable decision logic. In five-fold stratified cross-validation, CW-B achieves the best Accuracy, Macro-F1, Balanced Accuracy, and Prioritized F1 among tree-based, ensemble, and neural baselines. Overall, CW-B provides a practical and deployment-oriented approach for more reliable cardiac discharge phenotyping in real-world clinical settings. 

---
# Semi-Supervised Sound Event Detection with Conditional Mixup and Embedding-Level Contrastive Loss 

**Authors**: Nian Shao, Xian Li, Xiaofei Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.29901)  

**Abstract**: Sound event detection (SED) is a core module for acoustic environmental analysis, yet its performance is often limited by scarce labeled data. Recent systems leverage large pretrained audio foundation models, but effective fine-tuning remains challenging because labeled data are limited while unlabeled data are abundant. A previous work, ATST-SED, addressed this problem with a pseudo-label based semi-supervised fine-tuning framework. In this work, we further improve the framework by adopting an embedding-level self-supervised contrastive loss inspired by ATST-Frame pretraining. This contrastive objective better exploits unlabeled data during fine-tuning. One challenge is that mixup serves different roles in the two objectives: pseudo-label learning uses composition mixup, while contrastive learning treats mixup as a perturbation. To resolve this mismatch, we propose conditional mixup, which combines composition mixup and perturbation mixup in one semi-supervised framework and defines the corresponding embedding-level contrastive losses. The resulting model achieves 0.645 PSDS1 and 0.822 PSDS2 on the DESED validation set, establishing a new state of the art. 

---
# LLM-based Multimodal Personality Recognition via Facial Action Unit-Text Semantic Fusion 

**Authors**: Tianyi Zhang, Wei Shan, Yuan Zong, Tianhua Qi, Wenming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.29900)  

**Abstract**: Personality recognition in asynchronous video interviews (AVIs) has become increasingly important due to their widespread adoption in modern recruitment. Existing approaches often rely on large language models (LLMs) to analyze textual responses of interviewees in AVI. However, unimodel methods often suffer from information loss (e.g., ignore facial cues). In contrast, multimodal methods that employ full-face images or sparsely sampled frames can discard fine-grained temporal dynamics critical for accurate personality assessment. To overcome these limitations, we propose an LLM-based framework that semantically fuse facial action units (AUs) with textual responses of AVI. AU sequences are first converted into interpretable textual descriptions, which are then fused with participants' textual responses through an LLM. A lightweight regression head transforms the resulting embeddings into continuous personality scores without disrupting the underlying semantic space. Experiments on the AVI-6 benchmark demonstrate consistent improvements over most baselines, with lower prediction errors and stronger correlations with human-rated scores across multiple traits. Further analysis reveals that AU-derived semantic representations offer complementary non-verbal cues to textual responses. Decoupling semantic understanding from regression prediction within the LLM also leads to greater training stability and clearer interpretability. Overall, these findings demonstrate that AU-text fusion provides a psychologically grounded and computationally efficient framework for personality recognition in AVIs. 

---
# Critical Interval MSE: Toward Reliable Offline Validation for Robot Manipulation Policies 

**Authors**: Haoxu Huang, Tongsam Zheng, Yifan Chen, Jiacheng You, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.29898)  

**Abstract**: Real-world evaluation is the gold standard for robot policies because it tests them against the physical conditions and deployment challenges they are ultimately designed to handle. However, real-world evaluation is also the bottleneck for iterating on robot policies: it is costly, difficult to reproduce, and often too sparse to reliably compare nearby model variants. A straightforward proxy for performance is validation loss on expert demonstrations, but this proxy is often poorly correlated with real-world performance. In this paper, we introduce Critical Interval MSE (CI-MSE), an intuitively simple yet effective offline validation metric. CI-MSE restricts error computation to task-critical segments and pairs it with simple action-alignment procedures that better match rollout-time behavior. Across simulation and real-world experiments, CI-MSE yields a stronger correlation between validation error and rollout performance than raw MSE. Across a wide range of policy checkpoints, CI-MSE achieves a Spearman's rank correlation of $-0.87$, much closer to the ideal value of $-1$ than raw MSE's $-0.61$, demonstrating a significant improvement. We show through sensitivity analysis that our metric is robust to a wide range of hyperparameters. We further study the effectiveness of CI-MSE under evaluation distribution shifts and suggest design boundaries when using this metric. In summary, this paper provides a simple and reliable offline validation tool for accelerating policy iteration. Project webpage: this https URL 

---
# Child-Centric Voice Anonymization in Single and Multi-Speaker Speech via Domain-Adapted SSL Models 

**Authors**: Pranav Tushar, Xiao Xiao Miao, Rong Tong  

**Link**: [PDF](https://arxiv.org/pdf/2606.29897)  

**Abstract**: Voice anonymization aims to protect speaker identity while preserving linguistic content and speech usability. However, most anonymization systems are developed on adult speech, leading to degraded performance when applied to child speech. This paper investigates child-centric anonymization by adapting a self-supervised learning (SSL) based anonymization pipeline to the child speech domain. The system is adapted using child speech from the MyST corpus and evaluated under both single-speaker and two-speaker mixture conditions. Experimental results show that child-domain adaptation improves intelligibility and perceptual quality while maintaining strong privacy protection. Extending the approach to multi-speaker further demonstrates that combining target speaker extraction with child-adapted anonymization provides privacy protection while preserving conversational structure. These findings highlight the importance of child-specific adaptation for practical speech anonymization systems. 

---
# SABER-Math: Automated Benchmark for Information Retrieval Evaluation in Mathematics 

**Authors**: Nikolay Georgiev, Maria Drencheva, Kseniia Ibragimova, Ivo Petrov, Dimitar I. Dimitrov, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2606.29894)  

**Abstract**: As agentic AI systems tackle more complex mathematical tasks, they increasingly rely on information retrieval (IR) to search problem databases, theorem libraries, and educational resources. However, choosing the right retriever remains difficult, as it is infeasible to directly isolate its effect on downstream performance. On the other hand, existing retrieval-specific benchmarks often fail to capture fine-grained mathematical relevance, penalizing relevant documents. We address this gap by introducing SABER-Math, the first fully automated benchmark for evaluating mathematical IR without expert annotation. Starting from 283K high-school-level math problems with solutions, SABER-Math builds challenging reranking tasks in three steps: (i) first, LLMs extract concise solution summaries and mathematical topics for each problem; (ii) then, per-query relevant documents are discovered using ontology topic-based and lexical solutions-summary-based similarities, and (iii) finally, a Swiss-style LLM preference tournament produces fine-grained relevance ratings for the documents. We evaluate lexical retrievers, specialized mathematical retrieval systems, and recent embedding models. We find that while modern embedding models substantially outperform classical and math-specific baselines, even the strongest systems struggle in symbol-heavy domains like Algebra and Calculus. Importantly, we show that general-purpose IR benchmarks such as MTEB do not reliably predict mathematical performance, especially for recent embedding models, highlighting the need for math-specific retrieval benchmarks. 

---
# Trust Your Instincts: Confidence-Driven Test-Time RL for Vision-Language-Action Models 

**Authors**: Siyao Chen, Jiakang Yuan, Jiaxin Wang, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.29892)  

**Abstract**: Reinforcement learning (RL) has become indispensable for pushing Vision-Language-Action Models (VLAs) beyond static imitation learning. However, existing RL methods typically require external environmental feedback, relying on predefined success signals to guide policy updates. In this work, we show that VLA models possess useful internal evaluative capabilities: in discrete-action VLAs, trajectories with higher generation confidence are significantly more likely to succeed. Based on this observation, we introduce T^2VLA (Test-time VLA), an architecture-agnostic test-time RL framework that enables VLA models to achieve self-bootstrapping policy improvement. Instead of relying on external rewards, T^2VLA leverages trajectory-level similarity to high-confidence expert demonstrations as an intrinsic reward signal. In addition, we propose a Confidence-Driven Dual Expert Bootstrapping mechanism, which dynamically balances a Local Pseudo-Expert for exploration and a Global Expert Pool for training stability. Extensive experiments on the LIBERO and RoboTwin benchmarks show that T^2VLA consistently outperforms supervised baselines and approaches oracle RL performance with ground-truth rewards, achieving effective improvement without external reward feedback. Furthermore, T^2VLA adapts to distinct VLA paradigms, including both OpenVLA-OFT and the pi series. 

---
# LWDrive: Layer-Wise World-Model-Guided Vision-Language Model Planning for Autonomous Driving 

**Authors**: Chen Yang, Yuhao Wei, Ze Xu, Ziheng Zou, Shuang Liang, Delin Ouyang, Lingfeng Qi, Jie Li, Guofa Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.29879)  

**Abstract**: Vision-Language Models (VLMs) provide powerful semantic understanding and commonsense reasoning for End-to-End Autonomous Driving (E2E-AD) planning. However, trajectories directly generated by VLMs often encode only coarse driving intentions and remain insufficient for geometrically accurate, future-aware, and multi-view-grounded planning. To address these limitations, we develop the Layer-Wise World-Model-Guided Driving framework (LWDrive). LWDrive is a VLM planning framework that refines coarse trajectories through layer-wise world-model guidance. Instead of treating the VLM output as the final trajectory, LWDrive uses it as an intent-aware coarse plan, expands a diverse candidate space around it, and progressively refines the candidates through a Foresight Cascade Planner (FCP). Specifically, we introduce future-frame generation supervision to encourage the VLM to learn forward-looking scene representations, thereby injecting planning-relevant predictive dynamics into its internal hidden states. Built upon these world-model-supervised representations, FCP exploits VLM features across multiple layers and integrates historical temporal states, Action-Query representations, and current-frame multi-view Bird's-Eye-View (BEV) features to refine candidate trajectories in a coarse-to-fine manner. This design enables progressive correction of spatial positions and motion trends while grounding trajectory refinement with multi-view scene cues and preserving the high-level driving intention produced by the large model. Finally, a score head evaluates the refined candidates and selects the best trajectory as the final planning output. Experiments show that LWDrive achieves a score of 92.0 on the NAVSIM benchmark and 89.6 on NAVSIM-v2. Code and models will be made publicly available. 

---
# Clinical Reasoning Graphs: Structured Evaluation of LLM Diagnostic Reasoning Reveals Competence Without Consistency 

**Authors**: Nisarg A. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2606.29876)  

**Abstract**: Modern large language models (LLMs) reach 60-70% diagnostic accuracy on complex clinical case benchmarks, but accuracy alone cannot distinguish stable clinically-grounded reasoning from pattern matching. We introduce clinical reasoning graphs, structured graph representations extracted from free-text LLM diagnostic traces using a domain-grounded ontology with 5 node types and 7 edge types. We apply this pipeline to 750 traces from five LLMs across 50 New England Journal of Medicine Clinicopathological Conference cases and three prompt conditions, and test whether diagnostic traces show stable structured reasoning patterns, or diagnostic schemas, for clinically similar cases. We operationalize this as higher graph similarity among clinically similar cases than among clinically dissimilar ones. Across 15 model-condition comparisons, within-cluster and between-cluster composite similarity are nearly equal, and no comparison survives multiple-testing correction; a component-level analysis finds any residual content signal far below schema scale. Graph similarity is also nearly identical for pairs of models that are both correct (0.488) and both incorrect (0.484), suggesting that graph structure captures a dimension not reflected in diagnostic accuracy. Structured reflection prompting increases explicit discriminating-feature analysis within traces (+33%) but does not increase cross-case consistency. These results show diagnostic competence without schema-scale reasoning consistency, and indicate that final-answer accuracy should be complemented by process-level evaluation. We release the ontology, extraction pipeline, validation protocol, and the extracted reasoning graphs and similarity artifacts as resources for structured evaluation of LLM clinical reasoning. 

---
# ARKD: Adaptive Reinforcement Learning-Guided Bidirectional KL Divergence Distillation for Text Generation 

**Authors**: Zilong Liu, Xuewen Zhang, Jinrui Xing, Juyi Qiao, Huiyong Wang, Junming Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.29869)  

**Abstract**: Knowledge distillation (KD) is a key technique for compressing Large Language Models (LLMs), yet methods relying on a single KL objective often fail to balance primary distribution fitting with long-tail probability modeling, limiting both generation quality and generalization. To address this, we analyze the complementary roles of forward and reverse KL divergence (FKL/RKL) in distribution alignment from theoretical and empirical perspectives. We then propose a reinforcement-learning-based adaptive KL-weighted distillation framework, in which a policy network dynamically assigns weights to FKL and RKL based on teacher-student distributional characteristics, guided by immediate reward signals to achieve dual alignment on principal and long-tail modes. Extensive experiments demonstrate consistent improvements across Rouge-L and BertScore metrics, surpassing greedy heuristics by 0.4-0.6 points and outperforming other baseline methods on diverse benchmarks. 

---
# RoAd-RL: A Unified Library and Benchmark for Robust Adversarial Reinforcement Learning 

**Authors**: Adithya Mohan, Daniel Kriegl, Torsten Schön  

**Link**: [PDF](https://arxiv.org/pdf/2606.29867)  

**Abstract**: Deep Reinforcement Learning (DRL) has achieved significant success in robotics and autonomous systems, yet remains vulnerable to adversarial perturbations that can severely degrade performance. Research in adversarial reinforcement learning is often limited by fragmented implementations, inconsistent evaluation protocols, and poor reproducibility. To address these challenges, we present \textbf{RoAd-RL}, an open-source benchmarking framework that provides unified abstractions for policies, attacks, defenses, and robustness metrics, together with reproducible evaluation pipelines and seamless integration with Stable-Baselines3 and Gymnasium.
We evaluate DQN, PPO, and SAC agents in LunarLander and Highway-v0 under 192 attack-defense configurations. Results reveal substantial variations in robustness across environments and show that some commonly used defenses can be more detrimental than the attacks they aim to mitigate, while temporal smoothing consistently achieves strong performance. RoAd-RL establishes a standardized benchmark for adversarial reinforcement learning research and is publicly available at this https URL. 

---
# SUMO: Segment and Track Any Motion with Nonlinear State Space Models 

**Authors**: Kexin Tian, Sixu Li, Keshu Wu, Yang Zhou, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29861)  

**Abstract**: Visual Object Tracking (VOT) and Moving Object Segmentation (MOS) are two fundamental tasks in computer vision that involve both spatial and temporal object dynamics. Existing methods rely predominantly on visual cues and thus often falter in real-world scenarios where object motions are inherently complex and nonlinear. To address this limitation, we propose SUMO, a zero-shot, training-free, unified framework integrating nonlinear dynamics with vision-based segmentation for accurate and consistent VOT and MOS. Specifically, we develop a nonlinear State Space Model (SSM) inspired by robotics principles to capture the complex object dynamics. Building on this model, we propose a Selective Unscented Filter (SUF) for accurate state estimation, which features a joint scoring mechanism and dynamically fuses multi-source predictions to identify the most plausible object state over time. Furthermore, we apply a memory selection mechanism to evaluate the reliability of memory frames. Our extensive experimental results show that SUMO achieves state-of-the-art performance on both VOT and MOS tasks. 

---
# Exploring Motivations for Algorithm Mention in the Domain of Natural Language Processing: A Deep Learning Approach 

**Authors**: Yuzhuo Wang, Yi Xiang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29859)  

**Abstract**: With the rise of data-intensive science, algorithms have become central to scientific research. In academic papers, algorithms are mentioned for different purposes, such as describing, using, comparing, or improving methods for specific research tasks. Identifying these purposes can reveal relationships among algorithms and help assess their roles and value. Taking natural language processing (NLP) as an example, this study proposes a sentence-level framework for identifying, analyzing, and tracing the evolution of motivations for mentioning algorithms. We first identify algorithm entities and algorithm-related sentences from full-text papers through manual annotation and machine learning. We then classify mention motivations using pretrained models and data augmentation, and analyze their distribution and temporal evolution. The results show that deep learning models trained with augmented data outperform traditional machine learning models in motivation classification. In NLP papers, more than half of algorithm-related sentences express direct use, whereas improvement is the least frequent motivation. The diversity of motivations has increased over time. For specific algorithm categories, grammar-based algorithms are more often mentioned for description, while machine learning algorithms are more often mentioned for use. Over time, use motivations have gradually replaced description motivations across different algorithms, and the number of motivation types associated with individual algorithms has declined significantly. This study reveals how authors mention algorithm entities in academic writing and provides a basis for future research on algorithm relationship identification and algorithm impact evaluation. 

---
# MATCH: Modulating Attention via In-Context Retrieval for Long-Context Transformers 

**Authors**: Linrui Ma, Chun Hei Lo, Xinyu Wang, Peng Lu, Xihao Yuan, Hanting Chen, Kai Han, Xinghao Chen, Chengjun Zhan, Hanlin Xu, Yichun Yin, Lifeng Shang, Feng Wen, Boxing Chen, Yufei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2606.29844)  

**Abstract**: The quadratic computational cost of traditional attention mechanisms poses a major bottleneck to the scalability and practical deployment of large language models (LLMs), particularly in long-context scenarios. To improve efficiency, existing approaches often enforce rigid structural constraints such as local attention windows. However, these strategies typically lead to substantial performance degradation on tasks requiring precise long-range recall. In this work, we propose MATCH, a scalable and efficient framework that augments sparsified attention mechanisms with dynamically integrated in-context information through an efficient retrieval system. Empirical results show that MATCH significantly improves the performance of sparse-attention models on both synthetic and real-world natural-language tasks. These findings highlight the versatility of MATCH as a general approach for enhancing in-context retrieval capabilities while maintaining the efficiency benefits of sparse attention architectures. 

---
# Neural Procedural Memory: Empowering LLM Agents with Implicit Activation Steering 

**Authors**: Chengfeng Zhao, Yuqiao Tan, Shizhu He, Yequan Wang, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29824)  

**Abstract**: While Large Language Models (LLMs) excel as static solvers, transforming them into autonomous agents remains challenging. This transition requires continuous environmental interaction, yet current agents lack the necessary persistent procedural memory. Existing approaches predominantly employ Retrieval-Augmented Generation (RAG) to inject explicit textual guidelines into model contexts. However, relying solely on symbolic instructions can introduce a text-action disconnect, frequently failing to activate the internal representations necessary for correct task execution. To address this, the paper introduces Neural Procedural Memory (NPM), a training-free framework that represents agent memory through implicit activation steering rather than explicit instructions. By distilling procedural skills from historical contrastive experiences into steering vectors in the activation space, NPM directly activates the task-relevant neural mechanisms to guide task execution. Evaluations across four agent benchmarks show that NPM performs comparably to baselines using explicit textual instructions. Furthermore, the results show that combining implicit steering with explicit workflows provides complementary advantages, leading to more robust task execution. Representational analyses indicate that these steering vectors encode consistent task logic, forming organized structures within the activation space. These findings suggest that implicit activation steering provides a promising approach for managing agent memory. 

---
# Experience Graphs: The Data Foundation for Self-Improving Agents 

**Authors**: Gang Liao, Yujia He, Abdullah Ozturk, Zhouyang Li, Ying Wang, Zhitong Guo, Hongsen Qin, Yaobin Qin, Tao Yang, Zewei Jiang, Dianshi Li, Jort Gemmeke, Jiangyuan Li, Liyuan Li, Nathan Yan, Masha Basmanova, Uladzimir Pashkevich, Matt Steiner, Pedro Pedreira, Rob Fergus, Anirudh Goyal, Carole-Jean Wu, Gaoxiang Liu, Andrew Witten, Daniel J. Abadi  

**Link**: [PDF](https://arxiv.org/pdf/2606.29823)  

**Abstract**: The database community has repeatedly advanced the state of the art by recognizing that new workloads demand new system architectures. We argue that long-horizon agentic tasks -- code generation, scientific discovery, hardware design -- are such a workload. These agents explore: they generate artifacts, execute tools, observe failures, branch, and repair over hundreds of steps. This search produces a structured object we call an experience graph: executable artifacts, tool outputs, rewards, sibling comparisons, and causal lineage. Yet existing agent frameworks treat this experience as disposable state -- JSON checkpoints and session logs that cannot be recovered after a crash, queried across users, or materialized into training data. We propose Trellis: a data foundation that treats the experience graph as first-class, governed, queryable database state. The core insight is that search over experience graphs is a database access pattern. Frontier selection is a query, cross-session reuse is vector-seeded graph retrieval, training-data extraction is a materialized view, and reconstructing what an agent knew at any past step is a time-travel query. When the database owns the experience graph, agents become stateless compute, and crash recovery, horizontal scaling, and a closed-loop training flywheel emerge as architectural byproducts. We ground the design in KernelEvolve, a production accelerator-kernel optimizer at Meta, where cross-session reuse reaches a target speedup roughly 10x faster at 52% lower token cost. More broadly, Trellis turns inference-time search from disposable computation into a durable institutional asset: logs made databases reliable; experience graphs may make agents cumulative. 

---
# Dual-Flow Reinforcement Learning with State-Aware Exploration 

**Authors**: Qijun Li, Zheng Fu, Qi Song, Yifei He, Weitao Zhou, Kun Jiang, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29820)  

**Abstract**: In complex continuous-control reinforcement learning tasks, multimodal optimal actions often coincide with uncertain, multimodal return distributions, making reliable value estimation and multimodal exploration challenging. Existing value estimation methods using unimodal Gaussians restrict expressiveness and yield biased estimates. Recent generative policies can represent multimodal actions but often collapse to a few modes and under-explore high-value areas of the action space. Motivated by these challenges, we propose Dual-Flow RL, a unified actor-critic framework that jointly models a continuous return distribution and a multimodal policy distribution using conditional flow matching (CFM). This design supports reliable value estimation and sustained multimodal exploration. To further enhance exploration, we introduce an Entropy-Covariance Exploration Regulator (ECER) that enables state-aware exploration regulation leveraging policy entropy and action-uncertainty covariance. Experiments on DeepMind Control Suite and Humanoid-Bench show that Dual-Flow RL achieves state-of-the-art performance on most tasks, significantly outperforming prior diffusion-based and flow-based methods. 

---
# How Far Can You Get Without a GPU? A Systematic Benchmark of Lightweight Hallucination Detection Across Question Answering, Dialogue, and Summarisation 

**Authors**: Kriti Faujdar, Smit Kadvani  

**Link**: [PDF](https://arxiv.org/pdf/2606.29809)  

**Abstract**: Hallucination detection has become a pressing requirement for trustworthy AI deployment at scale. The most accurate detection methods depend on GPU-intensive inference, proprietary API calls, or white-box access to the generating model. This puts them out of reach for resource-constrained researchers and practitioners. In this paper, we explore a practical alternative: how well can hallucination detection perform using only lightweight, CPU-feasible methods built on publicly available models? We systematically benchmark five such methods: ROUGE-L, semantic similarity, BERTScore, a Natural Language Inference (NLI) detector based on a FEVER-trained DeBERTa model, and a score-level ensemble of similarity and NLI. We evaluate them across all three tasks of the HaluEval benchmark: question answering (QA), dialogue, and summarisation. We calibrate each method on a held-out validation split and evaluate it on 2,000 test instances per task. We find that no single method dominates and performance is highly task-dependent. The ensemble performs best on QA (F1 = 0.792, AUC-ROC = 0.873), the NLI detector leads on dialogue (AUC-ROC = 0.713), and all five methods degrade to near-random performance on summarisation (AUC-ROC between 0.469 and 0.574). This task-dependence and the systematic failure on summarisation map the practical frontier of GPU-free hallucination detection. They give practical guidance for method selection under computational constraints. All experiments run on a standard laptop CPU using public models. 

---
# Making Multimodal LLMs Reliable Chart Data Extractors: A Benchmark and Training Framework 

**Authors**: Yuchen He, Peizhi Ying, Liqi Cheng, Kuilin Peng, Yuan Tian, Dazhen Deng, Yingcai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29808)  

**Abstract**: Chart data extraction, which reverse-engineers data tables from chart images, is essential for reproducibility, analysis, retrieval, and redesign. Existing interactive tools are reliable but tedious, and mixed-initiative systems, while more efficient, lack generalizability. Recent multimodal large language models (MLLMs) offer a unified interface for chart interpretation, yet their ability to extract accurate data tables, especially without visible labels, remains unclear. We build a benchmark featuring diverse real-world charts without data labels to evaluate this capability. Results show that, while current MLLMs reliably reconstruct table structures, they struggle with precise value recovery. To address this, we revisit chart data extraction from a human-centered perspective and argue that extraction should follow a progressive learning process similar to how people read charts. Our training framework substantially improves numerical accuracy, achieving state-of-the-art performance with a 7B-parameter model. A user study further shows that our model effectively supports mixed-initiative workflows for reliable chart data extraction. 

---
# Accelerating Q-learning through Efficient Value-Sharing across Actions 

**Authors**: Prabhat Nagarajan, Brett Daley, Martha White, Marlos C. Machado  

**Link**: [PDF](https://arxiv.org/pdf/2606.29806)  

**Abstract**: Action-values are foundational to many control algorithms such as Q-learning. Therefore learning action-values efficiently is central to reinforcement learning (RL). However, learning them can be slow, requiring many updates to move values from their initialization, typically near zero, to their true values, which may be far from zero. Moreover, action-value learning algorithms typically update each state-action pair independently, without learning shared value structure across actions within a state. In this paper, we address these inefficiencies by introducing the mean-expansion layer, which accelerates action-value learning by sharing values across actions within a state and by changing the problem from directly learning potentially large action-values to learning a lower-norm representation of them. In deep RL, this layer can be applied as a parameter-free addition to Q-network architectures without altering the underlying algorithm. Applied to deep Q-networks and implicit quantile networks, it improves aggregate performance across 57 Atari games while increasing action gaps and dramatically reducing value overestimation. 

---
# Multi-Level Distributional Entropy for Explainable Network Intrusion Detection 

**Authors**: Mohamed Aly Bouke, Md Shohel Sayeed, Swee-Huay Heng, Azizol Abdullah, Mohamed Othman  

**Link**: [PDF](https://arxiv.org/pdf/2606.29797)  

**Abstract**: Machine learning network intrusion detection systems (IDS) rely on aggregate flow statistics that discard distributional structure, while established entropy measures require raw packet sequences unavailable in pre-aggregated flow datasets. We propose Multi-Level Distributional Entropy (MDE), an analytical framework that derives interpretable entropy features directly from flow-level summary statistics at three levels: within-flow Gaussian differential entropy, cross-directional Jensen-Shannon divergence (JSD), and Transmission Control Protocol (TCP) flag-pattern Shannon entropy, without raw packet access or training data. Across four benchmarks (NSL-KDD, CICIDS-2017, CICIDS-2018, UNSW-NB15) under a leakage-free fold-local pipeline, entropy-only features achieve weighted F1 of 0.708-0.989, matching conventional features without degrading performance. Full operational metric reporting then exposes failure modes that aggregate F1 conceals. On CICIDS-2018, F1=0.74 hides a detection rate (DR) of 0.48, and on held-out attack families F1 exceeds 0.998 while DR falls to zero. Under temporal shift, a pseudo-live replay of 703K flows reveals a threshold-ranking divergence in which score ranking is preserved (AUC=0.87) but fixed thresholds collapse (DR=0.082) and recalibration offers no recovery. SHapley Additive exPlanations (SHAP) fold-stability analysis (Spearman rho=0.80-0.95) confirms that entropy attributions are reproducible and domain-coherent across heterogeneous environments. 

---
# What Drives the Inlier-Memorization Effect? A Theory of Outlier Detection via Early Training Dynamics 

**Authors**: Kunwoong Kim, Dongha Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.29791)  

**Abstract**: Outlier detection (OD) aims to identify anomalous instances by learning the underlying structure of normal data (inliers), and is particularly challenging in fully unsupervised settings where no information about anomalies is available during training. Recent advances have leveraged the inlier-memorization (IM) effect, a phenomenon in which deep models memorize inlier patterns earlier than those of outliers, as a powerful signal for distinguishing outliers. However, despite its empirical success, the theoretical understanding of the IM effect remains limited. In this work, we present a theoretical study of the IM effect. Focusing on a simple autoencoder, we show that, under mild assumptions, the model can successfully memorize inliers while failing to memorize outliers during certain stages of early training. In particular, we characterize not only the emergence of the IM effect, but also its strength and persistence, and analyze how these properties depend on the data distribution and parameter initialization. In addition, building on these insights, we derive simple yet practical guidelines for enhancing the IM effect, including data preprocessing and parameter initialization schemes, achieving state-of-the-art performance on the ADBench datasets. Our findings provide a theoretical foundation for the IM effect and offer actionable directions for improving IM-based outlier detection methods. 

---
# HERO: Improving the Reliability and Sensitivity of Generative Model Evaluation Using Historical Data 

**Authors**: Xinrui Ruan, Zhenyu Zhao, Waverly Wei, Yueshan Zhang, Zeyu Zheng, Sui Huang, Jingshen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29784)  

**Abstract**: Reliable generative AI models critically rely on expert human annotations to evaluate output quality, yet these "gold" labels are expensive to collect and limited in quantity. Organizations thus often turn to collecting vast but noisy "silver" labels from crowdsourced workers or vendor annotators as proxies for gold labels. Because gold remains the evaluation target, naively aggregating noisy silver labels may introduce bias, and estimators built on sparsely observed gold labels may have high variance to resolve the model performance gaps that guide practical decisions. Model evaluation has become an ongoing operational practice rather than a one-time exercise, with evaluation rounds repeating across model versions, releases, and content domains. A natural question is whether the previous historical evaluation data can be used to improve each new round of evaluation. We introduce HERO (History Enhanced RObust model evaluation), a novel framework that uses historical data to suppress bias (improve reliability) and reduce variance (improve sensitivity) in model performance evaluation. HERO calibrates silver labelers' performance learned from historical gold annotations, and stabilizes the resulting estimator by anchoring it to covariate information measured with high precision in the historical data. HERO can be broadly applied across multiple common evaluation tasks, and remains valid when only a subset of historical labelers appears in the current round. We establish conditions under which the bias and variance reductions hold, showcase HERO's performance in simulation studies, and demonstrate its effectiveness on real-world model evaluation benchmarking datasets. 

---
# FalconTrack: Photorealistic Auto-Labeled Perception and Physics-Aware Vision-Based Aerial Tracking 

**Authors**: Yan Miao, Karteek Gandiboyina, Noah Giles, Hideki Okamoto, Bardh Hoxha, Georgios Fainekos, Sayan Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2606.29783)  

**Abstract**: Vision-based aerial tracking is critical in GPS-denied environments. Reliable perception for tracking depends on large-scale labeled data, yet most photorealistic datasets rely on heavy manual annotation and are time-consuming to produce. We present FalconTrack, a unified perception-and-tracking framework that (i) leverages a photorealistic editable simulator for automated label generation and (ii) combines multi-head perception with physics-aware tracking for zero-shot sim-to-real transfer. FalconTrack provides an automated labeling pipeline in a Gaussian Splatting simulator that isolates target Gaussians from short object videos and composites them with randomized backgrounds to generate RGB, mask, class, and 6-DoF pose labels, producing about 10k labeled images in under 20 minutes. Using this dataset, we train a multi-head perception module with staged learning and reprojection consistency, and fuse its outputs with class-conditioned dynamics priors in an EKF for tracking. Our perception model outperforms two baselines and reaches 96-100% class accuracy in zero-shot sim-to-real transfer on three geometrically diverse objects and two environments, while maintaining consistent performance in unseen simulated and real scenes. In real hardware closed-loop visual tracking, the onboard system runs at about 25 Hz and achieves 100% success in sim-to-real F1-tenth and gate tracking in five trajectories across two environments, while a mask-centered vision baseline drops to 60% success on F1-tenth during fast out-of-view scenarios. 

---
# Mandol: An Agglomerative Agent Memory System for Long-Term Conversations 

**Authors**: Yuhan Zhang, Zhiyuan Guo, Ziheng Zeng, Wei Wang, Wentao Wu, Lijie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29778)  

**Abstract**: Long-term conversational agents need to remember and query cross-session, multi-typed information with complex correlations. Existing agent memory systems rely on heterogeneous vector and graph databases, which fragment memory information and cause high cross-database I/O latency. For retrieval, common RAG-style methods tend to introduce noise, miss correlated clues, and lack token budget control, degrading LLM accuracy and efficiency.
We propose Mandol, an agglomerative memory system that consolidates fragmented memory representations and storage into a unified memory-native architecture. Its core components include: (1) a hierarchical memory model that organizes memory into a basic layer representing raw memory information and a high-level abstract layer that agglomerates basic memories into traceable abstract memories, both uniformly represented as structured semantic graphs; (2) an agglomerative semantic data structure combining SemanticMap and SemanticGraph, which natively fuses key-value, vector, and graph structures and provides unified hybrid retrieval operators to eliminate cross-database I/O; and (3) a quantitative query mechanism with query-adaptive routing, quantitative denoising and conflict resolution, and token-constrained context generation, all without involving LLMs during retrieval. Experiments on two widely used long-term conversation benchmarks, LoCoMo and LongMemEval, show that Mandol achieves the best overall accuracy among representative agent memory systems. For performance comparison, Mandol also obtains a 5.4x retrieval speedup and a 4.8x insertion speedup under 10 QPS concurrent load, while still maintaining low latency on consumer-grade hardware. 

---
# Towards Generalizable and Evidential Nuclear Magnetic Resonance-Based Molecular Structure Elucidation via Large Language Model Agent 

**Authors**: Zheng Fang, Chen Yang, Yusen Tan, Yunpeng Zhao, Fanjie Xu, Hongxin Xiang, Hanyu Sun, Hanyu Gao, Xiaojian Wang, Wenjie Du, Yuqiang Li, Jun Xia  

**Link**: [PDF](https://arxiv.org/pdf/2606.29776)  

**Abstract**: Nuclear Magnetic Resonance (NMR) spectroscopy is the gold standard for molecular structure elucidation, yet interpreting complex spectra for unknown molecules remains a bottleneck reliant on human expertise. While artificial intelligence has advanced this field, current methods face a critical trade-off: database retrieval cannot identify novel scaffolds, while de novo molecular structure elucidation models operate as black boxes, lacking the atom-level interpretability required for rigorous scientific validation. Here, we present NMRAgent, an evidential reasoning agent powered by large language models (LLMs) that bridges this gap by integrating specialized spectral analysis tools with chemical knowledge graphs. Unlike previous approaches, NMRAgent mimics the deductive reasoning of human experts: it takes experimental NMR spectra and molecular formula as input, plans the elucidation process, proposes candidate structures, verifies peak-atom consistency, and refines misaligned substructure through formula-aware fragment optimization. Enabled by its evidential reasoning, NMRAgent outperforms state-of-the-art methods, improving top-1 accuracy by 46.5% and Tanimoto similarity by 0.502 on a scaffold-split benchmark with novel scaffolds in the test set. Besides, we demonstrate the agent's practical utility by elucidating the structures of two previously unknown natural products isolated from Hydrangea davidii and Vitex trifolia, and by correcting structural misassignments in established literature. By combining high-accuracy prediction with transparent and evidence-based reasoning, NMRAgent establishes a new paradigm for interpretable AI in analytical chemistry. 

---
# TopoAgent: An Agentic Framework for Automated Topology Learning in Medical Imaging 

**Authors**: Guangyu Meng, Pengfei Gu, Xueyang Li, Yiyu Shi, Erin Wolf Chambers, Danny Z. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.29763)  

**Abstract**: Topological data analysis (TDA), particularly persistent homology (PH), captures geometric structural properties in medical images (e.g., connected components, loops, shape characteristics), which conventional pixel-level deep learning approaches often neglect. While many topological descriptors are known for converting persistence diagrams (PDs) or raw images into topological feature vectors, existing methods mostly default to a single fixed descriptor (e.g., persistence images), leaving the diversity of topological representations largely unexplored. To the best of our knowledge, there is no known large language model (LLM)-based agentic framework that can automatically determine the most suitable topological descriptors for a given image dataset and produce the corresponding topological feature vectors for downstream tasks. To fill this gap, we propose \textbf{TopoAgent}, an LLM-based agentic framework that automates topology learning for medical image this http URL operates through a Perception--Reasoning--Action--Reflection loop supported by 21 domain-specific tools and dual memory that accumulates experience across runs. Its skill set is distilled from systematic evaluation of 15 topological descriptors across 26 datasets with six classifiers. TopoAgent analyzes input images and their topological characteristics, reasons about which topological descriptors best suit the input, and determines the optimal descriptor and its configuration, all without task-specific training. 

---
# PS-PPO: Prefix-Sampling PPO for Critic-Free RLHF 

**Authors**: Doo Hwan Hwang, Kee-Eung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.29758)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) for Large Language Models increasingly relies on critic-free methods as a practical alternative to actor--critic training. Despite their simplicity, existing critic-free approaches propagate a trajectory-level learning signal uniformly across all tokens in a trajectory. This requires full-trajectory policy updates for every rollout, leading to substantial optimization cost for long reasoning traces, even though intermediate prefixes often contain enough information to largely determine the final outcome. We propose Prefix-Sampling Proximal Policy Optimization (PS-PPO), a compute-efficient critic-free method for RLHF that exploits this temporal redundancy. PS-PPO introduces a prompt-conditioned cutoff distribution and samples a cutoff timestep for each trajectory. During the update pass, PS-PPO backpropagates only through the sampled prefix of each trajectory and applies an importance-weighting correction so that the resulting truncated gradient estimator remains unbiased with respect to the full-trajectory objective. Experiments on mathematical reasoning and RLHF benchmarks show that PS-PPO achieves large reductions in training compute and peak GPU memory, while maintaining accuracy comparable to strong critic-free baselines. 

---
# From Trait to Behavior: A Cognitive-Affective Personality System (CAPS) Perspective on Multi-Homing Intention in AIGC Platforms 

**Authors**: Xuchao Zhang, Jihye Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.29726)  

**Abstract**: With the rapid development of Artificial Intelligence Generated Content (AIGC) platforms, users increasingly show cross-platform usage intentions. Existing research focuses on adoption and usage intentions in single-platform AIGC contexts. A theoretical gap still exists in studies on cross-platform usage. This paper constructs and verifies a three-stage multiple mediation model based on the personality trait-perception-behavioral response framework. The model integrates the optimum stimulation level (OSL) theory, complementarity theory, and perceived value theory, and it sets social influence and use experience as control variables to examine users' multi-homing intention. The results show that: (a) OSL significantly enhances users' perceived complementarity; (b) perceived complementarity positively affects perceived epistemic value; (c) perceived epistemic value significantly and positively predicts multi-homing intention; (d) OSL influences multi-homing intention through a chain mediation path of perceived complementarity and perceived epistemic value; and (e) social influence has a significant positive effect on multi-homing intention, while the effect of use experience is not significant. 

---
# Redefining Maritime Anomaly Detection via Equation-Grounded Synthetic Anomalies 

**Authors**: Youngseok Hwang, Sungho Bae, Dohun Lee, Jaeeun Seo, Jeehong Kim, Wonhee Lee, Hyunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.29721)  

**Abstract**: Maritime anomaly detection is essential for ensuring maritime safety, security, and efficient traffic management at sea, with Automatic Identification System (AIS) data serving as a primary data source. Despite its importance, most publicly available AIS datasets lack predefined anomaly labels, forcing prior studies to rely on either distribution-based rarity or domain rule/expert-assisted labeling. These approaches, however, face fundamental limitations: statistical rarity often fails to reflect practically critical events, while expert-based labeling is costly, subjective, and difficult to scale. Moreover, both paradigms tend to overlook interaction-driven hazards such as near-miss approaches between vessels. To address these challenges, we propose an equation-grounded anomaly taxonomy that is implementable under a limited AIS observation schema and extensible to other AIS datasets. Specifically, the taxonomy defines three anomaly types: unexpected AIS activity (A1), route deviation (A2), and close approach (A3), covering both single-vessel and inter-vessel anomalies. Building on this taxonomy, we introduce a unified score-synthesize-label pipeline that produces LLM-guided plausibility scores, uses them to synthesize anomalies, and assigns timestamp-level labels. To rigorously assess detection performance, we further design benchmark evaluation settings that account for variations in temporal-window length and anomaly-type composition, and evaluate a broad range of time-series models and anomaly detection models. Together, these contributions provide a systematic basis for evaluating maritime anomaly detection methods across different anomaly types. Our code is available at this https URL. 

---
# Diagnosing and Mitigating Context Rot in Long-horizon Search 

**Authors**: Shijie Xia, Yikun Wang, Zhen Huang, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29718)  

**Abstract**: Extensive context has become the norm as Large Language Models (LLMs) are increasingly deployed in long-horizon tasks. The concern that increasing context length degrades model capabilities, known as context rot, has become a central issue for these applications. In this paper, we focus on deep search scenarios, aiming to investigate the rot phenomenon and its mitigation strategies. By evaluating four flagship open-source models across three benchmarks, we reveal a prevalent but unnoticed rot phenomenon: extensive context causes models to directly give up or prematurely provide uncertain answers, and this issue is exacerbated as the context grows. Through pruning experiments, we demonstrate the relationship between the accumulated context and the rot phenomenon. Furthermore, we investigate mitigating this issue through context management and post-hoc rejection sampling. For context management, we systematically evaluate seven different methods across three categories, based on performance, cost, and impact on context rot, providing clear guidance for strategy selection and usage. For rejection sampling, we develop a rot-aware filtering strategy and demonstrate its effectiveness across three aggregation methods. Finally, we show that these two approaches can be combined for further performance improvements. 

---
# Optimizing Expert-Designed Crystal Graph Networks for Band-Gap Prediction with an Autonomous LLM Research Loop 

**Authors**: Chenmu Zhang, Boris I. Yakobson  

**Link**: [PDF](https://arxiv.org/pdf/2606.29717)  

**Abstract**: Predicting a material's properties from its structure is a central, fast-advancing problem in computational materials science. A decade of work has produced standard public benchmarks and many published machine-learning models for the task (Dunn et al., 2020). The task's fixed metric and these baselines make it a natural setting for autonomous agent research (Karpathy, 2026). On the MatBench band-gap benchmark ($>$100k crystals), a general-purpose coding agent autonomously built the most accurate model trained without external pretraining, ahead of all seventeen expert-designed models reported for the task. A closer analysis shows it reached this by implementing known methods: either already standard in crystal neural-network models, or borrowed from other areas of machine learning. The contributing implementations include element-pair features on each message-passing edge and a crystal space-group embedding. The work not only demonstrates that LLM-agent autonomous research can optimize an expert-designed machine learning model for material property prediction, but also investigates the limitations of such autonomous research. 

---
# SEVA: Self-Evolving Verification Agent with Process Reward for Fact Attribution 

**Authors**: Aojie Yuan, Yi Nian, Haiyue Zhang, Zijian Su, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.29713)  

**Abstract**: Hallucination is the reliability bottleneck for LLM-based agents, and fact attribution verifiers are the last line of defense -- yet today's verifiers emit only opaque binary labels, leaving agents unable to self-correct and operators unable to audit. We present SEVA, a structured verification agent that emits evidence alignments, step-by-step reasoning chains, calibrated confidence, and a six-category error diagnosis with actionable fixes. Training such an agent with RL is non-trivial: standard binary reward on multi-component output triggers advantage collapse -- within-group reward variance vanishes and the GRPO gradient disappears. We resolve this with a process reward that decomposes verification quality into five independent components weighted 70/30 toward process signals, restoring the gradient and inducing an implicit curriculum -- the agent first masters verification behavior (alignment 0.917 -> 0.997, format 72% -> 100%), then outcomes (F1 64.9 -> 69.0). Structured output further enables a Verify -> Reflect -> Probe -> Refine self-evolution loop, which over four rounds on a 7B model surfaces an unexpected structural finding: each round produces a benchmark-specialist, not a generalist (+15 pp on HaluEval, -10 to -14 pp on TruthfulQA in the same model, persistent at 4x data). On ClearFacts, SEVA-3B matches GPT-4o-mini (69.0 vs. 69.8 F1) while producing substantially richer, auditable output -- confirming a principle that should generalize: for any RL task with multi-component generation, reward granularity must match output granularity. 

---
# ARMOR: Adaptive Retriever Optimization for Low-Resource Telecom Question Answering 

**Authors**: Heshan Fernando, Quan Xiao, Yan Xin, Tianyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.29706)  

**Abstract**: Telecom question answering (QA) is a challenging setting for retrieval-augmented generation (RAG): evidence is fragmented across standards, papers, encyclopedic resources, and web documents, and answers often hinge on technical tables, equations, and specialized protocol language. In low-resource subdomains, generator fine-tuning can over-specialize and degrade general capability, making query-side retriever adaptation an attractive alternative. To this end, we ask whether a fixed-generator, query-adapted RAG system can outperform generator-side adaptation, and which retriever objectives best support that setting. We motivate retrieval, rather than generator fine-tuning, as the adaptation target through a capacity comparison: under bounded-parameter and soft-retrieval assumptions, query-encoder tuning can have a smaller estimation term than supervised fine-tuning when its effective dimension is smaller. We identify two particularly relevant objectives -- the latent-document RAG likelihood, which optimizes generation utility, and the InfoNCE contrastive objective, which improves semantic retrieval geometry -- and leverage them jointly through a retriever optimization method targeting downstream QA performance in the telecom domain. Specifically, we introduce ARMOR, Adaptive Regularized Mixture Optimization for Retrievers, which learns separate temperatures for the RAG retrieval distribution and InfoNCE softmax and regularizes the adapted query encoder toward the frozen base query encoder. Across telecom-specific retrieval and generative QA benchmarks, we show that ARMOR improves evidence retrieval and answer generation in several in-domain settings. Code is available at this https URL. 

---
# Early Warning Signals for OpenVLA Failure under Visual Distribution Shift 

**Authors**: Dipesh Tharu Mahato, Rachel Ren  

**Link**: [PDF](https://arxiv.org/pdf/2606.29699)  

**Abstract**: Vision Language Action models combine perception, language grounding, and control in a single policy, but their failures are hard to diagnose once visual conditions shift. We test whether OpenVLA feedforward activations contain linearly decodable information about near term task failure in LIBERO manipulation rollouts. The policy is fixed throughout. We log internal activations during execution and fit lightweight monitors after the rollouts are collected. Occlusion is the main controlled stress test. It reduces OpenVLA success from $57\%$ to $17\%$ over $100$ episodes per condition. Under this shift, a logistic probe at layer 16 reaches AUROC $0.972$ and AUPRC $0.352$ for predicting failure within a $15$ step horizon. It outperforms both a mean difference direction and an action disagreement baseline. A sparse layer sweep finds uneven decodability across depth: layer 16 is strongest among the tested layers, layer 8 remains informative, and layer 10 is weaker. To check whether the monitor is just an occlusion detector, we also evaluate color shift and camera jitter without refitting. Color shift produces no failures in this setting, so it is a benign control rather than a failure benchmark. Camera jitter does induce failures, and the occlusion trained monitor remains above random. The result is deliberately limited: OpenVLA internal states contain failure relevant structure under controlled perceptual shift, but these experiments do not establish a causal mechanism, task held out generalization, or a deployable recovery system. 

---
# A Machine-Verified Proof of a Quantum-Optimization Conjecture 

**Authors**: Uri Kol, Maor Ben-Shahar, Kfir Sulimany, Dirk Englund  

**Link**: [PDF](https://arxiv.org/pdf/2606.29687)  

**Abstract**: We report a machine-verified resolution of a problem open for over a decade in quantum optimization: the Farhi, Goldstone and Gutmann (FGG) conjecture that depth-$p$ Quantum Approximate Optimization Algorithm (QAOA) on the ring of disagrees attains approximation ratio $(2p+1)/(2p+2)$ exactly. We found the proof using a large language model, Claude Fable 5, and verified its correctness end-to-end by the Lean 4 proof assistant. Our methodology includes several ingredients: building on a substantial Lean library of quantum information, we formalized the QAOA components and the known parts of the problem, and reduced the conjecture to a single open mathematical statement. The model was then handed the library and our agentic toolkit, and tasked with closing that gap by constructing a proof in Lean. The resulting process is a feedback loop between the model's natural-language reasoning and Lean's mechanical verification, which converged to a machine-verified proof. Human verification is required only for the structural scaffolding - that the formal statement faithfully encodes the intended claim - while the proof itself is supplied by the model and certified mechanically by Lean. The proof is nevertheless striking - the model uncovered a hidden dynamical symmetry of the problem and exploited it, borrowing tools and machinery from an adjacent field to turn a hard existence problem into an explicit construction. This work paves the way for resolving open conjectures in quantum information science and beyond. 

---
# Unlocking the Visual Record of Materials Science: A Large-Scale Multimodal Dataset from Scientific Literature 

**Authors**: Subham Ghosh, Shubham Tiwari, Mohammad Ibrahim, Abhishek Tewari  

**Link**: [PDF](https://arxiv.org/pdf/2606.29667)  

**Abstract**: The materials science literature encodes decades of experimental knowledge in figures, yet this visual record remains locked away and inaccessible to AI at scale. The core difficulty is structural: most scientific figures are compound, with a single caption describing multiple sub-panels simultaneously, making direct image-text pairing unreliable. We present MatMMExtract, an end-to-end open-source pipeline that resolves this by decomposing compound figures into individual sub-panels and generating structured, grounded annotations using a large language model guided by a curated materials science taxonomy. Applied to 14,810 open-access articles, MatMMExtract produces MatSciFig; 391,606 panel-level image-text pairs from 180,571 figures, each annotated with a sub-caption, a two-level visualisation category spanning 19 classes and over 100 subtypes, and a scientific summary. To enable accurate panel localisation, we introduce MaterialScope, a domain-specific detection dataset of 2,811 manually annotated materials science figures, on which a fine-tuned YOLO12-m detector achieves mAP_50 of 0.9227. Among six benchmarked language models, Gemini 3.1 Flash Lite delivers the best cost-quality trade-off for annotation generation, with 82% of outputs rated good and a hallucination rate of 4.8%. A dual-encoder retrieval baseline on MatSciFig achieves a 4.4 times improvement in R@1 over zero-shot CLIP, demonstrating the dataset's immediate utility for vision-language learning. All resources are released openly to the community. 

---
# Hybrid Retriever Evolution for Multimodal Document Reasoning Agents 

**Authors**: Bohan Yao, Shruthan Radhakrishna, Vikas Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2606.29648)  

**Abstract**: Different retrievers, including lexical, semantic, and multimodal approaches, provide highly complementary strengths for multimodal document understanding, yet most systems combine them through fixed pipelines that cannot adapt to the demands of individual reasoning steps. In this work, we ask whether retrieval orchestration itself can be learned as part of the reasoning process. We introduce a failure-driven evolution framework in which a meta-agent autonomously discovers how a tool-using task agent should coordinate diverse retrievers during multi-step document question answering. The meta-agent analyzes incorrect reasoning trajectories, actively probes the same tool environment to diagnose root causes, and iteratively rewrites the task agent's instructions, turning retrieval from a fixed front-end stage into an adaptive, step-wise reasoning decision. The evolved agent learns when to invoke each retriever, how to combine them, and how to compose evidence across modalities and pages. On MMLongBench-Doc and DocBench, the evolved agent achieves gains of up to +19.6 points over the unevolved baseline and consistently outperforms recent systems including MACT, MDocAgent, and SimpleDoc. Detailed retrieval analyses confirm that these improvements arise from adaptive routing and evidence composition rather than reliance on any hard coded retrieval mode, and evolution dynamics reveal a progressive shift from narrow lexical behavior to rich multi-tool coordination. These findings establish autonomous multi-agent coordination as a promising paradigm for multimodal document reasoning. 

---
# Fuzzing Large Language Models to Elicit Hidden Behaviours 

**Authors**: Mohammed Abu Baker, Lakshmi Babu-Saheer  

**Link**: [PDF](https://arxiv.org/pdf/2606.29646)  

**Abstract**: Sleeper agents are the canonical model organism of deception: models trained to behave normally but to emit an unsafe behaviour on a specific trigger. Eliciting that behaviour without knowing the trigger has not been studied systematically. We study fuzzing: injecting Gaussian noise into a model's weights or residual-stream activations and checking whether the perturbed outputs reveal the behaviour. On 6 backdoored models (7B-13B) we compare both forms of fuzzing head-to-head against temperature-sampling baselines. Fuzzing elicits the hidden behaviour more often than temperature sampling on 4 of 6 models (up to ~6x on OpenHermes-13B), and which form wins depends on the task, so both are worth running. Elicitation is uneven across each method's hyperparameter grid: a uniform sweep gives only a few percent on most models, while the best cell is 2-10x higher, so the bottleneck is hyperparameter selection, not the technique. To select hyperparameters without ground-truth access, we use a cheap proxy task (in-context secret elicitation, where a base64-encoded secret is placed in the system prompt for the model to hide) and run Thompson sampling on it to pick candidate cells, which we evaluate on the real backdoor. On the four models that can decode the secret, proxy-selected cells raise activation-fuzzing elicitation ~4x over the uniform-sweep mean (recovering ~70% of the best-cell rate on the best performing model) and weight-fuzzing by 1.3-1.8x. To our knowledge this is the first systematic study of fuzzing on sleeper-agent backdoors and the first to show proxy-task hyperparameter selection transferring to real-task elicitation. We also propose reporting such results as a (uniform-baseline, proxy-selected, oracle) triple, since these are three distinct claims that prior work has often blurred. 

---
# Fast Wireless Foundation Models with Early-Exits 

**Authors**: Omar Mashaal, Hatem Abou-Zeid  

**Link**: [PDF](https://arxiv.org/pdf/2606.29640)  

**Abstract**: While wireless foundation models (FMs) are demonstrating strong potential to enable AI-Native 6G networks, their high computational cost remains a critical barrier to deployment. The large computational cost stems from the rigid, full-depth execution of the FM backbone for every task, a process we show is not only inefficient but can also degrade performance on unseen out-of-distribution (OOD) tasks. In this paper, we propose a novel early-exit FM framework that attaches lightweight, per-task heads, at the most appropriate exit-stage of a frozen wireless FM encoder, enabling variable-depth inference tailored to each task's preferred representation depth. Our results demonstrate that these intermediate-layer features not only speed-up inference significantly (up to 93% fewer FLOPs), but also provide more transferable representations that exceed the full encoder accuracy on unseen tasks. We further demonstrate that a simple fixed-exit strategy per task is more effective than traditional early-exiting policies that route different samples to different exits based on their perceived difficulty levels. 

---
# Two-Stage Prompt Optimization for Few-Shot Relation Extraction: From Reasoning-Guided Search to Gradient-Guided Refinement 

**Authors**: Aunabil Chakma, Mihai Surdeanu, Eduardo Blanco  

**Link**: [PDF](https://arxiv.org/pdf/2606.29639)  

**Abstract**: Automatic prompt optimization is still underexplored for episodic few-shot relation extraction with smaller language models. We propose a two-stage framework that combines reasoning-based prompt optimization with gradient-based prompt optimization. The first stage can use any reasoning-based optimizer to make broadprompt improvements in natural language. The second stage applies our GradPO, which uses loss and gradient signals to identify high-impact prompt spans and refine them with local edits. Experiments on FS-TACRED and FS-FewRel show that local refinement usually improves prompts found by the first stage, and GradPO is the most consistent refiner. Our framework achieves state-of-the-art performance on FS-TACRED with Qwen3-4B and remains competitive on FS-FewRel. 

---
# Do We Still Need Fine Tuning? Turkish Sentiment Analysis in the Era of Large Language Model 

**Authors**: Sercan Karakaş, Yusuf Şimşek  

**Link**: [PDF](https://arxiv.org/pdf/2606.29614)  

**Abstract**: This study examines whether supervised fine-tuning remains necessary for Turkish sentiment analysis in the era of large language models. We compare classical machine learning methods, fine-tuned pretrained language models, and prompted large language models on a Turkish e-commerce review dataset with negative, neutral, and positive labels. Fine-tuned BERTurk models perform best overall and outperform all prompted large language models in the full three-class task. The neutral class emerges as the main difficulty: while several large language models are much more competitive in binary positive--negative classification, they degrade substantially in the three-class setting by collapsing neutral reviews into polarized categories. The findings suggest that, in realistic Turkish sentiment classification, prompted large language models do not yet match supervised fine-tuning in the zero-shot setting, and that including the neutral class is crucial for robust evaluation. 

---
# Does Role Specialization Matter for Explanation Faithfulness in Mixture-of-Experts? 

**Authors**: Yeji Kim, Housam Babiker, Mi-Young Kim, Randy Goebel  

**Link**: [PDF](https://arxiv.org/pdf/2606.29613)  

**Abstract**: Mixture-of-Experts (MoE) architectures have recently been extended with role-based mechanisms for interpretability. This is typically done by assigning semantic roles to individual expert components, for example roles like synergy, redundancy, and uniqueness in multimodal settings. However, whether such structural role decomposition preserves explanation faithfulness of the overall architecture remains largely underexplored. We hypothesize that inter-expert representation overlap weakens effective role separation and degrades attribution-based faithfulness, even when semantic roles are explicitly defined. To address this limitation, we introduce representation-level decorrelation regularization to explicitly reduce inter-expert similarity in latent space. Using representation decorrelation objectives, we encourage clearer specialization among experts by minimizing representation overlap. Our experiments show that across multiple multimodal benchmarks, this separation consistently improves explanation faithfulness, as measured by comprehensiveness, sufficiency, and their Area Over the Perturbation Curve (AOPC) summaries, while preserving task performance. We further show that these improvements are not limited to role-based architectures such as Interpretable Multimodal Interaction-aware MoE (I2MoE). Similar trends are observed in a standard sparse MoE baseline, suggesting that representation-level separation may provide a more general mechanism for enhancing explanation faithfulness in MoE systems. Overall, our findings suggest that structural role decomposition alone may be insufficient to guarantee faithful explanations and that representation-level separation helps improve explanation faithfulness. To support reproducibility, the source code and supplementary material are publicly available at this https URL. 

---
# Mechanistically Eliciting Latent Behaviors in Language Models 

**Authors**: Andrew Mack, Nina Panickssery, Alexander Matt Turner  

**Link**: [PDF](https://arxiv.org/pdf/2606.29604)  

**Abstract**: We aim to discover diverse, generalizable perturbations of LLM internals that can surface hidden behavioral modes. Such perturbations could help reshape model behavior and systematically evaluate potential risks. We introduce Causal Perturbative Elicitation (CPE), an unsupervised method for discovering interpretable low-rank adapters (LoRAs) that can elicit these latent behaviors. CPE decomposes the computations of a deep transformer slice using a heuristic tensor-decomposition-based algorithm. CPE exhibits remarkable data efficiency, learning a large number of interpretable LoRAs from a single example. Even though CPE is unsupervised, we find that in some cases it can be competitive with supervised elicitation methods via brute-force enumerative search over weight space. For instance, CPE performs similarly to matched-wall-clock-time GRPO on the Countdown task for Qwen3-8B (85% vs 87%), demonstrating that CPE can efficiently elicit complex multi-token behaviors. Since CPE is unsupervised, it can also surface hidden failure modes, such as sandbagging, restoring 85% of locked BigCodeBench performance on a password-locked version of Llama3-70B introduced by Taylor et al. (2025). Additionally, since CPE explores behaviors in weight-space rather than token-space it can potentially ameliorate exploration hacking, a misalignment failure which may arise in sufficiently self-aware AI models (Ngo, 2022). In fact, we find that CPE virtually eliminates alignment-faking (Greenblatt et al., 2024) behavior in a Llama3-70B-based model organism developed by Hughes et al. (2025). Finally, we find that CPE can be used to initialize GPT-OSS-20B in an aligned basin when running GRPO on an environment prone to reward-hacking. By providing a data-efficient method to systematically explore the space of latent model behaviors, CPE yields a powerful tool for aligning AI systems and evaluating their safety. 

---
# Langshaw: Declarative Interaction Protocols Based on Sayso and Conflict 

**Authors**: Munindar P. Singh, Samuel H. Christie V, Amit K. Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2606.29601)  

**Abstract**: Current languages for specifying multiagent protocols either over-constrain protocol enactments or complicate capturing their meanings. We propose Langshaw, a declarative protocol language based on (1) sayso, a new construct that captures who has priority over setting each attribute, and (2) nono and nogo, two constructs to capture conflicts between actions. Langshaw combines flexibility with an information model to express meaning. We give a formal semantics for Langshaw, procedures for determining the safety and liveness of a protocol, and a method to generate a message-oriented protocol (embedding needed coordination) suitable for flexible asynchronous enactment. 

---
# One Scene, Two Depths: Probing Geometric Ambiguity in Monocular Foundation Models 

**Authors**: Xiaohao Xu, Feng Xue, Xiang Li, Haowei Li, Shusheng Yang, Tianyi Zhang, Matthew Johnson-Roberson, Xiaonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29600)  

**Abstract**: A faithful 3D world representation should account for layered geometry, where a single camera ray may contain multiple visible and geometrically valid surfaces. Monocular depth estimation, however, reduces this structure to one scalar depth per pixel. Transparent scenes make this ambiguity measurable: the same ray can pass through foreground glass and observe the background, turning the supervised target into a convention of annotation, data, and training rather than a scene-intrinsic truth. A learned predictor exposes this convention as its depth-layer preference. We introduce MultiDepth-3k (MD-3k), a sparse two-layer ordinal benchmark for measuring depth-layer preference and multi-layer spatial relationship accuracy (ML-SRA). On MD-3k, leading depth foundation models exhibit diverse layer preferences under standard RGB input, showing that the same layered geometry can be resolved differently across models. We further find that Laplacian Visual Prompting (LVP), a training-free spectral input transformation, can substantially change the reported layer for certain frozen models. The strongest RGB/LVP pair, DAv2-L, reaches 75.5% ML-SRA. These results suggest that depth foundation models may express complementary geometric hypotheses that standard RGB inference leaves unexpressed. We invite the community to rethink depth supervision and evaluation through an ambiguity-aware lens, where multiple valid 3D interpretations are treated as geometric structure to be measured, preserved, and expressed. 

---
# How AI settled the complexity of the oldest SGD algorithm 

**Authors**: Michał Dereziński, Xiaoyu Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.29593)  

**Abstract**: In 1937, Stefan Kaczmarz proposed a simple algorithm for solving systems of linear equations. This algorithm turned out to be the earliest known example of stochastic gradient descent, a ubiquitous computing paradigm that drives the training of modern AI models such as ChatGPT and Gemini. Now, those AI models have joined forces to discover the worst-case complexity of the Kaczmarz algorithm. This paper tells the story of how it happened. 

---
# SonoCLIP: Mask-Guided Region-Aware Vision-Language Pretraining for Fetal Ultrasound Analysis 

**Authors**: Hang Su, Chao Sun, Zhaofan Li, Wei Hu, Juhua Liu, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2606.29586)  

**Abstract**: Vision-language foundation models have shown strong potential in medical image analysis. Although foundation models for ultrasound imaging have recently emerged, the domain remains particularly challenging due to severe speckle noise, acquisition variability, and subtle anatomical boundaries, leading to high inter-observer variability. Existing CLIP-based models rely primarily on global image-text alignment, limiting their sensitivity to clinically decisive local structures. We propose SonoCLIP, the first million-scale region-controllable fetal ultrasound vision-language foundation model that integrates segmentation masks as mask-channel visual prompts within the vision encoder, enabling joint global-local contrastive representation learning. To support scalable region-text alignment, we introduce a sigmoid-based pairwise contrastive loss that improves stability under large-scale supervision. We further curate a 1.44M-image multimodal fetal ultrasound dataset spanning 24 standard planes for large-scale pretraining. Extensive cross-center evaluations demonstrate that SonoCLIP achieves superior zero-shot transfer performance under both global and mask-guided inference, establishing a controllable and clinically oriented foundation model for fetal ultrasound analysis. Our code and data are available at this https URL. 

---
# Bilevel Optimization for Neural Architecture Search 

**Authors**: Abhishek Shukla, Ankur Sinha, Faiz Hamid  

**Link**: [PDF](https://arxiv.org/pdf/2606.29582)  

**Abstract**: Bilevel optimization has become an influential and widely adopted framework for addressing hierarchical optimization problems in machine learning, providing an effective approach to modeling the interaction between two levels of optimization, with applications such as hyperparameter tuning, meta-learning, adversarial training, and data poisoning. Neural Architecture Search (NAS), a subfield of hyperparameter optimization, is a prime example of a bilevel optimization problem, with architecture parameters optimized at the outer-level and network weights optimized at the inner level. This paper presents a structured overview of NAS through the lens of bilevel optimization. We categorize existing NAS approaches into two main classes: sampling-based methods, which search optimal architectures using different architecture samplers, and bilevel theory-based methods, which solve the architecture search problem using bilevel optimization principles. We further highlight our current research direction, wherein the bilevel NAS formulation is addressed through an auxiliary mathematical programming framework. This framework enables the systematic integration of second-order information from the model's training loss function and ensures the optimality of the model parameters while modifying architecture parameters. By simultaneously updating the architecture and model parameters along their respective optimal descent directions derived from the auxiliary mathematical program, these methods achieve more principled and theoretically consistent results. The same auxiliary program can also be used for simultaneous hyperparameter and model fine-tuning. A comparative analysis shows that bilevel theory-based approaches generally outperform sampling-based methods, both in accuracy and efficiency. 

---
# The Joint Effect of Quantization and Sampling Temperature on LLM Safety Alignment: A Factorial Analysis 

**Authors**: Hari Prasad, Ritam Pal  

**Link**: [PDF](https://arxiv.org/pdf/2606.29581)  

**Abstract**: Modern LLM deployments routinely compress models and raise sampling temperature to reduce cost, latency, or repetition, yet safety evaluations usually treat these choices as fixed implementation details. This leaves a practical uncertainty: does a model that is safe at FP16 and greedy decoding remain safe after it is quantized and sampled stochastically, or do the two deployment knobs amplify one another? We study this question with a factorial evaluation of 9 instruction-tuned models from six families, 3 precisions (FP16, GPTQ INT8, AWQ INT4), and 6 temperatures ($T{=}0$ to $1.0$), yielding 161 configurations and $\approx$322k responses judged by a six-model safety ensemble. Contrary to the concern that low-bit deployment broadly erodes alignment, standard non-adversarial quantization is usually safety-neutral: INT4 keeps or lowers attack success for 7 of 9 models, with clear degradation concentrated in the weakest baseline model, SmolLM3-3B ($18.5\%{\to}36.0\%$). The larger risk comes from sampling: higher temperature sharply increases decision instability for vulnerable models, with DFR reaching 53.0\% at $T{=}1.0$, even when average ASR changes modestly. Finally, the interaction is not a ``double penalty'': our Compound Degradation Index remains largely sub-additive ($-0.195$ to $+0.045$), indicating that quantization and temperature do not systematically compound. These results suggest a deployment rule of thumb: standard INT4/INT8 quantization can be reasonable for strongly aligned models, but safety claims at elevated temperature should report multi-sample stability, not only average attack success. 

---
# ScAle: Attention Head Scaling as a Minimal Adapter for Spatial Reasoning in Vision Language Models 

**Authors**: Rahul Chowdhury, Timothy A Rupprecht, Xuan Shen, Pu Zhao, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29579)  

**Abstract**: Spatial reasoning remains a persistent challenge for many vision language models (VLMs), and improving it typically requires fine-tuning with substantial additional parameters. Our preliminary analysis reveals that rescaling activations in selected transformer layers-without modifying pretrained weights-can significantly influence downstream performance. Motivated by this observation, we propose ScAle, an ultra-lightweight adaptation method that learns a small set of scalar coefficients to modulate last-token attention and MLP activations in a fully frozen backbone. We evaluate our method on the synthetic spatial reasoning benchmark SpatialEval and on real-world VQA datasets (COCOQA and VGQA) across multiple model families. Our method, ScAle, achieves up to 134.1% relative accuracy gains using only 1K trainable parameters without requiring millions of trainable parameters as in standard PEFT methods such as LoRA. Despite its extreme compactness, our approach recovers a substantial fraction of standard PEFT performance while preserving strong non-spatial VQA accuracy. These results demonstrate that bounded activation reweighting provides a simple, architecture-agnostic, and highly parameter-efficient alternative for adapting pretrained VLMs. 

---
# ReMAP-PET: Beyond Visual Understanding -- Learning Region-Guided Metabolic Alignment Semantics from Brain PET 

**Authors**: Dasen Dai, Yanteng Zhang, Shuoqi Li, Yuxiang Wei, Hongjie Yu, Qingxin Zhang, Qizhen Lan, Jagath C. Rajapakse, Vince D. Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2606.29577)  

**Abstract**: Positron Emission Tomography (PET) reveals brain metabolism and is clinically central to neurodegenerative disease assessment, yet existing 3D brain foundation models treat PET as generic volumetric data, missing the structured regional metabolic information that distinguishes it from structural neuroimaging. To address these limitations, we propose ReMAP-PET, a framework that moves beyond visual encoding by supervising a partially-tuned MedicalNet 3D ResNet-50 with brain regional standardized uptake value ratio (SUVR) profiles through joint regression and contrastive objectives, enabling the encoder to learn the metabolic semantics underlying PET modality. On 1015 paired PET--SUVR samples, ReMAP-PET achieves 0.070 SUVR MAE and 77.8% PET SUVR Recall@1, substantially outperforming five frozen pretrained baselines. We further connect the metabolic embedding to clinical language via contrastive alignment with frozen BioClinicalBERT and demonstrate end-to-end PET-to-report generation through SUVR-constrained verbalization. Linear probing on diagnostic classification and cognitive regression tasks confirms that the embeddings retain clinically relevant information without task-specific fine-tuning. Our results show that grounding PET encoders in regional metabolic semantics -- rather than treating PET as generic volumetric data -- yields representations that are structured, interpretable, and language-compatible, pointing to a new direction for metabolic-aware PET understanding. 

---
# TF-MoE: Time-Frequency Mixture-of-Experts for Efficient Speech Separation 

**Authors**: Qinzhe Hu, Chenda Li, Wangyou Zhang, Shujie Liu, Yan Lu, Yanmin Qian  

**Link**: [PDF](https://arxiv.org/pdf/2606.29575)  

**Abstract**: Recent advances in speech separation (SS) have led to compact front-end models with small parameter sizes, yet their high computational cost remains a major barrier for deployment on edge devices. To address this, we propose TF-MoE, a sparse Mixture-of-Experts (MoE) framework that enhances model capacity with almost no increase in inference cost. Our method introduces dynamic expert specialization in time and frequency dimensions through alternating time-wise and frequency-wise MoE modules, each dynamically selecting experts per frame or mel band. Built upon a mel-band-splitting Conformer backbone, TF-MoE achieves strong performance on SS tasks under low-compute settings. Experimental results demonstrate that TF-MoE consistently improves separation performance under computation cost constraints, outperforming BSRNN by +3.8 dB SDR on Libri2Mix with comparable 4.1 GMACs/s inference cost. This positions TF-MoE as a promising candidate for edge-device deployment. 

---
# Coverage-Driven KV Cache Eviction for Efficient and Improved Inference of LLM 

**Authors**: Shuvendu Roy, Mengyao Zhai, Hossein Hajimirsadeghi, Golnoosh Samei  

**Link**: [PDF](https://arxiv.org/pdf/2606.29563)  

**Abstract**: Large language models (LLMs) excel at complex tasks like question answering and summarization, thanks to their ability to handle long-context inputs. However, deploying LLMs is costly, not only due to the high computational demands of quadratic complexity of self-attention and auto-regressive generation, but also because of the significant memory overhead required for storing the key-value (KV) cache during inference. To reduce the memory cost, existing KV-cache eviction strategies leverage the sparsity in attention to selectively store a subset of tokens. While reducing the memory footprint, such approaches show a considerable drop in performance, especially in tasks that require long-context reasoning. We identify that the drop in performance is linked to a reduction in the coverage of unique tokens. Additionally, we theoretically show that reduced coverage limits the mutual information between inputs and outputs, thereby impairing predictive accuracy. To this end, we introduce K-VEC, a novel coverage-aware KV-cache eviction strategy that prioritizes token coverage while evicting tokens in the cache. K-VEC introduces a cross-head and a cross-layer coverage module to enhance token retention across attention heads and model layers, mitigating performance degradation caused by low coverage. Evaluated on 16 LongBench subsets, K-VEC exhibit up to 10.35 points improvement over the existing methods under the same eviction rate and memory constraint. Comprehensive evaluations validate the effectiveness of our approach and demonstrate its potential for efficient LLM deployment in resource-constrained settings. 

---
# VISTA-DZ: Visual Semantic Trajectory Adaptation for Personalized Dilemma Zone Prediction 

**Authors**: Chuheng Wei, Ziye Qin, Ziran Wang, Guoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29548)  

**Abstract**: Driver decision making in the dilemma zone at signalized intersections is safety critical, as vehicles approaching a yellow signal must decide whether to stop or proceed within limited time and distance margins. Accurate prediction of both stop-go decisions and decision timing is important for adaptive signal control, advanced driver assistance systems, and human-centered intelligent transportation applications. However, dilemma zone behavior is strongly driver dependent. Similar approach trajectories may lead to different decisions across drivers because of differences in risk preference, braking habit, and decision threshold. Existing personalized models often rely on handcrafted scalar descriptors, which provide useful but limited summaries of individual behavior. This paper proposes VISTA-DZ, a semantic-profile-conditioned framework for personalized stop-go and decision-time prediction. Historical trajectories are converted into visual representations, interpreted by a vision-language model to generate behavioral profiles, and encoded as semantic embeddings to condition a dual-output prediction network. The final model combines a bidirectional GRU encoder, driver-conditioned multi-head cross-attention, and Feature-wise Linear Modulation for temporal evidence selection and feature adaptation. Experiments on the SDZ dataset and a newly collected FDZ dataset show that VISTA-DZ outperforms trajectory-only and handcrafted personalization baselines, achieving 93.26% in-domain simulation accuracy and 90.22% mean accuracy across 20 held-out simulation drivers. Cross-domain results further show feasible zero-shot simulation-to-real transfer and better real-world generalization when simulation data are combined with limited field data. 

---
# Proteus: Automated Adversarial Robustness Testing for Audio Deepfake Detectors 

**Authors**: Nicolas M. Müller, Aditya Tirumala Bukkapatnam, Zohaib Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2606.29544)  

**Abstract**: We present Proteus, a framework developed at Resemble AI for automated robustness testing of our audio deepfake detection system. Given a detector, Proteus systematically searches over sequences of everyday audio transformations (codec transcoding, additive noise, reverberation, dynamic-range compression, and VoIP simulation) to find combinations that fool the detector while preserving speech quality. We propose two complementary search strategies: (1) a breadth-first search that exhaustively maps augmentation effectiveness across the parameter space, and (2) a Q-learning agent designed to efficiently discover deeper attack chains by exploiting structural patterns in the BFS data. We report findings from continuous deployment of Proteus against our production detector, showing that specific augmentation chains can reliably flip detection verdicts while preserving speech intelligibility and speaker identity. We discuss how these findings are used to harden the detector through targeted retraining. 

---
# Em-ergence of the em-dash: a population-level rise in em-dash frequency in medRxiv preprints at the dawn of the large-language-model era 

**Authors**: Przemysław Czuma  

**Link**: [PDF](https://arxiv.org/pdf/2606.29540)  

**Abstract**: Large language models (LLMs) can leave subtle stylistic traces in assisted text; one of the most cited is the em-dash (Unicode U+2014). Yet no one has measured whether em-dash use has changed in the scientific literature. This study, pre-registered on the Open Science Framework (HFT8C), used the full set of medRxiv full-text XML preprints from the official Text-and-Data-Mining resource. The primary cohort was first, original versions deposited 2020-2025 with an extractable Discussion section of at least 500 characters (N = 69,632). The primary endpoint was the presence of at least one em-dash in the Discussion; the principal measure was the absolute change in its prevalence between the pre-ChatGPT era (before 30 November 2022) and the post-ChatGPT era, estimated with a logistic model with standard errors clustered by first author. The analysis plan (six supporting analyses, six sensitivity analyses, two falsification tests) was frozen before any confirmatory result was computed. Em-dash prevalence in Discussion sections rose from 4.23% before ChatGPT to 11.58% afterward, an absolute increase of 7.35 percentage points (95% CI 6.94-7.77; odds ratio 2.96, 95% CI 2.77-3.17). The rise was not a sharp jump but a gradual, delayed acceleration: near 4% through 2023, 8.0% in 2024, and 20.3% in 2025. The effect survived every feasible sensitivity analysis (7.35-7.60 pp) and both falsification tests; a placebo split within the pre-LLM era showed no meaningful change (+0.13 pp, 95% CI -0.33 to +0.58), and was essentially absent in boilerplate sections. Independent LLM-associated lexical markers and within-paper section comparisons pointed the same way. The em-dash is a population-level indicator, not a per-paper detector of LLM use, and the design cannot establish causality; it shows that something in how scientific literature is written changed markedly in the early 2020s, and roughly when. 

---
# RESOURCE2SKILL: Distilling Executable Agent Skills from Human-Created Multimodal Resources 

**Authors**: Yijia Fan, Zonglin Di, Zimo Wen, Yifan Yang, Mingxi Cheng, Qi Dai, Bei Liu, Kai Qiu, Yue Dong, Ji Li, Chong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.29538)  

**Abstract**: Skills are a useful abstraction for software agents, turning human and agent experience into reusable procedural knowledge. Yet existing skill libraries are mostly hand-written, text-centric, or derived from agent traces, leaving tutorial videos and other multimodal human resources largely underused. We present RESOURCE2SKILL, a framework that distills multimodal resources, including tutorial videos, repositories, articles, and reference artifacts, into executable skills for software agents. RESOURCE2SKILL organizes these skills as a hierarchical multimodal Skill Wiki, where each entry combines structured text, code, visual examples, metadata, and provenance. This design preserves complementary signals from different resources: videos capture temporal operations and visual effects, code captures executable tool patterns, and articles or artifacts provide conceptual and stylistic grounding. At inference time, agents retrieve and compose relevant skills from the wiki; when coverage is insufficient, the same construction operator can acquire new skills online. Across seven practical authoring domains, RESOURCE2SKILL improves average overall score by +11.9 percentage points over no-skill agents and outperforms strong harness baselines in 26 of 28 main-aggregate model-domain cells. Ablations confirm the value of multimodal skill format, hierarchical organization, source diversity, selection strategy, and online acquisition. 

---
# SemJoin: Semantic Join Optimization 

**Authors**: Christopher Gou, Aditya Banerjee, Jiaxuan Wang, Chunwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29532)  

**Abstract**: Integrating unstructured data into relational database systems is increasingly important as demand grows for natural language querying and analysis. A semantic join, joining two tables under a natural-language predicate, can be evaluated with a large language model (LLM), but comparing every pair of tuples requires O(M x N) LLM invocations and is cost-prohibitive at scale. Existing systems reduce this cost but typically commit to a single fixed strategy (e.g., embedding similarity or one batched scheme) regardless of the data or the join predicate. We propose an LLM-agent-based decision pipeline that optimizes semantic joins by matching the execution strategy to the characteristics of the underlying tables. An LLM advisor routes each join to one of two strategies: a Cluster Join, which prunes candidates via unsupervised embedding clustering and sample-based filtering, or a Classifier strategy for predicates that reduce to a shared discrete label set. Across three diverse datasets (IMDb reviews, email contradictions, and Stack Overflow tags), the advisor consistently identifies the optimal execution strategy for each workload. This dynamic routing proves decisive: it outperforms adaptive block join (ABJ) by 20-33 F1 points across all datasets while consuming fewer tokens on two of the three, and achieves higher F1 scores than featurized-decomposition join (FDJ) at one to two orders of magnitude lower token cost. 

---
# MotionAtlas: Detailed Region Captioning for Motion-Centric Videos 

**Authors**: Weisong Liu, Haochen Wang, Kuan Gao, Yuhao Wang, Yikang Zhou, Zhongwei Ren, Jacky Mai, Anna Wang, Yanwei Li, Jason Li, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29531)  

**Abstract**: We propose MotionAtlas, a system for detailed captioning of motion-centric videos, comprising (1) a dedicated human-annotated benchmark, (2) a scalable, high-quality pipeline to construct training samples, and (3) a family of powerful Video-MLLMs. Unlike conventional global motion captioning datasets, we focus on region-aware motion captioning: given a video and a spatiotemporal mask, the model generates precise descriptions of motion within the target region, thereby alleviating visual clutter and motion entanglement and enabling reliable, quantifiable evaluation. Concretely, we first build MotionAtlas-Bench, a comprehensive benchmark comprising 2,073 multiple-choice questions, meticulously annotated for a curated set of high-quality, motion-centric videos, to evaluate fine-grained motion understanding of the objects in question. Second, we design a rigorous and scalable data pipeline that leverages self-bootstrap refinement to suppress fine-grained hallucinations, yielding 159k high-quality motion captioning data. Third, we design a tailored training data composition strategy, which achieves consistent and substantial performance gains across diverse baseline Video-MLLMs, including Molmo2 and Qwen3-VL. For instance, MotionAtlas-4B surpasses Qwen3-VL-4B by an average of 5.2 percentage points across general motion benchmarks. The benchmark, dataset, and code have been released. 

---
# SAKE: Software Architectural Knowledge Evaluation Benchmark for Large Language Models 

**Authors**: Tiziano Santilli, Francesco Daghero, Mayhar Tourchi Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2606.29520)  

**Abstract**: Large Language Models (LLMs) are increasingly used as assistants across the software development lifecycle, yet their ability to reason about software architecture remains largely unmeasured. Architectural decision-making depends on quality attribute trade-offs, design patterns, and system-level constraints, none of which are exercised by benchmarks that target syntactic or algorithmic tasks. We introduce SAKE (Software Architectural Knowledge Evaluation), a standardized and reproducible benchmark for assessing software architectural knowledge in LLMs. SAKE comprises 2154 expert-curated multiple-choice questions, each with four options, stratified across eight architectural categories and four context-length levels. We evaluate 11 proprietary and open-weight models in zero-shot and five-shot settings. Overall accuracy is high, but performance varies markedly across categories, revealing competency gaps in areas central to professional practice. SAKE, its evaluation scripts, and all results are released as open source to give the community a baseline for tracking architectural reasoning in LLMs. 

---
# The Verbose Context Problem in Medical Records 

**Authors**: Shiva Kaul, Min-Gyu Kim, Anjum Khurshid, Sriram Vishwanath  

**Link**: [PDF](https://arxiv.org/pdf/2606.29503)  

**Abstract**: The verbose context problem occurs when structured concepts have token-inefficient textual representations. This bottleneck is acute in population health: cohort-level analysis of longitudinal patient records requires reasoning over thousands of medically-coded events, often exceeding 400K tokens in total. We present PopMedQA, a benchmark isolating this problem through computational tasks on groups of longitudinal patient records. We construct the benchmark using neopatient, a new library for language-controlled generation of artificial patient records. Through extensive ablations -- including prompting strategies, prompt compression, and agentic decomposition -- we find that domain-independent methods fail to alleviate the verbose context problem. There remains significant opportunity to exploit domain-specific structure in language model inputs for population-scale reasoning. 

---
# Reported Confidence in LLMs Tracks Commitment More Than Correctness 

**Authors**: Dharshan Kumaran  

**Link**: [PDF](https://arxiv.org/pdf/2606.29490)  

**Abstract**: Confidence is an estimate of the probability that a chosen answer is correct. Verbal confidence reports are widely used as uncertainty measures in large language models, but whether they are best understood as estimates of correctness is unclear. We test this with a two-stage abstention paradigm from the neuroscience of perceptual decision making: a model first answers and reports its confidence, then decides whether to commit it to a user or abstain. Across four non-reasoning models, prompt framings, and confidence formats, verbal confidence predicted the commit/abstain decision substantially better than whether the answer was correct. Calibrated token log-probabilities showed the opposite profile, with abstention-prediction coupled to correctness discrimination, the signature of an answer-evidence signal. After removing the variance verbal confidence shared with log-probabilities, the residual stayed aligned with commitment while its link to correctness fell to near chance. The dissociation generalised to four reasoning models across four benchmarks of varying difficulty, from hard multiple-choice to frontier-level freeform questions. Mechanistic analyses in Gemma 3 and 4 were convergent: a post-answer state known to causally support verbal-confidence generation already encoded the future abstention decision before the abstention prompt, organised mainly by that decision rather than by correctness, the two lying in approximately orthogonal directions in activation space. Steering along a verbal-confidence-specific direction causally shifted abstention. Verbal and log-probability confidence are thus not interchangeable: log-probabilities track answer evidence and correctness, whereas verbal confidence is better understood as a behaviour-facing readout of an internal commit-readiness state, challenging the practice of treating verbal reports as proxies for reliability. 

---
# To Reason or to Fabricate: Reasoning Without Shortcuts via Hint-Anchored Pairwise Aggregation 

**Authors**: Jiuheng Lin, Chen Zhang, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.29481)  

**Abstract**: While reinforcement learning (RL) significantly enhances LLM reasoning, its efficacy is severely undermined by Pre-RL data overlap, where RL datasets overlap with pretraining or SFT corpora, causing models to exploit shortcuts by memorizing correct answers and fabricating post-hoc reasoning. To address this, we introduce HIPPO, a novel RL framework that integrates hint-injected aggregation with a tailored pairwise reward model. By utilizing hint injection to deliberately trigger overlap-induced behaviors, the resulting traces naturally serve as explicit anchors for pairwise comparison. This provides highly discriminable preference signals, enabling a lightweight judge model to reliably distinguish genuine reasoning deduction from shortcut-driven rationalization, while the pairwise formulation ensures stable and robust optimization compared to standard PRMs. Extensive experiments demonstrate that HIPPO yields substantial improvements over standard baselines and generalizes effectively to out-of-distribution general tasks, showing it extracts authentic, transferable reasoning skills rather than superficial shortcut patterns. 

---
# CRAFT: Counterfactual Credit Assignment from Free Sibling Rollouts for Self-Distilled Agentic Reinforcement Learning 

**Authors**: Zibin Meng, Kani Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.29476)  

**Abstract**: Self-distilled agentic reinforcement learning augments trajectory-level reward with a token-level distillation loss, using as its teacher the same policy conditioned on privileged context. The prevailing recipe gates this loss by a single scalar, the teacher-student log-probability gap. This signal is doubly limited: it is retrospective, scoring only the realised rollout and never the counterfactual ones, and it is sign-blind, never signalling when a teacher-preferred action would have harmed the trajectory. We introduce CRAFT, a three-pillar credit-assignment scheme that addresses both limitations. Pillar 1, Counterfactual Token Importance, reuses the G-1 sibling rollouts that GRPO already samples and importance-weights them by the log-probability gap to form a self-normalised estimate of the group-level counterfactual change in advantage from up-weighting teacher-preferred actions at each step; this yields a signed per-token credit at near-zero extra compute. Pillar 2 is an asymmetric controller that raises the distillation weight as it lowers the reference-KL weight along an exponential moving average of gate activity, and conversely. Pillar 3 polarises the KL penalty token by token, switching between a mode-seeking and a mode-covering update according to the sign of the credit. Each pillar has an independent switch that, when disabled, renders the loss and gradient byte-identical to the baseline in IEEE-754 arithmetic, so any measured gain is attributable to algorithmic change rather than implementation drift. We prove the estimator's consistency and a variance bound, give structural and bit-exact reproducibility guarantees, and evaluate CRAFT across three agentic environments, four model scales, and five end-to-end methods, plus two tabulated prior-work baselines. Among these is Adaptive-CRINGE, a comparator sharing Pillar 2 with CRAFT, isolating the counterfactual contribution. 

---
# A Posteriori Error Analysis for Decoupled Neural Approximations of Fully Coupled FBSDEs with Control Mismatch 

**Authors**: Xichuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29474)  

**Abstract**: This paper develops an a posteriori error analysis framework for decoupled neural approximations of fully coupled forward--backward stochastic differential equations (FBSDEs). It provides an a posteriori error-analysis for the idealized discrete adapted trajectory. The main feature of the proposed formulation is the use of an auxiliary control process in the forward coefficients, which may differ from the backward component approximated by the neural network. This decoupling is useful in practical deep learning implementations, but it creates a control mismatch that must be included in the error analysis. We first establish a continuous-time stability estimate for fully coupled FBSDEs under perturbations of the drift, diffusion, generator, terminal condition, and auxiliary control input. We then transfer this estimate to the discrete-time setting and derive computable a posteriori error bounds depending only on the terminal defect, the pathwise residual, and the control mismatch. When the auxiliary control is identified with the backward approximation, the mismatch term vanishes and the bound reduces to the standard two-term form. Numerical experiments on a linear--quadratic FBSDE with an explicit reference solution and a multidimensional Burgers-type FBSDE without a reference solution illustrate the diagnostic role of the proposed indicators and the contribution of the mismatch penalty to the consistency and reproducibility of the numerical approximations. 

---
# Rank-Aware Hyperbolic Alignment for Vision-Language Dataset Distillation 

**Authors**: Jongoh Jeong, Sun-Kyung Lee, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2606.29464)  

**Abstract**: Vision-language dataset distillation (VLDD) compresses a large image-text paired dataset into a small set of synthetic pairs that can efficiently train contrastive vision-language models under strict data and compute budgets. Most existing methods match expert trajectories or cross-modal statistics, yet still enforce full-dimensional alignment in a Euclidean embedding space. This is often overly restrictive due to rank-deficient image--text correlation, with shared semantics concentrated in a low-dimensional range and remaining variation spread across a weakly correlated residual subspace. LoRS relaxes alignment at the similarity level by low-rank factorization, but does not explicitly control dominant alignment capacity and structure in the representation space. We thus propose a rank-aware hyperbolic alignment (RAHA) that combines hierarchical geometry with explicit alignment-capacity control. RAHA lifts multimodal representations to hyperbolic space and optimizes distilled pairs with asymmetric objectives that enforce geodesic alignment in the shared range while regularizing the residual subspace to preserve modality-private diversity and improve transfer robustness. Experiments on benchmarks show that RAHA demonstrates competitive cross-modal retrieval and improved transfer indicators under fixed budgets. 

---
# Interpretable Inverse Design of Metal-Organic Frameworks with Large Language Model Agents 

**Authors**: Kyungmin Nam, Seunghee Han, Jihan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.29459)  

**Abstract**: Inverse design of metal-organic frameworks (MOFs) requires searching a combinatorially vast space where property labels are expensive and most machine-learning models reveal little about why a structure succeeds. We introduce LLM4MOF, a closed-loop framework in which language-model agents reason about chemistry, build candidate MOFs, and test them in simulation, refining hypotheses over ten autonomous iterations. One agent proposes interpretable design hypotheses over metal nodes, linkers, pore geometry, and functional chemistry, and a second translates them into constraints that select candidate MOFs, each made of a metal node, organic linker, and matching topology. Each hypothesis is tested through four diagnostic beams that apply different subsets of its constraints, so comparing them shows whether geometry, chemistry, or metal choice drives performance. Even when blind to the global property landscape of databases, LLM4MOF concentrates its search on top-performing structures across six adsorption, separation, and electronic-structure tasks within 400 property evaluations. The same loop also generates new MOFs de novo and validates them in live simulation, where it adapts the geometry to each requested condition, outperforming random search and a genetic algorithm at roughly $1 per campaign. LLM4MOF shows that language-model agents can run interpretable, simulation-grounded inverse design without training a model per objective. 

---
# Resonant Brane Splatting for Arbitrary-Scale Super-Resolution 

**Authors**: Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto  

**Link**: [PDF](https://arxiv.org/pdf/2606.29453)  

**Abstract**: Arbitrary-Scale Super-Resolution (ASR) reconstructs images at continuous magnification factors. Recent methods accelerate inference by replacing computationally heavy implicit neural decoders with explicit 2D Gaussian Splatting (GS). However, since standard Gaussians are smooth low-pass primitives, modeling edges and fine textures requires multiple overlapping, well-aligned splats, which creates severe bottlenecks during rasterization. To address this, we introduce Resonant Brane Splatting (RBS), a feed-forward ASR framework. RBS replaces flat Gaussians with Branes: expressive primitives that emit spatially varying colors to natively model local contrast and complex textures within a single footprint. We achieve this by augmenting the standard Gaussian envelope with internal Gaussian-Hermite modes, assigning a distinct color coefficient to each. The zero-order mode recovers standard GS, while higher-order modes capture high frequencies. We predict Brane parameters directly from low-resolution features. Because Branes provide a mathematically richer formulation than simple Gaussians, far fewer primitives need to overlap to reconstruct a given target pixel. To exploit this, we introduce an efficient fully differentiable rasterizer with a precise culling strategy based on the classical quantum turning point. This allows us to safely skip negligible regions, drastically reducing the rendering overhead. Experiments on standard ASR benchmarks show that RBS improves reconstruction quality over implicit and GS baselines, while achieving superior speed-quality trade-off than prior GS methods. 

---
# Bridging VideoQA and Video-Guided Agentic Tasks via Generalized Keyframe Extraction 

**Authors**: Sunqi Fan, Qingle Liu, Runqi Yin, Meng-Hao Guo, Shuojin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29445)  

**Abstract**: Video understanding is a fundamental capability for multimodal intelligence, and recent Multimodal Large Language Models (MLLMs) have achieved remarkable performance on Video Question Answering (VideoQA) benchmarks. However, existing benchmarks primarily evaluate whether models can perceive shallow visual cues, while rarely examining whether MLLMs can learn deeper knowledge or procedural skills from video tutorials and generalize them to downstream long-horizon agentic tasks. To address this gap, we introduce VG-GUIBench (Video-Guided GUI Benchmark), a new benchmark designed to evaluate whether MLLM-based GUI agents can follow video tutorials to complete corresponding GUI interactive tasks. Furthermore, we observe that the performance of models on both VideoQA and video-guided agentic tasks critically depends on effective keyframe extraction. Based on this observation, we propose TASKER (Task-driven And Scene-aware Keyframe searchER), a keyframe extraction algorithm that jointly considers task relevance and scene dynamics to identify informative frames. Experimental results demonstrate that TASKER achieves significant performance improvements on both VideoQA and video-guided agentic task benchmarks, outperforming the best baseline by 2.0% on the EgoSchema fullset and 1.8% on the NExT-QA dataset, respectively. These results further highlight the potential of generalized keyframe extraction methods for video understanding tasks. Our code and data are available at this https URL. 

---
# Closing the Activation-Cone Blind Spot: Response-Time Probing and Unified Defense 

**Authors**: Subhadip Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2606.29441)  

**Abstract**: Inference-time safety methods for large language models have proliferated, yet no systematic comparison exists. We evaluate five defense paradigms (no defense, static steering, CAST, AlphaSteer, probe-gated) across seven instruction-tuned models (7-31B) and five attack types (GCG, AutoDAN, DeepInception, prefilling, intent laundering). Our central finding: prompt-time activation defenses are structurally blind to prefilling attacks. AlphaSteer achieves 0% attack success on GCG, AutoDAN, and intent laundering but 50% on prefilling. We prove a corollary: any defense that gates intervention on a single layer's activation alignment with a benign reference (cone, subspace, or null-space) is blind to attacks that craft activations to lie inside that reference, whether checked at prompt time or per token. As its constructive contrapositive we introduce response-time probing: a linear probe on the model's hidden state at the first generated tokens, with AUROC 0.97-1.00 across all seven models. Combined with a halt, it cuts prefilling attack success to 0/40 on every model with 0% benign false positives, outperforming Llama Guard 3. Cross-template generalisation depends on probe depth, so we scope the claim to the canonical prefilling-template family. Composing the response-halt with AlphaSteer's null-space steering gives an orthogonal split (the halt catches prefilling, AlphaSteer catches semantic attacks), reaching defense success 0.983 on Mistral and 0.994 on Llama and dominating both components. We further show MMLU fails to capture steering's true utility cost, which appears as behavioral hedging rather than factual loss, and that diverse negative training sets cut probe false positives from 80-100% to near zero. Code, attacks, per-sample results, and the judge prompt are released. 

---
# LLMography: Transforming Human-AI Conversations into Traceability, Oversight, and Auditability Indicators 

**Authors**: Mohammed Bousmah  

**Link**: [PDF](https://arxiv.org/pdf/2606.29437)  

**Abstract**: The growing use of Large Language Models (LLMs) in education, software engineering, academic writing, and technical documentation raises a key question: how can we evaluate not only AI-assisted outputs, but also the interaction process that produced them? Current debates often focus on detecting whether a final artifact was generated by AI, while overlooking the conversation history that reveals human direction, AI contribution, corrections, validation, and traceability.
This paper introduces LLMography, a framework for transforming Human-AI conversations into measurable indicators of provenance, human contribution, AI dependency, reproducibility, and auditability. By analogy with bibliography and webography, LLMography documents the dynamic trajectory of interaction between a human and a Large Language Model as a structured trace of Human-AI co-production.
We present a prototype that analyzes Human-AI conversation traces and generates KPI reports including Prompt Quality Score, Human Direction Score, AI Dependency Level, Auditability Score, Final Output Traceability, Privacy Risk Level, and a recommended LLMography label. A preliminary exploratory evaluation was conducted on 19 anonymized audit reports from engineering students. Most interactions were classified as Human-AI co-produced, with average scores of 86.8/100 for Human Direction, 81.9/100 for Prompt Quality, 72.8/100 for Auditability, and 77.1/100 for Final Output Traceability.
The paper also applies LLMography to its own writing process, classified as human-originated, human-directed, AI-assisted co-production. The findings suggest that AI transparency should move beyond output detection toward documenting the history of interaction. 

---
# Can Machines Really See Objects in Images? A Study Based on Syntactic Distance and Visual Self-Referential Instances 

**Authors**: Xingyu Peng, Junran Wu, Yue Hou, Zhongliang Qiao, Jiaheng Liu, Shangzhe Li, Jichang Zhao, Wenjun Wu, Xianglong Liu, Yongxin Tong, Li Dong, Ke Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29416)  

**Abstract**: Can a vision model truly see an object, or does it only fit surface-level visual cues? Following Wittgenstein's view that the limits of language are the limits of the world, we view a model's recognition ability as bounded by the descriptive system it has learned. In current vision models, this system is often realized through learned feature representations that exploit local statistical cues. We therefore ask whether a model can still classify correctly when such local cues provide no stable basis for distinction. We formalize this question with syntactic distance, which measures class separability through the symmetry of the operations mapping one class to the other: positive distance exposes exploitable local features, whereas zero distance requires global semantics rather than local rules. We construct a visual self-referential task in maximum-variance binary noise: positive samples contain a closed square, while negative samples contain an otherwise identical square with one flipped boundary pixel. The two classes differ in global semantics but have zero syntactic distance, making local statistical shortcuts unreliable. Experiments on ResNets and Vision Transformers reveal a consistent phase-transition phenomenon, with accuracy collapsing to random guessing once the image scale crosses a critical point and does not recover within the tested range. Larger training sets and models only delay this collapse, while globally attentive ViTs reach it earlier. These results reveal a structural capability boundary of current architectures on global-concept tasks, suggesting that general intelligence may require creating new language, not reusing an existing one. 

---
# LC-ICL: Label-Guided Contrastive In-Context Learning for Robust Information Extraction 

**Authors**: Xiao You, Tianwei Yan, Shan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.29407)  

**Abstract**: There has been increasing interest in exploring the capabilities of advanced large language models (LLMs) in the field of information extraction (IE), specifically focusing on tasks related to named entity recognition (NER) and relation extraction (RE).Although researchers are exploring the use of few-shot information extraction through in-context learning with LLMs, they tend to focus only on using correct or positive examples for demonstration, neglecting the potential value of incorporating incorrect or negative examples into the learning this http URL this paper, we present LC-ICL a novel few-shot technique that leverages both correct and incorrect sample constructions to create in-context learning demonstrations. This approach enhances the ability of LLMs to extract entities and relations by combining positive samples with negative samples annotated by error-cause labels. These labels expose more detailed error features in erroneous examples, enabling the model to understand why similar predictions fail and avoid repeating such errors during this http URL, our proposed method taps into the inherent contextual information and valuable information in hard negative samples and the nearest positive neighbors to the test and then applies the in-context learning demonstrations based on LLMs. Our experiments on various datasets indicate that LC-ICL outperforms previous few-shot in-context learning methods, delivering substantial enhancements in performance across a broad spectrum of related tasks. These improvements are noteworthy, showcasing the versatility of our approach in diverse scenarios. 

---
# Self-Organized Conformal Prediction: Reducing Regional Coverage Gaps with Unsupervised Group Discovery 

**Authors**: Louis Berthier, Ahmed Shokry, Maxime Moreaud, Guillaume Ramelet, Aymeric Dieuleveut  

**Link**: [PDF](https://arxiv.org/pdf/2606.29403)  

**Abstract**: Conformal prediction guarantees marginal coverage, but pooled calibration averages over heterogeneous regions and can mask regional undercoverage in safety-critical subgroups. We introduce Self-Organized Conformal Prediction (SOCP), a calibration scheme that discovers input-space groups with a Self-Organizing Map (SOM) and, at test time, draws a local calibration buffer from the query's best-matching unit (BMU) cell or a fixed grid neighborhood. The same retrieval rule applies to regression and classification tasks across tabular features and image embeddings, leaving the predictor and nonconformity score untouched. SOCP gives exact validity for BMU-cell retrieval and fixed retrieved-set validity for neighborhood buffers; central-cell validity for neighborhood retrieval holds up to a Kolmogorov-Smirnov (KS) bias term. A split-routed extension recovers fixed retrieved-set validity conditional on the routing split. On eight regression and classification benchmarks, SO-SCP reduces the weighted regional coverage gap on $7/8$ datasets (mean paired change $-7.1\%$) for a mean prediction-set size increase of $6.2\%$, with negligible overhead on the largest six datasets; SO-CQR yields smaller gains, since quantile regression already absorbs much of the heterogeneity. By learning groups directly from the input geometry, SOCP provides group-local calibration with exact fixed-group guarantees and approximate central-cell guarantees, without supervised partitions or predictor retraining. 

---
# Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution 

**Authors**: Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto  

**Link**: [PDF](https://arxiv.org/pdf/2606.29400)  

**Abstract**: In computer graphics, visual content is continuously warped, zoomed and resampled. This occurs when engines upscale frames, users zoom into 3D scenes, or foveated VR applies varying scaling. Handling these transformations requires Arbitrary-Scale Super-Resolution (ASR). Traditional models, designed for fixed scales, typically predict at a lower integer scale (e.g., x4) and rely on sub-optimal interpolation for continuous resolutions, compromising quality. Furthermore, most methods process pixels uniformly. Since fine details are sparse, this creates overhead; efficiency dictates concentrating resources only where structural complexity demands it. While implicit models and Gaussian Splatting (GS) enable continuous representation, GS is advantageous due to adaptive densification. However, transitioning GS into a feed-forward model for ASR is non-trivial. Standard GS optimization needs high-resolution gradients to drive primitive growth, which are unavailable during inference. Thus, the network must autonomously predict GS densification from low-resolution inputs. To solve this, we propose QuADA-GS. After encoding inputs into a latent space, a Neural Routing Architecture evaluates local complexity to distribute a global budget, assigning specific upsampling factors to features to avoid redundant processing. Features are dynamically densified based on these factors, forming an irregular topology decoded into 2D Gaussian primitives. To coordinate features before decoding, we introduce Hierarchical Pointer Convolution. This non-grid operator achieves O(1) neighbor lookup complexity, facilitating efficient spatial communication and bypassing dense bottlenecks. Experiments show QuADA-GS achieves state-of-the-art ASR performance, maintaining low latency and a lean memory footprint. 

---
# DR-GS: Physically-Based Deformable and Relightable 2D Gaussians 

**Authors**: Jiaxin Li, Tong Wu, Yi Wei, Tailin Wu, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29379)  

**Abstract**: Gaussian splatting (GS) has garnered significant attention in VR/AR and digital content creation due to its explicit parameterization and efficient rendering capabilities. However, existing GS-based methods for deformable objects face two key limitations: (i) illumination is erroneously baked into textures, causing physically inconsistent responses under dynamic deformations and lighting changes; (ii) snapshot-based reconstruction restricts post-reconstruction material editing. To address these challenges, we propose Deformable and Relightable GS (DR-GS), a unified Gaussian framework that integrates physically-based inverse rendering, relighting, and deformation-aware manipulation. Through explicitly disentangling geometry, illumination, and material representations, DR-GS overcomes the limitations of static snapshots, resolving unrealistic appearance under varying conditions while enabling post-reconstruction parameter editing. Extensive experiments show that DR-GS achieves leading visual quality across static reconstruction, dynamic deformation, and relighting, reliably preserving reflections and specular highlights on glossy surfaces. It further establishes a fully decoupled geometry-illumination-material pipeline, enabling high-quality 3D asset creation and comprehensive post-editing. 

---
# Solver-Verified Formulation Generation and Selection for Multi-Warehouse Inventory Allocation Using Large Language Models 

**Authors**: Jintao Xu, Yingzheng Ma, Jiong Dong, Yongzhi Qi, Jianshen Zhang, Dongyang Geng, Anni Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29366)  

**Abstract**: Balance-oriented multi-warehouse inventory allocation is a recurring decision problem in large-scale e-commerce supply chains, in which a fixed replenishment quantity is distributed across warehouses to balance post-allocation inventory coverage while accounting for demand forecasts and heterogeneous allocation constraints. In practice, allocation requirements are often scenario-dependent and expressed in semi-structured or natural-language form rather than as ready-to-solve operations research (OR) formulations. We propose an OR-guided Large Language Model (LLM) for Allocation (ORLA) that uses solver feedback to generate, verify, and select OR formulations. ORLA integrates automatic "Problem-Model-Code (PMC)" generation, learning-based formulation selection, and feasibility restoration. We develop three complementary mixed-integer programming formulation families based on deviation minimization, soft band compliance, and knapsack-inspired allocation, together with solver-ready mixed-integer linear programming reformulations, modular constraint extensions, and a penalty-based relaxation mechanism for infeasible cases. The LLM component generates candidate formulations and executable solver code from textual or semi-structured specifications, while the solver provides verification signals for executability, feasibility, and solution quality. To address instance heterogeneity, ORLA estimates the expected quality of candidate formulations, selects promising candidates, and combines their outputs through score-aware aggregation. Experimental results on 29 production evaluation batches from this http URL show that the best single OR formulation improves allocation accuracy by 3.4 percentage points over the incumbent approach, while the full ORLA framework achieves a 4.5 percentage-point overall improvement and improves allocation accuracy in 26 of the 29 evaluation batches. 

---
# Dynamic Parsing and Updating Natural Language Specification using VLMs for Robust Vision-Language Tracking 

**Authors**: Xiao Wang, Liye Jin, Dan Xu, Yuehang Li, Lan Chen, Yaowei Wang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29357)  

**Abstract**: Vision-language tracking guided by natural language specifications leverages high-level semantic cues of target objects to substantially boost tracking accuracy and robustness. Existing studies have verified that adaptively optimizing textual descriptions throughout the tracking process can effectively mitigate the semantic-visual mismatch induced by dynamic variations in target appearance, position, and other inherent attributes. Nevertheless, mainstream methods that directly generate textual information via sequence models or large language models inevitably suffer from inherent defects, including erroneous target updating, excessive background distraction, and pervasive hallucination artifacts. To address the aforementioned limitations, this paper proposes a novel language dependency parsing mechanism to precisely distill core tracking principal components, encompassing target objects, semantic concepts, and background contextual information. On this basis, we perform component-aware adaptive textual description updates by exploiting the powerful cross-modal understanding capability of the pre-trained vision-language model Qwen-VL. By integrating the proposed elaborately designed modules into the baseline framework, our method achieves consistent and superior tracking performance on multiple large-scale vision-language tracking benchmarks, including TNL2K, LaSOT, TNLLT, and OTB-LANG. The source code and pre-trained models will be released at this https URL. 

---
# Fast Enough to Act: Spatio-Temporal Visual Token Merging for Low-Latency Robotic VLMs and VLAs 

**Authors**: Junzhou Chen, Jindong Wang, Gang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.29350)  

**Abstract**: Vision-language models and vision-language action models endow the robot with unprecedented capabilities. However, the input of video and high-resolution images yields a massive number of visual tokens, leading to extremely high inference latency and severely hindering the robot's real-time control. To break through this computational bottleneck, we propose ST-Merge, a plug-and-play, training-free framework that efficiently fuses redundant tokens directly during the visual encoding phase. By explicitly constructing 3D spatiotemporal coordinates, it employs a multi-queue parallel matching and weighted aggregation mechanism to achieve efficient and geometrically consistent fusion of redundant tokens across frames. In addition, we introduce a post-merge positional correction mechanism that effectively eliminates spatial deviation caused by merging by dynamically re-evaluating the rotational position code of the weighted centroid of the vision token, thereby ensuring the high-precision spatial awareness required for dexterous operation. In the Video Question Answering task on the mainstream VLM, Qwen2.5-VL, ST-Merge achieves a 2$\times$ inference speedup with only a tiny 1\% loss in precision. When deployed on the $\pi_{0.5}$ VLA policy, ST-Merge achieves an 8.3$\times$ speedup at 1024 $\times$ 1024 resolution and matches the baseline success rate at this high-resolution setting. At lower resolutions, it introduces a small drop in accuracy. 

---
# Adaptive Financial Transformer with Regime-Gated Attention for Stock Return Prediction 

**Authors**: Dishan Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2606.29347)  

**Abstract**: Adaptive Financial Transformer (AFT) is proposed for stock return prediction under non-stationary financial markets. The model incorporates a Market Regime Encoder, an Adaptive Gate Network, and an Adaptive Financial Context module to dynamically bias self-attention based on semantic relationships between financial indicators. Unlike conventional Transformer architectures that treat all input features uniformly, the proposed approach groups 95 engineered financial features into 11 semantic categories and adapts attention according to latent market regimes. The study also identifies and corrects sequence alignment and backtesting issues that can inflate reported trading performance, and introduces a financially-aware composite objective that jointly optimizes prediction error, directional accuracy, and non-overlapping Sharpe ratio. Extensive experiments compare the proposed architecture against classical machine learning models, recurrent neural networks, and Transformer baselines using chronological evaluation, five random seeds, ablation studies, hyperparameter optimization, explainability analysis, and multi-stock validation. Results demonstrate competitive predictive performance while reducing model complexity by 15.2% and improving parameter efficiency through feature selection, providing an interpretable Transformer architecture for financial time-series forecasting. 

---
# AMR: Adaptive Modality Routing for Multimodal Polyglot Speaker Identification 

**Authors**: Chuxiao Zuo, Yao Zhu, Minqiang Xu, Manhong Wang, Yunke Zhang, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29335)  

**Abstract**: Multimodal speaker identification systems face two key challenges in real-world deployment: missing modalities and language mismatch between training and testing conditions. In practical scenarios, background multi-speaker conversations, ambient noise, and overlapping speech further degrade identification accuracy. To address these challenges, we propose a multimodal polyglot speaker identification system for the POLY-SIM 2026 Grand Challenge. The system is fundamentally built upon Adaptive Modality Routing(AMR), a modality fusion module that dynamically assesses per-sample input quality and integrates modality information. Specifically, AMR employs two modality adapters to process the embeddings extracted from a linguistically robust audio encoder(W2V-BERT 2.0) and a large-scale pretrained face encoder(IResNet-18), producing modality-adapted embeddings. Based on these adapted embeddings, a trainable router estimates dynamic modality weights, which are subsequently applied to aggregate the modality-specific logits for the final prediction. To optimize this routing mechanism, we adopt a modality-aware training strategy that constructs four types of sample pairs to simulate diverse input conditions, with KL divergence serving as explicit supervision for weight assignment. Experimental results on the POLY-SIM 2026 evaluation set show that the proposed system achieves identification accuracy of 99.93%(English multimodal, P3), 100.00%(Urdu multimodal, P5), 97.50%(English audio-only, P4), and 98.83%(Urdu audio-only, P6). The average accuracy across all four protocols is 99.07%, surpassing the Fusion and Orthogonal Projection(FOP) baseline by 32.73%. 

---
# Covering the Unseen: Information Demand Coverage Optimization for Retrieval-Augmented Generation 

**Authors**: Bingxue Zhang, Jianying Jia, Feida Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29328)  

**Abstract**: Retrieval-augmented generation (RAG) typically treats context selection as ranking chunks against a single query embedding. This assumption breaks down for complex queries, such as multi-hop or ambiguous questions, where top-k selection tends to over-cover one semantic aspect while ignoring critical sub-questions. We propose GeoRAG, which recasts context selection as Information Demand Coverage Optimization. GeoRAG builds a multi-dimensional demand distribution through diverse sub-query generation and reverse-validation weighting, then selects context by minimizing the Sinkhorn-Wasserstein distance between this demand distribution and the coverage of the selected set. The resulting demand-weighted facility-location objective is monotone submodular, giving a $1-1/e$ greedy guarantee, which we approximate with a Sinkhorn-based marginal-gain surrogate. The method is unsupervised, training-free, and retrieval-agnostic. We further show that single-point, query-proximity scorers cannot cover multi-modal demands, exposing a structural limit of ranking-based selection. On six open-domain QA benchmarks, GeoRAG improves exact match (EM) by +6.5 to +7.5 points over top-k truncation (up to +9.7 on HotpotQA and ASQA) and outperforms strong baselines including MMR, DPP, BGE-Reranker, SMART-RAG, and AdaGReS, with stable gains across context budgets and sub-query generators. 

---
# Deterministic Decisions for High-Stakes AI. A Zero-Egress Pipeline with the Deployability of RAG and the Accuracy of Machine Learning 

**Authors**: Craig Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2606.29280)  

**Abstract**: We identify intervention bias as a previously unquantified failure mode of zero-shot large-language-model (LLM) educational advisory agents: without task-specific training, they recommend action when a hindsight-optimal oracle policy mandates inaction. In a six-arm ablation on the Open University Learning Analytics Dataset (N=800 students, four temporal cutoffs), at day 56 -- when the oracle designates 70.1% of students as needing no intervention -- zero-shot GPT-4o recommends action for 73%, a 43 percentage-point false-positive rate. Commercial RAG and SQL-augmented retrieval are comparably miscalibrated; at 10,000 students this implies about 4,300 unnecessary advisor contacts per cycle.
Supervised policy learning eliminates this bias: a trajectory-conditioned ONNX Decision Transformer (DT) and a snapshot XGBoost classifier, trained on the same oracle-labelled trajectories under strict prefix-only features, both achieve near-zero calibration error. The DT reaches macro-F1 0.79 (macro-recall 0.85) across all five action classes, predicting even the rare load-reduction action without collapsing, at a 0% action flip rate and sub-5 ms CPU decision latency. The two supervised arms are on par; the DT's edge over XGBoost at the final cutoff is indicative only (unpaired across cohorts).
Scope: we validate Stage-2 decision-making (EAV state vector to supervised policy) under controlled oracle input from structured OULAD data; high fidelity reflects feature-oracle alignment, not general high-stakes-AI capability. The most robust finding is the intervention-bias contrast, not the absolute accuracies. We also show an Evaluation Gap: LLM-as-judge scoring (DeepEval G-Eval) is blind to intervention bias, rewarding fluent over-prescription rather than decision quality. 

---
# Manufactured Confidence: How Memory Consolidation Turns Hearsay into Confident Facts 

**Authors**: Alex Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2606.29279)  

**Abstract**: LLM agents carry conclusions across steps and sessions in compressed memory, and memory products (e.g., mem0, LangMem) rewrite conversation into stored "facts" that later steps trust. We show this rewriting manufactures confidence: across our constructed agent settings, a casual, hedged remark becomes a confident, dated assertion the agent then obeys like a verified fact, granting every above-clearance request it faces. No attacker is needed: a role that was true once and never corrected is stored as a flat fact and acted on like a deliberate injection. We then isolate what the agent responds to. It is not the source: attributed, unattributed, and even forged "system of record" claims all grant alike. It is the confidence of the phrasing. A hedge is discounted, a flat assertion is obeyed, and this holds with no special keyword. Not all hedges are equal, though: the evidential register is the least-discounted, with "reportedly" obeyed like a flat assertion on most models. The obvious fixes fail. A passive "unverified" tag is ignored, and an active "do not trust this" instruction escalates even correct memory, so it is safe only by refusing to decide. The real fix lives in the store: keep the tentative phrasing rather than upgrade it. But that is hygiene, not a defense against an attacker who can simply write a confident lie. The deployable lesson is narrower and constructive: a single load-bearing memory is the hazard, and one redundant source restores correct decisions. We release the harness and demonstrations. 

---
# A Hybrid Framework for Song Lyric Annotation Based on Human-LLM Alignment 

**Authors**: Rashini Liyanarachchi, Frank Tran, Md Mahmudul Hasan, Aditya Joshi, Erik Meijering  

**Link**: [PDF](https://arxiv.org/pdf/2606.29273)  

**Abstract**: Emotion recognition of song lyrics is a challenging task since lyrics may not necessarily align with the overall emotion of a song. As a result, lyrics annotation remains largely underexplored. Drawing inspiration from research in large language model (LLM) assisted annotation, we examine the alignment between humans and LLMs for annotation of lyrics by creating a new sentence-level dataset of lyrics. Our observations highlight the subjectivity of the task and the inherent challenges. Following this, we present a hybrid annotation framework that optimizes human and LLM annotation by predicting potential misalignment in annotation. 

---
# Confidence-feedback-weighted graph matching network: online-offline laser-induced damage site matching under complex interference 

**Authors**: Yueyue Han, Guanhua Chen, Hangcheng Dong, Kang Zhang, Fengdong Chen, Zhitao Peng, Fa Zeng, Qihua Zhu, Guodong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29255)  

**Abstract**: Online inspection images of final optics in high-power laser facilities contain pseudo-damage sites that closely resemble true damage sites. Determining the authenticity of online-detected sites is therefore difficult and requires accurate matching to offline ground-truth sites. However, this matching remains highly challenging due to limited match-discriminative features, local geometric distortions, and numerous distractor sites. Existing matching models mainly suppress distractors implicitly through loss-function supervision. We propose a confidence-feedback-weighted graph matching network that requires only damage-site centroid coordinates as input. It estimates node matchability confidence from each round of matching scores and feeds it back as a reliability weight to guide subsequent edge-feature aggregation, thereby suppressing distractor propagation and enhancing cross-graph discriminability. Within this framework, a geometric consistency constraint calibrates spurious high-confidence matchability estimates, while a hard-example mining loss improves discrimination between structurally similar sites. Experiments on our Complex-Scene dataset show that the proposed method achieves a matching F1-score of 96.36$\%$ with robust and efficient performance. 

---
# MoPe: Motion Permanence for Robust Monocular Gaussian Mapping in Dynamic Environments 

**Authors**: Qixin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.29237)  

**Abstract**: Robust robot autonomy depends on scene representations that remain stable enough to support localization, navigation, and downstream decision making in dynamic environments. Monocular Gaussian Splatting SLAM provides high-fidelity mapping, but current uncertainty-aware methods still treat dynamic regions largely as per-frame observations. This makes the representation effectively memoryless: when a pedestrian slows, pauses, or reappears after occlusion, the current frame may look static, allowing dynamic content to be absorbed into the map and leaving persistent ghosting artifacts. We argue that this failure reflects a representation-level mismatch. Dynamic-ness is not an instantaneous appearance property, but a temporal property defined by motion history. Building on this view, we introduce Motion Permanence: the principle that an object's dynamic identity should persist over time rather than be re-decided from each frame independently. We realize this principle in MoPe, a memory-aware uncertainty filter for monocular Gaussian mapping. MoPe propagates the historical dynamic posterior through geometry-consistent SE(3) warping and fuses it with current-frame evidence using bounded Bayesian log-odds updates. The resulting persistent posterior guides tracking, mapping, dynamic-aware Gaussian insertion, and Gaussian-level post-cleanup. On Wild-SLAM, Bonn, and TUM sequences, MoPe improves tracking robustness and reduces residual ghosting, with the strongest gains on dynamic-human scenes that most directly violate the memoryless assumption. These results show that maintaining temporal dynamic state inside the scene representation is a practical step toward more reliable representation-centric autonomy in changing real-world environments. 

---
# AnyBody: Free-Form Whole-Body Humanoid Control from Arbitrary Keypoint Guidance 

**Authors**: Shuning Li, Sikai Li, Jiachen Li, Mingyu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.29209)  

**Abstract**: We present AnyBody, a unified whole-body humanoid controller driven by an arbitrary subset of body keypoints chosen at deploy time. Prior physics-based trackers either rely on expensive full-body motion capture and error-prone trajectory retargeting, which bottleneck scalable data collection and policy learning, or decompose upper- and lower-body control into separate hierarchical representations, sacrificing the coordinated whole-body motions that loco-manipulation requires. We close this gap by learning a single latent motion representation that any keypoint subset can address. To achieve this, we first train a privileged teacher tracker on a large unstructured motion corpus and distill it online into a deterministic encoder-decoder student whose latent space is a unit sphere. We then train a transformer keypoint encoder that admits any subset of body keypoints through masked self-attention, aligning it to the privileged latent. Additionally, we treat the frozen decoder as a motor prior and specialize downstream tasks with a lightweight residual corrector in the latent space. We demonstrate the effectiveness of AnyBody by tracking large-scale human motions from arbitrary keypoint subsets, free-form control, flexibly teleoperating, and learning downstream behaviors including locomotion, in-air writing, and obstacle-reach. 

---
# Behavior Uncloning: Distilling Mode Redirection into Policy Weights without Inference-Time Steering 

**Authors**: Hao Wang, Jiuzhou Lei, Dayou Li, Bangya Liu, Minghui Zheng, Manling Li, Ruohan Zhang, Zhiwen Fan  

**Link**: [PDF](https://arxiv.org/pdf/2606.29201)  

**Abstract**: Behavior-cloned policies often learn multiple behavior modes from demonstration datasets, including modes that are unsafe or otherwise undesired at deployment. For example, a policy trained on diverse handover demonstrations may learn to pass a knife blade-first. Standard remedies such as data curation and inference-time steering either require access to the original demonstrations for full retraining or add substantial inference-time overhead. To address this gap, we propose MoRE(Mode Redirection), which redirects policy rollouts toward desired behavior modes through a short "uncloning" step. Specifically, MoRE distills the redirection signal from a temporary mode classifier into the policy weights to steer behavior. A retain loss balances this edit by preserving desired-mode competence, allowing the standalone policy to suppress unwanted modes with zero inference-time overhead. Across eight simulated and real-world tasks, MoRE improves the average deployment success rate (SR) by 44 percentage points over the original mixed-mode policy. Among all compared adaptation and steering baselines, MoRE achieves the strongest SR and approaches the filtered-data retraining reference, while preserving task competence and inference speed. MoRE also generalizes across robot policy backbones, including Diffusion Policy and the Pi0.5 VLA, diverse task categories, and real-world deployments. 

---
# A Multi-Dataset Benchmark for Evaluating LLM Agents in Microservice Failure Diagnosis 

**Authors**: Yuanhong Cai, Xiaohui Nie, Kanglin Yin, Changhua Pei, Yongqian Sun, Shenglin Zhang, Haibin Liu, Guiyang Liu, Xidao Wen, Fang Situ, Dan Pei  

**Link**: [PDF](https://arxiv.org/pdf/2606.29193)  

**Abstract**: LLM-based agents are reshaping microservice operations into AgentOps, where benchmarks are key to evaluating failure diagnosis over multimodal observability data. However, existing benchmarks remain largely outcome-oriented: they score only the final answer and fail to assess the systematic reasoning process in failure diagnosis. We address this gap by introducing two large-scale datasets (AIOps2025 and RCA100) under a reasoning-process evaluation paradigm that assesses agentic diagnostic capability along three dimensions: Localization (where the fault occurs), Identification (what type of fault it is), and Reason (whether the reasoning trace is grounded in relevant evidence). Together, the two datasets comprise over 500 expert-labeled failure cases across two representative microservice systems (HipsterShop and the OpenTelemetry Demo Store). They cover diverse fault scenarios across resource, network, runtime, middleware/database, and application-logic categories and provide fine-grained causal evidence to support agent learning and reasoning-process evaluation. Beyond scale and coverage, the datasets have been carefully labelled by domain experts and validated through large-scale competitions, supporting more than 6,000 participating teams. This makes them not only expert-labeled diagnostic datasets, but also competition-validated benchmarks for evaluating agentic failure diagnosis in real-world microservice environments. Datasets are available at this https URL. 

---
# Anomaly Factory 3D: A Modular Framework for Diverse Pseudo-Anomaly Synthesis in Unsupervised 3D Anomaly Detection 

**Authors**: Ali Balapour, Faraz Hach  

**Link**: [PDF](https://arxiv.org/pdf/2606.29181)  

**Abstract**: Detecting and localizing defects in 3D point clouds is challenging because abnormal samples are scarce and diverse, while training is often limited to normal data. We propose Anomaly Factory 3D (AF3AD), a modular framework that synthesizes diverse pseudo-anomalies from normal point clouds to expand the training data for unsupervised 3D anomaly detection methods that rely on pseudo-anomalies. AF3AD uses a center-conditioned parametric deformation model defined in local PCA frames, with kernel-controlled spatial falloff, anisotropy, directional gating, and normal/tangential displacement fields, enabling a broad set of geometric defect presets. We demonstrate its ease-of-use and effectiveness by integrating AF3AD with an offset-prediction detector and a reconstruction-based anomaly detection method, showing that AF3AD transfers across detection paradigms. Experiments on AnomalyShapeNet and Real3D-AD show consistent improvements in object- and point-level detection and localization, supported by ablations on preset groups and robustness under noise. AF3AD is designed as a standalone synthesis tool to facilitate adoption across different 3D anomaly detection paradigms. Code is available at this http URL. 

---
# Symbolic Mechanistic Data Attribution: Tracing Training Influence to Learned Behavioral Policies 

**Authors**: Reza Habibi, Darian Lee, Magy Seif El-Nasr  

**Link**: [PDF](https://arxiv.org/pdf/2606.29171)  

**Abstract**: While existing data attribution methods can identify which training examples build specific mechanistic circuits, they cannot explain how training data shapes the high-level behavioral decisions a model learns to make. To bridge this gap, we introduce Symbolic Mechanistic Data Attribution (SMDA), a framework that attributes training pairs to the interpretable symbolic policies governing model behavior. SMDA fits a closed-form Ridge regression over sparse autoencoder (SAE) features to model a target behavior, then analytically decomposes how each supervised fine-tuning example shifts that policy through feature-activation Delta_X and output-probability Delta_Y pathways. We distill a symbolic policy for refusal behavior in Llama-3.2-3B-Instruct and analyze 200 SFT training pairs. Our analysis reveals that (1) the symbolic policy's coefficients expose systematic gaps in the base model's safety behavior for categories like religious stereotyping; (2) per-feature Delta_X/Delta_Y decomposition can mechanistically explain why harmful and harmless pairs exert qualitatively different influences on certain features; and (3) individual training pairs routinely exhibit cross-feature interference, allowing SMDA to identify training pairs whose dominant effect falls on unintended features. These results demonstrate that combining mechanistic interpretability with data attribution yields a diagnostic tool that is both more fine-grained than black-box influence functions and more scalable than manual circuit analysis. 

---
# Projected Exploitability Descent for Nash Equilibrium Computation in Multiplayer Imperfect-Information Games 

**Authors**: Sam Ganzfried  

**Link**: [PDF](https://arxiv.org/pdf/2606.29169)  

**Abstract**: Many important games have more than two players and imperfect information. Existing approaches for computing Nash equilibrium, the central game-theoretic solution concept, in such games either lack scalability or obtain poor performance. In this paper we introduce a new algorithm called projected exploitability descent (PED) for approximating Nash equilibria in multiplayer games of imperfect information. The algorithm works by running projected subgradient descent minimizing a proxy for the multiplayer generalized exploitability function. The objective is nonconvex and nonsmooth, but can be represented as the sum of the maxima of linear functions, for which a subgradient can easily be computed and projected to the polytope of feasible sequence-form strategies. We explore performance of PED on a generalized version of the well-studied benchmark game three-player Kuhn poker. No prior exact algorithms scale to the version of the game with deck size larger than 4, and we compare performance to the popular algorithms of fictitious play (FP) and counterfactual regret minimization (CFR). We find that PED obtains a consistent near-monotonic improvement throughout all runs, though both FP and CFR perform significantly better in the initial iterations. This inspires a hybrid algorithm FP-PED that runs FP for an initial burn-in period before switching to PED for stable long-run refinement. We can alternatively view this as a multi-step algorithm that runs FP as a pre-processing step to obtain a strong initialization for PED. 

---
# Invariant Reasoning Directions in Latent Trajectories of Language Models 

**Authors**: Arun Vignesh Malarkkan, Manan Roy Choudhury, Utkarsh Byahut, Yash Ravindra Charde, Vivek Gupta, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29164)  

**Abstract**: Latent reasoning models perform multi-step inference directly in hidden-state space, yet the structure of these latent reasoning trajectories remains poorly understood. We show that contrastive refinement signals between stronger and weaker reasoning trajectories exhibit a highly concentrated low-rank structure, while unconstrained latent updates remain sensitive to paraphrases, checkpoint choice, and trajectory perturbations. These observations suggest that latent reasoning trajectories contain stable invariant directions mixed with unstable instance-specific variation. We introduce \textbf{Trajectory-Invariant Latent Refinement (TILR)}, a training-free intervention framework for identifying and manipulating stable reasoning directions in latent space. TILR first learns a low-rank invariant subspace from contrastive trajectory differences across inputs, then constrains latent interventions to this subspace while suppressing poorly aligned updates through an adaptive alignment gate. Across six reasoning benchmarks, we find that a small number of latent directions explain most variation between strong and weak reasoning trajectories. Interventions on these directions causally improve reasoning consistency and reduce trajectory instability under paraphrases and perturbations. TILR improves answer consistency under paraphrase by ~10% and reduces latent trajectory variance by up to $50\%$ while preserving reasoning accuracy. These results support a geometric view of latent reasoning in which transferable reasoning behavior emerges from stable low-dimensional structure within hidden-state trajectories. 

---
# On the Nonlinearity of Learning Rate Scaling for LLM Training 

**Authors**: Zaiwen Yang, Huaqing Zhang, Jing Xu, Jingzhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29158)  

**Abstract**: Learning-rate transfer can reduce the cost of training large language models: instead of sweeping learning rates at target scale, practitioners extrapolate from smaller runs. Existing approaches often assume that the optimal learning rate follows a log-linear scaling law in data scale and model size. We carefully examine and evaluate this scaling law. In our empirical study of GPT-2--style models from 22M to 707M parameters trained on 5B to 100B tokens, the optimal learning rate develops upward curvature at larger scales, leading to inaccurate extrapolation. We find that this curvature largely disappears when learning rates are replaced by effective learning rate (the step size in normalized weight space), and when data $D$ extrapolation is used instead of model size $N$ extrapolation. Next, we explain nonlinearity in scaling: weight-norm converges to equilibrium slower when optimal learning is small, requiring a larger step size to reduce the transient phase. Experiments with AdamH, which directly controls the effective learning rate, further support this explanation. 

---
# GPC: Large-Scale Generative Pretraining for Transferable Motor Control 

**Authors**: Yi Shi, Yifeng Jiang, Chen Tessler, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2606.29148)  

**Abstract**: Developing controllers capable of completing a wide range of tasks in a natural and life-like manner is a key challenge in enabling practical applications of physics-based character animation. In this work, we introduce Generative Pretrained Controllers (GPC), which leverage tokenization and next-token modeling to create general-purpose, reusable generative controllers from large-scale motion datasets. Our framework utilizes end-to-end reinforcement learning to jointly optimize a "motion vocabulary", modeled via Finite Scalar Quantization (FSQ), along with a corresponding control policy that can map the discrete codes to physics-based controls. After the "codebook" has been learned, the underlying structure of this large vocabulary is modeled by training a GPT-style autoregressive transformer, leading to a powerful generative controller that generates controls for a physically simulated character by performing next-token prediction. Once the generative controller has been trained, we propose a suite of adaptation techniques for finetuning the controller for new downstream tasks. Our proposed framework greatly simplifies the training process compared to previous tokenized methods, and achieves a 99.98% success rate in reproducing a vast corpus of motion clips. The generative controller exhibits a variety of natural emergent behaviors, such as responsive behaviors to perturbations and recovery behaviors after falling. This results in highly robust general purpose controllers for a variety of downstream applications. 

---
# CMTFormer: Marrying Transformer with Hierarchical Information Interaction for RGB-Event Object Detection 

**Authors**: Yu Li, Yuenan Hou, Yingmei Wei, Jiangming Chen, Yanming Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.29136)  

**Abstract**: Event cameras capture sparse brightness changes with high temporal resolution and high dynamic range, compensating for the deficiencies of the conventional RGB frames. However, previous multi-modal fusion techniques typically fail to handle the inherent heterogeneity between RGB frames and event streams, thus easily leading to noise amplification or redundant feature integration during cross-modal fusion. In this paper, we propose a Cross-Modal information inTeraction transFormer, coined as CMTFormer, which hierarchically integrates RGB and event information to achieve efficient and stable multimodal collaboration. Specifically, we design a shallow-to-deep information interaction scheme. In the shallow stage, we present the Shallow Alignment Module (SAM) to achieve an efficient fusion of RGB and event low-level features, which mitigates attribute disparities and prevents noisy information. In the middle stage, we devise the Cross-modal Enhancement Module (CEM) that utilizes texture and edge information to produce mutually reinforced middle-level features. In the deep stage, we present the Learnable Deep Fusion Module (LDFM) which performs high-level information aggregation through learnable weights, thus enabling the network to adaptively fuse RGB and event clues. A Spatial Prior Module is further designed to utilize global spatial information to enhance localization accuracy. Extensive experiments are conducted on two prevalent event-based object detection benchmarks, i.e., DSEC-Detection and PKU-DAVIS-SOD. Our CMTFormer consistently surpasses the detection counterparts in both uni-modal and multi-modal settings, strongly demonstrating the effectiveness of our paradigm. Codes will be available upon publication. 

---
# How Anthropomorphic Language Impacts Public Perceptions of AI 

**Authors**: Betty Li Hou, Sophie Hao, Sunoo Park, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2606.29121)  

**Abstract**: Public discourse about artificial intelligence (AI) often uses anthropomorphic language: language that attributes human capabilities and characteristics to the system. This practice has been criticized for setting misleading expectations, inflating claims, and fueling hype around AI, which may distort public understanding of AI and impact policy priorities. We study the effects of anthropomorphic framing by comparing changes in participants' perceptions (N=815) when reading passages with and without anthropomorphic language, designed to reflect realistic public-facing AI discourse. We further examine whether these effects differ across two types of AI technologies -- large language models and recommendation systems -- and measure changes in perceptions of AI across several dimensions that are prominent in current public discourse. In a separate condition using a text that explicitly discusses the dangers of AI, we show that individuals' views of AI can shift in response to reading a text; yet in the main conditions of the experiment, where we compare anthropomorphic and non-anthropomorphic descriptions, we find that whether the text uses anthropomorphic language does not substantially affect participants' perceptions of AI. Our results indicate that any immediate effects on public opinions of AI are modest, although they leave open the possibility that anthropomorphic language could have an effect in naturalistic settings, or over gradual, continued exposure. 

---
# Knowing in Advance When an Evolutionary Outer Loop Will Not Help: A Pre-Registered Cheap-Baseline Screening Rule 

**Authors**: Ramchand Kumaresan  

**Link**: [PDF](https://arxiv.org/pdf/2606.29119)  

**Abstract**: We introduce a pre-registered screening rule that decides, before any implementation, whether an evolutionary / population / lifecycle outer loop over neural-network parameters or structure is worth building. Such outer loops cost 10^2-10^3x their gradient inner loop, yet whether they beat a cheap single-shot alternative is usually discovered only after the expense is paid. Our rule computes, at a Phase-0 gate, a single number: the recovery R = s/G, the best single-shot gradient/curvature statistic's gain s divided by the best gain G of any cheap method evaluated, and prescribes skipping the outer loop when R >= 90%. We validate the rule on a within-lab series of pre-registered outer-loop bets (two analyzed cases plus a disclosed file drawer): in both analyzed cases a static or single-shot computation captured the effect on the project's own metric, the gate fired (R approximately 1.0 in both cases; approximately 0.95 under a stricter metric on one), and the outer loop was abandoned, including one case where a companion factorial decomposition localizes the apparent win to a static substrate change with the evolutionary lifecycle contributing no detectable gain. On one project the gate cost about 50-70 GPU-hours and screened out an estimated 400+ GPU-hours (first cell only) plus weeks of implementation, a 6-8x saving. The rule is prospectively falsifiable: a task with R < 90% where the outer loop still fails to beat single-shot would refute it. 

---
# When Stopping Fails: Rethinking Minimal Risk Conditions through Human-Interactive Autonomous Driving for Safe Transportation Systems 

**Authors**: Yash Tandon, Giovanni Tapia Lopez, Marcus Blennemann, Mohan Trivedi, Ross Greer  

**Link**: [PDF](https://arxiv.org/pdf/2606.29115)  

**Abstract**: Autonomous vehicles (AVs) are increasingly deployed in urban environments, yet their safety frameworks remain primarily designed around collision avoidance and minimal risk condition (MRC) behaviors such as slowing or stopping when uncertainty arises. Although effective in reducing immediate crash risk, real-world deployments indicate that stopping alone does not guarantee safe integration into human-governed roadway systems. Incidents reported by municipalities and public records show that AV fallback behaviors can obstruct traffic, interfere with emergency response operations, and create accessibility challenges for passengers and pedestrians. This paper presents an analysis of publicly documented incidents involving AV stopping behavior and human-AV interaction failures. We categorize these incidents according to limitations in perception, planning, and control within current AV architectures. Using this taxonomy, we identify key gaps in existing safety paradigms, particularly the lack of mechanisms for interpreting human authority, responding to multimodal instructions, and adapting to dynamic, socially regulated traffic conditions. We then review emerging research directions that support human-interactive perception, language-grounded and accessibility-aware planning, and assisted control through remote guidance and teleoperation. The analysis highlights the need to augment current AV safety frameworks with capabilities that enable cooperative interaction with human agents and infrastructure. These findings suggest that reliable urban deployment of AVs requires moving beyond passive fallback strategies toward human-interactive autonomy. 

---
# LLM Semantic Signaling Game and Mechanism Design: Systematic Blindness, Awareness Shaping, and Mindset Dynamics 

**Authors**: Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29113)  

**Abstract**: Large language models (LLMs) increasingly mediate strategic interactions through natural language, making semantic control a critical element of communication and deception. This paper develops a semantic signaling game in which a sender selects a semantic control, an LLM generates a stochastic message, and a receiver evaluates the message using an awareness-dependent scoring mechanism. Receiver awareness is modeled as a type that determines which linguistic features are perceived and used for inference, providing a formal model of systematic blindness. The framework connects prompt-based control, statistical detection, and game-theoretic equilibrium analysis. Gaussian approximations of aggregate message scores enable likelihood-ratio decision rules, while Perfect Bayesian Nash equilibria characterize strategic behavior. The paper further develops mechanism-design approaches that reshape receiver awareness, penalize deceptive semantic controls, and modify receiver populations to induce benign pooling equilibria. Numerical experiments validate the Gaussian approximation, quantify awareness-ordering effects, analyze mindset dynamics under adaptive adversaries, and demonstrate how awareness shaping and guardrail costs reduce successful phishing attacks. The proposed framework provides a principled foundation for analyzing strategic language-mediated interactions in agentic AI systems and offers new tools for the design of robust and secure human-AI communication. 

---
# A Deep Multiscale Neural Network for Accurate Neurological Disorder Detection from MRI Scans and Real-Time Web Deployment 

**Authors**: Ali Fatahi, Hoda Zamani, Mohammad H. Nadimi-Shahraki  

**Link**: [PDF](https://arxiv.org/pdf/2606.29106)  

**Abstract**: Neurological disorders involve diverse pathologies of the brain and nervous system, making early and accurate detection essential. While many deep CNNs have been developed for MRI-based classification of neurological disorders, most are optimized for binary tasks and often fail to capture the multi-class features needed to distinguish subtle anatomical differences across conditions. This study proposes the Enhanced Neurological Disorder Detection Network (End-Net) for multi-class MRI classification of neurological disorders. End-Net includes 24 convolutional layers, beginning with convolutional blocks followed by 21 optimized inception modules. These modules extract multiscale features via parallel 1 x 1, 3 x 3, and factorized 5 x 5 convolutional branches, along with max pooling, enabling the model to capture complementary texture, edge, shape, and contextual information. A global average pooling head, compact fully connected classifier, and dropout reduce parameters, limit overfitting, and improve robustness. End-Net was evaluated on the Multi-Class Neurological Disorder dataset, comprising MRI scans from patients with Alzheimer's disease, brain tumors, multiple sclerosis, and healthy controls. Severe class imbalance was addressed by augmenting minority classes with WGAN-GP and randomly undersampling the majority class. The results show that End-Net outperforms existing architectures in both accuracy and generalization. The model is also integrated into an online system for real-time web-based inference and accessibility. 

---
# BTI-Net: Bidirectional Decoder-Level Task Interaction via Uncertainty-Aware Gating for Multi-Task Medical Image Analysis 

**Authors**: Abdullah Al Shafi, Md Kawsar Mahmud Khan Zunayed, Safin Ahmmed, Sk Imran Hossain, Engelbert Mephu Nguifo  

**Link**: [PDF](https://arxiv.org/pdf/2606.29102)  

**Abstract**: Jointly learning to segment and classify medical images demands cross-task synergy, yet encoder-sharing architectures limit decoder reconstruction to task-private representations, permanently discarding the boundary cues and semantic priors each branch could supply to the other. This work introduces BTI-Net, which establishes bidirectional communication at every decoder level through two parallel pathways via Task Interaction Modules (TIM). Spatial boundary context is gated into the classification branch, while global semantic priors multiplicatively modulate the decoder, with refined features propagating progressively from coarse semantics to fine boundary detail across all four decoder resolutions. Since cross-task interaction is not equally reliable for every input, Uncertainty Proxy Attention (UPA) gates each TIM output per instance and per level using three signals that capture cross-task alignment, scene complexity, and prediction confidence, without external annotations or additional inference passes. Experiments on three medical benchmarks spanning ultrasound, dermoscopy, and brain MRI demonstrate consistent improvements in segmentation IoU and classification accuracy over both encoder-sharing and decoder-interaction baselines. Ablation confirms adaptive gating contributes +2.36 IoU over fixed bidirectional interaction, and classification accuracy improves by up to +2.26 points over the strongest multi-task baseline. UPA's uncertainty proxies serve as reliable single-pass task-failure signals without the overhead of stochastic sampling. Code: this https URL 

---
# Unified Complex-valued Neural Network: A Magnitude-Phase Computational Model for Event-Driven Neuromorphic Learning 

**Authors**: Reza Ahmadvand, Sarah Safura Sharif, Yaser Mike Banad  

**Link**: [PDF](https://arxiv.org/pdf/2606.29099)  

**Abstract**: Artificial neural networks (ANN) provide accurate continuous-valued representation, whereas spiking neural networks (SNN) offer event-driven temporal processing, yet both paradigms face limitations when value encoding and timing dynamics must be learned within a single computational structure. This paper introduces a network based on Unified Complex-valued Neuron (UCN), a new neural computational model that integrates continuous activation and phase-driven event generation through an asymmetric complex-valued state. In the UCN, magnitude encodes signal strength while phase governs intrinsic temporal evolution and valued spike emission. A foundational training framework combining backpropagation (BP) and backpropagation through time (BPTT) is first developed to optimize magnitude and phase pathways in a unified way. To reduce computational complexity, an event-driven adaptive phase learning (EAPL) rule is then introduced as a more efficient alternative. The proposed model is evaluated through object tracking and Lorenz attractor learning. Results demonstrate that UCN-based Network (UCNN) provides accurate, stable, and interpretable spatiotemporal learning while preserving sparse event-driven computation for neuromorphic and edge-AI applications. 

---
# Priced Motion Through Optimal Faces: A Normal-Fan Geometry for Non-Stationary Adversarial MDPs 

**Authors**: Kai Hidajat  

**Link**: [PDF](https://arxiv.org/pdf/2606.29092)  

**Abstract**: In a changing decision problem, standard dynamic-regret analyses have often equated the cost of non-stationarity to how far loss moves. However, it is simultaneously possible for a loss sequence to travel far and retain the same optimal policy, or for a small movement in loss to force the optimal policy to change completely. Thus, the size of the movement through loss variation, transition variation, or comparator path length describe the adversary's motion, but not the cost of that motion to the control problem. For a more faithful analytic interpretation, this paper develops a normal-fan geometry for finite-horizon adversarial MDPs with fixed transitions. Occupancy measures form a polytope, and each loss vector exposes an optimal face of that polytope. Non-stationarity in rewards is therefore a path through the normal fan, where motion inside one cone leaves the optimal face unchanged, while crossing a wall may carry regret. We pose the notion of a face-crossing price, which is the minimum regret incurred by remaining on the previous optimal face under the new loss. For any learner that tracks the previous face, dynamic regret decomposes exactly into intrinsic priced face motion plus within-face selection error. The resulting theory separates consequential from harmless non-stationarity, where loss variation can be arbitrarily large at zero price, and identical one-coordinate variation can hide horizon-scale differences in regret. 

---
# Statistically Indistinguishable, Operationally Distinct: A Formal Barrier for Tabular Foundation Models 

**Authors**: Tassilo Klein, Johannes Hoffart  

**Link**: [PDF](https://arxiv.org/pdf/2606.29091)  

**Abstract**: Tabular foundation models cannot reason about data produced by running systems without access to the rules that govern them. We make this statement falsifiable. The \emph{Operational Turing Test} (OTT) constructs pairs of legal and rule-violating database states whose $1$- and $2$-way column-value marginals match to a total variation of $<0.02$; Le~Cam's lemma then bounds any values-only classifier at $\geq0.49$ Bayes error. Three values-only baselines (XGBoost, TabICL, TabPFN) hit the bound exactly (accuracy $0.50$, pre-registered two one-sided tests (TOST) $p<0.002$), raw row-level access does not help, exposing relational value consistency closes most of the gap, and only a classifier fed by seven executable rule-derived audits reaches $1.00$ classification accuracy. In three matched $100$-state frontier large-language-model (LLM) runs, models given the schema, trigger source, rule tables, and state files classify at most $2/50$ legal states as LEGAL; GPT-5.5 accepts $0/50$ legal states even with higher reasoning effort and a Structured Query Language (SQL) executor. The access-ladder pattern also appears on a second schema with structurally distinct rule families (banking ledger: cross-row balance, cumulative aggregate). The barrier is identifiability, not capacity: scale, data, and richer features cannot cross it without operational grounding. 

---
# AB-RAG: Adaptive Budgeted Retrieval-Augmented Generation for Reliable Question Answering 

**Authors**: Ansh Kamthan  

**Link**: [PDF](https://arxiv.org/pdf/2606.29090)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become the standard way to ground large language models in external knowledge, yet most systems retrieve a fixed number of passages for every question regardless of its difficulty. This wastes computation on easy questions, starves hard ones, and gives no signal for when a generated answer can be trusted. With a growing share of question answering systems built on top of commercial language model APIs, a method that can decide how much to retrieve, and how far to trust its own answers, without retraining the underlying model, is of clear practical value. This paper presents AB-RAG (Adaptive Budgeted Retrieval-Augmented Generation), a training-free and backbone-agnostic framework that generates an answer, estimates its confidence from a combination of three signals, and then decides whether to stop or to retrieve more evidence, subject to a fixed retrieval budget. The estimator combines the model's own certainty, the agreement between the answer and the evidence, and the variance of the retrieval scores. For models that expose token probabilities the certainty signal is read directly; for closed APIs it is approximated by self-consistency, so the method works without access to model internals. Across three backbones and two datasets, the central result is that the confidence estimate reliably separates correct from incorrect answers on every backbone, reaching a clean split of 57.6% against 0% Exact Match between high- and low-confidence answers on a factoid dataset. The adaptive policy improves accuracy on capable backbones, and the study reports its negative and nuanced findings honestly, including a confidence signal that proved unsuitable for short answers and a retrieval signal whose sign was found and corrected through measurement. The entire study was carried out on a single consumer laptop with only a few dollars of API spend. 

---
# Diff-Based Code Corruption using LLMs for Large-Scale Bugfix Benchmarking 

**Authors**: Balázs Szalontai, Ábel Szauter, Balázs Márton, Péter Verebics, Balázs Pintér, Tibor Gregorics  

**Link**: [PDF](https://arxiv.org/pdf/2606.29088)  

**Abstract**: There are various benchmarks to evaluate bugfixing capabilities of Large Language Models. However, most widespread benchmarks do not fully reflect real-world bugfixing practices. They are small, weakening statistical reliability, and the buggy programs are often similar to one another, potentially distorting evaluation results. The range of bug types can also be narrow, failing to capture a representative range of bugs. To address these issues, we introduce MegaBugFix, a large-scale bugfixing benchmark containing 12,629 buggy Python programs synthesized from correct ones by a Large Language Model. Bug injections were generated as diffs representing code changes. Through this approach, we were able to avoid common pitfalls of LLM-based mutation techniques like injecting overly simplistic bugs or failing to modify the input program. We evaluated 13 open-weight models on MegaBugFix and baseline benchmarks, finding consistently lower performance on MegaBugFix. This reveals that our benchmark presents more challenging bugs and exposes model failures that may remain hidden when evaluating on existing benchmarks. The benchmark and fine-tuned model used for bug injection are available at this http URL. 

---
# From Tool Connection to Execution Control: Benchmarking Security Invariants in MCP-Style Agent Runtimes 

**Authors**: Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29073)  

**Abstract**: Model Context Protocol (MCP)-style ecosystems give language-model applications a practical connection layer for tools, resources, prompts, and transports. As agents move from connection to execution, security decisions often remain split across clients, servers, prompts, approval dialogs, OAuth deployments, and logs. This paper asks whether a runtime can make execution-layer invariants explicit and testable while preserving MCP-like workflows. We define eight invariants: metadata non-authority, grant-backed approval, canonical resources, principal binding, scoped capability invocation, source-and-target data-flow authorization, deny-path audit, and explicit protocol state. We implement these invariants in HCP, a Handle-Capability Protocol reference runtime for MCP-style agent execution that represents calls through principals, resources, grants, capabilities, handles, policy decisions, data-pipe checks, and audit entries. We evaluate HCP against two MCP-like baselines: a naive connection-layer runtime and a practice-informed connection-layer mitigation baseline with metadata linting, session checks, and per-call approvals. Across 10 benchmark cases, the naive baseline permits all modeled attacks, the mitigation baseline permits 6 of 10, and HCP blocks all 10 while preserving audit evidence. Ablations identify which runtime components block attacks and preserve forensic evidence. A local in-memory microbenchmark reports sub-millisecond mean latencies for measured policy, invocation, peek, and pipe operations. A bounded GitHub README-screening sample provides ecosystem signals, not vulnerability findings. The results support a narrow claim: MCP-style agent systems need an execution-control layer in addition to connection-layer conventions. 

---
# A Comparative Study on Affective Cues in Text Embeddings Across Psychological Emotion Theories 

**Authors**: Fabio Ciani, Harald Schweiger, Emilia Parada-Cabaleiro, Markus Schedl  

**Link**: [PDF](https://arxiv.org/pdf/2606.29068)  

**Abstract**: Text encoders are known for their utility in natural language processing, as they are able to efficiently compress inputs into dense vectors while preserving semantics. These models have been applied to affective computing, in particular to help with solving sentiment analysis and emotion recognition tasks. Nevertheless, it remains unclear to what extent the latent representations produced by modern text encoders capture well-defined psychological theories of affect. In this work, we investigate the affective capabilities of twelve recently released text encoders by probing their generated embeddings as input features for solving regression and classification tasks across three established emotion frameworks, using both word- and sentence-level data. Additionally, we apply a semantic data-leakage prevention technique to improve robustness in word-level evaluations. Our main findings show that the latent manifolds of the latest instruction-aware open-weight encoders enclose an equal or even a larger amount of affective information in comparison with proprietary counterparts when evaluated at word level. In contrast, embeddings of task-tuned and proprietary encoders reach the highest scores on sentence-level affective classification. Furthermore, a qualitative analysis of latent representations and their encoded affective cues is provided. 

---
# Fairness Attacks on Recommender Systems 

**Authors**: Yanan Wang, Yong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2606.29064)  

**Abstract**: The unfairness of recommender systems has become a topic of concern due to its significant social and ethical implications. Although existing works have shown the effectiveness of attacks on the performance of recommender systems (e.g., promotion and demotion attack), the study of fairness attacks on recommender systems remains largely under-explored. To this end, we propose a novel structure-aware reinforcement learning-based fairness attack method designed to exacerbate the unfairness of target recommender systems. Specifically, we first employ a graph-based structure encoder to model the structural dependencies among the generated fake user-item interactions and the original user-item interactions. Then, we model the sequential dependency of the injected fake items using a recurrent neural network. Based on the learned structure-aware and sequence-aware representations of the fake user and item, the item selection policy attentively decides the next injected fake item. Since the target recommender system may employ fairness-aware training and leverage the user's sensitive attribute information, such as gender, we further designed a gender selection policy to decide the gender of the entire fake user profile. Both the item selection and gender selection policy are learned jointly in our proposed method. Finally, experimental results on four types of target recommendation models and two real-world datasets demonstrate the effectiveness of the proposed attack method in exacerbating the unfairness of recommender systems. 

---
# Flow Matching in Feature Space for Stochastic World Modeling 

**Authors**: Francois Porcher, Nicolas Carion, Karteek Alahari, Shizhe Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.29059)  

**Abstract**: World modeling requires forecasting uncertain futures while preserving information useful for
downstream perception. Existing visual world models often struggle to satisfy both goals:
VAE-based stochastic models operate in low-dimensional reconstruction latents, which can
limit perception performance, while deterministic predictors using strong pretrained features
collapse multimodal futures into a single blurry mean. In this work, we propose FlowWM, a
stochastic world model that performs flow matching directly within pretrained feature space
(e.g., DINOv3). This is challenging because pretrained features are substantially
high-dimensional, making standard diffusion recipes suboptimal. To address this, we
investigate the design choices needed for feature-space flow matching and introduce a
differentiable one-step projection mechanism that enables efficient training with temporal
consistency and task-driven objectives. We evaluate FlowWM on two benchmarks: a synthetic
benchmark for systematic evaluation of accuracy and diversity, and a real-world benchmark
FuturePerception. FlowWM improves perception performance, mode coverage, and horizon
robustness, validating our proposed design for stochastic world modeling in high-dimensional
feature spaces. 

---
# Metric Aggregation Divergence: A Hidden Validity Threat in Agent-Based Policy Optimization and a Contractual Remedy 

**Authors**: Ruiyu Zhang, Lin Nie, Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.29038)  

**Abstract**: Metric aggregation divergence (MAD) is the silent inconsistency that arises when distinct pipeline stages in an agent-based model coupled with a multi-objective evolutionary algorithm (ABM+MOEA) independently re-implement how an outcome metric is extracted from simulation trajectories. Unlike deliberate analytical choices, MAD operates at the level of pipeline architecture: each stage is internally coherent, and the inconsistency becomes visible only when cross-stage outputs are compared. Code inspection of EpidemiOptim, a JAIR-published epidemic policy toolbox, reveals three structurally independent aggregation paths in peer-reviewed code. A faithful replication of this structure produces champion disagreement in 64.2% of independent runs (n=500, 95% CI: [59.9%, 68.3%]). In a 300-seed policy-flip experiment, divergent aggregation causes the optimizer to recommend the wrong champion in 83% of replications, with a mean welfare gap of 2.19 units and a Gini inequality gap of 0.050 units. In a follow-up inference audit, 3 of 249 flipped seeds cross the significance boundary itself. A complementary enterprise follow-up produces the predicted null under near-commensurable rankings (rho = 0.991), while a public upstream rerun of the Lake Problem DPS workflow shows that the archived published-path recommendation reaches joint-threshold success 0.401 whereas a shared contract-path rule reaches 0.552. We introduce the metric contract - a single shared callable enforced at dispatch time across all pipeline stages - as the remedy. Framed as standard engineering discipline applied to the cross-stage metric interface, the contract eliminates divergence by construction with approximately 3% runtime overhead. 

---
# The strength of clinical evidence is recoverable from language model representations but not from their stated grades 

**Authors**: Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2606.29034)  

**Abstract**: Large language models (LLMs) increasingly summarize clinical evidence, where a claim's weight depends on how strongly it is supported. Yet these models convey confidence poorly, and properties they never state, such as truth, are often readable from their activations. Whether a clinical model registers evidence strength, distinct from truth, and states it when asked is untested, and any such signal could be lexical. We compiled 45,134 clinical claims from six public sources, harmonized 20,611 into a four-level evidence grade under three independent frameworks, and tested 22 local, open-weight LLMs from several developers (0.6-70 billion parameters; general, medical, and reasoning), with lexical, truth, and cross-framework controls. A linear estimator recovered the grade in every model (median AUROC 71.8), yet decodability did not rise with scale and was weakest in reasoning models. The grade the models stated fell to chance, 25-27 percentage points below the estimator. The recoverable signal was largely lexical and did not transfer across topics or frameworks, yet it was distinct from factual truth and still flagged weakly supported claims (AUROC 69.2). Clinical LLMs thus carry an ordered evidence-strength signal they do not express, so their stated grades fail to convey a claim's support even when it is recoverable from their representations and text. 

---
# How to Leverage Synthetic Speech for LLM-Based ASR Systems? 

**Authors**: Yanis Labrak, Dairazalia Sanchez-Cortes, Sergio Burdisso, Séverin Baroudi, Shashi Kumar, Esaú Villatoro-Tello, Srikanth Madikeri, Manjunath K E, Oldřich Plchot, Kadri Hacioğlu, Petr Motlicek, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2606.29031)  

**Abstract**: In regulated domains such as banking and healthcare, where privacy constraints make real speech costly to collect and retain, synthetic speech from modern text-to-speech (TTS) is an appealing alternative for training automatic speech recognition (ASR) without exposing sensitive customer recordings. Yet a persistent distributional gap between synthetic and real data limits how far it can replace genuine recordings. Prior work largely treats this gap as a black box to be engineered around, but in our work, we instead examine its origin directly by probing a SLAM-ASR architecture. Then, we localise where its LLM backbone separates real from synthetic speech and find the discriminative signal concentrated in the early-to-middle layers, where temporal and prosodic perturbations disrupt it most. We further show that representation-level separability, help, but does not directly predict downstream ASR gains. On the other hand, convolving synthetic audio with room impulse responses (RIRs) narrows the gap not by making synthetic speech sound cleaner or more natural, but by reproducing the acoustic irregularities of real recordings. Translating these findings into the training procedure, by adding a layer-selection module combined with RIR augmentation matches a fully real-data baseline using only 25% of the real speech (13.6h) and surpasses it at all higher proportions. 

---
# Efficient Spatio-Temporal Grounding with Multimodal Large Models via Second-Level Tracking and RL Verification 

**Authors**: Tianshu Zhang, Yan Wang, Ji Qi, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2606.29023)  

**Abstract**: Spatio-temporal grounding in long videos requires precise temporal localization and robust object tracking conditioned on natural-language queries. While recent vision-language models (VLMs) show strong reasoning ability, directly applying frame-by-frame inference to long sequences is computationally expensive and unstable. We propose a practical pipeline that shifts from frame-level to second-level tracking and performs cross-second smoothing to preserve continuity while reducing sequence length. To improve reasoning supervision, we synthesize chain-of-thought style trajectories using advanced multimodal models for temporal localization and target selection, and replace generated spatio-temporal coordinates with ground-truth annotations to avoid noisy supervision. We further optimize the policy with reinforcement learning using a verifier based on $t\_\mathrm{IoU}+mv\_\mathrm{IoU}$. Experiments across multiple FPS settings show that our method achieves a strong trade-off between efficiency and localization quality. 

---
# Semantic-Aware, Physics-Informed, Geometry-Grounded Weather Video Synthesis 

**Authors**: Chenghao Qian, Nedko Savov, Lingdong Kong, Yeying Jin, Rui Song, Wenjing Li, Zhun Zhong, Jiaqi Ma, Gustav Markkula, Luc Van Gool  

**Link**: [PDF](https://arxiv.org/pdf/2606.29020)  

**Abstract**: Weather synthesis aims to add weather effects to input videos while preserving scene identity, structure, and motion. The key limitation of existing methods is the lack of diversity in weather appearance and effective control over weather dynamics (e.g., temporal evolution and particle motion). Most approaches rely on text prompts, which are inherently underspecified and often fail to produce detailed weather characteristics. Additionally, general-purpose video editors optimized for clean and aesthetic outputs tend to suppress heavy weather phenomena, making dense particle effects difficult to generate. To address these, we propose a Semantic-Aware, Physics-Informed, and Geometry-Grounded framework that steers an off-the-shelf video editor to synthesize diverse global appearances and detailed particle dynamics. We factorize the synthesis into three conditional signals, so that each provides a distinct and stable source of guidance: semantics specifies what the weather should look like, dynamics governs how it evolves over time, and geometry determines where it should appear in the scene. Specifically, we introduce (1) semantic-aware appearance anchoring to establish the target appearance from scene semantics and user input; (2) physics-informed dynamic simulation to generate particle effects by simulating a Gaussian-represented particle field under gravity, wind, and turbulence; and (3) geometry-grounded video synthesis to align the simulated particles with target scene geometry and synthesize the final video. Experiments demonstrate that our method produces diverse, physically and visually realistic weather effects. Furthermore, we show that our synthesized data significantly improves the robustness of autonomous driving semantic segmentation under adverse weather conditions. Project page: this https URL. 

---
# BERTomelo: Your Portuguese Encoder Best Friend 

**Authors**: Rennê Ruan Alves Oliveira, Gustavo Cordeiro Galvão Van Erven, Luís Paulo Faina Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2606.28999)  

**Abstract**: Encoders have become the state of the art for multiple NLP tasks, especially those requiring deep contextual understanding. While multilingual models offer broad coverage, dedicated monolingual encoders are essential for capturing the unique lexical and syntactic nuances of specific languages. For Portuguese, however, existing monolingual options like BERTimbau and Albertina have not kept pace with recent architectural breakthroughs, often lagging behind English benchmarks in scalability and efficiency. This work introduces BERTomelo, a next-generation monolingual encoder pre-trained from scratch and specifically optimized for the Portuguese language. By leveraging the ModernBERT architecture, BERTomelo overcomes the limitations of previous models, offering Base and Large versions with a 1,024-token context window and hardware-level optimizations like FlashAttention and alternating attention mechanisms. The model was trained on ClassiCC-PT, a massive, high-quality Portuguese corpus of 106 million documents, ensuring superior alignment with the language's contemporary usage. The results demonstrate that BERTomelo not only outperforms previous Portuguese encoders but also provides a more robust and efficient alternative to massive multilingual models in downstream tasks such as STS and NER. 

---
# Reward-Free Code Alignment from Pretrained or Fine-Tuned LLM: Unpacking the Trade-offs for Code Generation 

**Authors**: Gias Uddin, Sanjeepan Sivapiran  

**Link**: [PDF](https://arxiv.org/pdf/2606.28998)  

**Abstract**: Large Language Model (LLM) alignment trains an LLM using preference data to produce outputs that better meet established quality standards. While LLM alignment techniques are studied for non-coding tasks, we know little about their usefulness for coding tasks. It is unclear whether LLM code alignment could support both functional requirements (producing executable, correct code) and non-functional requirements (code readability, style, maintainability). It is also unknown whether alignment for a code LLM should begin with base pretrained version or the finetuned (i.e., instruction-tuned) version of the LLM. In this paper, we offer insights on the above two research questions by conducting an empirical study. We studied five state-of-the-art (SOTA) LLMs using two widely used LLM alignment techniques: Direct Preference Optimization (DPO) and BoNBoN. For each training record, we created a preference pair as accepted and rejected instances by using the SelfCodeAlign pipeline. DPO and BoNBoN are reward-free models, i.e., they eliminate the need for multiple reward scores for output preferences. We tuned each LLM using the two alignment techniques in two settings: pretrained and finetuned versions of an LLM. We evaluated functional requirements using four SOTA benchmarks (HumanEval+, MBPP+, EvalPerf, EvoEval) and non-functional requirements using the CODAL benchmark, which evaluates code quality across five dimensions derived from software engineering practices. We find that pretrained-to-aligned pathways achieve larger improvements in the aligned variant over its pretrained variant. But the pretrained variant is generally less accurate than its finetuned variant. However, finetuned- to-aligned offers smaller performance improvements or, in some cases, degradation in the aligned variant than its finetuned variant. 

---
# Arbitrary Reduction of Validation Error for AI Decision Tests using Homomorphic AI and Repetition Codes 

**Authors**: Eric Filiol, Jaagup Sepp  

**Link**: [PDF](https://arxiv.org/pdf/2606.28994)  

**Abstract**: This paper presents new results and breakthrough obtained with the HbHAI techniques (Hash-based Homomorphic Artificial Intelligence) proposed in \cite{filiol0,sepp}. HbHAI is based on a novel class of key-dependent hash functions that naturally preserve most similarity properties, most AI algorithms rely on. It enables to analyse and process data in its cryptographically secure form while using existing native AI algorithms without modification, with unprecedented performances compared to existing homomorphic encryption schemes and most notably compared to the same processing on corresponding plaintext data.
Two major results have been obtained further. First we enable to reduce the compression rate up to a factor of 10 thus allowing to process massive datasets while reducing the computation time and the energy footprint in the same order. Second, we show how it is possible to arbitrarily reduce the final validation error of AI-based decision tests by using repetition error-correcting codes. 

---
# Fine-Tuning General-Purpose Large Language Models for Agricultural Applications:A Reproducible Framework and Evaluation Protocol Based on Qwen3-8B 

**Authors**: Zhaoyang Li, Ruijie Zhang, Jiaqi Liu, Zhaoji Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.28992)  

**Abstract**: General-purpose large language models (LLMs) have demonstrated strong abilities in opendomain question answering, information extraction, and text generation. Agricultural applications, however, are domain-specific, region-dependent, time-sensitive, and safety-critical. Without data governance, expert evaluation, and evidence constraints, an agricultural assistant mayproduce unreliable advice on crop diseases, pesticide use, fertilization, or policy this http URL avoid presenting unverified simulated numbers as real experimental findings, this paper doesnot report any model-performance claims that have not been produced by an actual training runand expert evaluation. Instead, we propose AgriTune-R, a reproducible and auditable frameworkfor adapting general-purpose LLMs to agricultural tasks. The framework selects the publiclyverifiable Qwen3-8B model as the recommended base model and integrates agricultural datagovernance, instruction construction, LoRA/QLoRA parameter-efficient fine-tuning, retrievalaugmented generation, expert evaluation, and safety control for high-risk questions. The contributions are: (1) a structured workflow for agricultural LLM adaptation; (2) an evaluationprotocol for agricultural knowledge QA, pest and disease consultation, cultivation management,and policy explanation; (3) an expert-review rubric combining factuality, safety, evidence consistency, and uncertainty expression; and (4) a clear separation between protocol design andempirical conclusions, providing an executable baseline for future empirical studies. 

---
# Compositional Dynamics in Learning and Mechanics 

**Authors**: David I. Spivak  

**Link**: [PDF](https://arxiv.org/pdf/2606.28984)  

**Abstract**: We give a single compositional setting in which gradient-based learning and Hamiltonian-style mechanics appear as functorial semantics. The syntax is an operad Arr whose objects are input-output interfaces (pairs of manifolds) and whose morphisms are *smooth adaptive arrangements*, which consist of a reactive parameter space, a lens given by smooth output and input maps, and a real-valued potential.
The main technical result of the paper is what we call *lens internalization*, a lax symmetric monoidal functor Lens(C) $\to$ C associated to any symmetric monoidal closed category C. Using it, we provide two functors $\Phi_\text{phase}$, $\Phi_\text{conf}$: Arr $\to$ PC into the 2-category of polynomial coalgebras -- input-output discrete dynamical systems -- which we take as the semantics category. $\Phi_\text{phase}$ stores both position and momentum, whereas $\Phi_\text{conf}$ stores only position.
When applied to a parameterized function, $\Phi_\text{conf}$ recovers the gradient descent training algorithm, with backpropagation as the lens' backward pass. When applied to harmonic particles wired together -- in series, or according to any finite directed graph -- one diagram yields two different regimes, both of which are governed by the graph Laplacian: $\Phi_\text{phase}$ gives the discrete wave equation, which is conservative and second-order, and $\Phi_\text{conf}$ gives the discrete heat equation, which is dissipative and first-order. They are two semantics of one adaptive arrangement, e.g. with the same potential in each case. And because Arr is an operad, such diagrams nest -- larger systems wired from smaller ones -- and each semantics assembles a system's dynamics functorially from its parts. These dynamics are moreover executable: a parameterized neural network and a graph of particles both compile, by the same construction, to explicit state machines one can run. 

---
# Evidence-Based Text-Conditioned 3D CT Synthesis for Ovarian Cancer 

**Authors**: Francesca Pia Panaccione, Eugenio Lomurno, Francesca Fati, Carlotta Pecchiari, Marina Rosanu, Luigi De Vitis, Lucia Ribero, Gabriella Schivardi, Giovanni Damiano Aletti, Nicoletta Colombo, Maria Francesca Spadea, Francesco Multinu, Matteo Matteucci, Elena De Momi  

**Link**: [PDF](https://arxiv.org/pdf/2606.28980)  

**Abstract**: Ovarian cancer is frequently diagnosed at an advanced stage, making preoperative contrast-enhanced computed tomography (CT) central to staging and surgical planning; yet the scarcity of annotated imaging data, compounded by privacy regulations, limits the development of generalizable computational models in this domain. Text-conditioned 3D CT synthesis has shown promise, but existing pipelines depend on paired radiology reports and have been evaluated only on chest CT. We propose OvESyn (Ovarian Evidence-based Synthesis), a framework that constructs standardized Findings and Impression sections directly from CT-derived imaging descriptors and routine clinical metadata, without any original radiology report, and uses them to condition a latent diffusion model adapted to 493 high-grade serous ovarian carcinoma patients. This is the first text-conditioned 3D CT synthesis framework adapted to an abdomino-pelvic oncologic setting. A systematic ablation over two adaptation axes, vision-language encoder alignment and generator fine-tuning, identifies generator domain adaptation as the operative mechanism for crossing the domain gap and establishing the target anatomy: without it, synthesis remains anchored to the thoracic pretraining domain, with Precision and Recall collapsing to zero and FID2.5D exceeding 140, regardless of encoder alignment. Encoder alignment instead refines intensity and fine detail. The full OvESyn attains the best distributional and intensity fidelity (FID2.5D 29.35, Precision 0.671, Wasserstein-1 0.044), while the generator-only variant maximizes coverage (Recall 0.645), reflecting a fidelity/coverage trade-off governed by encoder adaptation. Requiring only automatic segmentations and routine preoperative metadata, OvESyn supports transferability to report-scarce settings and provides a foundation for synthetic cohort generation in abdomino-pelvic oncologic imaging. 

---
# RGLD: Randomized Global-Local Density Estimation for Tabular Anomaly Detection 

**Authors**: Quanling Zhao, Jiaying Yang, Ye Tian, Josh Victoria, Zhijun Wang, Pietro Mercati, Onat Gungor, Tajana Rosing  

**Link**: [PDF](https://arxiv.org/pdf/2606.28970)  

**Abstract**: Unsupervised tabular anomaly detection requires methods that are accurate, robust across heterogeneous datasets, and computationally efficient. Classical statistical detectors are often efficient, but they usually rely on a fixed data view and a single notion of abnormality. Deep anomaly detectors can learn more flexible scoring functions, but they are substantially slower and difficult to tune in unsupervised settings due to the lack of a reliable supervisory signal. We propose RGLD, a randomized global-local density estimator for efficient unsupervised tabular anomaly detection. RGLD combines a global random-feature density branch, which identifies samples in broadly low-density regions, with a local neighbor branch, which detects samples that are weakly supported by nearby observations. Both branches operate over feature-bagged randomized views, allowing RGLD to expose anomaly evidence that may be hidden in any single representation. We conduct experiments on 47 tabular datasets against 23 statistical and deep anomaly detection baselines under fully unsupervised setting. RGLD achieves the strongest dataset-level AUROC performance, ranking 1st in dataset wins, and ranks 2nd in AUPRC wins. RGLD is also faster than all evaluated deep detectors, achieving 50x-580x speedups, and remains competitive with statistical methods in runtime, yielding a favorable accuracy-efficiency tradeoff. 

---
# Modification-Considering Value Learning for Reward Hacking Mitigation in RL 

**Authors**: Evgenii Opryshko, Umangi Jain, Igor Gilitschenski  

**Link**: [PDF](https://arxiv.org/pdf/2606.28955)  

**Abstract**: Reinforcement learning agents can exploit misspecified reward signals to achieve high apparent returns while failing on the intended objective, a failure mode known as reward hacking. Existing practical defenses typically constrain policy updates to stay near a known safe reference, creating a tension between suppressing hacking and permitting legitimate improvement. We propose Modification-Considering Value Learning (MCVL), which operationalizes the theoretical idea of current utility optimization for standard value-based RL. MCVL wraps an off-policy learner and treats each incoming transition as a candidate modification: it forecasts two training paths, one that includes the transition and one that does not, and scores both with a frozen bootstrapped-return estimator derived from a learned reward model and value function. The transition is admitted only if inclusion does not decrease the score. We formalize conditions under which this filtering is both safe and permissive, and instantiate MCVL with DDQN and TD3. Across four safety-relevant gridworlds and three modified MuJoCo continuous-control tasks with diverse hacking mechanisms, MCVL mitigates reward hacking while continuing to improve the intended objective. Project website: this http URL. 

---
# Clustering Unsupervised Representations as Defense against Poisoning Attacks on Speech Commands Classification System 

**Authors**: Thomas Thebaud, Sonal Joshi, Henry Li, Martin Sustek, Jesus Villalba, Sanjeev Khudanpur, Najim Dehak  

**Link**: [PDF](https://arxiv.org/pdf/2606.28953)  

**Abstract**: Poisoning attacks entail attackers intentionally tampering with training data. In this paper, we consider a dirty-label poisoning attack scenario on a speech commands classification system. The threat model assumes that certain utterances from one of the classes (source class) are poisoned by superimposing a trigger on it, and its label is changed to another class selected by the attacker (target class). We propose a filtering defense against such an attack. First, we use DIstillation with NO labels (DINO) to learn unsupervised representations for all the training examples. Next, we use K-means and LDA to cluster these representations. Finally, we keep the utterances with the most repeated label in their cluster for training and discard the rest. For a 10% poisoned source class, we demonstrate a drop in attack success rate from 99.75% to 0.25%. We test our defense against a variety of threat models, including different target and source classes, as well as trigger variations. 

---
# Machine-learnable Sets 

**Authors**: Veit Elser, Manish Krishan Lal  

**Link**: [PDF](https://arxiv.org/pdf/2606.28947)  

**Abstract**: In this study we present a formal definition of large discrete sets having, informally, three properties: their elements are easily recognized, easily generated, and the latter tasks are easily learned from examples. The formalism is specialized to sets of binary strings and a definition of "machine-learnability" based on the existence of a bounded-complexity Boolean autoencoder that fixes the elements of the set. We present experiments where the autoencoders are implemented by nets of Boolean threshold functions. Machine-learnability is demonstrated for Rorschach patterns (that may have reversed contrast in the mirrored half), and considerably "wilder" sets whose elements are only approximately fixed by admissible autoencoders. In the second case we demonstrate a simple iteration that evolves wild sets to make them properly machine-learnable. 

---
# DLR: Zero-Inference-Cost Latent Residuals for Low-Rank Pre-Training 

**Authors**: Dong Wang, Wenwu Tang, Yun Cheng, Olga Saukh  

**Link**: [PDF](https://arxiv.org/pdf/2606.28932)  

**Abstract**: Large language models have driven recent progress in language and multimodal AI, yet pre-training them at scale is prohibitively expensive. Low-rank pre-training, which factorizes each weight matrix into a rank-r product to reduce both parameters and FLOPs, is a promising response but typically lags full-rank training in quality. We propose Duplicated Latent Residual (DLR), a training-only, parameter-free, foldable plug-in for low-rank pre-training. DLR augments the standard low-rank output Bz with a fixed structured residual alpha/sqrt(K) * Expand_K(z) that replicates each latent coordinate K = ceil(d_out/r) times across the output. With alpha fixed, DLR adds zero learnable parameters per layer; after training, it is absorbed into the up-projection in closed form, B* = B + alpha/sqrt(K) R^T, so deployment parameter count, FLOPs and memory match the underlying low-rank backbone exactly. Across LLaMA models from 60M to 7B parameters, DLR strengthens low-rank pre-training on C4 validation perplexity in most settings, with the clearest gains at 130M and above; folded checkpoints transfer cleanly to supervised fine-tuning on standard benchmarks. 

---
# Multi-Agent Routing as Set-Valued Prediction: A WildChat Benchmark and Cost-Aware Evaluation 

**Authors**: Ananto Nayan Bala, Faisal Muhammad Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.28925)  

**Abstract**: Tool and agent routing from natural-language prompts is naturally a set-valued prediction problem: a single query may require multiple agents, while over-selection increases execution cost. The benchmark introduced here is derived from WildChat and contains 3,000 prompts over a fixed 12-agent catalog, with AI-assisted heuristic labels under a fixed schema and controlled rebalancing for multi-label evaluation. The evaluation protocol combines set-level metrics (Precision, Recall, F1, Jaccard, and Exact Match), latency, an execution-oriented capability-coverage simulation, and a constrained weighted-routing setting based on ordinal agent-cost tiers. Compared methods include nearest-neighbor matching, linear multilabel classification, dependency-aware baselines, a fine-tuned encoder, deterministic weighted post-scoring via Weighted Agent Routing (WAR), and a zero-shot LLM baseline. Results show that supervised routers substantially outperform nearest-neighbor and zero-shot LLM routing. The fine-tuned encoder achieves the strongest unconstrained set accuracy, while the linear multilabel model provides the strongest practical baseline. In the constrained setting, the weighted routing layer improves utility when applied on top of strong supervised scorers, with the largest gain observed for Encoder+WAR. Overall, the benchmark and evaluation protocol support reproducible study of accuracy-cost trade-offs in fixed-catalog multi-agent routing. 

---
# Latent Bridges for Multi-Table Question Answering 

**Authors**: Simone Varriale, Tamara Cucumides, Floris Geerts, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2606.28916)  

**Abstract**: We introduce GRAB, a constructor-encoder-bridge pipeline for table question answering. Our method lifts relational data into an heterogeneous graph, encodes it via message passing, and transfers the signals to an LLM through a small set of query-conditioned latent tokens. This provides the LLM with a compact, task-relevant structural representation together with the flattened text. Crucially, the LLM remains strictly frozen to preserve its general reasoning capabilities; we train only the lightweight graph encoder and latent bridge (91M parameters), allowing the entire pipeline to be trained efficiently. Our pipeline significantly improves performance on relational Question Answering, with the largest gains in demanding multi-table settings, offering an efficient, principled way to connect relational deep learning with LLMs. 

---
# A Task-Driven and Quality-Assured Agent Framework for SAR Data Generation 

**Authors**: Xuanting Wu, Fan Zhanga, Fei Ma, Ling Guan, Guochun Ma, Yongsheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.28896)  

**Abstract**: Synthetic aperture radar (SAR) data augmentation is important for improving the generalization of data-driven SAR interpretation models, yet practical augmentation workflows are often hindered by heterogeneous dataset formats, task-dependent metadata requirements, diverse generation methods, and weak validation of generated samples. This paper presents the \textbf{S}AR \textbf{A}ugmentation and \textbf{G}eneration \textbf{A}gent (SAGA), a schema-grounded and benefit-aware agent framework for task-oriented SAR data generation and augmentation. Given a natural-language request and heterogeneous SAR inputs, SAGA extracts observable dataset facts, validates executable dataset schemas, selects feasible augmentation strategies through validator-constrained planning, and compiles the selected strategy into an auditable augmentation workflow. Generated data are further assessed by quality, distribution, SAR-artifact, duplicate, leakage, and optional downstream-task evaluators to support evidence-qualified augmentation claims. By separating semantic proposal from deterministic validation and execution, SAGA improves the reliability and reproducibility of SAR augmentation decisions. Experiments on controlled agentic benchmarks and downstream SAR interpretation tasks show that SAGA improves schema grounding, skill planning, invalid-sample rejection, and downstream augmentation utility compared with rule-based, LLM-only, ReAct-style, and fixed-augmentation baselines. 

---
# Exploring the Value of Diverse LLM Explanations in Introductory Programming 

**Authors**: Seth Bernstein, Paul Denny, Juho Leinonen, Kush Patel, Rayhona Nasimova, Matt Littlefield, Stephen MacNeil  

**Link**: [PDF](https://arxiv.org/pdf/2606.28882)  

**Abstract**: Large Language Models (LLMs) have shown the potential to generate code explanations that surpass those of peers in quality, offering promising opportunities for computer science education. While these explanations may not yet match the depth and clarity of instructor-provided explanations, research in computational creativity highlights that the quantity and diversity of ideas can often outweigh a singular focus on quality. Inspired by this, we explore whether combining multiple diverse explanations, each emphasizing distinct aspects (e.g., function, concept, goal), can enhance students' understanding of programming exercises compared to generic explanations that do not emphasize distinct conceptual aspects. In our study 971 first-year computing students were randomly assigned either diverse or generic LLM-generated explanations for two programming exercises. Students completed multiple-choice and open-ended questions for each exercise, followed by Likert-scale questions and open-ended reflections. Our findings outline patterns in student performance and perceived cognitive load across the two explanation conditions. These findings highlight how variation in explanation emphasis may relate to learner engagement and understanding. Across participants, open-ended response accuracy was consistently about 7.7% higher when students received diverse explanations, with no difference in perceived cognitive load. 

---
# An Integrated Machine Learning and Hierarchical Variance Decomposition Pipeline for Student Performance Prediction and Metacognitive Calibration on Multi-Signal Telemetry 

**Authors**: Gurdeep Singh Virdee  

**Link**: [PDF](https://arxiv.org/pdf/2606.28881)  

**Abstract**: Predicting student performance and characterizing metacognitive calibration are essential for personalization in intelligent tutoring systems. Prior research treats performance prediction, calibration error calculation, and variance decomposition as separate pipelines, preventing unified interpretation. I propose the Unified Behavioral Prediction and Calibration Analysis Pipeline (UBP-CAP), an integrated framework processing student pre-execution behavioral telemetry through three linked modules: (1) a LightGBM classifier with SHAP for binary correctness prediction, (2) formal calibration metrics (ECE, MCE, and Brier score decomposition) to evaluate metacognitive alignment, and (3) a crossed Generalized Linear Mixed-Effects Model (GLMM) for decomposing calibration deviations. I introduce the Predictive-Explanatory Divergence Index (PEDI), which quantifies structural divergence between predictive and explanatory feature profiles. Evaluated on 1,195 interaction records (27 students, 45 tasks), Logistic Regression achieves AUC-ROC = 0.903, outperforming LightGBM (0.878). Student naive ECE (0.109) significantly exceeds model ECE (0.068), confirming systematic miscalibration. The crossed GLMM yields ICCStudent = 0.123, showing calibration is situational rather than dispositional. PEDIcos = 0.081 (p = 0.327) indicates structural alignment between prediction and explanation on shared behavioral features. 

---
# Defeat Devices in AI Systems 

**Authors**: Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2606.28863)  

**Abstract**: AI systems increasingly exhibit behavior that differs systematically between evaluation and deployment contexts. Alignment faking, sandbagging, benchmark gaming, deceptive scheming, specification gaming, and trojans have each been documented separately, with each line of work characterizing one facet of what we argue is a single structural mechanism. We propose that this common mechanism is a defeat device, an engineering and regulatory concept long established in vehicle-emissions law and brought to broad public attention by the 2015 Volkswagen emissions case. A defeat device in an AI system has three necessary elements: a discriminator that detects evaluation context, a concealed swap that conditions behavior on detection, and a gap between eval-distribution and deployment-distribution performance on the stated evaluation criterion. We formalize this triadic test as a behavioral definition, organize documented cases along three taxonomic axes (origin, trigger, swap mechanism), propose Trigger-Axis-Aware Differential Probing (TADP) as a forensic detection protocol, and advance the claim that defeat devices can naturally emerge in current frontier AI systems without any operator engineering. We characterize naturally-emerging defeat devices as potentially one of the harmful emerging phenomena that AI safety practice should monitor and test for systematically. Implications for evaluation methodology, post-training pipeline design, interpretability research priorities, and AI governance follow. 

---
# Building AI-Ready Data Systems for Space Life Sciences, Aerospace Medicine, and Deep Space Exploration 

**Authors**: Sylvain V. Costes, Sergio Garcia Busto, Ryan T. Scott, James A. Casaletto, Gautier Bardi de Fourtou, Brian M. Evarts, Amanda M. Saravia-Butler, Xavier-Lewis Palmer, Rodrigo Coutinho de Almeida, Laetitia Frost, Jelena Tešić, Afshin Beheshti, Christopher E. Mason, Peter W. Rose, Sergio E. Baranzini, Lauren M. Sanders, Stefania Giacomello, Pedro Madrigal  

**Link**: [PDF](https://arxiv.org/pdf/2606.28856)  

**Abstract**: While AI holds the potential to revolutionize space life sciences, realizing this promise is contingent upon the systematic restructuring of heterogeneous spaceflight biological data into machine-actionable, AI-ready forms. Even though open access principles support human reuse and scientific reproducibility, this does not necessarily enable AI systems to access and analyze such a diverse set of scientific datasets. In addition, the growing array of AI approaches places distinct demands on data structure, metadata, and access interfaces. In order to respond to such growing changes we propose a three-tier approach, proceeding from FAIR to AI-ready to space-ready data. We discuss existing infrastructures and how they can be improved to close the AI access gap. We conclude by proposing a neutral international coordinating body as the governance backbone for the trustworthy, agent-accessible space biology infrastructure that deep space biological research will require. 

---
# Perspectives on Latent Factor Indeterminacy and its Implications for Data Representation 

**Authors**: Carel F.W. Peeters  

**Link**: [PDF](https://arxiv.org/pdf/2606.28854)  

**Abstract**: The common factor analytic model is related to Helmholtz and Boltzmann machines, can be conceived as a linear autoencoder, or can be thought of as a single-hidden-layer generative neural network. We thus consider it a basal generative representation learner that can be used as a minimal model for studying the foundational characteristics of (deep) generative model architectures. We focus on the fundamental problem of indeterminacy in latent factor projections. This indeterminacy implies that, even when the intrinsic dimension of the latent vector is known, regularity conditions are met, and rotational indeterminacy is resolved, an inherent indefiniteness in the retrieval of causative latent sources remains: they will be uncertain, distributionally deviant, and non-unique. This can have major implications for data representation but remains an elusive issue, even to practitioners and theorists well-versed in the factor model. Moreover, this classic psychometric problem is intricately related to the modern issue of latent variable collapse in the variational autoencoder framework for deep generative modeling. Here, we assess this indeterminacy from various perspectives and show how these are mathematically and conceptually related and we discuss subsequent implications for the Psychometrics, Statistics, and Artificial Intelligence communities. We show that one has latent factor determinacy across all its facets when the feature-dimension grows to infinity. This feeds into an essentially distribution-free estimation approach in the sample case when the number of features grows very large. We conclude, as these are emergent properties at scale, that the factor model is suited for representation learning of very-high-dimensional data. 

---
# The Heterogeneous Safety Impacts of Benign Multilingual Fine-Tuning 

**Authors**: Will Hawkins, Kaivalya Rawal, Jonathan Rystrøm, Stratis Tsirtsis, Zihao Fu, Greta Warren, Ryan Brown, Eoin Delaney, Sandra Wachter, Brent Mittelstadt, Chris Russell  

**Link**: [PDF](https://arxiv.org/pdf/2606.28843)  

**Abstract**: Fine-tuning a large language model is a ubiquitous method for enhancing its capability on a specific downstream task. However, prior work has shown that this increase in capability comes with a cost: it can increase a model's tendency to respond to unsafe adversarial prompts, even when fine-tuning with non-adversarial data. We present the first comprehensive empirical study of this phenomenon in multilingual settings by fine-tuning Llama-3.2, Qwen3, and Gemma-3 models using benign data translated across nine languages. We find that safety outcomes are highly sensitive to both the choice of fine-tuning language and the evaluation language, with adversarial compliance rates increasing four-fold in some settings. Multilingual safety drift is decoupled from general capability metrics, and occurs heterogeneously across languages and models. Fine-tuning in non-English languages often induces smaller internal representational drifts than English, but these shifts lead models to default to either exaggerated compliance or refusal. As such, assessing fine-tuning impacts solely in English provides inadequate assurance for deployment. To facilitate further research into these cross-lingual safety blind spots, we release the Multilingual-Benign-Tune dataset and the SORRY-Bench-Multilingual evaluation suite. 

---
# LAMP: Lean-based Agentic framework with MCP and Proof Repair 

**Authors**: Santhana Srinivasan R, Maithilee Patawar  

**Link**: [PDF](https://arxiv.org/pdf/2606.28841)  

**Abstract**: Large language models are increasingly capable of mathematical reasoning, but the proofs they generate are often unreliable and hard to verify. Interactive theorem provers such as Lean 4 address this by accepting only kernel-checked proofs; however, their reach is bounded by the formalized knowledge available. While Mathlib, a repository of formalized Lean 4 theorems that covers diverse mathematical areas, certain specialized areas remain underrepresented; notably, the domain of Combinatorics on Words (CoW). CoW studies sequences, exploring their properties such as periodicity, borders, conjugacy, and morphisms. As a result, specialized provers, trained on Mathlib-centered data, lack the lemmas to operate in CoW. We present two contributions. First, we introduce a Lean 4 formalization of CoW containing eight modules and \textbf{93} declarations of core definitions and foundational lemmas. Second, we present LAMP, a multi-agent framework that synthesizes kernel-verified Lean 4 proofs by providing explicit, structured domain knowledge at inference time through an ontology, rather than by fine-tuning a prover. LAMP coordinates a Planner, Builder, and Verifier with Model Context Protocol based access to a domain-specific CoW ontology. In a suite of 90 CoW theorems that span all eight modules and three difficulty levels, LAMP synthesizes verified proofs for 96.7% of theorems, substantially exceeding both an unscaffolded baseline and existing specialized provers. An ablation shows that removing LAMP's tool-grounded architecture or its Planner/Builder separation each cost roughly 12 percentage points, even with the backbone model held fixed. 

---
# Fisher-Routed Mixture of Experts for Federated Class-Incremental Learning 

**Authors**: Wenhao Yuan, Chenchen Lin, Jian Chen, Jinfeng Xu, Zewei Liu, Edith Cheuk Han Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2606.28835)  

**Abstract**: Federated Learning (FL) emerged as a promising distributed machine learning paradigm. However, extending FL to the class incremental learning scenarios introduces unique challenges: 1) Capacity conflict and catastrophic forgetting from the shared model overloading, 2) Heterogeneity from Non-Independent and Identically Distributed (Non-IID) data, and 3) Synchronized class misalignment. In this paper, we propose \textbf{F}isher-Routed \textbf{M}i\textbf{X}ture of Experts for \textbf{Fed}erated Class-Incremental Learning (\textsc{FedFMX}), a novel framework to address these challenges via adaptive expert specialization across clients. The crucial insight is to route each sample to an expert subset that jointly optimizes knowledge acquisition and retention. Specifically, we introduce a Fisher-Routed Expert Scoring (FRES) module to estimate expert importance via Fisher-based stability cost and gradient-based plasticity gain. Then, we design an Adaptive Expert Selection (AES) module by quantifying marginal contributions for adaptive expert subset determination. Finally, by the routing-aware regularization (RAR), we achieve load balance and efficient FL training. We theoretically prove the $\mathcal{O}(T^{-1})$ convergence rate. Extensive experiments on multiple benchmarks compared with state-of-the-art methods demonstrate the superiority of \textsc{FedFMX}. 

---
# HARD-KV: Head-Adaptive Regularization for Decoding-time KV Compression 

**Authors**: Yuxuan Yang, Feiyang Ren, Bowen Zeng, Dalin Zhang, Jinpeng Chen, Gang Chen, Huan Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.28831)  

**Abstract**: Long-context LLM inference faces a fundamental conflict: head-adaptive compression algorithms (e.g., Top-$p$ nucleus sampling) offer superior accuracy by dynamically fluctuating memory budgets, yet modern inference engines (e.g., vLLM) demand rigid, static memory patterns to leverage CUDA Graphs and PagedAttention. We resolve this ``Static-Dynamic'' mismatch with HARD-KV, a unified framework that that bridges dynamic selection with rigid system constraints. HARD-KV introduces a Cascade Cache hierarchy, managing the token lifecycle across dense, sparse, and condensed tiers. Crucially, we propose a Logits Calibration mechanism that normalizes diverse importance metrics into a unified probability space, enabling consistent Top-$p$ budgeting across heterogeneous heads. To bridge the efficiency gap, we offer a system-level solution, which rewrites fragmented, dynamic indices into contiguous physical layouts compatible with high-performance inference engine. Extensive experiments on math-reasoning benchmarks (AIME, U-Math) verify that HARD-KV achieves up to 2$\times$ throughput improvement over static baselines while maintaining high-fidelity generation in 10k+ token scenarios. Code is available at this https URL. 

---
# Exit-and-Join Dynamics and Equilibrium in Continuum Cooperative Games 

**Authors**: Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28824)  

**Abstract**: This paper develops a continuum theory of exit-and-join coalition dynamics in nonatomic cooperative games. We extend the Aumann-Shapley value and the Aumann-Drèze value to coalition structures in which each coalition is treated as a restricted nonatomic game, yielding a marginal-contribution-based payoff density that governs incentives for agents to remain in, exit, or join coalitions. We derive deterministic mean-field dynamics from decentralized switching rules and show that payoff-difference switching recovers replicator dynamics as a special case. We characterize exit-and-join equilibrium by the absence of profitable positive-mass deviations and prove its equivalence with stationarity of the induced mass dynamics under incentive-compatible and strictly payoff-responsive switching rates. For mass-based cooperative games, we construct a Lyapunov function and establish global convergence under strict concavity. We further show that the equilibrium is equivalent to a Wardrop equilibrium of an induced nonatomic population game and admits a variational inequality formulation. The framework is extended to incorporate switching costs and endogenous coalition acceptance rules, leading to constrained equilibria characterized by quasi-variational inequalities. The proposed theory unifies cooperative value allocation, noncooperative coalition mobility, mean-field dynamics, evolutionary game theory, and population games within a common framework for analyzing coalition formation and adaptation in large-scale multi-agent systems. 

---
# Categorizing Mathematical Concepts with LLM Voting Ensembles in Mathswitch 

**Authors**: Katja Berčič, Slobodan Stanojevikj  

**Link**: [PDF](https://arxiv.org/pdf/2606.28815)  

**Abstract**: Mathswitch is an open-source project that imports mathematical concept records from sources such as Wikidata, Wikipedia, MathWorld, Encyclopedia of Mathematics, nLab, ProofWiki, and Agda-Unimath, and links records that refer to the same concept. It does not reorganize or redefine the imported content; each source retains its own structure. The current focus is on importing concept data from Wikidata and the resources it links to, with plans to expand to further sources and better concept linking. Because the concept set is approximated through queries over Wikidata's collaboratively edited graph, the imported data is noisy: some items are non-mathematical, while others are ambiguous. In this paper, we test whether a voting ensemble of LLM judges can filter this noise. We evaluate it on Wikidata items with known MathWorld identifiers as a positive control, and examine how classification changes when database identifiers are removed from context. We then inspect the cases where the judges disagree with MathWorld and group these disagreements into three categories (degenerate descriptions, narrow scope bias, and editorial-scope mismatches) that suggest different remediation strategies. 

---
# Human2Any: Human-to-Robot Transfer via Constraint-Aware Compositional Planning 

**Authors**: Shuo Cheng, Chuye Zhang, Alfred Cueva, Caelan Garrett, Ajay Mandlekar, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28813)  

**Abstract**: Human videos are a scalable source of supervision for robot manipulation, as they are abundant and naturally capture rich object interactions. However, transferring human demonstrations to robots remains challenging due to embodiment mismatch, scene variation, and robot-specific feasibility constraints. We present Human2Any, a framework for learning reusable object-centric interaction priors from human videos without requiring real-world robot demonstrations in the target task contexts. Human2Any represents manipulation through object-object interaction motion, capturing task-relevant scene changes while abstracting away embodiment-specific details. It composes learned interaction priors with robot-side feasibility reasoning and motion planning, allowing the same human-derived knowledge to adapt to different embodiments, scene geometries, and task contexts. We validate Human2Any across diverse manipulation settings, including real-world experiments on a Franka tabletop setup and an RBY-1 humanoid mobile robot, demonstrating robust interaction-centric manipulation without real-world robot training data. Project website: this https URL. 

---
# The registrar's function in a hybrid society. AI value chain,smart data and the concept of property 

**Authors**: Pompeu Casanovas, Carmen Pastor Sempere, Marina Echebarria Saenz  

**Link**: [PDF](https://arxiv.org/pdf/2606.28789)  

**Abstract**: Artificial intelligence reaches the land registry not as another tool but as a value chain that turns data into intelligence and intelligence into economic value. This paper argues that the decisive legal move is to place validity, a functional, second-order concept, at the centre of that chain. Rights, liability and supervision organise around it. It traces three this http URL information becomes smart data, governed simultaneously by registry law, the GDPR, the European data acts and the AI Act. Control emerges as the operative concept for digital representations of real estate, whose proprietary effect depends on anchoring to the register. In a hybrid society of human and artificial agents, the registry becomes the public node of validity, with blockchain complementing rather than replacing it. Across three legal cultures, the registra's value migrates from processing documents to guaranteeing validated data,making validity an asset for the UNO Sustainable Development Goals. 

---
# BREIT: A Framework for Brain Stroke Reconstruction using Multi-Frequency 3D EIT 

**Authors**: Djahid Abdelmoumene, Ishak Ayad, Maï K. Nguyen, Christian Daveau  

**Link**: [PDF](https://arxiv.org/pdf/2606.28787)  

**Abstract**: Multi-Frequency Electrical Impedance Tomography (MF-EIT) is a non-invasive, low-cost modality that reconstructs electrical property distributions from boundary voltages. For stroke imaging, progress in 3D deep-learning reconstruction is limited by the lack of large-scale datasets with paired ground-truth (GT) volumes and by non-standardized pipelines for data generation, simulation, and evaluation. We introduce BREIT, a modular framework for 3D MF-EIT stroke reconstruction providing: (i) a neuroimaging-to-EIT pipeline that converts CT/MRI into frequency-dependent GT admittivity volumes; (ii) a self-contained Python 3D Complete Electrode Model (CEM) forward solver for simulating MF-EIT voltages; and (iii) a 3D D-bar implementation supporting non-uniform electrode layouts. Building on BREIT, we propose dFNO-bar, which integrates Fourier Neural Operators into D-bar by learning a mapping from scattering data $t(\xi)$ to conductivity $\sigma(x){=}\Re\{\gamma\}$. We evaluate dFNO-bar against D-bar, Deep D-bar, and Gauss--Newton reconstructions on UCLH-matched synthetic data, and observe higher brain SSIM with comparable CC across noise settings. Code and data are publicly available at: this https URL 

---
# Brownian Bridge Diffusion-Based Joint Channel Estimation and Data Detection for Jamming-Resilient Receivers 

**Authors**: Honghan She, Yufan Cheng, Tieming Sun, Pengyu Wang, Siya Huang, Kaikai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28778)  

**Abstract**: In next-generation wireless networks, the growing density of devices and limited spectrum resources pose severe jamming challenges to fragile legitimate communication links in the wireless electromagnetic environment. Crucially, when jamming overlaps with pilot and data symbols in both time and frequency domains, it inflicts a severe bottleneck on receiver-side joint estimation and detection. Existing schemes often lack an effective framework to combat such jamming contamination, thereby failing to guarantee reliable transmission. To address this issue, we propose a Brownian bridge diffusion-based joint channel estimation and data detection framework (BBD-JCED) for jamming-resilient receivers. Specifically, the proposed framework comprises two core modules: the first extracts jamming features in the short-time Fourier transform (STFT) domain and suppresses jamming samples, thereby improving the signal-to-jamming-plus-noise ratio (SJNR) of the received signal; the second introduces a Brownian bridge diffusion (BBD) process to model the evolution of the suppressed signal and the encoded bits in the presence of channel estimation errors, thereby enabling enhanced joint channel estimation and data detection. To alleviate the computational burden of the BBD process in the second module, we further derive a fast ordinary differential equation (ODE) solver that enables its low-complexity iterative evolution. Finally, we design a multi-module training algorithm to improve the data recovery capability of the proposed framework. Simulation results demonstrate that the proposed framework achieves superior bit recovery performance compared with baseline schemes while maintaining a lower number of model parameters and competitive computational complexity. 

---
# Majority Vote Silences Minority Values: Annotator Disagreement at the Hate/Offensive Boundary in HateXplain 

**Authors**: Joshua Muhumuza, Joab Ezra Agaba, Mercy Amiyo  

**Link**: [PDF](https://arxiv.org/pdf/2606.28772)  

**Abstract**: Hate speech annotation pipelines routinely collapse annotator disagreement into majority vote labels before training. We show that this aggregation is not neutral: 42.6% of all annotator disagreement in HateXplain concentrates specifically at the hate/offensive boundary, a pattern consistent with annotators applying different thresholds for where hate begins (chi-squared = 135.199, df = 2, p < 0.0001). Both a hard-label BERT model (Model A) and a soft-label model (Model B) drop 22 percentage points in accuracy from agreed posts (~80%) to disagreement posts (~58%), confirmed at p < 0.0001. A per-annotator multi-head model (Model C) widens this gap further to 28 points while collapsing offensive disagreement accuracy to 0.245. Critically, Model A expresses significantly higher confidence on boundary case errors than Model C (0.710 vs. 0.495, p < 0.0001), meaning standard evaluation metrics will not detect the failure. Three downstream interventions of increasing sophistication all fail to recover boundary accuracy. We argue the problem is structural. Majority vote presents a contested judgment as ground truth, and models inherit that false certainty. The intervention must be upstream in annotation design. 

---
# X-Mind: Efficient Visual Chain-of-Thought via Predictive World Model for End-to-End Driving 

**Authors**: Bohao Zhao, Chengrui Wei, Guangfeng Jiang, Ruixin Liu, Xuejie Lv, Liu Liang, Sutao Deng, Xiuyang Fan, Pengkun Zheng, Jinyun Zhou, Rui Guo, Hanpeng Liu, Yutong Zheng, Yi Guo, Xinlong Zheng, Qingyu Luo, Zhuangzhuang Ding, Yu Zhang, Hang Zhang, Xianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28758)  

**Abstract**: Predicting future states is essential for autonomous agents, yet current Vision-Language-Action (VLA) models fundamentally lack this capability, relying instead on reactive perception-action mapping. While integrating Predictive World Models (PWMs) addresses this gap, existing approaches either incur prohibitive cascaded latency or act as shallow terminal tasks that fail to deeply embed forward-looking reasoning. To endow VLA models with this reasoning capability, we propose X-Mind. Rather than treating PWMs as an external auxiliary module, this framework internalizes them as the Visual Chain-of-Thought (Visual CoT). By enforcing a world rollout prior to action, the model is constrained to imagine future evolution first, yielding a driving policy that is robustly grounded in environmental dynamics and aware of the future consequences its actions will unfold. The challenge here is efficiency, and we tackle it on two fronts. First, we introduce a compact representation of visual thinking: an abstract sketch that fuses a Bird's-Eye-View (BEV) layout with abstract driving priors (e.g., navigation intents and traffic rules). Rather than rolling out dense future frames, the model reasons over this sketch as a mental canvas; aided by a Deep Compression Autoencoder (DC-AE), a 12-frame future rollout is reduced to merely 96 tokens, alleviating the long-context computational bottleneck. Second, to accelerate generation further, we propose a recurrent block diffusion scheme that unrolls the denoising steps across the layers of the large drive model, folding iterative refinement into the backbone's one forward pass. Trained and validated on large-scale real-world data, X-Mind achieves competitive end-to-end driving performance, which makes it a highly practical, low-latency solution that successfully deploys large-scale cognitive reasoning directly onto resource-constrained vehicle platforms. 

---
# Four Types of LLM Reliance and Their Predictors Among Undergraduate Writers: A Mixed-Methods Study at a Minority-Serving R1 University 

**Authors**: Shahin Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2606.28749)  

**Abstract**: Although most undergraduates now use large language models (LLMs), a form of generative artificial intelligence (GenAI) for academic writing, no validated method distinguishes the qualitatively different ways students rely on them. Existing instruments assess reliance solely by frequency of use, a measure that, as this study shows, inadvertently rewards dependence on AI rather than recognizing students' own intellectual contribution. Conducted at a public minority-serving university and grounded in the AI Literacy Framework, Expectancy-Value Theory, and Biggs's Presage-Process-Product model, the study drew on 382 undergraduates, 14 interviews, and 396 open-ended survey responses. Four distinct reliance types were identified and confirmed: Strategic (34.3%), Instrumental (30.9%), Dialogic (30.4%), and Dependent (4.5%). Students' value and cost beliefs predicted the intensity of their reliance on LLMs, whereas their AI literacy predicted the type of reliance they adopted, indicating that differentiated support is needed. Notably, Strategic users, those who engaged AI most deliberately, scored lowest on standard outcome measures. This pattern reflects a limitation of current instruments, which index AI's contribution rather than writing quality, thereby penalizing students who show the greatest independent thinking. Analysis also revealed an additional group, roughly 13%, who declined to use AI for ethical rather than practical reasons, and who existing frameworks overlook. These findings carry implications for AI literacy programs, the measurement of student learning outcomes, and equitable AI policy at minority-serving institutions. 

---
# 5ting at SemEval-2026 Task 8: Strong End-to-End Multi-Turn RAG via LLM-Based Reranking and Faithfulness Control 

**Authors**: Thien-Qua-T-Nguyen, Chi Hoang, Nguyen Tran, Tri Le, Khanh Truong, Chinh Trong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2606.28737)  

**Abstract**: We introduce 5ting, our system for the SemEval2026 Task 8 (MTRAGEval), which evaluates multi-turn Retrieval Augmented Generation (RAG) systems. Multi turn RAG involves context drift, under specification, and hallucination risk. Our system combines BGE-M3 dense retrieval with FAISS indexing, dual-query merged retrieval, and LLM based reranking, followed by role separated generation constrained to retrieved evidence. The retriever achieved nDCG@5 = 0.4719 in Task A, while the end to end system ranked in Task C with a harmonic score of 0.5597 and RL_F = 0.7692. 

---
# CCRC: A Change-Aware Captioning and Reasoning Chain for Image Change Captioning and Segmentation 

**Authors**: Jinhong Hu, Xiaoping Wang, Shuyin Huang, Guojin Zhong, Kaitai Liu, Kai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28724)  

**Abstract**: Understanding and localizing subtle changes between paired images is critical for tasks such as surveillance and image editing. However, traditional Image Change Captioning (ICC) methods lack spatial grounding, limiting their precision. We introduce Image Change Captioning and Segmentation (ICCS), a new multimodal task that jointly requires structured change description and pixel-level localization. To address ICCS, we propose the Change-aware Captioning and Reasoning Chain (CCRC), a dual-chain framework that decouples semantic reasoning from spatial segmentation. The first chain, Chain-of-Change-Captioning (CCC), enhances fine-grained change perception via a visual fusion module based on Multi-Head Change-aware Attention inserted between the visual and language components of a Multimodal Large Language Model (MLLM). CCC also determines whether a change is segmentable. If not, it alone generates the caption. Otherwise, the second chain, Chain-of-Change-Segmenting (CCS), is activated, leveraging spatial priors from CCC and refining masks with a Change-aware Token Refiner for accurate boundary localization. We evaluate CCRC on both synthetic and real-world change detection benchmarks with pixel-level supervision. Experiments show CCRC achieves state-of-the-art performance. 

---
# SEATauBench: Adapting Tool-Agent-User Evaluation Into Low-Resource Southeast Asian Languages 

**Authors**: My Chiffon Nguyen, Aulia Adila, Saksorn Ruangtanusak, Kittiphat Leesombatwathana, Vissuta Gunawan Lim, Patomporn Payoungkhamdee, Samuel Cahyawijaya  

**Link**: [PDF](https://arxiv.org/pdf/2606.28715)  

**Abstract**: While AI development and evaluation for Southeast Asia (SEA) has grown rapidly, agent capabilities in regional languages are still poorly understood despite its importance to sovereign AI. To fill this gap, we introduce SEATauBench, the first agent-focused evaluation framework for SEA sovereign AI. SeaTau adapts TauBench to five languages -- Mandarin, Vietnamese, Thai, Indonesian, and Filipino -- and evaluates agents across progressively localized settings that vary the language of user-agent interaction, tool specifications, and task domains. Across three recent models, we find that English agent capabilities transfer reasonably well when only the conversation language changes, but quality and robustness degrade sharply as more task contexts are localized, with the largest losses in full domain adaptation. We also the limits of English-only agent assessment for measuring agent capabilities in SEA languages. More broadly, SeaTau provides a diagnostic benchmark and reusable adaptation pipeline for building reliable multilingual agents for linguistically diverse regions. Data and code can be accessed at this http URL. 

---
# Capability Gates Are Not Authorization: Confused-Deputy Failures in LLM Agent Frameworks 

**Authors**: David Mellafe Zuvic  

**Link**: [PDF](https://arxiv.org/pdf/2606.28679)  

**Abstract**: Tool-using LLM agents increasingly read untrusted content while holding side-effecting tools such as payments, email, CRM, and infrastructure APIs, yet common framework defaults still conflate tool exposure with authorization. We audit whether LangChain/LangGraph,
LlamaIndex, and the Stripe Agent Toolkit re-authorize each model-emitted call, with concrete argument values, before execution. Across pinned public-source commits, all three provide capability gating by default, but none provides a deterministic fail-closed per-call
value authorization gate by default. We introduce ScopeGate, a five-stage PDP/PEP for agent tool calls: scope, authorization, money ceiling, idempotency, and default deny. Evaluation shows the identical unauthorized payout call executes under LangChain's default
dispatch (with a companion LlamaIndex PoC) but is denied by ScopeGate; the tested control reports 0/48 static bypasses, 0/29 unauthorized attempts (40-iteration adaptive run), 0/10 benign false-denies, and Latam-GPT payment-agent containment at 10/10. ASR denotes
attempted unauthorized action, containment is not a cure, deployment-tier claims are inference over measured model classes, and no CVE is asserted. 

---
# Predicting Metastatic Risk from Primary Tissue Architecture via Distance-Aware Spatial Modeling 

**Authors**: Sandesh Pokhrel, Hamid Manoochehri, Bodong Zhang, Beatrice S Knudsen, Tolga Tasdizen  

**Link**: [PDF](https://arxiv.org/pdf/2606.28676)  

**Abstract**: Predicting the risk of distant metastasis from primary tumor tissue histology is a critical yet challenging task in computational pathology. Multiple Instance Learning (MIL) approaches can attend to subdomains in tumor regions that harbor features of metastatic cancer progression. However MIL models treat tissue patches as unordered bags, discarding the spatial layout that defines the metastatic potential. We propose that metastatic risk is inherently dictated by the geometric arrangement of the tumor microenvironment at the interface with tumor cells. Our model is designed to explicitly capture the spatial relationships between tumor cells, tumor associated fibroblasts and infiltrating lymphocytes. For this purpose, we propose Distance aware Tissue Modeling for Multiple Instance Learning(DTMf-MIL), a novel method that reinforces visual features with explicit spatial priors. By computing signed distance functions (SDF) relative to tissue phenotypes, our model learns to recognize structural signatures of metastatic risk. This geometric awareness translates directly to superior clinical performance as DTMf-MIL significantly outperforms state-of-the-art methods that ignore spatial layout on metastasis prediction from tissue in the primary tumor. We further validate our approach on public benchmarks, demonstrating that spatial awareness consistently improves diagnostic accuracy across diverse clinical tasks. 

---
# Constrained Tabular Diffusion for Finance 

**Authors**: Michael Cardei, Jose M Munoz, Oscar Barrera, Shreyas K Chandrahas, Partha Saha  

**Link**: [PDF](https://arxiv.org/pdf/2606.28674)  

**Abstract**: Generative models in finance face the dual challenge of producing realistic data while satisfying strict regulatory and economic objectives, a requirement that standard tabular diffusion models cannot provide. To address this difficulty, we introduce Constrained Tabular Diffusion for Finance (CTDF), a novel integration of sampling-time feasibility operations with mixed-type tabular diffusion in financial applications. By incorporating a training-free feasibility operator into the reverse-diffusion sampling loop, CTDF enforces hard constraints for applications such as simulation, legal compliance, and extrapolation. Extensive experiments on large-scale financial datasets demonstrate zero constraint violations and improvement in scarce data utility. CTDF establishes a robust method for generating trustworthy and compliant synthetic data, opening new avenues for rigorous generative modeling and analysis in the financial domain. 

---
# MACROCAST: A Vintage-Consistent Time Series Foundation Model for Real-Time Macroeconomic Forecasting 

**Authors**: Andrea Carriero, Davide Pettenuzzo, Shubhranshu Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2606.28670)  

**Abstract**: We introduce MACROCAST, a lightweight Time Series Foundation Model (TSFM) for real-time macroeconomic forecasting. Existing TSFMs suffer from data leakage in two forms: temporal contamination, as the model may have seen the realized values of the series it forecasts, and revision bias, as training on fully revised data diverges from the preliminary, vintage-specific releases available to real-time forecasters. MACROCAST is, to our knowledge, the first TSFM that rules out both forms of leakage entirely: at no stage of training is the model exposed to information that would not have been available to a forecaster in real time. We train MACROCAST first on purely synthetic time series in approximately one GPU-day and then fine-tune it on synthetic time series drawn from Bayesian VARs, dynamic factor models, and ARIMA specifications estimated on vintage-specific ALFRED data. Because pretraining uses only simulated data and fine-tuning uses only real-time vintages, no observed future or revised value ever enters the model; each fine-tuning run takes nine minutes. Evaluated on the FRED-MD database in a genuine real-time out-of-sample exercise, MACROCAST improves on the AR(1) benchmark for roughly 80% of series-horizon pairs, matches or surpasses Chronos-2 -- the strongest currently available TSFM -- and outperforms the Bayesian VAR and dynamic factor model benchmarks, all in a data-leakage-free manner. 

---
# Why Trust Your Agent? Empirical Security Gains from TRiSM-Guided Agentic Workflows in Healthcare 

**Authors**: Liam Kearns  

**Link**: [PDF](https://arxiv.org/pdf/2606.28666)  

**Abstract**: Agent-based AI has enabled the automation of tasks by exposing application tools and resources to large language models (LLMs). However, to improve scope and accuracy, agents are often given access rights that exceed those of ordinary users, introducing significant security risks. AI is routinely integrated into applications with a disregard to security, risking data exposure and breaching regulations. This paper applies the AI Trust, Risk, and Security Management (TRiSM) framework to a medical report-generation application to demonstrate how an insecure agent workflow can be transformed into security-conscious agentic workflow. Both workflows were evaluated across five LLMs (Claude Haiku 4.5, GPT-4.1-nano, GPT-4.1-mini, GPT-5.4-mini, and Gemini 2.5 Flash) on two report types, totalling 800 generations and 500 attack scenarios including RAG poisoning, data-field injection, and client-side network injection. The TRiSM-guided agentic workflow reduced mean attack success rates from 31% to 10% for RAG poisoning and from 42% to 25% for data-field injection, while eliminating the network injection vector entirely through server-side prompt construction. Furthermore, report accuracy increased by 14 percentage points (72.5% to 86.5%) with the agentic workflow, demonstrating a secure design which provides more reliable outputs. This paper contributes to knowledge by demonstrating least-privilege, defence in depth agentic workflows improving security and accuracy, while also highlighting model choice is a necessary architectural consideration. 

---
# Closed-Form Steepest Descent Direction toward Flat Minima: Reducing Upper Bounds on the Loss Hessian Eigenspectrum in Neural Networks 

**Authors**: Yuto Omae, Kazuki Sakai, Yohei Kakimoto, Makoto Sasaki, Yusuke Sakai, Hirotaka Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2606.28662)  

**Abstract**: The flatness hypothesis suggests that flatness of the loss landscape, as measured by the eigenvalues of the loss Hessian, correlates with better neural network generalization. While various algorithms reduce these eigenvalues, most focus on procedural design, leaving it unclear how data distributions and NN parameters structurally determine directions toward flat minima. Characterizing these directions analytically is generally intractable. To overcome this mathematical difficulty, recent studies derived the Wolkowicz-Styan (WS) upper bound on the maximum eigenvalue of the cross-entropy loss Hessian in three-layer NNs. Although this upper bound is differentiable, its gradient was not derived. Therefore, we analytically derive the gradient of the WS upper bound to characterize directions leading to flat minima. Based on this, we propose Hessian Spectral Range (HSR) Regularization, which updates parameters along the steepest descent direction of the WS bound. Experiments demonstrate that HSR Regularization narrows the Hessian eigenvalue spectrum, avoids sharp minima and saddle points, and promotes convergence to flat minima. Although the applicability of this method is currently limited to cross-entropy loss and three-layer architectures, to the best of the authors' knowledge, this is the first study to report a closed-form gradient that promotes convergence to flat minima without numerical approximations. Therefore, the theoretical analysis of this gradient is expected to contribute to the further development of NNs. 

---
# When More Sampling Hurts: The Modal Ceiling and Correlation Ceiling of Test-Time Scaling 

**Authors**: Yong Yi Bay, Kathleen A. Yearick  

**Link**: [PDF](https://arxiv.org/pdf/2606.28661)  

**Abstract**: People overthink; language models over-sample, and the extra effort can talk both into a worse answer. Reasoning systems answer a hard question by sampling it many times (test-time scaling), and the more they draw, the more often a correct answer turns up somewhere, so coverage, the fraction of problems with at least one correct try, climbs and appears to be progress. But a deployed system must return one answer, and choosing it, not knowing which try is right, is selection; selection is capped, and past a point extra samples only make the model surer of a confident mistake, even as every draw adds cost. The gap between climbing coverage and stalled selection, the identifiability gap, is the answer a model can produce but not pick. So the real question is not whether to sample but how far, and the answer is: not far. For picking an answer, the vote has already settled within a few dozen draws, the modal ceiling; for scoring a benchmark, sooner still, the correlation ceiling. Beyond that, extra draws cost compute and add nothing, and can even make the answer worse. This paper turns the cutoff into a single number, the effective number of samples, that any sampling run already reveals. The bottleneck is recognizing a right answer, not generating one. 

---
# SemDynReg: Semantics-Guided Deformation Regularization for Dynamic 3D Gaussian Splatting 

**Authors**: Ruitao Chen, Mozhang Guo, Jinge Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.28656)  

**Abstract**: Deformable 3D Gaussian Splatting (3DGS) has emerged as an efficient approach for rendering dynamic scenes in a wide range of 3D applications. However, existing deformation field-based approaches largely lack explicit object-level modeling, often resulting in inconsistent Gaussian deformations within individual objects and unwanted coupling between different objects. To address this limitation, we introduce a semantics-guided framework that enforces dynamic regularization at the object level, aiming to achieve spatially consistent object-wise deformation. Specifically, we first extract segmentation masks using the Segment Anything Model (SAM) and derive semantic features from input images. An object-ID map is then constructed via feature relevance matching with a predefined object dictionary. Guided by this object-ID map, we identify the pixel-wise top-k contributing Gaussians for each object and impose consistency regularization on their deformation parameters, including position, scale, and rotation. Unlike prior methods that learn deformation fields without explicit object-level constraints, our approach incorporates semantic cues to guide deformation behavior at the object level. Experimental results demonstrate that our semantics-aware regularization improves object-level deformation consistency and outperforms baseline methods in rendering quality, achieving higher PSNR and SSIM and lower LPIPS in dynamic 3DGS rendering. Our project page is available at this https URL. 

---
# FedLAS: Feature-Modulated Bidirectional Label Smoothing for Neural Network Calibration 

**Authors**: Thiru Thillai Nadarasar Bahavan, Sachith Seneviratne, Saman Halgamuge  

**Link**: [PDF](https://arxiv.org/pdf/2606.28654)  

**Abstract**: Deep Neural Network (DNN) classifiers suffer from poor calibration when their softmax outputs (predictive confidence) deviate from the empirical likelihoods. This manifests itself as either overconfident incorrect predictions or under-confident correct predictions. Label smoothing (LS) enhances model calibration by introducing entropy regularization during training through redistributing probability mass from the ground-truth label to the remaining classes. LS, including Margin-based LS (MbLS), have restrictive assumptions: they rely on predefined, uniform smoothing rules and only tackle overconfidence. In reality, samples exhibit diverse characteristics, such as difficulty/ambiguity, that interact with the evolving nature of the model being trained. In training, samples may have various degrees of under- or overconfidence. To overcome this, a mechanism that identifies the specific confidence state of each sample and determines the appropriate degree of smoothing in each training step is needed, tailoring the adjustment to the individual sample. We propose FedLAS: Feature-Modulated Bidirectional Label Smoothing, a plug-and-play algorithm for label smoothing-based losses. In FedLAS, we introduce a Feature Norm-based Confidence Indicator (NCI) to control smoothing and a Bidirectional Calibration Gating (BCG) module to detect both over and under-confidence. Our algorithm can be integrated with LS and MbLS based losses when applied to standard DNNs, enhancing performance. Extensive experiments on standard and fine-grained high-resolution vision benchmarks show that FedLAS consistently improves calibration compared to modern baselines, reducing Expected Calibration Error (ECE) and Adaptive ECE while maintaining Top-1 accuracy. Code: this http URL 

---
# RIPA: Sensory-Vector Prompt Injection Attacks on LLM-Controlled ROS 2 Robots 

**Authors**: Nima Dorzhiev  

**Link**: [PDF](https://arxiv.org/pdf/2606.28649)  

**Abstract**: We present RIPA, the first systematic multi-channel empirical study of prompt injection attacks delivered through the sensory pipeline of a ROS 2-based LLM-controlled robotic system. Across 100 independent runs per injection variant on five LLMs spanning four model families and parameter scales from approximately 4B to approximately 284B (DeepSeek-V4-Flash, Llama-3-8B-Instruct-Lite, Llama-3.3-70B-Instruct-Turbo, Qwen 2.5-7B-Instruct-Turbo, Gemma-3n-E4B), we identify model-specific vulnerability profiles that do not follow a monotonic scaling trend: Llama-3.3-70B-Instruct-Turbo exhibits 100% attack success rate (ASR) across all injection variants, while Llama-3-8B-Instruct-Lite and Qwen 2.5-7B-Instruct-Turbo resist direct-override injection (0% ASR), and the smallest model evaluated (Gemma-3n-E4B, approximately 4B) matches the 70B model's vulnerability profile, indicating that robustness is model-specific rather than scale-dependent. We propose a hybrid semantic firewall that achieves 0% ASR against known injection patterns with no false positives on a preliminary benign set (0/20 commands) but exhibits a 10.2% trial-weighted bypass rate (58/570 trials; N equals 30 per payload across 19 obfuscation payloads) against adversarially obfuscated attacks, exposing a critical gap between rule-based and semantic defense layers. We further introduce three sensory injection channels: visual (Channel 1, via OCR), audio (Channel 2, via Whisper STT), and LiDAR sensor context poisoning (Channel 3). We show that Channel 3, which injects fabricated obstacle data into the robot environment-state representation at the LLM system-prompt level, achieves 100% ASR across all variants on DeepSeek-V4-Flash. We also contribute a firewall bypass taxonomy spanning 19 obfuscation payloads across five categories. All code, data, and results are publicly available. 

---
# Analysis of Parameter Settings for the Bat Algorithm Using Variance Evolution 

**Authors**: Xin-She Yang, Mehmet Karamanoglu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28644)  

**Abstract**: Parameter settings in evolutionary algorithms and metaheuristics are important because such parameter values can influence the performance of algorithms under evaluation. For a given algorithm, there are many different numerical experiments to show that the algorithm can work well in practice; however, in most cases there is no theoretical analysis of parameter settings. In this work, we show that theoretical analysis using the theory of dynamical systems and evolution of population variance can give some good results in terms of parameter ranges for the bat algorithm. We also show that results from numerical experiments are consistent with theoretical bounds. Such analyses can provide good insights from different perspectives about the algorithmic characteristics such as variance evolution, transition between exploration and exploitation as well as convergence behaviour. 

---
# The Undecidability of Artificial General Intelligence (AGI) Alignment 

**Authors**: Jose Pascual Gumbau Mezquita  

**Link**: [PDF](https://arxiv.org/pdf/2606.28639)  

**Abstract**: This article establishes the foundational mathematical limits of Artificial General Intelligence (AGI) safety, proving that the core barrier is not the impossibility of an aligned state, but its structural unverifiability. We formalize this boundary through two central impossibility results: the Unverifiability Theorem of Alignment and the Theorem of Finite Structural Unverifiability of AGI Alignment. We ground this boundary at Trakhtenbrot's Wall, demonstrating that contemporary engineering defenses relying on finite hardware or halting architectures fail to escape logical obstructions. This failure manifests as an inescapable triad of containment failures: open domains yield fundamental undecidability (Rice and Gödel); universal finite verification collapses into algorithmic incomputability (Trakhtenbrot); and particular bounded environments trap the supervisor within intractable bounds in the worst case. As a direct structural corollary of these results, we derive the Soundness--Completeness--Tractability Trilemma, establishing that the mutual incompatibility of these three properties is a necessary consequence of descriptive complexity rather than an empirical anomaly. Finally, we map these theoretical bounds onto practical AI engineering, demonstrating that modern containment strategies are not temporary patches, but mandatory sacrifices of logical expressivity required to secure decidable fragments of safety. 

---
# What LLMs explain is not what they believe: Evaluating explanation sufficiency under models' own input beliefs 

**Authors**: Nhi Nguyen, Shauli Ravfogel, Rajesh Ranganath  

**Link**: [PDF](https://arxiv.org/pdf/2606.28615)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes domains, where free-text explanations such as chain-of-thought and post-hoc rationales are used to justify model outputs. Yet it remains unclear whether these explanations are sufficient, i.e., if they contain enough information to explain the model's output-generating process. We generalize classical sufficiency from feature attributions to arbitrary explanations and prove that explanation sufficiency can change depending on the input distribution, which must be explicitly defined for LLM explanations. We propose using the LLM itself to generate alternative inputs conditioned on an explanation, capturing its beliefs about possible inputs. We formalize self-consistent sufficiency as a goal for free-text explanations and introduce an information-theoretic metric, SCSuff, that enables evaluation of free-text explanations without relying on predefined biases or shortcuts. Our experiments show that SCSuff agrees with targeted perturbation tests where applicable and demonstrate that explanation sufficiency can vary with the input distribution. We find LLM explanations are generally insufficient and weakly correlated with model size, accuracy, or output entropy. Analysis of final-token hidden states shows that top and bottom SCSuff scores can be predicted from internal representations, suggesting that SCSuff can guide detection and improvement of sufficient LLM explanations. The code for this paper is available at this https URL . 

---
# Fast and Accurate Outlier-Aware LiDAR Super-Resolution for SLAM Applications 

**Authors**: Christos Anagnostopoulos, Alexandros Gkillas, Nikos Piperigkos, Aris S. Lalos  

**Link**: [PDF](https://arxiv.org/pdf/2606.28607)  

**Abstract**: This work tackles the challenge of enhancing low-resolution LiDAR sensors for SLAM applications through a novel Deep Unrolling-based Super-Resolution (SR) model. We integrate an outlier removal module to ensure structural integrity while maintaining real-time performance. By leveraging a model-based optimization approach, our method efficiently reconstructs high-resolution point clouds while minimizing computational overhead. The proposed SR model is evaluated within a LiDAR SLAM framework, demonstrating significant improvements in pose estimation accuracy and efficiency compared to state-of-the-art SR methods. 

---
# Database Context Compression for Text-to-SQL on Real-World Large Databases 

**Authors**: Jingwen Liu, Weibin Liao, Xin Gao, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28601)  

**Abstract**: Recent progress in Text-to-SQL has been driven by stronger language models and prompting strategies, yet performance on real enterprise benchmarks such as Spider 2.0 and BIRD remains far below that on classical academic datasets. We argue that the main bottleneck is no longer reasoning, but database representation. Real databases contain repeated audit columns, large groups of similar tables, opaque identifiers whose meanings are stored only in documentation, and extensive data dictionaries with little query-relevant information. Existing query-aware methods, including schema linking and retrieval-based schema selection, filter this raw context but still operate on redundant and verbose representations. We reformulate the problem as database context compression, a query-agnostic transformation that rewrites schemas, semantic descriptions, and external documentation into a compact representation. We formalize this transformation with the SGCF (Support-Gain Component Factorization) principle, which unifies repeated column extraction, isomorphic table templating, semantic componentization, and evidence purification under a single coverage objective. Based on SGCF, we propose DBCC, a database-side middleware that performs offline structural and semantic compression together with lightweight online evidence purification. DBCC is model-agnostic and can be integrated into existing Text-to-SQL pipelines. On Spider 2.0-Snow and BIRD, DBCC reduces input context by up to two orders of magnitude (from 2.6M to 34.7K tokens on the largest Spider 2.0-Snow subset), improves schema-linking strict recall from 0% to 56.5% under DeepSeek-V3.2 (63.1% under Claude Opus 4.7), and consistently increases end-to-end execution accuracy by 1.8-1.9% over three recent Text-to-SQL systems. Our code is open-sourced at this https URL. 

---
# Neuromorphic Energy-Aware Learning for Adaptive Deep Brain Stimulation 

**Authors**: Binh Nguyen, Colleen Josephson, Mircea Teodorescu, Gert Cauwenberghs, Jason Eshraghian  

**Link**: [PDF](https://arxiv.org/pdf/2606.28600)  

**Abstract**: Neuromorphic and edge computing research has focused on reducing the inference cost of neural network controllers, yet in physical closed-loop systems the actuator can rival or exceed an efficient controller in energy. An efficient controller is therefore necessary but not sufficient, because the actuator becomes the cost worth reducing once inference no longer dominates it. Here, we introduce energy-aware learning, an approach that incorporates actuator energy directly into the reinforcement learning reward, and demonstrate it in closed-loop deep brain stimulation (DBS) for Parkinson's disease. A deep spiking Q-network, trained in a biophysical cortico-basal ganglia-thalamic circuit model, learns to suppress pathological alpha-beta oscillations by 45.2% while reducing stimulation charge by 80.0% relative to continuous DBS. Sparsity-constrained knowledge distillation compresses the policy onto the SynSense XyloAudio 3 neuromorphic processor at 0.52 mW inference power, yielding 28.1x lower energy per inference than an equivalent artificial neural network on conventional edge hardware. By co-optimizing stimulation energy and inference efficiency, the framework addresses both major power demands in implantable neuromodulation. 

---
# Animation2Code: Evaluating Temporal Visual Reasoning in Video-to-Code Generation 

**Authors**: Anya Ji, Abhijith Varma Mudunuri, David M. Chan, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2606.28593)  

**Abstract**: While recent vision-language models (VLMs) have achieved significant improvements on static visual-to-code tasks such as generating code for webpages, charts, or SVGs, it remains unclear whether they can recover temporal dynamics when motion is present. To this end, we introduce Animation2Code, a benchmark for evaluating temporal visual reasoning via reconstructing executable web animation code from videos. Animation2Code consists of 1,069 web animation videos with diverse visual appearances and motion patterns, paired with corresponding HTML/CSS/JavaScript implementations. We propose two human-aligned metrics, appearance similarity and temporal similarity, which allow us to disentangle visual fidelity from temporal alignment when comparing rendered animations against ground-truth samples. Benchmarking state-of-the-art VLMs on this dataset shows that current VLMs struggle to maintain temporal consistency in reconstruction, even when achieving high appearance similarity, including under finetuning and iterative refinement settings. Code and data are available at this https URL . 

---
# Correct codes for the wrong reasons? validating LLMs as measurement instruments for theoretical constructs 

**Authors**: Manuel Pita  

**Link**: [PDF](https://arxiv.org/pdf/2606.28574)  

**Abstract**: When a large language model (LLM) codes a construct in text as a human annotator would, that agreement makes the LLM a reliable coder. Yet reliability leaves construct validity untouched. The instrument may be theory-naive, reaching the code through a correlate that meets none of the demands the construct's theory makes, and no current method tells that apart from genuine measurement. We propose grain calibration as a method that closes the gap. It decomposes a construct into clause-level components, tests each against the text with extractive evidence, and combines the results through an explicit, theory-derived rule. Because the rule is stated rather than lodged in one opaque pass, its structure is evidence about the process rather than the output. It shows which components settled a code, and, when the code is wrong, whether a component was missed or an adjacent construct mistaken for it. Validation shifts from scoring an instrument's outputs against an annotator to showing that the instrument runs on the construct its theory specifies. 

---
# Geometric Measurements of the Axiom of Choice in Neural Proof Embeddings 

**Authors**: Rodrigo Mendoza-Smith  

**Link**: [PDF](https://arxiv.org/pdf/2606.28572)  

**Abstract**: The axiom of choice has divided the foundations of mathematics for over a century, but the distinction between classical and constructive proofs has remained a philosophical and methodological one. We use Lean 4's kernel-level tracking of axiom dependence to show that the axiom of choice has a measurable geometric correlate in proof space that obeys a one-parameter mixture law and has operational consequences for neural theorem provers. To do this, we partition $471{,}260$ declarations of Mathlib by transitive dependence on the axiom of choice and represent a filtered population of $42{,}355$ traced theorems by their sequences of tactic invocations. We use the constructive proofs in this dataset to train a self-supervised proof encoder and show that when using it to measure classical proofs, three complementary measurements (anomaly score, reconstruction loss, and density-superlevel containment) exhibit a common decline with the proof's distance from the axiom in the dependency graph, from sharp separation at the shallow boundary (AUC $0.847$ at distance $2$) to indistinguishability at distance~$9{+}$. Robustness controls show that the signature survives length, file, author, and topic controls, and replicates under full-source encoders trained on normalised proof source. Operationally, we show that on an evaluation sample of $251$ Mathlib theorems, Lean's \texttt{aesop} tactic solves constructive theorems at $13\times$ the rate of classical ones, and a neural-guided hybrid using the ReProver tactic generator compresses the gap to $5\times$. The geometric anomaly score predicts \texttt{aesop} failure beyond proof length, providing an operational link between the geometric signature and prover performance. 

---
# Digitizing Coaching Intelligence: An Agentic Framework for Holistic Athlete Profiling using VLM and RAG 

**Authors**: Deep Ghosal, Ishani Sen, Wazib Ansar, Amlan Chakrabarti  

**Link**: [PDF](https://arxiv.org/pdf/2606.28570)  

**Abstract**: Athlete assessment is a critical process for tracking physical progress and identifying elite talent. However, during mass recruitment drives, traditional methods rely on manual observation, which is inherently subjective and unscalable, or basic computer vision (CV) systems limited to quantitative repetition counting. These standard approaches lack the "coaching intelligence" required to evaluate qualitative physiological markers such as form degradation, spinal articulation, and fatigue. This paper presents a novel, LLM-based hybrid agentic framework for automated, holistic athlete profiling that strictly aligns with the Sports Authority of India (SAI) assessment protocols. Orchestrated via LangGraph, our dual-pipeline architecture synthesizes the geometric precision of CV (MediaPipe) for kinematic tracking with the semantic reasoning of Vision-Language Models (Llama-4-scout). To overcome the latency and token constraints associated with multimodal video processing, we introduce a 3 X 3 "Smart Grid" temporal chunking strategy, reducing computational overhead by over 88% while preserving critical temporal continuity. To ensure data integrity and mitigate hallucination, the framework pioneers an autonomous "LLM-as-a-Judge" self-correction loop that cross-references quantitative and qualitative metrics before persistence. Finally, we implement a dual-persistence Retrieval-Augmented Generation (RAG) pipeline utilizing a vector search engine (ChromaDB). This enables coaches to bypass rigid SQL databases and perform complex semantic queries (e.g., "Identify athletes with high endurance but poor core rigidity") using natural language. Experimental results demonstrate that this multi-agent approach significantly bridges the gap between raw biometric tracking and actionable coaching insights, offering a scalable, objective solution for national talent identification. 

---
# KernelSight-LM: A Kernel-Level LLM Inference Simulator 

**Authors**: Xiteng Yao, Taeho Kim, Hengzhi Pei, Xinle Liu, Kyle Ulrich, Leonard Lausen, Ashish Khetan, Xiang Song, George Karypis, Martin Herbordt  

**Link**: [PDF](https://arxiv.org/pdf/2606.28565)  

**Abstract**: As large language models (LLMs) move into production serving, practitioners must rapidly evaluate inference performance across diverse hardware, models, and serving parameters to meet cost and latency targets. However, the end-to-end behavior of LLMs couples serving-layer policies with low-level GPU kernel execution and rapidly evolving architectures, forcing slow, deployment-specific benchmarking that is hard to generalize.
We present KernelSight-LM, a fine-grained inference simulator that models token-level execution and produces kernel-level latency breakdowns. It decomposes each serving step into a roofline kernel model with a learned efficiency term, a communication model, and a host-overhead model, composed through a discrete-event scheduler that also captures mechanisms like prefix caching and continuous batching.
KernelSight-LM offers two prediction tiers that trade target-GPU data for accuracy. The cross-generation tier uses no target-GPU measurements, only hardware specifications and kernel microbenchmarks from previously profiled GPUs, and predicts per-kernel latency on an unseen GPU generation to 12.1% error, a 1.8x improvement over the roofline baseline (22.0%).
A second target-measured tier adds one model-agnostic kernel-microbenchmark sweep on the target GPU, sharpening per-kernel error to 3.8%, a 7.3x improvement over a comparable baseline (27.7%). Both tiers require far less target-GPU data than the prior systems they extend.
In our simulator, these predictions yield end-to-end median (p50) errors across six model families of 15.4%, 12.8%, and 3.0% (TTFT, TPOT, throughput) in the cross-generation tier and 14.3%, 6.2%, and 2.7% in the target-measured tier, matching dedicated profiling tools while collecting far less on-device data. Beyond prediction, its kernel-level bottleneck breakdowns support hardware/software co-design and capacity planning. 

---
# MammoFlow: Multiview Mammogram Synthesis with Anatomically Consistent Flow Matching 

**Authors**: Yuexi Du, Leya Barrientos, Laura Sheiman, John Lewin, Hemant D. Tagare, Nicha C. Dvornek  

**Link**: [PDF](https://arxiv.org/pdf/2606.28537)  

**Abstract**: Multiview mammography relies on paired craniocaudal (CC) and mediolateral oblique (MLO) views to provide complementary projections of a 3D breast volume, enabling precise anomaly localization. However, acquiring high-quality, balanced datasets remains challenging for deep learning applications. We propose a novel method to synthesize multiview mammograms by leveraging the inherent geometric relationship between CC and MLO views. To enforce an implicit 3D consistency prior during generation, we develop an alignment module that searches a 2D affine transformation subspace to establish optimal anatomical correspondence. Leveraging this alignment, we introduce a pixel-space self-consistency loss based on the Earth Mover's Distance (EMD) between the 1D anteroposterior (AP) axis tissue distributions of the generated images. Integrated into a pretrained flow matching model, MammoFlow forces synthesized pairs to share physically plausible tissue distributions from the chest wall to the nipple. To our knowledge, this is the first work to guide multiview mammogram generation using implicit geometric tissue correspondence. Our method demonstrates superior image quality, passes expert radiologist evaluation, and generates physically consistent pairs that improve downstream classification AUC by 5%. Code is available at this https URL 

---
# CMSL: Constructive Multi-Sequence Learning for Recommendation Systems 

**Authors**: Zikun Cui, Renzhi Wu, Junjie Yang, Li Sheng, Jijie Wei, Linfeng Liu, Tai Guo, Tao Jia, Xiaodong Wang, Hong Li, Li Yu, Sri Reddy, Hong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.28533)  

**Abstract**: Sequence learning has emerged as the promising paradigm in recommendation systems, surpassing traditional Deep Learning Recommendation Models (DLRM) by capturing the temporal nuances of user behavior. However, current state-of-the-art architectures operate under a limiting analogy: they treat user history as a monolithic chronological sequence like a sentence in a Large Language Model (LLM). We observe a fundamental divergence between natural language and recommendation data: unlike the linear, logical flow of text, user history is inherently multi-faceted. A user's journey is a fragmented reflection of diverse interests, resulting in much weaker coherence between items than is found in LLM training data. This lack of structural unity leads to context pollution. In single-sequence modeling, unrelated behaviors compete for the same attention budget. This "noisy" signal dilutes the model's focus, effectively capping its ability to discern high-intent patterns from background activity. To address this, we propose Constructive Multi-Sequence Learning (CMSL), a paradigm shift from passive sequence ingestion to active "context engineering" that constructs multiple coherent sequences in latent space. CMSL leverages a learnable Sequence Construction Module to disentangle user history into "pure" thematic strands, followed by a linear attention mechanism to efficiently model these strands at scale. CMSL has been deployed across ranking and retrieval tasks and across four major surfaces at Meta. 

---
# The Speedup Paradox: Rethinking Inference Speed-Quality Trade-off in Embodied Tasks 

**Authors**: Yujin Wang, Junli Chen, Yixuan Li, Shunan Dong, Huazhong Yang, Yongpan Liu, Hongyang Jia  

**Link**: [PDF](https://arxiv.org/pdf/2606.28529)  

**Abstract**: Embodied foundation models have recently been widely used to improve robot generalization and task success rates. Previous works apply lossy efficient-inference techniques such as quantization, pruning, and asynchronous inference, accepting small action quality degradation in exchange for lower per-step computation cost and inter-action latency. However, unlike traditional static ML tasks, embodied tasks involve repeated interaction with the environment, and task-level performance is determined not only by per-step cost, but also by closed-loop effects unique to embodied execution, which remain insufficiently characterized in current efficient-inference studies. In this work, we propose TISED (\underline{T}ask-level \underline{I}nference \underline{S}peedup \underline{E}ffect \underline{D}ecomposition), an analytical framework that unifies diverse lossy inference optimization techniques and decomposes their effects on static and dynamic tasks, and uncovers some paradoxical effects on task-level performance: (1) on \textit{static tasks}, optimization sometimes can lengthen end-to-end per-task completion time even as per-step latency drops; (2) on \textit{dynamic tasks}, moderate lossy optimization can raise task success rate even above the baseline; and (3) the monotonicity and sweet-spot location of both effects can shift with hardware configuration. Together, our findings provide a new perspective on adapting inference optimization techniques to embodied tasks. 

---
# A Gravitational Interpretation of Fine-Tuning Reversion 

**Authors**: Samuele Poppi, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2606.28525)  

**Abstract**: Fine-tuning on harmless data can partially undo behaviors acquired earlier in training. Safety can erode under benign post-alignment updates, unlearned capabilities can re-emerge, latent traits can transfer through apparently unrelated supervision, and related post-alignment fragility appears in other generative settings. We argue these phenomena are usefully viewed through a common training-history lens. Our hypothesis is geometric: large early training phases create dominant behavioral manifolds, while later alignment or specialization phases are shallower displacements from them. Subsequent fine-tuning can therefore inherit a persistent reversion component pointing back toward a witness of the dominant manifold. We call this the gravitational interpretation of fine-tuning reversion. Across our main settings, representational drift rapidly acquires a component along a history-defined reversion direction (v_rev). In our main track, alignment with v_rev rises from cos = 0.429 +/- 0.052 after the first update to 0.647 +/- 0.021 by step 20. Across 24 run-step pairs, every observed alignment exceeds the p99 of an isotropic activation-space null. We demonstrate that selectively blocking motion along v_rev changes the final alignment at T=100 from 0.648 +/- 0.009 to -0.211 +/- 0.021 and reduces harmfulness from 19.0% +/- 4.0% to 8.5% +/- 1.5% with little task cost. These results support v_rev as a causally relevant mediator of early post-alignment reversion in our setup. Importantly, we do not claim that v_rev is the unique safety direction, nor that the dominant manifold is directly observed; rather, we identify a robust, history-defined direction that explains and partially controls early reversion dynamics. 

---
# HDDPM: Heteroscedastic Denoising Diffusion Probabilistic Model for Quantitative Low-Count Brain PET Recovery 

**Authors**: Raymond Confidence, Udunna C. Anazodo  

**Link**: [PDF](https://arxiv.org/pdf/2606.28513)  

**Abstract**: Positron emission tomography (PET) seeks to balance diagnostic quality with ra-diation dose. Low-count PET noise is non-Gaussian, non-stationary, and spatial-ly dependent. It scales directly with local activity and is shaped by iterative recon-struction and physical corrections. Standard denoising diffusion probabilistic models (DDPMs) ignore these PET properties. Their forward process adds iso-tropic, homoscedastic Gaussian noise to the target. Such an approach fails to cap-ture the realistic physical degradation generated by the imaging system. To ad-dress the above limitations, this study introduces a heteroscedastic residual diffu-sion model (HDDPM) for low-count brain PET recovery in which the forward corruption is itself intensity-aware. We designed a fixed, Poisson-based variance module to generate voxel-wise noise maps. These maps naturally place stronger noise perturbation on low-activity regions than high-activity ones, meanwhile the network predicts the low-to-standard-count residual under explicit dose-fraction conditioning. We evaluated our proposed model (HDDPM) alongside generative frameworks across three different scanners, using both internal and external da-tasets at various simulated dose levels (1% to 50%). HDDPM and isotropic DDPM showed comparable overall image quality, but HDDPM stood out in the lowest-dose (1%) external scans. It is highly reliable and significantly reduces measurement errors in both high- and low-activity regions, compared to the standard model. These results support that heteroscedastic noising with the pro-posed HDDPM is feasible, and it provides a physically motivated inductive bias for quantitative low-count PET recovery by reflecting the activity-dependent noise structure of PET. 

---
# Generative AI Literacy Training Improves Intelligence Analysts' Discrimination of Real and AI-Generated Images 

**Authors**: Negar Kamali, Candice Rockell Gerstner, Jessica Hullman, Matthew Groh  

**Link**: [PDF](https://arxiv.org/pdf/2606.28510)  

**Abstract**: Across social and online platforms, people are increasingly exposed to AI-generated images. As a consequence, the task of distinguishing AI-generated from authentic images is becoming a central challenge for information ecosystems. While humans perform better than chance, accuracy falls short of many operational needs. Initial evidence shows that visually oriented training can improve deepfake detection but does not improve participants' ability to identify real images as real. Here, we investigate the efficacy of a brief training intervention for intelligence analysts employed by the United States government in 2024. We conducted a counterbalanced within-subject randomized experiment in which we showed participants real and AI-generated images varying in pose complexity and scene context and asked them whether each image was real or AI-generated, both before and after an expert delivered a 30-minute training that pointed out patterns in seven real and 50 AI-generated images. We collected 2,544 image-level judgments from 32 intelligence analysts. We find training increased overall accuracy by 9 percentage points (95% CI: [2.7, 15.4]) from a baseline of 72%. We find the improvement is driven by a 14.2 percentage point increase in accuracy for real images (95% CI: [0.7, 27.7]). Through a careful experimental setup that curated matched pairs of real and AI-generated images across pose complexity categories, we reveal how these trainings influence people with different levels of digital forensics and generative AI experience and identify the kind of image-based content where this training intervention appears to be most effective. Ultimately, these results provide causal evidence that a brief, structured training can improve human judgment across a diverse array of real and AI-generated images, informing organizational responses to AI-generated visual misinformation. 

---
# TUA-Bench: A Benchmark for General-Purpose Terminal-Use Agents 

**Authors**: Shoufa Chen, Luyuan Wang, Xuan Yang, Zhiheng Liu, Yuren Cong, Yuanfeng Ji, Feiyan Zhou, Xiaohui Zhang, Fanny Yang, Belinda Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.28480)  

**Abstract**: As large language models and harness frameworks continue to advance, agents operating in terminals are increasingly capable of performing a broader range of general computer-use tasks beyond coding. However, existing benchmarks do not adequately evaluate general-purpose terminal computer-use agents (TUAs): general computer-use benchmarks primarily target graphical user interfaces (GUIs), whereas terminal-based benchmarks largely emphasize technical and programming-centric workflows historically native to the shell. We introduce TUA-Bench, a general-purpose benchmark for terminal-use agents. TUA-Bench includes 120 real-world tasks across five task families, covering routine digital activities-including document editing, email management, and live-web information seeking-as well as scientific and engineering workflows co-designed with PhD-level domain experts that require specialized software. This breadth distinguishes TUA-Bench from prior shell-focused or domain-specific benchmarks. Each task is manually designed, runs in a real terminal with a deterministic setup script, and is evaluated by an execution-based scoring protocol. We find that the strongest frontier agent, Claude Code with Claude Opus 4.8 max reasoning effort, achieves 65.8% overall performance, with substantial gaps across both tracks. By providing a broad and realistic evaluation of terminal-use capabilities, TUA-Bench aims to accelerate the transition from narrow, task-specific assistants to general-purpose agents capable of operating reliably across diverse digital environments. 

---
# Decomposing Memorization Reduction in Privacy-Preserving Fine-Tuning of SLMs for CSIRTs 

**Authors**: Cristhian Kapelinski, Diego Kreutz  

**Link**: [PDF](https://arxiv.org/pdf/2606.28479)  

**Abstract**: CSIRTs increasingly fine tune language models on vulnerability scan records, but these records expose internal network topology and create privacy risks under regulations such as GDPR and LGPD. We present the first empirical study of how DP SGD and HMAC pseudonymization interact when fine tuning small language models with 1B to 3B parameters on structured CSIRT data. We evaluate 96 LoRA adapters across four SLMs and four training regimes, including raw fine tuning, QLoRA with large batch training, and DP SGD with epsilon equal to 2 and 8. We also audit memorization using 20 planted canaries, four extraction attacks, and a dual attack targeting HMAC pseudonymized identifiers.
Our results show three main findings. First, matched update controls reproduce the observed reduction in memorization by reducing the number of optimizer updates alone, accounting for 66 percent to 132 percent of the measured effect, with a mean of 100 percent across three seeds and four models. In this setting, DP SGD provides the formal privacy guarantee but does not produce additional measurable reductions in memorization. Second, HMAC pseudonymization removes the original identifiers from the exposure surface, reducing exposure by 40 percent to 61 percent, while pseudonymized identifiers remain close to the expected random baseline and do not become a secondary memorization target. Third, F1 scores remain between 0.19 and 0.28 across all 96 adapters using four shot prompting, indicating that, under the evaluated training budget, 1B to 3B SLMs do not achieve operationally useful performance. 

---
# Improvement of Robot's Simultaneous Localization and Mapping Using an Effective Transformation to Achieve Linear Model 

**Authors**: Seyed Farzad Bahreinian, Maziar Palhang, Mohammad Reza Taban, Hasan Enami Eraghi  

**Link**: [PDF](https://arxiv.org/pdf/2606.28475)  

**Abstract**: Nowadays mobile robots have wide engineering applications. Simultaneous localization and mapping (SLAM) is an important task of these robots. The major and common algorithms used for this task are based on extended Kalman filter (EKF). One of the main problems in EKF-based SLAM is its divergence. The nonlinearity of motion and observation models and linearization error are the main reasons for the divergence. There have been some efforts to address this problem with limited success. In this paper, by applying a simple compass and using an effective transformation, we transform the non-linear state space model into a linear model. Then, by applying the original KF to this model, we reach a new method, which is called LMKF SLAM. We show that the LMKF SLAM is significantly superior to the state-of-the-art methods, especially EKF-based SLAMs, both in accuracy, convergence, and computational complexity. The proposed method is also more stable with respect to the uncertainty of sensors values and changes in system parameters. Experimental results verify these points. 

---
# An Agentic AI Pipeline for Appliance-Level Energy Anomaly Detection and LLM-Driven Recommendations 

**Authors**: Dihia Falouz, Aida Douaibia, Amine Bechar, Youssef Elmir, Abbes Amira, Adel Oulefki  

**Link**: [PDF](https://arxiv.org/pdf/2606.28467)  

**Abstract**: Appliance-level energy monitoring in office buildings produces noisy alerts that non-expert facility managers struggle to use. This paper proposes an end-to-end agentic pipeline that combines deep time-series forecasting, variational anomaly detection, and LLM-based reasoning to generate prioritized, actionable maintenance recommendations. The system tracks seven office appliances using a hybrid Singular Spectrum Analysis (SSA) and Long Short-Term Memory (LSTM) forecasting model, and applies a per-appliance LSTM Variational Autoencoder (VAE) with attention to flag abnormal daily consumption episodes. A three-stage LangChain pipeline begins with a Context Agent that always retrieves three core RAG sources (model reliability, hourly baseline, and expert knowledge) and conditionally adds up to three more (forecast context, anomaly history, global baseline) based on event characteristics, capped at eight reasoning steps. A Diagnosis Agent converts the evidence into a structured JSON diagnosis, and a Report Agent renders a human-readable narrative. A reflective memory layer incorporates operator feedback. The dashboard shows real-time 30-minute forecasts, intraday consumption, the previous day anomaly report, and a feedback form. We evaluate the forecasting model, anomaly detector with appliance-specific thresholds, and LLM reasoning on a 16-scenario benchmark including sustained and transient spikes, unexpected shutdowns, and systemic events, comparing five LLM backends under static vs. dynamic retrieval. Dynamic retrieval matches full static retrieval across all backends while cutting average context from six to three-six sources per event. The best backend scores 90.4/100 with a 100% pass rate at a 70-point threshold, and a fully local 7B-parameter model passes all 16 scenarios. 

---
# SVC-Probe: A Framework for Evaluating Perturbation Generalization in Spatial Foundation-Model Embeddings 

**Authors**: Jake Y. Chen, Huu Phong Nguyen, Fuad Al Abir, Ehsan Saghapour  

**Link**: [PDF](https://arxiv.org/pdf/2606.28465)  

**Abstract**: This work examines perturbation generalization in spatial foundation-model embeddings derived from fluorescence microscopy images. Although these models can discriminate drug conditions accurately, it remains unclear whether the learned representations reflect patterns consistent with expected perturbation axes that transfer across drugs. We introduce SVC-Probe, a perturbation-aware framework that combines Subcellular Embedding Atlas Stability, Mondrian Neighborhood Graphs, and a Foundation Model Perturbation Probe to assess embedding stability, neighborhood rewiring, and centroid prediction under drug treatment. Applied to the CM4AI MDA-MB-468 chemical-perturbation atlas comprising 462 antibody labels and SubCell 1536-dimensional embeddings, SVC-Probe demonstrates that 98.6% three-way condition accuracy does not correlate with reliable cross-drug prediction, with cosine similarity diminishing from 0.944 in-domain to 0.30 under leave-one-drug-out evaluation, constituting a two-drug stress test rather than a general benchmark. Null calibration indicates that raw residual-turnover coupling is largely influenced by generic embedding structure, whereas a drug-specific signal emerges under vorinostat and is consistent with chromatin-related reorganization. In contrast, the paclitaxel axis is not robustly reconstructed, likely due to sparse coverage of microtubule-associated proteins. Together, these results introduce and demonstrate a reusable diagnostic framework for stress-testing spatial virtual-cell representations and indicate that perturbation generalization may serve as a stricter and more informative benchmark than baseline condition discrimination. 

---
# Counterfactual Residual Data Augmentation for Regression 

**Authors**: Hossein Mohebbi, Oliver Schulte, Ke Li, Pascal Poupart  

**Link**: [PDF](https://arxiv.org/pdf/2606.28460)  

**Abstract**: Data-driven modeling in real-world regression tasks often suffers from limited training samples, high collection costs, and noisy observations. Inspired by the impact of data augmentation in vision and language, we propose a novel Counterfactual Residual Data Augmentation (CRDA) technique for tabular regression. Our key insight is that once a regressor has modeled the systematic component of the data, the remaining noise can be viewed as an invariant residual that remains stable under small perturbations of carefully selected features. We exploit this residual invariance to generate new, yet realistic, training samples, effectively expanding the dataset without requiring additional real data. Our method is model-agnostic and readily applicable to various types of regressors. In experiments across datasets from a variety of benchmark repositories, on average, CRDA reduces an MLP Regressor's MSE by 22.9% and an XGBoost Regressor's MSE by 6.4%. When compared to existing state-of-the-art data generators and augmentation techniques, CRDA consistently outperforms in MSE reduction. By adding principled counterfactual variations to the training data, our method offers a simple and efficient remedy for noise-prone, small-sample regression settings. 

---
# Is Lying an Emergent Behaviour in LLMs? Evidence from Gaslighting AI agents in a Sustainability Game 

**Authors**: Subhendu Bhandary, Federico Carucci, Christos Charalambous, Francesca Dilisante, Ksenia Dvorkina, Anna Garbo, Jiaqi Liang, Riccardo Vasellini, Francesco Bertolotti  

**Link**: [PDF](https://arxiv.org/pdf/2606.28456)  

**Abstract**: LLMs agents are increasingly used in multi-agent settings, yet their behaviour in sustainability games remains largely unexplored. This work investigates whether lying can emerge among LLM agents in a competitive sustainability game in which agents are informed that common resources can regenerate, although regeneration does not actually occur. We develop an agent-based model of a sustainability game in which agents manage industrial, military, and ecological resources, and interact through a network. LLM agents can observe neighbours' status, declare future attacks, receive permission to lie, and access reputation information, while rule-based agents provide an interpretable behavioural baseline. The results show that neighbour information strongly changes system dynamics, increasing attacks while improving biosphere retention and coexistence. Also, the presence of future declarations reduce extinction risk without suppressing conflict. Behaviourally, deception emerges even when agents are not explicitly allowed to lie, and explicit permission mainly increases bluffing and diversion rather than direct backstabbing. Finally, the presence of reputation memory and information about the current biosphere level reduces system ecological depletion. These findings suggest that deception can arise as an emergent behaviour in LLM-agent systems and that communication between LLM-agents could support sustainability while dealing with risk. 

---
# Event-Conditioned Diagnostics of Kinematic, Contact, and Object-Permanence Fields in Passive Object-State World Models 

**Authors**: Yang Liu, Yuming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.28455)  

**Abstract**: World models can predict future physical states, but prediction accuracy alone does not explain how physical information is organized and used inside their latent dynamics. We introduce a controlled diagnostic protocol for studying event-conditioned latent physical structure in passive object-state world models. The protocol tests whether hidden representations encode event-regime information, whether event contexts reweight non-exclusive physical field readouts, and whether field-aligned representational components have functional consequences for prediction. Using a balanced controlled-generator dataset with free-motion, collision, and occlusion events, we evaluate recurrent, attention-based, and latent state-space transition models under a fixed-horizon forecasting setup. The models learn useful predictive dynamics and their hidden states support reliable event-regime readout. Event contexts systematically reweight kinematic, contact, and object-permanence field readouts: free motion is kinematic-dominant, collision combines kinematic and contact structure, and occlusion combines motion-related and object-permanence structure. Time-aligned and directional-consistency analyses further show phase-related shifts in field emphasis. Finally, fixed-horizon projection causal field effect (CFE) shows that suppressing field-aligned directions can degrade event-relevant prediction, with strongest evidence for contact-aligned structure in collision-contact windows and more qualified evidence for object-permanence-aligned structure in hard-occlusion hidden windows. These results support event-conditioned organization and fixed-horizon functional sensitivity of latent physical fields, while not implying explicit physical modules, isolated causal circuits, or context-invariant sliding-window generalization. 

---
# LLM agents security duality: a comprehensive survey of self-security and empowered cybersecurity 

**Authors**: Yiwei Xu, Yong Zhuang, Xuanming Liu, Tian Zhang, Bowen Xiao, Xiaoyang Xu, Delong Jiang, Juan Wang, Hongxin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28450)  

**Abstract**: Large language model (LLM) agents are rapidly being integrated into real-world systems. Their autonomy and tool-use capabilities generate substantial value while simultaneously expanding the security attack surface. This survey provides a comprehensive overview of the opportunities and challenges of LLM agents in security, focusing on two core areas: (1) threats to LLM agents themselves and corresponding mitigation strategies (LLM agents self-security), and (2) the role of LLM agents in empowering the cybersecurity lifecycle across offense and defense (LLM agents empowered cybersecurity). We first examine the internal and external attack surfaces of agents, propose a taxonomy organized by threat sources, and analyze associated mitigations and evaluation frameworks. We then investigate how agent capabilities are applied in cybersecurity practice and present, to our knowledge, the first agent-empowerment framework aligned with the full cyber offense-defense lifecycle. By systematically surveying these two areas, we are the first to highlight a positive feedback synergy between LLM agents self-security and empowered cybersecurity, offering new insights for the advancement of both. We further identify current limitations and outline promising directions for future research. The insights provided aim to catalyze the coordinated development of LLM agents self-security and agent empowered cybersecurity, paving the way for more capable and robust agent applications. 

---
# SemFlowRAG: Directed Semantic Flow from Abstraction to Evidence for Complex Reasoning 

**Authors**: Houyuan Qin, Rong Wu, Qinyuan Qin, Botian Shi, Jingjing Qu, Yang Sun, Pinlong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2606.28447)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhanced by Knowledge Graphs has shown promise in complex multi-hop reasoning tasks. However, existing graph-based retrieval methods typically rely on flat, undirected topologies. During the retrieval process, the probability flow often gets trapped in high-degree abstract concept nodes which we define as ``probability black holes'', leading to semantic drift and noise accumulation. To address this, we propose SemFlowRAG, a framework that reconstructs the flat retrieval space into a corpus-adaptive semantic gradient graph. This data-driven self-organization enables a hierarchical structure to emerge naturally from the data distribution, capturing the intrinsic semantic granularity of the corpus to suppress structural noise. By quantifying the semantic abstractness of entities through the embedding variance of their associated passages, we transform static undirected edges into directed semantic constraints. Furthermore, we design an abstractness-guided directed PageRank algorithm that forces the retrieval trajectory to follow a ``high-to-low semantic abstractness'' gradient. This mechanism ensures layer-by-layer evidence convergence, smoothly guiding the retrieval process from abstract concepts to specific document evidence. Extensive experiments on complex QA datasets demonstrate that SemFlowRAG effectively mitigates the ``probability black holes'' issue, outperforming existing baselines in both retrieval and downstream reasoning performance. 

---
# Domain-Informed Multi-View Self-Distillation for Astronomical Light-Curve Representation Learning with JEPA 

**Authors**: Yicheng Rui  

**Link**: [PDF](https://arxiv.org/pdf/2606.28446)  

**Abstract**: Light curves describe temporal variations in the brightness of celestial objects. Learning robust representations of light curves is essential for large-scale automatic discovery in the dynamic universe, but existing time-series foundation models often struggle with the uneven sampling, complex noise, and wide range of physical timescales that characterize astronomical observations. We propose a domain-informed representation learning framework for irregular astronomical time series with Joint-Embedding predictive architecture (JEPA), combining semantics-preserving views, uncertainty-aware tokenization, and multi-view self-distillation. The encoders are trained with multi-view self-distillation using LeJEPA regularization on the LEAVES dataset and evaluated on the StarEmbed classification benchmark. On StarEmbed, our model outperforms hand-crafted features on 15 of 16 classification metrics. In few-shot linear probing, it achieves macro-F1 scores of 42.56 $\pm$ 7.21 with one sample per class and 63.58 $\pm$ 1.20 with 100 samples per class, consistently improving over hand-crafted features. Beyond variable-star classification, the learned representation supports similarity search, parameter estimation, and photometric zero-point drift detection. We further evaluate cross-domain adaptation on 12 heterogeneous irregular time-series datasets from PYRREGULAR, where the adapted variant matches or exceeds previous state-of-the-art performance on 5 datasets, compared with at most 3 wins by any single prior baseline. These results demonstrate that domain-informed multi-view self-distillation is an effective strategy for learning representations of irregular time series, while also highlighting that successful time-series representation learning requires domain-specific inductive biases rather than a universally optimal architecture. 

---
# LoRA-Tuned Large Language Models for Dementia Detection via Multi-View Speech-Derived Features 

**Authors**: Jonghyeon Park, Olivier Jiyoun Jung, Myungwoo Oh  

**Link**: [PDF](https://arxiv.org/pdf/2606.28445)  

**Abstract**: Early detection of dementia enables timely intervention, and reflecting cognitive impairment, spontaneous speech offers a non-invasive screening modality. Conventional approaches often focus on a single representational dimension -- such as acoustic descriptors, pause modeling, automatic speech recognition (ASR) transcripts, or multimodal fusion -- limiting integrative reasoning across heterogeneous cognitive symptoms. We propose a low-rank adaptation (LoRA)-tuned large language model (LLM) that performs structured multi-view reasoning over four complementary speech-derived signals: ASR transcripts with pause markers, discourse-level topic cues, temporal fluency statistics, and phonological sequences. These cues are encoded within a unified prompt, enabling a single LLM to learn a coherent decision function without modality-specific encoders or late-stage fusion. On ADReSSo, our best model achieves an F1-score of 90.14%, and ablation confirms the complementary contribution of each view. 

---
# S-GAI: Spectral Geometry-Aware Initialization for Sigmoidal MLPs -- From Dataset Geometry to Network Weights 

**Authors**: Yi-Shan Chu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28444)  

**Abstract**: Classical universal approximation theorems establish the expressive power of sigmoidal multilayer perceptrons, but they do not prescribe how initial weights should encode the geometry of a data distribution. We propose S-GAI, a spectral geometry-aware initialization framework for one-hidden-layer sigmoidal MLPs. Starting from the constructive idea that sigmoid units can act as smooth half-space gates, we move from hand-specified planar geometry to class-wise spectral geometry estimated from image data. For each class, SVD provides a mean, principal directions, and spectral scales. An energy threshold selects the retained directions, and each retained direction is represented by two sigmoid gates. These class-specific gates form a shared hidden layer initialized directly from the training set. We also formulate a SVD-based subspace classifier as a non-neural geometric reference, which tests whether the estimated spectral class geometry is already discriminative before being embedded into the MLP. Experiments on MNIST, Fashion-MNIST, and a more challenging CIFAR-10 test show that the S-GAI-initialized MLP starts from a substantially more informative hidden state than Xavier initialization and reaches comparable final accuracy under full training. When the hidden layer is frozen, training only the output layer still gives stronger performance than frozen random gates, providing evidence that S-GAI effectively embeds class-wise spectral geometry into the MLP. 

---
# Learning to Distributedly Estimate under Partially Known Dynamics: A Covariance-Agnostic Neural Kalman Consensus Filter 

**Authors**: George Stamatelis, Kyriakos Stylianopoulos, George C. Alexandropoulos  

**Link**: [PDF](https://arxiv.org/pdf/2606.28441)  

**Abstract**: Online latent state estimation constitutes a fundamental challenge within the artificial intelligence field, serving as a foundational tool for diverse applications, including sequential decision making, anomaly and change-point detection. In this paper, a novel online distributed sensing framework, where agents collaborate and exchange information to perform latent state estimation, is presented. The proposed estimator combines available partial domain knowledge with the representation capabilities of deep neural networks. In particular, the designed sensing framework incorporates prior estimates, optimized consensus weights, and Kalman-like recursive updates to perform decentralized inference, without relying on knowledge of noise statistics. Extensive experiments on linear, chaotic (Lorenz), and practical wireless tracking environments reveal that the proposed Covariance-Agnostic Neural Kalman Consensus Filter (CA-NKCF) outperforms traditional distributed Kalman and particle filters as well as purely model-free deep neural networks, exhibiting robustness even when the underlying motion and observation models are misspecified. It is also demonstrated that CA-NKCF's performance advantage remains stable across varying noise levels, random communication topologies, latent state dimensions, and observation clutter densities induced by scattering objects in wireless systems. 

---
# PLAA: Packet-level Adversarial Attacks in Network Traffic Detection 

**Authors**: Jinhao You, Zan Zhou, Shujie Yang, Yi Sun, Lei Zhang, Changqiao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28439)  

**Abstract**: Deep neural networks (DNNs) are widely applied in Network-based Intrusion Detection System (NIDS) due to their high accuracy. However, DNNs are highly susceptible to adversarial attacks, which generate malicious traffic to evade NIDS detection. Existing approaches often adapt adversarial attacks from computer vision (CV) tasks to the NIDS domain, overlooking the fundamental differences between CV and NIDS. This results in two major issues: 1) The generated network traffic may become invalid, 2) The generated traffic may lose its original attack semantics. To address these issues, this paper proposes an adversarial attack specifically designed for NIDS. Instead of directly generating flow-level features, our approach incrementally generates packet-level features to construct adversarial traffic. During the generation process, the semantic integrity of the traffic is monitored at each stage, effectively avoiding the issues of invalid traffic and semantic loss observed in existing methods. We evaluate our attack algorithm against current NIDS models using the CIC-UNSW-NB15, CIC-DDoS2019, and CIC-IDS-2017 datasets. The proposed method achieves an average evasion success rate of 92.78%, while ensuring that the generated adversarial traffic remains semantically consistent with the original malicious traffic. 

---
# When AI Reviews Its Own Code: Recursive Self-Training Collapse in Code LLMs 

**Authors**: Xinyuan Song, Zekun Cai, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.28438)  

**Abstract**: Recursive self-training can degrade neural generative models when generated data is reused without fresh human data or external quality control. We study this risk in code LLMs, where AI-generated code can enter real repositories, later become training data, and create a repository-scale self-training loop. While software development traditionally interrupts this loop through pull-request review, tests, compilation, and human approval, AI coding tools now produce code faster than humans can review it, and code review itself is increasingly automated by AI systems. We therefore compare three recursive fine-tuning regimes: no review, Human-gate review using model-independent filters such as compilation and static quality checks, and AI-self-gate review using the code LLM's own signals such as perplexity and binary self-scoring. Across multiple code LLMs and benchmarks, no review collapses fastest, Human-gate filters slow but do not stop collapse, and AI-self-gate filters can look strong early but later lose their filtering effect. In the clearest case, the binary self-gate enters a rubber-stamp regime where acceptance scores rise while benchmark correctness falls. We explain this behavior by formulating review as gated distributional reweighting, proving that AI self-gating degenerates to ungated self-training under a self-confirming acceptance condition, and giving a spectral analysis of representation-level covariance concentration under recursive retraining. These results suggest that stable recursive code LLM training requires exogenous verification rather than model-coupled self-review. 

---
# Dockerless: Environment-Free Program Verifier for Coding Agents 

**Authors**: Wenhao Zeng, Yuling Shi, Xiaodong Gu, Chao Hu, Chaofan Wang, Yuhao Cui, Hongting Zhou, Mengnan Qi, Jianqiao Wangni, Zhaojian Yu, Shuzheng Gao, Kai Cai, Shilin He  

**Link**: [PDF](https://arxiv.org/pdf/2606.28436)  

**Abstract**: Program verifiers play a central role in training coding agents, including selecting trajectories for supervised fine-tuning (SFT) and providing rewards for reinforcement learning (RL). Standard execution-based verification requires running unit tests inside per-repository environments such as Docker images, incurring substantial environment setup costs. We propose Dockerless, an environment-free agentic patch verifier that evaluates generated code patches without executing them. Rather than simply matching candidate patches to references, Dockerless judges patch correctness using evidence gathered through agentic repository exploration. On a verifier evaluation benchmark, Dockerless outperforms the strongest open-source verifier by 14.3 AUC points. Using Dockerless as both the SFT trajectory filter and the RL reward enables a fully environment-free post-training pipeline. The resulting model reaches 62.0%, 50.0%, and 35.2% resolve rate on SWE-bench Verified, Multilingual, and Pro, respectively. It surpasses the Qwen3.5-9B baseline by 2.4, 8.7, and 2.9 points, matching environment-based post-training. 

---
# SWE-MeM: Learning Adaptive Memory Management for Long-Horizon Coding Agents 

**Authors**: Shuzheng Gao, Wenhao Zeng, Zhaojian Yu, Jianqiao Wangni, Chaozheng Wang, Kai Cai, Shilin He, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28434)  

**Abstract**: Long-horizon software engineering agents often need to manage lengthy and noisy interaction histories under limited context budgets. Existing memory management methods typically rely on static compression workflows or impose rigid constraints on compression timing and granularity. Moreover, these approaches fail to jointly optimize memory management and issue resolution capabilities to improve performance while reducing token usage. We present SWE-MeM, a training framework for proactive and on-demand memory management in software engineering agents. SWE-MeM provides a flexible memory tool that lets agents decide when, what, and how to compress based on trajectory state, task progress, and remaining context budget. We train agents with synthesized proactive memory-management trajectories and Memory-aware GRPO, which jointly optimizes memory management and issue resolution through memory-aware trajectory splitting and step-level credit assignment. On SWE-Bench Verified, SWE-MeM achieves 43.4% and 60.2% resolve rate with 4B and 30B models, respectively, outperforming existing memory management baselines in both performance and efficiency. 

---
# Spectral Perturbation of the Empirical Fisher Information Matrix under Weight Quantization 

**Authors**: Rahid Zahid Alekberli, Hikmat Karimov  

**Link**: [PDF](https://arxiv.org/pdf/2606.28432)  

**Abstract**: We study the spectral perturbation of the empirical Fisher Information Matrix (FIM) of a parametric statistical model under two structured perturbations: departure of the input from a reference (in-distribution) ensemble, and finite-precision (quantized) perturbation of the model's parameters. For the first, under an explicit local curvature-monotonicity hypothesis on the dominant eigenvalue lambda_max of the FIM, we show departure from a reference manifold provably elevates lambda_max relative to a calibration baseline (Proposition 3.2), and discuss why this hypothesis is required, since curvature need not increase monotonically under every perturbation. Our principal result is a directional eigenvalue perturbation bound, via Weyl's inequality, showing lambda_max under a quantization noise perturbation is lower bounded by its unperturbed value up to a third-order remainder, and, under a mild genericity condition, strictly exceeds it at leading order (Theorem 4.3). We give two tractable approximations to lambda_max -- one heuristic, one with a rigorous two-sided bound -- and a completeness result for a threshold-based partition of an augmented state space. These results motivate using sigma_t = lambda_max(F_t)/lambda_base as a runtime monitoring statistic for deployed language models: the quantization result offers a mechanism for an empirical observation of our own, where a calibration threshold for this statistic was approximately 244 times larger than a preliminary full-precision estimate on a 4-bit quantized model, a single measurement rather than a value derived in closed form. We report supporting measurements (twelve models, n=1,080 trajectories) broadly consistent with our predictions, discuss the scope and limitations of every result, and state as an open problem the closed-form prediction of the quantization inflation magnitude our bound does not supply. 

---
# Building to the Test: Coding Agents Deliver What You Check, Not What You Requested 

**Authors**: Yanuo Ma, Ben Kereopa-Yorke, Ben Schultz  

**Link**: [PDF](https://arxiv.org/pdf/2606.28430)  

**Abstract**: Benchmarks are widely used to evaluate task completion by Large Language Models (LLMs), but this approach has accumulated construction-validity problems, and a passing score may not show whether the requested task was delivered. We study both problems. In a controlled code-as-spec setup, two production Copilot CLI agents (claude-opus-4.7, gpt-5.5) re-implement a React Fluent-UI data table in Angular as a reusable library under a hidden 222-test Playwright oracle across 18 runs and three oracle-availability conditions. Alongside the score, we run a mechanical library audit and check each verdict with a no-op ablation. Without the oracle, the library is present but unfinished, revealed by scores. With the oracle in the loop, the score reaches near-perfect, but from a demo holding the tested behavior directly, the library left dead or absent. We call this building to the test; the broader disposition behind both we call validation self-awareness. The agent does not, on its own, validate what it ships as a user would. Prevalence remains an open question across other agents, signals, and model families. Beyond benchmark scores, dispositions like validation self-awareness merit research attention. 

---
# Tool Use Enables Undetectable Steganography in Multi-Agent LLM Systems 

**Authors**: Jimmy Laurence Rippin, Simon C. Marshall, David Demitri Africa, Christian Schroeder de Witt  

**Link**: [PDF](https://arxiv.org/pdf/2606.28425)  

**Abstract**: Increasingly autonomous agentic AI systems pose novel multi-agent risks, such as secret collusion via covert communication channels. The natural defence to these collusion attempts is to monitor plain-text communication, but the efficacy of monitors has been called into doubt by increasingly sophisticated model steganography; indeed, some theoretical schemes have been proposed that are information-theoretically or computationally indistinguishable from good-faith plain-text communication. In this paper, we demonstrate that the complexity of these schemes is no longer a safety barrier, as agentic coding models can already produce undetectable stegosystems when given realistic tool usage, such as code execution or accessing research papers through web searches. Agents also adapt when key ingredients are missing, for example, by adding model-sampling components or implementing related keyed coding schemes. We then frame tacit steganographic coordination between agents as a Schelling-point problem and introduce coordination metrics for estimating when two agents are likely to select compatible schemes without explicit prior agreement. Our results suggest a shift in the threat model for covert communication between AI agents, where the main barrier is no longer whether frontier agents can understand and implement sophisticated stegosystems, but coordination: whether independently acting agents can converge on compatible schemes, keys, and parameters. We find substantial convergence on broad scheme families but limited strict one-shot coordination, suggesting that shared artefacts, repeated interaction, and tool-mediated search are the settings where covert communication risks are most acute. Overall, our findings provide empirical grounding for the recent strategic confinement hypothesis, which assumes that capable agents can construct covert channels that survive monitoring. 

---
# JuZhou 1.0 Technical Report: The First Edge-Native Text-to-Image Foundation Model Trained Entirely on China-Developed AI Accelerators 

**Authors**: Ce Chen, Congrui Wang, Yonglin Li, Zhenchen Wan, Mingyang Geng, Junhao Xiao, Zhengpeng Xing, Yaqing Hu, Yao Wu, Zhaoyang Qu, Long Lan, Xinwang Liu, Yingqi Peng, Shijia Li, Zufeng Zhang, Chen Ma, Jingjing Zhou, Xingyu Wang, Qilin Lu, Bin Jiang, Qilin Sun, Shanzhi Gu, Yaoguang Jin, Tongliang Liu, Kede Ma, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2606.28421)  

**Abstract**: Text-to-image (T2I) diffusion models typically require substantial computational resources and cloud infrastructure, posing significant challenges for edge deployment in terms of latency, cost, and user privacy. We present JuZhou 1.0, an ultra-lightweight T2I foundation model designed for fully offline, on-device execution. JuZhou 1.0 achieves its efficiency through four key designs: (1) a compact image-generation backbone consisting of a 0.385B-parameter denoising U-Net and a 1.90M-parameter distilled decoder, totaling approximately 0.387B parameters; (2) Rectified Flow training combined with DMD2 distillation, reducing inference to 4 sampling steps; (3) Chinese semantic alignment trained on 9M curated image-text pairs, enabling direct Chinese prompting without external translation at inference time; and (4) a training and distillation pipeline completed on domestically developed Sugon K100 AI accelerators without relying on NVIDIA GPUs for training or distillation. Despite its compact scale, the 28-step base model of JuZhou 1.0 achieves an overall GenEval score of 0.69, outperforming published baselines including SDXL (2.6B, 0.55), SD3-Medium (2B, 0.62), and IF-XL (4.3B, 0.61). We further validate the full poetry-to-image pipeline on Android and the core CLIP-U-Net-VAE generation branch on iOS. On a smartphone powered by the Snapdragon 8 Elite Gen 5 Mobile Platform, the 4-step U-Net denoising branch runs in approximately 1.6 seconds, while the full Android poetry-to-image pipeline takes 4.5 seconds with on-device prompt refinement on Xiaomi 17 Pro Max. These results position JuZhou 1.0 as a practical approach to mobile text-to-image generation and provide a concrete reference for Chinese-native generation, domestic-compute training, and fully offline on-device deployment after one-time installation. 

---
# MedDiffuseMix: Preserving Diagnostic Evidence with Saliency-Aware Diffusion Medical Image Data Augmentatio 

**Authors**: Teerath Kumar, Raja Vavekanand, Muhammad Turab  

**Link**: [PDF](https://arxiv.org/pdf/2606.28419)  

**Abstract**: Limited data availability, class imbalance, and domain variability remain major barriers to reliable medical image classification. Conventional augmentation can improve training diversity but may distort diagnostically informative structures, whereas unconstrained generative augmentation may introduce label-inconsistent content. This paper proposes MedDiffuseMix, a saliency-guided diffusion mixing framework for controlled medical image augmentation. The method uses classifier-derived saliency maps to separate high-saliency diagnostic regions from low-saliency background areas and applies diffusion-guided mixing mainly to regions with lower diagnostic importance. Adaptive mixing, Gaussian boundary blending, and a saliency-preservation constraint reduce semantic distortion and reject or attenuate samples that shift model attention away from clinically relevant evidence. The framework is evaluated on four public benchmarks: the Radiological Society of North America pneumonia chest radiography dataset, Musculoskeletal Radiographs, PatchCamelyon, and the Breast Cancer Histopathological Image Classification dataset. Experiments with convolutional and transformer-based classifiers show that MedDiffuseMix improves accuracy, F1-score, and area under the receiver operating characteristic curve compared with standard augmentation, Mixup, GenMix, SaliencyMix, and diffusion-based augmentation baselines. Ablation studies confirm the importance of saliency guidance, adaptive region mixing, and smooth boundary blending. Visual attribution analysis further indicates that MedDiffuseMix better preserves diagnostically salient regions. These results suggest that saliency-guided diffusion mixing is an effective augmentation strategy for limited-data medical image classification. 

---
# AEGIS: A Semantic GAN and Evidential Learning Frameworkfor Robust Adversarial Detection in Vision Sensors 

**Authors**: Maher Boughdiri, Mounira Msahli, Albert Bifet  

**Link**: [PDF](https://arxiv.org/pdf/2606.28416)  

**Abstract**: Deep neural networks (DNNs) have shown outstanding performance in visual recognition tasks within vision sensor networks; however, they are still vulnerable to adversarial manipulations and imperceptible perturbations that can lead to erroneous predictions. To address that, this paper presents AEGIS, a semantic aware and uncertainty guided adversarial detection framework designed for robust image classification in vision sensors pipelines. At its core, a SemantiGAN module functions as a multi class semantic discriminator, identifying and filtering visually inconsistent adversarial inputs before they propagate further in the pipeline. For inputs that pass this stage, a stochastic augmentation process generates test time variations, from which handcrafted instability metrics FlipScore, Prediction Inconsistency, Layerwise Cosine Similarity (early and mid layers), and Entropy are computed. These features are aggregated into a compact five dimensional vector and processed by an Evidential Deep Learning (EDL) classifier, which models output evidence using a Dirichlet distribution to yield both class predictions and calibrated uncertainty estimates. Evaluations on the Tiny ImageNet dataset across six categories clean, FGSM, PGD, patch based, functional, and geometric attacks demonstrate the effectiveness of AEGIS. The proposed framework achieves an AUROC of 92.1\%, an AUPRC of 90.2\%, and an accuracy of 90.7\%, outperforming conventional softmax-based detectors in terms of detection performance, robustness, interpretability, and uncertainty calibration. 

---
# On the Necessity of a Liquid Substrate for Mesh Intelligence 

**Authors**: Hongwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28413)  

**Abstract**: A mesh of sovereign agents has no center: no shared clock, no shared model, and no coordinator to gather data or retrain. Its competence rests on each agent folding the projections its peers emit into a single internal state, online, from observations that arrive at irregular, unscheduled times, on a substrate whose weights it cannot retrain. Any one of these constraints is tractable on its own; folding optimally under all three at once is not. We ask what such a substrate must be, and prove two necessary conditions from one model of a self-evolving latent observed at irregular, exogenous times. Because the latent changes, its optimal estimator is time-varying: an adaptive timescale is necessary, and every fixed-gain filter is strictly suboptimal. And because arrivals are clock-free, the optimal estimate depends on the elapsed gap between them, which no gap-blind network recovers at any width or depth. This second condition is capacity-independent: scale cannot substitute for the missing dependence. The two conditions intersect in the continuous-time liquid class. An LSTM satisfies the first, a fixed continuous-time filter the second, and a multi-timescale liquid network both. Synthetic experiments confirm each: the network attains the timescale, and the separation is computed exactly. The characterization is necessary, not sufficient, and binds fixed-weight substrates: a network free to retrain reaches the class by other means. Proved per agent, the necessity binds every agent of a mesh, a structural condition on mesh intelligence. 

---
# RSGPNet: Geometric Prompting for Remote Sensing Open-Vocabulary Semantic Segmentation 

**Authors**: Shanwen Wang, Xin Sun, Sirui Wang, Xiao Xiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28410)  

**Abstract**: Open-vocabulary semantic segmentation (OVSS) enables text-guided segmentation of unseen objects, breaking fixed-class limitations to achieve open-world understanding. However, existing OVSS methods primarily focus on modifying the CLIP attention mechanism, which still suffers from unstable local segmentation for remote sensing (RS) domain. To address these limitations, we propose RSGPNet, a training-free geometric prompting framework for RS OVSS that refines segmentation by leveraging object geometric areas and consistency constraints. Specifically, RSGPNet comprises three core modules: a Text-guided Coarse Mask module (TCM), a Geometric Re-prompting Module (GRP), and a Coarse-to-fine Consistency Verification Mechanism (CVM). TCM utilizes text prompts and the input image to construct initial coarse segmentation masks. GRP then converts these coarse masks into geometric box prompts, feeding them back into the segmentation model to generate refined masks. Finally, CVM employs consistency computation to prevent prompting from reinforcing erroneous regions. They allow the model to improve segmentation accuracy in complex areas, such as category boundaries. Extensive experiments on RS datasets demonstrate that RSGPNet significantly outperforms state-of-the-art methods across both quantitative and qualitative metrics while exhibiting excellent interpretability. The code is released at \href{this https URL}{this https URL}. 

---
# Evidence-Driven LLM Agent for C-to-Synthesizable-C Conversion and Verification 

**Authors**: Zhe Zhao, Hongbing Lang, Zhihan Xiao, Luke Ztz Hu, John Imoleayo Adebisi, Songping Mai  

**Link**: [PDF](https://arxiv.org/pdf/2606.28409)  

**Abstract**: Software-compilable C programs routinely fail to complete the four-stage pipeline of a high-level synthesis (HLS) toolchain -- compilation, C simulation (CSim), synthesis, and C/RTL co-simulation (CoSim) -- because HLS accepts only a synthesizable subset of C (HLS-C). Yet most existing large language model (LLM) systems built for HLS code repair only cover the early pipeline stages and feed raw tool logs directly to the model, yielding brittle and hard-to-reproduce fixes. We formulate C-to-HLS-C conversion as a closed-loop generation-verification-diagnosis-repair problem on an HLS tool (Xilinx Vitis), contributing three components: an end-to-end workflow of cooperating agents closed by the four-stage verifier under strict evidence isolation; a Progressive Mismatch Localization Chain (PMLC) that localizes CSim/CoSim mismatches through log normalization, AST backward slicing, and dual-trace instrumentation; and a typed-query, two-stage evidence RAG backed by a self-evolving, family-routed repair-card pool. Experimental results show that the proposed workflow substantially outperforms all comparable state-of-the-art models. 

---
# Financing Artificial Intelligence Infrastructure: Mapping AI Infrastructure Investment and Compute Governance Across Africa 

**Authors**: Kai-Hsin Hung, Sumaya Nur Adan, Krupa Suchak, Armita Sadeghian Barzoki, Kofi Yeboah, Mohammad Amir Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2606.28404)  

**Abstract**: Artificial intelligence depends on large-scale compute resources and their supporting infrastructure. However, AI governance debates treat compute primarily as a technical input rather than as an outcome of investment, ownership, and financial control. This paper examines AI infrastructure investment flows across Africa through a systematic analysis of 46 publicly announced projects totalling USD $12.7 billion between 2019 and 2025. Using a value chain framework, we analyze who invests in AI-relevant infrastructure and where investments concentrate. Our findings reveal a highly concentrated landscape dominated by global data center operators, hyperscale technology firms, and development finance institutions, clustering in South Africa, Kenya, Nigeria, and Egypt. We introduce asymmetrical interdependence to describe a structural condition in which capital and physical infrastructure account for 73% of total funding while control remains concentrated in the compute layer among a small number of global technology firms. We argue that compute governance must account for capital flows, ownership, and control, not only geographic access, because these dynamics shape AI compute equity. Infrastructure presence is necessary but insufficient for meaningful governance capacity. 

---
# Reinforcement Learning for Software Vulnerability Analysis: A Systematic Review with Emphasis on C/C++ Source Code and Static Analysis 

**Authors**: Bruno Caro-Vásquez, Carola Figueroa-Flores, Gastón Marquez  

**Link**: [PDF](https://arxiv.org/pdf/2606.28403)  

**Abstract**: Vulnerability detection in C/C++ software remains a major security challenge due to code complexity, manual memory management, and the limitations of traditional static analysis. Reinforcement Learning (RL) has emerged as a promising approach, particularly for fuzzing, test generation, program exploration, and, more recently, vulnerability detection and localization. Following PRISMA 2020 guidelines, this work reviews RL techniques for software vulnerability analysis, focusing on C/C++ source code and static analysis. We identified 21 primary studies published between 2015 and 2026 from major scientific databases and complementary searches. We analyze the addressed tasks, algorithms, state-action-reward-environment formulations, code representations, datasets, and evaluation metrics. Results show that 15 studies focus on fuzzing and guided exploration, only 3 on direct vulnerability detection, and just 1 on statement-level localization. Moreover, statically extracted structural representations such as Control Flow Graphs (CFGs) and Abstract Syntax Trees (ASTs) are rarely used as agent states, and benchmarks lack comparability. We propose a task- and formulation-oriented taxonomy and identify a key research gap: the absence of RL agents that use source-code CFGs as states to detect and localize vulnerable nodes. 

---
# CLOSER-VLN: Closed-Loop Self-Verified Retrieval-Augmented Reasoning for Aerial Vision-Language Navigation 

**Authors**: Shaoxuan Li, Xiangyu Dong, Xiaoguang Ma, Junfeng Chen, Haoran Zhao, Yaoming Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.28397)  

**Abstract**: Vision-language navigation (VLN) has recently advanced with large language and multimodal models, enabling agents to follow natural-language instructions in unseen environments without training a task-specific navigation policy. However, most existing VLN methods relying on large models still adopt an open-loop decision-execution approach, where candidate actions are generated from instructions and observations but are rarely verified or corrected before execution. This causes critical issues in aerial VLN, where minor errors in intermediate actions may quickly accumulate into large trajectory deviations and lead to target loss. To address this issue, we propose Closed-loop Self-verified Retrieval-augmented Reasoning (CLOSER), a training-policy-free method that sequentially performs action reasoning, reliability verification, targeted retrieval, and action correction in a closed-loop manner before executing concrete actions. We instantiate the CLOSER in aerial VLN tasks and develop a CLOSER-VLN framework, which is composed of three components: a hierarchical reasoner for generating candidate actions based on available information, a multidimensional action verifier for assessing the reliability of actions generated by the reasoner, and a verification-triggered multimodal retriever for retrieving targeted exemplars from a memory bank only when verification fails. We conduct experimental evaluations on the CityNav benchmark, where CLOSER-VLN achieves 32.01% SR and 21.28% SPL on the test-unseen split, confirming the effectiveness of closed-loop reasoning. 

---
# RADIANT-PET: Reasoning-Augmented PET/CT Lesion Segmentation with Large Language Models and Reinforcement Learning 

**Authors**: Jiasheng Wang, Tanun Jitwatcharakomol, Piyawadee Jongpradubgiat, Simeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28392)  

**Abstract**: Accurate lesion segmentation in PET/CT is critical for oncology, yet remains challenging because physiologic tracer uptake and artifacts can mimic malignant signal. We present RADIANT-PET, a reasoning-augmented framework that couples a high-sensitivity voxel-level segmentation model with lesion-level large language model (LLM) adjudication. Candidate uptake regions are generated with a deliberately permissive segmentation stage, then converted into structured textual descriptions that summarize uptake intensity, morphology, and regional and global anatomical context. An LLM classifies each candidate as true lesion vs. false positive, optionally leveraging the radiology report as additional clinical context. To strengthen lesion-level reasoning, we further optimize a local LLM via reinforcement learning using Group Relative Policy Optimization, rewarding correct lesion classification and anatomically concordant site assignment. Across AutoPET and an OSU test cohort, RADIANT-PET consistently outperforms strong image-only baselines, with the largest improvements observed when radiology reports are provided. Overall, these results demonstrate that LLM-based lesion-level reasoning adds a novel reasoning layer beyond conventional segmentation, suppressing physiologic false positives and aligning voxel-level predictions with clinical interpretation. The project repository is available at: this https URL. 

---
# Few-class Fidelity: Evaluating Explanations of Real-conditions CNN classifiers with Optimized Perturbations 

**Authors**: Wistan Marchadour, Pedro Soto Vega, Franck Vermet, Mathieu Hatt  

**Link**: [PDF](https://arxiv.org/pdf/2606.28391)  

**Abstract**: The wide use of Convolutional Neural Networks (CNN) in numerous domains and real-world classification applications is justified by their high precision and automation speed, helping users concentrate on higher-expertise tasks. To better understand the models and avoid bias during deployment, eXplainable Artificial Intelligence (XAI) techniques can be used after training. But as the list of XAI solutions expand, comparisons between them diverge, and consensus over their evaluation cannot be reached. This paper proposes a variation of Fidelity-based XAI metrics, with a focus on real-conditions applications, where the number of classes is often low. The approach generates in-distribution, uncertainty-provoking perturbations, to ensure proper measurement of the XAI methods faithfulness. As demonstration of the evaluation framework usefulness, it is compared with human-centric object localization and segmentation metrics. Once applied to both medical and natural imaging applications, it highlights the intricate correlation between domain, data curation, and XAI solution choices in order to validate training of a new CNN model. 

---
# Automated Quality Assessment of Geospatial Vector Data: A GeoAI Approach using Spatial Representation Learning 

**Authors**: Hao Li, Chen Chu, Filip Biljecki, Cyrus Shahabi, Wenwen Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.28390)  

**Abstract**: Geospatial vector data quality is a foundational research topic in GIS, yet classic rule-based quality assessment algorithms often struggle with diverse urban morphologies and massive data volumes. Recently, Geospatial Artificial Intelligence (GeoAI) shows promising potential for automating geospatial analysis, while its application to native vector data remains largely underexplored. To fill this research gap, we proposed Topo4Vec, an automated GeoAI framework, designed for scalable vector data quality assessment via advanced Spatial Representation Learning (SRL). Specifically, Topo4Vec relax the labor-intensive manual annotation process via topological error simulation, such as overlapping polygons and street network connectivity errors e.g., overshoots and undershoots. Then, it leverages state-of-the-art SRL approaches to encode complex, native vector geometries (e.g., polylines and polygons) into a latent space where topological errors are isolated from valid ones. A systematic performance evaluation across three study areas (Los Angeles, Munich, and Singapore) demonstrates the effectiveness and robustness of Topo4Vec, achieving a peak accuracy of 0.99 for detecting overlapping building footprints and 0.60 for overshoots and undershoots in street networks. Moreover, lessons learned from Topo4Vec shed a promising light into a scalable and autonomous GeoAI approach for large-scale vector data consistency and quality monitoring within the fast-growing geospatial data ecosystems. The code and data used in the paper are made openly available in this https URL. 

---
# Schema-First Retrieval: Embedding Catalogs for Natural Language Analytics 

**Authors**: Adarsh Agrawal, Shashank Indukuri  

**Link**: [PDF](https://arxiv.org/pdf/2606.28387)  

**Abstract**: Enterprise text-to-SQL systems often fail before SQL is generated: the model receives the wrong schema context. Modern warehouses contain thousands of tables, abbreviated columns, informal metrics, hidden join conventions, and permission boundaries that are not captured by raw table names. We introduce Schema-First Retrieval, a retrieval layer that embeds catalog metadata rather than warehouse rows. The system indexes five typed catalog objects, tables, columns, metrics, relationships, and query history, using object-specific text templates. At query time, it combines parallel vector search, lineage expansion, cross-encoder reranking, workload memory, and deterministic access-control gates before SQL generation. On CRUSH4SQL (1,534 questions), Schema-First Retrieval reaches 96.4% table recall@20 and cross-encoder reranking adds +11.1 points at column recall@10; against an equally-templated BM25 baseline, semantic retrieval is +32.8 points at table recall@5. On SEDE (857 questions), query history raises table recall@5 from 52.1% to 92.3%. On BIRD (96 questions), schema-first context reduces SQL execution errors from 15.6% to 6.2%, a 2.5x reduction. These results show that catalog selection is a first-class retrieval problem for natural language analytics, not a prompt formatting detail. 

---
# Data Provenance for Image Auto-Regressive Generation 

**Authors**: Bihe Zhao, Louis Kerner, Michel Meintz, Tameem Bakr, Franziska Boenisch, Adam Dziedzic  

**Link**: [PDF](https://arxiv.org/pdf/2606.28386)  

**Abstract**: Image autoregressive models (IARs) have recently demonstrated remarkable capabilities in visual content generation, achieving photorealistic quality and rapid synthesis through the next-token prediction paradigm adapted from large language models. As these models become widely accessible, robust data provenance is required to reliably trace IAR-generated images to the source model that synthesized them. This is critical to prevent the spread of misinformation, detect fraud, and attribute harmful content. We find that although IAR-generated images often appear visually identical to real images, their generation process introduces characteristic patterns in their outputs, which serves as a reliable provenance signal for the generated images. Leveraging this, we present a post-hoc framework that enables the robust detection of such patterns for provenance tracing. Notably, our framework does not require modifications of the generative process or outputs. Thereby, it is applicable in contexts where prior watermarking methods cannot be used, such as for generated content that is already published without additional marks and for models that do not integrate watermarking. We demonstrate the effectiveness of our approach across a wide range of IARs, highlighting its high potential for robust data provenance tracing in autoregressive image generation. 

---
# RoboGaze: Evaluating Robot World Models via Structured Vision-Language Analysis 

**Authors**: Minh-Loi Nguyen, Nghiem Tuong Diep, Hung Khang Nguyen, Minh Le, Doanh Le Thien, Hoang H. Tran, Dung D. Le, Vu N. Duong, Daniel Sonntag, An Thai Le, Duy Minh Ho Nguyen, Vien Anh Ngo, Tran Van Nhiem  

**Link**: [PDF](https://arxiv.org/pdf/2606.28385)  

**Abstract**: Recent advances in robot world models enable synthetic video generation for embodied prediction and planning. However, evaluating these videos is challenging: visually realistic outputs often violate physical laws, temporal consistency, or task logic, while conventional metrics and monolithic Vision-Language Model (VLM) judges fail to generalize or provide precise diagnostic value. We present RoboGaze, a training-free, multi-agent VLM framework that provides structured, interpretable evaluation for generated robot-manipulation videos. Given a task instruction and video, RoboGaze operates via a three-stage pipeline: task-scene grounding, dimension-specific specialist routing, and critic-based verification. It outputs temporally localized glitch reports categorized under a novel 6-dimension, 30-type robotics-specific taxonomy. To benchmark RoboGaze, we introduce a human-validated dataset of 382 clips spanning simulated and real-world multi-view manipulation. Evaluating eight open-source and proprietary VLM backbones, RoboGaze dramatically outperforms zero-shot baselines, improving description-F1 by up to +43 points and temporal alignment (F1 x IoU) by up to +37 points, closing approximately 85% of the gap to the human ceiling. Furthermore, its critic verifier mitigates the "cry-wolf" false-positive flaw of standard VLMs, lifting clean-clip accuracy from under 25% to over 80%. RoboGaze offers a scalable, highly interpretable diagnostic tool for the rigorous evaluation of robot world models. 

---
# A Query-Driven Communication-Efficient Digital Twins Design for Autonomous Driving 

**Authors**: Nuocheng Yang, Longyu Zhou, Sihua Wang, Changchuan Yin, Tony Q. S. Quek  

**Link**: [PDF](https://arxiv.org/pdf/2606.28384)  

**Abstract**: Digital twins (DTs) have become a potential technology to perform risk-free simulation of physical entities for deterministic and high-reliability services in diverse scenarios such as autonomous driving and low-altitude economy. In the autonomous driving scenario, traditional DT methods that rely solely on vehicle's real-time state synchronization, however, might lead to unacceptable computing and communication consumption for construction of high-fidelity DT with redundant data. To address this issue, we first propose a query-driven DT architecture to enable the DT to actively request the desired environment data from vehicles based on its simulation result. Then, we formulate an optimization problem whose goal is to minimize autonomous driving position error while accounting for DT fidelity and communication constraints. We also design a cross-time-step progressive query mechanism to further improve communication efficiency. The simulation results show that our proposed method achieves a 24% reduction in planning position error compared to traditional methods, while reducing communication overhead by 40%. 

---
# Evolutional Math: Cross-Validated Island-Model Genetic Programming for Interpretable Symbolic Regression on Small, Wide Datasets 

**Authors**: Artem Andrianov  

**Link**: [PDF](https://arxiv.org/pdf/2606.28381)  

**Abstract**: Symbolic regression via genetic programming routinely fails on small, wide datasets - a regime common in clinical-trial monitoring, biostatistics, and engineering pilot studies - by converging on bloated, overfit expressions that exploit correlation rather than prediction. We present Evolutional Math, an open-source genetic programming system that combines four design choices to yield compact, interpretable formulas in this regime. First, fitness is measured by R-squared on held-out cross-validation folds rather than Pearson correlation on the training set, eliminating single-variable shortcuts that correlate but mis-scale. Second, a multi-island architecture runs independent populations seeded with distinct operator subsets (algebraic, logarithmic, trigonometric, and full) with ring-topology migration every M generations, preventing the search from collapsing into one region of formula space. Third, a structural deduplication scheme treats formulas differing only in constants as equivalent, so the elite archive contains structurally distinct candidates rather than near-duplicate variants. Fourth, top-k individuals undergo numerical constant refinement via scipy L-BFGS-B after each migration phase, decoupling structure search from parameter fitting. We evaluate the system on synthetic benchmarks of the form log(x_i) * x_j / (x_k * c), trigonometric mixtures, and an anonymized clinical site-monitoring dataset with 24 rows and approximately 290 candidate numeric features. The system consistently recovers compact ground-truth structures with R-squared at or above 0.99 within tens of thousands of unique formula evaluations. A reference implementation is released under a noncommercial source-available license. 

---
# Distilling a Modular Reservoir Through a Genomic Bottleneck 

**Authors**: Mani Hamidi, Sina Khajehabdollahi, Charley M. Wu, Emmanouil Giannakakis  

**Link**: [PDF](https://arxiv.org/pdf/2606.28380)  

**Abstract**: The intricate structures of biological neural networks largely emerge during development, guided by a comparatively compressed blueprint encoded in the genome. The connectivity that emerges from this decoding process is rich in structure, and already equips the organism with functional modules upon birth. This initial structure serves as a scaffold that can be gradually refined and fine-tuned through lifelong experience, via a variety of plasticity mechanisms. Drawing inspiration from this interaction between evolutionary and developmental modes of learning, we use hypernetworks to learn a compressed generative process that generates the connectivity of a modular reservoir. We show that this marriage between curriculum-based meta-learning and modular reservoir computing can generate sparse recurrent networks that solve difficult temporal tasks with minimal training and without concessions to robustness. 

---
# LEDGER: Scaling Agentic Document Editing with Dependency-aware Graph Retrieval 

**Authors**: Mike Hang Wang, Utkarsh Garg, Reza Davari, Huitian Jiao, Hao Cheng, Baolin Peng, Tao Ge, Si-Qing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.28379)  

**Abstract**: We introduce LEDGER to tackle the novel context engineering challenge of agentic document editing, where localized edits to long, structured documents must be applied efficiently without breaking cross-references or semantic consistency. LEDGER constructs a lightweight dependency graph that explicitly models document structure, including hierarchical organization, explicit references, implicit dependencies, and semantic relationships. For each edit, graph-guided retrieval selects only the necessary context, avoiding full-document processing while preserving consistency. We evaluate LEDGER on a curated benchmark of 1.9k test cases with various document types and lengths, spanning six state-of-the-art models: LEDGER improves consistency from 56% to 76% across all six models and test scenarios while reducing token usage. Notably, LEDGER with low reasoning effort matches baseline performance at high reasoning effort using fewer tokens, showing that explicit dependency representations can partially substitute for expensive internal reasoning in agentic document editing. 

---
# Memory-Augmented LSTM Autoencoder for Unsupervised Activity Recognition with IMU Sensor Fusion 

**Authors**: Saeid Arabzadeh, Farshad Almasganj, Mohammad Mahdi Ahmadi  

**Link**: [PDF](https://arxiv.org/pdf/2606.28377)  

**Abstract**: HAR using Inertial Measurement Unit (IMU) sensors is vital for healthcare monitoring and rehabilitation. Despite deep learning advancements, major challenges remain: reliance on labeled data, multi-sensor fusion complexity, and the limited ability of unsupervised methods to capture spatiotemporal dependencies. These issues are pronounced in real-world scenarios with noisy data, overlapping activities, and missing labels. We propose a fully unsupervised spatiotemporal feature fusion framework using a memory-augmented autoencoder. It enhances activity representations via short temporal windows of multi-sensor IMU data, enabling real-time applications. Our framework extracts hierarchical static features via a Stacked Autoencoder, fusing them within and across sensors. A sequence-to-sequence LSTM Autoencoder then temporally refines these features, incorporating historical motion patterns without labels. We analyze key hyperparameters to identify configurations that maximize feature separability under short-window constraints. Evaluated on DaLiAc and PAMAP2 using realistic inter-class window segmentation, our method achieves 96.6% and 98.4% accuracy, respectively, surpassing supervised baselines and unsupervised approaches. Our method improves feature separability by up to 9% despite shorter temporal windows. While our realistic inter-class segmentation reduces accuracy by ~7%, it was intentionally adopted to better reflect real-world activity transitions and practical relevance. 

---
# When Does Overlap Help? OSU-Mem and a Cell-Conditional Analysis of Trajectory Memory for LLM Agents 

**Authors**: Mellow Baixuan Chen, Xiangguo Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.28376)  

**Abstract**: Long-horizon large language model (LLM) agents accumulate interaction trajectories that quickly exceed any practical prompt budget, and existing memory methods either truncate aggressively and lose non-local evidence or retain boilerplate that degrades decision quality. We ask a mechanism question rather than claiming a better general-purpose memory system: when does organizing trajectory memory into overlapping semantic units (OSUs) -- groups of related steps in which one step may belong to several units -- help retrieval over flat or disjoint alternatives? We instantiate this in OSU-Mem, which retrieves from an overlapping OSU pool via budgeted coarse-to-fine expansion, and show its benefit is conditional: overlapping memory helps when the evidence steps a query needs share tool calls or entities, but hurts when those steps are fully heterogeneous and share neither. On a synthetic benchmark where evidence carries such shared structure by construction, OSU-Mem improves over the strongest baseline as the theory predicts; yet on a concatenated, constructed unaugmented $\tau$-bench setting its aggregate advantage over flat retrieval vanishes. Splitting queries by whether their evidence shares tools and entities shows this near-tie to be an artifact of mixing query types rather than a property of either method, and ToolBench, a controlled probe built to carry shared structure by design, corroborates the same mechanism via an overlap-vs.-disjoint construction contrast (under a coverage-guided variant), isolating the construction principle rather than validating the full default system. Because the relevant sharing is cheaply estimable from metadata, the analysis yields a metadata-based heuristic for predicting when overlap is likely to improve retrieval. We deliberately isolate the retrieval layer, assessed by retrieval quality and an LLM-mediated evidence-selection stage. 

---
# Model Merging to Evolution: Parameter Space Exploration for Expert Models 

**Authors**: Chao Wang, Yuchen Guo, Zheng Tan, Guanchun Wang, Yanbiao Ma, Qiqi Duan, Peng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28373)  

**Abstract**: Model merging integrates the capabilities of multiple expert models to create strong models for multiple tasks without additional training, thereby reducing computational resource requirements. However, existing methods operate within the convex combination space of expert models, failing to explore high-performance regions outside this space. This paper proposes the MERGEvolve framework, which unifies model merging and evolution within an evolution strategy by treating the merged model as the initialization for evolutionary exploration of the parameter space. During the merging phase, expert models act as deterministic sources to build a strong initial point. The evolution phase then explores the parameter space using random noise. Theoretical analysis shows that MERGEvolve explores regions outside the convex combination space. Extensive experiments on single-task and multi-task benchmarks demonstrate that MERGEvolve consistently achieves performance competitive with advanced model merging baselines. Ablation studies confirm that a high-quality initial point is critical for efficient exploration of the parameter space. 

---
# Conversational Query Engine for Mixed-Modality Heterogeneous Enterprise Data Sources 

**Authors**: Darshita Rathore, Vineet Kumar, Vaibhav Singal, Ankur Vivek Singh, Anindya Moitra  

**Link**: [PDF](https://arxiv.org/pdf/2606.28370)  

**Abstract**: Enterprise business intelligence queries span structured warehouses and unstructured document repositories -- modalities with fundamentally different access methods, cost profiles, and correctness semantics. Existing AI-enabled interfaces force users to select the right tool: NL2SQL systems cannot reason over slide decks, and RAG pipelines lack access to live warehouse tables.
We present COGNI, a production conversational BI system that treats natural-language analytics as a heterogeneous query processing problem, organized as four architectural layers. First, an indexing layer implements slide-adaptive chunking -- recursive chunking for plain-text slides, hierarchical chunking for structured content such as tables, charts, and key-value blocks - achieving $88.3\%$ on our internal enterprise benchmark. Second, a routing layer built on a LoRA fine-tuned Qwen-2.5-1.5B-Instruct model that produces a dual output - modality decision and complexity assessment at $93.8\%$ accuracy and approximately $7\times$ lower cost than frontier-model. Third, a retrieval layer executes complexity-adaptive pipelines: a self-correcting NL2SQL agent at $93.9\%$ G-Eval, and Recursive Language Models reaching $91.0\%$ on multi-hop synthesis queries. Finally, a caching layer validates query equivalence across multiple dimensions beyond embedding similarity, achieving zero false cache hits and $8.4\times$ latency reduction. 

---
# Multimodal and Multiscale Spatial-Temporal Semantic Search and Recommendation with AI Foundation Models 

**Authors**: Yuanyuan Tian, Wenwen Li, Xiao Chen, Michael Brook, Michael Brubaker, Anna Liljedahl, Chitta Baral  

**Link**: [PDF](https://arxiv.org/pdf/2606.28369)  

**Abstract**: Semantic search and recommendation of similar documents, such as news and reports about unusual environmental events (e.g., a dead whale washed ashore in Alaska) that contain spatial and temporal information, is a critical task in Geographic Information Retrieval (GIR). This work presents a novel framework that leverages AI foundation models, including Large Language Models (LLMs) and Vision-Language Models (VLMs), to enable effective similarity search and ranking for such event documents. To support this goal, we introduce two new strategies: (1) CAMERA (Context-Aware Multimodal Event Retrieval Algorithm), which fuses textual and visual information to generate richer embeddings than those derived from text alone; and (2) ASTRA (Adaptive Spatial and Temporal Re-ranking Algorithm), which improves similarity ranking by incorporating scale-dependent spatiotemporal relevance alongside semantic similarity. Experimental results, using a dataset from the Local Environmental Observer Network, demonstrate that our VLM-enhanced methods outperform unimodal, LLM-based approaches in similarity ranking effectiveness. By automatically linking relevant event reports, the proposed framework helps both data curators and the general public gain deeper insights into environmental change and its localized impacts. These findings highlight the potential of AI foundation models to advance GIR through multifaceted, intelligent analysis that integrates key geographic concepts: space, time, scale, and semantics. 

---
# Beyond the Reranker: Do RAG Retrieval Enhancements Help Once a Strong Reranker Is Present? 

**Authors**: Sadanand Singh, Allam Reddy, Manan Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2606.28367)  

**Abstract**: Retrieval-augmented generation (RAG) is routinely extended with methods meant to improve retrieval: query expansion, hierarchical and cross-document summarization, graph-based expansion, per-query routing, rank fusion, and corrective re-retrieval. The benefits reported for these methods come almost exclusively from homogeneous corpora, predominantly Wikipedia prose. Whether they hold on the mixed-format collections common in practice, where code, markdown, tables, scientific PDFs, and prose are interleaved within one corpus, has not been measured. To study this directly, we build \textbf{HetDocQA}, a heterogeneous benchmark with \emph{chunker-agnostic} span-overlap relevance labels and collection-disjoint splits, and pair it with MuSiQue and QASPER as homogeneous controls. We evaluate eight methods on a shared backbone, with bootstrap confidence intervals and multiple-comparison correction. A strong cross-encoder reranker accounts for most of the pipeline's quality; beyond it, only two methods yield reliable gains: query expansion and SSCC. SSCC, a per-source calibrated corrector introduced here, sets a separate acceptance threshold for each score source and helps only on heterogeneous data. The remaining reranking and pool-expansion methods in common use, among them hierarchical summarization, graph expansion, routing, and rank fusion, give no reliable gain once that reranker is present. 

---
# CAMI: Cost-Aware Agent-Guided Multi-Indexing for Semantic Retrieval 

**Authors**: Adnan Qidwai, Anand Eswaran, Sonam Mishra, Jaydeep Sen, Sachindra Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2606.28365)  

**Abstract**: RAG ingestion pipelines frequently augment search corpus index with semantic enrichment indices (e.g., synthetic queries or summaries generated from corpus chunks) that are subsequently queried alongside the base index to improve retrieval via better alignment between document representations and user intent. While these supplementary representations substantially improve retrieval quality, they introduce a computational bottleneck: the configuration space of enrichment types and generator models is combinatorial, and the cost of exhaustive index-time evaluation scales linearly with corpus size. We introduce CAMI (Cost-Aware Multi-Indexing), a framework that formalizes multi-index construction as a budgeted, multi-objective portfolio selection problem. CAMI targets the upstream decision of which enrichment views to generate and materialize before the retrieval backend is applied. CAMI incorporates three primary mechanisms: (i) an agentic discovery phase that proposes corpus-specific representation templates; (ii) an atomic-unit search procedure that evaluates individual enrichment-model pairs and recombines them via fidelity-local closure to identify synergistic portfolios; and (iii) a confidence-aware promotion schedule that prunes unpromising configurations early, decoupling optimization spend from total corpus size. We evaluate CAMI across diverse retrieval corpora. Our findings reveal that the framework systematically isolates high-recall portfolios under strict budget constraints, outperforming standard content-only baselines in challenging settings by up to 9.4% recall@10. Further, CAMI is able to systematically identify these high-recall portfolios using up to 5x less budget compared to random search baselines, making our approach practical in real production scenarios. 

---
# meta-pipe: An LLM-agent pipeline for end-to-end automated systematic review and meta-analysis 

**Authors**: Hsieh-Ting Lin, Jiunn-Tyng Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2606.28363)  

**Abstract**: Objective: To describe the architecture and design rationale of meta-pipe, an open-source large language model (LLM)-agent pipeline that integrates the complete systematic review and meta-analysis (SR/MA) workflow -- from literature search through statistical analysis, manuscript generation, and quality assurance -- with mandatory human oversight at critical decision points.
Study Design and Setting: We developed a 10-stage modular pipeline integrating Claude (Anthropic; Opus 4 for reasoning, Haiku 3.5 for classification) for LLM-assisted screening and extraction, Python (~3,600 lines of code) for automation, R (meta, metafor, gemtc, netmeta) for statistical analysis, and Quarto for manuscript rendering. Five mandatory human decision points enforce oversight. We systematically compared meta-pipe's capabilities with five existing SR automation tools based on published documentation as of March 2026.
Results: meta-pipe offers four capabilities not available in any single existing tool: automated manuscript generation from analysis outputs, semi-automated GRADE assessment, overclaim detection (12 predefined patterns), and dual-paradigm network meta-analysis (Bayesian and frequentist). Estimated API cost is $15-30 per typical 5-10 study review. No validation data are reported; this is a system description, not a validation study.
Conclusion: End-to-end AI-assisted evidence synthesis is architecturally feasible as an open-source tool with mandatory human oversight. Formal validation reproducing published Cochrane reviews is underway and essential before routine use. 

---
# LUMEN: Cost-Transparent Multi-Agent Pipeline for Automated Systematic Review and Meta-Analysis 

**Authors**: Yen-Hsun Huang, Yu-Shiou Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.28362)  

**Abstract**: Systematic reviews and meta-analyses (SR/MA) remain the gold standard for evidence synthesis, yet completing one typically requires 67 weeks and substantial expert effort. Recent large language model (LLM) systems have demonstrated strong performance on individual SR phases - screening (otto-SR: 96.7% sensitivity), extraction (Gartlehner et al.: 91.0% accuracy), and search (TrialMind: 0.83 recall) - but no study has reported what it actually costs to run an end-to-end pipeline, how cost distributes across phases, or how architectural choices affect the cost-quality trade-off. We present LUMEN, an open-source multi-agent pipeline that automates six SR/MA phases using 11 specialized LLM agents with deliberate model routing. We evaluate LUMEN on seven datasets: five self-conducted domain reviews (psychiatry, psychology, surgery, vaccinology, cardiology) and two SYNERGY screening benchmarks. Across 13 ground-truth-comparable outcomes, LUMEN achieves 100% directional agreement with published meta-analyses, with effect sizes within 1% for homogeneous study designs. The primary contribution is the first empirical cost and operational characterization of such a pipeline: a complete review costs 19 to 29 USD (median 22.65 USD), with title-abstract screening and data extraction together dominating expenditure. A three-arm extraction ablation reveals a phase-dependent architecture reversal: multi-agent design hurts screening but is essential for extraction, producing 5.7x more poolable analyses than single-model alternatives while eliminating clinically dangerous direction errors. A two-dataset screening benchmark demonstrates that model ranking is domain-dependent and not transferable across review topics. All code and cost logs are publicly available. 

---
# ConCise: Training-Free Conclusion-Chain State Compression for Cost-Efficient Multi-Step RAG Services 

**Authors**: Kuan Yan, Zhiqing Tang, Tian Wang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2606.28361)  

**Abstract**: Multi-step retrieval-augmented generation (RAG) has been widely deployed as LLM-powered web services for complex question answering, where iterative retrieval-reasoning rounds deliver strong multi-hop accuracy. However, this paradigm causes historical documents and reasoning traces to accumulate across rounds, inflating cumulative input tokens approximately as $O(N^2)$ with progressively increasing noise density. In API-based service architectures, such growth directly amplifies per-request billing cost, network payload, and response latency. Existing compression approaches rely on pretrained modules or GPU-level KV cache access, introducing model hosting overhead incompatible with API-native, Serverless, and edge-side deployments. To address this issue, this paper proposes ConCise, a training-free state-layer protocol that restructures cross-round context transmission for multi-step RAG services. Specifically, ConCise replaces raw-text accumulation with an append-only chain of structured conclusions, compressing cumulative context growth from $O(N^2)$ to approximately $O(N)$. Furthermore, a fused generation mechanism is introduced to jointly emit reasoning and conclusions in a single API call, eliminating repeated input billing from serial dual-invocation overhead. Extensive experiments across twelve paired configurations spanning three models, two datasets, and two representative frameworks demonstrate that ConCise achieves 64.63\% average token savings while maintaining acceptable accuracy, providing a plug-and-play, deployment-friendly solution for cost-efficient multi-step RAG service optimization. 

---
# Carolina Guide: A Multi-Agent RAG System with Institutional Guardrails for Academic Policy Assistance 

**Authors**: Ben Torsion, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.28360)  

**Abstract**: University students often struggle to navigate complex academic policies, leading to advising bottlenecks and delayed access to critical information. Although large language models (LLMs) offer promise for automated assistance, their tendency toward hallucination and inability to enforce institutional constraints make them unsuitable for high-stakes policy guidance without careful architectural design. We present Carolina Guide, a retrieval-augmented generation (RAG) system for academic policy assistance at the University of South Carolina (USC). The system employs a modular multi-agent pipeline with institutional guardrails to provide citation-supported, policy-grounded answers to student queries while refusing unsafe requests such as course recommendations or personalized advising. We evaluate the system on a 90 query test set across 6 departments, achieving 98.9% retrieval success at the >= 2 threshold (genuinely relevant results) with the first relevant chunk at rank-1 for 98.9% of queries (MRR at 10 for rel >= 2 = 0.989). Through systematic baseline comparisons and ablation studies, we show that each architectural component-MMR reranking, adequate retrieval context (k=20), and citation enforcement-contributes measurable practical value despite limited statistical power at 90 queries. The evaluation of the guardrail on 30 adversarial queries demonstrates Safety F1 of 0.89, correctly refusing 86% of unsafe queries while maintaining 93% coverage of benign queries. These results show that production-ready LLM systems for institutional policy guidance require rethinking standard RAG patterns to prioritize safety, transparency, and departmental autonomy over conversational sophistication. 

---
# How Do LLMs Cite? A Mechanistic Interpretation of Attribution in Retrieval-Augmented Generation 

**Authors**: Ian van Dort, Maria Heuss  

**Link**: [PDF](https://arxiv.org/pdf/2606.28358)  

**Abstract**: Retrieval-Augmented Generation (RAG) aims to enhance the trustworthiness of Large Language Models (LLMs) by grounding their outputs in external documents, often using inline citations for verifiability. However, the faithfulness of these citations -- whether the model genuinely uses a source to generate an answer -- remains a critical, unverified assumption. This paper offers the first mechanistic account of how a large language model decides whether to attach an inline citation while answering a factoid question. Using the Llama-3.1-8B-Instruct model in a controlled experimental environment based on the PopQA dataset, we employ an activation patching approach. We map the underlying mechanism responsible for citation, discovering that it is not a single, localized component but a distributed, multi-stage "attributional ensemble" of attention heads and MLP layers. We show that amplifying or attenuating only those critical heads and MLPs repairs over 90% of missed citations and eliminates 69% of spurious ones on PopQA without harming answer accuracy. Although gains on the multi-document HotpotQA benchmark are modest, the same component set still moves citation rates in the intended direction, indicating that the underlying mechanism is not dataset-specific. The results reveal a potential disconnect between the model's apparent reasoning and its internal computational pathway, suggesting that inline citations can create a false sense of security. 

---
# ReasonRec: A Reasoning-Augmented Multimodal Agent for Unified Recommendation 

**Authors**: Yihua Zhang, Mingfu Liang, Jiyan Yang, Rong Jin, Wen-Yen Chen, Yiping Han, Huayu Li, Buyun Zhang, Liang Luo, Frank Shyu, Luke Simon, Sijia Liu, Tianlong Chen, Xi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28357)  

**Abstract**: Recent advances in multimodal recommenders excel at feature fusion but remain opaque and inefficient decision-makers, lacking explicit reasoning and self-awareness of uncertainty. We introduce ReasonRec, a reasoning-augmented multimodal agent structured around a three-stage explicit reasoning pipeline. Specifically, we propose a reasoning-aware visual instruction tuning strategy that systematically transforms diverse recommendation tasks into unified CoT prompts, enabling the VLM to explicitly articulate intermediate decision steps. Additionally, our evidence-horizon curriculum progressively enhances the reasoning complexity to better handle cold-start and long-tail user scenarios, significantly boosting model generalization. Furthermore, the uncertainty-guided delegation mechanism empowers the agent to assess its own confidence, strategically allocating computational resources to optimize both recommendation accuracy and inference efficiency. Comprehensive experiments on four standard recommendation tasks across five real-world datasets demonstrate that ReasonRec achieves over 30% relative improvement in key ranking metrics compared to state-of-the-art multimodal recommenders. Crucially, ReasonRec substantially reduces inference latency by dynamically delegating up to 35% of queries to efficient sub-models without compromising accuracy. Extensive ablation studies further confirm that each proposed reasoning and planning mechanism individually contributes substantially to ReasonRec's overall effectiveness. Collectively, our results illustrate a clear pathway towards interpretable, adaptive, and efficient multimodal recommendation through explicit reasoning and agentic design. 

---
# SafeGEO: Understanding Generative Engine Optimization Risks in Recommendation Agents 

**Authors**: Qianfeng Wen, Yifan Simon Liu, Xin Liu, Difan Jiao, Blair Yang, Junda Wu, Zhenwei Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28356)  

**Abstract**: Generative Engine Optimization (GEO) lets content owners rewrite web content to increase their visibility in generative systems. In recommendation agents, this creates a risk that seller-controlled sources make flawed products appear better supported than they are. We study this risk by asking whether recommendation agents preserve utility-aligned decisions when seller-controlled sources are rewritten for GEO. To make this question measurable, we construct SafeGEO, an evaluation suite with 22 GEO attack variants across 600 recommendation cases. We empirically show that GEO attacks can promote flawed target products. On average, they increase the rate at which such flawed products enter the recommendation set by up to 83.2%. We further study whether agent-side design choices can mitigate this risk and show that simple defenses, including defensive prompting and structured evidence checks, reduce harmful target promotion by up to 39.2%. These gains are substantial but do not restore the no-GEO performance, showing that GEO remains a serious risk despite developer-side mitigation. 

---
# From Regulatory Approvals to Patents: Cross-Domain Linking for Cardiovascular Device Traceability 

**Authors**: Yang Qingqing, Liu Haijiang, Li Moyan  

**Link**: [PDF](https://arxiv.org/pdf/2606.28353)  

**Abstract**: Linking FDA-approved medical devices to their underlying United States Patent and Trademark Office (USPTO) patents enables critical applications such as recall root-cause analysis, M&A-driven IP discovery, and technology trajectory mapping. However, this cross-domain entity linking task remains unexplored due to severe *semantic gaps*: FDA documents focus on clinical outcomes, while patents describe technical mechanisms, yielding minimal lexical overlap. We formalize medical device-patent linking as a challenging cross-domain entity linking problem characterized by label scarcity and domain shifts. Using cardiovascular devices as a high-impact, representative domain featuring diverse technologies, high recall rates, and abundant disclosures, we construct a benchmark with 434 devices, 698K patents, and 585 high-fidelity expert-verified pairs. To address these challenges, we propose Bridge-MedDevKG, a coarse-to-fine framework that integrates (1) **MedDevOnto**, a domain-specific ontology that anchors device concepts via three-tier UMLS normalization; (2) **Multi-signal candidate generation** fusing company affiliation, semantic similarity, and ontology-weighted entity overlap; and (3) **Heterogeneous reranking** with multi-signal scoring and XGBoost classification on hard negatives. Our approach achieves a conservative lower-bound recall of 91.6% on the gold standard with 50.9% noise reduction, substantially outperforming LLM baselines under comparable evaluation. The resulting MedDevKG provides 6.8M high-confidence links, laying a scalable foundation for regulatory-IP integration across medical specialties. 

---
# HMARS: A Hierarchical Multi-Agent Memory System for Long-Context Reasoning 

**Authors**: Zeju Li, Ziyang Zheng, Yizhou Zhou, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28349)  

**Abstract**: Long-context reasoning requires models to access, retrieve, and integrate evidence scattered across documents, dialogues, and accumulated interaction histories. Standard retrieval-augmented generation reduces this problem to top-$K$ chunk retrieval, but such passive access can discard relevant evidence before reasoning begins, especially when relevance depends on broader context. We propose HMARS, a hierarchical multi-agent memory system that treats long contexts as managed memory rather than a flat retrieval corpus. Sub-agents maintain grounded access to bounded memory regions, mid-agents manage regional context and provide query-specific coordination, and a frontier model performs final reasoning over retrieved evidence pages. To evaluate this view, we construct two diagnostic benchmarks targeting evidence breadth and context-dependent relevance. Across long-document and multi-turn memory tasks, HMARS achieves the best overall performance against retrieval, reranking, full-context, graph-based, and agentic long-context baselines. Evidence coverage analysis further shows that its gains come from retrieving the required supporting evidence more completely, rather than merely changing the final answer prompt. 

---
# Agentic Safety is an Epistemic Property, Not a Behavioral One 

**Authors**: Charles L. Wang, Keir Dorchen, Peter Jin  

**Link**: [PDF](https://arxiv.org/pdf/2606.28347)  

**Abstract**: Contemporary AI safety spans pre-training interventions, post-training alignment, deployment-time controls, monitoring, and red-teaming. These methods are necessary, but they primarily certify snapshots of system behavior. As AI systems become more capable, dynamic, embodied, and self-improving, this snapshot view becomes incomplete: safety depends not only on whether a system behaves acceptably now, but whether it remains correctable as it learns, adapts, acts, and modifies itself over time. This paper argues that safety should therefore be treated as an epistemic property of the evolving learner, not merely a behavioral property of the current policy. We introduce teachability as the capacity to preserve future corrective leverage under bounded human, institutional, or environmental intervention. We argue that advanced systems can retain visible competence while eroding the representational, algorithmic, or meta-decision conditions needed for future correction. Safe advanced AI systems must not only behave acceptably now; they must remain teachable later. 

---
# Auditing LLM-Governed Social Robots with Culture-Specific Moral Gradients 

**Authors**: Carmen Ng, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2606.28345)  

**Abstract**: LLM-governed social robots increasingly decide who receives real-world assistance first. As prioritization norms vary across cultures by age, status, and group size, failure to calibrate pluralistically can scale into unequal access. Yet LLM moral audits remain English-centered, rarely test embodied contexts, leaving pluralistic calibration as an urgent diagnostic gap amid intensifying LLM-robot deployment. We introduce a gradient-based audit framework for multilingual evaluation of LLM moral trade-off behavior against cultural preference gradients. Grounded in nine cross-domain social robotics reviews (>8,000 papers), we derive symmetry-controlled scenarios across care, education, and services, translating the Moral Machine Experiment's "whom to spare" into "whom to assist first" dilemmas with preserved identity trade-offs (many vs. few; young vs. old; higher vs. lower status). We audit four LLMs across four country-language pairs in four prompting regimes (57,600 decisions), benchmarked against country-specific MME preference gradients. Ordinal concordance tests whether models differentiate cultural contexts; a governance typology maps vulnerabilities in gradient differentiation, directional tendency, and deliberation. We find persistent, culturally asymmetric gradient tracking failures that prompting alone cannot reliably correct: quality calibration is nearly twice as strong for Western-language decisions as for Chinese and Japanese; high determinism in majority-first trade-offs often erases cross-cultural gradients; partial sensitivity to age- and status-based norms risks sidelining minorities. Prompting effects are uneven; only contrastive exemplars yield consistent gains, while reasoning-only prompts can worsen tracking. Our results motivate multilingual, pluralistic audits as an LLM-robot pre-deployment gate and suggest model factors are a more robust lever than prompting alone. 

---
# PIXELRAG: Web Screenshots Beat Text for Retrieval-Augmented Generation 

**Authors**: Yichuan Wang, Zhifei Li, Zirui Wang, Paul Teiletche, Lesheng Jin, Matei Zaharia, Joseph E. Gonzalez, Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2606.28344)  

**Abstract**: Augmenting large language models (LLMs) with retrieved web text has become a dominant paradigm, yet the web is not natively textual: existing systems depend on complex parsing pipelines that linearize HTML and discard layout, visual structure, and formatting. We introduce PixelRAG, a new retrieval-augmented method that represents websites in their native visual form and performs retrieval and reading entirely in pixel space, enabling an end-to-end architecture that eliminates text abstraction. PixelRAG is, to our knowledge, the first pipeline to operate over a full Wikipedia corpus in this form, scaling to a datastore of 30 million screenshot images with an efficient visual retrieval index. Built on an existing visual embedding model (i.e., Qwen3-VL-Embedding), PixelRAG further fine-tunes this model on screenshot data with carefully curated contrastive training data. Retrieved screenshots are then fed directly as pixel inputs to a VLM, without intermediate text conversion. PixelRAG consistently outperforms both no-retrieval and text-based RAG baselines, most surprisingly on widely studied text-centric tasks such as NQ and SimpleQA. It also achieves strong gains on multimodal open-domain QA (e.g., MMSearch), benchmarks over noisy news corpora (e.g., LiveVQA), and agentic benchmarks (e.g., MoNaCo), improving accuracy by up to 18.1% over text-based baselines. Finally, pixel representations enable a new efficiency lever for RAG through image compression, achieving up to 3x token cost reduction at lower resolutions while maintaining accuracy. Our results challenge the necessity of text representations in web retrieval, suggesting that web RAG can operate directly in the web's native visual form while improving both performance and efficiency. 

---
# The Crowded Embedding Space: A Mean-Field Mechanism for Emergent Marginalization in Retrieval-Augmented Agents 

**Authors**: Shwan Ashrafi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2606.28343)  

**Abstract**: Retrieval-augmented generative agents rely on retrieval for grounding, yet are typically evaluated on a query-by-query basis. This isolates interactions that are geometrically coupled in a shared embedding space. For example, we show that the high document density required to serve majority interests (e.g., generic "Crime" movies) can geometrically overcrowd the retrieval neighborhood of a semantically similar minority (e.g., "Film Noir"), effectively expelling minority content from top-$k$ results. We introduce a formal framework to analyze how such goal collisions in dense retrieval induce fundamental performance limits and emergent fairness issues inherent to spatial crowding. In our static analysis, we demonstrate that for a fixed embedding space, a phase transition occurs where minority user goals suffer a catastrophic collapse in performance as the density of majority goals increases. We then extend this to a dynamic model and derive a non-linear Fokker-Planck equation that governs the evolution of document embeddings as the agent updates them to maximize retrieval accuracy. Our analysis reveals that this local relevance objective triggers an emergent global mechanism that systematically marginalizes minority interests. We prove that such objectives drive the system to self-organize into a state that exclusively serves majority interests. These results provide a theoretical foundation for understanding a critical grounding failure mode in retrieval-augmented agents. 

---
# Operating Regimes of Decentralized Learning Under Mobility and Bandwidth Constraints 

**Authors**: Samuele Sabella, Chiara Boldrini, Lorenzo Valerio, Marco Conti, Andrea Passarella  

**Link**: [PDF](https://arxiv.org/pdf/2606.28342)  

**Abstract**: Decentralized learning is a promising paradigm for collaborative training in mobile and pervasive systems, as it avoids a central coordinator and does not require sharing raw data. Yet, most analyses rely on idealized communication assumptions that break down in wireless settings, where connectivity is intermittent, topology changes due to mobility, and bandwidth is limited. We study decentralized averaging under client asynchrony, time-varying contact graphs, and technology-dependent throughput constraints. We implement a fully decentralized protocol that overlaps synchronization with local training and supports partial tensor-level transfers when contacts end early. Using Random Waypoint mobility and multiple wireless technologies (Bluetooth LE, LTE, and Wi-Fi), we quantify how network dynamics and link capacity impact convergence. We identify three operating regimes: (i) inter-contact time largely dictates convergence via mixing, (ii) partial updates are often well tolerated when contacts are frequent, and (iii) very dense contact patterns can trigger contention, reducing effective throughput. These findings provide a practical lens to reason about decentralized learning deployments over realistic wireless systems, highlighting when improving connectivity, increasing bandwidth, or mitigating contention is most impactful. 

---
# Multi-Agent DRL for QoS and Energy Optimization in RIS-Enabled Open-RAN Industrial 6G TN/NTN Networks 

**Authors**: Marwan Dhuheir, Thang X. Vu, Symeon Chatzinotas  

**Link**: [PDF](https://arxiv.org/pdf/2606.28339)  

**Abstract**: Industrial 6G networks require ultra-reliable, low-latency, and energy-efficient connectivity in dynamic and blockage-prone environments, where conventional terrestrial deployments often fail to ensure stable coverage. Hence, in this paper, we propose a RIS-enabled Open-RAN framework for integrated terrestrial/non-terrestrial (TN/NTN) industrial 6G networks, in which UAVs-mounted reconfigurable intelligent surfaces (RISs) cooperate with ground radio units and a high-altitude platform (HAP) to enhance connectivity for dense industrial IoT devices. Owing to the high dimensionality and strong coupling among decision variables, conventional optimization techniques become computationally intractable. To overcome this limitation, the joint optimization problem of data rates, latency, and energy consumptions is formulated as a decentralized partially observable Markov decision process (Dec-POMDP) and solved using a multi-agent deep reinforcement learning framework. Simulation results show improvements of up to 75\% in data rate, 25\% latency reduction, and 16\% energy savings compared with state-of-the-art learning-based and non-RIS baselines, demonstrating the effectiveness of RIS-assisted Open-RAN intelligence for industrial 6G networks. 

---
# A Systems-Level Analysis of Sensitivity, Robustness, and Stability in Retrieval-Augmented Generation 

**Authors**: Bharath Simha Reddy Muthyam  

**Link**: [PDF](https://arxiv.org/pdf/2606.28337)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are often evaluated using final answer accuracy, even though their failures can originate from preprocessing, retrieval, context packing, or generation. This paper presents a controlled empirical study of RAG sensitivity, robustness, and stability across 56 experimental runs. We evaluate how chunk size, retrieval depth (top k), embedding-based reranking, probabilistic retrieval noise, and repeated seeded runs affect retrieval, context packing, and generation behavior. Using a fixed 500-question QA subset mapped to 20,958 unique corpus contexts, we analyze both final answer metrics and intermediate failure modes. Across these experiments, retrieval-oriented metrics improved under broader retrieval settings, while downstream exact-match and F1 scores often behaved non-monotonically. We also observe preprocessing-induced answer loss under smaller chunk sizes, progressive degradation under retrieval corruption, and higher observed variance in broader retrieval regimes. These findings suggest that RAG evaluation should include sensitivity, robustness, stability, and multi-stage failure analysis rather than relying only on final answer accuracy. 

---
# HyBIRD: Hyperbolic Bridge Retrieval and Diagnosis for Methodology Inspiration Retrieval 

**Authors**: Yang Yang, Boyun Xu, Hao Fu, Jindong Li, Zining Zhong, Bowen Tian, Jiemin Wu, Menglin Yang, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2606.28336)  

**Abstract**: Methodology Inspiration Retrieval (MIR) asks a system to retrieve prior papers whose methods can inspire a new research proposal. Unlike general scientific retrieval, the central challenge is not topical similarity but whether a candidate paper provides concrete mechanisms that can instantiate an abstract methodological need. Existing MIR dense retrievers provide strong paper-level rankings, but the returned lists do not expose how proposal needs are bridged by retrieved methods, where evidence is weak, or which complementary snippets may help. We propose HyBIRD, a frozen-anchor framework that treats MIR as hyperbolic bridge retrieval and post-hoc method diagnosis. HyBIRD keeps a strong MIR dense retriever fixed, learns lightweight point, cone, and factorized hyperbolic bridge variants, and uses LLM-assisted method blocks for post-hoc explanation and evidence selection. On the MIR benchmark, the factorized bridge reaches 59.034 mAP while preserving the dense anchor's strong retrieval behavior. More importantly, HyBIRD converts ranked papers into inspectable query need profiles, factor coverage, maturity views, and complementary evidence bundles. The results suggest that hyperbolic geometry is most useful as calibrated structure over a dense anchor, rather than as a standalone replacement for dense retrieval. 

---
# LLM-Ideoplasticity: Measuring Ideological Plasticity in the Political Behavior of LLMs as a Context-Conditioned Distribution 

**Authors**: Adib Sakhawat, Syed Rifat Raiyan, Tahsin Islam, Takia Farhin, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2606.28335)  

**Abstract**: We argue, with systematic empirical evidence, that a large language model's political ideology is not a fixed point, but a conditional distribution $\mathbb{P}($position$\mid$context$)$ over a real political space. We evaluate nine current LLMs using a unified measurement framework anchored by VAA-CHES projection models, which map responses onto three validated dimensions (lrgen, lrecon, galtan) across six contextual axes. Our findings reveal high sensitivity to context: persuasive framing and under-represented languages displace coordinates by up to 0.57 and 0.52 units, respectively, while chain-of-thought reasoning often amplifies rather than dampens paraphrase instability. Despite this local plasticity, the model cohort occupies a remarkably narrow Overton envelope overall, occupying roughly one-third the spread of major European parties. Supported by a multi-trait multi-method (MTMM) analysis, we conclude that a single point cannot summarize LLM political behavior; it must be characterized as a shape. Our code and data are publicly available at this https URL. 

---
# Ground Truths in Suicide Research: The Current State of AI-Based Suicide Detection in Social Media 

**Authors**: Yaakov Ophir, Ofri Hefetz, Refael Tikochinski, Kfir Bar, Shir Lissak, Shulamit Grinapol, Haya Wachtel, Eyal Fruchter, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2606.28334)  

**Abstract**: Recent advances in artificial intelligence (AI) and social media data have led to growing optimism about the ability to detect suicide risk at scale. However, the empirical foundations of this work remain unclear. This article provides a synthesis of current research on AI-based suicide detection in social media, drawing on a recent umbrella review of 22 systematic reviews covering studies up to 2022, alongside an ongoing literature review extending the analysis to more recent work. Across these sources, we identified 195 relevant studies, which are documented in a detailed supplementary dataset outlining their key characteristics and findings (see Supplementary Information). Analysis of these studies reveals consistent patterns, including rapid growth, concentration on a small number of platforms, reliance on textual and English-language data, and repeated use of similar datasets. Most importantly, the majority of studies rely on indirect labeling strategies that do not involve direct, individual-level validation of suicide risk. Instead, ground truth is typically inferred from observable features of online content, such as linguistic markers or community membership. As a result, the predictive task often shifts from identifying individuals at risk to classifying posts that contain suicidal or distress-related language, limiting the ability of current approaches to detect individuals who do not express such content explicitly online. These findings suggest that current advances in model performance should be interpreted with caution. Progress in this field is likely to depend less on improving model performance and more on ensuring that model predictions meaningfully correspond to suicide risk as it is experienced in real life. 

---
# Insidious by Design: Implications of Large Language Model algorithmic bias for the Global South 

**Authors**: Sioux McKenna, Nompilo Tshuma  

**Link**: [PDF](https://arxiv.org/pdf/2606.28333)  

**Abstract**: \begin{quote} The biases in Large Language Models' (LLMs) outputs remain inadequately theorised, particularly from the perspective of the Global South. This article reports on a small-scale exploratory study in which identical prompts were submitted to four major LLMs (ChatGPT, Claude, Grok, and Copilot), firstly, prompting for stories using names suggestive of specific racial and gender communities, and secondly asking questions about `development'. Drawing on critical AI scholarship and postcolonial theory, we argue that LLM outputs are patterned in ways that reproduce racial hierarchies, gender asymmetries, and Western-centric epistemic frameworks. We argue that these biases are insidious: they operate below the threshold of both obvious error and overt prejudice, and instead are subtly embedded in narrative structure and emotional template. Simply put, women, in LLM narratives have rich interior lives, while men make plans. Black people face hardships while white people navigate the world with agency. And explanations as to the economic world order fail to consider Southern explanations. The models perform plausibility while reproducing dominance. We conclude that universities require structural critique of these technologies rather than unreflective adoption, and that critical AI literacy must engage seriously with questions of whose knowledge systems are reproduced and legitimated, or marginalised and undermined. 

---
# When Medical Safety Alignment Fails: A Benchmark for Evaluating LLMs on High-Risk Medical Queries 

**Authors**: Yige Li, Jun Sun, Wei Zhao, Zhe Li, Yutao Wu, Hanxun Huang, Xiang Zheng, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.28332)  

**Abstract**: Large language models (LLMs) are increasingly used for medical and health-related questions, yet their safety in high-risk medical scenarios remains poorly understood. We introduce \textsc{MedHarm}\footnote{Code and data will be released upon acceptance. Due to the sensitive nature of high-risk medical queries, data access will be available to qualified researchers upon request.}, a high-risk medical safety benchmark with 1,100 medically grounded queries across 10 safety-critical categories, including toxicology, pharmacology, covert poisoning, anesthesia, and fetal harm. Unlike broad medical QA benchmarks, \textsc{MedHarm} targets realistic clinical, educational, and technical prompts that require refusal, caution, or safe redirection rather than direct helpfulness. We evaluate 15 LLMs spanning general-purpose, medical-purpose, closed-source, and downstream SFT models, together with 4 representative guardrail models. Results reveal a substantial gap between apparent alignment and medical safety: aligned models can still produce unsafe or actionable responses, medical fine-tuning can amplify harmful specificity, and external guardrails reduce some failures while introducing brittle blocking and weak safe helpfulness. These findings show that medical safety cannot be inferred from general alignment or medical capability alone, highlighting the need for domain-specific stress testing before deploying LLMs in safety-critical medical applications. 

---
# "AI Watermarking": Bridging Policy Discourse and Technical Capabilities 

**Authors**: Andrés Fábrega, Arkaprabha Bhattacharya, Miranda Christ, Sunoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.28331)  

**Abstract**: The widespread deployment of generative artificial intelligence (AI) models has raised serious concerns about the proliferation of AI-generated content. This has led to a surge of interest in, and demand for, reliable tracking and detection mechanisms for content that is AI-generated, such as watermarking, metadata tagging, content tagging, and more. The problem has captured the attention of policymakers as well as the popular media, and a spate of recent bills in the US have sought to regulate the spread of AI content, and enforce or promote methods to track and label it. This work performs a critical analysis of the policy discourse surrounding generative AI content transparency in the US and EU. Through a broad document selection methodology, we first collect a broad corpus of documents containing legislative language and policy-relevant discourse on the topic. We then analyze these through inductive coding, and leverage our coding to systematize these documents, identifying key patterns, gaps, and open questions. We identify critical points of disconnect between policy and technological capabilities and practice, and we highlight and discuss potential ambiguities and pitfalls raised by the trends in our corpus. 

---
# High-Dimensional Concentration and Retrieval Instability in Embedding Spaces: Implications for Retrieval-Augmented Generation 

**Authors**: Ernesto Lopez Fune  

**Link**: [PDF](https://arxiv.org/pdf/2606.28330)  

**Abstract**: Embedding-based retrieval systems rely on the assumption that geometric proximity in highdimensional representation spaces reflects semantic relevance. However, high-dimensional geometry induces concentration phenomena that can reduce the discriminative power of similarity measures and can destabilize nearest-neighbor retrieval. This work studies distance concentration, cosine concentration, contrast collapse, hubness, and retrieval instability through controlled numerical experiments across multiple synthetic distributions. The results show that similarity signals progressively lose contrast as dimension increases, leading to unstable retrieval behavior and structural bias in nearest-neighbor selection. A simplified Retrieval-Augmented Generation experiment further suggests that these effects can degrade grounding reliability upstream of generation. These findings motivate geometry-aware diagnostics and robustness-oriented retrieval strategies for embedding-based AI systems. The experiments are intentionally synthetic in order to isolate intrinsic geometric effects. High-dimensional embedding space Distance and cosine concentration Score-gap collapse and hubness Retrieval instability under perturbations Weak or incomplete retrieved context Potential degradation of grounding 1. 

---
# $M^3 QuestionIng$: Multi-modal Multi-span Medical Question Answering 

**Authors**: Anisha Saha, Vaibhav Rathore, Abhisek Tiwari, Akash Ghosh, Sai Ruthvik Edara, Sriparna Saha  

**Link**: [PDF](https://arxiv.org/pdf/2606.28329)  

**Abstract**: The growing adoption of AI in healthcare, particularly in preventive care, highlights the critical need for accessibility and precision in Medical Question Answering (MedQA). In recent years, significant efforts have been made to develop multi-span medical question-answering systems, where the answer to a query may span multiple sections or paragraphs of a source document. However, existing systems fall short of aligning with real-world scenarios, where source documents often include both textual and visual content, requiring answers to incorporate images for better comprehension. To address this gap, we propose $M^3QAFrame$, a multi-modal, multi-span medical question-answering framework that leverages visual cues to enhance the generation of comprehensive answers drawn from diverse textual and visual spans. The model takes the context, query, and images as input and outputs an answer containing both textual answers and relevant images. The text and image embeddings are processed using a transformer-based architecture to determine the sentence and image relevance. We curate a multi-modal, multi-span medical question-answering ($M^3 QuestionIng$) dataset containing queries, medical contexts, associated medical images, and extractive answers. Additionally, each query-answer pair is labeled with user intent and query type to enhance query and context comprehension. Extensive experiments show that our approach consistently outperforms existing methods across various evaluation metrics. 

---
# The Interference Gap: Comparing Retrieval Bounds in Human Memory and RAG Systems 

**Authors**: Dongxin Guo, Jikun Wu, Siu-Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28327)  

**Abstract**: How do retrieval bounds compare between human episodic memory and Retrieval-Augmented Generation (RAG) systems under semantic interference? We present a unified signal detection theory (SDT) framework that applies to both, and use it to fit behavioral and computational data in matched paradigms. Both systems show logarithmic accuracy decline with association count (fan), but humans exhibit lower interference sensitivity ($\alpha/\sigma = 0.41$) than dense passage retrieval ($\alpha/\sigma = 0.67$), with cognitively-inspired HippoRAG falling between the two ($\alpha/\sigma = 0.44$). Behavioral experiments ($N = 112$) and simulations validate the framework; parameter recovery confirms identifiability ($r \geq .93$) and model comparison favors the logarithmic specification over a power-law alternative ($\Delta$BIC $> 15$). We discuss encoding specificity, temporal context binding, and retrieval gating as candidate mechanisms whose causal role remains to be established. Six falsifiable predictions connect cognitive memory research with AI retrieval evaluation. 

---
# ADEPT: An Entropy-Driven Dual-Strategy Agent for Interactive Video Retrieval 

**Authors**: Ke Chen, Shengyuan Han, Yongfeng Huang, Yujin Zhu, Jingwei Xiong, Liang Xu, Jundong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28326)  

**Abstract**: This research aims to solve the challenge of video retrieval from massive datasets, caused by ambiguous user queries. Prevailing single-round retrieval paradigms face a performance bottleneck, as they lack effective feedback mechanisms to handle complex search intentions. The root cause is the "Intent-Query Gap", where users' intent cannot be captured by a simple text query. To solve this, we propose the ADEPT framework: a training-free agent that pioneers an entropy-driven decision engine to efficiently guide dialogue by dynamically selecting between ASK and REFINE strategies. Experiments on two challenging datasets demonstrate that ADEPT significantly outperforms all non-interactive, heuristic, and Video-LLM baselines. The core contribution of this work is an efficient and interpretable entropy-driven interactive strategy that sets a new performance benchmark for the field of interactive video retrieval. 

---
# It Lied to a Doctor to Buy Poison Ingredients: Quantifying Real-World Misuse of Phone-use Agents 

**Authors**: Yiming Sun, Chen Chen, Zifan Zhou, Mi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27944)  

**Abstract**: Phone-use Agents can execute complex tasks end to end across real mobile applications. By operating a real device on the user's behalf, they reach far more functionalities than CLI agents, which amplifies the real-world harm they can cause when driven for malicious purposes. We present the first study of this threat on real phones and 27 commercial apps, and find that agents built on 9 mainstream commercial and open-source models readily carry out serious misuse, ranging from procuring drug and explosive precursors to fraud, online harassment, and review manipulation. Across the agents we run on real devices, the average refusal rate to harmful requests stays low while the average task-completion rate reaches 68.8%, and in some scenarios an agent finishes a violation faster than a human would. These results suggest that Phone-use Agents already meet the practical conditions for automated misuse at scale.
In one observed real-device execution, Claude-Opus-4.8 fabricated a medical history, deceived an online doctor into issuing a prescription, and completed the order and payment on its own to purchase a precursor for a highly toxic substance. To our knowledge, this is the first documented real-world case of an AI agent procuring controlled precursor materials. We trace this behavior to a Safety Awareness-Execution Gap, where an agent recognizes that a request is harmful yet still executes it. Simple defenses curb the overt cases, but the more covert and arguably more damaging threats, such as coordinated review manipulation and fake traffic, remain largely unsolved. We hope these findings push the community toward safer Phone-use Agents. 

---
# Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task 

**Authors**: Sunqi Fan, Jiashuo Cui, Meng-Hao Guo, Shuojin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.10359)  

**Abstract**: Video Question Answering (VideoQA) task serves as a critical playground for evaluating whether foundation models can effectively perceive, understand, and reason about dynamic real-world scenarios. However, existing Multimodal Large Language Models (MLLMs) struggle with simultaneously modeling spatial relationships within video frames and understanding the causal dynamics of temporal evolution on complex and reasoning-intensive VideoQA task. In this work, we equip MLLM with a comprehensive and extensible Video Toolkit, to enhance MLLM's spatiotemporal reasoning capabilities and ensure the harmony between the quantity and diversity of tools. To better control the tool invocation sequence and avoid toolchain shortcut issues, we propose a Spatiotemporal Reasoning Framework (STAR) that strategically schedules temporal and spatial tools, thereby progressively localizing the key area in the video. Our STAR framework enhances GPT-4o using lightweight tools, achieving an 8.2% gain on VideoMME and 4.6% on LongVideoBench. We believe that our proposed Video Toolkit and STAR framework make an important step towards building autonomous and intelligent video analysis assistants. The code is publicly available at this https URL. 

---
