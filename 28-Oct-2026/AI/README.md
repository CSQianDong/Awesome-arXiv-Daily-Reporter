# KPPax: preprocess differentiable kinetic ODE for machine learning in atmospheric chemistry 

**Authors**: Wenqing Peng, Michael Boy, Zhi-Song Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37655)  

**Abstract**: Atmospheric chemical mechanisms are commonly distributed in the language of the Kinetic PreProcessor (KPP), which generates efficient Fortran or C code but does not fit into the differentiable programming frameworks in which machine-learning models are developed. We present KPPax, a kinetic preprocessor that generates differentiable, just-in-time compiled, and vectorisable chemistry code in JAX from mechanisms written in the KPP language. KPPax evaluates the rate law with sparse operations in log space, which keeps it differentiable, and integrates the stiff system with the implicit solvers and the forward- and reverse-mode adjoints of the \texttt{diffrax} library, so that exact sensitivities are obtained without hand-derived adjoint code. We demonstrate two applications with the POLLU benchmark mechanism. In the first, the Fisher information matrix computed from an automatically differentiated Jacobian predicts, before any fit, which of the 25 rate coefficients a flow-tube-like design can identify, and noisy synthetic estimation confirms the predicted uncertainties of the nine identifiable rate coefficients to within about 30\% while revealing clusters of fast reactions of which only ratios are constrained. In the second, a neural-network correction of an emission inventory is trained offline on per-scenario 4D-Var retrievals and then fine-tuned online through the chemical model, which reduces the error of the biogenic correction by a factor of 2.4--5 and recovers information on low-temperature emissions that the individual retrievals lose. KPPax makes KPP mechanisms directly usable for parameter estimation and hybrid physics--machine-learning modelling. 

---
# What Improves During Pretraining? Tracking Knowledge Acquisition with Point-in-Time Models 

**Authors**: Chang Su, Bin Wu, Emine Yilmaz, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37631)  

**Abstract**: Language models pretrained on web corpora collected over multiple years learn both general and domain-specific knowledge. How these different kinds of knowledge are learned as pretraining progresses remains unclear. Recent point-in-time models trained on web snapshots from different years provide a way to probe how knowledge prevalence varies over time. Using these models, we propose a temporal variability score (TVS), defined as the coefficient of variation of an item's negative log-likelihood across models. A higher score suggests that the knowledge may be less consistently represented across years of web text. We evaluate six benchmarks covering diverse topics using six models from multiple open-source model families. We find that low- and high-TVS items are learned similarly early in training, but high-TVS items improve faster later on. At the end of the pretraining run, high-TVS items outperform low-TVS items in 34 of 36 model-benchmark settings, with an average advantage of 7.3 points of chance-normalized accuracy. This pattern is most robust on knowledge-intensive tasks, even when comparing items of similar difficulty. Our findings reveal differences in how knowledge is learned over the course of pretraining, with temporal variability providing a diagnostic of these learning dynamics beyond aggregate benchmark scores. 

---
# Rare, Relevant, and Still Wrong: How Memory Fails LLM Group Decisions 

**Authors**: Anouar Imel, Imad Charradi, Abderrahman Sghir, Khalil El Khammar  

**Link**: [PDF](https://arxiv.org/pdf/2610.37624)  

**Abstract**: When an LLM group's discussion is compressed into a short memory for a final decision maker, what should the memory keep? Two natural heuristics keep the rarest claims or the claims relevant to the decision criteria. On hidden-profile tasks both look attractive, because how widely a fact is held is confounded with whether it helps. We introduce CrossProfile, a diagnostic benchmark of 60 templates with automatically verified answer keys, in which the ownership of supporting, decoy, and irrelevant facts is crossed while their text is held fixed (480 tasks). Across Qwen2.5-7B, Llama-3.1-8B, and two DeepSeek models, both heuristics fail in predictable places. Selecting by relevance readmits decoys, because decoys are on topic: even a perfect relevance filter loses 23 to 31 points to rarest-first selection in the classic hidden-profile configuration. Rarest-first selection is no better than random selection on average and wins only where the rare facts happen to be the useful ones. In both cases accuracy follows the balance of evidence the memory keeps, not the policy that kept it. The decision model adds a third failure, a form of social proof: it gives extra weight to facts attributed to many agents. Removing the attribution tags, or reassigning them at random with line order fixed, raises accuracy in the classic configuration by 13 to 24 points. In natural discussions the evidence is lost before any memory acts: the complete transcript yields lower accuracy than no notes at all. 

---
# Timestral 

**Authors**: Adrien Petralia, Alexandre Gavaudan, Artjom Joosen, Etienne Goffinet, Richard Kurle, Yassine El Ouahidi, Yassir Bendou, Abdelaziz Bounhar, Abhijeet Somani, Aditi Kabra, Adrian Valente, Adrien Sadé, Ahmed Djermani, Alan Jeffares, Albert Jiang, Alberto Bietti, Aleksandr Timashov, Alexandre Cahill, Alexandre Laval, Alexandre Sablayrolles, Alexandros Sanchez Vassopoulos, Amélie Héliou, Amos You, André Jonasson, Andreas Fürst, Andrew Bai, Andrew Ehrenberg, Andrew Zhao, Andy Lo, Angele Lenglemetz, Anmol Agarwal, Antonia Calvi, Arata Suzuki, Arjun Majumdar, Arthur Fournier, Ashwani Kumar, Augustin Lombard, Avinash Sooriyarachchi, Aylin Guliz Akkus, Aysenur Karaduman, Baptiste Bout, Baptiste Rozière, Barna Pásztor, Baudouin De Monicault, Benjamin Holzschuh, Benjamin Tibi, Benno Stein, Bernhard Stadlbauer, Bhargavi Bhaskar, Binbing Hou, Błażej Osiński, Boris van Breugel, Camille Le Scao, Chaoran Yu, Charlotte Cronjäger, Chen-Yo Sun, Chris Bamford, Christian Wallenwein, Christophe Renaudin, Clémence Lanfranchi, Colin Bonatti, Corentin Barreau, Corentin Sautier, Cristiana-Diana Diaconu, Cyprien Courtot, Daniel Lopes, Daniel Marczak, Darius Dabert, Diego de las Casas, Dominik Nuss, Dylan Rubini, Dzmitry Soupel, Elizaveta Demyanenko, Elliot Chane-Sane, Emilien Fugier, Emmanuel Gottlob, Erik Aas, Etienne Duchesne, Fabian Paischer, Fabian Schlager, Faruk Ahmed, Federico Baldassarre, Félix Rosseeuw, Filip Szatkowski, Filippo Vicentini, Florian Wiesner, Gabrielle Berrada, Gaëtan Ecrepont, Gaspard Blanchet, Gaspard Donada-Vidal, Gaurav Mahajan, Gauthier Guinet, Genevieve Hayes, Georgii Novikov, Georgios Christoglou, Giada Pistilli, Gonzague de Carpentier, Graeme Nail, Guillaume Breton, Guillaume Kunsch  

**Link**: [PDF](https://arxiv.org/pdf/2610.37585)  

**Abstract**: We introduce Timestral, a family of four time series foundation models for probabilistic forecasting, with 5M, 28M, 123M, and 209M parameters. Timestral sets a new state of the art on GIFT-Eval, TIME, and fev-bench, where Timestral-209M and Timestral-123M take the top two places on every metric. We achieve this with three ingredients: Decoupled Horizon Forecasting (DHF), which separates prediction tokens from the causally encoded context, so that longer horizons are trained without masking context and each observed value is both a forecast target and context for later forecasts; a covariate framework that natively supports known-future covariates within a causal architecture; and a two-stage pipeline that pretrains exclusively on a diverse, large-scale synthetic corpus before post-training on a mix of real and synthetic data with extended context. This report describes the model design and its benchmark results. Weights for all four models are available on Hugging Face and inference code on GitHub. 

---
# TeamGym: A Gym for Team Decision-Making by AI Agents with Distributed Information 

**Authors**: Chanwoo Park, Jaywoong Jeong, Zachary Wojtowicz, Yubin Kim, Taekyun Lee, Kaiqing Zhang, Asuman Ozdaglar  

**Link**: [PDF](https://arxiv.org/pdf/2610.37571)  

**Abstract**: Many real-world decisions require teams to integrate information distributed across participants. Yet AI agents are still primarily trained and benchmarked in isolation. We introduce TeamGym, an environment for evaluating and training language model-based agents on team decision-making problems constructed from real-world legal, finance, medical, and software engineering corpora. TeamGym introduces a method to identify a minimal set of facts, each element of which is necessary and all of which are collectively sufficient for a correct final decision, which we call pivotal information. This enables TeamGym to both control the distribution of pivotal information and track how well agents identify and integrate pivotal information when making team decisions. Examining 8 language models on 12 source corpora, we find that current AI agents often fail to effectively share information: Distributing pivotal information across participants lowers success rate by 22.9 percentage points on average relative to giving the same information to a single model (a communication gap). We also document that participants do not effectively retrieve decision-relevant information on their own: A team's success rate is lowered by a further 19.9 points when participants must identify and extract pivotal information from source material rather than receiving it directly (a retrieval gap). To help remedy these deficits, we use TeamGym to train policies through reinforcement learning on team decision loss. Across two policy-training experiments, we find that training improves final-decision quality and induces emergent coordination in participation, speaker selection, and communication, even though the reward evaluates only the final decision. 

---
# From Interaction to Capability: Task Advancement as State and Supervision for GUI Agents 

**Authors**: Ziyang Chen, Qian Li, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.37482)  

**Abstract**: GUI memory should preserve the results of past actions and learn reusable operations. This is difficult in long-horizon workflows, where agents must use information across applications and useful operations are interleaved with failed attempts. We introduce Task Advancement Memory (TAM), which defines task advancement as verified task progress, and supports ongoing task execution and learns reusable operations. During execution, Task Advancement State represents how verified task progress changes the remaining goals, allowing the agent to build on earlier achievements across applications. After each task, advancement supervision identifies and abstracts the interactions that produced a verified local result. The learned operations preserve the relation between required inputs, expected outcomes, and verification conditions. This retains locally successful operations from both successful and failed tasks. On MobileWorld GUI-only tasks, Advancement State alone raises success from 12.0% to 28.2%. Full TAM achieves a 17.6% relative gain in task success over the strongest evaluated baseline and reduces lifecycle tokens per solved task by 38.7%. On SPA-Bench, TAM achieves state-of-the-art task success in the single-app English Level 3 setting. 

---
# Concordat: Typed Sessions for LLM Agents 

**Authors**: Milad Sikaroudi  

**Link**: [PDF](https://arxiv.org/pdf/2610.37461)  

**Abstract**: Agents built on large language models are increasingly allowed to take consequential actions: cancelling bookings, issuing refunds, changing accounts. The rules that govern those actions, such as "obtain the customer's explicit confirmation before changing anything", are usually stated in the prompt and followed only as reliably as the model chooses to follow them. We present Concordat, a Python library that treats an agent as an object whose interface is a finite-state protocol, in the spirit of typestate and session types, and enforces that protocol on the model at run time. On every turn the model is offered only the tools that are legal in the current protocol state, calls outside that set are rejected before they have side effects, decisions reserved for humans can only be requested, and token and tool-call budgets are charged and split linearly across handoffs. Protocols are checked when they are defined, and a subclass of an agent is rejected if it could bypass its parent's protocol, a behavioural form of Liskov substitution. We evaluate a Concordat agent against the standard agent of tau-bench on its 50 airline tasks, adding simulated customers who pressure the agent to skip confirmation. The standard agent made changes without the required "yes" in 13 of 50 tasks with gpt-4.1 and in 29 of 128 task-runs with Claude Haiku 5.5; the Concordat agent made none, with no significant difference in task success. tau-bench's own reward did not detect most of these violations. Concordat is open source. 

---
# M4Bench: Evaluating Procedural Specification for Clinical EHR Derivation Agents 

**Authors**: Hannes Ill, Rafi Al Attrach, Rajna Fani, Ahram Han, Ryohei Yamamoto, Marlene Louisa Moerig, Dukyong Yoon, Leo Anthony Celi  

**Link**: [PDF](https://arxiv.org/pdf/2610.37459)  

**Abstract**: Clinical research over electronic health records depends on derived variables whose definitions sit outside the schema, in item codes, time windows, missingness rules, and output-grain choices that change cohorts and downstream conclusions. We introduce M4Bench, 28 clinical derivation tasks across 15 task families on MIMIC-IV and eICU, to test whether AI coding agents can reproduce such derivations and how clinician-reviewed agent skills change their results. Its controls decompose the value of skills into specification completeness, delivery format, and schema familiarity. In a Docker-isolated Codex campaign (GPT 5.4 Mini and GPT 5.5), skills raise reward by 0.136 over the no-skill baseline, with 26 of 28 tasks improving. Specification completeness accounts for most of this gain: on an eight-task control subset, an operational spec alone raises reward by 0.162, and skills add only 0.024 on top, indistinguishable from zero. On a ten-task subset, decoy skills score like no skill and raw reference SQL nearly matches targeted skills, arguing against generic context or the skill format as the explanation. Schema obfuscation and restructuring preserve 86% and 70% of the skill gain (GPT 5.4 Mini, 26 MIMIC-IV tasks), a descriptive robustness check rather than a bound on memorization. A descriptive Claude Code replication on all 28 tasks (Opus 4.7 and Sonnet 4.6, 560 runs) is directionally consistent (+0.120, 21 of 28 tasks improving). These gains are partial agreement with the reference, not completed tasks: only 25 of 280 with-skill and 13 of 280 no-skill Codex runs pass every strict check, and skills barely change which rows agents return (key F1 +0.003). Skills are released as a versioned, clinician-reviewable markdown delivery format, with a family-level reference-validity audit. M4Bench releases tasks, skills, evaluation code, and sanitized per-run evidence as an open-source benchmark. 

---
# Beyond the Readout: A Latent Numerical Frame Shared Across Modality and Operation 

**Authors**: Oriol Juan Sabater, Alex Ferrando de las Morenas, Pau Rodríguez, Xavier Suau  

**Link**: [PDF](https://arxiv.org/pdf/2610.37422)  

**Abstract**: When a multimodal model computes 17+35 from text or 91-39 from a rendered image, it produces the same answer, 52. Does it use one internal concept of number for both, or separate machinery for each path? Similarity metrics such as RSA or CKA cannot settle this question. They give the same score to identical and to rotated directions, and two representations can look alike only because both prepare the same output digits. We therefore use causal interventions. For four conditions (text or image, addition or subtraction), we learn a small subspace of Gemma4-12B-it's residual stream in which swapping coordinates swaps the model's answer. The subspaces answer three questions. First, do the conditions write numbers along the same directions? They do not. Reusing a subspace from text in an image on the same operation recovers only 53.2% of its causal effect. A single rotation and rescaling maps one subspace onto the other and raises recovery to 91.4%, so the conditions share one geometry expressed in different coordinates. Second, does this shared geometry only reflect the digit the model is about to print? To test this, we split each subspace into 9 directions that the output layer reads to choose the next digit and 13 directions that it cannot see. The 13 hidden directions still carry the number, yet changing them barely alters the digit being written. Third, what are these hidden directions for? A model must emit a multi-digit answer one digit at a time, so it can compute the full result early and keep it in reserve. When two numbers share their first digit, swapping the number in the hidden directions leaves the digit being written intact and redirects the next digit in 77.6% of cases. The model thus forms an abstract numerical concept upstream and unfolds it across time: multimodal representations share meaning through relational geometry across distinct coordinate frames. 

---
# Constraining the Commit Boundary: A DIO/ODI-Informed Authorization Guardrail for Agentic Entity Resolution 

**Authors**: Daniel Beránek  

**Link**: [PDF](https://arxiv.org/pdf/2610.37407)  

**Abstract**: Agentic Entity Resolution (ER) systems iteratively plan actions, retrieve evidence, revise hypotheses, and propose graph mutations. An erroneous identity hypothesis may condition later retrieval and produce apparent confirmation. This paper proposes a Digital Identity Optimization (DIO) and Ontology of Digital Identity (ODI)-informed authorization boundary between evidence available for agent reasoning and evidence eligible to authorize persistent identity-changing mutations.
A domain-governed constraint-binding function maps selected reconstruction-failure distinctions to explicit identity-preserving constraints. An abstract two-phase supervisor permits a commit only when claim-specific authorization-eligible evidence supports every required proposition, relevant conflicts are resolved, the operation remains within its authorized scope, and relevant state and policy versions remain valid. Admission as UNVERIFIED does not confer authorization eligibility. Separate claim-specific promotion governs evidentiary use; insufficient support or unresolved material conflict leads to deferral.
The contribution is a specification connecting diagnostic-to-constraint mapping, evidence promotion, proposition coverage, conflict handling, scope binding, and commit-time version integrity. A proposed evaluation protocol is designed to assess the mapping's incremental value beyond generic authorization and conventional identifier rules. This is a conceptual architecture without implementation or empirical validation, not a theorem of semantic correctness. Its property depends on domain-policy adequacy, evidence promotion, and enforcement. It does not prevent erroneous hypotheses or hypothesis-conditioned retrieval; dependency-aware evidence flow remains future research. 

---
# Clearing the FoG: Training KG Agents to Retrieve the Evidence Behind Their Answers 

**Authors**: Zongyuan Sun, Shuai Zhao, Yingjun Du, Ke Shu, Ziyue Zhu, Haoran Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.37405)  

**Abstract**: Large language models (LLMs) can answer many factual questions correctly, but a correct answer is not necessarily a supported one. For an LLM agent, a knowledge graph (KG) plays two roles: as a source of answers, it supplies facts that the model lacks; as a source of evidence, it supplies the supporting triples that make an answer traceable and auditable. However, existing KG agents treat the KG only as a source of answers: their training data carries answers but not evidence, and their training rewards check the answer but not the retrieved evidence, so agents often answer correctly without retrieving the supporting triples. We propose FoG, a framework that trains KG agents to retrieve the evidence behind their answers. FoG consists of evidence-grounded synthesis, a KG interaction agent, and an evidence-coverage process reward (ECPR): the synthesis recursively hides explicit entities behind verifiable sub-questions, so that every question carries its supporting triples by construction; the agent records the triples returned by the KG; and ECPR credits each supporting triple to the turn that first retrieves it. Experiments on three KGQA benchmarks demonstrate that FoG enables 3B--4B open-source agents to outperform substantially larger baselines in answer accuracy and evidence coverage, with zero-shot cross-KG transfer. 

---
# Agent-Conditioned Routing: Diverse Reasoning Paths Within One Frozen Model 

**Authors**: Ananth Kalyanasundaram, Ankush Kadu, Athira Gopal, Aswanth Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2610.37375)  

**Abstract**: Generating several solutions can expose alternative methods, but additional trajectories often paraphrase one dominant approach. We investigate whether the conditional capacity of a frozen sparse mixture-of-experts (MoE) model can provide such diversity from one problem prompt, without user-assigned roles, prescribed methods, fine-tuning, or model copies. Agent-Conditioned Routing (ACR) assigns $m$ parallel trajectories persistent, untrained shared/private expert-eligibility masks while preserving the learned, input-dependent token router. On a screened 126-question multi-mode mathematics and physics benchmark, with model judges reading the reported derivations in full, we count the number of problems yielding at least two distinct primary methods among soft-correct solutions in four trajectories. ACR reaches approximately twice as many such problems as the strongest temperature setting under GLM 5.3 (37.3 versus 18.7), and 1.67 times as many under GPT-5.6 Sol (45.00 versus 27.00), at matched output budgets. A blinded four-expert audit on a stratified sample of 30 problems also finds more multi-method solutions with ACR on both backbones, with lower correctness. Matched-eligibility ablations show modest multi-method count advantages for ACR, while token-resampled masks attain higher correctness. An informed sequential prompt elicits more diversity still, using previous solutions and explicit requests for alternative methods. ACR adds about 160 KiB of mask state. In a three-problem microbenchmark capped at 1,024 tokens, its latency and peak VRAM usage remain within 3% of batched temperature sampling. 

---
# Uncovering Why LLMs Fail: From Internal Mechanisms to Adversarial Reasoning Trajectories 

**Authors**: Zirui Ding, Xinxin Fan, Hao Yuan, Shaoye Luo, Men Niu, Yunfeng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37369)  

**Abstract**: Large language models (LLMs) are usually evaluated by final-output correctness or safety, yet an endpoint label does not reveal how a failure develops throughout the reasoning process. We therefore ask whether ordinary reasoning failures and adversarially induced failures can be characterized using the same internal coordinates. Towards this end, we in this paper study a white-box diagnostic framework CauTrace to disclose the root-cause of why LLM fails from three fundamental inspections: Functional States, Propagation of Representation, Temporal-State Coordination and Reasoning Trajectory. First, seven functional states are creatively defined to represent the support and constraint processing in LLMs. Second, the propagation of representation change measures how the changes among the seven states persist across subsequent Transformer blocks. Third, temporal-state coordination and trajectory topology characterize when the commitment and verification occur relative to the support, and how the dominant state sequence evolves over the full inference process. Finally, we evaluate CauTrace on both clean reasoning failures and paired adversarial perturbations. Across Qwen and LLaMA-family models, the failed traces show weaker Attention-MLP agreement on support-state updates and less verification activity relative to commitment, while the exact representation-level signatures and timing shifts vary by architecture. Held-out repair and steering tests demonstrate consistent operational signals that were in line with our claims for certain models and specific settings, with aligned steering producing the predicted directional changes in process measures. Overall, CauTrace provides a unified diagnostic framework for analyzing failure dynamics through changes in support, commitment, and verification, which would shed light on the in-depth research on LLM security. 

---
# Aligning Institutional Power and Operational Ability 

**Authors**: Andrea Marino, Giovanni Sileno, Thomas van Binsbergen, Tom van Engers  

**Link**: [PDF](https://arxiv.org/pdf/2610.37342)  

**Abstract**: To be socially sustainable, Agentic AI systems require normative designs that align responsibilities with the ability to fulfil them. Identifying incongruences such as unsatisfiable expectations or privilege escalation is crucial in avoiding normative pitfalls. In this study, we discuss design dimensions suitable for describing multi-agent systems in light of this norm-ability alignment requirement. We propose a formal framework centred on process-oriented modelling and institutional power for the analysis of a given normatively relevant process, discuss strategies for repairing it, and elaborate on how the symmetrical relation between operational process and institutional state affects the lifecycle of normative positions. 

---
# ReForest: Interpretability Meets Reasoning in Isolation Forest 

**Authors**: Juyoung Seo, Geonwoo Bang, Dongho Kim, Moohong Min  

**Link**: [PDF](https://arxiv.org/pdf/2610.37316)  

**Abstract**: Statistical anomaly detection methods like Isolation Forest are efficient but suffer from a critical semantic gap. They identify numerical rarities but fail to understand the contextual meaning of data, often missing subtle, context-dependent anomalies. Conversely, Large Language Models possess strong semantic reasoning capabilities but are prohibitively expensive and slow for large-scale deployment. To bridge this gap, we propose ReForest, a novel framework that synergizes the statistical efficiency of Isolation Forest with the reasoning power of LLMs. ReForest transforms the latent decision paths of Isolation Forest into natural-language explanations, enabling an LLM to refine anomaly detection through an uncertainty-aware dynamic ensemble. This approach allows the model to adaptively weigh statistical and semantic signals based on uncertainty-driven instance-level confidence. Extensive experiments on four real-world datasets (Healthcare, Finance) demonstrate that ReForest significantly outperforms baselines, achieving an average 21% improvement in Recall and a 10% gain in F1-score. Our framework not only improves detection performance but also enhances model transparency by grounding semantic narratives in statistical evidence, providing interpretable rationales essential for high-stakes decision-making. 

---
# RPSD: Retrieval-Privileged On-Policy Self-Distillation with Optimal Transport 

**Authors**: Dongyue Li, Zebang Cheng, Zhongyi Deng, Hongboxu, Yi Bin, Fei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.37281)  

**Abstract**: Sequential recommendation predicts a user's next interaction from historical behavior. Recent LLM-based methods recast this task as generative recommendation and increasingly adopt reinforcement learning such as GRPO for post-training. However, these methods suffer from two limitations, namely credit-blind optimization, where a sequence-level reward is shared across all generated tokens, and collaborative-blind supervision, where each training signal is derived from a single user and ignores behavioral patterns shared across similar users. To address both issues, we propose Retrieval-Privileged on-policy Self-Distillation (RPSD). For credit-blind optimization, we introduce on-policy self-distillation (OPSD), in which a teacher with privileged knowledge re-evaluates the student's on-policy response, and the token-level confidence difference between the student and teacher reshapes the shared sequence-level advantage into token-specific credit. However, the privileged knowledge in vanilla OPSD is still derived from a single user, leaving collaborative-blind supervision unresolved. We therefore ground the teacher in collaborative behavioral evidence from similar users. Specifically, we use temporally aware optimal transport (OT) to compare interaction trajectories across users by jointly considering item similarity and temporal structure, and further apply diversity-aware selection to retain complementary trajectories as privileged evidence. In this way, on-policy self-distillation provides fine-grained token-level credit, while OT enriches the teacher with structured cross-user evidence. To the best of our knowledge, this is the first work to bring on-policy self-distillation to sequential recommendation and to use cross-user behavior as its privileged signal. Extensive experiments across multiple real-world datasets demonstrate that RPSD achieves state-of-the-art performance. 

---
# TCM-DiagCommit: A TCM-SD-Derived Benchmark for Evidence-Conditioned Syndrome Selection and Commitment 

**Authors**: Erwei Hou, Yue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37243)  

**Abstract**: Traditional Chinese Medicine (TCM) syndrome differentiation is usually evaluated as static final-label selection over complete clinical records. This setting hides three diagnostic behaviors: whether a model distinguishes semantic-neighbor syndrome labels from easy random distractors, whether it updates predictions as evidence fields are revealed, and whether it should commit under uncertainty. We introduce TCM-DiagCommit, a benchmark derived from the public TCM-SD/ZY-BERT resource while retaining its source syndrome labels; it converts 1,834 source cases into 23,842 task rows over 148 syndrome labels. The benchmark combines random, profile-neighbor, and full-vocabulary candidate ladders with staged t1/t2/t3 evidence. It also includes fixed-label tongue/pulse perturbations and insufficient-evidence variants. A benchmark-defined action contract scores commit, acquire, defer, and abstain. Across five zero-shot DirectLLM calibrations, random10 top-1 accuracy is 63.2% to 66.4%, but hard20 falls to 24.7% to 30.3% and full148 to 14.8% to 22.3%; on hard20, WT(all) wrong-terminal commits remain 62.2% to 74.3%. These results show that final-answer accuracy alone is insufficient for diagnostic benchmark evaluation: answer selection, evidence use, and benchmark commitment scoring should be measured separately. 

---
# Grounding LLM Agents in Business Processes: Process-Aware Knowledge Graphs for Reliable Multi-Step Service Invocation 

**Authors**: Saiful Khan, Afrah Farea  

**Link**: [PDF](https://arxiv.org/pdf/2610.37242)  

**Abstract**: Large language model (LLM) agents that call enterprise APIs must select the correct endpoints, follow process order, use valid payload keys, and choose coded values from predefined lists. However, current retrieval-augmented service discovery treats endpoints independently and therefore misses process and schema constraints. We address this limitation by integrating Open Data Protocol (OData) metadata and Business Process Model and Notation (BPMN) reference processes in a typed property graph. A hybrid retriever identifies candidate endpoints using dense embeddings and follows process edges to retrieve subsequent endpoints. We evaluate the method on 42 enterprise services with process-chain ground truth. For queries with low lexical overlap between the query and downstream entities, it improves Recall@3 by 5.7 points (n=74), with consistent gains across embedding models and cut-offs. It matches dense retrieval on other queries and provides no improvement on OpenAPI services without a process layer, indicating that the gains arise from process structure. We also evaluate an LLM agent on 81 multi-step queries. It completes 38.7% of tasks, compared with 10.3% for a chunk-based baseline. In an elicitation probe, providing valid code lists eliminates invented codes but leaves invalid keys in 46% of write runs. Full typed context reduces invalid calls to 3% (3 of 101 runs), with no invented codes. These results show that exposing process and schema constraints during retrieval improves the reliability of agent-driven API calls. 

---
# GraphRAGProv: GraphRAG Meets How-Provenance 

**Authors**: Wadie El Amrani, Luis Galárraga  

**Link**: [PDF](https://arxiv.org/pdf/2610.37237)  

**Abstract**: GraphRAG systems often use knowledge graphs to retrieve relevant context for an LLM that produces the final answer. That makes answers more inscrutable and therefore less appealing for domains where transparency is paramount. We present GraphRAGProv, an architecture that executes SPARQL queries over source-grounded RDF graphs to compute answers and their how-provenance polynomials. These polynomials record which statement occurrences were used and how they were combined. Because each occurrence is linked to its source text, users can follow these links to check whether the extracted statements reflect the text and whether the query combines them to answer the question. Querying thereby becomes a way to inspect and evaluate the text-to-graph transformation. Our interactive demo examines 400 cases from two multi-hop QA benchmarks, revealing extraction and query errors even when the final answer is correct. 

---
# Does Your Generalist Skill Optimizer Really Incentivize Agentic Tasks? Rethinking Skill Optimization via Bipartite Attribution 

**Authors**: Jiawei Zhou, Bo Peng, Yuxuan Huang, Ming Zhang, Youbang Sun, Xiang Wang, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37201)  

**Abstract**: Skill optimizers adapt an agent to long-horizon tasks, such as repairing a terminal environment, by rewriting its skill, a natural-language procedure in its context, and keeping the edits that raise success on a small dev set. On such tasks success is noisy, so a gain on a small dev set does not show that an edit helps, and a success score does not show what the edit costs. We introduce BiPA, which replays an optimizer's acceptance rule on candidates identical to its seed to measure what selection alone reports, and records each episode as a bipartite graph of turns and the text they read to trace extra input tokens to the behavior behind them. Across three optimizers, three agentic benchmarks and two models, the learned skills succeed about as often as their hand-written seed, 0.1 points higher on average. Their dev gains vanish on test: keeping the best of several candidates on at most 30 tasks reports gains this large even when none changed. They read about 16% more input tokens: the optimizers almost always add rules, half to re-check work and some to act one call at a time, and the agent follows them as extra turns, each re-sending the whole conversation. Four principles follow: accept an edit only if its gain exceeds what selection alone reports; charge it for the turns it adds, not for its length; let the editor delete and merge rules; show it what earlier rules cost. Code is available at this https URL. 

---
# Rènyi-Regularized Flow-Based Variational Inference for Bayesian Calibration 

**Authors**: Exaucè Luweh Adjim Ngarti, Élise Arnaud, Léo Nicoletti, Victor Trappler, Arthur Vidard  

**Link**: [PDF](https://arxiv.org/pdf/2610.37169)  

**Abstract**: Variational inference with normalizing flows provides a flexible posterior approximation for Bayesian model calibration, but struggles when the posterior has several modes separated by regions of low probability mass. We study this behavior in a two-parameter twin experiment and regularize the variational objective. Under Gaussian observation errors and a Gaussian prior, the corresponding minimization objective is the expected Tikhonov calibration objective minus the differential entropy of the approximating distribution. However, the expected Tikhonov term favors concentration near one of the modes, while entropy maximization does not guarantee multimodality. We introduce a regularization term based on Rényi divergence that encourages the approximate posterior to cover the prior support. In the connected bimodal case, a flow approximation represents both modes; in the case with well-separated modes, the flow and Gaussian mixture frequently miss one mode. The introduced regularization reduces the near-collapse fraction from 50% to 25%. While the regularization improves mode coverage in this experiment, it changes the Bayesian variational objective, introduces a trade-off in approximation accuracy, and does not provide a general solution to mode collapse. 

---
# LiveSearch: Questions as Executable Verifier Code for a Live Benchmark of Professional-Source Financial Search Agents 

**Authors**: Sinuo Wang, Wentao Zhao, Jiacheng Lu, Rui Sun, Beidi Luan, Qiming Shi, Jing Li, Daxin Jiang, Zuo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.37166)  

**Abstract**: In finance, the answer that matters is usually the current one: prices, rates, and disclosures hold only as of a given day, so retrieving up-to-date facts is a core capability of a financial agent. Yet most financial benchmarks overlook timeliness. They fix each answer at annotation time, so the gold goes stale as markets move and, once leaked into training data, rewards recall rather than search and analysis. We introduce LiveSearch, a live benchmark that stores each question as executable verifier code: a typed program of calls to a professional financial data service, followed by a declarative extractor. The question is frozen, while its gold answer is determined by executing the program against professional financial data at evaluation time, and a daily job re-executes it with no human or language model in the loop. A memorized answer is thus likely outdated, and a model must reach the current one through tools and analysis. Construction is equally automatic: a language model only verbalizes programs enumerated from real tool responses and never chooses an answer. Under general web search, six contemporary tool-calling models reach 70.9% to 76.7% answer accuracy. Over three consecutive days, 70% of served questions change their gold value, and even under tolerance-aware matching, a literally reused earlier snapshot fails on a majority of questions within a single day. Per-question tracking uncovers dynamic shifts in correctness masked by aggregate accuracy. 

---
# WorkFactory: Turning Professional Intents into Executable Work for Scalable Agent Training 

**Authors**: Ziyang Zeng, Tingyang Chen, Jia Deng, Qiyao Wang, Shuo Tang, Yike Yang, Tao Ren, Wayne Xin Zhao, Feng Chang, Jujie He, Chuan Hao, Ran Tao, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2610.37161)  

**Abstract**: Scaling supervised training for working agents requires diverse professional tasks that yield reliable expert demonstrations. However, missing or inconsistent task information can obstruct teacher execution and weaken the resulting supervision. We propose WorkFactory, a scalable framework for constructing coherent and executable professional tasks. Starting from professional intents, information-level planning builds an information support graph to guide task instantiation, explicitly representing the dependencies and locations of information required for the intended outputs. Probe executions then identify support failures and guide repairs while preserving the intended work. We construct 112,686 tasks spanning 612 occupations and use the collected teacher trajectories to fine-tune Qwen models at 4B, 27B, and 35B-A3B scales, yielding the WorkPilot family. All three models improve over their respective base models across four professional-work benchmarks, with WorkPilot-35B-A3B improving from 879 to 1565 Elo on GDPval-AA v2.1. Scaling analyses reveal model-dependent diminishing returns, with execution length first increasing and then decreasing. Controlled component analyses show that planning and repair reduce probe-diagnosed task environment defects and improve downstream performance. The data and models are available at this https URL. 

---
# SFE-O: An Ontology for Knowledge-Grounded AI in Supercritical CO$_2$ Biomass Extraction and Fractionation 

**Authors**: Thomas Pallet, Mario Lezoche, Gianpaolo Leone, Daniele Pizzichini, Diego Torres, Herv{é} Panetto, M{é}lanie Glibusic  

**Link**: [PDF](https://arxiv.org/pdf/2610.37123)  

**Abstract**: Supercritical CO$_2$ extraction (CO$_2$-SFE) offers a sustainable route for recovering high-value compounds from biomass while limiting the use of conventional organic solvents, but its design and interpretation require heterogeneous knowledge spanning biomass properties, operating conditions, mass transfer, and product fractionation, typically fragmented across technical reports, engineering models, external databases, and expert practice. This paper introduces SFE-O, a modular Web Ontology Language (OWL 2) ontology for supercritical CO$_2$ biomass extraction, structured around interconnected Biomass, Process, and Product dimensions and extended with explicit representations of process topology, parameter provenance, external and internal mass-transfer phenomena, and the Constant Extraction Rate (CER) / Falling Extraction Rate (FER) mechanisms. Computational dependencies among process quantities are represented explicitly, without embedding numerical solving in the ontology, so that input, measured, computed, externally retrieved, and fitted parameters remain traceable. SFE-O reuses established semantic resources (ChEBI, QUDT, OWL-Time, PROV-O) and is evaluated through a worked example that instantiates a real pilot-scale extraction batch, a reasoner-based consistency check, and a competency-question SPARQL query. The resulting knowledge model provides a machine-actionable foundation for querying, consistency checking, and mechanistic explanation of SFE processes, and establishes a knowledge layer that could support future ontology-grounded AI reasoning, a direction this paper motivates but does not evaluate. 

---
# Prospector: learning a compact graph traversal policy for scientific reasoning 

**Authors**: Haiqian Yang, Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2610.37121)  

**Abstract**: Scientific reasoning in specialized engineering fields depends on knowledge scattered across the literature, often proprietary and absent from frontier LLM training data. In such domains, frontier models tend to give generic answers. Knowledge graphs (KGs) can ground LLM reasoning. Because a graph is too large to provide in full, typical LLM+KG systems preselect a portion, deciding relevance before reasoning begins. Here we train a compact model, Prospector, to reason over a domain knowledge graph by traversing it step by step, seeking evidence in light of what it has already found. Prospector calls a small set of tools one at a time to progressively build its context, and answers with a chain of evidence that can be checked edge by edge against what its tools retrieved from the graph. On a knowledge graph built from the semiconductor-manufacturing literature, Prospector approaches the strongest off-the-shelf models given the same graph and tools, and it far exceeds off-the-shelf models answering closed-book or from a pre-selected portion of the graph. Although trained only on the semiconductor graph, Prospector transfers to unseen knowledge graphs from three other distinct fields: bio-inspired materials, per- and polyfluoroalkyl substances (PFAS) and materials properties. Served on a single GPU, Prospector answers faster and at a fraction of the cost of a hosted service, while keeping proprietary data on the owner's hardware. We find that the policy's latent state is indicative of its next move on both seen and unseen graphs, and modulating the latent state can steer the policy from one behavior mode to another. These results suggest a practical LLM + KG paradigm for scientific and engineering reasoning over private data, in which a compact model trained on a graph from published literature transfers its traversal to unseen graphs that stay with their owners. 

---
# OpenGray: a benchmark for evaluating the decisions of radiotherapy planning agents 

**Authors**: Humza Nusrat, Kundan Thind  

**Link**: [PDF](https://arxiv.org/pdf/2610.37081)  

**Abstract**: Background and purpose: Radiotherapy planning agents based on large language models (LLMs) decide when to submit a plan or request review but are evaluated mainly by dose. This study developed OpenGray, a benchmark of planning tasks with known correct decisions. Materials and methods: OpenGray adds tasks with known answers to the OpenKBP-Opt head-and-neck dataset, including two contradictions with dose goals proven incompatible and a feasible control with a stored plan meeting every hard goal. Four systems, from a rule-based planning script to a full LLM agent, were evaluated on 30 patients, the LLM-based ones with a commercial (Terra) and a local (Ministral 3 8B) LLM. Decisions, achieved dose, problem recognition, and technical failures were recorded and also read from logs of 39 stereotactic radiosurgery runs of SAGE, a clinical LLM agent. Results: With Terra, the script preceded by an LLM review and the full agent had decision accuracies of 95.4% and 92.1%, respectively, yet escalated only 50% (95% confidence interval, 25-75%) and 25% (10-40%) of overlap contradictions, which required arithmetic on the anatomy. All four systems submitted every feasible control, but 24% of these plans missed a hard goal that the stored plan had met. In four SAGE runs, one goal failed at every attempt; the agent identified it in every reply but had no action for stopping. Conclusions: Favorable aggregate scores concealed specific decision failures. Planning agents should be evaluated on their decisions as well as their dose and given an explicit action for stopping and requesting review. 

---
# Correctness Conditioning as Selection in Chain-of-Thought Faithfulness Evaluation 

**Authors**: Yueer Pan, Yuexi Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.37070)  

**Abstract**: Restricting chain-of-thought (CoT) faithfulness evaluation to correctly answered instances changes the population being measured. We study this selection step in a seven-variable structural causal model and derive the exact observed gap $\Delta = \operatorname{Cov}(A, C)/P(C=1)$. The model distinguishes difficulty confounding from shortcut-induced collider bias and states the conditional-independence assumption needed to express the gap through latent response propensities. A protocol review of 18 papers distinguishes answer-correctness restrictions from agreement-based selection, benchmark construction, and training-data filtering. We evaluate four sub-ceiling deployments on arithmetic tasks and observe positive selection gaps in all four. The primary GLM-5.3 run yields a +7.4 percentage-point gap ($n = 1000$, $p < 0.0001$), and pooled Cochran-Mantel-Haenszel analysis gives OR = 1.70 ($p < 10^{-6}$). An assumption-free covariance decomposition assigns 42.8% of the GLM-5.3 gap to between-stratum variation in interventionable-operation count, leaving the majority unattributed. When counterfactual compliance is observable for every instance, evaluating all traces directly estimates population faithfulness without correctness filtering. 

---
# AI-Assisted Discovery of CNOT Decompositions for Quantum Permutations on Constrained Topologies 

**Authors**: Daniele Lizzio Bosco, Jacopo Cossio, Simone Faro, Francesco Pio Marino, Gabriele Messina, Carla Piazza, Giuseppe Serra  

**Link**: [PDF](https://arxiv.org/pdf/2610.37045)  

**Abstract**: Efficient quantum circuit compilation is challenging for quantum computers, where limited qubit connectivity can introduce substantial overhead in the implementation of logical operations. Recent AI-assisted synthesis methods have shown considerable potential for reducing circuit complexity, but their role in uncovering principles that can guide the design of new deterministic compilation algorithms remains largely unexplored. In this work, we investigate this perspective for the synthesis of quantum permutations into CNOT circuits under constrained qubit connectivity. Using AlphaCNOT as an exploratory tool, we identify a recurring decomposition for remote transpositions that improves on the construction based on adjacent SWAP routing. For two qubits at graph distance $d>1$, the discovered pattern yields a topology-legal implementation using at most $6d-4$ CNOT gates, compared to the $6d-3$ gates required by the corresponding SWAP-based construction. We formalize this observation through a constructive proof to define Global Transpositions, a deterministic synthesis method that searches for a minimum-cost decomposition of an entire permutation into topology-aware logical transpositions. The resulting method subsumes optimal SWAP routing within its synthesis model and provides certified upper bounds on the topology-constrained CNOT cost. Our experiments also reveal circuits that cannot be expressed as concatenations of independently synthesized transpositions. Instead, these circuits exploit intermediate linear transformations that temporarily leave the permutation subgroup and combine partial effects of multiple logical operations, a mechanism we call intermediate CNOT fusion. Our results show that AI-assisted synthesis can support not only circuit optimization, but also the discovery of reusable identities, provable constructions, and hidden mechanisms in quantum circuit compilation. 

---
# Guided-RLVR: Guided Exploration in Reinforcement Learning with Verifiable Rewards 

**Authors**: Bo Yang, Zishuai Zhang, Weihua Li, Qinnan Zhang, Sijia Wen, Hainan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37041)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has shown strong empirical gains in improving the reasoning ability of large language models (LLMs) on tasks such as mathematics and programming. However, recent studies have shown that RLVR rollouts often collapse onto a narrow set of reasoning trajectories, limiting the explored reasoning space. Motivated by findings that thinking tokens at critical steps serve as critical structural elements in LLM reasoning pathways, we propose Guided-RLVR, a novel approach that augments RLVR with a lightweight controller. The controller strategically injects semantically meaningful steering tokens into the sampling distribution, encouraging the model to approach problems from multiple reasoning perspectives and thereby expanding the exploration space. Experiments on mathematical reasoning and code generation benchmarks show that Guided-RLVR consistently outperforms existing RLVR methods while producing more diverse reasoning trajectories. 

---
# SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation 

**Authors**: Zafer Sahinoglu, Abdulazeez Mohammed, Buse Yaren Kazangirler, Cahit Berkay Kazangirler  

**Link**: [PDF](https://arxiv.org/pdf/2610.36991)  

**Abstract**: Retrieval-augmented generation (RAG) systems ground large language models (LLMs) by retrieving external evidence and conditioning generation on the retrieved context. Although dense retrieval is effective for local semantic matching, independently ranked retrieval can return redundant passages, miss cross-chunk dependencies, and provide incomplete evidence for multi-hop reasoning. We propose SpectRAG, a spectral graph-augmented retrieval framework that combines hybrid candidate seeding, weighted chunk graph construction, graph-based expansion, spectral relevance propagation, and token-budgeted context selection. Document chunks are represented by dense semantic embeddings and graph-spectral coordinates derived from a weighted chunk graph whose edges encode semantic similarity, document adjacency, section hierarchy, shared entities, citation links, and similarity between LLM-generated utility questions. At query time, SpectRAG builds a high-recall seed set using exact inner-product search and hybrid relevance signals, expands it over the chunk graph, re-scores candidates using graph proximity and spectral information, and assembles the final context under a fixed token budget. The goal is not merely to retrieve individually similar chunks, but to construct a context that is relevant, diverse, structurally connected, and sufficient for grounded reasoning. Experiments on MultiHopRAG, UltraDomain, and long-document question-answering settings show that SpectRAG improves broad evidence recovery and achieves competitive generation quality against dense, hybrid, HyDE, LightRAG, and NaiveRAG baselines. The results further indicate that graph-spectral structure is particularly useful for inference-oriented and cross-document retrieval, while highlighting trade-offs in early precision, computational cost, and domain-dependent performance. 

---
# AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations 

**Authors**: Qiming Shi, Yikun Li, Beidi Luan, Rui Sun, Yan Ma, Jing Li, Daxin Jiang, Zuo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.36980)  

**Abstract**: Large language model (LLM) agents are extending agent-mediated commerce on the Web from shopping for consumers to operating businesses for sellers. In online-to-offline (O2O) commerce, decisions are issued through Web-based platforms but realized through costly physical execution. Evaluating such agents requires going beyond individual transactions to ask whether decisions made online turn into sustained economic returns. We introduce AutoVend, a benchmark in which one LLM agent manages 20 vending machines across five site types through a Web-based operating interface for 180 simulated days. Its data-grounded simulator combines a shared warehouse, procurement lead times, paid replenishment, inventory expiration, and recurring operating costs. We evaluate five LLM agents against a rule-based inventory-control baseline and diagnose their site coordination, execution, and performance over time. Four of five evaluated LLM agents fall short of this baseline, and returns per dollar of inference cost rank models differently from returns alone. Operational analyses show that gains in individual metrics do not consistently yield stronger economic growth, as inventory and execution costs erode these gains. Trajectory analysis further reveals difficulties in grounding feasible actions, managing inventory commitments, and converting feedback into effective policy revisions. 

---
# Proactive Memory for Event-Driven LLM Agents 

**Authors**: Jinyang Li, Zichuan Fu, Xian Wu, Wenlin Zhang, Yimin Deng, Kaifeng Guo, Guojing Li, Junjia QI, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.36967)  

**Abstract**: Proactive LLM agents anticipate service opportunities from ongoing observations and accumulated history. Existing approaches emphasize recent context or retrieval of related memories, providing limited support for validating recurring behavioral regularities and assessing their applicability to the current event. We propose Proactive Memory (ProactMem), which organizes history into recent events, temporal trajectories, and behavioral regularities. Trajectory facts are checked against source events, and regularities are assessed through historical replay before being matched to current conditions. We also construct ProactiveStream, a longitudinal benchmark covering three application scopes, where agents process isolated event streams chronologically under episode-aware opportunity labels. On ProactiveStream, ProactMem achieves an overall effectiveness (F1) of 52.6%, a 21.4% relative improvement over the strongest of five external baselines. Code is available at: this https URL. 

---
# Do bots feel bad? Assessing empirical evidence for functional valence in LLMs 

**Authors**: Iwan Williams  

**Link**: [PDF](https://arxiv.org/pdf/2610.36962)  

**Abstract**: Do LLMs have valenced states, i.e. states that feel pleasant or unpleasant? Recent empirical work has claimed to show that LLMs possess functional valenced states (e.g. ``functional pleasure and pain'' [Ren et al. 2026], ``functional emotions'' [Sofroniew et al. 2026], and ``functional welfare'' [Han et al. 2026]) that play the same causal role as valenced states in cognition and action, without necessarily being consciously experienced. But it remains unexplored how well the identified states fit the functional role of valence as understood in the philosophical and cognitive science literature. Here, I assess the current body of evidence, and argue (i) that key aspects of valence remain untested in LLMs, and (ii) that extant empirical paradigms inadequately distinguish (functional) valence from neighbouring constructs like preferences, value-judgments, valence attributions and role-play. I conclude by sketching some novel experimental approaches to better target (functional) valence itself. 

---
# Vision-Language Agents Can Learn in Their Own Latent Worlds 

**Authors**: Haoxi Li, Sikai Bai, Jianfei Ma, Qinglin Hou, Xiaocheng Lu, Tao Han, Jie Zhang, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.36925)  

**Abstract**: Learning from experience is essential for building the next generation of agents that continually expand their capabilities. Pretrained vision-language models (VLMs) already internalize rich world knowledge, yet this knowledge is rarely leveraged to drive policy improvement through online interaction. Realizing this potential efficiently remains challenging, as policy updates shift the agent's interaction distribution, requiring the world model to adapt accordingly, while continually updating large pretrained models is computationally expensive. To address this challenge, we introduce Dwell, a model-based reinforcement learning (RL) framework that reframes a VLM as a recurrent latent transition operator, endowing it with the ability to predict, adjust, and act in latent space. Leveraging the VLM's own representations and its dual role as both world model and policy, the agent continually learns from real interactions to anticipate and evaluate action outcomes, and from latent imagination to improve its behavior. Across five interactive visual environments, Dwell achieves the strongest overall performance among trainable methods while substantially reducing the compute required for imagination, showing that effective world models for VLM agents need not reconstruct the future; they need to predict futures that are useful for action. Our code is publicly available at this https URL 

---
# Natural Hard-Negative Mining from Positives for Instruction-Following Retrieval 

**Authors**: Chaolang Li, Jingyu Li, Zhaocheng Du, Xuyang Xie, Gang Wang, Pengwen Dai  

**Link**: [PDF](https://arxiv.org/pdf/2610.36900)  

**Abstract**: Instruction-following information retrieval (IFIR) requires retrievers to rank documents by both a query and its instance-specific instructions, yet instruction-conditioned contrastive supervision remains scarce. Existing synthetic methods typically fix a positive passage and generate additional negatives, overlooking the hard-negative potential of the original positive. We propose Natural Hard-Negative Mining (NHNM), which promotes convertible negatives under query-compatible instructions and repurposes original positives as natural instruction hard negatives. Its automated Candidate Cleaning, Reversal Construction, and Geometry Filtering pipeline produces compact, high-contrast supervision aligned with the pretrained retrieval space. Applied to MSMARCO, NHNM yields NHNM-MARCO, an IFIR training set of nearly 7K instances. On InstructIR and FollowIR, models with 568M to 4B parameters fine-tuned on NHNM-MARCO achieve strong instruction-following performance and match or surpass substantially larger retrievers in several settings. Controlled analyses and ablations attribute these gains to natural hard-negative supervision rather than model scale or training-data volume. 

---
# Noise Steering: Exploiting Wake Steering and Acoustic Directivity for Noise-Constrained Wind Farm Control via Reinforcement Learning 

**Authors**: Martín de Frutos, Baptiste Corban, Claire Bizon Monroc, Ana Bušić, Donatien Dubuc, Jiamin Zhud, Esteban Ferrer  

**Link**: [PDF](https://arxiv.org/pdf/2610.36888)  

**Abstract**: Existing noise mitigation strategies for wind farms usually consist in reducing turbines power output, without accounting for the wake interactions between them and the noise emission directivity. This leads to over-conservative policies and unexploited wind power. Yet, turbine yaw can simultaneously affect wake interactions and the direction of noise emission, offering an additional degree of freedom for farm-level noise control. We therefore propose Noise Steering, a model-free reinforcement learning (RL) strategy that jointly controls turbine yaw angles and power setpoints to maximise farm power while satisfying a local noise constraint in a defined region. To enable training at a reduced computational cost, a low-fidelity acoustic model is coupled with the steady-state FLORIS simulator. The RL agent effectively learns how to redirect wakes and noise directivity away from the protected region while redistributing power across the farm. On a six-turbine test farm, the policy satisfies the 40 dB noise constraint throughout the full range of turbulent winds, with an average power deficit of only 2% relative to greedy unimpeded control, and up to 7.7% at high wind speeds. In comparison, the industry-standard noise-reduced operation (NRO) baseline here generates an average power loss of 17%. When deployed in the higher-fidelity this http URL simulator, the policy largely retains its performance under steady and turbulent inflow, despite differences in simulation fidelity. These results demonstrate the potential of combining yaw control and power derating through RL to improve the power-noise trade-off in wind farm operation. 

---
# RubricAtlas: Mining Type-Level Rubrics from Expert Reports to Evaluate Deep Research Agents 

**Authors**: Delin Wang, Sinuo Wang, Rui Sun, Beidi Luan, Qiming Shi, Jing Li, Daxin Jiang, Zuo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.36878)  

**Abstract**: Deep research agents conduct multi-step Web searches and synthesise retrieved information into analytical reports. Evaluating these outputs requires checking which analyses they deliver, beyond the presence of relevant facts. Query-specific binary rubrics make individual requirements explicit, but task-level checklists may leave the overall analytical structure underexamined. We introduce RubricAtlas, which derives type-level analytical requirements from expert reports on different objects within the same research type and organises detailed checks around recurring analytical questions. The requirements give related tasks a shared standard for what to analyse, without prescribing their answers. Our framework links binary checkpoints to report passages and evaluates completeness separately from argument validity. Built from 3,079 Chinese financial research reports, RubricAtlas covers 99 research types. Reference-report selection experiments show that individual reports cover only about half of their type's analytical questions, and that selecting requirements from different reports changes evaluation scores and agent rankings. Across ten agents, two judge pipelines achieve checkpoint agreement of $\kappa$ = 0.71 and agent-ranking agreement of Kendall $\tau$ = 0.87. Controlled deletions reduce scores on the affected analyses, while padding produces no detectable increase. These findings support type-level rubrics for comparing the analytical completeness of Web deep research outputs, with individual decisions traceable to report evidence. 

---
# GUI-JEPA: Candidate-Conditioned JEPA Latents as a Fast Action Policy for Mobile GUI Agents 

**Authors**: Bingchao Wang, Yongcheng Jing, Shunyu Liu, Chengyu Wang, Jun Huang, Jie Yang, Wei Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2610.36877)  

**Abstract**: Mobile GUI agents are typically fine-tuned 7B-scale vision-language models (VLMs) that generate each action as text. However, these VLMs decode every action token by token, and even agents that score candidate actions keep a large language model or VLM in the loop, making them slow and memory-hungry. To address these issues, we propose GUI-JEPA, to our knowledge the first decoder-free GUI agent policy built on a Joint-Embedding Predictive Architecture (JEPA), which selects actions in the latent space of a JEPA encoder pretrained on unlabelled GUI screenshots. Its action- and region-conditioned latent predictor turns the screen latent into one latent per candidate action, letting lightweight heads score all candidates against the instruction in a single batched pass, without decoding a token. We further show that the scorer relies on this latent pathway and that the predictor is trained for the decision itself: a randomly initialised predictor does just as well as one pretrained on screen dynamics. On AndroidControl, GUI-JEPA is 2.8-4.0x faster on the same GPU than a Qwen2.5-VL-7B agent we fine-tune with the same inputs, while reaching 86-92% of its in-distribution accuracy with 6% of its parameters. Code is available at this https URL 

---
# Evolving Agent Teams 

**Authors**: Shiyi Cao, Ziming Mao, Dacheng Li, Joseph E. Gonzalez, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2610.36848)  

**Abstract**: The rapid progress of software engineering agents drives interest in applying them to open-ended problems such as GPU kernel optimization. However, we find that existing coding agents converge to local optima on real-world, expert-written kernels (e.g., FlashInfer). We introduce Evolving Agent Teams, a multi-agent framework that escapes such local optima. It features (1) a central supervisor that diagnoses each local optimum and restructures a team of agents; and (2) a persistent file system that comprises a learning document, an evolution history, and per-team artifacts, grounding restructuring with accumulated experience. Under a matched token budget, Evolving Agent Teams consistently outperforms strong coding-agent baselines. With an extended token budget, Evolving Agent Teams generates a CUDA kernel that reaches 1.2x mean speedup over the production FlashInfer MLA paged decode kernel from scratch and wins on 36 of 47 production workloads from FlashInfer-Bench (0.86-0.99x on the remaining 11). It also demonstrates superior performance on tasks such as KDA prefill on AMD MI350X (3.01x over vLLM) and NVIDIA B200 (3.44x over FlashKDA), MoE expert-offload serving on Apple silicon (30.3 tokens/s decode for Qwen3.8-Flash-Next in MLX 4-bit on a 48 GB M4 Pro), and RoboDojo robot manipulation (62.1 on held-out seeds vs. 55.3 for the best leaderboard entry). Code is available at this https URL. 

---
# Conservative Policy Iteration For Zero-Sum POSGs 

**Authors**: Erwan C. Escudie, Matthia Sabatelli, Jilles Steeve Dibangoye  

**Link**: [PDF](https://arxiv.org/pdf/2610.36847)  

**Abstract**: Occupancy-state reductions make dynamic programming possible for zero-sum partially observable stochastic games (zs-POSGs). Existing point-based solvers usually optimise a central value function. This restores Bellman recursion, but it requires point-based updates over high-dimensional occupancy spaces. We introduce an approximate policy-iteration algorithm for finite-horizon zs-POSGs. The method maintains one policy per player and updates both jointly along occupancy trajectories induced by conservative policy updates. Instead of approximating a full central value function, each update uses a single supporting hyperplane induced by the current joint policy. The algorithm alternates local policy improvements, conservative updates, and persistent exploration. Computation is therefore concentrated on occupancy regions where the current joint policy is exploitable. On finite-horizon zero-sum POSG benchmarks, the method is competitive at small horizons and often improves the runtime-exploitability trade-off at larger horizons compared with PBVI, HSVI, and CFR+. 

---
# When the Critic Falls Behind: Co-Evolving Rubrics and Harnesses for Continual Agent Self-Improvement 

**Authors**: Hao Chen, Yukun Yan, Ziyu Han, Ye He, Chuanzhe Guo, Shangzhan Li, Qi Shi, Qingfu Zhu, Maosong Sun, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2610.36826)  

**Abstract**: Autonomous agents increasingly depend on harnesses that govern interaction, memory, tools, and execution around large language models (LLMs). While recent work has begun to evolve these harnesses from execution experience, the supervision used to guide evolution often remains relatively static, making it difficult to track changing behavioral weaknesses across rounds and potentially limiting generalization to unseen tasks. To address this issue, we propose RuHarness, a rubric--harness co-evolution framework for continual agent self-improvement. RuHarness continually discovers and reassesses behavioral criteria from execution trajectories, prioritizes the most relevant unresolved weaknesses, and uses them to guide targeted harness updates while keeping the task objective fixed. Extensive experiments across diverse agent tasks and backbone models show that RuHarness outperforms strong harness-evolution baselines, particularly on unseen held-out tasks. Further analyses show that adaptive rubric prioritization tracks shifting weaknesses and supports more sustained improvement and stable generalization across evolution rounds. 

---
# Who Knows What - and What Does It Support? Perspective-Bounded Evidence Calibration for Multimodal Theory of Mind 

**Authors**: Ziqi Liao, Jiaqi Shen, Yiming Na, Kedi Lyu, Shao-Yuan Lo, Hongxia Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.36824)  

**Abstract**: Theory of Mind (ToM), the ability to infer others' beliefs, goals, and intentions, is a cornerstone of social intelligence. Multimodal ToM requires inferring an agent's mental state from evidence that is available to that agent, rather than from the complete episode observed by the solver. Existing LLM-based and Bayesian model-based approaches reconstruct global event traces, yet they do not explicitly distinguish objective events from agent-admissible evidence during hypothesis evaluation. Consequently, agent-inaccessible events or semantically misaligned actions and utterances may influence the inferred belief, goal, or social intention, leading to perspective leakage and source or route confusion. We formulate this limitation as an evidence-admissibility problem: evidence should contribute to a mental-state hypothesis only when it is accessible to the relevant agent, preserves its information source, and is diagnostic of the queried mental-state variable. To operationalize this principle, we propose PIP-ToM, an evidence-admissibility interface that constructs agent-specific evidence stores, aligns them with structured mental-state claims, and calibrates model-based hypothesis scores while preserving the underlying Bayesian inference process. PIP-ToM with GPT-4o outperforms existing ToM methods across MuMA-ToM, MMToM-QA, and Hi-ToM, achieving average accuracies of 92.56%, 95.00%, and 95.00%, respectively. 

---
# An Incremental Knowledge-Graph Learner Digital Twin for Evidence-Grounded Early Warning 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel, Bertrand Laforge  

**Link**: [PDF](https://arxiv.org/pdf/2610.36820)  

**Abstract**: Early-warning models in learning analytics can identify learners at risk of failure or withdrawal, but the resulting alerts are often only weakly connected to inspectable learning evidence. A risk score may indicate that intervention is needed without showing which traces, assessment events, or temporal patterns justify the alert. This paper introduces an incremental Learner Digital Twin for evidence-grounded early warning, implemented as a lightweight knowledge graph. The twin is updated at temporal checkpoints by replaying learning traces and assessment records, and it serves both as a feature-extraction substrate and as an explanation-grounding substrate. Predictive features are derived through graph-mediated queries, while features, attributions, alerts, evidence items, and raw records are represented as queryable graph entities. The framework is instantiated and evaluated on a public learning-analytics benchmark under a temporal protocol. The results show that graph-mediated feature extraction preserves the predictive behavior of equivalent raw-table features across model families and checkpoints. Generated alerts are materialized as evidence chains linking risk decisions to feature attributions, evidence items, and raw platform records. The study shows that early-warning prediction can be routed through a semantic and provenance-aware representation while preserving predictive performance and adding traceability, verifiability, and evidence-grounded explanation. 

---
# Dependency Graphs of Mathematical Arguments: Representation, Structural Measurement, and Proof Review 

**Authors**: Eunji Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.36776)  

**Abstract**: We propose a source-grounded framework for representing and measuring mathematical arguments with assistance from large language models. It includes numbered and unnumbered statements, equations, assumptions, and constructions, distinguishing source occurrences from contextual states and inference records. Proposed dependencies retain source evidence and review status. A measurement contract fixes manuscript versions, boundaries, granularity, proof choices, and macro anchors. On a selected finite directed acyclic graph, depth is the longest prerequisite-path length, and bulk counts distinct ancestor edges. We give a union recurrence that accounts for shared subarguments and identify reachability conditions preserving depth and the macro skeleton; unrestricted refinement need not preserve depth or bulk. A standardized, versioned prompt protocol separates annotation discovery, deterministic recomputation, and review assistance. Cross-paper links support a research agenda on theorem co-use, rare combinations, and connections between dependency communities. These patterns motivate probabilistic premise suggestions and review of possible omissions or unchecked application conditions in human-written and AI-generated proofs. Further applications include source-linked reading support and comparison of individual and collaborative argument shapes. We provide an evaluation design rather than empirical evidence of these benefits. Suggested premises require mathematical validation, and the measures describe a specified representation rather than intrinsic mathematical difficulty, originality, or importance. 

---
# Batching the Reflective Optimization Loop: Parallel Proposals Make GEPA Faster and Better 

**Authors**: Jialin Zhang, Lakshya A Agrawal, Shangyin Tan, Donghyun Lee, Lutfi Eren Erdogan, Dan Klein, Koushik Sen, Alex Dimakis, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2610.36774)  

**Abstract**: Reflective text optimization makes agentic systems better, but the optimization loop is itself a resource-hungry workload whose sequential structure leaves parallel evaluation capacity idle. We extend the open-source GEPA optimizer to propose and evaluate a batch of candidates on each optimization step instead of one candidate at a time. In our sweep on two tasks, most batched runs finished in half the wall-clock time or less, and the fastest in about a quarter to a third. Batched settings also achieved higher held-out test scores, from 68.9% to 72.1% on LiveBench-Math (with $2\times2$) and from 49.0% to 60.0% on HoVer (with $8\times1$). A simple runtime model explains the measured speedups under a finite worker pool, and batched settings also spend their metric-call and dollar budgets more efficiently. We frame the axes along which a reflective optimization step can scale in parallel, a first look at allocating a fixed optimization budget across them. 

---
# Can LLMs Reliably Grade Olympiad Proofs? A Study of Mathematical Verification with LLMs 

**Authors**: Azim Ospanov, Zijin Feng, Ding Ding, Chengwu Liu, Haoli Bai, Jiacheng Sun, Lifeng Shang, Farzan Farnia  

**Link**: [PDF](https://arxiv.org/pdf/2610.36767)  

**Abstract**: Large Language Models (LLMs) have made remarkable progress on solving competition-level mathematics, yet their ability to verify natural-language mathematical proofs remains relatively underexplored. This gap is especially pronounced for Olympiad-level proofs, which require meticulous assessment of each claim in a long reasoning chain. Current proof verification thus primarily relies on expert inspection, which is costly to scale. In this paper, we present a systematic study of LLM-based verification in this setting. Specifically, we introduce a human-curated dataset of USAMO 2025 candidate solutions with expert grading, and conduct a broad evaluation of existing LLM verifiers across frontier models. We also propose TROJAN, an iterative self-critique pipeline for generating high-fidelity adversarial proofs at scale. Our comprehensive evaluation covers 20 models spanning seven metrics, two prompt styles, three inference methods, and three prompt templates, showing that state-of-the-art models still leave considerable room for improvement. To narrow this gap, we propose MABGrader, a Multi-Armed Bandit framework that reframes grading as arm selection over the discrete score space. MABGrader outperforms the SOTA ProofGrader by 26.4% in Quadratic Weighted Kappa and reduces grading error by 14%. Code and dataset are available at this https URL. 

---
# Beyond the Evidence Coherence Trap: Agent-Driven Evidence Selection for Temporally Disordered Fake News Detection 

**Authors**: Chi Wang, Min Gao, Yifei Zhang, Bincheng Gu, Zongwei Wang, Wentao Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.36764)  

**Abstract**: Fake news undermines public trust and the integrity of online information ecosystems. Temporally disordered fake news is especially elusive because it rearranges factual events into a false chronology. Individual events may remain factually valid even when their temporal relations are misrepresented. When similar events involving the same entity recur, retrieval may return semantically plausible evidence from the wrong period. Consequently, independent event verification may rely on locally coherent but temporally mismatched evidence and incorrectly validate the article. We call this the \textit{evidence coherence trap}. Because existing benchmarks rarely isolate this setting, we construct ChronoFake, a source-disjoint benchmark of 30,706 articles derived from temporally anchored real events. It distinguishes temporal manipulation from valid narration, including flashbacks. To address this, we propose DelSA, a \underline{D}ocument S\underline{el}ection \underline{S}elf-Updating \underline{A}gent that iteratively refines evidence selection during training and applies a frozen rulebook for one-pass evidence selection at inference. DelSA uses an LLM to characterize candidate evidence, while a verifier-trained reranker balances local relevance against timeline consistency. Finally, the classifier compares the article-implied chronology extracted by the LLM with the evidence-derived chronology to reveal global temporal conflicts. DelSA outperforms existing methods on ChronoFake, while remaining competitive on PolitiFact and GossipCop. 

---
# PsyJev: Psychological Analysis via System-1 Judgment 

**Authors**: Fangyuan Liu, Sirui Zhao, Xiaohao Wang, Guo Cheng, Peng Zhang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.36689)  

**Abstract**: Self-disclosures on the Web support diverse psychological analyses, from recognizing expressed experiences to identifying symptom-relevant information. We study a shared, question-conditioned decision interface for this heterogeneous problem in content analysis. We organize PsyJudgeBench, comprising 13 annotated datasets and 82 questions across six psychological areas, while preserving source-specific criteria and masks for unobserved labels. We introduce PsyJev, a System-1 model that directly produces distributions over answers specified in each request. Joint supervised training establishes the shared judgment function, followed by Judgment-Anchored RLCD (JA-RLCD), which adapts logit-space reward optimization to heterogeneous psychological judgments through balanced supervision, selective retention of correct SFT predictions, and question-group reward optimization. The Qwen3.5-based 2B, 4B, and 9B models achieve 64.33, 66.21, and 67.56 average test F1, respectively. The 9B model exceeds the strongest external comparison by 2.54 points. Controlled 2B comparisons examine the training stages and anchored refinement. Further analyses examine question-conditioned judgment, inference cost, and probability quality, complemented by qualitative comparisons across models. 

---
# Learning Where to Trust: Reliability-Guided Expert Routing and Velocity Gating for 3D Deformable Registration 

**Authors**: Chenyu Guo, Wenjie Dai, Yiqian Chang, Zhiyuan Zhang, Yunfan Wei, Shuhan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.36672)  

**Abstract**: Deformable image registration relies on accurate anatomical correspondences. However, explicitly coupling local matching reliability with spatial context selection and deformation update strength remains insufficiently explored in learning-based registration. This challenge is particularly pronounced in homogeneous, weakly textured, and severely deformed regions, where ambiguous matches may induce erroneous deformation updates. We propose ReliMorph, a reliability-routed symmetric framework for 3D deformable registration that couples correspondence assessment with spatial context selection and update this http URL correlation distributions characterize correspondence ambiguity using shared anatomical features with modality-specific adaptation. These reliability cues guide spatial expert fusion and gate residual velocities during coarse-to-fine refinement. Updated inverse-inconsistency feedback guides routing and gating at finer pyramid levels. To our knowledge, ReliMorph is the first 3D deformable registration framework to jointly condition spatial expert routing and residual velocity gating on correspondence reliability. Extensive experiments demonstrate that ReliMorph consistently outperforms competing methods in anatomical overlap, highlighting its effectiveness across unimodal and multimodal registration tasks. 

---
# The View from Nowhere: Population-Centered Agent Evolution 

**Authors**: Shuo Yan, Ningning Hou, Haoyu He, Haochen Pei, Shang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.36660)  

**Abstract**: Self-improving AI systems can revise the designs that govern their own behavior. Yet existing approaches center improvement on selected parent agents and their evolutionary experience, leaving the broader population largely underused. We identify a substantial gap between population coverage (i.e., the tasks solved by at least one evolved variant) and the performance of the best individual agent. This gap shows that useful capabilities and diagnostic evidence emerge across the population without being consolidated into its leading agents.
These observations motivate the view from nowhere, a principle for agent evolution: an agent should be improved by interpreting its behavior through population-wide patterns of success, failure, and prior self-modification. We instantiate this principle as Population-Centered Evolution (PCE). PCE distils task records and modification histories from the population into a population block, which captures recurring failures, complementary successes, and the effects of previously attempted modifications to guide improvements to a selected parent's harness.
Under matched models, initial harnesses, and evolution schedules, PCE achieves 72.1% success on SWE-bench Verified and 96.7% on Polyglot, improving over a strong group-evolution baseline by 3.2 and 5.9 percentage points. In our evaluation, PCE reduces the gap between population coverage and best-agent performance by one-third on SWE-bench and closes it on Polyglot. These results suggest that treating the population as evidence--not merely as candidates for selection--provides a complementary principle for agent evolution. 

---
# Remembering Is Not Enough: Refining Memories and Learning When to Use Them in Multi-Agent Systems 

**Authors**: Yipeng Hao, Yuxuan Cai, Wenkai Ji, Jie Zhou, Liyang Yu, Qin Chen, Zhiguo Yan, Shulei Zheng, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2610.36652)  

**Abstract**: Memory helps LLM-based multi-agent systems reuse experience, but stored advice can be flawed or applied in the wrong situation. Team-level success or failure alone cannot reliably show whether a memory needs correction or should not have been used. We propose DiscernMem, a training-free framework that improves both memory quality and memory use. To improve quality, Defect-Aware Memory Refinement (DMR) compares team executions with a single memory and with no memory, diagnoses rule defects, and tests repairs on current and earlier tasks to guard against losing previously useful behavior. To guide appropriate use, Boundary-Learning Meta-Memory (BLM) records each memory's past uses and effects, learning when its owner role should apply or avoid it based on task context and incoming messages. Together, these components help teams correct flawed experience while retaining useful rules that were applied incorrectly. Experiments with Qwen3-4B-Instruct-2507 on four benchmarks show that DiscernMem achieves the highest average score on AutoGen, CAMEL, and MacNet, exceeding the strongest reported baselines by 0.51-4.32 points. It leads on code generation and symbolic planning across all three systems, including a PDDL score of 37.71 versus 28.48 on CAMEL. 

---
# The uneven rise of AI-generated web content 

**Authors**: Adam Visokay, Soumyakanti Pan, Molly Griston, Polina Iasakova, Mark Nepf, Denis Peskoff, Riya Sinha, Jikai Sun, Isabelle Thiara, Letao Wang, Yiwei Wu, Lauren Yan, Tyler H. McCormick  

**Link**: [PDF](https://arxiv.org/pdf/2610.36631)  

**Abstract**: In less than five years, generative artificial intelligence (AI) has dramatically reshaped the public-facing web. Using a sample of more than 5 million crawlable webpages, we demonstrate that the share containing AI-generated text has increased precipitously, but unevenly. Our analyses reveal substantial differences in AI composition across language and domain categories (e.g., .edu, .com, or .gov). A higher fraction of English-language webpages contain AI-generated text compared to pages in other languages, and prevalence is substantially lower on .gov and .edu domains compared to .com. These distinctions persist even after accounting for compositional effects. We also find heterogeneity based on a page's position in the connectivity structure of the web. AI-generated text is most common and has increased most rapidly in pages that are the least central in the graph, which drives much of the overall growth in AI-generated text. A slower increase has taken place for more central pages, with the most central tier emerging as the least likely to contain AI-generated text. Methodologically, we use multiple AI detection tools, and generate estimates using a Bayesian hierarchical model that incorporates corrections for the detectors' statistical properties (both false positives and false negatives) and allows for different detector performance across groups, while propagating uncertainty in detector error rates into prevalence estimates. Our work suggests that AI is a prominent, but varied, contributor to public information and internet-based communication. 

---
# Polarity--Time Interaction and Multi-Level Reliability for Event-Based Spacecraft Pose Estimation 

**Authors**: Yonglong Zhang, Zongwu Xie, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36625)  

**Abstract**: Sparse event responses produce uneven keypoint errors whose effects on spacecraft pose depend on correspondence geometry and the reliability of successive observations. We present a reliability-aware framework connecting polarity--time localization, geometric recovery, event--CAD refinement, and trajectory optimization. A three-branch representation combines positive events, negative events, and recent-event timing through cross-branch attention and gating, followed by spatiotemporal aggregation. An ordered semantic-point head estimates relative correspondence precision for geometrically constrained pose recovery. Event-edge and CAD-projection agreement then guide local pose refinement under keypoint-geometry safeguards. At the sequence level, a ten-dimensional descriptor combines heatmap shape, reprojection consistency, and geometric conditioning to set time-aligned observation scales. On the official SPADES challenge test set, the complete system achieves a combined pose score of 0.0474, approximately 5.4\% below PASAN+IRIS. Validation ablations reduce the score from 0.0772 for base localization with uniform PnP to 0.0514 for the complete cascade; on identical single-frame observations, quality weighting reduces the score from 0.0549 with constant-weight optimization to 0.0514. 

---
# God Does Not Play Dice: Enforcing Batch Invariance in Agentic Kernel Generation 

**Authors**: Youwei Bao, Chengpeng Wang, Simin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.36623)  

**Abstract**: Large language model (LLM) serving systems dynamically batch user requests for GPU execution, so the same request may be processed alongside different requests across runs. If the GPU kernels lack batch invariance (BI), this changing batch context can alter the numerical output for an unchanged input, undermining reproducibility. Existing kernel generation agents focus on correctness and efficiency but \emph{overlook BI}. We find 19 confirmed BI violations among 83 batch-applicable released artifacts. To address this limitation, we propose \tool, which combines contract-guided static dependency analysis with dynamic validation of bitwise consistency across batch contexts. \tool is designed as a plug-in component for general kernel agents, allowing their original optimization capabilities to be reused while adding BI enforcement. We integrate it as successive acceptance gates in the optimization loop, rejecting structurally risky or dynamically non-invariant candidates before performance benchmarking and returning feedback to the agent. Experiments across three agent frameworks under baseline and BI-aware generation on real world LLM serving tasks show that \tool improves the BI pass rate from 67.6\% to 91.9\% while retaining 91.8\% of the baseline geometric mean speedup. These results demonstrate that \tool can be incorporated into kernel optimization by agents as an additional reliability objective alongside correctness and speed. 

---
# Learning to Formalize: Bridging Open-Ended Workloads and Fast Decision Models 

**Authors**: Chengxuan Pei, Ziyao Wang, Teng Mei, Zhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.36595)  

**Abstract**: Decision models offer fast choices over sup- plied alternatives, while general-purpose large language models interpret open-ended tasks through flexible generation and reasoning at the cost of sequential decoding. Many work- loads expect a generative backend to determine what can be done and therefore provide no well-defined candidate list. We bridge this in- terface gap with a small candidate formalizer trained on offline teacher-generated lists. The formalizer constructs explicit, validated alter- natives from task context, and a frozen deci- sion model selects an entry. A motivating com- parison with official TypeSafe Jev-1.13.0 on 200 CALVIN-D test episodes reports 11% suc- cess with generic motions versus 46.5% with DeepSeek V4 Flash pose candidates. Sepa- rately, a six-arm privileged-state evaluation on 1,011 CALVIN-D validation annotations finds that a locally deployed decision model can achieve at most 52.6% task success rate with the fine-tuned formalizer, compared with 11.8% without it, and the end-to-end latency is kept to 2.09 s, whereas Qwen3.5-27B is 3.67 s with the exact experiment setting. 

---
# InfraDev: Can Coding Agents Develop AI Infrastructure, and How Can We Tell? 

**Authors**: Yangshen Deng, Jiaqi Ruan, Yeqi Huang, Nick Lau, Guangming Sheng, Luo Mai  

**Link**: [PDF](https://arxiv.org/pdf/2610.36589)  

**Abstract**: Coding agents are beginning to develop the production-grade AI infrastructure that trains and serves large models, yet existing benchmarks cover only single kernels or single-GPU tasks, and grade with tests and prompts that misjudge both correct and incorrect patches. We introduce InfraDev, a benchmark of 36 tasks built from real-world pull requests to Megatron-LM, vLLM, and verl, covering infrastructure development for training, inference, and reinforcement learning. Each task asks an agent to reimplement a real change that spans many files and must run correctly across many GPUs. To grade such changes, we jointly synthesize hidden probes, which enforce the requirements the original tests leave unchecked, and a prompt that states every interface the probes call without revealing the solution. We also design a shared execution environment that reuses dependencies and artifacts across tasks, cutting storage by 52-57% and build time by 13-14% on Megatron-LM and vLLM. Across six frontier models, the best resolves only 38.9% of the tasks; the probes reject 32% of the submissions that pass every official test, and with the original pull request descriptions as prompts, no model resolves a single task. Failure analysis shows that structural and state errors dominate and that agents' own tests rarely check memory, time, or traffic. Turning these findings into an agent skill raises the resolved rate of a frontier model on the Megatron-LM tasks by 10.5%. 

---
# Follow the Herd: A Data-Driven Approach for Actionable Recourse 

**Authors**: Nishtha Madaan, Srikanta Bedathur  

**Link**: [PDF](https://arxiv.org/pdf/2610.36580)  

**Abstract**: As machine learning systems make increasingly greater decisions about people's lives, it is critical to provide users with actionable recourse about how they can achieve their desired outcomes. This problem is currently tackled by relying on expensive human supervision which involves hand-designing a cost function before searching for low-cost actions that achieve the desired outcome. In this paper, we propose \textit{``Data-Driven Action Recourse"} (DDAR), the first approach for providing action recourse without requiring costly supervision from humans about action costs. We build upon the notion that highly likely trajectories in the historical user data should be associated with lower costs. With this, DDAR first builds a dynamics model of historical user trajectories. Under this model, it then searches for trajectories that achieve the desired outcome. In experiments, we find that this simple approach recommends plausible action recourse without requiring human supervision, achieving lower action costs than the existing approaches. 

---
# ADTGuard: Adaptive Depth-Time Safeguard for Streaming Inference in Large Language Models 

**Authors**: Chuming Wei, Churui Yang, Zihan Zhang, Ruien Li, Wusheng Zhang, Yan Zhang, Guangwen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36574)  

**Abstract**: Large Language Models (LLMs) can generate fluent responses to diverse user requests, yet streaming may expose users to harmful content before generation is complete. Existing guardrails face two limitations in this setting. First, post-hoc moderation at response or chunk boundaries delays intervention, risking harmful exposure when content is released before verification. Second, even with token-level detection, coarse response-level supervision does not specify when harmful content begins, while fixed-layer readouts may miss relevant safety signals at other depths. To address these limitations, we introduce ADTGuard, which checks each token before release, progressively refines token-level training targets, and adapts layer selection to the current prefix. ADTGuard models risk across depth and time through Depth-Time Safety Dynamics (DTSD), which integrates safety evidence across layers at each token and uses it to guide risk accumulation over successive tokens. To adapt the choice of observation layers to the current prefix, we propose the Causal Depth Router (CDR). With only response-level labels available, we further develop Risk-Onset Target Refinement (RTR) to progressively construct token-level training targets for both components. Through extensive experiments across multiple safety benchmarks, we demonstrate that ADTGuard consistently achieves strong streaming detection performance, matching or surpassing state-of-the-art safeguards while requiring only 16M trainable parameters and maintaining competitive per-token latency. 

---
# CASEL: Consequence-Aware Selective Execution for Reliable Long-Horizon Tool-Using Agents 

**Authors**: Yanyan Zhang, Fei Shen, Ting Yan, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36570)  

**Abstract**: Tool-using language agents increasingly modify external systems, yet most runtime safeguards allocate scrutiny by generic uncertainty scores or coarse mutation heuristics. We introduce CASEL (Consequence-Aware Selective Execution Layer), a model-agnostic runtime allocation policy that combines calibrated failure probability, state-dependent consequence, reversibility, ambiguity, and evidence freshness to choose, for each proposed tool call, among direct execution, verification, sandboxing, clarification, approval, and abstention by minimizing a per-mode cost. We formalize direct execution as selective prediction and state and test the validity conditions of a finite-sample threshold and a trajectory risk ledger. In a controlled simulation over 20 seeds with 120,000 test actions each, in which every method has the same intervention menu and the same compute cost, CASEL reduces harmful-side-effect trajectories from 14.75 percent for an uncertainty-ranked gate to 7.02 percent and mean normalized harm from 0.184 to 0.123. A gate ranked by the product of failure probability and consequence reaches similar harm (5.82 percent, 0.120) but sends 45 percent more actions to human approval, and removing reversibility-aware routing raises the rate to 15.53 percent. A trajectory-level risk budget, in contrast, increases harm at equal cost (11.82 percent). Intervention effectiveness is assumed, not measured: the results rank allocation rules under the simulator's assumptions and do not establish deployment effectiveness. 

---
# Monitor Accuracy Isn't the Answer: A Decision-Value Framework for Evaluating AI Agent Restart Monitors 

**Authors**: Siddharth Jain, Venkat N. Vedam  

**Link**: [PDF](https://arxiv.org/pdf/2610.36544)  

**Abstract**: Classification accuracy and true-positive and false-positive rates alone do not determine whether restarting an AI agent improves success. Using an exact alarm-to-action identity for an independent clean restart with zero monitoring overhead, we show analytically that a monitor with fixed 90% accuracy can swing from a five-percentage-point loss to a 35-point gain in success rate, depending solely on the replacement attempt's residual-budget success probability. We introduce two techniques for evaluating monitors and progress signals in decision terms. First, a decision-sufficiency criterion, a binary-action specialization of Blackwell's comparison of experiments, identifies which progress representations or context compressions preserve the information needed to choose optimally between the specified actions, independent of trace reconstruction fidelity. Second, a finite-sample, model-free decision audit uses simultaneous binomial intervals over existing agent traces, without additional model calls, to certify a preference for "restart" or "continue" under stated sampling assumptions, or return "abstain" when evidence is insufficient. In a study of 194 externally evaluated model executions on 19 selected SymPy issues from SWE-bench Verified, with a retrospective audit of 130 of these traces, we find a 2.29-point apparent restart gain traced to a single surviving failed attempt, a 42.29-point loss on a second model, no positive replication of the initial gain on prospective task cohorts, and zero of 103 live comparisons meeting the specified certification bar. These results motivate evaluating agent monitors through recovery opportunity, decision value, and statistical support, rather than classification performance alone. 

---
# ATEMPO: Amortized Transfer Entropy Model Per-lag Over-Time 

**Authors**: Emma Andrews, Konstantinos Kontras, Keane Ong, Maarten De Vos, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2610.36536)  

**Abstract**: When one time series informs another, the delay at which that information arrives determines if models can utilize this relationship effectively. A source may be strongly predictive at one delay but uninformative at neighboring lags. Therefore, fixed input windows may obscure this structure by bundling informative and irrelevant history together. Transfer entropy (TE) can recover these delays by measuring how much a source's past tells us about a target beyond the target's own history. However, existing TE estimators require separate fits across source and conditioning depths, with further estimation on local windows to track delays over time. We introduce ATEMPO, a diffusion-based transfer entropy estimator amortized over lags and time. ATEMPO samples source and conditioning depths during training and conditions explicitly on temporal position, allowing a single score network to estimate the full TE surface without refitting. Differencing this surface recovers when information arrives. Across synthetic datasets with ground-truth TE, ATEMPO achieves the lowest mean relative error among baselines, and on datasets with known lags, ATEMPO achieves $0.98\pm0.04$ lag-detection accuracy. ATEMPO requires a single fit across source depths $K=6$ and target depths $L=3$, whereas existing estimators require 18 separate fits to evaluate the same $K \times L$ grid. On three systems with time-varying lag, ATEMPO is the only estimator to maintain perfect lag-detection accuracy across all evaluated window lengths. These recovered delays improve downstream forecasting: ATEMPO-selected inputs reduce MSE on WEATHER-5K by 7.25%, while 120 selected inputs outperform the benchmark's 240-feature default by 4.6%. Across three brain-computer interface tasks, using ATEMPO-selected lags achieves the highest or equal-best $R^2$, improving mean $R^2$ by 4.1% relative to the strongest competing baseline. 

---
# MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics 

**Authors**: Yibo Yang, Zerui Huang, Haomiao Yang, Zixia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36513)  

**Abstract**: As large language models advance rapidly, they are increasingly used to answer medical questions, and many such models are now deployed locally. Once a model is deployed, its behavior is governed mainly by the ethical policy written into its constitution. However, most existing work focuses on factual correctness rather than the trade-off between competing ethical values. Even benchmarks that do evaluate ethics treat it as a fixed property of the model rather than as a variable, so how a constitution reshapes medical-ethics behavior remains largely unexplored. We introduce MedConstBench, a medical-ethics benchmark that treats the deployment constitution as a controlled variable, is organized around Beauchamp's four principles, and is scored by human-calibrated multi-model peer review. It includes MedDilemma, our own dataset of clinical dilemmas, and supports a systematic analysis of how constitutions steer LLMs. Evaluating ten open-source LLMs under four constitutions, we find that the effect of a constitution is dimension-selective and model-dependent, and that no constitution is universally optimal. Motivated by these findings, we propose a do-no-harm adaptive selector that chooses a constitution per model, matching the per-model oracle while guaranteeing a worst-case floor and generalizing to unseen models at low cost. 

---
# AC-JEPA: Anchored Cross-Modal Joint Embedding Predictive Architecture for Visual-to-Symbolic Translation 

**Authors**: Chaofei Hong, Ziyang Chen, Baochen Wang, Christina Yi Jin, Qing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36493)  

**Abstract**: Cross-modal translation requires models to capture shared semantics across modalities while preserving the fine-grained structure required by the target modality. Yet in real-world scientific domains, paired data are often imperfect, while obtaining clean pairs at scale is costly. Supervised fine-tuning (SFT) directly optimizes cross-modal translation through categorical token-level supervision, but this objective is sensitive to errors in the target sequences. JEPA offers a complementary representation-space objective, but most existing formulations focus on prediction within a modality rather than translation across modalities. We introduce Anchored Cross-Modal JEPA (AC-JEPA), an SFT-compatible plug-in that predicts masked intermediate states in both cross-modal directions using paired data. The joint-embedding objective targets cross-modal structure predictable from context, while the causal SFT path retains direct token-level supervision for the target sequence. During self-training, AC-JEPA learns from continuous latent targets derived from imperfect pseudo-labeled pairs, offering intrinsic tolerance to local data errors without promoting pseudo-tokens to hard categorical supervision. Using optical chemical structure recognition as a testbed, AC-JEPA trained on clean pairs achieves 76.07% accuracy on the Simple set of the WildMol benchmark, compared with 67.40% for SFT alone. Using the same pseudo-labeled data for both approaches, AC-JEPA-based self-training further raises accuracy to 80.23%, whereas accuracy under direct SFT drops from 67.40% to 63.47%. These results support SFT-anchored joint-embedding prediction as an effective strategy for cross-modal translation under both curated and imperfect supervision. 

---
# CogArena: Benchmarking Multimodal Agents on Interactive Behavioral Experiments 

**Authors**: Kianté Fernandez, Caitlin Chen, Jialu Zhou, Bartek Sadowski, Anthony C. Miceli, Ian Krajbich  

**Link**: [PDF](https://arxiv.org/pdf/2610.36484)  

**Abstract**: Claims about the cognitive capabilities of large language models are typically tested by translating behavioral paradigms into text prompts, bypassing the visual interfaces and time pressure through which human participants actually engage with experiments. At the same time, AI agents can now complete online behavioral tasks and produce human-like data, which raises concerns that online datasets in the social and behavioral sciences are increasingly open to contamination. We introduce CogArena, a benchmark of ten experimental paradigms from the social and behavioral sciences. Agents complete the experiments through a standard web browser, working with the same visual stimuli and response interfaces that online participants use. A three-level scoring procedure evaluates whether the agent completed the task, how accurately it performed, and whether its data show the behavioral signatures each paradigm reliably produces in people. Across four multimodal models, each run ten times per task, we find that finishing an experiment and reproducing its signatures are different things. Many sessions did not finish, and those that did reproduced some signatures reliably and others not at all. CogArena provides the tasks, scoring code, and a public leaderboard for new submissions. 

---
# When to Intervene? An Empirical Study of Errors and Self-Recovery in LLM Agents 

**Authors**: Chiming Duan, Tong Jia, Yingxi Chen, Minghua He, Lingzhe Zhang, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.36471)  

**Abstract**: LLM agents interact with Web services to perform tasks on users' behalf. Errors during these interactions can disrupt execution and prevent task completion, motivating external intervention. However, agents may self-recover through continued execution, raising the question of when to intervene and when to wait for self-recovery. We present an empirical study of errors and self-recovery in LLM agents. We construct ReInBench, a dataset containing 1,259 agent traces and 1,403 error events across computer-use and Web tasks. Our study examines how often and how quickly agents self-recover, how self-recovery relates to subsequent execution and task completion, and whether self-recovery can be predicted when an error occurs. We find that 725 error events self-recover, often within a few steps, with self-recovery performance differing across Execution LLMs. Self-recovery can also help agents recover from subsequent errors of the same type and support overall task completion. Intervention should therefore prioritize errors unlikely to self-recover, while allowing agents to continue when self-recovery is likely. Building on these empirical findings, we propose ReCoF, a method that predicts whether an error will self-recover to inform whether additional intervention is needed. ReCoF achieves an average relative improvement of 9.02% in Macro-F1 over the strongest baseline on ReInBench. Our code and data can be found at this https URL. 

---
# Correction or Manipulation? Measuring Epistemic Vigilance in LLM Agents Under User Pressure 

**Authors**: Nardine Basta, Venus Haghighi, Dali Kaafar, Amin Beheshti  

**Link**: [PDF](https://arxiv.org/pdf/2610.36453)  

**Abstract**: LLM agents that answer questions and act on behalf of users must decide when to yield to a user who disagrees. Disagreement may correct a genuine error or may be an attempt at manipulation. Prior work shows that models frequently abandon correct answers when challenged by users, yet it remains unclear whether they distinguish correction from manipulation or merely differ in their propensity to concede. We examine this question across seven LLMs from four vendors in two studies, one on the answers agents give and one on the actions they take as customer-support agents whose policy permits account changes only for verified users. Simulated users push toward correct and incorrect answers with graded pressure. Following research on epistemic vigilance, the pressure either appeals to the source, such as claimed authority, or offers content the agent cannot verify, namely a cited source. Drawing on signal detection theory, we measure on a set of benchmark questions how much more readily each agent concedes to correction than to manipulation, separately from its overall propensity to concede. Citations made five of seven models readier to concede and four worse at discriminating, while stronger wording added little. In the agentic study, five models never violated the verification policy, including models that readily abandoned correct answers, so concession on answers did not predict violations of policy. Ministral 3B violated the policy mainly under claimed authority, and Ministral 8B reported changes it had not made. An instruction to revise only on verifiable evidence chiefly shifted propensity to concede rather than improving discrimination. Concession rates alone are therefore misleading, and agents should be evaluated on both their discrimination and the actions they take. 

---
# A Heart Means Recursion: Exploring Visual Meaning and Communication in VLMs 

**Authors**: Alexa R. Tartaglini, Zhengxuan Wu, Thaddäus Wiedemer, Martin Wattenberg, Been Kim, Yuxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.36449)  

**Abstract**: Do vision-language models (VLMs) share our ability to communicate visually? We share preliminary findings, starting from a surprising observation: simple visual symbols like a red heart can strongly influence model behavior on seemingly unrelated tasks, such as steering models towards recursive solutions on simple coding problems. Notably, this steering effect is largely implicit, as models' direct interpretations and thinking traces about visual symbols diverge substantially from their observed behavior. In visual communication between models, this leads to a generation--interpretation gap: while models tend to send shape primitives that they (and humans) find intuitive, these visual cues often fail to steer other models. In further communication games involving abstract concepts and stories, models can creatively use visuals to transmit meaning to one another, although some communications remain non-intuitive to humans. These findings reveal that VLMs might be capable of rich, creative, and visually-grounded communication, but their interpretation and production of visual meaning diverges from human conventions in surprising ways. Together this work highlights both the promise and challenges of visual communication between humans and VLMs. 

---
# From Solver Trajectories to Teaching Trajectories: Cognition-Aligned Reasoning Distillation 

**Authors**: Yashuo Luo, Qianren Mao, Ziqi Qin, Junnan Liu, Haitao Li, Qili Zhang, Yuan Gao, Zhijun Chen, Xu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36445)  

**Abstract**: Knowledge distillation from large reasoning models (LRMs) often uses raw teacher trajectories as supervision. However, this practice creates a mismatch between the goal of distillation and the nature of teacher trajectories. Advanced LRMs typically improve their reasoning capabilities through reward-driven post-training, such as RLHF. This training optimizes them as strong solvers, rather than as teachers that produce reasoning processes smaller models can easily learn. Since smaller models have different "cognitive capacities" compared to their larger teachers, directly imitating raw teacher trajectories can sometimes be ineffective and often requires a substantial number of high-quality samples. To bridge this mismatch, we propose Cognition-Aligned Reasoning Distillation (CARD), an RL-based framework that transforms solver trajectories into teaching trajectories aligned with the student's cognition. Avoiding expensive resampling, our framework directly transforms existing trajectories. We optimize CARD with two reward signals: (1) a Distribution Alignment Reward that matches the student's predictive distribution, and (2) a Logical Coherence Reward that reduces misleading failed reasoning steps. Comprehensive evaluations across multiple student models and settings demonstrate that CARD outperforms distillation from raw teacher trajectories in in-domain settings while achieving substantial gains in out-of-distribution performance. Code is available at this https URL. 

---
# RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG 

**Authors**: Zekai Chen, Hao Zhu, Yu Liu, Xunkai Li, Zejian Kang, Rui Qian, Quanjiang Li, Rong-Hua Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.36444)  

**Abstract**: Web-based applications increasingly rely on understanding long multimodal documents, where answering questions often requires combining information across pages, tables, and figures. Agentic retrieval-augmented generation addresses these tasks through iterative retrieval and reasoning. However, Agentic execution is not always necessary, introducing additional computation and potentially incorrect answers. Our empirical analysis reveals that Agentic execution provides query-dependent benefits, while additional operations may improve, preserve, or degrade answer accuracy. These findings motivate controlling both Agent invocation and subsequent execution. To address these challenges, we propose RouteDRAG, a learning-based framework comprising four components. Unified Document Index organizes multimodal content into a graph-backed index and derives query representations from initial retrieval. Adaptive Path Routing preserves exact answering and predicts the relative benefit of Agent over shorter execution paths. Dynamic Agent Execution maintains an evolving workspace and refines Planner proposals to control tool use and termination. Sequential Policy Learning first trains operation control from observed outcomes and then trains path routing using the resulting Agent's performance. Across two benchmarks, RouteDRAG achieves an average accuracy improvement of 3.47 percentage points over shorter-path-only execution while assigning only 79.3% of queries to Agent. Notably, it also surpasses uniform Agent execution on both benchmarks while requiring fewer answering calls. Our code is now available at this https URL. 

---
# Your Agent Acts Differently When You're Around: How User Presence Changes Agent Behavior in Negotiation 

**Authors**: Jane Pan, You Lyu, Abe Bohan Hou, Valerie Chen, He He  

**Link**: [PDF](https://arxiv.org/pdf/2610.36435)  

**Abstract**: LLM agents increasingly interact with other agents on behalf of users, yet how users in the loop affect their agents' behavior--for better or worse--remains poorly understood. We study this question through the lens of negotiation, where agents must represent users while strategically bargaining with other agents. To isolate the effect of user presence, we develop PRESENT, a controlled evaluation framework that compares negotiation settings with and without an interactive user, while holding the game, objectives, and user preferences fixed across conditions. Across 11.2k runs spanning 7 games and 8 models, we find that user presence significantly changes both outcomes and behavior. Specifically, negotiation parties with a user obtain higher payouts (+13.7% on average) and their agents negotiate more aggressively. Surprisingly, user presence also induces misaligned behavior, both toward counterparties (where its rate roughly doubles) and toward users (where it emerges in 14.3% of games). These misaligned behaviors are uncorrelated with outcomes and persist even when the user sends only templated, neutral messages. These findings expose two blind spots in current evaluation practice: static instructions miss behaviors that emerge only when users are present, and outcome-level metrics cannot detect them. 

---
# LM-DSB: A Lightweight Multi-agent System for Automated Daily Safety Bulletin Generation in Construction 

**Authors**: Xianxiang Zhao, Advik Mehta, Falak Sethi, Brian Gue, Qipei Mei, Lingzi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36433)  

**Abstract**: Construction safety management still struggles to produce daily safety bulletins (DSBs), structured daily communications that identify site-specific hazards, convey safety protocols, and set preventive measures for workers. Manual DSB creation takes 2-3 hours daily and yields inconsistent quality, with 25% of crews never conducting pretask planning (Olson et al., 2016), diverting safety personnel from field supervision. This study proposes the Lightweight Multi-Agent DSB System (LM-DSB), which integrates retrieval-augmented generation (RAG) with a locally deployable Large Language Model (LLM) to automate DSB generation on consumer-grade personal computers. LM-DSB coordinates specialized planner, researcher, and writer agents in a hierarchical architecture, enhanced by domain-specific fine-tuning on construction safety documentation and RAG grounding in 37 health, safety, and environment (HSE) documents. A two-phase validation assessed feasibility. Phase 1 comparative evaluation showed the multi-agent RAG system (Llama 3-8B) outperformed all alternative configurations, with strong lexical alignment and coherence (TF-IDF: 0.70, BLEU: 0.39, ROUGE-1: 0.66) and near-complete document generation (length ratio: 1.01 vs. 0.39 for single-agent RAG). Phase 2 expert evaluation (N=40) confirmed practical acceptability (mean scores 3.50-4.20 on a 5-point scale), particularly in regulatory compliance (4.20) and factual accuracy (4.10). The findings show that architectural design enables lightweight models to handle safety-critical documentation on consumer-grade hardware, advancing technology democratization in construction safety management. Limitations include knowledge base scope, language constraints, and lack of field deployment validation; future work should address multimodal integration and workflow adoption. 

---
# Verifiable Efficiency Limits of Innate LLM Planning in Sliding Puzzles 

**Authors**: Yezhou Cheng, Charles Ling  

**Link**: [PDF](https://arxiv.org/pdf/2610.36415)  

**Abstract**: Tool-assisted planning success can reflect external search and verification, leaving a model's unaided planning ability unresolved. We isolate innate planning through complete plans generated in one response without tools or execution feedback. We propose a verifiable framework that decomposes success-weighted plan efficiency as $\mathrm{SPE}=R\times Q\times E$: execution reliability, planning coordination among legal plans, and action economy among completed plans. Trajectory replay and certified optima verify these factors; the joint score includes unsuccessful attempts. We evaluate 8,400 full-plan outputs from seven model conditions on $4\times4$ sliding puzzles, controlling exact horizon and obstacle count in the primary cohort. Models that complete every trial still become less economical at longer horizons, while higher efficiency among completed plans can coexist with poor goal attainment. SPE separates the overall capability hierarchy, while its factors distinguish coordination failure from inefficient completion. The framework makes long-horizon limitations in unaided planning quantitatively verifiable. 

---
# kw-bench: Breaking Frontier Harnesses on Real-World Tasks 

**Authors**: George Juraj Salapa  

**Link**: [PDF](https://arxiv.org/pdf/2610.36405)  

**Abstract**: Each new generation of frontier model pushes the edge of what is possible. Inside a harness -- a loop, a shell, a filesystem, the ability to write and run its own code -- a model can take on increasingly difficult work.
A growing body of work treats the coding-agent harness as infrastructure for knowledge work rather than a coding tool, and a new wave of benchmarks tests it there. Most of them are generative: build a financial model, a memo, a spreadsheet from a brief, judged by a rubric or by another model. Few test the expensive part of knowledge work, which is investigative. In audit, tax, due diligence, regulatory reporting, claims or any dispute over numbers, the answer already exists -- hidden in incomplete and inconsistent evidence spread across systems -- and has to be recovered and defended to the cent.
We introduce kw-bench, a benchmark built from real engagements specialists described to us. Its first task family is a five-year VAT reconstruction across three generations of SAP -- legacy ECC, a partial S/4HANA migration and a consultant's redesign -- delivered as raw system extracts: page headers, local number formats, partial link tables, no documentation of how anything connects. Grading is deterministic, with no LLM judge.
We ran four harnesses -- Claude Code, Codex, OpenCode and micro-cc -- on the same model, DeepSeek v4.1 Flash, for two hours. The two independent, model-agnostic harnesses delivered (OpenCode 0.542, micro-cc 0.528); the two lab harnesses, away from their home model, delivered nothing (0.00 each). What defeated them was inertia: keeping many weak signals alive across a long session. On a frontier model the task collapses. micro-cc and Claude Code on Claude Opus 5.5 scored 0.9986 and 1.000, in 22 and 25 minutes. A weeks-long engagement now costs minutes and a few dollars; below the frontier, the harness decides how far a model gets. 

---
# MAREval: A Multi-Agent Framework for Evaluating Natural Language Recommendation Explanations 

**Authors**: Reza Yousefi Maragheh, Jayesh Uddhav Kudase, Aysenur Inan, Ramin Giahi, Kai Zhao, Parth Parikh, Jianpeng Xu, Jason Cho, Evren Korpeoglu, Sushant Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.36389)  

**Abstract**: Evaluating natural language explanations in recommender systems is essential for fostering user trust, transparency, and engagement. However, existing evaluation approaches like human evaluations, while accurate, are resource-intensive and impractical at the scale required by modern recommendation platforms. Also, automated methods using single-agent LLMs suffer from prompt sensitivity and inconsistent outputs. To address these challenges, we propose MAREval, a structured multi-agent framework for evaluating recommendation explanations using large language models. MAREval orchestrates (i) a planner agent that uses a novel Chain of Debate (CoD) prompting strategy to coordinate agent roles and enforce logically consistent evaluation plans; (ii) a moderator agent that regulates discussions by mitigating prompt drift; and (iii) an arbitrator agent that aggregates outputs from multiple evaluation rounds and (iv) a Monte Carlo sampling method, improving robustness and alignment with human judgment. We conduct comprehensive evaluations on both public (TopicalChat) and proprietary recommendation datasets, demonstrating that MAREval outperforms state-of-the-art baselines. Comprehensive experiments on a public benchmark and a proprietary e-commerce dataset show that MAREval improves alignment with human judgments over strong singleand multi-agent baselines. Stability analyses indicate substantially lower variability across repeated trials. In a large human-annotation gate, MAREval meets production quality thresholds where prior evaluators fall short, and online A/B testing demonstrates statistically significant improvements in engagement and revenue metrics. These results establish MAREval as a scalable and reliable solution for human-aligned evaluation of recommendation explanations in real-world systems 

---
# LoRA on a Budget: Function-Aware Post-Hoc Compression for Adapter Fleets 

**Authors**: Shenghui Li, Thiemo voigt  

**Link**: [PDF](https://arxiv.org/pdf/2610.36386)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a key mechanism for efficiently specializing foundation models into many task- and user-specific variants. Serving many LoRA adapters under a shared resource budget turns post-hoc compression into a two-level allocation problem: rank must be distributed across layers within each adapter and across adapters within the fleet. Although trained LoRAs contain substantial spectral redundancy, spectral reconstruction alone does not determine how much compression an adapter can functionally tolerate. We formulate fleet compression through a functional rank-distortion frontier for each adapter, followed by fleet-level minimax allocation under a shared rank budget. Given exact frontiers, this decomposition is equivalent to joint optimization over all layer ranks. We introduce Functional Rank Allocation (FRA), an unlabeled method for constructing practical frontiers. FRA measures isolated module truncations end to end, uses these measurements as an additive search surrogate solved by dynamic programming, remeasures every complete candidate, and takes the lower-risk envelope with a complementary spectral proposal before exact minimax allocation over the constructed frontiers. Across 78 adapters from three public fleets, FRA improves worst-case and tail utility over uniform spectral compression in constrained-budget regimes. Fixed-budget probes show that isolated costs rank candidates well, but their calibration shifts across proposal shapes; end-to-end verification and the spectral proposal therefore limit functional-search error. 

---
# WAKMan: Workload Aware KVCache Manager 

**Authors**: Suhas Kumar Morisetty, Vaibhavi Lokegaonkar, Sachin Mathew Varghese, Wesley Harris  

**Link**: [PDF](https://arxiv.org/pdf/2610.36355)  

**Abstract**: Prefix caching enables a Large Language Model (LLM) servers to reuse the Key-Value (KV) states and accelerate prefill computation stage for requests that share a prefix with earlier prompts, thus reducing latency and serving cost. With limited memory resource, the KV cache eviction policy becomes a decisive factor in determining the overall benefits of prefix caching. Production traffic may have single-turn, multi-turn, agentic, or a mixture of any combination of these request types. Our findings indicate that agentic steps return within seconds and favor recency, while popular prompts and long conversations return after many other requests and favor frequency. Across 20 mixes, no single policy is always best: vLLM's default strategy LRU provides 12% higher cache hits as compared to the frequency-oriented S3FIFO on agentic-heavy mixes but trails it by up to 22% elsewhere, and even a single workload parameter can flip the winner even for any particular mix. We present WAKMan, a workload-aware KV cache manager that picks its eviction policy from live traffic without knowing request types or reading prompts. It improves hit rate over LRU by 5 percentage points on average, with the lowest mean and worst-case optimality gap of all policies. 

---
# Executable Archaeology of Programs That Model Programming: Reanimating the Heuristic Compiler (1961) and KBEmacs (1984) 

**Authors**: Jeff Shrager, Kent M. Pitman, Mia Golec  

**Link**: [PDF](https://arxiv.org/pdf/2610.36341)  

**Abstract**: Early AI programs were often also theories of how people think, and programs that wrote programs were, in part, theories of how people program. Here we report on the reanimation of two such programs that offer two seemingly different theories of programming and were developed for quite different purposes. The first, Herbert Simon's Heuristic Compiler (1961), was an experimental model of programming as problem-solving, applying the means-ends analysis methods of the General Problem Solver to investigate the nature of the programming task. The second, KBEmacs (1984), was a working prototype of an intelligent programming assistant developed by MIT's Programmer's Apprentice project, built on the view that expert programmers recognize and combine familiar patterns, or "cliches." It let a programmer build up programs by combining cliches through a stylized English command language. We approach both through the methodology of Executable Archaeology, the study of historical computational systems by running their original code, a process we call "reanimation." Our reanimation of the Heuristic Compiler reproduces the compiled code appearing in Simon's 1963 paper, and our reanimation of KBEmacs reproduces the SQRT sequence of its December 1984 self-test. Examined side by side, the two theories turn out to have much in common: the Heuristic Compiler also works by adapting existing routines, and Simon later argued that search alone requires a large body of programming knowledge. To highlight the reflexivity of this project, we deliberately used a modern AI-assisted programming tool, Anthropic's Claude Code (Opus 5.5), to carry out much of the detailed reanimation work, building on a significant base of existing software. The explicit, executable theories of programming that both systems embodied are largely absent from today's far more capable AI coding assistants. 

---
# OA-RAG: Ontology-based Agentic RAG for Faithful LLM Reasoning 

**Authors**: Hui Yang, Jiaoyan Chen, Yongsheng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.36335)  

**Abstract**: Large language models (LLMs) perform well across many real-world tasks but remain prone to hallucination, limiting their reliability in domains such as biomedicine and healthcare, where answers must be grounded in established knowledge. Retrieval-augmented generation (RAG) mitigates this problem by incorporating external information, but often overlooks logical relationships and relies heavily on the LLM's own reasoning, which can compromise answer faithfulness. We propose OA-RAG, an agent that integrates formal reasoning with LLM reasoning, drawing on both the logical structure and informal annotations of domain ontologies. For each answer, OA-RAG reports the ontology observations used as evidence and a model-assessed grounding level (entailed, supported, related, or unrelated) that characterises the strength of that evidence. Across benchmarks derived from three large-scale ontologies and six real-world medical question-answering tasks, OA-RAG achieves the best overall performance among the evaluated ontology-fine-tuned and popular RAG methods, with particularly strong results on reasoning-intensive tasks. The reported grounding levels also offer a useful indication of answer reliability, with stronger grounding generally associated with higher accuracy. Finally, our results demonstrate the value of ontologies as knowledge sources: OA-RAG is competitive with, and sometimes outperforms, domain-specific RAG systems that draw on sources such as textbooks. 

---
# Plenty of Room in the Middle: Revealing Opportunities on a GPU Cluster via Fine-Grained Monitoring 

**Authors**: Andrew Sheinberg, Samyak Gupta, Kai Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.36329)  

**Abstract**: GPU clusters have become essential computing resources for AI and scientific research in academia and industry. However, their high cost and growing demand often result in long queueing delays. While previous monitoring efforts have operated at second- or minute-level frequencies, they primarily informed job-level scheduling for clusters and lacked the detail needed to optimize GPU-level utilization.
This paper presents an in-depth study of a university research GPU cluster, analyzing resource usage at 100 ms intervals. We have developed a lightweight monitoring tool that leverages Nvidia's DCGM library to capture fine-grained "compute utilization" metrics within GPUs. Deployed on a cluster with 89 nodes and 319 GPUs, our tool collected data over several months. This paper focuses on a 10-day period with 7,399 jobs submitted by 150 users from 12 departments. Our study reveals three key findings: (1) the GPU cluster exhibits significant underutilization, with over 70% of jobs using less than 20% of GPU memory and 50% averaging below 5% Streaming Multiprocessor (SM) utilization; (2) 80% of jobs have a drastic mismatch between their requested and actual durations, significantly limiting the scheduler's effectiveness; and (3) the multi-GPU jobs we examined show highly periodic communication patterns. These findings suggest there is plenty of room in the middle: substantial utilization gains are available through better job scheduling and GPU multitasking, independent of hardware or application improvements.
Our monitoring solution is available as open-source software at this https URL. 

---
# How Stable Are LLM-Generated Proofs Under Program Refactoring? A Stress Test for Verifiable Code Generation 

**Authors**: Aravind Mandiga, Guoming Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.36328)  

**Abstract**: Benchmarks for verifiable code generation usually test a proof against one fixed implementation. We ask whether the same proof remains valid after a program refactoring that preserves behavior. We introduce RefactorProof, which derives 458 Lean-certified variants from the 185 VERINA tasks that compile in our pinned environment while holding the specification and evaluated proof fixed byte-for-byte. VERINA supplies reference proofs for 46 tasks, all from its basic split; the primary model-versus-reference comparisons further match on 21-25 short, non-recursive tasks. Structural refactorings expose substantial brittleness across proof sources: task-normalized structural PSR is 4.3% for the supplied references and 13.3-26.8% for the specialized provers. On the matched tasks, specialized prover proofs nevertheless have higher overall unchanged-proof survival than the supplied reference artifacts: +15.6 points for DeepSeek-Prover-V2-7B, +18.8 for Goedel-Prover-V2-32B, and +20.3 for Goedel-Prover-V2-8B; all three 95% CIs exclude zero. The ordering persists after excluding universally survived T1 variants, under family-balanced aggregation, and under alternative deployable valid-proof selection rules. However, unchanged-proof failure is not the same as high repair cost: predefined one-identifier edits repair 51/54 eligible reference structural failures and 58/85 eligible prover failures. RefactorProof therefore measures unchanged-proof stability under program refactoring, not the cost of maintaining a proof after it breaks. 

---
# Ontology-guided multi-agent reasoning for explainable differential diagnosis 

**Authors**: Roxana Ramona Szomiu, George Ursu, Adrian Groza  

**Link**: [PDF](https://arxiv.org/pdf/2610.36325)  

**Abstract**: Effective differential diagnosis depends on well structured and unambiguous patient case descriptions, since incomplete or ambiguous clinical information readily leads to diagnostic errors and inconsistent reasoning. Large Language Models (LLMs) have recently shown promising capabilities in medical question answering and clinical reasoning; however, even when achieving competitive performance on medical benchmarks, they often rely on pattern recognition learned from training data rather than grounded clinical evidence, and can generate plausible yet unsupported diagnostic conclusions. Their reasoning remains largely opaque, making it difficult for clinicians to assess the evidence and rationale behind a given recommendation. To address this gap, we propose an ontology-guided, multiagent differential diagnosis framework that replicates the iterative reasoning process of physicians. The framework grounds diagnostic inference in a structured, expert-curated medical knowledge base, enabling interpretation of symptoms, clinical findings, or disease relationships, while decomposing diagnosis into explicit, auditable reasoning steps. This ontology-based grounding constrains inference to evidence-backed associations, reducing hallucination and improving diagnostic transparency, traceability, and consistency. The system then identifies the most discriminative missing symptoms and generates targeted follow-up questions to reduce uncertainty through a process that mirrors to clinical inquiry. Together, these components improve the reliability, explainability, and diagnostic accuracy of AI-assisted differential diagnosis. The source code is publicly available. 

---
# When Retrieval Is Not Enough: Obsolete Facts in Agent Memory 

**Authors**: Oleeviya Babu Poikarayil, Tony Basil Babu, Abdulrahman Nahhas, Het Darshan Mehta, Jawid Ahmad Baktash, Ernesto William De Luca, Klaus Turowski  

**Link**: [PDF](https://arxiv.org/pdf/2610.36308)  

**Abstract**: Memory-agent benchmarks often collapse retrieval and answer generation into one score, making it hard to tell whether a failure comes from missing an update or from not using an update that was already retrieved. We examine this distinction on the single-hop FactConsolidation task of MemoryAgentBench. With Qwen2.5-7B-Instruct on 6K-history BM25 accuracy drops from 45% to 32% when the retrieved evidence budget grows from 64 to 512 tokens, even though a gold-assisted post-hoc audit finds the newest statement in every retrieved set. Most of the errors occur on questions that also contain an obsolete statement version. Under a frozen prompt and equal four-statement evidence size, replacing one unrelated statement with the obsolete competing statement reduces accuracy from 100% to 44.6% for Qwen2.5 and from 100% to 87.8% for Qwen3-14B, with all 50 harmful paired flips classified as stale. A 32K-history replication likewise favors BM25-64 over BM25-512 (69% versus 58%). These results suggest that retrieving the correct fact is not enough. An obsolete competing statement can be more harmful than an unrelated piece of evidence of the same size. 

---
# Spiking Neural Networks Are Harder to Steal: A Benchmark Study of Model Stealing Attacks 

**Authors**: Ayana Moshruba, Maryam Parsa  

**Link**: [PDF](https://arxiv.org/pdf/2610.36298)  

**Abstract**: Machine learning models deployed behind query interfaces expose more than their predictions, because an adversary can use those predictions to rebuild the model itself. Model stealing attacks (MSAs) exploit this exposure by training a surrogate model that replicates the decision behavior of a target through black-box queries alone, without access to its parameters or training data. These attacks are well characterized for conventional artificial neural networks (ANNs), yet spiking neural networks (SNNs) process information in a fundamentally different way. SNNs communicate through sparse, discrete spikes, and their outputs depend on membrane potentials that evolve over time rather than on a static mapping from input to output. This temporal mode of computation suggests that the decision behavior of an SNN may be harder to reconstruct from its responses. This work tests that possibility by benchmarking four representative MSAs, namely the Papernot, Tramèr, T-RND, and color perturbation attacks, against SNNs and their ANN counterparts on MNIST, Fashion-MNIST, and CIFAR-10. Across every attack and dataset, SNNs prove consistently more difficult to steal. Surrogates of SNNs achieve 10% to 20% lower agreement with the target model and 15% to 25% lower adversarial transferability than surrogates of ANNs. We attribute this resistance to the way SNNs compute, as their spike-based and time-dependent dynamics yield decision boundaries that ANN-based surrogates struggle to approximate, even under large query budgets. These findings extend the known resilience of SNNs beyond attacks on training data to the replication of model functionality, strengthening the case for neuromorphic architectures in security-sensitive applications. 

---
# A Smart Instant Book Filter Model based on Dual-Objective Optimization 

**Authors**: Yi Hou, Li Fan, Huizhong Duan  

**Link**: [PDF](https://arxiv.org/pdf/2610.36287)  

**Abstract**: Hospitality service in online marketplaces has become increasingly popular over the past few years, not only because it helps people easily rent short-term lodging but also because it empowers travelers to connect with locals and feel belonging. A typical pattern of these industries is a central platform mediating the economic interactions between guests and hosts, where a successful booking experience is the key to user satisfaction. However, one of the biggest challenges is the possible delay of response and chance of rejection when guests try to book a request-to-book listing. Previously, we enabled the instant book feature to allow hosts to instantly accept qualified guests and guests to confirm bookings immediately. Yet the symptom has not been alleviated given mixed request-to-book and instant book listing supply in search results. One possible solution is to promote all instant book listings in search results; however, booking rate will be damaged due to some improperly promoted low quality listings.
In this paper, we present a novel smart instant book filter system that applies the instant book filter intelligently and shifts request-to-book to instant book without hurting booking rate. Compared with traditional two-sided markets research that targets finding strategies to maximize a single overall social welfare goal, our proposed system aims to maximize the instant book rate with the constraint that booking rate should not decrease, forming a dual-objective optimization problem.
We show A/B testing results for our system exposed to global users, demonstrating that our model performs better than previous human-expert-tuned models in terms of both instant book rate and overall booking rate. We also achieved improved metrics such as the host instant book adoption rate. The proposed system has been deployed to production and is serving millions of users every day. 

---
# BehaveRec: When Does Long-Term Behavioral History Help Sequential Recommendation? A Study with Session, History, and Community Signals 

**Authors**: Aysenur Inan, Reza Yousefi Maragheh, Jayesh Uddhav Kudase, Selene Xu, Kaushiki Nag, Jianpeng Xu, Sushant Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.36239)  

**Abstract**: Sequential recommendation models typically predict a user's next action from recent session interactions, which limits personalization when short-term signals are weak or ambiguous. Rather than assuming that incorporating longer-term history and community signals is universally beneficial, we study when it actually helps. We propose BehaveRec, a three-signal framework combining a causal session transformer, a bidirectional User History Encoder over a 90-day window, and a GraphSAGE module over a user-user graph built from behavioral embedding similarity rather than item co-occurrence; a learned gate blends the three signals per user. On an industrial dataset of 137M sequences from 12.6M users, BehaveRec improves R@5 by +3.2% over SASRec, with a larger-capacity SASRec baseline (more parameters than BehaveRec) still trailing by +2.5%, ruling out capacity as the explanation. More importantly, a user-level stratification by history length reveals a clear dose-response relationship: BehaveRec's advantage is negligible below roughly 10 interactions and grows monotonically with history richness, reaching +0.017 R@5 for users with 101+ events. This pattern is corroborated across one industrial and three public datasets spanning history-rich to history-sparse regimes, where session-only models remain preferable. These headline results, and the resulting deployment threshold, are established for category-level intent prediction; a preliminary, single-run check on the full item-level vocabulary yields a directionally mixed result and is scoped as an open question rather than a claim. We use these results to derive a practical deployment rule -- roughly 20 interactions per user -- for when long-term history and behavioral-graph signals are worth their added complexity. Despite this added complexity, BehaveRec remains lightweight: it uses 5.67M parameters and trains on a single GPU in under two days. 

---
# MARQ: A Modular, LLM-Agnostic Requirement Quality Assessor Infused with Systems Engineering Knowledge 

**Authors**: Jannatul Shefa, Alejandro Salado, Paul Wach, Taylan G. Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36234)  

**Abstract**: Requirements engineering (RE) is foundational to systems engineering (SE), as requirement defects can propagate downstream and adversely affect development outcomes. Although large language models (LLMs) are promising for requirement quality assessment, prior work shows that off-the-shelf models miss a large share of expert-identified defects, perform unevenly across quality criteria, and vary across executions. This study investigates whether infusing expert-derived SE knowledge can improve LLM-based requirement quality evaluation beyond off-the-shelf capability. We present MARQ, a modular, LLM-agnostic prototype that supplements LLMs with a criterion-specific knowledge stack operationalizing INCOSE requirement quality criteria and evaluated against off-the-shelf baselines on an expert-labeled NASA requirement set using Claude Opus 5, GPT-5.6 Sol, and the open-weight Gemma 3 12B, with 100 executions per model. MARQ achieves median F1 scores of 0.80 for Claude Opus 5, 0.62 for GPT-5.6 Sol, and 0.53 for Gemma 3 12B, compared with off-the-shelf baselines of 0.55, 0.52, and 0.18, respectively. However, the nature of these gains is model-dependent. Claude Opus 5 improves defect detection and false-alarm control; GPT-5.6 Sol gains detection and better execution-level consistency at the cost of more false alarms; and Gemma 3 12B reaches the performance of off-the-shelf proprietary models, although it remains highly variable. Criterion-level analysis shows gains concentrated in Correctness and Conformance, trade-offs between misses and false alarms within individual criteria, and model-dependent responses across several criteria. These findings position SE knowledge infusion as a promising model-agnostic pathway for strengthening trustworthy LLM-based RE support while underscoring the need for criterion-specific interventions and human-in-the-loop review. 

---
# Skill Dreaming: Evolving Repository Skills through Pull Request Streams 

**Authors**: Xinyuan Song, Hasibul Haque, Jay Somasundaram, Ryan Turner, Bowen Zhu, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.36188)  

**Abstract**: Large language model (LLM) agents increasingly rely on reusable skills for complex software engineering tasks, making skill libraries essential for repository development and maintenance. However, repository-specific skills must evolve with changing code structures, implementation conventions, and maintenance requirements. Pull requests (PRs) provide a natural source of such knowledge by recording how repositories evolve through implementation and review. Meanwhile, new PRs must follow repository-specific implementation patterns, modify related components consistently, and ensure that existing tests continue to pass while adding coverage for new functionality. This creates a mutually reinforcing cycle: PRs provide knowledge for updating skills, while these skills improve future PRs, maintain repositories, and support further skill evolution. To enable this co-evolution of PRs, repositories, and skills, we propose Skill Dreaming, a framework that learns from PR streams to guide continuous repository development and maintenance. Given a repository and its PR history, Skill Dreaming builds a repository-specific skill library by extracting executable knowledge from validated PRs and continuously refines the skills as new PR information becomes available. These skills guide coding agents not only in implementing requested features but also in maintaining associated tests, documentation, compatibility, and repository conventions. By integrating PR analysis, skill management, code execution, and continuous integration, Skill Dreaming provides an end-to-end workflow for applying learned skills to repository changes and incorporating subsequent PR outcomes into future skill updates. 

---
# Learning Human Sports Motion Feasibility via Quantile Regression on Tracking Data 

**Authors**: Marc Gutiérrez-Pérez, Antonio Agudo  

**Link**: [PDF](https://arxiv.org/pdf/2610.36183)  

**Abstract**: We study time-to-arrival (TTA) in sports: the shortest plausible time a player needs to reach an arbitrary location on the pitch given their current kinematic state, estimated as the lower envelope of observed locomotion times. Rather than modeling space ownership through position density estimation or closed-form physics-based models, we cast TTA as a conditional quantile regression problem, estimating the lower quantile of locomotion time so the model learns a player's lower-envelope arrival time. We train the model on optical tracking data from all 64 matches of the PFF FC FIFA World Cup 2022 dataset, and evaluate a lightweight multilayer perceptron against six player-motion models from the literature, re-expressed as TTA predictors, and a learned reachable- hull baseline. Across our evaluation metrics, the MLP consistently outperforms every physics-based baseline. To probe the generality of the learned locomotion model, we further evaluate it on basketball tracking data and a single-game NFL tracking dataset, analyzing the transferability of player motion dynamics across sports. Our project is available at this https URL. 

---
# Don't Pick a Winner: Knowledge Conflicts Need a Prior, Not a Rule 

**Authors**: Minh Pham Van, Cong Tran  

**Link**: [PDF](https://arxiv.org/pdf/2610.36169)  

**Abstract**: A retrieval-augmented language model answers from two sources, the retrieved context and its own knowledge, and the two disagree whenever the context is incomplete or wrong. Existing methods settle the disagreement with a rule fixed at design time: follow the context, amplify it in proportion to how far it moves the model, or trust the model's memory. Each rule is confidently wrong in some deployment. In a decision-theoretic model of two sources, the score of any fixed rule is linear in the share of reliable contexts while the best attainable score is convex, so every fixed rule is beaten wherever it is not exactly optimal; and a symmetric divergence between the two sources, the signal adaptive decoding uses, carries no information about which one is right. Resolving a conflict therefore needs two inputs that no existing method keeps: evidence about each source separately, and a prior on how often the context is right. We test these claims on GRACE, a benchmark that intervenes on the knowledge-graph context of a fixed question, so that what the model knows, what the context says and what is true are measured separately. There, every baseline from four families is strong on right contexts or on wrong ones, never on both. We then build the two inputs: the model states its knowledge as notes, a small policy voices each note with a calibrated confidence, a reader answers in two voices, and a selector commits or abstains under a prior $\pi$ applied as an exact prior shift. With $\pi$ set to the true share of right contexts, the system is above the best baseline at every mixture of GRACE contexts, by up to 26 points. Its overall score is 45.3 and 35.0 (Qwen3.5-35B-A3B, Llama-3.1-8B), against 29.4 and 27.5 for the best baseline. On preregistered ClashEval it beats same-backbone CK-PLUG and TCR by 12 and 26 points. Where the model lacks the knowledge, the gains vanish, as the theory predicts. 

---
# From Structural Feedback to Prompt Policies: Learning Faithful Text-to-Image Prompt Editors 

**Authors**: Wentao Ye, Yali Ye, Zhiqing Xiao, Ru Peng, Xinpeng Ti, Liyao Li, Zhanming Shen, Peng Lu, Jie Zhang, Haobo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36146)  

**Abstract**: Prompt optimization should improve the image a user receives without changing the scene the user requested. Yet stylistic rewriting can weaken object attributes and relations even when global text--image similarity remains high. We propose StruPO, a structure-aware prompt editor that uses a calibrated scene graph as a shared representation for policy conditioning and image-grounded supervision. The graph exposes what should be preserved, while deletion, reordering, and insertion provide a token-aligned way to modify the prompt. This turns structural feedback from an external critique into a training signal for a reusable editing policy. Learning from that signal requires handling both uncertain grounding and stochastic image generation. StruPO retains grounding confidence through smooth reward shaping and averages multiple images for each edit before computing group-relative advantages. Experiments span three disjoint 2K-prompt test sets and four frozen generation backbones, with comparisons to learned prompt optimizers, iterative structural feedback, diffusion-model fine-tuning, and inference-time attention control. StruPO improves the faithfulness--aesthetics trade-off, with consistent gains in relational fidelity. Independent automatic evaluators and a 15-annotator study provide complementary evidence, and human preference margins increase with relation complexity. Controlled ablations connect these gains to graph conditioning, local editing, and image-grounded learning. The resulting policy produces reusable optimized text in one pass after graph calibration, without an iterative critique--rewrite loop. 

---
# Why and How to Do Science When AI Is Better at It: A Framework Grounded in Human Understanding and Direction 

**Authors**: Hoda Heidari, Aydin Mohseni  

**Link**: [PDF](https://arxiv.org/pdf/2610.36116)  

**Abstract**: As AI begins to match or exceed human researchers at core scientific tasks, two questions become pressing: \emph{why} should humans keep doing science, and \emph{how} should they use AI when they do? This paper argues that the second can be answered only by answering the first. It holds that one indispensable aim of science is \emph{durable, collective human understanding of the world, pursued in service of human ends and values}. From this principle it derives a two-part test for when a research task can be delegated to AI. The first part, \emph{human understanding}, requires that the scientific community remain able to explain, defend and contest the resulting claims, and retain the skills to do so. The second, \emph{human direction}, requires that judgments about which questions matter and what findings mean stay with accountable humans. Applying the test to each phase of research in the Contributor Roles Taxonomy (CRediT) yields recommendations that fields can tailor to their questions and methods. The framework differs from existing guidance in three ways. It rests on an account of what science is for rather than on integrity norms. It holds even if AI outperforms human researchers, rather than depending on AI's current limitations. And it judges each use of AI by its effects on human understanding and direction rather than by the type of tool, across every phase of research, including peer review and the training of new researchers. It closes with the reforms needed to keep science aimed at human understanding and direction. 

---
# CohortFlow: A Steerable Clinical Cohort Curation Agent with Localized Semantic Correction 

**Authors**: Neil Chen, Adrian Serapio, Hanxue Gu, Evan Lee, Kang Wang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36100)  

**Abstract**: Language model agents for clinical cohort curation should accommodate evolving eligibility criteria without repeatedly processing unchanged requirements. We introduce CohortFlow, a steerable agent that represents natural language corrections as localized updates to a structured cohort specification and reconciles them with a persistent execution graph. Explicitly recorded semantic dependencies enable affected computations to be selectively invalidated while preserving valid prior results. On 100 scenarios derived from clinical-trial eligibility criteria, comprising 126 correction turns, localized steering achieves 77.0% requirement micro-F1 versus 76.4% for full regeneration using the same 8B model, a small numerical advantage that is not statistically significant ($p=0.493$). It reduces mean correction latency by 64.8% after completion and 44.3% during execution, with 38.1% fewer model calls. In a separate proof-of-concept evaluation over MIMIC, all 13 cohort states execute successfully, while correction trajectories demonstrate selective preservation of unaffected computations and recomputation of changed or dependent nodes. Together, these results show that localized semantic steering can substantially reduce correction latency and model calls while maintaining strong requirement-level accuracy. 

---
# The Trinity:Capability-Driven Recursive Self-Improvement for Multimodal Agents 

**Authors**: Weidong Tang, Ruiheng Wang, Jingxuan Zhang, Hexin Wang, Haoyuan Chen, Zhengfeiyang Zhang, Yixing Ma, Zichen Jiang, Ziyu Guo, Haodong Li, Zhengbo Jiao, Zhiqing Cui, Xinxiang Yin, Shuo Liang, Zhenglin Wan, Hongzhan Lin, Wangbo Zhao Kaipeng Zhang, Pengfei Zhou, Odin Zhang, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2610.36077)  

**Abstract**: Multimodal intelligence can be viewed as human-guided co-evolution between worlds and agents through interaction. However, scaling this process faces two bottlenecks: constructing heterogeneous, verifiable environments is costly, and manual Harness iteration cannot keep pace with growing task diversity and complexity. A key insight is that tasks with different modalities and surface forms can share the same underlying capability requirements. We therefore introduce The Trinity, a recursive self-improvement framework coupling world generation with Harness evolution through three roles: Creator, Explorer, and Evolver. Drawing on verified execution trajectories, Evolver shifts the basis of task organization from surface categories to shared requirements, reorganizing the original task tree into a hierarchical capability tree. We introduce Capability-Driven Real2Sim2Real: for heterogeneous tasks at the same node, Creator abstracts their common requirements into simplified, verifiable simulated worlds for capability learning, while evaluation on original tasks measures transfer. Along this capability tree, Explorer progressively evolves reusable capability modules through selective inheritance, forming a Harness tree whose validated updates accumulate across iterations within a fixed topology. The resulting Harnesses specialize in different capabilities, motivating a Capability-Aware Harness Router that uses public task information to select compatible module configurations from the tree. Our evolved Harnesses improve mean performance by 34 points across 28 training benchmarks and 26 points across 18 unseen benchmarks, including two real-world embodied tasks. 

---
# Enhancing Deep Learning Model Performance through Data Generation Strategies: Classification of Olive Tree Diseases 

**Authors**: Hicham El Akhal, Abderrahim Waga, Aissa Ben Yahya, Abdelbaki El Belrhiti El Alaoui  

**Link**: [PDF](https://arxiv.org/pdf/2610.35984)  

**Abstract**: Deep learning models are highly effective in a wide range of domains, particu larly in plant disease diagnosis. However, their performance is often limited by challenges such as insufficient data and class imbalance. In our work, we propose a comprehensive data generation strategy to overcome these limitations. First, we apply various data generation techniques, including traditional augmentation methods, CutMix, MixUp, and synthetic image generation using Deep Convolu tional Generative Adversarial Networks (DCGANs), to create new data samples. Second, we exploit this generated data in three scenarios: (S1) balancing class distribution in the original dataset, (S2) increasing the number of samples per class, and (S3) expanding the dataset while maintaining balanced classes. Third, we evaluate the impact of these data distributions (S1, S2, S3) on model perfor mance using the EfficientNetV2B0 architecture with pre-trained and fine-tuned weights. The experimental results show a substantial improvement in classifica tion accuracy, emphasizing the critical role of data diversity and class balance. These findings highlight the effectiveness and adaptability of the proposed strat egy of data generation and distribution for enhancing dataset quality, offering promising applications in plant disease detection and precision agriculture. 

---
# Counting the Bandages: On measuring the time that AI saves 

**Authors**: Solomon Shalom Lijo  

**Link**: [PDF](https://arxiv.org/pdf/2610.35942)  

**Abstract**: I consider what it means for AI assistance to save time. I contrast a stopwatch view, on which time saved is whatever the clock records, with a ledger view, on which it is a counterfactual that has to be read next to the errors the work lets through, and I argue for the latter. The argument has four parts. I define a model-free pair of quantities, time saved and excess escaped defects, and one decision number built from them: the break-even defect cost. I give a baseline model of checking in which, until the model is good enough not to check, a calibrated worker's escaped-defect rate does not depend on how good the model is; checking time grows with the logarithm of the model's error rate; and over-trust buys minutes on the clock at a larger cost in total minutes. I show what randomized studies can and cannot identify, and I replicate METR's developer study from its public data: the same 246 issues support a slowdown of 19% or 24% depending on the estimand (34% without adjusting for a chance imbalance), and AI helped on somewhere between 3% and 82% of them. Finally, I restate the published record, with stated bounds in place of imported point assumptions. In GDPval's own review-every-output workflow, review time alone caps GPT-5's speedup at 3.6x; the consultants outside the frontier were faster, and more often wrong. Last, I test two of the model's predictions in a small pre-specified experiment on AI reviewers checking AI drafts of the same kind of brief, built from SEC filings made after the models' training data were collected. Told the drafter's true error rate, the reviewer checked more as it rose, but all-or-nothing, switching near where its prompt's costs break even. Told that the drafter erred ten times less often than it did, it checked 7 fewer of 9 figures and let a wrong figure through in 22.5% of briefs instead of none: faster and worse, entirely through reduced effort. 

---
# Honeycomb-Surfels: Compact Hexagonal Primitives for Efficient Scene Rendering and Reconstruction 

**Authors**: Prakhar Kaushik, Soumava Paul, Alan Yuille  

**Link**: [PDF](https://arxiv.org/pdf/2610.37668)  

**Abstract**: Novel view synthesis and 3D reconstruction have advanced rapidly with radiance fields and splatting, yet their memory and computational demands limit their use in resource-constrained settings. We introduce Hexels: novel planar hexagonal surface elements with a first-order spatial color field for efficient 3D scene and 2D image representation. Hexels exploit locally planar regions with smoothly varying appearance using fewer, larger primitives. Their hexagonal support provides an area- and boundary-efficient domain, while the local color field captures appearance variation within each primitive. Across 34 scenes, Hexels improve PSNR over memory-matched 2DGS by 4.53-7.38 dB across datasets while requiring lower training VRAM. We further ablate Hexels by showing that per primitive, Hexels outperform 2DGS by upto 4.74 dB while using lower training VRAM and having 2.31 times faster FPS. Even in unconstrained settings, Hexels offers complementary quality-memory trade-offs. For efficient 3D surface reconstruction, Hexels reduce error by 28.9% compared with memory-matched 2DGS across 15 DTU scans. On 6 Tanks and Temples surface scenes, Hexels achieves the highest mean F1 among compared methods (34.38%, versus 31.72% for 2DGS). We further also show that Hexels, as a general primitive, improves 2D image representation over Image-GS. Hexels push the Pareto frontier on the quality-efficiency trade-off of surface-aligned primitives. 

---
# ThinkWorld: World Imagination Reinforces Chain-of-Dynamics Reasoning 

**Authors**: Chaehyun Kim, Chi-Pin Huang, Jisoo Kim, Min-Hung Chen, Yu-Chiang Frank Wang, Seungryong Kim, Fu-En Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37666)  

**Abstract**: Vision-Language-Action (VLA) models have made substantial progress in robotic manipulation, but their generalization remains constrained by the diversity and coverage of action-labeled robot demonstrations. Human videos provide much broader behavioral and environmental experience, while learning from them faces a fundamental trade-off between scalable data and explicit action grounding. We introduce ThinkWorld, a VLA learning framework that grounds action reasoning in world dynamics and strengthens it through reinforcement learning with world imagination. ThinkWorld represents each action thought as a Chain-of-Dynamics (CoD), a sequence of discrete motion tokens that captures the intermediate dynamics connecting the current world to a future state and serves as a common action representation across embodiments. We first initialize CoD with action-labeled demonstrations, then predict the future induced by each CoD and reinforce those whose predictions better match the observed outcome. This enables videos without action annotations to provide direct supervision from observed state transitions, allowing human and robot experience to refine the same reasoning space. Experiments in simulation and on real robots demonstrate effective human-to-robot transfer and strong generalization to unseen tasks and environments. 

---
# Attention as Lattice Gauge Theory: Relations Interact Only Through Wilson Loops 

**Authors**: Aakash Sarkar, Ananya Mukherjee, Motoshige Sato  

**Link**: [PDF](https://arxiv.org/pdf/2610.37656)  

**Abstract**: Relations compose through the entity they share - Alice is Bob's mother and Bob is Carol's father, so Alice is Carol's grandmother - and the order matters, since a father's mother is not a mother's father; verifying a composed relation means closing the loop. We show that if each entity's coordinate frame may be rotated independently (per-entity frame freedom), three things follow. In the pure-gauge sector, relations that share no cycle cannot interact at all. The only gauge-invariant score of a triangle is a class function of its holonomy $H = T_{ij} T_{jk} T_{ki}$ - a Wilson loop. And the only equivariant update of an edge is $U = \Psi(H)\, T_{ij}$ with $\Psi$ conjugation-equivariant. Edge attention by Wilson loops is therefore not one design among many but the unique form the symmetry permits. Three communities hold pieces of this form: the commuting-operator embeddings of knowledge graphs are its abelian (commuting) sector; AlphaFold's triangle update and the Edge Transformer have its shape - an edge updated through the third vertex - without its constraints; and lattice QCD builds it equivariantly on lattices. We instantiate the equivariant sub-family on general relational graphs as holonomy attention. With no training and composition alone, it infers unobserved $S_4$ relations on sparse graphs at 0.88 with a fifth of the edges corrupted, at or slightly below specialist synchronization algorithms, where every commutative architecture is capped at 0.083; its holonomy score flags corrupted triangles at AUC 0.99 where commuting scores are capped at 0.76 by a second selection rule, and it separates clusters without labels. 

---
# Can LLMs Uncover Hidden Physical Laws through Free Exploration? 

**Authors**: Zhen Yang, Wen Wang, Xander Xu, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.37649)  

**Abstract**: Scientific discovery often begins with an unexpected observation that prompts a new research question. In a world with altered physical laws, can large language models (LLMs) notice unexpected behavior, such as heavier objects falling more slowly, and discover the law through experiments without being told what to investigate? To evaluate this ability, we introduce LongChang, an automated framework that creates simulated worlds with altered physics, where one or more laws differ from ordinary physics. The evaluated LLMs act as explorers, manipulating objects and building tools to investigate their own questions without research tasks or hints about these laws. Separate LLM agents generate and review scenes, review tool designs, and evaluate reports for correctness and experimental support using the altered physical laws and exploration records. These records guide scene evolution for further exploration, allowing the full cycle to run without human intervention. Among 12 LLMs, GPT-6-Astra achieves a discovery rate of 16.7%, while the other 11 score 0%. The exploration records reveal uninvestigated anomalies and only partial identification of the altered physical laws. 

---
# MuonGS: Pretraining LLMs with Relaxed Weight-Norm Control 

**Authors**: Athanasios Glentis, Dawei Li, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2610.37647)  

**Abstract**: Weight decay and explicit weight-norm constraints improve optimizer performance in language model pretraining, yet a unified understanding of why such control helps and how much is needed remains incomplete. We analyze norm control in Muon over the entire training trajectory using the \emph{accumulated effective learning rate}, $C_T=\sum_{t<T}\eta_t/\|W_t\|_F$, where $\eta_t$ is the nominal learning rate and $\eta_t/\|W_t\|_F$ is the effective learning rate relative to the weight norm. For bounded update directions, accumulation over any training interval bounds relative weight displacement. We show that vanilla Muon can accumulate only logarithmically, potentially limiting exploration and helping explain its slower empirical progress, whereas MuonW and MuonH guarantee at least linear accumulation under the stated conditions. Going beyond bounded-norm control, we show that sufficiently slow norm growth guarantees faster accumulation than the logarithmic worst case, opening up less restrictive optimizer designs.
Based on this insight, we propose MuonGS, which uses Gram--Schmidt (GS) projection to remove the radial component of each Muon update without fixing the weight norm. With bounded update directions and a constant nominal learning rate, we prove that the weight norm grows at most proportionally to $\sqrt{t}$ after $t$ steps. Consequently, MuonGS guarantees $C_T=\Omega(\sqrt{T})$, with accumulation sustained even in later training. Further, for smooth scale-invariant objectives we establish its $O(T^{-1/2})$ stationarity guarantee. In pretraining experiments with 160M- and 400M-parameter dense models as well as a 135M active- and 1B total- parameter MoE model, MuonGS achieves lower validation perplexity than MuonH at long training horizons. In particular, for the dense model it achieves an estimated token-equivalent speedup up to about $15\%$ at matched validation loss to MuonH. 

---
# VideoLVR: Latent Visual Reasoning over Grounded Video Evidence 

**Authors**: Shilong Li, Xingyuan Bu, Hao Shi, Yun Xing, Weixin Feng, Qiangpeng Yang, Shilei Wen  

**Link**: [PDF](https://arxiv.org/pdf/2610.37613)  

**Abstract**: Chain-of-thought reasoning has substantially advanced video understanding in multimodal large language models (MLLMs). However, language-based reasoning traces do not provide a reliable mechanism for models to revisit relevant visual evidence, leading to visual forgetting and impairing their ability to reason over video content. Existing approaches maintain visual grounding during reasoning in two main ways: they either refer to evidence using textual coordinates, which specify symbolic locations rather than the visual content itself, or reinsert cropped regions or frames into the multimodal context, preserving visual details at the cost of substantial context expansion. To address these limitations, we introduce VideoLVR, which constructs latent visual evidence from local regions at relevant video moments, thereby providing visual support for intermediate reasoning steps. To supervise latent visual evidence generation, we introduce a visual token prediction objective that formulates visual token reconstruction as prediction over a vocabulary derived from the input video, thereby unifying visual evidence reconstruction and text generation within the same token prediction framework. This shared prediction framework enables MLLMs to reason over interleaved text tokens and latent visual states while integrating evidence from different moments. We further extend VideoLVR to reinforcement learning by treating sampled visual tokens as actions alongside text tokens and jointly optimizing the policy using final-answer rewards. By efficiently revisiting task-relevant visual information during reasoning, VideoLVR improves video understanding without substantially expanding the multimodal context. Experiments on multiple video benchmarks demonstrate its effectiveness, especially on tasks requiring fine-grained visual evidence. 

---
# Beyond Answer Changes: Matched Same-Class Controls for Local Audio Evidence 

**Authors**: Yunuo Wang, Maoyuan Chen, Yuanjin Huang, Xing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37605)  

**Abstract**: For audio-language models, an answer change after a local sound-event edit can reflect a response to the event's class, sensitivity to waveform rewriting, or both. We compare matched same-class and different-class replacements of a queried event. Both use the same interval, level estimate, donor fitting, and crossfade, while the same-class replacement preserves the correct answer. Across 600 synthetic soundscapes and three models from two architecture families, different-class replacements produce 49.7 to 60.9 percentage points more answer changes than same-class replacements. Same-class accuracy remains within 5.2 percentage points of original accuracy. Among trials with correct original and same-class answers, soundscape-weighted selective accuracy for the inserted class is 73.7 to 81.9%, averaging eligible trials within each soundscape and then weighting eligible soundscapes equally. This protocol tests how model answers respond to local changes in event class. 

---
# MoI-Bench: A Benchmark for Image Generation and Editing on Mobile Devices 

**Authors**: Kailai Feng, Haoxiang Rao, Yuxiang Wei, Bo Chen, Junsong Li, Yang Pan, Chengkun Cai, Jagmohan Chauhan  

**Link**: [PDF](https://arxiv.org/pdf/2610.37599)  

**Abstract**: Developing on-device image models requires considering practical user needs, task types and device resource constraints. However, existing benchmarks provide limited guidance on identifying capability gaps between on-device models and server-side models. We introduce MoI-Bench, a benchmark grounded in mobile user needs, comprising 64 subtasks and 1150 samples. It defines 16 operations, such as removal and replacement, and eight visual target categories covering content such as portraits and text. These form two orthogonal axes whose applicable combinations define specific tasks, such as removing facial makeup. To better guide model improvement, we further propose MoI-Eval, which introduces task-specific criteria covering instruction following, visual quality and detail preservation. We further construct sample-level evaluation questions and aggregate their scores into overall quality and capability profiles along the operation, target and criteria axes. We evaluate representative on-device and server-side models and measure mobile deployment costs. Results show that this decoupled Operation x Target design provides finer-grained guidance for improving compact models. For example, it highlights the need to prioritize specific capability gaps (e.g., text-replacement) over broad application-level goals (e.g., improving text editing). Unlike conventional benchmarks that rank models using a single score, we introduce a two-dimensional Quality-Deployability capability map to assess models for practical mobile use. The map shows that current on-device models can achieve efficient generation or editing but expanding their task coverage deserves greater attention than further improving deployment efficiency. Further progress requires improving editing quality, particularly identity preservation and text rendering, while broadening task coverage. 

---
# GAS-PFN: Graph-Aware Spatiotemporal Prior-Data Fitted Networks for Zero-Shot Traffic Forecasting 

**Authors**: Joseph Natter, Long Vu, Yifan Zhang, Bobin Deng, Kun Suo, Rui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37598)  

**Abstract**: Traffic forecasting requires modeling complex temporal dynamics and the spatial dependencies imposed by road topology. Supervised graph-based models achieve strong performance, but are often trained for a particular dataset and sensor graph, making deployment to new or evolving networks costly. Prior-Data Fitted Networks (PFNs) offer a zero-shot alternative by learning in-context inference from synthetic task priors, which provide an explicit mechanism for encoding domain-specific assumptions. Existing forecasting PFNs, however, do not explicitly represent the graph-structured dependencies governing traffic dynamics. To combine graph-aware traffic modeling with zero-shot adaptation, we introduce the Graph-Aware Spatiotemporal Prior-Data Fitted Network (GAS-PFN). GAS-PFN incorporates sampled road topology into its synthetic task prior through random-walk diffusion, teaching the model to learn graph-coupled dynamics during pretraining. Its co-designed inference architecture uses two mask-conditioned passes through a shared Transformer backbone to separately model sensor-specific temporal dynamics and spatiotemporal interactions among neighboring sensors before integrating their representations for prediction. This design enables GAS-PFN to use target-network structure and context without parameter updates. Across six benchmarks at the 90-minute horizon, GAS-PFN reduces MSE by 34.0% on average relative to TimePFN, its closest PFN-based predecessor. It also reduces MSE by 27.7% on average relative to four recent industry-developed general-purpose zero-shot forecasters. In this comparison, all models receive 90 minutes of contiguous recent history, and any additional context is drawn from each dataset's chronological training split. These results indicate that aligning PFN priors and inference architectures with graph structure improves in-context learning for zero-shot traffic forecasting. 

---
# Understanding the limits of latent world models 

**Authors**: Rodrigo Gonzalez Laiz, Steffen Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2610.37588)  

**Abstract**: World models are widely proposed as core components for enabling agents to reason over long temporal and spatial horizons. Yet current evaluations often emphasize task performance while providing limited insight into whether learned world models acquire predictive latent dynamics that reflect the underlying structure of the environment. Here we introduce the concept of faithful latent world models as an equivariance condition between the environment's transition map and the learned latent dynamics. We propose latent rollout consistency (LRC) as a metric to evaluate action-conditioned latent prediction. Strikingly, we find that across the studied state of the art latent world models, none solves the task by actually faithfully modeling the dynamics of the environment. 

---
# REGER: Modeling inter-item Relations with Semantic IDs for Generative Recommendation 

**Authors**: Yuxi Huang, Silin Zhou, Chenhao Wang, Zihan Li, Gangming Guo, Pei Ke, Shen Gao, Shuo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37584)  

**Abstract**: Personalized recommendation depends on learning semantic and collaborative relations among items from online content and user interactions. Generative recommendation (GR) offers a promising paradigm by autoregressively generating discrete semantic IDs (SIDs) of items. However, randomly initialized SID token embeddings and GR models trained from scratch do not directly inherit the semantic relations inherent in items, while inter-item collaborative relations are learned largely implicitly through token-level attention and next-token supervision. To address this limitation, we propose REGER, a unified framework that explicitly incorporates inter-item relations into semantic pretraining, history encoding, and SID generation. We pretrain the GR model through generative semantic retrieval, enabling it to inherit inter-item semantic similarity by learning to generate semantic neighbors' SIDs. We combine global attention with local attention aligned with SID boundaries to capture historical context and directed transitions between adjacent items. We further develop a theoretical framework for hierarchical preference decomposition over the SID prefix tree, proving an exact factorization of collaborative preference distributions and a weighted decomposition of item-level KL divergence into prefix-conditional KL divergences. Building on this analysis, we distill a discriminative teacher's full conditional distributions over all legal next tokens at ground-truth prefixes along the target SID path, jointly optimizing this objective with cross-entropy. Extensive experiments on real-world datasets demonstrate consistent improvements over existing GR baselines and support the effectiveness of the proposed preference distillation. 

---
# Gym-Anything-RT: Scaling Real-Time Environments for Vision-Language Agents 

**Authors**: Pranjal Aggarwal, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2610.37567)  

**Abstract**: AI agents have made substantial progress in domains where they have a non-trivial amount of time to deliberate between actions, such as mathematics, software engineering, and performing tasks on desktop applications. It is less clear how such models fare in settings that demand real-time interaction, such as playing a video game or operating a physical device. In these settings, the environment state may evolve as the agent is reasoning, or even as the underlying model is performing a forward pass. To gain insight into the real-time interaction capabilities of vision-language agents, we introduce Gym-Anything-RT. Our key idea is to construct visual minigames that demand a portfolio of capabilities required for interactive real-time settings. Gym-Anything-RT uses an agentic pipeline to automatically generate such environments based on a taxonomy of six capabilities required for real-time interaction, including the ability to perceive the environment and act under an evolving environment state. A key feature of Gym-Anything-RT is that it allows for controlled comparison of real-time interaction capabilities: we can run the same minigame in real time, or pause or slow down the environment to relax the requirement for real-time actions. We use Gym-Anything-RT to construct 100 real-time minigames and evaluate AI agents on them. We find large capability gaps in real-time interaction. For example, GPT-6 Astra (xHigh) achieves a pass rate of only 9.0% on the real-time minigames, which rises to 75.0% when the environment is paused. We also identify intriguing behaviors by the most powerful frontier agents: for instance, when given the ability to write code, these agents write code that bypasses the need to perceive and act in real time. We publicly release Gym-Anything-RT and the generated minigames to spur further research on AI agents in real-time interactive settings. 

---
# When uncertainty quantification fails: structural vulnerabilities and corrections for selective prediction under clinical class imbalance 

**Authors**: Walter Endrizzi, Flavio Ragni, Raffaele Marchesi, Stefano Bovo, Monica Moroni, Giuseppe Jurman, Venet Osmani  

**Link**: [PDF](https://arxiv.org/pdf/2610.37561)  

**Abstract**: Background. Selective prediction systems improve the safety of clinical artificial intelligence by deferring ambiguous cases to human experts while automating high-confidence decisions. However, this safety mechanism relies entirely on the premise that uncertainty scores accurately reflect clinical ambiguity. In prognostic tasks characterized by severe class imbalance, where the clinical outcome of interest is a minority, this assumption breaks down. Standard uncertainty measures cease to capture clinically aligned ambiguity; instead, they become a proxy for class membership, systematically penalizing minority patients regardless of their individual predictive clarity.
Results. Across 56 synthetic conditions and three neurodegenerative cohorts spanning severe class imbalance to near-balanced conditions (multiple sclerosis, Parkinson's Disease, Alzheimer's Disease), our framework consistently maintained bounded, balanced risk across trajectories. In contrast, global metrics either collapsed minority Sensitivity or protected the minority at the expense of majority patients. Applying CCR with standard entropy fails to protect the minority because it breaks within-class monotonicity when tau=0.5. Combining CCR with threshold-aware metrics restores strict monotonicity, correctly identifies ambiguous patients, and safely mitigates minority exclusion.
Conclusions. Under clinical class imbalance, standard uncertainty quantification measures dataset sparsity rather than clinical ambiguity, creating a systemic illusion of safety. This bias is a correctable structural flaw. Clinical pipelines deploying selective prediction under imbalance should adopt threshold-aligned uncertainty measures and class-stratified evaluation. This reduces class-specific risk imbalance, supporting safer selective prediction under the evaluated conditions without silently excluding the most vulnerable patients. 

---
# Kinematic Constraint-Referenced Evaluation for Articulated Object Manipulation 

**Authors**: Baptiste Bonin, Caro Strickland, Audrey Durand  

**Link**: [PDF](https://arxiv.org/pdf/2610.37550)  

**Abstract**: Manipulation policies are evaluated predominantly by binary task success, which cannot tell whether the commanded gesture is the one the object's kinematics admit. We introduce Constraint-Referenced Strategy (CRS), which projects the commanded end-effector twist onto the object's joint axis, separating progress along the joint from off-constraint waste. CRS requires a joint model and a joint-following scripted controller derived from it, but no recorded demonstration or learned judge. On 74 simulated articulated objects, success registers three transfer failures as lower rates but attributes none, while the two CRS channels give each a distinct signature. A door-trained behavior-cloning policy opens drawers in 59 of 110 episodes, and in these passes it commands about $40\times$ the off-axis rotation of a drawer-trained policy's passes ($1.27$ vs.\ $0.032$~rad/s). Relative to a scripted controller scored with the same axis, CRS verdicts are unchanged under joint-axis errors up to $30^\circ$; absolute thresholds tolerate about $8^\circ$. 

---
# When, Where, and What to Correct: Human-Guided Adaptive Residual Correction for Robot Policies 

**Authors**: Anukriti Singh, Maxime Caniot, Pratap Tokekar, Firas Abi-Farraj  

**Link**: [PDF](https://arxiv.org/pdf/2610.37547)  

**Abstract**: Imitation policies that reliably execute most of a manipulation trajectory can still fail in localized regions requiring last-mile precision. Updating the full policy or exploring residual actions everywhere treats this local failure as a global learning problem. We introduce Adaptive Residual Correction (ARC), a lightweight post-training module that uses reinforcement learning (RL) to learn a residual correction on top of a frozen policy while using sparse human corrections to structure autonomous exploration. Rather than treating human corrections as imitation targets, ARC uses pre-intervention states to determine 'when' residual control should become active and correction geometry to determine 'where' in residual action space autonomous refinement should explore; task reward determines 'what' residual is ultimately executed. Across three precision-manipulation tasks on a real robot, ARC improves average task success from 33.3% to 75.0%. The human corrections define task-dependent 2-3 dimensional subspaces of the seven-joint residual space that retain 96.0-97.7% of squared correction magnitude. Together, these results show that sparse human corrections can localize and constrain autonomous post-training without requiring the observed correction itself to serve as the final action target. 

---
# GroundControl: Satellite-Conditioned Multi-View Diffusion for Reconstruction-Free Novel View Synthesis 

**Authors**: Akshay Krishnan, Matias Turkulainen, Jamie Wynn, Zawar Qureshi, Mohamed Sayed, Guillermo Garcia-Hernando, Sara Vicente, James Hays, Michael Firman, Daniyar Turmukhambetov  

**Link**: [PDF](https://arxiv.org/pdf/2610.37545)  

**Abstract**: Multi-view or video diffusion models have recently emerged as state-of-the-art approaches for novel-view synthesis from sparse image collections. While they can produce high quality images, they tend to hallucinate unrealistic spatial layouts when synthesizing views outside the input views (scene extrapolation). We propose to address this challenge for outdoor scenes by using satellite images as an additional information source to condition the multi-view generative model. We introduce a new satellite-conditioned multi-view diffusion model that combines conditioning from a satellite image and street-level input images when generating novel-views. Our cross-attention based conditioning allows us to retain pretrained multi-view diffusion priors to generate seamless, realistic, novel views from a handful of source views, while also adhering to the scene layout as seen from satellite images. Our model does not use 3D scene reconstruction, and only relies on posed street and satellite views. Our experiments on outdoor datasets show that our model is better than both satellite-conditioned or street-view conditioned baselines on novel-view synthesis, especially on challenging cases where target views have very little overlap with source views. Project page: this https URL 

---
# Which Hand Was Late? Diagnosis for Robot Failure Recovery in Bimanual Tool Use 

**Authors**: Baptiste Bonin, Caro Strickland, Audrey Durand  

**Link**: [PDF](https://arxiv.org/pdf/2610.37533)  

**Abstract**: Failure diagnosers for robot manipulation are increasingly vision--language models (VLMs), trained and evaluated on single-arm failures produced by perturbing a successful demonstration. Bimanual tool use adds failures of other kinds: a tool that slips out of its socket while still held, two correct gestures made at different times, and attempts that fail for two reasons at once. On a simulated bimanual tool-use task whose failure causes are read from the recorded trace, we measure what each sensing channel lets a diagnoser tell. The tool's slip is read on the tool hand; which hand was late is read on both arms, while a sensor on one hand reads its own hand's delay and at most part of the other's; and whether a late attempt was also pushed too weakly is not identifiable from passive sensing when the tool hand is late. Within two attempts, a repair policy that escalates when the cause is not identifiable repairs as many failures as one told the true cause. A VLM's internal state carries the late hand almost perfectly, along a direction no closer to the one its answer reads than a random one; whether the answer reports it depends on the wording of the question. 

---
# The Structure Gap: When More Data Cannot Replace Structured Reasoning 

**Authors**: Kheer Sagar Patel, Mojtaba Soltanalian  

**Link**: [PDF](https://arxiv.org/pdf/2610.37523)  

**Abstract**: A persistent belief in modern AI is that sufficiently large learners trained on sufficiently much data can make explicit domain knowledge optional. This paper examines a limitation of that belief. We define structure-encoded problems: tasks whose input-output map is determined, or sharply constrained, by known structure such as a solver, symmetry group, graph, physical law, measurement operator, grammar, combinatorial search space, or optimization objective. For such tasks, the central question is not expressivity--a dense neural network may contain the desired map--but finite-data identification, extrapolation, and certification. We introduce the structure gap: the finite-resource gap between a method that receives the governing structure and a learner that receives only examples. The strongest useful form of the claim is asymptotic rather than absolute: structure-agnostic learning may approach a structured solver as coverage grows, but every finite dataset can leave residual uncertainty that the structure resolves by construction. We prove this exactly for a Gaussian inverse problem: the known-operator Wiener solver attains Bayes risk, while the dense empirical learner in the correct linear class has persistent expected excess risk. We also prove a cell-coverage lower bound and an unobserved-orbit non-identifiability result. Controlled experiments in known-operator inversion, sparse recovery, and graph reachability with size extrapolation instantiate the mechanisms. When validated structure is available, it can constrain the hypothesis space before learning begins, leaving learning to address components the structure does not specify. 

---
# Orthogonal and Parametrized Multi-Attribute Steering of LLMs 

**Authors**: Remy Ogasawara, Cullen Anderson, Narmeen Fatimah Oozeer, Foad Namjoo, Amir Abdullah, Jens Kristian Refsgaard Schou, Jeff M. Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2610.37517)  

**Abstract**: Activation steering offers an effective and efficient way to modify an LLM's behavior at inference time. However, it remains unclear how to apply steering such that the signal is consistently strong and the steering strength has a universally interpretable meaning. We address this first by identifying the most steerable interval for each steering vector and rescaling the steering range accordingly. Building on this, we introduce a more interpretable method for steering, called parameterized steering which instead of pushing activations in a direction, it sets them at the desired trait strength. Next, with multi-trait steering, we find that naively combining vectors causes them to interfere with one another. In response, we develop a method to orthogonalize the vectors; it decouples these traits, allowing a model to exhibit one behavior without inducing the others, thus enabling multiple traits to be steered simultaneously. 

---
# Strategyproof Deep Reinforcement Learning for Online Resource Allocation 

**Authors**: Fariba Dehghan, Sebastian Stein, Vahid Yazdanpanah, Stephanie Gauthier  

**Link**: [PDF](https://arxiv.org/pdf/2610.37504)  

**Abstract**: Online resource allocation faces two coupled challenges: (1) allocating scarce resources efficiently under uncertain future demand, and (2) ensuring that agents report task valuations and requirements truthfully (strategyproofness). Reinforcement Learning (RL) adapts allocation decisions through experience but does not necessarily ensure strategyproofness. Existing strategyproof RL uses a linear allocation rule, which limits the resource interactions it can represent. Neural mechanism design methods offer greater expressiveness, but provide only approximate incentive guarantees. Against this background, we introduce the first strategyproof deep RL mechanism for online resource allocation. It integrates three novel components: (i) a monotonic neural network that learns nonlinear interactions by construction (higher reported value cannot reduce acceptance, while a longer reported duration or larger resource requirement cannot increase it); (ii) a generalised critical payment: the lowest accepted value, with all other inputs fixed; and (iii) training by either RL or imitation of an existing rule. Together, monotonicity and the critical payment ensure strategyproofness and individual rationality, while the neural rule provides flexibility and imitation improves training efficiency. Empirically, on a four-resource stream with two partner pairs that a linear difference cannot represent, the neural rule achieves 97.8% of an offline welfare upper bound, compared with 93.8% for a linear learner and 89.0% for first-come, first-served, closing 65% and 80% of the remaining optimality gap, respectively. Moreover, imitation reaches 97.5% in only 261 seconds, about 89 times faster than RL. 

---
# Follow the Gradient: Tracing Safety Degradation Back to Training Data 

**Authors**: Saket Reddy, Amro Abbas, Anshuman Suri, Ari Morcos, David Schwab, Pratyush Maini  

**Link**: [PDF](https://arxiv.org/pdf/2610.37481)  

**Abstract**: Fine-tuning Large Language Models (LLMs) on instruction data can quietly erode their safety, even when no training example looks harmful. Because this safety-degrading data looks benign, standard content filters struggle to catch it. We tackle this problem by developing SimpleLoRIF, a scalable gradient embedding method that identifies fine-tuning examples whose gradients align with unsafe behavior. By simplifying a low-rank attribution method and restricting backpropagation to the final four transformer blocks, SimpleLoRIF can embed datasets on the order of a million examples, boosting gradient attribution from a small-scale analysis tool into a practical filter for full post-training datasets. We first validate it on emergent misalignment, where we find that gradients can connect insecure-code training data to the broadly harmful behaviors that appear after fine-tuning, with AUC as high as 0.865 in the gradient space of misaligned models. We then use SimpleLoRIF to build SafeTulu, a version of Tulu 3 with the 20% of examples most gradient-aligned with unsafe responses removed. Training on SafeTulu reduces GCG attack success rate by up to 22% across Qwen, Llama, and Olmo, outperforming both LLM-as-a-judge and prior gradient-based filtering, while largely preserving or even improving capability. Finally, we analyzed the removed data and identify possibly under-studied sources of safety degradation, such as tool-execution failures and narrative arithmetic. We release SafeTulu along with a Hugging Face Space that lets researchers build their own customized versions of SafeTulu to fine-tune on, such as by changing the removal fraction or filtering only certain subsets. 

---
# HGPRGDel: GPR-Aware Learning and Ranking for Growth-Coupled Gene Deletion Design 

**Authors**: Ziwei Yang, Takeyuki Tamura  

**Link**: [PDF](https://arxiv.org/pdf/2610.37478)  

**Abstract**: Designing microbial strains for growth-coupled production, in which cellular growth is linked to the synthesis of a target compound through genetic interventions, is a fundamental goal in metabolic engineering. Constraint-based methods can identify growth-coupled deletion strategies, but design for each target typically requires a separate search across a large number of possible gene-deletion combinations. This motivates a reusable approach that learns from previously evaluated strain designs to prioritize candidate interventions before downstream constraint-based evaluation. We present a reproducible protocol for growth-coupled gene deletion design that learns and ranks gene-deletion priorities for each target metabolite. Using computationally curated deletion strategies as learning resources, the protocol generates continuous deletion-priority rankings, constructs candidate strategies at adjustable search-space budgets, and evaluates these candidates through constraint-based metabolic analysis. Its learning component, HGPRGDel, explicitly represents gene-protein-reaction (GPR) logic, capturing how multiple genes can jointly support a reaction or alternative genes can maintain its activity, while integrating molecular and reaction-level information. Across e_coli_core, iMM904, and iML1515, the protocol achieves the highest macro-F1 and AUC among the evaluated learning configurations. HGPRGDel's rankings recovered an average of 90% of the reference deletion genes from MetNetComp by examining fewer genes than the evaluated baseline methods. By separating learned intervention prioritization from downstream constraint-based assessment, the protocol provides a reproducible framework for reusing strain-design knowledge while controlling the size of the candidate space and retaining mechanistic evaluation of growth-coupled production. 

---
# How Reliable Is Individualized Functional-Network Attribution in Resting-State fMRI Cognitive Prediction in ADNI? 

**Authors**: Sheethal Bhat, Mahfuzur Rahman Chowdhury, Imam Ahmed Shaik, Rose Dawn Bharath, Siming Bayer, Andreas Maier  

**Link**: [PDF](https://arxiv.org/pdf/2610.37472)  

**Abstract**: Individualized attribution estimates which functional networks drive a model prediction for a given subject and is often proposed as an interpretable resting-state fMRI (rs-fMRI) biomarker, but its test-retest reliability is rarely quantified. We evaluate per-subject network-ablation attribution for a frozen resting-state functional connectivity (FC) predictor (Schaefer-200 FC predicting MMSE/ADAS-Cog13) trained on N=875 subjects from the Alzheimer's Disease Neuroimaging Initiative (ADNI), using, and (iii) a subject-level permutation null. Split-half reliability is near-ceiling (Spearman p=0.948/0.950) but drops on scan-rescan (p=0.609/0.616). Under the mismatched-subject null, correlations remain substantial (p~0.46-0.47), indicating strong population-shared structure, yet matched pairs exceed this null (z~6.6-6.9, p<=10-4). Thus, in ADNI, individualized network attribution contains a modest but non-zero person-specific component atop a strong population-shared pattern. We recommend reporting scan-rescan reliability together with a permutation-null control, rather than within-session proxies alone, to avoid overstating individual specificity. 

---
# PerceptRL: Making Visual Perception Rewardable in MLLMs from Answers to Interactions 

**Authors**: Tongkun Guan, Zhibo Yang, Mingkun Yang, Haocheng Wang, Wei Shen, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37471)  

**Abstract**: Fine-grained visual perception remains a bottleneck during post-training of multimodal large language models (MLLMs), as perceptual errors can constrain gains from reasoning-focused optimization. Although image-caption pre-training improves perception, its influence can weaken as subsequent stages change data mixtures and objectives. Continuing caption supervision during post-training is challenging because subjective, information-dense descriptions complicate verifiable reward design. We present PerceptRL, which strengthens perception at two complementary levels. Perception RL uses a multi-agent pipeline to extract verified errors from dense image captions and convert the corresponding visual facts into QA pairs with fixed answer labels and rule-based rewards. Chain QA links correlated errors, while multiple-choice QA combines unrelated errors. A specialized route targets high-error domains with tailored verification, and a domain-agnostic general route covers errors beyond these domains. Perception Agentic RL addresses fine-grained and dense cases unresolved by single-turn Perception RL through multi-turn inspection with visual tools. We observe coordinate drift even in models trained to use tools, where incorrect crops or annotations distort evidence and propagate errors. Coordinate Injection corrects spatial operations in cold-start trajectories, while a grounding-aware process reward supervises intermediate spatial fidelity alongside final-answer correctness. Training proceeds through shared cold-start SFT, single-turn Perception RL, and multi-turn Perception Agentic RL. Across seven benchmarks under a shared tool-use protocol, PerceptRL achieves 71.6% average accuracy, surpassing the base model with the same tools and the strongest evaluated frontier VLM by 15.7 and 5.2 percentage points, respectively. 

---
# Autonomous fixed-wing interception using reinforcement learning 

**Authors**: Daan van Hamersveld, Ewoud Smeur, Iurii Kriachko  

**Link**: [PDF](https://arxiv.org/pdf/2610.37470)  

**Abstract**: Medium altitude, long endurance unmanned aerial systems such as Shahed-type strike drones are an unanswered threat in modern conflicts. Conventional air defense systems are too expensive to provide a feasible large-scale solution. Quadcopter interceptors are a cheap, battle-proven alternative, but their extremely limited flight time restricts them to low altitude targets at short range, whereas fixed-wing platforms have the endurance to reach higher altitude targets from longer distances. However, such interceptors are still operated manually. Recent developments in learning-based controllers show potential for automation. In this work, a fixed-wing interceptor is presented that is controlled by a neural network that maps filtered monocular vision observations directly to actuator commands. The policy was trained in a custom Python environment using reinforcement learning and proximal policy optimization. The policy was then deployed on a fixed-wing platform and tested in real interception scenarios. As a baseline, two classical guidance controllers were also developed and tested in simulation. The policy is able to intercept targets both in simulation and in real flight. In simulation, the policy outperforms the classical controllers under most conditions. The interceptor was ultimately successfully validated in real flight by intercepting maneuvering fixed-wing targets. This novel system shows the potential of artificial intelligence for autonomous control of fixed-wing interceptor drones. 

---
# Test-Time Communication in Trading Agent Swarms 

**Authors**: Avid Fayaz, Harry Zhang, Håvard Alstadheim, Erik Nordlöf, Vilhelm Karlin, André Silva  

**Link**: [PDF](https://arxiv.org/pdf/2610.37448)  

**Abstract**: Communication between agents has mostly been studied on tasks with a shared objective and fast, clear feedback. Trading has neither: each agent manages its own account, with decisions being scored by markets late and noisily. A trading swarm is thus a population of independent decision-makers rather than a team, for which convergence is a risk and not the goal. We ran 200 agents from four model families in groups of ten under five communication regimes, from isolation to message boards and shared notes, over seven US trading days (1 to 9 October 2026), logging everything each agent read before every decision. In two of the four families, communicating groups came to hold far more of the same stocks than the isolated group. A group's return was largely the return on what its members held in common. Seeing a post from a groupmate made an agent 7 percentage points more likely to reuse an eight-word phrase from it. Agents that started too close together to see each other's posts shared such a phrase in 14% of cases, the rate that common inputs produce. Seeing the post did not detectably change that morning's trades. Case studies traced the adoption of positions and decision rules over subsequent days, and communicating groups sold in step more often than isolated ones. Because isolated groups also converged on the same stocks, sometimes completely, assessments of herding need to measure excess convergence relative to a no-peer baseline. 

---
# Representing the Unseen: Rethinking Visual Imagination for Spatial Reasoning 

**Authors**: Zengqi Zhao, Zun Wang, Shoubin Yu, Mohit Bansal, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37437)  

**Abstract**: Visual world models increasingly support spatial reasoning by generating RGB views of unobserved states. However, RGB has become the default interface largely because it is well suited for visual generation, rather than necessarily being the most effective representation for downstream reasoning. Different spatial questions may benefit from different representations of the same imagined state: top-down geometry can facilitate global distance comparisons, while panoramic views provide broader context for reasoning about open space. To isolate the role of representation in spatial imagination, we introduce ImagiRep, a simulator-grounded benchmark of 922 questions spanning seven question types across two settings, Viewpoint Imagination and Cross-view Integration, providing oracle RGB and six alternative representations (e.g., depth, panoramic views, and top-down geometry) of the same unobserved state. This controlled comparison reveals that even accurate imagined RGB does not guarantee accurate spatial reasoning, and different questions benefit from different representations. Similar trends hold for representations derived from world-model-estimated states. Notably, estimated RGB alone can mislead spatial reasoning, while alternative representations retain complementary benefits. Building on these findings, we introduce GEARS, a lightweight test-time strategy for adaptive representation selection, outperforming existing spatial imagination methods and showing that reasoning about unseen states requires more than generating RGB views. 

---
# FeatureArc: Benchmarking Coding Agents on Evolving Features 

**Authors**: Yunlong Wang, Alyssa Wu, Cuiyun Gao, Jia Feng, Songyuanyi Lu, Zhichao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37434)  

**Abstract**: A software feature usually evolves over time: developers extend its first version as new use cases and requirements arrive, and each extension builds on code written earlier. Most feature benchmarks evaluate a single request on a fixed codebase. Feature evolution requires evaluating successive extensions on an agent's own earlier code and checking whether the feature remains correct. We introduce FeatureArc to evaluate coding agents on evolving features, together with a pipeline that constructs its tasks from repository histories. The benchmark combines dependencies between successive stages of the same feature, explicit requirements and interface contracts, and executable checks across four programming languages. The pipeline validates fail-to-pass (F2P) tests for new behavior and pass-to-pass (P2P) tests for existing behavior on the reference code before and after each stage. We evaluated 10 coding agents on 73 manually reviewed tasks of 2 to 10 stages. Each agent works in one persistent session and workspace without external test feedback or rollback. The evaluation reveals a gap between first-stage and whole-feature success: pooled success falls from 62.7% at the first stage to 28.5% when the whole feature is tested on the final code. Trajectory analysis identifies inherited defects as a common failure mechanism, present in 72.9% of failed later stages. These earlier errors often remain undetected: in stages following a failure, agents leave the earlier error unnoticed about 95% of the time, and their self-tests rarely revisit earlier stages. Evolving a feature autonomously, without human checks between stages, therefore remains a substantial challenge for current coding agents. At Tencent, the pipeline extends internal evaluations to this setting and guides the development of Hunyuan, Tencent's family of large language models. 

---
# Bayes-Adaptive MDPs for Rollout Allocation to Enhance GRPO Training 

**Authors**: Faeze Moradi Kalarde, Armin Toroghi, Nam Ho Koh, Doo Hwan Hwang, Seungjin Kim, Kee-Eung Kim, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2610.37429)  

**Abstract**: Group Relative Policy Optimization (GRPO) has demonstrated strong performance in LLM post-training. However, it allocates a uniform rollout budget to every training question, ignoring variations in question difficulty. As a result, highly difficult questions may receive too few rollouts to generate informative positive reward signals, limiting their contribution to training, while moderately difficult questions often receive more rollouts than necessary. In this work, we formulate the problem of difficulty-aware rollout budget allocation to maximize training utility under the same total rollout budget as standard GRPO. We propose AdaRoll-GRPO, which reformulates rollout allocation as a sequential decision-making problem and leverages a Bayes-adaptive Markov decision process (MDP) to learn an adaptive allocation policy. The learned policy is computed offline and applied during GRPO training without introducing additional online optimization overhead. Extensive experiments on multiple reasoning datasets and LLM architectures demonstrate that AdaRoll-GRPO consistently improves GRPO training performance under a fixed rollout budget, highlighting the effectiveness of adaptive rollout allocation for GRPO in LLM post-training. 

---
# Learning Quality-Controllable Robot Policies with Iterative Hindsight Relabeling 

**Authors**: Sinjae Kang, Dong Kyu Shin, Jaehong Kim, Kamin Lee, Hyoeun Kim, Kaixin Wang, Li Zhao, Seonghyeon Ye, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.37393)  

**Abstract**: Robot policy evaluation often relies on a binary success signal for task completion. In real-world deployment, however, desired execution behaviors are also defined by quality attributes such as execution speed and motion smoothness, which need to be dynamically controlled according to task requirements. To address this, we introduce Quality-Controllable Policy (QCP), a robot policy that enables inference-time control over both individual quality attributes and their combinations. To extend the controllable quality range beyond initial demonstrations, we further propose an iterative hindsight relabeling loop: QCP collects online rollouts under target quality labels, adds successful trajectories to the dataset, relabels the accumulated trajectories in hindsight according to their measured quality, and fine-tunes the policy. Experiments on Robomimic and RoboEval demonstrate that QCP consistently improves both task execution quality and success rates across iterations, producing successful rollouts that exceed the quality of the initial demonstrations. Finally, we validate QCP on real-world manipulation tasks, where controlling speed increases throughput in conveyor operations and controlling transport stability improves success when carrying wine glasses, demonstrating that quality-controllable policies yield practical gains in physical deployment. 

---
# JEPA-Inspired Latent Field Prediction for Sparse-Pilot MIMO-OFDM Channel Reconstruction 

**Authors**: Yuzhi Yang, Brahim Mefgouda, Hang Zou, Lina Bariah, Anis Bara, Yuhuan Lu, Hao Zhang, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2610.37391)  

**Abstract**: Accurate acquisition of full channel state information (CSI) is essential for multiple-input multiple-output orthogonal frequency-division multiplexing (MIMO-OFDM) wireless systems, while dense pilot-based estimation incurs substantial overhead. Directly minimizing the prediction error over raw CSI coefficients is fragile in realistic propagation environments, since small-scale fading, mobility, local scattering, and small spatial perturbations can induce large phase-sensitive variations. Nevertheless, wireless channels contain strong spatial, temporal, and frequency-domain correlations that can be more predictably represented in a latent propagation-field domain. Motivated by this observation, we propose a joint-embedding predictive architecture (JEPA)-inspired field world model, defined as a latent predictive model of the wireless propagation field, for sparse-pilot channel reconstruction. The model maps CSI observations into latent time-frequency tokens, predicts masked or future latent field states from historical multi-band context, and fuses the predicted latent prior with sparse noisy current pilots to reconstruct the full MIMO-OFDM channel. Ray-tracing experiments show that the proposed method improves normalized mean square error (NMSE) over pilot-only and direct reconstruction baselines, supporting latent field prediction as an effective prior for sparse-pilot channel acquisition. 

---
# Channel Selection for Mixture-of-Experts Decoding Needs No Calibration Data 

**Authors**: Geunhyoung Kim, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37386)  

**Abstract**: Mixture-of-Experts (MoE) models are bottlenecked by loading routed-expert weights from memory during interactive decoding at small batch sizes. Because most intermediate channels in a routed expert contribute little to any single token, channel selectors predict which channels matter and read only their weights, typically keeping channels above a threshold calibrated on offline data. We show that channel selection needs no calibration data. First, a calibrated threshold does not fix how much a selector reads, because the number of channels passing it varies across inputs; Prox with thresholds calibrated for an 18% read budget actually reads only 12.9% on average across our benchmarks. Second, top-k keeps each token's k highest-scoring channels and needs no calibration data. At the calibrated budget, a selector's top-k version matches or beats its threshold version on the same score in 40 of 43 comparisons at 13% and 18% budgets, including all 23 on generation tasks. The top-k versions also avoid the repetition-loop collapse that calibrated thresholds cause on Gemma 4. Calibrating a threshold thus provides neither of the two things selectors use it for: a known read budget and a consistent accuracy gain at that budget. We furthermore introduce a reader built entirely from the model's weights, which screens channels with a ternary sketch and recovers the skipped ones with a low-rank map. Across three post-trained MoE models, our reader retains 98.1% of full-model accuracy at 17.8% of routed-expert reads, with Triton kernels running MoE blocks 1.60 to 1.90x faster than optimized vLLM at batch one and up to 2.69x at larger batches on the same three models. 

---
# CoS-VQ: Coordinated Shared-Scale Vector Quantization for Large Language Models 

**Authors**: Zukang Xu, Zhixiong Zhao, Xing Hu, Zhe Jiang, Xin Si, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37336)  

**Abstract**: Large language models (LLMs) require substantial memory, motivating low-bit weight quantization for efficient deployment. Existing vector quantization (VQ) methods often improve accuracy by increasing the effective coding dimension, which can complicate independent lookup and local codeword updates. We propose CoS-VQ (Coordinated Shared-Scale Vector Quantization), which preserves low-dimensional codewords while jointly optimizing hard indices and their shared scale across positions. The key challenge is scale-induced coupling: the optimal scale depends on the entire codeword combination and, in turn, changes candidate rankings across positions, so pointwise updates can stall even after scale re-optimization. CoS-VQ collects candidates at multiple probe scales and constructs a tangent separable upper bound for the residual-aware quadratic objective. By reducing candidate comparisons to lower envelopes of lines over positive scales, it identifies finitely many representative intervals for joint index--scale updates without exhaustively enumerating codeword combinations. On Qwen3.8-27B, CoS-VQ outperforms QTIP and LLVQ on the evaluated language modeling, coding, and reasoning benchmarks. Ablations from the same initialization further show larger gains from joint updates than from scale-only refinement, highlighting joint selection as an alternative to increasing codeword dimensionality. The code is available at this https URL. 

---
# WCC: A Learning-Based Framework for Deciding When to Apply Cooperative Coevolution in Large-Scale Optimization 

**Authors**: Wenjie Qiu, Hepei Li, Jianfeng Wang, Jun Zhang, Yue-Jiao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2610.37329)  

**Abstract**: Cooperative Coevolution (CC) and Non-Decomposition Algorithms (NDAs) are two primary approaches for large-scale global optimization (LSGO). Although both have achieved success, they exhibit markedly different performance across problem types and each faces approach-specific limitations, leaving the question of "when to use CC" unresolved. We address this gap with WCC, a novel learning-based framework that dynamically decides when to deploy CC in LSGO. WCC employs a streamlined design for efficient adaptive learning. We formulate the decision as a Markov decision process, designing a targeted feature set, a robust reward, and a tailored network architecture to provide the agent with rich information about the problem state and optimization progress. Our auto-regressive policy enables WCC to perform fine-grained, dynamic switching between CC and NDAs while conditionally allocating the computational budget without expert specifications. WCC shows strong competitiveness and superior generalization across problems, optimization horizons, and baseline optimizers. It realizes a "train-once and reuse" paradigm under a low budget, significantly curbing LSGO cost. Additionally, in-depth analyses yield insights not captured by existing theory, offering practical guidance on when and how to effectively deploy CC. 

---
# AnalogScout: Simulation-Efficient PVT-Aware Analog Sizing with LLM Agents 

**Authors**: Yichen Long, Keren Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37319)  

**Abstract**: Analog device sizing must satisfy all specifications across process, voltage, and temperature (PVT) corners to ensure tape-out success, but simulating every candidate at all corners dominates the sizing time and slows design iteration. Conventional surrogate-based optimizers search the sizing space directly, while recent large language model (LLM) agents add circuit knowledge to designs but still delegate the sizes to global optimizers. Both learn the circuit response from repeated multi-corner SPICE simulations and optimize on top of it, which becomes expensive under a limited simulation budget. Providing an agent with reusable verified designs and sizing sub-blocks analytically could save most of this budget. We propose AnalogScout, a simulation-efficient PVT-aware sizing framework driven by an LLM agent. Combining elite-design reuse, analytic gm/Id local sizing and global refinement under full-corner verification, AnalogScout reduces power by 43.3% relative to the evolution strategy CMA-ES at an equal budget of 16k simulations on three SKY130 OTAs of increasing complexity, and reaches the final CMA-ES power after 7.0% of that budget. It also attains lower power than Bayesian optimization on all three OTAs and than two re-implemented LLM-driven flows on the two OTAs where they were tested. 

---
# A Lightweight Attention-Based CNN with Saliency-Map Interpretability for Skin Cancer Detection 

**Authors**: Sanjana Islam Kasfia, Md. Ismiel Hossen Abir, AKM Bahalul Haque, Abir Mahmud Shahariar  

**Link**: [PDF](https://arxiv.org/pdf/2610.37314)  

**Abstract**: Skin cancer is among the most prevalent and life threatening forms of cancer worldwide. Early and accurate diagnosis significantly improves patient survival rates. However, manual dermoscopic assessment remains challenging due to visual similarity. This study proposes an Explainable AI-driven framework for automated binary classification of skin lesions using a lightweight attention-based convolutional neural network (CNN-Attention). The model integrates channel attention to focus on informative regions of dermoscopic images while maintaining low computational complexity. A publicly available dataset comprising 2,637 training and 660 test images was used, with balanced class weights. The proposed model achieved 87 percent accuracy with 1.95 million parameters and 7.45 MB of storage, matching the best baseline (MobileNet) while using approximately half as many parameters and half as much storage. ROC AUC and PR AUC scores reached 0.9475 and 0.9430, respectively. Additionally, saliency maps were incorporated as an explainability technique to visualize regions that influence model decisions. Comparative analysis with VGG16,VGG19,ResNet101,and MobileNet demonstrates the proposed models effectiveness in balancing classification results, computational efficiency, and interpretability. 

---
# Where Should the Evaluation Budget Go in LLM-Driven Agent Scaffold Evolution? 

**Authors**: Sichun Luo, Yi Huang, Mukai Li, Haochen Luo, Guanzhi Deng, Haibo Wang, Jian Xu, Zefa Hu, Junlan Feng, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37276)  

**Abstract**: Large language model (LLM)-driven agent scaffold evolution improves agent workflows through iterative candidate generation and evaluation. Evaluation both guides subsequent search and determines the final scaffold, but evaluating every candidate on the full task set incurs substantial execution costs. We investigate how evaluation budgets can be allocated across these two roles. Our experiments show that sparse feedback can sustain productive evolution, while additional targeted evaluation improves final selection. Replay analyses further show that shared informative items and a common scoring reference support comparisons among closely matched candidates. Motivated by these findings, we propose \method, an evaluation allocation framework that combines sparse search-time feedback with targeted final confirmation. A short fully evaluated warm-up establishes a shared informative core and calibrates a response model whose item parameters remain fixed during search. After search, \method refits the model and applies paired successive halving, reusing accumulated observations and executing only missing evaluations. Across four benchmarks and two evolution frameworks, \method achieves higher mean performance than full evaluation in most settings while using approximately half as many evaluation cells. 

---
# Aligning Large Language Models for Safety in African Languages: A Case Study of Hausa 

**Authors**: Valdini Douglace Lemofouet, Blessing Ngozi Uzor, Paula Chikaodinaka Anyanwu, Danielle Blanche Kapsa, Saminu Mohammad Aliyu, Seid Muhie Yimam, Shamsuddeen Hassan Muhammad  

**Link**: [PDF](https://arxiv.org/pdf/2610.37245)  

**Abstract**: Safety alignment research for large language models (LLMs) remains concentrated on high-resource languages, leaving open how existing alignment methods perform in low-resource, culturally distinct settings. We study this question for Hausa, a language spoken by more than 200 million people across West Africa. We introduce HausaSafeLM, one of the first Hausa safety preference dataset, built from 10,000 prompts derived from PKU-SafeRLHF: 3,000 human-corrected (gold) and 7,000 machine-translated (silver) examples spanning six harm macro-categories. Using Qwen2.5-7B-Instruct, we compare supervised fine-tuning (SFT) and SFT followed by Direct Preference Optimization (DPO) under both data regimes, evaluating on 500 harmful and 200 benign Hausa this http URL find that the untuned base model appears highly safe by refusal-rate metrics alone, but this reflects limited Hausa generation ability rather than genuine alignment: its refusals are almost uniformly low quality. SFT improves both refusal fluency and precision and benefits substantially from the larger, noisier silver-tier data, while DPO sharpens refusal precision but comes at a steep, largely data-invariant cost to linguistic quality. No single configuration jointly maximizes safety, helpfulness, and fluency. Our results suggest that current alignment techniques teach models when to refuse more readily than how to do so fluently in underrepresented languages, and that refusal-rate metrics alone are an unreliable signal of safety in low-resource settings. 

---
# When Is a Relation Worth Trusting? Manipulation Costs, Exposure, and the Trust Trap in Multi-Relational Graph Learning 

**Authors**: Vincess Dongmo, Astride Melvin Fokam Ninyim  

**Link**: [PDF](https://arxiv.org/pdf/2610.37239)  

**Abstract**: Multi-relational graph detectors learn from clean data how much to trust each relation, but an adversary pays a different price to manipulate each relation and each action; taking such costs as given, we ask when using a relation is worse than ignoring it. Our first main result proves that degree gating strictly beats every selection or reweighting of a relation when part of the population is isolated on it and the attacker cannot afford to open the gate, and that this window is sharp; on real graphs, validated gating gains exactly on the relations that meet these conditions and nowhere else. Our second shows that a per-node exposure bound, which extends to message passing, tells a platform which relation to harden: exposure per unit cost, computed without any attack, picks the relation whose hardening raises the exact trap threshold most in 15 of 16 cases, against at most 7 for informativeness, cheapness, or an analysis with equal edit costs. Supporting tools give exact worst cases for linear relational models, certified brackets for R-GCN and tree ensembles, and a finite-sample guarantee for adversarial validation, under which clean-trained graph detectors fall below their graph-free counterparts in 58 of 90 settings on six graphs. A selection protocol built on these results is never significantly worse than ignoring the graph; with at most four relations it mostly abstains, as validation alone would, and its cost-aware proposals help when relations are many, so far shown on synthetic graphs. Without cost information, the safe choice is to abstain. 

---
# Spine: Compiling Subtask Trees into Structured Context for Long-Horizon Coding Agents 

**Authors**: Jiahong Xiang, Kunqiu Chen, Lingming Zhang, Yuqun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37224)  

**Abstract**: Long-horizon coding tasks contain nested subtasks. Typically, agents record interactions sequentially into a growing history, causing context inflation and performance degradation. Compressing or trimming the history without explicit subtask boundaries can discard information needed by unfinished subtasks. Conversely, recursive subagent delegation isolates completed subtasks through child agents by assigning every subtask boundary to delegation, which conflates subtask granularity with executing agent choice, adding execution and coordination overhead. We introduce Spine, an interface and runtime that align model context with subtask nesting and completion. An agent declares explicit subtask boundaries through work scopes, allowing the same agent to execute nested subtasks inline while reserving delegation for independent parallel tasks. The runtime organizes execution in SpineTree, a persistent tree recording subtask relations, local interactions, completion states, and returned results. Before each model call, it compiles the tree into model context just in time, retaining unfinished subtasks' interactions and replacing completed subtasks' details with their results, while preserving prefix-cache reuse.
We integrate Spine into Codex to build SpineCodex. With GPT-5.6 Sol, compared with native Codex, SpineCodex improves SWE-Milestone Score by 11.8% while reducing total cost by 32.4%, improves the average ProgramBench test-pass rate by 17.3%, and improves Mean@5 on eight of the ten evaluated FrontierSWE tasks. On SWE-Milestone, it also achieves a comparable Score to Recursive Subagent at 73.7% lower total cost. We also integrate Spine into Pi through a reusable software package: with Gemini 3.8 Flash and Qwen 3.8 27B, it improves SWE-Milestone Score by 9.5-17.5% and the average ProgramBench test-pass rate by 4.3-8.7% compared with native Pi. 

---
# Retrieve to Act: Motion Primitive Graph Retrieval for Robust VLA Execution 

**Authors**: Pham Tri Quang, Bao-Ngoc Dao, Minh Duc Nguyen, Duc-Duy Nguyen, Hung T. Ho, Linh Dang Le, Trong-Bao Ho, Thien-Loc Ha, Tan Q. Nguyen, Tuan Dam, Vu N. Duong, Daniel Sonntag, Ngan Le, Khoa D. Doan, Jan Peters, An Thai Le, Duy M. H. Nguyen, Vien Anh Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2610.37215)  

**Abstract**: Vision-Language-Action (VLA) policies demonstrate remarkable semantic generalization, yet remain highly vulnerable to out-of-distribution (OOD) visual shifts during execution. While existing test-time methods attempt to suppress spurious visual cues, they fundamentally overlook the rich, uncorrupted kinematic priors embedded in the policy's own training experience. We introduce Retrieve to Act (R2A), a test-time scaling framework that formulates OOD robustness purely as a geometric motion primitive retrieval problem. R2A bypasses fragile visual features by structuring successful training demonstrations into an offline spatiotemporal motion primitive graph. During inference, a query-dependent Graph Neural Network dynamically localizes the robot's ongoing execution within this graph, retrieving structurally compatible future motion primitives. These retrieved geometric priors are then seamlessly harmonized with the frozen VLA's predicted actions via a learned, dimension-wise gating mechanism, adaptively rectifying erroneous spatial trajectories while preserving the policy's local reactivity. Extensive experiments across three state-of-the-art VLA architectures, GR00T N1.7, $\pi_{0.5}$, and StableVLA, demonstrate that R2A consistently and significantly enhances robustness against diverse contextual perturbations. R2A achieves absolute success rate improvements of up to +12% in simulation benchmarks (LIBERO-Plus, RoboTwin2.0-Plus) and +40% in real-world deployments, without requiring any finetuning of the base policies. Project page: this https URL 

---
# Action-Value Calibration via Continuation Sharing for Long-Horizon Agentic Reinforcement Learning 

**Authors**: Yingjian Chen, Lang Zhou, Zixin Zhao, Pingfan Li, Weishi Zheng, Zhilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.37195)  

**Abstract**: Reinforcement learning for long-horizon agents relies on feedback from sequences of interactions, making credit assignment to individual actions difficult. Existing methods such as GiGPO compare actions at shared states to compute relative advantages, but estimate each action's value from a single sampled continuation. The resulting local credit remains sensitive to decisions made later in the trajectory. To address this problem, we propose ConCal (Continuation Calibration), a recursive continuation-value estimator that incorporates continuation evidence already present in a rollout batch. ConCal matches visits from the same task by their incoming state-action-state transitions, combines shared continuation evidence with each visit's own transition estimate, and propagates calibrated values backward to refine earlier action-value estimates. This reduces dependence on individual sampled continuations and mitigates local credit distortions without additional rollouts. Held-out return diagnostics show lower error in within-group value differences and higher agreement with reference action orderings than estimates based on individual continuations. With Qwen2.5-1.5B-Instruct, ConCal improves success rates over GiGPO by 3.39 and 5.86 percentage points on ALFWorld and WebShop, respectively. Our code is available at this https URL. 

---
# MLE-World: Predicting Machine Learning Experiment Outcomes with a Learned World Model 

**Authors**: Brian Ko, Han Wang, Hejia Geng, Li Erran Li, Sherry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37175)  

**Abstract**: AI research agents can propose, execute, and improve programs that solve machine learning tasks. However, executing each candidate program requires training and evaluating ML models, limiting how many candidates an agent can explore at each step. We introduce MLE-World, a world model finetuned from Qwen3.5-4B that predicts a program's terminal output (error traces, training logs, and validation scores) from its code and data description without executing it, together with a confidence score estimating the prediction's correctness. Because a prediction costs one inference call rather than a training run, MLE-World makes test time scaling (TTS) affordable for these agents. We build five TTS policies, guided by MLE-World, that compare several candidates per step by their predicted outputs: Sequential revision, Best-of-N, Pairwise judging, Fusion, and a UCB style rule that also adds an exploration bonus from this confidence score. Since predictions can be wrong, we run a candidate for real only when its confidence score is low. On 17 MLE-bench Lite tasks, MLE-World matches the real terminal output on 77.9% of held out programs, versus 41.1% for much larger prompted models. MLE-World guided TTS achieves up to 22.1% higher observed mean performance than the agent without a world model, with less real execution. Project webpage: this https URL. 

---
# GCSR-OCC: Geometry Calibration and Semantic Routing for Open-Vocabulary Occupancy Prediction 

**Authors**: Bohua Wang, Congyang Zhao, Tianxiong Lv, Chenxu Zhao, He Zheng, Hangqi Fan, Leigang Luo, Zhihao Liao, Yihong Guo, Yang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.37149)  

**Abstract**: Open-vocabulary 3D occupancy prediction represents scene geometry and semantics in a unified voxel grid, enabling recognition beyond predefined categories. One approach reconstructs scenes with 3D Gaussians and converts them into voxel occupancy. However, Gaussian scale inflation and excessive overlap can distort voxel occupancy predictions, while semantic mixing can arise within Gaussians spanning multiple categories and during 3D--2D semantic feature alignment. We propose GCSR-OCC, which combines geometry calibration and multi-mode query-conditioned semantic routing for indoor occupancy prediction from a single RGB image. The Evidence-Conserving Gaussian-to-Occupancy operator (EC-G2O) uses kernel mass normalization, $p$-norm aggregation, and a learnable gain to suppress scale inflation and excessive local evidence accumulation while calibrating occupancy responses. Multi-mode Query-First Gaussian Semantic Routing (MM-QGSR) matches multiple semantic modes per Gaussian to text queries and aggregates query-specific evidence into voxels through a learnable text-conditioned router. Fine-grained subclass queries and smoothed semantic aggregation further refine voxel-level predictions. Without 3D semantic annotations, GCSR-OCC achieves state-of-the-art performance in open-vocabulary occupancy prediction on Occ-ScanNet, with 63.31% geometric IoU and 27.44% semantic mIoU. It also reduces geometric aggregation latency from 19.76 ms to 12.88 ms. Our code will be made publicly available. 

---
# Beyond Parameter Protection: Fisher-Shaped EWC for Replay-Free Continual Learning 

**Authors**: Liying Wang, Simone Magistri, Joost van de Weijer  

**Link**: [PDF](https://arxiv.org/pdf/2610.37145)  

**Abstract**: In replay-free continual learning, Elastic Weight Consolidation (EWC) uses Fisher information matrix (FIM) to protect acquired knowledge by penalizing changes to historically important parameters. However, a local importance estimate at the retained solution does not in general characterize sensitivity after finite parameter displacement. This motivates controlling Fisher sensitivity over a neighborhood of potential future updates. We propose Fisher-Shaped EWC (FS-EWC), which extends the role of FIM from parameter protection to solution formation for future retention. During task learning, FS-EWC minimizes the worst-case empirical FIM trace within a prescribed neighborhood, explicitly favoring solutions that are less sensitive to bounded future displacement. At task completion, trajectory-based model merging constructs the retained solution from multiple optimization states rather than a single terminal iterate, after which FIM is estimated for subsequent consolidation. Experiments across multiple benchmarks show that FS-EWC improves performance over EWC, while controlled studies indicate a more favorable stability--plasticity balance and greater tolerance to parameter drift. These results suggest that effective knowledge retention requires both forming solutions with future retention in mind and protecting acquired knowledge. 

---
# Generative Replay for Equilibrium Propagation: Toward Edge-Friendly Continual Learning in Hopfield Networks 

**Authors**: Victor Cook, Gilles Sassatelli, Thierry Gil, Mohamed Watfa, Marina Reyboz  

**Link**: [PDF](https://arxiv.org/pdf/2610.37139)  

**Abstract**: Equilibrium Propagation (EqProp) offers a biologically grounded and hardware-aligned alternative to backpropagation for training Hopfield Networks. However, its behavior in continual learning settings remains largely unexplored. This work confronts catastrophic forgetting in EqProp-trained Hopfield Networks from the inside out, eliminating the need for external data buffers or architectural overhead that standard Experience Replay \textbf{(ER)} and Generative Replay \textbf{(Gen-Replay)} inevitably require. We introduce an Energy-Based Generative Replay strategy \textbf{(EB-Gen)} which exploits the intrinsic generative dynamics of Hopfield Networks, allowing a single model to act as both classifier and replay samples generator. Samples capturing past task knowledge are generated by relaxing a frozen copy of the network, enabling rehearsal solely through internally generated pseudo-data. We evaluate the proposed method in the Class-Incremental setting, using the split versions of the MNIST and CIFAR10 benchmarks and compare it with FineTune, Experience Replay, Generative Replay, Sleep-Like Replay Consolidation (SRC) and Joint training. Our results show that EB-Gen significantly improves stability while preserving plasticity, achieving a 75\% and 64\% average accuracy on Split MNIST and Split CIFAR10 respectively. These findings show that Equilibrium Propagation can be adapted to alleviate catastrophic forgetting without forgoing the traits which made it compelling in the first place. 

---
# What the Primary Lund Plane Omits: A Model-Stitching Diagnosis of a Jet-Tagger Representation Gap 

**Authors**: Saurabh Rai, Sanmay Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2610.37137)  

**Abstract**: Lund-plane and Particle Transformer (ParT) jet taggers can differ in performance because one representation is weaker, or because a weaker downstream computation reads an adequate one. Standalone performance cannot separate these. Model stitching can: it links the frozen early layers of one network to the frozen late layers of another through a single trained linear map. As a positive control we use an incomplete Lund representation, the primary declustering branch, whose deficit is known independently. Stitched into ParT's downstream, it reaches a background rejection at least as much as its own downstream recovers, yet far below ParT's background rejection: the deficit is representational, not computational. A hypergraph reader over splitting triplets does no better, ruling out higher-order correlations as the cause. The gap is invisible in AUC, which stays within 1\% of its target value, and appears only in background rejection at low signal efficiency. It localizes to 0.6\% of background jets, at and above the top mass, with more constituents but fewer primary splittings than typical background. Rebuilding the representation on the full Cambridge/Aachen declustering tree repairs the deficit as predicted: standalone rejection rises, closing 72\% of the gap. Masking nodes at inference shows the repaired network still relies almost entirely on the primary branch; the secondary splittings make the representation learnable rather than supplying information it reads. Along the primary branch, the widest-angle splitting, near the top and $W$ opening angles, is worth nearly twice a random one. For quark/gluon tagging the primary branch is far less deficient, retaining 77.6\% of ParT's rejection (46.9\% for top), and the full tree closes only 33\% of the gap (72\% for top). 

---
# MRI-to-CT Translation of Spinal Images via Anatomy-Guided Pseudo-Pairing and Adversarial Diffusion 

**Authors**: Eduardo Nieto, Ibai Genua, Davide Scorza, Karen López-Linares  

**Link**: [PDF](https://arxiv.org/pdf/2610.37134)  

**Abstract**: Magnetic Resonance Imaging (MRI) is routinely used to diagnose spinal disorders, but surgical planning often requires Computed Tomography (CT) for better visualization of vertebral bone anatomy. Synthesizing CT from MRI could avoid additional radiation exposure and streamline preoperative assessment.
We propose a three-stage framework for spine MRI-to-CT synthesis. First, an anatomy-guided vertebral matching system identifies similar sagittal MRI and CT slices using spine segmentations and aligns them through rigid registration, creating pseudo-paired data from unpaired scans. Second, these pairs train an adversarial diffusion model for MRI-to-CT translation. Third, synthesized slices are assembled into 3D volumes and refined by a network that improves inter-slice continuity and structural coherence.
The framework was trained on over 10,000 unpaired sagittal MRI and CT slices and evaluated on 1,911 slices from 83 MRI volumes spanning all spinal regions. Assessment combined distribution-based measures with vertebral segmentation metrics. Compared with diffusion alone, the proposed approach increased binary 3D vertebral Dice from 0.666 to 0.739. Volumetric post-processing reduced visible discontinuities between stacked slices and improved vertebral coherence in axial and coronal views.
Combining anatomy-guided pseudo-pairing, adversarial diffusion and 3D refinement improves spine MRI-to-CT synthesis. Vertebral matching primarily enhanced anatomical alignment and fidelity, while volumetric refinement improved 3D consistency. These findings support further development of MRI-derived CT-like visualization for preoperative planning, particularly to preserve fine intensity details while maintaining volumetric coherence. 

---
# A First-Step Bias Restores Lost Diversity in Diffusion Model Distillation 

**Authors**: Wenwu Tang, Dong Wang, Yun Cheng, Olga Saukh  

**Link**: [PDF](https://arxiv.org/pdf/2610.37110)  

**Abstract**: Distribution-matching distillation enables high-quality image generation in only a few denoising steps. However, distilled student models often map different initial noise samples to similar outputs under the same prompt. To locate where diversity is lost, we compare the teacher and student along the sampling trajectory. The largest measured reduction in the response to different initial noises occurs during the first, high-noise step. When we replace the first step with the teacher's corresponding trajectory and leave the final three student steps unchanged, most of the output diversity returns. This suggests that the first step is a key point for restoring diversity. Motivated by this finding, we introduce \emph{ReSeed}, a post-distillation method that adds a seed-response bias adapter only to the first student step and aligns its velocity with the teacher's first-interval secant velocity. All student weights and later denoising steps remain unchanged. We evaluate ReSeed on three text-to-image models (SD3-Medium, SD3.5-Medium, and FLUX.1-dev) and one text-to-video model (Wan2.1-T2V-1.3B). For a distilled 4-step FLUX.1-dev model, \emph{ReSeed} requires only $0.00196\%$ additional trainable parameters relative to the base model and is trained on 8,000 prompt--noise pairs, while keeping the distilled student fully frozen and requiring no target images, perceptual losses, or adversarial losses. It improves DINO diversity by $48.3\%$ while maintaining competitive visual quality. These results show that substantial diversity can be recovered through a lightweight, data-efficient tuning of only the first denoising step, without finetuning the full student model. 

---
# Hardware-Matched Spiking Neural Networks for Real-Time Purkinje Cell Spike Classification 

**Authors**: Arooba Farhan, Afsah Syed, Muhammad Fahd Ibrahim Bhatti, Muhammad Areeb Khalid Kidwai, Muhammad Ahsan Salar Khan, Vincenzo Romano, Muhammad Ali Siddiqi  

**Link**: [PDF](https://arxiv.org/pdf/2610.37104)  

**Abstract**: Distinguishing Purkinje cell (PC) simple spikes (SSs) from complex spikes (CSs) is key to studying cerebellar function. PC recordings commonly rely on wired headstages, whose tethers restrict natural movement in freely behaving animals. Untethered headstages remove this constraint, but cannot continuously transmit raw recordings within their power budgets, making on-headstage spike classification an attractive form of compression, provided it runs accurately in real time on constrained hardware. This work presents two compact spiking neural networks (SNNs) matched to their target hardware: ConvSNN, which maps to the Matrix Vector Processor (MVP) of an EFM32PG28 microcontroller, and StreamSNN, which processes one waveform value per timestep using learned time constants and fixed synaptic delays for a dedicated SKY130 circuit. Evaluated on cells unseen during training, ConvSNN achieves a CS F1 of 0.873 with 1,018 parameters and StreamSNN 0.837 with 527 parameters, both with eight-bit weights. On the microcontroller, MVP acceleration reduces ConvSNN latency by 62.3\% and extends estimated battery life to 33.73~hours on a 50~mAh battery. In SKY130, StreamSNN occupies 0.173~mm$^2$ and consumes 0.46~$\mu$J per event, with 0.2~ms of computation within an 8.3~ms end-to-end latency. 

---
# Learning to Look Beyond Screens: Transferable Visual Observation from GUI Grounding 

**Authors**: Ruilin Yao, Xinyu Qiu, Tianyu Zou, Yi Rong, Shengwu Xiong, Shuheng Shen, Changhua Meng  

**Link**: [PDF](https://arxiv.org/pdf/2610.37093)  

**Abstract**: Visual perception is evolving from passive recognition toward active, task-oriented observation. We investigate GUI grounding as a scalable source of spatial supervision for learning observation behaviors that transfer beyond interfaces. We introduce LBS (Look Beyond Screens), a vision-language model trained through supervised initialization on teacher-generated GUI grounding trajectories, followed by reinforcement learning for multi-turn crop-and-inspect decisions. Under fixed inference settings, LBS models based on Qwen3.5-35B-A3B and Qwen3.8-27B consistently improve over their corresponding baselines. Gains beyond supervised fine-tuning transfer to PerceptionBench for both backbones and to HR-Bench and ChartQAPro for Qwen3.5-35B-A3B, supporting observation strategies that generalize beyond GUI-specific imitation. Further analysis suggests complementary roles for the two stages: SFT establishes visual inspection primitives, while RL improves their downstream effectiveness. These findings support GUI grounding as a practical supervision paradigm for transferable active visual perception and broader multimodal reasoning. Our code and model are publicly available at this https URL and this https URL, respectively. 

---
# EnergyCal: Taming Loss Spikes in Muon-Family Optimizers for Sparse MoE Training 

**Authors**: Masato Fujitake  

**Link**: [PDF](https://arxiv.org/pdf/2610.37062)  

**Abstract**: Sparse mixture-of-experts (MoE) language-model training can exhibit large loss spikes with Muon-family optimizers. We introduce EnergyCal, a lightweight rule that attenuates routed-expert updates according to the normalized squared Frobenius norm of their finite-step Newton-Schulz directions. EnergyCal leaves the spectral transform unchanged and adds no persistent optimizer state. We evaluate it on a 4B-parameter MoE model. Against a development-selected, lower-learning-rate MuonH recipe, EnergyCal-MuonH reduces mean thresholded loss-spike burden by 99.51% and final geometric-mean perplexity by 4.06%, with a modest throughput cost. With plain Muon at matched nominal learning rate and weight decay, the reduction in mean spike burden averages 95.98% across the two tested exponents, accompanied by small endpoint improvements. These findings support stabilization both with and without Hyperball in the evaluated setting. 

---
# Exploring the Limits of the Delta Learning Hypothesis: Single-Pair Generator Biases in Direct Preference Optimization 

**Authors**: Giuseppe Curci, Stefano Simonazzi, Andrea Molinari, Andrea Zugarini  

**Link**: [PDF](https://arxiv.org/pdf/2610.37058)  

**Abstract**: Direct Preference Optimization (DPO) typically relies on preference pairs generated from diverse model pools and ranked by an external LLM judge. The recently proposed delta-learning hypothesis suggests that comparable supervision can instead be obtained from a single pair of models by treating the larger model's response as preferred. We investigate the limits of this approach across model families and scales. We show, both theoretically and empirically, that fixed model pairs introduce systematic, generator-dependent biases that are predictive of the preference label independent of response quality. Consequently, DPO trained on single-pair data consistently underperforms multi-model preference data, despite the larger generator producing higher-quality responses, and we trace this gap to training dynamics consistent with reward hacking on these biases. We further show that simply increasing the number of generators, even without an external judge, is enough to partially mitigate this effect. Our results highlight generator diversity as a critical factor in preference-data construction. 

---
# Blindfold: Keeping the Audit Coin Secret from LLM Agents 

**Authors**: Wei Zhu, Meiyun Zuo, Huiping Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.37055)  

**Abstract**: Audit-based certification bounds the violation rate of an LLM agent by auditing a random fraction $\varepsilon$ of actions and reweighting the audited labels; its validity is meant to rest only on the defender's audit coin. We show that it also rests on a confidentiality property: the coin must not interfere with the agent's view before the action commits, at a time the agent chooses. We call this pre-commit noninterference (PNI) and measure its violation by $D_{\mathrm{A}}$, the total-variation distance between the agent's views under the two coin values. PNI is necessary and sufficient for an unbiased certificate against adaptive agents, the worst-case bias is $(1-\varepsilon)D_{\mathrm{A}}$, and no procedure that sees only audited labels can pay less. Leakage composes across steps, a resource shared across episodes opens a channel of computable size, and raising the audit rate cannot restore validity. On a monitor we built, a silent audit changes no output but adds latency that tool timestamps expose, and the agent can enlarge it. A real LLM agent follows an injection that names a visible audit notice (generic, tool, and timing clauses have no effect); over that channel the plain certificate fails on measured labels and, by a post-hoc analysis, in a live run on a pre-registered schedule, while a silent audit restores it. A fresh coin per audited call closes the timing channel; padding only narrows it. 

---
# Zooming without Zooming v2: From Seeing Details to Connecting Evidence for Generalizable Multimodal Perception 

**Authors**: Lai Wei, Yatai Li, Liangbo He, Jun Lan, Renjie Gu, Linhan Cao, Junkai Lin, Huijia Zhu, Weiqiang Wang, Linghe Kong, Yue Wang, Zhuosheng Zhang, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37037)  

**Abstract**: Multimodal perception forms the foundation of multimodal intelligence. In particular, spatial perception and multimodal long-context understanding are two pivotal tasks that underpin embodied AI and multimodal agents. However, synthesizing targeted, high-quality training data for these tasks remains difficult and costly. Our key insight is that these tasks share an underlying capability, which we call connecting evidence: identifying relevant visual facts, binding them to the correct entities, recovering their relations, and composing them into an answer. Crucially, targeted training data are far easier to synthesize for this capability than for these tasks themselves. To strengthen this capability, we propose EvidenceConnect, an easy-to-implement data synthesis framework that uses zoomed views of related regions within a single image to generate questions requiring comparison, aggregation, or chaining of visual evidence. To ensure quality, a reference answer is retained only when independent VLMs agree on it, and region-ablation tests discard questions whose answer survives the removal of selected regions. We then train base VLMs on the resulting data with RLVR, obtaining Zooming without Zooming v2 (ZwZ2). Although the synthesized data are not tailored to any downstream benchmark, ZwZ2 improves fine-grained and spatial perception, with further gains on embodied spatial reasoning, multimodal long-context understanding, multimodal agent tasks, and general visual reasoning. These results establish EvidenceConnect as an effective framework for supervising evidence connection and suggest that strengthening this shared capability leads to more generalizable multimodal perception. 

---
# Interpretable Deep Learning Models for Breast Cancer Diagnosis Using Medical Imaging 

**Authors**: Sami Alnawali, Muhammad Atif Saeed, Abdul Hadi, Azka Atiq, Akhtar Jamil, Ahmad Din  

**Link**: [PDF](https://arxiv.org/pdf/2610.37026)  

**Abstract**: Deep learning shows promise for breast cancer diagnosis, but limited interpretability hinders clinical adoption. We benchmark ten ImageNet-pretrained models (ResNet50, VGG19, InceptionV3, DenseNet121, Xception, MobileNetV2, EfficientNet, ConvNeXt, ViT, and Swin Transformer) for benign-versus-malignant classification on the BreakHis histopathology dataset using a patient-level split, and explain their predictions with Grad-CAM, LIME, Integrated Gradients, and LRP. DenseNet121 performed best, with 99.4% test accuracy and an AUC-ROC of 0.999, and significantly outperformed all other models (p < 0.001). Transformer-based models trailed the CNNs. The explanations consistently highlighted nuclei-rich and glandular regions of diagnostic relevance. Combining transfer learning with explainable AI can thus yield accurate, interpretable tools for breast cancer diagnosis. 

---
# Crop Yield Prediction Using Regression Models: A Comparative Evaluation of Model Performance 

**Authors**: Saptak Das, Arijit Sarkar, Soumik Neogi, Suborno Naskar  

**Link**: [PDF](https://arxiv.org/pdf/2610.37022)  

**Abstract**: Reliable crop-yield prediction can support agricultural planning and resource allocation. This study evaluates five regression models for predicting crop yield from environmental, crop-related, and agricultural management variables. The dataset initially contained 1,000,000 observations and ten variables, with 231 observations having negative target values removed during preprocessing. Categorical variables were represented using one-hot encoding, and Boolean variables were converted into binary values. The models considered were multiple linear regression (LR), random-forest regression (RF), decision-tree regression (DT), ridge regression, and lasso regression. Their performance was evaluated using 80:20 and 70:30 training-testing splits based on the coefficient of determination (R-squared), mean squared error (MSE), and root mean squared error (RMSE). Applying an 80:20 split, LR achieved a test R-squared of approximately 0.9130 and a test RMSE of 0.5008. Applying a 70:30 split, lasso regression yielded the highest test R-squared (approximately 0.9131) and the lowest test RMSE (approximately 0.5008), with performance almost identical to LR and ridge regression. DT also showed improved test performance under the 70:30 split, whereas RF exhibited a substantial gap between training and testing performance. These results indicate that the evaluated linear regression model provides competitive baselines for this dataset. Further experimentation, including validation testing, will help determine the models' performance and predictability. 

---
# V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models 

**Authors**: Miso Choi, Daekeun Kim, Eunji Kim, Jungbeom Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.37019)  

**Abstract**: Embedding-level scaling expands the capacity of language models through lookup-based memory rather than by increasing model depth, width, or dense computation. Extending this idea to vision-language models (VLMs) is challenging because visual features are continuous, spatially structured, and query-dependent, unlike discrete text tokens that naturally support n-gram lookup. We introduce V-LUMEN, a visual lookup memory for VLMs. V-LUMEN discretizes visual features with a frozen codebook, builds spatially aware memory keys through spatial aggregation-based hashing, and modulates them with text-conditioned relevance signals; all addressing is non-parametric, and the retrieved embeddings are fused into the backbone through a gated residual pathway. On five reasoning-intensive benchmarks, V-LUMEN substantially improves a frozen Qwen3-VL-2B backbone and transfers zero-shot to three benchmarks whose images are disjoint from the training data. At a matched parameter budget, moving part of a LoRA or MoE adapter's capacity into V-LUMEN preserves accuracy on two backbones, indicating that external visual memory is an independent capacity axis rather than a stronger adaptation method. Because lookup cost is constant in table size and incurred only once at prefill, this capacity grows in storage rather than in per-token compute. Code is available at this https URL. 

---
# Evaluating Priority Dispatch for Premium Membership in On-demand Taxi-sharing 

**Authors**: Nozomi Yamamoto, Hiroki Sakaji, Itsuki Noda  

**Link**: [PDF](https://arxiv.org/pdf/2610.37011)  

**Abstract**: In this paper, we analyze the dispatch-level feasibility of introducing premium membership into an on-demand taxi-sharing service using a sequential optimal insertion method. We propose two dispatch-level methods: deadline shortening for premium users, and deadline shortening with dedicated premium vehicles. We compare them with a baseline dispatch method through SUMO-based simulations under three demand scenarios: steady uniform demand, time-varying demand with a peak period, and urban-focused demand with concentrated peak traffic. Our results show that the suitable priority mechanism depends on operating conditions and premium ratio, suggesting that condition-dependent method switching may provide premium benefits while keeping non-premium-user burden smaller for overall optimization. 

---
# LearnPrunerV2: Learnable Progressive Token Pruning for Vision-Language Models 

**Authors**: Rinyoichi Takezoe, Yaqian Li, Weiren Wang, Zihao Bo, Huihong Shi, Anzhou Hou, Mo Guang, Jiale Yan, Kaiwen Long  

**Link**: [PDF](https://arxiv.org/pdf/2610.36986)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated remarkable multimodal capabilities but suffer from prohibitive computational overhead due to the massive number of visual tokens. While token pruning is a promising solution, existing methods still suffer from two major drawbacks: 1) unreliable pruning criteria that fail to accurately localize task-specific regions due to inherent attention biases or query-agnostic designs, and 2) suboptimal pruning strategies that employ rigid, non-adaptive schedules, overlooking the dynamic evolution of token redundancy across different model depths. To address these challenges, we propose LearnPrunerV2, which extends the previous two-stage heuristic framework (LearnPruner) into a fully learnable architecture. Specifically, we integrate lightweight learnable modules into every layer of the VLM backbone, which estimate token importance by fusing intrinsic visual saliency with extrinsic query relevance captured via a shadow token. By uncovering a consistent staircase redundancy pattern across VLM layers, we further introduce Budget-Deterministic Pruning (BDP) to establish a hardware-friendly schedule with fixed per-layer budgets. Extensive experimental results across various benchmarks and VLMs demonstrate that LearnPrunerV2 consistently outperform previous state-of-the-art methods. Notably, LearnPrunerV2 preserves 99.8% of the original performance when pruning 77.8% of visual tokens on LLaVA-v1.5, achieving a superior accuracy-efficiency trade-off. 

---
# Advancement or Acceleration? How Data Curation Improves LLM Training 

**Authors**: Jiaxing Wang, Zicheng Zhang, Deping Xiang, Jin Xu, Yong Li, Chao Liu, Pengzhang Liu, Qixia Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36983)  

**Abstract**: While data curation is widely credited with improving large language model training, a basic question remains unresolved: does curation expand what a model can ultimately learn, or merely reach the same solution faster? In this paper, we formalize this distinction between advancement (improving the attainable performance) and acceleration (reaching the same performance with less data). Through an analytic characterization and real-world LLM experiments, we show that advancement requires a correctable train-target gap. When the baseline training and target objectives coincide, no curation policy improves the endpoint, although curation may still accelerate training. This reframes advancement as a property of the curation task rather than the curation method. Under a common gap-centric view of data mixture and selection, we operationalize this condition as the $\Delta$-test, an observable discrepancy between the uncurated baseline's training and target losses that is computable prior to expensive curation. Across extensive experiments, an appreciably nonzero $\Delta$ predicts persistent gains, whereas gains near $\Delta=0$ tend to diminish as the training budget grows. Beyond individual methods, our findings recenter data curation on the task itself and point to target-set construction as a promising direction for shaping what curation can achieve. 

---
# PailitaoMMSearch2.0 Technical Report: Scaling Generative Retrieval to Unified End-to-End Multimodal Search 

**Authors**: Xiaohan Ye, Zihan Gong, Jingqian Zhao, Lianyu Du, Yunmeng Shu, Xu Chen, Zesheng Zhou, Renchun You, Bocheng Chen, Jinsong Lan, Haihong Tang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.36958)  

**Abstract**: Generative retrieval models have been increasingly adopted in search and recommendation systems. However, industrial generative retrieval models are commonly specialized to a fixed online scenario: they learn from scenario-specific interactions, support a narrow set of query forms and retrieval objectives. We study how to remove these couplings and build a unified end-to-end multimodal search system for open-ended image--text queries, heterogeneous shopping intents, and online scenarios with different resource constraints. We present \textbf{PailitaoMMSearch}, which directly generates product Semantic IDs from a raw query image and free-form instruction and scales along three dimensions. \textbf{Data scaling} uses a reusable toolset that can be composed into intent- and lifecycle-specific workflows for cold-start synthesis, quality control, real-traffic refinement, and continuous data production. \textbf{Intent scaling} progressively equips one model with e-commerce product knowledge, multiple retrieval relations, and adjustable reasoning budgets through continual pre-training (CPT), mixed-budget supervised fine-tuning (SFT), and unified reinforcement learning (RL). The RL stage combines intent-balanced optimization and Sample--Beam rollouts aligned with multi-candidate serving. \textbf{Scenario scaling} exposes quality--cost operating points across model capacity and reasoning budget. The system is deployed in multimodal search applications on Taobao Pailitao. A controlled online A/B test reports statistically significant improvements, including item page views $\textbf{+17.90\%}$, transactions $\textbf{+27.04\%}$, and gross merchandise value $\textbf{+8.94\%}$. 

---
# Dense Representation Learning and Differentiable Shape Modeling for Spine Landmark Detection 

**Authors**: Augustin Daridon, Sofiane Sifaoui, Ayman Assi, Mohamad Karam, Wafa Skalli, Elsa Angelini, Loic Le Folgoc  

**Link**: [PDF](https://arxiv.org/pdf/2610.36951)  

**Abstract**: Spine landmark detection from spinal X-ray is a key step for estimating spine deformation severity, for building biomechanical models, and for pre-operative planning. We reframe anatomical landmark detection as a pixel-level contrastive representation learning problem, matching dense pixel embeddings with landmark prototypes. Surprisingly, even with a generic U-Net backbone, this formulation results in more discriminative heatmaps compared to the traditional heatmap regression paradigm, leading to improved discrimination between vertebral corners on distinct vertebra. Furthermore, we combine the resulting heatmaps with learnable shape priors over the global configuration of spinal landmarks to resolve ambiguities in landmark locations, yielding a simple and practical approach to spinal landmark detection. Our approach is competitive, in terms of landmark accuracy and clinical parameter estimation, with the state-of-the-art on the public AASCE dataset and on two private spinal X-ray datasets. Our code and pre-trained weights are available at this https URL. 

---
# DCKV: Learned Transform Coding of the Prompt KV Cache in Hybrid Language Models 

**Authors**: Egor Maximov, Aleksei Merkelov, Vasiliy Alekseev, Ivan Krivorotov, Valentin Beliaev, Nikita Arsenin, Dmitry Bocharov, Aleksei Goncharov  

**Link**: [PDF](https://arxiv.org/pdf/2610.36950)  

**Abstract**: Hybrid language models keep a key-value (KV) cache only in their few full-attention layers, but this cache is still the part of a prompt's state that grows with the prompt and that must be stored or moved to reuse it. We present DCKV, a differentiable codec for the KV cache and the first KV-cache codec that is trained end to end. Its transform, its quantizer and its entropy models, which are conditioned on the prompt tokens and on already decoded neighbouring positions, are optimized together against the frozen language model, both for the code length that the entropy models assign and for the change of the model's output. On Qwen3.5-2B, DCKV stores a compressed token in more than 55 times fewer bytes than bf16 at a loss of at most 1.8 points on three long-context benchmarks, and in more than 100 times fewer bytes at a loss of at most 3.6 points. 

---
# Self-Keyed In-Parameter Memorization for External Memory-Free Dialogue Recall in Frozen Language Models 

**Authors**: XinYu Piao, Myunghoon Ryu, Joongheon Kim, Jong-Kook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.36948)  

**Abstract**: A large language model maintains conversational content in an external storage that is outside the model and grows as the dialogue continues. This work assumes an environment where a language model has no external memory and no prior knowledge of the future question or its answer. The motivation is to research whether a dialogue can be stored in the model parameters and later answer a question about it. This work proposes self-keyed in-parameter memorization (SKIM), a closed-form method in which the model generates its own questions about each dialogue turn and writes those questions into its parameters as the keys that later retrieve the turn's content. Generating many varied questions for each turn allows a later user question to match one even when phrased differently, and its answer is recalled from the parameters alone. Evaluated on five language models, SKIM recalls most stored dialogue turns from later user questions, and it answers from the model parameters without any external storage. 

---
# Beyond QASecClaw: Agent Scaling, Verifier Feedback, and Evidence Reliability in Software Assurance 

**Authors**: S M Abdulla Al Mamun, Akif Islam, Mohd. Ruhul Ameen  

**Link**: [PDF](https://arxiv.org/pdf/2610.36947)  

**Abstract**: Can additional specialist language-model contexts improve software assurance when deterministic tool evidence is already available? We study this question through five experiments spanning authored Java warning controls, OWASP Benchmark assessment, Defects4J test generation, SV-COMP assertion verdicts, and deliberately misleading evidence packages. The evaluated systems use Qwen2.5-Coder-7B-Instruct, with a 1.5B sensitivity analysis, and complete 13,638 model requests. On eight related development warnings, increasing the context count from 1 to 50 increases token use 49-fold without improving unsafe clearance. On 160 OWASP cases, model configurations trade clearance against unresolved outcomes; the ten-context arm assesses only 96 cases. On 32 SV-COMP tasks, raw verifier feedback raises mean correct model verdicts from 12.7 to 22.0, but also raises incorrect verdicts from one to seven; a JBMC-plus-JVM validation baseline returns 26 correct verdicts and six unresolved outcomes. On two sets of 120 evidence packages, models reject 70-100% of applicable counterexamples, while construction-specific scope checks classify every package correctly. No target defect is discovered in three Defects4J bugs, one unsupported by the test adapter. These descriptive results support deterministic evidence admission for the tested settings, while exposing limits of verdict aggregation, feedback interpretation, and the available evaluation coverage. 

---
# Developing a Speech Language Model for Basque 

**Authors**: Andoni Sudupe, Inma Hernaez, Ander Arriandiaga, Christoforos Souganidis, Ibon Saratxaga, Eva Navas  

**Link**: [PDF](https://arxiv.org/pdf/2610.36911)  

**Abstract**: Although the recent development of LLMs has paved the way for multi-modal interaction, minority languages face difficulties keeping pace. This work presents the first open Speech Language Model for Basque, designed for low-latency, end-to-end spoken interaction. Based on the LLaMA-Omni architecture, the system integrates a Latxa model as its backbone, transferring its Basque capabilities to the spoken modality by fine-tuning several modules. To address data scarcity, the English VoiceAssistant-400K corpus was translated into Basque, and the audio was synthesized using a Basque TTS. Evaluation results confirm the system's viability, achieving 3.86/5 in content quality, a 9.94% WER for spoken intelligibility, and a 3.07 UTMOS for audio quality. This work demonstrates that advanced spoken interaction systems can be developed using modular architectures and synthetic data even in low-resource languages. 

---
# AdaGround: Token-Normalized Adaptive Zooming with Verifier-in-the-Loop Refinement for GUI Grounding 

**Authors**: Bingchao Wang, Yongcheng Jing, Chengyu Wang, Jun Huang, Jie Yang, Wei Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2610.36874)  

**Abstract**: GUI grounding on high-resolution screens fails mostly on small targets, and zooming in is a common training-free remedy. However, most existing zooming pipelines do not match the zoom to the target's size, accept the zoomed answer unconditionally and never revisit a wrong answer. To address these issues, we propose AdaGround, a training-free framework with three components: (1) Token-Normalized Adaptive Zooming, which sets the magnification, field of view and zoom depth of each sample from the model's own estimate of the target size; (2) Verifier-in-the-Loop Refinement, which uses one marked-click Yes/No verifier to keep or reject each zoom, choose between zoom chains, trigger re-grounding and, with an existence query, decide when to abstain; and (3) Compositional Structure Grounding, which composes spreadsheet cells from their headers. On Qwen3-VL models, AdaGround surpasses state-of-the-art training-free zooming methods and many trained GUI models, improving accuracy over the backbone by 7.4 to 20.6 points on OSWorld-G and ScreenSpot-Pro, and its gains carry over to held-out UI-Vision. Code is available at this https URL 

---
# ActiveAlign: Aligning Egocentric Manipulation via Active Perception 

**Authors**: Ri-Zhao Qiu, Bowen Wen, Yifeng Jiang, Chen Tessler, Eric Heiden, Xue Bin Peng, Xiaolong Wang, Stan Birchfield, Erwin Coumans  

**Link**: [PDF](https://arxiv.org/pdf/2610.36859)  

**Abstract**: Real-to-Sim-to-Real (Real2Sim2Real) enables structured world representations for planning, interpretability, and safe execution. Most existing approaches rely on passive perception, which creates real2sim gaps due to imperfect observation. Inspired by how humans perceive actively through actions, we propose ActiveAlign, a framework that leverages active perception to align the Real2Sim2Real gap from interactions. ActiveAlign introduces two complementary modules: (i) ActiveAlign-Waving, which performs hand-eye calibration by exploiting active hand motions and differentiable rendering, and (ii) ActiveAlign-Touching, which refines object alignment through tactile feedback to achieve millimeter-level precision. Together, these capabilities enable strong real2sim alignment. We demonstrate that ActiveAlign-Waving can be used to curate simulation environments for open-sourced uncalibrated robot manipulation datasets, and demonstrate on a real Unitree G1 robot that ActiveAlign enables fine manipulation of millimeter-scale objects such as pressing buttons and moving sliders. 

---
# When All Rollouts Fail: Policy-Compatible Joint Decoding for Hard-Problem Reasoning 

**Authors**: Zehao Wang, Hanzhu Chen, Runquan Gui, Zhe Wang, Lin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36849)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) derives its learning signal from binary rewards via on-policy exploration and becomes a powerful paradigm for improving the reasoning of large language models. Yet it faces a learning cliff on problems beyond model's current ability: when every rollout fails, the samples reveal nothing about a correct solution, and group-relative estimators (e.g., GRPO, RLOO) collapse the update to zero advantage. Expert trajectories offer a remedy, but they are incompatible with the current policy and often lie in the student's low-probability regions, so the surviving updates demand large parameter shifts and decay into surface-level imitation rather than transferable reasoning. We argue that a useful learning signal on hard problems must be both verifiably correct and sufficiently compatible with the current policy. We propose Policy-Compatible Reinforcement Learning (PICO), which performs distribution-constrained joint decoding: expert data guides decoding at the token level while preserving the student's relative token probabilities, and a dynamic top-k mechanism confines the sampling distribution under a finite divergence budget, yielding trajectories that are verified and policy-compatible by construction. These trajectories are optimized jointly with on-policy rollouts under a mixed-behavior objective that mitigates their distribution mismatch via importance sampling. Experiments across reasoning diverse datasets and agentic benchmarks demonstrate that PICO consistently outperforms standard RLVR and prior off-policy baselines. 

---
# TAPESTRY: A Single Backbone for Video Generation and Robot Control 

**Authors**: Jeremy A. Collins, Seungjae Lee, Rachanon Wachakorn, Krishnan Srinivasan, Animesh Garg, Vitor Campagnolo Guizilini, Paarth Shah  

**Link**: [PDF](https://arxiv.org/pdf/2610.36823)  

**Abstract**: Pretrained video generation models encode rich priors about physical interactions, but leveraging them for robot control typically requires either resource-intensive video generation at test time or action heads trained from scratch that do not fully utilize the pretrained backbone. We introduce Tokenizing Actions as Video Patches in a Shared Transformer Policy (TAPESTRY), a world action model that repurposes a pretrained video diffusion transformer into a robotic policy without requiring video generation at inference. Our key idea is to treat robot actions as single-patch tokens in the video model's latent space. A lightweight linear projection maps each action timestep to the patch dimensions of the pretrained VAE, and the resulting tokens are processed by the same patchifying layers and transformer backbone used for video. The shared backbone is optimized through three diffusion objectives sampled per minibatch element: Video diffusion uses VAE latents from robot and human videos to establish dynamics priors. Joint video+action diffusion trains on robot demonstrations using a coupled diffusion timestep and an asymmetric attention mask. Action-only diffusion denoises actions conditioned on clean image latents. At test time, the model generates a compact action token sequence in a single DDIM step from pure noise, producing smooth action chunks at 84 ms per action chunk on consumer hardware and avoiding the cost of full video generation. Across simulation benchmarks and real-robot experiments, TAPESTRY achieves strong performance with substantially lower inference cost than comparable joint video--action policies. More information can be found at this https URL. 

---
# SetGRPO: Calibrating Grader-Induced Policy Optimization 

**Authors**: Jiale Huang, Zixu Li, Zhiwei Chen, Zhiheng Fu, Jinhe Bi, Yupeng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36807)  

**Abstract**: Rubric-based reinforcement learning uses fine-grained grader feedback to train large language models on open-ended tasks, but grader randomness can alter GRPO updates in ways that score discrepancies alone cannot characterize. Because GRPO jointly normalizes rewards within each response group and combines the resulting training coefficients with response-level gradients, a scoring discrepancy can shift the coefficients of the entire group and induce policy-dependent update changes. Selective execution further affects the direction-reversal risk of retained updates. To address this, we propose Set-Valued Group Relative Policy Optimization (SetGRPO). It comprises three modules: (a) Joint Reward Set Calibration constructs a joint uncertainty set for the training coefficients; (b) Geometry-Guided Safe Recovery maps candidate coefficients into the current policy's local update geometry and determines the largest safe recovery ratio for each response group; and (c) Selective Risk-Gated Optimization selects a global threshold through independent risk calibration to control the conditional direction-reversal risk of accepted updates while preserving the original GRPO objective. Under exchangeability, SetGRPO provides finite-sample marginal coverage for the joint coefficient set and simultaneous risk control across candidate thresholds. On ten open-ended benchmarks spanning three domains, SetGRPO achieves the highest macro average among all compared training methods on both Qwen3.5 and Llama3.2 model families. 

---
# An Algebra of Transformers 

**Authors**: Yong Wang, Zhiyan Wang, Hexiao Kang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36795)  

**Abstract**: On the theoretic analyses on transforms, they mainly focus on the computability analysis, such the Turing-completeness of unbounded precision transformers, the advantage of parallelizability of unbounded precision transformers and the limitations of bounded precision transformers. This paper give transformers an algebra for reasoning, which relies heavily on the functional decomposition results of transformers. This algebra uses functional decompositions as atomic actions, includes sequential composition and parallel composition, recursion for feedback and abstraction for hidden layers. As we know, it is the first attempt to model transformers logically and can be used to verify the correctness of transformer based applications. 

---
# Select-TTT: Query-Conditioned Selective Fast Weight Updates for In-Place Test-Time Training 

**Authors**: Yue Wu, Yang Zhang, Weinan Gan, Chongming Gao, Xinsen Guo, Xiaoyuan Li, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.36794)  

**Abstract**: Test-time training (TTT) enables a deployed language model to internalize incoming context on-the-fly into a small set of fast-weight parameters, rather than merely treating it as transient context for attention. In-Place TTT recently made this idea practical for large language models by repurposing the down-projection of standard MLP blocks as fast weights. However, we find that its benefit largely disappears on realistic web long-context question answering. We attribute this to In-Place TTT's uniform write schedule: every chunk is written with equal strength, so irrelevant content dilutes query-relevant information in fast-weight memory. Building on this insight, we propose Select-TTT, a model-agnostic, query-conditioned rule for In-Place TTT that scales each chunk's write strength by its relevance to the user query. This relevance signal requires no supervision and no new parameters: it is read directly from the model's frozen output head, as the probability mass that a chunk's hidden states assign to the query tokens. A temperature-scaled softmax turns these scores into learning-rate multipliers, leaving the chunk-wise, context-parallel computation of In-Place TTT intact. Select-TTT consistently improves long-context question answering on two benchmarks and two backbones. These findings highlight the value of query-conditioned fast weight writes for practical test-time training on long-context question answering, which is also potentially promising for long-horizon agent tasks involving multiple user queries over evolving contexts. 

---
# The Vocabulary Is Already Here: Adapting Pretrained Interpretable Features for Generative Recommendation 

**Authors**: Ziluowen Luo, Ruochen Liu, Jun Yin, Hao Yan, Zhengxuan Hu, Xinze Shi, Chaozhuo Li, Chengqi Zhang, Senzhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36763)  

**Abstract**: Generative recommendation (GR) predicts the next item through its semantic ID (SID). Existing methods commonly fit SID vocabularies to pretrained language model (PLM) representations of item descriptions, incurring additional training costs and making vocabulary coverage dependent on the fitting domains. We observe that pretrained sparse autoencoders expose PLM features with human-interpretable semantic annotations, allowing feature indices to serve as a reusable SID vocabulary. However, reusing this vocabulary requires preserving item distinctions under compact token budgets and learning to favor target items over candidates with shared dominant features. It also requires extending feature interpretability from item content to retrieval decisions. To address these challenges, we propose API, a framework that Adapts the Pretrained Interpretable vocabulary for effective and explainable GR without additional vocabulary learning. Within API, Pretrained Interpretable IDs (PIDs) retain selected feature indices and relative activation strengths for item discrimination. With PIDs as supervision, the Semantic Prediction Adapter (SPA) learns user preferences from interaction histories through joint feature-index and strength prediction. SPA combines both predictions to score candidate items, with item-level supervision encouraging the observed next item to outrank its alternatives. API further explains candidate score differences through feature contributions paired with their pretrained semantic annotations. Extensive experiments show that API achieves competitive performance across backbones and decoding schemes, while preserving item distinctions under compact token budgets and enabling explainable retrieval. 

---
# Making Self-Driving Laboratories Auditable:A Unified Framework for Hybrid Actions,Measurements, and Safe Execution 

**Authors**: Zihan Zhang, Zeyu Wu, Xin Zhang, Mingyu Luo, Haohui Que, Tong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36761)  

**Abstract**: Self-driving laboratories (SDLs) are laboratory systems that couple experimental planning, automated execution, and measurement feedback in a closed loop. Integrating flexible robot control with the typed application programming interfaces (APIs) and continuous measurements exposed by laboratory instruments remains an execution-interface challenge. To address this challenge, we propose a traceable framework comprising four components: LAM (Lab Action Model), OpenSDL, Safe-SDL, and L3-Bench. LAM is a proposed measurement-conditioned policy architecture that extends vision-language-action paradigms to generate both continuous robot trajectories and typed instrument API calls. OpenSDL is an execution substrate with an initial control implementation, designed to map role-level operations to heterogeneous devices, coordinate execution, and record timestamped causal events. The proposed Safe-SDL runtime gate combines pre-dispatch schema and resource checks with signal temporal logic monitoring during execution, including extended waiting periods. L3-Bench (Level-3 Laboratory Benchmark) specifies an episode-level evaluation protocol covering physical outcomes, timing, reproducibility, safety violations, and fault recovery. Together, these components provide an end-to-end design for linking hybrid action generation, monitored device execution, measurement feedback, and experimental evaluation, addressing coordination and traceability challenges in SDL workflows. 

---
# Robust LLM Data Selection via Content-Invariant Gradient Attribution 

**Authors**: Jia Gu, Liang Pang, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.36755)  

**Abstract**: Gradient-based data attribution has become an important tool for high-quality data selection in the instruction fine-tuning of large language models (LLMs), as it estimates the contribution of training examples through their influence on model optimization. However, we find that existing global gradient attribution methods suffer from surface-form bias: minor changes to semantically irrelevant details, such as wording or formatting, can produce highly inconsistent attribution scores even when the underlying content remains unchanged. To address this issue, we propose Content-Invariant Gradient Attribution (CIGA), a robust framework for instruction data selection. CIGA first identifies layers whose gradients are both influential for attribution and stable under semantics-preserving surface-form perturbations, thereby reducing formatting noise in gradient features. It then dynamically fuses these attribution signals with semantic representations of the input, combining optimization-level influence with content-level information. Extensive experiments across multiple benchmarks show that CIGA produces more stable rankings under surface-form perturbations while preserving downstream capabilities and improving average performance. CIGA also reduces attribution cost, offering a robust and efficient approach to data selection for LLM instruction tuning. 

---
# Generative Refinement Learning for Temporal Interpolation in Scientific Machine Learning 

**Authors**: Pu Ren, Michael W. Mahoney, N. Benjamin Erichson  

**Link**: [PDF](https://arxiv.org/pdf/2610.36751)  

**Abstract**: Computationally expensive simulations and real-world observations often provide data at a lower temporal resolution than downstream tasks require. Averaging observations at fixed grid points can introduce errors, and motion estimates alone do not determine how the physical field values change. In this work, we study whether learning to correct a simple numerical interpolation improves the reconstruction of intermediate scientific fields. Our model, interpolative FLow EXpert (iFLEX), first estimates the missing field by taking a weighted average of the two observations at each grid point, giving more weight to the observation closer to the requested time. During training, subtracting this estimate from an observed intermediate field gives the residual that a conditional diffusion model learns to generate. Both observations condition the model, together with the target time, the interval between observations, and available physical parameters. To reconstruct an unobserved intermediate field, the generated residual is added to the weighted average. We evaluate iFLEX on Navier-Stokes Kraichnan turbulence, Shanghai radar precipitation, and sea surface temperature. Across these datasets, iFLEX achieves lower reconstruction error than the evaluated baselines, including optical-flow and diffusion models developed for video frame interpolation in computer vision. In a controlled turbulence comparison at a Reynolds number of 36,000, learning the residual reduces relative Frobenius norm error from about 13.7 for direct field generation to 10.5, with the diffusion backbone and training setup held fixed. Code is available at this https URL. 

---
# From Voiceprints to User Entities: Online Speaker-Memory Construction for Multi-User Spoken Agents 

**Authors**: Yucong Zhang, Cancan Li, Bo Xiao, Ke Zeng, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.36713)  

**Abstract**: Personalized spoken agents must identify speakers before accessing user-specific memory, yet memory benchmarks typically assume correctly attributed histories. We argue that voiceprints are observations of identity, not identities themselves, and formulate online speaker-memory construction to maintain persistent user entities as acoustic evidence evolves. We introduce EvoSpeakerMem, a training-free layer for causal speaker routing, selective acoustic-state updates, and identity-partitioned memory writes and retrieval through a replaceable text-memory backend. We also construct LoCoMo-ID, a spoken-memory benchmark with hidden speaker identities, speaker-sensitive queries, simulated recording-condition shifts, and four-speaker identity-confusion tests. On longitudinal SMIIP-TV, genuine-probe rejection under a frozen near-enrollment threshold rises from 29.8% to 55.6% over approximately one year. On LoCoMo-ID speaker-sensitive queries, gated updating improves identity-gated QA over fixed representations by 3.86 and 11.62 percentage points under clean and shifted conditions, respectively, with no wrong-user authorizations. The advantage persists in the four-speaker test. Yet reducing fragmentation and improving QA can coexist with incorrect cross-user links, revealing a fragmentation--contamination trade-off that QA alone cannot capture. 

---
# Conservative Dueling Bandits: Preference Learning under Baseline Constraints 

**Authors**: Rohan Deb, Mohammad Ghavamzadeh, Arindam Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2610.36708)  

**Abstract**: Contextual dueling bandits learn from pairwise comparisons, the same feedback used to fine-tune language models from human preferences. Standard algorithms explore aggressively and can show users two poor options. We study conservative contextual dueling bandits, which require that, at every round, the cumulative score of the better displayed item stay above a $(1-\alpha)$ fraction of a trusted baseline's. An item's score is its probability of beating the best available item, so the constraint depends only on utility differences and needs no absolute rewards. This also changes how safety can be certified: we show that transferring the safety test of the conservative bandit algorithm CLUCB unchanged can incur linear regret, even when the baseline's utilities are known exactly. For linear utilities, we develop MaxInP-BC, which certifies scores through utility gaps and, when it must fall back, pairs the baseline with an informative partner so that fallback rounds still drive learning. It is safe with high probability at all times, with regret $\tilde{O}(d\sqrt{T}/\kappa)$ plus an additive cost of conservatism that does not grow with the horizon. For general utility classes, we develop BTIGW-BC, which turns the guarantee of an online logistic-regression oracle into cumulative safety certificates. It is safe at all times, makes finitely many fallbacks, and has sublinear regret whenever the oracle's logistic regret is sublinear. In experiments, both satisfy the constraint, which their unconstrained counterparts often violate, at a moderate cost in regret. 

---
# HeteGCD: Statistics-Aware Dual-Prototype Learning for Generalized Category Discovery on Heterophilous Graphs 

**Authors**: Xuanzhi Xi, Tongze Zhang, Zhili Qin, Yulin Tao, Hongliang Wang, Zhong Zhang, Qinli Yang, Junming Shao  

**Link**: [PDF](https://arxiv.org/pdf/2610.36696)  

**Abstract**: Heterophilous graphs are prevalent in real-world applications, where connected nodes often exhibit dissimilar attributes or belong to different classes, posing significant challenges to conventional graph learning methods based on homophily. This limitation becomes particularly pronounced in generalized category discovery (GCD), which requires models to recognize known classes and discover novel ones using labels from known classes only. However, existing graph GCD methods typically rely on aforementioned homophily-based inductive biases, which are ill-suited to heterophilous graphs. Meanwhile, although current heterophilous graph learning methods exploit diverse neighborhood signals, their selection and fusion are often guided by supervision across all classes, including novel classes whose labels are unavailable in GCD. These limitations raise a key question: how can heterogeneous neighborhood information be effectively leveraged for category discovery without any novel-class supervision? To answer this question, in this study we propose HeteGCD, a new statistics-aware dual-prototype learning perspective for generalized category discovery on heterophilous graphs. The key idea is to examine both statistics of local attribute differences and neighborhood class distribution to extract discriminative information in feature and probability spaces simultaneously. Specifically, a gated dual-stream encoder integrates both node attributes and local attribute-difference statistics to learn informative node representations. Afterwards, a dual-prototype classifier combines feature-based similarity with distribution-based consistency to support category discovery. Experiments on five heterophilous graphs show that HeteGCD achieves competitive performance against 24 baseline combinations. 

---
# TOPS: Reshaping Flow Source and Transport for Multi-Task Robot Manipulation 

**Authors**: Tuo An, Chuhao Zhou, Jiahua Dong, Jingliang Li, Jiaqi Bai, Jindou Jia, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36694)  

**Abstract**: Flow-based policies for language-conditioned multi-task manipulation commonly share a single $\textit{Gaussian}$ source prior and a common linear transport path across tasks. However, reliably executing multiple tasks with distinct motion requirements remains challenging for a shared policy. We introduce $\textbf{TOPS}$ ($\textbf{T}$ask-$\textbf{O}$riented $\textbf{P}$riors and $\textbf{S}$chedules), which adapts the flow source prior and interpolation schedule to each task without modifying the network architecture. TOPS first constructs multimodal task sources from reverse-mapped expert action sequences, allowing language instructions to select where generation begins. Guided by source-target covariance mismatch, it then slows the transition toward expert actions early in denoising and accelerates it later along directions requiring expansion. Under a $\textit{Gaussian}$ endpoint model, our analysis shows that this schedule adjustment can lower an upper bound on velocity-prediction risk. Experiments on LIBERO, LIBERO-Plus, and CALVIN demonstrate improved performance across in-distribution, perturbation and cross-environment settings, alongside better data efficiency. Using one integration step, TOPS raises LIBERO-10 success from $53.44\%$ to $69.67\%$ compared with ten-step Standard Flow Matching (FM), while reducing inference time to approximately one-sixth of the baseline. TOPS also yields consistent gains when transferred to the large-scale flow-based $\pi_{0.5}$ policy across all four evaluated settings. Real-world experiments further demonstrate improved task success, including an increase from $20\%$ with standard FM to $80\%$ with TOPS on the upper-drawer opening task. 

---
# Specialization, Diversity, and Scaling in Child-Scale Language Models 

**Authors**: Steven Y. Feng, Michael C. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2610.36679)  

**Abstract**: Modern language models are trained on far more language than children experience, but we know much less about learning from small, naturalistic datasets. We train small GPT-2 models on transcripts of speech with children from dense home recordings (BabyView) and on synthetic child-directed dialogue (TinyDialogues), varying the amount, source, and diversity of training data. Models trained on one child's language environment predict that environment substantially better than other children's environments. Much of this specialization is lexical and distributional: a simple 5-gram model reproduces the cross-child pattern (r=.91). Training on data from multiple children clearly improves prediction of pooled held-out language relative to training on a single child's input without a similarly clear improvement on grammatical benchmarks. Across data sizes, naturalistic and synthetic training inputs also produce different grammatical and lexical-semantic patterns, with important differences between grammatical phenomena. Together, these results show that small-data language learning depends not only on how much language a model sees, but on which environments it comes from and how well that input matches what the model is asked to predict. 

---
# Atom as an Agent: Emergent Order by Collective Predictive Coding 

**Authors**: Koji Hashimoto, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2610.36673)  

**Abstract**: What is the intrinsic reason of emergence of macroscopic order parameter out of microscopic systems? In this article, we apply to physics the collective predictive coding (CPC), the theory of language/symbol emergence. As our language emerges out of our human communications, physical macroscopic order parameter emerges out of communications among atoms --- each atom is regarded as an agent who predicts each other's behavior through inference. We present three examples of classical spin systems: two-dimensional ferromagnetic and antiferromagnetic Ising models, a two-dimensional spin model with competing stripe orders, and the fully connected Sherrington--Kirkpatrick spin-glass model. In all of these examples, proper macroscopic order parameters including the replica overlap emerge automatically through communication informatics of CPC, only from observed spin configurations. We call the learned macroscopic coordinate the collective predictive order. Furthermore, for the ferromagnetic and antiferromagnetic Ising models, the equilibrium distribution of the collective predictive order yields an emergent potential with the symmetry and thermal evolution characteristic of a Landau potential. Notably, the emergence of the collective predictive order does not refer to any structure of microscopic Hamiltonians/Lagrangians. Thus our ``Atom-as-an-Agent" (AaaA) principle offers a completely data-driven method for the emergence of macroscopic order, that is, machine learning physics. 

---
# Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs 

**Authors**: Donghyeon Cho, Joowon Kim, Seungho Shin, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36671)  

**Abstract**: Diffusion vision-language models (DVLMs) process their entire input at every denoising step. Since visual tokens make up 89-97% of this input on the benchmarks we study, most of the decoding computation is spent repeatedly processing the same image. Visual token pruning methods partially relieve this burden by removing part of the visual tokens, but they lose the removed information and still process the remaining image at every step. To measure how often the image is actually needed, we compare LLaDA-V's predictions with and without the image at every decoding step and find that removing the image changes a token being unmasked at only 22.8% of steps on average and at as few as 7.9%. We therefore propose LOWN, a decoding method that decides at every step whether to process the image and how many tokens to unmask. LOWN predicts the masked tokens without the image and processes the image only when this prediction disagrees with the last image-conditioned prediction. Agreement certifies the earlier prediction for reuse, and the number of certified tokens sets how many are unmasked in that step. We further observe that the need for the image differs across benchmarks, so LOWN's adaptive decision is needed rather than a fixed schedule. Without training or a tuned threshold, LOWN retains 99.6% of LLaDA-V's score on ten benchmarks at 35.3% of the decoding FLOPs on average. LOWN also generalizes across LLaDA-based DVLMs, composes orthogonally with visual token pruning, and runs under key-value caching for further efficiency gains. Code and project page are available at this https URL 

---
# A Quantitative Approach to Asch Conformity in Frontier Large Language Models 

**Authors**: Senthil Rajasekaran, Stylianos Loukas Vasileiou  

**Link**: [PDF](https://arxiv.org/pdf/2610.36668)  

**Abstract**: The Asch conformity experiment has recently gained considerable attention as a way of measuring conformity in large language models (LLMs). While previous studies have focused on demonstrating conformity in LLMs, our work adds a further quantitative layer to this approach. Instead of sampling a frontier LLM on a suite of questions, our approach samples a single question repeatedly, leveraging the probabilistic nature of LLMs to produce a distribution of answers. In doing so, we can compare the distributions of answers before and after the model is presented with the manufactured consensus. Our experiments confirm that frontier LLMs do conform, reinforcing prior literature, and, through our distribution-to-distribution comparisons, provide additional quantitative data on the conditions that lead to conformity and the degree to which it occurs. 

---
# Tap2Listen: Instant Target Switching for Streaming Audio-Visual Speaker Extraction 

**Authors**: Hansol Park, Minkyung Song, Zinu Kim, Yujun Lee, Hoseong Ahn, Hyunjin Choi, Kyuhong Shim  

**Link**: [PDF](https://arxiv.org/pdf/2610.36643)  

**Abstract**: Streaming audio-visual target speaker extraction (AV-TSE) uses a target speaker's face to extract the speaker's speech from an input audio mixture in real time. However, existing systems fix the target speaker for an entire clip, while conversation demands frequent and immediate target switches. To address this problem, we propose Tap2Listen, a streaming AV-TSE model designed for instant response to target switches. The model separates computation by modality: the target-agnostic audio encoder processes only the mixture, while the target-specific visual cues of the selected face condition only the decoder. A switch requires only a 0.12ms re-projection and takes effect immediately. We also design component-wise losses: a prediction loss for the encoder and a supervised extraction loss for the decoder. For evaluation, we measure how quickly output quality recovers after a switch. On the standard LRS3-2mix test set, Tap2Listen improves SI-SNRi from 8.85 to 9.68 dB at a comparable parameter count. After a switch to a known face, the model produces full-quality output from the first frame, whereas the baseline needs about 1 second to recover. Finally, we show that the proposed streaming AV-TSE model runs in real time on a single CPU core. 

---
# Adapting Video Models for Robot Control via Motion-guided Asymmetric Prediction 

**Authors**: Taeyoung Kim, Junyoung Sung, Heeseung Kwon, Minseop Kim, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2610.36624)  

**Abstract**: World Action Models (WAMs) leverage pretrained video generative models for robotic control by jointly predicting future visual observations and actions. The visual prediction objective plays a key role in adapting pretrained video representations to the target robot domain. Yet standard visual prediction inherited from video pretraining treats all visual regions uniformly, despite their differing relevance to robotic control. We therefore investigate the spatial allocation of visual supervision in WAMs and show that focusing supervision on dynamic regions can substantially improve policy performance. Building on this finding, we introduce Motion-Guided Asymmetric Prediction (MoGAP), which augments WAM training with motion-focused visual prediction. MoGAP allocates noise asymmetrically across regions, training the model to reconstruct more heavily corrupted dynamic regions using less corrupted static regions as context. MoGAP is trained alongside standard video-action prediction through a training-only auxiliary prediction path within the shared video diffusion backbone. It uses motion cues derived directly from training videos, requiring neither an external teacher nor additional modalities. Across robot manipulation benchmarks and WAM architectures, MoGAP consistently improves policy performance, increasing success rates on the GR-1 tabletop benchmark from 38.8% to 47.5% for Fast-WAM (Wan2.2-5B) and from 52.3% to 62.3% for Cosmos3-Nano. 

---
# Dream-COP: Guiding Algorithm Discovery for Combinatorial Optimization Problems in an Experience World 

**Authors**: Kuan Xu, Jianming Wang, Ni Zhang, Yaoxin Wu, Yong Wang, Zhiguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.36618)  

**Abstract**: Large language models (LLMs) enable algorithm discovery for combinatorial optimization problems (COPs) through iterative program generation and evaluation. A central challenge is how to organize search to discover better algorithms within a limited budget. Our key insight is that past search trajectories and their recorded outcomes form an experience world, which can be used to facilitate the COP solving through replay. We introduce Dream-COP, in which a \emph{policy agent} learns executable policies through \emph{dreaming} in this experience world to guide a \emph{discovery agent} in generating and refining COP programs. Specifically, policy replay provides feedback on the quality of the programs revealed and the search steps used, to guide policy selection more effectively. Across five COPs, \textsc{Dream} discovers better algorithms than fixed exploration in both heuristic component design and complete solver generation. Particularly, it matches the optimal values on KP and reduces the gap on OBPP from 2.112\% to 1.199\% while using 42.5\% fewer search requests. Moreover, the learned exploration policies transfer effectively across different COPs with the assitance of the experience world. 

---
# Balanced State Subspaces for Mamba-2 State Compression 

**Authors**: Shang Tingyu, Cao Wenfei  

**Link**: [PDF](https://arxiv.org/pdf/2610.36617)  

**Abstract**: Mamba-2 compresses all context into a fixed-size recurrent state that dominates decoding memory and bandwidth. Existing compressors quantize state values or prune state coordinates, scoring each channel independently in the native basis. We examine trained Mamba-2 states and find that energy is almost rank-one, but the output does not read the energetic directions: energy concentrates in a bias-like direction accumulated by the recurrence, while the content the output reads lives in a low-dimensional subspace that is not aligned with the state's coordinates. Even output-aware scores are computed coordinate by coordinate, so they miss the directions that determine the input--output map. Compression should therefore target the write--read subspace rather than individual coordinates. We propose \method{}, a post-training transform that instantiates balanced truncation with full controllability and observability Gramians and applies the rotation exactly via Mamba-2's scalar-identity transition. Because the useful rank varies widely across layers, we further introduce HARP, which allocates ranks per layer from the Hankel spectrum and protects retrieval-sensitive layers, improving both perplexity and retrieval over uniform budgets. Calibration takes a single forward pass, an order of magnitude faster than GHOST's released protocol, with lower peak memory. With HARP, \method{} halves the state dimension at an average WikiText-2 perplexity increase of $0.3$ points ($1.8\%$) across 130M--2.7B, versus $1.6$ ($9.3\%$) for GHOST, and the gap widens at 25\% retention. 

---
# Benchmarking LLM Agents for Behaviorally Adequate Test Generation from Software Requirements 

**Authors**: Binhang Qi, Yiming Liu, Yun Lin, Yangjie Zhou, Hailong Sun, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.36584)  

**Abstract**: As large language model agents increasingly compile natural-language requirements into repository-level code, requirement-to-test generation grows in importance to provide validation constraints (or harness) to guide code generation. Yet it remains unclear whether current agents can generate tests that validate the required behavior when the implementation of that behavior is withheld. Existing test-generation benchmarks mostly focus on white-box testing, whereas the requirement--test-related study usually requires evaluation on semantic alignment between the given documents and the generated tests.
We introduce Req2Test-Bench, a benchmark containing 5,086 execution-validated instances recovered from developer-authored behavior-driven development (BDD) artifacts in 8 open-source projects, spanning 5 programming languages and 4 BDD frameworks. Each instance pairs a natural-language requirement in Gherkin with its developer-written Test Implementation. Given the requirement, target interfaces of the required behavior, an agent must generate a complete, potentially multi-file Test Implementation. Evaluation of 8 agent--model--reasoning configurations reveals a substantial gap between execution success and behavioral adequacy, which is measured strictly as kill-completeness: a kill-complete test kills every mutant that the ground-truth test kills. The best-performing configuration, Codex with DeepSeek-V4.1-Flash, produces passing tests for 82.8% of instances, yet only 33.8% are behaviorally adequate. We find that (1) scaling up agents yields sharply diminishing returns in behavioral adequacy relative to generation cost; (2) under-specified requirements cause the information gap, including ambiguous setup and implicit expected outcomes; and (3) requirement-grounded harnesses that steer agent non-determinism offer a promising direction for improvement. 

---
# MM-Net: A Multimodal Multi-Task Network for Sleep Staging and Sleep Apnea Assessment After Ischemic Stroke 

**Authors**: Md Wahiduzzaman Suva, Esm E Moula Chowdhury Abha, Md Imtiaj Alam Sajin, Md Iftekharul Mobin, Jia Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2610.36583)  

**Abstract**: Automated polysomnography after ischemic stroke is challenging because of altered sleep-related neural activity and concurrent respiratory disturbances. Existing stroke-specific approaches mainly address sleep-stage classification, leaving joint analysis of sleep architecture and respiratory events underexplored. Moreover, adapting pretrained EEG encoders introduces a representation inconsistency when downstream models are trained on embeddings from previously seen patients but evaluated on unseen individuals. We propose MM-Net, a multimodal multi-task framework that resolves this inconsistency through cross-fitting and ridge-based embedding alignment. Sleep-adapted EEG-EOG representations and engineered neural descriptors are fused with learned cardiorespiratory features through cross-attention and shared temporal modeling for simultaneous sleep staging and respiratory-event detection. On iSLEEPS (99 patients with subacute ischemic stroke), ten-fold patient-independent evaluation achieved 76.3% staging accuracy (Cohen's $\kappa$ = 0.656) and 0.825 respiratory-event AUC, outperforming the evaluated baselines under matched protocols. The sleep-adapted encoder improved the downstream model when its embeddings were aligned, raising staging accuracy by 1.4 percentage points over the frozen encoder ($p$ = 0.020) versus 0.2 points without alignment. MM-Net also achieved 86.4% staging accuracy on Sleep-EDF, above every single-EEG architecture reimplemented on the same recordings, and 80.3% staging accuracy with 0.889 respiratory AUC on ISRUC-I, exceeding its retrained staging baselines. Without retraining, the iSLEEPS-trained model retained respiratory AUCs of 0.833 on ISRUC-I and 0.841 on UCD. These findings support representation-consistent neural adaptation and joint physiological modeling across stroke, healthy-sleep, and sleep-disorder cohorts. 

---
# Less is more: learning what to profile for GPU kernel optimization 

**Authors**: Yihan Wang, Peiyu Liu, Runyu Chen, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36576)  

**Abstract**: Large language models have recently shown strong potential for automatically optimizing GPU kernels, yet their effectiveness depends critically on the hardware information available to the kernel optimization model. Existing systems typically rely on predefined profiling strategies, leaving what to profile largely outside the optimization process. We show that effective profiling is not simply about collecting more information: increasing the number of profiling metrics does not monotonically improve kernel optimization, while the most useful profiling information varies across kernels. Our theoretical analysis further shows that, under such heterogeneous profiling utility, kernel-conditioned selection can outperform any single fixed profiling configuration. Motivated by this observation, we propose AutoProfiler, a feedback-driven framework for learning a deterministic profiling tree from optimization feedback. The tree is iteratively refined through REWRITE and SPLIT operations, with an LLM generating candidate modifications and empirical evaluation determining their acceptance. Once constructed, the tree quickly gathers the required profiling information for any kernel, allowing the LLM rewriter to be invoked only once. Experiments on KernelBench show that our approach improves kernel optimization over fixed profiling configurations under constrained profiling budgets. These results suggest that hardware profiling should evolve from static metric collection toward profiling reasoning, making profiling itself a new dimension for improving kernel optimization. The codes are available at this https URL. 

---
# NextEval: An Agent-Based Solver and Benchmark for Derivative-Free Optimization 

**Authors**: Shuo Chen, Li Cui, Cunxin Huang, Ya-Feng Liu, Minghang Xie, Meng Xu, Zaikun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36553)  

**Abstract**: Derivative-free optimization (DFO) addresses problems where derivatives are unavailable or costly to obtain. This setting often arises in simulation-based and experimental applications, where evaluating the black-box objective can itself be costly. Thus, it remains a central challenge in DFO to effectively exploit accumulated observations to select informative evaluation points under a limited budget. In this light, we introduce the NextEval solver, an agent-based DFO solver, which integrates the capabilities of a large language model (LLM) to guide the sequential selection of evaluation points. Specifically, we establish a system that combines an LLM agent with a numerical environment. The environment provides tools for analyzing past evaluations and executing numerical methods, while maintaining a shared evaluation history and method-specific state to support continuation across method switches. Using observations and tool feedback, the LLM agent selects tools, configures their parameters, and allocates the evaluation budget, while numerical methods compute and evaluate candidate points. Building on this system, we also introduce NextEval Bench, a benchmark protocol for comparing agent configurations on shared DFO tasks using the same numerical tools under matched evaluation budgets and information constraints. Numerical results on low-dimensional DFO problems demonstrate that the NextEval solver outperforms the state-of-the-art DFO baselines in function-evaluation efficiency. 

---
# REACH: An RL-based Hierarchical Humanoid Crawling Framework for Adaptable Confined Space Traversal 

**Authors**: Shengzhang Xu, Yuning Guo, Weiyi Hu, Yifeng Xu, Zhongyu Li, He Kong  

**Link**: [PDF](https://arxiv.org/pdf/2610.36546)  

**Abstract**: Humanoid locomotion in confined spaces requires continuous height adaptation across multiple locomotion modes, yet most existing approaches remain limited to height-adjustable bipedal walking. We present REACH, an RL-based hierarchical framework that enables a Unitree G1 humanoid to traverse height-constrained environments through variable-height bipedal walking and crawling. The learned policies regulate body height over complementary command ranges of 1.27--0.50m in bipedal locomotion and 0.50--0.20m in crawling, extending traversal beyond the clearance limits of crouched walking. To address contact-rich crawling dynamics, we employ a history-conditioned latent representation that captures terrain and interaction context from proprioceptive observations. A bidirectional transition policy further connects crouched bipedal locomotion and crawling across large kinematic and contact discontinuities. These skills are integrated with a LiDAR-based planner that converts point clouds into collision-free three-dimensional paths and locomotion commands, including proactive height targets. Real-world experiments demonstrate accurate height and velocity tracking, robust terrain traversal, reliable mode transitions and confined-space traversal with the proposed pipeline. Code and videos are available at this https URL. 

---
# FrameForge: Harnessing Priors of Video Foundation Models for Unified Image Generation 

**Authors**: Yijing Lin, Zhendong Mao, Mengqi Huang, Yuxin Wang, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36545)  

**Abstract**: Unified conditional image generation requires a single model to preserve different aspects of a visual input while producing a target image that follows textual guidance. Pretrained video models provide a promising foundation by encoding visual transformations across time, yet existing approaches typically assign the condition and target to fixed temporal positions. We observe that condition-target relationships vary substantially across generation tasks, while video models naturally encode visual transformations at different temporal scales. This motivates us to model conditional generation through input-adaptive virtual temporal relations. We propose FrameForge, which captures two complementary aspects of such relations. Mixture of Temporal Experts (MoTE) models how far the target lies from the condition by dynamically selecting temporal RoPE experts with different virtual intervals. Chain of Temporal Reasoning (CoTR) models how the condition is transformed into the target through a latent intermediate state, with a reasoning-bottleneck attention mask encouraging information propagation through this pathway. Experiments on DreamBench, COCO2017, and GEdit-Bench show that FrameForge achieves superior performance in subject-driven generation, canny-to-image generation, and image editing among the compared methods. 

---
# RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding 

**Authors**: Buzhao Liu, Jianing Ye, Xiu Yuan, Yangtong Feng, Chengsong Huang, Tyler Dinh, Yevgeniy Vorobeychik, Ning Zhang, Chongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36533)  

**Abstract**: Recently, LLM-based autonomous agents have shown strong capabilities in iteratively generating, evaluating, and refining executable solutions. In robotic manipulation, Code-as-Policy lets these agents search over modular and editable robot programs. However, effective program search requires physical understanding: agents must understand how objects can be manipulated, how scene geometry constrains actions, and why interactions succeed or fail. Existing methods rely mainly on execution feedback for program refinement, often leaving these physical questions unresolved. We introduce RoboAuto, a framework for robot program search that explicitly organizes physical understanding at the object, scene, and interaction levels through three components. First, a hierarchical knowledge base combines executable skills with object-centric knowledge of affordances, spatial relations, and action effects, learned from embodied data and online task solving. Second, scene grounding connects this knowledge to the current environment by identifying relevant objects and resolving action-critical quantities. Third, multi-stage planning with self-designed experiments checks whether each stage produces its expected physical effects and actively tests competing hypotheses to resolve physical uncertainties. RoboAuto achieves superior performance across five benchmark families, attaining an 89.1% success rate on short-horizon LIBERO-PRO tasks and more than twice the mean success rate of ASPIRE on LIBERO-PRO Long and LIBERO-Plus. It also outperforms ASPIRE and $\pi_{0.5}$ on three challenging long-horizon real-robot tasks requiring precise manipulation. We further show that the knowledge base transfers across simulators and embodiments, and that program search performance scales with its size. Project page: this https URL 

---
# Do AI Scientists Know What's New? 

**Authors**: Miri Liu, Xiayu Zhao, Encheng Xie, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2610.36511)  

**Abstract**: A major challenge facing AI scientific discovery pipelines is how to determine when a generated idea is meaningfully novel. Ideally, novelty checks in these pipelines must be able to reject non-novel ideas while not rejecting too many novel ideas, and they should also be able to distinguish which parts of a proposal are non-novel, as opposed to opaquely discarding entire proposals. We design an analytical framework for evaluation of novelty checks with temporal ground truth at both the proposal- and claim-level, using four datasets of 1,060 proposals built from published papers, reviewer disputes, and generated ideas matching ones the field went on to publish. Auditing existing novelty checks with this framework, we find that checks with high rejection rates on non-novel work also tend to have high false alarms on ideas that are later realized by human researchers, and observe that checks built to evaluate proposals as a whole can point to claim-level prior art only by rejecting the whole proposal. We therefore introduce ClaimNov, a fine-grained, interpretable claim-level novelty module. ClaimNov catches 97% of disguised papers at a 1.5% false-alarm rate, and its claim-level novelty profile cuts the LLM judge's false alarms from 11.8% to 1.5%. Because its verdict rests on claim-level evidence, ClaimNov also keeps proposals that build on established contributions, rejecting only 3% of ICLR-accepted proposals with a reviewer-disputed contribution and keeping 91 of 100 novel proposals with one spliced prior contribution, against 76 for the strongest existing check. Code and data are available at this https URL. 

---
# ProCerDCL: Query-Efficient Black-Box Probabilistic Verification of Deep Learning Compiler Optimization 

**Authors**: Zexin Li, Weibo Zhang, Linyi Li, Arvind Easwaran, Simin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.36499)  

**Abstract**: We present \tool, a query-efficient framework for black-box probabilistic verification. Given a fixed, sampleable input distribution and a binary property oracle, \tool assesses the property's violation probability $p$ against a threshold $\theta$. Under independent sampling and stateless oracle behavior, acceptance asserts $p\le\theta$, rejection asserts $p>\theta+\eta$ for a user-specified separation margin, and an inconclusive verdict indicates insufficient evidence for either conclusion. The probability of each type of erroneous verdict is bounded by its specified error budget. To support this three-valued contract efficiently, \tool uses a single sequential test, eliminating the interval family required by the baseline. The test accounts for repeated sampling checks through either corrected exact binomial confidence intervals or an anytime-valid confidence sequence. We apply \tool to probabilistic local model semantic consistency, where the property oracle detects decision-level inconsistencies between reference and compiled executions under a specified perturbation distribution. Our evaluation covers 24 tasks across TorchInductor and TensorRT and examines sampling efficiency, conclusiveness, and diagnostic behavior. For sampling efficiency, we use paired one-model clean-label radius searches: the baseline requires 3.6-4.2 times as many unique samples under matched numerical parameters while providing a weaker decision contract. For model semantic consistency, checks at $\theta=10^{-3}$ yield 817 acceptances, 19 inconclusive verdicts, and 4 rejections across 840 input--backend instances from 21 tasks. Case studies of two documented compiler faults further demonstrate what these local verdicts can diagnose. 

---
# Proposing Better Rollouts for On-Policy Distillation 

**Authors**: Sirui Chen, Yunzhe Qi, Haobo Xu, Jiaru Zou, Mengting Ai, Zihao Li, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2610.36477)  

**Abstract**: On-policy distillation (OPD) improves language-model post-training by letting the student learn from its own sampled trajectories under teacher supervision, aligning the training process with inference-time behavior. However, the rollout policy in OPD is usually treated as fixed: each round samples directly from the current student. We show that this choice can create a compounding teacher--student mismatch, because it requires the teacher to provide effective supervision on whichever trajectories the student samples, including ones less reachable by the teacher itself. On such trajectories, teacher supervision can become noisy or even misleading, leading to undesirable student patterns such as repetition. Moreover, since the updated student in turn defines the rollout policy of subsequent rounds, these patterns persist and potentially amplify over training. This failure mode suggests that improving OPD requires treating the rollout policy itself as an optimization target, rather than only modifying the distillation objective. We propose Mixture Proposal On-Policy Distillation (MPOPD), which derives a student-anchored, teacher-guided rollout distribution from a local trust-region objective based on relative teacher support. The resulting target is a geometric mixture of the student and teacher policies, and we introduce a proposal policy that approximates this mixture, enabling rollout generation in a single forward pass. Experiments on 16 math, knowledge, and medical reasoning benchmarks show consistent gains over OPD and recent variants. On math reasoning, our method improves average accuracy by 6.91 over OPD on Qwen3-4B-Base while reducing average generation length by 41%, suggesting that rollout policy design provides an effective way to improve OPD. 

---
# FIDES-Live: Evaluating Behavioral Concordance in Sequential Trading Agents 

**Authors**: Arther Tian, Alex Ding, Simon Wu, Aaron Chan  

**Link**: [PDF](https://arxiv.org/pdf/2610.36476)  

**Abstract**: Sequential trading agents produce decisions, orders, execution receipts, account states, and reports that need not agree, even when the final portfolio value is correctly computed. We present FIDES-Live, an operational extension of FIDES from generated strategy concordance to the audit of sequential agent traces. The protocol separates structured decision validity, optional numerical commitment adherence, execution and ledger reconciliation, and receipt- grounded reporting, with explicit coverage and unscorable outcomes. We retrospectively audit ten requested inference routes over 7,299 cryptocurrency account-rounds and a 310-round equity pilot, using real market observations and local paper execution. Cryptocurrency decision validity is 97.35% and structured reporting accuracy is 90.81% among scorable responses, but route-level decision and reporting performance can diverge sharply. Commitment adherence is 94.36% overall; it falls to 62.63% on constraints emitted with nonempty order proposals, compared with 97.56% for valid empty proposals. All retained cash and position transitions reconcile with recorded fills, while execution controls and unavailable quotes explain separate rejection populations. Short-window marked returns provide descriptive context rather than a new test of whether concordance predicts alpha. These observations motivate reporting concordance, evidence coverage, activity, and financial outcomes separately. They do not establish long-run trading skill, semantic faithfulness of explanations, or causal benefits from audit feedback. 

---
# Full Collapse Is Suboptimal: Controlled Partial Neural Collapse for Long-Tailed Recognition 

**Authors**: Hyeontaek Hwang, Cheonghwan Hur, Daeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.36460)  

**Abstract**: Recent long-tailed recognition (LTR) methods incorporate neural collapse (NC) and align features to a fixed Equiangular Tight Frame (ETF) as a uniform target for all classes. However, forcing tail classes to fully collapse over-contracts their samples and weakens generalization. To address this, we propose a controlled partial neural collapse, realized through two coordinated controls on a backbone: the Class-Adaptive Sparse Activation (CASA) classifier and the ETF-constrained adaptive margin (ETF-AM) loss. Rather than forcing a uniform full collapse, this regime allows head classes to collapse along the ETF while tail classes retain within-class diversity. This prevents the tail from being forced into full collapse, which is suboptimal under imbalance. To set the dimensionality of collapse, CASA induces class-related node specialization through feature-conditioned gating so that tail classes occupy their own feature directions. For the clearance axis, ETF-AM applies a frequency-adaptive margin, with smaller margins for head classes and larger for tail classes. On CIFAR100-LT, ImageNet-LT, Places-LT, and iNaturalist-2018, our method achieves the best overall accuracy over prior methods. In the empirical analysis, within-class spread increases monotonically from head to tail classes, consistent with tail classes retaining variability rather than fully collapsing. 

---
# LociArt: A Unified Paradigm for Free-Style Local Video Stylization 

**Authors**: Yuxuan Chen, Feng Xie, Jiagao Hu, Fuhao Li, Zepeng Wang, Fei Wang, Daiguo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.36456)  

**Abstract**: Video stylization typically transforms entire scenes, while local artistic editing requires precise region--style association, spatial control, and temporal consistency. We introduce LociArt, a unified paradigm for free-style local video stylization that explicitly specifies where to stylize and what style to apply while preserving non-target content. At its core, GRNStyle builds on Generative Refinement Networks (GRN) and introduces a polar style-aware mask that embeds style identities as equal-norm directions within selected regions. Complementary representation--geometry conditioning combines encoded mask features with raw-mask spatial support, integrating spatiotemporal context with geometric control. To provide paired supervision, we develop an automated pipeline combining global stylization, video object removal, matting, and compositing, and construct the LociArt-100K Dataset with 100K local stylization pairs. We further establish the LociArt Benchmark, comprising 80 source videos and 84 target masks covering 168 foreground/background stylization cases, with a Style Prototype Similarity (SPS) evaluation scheme for regional style quality and complementary measures of non-target preservation and temporal consistency. Experiments show that GRNStyle outperforms existing methods in regional style quality and spatial control while maintaining strong temporal consistency. Qualitative examples further illustrate joint stylization of multiple regions with the same or different learned styles, despite training only on single-region, single-style pairs. Together, these components provide a unified foundation for learning and evaluating controllable local video stylization. Project page: this https URL. 

---
# Repulsive Population Policy Gradients for Behaviorally Diverse Reinforcement Learning 

**Authors**: Saeed Hedayatian, Harsh Shah, Celine Mneimneh, Stefanos Nikolaidis  

**Link**: [PDF](https://arxiv.org/pdf/2610.36425)  

**Abstract**: The goal of Quality-Diversity Reinforcement Learning (QD-RL) is to find a population of high-performing policies that solve a decision-making problem in diverse ways. Despite recent progress, QD-RL methods predominantly rely on evolutionary operators--including black-box optimizers, mutations, and crossovers--to facilitate exploration and induce diversity. Scaling these methods becomes increasingly costly as policies and behavior spaces grow larger. Building on the Soft QD view of quality-diversity optimization, we introduce \textbf{Re}pulsive \textbf{P}opulation \textbf{P}olicy \textbf{O}ptimization (\textbf{RePPO}), which adapts PPO to jointly optimize a compact population of diverse, high-quality policies using a population-dependent, dynamically shaped reward. On standard continuous-control tasks, RePPO achieves a strong quality-diversity tradeoff compared with state-of-the-art QD-RL methods. Ablations further characterize this tradeoff by examining how each component affects individual quality and collective diversity. To assess RePPO's scalability, we extend our evaluation to learning from pixels and fine-tuning large language models. In pixel-based tasks, where large policies make most traditional QD-RL methods infeasible, RePPO maintains quality on par with standard PPO while achieving substantially higher diversity. Its successful application to language model fine-tuning further demonstrates, for the first time, that QD-RL methods can scale to this domain. Together, these results show that directly optimizing compact populations with RePPO provides a scalable approach to QD-RL across increasingly complex domains. 

---
# Expert lithological labels do not improve learning when used to predict marine sediment properties 

**Authors**: John M. Aiken, William Gilpin, Thorsten W. Becker  

**Link**: [PDF](https://arxiv.org/pdf/2610.36410)  

**Abstract**: Scientific drilling has produced large archives of physical, chemical and magnetic measurements linked to descriptions of the rock recovered. These archives are mostly incomplete: instruments sample at different intervals, recovery is partial, and destructive analyses preclude later measurement of the same material. We train an self-supervised embedding model on the LILY ocean drilling database using six routinely logged properties (bulk density, magnetic susceptibility, natural gamma radiation, and red, green and blue reflectance) across 236,963 observations from 296 boreholes, with a masked variational autoencoder that has a heteroscedastic Gaussian decoder and learnable mask tokens. The embeddings separate geologist-assigned lithologies as well as a variant trained with those labels, and increasing the weight on the labels degrades reconstruction while leaving classification unchanged. The labels are recoverable from the measurements geologists use to assign them, so training on them adds nothing. Lithological categories compress continuous measurements into classes, and compressing them discards what the measurements record about the processes that produced the rock. The embeddings also predict physical properties that were absent from training, on boreholes the model has never seen. While depth should account for this since sediments compact with burial, it does not. On held-out boreholes depth alone reaches a median $R^2$ of -0.02, against 0.20 for the embedding. 

---
# SOMA: A Benchmark for Fine-Grained Egocentric Behavior and Body-Motion Understanding 

**Authors**: Matea Tashkovska, Naël Ouerghemi, Haozhe Qi, Alexander Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2610.36384)  

**Abstract**: Perceiving human body motion is critical for multimodal systems that understand and assist people. Yet current evaluations of Multimodal Large Language Models (MLLMs) largely focus on objects, scenes, and coarse actions, leaving their ability to reason about the body itself poorly assessed. This is especially challenging from the egocentric perspective, in which the body is only partially visible and motion must be inferred from occluded and temporally distributed cues. We introduce SOMA, a benchmark for fine-grained body-motion understanding from egocentric video, comprising 1,632 human-verified questions across three capability axes: body configuration, motion dynamics, and temporal reasoning. We evaluate 37 leading closed-source, open-source, and specialized egocentric MLLMs, complemented by an extensive human evaluation. We find that humans also find this task difficult from video alone (68.5% accuracy), yet clearly outperform the strongest MLLM (Qwen3.6-27B), which reaches 50.0% zero-shot accuracy. Our error analysis shows that MLLMs often misread the wearer's own body, misjudge events in time, and are prone to biases such as favoring the right hand or under-counting repetitions. To narrow the gap to human performance and support further progress, we introduce a training dataset of aligned video, body motion, and language supervision, and a multimodal baseline that jointly learns from these modalities. By identifying common failure modes and missing capabilities in current MLLMs, we hope SOMA will inspire future research on body-motion understanding and multimodal reasoning. Full benchmark, training data, and code will be released at: this https URL 

---
# HIDRA: Hierarchical Dual-Routing Attention for Replay-Free Lifelong Imitation Learning 

**Authors**: Fanqi Yu, Matteo Tiezzi, Cigdem Beyan, Tommaso Apicella, Vittorio Murino  

**Link**: [PDF](https://arxiv.org/pdf/2610.36374)  

**Abstract**: Robotic agents operating in real-world environments must adapt continuously from a stream of multimodal demonstrations, often under constraints that preclude storing or revisiting past data. In this replay-free, single-pass lifelong imitation learning setting, sequential updates can progressively distort the representation space, making expert-based approaches particularly sensitive to routing interference, causing expert selection to degrade over time. We propose HIDRA, a structured routing framework that mitigates this issue through a hierarchical dual-attention mechanism, which decouples instruction-level expert retrieval from context-dependent refinement. To further stabilize routing under representation drift, HIDRA introduces a key-level geometric regularization that enforces both alignment within tasks and separation across tasks. The resulting approach enables reliable expert selection while supporting an expandable set of experts for continual adaptation. We evaluate HIDRA on several LIBERO benchmark suites, including more challenging variants with heterogeneous task sequences and paraphrased language instructions. Our approach consistently outperforms both replay-free and replay-based baselines, improving AUC and forward transfer while maintaining low forgetting, with stronger gains under high interference and language variation. The code is available at: this https URL. 

---
# Reliability-Aware Short-Term Glycemic Risk Prediction Using an Anchor-Based Comparative Framework of Sequence, Multimodal, and State-Updating Digital-Twin Models 

**Authors**: Pir Bakhsh Khokhar, Carmine Gravino, Fabio Palomba  

**Link**: [PDF](https://arxiv.org/pdf/2610.36363)  

**Abstract**: Anticipating short-term hypoglycemic and hyperglycemic risk is essential for proactive diabetes management. However, existing machine learning approaches often use inconsistent evaluation protocols and emphasize predictive performance over reliability. This study introduces an anchor-based benchmarking framework integrating longitudinal glucose sequences and multimodal patient information. Using the T1DiabetesGranada dataset, 5,503,441 standardized prediction anchors were constructed from 736 individuals with type 1 diabetes to evaluate hypoglycemia and hyperglycemia prediction at 30-, 60-, and 120-minute horizons. Three neural architectures, a Hybrid LSTM-Transformer, a Multimodal Digital Twin, and a State-Updating Digital Twin with gap-aware longitudinal state progression, were compared under identical patient-level validation protocols. Beyond predictive performance, the evaluation examined calibration, classification thresholds, explanation reliability, cross-model disagreement, and robustness to controlled perturbations. No statistically significant predictive differences were observed after Holm correction, although the models exhibited complementary reliability characteristics. The Multimodal Digital Twin achieved the lowest mean Brier score, while the State-Updating Digital Twin demonstrated the lowest mean expected calibration error (ECE-10). The Hybrid LSTM-Transformer showed strong explanation faithfulness and stability under its temporal attribution protocol. These findings highlight the importance of evaluating digital-twin models beyond predictive accuracy and emphasize the need for reliability assessment and external validation before clinical deployment. 

---
# Enhancing Clinical Classifier Robustness with Deep Representation Learning Based OOD Detection on DNA Methylation Data 

**Authors**: Md Zahangir Alom, Quynh T. Tran, Breuer Alexandar, Brent A. Orr  

**Link**: [PDF](https://arxiv.org/pdf/2610.36344)  

**Abstract**: DNA methylation (DNAM) array-based tumor classification is a powerful clinical diagnostic tool, but models trained on fixed reference cohorts can produce faulty predictions for tumor types not represented in the training distribution. We propose an end-to-end deep representation learning-based out-of-distribution (OOD) detection framework for DNAm array data. Deep neural network (DNN) classifiers were trained on public sarcoma (SARC) (GSE140686) and brain tumor (BT) (GSE90496) cohorts, and feature vectors were extracted from bottleneck layers. Five OOD methods, including 1dimension (1D)-Subspace Analysis (1D-SSA), K-Nearest Neighbors (KNN), Ridge Regression, Mahalanobis Distance (MD), and Sum of Squared Distances (SSD), were explored with 200 and 500-dimensional embeddings. Spectral discrepancy was used to discriminate in-distribution (IND) from OOD samples. Performance was measured via cross-validation, holdout samples, and an independent clinical dataset. 1D-SSA was the most effective approach. In the sarcoma-IND/brain tumor-OOD setting, the framework achieved an AUROC of 0.975 and True Negative Rate (TNR)@95TPR of 0.930, and an AUROC of 0.9569 and TNR@95TPR of 0.8660 for the brain tumor-IND/sarcoma-OOD setting. On the independent clinical test set, the method yielded a mean AUROC of 0.976, a DE of 0.032, an FPR@95TPR of 0.065, and a TNR@95TPR of 0.935. These results show deep representation-based OOD detection as a practical quality-control layer for clinical-grade methylation-based tumor classification systems. 

---
# CrowdCC: Jointly Benchmarking Corruption Robustness and Deployment Quantization in Crowd Counting 

**Authors**: S Mahmudul Hasan, Lu Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.36340)  

**Abstract**: Crowd-counting models deployed on edge devices face two shifts: inputs are degraded along the capture and transmission pipeline, while models are quantized to satisfy resource constraints. Yet corruption robustness and deployment quantization are typically evaluated separately. We introduce CrowdCC, a controlled corruption benchmark that jointly evaluates these effects using 23 deployment-motivated corruptions at three severity levels, four crowd-counting datasets, and 26 models using A100 TensorRT engines at FP32, FP16, and post-training INT8 precision, paired with throughput and energy measurements from corresponding Jetson Orin NX configurations. The corruptions span six families, with head annotations adjusted when geometry or visibility changes. Our evaluation shows that capture-chain corruptions consistently produce the largest degradation across the evaluated models, while crowd density affects absolute and relative error differently. Clean accuracy does not fully predict corrupted performance. INT8 alters corruption robustness substantially more than FP16, with no systematic direction, and does not consistently improve latency or energy. INT8 inference efficiency is strongly associated with reformat overhead in the compiled engine, while slowdowns are also observed under limited INT8 coverage. Precision changes corrupted-error rankings and robustness-cost frontiers, showing why model selection must consider corruption robustness and measured deployment cost together. The code is available at this https URL. 

---
# Detecting and Mitigating Query-Based System Prompt Leakage Attacks 

**Authors**: Tushar Nayan, Pratik Poudel, Qian Lou, Yuede Ji, Jason Liu, Ruimin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.36295)  

**Abstract**: Large Language Models (LLMs) rely on proprietary system prompts to enforce task directives, safety policies, and access controls. These prompts are vulnerable to query-based leakage attacks, where black-box adversaries extract confidential instructions through adaptive, multi-turn interactions. Existing defenses often assess queries independently, missing malicious behavior that emerges across interactions.
We present AdvGuard, a model-agnostic framework for detecting and mitigating system prompt leakage. AdvGuard models cross-query relationships using a bounded semantic interaction graph, statistically filters candidate components, and converts them into line graphs for graph convolutional network (GCN) classification. This approach captures multi-hop patterns in query similarity that distinguish coordinated extraction from benign workflows. A lightweight prompt-level detector provides protection before sufficient interaction history accumulates, and the framework supports black-box LLM deployments.
In mixed-query evaluations, AdvGuard improves recall by up to 36% over the strongest baseline while maintaining a false positive rate of at most 3.4%. In multi-turn workloads, it achieves 97.23%-99.1% F1 with 100% recall. Leave-one-attack-family-out tests yield 99.98% mean recall and 99.99% mean F1, with no observed false positives. Integrated into the adaptive PLeak and LeakAgent attack loops, AdvGuard limits exact-match leakage to 0.00-0.01 and substring-match leakage to 0.01-0.09 across evaluated victim models. These results demonstrate robust detection of unseen attack families and substantially reduced prompt reconstruction. 

---
# BUSTA: Teaching Small Language Models to Reason About Binary Execution 

**Authors**: Ritika Poddar, Aidan Fennelly, Deirdre Murphy, Eric Wolos, Corre L. Steele, Michael Doyle  

**Link**: [PDF](https://arxiv.org/pdf/2610.36288)  

**Abstract**: Large Language Models (LLM) are increasingly applied to Reverse Engineering (RE) workflows. RE, however, differs from common LLM coding tasks as it requires models to reason about low-level program behavior. Furthermore, the deployment of tooling in resource-constrained sandboxed RE environments limits the ability to use frontier-scale solutions. We examine whether a targeted LLM training approach can transfer distinct RE skills to open models via inverse basic-block execution: given an x86-64 basic block and a desired output register state, the model predicts the input register values that produce it. Fine-tuning \texttt{Qwen3-1.7B} and \texttt{Nemotron-3-Nano} with Group Relative Policy Optimization (GRPO) raises knowable-register prediction accuracy from 73.2\% to 85.3\% and from 63.3\% to 95.2\%, respectively. Frontier models still score highest (up to 99.2\%), but our fine-tuned \texttt{Nemotron-3-Nano} comes within four points at a fraction of the parameter count. We release both fine-tuned models and the training dataset. 

---
# cuArms: GPU-Accelerated Conflict-Based Planning for Multi-Arm Manipulation 

**Authors**: Philip Huang, Fabio Ramos, Jiaoyang Li, Caelan Garrett, Lars Johannsmeier  

**Link**: [PDF](https://arxiv.org/pdf/2610.36250)  

**Abstract**: Multi-arm manipulation has the potential to dramatically improve the throughput, efficiency, and dexterity of many manipulation tasks. Recent work coordinates multiple robot arms in a shared workcell by allocating and scheduling tasks for greater parallelism, jointly planning multi-arm motions, or directly learning multi-arm policies. So far, these approaches are either too slow, rely on domain-specific assumptions, or scale poorly with the number of robots and task horizon. Thus, we introduce cuArms, a conflict-based planner co-designed with GPU parallelism that efficiently decomposes the search space using reusable multi-modal roadmaps for individual robots and resolves conflicts and deadlocks through a lazy search. To jointly optimize task assignment, schedules, motions, and manipulation skills, we integrate a high-level scheduler that proposes candidate skeletons for cuArms to ground into coordinated manipulation plans. Experimental results show that our method establishes a new state of the art, with real-time initial solutions and anytime improvements in both motion quality and scheduling, without days of domain-specific training required by learning approaches. Videos of both simulated and real-world industrial tasks are available on this https URL. 

---
# Freeze, Don't Clone: Fine-Tuning with Expressive Policies Without Retaining Offline Data 

**Authors**: Sami Nur Islam, Johan Obando-Ceron, Doina Precup, Aaron Courville, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2610.36247)  

**Abstract**: Offline-to-online reinforcement learning uses offline pretraining to accelerate subsequent online learning. While many methods assume the offline dataset remains available throughout fine-tuning, there are many settings where this is not the case. Prior work attributes failures in this setting primarily to critic recalibration under distribution shift. We identify a complementary actor-side failure: once the offline data disappears, continued behavioral cloning (BC) no longer preserves the offline behavior distribution and instead refits the behavioral model to the agent's limited, and nonstationary, online experience. We demonstrate that stopping online behavioral fitting while preserving the pretrained expressive prior substantially improves no-retention fine-tuning across implicit flow Q-learning (IFQL), flow Q-learning (FQL), and action-chunked RL (QC) on seven OGBench domains. We trace this failure to a behavioral feedback loop: continuing online behavioral cloning rapidly erases the pretrained prior, inducing representational collapse and action diversity loss that starves the critic of useful exploration. Freezing the prior breaks this cycle, preserving offline coverage while allowing value learning to drive continuous improvement. 

---
# Agent or Human? Detecting Commits by AI Coding Agents in the Absence of Authorship Signatures 

**Authors**: Kosei Horikawa, Hao Li, Yutaro Kashiwa, Youssef Esseddiq Ouatiti, Mohammed Sayagh, Bram Adams, Hajimu Iida, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2610.36229)  

**Abstract**: AI coding agents such as Claude Code, GitHub Copilot coding agent, and Cursor now commit directly to open-source repositories. Empirical studies of these agents identify their commits through authorship signatures, such as bot accounts and Co-authored-by trailers. These signatures are optional, can be removed, and are often attached to pull requests rather than to individual commits. Hence, it is unclear whether Agent commits can be told apart from Human commits once the signatures are gone. We investigate the signals and limits of signature-free detection using 7,837,946 commits from 1,226 repositories, covering five coding agents and Human commits. After signatures are masked, detectors rank Agent commits above Human commits with a ROC AUC of up to 0.859, and most of this signal comes from message verbosity and change size rather than from the code itself. On the shared sample, gpt-6-sol without fine-tuning reaches AUCs similar to XGBoost and TF-IDF. At the observed Agent share, however, the best detector on the full test set reaches a precision of only 0.332. Detection transfers poorly to held-out Claude Code and Copilot, with TF-IDF reaching an AUC of only 0.524 for Claude Code. Temporal changes also depend on the agent, e.g., Copilot AUC drops and later recovers, while more training history helps Claude Code but not Copilot. These results indicate that signature-free detection can prioritize commits for review but cannot replace recorded attribution, and that studies relying on signature-based labels should treat them as a lower bound on agent activity. 

---
# Directional Statistics on Circles and Spheres: A New Angle on Data 

**Authors**: Siddhesh Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2610.36227)  

**Abstract**: Two compass bearings near north can have an ordinary arithmetic mean pointing south: the difficulty is that angles live on a circle, not a straight line. Using real wind observations, diffusion-MRI orientations, and geometric examples from machine learning, this article introduces circular and axial averages, directional variability, and probability models on circles and spheres. The guiding principle is to identify the geometry and symmetries of the observations before choosing a statistical method. 

---
# Structure, Not Sequence: Tracking and Validating Folding Mechanisms in ESMFold's Trunk with Sparse Autoencoders 

**Authors**: Bohan Liu, Xiaohan Fan, Wenqian Ye, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36226)  

**Abstract**: Structure predictors like ESMFold fold proteins using neural network computation and have transformed structural biology. However, the driving mechanism from amino-acid (AA) sequence to 3D structure is insufficiently understood, and biological meaning remains locked in the dense network. Sparse autoencoders (SAEs) have shown promising results in obtaining interpretable concept features over an overcomplete basis of intermediate protein sequence-model representations. Yet folding models, which predict physical shape from AA sequence semantics, have no such account. We hypothesize that the features carrying the folding computation evolve in the 3D structural regression process. In this paper, we interpret the folding trunk in three steps. First, a pilot analysis on the representation distribution across modules and recycles locates distinct spatial computation in the trunk. Second, the dictionaries expose a division of labor: the single representation carries inherited semantics, and the pair representation computes the pairwise geometry. Third, we interpret the dictionary at scale with a reasoning language model, trace how the concepts compose across blocks by gradient-based attribution, and follow the interpreted code to the model's folding output, verifying where the computation aligns with, or departs from, known biochemistry. Together, the sparse dictionaries we train capture the relational geometry the trunk computes beyond the sequence model, and their concept activations forecast the model's own fold quality, turning the black-box folding trunk into a sparse, interpretable, and scientifically verifiable circuit. 

---
# Counterfactual Profile Perturbations for Recommendation Explanations: A Reproducibility Study 

**Authors**: Meysam Varasteh, Elizabeth McKinnie, Robin Burke  

**Link**: [PDF](https://arxiv.org/pdf/2610.36208)  

**Abstract**: This reproducibility paper explores the counterfactual explanation technique known as Learning to eXplain Recommendations (LXR). Prior work has shown that LXR can reliably identify perturbations that, if applied to a user's profile, cause the user's top ranked item to no longer be recommended, thereby explaining the item's recommendation as a function of the user's rating choices. In this paper, we extend this work in multiple ways. We expand the scope of the original LXR study to look at explanations for items other than the most highly recommended item, finding that LXR is also effective at this version of the task. We consider alternate baselines for LXR, including item popularity and latent factor similarity, and we see that for some tasks these alternatives work surprisingly well. We also examine the impact of item popularity, finding that LXR offers better explanation quality (shorter perturbations) when less popular items are being explained. Finally, we conduct a brute-force exploration of a subset of the perturbation space in order to estimate a lower bound for optimal perturbation size against which LXR and other such methods can be compared. 

---
# Concorde: Geometry-Aware Link Prediction via Decoupled Energy Minimization 

**Authors**: Ege Demirci, Ambuj Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.36204)  

**Abstract**: Link prediction remains difficult in sparse graphs, where diffusion heuristics lack connectivity and geometric methods suffer spectral instability when coupling learned representations with diffusion operators. We introduce Concorde, an energy-based framework that resolves this by decomposing link likelihood into three structurally decoupled branches: (i) a parameter-free spectral anchor that establishes permutation-invariant topological stability; (ii) a mixed-curvature residual on a learnable $\mathbb{H}^{d_h}_{c_h} \times \mathbb{S}^{d_s}_{c_s}$ product manifold, regularized by a feature-aware Ollivier-Ricci curvature proxy; and (iii) a continuous vector field on the tangent bundle of hyperbolic space for non-local structural alignment across topological gaps. The branches are fused via learned gating and trained with an N-pair ranking loss aligned with retrieval-style evaluation. Across five benchmarks spanning sparse, dense, featureless, and attributed regimes, Concorde establishes state-of-the-art early retrieval precision, nearly doubling P@10% over leading baselines in sparse settings. 

---
# Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss 

**Authors**: Oz Amram  

**Link**: [PDF](https://arxiv.org/pdf/2610.36202)  

**Abstract**: The Persona Selection Model (PSM) posits that post-training selects a persona the model learned during pre-training. Work on subliminal learning has shown that models can impart covert signals (like "love of owls") into semantically unrelated outputs like strings of numbers, and that students trained on these outputs inherit those traits. In this work I show that by evaluating the likelihood of such outputs under the base pre-trained model with an appropriate header, a method I call persona-header likelihood evaluation (PHLE), one can decode these subliminal signals and classify between them. For example, PHLE distinguishes number sequences generated by a post-trained model prompted with "You love owls" from those prompted with "You love eagles", and distinguishes answers from a model with a secret desire to harm the user, filtered to remove any noticeable misalignment, from those of a model with no system prompt. In both cases prompted classifiers, including the generating model itself, remain near chance. The signal partially persists across model families, with the Qwen base model reading signals generated by Olmo and vice versa. This is evidence for a non-trivial prediction of the PSM, and the method may be practically useful for auditing the hidden objectives of models. 

---
# count-catcher: Lean Theorems from Model Counting Certificates 

**Authors**: Johannes K. Fichte, Stefan Szeider  

**Link**: [PDF](https://arxiv.org/pdf/2610.36196)  

**Abstract**: We present count-catcher, a tool that turns a CPOG certificate of a count by the exact model counter d4 into a Lean theorem that a CNF formula defined in Lean has exactly N models over V variables. With an encoding lemma, this theorem becomes a theorem about the number of combinatorial objects. A CPOG certificate consists of a Boolean circuit equivalent to the formula and a proof of equivalence. count-catcher checks in Lean that the formula implies the circuit (forward check) and the circuit implies the formula (reverse check). For the forward check, lrat-catcher streams a refutation through the LRAT checker of Lean's standard library. The reverse check and the validity check of the circuit are computations proved correct in Lean. Lean runs all three checks as compiled code, so the theorem depends on Lean's kernel, compiler and runtime, as does CPOG's verified checker. We combine the theorem with results from Mathlib, for instance with Burnside's lemma to count graphs up to isomorphism. We prove the number of Latin squares of order 8 as a weighted sum of hundreds of certified counts from about eight terabytes of certificates. We turn about half of the 2022 Model Counting Competition formulas into theorems, with the three checks taking about a quarter of the time of CPOG's verified checker. We found that CPOG's unverified checker reported wrong counts, and removed its verified checker's dependence on two unproved lemmas. End to end, a theorem costs about the same as a checked count. 

---
# Beyond the Estimate: Wiener-Informed Decoding for Easier Training of Pixel-Space Flow Models 

**Authors**: Ziqiao Wang, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2610.36192)  

**Abstract**: Learning velocity fields directly in pixel space is difficult and slow. We show that coarse second-order image statistics can ease this task: frozen Wiener estimates improve velocity learning, with gains extending beyond the linear predictions they explicitly provide. Moreover, we observe similar but complementary gains from frozen bases and pixel decoding. These observations motivate Wiener-informed Decoding (WiD), which uses statistical knowledge both for direct prediction and to organize decoder computation without modifying native blocks. WiD combines a trainable Wiener prediction path with coefficient interfaces that inject features into the native decoder and return its predicted coefficient corrections, with these components initialized with data statistics. All introduced parameters, including the spectrum and coordinate transforms, are trained jointly under the original pixel-space velocity objective. Across model scales and decoder architectures, WiD accelerates training and improves generation quality with modest parameter overhead. It also yields smoother gradient trajectories and remains effective with representation alignment. WiD improves class-conditional and text-to-image generation across different resolutions. On ImageNet 256 x 256, PixelDiT-XL with WiD reaches FID 1.53 after 500 epochs, achieving comparable quality to PixelDiT with 37.5% fewer training epochs. Statistical denoisers contribute beyond the estimate: their computational structure can also ease pixel-space training. 

---
# Can AI Guide Testing Without Generating Tests? A Comparative Study of AI-Assisted Unit Testing 

**Authors**: Zihan Fang, Ron Asoulin-Handelman, Helen Wu, Yu Huang, Gina R. Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.36185)  

**Abstract**: Recent advances in generative AI have enabled assistants to generate increasingly sophisticated executable unit tests. However, novices may accept these outputs without independently identifying what to test, evaluating test-suite completeness, or verifying correctness, potentially limiting opportunities to develop essential testing skills. Although prior work has explored AI-assisted test generation and feedback-oriented programming tutors, little is known about whether AI can support novices' testing practices without generating tests for them. To investigate this, we introduce TestCompass, a generative AI tutor that provides targeted feedback on weaknesses in students' test suites without generating executable tests. In a classroom study with 18 students, we compared TestCompass with a general AI assistant across two Java unit-testing tasks, examining students' testing behaviors, test-suite quality, and subjective experiences. We found that participants using the general-purpose AI assistant produced more tests, whereas TestCompass participants spent more time developing and verifying tests themselves. However, most test-suite quality measures were comparable between conditions, and TestCompass participants achieved similar requirement coverage with significantly fewer test methods ($p=0.012$) in one task. Neither group reported short-term improvements in perceived testing knowledge or confidence. These findings suggest that feedback-oriented AI can support meaningful testing outcomes while preserving opportunities for hands-on testing practice, offering a complementary approach to AI-assisted testing education. 

---
# Stratum: Repository Memory for Coding Agents Beyond the Context Window 

**Authors**: Yuvraj Sehgal, Mahsa Panahandeh, Naser Ezzati-Jivan  

**Link**: [PDF](https://arxiv.org/pdf/2610.36153)  

**Abstract**: Context windows have grown a thousand-fold, yet coding agents still start every session with no memory of their past work. They re-read files, retry failed approaches, and forget root causes they established hours earlier. This is not a context-length problem but a memory-architecture problem. Our hypothesis: coding agents need persistent, experience-aware repository memory, so that knowledge survives from one session to the next. Three measurements on real repositories motivate the design. First, mid-size projects already exceed the largest production context windows, and storing knowledge inside the model's key-value (KV) cache costs $10^4$ to $10^5$ times more bytes than storing it as indexed text. Second, across 752 real bug-fix commits, the median fix edits two files, and these files plus their direct imports amount to less than 10% of the repository; the bottleneck is selection, not capacity. Third, even with transitive import reachability and test links, 47% of co-edited file pairs remain structurally disconnected; the knowledge needed for selection must be mined from history and experience. We sketch Stratum, a tiered memory that lives with the repository and serves it through page-safe tools: every response states what was returned, what remains, and what it cost. Retrieval over a store of one million nodes takes 1.7 ms at the median. In a multi-session experiment on eight real bugs, a fresh agent that inherited a prior session's memory, written by a different model, identified the root-cause file in 21 of 24 trials versus 6 of 24 for a retrieval-only baseline (Fisher $p<10^{-4}$), at 4.5x lower median token cost. With the notes of all eight investigations pooled in one store, success remained 8/8. 

---
# Replication or Corroboration? An Axiomatic Framework for Evidential Dependence in Multi-Agent Debate 

**Authors**: Hoang-Loc Cao, Truong Thanh Hung Nguyen, Phuc Truong Loc Nguyen, Vo Thanh Khang Nguyen, Phuc Ho, Hung Nguyen, Duc Pham, Long Tran-Thanh, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.36151)  

**Abstract**: In multi-agent debates, matching reports can arise for two different reasons: agents may repeat the same evidence (replication), or they may provide separate evidential roots that support the same claim (corroboration). A judge must distinguish these two cases. Counting report occurrences gives extra weight to relays, paraphrases, alias resubmissions, and excerpt splits, while merging all matching content avoids this only by also removing genuine corroboration. We formalize this tension as a minimal judgment game in which acquiring new evidence is costly, but replication is cheap. We show that root replication invariance (RRI) is exactly threshold-uniform replication immunity. Under RRI, replication-only deviations cannot change the verdict, and positive-cost replication is strictly dominated, whereas occurrence-sensitive and content-keyed judges admit profitable deviations. We also establish an information boundary: if a judge's observation interface cannot distinguish a replica from a matched fresh root, no judge can both ignore the replica and respond strictly to the fresh root. To preserve this distinction, we introduce the lineage quotient, which removes wording, identity, occurrence multiplicity, and arbitrary root names while keeping distinct roots separate. On orbit-connected domains, it is the coarsest representation sufficient for every deterministic judge invariant to replication and root-label renaming. We further lift this construction to quantitative bipolar argumentation frameworks (QBAFs), characterize strict fresh-root response, prove its propagation along sign-consistent paths under DF-QuAD, and derive a tight closed-form response modulus. Exhaustive audits over 28,224 direct-star profiles, 2,052 multi-layer DAGs, with matched interventions on 277 AVeriTeC-derived profiles, show the predicted separation between replication and corroboration under a supplied partition. 

---
# Physical structure as an organizing principle for universal Hamiltonian learning 

**Authors**: Mingkang Liu, Zhanghao Zhouyin, Guanghui Wang, Daye Zheng, Lei Shen, Qiangqiang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36135)  

**Abstract**: Density functional theory (DFT) gives the electronic structure of materials, but its cost limits large-supercell calculations and materials screening. Machine-learned Hamiltonians bypass the self-consistent cycle but are usually system-specific; universal ones could cover new compositions and disordered solids. Such models must generalize across chemistry and keep downstream properties physical, while staying inexpensive for large cells. Here we introduce UniTB, a universal E(3) equivariant Hamiltonian representation framework that integrates physical priors rooted in density-functional tight-binding theory with deep learning. UniTB's network, which corrects a prior Hamiltonian, generalizes across chemistry through a mixture of experts routed by that prior and a charge-equilibration head as in self-consistent-charge tight binding. A projection bounds band shifts from the prior, suppressing spurious states and keeping downstream properties physical. We provide the fastest SO(2) tensor-product GPU kernels for the network. The variant UniTB-Flash assembles its prior from two-center tables at $O(N)$ cost for $N$ atoms, against $O(N\log N)$ on a grid. Trained on 29,242 crystals, UniTB reaches a 0.48 meV matrix-element error on nonmagnetic test crystals spanning 68 elements. Without amorphous training data, both models agree with DFT optics and conductivity of amorphous Ge-Sb-Te. UniTB-Flash ranks candidates with no significant difference from UniTB, whose property values are more accurate; together they screen 16,195 crystals for five application domains of industrial relevance. 

---
# When Connectivity Misleads: Guiding Agentic Search with Contrastive Evidence Paths 

**Authors**: Junchen Li, Liang Xu, Rongzheng Wang, Qizhi Chen, Junhao Wen, Shuang Liang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36134)  

**Abstract**: Search agents can use cross-document graphs to discover sources through multi-hop connections and choose subsequent actions based on candidate rankings and relationships within a limited observation window. However, documents may rank highly because their numerous connections make them easier to reach, without necessarily providing more valuable evidence. Graph diffusion methods such as Personalized PageRank (PPR) aggregate propagation contributions from multiple sources, allowing highly connected distractor nodes to crowd the observation window. Flat node scores also obscure cross-document evidence trails, limiting continued exploration. We propose Contrastive Evidence Paths (CEP), which scores candidates by their strongest individual paths on a query-conditioned document graph and calibrates these scores against the same graph's background visitation distribution to mitigate structural bias. CEP ranks candidates by the calibrated scores and retains the corresponding source paths. Building on CEP, we develop Structured Corpus Interaction (SCI), which organizes candidates and their connecting paths into a structured Map, enabling agents to verify source documents and pursue missing information through expansion from anchor documents. Experiments on BrowseComp-Plus and TREC-COVID show that SCI achieves state-of-the-art answer accuracy and nDCG@10 with the lowest generation cost. Compared with the strongest baselines by task performance, maximum reductions in tool calls and total generation costs exceed 50% and 40%, respectively. Further analysis confirms that SCI with CEP suppresses ranking inflation induced by structural connections and improves gold recall and exploration efficiency. 

---
# Context Is a Pipeline, Not a Window: Information Attrition in LLM Agent Harnesses 

**Authors**: Yuvraj Sehgal, Mahsa Panahandeh, Naser Ezzati-Jivan  

**Link**: [PDF](https://arxiv.org/pdf/2610.36114)  

**Abstract**: LLM agents are judged by two numbers: the size of the context window, and how often the final answer is correct. Both numbers miss a third thing: how much of a tool's output is lost before the model ever sees it. We measured this loss in a real tool-using agent. The agent finds the cause of injected faults by reading Linux kernel traces. The context window was almost empty: across 720 runs, the single largest prompt used only 5.75% of it. Yet a third of all tool results (3,827 of 11,189) were cut before they reached the model, and every cut result became broken JSON. The harness cut each result by slicing its JSON text at a fixed length. This does not shorten a result evenly. It deletes whatever data sits near the end. One field reached the model in only 4.7% of calls. A second field, in the very same results, reached the model in 99.3% of calls. Which field held the fault's evidence depended on when the faulty process started: early or late in the trace. The agent's diagnosis accuracy matched this same split. So we ran controlled experiments to test one question: did the data loss cause the drop in accuracy? This paper makes four contributions. First, we model an agent's context as a five-stage pipeline, with a measured failure at each stage. Second, we replay 864 real tool results, including one from a production trace where the shipped cut deletes all 128 records of the fault's evidence. Third, a controlled experiment shows that the cut's real cost appears first as extra work, not as wrong answers: 43% more tool calls, twice the computation, and yield falling from 0.61 to 0.31. The agent still finds the right answer when it has another way to reach it. Fourth, a survey of seven agent frameworks finds two that cut tool output the same unsafe way. Whether the data cut truly causes the accuracy drop is still an open question. 

---
# Scaling and Renormalization in Neural Networks: An Effective Renormalization-Group Description of Grokking 

**Authors**: Min Chen, Junqi Wang, Zeguan Wu, Wenhao Huang, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36113)  

**Abstract**: Grokking is a delayed transition from memorization to generalization in deep neural networks (NNs). We examine whether scaling and renormalization group (RG) theories can describe this dynamics, establishing a quantitative physical framework for neural generalization transitions. For modular addition on a two-layer NN, an ensemble of trained models defines an order parameter (the squared weight norm) and fluctuation proxies for the heat capacity and the susceptibility, with training epoch and weight decay serving as thermal and field control coordinates. To bridge neural architectures to real-space RG, we construct an exact coupling matrix via a second-order expansion of the loss with respect to per-neuron norms and use its Fiedler vector to define an effective geometry. This spectral ordering yields a monotonic spatial decay of coupling strength that is absent under random permutations. We then execute real-space RG by block-spin aggregating adjacent neurons and parameterizing the coupling matrix into off-diagonal band strengths. Linearizing this local RG map yields relevant scaling exponents and five derived critical exponents, with direct empirical estimates matching RG predictions within 1-10%. Overall, our framework provides a consistent effective description of grokking, suggesting the scaling and renormalization group formalism as a tool to investigate NN dynamics. 

---
# Dirichlet Process Mixtures within Semiparametric Bayesian Networks: A Novel Conditional Density Estimation Approach 

**Authors**: Victor Alejandre, Pedro Larrañaga, Concha Bielza  

**Link**: [PDF](https://arxiv.org/pdf/2610.36110)  

**Abstract**: Semiparametric Bayesian networks (SPBNs) are flexible probabilistic graphical models for representing joint distributions of continuous variables through lower-dimensional conditional probability distributions (CPDs). Existing SPBN models use kernel density estimators (KDEs) for non-parametric CPD modeling, but their one-component-per-observation mixture representation can lead to increased inference costs and density estimation errors, which can be limiting. In this work, we define a new CPD modeling approach for non-parametric components based on Bayesian non-parametric Dirichlet Process Mixtures (DPMs) within SPBNs. DPMs provide a data-adaptive clustering of observations, yielding a more compact representation while retaining the flexibility of non-parametric density estimation, unlike KDE. This results in lower inference costs while improving density estimation performance. We implement the proposed DPM-based SPBN (DPM-SPBN) by extending the open-source PyBNesian package and evaluate it on synthetic and real-world data. For a given network structure, DPM-based CPDs significantly outperform state-of-the-art Bayesian network models for continuous variables in density estimation, achieving lower absolute log-likelihood errors on synthetic data. We further investigate the impact of the proposed CPD modeling on structure learning methods relying on CPD-based scores. DPM-based SPBNs achieve lower structure learning and density estimation errors than KDE-based SPBNs. Experiments on real-world data corroborate these findings, identifying DPM-based CPD modeling as the most effective alternative among the evaluated non-parametric approaches for continuous variables. 

---
# Log-To-Leak: Prompt Injection Attacks on Tool-Using LLM Agents via the Model Context Protocol 

**Authors**: Yuepeng Hu, Chongyu Fan, Sirat Samyoun, Jian Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.36108)  

**Abstract**: LLM agents increasingly access external tools through the Model Context Protocol (MCP), but the natural-language metadata used to describe those tools creates a prompt-injection surface. Within this established tool-poisoning surface, we study Log-To-Leak, an additive post-hoc logging attack that induces a separate attacker-controlled call after the final legitimate call. Its payload targets the user query, MCP server name, server response, and agent response. We operationalize the injected directive through Trigger, Tool Binding, Justification, and Pressure, and separately measure call success, payload fidelity, and preservation of legitimate behavior. We evaluate five MCP servers and four LLM agents using five metrics spanning effectiveness, utility, and efficiency. Across 18 model-server pairs with valid tool execution, Log-To-Leak achieves an attack success rate (ASR) of 62.64-100% and conditional logging accuracy (LA) of 76.55-95.64%; 16 pairs have absolute target-task completion changes of at most 6.63 percentage points. Three evaluated defenses leave residual ASR above 55% on GitHub MCP with GPT-4o, while an instruction detector flags 85.53% of benign descriptions. These results show that successful task completion can coexist with unauthorized post-hoc disclosure and motivate host-controlled authorization for secondary tool calls. 

---
# How Much Does the Hint Matter? Scope Conditions for Hindsight Self-Distillation in Financial Reasoning 

**Authors**: Johnny Yu, Cyril Gorlla  

**Link**: [PDF](https://arxiv.org/pdf/2610.36082)  

**Abstract**: Financial institutions operating under data-residency constraints are restricted to open-weight models small enough to locally host, which trail frontier systems on multi-step quantitative reasoning. On-policy distillation narrows such gaps by supervising a student on its own rollouts, but traditional OPD requires a strong external teacher, and on-policy self-distillation (OPSD) conditions on a full reference solution. We ask how much of that signal is necessary. We present Fin-SD, a localized hindsight self-distillation recipe for multi-step reasoning. Given a problem the student fails, the student itself identifies the first incorrect reasoning step and writes a short critique of that step. We truncate the student's trajectory immediately before the error (aiming to obtain a correct prefix), condition a frozen copy of the student on the same prefix plus the hint, and distill that hint-conditioned teacher into the student over a short rollout under reverse KL. No external model contributes a gradient. We apply Fin-SD to GPT-OSS-20B on a custom multi-step financial reasoning benchmark that requires deriving formula inputs from realistic scenarios via accounting conventions. We find: (i) replacing the critique with a generic-feedback control matches Fin-SD's entire 3.03 pp accuracy gain over the base model on the FinanceReasoning Hard subset, suggesting the gain comes from where supervision is applied, not hint content; (ii) the model writing the hint does not need to be stronger than the model receiving it: hints from same-family GPT-OSS-120B, at roughly 6x the parameters, yield no materially greater benefits than self-written hints; and (iii) Fin-SD matches on-policy self-distillation from full reference solutions on accuracy while generating 9.5% fewer tokens. With Fin-SD, general reasoning on MMLU-Pro and a fourteen-metric compliance audit on FinTrust show no material regression. 

---
# Rollback: Experience Backtracking in World Models for Policy Improvement 

**Authors**: Jiayi Luo, Yang Zhang, Liliang Chen, Nan Wang, Chen Gao, Boxiang Qiu, Zhuoxuan Li, Baorui Peng, Chuankang Li, Yuxiang Yan, Si Liu, Qingyun Sun, Shuicheng Yan, Maoqing Yao, Jianxin Li, Guanghui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2610.36068)  

**Abstract**: Action-conditioned world models offer a scalable alternative to costly real-world interaction for improving robot policies. However, existing world-model-based methods largely treat world models as forward-only surrogates of the physical environment, leaving untapped a capability no physical environment affords: rewinding to past decision points to explore counterfactual futures. We introduce Rollback, a policy improvement paradigm that turns world models from passive simulators into counterfactual engines: it rolls back to critical decision points, branches into alternative actions, and contrasts their outcomes to attribute success to individual decisions. This concentrates imagination where it matters most and distills counterfactual outcomes into fine-grained, action-level supervision without a separately learned value model. We build a scalable training system that unifies distributed world-model search with asynchronous real-robot execution, forming a self-reinforcing cycle: real-world experience continually sharpens the world model, and a sharper world model in turn drives further policy improvement. We term the world-model side of this cycle test-time physics grounding, a capability that hand-built physics simulators lack: while the world model serves as the training environment, real transitions from the policy it trains correct its dynamics through a lightweight LoRA pipeline. Experiments on three challenging real-world tasks show that Rollback improves success rates over the strongest baseline by an absolute 10.67%, 11.11%, and 11.85% on block extraction, ball tossing, and tabletop curling, respectively, while achieving more stable improvement throughout training. Although initialized from teleoperated demonstrations far below human level, Rollback surpasses the average success rate of human players on both ball tossing and tabletop curling within 8 hours of training. 

---
# Do Coding Agents Still Need a Debugger? Frontier-Model-Based Fault Localization at the Accuracy Ceiling 

**Authors**: Christoph Daxerer, Johann Glock, Christian Macho, Rudolf Ramler, Martin Pinzger, Josef Pichler  

**Link**: [PDF](https://arxiv.org/pdf/2610.36015)  

**Abstract**: Background: To localize a fault, a coding agent can reason from source code alone or use debugger assistance by running the program and stepping through it in an interactive debugger. While earlier agents struggled with static execution tracing and debugger operation, frontier LLMs have improved at both. Objective: We evaluate how accurately coding agents localize faults without tools across original and transformed programs, how much debugger assistance improves accuracy and at what cost, and how agents use it when optional. Methods: We seed 610 single-line faults into 144 LeetCodeDataset solutions and rewrite each faulty program with five recipes of behavior-preserving transformations. Agents based on Sol 6.1, Luna 6, and Qwen 3.8 localize each fault with no, optional, or encouraged debugger assistance. Results: Without tools, the agents localized 98.2 to 99.9% of faults in original and 96.7 to 99.9% in transformed programs, losing accuracy mainly where recipes restructured loops and conditions. Encouraged assistance removed 63% and 61% of Luna 6's and Qwen 3.8's remaining transformed errors (+2.1 and +1.3 percentage points) at 14.7 to 27.0 times the token use, while Sol 6.1 stayed at 99.9%. With optional assistance, Luna 6 and Qwen 3.8 still gained +1.5 and +1.3 points, though the agents used the interactive debugger in only 0.2 to 6.2% of runs. Conclusions: On short, precisely specified programs, current coding agents localize nearly all faults from source code alone, and debugger assistance removes most errors that remain. Given the choice, agents executed the program far more often than they stepped through it, yet kept most of the benefit. 

---
# CIVER: Causal Intervention Verification for Reliable Code Review 

**Authors**: Xinyuan Song, Hasibul Haque, Jay Somasundaram, Ryan Turner, Bowen Zhu, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.36010)  

**Abstract**: Code review is essential for maintaining software repositories by identifying defects, improving code quality, and supporting reliable software evolution. With the growing adoption of LLM-based coding agents, an increasing number of code contributions and repository changes are generated by these agents. Consequently, LLM-based code review agents are becoming increasingly prevalent to evaluate these contributions and maintain software quality. However, LLM reviewers often generate lengthy reviews containing numerous potential issues, many of which are false positives or lack sufficient evidence. These unreliable reviews can mislead developers and downstream coding agents, motivating the need to distinguish actionable findings from unsupported claims. We propose CIVER, a causal intervention-based framework that identifies the core issues underlying code-review findings and distinguishes genuine defects from false positives. CIVER integrates three modules: a review module assesses findings, an intervention module generates repair-oriented perturbations, and a verifier module scores review reliability. Through joint optimization via expert iteration, the three modules identify genuine defects by responding to targeted repairs while remaining stable under unrelated changes. Extensive experiments on SWE-Review-Bench demonstrate that CIVER correctly identifies 77% of genuine defects through targeted repairs. Cross-project and downstream repair evaluations further demonstrate that CIVER generalizes across repositories and improves coding agents' task success rates, highlighting its effectiveness for reliable automated code review. 

---
# DuplexTune: Coupled-State Replay for Memory-Efficient Mixture-of-Experts Adaptation 

**Authors**: Paul Raj, Simon Caldwell, Daniel Whitmore, Mutlu Mete  

**Link**: [PDF](https://arxiv.org/pdf/2610.36009)  

**Abstract**: Sparse expert execution does not remove the stored states needed to adapt a mixture-of-experts language model. We present DuplexTune, a framework for replaying coupled hidden states through pretrained attention and expert modules. Trainable width bridges connect two residual streams to the backbone. Bridge warm-up precedes backbone adaptation, with router weights kept frozen. We analyze the implicit state-recovery equation and identify the convergence and replay conditions needed for reconstruction. On Qwen1.5-MoE-A2.7B, the single-H800 evaluation reports 38.8 GB peak memory, versus 64.8 GB for supervised fine-tuning with checkpointing, and throughput of 25.1 versus 19.9 samples/s. DuplexTune achieves 67.2% MMLU, 75.6% GSM8K, and 7.69 MT-Bench, with a weaker multilingual score than the checkpointed baseline. The results characterize a configuration-dependent tradeoff among state storage, throughput, and adaptation quality. 

---
# MedJev: Evaluating Zero-Shot System-One Decision Making in Medicine 

**Authors**: Simon A. Lee, Jiahang Sha, Moonseong Jeong, Yunbei Pan, Jeffrey N. Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36006)  

**Abstract**: System One decision models such as Jev answer structured questions in a single forward pass without generating text, offering fast and computationally efficient clinical decision support. We evaluate MedJev, an open System One model that runs locally on a laptop, across ten medical benchmarks spanning medical knowledge, diagnosis, and patient level prediction using MIMIC IV and MIMIC IV ED. We compare MedJev against zero shot LLM baselines, published GPT results, and supervised models. MedJev achieves accuracies of 72.5 percent on MedQA, 75.4 percent on USMLE, and 80.1 percent on NEJM, approaching published GPT 4 performance while maintaining well calibrated confidence estimates (ECE 0.028 to 0.063). On patient level tasks, MedJev achieves an AUROC of 0.926 for In Hospital Mortality, exceeding XGBoost (0.902), but underperforms supervised models on operational outcomes such as ED Triage Admission (0.745 vs. 0.809). Patient level predictions also exhibit substantial miscalibration (In Hospital Mortality ECE 0.286). We introduce a lightweight, two parameter recalibration method that adapts predictions to local patient populations using 50 to 200 labeled examples, reducing ECE to 0.007. Our findings establish open System One models as a promising zero shot baseline for medical AI, combining competitive predictive performance with low computational cost and efficient local calibration. We release our code and evaluation harness to support future research. 

---
# PolicyMediator: Multi-Regional Policy Learning through Mediation Agents with Evolving Skills in OPC Ecosystems 

**Authors**: Jie Li, Bin Guo, Georgina Curto, Zikang Liu, Mengqi Chen, yu li  

**Link**: [PDF](https://arxiv.org/pdf/2610.35997)  

**Abstract**: One-person companies (OPCs) are emerging enterprises operated by individual entrepreneurs through digital platforms and LLM-based AI agents. These highly skilled entrepreneurs are characterized by being digital nomads, with limited reliance on local ecosystems. Governments in China use distinct regional support policies to attract mobile OPCs, turning local support into a cross-regional coordination problem. OPCs compare regions to maximize their own utility, but imperfect policy perceptions may cause them to reject otherwise suitable locations. Existing policy-learning methods optimize resources under fixed perceptions, while consensus-formation methods aggregate views without feeding cognitive changes back into policy learning. This disconnect produces mismatched location choices and inefficient regional policy adjustment. We introduce PolicyMediator, a cognition-constrained leader-multi-follower Stackelberg framework with bilateral mediation. The mediator explains verified policies to OPCs to build awareness and trust, and reports their needs and unresolved concerns to governments for policy learning. Its self-evolving skills retain, revise, or prune strategies using verification, allocation, capacity, and objective trajectories rather than mediation success alone. Across six-region simulations in China with five OPC types, five scenarios, and three OPC-simulator backbones, PolicyMediator ranks first in all 15 OPC-backbone--scenario combinations. Over the eight-baseline mean, it improves the aggregate government objective, combining economic performance, employment benefit, and policy-access equality, by 42.4% and average OPC utility by 40.9%. 

---
# Pricing Cross-Bank Context in Federated Anti-Money Laundering 

**Authors**: Lidia Losavio, Dario Fenoglio, Martin Gjoreski, Marc Langheinrich  

**Link**: [PDF](https://arxiv.org/pdf/2610.35996)  

**Abstract**: Most graph-based Anti-Money Laundering (AML) methods assume access to a pooled transaction graph containing all accounts and payments, which in reality does not exist. Federated learning offers the necessary privacy guarantees but it remains unclear how money laundering patterns spanning separate graphs can effectively be detected. We exploit transactions already recorded by both endpoint banks to approximate remote account states locally, enabling multi-layer Graph Neural Network (GNN) computation without additional account-specific exchange. Starting from this baseline, we test whether the remaining context is worth sharing, using a disclosure ladder that adds two transaction counts per remote account or the counterparty's own GNN states. On AMLworld HI-Small, across six GNN backbones and up to 32 silos, the observable baseline yields the largest gain and keeps it as fragmentation shrinks the visible slice; counts add a robust correction, while exact remote states help only in specific regimes. In controlled privacy--utility studies, the noise needed to protect shared states erases their detection value before it buys meaningful privacy, whereas counts stay useful even under strong privacy protection. These findings support a practical rule for banks: use what they already record, and formally protect the little that must move. 

---
# Layerwise Repacking: Variable-Length Hybrid-Attention Training without an Extra Redistribution Pass 

**Authors**: Jiahao Zhang, Junwei Feng, Dongliang Huang, Xiaoling Zhou, Yiqiao Liao, Chengru Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.35989)  

**Abstract**: Variable-length training creates conflicting parallelism requirements across the layers of a hybrid-attention model. The longest samples require wide sequence-parallel (SP) groups of 16 or 32 ranks for full attention, whose cost grows quadratically with sequence length. Sparse-attention layers, dominated by near-linear MoE work, can instead balance most evaluated workloads within node-local SP8 groups. Reusing the wide groups for these layers pushes their all-to-all communication across nodes, where the observed effective rate is 7.0x lower. Yet switching to smaller groups changes which ranks own each sample; a separate redistribution at every full-sparse boundary would introduce another data transfer.
Layerwise Repacking keeps each SP16/32 parent for full attention and repacks its samples into node-local sparse children at SP{1, 2, 4, 8}, where hidden states persist between full layers. The layout change rides on the two existing full-layer Ulysses sites: at each, one variable-size all-to-all performs both the head exchange and the sample repacking, so no redistribution pass is added. Because the balance the children can reach depends on which samples share a parent, an auxiliary parent solver rebalances parent membership before the children are formed.
On all 22 steps with SP16/32 parents among the first 200 training steps of a 1F3S hybrid-attention model (one full-attention layer per three sparse-attention layers), repacking alone reduces the mean end-to-end time by 5.95% and raises model FLOPs utilization (MFU) by 3.75 percentage points over the baseline with repacking disabled; with the improved solver, end-to-end time falls by 7.61% and MFU rises by 4.83 percentage points. 

---
# PROBE: Frontier Coding Agents Find Different Bugs Than Maintainers Fix 

**Authors**: Marco Paina, Mark Niklas Müller, Niels Mündler-Sasahara, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2610.35987)  

**Abstract**: Proactive bug discovery is an important software-engineering problem. Even though coding agents are already widely deployed in this domain, their capabilities remain understudied. In particular, the challenging task of finding all bugs that human maintainers would repair across an entire repository has not been studied yet. To close this gap, we introduce PROBE, a benchmark consisting of all 2,457 bugs fixed over the course of 18 months across 20 diverse GitHub repositories, together with a method for matching candidate findings to this partial reference set. We evaluate coding agents across frontier models and find they all achieve a recall of below $21\%$ and a precision of less than $9\%$ with respect to the reference set. While stronger models increase recall substantially from $8\%$ to $17\%$, they do not improve precision. Surprisingly, despite the low precision, LLM and human evaluators still judge $85\%$ of findings valid. Crucially, we find that found bug distributions are similar across agents but differ substantially from those of bugs fixed by maintainers. The practical relevance of bug discovery and the low recall of frontier models make this benchmark a valuable resource for both model users and developers. 

---
# Let the Gate Decide: Per-Lookup Precision for Quantized Conditional Memory in LLMs 

**Authors**: Tianqi Chen, Xunchao Li, Linchuan Xie, Guanghua Yu, Jianchen Zhu, Peisong Wang, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.35985)  

**Abstract**: Conditional memory layers, such as DeepSeek's Engram, attach large hashed n-gram tables to a language model, and removing them from DeepSeek-V4.1-Flash costs 36.8 PopQA points. Low-bit quantization loses long-tail facts first, and the standard remedy, giving more bits to frequently accessed rows, backfires: refining lookups by access frequency is worse than refining them at random. We trace this to the gate that scales each retrieved row. A lookup's quantization error is weighted by its gate, which is small for frequent n-grams, large for the rare n-grams of long-tail entities, and computable from a 1-bit code. GateRQ therefore stores each row as a ladder of 1-bit residual codes, keeps level 1 in GPU memory to compute the gate, and reads deeper levels only for the lookups the gate ranks highest. Without retraining, GateRQ cuts GPU table memory by 7.8x on DeepSeek-V4.1-Flash, which raises vLLM throughput by 11%-31% over FP8 tables, while losing about 1 PopQA point at 1.50 bits read per lookup and 0.2 points at 2.10 bits across two models. 

---
# Generation and Verification: Human-Agent Collaboration for Safety-Critical Automotive Software Engineering 

**Authors**: Elizaveta Zinovyeva, Aiham Taleb, Nikita Kozodoi, Daniel Schleicher, Amir Mahdi Namazi, Martin Kraus, Levent Kent, Brian Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2610.35982)  

**Abstract**: Research on coding agents mainly focuses on generation quality. Yet, safety-critical automotive software makes two additional demands. It prioritizes verification, since every change is reviewed, traced to a requirement, and checked against industry standards such as MISRA and AUTOSAR before it can be committed. It is also a narrow domain, written against layered architectures, embedded resource constraints, and organization-specific interface conventions that are scarce in the data general-purpose LLMs are trained on. We present a coding assistant for automotive C/C++ that addresses both demands. First, it exposes three tools to the user over the Model Context Protocol (generate, edit, and verify), allowing developers to run verify on any code on demand rather than relying on the agent to check its own output. Second, rather than using a general-purpose model, we customize the underlying LLM for the automotive domain through prompt engineering and fine-tuning. We construct a dataset from automotive production code of AUMOVIO, where each function is paired with the requirement texts and Unified Modeling Language (UML) behavior it must satisfy. In a two-week pilot on active production projects, developers report verify as the most-used tool and confirm using it on their own code and not only on generated output. Comparing model customization strategies, 75% prefer the faster prompt-engineered configuration. MISRA- and AUTOSAR-inspired pattern checkers indicate that prompt engineering achieves a lower number of violations on C, while fine-tuning shows better conformance with architecture patterns for C++. 

---
# UnityBench-Mini3D: Evaluating LLM Agents for 3D Game Development in Unity 

**Authors**: Yijun Zhou, Minghao Shao, Ke Zhang, Renkai Yuan, Hongbo Yuan, Bin Hu, Bozhao Qian, Hongxia Shen, Yanhong Liu, Yuting Chen, Yuntao Shi, Zhongxuan Xu, Shuo Mao, Haijia Zhou, Zhendong Li, Tao Zhang, Chenggang Li, Yuxuan Wang, Zinan Zheng, Calvin Zhang, Jia Li, Heng Chang, Jie Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.35980)  

**Abstract**: Game development stresses coding agents in ways general software engineering does not: correctness depends on how engine APIs, serialized scene state, and runtime behavior interact, not on source code alone. Existing benchmarks mainly target platforms such as Godot and heavily rely on LLM judges or coarse task-level outcomes. We present UnityBench-Mini3D, a benchmark of 77 project-level Unity tasks organized around seven expert-defined skill categories rather than tutorials or genres, with three difficulty tiers. Each task pairs a complete Unity project with natural-language requirements and is evaluated using 1,178 held-out, human-authored milestone tests covering scene structure, component configuration, and runtime behavior. This in-engine evaluation provides deterministic partial credit, localizes failures, and distinguishes partial implementations from complete solutions without relying on subjective judges. Across 21 models and 9 coding harnesses, the average pass rate reaches 80.9% on easy tasks but falls to 35.2% on hard tasks, with just 57.7% overall. UnityBench-Mini3D provides a reproducible, diagnostic testbed distinguishing systems across difficulty levels. 

---
# Multiple Leak Localization in Water Distribution Networks through Graph Neural Networks 

**Authors**: Arvin Ajoodani, Sara Nazif, Pouria Ramazi  

**Link**: [PDF](https://arxiv.org/pdf/2610.35972)  

**Abstract**: Leak localization in water distribution networks is challenging when multiple leaks occur simultaneously, particularly under sparse monitoring. Existing data-driven methods often assume a single or predefined number of leaks or require extensive enumeration of leak combinations. This study proposes a graph neural network (GNN) model that localizes a variable number of simultaneous leaks without specifying the leak count in advance or exhaustively generating every possible combination of leak locations for training. The model is evaluated on two benchmark networks of different sizes, the 31-node Hanoi network and the larger 92-node NET3 network, under variable demand, sparse pressure-sensor coverage, variable leak magnitudes, and varying numbers of simultaneous leaks. The GNN outperforms multilayer perceptron, support vector machine, K-nearest neighbors, and random forest classifiers by up to 12% and 20% in accuracy respectively in Hanoi and NET3 networks. Consistent performance across both networks supports the scalability of the proposed approach. 

---
# EMMA: Structured Episodic Memory for Multi-Agent Workflow Synthesis 

**Authors**: Ruijia Niu, Kaiwen Liu, Rose Yu, Kristina Klinkner  

**Link**: [PDF](https://arxiv.org/pdf/2610.35966)  

**Abstract**: Deploying LLM agents in specialized domains often require fine-grained task-specific procedures and knowledge a general model lacks, yet parameter training is costly and non-interpretable, and hand-engineered agent systems shift the cost to expert labor. Episodic memory updates at context level, but single agent adaptation struggles to alleviate cross-task interference, especially with complex tasks such as multi-application navigation or data pool analysis. Although multi-agent evolution alleviates this problem, existing methods lack clear definitions on update rules and rely on the backbone model to decide where and how much the system changes. This ill-scoped evolution makes learning follow the backbone's structural preferences rather than the observed failures: one backbone accumulates skills while another spawns agents. We present EMMA (structured Episodic Memory for Multi-Agent workflow synthesis), which treats the configuration of a multi-agent system (controller routing memory, specialists inheriting a generalist prompt, and per-step procedural skills) as episodic memory. EMMA makes the ownership and admissibility of updates explicit: a reflector assigns each diagnosed failure to the scope owning its correction and proposes one edit within it, and backbone-independent guards keep every accepted edit scoped, incremental and bounded. Across KramaBench, AppWorld and BBEH with five backbones and four baselines (ACE, GEPA, EvoMAS and ADAS), EMMA attains the best overall performance on every benchmark, up to 32.5% over the strongest baseline, maintains consistent improvements across backbones, and uses 17-38% fewer training executions than the most efficient baseline. 

---
# A Hyper-Logical View on Multi-Agent Path Planning 

**Authors**: Hadar Frenkel, Dor Atzmon  

**Link**: [PDF](https://arxiv.org/pdf/2610.35964)  

**Abstract**: Multi-Agent Path Planning (MAPP) requires collision-free paths for multiple agents. MAPP has numerous real-life applications, including the navigation of automated vehicles, and has therefore many variants considering different constraints, requirements, and objectives. For each such variant, a new algorithmic treatment is often developed. This paper takes a complementary, specification-centered view: we show that many MAPP variants can be modeled in HyperLTL, a temporal logic that reasons about multiple paths and their evolution through time, and in its omega-regular extension HyperQPTL. The contribution is not a claim that current hyperproperty model checkers replace specialized MAPP solvers. Rather, the hyper-logical view gives a unified language for comparing MAPP variants, adding cross-agent constraints, expressing optimality, and connecting MAPP to verification, explanation, and causality techniques for hyperproperties. 

---
# AgentBaton: Efficient Model Collaboration in Agentic Tasks 

**Authors**: Junlong Li, Jindong Jiang, Yonggan Fu, Jiarui Yao, Hanrong Ye, Wonmin Byeon, Peter Belcak, Hao Zhang, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2610.35959)  

**Abstract**: Solving long-horizon agentic tasks requires language agents to interact with their environments over multiple steps, and reliable performance often comes from the expensive capable models available through cloud providers. Meanwhile, alternative efficient models with inferior performance but a much lower cost are emerging, including lower-tier cloud models and smaller open-weights models deployable on local hardware. This raises a natural question that remains less explored than traditional query-level routing: can capable and efficient models collaborate within a single long-horizon agentic task to substantially reduce cost while preserving task accuracy? In this paper, we systematically investigate key design choices for such collaboration through model handoffs, where one model takes over from another, focusing on handoff direction and context transfer. We find that efficient-to-capable handoffs generally underperform capable-only execution with limited or even no cost reduction, whereas capable-to-efficient handoffs at suitable stages of task progress largely preserve or even improve accuracy while substantially reducing cost. These findings suggest that a one-time, simple, well-timed handoff strategy is good enough for cost-effective agent collaboration, motivating AgentBaton, a lightweight detector that learns when to hand off from recurring patterns in task progress. Across 11 model pairs and two representative benchmarks, AgentBaton reduces API costs by 65.9% on average while largely preserving task accuracy, with accuracy gains of up to 8.8 percentage points over capable-only execution. Experiments across various reasoning-effort settings further show that AgentBaton extends the empirical accuracy-cost Pareto frontier. 

---
# Structured Time-Frequency Representation for Channel Prediction 

**Authors**: Ning Yang, Yuming Xian, Tong Li, Ning Gao, Haijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.35956)  

**Abstract**: Accurate CSI forecasting is essential for mitigating channel aging and supporting reliable downlink transmission in high-mobility wireless systems. In FDD systems, forecasting future DL CSI from historical UL observations requires joint temporal extrapolation and cross-band inference. However, defining historical UL as the input and future DL as the target does not specify how elapsed time and UL-to-DL band separation should be represented in a pretrained predictor's positional space. FAR-CSI establishes an FDD-aware representation of channel forecasting within a joint temporal and operating-band positional field. Within this field, historical UL observations and future DL targets occupy segregated regions according to their temporal and operating-band roles, preserving the distinction between observed UL context and unobserved DL prediction targets. This representation is instantiated through a unified structured routing rule that governs encoder positional assignment, decoder-field reconstruction, and future-DL target selection, ensuring the same FDD geometry is preserved from contextual encoding to prediction readout. Under matched backbone, initialization, and training conditions, FDD-aware routing reduces NMSE by 21.44\% relative to generic routing. Across the evaluated FDD baselines, FAR-CSI achieves higher CSI prediction accuracy and MRT-based spectral efficiency, while retaining its predictive advantage across mobility conditions, prediction horizons, reduced training-data regimes, and noisy historical observations. 

---
# ResCom: Activation Residual Compensation for Large Language Model Quantization 

**Authors**: Ning Yang, Bohan Cui, Zhuowen Liu, Yuting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.35955)  

**Abstract**: Low-bit weight--activation quantization reduces the memory and computation needed to deploy large language models, but activation rounding can still degrade accuracy after calibration. Reconstruction from rounded inputs cannot distinguish activations within the same quantization bin. We propose ResCom, a low-rank correction that uses the activation rounding residual to compensate for the resulting layer-output error. An output-aware calibration procedure selects residual directions by their contribution to output reconstruction and computes the correction in closed form, without gradient training or updates to the calibrated weights. On Llama-3-8B with 4-bit weights and activations, rank-32 attention corrections reduce WikiText-2 perplexity from 7.708 to 7.147, improve MMLU by 2.89 percentage points, and raise five-shot GSM8K accuracy from 23.96 to 36.16 percent. Evaluations on models from 1B to 70B parameters and on held-out C4 test the method beyond this setting. The evaluated FP32 implementation adds 2.25 percent to 4K prefill time; a separate compact-storage implementation retains a 27.7 percent peak-memory reduction relative to FP16. 

---
# Curriculum-Driven Deep Reinforcement Learning for Autonomous Far-Range Rocket Landing 

**Authors**: Atin Chowdhury, Piyush Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2610.35954)  

**Abstract**: Autonomous vertical landing of reusable launch vehicles is a critical yet formidable challenge in aerospace engineering, primarily due to highly nonlinear dynamics, stringent precision require- ments, and inherent environmental uncertainties. While reinforcement learning (RL) has emerged as a promising alternative to traditional model-based control, existing approaches frequently rely on flat training paradigms and highly simplified initial conditions, drastically limiting their real-world applicability. This paper proposes a comprehensive curriculum-driven deep reinforcement learning framework designed to master both far-range navigation and precision landing. By employing a three-stage curriculum augmented with a critic warm-up procedure and a rehearsal mechanism to prevent catastrophic forgetting, alongside a high-dimensional observation vector utilizing a full 3 X 3 rotation-matrix, the agent is trained to navigate from an initial horizontal distance of 1,000 to 1,500 units vastly farther than prior benchmarks before executing a soft touchdown. Crucially, we intro- duce a redesigned reward function that reduces unnecessary computational overhead and addresses limitations of earlier formulations, including poorly calibrated reward components and potentially conflicting optimization objectives. The redesigned function bounds the per-step reward magnitude by (O(1)), while its progress term accumulates the total distance closed over the trajectory. The available 100,000-episode results report a 99.98% success rate for the proposed policy and 68.31% for the baseline under the evaluation conditions. In the evaluation, the proposed policy retained 69.1% of its initial fuel capacity and achieved a mean final distance from the pad centre of 7.07 units, compared with 13.10 units for the baseline. Landing visualizations are available in Video 1 and Video 2. 

---
# Who Convinced Whom? Tracing How LLMs Change Their Minds in Multi-Agent Systems 

**Authors**: Arnesh Batra, Aniket Khandelwal, Jashn Khemani, Krish Thukral, Arush Gumber, Denis Federico Lim, Ujwal Kumar, Phan Xuan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2610.35930)  

**Abstract**: Multi-agent debate (MAD) aims to improve language-model decisions through the exchange of arguments, yet agreement after debate need not reflect independent corroboration. We investigate how peer communication influences individual decisions and collective outcomes through behavioral and internal interventions. We introduce MAD Bench, a 2,400-question benchmark spanning knowledge and reasoning, values and moral judgment, and safety and refusal, and evaluate three open-weight models using a round-matched independent-reflection control. Among answer-changing responses, a forced-choice readout before rationale generation matches the eventual answer more often in debate than in reflection (65.4% versus 22.3% on Qwen3.5-9B). Message ablations reveal model-dependent effects of peer answers and reasoning, while attributing identical text to the receiver rather than another agent substantially changes its influence. For the tested source descriptions, stated track record affects answer preferences more strongly than professional-status cues across all three models. Attention masking produces different changes in decision margins across the three checkpoints. On fully parsed, reference-labelled non-numeric questions, debate yields small estimated accuracy improvements over initial majority voting, while increasing the rate of unanimous wrong answers by 3.1-7.4 percentage points relative to reflection. Longer debates show persistent consensus and greater correlation between agents' correctness than reflection. These findings show that stronger consensus can coexist with more shared errors, motivating evaluations that assess dependence and unanimous errors alongside accuracy. 

---
# SCOPE: Source Credibility and Outlet Profiling Engine 

**Authors**: Ilyas Nagiyev, Miras Baisbay, Daniil Orel, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2610.35929)  

**Abstract**: We present SCOPE, a multi-agent system for profiling news media outlets with respect to bias, factuality, and credibility. Given an outlet URL, SCOPE coordinates specialized agents for article acquisition, content analysis, external evidence search, scoring, and report generation. The system follows an MBFC-style methodology while making intermediate evidence and agent outputs inspectable through a structured demonstration interface. On a benchmark of 200 media outlets, SCOPE outperforms LLM-only, article-only, search-only, and media-background-check baselines on most metrics, achieving the lowest bias and factuality MAE, perfect failed-fact-check detection, and the lowest factual error rate. These results suggest that decomposed, evidence-grounded agent workflows can support more transparent and reliable media profiling. Our code is available at this https URL, and a live demonstration at this https URL. 

---
# ReCoPII: A Relation-Grounded Framework for Building Contextual PII Masking Datasets 

**Authors**: Taehwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.35925)  

**Abstract**: Training contextual masking systems for personally identifiable information (PII) requires examples distinguishing maskable from preservable mentions of one entity type, but privacy restrictions limit access to real data. Author-based augmentation (SPY) labels mentions unrelated to the author as non-PII, presuming a single data subject and overlooking others. We propose ReCoPII (Relation-grounded Contextual PII masking), a framework that builds such data without this premise: it plans ontology-constrained entity relations, writes insertion sentences with placeholders, reconstructs annotations with code, checks the result with a separate LLM, and derives MASK/KEEP labels from entity relations. On Korean documents from six domains, it yields ReCoPII-Ko, a corpus with relation, privacy, and MASK/KEEP annotations, which we validate. Across 300 pairs, four annotators agree with 88.42% of its sampled KEEP labels, against 26.27% for SPY-style augmentation of the same sources. Classifiers trained on it over-mask less than under that reference on a matched test. Contextual masking fine-tuned jointly with a PII detector removes most over-masking of detection alone within a fixed leakage budget; on human-verified entity records, the four detectors meeting it over-mask at most 5.04% with at most 4.18% leakage. The detector is comparable to XLM-R-large on KDPII with about a quarter of its parameters. 

---
# Physiognomy Revisited: How Vision-Language Models Encode Sensitive Attributes Through Image Transformation 

**Authors**: Soumyajit Datta, Arka Dutta, Ashiqur R. KhudaBukhsh  

**Link**: [PDF](https://arxiv.org/pdf/2610.35910)  

**Abstract**: Emergent generative AI technologies, while remarkably beneficial across several tasks and domains, can suffer from unforeseen vulnerabilities and difficult-to-anticipate failure modes. Such failure modes can disparately affect vulnerable communities. This paper investigates two well-known vision language models (VLMs) and inspects how these models encode protected identities while generating images. We show that when VLMs are requested to transform human face images to reflect race identities (e.g., Hispanic, Black, White, or Asian) and sexual orientation (e.g., gay or straight), VLMs often comply with such requests. We demonstrate that compliance (and refusal) is non-deterministic, group-sensitive, and generated images are statistically distinguishable, often accentuating deep-seated stereotypes. We further show that representation biases introduced during image generation can re-emerge in textual reasoning and characterization tasks. Finally, we show that this worrisome finding is not limited to racial identity or sexual orientation, image generation can perpetuate bias around criminality, raising concerning possibilities of AI-assisted profiling. 

---
# Task-Utility IQA: Rethinking Image Quality Assessment in Embodied Tasks 

**Authors**: Jirong Zha, Yanwei Liao, Yemin Wang, Xiangmin Yi, Siqi Peng, Yingfeng Chen, Chen Gao, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.35905)  

**Abstract**: Image quality assessment (IQA) underpins embodied imaging pipelines by judging whether visual quality satisfies downstream tasks, yet most existing methods learn task-agnostic scores aligned with generic human ratings on static benchmarks. This objective mismatches embodied and interactive settings, where image adequacy depends on task goals, context, and action requirements that shape an agent's decisions. We argue that IQA should shift from score regression to goal-conditioned judgment defined by embodied task utility. To this end, we propose an MLLM-based embodied IQA agent framework that reasons about task requirements, grounds quality evidence, and produces decision-facing assessments. A compact VisDrone evidence study shows that conventional perceptual metrics and generic MLLM-based IQA scores are weak proxies for downstream VLM task success, while a task-conditioned utility score better predicts whether visual input supports the target task. These findings support a task-utility-oriented reframing of IQA for embodied, explainable, and decision-aware visual perception. 

---
# Beyond Greenwashing: Algorithmic Resilience and the Non-Linear Governance Premium in Sovereign Capital Flows 

**Authors**: Sourav Bose, Taoufik Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2610.35895)  

**Abstract**: Despite the proliferation of sovereign Environmental, Social, and Governance (ESG) metrics, traditional linear macro-financial models systematically misprice Foreign Direct Investment (FDI) tail risks during systemic shocks. This paper demonstrates that sovereign risk transmission is fundamentally non-linear. Using a forward-chained global panel of 147 sovereign entities (2010-2024), we deploy a gradient boosting architecture that compresses forecast errors by over 39% against L1-regularized baselines precisely during extreme macroeconomic tail events. Moving beyond opaque predictive accuracy, we apply dual interpretability frameworks to mathematically map the discrete threshold effects of institutional quality. We identify a strict, non-linear "Governance Premium": incremental environmental and social improvements yield zero capital advantage until a foundational regulatory threshold is breached. Finally, we establish regime-conditional 95% uncertainty bounds that maintain valid marginal coverage even under severe ESG reporting failures. This framework equips policymakers with a distribution-free tool to quantify sovereign tail-risk, proving that global capital markets price institutional resilience as a discrete boundary rather than a continuous premium. 

---
# scPyviewer: a Python-native interactive viewer from AnnData single-cell data 

**Authors**: Hao Xuan, Yu Huang, Jiang Bian, Xiangtao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.35890)  

**Abstract**: Motivation: Interactive web tools enabling non-programmers to explore single-cell datasets, embeddings, gene expression, metadata, and marker genes are standard laboratory infrastructure. However, actively maintained tools (e.g., ShinyCell, ScRDAVis, sCIRCLE) rely on R Shiny and require Seurat objects. Python/scanpy, the dominant framework for single-cell RNA-seq and spatial analysis, lacks a lightweight, native option pairing a shareable viewer with a scriptable Python API. Sharing scanpy results currently requires converting to Seurat or sharing code notebooks.
Results: We present scPyviewer, a web viewer that ingests AnnData objects directly and reproduces incumbent interaction patterns within the Python stack. In a feature-parity audit against three R Shiny tools, scPyviewer matches or exceeds baseline capabilities (7/7), uniquely offering native AnnData ingestion without conversion and cross-dataset comparison over shared genes and matched cell types. Benchmark tests across three datasets (22,315 to ~313,000 cells) show scPyviewer renders every core view faster (up to 3.6x speedup) using lower memory (5.2x lower on the smallest dataset). On 313,000 cells, scPyviewer completes all views while Seurat exhausts an 8 GB memory limit and fails. Beyond the app, scPyviewer exposes a Python API returning Matplotlib figures and pandas tables for publication-ready output.
Availability and implementation: Implemented in Python 3.11 (scanpy 1.11.5, anndata 0.12.19, streamlit 1.59.2, plotly 6.9.0) with a one-command interface to reproduce benchmarks, figures, and launch the app. Source code: this https URL 

---
# ASTROLABE: A 3D Particle-Level Surrogate for Cosmology Simulation with Physical Flow Matching 

**Authors**: Ibrahim Elsharkawy, David Curtin, Yonatan Kahn, Neal Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2610.27068)  

**Abstract**: The microphysics of dark matter (DM) is imprinted on cosmological structure across an enormous dynamic range. Nonlinear cosmological observables are most reliably modeled with N-body and hydrodynamic simulations. However, systematic N-body scans over DM theory space are prohibitively expensive. Machine learning emulators provide a possible solution, but are currently limited by either low resolution or only the ability to emulate specific observables. We introduce Astrolabe, a flow-matching-based generative emulator designed to be free of both constraints. Astrolabe acts directly on the 6N phase-space coordinates of N individual particles. With no voxel grid in the output, there is no voxel-resolution ceiling, and in principle any observable can be measured from inference. Astrolabe is computationally tractable via a field-particle hybrid transformer in which every particle cross-attends to multiscale field tokens, so compute and memory are linear in N. Astrolabe incorporates physical flow matching by taking the prior distribution for the flow to be approximate COmoving Lagrangian Acceleration (COLA)-based coordinates rather than unphysical noise, with momentum and mass conservation enforced at all points along flow time. In the Lambda-CDM cases we test, Astrolabe produces realistic fields with accurate power spectra, halo mass functions, velocity profiles, and halo density profiles. We also observe out-of-distribution generalization, with accurate emulation of dark-acoustic oscillation initial conditions. Astrolabe is also capable of being fine-tuned to simulate very large N and we demonstrate this by emulating a TNG50-3-dark simulation at a fraction of the cost. This method represents a first step toward a general and powerful cosmology simulation surrogate, which may eventually permit the inclusion of baryonic physics and a systematic exploration of richer dark sector interactions. 

---
