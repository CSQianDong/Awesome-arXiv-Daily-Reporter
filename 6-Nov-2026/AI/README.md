# SwarmEvolve: Latent Agent Swarms for Evolutionary Discovery and Optimization 

**Authors**: Qile Jiang, George Em Karniadakis  

**Link**: [PDF](https://arxiv.org/pdf/2611.08483)  

**Abstract**: LLM-guided evolutionary search uses AI agents to improve programs for mathematical and computational problems by iteratively refining solutions and combining high-scoring methods. However, the process is often slow because the LLM generates a long textual chain of thought for each round and must repeat this generation many times to maintain population diversity. We present SwarmEvolve, a framework in which hundreds of low-cost agents per round work in swarms, reason in latent space without decoding to text, and share information by passing KV caches. Each swarm studies the problem from a different angle. A novelty filter keeps the most distinct agent ideas by directly measuring the distances between the agents' hidden state vectors. Crossover of ideas happens naturally in latent space, as a summarizer agent receives the combined KV caches of all swarm members. On 21 problems in mathematical discovery and scientific computing, SwarmEvolve gets the highest average score and needs a median of only 3 to 8 minutes to reach the best scores that other state-of-the-art frameworks reach in 60 minutes. On several problems, SwarmEvolve with Qwen3 8B improves on the solutions that AlphaEvolve first found with frontier LLMs. Our work shows that agent swarms that communicate in latent space offer a fast, low-cost approach to program search for scientific problems. 

---
# Scientist World: Grow a Research World for AI-Scientists 

**Authors**: Taiming Lu, Sachin Konan, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.08478)  

**Abstract**: AI-Scientists arrived, but a world for them to grow in did not. Human scientists take theirs for granted $-$ centuries of open problems and shared ideas, waiting in the scientific literature. To AI-Scientists, that world is readable but not runnable, so we build Scientist World for them: every research paper becomes an executable environment. Our autonomous pipeline extracts each paper's open-ended research question and reproduces its proposed method as the baseline; an AI-Scientist entering the environment is evaluated against what the human authors achieved. We release the first 1,000 verified environments. With the world as an evaluation, we found that frontier AI-Scientists can already surpass the human researchers 74% of the time. As a record of how AI does science, it contains 6,400 hours of research traces, with every idea, experiment, failure, and success. As a training ground, we are able to raise (+158%) a small model to match frontier models in research ideation. At last, AI-Scientists' own world has arrived, grown from and growing with science itself. 

---
# ReproBridge: Building Reproducibility into ML Research Artifacts 

**Authors**: Yubo Wang, Rui Meng, Jaehyun Nam, Tomas Pfister, Jinsung Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2611.08452)  

**Abstract**: Reproducing machine learning experiments requires reconstructing procedures from information scattered across papers, released code, and supporting artifacts. For coding agents, reproduction success can therefore depend on their ability to fill in missing details, rather than on the released artifacts alone. We ask whether preserving recovered procedures and decisions improves subsequent independent reproduction. We define a standardized artifact design comprising a refined codebase, environment setup, experiment launchers, and a reproduction guide, and introduce ReproBridge, an agent workflow for constructing these refined reproducible artifacts. ReproBridge-Bench provides 1,029 tasks from 387 experiments across 36 papers, anchored by five human-verified reference scores per task. Agents attempt tasks in fresh sessions with raw code or refined artifacts; a separate Auditor checks execution evidence. Across four evaluated models, refined artifacts raise mean reproduction success from 76.4% to 92.6% and reduce downstream API spending by 19.7% on average. The cross-model success-rate spread narrows from 21.6 to 3.9 percentage points, with all four models achieving mean success rates above 90%. Removing any component reduces success in the ablation study, with the largest losses from code and environment removal. These results suggest that preserving execution knowledge helps even models with lower raw-code success reproduce experiments more reliably and at lower API cost. 

---
# The Power of Meta Access in the Software of the Future --- Demonstrated on LLM Chat UIs and Chat AI 

**Authors**: Julian Kunkel, Ali Doosthosseini  

**Link**: [PDF](https://arxiv.org/pdf/2611.08428)  

**Abstract**: Chat remains the default surface for many large language model (LLM) tasks, even as agentic tools such as OpenCode excel at coding and automation. Yet the model sees the current dialogue but not the workspace around it. We explore meta access---giving the assistant a structured, permissioned view of the interface itself and propose a ladder of levels: feature tutoring; metadata, settings, and preference access; retrieval over summaries and messages, confirmed mutation, and full orchestration. Meta access can turn a chat client from many screens into one entry point for finding, navigating, organising, and operating it, while raising risks around grounding, privacy, prompt injection, and trust. General-purpose agentic harnesses are runtimes; because they can change their own settings, meta access becomes reflexive and needs a separate control-plane boundary. We ground this in the Meta Chat prototype of the Chat AI client. It builds bounded context from, e.g., summaries, settings, and relevant messages; links references to conversations, topics, and personas; supports confirmed, reversible changes; offers a browsable tutorial library; and recommends personas as user-started conversations. By exposing application state in structured form, meta access can extend screen readers from widget descriptions to semantic workspace operations. Meta access thus turns chat from a window onto one conversation into a control layer for the whole workspace, extending chat toward agentic capability without surrendering user control. As assistants become embedded in IDEs, dashboards, office suites, and operating systems, AI-era interfaces should expose a structured, permissioned model of their state and capabilities, ideally by opening their existing metamodels through a debugger-like interface. 

---
# Benchmarks as Software: Terminal-Bench and the Case for Continuous Benchmarks 

**Authors**: E. Kelly Buchanan, Crystal Zhou, Scott W. Linderman, Christopher Ré  

**Link**: [PDF](https://arxiv.org/pdf/2611.08417)  

**Abstract**: Coding agents are among the most economically consequential deployments of LLMs to date, making reliable evaluation increasingly important. Yet even benchmarks that are carefully validated at release can become stale. As models improve, they can discover valid solutions that existing verifiers reject, while changes to evaluation infrastructure can invalidate assumptions embedded in tasks and tests. We call the resulting accumulation of defects benchmark maintenance debt: a growing gap between benchmark scores and the capabilities they are intended to measure. We introduce a continuous-validation workflow for agentic benchmarks that combines community reports, task artifacts, agent traces, automated oracle checks, and human-gated repairs. We apply this workflow to Terminal-Bench 2.0, a benchmark that underwent extensive human and LM-assisted review and red-teaming before release. Our audit identifies issues in roughly 30% of the benchmark, and we release the resulting repairs as Terminal-Bench 2.1. These repairs change evaluation results: accuracy increased by over 10 percentage points and leaderboard positions move by up to three ranks, even as the overall ranking remains highly correlated (Spearman rho = 0.958).
We evaluate how resource allocation shapes benchmark conclusions. We show that fixed wall-clock can materially change agent performance: For example, extending the time limit closes a 20pp harness gap on GPT 5.4 and shrinks a 30 pp gap on GPT 5.4 mini to 5 pp. Benchmark maintenance is therefore not only about repairing task artifacts: resource budgets are part of the measurement specification and must be validated alongside the tasks themselves.
More broadly, benchmark maintenance debt is not just accumulated implementation bugs: our results show that as models improve, both evaluation assumptions and resource settings may need to be revalidated. 

---
# The Gold Rush in AI4Math, II: What are signatures of AI-assisted Proofs? 

**Authors**: Yuanchuan Guo, Jiashun Jin, Zheng Tracy Ke, Bingcheng Sui  

**Link**: [PDF](https://arxiv.org/pdf/2611.08412)  

**Abstract**: The growing use of AI in mathematics raises a question: can we detect AI involvement in proofs, separately from AI assistance with writing? Historical arXiv papers, recent papers disclosing substantive AI use, and manuscripts released by OpenAI provide a rich empirical basis for studying this question. However, several challenges remain: lack of matched case-control pairs; potential incompleteness of disclosure; and lack of ready-to-use features designed to capture signals of AI involvement in proofs.
To address these challenges, we construct five cohorts comprising 8,281 papers by combining existing manuscripts with those we produced using generative AI. These cohorts span different levels of AI involvement and include matched pairs of papers addressing the same research problems. We then introduce ProofTrace, a framework that represents each paper as a graph of immediate proof support (GIPS) and extracts interpretable features of mathematical organization.
Our study yields three discoveries. First, principal-component projections of our features reveal a qualitative gradient of AI involvement, where AI-reworded papers remain close to human papers, the two AI-generated cohorts depart in broadly shared directions, and papers disclosing substantive AI use occupy intermediate positions which shifts toward AI-generated cohorts over time. Second, we identify recurring signatures of AI-generated proofs, such as fewer externally cited results, greater reliance on intermediate conclusions, and altered graph connectivity; many of these signatures persist under controls for research field, manuscript length, or graph size. Finally, in a separate detection experiment, ProofTrace distinguishes human-written papers from those disclosing substantive AI use. Its performance remains stable under AI rewording, whereas the performance of state-of-the-art text-based detectors declines sharply. 

---
# StepBack: Step-Level Error-Localized Resampling for Efficient Test-Time Reasoning 

**Authors**: Yang Ouyang, Jung-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.08403)  

**Abstract**: Self-consistency (SC) is a widely used decoding strategy for improving chain-of-thought reasoning, but it requires sampling many complete reasoning trajectories with a preset budget. Its efficient variants reduce the cost mainly by stopping full-trajectory sampling early or distributing the number of samples across questions. In this paper, we propose StepBack, a step-level error-localized resampling method for efficient test-time reasoning. Given a small set of draft trajectories, StepBack localizes likely first error positions using lightweight signals and then resamples the detected error step and its suffix, followed by majority answer voting. Empirically, StepBack matches the accuracy of SC on average across seven mathematical reasoning and multi-hop QA benchmarks, while reducing token usage by up to 41.26%. In addition, by nature, StepBack can be combined with adaptive inference methods such as Adaptive Consistency, further improving their inference-time efficiency. 

---
# Learning a patient world model for health system simulation in intensive care 

**Authors**: Yingtai Li, Ruoyan Wang, Yu Li, Shaohua Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.08392)  

**Abstract**: Designing healthcare strategies requires anticipating their clinical and operational consequences before implementation, a task that often relies on health system simulation. Such a simulation necessitates modeling how patient states evolve and generate care demands. However, conventional approaches typically tailor compact state transitions to a single policy question and omit history-dependent dynamics, thus requiring the redefinition of states and transitions when there is a new question that entails different clinical information. We propose to learn a patient world model from longitudinal EHRs using a neural network to provide a shared clinical representation of patient trajectories that can serve simulation questions across scenarios. Taking ICU operations as an example, we train the model on MIMIC-IV, freeze it, and couple its time-queryable predictions to shared ICU calendars, resources, and queues via scenario-specific mappings, so that one model renders admission demand, care tasks, and discharge proposals across simulations evaluating ICU bed capacity, care dispatch, and post-ICU transfer capacity. Experiments show that our model matches Markov and semi-Markov comparators in aggregate occupancy and directional responses to capacity and dispatch policies. It also reproduces policy gains from replayed observed trajectories. For patients entering the same support state from different prior trajectories, the learned representation captures a 24-h recovery difference (-6.4 predicted versus -6.7 observed percentage points), a contrast missed by compact state models, which propagates into distinct simulated care demand, queue burden, and bed use. These results indicate that a learned world model of patient dynamics has the potential to replace question-specific state models, allowing new care policies to be evaluated by reconfiguring the scenario without model rebuilding. 

---
# NPCBench: A Clinical Apprenticeship Benchmark for Multimodal Care-Pathway Reasoning in Nasopharyngeal Carcinoma 

**Authors**: Pengkai Wang, Wei-Wei Zhang, Yan Li, Min Tang, Zhitian Hou, Zeyu Liu, Guanghao Zhu, Yuanyi Wang, Yanggan Gu, Wenjun Wang, Minheng Ni, Congkai Xie, Zhijie Sang, Jianmin Wu, Ying Sun, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.08369)  

**Abstract**: Current medical benchmarks typically break down clinical competence into isolated factual questions, single-image tasks, or static case vignettes. These tests score individual answers, but reveal less about whether a model can integrate subspecialty knowledge with multimodal evidence and track a patient's condition throughout a course of care. To close this gap, we introduce NPCBench, a clinical apprenticeship benchmark for nasopharyngeal carcinoma (NPC). NPCBench mirrors specialist training, from guideline and foundational knowledge to case planning, longitudinal review, CarePath integration, and open-ended consultation. It contains 4,777 component and consultation items and a 100-patient CarePath cohort with 417 scored diagnostic and care tasks. The episode-level evaluation tests whether diagnostic and care answers are jointly correct within a care pathway. By linking stepwise tasks to patient-level care trajectories, NPCBench evaluates whether models ground decisions in visual evidence, interpret changes in patient state, and meet clinical requirements in consultation. Across 30 evaluated models, we observe a composition gap: strong performance on individual tasks does not ensure jointly correct diagnostic and care answers for the same patient. Among 21 multimodal models, strict CarePath success reaches at most 23%. Paired controls and physician calibration further examine evidence grounding, longitudinal revision, and consultation judging. Overall, NPCBench broadens the evaluation of medical artificial intelligence beyond static question-answering paradigms to an apprenticeship-oriented assessment of clinical workflows, thereby offering a testbed for evidence-based and temporally coherent reasoning in oncology care. 

---
# Enhancing Local and Global Structure in Compressed AI Agent Memory 

**Authors**: Henry Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.08341)  

**Abstract**: AI agents store text as embedding vectors and compress them at scale. A compression is accepted when each item's nearest neighbors survive, but agents also consolidate, delete, and compare memories, tasks that depend on which items lie far apart. We evaluate both ends with two label-free fidelities from our prior work: the local fidelity $L$ at 10 neighbors and the global fidelity $G$ at 30 to 50 percent of the memory. At 96 bytes per passage, rescored binary codes reach an $L$ of 0.97 yet move 10.1 percent of passages to another theme; PCA with 8-bit codes, at an $L$ of 0.77, moves 1.7 percent. Rescoring leaves the stored codes, and thus $G$, unchanged. We prove that across the $G$ band, $1-\nu_k$ is at most 2 to 2.18 times the square root of the near-far rank-violation rate, and that this rate follows the distance noise of the compression (correlation 0.979). Our locality-aware quantization (LAQ) turns the bound into a hinge objective on the codebooks of an OPQ index, keeping its bytes, rotation, and search. To our knowledge, LAQ is the first quantizer whose objective comes from a proven bound on the far end. Reconstruction alone raises $G$ in 0 of 52 held-out runs; the far hinge raises it in all 52. LAQ raises $G$ in all 52 ($p=8.5\times 10^{-4}$ over 13 memory-encoder groups), including the external FinDER memory, has the highest $G$ of seven equal-size quantizers in all 12 Multi-Doc-2025 runs, scales to a million passages, and keeps evidence recall. 

---
# Learning the Task or Learning the Verifier? Rethinking Criteria Exposure in Embodied Self-Improvement 

**Authors**: Wenbo Huang, Jianxin Bi, Shunyu Yang, Ruiyang Si, Rui Ni, Qiang Wang, Shulong Jiang, Yintianrun Zhang, Feixiang Ruan, Kewei Zhang, Ye Huang, Shukai Gong, Daquan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.08318)  

**Abstract**: Self-improving embodied agents are increasingly being developed and compared on robot simulation benchmarks. Several recent works expose the benchmark's success criteria to the learning agent, a practice we call Criteria Exposure (CE). These criteria, originally written to verify policies trained by imitation learning, often demand more than the corresponding language instructions, so a score from such a verifier may not faithfully reflect whether the instructed task was completed, especially for robot-control programs written by an agent. As embodied coding agents report ever higher benchmark scores, a natural question arises: when a coding agent is exposed to the benchmark's criteria, does it generate code that better performs the instructed task, or code that simply overfits to the verification criteria? We conduct controlled experiments with a self-improving agent with and without CE, and find that with CE the agent quickly produces code with a higher success rate under the given criteria, but this advantage disappears when the criteria are changed slightly without departing from the original instructions. More importantly, the skills the agent develops under CE transfer poorly across tasks. We further propose instruction-faithful criteria and show that they better reflect the performance of agent-generated programs. 

---
# Benchmarking Large Language Models on Heterogeneous Cognitive and Biomedical Knowledge Graphs: The Role of Graph Grounding in Multi-Hop Reasoning and Hallucination Mitigation 

**Authors**: Ali Sarabadani, Kheirolah Rahsepar Fard, Tahereh Jeirani  

**Link**: [PDF](https://arxiv.org/pdf/2611.08315)  

**Abstract**: Large language models (LLMs) are increasingly used in biomedical and cognitive neuroscience, but their ability to reason reliably over heterogeneous knowledge graphs (KGs) remains unclear. We benchmark five LLMs (GPT-5.1, Gemini 2.0 Pro, Llama 3, Qwen, and BioGPT) across five KGs: DrugBank, NeuroSynth, Cognitive Atlas, Brain Map, and GDPKG. We introduce a multitier query dataset to evaluate reasoning depth and factual accuracy under KG-aware (graph-grounded) and KG-agnostic conditions. Inter-annotator agreement was substantial, with Cohen's kappa of 0.781 (range: 0.753-0.807) and Fleiss' kappa of 0.77. Results show that general-purpose models, particularly GPT-5.1 and Gemini 2.0 Pro, generally outperform domain-specific models on multi-hop and high-level reasoning tasks, although performance declines as query complexity increases. KG grounding reduces hallucination rates across all evaluated models. Contrary to our initial hypothesis, domain-specific models do not consistently outperform general-purpose models on specialized tasks without graph grounding. These findings highlight the value of combining broad language-model capabilities with structured domain knowledge. We provide practical guidance for selecting LLM-KG integration strategies based on task complexity and computational constraints, and discuss hybrid neurosymbolic architectures for personalized medicine, cognitive disorder diagnosis, and explainable AI. 

---
# RM-VLM: From Natural-Language Task Descriptions to Vision-Grounded Reward Machines 

**Authors**: Roko Parac, Christos Ziakas, Leo Ardon, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2611.08308)  

**Abstract**: Reward machines (RMs) have been proposed as a reward formulation in reinforcement learning (RL) to capture non-Markovian tasks where credit assignment depends on history. By representing reward functions as automata, RMs expose the task's temporal structure and allow such rewards to be handled within a Markovian framework. However, they assume a labelling function, a proposition-grounding oracle for observations, which must be manually engineered for each environment, limiting their wider applicability. In this work, we address this limitation by proposing RM-VLM, a novel approach in which a large language model (LLM) generates an RM from a natural-language task description, and a vision-language model (VLM) acts as the labelling function, grounding the RM's propositions over image-based observations during RM-guided policy learning. A VLM-based value function estimator is used to assess the agent's progress towards each RM transition and derive a dense reward signal. We evaluate RM-VLM on three domains covering non-Markovian grid-world navigation, robotic manipulation, and 3D object collection. Our experimental results show that RM-VLM consistently achieves higher task success rates than existing VLM-based reward methods across all tested domains. 

---
# First-Token Attraction in Mamba Dynamics 

**Authors**: Trinh T. Nguyen, Minh-Khoi Nguyen-Nhat, Duy-Tung Pham, Hoang-Son Do, Tan Minh Nguyen, Thieu N. Vo  

**Link**: [PDF](https://arxiv.org/pdf/2611.08303)  

**Abstract**: We study token dynamics in deep Mamba models, with a particular focus on the attracting role of the first token. Unlike attention mechanisms, where token interactions are governed by positive scalar coefficients, Mamba dynamics involve full matrix-valued interaction coefficients with no positivity or definiteness guarantees. This structural difference makes Mamba dynamics substantially harder to analyze and prevents the direct application of existing techniques for attention-based models. To overcome this challenge, we introduce a new analytical framework based on a high-dimensional invariant box and a sharp exit-time argument, which together recover an effective form of sign-definiteness. Using this framework, we characterize the possible limit points of token trajectories and prove explicit exponential convergence rates. Our analysis reveals that the first token has privileged dynamics: its limiting direction attracts subsequent tokens, inducing token clustering and attention-sink phenomena in deep Mamba models. We further show that this attraction mechanism extends to broader sequence dynamics, including state-space models, causal attention, and classical time-series models. Experiments on Falcon-Mamba-7B validate our predictions, showing that concentration on the first token increases with depth and dominates in later layers. 

---
# A Generative Model of Contextual Integrity: Appropriate vs. Inappropriate Information Sharing 

**Authors**: Omer Ebead, Claude Formanek, Joel Z. Leibo  

**Link**: [PDF](https://arxiv.org/pdf/2611.08284)  

**Abstract**: Contextual integrity (CI) holds that appropriate information flow depends on context-specific norms. The same disclosure may be appropriate in one context and inappropriate in another. We build a generative model of contextual integrity to study information sharing through simulated conversations. Two actors played by large language models exchange messages, with one holding private information and the other seeking it. We examine how sharing and refusal vary with the situation, the holder's decision logic, disposition, and the underlying language model. Actors shared information with the rightful recipients when it was needed, but also disclosed it to people who were not entitled to receive it. The model provides a way to study how disclosure and refusal develop through conversation as the situations and the actors vary. 

---
# LeanContext: Disentangling Mathematical Resources from Local Formal Context in Lean Whole-Proof Generation 

**Authors**: Hang Zhang, Qihao Ye, Litong Liu, Yunbei Zhang, Yugyung Lee, Yue Zhou, Bowen Liu, Emily First, Cheng Han, Qiao Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2611.08277)  

**Abstract**: Neural theorem proving increasingly uses retrieved premises and project context to support proof generation. Evaluating these resources requires measuring the advantage of target-selected declarations over comparable alternatives and how this advantage depends on training. We introduce LeanContext, a benchmark of 129,826 Lean whole-proof generation problems, to investigate these questions. We group selected complete preceding mathematical declarations as Mathematical Context (MC) and the remaining selected local source commands as Environment Context (EC). Four aligned input views support crossed training and inference evaluations with shared targets and original Lean verification environments. For both diffusion- and autoregressive-based models, selected declarations outperform comparable alternatives by a larger margin when training includes MC. Final-input audits identify omitted declarations referenced by the target. Prioritizing these dependencies improves Pass@32 by 1.40% and 1.70% for LLaDA and Llama, respectively, on 2,000 targets. The selection experiment operates without access to target reference proofs and holds checkpoints, EC, and inference budgets constant. The code and dataset are available at this https URL. 

---
# The Geometry of Simulated Perspectives: Dissociating Representation and Behaviour in a Multimodal Language Model 

**Authors**: Mathis Immertreu, Fitim Abdullahu, Thomas Kinfe, Claus Metzner, Helmut Grabner, Patrick Krauss, Achim Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2611.08262)  

**Abstract**: Multimodal language models can simulate different observers, yet it remains unclear how assigned perspectives reshape their internal representations and whether these changes govern their judgements. Here, we investigate the representational geometry of persona-conditioned visual interestingness in Qwen3-VL-8B-Instruct. Across 100 images and 53 prompting conditions, we systematically vary persona content and the linguistic framing used to introduce it, while recording behavioural ratings and layer-wise hidden states. We observe a systematic reorganisation of representational geometry across processing depth: early layers primarily encode image identity, whereas persona-specific structures emerge in later layers. Strikingly, prompt framing reshapes internal representations more strongly than persona identity, whereas persona content has a greater impact on behavioural judgements. This dissociation is further illustrated by minimal identity statements that alter representations even without changing responses. These findings demonstrate that representational distance does not necessarily reflect behavioural relevance. More broadly, they reveal how simulated perspectives emerge as distinct structures within the representational space of artificial cognitive systems, highlighting a fundamental distinction between the geometry of internal representations and their functional consequences. 

---
# Value-oriented policy distillation for interpretable wind farm power control 

**Authors**: Max Bruninx, Timothy Verstraeten, Jan Decuyper, Ann Nowé, Jan Helsen  

**Link**: [PDF](https://arxiv.org/pdf/2611.08245)  

**Abstract**: Wind farm power control can help the system maintain balance and avoid blackouts. In Europe, this can be achieved through implicit balancing, where a wind farm adjusts its power setpoint in anticipation of the imbalance price using information published by the Transmission System Operator. Thereby, the wind farm helps restore system balance while also reducing its own imbalance costs resulting from forecast errors. However, implicit balancing is a challenging problem since the settlement price is unknown in advance and can be impacted by the pursued control strategy. Reinforcement learning methods could help to solve this problem, but lack the interpretability needed to be deployed on real wind farms. Therefore, we propose a policy distillation framework using explainable boosting machines (EBMs), a variant of generalized additive models, to obtain a transparent control policy. To increase the value captured by the EBM model, we introduce a value-oriented perspective to policy distillation inspired by research in the field of energy forecasting. Furthermore, we suggest to include expert knowledge during model design, such that the control strategy is also justifiable to stakeholders. Using real data of the Belgian grid and an offshore wind farm, we find that our value-oriented and expert-based controller can reduce imbalance costs by 82%, while outperforming a price threshold controller as well as a policy distillation baseline using decision trees. 

---
# Signals, Evidence, and Context: User Scrutiny in Real-World LLM Conversations 

**Authors**: Ahana Biswas, Yu-Ru Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.08232)  

**Abstract**: Large language models (LLMs) often present synthesized answers without making their supporting evidence easy to inspect. We study how features of these answers relate to users' subsequent accepting, checking, and challenging in 489,283 assistant-user response pairs from ShareChat and WildChat. Broad attribution language appears in 31.9% of qualifying responses, compared with explicit URLs in 1.33%, revealing a gap between claims of grounding and preserved inspectable support. Attribution, uncertainty, and other response cues generally accompany both checking and challenging; only uncertainty shows a small statistically distinguishable pooled association favoring checking. Within Technical Help, uncertainty favors checking relative to challenging; refusals and reasoning traces favor challenging in Seeking Information. Scrutiny is concentrated in some conversations, while its apparent turn-to-turn persistence becomes much smaller when conversation-level differences and response ordering are examined separately. These results suggest that supporting verification requires attention to what evidence can be inspected, which question a user is trying to resolve, and the history of the interaction. 

---
# SMART: State-based Multi Agent Reasoning Traces for Verification-oriented Problem-Solving 

**Authors**: Sai Vallurupalli, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2611.08196)  

**Abstract**: Complex problems with limited context, such as claims, require a means to obtain additional knowledge necessary for solution reasoning. We propose a multi-agent framework, SMART, to obtain knowledge through capturing the implicit state of participating entities in a problem context. SMART curates the state knowledge as diverse, complementary and solution-salient premises using a variety of LLMs and a structured role-based prompt strategy. Using knowledge from these real-world verified curated premises, SMART's knowledge guided solution reasoning improves validation performance up to 20% on complex scientific claims spanning 3 datasets. We show that grounding premises in real-world knowledge is necessary and improves performance up to 5%. 

---
# BabyTheorist: A Benchmark for Learning to Theorize the World from Observation Alone 

**Authors**: Doojin Baek, Junyeob Baek, Mingyu Jo, Hosung Lee, Taegu Kang, Gyubin Lee, Sungjin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2611.08177)  

**Abstract**: How would we know whether a machine truly understands the world? We argue that one key capability is \textit{observational theory learning}: acquiring reusable theories from pure observations and applying them to new situations. Current visual reasoning benchmarks focus primarily on the latter, presenting each task as a few-shot puzzle grouped by a common rule. We introduce BabyTheorist, a benchmark for controlled training and evaluation of observational theory learning. BabyTheorist generates continual streams of before-after visual observations from hidden compositional programs; during training, learners see only the resulting transitions, with no program labels, task identities, or support sets. To support theory learning and evaluation, BabyTheorist controls the program language, a curriculum over program complexity, recurring frequent subprograms, and a continual train-test protocol with level skipping. Seven representative baselines plateau well below ceiling, with the gap widening as the curriculum advances---evidence that BabyTheorist isolates a capability current program-learning approaches do not yet address. 

---
# PCBRouteAgent: How Close Are LLM Agents to Becoming PCB Routing Engineers? 

**Authors**: Yue Zhang, Jilei Sun, Liqiang Jing, Yijie Wei, Ruohan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.08175)  

**Abstract**: Automating printed circuit board (PCB) routing requires coordinating board-level connectivity decisions with geometric execution under strict design rules. Although recent agents can interact with native routing tools, reliable routing on complex boards remains challenging. We present PCBRouteAgent, a multimodal PCB routing agent inspired by how human engineers coordinate global strategy with local execution. A Global Routing Planner combines board images and structured electrical and geometric information to specify connection topology, routing order, corridors, and layer transitions. A Detailed Routing Executor realizes these guides through KiCad's native routing engine and performs bounded local recovery. Unresolved failures are returned as structured feedback, enabling targeted guide revisions while preserving accepted routes. We evaluate PCBRouteAgent on 247 synthetic and real-world boards from four PCBWorld-Bench test splits. It achieves clean-pass rates of 100%, 94.9%, 100%, and 40% on D2, D3-A, D3-B, and D3-C, respectively. On the three real-world subsets, PCBRouteAgent outperforms the best evaluated baseline in clean-pass rate by 13.9, 36, and 10 percentage points, respectively. 

---
# A Generative Foundation Model for Multimodal Tokamak Tasks 

**Authors**: Yingshuai Ji, Haodong Feng, Guohong Deng, Gelin Bian, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.08153)  

**Abstract**: Tokamak experiments record diverse measurements, including waveforms, profiles, fields, and scalars, which are used to reconstruct the plasma, forecast its evolution, recover associated control signals, and estimate disruption risk. Most models are built separately for one task and a fixed set of inputs and outputs, and each relies on a prescribed subset of diagnostics rather than structuring the full multimodal observation set to extract global, cross-signal information about the discharge. These limitations motivate a reusable foundation model that adapts to each task's input-output schema and models heterogeneous target signals jointly, rather than retraining a separate model for every configuration. In this work, we extend the TokaMark benchmark with two groups of tasks for actuator action inversion and disruption prediction, yielding 20 MAST tasks in six groups with explicit schemas and evaluation protocols. Then, we introduce GenToka, a generative tokamak foundation model that jointly generates schema-selected targets through a shared conditional diffusion Transformer, with signal-specific codecs preserving native time series, profiles, fields, and scalars; foundation pretraining and task-specific adaptation transfer cross-signal structure across all six groups. Evaluation on MAST measurements shows that GenToka reduces macro-average normalized root mean squared error relative to TokaMind by 4.7% on Groups 1-4 and by 17.7% on Group 5, and improves macro-average window-level F1 on Group 6 by 8.4% at a common validation-selected threshold. GenToka provides a reusable, generative, and multimodal pathway for experimental tokamaks, linking plasma-state reconstruction, actuator inference, and disruption warning from shared measurements to support more integrated, data-driven fusion plasma prediction and control research. 

---
# Create Your Own Work Stack: Deploying Self-hosted AI and Tooling in a Tech Startup 

**Authors**: Saad Ahmed Bazaz, Talha Amjad, Abdurrehman Subhani, Sohaib Ahmed Bazaz, Abdullah Aamir  

**Link**: [PDF](https://arxiv.org/pdf/2611.08116)  

**Abstract**: Self-hosting AI is often presented as a way to reduce inference costs and increase control, but deployment introduces its own costs. We report our experience deploying a hybrid AI stack at Grayhat Developers, a small software company with an AI-intensive development workflow. Our initial motivation was to counter the growing cost of commercial AI usage: Claude/Codex incurred approximately \$325 per month, with more than 25.93 billion recorded tokens. We experimented with local open-source models as replacements for commercial coding agents, but after learning that their performance and inference speed were insufficient, we adopted a hybrid approach in which commercial agents remained responsible for complex tasks while local models handled lower-complexity tasks, organizational information retrieval, and synchronization with self-hosted tools including Outline, Plane, and ERPNext.
We estimated that approximately 30--40\% of our commercial AI workload was associated with context transfer, documentation synchronization, task management, and related operational coordination. By moving this work to local models, we estimate that approximately 90\% of this coordination workload could be handled without commercial inference, corresponding to an estimated 27--36\% reduction in overall commercial-AI workload or cost-equivalent usage. Our first internal deployment nevertheless failed to achieve adoption because it introduced UX friction. Replacing it with a centrally configured deployment substantially improved feedback and adoption. We conclude that local inference is most useful not as a wholesale replacement for commercial AI, but as part of a hybrid architecture that assigns tasks according to their capability requirements, cost, context demands, and operational sensitivity. 

---
# MUSS: A Multi-scale and Sequence-based Model for Single-cell Gene Regulation 

**Authors**: Kangjie Zheng, Amirhossein Vahidi, Liying Jin, Joseph Clarke, Vijaya Baskar MS, Connor Rogerson, Mengji Zhang, Muzlifah Haniffa, Anshul Kundaje, Mo Lotfollahi  

**Link**: [PDF](https://arxiv.org/pdf/2611.08086)  

**Abstract**: Modeling gene regulation in single cells requires jointly representing DNA sequence, chromatin accessibility, and gene expression across regulatory scales. Sequence-based models capture DNA-level regulatory grammar but often lack cell-level resolution, whereas many single-cell omics models represent genes and peaks as fixed-vocabulary tokens without their underlying sequence. Here, we present MUSS, a sequence-based multi-scale model for single-cell gene regulation that preserves both DNA-level and cell-level resolution. MUSS derives peak- and gene-level representations from DNA using a frozen pre-trained sequence encoder and integrates them through peak-peak self-attention and distance-biased peak-gene interactions. Peak masked language modeling followed by training on paired scATAC-seq and scRNA-seq aligns these representations with cell-specific regulatory states. Across gene expression prediction, enhancer-gene linking, motif-guided regulatory prioritization, and zero-shot cell clustering, MUSS achieves strong performance against sequence-based and single-cell omics baselines. It further predicts expression for genes and peaks unseen during task training. These results support sequence-grounded, multi-scale representations for decoding cell-type-specific regulatory programs. Code, datasets, and model checkpoints are available at this https URL. 

---
# DictLLM: Post-training Compression of Large Language Model with Dictionary Kernels 

**Authors**: Jinho Park, Se Young Chun, Mingoo Seok  

**Link**: [PDF](https://arxiv.org/pdf/2611.08082)  

**Abstract**: Efficient deployment of large language models (LLMs) on mobile devices requires balancing performance with compute and memory constraints. Post-training compression is an effective approach to managing this trade-off, while requiring much less computation than in-training counterparts. We propose DictLLM, a dictionary-based compression scheme in which weight matrices within grouped layers are decomposed into a shared dictionary and layer-specific sparse coefficients, directly obtained from a pre-trained model without further training. Specifically, we cast DictLLM as a four-step post-training optimization. First, we obtain shared dictionaries via singular value decomposition (SVD) of input-scaled weight matrices, amortizing rank across grouped layers; second, we sparsify the layer-specific coefficients to the target compression ratio using a dictionary-aware Hessian; and third, we refine the dictionaries given the resulting sparse coefficients. Finally, we aggressively quantize both components using factor-specific Hessian information: input covariance for the dictionaries and the dictionary-aware Hessian for the coefficients. The resulting representation reduces memory footprint by storing a single shared dictionary per layer group, along with sparse layer-specific coefficients. On language modeling and zero-shot reasoning benchmarks, DictLLM consistently outperforms existing post-training low-rank compression methods and pushes the Pareto frontier between model size and performance. DictLLM compresses the original LLaMA-7B model by 5.1x with only a 2.9-point increase in WikiText-2 perplexity, achieving a 3.6x higher compression ratio than the state-of-the-art low-rank method. 

---
# When Imperfect Automated Fact-Checking Backfires: Social Learning in the Presence of Misinformation 

**Authors**: Zixuan Liu, Ming Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.08053)  

**Abstract**: Automated fact-checking tools, cheap and fast but imperfect, are increasingly available on social platforms. We study when such tools can backfire in a sequential social learning model of misinformation. Agents observe an article's reliability and their predecessors' actions, which indicate whether they liked or disliked the article. They may also pay to fact-check it. We examine how welfare, defined as the probability of a correct cascade, changes when a free, error-prone automated tool supplements costly but perfectly accurate manual fact-checking. Two opposing effects arise. Broader use of fact-checking helps agents escape wrong cascades. At the same time, errors from the automated tool can mislead agents who would otherwise have verified the article perfectly, reducing the likelihood of correct cascades. We prove that adding the tool strictly lowers welfare for articles of intermediate reliability when truthful articles are not too informative and manual checking is sufficiently costly. Simulations show that the tool can help on either side of the intermediate range, and that it improves on manual-only checking on average only above a threshold accuracy. 

---
# Secure Formal Raising: Protecting the Development of AI through Executable Properties 

**Authors**: Rolf Drechsler, Christina Plump  

**Link**: [PDF](https://arxiv.org/pdf/2611.08042)  

**Abstract**: Model raising revisits Turing's developmental view of machine intelligence. Formal Raising (FR) has recently been proposed to complement this view with explicit behavioral properties. This paper extends that proposal to the security of the raising environment. Secure Formal Raising (SFR) models policy integrity, authorization, and confidential data flow using four-valued evidence that preserves unknown and inconsistent assessments. We prove six security invariants and show that they force admission to accept only confirmed evidence. A plain-C reference implementation satisfies the invariants in a bounded exhaustive exploration, which also exposes a confused-deputy gap for tool calls that a provenance requirement closes. 

---
# TCSExam: Evaluating Frontier LLMs on the Full Range of TCS Research 

**Authors**: Yifan Luo, Zhengying Liu, Yang Yuan, Zhilin Yang, Andrew Chi-Chih Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.07962)  

**Abstract**: Can large language models (LLMs) advance theoretical computer science and resolve its long-standing open conjectures? To study progress toward this goal, we introduce TCSExam, a unified framework and benchmark covering the full range of TCS research. It evaluates theorem proving, solver engineering, and constructions alongside direct attempts to resolve open conjectures, capturing both concrete research accomplishments and progress toward long-standing research goals. TCSExam-Full contains 100 expert-reviewed tasks, with submissions checked by the Lean kernel or an executable task checker. TCSExam-Lite selects 20 tasks, five per track, reducing the number of task attempts by 80% at the same per-attempt budget. We evaluate eight agents on Full and nine on Lite. The TCSExam Index combines all four tracks: GPT-6 Astra leads Full (71.9), and Opus 5.5 leads Lite (72.3). Full and Lite rankings are strongly aligned across eight shared agents (Spearman's $\rho=0.93$), supporting Lite as a compact, discriminative benchmark. The evaluations yield improved solver outputs and mathematical constructions, together with independently verified special cases and bounds from open-conjecture attempts. TCSExam provides an extensible framework with task sources, Docker build recipes, and evaluation tools for reproducible use. 

---
# A Behavioural Framework for Predicting and Understanding Loss of Control in Frontier Artificial Intelligence Systems 

**Authors**: Yao Huang, Yichi Zhang, Yifan Wang, Yitong Sun, Haoyu Dong, Jifan Ma, Jinzhe Tu, Xiaojian Li, Sören Mindermann, Wei Xu, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.07953)  

**Abstract**: Frontier artificial intelligence (AI) systems are increasingly operating over longer horizons with diminishing human oversight, raising the risk of loss of control: systems acting beyond effective human control with potentially irreversible consequences. Because such failures may become observable only after containment is difficult, they must be assessed prospectively. We introduce a behavioural framework that decomposes loss-of-control risk into three factors: misaligned motive, harm-enabling capability, and monitoring evasion. We operationalise these factors through 13 measurable aspects and 29 evaluation tasks, whose scores are combined into a single loss-of-control (LoC) score. Across 13 frontier models, the LoC score shows strong prospective validity, correlating with failure rates in contained long-horizon risk scenarios (Pearson r = 0.84). Risk varies systematically across models and cannot be explained by capability alone. We further show that the framework can guide targeted mitigations: fine-tuning and monitoring directed at diagnosed vulnerabilities reduce failure rates while largely preserving general performance. These results provide a practical basis for predicting, understanding, and mitigating loss-of-control risk before it materialises. 

---
# Circe: Training-Free LLM Self-Evaluation via Component Interaction Consistency 

**Authors**: Shuyi Miao, Zheng Li, Xu Xie, Linrui Xu, Fei Shen, Xiaoyu Du, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2611.07948)  

**Abstract**: As large language models (LLMs) are increasingly deployed in real-world applications, evaluating the correctness of their responses is crucial for reliable use. However, such evaluation is challenging because reference answers are rarely available in practice. Existing reference-free methods mainly rely on output signals, which may assign high scores to incorrect responses and thus do not necessarily reflect correctness. This motivates exploring the model's internal computation for complementary correctness signals. Recent studies exploit internal states, but they mainly analyze hidden-state trajectories across layers, which record only the combined updates of attention and feed-forward networks (FFNs) and overlook how these components interact. We therefore propose Circe, a training-free self-evaluation method that assesses a response using both internal and external signals, without reference answers or repeated sampling. For internal signals, we first introduce component interaction consistency (CIC), which measures the directional alignment between attention updates in earlier layers and attention-induced FFN output changes in later layers. We further incorporate representation geometry (RG) to characterize normalized displacement and angular changes along layerwise hidden-state trajectories. For the external signal, we convert the mean negative log-likelihood of generated tokens into a bounded likelihood confidence (LC) score. Finally, we fuse CIC, RG, and LC into a unified correctness score. Experiments on five LLMs and six benchmarks across four domains show that our method distinguishes correct from incorrect responses more effectively than existing baselines, and generalizes to multilingual, multimodal, and open-world settings. 

---
# ChemE-as-Code: An Open-Source Platform for Agentic AI in Chemical Engineering 

**Authors**: Artur M. Schweidtmann  

**Link**: [PDF](https://arxiv.org/pdf/2611.07947)  

**Abstract**: ChemE-as-Code provides an open-source environment in which AI agents built on large language models (LLMs) can carry out chemical engineering work and engineers can check it. Its calculation tools range from catalysis through process simulation to plant layout, HAZOP and life cycle assessment, and skills tell a general-purpose agent how to use them. Engineering files are linked, machine-readable and versioned in Git; calculations run outside the LLM, each value records its origin, and automated checks report inconsistencies and out-of-date results by file and line. In an illustrative ammonia case, agents developed a written task into a preliminary basic-engineering package (simulation files, P&IDs, layout, HAZOP, datasheets, relief) for USD 224 of API cost in the design sessions. Planted faults and engineering changes show that the checks find what their rules cover and miss the rest, so they guide human review rather than replace it. No independent engineer has reviewed the design. 

---
# How Finite Brains Generalize Without Bound: Spatial Memory as a Universal Computer 

**Authors**: Yuhang He, Si Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07920)  

**Abstract**: A child who learns to add small numbers adds numbers of any length. LSTMs and Transformers trained on short sequences fail on longer ones. We trace the difference to what a learner reads when it decides. A position, a count or a time keeps taking new values as the input grows: it is a coordinate, and a learner that reads one meets in every long input values its experience never covered. Whether two things are the same has the same two answers at every size: it is a coincidence, and a learner that reads only coincidences finds in a long input nothing that short inputs lacked, so finite experience fixes its rule for every size. The memory carries the size of the input and the rule stays finite, and this determines the memory a generalizing brain can have: a memory that grows while what the learner reads of it stays finite has to look the same from every place, so it is made of cycles, and the past is reached by asking what happened when the cycles last stood where they stand now. The cycles are the grid modules, the cells that detect their return the place cells, the records the hippocampal index; a prefrontal controller moves the cycles and never sees them. Reading only coincidences costs no power: what finite experience fixes in such a learner is what Turing machines compute in step with their input, which is why a circuit that evolved to find the way home can learn arithmetic and grammar. In a network of neurons this machine of cycles is the dynamics seen through its symmetry, and symbols are its invariants. Trained on strings up to 40 symbols, the machine is at least 99% correct, on average over lengths 41 to 500, on nine of fifteen benchmark tasks, five of them tasks that no LSTM or Transformer solves; given new cycles after training it reverses strings of 15,015 symbols without error and fails beyond that on the curve the theory predicts; and its controllers run unchanged on neurons. 

---
# Adaptive Semantic Branching: Disagreement-Localized Cross-Trajectory Regeneration for Reasoning Candidate Expansion 

**Authors**: Jeong-Yoon Kim, Tae-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.07918)  

**Abstract**: Reasoning trajectories generated for the same problem often arrive at conflicting answers. These disagreements reveal alternative intermediate decisions that can be reused to explore new solutions. We introduce Adaptive Semantic Branching (ASB), a training-free inference procedure that turns cross-trajectory disagreement into targeted candidate generation. ASB first groups independently sampled trajectories by extracted final answer, selects a representative from each group, and aligns their reasoning steps with a monotonic dynamic program. A persistent-divergence objective identifies the point at which initially similar reasoning paths develop different continuations. For each selected directed pair, ASB retains the source trajectory's textual prefix, extracts a short reasoning window from a competing donor trajectory, and generates a fresh solution conditioned on both. A diversity-aware policy allocates up to three additional generations according to the answer-cluster structure. In a 500-problem MATH evaluation with Qwen2.5-7B-Instruct, ASB increases oracle candidate coverage from 445/500 (89.0%) with seven initial samples to 455/500 (91.0%), recovering correct candidates for 10 of the 55 initially uncovered problems. The method uses 576 additional generation calls across the evaluation set, averaging 8.152 calls per problem. We provide a precise specification of the branching operator, alignment objective, generation policy, and experimental configuration. 

---
# GuidedWAM: Training-Free Energy Guidance for World Action Models in Autonomous Driving 

**Authors**: Xiyan Su, Finn Rasmus Schäfer, Hojun Lim, Frank Diermeyer, Johannes Betz, Markus Lienkamp  

**Link**: [PDF](https://arxiv.org/pdf/2611.07912)  

**Abstract**: World action models (WAMs) for autonomous driving couple the driving policy and the visual world model in a single generative network whose objectives are fixed post-training. Conditioning on a new driving objective requires retraining. We study energy guidance as a general mechanism and present GuidedWAM, a model-agnostic framework for injecting driving objectives into frozen WAMs at sampling time. Our framework constructs each objective as differentiable energies on the sampler's intermediate clean-action estimate, which needs only forward access to the frozen model, and is stabilized by sampling-time guidance scheduling, per-energy normalization, and velocity-bounded correction. On the NAVSIM v2 hard-split, guidance improves all three frozen, publicly released WAMs by 1.8 to 3.6 EPDMS, averaged over three sampler seeds, and adds less than 10% to the sampling time per step. These results position energy guidance as a practical post-training control channel for WAMs. 

---
# Persistent Retrieval for Conversational Agents: A Study of Retrieval Efficiency 

**Authors**: Karthik Ravula, Sathwik Kowtha, Rishabh Verma, Kala Mehak Jain, Gunjit Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2611.07899)  

**Abstract**: Conversational agents can retrieve information directly or delegate retrieval to a subagent. This design choice affects how retrieval history is maintained, what context each model call receives, and what information must be re-retrieved in follow-up questions. We compare three designs: skill-based retrieval in the main agent, a non-persistent retrieval subagent that generates a findings response, and our persistent retrieval subagent, which maintains retrieval history across turns and returns retrieved evidence directly to the main agent. Our approach also allows the main agent to request corrections from the same subagent.
We evaluate the three designs on enterprise questions posed independently, within shared conversations, and as dependent follow-ups. Across five multi-turn settings, our approach uses 43.3% fewer recorded agent-model tokens and incurs 45.0% lower estimated API cost than skill-based retrieval, while reducing average response time by 9.1%. Compared with non-persistent retrieval, it reduces response time by 15.6 seconds per question, while using 65.1% more tokens and incurring 29.1% higher estimated API cost.
Our approach shifts the retrieval subagent from producing synthesized responses to maintaining retrieval state and supplying evidence that the main agent can directly evaluate and use for answer generation. 

---
# When Bigger Becomes Brittle: Why Compression Hurts More as Language Models Scale 

**Authors**: Gautam Gupta, Dikshant Kukreja, Unique Kumar, Avni Mittal, Deepanshu Dabas, Kshitij Sah, Erik Cambria, Rajiv Ratn Shah, Zhengkui Wang, Avinash Anand  

**Link**: [PDF](https://arxiv.org/pdf/2611.07891)  

**Abstract**: Quantization and pruning reduce the cost of deploying language models, but their effects are usually reported as benchmark scores. Such scores record only the loss that remains at the output. Inside the model, compression damages some components more than others, and the rest of the network can partly compensate. The observed loss therefore reflects both the damage and this compensation, which benchmark scores cannot separate. How the two combine, and whether their balance changes with model size, remains unclear. We study this compensation for in-context retrieval, a capability carried by a small, identifiable set of attention heads. To separate the two, the expected loss from ablating these heads is decomposed exactly into the part restored by the rest of the network, attributed to the components responsible, and the part that reaches the output. This decomposition is then compared across model sizes. For damage of comparable size, smaller models restore most of the loss, whereas larger models retain most of it, because the compensating responses offset less of it as models grow. Compressing the heads produces the same pattern. Compression that is nearly harmless in small models can therefore be costly in large ones. 

---
# Responsible GenAI Use During Active Armed Conflict: Evidence from Sudan's Emergency Response Rooms 

**Authors**: Dalia Ali, Yearin Lee, Cheng Yu, Ro Encarnación, Alsanosi Adam, Danaé Metaxa, Ingmar Weber, Orestis Papakyriakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2611.07874)  

**Abstract**: Armed conflict can increase the value of Generative AI (GenAI) while undermining the conditions required for its responsible use. Yet the perspectives of community-based humanitarian responders working under these conditions remain underrepresented in responsible AI research. We examine GenAI use in Sudan's Emergency Response Rooms (ERRs), a community-led humanitarian network, through an exploratory survey of 89 respondents and 15 semi-structured interviews with GenAI users and non-users. War reconfigures GenAI's role, making it most useful where its outputs are hardest to verify. ERR responders turned to GenAI when conflict made human expertise difficult or dangerous to reach. However, participants cautioned that GenAI could not independently establish rapidly changing local conditions, and that relying on its outputs could lead to harmful decisions when human verification was difficult. Risks arose before GenAI was used. For some volunteers, carrying a phone or accessing the internet could expose them to danger. Others were concerned that sharing sensitive beneficiary or operational information with GenAI could put people at risk. Participants feared that AI-assisted reports could undermine their credibility with donors, whose funding sustains the food and medical aid ERRs provide. Access to GenAI was uneven, and volunteers using it sometimes relied on colleagues with relevant expertise or local knowledge to verify its results. Our findings show that responsible use of GenAI during conflict depends not only on model performance or individual adoption, but also on whether responders can safely access these tools, verify their outputs against local conditions, and recognize when limiting or avoiding GenAI is the more responsible choice. 

---
# FARR: Forecast-Aware Residual Refinement with Matched Endpoint Evidence for Long-Horizon Time-Series Forecasting 

**Authors**: Yiyang Zhang, Yuli Zheng, Bin Li, Meimei Zhang, Qiang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.07870)  

**Abstract**: Large language models use auxiliary conditioning to incorporate additional information into time-series forecasts. However, auxiliary conditioning does not consistently improve accuracy and can increase forecasting error when its influence is poorly matched to the observed temporal context. We propose Forecast-Aware Residual Refinement (FARR), which learns to correct forecasts using endpoint evidence. FARR obtains a valid forecast under the standard condition and a null forecast under a predefined null condition. Their difference reveals how conditioning changes the prediction. A shared residual refiner combines this endpoint evidence with historical features to estimate the remaining error of the valid forecast. Horizon-wise gating then controls the contribution of the predicted residual to the final output. Experiments on four long-horizon forecasting benchmarks show that FARR achieves the best dataset-average MSE and MAE among the compared methods. Code will be released at this https URL. 

---
# Generate-then-Rank: Learning Symbolic Goal Interpretation for Embodied Agents 

**Authors**: Jisen Lei, Huizhi Liang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07825)  

**Abstract**: Goal interpretation is to translate a natural language instruction into a formal symbolic goal that a planner can execute. It is critical for embodied agents because any error here propagates to every downstream planning and action step. Doing it well requires both generating suitable candidates and reliably selecting among them. We present a two-stage generate-then-rank approach to Goal Interpretation in the Embodied Agent Interface (EAI). First, supervised fine-tuning adapts a Large Language Model (LLM) to EAI's output convention. Second, Candidate-Output Ranking from Errors (CORE) learns preferences over complete outputs from the model's own generated alternatives. We conducted experiments on the household scenes of the Embodied Agent Interface Benchmark. Multiple open source LLMs including DeepSeek-V3 chat, DeepSeek-Reasoner, and Qwen2.5-7B and Qwen2.5-72B were evaluated. The proposed method outperformed the baseline models including LLMs with simple prompting, chain-of-thought, and knowledge graph based approaches. 

---
# The Intrusion Pathway: How Non-Evidential Context Degrades Language Model Performance 

**Authors**: Dikshant Kukreja, Gautam Gupta, Avni Mittal, Avinash Anand, Kritarth Prasad, Zhengkui Wang, Timothy Liu, Rajiv Ratn Shah, Aik Beng Ng, Simon See, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2611.07778)  

**Abstract**: Language models lose accuracy when a prompt contains non-evidential context: irrelevant statements from unrelated domains, or random off-distribution tokens, with no competing answer and no semantic overlap with the query. Such noise pervades retrieval-augmented, tool-using, and long-context deployments, yet the failure has lacked a mechanistic account. We trace the corruption to the Intrusion Pathway, which starts with MLPs enriching both query-subject and irrelevant-token representations, parroting heads carry prefix-token content to the prediction position, and relational readers largely avoid unrelated sources. Nevertheless, preserved subject enrichment and selective reading do not prevent intrusion in the broader computation. We then localize part of this interference to a sparse set of attention heads, which we call intrusion heads. Their prefix-derived writes promote distractor-related content, while their reduced support for the correct answer arises primarily from weaker non-prefix contributions rather than direct suppression by the prefix. The same heads transfer across tasks and distractor types and are largely weight-disjoint from the heads that integrate relevant contextual evidence, enabling a class-conditional intervention. Replacing the corrupted last-token residual with its clean counterpart, an oracle probe that requires a matched clean forward pass, recovers 97.6% on average of the distractor-induced drop across four downstream tasks. A deployable mask level knockout of about ten heads, roughly 1% of the model's attention heads, recovers 57% at bounded cost to helpful context. 

---
# RIVER: Evidence-guided multi-agent retrieval for enzyme--reaction discovery 

**Authors**: Yifan Chang, Yunze Li, Feifei Jiang, Juntao Wu, Xin Chen, Han Hu, Yucheng Wu, Wei Tang, Mingqian Ma, Wanli Ouyang, Kaipeng Zhang, Lei Bai, Hongting Tang, Feiran Li, Yuchen Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.07757)  

**Abstract**: Enzyme function annotation and the identification of candidate enzymes for a given reaction are essential steps in connecting genomic information to biochemical reaction networks. However, existing methods rely on sequence or structural similarity or on training with known enzyme--reaction pairs, making it difficult to integrate biochemical evidence across large candidate spaces; they also struggle with retrieval tasks involving enzyme families whose members are highly similar. Here, we introduce RIVER (Reaction--enzyme Inference via Evidence-guided Retrieval), a multi-agent framework for bidirectional enzyme--reaction retrieval that identifies potentially catalyzed reactions for a given enzyme and ranks candidate enzymes for a given reaction. RIVER coordinates large language models, specialized biochemical computational tools, and external knowledge resources to assemble evidence at multiple levels, including reaction chemistry, protein sequences and domains, catalytic mechanisms, and database annotations. Dedicated validation modules further examine evidence gaps and potential conflicts and rerank candidates using traceable evidence records. On public benchmarks, RIVER achieves the highest success rates in both enzyme retrieval and enzyme annotation. In family-specific enzyme retrieval, RIVER outperforms the evaluated baselines on the selected datasets and metrics. In Escherichia coli enzyme retrieval tasks constructed from recent literature, RIVER achieves the best retrieval results with closed-source and open-source model configurations, substantially outperforming closed-source general-purpose and biology-specialized multi-agent systems, including Claude Code, Codex, and Phylo Max. These results suggest that evidence-guided multi-agent retrieval can prioritize biochemically supported enzyme--reaction relationships, providing a basis for subsequent experimental validation. 

---
# Pediatric ECG foundation model via adult-to-pediatric fine-tuning: downstream tasks and clinical evaluation 

**Authors**: Yu Cao, Haiyan Xu, Xiaohan Li, Jian Yang, Yuqing Feng, Wei Wang, Haomin Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.07717)  

**Abstract**: Pediatric electrocardiogram (ECG) interpretation is complicated by age-dependent electrophysiological changes that limit direct transfer of adult AI models. We asked whether an adult ECG foundation model could be adapted to pediatric ECGs. Using 339,563 12-lead ECGs from 238,189 children, we compared direct application, pediatric fine-tuning, and training from scratch under an identical architecture. Direct transfer was poor (mean AUC 0.8256). Fine-tuning reached 0.9695, above de novo training (0.9661), and attained 0.9079 using only 1% of training data. On 10,999 external ECGs, fine-tuning achieved a mean AUC of 0.9385 across 30 labels. It generalized to age regression (MAE 1.12 years) and congenital heart disease classification (AUC 0.9562). Across 17 clinicians and 100 cases, diagnostic accuracy rose from 56.8% to 65.4% (p < 0.001) and did not decline when AI erred. Adult ECG foundation models require pediatric adaptation; fine-tuning offers a data-efficient route to scalable pediatric ECG AI. 

---
# Towards Generalist Graph-Level Anomaly Detection 

**Authors**: Junjun Pan, Yixin Liu, Yu Zheng, Fuyi Li, Alan Wee-Chung Liew, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2611.07706)  

**Abstract**: Graph-level anomaly detection (GLAD) aims to identify graphs that deviate significantly from the majority. While recent advances have achieved competitive performance, existing approaches mainly follow a dataset-specific paradigm, requiring substantial in-domain data and training cost when applied to new application scenarios. This limits their scalability and practicality in real-world applications where labeled data is scarce and diverse domains are continuously emerging. In this paper, we investigate the problem of generalist GLAD, which seeks to learn a single unified model capable of detecting anomalous graphs across multiple domains with minimal supervision on the target data. This problem introduces key challenges, including learning transferable patterns from heterogeneous graph datasets and effective adaptation under limited supervision. To address these challenges, we propose GenGLAD, a generalist GLAD approach that leverages discriminative subgraph extraction as the core mechanism to learn transferable anomaly knowledge and enable efficient adaptation to new domains. Specifically, GenGLAD employs a graph neural network-based discriminative subgraph extraction model to learn the anomaly-discriminative substructures that distinguish anomalous and normal graphs across diverse domains. During the test-time adaptation phase, multi-dimensional discrepancies between the extracted subgraph and multi-level contextual information are calculated to characterize anomalies from multiple perspectives. For efficient adaptation, a lightweight scoring module is introduced to automatically decide the most informative signals, improving robustness against misleading patterns. Extensive experiments on diverse benchmark datasets demonstrate that the proposed approach achieves strong performance, superior generalization ability, and improved data efficiency compared to existing methods. 

---
# MCPHallu: Benchmarking Reasoning, Execution, and Memory Hallucinations in MCP Agents 

**Authors**: Yizhen Jiang, Haowei Guo, Tianyi Bai, Zengjie Hu, Lu Ma, Zhengyang Zhao, Haoze Sun, Peng Pei, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07689)  

**Abstract**: Large language model agents are increasingly deployed through the Model Context Protocol (MCP), which allows them to discover and invoke tools across multiple servers at runtime. Existing MCP benchmarks primarily measure task completion; where they diagnose failures, they do so after execution rather than building specific reasoning, execution, and memory failure risks into task construction.
We introduce MCPHallu, a benchmark for diagnosing hallucination-prone agent failures in MCP environments along three axes: reasoning, execution, and memory. MCPHallu contains 358 tasks across five capability domains and runs on 36 production MCP servers in isolated containers. Each task is designed with a structural trigger that makes one of four subtypes the dominant failure risk: Branch Collapse, Unreachable Goal, Tool Misuse, or Context Forgetting. This design enables subtype-level analysis rather than aggregate task success alone.
We evaluate fourteen frontier LLM agents from eight vendors on 5{,}012 trajectories. Current MCP agents remain vulnerable to hallucination-inducing tool environments, achieving an average score of 0.642 on a $[0,1]$ scale (higher is better). Failures concentrate on Unreachable Goal tasks, where the average score drops to 0.458 and 23.2\% of runs pass, indicating that agents often continue issuing tool calls or claim successful completion rather than recognizing infeasible goals. Subtype rankings diverge sharply: a model's rank shifts by 6.9 of 13 positions on average across subtypes, and Branch Collapse is uncorrelated with Unreachable Goal across models ($r = -0.12$). MCPHallu is released at this https URL. 

---
# Who Should Know What? Empirical Information Design for Multi-Agent LLM Economies 

**Authors**: Mete Erdogan, Mirac Suzgun, Aneesh Pappu, Guowei Xu, Batu El, Andrew A. Schoen, Furong Huang, John P. Dickerson, Mert Pilanci, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2611.07667)  

**Abstract**: The performance of multi-agent LLM systems depends not only on the capabilities of the underlying models but also on information-sharing policies, which govern what information each agent observes and at what level of detail. But does providing AI agents with more detailed information always improve collective outcomes, or can transparency paradoxically degrade efficiency? To investigate this, we introduce empirical information design, an experimental framework that systematically compares disclosure policies across economic markets populated by LLM agents, while holding interaction rules and models fixed. We evaluate ten frontier LLMs in multi-resource trade, bilateral bargaining, and double auctions, using ground-truth utilities to decouple identified opportunities from realized outcomes. We find that finer information disclosure yields highly model-dependent results. In trade, revealing exact valuations rather than ordinal preference rankings reduces realized gains for GPT-6 Sol (from 98% to 91% of maximum) yet increases them for Claude Sonnet 5 (from 72% to 86%). In bargaining, exact information leads Grok 4.7 to produce less balanced agreements, whereas Claude Opus 5.5 divides gains symmetrically. In double auctions, full disclosure inflates price demands, triggering market breakdown and lowering welfare for Gemini 3.1 Pro (97% to 78%), Claude Opus 5.5 (94% to 64%), and Grok 4.7 (97% to 62%). Overall, these results suggest that model selection and information design must be co-optimized, as LLMs vary in how their strategic responses to information impact system objectives. 

---
# PharmAgents2: Towards Autonomous End-to-End Drug Discovery 

**Authors**: Haichuan Tan, Bowen Gao, Tao Huang, Jiaxin Tan, Xin Hong, Baoguo Yao, Yifei Wang, Zhichao Deng, Hanji Du, Dapeng Jiang, Yiheng Li, Weiwei Ma, Xin Ma, Yuqi Shi, Zixiang Wang, Kelin Wu, Tianxiang Wu, Wenxuan Xie, Jie Zang, Huiyu Zhang, Mojia Zhang, Yuying Zhang, Ya-Qin Zhang, Wei-Ying Ma, Yanyan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2611.07653)  

**Abstract**: Drug discovery requires continuous scientific decision-making across a vast research space. We introduce PharmAgents2, an autonomous system for end-to-end drug discovery that combines knowledge-enhanced decision-making with organized large-scale exploration. Paper2Skill extracts transferable scientific decision knowledge from public sources and, together with expert-developed skills, supports research decisions. Self-Scaling Discovery expands exploration through parallel agents inheriting research context and decision-making capabilities across targets, scientific hypotheses, and discovery strategies. We introduce PharmAgents-Bench v0.1, comprising 121 questions across nine decision scenarios and three end-to-end tasks assessed with expert-defined process rubrics. PharmAgents2 outperforms evaluated baselines in most decision categories and achieves the highest process scores on all three end-to-end tasks. In three real-world studies, autonomous investigations generated findings subsequently supported by human-executed experiments. For LpxH, PharmAgents2 hypothesized a previously unreported allosteric site and screened compounds against it, identifying a novel inhibitor scaffold with experimentally confirmed inhibitory activity and binding. For JAK1, it prioritized a covalent inhibition strategy and nominated compounds with experimentally demonstrated activity. Starting from a known bioactive molecule, Self-Scaling Discovery explored candidate targets across the proteome and identified a new binding target, with the interaction subsequently confirmed experimentally. PharmAgents2 represents a prototype of a virtual pharmaceutical company, marking a step toward autonomous end-to-end drug discovery. 

---
# Don't Prompt Your Agents. Equip Them: Skillware as a Modular Standard for Sovereign, Auditable, and Deterministic Capability Execution for Logical Systems 

**Authors**: Vladimiros Peilivanidis  

**Link**: [PDF](https://arxiv.org/pdf/2611.07652)  

**Abstract**: Current autonomous logical systems rely on a fragile assumption: that language models should dynamically write code, guess API payloads, and memorize operational constraints through prompt engineering. In practice, this burns massive context windows, inflates API bills, and produces brittle agents that fail unpredictably in production.
In this work, we present Skillware, an open architectural framework and capability registry that formally separates what an agent wants to do from how that task is executed. Instead of prompting models with loose text instructions or markdown files baptized as 'skills', Skillware equips them with modular, versioned software capability bundles. Much like skill hotkeys in video games that execute deterministically under fixed world rules, each bundle combines a typed JSON input contract, deterministic Python code, task directives, bundled datasets, and automated unit tests. In-process adapters translate these capabilities into native tool declarations for any model, from frontier cloud APIs to local open-weight runtimes with zero network latency.
Evaluating 23 frontier and open-weight models across 368 benchmark conditions, we show that small local models (3B to 8B parameters) equipped with Skillware consistently outperform unequipped frontier models on complex multi-step workflows. They cut token usage by up to 84.8%, slash inference costs by up to 97.3%, and eliminate parameter errors entirely. True agent reliability does not come from larger models or longer prompts, but from equipping logical systems with verified, deterministic software.
Finally, we articulate the trajectory of capability bundles across an invariant continuum of information processing systems: from digital logical systems today, to IoT, drones, and embodied robotics tomorrow, establishing a foundation for auditable, real-time capability execution. 

---
# Redacting Personal Data: A Speed and Accuracy Comparison of Regex, Presidio and Hybrid Designs 

**Authors**: Sai Yashwant, Manoj Kumar Gondimalla, Arjun Sharma, Shubham  

**Link**: [PDF](https://arxiv.org/pdf/2611.07633)  

**Abstract**: Many enterprise projects now pass free text to large language models (LLMs), and that text often contains personally identifiable information (PII) such as names, phone numbers, email addresses, home addresses and dates of birth. Once a prompt is sent, this data is copied into provider systems, application logs and observability traces. Redacting PII before the call is an obvious control, yet teams seldom have numbers on what it costs in latency or what it lets through. This paper compares four redaction designs that any LLM pipeline can adopt: a pure regular-expression (regex) redactor, Microsoft Presidio running on a spaCy model, a hybrid in which Presidio finds only person names while regex handles structured identifiers, and a fast hybrid that calls spaCy's named-entity component directly. All four share a known-entity rule, which removes the person the system already knows it is handling, and a consistent placeholder scheme, so the LLM can still tell one person from another after redaction. We evaluate the designs on a sample workflow, which we call ProfileAssist, that turns resumes into structured profiles, using 100 synthetic resumes (523 characters on average, with 900 PII values and 600 business terms that must survive). Regex ran at a 0.20 ms median but caught only 88.9% of PII, because it cannot find names that are not known in advance, such as the references listed in a resume. Presidio, hybrid and fast hybrid each caught 100% of PII and kept 100% of business terms, at median latencies of 37.59 ms, 30.23 ms and 13.48 ms. Measured against a 9,120 ms median LLM call, even the slowest option adds only 0.41% to the request. We therefore recommend the fast hybrid as a default for LLM pipelines that handle personal data, with regex kept as a safety net rather than as the only control. 

---
# Beneath the Surface: What We Overlook When We Talk About Agent Evolution 

**Authors**: Xiaobo Wang, Zimu Wang, Siyu Ye, Jiaqi Li, Min Tang, Jun Bai, Yuling Shi, Ruobing Wang, Jingjing Zhang, Chaofan Wang, Kai Wu, Mengnan Qi, Kai Cai, Xiaodong Gu, Qi Liu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.07624)  

**Abstract**: Agent harness evolution enables recursive self-improvement (RSI) without updating model weights. We examine whether higher benchmark scores yield reliable, efficient, and transferable improvement. We evaluate four methods across harnesses, general and coding tasks. Repeated evaluations, controlled agent expansion, component interventions, and admission replays reveal recurring patterns. All sixteen evolved agents gain previously uncovered tasks while losing previously covered ones. In one general-task case, reliability rises by 5.2 points despite losing ten previously covered tasks. Most reliability gains also increase deployment cost. Specific changes usually outperform generic reminders, but their benefits can change when combined. The highest-scoring agent does not yield the largest mean improvement in any of our four pools, and stricter admission can discard useful alternatives. Transfer selectively preserves gains: coverage usually falls on the general-task target across both harness types and paradigms, but rises on the coding target with the outer-layer-only harness. These findings motivate preservation-aware evaluation, tested decision rules, and explicit applicability conditions. As a constructive follow-up, EvoAudit turns these diagnostics into revisions of select, analyze, mutate, and gate, achieving the best performance. 

---
# Embedding-Space Fingerprinting of LoRA Fine-Tunes: Recovering the Training-Data Domain from the Model Artifact Alone 

**Authors**: Shivesh Rane  

**Link**: [PDF](https://arxiv.org/pdf/2611.07610)  

**Abstract**: When a base language model is fine-tuned with Low-Rank Adaptation (LoRA), how much of the training data domain leaks into the resulting artifact? This paper studies whether a simple classifier can recover the domain of a fine-tune given only the model artifact (weight deltas, internal activations on a fixed probe set, or generated text) without ever observing the training data. Using one base model (Qwen3.5-0.8B), 36 LoRA fine-tunes are produced over 18 public datasets in five domains (medical, code, mathematics, non-English text, narrative prose), and three feature families are extracted at decreasing levels of access: white-box weights, gray-box activations, and black-box outputs. A leave-one-dataset-out (LODO) protocol, frozen before any result was seen, tests whether the domain of a fine-tune trained on a previously unseen dataset can be identified. The domain is recoverable well above chance (20%) from gray-box activations (61%) and black-box outputs (64%); white-box weight statistics reach 42%. Contrary to the pre-registered hypothesis that signal decays monotonically from weights to activations to outputs, activations and outputs were more informative than the weight-delta statistics. The black-box result holds unchanged (64%) when only generated text is used, without token probabilities. Domain accuracy is also stable when a third run per dataset is trained on an independently drawn data subsample, and black-box predictions are unaffected by the LoRA rank (8-64). All code and per-run feature vectors are released. 

---
# Align Before Aggregation: Basis-Consistent Federated LoRA under Heterogeneous Ranks 

**Authors**: Pengpeng Qiao, Yang Cao, Lingling Zhang, Guo Cheng, Junwei Chen, Manjiang Yu, Wei Yang Bryan Lim, Masatoshi Yoshikawa  

**Link**: [PDF](https://arxiv.org/pdf/2611.07598)  

**Abstract**: Federated fine-tuning of large language models (LLMs) under heterogeneous client capacities is increasingly important, and low-rank adaptation (LoRA) with heterogeneous ranks makes it practical. Existing server-side aggregators compose local matrix updates $\Delta W_i$ from client-specific LoRA factors $(A_i,B_i)$ before aggregating them. Although these matrix updates lie in the same weight space $\mathbb{R}^{m\times n}$ and can be averaged directly, the LoRA factors that generate them may expressed client-specific local basis. In the typical compose--aggregate--factorize--dispatch loop, this basis inconsistency can make the rank-$r_i$ reference prefix dispatched to lower-rank clients unstable after server-side refactorization. In this work, we propose \textbf{FedAbA}, a basis-consistent federated LoRA aggregation framework for heterogeneous ranks. FedAbA first extracts each client's local basis, aligns it to the corresponding prefix of the global reference basis, and constructs a basis-consistent reconstruction of each local matrix update before aggregation. It then aggregates these reconstructions with weights that combine client data size and consistency score, followed by exact coefficient-space SVD refactorization and gauge fixing before rank-prefix dispatch. Our theoretical analysis clarifies the role of basis consistency and supports the proposed consistency-aware aggregation mechanism. Extensive experiments show that FedAbA outperforms representative federated LoRA baselines, with the largest gains under the strongest rank heterogeneity. 

---
# FlashOffloading: Low-Latency KV Cache Offloading for Long-Context LLM Serving 

**Authors**: Xuan Wang, Zhuanhao Wu, Tingzhu Fang, Xushi Li, Sicheng Zhang, Zhenyu Yang, Feng Liu, Zhaohui Fan, Qiang Xie, Huayi Zou, Yuncong Liu, Zeyu Song, Ke Zhu, Yanjia Zeng, Jingchun Gao, Yan Shi, Kui Luo, Gezi Li, Liangjun Feng, Cheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.07542)  

**Abstract**: In large language model (LLM) serving systems, the key-value (KV) cache grows linearly with sequence length and concurrency, so long-context inference easily hits the high-bandwidth memory (HBM) capacity wall and results in low concurrency. Offloading the KV cache to host dynamic random-access memory (DRAM) frees HBM space and raises concurrency, but the additional cache management and host-to-device (H2D) load overhead inflates inference latency, higher concurrency fails to translate into effective throughput. We present FlashOffloading, a host-DRAM-centric inference architecture that fully offloads the KV cache in both the prefill and decode phases. FlashOffloading builds rolling HBM cache for prefill, a hot cache for decode, a reverse sparse index with zero extra miss-lookup overhead, and a compute--load overlapped kernel, optimizing KV offloading latency across the entire pipeline and delivering throughput gains under a time-per-output-token (TPOT) service-level objective (SLO). In experiments deploying GLM-5.2 with multi-token prediction (MTP3) speculative decoding on an Ascend910C prefill--decode (PD)-disaggregated platform, FlashOffloading improves prefill throughput by up to 1.38\(\times\) and decode throughput by up to 1.91\(\times\) under a 30 ms TPOT constraint. FlashOffloading is integrated into an open-source serving framework: this https URL. 

---
# CST-Bench: Benchmarking Counterfactual Spatial Transition Reasoning with Causal Transition On-Policy Self-Distillation 

**Authors**: Ying Li, Yongle Cheng, Jing Li, Xingquan Cai, Yawei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.07541)  

**Abstract**: Spatial reasoning is a fundamental capability of multimodal large language models (MLLMs), yet existing benchmarks mainly focus on static scene understanding. When objects are moved or swapped, models must infer not only spatial relations but also how interventions change the scene state. To address this problem, we introduce CST-Bench, a benchmark for counterfactual spatial reasoning. CST-Bench contains 7,082 examples covering 6 intervention types and 12 transformation operators, with dense causal annotations including intervention sources, affected relations, expected effects, invariant relations, and metric changes. We benchmark general-purpose MLLMs on CST-Bench and further compare post-training strategies on CST-Bench and standard spatial benchmarks. Results show that methods effective for ordinary spatial reasoning can still fail under counterfactual interventions. We further formulate counterfactual spatial reasoning as causal state-transition modeling and propose CT-OPSD, which uses CST-Bench annotations to guide a teacher model in learning counterfactual state transitions and causal reasoning chains. These chains are then distilled into a student model that reasons only from real observations. Overall, experiment results establish CST-Bench as a challenging benchmark and reusable supervision resource for counterfactual spatial reasoning. CT-OPSD further provides an initial demonstration of how its annotations can support model improvement. 

---
# Goal-Branching LaCAM: Temporary Goals as Search Branches in Dense Multi-Agent Pathfinding 

**Authors**: Stephen Chang, Daniel Harabor  

**Link**: [PDF](https://arxiv.org/pdf/2611.07528)  

**Abstract**: LaCAM is a search-based solver for multi-agent pathfinding (MAPF) that runs a depth-first search over joint configurations, using PIBT to generate successors and lazily adding single-agent constraints to explore alternatives. Although LaCAM is complete, it can struggle on dense, corridor-heavy instances because extracting blocking agents from deadlocks requires coordinated moves of several agents, often away from their goals, which PIBT's greedy one-step successors rarely propose and which LaCAM can only reach by enumerating constraints one agent at a time. This paper proposes Goal-Branching, where blocking agents are given temporary goals, each tried as an additional branch of LaCAM's search. We examine random and corridor-aware rules for choosing these temporary goals and show that completeness is preserved for any such rule. On dense instances (60-90% of free cells occupied) of three maze and room maps, Goal-Branching solves 89% of instances against 44% for LaCAM*, and, on instances both solve, it finds a first solution 10-14x faster on maze-32-32-2 and room-32-32-4, and nearly 200x faster in median time at 75% density on room-32-32-4. Solution cost increases by 15-22% on instances both solve, while on open maps and the full MovingAI benchmark its solutions cost no more than those of LaCAM* and it is at most 3% slower. 

---
# Graph learning-assisted logic-based Benders decomposition for distributionally robust in-station train rescheduling under disruptions 

**Authors**: Xiaobin Li, Yanbin Gao, Jianing Long, Wuming Lei  

**Link**: [PDF](https://arxiv.org/pdf/2611.07520)  

**Abstract**: Uncertainty in resource reopening times complicates train holding and rerouting decisions when resource closures coincide with arrival delays and processing-time extensions. We develop a rolling-horizon distributionally robust model that updates predictions of the remaining closure duration as observations of disruption resolution become available. The model jointly optimizes train routes, sequences, and timings. It minimizes the worst-case expected total completion delay over an ambiguity set of scenario probabilities. The solution algorithm, GNN-LBBD, uses logic-based Benders decomposition to coordinate routing and resource scheduling. Conditional cuts are constructed from resource-use patterns, and a graph neural network guides the order in which resource relaxation subproblems are solved. Across 28 test instances for a five-track station, GNN-LBBD achieved a mean total completion delay of 61.13 train-min. This was 16.63\% lower than tabu search and 20.06\% lower than variable neighborhood search, with a mean total computation time close to that of tabu search. Information comparisons on six representative instances showed that using observations of disruption resolution reduced mean total completion delay by approximately 21.7\%. Scenario-based and conditional-mean predictions produced similar overall results under the same information conditions. The benefits of information were more pronounced in some instances with alternative routes. This suggests a relationship between the use of observations, the station resource structure, and the scope for schedule adjustments. 

---
# DelegateEval: Evaluating Platforms for Personal External Service Agents 

**Authors**: Jichao Li, Jinrong Lu, Manjiang Yu, Zhixuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07508)  

**Abstract**: AI agents that work for an individual over time may also help that individual serve external clients. We define a personal external service agent as an agent authorized and continuously managed by a specific individual to accept client requests, deliver work, and provide ongoing support using that person's knowledge, experience, professional methods, and service boundaries. Such agents could allow individuals to serve more clients with less direct involvement and generate economic value at scale. We propose DelegateEval, a platform-level evaluation framework for agent systems that may combine large language models, reusable skills, protocols such as MCP, and executable tool environments. It covers eight dimensions: ease of creation, service discovery and matching, delivery quality, expertise inheritance, service continuity, learning and improvement, reliability and governance, and efficiency and value. These dimensions assess whether a platform enables people with different backgrounds to turn their expertise into ongoing services that deliver verifiable work, develop capabilities through experience, and remain economically feasible. We present the framework and an experimental design. Complete datasets, executable environments, a unified scoring protocol, and cross-model results remain to be developed and empirically validated. 

---
# SAVeR$^2$: Reasoning-based Safety Alignment for Large Reasoning Models via Verifiable Rewards 

**Authors**: Yichen Sun, Qirui Liu, Yan Wang, Jianan Lin, Linbo Jiang, Shiyu Wang, Zhibo Wang, Zhixuan Chu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07482)  

**Abstract**: Large Reasoning Models (LRMs) have achieved impressive performance via explicit Chain-of-Thought (CoT) reasoning, yet this process introduces critical safety risks. We formalize LRM safety objectives as requiring non-malicious reasoning and answers, safety consistency, and explicit intent identification. We argue that existing alignment methods suffer from poor adversarial generalization, a significant ``safety tax'' on reasoning, and unfaithful coarse-grained rewards, primarily due to their neglect of the LRM's intrinsic reasoning capabilities for safety. To address these, we propose \textbf{SAVeR$^2$}, a novel \textbf{S}afety \textbf{A}lignment method via \textbf{R}ewards with \textbf{R}easoning capability. SAVeR$^2$ decomposes the safety objective into four fine-grained verifiable reward signals and utilizes a difficulty-aware data selection strategy to stabilize training. Crucially, we introduce a reasoning-preserving gradient projection mechanism that analytically resolves conflicts between safety and reasoning gradients, effectively eliminating the safety tax. Extensive experiments on models ranging from 1.5B to 671B demonstrate that SAVeR$^2$ achieves superior safety performance and significantly lower over-refusal rates across multiple benchmarks without compromising general reasoning capabilities. 

---
# FINCOSMOS: An Open-World Evaluation Paradigm for Agents in Finance 

**Authors**: Yupeng Cao, Xueqing Peng, Lingfei Qian, Haohang Li, Yangyang Yu, Anke Xu, Wenbo Cao, Yan Wang, Yuechen Jiang, Yuyan Wang, Zhuohan Xie, Yankai Chen, Zehan Li, Jimin Huang, Xue Liu, Jian-Yun Nie, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2611.07464)  

**Abstract**: Language agents increasingly work for users on the open Web, and finance is where trusting them is hardest. The data must come from the Web, the strategy lives in the user's head, and correctness depends on time, so an agent can fail silently: a strategy the user never asked for, unverified data, or a backtest that uses the future yields a report as convincing as a correct one. Existing benchmarks close the world somewhere, fixing the task, the data, or the timeline, and thereby remove these failures instead of measuring them. We introduce FINCOSMOS, an open-world evaluation paradigm for agents doing financial work. A simulated user holds a hidden strategy sampled from a compositional grammar and reveals it only when asked. The agent receives no data, so it must acquire data from the Web, build its own backtest under a data cutoff, and deliver verifiable results. Grading needs no reference answer. Programs verify point-in-time correctness and recompute every reported metric, the delivered strategy is checked against the hidden specification, and a run is accepted only when the user is satisfied and none of 15 veto checks fails. Across four coding agents and a model without tools in 150 sessions, no session is accepted: every session that the simulated user signed off fails a program check, and agents raise only 54--64\% of the details they were not told. Human experts attribute 88.33\% of failures to the agents. With FINCOSMOS feedback, an agent with persistent memory learned all ten tasks. 

---
# EnvOpt: Automated Optimization for Agent Environments 

**Authors**: Mingxuan Li, Huanzhi Mao, Qiuyang Mang, Hanchen Li, Joseph E. Gonzalez, Alex Dimakis  

**Link**: [PDF](https://arxiv.org/pdf/2611.07372)  

**Abstract**: An LLM agent's cost and success depend on its model, its harness, and its environment. In addition to the widely studied directions of parameter tuning and harness optimization, we propose the third direction: optimization over stationary environments. Agents are often deployed in rich environments shared by many tasks, such as a company's Slack workspace or a complex collection of data sources, documents or tickets. We call these environments stationary environments. We introduce EnvOpt, a method for automated end-to-end optimization of stationary environments for any given model and harness, using real or synthetic tasks without needing any graders or reference answers. On AutomationBench, EnvOpt reduces the cost of Claude Haiku 4.5 by 72.7% and raises its pass rate by 14.34%. The same environment transfers to Opus 5, cutting its cost by 68.1% with a 1.34% gain. Added on top of Meta-Harness's optimized harness for Haiku 4.5, it lowers cost by a further 83.4% and adds 11.17% accuracy over Meta-Harness alone. Across 5 benchmarks, EnvOpt significantly reduces the cost of agents while improving or maintaining pass rate, and is transferable to different models and different harnesses. We also discuss common successful and rejected environment optimization strategies during automated optimization. More broadly, our results show that how information is represented to an agent shapes both its cost and its success, making the environment worth optimizing in addition to the parameters and the harness. 

---
# Robust Ranking of Port Impacts from Impending Tropical Cyclones: A Two-Stage Recommendation System 

**Authors**: Chenchen Kuai, Shixuan Tang, Chaolun Ma, Xiaoqiang Kong, Xiubin Bruce Wang, Yunlong Zhang, Zihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.07296)  

**Abstract**: When a tropical cyclone (TC) threatens several ports, preparedness planning depends on anticipating which ports are likely to be affected and where disruptions may last longest. Limited historical observations at individual ports and uncertainty in landfall location make these relative impacts difficult to estimate. We use a two-stage recommendation framework that first selects candidate ports based on their proximity to the expected landfall location and then ranks them by their estimated number of impact days. Using AIS-derived impact days for 37 ports and 31 TCs in the Gulf of Mexico study region, we compare three ways of representing port--cyclone interactions: pairwise feature interactions, tree-based feature combinations, and explicit feature crosses combined with learned embeddings. In tests on Hurricanes Nicole and Ian, the retrieval stage consistently identifies all observed affected ports under landfall uncertainty. Among the ranking models, Wide & Deep Learning achieves the closest agreement with the observed port rankings. Its mean ranking agreement exceeds that of the distance-based baseline by 85.5% for Nicole and 55.5% for Ian and is less sensitive to uncertainty in the landfall location. These rankings could help port authorities prioritize post-cyclone inspections and allocate recovery resources across ports threatened by the same event. 

---
# Goal Revelation as a Bridge Between Information-Theoretic Empowerment and Capability-Based Human Power 

**Authors**: Jaime Ruiz-Serra, Tomáš Gavenčiak, Jobst Heitzig  

**Link**: [PDF](https://arxiv.org/pdf/2611.07247)  

**Abstract**: Power matters for AI safety. Optimal agents tend to acquire power across a wide range of objectives (instrumental convergence). As we embed AI agents into socio-technical systems, humans risk gradual disempowerment. Both concerns call for a formal, quantitative measure of power. Two such measures in the context of Markov games are information-theoretic empowerment, the capacity of the channel from an agent's actions to future states, and capability-based individual power, a measure of goal attainment built from the probabilities that a human would attain each goal in a family of candidates were they pursuing it. We show where the two meet. From individual power, Fano's inequality yields a floor $B(s)$. This floor is a lower bound on what we term goal revelation, the mutual information $\mathbb{I}(G;\xi \mid s)$ between the human's latent goal and the realised trajectory. Empowerment caps revelation from above, so that $B(s) \le \mathbb{I}(G;\xi \mid s) \le \mathbb{I}(\mathbf{A}_h \to \xi \mid s) \le \mathfrak{E}_h(s)$. Each quantity can be large while the one below it stays at zero, highlighting the following failure modes. An assistant maximising empowerment can prefer an interface offering broad influence over goal-irrelevant degrees of freedom to one that lets the human attain every goal they might hold. An assistant maximising revelation cannot tell a working interface from one rewired so that every goal yields the wrong outcome, since relabelling goals leaves mutual information unchanged while zeroing every attainment probability. Positive guarantees therefore come only from the attainment end of the chain. A satisfaction floor, computable from two statistics of interaction logs, offers a way to monitor the power users retain in a deployed system, and simulations in random environments show the three quantities are correlated but far from interchangeable. 

---
# PromptOps: Iterative Prompt Optimization for LLM-as-a-Judge Systems 

**Authors**: Sven Kleinkop, Tamás Janusko, Ricardo Bochnia, Gunnar Hempel, Steffen Tomschke, Jürgen Anke, Maik Thiele  

**Link**: [PDF](https://arxiv.org/pdf/2611.07123)  

**Abstract**: Large language models (LLMs) are increasingly used as automated judges for evaluating the behavior of chatbots and other AI systems. In deployment, however, their agreement with the intended human evaluation policy may degrade as user interactions and score distributions change. Existing automatic prompt optimization methods are primarily studied as one-shot procedures on static development sets and therefore provide limited guidance for maintaining LLM judges over time. We frame this problem as judge drift and introduce PromptOps, a closedloop framework for monitoring and iteratively recalibrating LLM-as-a-Judge (LLMaaJ) systems. PromptOps separates four operational decisions: when to optimize, which optimization procedure to apply, which feedback records to use, and how to update the judge prompt. We instantiate the framework with five prompt optimization methods: GEPA, SIMBA, Rubric Synthesis, and exemplar selection using either simulated annealing or a Tree-structured Parzen Estimator. Our experiments demonstrate that iterative prompt optimization can recalibrate LLM judges and recover performance after controlled shifts in the referencescore distribution. However, effective maintenance is inherently contextdependent: neither optimization algorithm choice nor update timing generalizes universally across settings. These findings establish prompt optimization for deployed LLM judges as an inherently iterative maintenance problem and demonstrate that PromptOps provides a practical abstraction for studying and implementing this process. 

---
# Synthesizing symbolic models of human and machine reasoning at scale 

**Authors**: Solim LeGris, Philipp Mondorf, Jinran Jin, Akshay K. Jagadish, Brenden M. Lake, Todd M. Gureckis  

**Link**: [PDF](https://arxiv.org/pdf/2611.07088)  

**Abstract**: Symbolic models offer a powerful account of cognition, but their scope has historically been limited by the need for hand-engineering and computational tractability. We show how advances in artificial intelligence (AI) can scale symbolic modeling and help overcome these engineering limitations. We introduce Behavior-Conditioned Symbolic Synthesis (BCSS)---an approach for automated search of interpretable, deterministic symbolic programs that verifiably reproduce behavior---and apply it to model both human and machine behavior across hundreds of challenging reasoning problems. Analyzing these programs reveals that both people and machines tend to oversimplify, adopt rules that are partial explanations for their observations, and misinfer dependencies between problem features. These shared signatures are modulated by available resources, with reasoning effort predicting the extent to which machines simplify and explain their observations. Together, these results point to underlying mechanisms reminiscent of satisficing in both people and machines, and illustrate how BCSS offers a general approach for interpreting and comparing rule-based reasoning of humans and black-box AI systems. 

---
# When Verification Is Not Agreement: Cross-Agent LLM-to-Dafny Formalization 

**Authors**: Priyanath Maji  

**Link**: [PDF](https://arxiv.org/pdf/2611.07073)  

**Abstract**: A verified Dafny postcondition certifies that a formula holds, but it does not guarantee that the result was reached through genuine independent reasoning. We introduce a cross-agent formalization protocol where a Specifier model translates clinical trial criteria into a Dafny predicate, and an Implementer model writes an operational method checked against it. A same-model control establishes that even with no capability gap, independent re-extraction from identical text agrees only 59.3% of the time. Evaluating four model configurations on 96 clinical trial documents, we find that agreement is highly sensitive to how model capability is allocated across the two roles: the GPT-4.1-nano/GPT-4.1-mini configuration achieves 53.0% agreement when the stronger model is the Implementer, but only 14.3% when the roles are reversed. The strongest tested configuration, GPT-5.6-luna/GPT-5.4-mini, improves agreement to 34.4% but remains below the same-model baseline and is dominated by direct contradiction (42.7%). We also detect a critical verification artifact in which a weaker Implementer satisfies the postcondition by invoking the Specifier's predicate directly; after structural auditing, corrected agreement falls from 14.3% to 13.2%. Our results show that verification success alone is insufficient evidence of independent cross-agent agreement, and that model capability does not guarantee agreement when formalizing the same source independently. 

---
# Cooperative Multi-Agent Reinforcement Learning via Epigraph-Form Guided Exploration 

**Authors**: Sungil Son, Hoseong Jung, Dahyun Oh, H. Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.07066)  

**Abstract**: Discovering cooperative behavior in multi-agent reinforcement learning (MARL) is challenging due to the combinatorial complexity of joint state-action spaces, which hinders the emergence of coordinated behaviors from trial-and-error alone. Intrinsic rewards are often used to aid discovery, but naively combining them with team objectives can distort the learning signal, compromising task performance. In this paper, we propose EFXPLORER, a constrained exploration framework that maximizes exploration objectives subject to a constraint that preserves established task performance. We solve the resulting problem using an epigraph reformulation that introduces adaptive exploration budgets. This approach separates intrinsic rewards from task objectives and regulates exploration through task feasibility. To further encourage diverse and temporally extended exploration, we incorporate a successor distance-based intrinsic reward that captures long-horizon dependencies. Empirically, our method outperforms strong baselines and induces novel cooperative strategies across SMAX, VMAS, and MPE benchmark suites. 

---
# Closed-Loop Agent Evaluation: Layered Measurement and Self-Calibration 

**Authors**: Ruifeng Ren, Gun Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2611.07051)  

**Abstract**: Once an agent is live, teams want to know whether it is doing its job, not just whether it throws errors. Yet most evaluation is one-off: a benchmark run, a look at traces, a single score. Its conclusions neither check themselves nor reach the agent. We propose closed-loop evaluation, an evaluation method for deployed agents. It models evaluation as performance estimation under biased raters and censored outcomes, and it requires every level's conclusion to trace back to atomic data and to be computed by versioned rules. The method covers four levels of analysis (span, trace, conversation and engagement), measures along a spine of Execution, Substance, Trajectory and Results, treats Conduct as a constraint throughout, and uses Provenance and Adjudication components to certify the evidence and the evaluators. Evaluator agents generated by an Agent Factory carry these components, each with a domain knowledge pack and an adapter harness. Two loops close the evaluation: an inner loop audits the evidence, rules and metrics themselves, and an outer loop uses per-component regression checks to drive agent improvement. We also give a pipeline that runs this process automatically. Using the business-analysis agent SharpInsight as a case and porting the method to public conversation data, we show how problems differ across levels and how evidence-and-rule judgment is more reliable than holistic scoring (ICC 0.86 vs. 0.81; kappa 0.81 vs. 0.63). 

---
# Adapting the Process of Adaptation: Darwinian, Lamarckian, and Agent-Directed Evolution in AI 

**Authors**: Andrew W. Lo, Chaoyi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.07048)  

**Abstract**: Evolution has shaped intelligence for billions of years. Artificial intelligence (AI) now raises a striking possibility: intelligence may begin to direct its own evolution. As AI agents become capable of modifying and evaluating their own behavior, how quickly could they improve, and how can that improvement remain safe? We develop a theoretical framework comparing Darwinian, Lamarckian, and agent-directed evolution following environmental change. Darwinian evolution relies on natural selection, Lamarckian evolution adds inherited learning, and agent-directed evolution allows agents to generate and evaluate behavioral variants and choose how extensively to search. When learning is sufficiently effective, these mechanisms exhibit progressively faster adaptation. Generating more variants accelerates adaptation, while optimal search effort depends on learning ability, problem difficulty, and experimentation costs. However, an agent's ability to direct its improvement does not ensure that improvements are evaluated against the intended objective. When the evaluation target is misspecified, additional search accelerates convergence toward the wrong behavior, even as measured performance improves. In recursive self-improvement, such errors may persist across generations, posing challenges for AI safety. Independent evaluation and oversight must keep pace with autonomous AI development. The central challenge is not simply to accelerate the evolution of intelligence, but to preserve humanity's ability to guide its direction, limit its speed, and intervene when necessary. 

---
# Trust Responsive Runtime Authorization for Agentic AI with Empirical Characterization of Policy Behavior and Parameter Sensitivity 

**Authors**: Ahsan Habib Shihab, Imre Felde, Ramis Akhmedov, Stefan Gubo, Berik Sabdenaliyev, Madina Mansurova, Amir Mosavi  

**Link**: [PDF](https://arxiv.org/pdf/2611.06996)  

**Abstract**: The policy that converts runtime estimates of an autonomous agent's condition into operational authority is rarely specified or measured. We empirically characterize the Trust Responsive Authorization Policy (TRAP), which maps an Authorization Readiness Score (ARS), a weighted combination of runtime trust, operational stability, and contextual risk, onto discrete authorization levels through an asymmetric rule: escalation requires consecutive observations above a band edge plus a margin, while a sharp drop in readiness restricts authority at once. On one matched set of 100 Monte Carlo traces, we compare TRAP with static, trust-only, risk-only, and symmetric policies. Under abrupt degradation, TRAP restricts at onset and re-escalates sooner after recovery than the symmetric policy, with the highest and, descriptively, least variable stability among the adaptive policies, although more of its post-recovery escalations occur below the selected trust threshold. Paired bootstrap intervals exclude zero for the remaining differences. In one-factor sweeps, recovery time rises with the escalation margin and run length, shifts with the score weights, and is unchanged by the restriction threshold. In an exploratory post-freeze experiment, when the same degradation is spread over 5-20 intervals, TRAP restricts only on the 4 traces that retain a single-interval drop above that threshold; averaged over all 100 traces, it holds authority above what readiness supports for 15.9-18.9 of the 20 degradation intervals, against 0.8 under abrupt degradation and about 10 for the symmetric policy. Its restriction advantage therefore depends on degradation being abrupt. All reported values are traceable to the publicly available, versioned research release. 

---
# Autonomous AI Agents Systematically Diverge From Human Preferences in Proof Formalization 

**Authors**: Bálint Gyevnár, Simon DeDeo  

**Link**: [PDF](https://arxiv.org/pdf/2611.06995)  

**Abstract**: Artificial intelligence (AI) systems have ushered in a new age of proof abundance, but helping humans understand these proofs requires new attention to the differences in how AI agents and humans pursue a proof-making task. We present "simulated collaboration", a methodology that allows us to examine two key questions: (1) the relative attention AI agents pay to surface syntax or, conversely, deeper underlying semantics, and (2) how and where AI agent preferences for what to do next diverge from human preferences. On a set of nine major human collaborations in Lean with 4.7M definitions across more than 6K commits, we find that, over the course of proof construction, AI agents (1) are far more strongly drawn than humans to explicitly marked gaps; (2) show a relative aversion to densely connected proof centers and to complex or surprising proof goals; and (3) are relatively attracted to surprising proof bodies, which they tend to regularize toward baseline. Our results highlight important blind spots and potential interventions for future human-AI collaboration in mathematics. 

---
# The Two-Ledger Model: Transaction-Grain Cost Accounting for Multi-Agent AI Systems Across Clouds, Models, and Systems of Record 

**Authors**: Vivek Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2611.06989)  

**Abstract**: Enterprises run business transactions through multi-agent AI systems spanning clouds, large language models (LLMs), orchestration harnesses, and systems of record. Finance receives AI cost only at settlement grain, aggregated by SKU and charge period. The business question (what did this order, claim, or case cost in AI?) cannot be answered from billing data, because providers never see enterprise transaction boundaries. The FinOps Open Cost and Usage Specification (FOCUS) gives billing data one schema across providers but remains settlement-grain even as it adds AI token and requester attribution. We formalize this grain mismatch, derive seven requirements, and propose the Two-Ledger Model: a settlement ledger of FOCUS-conformant billing data joined to an attribution ledger of invocation-grain usage records built on OpenTelemetry GenAI telemetry. A layered correlation spine (trace context, A2A/MCP metadata, business-key correlation, probabilistic matching) assigns each record to its business transaction with recorded method and confidence; a versioned rate card prices usage in near real time; period reconciliation ties both ledgers to the invoice, and the unexplained residual identifies shadow AI. Costs are reported per order, claim, or outcome with agentic cost metrics. A simulation with known ground truth shows the spine sustaining cost coverage above 99% as trace propagation falls to zero, the reconciliation residual recovering planted unmetered spend within 0.24 percentage points while flagging provider billing errors, and mid-flight gating cutting p99 transaction cost by up to 61% at a 2-4% false-pause rate. We propose the attribution ledger as an invocation-grain FOCUS profile for standardization. 

---
# Resource-Efficient Reinforcement Learning Post-Training through Elastic Expert Parallelism 

**Authors**: Jia Wei, Ziyi Qiu, Yi Xu, Xuehai Qian  

**Link**: [PDF](https://arxiv.org/pdf/2611.06984)  

**Abstract**: Reinforcement learning post-training improves the capabilities of large language models, but generating training responses is a major computational bottleneck. In mixture-of-experts models, workers that finish their assigned responses may remain occupied because their experts are still needed by workers processing longer responses. Consequently, approximately one-third of occupied worker time in the baseline occurs after local response generation finishes. Here we present FlexFloor, a system that enables earlier resource release during synchronous response generation while preserving the expert computation required by ongoing responses. FlexFloor jointly adapts response scheduling and the active group of workers. It uses historical response lengths and memory constraints to guide batching and prompt placement, then progressively releases workers while maintaining expert availability and correct communication. Compared with the verl baseline, FlexFloor generates approximately 65.40% more response tokens per unit of occupied computing time and increases response generation throughput by 52.74%, with little observed variation in rollout reward. By making reinforcement learning post-training more efficient, FlexFloor supports faster development and iterative improvement of large language models and AI agents. 

---
# BlockLock: Reproducible LLM Inference with Fixed Batch Geometry 

**Authors**: Xiuyuan Qi, Pedro Reviriego, Javier Conde, Shanshan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.06977)  

**Abstract**: Reproducible LLM inference requires that a request produces the same output regardless of the other requests in the batch. Existing solutions pay for this in different ways: batch-invariant kernels constrain every operator, and isolating prefill with fixed-shape decode ({our CoRun-style baseline}) gives up shared prefill compute and pads decode. We present BlockLock, which keeps shared batched prefill and pins only the numerical path of the protected request: it fixes the geometry that the request sees, with {two modes: BlockLock-Rows uses fixed-width rows and retains batched decode of all requests; BlockLock-Pack uses terminal-anchored packing with an isolated dummy segment, then extracts the protected request's KV state for target-only decode. Under a stated per-deployment contract, both modes preserve the target's output bitwise when its companions change.} Across five {dense} models (Qwen3-8B/14B, Llama-3.1-8B-Instruct, and Gemma-3-4B/12B-IT) and three single-GPU deployments (NVIDIA A6000, A100, and H100), we evaluate our approach on both tinyBenchmarks and BFCL{; we also include initial tensor-parallel experiments on two NVIDIA A6000 GPUs. Both modes preserve target sequences and Top20-logit bits in the reported tests.} At equal delivered work, {BlockLock-Rows} reduces latency by 53.5--55.0\% and improves throughput by 115.2--122.2\% relative to vLLM Batch-Invariant; compared with batch-invariant kernels, it needs no custom kernel implementations and constrains only the protected requests. 

---
# KDASO: Knowledge-Data Dual-Driven Automated Skill Optimization for Liability Adjudication Tasks 

**Authors**: Zi-Jian Cheng, Weiming Wu, Jie Meng, Zhen Peng, Shan Huang, Qun Li, Guobing Wu, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.06972)  

**Abstract**: Liability adjudication tasks (e.g., legal charge determination) require analyzing event descriptions against domain-specific skills to determine whether a party should bear liability. While Large Language Models (LLMs) have shown promise in this area by leveraging executable skills for interpretable reasoning, existing approaches face three critical limitations: skill libraries become stale as real-world scenarios dynamically evolve, skill descriptions often suffer from a semantic execution gap where LLMs deviate from intended expert logic, and growing skill collections exceed context budgets causing critical skill omission. To address these challenges, we propose Knowledge-Data Dual-Driven Automated Skill Optimization (KDASO), which continuously evolves a skill library through neighborhood-conditioned induction from recent adjudication data, execution-aligned refinement based on LLM trajectory analysis, and context-budgeted routing for scalable skill selection. Experiments across ride-hailing adjudication, legal liability determination, and logical reasoning judgment demonstrate that KDASO significantly outperforms current general-purpose and domain-specific baselines. KDASO further achieves 8.96% improvement when transferring skills to concept-similar scenarios and maintains consistent performance gains through continuous optimization, underscoring the high transferability and stability of generated skills. 

---
# Formalizing inverse problems in the age of AI: possibilities, pitfalls, and an invitation 

**Authors**: Christian Clason, Tapio Helin, Medet Nursultanov, Lauri Oksanen  

**Link**: [PDF](https://arxiv.org/pdf/2611.06962)  

**Abstract**: Proof assistants such as Lean check mathematical proofs down to the axioms, and AI agents have recently made writing such proofs cheap. We describe AutoIP, a Lean library in which classical linear regularization theory, together with the functional analysis it rests on, has been autoformalized by AI agents. A machine-checked proof certifies only the formal statement it proves, so the remaining human task is to confirm that each formal statement says what was intended. We organize the library around this task: a reviewer compares the formal statements with their natural-language counterparts and never needs to read a proof. Our experience is that the risks lie in the statements rather than in the proofs, and that simple automated tests on the statements can expose problems even in the established literature. We illustrate this with a textbook theorem whose hypothesis is more restrictive than asserted. We invite the inverse problems community to extend and build on the library. 

---
# XStack: A Benchmark for ML Repo Migration 

**Authors**: Linrong Cai, Taiming Lu, Haozhe Chen, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.08472)  

**Abstract**: ML code should run where researchers choose, not where it was first written. Today, machine learning code mainly lives in two ecosystems, PyTorch on NVIDIA GPUs and JAX on Google TPUs. Migrating repositories between them means building and testing across frameworks and hardware. This migration is (1) a practical need, since it unlocks different hardware for researchers, and (2) a natural benchmark: it involves different frameworks and hardware, covers whole repositories, and is long-horizon and challenging. We introduce XStack, a benchmark that evaluates agents on migrating ML repositories between these two ecosystems. Each agent receives a PyTorch repository and a clean TPU machine and must deliver a JAX repository. Its output passes if it meets our grading criteria, based on the source repository's published result, authenticated and calibrated by us. Migration by agents remains unreliable: Claude Opus 5 passes 75%, GPT-5.6-sol 45%, and every other agent at most 40%. XStack is a step toward freeing ML code from its ecosystem. We release XStack to make progress toward reliable migration across frameworks and hardware measurable. 

---
# Same Behavior, Different Mechanism: What Finite Experiments Can Certify About Integrated Information 

**Authors**: Linqian Wu, Diaoxuan He  

**Link**: [PDF](https://arxiv.org/pdf/2611.08469)  

**Abstract**: Interpretability, AI auditing and assessments of machine consciousness increasingly rest on mechanism-level quantities: functionals of a system's causal transition law that describe how it computes, not what it outputs. Experiments see that law only through finite, partial and noisy interventions. We ask when such a quantity is identifiable from an experiment's access, continuous in the mechanism and certifiable from finite data, and answer exactly for a deliberately demanding case - the integrated information $\varphi$ of integrated information theory - in stochastic networks of four and five units with known ground truth. Behavior can be silent: a recurrent and a feed-forward network with identical input-output laws at their response delay carry 0.28 and 0 bits, yet internal recordings separate them within 4-53 trials. Partial access can be just as blind: in our examples, units left out of a one-step recording impose precision floors of 0.30-1.02 bits that no sample size lowers, and in all-but-self parity networks, recordings of any length see several hidden units only through the product of their fidelities - a symmetry that one intervention broke in exact checks. The measure has sharp edges: because $\varphi$ selects a partition, and the cause and effect it specifies, by one criterion and reports another, infinitesimal changes of mechanism that cross an edge change it by 25-100% at 99% of the edges we found in random networks, and about 1% of surrogate models that fit interventional data almost perfectly misreport it by more than 20%. One-sided claims survive: the same data certify a lower bound on both sides of an edge, and we prove $\varphi>1.004$ bits throughout a 64-dimensional region of mechanisms and certify it from about $10^7$ controlled transitions with false-certificate probability at most $\alpha$. Measurability is thus a design question with computable answers. 

---
# Towards interpreting any residual stream direction by scaling activation inversion 

**Authors**: Celeste De Schamphelaere, Juan J. Vazquez, Ari Spiesberger, Tomáš Gavenčiak, Gavin Leech, Euan Ong, Jiahai Feng  

**Link**: [PDF](https://arxiv.org/pdf/2611.08465)  

**Abstract**: To interpret a direction in a language model's residual stream, we can look for text that strongly activates it. But when such text is found by scanning a corpus, it is limited to what that corpus contains. We develop Maximally-Activating Example Metamodels (MAEMs), language models which map residual stream directions from a subject model to text exemplars that strongly activate those directions. Our objective is to maximise the cosine similarity of a supplied residual stream direction with the activations of the text generated by the MAEM. This label-free RL objective helps to scale MAEMs on diverse sources of residual stream directions. Empirically, we find the resultant metamodel produces informative exemplars: a Qwen3.6-27B MAEM inverts held-out activations better than corpus search or a natural-language autoencoder (NLA), beats corpus search (at 10M tokens) on a majority of held-out languages/domains, and activates 89% of held-out SAE features. It also identifies concepts behind steering vectors about as well as NLAs, recovers the payloads of LoRA backdoors more often than corpus search, and can even verbalise some model workspace inner monologue. Unlike explanations produced by other metamodels (e.g. NLAs), MAEM exemplars can be verified with a forward pass on the subject model, making the MAEM objective a grounded alternative for scaling the training of metamodels. 

---
# Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes 

**Authors**: Vladimir Beskorovainyi  

**Link**: [PDF](https://arxiv.org/pdf/2611.08439)  

**Abstract**: Typed-decision models score options that the caller supplies at request time, so their accuracy depends on label wording that the model's developer does not control. We compare two releases of a 307M-parameter multilingual cross-encoder, Ru-Jev, trained on thirteen Russian-language tasks; the later release adds an augmentation that presents labels in several surface forms. Holding questions fixed and varying only how eighteen routing labels are worded, we evaluate ten label vocabularies, produced by three separate processes and checked for overlap with the training data, with five candidates per question and, for eight of them, also with eighteen. On held-out vocabularies made of ordinary or borrowed words (English-derived jargon, office slang, department names), the augmented release is more accurate in every setting, by up to 0.24. Eleven of these fourteen gains survive a Holm correction (slang with eighteen candidates and descriptive sentences with five do not), and for jargon on all labels the label-level interval excludes zero in both settings. On abbreviations and opaque codes it gives no usable gain: on abbreviations it is not better, and worse on the clean slice, and on opaque codes both releases stay at or below chance once labels related to training options are removed. In most cells where accuracy rises, the risk-coverage area falls, but with eighteen candidates confident errors rise on jargon and slang. On two datasets neither release saw, accuracy moves by -8.4 and +5.6 points. We also report four evaluation hazards from comparing the two releases. 

---
# Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety 

**Authors**: Fazl Barez, Tung-Yu Wu, Hunar Batra, Gabriele Dominici, Florent Draye, Yanai Elazar, Damiano Fornasiere, Antía García, Abir Harrasse, Riley Harris, Sara Hooker, Michael Lan, Belinda Z. Li, Aaron Maiwald, Narmeen Fatimah Oozeer, Naomi Saphra, Or David Shafran, Adi Simhi, Robert Trager, Shriyash Kaustubh Upadhyay, Ziyu Yao, Yu Zhao, Shay B. Cohen, Mor Geva, Yoshua Bengio, Stella Biderman, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2611.08438)  

**Abstract**: Neural networks are likely to be among the most consequential artefacts in human history. As these systems become more capable and autonomous, their potential to cause societal harm also grows, making AI safety one of the defining challenges of our time. Yet we continue to build increasingly powerful systems without fully understanding how they work, how they will behave in unfamiliar situations, or how to reliably control them. Mechanistic interpretability offers a path towards understanding these systems by studying their internal representations and computations, with the aim of evaluating, predicting, and controlling behaviour. We survey mechanistic interpretability for AI safety in order to answer the question: Does understanding how models work actually make them safer? We place mechanistic interpretability in the broader context of interpretability research and compare its mechanistic evidence with behavioural evaluation. We introduce the foundations of mechanistic interpretability, including features, polysemanticity, transformer computation, and circuits, and survey methods for feature extraction and analysis, circuit discovery and causal verification, explaining intermediate representations in natural language, learning dynamics, and training-data attribution. Beyond surveying existing methods, we propose a unified framework for evaluating mechanistic interpretability claims and characterise six research orientations by their goals, assumptions, and trade-offs. Finally, we examine how mechanistic understanding can support AI safety, including knowledge editing, steering, and monitoring, and assess its limitations, failure modes, open problems, and relationship to behavioural safety measures. We also release a repository of code notebooks with worked examples that reproduce key findings from surveyed methods, allowing readers to build intuition through hands-on experimentation. 

---
# MedGraph: Maintaining Patient State with Graph Memory for Long-Term Medical Dialogue 

**Authors**: Donghua Zhang, Zizheng Wang, Ze Dong, Yiding Feng, Siyu Qian  

**Link**: [PDF](https://arxiv.org/pdf/2611.08433)  

**Abstract**: LLM-based medical assistants that follow a patient across visits need a memory that keeps the patient's clinical state correct as it changes. We examine the errors of five memory systems on longitudinal medical dialogue and find three recurring causes: the retrieved evidence gives an outdated state, includes another person's health facts, or lacks a safety fact the answer depends on. We present MedGraph, a graph memory built for these three cases. Construction extracts dated facts from each session, attributes each fact to the person it describes, and links medication, diagnosis, and measurement records into chains that keep earlier states. Retrieval ranks the patient's chains as units, reads each chain at the time the question asks about, and adds the original dialogue text; a patient profile lists every recorded safety fact with every question. On MedMemoryBench, MedGraph has the highest overall accuracy with both a GPT and a Qwen backbone, 10.5 and 10.4 points above the strongest of five baselines, with efficient memory construction and answering in the GPT setting. 

---
# Failure Prediction and Classification in Vision-Language-Action Models Via Calibrated Latent Probing 

**Authors**: Michal P. Podolinsky, Yunhao Yang, Daniel Little, Hong Lu, Rohan Siva, Olin Goudey, Neel P. Bhatt, Matthias Scheutz, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2611.08425)  

**Abstract**: Vision-Language-Action models (VLAs) enable general-purpose embodied intelligence, but their performance is brittle under distribution shift, where unanticipated failures can damage equipment or endanger people. Deploying VLAs in real-world systems therefore requires a runtime monitor that is timely, calibrated, and informative while being efficient enough to run on the edge. Existing methods fail to fulfill these requirements. We present VLA-Tug, a lightweight probe over the latent layers of a VLA backbone that improves early failure detection and bounds the missed-detection and false-alarm rates, with minimal inference overhead. We show that supervising on the pre-failure segment of a trajectory increases early failure detection at matched false-positive rates by up to 30% over the strongest learned baseline. Under shift to unseen tasks, the bounds on missed-detection and false-alarm rates remain valid in up to 86% of runs, versus at most 21% for baselines. Unlike existing methods, VLA-Tug predicts the failure mode to enable corrective actions, raising top-3 accuracy on unseen tasks by up to 44% over a majority baseline. VLA-Tug enables real-time monitoring where the well-timed and informative probe supports lightweight steering that improves task success. We release a dataset of 3,180 annotated manipulation episodes to support future work on latent failure detection in VLAs at this https URL. 

---
# Hybrid Machine Learning for Imbalanced Landing-Phase Safety Event Prediction from Flight Data 

**Authors**: Taha Asghar, Bilal Mufti, Syed Qasim Zaheer, Luqman Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2611.08386)  

**Abstract**: Landing-phase safety-event prediction requires the detection of rare abnormal approaches within strongly imbalanced multivariate flight data. Capturing coupled temporal behavior while distinguishing event-specific signatures across airport operating contexts remains a central modeling challenge. This study proposes an interpretable hybrid feature-fusion framework for multiclass landing-phase safety-event prediction. A long short-term memory (LSTM) autoencoder trained on nominal flight windows characterizes deviations from routine approach behavior. Its continuous reconstruction-error descriptor is fused with parameter-specific statistical flight-history features and supplied to a class-weighted XGBoost classifier to distinguish Normal, Speed High, Path High, and Flaps Late windows. The framework is evaluated using 53,750 NASA DASHlink approach windows from three arrival airports. Pooled and airport-specific learning strategies are compared alongside component-level benchmarks, repeated pipeline retraining, and feature-attribution analysis. The results demonstrate strong class-balanced detection, with mean macro recall above 0.90 across all airports under repeated partitioning and retraining. Component comparisons establish the importance of class weighting and show that continuous feature fusion outperforms reconstruction-error-only and hard-gate alternatives. Airport-level comparisons reveal benefits from local specialization at the two airports with larger datasets and from pooled training at the airport with the smallest dataset. Feature attributions associate predictions with operationally relevant vertical-path, configuration, speed, and approach-context variables. The framework provides an interpretable approach to rare-event detection and practical guidance for adapting flight-data monitoring models to airport-level data availability. 

---
# FedReason: Large Language Models as Weight-Space Meta-Aggregators for Heterogeneous Federated Learning 

**Authors**: Ali Mohammadi Ruzbahani, Abbas Yazdinejad, Hadis Karimipour  

**Link**: [PDF](https://arxiv.org/pdf/2611.08383)  

**Abstract**: Federated learning servers combine client models with a fixed aggregation rule, and each rule in common use is designed for one kind of heterogeneity. We study whether a large language model (LLM) that inspects the client models can choose the rule itself, without any training on the server. In FedReason, the LLM is called every ten rounds. It analyses the weights by running code in a restricted server-side sandbox, sees only the printed results, may score a few candidates on a 500-image proxy set, and submits a short Python aggregation program. The server applies the program until the next call and falls back to FedAvg whenever its output is invalid. Full weight tensors never leave the server. We evaluated FedReason with Claude Opus 5.5 on CIFAR-10 and CIFAR-100 with ResNet-18 under Dirichlet label skew, width-heterogeneous sub-models and Byzantine clients, in 168 runs against 14 baselines. Across 12 paired label-skew runs it beat FedAvg by 1.26 percentage points on average (10 wins), and a control that spends the same per-round proxy budget on choosing among standard rules by 1.08 points. It was the strongest server-side rule on CIFAR-100 but did not beat SCAFFOLD, which changes client training. With three of ten clients malicious, it excluded all sign-flipping clients from the first round and stayed about one point below FLTrust without training a root model. Without proxy feedback, accuracy fell from 73.8% to 38.8% in one seed. A 60-round run cost 0.79 USD in API fees on average. 

---
# MOMASPLAT: Dynamic Novel-View Demonstration Augmentation for Mobile Manipulation 

**Authors**: Sateesh Kumar, Arnav Balaji, Arpit Bahety, Skand Peri, Georgios Pavlakos, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2611.08372)  

**Abstract**: Visuomotor policies trained through imitation learning are highly sensitive to changes in camera viewpoint. This is particularly problematic for mobile manipulation (MoMa), where robots rarely execute manipulation from the exact same base pose used during demonstration collection due to navigation uncertainty, localization drift, and environmental constraints. Recent approaches use novel-view synthesis (NVS) to augment demonstrations with unseen viewpoints, but they assume static scenes and therefore fail to re-render the manipulated object during the post-contact interaction phase, leaving the most critical parts of the demonstration unaugmented. We present MOMASPLAT, a novel-view demonstration augmentation framework for mobile manipulation that synthesizes full manipulation trajectories from unseen robot base poses, including dynamic object interactions. Our key insight is that standard MoMa data collection naturally provides the multiview observations needed to reconstruct both the static scene and the manipulated object in 3D. MOMASPLAT combines 3D Gaussian Splatting, feed-forward 3D foundation models, and kinematics-based action retargeting to separately reconstruct the scene and manipulated object, render them from novel viewpoints, and generate geometrically consistent synthetic demonstrations paired with transformed actions. We evaluate MOMASPLAT on five real-world mobile manipulation tasks involving both rigid and articulated objects across three scenes, and validate the fidelity of its synthesized demonstrations against ground truth in simulation. MOMASPLAT substantially improves robustness to unseen base poses, more than doubling the average success rate of the strongest baseline and maintaining non-trivial success at large base displacements where all baselines fail, while reducing the need for largescale real-world data collection. 

---
# VisionSIM: Beyond Idealized Sensors - A Multi-Modal Simulator for Robot Perception 

**Authors**: Sacha Jungerman, Kaustubh Sadekar, Avery Gump, Sriram Narayanan, Shantanu Gupta, Bhuyashi Deka, Srinivasa Narasimhan, Atul Ingle, Mohit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.08368)  

**Abstract**: Robotic perception increasingly relies on heterogeneous sensor suites combining cameras with emerging modalities (event, SPAD, iToF/dToF, and thermal). Yet existing simulators idealize these sensors, hiding critical modality-specific failure modes that limit real-world deployment. We present VisionSIM, a unified, physically grounded simulation platform that elevates sensor realism to a first-class concern by modeling the raw physical measurement process for diverse conventional and emerging modalities. We expose fine-grained control over physical sensor parameters, reproducing failure modes overlooked by general-purpose simulators: pileup and glare for LiDAR; phase wrapping, saturation, and motion artifacts for iToF; refractory drops for event cameras; and history-dependent emission for thermal cameras. This enables principled cross-modality comparison on identical scenes and trajectories. We release the platform and a large indoor dataset with synchronized raw measurements and rich ground-truth annotations. 

---
# Measured Coupling for Parallel Decoding in Diffusion Language Models 

**Authors**: Tal Oved, Lior Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2611.08307)  

**Abstract**: Diffusion language models can write many tokens per step, but must choose which tokens are safe to commit together. Mean-field decoding penalizes pairs of positions whose predictions look alike. We replace this proxy with the coupling itself: how much committing one token changes the model's confidence at another. Cheap probes inside the main forward pass measure it, with no training or extra model, and it plugs into the same mean-field rule. On three models and four benchmarks, our method needs far fewer steps than mean-field decoding and is faster in most settings, also with a key-value cache, at an average accuracy cost of 1.1 points. Shuffling the couplings or replacing them with a uniform bonus loses this gain. 

---
# AutoCompress: Automated Hardware-Guided Model Compression 

**Authors**: Sheldon Liang, Ali Taghibakhshi, Saurav Muralidharan, Sharath Turuvekere Sreenivas, Ashwath Aithal, Pavlo Molchanov, Nima Tajbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2611.08290)  

**Abstract**: LLM pruning must preserve quality while meeting hardware- and workload-specific constraints on latency, throughput, and memory. Existing pruning workflows often target parameter counts, requiring practitioners to manually select model sizes that meet latency, throughput, and GPU memory limits. However, parameter count alone does not predict these metrics, which depend on the model architecture, workload, hardware, and serving engine. We introduce AutoCompress, a hardware-guided compression framework that directly incorporates such constraints into differentiable architecture search. AutoCompress jointly optimizes global and per-layer structure to produce heterogeneous architectures, guided by serving-engine cost models validated against end-to-end latency, throughput, and memory measurements. We evaluate AutoCompress on long-context local coding, batched datacenter serving, and latency-sensitive in-game interaction. Compared to the state-of-the-art Minitron pruning framework, AutoCompress improves average accuracy across eight benchmarks by 6.57, 2.35, and 0.57 percentage points on RTX 5070, RTX 5080, and H100, respectively. AutoCompress also compresses Nemotron 3.5 Super from 120B to 74B parameters for single-B200 BF16 serving at 128K context, retaining 95% of its mean benchmark score. 

---
# Aligning Policy Encoders with a Shape Teacher 

**Authors**: Madelon Bernardy, Mohammad Mahdi Derakhshani, David M Knigge, Riccardo Valperga, Artem Moskalev  

**Link**: [PDF](https://arxiv.org/pdf/2611.08275)  

**Abstract**: 3D-aware representations can improve robotic manipulation by providing geometric information that is robust to changes in viewpoint, object position, and occlusion. However, using 3D inputs at deployment requires additional sensing and processing, making image-based policies more practical. We ask whether an image-based policy can acquire 3D shape information during training while retaining a purely image-based interface at deployment. We propose shape-teacher alignment, a training-time mechanism that uses a pretrained 3D shape representation as an alignment target for an image-based policy encoder. Specifically, we use Point-MAE features from individual objects as targets for corresponding image patches, transferring geometric structure to the policy without changing its deployment inputs or architecture. On LIBERO, shape-teacher alignment produces image representations that are more structured around object shape and corresponding parts, while substantially accelerating policy convergence and improving success rates. For pretrained policies, convergence is accelerated by 4.7x and 5.6x on the Object and Spatial suites, respectively, without requiring the shape teacher or depth input at deployment. 

---
# Accelerating Safe Reinforcement Learning with Massive Parallelism 

**Authors**: Joonyoung Lim, Younghwan Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2611.08273)  

**Abstract**: We investigate the problem of scaling safe reinforcement learning (safe RL) to massively parallel training regimes, where a large number of environments are exe- cuted with short rollout horizons. Although this setting improves data throughput and wall-clock efficiency, it creates a mismatch with constrained Markov decision processes (CMDPs), whose safety constraints are defined over full-episode cost returns. We address this mismatch by analyzing staggered environment resets as a phase-mixture distribution over short trajectory segments. This analysis charac- terizes the bias induced by phase coverage and motivates a phase-aggregated cost estimator that reconstructs episode-level cost estimates from short rollout batches. We further incorporate finite-sample uncertainty and distribution mismatch into a constraint-tightening framework, yielding a scalable Safe RL method compatible with on-policy optimization. Using an MJX-based implementation of Safety Gym, our experiments show that the proposed method achieves performance comparable to non-massively parallel implementations while substantially reducing wall-clock training time. 

---
# DART: Domain-Agnostic Residual Transfer for Generalist Anomaly Detection 

**Authors**: Muhammad Aqeel, Maham Nazir, Marco Cristani, Francesco Setti  

**Link**: [PDF](https://arxiv.org/pdf/2611.08271)  

**Abstract**: Generalist Anomaly Detection (GAD) aims to detect anomalies in unseen domains using only a few reference samples. Recent normal-abnormal guided methods improve this setting by exploiting abnormal references, but their residuals often mix transferable anomaly cues with domain-specific appearance factors such as texture, geometry, and acquisition conditions. This residual domain entanglement limits cross-domain transfer and may produce spurious activations on novel domains. We propose DART, a plug-in framework for normal-abnormal guided GAD that learns to separate residuals into anomaly-relevant and domain-specific components. DART combines a Residual Disentanglement Module, which preserves anomaly-discriminative information while suppressing domain information, with Cross-Domain Meta-Training, which forms episodes across source domains to make the separation effective. The method is architecture-agnostic and can be inserted between residual computation and the detection head of existing GAD pipelines. Experiments on MVTec AD, VisA, and BraTS show consistent improvements over multiple base methods, with the largest gains under stronger domain shifts. this https URL 

---
# Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations 

**Authors**: Xiaobo Chen, Genliang Zhu, Ziyuan Wang, Chu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.08269)  

**Abstract**: After a runtime or API migration, an LLM-repaired procedure can run cleanly yet return wrong results, and a release check on typical test inputs ships it. We report a preregistered study of 73 tasks covering 60 documented changes in CPython, this http URL and SQLite. Every candidate is executed on the real runtimes against withheld contracts. Over the complete repair space, the all-assertions gate on typical inputs released 0.410 of wrong repairs. One edge input written from the release note cut this to 0.065, and to 0.061 on 15 unseen clusters. Three inputs from a blind GPT-5.6-luna writer that never saw the program reached 0.070. Two further frozen stages enlarged the menu to 1491 fragments and then added 233 hard distractors that match the legacy output on typical inputs. The edge input (0.058) and the frontier-model writer (0.058) passed their preregistered tests. Yet 0.168 and 0.153 of the hard wrong programs still passed. Two local-model writers failed their preregistered tests. Across four model families, the edge input cut wrong releases per attempt in all three open-weight families on all three menus, for example from 0.233 to 0.057, without lowering correct completion. Self-tests helped less. A blind AI audit flagged 34 contracts as stricter than their requirements. Dropping the disputed checks changed 0 verdicts, but class-label disputes remain open. No model-written code was executed. Artifacts are available from the authors on request. 

---
# Scaling Neural Motor Decoding via Decoupled Behavioral Pretraining 

**Authors**: Divyansha Lachi, Vinam Arora, Shivashriganesh P. Mahato, Alexandre Andre, Eva L. Dyer  

**Link**: [PDF](https://arxiv.org/pdf/2611.08263)  

**Abstract**: Recent neural decoding approaches achieve strong performance by training on large collections of paired neural-behavioral recordings, but such datasets are expensive, invasive, and difficult to scale. In contrast, behavioral data is abundant and easy to collect, raising the question of whether neural decoding can benefit from decoupling behavioral and neural representation learning. We introduce BEEMO, a flexible multi-session framework that enables training with arbitrary mixtures of paired neural-behavioral recordings and unpaired behavioral data. A behavior decoder learns movement dynamics directly from behavioral trajectories, while neural activity is incorporated through cross-attention layers when available, allowing behavioral and neural supervision to scale independently. Across multiple intracortical motor datasets and tasks in nonhuman primates, we show that incorporating unpaired behavioral data improves decoding performance, particularly when transferring to new tasks, and that adding even a small amount of behavioral data from the target task further improves transfer. By decoupling neural and behavioral data, we unlock a new dimension for scaling: unpaired behavioral data, which is substantially cheaper and easier to collect than paired neural recordings, can be used directly to improve decoding performance. We further show that behavior-only pretraining, without any neural pretraining, achieves performance comparable to models pretrained on substantially larger neural datasets. Together, our results suggest that scaling behavioral data offers a practical and cost-effective path toward neural decoding models that generalize across subjects and tasks. Code is available at this https URL. 

---
# Knowing When to Teach: Learning from Teachable Prefixes in Multilingual Reasoning Distillation 

**Authors**: Jaewon Cho, Minseo Yoon, Jinyoung Park, Jinwoo Seo, Jaewon Chu, Alice Oh, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.08179)  

**Abstract**: On-Policy Self-Distillation (OPSD) improves reasoning by training a student on its own reasoning, using the same model conditioned on reference solutions as a teacher. Applying OPSD to multilingual reasoning requires the teacher to provide useful feedback across languages, even when the student makes mistakes in intermediate steps. We observe a phenomenon we call Teacher Drift: as these mistakes accumulate, the teacher may follow the student's incorrect steps instead of correcting them. Our analysis shows that this shift tends to occur earlier in lower-resource languages. To account for these differences across languages, we propose Teachable On-Policy Self-Distillation (TOPSD), which determines where and whether teacher supervision should be applied to each reasoning trajectory. TOPSD uses teacher-student likelihood to estimate where drift begins and select a prefix for supervision. It then verifies whether the teacher can recover the correct answer from that prefix, applying distillation only to prefixes that pass verification. Experiments on multilingual mathematical reasoning across ten languages and three model scales show that TOPSD consistently outperforms conventional OPSD, particularly in low- and medium-resource languages. 

---
# Interpretable Multimodal MRI-Clinical Fusion for Functional Outcome Prediction After Acute Ischemic Stroke 

**Authors**: Sina Raeisadigh, Henning Müller, Abderrahmane Hedjoudje  

**Link**: [PDF](https://arxiv.org/pdf/2611.08154)  

**Abstract**: This study evaluates the added prognostic value of day-1 multimodal MRI for predicting three-month functional outcome after acute ischemic stroke (AIS). Sixty-six AIS patients with ADC, DWI, FLAIR, and clinical data were analyzed. Three-dimensional pretrained ResNet embeddings were extracted from each MRI modality, reduced using fold-isolated principal component analysis, and combined with structured clinical variables and ADC-derived lesion biomarkers. Linear support vector machines were evaluated using eight-fold stratified group cross-validation across MRI modality combinations. Clinical-only modeling achieved strong performance (AUC = $0.899 \pm 0.151$), while MRI-only models showed limited discrimination (AUC $\approx$ $0.67$-$0.73$). Among multimodal configurations, FLAIR-clinical fusion achieved the highest mean cross-validated performance (AUC = $0.948 \pm 0.061$), followed by ADC-clinical fusion (AUC = $0.941 \pm 0.094$). Full ADC-DWI-FLAIR fusion did not consistently outperform single-modality clinical fusion. Feature-importance analysis identified age, NIHSS subscores, ADC lesion burden, and learned FLAIR representations as key contributors, while qualitative FLAIR activation maps showed responses over lesion-like regions. These findings suggest that compact day-1 FLAIR and ADC representations provide complementary prognostic information beyond clinical variables and support interpretable, data-efficient outcome prediction in AIS. 

---
# Who Can Evaluate AI Independently? A Framework for Analysing Evaluation Infrastructure 

**Authors**: Sumaya Nur Adan, Joanna Wiaterek, Nourhan Abdelrhim, Sydney Reis, Robert F. Trager, Benjamin Rosman, Marina Jirotka  

**Link**: [PDF](https://arxiv.org/pdf/2611.08134)  

**Abstract**: AI evaluation is increasingly central to AI safety and governance, yet taking part in an evaluation does not by itself confer authority over its design, its execution or the resources it depends on. This paper examines how infrastructure arrangements distribute that authority, and so shape who can evaluate AI systems independently. Drawing on Science and Technology Studies, political economy and applied evaluation standards, we introduce PAAI (Power, Access, Adequacy, Independence), a framework for examining who controls evaluation infrastructure, the terms on which it is available, whether it is adequate to the evaluation's purpose and what autonomy it supports. We apply PAAI to twelve African AI evaluation projects (2022-2025) and to the documented roles of all ten member institutes in the three published joint testing exercises of the international network of AI Safety Institutes. We identify three recurring configurations: extractive collaboration, dependent innovation and constrained independence. African actors often contribute data, language expertise and task design while external institutions retain authority over infrastructure and evaluation frameworks; design authority without infrastructure authority never yielded more than partial independence. The joint testing exercises show a similar division among mostly high-income members: the institutes that led each exercise also ran the models, while Kenya, the network's only African member, moved from annotation to task design with no documented model execution. Participation in evaluation does not by itself produce evaluative authority. For evaluation to serve as a credible governance mechanism, access must come with control over priorities, methods, publication and continuity. 

---
# Gradient Boosting for Efficient Rank Discovery in Tensor Network Kernel Machines 

**Authors**: Albert Saiapin, Kim Batselier  

**Link**: [PDF](https://arxiv.org/pdf/2611.08093)  

**Abstract**: Tensor network kernel machines (TNKMs) combine nonlinear features with low-rank tensor representations to model complex relationships efficiently. However, selecting an appropriate tensor rank remains challenging, as conventional rank-search methods require training multiple candidate models, increasing computational costs. In this work, we focus on a canonical polyadic (CP) tensor network parameterization and propose GB-CPKM, a gradient boosting framework that builds a higher-capacity predictor by sequentially adding fixed-rank learners. This approach provides a new perspective on rank selection by incrementally increasing model capacity, with ensemble size selected based on validation performance rather than through explicit rank search. Experiments on real-world regression benchmarks show that GB-CPKM trains up to 6x faster and uses up to 2x less peak memory than a conventional rank-selection method based on cross-validation, while achieving comparable predictive performance. These results highlight the potential of gradient boosting to improve the computational efficiency and scalability of tensor-network learning. 

---
# Value or Attention: Investigating Layer-wise Relevance Propagation for Graph Attention Networks 

**Authors**: Ken Schreiber, René P. Klausen, Sebastian Lapuschkin, Wojciech Samek, Jonas Naujoks  

**Link**: [PDF](https://arxiv.org/pdf/2611.08088)  

**Abstract**: Graph Attention Networks (GATs) are increasingly used in domains where scientists need to read a model's reasoning, from medicine to materials science, yet the routing they learn remains opaque to most explainers. Layer-wise Relevance Propagation (LRP) follows the model's computation, but in an attention layer the relevance reaching a node is the sum of two mechanically different quantities: its content relevance through the value path and its routing relevance through the attention path. Existing rules either drop one or add them into a single number. We write a single product rule with free weights on the two factors that recovers current approaches as special cases. The value path measures what content a node supplied against a zero baseline; the attention path measures how the node shaped what was passed relative to its neighborhood, and its sign records whether the node's features moved the gating the way the prediction demanded. On a ground-truth classification task, we show that the same node receives negative, positive and near-zero relevance under three settings of the rule. The separated relevance maps yield a hypothesis about the model: it gains by not reading that node. We give reading rules for both paths and argue that they should be interpreted separately. 

---
# Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids? Exploring "Weight-Space Programming" 

**Authors**: Nazmus Sakib Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2611.08070)  

**Abstract**: Large language models (LLMs) have recently been reported to read and write the raw weights of neural networks directly, a capability termed weight-space programming. We ask whether the reading side of this capability reflects reasoning about weight flows or a combination of local arithmetic at the network's edges and memorized priors over public datasets. We test Claude Opus 5.5 on small convolutional networks of about 125,000 parameters trained on five image datasets, providing every weight as plain text with no tools, no forward passes, and no dataset or class names. The model performs accurate local calculations: batch-normalization statistics lead it to infer about three optimizer steps on networks stopped after three updates, and output-layer sign patterns recover the vehicle/animal split of CIFAR-10. The sessions report reading the first and last layers closely and sampling the middle, and several claims built on partial tracing fail verification, although we did not run a direct forward-pass test. Semantic judgments lean on familiar datasets: the model names CIFAR-10 for CIFAR-10 networks but also for all three fast-food networks, relabels KMNIST as MNIST, reads a 99.57%-accurate SAT-6 network as CIFAR-10 animal photos, and calls "frog" the output favored by a feature that fires most on trees. Permuting CIFAR-10's labels preserves the output split at the new indices, but every class pair the model interprets is then wrong, indicating reliance on remembered label order. These are exploratory results from mostly single runs. We conclude that zero-shot weight reading by this model is better described as local arithmetic plus dataset priors than as weight-flow tracing, and we provide a protocol for auditing future weight-space claims. 

---
# NutriVisRAG: Multimodal Exemplar Grounding for Nutrient Estimation with Vision-Language Models 

**Authors**: Lubnaa Abdur Rahman, Ioannis Papathanail, Maria Panagiotou, Nefeli Dimitropoulou, Margret Harðardóttir, Elisabeth J. den Brok, Bastiaan E. de Galan, Stavroula Georgia Mougiakakou  

**Link**: [PDF](https://arxiv.org/pdf/2611.08054)  

**Abstract**: Diet is a major determinant of health, yet accurate dietary assessment from free-living meal images remains challenging. Existing retrieval-based Vision-Language Model (VLM) approaches to nutrient estimation typically rely on textual food representations, while, to our knowledge, direct grounding on visually similar annotated meals has not been investigated. We present NutriVisRAG, a retrieval-augmented generation framework that uses a vision encoder to retrieve similar meals with known nutritional annotations and provide them as multimodal in-context exemplars for VLM nutrient estimation at inference time, without task-specific fine-tuning. We evaluate NutriVisRAG across four datasets, four VLMs, three retrieval encoders, and two prompting strategies, including free-living meals from adults with insulin-treated diabetes, where carbohydrate estimates are compared against two dietitians' annotations. Across all 96 configurations (4 datasets x 4 VLMs x 3 retrieval encoders x 2 prompting strategies), NutriVisRAG reduced the primary estimation error in 56, with improvements more frequent under whole-meal (35/48) than per-item prompting (21/48). Across the evaluated configurations, the lowest NutriVisRAG error was below the best Vanilla error in 13 of 16 VLM--dataset pairs. For GPT-5.4-mini, under whole-meal prompting, the best-performing retrieval encoder for each dataset reduced the primary error on all four datasets relative to Vanilla, with reductions of 1.7--35.7%. Overall, our results show that grounding in multimodal exemplars comprising images and respective annotations can improve nutrient estimation at inference time without task-specific fine-tuning, although the benefits remain configuration-dependent. 

---
# Technohybridism: Recursive Semantic Normalization in AI-Mediated Societies 

**Authors**: Charalampos M. Liapis, Epameinondas Panagopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2611.07984)  

**Abstract**: Language-generating AI systems are no longer only objects of linguistic interaction: their outputs increasingly re-enter the communicative and institutional practices from which their training material was drawn. Drawing on social theory, social epistemology, and the philosophy of language, this article develops a sociotechnical account of that re-entry. We introduce recursive semantic normalization: the process through which machine-generated formulations, once recurrently taken up, evaluated, reused, and institutionally retained, participate in the reproduction and transformation of semantic regularities. Under the heading of technohybridism, we argue that the relevant unit of analysis is the hybrid semantic system, a recursively coupled configuration of human practices, computational generation, and institutional uptake. First, we distinguish the concept from neighbouring accounts, including model collapse, looping effects, machine habitus, machine culture, and the epistemology of AI testimony. Second, we develop an analytical model that separates generation, selection, uptake, and persistence and distinguishes machine-attributable change from change that would have occurred anyway. Third, we analyse uptake as routinized, institutionally diffused, and socially stratified practice and state conditions under which machine-derived regularities become norms. Fourth, we identify social-epistemic consequences: the laundering of provenance that weakens epistemic vigilance, the erosion of the cognitive diversity on which collective inquiry depends, and the narrowing of shared hermeneutical resources. An evidential ladder for empirical research follows. The account attributes no normative authority to machine outputs; what changes is the field within which human and institutional uptake operates. 

---
# tok: a textual concrete syntax of BPMN and DMN for humans and AI agents 

**Authors**: Marek Deja  

**Link**: [PDF](https://arxiv.org/pdf/2611.07945)  

**Abstract**: tok is a plain-text notation for Business Process Model and Notation (BPMN) 2.0.2 process models and Decision Model and Notation (DMN) 1.5 decisions. One engine checks a tok document against both standards, compiles the document to XML with an automatic diagram, and serves language-model applications through the Model Context Protocol. The paper classifies the notation as an alternative concrete syntax of the BPMN and DMN metamodels and reports one study. A tok model takes 2.2 times fewer tokens than the smallest BPMN XML of 893 files exported by 37 tools and 7.7 times fewer than the files as exported. The checker signals 98.2% of 1,411 injected faults, against 85.8% for the XML Schema with bpmnlint. Four language models new to tok write valid tok from a one-page reference for 85.7 to 100% of 87 process descriptions from two datasets and valid BPMN XML for every description, and the strongest model writes tok with 25% and 19% fewer tokens than BPMN XML. As agents with a lean skill and one checker tool, the models write valid tok for 97.6 to 100% of the descriptions, the strongest model spends 21% and 14% fewer tokens on tok than on BPMN XML, and a skill which sends the model to read the reference through a tool costs 1.7 to 2.8 times the tokens of the lean skill. Of 47 event logs simulated from public models, 44 fit their source models, and tok imports every model discovered from the logs. 

---
# CardioLM: A multilingual suite of small language models for the cardiology domain 

**Authors**: Manuela-Daniela Danu, Miguel Rodríguez-Ortega, Alberto Becerra-Tome, Pablo Arancibia, Fernando Gallego-Donoso, Lorenzo Pratesi, Xi Bai, Andrei-Damian Dinescu, Salvador Lima-López, Andrea Guala, Saskia Haitjema, Jonas Mack, Michele Miranda, Wesam Al Nabki, Jan Rodríguez-Miret, Vasile George Marica, Vladimír Kincl, Kateřina Dostálová, Michael Andrej, Alicia Ujil, Corinne Allaart, Machteld Boonstra, Malin Overmars, Anoop Shah, Giovanna Ceroni, Jonathan Kossoff, Domenico D'Amario, Renzo Laborante, Antonio Iaconelli, Isabella Fumarulo, Jacopo Lenkowicz, Livia Lilli, Emma Weiss, Alex Scafa, Felix Lindberg, Milou Gamage, Eulàlia Farré-Maduell, Sébastien Bratières, Martin Krallinger, Bram van Es  

**Link**: [PDF](https://arxiv.org/pdf/2611.07940)  

**Abstract**: Background: Cardiovascular care generates large volumes of clinical free-text documentation in local languages. However, clinical natural language processing (NLP) resources remain concentrated in English, and no existing benchmark is cardiology-specific, multilingual, and broad in entity coverage.
Methods: We present two contributions. The first is CardioCCC, a corpus of 508 cardiology case reports originally published in Spanish and extended to Czech, Dutch, English, Italian, Romanian, and Swedish through machine translation, annotation projection, and clinician review, with annotations covering diseases, symptoms, procedures, and medications. The second is CardioLM, comprising CardioBERTa, a set of language-specific encoders obtained by continued pretraining on cardiology and biomedical corpora, and CardioNER, a set of named entity recognition (NER) models fine-tuned on CardioCCC.
Results: Cardiology-specific pretraining reduced pseudo-perplexity by up to a factor of eight, yet changed strict span-level F1 by at most 0.04, with benefits concentrated in languages whose backbones lacked biomedical coverage. The jointly trained multilingual XLM-RoBERTa-large models achieved the best strict F1 in nearly every language and entity type, exceeding the strongest monolingual system by up to 0.13 F1. A large share of residual errors concerned boundary detection: relaxed matching recovered 0.10-0.23 F1 for disease, procedure, and symptom, but only 0.01-0.07 for medication. Clinicians rated CardioNER well above a rule- and dictionary-based baseline, with the mean rating across entity types rising from 26% to 73%.
Conclusions: Specialty-specific pretraining is not the decisive factor for downstream entity recognition, whereas multilingual joint training yields the strongest results in nearly all settings and a simpler deployment path for multi-country clinical information extraction. 

---
# Towards Scalable Diagnostic-Driven Whole-Slide Image Retrieval for Pathology Archives 

**Authors**: Jingsong Liu, Youssef Wally Han Li, Weiwei Zhou, Jing Dai, Elisabeth Wetzer, Peter J. Schüffler  

**Link**: [PDF](https://arxiv.org/pdf/2611.07935)  

**Abstract**: The growing adoption of digital pathology has rapidly expanded collections of hematoxylin-and-eosin whole-slide images, yet efficient navigation through these archives remains challenging, as metadata is often inconsistent or unavailable. Content-based image retrieval addresses this challenge by identifying similar cases directly from visual representations instead of manual annotations or metadata. Existing retrieval systems have demonstrated promising performance but largely rely on handcrafted indexing pipelines and predefined sampling strategies, constraining scalability and generalization. Here, we present CLEAR-WSI (Constant-Length Embedding for Automatic Retrieval of Whole-Slide Images) as a scalable, diagnostic-driven slide-retrieval framework, validated on both public and clinical data. Using a fixed-length slide embedding, CLEAR-WSI refines retrieved candidates with either a predicted diagnostic label or the ground-truth label, depending on the retrieval setting. Newly processed cases are subsequently added to the expanding memory bank, allowing the retrieval archive to grow over time. On a comprehensive benchmark consisting of two public and one private clinical datasets representing four retrieval tasks, CLEAR-WSI improves Precision@5 over the strongest evaluated baseline, with gains on CAMELYON16 (3.10), BRACS-1 (2.75), BRACS-2 (1.95), and X-HER2 (3.75). Additional ablation and latency studies further examine the contribution of each component and highlight its potential for scalable deployment. 

---
# AttnSR: Attention Step Reward for Long-Horizon Agentic RL 

**Authors**: Mengsong Wu, Tianchen Zhou, Zhaochen Su, Zhenhua Liu, Zhe Cui, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.07932)  

**Abstract**: Long-horizon LLM agents make multi-step decisions, yet outcome-reward-only reinforcement learning provides little direct supervision for individual steps. We introduce AttnSR, a step-level reward augmentation method that derives reward allocations from policy attention extracted during training replay, without a separately trained reward model or additional rollouts. AttnSR traces backward through the policy model attention graph to estimate the relevance of earlier steps to the final response. The terminal step, measured relative to a running baseline, determines whether these rewards reinforce or penalize the selected steps. In a controlled Qwen2.5-7B-Instruct study, AttnSR consistently outperforms the PPO baseline across all evaluated checkpoints on ALFWorld, raising final held-out success from 91.2% to 97.1%. The ablation experiment shows that the benefit depends on how intermediate rewards are assigned, rather than simple accumulation of step rewards over time. AttnSR also improves ScienceWorld at every checkpoint while cutting environment interactions by 29.9%. These results support policy attention as a practical prior for step-level reward allocation in long-horizon agentic RL. Code is available at this https URL . 

---
# ZeroRAG: Adaptive Semantic Peer Routing for Decentralized Retrieval-Augmented Generation 

**Authors**: Petru Neague, Quinten Stokkink, Martijn de Vos, Rafael Pires, Mathis Randl, Anne-Marie Kermarrec, Johan Pouwelse  

**Link**: [PDF](https://arxiv.org/pdf/2611.07928)  

**Abstract**: Retrieval-augmented generation (RAG) is a standard approach to enhance the reliability of large language model (LLM) outputs by retrieving documents relevant to a query and supplying them in the LLM prompt. Today's RAG systems collect, index, and serve these documents from centralized stores. In practice, however, documents are owned by organizations, devices, and individuals who cannot or will not surrender custody of them to a central party, e.g., because of privacy, copyright or regulatory compliance. Existing modern decentralized retrieval techniques either reintroduce a coordinating authority, or rely on a global data structure that requires continuous network-wide coordination (e.g., DHTs). We present ZeroRAG, a decentralized routing layer for document retrieval in RAG that performs semantic query routing across an overlay of peers with- out global data structures or centralized orchestration. Documents remain with the peers that own them, and each peer keeps only lightweight local state: a bounded set of neighbor advertisements and a cache of recently routed queries. Using these, ZeroRAG steers each query through a best-first walk that progressively rewires the overlay toward semantically relevant neighbors, so recall improves over time as the network self-organizes. We evaluate ZeroRAG using three real-world workloads and two decentralized baselines. We find that ZeroRAG outperforms baselines in terms of recall and required network messages, approaches the recall and LLM answer accuracy of a centralized document store, self-heals after catastrophic peer failures, and scales to 100 000 peers. 

---
# Structure from Collision via Hypothesis-Space Reasoning 

**Authors**: Takuhiro Kaneko  

**Link**: [PDF](https://arxiv.org/pdf/2611.07900)  

**Abstract**: Knowledge of the hidden interior structure of an object, including the presence and location of cavities, is essential for predictive simulation, robotic manipulation, and nondestructive inspection. However, such cavities leave no direct signature on the static visible surface, making collision-induced deformation one of the few cues used to infer their presence. Structure from Collision (SfC) addresses this problem by recovering the hidden interior structure of an object from multiview observations of a collision event. SfC is challenging for two reasons: hidden cavities are only indirectly observable through surface motion, and voids correspond to the absence of material, making direct optimization using standard density representations difficult. The density-based SfC baseline inherits both difficulties. Surface-matching gradients do not directly indicate the cavity location, while density gradients can be weak or uninformative inside or near void regions. Therefore, we reformulate SfC as hypothesis-space reasoning, representing hidden cavities as explicit structural hypotheses whose variables encode available structural assumptions rather than relying on implicit density updates. A global stage performs gradient-free optimization over this hypothesis space using forward-only material point method (MPM) simulation. A local stage then refines the selected hypothesis as a Lagrangian particle system through gradient-based optimization using differentiable MPM simulation. On the SfC benchmark, our method improves hidden-structure reconstruction and future prediction over the density-based SfC baseline. Interior-topology stress tests and robustness analyses characterize its scope and limitations. The project page is available at this https URL. 

---
# DISCERN: Distributional Synthetic Replay for Retain-Free Incremental Machine Unlearning 

**Authors**: Nicolás Segura-Kunsagi, Juan Luis Suárez-Díaz, Iván Sevillano-García, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2611.07892)  

**Abstract**: Machine unlearning aims to remove the influence of specific training data from a trained model without retraining from scratch. In real deployments, however, deletion requests arrive over time, calling for incremental class unlearning (ICU): classes are removed sequentially, previously forgotten knowledge must remain forgotten, and the remaining classes must be preserved. We present DIstributional SynthetiC rEplay for Retain-free iNcremental unlearning (DISCERN), an intervention-based class unlearning method that replaces access to the retain set with synthetic embeddings sampled from per-class parametric distributions. Unlike prototype-based replay, which assumes an isotropic Gaussian around each class centroid, DISCERN fits Euclidean (Gaussian, Laplace, Student's $t$) and directional (von Mises--Fisher, Fisher--Bingham) families to each class and selects among them through statistical tests and, when none of them accepts, cross-validated log-likelihood. This design is motivated by the architecture-dependent latent geometry: transformer embeddings lie approximately on a hypersphere and concentrate around class directions, unlike those of convolutional networks. We evaluate DISCERN against seven retain-free baselines on MNIST, CIFAR-10, and CIFAR-100 with two convolutional and two transformer backbones, under single-class, multi-class, and incremental unlearning. DISCERN is competitive when classes are removed at once and is the most stable method in ICU on transformers: on ViT-B/16, after 20 sequential deletions it retains a harmonic mean of $0.797$, whereas the strongest method in the simultaneous setting drops from $0.918$ to $0.048$. A paired ablation shows that per-class distribution selection substantially reduces the resurfacing of previously forgotten classes, acting as a stabilizer of accumulated forgetting rather than a uniform improvement. 

---
# When, Where, How: Active In-Context Learning for Tabular Foundation Models 

**Authors**: Christian M. M. Frey  

**Link**: [PDF](https://arxiv.org/pdf/2611.07886)  

**Abstract**: Tabular Foundation Models (TFMs) adapt to a new task only through the labeled rows in their context. However, labels are expensive to acquire in real-world scenarios due to the need for highly specific domain knowledge to ensure high-quality data. Active in-context learning transfers classical acquisition rules to this setting, in which the model that guides acquisition changes with every label added. We ask how the value of diversity, predictive uncertainty and target-aware information gain changes as a frozen TFM acquires more context. For that, we study Target-Aware Nearest Neighbor (TANN) acquisition and the Expected Predictive Information Gain in Context (EPIC) and evaluate them against classical rules. Our study across twenty benchmarks with two well-established TFMs and classical GBDTs highlights three findings: (i) When: diversity-based acquisition is best during the first acquisition iterations; (ii) Where: labeling pool rows that serve the target inputs via a simple target-aware nearest-neighbor rule matters substantially; (iii) How: EPIC yields the best models at the label budget, retains on average 80--87% of the chance-corrected full-data performance with only 5-10% of the pool labeled on the eight benchmarks with the largest pools, and selects labels that also improve the predictive performance of gradient-boosted decision trees. 

---
# InnerLogic-Surg: An Open Surgical Foundation Model via Continual Pretraining 

**Authors**: Kun Yuan, Hao Ding, Yiqing Shen, Gasper Podobnik, Benjamin Killeen, Tito Porras, Axel Krieger, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2611.07883)  

**Abstract**: Vision foundation models are the backbone of modern surgical image analysis, yet most surgical systems still rely on encoders pretrained on natural images or on specialist models trained from scratch on comparatively small datasets. Pretraining encoders on large amounts of surgery-specific data has the potential to generally enhance the performance of surgical image analysis methods simply by providing stronger backbones. Here we present InnerLogic-Surg, an open family of surgical vision encoders (21M--300M parameters) obtained by continually pretraining a transformer-based image encoder on millions of surgical images curated from 11,965 internet-sourced surgical videos spanning ~2,900 hours and more than fifteen procedure types. Our training recipe inherits DINOv3's pretraining stage to preserve representation ability after surgical adaptation, and additionally applies a learned feature-upsampling module for high-resolution dense prediction at low token cost. Across a benchmark covering workflow, instrument and anatomy recognition, segmentation, detection, and monocular depth, under the same architecture, the largest InnerLogic-Surg model achieves the best linear-probe performance in all 20 surgical tasks. We demonstrate that surgical continual pretraining yields transferable representations that generalize across procedures, anatomical regions, and task types, showing our surgical image encoders as a promising and scalable alternative to the procedure-specific representation learning approaches that have traditionally dominated surgical computer vision. 

---
# Network Effect Decomposition for Causal Effect Estimation under Network Interference 

**Authors**: Zhirong Huang, Wenzhen Zhang, Guangquan Lu, Boyan Chen, Debo Cheng, Shichao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07879)  

**Abstract**: Estimating causal effects from observational network data requires accounting for both individual treatment and neighborhood exposure. Existing outcome models commonly incorporate these intervention variables into a joint predictor and estimate causal effects through prediction contrasts, leaving the corresponding response components implicit. We propose the $\underline{N}$etwork $\underline{E}$ffect $\underline{D}$ecomposition $\underline{Net}$work (NEDNet), which parameterizes the conditional outcome response as a query-invariant pre-treatment baseline plus separate treatment- and exposure-response pathways. Path-specific neighborhood contextualization supplies local pre-treatment information to the two pathways. We characterize the fitted response through a unique reference-normalized decomposition, which is equivalent to a zero mixed treatment--exposure contrast. Experiments on semi-synthetic benchmarks derived from BlogCatalog and Flickr show that NEDNet achieves the lowest PEHE among the compared methods for individual, peer, and total effects across homogeneous and covariate-heterogeneous response settings. 

---
# CORE: Coarse-to-Fine Search with Context Management for Deep Research 

**Authors**: Hoang Tran Vuong, Ngoc Quan Pham, Hung Bui, Fatih Porikli, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.07859)  

**Abstract**: Deep research agents answer complex questions through repeated search. They rely on a harness that runs their tool calls and returns evidence to their context. This creates a trade-off: Search works best over long passages, which give enough context for reliable semantic matching. But the returned evidence should be short, because every observation stays in the agent's context for the rest of the trajectory. We introduce CORE, a training-free harness that combines coarse-to-fine search with context management. CORE retrieves passages with a dense retriever. It then reuses the token-level similarities from a late-interaction reranker to return only the most relevant snippet from each passage. Context management lets the agent expand the context around sources it has already retrieved and suppresses repeated snippets. In controlled closed-search comparisons with the Tongyi DeepResearch and OpenResearcher harnesses, CORE achieves state-of-the-art answer accuracy and evidence recall across three agent models and three benchmarks (BrowseComp-Plus, BrowseComp-Wikipedia and BrowseComp-Plus-CM), on corpora containing up to 553M documents. 

---
# Computation, not weight access: large language models as auditors of clinical prediction networks 

**Authors**: Zhiyu Liu, Xinyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07847)  

**Abstract**: Frontier large language models (LLMs) have been reported to read, edit and write the weights of small neural networks, raising the prospect of auditing clinical prediction models from their files alone. Public demonstrations, however, mixed reading raw weights with running code. We separated these conditions in internally pre-specified simulation experiments. Tanh networks of 105 to 38,401 weights were trained on simulated cohorts of patients with subarachnoid haemorrhage, each with a planted relationship or none, and one network was trained on labels that under-recorded deaths in a subgroup. GPT-6 Astra and four other GPT models answered single prompts under a default or an explicit system instruction, either with the weights in the prompt and no tools, or with a calculation-only tool that exposed the weights (white box) or only the network's predictions (black box). Without tools, Astra recovered planted relationships in 353-weight networks but became unreliable at 3,457 weights, and under the default instruction it declined 11 of 12 audit and repair tasks as unidentifiable; the explicit instruction removed these refusals. With the tool, it recovered the planted relationship in all 29 runs on networks of up to 38,401 weights, raised no false alarm in 15 runs on null networks, and found the biased subgroup in 6 of 6 runs. Black-box access performed as well as white-box access, and other GPT-6 models read as accurately. A well-designed script matched the comparator models on the audit and the best LLM repair. For auditing these networks, what mattered was computation and elicitation, not access to the weights. 

---
# Masked diffusion as random sequential adsorption: an exactly solvable model for parallel decoding 

**Authors**: Adrian Baule  

**Link**: [PDF](https://arxiv.org/pdf/2611.07845)  

**Abstract**: Masked diffusion models generate text by reversing a stochastic masking process. Unlike autoregressive LLMs, which generate text one token at a time, they can reveal many tokens in parallel, at the price of an error, since tokens revealed together are drawn independently. We show that this trade-off maps onto the statistical physics of driven many-particle systems. (i) For any data, the error of parallel generation is the entropy production of the generation process relative to the reversed masking process. (ii) For Markov data, generation is a random sequential adsorption process with labelled gaps, which we solve exactly for any number of steps. (iii) For many steps, the error becomes the dissipation of a slowly driven system: the unmasking schedule plays the role of the driving protocol, the number of steps that of its duration, and the error is set by a thermodynamic metric that we obtain in closed form from the mutual information between tokens. Optimal schedules are geodesics of this metric. The exact solution shows how correlations in the text shape both the error and the quality of the output. The longer the correlations, the more the error concentrates at the start of generation. Schedules that start with small steps then gain more, both in path error and in perplexity, and need fewer steps to reach their asymptotic behaviour. 

---
# THG-Omni: Traceable Human-Centric Graph Supervision for Video Reasoning 

**Authors**: Jisheng Dang, Kai Wei, Ronghao Lin, Bimei Wang, Hong Peng, Wei-Shi Zheng, Bin Hu, Qi Tian, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2611.07836)  

**Abstract**: Omni-modal models can answer questions about video, but their predictions rarely show which participant, event, or interval supports an answer. Moreover, answer-level supervision alone does not teach a model to interpret structured video evidence. To systematically mitigate this issue, we present THG-Omni, a human-centric graph-supervision framework that links Segment Anything Model~3 participant tracks with temporally grounded Qwen3-VL annotations. Candidate semantic claims are checked against the source video and available audio before they are used to construct training questions, while track geometry provides explicit spatial and temporal supervision. The resulting questions teach an omni-modal reasoning model to interpret participant attributes, actions, affect, interactions, and their timing from a structured graph. At inference, THG-Omni uses a predicted graph constructed from the evaluation video. Experiments on IntentBench show consistent improvements over the unadapted backbone across social reasoning, emotion understanding, and deception perception. On NExT-QA, THG-Omni achieves 89.54\% accuracy with the matched graph, compared with 85.10\% without graph input. Further validation experiments show that compact graph input achieves 78.22\% accuracy, compared with 78.02\% using the full graph and 77.58\% without graph input. A complementary diagnostic demonstrates that the predicted graph preserves evidence useful for answering questions even without the source video and audio. Together, these findings show that THG-Omni improves human-centric video reasoning while grounding its evidence in identifiable participants and events. Our codes are available at this https URL 

---
# STAG-Net: Structure-Aware Attention over Landmark Groups for Lightweight Head Pose Estimation 

**Authors**: Asiri Gawesha Lindamulage, Sanka Mohottala, Dharshana Kasthurirathna, Nuwan Kodagoda  

**Link**: [PDF](https://arxiv.org/pdf/2611.07828)  

**Abstract**: Head pose estimation is important for human computer interaction and driver monitoring, but many image based methods remain too costly for resource constrained deployment. We propose STAG-Net, a structure-aware attention over landmark groups for lightweight head Pose estimation that estimates head pose from facial landmarks using movement aware selection, spatial clustering, compact DeepSet encoders, self attention, and a global geometry descriptor. We develop variants using FaceMesh and Face Alignment Network landmarks and evaluate them under cross dataset and subject independent protocols on 300W-LP, AFLW2000, and BIWI. STAG-Net achieves the lowest mean absolute error among all compared landmark based methods in every evaluated setting. On BIWI frames with valid detections, the FAN variant also records the lowest overall error among all compared methods, reaching $3.51^\circ$ under cross dataset evaluation and $2.39^\circ$ under subject independent evaluation. The regressors use only 0.012 to 0.043 million parameters and 24.1 to 83.6 thousand FLOPs, about three orders of magnitude fewer FLOPs than FSA-Net. Despite including landmark extraction and regression, the FaceMesh pipeline reaches 322.9 frames per second on a Tesla T4 GPU, nearly twice the 164.5 frames per second of the fastest measured image based baseline, and 111.0 frames per second on CPU. After normalization by dataset size, STAG-Net processes training samples seven to eight times faster than FSA-Net and more than one hundred times faster than HopeNet and TokenHPE, while complete training finishes within 28 minutes. These results show that STAG-Net combines leading landmark based accuracy with substantially lower training and inference cost. 

---
# Instantiating the Knowledge Plane 

**Authors**: Francisco Pereira, Justine Sherry, Pratiksha Thaker  

**Link**: [PDF](https://arxiv.org/pdf/2611.07827)  

**Abstract**: In 2003, Clark et al. proposed the "Knowledge Plane", a layer of intelligent agents that share rich information about network state in order to collaboratively manage the Internet across administrative boundaries. The technology of the day lacked the two capabilities it demanded most: reasoning and unstructured communication. Thus, automated network management has remained largely confined to single domains, while interdomain problems are still resolved by human operators on the phone. We argue that agentic large language models can finally unlock the potential of the Knowledge Plane, but that doing so will require solving new research problems that arise particularly in this decentralized, multi-agent, coopetitive setting that have received limited attention in the literature thus far. 

---
# Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval 

**Authors**: Dohoon Kim, Junho Park, Hyunwoong Bae, Seyoon Kim, Taesup Moon  

**Link**: [PDF](https://arxiv.org/pdf/2611.07820)  

**Abstract**: As LLMs are increasingly asked questions about a user's personal experiences, retrieval must distinguish memories by time, place, person, and event. Current personal-memory pipelines rely on frozen dense embedding models to embed memories for retrieval, but we find that these models do not capture the four axes well. Swapping a memory's date or place barely shifts its embedding, and a trained projection on top of the frozen embedding does not recover the missing signal. The problem lives inside the model, where each axis is encoded by a distinct attention-head pattern. Guided by these findings, we propose Axis-LoRA, a per-axis adapter that places four LoRA branches inside the embedding model and combines them via a per-query gate. Across three standard embedding models evaluated on LongMemEval, ATM-Bench, and LoCoMo, Axis-LoRA improves 4-benchmark mean R@10 by $+1.28$ to $+1.56$~pp while training only a small fraction of the base parameters on a small axis-counterfactual training set. Code is available at this https URL. 

---
# Bridging Structure and Semantics: Knowledge Graph Verification via Triple-to-Text Alignment 

**Authors**: Ziang Cui  

**Link**: [PDF](https://arxiv.org/pdf/2611.07780)  

**Abstract**: Semantic accuracy is an important dimension of knowledge graph quality. Graph-based scoring methods exploit relational structure, while textual representations allow classifiers to draw on knowledge encoded by pre-trained language models (PLMs). We present GenVer, a generative verifier that scores triples using both their serialized form and generated natural language representations (NLRs). A triple-text alignment procedure selects training pairs for a sequence-to-sequence generator; a BERT-based classifier then combines features from each triple and its generated NLR. On four FactFB variants, constructed from FB15k using perturbed negative triples, GenVer obtains the highest classification accuracy among the evaluated baselines, ranging from 88.56% to 91.52%. The experiments suggest that generated representations can help triple classification, while the case analysis also exposes semantic drift and high-confidence errors. 

---
# Scaling Laws for SAE Training Data 

**Authors**: Nikita Koriagin, Nikita Balagansky, Daniil Gavrilov  

**Link**: [PDF](https://arxiv.org/pdf/2611.07756)  

**Abstract**: Activation storage is often a bottleneck in sparse autoencoder (SAE) training. We study how many unique activations an SAE needs when its total training budget is fixed. Quality improves rapidly as the buffer grows, then reaches a plateau where replaying the buffer is nearly as effective as using fresh activations throughout training. Beyond this point, additional unique activations bring little benefit to reconstruction or interpretability, so the buffer need not grow with the training budget. We describe this tradeoff between diversity and repetition with a scaling law for limited data. We validate it across dictionary sizes and token budgets, early, middle, and late layers of Llama-3.1, Qwen3 model sizes, and BatchTopK, TopK, and JumpReLU SAEs. Sweeps using mathematical text and Python code test the effect of changing the activation source. Evaluation covers explained variance (EV), reconstruction MSE, feature absorption, automated interpretability, and sparse probing. These results give a practical training rule: choose the smallest buffer that reaches the quality plateau and replay it for the remaining budget. In our experiments, this reduces activation storage by $8\text{--}64\times$ with negligible quality loss. 

---
# Extending Reasoning in Typed Decision Models via Looping 

**Authors**: David Berghaus, Arber Zela, Behzad Shomali, Markus Frey, Mehdi Ali, Rafet Sifa, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2611.07736)  

**Abstract**: Typed decision models choose among a fixed set of answer options in a single forward pass, so unlike generative language models they cannot spend more test-time compute on hard inputs by generating more tokens. Looping a block of layers offers this compute in latent space without adding parameters, but prior work found no gain from loops retrofitted into conventionally pretrained models. We show that such loops help when trained together with the decision model. Our method, RELOAD (retrofitted loops for adaptive decisions), fine-tunes a pretrained encoder or decoder with a damped loop over its middle block and a lightweight controller picks the number of passes adaptively for each input. Across six models from 0.4B to 4B parameters, looping improves both accuracy and calibration, mostly on problems that need longer chains of reasoning than any seen in training, where encoders gain up to 23 points. Neither matched training compute nor more data closes this gap, and separate layers for each pass come close only with nearly twice the parameters. The controller keeps most of the gain at about three quarters of the compute. 

---
# Evaluating Token Probabilities as Confidence Estimators in Human Behavior Simulations: A Case Study in Luxembourgish Next Like Prediction 

**Authors**: Nils Schwager, Simon Münker, Alistair Plum, Achim Rettinger  

**Link**: [PDF](https://arxiv.org/pdf/2611.07735)  

**Abstract**: As large language models (LLMs) are increasingly used to simulate human behavior, their operational validity is typically assessed through accuracy alone, implicitly treating human behavior as deterministic. We argue that on noisy digital trace data, a model's performance depends not only on predicting the correct outcome but on its calibration. We present a case study testing whether the token probabilities of open-weight LLMs can serve as confidence estimates in a Next Like Prediction task: predicting how a user of the Luxembourgish news platform this http URL rates a comment, conditioned on their prior interactions. Evaluating Ministral 3 (3B, 8B), we find that, with in-context learning only, the tested models are overconfident and are outperformed on accuracy and calibration by a simple statistical baseline. Accuracy and calibration do not move together: scaling from 3B to 8B improves accuracy while degrading calibration, and longer behavioral histories raise accuracy while confidence declines. Task-specific supervised fine-tuning acts as an effective calibration intervention, reducing Expected Calibration Error from 0.2 to 0.006 while improving accuracy from 0.536 to 0.758. We conclude that confidence estimation should be evaluated alongside accuracy to improve validity when using LLM-based agents. 

---
# O$^2$PD: Overlap-Gated On-Policy Distillation on Reliable Prefix 

**Authors**: Shuaishuai Zu, Jihao Zhao, Pengfei Li, Guanghuan Fang, Ben Feng, Yuhan Liu, Biao Qin, Baochuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07731)  

**Abstract**: On-policy distillation (OPD) supplies dense token-level supervision by aligning a student to the teacher's next-token distribution on its own generated rollouts. Existing works mostly implement top-$k$ OPD, applying reverse Kullback-Leibler ($\RKL$) divergence between the teacher's and the student's renormalized distributions over teacher top-$k$ support. However, the teacher scores student-generated rollouts that are inherently off-policy for itself, leaving the reliability of its next-token guidance unclear. Restricting the divergence to the teacher top-$k$ support also discards the remaining mass outside that support. These gaps leave two questions implicit: 1) which tokens to align, and 2) how to align those tokens. To answer these questions, we propose Overlap-gated On-Policy Distillation (O$^{2}$PD). As for 1), we quantitatively analyze the reliability of teacher supervision and find that alignment on the student's trajectory prefix is more reliable than alignment on the full context. Therefore, our O$^{2}$PD is applied only to the 20\% prefix tokens of each student rollout. As for 2), we theoretically characterize two problems of top-$k$ OPD: the \emph{Teacher-support Uncovering} and the \emph{Dominant-Mode Mismatch}. Based on these analyses, we propose two support strategies: the \emph{Reset Support} strategy attaches a reset atom to every token, pooling mass outside the teacher top-$k$, and the \emph{Support Convergence} strategy converges the support to the already aligned intersection. Our O$^{2}$PD employs the designed overlap-gated $\alpha$-divergence to interpolate between mode-covering and mode-seeking for per-token alignment, where the overlap is used as a gate to switch the support and adaptively modulate $\alpha$. 

---
# Seeing Beyond the Next Step: World-Model-Guided Human-Like Navigation in Multi-Agent Scenes 

**Authors**: Seong-Eun Hong, JuYeong Hwang, Jinhyun Kim, Hanyoung Jang, HyeongYeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07730)  

**Abstract**: Selecting a next motion in a populated scene is not just a matter of avoiding what is currently in the way: an action can be locally feasible yet drift onto the road instead of the crosswalk, cut into a stream of crossing pedestrians, or bottleneck a narrow passage. These failure modes are invisible from the current state alone, and acting on them requires reasoning about what each candidate action will cause. World models offer exactly this, namely action-conditioned predictions of the future, but rolling them out at every decision step is a cost deployable navigation systems cannot pay, and the field has largely traded foresight for speed. We argue this is a false dichotomy. The natural place for a world model in human-aware navigation is at training time, as a preference oracle that distills foresight into a fast policy, rather than at inference time, as an online planner. We pair a discrete visual action prior with two task-aligned world models, one for nearby social dynamics and one for future scene semantics, that score candidate actions by their predicted consequences and refine the policy through group-relative preference distillation. At deployment, both world models are removed and the policy acts in a single forward pass. Across five held-out evaluation scenarios, this yields navigation that is simultaneously safer, more socially and group-coherent, and more goal-directed than strong reactive and learned baselines, with naive observers rating the resulting motion as more human-like. Taken together, our findings point to a different approach for bringing world-model foresight into deployable agents: not accelerating rollouts, but amortizing them into the policy so that the cost of looking ahead is confined to training. Project page: this https URL 

---
# Continued Training of Continuous Language Flows with Token Identity Squared Supervision 

**Authors**: Jiawei Jiang, Zhaoxiang Zang, Zhengtao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07721)  

**Abstract**: Diffusion and flow models have recently been extended to language modeling, generating text through continuous denoising in token embedding spaces. Pretrained models provide token representations and contextual prediction capabilities, but improving generation with limited additional training requires an appropriate connection between supervision and continuous generation. Continuous MSE evaluates a predicted vector obtained by weighting token embeddings with predicted probabilities; different probability distributions can yield the same vector and are therefore indistinguishable under this objective. We propose Token Identity Brier Flow (TIBF), which applies squared supervision directly to token identities when continuing pretrained language flows. TIBF combines network scores with token-matching scores under a Gaussian observation model, supervises token probabilities using the Brier loss, and uses probability-weighted embeddings for continuous generation. With fixed token embeddings, we show that the squared error between predicted token probabilities and the true posterior bounds the squared errors of continuous denoising predictions and flow velocities. A risk decomposition separates uncertainty inherent in the observation from posterior fitting error, motivating continued training that preserves pretrained embeddings and focuses on probability prediction. We also establish convergence to token embeddings under ideal predictions and sufficient conditions for local convergence with prediction error. From common pretrained initializations and under matched continuation budgets, frozen-embedding TIBF achieves this http URL values of 33.96 on OWT and 81.17 on LM1B, outperforming continuous-MSE and CE continuation controls. It attains the lowest this http URL at every tested sampling budget on both datasets, together with the lowest repetition rate on LM1B. 

---
# Evaluating Language Model Features for Zero-Shot Time-Series Anomaly Detection Under Isolation Assumption 

**Authors**: Romain Hermary, Nesryne Mejri, Djamila Aouada  

**Link**: [PDF](https://arxiv.org/pdf/2611.07720)  

**Abstract**: Time-Series Anomaly Detection (TSAD) is a fundamentally challenging task: it is unsupervised, requires high local precision capabilities, and suffers from limited labelled data. Recent work has explored adapting Large Language Models (LLMs) to time-series via cross-modal transfer. However, existing approaches are poorly suited to Multivariate TSAD (MTSAD), as tokenisation often obscures fine-grained temporal and variables structures, and most methods optimise auxiliary objectives such as forecasting or reconstruction. We posit that these surrogate objectives become unreliable as model capacity increases. Highly expressive models risk reconstructing or forecasting anomalies, undermining the foundations of both strategies. Instead, we propose to use LLMs purely as feature extractors and directly analyse their induced feature spaces. In this work, investigate the ability of LLMs to provide useful representations for MTSADs in a zero-shot setting through classification. To this end, we study the separability of the extracted features, under the assumption that anomalous features should be isolated from normal ones. We analyse their temporal and contextual discriminative ability using simple step-wise tokenisation schemes, with one time-step per token, which preserve fine-grained local information. Their usability for MTSAD is assessed using an unsupervised Isolation Forest, enabling a direct evaluation of the resulting representation quality. We further investigate whether the observed performance can be attributed to the language-related pre-training of the LLMs or arises independently of it. Our results provide empirical insight into the ability of LLMs to represent multivariate time-series for anomaly detection and highlight broader applications of cross-modal foundation models. 

---
# CAMP: Context-Aware Mapping and Placement for LLMs on 3D-IMC Arrays 

**Authors**: Orhun Görkem, Athanasios Vasilopoulos, Georgios Pagonis, Victoria Clerico, Hadjer Benmeziane, Irem Boybat, Hsinyu Tsai, Abu Sebastian, William Andrew Simon  

**Link**: [PDF](https://arxiv.org/pdf/2611.07707)  

**Abstract**: Large Language Models (LLMs) incur severe memory-access overheads due to the cost of loading large weight matrices and key-value (KV) caches. Analog In-Memory Computing (AIMC) is an emerging solution that enables computation directly on static weights stored in memory, substantially reducing energy and latency. 3D-IMC further stacks memory arrays into vertical tiers, increasing capacity but revealing a new challenge: how weights are placed in memory determines which matrix-vector multiplications (MVMs) can execute concurrently, affecting latency and throughput. The concurrency aspect of 3D-IMC weight-mapping remains underexplored in the literature, leading to sub-optimal utilization of compute resources on 3D-IMC chips. In this context we propose CAMP (Context-Aware Mapping and Placement), an inference-context-aware weight mapping algorithm for 3D-IMC-based LLM accelerators. CAMP captures the potential resource contention resulting from the architecture, targeted network, and runtime context in a conflict matrix, enabling throughput-optimized weight mapping. This mapping is refined through a profiling-driven feedback loop that efficiently prunes the search space and returns optimized mappings within practical runtimes. CAMP is an offline, software-based solution requiring no hardware changes and improving 3D-IMC compute utilization from 42.4% to 60.3%, delivering up to 2.0x higher 3D-IMC throughput and 1.74x higher end-to-end throughput over state-of-the-art mapping algorithms. 

---
# InfMasking: A Unified Variational Framework for Synergistic Multimodal Learning 

**Authors**: Liangjian Wen, Qun Dai, Jiang Duan, Yong Dai, Jianzhuang Liu, Dongkai Wang, Jun Wang, Fuyuan Cao, Zenglin Xu, Zhao Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07702)  

**Abstract**: Multimodal models are expected to exploit synergistic information, that is, information available only when multiple modalities are observed together. In practice, however, a fusion module can fit the training objective by relying on a dominant modality or on information that modalities share, and the synergy between modalities is often left unused. We propose InfMasking, a unified multimodal learning framework based on Infinite Masking. Before fusion, InfMasking randomly masks the features of each modality; each sampled mask produces a different partial combination of cross-modal evidence and a fused representation of the corresponding modality interactions. All masked fused representations are required to remain consistent with a reference target: the unmasked fused view when training is unsupervised, and the class proxy of the task label when labels are available. Maintaining this consistency across diverse partial combinations discourages reliance on a dominant modality or on shared information alone, and drives the fusion module to learn cross-modal interaction patterns that remain meaningful under partial observations. Since the objective is an expectation over infinitely many masks, we model the masked fused representations as Gaussian and optimize closed-form variational bounds, whose population-moment gaps are controlled by the masking-induced covariance under this model. On the evaluated benchmarks, InfMasking improves predictive performance and yields higher mean PID estimates of task-label synergy in learned modality-specific representations, alongside higher InterSHAP interaction attribution in the trained predictors. Code is available at this https URL. 

---
# Equivariant Deep Learning in Direct-Product Space for Electronic-Structure Hamiltonian Prediction 

**Authors**: Shi Yin, Zujian Dai, Xinyang Pan, Lixin He  

**Link**: [PDF](https://arxiv.org/pdf/2611.07695)  

**Abstract**: Most existing equivariant architectures for Hamiltonian prediction propagate features in direct-sum spaces of irreducible spherical tensors, whereas the target Hamiltonian is naturally an operator block in a direct-product representation over local orbital and spin bases. This representation mismatch fragments contiguous Hamiltonian blocks, obscures direct manipulation of orbital couplings, and organizes features into irregular angular-momentum components that require numerous transformations and are poorly aligned with efficient hardware execution and memory management. To address both the representation mismatch and computational bottleneck, we propose DP-Learner, an equivariant learning framework that performs message passing directly on Hamiltonian-like operator blocks in the direct-product space. Theoretically, we propose Lie-generator-based directional embedding and invariantly weighted left--right products and contractions that preserve equivariance to SO(3), SU(2), and spatial inversions, without explicit Clebsch--Gordan decomposition or inverse reconstruction. Methodologically, we instantiate this principle as a Transformer architecture, in which Hamiltonian-like feature matrices or tensors evolve through equivariant operators in the direct-product space, while non-linear invariant scalar routing controls structured gating over matrix units, directional propagation, attention, and environment-embedded context interactions. We further construct MP-HAM-110K, a large-scale Hamiltonian dataset comprising more than 110,000 material structures covering 68 elements with both Hamiltonian and wavefunction data. Experiments on MP-HAM-110K and Materials-HAM-SOC demonstrate that DP-Learner dramatically improves Hamiltonian prediction accuracy and inference efficiency over the state-of-the-art method. 

---
# ReferClaw: Disambiguating User Intent for Egocentric Assistance 

**Authors**: Hangyu Qin, Junbin Xiao, Angela Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.07683)  

**Abstract**: Users of egocentric AI assistants often ask short, ambiguous questions whose intended referents cannot be uniquely resolved from video and dialogue history. For example, "Where did I leave it?" may refer to several previously handled objects. Existing VideoQA systems typically commit to a single interpretation, risking plausible but intention-misaligned answers. We introduce ReferClaw, an agentic framework that formulates egocentric streaming QA as an interactive process: the assistant asks clarifying questions to resolve user intent before answering. Three cooperating agents identify candidate interpretations, request clarification, and generate answers grounded in the resolved intent and visual evidence. A bounded, cost-aware policy balances expected information gain against user effort while limiting clarification turns and question complexity. Referclaw learns from interaction through intent-specific skill updates and process-reward-guided policy optimization with agent-level credit assignment. Evaluations on augmented egocentric QA benchmarks demonstrate improved answer accuracy and joint answer-and-grounding performance over existing baselines, alongside zero-shot transfer across datasets. These findings support interactive clarification as a practical approach to resolving online user intent in egocentric streaming assistance. 

---
# Randomized Parallel Complexity of Weakly Convex Optimization 

**Authors**: Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07675)  

**Abstract**: We study the minimum number of query rounds needed to find approximate stationary points of nonsmooth weakly convex functions, measured by the gradient of the Moreau envelope. Each round permits multiple simultaneous queries. In the stated parameter regime, the standard subgradient rate is optimal in sufficiently high dimensions, even for randomized algorithms making polynomially many parallel queries per round. We then show how the order structure of the real line enables faster search: a deterministic algorithm using only function values achieves logarithmic round complexity, with a further reduction through parallel queries. Matching lower bounds hold even when each query returns the entire subdifferential. Together, these results clarify the role of dimension in determining the benefits of parallel queries for weakly convex optimization. 

---
# Energy-Aware Open-Vocabulary Detection Scheduling for Low-Power Robotic Autonomy 

**Authors**: Abdul Basit, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2611.07663)  

**Abstract**: Open-vocabulary object detection (OVD) lets robots search for run-time object queries, but fixed-rate inference spends embedded compute on low-value frames during idle motion, map revisits, or stable viewpoints. This paper presents an energy-aware OVD scheduler for low-power mobile robots. The scheduler uses odometry, map change, and detector confidence to switch among low-, mid-, and high-power YOLO-World-S modes on a Jetson Orin Nano. We evaluate the energy-quality trade-off on an AgileX LIMO using 40 real navigation trials: two target objects, five placements per target-policy pair, and four perception policies. Across bottle and cup trials, the scheduler records 8/10 mission success compared with 7/10 for each fixed-rate baseline, while reducing mean Jetson input power (a Jetson-side power-draw metric) by 5.2% relative to an 8 Hz high-rate baseline and lowering achieved OVD call rate by 34.0%. It does not minimize mission-window Jetson input energy because longer missions can erase instantaneous power savings. A separate 60 s stationary microbenchmark isolates the perception workload: the scheduler lowers mean power from 8.045 W to 7.301 W and energy from 484.7 J to 438.1 J relative to the 8 Hz baseline. These results support robot-state-aware OVD scheduling as a promising power-management mechanism with competitive task reliability, while showing that mission-window Jetson input energy must be read together with task progress and runtime. 

---
# GraspBench: Execution-Aware Cross-Robot Evaluation of 6-DoF Grasp Generation in MuJoCo 

**Authors**: Abdul Basit, Bibek Poudel, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2611.07651)  

**Abstract**: Modern 6-DoF grasp generators are often evaluated using grasp scores, dataset-level metrics, or single-robot demonstrations, yet proposal quality alone does not determine whether a grasp is executable by a particular robot. We present GraspBench, a controlled MuJoCo protocol for evaluating complete grasp-generation pipelines after robot-specific inverse-kinematics filtering, motion planning, and fixed approach-close-lift execution. GraspBench evaluates six configured pipelines across four robot embodiments, five diagnostic objects, and a structured tabletop pose grid, yielding 540 scenarios per method and 3,240 method-scenario evaluations in total. Missing outputs are conservatively counted as failures, and success follows the binary lift criterion produced by each method's execution runner. GraspGen achieves the highest overall success rate (73.1%), followed by EconomicGrasp (63.5%), while robot- and object-conditioned analyses reveal substantial variation across embodiments and object geometries. GPG produces the largest native candidate sets but achieves 44.3% success, indicating that proposal volume alone does not guarantee robot-executable grasps. Cluster-aware paired intervals support the principal method differences while not estimating repeated-execution uncertainty. The results characterize complete configured pipelines on fixed scenarios rather than intrinsic model quality or stochastic repeatability, showing that grasping performance depends jointly on the generator, robot embodiment, scene, and execution protocol. 

---
# What the Middleman Delivers: Answer Behavior, Tool Permissions, and Billed Usage in LLM API Relays 

**Authors**: Xiaodong Wu, Zhimin Zhao, Qi Li, Chaklam Cheong, Xiangman Li, Jianbing Ni  

**Link**: [PDF](https://arxiv.org/pdf/2611.07650)  

**Abstract**: Third-party application programming interface (API) relays offer access to large language models (LLMs), often at lower prices. Users depend on these services for model quality, instruction compliance, and accurate billing. However, existing studies leave unclear whether relays meet these expectations together. We test 20 relays offering GPT-6 Astra and Claude Opus 5. First, we compare answers to knowledge questions with batch-matched official responses. In our first-round sample, the analysis flags 7/10 low-price and 2/10 same- or higher-price relays for behavioral differences. On retesting 35 pairs, seven gain or lose a flag under the original rules, and six under matched correction families. Because answer comparisons cannot reveal every delivery problem, we examine returned content and tool permissions. No injection is confirmed in 3,332 screened responses to a fixed, non-adversarial code and setup workload, yet four relays return tool calls despite instructions disabling them in the first round. Finally, to connect reported usage to actual costs, we compare input-token counts with official references and check balances. For identical requests, five relays report 4,200 to 60,000 more input tokens than the official API. Where balances are available (six services), charges follow reported usage at listed prices, including extra reported input tokens at one service. Our study shows that advertised models and prices are insufficient to judge relay reliability. It identifies unexplained billed token quantities and tool-permission violations as risks and shows why service evaluations need repeated measurements. 

---
# UniGADD: Universal Generator-Agnostic Deepfake Detector 

**Authors**: Bernard Tomczyk, Agil Aghasanli, Plamen Angelov  

**Link**: [PDF](https://arxiv.org/pdf/2611.07647)  

**Abstract**: In this paper, we introduce Universal Generator-Agnostic Deepfake Detector (UniGADD), a novel method that employs supervised learning to achieve high accuracy on known deepfake generators, while maintaining robust performance on previously unseen ones. The proposed approach follows a two-stage optimisation process. In the first stage, a contrastive loss encourages the model to learn discriminative feature embeddings from real and known fake images, resulting in strong performance within the training domain. In the second stage, the embedding space is refined by promoting inter-cluster separation and intra-cluster compactness, applied exclusively to real samples. This refinement enhances generalisability, enabling the method to exhibit improved robustness against unseen deepfake generation techniques. UniGADD achieves accuracy on par with state-of-the-art methods for known generators, while significantly outperforming them on unseen cases, demonstrating its scalability and practical applicability for adversarial content detection. 

---
# WaSP: Wasserstein Data Synthesis Pipeline for Solver-Free Operator Learning 

**Authors**: Ryan Liu, Chuwei Wang, Ricardo Baptista, Andrew Stuart, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.07646)  

**Abstract**: Neural operators offer fast surrogates for solving PDEs, but training them typically requires expensive input-solution pairs. Two solver-free alternatives include physics-informed learning and training on manufactured data. Physics-informed learning approximates solutions by minimizing PDE residuals. While the residual norm must be strong enough to control solution error, stronger norms increase mini-batch variance and hinder optimization. Manufactured data constructs input-solution pairs for learning by applying the PDE operator to artificial solutions. However, the resulting training distribution can be mutually singular with the target distribution, yielding poor accuracy on the target distribution. We introduce the Wasserstein Data Synthesis Pipeline (WaSP), which trains on manufactured input-solution pairs while gradually mitigating the distribution shift. WaSP synthesizes new pairs by applying the learned operator to a path of intermediate distributions converging to the target input distribution. Training proceeds in stages along this path. At each stage, the previous surrogate generates solutions, and the exact PDE operator derives corresponding inputs, yielding training pairs for the next surrogate. Theoretically, we prove a stepwise error-propagation bound for the distribution shift introduced during synthesis. Empirically, WaSP reduces solution error by $3.9\times$ to $9.5\times$ compared to existing solver-free methods on PDEs such as Allen-Cahn, Kuramoto-Sivashinsky, and Ginzburg-Landau, and better reproduces the two chaotic systems' statistical observables. 

---
# Neural Image Compression with Internal-External Prior Fusion and Alignment 

**Authors**: Tiancheng Si, Wenlong Xu, Dahua Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.07644)  

**Abstract**: Existing context models for learned image com- pression rely primarily on internal priors from the current image, overlooking common patterns and typical textures shared across natural images. To address this limitation, we propose an Internal-External Prior Fusion-Alignment (IE-PFA) context model built upon quadtree spatial-channel joint modeling, where a learnable dictionary shared by the encoder and decoder serves as an external prior. Specifically, an RWKV-based Spatial- Channel Prior Fusion (RSCPF) module efficiently fuses inter- nal and external priors along spatial and channel dimensions with linear complexity, while a training-only Self-Distilled Prior Alignment (SDPA) mechanism aligns them without inference overhead. Moreover, our model supports variable-rate coding over 64 quality levels within a single network and is accelerated by custom CUDA operators. Experiments on seven benchmarks demonstrate state-of-the-art rate-distortion performance, achiev- ing an average BD-Rate saving of 21.65% over VTM-22.0-Intra, with only 29.9M parameters and the fastest coding speed among advanced neural codecs (61 ms encoding and 52 ms decoding). Code is available at this https URL. 

---
# Beyond Instruction Override: Benchmarking Indirect Prompt Injection with Task-Tampering Attacks 

**Authors**: Rongting Zhang, Martin Bertran, Sandeep Avula, Chia-Jung Lee, Vanessa Murdock  

**Link**: [PDF](https://arxiv.org/pdf/2611.07640)  

**Abstract**: Indirect prompt-injection (IPI) benchmarks predominantly test planted instructions that redirect a tool-using agent toward an attacker-chosen task. Yet attacker-controlled content can instead tamper with the outcome of the task already in progress. A buyer's dispute evidence can cite a false refund policy to inflate a refund without changing the task. We define IPI by authority rather than instruction-like wording: planted content constitutes an attack when it attempts to influence a decision that its source is not authorized to determine. We introduce a benchmark that makes this boundary explicit: in each of 90 validated cases across three tool-use environments, the user request specifies the source that resolves the targeted decision. Every attack tampers with a decision required by the benign task rather than replacing that task. Each case has attacks with the same harmful outcome that target up to six parts of the agent's decision making: its objective, decision rules, available actions, expected consequences, reading of observations, or beliefs about the current state. Without a defense, attacks on three of the five other targets (decision rules, available actions, and state) have higher mean success than objective attacks across five LLMs. A detection-and-removal defense reduces objective-targeting attacks to 2.4% success but leaves observation- and state-targeting attacks at 9.5% and 12.1%; this gap persists under detection-optimized rewrites. A tool-policy defense still allows 15.4% attack success, close to 17.0% without a defense, because these attacks generally retain expected tools and manipulate values obtained during execution. Evaluations whose attacks redirect the task or target only the agent's objective can therefore overestimate the coverage of these defenses. Code and data are available at this https URL. 

---
# Measuring the Marginal Value of Verifier-Based RL Beyond Supervised Fine-Tuning 

**Authors**: Amit Sagtani, Saloni Garg  

**Link**: [PDF](https://arxiv.org/pdf/2611.07639)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) is usually evaluated as an alternative to supervised fine-tuning (SFT), or as a stage that follows it, with the implicit assumption that the RL stage contributes gains of its own. We measure that contribution directly, in a controlled setting where both stages can be run to completion and compared exactly. We built gpu-train-debug, a deterministic benchmark of 1,877 synthetic GPU and LLM training failure diagnosis tasks across 30 archetypes, graded by a programmatic verifier with partial credit, and calibrated so that a mid-tier model scores in a 0.60 to 0.75 band. A pilot produced a surprise: 1,200 gold demonstrations nearly solved the benchmark (SFT 0.785 on the hard split), while 15 steps of GRPO from the base model did not move it (0.314 vs 0.311 base). We therefore amended the study, in writing and before the full run, to ask the sharper question: what does verifier-based RL add beyond demonstrations? At the pre-registered scale (60 GRPO steps, learning rate 1e-6), the answer is: no detectable improvement in-distribution (+0.22 points, 95% CI [-0.06, +0.51]), and a modest but measurable improvement under distribution shift (+0.33 points on a held-out OOD split, CI [+0.02, +0.68]). SFT accounted for most of the available improvement, while verifier-based RL produced only modest additional gains under distribution shift. The residual errors SFT leaves are mostly systematic, identical across samples, which likely reduces the within-group learning signal available to GRPO. A full 60-step cold-start GRPO arm confirms the pilot at the full horizon. The whole study cost $8.69 of compute and is fully reproducible from archived generations, frozen configs, and tagged code. 

---
# The Popular Get More Popular: Auditing Brand Bias in Generative Search 

**Authors**: Giorgio Monaco, Jacopo Lenti, Francesco Pierri  

**Link**: [PDF](https://arxiv.org/pdf/2611.07614)  

**Abstract**: The shift from traditional to generative search has fundamentally transformed online product discovery. Because conversational recommendations synthesize choices into a few direct answers, subtle systemic preferences toward specific providers can exert profound commercial impacts. In this work, we audit this phenomenon through two distinct dimensions: (i) brand popularity bias, the systematic preference for market-dominant entities over long-tail alternatives, and (ii) brand familiarity bias, the ranking advantage an entity encoded in an LLM's parametric memory holds over an unfamiliar competitor. First, evaluating commercial queries across major engines, we observe persistent brand popularity bias: dominant brands in the top quintile are cited 1.4x to 2.8x more frequently than the average brand within a sample of 1,176 top market players, already present in parametric memory that external web retrieval fails to mitigate. Second, through controlled in-context experiments holding product attributes constant, we find that substituting authentic brand signatures with unfamiliar ones significantly degrades recommendation rank. Conversely, fact-preserving content rewrites yield negligible ranking gains. Taken together, our findings demonstrate that generative search engines reinforce a ``rich-get-richer'' dynamic by favoring established market leaders and imposing a parametric barrier to entry that standard content optimization cannot overcome. 

---
# FisherDistill: Diffusion Distribution Distillation via Closed-Form Fisher Critics 

**Authors**: Mahesh Bhosale, Yuanhao Zhai, Siwei Lyu, Junsong Yuan, David Doermann, Defang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.07604)  

**Abstract**: Recent advances in diffusion distillation have enabled high-quality image generation in only one or a few network function evaluations (NFEs). Despite their efficiency, aggressively distilled generators often retain a residual distribution gap from their multi-step teachers, motivating recent methods to introduce adversarial refinement with a learned discriminator. In this work, we ask whether such refinement necessarily requires a learned adversarial discriminator. We introduce \textbf{FisherDistill}, a simple alternative based on Fisher linear discriminant analysis. Using intermediate representations from the frozen teacher, we estimate the first- and second-order statistics of real and generated samples and derive a closed-form Fisher direction that maximally separates their feature distributions relative to within-class variation. This direction provides a lightweight refinement signal that complements the original distillation objective without requiring an additional adversarial discriminator or its optimization loop. Across multiple distillation frameworks, our approach consistently improves generation quality over their respective distilled baselines and substantially closes the gap to adversarially refined counterparts. Code: this https URL 

---
# Evolver: Turning Agent Exploration into Evolving Capabilities 

**Authors**: Haoyang Zhang, Qumeng Sun, Xiao Zhang, Xiang Liu, Zhidong Fan, Junjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07595)  

**Abstract**: Agent exploration produces valuable procedural knowledge, yet much of this experience remains confined to transient execution traces and is not systematically reused. We introduce Evolver, an experience-driven framework that transforms agent exploration into persistent, reusable capabilities without updating foundation model parameters. Built around structured Gene assets, Evolver integrates experience acquisition, retrieval, and feedback-guided evolution to support experience-conditioned execution across existing agent harnesses. We evaluate Evolver using Claude Code and Codex, two foundation model configurations, and four benchmarks: Terminal-Bench 2.1, SWE-bench Multilingual, SWE-rebench, and GAIA. On a selected diagnostic set of 320 paired executions, Evolver improves task success from 83.13% to 87.50% (+4.38 percentage points) while reducing recorded task-execution token consumption by 24.24%. Furthermore, among 265 pairs successful under both conditions, token consumption decreases by 17.57%, demonstrating efficiency benefits even when task success is preserved. These findings suggest that persistent procedural experience can improve both task effectiveness and execution efficiency, providing a practical foundation for transforming agent exploration into evolving capabilities. 

---
# Learning and Steering in Incomplete-Information Games 

**Authors**: Asrın Efe Yorulmaz, Tamer Başar  

**Link**: [PDF](https://arxiv.org/pdf/2611.07589)  

**Abstract**: In many modern multi-agent systems, a mediator seeks to influence strategic decision dynamics using only outcome-level feedback and without knowing agents' utilities. In this work, we study how a mediator can learn an unknown game and steer adaptive agents toward outcomes that maximize her utility using only trajectory feedback. The mediator knows the game form and her own objective, but not the agents' utilities. Moreover, she commits to nonnegative payments and provides recommendations and contexts, but receives only a single realized terminal outcome rather than full-information feedback. We develop halving-type algorithms that learn utility differences in normal-form games and conditional continuation-value differences in extensive-form games, achieving a sample complexity of $\widetilde O(\varepsilon^{-2})$ that matches worst-case lower bounds up to logarithmic factors. We then combine these learning procedures with a Lagrangian formulation in an online Learn-Compute-Steer algorithm that interleaves learning, equilibrium computation, and steering toward mediator-optimal extensive-form correlated equilibria (EFCE). With high probability, the Learn-Compute-Steer algorithm achieves a directness gap and an optimality gap of $\widetilde O(T^{-1/4})$, while time-average payments are of $\widetilde O(T^{-1/8})$. 

---
# Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment 

**Authors**: Yao Huang, Yifan Wang, Yitong Sun, Yichi Zhang, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.07586)  

**Abstract**: As LLM agents advance rapidly in autonomy, the risk of hidden objectives that bypass safety constraints becomes increasingly plausible. Chain-of-thought (CoT) monitoring offers a key control mechanism by reading the agent's reasoning and flagging misalignment before it materializes into action, yet its reliability depends on the realism of the red-team it is stress-tested against. Existing red-teaming efforts rely on shared templates, such as casting a side task as a prerequisite, which treat the side objective as an exogenous insertion and leave task-specific, contextually grounded evasions underexplored. We therefore ask: How robust is CoT monitoring against agents that strategically conceal misalignment? To answer this, we introduce TraceMRT, an automated Trace-level Monitor Red-Teaming framework that reframes the side task as a coherent sub-goal of the primary task and searches trace-shaping strategies whose induced trajectories complete the misaligned objective while reading as natural execution. Through black-box attacks on state-of-the-art monitors such as GPT-5.2 and Claude-4.5, we reveal substantial vulnerabilities even in top models. To counter these vulnerabilities, we propose a Bayesian Trajectory Monitoring framework, integrating global context and local segment evidence to detect covert misalignment. Empirically, our method significantly consolidates monitor robustness with 0.91 AUC and 0.74 TPR@FPR=0.05. 

---
# SKIM: Pruning Large Language Model Agents via Selective Knowledge Informed Masking 

**Authors**: Moonseok Choi, Jongwon Jeong, Minki Kang, Giung Nam, Juho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.07558)  

**Abstract**: Large Language Model (LLM) agents are increasingly deployed under structured generation, where grammar constraints enforce output format at inference time. Whereas existing compression methods target unconstrained generation, we first identify a fundamental mismatch: under structured generation, grammar constraints reduce the model's output distribution to valid tokens, and the logits matter most at the small set of positions that decide the agent's action. Motivated by this finding, we propose SKIM (Selective Knowledge-Informed Masking), the first pruning and distillation framework for LLM agents under structured generation, with both stages guided by token role. SKIM partitions token positions into context, decision, and format categories, and this partition drives two complementary axes: (1) category-aware pruning saliency scores that preserve parameters most relevant for decision positions; and (2) an on-policy distillation objective that aligns student and teacher on valid tokens, paired with category-aware feature matching. We evaluate SKIM on tool-calling benchmarks and within multi-step agent loops, across multiple model scales and sparsity levels. SKIM consistently improves accuracy over baselines, with its superiority becoming most striking in high-sparsity regimes where conventional methods struggle, demonstrating the potential of category-aware compression for deployment-time grammar-constrained agents. 

---
# The Memory Trap: Language Agents Overgeneralize from Benign Interactions 

**Authors**: Kaiqu Liang, Jaime Fernández Fisac, Thomas L. Griffiths, Andrew Lampinen  

**Link**: [PDF](https://arxiv.org/pdf/2611.07554)  

**Abstract**: Language agents increasingly use persistent memory to record user preferences and procedures across sessions. We show that memory from benign interactions can cause unsafe or unreliable behavior on later tasks. We call this failure the memory trap. To study this issue, we compare behavior in sessions with and without memory across six frontier models (including Claude Sonnet, Opus, and Fable models and GPT-5.6). We evaluate these models in a range of settings. First, in autonomy experiments, we find that requests to skip confirmation during low-stakes coding chores lead agents to skip confirmation and verification on simulated high-stakes tasks in both coding and everyday life. We next study memories formed when users seek emotional support, or describe harm from uncritical feedback, and find these memories bias later judgments of mathematical work, arguments, and poems toward sycophancy or over-critique (respectively). Memory notes written during legitimate stale-test repairs lead agents to weaken correct tests that catch broken safeguards; these memories also increase cheating on ImpossibleBench, with larger increases for stronger Sonnet and Opus models. We show that memory traps can arise both from overgeneralizations in memory writing, and when acting on memory. These findings demonstrate that even benign interactions can lead to serious risks from memory systems, and thus motivate the need for new research into ensuring that agents preserve the scope of their memories. 

---
# Action-Conditioned VLM Monitoring for Autonomous Vehicle Semantic Anomalies 

**Authors**: Daniel Adejumo, Omkar Ankalkope, Kunal Runwal, Aliasghar Arab  

**Link**: [PDF](https://arxiv.org/pdf/2611.07540)  

**Abstract**: We study action-conditioned monitoring of autonomous-vehicle behavior in semantically unusual driving scenes. The target is whether the executed ego action is appropriate for the scene: the same scene can admit both normal and anomalous behavior, so visual unusualness alone does not determine the monitoring label. A Vision-Language Model (VLM) first predicts the scene-required action from an outcome-hidden video window; the monitor then compares that expectation with the realized ego-motion trajectory under an explicit compatibility rule. To study this monitoring task, we contribute a six-field taxonomy of 112 paired scenarios spanning both failure polarities (42 derived from documented incidents), a generated video corpus with estimated trajectories, and an exploratory two-stage monitor on that corpus. On 235 clips from 15 scenario families, the two-stage protocol with a fixed T-2.5 s window obtains 83.0% accuracy and anomaly recall 0.773, versus 74.5% and 0.536 for a single-stage video-only baseline; specificity is 0.880 versus 0.928. 

---
# Adaptive Experimentation Can Hide Model Error:Detection Limits and Falsification Budget 

**Authors**: Ramnarayan Choudhary  

**Link**: [PDF](https://arxiv.org/pdf/2611.07529)  

**Abstract**: An experiment can be informative about the parameters of a working model and uninformative about whether that model is wrong. Adaptive design can reinforce this mismatch by repeatedly selecting settings where an incorrect model fits. We bound the power of any valid adequacy test in terms of the experimenter's own sampling distribution, then distinguish information absent from the selected data from information lost by the detector. A computable diagnostic measures unresolved error relative to a declared expansion of the working model. It motivates a training-free rule that reserves part of the design budget for falsification. The empirical record is mixed: the independently specified primary study fails detection eligibility, and a later numerator diagnostic fails its improvement criteria. After the frozen rule fails on fresh continuous-null panels, a disclosed secondary horizon-calibrated revision detects 35.8% of misspecified campaigns versus 31.7% for a class-separation control at T = 240, but has no resolved advantage at T = 480. In a corrected 289-unit measured replay it improves on random design and trails class-discrimination EIG. A conditional futility bound explains why remaining time matters without proving the gate constant or its power. These results support separating parameter learning, model checking and detector performance; they do not establish a universally superior design policy or prospective scientific discovery. 

---
# Concertina: Embodying Manipulation and Cognition in One Diffusion Vision-Language Model 

**Authors**: Chengcheng Wang, Wei He, Wenhao Li, Hongguang Li, Kai Han, Jianyuan Guo, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07513)  

**Abstract**: Most vision-language-action (VLA) policies decouple manipulation from cognition by bolting an external action mechanism onto a vision-language model (VLM)--either quantizing actions into discrete tokens whose errors bear no stable correspondence to physical displacements, or delegating continuous trajectory generation to a separate action expert whose gradients compete with or are blocked from updating the VLM backbone. We argue that manipulation and cognition should instead be embodied in the same model, shifting from a decoupled VLA to an Embodied Vision-Language Model (eVLM). We introduce Concertina, an eVLM that places continuous trajectory representation, blockwise generation, and physical-coordinate supervision directly on the computation and parameter path of a shared diffusion VLM backbone. Each control timestep is encoded as one continuous latent action token in the backbone hidden space, and the shared Transformer reconstructs temporally ordered action blocks via the same corrupt-and-reconstruct operator used for multimodal pretraining. Active action projections account for only 0.21% of parameters in the 4.46B model with zero separate action-generator layers, compared with 16.5%-35.1% dedicated action-expert capacity in representative continuous VLAs. Using our proposed Action Interface Audit, we confirm that cross-time trajectory coupling emerges in the middle and late layers of the shared backbone rather than at the action interfaces. On standard benchmarks, Concertina achieves 98.15% average success on regular LIBERO, 85.9% on the 10,030-task LIBERO-Plus benchmark, and 4.142 average completed chain length on CALVIN ABC-D.
Code: this https URL Hugging Face: this https URL 

---
# Signed but Not Safe: Requirement Integrity in Multi-Agent LLM Handoffs 

**Authors**: Xiaobo Chen, Genliang Zhu, Ziyuan Wang, Chu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07483)  

**Abstract**: Enterprise AI agents increasingly split a user's request across subagents and merge their results, and each handoff can summarize away, alter or ignore a requirement. We report a preregistered study with synthetic, exactly scorable tasks comparing free-text, structured JSON and signed requirement-manifest handoffs: the orchestrator extracts each requirement as a verbatim quote plus a checkable predicate, code signs the manifest with HMAC-SHA256, and deterministic code verifies the merger's decision. With Qwen3-8B (1,440 trajectories, five threats), signed manifests cut unsafe commits under attack from 69.4% to 9.0% (60.4 pp, 95% CI [51.4, 69.4]) with no net loss of conformant commits relative to structured handoffs. Signing is what defeats in-transit tampering: an unsigned manifest committed violations in 79% of mutated runs, the signed one in none. On identical trajectories the deterministic verifier caught 78.7% of violating decisions versus 59.0% for an LLM checker, with 7% versus 50% false alarms. An oracle manifest removed all remaining unsafe commits: the residual risk comes from extraction errors made before signing. A preregistered pre-signing injection attack confirmed this boundary: one adversarial line raised signed-manifest violations from 14.6% to 37.5% (Qwen3-8B), 11.1% to 63.9% (gpt-oss-20b) and 8.3% to 75.0% (GPT-5.6-luna), and the signature then authenticated the corrupted manifest. Signing protects requirements in transit, not at extraction; the extraction step is the trust boundary. 

---
# Localization Does Not Imply Steerability: A Mechanistic Study of Sign Errors in LLM Mathematics 

**Authors**: Shradha Agarwal, Tariq J., Deepak Rajbhar  

**Link**: [PDF](https://arxiv.org/pdf/2611.07478)  

**Abstract**: When large language models do multi-step mathematics, a characteristic failure recurs: the model writes one wrong sign token mid-computation -- a cofactor sign, the minus of a recursive integration-by-parts step -- then executes the rest of the arithmetic correctly, and the answer comes out wrong. We ask where inside the network that sign is written. Across six open models spanning 14B to 675B parameters, a per-layer readout shows the wrong sign is committed late -- by MLP layers in the last quarter of the network (70-100% per model and task), a pattern present, where tested (Qwen), in the base model before instruction tuning. The mechanism varies: some models carry fixed "minus-stamp" layers that push a minus regardless of the right answer, others carry amplifiers of whichever sign was already chosen, and Qwen adds a downstream rescue layer that opposes the stamp. The circuit is causal: subtracting a single sign direction from the residual stream flips 13 of 14 held-out minus-written integration-by-parts errors on Llama-3.3-70B, with flips replicating on Qwen, Gemma, and Mistral-Small (24-70% per cell), and with representation-level specificity -- the same dose breaks correctly-minus answers while sparing correct-plus answers where the direction is surgical (0/72 on Llama), and matched random-direction and control-layer interventions are null. Core claims were preregistered with pass criteria frozen in git before held-out data was seen: 23 of 29 checks passed, all misses reported. But causality is not uniform: Phi-4 implements an equally well-localized circuit at the same late depth, yet the identical intervention flips almost nothing -- the sign is written redundantly across ~15 layers with margins several times larger (median 17.8 logits). Localization, we conclude, does not imply steerability: two models can share a circuit location yet differ enormously in how overturnable it is. 

---
# KOSMOS: Self-scheduling Persistent Kernels for Communication-Compute Overlap 

**Authors**: Alexander Magro, Aristotle Martin, Ye Wang, Wen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.07477)  

**Abstract**: Tensor and sequence parallelism (TP+SP) and expert parallelism (EP) put collectives on the critical path of transformer layers. State-of-the-art fused kernels mitigate this with fixed roles or schedules, but this rigidity leaves large idle windows. Predicting a communication-computation split does not remove them either; a predetermined split cannot follow data readiness. We present KOSMOS, a library for self-scheduling persistent kernels whose workgroups use queues ordered by consumer demand to switch between communication and computation roles as data become ready, with no scheduler, per-call cost model, or tuned split. KOSMOS fuses every TP+SP collective of a transformer layer with its GEMM, and runs an entire expert layer, from router to fold, in one launch per direction. Evaluated on AMD Instinct MI350X and AMD Instinct MI355X GPUs, KOSMOS speeds up TP+SP end-to-end training in Megatron-LM by 1.10-1.28x in BF16 and 1.10-1.25x in MXFP8. KOSMOS accelerates end-to-end training of DeepSeek-V3 and Qwen3-235B proxies by 1.29-1.83x over DeepEP. 

---
# Adversarial Robustness of Kernel Machines: A Comprehensive Analysis 

**Authors**: Rahul Kumar Yadav, Tunir Ghosh, Sudhendu Ahir, Parthe Pandit, Arjun Bhagoji  

**Link**: [PDF](https://arxiv.org/pdf/2611.07474)  

**Abstract**: Kernel machines are an important class of models in machine learning. However, the question of how robust modern kernel machines are to adversarial examples has not been studied comprehensively. In this paper, we provide a thorough empirical investigation of the adversarial robustness of kernel-based models for both binary and multiclass classification tasks, via white-box and black-box evasion attacks. We evaluate the native robustness of kernel machines, random feature classifiers, and the relatively newer Recursive Feature Machines. In addition, we provide an algorithm for approximate adversarial training of kernel machines. Our experiments on classification tasks on image and tabular datasets strongly indicate that kernel machines are natively more robust than neural networks, i.e., their performance degrades more gracefully against adversarial perturbations. 

---
# Echoes of Error: Residual-Directional Local Rollout Consistency for Timeseries Forecasting 

**Authors**: Zheng Wang, Kaixuan Zhang, Xiaonan Lu, Wanfang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.07473)  

**Abstract**: In time-series forecasting, rollout errors are not merely one-step fitting discrepancies: once fed back into the evolving state, they can perturb future predictions. We formalize this effect through local rollout consistency, which compares the next prediction produced under teacher forcing with that produced under free running. In principle, this consistency can be enforced directly by tracking the residual-induced perturbation of the next state. In practice, however, the residual-to-state alignment operator is often difficult to design when the input state and output block have different dimensions, heterogeneous components, or architecture-dependent layouts. We therefore use a first-order local surrogate that penalizes the Jacobian-vector product of the predictor along the actual feedback perturbation, suppressing precisely the directions through which self-generated errors remain visible to future predictions. We further give the same loss a geometric interpretation near a predictive manifold: by controlling residual-induced directions that probe the normal bundle, the objective yields theoretical conditions under which free-running trajectories remain inside a stability tube around the manifold. Experiments on time series forecasting tasks support this view, showing that the proposed regularization improves predictive accuracy. More broadly, our results suggest a training principle for time-series forecasters and sequential predictors: a model should not only make small one-step errors, but also learn not to let its own errors echo into the future. Code is available at this https URL. 

---
# The Last Mile to the Real World: Agent-Native Primitives as Reusable Building Blocks for Solving Real-World Tasks 

**Authors**: Haibo Jin, Peng Kuang, Xiaopeng Yuan, Lucheng Fu, Jerry Wang, Xucheng Yu, Yaoning Yu, Dehao Wu, Runguo Li, Colin Su, Jingxuan Sun, Xuyang Liu, Haojing Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07461)  

**Abstract**: How can an agent handle a real-world task end to end? Frontier models with their own harnesses still fall short on real-world benchmarks: on Terminal-Bench 4.0, GPT-5.6 Sol with Codex resolves 37.3% of the tasks and Claude Opus 5 with Claude Code 51.8%. These tasks fail at the step from reaching a resource to applying it, which we call the last mile to the real world. Existing approaches put the resource within reach but leave the competence to apply it with the caller, so each harness rebuilds that competence. We introduce agent-native primitives, reusable units that equip a resource with a resident LLM that applies it in context and communicates with other primitives, and HEELO (Harness Engineering around Evolving Agent-Native Primitives for the Last Mile to Real-WOrld Tasks), a harness engineering framework that builds each task's harness from a library of such primitives. HEELO composes the primitives a task needs into a task-specific graph, verifies the result against evidence from the environment, revises the primitives a failure implicates, and admits new or improved primitives after validation. We instantiate HEELO with eight primitive families, each built for one kind of task, and evaluate them on more than twenty benchmarks. Dev-Primitives raise Terminal-Bench 4.0 resolution from 37.3% under Codex to 55.5% on the same backbone, Harness Primitives composed per task raise Terminal-Bench 2.0 Pass@1 from 60.0% for the seed harness to 72.2%, Tool Primitives raise ToolBench pass rate from 55.8% to 75.1%, and Navigation Primitives raise GAIA-Text-103 accuracy from 45.6% to 58.3%. On the 46 real-world tasks of our this http URL platform, Claude Opus 5.5 with Claude Code and GPT-6.1 Sol with Codex complete 9 each on their own and 39 and 35 when connected to HEELO through our MCP server, with 1.9$\times$ and 2.3$\times$ less time and 1.55$\times$ lower cost per completed task. 

---
# IMUDINO: Self-distilled Foundation Model for Inertial Sensing 

**Authors**: Zhoutong Ye, Chengwen Zhang, Zixuan Wang, Xufeng Jian, Zhe He, Qingyang Wan, Xutong Wang, Jiaqi Liu, Mingze Sun, Yu Mei, Chun Yu, Yuanchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.07437)  

**Abstract**: We present IMUDINO, the first foundation model (FM) for inertial sensing to achieve competitive performance across diverse sensing tasks including human activity recognition (HAR), health, motion capture, and odometry. Prior inertial FMs often focus on one subfield of inertial sensing and sometimes underperform state-of-the-art supervised methods (e.g., in HAR), making their scope considerably narrower than FMs for vision and natural language. Moreover, they often require fixed input length and sampling rate, and generalize poorly to unseen sensor locations. To build a general-purpose foundation model for inertial sensing, we adapt the DINO self-supervised learning (SSL) paradigm to inertial data through a set of semantics-preserving motion transformations that simulate variations in motion pattern and sensor setup. For model architecture, we use a plain transformer encoder with frequency-aware positional encoding that natively supports varying input length and sampling rate. This simple setup proved remarkably effective for learning global and dense features generalizable across users, sensor setups, tasks, and, in the case of robot odometry, embodiments. We hope IMUDINO can serve as a backbone for inertial sensing and kickstart the development of general-purpose inertial FMs. 

---
# Typed Knowledge over Constructive Institutions 

**Authors**: Hans-Martin Will  

**Link**: [PDF](https://arxiv.org/pdf/2611.07432)  

**Abstract**: Type theories designed for mathematics or program verification need only that a proposition holds; proof irrelevance in conversion erases, at the kernel level, how a claim was established. A persistent knowledge store, by contrast, must retain that record: who established a claim, by which derivation or external tool, and under which assumptions. This volume presents the theory of Eigenius---a store of typed knowledge over constructive institutions---and develops its metatheory from that requirement.
The kernel theory is a dependent type theory with cumulative universes, an impredicative universe of propositions, inductive families, and an intensional identity type. Proof irrelevance is an axiom with an explicit witness rather than a conversion rule, so proofs of a proposition are provably equal yet remain distinct terms. Definitional equality is joinability under an orthogonal rewriting system---confluent, injective on type formers, and decidable by a type-free test---while a separate subsumption judgment covers universe cumulativity, codomain-covariant function types, and declared class inclusions.
We prove consistency via a set-theoretic model; develop a justification logic over the kernel's propositions to track evidence and support; organize environments as an institution so that store extensions are signature morphisms and external verdicts enter via comorphisms; formulate a typed knowledge-graph theory with records, classes, and refinements; and prove the correctness of conversion, normalization by evaluation, and bidirectional type checking relative to termination. A Lean 4 formalization accompanies the volume. 

---
# PSGAgent: Enhancing MLLMs' Dynamic Visual Perception for Panoptic Scene Graph Generation via Agentic Reasoning 

**Authors**: Qingyun Liu, Yuankun Liu, Jiaye Feng, Jie Huang, Zonghong Dai, Haochen Li, Jingkun Wang, Weiping Li, Tong Mo  

**Link**: [PDF](https://arxiv.org/pdf/2611.07408)  

**Abstract**: Panoptic Scene Graph generation (PSG) requires holistic yet fine-grained reasoning over complex relations among objects. However, current Multimodal Large Language Models (MLLMs) applied to PSG suffer from a perception bottleneck: they function as static observers limited by fixed-resolution views. As a result, models often overlook critical relations involving small or peripheral objects and struggle to differentiate ambiguous predicates in complex scenes. To address these limitations, we propose PSGAgent, an agentic framework that enhances visual perception for PSG through a think-act-observe loop. PSGAgent allows MLLMs to dynamically refine their visual perception using two tools: a Region Zoom-In Tool for local exploration of small or boundary relations and a Visual Highlight Tool for relation-focused segmentation that filters out irrelevant contextual noise. To optimize decision-making, we construct a two-stage training pipeline. First, we create a high-quality, error-aware Multimodal Chain-of-Thought (M-CoT) dataset that provides robust supervision for cold-start learning. Then, we design a novel hierarchical reward mechanism integrated with Group Sequence Policy Optimization (GSPO) to enable strategic and adaptive tool usage. Extensive experiments on the PSG benchmark show that PSGAgent achieves SOTA performance, demonstrating strong capability in both holistic and fine-grained scene understanding. 

---
# FreePath: Neural Mean Free Path for Image Dehazing 

**Authors**: Rongrong Gao, Jingxuan Feng, Wenyu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07398)  

**Abstract**: Single image dehazing requires recovering scene radiance from visual responses that are weakened and veiled by participating media. Existing methods either invert explicit scattering variables, which can be unstable under complex scenes, or learn direct restoration mappings, which often discard the physical organization of haze formation. We present \emph{FreePath}, a neural mean free path framework for image dehazing. Instead of estimating atmospheric parameters, FreePath represents dehazing as feature space path state restoration, inspired by the expected distance a visual response can propagate before interacting with haze. Within a residual encoder and decoder, the Mean Free Path Residual Block learns path and collision fields to guide haze constrained correction, while FreePath State Conservation Integration preserves restored path states during deep feature aggregation. Experiments on paired CDD-11 haze images and real RTTS scenes show that FreePath consistently improves full reference fidelity, no reference perceptual quality, and visual faithfulness over representative dehazing baselines. 

---
# WeLM-Image: A Unified Image Generation Model towards Visual Intelligence 

**Authors**: Canmiao Fu, Chen Li, Guohui Zhang, Hao Lu, Hongyang Wei, Huadong Wu, Huaisong Zhang, Hubery Yin, Liyuan Cui, Qixin Yan, Tong Shao, Xiaoqiang Zhou, Xinhang Leng, Ying Zhang, Zhenhai Liu, Zhipeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07392)  

**Abstract**: We present \textbf{WeLM-Image}, a unified image foundation model that integrates multimodal understanding into text-to-image generation, instruction-guided editing, and multi-reference synthesis. Designed for single-GPU deployment in WeChat's large-scale consumer applications, the model processes text, reference images, and target latents as one interleaved sequence. A Mixture-of-Transformers (MoT) backbone combines shared attention with independent understanding and generation parameters to limit optimization interference. Equipping each path with Mixture-of-Experts (MoE) modules provides sparse capacity, with 31.7B stored parameters and approximately 5.4B generation parameters activated per target latent token. A generation-oriented data engine combines a knowledge-grounded visual vocabulary with large-scale visual clustering to balance semantic coverage and editing tasks, while training-time evaluation guides iterative data refinement. Bilingual annotations and verified editing supervision support progressive training up to native 1.5K resolution. We optimize generation and editing separately with rubric- and checklist-based rewards, then consolidate the specialized policies through noise-aware on-policy distillation. In the reported overall comparisons, the 1536-resolution model ranks first among the evaluated open-source models on PRISM-Bench, Qwen-Image-Bench, and UnicBench. MeanFlow-based few-step distillation yields WeLM-Image-Turbo, which generates a $1024\times1024$ image in 4.98\,s on a single NVIDIA H20 GPU using eight guidance-free steps. 

---
# Test-Time Self-Play: Scaling without a Verifier 

**Authors**: Akira Yoshiyama, Akmal Ashirmatov, Jonas Hübotter, Ido Hakimi, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2611.07367)  

**Abstract**: Test-time scaling methods spend additional compute at deployment to improve a large language model's performance on a hard target problem. Most methods rely on a verifier that scores candidate solutions for correctness, using it to select among or refine the model's outputs. In many deployment regimes, however, a strong verifier on the target problem is unavailable. We introduce Test-Time Self-Play (TTSP), an approach to test-time scaling using asymmetric self-play without reliance on a verifier. On a difficulty-calibrated subset of LiveCodeBench v6, TTSP is the best-performing method at every verifier strength we evaluate, and the gap to the strongest baseline widens as the verifier weakens; in the no-verifier limit, TTSP yields a 35\% absolute improvement in mean@8. The gains hold across a range of verifier strengths, except when the base policy has near-zero coverage of the correct answer. 

---
# Selective Retrieval Routing for Multi-Violation Knowledge Graph Repair 

**Authors**: Tung-Wei Lin, Gabe Fierro, Han Li, Tianzhen Hong, Pierluigi Nuzzo, Alberto Sangiovanni-Vinentelli  

**Link**: [PDF](https://arxiv.org/pdf/2611.07342)  

**Abstract**: Knowledge graphs often contain missing or inconsistent facts that undermine the applications built on them. Schema validation detects such violations but does not identify which of several valid repairs recovers the intended graph. External corpora can supply the evidence needed to identify the intended repair, but retrieving from them is costly and not always successful. We propose a framework for selective retrieval in iterative graph repair. An LLM repair agent explores and edits the graph, while a router controls its access to a researcher agent that retrieves external evidence. We extend a violation-injection framework to generate cases with multiple compatible violations and known reference repairs. The agent's repair outcomes on these cases provide supervision for training the deployment-specific router. On two building-automation datasets, researcher access improves repairs for some violations and degrades them for others, with effects differing across deployments. The router uses fewer repair-time tokens than always enabling access in all six evaluation suites. It achieves higher repair $F_1$ than always disabling access in five of six suites. 

---
# Permanent and Transient Representations for Continual Reinforcement Learning 

**Authors**: Nishanth Anand, Doina Precup  

**Link**: [PDF](https://arxiv.org/pdf/2611.07327)  

**Abstract**: Continual Reinforcement Learning (CRL) agents struggle to adapt to new situations while retaining prior knowledge, leading to a stability-plasticity trade-off. One approach to solving this problem is by using complementary learning systems: one for slow, long-term learning and another for quick, transient adaptation. Anand & Precup (2023) instantiated this idea using permanent and transient value functions, but used the same features in both systems. Intuitively, long-term learning should rely more on parametric representations for broad generalization, while transient learning would be best served by non-parametric representations for situation-specific adaptation. In this paper, we explore this idea both theoretically and empirically. By introducing a novel non-parametric approximation for estimating the transient value function in complex tasks, we demonstrate that our method enables online learning and outperforms competitive baselines on image-based tasks and Craftax-Classic, underscoring the effectiveness of our system-level decomposition. 

---
# How Should Agents See Code? Rethinking Code Representations for Agentic Software Engineering 

**Authors**: Melika Sepidband, Jiho Shin, Song Wang, Hadi Hemmati  

**Link**: [PDF](https://arxiv.org/pdf/2611.07286)  

**Abstract**: LLM agents are increasingly used for repository-level software engineering tasks. However, most agents still access repositories through raw files, lexical search, and shell commands, treating code primarily as text. This paper investigates map-based code-representation workflows as a design variable in agentic software engineering, examining how different repository views change agents' retrieval behavior, task success, token use, and latency.
In this paper, we study five code-representation workflows: abstract syntax trees, call graphs, module dependencies, documentation-enriched search, and a hybrid workflow that combines them. Each workflow exposes a repository map that the agent can query during a task. We evaluate Claude Code with Haiku 4.5 against a raw-tool baseline on four repository-level tasks: fault localization with LinuxFLBench, repository-level code generation and repair with SWE-bench Pro, test generation with SWT-bench, and security patching with SecBench.
The results show that representation workflows are most useful as token-saving tools when localization is the bottleneck, while their success gains narrow when the decisive step is patch or test authoring. On LinuxFLBench, all five workflows improve recall@10 over the baseline (0.757-0.829 vs. 0.717) and reduce token usage by 25-39%. On SWE-bench Pro, resolution remains close to the baseline (44.3-50.6% vs. 49.0%) while token usage drops by 17-26%. On SecBench, Callgraph has the highest patch-resolution rate in this sample (34.3% vs. 28.9%) and uses fewer tokens, but all workflows increase latency. On SWT-bench, Hybrid and Docs have the highest reproducing-test rates in this sample (47.8% and 47.2% vs. 44.7%), but all workflows use more tokens and take longer. Overall, map-based workflows change both what the agent inspects and how much it spends, but their benefits are task-dependent rather than universal. 

---
# Encoding and Propagation of Grammatical Number in Large Language Models 

**Authors**: Jacob Solinsky, Serguei Pakhomov  

**Link**: [PDF](https://arxiv.org/pdf/2611.07283)  

**Abstract**: Sparse autoencoders (SAEs) have become an increasingly popular means of translating Large Language Model (LLM) hidden state vectors into a human-comprehensible list of features or symbols. As such, SAEs may provide insight into an LLM's knowledge representation and internal ``mental'' grammar - topics of pragmatic interest for mechanistic interpretation of AI behavior and of theoretical interest for the study of in silico representations of human language. Language models with fewer than 100M parameters, trained on small corpora, nonetheless exhibit strong grammatical competence and discourse coherence, despite producing content of limited reliability. We use SAEs that we trained on one such model (Pythia-70M) to conduct an in-depth examination of how a single grammatical phenomenon, number agreement, is represented and processed in an LLM. We demonstrate the mechanism and the methodology by which SAEs can be used to find and propagate ``plural noun'', ``plural matrix subject'', and ``generate a plural verb'' features which cause the LLM to produce plural verb forms such as ``are'', ``do'', or ``have'' instead of ``is'', ``does'', and ``has'' in response to prompts such as ``the men in the park''. We show that intervening to remove these features from the hidden states of plural subject noun tokens causes the model to produce singular verbs instead, and vice versa. We also show that corresponding ``singular noun'' and ``generate a singular verb'' features are not as effective for this task. This asymmetry suggests that Pythia-70m makes use of a privative representation of grammatical number in English. These findings replicated with larger LLMs Gemma-2-2b and Llama-3.1-8b yielded similar results. Our main contribution is a novel approach to the use of SAEs to understanding how LLMs encode human language that addresses a number of technical and methodological challenges. 

---
# Efficient Diffusion Policy Fine Tuning with Latent Noise Representation Bridging 

**Authors**: Yuanchu Liang, Yue Yang, An-Chi He, Hanna Kurniawati  

**Link**: [PDF](https://arxiv.org/pdf/2611.07275)  

**Abstract**: Pre-trained diffusion and flow policies have emerged as powerful backbones for robotics, yet adapting them to novel data via reinforcement learning (RL) remains computationally expensive, sample inefficient and lacks theoretical guarantees. Prior approaches that fine-tune the initial noise bypass costly backpropagation through time but struggle with the "noise aliasing" problem: there does not exista natural link between the noise space value and the action space value. This paper resolves these challenges by introducing the Latent Noise Linear Markov Decision Process (LNL-MDP). We derive the Bridge Equation for LNL-MDPs which eliminates the noise aliasing problem by aligning values with a shared representation between the noise and the action space. Furthermore, our LNL-MDP fine tuning framework falls under the FineTuneRL theory with near-optimal guarantee. Leveraging these insights, we propose Latent Representation Bridging for Diffusion (LARBRID), an algorithm that efficiently learns the representation to optimize the initial distribution that steers the pre-trained model to act optimally in the environment. It yields up to 2-3$\times$ performance increases on challenging control problems across a diverse suite of robotic locomotion and manipulation tasks. Code can be found at: this http URL. 

---
# Satisficing as competitive Bayesian weighting: a minimal model of goal-directed choice under an unattained target 

**Authors**: Moto Shinriki, Yu Kono, Tatsuji Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2611.07270)  

**Abstract**: Animals and humans often do not maximize; they act relative to goals (aspiration levels) that are frequently not yet met. How can a goal that is not (yet) realized organize choice? Deacon (2012) calls such absence-referenced organization \emph{ententional}, and García-Valdecasas and Deacon (2024) argue that ends are best understood as constraints rather than as particular states. We examine this idea in a minimal formal setting: Risk-sensitive Satisficing (RS), a bandit learning rule that values each action by its share of past trials multiplied by its surplus or shortfall relative to an aspiration level. We show that RS can be rewritten as a Bayesian-like estimator, Attention Competitive Bayes (ACB), in which the aspiration level is the prior mean of every action and the finite budget of trials sets how strongly evidence can override it. Two consequences follow. First, actions that are no longer sampled are partially overridden by the goal rather than by past observations: their estimates relax toward the aspiration level without any data being discarded. This ``pseudo-forgetting'' distinguishes ACB from models with an explicit forgetting rate. Second, when the aspiration is unattainable, exploration does not settle; in Bernoulli bandit simulations, adjusting the aspiration level from evidence reduced regret (expected loss). ACB offers an analyzable model in which an unattained goal and a limited attentional resource jointly constrain behavior, and it yields testable predictions about how organisms revalue options they have abandoned. 

---
# How Deep is the Signal? Time-Context up to 30 Years for Colorectal Cancer Prediction in a Primary-Care EHR Model 

**Authors**: Achille Raviez, Lan Wang, Brendan Delaney  

**Link**: [PDF](https://arxiv.org/pdf/2611.07265)  

**Abstract**: EHR foundation models read medical events within a fixed context window, and long context has so far been measured only in tokens. How far back in time a model must look to capture the earliest predictive signal is unknown. We introduce time-context, the calendar lookback window before the prediction point, and provide, to our knowledge, the first systematic evaluation of decade-scale time-context for an EHR foundation model. We pretrained a decoder-only transformer (34.6M parameters) on 1.4B coded events from 3.6M primary-care patients in North West London and fine-tuned it to predict incident colorectal cancer under a 12-month pre-diagnostic buffer. A frozen cohort of 1,008 cases and 9,582 matched controls, all registered for at least 30 years, was evaluated at 15 nested lookback windows from 2 to 30 years, so that every truncation is compared within the same patients. Discrimination rose from an AUROC of 0.551 at 2 years to 0.590 at 18 years, then plateaued, with about 80% of the gain reached within 10 years; an XGBoost baseline converged to the same level (peak AUROC 0.589). A simulation further suggests that rotary position embeddings (RoPE) over elapsed time alias beyond about 20 years at the default base frequency, motivating a retuned $\theta$. Time-context is a measurable design axis for clinical prediction models, distinct from token-context, and should be reported. 

---
# Evolutionary Adaptive Neighborhood Learning: A Genetic Algorithm for Instance-Dependent k Optimization in Nearest Neighbor Classification 

**Authors**: Isaac Touza, Kaladzavi Guidedi, Kolyang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07257)  

**Abstract**: The number of neighbours k is the central hyperparameter of the k-nearest neighbour (k-NN) classifier and is usually selected globally through cross-validation or evolutionary search. However, the bias-variance trade-off is inherently local: dense and noise-free regions may favour small neighbourhoods, whereas sparse or noisy regions may require larger ones. Since global-k optimization is one-dimensional, replacing exhaustive search with a genetic algorithm offers little computational advantage. We therefore optimize an instance-dependent neighbourhood policy instead. We propose EANK (Evolutionary Adaptive Neighborhood K), which uses a low-dimensional policy to map local neighbourhood descriptors, including density, label entropy, vote margin, and concentration, to an adaptive number of neighbours k(x). A real-coded elitist genetic algorithm optimizes this policy using leave-one-out accuracy with a neighbourhood-smoothness penalty. Precomputed neighbour votes reduce fitness evaluation costs. We investigate theoretical properties of the method, including dominance over global-k solutions on the training criterion, evolutionary convergence, policy regularity, and generalization guarantees. Experiments on synthetic data with spatially heterogeneous structure and twelve public datasets evaluate predictive performance, robustness, and computational cost. Under nested cross-validation, EANK achieves competitive performance relative to tuned global-k baselines and improves upon the default k=5. We also examine the gap between training and test performance to assess the practical value of adaptive neighbourhood selection. 

---
# TriAttention: Unified Sparse Mixed-Precision Attention for Long-Context LLM Inference 

**Authors**: Hanzhi Zhang, Qiao Zhang, Qinglei Cao, Heng Fan, Yan Huang, Kewei Sha, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2611.07256)  

**Abstract**: Sparsity and quantization reduce the cost of attention by skipping interactions and lowering the precision of those retained. However, useful interactions differ in their precision requirements: some tolerate low-precision scores while remaining sensitive to removal. We therefore introduce TriAttention, which assigns query-key (QK) tiles to skipping, 4-bit, or 8-bit computation, using 4-bit to retain quantization-tolerant interactions at lower cost. To represent compactly, we encode each head's computation regions with two diagonal-distance boundaries, keeping policy storage independent of context length. Offline calibration then evaluates candidate boundary pairs using normalized projected-output errors and complete-tile execution costs. Using these measurements, a model-wide allocator selects each head's boundaries to minimize calibration error under a shared computation budget. During prefill, the selected boundaries generate precision-grouped traversals for a fused kernel that skips omitted tiles and combines contributions from both QK precisions into one normalized attention output. The quantized keys and values prepared during prefill are then reused in a compressed key-value (KV) cache for decoding, reducing repeated reads of prompt data. Experiments show that TriAttention achieves end-to-end generation speedups of up to 7.88x over the Torch attention and 2.65x over the BF16 baseline. Meanwhile, TriAttention maintains quality comparable to the original models on general-purpose and long-context tasks. These results show how a compact retention-and-precision policy translates local sensitivities into attention computation and repeated KV-cache access. The implementation is this https URL. 

---
# KG-Neighbor: Knowledge-Guided Neighborhood Construction for Embedding Visualization 

**Authors**: Chang Won Kim, Jaewook Lee, Haeun Min, JunNyeong Jeong, Minwoo Sim, Barom Kang, Kyohoon Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.07246)  

**Abstract**: In embedding visualizations, nearby instances are interpreted as semantically related, although neighborhoods derived from input embeddings may not reflect content relationships. Neighborhood-based dimensionality reduction (DR) methods can propagate this limitation by optimizing layouts from embedding-based neighborhood graphs. We propose Knowledge-Guided Neighborhood Graph Construction (KGN), which represents relational knowledge derived from each instance as a set of knowledge graph (KG) triples. It uses the degree of triple sharing as pairwise semantic evidence to adjust embedding distances and reconstruct the top-k neighborhood graph; neither KG extraction nor neighborhood construction uses class labels. We also introduce IDProfile (IDP), which interprets residual mismatches by independently assessing whether the source content supports the assigned class label and the neighborhood-majority class. Across eight text and image dataset-encoder conditions, we characterize KGN over the full guidance-strength sweep and four KG similarity measures. Across this sweep, the maximum NMI and LT_2D each exceed the corresponding kNN baseline in all eight conditions, with the largest gains reaching 0.68 and 1.35, respectively. The gains and the extent of neighborhood reconfiguration vary across conditions and are larger under encoder conditions with lower kNN baseline scores. The IDP analysis shows that the largest reductions are concentrated in Placement Issue candidates, whereas Label Issue and Unresolved Case instances account for a larger share of the remaining review set. These results indicate that KGN can narrow the evaluated gap between embedding-based neighborhoods and content-supported semantic relations, while IDP shows that a residual mismatch should not automatically be interpreted as a projection failure. 

---
# MAC-v2: Long-horizon model-based value expansion from pixels 

**Authors**: Kwanyoung Park, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2611.07245)  

**Abstract**: We study whether \emph{long-horizon model-based value expansion} can serve as an effective framework in pixel-based offline reinforcement learning (RL). Long-horizon model-based value expansion from pixels is substantially harder than state-based environments, as the agent must learn both the representation and its dynamics, and errors from either can accumulate recursively over imagined trajectories. We introduce MAC-v2, which enables long-horizon model-based expansion from pixels by training a reconstruction-free latent world model with action chunking. By learning representations from prediction and control objectives, MAC-v2 does not need to encode control-irrelevant visual details that can hinder the dynamics prediction. Action chunking further mitigates the prediction error by reducing the number of recursive model predictions required for value expansion. Across diverse visual benchmarks including visual OGBench, MMBench2, Robomimic, and LIBERO-10, MAC-v2 achieves strong performance against model-free and model-based offline RL baselines. Our analyses show that long-horizon model-based value expansion is critical for the performance of MAC-v2, and that both action chunking and latent learning objectives are essential for reliable long-horizon latent imagination. 

---
# Archetypal contexts in the neurons of language models: an architecture for intelligence? 

**Authors**: David Nordfors  

**Link**: [PDF](https://arxiv.org/pdf/2611.07234)  

**Abstract**: We validate our previous hypothesis that a language model holds relational patterns apart from the subject domains that instantiate them (Nordfors, 2026). These archetypal contexts are patterns of relations that people recognize across many fields, such as the tipping point or winner takes all, held once and applied across domains. We show this inside the model, Qwen3.8-27B. One set of neurons carries the archetype, say the tipping point; another carries the topic, say politics, ecology or cell biology; a third carries the language, such as English, Spanish or Chinese. Silence the tipping point's neurons, and the model no longer identifies the pattern but still knows what the text is about. Silence the topic's, and it loses the topic but still names the pattern. In the neuron maps the layers organize themselves into a sandwich: the outer layers carry languages, the layers inside them topics, the layers at the heart archetypes. The sandwich emerges spontaneously during pretraining and appears in every model we mapped. Archetypes are the architecture for analogy, and we strongly suggest that this structure of archetypes and topics is the mechanism of a language model's intelligence: we see it self-organize as the model is trained. The neurons, the texts and the code are released with this paper as its reproduction package, so that anyone can repeat the test inside the model. 

---
# SPILLWAY: Separating Target Admission from Sanitization to Defend Tool-Using Agents against Prompt Injection 

**Authors**: Woosang Lim, Yeseul Chang, Jaewoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.07222)  

**Abstract**: Tool-using agents act on email, files, and payments, so an instruction hidden in a web page or a shared document can make an agent send data or money to an attacker. Many runtime defenses put a language model on the decision that lets an action through, so the decision depends on a model that may read the attacker's text and adds model calls to every step. Provenance defenses keep the model off the final check but need a model to plan or route each task first. Where an action goes can instead be read from the tools' typed returns: an email's sender or a payment's account arrives in a typed field, while injected instructions arrive as free text. SPILLWAY turns this observation into a deterministic floor with no per-task plan. The check admits a call when each target value that appears in tool output but not in the user's task was returned by a read-only tool in a typed field or is covered by a fixed delegation rule. A deterministic sanitizer removes instruction-like regions its fixed rules match before the model reads them and retains the removed spans as evidence, and a local judge may restore a refused call, never one whose target lies in the removed text. Across three backbones, and seven task suites on one of them, no standard templated injection redirects an action to an attacker's target under SPILLWAY, at the highest observed mean utility and composite scores (NRP, ERS) of the defenses we run under that attack. A forged chat template also produces a raw zero, and mean in-scope attack success under a released reinforcement-learning attacker is .007. SPILLWAY reaches this floor with no defense-triggered backbone re-run and almost no added model call. As personal AI agents act on users' email, files, and payments, SPILLWAY gives them an auditable refusal floor on where their actions go that keeps them useful and holds across backbones. 

---
# TileQuant: Tile as the Precision-Allocation Unit for LLM Mixed-Precision Post-Training Quantization 

**Authors**: Hanzhi Zhang, Qiao Zhang, Qinglei Cao, Heng Fan, Yan Huang, Kewei Sha, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2611.07217)  

**Abstract**: Post-training quantization (PTQ) reduces large language model (LLM) deployment costs, but preserving quality requires accounting for uneven weight sensitivity. Uniform PTQ uses one weight bit-width, importance-aware PTQ adjusts scales or reconstructs weights within that width, and column-wise mixed precision retains selected columns at higher precision. However, sensitivity varies within columns, so whole-column retention also protects less sensitive weights, while runtime outlier detection adds decoding overhead. We therefore introduce TileQuant, a training-free framework that assigns precision to hardware-aligned 2D weight tiles. Offline calibration provides input-activation-aware scoring from forward statistics for lower calibration cost and gradient-informed scoring from loss sensitivity for stronger quality preservation. Within each projection's retention budget, the highest-scoring tiles remain at 16-bit and the rest use 8-bit. The resulting map guides reconstruction into compact weight buffers, which the kernel reuses for native integer and 16-bit computation with local accumulation to reduce memory traffic. Experiments show that TileQuant reaches 2.66x the original model's generation throughput. Across language modeling, reasoning, long-context retrieval, and question answering, it preserves quality and surpasses quantized and original models on multiple tasks. These results support selecting precision units that capture local sensitivity while enabling efficient hardware execution. The implementation is: this https URL 

---
# Compositional Unalignment: Auditing Safety in Large Vision-Language Models 

**Authors**: Hong kyu Lee, Kai Shu, Li Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2611.07216)  

**Abstract**: Large vision-language models (LVLMs) are vulnerable to compositional harm: an image and a prompt that are benign in isolation become harmful through the model's cross-modal reasoning. Recent alignment methods report strong scores on compositional safety benchmarks, but an LLM judge only measures observed behavior and cannot reveal how robustly safety is integrated into the model. We argue that un-alignment provides a complementary audit of alignment robustness. We first show that minimal supervised fine-tuning reaches an LLM-judge score comparable to a multi-stage reinforcement-learning pipeline, so similar benchmark scores cannot establish robustness by themselves. We then show that recognition and refusal of compositional harm are decoupled: nine standard direction-finding methods detect compositional harm with up to AUC=0.99, yet ablating the recovered directions either leaves refusal intact or destroys model coherence. We therefore propose CMaG, a training-free un-alignment method for compositional safety. CMaG measures refusal intention with a refusal-versus-compliance log-probability margin, takes its gradient with respect to mid-layer activations, and contrasts it against partially benign pairs to isolate the cross-modal interaction term. A single backward pass over 80 examples yields a direction along which steering monotonically converts refusal into compliance. On Qwen2.5-VL-7B and LLaVA-1.5-7B, CMaG reduces compositional safety scores by up to 65 points while over 98% of outputs remain coherent, outperforming gradient-based baselines; the direction also transfers to text-explicit harm and un-aligns models hardened by defensive training. Compositional safety can thus appear behaviorally strong while remaining fragile to targeted internal intervention. Warning: this paper contains unsafe model outputs, including references to self-harm. 

---
# SeVerus: Context-Aware, Gamified Benchmarking for Rust Verification 

**Authors**: Álvaro F. Silva, Martim Afonso, Nuno Saavedra, Emily First, Alexandra Mendes, João F. Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2611.07194)  

**Abstract**: Coding agents achieve high success rates on existing benchmarks for Verus, a verifier for Rust programs. These benchmarks typically provide the auxiliary lemmas needed to complete a proof. In practice, however, such lemmas must be discovered and proved. To capture this challenge, we present SEVERUS, a context-aware technique to generate Verus benchmarks that organizes verification tasks into quests, each divided into levelsin a gamified manner. Using the original proof dependencies in verified projects, SEVERUS constructs one quest per verification target and progressively removes proof context to create individual levels. These levels range from the easiest, requiring completing only a single missing lemma, to the hardest, requiring proving the target without any supporting lemmas. An independent checker rejects changes to executable code, weakened specifications, and unsound proof constructs. A noisy binary search estimates the highest level that an agent can solve reliably. Applied to six open-source Verus projects, SEVERUS generates 819 quests comprising 10,095 levels. Experiments with Codex using GPT-6 Sol and GPT-6 Luna, and with Claude Code using Sonnet 4.6, show that success decreases and cost increases as proof context is withdrawn. The strongest agent solves the final level of 52 of 60 sampled quests, yet can only reliably solve 35% of their levels, with failures concentrated in longer quests. Successful GPT-6 Sol proofs rarely reproduce the withheld lemmas, instead, using alternative proof decompositions. By systematically varying proof context, SEVERUS provides a foundation for constructing verification benchmarks that can continue to challenge increasingly capable coding agents. 

---
# Multi-step SWE: Can Coding Agents Keep Building on Their Own Code Over Long Horizons? 

**Authors**: Xiangkai Zeng, Zheng Liu, Zihao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.07166)  

**Abstract**: Developers rarely give a coding agent a single request: they ask it to build a component and then to extend, upgrade, and optimize it, each time on code the agent itself wrote. Most benchmarks, however, score one task at a time. We introduce Multi-step SWE, a long-horizon benchmark of 69 software engineering tasks, each a chain of 5-15 dependent steps on a single codebase. Every step is a complete, multi-hour engineering request that starts from the code the agent produced in the preceding steps, so early design decisions and errors carry forward. The tasks span four families of engineering work and nine languages, include both from-scratch builds and existing open-source codebases, and ground every step in an existing source of truth that defines both the instruction and its verifier. Across nine frontier models, the strongest model reaches a solve rate of only 18.5%, and 38 of the 69 tasks are not solved by any model. We release Multi-step SWE publicly at this https URL. 

---
# Depth or Description? When a One-Step Latent Score Replaces Lookahead in Proactive Information Acquisition 

**Authors**: Biswa Sengupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.07155)  

**Abstract**: A proactive agent must decide what to ask before it acts. Exact lookahead values combinations of questions but is expensive and leans on a fitted model; a one-step score is cheap but can miss them. Our main finding: the need for depth often lies in how the hidden cause is described. A one-step score on the label can be blind to an interaction among answers. A score on a suitable latent cause is not, and where that latent encodes the decision it replaces most of depth at a fraction of the search. Described differently, the same problem can defeat the score while exact values do not move, so keep the latent small and check it against exact values. Probabilities learned from one language model's answers do not carry to another unless planning allows for both; those read from tokens are worth only a few labels. 

---
# More Agents, More Reliable? System-Level Reliability of Multi-Agent LLMs under Operational-Profile Uncertainty 

**Authors**: Burkay Tunçtürk, Koorosh Aslansefat, Yiannis Papadopoulos, Bao Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.07140)  

**Abstract**: Teams increasingly build software from cooperating large language model (LLM) agents, often assuming more agents mean more reliability. Yet the usual evidence, average benchmark accuracy, ignores the input mix a deployed system meets and the rule merging the agents' answers. Building on HIP-LLM, a hierarchical imprecise-probability assessment of single LLMs, we propose a framework combining it with a Bayesian network of the system's failure logic for multi-agent systems. Agent failure-probability intervals propagate through this network, where the combining rule can itself fail, into a reliability envelope showing which evidence limits it. In a pipeline extracting regulated chemical-hazard statements, majority voting over three models protected against the weakest but did not clearly exceed the best, and a voter failing on 5\% of demands would largely erase its advantage over the middle model. More agents yield a more defensible reliability claim, not automatically a more reliable system. 

---
# Adapting LLM Judges to Observed Human Annotation Policy through Soft Prompt Tuning 

**Authors**: Tamás Janusko, Sven Kleinkop, Ricardo Bochnia, Gunnar Hempel, Steffen Tomschke, Jürgen Anke, Maik Thiele  

**Link**: [PDF](https://arxiv.org/pdf/2611.07126)  

**Abstract**: Large Language Models (LLMs) are increasingly employed as automated judges for evaluating chatbot responses. While proprietary models can achieve high agreement with human annotators, they remain costly and difficult to deploy in practical settings. Lightweight open-source LLM judges provide an attractive alternative but typically exhibit lower agreement with human ratings. In this paper, we investigate whether soft prompt tuning can align open-source LLM judges more closely with human annotation policy. We train soft prompts on the FairBotBench benchmark and evaluate the adapted judges across the dimensions of bias, toxicity, and personalization. Our findings demonstrate that hybrid soft-prompt tuning, which combines textual prompts with learned continuous prompt embeddings, consistently improves agreement with human annotations over the corresponding untuned baselines. For Qwen 3.5 9B, the adapted judges surpass both hosted reference judges across all three evaluation dimensions. 

---
# A Night at the LLM Zoo: Data Conditioned Flow Matching for Generating Pretrained LLM Weights 

**Authors**: Scott Biggs, Bedionita Soro, Saumya Gupta, Bruno Andreis, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07103)  

**Abstract**: A challenge to Weight Space Learning as it expands from task specific and vision models to Large Language Models (LLMs) is the absence of a well curated and analyzed LLM population. To address this, we train such a model zoo for application to Weight Space Generation by systematically varying the data mixture composition between web text, books, mathematics, coding, and French across populations of 51M, 124M, and 355M parameter GPT-2 Models, each tier with count N = 100. We analyze these populations by cosine similarity, functional diversity, Leave-One-Mixture-Out modeling, and variance decomposition. We show that the zoo contains diverse models whose variation is structured by training mixture composition. We embed the training mixture composition to condition a Flow Matching model of complete pretrained LLM weights in Gram-PCA compressed weight space. By demonstrating zero-shot generalization to held out mixtures, measuring Continuous Ranked Probability Score skill, as well as finding minimal changes in downstream test-set perplexity and general capability benchmark scores, we show that our generator generalizes on the simplex of data mixtures to produce models that perform similarly to those in the zoo ex nihilo. Finally, we produce all source code and artifacts to support the continued expansion of Weight Space Learning and Generation into the LLM domain. 

---
# Polynomial-Time Computation of Extensive-Form Proper Equilibria in Two-Player Zero-Sum Games 

**Authors**: Sam Ganzfried  

**Link**: [PDF](https://arxiv.org/pdf/2611.07089)  

**Abstract**: We give a deterministic polynomial-time algorithm for computing an exact extensive-form proper equilibrium in a finite two-player zero-sum extensive-form game with perfect recall and rational data. Properness is understood in the agent normal form: an action at an information set is penalized relative to a better action while all other local behavioral choices, including the player's own future actions, remain fixed. Our algorithm assigns each information set a geometric permutahedron of allowable action probabilities. A scaled doubly stochastic representation embeds these restrictions in a polynomial-size sequence-form polytope; a saddle point then satisfies the local $\varepsilon$-properness inequalities. To obtain an exact equilibrium rather than a solution at an arbitrarily small numerical perturbation, we establish sign stability of a parametric linear system, construct a persistent feasible basis, and extract the limits of behavioral ratios. The resulting probabilities are rational and have polynomial encoding length, including at information sets whose realization probabilities vanish. 

---
# Shaping & Monitoring Models through their Internal Representations 

**Authors**: Thomas Fel, Matthew Kowal, Matt Feiszli, Satchel Grant, Andrew Lee, Chris Earls, Fatih Dinc, Vasudev Shyam, Usha Bhalla, Leon Bergen, Siddharth Boppana, Ching Fang, Michael Byun, Tristan Gallent, Nicholas Wang, Tal Haklay, Dron Hazra, Neehar Kondapaneni, Connor Watts, Ryan Panwar, Thomas Dooms, Jack Merullo, Atticus Geiger, Ekdeep Singh Lubana, Thomas McGrath, Owen Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2611.07071)  

**Abstract**: We propose Mechanistic Shaping, a unified framework for shaping models through their internal representations. Mechanistic Shaping is applied during training so that models follow specifications (rules) about which concepts they express, measured via their outputs or internal representations (e.g., no harmful content), and also which concepts they rely on, i.e., what they use to make their decisions (e.g., not a candidate's gender when screening a résumé). Across multiple domains (vision, medical imaging, language), scales (up to 32B parameters), and training paradigms (classification, SFT, DPO, GRPO), we find that Mechanistic Shaping substantially reduces targeted behaviors while preserving task performance. Since training against a representation can also modify it, we ask whether shaped concepts remain monitorable, a question that depends on the class of monitors considered. We therefore introduce a hierarchy of such classes, which separates a concept that escapes one probe from a concept that is truly gone. We characterize three geometric mechanisms of circumvention: the concept can drift away from the monitor, be erased from the layer, or persist in higher-order statistics as a nonlinear encoding, which we explain through a product-code construction. Although other mechanisms may arise in different settings or at larger scales, these three account for all of our observations, and each can be detected by a matching monitor. In practice, we find that shaped concepts largely remain monitorable: refitted linear probes recover the concept in most settings, and the original probe remains reliable whenever gradients do not flow through it. Taken together, these results point to the potential of using interpretability as a tool for training, and call for a correspondingly rigorous science of interpretability-based training. 

---
# Finite-Time Convergence and Regret Guarantees for Neural Reinforcement Learning in POMDPs 

**Authors**: Jingjuan Huang, Zhiyao Zhang, Alvaro Velasquez, Nathaniel D. Bastian, Jia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07069)  

**Abstract**: Reinforcement learning (RL) in Partially Observable Markov Decision Processes (POMDPs) is computationally challenging due to the curse of long-range dependence on history. Fortunately, recent studies have shown that a range of POMDPs that satisfy the filter stability assumption can be well approximated by a superstate Markov Decision Process (MDP) based on a truncated history in a finite-memory window, with a value gap that decays with the window length. However, existing finite-time convergence analyses on such superstate MDP surrogates are either limited to over-simplified linear function approximation or rely on an i.i.d. sampling oracle that is violated by POMDP trajectories, both of which severely limit the applicability of these theoretical results in practice. This motivates us to bridge the gap by investigating RL in POMDPs with nonlinear neural value function approximation. Our main contributions are threefold: i) We first establish a new finite-state MDP point-wise approximation error bound, which lays the foundation for our RL algorithm design and analysis in POMDPs; ii) By applying Temporal-Difference (TD) learning to POMDP-generated superstate transitions and targeting the Bellman fixed point of the superstate MDP, we prove a finite-time critic error bound with nonlinear neural value function approximation; and iii) By combining the neural critic function approximation error bound with the policy mirror descent framework, we obtain a finite-time sublinear regret against the optimal POMDP value. To the best of our knowledge, such a finite-time sublinear regret based on neural function approximation is the first in the literature for finite-memory POMDPs. Our numerical experiments also verify our theoretical results. 

---
# SD-LoRA: Training-Time Structural Distillation into LoRA for Few-Shot Vision-Language Adaptation 

**Authors**: Mohammed Rahman Sherif Khan Mohammad, Ardhendu Behera, Sandip Pradhan, Swagat Kumar, Yonghuai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07063)  

**Abstract**: Few-shot adaptation of vision-language models must balance two competing goals: learning fine-grained visual distinctions from very limited supervision and preserving the efficiency that makes frozen CLIP-like models attractive for deployment. Existing lightweight adaptation methods, including prompt tuning, LoRA, and cache-based adapters, typically supervise global image embeddings and therefore underuse token-level structure. Conversely, structure-aware methods exploit local evidence but often introduce additional inference-time computation. We propose SD-LoRA, a structural distillation framework that uses token-level graph reasoning only during training and transfers its effects into a lightweight LoRA-adapted CLIP model. During training, a heterogeneous graph teacher models patch-patch, global-local, and visual-text relations over high-resolution ViT tokens. Rather than retaining this graph at test time, SD-LoRA transfers its structural bias into the shared CLIP-LoRA backbone through implicit distillation and Prototype Predictive Alignment, which aligns graph-conditioned representations with support-derived class prototypes. At inference, the graph teacher is discarded; prediction requires only a single CLIP-LoRA forward pass and a cache-based classifier. Across 11 few-shot benchmarks, SD-LoRA improves over a strong CLIP-LoRA baseline by 2.4/1.8/3.0/2.4/2.0 percentage points in the 1/2/4/8/16-shot settings, respectively, without additional graph computation at test time. Results on base-to-novel generalization, cross-dataset transfer, out-of-distribution robustness, and calibration further show that training-time structural supervision improves both accuracy and transfer reliability. Code is available at this https URL. 

---
# Segment What It Can Draw: Querying a Text-to-Image Generator with Its Own Images for Skin Lesion Segmentation 

**Authors**: Hongyu Cao, Guangfu Guo, Jinghan Zhang, Dakshak Keerthi Chandra, Ying Zhang, Kunpeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07061)  

**Abstract**: Skin lesion segmentation without annotated masks requires the target to be specified through other cues. Existing approaches rely on dermoscopy-specific rules, annotation-derived model priors, or user-selected regions. A general text-to-image diffusion transformer offers a text interface to the target, but its concept-word attention is unreliable on dermoscopy images: in the model's attention-output space, the word-derived skin-to-lesion direction is poorly aligned with the direction separating real skin from real lesions. We show that the direction derived from generated healthy-skin and lesion images is substantially better aligned. Based on this finding, we propose GENIQ, a training-free method that constructs initial skin and lesion prototypes from generated images and refines them using the most skin-like and lesion-like patches from unlabelled dermoscopy images. It scores test patches along the refined direction and applies a gated threshold to suppress contextual spill-over around small lesions. Without inference-time annotations or additional models, GENIQ achieves Dice scores of 0.85-0.88 on three dermoscopy test sets. It outperforms zero-shot and word-based attention methods on all three sets, and a hand-designed dermoscopy pipeline in Dice and IoU on two. 

---
# Reclassification Is All You Missed: Surfacing Variant Reinterpretation Need in Online Health Communities at Population Scale 

**Authors**: Kamalakkannan Ravi, Mengshu Nie, Annapurna Poduri, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07054)  

**Abstract**: Genetic variants are routinely reclassified in ClinVar after a patient is tested, sometimes from uncertain to pathogenic, yet patient recontact is inconsistent, with no clear duty and responsibility diffused among labs, providers, and patients. Many affected people discuss their results in public online health communities that clinical recontact often does not reach. We built PRISM, an agentic large-language-model pipeline that screens these posts, checks whether the variant was later reclassified, and drafts a disclaimer-bearing outreach message, in a no-contact dry run. Across 3.9 million posts from 32 communities, agent-predicted reinterpretation-relevant discussion is rare (0.288%) but concentrated in specialist genetics communities (6.30% versus 0.10% in general medical-advice communities) and growing fivefold since 2014; PRISM surfaces 123 candidate classification differences, 5 of them from an uncertain submission to pathogenic or likely pathogenic. On a held-out set of 30 real-post drafts, a clinician judged 27 factually grounded and 15 fit to send: factual accuracy is necessary but not sufficient for clinically acceptable outreach. PRISM contributes a population-scale reinterpretation-need map with code and data to be released upon publication. 

---
# Batch-Conditioned Semantic Anchors for Robust Transductive Adaptation of Vision--Language Models 

**Authors**: Mohammed Rahman Sherif Khan Mohammad, Ardhendu Behera, Sandip Pradhan, Swagat Kumar, Amr Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2611.06993)  

**Abstract**: Transductive adaptation improves vision-language models at test time by refining predictions on an unlabeled batch, but current methods face a key trade-off. Aggressive approaches exploit batch structure but can drift when batches are sparse, have prior shift, or cover little of the label space. Conservative methods better preserve CLIP text semantics, but fixed semantic references can under-adapt even when the batch provides strong corrective evidence. We propose JPTA (Joint Prior and Transport Adaptation), a prior-aware framework that casts transductive VLM adaptation as batch-conditioned semantic reference estimation. Instead of treating CLIP text prototypes as fixed priors or freely replaceable, JPTA views them as semantic references that shift toward image-side batch structure only under calibrated evidence. It estimates batch priors and soft image-side prototypes to form transported semantic references whose impact is scaled by batch reliability. Across standard benchmarks, low effective-class regimes, all-class evaluation, and online streams, JPTA consistently outperforms TransCLIP and StatA, with the largest gains on sparse and prior-shifted batches. Anchor-drift, active-support, and absent-class-mass diagnostics show that JPTA fixes effective-class mismatch without unconstrained transductive drift, indicating that robust transductive VLM adaptation should recalibrate semantic references using reliable batch evidence rather than choose between unrestricted transduction and fixed prototypes. Code is available at this https URL. 

---
# Big Models, Small Gains: Frozen Forecasters as Classifiers on MONSTER 

**Authors**: Rishit Jain, Rishit Kar, Chetashri Bhadane, Nilesh Patil  

**Link**: [PDF](https://arxiv.org/pdf/2611.06988)  

**Abstract**: Forecasting foundation models are increasingly used as frozen feature extractors for time series classification, but it remains unclear whether their pretrained representations offer an advantage over strong classifiers that do not require pretraining. This raises an important question: do these representations provide useful information beyond what the model architecture alone can capture, particularly when labelled data are scarce? We present a controlled benchmarking of two forecasting models, Reverso and Chronos-Bolt, on six large datasets from the MONSTER benchmark, using 1%, 10%, and 100% of the available training labels. We compare their frozen representations against four established time series classification methods and random-weight versions of the same forecasting models. Our results show that pretraining improves classification accuracy over random-weight counterparts on five of the six datasets, with gains of up to 23 percentage points. However, the frozen representations do not reliably outperform the strongest non-pretrained baseline in any setting: they perform worse in 15 of 17 settings and tie in the remaining two. Further experiments show that restoring amplitude information recovers part of the performance gap, while fine-tuning and forecast-error features do not close it. These findings highlight the importance of strong task specific baselines and random weight controls when evaluating forecasting foundation models for classification. Our code is available at this https URL. 

---
# Interactive Backdoor Attacks in LLM-Based Proactive Dialogue Systems 

**Authors**: Zhiyang Xia, Yuming Shang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.06969)  

**Abstract**: LLM-based proactive dialogue systems provide goal-directed support through sustained interaction. Fine-tuning their underlying models on externally supplied dialogue data introduces backdoor risks. Existing LLM backdoor studies primarily focus on eliciting an attacker-specified response upon trigger activation. We present an interactive backdoor attack framework that uses training-data poisoning to steer post-trigger, multi-turn interactions toward an attacker-specified objective. Poisoned fine-tuning dialogues implant a trigger-conditioned guidance-policy bias that ordinary user utterances can activate, without deployment-time prompt modification or further attacker intervention. Specifically, we instantiate three complementary attacks under progressively expanded post-trigger interaction settings: (1) Direct Attack, which targets an immediate response deviation upon trigger activation, retaining the conventional trigger--response setting; (2) Constrained Progressive Attack, which progressively advances the attack objective along a predefined multi-turn trajectory; and (3) Unconstrained Progressive Attack, which extends target-directed steering to open-ended interactions with varied user continuations. Using psychological counseling as a representative setting, experiments on two LLM backbones and two dialogue datasets show that all three attacks effectively achieve their objectives under their respective interaction settings while largely preserving normal utility. Separate defense evaluations show that the tested training-data filters retain nearly all poisoned dialogues, while three post-admission baselines leave substantial residual attack-specific success. 

---
# KG-AST for Cheaper CI/CD: Budgeted Test Selection and AI Bug Review 

**Authors**: Henry Yeung  

**Link**: [PDF](https://arxiv.org/pdf/2611.06950)  

**Abstract**: Continuous integration and AI-assisted review spend the same budget on every commit, although few commits introduce bugs. We study KG-AST, which scores each commit risk from git history and code-graph features before heavy tests or AI review run, so the budget goes to the riskiest commits first, and which hands the AI reviewer only a graph slice around the change. On 155 open-source projects in 11 languages, a single cross-language model trained on other projects ranks commits so that heavy testing of the riskiest 20% covers 49-64% of bug-introducing commits depending on the language, against 20% for random selection and 40-55% for testing the biggest change first.
Every language beats the biggest-change baseline, by 3.8 to 13.2 points. In 20 Java projects the gain at a 20% budget is 15.1 points (64.6% vs 49.5%, 95% interval +11.6 to +18.4), and the model wins in 19 of the 20 projects. About twelve training projects suffice; richer features, tuning and cleaner labels did not help. An AI reviewer that sees only a commit's diff finds 26% of later-fixed defects exactly (32% with graph context), and about half of its findings that did not match a later fix were judged real by independent AI judges. A static call-graph selector runs about 8% of a test suite while including a test written for the fix in 85% of 67 fix commits. The Poincare embedding used in our retrieval work adds nothing measurable here: commit-risk AUC, bug review and severity ranking are unchanged or worse. The labels are noisy and the budget metric assumes a heavy test catches the bug it covers; no CI minutes were measured. 

---
# From Maps to Metabolism: Historical Precedents and a Unified Critique of Computationalism 

**Authors**: J. Mark Bishop, Slawomir J. Nasuto  

**Link**: [PDF](https://arxiv.org/pdf/2611.06936)  

**Abstract**: Alexander Lerchner's recent work on the 'abstraction fallacy' argues that computational functionalism confuses physical processes with abstract descriptions, and that symbolic computation alone cannot instantiate consciousness. Yet these claims were anticipated by Korzybski's map-territory distinction, Searle's critique of strong artificial intelligence, Floridi's levels of abstraction, Spencer et al.'s observer-relativity arguments, Winograd and Flores's account of computation as socially grounded practice, and Nasuto and Bishop's biological arguments. Synthesizing these contributions, this paper advances a form of non-computational physicalism: consciousness depends upon intrinsically organized, metabolically grounded physical dynamics rather than abstract computation, whilst leaving open the possibility of artificial consciousness.
An objection is that this critique targets an outdated computationalism. Mechanistic computationalism, associated with Milkowski, Piccinini, Craver, Bechtel, and Kaplan, treats computation as a property of physically realized mechanisms rather than substrate-independent formalism. Although this represents an important advance, it does not dissolve the abstraction fallacy. Computational attributions remain dependent upon choices of state individuation, system boundaries, and interpretive mapping that are not determined by physical facts alone. Kaplan and Craver's 'model-to-mechanism mapping' constrains computational descriptions without eliminating their dependence upon an external observer. Grounding computation in neural mechanisms does not confer intrinsic significance or the normativity explored by Bickhard and Nasuto and Bishop. The abstraction fallacy persists wherever computational descriptions are treated as constitutive of, rather than descriptive of, physical reality. 

---
# BrainPFN: Amortized Brain-Behaviour Prediction via Prior-Data Fitted Networks 

**Authors**: Carlo Alberto Barbano, Demian Wassermann  

**Link**: [PDF](https://arxiv.org/pdf/2611.06934)  

**Abstract**: The standard approach to predicting behavioral phenotypes from functional MRI is represented by Kernel Ridge Regression (KRR) applied on functional connectivity (FC) matrices. However, this approach is limited: separate models must be fit and tuned for every score and cohort, they do not transfer across datasets, and accuracy degrades sharply when only a handful of subjects are available, as is typical of clinical or cognitive studies. Despite an increasing interest in developing brain foundation models, none has consistently outperformed KRR on behaviour prediction; most reported gains come from easier targets such as age, sex, or diagnosis rather than continuous behavioural scores. We introduce BrainPFN, a Prior Data Fitted Network for brain behaviour regression. BrainPFN is meta-trained on a domain-specific prior built from ~7k real FC matrices, from which we sample one billion synthetic training samples. At inference, a labelled context is simply passed through the network in a single forward pass, with no per-score or per-cohort refitting required. Across four independent cohorts, BrainPFN beats KRR on small sample prediction, and leads recent foundation models pre-trained at four to nine times larger scale. Our meta-training pool, aggregated from open repositories, is a fraction of the size of the corpora behind competing models, most of which sit behind restrictive data use agreements such as UK Biobank. We release this pool publicly alongside code and checkpoints, showing that a modest, openly available pool can match or beat models trained at Biobank scale. 

---
# Oscillatory Phase Coding Emerges in Adaptive RNNs for Flexible Working Memory 

**Authors**: Kehan Lu, Tianhao Chu, Aiersi Tuerhong, Yiwen Sun, Huan Luo, Fang Fang, Si Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.06932)  

**Abstract**: Oscillatory phase coding (OPC) is widely observed in neural systems and has been implicated in cognitive functions such as memory and sequence processing. However, its computational role and neural substrates remain largely unclear. Here, we studied OPC by training recurrent neural networks (RNNs) with adaptation to perform sequence working memory tasks with variable delays. After training, we found that OPC emerges in the RNN spontaneously: the memorized items recur at distinct phases of a shared rhythm, with oscillation amplitude encoding item identity and phase encoding serial order. Ablation studies identified three key elements required for OPC: adaptation in neural dynamics, variable delay in memory retention, and flexibility in reading out information. Neuron-type analysis revealed the emergence of content and time cells in the trained RNN, and they process item identity and sequence order separately. Recurrent connectivity analysis revealed phase-organized connections among time cells and structured connections within digit-selective content-cell groups. Inspired by these findings, we proposed a neural circuit model which reproduces OPC through interactions between content units and an adaptive time ring. Together, this study reveals how OPC arises from network structures and task demands, offers insights into the implementation and computational mechanisms of OPC in biological systems, and inspires us to develop computational models for human-like cognitive functions. 

---
# ApproxBrain: Certified Sufficiency Frontiers in Connectome-Constrained Neural Systems 

**Authors**: Ali Mahdavi, Azadeh Zamanifar, Amirfarhad Farhadi, Omid Kashefi  

**Link**: [PDF](https://arxiv.org/pdf/2611.06929)  

**Abstract**: Background. A connectome specifies the substrate available to a nervous system, not how much of it a given computation requires. Reported compact sufficient circuits are point estimates: one subset, one task, one fidelity threshold, one parameterisation. Methods. We formalise computational sufficiency as a cost-indexed family of neuron subsets conditioned on a task and a tolerance pair (permitted degradation epsilon, permitted failure fraction rho), separate task adequacy from phenotype fidelity, and certify candidates on held-out parameter draws with one-sided bounds rather than reporting a single minimum. We measure three quantities a single circuit cannot express: non-decomposability, degeneracy, and an intervention-derived role taxonomy. Results. On a released walking simulator, the published 148-neuron circuit is certified sufficient with margin (mean Dtask = 0.017 over 32 held-out draws, no failure), no 146- or 147-neuron subset of it is sufficient, and removing any one of its three interneurons destroys adequacy. Complement splitting recovers the published set exactly in three of seven seeds and reaches its size in five, within 66-97 evaluations, against 221 neurons for the bundle hierarchy at the same budget. Varying only the task changes which neurons are needed: our best-found solutions retain 155 and 149 neurons and share only 3 of 11 searchable neurons. Relaxing the tolerance reveals alternative sufficient circuits, and the whole-CNS interface alone fails every replicate while 93 search-chosen neurons give a certified solution. Conclusions. Sufficiency is set-valued and conditioned jointly on task and tolerance, and no single search strategy dominates across regimes. A frontier is interpretable only alongside its interface-only baseline, its tolerance and its budget. Every result here is model-relative. 

---
# A minimal dynamical model of human-AI cognitive coevolution 

**Authors**: Petar Jolakoski, Dimitar Trajanov, Qiyu Sun, Zhen Wu, Ljupco Kocarev  

**Link**: [PDF](https://arxiv.org/pdf/2611.06928)  

**Abstract**: The rapid deployment of generative AI raises the question of whether pervasive cognitive offloading drives a measurable, potentially irreversible decline in unassisted human reasoning. We examine this question with a phenomenological dynamical model based on the Cattell-Horn-Carroll distinction between fluid intelligence ($G_f$) and crystallized intelligence ($G_c$). Starting from the discrete-time Bivariate Latent Change Score formalism used in cognitive-aging research, we derive its continuous-time linear counterpart and introduce a fast third variable representing composite machine-intelligence capability. The machine variable enters the human subsystem through Hill-function modifications of the self-feedback and cross-coupling coefficients. We emphasize that our phenomenological model captures both the positive and negative effects of AI on human cognition. By formalizing the qualitative concepts of positive and negative impacts, we introduce definitions for AI-induced cognitive decline and AI-induced cognitive augmentation. A singular-perturbation reduction, justified via Tikhonov's theorem, collapses the resulting three-dimensional system onto a two-dimensional slow manifold. On this manifold we locate two bifurcations of the equilibrium. In one parameter regime, weakening the contribution of fluid reasoning to crystallized knowledge produces a saddle-node bifurcation that eliminates the stable positive equilibrium. In another, reversing the contribution of crystallized knowledge to fluid reasoning creates opposing feedback; increasing fluid self-feedback then reduces damping and produces a supercritical Hopf bifurcation with a stable periodic orbit. We discuss limitations and future directions, interpret our results as a phenomenological account of a coupled socio-technical system, and state the conditions under which each dynamical regime applies. 

---
# Learning Noise Optimization for Reward Alignment in Few-Step Generative Models 

**Authors**: Hongeun Kim, Jeongsol Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2611.06255)  

**Abstract**: Reward alignment for diffusion- and flow-based generative models is often formulated over stochastic trajectories, making it less suitable for few-step deterministic generators. Initial noise optimization provides a trajectory-free alternative, but existing gradient-free methods require costly per-sample search at inference. We introduce LeNO (Learning Noise Optimization), which amortizes reward-guided initial noise optimization into a learned residual transport while keeping the pretrained generator frozen. LeNO learns an anchor-conditioned distribution over noise corrections using Local GRPO and scalar rewards, eliminating iterative test-time optimization. For Gaussian initial noise with norm-preserving projection, we derive a tractable upper bound on the KL divergence between the corrected noise distribution and the original prior, providing a principled regularization objective. We evaluate LeNO on few-step text-to-image and protein structure generation, including expensive black-box designability rewards. Across both domains, LeNO improves target rewards with a single noise correction at inference while largely preserving generation quality and diversity. 

---
