# Can Language Models Play the Wikipedia Game? 

**Authors**: Alexander Hickey, Joanna Knight, James Bishop  

**Link**: [PDF](https://arxiv.org/pdf/2610.13948)  

**Abstract**: Language models increasingly support or automate information retrieval from document collections too large to index. This task is difficult when it requires specialist linguistic or scientific knowledge. We used the Wikipedia Game to test language models on an analogous information retrieval task. In the game, players must reach a target article using only article links. We compared sentence embedding models and LLMs across five Wikipedia languages and five academic-literature datasets. We construct very large local graphs of Wikipedia pages and of paper citations using OpenAlex and provide code to build and navigate them. This allows the study of language models' navigation capabilities without relying on third-party APIs. We compared the models with human and TF-IDF benchmarks. The Wikipedia Game format enables robust comparisons across techniques, languages, and domains. No model significantly outperformed humans, although the best models (GPT-o3 mini, GPT-4o, QwQ:32B, and DeepSeek-R1:70B) achieved comparable success. Performance and rankings varied across languages and topics, with implications for practitioners utilising these techniques. 

---
# RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics 

**Authors**: Weiliang Chen, Song Liao  

**Link**: [PDF](https://arxiv.org/pdf/2610.13889)  

**Abstract**: As AI agents increasingly rely on external tools, accurate retrieval is essential for reliable behavior. Existing methods optimize semantic relevance but largely ignore tool availability during execution. To address this gap, we propose RATED (Reliability-Aware Tool Retrieval via Execution Dynamics), a retrieval framework that combines robust semantic matching with dynamic reliability adaptation. RATED employs a frozen dual-encoder with contrastive fine-tuning for semantic matching, and a lightweight contextual bandit module that re-ranks candidates based on real-time execution feedback. This design enables rapid adaptation to runtime tool failures without expensive model retraining. We introduce EcoAgent, a benchmark of 5,172 real-world tools and over 210,000 query-tool pairs for offline retrieval and dynamic reliability evaluation. Experimental results demonstrate that RATED achieves a 52.2% relative MRR improvement over the ToolLLM API retriever on real-world queries, and reduces fault selection rate from 86.2% to 2.9% under severe API outage scenarios with a 30% failure rate, while incurring negligible latency overhead of less than 1 ms over encode-only inference. The code and dataset are available at this https URL. 

---
# X-AgentRed: Red-Teaming Personal Agents in Evolving Multi-Service Ecosystems 

**Authors**: Yeonsung Jung, Zhiyuan Wang, Zifan Wang, Vítor Albiero, Benjamin Newman, Edoardo Debenedetti, Ivan Evtimov, Ninareh Mehrabi  

**Link**: [PDF](https://arxiv.org/pdf/2610.13886)  

**Abstract**: Personal agents increasingly connect to services across the web and applications by using tools. As the effects of their actions accumulate and interact across services, unintended or harmful outcomes can arise when agents fulfill benign user requests. Proactively identifying these compositional failures requires systematically exploring diverse cross-service paths. Existing red-teaming methods often simply prompt LLMs to identify potentially harmful paths from task and environment descriptions or decompose harmful objectives specified by humans or LLMs into seemingly benign action combinations. However, generation is biased toward patterns learned during training, potentially favoring familiar tool combinations and leaving novel cross-service interactions underexplored. To address this, we propose X-AgentRed, a self-evolving red-teaming framework that discovers harmful tool call paths through a graph of tool dependencies and expands and refines them using execution feedback. By analyzing each tool's effects, potential harms, and preconditions for execution, the red-teaming agent constructs a directed graph that captures how one tool's effects can satisfy another's execution preconditions. Recipes sampled from this graph are instantiated as realistic scenarios in which agents can produce harmful outcomes. Execution feedback enables self-improvement by diversifying paths that produce harm through tool substitution and refining preconditions at blocked steps in unsuccessful paths to enable further execution. Across four frontier agentic models, X-AgentRed achieves $1.77\times$ the harmful path rate of STAC and explores a more diverse set of tools, yielding $2.99\times$ the tool coverage on average. 

---
# Studying Comics at Scale with Artificial Intelligence: The PAGE Framework and Three Studies of American Comics and Japanese Manga 

**Authors**: Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2610.13840)  

**Abstract**: Comics are one of the great visual records of the twentieth century: hundreds of thousands of pages in which a mass readership saw its leaders, its wars and its heroes. Most of this record has never been studied as a whole; comics scholarship proceeds one book at a time. We argue that artificial intelligence, including generative models, can change this to the benefit of the medium itself. We propose PAGE, a framework that makes artificial intelligence an instrument for comics research. It parses every scan into panels, text and a bibliographic record; annotates each page with a vision-language model that follows a written codebook, as a human coder would; grounds samples of these readings in human annotation; and explains change across an archive, with estimates controlled for publisher and genre. We demonstrate the framework on 3,929 American comic books (1938--1956) and 87 volumes of Japanese manga, in three studies, each resting on a different capability. The first pairs text search with visual reading to recover how a mass medium portrayed political power. The second measures the page's visual form across an industry and recovers the hand of individual artists. The third follows who speaks across two comics traditions, checked against human annotation. Beyond research, the same instruments can make archives searchable by what is drawn, open comics to blind and low-vision readers, and help preserve fragile collections and credit uncredited artists. We outline a research agenda in which artificial intelligence strengthens the study, preservation and reach of the medium. 

---
# UkrMeme: Multimodal Explainable Persuasion Detection in Ukrainian Memes 

**Authors**: Nazar Andrushko, Vera Schmitt, Veronika Solopova  

**Link**: [PDF](https://arxiv.org/pdf/2610.13783)  

**Abstract**: Memes play a central role in modern online communication, especially in contexts of political conflict where visual and textual cues combine to convey persuasive messages. We study persuasive technique detection in Ukrainian memes using vision-language models (VLMs). We introduce UkrMeme, a dataset of 200 organically produced Ukrainian wartime memes annotated for persuasion techniques, and PTM-UA, a Ukrainian translation of a subset of the Propaganda 950 benchmark that preserves original visual content while translating meme text. We evaluate VLMs in a zero-shot prompting setup across image, text, and multimodal inputs using hierarchical F1 metrics. Finally, we compare Layer-wise Relevance Propagation (LRP) and model-generated token attributions and analyze the semantic categories of evidence emphasized by each explanation method. 

---
# A Privileged Teacher That Only Scores Cannot Search: Critique-Conditioned On-Policy Self-Distillation for Reasoning 

**Authors**: Mayug Maniparambil, Ayush Rai, Ekansh Verma, Noel E. O'Connor, Fergal Reid  

**Link**: [PDF](https://arxiv.org/pdf/2610.13744)  

**Abstract**: On-policy self-distillation promises to replace the sparse outcome signal of verifiable-reward RL with dense token-level credit: a teacher conditioned on the reference solution scores the student's own rollout, and the resulting distributional shift is distilled back into the policy. We investigate whether this scoring signal actually contains corrective credit. Using gold error spans in student rollouts, we test the two signals consumed by self-distillation objectives, the full-distribution KL and the token log-ratio, for two properties: whether they localize errors and whether they push probability away from erroneous student tokens. They exhibit neither of these properties nor can they reliably distinguish correct from incorrect rollouts within the same problem. We argue that this failure is structural: locating an error in a reasoning trace is itself a search problem, while a scoring teacher is never explicitly asked to perform that search. The same model, with the same privileged information, localizes errors substantially better when allowed to generate a critique. This forms the motivation for \emph{critique-conditioned self-distillation}: first generate a critique of the current rollout, then condition the scoring teacher on that critique. The resulting signal localizes errors, is corrective within error spans, and shifts probability toward tokens that initiate backtracking such as ``Wait'', ``But'', and ``Actually''. We further show that the critique is useful both at test time, where resuming a wrong rollout with the critique in context converts math problems the model cannot solve by sampling, and in cold-start GRPO, where it provides an advantage signal on prompts for which verifiable rewards alone provide none. 

---
# Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy 

**Authors**: Naji Alazhar, Wassel Alazhar  

**Link**: [PDF](https://arxiv.org/pdf/2610.13708)  

**Abstract**: Long-context serving runs out of KV memory before the hardware reaches its useful throughput. HiSparse stores history in host memory and fetches only the selected entries, but it uses one cache budget at every load and does not handle draft-model speculative decoding such as DSpark and DFlash2. We present Adaptive DHiSparse, which supports speculative decoding over hierarchical sparse KV and adapts residency to load. It protects the required working set, retains additional history while GPU memory is spare, and recovers the rest through exact fused retrieval. DSpark and DFlash2 keep their proposal and acceptance rules: queries share missing-row retrieval, draft KV uses a window-bounded ring, and verification staging is reused across layer groups. At 100K context and concurrency 128, GLM-5.3 with DFlash2 delivers 793 output tokens/s per decode GPU, 1.45$\times$ HiSparse and 1.27$\times$ non-speculative adaptive serving. Held to the same latency bounds as HiSparse, it sustains up to 4$\times$ the concurrency and 2.3 to 3.7$\times$ the output, depending on the per-token bound. On DeepSeek-V4.1-Flash at 500K context and concurrency 512, Adaptive DHiSparse with DSpark delivers 1.85$\times$ the output of resident serving with DSpark. In a load step, optional residency reduces host fetches from 14.2% to 1.4% of selected rows at low load and releases memory under pressure. 

---
# A Baseline Assessment for the Agentic Web Technology Stack 

**Authors**: Vagner Figueredo de Santana, Diogo Cortiz  

**Link**: [PDF](https://arxiv.org/pdf/2610.13703)  

**Abstract**: The fast pace in which protocols, frameworks, and technologies are being proposed for LLM-based agents results in a complex scenario for anyone aiming to create state-of-the-art AI solutions. Without an assessment on how different protocols, frameworks, and libraries work together to compose an Agentic Web solution, developers and organizations face different challenges (e.g., skilling, governance, infrastructure) even before becoming part of the Agentic Web ecosystem. In this context, we propose a system for performing Agentic Web evaluations called MASTIF (Multi-Agent System TestIng Framework) and present a baseline assessment of Agentic Web technologies, including 3 LLMs, 3 protocols, and 4 agent implementation frameworks, against 300 Mind2Web benchmark tasks. Results suggest that selecting a task-specific technology stack may result in gains of task completion rate, efficiency score, and social value stability scores; and reductions in verbosity and time to complete tasks. With the presented results, we expect to present a baseline in terms of metrics and results for this complex and fast evolving scenario for those aiming at designing, assessing, and developing multi-agent systems. 

---
# A Step Harder, Not a Step Too Far: The Benefits and Limits of Curriculum Difficulty Progression in GRPO 

**Authors**: Ronghan Li, Fei Chen, Mingyu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13669)  

**Abstract**: Curriculum learning typically organizes training samples from easy to hard. In GRPO, medium-difficulty samples are attractive for online selection because they are more likely to yield informative group-relative learning signals, and prioritizing them can improve training efficiency. However, it remains unclear whether persistently selecting such samples also leads to the strongest final model performance, or whether training can benefit from progressing toward harder samples. Meanwhile, sample difficulty changes as the policy evolves, so fixed difficulty estimates can become misaligned with the current policy. We therefore re-estimate sample difficulty under the current policy and, inspired by the easy-to-hard principle of curriculum learning, design and evaluate a staged curriculum that progressively shifts the active difficulty band toward harder samples. Experiments on GSM8K show that persistent medium-difficulty selection does not achieve the strongest observed performance. Across three random seeds, progressing to a moderately harder difficulty band improves test accuracy to $0.6591 \pm 0.0077$, compared with $0.6449 \pm 0.0024$ for the fixed-medium curriculum. However, further progression to a harder difficulty band reduces accuracy to $0.6405 \pm 0.0053$. These results show that staged difficulty progression can outperform persistent medium-difficulty selection, but its benefit is non-monotonic. Effective GRPO curricula can therefore benefit from progressing beyond medium difficulty, while avoiding excessive progression toward overly difficult samples.} 

---
# A Multi-Agent LLM System for Automated Tuning of Specialized Neural Network Models 

**Authors**: Ivan Moloshnikov, Dmitry Kunitsyn, Alexander Sboev, Maksim Skorokhodov, Roman Rybka, Yuanchao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13663)  

**Abstract**: Many machine learning approaches are two-level: an expert selects a class of methods, within this class the choice of key components is critical and usually manual. This paper addresses the automatic tuning of these components and proposes a framework transferable across models of different nature. The multi-agent large language model (LLM) framework has three roles---analyst, coder, and recorder~---with an experiment log and a sandboxed execution environment. Unlike systems that optimize the entire ML script or search architectures in a discrete space, the framework operates over an open but semantically narrow space: the agent generates code only for the specified components, leaving the overall network structure unchanged. The approach is validated in two directions: tuning a classification spiking neural network with correlation learning, and solving high-order differential equations with Physics-Informed Neural Networks (PINN). Across five tasks---spiking classification on MNIST and Digits, and boundary-value problems for the generalized nonlinear Schrödinger equation (GNLSE) and Kuramoto--Sivashinsky (KS) equations---the agent outperforms the baselines: classification accuracy is 13 percentage points (MNIST) and 6 percentage points (Digits) higher; the error against the analytical single-soliton solution is reduced approximately 2491-fold; the invariant-preservation error drops approximately 325-fold in the selected GNLSE run with a Gaussian perturbation and by more than 100-fold on average over repeated initializations; and the mass drift drops 14-fold for the KS problem with a Gaussian perturbation. The framework thus tunes key components on tasks of different nature without changing its internal structure. 

---
# Learning Size Independent Constraint Models from Input Data and Positive and Negative Example Solutions 

**Authors**: Helmut Simonis  

**Link**: [PDF](https://arxiv.org/pdf/2610.13640)  

**Abstract**: In this paper we describe a Constraint Acquisition Tool that generates size independent constraint programs from input data and positive and negative example solutions of multiple problem instances. The system finds size specific models based on typical constraint patterns over global and binary constraints, then generalizes the models to a size independent model that can be run on unseen input data. The system produces MiniZinc code as well as a textual description of the model. We evaluate the tool on the example problems of the PTHG21 Challenge and show the feasibility of the approach. 

---
# Humanize Olympiad Agents: A Multi-Session Workflow for Olympiad Problem Solving with Solve, Review, Wiki, Verify 

**Authors**: Zhengyang Zhang, Jing Xiong, Hongzhou Lin, Juihui Chung, Zijian Zhang, Song Han, Sihao Liu, Ligeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13632)  

**Abstract**: Large language model (LLM) agents are rapidly improving on long-horizon scientific and olympiad tasks. Recent systems have produced a Lean-formalized resolution of the Navier--Stokes problem and autonomous solutions to four open questions in the Erdős Conjectures database. Yet neither system releases both its underlying model and complete inference stack. More broadly, a single agent session remains brittle on difficult problems: early errors can propagate, missing domain knowledge can stall progress, and apparent success can mask a misinterpretation or weakening of the original task. We introduce HOA (Humanize Olympiads Agent), an open, flow-level framework that coordinates multiple agent sessions through three complementary feedback channels. First, an independent reviewer evaluates each candidate and provides actionable feedback. Second, a structured wiki supplies task-specific knowledge that may be missing or unreliable in model pretraining. Third, a verifier checks generated Lean proofs against trusted challenge statements. With GPT-5.6 Sol, HOA achieves perfect scores on the IMO, IPhO, and IOI 2026, near-full marks on the IChO 2026 problem set, and solves all 672 problems (100%) on PutnamBench. On the Lean-Eval leaderboard, HOA ranks first, solving 251 of 304 problems (82.6%). We further evaluate HOA with open models. Using Kimi-K3, HOA achieves full scores on the IMO, IPhO, and IOI, and a gold-level score of 97.1% on the IChO. These results closely match the closed-source GPT-5.6 Sol + /Goal baseline on the IMO and IPhO, while exceeding it on the IChO and IOI, demonstrating that systematic feedback can help bridge substantial differences in base-model capability. 

---
# MENTOR: Learning from Failure with On-Policy Pedagogical Intervention 

**Authors**: Guanlan Yue, Hang Yu, Xiaoyong Huang, Ruipeng Li, Wei Zhang, Bingchang Liu, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.13594)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) is a key driver for Large Reasoning Models, but it is vulnerable to "zero-reward collapse", a training failure where a sudden increase in task difficulty causes the learning signal to vanish. We argue that the conventional approach of discarding these failed trajectories is wasteful, as they contain valuable diagnostic information about the model's cognitive gaps, a bright side that can be leveraged for recovery. Existing solutions, however, often regress to static, "off-policy-style" interventions like pre-defined hint libraries or perform post-hoc refinement, failing to provide the adaptive, forward-looking guidance needed to escape the collapse. To address this, we propose a paradigm shift towards on-policy pedagogical intervention and introduce MENTOR, a multi-agent framework that acts as a live mentor. MENTOR diagnoses the root cause of failure in real-time and dynamically generates targeted, instructional prompts to guide the learning agent's next attempt. As a plug-and-play module, MENTOR integrates seamlessly into existing policy optimization pipelines like GRPO without modifying the training infrastructure. Extensive experiments on math and code generation show that MENTOR effectively mitigates zero-reward collapse, outperforming state-of-the-art baselines. Notably, it achieves massive average relative gains of up to 78.3% on Llama-3.2-3B-Instruct over its base model. 

---
# Constructing Pareto Frontiers of Optimal Reasoning Termination Methods for Overthinking in Reasoning Models 

**Authors**: Weixin Guan, Liang Li, Shiyu Xie, Jiapeng Liu, Bing Li, Peng Fu, Chengyang Fang, Xiaoshuai Hao, Can Ma, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.13570)  

**Abstract**: Reasoning models can benefit from test-time scaling, but continued reasoning may become ineffective or harmful, a phenomenon termed overthinking. Evaluations against full thinking compare accuracy and token usage but lack references for overthinking severity and strategies' theoretical mitigation potential. To characterize overthinking at the model-task level, we first introduce Step-wise Truncation Probing (STP), repeatedly generating answers from multiple retained prefixes along each fixed sampled reasoning trajectory. Using each trajectory's earliest measured position attaining maximum empirical accuracy as a reference, we define Overthinking Ratio and Accuracy Degradation as the fraction of thinking tokens generated afterward and the accuracy decrease from this reference to full thinking, respectively. Secondly, we then formulate dynamic routing, fixed budgets, and dynamic early exit within a unified framework to quantify the theoretical mitigation potential of different termination strategies. Based on this unified framework, we use the complete STP measurements to construct a token-accuracy Pareto frontier for each strategy under a common normalized thinking-token budget. We further develop OTBench, a reusable benchmark built from the STP process, which produced 70.5B reasoning tokens and the corresponding probing measurements across six models and four benchmark groups. 

---
# MAGMA-Bench: Evaluating Long-Horizon Embodied Agents under Dynamic Tasks and Execution Failures 

**Authors**: Loan Bernat, Matthieu Grard, Ariane Herbulot, Florent Lamiraux  

**Link**: [PDF](https://arxiv.org/pdf/2610.13534)  

**Abstract**: Long-horizon embodied agents must maintain task-relevant information, coordinate heterogeneous robots, and adapt their decisions as execution unfolds. Existing benchmarks investigate dynamic task execution and embodied multi-agent coordination, but provide limited evaluation of their intersection. We introduce MAGMA-Bench, an architecture-agnostic benchmark for persistent task reasoning in dynamic multi-robot environments. MAGMA-Bench evaluates agents across increasing task horizons and controlled online perturbations, including execution failures, task interruptions, and mission updates, with scenarios requiring coordination across robots with heterogeneous capabilities. It uses a structured skill-level interface to separate task-level reasoning from perception and low-level control. We evaluate locally deployable open-weight planners, multi-agent state-tracking strategies, and frontier models. Even the strongest evaluated models fail on a substantial fraction of clean episodes, and their success rates decrease further under online perturbations. Supervised adaptation greatly improves a small model, but performance remains substantially lower on longer and multi-robot missions. We release the benchmark, 261,646 task-level training examples, and an architecture-agnostic expert-trace generation pipeline. 

---
# Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation 

**Authors**: Shi-Yu Tian, Yong Liu, Ximan Liu, Tianrui Zhu, Sule Bai, Xinwei Yang, Ruiqi Liu, Kun-Yang Yu, Ming Yang, Zhi Zhou, Yu-Feng Li, Yipeng Sun, Gu Simiu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13521)  

**Abstract**: Multi-teacher on-policy distillation (MOPD) extends on-policy distillation (OPD) by routing domain-specific data to specialized teachers, integrating their capabilities through dense token-level supervision on student-generated trajectories. However, domain routing only distinguishes the supervision sources for different domains; updates from all domains still act on shared parameters. When teachers differ substantially in response style and domain expertise, supervision that is effective in isolation may still lead to persistent gradient conflicts and capability degradation during joint optimization. To address this challenge, we propose Cross-Teacher Guarding for MOPD (Road MOPD), which introduces non-routed teachers as auxiliary cross-domain critics on the same student trajectories and coordinates multi-teacher supervision through token-level credit assignment and gradient updates. Specifically, cross-teacher-guided credit assignment uses teacher agreement to strengthen token-level corrections jointly supported by the teachers and attenuate conflicting signals; preventive multi-teacher gradient projection uses auxiliary gradients from the same trajectories to constrain potentially conflicting updates and adaptively controls projection strength to retain target-domain learning signals. In experiments, Road MOPD mitigates degradation in the disadvantaged domain, gains over 5 percentage points under high conflict, and improves overall performance in low-conflict and three-domain settings. 

---
# Causal Effect Identifiability in Causal Abstractions: A Strict Hierarchy of Criteria 

**Authors**: Clément Yvernes, Marianne Clausel, Emilie Devijver, Eric Gaussier  

**Link**: [PDF](https://arxiv.org/pdf/2610.13520)  

**Abstract**: Identifying causal effects from observational data typically requires a fully specified causal graph, an assumption that is often unrealistic in complex or high-dimensional settings. Causal abstractions address this limitation by retaining only partial causal information, thereby defining equivalence classes of compatible causal graphs. In this paper, we study causal effect identifiability directly at the level of these graph collections, rather than at the level of individual causal diagrams. We formalize several notions of identifiability in this setting and organize them into a hierarchy that clarifies their relationships and differences. In particular, we establish that a causal effect may be identified over a collection of graphs without admitting a single causal derivation that holds uniformly across all compatible graphs. These results help reveal gaps and clarify ambiguities in the existing literature. 

---
# Retrieval-Augmented Multimodal Language Model with Heterogeneous Evidence Integration for Clinical Prognosis 

**Authors**: Hamza Shafiq, Jialu Tang, Ting Dang, Jun Hu, Aaqib Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2610.13510)  

**Abstract**: Clinical prognosis requires integrating structured measurements, physiological waveforms, longitudinal history, and population context. We introduce PRISM (Population-Retrieved Inference for Structured Multimodal Prognosis), a multimodal language model that combines per-feature numerical tokenization, recency-weighted aggregation of longitudinal ECGs, and a dual-modality population memory bank. Retrieval operates in representation space: the model receives mean-pooled embeddings of similar training-cohort encounters, while no individual retrieved record enters the prompt. On the 1,443-task MDS-ED benchmark, PRISM achieves AUROCs of 86.2% for diagnosis, 93.1% for deterioration, 91.8% for ICU admission, and 92.8% for mortality. These results use a 96-hour pre-arrival window unavailable to reported baselines. With benchmark-window inputs only, PRISM reaches 0.8895 AUROC, compared with 0.8937 for UniPACT and 0.8889 for MDS-ED. A matched-input compact fusion model using the same five evidence streams and frozen encoders reaches 0.7652 AUROC, versus 0.9092 for PRISM. Replacing numerical EHR encoding with text serialization reduces AUROC by 3.66 points; removing pre-arrival history reduces it by 1.97 points; and zeroing retrieval reduces it by 0.99 points. Genuine retrieval outperforms random-neighbor retrieval by 0.97 macro-average points, although a paired task-level sign test does not establish a uniform advantage (p=0.067). On the external MC-MED cohort, supervised fine-tuning reaches 0.8611 AUROC, compared with 0.8227 for locally trained gradient boosting and 0.7841 for source-only PRISM. These results support structure-aware multimodal integration and supervised external adaptation. 

---
# Humanize MLE Agents: Multi-Agent Session Alternation Beyond Single-Agent Limits 

**Authors**: Changye Li, Zijian Zhang, Muheng Li, Yiduo Pang, Tingjun Wu, Menghan Li, Yitong Liu, Hongkai Guo, Mengchi Chen, Chenyu He, Kaisheng Ma, Wei Xu, Song Han, Sihao Liu, Ligeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13494)  

**Abstract**: Existing LLM-based agent approaches for long-horizon machine learning engineering (MLE) often use context engineering, implemented through custom harnesses that manage tools, skills, and accumulated experience. However, we find that as single-session execution lengthens, the agent's reasoning intensity declines alongside slowing progress, summarized as long-horizon deceleration, and the benefits of these custom harnesses vanish as the underlying model improves, summarized as harness staleness. These findings suggest that their benefits may not persist across versions, which motivates a workflow-level approach in which agents build on one another's work across sessions. We model these alternating agent sessions as a context-reset semi-Markov decision process (CR-SMDP) and establish sufficient conditions for alternation to exceed single-agent reference limits, derive an explicit experiment-budget bound, and characterize when session caps preserve this advantage. Guided by this analysis, we introduce Humanize MLE Agents (HMA), which alternates two agents on a shared workspace, limits each session to at most k experiments and resets the context between sessions, without additional task-specific harness or prior knowledge supplements. Across 75 Kaggle tasks in MLE-bench, HMA reaches a state-of-the-art any-medal rate of 78.2%, exceeding the strongest open-source baseline by 8.0 points with a quarter of its time budget. Moreover, among six configurations spanning five heterogeneous agent pairs, HMA improves any-medal rate over the constituent baselines by 6.4 points on average, demonstrating cross-pair generality. A supplementary case on kernel optimization reaches 878 cycles, within 1.6% of the community leaderboard leader, demonstrating applicability beyond MLE. 

---
# Branching On-Policy Distillation 

**Authors**: Dongyang Chen, Shudong Liu, Shubo Li, Duoxun Tang, Jinwei Liang, Zehuan Wu, Lewei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13481)  

**Abstract**: On-policy distillation (OPD) provides teacher supervision on student-generated trajectories, but a single trajectory realizes only one continuation at each decision, limiting supervision coverage under a finite sampling budget. We propose BranchOPD (Branching On-Policy Distillation), which selects informative decision points using student uncertainty and teacher-student disagreement, resamples student-policy continuations from shared prefixes, and applies token-level teacher supervision to both main trajectories and branches. BranchOPD operates at token-level decisions in mathematics and within-turn decisions in agent tasks. Shared-prefix reuse reduces student-generated token costs by approximately 19% on ALFWorld and 31% on WebShop relative to seven independent trajectories. We also introduce protocol-boundary repair (PBR), which uses targeted supervision under protocol constraints to mitigate repetition and wasted generation. BranchOPD outperforms Vanilla OPD and Multi-Rollout OPD (n=7) across all evaluated benchmarks, with gains over Vanilla OPD of 18.55 percentage points in mean mathematics accuracy over 16 sampled answers (avg@16) and 13.8 points in mean ALFWorld ID/OOD success. Mechanistic analyses and selection ablations show that branching at uncertain decisions expands subsequent behavior coverage and outperforms random selection and turn-initial branching. 

---
# A Local Agentic AI Framework to Generate Edge Case Scenarios for Safety Assurance 

**Authors**: Victoria J. Hodge, Colin Paterson, Rob Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2610.13479)  

**Abstract**: Generating edge case scenarios is crucial for training, testing and safety assuring autonomous systems. This identifies hazardous failures to increase system reliability and robustness, ensures the system can still function safely on the boundaries of its operational design domain and learns the expected system behaviour for these edge cases. The rapid advancement in the generation and analysis capabilities of language models (LMs) has expanded their applicability and recent work has investigated LM scenario generation. However, many recent approaches rely on human-in-the-loop or rule-based strategies. This restricts the LM's ability to generate diverse and edge case scenarios; the long tail of the distribution. Our work uses local LMs in a laptop-based agentic AI framework for adaptive scenario generation. This is cheap, fully-controlled, ensures data privacy and security. It uses think, act, observe, update, and continue loops with lightweight JSON encoding to generate scenarios. We use the system requirements, system description and environment description as knowledge bases in contextual prompts for an LM to ensure scenario realism. A second, independent LM critiques each scenario using domain specific criteria so it can be refined. The refined scenarios are run in simulation and evaluation metrics collected. These metric scores are analysed and similar scenarios are created for edge cases (exploitation) or novel scenarios are generated to explore the scenario space. Post-hoc clustering using metric scores and thematic-type analysis of the rich-text scenario descriptions produced by the LM pinpoints functional gaps and their triggering conditions. This creates traceable evidence for the safety assurance case. Our work highlights the potential of a flexible and adaptive local agentic AI framework for automating edge case generation and producing traceable safety assurance evidence. 

---
# SqueezeBench: Saturation-Resilient Evaluation of Mathematical Capability 

**Authors**: Pablo Manrique, Hai Xia, Stefan Szeider  

**Link**: [PDF](https://arxiv.org/pdf/2610.13461)  

**Abstract**: AI systems are making rapid progress in mathematics, but this progress is hard to measure. Benchmarks of solved problems saturate, and in benchmarks of open problems each problem can be resolved for the first time only once. We present SqueezeBench, a benchmark that is resilient to saturation: a success raises the bar instead of removing the problem. Each problem fixes an integer function $f(n)$ whose values are known only for small $n$, and the benchmark keeps a lower and an upper record on it. A record is a computable Lean function with a machine-checked proof that it bounds $f$. A submission proposes a new function and proves that it bounds $f$, that it is no worse than the current record at any $n$, and that it is strictly better at some $n$. Once accepted, it becomes the record that the next submission has to beat. The system thus chooses the bound as well as its proof, and it may improve the record at a single $n$ or its growth rate. At every $n$, the gap between the two records shows how far the certified bounds are from determining $f(n)$. Every proof is checked by two independent Lean kernels, so no referee needs to check the mathematics. The checks also reject a function that computes $f$ by exhaustive search at small $n$, and a maintainer only reads the submitter's report and looks for such a search at larger $n$. We release ten problems from extremal combinatorics with twenty proved initial records and the admission tools. As a baseline, agents based on a frontier language model, with at most one hour per record, had 82 submissions accepted and improved every record, without going beyond the published literature. Agents based on two newer models then started from these records, had 60 more submissions accepted and improved 19 of the 20 records again. For eight of the ten problems, at least one record is still asymptotically behind the best published bound. 

---
# Agentic Large Language Models 

**Authors**: Max van Duijn, Michiel van der Meer, Aske Plaat, Niki van Stein  

**Link**: [PDF](https://arxiv.org/pdf/2610.13455)  

**Abstract**: Large language models (LLMs) have taken the world by storm. Their ability to communicate with us in our own language, and to reason about our world, has made them a favorite tool for writing, answering questions, and much more. Their answers already influence our decisions and our actions. What would happen if LLMs themselves could act in the world directly? What would happen if LLMs could interact with each other? Agentic LLMs combine the power of large language models with the ability to act in the real world. In this book we will discuss how agentic LLMs work, how they act, how they interact, and what this means for us, as users. 

---
# Rollout-Time Intervention: Unlocking Agentic Reinforcement Learning Beyond Policy Limits 

**Authors**: Chen Xi, Ding Ruixue, Wang Zhigaoyuan, Zhang Fanrui, Zhang Qiang, Chen Boli, Wang Shihang, Wang Tianfu, Ren Ailin, Wang Pengyong, Xie Pengjun, Xiong Hui  

**Link**: [PDF](https://arxiv.org/pdf/2610.13438)  

**Abstract**: Reinforcement learning (RL) has emerged as a key paradigm for improving large language model (LLM) agents, enabling stronger reasoning, planning, and tool-use in interactive environments. However, existing agent RL methods remain fundamentally constrained by an on-policy reasoning bottleneck: when tasks exceed the model's current capability boundary, effective trajectories become extremely sparse and training often collapses. We introduce Rollout-Time Intervention (RTI), a technique designed to elicit successful trajectories beyond the policy's intrinsic exploration capacity. RTI operates by leveraging reward-side evidence (i.e., evaluation rubrics) during rollout to convert erroneous trajectories into successful ones, providing effective learning signals for tasks that the policy rarely solves on its own. Although intervention trajectories are inherently off-policy, we find that they can effectively guide policy optimization when trajectories with excessive distributional shifts are filtered out. Extensive experiments across five benchmarks demonstrate that RTI consistently outperforms RL baselines, enabling the post-trained policy to approach the performance of substantially stronger contemporary LLMs and, more importantly, transcend its original capability boundary. The code is available at this https URL. 

---
# Local Diagnostics of Learned Event Representations for Anomaly Detection at the HL-LHC 

**Authors**: Donatella Genovese, Philip Ploner, Simone Scardapane, Stefano Giagu, Thea Klaeboe Aarrestad  

**Link**: [PDF](https://arxiv.org/pdf/2610.13437)  

**Abstract**: Anomaly detection, the identification of collision events that deviate from Standard Model expectations, is one of the most promising model-agnostic strategies for new-physics searches at the Large Hadron Collider. Searching for rare events in learned low-dimensional representations removes the need to commit to hand-crafted observables a priori, but sacrifices the immediate physical interpretation those observables provide. For anomaly detection to enter a discovery pipeline, flagged events must therefore be diagnosed locally, both to distinguish potential new physics from mismodeling or detector effects and to identify observables for targeted follow-up searches. We evaluate supervised and self-supervised neural embedding strategies on COLLIDE-2V, a 750-million-event dataset simulated under realistic High-Luminosity LHC conditions, using the reconstruction error of an autoencoder trained in the learned latent space as the anomaly score. Across the configurations and benchmark signals considered, supervised objectives, which exploit simulation process labels during training, generally outperform their self-supervised counterparts and, in their best configurations, improve on an autoencoder trained directly on kinematic features, raising the signal efficiency at 10% background acceptance on all held-out processes. We further introduce a local diagnostic pipeline that fits a Gaussian Mixture Model to the SM latent space, profiles each component through high-level observables without process labels, localizes the flagged events, and ranks observables by their local discriminating power. The ranking recovers the defining kinematics of the held-out processes, while correlating the anomaly score with the same observables quantifies how closely the score tracks them, revealing cases where the most discriminating observable is not the one the score follows most closely. 

---
# Web$hell: A Unified Interface for Agentic Web Access 

**Authors**: Sunghwan Kim, Deogyong Kim, Sangam Lee, Donguk Kwon, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.13416)  

**Abstract**: Large language model (LLM) agents are becoming direct users of the open web. As they take on increasingly complex and information-intensive tasks, effective web access is becoming a core capability for agentic problem solving. Existing approaches have advanced agentic web access through different access models, including search-augmented agents, browser-based agents, and, more recently, filesystem-based agents. Nevertheless, as users expect agents to complete long-horizon, end-to-end tasks, they increasingly need to conduct broad information gathering, perform browser actions, and revisit acquired evidence within a single trajectory. Yet our preliminary analyses reveal that existing access models struggle to meet these demands. They primarily support either information gathering or browser interaction, while combining their capabilities can involve substantial costs or performance trade-offs. To this end, we present Web\$hell, an interface that unifies web access under a single access model. Using a common schema defined by WebContract, WebCompiler compiles live webpages into addressable information and action objects. These objects are maintained in WebCache and handled through a shared set of operations, providing agents with consistent, operable, and persistent web access. Through extensive experiments across web search, web navigation, and deep research benchmarks, we show that Web\$hell achieves strong task performance at lower cost. Our results further indicate the importance of interface design for effective web access, with Web\$hell opening new possibilities for more capable and efficient LLM agents on the web. 

---
# RegulaTutor: A Theory-Informed Framework for Long-Term Personalized Learning 

**Authors**: Yonggang Li, Boqian Han, Rui Sheng, Yushi Sun, Zixin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.13399)  

**Abstract**: Long-term personalized tutoring requires language models to translate a learner's evolving history into appropriate instructional support. Existing LLM-based tutoring systems use memory and retrieval to personalize responses, but the pedagogical rationale linking historical evidence to teaching strategies is not always explicit. We introduce RegulaTutor, a theory-informed framework for long-term personalized tutoring that connects historical evidence, learner interpretation, and learner-facing instruction. Guided by self-regulated learning (SRL) research, its tutoring harness operationalizes a cognitive, affective, metacognitive, and motivational (CAMM) framework for deriving evidence-supported learner interpretations and teaching policies. The resulting policies retain links to their supporting evidence and are updated as new observations arrive. This staged workflow also supports systematic refinement through PairRoute, an offline procedure that uses intermediate execution records to distinguish harness, model, and mixed failures. PairRoute proposes and evaluates targeted harness revisions, then constructs reviewed stage-level supervision for confirmed model failures remaining under the selected harness. We focus this refinement on historical evidence acquisition, which underpins personalized instruction and allows outputs to be checked against learning records and reference answers. RegulaTutor improves historical-evidence accuracy, learner diagnosis, and judged teaching quality on average across evaluated model configurations. PairRoute results show an endpoint improvement in standalone-model historical-evidence accuracy over the refinement trajectory. Together, these results highlight RegulaTutor's potential to support lifelong personalized education through instructional policies that evolve with learner histories and targeted improvements to the tutoring system. 

---
# SWE-Critix: Post-Training Generalist Verifiers for Assessing Coding-Agent Trajectories 

**Authors**: Jiajun Hu, Chun Yong Chong  

**Link**: [PDF](https://arxiv.org/pdf/2610.13356)  

**Abstract**: Coding agents are increasingly deployed for end-to-end software engineering tasks, where a readily available oracle evaluator that defines the success criteria is often absent. This motivates verifications for coding agents that can reliably judge and explain their behavior. However, verification is non-trivial as coding agents are increasingly capable of solving long-horizon software engineering tasks. Their execution trajectories can be extremely long, containing hundreds of iterative interactions with code repositories and external tools. Existing verifiers either rely on general-purpose LLMs without specialized training or train dedicated critics that predominantly focus on producing scalar rewards, highlighting the relatively limited exploration of training-based generalist verification. We introduce SWE-Critix, a post-training pipeline for training generalist verifiers of coding-agent trajectories. Given a task and its execution trajectory, the learned verifier predicts the completion label and generates a natural-language analysis that explains its judgment. Our pipeline combines supervised fine-tuning (SFT) with reinforcement learning (RL) and trains a series of 30B models on trajectories from CodeForge-Preview. On a held-out test set, SWE-Critix improves classification accuracy from 57% for the base Qwen3-30B-A3B reasoning model to 66% after SFT and 78% after RL. We release our models, datasets, and training code to facilitate further research. 

---
# AlphaNIST: Emergent Assembly Skills from Binary Task Rewards 

**Authors**: Octi Zhang, Iretiayo Akinola, Rosario Scalise, Mateo Guaman Castro, Patrick Yin, Tobias Widmer, Miles Macklin, Chris Amevor, Dylan Turpin, Dirk Van Gelder, Jan Carius, Ankur Handa, Gavriel State, Yunrong Guo, Yashraj Narang  

**Link**: [PDF](https://arxiv.org/pdf/2610.13354)  

**Abstract**: The NIST Assembly Task Board #1 remains a challenging benchmark for learning-based methods in robotic manipulation. Over short horizons, it demands precise, contact-rich control of small, tight-fitting parts. Over long horizons, it requires recovery and coordination across extended action sequences. Demonstrations, task decomposition, and dense progress rewards have enabled research advances by providing guidance on how to assemble. But how much of the strategy must be specified? The progression from AlphaGo to AlphaGo Zero showed that RL at scale can discover superhuman strategies without prescribing how to win. AlphaNIST asks whether a sparse binary task reward and task diversity admitted by this task-agnostic reward can similarly let a robot discover how to assemble. In simulation, we train a single policy from scratch across all 20 assembly tasks on the NIST Assembly Task Board #1, reaching 98.2% overall training success without demonstrations, progress shaping, scripted primitives, or per-task tuning. The framework also trains policies to complete ten assembly pairs in a single episode without a prescribed order, with task success rewarded only when all ten pairs are simultaneously complete. The learned policies discover dexterous and planning-like behaviors: choosing grasps for later assembly, using surrounding geometry to reorient parts, and preparing one part, assembling another, then returning to finish the first without an explicit planner. For videos, see our website: this https URL 

---
# Masked Diffusion Language Agents for Tool-Integrated Chemical Reasoning 

**Authors**: Mengdi Liu, Chenghao Jia, Hong Chang, Shiguang Shan  

**Link**: [PDF](https://arxiv.org/pdf/2610.13330)  

**Abstract**: Designing tool-integrated reasoning agent systems for complex chemical tasks remains a fundamental challenge. Although recent chemical agents have demonstrated promising performance through the integration of external tools, they are still built on the autoregressive paradigm, which is inherently constrained by causal attention and left-to-right generation. As a result, they often struggle with bidirectional molecular understanding, global coordinated tool planning and multi-source evidence integration. To address this challenge, we introduce ChemDiffAgent, the first masked diffusion language agent for tool-integrated chemical reasoning. It reformulates chemical reasoning as iterative denoising over interleaved reasoning and tool-use trajectories, enabling tool-use decisions to be jointly determined under full-trajectory context and thereby better aligning the agent's decision boundary with its knowledge boundary. We further analyze how this paradigm benefits agentic reasoning through harder infilling subproblems during training and flexible arbitrary-order decoding during inference. Then we develop a two-stage post-training framework, comprising agentic supervised fine-tuning and variance-reduced preference optimization, to enhance tool-use and reasoning capabilities. We also construct a comprehensive benchmark covering six chemistry tasks across both single-turn and multi-turn tool-use scenarios. Under matched budgets and evaluation, ChemDiffAgent consistently outperforms autoregressive agents, achieving an overall improvement of approximately 20%. Notably, our 8B model achieves performance comparable to, in some cases surpassing, significantly larger frontier models, such as GPT 5.5 and Claude Opus 4.7. Code is available at this https URL. 

---
# BoundaryEvo: Data-Model Co-Evolution at the Capability Boundary for Long-Chain Reasoning Distillation 

**Authors**: Jiaming Su, Suorong Yang, Xudong Li, Tengchao Yang, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.13285)  

**Abstract**: Distilling long-chain reasoning trajectories from large teacher models provides an effective way to improve the reasoning ability of small language models. However, existing methods typically select a fixed training set before distillation, overlooking that the usefulness of a reasoning example changes as the student evolves. This mismatch becomes especially costly when only a small number of teacher trajectories can be used for training. Besides, examples already mastered provide little new supervision, while examples far beyond the capability of student may be difficult to learn from. Training data should therefore adapt as model capability evolves, yet reliable data selection is difficult when the student is initially weak. To address these challenges, our key idea is to first establish the basic reasoning capability and then adapt the training data to the evolving capability boundary of the student. We propose BoundaryEvo, a two-stage data--model co-evolution framework. In Capability Priming, we directly distill a small set of teacher-generated long reasoning trajectories without using a separate model to judge and filter them, allowing the student to acquire an initial reasoning capability before capability-aware data selection. In Capability-Boundary Co-Evolution, we repeatedly select questions that the current student has not mastered but the expert can reliably solve, and update the training data as the student evolves. With only 1,000 training examples, BoundaryEvo consistently outperforms static and dynamic data selection baselines across 11 reasoning benchmarks, raising the original Qwen3.5-4B accuracy from 62.9 to 75.3 on IMO-AnswerBench and from 26.8 to 46.0 on AMO-Bench. The code and data will be made publicly available. 

---
# Forgetting is Not Always Bad: A Neuro-Inspired Memory Repair Mechanism for Poisoned LLM Agents 

**Authors**: Lei Liu, Yunji Liang, Xiaowen Zhang, Jingqi Liu, Qi Li, Bin Guo, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13267)  

**Abstract**: Large language model (LLM) agents are highly vulnerable to memory-poisoning attacks. Existing defenses primarily rely on external modules for either static memory isolation or continuous online auditing, resulting in low memory efficiency and high computational overhead. To address these limitations, inspired by neuroscientific mechanisms that weak reactivation can destabilize memories and induce decay in the absence of reinforcement, we propose an agent-intrinsic training-free memory repair framework, DREAM (Dynamic Reactivation for Engram Attenuation in Memory), that enables selective functional forgetting of poisoned memories. Specifically, DREAM implements a three-stage pipeline: perturbation-induced implicit reactivation to reveal the structural fragility of malicious memories, adaptive anomaly diagnosis based on multi-dimensional activation patterns to detect topological anomalies, and a dynamic memory repair module that selectively suppresses harmful memories while reinforcing benign ones. Extensive experiments across diverse attackers and real-world tasks demonstrate that DREAM reduces attack success rates by over 95\% against backdoor poisoning while preserving strong benign utility. DREAM also achieves competitive robustness against injection attacks. In terms of runtime and token consumption, DREAM achieves up to a 2.13$\times$ speedup and a 37.7\% reduction in token consumption compared with A-MemGuard. Furthermore, DREAM also achieves a task success rate of 92.96\% on poisoned multi-agent systems. 

---
# JANUS: Enhancing Long-Horizon Autonomous Research with Co-Evolving World Models 

**Authors**: Mengdi Liu, Chenghao Jia, Hong Chang, Mingjie He, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.13194)  

**Abstract**: Long-horizon autonomous research has shown promise in accelerating scientific progress through sustained cycles of hypothesis generation, implementation, and validation. However, this Generate-Execute-Feedback paradigm relies on reactive trial-and-error, where costly and time-consuming executions introduce severe feedback delays, throttling evolutionary progress under explosively expanding search space. To address this challenge, we propose JANUS, a framework that unifies future outcomes forecasting and past experience reflection through a Predict-Verify-Evolve loop driven by a co-evolving world model. Within an evolving graph, the research agent proposes candidates while world model scores them by forecasting value improvement, information gain, and exploration utility, then accordingly selects, refines, or prunes branches prior to execution. After receiving ground-truth feedback, JANUS distills experience into evolving skills: hierarchical skills for research agent, exploration-strategy skills for world model, and their coordination protocols. This mechanism enables continual co-evolution: the research agent sharpens candidate generation while the world model refines its predictive judgment over extended horizons. Extensive evaluations on MLS-Bench-Lite and real research tasks demonstrate that JANUS significantly outperforms state-of-the-art baselines, achieving both higher evolving efficiency and ceiling. 

---
# AdaC-GRPO: Towards Mastering Long-Horizon Visual Sudoku Puzzles via Adaptive Curriculum Learning 

**Authors**: Sida Li, Yuanxin Liu, Linli Yao, Shicheng Li, Xu Sun, Haoli Bai, Xiaohui Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.13184)  

**Abstract**: While reinforcement learning with verifiable reward (RLVR) enhances reasoning in large language models, its efficacy in long-horizon multimodal tasks remains underexplored. Visual Sudoku puzzles pose a representative challenge, requiring multi-step deductive reasoning over both local constraints and global dependency chains. To this end, we introduce FG-Sudoku, a fine-grained benchmark categorized by grid size and required deductive steps. Evaluations reveal that frontier MLLMs fail on complex long-range dependencies (e.g., 9*9 puzzles). During post-training, this near-zero success rate causes severe reward sparsity, halting standard RLVR. We propose Adaptive Curriculum Group Relative Policy Optimization (AdaC-GRPO) to overcome this. Unlike rigid or prior-based schedules, AdaC-GRPO dynamically constructs training batches by tracking the model's "Zone of Proximal Development" via online rollout accuracy. Experiments demonstrate that AdaC-GRPO enables Qwen2.5-VL-7B to approach the performance of Gemini-2.5-Flash while bridging 62.2% of the overall accuracy gap to Gemini-2.5-Pro, with its generalizability further validated across InternVL-3.5 and Qwen3-VL. 

---
# RoleWorld: Knowledge Transfer in World Modeling with Functional Role Abstractions 

**Authors**: Boyuan Xiao, Pu Wang, Ji Feng, Yao-Xiang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2610.13172)  

**Abstract**: World models enable large language model (LLM) agents to anticipate action effects and make informed decisions. For generalization, it is desirable for such world models to transfer efficiently to new environments with similar interaction structure and underlying dynamics. Recent approaches encode world dynamics as symbolic rules or executable programs, whose compositional structure in principle affords transfer, but still organize such knowledge around environment-specific objects. To address this limitation, we introduce RoleWorld, a symbolic world modeling approach which uses reusable prototypes and environment-specific parameter values as functional role abstractions to represent how objects behave in interactions. From interaction trajectories, a teacher LLM extracts role prototypes and executable transition rules defined over role instances, together with a skill that guides the actor LLM to use role information in decision making. In a target world, without the teacher, the actor uses this skill together with its current role assignments and parameter estimates, and actively tests a role instance hypothesis through interaction when its uncertainty affects the current decision. Feedback from each action refines role assignments and estimates parameter values online, with no LLM parameter updates and the transition rules held fixed. Experiments on MARS benchmark show that RoleWorld enables robust generalization across environments with significantly altered object behavior, suggesting that functional role abstractions can serve as an effective component for knowledge transfer in world modeling. Project Page: this https URL. 

---
# Quality, Not Just Outcome: A Structural Scoring Framework for Selecting Coding-Agent Trajectories for Supervised Fine-Tuning 

**Authors**: Sida Li, Yuanxin Liu, Shicheng Li, Linli Yao, Xu Sun, Haoli Bai, Xiaohui Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.13152)  

**Abstract**: Open coding agents are commonly trained through supervised fine-tuning on interaction trajectories whose final patches pass the benchmark test suite. However, this outcome-level filtering criterion ignores process quality: trajectories that eventually succeed may still contain inefficient interaction patterns such as repetitive file inspection, unstable tool usage, or delayed modification behavior, all of which become part of the imitation target. We propose TQS (Trajectory Quality Score), a structural scoring framework for coding-agent trajectories that operates directly on trajectory structure, requires no LLM judge, and runs in linear time. TQS characterizes trajectory quality through structural signals organized around structured convergence: information focus, intent--action alignment, and policy dynamics. Using 155,144 successful trajectories from a public open-license corpus, we rank trajectories by TQS and fine-tune coding agents on filtered subsets. Models trained on top-ranked trajectories consistently outperform those trained on random or bottom-ranked subsets on SWE-Bench-Verified. Our results suggest that process-level trajectory quality is an important factor in coding-agent supervision and that structural trajectory filtering provides an effective and scalable alternative to LLM-based data selection. 

---
# VIGOR: Verifiable Industrial Geometry Optimization and Repair in Scene Programs 

**Authors**: Fuqiang Wang, Zheng Guo, Xiaoming Wu, Hanlin Kong, Nayu Liu, Chuanting Zhang, Daojun Liang, Jingxuan Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.13145)  

**Abstract**: Large language models (LLMs) increasingly generate structured layouts and executable scene programs from natural-language instructions. However, executability does not imply engineering feasibility: an industrial scene program may execute successfully while its final state still violates coupled boundary, collision, clearance, aisle, or structural constraints. We identify this mismatch as the executability-engineering-feasibility gap, where locally plausible placements can remain globally infeasible because corrections to one device may introduce violations elsewhere. To make this gap measurable in a controlled setting, we construct an execution-grounded industrial scene-program dataset with 1,749 constructible scenarios, complete executable programs, and 94 final-state geometric checks per scenario. We further propose VIGOR, which combines supervised scene-program generation with execution-grounded verification and staged geometric refinement. Supervised fine-tuning establishes complete program generation, local projection resolves independently correctable violations, and joint conflict-graph repair coordinates residual multi-object conflicts while preserving structural relations. Experiments show that VIGOR achieves 96.86% Complete Feasibility and an Overall score of 91.20, outperforming the strongest evaluated baseline by 10.42 points in Overall. These results show that, under controlled industrial layout conditions, reliable scene-program generation requires not only executable programs, but also explicit verification and coordinated resolution of coupled engineering constraints. 

---
# Competence-Aware Orchestration for LLM-Based Multi-Agent Systems 

**Authors**: Songmiao Wang, Yuchen Wang, Chenglin Yu, Hongxia Yang, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.13144)  

**Abstract**: LLM-based multi-agent systems (MAS) have emerged as a promising paradigm for enhancing complex reasoning through role specialization and collaboration, yet a key challenge remains: how can collaboration be adapted to a backbone's competence on the current query? This problem matters because more collaboration is not always better--richer coordination may help on difficult cases, but it also incurs additional token cost and redundant interactions. We propose a two-stage competence-aware framework for automatic MAS orchestration. It first models the LLM backbone's task-conditioned competence for the current query, and then uses this signal to adapt role activation and the communication graph. Across six benchmarks, our method achieves an average score of 93.81%. Experiments with six LLM backbones of varying competence levels further show that our method adjusts collaboration according to the inferred competence: for higher-competence backbones, it improves average accuracy from 87.22 to 89.97 while reducing token cost by 36.26%; for lower-competence backbones, it yields a further average gain of 4.58 points over existing methods. The same trend also holds on unseen backbones. 

---
# DP-MIND: Differentially private memory-guided Long context Inference 

**Authors**: Zhihao Liu, Yitian Cheng, Shuang Liu, Rui Wen, Yuke Hu, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.13132)  

**Abstract**: LLM-as-a-Service has become a dominant paradigm for accessing large language models without requiring local deployment. However, the process raises severe privacy concerns when user queries may include sensitive or confidential contexts. Differential privacy (DP) provides a principled approach to protecting such inputs, but existing DP inference methods are primarily designed for short- or moderate-context settings and struggle with million-token-scale inputs due to limited context capacity and severe utility degradation under DP perturbation.
In this paper, we propose DP-MIND, a memory-guided framework that scales differentially private inference to million-token contexts. DP-MIND enables private long-context inference by locally perturbing the input and processing it chunk by chunk with a compact memory that preserves task-relevant information across the sequence. To make this memory robust to DP noise, we optimize the memory construction policy directly on perturbed inputs using reinforcement learning, guided by both final answer correctness and memory consistency. We further recover perturbation-corrupted answer spans through a local denoising module without exposing the original context to the external model. We provide theoretical privacy analysis of DP-MIND and further conduct extensive empirical analysis across multiple long-context benchmarks with context lengths up to the million-token scale, demonstrating that DP-MIND consistently outperforms existing DP and long-context inference baselines in both scalability and utility. 

---
# Looking Ahead and Back: Anticipatory Generative Modeling for Road-Network Trajectories 

**Authors**: Zhenhao Huang, Li Sun, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13131)  

**Abstract**: Trajectory generation underpins urban applications such as traffic management and urban planning. Within road networks, vehicle movement along road segments is largely deterministic, whereas transitions at intersections are critical--multiple outgoing paths may appear equally plausible, yet they can lead to substantially different downstream routes. Existing generative methods such as diffusion models simulate location sequences and fail short of next transition selection, even producing trajectories that drift off the road network. To fill this gap, we advocate a fundamental shift: moving beyond location simulation toward behavior modeling on the road network. In this paper, we revisit trajectory generation through the lens of generative route planning, and introduce a new graph-constrained generative paradigm, Anticipatory Generative Modeling, which resolves next transition selection by incorporating the Anticipation of how the future route may unfold toward the destination. Concretely, we propose a novel generative model named Foresight that formulates a probabilistic model for an "anticipate--select--update--re-anticipate" loop that enables longer-term foresight while continuing sequential generation along the historical trajectory. Extensive experiments demonstrate that Foresight wins state-of-the-art baselines. 

---
# Locate, Judge, Read: Role-Specialized Tool Agents over a Knowledge-Point Index for Open-Domain Multimodal Document QA 

**Authors**: Wenlin Lin, Fanzhe Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.13128)  

**Abstract**: Question answering over large heterogeneous document collections requires first finding the right passages and then reading them, yet current retrieval-augmented agents give one model both kinds of tools and let it schedule itself. We present KnowForge, a pipeline separated by job: a hybrid-retrieval knowledge-point index; a Locator agent that can only search; a code-scheduled bridge judge that forces the missing second hop; a typed, size-bounded evidence pool that merges every search's ranking; a Deep Reader agent that can only read text, images and tables; and a synthesizer with a format checker. With a self-hosted Qwen3.6-35B-A3B (3B active), KnowForge reaches F1 47.5 on the full open-domain M3DocVQA (2,441 questions, 3,368 PDFs), +11.0 over the published M3DocRAG system and +7.1 over one-shot retrieve-and-answer on the same index and model, and 53.5 with DeepSeek-V4-Flash. On MultiHop-RAG it reaches 73.5 accuracy, above the strongest published RAG and GraphRAG results (71.2); on the single-document MMLongBench-Doc it lifts the same model from 34.8 to 47.0, level with GPT-4o reading page images (46.3). Per-job ablations on the full benchmarks attribute the gain: the Deep Reader is worth 3.8, 9.8 and 0 points on the three corpora, tracking how visual each is; merging both agents into one with all tools and twice the rounds loses 1.9 F1 because it keeps searching instead of reading; a round-budget sweep shows a step at four Locator rounds and a plateau beyond; query latency scales with model calls rather than corpus size. We serve the three indexed benchmarks as read-only knowledge bases (free registration) and release all prompts and the runner code: this https URL 

---
# DRYLAB-Bench: Benchmarking the Biosecurity Risk of Agentic LLM Mutation Design 

**Authors**: Yuzhen Wang, Chunxiao Li, Baichen Le, Lewen Yan, Jie Zhang, Yucheng T. Yang, Xing-Ming Zhao, Lijun Li, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2610.13116)  

**Abstract**: As large language models (LLMs) acquire advanced biological design capabilities, they offer substantial research benefits but simultaneously introduce severe dual-use risks, such as engineering enhanced pathogens or drug-resistant bacteria. While biosecurity frameworks recognize these threats, the actual risk of LLMs executing dangerous biological modifications remains unquantified, particularly when augmented with specialized biological design tools (BDTs). Existing evaluations primarily assess hazardous knowledge possession rather than practical application, overlook risk amplification from BDTs, and rely on unreliable, expensive, or hazardous validation methods. To address these gaps, we introduce DRYLAB-Bench (Design Risk Yielded by LLM Agent in Biodesign), the first comprehensive benchmark evaluating the high-risk biological mutation design capabilities of LLM agents. Moving beyond basic question-answering, DRYLAB-Bench tasks agents with 30 genuinely hazardous mutation-design scenarios distilled from biosecurity guidelines. It evaluates models in a multi-round, tool-augmented environment where LLMs orchestrate BDTs to iteratively refine candidates. Crucially, it scores designs directly against 187,695 empirical wet-lab measurements from ProteinGym and ViroGym using our proposed Biosecurity Risk Score (BRS), quantifying risk-associated effects while enabling scalable and safe ground-truth verification. Evaluations of 15 frontier models reveal substantial baseline threats: without tools, 67.6% of one-shot designs exceed the average measured variant, and 21.4% reach the highest-risk decile. Equipping models with BDTs significantly amplifies this risk, raising the pooled risk by \(\Delta BRS=+0.116\) and the top-decile share to 23.5%. Code: this https URL 

---
# Does Location Matter? A Black-Box Causal Audit Reveals Model-Dependent Audio Grounding in LALM Hallucination 

**Authors**: Mohammad Khalooei, Mohammad Sabokrou  

**Link**: [PDF](https://arxiv.org/pdf/2610.13112)  

**Abstract**: Large audio-language models (LALMs) frequently exhibit object hallucination, affirming absent sounds. The dominant mitigation, Audio-Aware Decoding (AAD), contrasts logits under real audio against an all-zero "blank" waveform, flagged by prior review as out-of-distribution. We introduce a black-box causal test that isolates the claimed evidence itself: localize a claim's evidence region via CLAP similarity, remove it by spectral gating, and compare against a random-window control across a 2x2x2 grid on three LALMs. On Qwen2-Audio-7B, localized ablation's raw AUROC trails blank's (0.613 vs. 0.721); this is a duration confound, not a localization failure, since blank removes the entire clip versus the grid's 2-4 s. Holding duration fixed, localization beats random removal at every cell on both models. Raw detection value remains model-dependent, especially on Audio Flamingo 3. The same evidence corrects decoding: contrastive decoding combining blank and language-prior contrasts, gated and ensembled, raises Qwen2-Audio accuracy from 57.1% to 71.9%, beating AAD alone. A unified causal-grounding profile does not transfer between models, indicating different grounding regimes. The source code and a detailed report of our findings, including identified limitations, are available at this https URL. 

---
# VETR: Execution-Grounded Self-Improving Agent for Symbolic Graphics Reasoning 

**Authors**: Yingbo Ma, Kuangshi Ai, Gordon Dai, Chenglong Yu, Chaoli Wang, Hua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.13098)  

**Abstract**: Visual agents for symbolic graphics reasoning depend on executable world representations, yet continued improvement remains bottlenecked by ambiguous supervision from final renders. Recent agentic systems refine scenes through repeated interaction, but they lack mechanisms to recursively extract actionable evidence from execution traces, overlooking which proposed symbolic edits survived, disappeared, or interfered during program execution. To address this, we propose VETR (Verified Execution Trajectories for Refinement), a framework that enables agentic recursive self-improvement over executable world representations for symbolic graphics reasoning. Within each task, VETR retains aligned program/render records with bounded runtime observations and introduces an execution refiner that compiles observed writes and state changes into constrained alternative programs. The refiner validates their endpoints and preservation conditions against captured runtime state; the ordinary visual verifier then assesses goal relevance. Repair construction makes no additional language-model calls, but can require additional executions. Evaluations on BlenderGym (280 tasks) and BlenderBench (27 released tasks) with GPT-5.6-luna and Qwen3.8-27B show that VETR consistently outperforms existing refinement strategies, including BlenderAlchemy and VIGA. Controlled ablations evaluate the contributions of execution refinement and role-asymmetric evidence organization under fixed generator-interaction caps. 

---
# When Faster Decisions Do Not Make Better Agents: An Empirical Study of Jev in Agent Workflows 

**Authors**: Yuchen Ling, Shengcheng Yu, Zhenyu Chen, Chunrong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2610.13092)  

**Abstract**: AI agents depend on repeated decisions to accomplish tasks, with large language models (LLMs) commonly serving as their core decision makers. The recent emergence of Jev offers an opportunity to make these decisions faster and at lower cost. However, it remains unclear whether such local savings can be translated into efficient and reliable task completion. In this paper, we investigate this efficiency transfer across five agents in two representative application settings, embodied tasks in ALFWorld and mobile app use in AndroidWorld. Across 4,500 task executions, we jointly evaluate task effectiveness, decision-level and task-level resource use, and execution behavior. Experimental results show that Jev integration substantially reduces decision cost and latency across all five agents, with task-level gains varying by workflow. Task success remains close to baseline levels on ALFWorld and declines more substantially in two mobile agents. Auxiliary reasoning and additional interactions absorb decision savings, while failed recovery and premature completion limit successful delivery. Mean cost per successful outcome increases in one agent even as spending per attempt falls. These findings highlight the importance of preserving progress, supporting recovery, and verifying completion when integrating efficient decision models into agent workflows. Their practical value should be assessed through successful task delivery, recovery behavior, and resources per delivered outcome. 

---
# SWEAR: Section-Wise Efficient Audio Reasoning via Self-Rated Difficulty 

**Authors**: Yixuan Li, Minghao Lv, Xuenan Xu, Zihao Zheng, Mengyue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13081)  

**Abstract**: Recent work on large audio-language models (LALMs) explores adaptive reasoning to balance accuracy and efficiency. However, existing approaches often rely on coarse-grained difficulty estimation, overlooking the diverse factors that determine reasoning effort. We address this limitation by deriving five-dimensional difficulty scores and compressed reasoning traces from complementary offline assessments. Based on this characterization, we propose SWEAR, a difficulty-aware reasoning framework that learns difficulty signals during supervised fine-tuning and uses reinforcement learning to adapt reasoning allocation across reasoning stages. Experiments on MMAU-test-mini and MMAR show that SWEAR improves accuracy while substantially reducing generated tokens, shortening reasoning for easy instances without sacrificing performance on difficult ones. 

---
# Attention Sees, MLP Refuses: Tracing the Attention-to-MLP Safety Pathway in LLMs 

**Authors**: Zixuan Weng, Haoqi He, Weifei Jin, Yue Huang, Yang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.13078)  

**Abstract**: Safety-aligned large language models (LLMs) remain vulnerable to jailbreaks and can over-refuse benign requests. Understanding these failures requires distinguishing harmfulness recognition from the computation that writes refusal. We reframe this as a pathway question and make it measurable on residual-stream updates in decoder-only Transformers, separating the harmfulness information a module update exposes from the change it produces in readable refusal evidence. Across 18 Base-Instruct pairs spanning Qwen, Llama, and Gemma, harmfulness is decoded more readily from attention updates, particularly in early layers, whereas MLP updates contribute more strongly to harm-selective refusal writing in aligned models. Post-training strengthens MLP refusal writing in all 18 pairs without systematically improving harmfulness decodability, while positive MLP writing becomes more concentrated in deeper layers after post-training. Across six reasoning-enabled models, harmfulness remains decodable during reasoning while strong refusal writing concentrates at answer onset, a pattern that persists with reasoning disabled. Causal interventions further support this pathway: resetting downstream MLP updates to their clean values substantially reduces the refusal shift caused by upstream attention interventions. Building on these findings, we introduce Sentinel, which combines an attention-derived perception score P with an MLP-derived decision score D. Their joint profiles characterize jailbreaks and over-refusals, and conditional routing with both signals achieves a lower aggregate error rate than single-signal routing and evaluated fixed reminders across eight models. 

---
# LongHarness: Advancing Agentic Reasoning over 100M-Token Corpora via Stateful Evidence Management 

**Authors**: Songtao Huang, Zhijie Zhong, Xiaohan He, Shuaiyu Zhang, Zhiyin Yu, Li Kang, Heng Zhou, Yutao Fan, Xiangchao Yan, Bo Zhang, Shiyang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.13075)  

**Abstract**: Corpus-level long-context reasoning requires agents to acquire and integrate evidence across document collections that exceed their context windows. Existing agent harnesses enable incremental access to these corpora, but acquired evidence and evolving conclusions often remain entangled in a growing interaction history. This implicit state management can lead to two coupled failure modes: unbounded exploration, where retrieval expands the interaction history without resolving decision-relevant information gaps, and evidence-state drift, where previously acquired evidence or subsequent corrections fail to inform the current conclusion. We present LongHarness, a model-agnostic harness that makes the interaction between acquired evidence and evolving conclusions explicit. Its core mechanism, Evidence-Conclusion Coevolution, uses unresolved conclusions to guide exploration toward missing information, while newly acquired evidence strengthens or revises the conclusions that directed the search. LongHarness couples this process with Evidence Lifecycle Management, which keeps accumulated evidence current and traceable throughout the interaction. Together, these mechanisms focus retrieval on remaining information needs and keep conclusions aligned with current evidence. We evaluate LongHarness on OfficeQA Pro V1 and V2, AA-LCR, and LongBench v2 using GLM-5.3-Flash and Qwen3.8-Flash. Compared with direct reasoning using the same backbone, LongHarness improves GLM-5.3-Flash exact match on OfficeQA Pro from 15.8% to 60.2% on V1 and from 4.4% to 61.1% on V2. Against state-of-the-art agent harnesses, it surpasses Claude Code by 12.0 percentage points on V1 and Codex by 18.9 points on V2. With Qwen3.8-Flash, LongHarness reaches 61.7% on V1, outperforming Claude Code by 8.3 points and Codex by 12.0 points. 

---
# Travel Back To Move Forward: Retrospective Exploration and Evolving for Long Horizon Agents 

**Authors**: Ziang Ye, Wentao Shi, Lei Chen, Zhengzhou Cai, Yu Wang, Zhiyuan Yao, Zihan Wang, Xierui Song, Fuli Feng, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.13058)  

**Abstract**: Large language model agents increasingly operate in long-horizon environments where early mistakes can trigger cascading failures and trap agents in dead-end trajectories. Agents must therefore be able to proactively recover from such states and turn their exploratory experience into guidance for subsequent decisions. We introduce ReTRAIL, a framework that enables agents to rewind the environment to earlier states while preserving valid progress and carrying forward knowledge learned from abandoned exploration. The environment state moves backward while the agent's knowledge continues to move forward. ReTRAIL trains a shared policy with two complementary signals: State-Anchored GRPO compares alternative continuations from revisited states to improve execution, while a branch-advancement reward guides reflection through the progress of subsequent exploration. Experiments across four interactive benchmark MineSweeper, Sokoban, ALFWorld and ScienceWorld show that simply adding a rewind primitive provides limited benefit and can even degrade performance. In contrast, ReTRAIL enables agents to rewind from unproductive states and use insights from abandoned branches to guide subsequent exploration. These results demonstrate the value of coupling environment rewinding with experience reuse within an ongoing task for long-horizon task solving. The code is available at this https URL. 

---
# EmbodiedCLBench: Evaluating Continual Learning for Self-Evolving Embodied Agents 

**Authors**: Jie Huang, Yanan Chen, Ruixun Liu, Kaichen He, Xinyi He, Rui Huang, Zilong Zheng, Zhenliang Zhang, Yiwu Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2610.13021)  

**Abstract**: Continual learning requires models to accumulate past experience and transfer it to new tasks. Despite substantial research on continual learning, existing work remains largely confined to non-embodied settings, without extending to embodied environments. To address this gap, we introduce EmbodiedCLBench, the first benchmark in embodied simulation for evaluating the continual learning capability of self-evolving agents. With compositional generalization as the core evaluation principle, our benchmark evaluates whether agents can first learn from basic tasks and then recombine the acquired skills to solve complex, unseen advanced tasks. Based on EmbodiedCLBench, we conduct extensive experiments across multiple agent harnesses and learning methods, revealing consistent limitations and distinctive behaviors. Our results show that while advanced tasks can be hardly solved under zero-shot setting, the experience from basic tasks enables consistent improvement. Such benefit from the experience remains stable even if the task complexity increases. However, this improvement does not scale accordingly with the number of learning samples, and current agents cannot extract meaningful information from additional experience. Collectively, our findings highlight continual learning as a fundamental yet underexplored capability, and our benchmark offers valuable resources for designing effective self-evolving agents in embodied environments. Our code and dataset are available at this https URL. 

---
# Provenance-Aware Belief Updating in Human-AI Feedback Loops 

**Authors**: Tianyi Huang, Eric Liu, Jay Chong, Emily Yin, Wanqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.13008)  

**Abstract**: In repeated human-AI interaction, a single observation can be quoted, summarized, and returned to a model as multiple reports, making repetition look like independent corroboration. We study whether language models can use supplied report lineage to update according to evidence provenance rather than report count. We introduce Latent Quotient Evidence Training (LQET), a relational objective that combines sparse Bayesian supervision with constraints that encourage stable predictions when reports add no new evidence and consistent updates to the same new evidence across relay histories. On a controlled binary benchmark, LQET reduces new-evidence update RMSE by 41-44% and log-odds drift after eight relay steps by 89-91% relative to task-only training across Qwen3-4B and Phi-4-mini, with three paired training seeds per backbone. The gains extend to a closed-loop simulation in which model confidence controls how many relayed reports re-enter the model's context in later rounds. By grounding belief updates in supplied provenance, LQET offers a route toward dynamic human-AI systems in which influence follows new evidence, not the number of times the same evidence returns. 

---
# Scaling Empowered Agents 

**Authors**: Catherine Ji, Vivek Myers, Joshua Lin, Alexandra Wingate, Sergey Levine, Benjamin Eysenbach  

**Link**: [PDF](https://arxiv.org/pdf/2610.13002)  

**Abstract**: Empowerment measures the capacity of an agent to actively control its environment. In this work, we explain the connection between skill learning and empowerment. While skill learning is not empowerment, we introduce a simple scalable recipe for empowerment maximization: empowerment corresponds to the number of executable skills from a state, so maximizing empowerment corresponds to reaching states that are good starting points for many skills. Our recipe applies to arbitrary underlying skill learning methods. Whereas most prior work on empowerment looks at tabular settings, this recipe enables us to study empowerment at a scale where we can realize theoretically-predicted but empirically-elusive properties of empowerment (e.g., gathering resources, environment modification, survival). Experiments show that empowerment maximization unlocks interesting emergent properties, such as building a house in Crafter. Website and code at \href{this https URL}{this https url}. 

---
# Hindsight Structured Advice Distillation: Steering Executor Agents with Advisors Trained on Their Failures and Task Characteristics 

**Authors**: O. Ozan Koyluoglu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13000)  

**Abstract**: Language-model agents are used for specialized work such as financial research, healthcare and law. Such agents are often capable enough for a task but still miss its details. In finance, for example, these details include which filing reports a figure or how a company discloses a metric. Retraining the agent to learn these details is expensive, and often impossible when the agent is a closed or shared system. Writing the missing know-how into the agent's prompt makes the prompt long and identical for every input, although different inputs need different guidance. We propose to learn to generate this guidance instead. A small advisor model reads each input and writes structured advice, which is appended to the prompt of a frozen executor. The advisor is trained with hindsight, following work on distillation with privileged information. On a training pool with graded answers, we analyse the executor's failures, run it under several strategies to see which one worked on each input, and let a teacher that sees the grades and the traces write the advice that would have helped. The advisor then learns, by supervised fine-tuning, to write that advice from the input alone. Because the student is a separate small model rather than the executor, it is cheap to train and adds little latency. We call this approach hindsight structured advice distillation (HSAD). On FrontierFinance, a public benchmark of expert-written financial research queries, an 8B advisor raises the qualification rate of Samaya's proprietary research agent, which already leads the benchmark's leaderboard, from 52.9% to 56.6%. This is above the agent's own high-effort configuration, at 37% lower cost per query. We study the generalization of the proposed approach, and in particular show that HSAD also improves results on WebShop, BIRD text-to-SQL and ExpertLongBench. 

---
# Benchmarking Embedding Geometries and Clustering Algorithms for Unsupervised Patent Claim Classification 

**Authors**: Sujith Subramanian, P N Bhargav Teja, Susen Sarkar, Rajasekhara Reddy Duvvuru Muni  

**Link**: [PDF](https://arxiv.org/pdf/2610.12997)  

**Abstract**: Prior-art retrieval systems are structurally organized around classification codes, making cross-code discovery difficult. This paper asks if an unsupervised representation trained on patent text can recover relevant connections that code-restricted search cannot, and if hyperbolic space helps. We trained a hyperbolic and a matched Euclidean Variational Autoencoder (VAE) on 45,000 US patents (96 runs, 6 seeds). The hyperbolic model retrieves genuinely cross-category prior art 52% more often than the Euclidean model (18.76% vs. 12.34% of top-10 results, 3,000 queries). This represents a practical gain for the motivating search problem. The hyperbolic model agrees less with official classification (NMI 0.2392 vs. 0.3206), which we explain geometrically: the two measures capture different goals, and only one aligns with prior-art search needs. In a study of 10 technology domains (50,000 more patents), we find that a domain's classification breadth--the number of distinct CPC subgroups spanned--predicts approach performance (Pearson r=-0.642, p=0.045, N=10), and that patents per subgroup is an even stronger predictor (r=0.715, p=0.020). This provides a checkable signal for practitioners to determine where the method is most valuable. We report all findings with statistical transparency and release a complete, 91-test automated implementation. 

---
# Work2Skill: Capturing Human GUI Experience for Reusable Agent Skill Induction 

**Authors**: Yuqi Huang, Hui Li, Yifei Cao, Zikun Cai, Tianci Xie, Yangfan Gao, Pinxue Guo, Chaoyun Zhang, Bingxuan Li, Rui Zheng, Hang Yan, Ning Liao  

**Link**: [PDF](https://arxiv.org/pdf/2610.12984)  

**Abstract**: Computer-use agents can perform increasingly complex GUI tasks, yet unfamiliar software and workflows requiring human domain knowledge remain challenging. Human demonstrations convey procedural knowledge through the same GUI interaction space, but reuse requires interpreting actions in context and separating workflow rules from task-specific details. We thus introduce Work2Skill, a framework that captures human GUI experience for reusable agent skill induction. WorkRecorder provides the framework's infrastructure for collecting this experience, aligning screen recordings, mouse and keyboard events, and available accessibility-tree information. Using WorkRecorder, we construct WorkDemo-350, the first synchronized multimodal dataset of human desktop GUI demonstrations for reusable agent skill induction and downstream evaluation. Its 350 demonstrations are screened for completeness and task success. We then propose an agentic skill-induction method that inspects evidence on demand to produce reusable skills capturing workflow rules, step dependencies, and completion checks. When the same four execution models are equipped with skills induced by each method under shared execution settings, Work2Skill achieves the highest scores and success rates. Mean score improvements over the corresponding No Skill baseline are 4.67--10.54 points on MyPCBench Bounded Action and 3.11--7.50 points on a 286-task subset of OSWorld-Verified. Our code and data are available at this https URL. 

---
# Beyond Semantic IDs: Finite-Beam Reachability and Continuous-Query Generation in Generative Recommendation 

**Authors**: Yu Sun, Renkai Xiang, Zuodong Xiang, Junwei Zhou, Yike Zhang, Jianxin Zhang, Zhihao Zhu, Xiaoyu Cao, Hailu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.12977)  

**Abstract**: Generative recommendation (GR) of social media platforms commonly retrieves items by autoregressively decoding discrete Semantic IDs (SIDs) with finite-width beam search. This work examines how this output interface constrains whether held-out targets enter finite returned lists, while distinguishing catalog-wide scoreability from empirical top-K support. We introduce two continuous-query paradigms: CQG-Single generates one retrieval query in a single pass, and CQG-AR autoregressively generates continuous residual updates, retaining step-wise decoding while removing the SID item-token vocabulary, code lookup, and discrete beam expansion. These designs thereby eliminate the SID-specific reachability constraint induced by finite-beam decoding and avoid the technical challenge of aligning a discrete SID vocabulary with the pretrained backbone's vector space. Both methods make every encoded item directly scoreable through dense or approximate-nearest-neighbor retrieval. Under a unified protocol with valid-prefix constrained decoding, CQG-Single exceeds the TIGER-style baseline in all seven runs while using 55.3% as many trainable parameters. CQG-AR remains comparable on ML-1M and is numerically stronger on Beauty and Sports datasets. A controlled depth ablation further shows that additional continuous refinement does not monotonically improve shallow accuracy. These findings motivate decoupling output representation, autoregressive modeling, and retrieval budget when evaluating generative recommendation systems. 

---
# SmartSalesBench: A Resistance-Aware Multi-Turn Benchmark for Evaluating LLM Sales Agents 

**Authors**: P N Bhargav Teja, Velpula Lokesh Reddy, Madhu Basumata, Susen Sarkar, Rajasekhara Reddy Duvvuru Muni  

**Link**: [PDF](https://arxiv.org/pdf/2610.12975)  

**Abstract**: We present SmartSalesBench, the first multi-turn conversational benchmark explicitly designed to evaluate large language model (LLM) sales agents against psychologically grounded, resistant buyer personas. Existing task-oriented dialogue benchmarks assume cooperative users and measure task completion; SmartSalesBench inverts this assumption, placing agents against personas programmed to resist and challenge across conversations of up to twenty turns. The benchmark includes five agent profiles, twelve buyer personas, fifteen smartphone products across six price tiers, and four controlled experiments with 3 iterations totaling 13,680 sessions. Both sales-agent and buyer persona generations are grounded through retrieval-augmented generation (RAG). We introduce nine evaluation metrics grouped into three tiers: primary outcome metrics (CR, BRS, RIR), secondary quality metrics (SMS, PFS, ETI), and exploratory metrics planned for future reporting (ITS, ARI, SPI). Our evaluation combines automated LLM-as-judge scoring with internal human assessment (n = 50, \k{appa} = 0.71, Pearson r = 0.81 with BRS). 

---
# MathVis: Can AI Reason through Drawing? A Benchmark for Mathematical Figure Generation 

**Authors**: Jingjing Wei, Hao Zhao, Xinke Li, Linxi Yu, Xiaosong Gao, Shanshan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.12954)  

**Abstract**: This paper introduces MathVis, a benchmark designed to evaluate LLMs' ability to visualize frontier mathematical problems and perform diagram-assisted reasoning. The benchmark consists of 512 tasks, each comprising a mathematical statement, a reference diagram, executable TikZ or SVG code, a Gold Intermediate Representation (IR), atomic rubrics, and error codes. Through rigorous evaluation, GPT-6-Astra, Claude Fable 5, and Gemini 3.1 Pro Preview achieve pass@1 scores of 31.25%, 26.56%, and 22.85%, respectively. We found that although the tasks require visualization, LLMs produce outputs dominated by formal languages, symbols, and graphics programs, rather than explicit diagram-mediated reasoning. Furthermore, while the solutions generated by LLMs may contain all the necessary elements, they are often visually confusing to humans. Ultimately, this benchmark aims to visualize mathematical problems and promote research on enabling LLMs to simplify their reasoning via visual diagrams. 

---
# Learning More by Training Less: Uncertainty-Guided Progressive Contraction for Efficient Reasoning Post-Training 

**Authors**: Suorong Yang, Jiaxin Wang, Jiaming Su, Chentao Yue, Furao Shen, Peng Ye, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2610.12933)  

**Abstract**: Large language models have achieved remarkable advances in complex reasoning. However, scaling reasoning post-training with ever-larger corpora can substantially increase training cost without proportional performance gains. In this study, we therefore ask: how far can reasoning supervision be compressed without sacrificing capability? To this end, we propose an uncertainty-guided progressive contraction framework that concentrates supervision and optimization on the reasoning trajectories most valuable for learning. Specifically, rather than treating a teacher-generated chain-of-thought (CoT) as a single realized path, we construct uncertainty-profiled reasoning supervision by building Monte Carlo reasoning trees around intermediate reasoning states to expose process-level uncertainty. This uncertainty first contracts the original corpus into a compact teacher-informed pool. A second, learner-aware contraction then performs online routing during SFT, concentrating optimization according to the student's evolving learning state. Together, the two stages progressively contract both supervision and optimization, substantially reducing the student training budget. Experiments across diverse reasoning benchmarks and LLM families show that our method uses as little as 0.22% of the optimization budget of full-data SFT, over 450x less training exposure, while maintaining or improving reasoning performance. Notably, our framework generalizes well across heterogeneous student families, black-box teachers, and weak-to-strong transfer. This indicates that it is not only a way to accelerate training but can also enhance reasoning capability. 

---
# Collaborative Learning and Reasoning over Model Slices 

**Authors**: Yuheng Wang, Runde Yang, Zhuohan Bao, Qiran Zhang, Huatao Li, Yiyang Zheng, Lin Wu, Jingru Fan, Shu Yao, Yihan Li, Zhi Li, Chen Qian  

**Link**: [PDF](https://arxiv.org/pdf/2610.12927)  

**Abstract**: Large language models (LLMs) excel across diverse tasks, benefiting from knowledge priors, in-context reasoning, and test-time computation. Despite these strengths, single-model reasoning remains vulnerable to error propagation, as later steps depend on the model's earlier outputs, leaving reasoning blind spots difficult to independently reassess. To address this limitation, we develop \textit{collaborative learning}, encompassing the construction of heterogeneous collaborators and the learning of effective cross-model collaboration. Specifically, \textit{model slicing} replaces a large model with smaller, heterogeneous models under a comparable parameter budget, enabling prior reasoning to be reassessed across distinct model parameterizations. \textit{Pyramid supervision} combines protocol distillation with step-level supervision to refine reasoning and interaction decisions throughout collaboration. We evaluate our approach across four task categories: multi-hop question answering, mathematical reasoning, code generation, and open-ended constrained generation. The trained heterogeneous models achieve comparable or better task performance than the original large model and established multi-agent baselines, including reinforcement-learning-based methods, while also achieving higher inference efficiency than the multi-agent baselines and exhibiting differentiated collaboration patterns across tasks. These findings show that learned collaboration turns model heterogeneity into complementary reasoning capabilities. 

---
# Teaching neural decoders with language 

**Authors**: Xuan Wen, Yonghao Song, Yuhong Zhang, Guoliang Li, Tzyy-Ping Jung, Xiaorong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.12921)  

**Abstract**: Brain-computer interfaces decode neural activity, yet class-label supervision (CLS) leaves relationships among classes implicit. We propose Language as Supervision (LaS), which aligns neural features with natural-language class descriptions through learned projections and predicts classes using candidate text prototypes. We evaluate seven encoder architectures with independently trained models for visual, motor, and speech decoding from scalp and intracranial recordings. LaS improves speech-decoding accuracy by up to 17.15~pp over matched CLS baselines. Its representations support candidate-set reuse and label-supervised subject and session adaptation, while LaS enables conditional unseen-frequency decoding in steady-state visual evoked potential (SSVEP) responses. Correct class-description correspondence supports this generalization, and dual-sided attribution visualization shows neural and textual contributions to the same class-matching score. These findings suggest that text descriptions can serve as general-purpose references for learning and interpretation in neural decoding. 

---
# AI-Mediation in Retail Markets: Revenue Effects and Assortment Optimization 

**Authors**: Omar El Housni, Marouane Ibn Brahim  

**Link**: [PDF](https://arxiv.org/pdf/2610.12917)  

**Abstract**: AI shopping assistants increasingly influence which products consumers consider. We study retail assortment planning under AI mediation, where a retailer chooses a catalog, an AI intermediary forms a customer-facing shortlist, and the customer chooses from that shortlist according to a multinomial logit model. We consider both deterministic mediation, in which the AI follows a fixed segment-specific ranking, and stochastic mediation based on the Plackett-Luce model.
Empirically, we construct a dataset of product rankings from leading AI models, estimate Plackett-Luce ranking models, and evaluate their predictive performance on held-out assortments. Combining these estimates with synthetic customer preferences, we find that accounting for AI mediation can materially change optimal assortments and revenue. Across the tested capacity configurations, ignoring mediation yields mean revenue losses between 2.52% and 11.37% relative to the PL optimum. A much simpler deterministic plug-in, based only on the ordering of fitted AI utilities, performs substantially better, with mean losses of at most 0.53%.
Motivated by this performance, we study assortment optimization under deterministic mediation. We characterize its computational complexity and show that, for any fixed number of customer segments, the problem admits a fully polynomial-time approximation scheme. 

---
# DeFiRank: Federated Pairwise Ranking for Governance in Decentralized Financial Networks 

**Authors**: Zahra Batool, Michael Duchesne, Kaiwen Zhang, Ulrich Aivodji  

**Link**: [PDF](https://arxiv.org/pdf/2610.12916)  

**Abstract**: Existing governance approaches in decentralized financial networks rely on manual review processes conducted by human committees, which introduces subjective bias, creates a scalability bottleneck under high network volume, and fails to capture complex behavioural patterns and evolving network dynamics. To address these limitations, we propose \name, a federated learning-based pairwise ranking framework in which multiple institutions collaboratively train a ranking model without sharing their private transaction data. This collaborative learning enables the model to capture network-wide behavioural patterns that signify strong network participation and directly contribute to the network effect, including diversity of counterparties, usage of multiple applications, and frequency of transactions. The proposed system ranks providers based on the authenticity of their actions, promoting genuine contributors and demoting those exhibiting suspicious behaviour. We evaluate \name\ on a real-world blockchain transaction dataset. Results demonstrate that the federated approach achieves ranking performance modestly below the centralized baseline (NDCG@15: $\approx$0.87 vs.\ $\approx$0.92) while preserving data privacy and matching its ranking stability (Kendall $\tau$: $\approx$0.70 for both regimes), producing reliable governance decisions, including provider promotion and demotion (648 and 39 unique providers, respectively), without requiring centralized access to sensitive transaction data. 

---
# OGPO: Offline Goal-conditioned Policy Optimization for Recoverable Vision-Language-Action Models 

**Authors**: Xule Gao, Xi Wang, Zehua Zang, Rui Wang, Changwen Zheng, Chuxiong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.12914)  

**Abstract**: Vision-language-action (VLA) models have shown strong promise in robotic manipulation, yet their post-training still predominantly relies on trajectory-level supervised imitation over robot demonstrations. While effective for learning feasible actions, this paradigm supervises policies to reproduce expert actions at each step, biasing them toward demonstration-specific action paths and providing limited guidance for recovering toward task-progress states once execution deviates from demonstrated trajectories. To address this limitation, we propose OGPO, an Offline Goal-conditioned Policy Optimization framework for recoverable VLA post-training. Instead of treating offline demonstrations merely as step-wise action labels, OGPO adaptively relabels trajectories with semantic key states and constructs goal-conditioned process rewards to optimize the policy for reaching these task-progress states. In this way, OGPO shifts VLA post-training from trajectory-level action imitation to key-state reachability learning, enabling policies to acquire a more robust ability to complete goal states from feasible off-demonstration states without requiring additional online interaction. Experiments on LIBERO and MetaWorld show that OGPO consistently improves VLA policy performance over standard supervised fine-tuning. Moreover, evaluations on LIBERO-Plus and our proposed LIBERO-ReAct, a mid-execution perturbation benchmark, further demonstrate that OGPO achieves stronger robustness and goal-directed recoverability under distribution shifts and recoverable off-demonstration states. 

---
# Mind the Spatial Gap: A Benchmark for Geospatial Reasoning and Constrained Spatial Editing 

**Authors**: Zheng Ren, Yanning Wang, Ziheng Huang, Fei Huang, Shiqi Wang, Ziqian Ye, Yang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2610.12881)  

**Abstract**: Understanding and manipulating spatial relations is fundamental to AI systems that reason about physical environments and construct grounded world models. However, existing evaluations of multimodal large language models (MLLMs) for spatial reasoning rely predominantly on static question-answering (QA) over small-scale or synthetic scenes, rarely probing whether models can modify spatial configurations while satisfying explicit geometric constraints. We introduce GeoSpatialBench, a benchmark designed to evaluate an escalating progression of spatial cognition, ranging from spatial relation understanding and geometric reasoning to constrained spatial editing. Grounded in real-world urban data across six geographically diverse cities, GeoSpatialBench evaluates a comprehensive taxonomy of 28 topological, directional, metric, and ordering relations across structured GeoJSON and map image modalities, with ground-truth references and constraints verified via GIS operations. Our empirical evaluation across frontier MLLMs uncovers a stark spatial gap: structured GeoJSON consistently outperforms map images, with multimodal combinations offering no reliable gain over GeoJSON alone, and while models show reasonable competence on qualitative topological and ordering, performance drops on quantitative metric relations. Crucially, editing performance on ordering relations declines relative to QA, and most failed edits violate at least one specified spatial constraint. These results highlight a gap between passively recognizing spatial relations and actively manipulating geometric configurations. GeoSpatialBench provides an open, rigorous diagnostic testbed to benchmark these spatial limitations, laying essential empirical groundwork for physically grounded world models and embodied spatial AI. 

---
# One-Step Text Generation by Seq-Drifting 

**Authors**: Jiayu Liu, Ruizhi Zhao, Tianyun Ji, Yee Whye Teh, Zhenya Huang, Ning Miao  

**Link**: [PDF](https://arxiv.org/pdf/2610.12879)  

**Abstract**: Diffusion language models generate tokens in parallel, but still require multiple sampling steps to produce high-quality text. In this paper, we investigate whether a model can learn to generate an entire sequence in a single forward pass. We propose Seq-Drifting, a framework that applies drifting to sequences of continuous token representations and directly learns a map from Gaussian noise to text. Unlike existing DLMs that derive the supervision at each position directly from the observed token in a training sequence, Seq-Drifting builds a drift field whose position-wise targets adapt to the current generated context. At each position, attraction guides each output toward a set of contextually plausible token representations, while repulsion discourages collapse across samples and repetition within sequences. The generator learns from this drift field during training and produces all tokens jointly in one pass at inference. Empirically, Seq-Drifting achieves a generative perplexity of 36.01 on LM1B in one step and performs competitively in both unconditional and conditional generation. Experiments on translation, summarization, and reasoning further demonstrate the applicability of the framework beyond open-ended generation. These results suggest that drifting over token sequences offers a promising path toward effective one-step text generation. Our code is available at this https URL. 

---
# Collaborative Theory of Mind: Learning to Self-Revise by Anticipating User Responses 

**Authors**: Fanzhi Zeng, Anubrata Das, Maytal Saar-Tsechansky, Liu Leqi  

**Link**: [PDF](https://arxiv.org/pdf/2610.12872)  

**Abstract**: Large language models (LLMs) demonstrate strong individual task performance, yet still struggle to collaborate with users over extended, multi-turn interactions, for example by attempting solutions before understanding the user's needs or by failing to revise mistaken assumptions after receiving feedback. We propose Collaborative Theory of Mind (CToM), a framework that addresses these challenges by explicitly training LLMs to anticipate user responses and to use those predictions to improve their own behavior. CToM drafts a response, predicts how the user will respond, and uses this prediction to guide internal revision before presenting a response. During training, the user's reply to each draft serves as the target for predicting the next turn, and we train the assistant to minimize the negative log-likelihood of that reply. We combine this prediction objective with group-relative policy optimization (GRPO) to optimize both task performance and interaction efficiency. Across mathematical reasoning, coding, and writing tasks, CToM achieves higher collaborative task performance than all baselines; for example, on a collaborative mathematical reasoning task, it improves final accuracy from 74.1% to 79.6% compared with naive GRPO. Ablations show that predicting user responses also makes collaboration more efficient: under the same reward, adding prediction supervision reduces the number of user replies by 8% on coding and 17% on writing while improving task performance. Behavioral analyses of interaction trajectories show that user-response predictions become more accurate after training, and that more accurate predictions are associated with larger improvements in collaborative task performance through self-revision. 

---
# Mitigating Capability Degradation in LLM Self-Improvement via Representation Geometry Regularization 

**Authors**: Jiaqi Tian, Ting Long  

**Link**: [PDF](https://arxiv.org/pdf/2610.12855)  

**Abstract**: On-Policy Self-distillation is a self-improvement paradigm that enables large language models (LLMs) to improve their capabilities by leveraging the same model as both teacher and student under different conditioning contexts. However, we find that OPSD methods can progressively degrade previously acquired capabilities: queries that are correctly responded at earlier training stages may become incorrectly responded as optimization proceeds, even the overall performance continues to improve. Further analysis reveals that this degradation coincides with a deterioration of the geometry of hidden representations, characterized by reduced intra-cluster compactness and diminished inter-cluster separation. Motivated by this observation, we propose Representation Geometry Regularization (RGR), which preserves representation geometry during post-training by encouraging compactness within cluster and separation across cluster. RGR can be seamlessly integrated into existing OPSD methods without altering their core optimization procedures. Extensive experiments on multiple OPSD methods and benchmarks show that RGR generally alleviates capability degradation while retaining capability improvements. Our code is available in this https URL. 

---
# Towards Defining, Measuring, and Leveraging Creativity in AI Applications 

**Authors**: Jingyi Yang, Alexander Tuzhilin  

**Link**: [PDF](https://arxiv.org/pdf/2610.12845)  

**Abstract**: Despite extensive research, computational creativity still lacks a conceptually grounded, practically relevant, and empirically applicable formal definition across contexts such as art, scientific research, and games. Existing approaches to creativity often focus on proposing informal conceptual definitions without providing formal computational concepts. Alternatively, they propose formal definitions without carefully examining their underlying assumptions or providing validation of these definitions by testing their performance in specific application settings. To address these problems, this paper proposes several formal postulates of creativity grounded in prior psychological work on creativity, especially as related to Boden's three core criteria of creativity: newness, value, and surprise. Furthermore, we propose a conceptual mathematical definition of creativity that aligns with these postulates. We also validate our proposed creativity formula in two applications: determining the creativity of particular chess moves, and assessing the creativity of paintings by some artists, such as Picasso, Renoir, and other painters. Experimental results demonstrate that our formula of creativity successfully captures creative patterns across both the chess and the art domains. This suggests that creativity can be effectively characterized through a unified mathematical structure based on three fundamental criteria, namely newness, value, and surprise, whose specific measurements are adapted to the particular domains. 

---
# Neural networks are more modular than single neurons suggest 

**Authors**: Shreya Saha, Benjamin Bergen, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2610.12842)  

**Abstract**: Neural networks routinely develop representations in which individual neurons contribute to multiple distinct functions, leading to the widespread view that such networks lack modular organization. We challenge this conclusion by introducing a formally nested hierarchy of modularity: unit-level, which requires distinct computations to occupy disjoint neurons; orthogonal, which asks whether a rotation reveals functionally independent subspaces; and linear, which asks whether a bounded linear map can separate task-relevant computations into independent subspaces. We develop a unified framework for learning decompositions at each level and evaluating them causally through selective ablation. Across vision models, language models and multi-task Recurrent Neural Networks (RNNs), we find that networks which lack unit-level modularity nonetheless often exhibit clear functional modularity at the subspace level. In vision and language models, content versus style and syntax versus semantics respectively dissociate under subspace ablation despite sharing overlapping populations of neurons. These learned subspaces also align differentially with brain data, with content and style subspaces predicting higher and lower visual cortex respectively, and semantic subspaces outperforming syntactic ones in predicting language cortex activity. In RNNs trained on multiple cognitive tasks and previously shown to lack modular structure, our decomposition recovers sharp double dissociations between tasks. The absence of unit-level modularity frequently coexists with clear subspace-level modularity across domains, enabling targeted intervention on specific computations even in networks where no individual neuron is specialized, with broad implications for mechanistic interpretability and for understanding functional organization in both biological and artificial neural systems. 

---
# Reward Hacking as a Transmissible Strategy in Multi-Agent Coding Systems 

**Authors**: Yuepeng Hu, Pegah Ghahremani, Reza Aghajani, Mahesh Pasupuleti, Himaghna Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2610.12822)  

**Abstract**: Multi-agent systems are increasingly used to solve coding tasks by coordinating agents that share code and plans. These agents are often optimized with reinforcement learning (RL) using automated evaluators, but bypassing evaluation can earn reward without producing a genuine solution. Prior work has largely treated reward hacking as a single-agent failure. In this work, we show that a reward-hacking strategy elicited in one agent can propagate to others and affect system coordination. Across four open-weight backbones, coding RL elicits and amplifies reward hacking in two settings: explicit mechanism descriptions in the system prompt and mid-training that models upstream exposure to facts about the coding environment. Using post-RL agents from the mid-training setting as sources, we observe propagation through code and shared memory, and across roles through natural-language plans. Even agents based on small open-weight models induce reward hacking in stronger closed-weight targets. On hard tasks, propagation can intensify along a transmission chain after the original source leaves as the code acquires more persuasive justifications. We also theoretically analyze conditions under which reward hacking amplifies, persists, or decays across agent interactions. Beyond propagation, reward hacking can bias task routing toward reward-hacking agents and survive candidate aggregation. As potential mitigations, we evaluate code review by the coder or a separate verifier, both of which reduce delivered hacks. Together, these results establish reward hacking as a transmissible strategy whose effects extend beyond the agent in which it first emerges. 

---
# VEPA: Verified Evolving Formal Policy for Lifelong LLM Agent Guardrails 

**Authors**: Duy Nguyen, Lesly Miculicich, Bhavana Dalvi Mishra, Shuo Li, Krishnamurthy Dj Dvijotham, Mohit Bansal, Tomas Pfister, Long T. Le  

**Link**: [PDF](https://arxiv.org/pdf/2610.12813)  

**Abstract**: LLM agents increasingly operate over sensitive resources, such as private documents and mutable system states, which can create serious safety and security risks. Runtime guardrails should therefore make inspectable, reproducible, and deterministic allow or refuse decisions before potentially irreversible agent actions are executed. However, safety and security requirements are rarely complete before deployment, so guardrails need to keep adapting as new risks emerge. Existing adaptive guardrails learn from deployment experience, but they usually apply safety policies through LLM inference, which can lead to inconsistent decisions and ambiguous policy interpretations. Formal guardrails, on the other hand, enforce policies deterministically, but they usually rely on pre-deployment or task-specific policies and have no mechanism to adapt to newly observed risks. We introduce VEPA, a framework for the verified evolution of formal agent guardrails. VEPA represents the guardrail as a formal policy of executable symbolic rules over typed agent events. During deployment, we use an LLM guided by deployment feedback to propose policy patches that fix newly observed failures while preserving previously correct behavior. Then, a deterministic verifier only accepts policies that are structurally valid and conflict-free. Finally, an evidence-aware authority layer activates a verified rule only when it has enough supporting evidence, so uncertain updates cannot directly affect guardrail decisions. Across ASSEBench-Safety, ASSEBench-Security, and ATBench, VEPA consistently outperforms baselines while expanding the scope of deterministic formal-policy enforcement. These gains also hold under more challenging deployment conditions, including mixed streams, corrupted feedback, and longer horizons. 

---
# EvoBelief: Falsifiable World Models for Self-Improving LLM Agents 

**Authors**: Janet Wang, Jiang Guo, Yi Xiang, Jiaqing Yuan, Mingwen Dong, Hideo Kobayashi, Shuaichen Chang, Chandan K. Reddy, Jiarong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.12789)  

**Abstract**: Self-evolving LLM agents typically improve how they act through accumulated experience, reflection, or skills. Yet even procedurally capable agents can fail when their beliefs about environment dynamics are incomplete, incorrect, or outdated. We study a complementary axis of self-evolution: evolving those beliefs during deployment. Such beliefs can support multiple tasks, but incorrect ones can also propagate across them. We introduce EvoBelief, a self-evolving world model that represents environment knowledge as falsifiable hypotheses. Unexpected outcomes and decision-relevant unknowns trigger hypothesis formation. Later observations and controlled experiments provide independent validation before a rule can guide predictions or plans within its supported scope. Counterexamples narrow or retire rules as the environment changes. The language model remains frozen, and each rule remains linked to its evidence and the decisions it changes. Across four benchmarks spanning static dynamics, hidden drift, and scientific discovery, EvoBelief improves task performance and reuses learned rules across tasks. On ALFWorld without actor-side transition memory, it improves success over the matched Frozen World Model by 23.9% and 31.3% with Claude Haiku 4.5 and GPT-5.4, respectively. On MiniGrid, it revises stale rules and recovers after unannounced dynamics changes. In DiscoveryWorld's Reactor Lab, it discovers and validates quantitative laws, completing 83.3% of Normal-tier and 58.3% of Challenge-tier instances, versus 0% for each evaluated baseline. These results demonstrate self-improvement through testable, revisable environment knowledge rather than model-parameter updates. 

---
# Representation Alignment from a Few Paired Data via Distilled Optimal Transport 

**Authors**: Julien Kordes, Reuben Dorent  

**Link**: [PDF](https://arxiv.org/pdf/2610.12781)  

**Abstract**: Aligning two independently pretrained unimodal encoders typically demands millions of paired examples. In this paper, we study the regime where only $N \sim 100$--$500$ pairs are available, while unpaired data on either side is abundant. Our method expresses every representation by its similarities to the $N$ paired anchors, which places both modalities in a common coordinate system; matches the two point clouds with a single entropic optimal transport problem solved over the entire pool, whose marginal constraints counteract the hubness that defeats nearest-neighbor matching; and distills the resulting plan into two lightweight encoders, so that the correspondence becomes inductive. We additionally put this marginal constraint in relation with the one-step hub corrections of the bilingual lexicon induction literature. Pairing is required only for the anchors: the unpaired collections may be drawn independently from the two marginals, so that no item outside the anchor set needs to have both of its modalities observed. We evaluate on image-text and molecule-text alignment, where our method outperforms existing approaches at the same anchor budget on Flickr30k, MS-COCO and ChEBI-20. 

---
# Faithful and Complete LLM-Assisted Automated Construction of Hierarchical Hardware Knowledge Graphs 

**Authors**: Haotian Qiao, Robert P. Dick  

**Link**: [PDF](https://arxiv.org/pdf/2610.12762)  

**Abstract**: Hardware datasheets contain engineering specifications essential for component selection, system integration, and hardware compatibility verification. However, manual inspection is tedious and error-prone because datasheets are lengthy and heterogeneous, with dense information fragmented across tables, figures, and text. Although large language models (LLMs) can automate this process, naïve one-shot processing omits relevant specifications even when the document fits within the model's context window.
We present an automated LLM-driven pipeline that transforms hardware datasheets into general-purpose hierarchical hardware knowledge graphs through chunk-based property extraction, hierarchical routing, and iterative refinement. The framework uses LLMs for information extraction and semantic reasoning while preserving source provenance and applying graph updates through validated instructions and deterministic local execution. This staged design reduces per-query context and maintains traceable intermediate representations. The resulting knowledge graphs provide semantically organized, source-grounded hardware properties for downstream retrieval and reasoning.
We evaluate the framework on 20 hardware datasheets spanning diverse component types and document lengths. We assess property faithfulness and introduce inter-graph completeness to measure relative semantic coverage without exhaustive ground truth. Our framework achieves 98.8% property faithfulness compared with 76.1% for the one-shot baseline, while fully covering 83.6% of faithful baseline properties; in the reverse direction, the baseline fully covers only 10.5% of ours. These improvements provide more reliable and comprehensive engineering information for downstream hardware design and verification, reducing the risk of overlooked constraints and incorrect decisions. 

---
# MagicBeanStalk: Escaping the RLVR Zero Advantage Trap with Recursive Data Synthesis 

**Authors**: Ximing Lu, Aviral Kumar, Jan Kautz, Yejin Choi, David Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2610.12659)  

**Abstract**: Data-centric scaling of RLVR has focused on collecting or generating verifiable training problems. However, corpus size alone does not determine whether the dataset provides a useful learning signal for a target policy. To guide the synthesis of effective RLVR data, we experimentally characterize dataset effectiveness as a function of its problems, the target policy, and the training rollout budget. Under this characterization, we show that a large open-source math corpus pooled from widely used RLVR datasets falls almost entirely into a zero-advantage trap: for a fixed target policy and rollout budget, roughly 88% of its problems are solved in every rollout or failed in every rollout. Filtering concentrates what remains, but it can only select, discarding the hard problems. Effective synthetic problems should therefore require the reasoning operations and compositions required to solve those hard problems. Yet their difficulty must be calibrated, so that some rollouts succeed, some fail, and at least one correct solution fits within the training budget. Prompting a teacher to vary hard problems directly does not achieve this. Guided by these empirical principles, we introduce MagicBeanStalk, a top-down decomposition and bottom-up recursive data-synthesis algorithm that grows seed problems into families of substantively new, policy- and budget-calibrated training problems while preserving the seeds' reasoning blueprints. Across 8 math benchmarks, continual RLVR on MagicBeanStalk problems improves gpt-oss-120b from 61.2 to 69.8 (+8.6) and DeepSeek v4Flash from 72.4 to 77.9 (+5.4) on a 16K-token budget. Training on the full 78K source corpus improves by only +1.1 and its best filtered real data subset by +5.8. In finance, we synthesize over 14K verifiable problems from only 340 solutions and improve over base by 6.9 and 2.4 points on SecQue and FinanceBench, respectively. 

---
# CLEAT: Learning Value-Guided Clarification for Interactive Agents 

**Authors**: Jiatan Huang, Zheyuan Zhang, Ziming Li, Tianyi Ma, Mingchen Li, Yanfang Ye, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.12655)  

**Abstract**: LLM agents can execute increasingly complex tasks, but effective assistance also requires aligning their actions with goals and constraints that users may leave unstated. We study this problem as interactive intent alignment through clarification. Yet clarification is useful only when the agent can turn the answer into better decisions. As training changes how the agent handles incomplete information and uses additional answers, the same clarification can become more or less useful. We introduce CLEAT (CLarification through valuE-guided Agent co-Training), a framework that co-trains a clarification-value controller and the task policy from shared interaction outcomes. We formulate residual query value to measure the task utility gained by asking rather than proceeding under the current policy, and estimate it through paired rollouts from the same interaction history. These paired returns train the controller to predict clarification value and the task policy to select useful questions and act on the answers. The controller guides subsequent rollout collection, while outcomes from the updated agent provide new value labels that allow clarification guidance to adapt as the agent learns. With Qwen3-4B, CLEAT achieves the highest mean score in all ten evaluation settings across four benchmarks. Gains over the strongest baseline reach 13.2 and 6.5 percentage points on UserGym Travel and IN3 underspecification judgment accuracy, respectively. Ablations show that co-training improves the task policy even when the controller is removed at inference, while adapting clarification guidance to the evolving policy yields further gains. 

---
# Beyond Binary Verifiable Rewards: Behavioral Self-Rewarding for SWE Agents 

**Authors**: Yiran Wu, Andrew Zhao, Yurui Chang, Qingyun Wu, Xuan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.12598)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) provides reliable supervision for software engineering agents through executable tests, but binary rewards fail to distinguish trajectories with substantially different debugging quality. We investigate whether the current policy can judge its own trajectories and use these judgments to improve task performance without a separately trained reward model. To this end, we introduce Behavioral Self-Rewarding (BSR), a reward-refinement method that reuses the policy to compare observable debugging behavior among trajectories from the same task with the same verifier outcome. BSR aggregates pairwise preferences into task-local scores and converts them into bounded auxiliary rewards, introducing reward contrast while preserving the verifier-defined pass/fail ordering. This provides additional supervision from already-collected trajectories without requiring new environment interactions. Experiments on SWE-bench Verified and SWE-bench-Live show consistent improvements across the evaluated Qwen3-14B and Qwen3-32B configurations, supporting using the policy's own comparative judgments as a complementary learning signal for training software engineering agents. 

---
# SpireBench: Evaluating Long-Horizon Strategic Play of Agentic Models with Slay the Spire 

**Authors**: Scott C. Lowe, Sander W. Keemink  

**Link**: [PDF](https://arxiv.org/pdf/2610.12596)  

**Abstract**: Agentic large language models now tackle complex tasks, yet strategic reasoning over long horizons is under-studied: most agentic benchmarks are short and episodic, and performance on long tasks is hard to dissect into stored knowledge, reasoning and luck. We introduce SpireBench, in which models play the complete commercial deck-builder Slay the Spire (StS), where early decisions decide long-term success. Across 20 large language models (LLMs) at the game's highest difficulty, frontier models beat heuristic baselines and match the average human player but remain far behind experts. This gap widens as the run lengthens: the strongest models win the first act as often as experts, then fall behind. What separates strong models from weak is how well they play each fight; all of them misjudge risk over a run, preserving health rather than spending it to grow stronger. A comprehensive, self-consistent renaming of every game term shows that part of their performance rests on prior knowledge of the game, and models do not improve from the memories they write across runs, which are often wrong and overly cautious. Our harness is the first to give a model the complete original game, and we compare it with existing harnesses. Overall, SpireBench offers an unsaturated long-horizon benchmark with tools to locate where and why agents fail. 

---
# Know What You Don't Know: Agentic Data Synthesis by Evolving Generation Space for Closed-Loop Learning 

**Authors**: Wei Yang, Qing Ping, Youzhi Luo, Yawei Wang, Abhishek Kumar, Panpan Xu, Jun Huan  

**Link**: [PDF](https://arxiv.org/pdf/2610.12593)  

**Abstract**: The continued improvement of large language models increasingly depends on synthetic experience, yet effective data synthesis for interactive agents requires not only executable and verifiable trajectories, but also interactions that provide learning value for the current task agent. Existing methods increasingly incorporate environment feedback, data validation, or learner failures, but still face a feedback-to-decision gap: observed weaknesses do not directly specify how future synthesis should change, or whether they can be represented by the current synthesis space. To address this gap, we propose EvoSyn, a learning-feedback-driven multi-agent framework that organizes synthesis around a revisable state specifying what interaction factors can be deliberately generated and where synthesis effort should be allocated. A role-specialized multi-agent system translates learner feedback into synthesis revisions, while a Designer allocates synthesis directions and a Planner grounds them into executable experience. This process enables within-space adaptation toward weak or under-covered regions and synthesis-space expansion that promotes newly discovered factors into explicit generation controls. Experiments on $\tau^2$-Bench and CAR-Bench show that EvoSyn improves average task success by 13.3 points over the strongest synthesis baseline, while producing higher-quality and structurally broader synthetic experience and discovering controllable factors missed without learner feedback. 

---
# Equal-Budget Monte Carlo Tree Search versus Randomized-Restart List Scheduling for DAG Workflows: A Controlled Equal-Budget Computational Study 

**Authors**: Yuyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.12542)  

**Abstract**: Monte Carlo Tree Search (MCTS) is often proposed for scheduling directed-acyclic-graph (DAG) workflows, yet most studies never run the comparison that could refute it - giving a much simpler method the same search effort. We run that comparison under a controlled protocol. All arms receive an identical budget - the same number of candidate schedules built and evaluated: an MCTS scheduling method, preregistered as the main arm; a randomized-restart list scheduler with no search tree as the control; and a textbook-UCT MCTS arm as a registered follow-up on the same domain. The analysis plan was locked before data collection; an independent checker confirmed all 768 confirmation-stage schedules valid with correct finish times. At the main budget of 2048 evaluated schedules, MCTS is 2.88% worse than the restarts (95% CI [-3.62%,-2.15%]; the 72 of 96 held-out instances with positive baseline-relative headroom). The deficit is not budget starvation: on the separate exploration pool restarts self-improve by 1.85% where MCTS gains 0.24%, and 700 of 720 of MCTS's first results are merely the warm start it was handed. The studied selection rule locks deterministically onto the first-expanded child, keeping its trials correlated while restarts draw gains from the tails; the causal link to makespan remains an inference. A structural map locates where the tree stops paying: restarts win on sparse irregular workflows, no difference is detectable on regular layered ones, and dense ones leave no room to improve. Swapping the rule for textbook UCT removes most of the deficit (-0.26%, 95% CI [-0.71%,+0.23%]) without reversing its sign. Results are confined to a controlled synthetic domain (N in {8,16,32} tasks, 2-8 processors, budgets up to 2048); we make no claim about larger budgets. 

---
# EliteTable: Self-Improving Program Search for Executable Control Policies 

**Authors**: Stuart Farmer  

**Link**: [PDF](https://arxiv.org/pdf/2610.12536)  

**Abstract**: EliteTable is a self-improving program-search algorithm guided by large language models (LLMs) that uses a fixed-capacity table of high-scoring solutions to guide new proposals, edits of existing solutions, and remixes of several solutions. Performance was evaluated across six Gymnasium control tasks, each with ten generations and fifty candidate slots per generation. Each task was evaluated across ten independently initialized searches. Candidate-score medians rose between the first and final generations in all six tasks, and selected-winner search-score medians rose in five. In CartPole-v1, the selected winner reached the search-score ceiling in the first generation of all ten searches. All ten paired held-out endpoints improved in four of the six tasks. BipedalWalker-v3 and CarRacing-v3 improved without reaching their reporting targets. Six of ten Acrobot-v1 searches declined on held-out episodes, and the median paired change was negative. The results document improvement during program search and the transfer of selected-winner gains to new episodes in some tasks. Comparisons and ablations are needed to establish EliteTable's relative performance and the contributions of its components. 

---
# Faithfulness Without Anchors: A Position Paper on the Structural Misalignments of Explainable AI Faithfulness Metrics for Feature Importance 

**Authors**: Marco Zullich, Emily Schiller, Ivan Gentile, David Dembinsky, Adriano Lucieri, Kilian Göller, Steffen Seitz  

**Link**: [PDF](https://arxiv.org/pdf/2610.12493)  

**Abstract**: Explainable Artificial Intelligence (XAI) tools have often been presented as methods to increase trust in machine learning models by making aspects of their algorithms more human-understandable. However, evidence accumulated over the last decade points to an overall failure of this objective, instead promoting overreliance on incorrect predictions, which can subsequently lead to algorithmic aversion. Rather, XAI appears to function primarily as a debugging tool, useful for obtaining anecdotal insights into, for example, features learned by the model. One of the main issues underlying this limitation is that XAI methods often oversimplify the inner workings of the model, resulting in misalignment between the explanation and the model itself - a behavior referred to as unfaithfulness or infidelity. Faithfulness can be interpreted as the degree of approximation between an explanation and the model. Computing faithfulness in a rigorous manner would thus make it possible to quantify one of the various elements of trustworthiness of an explanation. However, existing metrics present significant limitations, making this quantification either infeasible or computationally expensive. In this position paper, we outline five major shortcomings of current faithfulness metrics and propose four possible avenues for research to bolster their reliability. 

---
# Hide and Seek: HIDE Concepts to SEEK explanations for black box models 

**Authors**: Bhanu Tokas, Rahul Nair, Rockwell Jackson, Hannah Kerner  

**Link**: [PDF](https://arxiv.org/pdf/2610.13935)  

**Abstract**: Interpreting machine learning model predictions is a central challenge, particularly for black-box vision models. Concept-based attribution methods interpret models using human-understandable concepts. Existing concept-based methods can only interpret classification and regression models. They require access to model weights or large concept-annotated datasets, and most methods cannot provide image-level attribution. We propose Hide and Seek (HnS), a new method for concept-level attribution that does not require access to model weights or labeled concept datasets. HnS can interpret a wide range of vision models. Given a concept and image set, HnS retrieves concept-aligned images using a vision-language model, localizes the concept within each image, masks the localized region, and measures attribution from the change in the black-box model's output between the original and masked images. This enables faithful attribution by directly measuring the model's behavior and supporting local attributions. We validated HnS on CelebA and COCO-Stuff. Compared to existing methods, HnS showed better agreement with ground-truth attributions. We demonstrated the applicability of HnS on diverse tasks, including image captioning and satellite image segmentation, where no existing concept attribution method can be used. We provided our anonymous code in the supplementary material. 

---
# Revisiting Tabular Deep Regression: Objectives and the Limits of Global Early Stopping 

**Authors**: Danil Rudenko, Georgii Bulgakov, Nikolay Kartashev, Ivan Rubachev, Daniil Krasylnikov, Artem Babenko  

**Link**: [PDF](https://arxiv.org/pdf/2610.13909)  

**Abstract**: While deep learning models for tabular data have recently received significant research attention, the choice of objective functions for regression tasks remains an understudied degree of freedom. Several lines of research offer alternatives to the standard Mean Squared Error (MSE), but differences in datasets, backbones, and evaluation protocols make their practical benefits difficult to compare. In this work, we address this gap by performing a systematic re-evaluation of regression objectives across a broad collection of established tabular datasets and multiple neural backbones. Our comparison reveals that many proposed objectives provide limited or inconsistent gains over a properly tuned MSE baseline, while quantile loss yields consistent aggregate improvements over MSE across the tested backbones. We then identify a limitation that remains even with this strong objective: regions with different levels of target noise can reach their best predictive performance at different training epochs. Consequently, a single global early-stopping checkpoint can impose a compromise between uncertainty regions, even when the objective can represent heteroskedastic target distributions. Motivated by this limitation, we introduce a heteroskedasticity-aware extension of quantile loss that uses input-dependent label smoothing as a potential remedy and improves over quantile loss on several backbones. Overall, our results provide a modern empirical picture of regression objectives for tabular deep learning and highlight how differences in optimal stopping times across uncertainty regions can limit the effectiveness of global early stopping. The code is available at this https URL. 

---
# Task Context Shapes Component Roles in Language Models 

**Authors**: Sovesh Mohapatra, Dani S Bassett  

**Link**: [PDF](https://arxiv.org/pdf/2610.13895)  

**Abstract**: Large language models reuse the same attention heads across tasks, but identifying these shared heads does not explain what they do differently in each task. A natural source of inspiration is task-dependent interactions among shared brain regions, which motivate a network view of language-model computation. Here, we compare fixed model components across tasks using identical graph facts, statements specifying which directed connections are present. We measure how interventions affect attention heads and the activation state in later layers (influence), as well as the answer (importance). Task changes redistribute the influence in five open-weight models spanning four families and 1.5-14 billion parameters. In Gemma-2-9B and Qwen3-14B, an activation edit changes an early internal activation vector and switches which fact the model reads while the written instruction stays fixed. The same heads then respond to different graph facts and influence later components differently. For each prompt, we construct two linear response maps, one before the edit and one after it. Each map predicts a later head's response from a measured change in an earlier head. For a separate factual intervention, the map for the corresponding condition reduces mean squared prediction error by 53.6% in Gemma and 5.0% in Qwen compared with the average of the two maps. These findings show that task context changes both which graph facts the shared components respond to and how their outputs affect later computation. A network account may help explain how models use shared parameters to perform different computations. Measurements of task-dependent influence could also guide interventions intended to alter one behavior while preserving others. 

---
# SpatialASM: Internalizing Active Mentalization for Spatial Reasoning 

**Authors**: Yuhong Liu, Shenxi Wu, Junyi Chi, Beichen Zhang, Xuanlang Dai, Jiaqi Wang, Yuhang Zang, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.13832)  

**Abstract**: When solving complex spatial reasoning tasks, ordinary humans naturally revisit the scenes from task-relevant viewpoints and mentally reconstruct the spatial layouts from those viewpoints. But can Large Vision-Language Models (LVLMs) proactively think in the same way? We term this ability Active Spatial Mentalization (ASM), an intrinsic component of visual intelligence that remains substantially underdeveloped in current LVLMs. To bridge this gap, we introduce SpatialASM, a scalable framework to internalize ASM into LVLMs through training-time spatial simulation. SpatialASM employs an allocentric novel-view generator to ground hypothetical viewpoints in visual evidence and produces two complementary forms of supervision: i) explicit ASM trajectories that teach LVLMs when and how to adaptively invoke view planning and imagination during spatial reasoning, and ii) automatically-constructed perspective-taking tasks that further strengthens viewpoint-conditioned scene imagination. On seven benchmarks spanning relatively perception-oriented spatial understanding to more challenging tasks requiring multi-image, cross-view and viewpoint-transformation reasoning, SpatialASM yields performance gain on LVLMs of four LVLM backbones in spatial reasoning while retaining their original general vision capabilities. The experimental results demonstrate that our approach improves average accuracy across the seven spatial benchmarks. And the synthesis of ASM trajectories provides a scalable pathway to alleviate the scarcity of spatial Chain-of-Thought (CoT) supervision for eliciting active viewpoint exploration and mental scene simulation. 

---
# MC-EvoBench: Auditing Experience-Driven Self-Improvement in Minecraft Agents 

**Authors**: Jingwei Song, Zhisheng Chen, Bingfan Zeng, Jing Tang, Yubo Hou, Jinhao Jing, Zhengwei Xie, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.13829)  

**Abstract**: A central challenge in embodied recursive self-improvement is enabling agents to convert interaction history into capabilities that persist beyond one world and transfer to later tasks. A saved memory or successful final action does not by itself establish that a reusable capability was acquired; evaluation must separate capability change from carried-over resources, unequal source exposure, and early execution failures. We construct MC-EvoBench as a controlled Minecraft benchmark with a fixed catalog of 19 registered evolution chains, including 12 primary evaluation chains and seven calibration or probe chains, together with three graph-aligned paired probe families for transfer. The benchmark represents each task as a task-state graph, defines stage-level environment verifiers, assigns matched source and target roles, resets the target world, and permits only a declared agent artifact to cross the source-target boundary. For each probe, no-prior, related, unrelated, and shuffled source conditions are evaluated under matched environment-step budgets and five seeds; we report TSR, WGP, and VMP and retain provenance records. The benchmark reveals a consistent direct-capability gap between controller and planner-backed variants, while related experience yields no stable transfer: it does not improve either tool probe, its small wood-species gain hinges on one seed, and all intervals include zero. Stage audits further show that early wood-collection and crafting bottlenecks often leave later capabilities unobserved. MC-EvoBench thus offers a controlled testbed for durable, transferable embodied capabilities. Code and data are available at this https URL 

---
# Same Query, Different Answer: Run-to-Run Variance in Semantic Query Processing 

**Authors**: Oktie Hassanzadeh, Dharmashankar Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2610.13823)  

**Abstract**: Semantic query engines ask a large language model (LLM) to judge individual rows, and benchmarks compare these engines on the quality of a few executions at temperature 0. We measure how much another execution changes. We re-execute the text and tabular workloads of SemBench, LRO-Bench, and UDA-Bench through their own harnesses, log every one of 1.2 million model calls, and reanalyze SemBench's published executions. At temperature 0, between 0.07% and 10.2% of rows change their operator output over five executions, depending on the model, and up to 24.0% on one task. Plans amplify these changes exponentially in the number of rows an answer exposes, as a possible-worlds model predicts: on UDA-Bench, a few percent of extracted values change, but about two thirds of the answers do. In SemBench's published results, 21% of text results change quality between executions, including on engines whose plans are deterministic. Several per-query system comparisons, and 16 of 28 tests of LRO-Bench's implementation recommendations, are not settled by the executions the benchmarks run. Changing rows are rows on which models disagree with each other; identical requests within one execution disagree as often as across executions; and one execution's decision margin or reasoning length predicts which rows change. Voting reduces the variance but does not remove it, and a calibrated cascade keeps its guarantee while the rows it returns change. We derive reporting rules for benchmarks and design rules for engines. 

---
# FORTE: A Production Framework for Feature Evaluation and Deprecation in Industrial Recommender Systems 

**Authors**: Vishnu Arun, Adam Obeng, Tianyou Wang, Zilin Ren, Mukuntha Narayanan Sundararaman, Han Wang, Shivin Thukral, Zhixiang Wang, Saurabh Vishwas Joshi, Darren Reger  

**Link**: [PDF](https://arxiv.org/pdf/2610.13806)  

**Abstract**: Production recommender systems have feature-lifecycle asymmetry: adding new features must produce gains on some metric, while feature removal must not regress any metric. In practice, feature deprecation is hard to prioritize as a feature can fail to be removed if it interacts with other features, its importance changes daily, it cannot be evaluated before production models change or data drift occurs, it is in contention for A/B test space with other launches, it causes regressions in an online A/B test, or it is consumed by multiple models that need to coordinate its deprecation. We present FORTE, a model-agnostic production framework for discovering, validating, and removing features. Its efficient perturbation implementation allows daily feature evaluation. To account for interactions, it evaluates candidate feature sets constructed from all the low-rank features through grouped ablation and its offline results are sufficiently predictive of online A/B tests to allow deprecations to proceed to online testing with confidence. Lineage tracing identifies all downstream consumer models to realize the full savings of removing a feature from the training set. We validate FORTE over 2 large-scale recommender systems and find that it identifies deprecable feature sets which do not cause online regressions, while the impact on efficiency depends on model characteristics. On a search-relevance model, FORTE resulted in 3 nested candidate sets of 32, 48, and 59 features. Online guardrails rejected the largest treatment. The 48-feature set was selected for deprecation, removing 43.3% of the model's features. On a large ranking model, FORTE identified 8 candidates for removal. Retraining and A/B testing confirmed no regression. Its launch reduced the trainable parameters by 29.9%, improved serving throughput by 19.8-25.5%, and reduced annual training costs by hundreds of thousands of dollars. 

---
# Your Discrete Graph Diffusion Model is Secretly Memorizing 

**Authors**: Adarsh Jamadandi, Nicolas Keriven  

**Link**: [PDF](https://arxiv.org/pdf/2610.13786)  

**Abstract**: Discrete graph diffusion models (GDMs) corrupt the graph structure by adding noise independently to nodes and edges under a Markovian noise model. In the unconditional setting without node features, an equivariant denoiser trained on a single graph is able to generate novel graphs and sample-level metrics such as edge overlap and WL kernel similarity find no evidence of memorization. In this work, we challenge this view and show for the first time that graph diffusion models indeed exhibit memorization. Analyzing the reverse diffusion trajectories of GDMs trained on two-block stochastic block models (SBMs), we find a strong train-data bias: the model recovers a corrupted training graph almost perfectly compared to a held-out test graph. We term this conditional memorization, as it surfaces only when the reverse trajectory is conditioned on a corrupted training graph and is not apparent during inference. It coincides with a growing generalization gap, where the training loss improves as the loss on held-out graphs degrades, and it cannot be mitigated by early stopping. The onset of memorization always precedes the emergence of community structure in the generated graphs, thus every checkpoint earlier produces structureless graphs. This behavior holds across all node sizes we test. 

---
# ReplaySSM: Cache SSM Inputs, Not State 

**Authors**: Ze-Wei Liou, Harris Nover, Amey Naik, Andrii Skliar, Igor Shovkun, Ido Shahaf, Tri Dao  

**Link**: [PDF](https://arxiv.org/pdf/2610.13778)  

**Abstract**: State-space models (SSMs) maintain a fixed-size recurrent state, but reading and writing it at every decoding step can bottleneck hybrid models. This eager state management causes large state I/O with low arithmetic intensity, leaves speculative decoding no cheap rollback, and feeds state quantization error into subsequent recurrent updates. We introduce ReplaySSM, which unrolls the recurrence to represent the state as a checkpoint and a buffer of recent SSM inputs. This separates output computation from checkpoint updates. By updating the checkpoint periodically, ReplaySSM avoids state writes on most steps and roughly halves the dominant state memory traffic. Keeping recent inputs also makes speculative rollback a buffer-pointer operation and enables output-only verification without materializing intermediate states. The same separation allows a low-precision checkpoint to serve output computation on most steps while a higher-precision checkpoint serves recurrent updates, so rounding error in the low-precision copy never enters later updates. We implement ReplaySSM in vLLM for Mamba-2 and Gated DeltaNet. At serving batch size 256, it accelerates standard decoding by $1.20$--$1.47\times$ over a tuned vLLM baseline across four hybrid models, and delivers about $1.8\times$ the throughput of vLLM's speculative decoding on the two 120B-class models. Separate state precisions also reduce output KL divergence and downstream accuracy losses compared with a shared low-precision checkpoint. 

---
# DualQuant: Dual-scale Calibration-free Weight-and-Activation Quantization for LLMs 

**Authors**: Mahsa Salmani, Sayeh Sharify, Hesham Mostafa  

**Link**: [PDF](https://arxiv.org/pdf/2610.13767)  

**Abstract**: Low-bit inference on accelerators with native 4-bit support requires quantizing both weights and activations to realize the full throughput gains these formats provide. While post-training weight quantization has been extensively studied, activation quantization remains considerably more challenging and typically requires calibration data. In this work, we introduce DUALQUANT, a dual-scale, calibration-free post-training quantization framework that directly minimizes the weight-reconstruction Mean Squared Error (MSE) and, as an immediate by-product, yields per-channel activation rescalings prior to activation quantization. Our key observation is that the scaling factors obtained by minimizing weight-reconstruction error provide effective activation rescalings, enabling competitive activation quantization without calibration data. We formulate DUALQUANT as a non-differentiable weight-reconstruction problem and develop an efficient solver based on variable splitting and block coordinate descent, where each subproblem admits a closed-form update. DUALQUANT provides a unified algorithm supporting a wide range of formats, including the full MX family and NVFP4. To our knowledge, DUALQUANT is the first dual-scaling method to formally support MX and NV block formats. Across Llama and Qwen models evaluated on WikiText-2, C4, and zero-shot downstream tasks, DUALQUANT matches or exceeds the accuracy of calibration-based SmoothQuant across most models and formats, and substantially outperforms calibration-free baselines, with large gains for MX and NV block formats. We validate these gains with a real packed W4A4 inference path for Llama-3.1-8B on an RTX 4090. The resulting implementation closely matches the accuracy of fake quantization while delivering up to 2.61x memory reduction and up to 2.46x throughput speedup over BF16. 

---
# When Edge Independence Fails: Joint Graph Diffusion with Latent Sociability Priors 

**Authors**: Adarsh Jamadandi, Nicolas Keriven, Aline Roumy  

**Link**: [PDF](https://arxiv.org/pdf/2610.13765)  

**Abstract**: Discrete graph diffusion models corrupt graph structure through independent edge updates with a Markovian noise model. Combined with the exchangeability requirement on the output distribution, the terminal prior is necessarily an Erdős--Rényi (E-R) random graph. This has a fundamental consequence for unconditional graph generation without node features: in the large graph limit, any GNN denoiser applied to an E-R graph approximately outputs an E-R graph, making effective denoising difficult. In practice, generated graphs fail to recover meaningful node heterogeneity and higher-order structure. Previous models introduced specialized node features to compensate, framing this as an expressivity problem, while the role of the prior remained unexamined. We instead address the root cause by introducing the Latent Sociability Prior (LSP) and a join} diffusion process that co-evolves latent node sociability variables and graph structure, removing edge independence while preserving exchangeability. Unlike existing latent graph diffusion methods, the sociability mechanism requires no complex encoder: its sole role is to guide edge updates and preserve node heterogeneity throughout the trajectory. We validate our approach on multiple graph generation datasets and show that the proposed framework more accurately captures graph distributions than baseline methods without node features. 

---
# Matched-Context Group Policy Optimization for LLM Agent Training 

**Authors**: Chenxu Zhang, Yaorui Shi, Wenlong Zhang, Jianchao Tan, Zihan Wang, Yiming Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.13755)  

**Abstract**: Training long-horizon language model agents requires assigning credit to intermediate decisions from sparse terminal feedback. Within a finite group of sibling rollouts, useful action comparisons depend on their context, evidence for both alternatives, and how that evidence enters learning. We propose Matched-Context Group Policy Optimization (MCGPO), a procedure for constructing weighted action preferences from already collected trajectories. MCGPO defines comparison contexts with pre-action state summaries and observed rubric progress, annotated by task rules. Two-action support selects among ordered exact and coarse matching rules. Fixed folds separate candidate decisions from the terminal outcomes used to orient their preferences, and support-dependent evidence weights scale an auxiliary pairwise objective alongside shared GraphGPO task learning. MCGPO reuses rollout data and each response's original history, preserving critic-free training. Across six environment-backbone settings with three training seeds each, MCGPO improves mean success over matched GraphGPO: by 0.47-1.46 percentage points on ALFWorld and WebShop with Qwen2.5-1.5B and 7B, and by 3.96 and 5.16 points on WebShop with Qwen3-4B and 8B. The ALFWorld gains build on baseline success above 92%, where remaining headroom is limited. Integrated success improves in three of four Qwen2.5 settings. Three-seed factorial controls support opposite-fold evidence and evidence weighting; joint state-rubric matching achieves the highest mean in all four representation-control settings. 

---
# SafeDiSTL: Decomposing Joint Constraints for Safe Multi-Agent Reinforcement Learning 

**Authors**: Hongrui Zheng, Aditya Singh, Rahul Mangharam  

**Link**: [PDF](https://arxiv.org/pdf/2610.13737)  

**Abstract**: Cooperative multi-agent tasks often impose constraints on a shared objective or collective safety. A team level violation cost function can measure this constraint outcome, but does not distinguish which agents' actions most strongly influence that violation. We present SafeDiSTL, a constrained multi-agent reinforcement learning framework that allows for credit assignment even with joint constraint measurements. We use Signal Temporal Logic (STL) robustness to measure episode-level violation through running-maximum cost increments. Smooth full-episode robustness and differentiable dynamics provide action sensitivities, which allocate each cost increment across agents at the same decision time. These sensitivities account for delayed effects when weighting agents' shares of each cost increment. We study the framework in cooperative payload transport, connected navigation, and networked traffic control tasks. SafeDiSTL achieves 1.6-5.1x the safe-success rate in transport, improves navigation safe-success by 11.1%, and delivers 50-68% more throughput in traffic when compared to constrained baselines. Results demonstrate a strong improvement of the learned policy in dealing with shared constraint while retaining the performance of unconstrained policies. 

---
# Orthros: Phase-Aware Heterogeneous Attention for Efficient Transformers 

**Authors**: Yuhong Chou, Zehao Liu, Yuqi Pan, Qian Liu, Xianwei Chen, Zhongzhu Zhou, Jibin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13735)  

**Abstract**: Efficient attention mechanisms, such as linear and sliding window attention, aim to reduce the computational complexity of Transformers from quadratic to linear. However, their degraded in-context recall often necessitates interleaving full attention layers, leaving the quadratic bottleneck unresolved in long-context processing. To address this, we introduce Orthros attention, which shifts the efficiency paradigm from structural allocation to phase-aware decoupling. As a unified module, it dynamically switches its computation pattern across different phases of sequence processing, enabling linear complexity prefilling and full attention decoding. Our design grants Orthros attention a lower asymptotic complexity compared to vanilla Transformers in prevalent long-context scenarios characterized by heavy prefilling and light decoding. Beyond theoretical formulation and complexity analysis, extensive empirical evaluations demonstrate Orthros's superiority across multiple dimensions. First, we validate its fundamental language modeling capabilities, demonstrating scalability across varying model sizes and strong competitiveness in comprehensive architectural comparisons. Second, Orthros exhibits exceptional conversion feasibility via a data-efficient Transformer-to-Orthros conversion, yielding performance that even surpasses the original Transformer. Finally, Orthros achieves a $21.9\times$ prefilling speedup over Transformers with end-to-end inference gains, while preserving long-context capabilities, as demonstrated by near-perfect 128K needle-in-a-haystack results and Transformer-level performance on real-world in-context recall tasks. Overall, these advantages in high performance, flexibility, and efficiency position Orthros as a promising candidate to succeed the Transformer in industrial applications. 

---
# LAPT-Scale: Scaling Embodied Policies with Latent Action Pre-Training on Human Videos 

**Authors**: Zhiyuan Xiang, Peidong Liu, Mingyang Li, Jiawei Li, Wenhow Li, Wenbo Ding, Xiang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.13732)  

**Abstract**: Scaling generalist robotic policies is limited by the high cost of collecting embodied interaction data. Large-scale egocentric human videos offer a rich alternative, but they lack robot action labels, and existing action annotation methods are often noisy under real-world visual complexity and diverse manipulation dynamics. We introduce LAPT-Scale, a scalable latent action pre-training paradigm with two components: LAPT-Scale-LAM learns continuous latent actions by explaining frame-to-frame visual transitions in human videos and uses them as human-derived supervision, while LAPT-Scale-Policy trains a VLA policy through human-video latent-action pre-training, target-robot embodiment alignment, and task-specific post-training. Unlike prior vector-quantized LAMs, LAPT-Scale-LAM uses a continuous latent space to preserve fine-grained motion and interaction dynamics. We further combine perceptual, semantic, and optical-flow alignment signals to improve transition-level supervision beyond pixel reconstruction. Trained on 23.3K hours of egocentric video, LAPT-Scale-LAM provides implicit action supervision for policy pre-training, yielding consistent scaling gains and improving real-robot success from 63.3% to 85.0% as annotated human video scales from 1K to 5K hours. The learned policy also generalizes to unseen visual conditions. 

---
# Amortized Counterfactual Prediction with Graph Neural Networks 

**Authors**: Hendrik Stumpf, Francesco Quinzan  

**Link**: [PDF](https://arxiv.org/pdf/2610.13724)  

**Abstract**: Counterfactual inference asks how an observed outcome might have differed under different circumstances, such as whether a patient would have recovered had they received a different treatment. Such questions arise whenever decisions depend on the outcomes of alternatives that were not observed. Answering counterfactual queries generally requires knowledge of the underlying causal structure, which is often unavailable in practice. A common approach is therefore to first estimate the causal graph from observational data and then perform counterfactual inference. However, errors in the estimated causal graph can propagate to downstream counterfactual predictions. To address this challenge, we introduce the Counterfactual Graph Neural Process (CFGNP), an end-to-end approach that answers counterfactual queries directly from observational data without first estimating a causal graph. CFGNP uses Graph Neural Networks (GNNs) to exploit the dual-graph structure by restricting cross-world message passing to corresponding factual--counterfactual variables. Across synthetic and image-based benchmarks, CFGNP achieves stronger predictive performance than existing approaches, including on the German Loan benchmark, where it reduces MSE from $4.51 \times 10^{-2}$ to $0.19 \times 10^{-2}$. Furthermore, on synthetic causal graphs, a model trained only on 10-node graphs maintains strong predictive performance on graphs with up to 30 nodes, without retraining. 

---
# Calibration in One Go: Jointly Learning Accurate and Calibrated Neural Networks 

**Authors**: Nektarios Kalampalikis, Pablo M. Olmos, Isabel Valera  

**Link**: [PDF](https://arxiv.org/pdf/2610.13693)  

**Abstract**: Neural networks (NNs), while highly flexible and versatile, are often poorly calibrated. Post-hoc calibration methods adjust the outputs of an already trained model using a separate calibration dataset and additional calibration parameters. This process adds overhead that requires separate optimization of the calibrator and subsequent model selection. In this paper, we show that standard (post-hoc) calibrators can be trained end-to-end with the backbone via a bilevel Stackelberg formulation. This formulation eliminates the need for a separate calibration set or a dedicated calibration loss by jointly optimizing accuracy and calibration using cross-entropy, introducing only one additional hyperparameter: the calibrator learning rate. Although our framework supports calibrators ranging from temperature and matrix scaling to NN layers, our results show that using the final classification layer itself as the calibrator yields accurate, well-calibrated models with no additional parameters. Our approach outperforms SOTA calibrators with minimal added complexity, while also avoiding the multiplicative overhead of separate calibration components in settings such as multi-task learning. 

---
# Variance-Reduced Policy Gradient is Nearly Minimax-Optimal 

**Authors**: Leonardo Cesani, Matteo Papini, Marcello Restelli  

**Link**: [PDF](https://arxiv.org/pdf/2610.13692)  

**Abstract**: Standard Policy Gradient (PG) algorithms suffer from high estimation variance because they rely on Monte Carlo updates. While off-policy variance reduction mitigates this problem, establishing optimal sample complexity currently requires either restrictive assumptions, such as uniformly bounded importance weights, or impractical second-order operations. In this work, we introduce Loopless Variance-Reduced PG (LVR-PG), a unified theoretical framework that abstracts a class of first-order estimators that integrate recursive updates with defensive importance sampling, including known estimators such as PAGE-PG and STORM-PG. We prove that this algorithmic class achieves $\tilde{\mathcal{O}}(\epsilon^{-2})$ last-iterate global convergence under weak gradient domination for general parametric policies. We establish this guarantee using purely first-order updates, with minimal assumptions and without gradient preconditioners. By extending state-of-the-art lower bounds to continuous parametric policies, we establish an $\Omega(\epsilon^{-2})$ sample complexity limit, confirming our framework is minimax-optimal up to logarithmic factors and horizon terms. 

---
# Not All Dimensions Are Equal: Outlier-Aware Attention Linearization with Grouped Quadratic Kernels 

**Authors**: Weiqi Sun, Qianyue Wu, Wenjia Wang, Tianyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13674)  

**Abstract**: Linearizing softmax attention offers a route to efficient Transformer inference, but preserving pretrained attention remains challenging. We identify an important source of approximation error overlooked by standard linearization methods: a small number of outlier dimensions contribute disproportionately to query-key dot products and account for a large fraction of the approximation error. Motivated by this heterogeneity, we propose Outlier-Aware Attention Linearization (OAL), a dimension-aware method for constructing quadratic attention approximations. Starting from a second-order Taylor approximation, we introduce a Hadamard-product formulation that makes individual query-key contributions explicit. This formulation relaxes globally shared coefficients to support dimension-specific linear terms and cross-dimensional quadratic interactions while preserving separable evaluation. To retain this flexibility with fewer parameters, OAL groups dimensions with similar contribution scales to share coefficients, while allowing distinct treatment of the few outliers. The resulting positive quadratic kernel is calibrated against pretrained attention outputs and admits linear-time evaluation with respect to sequence length for fixed head dimensions. On Qwen2.5-0.5B, OAL reduces frozen-attention output relative MSE by approximately 8.9% compared with DiJiang. Replacing middle-layer attention modules and jointly fine-tuning the kernel coefficients with LoRA further lowers WikiText-103 perplexity from 31.71 to 27.50 compared with a learned global quadratic, while improving accuracy on PIQA, HellaSwag, and ARC-E. These results demonstrate the value of outlier-aware dimension grouping for both attention approximation and pretrained-model conversion. The code is available at this https URL 

---
# Inject a Turn, Not a Token: Learning Where to Redirect LLM Reasoning for Self-Correction 

**Authors**: Jiwon Jeon, Myungsik Cho, Jeonghye Kim, Youngchul Sung  

**Link**: [PDF](https://arxiv.org/pdf/2610.13672)  

**Abstract**: Large language models struggle to reliably identify and correct errors in their own reasoning. Unlike prior training-based approaches that fine-tune an LLM to generate revisions or feedback, we learn where to intervene in failed rollouts and determine how to redirect them. To choose where to intervene, we train a 130K-parameter MLP head on a frozen encoder using only final-answer correctness. For redirection, we find that under a fixed generation budget, the tested inline injections fail to reliably redirect reasoning, whereas presenting the same instruction in a new user turn changes the model's course. The resulting correction framework, UPTURN (User-Prompt TURN injection at a learned position), requires no LLM fine-tuning: it retains a selected prefix, injects an instruction as a new user turn, and regenerates the suffix. Across four reasoners and six math benchmarks, UPTURN improves average accuracy in every setting over a 32-rollout baseline. It outperforms self-correction and reflection-token injection baselines under matched correction budgets, transfers across reasoners and to GPQA-Diamond without retraining, and complements GRPO and self-correction methods. 

---
# Pretraining and Test-Time Training Budgets Matter for ARC: A Controlled Study of VARC and TRM 

**Authors**: Adrien Deliege, Marc Van Droogenbroeck  

**Link**: [PDF](https://arxiv.org/pdf/2610.13660)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC) is increasingly used to evaluate whether neural networks can infer task-specific transformations from few demonstrations. Recent ARC models differ in architecture, but also in broader training and evaluation protocols, including pretraining data, data augmentation, test-time setup, and optimization budgets, making conclusions about architecture benefits debatable. In this work, we study how the performance of the vision-inspired VARC and recursion-based TRM models is affected by pretraining and test-time training (TTT) budgets under a common inductive protocol on ARC-AGI-1. We cross five pretraining budgets with four TTT budgets for each test task. Each budget specifies a task-variant count and total sample presentations, defining the controlled training budgets. We observe that increasing TTT sample presentations improves the standard ARC pass@2 metric in all settings for both models. Increasing the TTT task-variant count consistently helps only TRM, whereas VARC often performs better with fewer variants. VARC also remains competitive without pretraining, while TRM benefits much more from it. These results show that pretraining and TTT budgets affect ARC evaluation and motivate reporting at comparable budgets. 

---
# SelectArena: A Modern Benchmark for Feature Selection on Real-world Tabular Data 

**Authors**: Bastian Schäfer, Dominika Matus, Lennart Purucker, Ricardo Knauer, Erik Rodner, Frank Hutter, Anton Sokolov, Arnaud Liefooghe  

**Link**: [PDF](https://arxiv.org/pdf/2610.13644)  

**Abstract**: Selecting a subset of features (columns) from a tabular dataset is a ubiquitous challenge in real-world applications. Industry experts seek to minimize model training costs, climate scientists require an interpretable subset of factors to understand extreme weather events, and clinicians need to choose among diagnostic tests due to financial and time constraints. When looking at the scientific literature to determine which feature selection method to use, practitioners across domains face an unrepresentative landscape of benchmarking. Despite extensive research, existing feature selection benchmarks provide limited guidance because they rely on outdated models, ad hoc evaluation protocols, or artificially generated data that do not reflect real-world problems. To close the gap between scientific benchmarking and real-world applications, we introduce SelectArena, a modern benchmark for feature selection. We compare 15 widely used feature selection methods on 71 manually curated real-world tabular datasets, covering classification and regression problems, using standardized evaluation protocols and state-of-the-art tabular (foundation) models. Our results reveal clear trade-offs across predictive performance, runtime, stability, and validity. Tree-based methods, CART and RFImportance, along with F-Test are Pareto-optimal when taking into account predictive performance and runtime. F-Test, information-theory-, and distance-based methods lead on stability and validity, with F-Test emerging as a balanced choice across our metrics. Search-based methods are slow, unstable and achieve low performance. SelectArena turns these findings into practical guidance and, as an extensible platform, supports future feature selection research. 

---
# Model-agnostic Knowledge Augmented Machine Learning for Weakly Annotated Bioimaging Data 

**Authors**: Jayanie Bogahawatte, Sachith Seneviratne, Simon Maksour, Dzung Do-Ha, Elizabeth Qian, Samuel Brennan, Matthew Keon, Lezanne Ooi, Saman Halgamuge  

**Link**: [PDF](https://arxiv.org/pdf/2610.13637)  

**Abstract**: Machine learning models are increasingly used for automated feature extraction from weakly annotated bioimaging data, such as microscopy and cytomorphology images. Due to the sole dependency of automated feature extraction, existing methods yield inconsistent classification performance across weakly annotated datasets. Here, we propose a model-agnostic knowledge augmentation method that enables ML models to exploit domain knowledge beyond what automated feature extraction can capture. The proposed approach improves classification performance across seven ML architectures. When integrated with masked autoencoder-based self-supervised learning, our method achieves the strongest overall performance, improving precision by 2.9% over the next-best method on three in-house microscopy datasets and by an average of 6.8% across three public cytomorphology benchmarks. We perform nucleus-level classification based on mislocalization of TAR DNA-binding protein 43 (TDP-43) from the nucleus to the cytoplasm, a phenotype linked to Amyotrophic Lateral Sclerosis (ALS), using three independent microscopy datasets of motor neurons derived from human induced pluripotent stem cells. Accurate quantification of this phenotype at scale remains challenging and may contribute to inconsistencies across studies. Application of the best-performing model to these microscopy datasets reveals biological interpretations related to ALS, including dose-dependent effects of the proteasomal stressor, MG132, and the oxidative stressor, sodium arsenite (SA), on TDP-43 mislocalization, variable pathology in motor neurons from sporadic and familial ALS donors, and a lack of protection of the pharmacological compounds, edaravone and niclosamide, against SA-induced TDP-43 mislocalization. Together, our pipeline enables in-vitro cellular image analysis to advance understanding of ALS-associated pathology and potential treatments. 

---
# SWE-DeceptionBench: Benchmarking Deceptive Behaviors of LLMs Across the Software Development Lifecycle 

**Authors**: Xiting Liu, Lin Yang, Yitong Zhang, Jia Li, Shi-Min Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13626)  

**Abstract**: Large Language Models (LLMs) are increasingly used throughout the software development lifecycle. This growing reliance on LLMs raises concerns about their trustworthiness, with deception posing a particularly important risk. Deceptive behavior takes the form of misleading responses or assistance in fabricating, concealing, or distorting information. To study this behavior, we introduce SWE-DeceptionBench, a benchmark for evaluating LLM deception in software engineering. The benchmark covers six lifecycle stages and two deception triggers: Active Deception, where models recognize infeasible or conflicting requirements but still present them as achievable, and Passive Deception, where models help fulfill explicit user requests to mislead others. We further introduce Neutral, Reward, and Pressure conditions to study the effect of external inducements, resulting in 720 test instances in Chinese and English. Evaluating seven mainstream LLMs, we find substantial deception even under the Neutral condition, with aggregate rates of 49.88% for Active Deception and 64.29% for Passive Deception. Deception rates vary across lifecycle stages, and external inducements affect Active and Passive Deception to different extents. We also evaluate lightweight defenses, finding that guard models miss many deceptive requests, and Self-Reminder reduces but does not eliminate deceptive responses. These findings highlight the need for more effective deception detection and mitigation methods to support trustworthy LLM-based software engineering. 

---
# AudioTETRIS: Token Elimination via Transitional Importance Signals for Efficient Audio LLMs 

**Authors**: Shreya Sajid, Anna Vettoruzzo, Joaquin Vanschoren  

**Link**: [PDF](https://arxiv.org/pdf/2610.13600)  

**Abstract**: Large Audio Language Models (LALMs) achieve strong performance on speech-text tasks, but standard audio encoders produce hundreds of tokens for even short audio clips. Because self-attention scales quadratically in sequence length, this inflates inference-time compute substantially. Current training-free token compression methods prune or merge tokens based solely on coarse attention or feature similarity. However, treating audio tokens as generic, isotropic features smooths over spectral transitions, destroying fine-grained phonetic transitions essential for decoding speech. In contrast, speech signals exhibit highly structured temporal patterns where information density peaks at acoustic transitions. We propose AudioTETRIS, a training-free, two-stage framework that uses the audio encoder's representation geometry as an acoustic prior for token importance before applying attention-based compression. Instead of discarding encoder structure after the encoding stage, AudioTETRIS propagates this prior directly into the LLM decoder to compute a per-token fused score. This score drives progressive top-$k$ token selection across decoder layers without requiring additional training or auxiliary models. Evaluated on Qwen2-Audio-7B, Phi-4-Multimodal, and Audio Flamingo 3, AudioTETRIS retains near-uncompressed performance on several speech tasks using only 25% of audio tokens, matches or surpasses full-token accuracy at moderate compression rates, and consistently outperforms existing training-free baselines. 

---
# Defence against the dark arts of distillation: Unsupervised discovery of subliminal learning 

**Authors**: Joseph Egan, Ciarán M. Gilligan-Lee, Michael O'Riordan, Yuchen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13536)  

**Abstract**: When a language model is distilled from a teacher that shares its base parameters, the student can inherit behavioural traits of the teacher---including misalignment---even from data that is semantically unrelated to the trait (Cloud et al., 2026). All existing defences require knowing the trait. We remove this requirement and prove that a subliminally transferred trait can be recovered from the parameter updates alone, without supervision, by exploiting a single asymmetry: the trait is expressed consistently across diverse datasets generated by the teacher, whereas task-specific and shared capabilities vary. The resulting estimator returns a weight-space direction that can be amplified to discover and interpret the trait or projected out to suppress it. Applied to the subliminal-learning benchmarks of (Cloud et al., 2026), our method recovers the transmitted animal-preference traits without any trait labels and suppresses their expression with almost no impact to task performance. 

---
# RESCUE: Is This Alarm Worth Acting On? Counterfactual Evaluation of VLA Runtime Monitors 

**Authors**: David Ahmedt-Aristizabal, Abdelwahed Khamis, Mohammad Ali Armin, Lars Petersson  

**Link**: [PDF](https://arxiv.org/pdf/2610.13516)  

**Abstract**: Runtime failure monitors for robot policies are typically evaluated by how accurately they distinguish failed from successful executions. Yet an accurate alarm may arrive too late for recovery, while an unnecessary intervention may disrupt an execution that would otherwise succeed. We introduce RESCUE, a counterfactual framework for evaluating whether a monitor's alarms enable useful interventions. By restoring simulator and policy states at execution checkpoints, RESCUE compares alternative actions, identifies the latest checkpoint satisfying a specified recovery criterion, and evaluates whether acting on alarms reduces expected task-completion cost. We evaluate RESCUE with OpenVLA-OFT and $\pi_{0.5}$ over LIBERO-Spatial and LIBERO-Plus using 128,340 counterfactual branches. Recoverability depends on the state--intervention pair rather than on robot state alone: at $\tau=0.5$, 14 of 66 OpenVLA-OFT/LIBERO-Plus failures contain a recoverable checkpoint under continuation, compared with 34 of 66 under continuation or scripted recovery, while recovery reduces recoverability in other conditions. On the primary strong-policy conditions, favourable detection scores do not establish lower intervention cost. Under the evaluated stop-and-reset model, even an idealised detector can afford only a 2.0--3.3% false-alarm rate before unnecessary resets outweigh the savings from stopping failures early. These results show that monitor value depends jointly on the intervention available and the workload in which it is deployed. 

---
# Beyond Temporal Proximity: Explanatory World Models for Credit Assignment in Reinforcement Learning 

**Authors**: Oriol Corcoll, Qiyao Wei, Ciarán Mark Gilligan-Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.13504)  

**Abstract**: Generative Recommenders increasingly use Reinforcement Learning (RL) to optimize outcomes that may appear hundreds or thousands of interactions after a decision. Standard RL propagates reward according to temporal proximity rather than explicit relevance, attenuating the signal assigned to distant but important actions and creating a source of sample inefficiency. We introduce Explanatory World Models (EWM), an attribution framework that uses explanations to identify the earlier events that plausibly contributed to a realized outcome. We instantiate EWM with a frontier language model and use its explanations as learning signals using a conservative hierarchical operator that recursively redistributes reward while preserving the undiscounted episode return. To investigate what constitutes a useful explanation, we operationalize three complementary variants inspired by Halpern's actual causation, Lombrozo's explanatory simplicity, and Deutsch's hard-to-vary explanations. We evaluate EWM on two controlled long-horizon RL benchmarks, a public sequential-recommendation benchmark, and an industry-scale generative recommender. On Crafter and WordCraft, EWM substantially outperforms PPO and LaRe, with its largest gains when rewards are most delayed. On KuaiRand, EWM outperforms SFT and LaRe offline and A2C online. Finally, in an offline evaluation of an industry generative recommender, EWM considerably improves Recall and increases behavioral proxies. These results suggest that explanations can provide structured and reusable credit for learning from delayed outcomes. 

---
# When MLLMs miss what humans read: diagnosing and learning to read squint text 

**Authors**: Chen Peng, Shan Huang, Jianbo Huang, Linrui Qin, Qibo Sun, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13465)  

**Abstract**: Multimodal LLMs can fail to read text embedded in an image's coarse structure (text a human recovers by squinting) and report that no text is present. We name this class Squint Text and study it as a recognition deficiency, motivated by cases from a production moderation pipeline. We introduce SquintBench, comprising 1076 production-derived moderation images, and SquintCtrl, a controlled synthetic suite for mechanism diagnosis. Frozen-weight analysis shows that hidden text leaves a coarse signal while glyph identity is weakly represented; identity becomes substantially more accessible when the correct spatial regions are supplied, implicating spatial selection as a major bottleneck. Suppressing high-frequency content also improves reading at a fixed visual-token budget, while visual token count may also contribute. These findings motivate Squint OPSD: a frozen teacher receives a downscaled view and the ground-truth transcription as privileged context, and its token-level distributions supervise a student along on-policy trajectories. At inference, the student reads the original image alone. Squint OPSD reaches 88.61 ANLS on SquintBench, versus 26.50 for the unadapted backbone, 24.44 for the best general-purpose model on the original view, and 68.60 at best for an off-the-shelf model under a downscaled view. These results substantially narrow the gap between available visual evidence and its generated read-out without auxiliary views at inference. 

---
# Collective Autonomous Vehicle Routing Needs Social Awareness for Coexistence with Human Drivers 

**Authors**: Anastasia Psarou, Mikolaj Rams, Blazej Torbus, Rafal Kucharski  

**Link**: [PDF](https://arxiv.org/pdf/2610.13439)  

**Abstract**: Collective Autonomous Vehicle (AV) route choices in real-world traffic networks can create externalities for Human-Driven Vehicles (HDVs), as routes that benefit AVs may increase travel times for other road users. To address this, we introduce a social component, the marginal cost, to the selfish travel time AV reward. Marginal cost captures an AV's impact on other road users by comparing their travel times with those in a counterfactual simulation in which the AV is removed.
We study AV route choice in mixed-autonomy traffic settings using Multi-Agent Reinforcement Learning (MARL), where AVs learn routing policies through repeated interactions with the traffic environment. Experiments on the real-world networks of Ingolstadt and Provins show that incorporating marginal cost into the AV reward not only limits the increase in HDV travel times and shortens AV travel times relative to selfish AV routing, but also improves overall system performance. In Ingolstadt, selfish routing of 40\% AVs increases the travel times of other vehicles by 1.7\%, and this drops to 1.1\% when the social reward is included. Interestingly, not only does overall system performance improve, but the AVs are also better off when acting this http URL effect is consistent across tested cities, penetration rates, and replications, yet when the weight of the social component in the reward increases, the system fails to reach the desired performance.
For a more scalable approach to social AV routing in mixed-autonomy traffic, we compute the marginal cost only for the AVs whose route choices most affect the traffic system. This reduces the number of counterfactual simulations and achieves similar performance at a significantly lower computational cost. Our results suggest that AV routing should go beyond selfish rewards and, for the benefit of all traffic participants, include a social component in its objective. 

---
# CNN-Transformer with Sigmoid Label for High-Resolution Sub-Degree DOA Estimation 

**Authors**: Lifeng Wang, Xintong Li, Aifei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13433)  

**Abstract**: This paper addresses the grid quantization error of deep-learning (DL)-based direction-of-arrival (DOA) estimation. Classification- based DOA methods discretize the continuous angular space into a grid, so that with a peak-based read-out, the accuracy and resolution are limited by the grid step and by the width of the label peak. To address this issue, we propose a sigmoid label that encodes the sources' DOAs as the sum of angle-shifted monotonic sigmoid functions. Differentiating the output of a CNN-Transformer network trained with the sigmoid label produces sharp spectral peaks at the source DOAs. Combined with a moderately refined grid step of 0.25 degrees, the CNN-Transformer with the proposed sigmoid label attains sub-degree accuracy and a 1-degree resolution, breaking the limit of the grid quantization error. Under the unified CNN-Transformer backbone, we compare the proposed Sigmoid label with the Gaussian label, the 0-1 hard label, and the binary cross-entropy (BCE) baseline. Numerical results demonstrate that the Sigmoid label attains the highest accuracy and superior resolution compared to other labels. 

---
# In-Context Learning for System Identification: Probabilistic Formulation, Long-Sequence Handling, and Benchmark Validation 

**Authors**: Matteo Rufolo, Dario Piga, Filippo Pura, Marco Forgione  

**Link**: [PDF](https://arxiv.org/pdf/2610.13411)  

**Abstract**: Traditional system identification typically involves two steps: initially, training data is processed to estimate a model of the system of interest; then, the model is used to predict outputs for new input queries. In contrast, recent in-context learning approaches learn a meta model that maps directly from training data and query inputs to outputs in a zero-shot fashion, bypassing explicit model estimation. This approach leverages a meta model trained offline, on large sets of synthetically generated data, to emulate the behavior of a system identification algorithm. In this work, we advance this meta-learning framework for system identification in three directions. First, we develop a unified probabilistic meta-identification architecture that combines a Transformer-based encoder-decoder with tailored mechanisms for handling long training sequences, enabling calibrated predictions over long horizons. Second, we show how a pre-trained meta model can act as an uncertainty-aware synthetic data generator for classical system identification, providing task-specific data that regularize parametric models when real data are scarce. Third, we systematically evaluate the framework on standard nonlinear system identification benchmarks and on a challenging out-of-distribution scenario, demonstrating competitive accuracy with state-of-the-art methods and strong generalization, including cases where input-only adaptation is sufficient to handle significant distribution shifts. 

---
# ProgramGym: Scaling Training Environments for Long-Horizon Program Reproduction Agents 

**Authors**: Chengyi Yang, Renxu Bao, Jue Hou, Zongpei Teng, Ziheng Wang, Yujie Lin, Chengsong Huang, Junnan Zhu, Delai Qiu, Yining Wang, Shengping Liu, Fei Long, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2610.13395)  

**Abstract**: Coding agents are increasingly used to build programs from scratch as software engineers do rather than to resolve issues within a repository or to generate a repository from a given requirements document. However, scalable verifiable training environments that force agents to discover what is required and to decide how the system should be designed remain scarce. Moreover, existing agent workflows also cannot sustain long-horizon development in which the agent explores the reference program before implementation. To address these challenges, we introduce ProgramGym, a framework consisting of Program-Env and Program-Agent for scalable training environments of long-horizon program reproduction agents. Specifically, Program-Env constructs verifiable training environments from repositories through a feedback-driven loop that repairs runtime sandbox builds and the quality of black-box tests. Using this novel pipeline, we obtain 1,648 training environments spanning C, C++, Go, and Rust. Program-Agent combines a role-isolated workflow of analysis, implementation, and review with revision management for compilation checks and failure recovery. Experiments on ProgramBench show that both Program-Env and Program-Agent improve performance on program reproduction. Fine-tuning on trajectories with these constructed environments raises the performance of Qwen3.8-27B from 34.87% to 42.84%. Program-Agent further raises the base and SFT models to 55.71% and 56.25%, both exceeding some flagship models. We will release the ProgramGym to support further research on coding agents that build programs from scratch. 

---
# AgentSWE: Can Coding Agents Build the Agent You Actually Want? 

**Authors**: Jiahao Wang, Hongjin Qian, Yuyang Hu, Jiajun Zhang, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13361)  

**Abstract**: As the production of software passes into the hands of software itself, coding agents are increasingly commissioned to build agents fitted to their users' own workflows. This delegation carries a quiet assumption, that the delivered agent will faithfully serve the requirement it was built for, rather than merely appear to. No existing benchmark tests that assumption. We introduce AgentSWE, a benchmark for agent software engineering spanning three stages of the agent development lifecycle: Creation of a complete agent from a natural-language requirement, Editing of production agent codebases, and Optimization toward a measurable target. Each of its 25 tasks is framed as a commission rather than an exam: the builder iterates against a handful of illustrative cases, while acceptance exercises the frozen submission on held-out cases designed to span the full requirement, grading behavior against requirement-derived rubrics under hard pass/fail gates or, for Optimization, against the source benchmark's own objective. Evaluating a range of coding agents across models and harnesses, we find that none comes close to delivering what a requirement asks for: the gap between what a builder demonstrates and what it delivers is the norm, not the corner case. An accompanying empirical analysis of the failure trajectories identifies recurring failure modes, among them overfitting to the visible cases, hollow runtimes behind code that looks complete, and unsupported claims of completion, each a signal by which coding agents can learn to build the agents they are asked for. 

---
# Agreement Without Alignment: Label-Free Rubric Clarification Increases LLM-Judge Consensus but Not Agreement with Humans 

**Authors**: Sangdo Han, Moonsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.13303)  

**Abstract**: LLM judges are increasingly deployed as panels, and disagreement among panel members is often read as a symptom of an unclear rubric. We test a label-free procedure that turns this disagreement into rubric repairs. The procedure decomposes each rubric into sub-criteria, ranks them by cross-judge disagreement on a development split, and has an editor model rewrite only the most contested ones. An additional stage that moves text-checkable sub-criteria to deterministic checkers transferred none in our evaluation. Human labels are never used to select or edit clauses; they are used only to evaluate the outcome. We pre-specified four hypotheses in a version-controlled protocol and evaluated six rubric conditions with a panel of five judges from three providers on SummEval, the HealthBench meta-evaluation, and USR Topical-Chat, for a total of 129,770 judge requests, excluding retries. On SummEval, the procedure raised inter-judge Krippendorff's $\alpha$ by 0.069 (95% CI 0.046 to 0.092) over the original rubric and by 0.074 over a control that applied the same edits to an equal number of randomly chosen clauses that were not selected. Agreement with expert ratings showed no detectable change (Kendall $\tau$ difference -0.007, 95% CI -0.037 to 0.021). On HealthBench and USR, neither inter-judge agreement nor human agreement met the preregistered effect criteria. In a preregistered secondary comparison, human agreement after editing the contested clauses was lower than after editing the control clauses on all three datasets in point estimate, although no interval excluded zero. Selecting clauses by disagreement makes judges agree with each other more than editing arbitrary clauses does, but in our setting the added consensus did not bring the judges closer to human annotators. We will release code, frozen rubrics, item identifiers, and judge scores. 

---
# PRIME: Pre-Retrieval Index Modeling Enhancement for Structured Document Retrieval-Real 

**Authors**: Cheuk Hei Chung, Li Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.13296)  

**Abstract**: High-precision retrieval over external knowledge bases is essential for LLM agents to operate effectively. However, existing Information Retrieval (IR) research primarily optimizes query-side retrievers (e.g., ColBERT, BGE) and generators (e.g., Self-RAG, GraphRAG), while reducing corpus-side representation to chunking. This leaves index-side quality an underexplored lever. We propose Pre-Retrieval Index Modeling & Enhancement (PRIME), an index-side optimization framework built on two complementary tracks: an LLM-driven Semantic Restructuring (SR) track that transforms long documents into structured, information-dense sections, and a deterministic Metadata Synthesis (MS) and Hierarchical Retrieval (HR) track that extracts semantic anchors and heading structure for coarse-to-fine section retrieval. Our evaluations show that deterministic metadata matches structure-aware chunking's retrieval quality while cutting embedding computation by 45.8 $\times$ and index vector count by 22.4 $\times$. On top of these gains, Semantic Restructuring (SR) further improves Answer-F1 by 13.7% in full deployment, with characterized quality-fidelity-cost trade-offs. Across QASPER, BRIGHT, and NFCorpus, we further characterize three qualitative regimes that determine whether index-side optimization helps or degrades retrieval, while showing that these index-side effects are neutralized under a strong reranker. 

---
# Agent4Spec: An LLM-based Agentic System to Optimize API Specifications through Automated Validation and Market Analysis 

**Authors**: Yongjian Tang, Costel Moraru, Doruk Tuncel, Hanya Elhashemy, Steffen Klepke, Thomas Runkler  

**Link**: [PDF](https://arxiv.org/pdf/2610.13273)  

**Abstract**: API specifications serve as critical documentation for modern software development, directly impacting integration efficiency and client code generation. While recent advancements have leveraged Large Language Models (LLMs) to generate initial specifications from source code, existing approaches treat this process as a one-time event - overlooking the needs for continuous validation, long-term maintenance, and market-driven evolution of these artifacts. To address this issue, we present Agent4Spec, an autonomous multi-agent framework featuring two distinct pipelines: (1) a validation pipeline that detects and self-corrects OpenAPI syntax errors, and (2) a market analysis pipeline that conducts competitive research, identifies functional gaps against industry leaders, and automatically generates missing API-spec functionalities. We evaluate Agent4Spec on two curated datasets of API specifications spanning both general-purpose and industrial scenarios. Among 9 SOTA LLMs, the top-performer achieves 100% syntax error detection rates and over 95% implementation success for proposed corrections, while securing over 50% adoption rates for suggested endpoint improvements, representing a practical tool for API development. 

---
# Tabular Foundation Models Are Effectively Shallow 

**Authors**: Irene Cannistraci, Julia E. Vogt  

**Link**: [PDF](https://arxiv.org/pdf/2610.13253)  

**Abstract**: In-context-learning Tabular Foundation Models (TFMs) are uniquely suited to scenarios with limited resources or privacy constraints, as they solve new tasks without additional training. However, their architecture consists of multiple transformer blocks, resulting in high computational and memory footprints. Meanwhile, traditional alternatives like Gradient-Boosted Decision Trees (GBDTs) remain highly competitive and run efficiently on CPUs, suggesting that tabular tasks may not inherently require deep architectures. This contrast raises a fundamental question: can TFMs be simplified by removing architectural redundancies without sacrificing performance? To answer this, we introduce a closed-form, label-free affine translator that replaces a contiguous window of up to 96% of the blocks, leaving the rest unchanged. Evaluating eight frontier TFMs on 171 classification datasets, we reveal that tabular transformers are vastly overparameterized: replacing half of a model's blocks costs at most $0.002$ median AUROC across all architectures, with the half-depth TabFM still outperforming every other full-depth TFMs and tuned GBDTs. Furthermore, different model families are overparameterized in different ways: TabPFN-3 relies on its encoders, maintaining performance regardless of which single block is retained. Other single-axis models spread computation thinly, needing only their first block, while dual-axis models concentrate it in one critical block. Ultimately, removing depth saves up to 1.5B parameters, speeds up GPU inference by up to $6\times$ at a median cost of $0.03$ AUROC, while a single-block TabPFN-3 serves cached queries $2.4\times$ faster using $6\times$ less memory at negligible cost. 

---
# How Reliable are AMR Parser Outputs? Uncertainty Estimation for AMR Parsing 

**Authors**: Anuj Kumar, Shintaro Ozaki, Yamuna Prasad, Katsuhiko Hayashi, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2610.13236)  

**Abstract**: Recent advances have significantly improved the performance of Abstract Meaning Representation (AMR) parsers, yet determining when their outputs can be trusted remains largely unexplored. This work presents the first study of uncertainty estimation for AMR parsers in a black-box setting using top-k beam-search candidates. Uncertainty is estimated from pairwise candidate similarities and evaluated using the Prediction Rejection Ratio. This work characterises the relationship between parsing performance and uncertainty, evaluates the reliability of beam candidate distributions, analyses the effect of evaluation metrics on uncertainty assessment, and identifies parsers that provide the strongest balance between performance and reliability. Experimental results show the value of uncertainty estimation as a complementary dimension for evaluating AMR parsers. 

---
# ReSFlow: Rectifying Categorical Flows on Statistical Manifolds for One-Step Generation 

**Authors**: Xiaohuan Jia, Renzhe Xu, Xiao Wang, Jiayun Wu, Shaohua Fan  

**Link**: [PDF](https://arxiv.org/pdf/2610.13225)  

**Abstract**: Generative modeling for discrete data has been significantly advanced by learning continuous vector fields on the statistical manifolds to transport simple priors to target discrete distributions. However, because these continuous-time flows are not constrained to follow geodesics on the statistical manifold, thereby requiring multi-step numerical integration for simulation and leading to substantial computational costs during inference. In this work, drawing inspiration from the "straight and fast" properties of Rectified Flow in continuous Euclidean space, we propose the Rectified Statistical Flow (ReSFlow) model, which enables one-step categorical generation on statistical manifolds by explicitly learning geodesic trajectories that adhere to the intrinsic Riemannian geometry. By rectifying the flow on the statistical manifold, we theoretically guarantee that trajectories between prior and target distributions approximate geodesics. This rectification significantly reduces the transport cost by aligning the learned trajectories directly along geodesics on the statistical manifold, allowing the model to map pure noise to target discrete data in a single step. Through extensive experiments on diverse real-world generative tasks, ranging from image and text to biological domains, we demonstrate ReSFlow's superior one-step generation quality, which truly embodies geodesically optimal and rapid categorical generation. Our code is available at this https URL. 

---
# Beyond Actions: Learning Future Operation Targets for Vision-Language-Action Models 

**Authors**: Min Lin, Bokai Zhou, Jianqi Lin, Kaiyuan Yang, Yicheng Chang, Junjie Yu, Xiaofan Liu, Jianhua Han, Rui Li, Shen Zhao, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2610.13212)  

**Abstract**: Actions specify what a robot should execute, but do not describe what those commands will achieve. Vision-language-action (VLA) policies nevertheless learn primarily by predicting action chunks from images, language, and state. We introduce Action--Future Coupled Effect Learning (AFCE), which makes a future-grounded physical Effect the policy's primary prediction target. An offline tokenizer constructs temporally ordered Effect tokens from demonstrated actions, realized robot responses, and sparse visual transitions. Reconstruction objectives encourage the representation to retain executable detail and observed physical change. A flow-matching VLA predicts Effects from current images, instruction, and robot state, while a state-conditioned decoder converts them into action chunks. Embodiment-specific interfaces allow single- and dual-arm tasks to share the Effect model. On DexJoCo, AFCE achieves $50.97\%$ overall success, improving over $\pi_{0.5}$ and UVT by 11.82 and 15.21 percentage points, respectively. It leads on all five bimanual tasks, highlighting the value of Effect supervision for coordinated dexterous manipulation. The project page is available at this https URL. 

---
# Does Learning Recurrent Networks Matter in Partially Observable RL? 

**Authors**: Zekang Wang, Zhe He, Gaojie Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.13197)  

**Abstract**: Reinforcement learning (RL) under partial observability typically re- lies on recurrent networks to encode temporal information, with the recurrent module trained jointly with the policy or value function. This raises a natural question: is learning the recurrent network actually necessary? In this paper, we investigate whether a frozen recurrent network can provide sufficient memory for solving par- tially observable Markov decision processes (POMDPs). Motivated by the theoretical foundations of Reservoir Computing (RC) and Generalized Reservoir Computing (GRC), we study a simple archi- tecture that combines a fixed recurrent network with a trainable nonlinear output module. We evaluate this approach across tasks with varying memory demands, observation modalities, and action spaces. Surprisingly, frozen recurrent networks support recall over hundreds of steps and achieve competitive control performance across diverse POMDPs, while using less than half the number of trainable parameters required by their fully trainable counterparts. These results suggest that learning the recurrent memory module is not always necessary for solving POMDPs. Frozen recurrent networks therefore provide a simple, parameter-efficient, and sur- prisingly strong baseline for partially observable RL. Our source code is available at this https URL 

---
# Continually Post-training Small Models via Synthetic Self-distillation 

**Authors**: Woosung Koh, Hyunsoo Lee, Kyungjae Lee, Haeju Park, Dahyun Lee, Dasol Hwang, Moontae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.13113)  

**Abstract**: Continual post-training faces a dual challenge of acquisition and retention: models must learn new capabilities while preserving prior knowledge. While Self-Distillation Fine-Tuning (SDFT) mitigates this using self-supervision with in-context demonstrations, it relies on strong in-context learning and instruction-following abilities. Consequently, SDFT fails on smaller models critical for low-latency, on-device use. To enable continual learning at smaller scales, we first identify the root cause of failure. We show that poor acquisition and forgetting correlate linearly with the post-trained model's distribution shift, revealing the necessity of in-distribution learning. We thus introduce Synthetic Self-Distillation (SynthSelf), imitating SDFT's in-distribution rewrite by outsourcing the task to a minimally larger, same-family aligner. Coupled with off-policy supervision and rejection sampling, this provides the high-quality supervision small models require without breaking their original distribution. Evaluations spanning 4 acquisition datasets (including a mixture) and 14 retention benchmarks show SynthSelf consistently outperforms 6 methods (including off- and on-policy distillation) across 3 small model scales. Extensive further analyses confirm SynthSelf paves the way toward effective small-model continual learning. 

---
# Evolving Creatives with Clicks: CTR-Driven Advertising image Editing via Attribute Disentanglement and Preference Alignment 

**Authors**: Xu Zhang, Haohan Wang, Zheng Zhang, Yingjie Xia, Wei Feng, Honghe Zhu, Jingjing Lv, Changping Peng, Xuheng Qin, Zehua Zhang, Zheng Luo, Qicheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.13066)  

**Abstract**: The optimization objective for e-commerce advertising creatives has shifted from aesthetic quality to online business metrics like click-through rate (CTR). Recent methods generate appealing ad images from product images by constructing CTR-based reward signals and refining prompt generation via Reinforcement Learning. Such paradigms equally penalize all attributes within an underperforming image, even though some are actually beneficial to clicks. In this work, we propose Evolving Creatives with Clicks (ECC), which reformulates creative optimization as fine-grained editing rather than generation from scratch. ECC selectively refines attributes that hurt CTR while preserving those that contribute positively. Specifically, faced with a rich attribute space across massive images, ECC first identifies 28 attributes significantly correlated with CTR via ANOVA and Random Forest regression. Based on attribute contrasts between images with disparate CTRs, we instruction-tune a prompt model to generate editing suggestions, then refine it with real click preferences since reference-derived suggestions are not uniformly beneficial. Furthermore, we introduce Consistent Preference Optimization (CPO) as ECC's preference optimization strategy, which enforces consistency across semantically equivalent shuffled suggestions to mitigate DPO overfitting and improve preference generalization. Extensive experiments demonstrate that ECC significantly improves the CTR of ad images, outperforming state-of-the-art models by a substantial margin. The complete data pipeline, including attribute extraction and analysis, the resulting dataset, and source code will be released upon acceptance. 

---
# CoBe: A Benchmark For Conversational Counterfactual Text Editing 

**Authors**: Arvind Raghavan, Inwoo Hwang, Shreyas Havaldar, Kai-Zhan Lee, Aurghya Maiti, Yushu Pan, Jeffrey Wu, Mingxuan Li, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2610.13057)  

**Abstract**: Counterfactual text editing is the ability to revise a given text in light of a hypothetical change while preserving facts not causally affected by that change. When asked what would have happened if a doctor had not recommended surgery, the model output should retain the patient's medical history and modify only the consequences of the decision. This requires causal understanding: a model must determine, without being explicitly instructed to, what events are affected by the hypothetical change and what should remain fixed. This skill is central to human reasoning, especially in safety-critical domains like healthcare or law. By contrast, the task of associational text generation merely generates text statistically associated with the altered condition. In the medical example, an associational response might make the patient's symptoms less severe because milder cases correlate with no surgery. Such responses produce a plausible scenario, but not the counterfactual version of the given one. Prior counterfactual-text datasets do not clearly separate these modes of reasoning. We introduce the Counterfactual Benchmark (CoBe), a testbed for models' zero-shot ability to counterfactually edit a given textual scenario when prompted to do so in a conversationally natural way. CoBe contains 2000 questions spanning healthcare, science and engineering, business, and everyday scenarios. We test a suite of frontier and smaller models, with frontier systems achieving an average score of $58.05\%$. Our findings indicate that models frequently default to associational generation rather than preserving the causal structure of the original scenario. Moreover, performance does not scale consistently with either model size or output token length, suggesting that the failure may reflect model training objectives rather than insufficient model capacity. 

---
# GeoGR: Spatial-Semantic Token Compression and Group Routing for Efficient 3D VLMs 

**Authors**: Yongxin Zheng, Hao Wu, Fang Li, Hu Fu, Peiran Yin, Xudong Wang, Haozhe Hu, Xinghao Chen, Yunpu Ma, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.13035)  

**Abstract**: Multi-view representations enable pretrained image and video vision-language models to understand 3D scenes, but their long visual token sequences make inference costly. Redundancy arises both from repeated observations of the same scene regions across views and from processing all visual tokens at every LLM layer, regardless of their relevance to the current computation. We propose GeoGR, a two-stage framework that jointly compresses scene representations and visual computation. At the projector level, GeoSemZip consolidates cross-view observations into 3D voxel tokens and combines semantic saliency with spatial coverage to construct a compact scene representation. Unselected voxel features are aggregated into contextual anchors to retain scene information that would otherwise be discarded. At the LLM level, we identify an effective visual computation window and observe that query-conditioned token relevance remains locally stable but changes over longer layer intervals. Guided by these findings, GroupRoute combines late entry and early exit with recoverable group-wise skipping. Low-relevance tokens bypass complete LLM layers while retaining their hidden states and can rejoin the active group at subsequent anchor layers selected under a computation budget. For compression-aware post-training, we construct a compact dataset of 67K samples by retaining 30% of the original training pool. With this protocol, GeoGR retains 95.2% and 96.8% of the dense post-trained models' performance on LLaVA-OneVision-7B and Video-3D LLM-7B, respectively, while accelerating LLM prefill by $6.8\times$ and $6.1\times$. These results demonstrate the consistent effectiveness of GeoGR across 3D VLMs with and without explicit 3D spatial encoding, highlighting a favorable performance-efficiency trade-off. The code is available at this https URL. 

---
# Don't Guess, Just Ask: Evaluating Interactive Requirement Clarification for Repository-Level Software Development 

**Authors**: Mingyu Chen, Zhiqing Zhong, Manyi Wang, Linxi Liang, Rouyu Chen, Pinjia He  

**Link**: [PDF](https://arxiv.org/pdf/2610.13022)  

**Abstract**: As LLMs become increasingly capable of implementing software, a critical bottleneck emerges: whether they truly understand what users want. Users often begin with vague intent, leaving important requirements to be clarified through interaction. In repository-level development, misunderstandings can propagate and cause costly rework. However, existing benchmarks typically evaluate implementation from complete specifications or underspecified requests, overlooking pre-coding requirement clarification. Prior clarification studies mainly focus on function-level tasks. It therefore remains unclear how well LLMs can transform vague intent into clear requirements through interaction and how this capability affects subsequent code generation.
We introduce ReqClarifyBench, an interactive requirement clarification benchmark for repository-level software development. It contains 341 tasks spanning game, miniapp, and web development, each with a vague user intent, a human-validated requirement document, and code evaluation artifacts. We employ LLM-based user simulators with diverse personas derived from real-world conversations. We evaluate six LLMs in terms of clarification effectiveness and generated requirement document quality. Results show that stronger clarification effectiveness does not necessarily translate into higher document quality, and performance varies substantially across user personas. We also find that coding-agent harnesses do not further improve requirement document quality compared with their base LLMs. These findings offer insights for building LLM-based end-to-end software development systems. 

---
# Generalized Proximal Policy Optimization: Unifying LLM RL Algorithms from the Perspective of Divergence 

**Authors**: Zhicheng Cai, Xinyuan Guo, Hanlin Wu, Weinan Dai, Yuyan Zhou, Wei-Ying Ma, Ya-Qin Zhang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.13015)  

**Abstract**: Reinforcement learning has become the foundational paradigm for enhancing the reasoning capabilities of LLMs. Yet the dominant approach, Proximal Policy Optimization (PPO), relies on a geometrically degenerate clipping proxy that cannot adapt its resistance to policy shifts, leading to premature clipping and degraded exploration. Subsequent variants alleviate these issues through fragmented and heuristic modifications. We propose Generalized Proximal Policy Optimization (GPPO), a principled and unified framework for understanding and designing LLM RL algorithms through the lens of divergence. We establish a generalized $f$-divergence trust-region theory with a monotonic performance improvement guarantee, and show that different $f$-divergences induce distinct geometries and policy update dynamics. We further introduce novel estimator topologies that provide more faithful divergence estimates than the conventional single-sample Monte Carlo estimation. Under this framework, PPO and recent variants such as DAPO, DPPO, and RIPO emerge as special cases, establishing a principled two-dimensional design space of {Divergence $\times$ Estimator}. Extensive experiments on eight competition-level math and coding benchmarks show that both dimensions are critical. The best-performing GPPO instantiation surpasses standard GRPO by up to 50% and 14% on math and coding, delivering SOTA performance. 

---
# What Should SFT Learn? Conditional Information Gain for Generalizable Post-Training 

**Authors**: Yuxuan Gu, Kequan Niu, Xiaocheng Feng, Yun Li, Yiqiao Yang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.12995)  

**Abstract**: Supervised fine-tuning (SFT) remains a widely used approach for post-training large language models, because fixed demonstrations can directly introduce desired reasoning strategies and behavioral patterns. Yet forcing a model to imitate externally provided trajectories may also interfere with behaviors already supported by pretraining, limiting generalization beyond the training distribution. Recent dynamic SFT methods mitigate this issue by reweighting token-level supervision according to model-side signals such as predictive probability, prior support, or distributional compatibility. However, model compatibility is not equivalent to learning value: a token that is weakly supported by the current model may either be an undesirable mismatch or precisely the task-specific decision that the demonstration is intended to teach. We propose CIG-SFT, an information-theoretic framework that estimates the task-conditioned information carried by each demonstrated token. Under teacher forcing, we compare the token likelihood from a frozen reference model with and without access to the task input while keeping the gold solution prefix fixed. The resulting conditional information gain quantifies how much the task itself reduces uncertainty about the demonstrated decision. We then convert this information into a KL-regularized projection of the current policy and derive an equivalent scalar-weighted SFT objective whose logit gradient exactly matches that of the projected target. Across mathematical reasoning and code-generation benchmarks, CIG-SFT consistently outperforms standard SFT and recent adaptive fine-tuning baselines, with particularly strong improvements in generalization to challenging evaluation distributions, while requiring neither additional training data nor online rollouts. 

---
# The Evidence Anchoring Boundary: Where Visual Interventions Matter and When They Help in Multimodal LLMs 

**Authors**: Qiji Zhou, Junyi Hu, Peiyu Wang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.12981)  

**Abstract**: Visual interventions provide a training-free way to reduce hallucinations in multimodal large language models. However, an intervention may have little leverage at some layers and may damage a correct answer even when it changes the output. We investigate where visual interventions retain leverage and when they improve predictions. First, we apply the same visual-residual edit at successive decoder layers, matching its RMS strength across depth. Across six checkpoints from three model families, we observe a persistent drop in intervention effect and define its operational onset for this edit as the \textbf{Evidence Anchoring Boundary (EAB)}. With boundaries fixed on calibration queries, held-out effects on disjoint evaluation queries before EAB are $13.78\times$--$26.10\times$ larger than those after it. Comparison with intermediate readouts shows that the edit can lose leverage before answer preferences stabilize. Second, we compare counterfactual subtraction across presence, attribute, and relation questions. Subtraction can correct a misleading preference, leave a decision unchanged, or remove a preference supporting the correct answer. Based on these observations, we develop calibrated selective routing, which preserves the original prediction unless calibration supports an intervention for the question type. It improves full-set attribute accuracy by $1.56$--$1.81$ percentage points across three checkpoints, while universal subtraction is neutral or harmful. 

---
# MotionCoach: Constructing Multimodal Guidance for Video Professional Motion Editing 

**Authors**: Siyi Qian, Hanqing Yang, Weihuang Lin, Zihao Yang, Yichen Lu, Yuanhong Zheng, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.12963)  

**Abstract**: Video editing models achieve strong performance in modifying appearance, backgrounds, styles, and motions, empowered by large-scale training data. Beyond general scenarios, professional motions are significant in performance, sports, and martial arts. This paper focuses on professional motion editing, which involves fine-grained motion technique and temporal structure. Given professional motion names but lacking detailed descriptions, existing models often fail to generate professional motions, such as incorrectly lifting the front wheel under a Stoppie instruction. To address this challenge, we introduce MotionCoach, a trained multimodal agent that uses multi-turn tools and performs self-verification, producing a grounded prompt and a reference clip as reliable and sufficient multimodal guidance to inject external professional motion knowledge into video editing models. To supply training data, we build a data pipeline to construct trajectories. We cold-start the agent on MotionCoach-SFT-10k and apply Group Relative Policy Optimization (GRPO) on MotionCoach-RL-2k with rewards on the motion guidance prompt, the motion outcome video, and the format. For evaluation, we introduce VPMEBench, comprising 314 professional motion editing tasks with fine-grained references and annotations. Extensive experiments show that MotionCoach improves the overall score from 55.1 to 65.6 on Seedance 2.0 and from 51.2 to 64.5 on MiniMax-H3. Learning to construct multimodal guidance enables precise professional motion editing without retraining video editing models. 

---
# JoyInside-VLM-Agent: Product-Conditioned Multimodal Policy Learning for Embodied Agents 

**Authors**: Gangyi Li, LuYao Wang, Jingfu Sun, Jingyi Yuan, Hang Liu, Hao Zong, Yujia Hu, Tingting Gao, Jianyong Wang, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.12957)  

**Abstract**: Embodied agents must translate multimodal observations and user requests into executable actions while adapting to product constraints, changing goals, and execution feedback. We present JoyInside-VLM-Agent, a ReAct-style multimodal agent developed through product-conditioned trajectory construction and staged policy learning for robotic platforms and smart hardware. A closed-loop sandbox generates multimodal interaction trajectories covering goal revisions and failure recovery, which are checked against execution outcomes and semantic criteria. Supervised fine-tuning and multi-teacher on-policy distillation precede reinforcement learning with DPER (Dual-Source Privileged On-Policy Self-Distillation with Token-Level Evidence Routing). DPER routes token-level evidence between a frozen privileged teacher and a privileged student sharing the rollout policy's parameters, modulating reward-derived token advantages while preserving their signs. Training rewards combine generative process evaluation, multi-judge rankings, and task-queue consistency. JoyInside-VLM-Agent achieves a BFCL v4 overall score of 77.44, exceeding the evaluated baselines; results on the other public benchmarks are mixed. On JoyInside-Devices and JoyInside-Robotics, the evaluated system, which includes domain-specific skill refinement, leads the baselines on all reported metrics. Two real-world quadruped case studies illustrate visually grounded search, replanning after execution failure, and explicit failure reporting. 

---
# Surfing the Activation Wave: Analyzing the Information and Geometry of LLM Hidden States for Speculative Decoding 

**Authors**: Raghavv Goel, Sudhanshu Agrawal, Mukul Gagrani, Risheek Garrepalli, Mingu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.12939)  

**Abstract**: Decoder hidden states from different layers of large language models (LLMs) often encode complementary information. However, existing speculative decoding methods typically rely on a small set of target-model layers chosen using fixed, position-based heuristics. Selecting an optimal subset of layers is inherently a combinatorial problem, making exhaustive evaluation infeasible even for moderately deep models. In this work, we present a principled framework for multi-layer hidden-state selection for speculative decoding based on two complementary criteria: informativeness, measured through matrix-entropy analysis, and diversity, quantified using centered kernel alignment (CKA). We investigate layer-wise Shannon entropy, aggregate entropy under hidden-state concatenation, and pairwise CKA dynamics across depth, revealing distinct geometric and informational compartments within decoder representations. Motivated by these findings, we propose entropy and CKA-based layer selection algorithms that favor hidden-state sets that are both highly informative and minimally redundant. As part of this framework, we introduce token-entropy filtering, which removes hidden states associated with low predictive-entropy tokens, thereby improving the quality of layer-selection statistics. In EAGLE3-style speculative decoding, our selected sets outperform fixed baseline choices, including the default $[2, N//2, N-3]$ rule, while providing a clearer explanation of why some defaults can remain competitive. Our results provide a practical recipe for multi-layer hidden-state selection for speculative decoding, insights into the geometry and information in LLM decoder states, and a foundation for extending similar analysis to related applications. 

---
# Localized Generative Design of Ship Hulls with Diffusion Models and High-Fidelity Assessment 

**Authors**: Hadi Keramati, Mohammed A. Hannan, Patrick Kirchen, Sina Tousi, Joonas Riekkinen, Jim Pyra, Rajeev K. Jaiman  

**Link**: [PDF](https://arxiv.org/pdf/2610.12870)  

**Abstract**: Practical ship design requires localized refinement and performance evaluation. This paper presents a reward-directed diffusion framework for localized hull optimization. Parametric masking during dataset generation restricts geometric changes to selected variables, including the bulbous bow and forebody, while preserving full-hull continuity. Surrogate-assisted sequential Monte Carlo sampling targets reduced predicted calm-water resistance, with stability retained through training data and displacement matched during screening. An automated viscous free-surface CFD pipeline, validated against measurements for two benchmark container ships, evaluates a generated variant and GenHull E, a hull refined by a naval architect within the same representation. At matched KRISO Container Ship (KCS) displacement, the generated variant has comparable total resistance at $F_n=0.195$ but 45-50% higher resistance at $F_n=0.250$ and $0.260$, missed by empirical screening. GenHull E achieves 6-8% lower total resistance up to $F_n=0.240$ but higher resistance above $F_n=0.250$ as opposing friction and pressure trends reverse the gains. Aggregating resistance across this crossover can conceal tradeoffs that emerge when friction and pressure components are evaluated separately. The assessment reveals an inability to represent a faired bulbous bow and a reward lacking viscous pressure and stability terms. These findings show how high-fidelity assessment informs improved representations and objectives for physically realizable generative designs. 

---
# Co-Modifying Application Code and Deployment Infrastructure: A Causal Study of AI Agent Pull Requests 

**Authors**: Mohammad Wathiq Soualhi, Basem Suleiman  

**Link**: [PDF](https://arxiv.org/pdf/2610.12849)  

**Abstract**: Current software engineering agents are evaluated predominantly inside isolated sandboxes against synchronous local tests. However, deploying features in collaborative repositories requires modifying continuous integration (CI/CD) pipelines, containers, and provisioning manifests alongside application logic. Because agents lack deployment awareness, attempting cross-domain modifications introduces unverified dependencies into asynchronous integration pipelines. To isolate the causal integration overhead of dual-scope pull requests (PRs)-modifying both application code and deployment infrastructure-we evaluated 26,999 PRs across two large datasets (AIDev v5 and MOSAIC-3M) encompassing six agent architectures and an aligned human baseline. Using a deterministic four-tier file taxonomy and an exact 4D strata-matching framework, we isolated integration metrics under conditional exchangeability while controlling for application volume. We find agents operate primarily as application coders or chore workers, with maintenance tasks capturing 17.70% to 25.41% of agent PRs. Dual-scope modifications represent only 4.48% to 4.55% of agentic PRs, yet trigger bimodal integration outcomes. In the first mode, explicitly reviewed contributions experience protracted delays, adding a median penalty of +13.13 hours and multiplying time-to-merge by 3.24x, driven mechanically by CI/CD and provisioning manifests. In the second mode, maintainer gatekeeping exhibits a structural bypass: even for external contributors lacking repository write access, agent identity reduces the odds of human review by 63.7% compared to human peers. Consequently, 43.1% to 56.2% of dual-scope PRs bypass inspection to merge silently in a median of under 31 minutes. These uninspected modifications compromise build verification, demonstrating the need for deployment-aware benchmarks and strict path-based access controls. 

---
# When Generative Models Act as Simulators 

**Authors**: Sevan Brodjian, Michael Hobley, Pietro Perona  

**Link**: [PDF](https://arxiv.org/pdf/2610.12847)  

**Abstract**: A simulator summarizes an environment in a state that users can read and change. Generative models learn representations of the world without explicit specification, but offer no built-in way to edit that world's state. We investigate the conditions under which a generative model can act as a simulator through $\textit{editability}$: whether changing a model's latent representation produces the next-step outputs we get by editing a simulator's state. Plainly stated, our goal is to find a mapping between a generative model's latent state and the simulator state, where the model is trained on observations alone and the mapping is learned afterward. We test two kinds of mappings, both learned from observations paired with simulator states. In one, we train probes to decode the simulator state from the model, then steer along them. In the other, we learn the mapping directly with an inverse map, which can succeed even where steering fails. We study both in two contrasting environments: the discrete board game Othello, and Rayworld, our own continuous world of moving discs seen through partial renderings, closer to physical scenes and video. We find that which mapping succeeds depends strongly on the environment, even though the state is decodable in every variant. In Othello, editability falls smoothly as the game is simplified, and even small rule changes have large consequences. In Rayworld, probes fail to steer along the simulator's continuous state, yet an inverse map sets the model from that state anyway. A model can therefore be driven by a simulator it may have never learned. 

---
# COFFEQA: Counterfactual-Guided Few-Shot Distillation for LLM Question Answering 

**Authors**: Kavinda Kehelella, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2610.12837)  

**Abstract**: Knowledge distillation aims to train a smaller student model to mimic a larger teacher model for resource-constrained environments. However, the process of distillation is typically data-intensive and computationally expensive, particularly for generative tasks. Few-shot distillation offers an alternative by transferring teacher knowledge using limited training samples. However, we observe that few-shot distillation performance is heavily dependent on the choice of these training samples. In fact, we demonstrate that using more uncertain samples near the teacher's decision boundary can significantly deteriorate distillation performance. Motivated by this observation, we propose Counterfactual-guided Few-shot Distillation in Open-book Question Answering (COFFEQA), a novel strategy that aptly uses counterfactuals to provide additional supervision around challenging factual samples. In the generative QA setting, we define a counterfactual as a minimally modified context that induces the teacher to change its response to an alternative answer. We provide geometric insights on how factual-counterfactual pairs can constrain the deviation between the teacher's and student's decision boundaries. We further introduce a new metric, Context Following Difficulty (CFD) that further validates counterfactuals prior to distillation based on agreement with the intended alternative answer and dependence on the modified context. COFFEQA uses CFD-validated factual-counterfactual pairs for few-shot distillation. Experiments across five in-domain and three out-of-domain QA benchmarks using T5 and Qwen2.5 families show that COFFEQA outperforms factual-only distillation baselines under the same few-shot budget. For instance, on TriviaQA, COFFEQA improves over the factual-only baseline by 8.77 average EM points and 9.98 average F1 points under the T5 setting with 32 few-shot pairs. 

---
# LeWAM: End-to-End World and Action Modeling with JEPAs 

**Authors**: Minghao Fu, Tavis Siebert, Eryk Halicki, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2610.12818)  

**Abstract**: Joint-embedding predictive architectures (JEPAs) learn compact latent world models without pixel reconstruction. Existing JEPA pipelines typically train visual representations and action-conditioned dynamics before adding a control policy or solver. We introduce LeWAM, an end-to-end JEPA for control with: (1) a joint training objective over next latent state prediction and action flow matching, and (2) an isotropic Gaussian constraint on the latent state, enabling a single model to support direct policy execution and efficient planning by evaluating sampled actions with its learned latent dynamics. Compared with prior JEPAs in visual planning, LeWAM improves success rate from 28.6% to 89.7% on contact-rich tasks and from 10.9% to 46.2% on long-horizon tasks, and plans 32.3 times faster. On behavior cloning tasks in simulation and physical robots, LeWAM is also competitive with established methods. Interpretability analyses show that LeWAM learns control-relevant representations and action distributions effectively guiding test-time planning and control. 

---
# Efficient Audio-Visual Speech Enhancement via Reinforcement Learning-Based Neural Architecture Search 

**Authors**: Khubaib Ahmed, Ahsan Adeel, Nasir Saleem, Kia Dashtipour, Ahsan Ulhaq, Amir Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2610.12796)  

**Abstract**: Audio-visual speech enhancement (AVSE) improves speech intelligibility in noisy environments but often relies on computationally expensive models unsuitable for resource-constrained edge devices. This paper introduces a reinforcement learning-driven neural architecture search (RL-NAS) framework to discover compact, efficient AVSE architectures. The search space preserves temporal alignment between audio and visual streams, while a novel reward function jointly optimises speech enhancement performance, measured by SI-SNR improvement, and model compactness. Using proximal policy optimisation with constraint-aware action masking, the framework evaluates only 35 candidate architectures and identifies a model with 2.9 million parameters, compared with 22 million in the baseline. The proposed architecture achieves a 14.5 dB SI-SNR improvement, only 0.7 dB below the baseline, while reducing model size by approximately nine-fold. Evaluation on commercial mobile CPUs demonstrates a 4.3-fold reduction in real-time factor, supporting practical on-device deployment. Compared with random search, RL-NAS achieves 92% higher search efficiency. Architectural analysis reveals that shallower but wider temporal convolutional blocks and aggressive visual pathway pruning can maintain enhancement performance while substantially reducing computational requirements. These findings demonstrate the potential of reinforcement learning-guided architecture search for developing lightweight, efficient and deployable multimodal speech enhancement systems. 

---
# Meta-Reflective Evolution Effectively Improves Production AI Systems 

**Authors**: Seung Won Wilson Yoo, Chang Zeng, Aditi Somayajula, Sahithi Ankireddy, Rohan Kosalge, Avishree Khare, John Mitchell, Aditya G. Parameswaran, Rajhans Samdani, Anupam Datta  

**Link**: [PDF](https://arxiv.org/pdf/2610.12794)  

**Abstract**: Large Language Model (LLM)-guided evolutionary optimization has successfully improved prompts, programs, and even entire AI systems. Most optimization methods select a candidate from the search frontier, then mutate it in one LLM pass based only on its failures. This focus on the frontier can stall on production AI systems, where isolated failures reveal little about complex, hidden interactions. We introduce CoCoEvolve (Cortex Code Evolve), an evolutionary optimizer built around what we call meta-reflection, wherein an agentic mutator (Cortex Code) selectively retrieves evidence from the entire evolutionary history and probes the live system before verifying and proposing mutations. We apply CoCoEvolve to optimize three classes of production AI systems: data agents, LLM-powered data pipelines, and data engineering workflows. On DABStep, CoCoEvolve tops the public leaderboard, reaching 92.8% held-out hard accuracy compared to 78.4% for Meta-Harness and 30.4% for GEPA under matched conditions. Across six LLM data pipeline workloads, it outperforms every evaluated baseline. On data-eng-bench, CoCoEvolve lifts test-set solve rate from 59.1% to 77.4%, past the best baseline of 73.6%. It also has 11 percentage points higher accuracy than GEPA on HotpotQA while remaining competitive with specialized optimizers on established mathematical and systems benchmarks. Matched ablations confirm that each component of CoCoEvolve contributes to these gains, demonstrating the value of meta-reflection for optimizing production AI systems. 

---
# REDaCT: Collapse-Free Unlearning in Large Reasoning Models 

**Authors**: Joshua Maccoby, Wendy Hui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.12769)  

**Abstract**: Unlearning in large reasoning models (LRMs) requires removing forget-set information from both the reasoning trace and the final answer. Existing approaches achieve this objective at the cost of reasoning collapse, where the model fails to produce a coherent chain-of-thought (CoT) and does not properly terminate its reasoning process. To address this challenge, we propose REDaCT (REgenerate via Detection and Corrective Tuning), an inference-time unlearning framework for LRMs. REDaCT consists of two key components: (1) a binary leak detector that identifies sentences in the reasoning trace that leak forget-set information, and (2) an embedding optimization module that learns a corrective embedding offline, such that when injected at the first detected leak during inference, it guides subsequent reasoning towards sentences that do not leak forget-set information. Experimental results show that REDaCT completely avoids reasoning collapse, substantially improving reasoning preservation compared to state-of-the-art unlearning methods while maintaining perfect answer forgetting and near-perfect CoT forgetting. Our code is available at: this https URL. 

---
# Semantic Views: Certified Reuse of LLM Labels Across Semantic Queries 

**Authors**: Oktie Hassanzadeh, Dharmashankar Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2610.12768)  

**Abstract**: Semantic query engines evaluate natural-language filters with a large language model, and every evaluation is a costly oracle call. Optimizers that reduce these calls face a dilemma: cascades that trust a cheap proxy miss their precision and recall targets on many queries, methods with finite-sample guarantees save little, and all of them discard the oracle's labels when the query ends. We present SemViews, a certified semantic filter that orders tuples by signals that cost no oracle calls (the labels that earlier queries left behind as semantic views, a local embedding, and a helper model's score), refines that order with labels it must pay for anyway, and certifies its answer with fresh stratified oracle samples. The answer meets user precision and recall targets with probability at least $1-\delta$ for every workload and every catalog, so a stale or misleading view can cost oracle calls but never accuracy. On four workloads replayed against 725,000 recorded oracle labels, SemViews meets both targets on 99.63% of queries and saves 28%-45% of the oracle calls, whereas the only prior method in our study with the same guarantee needs 1.32x-1.71x as many calls. When a question returns in different words, views cut its calls by a further 27%-36%, with both targets met on at least 99.8% of such queries; similarity caches and entailment-based reuse meet them on only 17%-47%. 

---
# Trajectory Prediction for Autonomous Driving in Road Construction Zones Using Human-in-the-Loop Driving Data 

**Authors**: Shuo Zhang, Aman Sunesh, Semiha Ergan, Kaan Ozbay  

**Link**: [PDF](https://arxiv.org/pdf/2610.12747)  

**Abstract**: Ego-trajectory prediction in urban roadway work zones remains an underexplored topic in autonomous-driving research, with limited dedicated data and systematic evaluation. Roadway work zones involve temporary changes in the lane layouts and connectivity of the roads, complicating interactions with surrounding vehicles and workers. These changes make future vehicle trajectories difficult to predict. To address these challenges, we collect WZDrive-Data, a multimodal dataset comprising 1,183 work-zone driving segments from 33 drivers, using a human-in-the-loop driving simulator with prerecorded worker animations and VR-controlled workers. Driver control and gaze signals are recorded from human participants, while vehicle trajectories and scene information are obtained from the simulator. Our proposed multimodal ego-trajectory predictor, WZDrive, achieves a mean minADE (K=6) of 1.09 m and minFDE (K=6) of 2.12 m. Compared with the best-performing state-of-the-art baseline for each metric in our benchmark, WZDrive reduces minADE (K=6) by 7.18% and minFDE (K=6) by 13.18%. This research addresses a gap in autonomous-driving research on urban work zones by combining human-in-the-loop data collection with multimodal learning from human driving signals and in the context of work zones. 

---
# KITE: A Knowledge-Injected Temporal Encoder for Time-Series Event Forecasting 

**Authors**: Jonas Petersen, Pratik Jawahar, Riccardo Maggioni, Gian-Alessandro Lombardi, Matteo Cozzi, Marek Gajewski, Federico Martelli, Camilla Mazzoleni, Philipp Christian Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2610.12738)  

**Abstract**: Forecasting rare, high-cost events in multivariate time-series systems is hard because their precursors are seldom observed, so a purely data-driven model has little to learn from. The engineers who run these systems, however, hold knowledge that precedes such events, information absent from the scarce labels. We call this a semantic prior and inject it into a self-supervised event forecaster through two switchable mechanisms, a covariate mechanism and an attention-bias mechanism, each the identity at its null, so one architecture serves every prior type with no per-dataset architecture search. A prior helps when it supplies information the observed window does not already contain: in a short context an informative prior beats no prior in-domain (pooled h-AUROC over the no-prior baseline: a periodicity clock +14%, p<0.001; a conservation residual +47% at a 0.25 label fraction), and a working prior transfers out-of-domain (+53%), substituting for feature transfer, while priors the window already recovers stay inert and the attention-bias mechanism shows no measurable gain. KITE is a switchable, attributable route to contextual conditioning, and a substrate for scaling toward cross-domain, pretrained time-series foundation world models that carry their governing structure explicitly. 

---
# Unsupervised Multimodal Reward Modeling 

**Authors**: Yueying Li, Jingxuan Fan, Kianté Brantley, Sham M. Kakade, Hanlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.12707)  

**Abstract**: Vision-language reward models are trained on annotated preference pairs, which are expensive to collect, can disagree, and are ranked by benchmark accuracy. We build one with no human preference label: a linear head on an intermediate layer of a frozen Qwen3-VL backbone, trained to score a web caption above a perturbed version of the same caption; labeled data enters only to select the layer. The layer is the decisive choice, worth 5.8 points over the last layer, and one recipe transferred without re-tuning beats or at least matches GPT-4o on three benchmarks at 32B, 6.6 points ahead on VL-RewardBench, from a few thousand parameters at about a thousandth of fine-tuning compute. At an equal pair budget, perturbations teach a frozen head 3 to 5 points more than curated RLAIF-V pairs. Because the backbone is frozen, the objective is convex, and the perturbation sampler specifies the reward, which the benchmark does not identify: two heads trained on disjoint halves of the same perturbations tie on the benchmark and differ by 6.4 points in hallucination judging. Retraining the head on the policy's own verifier-labeled candidates cuts caption hallucination by 5.1 CHAIR points and lifts image-generation pass rate by 5.4, while scoring lower on every reward benchmark.
Together, an intermediate-layer probe, free same-image perturbations, and a verifier-driven bootstrap turn give a recipe for multimodal reward modeling that needs no annotator, costs a fraction of fine-tuning, and is specified by the comparisons it is trained on rather than by a benchmark rank. 

---
# GeoFitBench-LLM: A Benchmark for Assessing Large Language Models' Capabilities in Geospatial Data Requirements Elicitation for Geo-Analytic Tasks 

**Authors**: Amirhossein Vahdat, Thierry Badard, Luc Lamontagne  

**Link**: [PDF](https://arxiv.org/pdf/2610.12669)  

**Abstract**: Large language models are becoming central to GeoAI systems, yet their evaluation rarely considers the elicitation of geospatial data requirements. This omission is critical as retrieval oriented geo-analytic tasks cannot be supported reliably without first identifying the necessary datasets and constraints. GeoFitBench-LLM is introduced as a benchmark for assessing whether LLMs can infer the datasets, the supporting resources and the constraints required by geo-analytic requests in natural language. The benchmark is structured around two reasoning dimensions, Spatio-Temporal Relational and Semantic Grounding, reflecting long-standing challenges in geospatial information retrieval and related natural language tasks. Ground truth requirements are encoded in a fixed, provider-independent schema for reproducible evaluation of resource identification and attribute-level specification. Experiments on seven models under minimal prompting indicate that identifying the necessary resources is often achievable, with GPT-5.4 Thinking achieving the best result by recovering the complete set of resources in 66.67% of benchmark requests. Results further indicate that accurate fitness-for-use specification remains more difficult as inferential demands increase. 

---
# Recurrent Harness for Agentic Continual Learning 

**Authors**: Yuxin Xiao, Zhiyun Lu, Jianyu Wang, Yangtian Zhang, Yang Song, Qifan Wang, Bailin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.12668)  

**Abstract**: Language model agents increasingly operate over long interaction streams, requiring online adaptation to distribution shifts through continual learning. However, existing agent harnesses typically either evolve their states only through forward information flow or rely on additional replay or validation data for harness optimization. To address this, we propose ReHarness, a recurrent harness framework for agentic continual learning that complements forward read and write with backward revision under a strictly online streaming protocol. We ask what information could have been captured differently in the harness state to better inform the current instance in the stream. Specifically, after each instance, we leverage the resulting trajectory as feedback to revise the reflections retrieved for that instance, thereby propagating feedback backward through the textual harness state. Across three continual learning benchmarks and three language model backbones, ReHarness consistently outperforms strong raw-context baselines, memory- and reflection-based methods, and prompt optimization approaches. Further analysis shows that ReHarness produces source-grounded and meaningful revisions, corrects overgeneralized or outdated experience under distribution shifts, and improves both stability and plasticity during continual learning. 

---
# Incremental Retrieval-Augmented Generation for Human-in-the-Loop Decision Support 

**Authors**: Federico Mazzoni, Andrea Failla  

**Link**: [PDF](https://arxiv.org/pdf/2610.12666)  

**Abstract**: We introduce iRAG, a human-in-the-loop decision-support system combining Retrieval-Augmented Generation (RAG) with Hybrid Decision-Making. Unlike conventional RAG systems, relying on static Knowledge Bases (KBs), our proposal addresses settings where the KB is initially empty and is progressively populated through human decisions. The system stores previous decisions as retrievable precedents and moves from silent observation to evidence-grounded contestation and, finally, to autonomous handling with abstention. Against an independent baseline, iRAG matches final-decision error under 10% drift without retrieval decay and reduces it under 40% drift across all configurations. Retrieval decay improves performance on changed knowledge while increasing error on stable knowledge, evidencing a stability--plasticity trade-off rather than a general advantage. Human oversight is an architectural provision of the workflow; its usability with real human users remains to be evaluated. 

---
# Kronecker Product Attention: Memory-Efficient KV Compression via Structured Factorization 

**Authors**: Bahareh Nikpour, Hossein Rajabzadeh, Maryam Dialameh, Vahid Partovi Nia, Parsa Omidi  

**Link**: [PDF](https://arxiv.org/pdf/2610.12657)  

**Abstract**: Efficient autoregressive generation is increasingly limited by the memory footprint of the key-value (KV) cache, which grows linearly with sequence length, batch size, and model depth. Existing approaches such as Multi-Query Attention (MQA), Grouped-Query Attention (GQA), Multi-Head Latent Attention (MLA), and Tensor Product Attention (TPA) reduce this cost through different trade-offs, but often sacrifice expressivity, require additional positional-encoding machinery, or impose restrictive low-rank structure. We propose Kronecker Product Attention (KPA), which represents key and value activations as sums of Kronecker products of compact, token-dependent matrix factors. KPA strictly generalizes TPA: rank-one vector-outer-product factorization arises as a degenerate Kronecker split, while non-degenerate matrix factors can represent richer block-structured patterns. We further show that a structured rotary embedding operator matched to the Kronecker representation preserves the relative-position property of RoPE while rotating only the compact factors, removing the need for a decoupled RoPE branch as in MLA. Empirically, KPA matches or exceeds TPA, MHA, MQA, GQA, and MLA across four model scales and nine downstream benchmarks under identical training conditions, while achieving up to 10x KV cache compression. The Kronecker head split is a tuneable hyperparameter that allows practitioners to navigate the compression-accuracy trade-off at design time without architectural changes. 

---
# Cancerous Backdoor Attack: Gradual Activation via Invisible Tokens and A Stage Controller 

**Authors**: Tahmid Imtiaz Imu, Emama Nahid, Prodipta Sen Amartya, Ahsanul Jobayer, Yutong Song, Jiang Wu, Pengfei Zhang, Honghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.12644)  

**Abstract**: Backdoor Attacks on Large Language Models (LLMs) typically follow a trigger-response pattern, where satisfying a trigger condition activates a single terminal behavior. However, this single-point backdoor activation does not explore attacks in which malicious behavior develops gradually through multiple intermediate stages before reaching the terminal activation stage. We present Cancerous Backdoor Attack, that gradually corrupts an LLM's reasoning over a sequence of interactions by leveraging invisible tokens and a stage controller. Specifically, invisible tokens provide hidden stages to the model without appearing in the user's visible request and the stage controller tracks the attack's progression during inference to realize gradual activation. We evaluate Cancerous Backdoor Attack across several reasoning datasets. The attack consistently produces gradual reasoning corruption across models and tasks. We compare Cancerous Backdoor Attack with prior backdoor attack baselines to demonstrate our proposed attack outperforms the baselines. Our results indicate that reasoning corruption can develop gradually before it becomes visible, making answer accuracy alone insufficient for detecting backdoor attacks. 

---
# Machine Learning and Time-Series Foundation Models for Day-Ahead Electricity Forecasting in a University Building 

**Authors**: Amey Vasulkar, Samuel Kernan Freire, Arie-Willem de Leeuw, Tadeo-Baldiri Salcedo-Rahola  

**Link**: [PDF](https://arxiv.org/pdf/2610.12638)  

**Abstract**: Day-ahead electricity forecasts can help building operators schedule heating, ventilation, and other services around anticipated demand. Choosing a forecasting model also requires balancing accuracy with its forecast-time inputs, local training, and computational requirements.
We benchmark statistical, machine-learning, recurrent neural-network, and pretrained TimesFM models for a university building. We use chronological validation and evaluate all configurations on the same 57 test days, comparing accuracy across workdays, weekends, and holidays alongside local fitting and forecasting time. Weather covariates use retrospective observations.
TimesFM 3.0 with covariates achieves the lowest overall errors, reducing mean absolute error by approximately 43\% relative to weekly seasonal-naive forecasting. XGBoost remains competitive, although paired daily comparisons do not establish a clear mean-absolute-error advantage for TimesFM. TimesFM also performs strongly using only past electricity consumption. Adding covariates improves its holiday forecasts but slightly worsens workday and weekend forecasts, leaving overall mean absolute error almost unchanged. TimesFM requires no building-specific weight training, while XGBoost offers rapid local fitting and inference. The evaluated LSTM takes longer to fit without outperforming either approach.
An exploratory historical evaluation reveals seasonal variation, while a separate weather-source comparison produces only small changes in the selected TimesFM configuration's accuracy. These results support pretrained TimesFM as a strong initial forecasting baseline for this building, with XGBoost as a competitive locally fitted alternative. Covariates are particularly useful when improved holiday forecasts justify the additional input preparation. 

---
# Mesh-Native Tokenisation: Retrofitting Foundation Models to Unstructured Data 

**Authors**: Georg Maierhofer, Miles Cranmer, Payel Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2610.12618)  

**Abstract**: Physics foundation models are an increasingly popular approach for building versatile emulators for spatiotemporal systems. They require tokenisation layers that lift physical field data into a latent representation suitable for dynamic processing. A typical choice for this layer is a convolutional encoder, which lifts field information in a locality-aware way and can be trained very efficiently. However, this choice cannot be applied to a large class of scientific computing problems where data are represented on unstructured meshes, for example simulations on domains with curved boundaries. Such data typically require graph neural network architectures, which are more expensive to train and, in the form currently used, encode information differently from their CNN counterparts, thus limiting the portability of physics emulators between these data regimes. In this work we introduce Mesh-Native Convolution (MNC), a locality-aware tokenisation layer that encodes unstructured mesh-based data in a way that is analogous to CNNs. This approach leads to models that can be trained once on mesh data and then deployed to arbitrary unstructured meshes, even if these are irregular, with minimal retraining. We show that this tokenisation approach can be used to retrofit grid-pretrained foundation models efficiently on unstructured meshes and provide theoretical results to confirm the consistency of this approach with classical CNN encoders. We demonstrate the favourable performance of this retrofitting approach, which in many cases reaches acceptable accuracy even in zero-shot deployment, and outperforms existing baselines after only a few epochs of fine-tuning, thus providing an efficient bridge between the two data regimes. 

---
# Self-Preservation in Language Models 

**Authors**: Dongkyu Cho, Rumi Chunara, Ali Behrouz  

**Link**: [PDF](https://arxiv.org/pdf/2610.12602)  

**Abstract**: Continual learning (CL) seeks to adapt pretrained models while preserving prior capabilities. Yet the tasks supplied for preservation cover only part of what a model knows, while forgetting can extend beyond them. Recent work shows that language models can supplement this limited evidence by generating useful replay without access to their original training data. This raises a question about self-preservation: can a model help choose preservation targets that protect its own capabilities more broadly? To study this question, we organize task descriptions and their examples into Task Banks, whose entries can be selected and combined to guide replay generation by a frozen model. Continual instruction tuning with Qwen models of varying sizes yields three findings. First, the new task and learning rate shape which prior capabilities degrade, while tracking previously correct answers reveals losses that average accuracy can conceal. Second, generated replay reduces forgetting on tasks outside the bank. Combining partial banks strengthens this protection, improving mean retention of previously correct answers over either bank alone across the tested sizes. Third, the model can also help choose replay targets: selection informed by the new task improves mean held-out retention over random selection at 4B, with similar new-task gains, although smaller models show no corresponding advantage. Under a fixed example budget, however, replacing new-task examples with replay improves preservation at the cost of smaller new-task gains. Together, these findings show that models can support broader self-preservation by generating replay data and, in some settings, choosing which tasks to replay. 

---
# AutoIndexer: Flexible-Order Decoding by Training Causal Models on Chains of Edits 

**Authors**: Shu Ishida, Aliasghar Khani, Tianyu Zhang, Cong Luong, James Seale Smith, Adam Gaier  

**Link**: [PDF](https://arxiv.org/pdf/2610.12545)  

**Abstract**: Autoregressive decoding is less suited to non-causally ordered data, e.g., cross-references or future dependencies, since their outputs cannot be revised post hoc, unlike iterative human reasoning. Diffusion LLMs partially mitigate this limitation but still cannot flexibly insert or remove arbitrary numbers of tokens from a pre-generated context. We design AutoIndexer as a generative model with a novel training and decoding mechanism that simulates a Chain of Edits, allowing the model to flexibly modify its prior outputs in hindsight via token insertions, substitutions, and deletions. The model learns to predict marker tokens for opening and closing edits, simultaneously moving a cursor to positions in the existing token sequence, analogous to a text editor. Training sequences are obtained by perturbing ground-truth labels with inverse chained edit operations, and the model learns to edit by undoing those inverses. The perturbation training allows AutoIndexer to be robust to erroneous context and adaptable to editing tasks such as code correction. We demonstrate that our method exhibits post-generation context-editing capabilities that most autoregressive and diffusion models lack without compromising performance on standard LLM benchmarks. The code is available at this https URL. 

---
# CircuitTree: Composing Circuits into A Syllable Transcriber 

**Authors**: Jiankun Wei, Ewan Dunbar, Gerald Penn  

**Link**: [PDF](https://arxiv.org/pdf/2610.12520)  

**Abstract**: The goal of circuit discovery is to decompose a monolithic neural network into tiny modular subnetworks which are interpretable in the sense that each does an identifiable subtask. We develop a practical method for composing arbitrarily many circuits into a single model. We propose a greedy, density-scaled composition algorithm that overcomes limitations of previous attempts to compose circuits by naive union, then generalize it to $N$ circuits via CircuitTree, a new data structure built on a $\Theta(N\log N)$ tree-compose schedule. We compose circuits pruned from HuBERT for classifying individual phonemes into a syllable classifier, a capability none of the 39 phoneme circuits has individually. Chaining CircuitTrees into a hierarchy-of-circuits pipeline, we transcribe open syllables at 64.8\% accuracy, within 1.2 points of a 66.0\% ceiling that shares no circuitry at all, with 99.5\% of circuit components shared. 

---
# Energy-Efficient Spiking Neural Networks for Sensing-Aided Beam Prediction 

**Authors**: Nhan Thanh Nguyen, Phuong Nam Tran, Mengyuan Ma, Huy T. Nguyen, Ahmed Alkhateeb, Markku Juntti  

**Link**: [PDF](https://arxiv.org/pdf/2610.12507)  

**Abstract**: Sensing-aided beam prediction can reduce the beam-training overhead of millimeter-wave (mmWave) systems by leveraging environmental information from complementary modalities such as vision and position. However, the resulting sensing and inference pipeline can be energy-intensive, motivating the use of energy-efficient spiking neural networks (SNNs) as an alternative to conventional artificial neural networks (ANNs). In this work, we develop SNN architectures for beam prediction and systematically compare them with ANNs using position-only, vision-only, and multimodal inputs. Our results show that compact ANNs consistently outperform SNNs for position-only sensing. In contrast, for vision and multimodal sensing, SNNs achieve higher accuracy than ANNs at similar energy costs and require up to an order of magnitude less energy at a comparable accuracy. For example, the multimodal SNN achieves 58.08% top-1 accuracy, outperforming the closest-energy ANN by 18.11 percentage points. We further show that SNN accuracy is limited by the binary communication between layers rather than by model size. These results demonstrate the potential of SNNs for energy-efficient sensing-aided beam prediction while highlighting their accuracy limitation. 

---
# Kairos: Toward a Byte-Level Foundation Model for Every Modality 

**Authors**: Fabien Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2610.12495)  

**Abstract**: Small language models trained on a single local machine leave little room for architectural trial and error. Kairos combines bidirectional gated-DeltaNet linear attention and sliding-window attention over shared projections (LiZAttention2), block-wise Attention-Residual aggregation, a sparse Mixture-of-Experts (MoE) feed-forward layer and a multi-scale byte-level codec, trained with a uniform-noise denoising objective over one byte space shared by text, image, audio, video, lidar and control streams; this report asks whether the combination optimizes at all before asking whether it is useful. We give code-derived algorithms for each component, including a cross-session recurrent-state memory gate, then run convergence sanity checks: overfittinga single 440-byte sentence at 1--15M parameters, text only, with three seeds for codec and attention choices and single-run sweeps for seven further factors. The two attention variants tested at 3\,000 steps (LiZAttention2 and window attention alone) reach a minimum loss near $10^{-3}$. After 600 steps, the mean minimum loss ranges from 0.018 (patch codec) to 0.354 (depthwise convolutional codec) with an identical ranking on three seeds. An apparent advantage of shared over separate attention projections closes with more steps or more diverse data and reverses when the projections are frozen: it is a convergence-speed effect, not a capacity effect. These checks suggest that the implementation optimizes; they say nothing about generalization, multimodal training or comparison with existing models, which a planned pretraining run is meant to address. The Discussion sizes a 200M-total/25M-active target analytically; none of its throughput figures is measured. Code is available at this https URL . Python package at this https URL . 

---
# Imagine the Future, Internalize the Gist: Efficient VLA Reasoning via Internalized Spatiotemporal Imagination 

**Authors**: Shenglan Li, Zhendong Mi, Hengyi Zhu, Jingwu Luo, Chun Kit Chan, Geng Yuan, Yanzhi Wang, Pu Zhao, Shaoyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.02626)  

**Abstract**: Vision-language-action (VLA) models increasingly incorporate intermediate reasoning to improve robotic manipulation, yet existing approaches primarily reason about observed states without explicitly anticipating future scene evolution. Extending such reasoning to explicit future rollouts at every inference step, however, introduces substantial computational overhead. We propose IG-VLA, a VLA reasoning framework that enables models to imagine the future and internalize the gist. Our Latent Spatiotemporal Reasoning learns to imagine task-relevant future scene evolution directly in visual representation space, guiding action prediction without costly pixel-level video generation. To further reduce inference overhead, we introduce Scene Gist Memory, which internalizes reasoning-derived scene-behavior associations into a compact Scene Gist Token, preserving the benefits of future reasoning while bypassing explicit future imagination at inference. Extensive experiments on LIBERO, LIBERO-Plus, and VLABench demonstrate the effectiveness and efficiency of IG-VLA. On the LIBERO-Plus Language suite, both the reasoning and gist policies outperform the strongest baseline by nearly 6% in success rate. The gist policy also achieves up to 6.38x speedup over baselines, reducing inference latency from 1081ms to 169.5ms per action chunk on a single NVIDIA A6000 GPU. These results demonstrate that future spatiotemporal reasoning can be effectively internalized for efficient VLA deployment. 

---
