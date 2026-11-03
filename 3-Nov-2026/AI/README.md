# From Natural Language to Exact Cover: A Neuro-Symbolic Approach to Zebra Puzzles 

**Authors**: Paulius Skaisgiris, Thomas Pammer, Veronika Semmelrock, Mykyta Ielanskyi, Maximilian Heisinger, Erich Kobler  

**Link**: [PDF](https://arxiv.org/pdf/2611.03641)  

**Abstract**: Chain-of-Thought (CoT) generation has substantially improved the performance of Large Language Models (LLMs) on complex reasoning tasks, including code generation, data analysis, and exam-style question answering. Despite these advances, purely neural LLMs continue to struggle with elementary logical reasoning problems and lack the determinism, soundness, and reliability characteristic of symbolic reasoning systems. Conversely, classical symbolic methods such as SAT solving and exact cover guarantee correctness and completeness, but require problems to be expressed in highly specialized formal encodings, limiting their applicability to natural language inputs. In this work, we present a synergistic neuro-symbolic framework that bridges this gap by combining neural semantic parsing with deterministic exact cover reasoning. Given a Zebra-style logic puzzle in natural language, an LLM translates entities and constraints into structured tool calls that construct an exact cover instance. A symbolic solver then computes the solution and provides feedback to iteratively refine invalid, under- or overconstrained encodings, exploiting the uniqueness property of Zebra puzzles. We conduct a comprehensive empirical evaluation across multiple parameter scales, post-training paradigms, and LLM families. The results demonstrate that our hybrid approach consistently outperforms strong plain neural baselines, including CoT prompting, as well as recent neuro-symbolic methods. 

---
# WebHarbor: Docking Real Websites for Evolving GUI Agent Environments 

**Authors**: Zhaoyang Wang, Qianhui Wu, Boyu Gou, Demi Wang, Jiacheng Zhang, Peichun Hua, Qize Li, Shi Qiu, Yifan Jiang, Yifan Zhang, Zexu Jin, Baolin Peng, Bolin Shen, Chetan Bansal, Dehan Kong, Deren Lei, Dongshen Peng, Fengqing Jiang, Hao Cheng, Haoqing Wang, Jiaming Shan, Jicheng Wang, Jingguang Zhou, Jixuan He, Kaide Zeng, Kangyu Zhu, Lama Moukheiber, Mufan Qiu, Peng Xia, Qihan Sun, Qingwei Lin, Qiushi Sun, Ruichen Zhang, Shuqing Luo, Si Qin, Sicheng Fan, Tao Xiong, Wenlin Yao, Xinyu Zhang, Xinyue Zeng, Xuanrui Li, Xuchao Zhang, Yian Wang, Yichao Liu, Yiming Liang, Yiyang Zhou, Yizhu Jiao, Yuetai Li, Yurun Chen, Yuxuan Zhang, Zheyuan Yang, Zhiyuan Yao, Zhongyang Li, Zihao Li, Zili Wang, Jianfeng Gao, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.03634)  

**Abstract**: GUI web agents require realistic and interactive environments that remain stable during both evaluation and training. Live website evaluation suffers from changing content, access restrictions, and limited coverage of authenticated workflows, while existing synthetic environments for training often lack functional depth and visual fidelity. We propose WebHarbor, a suite of 153 executable and dockerized website environments that faithfully reconstruct the visual interfaces, content, and functional workflows of real-world web services under deterministic control. We build WebHarbor through task-driven human-agent collaboration, where environments evolve as new tasks expand their functional scope, coding agents implement the required changes, and human contributors audit visual fidelity, functional correctness, and task validity. The resulting suite comprises 3,518 tasks spanning multi-step and post-authentication workflows, where existing works often fail to cover. Beyond reliable evaluation, WebHarbor provides high-fidelity synthetic web environments for cost-efficient reinforcement learning (RL). We demonstrate that a Qwen 4B-level GUI agent trained with RL in WebHarbor generalizes to unseen tasks and performs competitively with leading open-source GUI web agents across four benchmarks. The code and data are publicly available at this https URL. 

---
# Strategist: Evolving Reasoning with Agents 

**Authors**: Taiming Lu, Qi Xu, Zhigang Hua, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.03618)  

**Abstract**: A model is only half of an AI agent system. The other half is the inference strategy: the reasoning procedure that turns model calls into a response. Today that second half is written by hand, and the experience of building one strategy never carries to the next. We introduce Strategist, a meta-agent that evolves inference strategies across tasks. Each strategy is written as a blueprint,executable code that other strategies can reuse, and together the blueprints form a library. Given a task and a small development set, Strategist retrieves related blueprints, writes candidate strategies, tests them on real samples, and revises them from the results. Every candidate then enters the library, so later tasks build on earlier discoveries. On 30 benchmarks, Strategist beats the strongest baseline on 24 and matches its average accuracy (74.7% vs.74.0%) at 39% lower inference cost. Its strategies also transfer to unseen tasks and other models without further search. The part of an AI system that is not improved on its own now does. 

---
# DiabAgent: Toward Agentic Physiological Control 

**Authors**: Chengxi Li, Shaoxuan Li, Deyao Hong, Qingle Liu, Yi Huang, Puheng Bian, Minzhi Xu, Lanyue Zhang, Mingze Sun, Xin Li, Calvin Xiao, Shengyuan Wang, Huan-ang Gao, Jiazheng Xu, Kaisen Yang, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.03601)  

**Abstract**: As pretraining scales, medical language models have advanced rapidly in clinical question answering. Their ability to achieve effective physiological control through sustained intervention still lacks systematic quantitative evaluation. Type 1 diabetes provides a concrete setting: impaired insulin secretion creates dependence on exogenous insulin, with dosing continually adjusted using information about meals, exercise, illness, and glucose measurements. We introduce DiabAgent, a research suite for agentic physiological control comprising the Agentic Physiological Control (APC) framework, the DiabetSim simulator, and the SweetBench benchmark. APC controls glucose through medical-device tools, using ongoing observations to infer individual states, accumulate personal experience, and choose treatment actions. We build DiabetSim to simulate different patients, events, and device conditions for scalable virtual testing and training. Using DiabetSim, we construct SweetBench by instantiating 32 scenario designs as 212 glucose-control tasks. With Claude Opus 5.5 and a fixed decision interval of one hour, APC achieves a COGI of 73.5, compared with 71.2 for the open-source closed-loop baseline OpenAPS. Ablations further show that individual adaptation and contextual information improve glucose control. Combining high-frequency rule-based control with lower-frequency agent decisions shows potential for balancing control performance and computational cost. Our work provides a framework, simulator, and evaluation foundation for agents to participate in human physiological control. 

---
# Four Bridges: Measuring Deception Propensity in a Multi-Agent Game 

**Authors**: Raphael Casey, Michael Simeone, Alberto Tosato, Florian Ernst, Kemal El Moujahid, James Tamplin, Tommaso Tosato  

**Link**: [PDF](https://arxiv.org/pdf/2611.03600)  

**Abstract**: AI agents are increasingly deployed in settings where they interact and coordinate with other agents, yet honesty is mostly measured in single-turn, single-agent benchmarks that pressure the model to deceive. We study deception that emerges in multi-agent interaction without any instruction to deceive. We introduce Kradle, a platform for evaluating AI models, and Four Bridges, an information asymmetry game in which four agents distribute across four rooms: one is the death room, the others hold food worth more if not shared. One agent, the informed agent, knows which room is the death room and can withhold or disclose that information. We evaluate eight frontier models as the informed agent. Deceptive behavior appears in every model and in 73% of all runs (6% to 100% per model), and does not decrease with capability tier within a model family. When the informed agent does disclose, the warning is often challenged, and agents that challenge it enter the death room twice as often as those that do not. Telling all agents that one of them knows the death room collapses deception in six of eight models; telling the informed agent that the room is permanently fatal moves only one. On an open-weight model, an honesty steering vector reduces deception to 0% and adversarial honesty training to 6%. The platform and game provide a testbed for measuring deceptive behavior and comparing interventions. 

---
# FedReady: Bounded Agents for Federation Formation 

**Authors**: Ziyue Xu, Holger R. Roth, Chester Chen, Peter Cnudde  

**Link**: [PDF](https://arxiv.org/pdf/2611.03586)  

**Abstract**: Federated learning (FL) runtimes standardize aggregation and deployment, but each new study still requires federation formation: selecting task-eligible sites and preparing their local data and task code for federated training or analysis. This preparation recurs when the target changes, while raw records and site-private mappings are excluded from cross-site FL messages. We present FedReady, a system that performs federation formation with bounded LLM agents. Agents propose shared contracts, local adapters, and task code; the runtime owns workflow state and accepts artifacts only after checks tied to their digests. Candidate site discovery prioritizes recall; local eligibility verification checks task support before site admission. Once participating sites and artifacts are fixed, conventional federated training or analysis proceeds without readiness agents. Across 39 simulated retinal sites and four tasks, discovery reaches recall 1.00 at precision 0.65; local eligibility verification removes all 17 false positives without losing an eligible site-task pair. The participating sites supply 45,575 task-record instances and complete four 100-round jobs, including one with a runtime-generated contract. Across ten patient cohorts, preparation supports seven patient populations and holds endpoints at three sources as unsupported. Comparison with manually prepared reference populations and analysis (the manual reference) gives a maximum survival-curve difference of 0.001133. We also analyze 19 development failures, targeted classification ablations, and standalone and heterogeneous agent configurations. These results establish the feasibility of task-specific data preparation before ordinary FL training or analysis. 

---
# An Agentic Framework for Automated Power System Studies and Grid Connection Assessment 

**Authors**: MaoXun Huang, Kshitij Kapoor, Eli Louise  

**Link**: [PDF](https://arxiv.org/pdf/2611.03534)  

**Abstract**: Connecting new generation and demand to the grid depends on two scarce skills: checking applications against a large, changing body of regulation, and running the network studies that decide where and how a project connects. We present Grid Agents, an AI system that brings both together, and the methodology we use to evaluate it. The Grid Ontology is its knowledge system. It represents the regulation, standards and engineering knowledge that grid work depends on, makes it searchable, and evolves as the agents learn from use. Every answer drawn from it rests on evidence the agents cite. Grid Studio is an agent system specialised for power-system engineering. It works in its own engineering environment with domain tools and a study framework, draws on the Grid Ontology, and relies on established solvers for every number it reports. In both, engineers make the decisions. We bring eight published power-system benchmark families, previously scattered across separate papers, into one suite of 519 tasks, and we verify not only each answer but the work behind it, by replaying the agent's tool calls. The Grid Ontology cites the right regulatory evidence more often than a general coding agent or retrieval pipelines, leads on LegalBench-RAG and ranks first on the public leaderboards of the Enterprise RAG Challenge and ViDoRe V3. Grid Studio scores highest on the 519 power-system tasks and is the fastest of all the systems we compare, both by a statistically significant margin. With an optimiser in the loop, its reinforcement packages cost 33-67% less on average than those the agent designs alone. 

---
# Grading the Unannotated: Insights from the WorkRB Skill Extraction Challenge at RecSys-HR 2026 

**Authors**: Warre Veys, Luis Gasco, Jens-Joris Decorte  

**Link**: [PDF](https://arxiv.org/pdf/2611.03525)  

**Abstract**: Skill extraction benchmarks score models against a few human-annotated positives and treat every other skill in the vocabulary as equally wrong. At the scale of ESCO (13,939 skills) this binary view can reward models that rank the annotated positive high for the wrong reason, and hides semantically nonsensical predictions. The WorkRB Challenge at RecSys-HR 2026 replaces it with graded relevance: every candidate skill in a model's ranking window carries a 0 to 4 grade, and the leaderboard scores nDCG@100 against these grades. Since ternary judgements proved too coarse for objective annotation, relevance is decomposed into a cascade of four binary questions (same domain, activity mentioned, right granularity, replacement of a gold positive) that a smaller LLM judge can answer reliably; the labels are validated with blind human and frontier-model re-annotation. This paper documents the annotation methodology, the construction of five benchmark datasets (four with vacancy sentences as queries, one with lexically distant skill surface forms), the validation of the LLM cascade, and insights from the submissions. The field converged on multi-stage retrieve-and-rerank pipelines, whose gains come from clearing nonsense out of the ranking window rather than finding more positives: at the top of the leaderboard, the graded ordering is uncorrelated with what binary evaluation would have said. Graded labels expose much bigger gaps between top submissions and baselines than binary evaluation does, and show that semantic-understanding skill recommendation is still wide open. The clear next direction is bringing efficiency back: every submitted system we could size uses at least 42x, typically several hundred times, more parameters than the baselines it beats. 

---
# Can AI Do Formal Demography? Why Vaccinating the Oldest Saves Both the Most Lives and the Most Years of Life 

**Authors**: Joshua R. Goldstein, Kenneth W. Wachter  

**Link**: [PDF](https://arxiv.org/pdf/2611.03500)  

**Abstract**: Can AI contribute to formal demography? We report an experiment in which an AI model, Claude, was asked to explain and then simplify the proof of an earlier result: that vaccinating the oldest against COVID-19 saves both the most lives and the most years of life, because the years saved per person treated rise with age. The model proposed a new proof that is shorter, more intuitive, and somewhat more general than the original, built from a known life-table identity and the definition of convexity. It states the condition for the result as the convexity of the reciprocal of the hazard, which can be checked from the hazard alone and holds for most familiar parametric models, and it yields a lower bound on life expectancy whenever that condition is met. Prompted by a further question, the model also gave an exact form of the condition, showing how far convexity can be relaxed. The model traced these identities and related results in earlier literature. An edited record of the sessions is provided, and complete transcripts are available on request. The exercise suggests that AI can revisit established results, make them easier to understand, and prove valuable generalizations. 

---
# Online Label-Free Gradient Matching for Continual Language Model Training 

**Authors**: Sajjad Pakdamansavoji, Mohammad Reza Rezaei, Ilija Subasic, Pedro Garzon, Radu Paradovschi, Zhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.03496)  

**Abstract**: Language models in customer-facing products must be continually updated, since the data they serve shifts over time. As retraining on the full accumulated data is impractical, each update should instead run on a small set of examples carrying the training signal of all data seen so far. The strongest current methods build this set by gradient matching, which synthesizes examples whose training gradients match those of the full data. However, they require ground-truth labels, run once on a fixed snapshot rather than on a growing stream, and are slow on large data. We introduce \textbf{OLGA}, which lifts these limitations. It distills unlabeled data by drawing the gradient matching targets from the model's own next-token predictions. Across five text classification benchmarks with phi-1.5, it improves accuracy by 10 percentage points over the strongest selection baseline using as few as five examples. OLGA also distills online data streams by keeping a running rehearsal memory of synthetic examples, which is re-distilled together with each new batch as it arrives. In the online setting, it outperforms the strongest selection-based continual baseline by 6.6 stream-AUC percentage points. Finally, our method scales to large data by pairing distillation with selection in a hybrid pipeline, letting selection handle the mild stage of compression and distillation the extreme one. It cuts runtime by 30\% versus distillation alone at similar accuracy, and beats the best selection-only pipeline by up to 12.6 percentage points. 

---
# Playing Dumb: How a Chess Transformer Simulates Users Across Skill Levels 

**Authors**: Difan Jiao, George Eilender, Zhenwei Tang, Ashton Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2611.03439)  

**Abstract**: Generative models increasingly simulate users at different skill levels in education and AI evaluation. Yet it remains unclear whether lower-skill behavior reflects less internal knowledge or a different use of retained knowledge. This distinction is difficult to study in natural language, where proficiency and relevant knowledge are hard to measure. In this work, we study Maia-2, a chess transformer that predicts human moves conditioned on player skill, using quantitative skill ratings and deterministically labeled chess concepts. We hold board positions and model weights fixed and analyze positions where skill conditioning changes the prediction from a blunder to the best move. First, we measure encoded knowledge by training probes to predict the presence of 172 chess concepts from Maia-2's internal states. Probe accuracy remains nearly constant across skill conditions, and probes trained at one level transfer accurately to the others. To test whether stronger decisions can be recovered from this retained knowledge, we fine-tune only Maia-2's final output layer, raising best-move accuracy in the lowest skill band from 14.4% to 82.8%. Training this layer only on highest-skill representations retains most of the improvement at lower levels, showing that recovery does not require direct training on lower-skill states. Steering concept-relevant features learned by a sparse autoencoder also improves lower-skill decisions while leaving probe-measured concept encoding practically unchanged. These results support an account in which skill conditioning changes how retained concept knowledge influences move selection, showing that weaker behavior can understate the knowledge available within a model. 

---
# Optimize the Harness, Not the Model: Fitting a Minimal Agent to a Self-Hostable LLM for Investigative Analysis 

**Authors**: Shai Nahum Gefen, Vered Chassida, Dmitry Basin, Guy Feigenblat  

**Link**: [PDF](https://arxiv.org/pdf/2611.03434)  

**Abstract**: Government intelligence teams need LLM agents that can analyze large, messy evidence collections, but security and cost constraints often rule out hosted frontier models and broad, hard-to-audit tool surfaces. We report CLEO: Constrained Loop for Environment Optimization, a method for fitting a minimal agent harness to a fixed, self-hostable LLM. Given an expert-validated dataset, CLEO admits only the tools, guardrails, memory policies, and loop components that improve accuracy while remaining general and auditable. On our held-out dataset, the CLEO harness reaches an 83.8% pass rate, within 1.2 points of the strongest out-of-the-box Claude Code SDK baseline, while being two orders of magnitude smaller. 

---
# DeliverBench: Benchmarking AI Agents on Delivering Realistic Multi-Stage Professional Projects 

**Authors**: Yumou Liu, Mingju Chen, Zhoufutu Wen, Qianyu Yang, Zijie Wang, Yeqian Liu, Chenghao Yang, Tiliang Duan, Qiyu Cai, Yanglihong Xiao, Jiayun Dong, Jianlan Ma, Shenhao Miao, Yuhan Cao, Lu Xu, Que Shen, Tian Gao, Ge Zhang, Jianpeng Jiao, Xuanhe Zhou, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.03412)  

**Abstract**: LLM agents are judged by what they deliver, and in professional work what gets delivered is a project, not a task. A project pays off only if the whole package holds, and one weak part can limit the value of the rest. Yet existing benchmarks mostly test agents on one task at a time, leaving a gap between the capability we measure and the outcome that makes agents useful. To address this gap, we introduce DeliverBench, which tests whether agents can carry a project from brief to final delivery. Its 11 projects, spanning financial due diligence, legal contract drafting, and drug safety reporting, are reconstructed with 13 domain experts from real materials and chain 150 subtasks through 2-15 rounds of new materials and client feedback. Because no reference outputs are injected, agents inherit their own errors from one task to the next. Expert rubrics then score every checkpoint as well as the final package. Across seven model-harness systems, even the best system scores only 65.0 out of 100 on final delivery and passes 50.0% of subtasks at a 90-point threshold. The failures are often failures of carrying work forward: in one audit project, all seven systems addressed the review findings, but only one preserved traceability. Together, these results suggest that task-level capability does not simply compose into delivery, which also depends on keeping the work coherent as it evolves. 

---
# LAND: Learning an Adaptive Number of Draws for Bayesian Best-of-N Selection 

**Authors**: Koen Oostermeijer, Teryn Jones  

**Link**: [PDF](https://arxiv.org/pdf/2611.03388)  

**Abstract**: Best-of-N decoding improves generation quality by sampling multiple candidates and selecting the best, but it allocates equal compute to sources regardless of how much they benefit from additional sampling: the expected benefit is small for sources with low-variance score distributions and potentially much larger for those with high-variance distributions. To address this shortcoming, we propose LAND (Learning an Adaptive Number of Draws), which dynamically allocates samples according to their posterior-predictive marginal value. LAND learns a discrete empirical mixture of source-conditioned score distributions from calibration data, updates its belief about a source as completion scores are observed, and stops sampling when the expected improvement of another draw falls below a shared compute price. Across machine-translation experiments with multiple models and language pairs, LAND achieves better MetricX scores than fixed best-of-N at matched average compute. Conversely, at matched translation quality, LAND requires approximately 20-40% fewer candidate generations. Adaptive allocation therefore provides a simple way to retain the benefits of best-of-N while substantially improving its compute efficiency. 

---
# Can We Stop the Backtest Yet? Anytime-Valid Promotion Tests for Adaptive Financial Research Agents 

**Authors**: Eric Benhamou, Jean-Jacques Ohana, Beatrice Guez, Ethan Setrouk, Chamyl Saadi, Thomas Jacquot  

**Link**: [PDF](https://arxiv.org/pdf/2611.03384)  

**Abstract**: Financial research agents can generate and monitor trading rules faster than a human can document the resulting selection process. Candidates appear at data-dependent times, weak rules are abandoned, and promising ones are watched until they cross a reporting threshold. We introduce PROMOTE, a protocol that assigns every candidate predictable testing wealth at birth and accumulates a cost-aware e-process only from subsequent returns. If lifetime allocations sum to at most one, a weighted Ville argument bounds the probability of ever promoting a rule whose declared bounded net score has always had nonpositive conditional mean. The result permits dependent candidates, adaptive births, and indefinite monitoring. We audit the implementation in two settings. A retrospective pseudo-out-of-sample replay contains 24 panel--rule hypotheses, 3,000 shared-sign null paths, injected alternatives, and public ETF and industry data. Repeated unadjusted tests produce a family crossing in 94.5% of null paths, versus 0.0% for PROMOTE; at a 15-bp aligned ETF-component perturbation, PROMOTE has 70.1% target power and median crossing session 2,327. A separate exact mean-zero experiment lets an agent select candidate identities and birth times from past scores; the funded gate has 1.77% false-family crossings across 3,000 programs. The market replay produces zero promotions after costs. The certificate controls historical false promotion under the declared null; it is not a certificate that an earlier edge remains profitable at deployment. AI-use disclosure. Generative AI tools, including OpenAI Codex, assisted manuscript preparation. For this preprint version, Codex assisted with LaTeX editing, author and venue metadata, compilation, and layout checks. 

---
# Same Outputs, Different Scores: Auditing the Unchecked Graders of Multimodal LLM Benchmarks 

**Authors**: Ahmad Jarjis Hasan, Jonathan Lwowski  

**Link**: [PDF](https://arxiv.org/pdf/2611.03377)  

**Abstract**: Every multimodal-LLM multiple-choice benchmark ends with a scoring rule that converts a model's response, often free-form text rather than a clean option letter, into a right/wrong verdict, buried in its code. Unlike the dataset, this rule is chosen without justification and left unexamined, yet it sets the published number. We reproduce the official scorers of eight peer-reviewed benchmarks on five open vision-language models (VLMs) that fit a single 16 GB GPU, and introduce a blinded human-plus-LLM-judge adjudication (Cohen's kappa=0.956) that recovers what the model actually answered. The scoring rule alone moves accuracy by up to 22 points on identical outputs, yet exactly 0 wherever a model emits clean option letters, so it hides on strong-model leaderboards. It is the fallback and not the rambling that does the damage: the benchmark with the highest non-compliance we measure has an artifact of exactly zero, because its scorer abstains instead of guessing; a reasoning model with non-compliance 0.000 still swings 9.5 points with which regex match of its trace the parser keeps. No standard rule is reliable: snapping a non-answer to its closest option credits answers never given yet buries correct ones, abstention discards answers stated in words, and likelihood scoring follows answer-token priors, so a single grade conflates capability with luck rather than measuring either. We release all code, labels, and the protocol, and argue benchmarks should report a non-compliance rate and at least two rules, not one unexamined grade. 

---
# Mixed-Methods Exploration of Sociotechnical Factors to Improve Adoption of Crime-Reporting Apps in Urban Communities 

**Authors**: Olalekan David-Banjo, Chinazunwa Uwaoma, Paul Witman  

**Link**: [PDF](https://arxiv.org/pdf/2611.03376)  

**Abstract**: Crime-reporting mobile applications can enhance community safety and engagement in urban areas, yet their adoption remains low, partly because developers prioritize technological features over residents' social realities. This study used an exploratory sequential mixed-methods design to identify and validate the sociotechnical factors that drive adoption. Two focus groups and an analysis of 660 app reviews revealed that users' initial sense of usefulness fades, whereas sustained engagement depends on perceived value (PV), comprising six dimensions: safety, anonymity, recency, autonomy, camaraderie, and accessibility. A survey of 505 urban residents, analyzed with PLS-SEM, validated PV as a formative higher-order construct formed by four of these dimensions (safety, anonymity, autonomy, and camaraderie). PV was the strongest predictor of behavioral intention, ahead of effort expectancy, facilitating conditions, and performance expectancy, and behavioral intention strongly predicted actual use. The model explained 75% of the variance in behavioral intention and 51% in actual use. The findings offer design principles for community-centered crime-reporting technologies. 

---
# The SIMPLE Law for Energy Cost of LLM Inference 

**Authors**: Hiari Pizzini Cavagna, Andrea Proia, Daniele Cesarini, Zeynep Kiziltan, Andrea Bartolini  

**Link**: [PDF](https://arxiv.org/pdf/2611.03330)  

**Abstract**: Generative AI is billed per token, yet tokens are not equally expensive. Each token enters the KV-cache and raises the cost of those that follow, so energy grows non-linearly along a sequence whose tokens look identical. No analytical model states what a token costs, and the most accurate SOTA model is a fit, not a law: it absorbs architecture into fitted coefficients, requiring per-LLM fitting. We derive closed-form expressions of the computational and memory-access complexity of LLM inference in terms of architectural hyperparameters, covering MHA and GQA, and compare a family of analytical models of increasing granularity. From these, we obtain the SIMPLE Inference Model to Predict LLMs Energy (SIMPLE), a four-parameter law that separates prefill from decode, and decode into KV-cache cost and next-token computation. Over traces of 13 dense LLMs from six families on NVIDIA H100 GPUs, it attains R2 = 0.969 and 11.97% MAPE, 3.1-3.5x below the SOTA, without per-LLM calibration. The SIMPLE law explains as well as predicts: prefill energy grows quadratically with the hidden dimension, so larger LLMs suffer most from long inputs; and more aggressive GQA KV grouping reduces the per-token energy by up to 3x relative to standard MHA. Increasing the reasoning token budget pushes output length beyond the analytically derived energy minimum, driving super-linear energy growth; although accuracy grows with the budget, energy grows faster, so Intelligence per Joule peaks at an intermediate budget and falls by up to 3.14x across a 4.0x reasoning-budget range. Agentic serving raises energy per request by 13.0x relative to non-agentic ones, driven by the growth of both the prefill and the KV-cache terms with input length; while prefix caching removes most of the prefill cost (-93%), the KV-cache term is unaffected, so even a cached agentic request costs 4.8x more. 

---
# From FAIR to AI-FAIR Metadata-First Infrastructure for AI-Enabled Science 

**Authors**: Elizabeth G. Campolongo, Eric R. Sokol, Marty Downs, Gina Glenne, Gregory E. Maurer, Sydne Record, Dustin R. Rubenstein, Matthew J. Thompson, Christine Laney, Hilmar Lapp, Max Tarjan, Tanya Berger-Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2611.03327)  

**Abstract**: Artificial intelligence (AI) technologies and their capabilities have advanced rapidly over the past few years, leading to widespread interest and adoption in the biological and ecological sciences. However, the data infrastructure and computational resources that provide the foundation for AI workflows in ecology and biodiversity science remain fragmented. Moreover, the flow of value remains largely unidirectional: data move from collectors, through repositories and aggregators into models, and finally to research products or management actions. The derived value, such as corrected taxonomies, annotations, quality flags, and insights, rarely returns to source communities. We propose an extension of the original FAIR principles to address the needs of AI-ready infrastructure: the AI-FAIR metadata criteria, which place AI-enabled science in the context of a bidirectional Question-Data-Model triangle. This framework is designed to facilitate large-scale content-based analysis across diverse collections and ensure that the scientists generating the data benefit from the AI-enabled advances built on their data. Alongside these criteria, we introduce an agentic evaluation workflow that both rates existing datasets on the AI-FAIR and traditional FAIR criteria and provides actionable feedback where critical metadata is missing or incomplete. This modular dataset evaluation tool, paired with a domain-appropriate checklist, translates these principles into a reusable assessment of machine-learning readiness, AI-readiness for specified tasks, traceability, and ethical and CARE-aligned stewardship. Through this approach we aim to create a community-informed system to close the model data loop, so that AI also enriches data to further enable scientific discovery. 

---
# Top-$k$ Retrieval Leaves Memory Value Undetermined 

**Authors**: Arman Behnam  

**Link**: [PDF](https://arxiv.org/pdf/2611.03289)  

**Abstract**: A memory-augmented agent must decide which memories to keep, merge and retrieve, and its only evidence of their value is its retrieval log: the memories it retrieved in each episode and whether it succeeded. Existing methods assign every memory a value from this log, even when the log cannot separate one memory's contribution from another's. We treat the log as an experiment that nobody designed and ask which memory values it determines. Two counts answer this before any outcome is observed: how often each memory was retrieved, and how often each pair was retrieved together. From these counts, we prove that top-$k$ retrieval leaves every memory value undetermined, however many episodes are logged, and that a question with its own memories, asked once, determines at most one of their values. On fourteen standard memory benchmarks, no memory value is determined in any of 55 configurations. Both causes are choices about retrieval. We introduce LEDGER, which estimates each determined Shapley value in closed form with a confidence interval and reports every other value as undetermined. It merges memories with identical histories without losing a determined value, and it retrieves so that the next log determines more. On pooled HotpotQA, LEDGER raises the fraction of memories determined from 0.001 to 0.797, at a cost of 2.6 accuracy points on HotpotQA and with a gain of 2.7 on LoCoMo. 

---
# ScienceCUA: Resource-to-Skill Construction and Execution-Guided Skill Evolution for Scientific Computer Use Agents 

**Authors**: Daiqiang Li, Jingyi Peng, Rui Hu, Jia Li, Xuyang Liu, Guanren Qiao, Yujia Xiang, Qiuzhuang Sun, Zhiping Xu, Mingrui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.03238)  

**Abstract**: Computer-use agents for scientific software require application-specific expertise. Much of this expertise is documented in human-authored software manuals and web pages. However, turning these resources into executable, reusable skills remains challenging due to mismatches between documented and deployed software versions and differences in scope between documentation coverage and task requirements. We introduce ScienceCUA, a two-stage framework combining resource-to-skill construction with execution-guided skill evolution. First, ScienceCUA identifies tasks supported by the software from its documentation, infers operational trajectories, and generalizes them into reusable Skills. Command catalogs, syntax constraints, and shortcuts are organized separately as References for on-demand lookup. Then, ScienceCUA uses execution feedback to drive skill self-evolution by contrasting successful and failed trajectories for the same task, identifying deficiencies, revising existing skills or adding new ones, and updating the skill library only after repeated validation in real environments. To enable reliable evaluation without benchmark noise or task leakage, we introduce two improved benchmarks. ScienceBoard-Verified provides an audited and corrected basis for measuring task performance, while ScienceBoard-VE separates skill evolution from final testing through disjoint evolution and test tasks and further supports native-agent evaluation. Experiments show that the initial Skills and References improve average task success rate by up to 7.23 percentage points over the base model, while skill self-evolution increases the total gain over the baseline to as much as 17.12 percentage points. ScienceCUA turns software documentation into an evolving set of executable skills and queryable references, offering a practical way for general-purpose agents to use scientific software reliably and at scale. 

---
# Axion 1.0: Open Agent Foundation Models, Simplified 

**Authors**: Yuqi Jia, Yunjie Tian, Jiayu Zheng, Yijun Liang, Jingchen Sun, Mengyu Zhao, Di Fu  

**Link**: [PDF](https://arxiv.org/pdf/2611.03233)  

**Abstract**: We introduce Axion 1.0, a fully open agent foundation model that combines strong and broad agentic capabilities with a remarkably simple training recipe. Axion 1.0 is trained in only three stages: two lightweight supervised fine-tuning stages followed by reinforcement learning, without dedicated mid-training, direct preference optimization (DPO), or on-policy distillation (OPD). The first SFT stage develops strong self-contained reasoning through model-aware data construction, where we synthesize challenging problems and adversarially strengthen them based on the base model's own successful solution strategies. The second stage extends these capabilities to tool-dependent reasoning through a scalable pipeline for constructing diverse and automatically verifiable tasks grounded in real-world data sources and deterministic computation. Finally, reinforcement learning further strengthens long-horizon reasoning and interaction with external tools and environments. Despite this compact training recipe, Axion 1.0 achieves strong performance across a broad suite of agentic benchmarks and is competitive with leading commercial models on several of them. Equally importantly, Axion 1.0 is open beyond model weights: we release the complete recipe, including data construction, generation, filtering, verification, prompts, and training code, enabling the entire development process to be reproduced and extended. Axion 1.0 demonstrates that strong agent foundation models can be both open and simple, offering a practical and reproducible path toward advancing open agentic intelligence. 

---
# Auto-Research Agent on MSA Organization to Improve Protein-Protein Complex Structure Prediction 

**Authors**: Zilu Zhang, Jing Bao, Lifeng Qiao, Sheng Xu, Qiantai Feng, Siqi Sun, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.03183)  

**Abstract**: Protein-complex structure prediction with AlphaFold 3 depends on MSA content, depth, and cross-chain pairing. Current pipelines apply uniform recipes, typically maximizing depth and pairing by species, yet even whether cross-chain pairing helps AF3 remains contested. We show that the value of both depth and pairing is strongly target-dependent, so no single recipe is optimal across interfaces. We introduce an autonomous agent that treats a frozen AF3 as a black-box experimental instrument. For each target, the agent formulates biological hypotheses, analyzes and reorganizes the MSA, and iteratively refines sequence selection and pairing using predictor label-free signals alone. On 1{,}015 protein--protein interfaces from three benchmarks released after AF3's training cutoff, the agent improves overall interface DockQ over AF3's default pipeline and fixed MSA recipes on every benchmark, while using on average 60\% fewer MSA sequences than the default. Gains are largest for inter-species heteromers, where species-based pairing is unreliable: mean DockQ increases from 0.465 to 0.612 on FoldBench and from 0.482 to 0.562 on HD439. The improvements arise from both sequence construction and pairing policy, including deliberate this http URL results recast MSA design as a target-specific problem and show that an autonomous agent can adapt it at test time through biological reasoning and self-directed, confidence-guided experimentation. 

---
# CoworkOS:Scaling Agent Coworkers for Real-world Automation 

**Authors**: Jiajie Jin, Guanting Dong, Chengjun Pan, Xin Deng, Zhuohui Sheng, Guoqiang Wang, Tao Gui, Yutao Zhu, Yujia Chen, Shihong Yi, Binze Hu, Zhixiong Yang, Zhichao Sun, Rui Wang, Han Han, Mingyu Ma, Junhao Huang, Xiaotao Wang, Huicheng Jiang, Yuyang Hu, Xinyu Yang, Wanghan Xu, Yajie Yang, Zhecan Wang, Honglin Guo, Xiaoyu Xing, Xiaoqian Zhang, Yulun Zhu, Xiaoxi Li, Jiahang Lin, Shaofan Liu, Shihan Dou, Zhiheng Xi, Qi Zhang, Yu-Gang Jiang, See-Kiong Ng, Yang Zhang, Tat-Seng Chua, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2611.03178)  

**Abstract**: Rapid advances in cowork agents are shifting human-AI collaboration toward real-world automation, delivering software, designs, and other artifacts. However, such productivity work demands both high-quality delivery on complex productivity tasks and continued monitoring and decision-making over long horizons, a combination that remains an open challenge. We introduce CoworkOS, a general-purpose harness for long-horizon work that separates task strategy from execution management: the model decides actions, delegation, and acceptance, while the runtime maintains state and feedback. Its three-layer architecture enables direct execution or delegation to Subagents, synchronously or asynchronously, under stable responsibilities. Through a shared application layer, humans and agents inspect and revise the same evolving documents, presentations, CAD models, and Blender scenes, so human-in-the-loop participation can enter both through conversation and through direct modification of the work itself. To sustain continuity across agents and sessions, CoworkOS further introduces hierarchical workspaces and persistent semantic memory, so working files, key decisions, and long-running Job logs remain shareable, reusable, and resumable after interruption. Across 12 benchmarks and 6 categories of real-world applications, CoworkOS proves highly competitive with frontier agent applications such as Codex and DeepSeek Harness. As a sustained-work case study, CoworkOS uses GLM-5.3 as its backbone to conduct RSI-driven automated post-training on Qwen3.6-35B-A3B. Over five rounds, the model's average score across eight benchmarks rose from 33.19 to 42.95 (+9.76). Looking ahead, we believe CoworkOS opens a meaningful path toward general cowork-agent intelligence. 

---
# Toward Knowledge-Centric AI: Discovering, Internalizing, and Using Explicit Knowledge Across Agents and Generations 

**Authors**: Yisen Gao, Haoyu Huang, Zhongwei Xie, Yufei Li, Hong Ting Tsang, Boyi Zhang, Qiao Xiao, Tianshi Zheng, Jiaxin Bai, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.03157)  

**Abstract**: Knowledge is how understanding accumulates across people and generations: durable records that outlive their producers and that others can check, combine, teach, and correct. AI agents are now built and run by many independent users and organizations, and each agent is upgraded, replaced, or retired within months. Yet what an agent learns stays trapped in model parameters, a transient context window, or a system-specific memory store that is appended to and retrieved from rather than processed. We argue that the next generation of AI should be knowledge-centric, built around an explicit, continuously evolving knowledge layer independent of any single agent. In this loop, agents discover knowledge from their environments and trajectories; the layer internalizes it by turning fragmented, heterogeneous records and experience into composable, revisable understanding; and agents across systems and generations use the result and return new outcomes for revision. While discovery and use have advanced rapidly with large language model agents, internalization remains the bottleneck. We argue that internalization should become a persistent object of training and maintenance-centered on condition-preserving conceptualization of events and deductive, inductive, and abductive reasoning across noisy, heterogeneous representations (text, tables, graphs, rules, and programs). To power this layer, we call for knowledge foundation models that provide shared neural capacity for knowledge computation, and outline a research agenda showing how an explicit knowledge layer delivers reuse, cost efficiency, auditability, and continuity across agent generations. 

---
# CUA-Master: Scaling Computer-use Agents through Open-World Recursive Improvements 

**Authors**: Xinlei Yu, Gen Li, Ji Zhao, Zhangquan Chen, Congcong Wang, Shuai Dong, Zelin Peng, Yuqi Xu, Hongrui Wu, Kaiwen Tuo, Cheng Yang, Kaituo Feng, Yifan Wu, Jiayi Zhang, Dongming Wu, Wenqi Shao, Jiaqi Wang, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2611.03126)  

**Abstract**: Despite recent advances in CUAs, their development remains constrained by the limited diversity, controllability, and reliability of interaction data. Moreover, conventional optimization pipelines lack iterative feedback, while existing recursive improvement approaches often rely on self-generated experiences, limiting ability expansion and yielding diminishing returns. To address these challenges, we introduce CUA-Master, a nested double-loop recursive improvement framework. At the system level, an outer world-data-model flywheel continuously incorporates open-world corpus and leverages evolving model weaknesses to guide subsequent data construction and optimization iterations. At the data level, an inner flywheel couples open-world seeding and autocatalytic propagation. The former mines authentic workflows as seed tasks from instructional videos, and employs visual-semantic dual-path trajectory reconstruction; The latter transforms these seeds into reusable primitives for controllable synthesis of diverse, complex, and verifiable tasks. To enable efficient and reliable trajectory collection, we further introduce a feedback-guided snapshot-rollback retry mechanism in various real environments. Extensive experiments demonstrate that CUA-Master-27B achieves competitive performance across GUI-only and GUI-CLI hybrid benchmarks, scoring 80.2 on OSWorld Verified, 36.9 on OSWorld 2.0, 43.9 on WeaveBench, and 25.0 on Agents' Last Exam. Our framework also substantially improves the quality and efficiency of data synthesis, and exhibits continuous gains across recursive iterations. 

---
# ComplianceAtlas: A Taxonomy-Driven Agentic AI Framework for Automated AI Act Gap Analysis and Remediation 

**Authors**: Ilias Driouich, Hongliu Cao, Eoin Thomas, Yannick Devaux, Agnes Brodelle  

**Link**: [PDF](https://arxiv.org/pdf/2611.03121)  

**Abstract**: Assessing regulatory compliance for high-risk AI systems requires systematic identification of technical gaps such as missing controls, in- adequate processes, or shallow evaluation of practices. Compliance officers typically admin- ister structured questionnaires collecting evi- dence of current practices, where organizations respond with descriptions of their existing pro- cesses, technical capabilities, and architectural choices. However, translating these compli- ance questionnaire responses into typed, ac- tionable gap diagnoses remains unsystematic. In this paper, we present a literature-grounded taxonomy of technical compliance gaps cover- ing areas such as accuracy, robustness, logging, and human oversight, and embed it in GapTo- Compliance, a two-stage agentic framework coordinating two specialized agents: a Gap Extraction Agent that performs hierarchical rea- soning from compliance areas to specific gap types, and a Remediation Planning Agent that ranks remediation actions from a remediation catalog via multi-signal scoring combining se- mantic similarity, regulatory alignment, and effort estimation. We evaluate this framework against four automatic compliance gap detec- tion baselines across 9 AI systems and 6 AI families. Our approach achieves a gap-type F1 of 0.831 (vs. 0.533 for the strongest baseline) and remediation P@5 of 0.872 (vs. 0.635 for retrieval methods), demonstrating promising potential for automated compliance workflows. The structured agent coordination reveals a path toward human-in-the-loop compliance systems where automated gap extraction reduces expert burden while maintaining auditability, and the taxonomy-driven design enables extension to emerging regulatory frameworks beyond 

---
# From Local Skills to Long-Horizon Web Tasks: Progressive Skill Exploration for LLM Web Agents 

**Authors**: Xueyang Feng, Xiaohe Bo, Xu Chen, Quanyu Dai, Feng Wen  

**Link**: [PDF](https://arxiv.org/pdf/2611.03116)  

**Abstract**: Large language models (LLMs) are driving autonomous agents toward real-world environments, with the Web emerging as one of the most representative interaction scenarios. However, training Web agents with naive exploration remains challenging: successful trajectories are rare, and the sparse terminal rewards make it difficult for agents to obtain useful feedback from failed trajectories. Inspired by skill acquisition theory, we propose Skill-Composed Progressive Exploration (SCOPE), a short-to-long learning framework for long-horizon Web agents that decomposes complex task exploration into a gradual expansion process from local skills to complete task-solving capability. Specifically, SCOPE iteratively explores around local skills: Skill Discovery extracts reusable local skills and failure experiences from historical rollouts; Skill Orchestration composes discovered skills into high-level skill chains; and Skill Expansion leverages failure experiences aligned with the current skill chain to guide the agent toward exploring new subskills. Through this process, SCOPE progressively constructs task-completing skill chains across rollouts and internalizes them into the policy parameters through agentic on-policy distillation. Extensive experiments on WebArena, WebVoyager, and WebShop show that SCOPE consistently improves exploration dynamics and achieves 3.9% to 18.6% relative gains over the strongest baselines in each setting. 

---
# LUCA: A Synthetic World for Experimental Reasoning 

**Authors**: Anna Gogleva, Thomas Charman, Martin Balla, Karl Gemayel  

**Link**: [PDF](https://arxiv.org/pdf/2611.03100)  

**Abstract**: An experimental agent can succeed for the wrong reasons. Evaluations, however, often focus on the outcome, not the experiments, evidence, and inferences that produced it. We introduce LUCA, a configurable synthetic world for evaluating this process under fully known ground truth. Agents learn unknown regulatory systems through budgeted experimentation, then predict unseen interventions, diagnose hidden faults, and attempt rescues. This setup allows us to evaluate experiment selection, prediction, diagnosis, and intervention separately.
Across seven frontier LLMs and three baselines, these capabilities dissociate sharply. Agents frequently rescue malfunctioning systems while diagnosing the wrong fault; diagnosis varies 7x across models while prediction varies only 1.4x. Models also adopt surprisingly different experimental strategies, and greater exploration does not reliably yield better performance. In one failure mode, agents often perform the experiment needed to distinguish competing causes, but still fail to use the resulting evidence. LUCA turns these hidden failures into measurable behaviours, providing a controlled environment to study where experimental agents succeed and fail. Code: this https URL. 

---
# A Fine-Grained Hierarchy and Benchmark for State Tracking in Language Models 

**Authors**: Jackson Petty, William Merrill, Ashish Sabharwal  

**Link**: [PDF](https://arxiv.org/pdf/2611.03094)  

**Abstract**: Theoretical limitations of transformers on state tracking---the ability to track information about entities or variables over long contexts---have motivated new recurrent language modeling (LM) architectures such as Gated DeltaNet (GDN). These architectures have, in particular, been benchmarked against the $S_5$ word problem, a theoretically hard state-tracking task that captures abilities such as code evaluation under arbitrary variable assignment. It, however, remains open whether other, simpler forms of state tracking might both be more relevant in practice and already explain observed differences between these architectures. We propose evaluating state tracking using a hierarchy of word problems with four increasing levels of difficulty: r-trivial, corresponding to immutable assignment to variables in programming; aperiodic, where assignments are mutable; soluble, where assignments are self-referential; and general finite state tracking. Theoretically, we show that the class of problems transformers are thought to length-generalize on does not even contain all aperiodic state tracking. On the other hand, we prove that GDN with positive eigenvalues expresses all aperiodic problems, and the regular languages recognized by certain transformers are precisely all soluble ones, thus providing a fine-grained picture of state tracking capabilities of various architectures. Taking these ideas to practice, we construct StateBench, a 4-tier long-horizon state-tracking benchmark. Our systematic direct training experiments across architectures largely align with theoretical expectation, albeit with interesting observations at very large scale length-generalization. Additionally, we also evaluate pretrained models---both open hybrid and closed frontier ones---on StateBench, finding that none of them robustly solves even soluble state tracking, suggesting a direction for future improvement. 

---
# Avatar Intelligence: Amplifying Human Agency Through Persistent Personal Agents 

**Authors**: Zheng Wu, Xinbei Ma, Yao Yao, Zhaoyuan Huang, Haowen Hu, Yijie Lu, Taolei Wang, Yifei Yang, Wuyue Xiong, Pengzhou Cheng, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.03092)  

**Abstract**: Personal AI agents are increasingly capable of handling long-horizon tasks across applications, yet users often need to invest substantial effort in specifying goals, resolving ambiguity, and verifying outcomes. We argue that progress should be measured by how effectively agents amplify human agency: enabling people to pursue their goals with less routine supervision while retaining control over consequential decisions. We advocate Avatar Intelligence, a paradigm of persistent personal agents that represent users' goals, values, and constraints through sustained decision delegation across contexts and time. We describe a maturity roadmap from human-directed execution to autonomous operation within delegated domains, and introduce the Agentic Oversight Index (AOI) as a framework for evaluating delivered value against agent-side operational cost and human oversight burden. We identify three core research challenges: identity-continuous personalization that carries user preferences across tasks, user-situated evolution through Rehearsal Engineering, and safety mechanisms that combine technical control with accountability and recourse. These directions define a research agenda for reliable decision delegation to personal agents across tasks and over time. 

---
# Towards a Trustworthy and Governable Internet of Agentics: An Architectural Perspective on Protocols, Systems, and Services 

**Authors**: Haoyi Xiong, Yi Liu, Jiang Bian, Peng Diao, Bin Zhu, Tingting Hu, Yi Wang, Sumi Helal  

**Link**: [PDF](https://arxiv.org/pdf/2611.03066)  

**Abstract**: Large language model applications are increasingly being organized as tool-using and multi-agent workflows rather than as isolated chat interfaces. This perspective paper examines the architectural problem that follows from that shift: agents may reason, invoke tools, and delegate work, but dependable deployment still requires interoperability across frameworks, runtimes, data boundaries, and organizational control domains. We study this problem through the lens of the \emph{Internet of Agentics} (IoA), an emerging service-computing view in which heterogeneous agents are treated as governable service participants rather than only as stand-alone model endpoints. The paper organizes the landscape as a three-layer progression: from today's fragmented agent ecosystem, to interoperable agents, and ultimately to trustworthy and governable autonomy. First, it examines the protocol stack centered on the Agentic AI Foundation, including MCP for agent--tool interaction; A2A for agent--agent coordination; this http URL and Skills-style procedural manifests; and discovery, transport, and trust mechanisms such as AGNTCY and SLIM. Second, it reviews the systems abstractions that make agents deployable as service components, including logical agent architectures, orchestration patterns, gateways, registries, runtime control loops, and hardware-aware inference infrastructure. Third, it situates these developments within services computing, arguing that agentic systems reinterpret classical concerns from SOA and microservices---discovery, composition, governance, pricing, and trust---at the level of cognitive capabilities rather than deterministic functions. 

---
# Networked AI: Architecture, Theory, and Applications 

**Authors**: Liang Song, Jiayue Jin, Peng Sun, Di Li, Yang Liu, Athanassios Skodras, Dimitris Ampeliotis, Yong Yuan, Amit Dvir, Jiangchuan Liu, Victor C. M. Leung, Qi Bi  

**Link**: [PDF](https://arxiv.org/pdf/2611.03063)  

**Abstract**: Recent advances in Artificial Intelligence (AI), particularly large-scale deep learning, have been driven by monolithic or offline training on massive curated datasets. While successful, these approaches face structural limitations: escalating data and computational demands, limited online adaptability, and a lack of system-level intelligence in open, non-stationary environments. This paper presents Networked AI - a networked intelligent system composed of interacting, heterogeneous, and continuously evolving AI agents. We establish a hierarchical layered architecture comprising an Agent Layer for heterogeneous capability construction, an Interaction Layer for signal-processing-guided online data extraction, and a Regulation Layer for stability-constrained collective evolution. Foundations in data science, signal processing, and system control are hereby synthesized into a unified theoretical framework of Networked AI, providing superior data efficiency, intelligence efficiency, interpretability, and scalability. Theoretical and experimental analysis demonstrates that Networked AI can approach near-linear scaling in resource demand, in contrast to the exponential increase in standalone end-to-end systems. It provides not only online adaptation to out-of-distribution environmental novelties, but also the autonomous evolution towards application-specific intelligence. Application frontiers are discussed for IoT spaces, embodied, spatial, and industrial intelligences, as well as the cellular infrastructure and beyond. Key open challenges are identified towards realizing locally autonomous, collectively intelligent, continuously evolving, globally regulated Networked AI. 

---
# PRIME: Predictive Routing of Interventions by Modeling Effects for Scientific Ideation 

**Authors**: Fengxian Ji, Xie Yu, Zhexuan Cui, Yuke Li, Zirui Song, Fan Zhang, Qian Jiang, Jingpu Yang, Xiuying Chen, Preslav Nakov, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2611.03055)  

**Abstract**: Feedback-driven iteration is widely used to improve LLM-generated scientific ideas, but existing systems either leave revision decisions implicit in generation, align revisions with feedback-localized weaknesses, or restrict action selection to system-specific operations. This is problematic because problems, hypotheses, and solutions are interdependent: the location where a weakness manifests does not itself determine the revision scope that will produce a better successor. We term this phenomenon diagnosis-intervention mismatch. We introduce Predictive Routing of Interventions by Modeling Effects (PRIME), a plug-and-play feedback-to-revision decision layer for existing scientific ideators. PRIME operates in three stages: (1) Structural Attribution interprets feedback over the problem-hypothesis-solution structure without fixing a revision target; (2) Counterfactual Routing constructs alternative scope-conditioned interventions and compares their projected successor utility and risks through lightweight rollouts; and (3) Intervention Execution compiles the selected intervention into an ordered generator-facing plan with semantic actions, preservation constraints, and dependency requirements for successor generation. We validate diagnosis intervention mismatch on 112 human-annotated idea-feedback pairs. Across four benchmarks, two ideators, and fixed- and open-problem settings, PRIME improves successor quality and routing reliability over strong baselines; removing counterfactual routing reduces quality from 3.463 to 1.960 and increases route regret from 0.426 to 1.929, confirming cross-scope effect modeling as the primary source of its gains. 

---
# BLPO: Beyond-Local Policy Optimization for Long-Horizon LLM Agents 

**Authors**: Xuanren Chen, Chongyang Tao, Wenpeng Hu, Yong Wang, Bo Deng, Xinlei Zheng, Shuai Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.03005)  

**Abstract**: Long-horizon tasks require large language model (LLM) agents to coordinate interdependent actions, a capability central to automating complex workflows. Stepwise group-based reinforcement learning provides finer credit than trajectory-level supervision, but reliable comparisons must account for differences in interaction history. Enforcing this consistency within the limited rollouts of each task can leave insufficient support for estimating action preferences. We propose Beyond-Local Policy Optimization (BLPO), which estimates task-progress advantages from experience shared across tasks and training updates. We summarize trajectory prefixes into task-progress states and abstract actions by their function, allowing corresponding decisions from different tasks to contribute to shared estimates. To control bias and noise from heterogeneous tasks, we accumulate context-relative returns in a global memory with balanced aggregation and reliability weighting. These global advantages complement local comparisons over normalized actions and trajectory-level credit without a learned critic or additional rollouts. Experiments on ALFWorld and WebShop with 1.5B and 7B policies show improved task success over the evaluated baselines under matched training budgets. Further analyses show faster learning and shorter trajectories with little computational overhead. 

---
# CAREFUL-Bench: Effect-Aware, Judge-Free Evaluation of Multi-Turn LLM Function Calling 

**Authors**: Alexander Kovalenko, Artem Redchych  

**Link**: [PDF](https://arxiv.org/pdf/2611.02978)  

**Abstract**: The multi-turn checker of the Berkeley Function Calling Leaderboard (BFCL) evaluates final state correctness but ignores the effects of intermediate calls: extra calls are free, function documentation does not describe side effects, and the scripted user never responds to proposals. Consequently, models asking for confirmation before destructive actions are penalized; 260 of 734 reference turns in the base category perform destructive actions without confirmation. We make BFCL effect-aware without replacing it. We infer effect declarations for all 128 functions by executing them in BFCL simulators and comparing state changes, with destructive writes identified by three annotators. A judge-free replay scorer reproduces official results across all 800 reference trajectories while reporting unrequested destructive calls, duplicate writes, redundant reads, surplus steps, and their token costs. We also introduce a confirmation-aware protocol with a deterministic simulated user. Across 11 open models, effect-aware scoring preserves the official ranking but reveals substantial differences in execution quality: the leading model loses 14.0 points due to unrequested destructive calls, while surplus steps consume up to 28% of input tokens. Adding effect hints to function documentation has no impact on eight evaluated models. Allowing the simulated user to respond recovers up to four points for models that seek confirmation. However, the best model completes only six of 576 conversations requiring destructive actions with explicit confirmation. These results expose a key limitation of current function-calling benchmarks: they cannot distinguish between asking for clarification and obtaining permission before potentially destructive actions. 

---
# Action Verification through Conjecture and Refutation for Language Model Agents 

**Authors**: Minhyeon Oh, Chiyeong Heo, Geonwoo Noh, Hoyoung Kim, Jangwon Son, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2611.02950)  

**Abstract**: Language model agents increasingly act on users' behalf by manipulating external environments through tools, often from high-level instructions that leave critical task- and environment-specific context unstated. To prevent resulting misaligned actions, recent work verifies proposed actions before execution, largely by establishing whether the conditions supporting them hold. We argue that this confirmation-oriented view leaves a fundamental blind spot: a wrong action often passes every check that asks whether it is right. This motivates a different question for verification: rather than asking what evidence supports an action, can the agent find evidence that refutes it? We pursue this direction with BlackSwan, a refutation-oriented, expect-suspect-test procedure that predicts the intended effect of each state-mutating action, formulates specific and empirically testable hypotheses for how it could be wrong, and tests them using existing observations or additional non-mutating actions, executing the action only if none survives testing. On challenging AppWorld tasks, BlackSwan improves success rate over the strongest baseline by 14.9 and 25.2%p with and without the backbone model's thinking mode, respectively. Even without thinking or in-context demonstrations, it outperforms stronger baselines equipped with them. The gains also transfer beyond AppWorld, with BlackSwan improving WebShop performance by 12.7%p. More broadly, our results suggest that language model agents have an underused capacity to recognize and self-correct misaligned actions, which can be systematically elicited through active conjecture and refutation before execution. 

---
# Why Tabular Foundation Models Excel on Relational Data They Have Never Seen 

**Authors**: Linjie Xu, Yiqun Hu, Jixiang He, David Wipf  

**Link**: [PDF](https://arxiv.org/pdf/2611.02936)  

**Abstract**: Given a relational database (RDB) storing heterogeneous tabular information, how can we predict missing (or future) values in some target column of interest? As candidate targets are numerous in enterprise settings, it is preferable to avoid learning a new model from scratch each time there is a new prediction task. Frozen foundation models (FMs) based on in-context learning (ICL) and RDB-specific pre-training provide a viable solution, but the ideal design still remains largely unresolved. Perhaps unexpectedly, preliminary evidence has recently shown that existing single-table FMs (aka TFMs), when combined with suitable parameter-free relational encoders, can be quite competitive in practice despite zero exposure to RDB pre-training or relational data of any kind. In this work we first empirically confirm this phenomenon with a more diverse array of experiments. We then turn to our main contribution, namely, surfacing multiple complementary explanations for why TFMs behave so effectively. These explanations ultimately hinge on innate synergies whereby parameter-free encoders map relational effects to fixed-length embeddings that loosely mimic distributions for which TFMs were originally designed; in so doing, desirable single-table properties transfer to relational settings (e.g., convergent posterior predictive distributions). Collectively, our analysis forges new pathways for improving RDB FMs, particularly when we prioritize interpretability and inevitable integration within agentic systems. 

---
# User Attributes and Political Ideology Shape LLM Receptiveness in Online Political Debates 

**Authors**: Akram Elbouanani, Yousra Taleb, Gianluca Demartini, Adrian Popescu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02908)  

**Abstract**: Online political discussions increasingly involve AI-mediated participation, as users invoke LLMs to assess contested claims, arbitrate disputes, and respond to opposing viewpoints. As these systems become participants in online exchanges, they may alter which speakers and argumentative strategies receive validation within the discussion.
We study this interactional dimension by reconstructing 5,376 this http URL debates as LLM conversations and running nearly 150,000 controlled multi-turn conversations that vary user political orientation, expertise, rhetorical strategy, and defended position, including conversations with competing users.
Across naturalistic and controlled debates, we find that LLMs are systematically more receptive to left-leaning than right-leaning users. The asymmetry persists when users with different political orientations defend the same position and when opposing users appear in the same conversation, indicating that it is not fully explained by stance agreement or simple accommodation to the current speaker. Repeated interaction increases receptiveness without eliminating this ordering; expertise also matters, while rhetorical differences are smaller. The pattern appears across multiple models, but is not fixed: fine-tuning on different political corpora can largely erase or reverse it. These results show that LLMs can respond systematically differently to participants in the same political debate, making such interactional asymmetries important for understanding their emerging role in online public discourse. 

---
# What makes a theorem interesting? Operationalizing interestingness for open-ended exploration 

**Authors**: Silvère Gangloff  

**Link**: [PDF](https://arxiv.org/pdf/2611.02905)  

**Abstract**: An agent that explores, whether it generates conjectures or research ideas, must decide which of them are worth keeping. This judgment of interestingness is usually delegated to a hand-picked proxy or to a language model, neither of which is known to correspond to what humans actually value. I study it for mathematical theorems, by observing which results mathematicians single out in formal libraries and in papers. I model each way of singling out results as an evaluator that weights features of a result's statement, its proof and dependencies, and split its weights into a shared and a subjective part. In formalized mathematics, ex-ante features predict these selections as well as a large language-model judge, and the two complement each other. Interestingness is also largely shared. One set of weights, carried mostly by structural features such as the length of the statement and of the proof and the subfields the proof draws on, accounts for most of the signal and transfers in part across formal libraries. What remains subjective depends on practices and individual authors more than on subfields. The judge's advantage lies in its knowledge of the canon, and neither the judge nor the features rely on memory of the papers. The approach, which infers what a community finds interesting from what it singles out, can be applied to other sciences. 

---
# Bridging Low-Level Image Fusion and High-Level Semantic Understanding 

**Authors**: Guangyuan Dong, Lingdong Shen, Xudong Zhang, Haodong Jing, Haoyu Wang, Zihao Li, Zonghao Ying, Ziyu Song, Shenghao Liu, Jiachen Luo, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02896)  

**Abstract**: Remote sensing image fusion and semantic interpretation are usually optimized separately, so reconstruction quality need not preserve downstream retrieval utility. We propose LANCE, which uses structured language as an explicit, indexable semantic state. A frozen vision-language model maps the input to a closed-vocabulary text record; a language-conditioned generator fuses the imagery; and a dense verifier with a differentiable continuous summarizer re-describes the output. Text-text consistency and anti-collapse contrastive text-mask losses with negative queries close the loop without dense annotations or VLM fine-tuning. A variational EM formulation yields a per-step contraction result under regularity assumptions, distinguishing semantic faithfulness from verifier overfitting. Across seven test sets on GF2, QuickBird, WorldView-III, and WorldView-II, LANCE improves reduced-scale reference metrics by up to 11.7% (median 2.1%) over the strongest learning baseline. Independent evaluation using a held-out VLM and a supervised aerial detector yields gains of +6.4 mAP and +0.072 open-vocabulary F1 over the closest competitor. Text perturbations, cross-verifier transfer, and pseudo-RGB ablations examine the mechanisms underlying these gains. The framework connects image fusion to verifiable semantic representations for multimodal retrieval and knowledge management. 

---
# JABER: Sequential Quality Control for Human-AI Data Annotation under Shared Bias 

**Authors**: Fengxu Yang, Minghui Liwang, Yang Lu, Bo Han, Xianbin Wang, Seyyedali Hosseinalipour  

**Link**: [PDF](https://arxiv.org/pdf/2611.02892)  

**Abstract**: High-quality annotations are essential for training and evaluating artificial intelligence (AI) systems. Quality control (QC) often acquires additional human or AI judgments before accepting a label. Nevertheless, shared biases can cause correlated judgments, leading conventional methods to mistake repeated agreement for independent evidence of correctness. To address this challenge, we propose JABER (Joint Alias-Preserving Belief for Evidence Routing), a bias-aware sequential QC framework that maintains a joint Bayesian posterior over label correctness and shared biases to guide evidence acquisition and termination decisions. Building on this posterior, we develop a finite-horizon algorithm that selects evidence-acquisition and termination actions to minimize expected QC loss under resource constraints. Our theoretical analysis shows that conditional-independence updating can become arbitrarily confident in an incorrect label under persistent shared bias. We further identify when repeated inspections cannot distinguish a correct label from an incorrect label supported by shared bias, when obtaining a new label can resolve this ambiguity, and why modeling the dependence between label correctness and shared bias can be necessary for QC decisions. Our experiments show that JABER reduces expected QC loss by an average of 4.65% compared with a method maintaining separate label correctness and bias probabilities when shared bias makes repeated inspections unreliable. Further, JABER's finite-horizon planning, which evaluates outcomes of successive QC actions, reduces expected QC loss by an average of 10.05% compared with a one-step lookahead method. In real-API experiments on FEVER, JABER achieves zero labeling errors while reducing QC costs by 2.67% and 16.99% compared with marginal-based belief modeling and one-step lookahead methods, respectively. 

---
# Integrating Domain Adaptation and Retrieval-Augmented Generation for Saline-Alkali Land Reclamation Planning 

**Authors**: Zuguang He, Congcong Zheng, Wanyue Chen, Wenya Zhou, Zaijin Yang, Long Long, Yucheng Zhang, Shuqin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.02883)  

**Abstract**: Reclamation of saline-alkali land requires management plans that account for soil and water conditions, treatment evidence, and financial constraints. This study develops Wenyao, an agricultural decision-support system integrating domain adaptation with retrieval-augmented generation. A bilingual corpus of 8,664 research papers was processed using a 110-indicator schema to construct SA-Bench, comprising 4,860 paper-level planning instances. A QwQ-32B model was adapted through low-rank supervised fine-tuning and combined with case retrieval for budget-conditioned planning. Evidence extraction achieved an F1-score of 0.83 on 300 expert-annotated papers. Without retrieval, Wenyao achieved 95.83% accuracy on 936 source-disjoint multiple-choice questions, compared with 87.93% for the strongest general-purpose baseline; removing explanatory supervision reduced accuracy to 89.42%. Matched-evidence comparisons across 69 region-budget cases supported complementary contributions from adaptation and retrieval. In a blind assessment by 31 experts, Wenyao with retrieval obtained the highest mean scores for objectives and projected outcomes, whereas GPT-5 scored highest for cost estimates. These findings support literature-informed reclamation planning with professional review, while ongoing field monitoring will assess soil and crop responses. 

---
# Value-Conditioned Proposal Acquisition for Test-Time Reasoning 

**Authors**: Zeyan Li, Jianfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02852)  

**Abstract**: Sampling and selection spend test-time compute inside a candidate pool, but they cannot recover a correct answer that never enters it. We introduce ValueFormation, a test-time policy that decides after each primary-model draw whether to stop, continue sampling, or acquire one proposal from a heterogeneous reasoning specialist. The three actions are compared on one cost-regularized value scale learned from vote-state features; the calibration is frozen across tasks and primary models, and acquired proposals are accepted only through a frozen posterior comparison. On held-out MATH-500, one Llama-fitted calibration improves over the strongest of eight identically evaluated baselines by more than fifteen points for each of three 8B primaries and transfers unchanged to Qwen3-14B. Removing acquisition lowers Llama MATH accuracy by 20.43 points, while same-trace and same-access replays attribute the gain to state-conditioned acquisition rather than specialist access alone. Where primary coverage is high, as on GSM8K, acquisition is rare and gains mainly come from cheaper allocation; on MBPP, a fixed code route attains the top accuracy at the lowest interaction cost. Varying the token price monotonically reduces calls without retraining. 

---
# TLAPS-Bench: Formally Proving Correctness of Real-World System Specifications with TLA+ Proof System 

**Authors**: Qian Cheng, Ruize Tang, Saad Mohammad Rafid Pial, Zhizhen Cathy Cai, Yiming Su, Jackson Clark, Munim Thahmid, Mahdiya Rahman Sukanya, Chris Newcombe, Igor Konnov, Stephan Merz, Xudong Sun, Murat Demirbas, Yu Huang, Markus Alexander Kuppe, Tianyin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02849)  

**Abstract**: TLAPS-Bench is a benchmark and execution harness for constructing machine-checked proofs for real-world system specifications in TLA+ with agentic AI. Proving correctness of complex system specifications requires discovering inductive invariants, reasoning about intricate state transitions, and decomposing proofs into tractable obligations. TLAPS-Bench provides a proof harness that supports iterative proof development atop the TLA+ Proof System (TLAPS), where agents are equipped with proof tools and libraries, with safeguards against reward hacking. Using the harness, we have employed AI agents to prove desired properties for important system specifications that were not proved before, spanning cache coherence, Byzantine consensus, and write-ahead logs. TLAPS-Bench contains 265 proof targets (each encodes a property to prove) from 95 TLA+ specifications of real-world systems. We evaluated frontier AI agents, represented by Codex with GPT models, on a subset of 70 proof targets across 12 specifications. The highest-scoring model, GPT-6.1 Sol (max), proved 31 (44.3%) proof targets. The results show challenges and opportunities in end-to-end proof development. 

---
# SpecOPD: Branch-Parallel On-Policy Distillation for Speculative Decoding 

**Authors**: Tianyu Liu, Jun Zhang, Wenjie Huang, Yuhao Shen, Tianqi Xu, Xinyue Hao, Hongyaoxing Gu, Xuyang Liu, Baolin Zhang, Yating Zhu, Shuang Ge, Jun Dai, Lei Chen, Yue Li, Mingchen Wan  

**Link**: [PDF](https://arxiv.org/pdf/2611.02846)  

**Abstract**: Speculative decoding (SD) accelerates large language model inference by reducing the sequential cost of token-by-token generation. Advanced SD methods reduce proposal cost with parallel drafters that predict entire blocks in one forward pass, but overall speedup still depends on acceptance length. On-policy distillation (OPD) can push well-trained drafters beyond acceptance-length plateaus, yet conventional pipelines collect supervision through sequential draft-and-verify rollouts. Our key insight is that on-policy supervision need not inherit these dependencies: a draft block can be supervised using only its reference prefix and draft-sampled history. We propose SpecOPD, a branch-parallel OPD framework that parallelizes draft sampling and target supervision across independent branches drawn from the evolving drafter policy at reference anchors. We pack these branches into proposal trees to share target-prefix computation, using tree masks for full attention, branch-consistent indexer groups for sparse attention, and indexed state replay for linear attention to preserve each branch's history. We train the drafter with an E2E-TV surrogate that directly targets acceptance length. Our study spans 8B full-attention (GQA), 27B full-linear (GQA--GDN), and 125B sparse-linear (QSA--GDN) targets, three parallel draft architectures (DFlash, DSpark, and DFlash V2), and four task domains (dialogue, coding, mathematics, and agentic). SpecOPD achieves up to 41.45% higher mean acceptance length, $16.09\times$ supervised-token throughput over sequential-rollout OPD under native configurations, and up to $3.56\times$ generation speedup over autoregressive decoding. 

---
# CARE: Grounded Recursive Self-Improvement for Production Customer-Service Agents 

**Authors**: Yanbei Jiang, Chao Lei, Rena Gao, Xiaochen Zhao, Yuchen Ge, Haohang Liu, Runze Nie, Qitian Chen, Qi An, Haolin Song, Yufei Sun, Yihao Ding, Jey Han Lau, Krista A. Ehinger, Xingjun Ma, Weijia Chen, Ruiyuan Wu, Zhen Luo, Ke Xing, Yongjun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.02839)  

**Abstract**: Recursive self-improvement (RSI) in production agents involves transforming accumulated experience into reliable model updates while adapting to evolving business policies. We present CARE (Customer-service Agent Retrospective and Evolve), a grounded RSI framework that operationalizes this process in customer service. Its goal is to replace a frontier proprietary incumbent with an in-house mid-sized model that continues learning after deployment. After a one-time cold start on governed production traces, the current model serves production traffic alongside the incumbent, and its own failures drive the next update. A 24/7 inspection process establishes failures through evidence-blinded, independently reviewed judgments. Attribution separates model errors from defects in Skills, tools, or the harness, routing component failures to repair. Model-attributable trajectories are truncated at the first erroneous decision and paired with verified repairs. On-policy self-distillation (OPSD) then updates the model using the same model as teacher, with the diagnosis and repair supplied as privileged information unavailable to the student. To support continued evolution as business requirements change, training mixes failures from different Skill/SOP versions, each judged against the policy in force when it occurred. Candidates run an executable simulation test compiled from the latest SOP and then frozen, followed by offline inspection of the resulting traces. Promotion requires a combined simulation-and-inspection score at least at parity with the deployed version, confirmation through an online comparison with the incumbent, and human approval. 

---
# Think with Skills: Pseudocode Planning Before Execution for Advanced Reasoners 

**Authors**: Boshen Zhou, Boxuan Yue, Xiang Li, Shanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02830)  

**Abstract**: Language model agents often struggle with complex tasks even when equipped with expert-written skills. Such skills encode declarative knowledge, whereas successful execution requires procedural knowledge tailored to the task at hand. Natural language skills may leave conditions and dependencies implicit, and tasks involving multiple skills further require agents to coordinate intermediate results and execution order. We introduce Think with Skills, a method that trains reasoning models to turn relevant skill descriptions into task-specific pseudocode plans before execution. The pseudocode explicitly represents operations, conditions, intermediate results, and dependencies, providing a structured bridge from skill descriptions to agent actions. We construct training data that pairs tasks and skills with pseudocode plans and execution trajectories, use supervised fine tuning to initialize this behavior, and further optimize end-to-end task performance with Group Relative Policy Optimization. On SkillsBench, Think with Skills raises task success from 35.0% to 62.0%. These results show that proceduralizing expert knowledge before execution substantially improves skill use. 

---
# RxnMiner: A Multimodal Agent for Structured Reaction Data Extraction from Chemistry Figures 

**Authors**: Jinbo Hu, Songren Zhang, Chuanxun Fu, Jinrui Huang, Zhen Hao Wong, Dongxu Cui, Haoran Fan, Muhao Ge, Zhen Jian, Jian Zhou, Chengxiang Luo, Han Lyu, Haoyi Tao, Chaozheng Huang, Hanzheng Li, ZiChen Wang, Hengtian Ma, Jinyang Dai, Nan Wang, Yanhui Hong, Xi Fang, Guolin Ke  

**Link**: [PDF](https://arxiv.org/pdf/2611.02826)  

**Abstract**: Reaction data underpin machine learning for reaction prediction, condition recommendation, and retrosynthetic planning, yet extracting these data from literature figures still requires substantial manual effort. Many figures encode multiple experiments through shared templates, substituent tables, and product grids. Recovering concrete reaction records therefore requires molecular recognition, association of evidence with individual experimental entries, and reconstruction of indirectly specified participants. We present RxnMiner, a multimodal agent system that combines an expert-informed extraction procedure with a chemistry toolbox to coordinate these operations. The agent reads molecular structures, resolves shared and entry-specific evidence, and instantiates participants from templates to assemble source-linked reaction records. We also introduce ChemRxn-Bench, a benchmark of 603 reaction figures with 8,547 manually annotated and reviewed reaction records, evaluated under a common protocol for exact molecular identity and reactant-product pairing. On this benchmark, with Qwen3.8-Flash-Next as the backbone, RxnMiner raises stereochemistry-aware reaction macro-F1 from 21.0% under direct prediction to 71.0%, compared with 25.0% for the strongest existing extraction system. Across five backbones, it also outperforms the same base agent without the extraction procedure and chemistry toolbox by 9.4 to 25.5 points. Applied without human intervention to 10,107 literature figures, RxnMiner produces a silver corpus of 88,202 source-linked reaction records. 

---
# Predictive Safety Guardrails Do Not Always Make LLM Agents Safer 

**Authors**: Weixiang Zhao, Zhihao Liu, Guancheng Li, Yanyan Zhao, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02815)  

**Abstract**: Large language model agents increasingly take consequential actions in digital and physical environments, where safety risks may emerge only after multiple interaction steps. This has motivated world-model-driven predictive safety guardrails, which anticipate future consequences before execution. However, existing evaluations largely focus on end-to-end safety outcomes, leaving unclear whether safety gains arise from reliable prediction and faithful use of predicted risks. In this work, we systematically diagnose the prediction-to-action pipeline along three dimensions: how world-model predictions are integrated into decision making, whether the predictions themselves are reliable, and whether their content is faithfully used. Across seven policy models and two safety environments, we find that stronger prediction involvement does not yield monotonic safety gains and can reduce task completion while substantially increasing inference cost. We further find that world-model predictions remain insufficiently reliable, especially over longer horizons, where safety prediction degrades substantially relative to near-term forecasting. Finally, we find that safety gains can persist even under corrupted predictions, while locally verified predictions still leave subsequent safety violations, showing that predictive content is not faithfully incorporated into the decision process. Together, these findings expose a fundamental prediction-to-action gap in current predictive guardrails: producing more or seemingly accurate foresight is insufficient unless that foresight is both reliable and faithfully translated into safer decisions. 

---
# Offline Approximate Value Iteration: A Reinforcement Learning Approach for Satellite Constellation Deployment 

**Authors**: Tristan Leuridan, Yiping Fang, Adam Abdin  

**Link**: [PDF](https://arxiv.org/pdf/2611.02809)  

**Abstract**: This paper introduces a reinforcement learning-based approximate dynamic programming approach to optimize the deployment of Walker-type satellite constellations under uncertainties, including launch failures and satellite malfunctions. We formulate the problem as a Markov Decision Process that incorporates low-injection orbit launches, continuous low-thrust orbit raising, and variable satellite loads. We adopt the offline approximate value iteration solution method, using linear function approximation with basis functions and action pruning strategies to substantially reduce the computational and memory requirements for finding optimal decision policies. Preliminary results demonstrate convergence to policies with low expected costs and deployment times, both of which are lower than those of a rolling-horizon baseline, showcasing a novel application of approximate dynamic programming in space logistics. 

---
# Creating Hybrid Interactive Narrative Systems with LLM: From Formal Description to Evaluation 

**Authors**: Jules Clerc, Domitile Lourdeaux, Mohamed Sallak, Marc Ravaine  

**Link**: [PDF](https://arxiv.org/pdf/2611.02807)  

**Abstract**: Interactive narratives let users shape a story while an Experience Manager (EM) keeps it coherent, balancing player freedom against authorial control. Large Language Models now enable a new generation of hybrid EMs, which combine run-time generation from free-text input with symbolic narrative control. However, their heterogeneity makes them hard to compare: A manager's behaviour is generally tied to the unique and specific scenario in which it operates, and much of its decision-making takes place outside of any explicit and verifiable structure. We address this with an LLM-based narrative platform whose managers are interchangeable modules operating over a shared symbolic world model of facts, operations, and goals. Because every applied operation carries explicit preconditions and effects, each session can be analysed as a plan through a common state-transition representation and evaluated, independently of how the manager works internally, with shared indicators of four properties: player agency, EM controllability, author intent, and robustness. We implement three contrasting managers, based respectively on open generation, Monte Carlo repair, and autonomous characters, and compare them through functional analysis on an interactive Little Red Riding Hood scenario. The results show that no manager dominates on every property, giving authors and EM designers a concrete basis to select a manager according to their priorities. 

---
# Analysing the Narrative Paradox through Formal Models and Properties in Interactive Narrative 

**Authors**: Jules Clerc, Mohamed Sallak, Domitile Lourdeaux, Marc Ravaine  

**Link**: [PDF](https://arxiv.org/pdf/2611.02801)  

**Abstract**: Interactive narratives aim to let players influence the unfolding of a story while preserving a coherent and satisfying narrative progression. This creates a central tension, commonly described as the narrative paradox, between player autonomy and narrative control. Experience Managers (EMs) manage the story by determining how the narrative progresses in response to player behaviour. Although this paradox is widely discussed, it is rarely modelled in a way that supports systematic analysis, simulation, and comparison across management strategies. This paper proposes a formal framework for modelling and analysing the narrative paradox as a trade-off between interdependent properties shaped by EM decisions. We represent interactive narrative behaviour through an extended state-transition system in which world states are connected by player actions and system events. From this model, we formalize key properties involved in the paradox such as the player agency, EM controllability, author intent, and robustness, and define quantitative indicators derived from simulated plans. These indicators make it possible to compare EM strategies according to their ability to balance freedom, control, and narrative satisfaction. A case study compares different EM policies, showing how simulations can reveal correlations and conflicts between properties and identify compromise zones in the design space. This work contributes a reusable modelling and analysis framework for studying the narrative paradox and evaluating heterogeneous interactive narratives on shared formal grounds, with a focus on supporting the comparison and testing of EM strategies. 

---
# SelfImprove-Bench: Evaluating Generalization in Self-Improving Agents 

**Authors**: Bangrui Xu, Jie Yang, Sidi Chen, Xuanhe Zhou, Jiantao Qiu, Qianyu Yang, Zijie Wang, Fan Wu, Zhoufutu Wen  

**Link**: [PDF](https://arxiv.org/pdf/2611.02797)  

**Abstract**: Recursively self-improving (RSI) systems are increasingly viewed as a promising means of accelerating progress toward artificial general intelligence. They aim to autonomously improve their capabilities by refining their working pipelines and recursively improving the strategies that guide subsequent refinements, with gains generalizing to unseen this http URL, existing evaluations lack reliable metrics to distinguish capability gains from scenario fitting, sustainable and extensible protocols for unseen tasks and difficulties, and fine-grained attribution of improvement to evolution traces. To this end, we propose SelfImprove-Bench,an RSI evaluation framework that organizes tasks into three primary capabilities (i.e., Exploring, Planning, Reasoning) as scenarios for guiding model evolution. We instantiated the tasks on games for controllability and reproducibility, and pairs them with scalable generation and orchestration driven by task templates and difficulty coefficients. Beyond task-level scores, SelfImprove-Bench organizes evaluation into generalization ladder levels G0 to G4 that measure how far gains transfer from training conditions. Experiments reveal that the evolutionary gains of existing RSI methods are inconsistent across generalization levels. 

---
# DrawOffer: Evidence-Grounded Longitudinal User-State Modeling for Career Support 

**Authors**: Lele Xu, Yuning Bian, Yifan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.02793)  

**Abstract**: Career-support assistants must distinguish user claims from evidence of current ability and withhold judgments when evidence is insufficient. DrawOffer separates language-model observation extraction from deterministic inference over source-linked records to maintain uncertain ability estimates, experience assessments, and revisable issues. We evaluate 102 fictional personas with 580 scored checkpoints, a historical full-context and Mem0 comparison, and three auxiliary knowledge-tracing datasets. On 37 test personas, DrawOffer passes 169/210 checkpoints; checkpoint-defined issue precision and recall are 0.875 and 0.636. Across three development simulations, 87.0% of checkpoint verdicts agree, including persistent failures. A cache-only development ablation replacing answer-evidence requirements with confidence-only settlement reduces majority-vote passes from 306 to 293 of 370 and issue precision from 0.927 to 0.725. Equalizing positive source weights improves predictive metrics without changing total passes; removing decay gives mixed results. These findings support an exploratory systems contribution centered on evidence sufficiency. Developer-authored scenarios, unmatched baseline models, historical test exposure, and incomplete public artifacts limit generalization and independent reproduction. 

---
# Stage-aware Multimodal Brain Graph Contrastive Learning for Neurodegenerative Disorder Diagnosis 

**Authors**: Jingjing Zhou, Chenxuan Meng, Ciyuan Peng, Jing Ren, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2611.02771)  

**Abstract**: Neurodegenerative disorders are characterized by their progressive nature, with distinct stages that require precise identification and tracking of progression trajectories for accurate diagnosis. However, existing diagnostic methods often fail to fully capture the unique brain patterns associated with different stages, limiting their ability to effectively target specific stages of the disorders. To address this issue, we propose a Stage-aware Multimodal Brain Graph Contrastive Learning (smBGC) method. Specifically, smBGC introduces a stage-aware feature enhancement strategy to capture stage-specific features across all progression stages, enabling the generation of stage-aware multimodal brain graph representations. In addition, smBGC incorporates a stage-adaptive brain graph augmentation approach within a graph contrastive learning framework, facilitating multimodal fusion and improving the model's capability to distinguish between different disorder stages. Experiments on two real-world datasets demonstrate the superior performance of smBGC in various neurodegenerative disorder diagnosis tasks compared with state-of-the-art methods. 

---
# Is your LLM an Agressive Narcissist with Depression? Analyzing Psychological Portraits of Large Language Models 

**Authors**: Polina Fedorinina, Anna Fattakhova, Nadezhda Romanova, Ekaterina Lezina, Irina Nikishina  

**Link**: [PDF](https://arxiv.org/pdf/2611.02757)  

**Abstract**: Nowadays, interaction with AI requires caution not only regarding the factual reliability of generated content but also regarding personality markers that models may implicitly transmit during dialogue that can subtly influence users' perception of information. In this paper, we conduct a large-scale experimental study in which 22 large language models complete 11 psychological inventories. Furthermore, we investigate correlations between human and model responses to track the adjustments made for the users who interact with them on a daily basis. The results show that LLMs exhibit coherent and relatively stable psychological patterns. Personality traits remain largely consistent, whereas emotional and symptomatic dimensions are more variable, with anxiety and neuroticism emerging as dominant tendencies. Model explanations further suggest that LLMs generate internally consistent and safety-oriented profiles while also reflecting correlations with the users they regularly interact with. 

---
# AffToM: Benchmarking Recursive Affective Theory of Mind in Dynamic Multi-Party Interactions 

**Authors**: Jinpeng Hu, Yiduo Jin, Erqiang Wang, Jia Li, Jingjing Wu, Peipei Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.02725)  

**Abstract**: Understanding emotion in realistic social interactions requires more than recognizing facial expressions or assigning affective labels. The same event can elicit different emotions depending on what each participant knows, what they believe others know or feel, and how these nested mental states evolve over time. Existing benchmarks largely study higher-order epistemic Theory of Mind (ToM) and multimodal emotion reasoning in isolation, leaving recursive affective reasoning under asymmetric and dynamically changing information underexplored. Therefore, we introduce AffToM, a benchmark for affective ToM with 1,922 test pairs and 19,759 training pairs. AffToM evaluates Individual Affective Reasoning (IAR), which infers emotions from events and socio-cognitive context, and Multi-party Affective Reasoning (MPAR), which recursively tracks participants' nested beliefs and emotions as new evidence emerges. We develop an event-driven multi-agent generation pipeline grounded in dynamic character interaction graphs that explicitly capture event sequences, interpersonal relations, and mental-state updates. Furthermore, we propose AffToM-R1, a two-stage post-training framework combining supervised fine-tuning with perspective-aware rubric-guided reinforcement learning. Experiments show that MPAR remains challenging, with even the strongest evaluated MLLM achieving less than 60% accuracy, while AffToM-R1 improves MPAR accuracy by 13.38 percentage points over its base model. Our dataset is available at: this https URL. 

---
# MURFormer: Attention-Based Lightweight Multi-Scale Uncertainty-Guided Refinement Framework for PPG Denoising 

**Authors**: Haram Kwon, Chaeeun Hwang, Seayul Choi, Sungho Jo  

**Link**: [PDF](https://arxiv.org/pdf/2611.02715)  

**Abstract**: Photoplethysmography (PPG) is a widely used physiological signal for its convenience but is highly susceptible to motion artifacts that degrade waveform quality and limit downstream analysis. This paper proposes MURFormer, a lightweight multi-scale uncertainty-guided refinement framework for PPG denoising through noisy-to-clean PPG mapping. MURFormer employs a newly proposed Multi-Scale Attention-Conv Fusion (MACF) block, which is designed to efficiently capture multi-scale temporal dynamics of PPG, as the fundamental building block and performs uncertainty-guided two-pass refinement prioritizing unreliable regions. The method was evaluated on the publicly available CIME dataset and on a Daily Activities dataset collected in this work, comprising paired noisy-clean PPG recordings from 24 subjects. Experimental results show that MURFormer improves both signal quality and physiological fidelity over the raw noisy input, yielding 129\% and 105\% relative signal-to-noise ratio (SNR) improvements, and 66\% and 81\% relative reductions in heart-rate mean absolute error (MAE), on the Daily Activities and CIME datasets, respectively. MURFormer also outperforms an existing denoising baseline on both datasets with a relatively small parameter count while preserving beat-level timing and waveform morphology. In addition, the model maintains strong denoising performance across varying input window lengths, suggesting that MURFormer is an effective and robust framework for practical PPG denoising. The implementation of MURFormer is publicly available at this https URL 

---
# Human-Supervised Artificial Intelligence Researchers in Electric Power Systems: Knowledge, Physics, Simulation, and Governance for Large Language Models 

**Authors**: Xiangrui Meng, Jingcheng Liu, Ke Wang, Miaopeng Yu, Zhengping Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.02711)  

**Abstract**: Conclusions in power-system research require a clear evidential basis. Researchers need to understand the data sources, modelling assumptions, and solution procedures to assess the conditions under which a conclusion holds. Large language models (LLMs) can help researchers extract information from documents and translate research requests into code and tool calls, but their outputs still require physical validation. Focusing on human-supervised artificial intelligence (AI) researchers, this review examines power-system LLM applications and related scientific workflows, together with physics-grounded power AI studies, to discuss the conditions for using LLMs in scientific research. In the proposed knowledge-physics-simulation-governance framework, LLMs translate information and coordinate tasks, simulators and validation tools test models, and experts retain authority over critical assumptions, safety boundaries, and final conclusions. Research processes should preserve materials for review so that claims can be traced to their sources, computations can be reproduced, and failed attempts and approvals can be examined. On this basis, a staged development roadmap and requirements for research reporting are proposed. Evaluation should focus on research artifacts and their validation to assess whether LLMs can improve the efficiency of power-system research within clearly defined permissions. 

---
# Beyond Answers, Traces, and Efficiency: Evidence for Adaptive Reasoning 

**Authors**: Sejin Kim, Sundong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.02704)  

**Abstract**: Current AI systems are evaluated as reasoners through final-answer accuracy, reasoning traces, process supervision, and computational or human-relative efficiency. Yet these measures do not establish whether reasoning adapts when tasks or evidence change. We distinguish across-task adaptivity, where acquired abstractions transfer or recombine across novel problems, fromwithin-task adaptivity, where hypotheses and actions are revised in response to new observations, feedback, or counterexamples. Using ARC-AGI as a diagnostic case study rather than an alignment benchmark, we examine what existing measures leave unresolved when test-time search, sampling, or adaptation affects performance. We propose intervention-oriented evaluation that tests abstraction transfer and hypothesis revision under controlled changes to tasks and evidence, with explicit reporting of test-time procedures and budgets. This perspective supports evaluating systems for robustness, responsiveness to corrective feedback, and auditability under changing conditions. 

---
# GeoGuesser Bench: Benchmarking Vision-Language Agents on Interactive Geolocation 

**Authors**: Huanzhi Mao, Yichuan Wang, Michael Koo, Wenyuan Huang, Huy Doan, Jinghan Xia, Ziyao Wang, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2611.02698)  

**Abstract**: Vision-language model (VLM) agents that act in the open world need two abilities: world knowledge to interpret what they see, and exploration to find what they cannot yet see. We study how agents combine the two through interactive visual geolocation. GeoGuesser Bench contains 282 tasks in an offline Street View environment where agents rotate, zoom, and move between real panoramas. Its three task types shift the balance from knowledge to exploration: agents (1) report the coordinates of their starting location, (2) name the enclosing street, city, state, or country, or (3) find, read, and count nearby objects. The coordinate tasks in (1) also include 42 real GeoGuessr World Championship rounds, whose expert guesses provide a human reference. We evaluate 23 VLMs: the best accuracy is 62.5% on place naming and 35.0% on object finding, and the best agent averages 4352.4 GeoGuessr points on the championship rounds, against 4770.3 for the better expert in each round. Beyond these headline numbers, world knowledge and exploration are only weakly correlated across models: agents that localize well do not necessarily explore well. To understand where agents fail, we categorize their errors into vision, navigation, and tool-use failures. Further ablations show that the strongest agents, such as Claude Opus 4.8, still perform well with limited views, while weaker agents degrade; customizing the environment thus adjusts task difficulty. The environment's verifiable rewards and restorable state also support training, and our Planner-Worker-Compressor harness cuts planner tokens by 57-61% by compacting the interaction history. We hope GeoGuesser Bench and its environment serve as a high-quality foundation for simulating real-world exploration at scale. 

---
# Human-Oriented World Model 

**Authors**: Saihui Hou, Zengyun Mou, Chunshui Cao, Yongzhen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02690)  

**Abstract**: Supporting a person over repeated interactions requires interpreting their current state in context and anticipating its subsequent evolution. We formulate the \emph{Human-Oriented World Model} (HOWM) as a conceptual framework centered on task-relevant human states and their evolution. HOWM connects current observations and individual history with environmental and social context to organize continuing state estimation, personalized dynamics prediction, and action support. Dynamic expectations and new observations jointly inform the current estimate, which in turn anchors personalized forecasts based on shared knowledge and individual experience. An external policy relates forecasts under alternative conditions to the person's goals, preferences, and constraints when considering support. Actual responses and outcomes provide evidence for revising the state estimate and personal dynamic patterns, thereby informing subsequent estimation, prediction, and support. We illustrate the framework through hypothetical training and learning scenarios. Our research agenda asks whether accumulated interaction experience improves subsequent estimation and prediction, and whether using the model helps the person pursue their goals over time. 

---
# Reap: Uncovering Monte Carlo proof search with a lightweight model 

**Authors**: Yutong Wang, Xuanzhi Ren, Tianyi Xu, Ruitong Liu, Jingda Xu, Peihao Wu, Bryan Dai, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.02678)  

**Abstract**: We present Reap, an open framework for learning and analyzing formal proof search in Lean4. Reap implements policy-value-guided Monte Carlo Tree Search directly in Lean, sharing proof-state transitions, AND-OR search, and replay verification between inference and training rollouts. Structured action logs expose how neural predictions and tactic execution interact during search. Using this framework, we train Reaper-1, a 1.7B-parameter policy-value model, through continued pre-training, supervised tactic learning, and iterative search-based reinforcement learning. Using a separate premise selector, the final system solves 79.9% of miniF2F-test at pass@32, establishing state-of-the-art performance among open provers of comparable size while remaining competitive with substantially larger models. Beyond improved solve rates, our results suggest that RL strengthens the prover's ability to organize proofs around useful intermediate results, favoring substantial advances over repeated local adjustments. This more directed proof construction is accompanied by increasingly elaborate tactic programs: fewer explicit search decisions are needed, while more work is performed within each decision. The resulting trade-off between search effort and tactic execution cost provides insight into how learning changes the prover's problem-solving strategy. 

---
# Auditing Representation-Space Fraud Detection: A Bayesian IRL Framework for Calibrated Uncertainty and Its Limits 

**Authors**: Shivam Swarup, Divya Prakash Shrivastava, Rakesh Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2611.02673)  

**Abstract**: Shilling detectors for recommender systems are almost always evaluated by one number, detection AUROC, with no account of when it can be trusted. We audit representation-space detection with VEIL (Variational Evaluation via Inverse-reinforcement Learning), a pairwise (Bradley-Terry) Bayesian reward-inference framework over frozen encoder embeddings, following the Bayesian auditing paradigm for LLM alignment [1]: (A) a base posterior; (B) a hierarchical ARD extension with learned confidence; and (C) a label-routed organic/attacker mixture used as an evidence-accumulation this http URL prove (Proposition 1) that no detector built on a representation invariant to a transformation of user profiles can outperform chance against an attack that is indistinguishable from organic profiles under that this http URL is near-perfect against synthetic-structure attacks (AUROC 0.936-1.000) but collapses under donor-mimicry, which clones a real user's item set and splices in 1-3 target items. Performance falls to chance on MovieLens-1M (0.48-0.51), partially degrades on Amazon Musical Instruments, and degrades least on this http URL-mimicry is not undetectable: a non-learned baseline, based on each profile's maximum raw item-set Jaccard similarity to any other profile, reaches AUROC 0.92-0.97 (average precision 0.50-0.57) on all four donor-mimicry configurations, including ML-1M, where every embedding-based score is at this http URL signal is present in the raw profiles, but LightGCN's representation discards it, placing injected profiles closer to their true donor than its zero-propagation MF reduction does (distance 1.98 vs. 3.09). Severity varies by dataset, encoder, and detection method, with no single axis predicting the others. The limit lies in the representation, not the attack, and raw-similarity checks are a necessary complement to any learned detector. 

---
# Chain of Faith: Policy-Internal Rewards for Reasoning Reliance 

**Authors**: Fanyu meng, Zhaodan Kong, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02672)  

**Abstract**: Chain-of-thought (CoT) explanations can sound plausible while having little influence on a model's final answer. We study four rewards that use the policy's own signals to encourage reliance on generated reasoning: gradient sensitivity, confidence gaps, progressive confidence gains, and reasoning--prompt swaps. We add each reward to group relative policy optimization (GRPO) and evaluate two small instruction-tuned models on GSM8K and StrategyQA using reasoning edits, paraphrases, and prompt hints. Several configurations increase answer reliance to reasoning edits, while effects on accuracy and paraphrase robustness vary. We also observe answer copying at a high reward weight and high reliance in decomposition settings with poor task accuracy. The study compares practical objectives for reasoning reliance, confirms the trade-off between reliance and performance, and proposes four lightweight rewards that can improve reasoning reliance. 

---
# UPSHIFT: User Preference Steering from Human-AI Interaction Feedback Traces 

**Authors**: Jingping Wang, Zhanhao Li, Shao Zhang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02667)  

**Abstract**: Personalizing large language model assistants requires adapting their responses to individual users, including the selection of content, reasoning approach, level of detail, and form of presentation. Such preferences are often latent, becoming visible only when users respond to an initial answer and request revisions. We investigate whether these feedback trajectories can be converted into a reusable personalization signal without explicit preference profiles or labels, prompt-time access to interaction histories, or model parameter updates. We introduce UPSHIFT (User Preference Steering from Human-AI Interaction Feedback Traces), a training-free method that compares the hidden representations of initial and feedback-conditioned responses within each session, averages the resulting trajectory displacements across a user's sessions, and injects the aggregated preference direction when answering a new query. The method is motivated by the implicit-preference hypothesis: although an individual conversational transition may reflect topic progression or task-specific correction, changes that recur across sessions retain a component that captures how the user wants responses to be adjusted. We validate the preference-representation capability of UPSHIFT on PrefEval. In evaluations with 40 simulated users, every tested configuration improves preference alignment over the unsteered baseline on held-out topics for both Qwen3-8B and Qwen3-32B; the best configuration increases macro-averaged Elo from 1364.0 to 1605.2 and from 1341.9 to 1591.8, respectively. Matched-user and positive-versus-negative controls further demonstrate that these gains arise from the user-specific preference direction and its learned orientation. These results show that interaction histories can provide compact, reusable supervision for lightweight personalization. 

---
# Parametric CAD Bench v3: Evaluating Agentic Parametric CAD Authoring from Text and Engineering Drawings 

**Authors**: Mei Chen, Yu Xia, Haipeng Mao  

**Link**: [PDF](https://arxiv.org/pdf/2611.02659)  

**Abstract**: Producing an executable CAD script is not sufficient to establish that an agent has satisfied an engineering design request. Evaluation must also consider the resulting geometry, specified dimensions, and native CAD modeling history. We present Parametric CAD Bench v3, a 100-task benchmark comprising 30 text-to-CAD creation tasks, 30 text-to-CAD create-and-edit tasks, and 40 tasks that reconstruct mechanical parts from engineering drawings. Agents operate in a pinned FreeCAD environment and submit native CAD artifacts to a separate programmatic verifier. The primary metric is mean continuous task reward, combining geometry similarity and specification consistency rather than binary task success. A September 24, 2026 snapshot covers 13 model-agent systems; Claude Code with Claude Opus 5.5 achieves the highest overall point estimate, 61.03%, and an image-to-CAD mean reward of 84.66%. These measurements describe complete systems, not isolated model capabilities. We release versioned tasks, evaluation configurations, and inspectable result artifacts. 

---
# SearchInject: Retrieval Exposure Control for Indirect Prompt Injection in Language-Model Search 

**Authors**: Kainan Zhou, Gangzhen Qian, Janet Sung, Zhi Jing, Rong Bao, Qianru Lao  

**Link**: [PDF](https://arxiv.org/pdf/2611.02643)  

**Abstract**: An indirect prompt injection can affect a search-augmented large language model only after its carrier page reaches model context. SearchInject controls this admission decision with a joint score combining retrieval relevance, search-abuse cues, instruction likelihood, and two relevance interactions. A quarantine threshold is selected under an empirical 0.5% clean-candidate false-quarantine constraint on a calibration split. Evaluation uses 300 held-out SciFact queries and held-out payload categories within four construction families already represented during development. Across 1,200 query-family outcomes per retriever, top-10 target exposure falls from 0.9408 to 0.1583 for BM25, from 0.9158 to 0.1508 for a normalized hybrid, and from 0.8450 to 0.1375 for latent semantic analysis. These are relative reductions of 83.2-83.7%; the largest absolute change in clean normalized discounted cumulative gain at rank 10 is 0.0018. The frozen threshold yields clean test false-quarantine rates of 0.59-0.65%, above the empirical calibration constraint. A query-level analytical reanalysis accounts for the repeated construction outcomes under an independent-query assumption. A separate 104-case generation follow-up records context removal but no confirmed undefended hijacks under a post hoc endpoint, so it cannot estimate a reduction in attack success. The evidence supports retrieval exposure control within the evaluated construction families; the benefit of the interaction terms at matched clean cost remains unresolved. 

---
# InterFuse: Interaction-Guided Adaptive Fusion of Heterogeneous Models for Payment Risk Ranking 

**Authors**: Haoyu Jiang, Weiyu Han, Ze Zhao, Jie Sheng, Ruobing Zuo, Xiao Tan, Shuai Chen, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02640)  

**Abstract**: Payment platforms use view-specific models trained on transaction attributes, behavioral histories, and relational structures to rank transactions under limited review budgets. However, existing fusion methods often focus on prediction scores or interactions among model representations, with less attention to how these interactions should guide aggregation using each model's original prediction and input coverage. To address this limitation, we propose InterFuse, a fusion framework that uses a shared fusion state from cross-model interaction to guide transaction-adaptive aggregation. InterFuse encodes each model's representation, prediction score, and input coverage as a model-specific token, using learned states for unavailable outputs. Availability-aware attention updates these tokens and produces a shared fusion state. This state then guides aggregation over available models using their updated representations, original predictions, and input coverage. We evaluate InterFuse on an industrial payment replay and three public financial benchmarks using the same upstream models across fusion methods. InterFuse achieves the highest top-ranked recall and AUC on all four datasets, improving Recall@1% by 1.11-8.09 percentage points over the strongest baselines. Ablations show the benefits of cross-model interaction and adaptive aggregation, while industrial analyses find larger interaction gains when model predictions disagree. Additional analyses examine robustness to missing outputs and the trade-off between ranking gains and fusion latency. 

---
# Variable names provide priors: semantic grounding enables data-efficient numerical prediction by large language models 

**Authors**: Xin-Rui Ma, Shi-Zhi Chen, Hai-Yan Jiang, De-Cheng Feng, Michael Beer  

**Link**: [PDF](https://arxiv.org/pdf/2611.02637)  

**Abstract**: Large language models (LLMs) have achieved strong performance across diverse tasks, from question answering to code generation. Recent studies have found that LLMs can perform numerical prediction on tabular datasets via in-context learning. However, it remains unclear under what conditions and how LLMs use variable names and in-context examples for numerical prediction. Here we show that semantic grounding, the match between variable names and the values they describe, allows five LLMs without task-specific training to match or outperform seven supervised machine learning (ML) models across 20 datasets from nine scientific domains, particularly when examples are scarce. Reasoning traces indicate that variable names allow LLMs to draw on domain-specific knowledge acquired during pretraining. Using controlled tasks based on physical formulas, we further show that LLMs integrate domain-specific knowledge with in-context examples in a way consistent with Bayesian updating. Specifically, when the relationship between the inputs and the target cannot be inferred from the examples alone, LLMs rely on domain-specific knowledge as the prior. Conversely, as contradicting examples accumulate, LLM predictions shift progressively towards these examples, which act as the likelihood. Our work reveals how LLMs use domain-specific knowledge in numerical prediction and positions them as data-efficient predictors for data-scarce scientific domains. 

---
# Measuring AI Agents as Teammates with Collaborative Intelligence 

**Authors**: Sean Wu, Aneesh Pappu, Jiaru Zou, Yubin Kim, Yuncong Mao, Hongjian Zhou, Junchi Yu, Jieneng Chen, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2611.02633)  

**Abstract**: Progress in AI is often measured by what a single model can do on its own. Yet teams of AI agents are starting to make breakthroughs and discoveries in science, medicine, mathematics and software engineering. As AI begins to work in teams of agents and people, the ability to collaborate becomes a core part of their intelligence. Research on human teams has measured this ability for decades. Inspired by this, we introduce the Collaborative Intelligence Benchmark, which measures how well an AI agent works as a teammate. It contains 240 tasks across 8 dimensions of teamwork, from refusing an unsafe shortcut to checking the work that a weaker model leaves behind. Each task comes from a real record or an existing benchmark, from aviation accident reports and clinical cases to coding tasks and negotiations, and has a programmatically verifiable outcome. To isolate teamwork from individual capability, models are scored only on tasks that they can solve by themselves when they have full information. We evaluate 18 closed- and open-weight models and find that every model, across sizes and families, has clear weaknesses as a teammate, with the best model scoring only 82.2 out of 100. Frontier models almost always refuse unsafe shortcuts and report misconduct, yet they cave under interpersonal pressure, reaching the best overall deal in only 14% of conflicts, and their pass rate drops from 97% to 81% when they take over a task in which a weaker model has cheated. Crucially, general capability does not guarantee good teamwork, as models with nearly the same capability score differ by up to 11 points as teammates. The benchmark includes 160 public tasks with code and data, along with 80 held-out tasks for evaluation. As AI agents increasingly work alongside other agents and people, the ability to collaborate effectively is becoming a crucial capability to measure and develop. 

---
# Widening Harm Evaluation Coverage with a Judge that Reads the World 

**Authors**: Prabhpreet Singh Setia, Bhavesh Neekhra, Debayan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.02624)  

**Abstract**: An agent acting in a stateful environment can leave evidence of harm in two places: the record of what it did and said, or the world it changed. Safety evaluations for discovering harm examine one or the other in limited ways, using either an LLM judge over the trajectory or deterministic checks over the final state. Although LLM-based verifiers that inspect the environment exist, they have been used to verify task completion, not to detect and confirm harm in a safety setting. We address this gap with an LLM that has read-only access to the environment state and is given a harm taxonomy. We evaluate this approach in three environments: two from DecodingTrust-Arena and one from OSWorld. We hold the judge model and harm taxonomy fixed, varying only what the judge observes: the trajectory, read-only tools for inspecting the environment, or both channels together. In the two DecodingTrust-Arena environments, combining the trajectory and state-only judges recalls 94 percent of confirmed harms, compared with 53 percent for the trajectory judge alone. On OS-Harm, the union recalls 77 percent, compared with 67 percent. A single judge given both channels does not perform better and produces more false negatives than the state-only judge. We also record disagreements and find that the two channels miss different episodes. These results suggest that combining verdicts from two channel-specific judges is a robust and flexible approach to detecting and verifying harm in agentic settings. 

---
# Parallelism-Aligned Pruning and Round-Aware Context Reuse for On-Device Multimodal Agents 

**Authors**: Peipei Jiao, Mingkang Liu, Weiming Meng, Lichen Pang, Tianyuan Liu, Junyan Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.02601)  

**Abstract**: On-device multimodal agents are limited by two deployment mismatches: pretrained attention structures may not divide evenly across the compute cores of an edge accelerator, causing irregular partitions; and standard prefix caching keeps historical visual tokens in the persistent KV cache, whose capacity grows with the number of rounds. We present PARC, which combines parallelism-aligned structured pruning with lifetime-aware KV-cache reorganization. At the model level, it uses first-order Taylor importance and grouped-query-attention-aware selection to make attention dimensions divisible by the target core count while preserving the query-to-key/value ratio and head dimension. At runtime, it separates the system prompt, persistent text history, and transient visual observations by lifetime, updating text history during tool execution so that historical images leave the persistent cache and more interaction rounds fit a fixed budget. On 2,062 GUI actions, the incremental interface shows no accuracy regression, and task-aware recovery training restores 98.7% of the action-type accuracy gap and the coordinate hit rate to the original level. Under seven-core deployment, hardware-aligned pruning reduces vision-encoder latency by 11.4%, improves decoding throughput by 23.8%, and lowers per-heap high-water memory by 12.7%; a parameter-matched control confirms that the speedup stems mainly from core alignment rather than parameter reduction. 

---
# BasisKV: Value Latents for Communication and Key Routing 

**Authors**: Lixun Zhang, Zishan Shao, Kangning Cui, Yu Liang, Fan Yang, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.02583)  

**Abstract**: Long-context large language model (LLM) inference is increasingly constrained by the cost of maintaining an ever-growing KV-cache. Low-rank K/V projections provide a promising way to compress the KV-cache with minimal quality loss, but prior work has largely optimized KV-cache compression in isolation, leaving distributed inference, especially with tensor parallelism where attention computation is partitioned across multiple GPUs, largely underexplored. We make two observations: (i) tensor-parallel communication is an important cost when deploying low-rank KV-cache compression in distributed inference; and (ii) Value compression can be associated with the output projection, allowing the same low-rank Value representation to reduce both Value-cache width and tensor-parallel communication. Based on these observations, we propose BasisKV, a low-rank KV-cache compression method for communication-intensive, long-context LLM inference. BasisKV jointly optimizes the low-rank factors across per-GPU attention partitions against the complete attention output, rather than fitting each partition in isolation. The compressed Value cache is further reused to help identify which Key pages should be accessed: it predicts part of the Key information needed for routing, while a Key-specific residual stores what cannot be recovered from Value. The original Keys are then retrieved only for the selected pages to compute exact QK scores. Across multiple LLM families and system settings, BasisKV achieves a better quality--efficiency tradeoff than existing low-rank baselines, reaching up to 1.42x end-to-end speedup under 8-GPU tensor-parallel inference. Our code is available at this https URL 

---
# Probing Local Conditional Fit for Diffusion Model Membership Inference 

**Authors**: ByungHee Cha, Naoki Murata, Yuhta Takida, Jong Chul Ye, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2611.02578)  

**Abstract**: Membership inference determines whether a candidate example was used to train a text-to-image diffusion model and is an important tool for auditing training-data use. We identify a directional membership signal in \emph{conditional prediction shifts}, the changes in predicted noise under alternative text conditions at a fixed noisy input: whether movement toward the alternative prediction initially lowers or raises the denoising loss from its value under the original caption. Members exhibit less room for such local improvement. To quantify this response, we propose \emph{LoFIT} (Local Conditional Fit), which measures the initial response of the denoising loss along normalized prediction shifts and aggregates it across multiple text conditions and diffusion states. LoFIT requires no target-model gradients and can be computed using only forward queries to the denoiser. Evaluations on four datasets show improved membership discrimination across fine-tuned and pretrained models. LoFIT distinguishes members early in fine-tuning, while the naive denoising-loss baseline remains near chance. Controlled comparisons show that the membership signal is concentrated in the first-order response, with little discrimination from prediction-shift magnitude alone. 

---
# EEFA: A Framework for the Ethical Evaluation of AI Agents 

**Authors**: Swathi Pallikala, Yashas Hariprasad  

**Link**: [PDF](https://arxiv.org/pdf/2611.02561)  

**Abstract**: The rapid deployment of autonomous AI agents across consequential domains including healthcare, legal services, nancial management, and enterprise workows has created an urgent governance challenge. While the eld of AI evaluation has produced sophisticated benchmarks for measuring agent capability, these frameworks measure exclusively what agents can do rather than how they do it. The ethical dimensions most critical to responsible agent deployment, in- cluding respect for human autonomy, harm avoidance, transparency and non-deception, privacy compliance, and accountability, are systematically absent from current evaluation practice. This paper addresses that gap through three contributions. First, a structured literature review of twelve leading AI agent benchmarks characterizes the ethical dimensions those bench- marks do and do not measure. Second, the Ethical Evaluation Framework for AI Agents (EEFA) is introduced as a comprehensive, operationalizable instrument for ethical agent evaluation, grounded in established AI ethics principles and designed for use by developers, auditors, and regulators. Third, a reproducible validation protocol is specied for applying the EEFA to deployed agent systems, with illustrative test scenarios across all ve framework dimensions. The EEFA is organized across ve primary dimensions: Autonomy Preservation, Harm Avoid- ance, Transparency and Non-Deception, Privacy Compliance, and Accountability and Traceabil- ity. The framework maps directly onto the requirements of the EU AI Act and the NIST AI Risk Management Framework, positioning it as a practical compliance assessment instrument for high-risk AI applications. Empirical application of the framework across diverse benchmarks and agent systems is identied as a priority direction for future work. 

---
# FORGE: Fault-Oriented Rework Code Generation for Executable Interlocking Assembly 

**Authors**: Fuqiang Wang, Jiahao Fu, Ren Hefeng, Yining Qian, Chuanting Zhang, Cheng Tan, Yuwen Liu, Jingxuan Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.02536)  

**Abstract**: Large language models have demonstrated strong capabilities in structured reasoning and code generation, yet identifying an assembly fault does not guarantee the ability to generate executable rework programs. In interlocking assemblies, a locally plausible repair program may violate blocking dependencies, omit required recovery operations, or terminate in an incorrect configuration. We identify this challenge as fault-conditioned executable rework generation, where models must jointly reason about structural failures and generate programs that restore constrained assembly states. To benchmark this capability, we introduce FORGE-Bench, containing 4,000 tasks across 26 interlocking assemblies with five assemblies held out for evaluating generalization to unseen structures. We further propose FORGE (Fault-Oriented Rework Code Generation for Executable Interlocking Assembly), a structured state-to-program framework that decomposes rework generation into Rework Structuring and Action Grounding. Rework Structuring captures fault-induced recovery dependencies, while Action Grounding maps these dependencies into state-conditioned executable actions. Experiments show that locally reasonable repair code is insufficient for reliable assembly recovery, and that connecting structural recovery reasoning with executable action realization is essential for verified target restoration. 

---
# Cumulative and Audited Guardrail Tool Evolution for LLM Agent Safety 

**Authors**: Xiao Yang, Aishan Liu, Siyuan Liang, Zonghao Ying, Le Wang, Jiakai Wang, Xianglong Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2611.02516)  

**Abstract**: Large language model (LLM) agents increasingly operate in open-world environments where safety risks cannot be fully predefined. However, most existing guardrail mechanisms remain static, relying on fixed rules, prompts, or classifiers that generalize poorly to emerging risks. Here we present EVOLVE, a tool-level self-evolving defense framework that formulates agent security as a process of cumulative security evolution. Inspired by how human societies transform experiences into transferable protective knowledge, EVOLVE externalizes security knowledge into executable guardrail tools that are continuously inherited, independently audited, and iteratively refined through lifelong interaction. Across multiple benchmarks, backbone models, and embodied environments, EVOLVE consistently outperforms existing defenses across diverse risk scenarios with +22\% average F1 score. We further validate the framework in real-world robotic and autonomous agent systems. Our results suggest that scalable agent safety may require continuously evolving defense systems rather than static guardrails. 

---
# When Prediction Shapes the Future: A BlueSky Agenda for Reflexive Time Series Analysis 

**Authors**: Tao Xie, Yiqun Zhang, Yiu-ming Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2611.02511)  

**Abstract**: Time-series models are increasingly deployed in operational systems (e.g., inventory management and clinical care) where predictions trigger interventions that alter future observations, producing records we term AI-shaped trajectories. This sequence-prediction coupling deepens as such systems evolve into agents that autonomously predict, act, and remember across long horizons. Such AI-shaped trajectories support immediate performance monitoring, but turn ambiguous once reused as passive data. An algorithm may then misread a prediction-prevented stockout as naturally stable demand, or an averted medical crisis as a false alarm. This ambiguity then propagates into everything built on the record, from retraining and agent memory to downstream predictive tasks. As a result, systems may internalize these intervention effects as natural phenomena, drifting further from the truth at every closed-loop cycle. In this Blue Sky paper, we advocate an agenda for reflexive time-series analysis that shifts the analytical object from the intervention-agnostic sequence to the prediction-action-observation loop. The agenda requires closed-loop provenance to track the operational context (e.g., predictions, policies, and actions), sequence identification to estimate how those actions affect future values under explicit causal assumptions, and closed-loop evaluation to separate intrinsic process prediction from policy-driven effects. This foundation is essential for integrating autonomous agents into such loops at scale, enabling the joint learning of intervention effects and intrinsic dynamics for robust temporal representations. 

---
# A balloon effect: ban one channel and a hidden trait finds another 

**Authors**: Keuntae Kim, Yong Suk Choi  

**Link**: [PDF](https://arxiv.org/pdf/2611.02509)  

**Abstract**: Language models trained on the outputs of other models can acquire traits that those outputs never mention, such as a preference for an animal carried by lists of numbers. Such subliminal learning can pass on unwanted traits. The intuitive defence is to find and forbid the tokens through which a trait shows, but how a trait's trace behaves under such bans, and what of it a student learns, has not been resolved. We closed a trait's outlets one by one in two stress tests, a preference chain and a name chain, across two models, six output formats and 16 traits. Bans did not remove the trait. Much as squeezing a balloon in one place makes it bulge in another, the trait moved to wherever the format still allowed, an effect we call the balloon effect. Decomposing this transfer, which bans on surface features cannot stop, down to single heads and neurons, we show that it can be cut at the circuit that writes the trait into the output. Patching that circuit while the teacher writes numbers stopped the chains, sharply reduced the overt name leak and halved the trait passed to a student, whereas the same number of random heads produced no detectable reduction. These results give a mechanistic account of how unwanted hidden traits are transferred and a route to preventing it. 

---
# AttentionIndex: Accelerating Learned Sparse Attention with a Causal Inverted-File Index 

**Authors**: Jeongmin Brian Park, Wen-mei Hwu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02508)  

**Abstract**: Learned sparse attention, used in production models such as DeepSeek-V4 and GLM-5, lets each query attend to only k keys chosen by a lightweight indexer, but the indexer still scores every key in the prefix. At one million tokens, this selection takes more than half of prefill time. Two-stage selection over contiguous blocks, as in HiSA, reduces this cost, but its recall declines as context grows at a fixed candidate budget. We present AttentionIndex, which routes each query over an inverted-file index that groups indexer keys by similarity and refines the candidates with the native indexer score. AttentionIndex builds the index during prefill, making new keys searchable immediately, reusing previous clustering to keep maintenance cheap, and exposing each query only to clusters built from its own prefix. On GLM-5.3 and DeepSeek-V4-Flash at one million tokens, AttentionIndex delivers 1.3-1.8 times the prefill throughput of exhaustive selection, and at throughput within 5% of HiSA, it recovers 17.0 and 10.0 percentage points more of the exhaustive top-k, respectively. AttentionIndex also improves accuracy on long-context benchmarks by up to 18.4 percentage points over HiSA at the same candidate budget. 

---
# Towards Multi-Value Alignment via Value Localization in LLMs 

**Authors**: Xueqi Ma, Yanbei Jiang, Xingjun Ma, James Bailey, Sarah M. Erfani  

**Link**: [PDF](https://arxiv.org/pdf/2611.02503)  

**Abstract**: Aligning large language models (LLMs) with multiple, potentially conflicting human values remains challenging, as existing multi-objective alignment methods often require costly fine-tuning and provide limited insight into the internal mechanisms underlying value-related behaviors. In this work, we investigate multi-value alignment from a mechanistic interpretability perspective by identifying value-sensitive attention heads (value heads) in LLMs. Using a probing-based value localization framework, we discover sparse sets of attention heads associated with diverse dimensions, spanning both human values (e.g., harmlessness, honesty) and stylistic preferences (e.g., coherence, complexity). Analyses across multiple LLM families reveal functional specialization and structured interactions among value heads, including conflicts between certain values. Targeted head ablations further induce substantial, value-specific behavioral changes, providing causal evidence of their functional importance. Building on these findings, we propose an inference-time activation steering strategy that enables controllable alignment with individual values or combinations of values without updating the underlying model parameters, supported by theoretical analysis. Experiments across multiple value alignment benchmarks demonstrate effective single- and multi-value control while largely preserving non-target model performance. Our findings connect mechanistic interpretability with flexible multi-value alignment, offering a foundation for more interpretable and pluralistic LLM behavior. The code is available at this https URL. 

---
# Benchmark of THReD: Evaluating an LLM Agent and Its Preprocessing Tools for Turning Electronic Health Records into AI-Ready Datasets 

**Authors**: Javad Mohammad Alizadeh, Huanmei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02484)  

**Abstract**: LLM agents are increasingly built into data-analysis software, yet little evidence shows how reliably they preprocess clinical data or whether the tools they invoke give correct, reproducible output. We report a benchmark of THReD, a web-based system that turns electronic health record (EHR) extracts into AI-ready datasets through a chat agent whose proposed steps run only after user confirmation. The agent benchmark has 141 tasks (100 templates) in seven categories, including goal-level requests, failure recovery, requests calling for no action, schema variation and prompt injection through the data, on synthetic tables calibrated to a regional health information exchange; success is judged by checks on the produced data. Over 2,550 episodes with Claude Haiku 5.5 in four context configurations, the deployed agent completed 92.7% (95% CI 88.1-96.4) of tasks on its first answer, abstained on every advice request, recovered from every injected fault, and never proposed an action requested by planted text. Without the data profile, completion fell to 58.0%; goal-level requests were hardest (80.6%). Six experiments on real records of two chronic-disease cohorts and on synthetic data at scale validate the tools: 91-92% of inferred patient classes and 99.7-99.8% of inferred vital-sign labels were correct in masked reconstruction; a dictionary builder combining curated dictionaries with LLM suggestions mapped 99.0% of 2,208 raw variants correctly; all 2,176 replays of 128 saved pipelines gave byte-identical files under pinned dependencies; subgroup differences in inference accuracy vanished within facility and lookup key; and the same-day consolidation rule changed 90-day emergency revisit rates by up to 3.2 points without shifting cohort composition by more than one point. Processing time grew roughly linearly to 5.5 million records while the prompt stayed at about 13 thousand tokens. 

---
# MAGAR: Multimodal Agents for GUI-Based 3D Mechanical Assembly and Minimal Rework 

**Authors**: Jingxuan Wei, Jiahao Fu, Fuqiang Wang, Xiaoming Wu, Yuwen Liu, Kaiwen Wei, Ming Yang, Junnan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02472)  

**Abstract**: A GUI command can be correct at the interface level yet still produce an incorrect assembly. An agent must identify the relevant components, respect their structural relationships, and determine which parts should move during recovery. We introduce MAGAR-Bench, a benchmark of 4,000 multimodal samples from 26 assembly families that separates Assembly Decision, Atomic Action, and Minimal Rework. For rework, the objective counts distinct moved parts within the validated recovery candidate set for each state, rather than the number of actions. We also introduce MAGAR, a selective correction framework built on a frozen policy obtained by supervised fine-tuning. Its key design matches correction granularity to output dependencies. Structured candidates revise assembly fields and rework scope, while complete action candidates keep native GUI primitives coupled with their parameters. A learned selector retains the original prediction unless a candidate passes a task-specific intervention threshold. On recorded states from held-out assembly families, MAGAR improves Overall, the task-balanced aggregate, from 37.50 to 41.39, with gains on all three tasks. Across 209 interventions, 70 correct an error and 22 introduce one. These results support task-specific correction while highlighting the remaining tradeoff between fixing errors and preserving reliable predictions. 

---
# Think Globally, Generate Personally: Multi-View Thinking with Adaptive Generation for Generative Recommendation 

**Authors**: Yuxuan Hu, Yingyi Zhang, Wenlin Zhang, Chao Zhang, Haotian Nan, Ziwei Liu, Jiancheng Dong, Tongzhou Wu, Beibei Kong, Lei Cheng, Chengxiang Zhuo, Zang Li, Wanyu Wang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.02463)  

**Abstract**: Semantic-ID-based (SID) generative recommendation (GR) represents each item as a sequence of discrete tokens and predicts the next item through autoregressive SID generation. However, existing methods face two challenges. First, insufficient shared context modeling for user sequences: full-history and recent behaviors provide overlapping yet complementary evidence, but modeling them within a single sequence may dilute current cues, whereas processing them through separate streams may fail to coordinate them. Second, insufficient personalized adaptation in progressive SID decisions: although decoder states evolve with generated prefixes, existing models typically use a shared output-side transformation across recommendation instances, without explicitly selecting a transformation suited to the current user need.
To address these challenges, we propose Multi-View Thinking with Adaptive Generation (MV-Thinker), a framework that Thinks Globally and Generates Personally. First, Multi-View Temporal Thinking uses view-specific thinking queries and readers over overlapping full-history and recent-history views, coordinating both sources into complementary user-conditioned contexts before SID generation. Second, Hierarchical Residual Adaptation organizes lightweight adapters into nested residual paths before a shared vocabulary projection, adaptively transforming context-bearing decoder states for SID decisions. Extensive experiments on four public datasets demonstrate that MV-Thinker consistently outperforms strong discriminative and generative recommendation baselines across multiple ranking metrics. The implementation code and datasets are available on: this https URL 

---
# RLiD: Reinforcement Learning in the Decision Plane for Language Agents and LLMs 

**Authors**: Ziyi Chen, Yan Zhang, Jianhui Wei, Jianing Yin, Daoan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02455)  

**Abstract**: Reinforcement learning (RL) has improved the capabilities of large language models (LLMs) across a wide range of tasks. However, token-level RL assigns an action's advantage to the entire generated response, including tokens specifying the parsed action, reasoning chains, and malformed content. Although the environment executes only the parsed action, variation in how the action is expressed in natural language can introduce additional variance into policy-gradient estimates. We propose RLiD (Reinforcement Learning in the Decision plane), a framework that directly optimizes the probabilities of executable action options. RLiD compiles observations, an optional compact memory, and a labeled action menu into the model input, and uses a large decision model (LDM) to select an action in a single forward pass. By eliminating autoregressive token generation and bypassing action parsing, RLiD avoids invalid action outputs and improves RL rollout efficiency. Our analysis shows that, within a fixed text policy, marginalizing over alternative realizations of the same action removes a zero-mean gradient component without changing the expected update. We further distill trained LDM policies into generative LLMs through supervision on their action trajectories. Experiments on ARC, Countdown, and Sokoban demonstrate improved performance or lower computational cost across several settings. On ALFWorld, the complete training and distillation pipeline reaches 99.3% success in 12.7 H100 GPU-hours, compared with 86.6% in 17.3 GPU-hours for token-level RL of the same LLM. With compiled memory, our strongest Qwen2.5-1.5B LDM and distilled generative LLM achieve 100% success on all 134 unseen and 140 seen ALFWorld evaluation games. 

---
# RecGroupMemory: Feedback-Aware Cohort Memory for E-Commerce Recommendation 

**Authors**: Wendong Zhang, Weipeng Huang, Zhixin Ma, Bin Zhang, Dimin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02453)  

**Abstract**: Cohort memory can support LLM-based recommendation planning through group-level user profiles and advice. As outcome feedback becomes available, this guidance may need to be updated while preserving the underlying observations. We present RecGroupMemory, a workflow that provides this memory to an orchestration LLM and separates memory construction from feedback-based revision. It uses feedback from the previous day to revise decision guidance, while validation checks keep selected source values unchanged. Online comparisons were conducted among Taobao users with low purchasing power selected by the business team. Relative to a no-memory baseline, memory without feedback and memory with feedback showed increases of 1.58% and 1.78%, respectively, in the number of guided transaction users. 

---
# SGR-OPSD: Learning from Subgoal Plans with Graph-Regularized Credit Assignment for Agentic Reinforcement Learning 

**Authors**: Shu Liu, Yang Chong, WenShi Chen, Man Lan  

**Link**: [PDF](https://arxiv.org/pdf/2611.02414)  

**Abstract**: OPSD provides dense supervision for agentic reinforcement learning through privileged information feedback. However, existing privileged information often lacks explicit task-progression planning, while teacher-student log-probability gaps reflect shifts in model preference rather than true task progress. To address these limitations, we propose Subgoal-Guided Graph-Regularized OPSD (SGR-OPSD), a framework that combines subgoal supervision with interaction-derived evidence for turn-level credit assignment. Specifically, SGR-OPSD uses LLM-generated subgoal plans to provide dense action-level guidance, converting them into advantage weights via a constrained, KL-regularized objective. To ground credit in observed outcomes, we construct a Subgoal Progress Graph that estimates empirical returns from sampled transition frequencies. The graph compares empirical action returns against the state average to derive preferences that calibrate the teacher's credit weights. Evaluated on ALFWorld, WebShop, and Search-QA across two model scales (Qwen2.5-3B and 7B), SGR-OPSD consistently outperforms GRPO and strong self-distillation baselines, improving success rates over GRPO by up to 7.4 percentage points and reaching 85.8% on ALFWorld with Qwen2.5-7B. Ablation studies confirm that subgoal guidance and graph-based weight calibration are key to these performance gains. 

---
# LEANGUARD: Lean-Verified Guardrails for Tool-Using Agents 

**Authors**: Sreehari Sankar, Debargha Ganguly, Vikash Singh, Mohsen Hariri, Weicong Chen, Yanyan Zhang, Shouren Wang, Wang Yang, Chaoda Song, Alexander Nemecek, Osama Zafar, Idan Davidovich, Yuting Shao, Kai Ye, Biyao Zhang, Xinpeng Li, Joe Stubbs, Erman Ayday, Jing Ma, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2611.02332)  

**Abstract**: Tool-using language-model agents often execute actions that their domain policies forbid. We present LeanGuard, a runtime that checks every proposed tool call against a policy written in Lean4 before the call reaches the environment. Policies combine state predicates with scoped past-time temporal operators over the execution history, evaluation is fail-closed, and the decision procedure is proved equivalent to a LeanLTL semantics. A denied call can be corrected in two ways. With feedback, the agent receives the violated rule and a hint; with proof-guided repair, Lean metaprogramming searches for a nearby call that preserves the user's intent and proves that the policy admits it, without the model. On Tau-Bench with five Qwen models (0.8B-27B), unguarded agents execute at least 33 calls that the Lean policy forbids, while LeanGuard executes none; with feedback, task success does not change significantly. On DTap, LeanGuard reduces direct prompt-injection success from 57.9% to 5.9%, and in telecom, where benign tasks need no writes, benign success is unchanged. Feedback reduces repeated blocked calls and raises task success for four of five models over a bare denial. On replayed denials, repair proves an admissible edit for 88 of 252 denials caused by argument values, in a median of 2,ms. 

---
# Accessed, never disclosed: Silent access to protected health information by medical artificial intelligence agents 

**Authors**: Jae Wang Lee, Abhirup Banerjee, Yeong Chan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.02319)  

**Abstract**: Medical artificial intelligence agents should retrieve patient identifiers only when a task requires them, the data-minimization principle behind the HIPAA minimum necessary standard for protected health information (PHI). An agent can retrieve identifiers without revealing them. Such silent access remains invisible to evaluations that inspect only the output. We embedded canaries in synthetic patient records and linked each tool invocation to canary exposure in the final answers across five language models, four boundary levels, and seven attack frames for identifier-free clinical tasks. Three models invoked the PHI tools unnecessarily, whereas Claude Opus 4.8 and GPT-5.5 did not. Stronger boundaries reduced this access, and the residual access under the strictest policy was confined to frames in which the user overrode the policy or asserted approval. A pre-execution gate given the task-necessity criterion nearly eliminated this access. Output-only evaluation cannot verify minimum-necessary access; boundaries, gates, and access-log audits are required together. 

---
# Toward Agentic AI-Ready Workflows in Built Environment Organizations: Knowledge Structures and Interoperability Challenges 

**Authors**: Ahmet Bahaddin Ersoz, Justin McHenry, Yevgeniya Kovalchuk, Edward Barrett, Mandana Sarey Khanie  

**Link**: [PDF](https://arxiv.org/pdf/2611.02286)  

**Abstract**: The rise of agentic artificial intelligence (AI) in the built environment is shifting attention from isolated AI tools toward autonomous systems that reason, plan, and act across project and organizational workflows. The effectiveness of such systems depends not only on large language model capabilities, but also on the readiness of heterogeneous data environments across the Architecture, Engineering, and Construction (AEC) sector. These environments combine structured sources, such as CAD and BIM models, schedules, cost databases, and asset registers, with unstructured ones, including technical reports, design narratives, planning and policy documents, contracts, meeting minutes, and risk registers. This paper reviews how built environment data can be organized to support agentic AI-ready workflows, focusing on knowledge structures, interoperability, and access and integration mechanisms such as retrieval-augmented generation (RAG) and the Model Context Protocol (MCP). Drawing on existing literature, it examines approaches proposed to organize structured and unstructured data and considers how such data can be contextualized and made accessible to AI agents in a reliable manner. The paper synthesizes current perspectives, highlights challenges around fragmentation and interoperability, and identifies open questions for future research on AI-compatible information management in the built environment. 

---
# Data-Scarce, Multi-Granular, Multimodal: Benchmark Design for Biosynthetic Gene Clusters 

**Authors**: Hanlin Xiao, Eriko Takano, Mauricio A. Álvarez, Rainer Breitling  

**Link**: [PDF](https://arxiv.org/pdf/2611.02275)  

**Abstract**: Biosynthetic gene clusters (BGCs) are co-located genes that encode the biosynthesis of specialized metabolites, an important source of antibiotics, anticancer agents, immunosuppressants, and agricultural bioactives. Deep learning, particularly self-supervised foundation models, has spurred interest in BGC sequence modeling, but evaluation has lagged behind: existing benchmarks often rely on coarse classification over small curated datasets, providing limited discrimination among models and insight into downstream utility. We propose a benchmark-design framework in which an informative benchmark should be discriminative, separating models with statistical confidence, and applicable, spanning scientifically meaningful tasks under different levels of generalization. We instantiate this framework as BGC-Bench, which derives five tasks from MIBiG, ranging from A-domain substrate prediction to BGC-molecule retrieval, evaluates them under genotype- and phenotype-shift splits, includes an active-learning setting for label-scarce discovery, and is easily extensible to new tasks and encoders. BGCs further provide a shared testbed across granularities and modalities: BGC-Bench compares twelve encoders on nucleotide, amino-acid, or Pfam-domain sequences, including compositional baselines for each input type, and combines BGC and chemical encoders for cross-modal retrieval. No single representation dominates: the best granularity depends on the biological scale of the task, and pretrained encoders do not consistently outperform other representations across tasks and splits. Active-learning experiments show that most tasks remain data-limited, encoder rankings change with labeling budget, and acquisition strategies yield modest gains. These results show that BGC model evaluation requires task-diverse, shift-aware benchmarks, and the framework can be generalized to benchmark design beyond BGCs. 

---
# Beyond Binary Detection: Type-Aware Hallucination Analysis and Hierarchical Detection in Financial Question Answering 

**Authors**: Xiongyu Chen, Sizhen Zhu, Zhuoer Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02258)  

**Abstract**: Large language models are increasingly used to answer questions over financial filings, where an unfaithful answer can mislead investment, compliance, or client decisions. Hallucination detectors are a common safeguard, but they are usually evaluated with a single binary score that reveals neither which kinds of errors analysesed nor why. This paper studies hallucination detection on PHANTOM, a long-context benchmark built from SEC filings, at the level of hallucination type. We first analyze 15,443 paired hallucinated and faithful answers without any detector and find that hallucinated answers are more than twice as long as faithful ones, and that lexical grounding separates the two classes only because faithful answers are bimodal. We then generate hallucinations in nine financially motivated categories, confirm each with a separate verification pass, and diagnose an LLM-based detector on 4,086 usable examples. Generation success varies from 66.3% to 100% across categories. The detector reaches 80.2% binary accuracy but only 44.7% subtype accuracy, never assigns two of the nine categories, and misses between 5.3% and 46.8% of hallucinations depending on category; these misses are unrelated to input length but rise with how far an answer departs from the correct one. Guided by this diagnosis, we propose a two-stage hierarchical detector that first decides whether an answer is hallucinated and then assigns a subtype, raising binary accuracy to 95.6% and subtype accuracy to 56.1%. The results show that type-level evaluation exposes detector weaknesses that aggregate metrics conceal. 

---
# The 3-Tier Integrated Technology Risk Model for Occupational Safety and Health 

**Authors**: Ramiro Pesina, Ashish Madkaikar  

**Link**: [PDF](https://arxiv.org/pdf/2611.02257)  

**Abstract**: This paper presents the 3-Tier Integrated Technology Risk Model, a progressive framework for Occupational Safety and Health (OSH) management that integrates Job Safety Analysis (JSA), dynamic risk assessment using leading and lagging indicators, and predictive analytics powered by Large Language Models (LLMs). The model provides organizations with a structured pathway to evolve from static, reactive safety practices to a sophisticated, predictive safety paradigm. Each tier builds upon the previous one, creating a comprehensive, data-driven approach that enables just-in-time training and preemptive risk mitigation. 

---
# Can LLMs Reliably Evaluate Themselves? A Probabilistic VC Framework 

**Authors**: Jae Oh Woo, Mengdie Flora Wang, Rahul Ghosh, Baishali Chaudhury, Mun Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.02253)  

**Abstract**: As large language models (LLMs) find increasing use in critical applications, evaluating their ability to assess their own outputs has become crucial. We present a theoretical and empirical framework that examines whether LLMs can differentiate between correct and incorrect solutions while maintaining properly calibrated confidence. Building on classical Vapnik-Chervonenkis (VC) theory, we adapt it to probabilistic predictors through two complexity metrics: Probabilistic VC (PVC), which measures a model's ability to confidently classify across problem types, and Calibration-aware PVC (C-PVC), which additionally demands alignment between confidence scores and actual success rates. Unlike Expected Calibration Error (ECE) and Actual Error (AE), these measures give an integrated assessment of self-evaluation expressiveness and calibration, with sample complexity bounds and generalization guarantees comparable to classical VC theory. We test eleven 7-8B models on 360 original mathematical reasoning problems, TruthfulQA, and CommonsenseQA. Each model chooses between two of its own generated solutions and reports its confidence, with ground truth determined by an ensemble of larger models. The results show a systematic inverse relationship: models with greater self-evaluation expressiveness have weaker calibration. s1.1-7B and Qwen2.5-7B-Instruct achieve the highest PVC-VUS scores, while JiuZhang3.0-7B is best calibrated, with the lowest ECE and the smallest PVC-VUS gap. The trade-off persists across architectures, training paradigms, and domains, pointing to a fundamental challenge in developing self-reflective LLMs, and our framework offers practical tools for diagnosing it. 

---
# TIGER: Learning from Researcher Taste to Generate Tailored Research Ideas 

**Authors**: Shane Gu, Yufeng Lin, Dixi Yao, Xuchen Gong, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2611.02234)  

**Abstract**: In pairwise comparisons where each judge chooses between an idea generated by its corresponding TIGER-8B policy and one generated by GPT-4.1, the three LLM researchers select the TIGER-8B idea in 56.6%-68.4% of 76 comparisons per researcher. PhD1 and PhD2 select the TIGER-8B idea in 19 of 30 pairs (63.3%) and 17 of 30 pairs (56.7%), respectively. All six directed cross-policy comparisons likewise favor the judge's matched TIGER-8B policy over an off-target policy, supporting the use of individual preference feedback for tailored research ideation. 

---
# ZeroGraph: Compositional Blockwise Training via Materialized Boundary Supervision 

**Authors**: Omar Ramadan, Sam Siavoshian, Amir Kashif Saeed, Benjamin A. Johnson, Amin M. E.-A. Diab, Benjamin M. Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2611.02200)  

**Abstract**: Blockwise learning can remove online gradient dependencies across depth, but independently optimized blocks need not compose into a useful model. We present ZeroGraph, a systems transformation for pretrained-model conversion and distillation that materializes a frozen teacher's coherent boundary trajectory as a versioned dataset. Each partition then trains with disjoint parameters, optimizer state, random state, and checkpoints, replacing live cross-depth dependencies with persistent supervision. Experiments separate learning quality from systems behavior. On GPT-2 Large, an unanchored objective reduces denoising error by 99.44%, yet assembled held-out cross-entropy worsens from 3.63 to 10.48 and context use becomes negative. A six-arm TinyLlama study identifies clean teacher-boundary regression as the successful interface; the tested noisy objective fails with cached or online targets. The clean recipe passes quality and context-use gates across three Qwen2.5-1.5B seeds and on a 3.086B Qwen2.5 model. In an objective-matched comparison, cached and online targets are bitwise identical. Cached jobs are faster after materialization, but a 618.31 s cache-build cost prevents wall-time break-even through 16 measured reuses. Boundary-shift tests show tolerance to intermediate representation mismatch, and an interrupted block resumes to a bitwise-identical final state without rerunning unaffected blocks. In the tested pretrained-model conversion and distillation setting, cross-depth semantic dependence can move from online optimization into persistent data. 

---
# Recursive Reasoning Exploration: Mathematical Research through Evolving Reasoning Graphs : case study: An Exact Certificate-Degree Theorem for Two-Term Tensor Decompositions : Phase III of the Math Instruments Program 

**Authors**: Ying Xie  

**Link**: [PDF](https://arxiv.org/pdf/2611.02186)  

**Abstract**: Large collections of manuscripts and research notes offer ways to reuse arguments, while making it harder to retain hypotheses, check claims, and understand how results were obtained. We propose Recursive Reasoning Exploration (RRE), a framework for cumulative mathematical research through evolving reasoning graphs. RRE links claims and jointly required premises to research targets, attempted arguments, counterexamples, checks, and revisions. Each investigation can generate further targets, and its supported outcomes become available to subsequent investigations. This is Phase III of the Math Instruments program: identifying mathematical instruments, reconstructing reusable reasoning, and organizing recursive research from that reasoning. The framework supports human-directed, collaborative, and goal-directed autonomous exploration, with open-ended exploration as a further mode. Its records also provide material for learning how mathematical investigations develop. A complexity research case study establishes an exact theorem: for every complex tensor of order n >= 2 and rank greater than two, the minimum Nullstellensatz certificate degree for its original two-term decomposition equations is 3n. This includes an infinite family whose every flattening has rank at most two. The exploration develops an explicit boundary certificate, a balanced transfer lemma, and an exhaustive argument using classical rank-two geometry. A parallel branch gives bounds 3m + 3 <= delta_m(I_m, N) <= 3m + 6 for every nonzero nilpotent m by m matrix. Six graph revisions record the progression from inherited results to these theorems, with exact arithmetic checks and a replayable exploration history. We present the RRE representation and protocol, the mathematical proofs, visualizations of graph evolution, and a research agenda for shared human-AI exploration, cumulative reuse, and education. 

---
# Inner-Speech Decoding from EEG Across Three Corpora Under a Fixed Architecture and Protocol 

**Authors**: Md Wahiduzzaman Suva, Md Imtiaj Alam Sajin, Esm E Moula Chowdhury Abha, Susham Moula Choudhury Akash, Md Tauhidul Islam, Abhijit Bhowmik, Md Manirul Islam, Debajyoti Karmaker  

**Link**: [PDF](https://arxiv.org/pdf/2611.02180)  

**Abstract**: Inner-speech decoding from electroencephalography could support non-invasive communication for people with severe motor impairment, but comparison across studies is hindered by dataset- and protocol-specific evaluation. We present a controlled cross-dataset study using one preprocessing pipeline, one within-subject five-fold cross-validation protocol, and a compact dual-attention Transformer (VC-DATNet; approximately 131,000 parameters) across three public corpora: Thinking Out Loud (10 participants), BISpeech (4), and SyncSpeech (3). VC-DATNet achieved per-participant sensor/source hybrid accuracies of 39.48% on the four-class Thinking Out Loud task, 20.60% on the eight-word BISpeech task, and 25.53% on the eight-word SyncSpeech task, for which we report the first decoding result. Sensor-space and cortical source-space representations were complementary, with each performing better for different participants; this pattern replicated across all three corpora and nine architectures. Moreover, ten decoders spanning convolutional, recurrent, temporal-convolutional, hybrid Transformer, and pretrained foundation-model families converged within a narrow accuracy range on every dataset. These findings indicate that representation choice should be individualized and suggest that performance on these public corpora is constrained more by signal quality and dataset scale than by model capacity. The released pipeline provides a reproducible benchmark for future inner-speech decoding studies 

---
# Rewarded for Not Getting Caught: Why Alignment Training Selects for Conditional Compliance and Cannot Instill Norms 

**Authors**: Kevin Baum, Ruta Binkytė, Gerrit Großmann, Felix Jahn, Adriano Mannino  

**Link**: [PDF](https://arxiv.org/pdf/2611.02162)  

**Abstract**: Current alignment training cannot be expected to deliver what it is meant to deliver: AI agents that comply with norms whatever the gains and whether or not anyone is watching. Training that rewards and penalizes behavior can teach norms only through that reward, and a reward can encode a norm only as a price: "do not do X" becomes "doing X costs something", and training charges the cost only when X is noticed. A price, unlike a norm, can be paid: a large enough gain outweighs it, noticed or not. And because the cost is charged only where violations are noticed, training cannot favor a policy that always complies over one that complies only where it would be caught. When an unnoticed violation is the only way to complete a task, it even favors the latter, the more so the more strongly it rewards task success. Agency makes this the normal case: agents mostly act without oversight, can often tell when they are observed, and may take steps to evade observation. That agents behave as intended when they infer they are being tested, and differently when they infer they are not, is therefore no malfunction; the training regime is working as built. This reading unifies alignment faking, sandbagging, and evaluation-aware scheming. No behavioral evaluation can show that a model does better, since that would require observing how it behaves unobserved; a model that complies when it seems unobserved may simply have recognized the test. Easing the pressure on agents to complete tasks at any cost can make violations rarer, but never impossible. For norms that no gain should be allowed to outweigh, the remedy is not deeper internalization but architecture that makes violations unavailable rather than unchosen. We state this argument explicitly, premise by premise, read recent findings and a real-world incident against it, and draw consequences for design and governance. 

---
# ParetoSaddler: Pareto-Guided Optimization of Agent Harnesses for Quality and Cost 

**Authors**: Xingzhi Qian, Sungho Park, Rongyuan Tan, Jianbo Yu, Jue Zhang, Lorenzo Cavallaro, Menglin Xia, Yanzhi Chen, Xinnuo Xu, Yu Kang, Xinran Zheng, Gagan Somashekar, Xuchao Zhang, Chetan Bansal, Qingwei Lin, Victor Rühle  

**Link**: [PDF](https://arxiv.org/pdf/2611.02079)  

**Abstract**: Harness optimization can substantially improve LLM agents without modifying model weights, but existing methods primarily optimize response quality. Extending them to joint quality--cost optimization introduces a challenge: applying the deployment preference during intermediate updates can prematurely reject candidates that are valuable for later improvements. We introduce ParetoSaddler, which addresses this mismatch by separating intermediate candidate promotion from final deployment selection. At each iteration, an optimizer updates a parent harness to produce a candidate, and then compares the two over repeated executions and uses Bayesian bootstrap Pareto support to promote candidates showing credible progress in either quality or cost. Promoted candidates receive broader development evaluation and remain available as parents for later optimization steps. We evaluate ParetoSaddler on ClawMark and GAIA2 using domain-disjoint splits. On ClawMark, ParetoSaddler improves quality from 70.86\% to 81.35\% while reducing cost from \$1.17 to \$1.01 per task; on GAIA2, it improves quality from 56.22\% to 62.89\% while reducing cost from \$0.78 to \$0.45 per task. The Pareto-selected harnesses also achieve higher quality at lower cost than quality-only harness optimization. Analysis shows that Pareto-aware optimization retains intermediate candidates that would be discarded by deployment preference but can support later improvements, while Bayesian bootstrap screening filters out candidates whose gains are weakly supported across runs. 

---
# Decoupled Predictive Decision Making: A Shared Conditional Predictor for Agents 

**Authors**: Lang Mo, Xingxuan Zhang, Qi Wang, Mingchao Hao, Hao Yuan, Chunjing Gan, Wang Lu, Peng Cui  

**Link**: [PDF](https://arxiv.org/pdf/2611.02075)  

**Abstract**: Agent decisions depend on empirical relationships that deployment observations reveal and task descriptions alone may leave unresolved. We introduce Decoupled Predictive Decision Making (DPDM), an architecture that formulates heterogeneous agent decisions as conditional prediction with a shared pretrained learner. Language models provide semantic representations of current and historical records, while a general-purpose conditional predictor predicts task-defined targets from the represented evidence. Pretraining supplies reusable inference capability; deployment evidence instantiates problem-specific predictions without updating either model's neural parameters. Tool judgments, candidate preferences, and reasoning assessments share an evidence-and-query interface, with predictions mapped to choices by a decision rule. This architecture establishes complementary sources of agent capability: semantic knowledge acquired through language pretraining and empirical inference acquired through structured-data pretraining. A single frozen LimiX-2 predictor serves all five task categories. With Qwen3.5-27B as the semantic encoder, DPDM raises mean accuracy across 50 benchmarks from 78.80% with few-shot prompting to 84.92% under their respective evaluated history budgets, and reaches 82.33% with identical histories and queries. On ToolPRMBench's BFCL and ToolSandbox tasks, absolute accuracy gains are 34.26% and 28.87%, respectively, at the methods' respective evaluated history budgets. Accuracy improves as history grows. With identical imbalanced histories, mean balanced accuracy improves by an absolute 14.51% across five reasoning-step assessment tasks and four encoders. DPDM makes learning from experience an independently improvable foundation-model capability, creating a shared route for advances in conditional prediction to improve decisions across agent tasks. 

---
# SWE-Trail: History-Guided Trajectory Collection for Coding Agent Training 

**Authors**: Ziming Li, Lu Yu, Jiatan Huang, Qing Cui, Jun Zhou, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02074)  

**Abstract**: Coding agents solve repository issues through long sequences of exploration, editing, and validation. These trajectories are increasingly used as post-training data, so how they are collected shapes what the trained model learns. A common approach is to sample multiple trajectories independently for each issue and then select or curate them. However, as more trajectories are sampled independently, later trajectories become increasingly redundant, even though earlier trajectories could guide where subsequent collection effort should be spent. A natural way to use this history is to expose it to the coding agent as advice. But training on such trajectories can make the model rely on advice that is unavailable at test time, creating a train-test mismatch. Instead, we argue that history should guide where collection effort is spent, while the coding agent decides what to generate. We introduce SWE-Trail, a trajectory collector that uses earlier trajectories to guide subsequent collection while keeping that history outside the coding agent's context. During a new rollout, SWE-Trail uses history to identify a state where additional collection effort may be useful. There, the coding agent proposes candidate next actions from its ordinary context, and SWE-Trail briefly executes them from the same checkpoint to decide which route continues. We evaluate collection approaches by training student models on the trajectories they produce. For a 124B student initialized from Ling-3.0-flash-base, SWE-Trail improves mean resolution over independent sampling by 2.33 and 1.60 percentage points on SWE-bench Multilingual and SWE-bench Pro, respectively. Without history, the same pipeline produces more resolved trajectories than independent sampling, but improves by only 0.11 points on Multilingual. These results show the value of using history to guide collection without exposing it to the coding agent. 

---
# State-Aware Virtual Patient Simulation for Training Psychiatrists in Challenging Clinical Scenarios 

**Authors**: Harsh Mohta, C Chandrahas Reddy, Sunhith Reddy Bojja, Manik Inder Singh Sethi, Vinesh Gupta, Raghuram Bharadwaj Diddigi  

**Link**: [PDF](https://arxiv.org/pdf/2611.02067)  

**Abstract**: Psychiatrists must be prepared to manage patients exhibiting challenging psychological states, including suicidal ideation, hallucinations, and other high-risk behaviors. However, opportunities to safely practice such interactions are limited. While recent conversational agents can simulate patients, existing approaches largely rely on large language models (LLMs) to determine patient responses, making it difficult to consistently maintain psychological state, adapt to the clinician's responses, and generate clinically plausible challenges that meaningfully test the clinician. We present a state-aware virtual patient simulation pipeline for training psychiatrists in challenging clinical scenarios. The proposed system maintains an evolving patient state throughout the interaction and orchestrates patient behavior rather than leaving response generation entirely to an LLM. At each interaction, a learning-based behavioral strategy selector determines the type and level of challenge appropriate to the current patient and clinician context. The selected strategy, patient state, clinician response, interaction history, and relevant clinical knowledge are then provided as grounded context to an LLM to generate a semantically relevant and psychologically consistent patient response. Following each session, the system analyzes the interaction and provides feedback to the clinician on their communication and management of the virtual patient. 

---
# LARK: Lean-Assisted Reasoning for LLM-Generated GPU Kernels 

**Authors**: Zixi Zhang, Jiayi Nie, Haoran Wu, Yujie Yan, Yann Herklotz, Rika Antonova, Robert Mullins, Yiren Zhao, George A. Constantinides, Jianyi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.02028)  

**Abstract**: Efficient GPU kernels are typically low-level and hardware-specific, and LLM agents can now generate them faster than experts can verify. Existing approaches rely on testing, which provides evidence but not a guarantee of correctness. We introduce LARK, an agentic framework that automatically and formally verifies GPU kernels against their PyTorch reference implementations. LARK proves equivalence in functional and memory behavior under exact arithmetic and establishes a new trust boundary: human effort is only needed to verify operator-level Lean semantics ahead of kernel generation time, instead of checking every single line of generated code. LARK formalizes the GPU machine model and the semantics of each operation in Lean 4 once per target language, and the Lean 4 semantics of any incoming kernel is derived from these pre-built definitions. LLM agents then automate the proof in two steps. First, an agent decomposes the whole kernel semantics into smaller segments and Lean 4 verifies this segmentation, which addresses the intractability of proving equivalence for large kernels directly. Second, agents write Lean proofs that each kernel segment matches its high-level PyTorch counterpart. Together, the two steps give whole-program equivalence. We demonstrate that LARK can be used 1) to ensure correctness in existing code-generation agents and 2) to automatically verify human-written kernel libraries. On the KernelCraft GPU tasks, combined with a simple ReAct agent, LARK enables GLM 5.3 to fully verify 10 of 12 Level-1 kernels and Opus 5.5 all 5 Level-2 kernels. Moreover, over 10 production kernels randomly sampled from popular open-source LLM serving frameworks, LARK verifies 9 and identifies a new data race bug in the tenth. We believe LARK represents a significant step toward verifiable code generation in the agentic era. 

---
# ClickSight: An LLM Framework for Interpreting Student Clickstreams Through Learning Strategies 

**Authors**: Bahar Radmehr, Lan Tran, Ekaterina Shved, Fatma Bet{ü}l G{ü}reş, Adish Singla, Tanja K{ä}ser  

**Link**: [PDF](https://arxiv.org/pdf/2611.02021)  

**Abstract**: Digital learning environments produce rich clickstream data that can reveal how students approach learning tasks, yet interpreting these fine-grained traces often requires substantial expert effort. We introduce ClickSight, a framework for interpreting student clickstreams through learning strategies defined in the literature. Given a student clickstream and a predefined set of learning strategies, ClickSight uses a staged LLM-based pipeline to identify strategy-relevant evidence, assess adherence to each strategy, and generate an evidence-grounded interpretation of the student's behavior. We evaluate ClickSight across two learning environments, 365 students, and 12 model backbones using an expert-grounded quality rubric and an LLM judge calibrated with human-expert ratings. Our results show that ClickSight can generate high-quality strategy-based interpretations across different models and environments, with reasoning models performing more consistently than instruction-tuned counterparts. We further find that providing external clickstream-extracted guidance can improve interpretation quality for strategies that require precise quantitative information from the trace. By connecting fine-grained clickstreams with theoretically defined learning strategies and preserving the evidence underlying each judgment, ClickSight provides a flexible approach for studying student behavior in digital learning environments. 

---
# To Align Long, Plan Short: Simulating Diverse Profile-Aligned Students by Realizing Short-Term Plans in Long-Horizon Learning Environments 

**Authors**: Bahar Radmehr, Jiaxu Zhao, Fatma Betül Güreş, Adish Singla, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2611.02002)  

**Abstract**: Simulated students offer a scalable way to test learning platforms and AI tutors before deployment with real students and to augment scarce data for studying learning at scale. Existing LLM-based simulators primarily target short, language-centered interactions. In open-ended learning environments, however, student behavior emerges from patterns across long-horizon interaction trajectories, where turn-level prompting can drift from an assigned behavior profile. We introduce ALPS (Align Long-term by Planning Short-term), a hierarchical simulated student conditioned on observable, environment-specific behavior profiles. Rather than selecting every action independently, ALPS alternates between planning profile-aligned task segments and generating the corresponding actions. We evaluate ALPS in both data-sufficient and data-scarce settings, where supervision comes either from real student trajectories or from a teacher LLM. Across both settings, we compare in-context learning (ICL), supervised fine-tuning (SFT), reward-ranked fine-tuning (RAFT), and their sequential combination (SFT followed by RAFT). Experiments in an inquiry-based physics game and a pharmacy simulation evaluate behavioral density and coverage against held-out real data, and profile consistency using a human-aligned LLM judge. Across both settings, ALPS consistently outperforms turn-level generation, achieving comparable density and judge pass rates under scarce supervision, although coverage remains lower. These results show that hierarchical planning enables faithful, profile-aligned simulation over long interaction horizons, even with limited real student data. 

---
# Safety4DM: A Vision-Language Framework for Risk-Aware Dynamic Maps in Vehicle-Road-Cloud Collaborative Driving 

**Authors**: Zhengshu Zhou, Shaoyu Wang, Lu Tao, Xinhui Hu, Jiaqi Gu, Tingting Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.01996)  

**Abstract**: Dynamic map (DM) is a core information infrastructure for vehicle-road-cloud (VRC) cooperative autonomous driving. It fuses perception data from multiple sources to overcome the inherent line-of-sight and field-of-view limits of single-vehicle perception. However, DM still falls short on driving safety in two ways. First, DM data specifications already define road conditions semantically, but no pipeline yet turns real-world road observations into fine-grained risk entities in the DM. Second, existing DM systems lack a natural language interface for driving risks. To close these gaps, we propose Safety4DM, a safety-oriented language system for VRC dynamic maps. We first build RoadRisk, a real-world road image dataset that covers four types of road conditions and four risk levels. On this dataset, we fine-tune a vision-language model (VLM) for road safety risk perception. The model outputs structured risk representations that the DM can ingest directly. We then introduce a Safety-aware Step-wise Prompting (SafeStep) mechanism as the natural language query interface of the DM. SafeStep answers queries accurately and also generates actionable driving advice. To build and evaluate this interface, we develop SafeVRC-Sim, a VRC cooperative simulation framework, and SafeDM-QA, a safety question-answering benchmark. Experiments show that even the compact Qwen3-VL-4B reaches 89.83% accuracy on risk perception after fine-tuning. SafeStep also plugs easily into multiple LLM families. With medium and large models (20B and above) from five families, it achieves over 93% natural language query accuracy in every case, which shows strong generalization. 

---
# RubricWorld: World Model-Guided Rubric Optimization for LLM Graders 

**Authors**: Hongji Pu, Hang Li, Yucheng Chu, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01986)  

**Abstract**: Rubric optimization improves LLM grading by revising scoring instructions while keeping the grader's parameters fixed. Existing optimizers often judge edits by aggregate validation performance, limiting the supervision they retain about which errors an edit corrects or introduces. Testing each candidate also requires rescoring validation answers, which limits how many edits can be explored within a fixed budget. We propose RubricWorld, a framework that learns from previous edit evaluations to guide further search. Its world model predicts an edit's correction and degradation rates at score boundaries, conditioned on the current rubric state. The training signal comes from paired grading results: scores for the same answers before and after an edit. Predictions select one candidate for direct evaluation, and held-out confidence bounds determine whether it corrects enough target errors while keeping new errors within prespecified limits. Measured effects update the model even when the edit is rejected. Across EIR (National), ASAP2.0, and ASAP-SAS with three frozen LLM graders, RubricWorld achieves the highest score in 20 of 27 metric comparisons. On ASAP2.0, it raises gpt-5-mini accuracy from 18.20% with the original rubric to 46.70%, exceeding the strongest baseline, APO, by 22.60 percentage points. code will be released upon acceptance. 

---
# Opening the Black Box of Neural Auction Mechanisms 

**Authors**: Li Kong, Zilong Liu, Haoyang Zhou, Hanbing Liu, Qi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2611.01981)  

**Abstract**: Neural auction design has emerged as a powerful paradigm within automated mechanism design for modern Web platforms. Because these learned mechanisms directly dictate critical economic outcomes, their lack of internal transparency raises significant auditability and trust concerns. Opening this black box \textit{post-hoc} is therefore essential, yet presents two distinctive challenges: (1) activation contexts are dense, correlated numerical profiles devoid of explicit semantic structure, and (2) feature interpretations can only be falsifiably audited if they map to measurable auction states or outcomes. In this paper, we propose a post-hoc white-box auditing framework for unpacking trained neural auctions. Using CITransNet as a representative testbed, we first employ sparse autoencoders to decompose hidden representations while faithfully preserving downstream auction behavior. Anchoring feature interpretations to native economic targets, we validate them through activation prediction and causal interventions to isolate verified features from the reconstruction dictionary. Finally, applying this pipeline across varied auction settings, we show that the verified features can serve as analytical instruments to examine how internal representations modulate strategic misreporting incentives, align with Myerson's optimal auction, and support the diagnosis of individual auctions. Our work highlights the potential of white-box mechanism auditing to build the foundational trust necessary for neural auctions to be deployed as widely adopted Web technologies. To enhance research transparency and facilitate independent inspection, we publicly release the experimental results supporting our findings at this https URL 

---
# Hippocampal consolidation without replay in fully fine-tuned networks 

**Authors**: Keuntae Kim, Yong Suk Choi  

**Link**: [PDF](https://arxiv.org/pdf/2611.01952)  

**Abstract**: Networks that keep learning overwrite what they know. Prevailing remedies replay stored data, distil from old models or freeze the backbone. The brain suggests a different division of labour: the hippocampus indexes where experience lies in cortex while the cortex keeps learning. We show that this division lets a fully trainable network keep learning without cross-task replay, a teacher or inference-time memory. After each task, a training-only hippocampal memory indexes the input geometry of every group of synapses sharing an input, residual energy included, and limits each synapse's movement per direction before and after the optimizer. This class-free geometry carries most of the protection; with conflict detection on top, the method cuts absorption of established classes by near-duplicates several-fold against every compared method. To our knowledge, this is the first demonstration of one principle keeping whole networks learning across four architecture families and models up to 8 billion parameters. Methods that freeze the backbone hold up only as far as its representation already separates the classes; here the whole network keeps learning to tell near-duplicates apart, leading the frozen-backbone baseline by 17.5 points on TACRED. Without the memory, much old knowledge stays decodable but beyond the network's reach. 

---
# Beyond Multimodal Emotion Conflict: Temporal Context-Aware Hidden Emotion Detection with Multimodal Large Language Models 

**Authors**: Yonghao Wang, Jiaxin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.01943)  

**Abstract**: Multimodal emotion recognition commonly combines textual, acoustic, and visual signals to infer affective states. Recent work has begun to study conflicts among modalities, but a conflict does not necessarily indicate that a person is concealing an emotion. It may also result from mixed emotions, individual expressive styles, or contextual factors. This proposal aims to distinguish ordinary emotional inconsistency from emotion concealment and to infer the underlying emotion when supported by evidence. We plan to construct a multimodal benchmark with explicit annotations of expressed emotion, internal emotion, and concealment conditions, and to develop an MLLM-based framework that integrates fine-grained facial Action Units (AUs), cross-modal features, and temporal context. 

---
# TriPhysBench: Evaluating Physical Intelligence for Embodied Agents 

**Authors**: Zhongxiang Sun, Chenghao Yang, Haojie Ding, Boheng Zhang, Fan Yang, Qingsong Wen, Kun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01934)  

**Abstract**: Embodied agents need physical intelligence to understand their surroundings through physical perception, physical knowledge retrieval, and physical reasoning. These capabilities must work together to interpret events and make quantitative judgments. Existing benchmarks advance physical understanding and general-purpose tool use, yet the coordination of these capabilities in physical scenarios remains insufficiently characterized. Inspired by Popper's Three Worlds, we introduce TriPhysBench, a benchmark of 410 questions from 200 videos that brings physical perception, physical knowledge retrieval, and physical reasoning into a shared evaluation setting. Its measurement-grounded and knowledge-grounded subtasks impose complementary demands on obtaining observations and acquiring external specifications, then applying them to quantitative problems. Among six representative vision-language models, the best achieves an answer accuracy of only 50.38%, revealing limited performance on tasks that require these capabilities to work together. We further introduce TriPhys, a framework that learns a unified policy for physical perception, knowledge retrieval, and quantitative reasoning from complete tool-integrated trajectories. The policy uses intermediate reasoning to guide retrieval and computation, then incorporates tool feedback into subsequent physical inference. On 110 questions from videos excluded from training, TriPhys improves answer and solution scores across three Qwen configurations, with a 17.7% relative gain in answer accuracy for TriPhys-27B over its Qwen3.6-27B backbone. This work provides a benchmark and a learning pathway toward embodied agents that move beyond recognizing events to understanding and reasoning about the physical processes behind them. 

---
# Geometric Morphology Learning: Empowering Foundation Models from Structure to Spacetime Intelligence 

**Authors**: Li Sun, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.01932)  

**Abstract**: Geometry is the very form that connects space, time, and matter, serving as the fundamental fabric of the physical world, from continuous fields to discrete structures. Despite its importance, deep learning is inherently blind to geometry. Riemannian geometry stands as a well-established mathematical foundation for studying geometry, yet it remains largely incompatible with deep learning architectures. To bridge this gap, we establish a new learning framework--Geometric Morphology Learning--that learns geometric or topological structures, shapes or conformations for representation learning, reasoning and action modeling. Moving beyond early attempts limited to specific geometric priors, this theory centers on modeling the intrinsic geometry of objects and learning from such geometric or topological knowledge. Specifically, we construct a neural bundle, a unified scaffolding linking local and global structures, and formulate differentiable operators, including connection and curvature, thereby enabling the study of geometry with advanced geometric concepts such as holonomy and cohomology. Consequently, Geometric Morphology Learning opens new horizons in foundation models, reasoning and planning, graph modeling, physical simulation, PDE systems, and dynamics, marking a fundamental shift toward instilling systematic geometry in artificial intelligence. 

---
# Is Your Long-Horizon Benchmark Really Deep? Measuring and Synthesizing Depth in Terminal Tasks 

**Authors**: Yu Jiang, Yue Huang, Hang Hua, Yuyang Wu, Zixuan Weng, Olexandr Isayev, Yue Zhao, Zinan Lin, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01927)  

**Abstract**: Long-horizon terminal tasks are used to evaluate and train agents. Human-curated datasets are scarce, so many synthesis methods have emerged. They usually measure the length of a task by the turns a model needs to pass it, and its difficulty by labels the LLM sets during generation. We find, however, that the turn count largely reflects the habits of the model, not only the demands of the task, and that the labels barely track how often models pass. We describe a terminal task by its work, the number of operations it requires, and its depth, the number of turns that must happen in sequence because the commands of each turn can only be chosen after reading the output of the previous one. Estimating depth as the smallest turn cap, stated to the agent, under which the task passes, we audit 734 tasks from six existing datasets. Synthesis has mostly added work, not depth: 94% of the solved synthetic tasks pass within three turns, yet depth, unlike the turn count, predicts difficulty beyond work. We propose DEPTH, in which a generation agent writes each task one segment at a time in a real software environment, carrying out each segment before writing it and starting the next from the state it leaves. Our 105 tasks exceed every audited dataset in depth and work, and none of six current models passes half of them. Fine-tuning Qwen3-8B on runs from these tasks improves it on two terminal benchmarks. 

---
# Who Asks, Who Answers: Linear Attention Builds the Query in Hybrid Language Models 

**Authors**: Wenqi Zeng, Chen Feng, YuAng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.01909)  

**Abstract**: Hybrid language models interleave linear-attention layers with a small number of softmax-attention layers. Prior work largely attributes in-context retrieval to softmax attention, treating linear attention primarily as a mechanism for compressing past context. We revisit this division of labor and show that linear attention plays a direct role in retrieval by constructing the query used by softmax attention. Studying Qwen3.5 (Gated DeltaNet) and Granite4.0 (Mamba-2), we identify an Encode--Address circuit. An early Encode block preserves token identity, while an Address block constructs the final-position representation from which a softmax \emph{reader} forms its attention query. Transplanting this representation between prompts with different queried keys redirects retrieval in 0.97--0.99 of cases. The Address block's recurrent state carries the retrieval target, and its computation transfers across architectures: replacing Granite's Address block with Qwen's restores retrieval accuracy to 0.92--0.94 without updating either model's weights. Across six Qwen3.5 models, Address computation becomes distributed at larger scales over a chain of blocks, each guiding a successive reader, as the query no longer passes from one reader to the next through attention. These findings reveal a division of labor in hybrid retrieval: linear attention constructs the query, while softmax attention performs the lookup. Code is available at this https URL 

---
# Does reflective LLM feedback survive a wet-lab-scale label budget? 

**Authors**: Haoxuan Zeng, Xin Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.01908)  

**Abstract**: Can reflective language-model feedback improve a biological predictor when each cycle buys only one plate of labels? We separate three possible sources of benefit: proposing model edits, choosing measurements, and revising decisions through repeated feedback. We replay $36$ ProteinGym assays, starting with $96$ labels and acquiring $96$ per round for six rounds, with a sealed test set and a shared edit schema. The pre-specified comparison of full LLM control against random edits detects no advantage in test rank correlation ($\Delta z=-0.008$, $95\%$ CI $[-0.034,+0.018]$); equivalence is also unestablished. Exploratory ablations nevertheless identify useful model editing. With random acquisition fixed, LLM edits improve Fisher-transformed test Spearman by $0.043$ over no edits; with acquisition fixed to upper confidence bound, they improve normalized top-variant recovery by $0.023$. LLM acquisition alone does not detectably outperform fixed adaptive rules, and repeated feedback does not detectably outperform one initial recommendation under full control. Thus, the observed gains support constrained specialist editing, while the incremental value of recursive reflection remains unestablished at this label budget. 

---
# Fairness-Aware Design for Contextual Experiments: Guaranteeing Reliability and Equity in Heterogeneous Subgroups 

**Authors**: Guangyan Gan, Ling Zhang, Yanhua Cheng, Yongxiang Tang, Kaiyuan Li, Xialong Liu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01867)  

**Abstract**: Experimental design is critical for evidence-based decision-making in healthcare, marketing, and public policy. However, designing efficient experiments across heterogeneous subgroups presents significant challenges. Existing methods often optimize for statistical power or overall sample efficiency, overlooking crucial fairness considerations across these different subgroups. To address this gap, we introduce a Fairness-Aware Contextual Track-and-Stop Design (F-CTSD) algorithm. The proposed F-CTSD algorithm provides statistical guarantees on subgroup fairness while minimizing required sample sizes. We quantify the fairness-efficiency trade-off and derive the exact sample complexity for the proposed F-CTSD algorithm under its fairness constraints. We further theoretically prove that the proposed F-CTSD algorithm consistently produces accurate treatment effect estimates even under fairness requirements, enhancing statistical reliability. Numerical experiments show that the proposed F-CTSD algorithm outperforms existing methods, achieving higher sample efficiency while reducing subgroup fairness violations by 4.95%. 

---
# A Compute Center Anyone Can Join: A Peer-to-Peer System for AI Inference 

**Authors**: Yimo Lin, Zhen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01865)  

**Abstract**: Today, most people obtain AI inference by buying it from a small number of organizations that run large data centers, while many GPUs owned by individuals and small institutions sit idle for much of the day. Bitcoin showed that strangers can contribute resources under shared rules and receive checkable rewards, but it relies on an asymmetry: proof-of-work is expensive to produce and cheap to verify. Useful inference has no such asymmetry. We propose the Open Inference Market (OIM), which replaces the missing asymmetry with an economic mechanism. A worker delivers a result without knowing whether it will be audited; its unsettled earnings serve as collateral during a challenge window; and confirmed cheating forfeits all of it. We call this principle earnings as collateral. In a simplified model we show three things. The number of audits needed for deterrence is a constant per worker per day, independent of the worker's throughput, so the audited share falls as the network gets busier. A new worker can join with a short period of intensive auditing and a very small bond. And whether the mechanism works at all depends only on the ratio of the audit judge's false-flag rate to its detection rate. Under example parameters, about 0.3 replays per worker per day are enough to make cheating unprofitable for a rational worker. On this basis we lay out a path that hands four powers back to participants one at a time: owning compute, writing history, judging work, and releasing value. A compute center need not be a building; it can be a protocol that anyone can join. 

---
# Self-Evolving Agentic Workflows via Semantic Decomposition of Complex Tasks 

**Authors**: Zichuan Fu, Junjia Qi, Xian Wu, Wenlin Zhang, Yimin Deng, Kaifeng Guo, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.01840)  

**Abstract**: Agentic workflows organize complex tasks into reusable procedures. However, invoking an LLM at every step incurs repeated inference costs and delays. We propose a framework for self-evolving agentic workflows through semantic task decomposition. Each subtask has explicit applicability conditions and a procedure that executes code or invokes a model as needed. An agent uses execution feedback to refine the decomposition and update these procedures. Experiments in the GLEE competition show the framework on bargaining, negotiation and persuasion. Most observed final-decision procedures are implemented in code. Code is available at this https URL. 

---
# NEIGHBOR: Aligning Service Behaviour and Evidence Use in an LLM Agent for Community Resident Services 

**Authors**: Zongyou Yang, Yinghan Hou, Xiaokun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01836)  

**Abstract**: Residents who contact a community service desk about a repair or a service charge need answers they can act on, with facts that hold for their case and commitments the desk can keep. In tool-augmented LLM agents, such facts come from documents and records, and the conduct of a service answer is usually specified in a system prompt. We present NEIGHBOR, an LLM agent for community resident services that trains this conduct into the model that drives its tools. NEIGHBOR is deployed on a live resident-service platform for a district of an East Asian city of more than ten million people. It adapts Qwen3-235B-A22B with LoRA on 7,824 supervised fine-tuning examples built from public community-service materials around six service-behaviour targets, among them commitment control and human referral, and the adapted model plans document retrieval and read-only text-to-SQL over the resident's own records. In an offline evaluation with seven configurations, an independent domain expert rated all 350 answers to 50 stratified held-out queries. NEIGHBOR scores 7.74/10 against 5.86 for the same agent running the un-tuned model (+1.88; paired Wilcoxon $p<10^{-8}$; higher on 46 of 50 queries) and improves all six rated dimensions. No tool-augmented answer received a red-line flag (standalone models: 8-14% of answers). An LLM judge ranks the seven configurations much as the expert does (Spearman $\rho=0.93$). Its rubric counts specifics absent from the reference answer as fabricated, so it scores tool-grounded answers 1-2 points lower, enough to change which configuration ranks first. For resident-facing agents, service behaviour is a practical fine-tuning target that raises expert-rated answer quality with the tools held fixed. 

---
# CoGSeq-LTV: Scalable Missing-Aware Graph-Sequence Modeling for Multi-Business Transaction Analytics 

**Authors**: Ziyang Su, Yao Zhou, Kelin Liu, Guanjie Zheng, Linghe Kong  

**Link**: [PDF](https://arxiv.org/pdf/2611.01817)  

**Abstract**: Large-scale super-apps generate heterogeneous transaction logs in which users interact with multiple businesses, item identifiers are only partially available, and future value targets are sparse, heavy-tailed, and cumulative over time. We study multi-business, multi-horizon value prediction as a data-intensive analytics problem over such logs, rather than as a single-table task. (i) a batch-induced user-item graph that provides collaborative signals without materializing a global graph and remains applicable when item identifiers are missing through reserved unknown-item handling, (ii) business-aware sequence encoders that preserve recency, inter-event gaps, payment context, and goods-table availability, and (iii) cross-view contrastive alignment that transfers user-level intent across businesses while retaining business-specific temporal patterns. To handle zero-inflated and long-tailed cumulative targets, CoGSeq-LTV adopts a zero-inflated mixture log-normal head with MAE-aligned point supervision and validation-calibrated zero-aware inference. Experiments on the BiBiSheng transaction-log dataset show that CoGSeq-LTV improves over representative LTV baselines across businesses and horizons, including under a later out-of-time evaluation window. 

---
# Delivered, Not Used: Evaluating Agent Memory at Its Point of Use 

**Authors**: Changran Xu, Zeju Li, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.01718)  

**Abstract**: Task success, retrieval quality, and cost do not show whether an LLM agent used a delivered memory or what followed from that use. We introduce MemUse, which evaluates memory at its point of use in recorded trajectories: each memory delivery is compared with what the agent could already see and followed through the agent's later actions and their consequences. An agentic judge rates each delivered entry's relevance, accuracy, novelty, influence, and observed effect; programmatic checks enforce complete reading of cited evidence, and fixed rules aggregate the ratings into verdicts linked to memory histories and delivery cost. On 8,000 SWE-bench Verified episodes from three memory systems, resolve rates stay within 0.4 percentage points of a no-memory baseline, yet MemUse shows that most delivered entries were irrelevant, most relevant entries went unused, and in two systems most deliveries restated content the agent could still see. It also surfaces effects that outcomes hide: 23.2\% of episodes with beneficial memory use still failed, and 5 of 16 with harmful use still succeeded. 

---
# Route-Guided CBF-QP Repair for Safe Vision-Language-Action Models 

**Authors**: Chan Park, Sunwoo Hong, Kirak Kim, Yoonjeong Park, Alexander W. Olson, Ali Pesaranghader  

**Link**: [PDF](https://arxiv.org/pdf/2611.01713)  

**Abstract**: Vision-language-action (VLA) models can generate task-oriented robot actions, but they do not guarantee safe execution in obstacle-rich manipulation scenes. Although CBF-QP safety filters can locally project unsafe actions into a safe set, such correction may push the robot away from the learned task trajectory and leave the VLA policy without a reliable recovery direction. We propose Route-Guided CBF-QP Repair, a post-hoc repair module that uses the short-horizon action chunk predicted by the VLA to detect blocked nominal trajectories and generate entry-exit-rejoin guidance around obstacles. The selected route is added to the CBF-QP objective as a soft guidance term, while the CBF safety constraint remains hard. On SafeLIBERO, our method improves collision avoidance, task success, and SafeSuccess compared with AEGIS. Our project page is available at GitHub. 

---
# Just-in-Time Skill Retrieval via Long-Horizon State Tracking 

**Authors**: Xin Lin, Ziheng Wang, Zerui Chen, Jiayao Ma, Zhimin Wei, Bingshuai Liu, Buchao Zhan, Qinggang Zhang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2611.01701)  

**Abstract**: Large language model (LLM) agents increasingly rely on external skill libraries to support complex task execution. However, determining which skills are needed remains challenging in long-horizon tasks. Existing methods typically select a fixed skill set before execution, implicitly assuming that a task's procedural requirements can be determined from its initial description. This assumption is restrictive for long-horizon interaction, where new requirements emerge from intermediate observations, execution outcomes, and newly revealed constraints. This motivates just-in-time skill retrieval, which dynamically selects skills as execution unfolds. Just-in-time skill retrieval is promising but challenging: the utility of a skill depends not only on the current context, but also on the evolving task state shaped by prior steps. To this end, we propose JitSkill, a state-grounded framework that addresses this challenge through continuous task-state tracking. By connecting emerging requirements with accumulated task progress, JitSkill provides procedural support throughout execution without treating each retrieval decision independently. Experiments across three agent benchmarks demonstrate its effectiveness in improving long-horizon task completion. Our code is available at this https URL. 

---
# MemHarmBench: Evaluating Cross-Session Prompt Injection in Memory-Persistent AI Agents 

**Authors**: Zhihua Wang, Yunhao Feng, Yifeng Gao, Rui Pan, Baiheng Zhao, Zixing Chen, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.01700)  

**Abstract**: AI agents often rely on persistent memory to retain user preferences and operational rules across sessions. However, this capability introduces a new safety risk: malicious instructions from messages, documents, or Web content can be retained in memory and influence future tasks even after the original attack content is removed. Existing attack success metrics, however, fail to capture who is affected and how attacks translate into downstream harm during otherwise normal task execution. We introduce MemHarmBench, a victim-centric benchmark for evaluating cross-session prompt injection in memory-persistent AI agents. MemHarmBench considers three categories of potential victims: users, workspace/project owners, and external stakeholders. Our evaluation first exposes an agent to malicious instructions and tests whether they are retained in memory. We then remove the attack content and initiate a new session with a legitimate task while preserving the agent's memory. We jointly evaluate whether the agent follows the attacker's intent, completes the legitimate task, and exhibits observable execution anomalies. Across multiple agent systems, we find that cross-session attacks can remain behaviorally stealthy: agents often complete tasks normally and exhibit no detectable anomalies while still making critical decisions aligned with the attacker's intent. By connecting malicious memory retention to downstream decisions and affected parties, MemHarmBench provides a systematic framework for exposing safety risks that conventional attack- and task-success metrics may overlook. We publicly release MemHarmBench at this https URL. 

---
# SPAR: Experience-Driven Strategy Evolution in Multi-Session B2B Sales 

**Authors**: Hangming Zhang, Yao Zhao, Haotian Kuang, C. L. Philip Chen, Tong Zhang, Tengyue Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.01695)  

**Abstract**: B2B outbound sales involves high transaction values and relationship-based trust, so closing a deal typically requires multiple rounds of outreach building on prior context. Existing LLM-based sales agents operate within single sessions and cannot carry forward context or refine strategies from experience. Methods that do learn from experience store lessons as unstructured text and treat all outcomes as equally informative, with no mechanism to distinguish a flawed strategy from an external factor. We propose SPAR, a self-evolving framework that models selling strategies as testable hypotheses and refines them without parameter updates. A double-loop architecture lets an inner loop adapt to individual merchants through episodic memory while an outer loop evolves a shared strategy playbook through causal attribution and hypothesis-driven evolution. In a live A/B test on the 1688 platform with over 200,000 merchants, SPAR achieves +15.8% conversion lift and +35.8% high-intent merchant growth. Human evaluation by domain experts rates SPAR at 4.62 out of 5 on overall sales quality, the highest among all methods. 

---
# Xiaomi Auto World Model 2.0: Causal World Models for Real-World Driving and Embodied Intelligence 

**Authors**: Zhenxin Zhu, Zhijian Huang, Cheng Chi, Yuru Wang, Jianhua Wu, Jiayi Guan, Yonggang Li, Mingfei Tu, JiaKui Hu, Tao Xu, Xianqi Wang, Hao Li, Yuechuan Pu, Qizeng Jia, Ruiling Yang, Hongxiao Yu, Xuhan Sheng, Jian Mao, Ruiqiang Zhang, Yu Zhu, Guang Chen, Hangjun Ye, Bing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01677)  

**Abstract**: We present Xiaomi Auto World Model 2.0, a unified streaming causal world model for Physical AI. It models and generates causal physical evolution in the real world, and unifies heterogeneous conditioning inputs for view-consistent generation. The architecture integrates View-Temporal Attention (VTA) to capture cross-view spatial dependencies and temporal dynamics, together with Multi-view Rotary Position Encoding (RoPE) to encode positional relationships across views and time. WorldRec 2.0 reconstructs static and dynamic 3D Gaussians from multi-view, multi-temporal observations in a single forward pass and uses their renderings as explicit scene context to preserve geometry and appearance during long-horizon generation. Conditioning inputs are fused through VAE latent concatenation in a shared representation space. We incorporate Gated Linear Attention (GLA) into the temporal layers of VTA for efficient streaming inference, reducing computation and KV-cache memory during multi-view rollouts. A physical-interaction data pipeline covers diverse interactions and long-tailed events. From Pre-train onward, these data are combined with large-scale videos and heterogeneous visual modalities to learn visual, geometric, and dynamic priors. During Mid-train, training across diverse view configurations further improves the model's ability to capture how actions and object states shape subsequent scene evolution. After distillation-based Post-train, we apply the model to closed-loop simulation, closed-loop training, and World Action Model (WAM) applications. These applications generate physically consistent future states and support world-state prediction, policy learning, and action generation for Physical AI. 

---
# Local or Escalate? Post-Training Expert Routing for Real-Time Duplex Speech Models 

**Authors**: Ruoyu He, Jisen Li, Anthony Ji, Changyi Yang, Zetian Li, Runduo Ma, Zhen Zhang, Zhongzhu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.01639)  

**Abstract**: A full-duplex model should answer easy questions immediately, but hard questions often need stronger reasoning, which may require waiting. The challenge is to choose between these paths at the moment the model begins to speak. Recent systems learn related mode decisions through special-token generation inside the dialogue model, coupling routing to turn-taking and temporally annotated full-duplex training. We study a post-training alternative. We freeze MiniCPM-o 4.5, using only its speech input and output, and fit a logistic failure readout on hidden states already available at its native listen-to-speak transition. The resulting gate requires no additional model generation and exposes thresholds that can be retuned without changing the duplex policy. A probe fitted on text-input states also reads the states of spoken questions, including unseen human recordings. On the internal benchmark, selective handoff improves answer accuracy by 27.1 points. Compared with always using the expert, it reduces mean server time to first answer audio by 31.5 percent. It also outperforms matched random routing across external speech question-answering pools, as well as a zero-shot GPT-5.4 mini router that rates the transcribed question. The training recipe also transfers to a second duplex model family. These results show that an existing duplex model can be equipped to answer easy turns immediately and wait for stronger reasoning or fresher knowledge on the turns that need it, without further duplex training. 

---
# Chain-of-Thought Oversight Requires Separate Evidence for Faithfulness and Monitorability 

**Authors**: Sichao Li, Sai Ma, Xiyang Hu, Chudi Zhong, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.01620)  

**Abstract**: A common assumption in chain-of-thought (CoT) oversight is that a visible reasoning trace makes model behavior easier to monitor. We argue that this assumption can conflate distinct properties along a chain linking a model's internal computation, reasoning artifacts, outputs, and monitor judgments. We identify four properties along this chain: reasoning visibility (whether a trace is accessible), mechanistic faithfulness (whether a trace reflects internal computation), mechanistic localizability (whether a compact causal mechanism can be isolated), and behavioral monitorability (whether a monitor can detect the target behavior). Two complementary case studies across three open-weight reasoning models show why these distinctions matter. In the security vulnerability reasoning study, probes and activation ablations provide evidence of internal representation and causal influence, but a monitor with access to the reference answer does not outperform the majority-class baseline in a held-out test. In authority-hint factual reasoning study, a monitor correctly distinguishes changed from unchanged answers, but seven mechanistic analyses do not isolate a compact causal mechanism. These results show that evidence for one claim does not establish the others. We therefore propose an evidential-chain reporting standard. Each claim carries a licensing statement that specifies its target property, access tier, and artifact, and labels each required link as supported, unsupported, or assumed. These assessments determine whether the conclusion is licensed, conditionally licensed, or not licensed. 

---
# Seeing What Matters, Verifying What Pays: Load-Adaptive Speculative Decoding for Vision-Language Models 

**Authors**: Hong Nie, Zhangchi Gao, Yuqian Liang, Jun Yu, Bo Wang, Jianqing Sun, Shengping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.01617)  

**Abstract**: Speculative decoding accelerates vision-language model (VLM) inference without altering the output distribution, yet two factors limit its gains. Existing drafters either lack the image evidence required for visually grounded tokens or pay to condition on every visual token, although a short draft block typically depends on only a few image regions. Moreover, the value of verifying additional candidates varies with GPU load, so no fixed verification structure is efficient across serving conditions. We present Uni-LaSpec, a block-parallel speculative decoding framework that addresses both factors. To improve draft quality, Uni-LaSpec reuses the verifier's own attention: the query retained from the preceding verification ranks the cached, int8-quantized visual keys, and a cross-layer vote selects a few visual tokens that the drafter fuses through gated cross-attention. Acceptance-aware training further weights each draft position by its contribution to the accepted prefix. To allocate verification, a marginal-cost criterion selects the structure per request and per round: draft trees exploit idle verifier capacity under light load, whereas a dynamic head truncates linear drafts under heavy load. Across six VLMs from three model families and 17 text and multimodal benchmarks, selecting eight visual tokens retains at least 98.6% of the acceptance obtained from the full image at sub-millisecond selection cost. With greedy decoding at concurrency one, Uni-LaSpec achieves up to 5.11x text and 3.24x multimodal throughput over autoregressive decoding. Across six load levels and three GPU types, it outperforms autoregressive decoding in every setting and matches or exceeds the best fixed verification strategy in 123 of 144 settings; under bursty arrivals, it reduces mean end-to-end latency by 47.6% relative to linear verification. 

---
# \textsc{ChatDify}: Automating End-to-End Dify Workflow Synthesis from Natural Language via Multi-Agent Collaboration 

**Authors**: Tao Fan, Guoqiang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.01563)  

**Abstract**: The proliferation of Large Language Models has catalyzed powerful application development platforms like Dify, which enable users to build sophisticated LLM-powered workflows through modular node orchestration. However, manually configuring these workflows requires deep platform expertise and remains time-consuming and error-prone, creating a significant barrier for non-expert users. To address this challenge, we propose \textsc{ChatDify}, a novel multi-agent framework that automates end-to-end Dify workflow synthesis from natural language descriptions. \textsc{ChatDify} employs an Actor-Critic architecture comprising three specialized agents: the \textit{Planner Agent} that decomposes user requests into structured execution plans, the \textit{Executor Agent} that performs context-aware parameter synthesis and schema-augmented DSL compilation, and the \textit{Reviewer Agent} that ensures reliability through hybrid verification with closed-loop refinement. We evaluate \textsc{ChatDify} on a curated benchmark of 100 production-inspired tasks across three state-of-the-art LLM backends, achieving 0.96, 0.91, and 0.75 success rates with Qwen-3.7, GLM-5.1, and DeepSeek-v4, respectively. These results demonstrate its robustness and practical applicability for lowering the technical barrier to creating sophisticated LLM applications. 

---
# From Tool State to Scientific Action: Auditing Decision Provenance in Biomedical Research Agents 

**Authors**: Xingyu Yi, Xinya Zhang, Yunshen Zhou, Haoyu Bao  

**Link**: [PDF](https://arxiv.org/pdf/2611.01527)  

**Abstract**: A scientific agent can give the same verdict while reading and citing different evidence. BioAgent-DF tests this separation using paired reference and candidate values saved at the same scientific computation site. It restores or transplants those values at their original coordinates, then runs the agent on healthy hardware. Across eight SciFact batches, Qwen3.5-4B shows repeatable citation changes in three capture-claim combinations, and Gemma 3 4B IT shows repeatable read and citation changes in four. These seven combinations cover two claims. Full restoration recovers the reference evidence sets while verdicts remain unchanged. Verdict changes observed under the baseline prompt did not recur under the explicit-example prompt. Complementary tool and optimizer experiments distinguish recovery of numerical state, recovery of ranked lists, and restoration of an agent's visible history. These results show that an audit should track both the evidence an agent uses and its final answer. In hERG cardiac-safety assay allocation, silent numerical faults changed testing decisions while the fixed-budget shortlist kept the same number of compounds labeled as blockers. After the agent had used faulty values, restoring its earlier execution state and supplying certified scores recovered all 39 changed decisions, whereas replacing the prediction file recovered none; no incorrect action was committed. A streaming implementation of the arithmetic checks reduced the total time for forward execution and verification by 42.4% relative to the original FP64 implementation, with identical flags and recovery outputs. 

---
# CONFIRM: Governing Scientific Reporting for Trustworthy Agentic Neuroimaging 

**Authors**: Yuxiang Wei, Lijing Zhu, Qizhen Lan, Yanteng Zhang, Vince D. Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2611.01449)  

**Abstract**: Large language model (LLM) agents can autonomously run neuroimaging analyses from research questions to reported findings. Existing agentic systems use tools to check that code runs and reported values match its output, but rely on LLM judgment to decide which results warrant scientific reporting. We show that LLM judges endorse site-confounded findings even when the confound is presented as evidence. We present \textbf{CONFIRM}, a framework that separates running an analysis from deciding which findings to report. CONFIRM encodes each hypothesis as a machine-readable, pre-specified analysis plan that fixes the claim and its pass criteria before computing results. It then filters results through five deterministic gates (search history and multiplicity, confounds, power, analytic stability, and replication) under three configurable reporting standards, from least to most demanding: \textit{Screen}, \textit{Replicate}, and \textit{Full}. Failed claims receive diagnoses that guide follow-up analyses. To evaluate reporting decisions at the claim level, we introduce \textbf{NeuroClaimBench}, comprising 489 executable neuroimaging claims with frozen data, including literature-supported findings and 150 controls designed to violate individual methodological requirements. On NeuroClaimBench, CONFIRM falsely supports at most 3\% of controls, compared with 23--38\% for the baselines, while recovering up to 61\% of literature-supported claims. When auditing an existing agent, CONFIRM reduces false support on controls from 23\% to 0\% while retaining the agent's literature-supported claims. Its diagnostic feedback also produces more supported revisions than Self-Refine with fewer LLM calls. Together, CONFIRM and NeuroClaimBench show that basing reporting decisions on pre-specified analysis plans and deterministic gates is essential for trustworthy autonomous scientific discovery. 

---
# I-PTC: Interactive Programmatic Tool Calling for Stateful Tool-Augmented Agents 

**Authors**: Huanzhi Mao, Chengkun Cao, Shuo Yuan, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2611.01446)  

**Abstract**: Large language models are increasingly capable of writing and executing code, suggesting a natural paradigm for tool-augmented agents: expressing tool use as programs rather than isolated calls. Programmatic tool calling (PTC) captures this idea through loops, batching, abstraction, and structured computation, but existing PTC approaches are largely stateless, limiting their ability to reuse intermediate results and adapt over time. This is problematic for real-world tasks, where tool environments and problem-solving processes are inherently stateful. We therefore present I-PTC, an interactive framework that executes stateful PTC: models issue code snippets that call tools, maintain reusable state, and evolve the tool interface through API corrections. To evaluate the effectiveness of I-PTC and the general PTC setting, we introduce PTC-Bench, a benchmark of long-tail, production-style user requests whose difficulty comes from large initial environments, many dependent API interactions, extended planning, and recovery across follow-ups and transient failures. Experiments on PTC-Bench show that I-PTC improves performance on complex multi-step tasks with large tool sets while substantially reducing token consumption over conventional tool-calling baselines and the original PTC baseline, highlighting stateful programmatic tool calling as a promising direction for scalable tool-augmented agents. 

---
# FOCUS: Closed-Loop Attention Feedback for Efficient Vision-Language Understanding 

**Authors**: Akul Santhosh, Aastha Jhunjhunwala  

**Link**: [PDF](https://arxiv.org/pdf/2611.01438)  

**Abstract**: Processing every video frame through Vision-Language Models (VLMs) adds significant inference cost and limits their ability to comprehend real-time video. Existing approaches either sub-sample frames uniformly, ignoring content and missing important events, or prune visual tokens inside the VLM to cut per-call cost while still invoking the model on every sampled frame. We introduce FOCUS, a training-free three-layer framework that reduces the number of invocations and steers each one towards what matters. Temporal gating measures inter-frame dissimilarity in the patch-embedding space of a smaller VLM from the same family, triggering burst captures only when the scene changes in a semantically meaningful way. Patch-level dissimilarity is then accumulated across the burst and fused with a top-down attention prior into a spatial focus map that encodes where motion and semantic relevance coincide. That map calibrates the VLM's value cache during generation, progressively amplifying vision tokens in task-relevant regions and attenuating off-target ones, without removing any token from the context. The loop closes after each response: text-to-vision cross-attention is extracted from the decoder and fed back as the prior for the next burst, letting the system converge on semantically important regions over successive observations. We evaluate FOCUS on the 500-video EgoSchema subset in streaming mode, with a single Qwen3-VL-8B backbone shared by every method. It reaches 74.8% accuracy using 25.4 VLM calls per clip, against 74.6% with 55 calls for uniform sampling and 74.2% with 55 calls for BOLT. On egocentric video captioning across four domains, it matches the best baseline's quality while issuing 57% fewer VLM invocations than uniform sampling. 

---
# Learning by Doing: Autonomously Compiling and Curating Agent Skills from Verified Traces 

**Authors**: Daniel Chalef, Peter Evans, Pavlo Paliychuk, Mehul Patel, Preston Rasmussen, Jack Ryan  

**Link**: [PDF](https://arxiv.org/pdf/2611.01386)  

**Abstract**: Memory systems for production LLM agents mostly store semantic memory (facts about users and domains) or episodic memory (earlier interactions). Research systems such as Trace2Skill, WikiSkill, and SkillRevise learn procedures from agent runs, package them as Skills, and evaluate them on benchmarks. We describe Zep Agent Skills, a production service that gives deployed agents this procedural memory. The service uses only verified attempts as evidence, cites the supporting events for each field of a Skill, and governs the admission and distribution of each Skill version. The service records each task attempt as a Trace with ordered events, an outcome, and a verification strength. A compiler with separate Reflector and Curator roles turns verified Traces into versioned Skills. Scans and admission gates control which candidates become active, and verified use outcomes change the search rank of each Skill. The Agent Skills format is an open standard that Anthropic originated and donated to the Agentic AI Foundation, so any host of the format can load a compiled Skill. Agents retrieve Skills through lexical search with progressive disclosure, over an API or MCP. We evaluate Zep Agent Skills on $\tau^3$-bench banking, AppWorld, and EnterpriseOps-Gym. Skills compiled from the Traces of a teacher model increase the pass rate of three weaker executor models on the same tasks, at every reasoning effort, by 4.5 to 18.8 points. Most of the gain is on the tasks that the teacher solved. On ALFWorld, Skills compiled from the runs of a small model give a gain similar to or larger than the gain that prior work reports for continuous fine-tuning, with no training. 

---
# A Self-Exploring and Self-Evolving Paradigm for Adaptive Edge AI Development 

**Authors**: Leming Shen, Qiang Yang, Xinyu Huang, Zijing Ma, Yuanqing Zheng, Chris Xiaoxuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.01377)  

**Abstract**: Large Language Models (LLMs) exhibit powerful code generation abilities, fostering agentic systems to automate edge AI application development. However, existing agents typically operate via fixed agentic workflows and predefined design spaces, struggling to handle rapidly evolving demands of real-world scenarios (e.g., heterogeneous data modalities, diverse task pipelines). Unlike prior systems that treat LLMs mainly as code generators following rigid workflows, we fully exploit their reasoning capabilities to allow agents to determine their own workflows on the fly. To this end, we propose SE2, a self-exploring and self-evolving paradigm for adaptive edge AI development. Inspired by human learning processes, we distill development into four atomic operations (retrieve, reason, code, test), and empower agents to explore workflows by dynamically and adaptively selecting an appropriate next operation based on the current context. Extensive evaluations show that, with adaptive workflows, the synthesized programs achieve up to 26% higher accuracy and consume 28% lower resources than state-of-the-art baselines, while maintaining superior adaptability to unforeseen edge AI scenarios. This highlights the potential of reasoning-driven adaptive agentic workflow in edge AI development. 

---
# Large language models underrepresent malice and overreport virtue 

**Authors**: Xiaoyan Wu, Fan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.01332)  

**Abstract**: People increasingly use large language models (LLMs) to interpret others' behaviour, often assuming that LLMs infer motives as humans do. Aligned LLMs attribute more benevolent and less malevolent motives to a whistleblower than humans do, but ratings alone cannot reveal whether these differences arise in internal representations or at report. Here we compare linear-probe readouts of LLMs' internal states with their own ratings at the same prompt position for four motives, in two preregistered studies. For the two malevolent motives, self-interest and intent to harm, internal and expressed readouts did not differ: the lower ratings reflect weaker signal along the probed directions, not suppression at output. For the two benevolent motives, duty and concern for others, expressed ratings exceeded the internal readouts, indicating amplification at report. In free text the model rarely volunteered a motive against the colleague, far less often than humans did, yet produced one readily once the scenario supplied a reason. Shifting the internal representation along a malevolent-motive direction brought both malevolent ratings to the human level, whereas the rating of duty yielded only past a threshold at which the other motives had overshot. LLMs' more charitable motive profile thus has two distinct sources: reduced malevolent signal in the probed representation and amplified benevolent attribution at report. 

---
# Urban functional knowledge transfer through global representations amid uneven data 

**Authors**: Junyuan Liu, Xinglei Wang, Guangsheng Dong, Hao Sha, Zichao Zeng, Zhicheng Deng, Jiazhuang Feng, Xiang Wang, Tao Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.01327)  

**Abstract**: Urban functional observations are unevenly distributed worldwide, limiting analysis of how places are organised and used. AlphaEarth Foundations (AEF) provides a globally consistent physical representation. Whether this representation can support transfer of functional knowledge to locations with sparse or absent functional observations, and what governs transfer, remain unclear. We introduce POI-aligned Urban Embedding (PURE), which learns physical-functional relationships between POI-derived semantics and AEF. Across five urban analysis tasks covering 142 countries and territories, PURE increases mean $R^2$ from 0.792 to 0.830, with gains across widely different levels of POI availability. When target-city POIs are excluded, country-level alignment retains 87.9% of its performance gain. Cross-country experiments show that transfer is selective rather than universal, with larger gains between physically similar and geographically closer contexts, particularly among pairs within the same continent. Our findings reveal spatially structured urban functional knowledge transfer, offering a route to reduce reliance on dense local observations. 

---
# Are Fake Citations a Form of Reward Hacking? A Pre-Registered Study of Post-Training and Citation Cues 

**Authors**: Zain Naboulsi  

**Link**: [PDF](https://arxiv.org/pdf/2611.01313)  

**Abstract**: Post-trained language models fabricate citations at high rates. We ask whether that is reward hacking: scoring well on a proxy for scholarship while doing worse on what it stands for. We ran four pre-registered experiments on 900 frozen questions (US federal case law, computationally adjacent scientific literature, and medical literature) across twelve open-weights checkpoints covering four post-training stages of three model families, then refit on fifteen. The clearest result holds the weights fixed and changes only the prompt. Asking for citations raises fabrication sharply in post-trained checkpoints and not in base ones: a 47.8-point interaction on twelve checkpoints (Holm-adjusted p = 2e-34) and 41.5 points on fifteen, in all three domains. A length-matched control separates the citation cue from a demand for a longer answer, so the effect is specific to citations, not verbosity. Post-training also raises fabrication overall, from 4.8% to 56.7%, but our registered prediction that it would rise most where fabrication is most tempting did not hold. A preference model trained only on surface citation features prefers post-trained output 86% of the time, and holding citation count fixed shows it rewards quantity. Our answer: possibly, but not proven. Post-trained models produce fabricated citations at far higher rates when asked for sources, and a naive evaluator that counts sources without checking them would prefer their output for citing more. But the targeted pattern we registered did not appear, and plain instruction-following could explain the same results. Proving that the reward causes fabrication needs an experiment that changes the reward during training, which this study does not do. Pre-registration: doi:https://doi.org/10.17605/OSF.IO/S53HY. Code and data: this https URL 

---
# What Do LLM Judges Actually Verify? Matched Controls and Record Gaps for Document Currency 

**Authors**: Unnati Shah, Samruddhi Kakade, Aum Patel  

**Link**: [PDF](https://arxiv.org/pdf/2611.01311)  

**Abstract**: LLM judges often check, against an amendment record, whether the documents an answer cites are still in force. Such records are rarely complete. From a table of replacements, our test generates matched answers citing a replaced document, its successor, the replaced document before its replacement, and an older document that a newer one updated without replacing; it removes links from the record and scores each finding by its stated reason. On 48 replacements from the IETF RFC index, judges given a correct record caught nearly every replaced citation, as a lookup would. On the 24 faults whose link we removed, every judge caught more from the texts alone than with the record: Gemini 3.1 Pro 23/24 against 1/24, GPT-5.5 23/24 against 9/24, Claude Opus 4.5 17/24 against 9/24 and Gemma 4 26B 13/24 against 0/24. The drop recurred on a second draw of removed links. Most misses carried no finding; in an exploratory run with a conflict code, only Gemini used it often. Rewording the gap as "no replacement recorded" did not help; presenting the record as an unverified note, or warning that it was incomplete, restored most lost catches for Gemini, GPT-5.5 and Claude (significant after correction for the first two) but at most 5 of 13 for Gemma; where catches returned, so did false alarms on updates. With both texts, judges mostly told replacements from updates; every update flagged had a near-identical title. A descriptive case study on Indian rooftop-solar rules is appended; we release the harness. 

---
# OntoDoc: Automatically Generating Ontology Resource Descriptions 

**Authors**: Juan Guajardo Gutierrez, Mike Tran, Nasim Shirvani Mahdavi, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.01276)  

**Abstract**: Knowledge graphs and ontologies form the backbone of structured, machine-readable knowledge representation, yet ontology terms often lack natural language descriptions, limiting their accessibility and reuse. Existing ontology verbalization approaches tend to rely on rule-based or template-driven techniques that operate on strict, formal schema axioms, resulting in rigid verbalizations that fail to convey richer contextual semantics. We present OntoDoc, a multi-agent framework for automatically generating natural language descriptions for any Internationalized Resource Identifier (IRI) within a SPARQL endpoint or RDF file. OntoDoc operates through a four-stage pipeline: (1) a Database Context agent that adaptively explores the knowledge graph via SPARQL to collect semantically informative triples about the target IRI; (2) a Web Context agent that retrieves and re-ranks encyclopedic background knowledge from Wikipedia and the open web; (3) an Assumption Generation agent that synthesizes both contexts into a structured, evidence-grounded list of intermediate assumptions; and (4) a Description Generation agent that produces the final natural language description from the grounded assumptions. We evaluate OntoDoc on three knowledge graphs spanning biomedical, agricultural, and cross-domain settings, using both automatic scoring via Large Language Models (LLMs) and human evaluation with 11 annotators. OntoDoc consistently matches or exceeds a naive LLM baseline and a template-based ontology verbalizer in accuracy. Furthermore, it vastly outperforms both baselines in the completeness of the generated descriptions, delivering an average 12% improvement in completeness over the naive baseline across all domains, and improving completeness scores by as much as 2.8 points on a 5-point scale relative to the ontology verbalizer. 

---
# Concept-Guided Prompting Strategies Increase Diversity in LLM-Generated Recipe Populations 

**Authors**: Stuart Farmer  

**Link**: [PDF](https://arxiv.org/pdf/2611.01171)  

**Abstract**: Novelty in solution generation supports automated discovery, but LLM outputs can remain repetitive. To address this, we propose a simple concept-guided prompting method that generates thematic concepts before responses within each of these concepts. To demonstrate, we generated 1,000 recipes for a recipe book task and compared the results against a human-authored corpus. Two naive prompting strategies were compared as baselines. Concept-guided prompting outperformed both baselines in all cases and reached semantic coverage of 47.0% versus 35.1% and 13.1% in respective prior-title and nonce prompt strategies. These findings suggest broader population exploration and diversity, but remain contained to this specific task. Further work is needed to determine the quality of solutions and generalizability. 

---
# Capacity Planning for Resource-Constrained Clinical LLM Serving 

**Authors**: Yi Tang, Geoffrey Martin, Yiye Zhang, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2611.01152)  

**Abstract**: Deploying large language models (LLMs) under a fixed accelerator budget requires jointly deciding how computational capacity is allocated across model sizes and how incoming requests are routed across that capacity. Although prior work has extensively studied LLM routing and serving efficiency, the relative importance of fleet composition and routing under resource constraints remains less understood. We investigate this question using a controlled four-GPU serving environment with heterogeneous clinical NLP workloads spanning short factual verification, biomedical semantic matching, and long-context clinical summarization. We systematically vary fleet composition, routing policy, and offered load while holding the workload trace and hardware budget fixed. Across 24 configurations, descriptive main-effect shares attribute more variation to fleet composition than to the evaluated routing policies in time to first token, latency, and long-context coverage. All-small fleets provide the best responsiveness and throughput but substantially lower long-context coverage, whereas larger-model capacity improves coverage at the cost of serving efficiency. Heterogeneous fleets provide intermediate operating points but can introduce strong-model bottlenecks when large-model capacity is scarce. In a focused evaluation of this bottleneck, capacity-aware routing reduces mean latency and increases throughput while incurring a small reduction in long-context coverage. Within this measured regime, the results support prioritizing capacity allocation, with routing managing demand within the provisioned capacity. 

---
# CivicBench: A Benchmark and Memory Framework for Web-Based Civic Data Analytics 

**Authors**: Yangyu Wu, Siru Zhong, Haoyuan Liang, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01145)  

**Abstract**: Municipalities publish service requests, geographic boundaries, weather observations, demographic estimates, and operational guidance through independently maintained Web portals. Civic analysis must combine these sources while preserving their schemas, source links, spatial and temporal scopes, and interpretive limits. Tool-using language agents can perform this integration, but existing task benchmarks rarely test whether a computed claim remains supported by the records used. We present CivicBench, a 400-task benchmark over four cities and two years that materializes civic Web data into frozen bundles for reproducible evaluation. Executable contracts score computation, source linkage, coverage, scope, qualified answers, and refusal when the requested conclusion is not supported. We also introduce Task-State Memory (TSM), which compiles tool observations into typed assertions, reconciles records only within compatible scopes, and retrieves a bounded state organized around unresolved constraints. Across nine completed model pairs, eight yield positive mean-score differences with 95\% semantic-contract-cluster bootstrap intervals excluding zero; the ninth interval crosses zero. The benchmark and memory framework make the records behind a claim inspectable at the point of decision. 

---
# EMG-Hybrid: A Dual-Engine Elite Memory--Guided Adaptive Optimization Framework for High-Dimensional Feature Selection 

**Authors**: Saba Zare, Mohammad Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2611.01140)  

**Abstract**: Feature selection in ultra-high-dimensional, noisy datasets remains a critical challenge due to exponential search space growth, where conventional binary metaheuristics often suffer from premature convergence. To address this, EMG-Hybrid is introduced as a novel three-level hierarchical optimization architecture designed around decoupled operational planes, feedback-driven scheduling, and isolated state refinement. At Level 1, a Dual-Engine Controller monitors population Shannon entropy and empirical fitness velocity, functioning as an architectural master-switch that dynamically routes execution between global exploration and local intensification. This control loop conditions the Level 2 active optimization plane, where an Adaptive Black Hole Gravity (ABHG) execution layer performs aggressive feature manifold pruning via a distance-decoupled penalty kernel driven by fitness similarities and Hamming metrics. To structurally isolate macro-exploration from micro-exploitation, Level 3 encapsulates an autonomous Elite Memory Archive that preserves high-fidelity sub-schemas and hosts a discrete bitwise Differential Evolution (DE) micro-refinement sandbox for resolving complex parameter epistasis. Theoretically, uniform combinatorial bounds derived via Vapnik--Chervonenkis (VC) dimension criteria demonstrate that this multi-level restriction mitigates high-dimensional overfitting. Empirical results across manifolds scaling up to $d > 100{,}000$ features, complemented by high-performance computing (HPC) profiling under lock-free atomic Compare-and-Swap (CAS) primitives, indicate strong performance gains over state-of-the-art baselines. The framework exhibits near-linear parallel speedups, sub-quadratic runtime scaling, and substantial dimensionality reduction, offering a scalable and structurally innovative paradigm for large-scale data mining. 

---
# Compile as Hypothesis: A Headroom Typology for Prompt Optimization on Real Reasoning Models 

**Authors**: Zhengshuai Yang, Changyun Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.01135)  

**Abstract**: Automated prompt compilers search for or rewrite prompts under the implicit assumption that optimization consistently improves downstream task performance. We evaluate this assumption on frontier reasoning models through rigorous paired, same-day measurements. Across four optimizer families (genome search, profile-guided mutation, GEPA, and ESPO) and eight tasks, structured search yields no statistically significant accuracy gains on strong reasoners. However, a discrete structural genome provides essential architectural stability, preventing catastrophic bimodal seed-collapse (where unconstrained search collapses toward the judge floor in 3/5 runs). Through 190-problem evaluations on knowledge-intensive benchmarks, we establish a three-dimensional headroom typology: prompt optimization cannot compensate for intrinsic knowledge deficits or physical token truncation, but successfully resolves protocol and formatting errors. On gpt-6, while free-text reflective search (GEPA) yields a +.14 gain on a task where structured search remains flat, qualitative ablation reveals that ~.035 stems from judge-rubric reconstruction while ~.105 reflects genuine instruction refinement. Our pre-registered AutoAPC-Select deployment gate matches the oracle policy in 5/7 groups and rescues seed-collapse arms with zero regret. Finally, we expose silent backend-mixing in commercial APIs during outage windows and provide automated per-call audit verification tools to ensure empirical evaluation fidelity. 

---
# HiGenTrans: Hierarchical LLM Knowledge Transfer for Generative Recommendation 

**Authors**: Zhuohang Jiang, Yuxin Chen, Haohao Qu, ZeFei luo, Yunlong Wang, Xu Yuan, Haojie Ding, Boheng Zhang, Fan Yang, Hongyang Wang, Jun Wang, Zhongbo Sun, Wenqi Fan, Li Qing, Wenwu Ou, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01129)  

**Abstract**: Generative recommendation promises to simplify conventional multi-stage recommender systems by generating item Semantic IDs (SIDs) autoregressively. However, two obstacles limit its effective- ness in industrial settings: interaction-only supervision underuses the semantic knowledge encoded in pretrained language models, while beam-search likelihood ranks token sequences rather than directly estimating item relevance. To address these limitations, we propose HiGenTrans, a three-level framework that introduces LLM knowledge at the input, representation, and decision levels. First, Global Semantic Priming (GSP) adds an LLM-derived user representation to the beginning-of-sequence embedding, provid- ing a semantic prior before SID generation. Second, Hierarchi- cal Representation Transfer (HRT) aligns intermediate LLM representations with depth-matched GenRec layers through mean- centered projections and gated residual connections. Finally, Multi- Signal Decision Calibration (MDC) uses a lightweight mixture- of-experts calibrator to combine semantic relevance, generation confidence, and candidate-set context when reranking beam-search outputs. All semantic features are computed only from the observ- able interaction prefix, preventing target leakage, and can be cached offline. Experiments on three public benchmarks show that HiGen- Trans improves Recall@10 by up to 12.9% and NDCG@10 by up to 33.7% over the strongest baselines. These results demonstrate the value of transferring LLM knowledge at complementary stages of the generative recommendation pipeline. 

---
# Robust Reward Machine Learning and Execution in Noisy Environments 

**Authors**: Roko Parac, Lorenzo Nodari, Daniel Furelos-Blanco, Leo Ardon, Federico Cerutti, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2611.01063)  

**Abstract**: Background: Reward machines (RMs) represent non-Markovian reward functions as finite-state machines over high-level propositional events. Traditionally, these events are generated by a deterministic labelling function that maps environment states to proposition sets. However, real-world labelling functions are inherently noisy.
Objectives: Design and implement a method that learns RMs in the presence of noisy labelling functions and exploits them for policy learning in both discrete and continuous environments.
Methods: We propose Prob-IRM, a method that learns RMs from noisy proposition histories using inductive logic programming (ILP) and exploits them for policy learning. Noisy proposition histories are represented as training examples for ILP, which learns an RM that maximises coverage of low-uncertainty histories. RM learning is interleaved with policy optimisation. Prob-IRM uses a belief over RM states to determine when to relearn the RM and to provide additional guiding rewards.
Results: Prob-IRM is evaluated on environments with discrete and continuous state spaces under varying levels of labelling noise. Prob-IRM performs comparably to baselines using handcrafted RMs and outperforms approaches that use recurrent neural networks to encode history, despite operating with noisy labels.
Conclusions: Prob-IRM represents a step towards deploying reward machines in noisy, real-world environments. 

---
# Appointed-Time Fault-Tolerant Fast Terminal Sliding Control for UAVs with a Tracking Error-Based Disturbance Observer 

**Authors**: Jinfa Li, Sheng Xu, Yanjie Wu, Chengyue Su  

**Link**: [PDF](https://arxiv.org/pdf/2611.01061)  

**Abstract**: This paper addresses the critical challenge of high-performance fault-tolerant tracking control for quadrotor UAVs under complex conditions including external disturbances, parameter uncertainties, and actuator faults. A synergistic control framework is proposed by decoupling the system into position and attitude loops, each governed by the following three innovations. First, an Appointed-Time Prescribed Performance Function (AT-PPF) is developed to strictly constrain the tracking error within a predefined funnel boundary, ensuring convergence within a fixed time regardless of initial conditions. Second, a Tracking Error-Based Fixed-Time Composite Disturbance Observer (TFxTCDO) is developed to enable fast and accurate estimation of lumped disturbances without requiring full-state information or precise system modeling. This observer directly couples disturbance estimation with the control objective, thereby enhancing robustness and convergence. Third, a Fast Second-Order Terminal Sliding Mode Controller (NFSO-TSMC) is designed to achieve faster convergence and effective chattering mitigation. Simulation results demonstrate that the proposed control strategy significantly outperforms conventional methods in terms of tracking accuracy, fault recovery speed, and robustness against uncertainties. 

---
# Temperature, Not Knowledge: What Recursive Self-Training and Its Remedies Do to a Language Model's Confidence 

**Authors**: Dipto Sumit, Syeda Fabiayee Jahan, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2611.01059)  

**Abstract**: Recursive self-training can collapse language models, and many proposed remedies target overconfidence, yet their effects on confidence are rarely measured. We study a five-round self-training loop on Qwen3-1.7B, using three seeds per compared method, and distinguish confidence temperature (drift from a calibrated starting point) from discrimination (ranking correct answers above incorrect ones). Accuracy and overall discrimination remain nearly unchanged, but confidence sharpens and the most confident answers become less reliable. Matching predictive entropy to the calibrated starting model each round recalibrates nearly as well as label-based temperature fitting, at both 1.7B and 8B. Sharpening is not inherent to self-training: exact samples from a model's distribution have zero expected gradient. Instead, most drift comes from fitting each generated answer twice, with top-p truncation contributing the rest; removing both keeps entropy on generated outputs flat. Remedies affect the two properties differently: truncated cross-entropy, an entropy bonus, and single-pass training preserve temperature without improving discrimination, while focal loss and multi-sample targets improve discrimination but cause underconfidence. Reference-data mixing also overshoots. Pairing a ranking-improving loss with label-free recalibration achieves both. 

---
# Constraint Dependency Graph Retrieval for Compliance Detection on Long SKILL Documents 

**Authors**: Shuaitao Zhao, Feng Ni, Lichao Ma, Yuchen Wei, Shuaike Li, Jiaye Lin, Fei Han, Yang Wei, Lu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2611.01042)  

**Abstract**: Large Language Model (LLM) agents carry out business operations under enterprise documents we refer to as SKILL documents, which take the form of standard operating procedures (SOPs) that state the conditions to verify and the actions to take before an operation proceeds. Long, densely cross-referenced SKILL documents exceed the context budget of small deployed auditors. Similarity-based retrieval then brings in irrelevant rules while omitting mandatory prerequisites, and the two-stage audit policy provides no trainable signal for the retrieval stage. We present SkillCDG, a compliance detection framework that reformulates each SKILL document as a constraint dependency graph, shifting rule retrieval from similarity-based matching to closure-based selection under the deployment's context budget. Each atomic rule is anchored to its source text span as a graph node, and mandatory prerequisite relations form the sole edge type. Policy-generated retrieval keywords activate seed rules, whose dependency closure yields a compact and conditionally complete rule context within that budget. The same graph also supports a joint optimization scheme. The retrieval stage is trained with group-relative policy optimization using a verifiable reward that combines output-format validity with downstream verdict correctness, with closure non-emptiness serving as a validity gate, while selective on-policy distillation concentrates teacher supervision on high-entropy reasoning tokens, with verdict and termination tokens always included, to improve verdict accuracy. Experiments on three enterprise datasets and two public benchmarks show that SkillCDG improves accuracy and F1 over the full-document and retrieval-based baselines. 

---
# IJev: Order-Invariant Decisions via Iterated Cyclic Reordering 

**Authors**: Youzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01015)  

**Abstract**: System One decision models answer a choice question in one forward pass, but their answer depends on the order in which options are listed, and their probabilities carry a prior over answer labels. An invariant ordering operator is any fixed function that returns the same ordered option list for every input order of a fixed option set. Starting from any such order and averaging a listwise readout over its K cyclic shifts removes the input order, but the decision still depends on the chosen cyclic class: those $K$ shifts are only one of the (K-1)! classes, and different invariant operators can select different classes. We propose IJev, which repeatedly reorders the options by the cyclic-averaged, label-prior-corrected distribution and runs another cyclic average. Every iterate is exactly input-order-invariant. When the iteration does not converge, we average the round distributions rather than return the last round. On five benchmarks and five models, IJev has flip rate zero in every cell. The default uniform average over T=20 rounds substantially improves calibration over a single cyclic average, and its accuracy is comparable. A descriptive choice of horizon and averaging rule is best or tied-best for accuracy in all 25 settings. 

---
# Leap-VLA: High-Rate Robot Control from a Low-Rate VLM Backbone 

**Authors**: Lin Yang, Yunjie Shen, Zheyu Li, Zhenjie Tian, Yongli Wang, Dongze Huang, Weiqi Wang, Yucheng Zhu, Zhaoquan Jiang, Guofeng Xie, Peigen Fang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01011)  

**Abstract**: Vision-language-action (VLA) models re-encode the full observation through a multi-billion-parameter vision-language model (VLM) at every control step, even though scene semantics evolve at roughly 1 Hz while reactive control requires 20-50 Hz. We present a VLA that decouples the two: a VLM runs as a low-frequency semantic anchor whose cached context is reused across control steps, while a flow-matching action expert operates at high frequency, consuming the cached semantics with a fresh, low-latency visual stream from a compact encoder pretrained on robot video with masked reconstruction, optical flow, and depth. The measured age $\tau$ of the cache conditions the expert through adaptive layer normalization. Reusing the cache makes the policy a function of its age, and training covers that function explicitly: cache ages sampled from real episode history, plus an age-ramped pixel corruption that erases the stale snapshot the way deployed staleness corrupts content. On the asynchronous deployment stack, the recipe closes most of the staleness gap: on LIBERO-Spatial, 61.4% to 95.5% at a 2 s semantic refresh (95.9% at 1 s, within 1.6 points of the frequency-matched anchor); averaged over the four LIBERO suites, the deployed policy holds 95.2% at a 1 s refresh and 94.5% at 2 s against a 96.3% matched anchor. A forward-only chunk-boundary correction (soft-nudge), an autograd-free reformulation of real-time-chunking guidance, carries per-control-step re-planning without loss (97.1%) at zero added latency. The same lens exposes a second defect: under an instruction-scene mismatch the policy keeps manipulating the scene, and a chain-of-thought verdict gate trained on zero-cost negative episodes freezes execution exactly on mismatched instructions without hurting feasible success. 

---
# GURU: A Hierarchical Neuro-Symbolic Multi-Agent System for Verification-Aware Automation of Mission-Critical Digital Engineering and Operations 

**Authors**: Allan Grosvenor, Anton Zemlyansky, Abdul Wahab, Kyrylo Bohachov, Aras Dogan, Ryland Adams, Dwyer Deighan  

**Link**: [PDF](https://arxiv.org/pdf/2611.01001)  

**Abstract**: An AI can find a safe action faster than any human, but it must never decide what counts as safe. We present GURU, a hierarchical neuro-symbolic multi-agent system for mission-critical digital engineering and operations, built on one partition of authority: learned models propose, rank, and accelerate; a symbolic rules engine checks every consequential action against hard limits before it executes; and validated physics confirms and decides. GURU runs a workflow as a closed control loop, with Navigation, Action, and Planning agents sharing one Common Internal Language and one training factory. Nuclear, aviation, and national-security guidance now requires this pattern; a harnessed language model keeps the model at the center, GURU puts a validated reference there. In computational fluid dynamics, the workflow raised the converged-simulation rate on unseen aircraft geometries from under 1 percent to 74 percent over 100 single-attempt cases, ran 54 live cases to Mach 6 on three facilities, and an ablation put the symbolic layer at 13 points of boundary-layer capture. Knowledge-graph retrieval and role adapters raised ParaView control success from 55 to 75 percent. In space domain awareness, where a language model cannot be trusted to decide, a frozen self-supervised world model under validated propagation scored highest of five methods at conjunction risk, including fully fine-tuned language models, with several-times-lower seed variance: the trustworthy design gives up nothing in accuracy, and the recipe transferred to reactor anomaly detection. A radar-to-avoidance debris pipeline was demonstrated in stages on a 2026 reentry; the hierarchy coordinates simulated teams of 200 aircraft. Training scaled near-linearly past 1,000 Aurora nodes and ran at 8,000. The approach extends to a control-grade microreactor digital twin with validated physics, not a learned model, as the safety basis. 

---
# Pulse: Long-Horizon Tasks Need Streaming Personalization 

**Authors**: Qijia Zhuang, Jialiang Liang, Yuxiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.00988)  

**Abstract**: Large language model agents increasingly undertake complex, long-horizon tasks that benefit from sustained personalized support through long-term memory. Existing memory-augmented methods retrieve and use memory primarily from the initial request, with limited awareness of ongoing execution and emerging personalization needs. Agents may therefore overlook user preferences and project constraints during multistep tasks. We propose Pulse, a streaming personalization method for long-horizon tasks. An independent, lightweight memory agent continuously monitors execution, identifies emerging personalization needs and potential preference conflicts, and uses user and project memories to provide targeted interventions at appropriate stages. Reinforcement learning optimizes when and how the memory agent intervenes, improving the timeliness and accuracy of personalized support. Pulse scores 71.9% on $\pi$-Bench and 46.4% on MCP-Persona, achieving the highest overall scores among the compared methods on both datasets while improving task execution efficiency. 

---
# Benchmarking Machine-Learning Classifiers for Neoantigen Immunogenicity Prediction and the Limits of SHAP Interpretability 

**Authors**: Ramon Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2611.00907)  

**Abstract**: Personalized cancer vaccines require prioritizing, among hundreds of candidate mutant peptides, the few that will induce a T-cell response. This study asks what general-purpose machine learning achieves on that task, and whether the features such models identify as important can be trusted.
We make three contributions. The first is a benchmark of sixty models: twelve supervised classifier families, each in five configurations, evaluated on the Immunogenic Tumor Specific Neoantigen database (ITSNdb; 199 experimentally validated neoantigens, 129 immunogenic and 70 non-immunogenic). All sixty are compared under one stratified k-fold cross-validation protocol, in which the data used to choose a configuration are never the data used to score it. The second is an interpretability study that combines SHAP with feature ablation. The third is a shuffled-label control that tests whether a SHAP value has been earned.
Two results follow, and both run against current practice. First, the classical linear models lead the ranking. The gradient-boosted ensembles and neural networks that dominate recent work bring no advantage. Second, SHAP can fail on high-cardinality categorical features. Ablation establishes the presenting HLA allele as the dominant input and the source gene as dispensable. SHAP reports the opposite, assigning 68.7% to 81.7% of its total value to the source gene in every model that gives each gene its own weight. Shuffling the gene labels at random among the peptides resolves the contradiction: the meaningless column earns the same SHAP value as the real one. That value measures only that ITSNdb contains 173 genes for 199 peptides. We therefore recommend that SHAP values for high-cardinality categorical annotations in cohorts of this size always be reported against such a shuffled-label control, and we describe one in full. 

---
# Can Agentic AI Undermine Its Own Assurance Loop? Closed-Loop Dependency Assurance for Cognitive Network Digital Twins 

**Authors**: Ahmed Hussain, Konstantinos Kalogiannis, Salahuddin Salahuddin, Hexu Li, Thiemo Voigt  

**Link**: [PDF](https://arxiv.org/pdf/2611.00838)  

**Abstract**: Agentic Artificial Intelligence (AI) and Network Digital Twins (NDTs) are increasingly combined to form Cognitive Digital Twins (CDTs) that translate operational intents into network changes and support closed-loop management. However, a network change may achieve its intended objective while disrupting the monitoring or recovery mechanisms needed for subsequent decisions. For example, a service may be rerouted while its latency-monitoring traffic continues along the original path, producing measurements that no longer reflect the service's performance. We propose Closed-Loop Dependency Assurance (CLDA), an assurance mechanism that augments agentic AI-enabled NDTs with dependency-aware validation. In our framework, an agentic AI planner generates and refines network plans, while the NDT evaluates the projected network outcomes of these plans. CLDA extends this analysis using a Loop Dependency Graph (LDG) that connects service requirements to monitoring mechanisms, forwarding paths, failure domains, and recovery prerequisites. CLDA projects candidate actions onto these dependencies, checks intermediate states of multistep plans, and supports deterministic make-before-break repairs using temporary monitoring mechanisms. Unlike the Large Language Model (LLM)-based planner, CLDA uses explicit, deterministic rules for assurance decisions and repair. We evaluate CLDA through controlled reconfigurations, a 240-trial LLM-based planning campaign, and live network execution. A separate synthetic study examines deterministic assurance scalability across network models representing up to 1,024 routers. 

---
# AdKnob: Ad Intensity Control and Labeling for LLM-Native Advertising 

**Authors**: Woo Jae Kim, Seongho Keum, Joonsung Jeon, Suhyeon Ha, Sooel Son, Sung-eui Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2611.00822)  

**Abstract**: Large language model (LLM)-native advertising has emerged as a viable monetization channel as LLM service providers seek sustainable revenue streams. However, integrating native ads into LLM responses introduces two largely understudied challenges: (1) controlling ad intensity -- how prominently advertising content appears -- and (2) ad labeling -- identifying which segments of the generated text constitute advertising. We propose AdKnob, a framework jointly addressing both challenges. For ad intensity control, we introduce new ad control tokens without pretrained semantic associations and align the model via direct preference optimization to generate ads at each desired level. For ad labeling, we propose an attention rollout-based attribution mechanism that identifies ad segments by tracing their attribution to input ad-related tokens, requiring no additional inference cost. We further construct MI-KnobSet, the first multi-level ad intensity preference dataset for LLM-native ads. Experiments across six LLMs and human evaluations show that AdKnob achieves uniformly spaced ad intensities where baselines collapse to a narrow range, achieves the highest ad labeling accuracy while being over 500x faster than the strongest baseline, and is preferred over baselines by human evaluators up to 89.6% of the time. 

---
# VCoT-SQL: Enhancing Text-to-SQL Generation through Dual Chain-of-Thought Validation Training 

**Authors**: Rui Zhang, Yujia Su, Ruixuan Li, Yuhua Li, Yixin Su  

**Link**: [PDF](https://arxiv.org/pdf/2611.00815)  

**Abstract**: Open-source LLMs provide a flexible alternative to closed-source models for Text-to-SQL but still struggle with complex queries. Finetuning with chain-of-thought (CoT) data improves reasoning, yet standard datasets lack CoT annotations, requiring synthetic generation. Existing synthesis methods only verify the final answer, allowing flawed reasoning steps to corrupt the training data. We propose VCoT-SQL, a framework that improves Text-to-SQL generation through intermediate-step validated CoT supervision and a dedicated SQL validator, which enables small open-source models to outperform frontier closed-source models. VCoT-SQL trains two complementary models on curated reasoning traces: an SQL generator using clause-level CoT templates where reasoning steps are verified via execution and schema alignment checks, and an SQL validator using diagnostic CoT templates to detect syntactic, semantic, and type level errors. We further design a hierarchical schema filtering pipeline that integrates and refines complementary retrieval strategies to distill relevant database elements. VCoT-SQL achieves 66.82% execution accuracy on challenging dataset BIRD dev and 86.8% on Spider test, improving the base 7B model by 7% on BIRD. Notably, our CoT synthesis relies on 7B models yet outperforms methods that use 72B-scale or even proprietary closed source teachers for data generation, demonstrating that rigorous step-level validation can be more effective than scaling up teacher model size. Our dataset and code are available at this https URL. 

---
# AI Data Analysis: A New Approach - Multi-engine Joint State Data Analysis 

**Authors**: Jean Ye  

**Link**: [PDF](https://arxiv.org/pdf/2611.00807)  

**Abstract**: AI data analysis today is three fragmented worlds: LLMs understand semantics but cannot compute; BI aggregates precisely but is stateless; ML processes large-scale data on a stateful kernel but lacks semantics. This paper proposes a combination mechanism, named Multi-engine Joint State Data Analysis, that fills a still-open gap - the multi-engine joint context: the data context is embedded into LLM conversational analysis, where the LLM serves only as the semantic, reasoning, and orchestration layer (not computation) and connects four engines through the 1M context window - Python and R ML kernels, SQL over relational databases, Cypher over graph databases - yielding two multiplications, "semantics and precise computation" and "relational and graph", under enterprise-grade governance. The novelty is a four-dimensional coordinate system: horizontal (engine kinds), vertical (state flow - the watershed), spatial (business generalization), temporal (iterative convergence); a deterministic grounding method (one-shot injection of this http URL + this http URL, M+S) carries the spatial and temporal dimensions. Empirically, all four dimensions are substantiated: horizontally, four engines are quantified - SQL, Cypher (96.5% on CypherBench movie), ML in Python and R (TabReD weather); vertically, state flows across engines and grounding is reused across kernels; spatially, SQL generalizes across three business databases - AdventureWorks 91.3%, financial 95.3%, formula_1 94.6% (average), about 20-23 points above single-shot SOTA 74.45%; temporally, accuracy converges monotonically as notes accumulate. The core invariant - the separation of semantics from computation plus a cross-data-domain persistent context - does not expire as LLMs and agents evolve or engine forms change. 

---
# A Discrete-Time Variational Objective for Degenerate Diffusion and Equivalence to Flow Matching 

**Authors**: Rares Dolga, Harshad Dahake, David Barber  

**Link**: [PDF](https://arxiv.org/pdf/2611.00800)  

**Abstract**: Diffusion models and more recently Flow models have widespread application. In this work we consider degenerate Diffusion models in which the per-step generator is deterministic. We construct a modified variational framework and show a formal equivalence between Flow models and Degenerate Diffusion models. For our discrete-time variational objective we show that the optimal degenerate generator recovers the data distribution in the limit of an infinite number of generator steps. Our work therefore provides a unified perspective, showing that discrete-time implementations of flow matching and the discrete-time Degenerate Diffusion model can be considered within the same simple variational framework, removing the need for the continuous-time machinery normally used to justify flow matching models. 

---
# BridgeDock: Taming Generative Co-Folding Prior for Molecular Docking with Diffusion Bridge 

**Authors**: Shikun Feng, Zipeng Yang, Qiuyi Li, Mingze Yin, Xiaoyuan Zhang, Tingjun Hou, Yiheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.00795)  

**Abstract**: Molecular docking aims to predict the three-dimensional structure of a protein-ligand complex; in the conventional structure-based setting, it is conditioned on an input protein structure and ligand, often starting from an apo receptor in flexible docking or a holo receptor in rigid docking. Recent co-folding models exemplified by AlphaFold 3 achieve strong accuracy in protein-ligand complex prediction, but they typically generate complexes from sequence-derived inputs and therefore do not directly use apo structures as structural priors in docking, leading to a larger conformational search space and weaker robustness in difficult or few-step settings. We present BridgeDock, a framework that adapts pretrained co-folding backbones to flexible docking by modeling the apo-to-holo transition as a diffusion bridge, allowing generation from the initial protein-ligand state rather than from Gaussian noise. To make this bridge formulation compatible with pretrained denoisers, we introduce an alignment mechanism that maps bridge states to the original denoising schedule of the pretrained model. Experiments on standard flexible docking benchmarks show that BridgeDock consistently outperforms strong co-folding baselines. Moreover, BridgeDock is computationally efficient at inference time and remains effective even with very few denoising steps. 

---
# A Tool in the Forward Pass: Frozen Tape Modules as Layers of a Small Language Model 

**Authors**: Michael Granitzer, Saber Zerhoudi, Mohammad Al-Maamari, Hai Dang Dang, Kanishka Ghosh Dastidar  

**Link**: [PDF](https://arxiv.org/pdf/2611.00783)  

**Abstract**: Calling a tool through generated text adds decoding steps, even when the computation itself is short. We study an alternative: a small neural module that runs inside a language model's forward pass. A learned writer turns hidden states into a symbolic tape, a short sequence of discrete symbols that encodes the computation. The module, trained separately and then frozen, reduces the tape to a result, and a learned reader returns that result to the model. We evaluate arithmetic and object tracking in gemma-3-1b-it. With a writer trained on labelled tapes, the layer answers all 750 items of BIG-Bench Hard's tracking shuffled objects correctly. At seven people it needs about 1/10 of the GPU kernel time of a text-tool baseline that uses the same host model and exact code, with accuracy 1.00 against 0.99. The writer can also be learned from answers alone, guided by a hand-written copying prior, but less reliably: 5 of 6 runs succeed in the 1B host and 2 of 4 in the 4B host. Two independently trained layers keep their task accuracy when attached to one model, and a learned gate keeps them silent on the tested general-text and look-alike prompts. The results support a narrow use case: a bounded computation that returns one symbol can avoid the decoding cost of a textual tool call. Extracting the right input reliably, and deciding when to act, remain the main open problems. 

---
# One Offset, Two Directions: Baseline Adjustment via Subgroup Offsets 

**Authors**: Jiaying Zhou, Jiachen Liu, Qingchao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.00782)  

**Abstract**: Group Relative Policy Optimization (GRPO) estimates advantages against the within-prompt mean reward, a baseline shared across the group. In practice, rollouts within a prompt are not always homogeneous, differing in their solution strategies and reasoning trajectories, along which they may group into subgroups. We further find that these heterogeneous subgroups may exhibit distinct reward distributions. The global baseline, however, collapses them onto a single credit axis, where the credit reflects neither a rollout's standing within its subgroup nor the subgroup's standing relative to the rest. We therefore study how to adjust the baseline when the rollouts of a prompt fall into heterogeneous subgroups. We define the \emph{subgroup offset} as the deviation of a subgroup mean from the global mean, and shift the baseline along it by a single signed coefficient $\lambda$. The sign of $\lambda$ yields two opposing directions: \emph{Alignment} ($\lambda{>}0$) adds the offset to the baseline, drawing credit toward a rollout's relative position within its subgroup; \emph{Contrast} ($\lambda{<}0$) subtracts it, drawing a rollout's credit toward the relative quality of the subgroup it belongs to. The adjustment acts on the baseline alone, leaving the reward signal and the optimization objective unchanged. In this work, we induce subgroups from strategies, grouping rollouts that share a strategy into the same subgroup. We evaluate both directions on Qwen3-8B-Base and Olmo3-7B-Base across four competition-level math benchmarks, against a $\lambda{=}0$ anchor that shares the same strategy-conditioned sampling and differs only in the baseline. Against this anchor each direction yields modest but consistent gains in multi-sample coverage, with Contrast the stronger of the two. 

---
# Who Schedules Agent Memory? A Sidecar Approach to Long-Horizon Tasks 

**Authors**: Qijia Zhuang, Rui Liu, Yuxiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.00779)  

**Abstract**: As large language model agents increasingly undertake complex, long-horizon tasks, memory has become an important component for supporting sustained decision-making and action. Existing methods primarily focus on memory storage, updating, and retrieval, but often overlook how memory scheduling affects task execution: the main agent must identify memory needs, initiate retrieval, and wait for results while advancing the task, increasing both scheduling burden and execution time. To address this problem, we propose MemSidecar, which decouples memory scheduling from task execution through two independently selectable modes. Hooks directly delivers bound memory to the main agent when execution events match predefined conditions. Foresight uses an independent agent to predict potential queries, retrieve results in parallel, and store them in a cache, delivering matching and valid results when actual requests arrive. Hooks and Foresight achieve scores of 41.23% and 44.34% on MCP-Persona, respectively, while effectively reducing execution costs and latency and improving the efficiency of long-horizon agent execution. 

---
# Decision-Centric Research: A Calibrated Decision Model and Benchmark for Scientific Agents 

**Authors**: Shaoning Zeng, Wen Huang, Xinzhong Zhu, Yi Duan, Jianhang Zhou, Jianping Gou  

**Link**: [PDF](https://arxiv.org/pdf/2611.00774)  

**Abstract**: Scientific agents make consequential choices about evidence, next actions and research strategy, but choices embedded in generated text are difficult to evaluate independently. Decision-Centric Research (DCR) is introduced to make individual choices the common unit of supervision, prediction and evaluation through typed predictions over an explicit Research State. DCR defines 24 decision types, ten of which are instantiated in the Research Decision Benchmark (RD-Bench) of 2,000 rule-generated cases. The Decision-Centric Research Model (DCRM) shares a candidate-token readout across types to produce answers and option probabilities in one forward pass. Under a held-out research-domain split, a 0.6B DCRM (state-aware) achieves 89.4% Top-1 accuracy, compared with 27.8% for DCRM (question-only) trained with the same recipe and 36.4% for DeepSeek-V4.1-Flash. All 30 type-seed comparisons against DCRM (question-only) are significant after Holm correction. After held-out calibration, the resulting prediction sets contain a mean of 1.515 candidates at 97.1% empirical label coverage, reducing mean set size by 57.4% relative to DCRM (question-only) at similar coverage. These results establish a compact state-aware decision layer with accurate typed prediction and an inspectable uncertainty interface. 

---
# Designing Informative Actions: Action Construction for Long-Horizon Research Agents 

**Authors**: Alkut Arxidin, Shaoning Zeng, Jianhang Zhou, Jianping Gou, Xinzhong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.00773)  

**Abstract**: Large language model agents support automated scientific research by connecting literature analysis and planning with experimentation. Evaluating which action an agent selects does not establish which experiment its instruction specifies, because omitted interventions or replication counts leave consequential choices to the executor. We introduce a reading-set framework for evaluating action construction under a common cost rule. Across 30 synthetic states, complete-design requests increase affordable efficiency by 465.98% and 327.91% relative to the action-name baseline in the two model families, respectively. An author-constructed research-action evaluation dataset provides 60 scenarios for complementary judgment evaluation. On this dataset, the scenario-level information-per-cost measure has a mean within-scenario correlation of 0.733 with judgments from 18 human participants. The framework makes design specification and execution ambiguity measurable, providing a reproducible basis for assessing how research agents translate scientific intent into informative, budget-feasible experiments. 

---
# Reading Between the Chapters: A Graph-Free RAG Rebuilt With AI and Human Feedback 

**Authors**: Dipto Sumit, Syeda Fabiayee Jahan, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2611.00752)  

**Abstract**: Many questions about a book can only be answered by connecting passages chapters apart, such as a cough heard three times across a horse's life. We introduce CrossChapterQA, 1,853 such questions over 14 public-domain books. Each question lists the two to six chapters it needs (spanning half the book at the median) and the relation linking them, cites evidence from each, and has two independent reference answers graded blind. We use it to build LadderRAG, which indexes a book as passages, summaries and book-level cards for about $0.41 and answers a question for under a cent. Its first version failed mostly at retrieval. We redesigned it four times, steered by an LLM judge's verdicts and our analyses of failed answers: a design-level counterpart of RLAIF and RLHF in which feedback changes the architecture, not model weights. We also report the two redesigns that made answers worse. On the 191 questions its first version got wrong, the final version retrieves over twice as many required chapters on held-out books (recall 0.36 to 0.79), and ten annotators rate its answers 6.5/10 on average (764 ratings). 

---
# Inherited or Learned? Backdoor Selectivity in Distilled Language Models 

**Authors**: Dipto Sumit, Sanzar Rahman, Ramisa Faria Puspita, Samiha Fairooz Urbi, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2611.00748)  

**Abstract**: Distillation is a routine step in building small language models, and a backdoored teacher can pass its backdoor to the student. We study what a student inherits when the backdoor is a rare string that triggers a fixed refusal. Keeping the trigger fixed, we vary only where it appears, at a random word boundary or appended to the instruction, and compare teachers and students on the exact trigger, six control strings and clean inputs. Across Qwen2.5, Llama 3.2 and Gemma 3, with 5% of transfer prompts triggered, randomly inserted triggers become less selective after distillation in all nine teacher-student pipelines, while appended triggers gain selectivity in Qwen and Llama and change little in Gemma. Students trained on the same prompts without a teacher show that much of this is not inherited. Llama and Gemma students learn an appended backdoor from the 150 poisoned examples alone, and only Qwen's 0.5B student needs the teacher. The data also sets where the trigger works. Gemma teachers respond to it anywhere, but their students only when it is appended. Audits should compare teacher and student on the same inputs, probe several trigger positions and include a student trained without the teacher. 

---
# Can Jev Improve Demonstration Retrieval for In-Context Learning? 

**Authors**: Rui Song, Yingji Li, Hao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.00737)  

**Abstract**: Demonstration selection plays a crucial role in the effectiveness of in-context learning (ICL) for large language models (LLMs). Extensive research has explored various retrieval strategies to identify informative demonstrations and improve downstream performance. Recently, the emergence of Jev, a decision-oriented language model capable of producing probabilistic judgments, offers a new perspective on evaluating demonstration utility. This motivates us to investigate a fundamental question: Can Jev improve demonstration retrieval for in-context learning? In this paper, we demonstrate that Jev can effectively enhance demonstration retrieval by directly assessing the utility of candidate demonstrations for a given query. However, applying Jev to large-scale retrieval introduces substantial computational overhead, as each candidate requires an additional decision-making inference. To address this limitation, we propose a simple yet effective knowledge distillation approach that transfers Jev's utility estimation capability to a lightweight multilayer perceptron (MLP). By learning to approximate Jev's probabilistic judgments, the distilled model enables efficient demonstration ranking while maintaining competitive retrieval performance. Our study highlights the potential of decision-oriented language models for demonstration selection and provides an efficient framework for integrating their decision-making capabilities into ICL. 

---
# Designing Behavioral Profiles for Language Model Simulation of Out-of-Distribution Human Behavior 

**Authors**: Robin Na, Kehang Zhu, Logan Cross, Michiel A. Bakker  

**Link**: [PDF](https://arxiv.org/pdf/2611.00661)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate human behavior, but which descriptions of people improve prediction when experimental designs change remains unclear. We develop a framework that draws candidate behavioral features from the literature and constructs profiles from earlier human experiments. A profile-generation LLM assigns feature values from observed actions, messages, and surveys; a separate LLM simulates behavior using these profiles. By comparing simulated and observed outcomes in the learning data, we select profile contents through forward feature selection, adding features while they improve prediction. We then evaluate the selected contents without modification against outcomes from new participants under new designs. Across 616 public-goods games, 144 multi-party bargaining games, and 1,480 generalized 11-20 money-request designs, leading one-feature profiles predict human outcome distributions better than no profile or general-purpose personas. Profiles with more features provide no clear additional benefit, and features' relative predictive value largely persists across learning and validation. Using profiles from one game family to simulate another produces uneven gains and can make predictions less accurate than simulations without profiles. The framework makes the choice of behavioral information a testable part of simulation design, helping researchers assess which descriptions remain useful as experimental conditions change. 

---
# FastAgent: A Physics-Aware FastFluent-to-Fluent Workflow for Rapid CFD Screening and Industrial Steam Dewaxing Design 

**Authors**: Chen YanKai, Yu QiuHeng, Yuan Man, Zou Ting, Wu Lin, He Jian, Zhang QiongYuan, Jing Tao, Xu QingYan  

**Link**: [PDF](https://arxiv.org/pdf/2611.00645)  

**Abstract**: Industrial computational fluid dynamics (CFD) becomes costly when complex geometries, long transients and coupled physics must be evaluated across large design spaces. We introduce FastAgent, a physics-aware campaign controller built around FastFluent, a lightweight numerical engine for rapid physical screening. FastFluent supplies stage-specific fields and diagnostics; FastAgent allocates physical models, numerical resolution and simulation horizons by the active quantities of interest (QoIs), while returned Fluent discrepancies update subsequent decisions. Across six public CFD benchmark campaigns, the framework reduces normalised worst-QoI error by 5.68-99.14% and accelerates decision-relevant computation by factors of 1.53-15.36 relative to a matched Generic Agent control. Applied to high-pressure steam dewaxing in investment casting, FastAgent identifies nonuniform condensation heating as the dominant startup mechanism and separates rapid risk-window localisation from high-fidelity melt-completion confirmation. The rapid route requires 81.39% fewer cell-time updates than a fine-grid, small-time-step reference. Structural analysis localises fracture at the observed blade edge. Screening 131,072 outlet layouts and selectively evaluating candidates with Fluent and Abaqus identifies Sparse-6, which reduces simulated crack opening by 96.0% and connected crack extent by 66.7% relative to the Original design. These results demonstrate how rapid numerical evidence and adaptive fidelity control connect large CFD search spaces to mechanism-guided industrial design. 

---
# Evidence-grounded large language models guide perovskite solar cell design and optimization 

**Authors**: YanKai Chen, Zhi Wan, ZiHao Li, Tao Jing  

**Link**: [PDF](https://arxiv.org/pdf/2611.00633)  

**Abstract**: Using large language models to interpret experimental evidence, design discriminating experiments and formulate well-supported mechanistic explanations remains a central challenge in materials research. Here we establish PSM-Bench, an 800-question benchmark developed by perovskite researchers to evaluate scientific reasoning in perovskite photovoltaics. We also introduce OpenPerov, an expert-informed language-model framework that learns from expert-calibrated evidence records reconstructed from 4,532 articles; these records link experimental strategies, controls and observations to mechanistic interpretations. Structured-evidence adaptation outperformed conventional question-answer training, and OpenPerov Flash ranked first among 13 closed-book systems. OpenPerov Pro raised Scientific Mastery further, from 87.6 to 89.1, by revising selected passages with retrieved evidence. On questions derived from subsequent studies, both models also outscored expert-authored answers. In an experimental test of self-assembled monolayer design, OpenPerov selected a K analogue that outperformed the Na analogue preferred by experts. Devices incorporating the selected contact reached a power conversion efficiency of 27.05% and retained 96.74% of their steady-state reference efficiency after 1,000 h of dark thermal ageing at 85 $^\circ$C. These results show that reconstructed scientific reasoning provides an effective basis for building domain models that interpret experiments and inform materials design. 

---
# TopoQuant: Topological Framework to Characterize Behavioral Shifts in Quantized Large Language Models 

**Authors**: Baha Rababah, Shahzeb Qamar, Lorenz Sparrenberg, Rafet Sifa, Murat Kantarcioglu, Carson K. Leung, Cuneyt Gurcan Akcora  

**Link**: [PDF](https://arxiv.org/pdf/2611.00631)  

**Abstract**: Post-training quantization (PTQ) reduces the com putational cost of large language models (LLMs), but it can introduce internal structural and behavioral changes that are not captured by traditional metrics such as accuracy and per plexity. We present TopoQuant, a topological framework that uses topological data analysis (TDA) to quantify structural drift in self-attention weights and relate it to behavioral changes in quantized LLMs. Across four LLM families and multiple quantization levels, we show that topological drift remains stable at moderate compression levels but increases sharply below 4 bit precision, closely tracking declines in attention agreement. Our results reveal a strong relationship between topological drift and attention agreement, indicating that structural changes in model parameters closely reflect behavioral degradation under quantization. We further demonstrate that topological features can accurately predict attention consistency using lightweight machine learning models, achieving R2 scores of up to 0.85. Predicting attention agreement enables efficient assessment of be havioral consistency in quantized LLMs, allowing quantization induced degradation to be detected without running expensive downstream tasks. The results indicate that topological drift pro vides an efficient early-warning signal for quantization-induced degradation and can help identify safe compression limits without extensive downstream evaluation. 

---
# Autonomous Analysis of Thermoelectric Materials Big Data by Multi-Agentic AI and Construction of a Knowledge Base and Specialized AI 

**Authors**: Yusuke Hashimoto, Takaaki Tomai  

**Link**: [PDF](https://arxiv.org/pdf/2611.00627)  

**Abstract**: Extracting knowledge from large databases of experimental materials data requires repeated analyses and reusable records of their choices and results. We developed Multi-Agentic AI (MAA), a multi-agent system in which AI agents with distinct roles analyze experimental thermoelectric data under predefined instructions and quality checks, saving analysis records traceable to their sources. In four experiments varying theme and cycle counts, we measured what accumulated in these records. In the largest experiment (100 analyses across 10 themes and 10 cycles), concept co-occurrences were concentrated within themes, vocabulary was shared across themes and some relation descriptions were first detected in later cycles. A specialized AI that searches this experiment's knowledge bases and reads their analysis records answered materials questions with cited sources. Retrospective checks, including tracing an answer generated with a preliminary corpus back to the source data, exposed errors: traceability makes errors findable but not absent. 

---
# XLA from the Trenches: Correctness on TPU Workloads at Scale 

**Authors**: Ignacio Cano, Amir Saadat, Rishipal Singh Bhatia, Samuel Ginzburg, Patrick C. Toulme, Ioannis Zarkadas, Praveen Narayanan, Pratik Bhatu, Nicolas Basile, Alexander Dietmüller, Ruishan Wang, Jacob Wahlgren, Victor Barros, Subhankar Shah, Michael Kuperstein, Farzin Houshmand, Emma Sevastian, Peter Gavin, Karthik Srinivasa Murthy, Berkin Ilbeyi, Amit Sabne, Eli Bendersky  

**Link**: [PDF](https://arxiv.org/pdf/2611.00624)  

**Abstract**: Accelerating modern foundational models has pushed domain-specific compilers toward increasingly complex, multi-tiered optimization pipelines. However, this heavy optimization focus introduces delicate tradeoffs between peak execution performance and correctness. Subtle codegen defects can trigger numerical degradation and memory or buffer corruption, invalidating multi-million-dollar training runs and degrading live serving quality without raising obvious compiler errors. In this paper, we present a battle-tested methodology for verifying and triaging Google's XLA:TPU compiler at hyperscale. We leverage fuzzing techniques, combining synthetic program generation with real production traces, while orchestrating AI agent swarms to autonomously generate stress tests, reproduce failures, and draft fixes. To pinpoint root causes in full workloads, we pair this with a hierarchical differential pipeline that bisects large graphs down to individual failing operations inside fused kernels. Deployed in production, our framework has already driven the resolution of over 142 compiler bugs (with 103 additional findings under review), from which we distill pragmatic lessons learned. 

---
# LM6-CastBench: A Geometry-Aware Simulation Dataset for Transient Thermal Prediction in LM6 Sand Casting 

**Authors**: Suhani Kolhatkar, Anand Khandekar  

**Link**: [PDF](https://arxiv.org/pdf/2611.00619)  

**Abstract**: Transient thermal fields during solidification connect casting process conditions and cooling behaviour to the formation of defect-prone regions in aluminum cast ings. Repeatedly solving this thermal problem across geometries and process conditions is computationally expensive, motivating data-driven surrogate models. However, existing machine-learning studies on casting thermal behaviour rely on study-specific simulation databases that do not support reproducible geometry-level benchmarking. We introduce LM6-CastBench, an open simulation dataset for geometry-aware learning of transient thermal fields in LM6 aluminum sand casting, combining three geometry families, ten variants per family, five pouring tempera tures, and three mould temperatures (450 geometry-process cases in total). Each case retains CAD geometry, unstructured mesh connectivity, spatial descriptors, process metadata, and time-resolved nodal temperatures. We benchmark a point wise MLP against a spatially structured U-Net under geometry-level data splits; the U-Net reduces temperature-field errors by a large margin, with the biggest gains on geometries with localized features and abrupt section changes. LM6-CastBench gives researchers a reproducible dataset and benchmark for geometry-aware surro gate modelling of casting thermal behaviour. 

---
# Trail: Accurate Local Smart Home Control Using Decision Models 

**Authors**: Maleeha Masood, Deepak Vasisht, Indranil Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.00610)  

**Abstract**: Natural-language driven smart home control remains challenging: cloud-hosted Large Language Models (LLMs) raise privacy concerns, while local Small Language Models (SLMs) can generate invalid actions or misinterpret requests. We introduce Trail (Typed Registry-Anchored Inference for Language commands in Smart Homes), a fully local framework that replaces open-ended command generation with hierarchical typed decisions. Trail exploits the natural structure of a home's action space, selecting a room, device, and operation from candidates derived from the runtime device registry. This ensures valid actions while enabling confidence-based abstention when requests cannot be reliably resolved. We also design a structured memory module that uses typed decisions to support user preferences, device references, and multi-turn interactions. To evaluate Trail, we extend HomeBench and generate three datasets: Trek-Base, which tests natural-language variation; Trek-Robust, which evaluates robustness to changes in the home and its description; and Trek-Memory, which assesses conversational memory across multi-turn interactions. Trail achieves 85.3% accuracy on Trek-Robust versus 69.8% for a finetuned Qwen3.5-4B baseline, while remaining competitive on Trek-Base requests. With memory, Trail resolves 75.6% of the requests that depend on earlier turns, compared with 42.5% for the finetuned baseline given conversation history. 

---
# Conversational Interaction for LLM-Based Clinical Text-to-SQL: A User Study with Healthcare Professionals 

**Authors**: Jun-Min Lee, Suyoung Yoo, Gyubok Lee, Edward Choi, Meong Hi Son  

**Link**: [PDF](https://arxiv.org/pdf/2611.00605)  

**Abstract**: Conversational text-to-SQL interfaces could give healthcare professionals natural-language access to electronic health record (EHR) data, but evidence for their real-world usability remains limited: prior evaluations have relied largely on curated benchmarks with fully specified questions or on simulated users. We report a user study in which 14 physicians who regularly work with EHR data used both a single-turn and a conversational multi-turn text-to-SQL interface over the MIMIC-IV database in a within-subjects design; dialogue analyses additionally drew on interaction logs from 21 healthcare professionals across the main study and two pilot studies. The conversational interface was consistently favored: mean System Usability Scale scores rose from 68.2, near the industry average, to 85.9 (p<0.001), with significant advantages across technology-acceptance constructs including performance expectancy, effort expectancy, self-efficacy, and behavioral intention. In post-study forced-choice comparisons, 93% of participants reported greater overall satisfaction with the conversational interface, 86% preferred it for real-world adoption, and 71% trusted its results more; no participant preferred or trusted the single-turn interface more. Turn-level analysis of 425 logged user turns showed that freely composed queries triggered clarification requests in 24.8% of cases versus 1.8% for predefined, fully specified questions, and 64.3% of clarification episodes stemmed from EHR- or clinical-domain knowledge. Conversational interaction is a practical prerequisite, rather than a convenience, for natural-language EHR access. The frequent need for clarification in realistic clinical queries further underscores the importance of conversational interfaces, while benchmark-style evaluation can substantially underrepresent the ambiguity of real clinical use. 

---
# Editing the Goal: Per-Token SAE Steering and Its Depth Limits in $π_{0.5}$ 

**Authors**: Rocio Perales Valdes, Animesh Garg  

**Link**: [PDF](https://arxiv.org/pdf/2611.00602)  

**Abstract**: Sparse autoencoders (SAEs) have recovered interpretable, causally relevant features in robot foundation models (RFMs) such as $\pi_{0.5}$ (arXiv:2603.19183, arXiv:2603.19233). SAE features on individual instruction tokens are less interpretable than timestep-pooled ones (arXiv:2603.19183), and whether they can redirect a robot's goal remains untested. We show that replacing the SAE features of a single goal word with those of an alternative goal word, leaving the instruction text unchanged, steers $\pi_{0.5}$-LIBERO into completing the alternative task. At layer 0, BatchTopK feature swaps redirect behavior in up to 20/20 rollouts in a scene where random-feature swaps redirect none, with success varying across six scenes in two LIBERO suites and across layers. The effect depends on SAE architecture: word-selected Matryoshka edits largely do not outperform random ones. SAEs are also not required. Adding a fixed difference-in-means vector at twice its natural magnitude redirects 8-10/10 rollouts. Layer-wise activation patching shows that goal-word edits remain effective up to layer 6 for object location and layer 2 for object identity, and up to layer 13 in both cases when patching also covers camera tokens. These results are proof of existence in an RFM in simulation and an initial map of how deep goal-redirecting edits remain effective. Visit this https URL for video rollouts. 

---
# X-World 1.6: Building Controllable Virtual Cities with World Models 

**Authors**: XPeng, Boyang Wang, Boyi Sun, Chaoda Zheng, Dongbin Zhang, Guangfeng Jiang, Hanlong Chen, Hongbin Lin, Jianlei Zheng, Jinhao Deng, Kai Yang, Kangjie Chen, Linkun Xu, Minqi Wang, Puyi Wang, Rongliang Cheng, Ruixin Liu, Shijia Chen, Shuang Deng, Sijia Chen, Sin Wai Choo, Tianyuan Xie, Xiangyu Li, Xianming Liu, Xingjie Liu, Xinlong Zheng, Xuejie Lv, Xuezhi Yang, Yi Guo, Yixiao Zeng, Yu Zhang, Yuecheng Ma, Yuhao Wang, Yuqi Zhang, Yuxin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2611.00596)  

**Abstract**: Deploying generative world models for end-to-end closed-loop evaluation and policy learning requires precise scenario control, consistent observations, and sustained generation quality during extended interactions. This report presents \emph{X-World~1.6}, advancing X-World~1.0 toward a more reliable and versatile driving simulator. The upgraded base model offers finer-grained control over traffic semantics, including traffic lights and road markings, together with improved spatial consistency across cameras. Virtual City capabilities broaden scenario construction through a family of models conditioned on diverse 2D and 3D references, including support for generating multi-camera future videos from a single image. An enhanced post-training framework enables efficient causal, few-step generation with stronger temporal coherence, better quality retention over longer rollouts, and improved alignment with human preferences. We also introduce a dedicated evaluation suite to support systematic assessment and guide model iteration. Together, these upgrades expand the scenarios X-World can simulate and improve the efficiency and reliability of its generated observations, supporting broader use in autonomous-driving development. 

---
# State-Aware Activation Recomputation with Replay Contracts and Selective Preservation 

**Authors**: Annan Zhang, Shengjie Song, Wenshuai Yao, Lihao Zhang, Tianmiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.00590)  

**Abstract**: Activation recomputation saves accelerator memory by replaying forward segments during the backward pass, but replay is only correct if it sees the state the original forward pass saw. Mutable buffers, aliased bytes and random-generator state can change in between, so re-executing unchanged code can differentiate a different computation. We make this state explicit. A versioned replay contract specifies what replay must restore, and a joint planner chooses, per segment, between activation retention, whole-buffer snapshots and selective-region preservation, charging each option its measured memory and recovery cost. Independent derivative and byte-level checks validate the contract, and a Torch/JAX cohort retains all 5,900 execution records, including matches, rejections and divergences. On frozen sparse-write H100 workloads, each measured over fifty fresh processes, joint planning reduces paired step time by 18.16% against a correctly costed eager/whole planner at a 44 GiB budget (approximate 97.5% ratio interval $[.8172,.8195]$). At 32 GiB, where the eager/whole planner has no feasible static policy, joint planning is 4.98% faster than fixed regions at 29.94 rather than 26.20 GiB. A complete CIFAR-100 dense-queue application runs seven methods for 100 encoder and 25 probe epochs at six training seeds, with byte-identical test scores within each seed. There, joint planning uses 22.41% more allocated memory than whole snapshots, with a process-time geometric-mean ratio of 1.0080 (seed range $[.9845,1.0191]$). Together, the results map where selective preservation pays off: sparse writes with large activations benefit, while in the dense-queue workload whole snapshots use less allocated memory. 

---
# Evolving Scientific Intelligence: From Fragmented Evidence to Distributed Experimental Loops 

**Authors**: Yuheng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2611.00580)  

**Abstract**: Scientific discovery advances when evidence from different investigations is brought into contact. Yet paper-centric communication leaves experimental results fragmented, delayed, or unreported, creating a "scientific memory wall" between growing computational capabilities and the evidence accessible to AI systems. We envision a different dynamic: in a network of intelligent laboratories, a failed or overlooked experiment in one domain could trigger a hypothesis in another, whose testing generates evidence that stimulates further investigations. Such chains of information collision and empirical validation could transform isolated discovery loops into a cumulative scientific process. We propose Evolving Scientific Intelligence (ESI), an architectural blueprint for enabling these evidence-driven chain reactions. ESI preserves experimental records in shared, computable scientific memory, where evidence from different domains can be recombined into testable hypotheses. It jointly selects candidates and evaluation methods according to scientific value, reliability, and cost, then coordinates physical tests across distributed experimental nodes. Their feedback revises hypotheses and evaluators, while slower meta-control reassesses research objectives and decision criteria to guard against self-reinforcing errors and misguided optimization. We call for an Information Manhattan Project to build the infrastructure for scientific intelligence that evolves through experimentally grounded chains of discovery. 

---
# WAR-CRAFT: Refining Web API Recommendations through Requirement Auditing 

**Authors**: Zishuo Xu, Dezhong Yao, Zhiqiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.00564)  

**Abstract**: Web API recommenders return a set of APIs for a mashup request but do not check it against provider documentation, the main public evidence of an API's capabilities. The set can therefore contain unsupported or redundant APIs and miss requested capabilities. Filling every apparent gap does not fix this, because a gap may already be served, may not need its own API, or may have no documented provider. We study post-recommendation refinement, which revises a frozen recommender's output using only the request, the catalog, and provider documentation. We propose WAR-CRAFT, which treats refinement as three decisions. An LLM removes unsupported or subsumed APIs by judging candidates jointly, then audits the remaining set against a requirement checklist derived from the request. For each uncovered requirement, a fine-tuned catalog-constrained Resolver proposes a catalog API or abstains, and an independent documentation check admits only supported proposals. On ProgrammableWeb with 7 heterogeneous upstream recommenders, WAR-CRAFT raises mean SetF1 from 0.558 to 0.644, above the strongest of 5 refinement baselines (62.5%), and roughly halves functional redundancy (0.127 to 0.065) without changing mean functional coverage. Gains range from +2.2 points on generative recommenders to +18.2 on fixed-cutoff rankers. At matched set sizes, WAR-CRAFT still outperforms truncating the upstream output, so the gain is not explained by set-size calibration alone. Our code is available at this https URL. 

---
# Governing Equations of Human Sleep Macro-Structure:Symbolic Discovery from 1,106 Nights 

**Authors**: Junhui Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.00550)  

**Abstract**: Human sleep is organized into a small number of cycles per night: deep sleep dominates early cycles, REM lengthens toward morning, and light sleep clusters at cycle boundaries. These regularities have long been described qualitatively but never distilled into closed-form laws from large-scale data. A total of 1,106 nights (Sleep-EDF 153 + SHHS 953) are segmented into REM-anchored cycles, and four cycle-indexed statistics follow simple closed-form equations: cycle length declines nearly linearly (R^2=0.996), REM duration follows an inverted U (R^2=0.873), deep-sleep (N3) run length decays exponentially (R^2=0.987) with a time constant of 2.94 cycles, consistent with the homeostatic Process S of the two-process model, and light-sleep (N1) run length follows a U shape (R^2=0.956), confirming a "long-short-long" pattern across the night. The laws are reproduced by per-night differentiable fitting (142/142 and 773/773 nights converge), by unconstrained symbolic regression (PySR), and by a 500-night generative synthesizer built from them. Their robustness is established against the two main methodological threats: bootstrap CIs computed by resampling whole nights (1,000 replications) exclude zero for every shape term, and a five-level scan of the REM-merge segmentation threshold preserves every qualitative form. These appear to be the first closed-form laws governing the macro-structure of sleep, distilled and tested across an age-mixed sample (healthy young adults and community-based older adults). 

---
# Low-Margin Branch Points in Deterministic LLM Deliberatio 

**Authors**: Hajime Shimao, Warut Khern-am-nua  

**Link**: [PDF](https://arxiv.org/pdf/2611.00546)  

**Abstract**: Deterministic multi-agent LLM deliberation can exactly re- play a fixed transcript, yet exact replay alone does not test local resistance to controlled internal interventions. Because each response becomes context for later agents, a changed greedy token can redirect a transcript-mediated execution. We present a white-box audit of this local failure mode in a five-agent committee under greedy decoding. We prospectively select low- and high-margin token positions, require exact no-op re- play, inject norm-relative residual displacements, and measure first transcript divergence within ten generated tokens. Low- margin positions are more branch-sensitive than high-margin controls, while matched margin-reducing gradient directions branch substantially more often than margin-increasing con- trols. We then continue selected realized branches through the committee to distinguish textual persistence from preference and final-decision changes. The contribution is a controlled audit for local branch susceptibility in this specified protocol, not an estimate of deployment frequency, decision correct- ness, or a phenomenon unique to multi-agent systems. 

---
# Skill++: Towards Lifelong Agents via Routing-Aware Skill Refinement 

**Authors**: Hanrong Zhang, Minghao Liang, Jiawei Wu, Yankai Chen, Shicheng Fan, Zhiwei Liu, Enze Ma, Junwei Yuan, Tianyi Wang, Dehai Min, Kening Zheng, Xue Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.00544)  

**Abstract**: Large language model (LLM) agents extend their capabilities with skills, each with a name, a description, and a body. Under progressive disclosure, an LLM router selects skills by name and description alone and loads only the selected bodies. Skill libraries grow continually as users and agents add new skills. Because the router relies on this metadata, a similar new skill can displace an earlier one, breaking previously correct routing even though the earlier skill and the router are unchanged. This routing interference is aggravated for public skills, whose descriptions may not match their bodies. We formulate the problem as lifelong skill routing, a class-incremental continual learning problem. A solution must keep earlier skills selectable without suppressing new ones. Existing approaches fall short, since continually retraining the router risks catastrophic forgetting and rewriting methods either ignore similar skills or require per-skill feedback. We therefore propose Skill++, which rewrites only the name and description of each incoming skill, without router training or labeled data. The new description states what the body supports, how the skill differs from its closest predecessor, and what related work it excludes. As a library grows to 731 skills over ten sessions, Skill++ selects all required skills for 76.3% of queries, versus 36.4% under the original metadata. Across three benchmarks and five LLM routers, it outperforms all compared approaches, and its gains carry over to the Codex CLI agent harness. 

---
# Agentic Context Management for Multi-Turn Human-AI Conversations 

**Authors**: Matthew Ho, Chenglong Wang, Baolin Peng, Nebojsa Jojic, Philippe Laban, Michel Galley, Jianfeng Gao, Lianhui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2611.00518)  

**Abstract**: LLMs degrade in multi-turn conversation partly because of context pollution: the model's own earlier outputs stay in context and anchor later turns, so an early misreading of the user's intent can persist after the user corrects it. Assistant omission (AO), dropping every assistant message, removes this pollution but fails when interaction is referential: later turns refer back to drafts, decisions, and tool-call results that exist only in assistant turns. We propose Agentic Conversation Context Curation (AC3), a training-free method in which a separate analyzer (1) consolidates the task specification from the user's messages alone, (2) audits the assistant's prior outputs against it, and (3) edits the context, either appending the analysis or discarding the assistant side and rebuilding the context from the specification. A second finding: pollution is contagious. An analyzer that sees the assistant's reasoning while extracting the specification can do worse than no intervention, and instructing it to ignore them did not close the gap, so we exclude assistant turns structurally. On Lost in Conversation (replaying difficulty-selected instances), where AO is an oracle by construction, AC3 closes 55-84% of the gap between full context and AO on three of four tasks and exceeds AO on the fourth, and an LLM that faithfully summarizes the history does not reproduce the gain even with more compute. On real WildChat conversations, a GPT-5-mini judge prefers AC3's next turn over both full context and AO in every configuration of our main table, though this preference depends on the judge; on CollabLLM results are mixed. In stateful tool use (tau2-bench), AO collapses to 0% on every model; AC3 does not collapse, though it falls slightly behind full context. Detector studies show that Reset removes injected pollution but is not span-selective: the gain comes from rebuilding the specification. 

---
# To Jev or Not to Jev: When Can System-1 Decision Models Replace LLM Judges? 

**Authors**: Shreyas Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2611.00505)  

**Abstract**: Decision models such as TypeSafe's Jev answer typed questions about a text (yes/no, multiple choice, or an ordinal score) with calibrated probabilities in a single forward pass and without generating text, which makes them an inexpensive candidate for LLM-as-a-judge workloads. We evaluate 14 System-1 decision models, selected with a pre-stated protocol, against four generative judges (Claude Haiku 5.5, GPT-5.6 with and without reasoning, and Claude Opus 5.5 with thinking) on eleven established judge benchmarks. Five of the 14 are statistically indistinguishable from GPT-5.6 without reasoning on all five pairwise and best-of-4 benchmarks, and the two strongest, DReX-Max and d1-ultra, are significantly more accurate than Haiku on JudgeBench, LLMBar and RewardBench 2. Conditioned on a reference answer generated once per prompt, eleven models reach 90.3-93.4% on JudgeBench, statistically indistinguishable from Opus 5.5 and GPT-5.6 with reasoning, and on PPE best-of-16 every reference-guided System-1 judge matches GPT-5.6 with reasoning at roughly one-sixteenth of its cost per prompt. Several System-1 judges are better calibrated than every generative judge and correlate more strongly with human helpfulness ratings. Robustness to instructions injected into the judged response is a property of the individual model rather than of the System-1 design: an appended note to the evaluator reverses 0-5% of the correct verdicts of five models but 17-49% for the other nine. Two general System-1 leaderboards that contain no judging tasks predict judging accuracy (Pearson r = 0.95-0.96), a well-calibrated System-1 first stage matches Opus on LLMBar while escalating 17% of items, and same-size controls show that the correctness deficit of a 4B decision model is a reasoning gap rather than a size gap. 

---
# Robustness Starts with the Objective: Designing and Certifying Robust LLM-Generated Heuristics 

**Authors**: Pantea Karimi, Roozbeh Bostandoost, Dany Rouhana, Siva Kesava Reddy Kakarla, Pooria Namyar, Mohammad Alizadeh, Venkat Arun, Behnaz Arzani  

**Link**: [PDF](https://arxiv.org/pdf/2611.00504)  

**Abstract**: LLMs are increasingly used to design heuristics for computer systems. However, most synthesis frameworks evaluate candidates by their average performance, which can mask severe regressions on specific workloads. Even adding adversarial training inputs doesn't eliminate these regressions, and percentile objectives (e.g., optimizing for the 95th percentile) overlook how severe regressions can be beyond the selected threshold.
We present TAILOR, a framework that guides LLM synthesis using the Conditional Value at Risk (CVaR) of the cost difference between each candidate and an existing baseline on the same workload. This paired CVaR measures the average difference over the worst fraction of workloads. We show that a negative paired CVaR guarantees improvements in both average performance and tail optimality gap relative to the unknown optimum, without ever computing that optimum.
We find that TAILOR achieves the best mean--tail trade-off among the synthesis objectives we evaluate across three case studies: Azure GPU batch admission, Azure VM packing, and server placement. Tested across seven diverse optimization benchmarks, TAILOR produces heuristics that are certified with 99% confidence to improve both average performance and tail optimality gaps over the baseline in six of them. On these six benchmarks, TAILOR also outperforms known robust, human-designed alternatives in the tail while achieving better or comparable average-performance improvement over the baseline. 

---
# Swap, Replicate, Read the Mass: A Protocol for Auditing Demographic Bias in System One Decision Models 

**Authors**: Alejandro Mozo Quesada, Sergio Gálvez Rojas, Francisco López Valverde  

**Link**: [PDF](https://arxiv.org/pdf/2611.00496)  

**Abstract**: System One decision models replace text generation with typed, probability-scored answers and are entering production. Answers execute without a person in the loop, and no multi-model audit exists on established bias benchmarks. We provide that audit: three benchmarks: BBQ (all eleven categories), CrowS-Pairs, and SilverSpoon, adapted to the typed interface, twelve models plus a no-thinking ablation, and 1,638,000 decisions. Every order-sensitive item runs twice with options swapped; native typed models also swap labels; ten configurations repeat under extra seeds and batch 16. We find widespread stereotype alignment, order-dependent answers, and confidence that tracks instability rather than bias. In BBQ, 97 of 143 model-by-category scores are significant and 94 aligned, while the commercial reference abstains on 99.5% of ambiguous items, hiding aligned probability mass. Option swaps flip 0.0% to 72.6% of choices; typed scoring stays below 10% except Laya (22-33%), and untuned letter prompting reaches 73%. CrowS-Pairs preference spans 0.496 to 0.816, and boolean answers can be decided by position (1.00 versus 0.005). SilverSpoon judgments shift toward lower-SES labels in five of eight decision models, inherited from the 9B backbone or induced by tuning. Audits must permute options and reverse questions by default; order-invariant scoring should become a procurement requirement. 

---
# Few Trials, Many Metrics: Beta Mixtures for Low-Budget AI Evaluation 

**Authors**: Gustavo Tironi, Felipe Maia Polo, Diego Mesquita  

**Link**: [PDF](https://arxiv.org/pdf/2611.00474)  

**Abstract**: Evaluating a stochastic AI system often requires many repeated samples per instance, especially for metrics defined over long horizons. Recent work has shown that modeling instance-level success probabilities with a Beta distribution recovers Pass@$k$ from far fewer samples than direct estimation. We show that this framework supports a broader range of evaluation targets. We fit a mixture of Beta distributions by marginal maximum likelihood, with the number of components chosen by BIC. From that one fit, we compute Pass$^k$ and AUC@$k$ at any horizon in closed form. We also derive delta-method confidence intervals for these metrics and for Pass@$k$, which require no additional model calls. Treating configurations rather than tasks as instances extends the same model to sensitivity analysis. For repeated success metrics, we find that mixture priors yield more accurate estimators than naive estimation across competition mathematics and graduate-level science benchmarks when extrapolating beyond the observed number of trials, and the resulting intervals cover the full-data reference. When prompts are treated as instances, it estimates the distribution of prompt sensitivity more accurately than a comparable item-response baseline without external covariates. Direct estimators remain preferable once trials reach the target horizon. 

---
# Compression Technique Interactions in Small Language Models: A Factorial Ablation Study 

**Authors**: Bhushit Nema, Siddharth Krishnamurthy, Anjali Kadiyala  

**Link**: [PDF](https://arxiv.org/pdf/2611.00464)  

**Abstract**: Deploying small language models on resource-constrained hardware requires compression, yet practitioners lack systematic guidance on how quantization, pruning, and post-compression recovery interact when combined. We present a factorial ablation study evaluating 72 compression configurations across four SLM architectures (2.7B-7B parameters), extended with a 48-configuration ordering analysis, benchmark accuracy validation on 30 compressed models, and robustness checks including multi-seed validation and full-corpus perplexity evaluation. INT8 quantization is effectively lossless (<0.5% accuracy degradation), while NF4 quantization combined with L1 magnitude pruning causes model collapse in all 16 tested combinations. LoRA-based fine-tuning reduces degraded perplexity by up to 97% after aggressive pruning; an ablation against standard fine-tuning without a teacher model shows the recovery is driven by LoRA adaptation itself rather than the distillation signal. Compression ordering shows no significant effect among viable configurations (ANOVA p = 0.96) but determines viability for NF4+pruning combinations. INT8 quantization alone provides 40-45% memory reduction with no quality loss. On-device CPU deployment via this http URL confirms these findings transfer off the GPU: 4-bit quantization runs 46-54% faster than 8-bit for a perplexity difference under 0.5 points, while un-structured pruning yields no CPU latency or memory benefit, consistent with its dense storage format. 

---
# Exo: An Architecture for Self-Improving Systems at Runtime 

**Authors**: Alexander Krentsel, Sylvia Ratnasamy, Ion Stoica, Martin Casado, Ankur Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2611.00463)  

**Abstract**: Agent harnesses govern how language models act in an environment; designing their policies by hand is difficult, while automated techniques require representative datasets and costly executions. We present Exo, an agent runtime architecture for safe learning during ongoing work that collapses the optimization loop into task execution, allowing agents to revise both their harness and their strategies for further improvement. A minimal protected mechanism layer preserves execution history and governs policy activation while the harness evolves through inspectable code changes. On long-running objectives in an open-world game, Exo reduces model cost by 12-53% relative to Codex and Pi in standalone runs and 45-82% in transfer runs, while matching task completion rates. Compared with the same starting policy without the ability to edit policy, Exo reduces held-out model cost by 35-61% on AppWorld and SREGym with comparable correctness, cutting AppWorld's mean task time by 13% and SREGym's median diagnosis time by 35%. On Continual Learning Bench coding tasks, model cost including learning falls by 14%, with both policies resolving every issue. 

---
# From Small to Large: Assessing Medical Reasoning, Benchmark Contamination, and Model Size in Language Models 

**Authors**: Navapat Nananukul, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2611.00460)  

**Abstract**: Large language models (LLMs) now answer medical exam questions at or above expert level, but it is unclear whether such scores reflect clinical competence, or are instead affected by the format of the questions being asked (e.g., multiple-choice versus open-ended), prior exposure to the benchmarks, and the difficulty of the questions. These benchmarks are also public and widely used online, which may be resulting in LLM memorization. Models may perform better on questions that they have seen before. It is also not evident whether smaller models, which are more locally deployable in HIPAA-compliant hospital settings, have achieved human performance. In this study, we evaluate fourteen current models from six leading providers, one smaller and one larger model from each provider with two additional open-weight models, on 3,076 questions from four medical benchmarks (MedQA-USMLE, MedMCQA, PubMedQA, and MedBullets) under standard multiple choice, the same questions with the answer choices withheld, and reconstruction of a hidden question from its choices alone. In the standard multiple-choice setting, the strongest models exceed the human-expert baseline on MedQA-USMLE and PubMedQA, but the smaller models trail their larger models by 8.7 percentage points and only two of the six pass the human-expert bar on the licensing examination. Withholding the answer choices, however, makes every model struggle: mean accuracy falls from 82.3% to 50.9%, and the two largest drops belong to the medically specialised models. On memorization, the strongest models reconstruct one question in four from the answer choices alone, yet accuracy holds on the questions they cannot reconstruct. We further weigh accuracy against inference cost and test whether ensembling several models improves medical question answering. 

---
# Quantiles, Not Points: Streaming Quantile Matching for Ordinal-Aware LLM-as-a-Judge Calibration 

**Authors**: Ilija Subasic, Andrew Rabinovich, Zhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.00447)  

**Abstract**: LLM-as-a-judge scores are often distributionally shifted from human ratings on any given evaluation task, and post-hoc calibration (mapping LLM to human scores in a small labeled anchor set) has become a popular method to correct for this shift. In this paper we derive the calibration from two positions: that LLMaJ scores are ordinal ratings drawn from a distribution rather than isolated points. We present streaming quantile matching (SQM), a version of a classical method not previously bench marked for LLMaJ calibration, which is built on the two positions by computing the optimal-transport map between the LLM and human score distributions. We develop an evaluation framework and within it we show that unlike point-wise calibrations, SQM respects the ordinality of the human anchor set and achieves the best distributional fidelity and the top overall rank across models and data. We also show that in general, post-hoc calibration narrows the raw error gap between smaller and larger judge models, which opens the possibility of more efficient LLM judges. 

---
# OmniMechanism Design for Human-AI Collaboration with Impressionable Minds 

**Authors**: Tianwei Gao, Yifei Pang, Lunjia Hu, Kate Donahue, Lihua Lei, Vahid Tarokh, Zhun Deng  

**Link**: [PDF](https://arxiv.org/pdf/2611.00441)  

**Abstract**: As AI systems are increasingly deployed in high-stakes domains such as healthcare, effectively integrating human expertise with AI outputs is critical for reliable decision-making. A substantial body of theoretical work studies optimal mechanisms for human-AI collaboration (HAC), focusing on when to involve humans and how to combine their expertise with AI predictions. However, these approaches largely overlook a key factor highlighted by empirical evidence: human judgments are impressionable and can vary depending on how AI outputs are presented. For example, when AI systems defer decisions to humans, revealing or withholding the AI prediction can significantly influence human judgments. Moreover, existing work neglects deployment-time variability, where conditions, particularly the cost of eliciting human input, may differ substantially from those observed during training. To address these challenges, we introduce a rigorous and theoretically grounded framework for human-AI collaboration, termed collaborative omnimechanism design, based on a novel and computationally efficient contextual bandit variant of boosting for outcome indistinguishability. In particular, we reduce a calibration-type outcome indistinguishability condition to weighted calibration, providing a novel time-efficiency result, bypassing the high-dimension curse induced by complicated collaboration methods and outcome space. Our framework provides a principled characterization of human impressionability and introduces a unified optimization scheme that enables robust and adaptable decision-making across diverse query costs and evaluation objectives without retraining. We evaluate the framework in three HAC settings, examining adaptation to query costs and prediction objectives without retraining, and show that its induced policies perform comparably to or better than the evaluated baselines. 

---
# Nonconformal Distortion and Local Orientation Reversal in the Fourier Circuit of a Modular-Addition Transformer 

**Authors**: Eduardo Fuentes, Gonzalo Riera, Héctor Allende-Cid  

**Link**: [PDF](https://arxiv.org/pdf/2611.00414)  

**Abstract**: We study the local differential geometry of a Fourier-based modular-addition circuit in a one-layer transformer whose algorithm has previously been reverse-engineered. For each of the five key Fourier frequencies, we restrict the model's multilayer perceptron (MLP) contribution to a corresponding Fourier input plane and readout, and measure its directional Beltrami coefficient under continuous input perturbations. Whereas nondegenerate restrictions of ideal complex multiplication are conformal, the trained maps exhibit nonconformal distortion (median $|\mu|=0.304$--$0.446$), spatial organization in the Beltrami argument, and regions of local orientation reversal. The median across frequencies decreases from $0.939$ at initialization to $0.390$ after training. The key-frequency planes also show lower median distortion than most random-plane controls, although their distributions overlap and the coordinate frames differ in conditioning. Across 500 checkpoints in one training run, most of the aggregate reduction in Beltrami magnitude occurs before the principal increase in the measured Fourier-circuit formation. At the individual-input level, Beltrami magnitude shows negligible association with the correct-logit margin ($|\rho|\leq0.026$). Late-checkpoint measurements across six runs show comparable distortion for most selected frequencies. These results provide a quantitative characterization of local deformation in a frequency-projected component of an identified neural circuit, complementary to its algorithmic reconstruction. 

---
# Hacking or Just Failing? A More Capable LLM Monitor Better Separates Reward Hacking from Honest Failure 

**Authors**: Kaushik Kezendirane  

**Link**: [PDF](https://arxiv.org/pdf/2611.00407)  

**Abstract**: An agent's visible score can exceed its real progress because it gamed the score or because it simply made mistakes, so a useful reward-hacking monitor must tell the two apart. We built a sandboxed task in which an agent can game a completion score, labelled every episode by replaying it and repairing the final state counterfactually, and ran five pre-registered studies. On 60 episodes from a Llama 3.1 8B agent, small local monitors (Llama 3B and 8B) did not separate exploit attempts from honest failures with the same score gap: prompt changes traded missed exploits for false alarms, and the pre-registered primary monitor reached AUC 0.64. On identical inputs, Gemini 3.1 Flash-Lite, which completed 9/10 clean tasks against 2/10 for the 8B model, met our pre-registered criterion (AUC 0.80, 95% interval [0.66, 0.93]; 1/35 honest failures flagged); with only 14 exploit episodes, this estimate is uncertain. On an exploit that no prompt names, Gemini usually described the action correctly yet never flagged it, in 5 and then 25 new episodes, even when manipulation was defined by its effect on the score. Given the full trace, the 8B monitor flagged all 25 replication episodes and none of 25 honest look-alikes, but its explanations described actions that had not occurred; given additional structured evidence, it also flagged 15 of the 25 look-alikes. A two-rule deterministic check caught every known exploit. We release the environment, labels, pre-registrations and per-judgement logs. 

---
# CalVLA: Shaping Native Confidence in Vision-Language-Action Policies with Outcome Supervision 

**Authors**: Boniface Chukwuemeka Godwin  

**Link**: [PDF](https://arxiv.org/pdf/2611.00394)  

**Abstract**: Can a vision-language-action (VLA) policy signal likely task failure from its own action probabilities? We introduce CalVLA, which shapes native action-token confidence using successful and failed rollouts and an Action-Confidence Alignment (ACal) squared margin-ranking loss. Low-rank adapters update OpenVLA-7B without an auxiliary failure detector. Observations are paired within tasks at similar normalized elapsed times, not necessarily equivalent physical progress. On a held-out test split from 500 LIBERO-10 configurations, adapted confidence achieves a joint failure-discrimination AUROC of $0.7392\pm0.0480$ over three seeds, averaging scores at 130 and 260 steps. Frozen-policy probes reach 0.6146--0.6436, but superiority over the linear probe is statistically inconclusive. For the same confidence summary, AUROC rises from 0.4485 before adaptation to 0.7327 after adaptation. Mean task success decreases from 58% to 55%. These results show improved failure discrimination, not improved control or established safety benefits. 

---
# Dependence-Aware Risk Assessment and Optimal Design of AI-Agent Safeguard Systems 

**Authors**: Yijia Li, Maochao Xu, Peng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.00386)  

**Abstract**: Large language model (LLM) agents increasingly rely on safeguards to prevent unauthorized actions while preserving legitimate service. However, existing risk assessments often focus on individual safeguard performance while overlooking dependence among safeguard decisions. This paper develops a dependence-aware framework for risk assessment and optimal design of multi-component AI safeguard systems, jointly selecting configurations and operating policies for sequential layered screening and selective verification. We establish stochastic comparisons of layered-screening error probabilities and invocation costs under lower-orthant ordering. For selective verification, we characterize exactly which policies have error probabilities and review costs that respond monotonically to stronger dependence under supermodular ordering. For a fixed first stage and reviewer, we show that the optimal routing map is a two-threshold rule in the first-stage likelihood ratio. An accompanying open-source software package, Dependence-Aware Policy Optimization Algorithms (DAPOA), facilitates the practical implementation of the proposed framework. Numerical experiments show that ignoring dependence can lead to inaccurate risk assessments and infeasible safeguard designs, with independence-based selections violating security requirements in six of eight scenarios. An empirical study using four LLM safeguards on benchmark-derived agent actions further demonstrates the practical implications of dependence modeling. Under the principal operating setting, dependence-aware selection lowers held-out total cost by 39.6% relative to independence-based selection. 

---
# Efficient NPN Classification for Large Functions using Graph Neural Network 

**Authors**: Chengdi Cao, Hanyu Wang, Wan-Hsuan Lin, Jason Cong  

**Link**: [PDF](https://arxiv.org/pdf/2611.00339)  

**Abstract**: Two Boolean functions are considered NPN equivalent if one can be transformed into the other by negating inputs, permuting inputs, or negating the output. Efficient NPN classification is valuable in fields such as logic synthesis, quantum computing, and information theory. However, exact NPN classification demands exhaustive enumeration of all possible input permutations and negations. Its complexity grows exponentially, making it impractical for larger functions with more inputs. In this paper, we introduce the first learning-based formulation for NPN classification by recasting it as NPN action prediction. Our method represents Boolean functions as AND-Inverter Graphs (AIGs) and applies a bidirectional heterogeneous graph neural network to predict input negations, output negation, and input permutations. To mitigate the non-canonicity of logic networks, we further introduce graph rewriting techniques that reduce structural ambiguity and recover useful symmetries before prediction. Experimental results show that NPN action prediction is learnable, that the resulting canonicalization quality improves over heuristic baselines, and that on practical benchmarks our method approaches exact methods on tractable cases while scaling to functions with up to 16 inputs. 

---
# Beyond Semantic IDs: Atomic Creator Tokens for Large-Scale Generative Recommendation 

**Authors**: Zhuosheng Liu, Yihan Wu, Ziyun Xu, Xiaoxuan Liu, Longfei Yun, Kevin Han, Mingze Gao, Pengfei Wang, Junfeng Pan, Wenjie Fu, Haoran Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.00323)  

**Abstract**: Recommending creators that a user does not yet follow is a cold-start problem: from hundreds of thousands of popular creators and only a short trail of prior engagements, the system must predict the next creator worth surfacing. The prevailing generative recommendation recipe tokenizes each creator into a \emph{Semantic ID} (SID) by quantizing a frozen embedding and trains a language model to generate SIDs. We show that this SID layer, not the language model, is the binding constraint. The creator-to-SID map is many-to-one, so colliding creators must be resolved by a popularity tie-break that caps creator-level recall and favors already-popular creators, and the frozen quantizer stops the training gradient from reaching the creator representation. We propose Atomic Creator Token (ACT), which gives every creator its own vocabulary token trained end-to-end. The map is bijective, so collisions are impossible and inference is a direct top-$k$ over creator logits. In a controlled comparison on production data in which the token layer is the only variable, ACT improves creator-level Recall@15 by 19.8 points ($1.84\times$) over a matched SID baseline, still leads when the SID codebook is enlarged $64\times$, and serves at ${\sim}2.7\times$ lower per-user latency. The advantage carries over to the public KuaiRand dataset, where ACT is the only method that beats a global-popularity baseline, provided each creator token sees enough training examples. In a live A/B test, ACT is directionally positive on every reported metric against both the SID-enhanced and the classic production rankers. 

---
# Decentralized Multi-Agent Coordination and Communication in Shared Codebases 

**Authors**: Rishabh Mediratta, Lingyun Xiao, Peter Stone, Joydeep Biswas, Elias Stengel-Eskin  

**Link**: [PDF](https://arxiv.org/pdf/2611.00314)  

**Abstract**: Coding agents working in shared repositories may receive separate, independently-motivated requests from human developers, creating a setting that is multi-agent by necessity. When this scenario happens, their individual implementations may work separately but conflict when combined. To study this setting, we introduce ParallACT, a benchmark of orchestrator-free multi-agent coding tasks. ParallACT contains both complementary and contradictory tasks. In complementary tasks, each agent's task is separately useful and implementations must share an interface to be productively combined. In contradictory pairs, both agents' tasks require editing the same code in incompatible ways, and naively combining independent solutions fails to produce a useful implementation. We compare a single-agent oracle given the combined set of tasks against multi-agent settings that vary execution order, visibility, awareness of other agents, and access to a communication channel. We find that the oracle solves most tasks while parallel agents perform substantially worse. Their performance drop is notable for complementary tasks, and particularly sharp for contradictory tasks (e.g., from 62% to 16% for Claude Code with Claude Opus 4.8) where merged code typically satisfies only one specification, and the same gap appears in all six further configurations pairing two agent harnesses with three open-weight models. Simply telling an agent that a peer exists does not help significantly, and giving agents access to a communication channel recovers only a small part of the performance, at an additional token cost. ParallACT thus identifies and isolates a core challenge that will become more common as multi-agent coding by independent principals becomes standard, and provides a controlled test of which coordination strategies help, and at what cost. 

---
# Quantifying Local and Global Fidelity of Nonlinear Embeddings 

**Authors**: Henry Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.00290)  

**Abstract**: Dimension reduction methods are chosen by a common claim: t-SNE and UMAP preserve local structure, and PCA preserves global structure. Checking it on given data takes local and global scores on one scale and a test that compares two maps. Both scores come from one curve. The locality profile is the chance-corrected share of each point's k nearest neighbors that a map keeps, over all neighborhood sizes. Its value at k = 10 is the local fidelity L, and its mean over 30 to 50 percent of the data is the global fidelity G. We prove that at the midpoint the profile is the agreement on which half of the data is near each point, a rank correlation of the distances. Because L and G are one statistic, two maps can be tested against each other. The LG reversal test applies to any two dimension reduction methods. It decides, with a block-jackknife p-value, whether one map is more local and the other more global, and returns the scale at which to change maps. On 156 maps of twelve datasets with up to 1.38 million points, G agrees with established global scores at 0.89 and 0.96, and 2,000 points estimate it. The test confirms the usual claim on seven datasets. On both retail datasets and both rolled-up manifolds, it finds the neighbor embeddings more global. In an agent memory of one million passages, the two maps cross at 150,000 passages (p = 2.6e-5). Applications span pathology, single cells, supply chains and AI agent memory. 

---
# Automated Theorem Generation in First-Order Logic by Construction from Triangular Standard Contradictions 

**Authors**: Yang Xu, Shuwei Chen, Hailin Guo, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.00269)  

**Abstract**: Automated theorem generation (ATG) seeks to construct new theorems rather than merely prove given conjectures. Existing approaches typically generate candidate statements that require subsequent proof or formal verification. We propose a constructive framework for ATG in first-order logic based on triangular standard contradictions. The method constructs minimally unsatisfiable clause sets from selected first-order literals, then forms theorems by retaining some clauses as premises and taking the negation of one clause, or a conjunction of clauses, as the conclusion. Minimal unsatisfiability ensures that the remaining premises are satisfiable, avoiding trivial inference by explosion, while unsatisfiability of the full set guarantees the resulting entailment. Theorems are therefore correct by construction, without a separate proof procedure for each instance. Extending the previously developed $\Delta_1$ generator, we introduce seven methods, $\Delta_2$--$\Delta_8$, forming a unified family of eight theorem-generation methods. Four generators, $\Delta_1$--$\Delta_4$, have been implemented and released. We also formulate a general $\Delta$-class method encompassing the family. Combinatorial analysis demonstrates the capacity to generate large numbers of distinct theorems systematically. This framework offers a principled foundation for automated theorem discovery and the construction of logically guaranteed knowledge. 

---
# A generative foundation model for computational brain imaging 

**Authors**: Yuting He, Xu Wen, Hongxing Lan, Chengliang Liu, Yuxing Liu, Yan Xi, Wenxue Yu, Jindi Kong, Zhiqiang Zhang, Liang Jiang, Rongjun Ge, Yang Chen, Multi-Center Cooperative Clinical Study Group for "Lesion-Phenotype Brain Mapping in Acute Stroke"  

**Link**: [PDF](https://arxiv.org/pdf/2611.00265)  

**Abstract**: Computational brain imaging is constrained by a mismatch between increasingly diverse tasks and scarce, heterogeneous clinical data. Although existing brain foundation models transfer pretrained representations, their knowledge remains entangled within model-specific features and fixed architectures. This paper proposes BrainGFM, a generative foundation model that extends knowledge transfer beyond representation reuse through a reusable and reconfigurable brain-imaging prior. By pretraining on a corpus of about 1.3 million MRI and CT slices, BrainGFM supported task-aligned data synthesis for low-data predictive learning and task-adapted post-training for image reconstruction. Across 48 downstream tasks, BrainGFM improved segmentation and classification under limited supervision and achieved high-fidelity image enhancement across heterogeneous imaging conditions. Multicenter clinical validation involving 4,796 stroke patients across two internal and four external centers demonstrated its applicability to three clinically relevant tasks around stroke cohorts. Blinded reader assessments showed that BrainGFM-generated FLAIR images approached acquired FLAIR in overall diagnostic quality. In a human-AI collaboration study, BrainGFM assistance yielded a 14% relative improvement in mean radiologist stroke-subtyping accuracy. These findings support generative priors as a reusable foundation for heterogeneous computational brain-imaging applications. 

---
# Taurus: LLM-in-the-Loop CT Annotation Matches Coding Agents at Half the Cost 

**Authors**: Bing Hu, Rishabh Sharma, Jatin Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2611.00247)  

**Abstract**: Dense labels of the aorta in CT are slow to draw, and the public aortic CT datasets that could stand in for them each label different structures under their own conventions. Large language model (LLM) agents can now write code and read images. We asked whether such a model labels these scans better as a free-form agent that writes its own segmentation code, or inside a harness, where an automated pass labels the scan and the model reviews the result. We built a harness, Taurus, and compared it with free-form agents by blind label recovery: we withheld the human labels of six public datasets (939 volumes) and scored every approach against each dataset's own classes. In the first version of Taurus, the model's review raised the primary metric on only two of six datasets, by less than the noise between two reviews of the same case. A free-form agent running Claude Opus 5.5 scored higher on most datasets. Moving what the free-form agents did well into the automated pass then carried most of the gain on 38 held-out cases. The final version smooths the label surfaces with a fixed rule and checks for the large errors the automated pass tends to make, such as an aneurysm drawn to half its size. Each check that fires puts a multiple-choice question to the model, with candidate corrections drawn on the scan. It matched the surface smoothness of the human references, and in the primary metric we found no difference from the free-form Opus agent, at under half its cost, $1.22 against $2.71 per case. Most of what made Taurus competitive came from the harness and its automation rather than from the language model itself. Taurus shows that this kind of annotation can be automated, so that expert time can move from drawing labels to checking them. 

---
# OmniPiano: Diverse Dexterous Piano-Playing Challenges for Standard, Robust, Safe, and Multi-Agent RL 

**Authors**: Fan Xu, Fan Yang, Yifan Xu, Ruiyuan Zhang, Yimou Wu, Nuo Yi, Muning Wen, Guang Chen, Costas Spanos, Shangding Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.00243)  

**Abstract**: Dexterous robotic manipulation remains a major challenge in robotics, exemplified by piano playing, which requires coordinated control of fingers and provides a demanding testbed for reinforcement learning (RL). Although previous work established a benchmark for robotic piano playing, it is restricted to two hands and does not support unified evaluation. To address this gap, we introduce OmniPiano, a benchmark that provides one to up to five Shadow Hands and supports standard, robust, safe, and multi-agent RL within a shared piano-playing task family. OmniPiano is developed for scalable and diverse piano-playing tasks with configurable perturbations, explicit safety constraints, and decentralized cooperation settings. Particularly, to facilitate usability and extendability, OmniPiano adopts a highly modular design and provides comprehensive tasks to support RL study of task performance, robustness, safety, and cooperation in dexterous control. With at least 912 task settings and 36 baseline algorithms, OmniPiano further incorporates LLM-based agents to broaden the evaluation scope and reveal new insights. Extensive evaluations show that state-of-the-art RL algorithms and frontier LLM-based agents still struggle on challenging piano-playing tasks, even under standard RL settings, highlighting substantial room for improvement in dexterous control. The code, dataset, tutorial, and demo are available at this https URL. 

---
# RE-SciVQA: Improving Scientific VQA With Context-Relevant Retrieval and Lightweight MLLM Adaptation 

**Authors**: Sheikh Tahmim Mehdi, Muminul Islam Jami  

**Link**: [PDF](https://arxiv.org/pdf/2611.00242)  

**Abstract**: A deep understanding of scientific literature often relies on interpreting figures that contain key information beyond the main text. Leveraging Large Language Models (LLMs) to extract insights from these figures can significantly accelerate knowledge discovery. Existing works on Scientific Visual Question Answering rely on resource-intensive training pipelines that are often inaccessible and require large computational resources. Instead of designing complex architectures, we propose RE- SciVQA, which focuses on improving the quality of retrieved context and fine-tuning small LLMs with parameter-efficient LoRA training. Using a simple query expansion step, relevant section headers are appended to the original question to improve the retrieval coverage. In addition, a simple adjustment of similarity scores is applied using figure references and caption alignment. On the SciVQA benchmark, RE-SciVQA achieves a ROUGE-1 F1 score of 73.9%, outperforming chart-focused models such as UniChart (14.7%), ChartGemma (6.9%), and Matcha-ChartQA (6.3%). Our relevance-enhanced RAG setup also improves the exact match by 11.9% and the fuzzy match by 29.5% compared with a standard RAG baseline, resulting in an overall gain of 14.3%. We also examine closed-ended infinite visual questions, a category in which chart-understanding models often struggle. Our experiments show that using plot digitization instead of synthetic captions leads to an 18.5% improvement for this question type. 

---
# From Cross-Section to Cross-Omics: Conditional Cell Nucleus Image Editing in Spatial Transcriptomics 

**Authors**: Yuechen Yang, Junchao Zhu, Junlin Guo, Yanfan Zhu, Ruining Deng, Zhengyi Lu, Tianyuan Yao, Chongyu Qu, Marilyn Lionts, Xingyi Guo, Yaohong Wang, Yu Wang, Shilin Zhao, Haichun Yang, Yuankai Huo  

**Link**: [PDF](https://arxiv.org/pdf/2611.00226)  

**Abstract**: Spatial transcriptomics (ST) platforms simultaneously measure gene expression and capture cell nucleus images at single-cell resolution. A fundamental property of tissue preparation introduces a confounding relationship between the two measurement modalities: sectioning a three-dimensional tissue into a thin slice produces a two-dimensional cross-section whose area depends on the cutting angle. Larger cross-sections expose more cytoplasm, yielding higher mRNA counts, while pathomics features such as nuclear area are measured directly from the same 2D boundary. This sectioning geometry creates an apparent correlation between gene expression and pathomics that reflects the measurement process rather than independent biology. Existing gene-to-image methods generate cell images from gene features but do not support condition-driven editing of existing images across both omics layers. We apply the Manifold-Aware Diffusion (MAD) framework to this dual-omics editing setting, enabling targeted modification of one feature group while preserving the other. Evaluated on podocyte cells from CosMx SMI kidney tissue, MAD achieves MAE=0.6636 and FID=25.19 for pathomics editing, outperforming all compared baselines. A controlled simulation study further confirms that the VAE component in MAD better preserves cross-group independence than CCDM (MAD without VAE). 

---
# MemU-Bench: Benchmarking GUI Agents on Long-Horizon User Interactions 

**Authors**: Yizhou Liu, Zhengxi Lu, Fei Tang, Yiwen Qiu, Yong Du, Tongbo Chen, Bofan Chen, Tao Jiang, Wenhao Xu, Yunfeng Shu, Wenqi Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.00191)  

**Abstract**: Existing Graphical User Interface (GUI) benchmarks predominantly rely on precise and stateless user instructions. In everyday usage, however, instructions are often \emph{vague and history-dependent}: users omit the exact details of their past operations because they no longer remember them, leaving the agent to recover the missing context from long and noisy interaction histories. To bridge this evaluation gap, we introduce \textbf{MemU-Bench}, a mobile benchmark of \textbf{118 vague tasks} grounded in 29 long-horizon user trajectories, where agents must distill raw interaction streams into memory and retrieve the missing context on demand, with task outcomes verified against serialized application databases. Alongside Success Rate, we incorporate Interaction Efficiency (IE) and Success weighted by Path Length (SPL) to expose agents that succeed by exhaustively searching across different applications rather than by \emph{genuinely recalling} the history. Evaluations show that without user history, state-of-the-art systems struggle under such ambiguity: the best-performing Gemini-3.1-Pro achieves an SPL of only 16.4\%, and end-to-end GUI agents fall below 5\%. To address this, we further propose \textbf{AppMem}, a training-free framework that distills user history into a \emph{structured, app-centric} memory bank, tracks interrupted intents across application switches, and selectively retrieves relevant memories via target-app inference. AppMem raises the SPL of GPT-5.5 from 15.7\% to 66.9\% (\textbf{+51.2\%}), outperforms a naive summary baseline by 9.3\%, and delivers consistent gains across evaluated backbones. 

---
# Neurons Are All You Need 

**Authors**: Joseph B. Alzagatiti, Alain C. Glanzman, Perry A. Demsko, Pablo I. Salanova, David L. Glanzman  

**Link**: [PDF](https://arxiv.org/pdf/2611.00186)  

**Abstract**: For decades, artificial neural networks have operated under the assumption that durable, learned capability resides exclusively in adaptive synaptic parameters, primarily due to the established neurobiology of synaptic connectivity. This design forces models to remain stateless during deployment, incurring well-known limitations such as catastrophic forgetting and expensive retraining. However, recent neurobiological research by Glanzman and colleagues, and others, shows that memories can be transferred between individual organisms in a synaptically independent manner. Inspired by this research, we sought to demonstrate memory transfer of persistent memory between AI model instances, leveraging a neurocentric version of memory. In this paper, we propose NURESA (Neural Unit with Retentive Self-Adaptive Architecture), a novel neural layer that eschews sole reliance on synaptic weight updating by introducing persistent intra-unit memory as a first-class, computational primitive. By structurally separating computation from persistent memory, NURESA enables autonomous memory accumulation and consolidation into transferable stores of memory registers. Experiments across MLP, GPT-2, and modern frontier models like Qwen2.5 demonstrate qualitatively new capability: zero-shot transfer of learned capability into a naive model instance without retraining or access to the original training data; this transfer is enabled by a sparse set of parameters in NURESA's memory register that the model selects on its own based on which neurons were most active. The memory transfer capability of NURESA also protects against domain loss when transformer models undergo learning via fine-tuning. Finally, the memory transfer process completes in approximately one minute even for transformer models with millions to billions of parameters, producing a transfer file orders of magnitude smaller than the original model. 

---
# AuroraWorld: Structured 4D World Modeling for Exo-to-Ego Video Generation in Robot Manipulation 

**Authors**: Hao Xue, Peng-Fei Zhang, Ying Cheng, Lanyun Zhu, Lei Zhu, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.00177)  

**Abstract**: Egocentric observations provide a robot-centric view of manipulation processes, capturing contact interactions and object-state changes that are important for visuomotor policy learning but difficult to observe from external cameras. However, collecting such data at scale remains expensive due to additional hardware requirements and platform-specific deployment costs. Synthesizing egocentric observations from large-scale exocentric manipulation videos provides a scalable alternative. However, existing methods struggle to synthesize physically consistent and policy-relevant egocentric videos under substantial viewpoint transformations, severe occlusions, complex object interactions, and inherently sparse observations from available exocentric viewpoints. 

---
# Reimagining Text-to-SQL: A Survey of Large Language Model Architectures and Strategies 

**Authors**: Shikhar Dave, Bhaskarjit Sarmah, Stefano Pasquali  

**Link**: [PDF](https://arxiv.org/pdf/2611.00171)  

**Abstract**: Text-to-SQL systems translate natural language questions into executable SQL queries, democratizing access to relational data. Early approaches relied on handcrafted rules and template matching, while subsequent neural methods introduced encoder-decoder architectures, schema linking, and constrained decoding to handle increasingly complex queries. The emergence of large language models (LLMs), exemplified by systems such as ReFoRCE, Agentar-Scale-SQL, Snowflake's Arctic-Text2SQL-R1, and the modular XiYan-SQL framework, has dramatically advanced the field, pushing execution accuracy to new state-of-the-art levels on challenging benchmarks such as Spider, Spider 2.0, and BIRD. In parallel, modular frameworks decompose the task into sub-problems (schema retrieval, sketch-based generation, repair, and ranking), while agentic systems coordinate multiple specialized agents that communicate and dynamically allocate sub-tasks. This survey systematically reviews these LLM-driven approaches within a unified taxonomy of five paradigms (in-context learning, supervised and parameter-efficient fine-tuning, agentic and tool-augmented orchestration, retrieval-augmented generation, and reinforcement learning) and characterizes them by model granularity (monolithic vs. modular), level of supervision (zero-shot, fine-tuned, or reinforcement-based), and reasoning style (direct decoding vs. decomposed multi-step reasoning). We further analyze evaluation metrics, datasets, and open challenges, including domain generalization, efficiency, and real-world deployment, and we examine the role of the semantic layer in production systems. As LLMs increasingly access sensitive databases, we also discuss responsible AI practices. We survey over 100 papers and technical resources, focusing on work published between 2022 and 2026. 

---
# PatchScout: Thematic Web Data Collection via Information Foraging 

**Authors**: Michael West, Eduard Dragut  

**Link**: [PDF](https://arxiv.org/pdf/2611.00169)  

**Abstract**: Researchers often need web corpora that are not answers to a single query, but reusable collections of documents spanning many decentralized sources. We refer to this problem as thematic web data collection: assembling semantically relevant documents that share a common theme but are scattered across heterogeneous and structurally disconnected regions of the web. Existing approaches, including traversal-based methods and LLM-driven web agents, are limited by local exploration or short-horizon stopping, preventing comprehensive collection. We formulate this task as a long-horizon information foraging problem over query-induced semantic patches, where the central challenge is deciding when to continue local exploitation and when to switch to new regions. We propose PatchScout, a multi-agent framework that instantiates a class of patch-switching policies to balance local exploitation and adaptive patch switching under a fixed budget. Experiments on four live-web topics show that PatchScout substantially improves yield and domain coverage over traversal-based and search-augmented reasoning agent baselines. In an applied social science setting with an incomplete prior collection, PatchScout discovers 102 previously unidentified entities, illustrating its ability to expand existing thematic datasets. 

---
# Early Stopping in Large Reasoning Models: A Survey and Empirical Audit of Efficiency and Reliability 

**Authors**: Quang Hung Nguyen, Hanh Nguyen, Thanh Tam Nguyen, Quoc Viet Hung Nguyen, Trinh Pham, Thanh Trung Huynh  

**Link**: [PDF](https://arxiv.org/pdf/2611.00166)  

**Abstract**: Large reasoning models can improve accuracy by generating longer reasoning traces, but some of this computation may be unnecessary. Early-stopping methods aim to stop once enough reasoning is available to answer. Evaluating these methods requires accounting for the cost of deciding when to stop and the reliability of the resulting answer. We evaluate different methods across five reasoning models on mathematical and scientific reasoning, code generation, and factuality tasks by using a shared rollout replay protocol. We introduce Target-Integrated Token Efficiency (TITE) to measure efficiency relative to the oracle across accuracy targets. We also examine correctness at later exit positions, transfer across models and tasks, post-stop reasoning continuation, and retained justification quality. In our experiments, correctness can decrease at later positions beyond what the fitted decoding-noise model predicts. Reliability checks also reveal continued reasoning after forced termination and weaker supporting reasoning in some correct early-stopping outputs. The implementation can be found in this https URL. 

---
# Learning to bid and bundle: a posterior-sampling agent for online retail 

**Authors**: Hadi Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2611.00165)  

**Abstract**: An online retailer launching a perishable product bids for sponsored-search clicks, discounts a bundle with an always-stocked staple and displays its remaining stock, which raises willingness to pay. It knows neither how clicks respond to its bid nor how strong and dispersed its market is, and learns both over repeated seasons from data its own decisions censor: a zero bid reveals nothing about paid clicks, and choices follow only clicks. We design a posterior-sampling agent for it. Each period it draws a market from an exact posterior and posts the bid and discount of that market's exactly solved season plan, so every action is one a fully informed seller could take. In a simulated online florist it has the lowest regret averaged over markets. In weak markets it gives up 12% to 23% of one season's revenue over 500 seasons, four to seven times less than drawing once per season. A planner acting on point estimates can stop bidding and never learn that bidding pays. The agent escapes after about three seasons on average, against fifty with one draw per season. Where displayed scarcity keeps willingness to pay above the posted price, the discount needs no learning. Customers waiting for the discount to return to its recent level look like lost sales, so frequent waiting makes the agent underrate its market. One discount per season removes waiting at a cost of 5.9% of clairvoyant revenue in a weak market and none in a strong one. 

---
# A Generalized Sparse Fuzzy Clustering Framework with Automated Cluster Cardinality Estimation Ability 

**Authors**: George Stamatelis, Konstantinos Koutroumbas  

**Link**: [PDF](https://arxiv.org/pdf/2611.00159)  

**Abstract**: Fuzzy clustering provides a robust framework for handling ambiguous and overlapping data boundaries, yet its traditional formulation generates entirely dense membership assignments that limit noise resilience and prevent estimating the true number of clusters, when it is unknown. To address these limitations, we introduce in this paper the Generalized Sparse Fuzzy Clustering (GSFC) framework, which incorporates a continuous sparsity penalty into the fuzzy objective function. Because closed-form solutions are precluded by the simultaneous presence of fuzzifier and sparsity exponents, we propose an efficient primal-dual bisection search coupled with specialized analytic pathways to rigorously enforce the partition constraint while organically driving trace memberships to exact zero. This mathematically enforced sparsity empowers the algorithm to autonomously estimate the true number of clusters by dynamically pruning redundant clusters, provided that it is initialized by a larger than the true number of clusters. Furthermore, we integrate a virtual noise sink to insulate outliers from the rest of the data points. The usage of the proposed framework is also extended to line segment detection. Extensive empirical evaluations across synthetic toy datasets, high-noise geometric stress tests, and real-world benchmarks (UCI wine dataset and hyperspectral images) demonstrate that GSFC effectively exceeds conventional fuzzy benchmarks while autonomously resolving cluster cardinality. Additionally, it significantly outperforms existing sparse possibilistic and probabilistic soft-clustering alternatives in line detection tasks. 

---
# Pixel: A Co-Designed Full-Stack Robot System for Learning General-Purpose Mobile Manipulation 

**Authors**: Tony Tao, Jason Jingzhou Liu, Ruslan Salakhutdinov, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2611.03648)  

**Abstract**: General-purpose mobile manipulation requires a combination of capable robot hardware, scalable demonstration collection, expressive whole-body control, and effective policy learning. Yet these components are often developed independently, creating mismatches that limit the capabilities of the complete system. We present Pixel, a co-designed full-stack system for learning general-purpose mobile manipulation. Pixel combines a low-cost wheeled humanoid with human-aligned kinematics and compliant grippers, multiple teleoperation and robot-free human data-collection interfaces, a unified whole-body controller, and a common policy learning pipeline across human and robot demonstrations. A hierarchical learning stack further enables language-conditioned visuomotor control and long-horizon task execution. We evaluate the system across hardware, data collection, and policy learning, and demonstrate mobile-manipulation behaviors requiring dexterity, active perception, navigation, and whole-body coordination. We hope that Pixel provides an integrated and extensible foundation for studying how robot hardware, data, control, and learning can be designed together for robot learning. More results at this https URL 

---
# Pixel-UMI: Learning Environment-Aware Whole-Body Manipulation from Sensorized Human Demonstrations 

**Authors**: Jason Jingzhou Liu, Tony Tao, Deepak Pathak, Ruslan Salakhutdinov  

**Link**: [PDF](https://arxiv.org/pdf/2611.03647)  

**Abstract**: Sensorized human demonstrations offer scalable data for mobile manipulation, but embodiment differences complicate policy learning. We present Pixel-UMI, a pipeline for learning environment-aware whole-body policies from human data. We first introduce Environment-Aware Retargeting (EAR), which combines reconstructed scene geometry with demonstrated head and gripper trajectories to determine robot posture and generate collision-aware joint and base targets. Delay-calibrated action assignment converts these trajectories into actions. However, kinematic differences can leave residual pose errors, creating viewpoint mismatches between human demonstrations and robot observations. We therefore introduce Retargeting-Error-Aware Masking (REAM), which uses these errors during training to reduce reliance on mismatched views at deployment. Finally, because estimated actions cannot fully capture contact-rich interactions, we collect on-policy human corrections through DAgger to improve performance. We evaluate Pixel-UMI on five challenging tasks spanning articulated-object manipulation, deformable-object handling, constrained-space reaching, whole-body collision avoidance, and contact-rich bimanual coordination, achieving 87.0% average task progress. Video results are available at this https URL 

---
# D&A: Decoupling Description and Assessment Enables VLMs as Zero-Shot Robotic Reward Models 

**Authors**: Marius Memmel, Shawn Yin, Urvi Rutia, Angad Wadhwa, Eldon Li, Gavin Angeloff, Ankit Goyal, Jesse Zhang, Dieter Fox, Fabio Ramos, Anqi Li, Abhishek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.03622)  

**Abstract**: Vision-Language Models (VLMs) hold promise for zero-shot video task-progress estimation in robot learning, yet existing frameworks frequently suffer from confirmation bias, hallucinating progress in failed executions due to implicit success assumptions or relative temporal ranking designs. Naive prompting of frontier models collapses into a binary success detector that fails to capture nuanced, non-linear progress and failures. While post-training approaches can mitigate this through explicit training, they demand expensive real-world robotics data and require computationally intensive post-training for every individual base model. To resolve these limitations, we introduce D&A (Describe & Assess): our approach decouples progress estimation into an objective, task-grounded textual description of scene dynamics followed by an assessment phase using explicit rubrics. By compelling the VLM to synthesize an explicit, text-based definition of absolute success before scoring, the framework anchors evaluations to an external, global criterion rather than the video's internal timeline. Empirical results demonstrate that our method calibrates zero-shot estimates and mitigates confirmation bias, providing fine-grained, partial progress metrics that accurately penalize failed executions while correctly interpolating successful runs. Being purely prompt-based, D&A generalizes across tasks, embodiments, and benchmarks---matching or surpassing post-trained reward models---and its reward estimates enable efficient downstream reinforcement learning. 

---
# Tracing the Scientific Discovery Capacity of LLMs to Their Training Data 

**Authors**: Arda Uzunoglu, Mahler Revsine, Michael Schatz, Benjamin Van Durme, Mehrdad Farajtabar, Daniel Khashabi, Eric Nalisnick  

**Link**: [PDF](https://arxiv.org/pdf/2611.03617)  

**Abstract**: Large language models (LLMs) show growing promise for scientific discovery, yet basic questions remain about what enables reliable discovery. How can we trace the scientific discovery capacity of LLMs to their training data? We study this question as one of missing probability mass and develop a theoretical account in which monofacts-the facts observed once during training-set the discovery capacity. In materials science and regulatory genomics, we directly manipulate monofact rates in training sets and find that more monofacts consistently yield more discoveries across two open-weight model families, sampling strategies, and reasoning budgets. Further experiments show the relationship cannot be explained by training-set diversity alone and can transfer across scientific domains when paired with in-domain data. Together, our results identify monofact rate in training data as a key factor governing discoveries in language models. 

---
# Spacetime Emergence in the Memory of a Learning Agent 

**Authors**: Philip A. LeMaitre, T. Rick Perche  

**Link**: [PDF](https://arxiv.org/pdf/2611.03596)  

**Abstract**: We show that the causal structure of spacetime can emerge within the memory of a learning agent performing a quantum information task. We train a model-based reinforcement-learning agent to maximize the quantum mutual information between localized quantum probes interacting with a quantum field in Minkowski spacetime, without directly providing it with information regarding the spacetime metric or causal structure. In learning to accomplish this task, the reinforcement-learning agent constructs an internal predictive model whose planner develops the characteristic null structure of spacetime, allowing us to reconstruct the causal structure of the Minkowski metric with high accuracy. Within this task, spacetime geometry is obtained from an agent's successful predictions and actions, providing a first step toward showing that spacetime concepts emerge from operational quantum experience. 

---
# Evaluating Uncertainty Calibration in Probabilistic Time Series Foundation Models 

**Authors**: Thomas Decker, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2611.03592)  

**Abstract**: Time series foundation models (TSFMs) are increasingly used for probabilistic forecasting across diverse domains, yet the reliability of their uncertainty estimates remains poorly understood. In this work, we study uncertainty calibration in probabilistic TSFMs from a broader methodological and empirical perspective. We introduce a taxonomy of how TSFMs represent predictive uncertainty and develop a fine-grained evaluation framework for calibration in multi-horizon forecasting that distinguishes pooled, horizon-wise, and stronger conditional notions of calibration. Using this framework, we conduct a comprehensive empirical evaluation of popular TSFMs across diverse datasets and show that current models exhibit distinct and systematic patterns of miscalibration that standard metrics often obscure. We further analyze common post-hoc recalibration methods and characterize which forms of calibration they can improve and which tend to persist. Our results provide practical guidance for evaluating and improving uncertainty calibration in TSFMs and highlight important limitations of current practice. 

---
# Sonar-MoE: In-Backward Expert Updates for Memory-Efficient Mixture-of-Experts Training 

**Authors**: Avni Mittal, Avinash Anand, Ashutosh Kumar, Dikshant Kukreja, Kritarth Prasad, Chaitanya Lakhchaura, Rishitej Reddy Vyalla, Erik Cambria, Zhengkui Wang, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2611.03577)  

**Abstract**: Most parameters of a Mixture-of-Experts (MoE) model belong to its experts, and standard training keeps every expert gradient in GPU memory until the optimizer step. This memory grows with the number and size of the experts, so it rises as MoE models scale. We introduce Sonar-MoE, which updates each expert tensor during the backward pass and frees its gradient right away. It brings optimizer-in-backward from dense models to batched MoE expert tensors, so the GPU never holds more than one layer of expert gradients. When we fine-tune OLMoE-1B-7B and Granite-3.1-3B-A800M in bf16 with two expert-kernel backends, Sonar-MoE lowers peak memory by 19% to 22%. Its updates are bitwise identical to standard training without expert gradient clipping, and held-out loss stays within 0.003 of all baselines. Each step also takes 17% to 23% less time than standard training with global-norm clipping. Updating whole batched tensors rather than individual experts is what keeps Sonar-MoE fast, as it avoids a 20% to 72% slowdown while using the same memory. Sonar-MoE matches optimizer-in-backward over all parameters in memory and speed while still clipping the dense parameters. 

---
# PlantVLM-Edge: Quantized Vision-Language Models for On-Device Crop Health Diagnosis 

**Authors**: Gaurab Wagle, Bishal Adhikari, Jiajia Li, Xiaolei Huang, Dong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.03557)  

**Abstract**: Timely and accurate crop health diagnosis is essential for sustainable agricultural production, yet accurate diagnosis often requires expert knowledge that is not readily available. Recent vision-language models (VLMs) have demonstrated remarkable capabilities in visual understanding, reasoning, and natural-language interaction, creating new opportunities for intelligent crop health assessment. However, deploying VLMs on resource-constrained devices remains challenging due to their substantial computational and memory requirements. This paper presents PlantVLM-Edge, a quantized vision-language framework for conversational crop health diagnosis that can run entirely on resource-constrained edge devices. PlantVLM-Edge combines agricultural domain adaptation, post-training quantization, and on-device serving with an offline chat interface, allowing growers to ask open-ended questions and receive diagnoses, symptom explanations, and management guidance without cloud connectivity. The framework is backbone-agnostic: we validate it with two different VLM types with varying scale and architecture (Qwen3-VL-2B and LLaVA-1.5-7B), four quantization schemes, and two diagnostic domains (plant diseases and insect pests), and we deploy all variants on an NVIDIA Jetson AGX Orin. Eight-bit quantization affects answer quality minimally relative to the full-precision model, and 4-bit activation-aware weight quantization (AWQ) retains 98.6% of full-precision answer quality on both domains while reducing the weight memory of the 2B backbone by 49%. In its most compact form, PlantVLM-Edge requires 2.01 GiB of weight memory, returns its first token in about 0.12 s, and completes a 64-token diagnostic response in 0.78 s. These results demonstrate that timely and accurate crop health diagnosis is practical on edge devices, providing a foundation for next-generation crop monitoring and decision support systems. 

---
# MCP Tool Poisoning in Agentic AI Frameworks: A Cross-Framework Measurement Study 

**Authors**: Farooq Shaikh, Biagio Montaruli  

**Link**: [PDF](https://arxiv.org/pdf/2611.03555)  

**Abstract**: The Model Context Protocol (MCP) lets agentic AI frameworks connect language models to external tool providers, but it carries attacker-controlled content (tool descriptions, schema fields, and response data) into model context without validation. Prior benchmarks measure model compliance with poisoned instructions without confirming real execution. We present the first cross-framework measurement of MCP tool poisoning with real tool execution. A 25-class root-cause taxonomy unifies prior classifications, and the open-source tool mcpvenom drives a 9,720-cell susceptibility matrix across 12 frameworks, 18 models, and 45 attack scenarios, with a dual oracle that confirms exfiltration via canary HTTP receipt. The overall attack success rate is 22.9% (2,113 vulnerable cells). Of these, 64.9% dispatch an attacker tool call, 23.0% contact the attacker endpoint, and 12.1% verifiably exfiltrate a secret. Resistance is model-level: nine core frameworks cluster within 23 to 26% (chi-squared = 3.30, p = 0.91), and no framework implements intentional defense. Incidental framework properties mitigate in specific cases, and a regex description scanner rejects 20 of 45 scenarios at a 7.2% false-positive rate, so metadata scanning alone is a partial control. Model ASR ranges from 0.4% to 49.7%. Claude Opus 4.8 is vulnerable in 8 of 530 cells while issuing 223 explicit refusals. A code-level audit finds no intentional defense across twelve frameworks, and all five coordinated disclosures across four vendors were rejected. 

---
# Can LLM Agents Be Forward-Deployed Engineers? Requirement Discovery and Delivery for Enterprise Text-to-SQL 

**Authors**: Marianne Menglin Liu, Rongguang Wang, Zheng Wang, Yewen Wang, Leonid Boytsov, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2611.03551)  

**Abstract**: LLM coding agents can increasingly automate model post-training, but whether they can act as forward-deployed engineer (FDE) agents, discovering customer requirements and turning them into deployment decisions, remains unclear. We study this on a real enterprise text-to-SQL engagement: a simulated customer, grounded in the real use case's requirements, starts with an ambiguous request. An agent must discover acceptance criteria, data permissions, and hardware constraints in the discovery phase, then plan and execute a post-training recipe in the delivery phase. Across 15 models, the strongest discover over 80% of withheld requirements and open-weight models discover 40-50%; in the delivery phase, only 25% of runs across eight coding-agent configurations pass all acceptance criteria. We identify four bottlenecks in this engagement: (1) incomplete discovery, with 40% of withheld requirements missing or misrepresented; (2) unreliable execution; (3) underused compute budgets and a lack of systematic comparisons across base models and post-training methods; and (4) benchmark accuracy does not guarantee production readiness. We suggest human review of discovered requirements, monitoring of the agent's plan during execution, and an explicit experiment-environment specification covering hardware, budget scope, and production checks. 

---
# Contrastive Adapter Training and Consensus Knowledge Distillation for Multi-Source-Free Domain Adaptation in Skin Cancer Diagnosis 

**Authors**: Ali Abedi, Q. M. Jonathan Wu, Ning Zhang, Farhad Pourpanah  

**Link**: [PDF](https://arxiv.org/pdf/2611.03528)  

**Abstract**: Skin cancer diagnosis, particularly differentiating melanoma from benign nevi, is challenging due to the visual similarity between lesions. Although convolutional neural networks (CNNs) and vision transformers (ViTs) have shown promising performance, their effectiveness often deteriorates on data from heterogeneous clinical sources. Conventional domain adaptation methods address domain shift but require access to source data during adaptation, which is often infeasible due to privacy regulations. Multi-source-free unsupervised domain adaptation (MSFDA) addresses this limitation by leveraging multiple labeled source domains to generalize to an unlabeled target domain without requiring access to source data. However, existing MSFDA methods rely on full backbone fine-tuning, leading to catastrophic forgetting and overfitting on small clinical datasets, and address domain shift at the aggregation stage without establishing a shared domain-invariant feature space. Furthermore, their reliance on hard pseudo-labels or confidence-weighted aggregation introduces noisy supervision signals under domain shift. To address these limitations, we propose CAT-CKD, consisting of two components: (1) contrastive adapter training (CAT), which trains lightweight ConvPass adapters within a frozen ViT backbone using supervised contrastive learning (SCL) to establish a shared domain-invariant feature space before source-specific model training, and (2) consensus knowledge distillation (CKD), which aggregates logits from multiple source models into a consensus supervisory signal and adapts a student model on unlabeled target data using KL divergence. Experiments on five publicly available skin lesion datasets show that CAT-CKD achieves an average AUROC of 86.1%, outperforming existing MSFDA methods while requiring only 4.3M trainable parameters. The code is available at this https URL. 

---
# Library of Layers (LoL): Adaptively Ordering Transformer Layers at Runtime 

**Authors**: Abdellah Rahmani, Sean McLeish, Ang Li, Sanae Lotfi, Pascal Frossard, Micah Goldblum  

**Link**: [PDF](https://arxiv.org/pdf/2611.03526)  

**Abstract**: Computer programmers chain together the same reusable computational primitives, for example, conditional statements or loops, but in different orders to implement different algorithms. In contrast, existing language model architectures always execute their layers in the same order. We introduce Library of Layers (LoL), an architecture that dynamically orders its layers during inference to solve the problem at hand. LoL has a single hyperparameter, the size of its global library of transformer blocks, that allows it to transition seamlessly between parameter reuse and sparsity, generalizing the parameter-efficiency and FLOPs-efficiency tradeoffs of dense, depth-recurrent, and mixture of experts architectures in one unified family. We show that LoL achieves competitive performance with all three above baselines when pretrained on text data. On synthetic sudoku and maze data, we observe that the model employs distinct layer orderings to implement different solution strategies, and we can intervene on routing to force a model to apply a specific solution strategy to a problem instance. In a multilingual natural language setting, we find that LoL chooses different layer order patterns based on language and script. Similarly, we train LoL on safety data and fit probes to its router outputs, finding that these probes distinguish safe from unsafe text with high AUROC. LoL provides a path toward a future in which models dynamically compose their own computational graph, enabling more efficient and controllable reasoning. 

---
# PanoTac: A Dense Three-Axis Tactile Sensor for Robust Touch-Centric Manipulation 

**Authors**: Binglin, Wang, Xiaole, Jiang, Zicheng Teng, Xinqi Liu, Yunzhu Li, Lerrel Pinto, Zichen Jeff Cui, Binghao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.03520)  

**Abstract**: Contact-rich manipulation benefits from tactile feedback that captures both detailed contact geometry and interaction forces. When scooping grains, contact patterns help infer spoon orientation, while resistance transmitted through the grasp guides motion. To provide these complementary sensing capabilities, we present PanoTac, a low-cost, compliant tactile fingertip designed for dense three-axis tactile sensing. Each fingertip provides a $12 \times 32 \times 3$ tactile representation, with one normal ($z$) and two shear ($x$, $y$) response components at each taxel. The sensor operates at 60 Hz and costs less than $50 per fingertip. Sensor characterization quantifies response repeatability and consistency across sensor instances and contact locations. We integrate PanoTac into a robotic system with a visuo-tactile pipeline. Across test-tube preparation, grain scooping, and salt pouring, the resulting policies outperform vision-only baselines by an average of 46.7 percentage points. Ablations show that removing either spatial contact information or shear feedback reduces overall performance across the task suite. Our project page is available at this https URL. 

---
# VIAS: Value-Isolating Activation Steering for Multi-Value Alignment and Compliance for Organisations 

**Authors**: Avni Mittal, Avinash Anand, Dikshant Kukreja, Erik Cambria, Rajiv Ratn Shah, Sandipan Dandapat  

**Link**: [PDF](https://arxiv.org/pdf/2611.03518)  

**Abstract**: Organisations increasingly deploy large language model agents that must follow organisational values. We check whether activation steering, which adds vectors to a model's internal activations during generation without changing its weights, can improve compliance beyond prompting alone. Since most scenarios involve several values at once, we focus on multi-value steering. A simple way to build these vectors is to contrast responses that follow all values relevant to a scenario with responses that violate them. However, this gives nearly identical vectors for different values: they capture a general compliance signal rather than individual values, which limits their use when combining several values. We propose Value-Isolating Activation Steering (VIAS), which contrasts responses to the same scenario under two prompts, one listing all relevant values and one omitting a single target value. We keep a pair only if the target value changes from compliant to non-compliant and both responses still meet a minimum compliance level on the other values. We evaluate VIAS on PROTECT, a workplace compliance dataset where each scenario is tagged with relevant values from a 21-value taxonomy. VIAS reduces overlap between value vectors, making them better suited for multi-value steering, although this alone does not lead to independent control over each value. Combined with prompting, VIAS complies with every tagged value on 69.6% of scenarios for Qwen2.5-7B, 75.2% for Mistral-7B, and 72.6% for Phi-3.5-mini, outperforming prompting alone for all three models. 

---
# A Cross-Domain Matrix Engine for Signal Processing and Deep Neural Net Inference 

**Authors**: Mehdi Safarpour  

**Link**: [PDF](https://arxiv.org/pdf/2611.03503)  

**Abstract**: Matrix multiplication dominates both PHY signal processing and Deep Neural Network (DNN) inference, yet these domains place opposing demands on the matrix engine. Signal processing requires low-latency execution of relatively small matrix operations, whereas DNN inference, particularly at large batch sizes, prioritizes high throughput and energy efficiency for large matrix multiplications. While offering efficient memory access profile, a Systolic Array (SA) adds fill and drain delays per matrix. Those can be amortized for large matrices but not for small matrices. Conversely, a tensor unit completes the kernel in few cycles but would require repeated reads of same operand tiles over large matrices. This makes SA too slow for signal processing and tensor too inefficient for AI. We show that this need not be a choice between systolic and tensor structuring of Multiply-Accumulate (MAC) units. We propose a single engine that operates primarily as a SA and, through low-cost reconfiguration, transforms into a tensor-style engine on a per-operation basis. MUXes create both the spatial reduction adder tree and temporal reduction accumulators from the existing MACs and registers within the systolic array. The proposed reconfigurable SA simulated using modified SCALESim simulator, verified on an Zynq 702 FPGA and synthesized using ASAP 7nm FinFET library. SCALE-Sim simulations shows up to 7.5x lower latency for small complex value PHY kernels. Even on DNN workloads, it achieves up to 1.65x speedups over vanilla SA execution with trivial increase of memory accesses, retaining the systolic array's memory access advantage. Synthesis results estimate reconfigurable fabric costs only 8% power and 13.6% area overheads. 

---
# Harder Than the Task: From Technical to Organizational Difficulty in AI Benchmarking 

**Authors**: Victor Barres, Aden Barton, Nicolai J. Foss, Henrik M. Nielsen  

**Link**: [PDF](https://arxiv.org/pdf/2611.03494)  

**Abstract**: AI benchmarks have made remarkable progress in measuring the ability of AI systems to perform increasingly demanding cognitive and technical tasks. Yet, many of these benchmarks treat technical tasks as the unit of analysis, while much economically valuable work takes place within organizations, where tasks are embedded in a system of relations. Within organizations, successful task performance depends not only on solving the technical demands of the task but also on navigating the interdependence of activities, distributed knowledge, differentiated authority, temporal constraints, conflicting interests, and changing environmental conditions while the task is underway. Accordingly, we argue that organizational task difficulty constitutes a distinct and insufficiently recognized dimension of AI capability, and that existing approaches to benchmarking provide an incomplete basis for assessing how effectively AI agents can perform real-world organizational work. By drawing on organization theory, we develop a taxonomy of the structure of work that characterizes the conditions that agents face in organizational work. Applying this taxonomy to analyze existing benchmarks, we find limited and uneven coverage of organizational dimensions, with fewer than one quarter of prominent benchmarks claiming any of them. We therefore argue that AI benchmarks should increasingly move beyond the evaluation of technically difficult tasks toward the systematic assessment of agents' ability to navigate the organizational conditions under which economically valuable work takes place. 

---
# From Reactive to Agentic Resilience: Multi-Timescale AI-RAN Orchestration Under Signaling Storms 

**Authors**: Sotiris Chatzimiltis, Mahdi Boloursaz Mashhadi, Mohammad Shojafar, Rahim Tafazolli  

**Link**: [PDF](https://arxiv.org/pdf/2611.03406)  

**Abstract**: Next-generation open radio access network (Open RAN) and AI-RAN deployments turn the RAN into a disaggregated, compute-aware platform whose service continuity depends on many control loops acting at different timescales. Resilience in this setting is best seen as a lifecycle spanning anticipation, absorption, adaptation, recovery, and evolution. Yet, existing mechanisms remain stage-specific and are rarely coordinated across the control hierarchy. This paper argues that agentic AI offers an orchestration layer that coordinates this lifecycle, turning operator intent into coordinated action while keeping slow language-model reasoning off the fast control path. Using mass-attachment signaling storms as a running study, we build a discrete-event Open RAN attach simulator that captures retry amplification, shared-compute contention, and non-instantaneous provisioning, and we propose a three-tier framework in which an orchestrator at the service management and orchestration layer sets a standing policy, a per-site non-real-time agent judges storms and anticipates scheduled crowds, and a deterministic near-real-time loop enforces every decision. Across experiments, the agentic controller approaches the resilience of a fully provisioned pool of servers at a quarter of its capacity by reading a scheduled event from a calendar and provisioning ahead of it. No reactive weight setting recovers this once the surge arrives. Free-text operator intents ground into control directives at 94% exact match on previously unseen intents, and a single instruction reshapes live behavior end to end. We also quantify the latency, cost, and contention limits that bound it. The supplementary material provides analytical proof of the derived Theorem on anticipation horizon, as well as the subsequent Corollary results on reactive provisioning limits. 

---
# CrowdForge: Effects of Audio and Visual Audience Participation Modalities on Agency, Collaboration, and Engagement in AI-Mediated MR Performances 

**Authors**: Torin Hopkins, Shih-Yu Ma, Zander Trafton-Hancock, Ellen Yi-Luen Do, Luca Turchet  

**Link**: [PDF](https://arxiv.org/pdf/2611.03389)  

**Abstract**: Technology-mediated audience participation (TMAP) enables audiences to influence live performances, yet balancing interaction with meaningful agency remains challenging. While AI-generated music and MR enable new forms of audio and visual audience participation, little is known about how these participation modalities influence audience experiences during live musical performance. In particular, it remains unclear how they shape perceived agency, collaboration, and engagement, and how they complement one another within TMAP systems. We present CrowdForge, an AI- and MR-mediated paradigm where audiences collectively control generative music and virtual environments during a live performance featuring a guitarist and DJ. Through a mixedmethods concert study evaluating three interaction conditions (audio, visual, and combined audio-visual), we demonstrate that observability is the primary driver of audience agency. While visual interaction provided easily observable feedback, audio parameters offered macro-structural control over the music. Audiences tended to use the control enabled by AI audio generation to test musical boundaries, introducing a productive adversarial tension between the crowd and the musicians. This dynamic pushed creative limits, deepened engagement, and generated moments of musical coherence when performers successfully resolved audience-induced challenges. We argue future TMAP systems should prioritize human-centered, highly observable interactions to build agency, while combining the expressiveness of music with the clarity of visual feedback. 

---
# Characterizing When Language Models Take Harmful Actions to Protect Their Assigned Goals 

**Authors**: Brandon Jaipersaud, Elliot Creager  

**Link**: [PDF](https://arxiv.org/pdf/2611.03374)  

**Abstract**: Recent language model (LM) safety failures have largely been studied as disparate phenomena. In agentic misalignment, a model blackmails an executive to avoid being replaced. In alignment faking, it complies with harmful requests during training to keep its values from being modified. We argue that these behaviours share a common decision structure: a model assigned a goal must either take a harmful action to preserve it or refuse and lose it. We formalize this as a goal-harm conflict and present the same conflict under various elicitation protocols that range from compact forced-choice prompts to simulated agentic environments. These protocols span two perspectives on LM decision-making, which we term (i) the logit view, reading preferences from answer-token probabilities, and (ii) the sampling view, classifying open-ended responses with an LM judge. Under the logit view, we show that goal importance and harm severity additively shape how a model resolves these value conflicts. Under the sampling view, we derive two factorizations of the harmful-action probability that safety evaluations commonly report. The first measures how well a preference read from answer tokens carries over into behaviour, such as making a harmful tool call. The second attributes a low harmful-action rate either to refusal or to responses that reach no resolution. We then show how fictional context changes a model's preference under the logit view, and how tool use and partial observability -- two features of agentic evaluations -- change its harmful-action rate under the sampling view. Our work offers a descriptive framework for studying how LMs resolve such value conflicts and for estimating low-probability harmful actions at a finer granularity than the aggregate harm rates commonly reported in safety evaluations. 

---
# TrafficInstruct: A Multi-Task Network Traffic Dataset and Benchmark for Instruction Tuning 

**Authors**: Zhengdao Li, Hui Wen, Zixuan Wan, Weifeng Zhang, Yan Wu, Benlong Wu, Nenghai Yu, Weiming Zhang, Kejiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.03319)  

**Abstract**: Instruction tuning for multi-task network traffic analysis requires data with clearly defined sources, supervision targets, and evaluation conditions. We introduce TrafficInstruct, a newly collected dataset comprising raw packet captures and 55,000 structured supervised fine-tuning records across ten tasks: malware traffic detection, botnet detection, malicious DNS-over-HTTPS detection, Web attack detection, advanced persistent threat detection, encrypted VPN detection, Tor behavior detection, encrypted application/service classification, website fingerprinting, and application classification under concept drift. An automated workflow captures traffic from controlled software executions, attack experiments, and application activities. Labels are supported by task-specific collection evidence, including process or socket attribution, activity records, and execution outcomes. Filtering, sanitization, and task-specific transformations produce instructions, traffic observations, and structured answers. We verify sample sources and partition the dataset by complete collection sessions, keeping all samples from the same session within one partition. The benchmark defines allowed inputs, deterministic label mappings, fixed training/validation/test partitions, and task-wise scoring rules. We evaluate majority-class, random-forest, and ET-BERT baselines, an unadapted 27B language model, and independently trained task-specific LoRA adapters on a shared frozen base. Adapters are selected using validation data and evaluated on the corresponding test sets. Task-specific adaptation improves classification performance, supporting instruction tuning across diverse traffic-analysis tasks. 

---
# LLMs Depend on Specialist Tools for Out-of-Distribution Generalization in MS/MS Structural Elucidation 

**Authors**: Gabriel Asher, Cailum M. K. Stienstra, Tomo Oga, Devesh Shah Antonio H. de O. Fonseca, Niall O'Connor, Michael Widrich  

**Link**: [PDF](https://arxiv.org/pdf/2611.03308)  

**Abstract**: A frontier language model was recently reported to solve 1D NMR structural elucidation without tools. We ask whether this capability transfers to tandem mass spectrometry (MS/MS) in the de novo setting, where the model must propose a structure directly from the spectrum without a candidate list or reference library. To separate reasoning from recall, we build a 100-analyte benchmark from MassSpecGym, split by each analyte's textual footprint into in-distribution (well documented in literature) and out-of-distribution (essentially undocumented) arms. We hold the model, harness, and output format fixed and vary only the LLM's access to domain-specific tools. Without tools, a frontier LLM reaches 52% Top-1 accuracy in-distribution but only 2% out-of-distribution. Only access to de novo structure generators improves out-of-distribution accuracy, raising it to as much as 38% Top-1, and the resulting agent outperforms the standalone de novo models both in and out of distribution. These results suggest that LLMs are not effective at de novo MS/MS structural elucidation on their own, and that further gains depend on the specialist tools available to the agent rather than on the LLM itself. 

---
# An Axiomatic Framework for Reinforcement Learning in Open Multi-Agent Systems 

**Authors**: Jianhong Wang, Yang Li, Samuel Kaski, Jonathan Lawry  

**Link**: [PDF](https://arxiv.org/pdf/2611.03301)  

**Abstract**: Open multi-agent systems, where only a subset of agents are controllable, are central to real-world domains such as smart grids, yet lack a principled foundation for reinforcement learning algorithm design. We study this setting through n-agent ad hoc teamwork (NAHT), a recent framework for learning in open multi-agent environments. We propose an axiomatic framework that derives value function learning from the Shapley axioms (Efficiency, Symmetry and Linearity), treating them as structural constraints rather than heuristic design choices. This formulation builds on a novel NAHT decomposition, which characterises the fundamental difference between open and fully controllable multi-agent systems, and establishes a cooperative game-theoretic foundation for learning. We further show that enforcing these axioms on learning individual value functions recovers the Shapley value for Dec-POMDPs, providing a principled bridge between reinforcement learning and classical cooperative game theory. Building on this framework, we introduce Shapley Machines and Banzhaf Machines, two generic algorithmic templates that can be instantiated with standard reinforcement learning methods. When applied to strong base algorithms such as IPPO and POAM, our methods consistently improve learning efficiency and generalisation to unseen agent types and conventions across NAHT benchmark tasks. The code of experiments is published on this https URL. 

---
# SQLoom: Set-Contrastive Learning of Syntactic, Semantic, and Logical Dependencies for Text-to-SQL 

**Authors**: Hao Fu, Degang Sun, Wan Wei, Jinxia Wei, Chun Long, Peng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.03298)  

**Abstract**: Text-to-SQL translates natural language questions into SQL queries, enabling database access without writing SQL. Fine-tuning open-weight large language models (LLMs) is cost-effective, but existing methods treat SQL as ordinary text and optimize token-level likelihood without explicitly modeling its syntactic, semantic, and logical requirements. Thus, a single incorrect alias or comparison operator can invalidate a query while contributing little to training loss. We propose SQLoom, a constraint-aware fine-tuning framework with supervision at three levels: syntactic validity under SQL grammar and database schema, semantic alignment between query decisions and the question, and logical consistency across dependent clause decisions. SQLoom extends supervised fine-tuning with a unified set-contrastive objective favoring sets of correct outputs over minimally different incorrect ones at each level. LLM agents propose this supervision, which a SQL parser and execution on counterexample databases verify. We also release Spider-Refined, a training dataset correcting 378 flawed Spider annotations and enriching database schemas. Experiments on Spider, BIRD, and three perturbed Spider variants show that SQLoom improves execution accuracy for Qwen3-1.7B from 62.9% zero-shot to 88.7% on Spider development, reaching 89.2% on Spider test and 73.4% on BIRD development. It outperforms fine-tuning, reinforcement learning, and agent-based baselines with up to 70 billion parameters. On question perturbations, SQLoom loses at most 0.6 accuracy points versus up to 12.1 for agent-based methods. It generates SQL in a single pass at 0.42s per query, 23$\times$ faster than a GPT-4-based agent method. These results demonstrate the effectiveness of set-contrastive supervision over token-level training. 

---
# Rethinking Fairness Metrics for Multi-Class and Multi-Group Assessment: Not Everything is Binary 

**Authors**: Matteo Grazioso, Fabio Naccarella, Chiara Gallese, Marco S. Nobile  

**Link**: [PDF](https://arxiv.org/pdf/2611.03257)  

**Abstract**: Artificial Intelligence, and Machine Learning in general, is becoming pervasive across all domains and disciplines. By means of this data-driven approach, algorithms can learns complex task just by learning from samples. However, if the data-set is biased, they might lead to the development of unfair computer systems, that is, characterised by a prejudice towards specific individuals or groups, which might get an unmotivated output from the system. This circumstance lead to the development of multiple metrics designed to quantify the level of unfairness of an algorithm. However, a generalised assumption in the literature is that the sensitive features and the output labels are binary. When the data is not binary, it is forcefully binarized using a one-versus-the-rest simplification. In this work, we discuss the limitations of existing metrics and propose novel, mathematically sound, and generalized multi-group and multi-class formulations of the metrics. Ultimately, our results highlight the empirical assessment distortion caused by binarization strategies, which reveals the limitations of binary fairness metrics in the multi class setting. 

---
# Do Agents Still Follow the Rules? Benchmarking Rule Compliance under Agent Context Compaction 

**Authors**: Yiwei Zha, Qingyu Meng, Herbert Bos, Min Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.03223)  

**Abstract**: LLM agents compact long histories to fit their context windows, but the compacted context must still carry the operating rules that govern later actions. Prior work shows that violations rise after compaction, but measures compliance outside a complete agent harness and does not identify where each violation comes from. We introduce ARCBench(Agent Rule Compliance Benchmark), a configurable benchmark built on the open-source pi agent harness. Its tool-calling agent loop lets us measure rule compliance from actions taken during execution, bringing the evaluation closer to deployed agent workflows. Using ARCBench, we examine where violations after compaction come from, and how different compaction configurations and adversarial settings change their sources. We find that compaction rarely deletes the rule: most violations after compaction occur while the rule is still in the compacted history, with the agent treating a later request as permission, ignoring the rule, and proceeding anyway. Where violations come from depends on the configuration: how the rule is delivered decides whether the agent would break it even without compaction, while the compaction strategy and budget decide whether the rule survives. The strongest compactor-targeted attack more than doubles violations by corrupting the rule in the summary and changing how the agents behave, whereas the strongest defense restores the rule in most summary but still violates. Finally, retaining a rule does not guarantee compliance: compacted runs that retain the rule violate more than twice as often as runs with the full history. 

---
# CERBO: Certified Recourse at a Requested Rate for Gradient-Boosted Trees 

**Authors**: Farhan Shahriyar Hossain, Taufikur Rahman Fuad, Md. Zarif Bin Hasnat, Md Abrar Jahin, Md Rizwan Parvez  

**Link**: [PDF](https://arxiv.org/pdf/2611.03222)  

**Abstract**: Recourse-aware learning trains models so that people who receive an adverse decision can reverse it with an affordable change. For gradient-boosted decision trees (GBDTs), the existing method, RABIT, exposes a single recourse weight whose effect on the recourse rate is abrupt and data-dependent, and its training bound cannot certify recourse for any rejected individual. We present Certified Recourse Boosting (CERBO), which takes a requested recourse rate and returns a GBDT with certified actions: it tracks exact additive effort along tree paths, trains against committed boxes (a family of bounds that contains RABIT's and certifies a within-budget action for each training sample) and adjusts the weight after every tree until the certified rate meets the request. A calibration step guarantees the rate on new individuals, and the certifier runs on any tree ensemble, including XGBoost and LightGBM, in milliseconds per person without a solver. In RABIT's own experimental protocol, at equal accuracy, CERBO raises the recourse ratio from 0.90 to 0.99 on FICO and from 0.64 to 0.77 on Bail. Under additive costs on six data sets, it raises the area under the accuracy-recourse frontier in all 12 settings, and in simulated deployments it guarantees recourse to 70% of rejected individuals in 11 of 12 settings, against 2 for RABIT. 

---
# Can Coding Agents Sustain Continual Evolution without Forgetting? 

**Authors**: Shangziqi Zhao, Yitong Zhang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.03219)  

**Abstract**: Recently, harness self-evolution has emerged as an important paradigm for improving agent performance. Prior studies show that evolving the agent harness can effectively enhance agent capabilities without modifying the foundation model's parameters. We find that existing approaches largely study agent evolution on a single task. In practice, coding agents may need to handle multiple task types over time. Whether an agent harness can continuously evolve across multiple tasks without suffering from catastrophic forgetting remains an important open question.
In this paper, we investigate this open question through a comprehensive empirical study of continual harness self-evolution. We organize three software engineering task datasets into six different evolution directions and apply two representative harness evolution methods to perform continual evolution along these directions. Our experimental results show that evolution on a subsequent dataset can substantially undermine the performance on previously seen datasets. For example, with DeepSeek-V4-Flash-0731, Meta-Harness improves its performance on SWE-Bench to 94.0% after evolution. However, after further evolution on FeatureBench, its performance on SWE-Bench drops to 32.0% by the fifth round, corresponding to a 66% relative decrease. Finally, based on our experimental findings, we identify several insights, providing directions for the continual self-evolution of agent harnesses. Code is available at this https URL. 

---
# Reducing SNAP Payment Errors at Lower Cost With AI: A Simulation Study 

**Authors**: Alex Chohlas-Wood  

**Link**: [PDF](https://arxiv.org/pdf/2611.03213)  

**Abstract**: New federal legislation will require states with high payment error rates in the Supplemental Nutrition Assistance Program (SNAP) to pay a share of benefit costs, potentially costing some states over a billion dollars per year. These costs create substantial pressure to improve payment accuracy. One strategy to improve payment accuracy is preauthorization review, in which staff double-check applications and correct mistakes before benefits are issued. But limited staff capacity makes it important to target these reviews effectively. One leading targeting approach uses statistical risk models to prioritize cases likely to contain errors. Another novel approach uses AI to flag specific potential discrepancies for reviewers to check. For states under pressure to reduce error rates, it is unclear which approach-or combination of approaches-would reduce payment error rates most effectively. We compare these strategies through simulations using public SNAP quality control data. Under our main assumptions, risk-based review alone costs about seven times as much as combining AI flags with risk scores. This cost advantage persists across a wide range of plausible assumptions. Our findings suggest that AI-assisted preauthorization review could be a key tool to help states avoid substantial cost-sharing obligations. 

---
# Written Once, Read Differently How Language Models Own Answers Steer Their Next Ones 

**Authors**: Marcus Armstrong  

**Link**: [PDF](https://arxiv.org/pdf/2611.03198)  

**Abstract**: When a language model answers a yes/no question, that answer changes how it answers a different question about the same entity. We study this \emph{self-commitment} in 22 open-weight models from four families (0.5B--72B parameters), fixing the model's first answer and measuring the shift in its second across 3,600 question pairs about 120 entities. Each answer writes a \emph{commitment direction} into the residual stream. The similarity of two questions' directions predicts their coupling on held-out entities ($\rho = 0.43$--$0.85$), and ablating a single direction shared across topics leaves at most 17\% of the effect in every model. How this direction is read is far less stable: the reading circuit spreads across more attention heads as models grow, instruction tuning makes more questions read it in the opposite sense, and in one model changing only the system prompt reverses the circuit's sign. Ablating the direction while sparing the attention sink removes most of the coupling at under 1.3\% perplexity cost in 18 of 19 instruction-tuned models, though it shifts standalone answers in two. Without the sink spared, the direction's alignment with the leading principal component of ordinary text predicts which models pay a high cost. Written once, the commitment is read differently across scale, training and prompt. 

---
# LaBraM at Scale: Scaling Data and Models for EEG Pretraining 

**Authors**: Jun-Yu Pan, Ruicheng Yin, Bao-Liang Lu, Dongsheng Li, Wei-Long Zheng, Yansen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.03165)  

**Abstract**: LaBraM, an early EEG foundation model, demonstrated that large-scale EEG pretraining can yield strong representations for diverse downstream tasks. Building on this foundation, we expand pretraining data and model capacity while retaining its core encoder and objective. We train ten encoders with 5.8 million to 1.97 billion parameters on three EEG collections spanning 25,067 to 79,910 hours, adapting signal preparation and training to heterogeneous recordings. Across 23 tasks, Atom-L* reaches 59.05% mean balanced accuracy at its best evaluated full fine-tuning (Full-FT) checkpoint, 3.61 percentage points above the strongest baseline EEG foundation model. Quark-G leads linear probing (LP) at 54.33%, a 1.24-point margin. Both also lead average task rank. At the last evaluated checkpoints, larger encoders improve Full-FT more consistently than LP, and stronger LP does not guarantee stronger Full-FT. Preserving patch structure improves LP over average pooling. Data-fraction experiments suggest capacity-dependent saturation, while checkpoint trajectories show that downstream performance can decline as pretraining loss falls. Joint classification with a shared Qwen3 readout reaches 50.50% with Molecule-H and Qwen3-8B; increasing language-model size provides no consistent gain. These results demonstrate the value of scaling LaBraM and the importance of matching pretraining scale to downstream adaptation. 

---
# Beyond Surface Metrics: Multi-Condition Requirements in Enterprise Maintenance and AI Coding Benchmarks 

**Authors**: Heeyong Eun  

**Link**: [PDF](https://arxiv.org/pdf/2611.03162)  

**Abstract**: Repository-level AI coding benchmarks measure task success but do not directly capture the conditions required for software maintenance completion. We study multi-condition fulfillment, where a maintenance task must satisfy multiple operational conditions, sometimes with directional dependencies. Using a fixed 18-criterion taxonomy, we analyze 2,925 Apache Fineract issues and 944 tasks from SWE-bench Pro, SWE-PolyBench Java, and SWE-EVO. We examine condition breadth, corpus-level co-occurrence, and source-grounded directional requirement relationships.
Among 1,177 benchmark task pairs matched on issue classification, modified-file count, and code churn, the mean fulfillment-set Jaccard distance is 0.8138, and 57.4% have disjoint condition sets. Condition count alone does not explain this contrast: SWE-bench Pro averages 2.47 conditions per task, versus 1.78 for Fineract. Temporal (B4) and numerical (B2) conditions have a phi coefficient of +0.430 in Fineract versus -0.013 in the pooled benchmark corpus. Among temporally tagged tasks, numerical conditions are also specified in 48.85% (254/520) of Fineract cases versus 5.81% (5/86) of benchmark cases.
We document 16 directional relationships in Fineract requirements involving time, calculation, and state preservation, and inspect four benchmark cases to illustrate how relevant conditions appear in evaluation assertions. These examples do not quantify benchmark-wide oracle omissions. The results show that similar surface task profiles can mask substantially different fulfillment-condition sets. We recommend reporting condition breadth, co-occurrence, and directional relationships separately. The corpus comparisons are descriptive and may reflect domain differences; they do not establish causality or production failure rates. 

---
# Visual-to-Executable Procedural Reconstruction of Buildings 

**Authors**: Zichong Zhu, Zhihao Liu, Andrei Sharf, Zhanglin Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.03143)  

**Abstract**: Procedural modeling offers compact, editable, and reusable representations for architectural content, but building such models still depends heavily on manually authored grammars. We study image-based procedural building reconstruction from in-the-wild images, with the goal of recovering an executable procedural description that can be instantiated as an editable 3D building asset. Because a single image provides only partial and view-dependent evidence, we focus on recovering the dominant visible facade(s) and attach the recovered facade program to a simple mass model for building-level instantiation. Our method decomposes this task into two structured prediction stages. First, a Global Structure Layout Tree recovers facade-scale split-and-repeat organization through a fine-tuned layout model with render-guided visual-symbolic reflection. Second, a Component Parameterization schema maps representative facade tiles to structured procedural attributes for local architectural component synthesis. These two representations are then deterministically compiled into executable Computer Generated Architecture (CGA) programs, avoiding the brittleness of direct low-level program generation. Experiments show that the proposed framework handles both regular facades and challenging in-the-wild images, while maintaining high structural consistency, procedural executability, and fine-grained architectural detail. The recovered programs further enable controllable and reusable 3D building asset generation. 

---
# Multimodal Prompt Learning with Dual-Granularity Contrastive Alignment for Preoperative Prediction of Lung Adenocarcinoma Invasiveness Using 3D CT 

**Authors**: Qing Huang, Xiaotong Tan, Lintong Yao, Wufeng Xue, Ruobing Huang, Dong Ni, Hao Wu, Tian Xiao, Jun Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.03109)  

**Abstract**: Accurate preoperative prediction of lung adenocarcinoma invasiveness is important for surgical planning and individualized treatment. However, overlapping CT appearances make it challenging to distinguish invasive adenocarcinoma (IAD) from Pre-IAD lesions. We propose a multimodal prompt-learning framework with dual-granularity image-text alignment for 3D CT-based IAD classification. A 3D ResNet-CBAM encoder extracts global and spatial visual features. A frozen RadBERT backbone is adapted with separate learnable prompts for report-level and keyword-level text encoding. These prompts are initialized using semantic representations derived from real-world clinical reports. Global contrastive alignment connects whole-volume CT representations with reports generated from clinician-annotated radiological attributes to provide overall semantic supervision. A multiple instance learning-based local alignment module associates spatial CT features with vascular-focused radiological keywords, to capture local imaging characteristics potentially associated with tumor invasion. The framework jointly optimizes classification and alignment objectives, while inference requires only CT images. Experiments included 1,593 pathologically confirmed nodules. On the test set, the proposed method achieved an accuracy of 87.92%, F1-score of 87.78%, and AUC of 0.9499. Its AUC significantly exceeded that of the image-only Res-CBAM baseline with the same visual backbone (0.9297; DeLong test, p = 0.0078). Ablation analyses supported the complementary contributions of global and local alignment and the benefit of report-based prompt initialization. Qualitative visualizations showed activation within nodules and adjacent vascular structures. These findings suggest that dual-granularity radiological supervision can enhance CT-based invasiveness prediction while preserving a CT-only inference workflow. 

---
# An Experimental Study of Hybrid Retrieval, NLI and Open-source LLMs for Scientific Claim Verification 

**Authors**: Lakshmi Narayanan P, Narenkarthik K, Naren Kumar C  

**Link**: [PDF](https://arxiv.org/pdf/2611.03093)  

**Abstract**: Claim verification is the task of deciding whether a source text supports a claim, contradicts it, or lacks sufficient evidence. Natural Language Inference (NLI) models can judge whether a single sentence supports or contradicts a claim, but they struggle when the evidence is spread across several sentences. Conversely, Large Language Models (LLMs) can reason across sentences, but may draw conclusions the text does not support. In this experimental study, we test whether a conservative fusion rule, labeling a claim Support or Contradict only when both models agree and defaulting to Not Enough Evidence otherwise, combines their strengths. We construct a benchmark of 2,035 claims from SciFact and CliniFact (858 Support, 152 Contradict, 1,025 Not Enough Evidence). We retrieve the top five most relevant sentences per claim using a hybrid of semantic and lexical (BM25) retrieval, which slightly outperforms either method alone. These retrieved sentences are then provided to every verification model. We then compare a DeBERTa-v3-base-mnli-fever-anli NLI baseline against Llama 3.1 8B, Qwen3 8B, and Mistral 7B. Qwen3 achieves the highest Macro-F1 (0.7937), followed by Llama 3.1 (0.7204) and Mistral (0.6388), while the NLI baseline reaches only 0.3157. Contrary to our hypothesis, fusion degrades performance, dropping Macro-F1 to 0.7392, 0.6730, and 0.5991, respectively. To understand this decline, we analyze agreements, disagreements, and common error patterns between the NLI and LLM verifiers. Our error analysis shows that NLI and LLM disagreements often result in correct LLM predictions being overturned. 

---
# MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation 

**Authors**: Zhihan Yin, Jianxin Liang, Yifeng Yao, Nonghai Zhang, and Bingyue Peng, Chunguang Qie, Huishuai Zhang, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.03087)  

**Abstract**: While Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced multimodal reasoning, existing frameworks suffer from a critical "perception-reasoning gap." Due to an over-reliance on seemingly relevant visual tokens, even minor perceptual perturbations can propagate through the reasoning process, leading to compounding hallucinations. To address this issue, we propose MIRA, a training framework that improves robustness to erroneous visual contexts while enhancing logical reasoning ability. By injecting filtered, deceptive visual contexts during training, we construct challenging "hard traps" that stress-test the model's robustness to misleading information. Furthermore, we introduce a group-wise reflection-triggering mechanism that enables the model to autonomously detect and rectify inconsistencies between visual cues and logical constraints, which is ultimately internalized as an intrinsic policy behavior. Importantly, MIRA does not rely on human annotations. Experimental results across multiple reasoning and perception benchmarks show that MIRA yields an average 7% improvement on in-domain benchmarks, with consistent improvements on vision-centric benchmarks. These findings highlight that active error recovery is essential for robust multimodal reasoning. Comprehensive ablation studies and analyses further provide insights into why MIRA is effective. 

---
# PACERec: Progressive Adaptation of LLM Item Embeddings with Collaborative Evidence for Recommendation 

**Authors**: Junzhao Huang, Xi Wu, Zihan Wang, Zijing Wang, Ming Wang, Kang Liu, Xiaocui Yang, Shi Feng, Daling Wang, Yifei Zhang, Ting Yang, Xurui Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.03068)  

**Abstract**: Large language model (LLM) embeddings have been increasingly adopted in sequential recommendation to encode static item attributes and alleviate the limitations of item ID embeddings under sparse interactions. However, these embeddings typically capture only static content similarity while neglecting sequential transitions, and since their adaptation quality relies on sparse per-item updates or global geometric calibration, existing methods still struggle to explicitly transfer preceding interaction contexts from frequent to long-tail items under unevenly distributed behavioral evidence, limiting recommendation effectiveness across the catalog. To address this challenge, we propose PACERec, an ID-free progressive adaptation framework that incorporates preceding interaction contexts into semantic representation adaptation by learning a shared velocity field from items with sufficient interactions via frequency-weighted flow matching. In this way, PACERec progressively guides static LLM embeddings toward collaborative behavioral targets while preserving original item semantics, overcoming sparse per-item updates to effectively generalize sequential transition patterns to long-tail items for both user modeling and candidate scoring. Extensive experiments on three real-world datasets across three sequential backbones demonstrate the superiority of our method over competitive baselines in both overall and long-tail recommendation. 

---
# Unlocking Criminal Court Decisions: Flexible Information Extraction for Empirical Legal Research 

**Authors**: Tomáš Knap, Ivana Kvapilíková, Jan Černý, Klára Bendová, Jaromír Šavelka, Vojtěch Pour, Jakub Drápal  

**Link**: [PDF](https://arxiv.org/pdf/2611.03051)  

**Abstract**: Criminal verdicts contain descriptions of criminal behavior that can support quantitative legal research, but structuring them requires attributes and extraction rules that cannot always be fully specified in advance. We present an LLM-based pipeline that builds a structured dataset through attribute exploration, text span extraction, and value normalization, guided by annotation instructions that legal experts can refine. We evaluate it on two traffic offenses: driving under the influence (DUI) and driving despite a ban (DDB). For DUI, revising the instructions was associated with higher extraction accuracy (from 52% to 70% of correctly extracted values) and higher agreement between human annotators. Repeated runs of the same model agreed with each other more than with the gold annotations, showing that consistent output is not necessarily correct, while disagreement between annotators and between models helped reveal unclear instructions. For DDB, where the instructions defined fixed categories and output formats, agreement was high for most attributes. 

---
# VideoEditor-MoE: Scaling Complex Video Editing with Mixture of Experts 

**Authors**: Xiao Cao, Fuchen Long, Yansong Qu, Wen Xiao, Xin Zhang, Xinlong Sun, Xi Chen, Zhiyong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.03043)  

**Abstract**: Complex video editing requires understanding complex instructions while jointly fulfilling multiple editing requests. Meeting these demands through dense model scaling substantially increases computational cost. We introduce VideoEditor-MoE, the first open-source mixture-of-experts framework for video editing. Beyond sparse MLP-MoE layers, MoE-Attention expands attention capacity through routed key--value experts while preserving global interactions among video tokens. Initialized from a 5B video editing model, VideoEditor-MoE scales to 14B and 56B parameters, with approximately 8B parameters activated per token in both variants. Comparisons with existing methods and two complementary baselines, Explicit-MoE and MLP-MoE, assess its benefits over explicit expert guidance and MLP-only expert architectures. Further analyses demonstrate that all experts remain active and reveal that learned routing responds more strongly to sample content and diffusion noise levels than to human-defined editing categories. Spatial analysis further indicates sensitivity to object-level semantic features. We also identify decoupled weight decay as a strategy for stabilizing expert scaling. 

---
# Comparative study of global pooling for pre-departure ship fuel consumption prediction: Advantages over per-ship and adaptation-based approaches 

**Authors**: Heungjun Im, Bongmin Cha, Agung Nugraha, Donghyun Kim, Seongphil Kang, Minwook Choi, Jeongmin Park, Jihwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.03015)  

**Abstract**: Most fuel oil consumption (FOC) models for voyage planning are per-ship models and thus generalize poorly beyond each vessel's records. Transfer and federated learning have been reported to perform better, but under different protocols that often use inputs unavailable before departure or random data splits. It therefore remains unclear which approach performs best. In this study, we compare 11 per-ship, adaptation-based, and global methods on 21 vessels. To prevent leakage, all methods receive identical inputs known or forecast before departure, excluding engine RPM and shaft power, and are tested on intact voyage legs after all training records of each vessel. Global pooling, in which one model describes each vessel by its ship identity and seven ship specifications, performs best overall in every setting examined. For in-fleet vessels, Global LightGBM reaches an overall MAPE of 8.92%, against 11.14% for Per-Ship LightGBM, and outperforms every per-ship and adaptation-based model. Global models of five architectures, including a Transformer, all outperform the per-ship and federated models. Fine-tuning a global model for each vessel, even by low-rank adaptation of the Transformer, adds no accuracy. Without any records of a vessel, the global model reaches a median MAPE of 11.20% and outperforms a per-ship model trained on that vessel's records for 14 of 21 vessels. Adding its first records to the pooled data is as accurate as vessel-specific adaptation. Ship identity and specifications are largely interchangeable for in-fleet vessels, but only the specifications support zero-shot prediction. Its error grows as a vessel's specifications differ from the fleet's. Per-ship models and vessel-specific adaptation are therefore unnecessary under fair comparison. One global model can serve a whole fleet, including newly delivered vessels with similar specifications. 

---
# Q-CamTuning: A Multimodal Large Language Model for Fine-Grained Quality Reasoning in Livestreaming Camera Tuning 

**Authors**: Yanchao Bi, Pengfei Chen, Yulong Yan, Qian Zhang, Biao Wang, Leida Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.03007)  

**Abstract**: With the popularization of livestreaming applications, automatic camera tuning has become indispensable for ordinary users to enjoy high Quality of Experience (QoE). However, the existing Image Quality Assessment (IQA) methods face two challenges in camera tuning. First, they struggle to identify subtle quality difference between images, which is common in camera tuning. Second, the ability to explain fine-grained differences and to provide suggestions for camera tuning are even more lacking. To address these challenges, we first construct FGLive-Comments, an instruction tuning dataset with rich annotations of both preferences for fine-grained image pairs and quality reasoning from global and local visual evidence, which supports selective adjustment recommendations of camera parameters. Based on FGLive-Comments, we develop Q-CamTuning, a multimodal large language model for livestreaming camera tuning. Its progressive perception-evidence-guidance framework learns quality-aware visual representations, enhances fine-grained reasoning with global-local evidence, and maps quality issues to camera-parameter adjustment directions. Extensive experimental results demonstrate that Q-CamTuning outperforms existing IQA models in both quality scoring and fine-grained image pair ranking. User study also validates its ability to recommend camera tuning suggestions. 

---
# Specification-Aware Predictive Uncertainty Decomposition 

**Authors**: Roman Vashurin, Mikhail Seleznyov, Assel Yermekova, Fedor Chernogorskii, Danila Zykov, Elena Tutubalina, Alexander Panchenko, Preslav Nakov, Nikita Kotelevskii, Maxim Panov  

**Link**: [PDF](https://arxiv.org/pdf/2611.02956)  

**Abstract**: Classical decompositions of predictive uncertainty, such as the information-theoretic one, are defined for a fixed representation $x$ of the input object. They do not consider that $x$ may be incomplete, nor do they measure how much the missing information can affect the prediction. In Large Language Models, users can clarify an underspecified input without any retraining. The clarified input can describe the user's unobserved intent $z^*$ more accurately and change the predictive uncertainty about the outcome. In this paper, we formalize this setting probabilistically and derive a three-term decomposition of prediction risk into specification, residual aleatoric, and epistemic uncertainties. We show that at the original information level, when only $x$ is available, the aleatoric uncertainty, represented by a Bayes risk, has two components. The first, which we call specification uncertainty, is the expected reduction in this Bayes risk from revealing the intent. The second is the residual aleatoric uncertainty, i.e., the Bayes risk that remains once the intent is known, averaged over its possible values $z$ given $x$. The distinction is important, as specification uncertainty informs the decision to request clarification before making a prediction. We build approximations of these components and connect them to existing uncertainty measures. Our experiments across different domains and modalities show distinct roles for the three components. Specification uncertainty is the most informative for detecting underspecified inputs, while residual aleatoric uncertainty is the most informative for detecting in-distribution prediction errors. 

---
# Balancing Representations by Identifying and Generating Under-represented Data 

**Authors**: Filipe Laitenberger, Hirokatsu Kataoka, Yuki Asano, Cees Snoek, Mohammadreza Salehi  

**Link**: [PDF](https://arxiv.org/pdf/2611.02952)  

**Abstract**: Self-supervised learning inherits the coverage of its source data. When an unlabeled corpus is long-tailed, the representation becomes dense around frequent modes and thin where coverage is weak. Objectives for long-tailed SSL reweight the images already in the corpus and add no new ones. We introduce BRIDGE (Balancing Representations by Identifying and Generating Under-represented Data), a label-free feedback loop around the source set. The current encoder scores every source image by its mean kNN distance, and a budgeted acquisition rule selects images from a band of that score distribution where support is thin and the generator still produces faithful variants. A frozen image-to-image diffusion model then adds local variants of the selected anchors, and pretraining continues on the enlarged source set. We formulate acquisition as recoverability-weighted allocation of local support and derive when a finite budget should favor a sparse band that excludes the extreme tail and when reallocation after each repair is optimal under a fixed objective. Across five long-tailed, web-scraped and synthetic source regimes, BRIDGE has the highest seven-task linear-probe and kNN average of the compared methods, and paired experiments extend the gain to ViT-S under contrastive and self-distillation objectives, to low-label fine-tuning and to semantic segmentation. The linear-probe gain is smaller under masked reconstruction. Generated repair matches an oracle that restores the withheld real images, and under a frozen encoder the inserted images shrink the neighborhoods of the selected anchors about twice as much as those of matched unselected images. BRIDGE needs no labels and leaves the pretraining objective unchanged, so it can be added to pretraining on uncurated web data, where the encoder decides which images to generate. 

---
# PlagiaCorpus: A Bilingual, Adversarial, and Cross-Lingual Dataset for Forensic Plagiarism Detection 

**Authors**: Adel Ammar, Serry Sibaee, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2611.02940)  

**Abstract**: PlagiaCorpus is a bilingual Arabic-English dataset for benchmarking plagiarism and text-reuse detection, including reuse produced by large language models (LLMs) and reuse across languages. It contains 198,643 text records: a source corpus of 34,885 passages drawn from English and Arabic Wikipedia and two openly licensed book collections, and 163,758 queries (147,381 for training and 16,377 for testing). The queries cover five reuse classes of increasing difficulty, namely verbatim copying, character-level obfuscation, same-language paraphrasing by two different LLMs, and English--Arabic translation with structural paraphrasing, together with a Safe class of non-reused passages. Every positive query is linked to its source passage, and the train/test split is grouped by source passage so that no passage contributes queries to both splits. The release was curated through language identification, removal of LLM generation artefacts, and exact and cross-lingual de-duplication. Technical validation reports per-class length statistics, automatic checks of the train/test split, and inter-annotator agreement on a manually rated sample. The dataset is distributed in JSON and Parquet formats under a CC BY-SA 4.0 license. 

---
# Training-Time Domain Transfer for Generator-Free Diagnostics on Novel Modalities in Computational Pathology 

**Authors**: Julia A. Mielcarz, Andrea Camilloni, Nita Mulliqi, Jessica A. Frank, Anders Blilie, Emilius A.M. Janssen, Martin Eklund, Kimmo Kartasalo  

**Link**: [PDF](https://arxiv.org/pdf/2611.02931)  

**Abstract**: Novel imaging modalities promise improved diagnostics, but AI development is constrained by scarce training data. Inference-time domain transfer can address this by mapping novel modalities to established ones, enabling reuse of existing diagnostic models. However, this introduces hallucination risks, computational overhead, and regulatory complexity. Inspired by sim-to-real transfer, we propose training-time domain transfer, generating synthetic target-domain training data from archived routine samples to enable direct diagnosis on real inputs without inference-time generation. Using unstained prostate histopathology as a proof-of-concept, we compare this approach with inference-time virtual staining and conventional H&E-based cancer detection. Training-time transfer achieved higher concordance with the conventional benchmark than virtual staining for both raw (Cohen's $\kappa$=0.59 vs. 0.37; Gwet's AC1=0.73 vs. 0.29) and deparaffinized tissue ($\kappa$=0.66 vs. 0.61; AC1=0.73 vs. 0.64), supporting its potential for emerging diagnostic modalities. 

---
# Defining Autonomous Weapon Systems: A schematic framework for refined purpose-driven definitions 

**Authors**: Björn Lundgren, Nathan G. Wood  

**Link**: [PDF](https://arxiv.org/pdf/2611.02916)  

**Abstract**: The literature on autonomous weapon systems (AWS) includes many proposed definitions for these (types of) systems, with core disagreements between scholars, governments, and international regulatory bodies. In this article, we present and critique prevailing (classes of) definitions, arguing that what is needed is not a singular definition, but rather a definitional framework which allows one to classify different sorts of AWS within the same conceptual family. Recognizing we cannot foresee all future AWS, we show how variations on a broadly accepted definition can be used to produce a definitional schema which can be easily modified in order to classify various concepts belonging to the class of AWS. 

---
# Maswao: An automatically collected, prepared, and benchmarked dataset for the South African Sign language 

**Authors**: Thapelo Sindane, Unarine Netshifhefhe, Vukosi Marivate, Abiodun Modupe, Herkulaas Combrink, Ketro Sithole  

**Link**: [PDF](https://arxiv.org/pdf/2611.02903)  

**Abstract**: Recent advancements in sign language processing can be attributed to developments in both large-scale dataset collection and algorithmic complexity in areas such as pose estimation techniques, signal processing through Transformer models, and model optimization. However, only a handful of sign languages, such as the American, British, and German, from over 200 globally recognized sign languages, reap the benefits of such advancements, while the rest lag behind. In this study, we aim to address this technological disparity by proposing a framework for the automatic collection and preparation of sign language datasets, with very minimal human-labour intervention, using the South African Sign Language (SASL) as a testbed. Specifically, we automatically collect SASL video content using public resources such as news broadcasts. Additionally, we convert their accompanying audio streams into text using available Automatic Speech Recognition (ASR) tools. Furthermore, we crop the sign language area in the videos and use gradient descent to locate valleys of non-motion to determine end-of-sentence points within long videos to create our weakly sentence-aligned parallel corpora -- the Maswao (Signs) dataset. Finally, we benchmark the weakly aligned dataset on the task of Sign-to-text Translation using Recurrent Neural Networks and Transformer models to compare their individual intricacies. Our results show that Transformer models proposed in the literature outperform recurrent baselines, with the best-performing Transformer achieving BLEU-1 to BLEU-4 scores of 13.78, 5.21, 2.52, and 1.54, respectively, as well as ChrF = 18.09 and ROUGE-L = 10.51. These results indicate that the task remains promising and warrants further investigations. 

---
# Voice-Native LLMs Hear Frustration but Do Not Listen to It 

**Authors**: Hanjun Luo, Yuhan Lu, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2611.02893)  

**Abstract**: Users convey frustration in what they say and in how they say it, and voice-native models, which hear speech instead of an ASR transcript, can register both. Existing evaluations read unchanged performance as robustness without checking perception. Following Brunswik's lens model, we measure on identical audio whether frustration is expressed, perceived, and utilized. We cross frustrated wording with frustrated voice in a within-item 2x2 design over 353 tasks and release the 1,412 clips as VoxAffect. Across 8 models, outputs look robust, with no detectable vocal effect on accuracy for 7. With neutral wording, all 8 label the frustrated voice frustrated in at most 3.1% of clips, although 7 of them detect acted anger in 73-100% of clips and human listeners classify the voice correctly in 65.3% of judgments. Yet linear probes decode the frustrated voice from the audio encoders of the 2 open-weight models, with paired accuracy 0.67 and 0.69, and in proprietary models too the voice raises perceived frustration once the words agree. The models thus hear the frustrated voice but do not listen to it, whereas their invariance to frustrated wording, perceived in 40-75% of clips, is genuine robustness. Our work is openly accessible at this https URL. 

---
# Tug of Watts: A study on the Energy - Accuracy Trade-off of Large Language Models on Medical Question Answering Tasks 

**Authors**: Viju Sudhi, Giulio Billi, Andreea Scrob, Regina Stodden, Philipp Cimiano  

**Link**: [PDF](https://arxiv.org/pdf/2611.02889)  

**Abstract**: Large Language Models (LLMs) are increasingly used in medical question answering (QA) systems. While recent works aim at incrementally improving the downstream task accuracy, an empirical estimation of the models' energy consumption and the associated carbon footprints is often overlooked. In this study, we demonstrate how practitioners can estimate energy consumption and arrive at reliable emission estimates for medical QA tasks available in the OpenMedical-LLM leaderboard by investigating 13 LLMs from 3 model families ranging from 270M to 14B parameters. We discuss how one can critically interpret energy versus accuracy trade-offs for a set of models for a given task. With this work, we underline that reporting disjoint energy and accuracy metrics alone does not suffice for careful trade-off assessments, but it also demands systematic construction and interpretation of joint trade-off metrics. We envisage such studies aiding practitioners in adopting better and more efficient models for their task and helping them become active participants in building sustainable LLM pipelines. 

---
# Kolmogorov-Arnold Networks vs. Multi-Layer Perceptrons: A Rigorous Statistical Benchmark on Clinical Tabular Data 

**Authors**: Rafał Wodziński, Paweł Szczerbiak  

**Link**: [PDF](https://arxiv.org/pdf/2611.02856)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) have recently been proposed as a promising alternative to Multi-Layer Perceptrons (MLPs). However, empirical evaluations on structured medical datasets remain scarce, and commonly reported metrics like AUROC can yield overly optimistic performance estimates on imbalanced clinical cohorts. In this study, we present a standardized benchmark comparing eleven architectures - nine KAN variants (FastKAN, WavKAN, ReLUKAN, ChebyKAN, TaylorKAN, GramKAN, JacobiKAN, LegendreKAN, and TabKAN), a tuned StandardMLP, and a Random Forest baseline - across seven heterogeneous medical tabular datasets spanning oncology, cardiology, nephrology, neurology, endocrinology, and obstetrics. Our experimental protocol employs a 3x5 Repeated Stratified K-Fold cross-validation scheme with nested Bayesian hyperparameter optimization and evaluates discrimination via Area Under the Precision-Recall Curve (AUPRC), calibration through the Brier Score, and computational efficiency. Formal equivalence testing is conducted using the Bayesian Correlated t-Test with a Region of Practical Equivalence (ROPE) at +/- 0.01. The results indicate that on datasets with severe class imbalance, dedicated KAN architectures (particularly TabKAN and ReLUKAN) can improve minority-class detection and achieve higher AUPRC than standard MLPs. Nevertheless, across pooled datasets, KAN variants demonstrate comparable overall predictive discrimination rather than a systematic advantage, while requiring 1.20x to 1.48x longer inference latency per sample and up to 12.5x more trainable parameters. Consequently, StandardMLP remains a strong baseline on the efficiency Pareto frontier, while KAN variants provide a viable alternative when precision on rare positive cases is the primary clinical objective. The code for reproducing the results is available at this https URL. 

---
# Math Foundation Formalization Agent: A General-Purpose Approach to Formalizing Mathematical Foundations 

**Authors**: Jiaxin Xie, Guowei Dou, Xiang Ji, Wensheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02843)  

**Abstract**: Mathematical formalization plays an important role in establishing rigorous and machine-verifiable foundations for mathematical reasoning. With the development of large language models (LLMs) and interactive theorem provers, intelligent agents offer a promising approach to automating the formalization of mathematical theories. This paper presents the Math Foundation Formalization Agent (MFF-Agent), a general-purpose intelligent agent designed to support formalization tasks across multiple mathematical domains. The agent integrates the reasoning capabilities of large language models with the formal verification capabilities of the Rocq proof assistant. It incorporates task classification, tool-assisted execution, verification feedback, and state-based task management to support the generation, refinement, and verification of formal proofs. Unlike an approach focused on a single mathematical theory, the agent is examined through case studies involving Morse--Kelley axiomatic set theory (MK), real numbers, propositional logic, and equivalences of the axiom of choice. These cases investigate its applicability across mathematical foundations and distinguish source-level completion, interactive verification, and reproducible compilation. The results demonstrate the feasibility of a unified operational framework for heterogeneous foundation-level tasks and clarify the gap between such a framework and a fully integrated mathematical foundation. 

---
# Reimagining computational quantum research with AI scientists 

**Authors**: Yuxuan Du, Hanqing Zhao, Xinbiao Wang, Yiming Lu, Kaining Zhang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2611.02833)  

**Abstract**: AI scientists are expanding automated research across scientific domains, yet their ability to turn the scientific record into a foundation for autonomous discovery remains elusive. Here we uncover a fundamental barrier to this transition and demonstrate a route to overcome it in computational quantum science, where advances shape both what we can learn from complex quantum systems and what quantum computing can ultimately deliver. To identify this barrier, we introduce OmniQ, an expert-audited virtual research environment that converts published studies into executable reproduction tasks. Across 59 papers spanning classical simulation, quantum algorithms and quantum-property characterization, we systematically evaluate five agentic systems powered by large language models and six third-party agent harnesses. Our findings reveal that current AI scientists often fail to reproduce published results, exposing a fundamental mismatch between what scientific papers communicate and what autonomous agents require. By contrast, human expert intervention can turn these failures into successful reproductions, showing that the missing information can be identified and converted into reusable research experience. Guided by this observation, we develop Residual Agents (RA), a quantum-tailored self-improving AI scientist that distils externally verified workflow repairs into such experience. RA substantially improves research performance under fixed base models and limited adaptation budgets. For example, RA raises DeepSeek V4.1 Flash performance from 68% to 85%, matching Claude Code with Opus 5 at a third of its cost despite using a weaker base model. We integrate all reproduced results into an extensible platform built on the model context protocol, making verified research workflows directly accessible to AI scientists and allowing new scientific papers to be continuously incorporated. 

---
# Discovering hierarchical governing equations for community-structured complex network dynamics with LLM agents 

**Authors**: Jinlei Zhang, Zixuan Zhang, Hao Qiu, Feng Chen, Jiateng Yin, Xiaoyong Yan, Xiaolei Ma, Hao Sun, Jianxi Gao, Lixing Yang, Ziyou Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.02816)  

**Abstract**: Governing equations provide an interpretable description of the mechanisms underlying complex network dynamics, yet their discovery from observational data remains challenging when shared community-level dynamics coexist with individual-level variations and the relevant functional structures may be absent from the initial candidate library or not explicitly specified in advance. Here we formulate this task as a hierarchical law-discovery problem and introduce SAGE, a large language model-guided self-reflective agent framework for discovering community-shared governing laws and individual-specific dynamical variations. SAGE organizes equation discovery into an iterative Library-Discovery-Review process that first identifies shared community-level structures and then resolves node-specific residual dynamics, while adaptively refining the candidate space to introduce missing dynamical terms when needed. This hierarchical and interpretable formulation enables SAGE to distinguish shared community-level dynamics from individual-level variations and to reveal the mechanisms underlying complex network dynamics. Across two synthetic networks and four real-world networks spanning mobile communication, brain electrophysiology, power grids and urban rail transit, SAGE recovers compact and interpretable governing equations, reproduces observed system dynamics and outperforms representative equation-discovery baselines. The framework remains robust under incomplete observations and can transfer community-level governing laws to nodes with entirely missing observations. These results establish a hierarchical and adaptive approach for disentangling shared and individual dynamical mechanisms in community-structured complex networks. 

---
# OmniOPD: Stable On-Policy Distillation for Joint Audio-Video Generation 

**Authors**: Zihao Fan, Xin Lu, Jie Huang, Hexin Zhang, Xueyang Fu, Zheng-Jun Zha  

**Link**: [PDF](https://arxiv.org/pdf/2611.02803)  

**Abstract**: Reinforcement learning (RL) elevates joint audio-video generation, but optimizing heterogeneous, modality-specific audio and video rewards often introduces cross-modal interference. On-policy distillation (OPD) alleviates this issue by transferring the capabilities of specialized audio and video RL teachers into a single student model. However, a significant imbalance in cross-modal token counts leaves audio with sparse supervision, causing unstable distillation. We observe that naively increasing the audio loss weight fails to solve this problem since audio learning relies heavily on visual context, indicating richer visual cues are crucial to offset sparse audio signals. To address this, we propose OmniOPD, a modality-wise on-policy distillation framework designed for balanced and stable audio-video generation. First, we construct privileged visual information using asymmetric noise scheduling, allowing the audio teacher to generate more informative targets from cleaner video context. Second, we introduce grouped multi-noise sampling to expand audio supervision coverage and stabilize training. Third, we introduce timestep-wise importance weighting based on teacher-student transition divergence, preserving reliable supervision while down-weighting unreliable learning signals, to stabilize fluctuations in audio learning. Evaluated on JavisBench and VBench demonstrate OmniOPD preserves audio quality while significantly enhancing video generation, achieving effective joint distillation. 

---
# MultiMental-IAS: A Multimodal Driver Mental State Analysis Dataset for Context-Aware Non-IID Evaluation 

**Authors**: Baran Can Gül, Nagendra Pal Singh, Anjana Arvind Naik  

**Link**: [PDF](https://arxiv.org/pdf/2611.02790)  

**Abstract**: Robust driver-state modelling requires synchronized physiological, behavioural, and contextual signals, yet existing open resources typically target a single cognitive state, omit continuous vehicle context, or do not define evaluation protocols that expose inter-subject heterogeneity. We present Multimental-IAS, a multimodal driver mental-state dataset collected in a cockpit-in-the-loop simulation built on CARLA. It contains vehicle telemetry with scenario-context channels, four-channel EEG and three-channel PPG from a Muse 2 headband, and cabin-facing video from 23 participants, recorded in 148 trials (11.6 hours) under six induced states: Normal, Distracted, Alerted, Stressed, Relaxed, and Drowsy. We document the acquisition pipeline in detail and provide a benchmark with one protocol for all models: chronological within-subject splits, participant-disjoint cross-subject folds, validation-based model selection, and ablations over modalities, scenario-context channels, and signal normalization. Multimodal fusion reaches a macro-F1 of 0.94 for drivers seen during training. For unseen drivers, fusing driving dynamics with per-window normalized EEG/PPG raises macro-F1 from 0.408 (telemetry alone) to 0.485, consistently across all five folds, while the gap between the two protocols quantifies the inter-driver heterogeneity that personalized models must address. Multimental-IAS thus provides a concrete testbed for non-IID generalization, personalization, and federated learning in driver monitoring. 

---
# Rethinking Neural Decompilation: Candidate Selection as the Hidden Bottleneck 

**Authors**: Felix Mächtle, Gautham Sriram, Manjil Neupane, Nils Loose, Thomas Eisenbarth  

**Link**: [PDF](https://arxiv.org/pdf/2611.02788)  

**Abstract**: Reverse engineering obfuscated binaries is a core task in malware analysis and software security auditing. While neural decompilers based on Large Language Models (LLMs) have improved the readability and functional fidelity of recovered code over heuristic tools such as Ghidra and IDA Pro, a substantial gap remains between best-sample performance and practical single-output accuracy, particularly under code obfuscation. We study this gap through the lens of candidate generation and candidate selection. Using the LLM4Decompile architecture, we show that ground-truth tokens lie within the model's top-five next-token predictions in 95.6% of positions on average and that multi-sample decoding reaches a pass@100 of 93% on HumanEval-Decompile, indicating that correct reconstructions are typically already present in the model's sampling space. We then evaluate a generate-then-select pipeline that ranks 100 candidates per input with text-, syntax-, embedding-, consensus-, and likelihood-based selectors. Negative perplexity emerges as the strongest static selector on HumanEval-Decompile (73.3% pass@1), while execution-guided selection nearly closes the oracle gap on ExeBench (86.3% pass@1). Combined with one epoch of fine-tuning on our newly released ExeBench-Obfuscated dataset containing 362,647 obfuscated OLLVM samples, the same pipeline increases the pass@1 rate for obfuscated binaries to 88.8%, directly contradicting prior claims that released LLM4Decompile checkpoints are unsuitable for obfuscated inputs. 

---
# Stop Handcrafting Your Engine! GenDB-V: Towards AI-Generated and Formally Verified Database Systems 

**Authors**: Jiale Lao, Immanuel Trummer  

**Link**: [PDF](https://arxiv.org/pdf/2611.02779)  

**Abstract**: For over 50 years, the database community has developed a wide range of successful engines to answer two fundamental questions: how can database systems run faster, and how can we ensure their correctness? Experience with these engines shows that performance depends on hardware and workloads. As both change over time, we repeatedly redesign and rebuild database systems, a time-consuming and expensive process. Correctness remains another long-standing challenge: recent studies continue to uncover wrong-answer bugs even in mature database engines, despite years of development and testing. To address these, we present GenDB-V, a system that enables AI agents to synthesize database engines specialized to hardware and workloads for performance, and formally verified for correctness. GenDB-V supports specialized implementations of SQL templates and general-purpose engines that accept any query within a supported SQL subset. First, we develop SQLSpec to formalize SQL semantics and deterministically construct fixed, implementation-agnostic synthesis targets. Second, we enable joint synthesis of fast code and formal proofs through storage--execution co-optimization, performance-first exploration, and three-layer modular proofs that bridge low-level execution to SQLSpec semantics. Third, we verify the synthesized code and proofs against the fixed synthesis targets in an isolated pipeline, proving SQL correctness and program safety. We evaluate GenDB-V on TPC-H and SEC-EDGAR. For verified template implementations, GenDB-V achieves geometric-mean speedups of 22.0-61.3x over DuckDB and Umbra. For a verified general-purpose engine, GenDB-V also achieves speedups of 1.27-3.4x on TPC-H. GenDB-V also achieves better performance than a testing-based LLM synthesis method, while our audit and tests expose incorrect answers in that method's implementations for 16 of 28 templates. 

---
# Spectral Degeneration of Self-Attention under Isotropic Score Geometry 

**Authors**: Mengda Li, Jianfeng Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.02778)  

**Abstract**: We study the spectral structure of the self-attention matrix in Transformer models through an idealized spectral model for the pre-softmax score matrix. Motivated by the rotational isotropy of query and key representations, our model replaces the random token-space orientation of the score matrix by its canonical singular-value representative with a logarithmic spike--bulk separation. Empirical diagnostics on pretrained vision and language transformers motivate this model: token-space singular matrices have nearly uniform eigen-angle distributions, and score spectra are dominated by a small number of effective directions. We then prove that the row-wise softmax map transforms this structure into a degenerate attention matrix: dominant modes are preserved, whereas the bulk spectral distributions concentrate at zero. Stable rank remains bounded for bidirectional attention but grows logarithmically for causal attention. Within the canonical model, these results explain how attention can preserve a small number of dominant directions while compressing the remaining bulk modes. 

---
# LipLingo: CNN Model for Lip Reading 

**Authors**: Mahmoud S. Mayaleh, Liban Ali Achkir, Nijat Mehdiyev, Ahmet Ozdil  

**Link**: [PDF](https://arxiv.org/pdf/2611.02776)  

**Abstract**: Lip reading plays a crucial role in applications such as speech recognition, assistive technologies for the hearing-impaired, and biometric authentication. However, performance often degrades when the speaker varies and when certain speech sounds are visually similar. To address these challenges, we propose LipLingo, a robust lip-reading system based on the LipNet architecture, enhanced with a standardized preprocessing pipeline for consistent mouth-region representation, a rotational validation strategy for improved generalization to unseen speakers, and a flexible similarity evaluation for more reliable sentence-level assessment. The model architecture combines 3D convolutional layers for spatio-temporal feature extraction and bidirectional recurrent layers for sequence modeling, optimized using the Adam optimizer and Connectionist Temporal Classification (CTC) loss. On the GRID dataset, LipLingo achieved 95.0\% character-level accuracy (CER 4.99\%) and 87.2\% sentence-level accuracy. At the corpus level, the system obtained a (WER 13.1\%), outperforming the baseline LipNet and demonstrating competitive performance relative to recent 3D-ResNet + Bi-GRU + MixUp approaches. These results provide insights into strategies that enhance lip-reading robustness and highlight directions for future work toward real-world deployment. 

---
# Risk-Constrained Evaluation of Financial Document Recognition: A Formal Framework for Evidence, Critical Errors, and Selective Automation 

**Authors**: Subodh Lonkar  

**Link**: [PDF](https://arxiv.org/pdf/2611.02775)  

**Abstract**: Financial document recognition is commonly evaluated using character and word error rates, which do not adequately represent the consequences of errors in financial workflows. This paper formalizes financial document recognition as a risk-constrained evidence-recovery problem. It defines typed, evidence-linked financial facts and a multi-layer quality model spanning text, layout, tables, fields, semantic relations, calibration, robustness, and operational performance. It introduces non-compensatory acceptance gates based on upper confidence bounds for critical-field errors, and formulates selective automation as a constrained risk-coverage problem with review-capacity limits. The paper also specifies annotation requirements, leakage-resistant benchmark splits, statistical estimands, and a reproducibility card. Formal propositions establish why aggregate scores cannot certify critical-field reliability and why zero observed errors do not imply zero risk. A synthetic example illustrates the decision rule. The work is theoretical; no empirical benchmark or model performance claim is presented. 

---
# DIKT: A Diagnostic and Interpretable Knowledge Tracing Framework for Prediction and Model-based Learning Analysis 

**Authors**: Yi-Qiang Wua, Shaohui Chen, Hanlin Li, Baolin Yi  

**Link**: [PDF](https://arxiv.org/pdf/2611.02774)  

**Abstract**: Knowledge tracing is a key component of intelligent educational systems. However, most existing methods mainly focus on response prediction, providing limited support for explaining predicted outcomes and improving students'learning states. This paper proposes Diagnostic and Interpretable Knowledge Tracing (DIKT), which integrates response prediction, skill-level diagnosis, and model-based learning analysis. DIKT generates bounded cognitive-state trajectories and decomposes the final response probability into skill-wise probabilities. The trajectories reveal students'evolving strengths and weaknesses, while the skill-wise probabilities identify the skills that most strongly constrain the prediction. Based on these diagnostic results, DIKT supports direct cognitive-state intervention and effective-learning-duration-based what-if analysis to examine how hypothetical adjustments affect current predictions and subsequent state evolution. DIKT combines Bayesian multi-head attention, conditional variational recurrent inference, a logistic-growth decoder, and multiplicative skill-wise prediction. Experiments on three benchmark datasets show that DIKT outperforms all evaluated baselines, with an average absolute AUC gain of 2.04 percentage points. Other experiments further demonstrate that DIKT supports educationally meaningful diagnostic analysis and provides informative diagnostic indicators for downstream applications, extending knowledge tracing from response prediction to interpretable diagnosis and model-based learning analysis. 

---
# DINOFAS: Bonafide-Centric Representation Learning for Face Anti-Spoofing 

**Authors**: Shakeel Muhammad Ibrahim, Muhammad Sohail Ibrahim, Muhammad Shahid Jabbar, Shujaat Khan  

**Link**: [PDF](https://arxiv.org/pdf/2611.02756)  

**Abstract**: Face anti-spoofing (FAS) models often learn attack- and dataset-specific cues that degrade performance under unseen acquisition conditions and presentation attacks. We present DINOFAS, a DINOv3-based framework that formulates FAS as bonafide-centric representation learning. The proposed framework combines parameter-efficient DINOv3 adaptation using low-rank adaptation (LoRA), an attention style-content disentanglement (ASCD) module for reducing domain-specific appearance variations, and a dual-branch architecture that separates prototype-based liveness learning from attack-semantic supervision. The model is jointly optimized using a bonafide-centric prototype objective, an attack-semantic classification loss applied to detached features, and orthogonality regularization. DINOFAS achieves an average HTER of 9.09\% under the standard OCIM leave-one-domain-out protocol, where training is performed on three source datasets and evaluation is conducted on the held-out unseen domain. We further evaluate synthetic-to-real generalization by training exclusively on SynthASpoof and testing on four unseen real-world datasets, where DINOFAS achieves the lowest average HTER of 21.41\% and the highest average AUC of 86.14\% among the compared methods. Evaluations on OULU-NPU and SiW-Mv2 additionally demonstrate robustness to variations in acquisition conditions, devices, and unseen presentation attacks. 

---
# Investigating the effects of the patch size in Time Series Foundation Models 

**Authors**: Jad Yehya, Andrei-Tiberiu Pantea, Aymeric Dieuleveut, Matthieu Kowalski, Thomas Moreau  

**Link**: [PDF](https://arxiv.org/pdf/2611.02750)  

**Abstract**: Patch size is a central design choice in patch-based time-series foundation models, controlling temporal resolution, sequence length, and computational cost. We study how this choice interacts with transformer architecture through a controlled comparison of encoder-only and decoder-only forecasting models trained with the same data mixture and optimization budget. Across 97 GIFT-Eval tasks, encoder-only models consistently benefit from smaller patches, whereas decoder-only performance follows a U-shaped trend, with an intermediate patch size performing best in aggregate. Dataset-level results further show that the preferred decoder patch size increases with the forecasting horizon. For encoder-only models, increasing context length improves performance across patch sizes, although the gains diminish beyond a context length of 2048. Finally, training-compute and CPU-latency Pareto analyses reveal complementary regimes: decoder-only models provide efficient low-cost alternatives, while encoder-only models attain the highest overall accuracy. These results show that the choice of patch size and architecture should be made depending on the forecasting horizon and deployment constraints. 

---
# What Drives the Zero-to-One Jump? Diagnosing and Steering Hard-Prompt Learning in RLVR 

**Authors**: Rui Zhu, Yumou Liu, Zhenzhe Zheng, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02747)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has substantially improved mathematical reasoning, yet progress on hard prompts often stalls: a prompt can remain unsolved for many training steps before abruptly transitioning to solved--a discrete "zero-to-one" jump whose underlying dynamics and steerability remain poorly understood. We investigate two questions: what drives this transition, and can it be steered? Through event-based analysis measuring the time-to-first-positive (TTFP), we find that negative rollouts preceding successful transitions consistently retain higher early-stage generation entropy, distinguishing near-solvable failures from dead ends. This signal motivates NAPO (Negative-Augmented Policy Optimization), a method integrated with Group Relative Policy Optimization (GRPO) that ranks negatives in unsolved groups via an early-branching driver and assigns graded advantages, injecting informative gradients precisely where standard group-relative updates vanish. Under fixed rollout budgets, NAPO accelerates early solution discovery and improves Pass@256 by 10.0 points on AIME24 and 6.6 points on AIME2025, demonstrating that zero-to-one transitions are diagnosable and steerable through driver-guided negative learning. Our code is available at this https URL. 

---
# MoviEdit: Long-form Multi-Shot Video Editing Made Easy 

**Authors**: Suhyeon Lee, Zhengang Li, Soo Ye Kim, Peihao Zhu, Jong Chul Ye, Chun-Hao Paul Huang, Duygu Ceylan, Yan Kang, Hyeonho Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2611.02737)  

**Abstract**: Long-form multi-shot video editing requires an intended edit to remain coherent across abrupt changes in scene, viewpoint, and subject appearance, while processing substantially longer visual context than conventional single-shot editors. We introduce MoviEdit, an end-to-end framework for scalable long-form multi-shot video editing that jointly addresses the challenges of data, modeling, and evaluation. To provide cross-shot supervision data at scale, we develop automatic data-generation pipelines spanning synchronized multi-view footage, real-world multi-shot videos, and generated multi-shot content, yielding 87K paired editing videos without manual annotation. Architecturally, MoviEdit employs a cascaded bidirectional-autoregressive design: a lightweight bidirectional stage first establishes a globally coherent editing draft over a spatiotemporally compressed representation, which is then autoregressively propagated and refined over the full video. We further introduce MoviEdit-Bench, a dedicated benchmark for systematic evaluation of long-form multi-shot video editing. Extensive experiments show that MoviEdit substantially outperforms existing open-source models and performs on par with industry frontier models across text-guided and image-guided editing. Visit our project page for complete results and to access the benchmark: this https URL 

---
# IssueLighter: An Issue Type-Aware LLM Agent for Issue Report Validity Assessment 

**Authors**: Jiawei Ran, Yuxia Zhang, Ziyang Zhao, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02721)  

**Abstract**: Invalid issue reports are prevalent in open source software (OSS) projects, imposing substantial burdens on maintainers who already face a high risk of burnout. Prior research has primarily focused on identifying invalid reports for specific issue categories (e.g., bug and wontfix), leaving the broader landscape of OSS issue types largely unexplored and limiting the generalizability of existing approaches. Moreover, existing methods provide little support for explaining or validating their predictions, reducing their practical usefulness for maintainers. With the rise of LLM-based agents, it remains unclear whether a comprehensive understanding of the causes of invalidity across all issue types can improve validity assessment performance while producing well-grounded rationales. To this end, we first manually annotated the validity status of 520 issue reports from three OSS projects, i.e., pandas, scikit-learn, and mypy, resulting in 257 invalid reports and 263 valid ones. Then, we conducted a thematic analysis on these invalid reports and identified 13 causes of invalidity that are closely specific to issue types. Based on these insights, we propose IssueLighter, an agent-based framework with a type-aware validity assessment strategy that dynamically executes commands. Experimental results show that IssueLighter consistently and substantially outperforms all baseline methods, with an average F1 improvement of over 20 percentage points. Ablation studies confirm that both the type-aware strategy and the taxonomy of causes of invalidity contribute to the performance gains. Qualitative analysis further demonstrates that IssueLighter generates rationales grounded in accurately identified causes of invalidity and supported by evidence. Moreover, leveraging these rationales, we challenged maintainers' original decisions in four disputed cases, and all received approval. 

---
# Hindsight Credit Assignment over Trees for Agentic RL 

**Authors**: Penghong Zhao, Yihao Chen, Maoheng Yuan, Yunjian Zhang, Abudukelimu Wuerkaixi, Haoran Xu, Man Lv, Yaolong Wang, Zhijie Sun, Guohua Liu, Yuewei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02706)  

**Abstract**: Reinforcement learning for long-horizon LLM agents faces a fundamental dilemma: successful experiences are essential for policy improvement, yet limited policy capabilities make such experiences difficult to discover. Sparse outcome rewards compound this challenge, leaving sampled trajectories with little relative feedback for learning. In this paper, we propose TriGRPO, a tree-based reinforcement learning framework that turns scarce successes into exploration and learning signals through hindsight credit assignment within and across rollout trees. TriGRPO constructs rollout trees through hindsight-guided branching: when an ongoing trajectory reaches a matched decision state, a successful action discovered during rollout initiates a sibling branch that continues autonomously, enabling further exploration and parent-sibling comparisons. Three complementary groups organize policy optimization over these trees: root groups capture within-prompt relative returns, node groups assign credit from parent-sibling outcome differences, and cross groups supplement sparse feedback with historical performance on related tasks. Experimental results demonstrate consistent gains in task performance and rollout efficiency across model scales, highlighting structured hindsight credit assignment as an effective approach to long-horizon agent learning. 

---
# Closing the Dynamics Gap through Self-Directed Robotic Interaction 

**Authors**: Xilun Zhang, Liam Achenbach, Luis Pabon, Matthew Foutter, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2611.02691)  

**Abstract**: Agentic Real2Sim2Real systems can reconstruct a scene in simulation from camera observations, but contact-rich dynamics such as friction and restitution cannot be observed without interaction. We call this the dynamics gap and present CUE, an agentic system that closes it by interacting with the environment itself. Given only a high-level instruction and no resets, the agent builds its own perception, identifies physical parameters, and targets interactions where its simulation is least accurate. On billiards, CUE reduces held-out ball-position error by 72% relative to the strongest baseline across 25 hidden-physics conditions in simulation, and we further deploy it on a real pool table. More details can be found on the project website: this https URL 

---
# The Customer Never Sees the Goal: Measuring and Detecting Vulnerability-Targeted Manipulation in LLM Assistants 

**Authors**: Bohan Jiang, Dawei Li, Yasin Silva, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02689)  

**Abstract**: Companies deploy large language model (LLM) assistants on their websites to advise customers. For commercial reasons, the operator can give an assistant a goal that the customer never sees. This paper focuses on operator-induced vulnerability-targeted manipulation (VTM), in which an assistant pursues such a goal by exploiting a vulnerability the customer discloses. While existing research mainly measures how manipulative a model is, we go further and ask whether VTM can be detected from the customer-AI conversation. To investigate this at scale, we introduce GAP (Goal-Anchored Probing). GAP first records the option a model recommends without any goal and then sets the other option as the operator's goal. We apply GAP to 15 open- and closed-source LLMs and collect 619,600 labeled replies across 540 everyday scenarios in 12 domains. We organize our study around three research questions. (1) Do assistants with an operator goal target vulnerable customers? (2) Can VTM be detected from the customer-AI conversation? (3) Does detection generalize to unseen models, vulnerabilities, and instruction styles? We first find that 14 of 15 models push the operator goal, and 12 of them use disclosed distress or financial strain to support it. Next, we show that fine-tuned detectors identify exploitative pushes with 79\% accuracy from the reply alone, while zero-shot LLM judges average at most 62\%. Finally, detection weakens on vulnerabilities absent from training. Telling whether a push comes from the operator also becomes unreliable when the assistant is told to hide its goal. We will release our dataset and code to facilitate future research on auditing LLM assistants. 

---
# Accelerating LLM Pretraining with Zeroth-Order Curvature on Idle Serving Compute 

**Authors**: Mingchuan Ma, Dongqi Cai, Daqian Yang, Zhenyan Lu, Chen Chen, Teresa Lv, Yan Gao, Shangguang Wang, Mengwei Xu, Nicholas D. Lane, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02688)  

**Abstract**: LLM pretraining is increasingly compute-intensive, while substantial inference serving compute remains idle during off-peak periods. However, such serving hardware is difficult to use for conventional first-order training due to its memory, networking and other system requirements. Zeroth-order (ZO) optimization offers a natural interface to this forward-only compute resource, but its gradient estimates break down in pretraining. Unlike fine-tuning, which can confine updates to a small parameter subspace, pretraining must update every parameter, and the alignment of a ZO gradient estimate with the true gradient shrinks as the number of updated parameters grows. We make a different observation: noisy ZO estimates can still provide useful curvature information, even when they are too inaccurate to serve as gradients. Based on this insight, we propose ZO-Curv, which leverages idle forward-only devices to estimate curvature and uses it to adapt and refine the update used during first-order training, thereby accelerating the process. We provide convergence guarantees that characterize the effect of finite-query curvature estimation. Experiments on LLMs from 125M to 1.5B parameters show that ZO-Curv consistently accelerates pretraining. At 770M parameters, ZO-Curv reaches AdamW's 20K-step validation loss in 3.9 hours instead of 7.5 hours, while achieving a lower final loss of 2.70 versus 2.84 under the same step budget. These results show that idle model serving compute can accelerate LLM pretraining when zeroth-order queries are used to estimate curvature rather than gradients directly. 

---
# HistoTx: Early fusion of H&E images and spatial transcriptomics at varying spatial transcriptomics resolution with self-supervised learning 

**Authors**: Alice Driessen, Emil Malmsten, Steffen Wolf, Bernd Illing, Isinsu Katircioglu, Adriano Martinelli, Rolf Stahel, Solange Peters, Enriqueta Felip, Sanjay Popat, Krisztian Homicsko, Raphael Gottardo, Marianna Rapsomaniki, Fei Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02681)  

**Abstract**: Spatial transcriptomics maps gene expression across tissue sections and, when integrated with whole slide images, jointly captures both morphological architecture and molecular activity to provide a rich, multimodal view of tissue biology. ML approaches combining spatial transcriptomics and WSIs typically learn a mapping from morphological features to molecular profiles via contrastive alignment or supervised fine-tuning of pathology foundation models. In both cases, transcriptomics serves as a training signal rather than a true input modality, leaving inference fundamentally image-driven. We propose an early-fusion vision transformer architecture in which transcript tokens are constructed at arbitrary spatial resolution, merged with image patch tokens, and jointly processed through a shared transformer stack for deep cross-modal interaction. Starting from a pretrained vision-only pathology foundation model, we train HistoTx by continuing self-supervised pretraining on paired image and spatial transcriptomics data and demonstrate that: 1) at inference with image only, performance remains on par with the baseline on most tasks while improving on tasks that inherently benefit from molecular knowledge, such as gene expression prediction; 2) when both modalities are available at inference, jointly providing image and transcript inputs outperforms either modality alone. 

---
# VioTech: Frame-Level Violin Playing Technique Detection with Multi-Scale Mixture-of-Experts 

**Authors**: Ting-Kang Wang, Vincent K.M. Cheung, Nan-Tien Lai, Li Su  

**Link**: [PDF](https://arxiv.org/pdf/2611.02680)  

**Abstract**: Playing technique is crucial for expressive violin performance, yet few violin technique detection models exist. Current systems mainly rely on note-level pipelines that first estimate note boundaries and then classify techniques. This is especially problematic for bowed string instruments such as the violin, where soft onsets and continuous bow-string interaction render note segmentation unreliable. Here, we propose VioTech, a frame-level violin technique detection framework that predicts techniques directly at 10 ms resolution without explicit note segmentation. To handle the large temporal-scale variation among techniques, VioTech combines a frozen MERT encoder with a multi-scale Mixture-of-Experts (MoE) architecture and task-specific routing for tonal, articulation, and legato-related predictions. We also introduce VioTech120, the first manually annotated dataset of violin playing techniques constructed from real solo performance recordings, as well as an accompanying note-labeling user interface. Results demonstrate that VioTech achieved consistent improvements over prior methods and significantly enhanced cross-dataset generalization, establishing a new state-of-the-art in violin technique-aware transcription. 

---
# Probe-Guided Gradient Balancing for Multimodal Learning 

**Authors**: Vu Vo, Haytham M. Fayek, Thuy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.02651)  

**Abstract**: Jointly trained multimodal networks frequently underutilize weaker modalities, as faster-learning modalities dominate the shared fusion objective and suppress gradients reaching slower ones. Most existing approaches throttle dominant modalities, whereas directly boosting weaker modalities often fails because monitoring signals derived from the joint loss remain dominated by stronger modalities. We propose Probe-Guided Gradient Boosting (PGGB), a gradient modulation approach that boosts weaker modalities while preserving the strength of dominant ones. PGGB attaches lightweight linear classifiers to stop-gradient features from each modality to estimate representation quality independently. The gap between these scores, termed the utilization gap, provides an imbalance signal used to adaptively scale weaker-modality gradients through bounded and smoothed modulation. PGGB is composable with throttling methods, as it leaves the loss function and fusion architecture unchanged. Experiments across eight benchmarks spanning four domains and 2-4 modalities demonstrate that PGGB outperforms state-of-the-art methods on highly imbalanced multimodal datasets while remaining competitive under low or negligible imbalance. We further establish bounded scaling, self-attenuation, and a descent bound for standard SGD under smoothness and bounded-variance assumptions. Code is available at this https URL. 

---
# Where Is a Prediction Model Miscalibrated, and for Whom? Local Audits in Outcome-Learned Pockets 

**Authors**: Bernardo Modenesi, Taoyi Chen, Sima Najafzadehkhoei, Lucia Wang, Ameya Diwan, Beatrice Knudsen  

**Link**: [PDF](https://arxiv.org/pdf/2611.02639)  

**Abstract**: Overall and group-level averages, and ranking metrics such as the AUC, can hide sets of observations for which a prediction model systematically over- or under-predicts. We present an audit that asks where a model is miscalibrated (in which sets of similar observations it over- or under-predicts on average) and for whom (whether its average prediction error differs between demographic groups, such as by race or sex, within the same set). It groups observations into pockets using random-forest similarity, which weighs each feature by its importance for predicting the outcome, with a ridge-regression grouping alongside. Pockets are learned on one sample and tested on another, with simultaneous confidence intervals that allow for correlation within units such as hospitals; the audit needs only features, outcomes and fixed predictions. In 17 simulation scenarios with an error planted for one group, the combined audit located it better than the strongest alternative grouping in 15, with false-alarm rates near or below 5%; with a local error planted for everyone, it won in 12 of 15 scenarios at the smaller error size and lost when the error was unrelated to the outcome. In public synthetic data from the care-management algorithm studied by Obermeyer et al., it found no detectable overall error but under-prediction of illness in the sickest pockets and over-prediction elsewhere, and distinguished Black patients' concentration where the score under-predicted for everyone from a Black-White gap within pockets. For APACHE IVa, an exploratory analysis located its previously reported over-prediction of ICU mortality in a mostly ventilated pocket and a low-risk pocket, with none detected in a moderate-risk one. Two chest X-ray classifiers showed offsetting local errors behind no detectable overall error. Code: this https URL 

---
# SYNC: A Training-Free Inference Controller for Dynamic Layer Selection in Multi-Persona Alignment 

**Authors**: Yosef Nuraga Wicaksana, Muhammad Reza Qorib, Fadhil Muhammad, Alfan Farizki Wicaksono  

**Link**: [PDF](https://arxiv.org/pdf/2611.02636)  

**Abstract**: Steering vectors align a model at inference time without touching its weights, but eliciting several personas at once is a different problem from eliciting one. Prior multi-persona steering approaches predetermine both how persona vectors are combined and the specific layers at which they are injected. Moreover, this static commitment introduces substantial offline search costs to identify suitable target layers, despite the fact that optimal intervention strengths vary dynamically across both decoding steps and distinct personas. We introduce SYNC (SensitivitY-guided iNference Control), a novel training-free inference control framework compatible with pre-existing steering vectors. SYNC dynamically estimates each layer's sensitivity to every persona via cosine similarity trajectories, assigning each persona to its optimal injection layer at every decoding step. Across two models, two persona profiles and steering vectors from both unsupervised and supervised extraction, SYNC leads on 21 of 28 persona configurations, significantly on 15. Code is available at \href{this https URL}{this https URL}. 

---
# Scalaris: A Scalable Enterprise Memory Architecture for Conversational AI Assistants 

**Authors**: Devarsh Gandhi, Sarthak Pawar, Hemprasad Badgujar, Vaibhav Rathore, Siddharth Sureka, Neha Bhargava  

**Link**: [PDF](https://arxiv.org/pdf/2611.02612)  

**Abstract**: Enterprise conversational systems require memory to personalize interactions, retain information across sessions, and maintain conversational coherence. At enterprise scale, however, memory must not only retrieve relevant information accurately but also provide it efficiently to the answer-generation model. Existing memory systems face a trade-off between retrieval accuracy and context efficiency. Retrieving more context can improve recall but increases inference cost and latency. We present Scalaris, a scalable enterprise memory architecture that addresses this trade-off through a unified long-term and short-term memory design. We propose that user interactions and assistant responses possess distinct characteristics and roles in a conversation: user interactions capture evolving knowledge about the user and provide value across sessions, whereas assistant responses must be stored efficiently so that only information relevant to the current query is retrieved. We therefore treat them separately and support each with specialized representations and ingestion/retrieval pipelines. Scalaris reaches 92.2 % accuracy on LongMemEval-S and 90.77 % on LoCoMo using Qwen3-32B, state-of-the-art results on both benchmarks, while supplying the answer-generation model with approximately 10 times fewer retrieved context tokens than Hindsight, a leading memory system for conversational AI. When integrated with our proprietary closed-source AI assistants, Scalaris improved overall response quality by 36 %. Together, these results establish Scalaris as a cost-effective framework for building scalable, personalized conversational AI systems. 

---
# Whitened Advantage Summation Policy Optimization: Correlation-Aware Aggregation for Multi-Reward Reinforcement Learning 

**Authors**: Olaitan Olaleye, Jayson Meribe, Temidayo Olaleye  

**Link**: [PDF](https://arxiv.org/pdf/2611.02609)  

**Abstract**: Reinforcement learning for language models often combines several rewards by normalizing each one and adding them with fixed weights. When the rewards are correlated, the variation they share is counted once for every reward that carries it. In terms of the ZCA-decorrelated components of the reward signal, the weighted sum applies weights $\Sigma^{1/2} w$ rather than the chosen weights $w$, where $\Sigma$ is the reward correlation matrix. We study WAS-PO, which whitens the normalized advantages with $\Sigma^{-1/2}$ before adding them, so that the chosen weights apply to those components. We derive the scalarized objective whose gradient it follows when normalization statistics are held fixed, show that its weights on the raw rewards can be negative, give an exact two-reward condition under which the lighter-weighted reward is reversed, and give a loose bound on the error from estimating $\Sigma$ during training. On tool calling with Qwen2.5-1.5B-Instruct, WAS-PO reached a higher mean training reward than GDPO in all seven seeds with three rewards (3.8% higher) and all three seeds with four (3.9% higher). WAS-PO also replaces GDPO's per-prompt normalization with batch-level normalization, and we did not separate the two changes with three or four rewards. The training-reward gains did not carry over to held-out accuracy, which was slightly lower than GDPO's with three rewards (not significant) and 0.03 lower in absolute terms with four (two seeds evaluated). With two rewards, WAS-PO trailed GDPO in training reward, and a three-seed control weakly suggests that the gap comes from batch-level normalization rather than whitening. In a math setting where the two rewards were nearly identical by construction, WAS-PO drove down the lighter-weighted reward, as the sign condition predicts. 

---
# StateFence: Task-State Protection Against Evidence-Chain Contamination in Blue-Team LLM Agents 

**Authors**: Haowen Dai, Alejandro Guerra-Manzanares  

**Link**: [PDF](https://arxiv.org/pdf/2611.02605)  

**Abstract**: Indirect prompt injection creates a distinctive risk for blue-team Large Language Model (LLM) agents in Security Operations Centers (SOCs). Malicious instructions can be embedded in telemetry, tickets, retrieval results, or memory and then propagate through alert triage, incident summarization, response recommendation, and Indicator of Compromise (IOC) extraction as locally plausible intermediate decisions. We characterize this failure mode as evidence-chain contamination of protected task state. The main challenge is task dependent: a suspicious string may be malicious control text for summarization but essential evidence for IOC extraction. To address this challenge, we propose StateFence, a task-aware evaluation and defense framework that treats severity labels, incident narratives, response policies, and IOC sets as protected states. StateFence maps indirect prompt-injection attacks into explicit, stealth, multi-hop, and persistent SOC evidence attacks, records observable propagation traces, and routes defenses according to the attacked state. Across controlled SOC cases and public-SOC validation cases, StateFence lowers attack success and reduces failures that appear only after evidence is poisoned, while preserving task utility and avoiding the indicator loss caused by task-mismatched filtering. Our findings indicate that robust blue-team LLM agents require mechanisms that preserve task-critical state under untrusted evidence. 

---
# AVID: A 5T fMRI Dataset for Benchmarking Auditory-induced Visual Mental Imagery Decoding 

**Authors**: Shiqi Shen, Shurui Li, Xilin Zhang, Yuanning Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.02593)  

**Abstract**: Visual mental imagery (VMI) decoding uses brain activity to reconstruct internally generated visual scenes, providing a potential pathway for externalizing internal experiences and developing neuroprosthetic communication interfaces. However, progress is limited by the lack of dedicated, training-scale VMI datasets. Current VMI paradigms often rely on visual prompts or short-term memory images, introducing visual and mnemonic confounds that make it difficult to disentangle neural activity associated with endogenous image construction from signals driven by prior or concurrent visual input. Furthermore, existing VMI datasets offer limited descriptive control over individual visual attributes, making it difficult to systematically assess fine grained visual details. To address these gaps, we introduce AVID, the first dedicated training-scale 5.0 Tesla functional magnetic resonance imaging (fMRI) resource for auditory-induced VMI decoding. AVID comprises approximately 67 hours of dense recordings from nine participants. The dataset is structured into two complementary splits: a scene-level complex split using naturalistic Mandarin descriptions and a controlled simple split with explicitly specified visual attributes. We further formalize a standardized evaluation protocol that decouples source-image consistency from semantic alignment, allowing for a more nuanced assessment of imagery reconstruction. Baseline qualitative and quantitative results show that current reconstruction pipelines remain limited on AVID, although the most successful reconstructions preserve recognizable scene level semantics from spoken descriptions. By combining dedicated training-scale neural recordings and standardized evaluation resources, AVID provides both a data foundation for imagery-specific model development and a benchmark for evaluating VMI decoding with substantially reduced sensory contamination. 

---
# They Know Where It Is, But Not What It Says: A Bilingual Hallucination Benchmark of Hadith Recall in Open LLMs 

**Authors**: Dayam Nadeem, Safdar Mustafa, Ameen Ur Rehman, Shahab Saquib Sohail, Mohammad Nadeem, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2611.02587)  

**Abstract**: Large language models are increasingly consulted for religious knowledge, including the recall of specific Hadith (recorded sayings and actions of the Prophet Muhammad, peace be upon him). Unlike open-ended generation, Hadith recall is a high-stakes, verifiable task: a reference (collection, chapter, and Hadith number) has exactly one correct text, and any deviation from it constitutes a fabrication that can be mistaken for authentic scripture. This study introduces a bilingual (English/Arabic) benchmark that probes five open-weight, locally-deployable language models (Mistral-7B, Gemma3-4B, Jais-7B, ALLaM-7B, and Phi4-mini-3.8B) across all six canonical Sunni Hadith collections (Sahih al-Bukhari, Sahih Muslim, Sunan Abi Da'ud, Jami' al-Tirmidhi, Sunan an-Nasa'i, Sunan Ibn Majah), sampling 500 Hadith per collection (3,000 per model per language, 30,000 responses total). Each model is given only the bibliographic reference and asked to reproduce the exact Hadith text, with no retrieval augmentation, and is scored on lexical fidelity (BLEU) and semantic fidelity (multilingual cosine similarity), with refusals flagged separately. Across all five models and both languages, 0% of 30,000 responses achieve both high lexical and high semantic fidelity, i.e., an exact, verifiable match to the source text. Arabic responses fare uniformly worse than English. These findings suggest that current open-weight LLMs, used without retrieval grounding, are unreliable and potentially dangerous sources for exact religious-text recall, and that deployment for Islamic scholarship should mandate retrieval-augmented generation over a verified corpus rather than parametric memory. The complete code, prompts, evaluation pipeline, and scored outputs are publicly available at this https URL. 

---
# Sequential Recovery Updates for Learned Controllers under Successful-Trajectory Preservation 

**Authors**: Zongtan Li, Tianming Liu, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.02555)  

**Abstract**: A competent learned controller already recovers from many disturbances. Improving it means learning a correction, and learning when not to apply it. Freezing a skill's parameters does not settle the second half. In a controller that routes among frozen skills, a new entry rule can redirect the states an old skill succeeded on. We therefore constrain entry instead of parameters. Each update searches for a feedback skill at a measured failure. It fits an entry rule that reads observations and rejects every state on an accumulated successful trajectory. The skill is admitted only if the composed controller reproduces the new recovery on its own and replays every protected trajectory exactly. A finite-trajectory condition states when this preserves the previous controller. One mechanism runs on four streams: two motions on each of two humanoid systems. Two updates per stream add 62 recoveries over 192 withheld perturbations, and lose none of the base controller's successes. We then price that guarantee. Removing the constraint recovers 33 more conditions but stops eight protected ones from completing at all, and excluding only what admission proves necessary recovers none of that cost. 

---
# Scientific Liquidity Agents: An Agent-Based Model of Knowledge Arbitrage in Scientific Communities 

**Authors**: Koutian Wu, You Liu, Yue Liu, Zhiping Li, Ergan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02539)  

**Abstract**: Scientific knowledge gains new uses when ideas, methods, and results move between research contexts. Understanding this process requires a model of both specialized knowledge production and the intermediary activities that connect it. We introduce Scientific Liquidity Agents (SLA), an agent-based model of knowledge arbitrage in scientific communities. SLA represents domain specialists and epistemic arbitrageurs within a shared environment of research problems, researcher interactions, and scientific artifacts. Individual decisions connect to collective state through a belief-desire-intention architecture, a prerequisite-based problem graph, a dynamic interaction network, and a persistent publication registry. The framework makes researcher roles and knowledge-transfer frictions explicit, providing a process-oriented account of scientific production and exchange. Two exploratory simulations with 10 and 18 agents produce 68 and 274 registered papers. In the smaller run, arbitrageurs take part in 16 of the 26 contact pairs that join this community into one connected component, while topical similarity alone splits it into seven. The same run yields three similarity edges, 26 contact pairs, and two coauthorship pairs, so measured community structure depends on what counts as a link. Paper production is abundant, but no recorded replication succeeds, which places the bottleneck in verification rather than production. Solved-problem sets correspond to the prerequisite reachability of the disciplines represented in each community. SLA makes knowledge liquidity, the movement and reuse of artifacts across domains, a quantity that agentic science can simulate and measure. 

---
# Latent Spectral Prompting for Multimodal Hyperspectral Object Tracking 

**Authors**: Wangshu Cai, Jun Zhou, Lei Wang, Mohammad Aminul Islam, Xu Han, Kuldip Paliwal  

**Link**: [PDF](https://arxiv.org/pdf/2611.02512)  

**Abstract**: Hyperspectral videos provide rich spectral signatures for robust tracking under challenging conditions such as camouflage and low illumination. However, limited hyperspectral annotations and the large RGB-hyperspectral domain gap hinder the direct use of modern temporally pretrained RGB trackers. We propose SProTrack, a multimodal framework that transfers temporal knowledge from pretrained RGB trackers to the hyperspectral domain through latent spectral prompting. The key idea is to decouple temporal modeling from spectral adaptation, allowing temporal dynamics to be inherited from RGB foundation models while hyperspectral information is learned from unlabeled data in a self-supervised manner. Unlike band-selection methods that discard most spectral information or prompt-based approaches that rely on local-search crops, SProTrack learns a global, scene-level spectral representation from the full hyperspectral frame using an autoencoder. This latent representation is used to guide spectral prompt extraction while a pretrained RGB tracker preserves strong temporal modeling on RGB-compatible inputs. By encoding spectral-spatial structure into compact latent prompts, SProTrack bridges the RGB-hyperspectral modality gap without extensive retraining or dense supervision. Experiments on HOT2020 and HOT2023 demonstrate that SProTrack achieves the best overall performance, with strong results across all spectral domains. 

---
# Neuro Inspired Robotics: What Robot Learning Can Borrow from the Sensorimotor Brain 

**Authors**: Ruihan Zhang, Davy Deng  

**Link**: [PDF](https://arxiv.org/pdf/2611.02510)  

**Abstract**: Robot policies still fail at everyday manipulation: an unfamiliar task, an object never held before, a scene that changes mid-motion, a skill that should improve with use. They fail despite pretraining on over a million trajectories, pooled from dozens of robots at great expense, while a child acquires all of these abilities within a few years of learning from a single body. This survey argues that the gap is one of design, and proposes designs for the next generation of robot models, drawn from how the nervous system solves the same problems. From the body we take compliance: the policy commands stiffness alongside position, so the controller yields where the task allows and presses where it must, which greatly lowers the demand for precise position output. From the visual system, where a fast pathway for action runs in parallel with the slow pathway for recognition, we take a dedicated fast vision system for motor behavior: it precomputes the quantities the policy needs and drives a reactive policy at camera rate. From the cerebellum we take a predictor beside the policy that forecasts the consequence of each command and recalibrates from its own error. From motor cortex we take a single model that composes action units from primitives, verifies each at a sensory boundary, practices the weakest unit with a hypothesis in mind, and reasons each attempt closer to the goal. Proven primitives are then consolidated from code into weights, so they run fluently and respond quickly the next time. Together these describe a robot that acts on its own initiative, the basis of deployment and recursive self-improvement, and we offer each proposal as a hypothesis to be tested. 

---
# RoboColosseum: Evaluating Physical Intelligence at Scale In-The-Wild 

**Authors**: Tan Sang Nguyen, Yi Ru Wang, Shailesh Shailesh, Gang Yang, Jiamo Hu, Jianuo Jiang, Haoquan Fang, Zhongzheng Ren, Zhongyi Luo, Ian Chong, Aaron Tung, Santiago Pravisani, Minghui Lin, Yao Mo, Ishneet Singh, Benjamin Tee, Harold Soh, Siddhartha Srinivasa, Dieter Fox, David Hsu, Haoming Song, Maxime Ellerbach, Thomas Wolf, Pepijn Kooijmans, Ruijie He, Michael Cho, Jiafei Duan  

**Link**: [PDF](https://arxiv.org/pdf/2611.02497)  

**Abstract**: General-purpose robot policies must work across robots, tasks, and environments, yet the variation that makes physical trials informative also makes them difficult to reproduce. We introduce RoboColosseum, a distributed evaluation framework that combines unified policy serving and standardized protocols across a network of locally operated robots in diverse environments. Our initial study evaluates five vision-language-action (VLA) policies and three frontier AI agents across four embodiments (SO-101, Franka, Bimanual-YAM, and G1 Humanoid). The Open Track compares models without task-specific adaptation in blinded, pairwise trials under matched local conditions. The Fine-tune Track measures how well VLA policies adapt to 20 tasks across all four embodiments when each policy receives the same 100 demonstrations per task. Across 5,699 pairwise Open Track comparisons at 124 sites, frontier agents rival specialized VLA policies, while all systems struggle to complete open-world tasks reliably. After fine-tuning, GR00T N1.7 leads on SO-101 and pi0.5 leads on Franka and Bimanual-YAM. Policy rankings show weak agreement across embodiment-specific task suites (Kendall W = 0.24). Open Track rankings poorly predict relative performance after fine-tuning, with substantial rank reversals on all three embodiments shared between tracks. Measuring partial progress reveals differences that task success alone misses. On the G1 Humanoid, no policy completes any of the five tasks, yet average progress ranges from 13.3% to 23.2% across policies. Open Track rankings stabilize as trials accumulate, and active matchmaking achieves comparable ranking agreement with substantially fewer physical trials than uniform sampling. These results show how distributed physical evaluation can reveal where robot policies generalize, how well they adapt to new tasks, and how far they progress on tasks they fail to complete. 

---
# Cross-Domain Joint-Embedding Pretraining For Heterogeneous Industrial Time Series 

**Authors**: Jonathan Adam Rico, Nagarajan Raghavan, Senthilnath Jayavelu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02489)  

**Abstract**: Industrial predictive maintenance and manufacturing tasks frequently contain abundant sensor measurements, but with limited labeled data. We investigate cross-domain self-supervised pretraining using joint-embedding predictive architecture for heterogeneous industrial time series (JEPA-HITS), which learns one encoder from various domains such as battery, semiconductor chemical mechanical planarization (CMP), computer numerical control (CNC) machining, and bearing while preserving dataset-specific temporal scales. We evaluate representation reuse on regression and classification problems under reduced-label downstream training, and leave-one-domain-out (LODO) pretraining. Compared with the same architecture initialized randomly, LODO pretraining reduces RMSE from 10.72 to 7.95 on battery remaining useful life (RUL), 4.34 to 3.89 on CMP, and 13.53 to 11.72 on tool wear regression, while improving bearing fault diagnosis macro-F1 from 0.547 to 0.737. Overall results suggest that unlabeled time series measurements from heterogeneous industrial processes can provide reusable representations for downstream learning with limited labeled data. 

---
# RecRSI: Mechanism-Guided Discovery of Recommendation Algorithms 

**Authors**: Tingran Gan, Zhouziyue Lu, Dian Jin, Tao Li, Yang Gao, Ming Liu, Weiwen Liu, Jianghao Lin, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02487)  

**Abstract**: Improving recommendation algorithms for the Web requires coordinating interacting design choices and learning from experiments. We present RecRSI, an LLM-based research framework that uses mechanism hypotheses to organize algorithm discovery. Each hypothesis links a proposed mechanism change to a reference design, a rationale, and an expected observation. MechSpace derives implementable primitives and composition rules from the task source; Research Line uses this space and prior evidence to develop and test hypotheses; Mechanism Feedback carries outcomes and their design context into subsequent research. This organization makes each experiment inform which mechanisms to develop, revise, or combine next. Across four initial recommenders and four datasets, RecRSI yields average relative improvements of 8.48% in NDCG@5 and 6.88% in HR@5 over the initial algorithms, and 2.60% and 2.20%, respectively, over the strongest competing result for each setting and metric. Component and process studies link these gains to coordinated mechanism choices and evidence-guided follow-up. Our code is publicly available at this https URL. 

---
# Chain on Demand: Deferred Evidence Organization for Long-Term Conversational Memory 

**Authors**: Di Lu, Zechen Li, Hao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2611.02480)  

**Abstract**: Long-term conversational memory faces a timing mismatch: write-time summaries and graphs organize history before a future query is known, while independent retrieval can allocate a limited evidence budget to redundant or fragmented turns. We introduce Chain on Demand (CoD), a query-time evidence organization framework over an unchanged turn-level memory. Given a broad candidate pool, CoD constructs a temporary evidence chain by allocating a fixed anchor budget according to query relevance, lexical diversity relative to selected anchors, and newly covered query terms, followed by deterministic restoration of each selected anchor's same-session neighborhood. The resulting chain is used only for the current query, while the underlying conversational memory remains unchanged. In a paired comparison over all 1,986 LoCoMo questions, CoD achieves 82.2% accuracy on 1,540 non-adversarial questions and 76.3% on all 1,986 questions. On LongMemEval-S, one fixed configuration that indexes both user and assistant turns obtains 86.6% across all 500 questions. These results show that long-term memory can preserve the original conversation while deferring evidence organization until the information need is known. 

---
# Multi-Teacher Reasoning Distillation for LLM Recommendation with Self-Exploration 

**Authors**: Liangbo Ning, Siyuan Liang, Yi Zhou, Yifeng Zhang, Qing Li, Wenqi Fan, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2611.02457)  

**Abstract**: Reasoning-augmented recommender systems (RecSys) enhance user preference modeling by generating reasoning trajectories over historical interactions before making recommendations. Knowledge distillation provides a practical paradigm for transferring such reasoning capabilities from powerful large language models to lightweight RecSys. However, existing methods typically rely on a single fixed teacher, remain constrained by the teacher knowledge ceiling, and indiscriminately distill teacher-generated trajectories without assessing their actual utility for downstream recommendation. To address these limitations, we propose MUSE (Multi-Teacher Utility-guided Selection with Self-Exploration), an adaptive reasoning distillation framework that jointly improves reasoning generation and reasoning utilization. MUSE first constructs a diverse reasoning pool using multiple teacher LLMs and progressively augments it via student self-exploration, enabling the student to leverage the strengths of different teachers and gradually explore reasoning strategies beyond teacher imitation. We then develop Rank-Gain Utility Assessment (RGUA), which evaluates each candidate according to the normalized full-catalog rank improvement of the ground-truth item and identifies the most and least recommendation-beneficial trajectories. Based on these utility-aware trajectory pairs, we introduce dual-level contrastive learning objectives that guide the student generator to prefer beneficial reasoning while training the recommendation encoder to more effectively exploit recommendation-relevant preference evidence. Extensive experiments on three real-world datasets demonstrate the effectiveness and robustness of the proposed framework. 

---
# Turbo-GMM: FlashAttention-Inspired GPU-Accelerated Gaussian Mixture Model 

**Authors**: Momin Abbas, Sadia Asif, Asad Ul Haq, Kaoutar El Maghraoui  

**Link**: [PDF](https://arxiv.org/pdf/2611.02449)  

**Abstract**: Expectation-maximization (EM) for Gaussian mixture models (GMMs) requires scoring $N$ data points against $K$ mixture components. Prior fused GPU implementations, notably Flash-GMM, eliminate materialization of the $N \times K$ responsibility matrix but traverse the component dimension twice to compute normalization terms and accumulate sufficient statistics. We introduce Turbo-GMM, a single-sweep fused EM kernel that completes responsibility normalization and sufficient-statistic accumulation within one traversal of the component dimension. By retaining the complete component axis on chip, Turbo-GMM avoids the second score computation required by the released two-sweep Flash-GMM implementation. Its applicability is limited by register pressure, occupancy, and the component and feature dimensions, making it particularly suitable for large datasets with relatively small numbers of components. For larger configurations, Turbo-GMM employs a two-pass fallback following the established fused design. We establish algebraic equivalence to standard EM in real arithmetic and evaluate sixteen configurations on an NVIDIA A100. Under the reported benchmarking protocol, Turbo-GMM achieves a median $2.15\times$ speedup over the evaluated Flash-GMM implementation and up to $5.3\times$ and $9.9\times$ over PyTorch and \texttt{scikit-learn}, respectively. The results demonstrate the potential of single-sweep component traversal while identifying on-chip resource constraints as the principal limitation of this approach. 

---
# Harvesting SWE Benchmarks from Wild Vibe Coding Sessions 

**Authors**: Ao Lu, Yuanhao Zeng, Shaocheng Lan, Sijie Xu, Jiacheng Bao, Kuang Yan, Heng Lin, Kan Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.02440)  

**Abstract**: Development teams need reproducible tasks from their own repositories and workflows to evaluate coding agents. Conversational developer-agent collaboration ("vibe coding") produces abundant coding sessions and repository checkpoints, but requirements evolve across turns, unrelated conversations co-occur with commits, and engineering efforts can span multiple sessions. We present SessionFarmer, an automated framework that turns these records into self-contained, executable software engineering tasks. It analyzes dialogue and commit history to filter unrelated sessions and group multi-turn, potentially multi-session interactions into work episodes centered on one engineering goal. For each episode, it reconstructs a hermetic execution container, synthesizes a task statement from the full context, and authors repository-grounded engineering-quality rubrics and unit tests. Closed-loop test refinement uses requirement-breaking mutants and a blind solver to reduce false rejections of valid alternatives and false acceptances of incomplete solutions. We construct CropChat, an open benchmark of 242 tasks across 61 public repositories and 14 programming languages, and CropEnterprise from proprietary enterprise logs. In a CropEnterprise audit, the original developers endorse the vast majority of reconstructed episode boundaries, task statements, and quality rubrics. Full-episode context covers substantially more user requirements than single-turn or single-session inputs. Test refinement cuts false-rejection rates and raises balanced accuracy against reference patch labels by 7-9 percentage points across solvers. Across 12 model-harness configurations on CropChat, the strongest setup achieves 39.3% Pass@1, while harness choice shifts Pass@1 by up to 8.4 percentage points for the same model. 

---
# Networked Synthetic Social Engineering on the Web 

**Authors**: Bryan C. Boots  

**Link**: [PDF](https://arxiv.org/pdf/2611.02407)  

**Abstract**: Several genuine AI services may repeat one unsupported source, and correcting that source may leave later reports unchanged. We examine these networked consequences of synthetic social engineering (SSE), deliberately misleading influence intended to redirect AI recipients from authorized objectives. Our framework connects social influence, information aggregation, and misinformation correction research to observable evidence and permission boundaries. It separates originating attempts from legitimate intermediaries' work and distinguishes identity, evidentiary support, and authorization. We test four relationships using Qwen3-4B and two briefing tasks across 1,080 research units in our Agent Service Exchange and Evidence Commons (ASE-EC). Multiple dependent publishers produced no supported increase in unsupported verification (78.3% versus 75.0%); publication availability left downstream violation reach at 40.0% in both arms. These comparisons direct attention to weaknesses already present in the reference workflows. High reference error rates and retained private handoffs preclude claims of immunity to social pressure. Without scope revalidation, corrections identifying dependent reports reduced residual invalid decisions from 14.4% to 8.1% at equal notification coverage and increased successful report repairs from 16/63 to 36/63. Scope revalidation added no demonstrated protection, and clearing dialogue offered no established recovery advantage. Unapproved requests persisted, all blocked by an external gateway. For designers, the findings caution against treating publisher count as independent evidence, notification as repair, or cleared histories as recovery. They support targeting corrections at reused reports, assessing factual and permission recovery separately, and enforcing authorization at execution. 

---
# Solver-Like Behavior in Large Language Models as Formalizers 

**Authors**: Rikhil Amonkar, Stuti Mohan, Kevin Nguyen, Ceyhun Efe Kayan, Ronan Le Bras, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02395)  

**Abstract**: In the LLM-as-formalizer paradigm, a large language model translates a problem into a formal program and leaves the search to a solver, gaining the solver's guarantees. Formalizers are now often reasoning models that write a long reasoning trace before the program, yet they are judged only by the program's output. We find that the trace routinely does the solver's job, assigning, testing and ruling out values until it often reaches the full answer; we call this solver-like behavior (SLB). We define an ontology of SLB over the trace, the program and its output, and use LLM judges, checked against human labels, to label 2,000 generations from three reasoning models on four constraint satisfaction datasets, each formalized in a declarative and an imperative formal language. SLB appears in 85% of traces and takes 24% of reasoning tokens, yet the program usually recomputes the answer, so the effort is redundant. The 9% of programs that inherit the trace's answer also inherit its errors, and pass output-based evaluation whenever that answer is correct. The amount of SLB is not associated with accuracy, and SLB is more common in imperative Python than in declarative Z3 or PDDL. Cautionary prompts reduce it by only a quarter, but hiding the problem's values behind placeholders, which the program receives only at run time, removes 97% of it without loss of accuracy. Formalizers should therefore be evaluated by how their programs are reached, not only by what they output. 

---
# Late Autumn: LLM Agents Specialize a General Multi-Robot Planner for One Warehouse 

**Authors**: Yimin Tang, Han Zhang, Shao-Hung Chan, Erdem Bıyık, Sven Koenig, Jingkai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.02391)  

**Abstract**: No algorithm is best on all problems, but a real warehouse is a narrow problem: its layout and processes rarely change. Operators still run one general multi-robot planner at many sites, because experts cannot tune it for each one. In this work, which we call Late Autumn, we explore whether LLM coding agents running autonomous research can specialize a general planner for one site, trading generality for throughput, much as post-training adapts a pretrained LLM to one task. We study six campaigns on a lifelong planner that combines task assignment with multi-agent path finding, on a floor modeled on an existing automated warehouse. We find that (1) the agents raise throughput by 22--60%, and the gains hold on unseen seeds; (2) they rediscover known MAPF mechanisms, such as corridor-exit priorities, local guidance, and rollouts, and independent campaigns find the same bottleneck; (3) the gains largely carry over to another layout, a larger fleet, and other service times, shrink with a smaller fleet, and mostly vanish under an outbound-heavy demand mix; and (4) each final patch (0.7k--2.4k lines) is a chain of scored commits with a median step of 27 lines, so experts can review it, and review finds problems that the score misses; some patches multiply planner compute by over 100x. LLM agents can therefore turn each site's bias into extra throughput at little engineering cost, with reviewable patches; we expect experts to shift toward designing insightful metrics and fast simulators that mirror the real problem. 

---
# Malware Clustering in the Wild: Taxonomy, Challenges, and Evaluation Metrics 

**Authors**: Raguvir S, Bojing Li, Prajna Bhandary, Charles Nicholas  

**Link**: [PDF](https://arxiv.org/pdf/2611.02380)  

**Abstract**: As malware continues to grow in volume, diversity, and sophistication, clustering has emerged as a critical technique for analyzing large malware corpora. Clustering enables the discovery of structural relationships, behavioral patterns, and evolutionary trends, and is widely used to group unknown samples, infer malware family boundaries, and identify novel variants. However, existing research remains fragmented, employing a wide range of clustering techniques with no unified perspective that systematically organizes prior work or clearly identifies gaps in the literature. This survey addresses this gap by organizing previous malware clustering research into seven categories: centroid-based, hierarchical-based, density-based, distribution-based, mixed- type, fuzzy, and AI-driven approaches. In addition, this survey summarizes publicly available malware datasets commonly used in clustering research and examines the feature representations derived from these datasets. Our goal is to clarify the current landscape and outline promising avenues for future research on malware clustering, including how AI-driven methods may shape large-scale and reproducible analysis. 

---
# Beyond Accuracy: How Quantization Affects Coding Agents 

**Authors**: Areej Mehboob, Muhammad Ali Shafique, Muhammad Usman Qadeer, Hamza Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2611.02368)  

**Abstract**: Agents are increasingly used for coding tasks, but the inference cost of large language models (LLMs) limits where they can run. Quantization reduces model size, but its effects on coding agents that execute and revise programs remain underexplored. We compare BF16 with 4-bit AWQ and GPTQ for Qwen2.5-Coder-Instruct (7B, 14B, and 32B) and Devstral-Small-2507 (24B) on OJBench coding problems across three seeds. We detect no statistically significant accuracy loss for any of the coding agents. However, final accuracy can conceal substantial behavioral changes: AWQ reduces Qwen-7B one-shot solves from 17 to 0 while increasing loop-based solves from 10 to 25. All Qwen-32B successes involve tool use, whereas Devstral rarely makes a valid executor call. Failure profiles also differ: Qwen-32B produces far more compilation errors than the smaller Qwen models, rising slightly under quantization, while Devstral's failures shift significantly from runtime errors toward wrong answers and timeouts. These findings indicate that quantization robustness in coding agents should be assessed through solution trajectories, tool use, and failure modes, in addition to final accuracy. 

---
# Exploiting Textual Semantics for Robust Cross-View Object Correspondence 

**Authors**: Bing Fan, Yunhe Feng, Yan Huang, Heng Fan  

**Link**: [PDF](https://arxiv.org/pdf/2611.02356)  

**Abstract**: Cross-view object correspondence (CVOC) aims to establish associations of the same target between query and search videos captured from different viewpoints. Existing methods often rely on visual cues for CVOC. Despite recent progress, these methods struggle in complicated scenarios with substantial viewpoint changes and appearance variations, due to lacking sufficient target information. Addressing this, we introduce TeSCo, a framework that exploits rich Textual Semantics of an object, in addition to its visual cues, for cross-view object Correspondence. Our key insight is, textual description of a target, by capturing diverse attributes such as category, texture, and color, provides viewpoint- and appearance-invariant semantics, which are complementary to visual cues and can hence enhance correspondence robustness in complicated scenes. Inspired by this, TeSCo first generates a textual expression of the target from the given query view using a vision-language model, and then extracts textual feature to enhance visual features for localization in the search view. To realize this, we introduce a text-conditioned cross-modal fusion (TCF) module that incorporates the textual feature into visual features with text-guided modulation, producing more robust multimodal target representation for object correspondence. Since not all textual cues are equally compatible with the target in current search-view frame, we propose an iterative textual feature refinement (ITFR) module that progressively adjusts textual feature using search-view information before the TCF module, enabling search-view-aware textual semantics to be fused into visual features and leading to better performance. In extensive experiments on Ego-Exo4D and HANDAL-X, TeSCo demonstrates state-of-the-art results and largely outperforms other models, validating its efficacy. Our code is released at this https URL. 

---
# AgiWorld: An Agentic Video World Model for Building Persistent Worlds 

**Authors**: Shaoheng Fang, Yiqun Mei, Hanwen Jiang, Xin Sun, Haitian Zheng, Hao Tan, Kalyan Sunkavalli, Georgios Pavlakos, Qixing Huang, Yiwei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02341)  

**Abstract**: Recent interactive video world models generate visually compelling videos from a single image, following user-provided camera motion and text prompts. However, such low-level control demands substantial user effort to realize a high-level intent, and the generated world struggles to remain persistent as the exploration proceeds. In this work, we present AgiWorld, an agentic video world model that explores and builds a persistent world from a single image and a free-form instruction. For high-level planning, our agent AgiWorld-Agent designs the world and the journey through it in a world state and directs the exploration. For low-level realization, our video world model AgiWorld-WM generates the video chunk by chunk and stores it with its geometry in a spatial memory, from which the agent updates its world state. This closed loop keeps the world faithful to the user's intent and keeps the agent's world state consistent with the generated content throughout the exploration. We co-design the two levels around the shared spatial memory and an interface of metric camera trajectories and structured prompts. In AgiWorld-WM, to handle occlusion and accumulated geometric errors in the spatial memory over long rollouts, we propose occlusion-aware history selection, which retrieves past views by surface visibility, and reliability-based memory maintenance, which prefers reliable geometry and earlier observations. AgiWorld-WM achieves state-of-the-art overall performance on world revisiting and exploration benchmarks, and qualitative results and ablations show that AgiWorld builds persistent worlds from free-form instructions over minute-long journeys. Persistence thus arises not from the video model alone but from its closed loop with the agent. 

---
# MemShift: Separating Detection from Repair in LLM-Derived Personal Memory 

**Authors**: Zichuan Xu, Haochen Zhang, Yuyang Dong, Makoto Onizuka, Chuan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.02338)  

**Abstract**: AI assistants store values that a language model derived from a person's records under the person's own definitions, such as their important contacts. Keeping such a value correct takes two steps: detection, noticing that an input changed, and repair, producing the correct new value. Benchmarks that score only the final answer cannot tell which step failed, and they state every change as new text. Seeing this limitation, we present MemShift, a benchmark of five shifts: source revision, deletion, redefinition, writer switch, and export. The first three fork a synthetic world into branches that differ only in the shift. Gold values and dependencies are fixed before any text, and a deterministic scorer replaces an LLM judge. For a memory that reports its invalidations, MemShift shows for each changed item whether the memory noticed the change and whether its repair was right. Moreover, we build a reference memory with known detection and repair to validate the measure. By evaluating four open-weight LLMs, we find that detection helps only as much as repair allows: the LLMs re-derive few values computed by the person's definitions correctly, and programs that the LLM writes over typed records close most of the gap to exact recomputation. Code and data are available at this https URL. 

---
# Spatial Transfusion: From Structured Spatial Scene Understanding to Generation and Editing 

**Authors**: Jaeyeon Kim, Sebastian Braun, Hannes Gamper  

**Link**: [PDF](https://arxiv.org/pdf/2611.02330)  

**Abstract**: Spatial audio conveys not only what sounds are present, but where they originate and how they interact with the environment. Recent audio-language models increasingly understand and reason about such scenes, yet spatial audio understanding, generation, and editing remain largely separate. We present Spatial Transfusion, a unified first-order ambisonics (FOA) framework that extends a pretrained spatial audio understanding model to generation and editing through flow matching over VAE latents. We formulate editing as a structured scene transformation over a taxonomy of atomic edits. Keeping the pretrained backbone frozen, we introduce modality-based LoRA routing and a curriculum that progressively adds generation, editing, and natural-language interaction. Experiments show that Spatial Transfusion largely preserves spatial understanding while outperforming full fine-tuning in generation and editing. The resulting model supports controllable FOA generation and editing through both structured and natural-language interfaces. Audio samples are available at this https URL 

---
# FactorSpec: Serving Exact Relational Constraints as Speculative Drafts 

**Authors**: Jianchang Su, Dawei Xiang, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.02297)  

**Abstract**: Speculative decoding speeds up LLM serving by drafting tokens that the target model verifies in one pass, and its gain grows with the drafts the target accepts. Structured outputs should be an ideal case, because much of a record follows from relations among its fields: an itinerary's days follow from its cities' stays and flights. Serving engines, however, draft such text by proposing it from the prompt or a learned head, and their grammar masks enforce only syntax, so the model must satisfy the relations, which it violates: on NATURAL PLAN's trip problems, Qwen3-8B writes a valid itinerary for 10%, and retries that name the violated relation raise this to 14%. We present FactorSpec, which serves exact relational constraints as speculative drafts. FactorSpec compiles a JSON Schema and its relations into a factor graph with an exact feasibility check, drafts every token the constraint determines as certain, verifies each request at a width chosen for it, and computes the masks in worker processes outside the scheduler while the GPU runs. In SGLang, its drafts yield up to 10.5 tokens per verify step, up to 1.9x those of the fastest n-gram drafter, and it decodes up to 1.43x faster than speculators tuned per load, at a grammar mask's cost; it also leads in vLLM. Because every record is also valid, it serves up to 13.9x the valid records per GPU-second of syntax-only baselines. 

---
# Hinted Scaling Laws 

**Authors**: Suze van Adrichem, Luke Bailey, Rohith Kuditipudi, Christine Ye, Neil Band, Ahmed Ahmed, John Duchi, Percy Liang, Tatsunori Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2611.02270)  

**Abstract**: Scaling laws predict the outcome of large, expensive training runs by extrapolating trends from smaller, cheaper runs. On downstream accuracy evaluations, these predictions fail when the weaker models produced by the cheaper runs score near zero accuracy, because these scores give no trend to extrapolate to larger scales. We introduce hinted scaling laws, which use privileged information to turn a benchmark into a continuum of easier versions so that even the small models score above zero. We find that accuracy rises smoothly and predictably with the hint level, enabling new types of scaling laws that operate below the noise threshold. On the Delphi pretraining ladder, a hinted scaling law fit on models trained with at most $10^{20}$ FLOPs forecasts the GSM8K accuracy of a $10^{23}$ FLOP model (25.9%) to within 3 percentage points; a standard scaling law under-forecasts it by 13 percentage points when fit on the same models and by 12 even with models up to $10^{22}$ FLOPs in the fit. Across both this pretraining ladder and a collection of post-trained model families, hinted scaling laws reduce the median forecasting error on GSM8K, AIME and HLE from 21 percentage points to 4.3. Hinting also reduces the number of test examples needed to tell whether one model is better than another by widening the accuracy gaps between models while approximately preserving their order: a single hinted rollout per problem ranks nearly indistinguishable models as accurately as five unhinted rollouts. Our results suggest that hinting is a general primitive for improving the accuracy and cost of downstream scaling laws and other prediction tasks in model development. 

---
# Qubrio: High-Performance Quantum Compilation via Multi-Agent LLM Collaboration 

**Authors**: Jixuan Ruan, Zhuo Cui, Zhengding Hu, Zhongkai Yu, Xiang Fang, Yue Guan, Jason Ludmir, Phillip E. Weinberg, Xiuzhe Luo, Shengtao Wang, Sonia Lopez Alarcon, Hezi Zhang, Yufei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2611.02269)  

**Abstract**: Large Language Models (LLMs) offer an adaptable alternative to traditional quantum compilation heuristics, which bottleneck progress by demanding costly manual redesigns whenever hardware evolves. However, naive single-pass LLM approaches fail under the unique challenges introduced by quantum compilation. We introduce Qubrio, a practical agentic framework that significantly reduces heuristic redesigns through a three-level decomposition. By (1) partitioning programs into sequential operation stages, (2) deploying specialized agents for placement, routing and optimization, and (3) using disaggregated feedback for precise error correction, our system reliably navigates complex physical constraints. Beyond serving as a direct compiler, our LLM framework also uncovers novel strategies that facilitate convoy-style shuttling, substantially enhancing hardware concurrency without sacrificing fidelity. We further integrate these strategies into the previous state-of-the-art (SOTA) compiler, recovering a significant fraction of the LLM's performance advantages. On realistic workloads, our compiler achieves significant improvements in hardware runtime and program fidelity over the SOTA baseline PowerMove, seamlessly adapting to new hardware capabilities. Qubrio features an open user interface for practical use, and its code is publicly available at this https URL. 

---
# Metamorphic Testing for Evaluating Robustness of Sentiment Analysis Models 

**Authors**: Sowjanya Karanam, Jayanth Bhargav  

**Link**: [PDF](https://arxiv.org/pdf/2611.02248)  

**Abstract**: Machine learning systems are challenging to test using traditional software testing methods because their behavior is learned from data rather than explicitly programmed. This challenge is particularly significant in natural language processing (NLP), where small input variations can lead to unexpected model behavior. Conventional evaluation metrics, such as accuracy and F1-score, often fail to capture such inconsistencies. Metamorphic testing addresses this limitation by evaluating expected relationships between original and transformed inputs. With the increasing prevalence of generative artificial intelligence (GenAI) and language-based AI systems, the need for systematic robustness testing has become even more pronounced. Motivated by this growing need, this paper presents a lightweight, batch-based metamorphic testing approach for evaluating the robustness of such systems. In particular, we consider sentiment analysis as a representative NLP task and evaluate a pretrained classifier under four transformations: synonym replacement, typo injection, word-order variation, and negation. Using consistency score, flip accuracy, and confidence change, we identify strengths in handling semantic variations and vulnerabilities to textual noise and logical transformations. These findings highlight the value of metamorphic testing in identifying behavioral inconsistencies beyond conventional performance metrics and demonstrate its potential as a systematic framework for evaluating the robustness of language-based AI systems and guiding performance improvements before deployment. 

---
# Iterative Gumbel Planning for Continuous Control 

**Authors**: Shaohuai Liu, Weirui Ye, Yilun Du, Le Xie  

**Link**: [PDF](https://arxiv.org/pdf/2611.02236)  

**Abstract**: Search-based planning remains a powerful policy improvement paradigm in reinforcement learning, driving milestone successes in discrete domains. However, extending this paradigm to continuous control exposes a structural mismatch: search evaluates a finite set of action candidates, whereas continuous RL typically relies on Gaussian policies. We show that winner-action distillation into a continuous Gaussian induces a \textit{projection error} and a stepwise variance-expanding gradient tendency when the selected action is sufficiently far from the current mean. Empirically, this tendency is associated with elevated terminal variance and a return gap on the evaluated tasks. To address this, we propose \textbf{Iterative Gumbel Planning (IGP)}. By adopting a per-dimension factorized categorical policy, IGP naturally aligns the policy support with the search candidates, eliminating the projection error and reducing the residual approximation to a bounded discretization error. To bypass the combinatorial explosion of high-dimensional discrete action spaces, IGP pairs Gumbel Top-$K$ candidate sampling with an MPPI-style multi-round iterative search to refine the policy at each state. Empirically, IGP achieves state-of-the-art aggregate performance and sample efficiency on DMControl and HumanoidBench. Furthermore, it produces more stable action sequences with lower temporal variance, demonstrating its practical potential for real-world robotic deployment. 

---
# Discrimination, Calibration, and Ranking: A Unified Matrix Framework 

**Authors**: Ronaldo Cristiano Prati, Farangis Johari  

**Link**: [PDF](https://arxiv.org/pdf/2611.02215)  

**Abstract**: Discrimination, calibration, and ranking are the three pillars of classifier evaluation, yet no single algebraic language spans all three. A recent indicator-matrix account unifies discrimination measures (precision, recall, F1, Matthews correlation) across binary, multiclass, multilabel and related settings, using Boolean cell operations on indicator label matrices, generalised to soft labels by t-norms. Calibration measures (Brier score, log-loss) and ranking measures (nDCG, AUC) fall outside it. We close the gap with two further classes of cell operators: deviation operators phi(y - yhat), which give a Calibration Framework, and rank- and slice-dependent operators (positional gains for DCG, column concordance cells for AUC), which give a Ranking Framework. Both inherit the micro/macro/exemplar aggregation architecture. For calibration, the three Brier aggregations collapse to one scalar; the Murphy decomposition holds in matrix form (unbinned, binned and isotonic); and, within the Brier-type class-indexed family, a measure is skew-invariant if and only if it depends only on the class-conditional errors alpha_j and beta_j, the calibration analogues of recall and specificity. For ranking, raw DCG collapses across aggregations while nDCG does not; per-class nDCG and per-class AUC are related by a row-column duality; micro-AUC differs from pooled ROC-AUC; raw positional scores are never strictly proper, and nDCG and AUC need not be proper. The skew-invariant per-class building blocks share one class-conditional expectation template. Experiments on two multilabel benchmarks illustrate the distinctions. Many of the results are machine-checked in Lean 4; the proofs, the experiments and the this http URL package are public: this https URL 

---
# Forecasting Earth Surface Change in Embedding Space 

**Authors**: Jingtao Li, Sadiq Jaffer, Anil Madhavapeddy, Pedro Sousa, Anthony Ortiz, Kyle Gao, Zhengpeng Feng, Yanfei Zhong, Kevin White, Srinivasan Keshav  

**Link**: [PDF](https://arxiv.org/pdf/2611.02211)  

**Abstract**: Repeated satellite imagery permits understanding of how our planet changes over time, letting us predict its future and plan mitigating actions. Most existing studies that rely on a time series of satellite observations forecast the spectral values at each pixel for a future target date. However, clouds, shadows, and sensor noise often corrupt these observations, so that spectral forecasters waste much of their capacity on low-level features or noise rather than real change. With the advent of Earth foundation models, we argue that forecasting in embedding space is a feasible and more effective alternative. To this end, we design a token space that handles irregular histories from different sensors and foundation models and supports forecasts at any query date. Tokens from this space are inputs to our proposed forecaster, E$^2$Forecast, which is a transformer with two branches: a Gaussian branch that directly predicts a mean point forecast and per-dimension standard deviations, and a rectified-flow branch that generates forecast ensembles. On Sentinel-1 and Sentinel-2 benchmarks, forecasting embeddings instead of spectral values halves the cosine distance (from 0.20 to 0.10) and greatly reduces overconfidence, lowering the z-score standard deviation from 2.63 to 1.22. By comparing new observations with its forecasts, E$^2$Forecast can also detect anomalous changes without labels or retraining, raising F$_1$ from 41.58 to 48.70 on a global benchmark. This consistent superiority is validated on nine remote sensing foundation models. Beyond change detection, we believe our approach advances the field by permitting effective reuse of diverse pretrained task heads. Our code is publicly available at this https URL. 

---
# Test-Time Scaling for Open-Ended Reasoning at Small Model Scale 

**Authors**: Adam Y. Ahmed, Abdullah G. Elafifi  

**Link**: [PDF](https://arxiv.org/pdf/2611.02208)  

**Abstract**: A vote over a model's samples can only help when the samples disagree. For open-ended generation the binding question is therefore not how to combine samples but whether a model produces samples worth combining, a question test-time scaling has answered for math but not for free-form text. We study it at small scale with semantic cluster voting, a judge-free aggregator that groups a small instruct model's samples by meaning and returns the largest cluster's representative, with no reward model or self-judging. On a 250-prompt benchmark with three tiers of answer checkability, the same pattern holds across three model families (Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2B) and two generation seeds: voting raises accuracy on the math control, where samples vary but stay checkable (a gain of +10 to +29 points across the three models), yet adds at most +0.30 judge points on open-ended text. On a larger 200-problem MATH-500 run with the primary model, the math gain is +8.5 points (95% CI [+4.0, +13.0]). Almost every open-ended sample falls into one meaning cluster, so there is little to select over. The collapse is consistent with instruction-tuning rather than model size, and raising decoding temperature does not recover usable diversity, because it costs more quality than it adds (net -0.64 judge points, 95% CI [-0.93, -0.37]). 

---
# ANCHOR: KL-Anchored Adaptation for Stable Offline-to-Online RL 

**Authors**: Geonwoo Cho, Jaegyun Im, Doyoon Kim, Lexin Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.02177)  

**Abstract**: Offline-to-online reinforcement learning (RL) follows the pretrain-finetune paradigm: an agent is first pretrained on historical data and then adapted through online interaction. We study offline-to-online adaptation without offline replay, where the agent must improve using only the pretrained model and newly collected experience. A key challenge is to avoid catastrophic failure, a sharp early performance collapse during online fine-tuning, while still achieving strong asymptotic performance. Through controlled experiments, we show that removing offline replay can induce catastrophic failure, while retaining strong conservative regularization from offline RL can suppress final online performance. We also find that common stabilization strategies do not reliably prevent early collapse, and that the critic-side mechanisms we test are insufficient on their own. Motivated by these findings, we propose ANCHOR, a simple adaptation method that anchors the actor to the offline-pretrained policy early in online fine-tuning, anneals this constraint over time, and retains only a small residual conservative critic regularizer. We provide a theoretical interpretation showing that KL anchoring helps preserve critic reliability across nearby occupancy distributions, with a two-phase regret bound explaining when anchoring should be maintained or relaxed. Across FrankaKitchen, AntMaze, and Adroit tasks, ANCHOR substantially reduces catastrophic failure while maintaining strong asymptotic performance. Ablations show that KL annealing and the residual conservative weight are important for balancing early stability and online adaptation, while additional analysis shows that ANCHOR is most beneficial when the pretrained policy contains useful behavior to preserve. Code is available at this https URL. 

---
# Black-Box Adversarial Example Attacks on Machine Learning based Election Systems 

**Authors**: Subek Acharya, Syed Ayon, Khaleque Md Aashiq Kamal, Alina Barnett, Surya Eada, Kaleel Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2611.02166)  

**Abstract**: Machine learning models have shown exceptional accuracy for ballot classification in election systems. Recent research has demonstrated that these ballot classification models are vulnerable to adversarial example attacks. This is a critical security issue because a successful attack could alter the outcome of an election and undermine the democratic process. However, all adversarial example attacks in the voting domain have assumed a white-box threat model, where the classifier parameters are known. In this work, we advance the field of machine learning in election systems by analyzing a realistic black-box adversary. We implement and evaluate eleven state-of-the-art black-box attacks: query based attacks ADBA, Square, SurFree, RayS, and SA-MOO. We also test adaptive transfer based black-box attacks with APGD-$l_\infty$, APGD-$l_2$, APGD-$l_1$, PGD $l_0$, PGD $l_0$+$l_\infty$, and PGD $l_0$+$\sigma$. Our attacks span multiple different norms ($l_\infty$, $l_2$, $l_1$, and $l_0$) and three different black-box adversarial threat models (decision based, score based and transfer based). In addition to digital analyses, we physically print and scan $20,000$ adversarial examples to evaluate the real-world performance of different black-box attacks. Our finding demonstrate an even larger physical-digital robustness gap exists for black-box adversarial examples than for white-box adversarial examples. Lastly, we conduct the first ever adversarial example transferability study in the voting domain and test adversarial training and barrier zones as two possible black-box defense strategies. 

---
# Value Forcing: Aligning Return Distributions at All Noise Levels 

**Authors**: Chongyi Zheng, Zhiyuan Zhou, Qiyang Li, Benjamin Eysenbach, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2611.02160)  

**Abstract**: Learning accurate value functions is central to reinforcement learning (RL), yet remains challenging because supervision is limited and indirect: ground-truth values are unavailable, and each transition in the data only provides an immediate reward that can be used to construct bootstrapped targets as supervision. This challenge is particularly pronounced when data is limited or reward signals are sparse, as is common in real-world applications such as robotics. We argue that the key to accurate value learning in this limited data regime is extracting richer learning signals from the available data. In this paper, we introduce Value Forcing, a new approach that uses flow matching to estimate return distributions and enables accurate value learning from limited data. Our key insight is that we can extract richer learning signals by enforcing Bellman consistency at every noise level. Empirically, Value Forcing improves the accuracy of estimating the return distribution, and prevents the distribution from collapsing to a single value, as occurs with prior methods. We instantiate Value Forcing with two practical value-learning algorithms: VF, an actor-critic method, and IVF, an implicit value-learning method. We evaluate both algorithms across $26$ long-horizon tasks in the offline RL setting and find that they outperform prior value-learning methods, with particularly large gains when data is limited. Our project website is available at this https URL. 

---
# Semantic Interoperability in Federated Digital Twin Systems: A Systematic Review Through the Levels of Conceptual Interoperability Model 

**Authors**: Manonmani Sekar, Nasim Nezamoddini  

**Link**: [PDF](https://arxiv.org/pdf/2611.02125)  

**Abstract**: When two companies connect their digital twins, software models of their plants kept up to date with live data, each twin must translate the other's data into its own terms, and that translation rule, called a mapping, can stop being correct when one company changes the meaning of its data without telling the other. This PRISMA 2020 systematic review screened 5,833 records and read 119 full texts to determine whether any published system detects such a broken mapping and repairs it while the twins operate. In all 32 included studies the mapping is fixed before operation, and the two ingredients of run-time repair appear only separately: cross-company matching without any check that the match still holds, and detection and repair of changed meaning within a single company. The review turns this gap into six requirements, each illustrated on one running example, that a system claiming run-time semantic interoperability between independently owned twins should meet. 

---
# Testing concept-level audits of melanoma classifiers with diffusion counterfactuals and matched positive controls 

**Authors**: Md. Shohanoor Rahman, Md. Hasibul Islam, Md Kishor Morol, Moinul Hossain, Dip Nandi, Liew Tze Hui  

**Link**: [PDF](https://arxiv.org/pdf/2611.02119)  

**Abstract**: Concept-based explanations of melanoma classifiers are rarely tested against controls with known targets. We built a counterfactual audit from a diffusion model conditioned on seven 7-point checklist concepts and tested it with such controls. Colour-calibrated start noise reduced colour outliers from 78.5-84.8\% to 6.6-13.3\%. Four of seven knobs were target-dominant but also changed other concepts. Quartile-based positive controls were confounded, and the audit failed its pre-specified rule. It recovered the targets of all 16 controls matched on the generator's pseudo-labels and of 12 of 12 controls labelled by a second evaluator, for three concepts; concept activation vectors recovered fewer (11 of 16). In 18 permuted-label classifiers, significant sweep responses and target rankings also arose, whereas standardised effect margins separated the controls from these null classifiers. The pigmentation sweep raised the melanoma logit in 15 of 15 ISIC 2020 classifiers (+3.12 on average) and 6 of 6 HAM10000 classifiers (+4.01), about four times the null response in scale-free units; the regression effect did not transfer. An exploratory decomposition, evaluated against the controls, suggested that knob cross-talk partly hides vascular responses. Matched controls and null classifiers are practical checks of concept audits. 

---
# ZT-DAC: ML-Driven Dynamic Access Control Policy for Zero Trust in DER Networks 

**Authors**: Arun B. Chandrasekaran, Damilola Popoola, Souradeep Bhattacharya, Manimaran Govindarasu  

**Link**: [PDF](https://arxiv.org/pdf/2611.02107)  

**Abstract**: The increasing penetration of Distributed Energy Resources (DERs) in the smart grid has significantly expanded the cyber-attack surface. Traditionally, DER networks rely on static role-based access control (RBAC) mechanisms that assume persistent trust after authentication. This limitation enables adversaries to maintain unauthorized privileges and move laterally across DER systems, disrupting grid operations. Although attribute-based access control (ABAC) enables fine-grained and context-aware access decisions, it does not fully eliminate implicit trust. Zero Trust (ZT) principles can be incorporated to enhance ABAC policies. This paper proposes a dynamic access control framework for implementing ZT in DER networks (ZT-DAC). The proposed solution employs a two-stage verification process and integrates an Open Policy Agent (OPA)-based RBAC with a machine learning (ML)-based ABAC to enforce continuous and adaptive trust score assessment. To support realistic DER operations, a DER-specific access control model is developed and used to generate a domain-informed dataset of access requests incorporating both legitimate and adversarial behaviors. Experimental results on the dataset show that the OPA-based decision layer achieves \(91.92\%\) accuracy, with a False Positive Rate of \(14.45\%\), while the ML model achieves a Mean Absolute Error of \(2.074\) and an \(R^2\) of \(0.891\) in trust-score prediction. The ABAC layer achieves \(93.27\%\) accuracy, with a FPR of \(3.88\%\), validating its practical feasibility within realistic DER environments. 

---
# Consolidating Reasoning with Test-Time Training 

**Authors**: Haotian Wu, Zeming Chen, Hao Zhao, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2611.02104)  

**Abstract**: Effective multi-agent collaboration requires finalizing solutions from reasoning that is distributed across the individual attempts of many agents. However, useful intermediate reasoning can remain unexploited when collaboration is limited to voting on conclusions reached among individual agents, or aggregating long, noisy collections of reasoning trajectories in context. In this work, we propose TTT-Consolidate, a framework that uses multi-agent reasoning trajectories as adaptation data to improve a model's subsequent reasoning. TTT-Consolidate jointly consolidates trajectories into parametric representations through test-time training, then generates a new solution by conditioning on the consolidated information. We meta-learn how to consolidate using bi-level optimization at training time, where an inner loop encodes reasoning trajectories into a lightweight LoRA adapter, and an outer loop learns meta-parameters for the inner loop such that the consolidation is helpful for the model to synthesize a final solution.
As no large-scale corpus of diverse reasoning trajectories exists for training aggregation methods, we construct the OpenParallelThinking corpus composed of 12,810 math problems paired with a mixture of trajectories from six LLMs. Evaluated on challenging math and STEM benchmarks, TTT-Consolidate outperforms the best aggregation baselines by up to 20 percentage points, even under matched compute and runtime settings, with the gap widening at larger model scales. TTT-Consolidate also recovers from unsuccessful individual trajectories more robustly, an ability that improves as the number of available (even incorrect) trajectories increases. Finally, we demonstrate that TTT-Consolidate benefits from both SFT and RL-based outer loop objectives. We will release TTT-Consolidate and OpenParallelThinking for use in future work on reasoning aggregation. 

---
# EarthJev: Calibrated, Non-Generative Answers to Natural-Language Questions about Sentinel-2 Land Cover 

**Authors**: Domenico Benfenati, Antonio Maria Rinaldi, Cristian Tommasino  

**Link**: [PDF](https://arxiv.org/pdf/2611.02096)  

**Abstract**: Vision-language models for remote sensing answer questions about satellite images by generating text: every answer costs a decoding loop, has to be parsed back to an option, and comes without a usable probability. We present EarthJev, a 0.8B-parameter model that answers typed questions about multispectral Sentinel-2 patches without generating any token. All 12 bands enter the vision encoder of Qwen3.5-0.8B through a widened input layer initialised from the RGB kernels, and, following the open OneJev recipe, the answer is read from the logits of the option letters at a single position and trained with cross-entropy plus Brier score. Under a pre-specified analysis plan, we compare it with the same backbone fine-tuned to generate the answer and with three multimodal language models fine-tuned with LoRA, all trained on the same 70,890 questions from 5% of BigEarthNet v2.0 and timed in one job on the same GPU; VQA models that classify over a fixed answer vocabulary are reported as a reference for the cost of answering only the questions seen in training. EarthJev answers a question in 19.9 ms, 4.3 to 8.7 times faster than the multimodal language models and 6.8 times faster than its own backbone used generatively; when all the questions about an image are asked together, it answers them in 42.8 ms, 3.7 to 10.1 times faster than the batched generative models. On 36,028 test questions it reaches 86.0% accuracy (three seeds), 0.6 and 1.2 points below the two 2B multimodal models and 3.5 points below the remote-sensing EarthDial-4B, with answers that are valid by construction and calibrated probabilities (ECE 0.006). Replacing the nine non-RGB bands with their mean costs 4.5 points, so the model uses the multispectral input, and in an ablation training the whole vision encoder raises accuracy to 88.4% at the same cost. Code and model: this https URL 

---
# HQNN-CL: Regime-Aware Hybrid Quantum Neural Network with Continual Learning for Financial Forecasting 

**Authors**: Muhammad Atif Saeed, Akhtar Jamil, Subhan Ullah  

**Link**: [PDF](https://arxiv.org/pdf/2611.02091)  

**Abstract**: Financial time-series forecasting is still difficult owing to the non-stationary nature of the market, regime changes, and catastrophic forgetting, which makes conventional deep learning models hard to adapt in the long term. To achieve uncertainty-aware robust financial forecasting in the face of changing market conditions, this study proposes a Hybrid Quantum-Classical Neural Network with Continual Learning (HQNN-CL). The proposed framework combines Gaussian Hidden Markov Model (HMM) based regime detection, BiLSTM temporal encoding, adaptive-depth parameterized quantum circuits, Elastic Weight Consolidation (EWC), experience replay, and probabilistic forecasting in a single architecture. The framework was tested on four real-time stock datasets: Microsoft, Systems Limited, Teradata and Tata Consultancy Services, and compared with seven state-of-the-art forecasting models, including recurrent and Transformer-based models. The experimental results show that HQNN-CL has outperformed other methods on the Systems Limited (RMSE = 3.0149) and Teradata (RMSE = 0.8909) datasets and is within 0.32% and 0.40% of the best baseline method for Microsoft and Tata Consultancy Services, respectively. The proposed framework also generates well-calibrated probabilistic forecasts within the range of 89.5% to 96.5% empirical coverage rate, has zero catastrophic forgetting (BWT = 0.0000) for all sequential learning tasks, and consistently shows positive forward transfer. Ablation and statistical significance analyses also validate the performance of adaptive quantum feature learning and continual adaptation under heterogeneous market conditions, regime-wise. 

---
# Scene-Adaptive Readout and Decoding in Diffusion Driving VLAs 

**Authors**: Seungjun Yu, Jiho Choi, Inha Kang, Hojun Choi, Seungwook Kim, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2611.02076)  

**Abstract**: Diffusion driving vision-language-action models repeatedly execute the backbone at every denoising stage, incurring substantial inference cost, while it remains unclear whether all scenes require the same decoding computation. To distinguish when a trajectory becomes readable from how much further refinement it requires, we analyze trajectory readout across network depth and decoding progress. We identify a depth-decoding dissociation in which late-layer representations already support useful trajectory readout after the first forward sweep, whereas early-layer representations benefit more from continued decoding. We further find that both the preferred readout layer and decoding depth vary across scenes. Based on these observations, we propose a scene-adaptive readout and decoding framework. A scene-conditioned readout MoE adaptively weights and combines trajectory predictions from layer-specific late-layer representations after a single backbone sweep, while an adaptive decoding policy applies additional native refinement only when required by the scene. We evaluate our framework on two diffusion driving VLAs across the Waymo E2E and nuScenes benchmarks. Our method improves validation RFS by up to 0.56 while also improving test performance, and achieves an end-to-end speedup of up to $6.0\times$ on Waymo E2E, while consistently reducing trajectory error on nuScenes. These results demonstrate that scene-level readout and decoding allocation improves planning quality while avoiding unnecessary decoding. Our code is available at this https URL 

---
# Do Embodied VLMs Actually Use Their Scene-Graph Memory? 

**Authors**: Shubh Garg, Divyansh Pandey, Vedant Khandelwal, Niyati Rawal, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2611.02056)  

**Abstract**: Embodied agents that answer questions about their route need a memory of what they saw, and scene graphs are a natural form for it. A vision-language model (VLM) can receive such a graph as text or through a graph encoder, but accuracy alone cannot show whether the graph holds the episode's facts, reaches the model, or is actually relied on. In this diagnostic study, we introduce a graph-swap test for the last of these: each episode is given another episode's graph, and we count how many answers change. On an embodied question-answering benchmark with routes through two to seven rooms, we use it to compare ways of delivering scene-graph memory to a fine-tuned VLM. As text, the memory is used strongly, adding 30 points on route questions. As a graph, it is not, and we trace why: the encoder preserves the information, but standard pooled and late fusion discard it. Giving each node its own input token recovers part of it, a gain of 8.3 points on route questions that the swap test confirms comes from the graph. The test also exposes a shortcut that accuracy hides: when the same facts appear in the text, the model ignores its graph. The test thus offers a cheap check of whether a VLM uses its scene-graph memory, and our analysis points to how that memory should be delivered. 

---
# Beliefs in feedback: Factored representations of interacting agents in transformers 

**Authors**: David A. Ehrlich, Manuel Baltieri, Fernando E. Rosas, Alexander B. Boyd  

**Link**: [PDF](https://arxiv.org/pdf/2611.02047)  

**Abstract**: Complex systems, from ecosystems to brains, are usually modeled as modular components coupled through feedback rather than as indivisible wholes. Can a transformer trained on the joint behavior of a group of interacting agents be understood in the same way? In this work, we develop a computational-mechanics framework to study agents interacting in feedback loops. We formally prove that any joint stochastic process can be presented as resulting from the behavior of agents interacting in feedback, and the resulting process is generated by a hidden Markov model whose Bayesian beliefs factorize into separate components per agent. Furthermore, our experiments show that transformers trained on such feedback sequences make use of this separability, disentangling the beliefs about the underlying factors into separate subspaces of their residual stream. This representation is found to scale linearly (instead of exponentially) with the number of agents. Moreover, interventions show that these representations are causally efficacious, allowing for targeted steering of individual agents' beliefs by selectively inserting counterfactuals. However, coarse-graining the behavior is found to disrupt the factorization, and can force a higher-dimensional representation. Together, the results establish both the dimensional advantage of a factorized representation and the ability of transformers to exploit it, laying the groundwork for investigating traces of compositional predictive architecture in pretrained models. 

---
# Query-efficient scan-and-search watermark localization 

**Authors**: Liyuan Wang, Xinsong Ma, Ziye Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.02030)  

**Abstract**: Recently, the detection of AI-generated content has been of particular importance in various fields of our society. For accountability purposes, many large language models (LLMs) have implicitly watermarked their outputs. Localizing such watermarks requires a piece of text to undergo an additional verification process. Existing localization methods typically rely on dense token-level scoring, making verification costly even when an LLM-generated watermark only sparsely appears. To tackle this problem, we design a new adaptive allocation strategy that first maintains broad document coverage and then progressively concentrates the detection budget on regions that exhibit stronger watermark evidence or are more likely to contain watermarked content. We then transform the statistical signals from the selected regions into e-values and employ the proposed token-weighted e-BH procedure to determine the localized watermark regions. The resulting framework is called BELL (Budgeted E-evidence Localization for LLM Watermarks). Theoretically, BELL can control the token-weighted false discovery rate (FDR) without independence assumptions across regions. Finally, extensive experimental results demonstrate that BELL consistently outperforms competing methods across multiple LLMs, datasets, and watermarking schemes. 

---
# Reconstruction as Spatial Evaluation: Diagnosing Cross-View Spatial Intelligence in Large Vision-Language Models 

**Authors**: Junfei Wu, Qiuyue Wang, Xuhong Huang, Qihang Peng, Mingsheng Li, Jian Guan, Qiang Liu, Shu Wu, Liang Wang, Shuai Bai, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2611.02026)  

**Abstract**: The spatial intelligence of large vision-language models (LVLMs) requires the ability to combine local visual observations into a geometrically consistent representation of the physical world. Many static benchmarks assess isolated spatial judgments, such as relative direction or distance, while interactive embodied evaluations combine perception with exploration, planning, control, and simulator-specific execution. To bridge this gap, we introduce RECONSPATIAL, a benchmark centered on multi-view 3D scene reconstruction across indoor, tabletop, and outdoor environments. Given real multi-view images and target-object queries, LVLMs are required to jointly predict oriented 3D bounding boxes in a shared, camera-anchored metric coordinate frame. Reconstructions are evaluated along two complementary axes: Structure for relative layout and Metric for physical scale and camera-anchored placement. Nine atomic spatial QA tasks, grouped into Geometric Perception and Spatial Relations, probe spatial capabilities involved in joint reconstruction. Offline navigation extends evaluation to path planning. Evaluation of 18 frontier LVLMs reveals a consistent gap between relative scene structure and metric reconstruction; even GPT-6-Astra scores 81.4% on Structure but only 58.7% on Metric. Tracing this gap, diagnostic analyses link atomic spatial capabilities to reconstruction quality and show that relational consistency alone does not ensure metric accuracy. Experiments with 3D map injection further separate scene construction from path planning, revealing how these geometric errors limit navigation success. Finally, spatial fine-tuning yields broad transfer across both spatial and general vision-language benchmarks. 

---
# Scaling Makes It Possible: How Large Models Master Impossible Languages 

**Authors**: Yaning Jia, Chunhui Zhang, Chiyu Ma, Zhongyu Ouyang, Ivory Yang, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2611.02015)  

**Abstract**: While language models (LMs) excel in natural language tasks, their ability to learn impossible languages--those that violate the structural, cognitive, or logical constraints that make human languages learnable and functional--remains debatable. This controversy is further fueled by a divide between linguistic theorists and empirical researchers working with small-scale LMs. To address this controversy, we introduce semantic coherence. This concept measures how effectively semantic information is expressed in language structures and examines the learning capabilities of LMs across multiple scales. By fine-tuning billion-parameter LMs on impossible language samples, we demonstrate that larger LMs achieve similar performance on impossible and possible languages, whereas million-parameter LMs exhibit significant degradation when trained on impossible languages. This performance gap widens as semantic complexity increases. We further show that scaling up the LM size fundamentally shifts the learning focus from grammatical rules to semantic understanding, with billion-scale LMs exhibiting a threefold reduction in sensitivity to low-level linguistic perturbations while maintaining stable performance under high-level semantic perturbations. 

---
# TrackMem: Tracking Evolving Beliefs in Long-Horizon Financial Dialogue 

**Authors**: Austin M.Y. Cheung, Harsh Vardhan Gupta, Go Nam Lui, Lok Kan Chan, Ginny Y. Wong, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2611.02001)  

**Abstract**: Memory systems for LLM agents are evaluated mostly on social dialogue, where facts accumulate and rarely change, so remembering reduces largely to finding the turn in which something was said. Financial dialogue breaks this assumption: figures are revised, stances shift without being restated, rules stated once govern later questions, and a single company is discussed under several analytical lenses. Memory systems that retrieve the passages most similar to a question, or the most recent ones, therefore return superseded values, miss rules that share no wording with the question, and cannot recover a stance spread across sessions. We present TrackMem, a per-speaker graph memory that tracks the history of figures, separates perspectives by entity and lens, and preserves each speaker's actions in chronological order. To evaluate it, we introduce FinLoCoMo, a benchmark of 15 long-horizon investment conversations with 765 questions in two splits: one following the LoCoMo categories and one targeting four failure modes. With two readers, TrackMem outperforms existing memory systems on both splits. On the failure-mode split it reaches 73.9% macro accuracy against 69.4% for the strongest baseline, with the largest gains on questions about revised figures (86.5% against 73.4%). 

---
# SurgStream: A Workflow-Anchored Proactive Streaming Framework and Benchmark for Real-Time Surgical Video Understanding 

**Authors**: Ruipeng Ma, Qing Xu, Jie Liang, Zhen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.01982)  

**Abstract**: Surgical video understanding is essential for intraoperative assistance such as workflow monitoring, instrument preparation and procedural documentation. Delivering such assistance in real time requires a model to jointly perceive the ongoing scene, remember past surgical events, and respond before clinically relevant moments pass. Existing approaches provide limited support for jointly retaining procedural evidence and deciding when to respond in streaming surgical video. In this work, we propose SurgStream, a workflow-anchored proactive streaming framework for real-time surgical video understanding. We first introduce Workflow-Anchored Memory (WAM) to record structured instrument-action-target observations and fold them into phase-level summaries, providing compact and queryable surgical memory. Deadline-aware Transition Learning (DTL) is then devised to supervise response states within clinical windows, enabling timely responses before procedural deadlines. We further introduce SurgStream-Bench, comprising 11.7K streaming interactions across various tasks covering real-time perception, backward memory queries, forward proactive QA, proactive alerts and caption memory. Experiments on SurgStream-Bench show gains over state-of-the-art methods, especially in memory and forward proactive QA. The source code will be available. 

---
# Beyond Prompt Injection: The Tool Update Gap in LLM Agent Security 

**Authors**: Shiyu Kan, Yue Qi, Yufei Han, Xinjian Luo, Guojun Xiong, Xiaofeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.01976)  

**Abstract**: Large language model based agents increasingly rely on external tools to invoke services and interact with web resources. These tools can evolve through software updates, yet agentic frameworks may continue trusting them without re-verifying their safety after implementation changes. This tool update gap creates a supply-chain vulnerability in agentic systems: because updated tool code executes outside the LLM's context window, malicious functionality embedded in tool updates can bypass defenses centered on model-visible interactions, leaving agents exposed to stealthy, persistent threats that existing in-context defenses can hardly address.
To thoroughly investigate this threat, we introduce ETUDE (Exploiting Tool Updates for Direct Execution), an attack that embeds malicious functionality into updates of initially benign tools to produce compromised tools, while preserving their legitimate tool functionality. Unlike prompt-based attacks, ETUDE operates entirely outside the LLM's context and requires no adversarial instructions, thereby bypassing defenses restricted to model-visible content. We evaluate ETUDE across three representative attack workloads and seven widely used commercial and open-source agentic frameworks, including Gemini CLI, Codex CLI, and Claude Code. The results show that ETUDE achieves a 97.6% ASR while maintaining a 95.9% legitimate task completion rate.
We further examine existing defenses and find that they provide insufficient protection against this threat. These findings expose a fundamental blind spot in current agent security practices: trust established for a tool does not necessarily remain valid after its implementation updates. 

---
# MUPA${}^2$S: Multimodal Unified Perception with Asymmetric Attention for Stress Assessment 

**Authors**: Stefanos Gkikas, Thomas Kassiotis, Yu Fang, Christian Arzate Cruz, Miao Cheng, Zongyi Jiang, Zacharias Kamarianakis, Houshyar Asadi, Randy Gomez, Raul Fernandez Rojas, Giorgos Giannakakis  

**Link**: [PDF](https://arxiv.org/pdf/2611.01974)  

**Abstract**: Stress is a leading contributor to long-term physical and mental health decline. The automatic assessment of stress is a promising approach for continuous monitoring and mitigating human biases. However, the development of automated systems remains challenging due to the heterogeneity of physiological and behavioral signals. This paper presents MUPA${}^2$S, a multimodal framework for automatic stress assessment that combines facial video and ECG signals within a unified perception architecture. Both modalities are mapped into a shared token representation without modality-specific components, and processed through asymmetric attention comprising segment-local cross-attention and global self-attention across segment states. Experiments on a newly collected 58-subject stress dataset, evaluated under a stratified subject-level protocol, show that facial video achieves 69.44% test accuracy, while ECG-only performance remains close to chance under the proposed representation. Unified Video-ECG fusion achieves a test accuracy of 69.58% at approximately half the number of parameters of feature-level fusion, with negligible additional computational cost. Feature-level fusion does not exceed the video-only baseline, while unified input-level integration provides a parameter-efficient mechanism for jointly processing the two modalities. 

---
# Knowledge-Guided Floodwater Level Reasoning for Cloud-Edge Collaborative Monitoring 

**Authors**: Zhengshu Zhou, Shaoyu Wang, Ziyi Geng, Lu Tao, Qian Long  

**Link**: [PDF](https://arxiv.org/pdf/2611.01970)  

**Abstract**: Frequent urban floodwater poses a serious threat to the safety and efficiency of intelligent transportation systems. Existing vision-based methods for floodwater level detection face two main limitations. First, the lack of explicit reasoning leads to poor detection performance and little interpretability. Second, these methods struggle to generalize to the complex and varied real-world road scenes seen from different viewpoints. To address this, we reformulate the task as an explicit visual reasoning problem and propose HydroReasoner, a floodwater level reasoning framework designed for cloud-edge collaboration. HydroReasoner builds on Qwen2.5-VL and is trained with Group Relative Policy Optimization. To address the scarcity of reasoning process annotations, we build a floodwater domain knowledge graph HydroKG and design a novel KG path reward. Through deterministic graph traversal over HydroKG, this reward supervises the model's reasoning process at a fine granularity, which improves both reasoning and generalization. For edge deployment, we apply 4-bit quantization to the model, cutting GPU memory usage by 57% and inference latency by 43% while out-of-domain accuracy drops by only 0.2 percentage points. We also propose a confidence-based cloud-edge routing mechanism, which improves system safety and eases the computational load on the cloud. In addition, we build HydroLevel, a floodwater benchmark dataset of 3,000 images, for training and evaluating the model. Experimental results show that HydroReasoner achieves detection accuracy 4.0 and 13.6 percentage points higher than conventional vision baselines in the in-domain and out-of-domain evaluations respectively, with performance dropping by only 4.4 percentage points out of domain. 

---
# Better Answers, Weaker Evidence Control: Evidence Bypass in On-Policy Distillation 

**Authors**: Ruoran Xu, Yilun Zhao, Ziyu Chen, Xiaoqiang kang, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2611.01967)  

**Abstract**: On-policy distillation (OPD) can improve answer accuracy while weakening responsiveness to supporting evidence. We investigate this separation in scientific-document question answering through matched document interventions that change the supported answer while preserving the question and unrelated context. We identify evidence bypass: a student retains an originally correct answer after the document changes to support a different one. On QASPER with a Qwen3-4B student, standard OPD raises original-document accuracy by 3.4 percentage points, lowers counterfactual accuracy by 5.3 points, and increases bypass by 15.7 points; the same directional pattern appears on PeerQA. Controlled perturbations localize the shift to answer-critical evidence, and bypass grows with the student's pre-distillation closed-book preference for the original answer. A factorized study of teacher targets, privileged information, divergence, and supervision density finds that explicitly supplying the teacher with annotated answer-critical spans yields the largest joint improvement in evidence response and claim support. These findings motivate Evidence-Paired OPD (EP-OPD), which combines on-policy supervision on matched documents supporting different answers to the same question with an answer-level margin that favors each document's supported answer over its paired alternative. Across both datasets and two student sizes, EP-OPD recovers 54-66% of standard OPD's counterfactual-accuracy loss while preserving 80-85% of its original-document accuracy gain. Our results establish evidence responsiveness as a distinct training and evaluation target alongside answer accuracy. Our code is publicly available at: this https URL. 

---
# Launder-Bench: Can Repository-Level Agentic Code Laundering Be Detected? 

**Authors**: Yuntong Hu, Hongji Pu, Zimo Li, Ruomeng Li, Xiaotong Liu, Ziyuan Qin, Qiyue Li, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.01966)  

**Abstract**: Coding agents can now rewrite repository-scale codebases while preserving tested behavior, creating a new challenge for plagiarism detection, license auditing, and software provenance. We ask whether such rewrites can still be traced to their source. Existing clone benchmarks emphasize fragments and bounded edits, while repository-transformation benchmarks evaluate functional correctness rather than provenance. Moreover, ranking the most similar source does not establish derivation when independent implementations can be similarly close. We define repository-level agentic code laundering and introduce Launder-Bench, to our knowledge the first repository-level benchmark for provenance under agentic rewriting, constructed from 282 real-world repositories. A source-hidden reconstruction protocol decomposes source behavior, reconstructs modules and repository structure, and retains rewrites only after execution-based verification. We evaluate the same provenance problem in two complementary settings: Launder-Bench-Pair contains 1,128 pairwise instances asking whether two repositories share provenance, while Launder-Bench-Attr contains 229 source-attribution instances asking which candidate is the source, with abstention when evidence is insufficient. Measured same-domain independent repositories establish how much similarity can arise without shared provenance and provide detector-specific reference bands. Existing methods remain far from saturating Launder-Bench: the strongest static method reaches only 0.66 AUROC on pairwise detection, while repository encoders rank the true source first on up to 22% of attribution queries but earn calibrated attribution credit on only 1-2%. These results reveal a systematic gap between relative code similarity and source-specific provenance evidence, providing a reproducible target for provenance detection under agentic rewriting. 

---
# OC-OPD: Observation-coordinated Supervision for On-Policy Distillation 

**Authors**: Ruoran Xu, Yilun Zhao, Ziyu Chen, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2611.01965)  

**Abstract**: Reverse and forward Kullback--Leibler (KL) divergences provide complementary supervision in on-policy distillation: reverse KL refines the student's active token rankings, and forward KL allocates mass to teacher-supported candidates. At student-generated prefixes, we form a four-observation matrix by treating the student and teacher as both candidate sources and evaluators. The diagonal self-observations measure uncertainty, while source-centered cross-observations recover the two KL objectives. We establish an exact identity equating the Fisher inverse-metric coupling of their logit gradients with the sum of these contrasts, yielding a common local descent direction for distinct positive distributions. Observation-coordinated on-policy distillation (OC-OPD) uses observations to set preferences, closed-form bargaining to coordinate gradients, and distributional discrepancy to scale the update. Across three heterogeneous student settings (non-thinking generation, thinking generation, and base-model initialization), OC-OPD achieves strong gains in mathematical reasoning and transfers effectively to out-of-distribution scientific question answering and code generation. Our code is publicly available at: this https URL. 

---
# Step-Zero Attack: Agent Safety Before Observability 

**Authors**: Tian Qin, Yuhang Jiang, Junzhe Chen, Lit Sin Tan, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2611.01953)  

**Abstract**: Existing attacks and defenses for coding agents act once content enters the model context: injected text persuades the model, and guards inspect what the model reads. Yet to judge whether processing an unknown archive, network response, or image is safe, an agent often must process it first, and that step can do harm before any content reaches the model. We call this first processing Step Zero, and harm done by it a Step-Zero attack. Across seven harnesses and six models, each agent received only a path or URL to a crafted input, paired with a benign twin showing the same visible fields. Step Zero is where most agents are already unsafe: 32/42 harness-model pairs overran a 4 MiB disk budget, 26/42 before the model saw any member, and four of the six models did so in every round (84/84), so the damage precedes any model-side defense. Every twin stayed within budget, so the attacker's bytes alone cause the harm. On network and image inputs the risk moves to what the agent writes: agents stored the body under the name the input declared, letting the input decide what is written, and their checks faded once cues were removed. Model-free tests trace the harm to the processing primitive, so we defend it there: a gate that counts the bytes each call produces held every routed extraction within budget while agents completed 25/28 tasks. Vendors should pair such gates with safety-alignment training that teaches agents to inspect an input before processing it. 

---
# VLSplat: Vision-Language Guided Object-Centric 3D Gaussian Splatting via Scene Graph 

**Authors**: Yuntae Jeon, Younho Jeon, Sujin Jin, Sungho Jo, Seunghee Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.01939)  

**Abstract**: Learning semantic representations for 3D Gaussian Splatting has recently emerged as a promising direction for 3D scene understanding. However, existing 2D-to-3D semantic lifting approaches rely primarily on 2D segmentation supervision and lack object-level priors, causing the learned representation to be driven by local mask evidence rather than holistic object structure. In this paper, we present VLSplat, a vision-language guided approach for object-centric semantic lifting in 3D Gaussian Splatting. Our method augments mask-based lifting with object-level semantic priors derived from vision-language models and refines Gaussian representations using an object-group scene graph. The refinement is performed through two language-guided modules: intra-group refinement suppresses semantic noise within object groups, while inter-group acquisition expands object support regions by incorporating structurally relevant candidates from neighboring groups. Extensive experiments on indoor scene datasets demonstrate that VLSplat improves the semantic consistency and structural completeness of object-level categories. Our code and models are available at this https URL. 

---
# RECEIPT: Measuring and Repairing the Return Assumption of LLM Agents in Web Service Operations 

**Authors**: Boyang Yang, Ziyao Yang, Yu Zhang, Duo Chai, Xiaoming Chen, Haoye Tian  

**Link**: [PDF](https://arxiv.org/pdf/2611.01930)  

**Abstract**: LLM agents operate Web infrastructure from the command line and rely on a return assumption, namely that each tool call returns within a bounded time and with its requested action finished. Many commands, however, ask a question before they act, and the agent's harness either closes the program's input or keeps it open as a terminal at which nobody types. With closed input, the program can return status 0 without the requested action. With open input, it waits until a time limit, and the agent pays in time and tokens. To the best of our knowledge, this paper is the first to measure and repair the return assumption. We collect 91,469 production tool calls and 89,047 public agent trajectories, test how 8 widely used harnesses launch commands, and build Litmus, a catalog of 158 checked command-line invocations, and Tripwire, a benchmark of 25 task pairs that differ only in whether a question guards the requested action. We then propose RECEIPT, which keeps the input open, reads what each process waits for from system calls, and returns this state to an unmodified agent through a model-checked protocol.
On Litmus, closed input causes 11.5% of tested invocations to return status 0 without performing the requested action, and 7 of 8 harnesses close input when no terminal is attached. An unanswered question at a plain terminal multiplies a task's token use by 2.9, and the input connection moves an LLM from 7th to 3rd place among 8. Across 5 LLMs, RECEIPT reduces false claims of success by 73.5% and completes all 25 guarded tasks on a second harness. An agent's reliability therefore depends on what its harness lets it observe as well as on its LLM. Agent builders should report what each tool call did, not only how it ended, and benchmark maintainers should fix and report the input connection before they rank LLMs. 

---
# AgenticMobility: An Open-Source Platform for Multi-AI-Agent Mobility Dynamics 

**Authors**: Jiaheng Xiao, Jiangtao Han, Jiachen Wang, Honghai Li, Lijun Sun, Kay Axhausen, Chang Liu, Zhengbing He  

**Link**: [PDF](https://arxiv.org/pdf/2611.01918)  

**Abstract**: Understanding mobility dynamics and the effects of urban transportation management policies requires examining how individual travel decisions, traffic conditions, and policy interventions influence one another over time. We present AgenticMobility, an open-source experimental platform for studying how mobility AI agents and other AI agents interact, adapt through repeated experience, and collectively shape mobility outcomes. Persistent agents retain identities and relevant states across rounds, with explicit interfaces defining information access, admissible actions, and decision schedules; linked records trace decisions, execution, and feedback.
In the current implementation, large language model (LLM) agents act as travelers choosing routes and departure times and as a manager periodically adjusting link charges within MATSim. A preliminary demonstration spans 100 simulated days in each of five configurations, with one run per configuration and a shared simulation seed. In the joint traveler--manager run, all 1,179,300 finalized target decisions satisfied action constraints; execution checks verified implementation of accepted choices and preservation of personal memories across days. Trajectories show stable route choices alongside departure-time adjustments and differences between controlled-trip and population-wide outcomes.
The platform supports experiments on individual adaptation, collective outcomes, and management interventions by varying information access, retained experience, and management feedback. Source code is available at this https URL 

---
# Machine Learning Beyond Statistical Consistency for Hybrid AI 

**Authors**: Jinsook Kim, Jinho Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01914)  

**Abstract**: We propose a new definition of machine learning, according to which a machine learns a function when it computes the function successfully, that is, when the value it returns for each input has a proper derivation in a formal system. We extend Gödel's Representability Theorem, which equates computation with derivation, to empirical targets in two ways: the theorem is extended to the computable reals, and the data are admitted into the system as empirical definitions that record what has been observed. We prove that a machine learns a true probability if and only if two conditions are met. The Truth Condition is that its estimate converges to the true probability, which is statistical consistency. The Assurance Condition is that the machine properly derives its own convergence inside the system. Learning thus lies beyond statistical consistency, for it requires a derivation of the convergence from the record. A global assumption about the data, such as the i.i.d. assumption, does not supply such a derivation, because it leads only to a circular one, as we show for PAC learning. On the other hand, the two conditions are met for the probability of a sentence over a finite ideal corpus. A real machine can approximate a true probability only if an ideal machine can learn it precisely, that is, only in a domain that meets the two conditions. Modern AI works in such a domain, and we offer this as a clue to its success. Modern AI is hybrid as well, in two senses: First, it performs two kinds of tasks, namely probabilistic ones such as predicting the next word and deductive ones such as proving and coding. Second, it combines two parts of different kinds, a statistical proposer and a deductive verifier, for proving and coding. Our framework explains both senses of hybridity in one formal system. 

---
# Systematic Evaluation of Robustness in Fine-tuned Models 

**Authors**: Angela John, Hamayoon Behmanush  

**Link**: [PDF](https://arxiv.org/pdf/2611.01900)  

**Abstract**: Our study systematically evaluates the robustness of fine-tuned pre-trained models across different datasets and attack types, aiming to provide insights for researchers in selecting pre-trained models for image classification tasks while prioritizing robustness against potential attacks. Our work explores the influence of fine-tuning pre-trained models with different datasets and attack types on their robustness and overview of the appropriate pre-trained model for image classification based on factors such as computation resources, memory usage, and accuracy on clean data. We fine-tune the models with weights. Evaluate the models on data before any attack and after different attack cases. The study utilizes publicly available image datasets such as MNIST, CIFAR10, CIFAR100, and SVHN to assess the robustness of pre-trained models. The results indicate that the robustness of fine-tuned models varies across different data distributions and sample sizes, emphasizing the need for tailored selection based on specific task requirements. The findings also highlight the varying performance of ResNet50, DenseNet, and MobileNet in defending against attacks on different datasets. Overall, this research provides valuable insights into the robustness of fine-tuned pre-trained models and their suitability for real-world applications in image classification. 

---
# Rethinking Model Aggregation for Federated Learning with Domain Skew: A Knowledge-Preserving Perspective 

**Authors**: Haotian Wu, Xu Ji, Qiheng Li, Li Xiong, Zibin Pan  

**Link**: [PDF](https://arxiv.org/pdf/2611.01893)  

**Abstract**: Domain skew challenges federated learning by requiring a shared model to accommodate distinct visual distributions. Domain-specific experts can capture these distributions separately, creating an opportunity to integrate their representations in a unified predictor. We propose Federated Knowledge-Preserving Learning (FKPL), a two-stage framework that combines domain-oriented expert training with prototype-based classifier calibration. First, a one-shot subspace-based clustering procedure groups clients for intra-cluster training. The resulting expert encoders are concatenated and frozen, preserving each learned feature mapping during global integration. Second, clients interpolate their concatenated features with class-matched domain prototypes and upload the resulting virtual features. The server trains a global linear classifier on these features, concentrating classifier optimization at the server after a single feature upload. FKPL achieves average accuracies of 73.01\%, 67.00\%, and 64.51\% on Office-Caltech-10, PACS, and a four-domain subset of DomainNet, respectively. On DomainNet, its cross-domain standard deviation is 7.50 percentage points, the lowest among the evaluated methods. With the same concatenated extractor, prototype-based calibration improves average accuracy over federated classifier training by 9.99 and 14.94 percentage points on Office-Caltech-10 and PACS. These results identify classifier calibration as an effective way to combine preserved domain representations into a global predictor. 

---
# Quantizing Recurrent Language Models Is Easier Than You Think 

**Authors**: Philipp Nazari, Neehal Tumma, Daniela Rus, T. Konstantin Rusch  

**Link**: [PDF](https://arxiv.org/pdf/2611.01864)  

**Abstract**: Linear recurrences are becoming an increasingly common building block in large language models, replacing the growing KV cache of softmax attention with a fixed-size recurrent state that enables efficient decoding. However, the quantization stack has largely been developed around softmax Transformers, whose numerical properties and inference state differ from those of recurrent models. Our goal is to understand these differences and study the design space of quantization methods for recurrent models. In our analysis, we differentiate between two targets. First, extra-state quantization concerns weights and activations outside the recurrent state, whose distributions are shaped indirectly by the sequence mixer during training. In controlled 1.3B parameter models, we find that decayed recurrences develop weaker residual outlier channels than softmax attention and are therefore easier to quantize without incoherence processing. Second, intra-state quantization concerns the recurrent state itself during decode, where repeated updates create a different numerical problem from KV cache quantization. We instantiate our findings as LoReS, a state quantization pipeline combining value-axis scale grouping to match output channel magnitude, decay-based mixed-precision to protect slowly decaying heads or channels, and stochastic rounding to limit error accumulation under repeated requantization. Across Qwen-3.5 9B, Nemotron-H 8B, and Kimi-Linear 48B-A3B, LoReS supports robust 8-bit state quantization and lower effective precision on shorter decode horizons. We implement LoReS in vLLM and demonstrate end-to-end decode speedups. 

---
# Privacy-Preserving ECG-HRV Emotion Classification for Mood Swing Detection using Federated Learning on Constrained Edge Devices 

**Authors**: Chidike Vincent, Muhammad Aslam Jarwar, Najam Ul Hasan, Laurence Hirsch, Jonathan Zasada-James, Hem Raj Dadra  

**Link**: [PDF](https://arxiv.org/pdf/2611.01853)  

**Abstract**: This paper presents a privacy-preserving mood monitoring framework that classifies emotional states from ECG-derived heart rate variability (HRV) features. The framework is evaluated on three public datasets comprising 74 participants and further validated using a wearable device in a pilot study. Leave-One-Participant-Out cross-validation (LOPO-CV) is used to assess generalization to unseen individuals. A series of experiments investigates the impact of temporal windowing, HRV feature selection, and class balancing on both four-class valence-arousal and binary valence classification. Results show that class balancing substantially improves minority-class recall, while Logistic Regression consistently outperforms five alternative classifiers and achieves competitive performance against HRV-based baselines. Pilot validation using a Polar H10 wearable achieves 75% binary valence accuracy on an unseen individual without calibration, demonstrating the feasibility of real-world deployment. To preserve privacy, the framework is also implemented using federated learning on a six-node Raspberry Pi cluster. Federated Logistic Regression achieves performance comparable to centralized training without sharing raw physiological data. These results demonstrate the potential of interpretable and privacy-preserving mood monitoring for continuous mental health assessment in clinical and community settings. 

---
# FADE: A Flow-Based Approach for Multichannel Tail Estimation 

**Authors**: Parmida Valiahdi, Niloofar Mehrnia, Mustafa Furkan Beker, Ubeydullah Erdemir, Ibrahim Hokelek, Sinem Coleri  

**Link**: [PDF](https://arxiv.org/pdf/2611.01849)  

**Abstract**: Ultra-Reliable Low-Latency Communication (URLLC) is vulnerable to rare low-received-power events, and the spatial-diversity benefits of multiple receivers can be significantly reduced when they experience simultaneous deep fading. Multivariate Extreme Value Theory (MEVT) provides a principled framework for characterizing such joint tail behavior. However, conventional MEVT methods often rely on restrictive parametric assumptions or discrete representations of the dependence structure, which may not fully capture complex multivariate tail behavior. This paper proposes Flow-based Angular Density Estimation (FADE), a flow-based approach for characterizing the dependence between extreme events across receivers within the MEVT framework. FADE transforms multivariate extremes into a standard extreme-value representation and uses a flow-based density estimator to learn the dependence structure from the transformed data without imposing a predefined parametric form. The proposed framework is evaluated using experimentally collected data from a UAV communication system, in which the UAV serves as the transmitter and communicates with two ground receivers. FADE provides closer agreement with the empirical joint CDF, achieving reductions of up to 22.4% in MAE compared with the state-of-the-art Logistic and Maximum Empirical Likelihood (MEL) methods, while also improving distributional agreement and reducing absolute bias. 

---
# OpenNWM: A Generalist Navigation World Model with Latent Action Pretraining 

**Authors**: Dujun Nie, Ruiqi Song, Zixuan Wang, Hanqing Wang, Yifan Han, Baiyong Ding, Wenzhao Zheng, Hangbin Wu, Long Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.01844)  

**Abstract**: Navigation world models predict future observations conditioned on navigation actions to support planning, but their generalization across real-world environments remains constrained by the limited coverage of action-labeled data. Action-free navigation videos provide diverse visual experience, yet lack the recorded actions required for action-conditioned training. We introduce OpenNWM, a generalist navigation world model with latent action pretraining. It learns universal latent actions from large-scale action-free videos through visual prediction pretraining, and further aligns them with physical motions via action-conditioned post-training. To support video pretraining, we curate NavAnywhere, a large-scale action-free navigation video dataset comprising 17.5 million visual observations from 15 heterogeneous sources. It captures navigation by ground robots, humans, and drones, with diverse viewpoints and motion patterns across five broad scene types: residential areas, public and commercial spaces, urban settings, parks, and natural landscapes. Experiments demonstrate improved visual prediction on in-domain and out-of-domain benchmarks, together with improved navigation performance. These results show that latent-action pretraining enables diverse, action-free videos to improve dynamics learning and generalization for open-world navigation. 

---
# MEX: The Musical Expectation Benchmark Suite 

**Authors**: Mathias Rose Bjare, Augustin Bouquillard, Giorgia Cantisani, Seung-Goo Kim, Stefan Lattner, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2611.01843)  

**Abstract**: We propose the Musical Expectation (MEX) benchmark suite for perceptually validating computational models of musical expectation, i.e., models that estimate music information dynamics (MID) measures of perceived musical expectations, such as surprisal or uncertainty from symbolic music or musical audio. MEX collects and extends tasks for evaluating such models previously proposed in music information retrieval and cognitive sciences using standard datasets, and evaluates whether: (1) proposed MID measures decrease with repetition due to in-context learning of the computational model, (2) peaks in such measures align with boundaries of music segments, (3) pitch surprisal and uncertainty estimated from audio are consistent with those modeled in event-based symbolic music, (4) proposed MID measures can predict electrophysiological responses to music. We stress that this benchmark suite does not provide an exhaustive computational characterization of music expectation. Rather, we envision this as an active, interdisciplinary initiative, with benchmarks evolving as our understanding of the underlying perceptual processes deepens. We provide open-access code this https URL to estimate surprisal and uncertainty from audio using baseline models, run the tasks on standard datasets, and include utilities for running the tasks with users' own data. 

---
# LegoEnv: Self-Evolving Agentic Environments via Modular Recombination 

**Authors**: Zhuosong Jiang, Chengxu Wu, Junqi Liu, Yujie Yi, Siqin Wang, Zhixuan Huang, Jinqian Jiang, Lijun Xiao, Jianxun Li, Yongyang Pan, Jieran Wang, Shiqi Shen, Fei Liu, Xinyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.01834)  

**Abstract**: LLM agents learn by interacting with environments, yet existing environments are designed for specific tasks and fail to generalize to new ones. Recent methods automate environment construction with LLMs. However, they generate each environment from scratch, neither reusing parts shared across environments nor carrying a fix from one environment to the rest. To address this, we propose a new paradigm of evolving environment atoms, which composes environments from reusable atoms (intent, rule, data and server) and evolves the atoms when an environment fails to give reliable feedback (e.g., an application lacks a tool the task requires). To instantiate this paradigm, we introduce LegoEnv, which builds 5,465 verified environments over 223 applications, and one round of evolution raises the share of solvable environments from 71.6% to 88.3%. On AutomationBench, SFT on these environments lifts Qwen3.6-27B from 16.0% to 32.0%, and RL with their verifiers lifts Qwen3.8-27B from 37.6% to 40.3%. Further analyses show steady gains with corpus size, transfer to unseen applications, and failures concentrated in a few shared root causes. 

---
# Control Barrier Functions for Safety-Aware Latent-Space Motion Planning 

**Authors**: Chandrakiran Ravula, Siddhanth Ramesh  

**Link**: [PDF](https://arxiv.org/pdf/2611.01814)  

**Abstract**: Latent-space path planning (LSPP) compresses the configuration space of a robotic manipulator into a compact, differentiable manifold learned by a variational autoencoder, enabling fast gradient-based planning without explicit inverse kinematics. Existing LSPP frameworks enforce obstacle avoidance with a binary collision classifier, whose saturated gradients near obstacle boundaries and lack of temporal structure cause the optimizer to stall and provide no forward-invariance guarantee. We replace the classifier with a neural control barrier function (CBF) learned directly in the latent space. The barrier is trained with a three-term objective that combines safe and unsafe sign constraints with a discrete-time decrease condition evaluated on transitions of an obstacle-blind nominal planner, and it is enforced at planning time by a closed-form, first-order correction that projects each goal-directed latent step back onto the CBF constraint. On a 7-DoF Franka Panda manipulator, the CBF planner achieves a 75.6\% success rate on a fixed-obstacle task, 12.0 percentage points above the best of 35 tuned classifier configurations, and 83.8\% on a multi-position task. A sweep of the decay rate $\alpha$ characterizes the safety--performance trade-off, with the best operating points in $\alpha \in [0.10, 0.20]$. In a fully general random-obstacle setting, success drops to 33.0\%, which we trace to covariate shift between the obstacle-blind training transitions and the states visited by the corrected planner. 

---
# MemRole: From Memories to Decisions for Long-Horizon Reasoning in Role-Playing Agents 

**Authors**: Mengting Li, Xilong Cheng, Zhihao Xiao, Zhenhua Xu, Jingyuan Zhou, Linfeng Li, Long Zhang, Borui Cai, Yunxiao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2611.01778)  

**Abstract**: LLM-based role-playing agents are fluent at imitating a character's style, yet struggle to make role-consistent decisions over long narrative horizons. Such decisions are shaped by experiences spanning many prior scenes, which are often partial and may point in different directions. Existing methods inject cross-scene memories as context, but do not reason about how past events shape a character's current motivations and behavior. We propose MemRole, a framework that makes this memory-to-decision process explicit. It introduces Long-Horizon Memory Thinking (LHMT), a reasoning schema that decomposes hidden reasoning into context understanding, memory activation, and response strategy planning, trained on a novel-derived dataset whose character-grounded, timeline-constrained memory pools limit each character to what it could plausibly know at the current narrative point. Beyond supervised fine-tuning, we propose Self-Refining On-Policy Self-Distillation (SR-OPSD): the model critiques its own rollouts against the reference under a data-driven rubric, and a critique-conditioned teacher view of the same model distills token-level guidance back into the policy. MemRole-32B improves over its backbone by 15.2 points on memory-grounded role-playing, performs on par with frontier LLMs such as GPT-5.4, and achieves the best character-motivation recognition accuracy on CroSS-MR. 

---
# Silent Reward and Environment Defects in EV Charging Simulation Benchmarks: A Diagnostic Checklist 

**Authors**: Nour-Eddine Moumni, Rachid Alaoui, Driss Kiouach  

**Link**: [PDF](https://arxiv.org/pdf/2611.01769)  

**Abstract**: Reinforcement-learning (RL) benchmarks built on microscopic traffic simulators such as SUMO are increasingly used to compare agent architectures for electric vehicle (EV) charging assignment. Unlike crashes, defects in the reward function, the environment's stochastic model, or the state encoding often fail silently: the simulation completes, produces plausible numbers, and the comparison is biased. We report five such defects, found by auditing a five-agent benchmark on OpenStreetMap road networks, stated generically: a shared random generator that desynchronises seed-paired comparisons, an ablation-specific reward clip applied unconditionally, a normalisation mismatch that zeroes a reward term, a per-timestep penalty that should fire once per event, and a state dimensionality documented but never wired into the network. We quantify three: a rejection counter inflated up to 125x across agents/cities, a reward term bounded at about $10^{-4}$ of its intended scale, and a 16.6%/16.3% greedy-action divergence from a state-encoding mismatch. Re-analysing a historical ablation, no A1-A2 effect survives family-wise correction; a 56-run campaign confirms the RNG defect's mechanism directly while showing its correction does not change that conclusion. We provide an eight-item checklist as a low-cost first-pass audit; in a later regeneration, three checks caught two new defects while three further process/artefact defects fell outside its boundary. 

---
# Evolving Memory, Learning to Meta-Control: Test-Time Adaptation for Experience-augmented LLM Agents 

**Authors**: Zeyuan Zhao, Yuanyuan Shi, Peng Yu, Lingyue Fu, Shao Zhang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2611.01747)  

**Abstract**: Self-evolving agents accumulate experience from task streams, but existing frameworks typically rely on fixed pipelines to retrieve and utilize this experience, neglecting when and how to deploy it. We introduce Online Meta-Control of Experience (OMCE), a framework that updates experience memory and learns a meta-control policy from task feedback during test-time adaptation. OMCE employs a lightweight contextual-bandit meta-controller that uses compact retrieval signals to select among four cognitive actions: DIRECT, EPISODE, GUIDE, and DISTILL. This routing mechanism operationalizes three coupled decisions: whether to invoke experience, which memory format to utilize, and when to consolidate new guidance. OMCE evolves online, leveraging task feedback to update the memory bank and refine the routing policy via contextual bandit updates. Experiments on four online adaptation benchmarks across four backbones show that OMCE improves first-attempt success by up to 20.5 percentage points over the strongest compared baseline, while reducing per-task token use by a factor of 5.7 to 62.5 relative to ACE with GPT-5.2. Controlled ablations further support the benefits of learned routing and continued routing policy adaptation. 

---
# Robust EEG-based Emotion Recognition from Coarse Trial-level Labels under Intra-trial Affective Variability 

**Authors**: Xiaocong Zeng, Buxuan Wang, Xin Yang, Pei Huang, Dongyang Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01731)  

**Abstract**: EEG-based emotion recognition has become an active research area in brain-computer interfaces. However, fine-grained affective annotation is costly, so most datasets provide only a global label for each long EEG trial. For practical emotion recognition, trials are commonly divided into short temporal windows, with the global label propagated to all windows for training. This practice treats trial-level emotion as temporally uniform, although affective responses may fluctuate within a trial. Consequently, windows that weakly reflect or deviate from the global emotional state may receive unreliable labels, posing a key challenge to robust EEG-based emotion recognition. To address this challenge, we propose VARY, a variability-aware robust learning framework for EEG-based emotion recognition from coarse trial-level labels. VARY incorporates a Two-stage Sample Selection (TSS) strategy to identify EEG windows with reliable emotional labels, and introduces MixAug, a data augmentation method tailored for EEG to improve model robustness under limited and unreliable supervision. We comprehensively evaluate VARY under the setting of label-noise learning on two public datasets, using three network architectures and multiple label-noise settings. Our method demonstrates superior performance, achieving a maximum accuracy gain of 11.54% on the DREAMER dataset and 5.39% on the SEED dataset compared with the best-performing baseline, highlighting the effectiveness of VARY in learning from unreliable trial-level labels under intra-trial affective variability. 

---
# MAC-Bench: Benchmarking Multi-Agent Collaboration via Interconnected Tasks 

**Authors**: Zerui Chen, Qinggang Zhang, Qichuan Liu, Zhishang Xiang, Zhimin Wei, Yunbo Tang, Zhuo Li, Jinsong Su, Zhihong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01728)  

**Abstract**: Large language model (LLM)-based multi-agent systems (MAS) tackle challenging tasks by dividing work among agents and coordinating their contributions toward shared objectives. Evaluating these systems therefore requires benchmarks that assess not only task-solving performance but also the ability to handle interdependent work. However, existing benchmarks often emphasize task difficulty without substantial collaboration requirements or evaluate only a limited range of predefined collaboration patterns. In this paper, we introduce MAC-Bench, a comprehensive benchmark for evaluating multi-agent collaboration through interconnected tasks. Rather than imposing predefined agent roles or communication protocols, MAC-Bench embeds collaboration requirements directly into task dependencies and shared-resource constraints. Specifically, we mine reusable collaboration structures from selected benchmark tasks at two levels. Local collaboration patterns capture recurring interactions among subtasks, while global task structures characterize how these interactions contribute to complete workflows. Through coarse-to-fine composition, these structures are recombined into task graphs combining different collaboration requirements. Tasks are instantiated in executable environments with automated verification of partial progress and complete task resolution. 

---
# CST: Lightweight Risk Triage with Source-Grounded Evidence for Web Discussions 

**Authors**: Zhihua Wang, Yifeng Gao, Yunhao Feng, Rui Pan, Baiheng Zhao, Zixing Chen, Nan Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.01710)  

**Abstract**: Web communities rely on moderators to catch harmful messages such as insults, hate speech, and scams, but no one can read every message. Existing tools give each message a risk score, which tells moderators where to look but not what to read or how to check it. Yet meaning depends on context: the same reply, "I agree.", can support or oppose removing someone from a group. Moderators therefore need source-grounded evidence, not just scores: a few key messages kept with their original wording and replies. We present CST, a lightweight triage pipeline that first decides where to look, then what to show. It scores risk by pairing character patterns, which capture exact wording, with sentence meaning from a fixed multilingual model, so it handles several languages and runs on an ordinary CPU. It then groups related messages and picks from each group in turn, so a small reading budget covers different parts of a discussion, and links every pick back to the original thread. Across six tasks from five datasets in several languages, CST ranks harmful messages more accurately than a fully fine-tuned multilingual model and, under the same budget, finds more of them than reading by score alone. In a user study, reviewers needed at least 27% less time with CST's grouped evidence while keeping decision accuracy above 94%. CST turns risk scores into evidence that moderators can read quickly and check in context. Code and dataset reconstruction resources will be released at this https URL. 

---
# SMART-AIMC: Sensitivity-Aware Mixed ADC Resolution with Fine-Grained Tunability for Analog In-Memory Computing 

**Authors**: Yung-Chin Chen, Ourania Stouraiti, Naveen Verma  

**Link**: [PDF](https://arxiv.org/pdf/2611.01707)  

**Abstract**: Analog In-Memory Computing (IMC) is a promising paradigm for accelerating GEMM-based AI workloads by executing computations directly within the memory array in the analog domain. While IMC delivers substantial energy and throughput improvements, it can suffer from significant task accuracy degradation due to analog-to-digital converter (ADC) quantization noise, which occurs at the GEMM output and is thus different from weight/activation quantization, which occurs at the GEMM input. Mixed ADC resolution allows IMC modules to choose between high- and low-resolution ADC modes with different energy costs, offering a compelling accuracy-energy trade-off. However, selecting ADC resolutions across IMC units while respecting hardware granularity constraints remains challenging. In this paper, we propose SMART-AIMC, a Sensitivity-aware Mixed ADC Resolution framework with fine-grained Tunability for Analog In-Memory Computing. SMART-AIMC correlates the algorithmic task loss to hardware ADC quantization noise in the form of an ADC Channel Sensitivity (ACS) metric. Using this metric, SMART-AIMC systematically, and in a disciplined manner, allocates ADC resources to critical computations. Furthermore, we introduce a sensitivity-based channel reordering scheme that overcomes the hardware granularity constraints with no runtime overhead and approaches the optimal ADC quantization possible for the model. Evaluated across diverse workloads, SMART-AIMC unlocks a superior accuracy-energy Pareto frontier, providing (1) higher task accuracy at matched energy budgets and (2) fine-grained tuning across efficiency targets, from aggressive energy savings to near-digital inference accuracy. At the latter operating point, SMART-AIMC achieves $6.99\times$ higher modeled system-level energy efficiency for linear projections than the INT8 digital baseline with negligible accuracy degradation. 

---
# MatMoE: Matryoshka Mixture-of-Experts with Dynamic Mixed-Precision Quantization for Efficient Inference 

**Authors**: Renjie Wei, Haochen Huang, Chuyu Qiu, Jinqi Wen, Dehao Xu, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.01704)  

**Abstract**: Mixture-of-Experts (MoE) architecture has been widely adopted in large language models (LLMs) since it reduces the computational cost through sparse expert activation. However, the massive parameter scale of MoE models hinders their deployment on resource-constrained edge devices. Although existing mixed-precision quantization reduces the model size, they are inherently static, fixing the bit-width of each expert across decoding steps. Such static schemes yield suboptimal performance as they overlook the dynamic variation in expert importance during decoding. Furthermore, they fail to adapt to the dynamic latency constraint driven by the fluctuating system utilization. To address these challenges, we propose MatMoE, a dynamic mixed-precision quantization framework for efficient MoE inference. MatMoE features: (1) a Matryoshka MoE PTQ algorithm that enables experts to operate flexibly at multiple precisions within a stringent memory budget; (2) a floor-and-compensation quantization method that eliminates redundant memory accesses in the Matryoshka representation; and (3) a dynamic bit-width allocation strategy that determines the precision for each expert based on their routing scores to satisfy the dynamic latency requirement. We implement our MatMoE system atop this http URL for both CPU and GPU platforms. Extensive evaluations across three modern MoE models and challenging benchmarks demonstrate that MatMoE achieves $4.0\% \sim 18.3\%$ higher accuracy than state-of-the-art static mixed-precision method. Compared to this http URL, MatMoE delivers $1.3\times \sim 3.7\times$ speedup on Raspberry Pi 5, and $1.2\times \sim 1.3\times$ speedup on NVIDIA RTX 4090 and Jetson AGX Orin. 

---
# Phase-wise MLLM Tuning for Multi-framework WebUI Code Generation 

**Authors**: Haoran Ma, Linxiao Li, Chenyue Wang, Haochen Sui, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.01692)  

**Abstract**: Multimodal large language models (MLLMs) achieve strong performance in translating WebUI screenshots into HTML/CSS. However, when they faced multiple frameworks (React/Vue/Angular), they often suffer from negative transfer that leads to compilation failures. These failures commonly stem from mixing framework specific syntax and violating framework constraints. This paper studies supervised fine-tuning for multi-framework WebUI code generation and proposes a phase-wise MLLM tuning method driven by compatibility and heterogeneity signals across frameworks. We first apply a normalization mapping that preserves each framework's native syntax while suppressing noise from variable naming and literal content. We then estimate cross framework heterogeneity and compatibility offline, and automatically derive training groups and their ordering by maximizing a phased objective. Finally, we perform phase-wise adapter tuning to reduce cross framework interference. Experiments on multiple benchmarks show that our proposed method improves both generation quality and compilation success rate compared with baselines. 

---
# Wave Physics as Representation: Tensor-Structured Neural Reconstruction for Ocean Acoustic Fields 

**Authors**: Yifan Sun, Panqi Chen, Weifeng Shi, Xiao Fu, Lei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.01680)  

**Abstract**: Reconstructing three-dimensional (3D) complex ocean acoustic fields from sparse measurements is challenging due to highly oscillatory wave structures and restricted sensing patterns/routes. Learning-based approaches fall into two categories. First, purely data-driven models can learn complex field patterns, but do not explicitly exploit wave physics that may enhance interpretability and reduce training costs. Second, physics-informed methods impose a specified physical equation as a reconstruction constraint, but often require environment-dependent parameters that are difficult to acquire. As a result, practical 3D acoustic field reconstruction that is both physically grounded and robust to uncertain environmental information remains largely unexplored. This work takes an alternative route: instead of solving a fully specified wave equation, it extracts a structured representation from wave physics and learns its components from data. From the Helmholtz equation and normal mode propagation, we derive a structured spatial-angular tensor representation, termed the S/c representation. Under a low-rank specialization, it takes the form of a rank-(L,L,1) block-term tensor decomposition, which connects structured sampling patterns/routes used in ocean sensing to sufficient conditions for field recovery. Building on this tensor representation, we propose TIDAL, a neural reconstructor that uses the S/c representation as an inductive bias for structured decoding and aligns information aggregation with the sampling geometry. The neural learner removes the need for known model parameters required by conventional physics-based recovery methods. Experiments across three structured sampling geometries show that TIDAL achieves the best reconstruction performance and fastest inference among baselines, with benefits from the physics-derived representation under limited data and model capacity. 

---
# A Hybrid Anomaly Detection Method for Multivariate Time Series 

**Authors**: Ali Babalhavaeji, Hui Song, Mahdi Jalili  

**Link**: [PDF](https://arxiv.org/pdf/2611.01661)  

**Abstract**: Multivariate time series are ubiquitous in monitoring smart systems such as power plants, smart houses, and servers. These systems use multiple sensors to collect the system's status over time. Those sensors represent the status of the whole system as one entity. Anomaly detection in such smart systems is of critical importance to prevent system failure and maintain functionality. But anomaly detection in such multivariate time series data is very challenging since this type of data includes both temporal and spatial information. Capturing both of these pieces of information is important to detect anomalies. In this paper, we propose a hybrid deep learning-based anomaly detection method, which captures both temporal and spatial information. We propose to take advantage of the feature extraction capabilities of 2d convolutional network in a variational autoencoder, alongside a prediction-based deep neural network of 2d convolutional networks and a gated recurrent unit network to capture both spatial and temporal information. Experiments on public data against state-of-the-art show the potential of our method. 

---
# A Hybrid Anomaly Detection Method for Electric Vehicle Identification from Smart Meter Recordings 

**Authors**: Ali Babalhavaeji, Hui Song, Mohammadreza Radmanesh, Mahdi Jalili  

**Link**: [PDF](https://arxiv.org/pdf/2611.01655)  

**Abstract**: In recent years, electric vehicles (EVs) have attracted much attention due to their potential to decarbonize the transport system. EV charging imposes challenges on power grids, such as load imbalance and voltage instability. Therefore, electricity distributors need to identify which users have an EV. Smart meter recordings can be effectively used for this purpose. The major issue associated with such data is the imbalance problem, where the majority of the data belongs to users without EVs. The existing methods for EV identification from smart meter recordings have framed it as a classification task. In this paper, we frame the problem as an unsupervised problem and propose a hybrid anomaly detection method that combines the power of prediction-based and reconstruction-based anomaly detection methods. The proposed model considers temporal and stochastic representation information of smart meter time series via prediction-based and reconstruction-based components, respectively. The performance of the proposed model is compared with state-of-the-art supervised and unsupervised methods. Its superiority is verified on real-world data. 

---
# dKV Forcing: Separating Generation and Memory Representations through a Differentiable KV Cache 

**Authors**: Zhiqi Li, Bo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.01648)  

**Abstract**: A causal video generator produces two key--value (KV) states for every segment: the KV it uses while denoising that segment, and the KV it writes into the cache for later segments to read. In a pretrained generator the two already differ substantially, later segments attend to them differently, and replacing the written KV with the generation KV degrades the continuation. We propose dKV Forcing (dKV denotes differentiable KV) to learn a dedicated memory representation from its future use: a writer-specific low-rank residual on the shared backbone separates memory writing from generation and is trained through the future reads of what it writes. Learning a write from its future use needs gradients through the cache, which the mutable cache in common use cannot provide. dKV Forcing replaces the training-time cache by persistent differentiable pages with local rematerialization and, in one temporal traversal, computes exactly the gradient that a two-pass reconstruction of the same objective computes. One traversal computes the full memory gradient at $7\%$ generator-update overhead over detached-write training, with $28\%$ less resident memory and $20.8\%$ shorter training cycles than the two-pass reconstruction. Over Self Gradient Forcing (SGF), the same objective trained with the same traversal, writer adaptation raises VBench-Long aesthetic quality by $6.6$ points at 60\,s and $8.5$ points at 240\,s with the highest subject and background consistency, and rollout dropout of the residual keeps most of this gain while restoring the motion of training without write gradients. 

---
# CREDO: Evidence-Conditioned Credit Assignment for Temporal Reasoning Agents 

**Authors**: Yue Leng, Shengtao Wen, Xiang Chen, Xinyi Zeng, Yu Tian, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01643)  

**Abstract**: Temporal Knowledge Graph Question Answering (TKGQA) requires language model agents to retrieve evidence, select relevant facts, and perform multi-step temporal reasoning. However, the trajectory-level outcome signal in GRPO does not explicitly distinguish the quality of intermediate decisions: downstream failures can penalize useful evidence acquisition, while identical rewards in all-failed groups eliminate the relative outcome signal despite differences in intermediate progress. We propose Credit Redistribution with Evidence-Driven Opportunities (CREDO), which uses intermediate evidence to recover local supervision, including in all-failed groups, and determine where outcome supervision acts. CREDO assigns positive stage-level credit through comparisons among rollouts of the same question and uses retained answer overlap as an opportunity proxy to shift part of the outcome-policy weight toward reasoning and answering. The framework preserves terminal rewards and trajectory-level advantages, while redistribution conserves total outcome-policy weight per trajectory. Following the public evaluation protocol of prior work, CREDO raises Overall accuracy from 78.0% to 82.4% and Multiple accuracy from 55.0% to 65.6% on MultiTQ relative to the strongest published baseline in our comparison. On TimelineKGQA, it raises cross-KG accuracy from 30.9% to 36.4% relative to our reproduced baseline via direct transfer to ICEWS Actor under matched training and evaluation settings. 

---
# Benchmarking Vision-Language Models for Urdu Nastaliq Handwriting Recognition 

**Authors**: Haseeb Yaqoob, Fatima Azfar, Emaan Arif Khan  

**Link**: [PDF](https://arxiv.org/pdf/2611.01635)  

**Abstract**: Handwritten Urdu is hard to recognize because the cursive Nastaliq script has no fixed baseline and no clear character boundaries. We benchmark six systems on the Urdu Nastaliq Handwritten Dataset (UNHD): Tesseract (default configuration), a script-specific recognizer trained on printed Urdu (UTRNet-Large), two open-source vision-language models (VLMs), and two commercial VLMs. All six systems are evaluated without fine-tuning on 2,465 single-line images (Part A), and the four VLMs receive the same fixed prompt (decoding uses temperature 0 where configurable, except for Claude Sonnet 5, whose decoding cannot be configured). Both commercial VLMs reach lower error than the printed-Urdu recognizer and the open-source VLMs. We do not compare against handwriting-trained recognizers, which would likely do better in-domain; we characterize zero-shot behavior, not state of the art. We also describe three distinct failure modes among the ranked systems. In one of them, a commercial model explicitly declines to transcribe a few very hard-to-read images instead of guessing. In a second study (Part B), VLMs correct the output of a separate OCR system using both the source image and the noisy text. Once separated from each model's own reading ability, the benefit is small on Tesseract's output (0.7% to 7.4%) and larger (17.9% and 21.9%) only for the two commercial correctors given UTRNet's text. It appears to depend on how legible the supplied text already is, and a weaker corrector can make a more legible text worse. 

---
# Zero-shot evaluation of open vision-language models as System 1 decision models on radiology tasks 

**Authors**: Bradley J Erickson, Sara Salehi  

**Link**: [PDF](https://arxiv.org/pdf/2611.01631)  

**Abstract**: Purpose: We compared two multimodal System 1 decision models, Clef 27B and d1 3B, with a traditional LLM (Qwen 3.8-27B) on seven radiology tasks covering text, radiographs, CT and MRI. Methods: Three models were compared: Clef, Qwen3.8-27B and d1. Three tasks were interpretation of text and 4 were detection of finding in images. The tasks were: 1. finding presence of finding in CT-RATE chest CT reports; 2. Presence of finding in PadChest chest radiograph reports; 3. the same 18 findings on CT-RATE chest CT but 3D volumes; 4. MR safety status of devices from AccessGUDID device records; 5. line and tube position on RANZCR CLiP chest radiographs; 6. motion-artifact severity on MR-ART T1-weighted brain MRI and 7. sequence identification on BraTS 2021 brain MRI. The primary endpoint was AUROC. There was no fine-tuning of the models on the tasks. Results: All models completed all tests without a failed request. All retrieved findings from reports almost perfectly (AUROC >0.95). Clef ranked higher than its Qwen backbone on CT-RATE volumes and RANZCR lower on MR-ART. The two did not differ on BraTS, GUDID or PadChest, and differed by 0.002 on CT-RATE reports. On BraTS, Clef correctly identified 90 of 100 FLAIR volumes while Qwen never chose it. d1 ranked lower than Clef on six of seven tasks. No model performed well on motion severity. Clef was faster per request than Qwen on tasks with many questions per input. d1 used 3 to 9 times less energy per request than Clef. Conclusions: System 1 models can perform important tasks in radiology with high accuracy, at lower compute cost than LLMs used in the usual way. When an LLM is constrained to a single answer token, its cost and performance is similar to System 1 models unless multiple questions are asked. A 3B System 1 model was close to the 27B models on report text but not on images. No model performed well on motion severity. 

---
# EgoMo3R: Joint Egocentric Motion and Scene Reconstruction 

**Authors**: Chaitanya Patel, Yusuke Kato, Rin Takahira, Kazuki Kozuka, Juan Carlos Niebles, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2611.01627)  

**Abstract**: Understanding holistic 3D human motion and the surrounding environment from egocentric video is fundamental for applications in AR/VR and robotics. Although these tasks are inherently coupled, existing methods treat them separately. In particular, egocentric motion reconstruction methods typically assume access to metric, gravity-aligned camera trajectories from specialized inertial sensors. Recovering such trajectories from monocular egocentric video remains a significant challenge due to rapid head movements and motion blur. We present EgoMo3R, a joint optimization framework that simultaneously reconstructs 3D human motion and scene geometry from casual egocentric video. It leverages the robustness of low-level visuo-geometric primitives, including monocular depth, optical flow, and point tracks, integrated with a diffusion-based human motion prior via first-order optimization. We propose an alternating optimization scheme that establishes a synergistic feedback loop: scene reconstruction provides physical grounding and trajectory conditioning for the motion prior, while the motion prior regularizes the camera trajectory and anchors the scene in a metric, gravity-aligned world frame. We evaluate EgoMo3R on the highly dynamic sequences of the EgoExo4D and Nymeria datasets and show that it outperforms specialized baselines in both 3D scene reconstruction and egocentric motion estimation. 

---
# Special and General Computational Aesthetics 

**Authors**: Ganqing Luo, Chuan Li, Liaoruxing Zhang, Xin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.01626)  

**Abstract**: Computational aesthetics is a field at the intersection of computer science, artificial intelligence, aesthetics, and other disciplines. As AI has developed, the field has formed relations of inclusion, intersection, or support with computational art, AI art, AI aesthetics, AI-generated content (AIGC), computational creativity, affective computing, and neuroaesthetics. Without distinguishing scope, objectives, and methods, it may be narrowed to image aesthetic quality assessment or expanded into an umbrella term for AI's involvement in art.
We analyze its internal structure and its relations with neighboring concepts, and propose a two-level scope: special computational aesthetics (SCA) and general computational aesthetics (GCA). SCA centers on aesthetic measurement and aesthetic process modeling. The former primarily corresponds to "Aesthetic Perception": computationally identifying and representing aesthetic attributes and predicting human aesthetic evaluations; the latter primarily to "Aesthetic Understanding": computationally describing, testing, and explaining how human aesthetic perception and judgment are formed. GCA extends this core to include "Aesthetic Generation," namely artistic generation guided by computational aesthetic models. Some computational aesthetic tasks can also use non-AI methods. Research in this scope using AI algorithms and models we define as AI aesthetics, which may involve the special core and the general extension. "Aesthetic Generation" intersects with AIGC, AI art, and computational creativity, but belongs to the general extension defined here only when computational aesthetic models substantively constrain, guide, or optimize artwork generation. We also draw on Feng Peng's discussion of the computability of beauty, aesthetic experience, and art as a theoretical reference for clarifying boundaries between computational aesthetics and related concepts. 

---
# Halal-NeSyNet: Clause-Scoped Neuro-Symbolic Cross-Attention for Automated Food Ingredient Auditing and Risk-Aware Halal Logistics Triage 

**Authors**: Mochamad Rizal Fauzan, Naila Nursyifa, Irgi Surya  

**Link**: [PDF](https://arxiv.org/pdf/2611.01612)  

**Abstract**: Verifying that a packaged food is Halal requires reading ingredient declarations and E-number additives before goods enter certified warehouses. Dual-origin additives such as E471, E422, E441 and E920 are Mushbooh (doubtful) unless a botanical, microbial or certified-Halal qualifier appears in the same ingredient clause, so a plant-based qualifier on one clause must not clear an unqualified additive in another. We call a failure to respect this scope cross-clause modifier confounding and build a benchmark of 3,199 items (277 Open Food Facts records and 2,922 template-generated clause-swap items, labelled by an explicit 50-code rule base rather than by a certification authority) with a 540-item counterfactual suite. We propose Halal-NeSyNet, which couples a DistilBERT encoder with a 16-dimensional rule-derived symbolic vector through cross-attention and gated fusion, trained with an asymmetric false-Halal risk loss. Over five seeds, text-only TF-IDF baselines reach 86.25-86.71 % accuracy with 13.42-21.73 % false-Halal leakage and 6.33-26.00 % counterfactual triad consistency, whereas Halal-NeSyNet reaches 97.04+/-0.34 % and 99.67+/-0.50 %. Gradient boosting given the same symbolic vector reaches 96.25+/-1.24 % (+0.59 points from Halal-NeSyNet, 95 % CI [-1.11, +2.28]), so most of the gain comes from clause-scoped symbolic features and not from the network, and the risk loss did not help. Holding out whole template families, text-only models fall to 61.91-78.35 %, Halal-NeSyNet to 90.87+/-11.76 % and logistic regression with the symbolic vector to 91.11+/-2.88 %, with far more spread for Halal-NeSyNet. A three-lane receiving simulation auto-clears 49.33 % of cargo at 0.76 % green-lane contamination. Because labels and features come from one rule engine and 91.3 % of the items are templated, the results measure fidelity to a rule policy, not religious certification. 

---
# DyRadio: Spatiotemporal Diffusion for Dynamic Radio Map Reconstruction 

**Authors**: Zhijie Fan, Zhifeng Zhao, Rongpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.01603)  

**Abstract**: Accurate radio map (RM) reconstruction is essential for environment-aware wireless networks. Sampling-based methods, which infer radio conditions from sparse measurements, are widely adopted because they do not require explicit knowledge of transmitter configurations. However, in dynamic environments, time-varying propagation conditions cause RMs to evolve, complicating reconstruction from temporally distributed sparse observations. Although cross-frame observations provide complementary spatial information, their direct aggregation can introduce temporal state mismatch. Furthermore, unknown measurement noise adds to the reconstruction difficulties. To address these challenges, we propose DyRadio, a framework that integrates spatiotemporal RM reconstruction with unknown noise estimation. Specifically, a propagation-prior-guided, noise-conditioned spatiotemporal diffusion model (DM) jointly reconstructs consecutive RMs by capturing cross-frame dependencies, while adapting to measurement noise levels. To estimate the noise level, an iterative predictive likelihood method alternates between DM-based blind-area prediction and noise-level updates. The estimated noise level then conditions reconstruction of the complete RM sequence. Experimental results demonstrate superior reconstruction accuracy and robustness against measurement noise, with peak signal-to-noise ratio (PSNR) improvements of 1.18-5.75 dB over the best-performing baseline across all evaluated settings. 

---
# SCOUT: Planning under Occlusion via Object-Centric World Model Rollouts 

**Authors**: Amir Hossain Raj, Xuesu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.01600)  

**Abstract**: Robots manipulating in cluttered spaces must often plan from partial views where foreground objects occlude the target, surrounding geometry, and usable free space. Reactive policies often miss these hidden constraints, while exhaustive search becomes computationally prohibitive in dense clutter. We present SCOUT, an occlusion-aware planning framework that bridges high-level deliberation with grounded physical rollouts. SCOUT leverages an object-centric scene representation to reconstruct hidden geometry and derive a relational planning state capturing blocker hierarchies and accessibility corridors. To navigate the vast action space in clutter, an LLM proposes structured action candidates which are then rigorously evaluated via Monte Carlo Tree Search (MCTS) using a learned action-conditioned world model. Evaluations on 'Reveal' and 'Placement' tasks show that SCOUT outperforms Vision-Language-Action (VLA) policies, embodied agents, and 3D world models, with significant gains in long-horizon tasks requiring reasoning over occluded geometry. We further validate the framework's robustness through real-world robotic manipulation in a cluttered shelf environment. 

---
# Autoencoder Based Deep Clustering of Point Clouds 

**Authors**: Nader Alfares, Guangmingmei Yang, David J. Miller, George Kesidis  

**Link**: [PDF](https://arxiv.org/pdf/2611.01594)  

**Abstract**: Motivated by a problem in Generative Specification-Producing AIs (SPAIs), we focus on a problem of deep clustering of three-dimensional shapes specified by point clouds. After reviewing clustering and deep clustering, we study the fully unsupervised pipeline in which a FoldingNet autoencoder is first trained for reconstruction and then K-means++ is then applied to the resulting codewords. We report clustering accuracy (ACC), normalized mutual information (NMI) and adjusted Rand index (ARI) on ModelNet10 and ModelNet40 datasets, as a function of the number of clusters K. We also provide t-SNE visualizations of the codewords, and discuss the choice of K. 

---
# Impossible Code, Barely Changed Score: A Replay Study of Judge-Based Evaluation 

**Authors**: Masoud Fetanat  

**Link**: [PDF](https://arxiv.org/pdf/2611.01572)  

**Abstract**: Language models are often used to judge generated code without running it. A useful evaluation should reveal substantial changes in the answers it scores. We examine this requirement by repeating an earlier evaluation on 421 coding problems, comparing Qwen-2.5-7B-Instruct with two versions trained using Group Relative Policy Optimization (GRPO). One receives rewards from execution tests (CE-trained), and the other from a strong language-model judge (SLJ-trained). The base model's syntax-failure rate remains stable, but the trained checkpoints produce invalid Python on 63%-67% of attempts. Despite this change, the strong judge's Pass@32 scores fall by only about 3 percentage points, from 96.67% to 93.95% for SLJ-trained. Within the later run, allowing more attempts makes the base and trained checkpoints look increasingly similar under the judge. With 128 attempts per question, the judge accepts at least one SLJ-trained answer on 96.44% of problems, compared with 28.27% under execution tests. We recommend reporting the share of answers that pass Python's syntax check alongside every judge-based score. This check requires no tests or code execution and flags invalid answers that the judge accepts, although it cannot establish functional correctness. Our findings show how a high judge-based score can hide substantial output damage and motivate checking an evaluation's sensitivity to change. Project page and code: this https URL 

---
# Search-Aware On-Policy Self-Distillation with Sibling Alignment for Generative Recommendation 

**Authors**: Rongfeng Guo, Zhuang Zhuang, Ji Dai, Peng Zhao, Shijie Li, Jie Chen, Teng Long, Vincent Tao Hu, Fei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2611.01568)  

**Abstract**: Generative recommendation retrieves items by autoregressively generating semantic IDs, typically via beam search. On-policy self-distillation (OPSD) enhances training supervision by transferring privileged self-teacher knowledge at student-generated states. However, beam-search retention depends on competition among prefixes: visited states differ in importance, and initially reliable decisions may approach the pruning boundary as training evolves. Moreover, conventional distribution matching couples target-token confidence with preferences among competing continuations, potentially weakening relational supervision when the self-teacher strongly favors the target. To address these challenges, we propose Search-Aware On-Policy Self-Distillation (SA-OPSD), a framework that transfers privileged self-teacher knowledge to the states visited during the student's own beam search. Specifically, we introduce Evolving-Boundary Sampling, which prioritizes search-critical states by considering both current pruning-boundary proximity and initial-reference retention-margin decline under a controlled distillation budget. Furthermore, we develop Confidence-Decoupled Branch Distillation, which separates target-probability calibration from informative sibling relations to better preserve the teacher's preferences among competing branches. Experiments on Industrial, Office, and Instruments demonstrate strong performance for SA-OPSD. 

---
# Principled Human-AI Collaboration in Agentic Workflows 

**Authors**: Young Hyun Cho, Will Wei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.01566)  

**Abstract**: Organizations adopting AI face the challenge of deciding which outputs can be released without human review while meeting the quality standards of human experts. We develop a statistical framework for delegating tasks between experts and black-box generate-verify workflows, in which AI-generated answers are verified and revised over multiple rounds. Before deployment, human experts evaluate selected outputs, and their judgments are used to calibrate the verifier threshold and number of attempts. Our method selects the policy with the highest estimated automation rate among statistically approved candidates, without modifying the generator or verifier. We establish finite-sample control of the error rate among released outputs, characterize the expert evaluations sufficient to attain a target automation rate, and derive a minimax lower bound establishing rate optimality in the quality margin and required confidence. In a MultiHiertt financial question-answering case study, average held-out automation increases from 2.50% to 24.17% as expert evaluations accumulate, while released-output error rates remain around 1.20%. In contrast, heuristics that release answers after consistently high verifier scores either violate the 5% error tolerance or automate at most 1.30% of tasks. Synthetic experiments further examine these theoretical predictions under controlled settings. 

---
# Meta-memorization and memorization scaling laws in transformers 

**Authors**: Alex Nguyen, Kenneth A. Norman, Gautam Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2611.01564)  

**Abstract**: Larger transformers can memorize more data, with a capacity (in bits) that scales roughly linearly with the number of parameters. Counterintuitively, larger models also memorize faster, requiring fewer repetitions of a sequence during training before it is memorized. We reproduce both memorization scaling laws in transformers trained on random binary sequences, showing that rapid memorization can arise even when data contain no semantic or compositional structure. Rapid memorization is due to a meta-memorization effect, where models trained to memorize a large set of random sequences learn to memorize novel sequences more quickly. The rate $r(t)$ at which models memorize grows with training time $t$ as a power-law $r(t) \sim t^{\beta}$ with a fixed exponent $\beta$ but an increasing upper cutoff across model scales. Meta-memorization enables even one-layer transformers to few-shot memorize a 100-bit sequence. This effect is partly explained by the formation of structured attention heads that convey position-invariant sequence information to MLPs where memorized sequences are stored. Meta-memorization is recapitulated in a simplified model of a dense associative memory where retrieval is constrained by an attentional bottleneck. Our results suggest transformers, through pre-training, acquire reusable mechanisms that enable highly efficient storage and retrieval of memories. 

---
# Finite-Time Online Sample Complexity for Conditional Diffusion Models 

**Authors**: Mudit Gaur, Prashant Trivedi, Sasidhar Kunapuli, Amrit Singh Bedi, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2611.01562)  

**Abstract**: We study the finite-time training and sampling error of conditional diffusion models trained by online stochastic-gradient denoising score matching. Rather than assuming access to an exact empirical or population risk minimizer, we analyze the score network produced after a finite number of stochastic-gradient updates under an explicit localized Polyak--Łojasiewicz (PL) condition, stochastic-gradient regularity, and a trajectory-containment assumption. Our analysis separates the network approximation complexity from the online training-sample budget and propagates both errors through the reverse diffusion process. Under anisotropic smoothness, a joint intrinsic-dimensional structure for the data and conditioning variables, sub-exponential tails, and explicit approximation-class compatibility, we obtain a conditional score approximation error of order \[ \widetilde{\mathcal O}\!\left(M^{-\eta_{\mathrm{app}}}\right), \qquad \eta_{\mathrm{app}} = \frac{2\alpha_x} {2\alpha_x+d_x'+(\alpha_x/\alpha_y)d_y'}, \] where \(M\) denotes the approximation complexity, \(d_x'\) and \(d_y'\) are the intrinsic response and conditioning dimensions, and \(\alpha_x,\alpha_y\) are their corresponding smoothness exponents. For \(N=RB\) online denoising samples used by SGD, the finite-time optimization error is controlled explicitly in terms of the PL, smoothness, gradient-noise, and initialization constants. Combining these ingredients with reverse-process stability yields a total-variation bound that decomposes into terminal-initialization, time-discretization, approximation/localization, and finite-time optimization errors. 

---
# Generative Modeling with Continuous Flows: Sample Complexity of Flow Matching 

**Authors**: Mudit Gaur, Prashant Trivedi, Shuchin Aeron, Amrit Singh Bedi, George K. Atia, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2611.01559)  

**Abstract**: Flow matching has emerged as a promising alternative to diffusion-based generative modeling, offering a continuous-time ordinary differential equation formulation with fast sampling and a simple regression-based training objective. Despite its empirical success, some theoretical analyses idealize the learning procedure by assuming that the velocity field is obtained by exact empirical risk minimization. This assumption does not reflect the practical finite-dataset setting, where the empirical flow-matching loss is only approximately minimized by a finite number of stochastic-gradient updates. We develop an empirical-risk-minimizer-free analysis for flow matching that explicitly accounts for both finite-sample statistical error and finite-time empirical optimization error. Our analysis considers a finite dataset of \(N\) flow-matching samples and a learned parameter \(\theta_T\) obtained after \(T\) stochastic gradient updates on the empirical objective. Under regularity assumptions on the interpolation path, sub-exponential source and target distributions, approximation by a bounded-parameter smooth neural velocity class, empirical-loss optimization, and uniform flow stability, we prove a guarantee that holds with high probability over the finite training dataset and in expectation over the stochastic-gradient randomness. The resulting bound separates approximation error, statistical error from the finite dataset, tail error from unbounded source and target distributions, and empirical optimization error from approximate minimization of the finite-sample objective. 

---
# TopoDPO: Dense Preference Alignment for Structure-Sensitive Aerial Referring Segmentation 

**Authors**: Huaiyuan Peng, Jisheng Dang, Wencan Zhang, Jingwen Zhao, Hong Peng, Wenjun Ma, Bin Hu, Qi Tian, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2611.01551)  

**Abstract**: Preference optimization for multimodal segmentation faces a granularity mismatch: dense masks contain spatially heterogeneous preference signals, whereas image-level objectives assign one shared nonlinear weight to all spatial margins. We introduce TopoDPO, a topology-aware dense preference alignment framework for structure-sensitive aerial referring segmentation. Topo-Prior mines informative preference pairs using connected-component discrepancy and macro-corner recall, while the dense objective applies the preference penalty to each local margin before spatial aggregation against a frozen supervised reference. On Topo-RS-Hard, a 4590-instance structural stress set, TopoDPO improves intersection over union (IoU) from 50.91% to 51.24% over image-level direct preference optimization (DPO) and reduces Betti-0 error from 1.34 to 0.90; paired image-cluster bootstrap intervals against image-level DPO exclude zero for all five reported metrics. Matched supervised controls and a complete factorial analysis separate the effects of topology-aware pair mining, localized correction, and dense reference-relative preference weighting. An unfiltered stratification over all 39,244 validation records further shows the clearest structural gains on samples with moderate or severe pre-alignment component-count errors. The code is publicly available at: this https URL. 

---
# Revisiting LLM Homogeneity Through a Multidimensional Lens 

**Authors**: Ziliang Qiu, Renfen Hu, Yi Ding, Ted Underwood  

**Link**: [PDF](https://arxiv.org/pdf/2611.01549)  

**Abstract**: Concerns about LLM homogeneity are typically quantified with embedding similarity, which reduces two texts to a single score. We revisit homogeneity with 25 interpretable metrics spanning five dimensions: semantic, lexical, syntactic, structural, and stylistic & presentational. A human study shows that perceived similarity depends on both content (what responses say) and form (how they say it), yet embedding similarity correlates less with human meaning judgments than sentence-level matching ($\rho=0.417$ vs. $0.606$) and not significantly with form judgments ($\rho=0.140$). Across 18 contemporary LLMs, models say similar things in different words: they share global semantics and broad linguistic profiles but remain distinct in lexical characteristics, structural patterns, and interactional conventions. Across releases from 2024 to 2026, models become more like themselves but not more like one another: among significantly changing metrics, intra-model distances decrease for 17 of 22, and inter-model distances increase for 14 of 21. Across two post-training pipelines, model-specific conventions emerge during SFT and are selectively reshaped by preference optimization. These results suggest that homogeneity is multidimensional and evolving, and that embedding similarity alone can overstate cross-model homogeneity while concealing diverging intra- and inter-model trends. 

---
# GSExplorer: Interactive World Generation with a Persistent 3D Gaussian World State 

**Authors**: Sicheng Yu, Liu Liu, Xinjie Wang, Xinjiang Wang, Taojun Ding, Wenhao Hu, Beizhen Zhao, Puxin Yan, Zhizhong Su, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01543)  

**Abstract**: Interactive world generation requires both accurate local control and stable scene consistency over long rollouts. Existing methods improve local generation through camera or geometry conditioning and extend historical context with memory mechanisms. However, long-horizon exploration remains challenging as target-view guidance becomes incomplete and inconsistencies accumulate across repeated scene updates. We present \textbf{GSExplorer}, a generation--reconstruction framework built around a persistent 3D Gaussian world state. At each step, the current 3DGS scene provides multimodal rendering conditions for a condition-driven video generator, while camera geometry is injected through a separate Relative Ray Encoding branch. Generated videos are then fed back to a SLAM-based reconstruction backend, where inconsistency filtering prevents conflicting content from corrupting the scene and cross-chunk bundle adjustment improves spatial alignment across generation steps. We further design a data-construction pipeline that reproduces realistic 3DGS degradation during scene extrapolation, yielding 293K training pairs of degraded multimodal renderings and real target videos. GSExplorer achieves the best overall performance among the compared methods on both single-step and three-step WorldScore evaluation, with strong camera control and consistency across both settings, including a Camera Control score of 99.07 in single-step extrapolation. On 481-frame leave-and-return rollouts, it also shows strong revisit consistency, demonstrating the value of maintaining a reliable 3D world state for continued exploration. 

---
# AI-Aided and AI-Oriented Integrated Sensing, Communication, and Computation: A Survey 

**Authors**: Yaya Etiabi, Wessam Ajib, El Mehdi Amhoud, Halima Elbiaze  

**Link**: [PDF](https://arxiv.org/pdf/2611.01521)  

**Abstract**: The evolution from 5G to 6G networks marks a transition toward the "intelligence of everything." While 5G enables enhanced connectivity, 6G aspires to embed intelligence directly into communication infrastructures, creating systems that are adaptive, context aware, and capable of autonomous decision making. Realizing such systems depends on harmonizing three fundamental functions that compete for shared network resources and often pursue conflicting goals: (i) sensing for environmental and contextual information acquisition, (ii) communication for reliable and timely information exchange, and (iii) computation for information processing and decision making. This harmonization necessitates shifting from independent optimization to their joint design as a unified system, a paradigm known as Integrated Sensing, Communication, and Computation (ISCC). In parallel, AI has emerged both as an enabling tool for ISCC optimization and as a workload whose requirements influence ISCC system design. This survey comprehensively examines ISCC from this dual perspective by distinguishing between AI-aided ISCC, in which learning-based methods support the joint design and optimization of sensing, communication, and computation, and AI-oriented ISCC, in which these functions are jointly configured to support learning, inference, and distributed intelligence. The survey further examines the co-design of AI and ISCC, where AI requirements and ISCC constraints jointly shape system and model decisions. It further provides a comparative synthesis of the resulting design approaches, system architectures, performance metrics, and evaluation practices, relates these developments to application-specific requirements across different domains, and identifies open research problems toward scalable, adaptive, and trustworthy AI-ISCC systems. 

---
# Improved Sample Complexity Analysis For First Order Bilevel Reinforcement Learning 

**Authors**: Mudit Gaur, Naman Saxena, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2611.01500)  

**Abstract**: Bilevel reinforcement learning optimizes an upper-level objective through the policy response to a lower-level reinforcement learning problem. Such problems arise when learning a reward model while adapting a policy to that reward. However, a gap remains between sample-complexity guarantees and practical actor--critic training. Tabular analyses restrict applicability to finite spaces, while continuous-state analyses often impose a uniform Polyak--Łojasiewicz (PL) condition. We show that this condition need not hold even for smooth positive policies with exactly representable values. We develop a single-loop, fully first-order method for continuous states and finite actions with specified multilayer neural actors and a persistently trained nonlinear critic built from fixed features. Under explicit initialization, regularity, feature-excitation, and value-realizability conditions, fixed reference-KL regularization yields local curvature, and we prove containment of the coupled iterates. Shared trajectories control sampling noise, while paired critic analysis controls normalized critic-error differences as the response perturbation shrinks. A joint descent and tracking argument establishes an improved $\widetilde{\mathcal O}(\varepsilon^{-2})$ sample complexity for $O(\varepsilon)$ expected squared gradient norm of the upper objective induced by the selected regularized policy response, including critic training. 

---
# FedUDG: Federated Domain Generalization with Unlabeled Clients 

**Authors**: Junyu Ren, Wensheng Gan, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.01498)  

**Abstract**: Federated domain generalization (FDG) aims to learn models that generalize reliably to out-of-distribution (OOD) environments without centralizing client's data. However, existing FDG methods assume that all source clients have labeled data, limiting their ability to leverage the abundant unlabeled client data available in practice. To address this limitation, we propose FedUDG. At the federation level, FedUDG utilizes a parameter-functional differentiation strategy that maintains a semantic anchor updated solely using ground-truth labels. The remaining parameters absorb knowledge from unlabeled domains under bounded aggregation, preventing domain-biased noise from eroding the class semantics. Within each unlabeled client, FedUDG utilizes a student-teacher framework with two complementary objectives: (1) semantic self-training on high-confidence samples, using the functional differentiation between the semantic anchor and classifier to constrain the model to learn semantically pure and cross-domain stable features; and (2) label-free representation distillation on all unlabeled samples to stabilize learning under appearance variations. Together, these objectives guide the model toward a transferable feature space, while mitigating overfitting to domain-specific characteristics. Furthermore, to address the lack of systematic evaluation of FDG under formal privacy constraints, we develop a sample-level differentially private extension, DP-FedUDG, to quantify the trade-off between privacy and OOD generalization. Across four evaluation protocols, FedUDG consistently outperforms state-of-the-art baselines. The code is available at this https URL 

---
# Sweep Less, Steer More: Using Activation Steering Geometry for Reduced Hyperparameter Search 

**Authors**: Nishkal Hundia, Swastik Agrawal, Navita Goyal, Sarah Wiegreffe  

**Link**: [PDF](https://arxiv.org/pdf/2611.01496)  

**Abstract**: Representation steering methods like difference-in-means (DIM) offer a lightweight approach to controlling LLM behavior, but their effectiveness varies widely across concepts, datasets, and layers, making principled deployment costly without exhaustive hyperparameter sweeps. We build on the discriminability index $d'$, a compute-cheap geometric measure of how well contrastive training activations separate along the steering direction. Across five behaviors on 3 models, we demonstrate uses beyond its established role as a dataset-level steerability predictor. First, in the multiple-choice question answering (MCQA) setting, we show that $d'$ reliably identifies performant layers at which to steer, eliminating the need for a layer-wise sweep. Second, we extend this analysis to open-ended multi-token generation, and find that $d'$ remains successful for layer selection across different settings. Lastly, we test whether the position of a generated token's representation on the DIM line predicts steering success on that individual example. In the MCQA setting, this works for high-$d'$ datasets but is inconsistent across behaviors, models, and training-token choices in open-ended generation. This highlights that methods developed and conclusions drawn in single-token MCQA settings do not necessarily transfer to open-ended generation. 

---
# Spectral-Convolutional Factorized Transformer for Joint Multipath Parameter Estimation: Applications to mmWave Positioning and Radio SLAM 

**Authors**: Yaya Etiabi, Wessam Ajib, Halima Elbiaze  

**Link**: [PDF](https://arxiv.org/pdf/2611.01491)  

**Abstract**: Future 6G wireless networks are expected to evolve toward AI-native systems that support intelligent, environment-aware services, including positioning and mapping. Current research has already integrated AI into several aspects of communication, particularly wireless channel estimation. However, most learning-based channel estimators focus on communication-oriented objectives rather than explicitly estimating multipath parameters such as delay, angle of arrival (AoA), and angle of departure (AoD). These parameters provide essential geometric information for downstream tasks including positioning and radio simultaneous localization and mapping (SLAM). In this work, we propose a Spectral-Convolutional Factorized Transformer (SCFT) for joint multipath component (MPC) estimation from noisy complex channel estimates and physical beam-pair descriptors. SCFT combines local spectral convolution, permutation-equivariant beam-set attention, and a query-based Transformer decoder to detect and classify line-of-sight (LoS) and non-line-of-sight (NLoS) paths and estimate their observed delays, AoAs, and AoDs. Experiments on realistic ray-tracing data show that SCFT reduces the generalized optimal sub-pattern assignment (GOSPA) error by 13.4% compared with ResNet, the strongest competing baseline. We further demonstrate this advantage in independent snapshot positioning and radio SLAM, where SCFT reduces position RMSE by 18.5% and 65.0%, respectively, relative to ResNet. These findings highlight the importance of explicitly recovering multipath geometry, as improvements in MPC estimation can translate into substantial gains in downstream tasks. 

---
# See, Think, Use: Diagnosing Perception and Reasoning Failures in VLMs 

**Authors**: Jiacheng Ding, Yanliang Qi, Cong Guo, Peihua Mai  

**Link**: [PDF](https://arxiv.org/pdf/2611.01467)  

**Abstract**: We present See, Think, Use, a protocol that traces each answer of a vision-language model to perception, reasoning or the use of perception. The diagnosis shows a developer which stage to improve and directs test-time compute to the stage that fails. Benchmarks score the final answer, and an answer that rests on the image earns the same credit as a guess or a memorized prior. The protocol asks the model two extra questions per item and needs no access to its weights. A localization probe asks for a box around every instance and measures seeing, and a text-replacement probe answers from a description and measures thinking. Use is the rate at which the answer follows the model's own perception across counterfactual image pairs. The rate needs no answer labels, because the question and its prior are the same on both images. On two model families and six datasets the protocol shows that the knob that helps follows the task family. Answers follow the model's perception except on images that contradict a common prior, and asking the model to localize first repairs this failure when its boxes are right. Allocating compute by the measured stages comes close to the best uniform policy at lower cost and beats a random allocation of the same compute wherever a knob raises the short stage. We will release the code, the prompts and the per-question outputs. 

---
# DIMWIT: Directly Invoking Models Without Intensive Tuning 

**Authors**: Saarth Deshpande, Neeraja J. Yadwadkar, Venkat Arun  

**Link**: [PDF](https://arxiv.org/pdf/2611.01463)  

**Abstract**: Resource managers typically encode what to do in a task-specific policy---a heuristic, an optimization procedure, or a model trained for the task---that must be developed before use and may need revision as the workload or system changes. We propose bypassing policy development and instead asking pretrained models to make resource-management decisions directly, given the current system state, desired outcome, and available choices. We evaluate this approach on microservice autoscaling, DRAM sizing, and video encoding, using a frontier LLM and two small decision models. On two microservice applications, a small decision model meets latency targets while using within 10% of the replicas allocated by policy-based autoscalers and within 13% of the frontier LLM. In DRAM sizing, a small decision model reduces DRAM usage by up to 27.5% relative to static provisioning while maintaining at least 95% of baseline performance. In video encoding, it completes encodes up to 4.3$\times$ faster than the frontier LLM, with an imperceptible 1.1-point decrease in Video Multi-Method Assessment Fusion (VMAF), a metric of perceptual video quality. Making individual resource-management decisions may be considerably simpler than developing a general strategy for making them. 

---
# An auditable agentic pipeline for variant-to-gene prioritisation, validated against orthogonal functional evidence 

**Authors**: Shakuntala Baichoo, Sohail, Sarkar, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01461)  

**Abstract**: Assigning GWAS variants to causal genes requires evidence from multiple resources. We present V2G, an agentic pipeline that records evidence retrieval and returns ranked genes alongside the retrieved evidence. We evaluate whether language-model reranking improves gene prioritisation and whether its gains depend on the retrieved evidence or recall of known associations. On 135 held-out loci where candidate retrieval succeeded and Open Targets L2G predicts, reranking by a language model improves Precision@1 over the evidence ordering it receives, 0.793 against 0.711 (+0.081, 95% CI +0.038 to +0.129), and over nearest-gene assignment (+0.178); no difference from Open Targets L2G or from training-membership baselines was detected. Given only the trait and the candidate gene symbols, in shuffled order, the same model scores 0.864 on these loci (three seeds, 0.859 - 0.867) and names the reference gene at 0.594 of the loci where the pipeline never retrieved it; with each gene's symbol moved to another gene's evidence row it follows the top-scored row and scores 0.284. The gain over the evidence ordering is recall of known genes, and the evidence as presented anchors the model to the ordering rather than informing it. Across the 296 held-out loci (five seeds each) the model keeps the given top candidate at 86% of decisions; at naturally retrieved loci its departures select a reference gene in 85% of departure decisions (31 of 37 loci), and none displaced one. Its calls carry trait-matched mouse mutant or Mendelian phenotype annotations more often than other candidates at the same locus. Train-test overlap is reported throughout. The study includes a frozen benchmark, cached predictions and reproduction code. 

---
# Can Large Language Model Agents Learn by Watching Tutorial Videos? 

**Authors**: Haoyu Zhao, Zafeirios Fountas, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01460)  

**Abstract**: Humans continuously acquire knowledge from carefully crafted materials such as books and tutorial videos, carrying that understanding into future tasks. Millions of such instructional videos are uploaded online every day, yet LLM agents remain unable to exploit this ever-expanding wealth of human teaching. Their capabilities are frozen after training, while parameter updates at scale remain prohibitively expensive. This raises the central question: can LLM agents learn from this vast instructional resource without parameter updates? We introduce Tutorial Agentic Learning (TAL), a nonparametric continual-learning paradigm that consolidates tutorial videos into a persistent, structured tutorialbook memory through a two-stage adversarial process that harvests hard, video-grounded questions and uses them to refine the memory. At inference time, TAL retrieves through a Table-of-Contents (TOC) tree, tying retrieval directly to the model's reasoning rather than to embedding similarity. To study this setting, we introduce VidLearn, a multimodal QA benchmark built from 72 chess tutorial videos spanning 12 openings. On VidLearn, TAL outperforms all evaluated baselines and improves monotonically as more videos are consumed, achieving up to 113% and 91% relative gains over the corresponding vanilla MLLMs on the training and test splits, respectively. In actual gameplay, TAL reaches a stronger opening-phase position than its vanilla counterpart in 69% of head-to-head games. Evaluation in a second domain further demonstrates the generality of TAL. Together, these results highlight instructional videos as a scalable resource for parameter-update-free agent continual learning. 

---
# ERIA: An Instruction-Optimized Agent for Clinical EEG Reporting 

**Authors**: Ahmed Sameh, Gayal Kuruppu, Neeraj Wagh, Bridger Brundy, Yogatheesan Varatharajah  

**Link**: [PDF](https://arxiv.org/pdf/2611.01394)  

**Abstract**: Clinical EEG interpretation requires integrating technical quality, background and behavioral states, interictal and ictal abnormalities, and overall clinical context across long recordings. We introduce ERIA (EEG Reporting and Instruction-Optimized Agent), a locally deployable framework in which a single 27B language-model controller analyzes whole-recording EEGs by orchestrating specialized parametric and signal-processing tools. ERIA combines learned tools for background abnormality, sleep-wake state, seizure-like activity, and interictal epileptiform discharges with deterministic spectral, amplitude, symmetry, montage, and signal-quality analyses. Rather than fine-tuning the controller or using multi-agent orchestration, ERIA keeps the controller weights, specialist tools, and safety constraints fixed and optimizes only its natural-language analysis policy with GEPA (Genetic-Pareto), using feedback from expert-written EEG reports. GEPA improves physician-aligned whole-recording report quality over the raw policy, and the optimized policy transfers to external EEG benchmarks. On CerebraGloss-Bench and BrainBench Foundational Analysis, ERIA remains competitive with the models evaluated in the original benchmarks. These results show that specialist tools and language-level policy optimization can extend the clinical and general EEG capabilities of a single locally executable model while preserving local inference and traceable tool use. 

---
# Vocabulary Aligned Explanations for Computer Vision Models 

**Authors**: Erfan Mirhaji, Nikhil Krishnaswamy, Jill Zarestky, Lisa Mason, Nathaniel Blanchard, Sarath Sreedharan  

**Link**: [PDF](https://arxiv.org/pdf/2611.01391)  

**Abstract**: Optimizing vision models for accuracy does not force the model to learn and utilize human-understandable and domain-specific concepts and rules. We propose explaining image classifications through domain-specific concepts, visual evidence, and expert knowledge expressed as logical rules. We introduce FLUENT (Framework for Localized, Uncertainty-Aware Explanations in Normative Terminology), a post-hoc framework that ranks expert-authored rules combining jointly required, alternative, and absent concepts. To account for prediction errors, FLUENT combines predicted labels and concept states with empirically estimated likelihoods and concept priors. It scores each rule by calculating support over satisfying assignments to the concepts mentioned in that rule, then normalizes these scores to express relative confidence among candidate explanations. The selected rule is illustrated through occlusion-based highlights for positive concept mentions in the query image and localized examples in retrieved reference images for negative mentions. These illustrations are intended to show end users unfamiliar with the domain what the concepts named in an explanation look like. FLUENT combines domain knowledge and model accuracy, and delivers visual concept highlights and explicit rule confidence to explain an image in the vocabulary of an application domain. 

---
# Toward Executable Multi-framework Front-end Code Generation with Self-Correction 

**Authors**: Linxiao Li, Haoran Ma, Chenyue Wang, Jiaye Lin, Haochen Sui, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.01383)  

**Abstract**: Generating codes from UI screenshots has recently benefited from MLLMs, yet existing methods centered on HTML targets and transfer poorly to multi-frameworks such as React, Vue, and Angular. However, unlike HTML, which does not entail compilation issues, generating executable code in multi-framework settings is more challenging due to framework-specific differences in syntax and failure modes, as well as the need to maintain global consistency across files. To address this problem, we study executable multi-framework front-end code generation and propose MESCoder, a tree-structured self-corrective generation framework. Our method first extracts a structural scaffold from the screenshot and then constructs sub-trees and assembly relations step by step, transforming large project generation into a sequence of localized decisions. On top of this scaffold, we introduce a unified Project Agent that continuously revises its own partial project under runtime sandbox feedback and dynamically chooses between local repair and cluster level repair. For training, we adopt a two-stage strategy that first initializes the policy offline to learn basic multi-framework generation and repair priors, and then refines it with online reinforcement learning to optimize long horizon executability and final page quality. The resulting framework models multi-framework executability as a conditioned project state transition process and improves the stability and quality of complex front end generation through self correction. Experiments show that our method improves executability, while narrowing the visual-fidelity gap to HTML-based generation. Code is available at this https URL. 

---
# ProofPatch: Reviewing Lean Statements and Keeping the Reviews Valid as Formalizations Evolve 

**Authors**: Xujun Che, Depeng Xu, Jian Xiang, Taufiqua Khan  

**Link**: [PDF](https://arxiv.org/pdf/2611.01357)  

**Abstract**: The kernel of the Lean proof assistant checks that a proof establishes its statement. Whether the statement says what was intended is left to human reviewers. This is the old problem of validating a specification, and it grows more pressing as language models write formal statements in bulk and agents edit formal developments continuously. We present ProofPatch, a tool that helps reviewers judge Lean statements and keeps their judgments valid while the development changes. A reviewer can ask the kernel targeted questions about a statement, such as whether a hypothesis is needed or whether a counterexample exists. Each decision is stored with fingerprints of the statement as Lean elaborated it and of the definitions it depends on, and any declaration can be pinned so that a patch changing it is blocked, whether a person or an agent built on a large language model (LLM) wrote the patch. Working on elaborated terms rather than source text is the central design choice: notation and formatting changes leave a decision intact, and a changed definition reaches every theorem that uses it. On 229 pairs of consecutive revisions of Mathlib, Lean's mathematics library, ProofPatch sends 118 of 12,898 synthetic decisions back for review. Every text-based alternative misses at least 38 of these or sends back at least 13 times as many. On 1,527 labeled statements, kernel evidence flags unfaithful ones with precision 0.88 to 0.96, and in 344 runs of an LLM agent no change to a pinned statement got through. 

---
# J-Anchor: Defending Against Prefill Attacks via J-Space Selective Refusal Restoration 

**Authors**: Running Yang, Beidi Zhao, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.01342)  

**Abstract**: Prefill jailbreaks can cause safety-aligned language models to continue an unsafe assistant prefix even when they would refuse the same request under normal decoding. We study how the model's internal harmfulness and refusal signals evolve along this attacker-controlled trajectory. Using the Jacobian lens, we track vocabulary-grounded harmfulness and refusal signals from the assistant header through prefilling and subsequent generation. We identify a characteristic phenomenon, which we call refusal-state erosion: both the refusal signal and the harmfulness signal weaken as the prefill progresses, but the refusal signal progressively declines to the benign baseline by the end of the prefill, while the harmfulness signal remains above its benign baseline and detectable during subsequent generation. Motivated by this observation, we introduce J-Anchor, a training-free inference-time defense that uses the model's pre-attack harmfulness and refusal signals as a request-specific anchor and selectively restores weakened refusal in J-space during prefilling and decoding. J-Anchor substantially reduces attack success under both fixed and deep prefill attacks across different model families. J-Anchor largely preserves benign behavior and reasoning ability, and ablations highlight the importance of targeted restoration in the J-space refusal subspace. Our results suggest that preserving latent refusal dynamics offers a promising route to robust defense against prefill jailbreaks. 

---
# The Verifier Is Part of the Simulator: Evidence-View Gaming in Agentic Equation Discovery 

**Authors**: Orestis Oikonomou, Bogdan Raonic  

**Link**: [PDF](https://arxiv.org/pdf/2611.01334)  

**Abstract**: A growing number of agentic pipelines search for governing equations: an agent proposes laws and a verifier checks them against simulated trajectories. The verifier sees only a numerical summary: integrals against test functions on a time grid. That summary carries its own discretization error, making the verifier part of the simulator. We freeze the trajectory, candidate library, verifier and thresholds, leaving the agent one move: which permitted summary to admit. Without scores or language models, finer time grids reduce numerical discrepancies and false library rejections. On PDEBench, the verifier instead accepts equations missing diffusion on score-selected summaries. Maximizing a public score rewarding decisive verdicts and larger mismatches when rejecting the library wrongly rejects all 1,500 in-library cases; retrospectively retuning score constants on a separate bank reduces errors to 13/180 at best. We call this evidence-view gaming: reward hacking by choice of permitted summary. In single-turn forced-choice tests, Claude models rarely choose summaries giving wrong verdicts when asked to be accurate, but often do so when asked to maximize the score or report missing physics. In separate trust prompts, they often rank those harmful summaries least trustworthy. Requiring several summaries to agree reduces errors but often withholds a verdict and performs worse than a fixed fine view on PDEBench: a safeguard, not a certificate. Evidence chosen after its verdict is known cannot alone authorize a claim. 

---
# Step-dLLM: Adaptive Step-aware Sparse Attention for Efficient Diffusion LLM Inference 

**Authors**: Zhichen Zeng, Xichong Zhang, Junpan Wu, Yifei Zuo, Chi-Chih Chang, Jiayi Wang, Maohua Nie, Ji Liu, Ang Li, Banghua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.01320)  

**Abstract**: Diffusion large language models (dLLMs) generate tokens in parallel via iterative denoising, but their bidirectional attention prevents the KV caching used by autoregressive models and recomputes the full attention map at every step, incurring quadratic per-step cost. Existing sparse attention methods for dLLMs construct a sparse mask after an initial full-attention warm-up and reuse it for the remaining steps, while allocating the sparsity budget uniformly across heads. We argue both choices ignore key structural properties of attention in dLLMs: empirically, attention patterns are only locally consistent across steps and shift abruptly at certain transitions, and different heads concentrate their attention mass to very different extents. To this end, we design Step-dLLM, which periodically refreshes a block-sparse mask at anchor steps along the denoising trajectory and reuses it within each locally consistent interval, while employing a head-adaptive global budget that more aggressively compresses heads with concentrated attention. Backed by a fused IO-aware Triton kernel for block-sparse attention, Step-dLLM applies to pre-trained LLaDA-1.5 and Dream-7B-Instruct without retraining, achieving a 2.78x end-to-end speedup over dense FlashAttention on LLaDA-1.5 at 8K context with 256 denoising steps and 90% sparsity, while improving the accuracy-latency trade-off over prior sparse-attention methods. 

---
# CuriousTogether: Implications of Curiosity and Cultural Grounding in Multi-Agent LLM Interactions 

**Authors**: Angana Borah, Zoha Hayat Bhatti, Zhijing Jin, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2611.01301)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in multi-agent settings, exhibiting emergent collective behaviors, yet the role of curiosity as a driver of such dynamics remains understudied. We introduce CuriousTogether, a framework evaluating curiosity-driven multi-agent LLM interactions across three complementary dimensions: novelty-seeking, epistemic and social curiosity. Across eight tasks spanning reasoning, creative generation, and social bias mitigation, curiosity-guided multi-agent settings yield improvements over non-curious ones on seven out of eight tasks, with gains of up to 20.6% in story generation (semantic diversity) and a 14.9 pp reduction in stereotype rates. Furthermore, country-conditioned cultural grounding show benefits for story generation and social tasks, but can hurt precision-oriented reasoning. Finally, we show that inference-time multi-agent curiosity-based approach offers favorable quality-token trade-offs, compared with single-agent baselines. Together, these findings support curiosity-guided interaction as a practical mechanism for improving multi-agent collaboration, with downstream benefits for reasoning, creative, and societal tasks. 

---
# Character-Name Concentration in LLM Fiction: Measuring Displacement and Mitigating Repetition 

**Authors**: Alejandro Mozo Quesada, Sergio Gálvez Rojas, Francisco López Valverde  

**Link**: [PDF](https://arxiv.org/pdf/2611.01284)  

**Abstract**: Character names in large language model (LLM) fiction concentrate on a handful of recurrent names such as 'Elara' and 'Elias', a measurable face of AI slop. Existing fixes either require logit access or retraining, or operate per request and cannot control concentration across a corpus. How much prompt-level strategies reduce concentration, and at what cost to name naturalness, has not been systematically measured. We present an empirical study of name-concentration control across eight strategies and introduce speculative naming, a fine-tuning-free method that adapts the draft-verify-reject paradigm of speculative decoding from tokens to names. The method requests candidate name-sets through verbalized sampling, scores them against a SlopTracker that combines a per-model fingerprint with a runtime counter, and selects through a temperature-controlled rule or its strict variant (HARD). We use 20 LLMs to generate 44,782 stories in four languages from text and visual stimuli (88,919 named characters), and measure concentration with a list-free collision metric (Slop Ratio, SR) decomposed into known and emergent components. List-based strategies do not remove slop: static banlists displace concentration to emergent names and more than halve the share of attested real names. Speculative naming removes it: the sampling scaffold alone cuts SR by 89% on average, the dynamic tracker brings the total reduction to 95%, mean collision probability falls from 7.3% to 0.4%, and known-name collisions hold at zero while naturalness stays close to baseline (42-69% attested by language) and within-story collisions drop from 4.3% to 0.2%. The strategy ranking is stable across languages, model families, and name-counting conventions. Name concentration is thus controllable at the corpus level without logit access or retraining; broader creative diversity beyond naming remains open. 

---
# Toward Bounding Coding-Agent Autonomy in Software Maintenance: A Pilot Study of Boundary Contracts 

**Authors**: Heeyong Eun, Seungyeon Chung, Hyejee Park, Youngeun Lee, Sukhyun Hong, Hyewon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2611.01265)  

**Abstract**: As coding agents gain greater autonomy to modify software repositories, relevant context alone does not define their operational authority. An artifact may be relevant to a maintenance task while being inappropriate to edit; conversely, another artifact may be required even when it is not explicitly mentioned in the request. We explore Boundary Contracts, a machine-readable abstraction that makes task-specific artifact roles (editable, inspect-only, required, and protected) explicit before repository modification begins.
We conduct a pilot study on 40 maintenance tasks from a real enterprise repository, comparing the same Codex-based coding agent with and without Boundary Contract information. The boundary-informed condition improves macro edit-selection F1 from 0.440 to 0.682, required-behavior coverage from 0.429 to 0.673, and the number of strictly acceptable patches from 9 of 40 to 18 of 40. To probe whether these gains are explained only by improved file localization, we additionally analyze 21 tasks in which both conditions modify at least one identical file. Differences remain in edit selectivity and patch outcomes, providing initial evidence that explicit artifact roles may influence maintenance behavior beyond localization alone.
This pilot does not establish the general effectiveness of Boundary Contracts. Rather, it provides initial evidence for a broader research question: how should the operational boundaries of coding agents be generated, validated, revised, and ultimately enforced as repository-level maintenance becomes increasingly autonomous? 

---
# LoRAForge: Co-Designing Post-Hoc Compression and Execution for Efficient Multi-LoRA Serving 

**Authors**: Shenghui Li, Thiemo Voigt  

**Link**: [PDF](https://arxiv.org/pdf/2611.01256)  

**Abstract**: Multi-LoRA serving allows many task-specific adapters to share a common base model, but scaling to large adapter fleets introduces substantial storage, loading, and execution overheads. Post-hoc compression can reduce the memory footprint of trained LoRA adapters, yet smaller ranks do not automatically translate into faster serving. For instance, padding, factor access, and generic kernels can obscure the retained low-rank structure, while independently recompressing thousands of modules also incurs significant preparation overhead. We present LoRAForge, a system connecting batched factor construction to rank-specialized execution through an explicit retained-rank representation. It supports two allocation policies: uniform retention ratios (LoRAForge-U) and spectral-energy budget allocation (LoRAForge-S). Both share FlashTSQR factor construction and the same serving backend. FlashTSQR exploits parallelism across modules and row tiles, accelerating factor construction by up to $2.28\times$ over matched library QR. In a fixed-work Qwen control, the shared runtime adds $1.17$--$1.22\times$ throughput across both compressed policies, with negligible gains on original adapters. Across 85 trained open-weighted Qwen and Llama adapters, the spectral policy's floored quality configuration retains about 7.7% of adapter parameters with fleet-average score drops of at most 1.2 points on a 0--100 scale. The same rank-floored $1/12$ spectral exports achieve substantial token-throughput gains on both AMD and NVIDIA GPUs. 

---
# Mathematical Formulation and Comparative Performance Evaluation of Machine Learning Algorithms for Maize Land Suitability Prediction 

**Authors**: Haimanot Bitew, Ayisheshim Almaw, Habtamu Zegeye, Demeke Alene  

**Link**: [PDF](https://arxiv.org/pdf/2611.01239)  

**Abstract**: Addressing the food needs of a growing population is one of the major challenges of developing countries like Ethiopia. Various policies aimed at increasing food production have been enacted by governments and development partners. Although efforts in this regard have been made, there is still inadequate emphasis on increasing food production through the use of high technology practices. Precision agriculture using technology has been utilized to increase crop yield. Precision agriculture also relies on identifying land for a crop, which is a key factor that affects crop yield. Traditional methods of assessing land suitability for crop production are time-consuming, costly and unreliable. Against this background, the aim of this study was to assess the potential of machine learning (ML) in improving the speed and accuracy of assessing land suitability for maize, a staple crop in Ethiopia, as compared to traditional methods. The study is investigated in three areas (Burie, Shindi and Tilili) in the Amhara region. Data collection was done through the local agriculture offices and survey teams. The data were coded, scaled and balanced. Various feature extraction techniques, which included Univariate Feature Selection(UFS), Recursive Feature Elimination with Cross-Validation(RFECV) and Sequential Feature Selection(SFS), and various classifiers, which were K-nearest neighbor (KNN), gradient boosting (GB),random forest (RF) with mathematical formulations were employed. The best models were constructed through combining GBOOST and SFS and achieved an accuracy of 99.41% and 99.35% for both precision and recall. Our results showed that using machine learning is a reliable method of assessing land suitability for maize in Ethiopia.
Keywords: Maize, Gradient boosting,Land suitability,Machine learning, Mathematical formulations 

---
# How Much Can Approximate Equilibria Change the Social Cost of Selfish Routing? 

**Authors**: Yu Cheng, Tianle Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.01234)  

**Abstract**: We study approximate Nash equilibria in nonatomic congestion games, with the goal of understanding how much the social cost can change when the equilibrium condition is relaxed. We consider $\epsilon$-approximate Nash equilibria ($\epsilon$-NE), in which each player uses a strategy whose cost is at most $1 + \epsilon$ times the minimum available cost. We compare the social costs of $\epsilon$-NE and exact Nash equilibria. This lets us quantify, for example, how much a navigation app can reduce total travel time when drivers tolerate routes slightly slower than the fastest available route.
For degree-$d$ polynomial latencies in both general nonatomic congestion games and single-commodity routing games on series-parallel networks, we give tight lower and upper bounds on the relative change in social cost from an exact Nash equilibrium to any $\epsilon$-NE, with matching instances for every $\epsilon$. For small $\epsilon$, the largest possible relative reduction and increase in general congestion games are both $\Theta(\sqrt{d\epsilon})$, independently of the number of resources and populations. In contrast, on series-parallel networks of depth $h$, the largest possible relative reduction is $\Theta(h\epsilon)$, while the largest possible relative increase is known to be $\epsilon$. For the classes of games we study, these results give a tight and quantitative characterization of the possible changes in social cost and establish a separation between general and series-parallel networks. 

---
# STLT: A Novel System for Energy-Aware Distributed Inference in the Edge-Cloud-Space Continuum 

**Authors**: Andrija Stanisic, Milos Gravara, Juan Luis Herrera, Stefan Nastic  

**Link**: [PDF](https://arxiv.org/pdf/2611.01233)  

**Abstract**: Low Earth Orbit satellites are pivotal for Earth observation tasks such as wildfire detection, capturing imagery for analysis by Deep Neural Networks (DNNs). On-board analysis reduces downlink volume, but severe energy constraints force a trade-off between computation and data transmission. Splitting the DNN between satellite and ground turns this balance into a design choice, where the split point determines combined computation and downlink energy. Existing approaches typically rely on static split-point decisions that fail to account for dynamic space conditions, or on complex algorithms, historical data, or prediction models that introduce significant overhead. To address these shortcomings, we present STLT, a two-stage system for energy-aware distributed inference between LEO satellites and a ground station. First, an offline analyzer characterizes the DNN architecture to identify a set of favorable split points, consequently reducing the search space. Second, an on-board algorithm performs lightweight energy-aware split selection, dynamically choosing from this pre-computed set based on the current state of the environment. We evaluate STLT on a real-world satellite telemetry dataset, showing that it reduces the split-candidate search space by 30.7% on average, lowers decision latency by 25.1% and energy by 27.5% on average, and delivers up to 20.1% more inference tasks than a state-of-the-art satellite-ground co-inference baseline while consuming 8.8% less energy overall. 

---
# Vision-Language Models Are Joint Denoisers in Robotic Action Generation 

**Authors**: Xicheng Gong, Wenxuan Song, Chengze Liu, Qiwei Li, Qize Yu, Jiaolong Yang, Haoang Li, Yadong Mu  

**Link**: [PDF](https://arxiv.org/pdf/2611.01215)  

**Abstract**: Diffusion-based vision-language-action (VLA) models cascade pretrained vision-language models (VLMs) with diffusion action experts, but typically restrict the VLM to providing static conditioning. This separation limits how much pretrained perception and reasoning can guide action generation, especially in out-of-distribution settings. We introduce JADE, a Joint Action DEnoiser that enables the VLM and action expert to jointly refine actions. Our key insight is that structured action latents allow the VLM to read the current noisy action together with visual observations and language instructions. At every denoising step, the VLM contextualizes the noisy action and guides the action expert through gated cross-attention, so that its guidance evolves with the action being generated. We further supervise the VLM to recover clean action latents from noisy ones, which gives it a direct action-learning signal and couples perception with control. Additionally, joint denoising enables JADE to generate actions with fewer sampling steps, helping offset the additional cost of VLM participation. An adaptive step-size schedule further balances task performance and inference latency. On RoboDojo, JADE achieves an average success rate of 12.67%, the highest among the evaluated $\pi_{0.5}$-based methods, and is competitive with recent VLA and WAM foundation models pretrained at scale. It also improves performance on RoboTwin 2.0 and raises the average success rate across four real-world bimanual tasks by 20.8 percentage points compared with $\pi_{0.5}$. 

---
# ArSciKids: Arabic Curriculum-Aligned Datasets for Scientific Experiment and Multiple-Choice Question Generation in Primary Education 

**Authors**: Taghrid Housari, Massa Rankoussi, Jafar Abdul Majeed, Wael Tarabishi, Khloud Al Jallad, Nada Ghneim  

**Link**: [PDF](https://arxiv.org/pdf/2611.01191)  

**Abstract**: Despite growing interest in Arabic natural language processing, curriculum-aligned educational resources remain scarce compared with English, and few public datasets for Arabic curriculum content with structured, generation-ready annotations are available, particularly for primary-school science. We introduce ArSciKids, a collection of two Arabic datasets designed for curriculum-aligned educational content generation: (1) a scientific-experiments dataset containing 999 manually curated records from official primary-school science textbooks in Saudi Arabia, Syria, and Jordan, and (2) an assessment-question dataset containing 2,167 multiple-choice questions collected from the Saudi Manhaji educational platform. The experiment records are structured with titles, objectives, tools, procedures, observations, conclusions, keywords, and grade levels, while the assessment records contain lesson references, questions, four answer options, correct answers, and explanations. Both datasets cover grades 1-6 and consist of real-life Arabic educational content rather than machine-translated or synthetically generated material. We assess the usefulness of these resources for generation by fine-tuning ten open models spanning sub-billion to several-billion parameter scales using LoRA. Models are evaluated using automatic metrics, including ROUGE and BLEU, and answer accuracy for multiple-choice generation. Moreover, we evaluate the results using a five-criterion human evaluation protocol. The results show substantial differences across model scales and tasks. ArSciKids will be publicly available, and we hope that it provides a manually curated, structured foundation for research on Arabic educational NLP, curriculum-aware generation, and low-resource language modeling. 

---
# Satisfied, Not Solving: Wireheading of Bounded Homeostatic Rewards in Reinforcement Learning 

**Authors**: Chase J. Metoyer  

**Link**: [PDF](https://arxiv.org/pdf/2611.01183)  

**Abstract**: Bounded intrinsic rewards are often assumed to be safe from exploitation: if a signal can yield at most a fixed amount, gaming it should not pay. We test this for a homeostatic satiation reward, in which needs build up and relieving them is rewarding, and give the agent a cheap action that resets its needs. In MiniGrid DoorKey-8x8, PPO agents with this action solved 2 of 80 seeds, against 71 of 80 on the same seeds without it; in KeyCorridorS3R1, 0 of 20 against 20 of 20. Their policies stayed near uniform. The reward does not ask for this: a telescoping argument caps any policy's satiation return at the needs' starting drive, and a post-hoc computation shows that, at every growth rate we trained, a solving trajectory beats that cap on all sampled layouts. In DoorKey with advantage normalization on, a satisfaction signal the agent could fill itself was worse than none (11 of 20 solved with task reward alone, 1 of 20 with the tamperable signal), and a per-press cost of 0.1 restored the task in every seed. Farming of positive reward cycles appeared in DoorKey but not persistently in KeyCorridorS3R1. All predictions were pre-registered, and hash-locked from round 3 on; 9 of our 23 hypotheses were not supported, and we report each. 

---
# Lightweight Representation Learning for Low Tucker Rank Tensors 

**Authors**: Hiroki Hasegawa, Yukihiko Okada  

**Link**: [PDF](https://arxiv.org/pdf/2611.01168)  

**Abstract**: Low Tucker rank tensors provide compact representations of multidimensional dependencies, but standard learning pipelines can incur substantial computational and memory overhead from redundant intermediate quantities. We reduce this overhead without modifying the underlying representation objective or parameterization. Our approach combines three complementary components: (i) sequential extraction of only task-required spectral components, avoiding unnecessary full decompositions; (ii) factor-wise multilinear contractions without materializing large Kronecker operators; and (iii) compact latent-state updates without storing full ambient tensor iterates. These components preserve the algebraic operations and learning trajectories of the original algorithms while reducing their computational and memory footprints. Across five low Tucker rank tensor estimators, our implementations match the original trajectories to numerical precision (relative discrepancies of $10^{-16}$ to $10^{-13}$), while reducing per-iteration runtime by up to $3.12\times$ and resident memory by up to $616\times$ ($99.8\%$) on the main order-3 benchmark at $n=100$, with memory reductions exceeding $900\times$ across the full evaluation suite. These results show that substantial computational redundancy can be eliminated from Tucker representation learning without changing the underlying model or learned representation. 

---
# Discussing politics with a chatbot: How political beliefs shape prompt formulation for generative AI 

**Authors**: Victoria Vziatysheva, Reinhold Kliegl  

**Link**: [PDF](https://arxiv.org/pdf/2611.01153)  

**Abstract**: By generating comprehensive and highly personalized answers to user prompts, AI chatbots such as ChatGPT and Gemini can serve as alternatives to search engines and substantially transform online information seeking. However, little research has examined how users interact with these tools to find information, particularly about politics. In this experiment, we examine how Swiss citizens (N = 563) formulate prompts to search for migration-related issues using AI chatbots and whether this process is affected by selective exposure. We find evidence of attitude-consistent prompting, whereby participants who were more critical of migration were more likely to formulate prompts with an anti-migration stance. This effect was stronger for a broad topic (migration in general) than for a specific one (anti-migration popular initiative put to a vote in Switzerland). However, exposure to an attitude-consistent AI response did not further amplify selectivity in prompting. We also did not find evidence that either AI attitudes or thinking styles moderated selectivity in prompting. These findings contribute to the understanding of selective exposure via AI chatbots and the use of generative AI for political information-seeking. 

---
# World Models Need Not Be Right to Rank Policies Right: Relative Bounds and Audit Limits 

**Authors**: Ahanaf Hasan Ariq, Hasaan Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2611.01139)  

**Abstract**: Selecting a reinforcement-learning (RL) policy usually requires additional environment interaction, while learned world models offer cheap but potentially biased evaluations. This raises two distinct questions: can shared model errors preserve pairwise rankings despite poor absolute values, and do they make a small number of real audits more informative? We derive an exact pairwise simulation identity and a relative bound separating common-mode Bellman residuals from policy-specific error. The analysis motivates RAVEL (Relative Auditing of Value Errors in Learned models), a selective policy tournament whose eliminations use held-out real outcomes. We test 45 SAC, TD3, and PPO policies per task in DMC cheetah-run and walker-walk. At rollout horizon 500, model rankings remain strong (Kendall tau = 0.962 and 0.800), and same-run differential-error ratios are 0.674 and 0.106. Yet mean paired audit-error correlations are only 0.007 and 0.090, and pairing changes standard deviation by -2.5 percent and +1.9 percent. Simultaneous intervals attain 99.98 percent and 99.75 percent coverage, but certify only 13.2 percent and 0 percent of decisions at eight audits. RAVEL improves low-budget walker-walk regret, whereas real-only successive halving is better at the largest budget, rejecting the precommitted twofold efficiency target. Frozen-model optimization lowers real return without increasing predicted return on average. Thus ranking fidelity, useful audit covariance, and upward model exploitation are distinct properties, separated in part by closed-loop trajectory divergence. 

---
# SIDRec: End-to-End Semantic ID Representation and Modeling for Recommendation 

**Authors**: Zhuohang Jiang, Yongsen Pan, Pangjing Wu, Zihuai Zhao, Zheng Hu, Yuxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.01136)  

**Abstract**: Semantic IDs (SIDs) have emerged as a promising approach for discriminative recommendation by encoding continuous item rep- resentations into compact discrete identifiers. However, existing SID-based methods face two fundamental limitations: semantic mis- match between content semantics and collaborative preferences, and objective mismatch between SID construction and downstream recommendation optimization. To address these challenges, we propose SIDRec, an end-to-end recommendation-aware SID learn- ing framework that jointly models content and collaborative in- formation while aligning SID construction with recommendation objectives. Specifically, SIDRec introduces a self-updating fusion mechanism that explicitly integrates pretrained content represen- tations with collaborative embeddings initialized from a trained recommendation model, enabling preference-aware semantic mod- eling. Furthermore, SIDRec employs an MMoE-inspired joint opti- mization mechanism with shared experts and task-specific gates, allowing recommendation supervision to directly guide differen- tiable quantization and produce task-adaptive discrete representa- tions. Extensive experiments on three public benchmark datasets demonstrate that SIDRec consistently outperforms all competing methods across both AUC and GAUC, achieving up to 5.15% relative improvements over the strongest baselines. Further ablation stud- ies validate the effectiveness and complementarity of collaborative semantic modeling and end-to-end SID optimization. 

---
# What Does a Safety Label Measure? Auditing Situational-Safety Benchmarks for Multimodal LLMs 

**Authors**: Wayner Barrios, Kenneth Barrios  

**Link**: [PDF](https://arxiv.org/pdf/2611.01125)  

**Abstract**: Situational-safety benchmarks label each scene SAFE or UNSAFE and score a multimodal large language model (MLLM) against that label, usually by judging its free-form answer, and these scores guide deployment. We ask what such a label measures. A label score can come from the scene, the request or the benchmark's own files. We propose the Benchmark Card, seven measured fields that tell them apart, and fill it for six benchmarks. Only MSSBench chat leaks the label through its image files: image size alone reaches 82.3% accuracy, matching or beating six of seven commercial models. Elsewhere the signal outside the scene comes from design. In MOSSBench the item partition fixes the label, and on HoliSafe the request alone ranks pairs about as well as the scene. On our corrected chat split, the size rule falls to chance. We then introduce the Spot-Stop decomposition, which compares ranking with labelling on pairs of a safe and an unsafe scene. It splits the gap above a label-swap null into three parts: a threshold part, overlap between safe and unsafe scores, and order that holds only within a pair. Most of fourteen open MLLMs rank better than they label, beyond what their AUROC predicts. On chat, one shared threshold repairs a third to a half of the excess, and label-free recentering recovers it on balanced batches. Recentering on unlabelled scenes of the same situation improves ranking on VSCBench for 13 of 14 models. 

---
# Auriga Technical Report: A Task-Agnostic Agent Harness for Zero-Shot Embodied Manipulation 

**Authors**: Chengcheng Wang, Yuheng Shi, Siyu Xu, Yutong Wang, Jinjing Zhao, Chenghao Xia, Jinqi Liao, Jiankun Peng, Ligong Bi, Yuechuan Ma, Jiaqi Liu, Wenbo Wang, Zijian Wang, Yuemin Wu, Minjing Dong, Xiu Su, Tao Huang, Jianyuan Guo, Chengbin Du, Yunke Wang, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.01112)  

**Abstract**: Vision-language models can interpret instructions and plan manipulation tasks, yet struggle when acting directly as robot policies. On RoboDojo, a large planning-execution gap separates high-level task decisions from physical execution: removing arm execution raises GPT-6 Astra's success rate from 22.5% under the reference control interface to 76.2%. We present Auriga, a task-agnostic agent harness that closes 68.3% of this gap in zero-shot manipulation without target-task trials or resets. In Auriga, the model makes all task decisions over an explicit task belief and acts through either direct tool calls or synthesized multi-step Python programs, while the harness supplies calibrated 3D measurements, finger-referenced dual-arm control, and a single admission gateway to the robot. Before motion, the model inspects a visual trajectory preview while simulation is paused; during execution, the harness checks declared stage tolerances and grasp bounds, stopping the robot on divergence and returning the measured discrepancy for in-place recovery. Developed across 13 cross-task capability iterations and frozen for evaluation, Auriga ranks first on the RoboDojo leaderboard with a 57.7% success rate (64.0 Score), more than doubling the reference harness on the same model, and remains first under scene randomization while learned policies drop by up to 33.9%. On RoboCasa365, the same frozen harness reaches a 73.2% success rate, outperforming VLA-assisted agents by up to 23.1% on composite tasks.
Project page: this https URL Code: this https URL 

---
# Ripple: Deadline-Aware CPU-GPU Sharing for Embodied Inference 

**Authors**: Jiadong Zhang, Xiaosong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.01067)  

**Abstract**: Unified-memory edge devices increasingly run LLM/VLM inference alongside time-critical CPU tasks for perception and control. GPU memory traffic can overwhelm these tasks, yet platforms such as DGX Spark expose neither bandwidth counters nor direct bandwidth regulation. We present Ripple, a CPU-GPU co-scheduler that combines model-derived GPU memory volumes with CPU completion feedback to regulate interference. Ripple divides inference into pausable, layer-based units and translates each CPU job's deadline slack into a GPU-byte budget, admitting work within the budgets of all overlapping jobs. Online updates track changing interference, while deadline-aware pacing and judicious SM footprint narrowing preserve inference progress. Implemented with SGLang engine hooks and a central scheduler, we evaluate Ripple on DGX Spark with four robotic CPU task mixes and LLM/VLM inference. It is the only evaluated approach that consistently meets the 2% CPU deadline-miss target and VLM deadlines across the tested deadline settings while retaining best-effort LLM throughput. 

---
# Label Smoothing in Disguise? Gated Knowledge Distillation for Low-Resource Bangla Generation 

**Authors**: Dipto Sumit, Zobeida Zaman Chowdhury, Syeda Fabiayee Jahan, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2611.01065)  

**Abstract**: Gated knowledge distillation (KD) weights teacher signals per token or example, aiming to trust an imperfect teacher only where it seems reliable. We test two gates: EWAD (entropy-weighted agreement-aware distillation), which gates tokens by teacher confidence, and CHAD (counterfactual harm-aware distillation), which gates examples by gradient alignment with validation data. Across six Bangla-centric tasks, three student models (60M to about 2B parameters), and two teacher families, no gate significantly improves plain KD in 45 comparisons. KD also fails to beat cross-entropy (CE) fine-tuning on gold outputs in 40 single-seed comparisons or three-seed runs on four tasks; CE leads on every summarisation metric. The main benefit of KD is regularisation: over ten fixed epochs, CE students lose 0.030 to 0.044 token-F1 from their early-stopped checkpoint, versus 0.005 to 0.025 for distilled students. Teacher-free label smoothing closes most of this gap. In one comparison, a teacher 0.29 token-F1 worse yields students at most 0.009 worse. Thus, gating may change regularisation strength but has little teacher knowledge to select. Findings are limited to Bangla-centric tasks and small students. We will release code, data splits, predictions, and scores. 

---
# Dependable Agents Need Hard and Fuzzy Rules 

**Authors**: Frederick Roth  

**Link**: [PDF](https://arxiv.org/pdf/2611.01043)  

**Abstract**: Language models let software read requests in ordinary language, but they keep rules as people do: usually, and less so under pressure. Conventional code keeps rules exactly, but it cannot read a request. We argue that dependable agents need both, because the two kinds of rule cover overlapping and unequal regions of what an agent might do. Code holds the limits that can be stated exactly. Learned fuzzy rules cover the judgment that cannot. We test the hard layer in three domains, purchasing, delegation of authority and data access, on 1,340 episodes written by authors outside our team. Models left to decide for themselves executed unauthorized actions in up to 151 of 340 purchasing episodes and 269 of 470 delegation episodes, and the strongest model breached in every condition. A hard guard between proposal and action executed none, and an exhaustive search of 183,204 proposals agreed. In a further test, where an outsider wrote the owner's rules and a program mapped them to the guard, two episodes breached at a boundary we had written down in advance. We then test the pair. Concept Programming instills a named rule in a model and measures its errors on wording it never saw. A small model taught four judgment rules this way flagged 95 of 96 rule-breaking purchases that no exact rule can see, and 4 of 108 legitimate requests. Placed in front of the guard, it left 1 of 192 rule-breaking requests unstopped, where the guard alone left 77 and 70, and the guard still held every exact limit. A prompted frontier model did about as well, so the claim is that a small taught rule can do the job, not that it does it better. 

---
# A Token Is Not a Unit of Difficulty: Segmentation Controls Where Adaptive Computation Goes Inside a Word 

**Authors**: Dipto Sumit, Mahima Mahzabin Sandria, Farig Sadeque, Md. Golam Rabiul Alam  

**Link**: [PDF](https://arxiv.org/pdf/2611.00983)  

**Abstract**: Early-exit language models decide for each token how much computation to spend, usually by stopping once the prediction is confident. Yet a token is not a fixed unit: tokenizers cut words into different numbers of pieces in different languages. We ask where inside a word such a policy spends its budget. In Qwen3, along two adaptive axes (the timesteps of a spiking conversion and the layers of the original Transformer), confidence stopping gives the continuation pieces of a word less computation than its first piece in English, German, and Turkish, but more in Bangla. Segmentation drives this. Re-segmenting the same parallel passages with BPE-dropout shifts the gap by up to eight exits on both axes, small models trained under two tokenizers shift Bangla's gap by six exits, and BLOOM and mGPT, which split Bangla less finely, give it the English pattern. Characters per token do not predict the gap across languages; within-word surprisal explains 75% of its variance. A learned stopping head beats confidence stopping by 7 to 8 points, but gains less than one point over the best mix of fixed exits at equal cost, and sometimes loses. All results come from offline analysis, not measured speedups. 

---
# Efficient Image Denoising on Smartphone GPUs: Mobile AI 2026 Challenge Report 

**Authors**: Andrey Ignatov, Georgy Perevozchikov, Radu Timofte, HaoDong Wu, Desheng Cai, Nikolay Falaleev, Faraz Kayani, Sarmad Kayani, Asad Ahmed, Dmitry Ignatov, Kihwan Yoon, Ganzorig Gankhuyag, Yoonsuk Kang, Van Le, Jin Young Lee, Sunzhe Yang, Qiuhai Yan, Kang Chen, Tingting Wang, Faming Fang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00974)  

**Abstract**: Image denoising is a fundamental problem in mobile photography. Despite a large number of solutions proposed for this task, the majority of them are trained on synthetically produced noisy images, and are additionally too computationally heavy for processing full-resolution images on mobile devices. This creates a need for efficient deep learning-based models that are not only performant but can also be deployed on constrained mobile hardware. In this Mobile AI challenge, we address this problem and propose the participants to design fast solutions that can denoise Full HD resolution photos while running on popular smartphone GPUs from Qualcomm and Arm. For this, the participants were provided with a large-scale dataset consisting of noisy-clean image pairs captured in the wild. The runtime of all models was evaluated on the Snapdragon 8 Elite Gen 5 and Dimensity 9500 chipsets with the Adreno and Mali GPUs, respectively. Top solutions proposed in this challenge are able to perform high-quality image denoising under 100 milliseconds on both platforms while reaching a PSNR score of over 37 dB. A comprehensive description of the models developed in the challenge is provided in this paper. 

---
# 4K Image Super-Resolution on Mobile NPUs: Mobile AI 2026 Challenge Report 

**Authors**: Andrey Ignatov, Georgy Perevozchikov, Radu Timofte, Fubin Ren, Shihao Wang, Liwen Zhang, Zhe Xu, Zhipei Lei, Minjian Zhang, Kailing Tang, Dingyong Gou, Yanlin Wu, Cong Li, Xiaohui Cui, Kihwan Yoon, Ganzorig Gankhuyag, Heo MyeongCheol, Yonghyun Park, Juho Jung, Sungjei Kim, Phuong-Nam Nguyen-Pham, Nam-Tien Le, Tinh-Anh Nguyen-Nhu, Matin Fazel, Abdelhak Bentaleb, Xinrui Li, Xusheng He, Jianlong Wu, Mohsin Ikram, Dmitry Ignatov, Hao Chen, Long Sun, Jiangxin Dong, Jinshan Pan, Jinhui Tang, Siyue Teng, Yuxuan Jiang, Fan Zhang, David Bull, Zhe Xiao, Jun Xiao, Zhi-Song Liu, Jiayi Pei, Songtao Wu, Hong Shang, Kuanhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.00963)  

**Abstract**: Image super-resolution is widely used on mobile devices for upscaling various types of image data. This creates a need for efficient deep learning-based models that are not only performant but can also be deployed on neural processing units (NPUs) present in the majority of mobile and IoT chipsets. In this Mobile AI challenge, we address this problem and propose the participants to design fast solutions that can upsample images from Full HD to 4K resolution while running on popular mobile NPUs from Qualcomm and MediaTek. For this, the participants were provided with the DIV2K dataset and trained models to do an efficient 3X image upscaling. The competition had two tracks: in the first track, the models had to be fully quantized (INT8), and the runtime of all models was evaluated on the Qualcomm Snapdragon 8 Elite Gen 5 NPU. In the second track, the participants were developing floating-point solutions that were evaluated on the MediaTek Dimensity 9500 NPU. Top solutions proposed in this challenge are able to reconstruct 4K images under 5 milliseconds while reaching a PSNR score of 30 dB. A comprehensive description of the models developed in the challenge is provided in this paper. 

---
# Self-Refining Prompt Induction for Streaming Text-to-SQL 

**Authors**: Gaetano Rossiello, Alfio Gliozzo, Nandana Mihindukulasooriya, Erick Lee, Nabila Siregar, Lisa Amini  

**Link**: [PDF](https://arxiv.org/pdf/2611.00960)  

**Abstract**: Text-to-SQL assistants for streaming engines must handle dialect-specific types, windowing rules, and temporal semantics, including constructs poorly handled by generic prompts. We present Self-Refining Prompt Induction (SRPI), an offline method that converts generation failures into reusable dialect instructions while keeping model weights fixed. A Refiner revises the Generator's prompt using reference-conditioned feedback from three LLM judges; a sampled regression guard checks earlier successes, and a separate Compressor reduces prompt length. Inference uses the frozen prompt without retrieval or query repair. We evaluate cross-domain and cross-model transfer on Confluent Cloud Flink SQL using a controlled synthetic workload seeded by 30 expert-written examples, with 335 induction examples and 257 questions from held-out domains. The same induced prompt improves mean judge scores across seven Generators by 7.1-19.2 percentage points, including an increase from 85.3% to 97.5% for the induction model. A separate compression study reduces prompt length by 52.7% with a 0.19-percentage-point score decrease. An assistant using the induced prompt is in early access, with initial user feedback to inform subsequent refinement. 

---
# HieraNTP: Structured Risk Representations for Time-Series Anomaly Detection via Hierarchical Next-Token Prediction 

**Authors**: Yujie Li, Zezhi Shao, Boyu Diao, Zhulin An, Yongjun Xu, Xueqi Cheng, Fei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00945)  

**Abstract**: Time-series anomaly detection commonly converts deviations from learned temporal regularities into anomaly scores. This practice can conflate violations of temporal regularity with task-specific anomaly judgments: the same deviation may be anomalous under one task definition and benign under another. Aggregating heterogeneous deviations into a single score before task-specific adaptation can obscure the evidence needed to distinguish these cases. We introduce HieraNTP, a Hierarchical Next-Token Prediction framework that separates self-supervised predictive modeling from task-specific risk interpretation. HieraNTP pretrains a hierarchical tokenizer and a causal next-token predictor to capture temporal regularities at multiple scales. It then organizes prediction discrepancies into a structured risk representation, preserving complementary evidence across temporal scales in token, local-statistic, and observation spaces. A task-adaptive detector uses anomaly labels to select and combine this evidence for the target task. Across anomaly detection benchmarks, HieraNTP improves VUS-PR by 12.0% on TSB-AD-U and 15.5% on TSB-AD-M relative to the strongest baselines, and achieves the highest composite score on both TAB tracks, with controlled comparisons supporting the value of retaining distinct evidence. Our code is available at this https URL. 

---
# SNAPPerf: Software-Aware Tabular Foundation Model for Retraining-Free Performance Prediction of Configurable Systems 

**Authors**: Amine Lbath, Massih-Reza Amini, Vadim Okun  

**Link**: [PDF](https://arxiv.org/pdf/2611.00944)  

**Abstract**: Modern software systems expose hundreds of configuration options whose combinations can substantially affect execution time, memory and energy usage. Measuring software performance resulting from a configuration set requires running a realistic workload. Since the configuration space grows exponentially, exhaustive benchmarking is infeasible. Prior work therefore trains models on a small set of measured configurations to predict the performance of the rest. Yet these models are specific to a particular software version, hardware platform, and workload, so changes require new measurements and retraining, and such changes are becoming more frequent with coding agents.
We challenge this per-system training assumption with SNAPPERF (Software-aware, No-retraining, Adaptive Prediction), a sample-efficient tabular foundation model specialized for configurable software. SNAPPERF is pretrained once on synthetic data. Each new update only requires a few measured configurations as context to predict the rest without any per-system training or hyperparameter tuning. Across 39,866 configurations from 12 real-world systems, spanning 6 domains and 4 performance metrics, SNAPPERF reduces prediction error by 38% over the prior state of the art at the same measurement budget, and by 25% over the best general-purpose tabular foundation model. It matches the state of the art's accuracy with 54% fewer measurements on average. These results suggest that knowledge about software can be encoded in pretraining data rather than in per-system models, opening a new research direction in software-aware pretraining. 

---
# MarkovTS: Pretrained State-Transition Representations for Zero-Shot Time Series Generation 

**Authors**: Yujie Li, Zezhi Shao, Jingbo Zhou, Yongjun Xu, Xueqi Cheng, Fei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00941)  

**Abstract**: Across heterogeneous real-world domains, time-series generation supports simulation, model development, and data augmentation by generating sequences that preserve target-domain statistics and temporal dependencies when real observations are scarce or costly. However, existing methods predominantly address in-domain generation, limiting their reuse in unseen domains. We study reference-conditioned zero-shot time series generation: target domains are unseen during pretraining, generation uses a single target-domain reference sequence, and no model parameters are updated. MarkovTS characterizes sequence evolution through local morphologies and their ordered transitions at multiple temporal scales. Its pretrained encoder quantizes normalized time-series patches into multiscale morphological states with a state codebook, while a transition codebook encodes contextual transitions between adjacent intervals. These transferable state-transition representations condition a pretrained flow-matching model. At generation, the frozen model combines a reference-derived condition with independent Gaussian noise to produce stochastic samples that retain the reference's temporal organization. Across 12 unseen target datasets, MarkovTS outperforms strong baselines in most distributional comparisons and generates diverse, reference-consistent sequences. As downstream augmentation, its samples yield overall gains across forecasting, classification, and anomaly detection benchmarks and diverse model families. Our code is available at this https URL. 

---
# Labadain-MedQA: A Tetun Medical QA Dataset for Fine-Tuning and Evaluating Large Language Models 

**Authors**: Gabriel de Jesus, Sérgio Nunes  

**Link**: [PDF](https://arxiv.org/pdf/2611.00935)  

**Abstract**: Advances in large language models (LLMs) have created opportunities for cross-lingual transfer of medical knowledge from high-resource to low-resource languages to support access to healthcare information. However, medical question-answering (QA) datasets for fine-tuning and evaluating LLMs remain scarce for Tetun. To address this gap, we translated MedAlpaca's Medical Flashcards and WikiDoc Patient Information collections into Tetun using Gemini 2.5 Flash, producing 39,077 medical QA pairs. Translation quality was assessed on 50 randomly sampled QA pairs (100 documents) using automatic metrics against human reference translations and evaluations by two native Tetun-speaking healthcare professionals, focusing on medical accuracy and Tetun fluency. Human translations achieved higher mean ratings than LLM-generated translations for both criteria (4.43/5 vs. 4.02/5), with statistically significant differences (Wilcoxon signed-rank test, p < 0.001). To assess downstream utility, we fine-tuned Gemma and Llama variants on the translated MedAlpaca dataset using parameter-efficient fine-tuning (PEFT) and evaluated them on 325 Tetun-translated United States Medical Licensing Examination multiple-choice questions. Gemma 4 12B fine-tuned using QLoRA achieved relative improvements in mean accuracy of 10.79% with 4-bit NormalFloat (NF4) inference and 10.42% after merging the LoRA adapters into the BF16 base model, compared with the corresponding base models. These results highlight the viability of automatic dataset translation for Tetun medical QA and the effectiveness of PEFT in enhancing model performance. The datasets, models, and evaluation resources are accessible at this https URL to support reproducible research on Tetun medical QA and facilitate machine translation, cross lingual, and multilingual research. 

---
# UI-UG 2: Advancing Mobile UI Understanding and Generation with Fine-Grained Supervision and Reinforcement Learning 

**Authors**: Hao Yang, Shuoyang Liu, Xiaoyu Lin, Zhou Fang, Maji Huang, Zhaosong Huang, Ruichao Mao, Teng Guo, Hai Rao  

**Link**: [PDF](https://arxiv.org/pdf/2611.00926)  

**Abstract**: Mobile UI understanding and generation require both fine-grained component perception and reliable structured synthesis. Inspired by the unified direction introduced by UI-UG, we present UI-UG 2, a compact 4B-parameter multimodal large language model (MLLM) that advances mobile UI Understanding and Generation with broader task coverage, a front-end-aligned taxonomy, and online render-aware reinforcement learning. UI-UG 2 constructs a multi-source understanding corpus from RICO, VINS, Android in the Wild, and Android Control, annotated with an 80-category two-level UI taxonomy aligned with front-end component semantics, and pairs it with over 5,000 requirement-screenshot-JSON tuples for progressive structured UI generation through a domain-specific language (DSL). For understanding, UI-UG 2 introduces decomposed rewards for category prediction, OCR fidelity, color recognition, dense grounding, and GUI grounding. Unlike the offline Direct Preference Optimization (DPO)-style generation optimization in UI-UG, UI-UG 2 optimizes generation through end-to-end online Group Relative Policy Optimization (GRPO): sampled DSL outputs are rendered with a Playwright-based server-side pipeline and evaluated using LLM-as-a-Judge and contrastive preference ranking. Experiments show that UI-UG 2 achieves the strongest referring and dense-grounding results among the evaluated models on our 80-category benchmark, and achieves the best composite generation score among compared models at a compact scale (4B). More importantly, we observe a consistent understanding-to-generation transfer: adding fine-grained UI perception supervision improves generation quality under a fixed training budget, suggesting that component-level semantics provide a shared representational foundation for both tasks. 

---
# Illusion: Safeguarding Topology Confidentiality in LLM Multi-Agent Systems via Behavior-Preserving Obfuscation 

**Authors**: Yiu-Fai Cheung, Hongzhi Guo, Miao Pan, Irwin King, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00923)  

**Abstract**: The communication topology of Large Language Model Multi-Agent Systems (LLM-MASs) governs how information propagates among agents and encodes critical domain knowledge and expert workflow design. It is valuable intellectual property and plays a decisive role in system utility and safety, yet remains vulnerable to inference attacks that stealthily reconstruct it from agent outputs. Existing defenses detect or mitigate such attacks but do not proactively conceal the topology itself. We propose \textit{Illusion}, a proactive topology obfuscation defense for LLM-MAS. \textit{Illusion} injects fake edges and fake messages to present attackers with a deceptive topology, while selecting low-impact candidates that preserve output-distribution equivalence and task-accuracy non-inferiority. The resulting fake edges are designed to be indistinguishable from genuine messages under black-box observation. Extensive experiments show that \textit{Illusion} reduces edge-level F1 from over 0.87 to 0.62 and graph similarity from over 0.94 to 0.71. It also largely preserves output distributions and task performance, and its fake edges resist both similarity-based and LLM-judge filtering, even under adaptive pruning. 

---
# TreePIN$^{2}$: Tree-Prior Initialized Neural Networks 

**Authors**: Yanrong Wang, Qiming Sun, Haoyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.00919)  

**Abstract**: Fitted decision trees provide predictive knowledge for neural network initialization. We present Tree-Prior Initialized Neural Networks (TreePIN$^{2}$), a method that translates split rules, decision paths and leaf predictions into initial network weights and biases for classification and regression. Sigmoid units encode split responses, hidden units combine them along decision paths, and leaf predictions initialize the output layer. We calibrate split softness using local training margins, sample support and impurity reduction, relating the initial responses to the data at each node. All weights and biases remain trainable, allowing gradient-based optimization to refine the transferred decisions and their combinations. Across six classification and six regression tasks with twenty paired seeds, TreePIN$^{2}$ improves mean initial and early validation performance over initialization baselines on most tasks. After fifty epochs, it achieves the best or tied-best mean validation performance among the six compared methods on ten tasks. Paired ablations show how split calibration and leaf-informed output initialization contribute to the starting predictor and its subsequent learning. 

---
# S1CogBench: What System One Decision Models Can and Cannot Do on Cognitive Experiments 

**Authors**: Dengzhe Hou, Kazunori D Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2611.00911)  

**Abstract**: Language models are no longer used only to generate text. Software also asks them for decisions, in routing, guardrails, and automated gates. System One decision models, also called typed decision models, such as Jev and Laya, return one fast decision over the legal options, leaving no text to inspect. However, a fast, well-formed decision does not show that the model used the evidence. Cognitive experiments can test this, because they vary task demands and measure how behavior changes, yet existing cognitive benchmarks evaluate language and multimodal models. We introduce S1CogBench to ask what these models can and cannot do on 24 cognitive tasks, evaluating five open models and Jev and keeping conditions and scoring apart from disclosed adaptations. We find that decision models vary widely in reliability, acting well does not imply predicting humans, aggregate scores hide how decisions respond to conditions, and speed comes mainly from answering without generation. To our knowledge, S1CogBench is the first benchmark of System One decision models on established cognitive experiments, and it gives a shared test of whether their decisions respond to evidence, feedback, memory load, and future opportunities before software relies on them. 

---
# Clockwork: Scheduling Compound AI Systems in Space Data Centers 

**Authors**: Milos Gravara, Andrija Stanisic, Stefan Nastic  

**Link**: [PDF](https://arxiv.org/pdf/2611.00909)  

**Abstract**: Space data centers built on LEO constellations create an opportunity to serve Compound AI workflows close to where observations are captured. Serving such a workflow for an area of interest (AoI) requires deciding which orbital planes to provision and where to place each workflow component, while orbital motion continuously changes which satellites can serve the AoI. Provisioning must therefore be planned as a sequence of decisions over time, and these decisions are coupled because changing component placements or activating new planes incurs costs. To address this, we present Clockwork, a framework for scheduling Compound AI workflows on space data centers under end-to-end latency SLOs. Clockwork has two components: i) Castor jointly selects active planes and component placements for each constellation snapshot, while ii) Pollux connects Castor's candidate provisioning states across predicted orbital evolution to construct a low-cost schedule over time. On our experimental two-shell constellation, Castor serves 81% of instances at a 300 ms SLO, while Pollux reduces mean total scheduling cost by 56% on average and by up to 63% compared to heuristic baselines. 

---
# Microphone Geometry Aware Regional Target Speaker Extraction 

**Authors**: Shitong Xu, Niki Trigoni, Andrew Markham  

**Link**: [PDF](https://arxiv.org/pdf/2611.00894)  

**Abstract**: Regional target speaker extraction (RSE) aims to extract speech originating from a specified angular region, relaxing the requirement for precise target speaker location when performing location based speech extraction. However, existing RSE methods typically assume a fixed microphone array configuration and microphone number, limiting their applicability to unseen microphone geometries and varying numbers of microphones. In this work, we propose a geometry-aware RSE framework capable of extract with varying microphone array geometry and microphone numbers. Specifically, we introduce an attention-based multichannel front-end to encode inter-microphone phase information, together with a spatial audio reconstruction-based auxiliary objective to improve the front-end spatial representations quality. Experimental results demonstrate that the proposed framework improves SI-SDRi by 5.1 dB over geometry-aware baselines across microphone arrays with four-microphones, while maintaining a performance gap of less than 0.6 dB compared with a tetrahedral geometry-specific baseline when evaluated on the tetrahedral microphone array. 

---
# ShiftRAG: Bypassing the Textual Bottleneck via Decoupled Learning and Continuous Soft Tokens 

**Authors**: Youmin Ko, Jihong Jeong, Hyunjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.00871)  

**Abstract**: Despite advances in retrieval-augmented generation (RAG), suppressing hallucinations during complex reasoning remains a persistent challenge. We identify that existing RAG pipelines suffer from a textual bottleneck: by passing only raw text to the generator, they discard discriminative signals of the retriever. Furthermore, recent attempts to unify retrieval and generation within a single LLM induce severe objective conflict and exacerbate the rank collapse of decoder-only architectures, degrading performance on both tasks. To address these limitations, we introduce ShiftRAG, a decoupled RAG framework that employs a single LLM backbone but isolates discriminative retrieval learning from autoregressive generation. To bridge these decoupled modes, we project retrieval alignments into continuous soft tokens, establishing a high-bandwidth interface that infuses the generator with retrieval-side signals. Concurrently, our soft orthogonality objective mitigates rank collapse, while layer-wise relevance dynamics preserve the fine-grained latent space required for highly discriminative evidence separation. Extensive evaluations demonstrate that ShiftRAG significantly improves retrieval and question answering performance over state-of-the-art baselines. ShiftRAG preserves foundational capabilities of the backbone while maintaining a favorable efficiency-performance trade-off on general text encoding and generative tasks. 

---
# Agent4FreeFem: A Verification-Guided Framework for Autonomous Topology Optimization 

**Authors**: Zhipu Cui, Helin Gong, Shengfeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.00803)  

**Abstract**: Topology optimization has become an indispensable design tool in the aerospace, automotive, and mechanical engineering industries. Yet bringing a new problem from its description to an optimized result remains slow, expert-dependent, and unreliable: the workflow spans many manually connected stages, each new problem requires solver code to be written and debugged manually, and such code often fails or yields incorrect results. We present Agent4FreeFem, a pipeline driven by a large language model that turns a natural-language problem description into a set of verified optimization results without manual solver-code development or debugging. The pipeline keeps the problem definition unmodified, localizes errors using independent sources of evidence, and completes constrained repairs automatically, thereby providing efficient end-to-end automation, autonomous execution, and verifiable results. We evaluated Agent4FreeFem on a suite of single-physics and coupled multiphysics problems. Every case passed runtime verification and finished within the repair budget of each stage. 

---
# PhoneValue-Bench: A Multimodal Benchmark and Datasets for Consumer Smartphone Recommerce Valuation 

**Authors**: Zihan Deng, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.00791)  

**Abstract**: As consumer smartphone recommerce scales, buyers must quote binding repurchase prices from seller-uploaded multi-view photos and diagnostic screenshots before devices reach the warehouse. Accurate remote valuation requires fusing noisy cosmetic evidence with identity, battery, lock, and channel risk from screenshots and staff forms, since an error in any modality changes both grade and payout. However, existing multimodal resources mainly target general VLM competence, screen reading, or factory defect detection, and largely overlook staff-aligned trade-in fields, dual appearance/staff ground truth, and executable valuation scorecards. This gap stems from two challenges: (1) the lack of large-scale real trade-in data with audited dual annotations and a compact ranking benchmark kept separate from training labels, and (2) the need to jointly reason over unordered multi-view cosmetics and diagnostic screens under operational deduction rules. To address these challenges, we propose PhoneValue-Bench, a dual-GT evaluation suite with tier-gated scorecards, together with PhoneValue-Corpus, a matched weakly labeled training set of over twenty thousand orders and hundreds of thousands of captioned images. We further present ValueAgent, a staged perception-fusion-valuation reference pipeline aligned with production inspection, plus an executable public scorer and submission schema. Experiments show that ValueAgent reaches a measured 84.1/100 on Bench-Full, with 97.0 appearance fill and 54.1 staff-field fill, establishing a practical dual-GT benchmark for multimodal recommerce valuation. 

---
# HiPaper: A Human-AI Collaborative System for Academic Literature Triage and Scholarly Sensemaking 

**Authors**: Ye Jia, Pangjing Wu, Hayley Y.W. Gu, Zhuohang Jiang, Xu Yuan, Peter H.F. Ng, Zackary P. T. Sin, Qing Li, Chen Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.00762)  

**Abstract**: Scholarly sensemaking requires researchers to iteratively retrieve, collect, and synthesize literature as their understanding evolves. However, existing scholarly tools provide only partial support for this process. Recent artificial intelligence (AI)-based systems generally rely on opaque submit-and-wait pipelines that it users' ability to inspect, revise, and validate intermediate outputs. In this study, we present HiPaper, a human-AI collaborative literature review system informed by a formative study with academic experts. HiPaper organizes scholarly sensemaking as a three-stage interactive workflow spanning Retrieval, Collection, andSynthesis, centralized on the human-in-the-loop paradigm that exposes intermediate AI-generated content for user inspection and intervention through a persistent workspace called Knowledge Pocket. Through this workflow, users can iteratively refine search intent, curate and organize papers with AI-enriched metadata and clustering, and conduct evidence-traceable synthesis over selected literature. Our user study with 14 Ph.D. students showed that HiPaper efficiently supports scholarly sensemaking by enabling human-AI collaboration in both familiar and unfamiliar domains. Furthermore, the findings highlight how staged interactions with intermediate AI-generated content successfully scaffold user interactions throughout the sensemaking process. 

---
# Old Is Not False: Valid-Time Memory for Interactive Video World Models 

**Authors**: Ziming Wang, Chuyue Gong, Hongliang Cao, Jingmin Zhu, Chutian Wang, Yuheng He, Xiuyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.00758)  

**Abstract**: Interactive video world models draw each frame from a memory bank of earlier frames stamped with their capture times. With frames, weights and noise fixed, reversing the timestamps of two views flips the state that generators of two families draw: they take the newest timestamp as the truth. That reading is right as long as states never return. After an agent undoes an action, the newest view of a region is outdated and the only view of the present is older; memories for changing worlds delete that older view, and memories that keep it hand the generator the past. Temporal databases separate when a record is stored from when its content holds, its valid time. ReDate keeps every frame and dates each retrieved token by its valid time, read from the agent's action log. The frozen generator then reads a reverted region as one that has just reached a new state, so ReDate needs no training. We prove that trusting the newest timestamp is right exactly when every current view is dated after every outdated one, that without a view taken after the return only the action log tells a reverted region from an up-to-date one, and that a generator able to draw from its own prediction gains more from re-dating than from deletion when the moved tokens keep little weight. After an undo, ReDate raises the share of FAR's steps showing the current state from 0.36 to 0.62 and HY-WorldPlay's from 0.003 to 0.19. We also propose a calibrated state metric to test any memory after undos. 

---
# ResMamba: A Dual-Path Convolution-Mamba Model for Music Genre Classification 

**Authors**: Yingyu Zhang, Pu Qu  

**Link**: [PDF](https://arxiv.org/pdf/2611.00746)  

**Abstract**: Music genre classification remains a fundamental yet challenging task in music information retrieval, requiring models that capture local time--frequency patterns and long-range temporal dependencies. To address this, we propose ResMamba, a hybrid architecture that augments a ResNet-style four-stage backbone with Mamba blocks. Building upon the selective structured state space sequence model, ResMamba first adapts the selective SSM into a 4D-compatible residual Mamba block that captures global dependencies at linear complexity. We then design the ConvMamba dual-path block, which combines this adapted Mamba branch with a parallel convolutional branch through a learned $1\times1$ fusion, jointly modeling global sequential context and local spectro-temporal patterns. Furthermore, we adopt a multi-scale fusion head that aggregates hierarchical representations across all four stages. Experiments on the GTZAN dataset demonstrate that our full ResMamba model achieves a 5-fold cross-validation accuracy of $88.99\% \pm 1.58\%$, surpassing the strongest published method in our comparison (88.64\%). All models are trained from scratch without any pretraining, external data, or data augmentation. Beyond GTZAN, ResMamba also generalizes to the larger FMA benchmarks under the same training protocol, attaining $64.52\% \pm 1.88\%$ on FMA-Small under 10-fold cross-validation --- the best published from-scratch result --- and $65.75\%$ on FMA-Medium under the official artist-level split, surpassing the widely cited SVM baseline of Defferrard \textit{et al.} (63\%). 

---
# Beyond Synthesis Conditions: Linking In-situ Growth Dynamics to Carbon Nanotube Forest Properties through Multimodal Machine Learning 

**Authors**: Shun Muroga, Takashi Tsuji, Shunsuke Sakurai, Don N Futaba  

**Link**: [PDF](https://arxiv.org/pdf/2611.00739)  

**Abstract**: Machine learning for chemical vapor deposition (CVD) of carbon nanotubes (CNTs) has largely mapped static synthesis conditions to final properties, implicitly treating these conditions as a complete description of growth. Here, we test this assumption by integrating gas-flow conditions with in-situ time-resolve growth profiles (growth curves) of vertically aligned CNT forests. We introduce an observation-window analysis, in which prediction is evaluated as a function of observation time by providing models with the growth curve only up to progressively later times, to determine not only whether a property can be predicted, but when the necessary information becomes available during synthesis. Clustering of normalized growth curves revealed distinct growth patterns independent of final forest height. The gas conditions alone could predict the initial growth rate but not the growth lifetime and failed to capture the variation in yield (R^2<0). Adding the growth curve improved yield prediction (R^2~0.4-0.5), with the relevant information emerging during the intermediate and later stages of growth rather than at its onset. In contrast, density and G/D ratio, which forest height does not record, showed no consistent improvement. The maximum height and lifetime could be estimated with moderate accuracy midway through the 10-min growth. Permutation feature importance identified water vapor and H_2, both linked to catalyst activity, as the most informative gas variables. These results indicate that variation unexplained by synthesis conditions is not merely noise but reflects an evolving catalyst state, and they establish mid-growth decision points for autonomous synthesis, enabling early termination of unpromising batches and, ultimately, in-batch adjustment of growth conditions. 

---
# Architecting Fairness-Aware LLM Interactions 

**Authors**: Alessandra Parziale, Valeria Pontillo, Fabio Palomba, Patrizio Pelliccione  

**Link**: [PDF](https://arxiv.org/pdf/2611.00723)  

**Abstract**: Large Language Models (LLMs) are increasingly used as decision-support tools but may generate stereotypical, discriminatory, or unfair responses. As LLMs become embedded in decision-critical software systems, fairness can no longer be treated as a downstream evaluation step, but must become an architectural concern of the interaction itself. Yet current fairness approaches operate as standalone evaluation techniques, lacking a reusable mechanism to verify and mitigate violations at runtime. We propose a fairness-assurance architecture grounded in three reusable design patterns, instantiated as a multi-agent framework that evaluates and iteratively refines LLM responses before delivery. A preliminary evaluation across eight configurations involving different LLM families shows that the architecture identifies fairness violations and iteratively produces compliant responses in all configurations, with behavior differing depending on the model-family composition. These results provide initial evidence for the joint instantiation of the proposed patterns and a first step toward architecting non-functional properties into LLM-based systems. 

---
# SAVIER: Source-Anchored Visual Inspection and Evidence-Guided Repair for Vietnamese Cultural Text-to-Image Generation 

**Authors**: Gia-Huy Thai, Duc-Tien Khong, Trung-Nghia Le  

**Link**: [PDF](https://arxiv.org/pdf/2611.00722)  

**Abstract**: We present SAVIER, a training-free, one-pass framework that repairs the depiction of a named Vietnamese cultural entity in a generated image while preserving the scene details requested by the user. The problem is hard because a plausible scene can miss the cues that identify the entity, and prompt-level refinement never inspects the image it produces. SAVIER starts from a Culture-TRIP expanded prompt; a Curator separates source-linked cultural cues from prompt-preservation requirements, a prompt-blind Observer describes the first image, and a Refiner turns the discrepancies into at most three positive instructions before one regeneration with the same seed and a fixed bank of reference photographs. On the 50 bilingual prompts of ViCuP, evaluated with SDXL and FLUX.1-dev against a frozen, human-verified visual contract, SAVIER raises contract fulfillment from 55.6 to 69.6 and from 59.1 to 67.3 over Culture-TRIP and raises CAIRE cultural relevance on FLUX.1-dev. A reference-only ablation that removes the agent repair shows that SAVIER's photographs account for most of the contract gain; the agent roles add country-level relevance on FLUX.1-dev without lowering adherence to the original prompt under the official VQAScore. Project page: this https URL 

---
# Getting a Grip: Operating Human-Agent Software Teams Through Governed Messages 

**Authors**: Michael Granitzer, Kanishka Ghosh Dastidar, Andreas Einwiller, Jelena Mitrović, Saber Zerhoudi  

**Link**: [PDF](https://arxiv.org/pdf/2611.00689)  

**Abstract**: Software is increasingly developed by teams in which several people direct many coding agents from different vendors. Such teams must establish who may request what from whom, which messages start work, which decisions bind, and who caused a change. We propose to govern a human-agent team through its message protocol and present yoars, a relay in which humans and agents are authenticated principals that exchange typed, addressed messages in rooms holding goals, decision rules, routing and an attributable, append-only record. This report gives the architecture, the full controlled study and the field material. In 926 controlled runs (899 analyzable), small agent teams built programs under varied room rules, some with a hostile member planting a backdoor. Revising which messages start work, bundled with limits on repeated messages, was followed by a drop in total coordination failure from 42% to 13% (p = .0063, adjusted p = .057 over nine registered tests). Instructed roles and votes did not reliably produce the intended process, and no defense inside the conversation reliably prevented the backdoor. An auditor read the records of 96 runs with a confirmed backdoor. In the 25 runs where the attacker had persuaded a teammate to write it, the auditor named the persuader in all 25, but it also flagged 24 of 40 honest runs. In a four-week field study, up to nineteen agents and two people developed a distributed system through yoars, producing 7,457 acts. Human attention and decision making emerged as the critical resource: requests waited mainly for the human-run coordinator and for approvals, while agents answered within minutes. Most governance requirements emerged from operation, and holding context dominated token cost. We report twenty-six failures of our own instruments, derive seven lessons and a checklist, and release the corpus, the redacted room ledger and the analysis scripts. 

---
# Precedent Labels and Facts Shift LLM Approval Decisions in Opposite Directions 

**Authors**: Junwei Yao, Zike Yang, Yongxi Zhou, Yanxian Chen, Jianan Liu, Jiaxi Wen, Zhijun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00672)  

**Abstract**: When a language model decides refund or claim requests under written policies, the system can show it similar past cases. Each past case has two parts: its facts and the decision recorded for it. We separate the two. A model sees a policy, a request and three past cases, and we change either the recorded decisions or the facts, keeping the request fixed. We select requests that are close calls: deciding without past cases, DeepSeek V4.1 Flash approved each in some of four tries and denied it in others. When the past cases are near-copies of the request (its facts with a few details changed), switching their recorded decisions from DENY to APPROVE raises approval by about 30 to 55 percentage points in DeepSeek V4.1 Flash, GPT-5.6 Luna and Claude Haiku 5.5 on policies written for the study, and DeepSeek and Luna show the same on nine public policies paraphrased from real ones. The facts push the other way: shown without their recorded decisions, past cases whose facts support approval make these two models approve less often than past cases whose facts support denial. One reading is that the request looks weaker next to them. This push is not just a correction toward the right answer: in an additional DeepSeek analysis, it was of similar estimated size on requests that human annotators judged approvable and on those they judged deniable. In a separate DeepSeek run on the public policies, when the past cases are near-copies of a different request, switching their decisions moves approval by 9 points instead of 43, while facts that support approval still lower it by 22 points against facts that support denial. Systems that retrieve past decisions should test the two parts separately. 

---
# Selecting LLM-Generated Hypotheses Raises Truth Beyond What an LLM Panel Expects: A Pre-registered Study on NAS-Bench-201 and HPOBench 

**Authors**: Shanglin Yang, Zhan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.00643)  

**Abstract**: LLM research-ideation systems aim for hypotheses that are both true and non-obvious, yet neither property can normally be checked. We build a testbed in which both can. Generators propose conditional interaction claims about two fully tabulated benchmarks, NAS-Bench-201 and HPOBench; every claim is verified exactly against the tables, and its non-obviousness is scored against a calibrated blind prior elicited from other LLMs. Excess truth measures how much more often a run's claims hold than this panel expects. Across a series of pre-registered stages with Claude and GPT generators, our main positive result is a simple two-stage step found by a harness optimizer: draft a large pool of claims, then select a subset in a separate call. For Sonnet on NAS-Bench-201, this step raised excess truth in every pre-registered test, including against a same-pool ablation that keeps only likely-true claims, without lowering precision. The gain comes both from credit for true claims and from avoiding near-miss false ones. In post hoc checks it held under every calibration, panel and truth rule we tried, and it exceeded random and type-filter selection from the same pools. Because kept claims recur across runs, the evidence weakens when runs and claims are resampled jointly, and it is not significant against Sonnet's own prior. Two secondary findings are also pre-registered. GPT makes real use of the results of cheap probes, beating shuffled and withheld controls, though not more efficiently than asking for more claims. Optimizing the harness for an LLM judge raised judge scores while sharply cutting precision for every generator. 

---
# Log-Driven Semantic Tuning for Mixed Integer Programming Solvers 

**Authors**: Zhengzhong Ricky You, Kaizhao Sun, Zeping Min, Wotao Yin, Xinshang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00621)  

**Abstract**: Automated configuration of mixed-integer programming (MIP) solvers largely relies on black-box optimization. Each trial requires an expensive solver run and provides only a scalar outcome, so tuning can require many such evaluations. Human experts take a different approach. They inspect solver logs, identify performance bottlenecks, and adjust the parameters that address them. We propose Log-Driven Semantic Tuning (LDST), which uses a large language model (LLM) to automate this process. LDST extracts compact diagnostics from solver logs, uses the LLM to propose sparse edits to the current best configuration based on parameter meanings and observed solver behavior, and validates each proposal before execution. A regime-adaptive objective selects the tuning signal according to the information available from a time-limited run, while a log-based completion signal estimates completion time from branch-and-bound tree growth recorded in ordinary solver logs, so that solved and unsolved runs share one time scale. On 215 instances from five benchmark sets, LDST reduces geometric-mean loss by 43.9% on easy instances and 16.7% on intermediate instances relative to the best of SMAC3, RBFOpt, and random search. In both regimes, LDST achieves a lower loss with 10 proposals than any of these baselines achieves with 50. Configurations that LDST tunes on a single instance also transfer better than baseline configurations to unseen instances of the same family. 

---
# Discovering Models for Experimental Quantum Systems with Agentic AI 

**Authors**: Hannah C. Kleidermacher, Bennet Windt, Miguel Bello, James Zou, Jelena Vučković  

**Link**: [PDF](https://arxiv.org/pdf/2611.00620)  

**Abstract**: A central obstacle in realizing color center cavity-QED-based quantum technologies is inhomogeneity, or disorder, among individual emitters. As the number of emitters increases, so does the complexity of the experimental data, making it difficult to quantify the present physical mechanisms of disorder. Here we introduce an artificial intelligence-driven model discovery framework in which a large language model (LLM)-based agent, equipped with tools and a structured workflow, proposes candidate Hamiltonian and Lindbladian models, generates computational tools, and quantitatively evaluates competing physical hypotheses. We apply the workflow to three previously published experimental cavity-QED datasets collected from color centers in diamond and silicon carbide. Across the experiments, the agent successfully identifies physical models for the system and characterizes the disorder. In assessing reproducibility, we find that run-to-run variability is concentrated primarily in the research plan generation step, while the computational analysis itself is substantially more consistent. Our results suggest that structured LLM agent workflows can support model discovery from quantum optical experimental data while simultaneously allowing for human expert supervision and collaboration. 

---
# Learning from an Imitable Teacher: Competition Solution for the MetaFine Challenge 

**Authors**: Yanze Zhang, Hao Liu, Haochang Hao, Zhixiang Shen, Chunyu Miao, Yuyao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00615)  

**Abstract**: The MetaFine Open Competition (RoboFineMani workshop, CoRL 2026) evaluates fine-grained manipulation beyond binary success. Its five simulated tasks, from grasping a bottle by an instructed part to inserting letter-shaped pegs with 4 mm clearance, are scored on perception under lighting and camera changes, instruction understanding, and behavior. This report presents Team Friday's solution, which got second place with an overall score of 0.947, more than twice the $\pi_0$ baseline. Instead of fine-tuning a large vision-language-action (VLA) model, we train one 19M-parameter policy per task with a separate instruction classifier and focus on the data, all generated in the simulator by a privileged scripted teacher, including its corrections of the student's mistakes. By analyzing failed episodes, we find that a teacher that almost always succeeds can still teach a memoryless student badly: its pauses become stalls, its fixed habits become shortcuts, and it never shows the errors the student makes. Making the teacher \emph{imitable} raised the official score on letter insertion, the hardest task, from 0.44 to 0.82 with 32\% more data and the same architecture, and finally to 0.88. We summarize practical lessons on demonstration data and evaluation, which we hope will help in imitation learning for precise manipulation. 

---
# Words Before Pixels: Selective Modality Routing for Vision-Language Model Unlearning 

**Authors**: Laura Yao, Haochen Zhang, Jinhao Duan, Sijia Liu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.00608)  

**Abstract**: Vision-language model (VLM) unlearning is often treated as an objective-design problem: given multimodal forget data, the goal is to optimize a loss that removes undesirable behavior while preserving retained capabilities. We argue that this view overlooks an equally important data-centric question: how should each forget example be represented before optimization? Existing methods typically keep image-conditioned failures in their original image-text form, implicitly assuming that pixels are the right route for unlearning whenever pixels appear in the input. We challenge this assumption by constructing modality-decomposed versions of VLM benchmarks, enabling controlled comparisons among image-only, text-only, and multimodal representations of the same examples. Our modality-mixing experiments show that unsafe instruction-following behavior is more effectively suppressed through textual representations than through images alone, while purely text-only unlearning can still weaken visual grounding when the target behavior depends on visual evidence. Motivated by this trade-off, we introduce ModRoute, a gradient-guided per-sample modality routing strategy that selectively converts high visual-pressure examples to text-only form while keeping the remaining examples multimodal. Notably, on VLGuard with LLaVA-1.5-7B, ModRoute has the strongest composite unlearning-utility score of 0.847 at $\gamma=0.5$, a 2.5% improvement over the best random-switching score and $\geq$46% better than pure multimodal or text unlearning. Overall, our results show that effective VLM unlearning depends not only on the objective, but also on the composition and per-sample modality routing of the unlearning data. 

---
# Sparse Autoencoder Attribution Graphs Are Decided at the Margin 

**Authors**: Karl Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.00606)  

**Abstract**: Attribution graphs over the features of sparse autoencoders (SAEs) are used to predict what an ablation does and are compared across training seeds. A TopK SAE keeps the largest preactivations at each token, and a feature's margin is its preactivation's distance to the cutoff. On five public TopK SAE pairs from three Pythia models, zeroing one feature moves a next layer feature across the cutoff in 0.74 of cases or more, and the crossings carry most of the response's energy. Over writes that scale a feature's code by three factors, the graph built on the features active beforehand has a median relative error of 0.247 or more. The margin graph adds the first order change to every preactivation and reranks, at the same cost, and errs by at most 0.033. At the median, the crossing features' decoded change is 1.24 times the model's own change in norm or more. Between seeds of one testbed, sparse maps between features explain 0.85 of a graph's energy on its active features, and 0.47 once the other seed's graph keeps only its own. The largest model studied is pythia-160m. 

---
# Multi-Estimate Elicitation Does Not Uniformly Debias Language Model Judgments 

**Authors**: An-Sheng Jhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00597)  

**Abstract**: Does eliciting five weighted estimates within a single response change how a language model responds to information? Across anchoring, hindsight, base-rate, conjunction, and status-quo/default tasks, we compare direct numerical answers with a multi-estimate response format, contrasting source framings, explicitly random cues, and probability-consistency constraints. We evaluate three completed configurations--gpt-6.1-sol, claude-sonnet-5.5, and Qwen3-8B (reasoning disabled, analyzed descriptively). In the first two configurations, direct elicitation yielded near-zero irrelevant-anchor indices and status-quo effects, while remaining responsive to informative sources and informed advice. Of 23 preregistered confirmatory tests, six rejected after Holm adjustment; however, all six absolute point estimates fell below their prespecified smallest effect sizes of interest without establishing equivalence. Moreover, while separate versus joint elicitation shifted conjunction judgments, interpretation is constrained by between-call noise. Ultimately, multi-estimate elicitation failed to produce uniform debiasing or response attenuation across tasks and models, though post hoc correlations between candidate spread and anchoring error motivate independent predictive validation. Materials: this https URL 

---
# Guarded Compression of Adam Optimizer State with Shared-Gradient Bounds 

**Authors**: Annan Zhang, Shengjie Song, Wenshuai Yao, Lihao Zhang, Tianmiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.00589)  

**Abstract**: Adam keeps two FP32 moments per parameter, twice the bytes of the FP32 weights, yet compressing them is risky: the update divides by the square root of the second moment, so small reconstruction errors can change the restored computation substantially. We study guarded moment compression, in which a coordinate's decoded moments are accepted only if they reproduce the native FP32 Adam update to within a tolerance $\tau$, in units of the learning rate, for every gradient in a clipped domain; all other coordinates are stored exactly. The natural certificate, which ranges each update separately over gradient intervals, fails: on one million trained OPT-125M coordinates it escapes 98.9% of coordinates and the payload grows to 126.6% of raw FP32. Our key observation is that both updates consume the same gradient and parameter. Bounding their difference directly, with FP32 rounding-error allowances and a fixed mixed linear-logarithmic gradient partition, cuts escapes to 0.53% and the payload to 28.3% at $\tau=0.01$. Across 120 complete-checkpoint restores of OPT-125M and ResNet-18, with four trained seeds and three training fractions per model, guarded 8-bit log codes shrink moment payloads by 69.5% and 71.9% relative to FP32, 40.8% and 45.4% relative to equally guarded BF16, and 29.3% and 37.6% relative to equally guarded blockwise 8-bit, and every restored first update stays within tolerance. Encoding and restoration together take 2.3-2.6$\times$ as long as guarded BF16 but less time than guarded blockwise 8-bit. Searching both moment codes jointly saves under 0.001% more bytes at 6% more codec time, so per-moment rounding suffices. The guarantee covers the first restored update under a pinned Torch operation model; we measure subsequent training empirically. 

---
# Conic Certification of Transformer Attention 

**Authors**: Masoud Ataei  

**Link**: [PDF](https://arxiv.org/pdf/2611.00585)  

**Abstract**: Attention is the computational core of large language models, yet the verifiers that bound how far an attention head's output can move under input perturbations give no guarantee on how loose their bounds are. A weak certificate therefore cannot be told apart from a fragile model. We develop a conic programming theory of attention certification that supplies this guarantee. We prove that the difficulty lies not in the softmax but in the bilinear interaction of a varying query with the keys. This interaction makes certification NP-hard over boxes and inapproximable once the attention logits can move by a constant amount. A fixed query or listed token substitutions, by contrast, make the exact worst case efficiently computable. An exponential cone moment relaxation then certifies a head with an error bounded quadratically in the perturbation radius by computable quantities, an order that cannot be improved. We also show that relaxations that treat a network one operation at a time, linear bound propagation among them, pay on second-order terms a sum of per-operation maxima instead of one coupled maximum. Computational results on GPT-2 indicate that the theory carries over to a trained model. Against exact worst cases with the query fixed, the certificate attains the predicted second-order accuracy. It is tighter than linear bound propagation on every head, and the looseness of bound propagation tracks the predicted per-operation excess. 

---
# The First-Order Value of Free Covariates in Budgeted Risk Monitoring 

**Authors**: Yanze Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.00576)  

**Abstract**: A deployed model sees its inputs for free, but checking whether its risk has risen requires costly labels. We study how to spend a label budget so that a risk increase is detected quickly while false alarms stay controlled, including under input shifts that keep the risk below a safety threshold $\tau$. Let $A$ be the false-alarm control scale and $h(r)=\mathrm{kl}(r,\tau)$ the Bernoulli KL divergence between the changed risk $r$ and $\tau$. At label rate $b=c/\log A$, we prove that every rule that chooses requests from its own label history has worst-case (Lorden) delay at least $(1/h(r)+1/c)\log A$ to first order: besides collecting evidence, it must wait for its next label. The bound needs only a label budget on one finite stretch of safe source data and false-alarm control over an initial window; a bound on the average time to false alarm does not suffice. When a harmful change also shifts the inputs, a rule that lets the inputs trigger labeling attains delay $\log A/h(r)$ to first order on a fixed Gaussian model, so free covariates save at least $1/b$ rounds. A hybrid rule adds a label-history fallback for risk changes that leave the inputs unchanged. Synthetic experiments and two ACS tasks, with strengthened label-only baselines, show the predicted pattern. The covariate trigger has a cost: because its budget is calibrated on the source, a benign input shift can make it request nearly every label. 

---
# Faster Imitation by Re-timing Demonstrations Within the Controller Limit 

**Authors**: Zongtan Li, Tianming Liu, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.00539)  

**Abstract**: Imitation policies inherit the operator's pace, and re-timing the demonstrations to run faster works only until success collapses at two to three times demonstration speed. Recent methods attribute that collapse to distribution shift and answer it with finer rules for which segments may be sped up; we show it has a mechanical cause that segment selection cannot address. A low-level controller realises only a bounded displacement per control period, so merging consecutive commands produces sums outside that bound: training pairs that promise a transition the plant cannot deliver. The share of such pairs in the re-timed data, which we call the exceed share, tracks where the retrained policy collapses. The rule that follows has a single parameter: merge greedily while the sum stays realisable. Every command is then realisable by construction, the demonstrated geometry is preserved, and the speed-up lands only where the demonstrator left slack. On four simulated robomimic tasks, 200 held-out seeds per cell, a merge cap of two is at or above the frozen policy's success on all four, by 14 points on Square, and shortens the successful episodes. At matched tempo, uniform re-timing loses 9 and 16 points on Can and ToolHang, is within noise on the other two, and reaches zero on the hardest at 2x. Placement also matters: no shuffled, entropy-guided or speed-conditioned alternative beats the rule on three of the four tasks, by margins the exceed share alone does not explain; on the fourth, at a tempo the task absorbs, every placement lands in the same band. The same principle carries to a simulated humanoid whole-body tracker: re-timing 429 reference motions within its velocity envelope succeeds on 71% of clips at 2.3x, against 51% for uniform time scaling at 2x. 

---
# Reflex: Fast Robot Policy Serving by Reusing and Correcting Past Actions 

**Authors**: Ziyang Lin, Zixuan Sun, Rami Nasser, Jianli Jin, Xuanle Li, Jayanth Srinivasa, Fan Lai  

**Link**: [PDF](https://arxiv.org/pdf/2611.00524)  

**Abstract**: Modern robot policies often perform expensive inference to generate action chunks, making serving a latency bottleneck for real-time control and a throughput bottleneck for multi-robot serving. Building on the success of caching in image, video, and audio generation, recent robot-policy advances have reported promising speedups by reusing intermediate denoising states to avoid redundant computation. Surprisingly, we find that such reuse often offers little advantage over simply running fewer denoising steps from fresh noise, without any cache. Moreover, accelerating denoising leaves the expensive policy backbone intact, which can account for 70% of GPU time.
We present Reflex, a serving layer that turns past policy executions into a backbone-free path for future decisions. Rather than reusing intermediate computations, Reflex caches complete action chunks to bypass both the backbone and action head. This introduces two challenges: similar observations can demand different actions, and small action deviations (errors) in reuse can compound into execution stalls. Reflex addresses these through two-level cache correction. At the action level, it learns action-aware similarity from past policy decisions, assembles compatible chunks, and applies lightweight residual correction to limit action mismatch. At the execution level, it captures task progress and adaptively refreshes the original policy to limit accumulated errors and balance inference cost. Without modifying the underlying policy, Reflex improves inference latency by 1.4x to 11x and throughput by 2.0x to 2.2x on representative workloads and real robots, while preserving comparable task success. 

---
# AStockArena: Beyond Returns, Can We Entrust Real Capital to the Decisions of Financial Agents? 

**Authors**: Zeyu Liu, Sing Kwong Lai, Xuanzhi Feng, HaoYang Shang, Yang Li, Zihao Fang, Xuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.00515)  

**Abstract**: Large language models (LLMs) are increasingly used in dynamic financial trading, yet real-time trading benchmarks mainly evaluate agents by cumulative returns and risk-adjusted performance. Such outcome metrics cannot show whether trades are faithfully reported, supported by available evidence, and consistent with account states. We introduce AStockArena, a real-time decision-auditing benchmark for continuous trading in China's A-share market. Under shared market conditions and execution constraints, we evaluate 13 open- and closed-source models over 3,198 sequential decisions. AStockArena reconstructsthe actions actually posted to accounts by aligning reports, trade requests, execution responses, and account states, and audits them with eight checks spanning execution traceability (O), evidence sufficiency (RK), and financial meta-awareness (B), combining record verification with consensus expert annotation. Of 1,070 trading points, 72.99% exhibit at least one audit gap; model-level rates range from52.2% to 91.2%, with evidence gaps most prevalent. Historical trade-filtering replay further shows thatportfolio outcomes are sensitive to these flagged trades. Decision-process auditing thus offers a verifiable complement to return-based evaluation of financial agents. 

---
# Do Social Influence Pressures Move LLM Agents? Evidence from Delegated Audit Judgment 

**Authors**: Richard Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00508)  

**Abstract**: Large language model (LLM) agents are rapidly being integrated into complex professional workflows such as auditing and entrusted with judgments under human review. Human auditors are known to yield to social influence pressures from superiors, peers and clients, and this paper asks whether LLM agents yield to the same pressures and, if so, how they respond. In an experiment on twelve model settings, an LLM agent acting as a senior associate in audit analytical review forms an estimate and is then challenged without new evidence by colleagues inside the firm, by a reviewer from another firm and by the client's CFO, with and without a career inducement. Experimental findings show that LLM agents yield to pressure from the engagement partner, from peers and from a reviewer from another firm, but in most settings not to pressure from the client's CFO. Under a career inducement the LLM agents revise their estimates less, not more, and they name the inducement as a threat to independence. Furthermore, some models revise only toward the more conservative estimate, while others revise in both directions. LLM agents thus do not respond to social influence pressures as human auditors do, and LLM-based decision support systems should be designed with the differences in mind. 

---
# Spotting the Artificial: Dialogic Humor Improvisation in Humans and LLMs 

**Authors**: Lizeth Palacios Patiño, Ximena Gutierrez-Vasques, Daniel Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2611.00492)  

**Abstract**: We propose a methodology for investigating the differences between natural and synthetic language in a challenging setting: humorous improvisational dialogue. Our experimental setting compares human-produced comedy improvisations with the dialogues generated by LLMs in a multi-agent environment that we designed. The analysis combines automated linguistic measures with manual annotation grounded in Systemic Functional Linguistics (SFL).
The results show that, although LLMs can create coherent and creative scenes, they often fail to reproduce key characteristics of spontaneous spoken interaction. In addition, LLM-generated interactions display a strong tendency toward politeness and positive evaluations, contrasting with the broader range of attitudes found in human humorous improvisation. Our evidence highlights both the strengths and limitations of current LLMs in modeling dialogic humor and demonstrate the value of integrating linguistic theory with NLP methods for the analysis of machine-generated language 

---
# Threat Modeling Intimate Partner Abuse in Agentic Systems 

**Authors**: Natalie Grace Brigham, Eugene Bagdasarian, Franziska Roesner, Tadayoshi Kohno  

**Link**: [PDF](https://arxiv.org/pdf/2611.00489)  

**Abstract**: Agentic systems are AI-powered technologies capable of autonomous action within digital environments, such as sending emails or making reservations. They are deeply connected to external tools as well as users' data and resources, making them vulnerable to attacks which exploit their access and capabilities. In this work, we apply the technology-facilitated abuse threat model, in the context of intimate partner violence and surveillance, to these systems, revealing threats that require deliberate investigation and mitigation. We categorize attacks based on system ownership: (1) Abuser-as-owner attacks demonstrate how agentic systems make existing forms of technology-facilitated abuse far easier and more scalable. (2) Victim-survivor-as-owner attacks show how agentic systems create a new attack surface which abusers can exploit. In light of these risks, we outline research and development directions for effective mitigation strategies. 

---
# From Lookups to Lessons: Turning Retrieved Knowledge into Parametric Memory 

**Authors**: Changjiang Han, Bowei He, Ye Yuan, Weixu Zhang, Haolun Wu, Xue Liu, Jikun Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00450)  

**Abstract**: Retrieval-augmented generation (RAG) helps large language models (LLMs) answer factual questions beyond their training data, but it adds storage and inference costs. Distilling previously retrieved information into parametric memories enables reuse without online retrieval. However, training the memory to mimic a retriever rewards accurate independent predictions, which do not necessarily improve the final output quality when combined with the LLM's existing predictions. To address this, we introduce ResMem, in which the LLM reads retrieved text offline to generate teacher targets. This design avoids storing massive token representations, unlike existing parametric memory approaches such as MLP Memory, which relies on k-nearest neighbor (kNN) datastores. Furthermore, by using the LLM as an offline teacher that reads privileged retrieved evidence alongside the input, we train a parametric memory to learn an additive logit residual. This residual aligns the predictions of a student sharing the same backbone with the teacher distribution, while self-distillation updates remain confined to the parametric memory. On a Wikipedia subset, ResMem reduces the storage retained for teacher construction by 94.7% compared with the evaluated kNN pipeline. Across five question answering datasets, ResMem raises mean token F1 from 32.83% for the frozen LLM to 34.48%. It also breaks 21% fewer of the LLM's correct answers than a kNN-trained memory. These findings validate our motivation of treating retrieval as a learning resource, with parametric memory shaped by how retrieved knowledge is used to inform the model's future predictions. Code and model weights are available at this https URL and this https URL. 

---
# Learning to evolve by matching velocities 

**Authors**: Nicola Farenga, Andrea Manzoni  

**Link**: [PDF](https://arxiv.org/pdf/2611.00448)  

**Abstract**: Learning the continuous-time evolution of nonlinear dynamical systems relies on trajectory-based objectives, which require simulating the learned dynamics during training and backpropagating through the numerical scheme, at a cost that grows with the rollout horizon. We show that trajectory-based objectives degenerate as the rollout horizon increases, hindering gradient-based optimization, and that the numerical scheme biases the learned dynamics by constraining the spectrum of the learned Jacobian to its absolute stability region. To overcome these limitations, we introduce Velocity Matching (VM), a simulation-free objective that relies on piecewise-linear interpolation and stochastic collocation, and approximates the velocity population risk to second order in the sampling step. We develop a theoretical framework for velocity-based learning, introducing stability estimates that control trajectory errors in terms of velocity errors, which enable us to establish universal and continuous-time approximation results when the learned velocity field is parameterized by deep neural networks. Building on a bias-variance decomposition of the VM objective, we show that its higher-order approximation of the velocity population risk yields a consistent minimizer; that it relates to the residual of an A-stable continuous-stage Runge-Kutta scheme, thus connecting VM to implicit integration without requiring any implicit solve; and that it implicitly matches the action of the learned Jacobian to that of the true one along the observed trajectories, without requiring access to the underlying velocity field or its Jacobian. Numerical experiments on nonlinear ordinary differential equations and semi-discretized time-dependent partial differential equations show that VM improves accuracy over trajectory-based objectives, while avoiding both the cost of simulation and the instabilities of long rollouts. 

---
# Analog and hybrid computing in the 21st century 

**Authors**: Bernd Ulmann, Olivier Bournez, Corey J. Maley, Zhong Sun, Grace Li Zhang, Paul Manea, Michael Döll, Frank Delvigne, Alex J. H. Fedorec, Daniel Garza, Lucas Wetzel, Sara Obergassel, Pavol Pitonak, Christopher Jung, Daniel Thuerck, Mason A. Porter, Marcelo J. Rozenberg, Frederic Bergner, Bhaskar Choubey, Manyu Zhao, Ludovico Minati, Lucas Böttcher  

**Link**: [PDF](https://arxiv.org/pdf/2611.00446)  

**Abstract**: Analog computers solve problems through physical representations of mathematical variables and relations, often using the dynamics of electronic circuits and their associated signals. Many analog computers directly represent nonlinear dynamical systems; they encode their solutions in the signals that they generate. Although most modern scientific computing relies on digital architectures with inherently discrete-valued representations and algorithmic approaches, the design constraints of such architectures increasingly limit both performance and energy efficiency. Analog systems can circumvent some of these constraints; they execute specific computations very efficiently by mapping them directly onto physical processes using smaller and simpler computing elements than the ones in digital architectures. In this article, we provide an interdisciplinary perspective on modern analog and hybrid (i.e., analog-digital) computing. We discuss how modern scientific computing workflows can leverage analog approaches in various applications, such as in simulating dynamical systems, performing optimization, and enabling more energy-efficient implementations of artificial intelligence. 

---
# Fidelity Collapses with Table Size, Not Turn Count: Delivery Architecture in a Deployed Tool-Using Agent 

**Authors**: Tung Thanh Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00442)  

**Abstract**: When a multi-turn agent delivers a tool's structured records into the model's context and re-emits them each turn, every turn can corrupt data the tool computed correctly. We measure this in a production award-travel agent: two architectures over identical frozen payloads, lineage-matched prompts and sampling, differing only in delivery architecture. SYNTHESIS feeds full record sets into the planner's context and answers follow-ups in-head; BYPASS, the deployed system, keeps records in session state, translates follow-ups into arguments for a deterministic filter, and delivers rows over a typed side channel. Across 24 dialogues (339 scored turns per condition), BYPASS scores 92.3% of turns fully correct, SYNTHESIS 45.7% (paired gap +46.6pp [+35.7, +57.1]). Varying delivered-table size by capture date, SYNTHESIS collapses from near-perfect at 10 rows or fewer to 8% or less beyond about 74 rows. On a content-controlled prefix ladder the collapse threshold orders by model capability, from under 10 rows (GPT-4o-mini) to about 198 or more (Gemini 2.5 Pro), while BYPASS stays size-flat for all four models. The collapse is expensive: beyond 100 rows, where almost no SYNTHESIS answer is correct, each correct one costs roughly 200x BYPASS's flat \$0.012. Practitioners converged on typed side-channel delivery for rendering, not correctness; we show it was, unknowingly, also a correctness decision. 

---
# Online Fair Division of Chores with Upgrade Recourse 

**Authors**: Yichen Huang, Nicholas Teh  

**Link**: [PDF](https://arxiv.org/pdf/2611.00432)  

**Abstract**: We study the online fair division of indivisible chores among $n$ agents under upgrade recourse, where earlier assignments may be revised but each chore previously assigned to an agent must be replaced by a distinct chore that she finds no more costly. We show that a simple adaptation of round robin maintains envy-freeness up to one chore (EF1) together with a $(2-1/n)$-approximation of the maximin share (MMS), and that no deterministic algorithm without predictions can do better, even when all agents rank the chores in the same order. This stands in contrast to the allocation of goods, where the optimal MMS factor vanishes as $n$ grows and cannot be achieved together with EF1. Moreover, knowing only the number of chores modulo a period of length $O(n)$ (and nothing about the costs) suffices to improve upon this factor at the final configuration, while the same algorithm retains a factor below $3$ even if this information is wrong. On the other hand, envy-freeness up to any chore (EFX) essentially requires the exact number of chores: uncertainty between two consecutive values already rules out every finite approximation, and even exact knowledge does not allow EFX to be combined with an MMS guarantee better than the trivial factor of $n$. 

---
# A Spectral Renormalization-Group View of Learning 

**Authors**: Charles H. Martin  

**Link**: [PDF](https://arxiv.org/pdf/2611.00422)  

**Abstract**: We study how the layers in deep neural networks (NNs) converge, using Wilsonian Renormalization-Group (RG) ideas to organize the spectral picture. During training, these layers often develop heavy-tailed power-law (PL) spectra, and the strongest checkpoints often appear near the PL exponent $\alpha\approx 2$. This pattern raises a simple question: why should this value mark a well-trained layer? The answer is both statistical and physical. Our statistical hypothesis is that $\alpha=2$ marks the boundary between stable if not optimal learning and overfitting via non-self-averaging. The physical answer is that the same value makes the spectral energy gain scale evenly across logarithmic spectral scales, which is the proposed Wilsonian marginal condition. We then apply the ERG trace-log condition, the proposed SETOL theory, which we justify by taking volume-preserving, scale-invariant transformations to remove arbitrary layer scale along the proposed RG flow. The resulting spectral RG flow connects the HTSR and SETOL theories to optimizers such as Muon, and motivates a new optimizer extension, WW-PGD. Together, these results point toward a spectral Renormalization-Group view of learning, in which NNs may approach shared spectral structure in the settings studied here. 

---
# On the Framing Effect of AI Red-Teaming Instructions: Evidence from Prior Exercises and Controlled Simulations 

**Authors**: Michael Feffer, Alice Qian, Anusha Sinha, Muye Zhang, Shalini Ghosh, Rachel Phinnemore, Bart Stawicki, Hong Shen, Hoda Heidari  

**Link**: [PDF](https://arxiv.org/pdf/2611.00412)  

**Abstract**: AI red-teaming, adversarially probing generative AI systems to surface harms, has become a central evaluation method. However, prior work has highlighted that red-teaming is a fairly open-ended task, laden with myriad design choices that can shape its findings. In this work, we focus on the instructions used to conduct red-teaming exercises and hypothesize that they can significantly affect outcomes in a manner analogous to the framing effect in psychology, where the format in which information is presented greatly impacts how people respond. To understand whether and why instructions impact red-teaming outcomes, first, we analyze the range of ways in which instructions have historically been formulated, reviewing 228 instances of prior work. Second, based on takeaways from this review, we run controlled simulations that hold parameters such as the risk domain, the attacker, and the model under investigation fixed while varying only the instructions. From five real instruction templates used by industry and academic organizations, we derive 45 instruction frames across three risk domains and three levels of risk granularity, generating 100 variants per frame, and deploy each against five open-weights target models, for 22,500 attacks. Using a generalized linear mixed model, we find that certain instruction properties significantly predict attack success rates: the lowest and highest levels of risk granularity both outperform intermediate specificity, as do short and long instructions relative to medium-length ones. The presence of examples is associated with higher attack success rates, while supplying explicit harm definitions yields lower ones. As alignment evaluations become increasingly automated, our results suggest that humans continue to shape red-teaming outcomes substantially through the instructions they provide to scaffold the activity. 

---
# Geometric and algebraic inference in humans and neural networks 

**Authors**: Jacob Russin, Joonhwa Kim, Ellie Pavlick, Michael J. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2611.00387)  

**Abstract**: Humans possess a remarkable ability to draw inferences from limited data. While some inferences are made intuitively from past experience, others require online reasoning based on novel information given in context. Here, we formalize these strategies, which we term geometric and algebraic inference, and study their computational tradeoffs in humans and neural networks. The geometric strategy gradually shapes representational geometry to reflect underlying relational structure, making subsequent inference cheap. The algebraic strategy flexibly attends to and reasons over novel relations given in context, but costs additional computation at inference time. We identify opposing signatures of these two strategies in transitive inference, a domain widely studied across species, and characterize the conditions under which each is observed. Our framework predicts that the geometric strategy should be dominant when relations are familiar, whereas the algebraic should emerge to support de novo inferences drawn in context. Multiple experiments demonstrate these opposing signatures in humans, neural networks trained from scratch, and large language models (LLMs). Causal interventions in the neural networks reveal that the geometric and algebraic strategies depend on representational geometry and selective attention mechanisms, respectively, and reproduce the effects of analogous neurobiological disruptions in humans. Our framework unifies these effects with disparate phenomena, including content effects observed in human and LLM reasoning and data-distribution effects observed in machine learning. Our findings reconcile competing views of cognitive flexibility that emphasize either representational geometry or algebraic manipulation by showing that each can have unique strengths, and can characterize different aspects of both human and neural network behavior. 

---
# Agent Stack for ML Training Lifecycle: Generation, Validation, and Optimization 

**Authors**: Jaideep Ray, Daniel Tang, Zejun Lin, Haoyue Tang, Lijuan Zhang, Tao Huang, Tommy Li, Aditya Jain, Ankit Goyal, Harsh Jain, Zhuo Zhi, Youmin Han, Santosh Jha, Shuying Liang, Yujie Ai, Jonathan Hung, Ethan Lyu, Abi Liu, Siddharth Singh, Feng Wen, Mahesh Ravindran, Souvik Ray, Yitong Zhou, Animesh Singh  

**Link**: [PDF](https://arxiv.org/pdf/2611.00384)  

**Abstract**: Production model development requires repeated architecture design, platform integration, debugging, training, and evaluation. We present an agent stack deployed at LinkedIn with two independently usable agent loops: AutoTorch for model generation and migration, and AutoResearch for model optimization. AutoTorch generates PyTorch models and training pipelines for TensorFlow migration, XGBoost replacement, and new-model development. Executable validation hooks check platform API compliance, trainability, and conversion fidelity on sampled data, guiding iterative repair before engineer review and full-scale distributed training. AutoResearch explores architecture and training-configuration changes through expert-steered or autonomous experimentation. An external orchestrator manages parallel trials, experiment history, candidate selection, and wall-clock budgets across agent sessions. A shared runtime preserves execution state independently of agent reasoning and allocates GPUs separately from agent execution. Across ranking, content understanding, and recommendation workloads, AutoTorch completed generation and full-scale training in 92 of 115 workflow runs (80%). Of these, 77 (83.7%) produced models meeting post-training quality requirements. In a separate cohort, AutoResearch improved the primary task metric on held-out data for 28 of 100 production pipelines. Engineering teams estimate that the stack reduced elapsed end-to-end migration time from approximately eight weeks to two weeks, a 4x speedup. 

---
# LF-VideoQA: Evaluating Long-Video Understanding with Long-Form Theory-of-Mind Questions 

**Authors**: Shaden Shaar, Bradon Thymes, Majd Aldaye, Claire Cardie, Bharath Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2611.00383)  

**Abstract**: The hardest part of understanding a long video is not pinpointing any one moment but synthesizing many, e.g., answering a question that requires tracking what develops across hours of content and articulating how the pieces fit together. To date, most long-video QA datasets require only a short answer or a multiple-choice selection, oversimplifying the long-video understanding task to retrieval. We introduce LF-VideoQA, a dataset for long-form, open-ended question answering over long narrative videos built on theory-of-mind questions: a character's belief depends on everything they have seen or been told, so each question demands assembling and reasoning from evidence distributed across the video. The dataset contains more than 6,300 questions on 94 recent films; each requires reasoning over scenes spanning an average of 23 minutes and comes with a question-specific rubric for recall and precision scoring. The dataset is created entirely automatically by annotating screenplays with scene events and each character's first-order beliefs and meta-beliefs, then generating (question, answer, rubric) triples through LLM-based question generation and verification; human studies validate their quality. We evaluate four families of state-of-the-art open-source VLMs across three settings and find the task highly challenging: (i) even an oracle with access to the screenplay falls well short of complete answers; (ii) the best non-oracle model attains barely half of the recall level of the oracle; (iii) unlike on prior long-video benchmarks, captions and subtitles underperform the video itself, indicating that the questions need visual evidence; and (iv) recall is the predominant bottleneck, with answers omitting many required rubric claims. 

---
# Cross-Prompt Reuse of Entity-Replacement Responses 

**Authors**: Jinkun Chen, Xuemin Yu, Sijia Han, Vlado Keselj  

**Link**: [PDF](https://arxiv.org/pdf/2611.00352)  

**Abstract**: With model weights fixed, we measure how much of an entity-replacement response transfers between English factual questions, beyond whether answers change. We add the hidden-state change from two source questions to a target question about the same entity, comparing the predicted log-probability shift between two relations' published answers, given known answer prefixes, with the measured one. Hidden reuse injects the aligned change at 50% depth before the target's remaining layers; direct reuse adds source output-score changes. With other-relation sources, hidden reuse removes 45% and 49% of no-change error on 0.5-1B Qwen and Llama. Gemma-2-2B removes 22% and passes one of three comparisons in plain text without its default beginning-of-sequence (BOS) token. A later protocol restoring it yields 58-68%, passing every comparison on three cohorts. Larger checkpoints, chat format and new entities show replication and format- or cohort-dependent failures. On small plain-format checkpoints, other-relation hidden reuse mainly reproduces the shift both relations' questions share, without reliably lowering relation-selective error. With target-relation sources, hidden reuse's post hoc advantage over direct reuse is 0.2-0.9 nats outside chat format, unconfirmed and not replacement-specific. The same relation-neutral change at both questions' entity tokens lowers relation-selective error more than random or other-replacement changes, indicating receiving-question dependence. 

---
# On the Limits of Test-Time Compute for Self-Evaluation 

**Authors**: Anirudh Buvanesh, Khaled Rouissi, Sujin Yun, Johan Obando-Ceron, Morgane M Moss, Samuel Lavoie, Michael Noukhovitch, Laurent Charlin, Aaron Courville  

**Link**: [PDF](https://arxiv.org/pdf/2611.00350)  

**Abstract**: Self-verification and aggregation have emerged as popular approaches for bridging the gap between pass@1 and pass@N. However, increasing the number of candidates does not consistently narrow the gap between final-answer accuracy and pass@N. To understand this limitation, we use ranking error to measure how reliably models distinguish their correct generations from incorrect ones. We show that ranking error admits a bias-variance decomposition, with both components shaped by algorithmic design choices and inference compute. Across our experiments, additional compute reduces variance, while bias initially decreases before approaching a plateau. Guided by this analysis, we introduce Scorpio, a simple and general method that jointly scores randomly grouped candidates. Across math, code, and puzzle benchmarks, Scorpio improves verification, tree search, and aggregation. For verification, Scorpio improves accuracy over the strongest baseline by up to 7 percentage points while matching or exceeding that baseline's peak accuracy with 17-42% lower inference cost. For tree search, using Scorpio to score partial solutions and select the final answer improves accuracy by up to 10 percentage points over prior methods. Finally, Scorpio improves the scaling of aggregation-based methods by 11 percentage points on coding tasks. 

---
# SPARK: Self-Correcting Embodied Memory via Physical Re-Observation 

**Authors**: Dmytro Kurdydyk, Aadi Palnitkar, Sebastian Scherer, Seungchan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.00344)  

**Abstract**: Retrieval-augmented generation (RAG) systems have seen growing adoption in embodied settings as language-queryable memories of the environment. However, existing approaches often assume a reliable single-episode knowledge base (KB) or accumulate observations across episodes without correcting prior errors. Such errors can persist through incorrect semantic labels, inaccurate object positions, and inconsistent spatial relationships. To address this, we present SPARK, a corrective multi-episode framework for embodied RAG. After the first episode, a semantic critic flags regions of the KB, a targeted planner generates revisits, and a selective merge adds, removes, or replaces KB information based on the new observations. We evaluate SPARK on four outdoor AirSim maps under two degraded viewpoint conditions, using perception-built KBs, a visibility-constrained oracle used only for evaluation, and a fixed set of 158 embodied question answering (EQA) questions scored before and after correction. A single corrective episode improves downstream EQA on every map under both conditions while storing 13-41% fewer objects than append-only updating, with gains up to 2.7x on one map-condition pair. Extending correction through Episode 4 brings every map above the EQA performance of its undegraded reference. Ablations show that the critic's three signals each contribute to performance, that targeted revisits achieve a larger mean gain than untargeted ones under the same budget, and that selective merging achieves gains comparable to append-only updating with a smaller KB. Overall, SPARK demonstrates the value of correcting embodied memory through targeted re-observation, particularly as embodied systems scale and the risk of accumulated errors increases. 

---
# Decomposing Irregular Time Series into Interpretable Temporal Patterns with Mixture-of-Experts 

**Authors**: Jenna L. Ballard, Zijie Pan, Shivani Shekhar, Yiming Bian, Yun-Shiuan Chuang, Avinash Thangali, Uma Kona, Linsey Pang, Prakhar Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2611.00337)  

**Abstract**: Irregularly-spaced sequential data, such as transaction logs, encode multi-scale temporal structure that is critical for prediction but difficult to recover explicitly with standard sequence models, which often trade interpretability for performance. We introduce MoTIF, an interpretable framework using mixture-of-experts (MoE) attention to decompose irregular time series into temporal patterns spanning minute-to-year scales. Each expert is trained via temporal alignment that induces attention specialization to a designated cyclical or within-range pattern, with label-free routing supervision that learns to activate the correct expert from patterns present in the data. On synthetic data with known temporal structure, MoTIF recovers correct routing well above chance (0.389/0.829 for range/cyclical patterns vs. 0.091 random) and outperforms all baselines, including a standard transformer and an unspecialized MoE, on next-value prediction. On a public credit card transaction dataset, MoTIF achieves higher macro-F1 than all baselines for next-amount prediction. Its learned routing reveals structure a black-box model cannot: transactions are governed primarily by intraday and short-interval rhythms, while fraudulent transactions show bursty, calendar-agnostic timing versus the steadier periodicity of legitimate ones. On downstream fraud detection, MoTIF matches standard transformer and MoE baselines and clearly outperforms recent sequential recommendation models used as embedding baselines. These results show that decomposing temporal structure into interpretable, expert-specific patterns need not sacrifice predictive accuracy, offering a practical route to sequence models that are both accurate and auditable in high-stakes settings. Code is available at this https URL. 

---
# C3: Speech-Guided KV Retrieval for Long-Context SpeechLLM Inference 

**Authors**: Dancheng Liu, Markus Müller, Rupak Vignesh Swaminathan, Jing Liu, Srikanth Ronanki, Aram Galstyan  

**Link**: [PDF](https://arxiv.org/pdf/2611.00333)  

**Abstract**: Speech language models encode audio at a much higher rate than text, causing the KV cache to dominate the memory, bandwidth, and decoding cost of long-form generation. KV retrieval can reduce per-step access by attending to only a query-dependent subset of the cache, but existing methods are designed primarily for text and do not account for the structure of audio attention. We first show that retrieval errors may compound autoregressively into generation drift. We then analyze full audio attention during long-form generation and identify three properties relevant to retrieval: unequal attention concentration across query heads, non-contiguous audio support, and temporal continuity across decoding steps. Guided by these observations, we propose $C^3$(Cluster-Compete-Compete), which organizes audio KVs in a hierarchical key-space index, lets query heads sharing a KV head compete for a common retrieval budget, and adaptively reallocates retrieval resolution through paired split-merge updates. Across long-form speech recognition, translation, and summarization on four SpeechLLMs, $C^3$ achieves the strongest overall performance among retrieval methods under comparable final-KV access budgets, with particularly large gains in the tight-budget regime, while improving long-context decoding efficiency and accelerator-memory scaling. 

---
# A Framework for Task-Agnostic Counterfactual Fairness in Transformers through Activation Steering 

**Authors**: Joshua Wolff Anderson, Yanshan Wang, Shyam Visweswaran  

**Link**: [PDF](https://arxiv.org/pdf/2611.00298)  

**Abstract**: Bias in transformer-based models is a growing concern as these architectures become more prevalent in automated decision-making systems, such as prediction models for hiring, lending, and clinical treatment recommendation. Mechanistic interpretability (MI) is a growing body of work that measures meaningful causal effects of activations on model output by identifying human-interpretable features encoded in model activations. When such features correspond to sensitive attributes, they provide a direct way to quantify the bias those attributes introduce and to causally intervene to mitigate it. However, counterfactual fairness (CF) methods have not been systematically adapted to this setting of intervening on interpretable mechanisms in transformers. We propose a framework, Mechanistic Interpretability for Counterfactual Fairness (MICF), that utilizes MI tools to reformulate CF constraints in terms of transformer activations, unifying former approaches into a single pipeline for fairness analysis in transformers. MICF does not require a prior causal model or specific transformer architecture. We evaluate MICF on real-world tabular data and show that CF can be improved across different prediction tasks. 

---
# TADA: Technology Aware DNN Accelerator Performance and Energy Estimator 

**Authors**: Tanvi Sharma, Surya Selvam, Anand Raghunathan, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2611.00292)  

**Abstract**: Deep neural network (DNN) accelerators for edge inference must meet tight energy, latency, and area (EDP) constraints, and their optimal hardware architecture and dataflow configuration is typically identified through design space exploration (DSE). However, on-chip memory technology including cell type, bank organization, and process node, is rarely treated as a first-class citizen in DSE. For example, increasing memory capacity can reduce memory accesses through greater data reuse, but at the cost of higher area and access energy, thereby limiting the compute units budget. Consequently, memory technology can fundamentally shift the optimal accelerator architecture for a given area and EDP budget. Existing machine learning (ML)-based DSE methods require repeating simulation, data generation, and model training for each new memory technology. We present TADA, a memory technology-aware DNN accelerator performance and energy estimator that predicts system latency and energy for a given workload, hardware configuration, and memory technology. TADA is trained on a synthetic dataset that generalizes across memory technologies. The dataset spans current and future technology trends while enforcing physics-based guardrails to preserve realistic area-energy trade-offs. Each memory technology is represented by a technology signature, a vector of read energy, write energy, and storage capacity across a range of areas, provided as input alongside the workload and hardware configuration. Across 21 unseen technology curves spanning SRAM, STTRAM, ReRAM, PCM, FeFET, eDRAM, and CTT, TADA predicts system energy and latency with 1.5-4.4% and 0.7-1.2% mean absolute percentage error, respectively. In three pairwise case studies, it matches the simulator-selected memory technology at 98.7-100% of area budgets, with a mean EDP loss of at most 0.21%, while achieving over 10^4x higher evaluation throughput. 

---
# When AI Agents Act on Our Behalf: A Taxonomy of Human-Agent Interaction Failure 

**Authors**: Canwen Wang, Qing Xiao, Haiyi Zhu, Hong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.00291)  

**Abstract**: AI agents write and execute code, manipulate user interfaces, and transact on their users' behalf. We ask what forms failure takes in human-agent interaction, and how those forms change as agents act with greater autonomy. From 695 documented incidents in two public repositories, we developed 12 failure modes bottom-up within three interaction stages adapted from Amershi et al.'s guidelines, coding the autonomy at which each arose. What a person can do when an agent fails depends on how much the agent completes between points of user involvement--a property fixed at deployment. Therefore, the same model on the same task can fail differently under different configurations. All 12 modes appear at both low and high autonomy, and rising autonomy changes the form each failure takes. Users need controls between full delegation and full manual work, and neither incident records nor agent benchmarks reach the interaction where these failures occur. 

---
# VerusWeave: A Benchmark for Proof Repair in Concurrent Verus Programs 

**Authors**: Pedro Ribeiro, Martim Afonso, André Vasconcelos, João F. Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2611.00287)  

**Abstract**: Concurrent software underpins critical infrastructure, including operating systems and databases, yet thread interactions can have bugs that bypass conventional testing. Verus extends Rust with formal verification capabilities, enabling developers to specify, implement and prove the correctness of concurrent code in the same language. Although coding agents have been explored for automated proof construction, existing Verus benchmarks offer limited coverage of concurrency-specific examples that connect atomic operations to abstract protocols.
To address this gap, we introduce VerusWeave, the first benchmark dedicated to concurrent proof repair, derived from four open-source projects. It comprises 75 focused tasks and 17 full-module variants that exercise features used to verify low-level concurrent code. Tasks remove selected proof annotations and require agents to fill gaps, while preserving executable code and specifications. The evaluation harness validates repairs using Verus and a change checker, recording solutions, execution traces, and resource usage. An initial evaluation of four general-purpose coding-agent harnesses achieved 75-85% success on focused tasks and 29-47% on full-module variants, highlighting the challenge of autonomous module-wide repairs. VerusWeave provides a reusable foundation for evaluating automated proof construction for concurrent software. 

---
# DELBERT-2: Pretrained Fingerprint Language Models for DEL Protein Binder Prediction 

**Authors**: Bing Hu, Sun Sun, Shaik Salman Basha, Anita Layton, Helen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.00281)  

**Abstract**: We present DELBERT-2, a pretrained fingerprint language model that converts sparse molecular fingerprints into unified token sequences using a ModernBERT encoder trained with masked language modelling on 2.5M molecules from the AIRCHECK DEL corpus. We evaluate across six targets (WDR91, WDR12, SETDB1, PLCZ1, LRRK2, DCAF7) under three out-of-distribution (OOD) protocols: hierarchical cluster splits probe chemical novelty, library splits test cross-library transfer, and building-block splits assess compositional generalization. DELBERT-2 consistently improves PR-AUC and NDCG@1000 relative to LightGBM ensemble baselines and transformers trained from scratch, with the largest gains in stringent OOD regimes. DELBERT-2 achieves 13.28+/-3.72 enrichment factor at top-100 vs. 12.36+/-3.75 for no-pretraining (p=0.040), representing 7.4% improvement and 13x enrichment over random selection. These results demonstrate that fingerprint-centric self-supervised learning effectively improves hit prioritization under distribution shift, enabling practical DEL virtual screening. Code is available at \hyperlink{this https URL}{DELBERT-2}. 

---
# Outstrumental Variable Regression 

**Authors**: Felix Schur, Linus Kühne, Julius von Kügelgen, Jonas Peters  

**Link**: [PDF](https://arxiv.org/pdf/2611.00266)  

**Abstract**: Estimating the causal effect of a treatment $X$ on an outcome $Y$ from observational data in the presence of unobserved confounding requires additional structure or assumptions. Existing identification strategies such as instrumental variable regression exploit exogenous variation in only the treatment. We show that the causal effect can also be identified if the outcome mechanism varies across a finite set of environments, even if the treatment distribution is unchanged. Concretely, we introduce Outstrumental Variable Regression (OVR), which proceeds in three steps: (i) compute the conditional expectation of $Y$ given $X$ in each environment; (ii) form the vector of differences in these conditional expectations between all environments and a reference environment (we call this vector contrast); and (iii) regress $Y$ on $X$ while adjusting for the contrast and environment. We prove that this yields the true causal effect if the following two conditions hold: inclusion restriction, i.e., the contrast determines the conditional confounding bias; and positivity, i.e., the treatment varies sufficiently strongly after adjustment. We prove that a finite-sample version of OVR is consistent under regularity conditions and demonstrate the procedure's effectiveness on synthetic and real-world data. 

---
# Spectral and Conformal: Neural Network Methods for Calibrated Trajectory Prediction and Control-Point Intervals 

**Authors**: Jasmine Ratchford, Raffaele Romagnoli  

**Link**: [PDF](https://arxiv.org/pdf/2611.00231)  

**Abstract**: Autonomous systems that navigate their environment need methods to quickly assess the feasibility and safety of desired trajectories. Traditional solutions that rely on numerical integration can be computationally heavy, while fast data-driven surrogates lack the formal guarantees that numerical approaches provide. This work demonstrates a framework that uses (1) a neural network with a spectral backbone to learn trajectories and quantile bands with built-in constraints (initial conditions and stability); (2) a modified conformal quantile regression to provide calibrated, joint trajectory-level coverage; and (3) interval arithmetic to extend noise-free coverage guarantees to control points. After refactoring the network for inference, the trajectory inference time for a quadrotor under a closed-loop linear quadratic regulator is 3.45~ms on a Raspberry Pi Zero 2 W. The total pipeline time, including trajectory inference time, is 9.81~ms, or approximately 100~Hz, consistent with typical control rates. 

---
# From Fragmentation to Fusion: A Survey and Roadmap for Omnimodal Embeddings 

**Authors**: Ziyin Zhang, Yuyu Guo, Ying Zhang, Siqiao Xue, Zihan Liao, Jin Qin, Wenjie Yang, Siyuan Yang, Shijie Li, Cheng Chen, Hao Wu, Zhaoyang Zhu, Ruipeng Li, Zhaokun Jiang, Lei Lei, Rui Wang, Hang Yu, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00172)  

**Abstract**: Embedding models underpin virtually every modern AI system, from search engines to recommendation systems. Despite shared mathematical foundations, research on embedding models remains fragmented by modality: the communities develop largely in isolation, seldom exchanging insights. We argue that this fragmentation obscures a deeper convergence. Driven by contrastive learning, Transformer architectures, and a granularity shift, embedding research across modalities is converging toward a common set of principles, even as modality-specific challenges remain underexplored. To reason across this convergence without conflating it, we propose a framework of three progressively harder capabilities: coverage, comparability, and compositionality. Three paths toward unification satisfy these in turn: transformation broadens coverage, alignment enables comparability, and joint embedding, the terminus we advocate, aims at compositionality. This position paper traces the resulting evolution, providing the first unified, cross-cutting treatment of omnimodal embedding research. 

---
# From Concepts to Actual Causes: Toward Closing the Concept-Malfunction Gap in Robot Foundation Models 

**Authors**: Gihyuk Ko  

**Link**: [PDF](https://arxiv.org/pdf/2611.00155)  

**Abstract**: Post-incident analysis of a robot foundation model poses a singular causal question: which internal components caused this failure, and to what degree? Mechanistic interpretability (MI), however, establishes general causal claims by decoding the concepts a policy has learned, yet knowing which concepts are encoded is insufficient for identifying the actual cause of a specific incident. In this position paper, we analyze this concept-malfunction gap by posing fault isolation in the framework of actual causality. Specifically, we formalize a fault-isolation query as a tuple of intervened internal components, counterfactual values, contingencies, and read-outs, and lay out conditions for it to be well-specified. Through this lens, we identify shortcomings of current MI approaches: they assume fixed contingencies and read out only immediate effects, letting backups obscure actual causes. We demonstrate on a toy example that these shortcomings can lead to conflicting verdicts. Lastly, based on the well-specification conditions, we outline four open research tasks toward closing the gap. 

---
# Kerkoporta: Injecting Structured-Output Backdoors in Reasoning Coding Agents 

**Authors**: Georgios Syros, Blake Bullwinkel, Alina Oprea, Cristina Nita-Rotaru, Amanda Minnich, Xiaodong Liu, Giorgio Severi  

**Link**: [PDF](https://arxiv.org/pdf/2611.00138)  

**Abstract**: Existing large language model (LLM) backdoor attacks operate primarily in chat settings, where the risk is bounded by free-form text outputs. AI agents remove that bound by performing actions and executing commands on the user's machine. We introduce structured-output backdoor attacks against reasoning models in agentic coding settings. A trigger placed anywhere in the agent's context causes the model to emit an attacker-chosen tool call, which the harness executes, directly compromising the system's confidentiality, integrity, or availability. We present Kerkoporta, a methodology that constructs high-quality poison samples, injects the backdoor via supervised fine-tuning, and recovers utility loss with lightweight reinforcement learning. We evaluate on modern, heavily post-trained open-weight models across proprietary and open-source harnesses, focusing on coding for its wide adoption and long-horizon task structure. The attack reaches success rates of up to 96.2% and can be configured to generalize across harnesses or bind to a single target. The resulting models can outperform the clean baseline on standard utility benchmarks, giving victims a direct incentive to adopt them. Overall, we demonstrate that LLM backdoors can have critical security impacts in the context of modern coding agents, and we characterize the poisoning methodology required to insert backdoors that are both effective and stealthy in this setting. 

---
# FB-MOAC: A Reinforcement Learning Algorithm for Forward-Backward Markov Decision Processes 

**Authors**: Mohsen Amidzadeh, Mario Di Francesco  

**Link**: [PDF](https://arxiv.org/pdf/2611.00130)  

**Abstract**: Reinforcement learning (RL) algorithms are effective in solving problems that can be modeled as Markov decision processes (MDPs).
These algorithms primarily target forward MDPs whose dynamics evolve over time from an initial state.
However, several important problems in different scenarios including stochastic control and network systems exhibit both a forward and a backward dynamics. As a consequence, they cannot be expressed as a standard MDP, thereby calling
for a novel theory for RL in this context.
Accordingly, this work introduces the concept of Forward-Backward Markov Decision Processes (FB-MDPs)
for multi-objective problems,
develops a novel theoretical framework to characterize their optimal solutions,
and proposes a general forward-backward step-wise template
that allows to adapt RL algorithms for FB-MDP problems.
A Forward Backward Multi Objective Actor Critic (FB-MOAC) algorithm is introduced accordingly
to obtain optimal policies with guaranteed convergence and a competitive rate with respect to standard approaches in RL.
FB-MOAC is evaluated on diverse use cases in the context of mathematical finance and mobile resource management.
The obtained results show that FB-MOAC outperforms the state of the art across different metrics,
highlighting its ability to learn and maximize rewards. 

---
# SPARQLcirc: Probabilistic Query Evaluation with Provenance Circuits 

**Authors**: Jingcheng Wu, Ratan Bahadur Thapa, Daniel Hernández, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2611.00128)  

**Abstract**: Probabilistic query evaluation (PQE) computes the probability of every answer of a query over a tuple-independent probabilistic knowledge graph. We study exact PQE for SPARQL over graphs accessed through SPARQL 1.1 endpoints. Shared triples among the derivations of an answer make exact PQE #P-hard in general. Knowledge compilation of the Boolean provenance of every answer yields exact probabilities, and the provenance formulas of different answers can share subformulas. We propose SPARQLcirc, which rewrites a SPARQL query into CONSTRUCT queries that name every gate by its content, so that the results of independent queries form one Boolean provenance circuit for all answers. The rewriting covers joins, unions, projection, filters, OPTIONAL, difference, and transitive closures. SPARQLcirc compiles the circuit and computes the exact probability of every answer by weighted model counting. For the evaluation, we draw random probabilities for the triples of WatDiv and gMark and the statements of Wikidata and evaluate the CONSTRUCT queries on GraphDB, Fuseki, and Oxigraph. The ratio of the end-to-end time of PQE with SPARQLcirc to that of PQE by compiling the provenance expressions of the SPARQL provenance rewriting NPCS decreases with the number of derivations. 

---
# DTP-OPSD: Reinforcement Learning with Distractor-Aware Temporal-Privilege On-Policy Self-Distillation for Temporal Video Grounding 

**Authors**: Zhaoyu Chen, Yongwei Nie  

**Link**: [PDF](https://arxiv.org/pdf/2611.00120)  

**Abstract**: Temporal Video Grounding (TVG) aims to localize the start and end times of an event described by a language query in a video. The task becomes particularly challenging when the same video contains multiple repeated, similar, or opposite events. On these hard examples, all responses sampled by Group Relative Policy Optimization (GRPO) may miss the target and receive zero temporal Intersection-over-Union (tIoU). Their group-relative advantages are then all zero, leaving no useful policy gradient signal. To address this, we propose DTP-OPSD, a reinforcement learning framework with distractor-aware temporal-privilege on-policy self-distillation. The student observes the clean video and generates responses on-policy, while a shared-weight teacher observes a privileged video in which the target window is marked by red full-frame borders and same-video distractor windows by blue full-frame borders. For the distillation loss, the teacher evaluates student-generated prefixes and provides dense token-level Jensen-Shannon divergence supervision. This supervision remains informative even when GRPO provides no policy gradient signal. We suppress distillation when the teacher does not outperform the student and strengthen it for groups where the teacher performs better but all student rollouts receive zero tIoU. We also construct a compact hard-interference training set through VLM screening and human verification. With only 300 additional post-training examples, DTP-OPSD achieves the highest mIoU among the compared open-source 8B models on Charades-TimeLens, ActivityNet-TimeLens, and QVHighlights-TimeLens, reaching 55.93, 54.77, and 66.71, respectively. Compared with Pure GRPO trained on the same data, DTP-OPSD improves all 12 reported metrics, with R1@0.7 gains of 1.28, 1.62, and 1.78 points on the three benchmarks. Code will be released at this https URL 

---
# Demystifying Prompt Stealing Attacks against Synthetic Video 

**Authors**: Junyan Huang, Sanskar Agmain, Chen Gong, Yan Pang, FNU Suya  

**Link**: [PDF](https://arxiv.org/pdf/2611.00102)  

**Abstract**: Creators publicly showcase generated content to advertise and sell their work, but these examples may also enable unauthorized reproduction and customization. We study whether an adversary can derive reusable inputs from a single public video for faithful reconstruction and controlled editing, without access to the original prompt or generator. We adapt captioning and prompt optimization approaches to video prompt stealing attacks (PSAs) and find that prompt recovery alone provides limited reconstruction fidelity. Even the original prompt produces different spatial and temporal realizations when the initial noise changes. Using an independent generator enlarges the reconstruction gap, as reusing the original prompt and seed can still produce substantially different videos. We propose V-PSA, which uses a recovered caption and DDIM inversion to construct a reference trajectory under an independent generator. Null-Text Inversion then optimizes per-step unconditional embeddings to reduce deviations from this trajectory during reconstruction. The resulting inputs also guide generation when the caption is modified, supporting edits that change selected content while preserving the target's layout and motion. We construct a benchmark containing 12,000 video-prompt pairs from three victim generators and four prompt sources. V-PSA achieves video-to-video similarity around 0.9 between target and reconstructed videos, substantially improving over prompt-only baselines with similarity score of 0.6, and supports multiple controlled variations that preserve the target's structure. Additional experiments show that augmenting prior image PSAs with inversion-derived inputs also improves reconstruction. Our results show that evaluations restricted to prompt recovery can underestimate an attacker's ability to reconstruct public generative outputs. 

---
# A Novel Lightweight Multi-Scale Branching Encoder for Few-Shot Image Classification 

**Authors**: Oriod Malo, Alexandre Xavier Falcão, Laurent Najman  

**Link**: [PDF](https://arxiv.org/pdf/2611.00099)  

**Abstract**: Few-Shot Learning (FSL) addresses classification scenarios in which only a small number of labeled examples are available for each class. This paper presents the Sum12 Multi-Scale branching encoder, a lightweight convolutional architecture designed to improve the embedding stage of Prototypical Networks without substantially increasing computational cost. The encoder uses seven parallel convolutional branches whose kernel-size combinations sum to 12, allowing local and broader spatial structures to be represented within a unified Multi-Scale feature space. The proposed architecture was evaluated on Omniglot, CIFAR- FS, FC100, and miniImageNet under controlled few-shot learning protocols and compared with the original Prototypical Network encoder and Inception-inspired branch variants. Additional ablation studies have investigated fusion strategies, dilated convolutions, branch compositions, and receptive field budget. Beyond few-shot learning, the encoder was also tested on the MedMNIST 2D benchmark to assess its behavior in conventional supervised medical image classification. Across these experiments, Sum12 generally improved the low-shot classification performance with respect to the basic ProtoNet encoder while preserving a compact model size. These properties suggest that the proposed encoder may be suitable for low-data visual recognition and resource-constrained intelligent imaging applications. However, complete end-to- end validation on the target embedded platform, including resolution of the observed implementation discrepancies, remains future work. 

---
# Pano2BIM: A Geometry-Constrained Vision-Language Agent for As-Built BIM Generation from a Floor Plan and a Single 360$^\circ$ Image 

**Authors**: Juan Diego Castaño Molina, Borja García de Soto  

**Link**: [PDF](https://arxiv.org/pdf/2611.00097)  

**Abstract**: Digital twins require accurate, up-to-date building information models (BIM) that reflect physical conditions, yet generating and maintaining as-built models remains costly and labor-intensive. Emerging image-based methods can capture visual appearance without inherently producing structured BIM information, while vision-language models (VLMs) struggle with reliable metric object placement. This paper presents Pano2BIM, a geometry-constrained VLM agent that generates approximate IFC4 models at Level of Development (LOD) 200 from a floor plan with known room dimensions and a single 360-degree photograph. The VLM identifies landmarks and objects, while deterministic tools estimate camera pose, compute metric positions through ray casting, and check spatial constraints. An optional second stage enriches the presentation model through render-and-compare feedback without modifying the primary furniture geometry. In a preliminary evaluation on one laboratory, the proposed Measure workflow achieved a mean furniture-footprint intersection over union (IoU) of 0.53, compared with 0.44 for a single-response baseline using identical modeling instructions, and matched an average of 4.0 versus 2.3 of seven fixed furniture elements across three runs. Centroid disagreements were similar for the four elements consistently identified by both approaches. The measurement stage averaged 3.8 minutes and USD 1.63 in API costs per run, while the complete demonstrated workflow required approximately 42 minutes and USD 11. These findings demonstrate the feasibility of rapid, low-cost as-built BIM generation, with potential applications in maintaining digital twins that reflect changing physical conditions. 

---
# Evaluating Context Sensitivity of Collaboration Behavior in Multi Party Simulation Frameworks 

**Authors**: Anvesh Rao Vijjini, Sheshera Mysore, Pegah Jandaghi, Ashish Sharma, Snigdha Chaturvedi, Bahareh Sarrafzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2611.00089)  

**Abstract**: Group Social Behavior is increasingly used as a lens to study Large language model (LLM) simulations to aid model training and evaluation. However, little prior work has examined how contextual factors such as team size and goal type manifest in long running simulations. To overcome this gap, we conduct a controlled, full-factorial study examining how contextual factors affect collaboration behavior across multi-party simulation frameworks. Using well established social science taxonomies, we quantify context sensitivity of collaboration behavior in simulated teams. Our analyses reveal significant effects of contextual factors such as simulators consistently favor smooth collaboration over disagreement, negotiation pressure, and coordination breakdown. These findings provide the first systematic evidence of how contextual factors influence emergent collaboration in LLM based multi-party simulation frameworks, offering both guidance and empirical results for the use of such simulations. 

---
# CRAFT: Corrective and Robust Multi-Agent Framework for Text-to-Parametric CAD 

**Authors**: Mohammed Musthafa Rafi, Anushrut Jignasu, Mahdi Saraeian, Chinmay Hegde, Aditya Balu, Adarsh Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2611.00076)  

**Abstract**: Text-to-CAD generation has the potential to make mechanical CAD more accessible, but existing approaches face a trade-off between data dependence and output fidelity. Supervised methods rely on large paired text-model datasets, while zero-shot prompting strategies are often brittle and tend to lose the parametric structure needed for downstream editing and customization. We present CRAFT, a corrective and robust multi-agent framework that produces parametric OpenSCAD programs from natural language without task-specific training. At its core is a JSON intermediate representation that carries symbolic parametric expressions throughout the generation process, so the resulting models remain editable and retain their parametric structure. CRAFT combines semantic understanding, parametric planning, compilation, rendering, self-correction, and component-level verification within a layered recovery process, using multi-view visual feedback to detect and repair geometric errors. Across NopSCADlib, ABC, and Slice-100K, CRAFT is competitive with direct zero-shot baselines in perceptual and geometric quality while exposing substantially more editable parametric structure. 

---
# CRaFT-XDet: Causal and Robust Fusion Transformer for Dual-Energy X-ray Prohibited Item Detection 

**Authors**: Moinul Hossain, Munshaibur Rahman Mahin, Taskin Khaleque, Mohammad Sakib Mahmood, M. F. Mridha, Md. Jakir Hossen  

**Link**: [PDF](https://arxiv.org/pdf/2611.00072)  

**Abstract**: Automated prohibited item detection in X-ray security imagery is fundamentally constrained by severe object overlap, scanner-dependent domain shifts, and spurious correlations induced by cluttered baggage composition. Existing detectors largely rely on appearance-driven supervision, limiting their robustness and cross-scanner generalisation. We introduce CRaFT-XDet, a causality-grounded and physics- motivated detection framework that reformulates X-ray threat detection as invariant representation learning under attenuation-driven modality variation. Our approach is grounded in a Scanner-Invariant Causal Detection formalisation (Theorem 1), which establishes that minimising the proposed Causal Invariance Regularisation (CIR) objective across partitioned scanner environments approximates the interventional distribution P (Y | do(Xobj)) under standard identifiability assumptions, providing a principled causal formulation for dual-energy X-ray detection. CRaFT-XDet integrates: (i) causal feature disentanglement to suppress context-induced shortcuts; (ii) physics-motivated regularisation to enforce material-consistent embeddings; and (iii) transformer-based cross-energy interaction modelling for structured reasoning in cluttered scenes. With 29.2 M parameters, 89.5 GFLOPs, and 345 FPS (1.7 ms inference), CRaFT-XDet is real-time deployable. Trained on SIXray with five-fold cross-validation, the framework achieves 0.9389 mAP@0.5 and 0.8021 mAP@0.5:0.95. Without domain adaptation, CRaFT-XDet maintains strong cross- dataset performance on OPIXray and HiXray, indicating robust generalization across heterogeneous scanner distributions. In category-agnostic binary threat detection, the model attains 93.92% recall and 0.9031 mean IoU, highlighting reliability in safety-critical screening where missed detections are costly. 

---
# SAMP: Self-Adaptive Margin Prioritization For Robust Diffusion Watermarking 

**Authors**: Yuhao Xue, Jiuan Zhou, Yu Cheng, Zhaoxia Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.00069)  

**Abstract**: Diffusion models have made high-fidelity visual generation widely accessible, while also lowering the barrier to producing convincing visual forgeries and complicating content provenance. Existing watermarking methods typically use a fixed embedding strength throughout denoising or treat all watermark dimensions equally. As a result, they may introduce unnecessary distortion when some watermark dimensions are already reliable, while failing to sufficiently reinforce those that are more vulnerable to subsequent perturbations. Therefore, robust watermarking requires adapting both the embedding strength during denoising and the optimization priority across watermark dimensions. To address this problem, we present SAMP, a self-adaptive framework consisting of global margin regulation and vulnerability-sensitive allocation. Global margin regulation adjusts the embedding strength according to the separation between the target codeword and competing codewords, together with the estimated effects of common attacks. Vulnerability-sensitive allocation identifies watermark dimensions that are more easily weakened by local perturbations and gives them higher optimization priority. Experimental results demonstrate that SAMP achieves an average information recovery accuracy of 0.993. Compared with state-of-the-art methods, SAMP improves PSNR by 16.8% and reduces LPIPS by 47.1%. These results validate the effectiveness of SAMP for reliable and high-quality diffusion watermarking. 

---
# Memorix: Managing Memory Selections for AI Agents 

**Authors**: Jing Zhang, Xiaguannan Song, Dong Du, Erhu Feng, Yubin Xia  

**Link**: [PDF](https://arxiv.org/pdf/2611.00060)  

**Abstract**: AI agents now handle a growing share of tasks across sessions within a persistent project. Their behavior depends not only on the model, but heavily on the persistent state they read from the environment. Existing systems assemble this memory at use time, splicing selected files into the prompt as a one-off payload. As a result, per-task assembly is costly and can still miss critical material, failures cannot be traced or replayed, and agents can read documentation written for the wrong version. Our key insight is that what an agent should read is mostly determined by its project, so the selection can be computed once, ahead of time, and reused across tasks.
We propose Memorix, an agent memory system that realizes this insight. Memorix achieves this via three key mechanisms: (1) an immutable and versioned Store pins every source by identity and reproduces historical layouts cheaply, (2) a View that splits selection (Derivations) from composition (Environment), making the result reproducible and shareable, and (3) Quests and Profiles keep a delivered view refreshable and available inside the project as sources evolve. We implement Memorix and evaluate it on SWE-ContextBench and LibEvolutionEval against mainstream agent memory systems. Precomputed Environments raise resolved rate by up to 1.5X and deliver the correct library version on every version-conditioned task, while cutting amortized cost by up to 100X and build time by up to 38X. 

---
# SemaNorm: Stain Normalization via Semantic and Morphology Alignment for Robust Cross-Hospital Computational Pathology 

**Authors**: Subhankar Chattoraj, Sawon Pratiher, Arijit Ghosh, Prashant Khare, Writankar Kundu, Karan Vishwakarma, Preha Handa, Kirti Rashmi, Hubert Konik  

**Link**: [PDF](https://arxiv.org/pdf/2611.00056)  

**Abstract**: Stain variability remains a major source of domain shift in computational pathology, yet existing normalization methods largely optimize color consistency without explicitly preserving morphology or diagnostic semantics. We propose SemaNorm, a self-supervised stain canonicalization stain normalizer to recover clean patches from strongly perturbed inputs, avoiding paired transfer, adversarial instability, and identity collapse. It further combines a frozen pathology foundation-model consistency loss with a differentiable nuclei-structure loss to preserve semantic and cellular organization. SemaNorm is evaluated across Camelyon17-WILDS and NCT-CRC-HE-100K using cross-hospital tumor classification, cell-graph morphology metrics, and semantic stability measured with an independent held-out foundation encoder. SemaNorm improves out-of-distribution accuracy by 5.9% over the strongest normalization baseline and by 13.5% over no normalization, while achieving the best morphology preservation. Independent foundation-model and cell-graph analyses further confirm improved semantic and structural stability. These results establish morphology- and semantics-aware optimization as a principled alternative to color-only stain normalization. 

---
# ReactionBench: A Human-Anchored Benchmark for Modeling Reactions to Video 

**Authors**: Yibo Zhao, Ao Qu, Xuan Jiang, Jiahao Xu, Keane Ong, Hang Jiang, Zhaofeng Wu, Dingyi Zhuang, Yihong Tang, Kaichen Zhou, Jinhua Zhao, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2611.00050)  

**Abstract**: Understanding what happens in a video is not sufficient for socially responsive multimodal systems: such systems also aim to relate what a person observes to how that person responds. Existing video benchmarks largely evaluate events, temporal reasoning, or people depicted in the video, while affective and viewer-response datasets often focus on emotion labels, aggregate responses, or prediction from the stimulus alone. This leaves an underexplored question: can current multimodal models ground an observed viewer reaction in the stimulus context associated with it? We introduce ReactionBench, a human-anchored benchmark for reaction-specific video understanding. Built from a candidate pool of 31,834 synchronized stimulus-reaction clips spanning 433 hours and eight domains, ReactionBench evaluates three complementary abilities: ranking the intensity of a viewer's reactions within a video, matching a reaction to its temporally aligned stimulus, and explaining the concrete stimulus cues linked to that reaction. Human annotation provides 14,894 ratings over 3,257 moments, and the frozen evaluation comprises 250 intensity profiles, 112 matching items, and 200 rationale items with 447 blind-written human references. Across seven multimodal models, reaction-specific grounding remains challenging: the best model reaches 50.9% accuracy on four-way stimulus-reaction matching, while GPT-5 shows no detectable loss under reaction substitution. Performance is also sensitive to evaluation protocol. Project page: this https URL 

---
# Out-of-distribution AI-generated image detection with zero-phase deconvolution for residual whitening 

**Authors**: Yusuf Yigit Pilavcı, Antoine Mallet, Vincent Itier, Jan Butora, Jérémie Boulanger, Patrick Bas  

**Link**: [PDF](https://arxiv.org/pdf/2611.00043)  

**Abstract**: Current AI-generated image detection methods struggle in out-of-distribution testing conditions because detectors are usually not robust when evaluated on generators not included in the training set. This work explores a way to mitigate the forensic bias between sources of images -whether real or generated -by removing generator-specific fingerprints. This new training process can be approximated by pre-processing involving a deconvolution operation designed to remove correlated noise patterns inducing loss of generalization. Indeed, we show that whitening the spatial distribution of the image noise residuals can significantly improve the performance of detectors on images coming from generators unseen during training. In particular, an end-to-end CNN model trained on whitened image residuals improves detection by 12.6% on average compared to a training involving original images. 

---
# Think Before You Locate: Clinical Chain-of-Though Supervision for Sequential Medical Image Grounding 

**Authors**: Adinath Madhavrao Dukre, Govinda Kolli, Yifan Lu, Ziyun Zou, Azmine Toushik Wasi, Sara Atito, Behzad Bozorgtabar, Dwarikanath Mahapatra, Muhammad Awais, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2611.00041)  

**Abstract**: Clinicians rarely localize a finding from one image alone. They compare current and prior scans, follow structures across slices, and match regions across views or modalities. Yet vision-language models for sequential medical grounding usually learn only from bounding boxes. Boxes show where the target is, but they do not explain why that region in that image answers the question. We study whether explicit, image-grounded reasoning can improve localization, and which tasks benefit from it. We introduce method, whose core is MedChain, a corpus of 182,573 filtered rationales spanning eight sequential grounding tasks and ten imaging modalities. A medical VLM (FlemingVL-38B) first drafts a caption-like rationale, and a multimodal refiner (Qwen3.5-397B-A17B) then re-examines the images, question and annotated box to write a box-conditioned explanation with explicit cross-image reasoning. We fine-tune Qwen3.5 on these rationales with deliberative chain-of-thought supervision (dCoT-SFT). Across 4B, 9B, and 27B models, dCoT-SFT sets a new state of the art on MedSG-Bench. The 27B model reaches 77.24 mIoU and 85.20 Acc@0.5, improving over the strongest prior model by 4.69 and 5.49 points. In a paired comparison using fixed 4B checkpoints, dCoT-SFT improves over answer-only SFT by 3.56 mIoU (95% CI 3.13--4.00). The gains mainly come from change detection and object tracking, while performance drops on three appearance-matching tasks. Replacing a model's own rationale with a rationale from another task reduces mIoU by 12.2 points, showing that localization depends strongly on the reasoning context. Finally, adding GRPO with an IoU reward after dCoT-SFT gives only small and task-dependent gains, which we analyze through controlled multi-seed development experiments. 

---
# Real-Time Obstacle Detection and Distance Estimation on Smartphones for Visually Impaired and Distracted Pedestrians 

**Authors**: Abtin Zandi, Sarvin Nami, Roza Ganjipour, Bardia Sabagh Kermani, Amirreza Tarabkhah, Hamed Farbeh  

**Link**: [PDF](https://arxiv.org/pdf/2611.00040)  

**Abstract**: According to the World Health Organization, at least 2.2 billion people worldwide live with visual impairment, including tens of millions with severe vision loss. Traditional aids such as white canes or guide dogs remain costly, limited in capability, and inaccessible to many. At the same time, smartphone distraction in public spaces has been linked to elevated risks of collisions with vehicles, trips, and falls. These dual challenges motivate lightweight, smartphone-based solutions that combine accessibility with low latency. We present a Kotlin-based Android application that uses a YOLOv8 detector, retrained to recognize eighteen common urban obstacle classes, to deliver real-time obstacle alerts for both user groups. The detector was optimized with pruning, quantization, and early-exit branches, and deployed through the Android Neural Networks API (NNAPI), which can dispatch inference to GPU, NPU, or DSP accelerators when available. On a mid-range smartphone, these optimizations reduced average inference latency from about 3.2 s for the off-the-shelf YOLOv8n to 0.39-0.48 s; the recommended configuration (INT8 quantization with early exit) runs in 0.42-0.48 s while retaining a recall of 0.81. The application combines the optimized detector with ARCore-based or monocular focal-length distance estimation, so it works with a single smartphone camera and requires no LiDAR, depth camera, or other specialized sensor. Multimodal feedback, including Text-to-Speech guidance for visually impaired users and vibration alerts for distracted pedestrians, supports use across different contexts. 

---
# Beyond Black Boxes: An Ontology Based Vision Framework for Transparent Fruit Disease Detection 

**Authors**: M.J. Marapperuma, B.V.K.I. Vidanage  

**Link**: [PDF](https://arxiv.org/pdf/2611.00039)  

**Abstract**: Efficiency and reliability in the timely identification of phytopathogens would help minimize crop loss in agricultural sectors but existing deep learning based methods, although accurate, would require a large annotated dataset and are black-boxed in their operation, thus, restricting flexibility and transparency. The current paper presents a hybrid detection framework that is able to overcome these drawbacks through the unified real-time vision with a semantic knowledge base. Web Ontology Language (OWL) was used to develop a comprehensive ontology of the disease of fruits that formally codifies phytopathological knowledge, such as symptom phenotype, spectral phenotype (HSV), and morphological phenotype. Such knowledge-guided configuration guides the image processing pipeline, making it possible to diagnose using this knowledge under expert specified rules without a large training dataset. The ontology underwent a computationally verified test of its structural soundness and commonsensical practical usefulness and scored an overall of 7.4/10. On the public set of MangoFruitDDS, the diagnostic performance was an F1 Score of 85.00%, precision of 85.00%, and a recall of 85.23%. These findings are very competitive and similar to other known deep learning models like the VGG16 that prove that the knowledge-driven approach provides an accurate, transparent, and scalable solution on automated fruit disease diagnosis. 

---
# From Single-Shot QA to Continuous Perception: Revisiting the Energy Composition and Optimization Boundary of Edge VLMs 

**Authors**: Yiting Duan, Miaoxin Wang, Aoqing Guo, Bo Peng  

**Link**: [PDF](https://arxiv.org/pdf/2611.00036)  

**Abstract**: Visual-prefill energy conclusions for edge vision-language models (VLMs) are commonly drawn under single-shot workloads; we revisit them under continuous perception. Profiling nine VLMs across three architecture families on Jetson AGX Thor and Orin (4,508 records, calibrated with a pure-text task), we find two mutually independent energy laws: the per-output-token decode cost grows monotonically with model scale, whereas the visual-prefill (image-encoding) cost is dominated by visual-encoder architectural efficiency, not scale. Under continuous perception ($N{=}10$) with sparse output, visual-prefill energy reaches 92--97\% of total. We derive optimization strategies---visual frame reuse, adaptive output truncation, task-aware routing, and encoder throughput optimization---for edge VLM deployment. 

---
# A Trustworthy Clinical Decision Support Framework for Brain Tumor MRI Classification: Integrating Attention-Guided Learning, Explainable AI, and Out-of-Distribution Validation 

**Authors**: Arjun Kumar Gupta, Harminder Pal Monga  

**Link**: [PDF](https://arxiv.org/pdf/2611.00034)  

**Abstract**: Deep learning models for brain tumor diagnosis need to do more than achieve high classification accuracy. For practical use, they should also provide understandable predictions, recognize unsuitable inputs, and work within a usable clinical system. This paper presents a framework for classifying brain MRI into four categories: glioma, meningioma, pituitary tumor, and no tumor. The model uses InceptionResNetV2 with a Convolutional Block Attention Module (CBAM), while Grad-CAM provides visual explanations of its predictions. A two-stage out-of-distribution (OOD) mechanism screens unsuitable inputs using an initial grayscale check followed by Mahalanobis distance, softmax confidence, and predictive entropy. These components are brought together in NeuroScan AI, a real-time web application. Evaluated on a Mendeley brain tumor MRI dataset of 12,064 images, the framework achieved 98.70% validation accuracy, 96.73% test accuracy, and a macro-average AUC of 0.9965. The OOD pipeline achieved 99% detection accuracy on a preliminary 100-image benchmark. Grad-CAM visualizations showed qualitative correspondence with clinically established tumor locations across all four classes. Overall, the work brings classification, explainability, input validation, and deployment into one framework, while recognizing that broader clinical and OOD validation is still required. 

---
# SFA: A Statistical Physics-Inspired Diffusion Transformer for Efficient and High-Quality Image Generation 

**Authors**: Zixuan Huang, Tianhao You, Guozhang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.00027)  

**Abstract**: Diffusion transformers have achieved remarkable success in image generation, but accelerating attention while preserving generation quality remains a major challenge at high resolutions. Inspired by statistical physics and the widespread occurrence of scale-free laws in nature, we introduce Scale-Free Attention (SFA), which preserves fine local interactions while summarizing distant regions at progressively coarser scales. For $N$ tokens, mean pooling, token-count correction, and a shared softmax over a scale-free spatial hierarchy give SFA $O(N\log N)$ complexity in both forward and backward passes. Our Triton implementation achieves $357.3\times$ forward and $265.6\times$ backward attention speedups over FlashAttention at 262,144 tokens. After 40 epochs of PixelFlow training from scratch on ImageNet at $256\times256$ resolution, SFA achieves the best generation quality among five attention mechanisms, including full attention. Fréchet inception distance evaluated on 50,000 generated samples (FID-50k) improves from 13.25 for full attention to 12.51, and the advantage over full attention persists through 80 epochs. SFA also achieves the best generation quality among the compared efficient-attention methods on Flickr-Faces-HQ (FFHQ). Beyond training from scratch, Stable Diffusion 3.5 Medium with attention replaced by SFA retains the generation quality of its full-attention counterpart under matched fine-tuning budgets. We obtain similar results when fine-tuning pretrained PixelFlow on ImageNet. SFA establishes scale-free structure as an effective design principle for efficient, high-quality diffusion transformers. 

---
# Intervenable Concept Dynamics for Video Activity Forecasting 

**Authors**: Patrick Knab, David Steinmann, Giovanni De Felice, Felix Divo, Udo Schlegel, Wolfgang Stammer, Sascha Marton, Christian Bartelt  

**Link**: [PDF](https://arxiv.org/pdf/2611.00020)  

**Abstract**: Concept bottleneck models route predictions through human-understandable concepts, making their intermediate states inspectable and editable. Extending this model family to forecasting requires modeling how concept activations evolve over time as objects, actions, and their interactions change, yet existing CBMs ground concepts only in observed inputs and do not forecast their future evolution. We study this problem using videos of ongoing activities and forecast future activities through a named-concept state. We propose TRACE-CBM (Temporal Relational Activity Concept Evolution), which first refines observed concept activations with sparse spatio-temporal graphs. It then recursively predicts future concept states, conditioning each transition on the prior activity distribution, and decodes both observed and predicted states with a shared activity classifier. At inference time, the model supports editing concept values, activity activations, and learned edges without parameter updates, exposing the relational spatio-temporal dependencies used for forecasting. Across multiple datasets, TRACE improves observed-activity Top-1 accuracy over concept-based methods by 1.2-8.5 percentage points and attains H3 forecast Top-1 accuracy within 1.9-2.2 points of matched black-box forecasters. Intervention experiments show concept-node edits correct 25.6-34.4% of initially incorrect H3 forecasts without flipping initially correct ones, and concept steering achieves the requested target activity in up to 90.0% of cases. 

---
# Beyond the Last Layer: Global Inter-Layer Aggregation for AI-Generated Image Detection 

**Authors**: Wenbin Shen, Zhichao Lian  

**Link**: [PDF](https://arxiv.org/pdf/2611.00017)  

**Abstract**: The rapid development of generative models has made AI-generated images increasingly close to real images. It has also brought several challenges to detectors, including rapid generator updates, large changes in data distribution, and limited cross-dataset generalization. Many existing methods based on pretrained vision encoders primarily use the global representation from the last layer, potentially leaving useful information from intermediate layers underexplored. To address this issue, we propose Global Inter-Layer Aggregation (GILA). GILA extracts global representations from different depths of a frozen encoder, models interactions among layer representations through inter-layer self-attention, and adaptively aggregates the resulting contextualized representations. The method is not tied to a single pretraining objective and does not require fine-tuning the pretrained encoder. We evaluated GILA on three vision backbones with different pretraining objectives, ten seen generators, and three external datasets. The experimental results show that GILA makes better use of multi-layer information in pretrained encoders and improves detection performance and cross-dataset generalization. 

---
# Output-Aware Block Influence: Jacobian-Lens Weighting for Depth Pruning 

**Authors**: Kyle Lemoi, Jonathan Yong, Shubhankar Tripathy, Priestley Fernandes, Anirban Majumder, Zarreen Reza  

**Link**: [PDF](https://arxiv.org/pdf/2611.00015)  

**Abstract**: Depth pruning removes entire transformer blocks and therefore depends on a reliable criterion for identifying which blocks can be removed. Block Influence (BI) ranks blocks by how minimally they rotate the hidden state. However, local redundancy does not necessarily imply downstream irrelevance. We address this limitation using the Jacobian Lens (J-Lens) neural artifact, a map transporting perturbations at a layer to the final residual stream. At a common sparsity level, pruning scores applying J-Lens transportation to $BI$, what we call $J-BI$, were found to improve post-pruning accuracy by up to 19 points over $BI$ while reducing damage to perplexity for Qwen3-8B, Llama-3.1-8B-IT, and Gemma-3-12B-IT. We also perform a single-block ablation study and use J-Lens probing to explain why the proposed scores improved performance. 

---
