# Language Models Steer on Injected Uncertainty Tokens, but Fail to Self-Monitor In Situ 

**Authors**: Daniel Theron  

**Link**: [PDF](https://arxiv.org/pdf/2611.31228)  

**Abstract**: Prompting language models to verbalize stepwise numerical confidence during generation is an increasingly popular approach for triggering backtracking and error correction in agentic reasoning. This strategy presupposes two capabilities: (1) in-situ monitoring fidelity (does confidence drop when an error occurs?) and (2) downstream steering control (does an uncertainty token causally steer the model to repair its error?). We evaluated four open-weights models (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B, and Phi-4, 14.7B) across 15,200 math benchmark attempts, isolating 2,017 reasoning errors within each model's capability frontier. Applying 27,568 counterfactual continuations across these failure points, we demonstrate a consistent decoupling between monitoring and control.
On monitoring, all four models fail to detect their own errors. Verbalized confidence does not distinguish mistake steps from sound reasoning, yielding near-chance within-position AUROCs between 0.500 and 0.561. Even at a model's first reasoning error, trajectory confidence remains saturated at 95.1% to 99.9%, with noticeable drops ($\ge 10$ points) appearing on fewer than 9% of errors in Llama and under 1.6% in the other three models. In contrast, models respond readily to external uncertainty: injecting a low-confidence tag (10%) at the error step significantly increases downstream correction across all models (+3.7 to +22.5 points over format-matched 95% certainty; all $p \le 0.0022$), reaching up to 38.5% under active rethinking instructions. However, injecting doubt onto already-correct steps causes models to discard valid reasoning, reducing solve rates by up to 22.8 points. Inference-time self-correction is thus bottlenecked by in-situ error detection rather than downstream repair capacity: models readily steer toward recovery when prompted with doubt, yet fail to self-monitor during generation. 

---
# Seeking in Space: Benchmarking Purposeful Active Exploration of Vision-Language Models for Spatial Reasoning 

**Authors**: Xiwei Liu, Yulong Li, Xuhui Li, Xinlin Zhuang, Zhixiang Lu, Haolin Yang, Jianxu Chen, Yifan Lu, Zhiran Zhang, Yutong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2611.31215)  

**Abstract**: Humans explore unfamiliar spaces with purpose, deciding where to look until they have gathered the evidence a spatial question requires. However, can vision-language models (VLMs), which reason increasingly well over curated images and videos, also seek in space when the evidence is not handed to them? We present ActSpace-Bench, a benchmark of 994 questions across 234 exploration episodes in 72 real-scanned indoor scenes, where a model starts from a single view, navigates real captured frames, and must gather evidence before answering. To assess whether such exploration is purposeful, we introduce a controlled evaluation protocol that fixes each model as the reasoner and compares its own exploration against a same-budget random walk and an oracle route, separating evidence seeking from spatial reasoning. Every question is verified to require exploration and answered from 3D annotations, masked-evidence questions test whether models know when to abstain, and over 95% of sampled questions are confirmed correct by human verification. Evaluations of 23 proprietary, open-weight, and spatially specialized VLMs reveal that the strongest models do seek evidence with purpose but lose much of it to limited memory, whereas weaker and specialized models explore no better than a random walk or fail to reason over the evidence they are given, pointing to memory and evidence-grounded reasoning as the next targets for active spatial intelligence. 

---
# Know When to Abstain: Conformal Confidence Estimation for Text-to-SQL 

**Authors**: Avinash Kumar Singh, Bhaskarjit Sarmah, Stefano Pasquali  

**Link**: [PDF](https://arxiv.org/pdf/2611.31190)  

**Abstract**: Text-to-SQL systems often return queries that execute successfully but answer the wrong question. Detecting these errors at deployment is hard because execution correctness is defined against a gold query, which exists only in labeled data. We formulate confidence estimation for Text-to-SQL as probabilistic query validation under a latent correctness variable that is observable at calibration time through gold-SQL execution and latent at deployment. Our estimator separates generation from validation. A generator LLM writes and executes the query, and a different critic LLM judges the query and its execution result. The critic's validation and execution-grounded alignment scores are fused with the generator's verbalized confidence, token-level uncertainty, and SQL-structural features in a Random Forest, using two sequential LLM calls. Two calibration steps give finite-sample guarantees. A Learn-then-Test threshold ensures, with probability $1-\delta$, that the error rate among accepted queries stays below a user-chosen budget. A conformal p-value bounds the false abstention rate. On Spider we use GPT-OSS-120B as generator and Gemma-4-31B as critic. The estimator obtains an AUARC of 0.899. Without abstention, 22.6% of queries are wrong. With a 15% risk budget ($\delta=0.1$), a threshold is certified on all 10 resampled splits, and the system accepts 52.4% of queries on average with 9.1% observed error. Five different critic models give similar results. In a controlled comparison on Spider with Olmo-3-32B as generator, the estimator obtains an AUARC of 89.5, compared with 81.2 for the strongest logit-based baseline. Applied to BIRD without retraining, the confidence scores remain informative (AUROC 0.72, against 0.79 on Spider), but the risk guarantee does not transfer, as expected without exchangeability. 

---
# Needwise: Generation-Free Conversational Recommendation from Need-Level Typed Decisions 

**Authors**: Sungchul Kim, Weixiao Huang, Mekala Raja Sekhar Reddy, Amrutha Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2611.31175)  

**Abstract**: Conversational recommenders must rank items against needs that catalogs record only in part, while users add, withdraw and reverse needs and switch between requests. LLM rankers re-read the dialogue at every turn: they are slow, and their ordering does not show which needs an item meets. We show that a conversational recommender can be built from typed decisions instead: named questions over fixed options, answered with option probabilities rather than generated text. On each turn, one typed request maps the message onto the catalog's fields, keeps what no field expresses in the user's words, and decides how the turn changes each earlier need. Needwise then ranks candidates must-first: the needs on the kind of item, its place and its price form a tier, and within it one holistic decision and need-level verdicts (supported, contradicted or unknown) are combined with equal weight (Needwise-HF). With the graded attributes withheld from the records, so that needs must be inferred from names, descriptions and reviews, Needwise-HF ranks Google Local requests significantly better than GPT-6 Sol (0.383 against 0.343 nDCG@10) and is not significantly different from the strongest LLM ranker on Google Local and Yelp; these comparisons keep their sign under every treatment of unrecorded attributes we test. It costs a fraction of GPT-6 Sol's API price and answers in 2.5 s at the median on Google Local, against 5.6 s for GPT-6 Sol. In conversations, the typed state keeps the ranking across switches and returns, but it recognizes only 19-27% of withdrawn needs. 

---
# Reducing Order Sensitivity in Decision Models 

**Authors**: Bodong Liu, Dongyoung Go, Sofian Zalouk, Kilian Q. Weinberger  

**Link**: [PDF](https://arxiv.org/pdf/2611.31174)  

**Abstract**: Decision models are emerging as fast components of larger AI systems. Yet recent decision models, such as Jev, process options sequentially, leading to different predictions based on their ordering. Across ten open- and closed-source decision models, we find that \emph{order-sensitivity} is pervasive, causing substantial accuracy degradation and unreliable decision-making. These effects become more pronounced as the number of options grows and can compound across multi-step decisions. To address this problem, we systematically compare eight methods for mitigating order sensitivity in terms of robustness, predictive accuracy, and inference efficiency. Among these methods, we find that \emph{tree attention} is a surprisingly effective and efficient method. It achieves order invariance by processing options as parallel branches of a shared context. When used during training, it matches the baseline's average accuracy. Applied to the baseline without retraining, it also eliminates order sensitivity with only a 1.1-point drop in accuracy. Our findings demonstrate that order invariance can be enforced directly in decision models with little or no loss in predictive accuracy. 

---
# When Does Overriding the Majority Vote Help? 

**Authors**: Terry Ma, Aidan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31165)  

**Abstract**: When should additional reasoning or verification replace a majority-vote answer? We call such a replacement an override. We examine two obstacles: the availability of correct alternatives and the reliability of methods that produce or select them. We compare additional reasoning, executed programs, and cross-model reasoning on 150 competition-mathematics questions with three open reasoning models. Starting from a majority vote over 16 responses, we measure how often overrides occur and how many errors they correct or introduce across agreement thresholds. When the vote is incorrect, the correct answer is often absent from the original responses. Additional same-model reasoning rarely overrides, while executed programs correct more errors than they introduce at every evaluated threshold. Cross-model overrides improve some weaker baselines but do not establish an advantage over the more accurate model alone, even with only eight standalone responses. Stricter agreement reduces how often overrides occur but does not consistently increase the fraction that correct errors. At the strictest thresholds, overrides occur on few distinct questions, limiting conclusions about their reliability. 

---
# Habitats: A Framework for Open-Ended Evolution of Agent Objectives 

**Authors**: Nicolas Bolouri, Gaya Mehenni, Daniel Toyama  

**Link**: [PDF](https://arxiv.org/pdf/2611.31160)  

**Abstract**: Designing what an agent should optimize can be as difficult as optimizing the agent itself. Existing language-model-based reward-generation methods typically refine objectives for predefined tasks and couple the decision of what to explore with the synthesis of how to implement it, potentially narrowing the resulting behavioral repertoire. We introduce Habitats, a framework for open-ended evolution of executable agent objectives. Habitats maintains a population of reward-policy pairs and evaluates every policy under every population reward, producing a cross-fitness matrix that exposes self-mastery, transfer, and functional diversity. A population-level Planner uses this information to select parent rewards and issue targeted mutation directives, while a separate Mutator translates those directives into validated reward programs. We evaluate Habitats in a discrete Card Sandbox and three continuous-control environments (Cart-Pole, Hopper, and Cheetah) against Independent Generation and Direct Mutation baselines. Across four environments and three random seeds, Habitats achieves the highest macro-averaged final Specialist Quality and Behavioral Diversity, with final diversity reaching 1.8x that of Direct Mutation and 2.5x that of Independent Generation. Finally, across ten downstream objectives unseen during evolution, frozen Habitats policy libraries achieve higher paired win rates than size-matched Direct Mutation and Independent Generation libraries at 10,000 adaptation episodes, while also surpassing task-specific policies trained from scratch for 20,000 episodes. These results suggest that separating population-level planning from reward synthesis enables the discovery of diverse, learnable, and reusable behaviors. 

---
# Genome-to-Image Translation: Reconstructing Escherichia coli Morphology from Genomic Sequences 

**Authors**: Son T. Huynh, Thuy T. Le, Thu H. A. Huynh, Binh T. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.31132)  

**Abstract**: Antimicrobial resistance in Escherichia coli is established in routine practice by culture-based susceptibility testing, which is slow relative to the clinical need for an early treatment decision. Resistance status also leaves a measurable trace in cell morphology, and whole-genome sequencing of clinical isolates is increasingly available, which raises the question of whether colony appearance can be predicted directly from an isolate's genome. We formulate this as a conditional image-generation problem. The assembled contigs of each isolate are encoded into a single 768-dimensional vector by a frozen genomic language model through a segment-contig-isolate aggregation hierarchy, and a conditional Wasserstein GAN with gradient penalty synthesizes colony images from this embedding together with the applied antibiotic and the resistance label. We train on 167,356 single-cell images from 22 clinical blood isolates imaged under azithromycin, ciprofloxacin, tetracycline, and a no-drug control, compare seven pretrained convolutional backbones inside the generator, and select ConvNeXt-Tiny on distributional image-quality metrics. The synthetic images are then assessed extrinsically: added to a reduced real training set, they improve per-antibiotic resistance classification, with a significant gain in resistant-class recall. 

---
# Self-Upgrading Embodied Agents 

**Authors**: Jiajun Fan, Neil He, Qinglong Zheng, Changyi Yang, Tong Wei, Andy Mao, Zetian Li, Chaoran Cheng, Hui Ren, Ruihan Guo, Xianrui Zhong, Xizheng Yu, Jingyuan Li, Xi Zhu, Kaiqu Liang, Chang Xu, Zun Qian, Zidi Xiong, Yuezhu Zhao, Xu Huang, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.31117)  

**Abstract**: An embodied agent built on a language model goes to work with the weights it was given and keeps them for life. To get better, it needs more training, or reflection, memory, search or prompts that cost every step and are lost when the task changes. Editing weights by reasoning, weight-space programming, could end this, but has stayed with models that only reason or relied on a far larger commercial model. We introduce the self-upgrading embodied agent, to our knowledge the first that learns from its own experience by upgrading its own weights on the job, between its tasks, with no training, no gradient, no reward model, no second model, no test-time scaling and no harness or prompt engineering. By day a 27B agent works in a household environment. At night it reads its record, finds many of its steps wasted on replies that reach the token limit before any action, traces the habit to one direction in its hidden states, and rewrites one tensor of its weights. An agent that rewrites itself must first never break itself, so it is its own examiner: it tests every change against its unchanged self on practice tasks and keeps only what holds. When an edit holds, the next day it is a better agent, at the same cost per step: over thousands of held-out episodes, 48 fresh seeds a task, its one kept edit makes it complete significantly more tasks, in fewer steps, with fewer wasted and fewer tokens. This self-improvement is multi-round, stable and continual: through the nights that follow, the upgrade holds, without collapse. Of four sizes of its family, only the 27B agent does this. From here on, an embodied agent need not keep training and inference apart: as it works, it gathers its own experience and upgrades itself, with no separate pre-training or post-training and no human intervention. We believe this opens lifelong continual learning for robots: stable, continual, recursive self-improvement. 

---
# Preflight: Agentic Last-Mile Verification for AI Software Supply Chains 

**Authors**: Yaman Jandali, Nojan Sheybani, Farinaz Koushanfar, Tara Javidi  

**Link**: [PDF](https://arxiv.org/pdf/2611.31110)  

**Abstract**: Software dependencies execute within the trust boundary of downstream applications, yet provenance, static analysis, and package-level behavioral approaches do not directly determine how a released artifact behaves for a specific program. We present Preflight, a program-scoped pre-execution verifier for Python dependencies that extracts application-referenced symbols, iteratively synthesizes and validates function-level harnesses, executes resolved artifacts in a restricted environment, records behavioral fingerprints, and compares target behavior against prior versions when suitable baselines exist. We evaluate Preflight on four real software-supply-chain incidents under five application contexts, a 100-package benign corpus, and a controlled function-level versus import-only ablation. Preflight returns malicious verdicts in four incident contexts and a clean verdict in the LiteLLM client context, where the affected proxy-server path is not exercised. All 100 evaluated benign packages receive clean verdicts, with zero observed false positives. In a controlled ablation on Ultralytics and ctx, function-level harnesses expose malicious behavior in both incidents, whereas import-only probing exposes neither. Together, these results show that conditioning behavioral verification on the downstream program can expose security-relevant dependency behavior while remaining selective on benign workloads. 

---
# LiveLoRA: Exact KV Cache Migration Across LoRA Updates for LLM Agents 

**Authors**: Tianyi Huang, Samuel Xu, Ivy Gu, Nathan Huang, Siqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31105)  

**Abstract**: Updating an LLM adapter should not require replaying the full interaction history of every active session. Yet reusing key-value (KV) cache entries computed under an earlier version mixes model versions, while full-history re-prefill becomes increasingly costly as histories and session fleets grow. We introduce LiveLora, a model-runtime co-design for exact final-block KV cache migration. Frozen lower decoder blocks and fixed K/V adapter down-projections preserve each token's base projections and low-rank coordinates across adapter versions. The runtime reconstructs candidate K/V, reevaluates the latest token through the final block, and commits the adapter, cache, and next-token logits together. We prove continuation equivalence to full-history re-prefill in exact arithmetic under the stated model assumptions. On Qwen2.5-3B-Instruct, median paired publication speedups over full-history re-prefill average $68.98\times$ across five runs for 32 active sessions with 8K tokens each. Serving benchmarks show approximately 4.8% decoding overhead relative to native cached decoding. Compared with retained-boundary replay, compact retention gives a $3.42\times$ reduction in additional session storage at 8K, with similar measured publication-plus-decoding cost. LiveLora allows persistent LLM agents to adopt new adapter versions without replaying their full interaction histories. 

---
# SayWhen: Set-Valued Semantic Commitment for Low-Regret Early Speech 

**Authors**: Tianyi Huang, Lucas Su, Fred Zhang, Andy Liang, Gordon Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.31101)  

**Abstract**: Speaking early turns a prediction into an irreversible public commitment. We introduce SayWhen, a test-time controller that selects a task-directed act shared across typed continuation plans and controls its release as speech. A persistent veto blocks playback when intervening transcripts invalidate the candidate. We define regret-free early speech (RFES) to measure supported pre-endpoint releases without scored incompatibility. On a 72-prefix branching diagnostic, the generated-plan frontier attains 6.0% RFES with zero observed mean regret, versus 8.2% RFES and 0.068 regret for Matched Top-One. Measured planner latency constrains this path; a deterministic retail schema attains 31.7% RFES, compared with 32.0% for the complete policy. In a separate 30-task $\tau$-Voice simulation, schema-derived candidates with cached text-to-speech (TTS) audio attain 21/30 task-level RFES. Eligible audio begins 1.51 seconds before the user-audio endpoint among the 22 tasks with early speech, or 1.11 seconds over all 30 tasks with zero for tasks without eligible early speech. Semantic veto and endpoint waiting each pass 16/30 tasks. Together, the studies demonstrate a controlled coverage-compatibility trade-off and schema-based pre-endpoint operation, making the supported public commitment an explicit unit of voice-agent control. 

---
# Tradeoffs between objective outcomes and human enjoyment of work in human-AI task allocation 

**Authors**: Katherine M. Collins, Elizabeth Mieczkowski, Raja Marjieh, Marlene Berke, Joshua B. Tenenbaum, Max Kleiman-Weiner, Ilia Sucholutsky, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2611.31095)  

**Abstract**: For people, tasks at work are defined by more than their outcome: they are colored by the experience of engaging in that task. As AI systems become increasingly capable of performing human-level work they may be deployed to tasks that people find enjoyable while simultaneously creating comparatively more aversive tasks such as monitoring AI outputs. This creates a thorny tradeoff for managers: whether work should be assigned purely based on the expected performance of the workers in question, or also account for the subjective experience of the person potentially doing the task. Most economic accounts focus on expected utility calculations based on the former. Here, we assess how people balance objective outcomes and more subjective qualities like enjoyment. In a series of large-scale (N=800) behavioral studies in which people allocate tasks to either an AI or human worker we find that people's allocation decisions are best captured by a model that systematically balances relative objective performance with the human worker's expected enjoyment, weighting human experience higher when a task is framed as inducing less enjoyment and adjusting based on the stakes of the task. As AI also makes decisions about work allocation, we compare a set of LLMs. We find many models also deviate from purely objective decisions, particularly when the project stakes are low, but there is wide heterogeneity as to how much and in what manner they weigh enjoyment. These results raise questions about what role human experience may play in decisions around where and how to integrate AI into workplaces. Integration may be jagged across settings due to consideration not only of the relative capabilities of workers, but also the human experience of work, the valence of that experience (e.g., more strongly weighing aversion), the context (e.g., stakes), and who is making or guiding such choices (human or LLM). 

---
# Revisiting Learning to Delegate: LLM Discovered Heuristics for Large-Scale Vehicle Routing 

**Authors**: Zhengyang Wang, Renfei Tan, Shuze Daniel Liu, David Simchi-Levi  

**Link**: [PDF](https://arxiv.org/pdf/2611.31090)  

**Abstract**: Efficient vehicle routing is critical to logistics and transportation, yet solving large-scale capacitated vehicle routing problems (CVRP) remains computationally challenging due to their combinatorial complexity. One approach to solve large-scale vehicle routing optimization is to identify promising subproblems and then optimize them using solvers. In this paper, we propose a framework of automatic heuristic design (AHD) in which a large language model (LLM) autonomously refines heuristics for selecting subproblems, enabling effective optimization of large-scale CVRP instances. Compared to prior approaches that train task-specific neural networks to select among heuristically constructed subproblems, our method discovers generalizable, interpretable LLM-generated heuristics to directly construct subproblems. Experiments show that our approach achieves comparable in-distribution performance while outperforming neural network-based policies on out-of-distribution instances with large-scale problems. 

---
# Orthogonally Mixed Sign Coding with Progressive Residual Refinement for Transformer Weight Compression 

**Authors**: Anindya Roy Chowdhury, Atanu Barai, Nandakishore Santhi  

**Link**: [PDF](https://arxiv.org/pdf/2611.31089)  

**Abstract**: LLM quantization reduces storage but often degrades predictive fidelity at low bit rates. We propose a training-free, CDF-free, Gaussian-CLT-motivated progressive residual-sign quantization framework that uses randomized orthogonal mixing to promote Gaussian-like residual distributions and improve sign based reconstruction of Transformer weights. Each stage independently normalizes residuals and analytically calibrates reconstruction, progressively recovering quantization errors. The framework enables flexible storage-fidelity trade-offs without learned codebooks, empirical CDFs, or trellis optimization. We evaluate GPT-2 Large and Llama-3-8B across attention and MLP projections. At five residual stages, joint quantization preserves near-baseline predictive fidelity, achieving perplexities of 32.16 and 12.73, compared with full-precision values of 29.89 and 11.07, respectively. Ablations further demonstrate that orthogonal mixing substantially improves reconstruction over identity-based sign coding, while activation-aware scaling improves output fidelity and reduced-precision statistics lower storage overhead with negligible perplexity degradation. Comparisons with conventional quantizers establish the method's progressive fidelity recovery and characterize its storage-quality and runtime trade-offs. These results demonstrate the effectiveness of orthogonally mixed residual-sign coding for flexible, training-free Transformer weight compression. 

---
# ChiProgramBench: Can LLM Agents Rebuild Efficient Circuits from Pin-Level Experiments? 

**Authors**: Tianrun Gou, Qingyao Yang, Wendong Xu, Liang Chen, Kuan Li, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2611.31082)  

**Abstract**: Understanding a system that can be run but not inspected is a core engineering skill: one designs experiments, forms hypotheses about hidden state and timing, discards those that observations refute, and finally builds a replacement that works and is efficient. To evaluate this capability of LLM agents in the hardware domain, we present ChiProgramBench, 18 tasks derived from open-source processors, accelerators, codecs, transforms, interconnects, and control circuits. Agents can only drive and read the pins of a hidden, stateful simulated reference, then submit a gate-level netlist made of public library cells. Agents work in two settings: black-box discovery gives only anonymized functional pins and the interfaces of fixed memories, while function-guided reconstruction adds the circuit's purpose and each pin's role. Hidden test cases compare outputs with the reference cycle by cycle; power, delay, and area earn credit relative to the reference only when every case passes. Across 432 trials of 6 LLM agent configurations, successful agents recover algorithms and numerical conventions, yet only 12.0% of black-box and 14.8% of function-guided trials pass every case. Function-guided reconstruction raises the mean functional score, the cycle-weighted share of passed cases, from 24.6% to 30.9%. Of the 58 fully passing designs, only three improve on the reference's combined power, area, and delay, so behavioral recovery and efficient implementation remain distinct challenges. 

---
# AlphaDefect: Expert-Level AI for Slay the Spire 2 

**Authors**: Qiyue Chen, Taian Yu, Junpei Komiyama, Haobo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31076)  

**Abstract**: Slay the Spire 2 (STS2) is a single-player deck-building game that requires long-term planning under heavy randomness. We present AlphaDefect, an expert-level AI system for STS2: for each character, one network trained with proximal policy optimization (PPO) makes every decision of a run, in combat and outside it. At the highest difficulty, Ascension 10, on two versions of the game current in November 2026 and on test seeds not played in training, AlphaDefect wins 62-77% of runs across six characters with greedy play and 68-80% with test-time search, after 1.3 to 6.7 billion decisions of training per character; runs uploaded by human players of all levels are won 11-20% of the time at the same difficulty. The networks read the game as entity records that identify each card, relic and enemy through a learned embedding and attributes from the game data, together with consequences of public information computed by the engine. Ablations show how much the agent depends on this encoding: without these identities, the network did not win a single run, and without the computed consequences it won 8.3 points fewer. To our knowledge, AlphaDefect is the strongest publicly reported STS2 agent as of November 2026. 

---
# Decoupling Solving from Abstention: Single-Rollout Conformal Risk Control for Hard-Budget LLM Reasoning 

**Authors**: Qianyi Chen, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.31065)  

**Abstract**: Hard token budgets often interrupt LLM reasoning before an answer is produced. Forcing reasoning to terminate recovers correct solutions but also increases confident errors, creating a joint challenge of solving within budget and deciding when to abstain. Existing remedies are costly or lack guarantees: supervised fine-tuning offers no error control, penalizing wrong answers during reinforcement learning (RL) can suppress exploration, and multi-sample conformal methods multiply inference cost. We propose Tail-CRC, a single-rollout framework that separates candidate generation from submission control. We first adapt the solver to budget forcing with pure binary correctness rewards; after freezing it, we apply conformal risk control to upper-tail token surprise in the answer span. Each rollout yields both an answer and an uncertainty score from token log probabilities, enabling finite-sample control of expected wrong submissions under exchangeability and recalibration across risk targets or domains without retraining. We show that wrong-answer penalties create failure-only comparisons in group-relative RL, derive an exploration-starvation boundary, and identify length-induced ranking reversals and single-token spike sensitivity in answer scores. Across Qwen3.5-4B and Qwen3-8B, 50 RL updates on 400 problems improve correct-submission rates by 15.0--29.6 points on DAPO and 4.3--12.8 points on out-of-domain Omni-MATH over inference-only budget forcing, while reducing generation cost by 3.5--12.6\%. On 8B Omni-MATH, target-domain calibration cuts wrong submissions from 30.3\% to 5.8\% at 4K and from 41.8\% to 6.5\% at 8K. At 4K, generation is 2.05$\times$ faster for 4B and 2.23$\times$ faster for 8B than unforced 8K decoding in local amortized batch time. Tail-CRC thus enables reliable, efficient budgeted reasoning from a single rollout. 

---
# Automatic Occupational Coding through Retrieval-Augmented Generation 

**Authors**: Mathijs A. Langezaal, Egon L. van den Broek, Lukas P.A. Arts, Marcel Goldberg, Grégoire Rey, Melissa C. Friesen, Sarah J. Locke, Nathaniel Rothman, Qing Lan, Roel C.H. Vermeulen, Susan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2611.31051)  

**Abstract**: Annually 1.9 million deaths and 89.7 million disability-adjusted life years are caused by exposure to agents during work (e.g., welding fumes, silica dust, and noise pollution). To improve worker health, high-quality exposure assessments are crucial. Therefore, free-text job descriptions from large-scale occupational cohorts are manually standardized through occupational classifications and linked to hazardous exposures using Job Exposure Matrices (JEM). However, manual coding is slow, unreliable (i.e., 44-89% inter-coder reliability), and costly, due to i) short, ambiguous free-text job descriptions, ii) classifications containing hundreds to thousands of categories, and iii) the absence of gold standards. With 46%-79% accuracy, state-of-the-art automatic occupational coding tools achieve classification performance on human level. However, their performance drops to 17-26% on out-of-distribution datasets. Furthermore, their supervised architectures limit them to occupational classifications present in the training set. Consequently, JEM using other classifications are incompatible, leaving vital exposure-response associations unexplored. Our Retrieval-Augmented Generation framework, OPERAG, enables classification into any occupational classification without expert-labelled databases. It retrieves coding index information and combines this with Large Language Model-based classification. Validated on AsiaLymph's 36,062 real-world job descriptions, OPERAG achieved 42.6% occupational classification accuracy on the 4th coding level, substantially outperforming existing tools and approaching human coders' 51-56% inter-coder reliability. Using two JEM named ALOHA+ and SHOCK, 89.9-98.6% accuracy was achieved on exposure assessment. OPERAG approaches human-level, universal occupational classification, enabling efficient and effective large-scale exposure assessment to improve worker health. 

---
# Grower-in-the-loop AI for Deployable Precision Fertilisation in Sustainable Agriculture 

**Authors**: Xulong Wang, Yu Zhang, Gaoshan Bi, Menghui Zhou, Simon Jeffery, Max Davis, Sam Maycock, Paul Harris, Po Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31044)  

**Abstract**: In agriculture, typically over half of the applied nitrogen fertiliser is lost to water and to air. Precision fertilisation aims to reduce these losses while maintaining yield and quality, yet existing tools struggle to capture farm-specific challenges in determining optimal nitrogen rates. Static recommendations are common and overlook on-farm constraints, such as delays due to weather or machinery availability, or change in budgets. Each constraint governs when and how growers apply fertiliser in practice. Here we present a grower-in-the-loop framework for winter wheat in England, combining baseline fertilisation planning from a machine learning-derived nitrogen-response curve with in-season adaptation via a large language model. We evaluated the framework using data from long-term experiments (LTEs), replicated field trials, and farm deployments. Retrospective model validation using LTEs confirmed low economic decision-loss across multiple seasons. Using field and farm trials, baseline schedules were shown to match standard-practice yields with less nitrogen, whereas in-season adjusted schedules additionally retained grain quality standards. Given this enhanced performance, in-season adaptation absorbed weather, machinery and budget constraints to keep recommendations executable. The framework therefore shows promise for translating static recommendations into adaptable, grower-informed decisions for sustainable fertiliser management. 

---
# Boltz-Flow: Generating High-Quality Robot Action from Boltzmann Machine 

**Authors**: Hengzhi Jiang, Haixu Liu, Jindou Jia, Tinglan Wang, Shaobo Chen, Yiren Zhao, Ziyang Rao, Laiyuan Cui, Jiaxin Dong, Yin Ma, Hai Wei, Jiaqi Bai, Tuo An, Xinyu Zhou, Hao Ban, Hongdong Zhu, Yanqi Wang, Jiaqi Dou, Nuo Qian, Qi Liu, Bin Jin, Chang Wu, Qingfeng Li, Hui Yang, Weiyu Guo, Dicong Qiu, Qi Gao, Jianwei Niu, Kai Wen, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31038)  

**Abstract**: Flow-matching policies integrate a velocity field from a fixed, observation-independent Gaussian source. However, since the source is unimodal and identical across all tasks, the limited representational capacity of Gaussian distributions constrains how far a policy can scale with growing task diversity. Moreover, the source factorizes over time steps and control dimensions, discarding the coordination that manipulation requires. We introduce Boltz-Flow, which replaces Gaussian initialization with a more powerful, observation- and instruction-conditioned Boltzmann source while leaving the base VLA unchanged. Boltz-Flow parameterizes this source with a conditional Boltzmann machine that assigns an energy value to each binary source state. A listwise objective, supervised entirely by offline demonstrations, aligns this energy with the quality of actions produced after frozen flow integration. At inference, the energy defines an Ising instance whose solution is decoded into continuous noise before a single standard integration. We use a physical coherent Ising machine during training. Experiments show that Boltz-Flow improves $\pi_{0.5}$ across three benchmarks, outperforming the compared strong policy baselines and competing noise optimization methods on LIBERO and RoboTwin 2.0 and ranking second only to LEAP on LIBERO-Plus. Additional experiments show gains under noise-adapted fine-tuning, with action-to-action warm starts, and on the SmolVLA backbone. 

---
# AI-Assisted Auto-formalization for Trustworthy Scientific Computing: Case Studies and Lessons Learned 

**Authors**: Mohit Tekriwal, Het Mankad, Naifeng Zhang, Keita Teranishi, Sahil Bhola, Daisuke Uchida, Jean-Baptiste Jeannin, Franz Franchetti, Karthik Duraisamy, Jeffrey Vetter, Yohann Dudouit  

**Link**: [PDF](https://arxiv.org/pdf/2611.31033)  

**Abstract**: Agentic AI tools have recently had success in mathematical reasoning. Both general purpose coding agents such as Claude Code and Codex, and specialized mathematics agents such as Aristotle, have been used for formalizing mathematical problems such as Fermat's last theorem, Erdős problems and several IMO level math problems. However, much of these efforts are concentrated in the pure mathematics domain, which benefits from relatively self-contained theorem statements, fixed mathematical representation and extensive support in existing formal libraries. Applied mathematics, on the other hand, presents additional challenges because the source may describe a computational procedure or modeling transformation rather than a declarative theorem. In this work, we present a systematic study of AI-assisted auto-formalization for scientific computing (SC), a domain in applied mathematics central to U.S. Department of Energy applications. Through case studies spanning hierarchical abstraction layers in SC software stack, we evaluate not only whether AI agents can complete numerical proofs, but also where and why they fail across the stack. Our primary contribution is a cross-layer taxonomy of failure modes and an analysis of the form of expert feedback required at each layer to recover mathematical soundness. From these observations, we derive a workflow for recurring proof patterns and practical guidance for using AI agents while preserving mathematical soundness. Our results characterize both the practical value and the current limitations of AI-assisted formalization for trustworthy scientific computing. 

---
# ComputeWise: Measuring and Fixing GPU Waste by AI Research Agents 

**Authors**: Egor Pakhomov, Nina Pakhomova, Anurag Koul, Shafiq Joty, Erik Nijkamp  

**Link**: [PDF](https://arxiv.org/pdf/2611.31027)  

**Abstract**: AI agents are given GPUs to do AI research on their own. Do they use those GPUs well? A busy GPU does not tell us, because it can be busy with work the answer did not need. This matters, because the tokens the agent spends cost almost nothing next to the GPUs its experiments run on, so using those GPUs better has a huge impact. So we built ComputeWise, an open benchmark of research tasks on which this waste can be measured and reduced. For it we developed our own simulated GPU, which replays recorded runs to the agent's code. With no real GPU needed, we ran hundreds of experiments, far more than would be reasonable on real GPUs. With no help, the agents put only 52% of their GPU time into work the answer needed. Another 22% went to work it did not need, and 26% sat idle. Asking the agent to finish the work faster, telling it to keep every GPU busy, or giving it a job scheduler does not fix this. So we built an open-source harness, part prompting and part deterministic code. Across five models, it removed over 80% of the extra time to the answer. 

---
# Triage Accuracy of Jev and Open-Weight System One Decision Models and Their Use for Early Emergency Recognition in Symptom-Assessment Applications 

**Authors**: Marvin Kopka, Austen El-Osta, Christoph Kollwitz, Felix Balzer, Florian Bontrup  

**Link**: [PDF](https://arxiv.org/pdf/2611.31012)  

**Abstract**: Laypeople increasingly use large language models (LLMs) for triage advice, but LLMs answer in variable free text, which makes it hard to evaluate them for use in medical devices. Decision models, which assign a probability to each predefined triage category, may be easier to evaluate. To test their triage accuracy, we evaluated 16 decision models (14 open-weight, 2 proprietary) on 233 synthetic and 10,000 real-world cases, yielding 163,728 total assessments. The best models triaged 72.1% of synthetic cases correctly and agreed with an existing algorithm in 56.1% of real-world cases. They were more promising for recognising emergencies early: given only demographics, chief complaints, risk factors and red-flag answers, the best model reached an AUROC of 0.87. At a 5% error margin, it could decide for or against emergencies for about half of users. Decision models should therefore not replace triage engines but could shorten symptom assessments at setting-specific error margins. 

---
# MedUIGym: Evaluating and Training GUI Agents for Clinical Workflows 

**Authors**: Yitong Qiao, Lei Liu, Yue Shen, Jian Wang, Jinjie Gu, Zhixuan Chu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30983)  

**Abstract**: Automating clinical workflows requires vision-language agents that can navigate dense graphical user interfaces (GUIs) in electronic health records (EHRs) and execute precise actions, yet many medical agents still rely on text or structured tool calls. In this work, we introduce MedUIGym, a scalable interactive environment comprising 15,739 tasks from 17 benchmark sources across six clinical task families for training and evaluating clinical GUI agents. In MedUIGym, agents perceive EHR states from screenshots and execute mouse-and-keyboard actions, as well as a paired native-tool interface over identical clinical data. The environment verifies both final answers and required clinical actions, and supports parallel trajectory collection for training. Benchmarking twelve vision-language models shows that current agents struggle across diverse clinical workflows, and simply increasing reasoning effort fails to yield consistent gains. Furthermore, MedUIGym supports offline learning from GUI trajectories and online reinforcement learning from task rewards. After training, both Qwen3.5 models perform better on in-domain clinical tasks and generalize to tasks from external sources. By unifying tasks from existing benchmarks, dual-interface evaluation, and GUI trajectory collection, MedUIGym provides a testbed for studying how vision-language agents complete clinical tasks through the EHR's graphical interface. Our project page is available at this https URL. 

---
# Self-Evolving Evaluation Environments for Computer-Use Agents 

**Authors**: Wenhao Wang, Panrong Tong, Quyu Kong, Pengxiang Li, Haowen Yang, chenglin Cai, Siheng Chen, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30962)  

**Abstract**: Computer-use agents (CUAs) are rapidly improving, yet their evaluation remains largely static. Existing benchmarks provide a strong foundation, but fixed difficulty ceilings, limited feature coverage, and coarse aggregate scores progressively weaken their discriminative and diagnostic value. We introduce CUA-SEE, a closed-loop evaluation environment that co-evolves tasks and executable verifiers from rollout evidence. It disentangles three objectives: benchmark evolution expands evaluation scope through cohort-calibrated difficulty and coverage growth; diagnostic evolution increases resolution through agent-conditional capability-boundary probes; and verifier evolution improves precision by translating changing requirements into state-grounded checkpoints with adversarial validation. Across multiple agents, applications, and rounds, CUA-SEE generates challenging, measurable tasks while preserving process-level evidence. Our results position adaptive evaluation as a scalable counterpart to self-evolving agents, continuously tracking what agents can do, where their boundaries lie, and how reliably progress is measured. 

---
# Aggregation Rules Induced by KL-Regularised Preference Dynamics on Networks 

**Authors**: Milan Vojnovic  

**Link**: [PDF](https://arxiv.org/pdf/2611.30941)  

**Abstract**: We study agents that each align a model to their own preference data while observing one another, and ask which aggregation rule the resulting policies implement. Each round, an agent takes one KL-regularised step. The step involves a reference that the agent is penalised for leaving and a challenger against which its responses are compared, and each of the two may be the agent's own or an average over the agents it observes; attention need not be reciprocal. Aggregating the reference makes the rule collective. Agent $v$ then has the weight $\pi_v/\beta_v$, its share of the network's long-run attention divided by its temperature. If, in addition, the challenger is fixed, the agents concentrate on the response maximising $\sum_v(\pi_v/\beta_v)r_v(y)$, where $r_v$ is agent $v$'s reward, and the rule is cardinal. If the reference is fixed and the challenger aggregated, the agents' policies at a fixed point generally differ, and any weighted mean of them is an approximate maximal lottery whatever the weights, so the dynamics select no weighting. If both are aggregated, the rule is ordinal: averaged over rounds, the collective policy is a maximal lottery of the influence-weighted preference kernel, up to a slack that we bound. Where a collective rule exists, its weights depend only on the attention weights and the temperatures. 

---
# AeroJEPA: A Two-Stage Framework for Aerosol Forecasting with Joint-Embedding Predictive Learning 

**Authors**: Zitong Shuai, Zengliang Zang, Chunlei Shi, Lang Liu, Ning Liu, Yiwen Hu, Dan Niu, Chenyan Zhou, Shangjian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2611.30891)  

**Abstract**: Aerosol forecasting requires integrating sparse ground observations and modeling meteorologically driven concentration evolution and spatial structure. However, analysis-initialized forecasts make limited explicit use of historical observations, while pointwise regression tends to smooth local variations and provides limited constraints on fine spatial structures. In this work, we propose AeroJEPA, a two-stage joint-embedding predictive framework for regional surface PM$_{2.5}$ forecasting. Specifically, Stage I fuses historical observations, model backgrounds, and meteorological information to predict an initial future state, meteorological conditions, and multi-scale structures. Morphology-guided encoding and joint-embedding predictive supervision support learning future spatial representations, supplementing concentration regression with structural constraints. Stage II uses predicted meteorology to guide state refinement and incorporates coarse- and fine-scale structures into state updates and decoding to generate the concentration field at a single future time step. Experiments show that AeroJEPA reduces gridded and independent-station RMSE by 17.23\% and 7.90\%, respectively, relative to PFGNet, while improving spatial consistency with analysis fields. In a 48 h recursive forecasting case study, AeroJEPA generally achieves lower RMSE and MAE than the numerical forecasts across the forecast period, indicating promising short- to medium-range forecasting performance. 

---
# Unpacking the Evaluators: How Configuration Shapes the Evaluation of Alignment in Explainable AI 

**Authors**: Gizem Karagoz, Tanir Ozcelebi, Nirvana Meratnia  

**Link**: [PDF](https://arxiv.org/pdf/2611.30816)  

**Abstract**: Explainable AI (XAI) evaluation fundamentally requires human-centered evaluation, assessing whether explanations align with human reasoning and expert knowledge. However, such evaluations often require expert-driven manual validation and are difficult to scale, leading to the widespread use of alignment-based metrics, where attribution maps are compared against predefined ground-truth (GT) representations. Despite their popularity, alignment evaluations are implicitly bias towards multiple design choices made during the evaluation process, for example GT representations, metric families, metric formulations (soft vs. hard), and thresholding strategies. In this paper, we formalize alignment evaluation as a structured evaluation configuration space through which the influence of these design choices can be systematically analyzed. We further introduce a structured alignment evaluation framework that models interactions between XAI methods, GT representations, metric families, metric formulations, and thresholding strategies across 2D and 3D synthetic and real-world datasets. In addition, we propose a novel taxonomy of XAI output behavior based on local entropy and boundary mass ratio, enabling structured characterization of attribution patterns for alignment evaluation. Our experiments reveal systematic flows (triples of XAI-GT-metric configurations), showing that alignment outcomes provide distinct behavior patterns under certain configurations, such as soft/hard metric implementations. We further observe varying levels of threshold sensitivity across metric families and attribution structures. These findings demonstrate that alignment evaluation cannot be treated as a fixed or invariant measure of explanation quality, and provide practical guidance for designing more reliable and behavior-aware XAI evaluation protocols. 

---
# GameDojo: Benchmarking Real-Time Interactive Agents in Video Games 

**Authors**: Kuan Zhang, Zhihao Yang, Yukun Chen, Xingyilang Yin, Mingyu Ouyang, Yue Su, Xiangnan Wu, Jiaming Li, Longze Chen, Run Luo, Zirong Chen, Lu Wang, Yiming Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.30771)  

**Abstract**: Interactive intelligence requires agents to act on observations and adapt to the consequences of actions. Studying these abilities calls for diverse, challenging tasks under controlled conditions. We introduce GameDojo, a controllable interactive gameplay benchmark of 120 boss encounters across 20 games for evaluating model capabilities, learning algorithms, and agent harnesses. Coding agents reconstruct the environments and write gameplay scripts, producing GameDojo-Data with 600 hours of gameplay trajectories. Experiments reveal substantial challenges for frontier models, especially in real-time interaction. Training on GameDojo-Data brings GameDojo-4B to frontier performance, with the strongest real-time performance. We also find joint training improves individual tasks, and learned policies transfer across bosses. Together, these findings establish GameDojo as a challenging and valuable interactive gameplay benchmark for the community. Our project page is at this https URL , and our code is at this https URL . 

---
# SchemaCaps: When a Compatible Tool Update Silently Widens an AI Agent's Authority 

**Authors**: Yulliwas Ameur, Samia Bouzefrane, Soumya Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2611.30750)  

**Abstract**: The tools users approve for their AI agents keep changing their JSON input schemas. Of 8 surveyed MCP clients and scanners, none is observed or documented to re-approve a tool whose input schema widens (two not determinable). We define non-amplification over the coordinates visible in the schema (declared arguments and undeclared fields, on effective values) and a re-binding rule decided by subschema inclusion, trusting no field labels by default and conservative up to its checker and tested case analysis. Evidence: a Coq proof of composition; 48 regression tests; an exact-reference benchmark, where the reviewed version of the rule silently re-bound 27 amplifying changes, now fixed; and a pilot on 604 releases of 39 public MCP servers. There, 412 of 580 compatible schema changes widen some coordinate, yet compatibility re-binds them silently. The authors' adjudicated double annotation finds at least 76 that add or widen an authority-bearing argument (66 of the 84 a name-based labeller flags, plus 10 it misses). On existing tools, the rule asks 645 re-approvals, against 1,197 for pinning the input schema (2,004 with descriptions) and 617 for compatibility. Behaviour change behind an identical schema remains out of reach. 

---
# Intent-Conditioned Argument Mediation for Tool-Using LLM Agents: Blocking Trusted-but-Unauthorized Arguments 

**Authors**: Yulliwas Ameur, Miqeas Pedro Mendes, Samia Bouzefrane  

**Link**: [PDF](https://arxiv.org/pdf/2611.30745)  

**Abstract**: LLM agents turn a user's request and retrieved content into tool calls, and a call can reach the wrong recipient, endpoint, file or participant. Argument-level provenance blocks such values when they come from untrusted content, but accepts a trusted value that the user never requested. IntentFlow, a mediation layer between an unmodified planner and its tools, checks each authority-bearing argument against an intent contract and a provenance policy. On a synthetic benchmark of 200 attacks (oracle intent and injection labels, declared roles), attack success falls from 50.5% with provenance alone to 12.0% [8.2-17.2%]; every remaining success alters a content or payload field. IntentFlow wrongly blocks none of 200 paired benign tasks and 4 of 50 hard ones. On 50 trusted-but-unauthorized attacks, provenance-only mediation executes all 50 and IntentFlow none; a CaMeL-style check also blocks all 50 but rejects 16 of 50 benign tasks. On an offline AgentDojo-derived replay, IntentFlow blocks all 20 attacks at 75% benign success, as does a CaMeL-style check; a Progent-style allowlist blocks 19 and keeps all benign tasks. With one live planner and the same oracles, the mediator blocked all 17 corrupted calls proposed in 110 attacks. 

---
# Transport-Safe Evaluation of LLM Agents: Trajectory-Dependent Judge Error and Anytime-Valid Auditing 

**Authors**: Yulliwas Ameur, Samia Bouzefrane, Soumya Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2611.30742)  

**Abstract**: Large language model (LLM) agents are often ranked by an LLM judge whose error rates are measured once and reused for every agent. The corrected success rates can then be biased when the judge's errors depend on the trajectory and agents differ in the trajectories they produce for successes or failures. We derive the exact asymptotic bias and give a chain of checks from identification to simultaneous ranks. On a four-agent semi-synthetic benchmark, a scalar correction with oracle calibration has maximum bias 0.245 and mean pointwise coverage 0.016, and never recovers the full ranking (0/250 replications). Augmented inverse-probability weighting (AIPW) with a uniform or a design-aware random audit keeps the maximum bias below 0.003, with mean pointwise coverage of 0.939 and 0.950. In a 48-cell stress test, effective target support rises only from 0.209 to 0.805 as calibration grows from 600 to 4,800 labels, while audited AIPW keeps a mean absolute error of 0.018-0.020. Without calibration support, or with judge dependence within an externally justified bound, we report valid bounds or sensitivity sets and carry them into rank sets. An e-process confidence sequence stays valid, for a stable target, under predictable adaptive auditing and arbitrary stopping. 

---
# Can Cities Have Personalities? Introducing the SCD City Personality Assessment 

**Authors**: Andrei Khurshudov  

**Link**: [PDF](https://arxiv.org/pdf/2611.30733)  

**Abstract**: Descriptions of cities are shaped by human experience and institutional narratives. We introduce the SCD City Personality Assessment (SCD-CPA), a method that instantiates document-grounded ChatGPT city personas and interrogates each with an identical 24-statement instrument inspired by the four-pair structure of the Myers-Briggs Type Indicator. Rather than asking a general-purpose model to characterize cities freely, we configure a separate document-grounded conversational agent for each city and use alternative evidence portraits to compare documentary inputs. The resulting four bipolar dimensions and four-letter summaries use independently defined urban concepts and scoring rules. Tokyo, New York, and Paris form the methodological evaluation (33 assessments across repeated sessions, three GPT-6 effort settings, and alternative portraits): Tokyo's original portrait yielded 85.4% exact within-setting item agreement, while evidence substitutions shifted some dimension strengths. A five-city application to Shanghai, Dubai, Mumbai, Lagos, and Sao Paulo (15 assessments) yielded 77.8% exact and 99.4% within-one-point cross-setting agreement, indicating consistency under the tested configurations. By treating each city conversational agent as a reproducible, evidence-conditioned experimental configuration, SCD-CPA makes documented city-character perceptions comparable and exposes their sensitivity to source selection. Although the documentary evidence is explicitly controlled, the model's internal interpretation of that evidence cannot be fully observed. We also examine the robustness of SCD-CPA across three AI reasoning-effort settings. The observed consistency suggests that the methodology is relatively insensitive to the tested configuration differences and provides a framework for evaluating its stability as future generations of AI models become available. 

---
# Testing Quantized KV-Cache Maintenance: Encoding Layout and Historical Restoration 

**Authors**: Ziyang Huang, Songwei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30720)  

**Abstract**: Repeating rows then selecting originals retains layout-dependent drift despite equal encoding counts. We measure when existing checks fail to establish preservation contracts in fixed quantized KV-cache implementations. In four-bit HQQ axis zero at sixteen cycles, median paired relative-L2 increases over same-shape re-encoding are 7.22 percentage points for Qwen3-1.7B and 11.59 for SmolLM3-3B. Matched encoded states cover sixteen shared questions per model at one and sixteen cycles. Axis one matches the shape-preserving control; lazy reorders preserve the initial state. On one known input, fixed update counts retain the difference; preserving group members removes it. Our prefix-view extension passes current-prefix checks yet changes history and next logits in six model/codec settings at residual capacities four and 128. Public lookup reaches this boundary, including one of four precommitted project-new GPU inputs; fixed query swaps isolate a conditional old-history effect. Checkpoint/dense rollback is exact. An explicit delayed lookup client admits a supported 129-token path excluded by native AR on one known context. Failed task-harm expansion precludes population failure-rate or general answer-degradation claims. Native MLX-LM fixtures expose stale caller caches; synchronization restores the tested in-place contract with native computations preserved. In rejection tests, stale callers change logits but not greedy tokens. 

---
# Skill-Composer: Requirement-Driven Skill Selection from Multiple Marketplaces 

**Authors**: Zicheng Chen, Jia Li, Xiaohong Chen, Zhi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.30715)  

**Abstract**: Agent skills have become the primary vehicle through which agents reuse specialized capabilities, yet public skill marketplaces provide skills with individual items: a query, whether matched through keyword or embedding similarity, returns a list of single skills ordered by relevance or by popularity. Such an interface cannot answer the question a developer actually faces: which set of skills should be installed to cover a task's complete requirement? The highest-ranked skills are often functionally redundant, while capabilities that are genuinely required remain uncovered. In this paper, we argue that skill acquisition should be reframed from ranking to set selection. We propose Skill-Composer, an agent-native framework that selects a non-redundant skill bundle for the user requirement, which aims to cover all available required capabilities and report any gaps. Specifically, it clarifies user requirements into atomic requirement points, retrieves and merges candidate skills across fragmented marketplaces, decomposes each candidate's capabilities, and progressively discards overlapping ones, so that each retained skill provides different capabilities. This moves the decision of which skills to acquire from the developer to the agent itself. We position this reframing and sketch its workflow. 

---
# How Many Parameters Does Forecasting Need? A Survey and Code-Level Audit of Lightweight Long-Term Time Series Forecasting 

**Authors**: Issam Ait Yahia, Abdelkader El Mahdaouy, Ismail Berrada  

**Link**: [PDF](https://arxiv.org/pdf/2611.30685)  

**Abstract**: Long-term time series forecasting supports decisions in energy, transportation, weather and finance, where forecasts are produced continuously and often on constrained hardware. Recent work has moved in two directions. One scales Transformers and pre-trained foundation models to millions or hundreds of millions of parameters. The other shows that models with only hundreds to tens of thousands of parameters remain competitive on the standard benchmarks. Existing surveys organize forecasting models by backbone and treat efficiency as a side topic, so the reasons these small models work, and the way their sizes are reported, remain unexamined. This survey focuses on lightweight long-term forecasting. It provides a taxonomy of about thirty-five models in six families, with a common notation and an architecture diagram for each core model. It offers a structured-linear view in which the smallest models are linear maps restricted by frequency truncation or period folding, and it tests each restriction in closed form on eight benchmarks. It complements the literature review with an audit of sixteen official code repositories, which reproduces published parameter counts and reveals reporting gaps of up to 289 times. Finally, it gives practical guidance on model selection and reporting, and summarizes open problems. 

---
# Uncovering Hidden Behaviours in Language Models via Limited-Parameter Finetuning 

**Authors**: Boyd Kane, Jo J. Jiao, Elizabeth Donoway, Bryce Woodworth, Alex Cloud, Alexander M. Turner  

**Link**: [PDF](https://arxiv.org/pdf/2611.30672)  

**Abstract**: Large language models can have behaviours which only arise in hard-to-anticipate contexts. These hidden behaviours pose a challenge to alignment audits, which seek to uncover problematic model behaviour but are limited by the scenarios the auditor can anticipate. To address this challenge, we explore limited-parameter finetuning (LPF) on examples of a target behaviour. While training 3-25% of LoRA parameters, LPF reveals whether the model has a hidden behaviour, even when standard behavioural evaluations cannot. LPF succeeds on a suite of model organisms with hidden behaviours (8-14B parameters), and also surfaces naturally occurring hidden behaviours in open-weight models (27-35B parameters). LPF can thus distinguish between models which appear behaviourally identical to an auditor, and does not require the context in which the hidden behaviour surfaces. LPF provides a novel affordance that may be useful for auditing frontier AI models. However, we do not evaluate LPF against baselines in a realistic auditing setting, so further work is needed to determine its practicality. 

---
# Beyond the Black Box: A Diagnosis-Driven Agent for Interpretable MIP Solver Tuning 

**Authors**: Yidong Luo, Weihuang Wen, Xuanhao Pan, Yinya Wang, Shanwen Pu, Chenghao Lan, Dongdong Ge, Tianshu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30670)  

**Abstract**: Tuning a mixed-integer programming (MIP) solver is an instance-specific problem, where a poor configuration can increase solving time by orders of magnitude.
Existing Bayesian optimization tuners treat solver tuning as a black-box optimization problem: each evaluation returns a single score, allowing the tuner to learn which configuration is faster, but not why.
Inspired by the use of profilers in GPU-kernel optimization, we introduce APEC (Agent for Per-instance Explainable Configuration), a log-grounded tuning agent for MIP solvers.
APEC has three layers: a deterministic layer that profiles each log, an interpretation layer that guides hypothesis-driven diagnosis, and a system layer that controls evaluation, enabling more reliable and interpretable solver tuning.
On the full MIPLIB 2017 benchmark set, APEC achieves a 1.58X improvement over the default configuration with over 38% fewer parameter changes than the baselines, making the results easier to read. 

---
# WeMM-Ecom: Advancing Multimodal Content Understanding for WeChat E-Commerce 

**Authors**: Binhui Xie, Jinmin Li, Haomiao Tang, Guangting Wang, Song Jin, Jianwei Zhu, Fengyun Rao, Jing Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30662)  

**Abstract**: We present WeMM-Ecom, a family of multimodal large language models that deliver frontier e-commerce capabilities while achieving state-of-the-art general-purpose performance at their respective scales. The family spans five sizes, from a 0.8B dense model to a 35B mixture-of-experts model with 3B active parameters. Our training pipeline begins with two-stage continued pre-training that progressively raises the proportion of e-commerce data within a diverse multimodal data mixture, followed by hybrid-thinking post-training that combines on-policy distillation and reinforcement learning to support both non-thinking and thinking modes. For evaluation, we construct a comprehensive suite of 28 e-commerce benchmarks (23 in-house and five open-source) covering eight business scenarios and complement it with 64 public general-purpose benchmarks. On the e-commerce suite, WeMM-Ecom-35B-A3B surpasses substantially larger open-source flagship models, including Qwen3.8-Flash-Next, HY4, and DeepSeek-V4.1-Flash. On the general-purpose suite, every WeMM-Ecom model surpasses its similarly sized Qwen3.5 counterpart, with the lightweight variants (e.g., WeMM-Ecom-2B) showing clear advantages over leading lightweight models such as Qwen3.5-2B, MiniCPM-V-4.7-1B, and LFM3-VL-2B. Beyond benchmarks, WeMM-Ecom now powers hundreds of production use cases in WeChat, including catalog management, multimodal product search, and livestream commerce, serving billion-scale requests per day. We publicly release all model weights and detail our data curation, training recipe, and benchmark construction to facilitate future work on domain-specialized multimodal models. We have released the model weights and code to facilitate future research at this https URL. 

---
# ActField: Learning Action Effects as Continuous-Time Latent Dynamics 

**Authors**: Yunpeng Qing, Yuheng Huang, Changxu Wei, Yilun Kong, Yichen Shi, Shunyu Liu, Changqing Zou  

**Link**: [PDF](https://arxiv.org/pdf/2611.30659)  

**Abstract**: Learning how actions change visual states is essential for prediction and decision-making in embodied tasks. Existing action-conditioned JEPA world models typically predict future representations at discrete observation intervals, leaving the temporal evolution of action effects implicit. We introduce ActField, a continuous-time JEPA world model that explicitly models latent state evolution through an action-conditioned vector field integrated over physical time. ActField separates action-independent evolution from action-induced changes, with the latter modulated by the latent context. A reference-conditioned representation and multi-step dynamics loss further support accurate long-horizon prediction. Together, these components provide a structured and temporally continuous formulation for predicting action-conditioned visual dynamics. Experiments across LIBERO, DROID, and BEHAVIOR demonstrate improved visual prediction fidelity and more accurate action-induced motion prediction over existing world models. Further evaluations on simulated and real-world manipulation tasks demonstrate the utility of ActField in improving downstream policy planning through VLM-guided action selection. 

---
# Learning Multimodal Representations from Partially Paired, Small-Scale Data 

**Authors**: Merel Kuijs, Till Richter, Rushin H. Gindra, Korbinian Traeuble, Christian Matek, Rebeka Luknárová, Tingying Peng, Fabian J. Theis  

**Link**: [PDF](https://arxiv.org/pdf/2611.30601)  

**Abstract**: Modeling multimodal data from partially paired samples is critical for advancing domains like biomedicine, where vast unimodal datasets and foundation models exist but paired data remains scarce. Existing fusion methods rely on large-scale paired datasets, limiting their use in scenarios with incomplete pairing. We introduce CAMEO, an adversarial learning-based modality fusion framework that integrates modalities from small, partially paired datasets. Combining any pre-trained unimodal encoder with a cross-modal latent alignment mechanism, CAMEO learns shared representations requiring only minimal paired samples. Evaluated on computational pathology tasks such as niche classification and cell type composition prediction, CAMEO achieves superior data efficiency, outperforming contrastive approaches like CLIP in low paired-data regimes, and highlighting the benefits of adversarial alignment when paired annotations are scarce. To facilitate further research, we additionally release a fully annotated HuggingFace dataset comprising three organs and paired image and gene expression modalities. By extending fusion methods to address limited pairing and small-scale datasets, we provide a framework that advances multimodal learning and broadens its applicability to real-world biomedical problems. 

---
# JaxNaviSim: A GPU-Accelerated, Data-Driven Simulator and Learning Benchmark for Maritime Navigation 

**Authors**: Quang Anh Pham, Janaka Chathuranga Brahmanage, Hue Tse Leong, Akshat Kumar, Tien Anh Mai, Siu Loon Hoe, Venky Shankararaman  

**Link**: [PDF](https://arxiv.org/pdf/2611.30600)  

**Abstract**: Coordinating vessels through the shared waterways of busy ports grows more challenging as vessel traffic increases. Port authorities need realistic simulations to assess how arrival schedules, anchorage assignments, and routes affect vessel throughput and navigation safety. Existing AIS-based trajectory models predict a vessel's track from its past trajectory in an open-loop fashion. They do not simulate step-by-step responses to other ships or test whether a vessel collides or reaches its goal. To address this problem, we introduce JaxNaviSim, a data-driven simulator and learning benchmark coupling Automatic Identification System (AIS) data with fast, controllable traffic simulation, in which learned models steer vessels through recorded traffic. First, we release a dataset of ~141 million AIS records from over 30K vessels across eight high-density regions of the Singapore Strait, among the world's busiest waterways, with ~6.7 million vessel-hours. Unlike several existing AIS datasets which mostly cover open seas, ours captures close-quarters vessel traffic in the narrow strait. Second, we define a navigation task for each recorded vessel trajectory. A model steers that vessel every 10 s from its recorded state to its anchoring point or region exit, while surrounding ships follow their recorded tracks or are steered by the model. We implement a JAX simulator that runs these tasks in large batches on a GPU. We also incorporate multiple vessel dynamics models and sea surface currents. Finally, we benchmark imitation learning (IL), offline and online RL, and include Predictive Action Refinement (PAR) as a safety baseline that cuts collision rates to 1-3%. The comparison shows that low tracking error does not ensure arrival, and learned policies can reach most destinations yet collide in 20-45% of episodes. 

---
# Interrogation Protocols for AI Control 

**Authors**: Aniruddh Pramod, Matthew Khoriaty, James Lucassen, Adam Kaufman  

**Link**: [PDF](https://arxiv.org/pdf/2611.30583)  

**Abstract**: As the intelligence of an AI agent increases relative to a trusted monitor, so does the risk that the monitor will fail to understand the agent's actions or to evaluate them confidently. We introduce interrogation protocols to a high-stakes AI control setting, and allow the monitor to ask the untrusted agent questions about confusing actions. For interrogation to improve safety, a weaker model must be able to distinguish misleading explanations produced by an attack policy from genuine ones produced by an honest policy. We find that on a held-out subset of BashArena, an interrogation protocol improves safety at a 1% audit budget from 72.6% for an actions-only baseline, which does not use interrogations, to 79.5% (+7.0 pp [+0.9, +20.1], 95% paired bootstrap interval), but most of this uplift can be captured by having a trusted model state a specific concern. More broadly, our work suggests that interrogating agents can improve safety, allowing monitors to catch more attacks at a larger capability gap. 

---
# Learning What to Change: State-Verified Reinforcement Learning for Educational Agents 

**Authors**: Shengyao Wang, Siyuan Su, Siyu Song, Mengsong Wu, Yiming Luo, Wenbo Wu, Hao Hao, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.30559)  

**Abstract**: Persistent tutoring agents do more than answer questions: they manage course files, practice records, and learner profiles that govern future lessons. In such stateful settings, knowing what to teach does not guarantee knowing what to change. We identify a pervasive failure mode, content-state decoupling, where agents deliver pedagogically sound responses while silently corrupting workspace state through unwarranted updates, missed prerequisites, or unintended side effects. Text-level judges overlook these state errors, whereas isolated tool-call checks break down because whether a write action is valid depends jointly on the learner request, prior state, and execution order. We propose State-Verified Reinforcement Learning, which supervises online policy optimization directly through verifiable workspace transitions. Within isolated sandboxes that snapshot state around every tool call, a deterministic verifier audits tool effects, workflow ordering, and native auto-sync side effects, pairing graded partial credit with a hard gate that zeroes episode reward upon any critical violation. Whenever no critical violation occurs, an evidence-grounded judge scores actual saved artifacts rather than agent claims. We pair 1,000 verifiable training instances with an independently authored, human-reviewed evaluation benchmark of 60 tasks spanning 11 educational categories and 16 native tools. After online GRPO training, Qwen3.8-27B raises category-weighted strict success on our benchmark from 51.47% to 80.74% (+29.27 points), cutting content-pass-only execution failures from 45.6% to 13.7% and required state-check failures from 63 to 1 across 180 trials. The resulting policy outperforms seven reference baselines by over 20 points while reducing output tokens by 17-39% across seven of eight external benchmarks without regressing on general reasoning or tool use. 

---
# Splitting Belief Bases Conditionally based on Constraints for c-Representations and their Solutions 

**Authors**: Lars-Phillip Spiegel, Christoph Beierle, Gabriele Kern-Isberner, Alexander Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2611.30556)  

**Abstract**: Splitting notions for conditional belief bases aim at localizing reasoning and computation tasks to subbases containing only the relevant parts of the belief base. Among the different reasoning mechanisms, c-representations, a special kind of Spohn's ranking models, have been shown to provide high-quality inferences. c-Representations can be conveniently specified via a constraint satisfaction problem. In this article, we introduce the notion of conditional constraint splittings, extending constraint splittings by also allowing the subbases to share conditionals. Taking inspiration from conditional syntax splittings, we identify the subclasses of safe and fully localized conditional constraint splittings, allowing us to locally and independently solve the constraint system on the subbases. We also generalize solution splittings for c-representations to safe conditional c-solution splittings and provide an in-depth analysis of their relationship to conditional constraint splittings. Additionally, we elaborate on the relationship of these new splitting notions to safe and generalized safe conditional syntax splittings and also to safe covers, which have been proposed for localized computation of c-representations. 

---
# Reproducible by Seed? Not on GPUs: Batch-Dependent Non-Determinism in Text-to-Image Models 

**Authors**: Javier Conde, Xiuyuan Qi, Shanshan Liu, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2611.30545)  

**Abstract**: In text to image AI models, fixing the prompt and the random seed is widely assumed to fix the image. We show that it does not on GPUs. We evaluate six text-to-image models (diffusion and autoregressive) under four numerical precisions and quantizations on three GPU types (RTX A6000, RTX 6000 Ada, and H100 NVL), across up to 100 prompts per set for all six models. For each configuration we compare, with the prompt and seed fixed, the image generated alone with the image generated inside a batch, and we also compare across GPUs. Every replay control reported here is bit-exact, so the variation is not random noise: the execution setting, mainly the batch composition and the GPU, decides the output. The effect is large and model dependent. Structural similarity between the single-image and the batched output falls to 0.580 for an autoregressive model in bf16 and to 0.551 with int8. For a second autoregressive model in bf16 it falls to 0.261 with sampled decoding and to 0.229 with greedy decoding, and for an fp8 diffusion model to 0.68. A diffusion model in bf16 stays close to the batch-1 output (0.96-0.98) but changes every image. In most cases the batched image is a different but equally valid image; in some, such as prompts with digits and counters, it is visibly worse. Cross-GPU differences are of the same order as the batch effect: for one autoregressive model in bf16 compared across two GPU architectures (RTX 6000 Ada versus RTX A6000), no image is bit-identical and the cross-GPU LPIPS (0.355 to 0.383) matches the batch effect within one GPU (0.347 to 0.381). The consequences are direct: seeds do not make results reproducible, benchmarks and image-quality comparisons that mix batch sizes or hardware can be biased, and safety audits and A/B tests on generated images need the serving setup reported. 

---
# Revisiting LLMs as Context Engineers 

**Authors**: Junrui Zhang, Zeyu Xu, Runchuan Zhu, Jie Peng, Long Zhang, Zixin Ding, Jiaxin Lin, Kai Wang, Jianmin Ji, Junyuan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2611.30494)  

**Abstract**: As large language models (LLMs) are used in more and more new tasks, context engineering becomes a fundamental method to efficiently steer LLMs by constructing instruction prompts or metadata without intensive training. A growing number of LLM context engineering (LCE) approaches automate the task by casting the LLM itself as the engineer that draws generalizable instructions or experiences from offline trials as context. Although these methods resemble how humans learn from experiences, it remains unclear \textbf{when and how} LLMs can engineer effective context from data. To answer the question, we first establish a benchmark composed of datasets with different reasoning patterns. We compared different LCE methods with different LLM families in 8B size, showing that there is no single winner for all tasks, even compared with simple baselines like chain-of-thought. Even when scaling model capabilities via more parameters or generation iterations, LCEs do not consistently boost existing performance. Across all datasets and models, the performance gains of LCEs are negatively correlated ($r=-0.5, p\le 0.05$) with the diversity of the failure modes of reasoning trajectories. Investigating how LCEs work, we find that (1) the improvement in instruction adherence can explain some success of LCEs; (2) one major bottleneck is whether LLMs can extract effective experiences out of natively diverse failures. Through this study, we uncover the limited capabilities of LCEs in abstracting principled contexts from diverse failure modes and understand how LCEs work, providing guidelines for future work. 

---
# Automated Call for Action: RAG-Based Environmental Law Enforcement Screening System 

**Authors**: Matas Jurkevicius, Yen-Chia Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30423)  

**Abstract**: Environmental law inspectors face a structural issue in identifying potential violations from data: they often do not have the capacity to deal with large volume of data present in the emission events. This paper examines to what extent an LLM-based system can support evidence assembly and permit-based screening of such events, using the coke plants of a steel producer and a regional environmental service as a case. A simple puff-trajectory model aligns air quality measurements, weather and citizen complaints with each emission event. A retrieval-augmented generation (RAG) pipeline then retrieves facility-specific permit conditions and generates event reports and a period report with an investigation recommendation. Across 74 events from March 2025, plant-aware re-ranking reduced reports citing the other plant's conditions from 58.1% (hybrid retrieval) to 0% by rule-based plant labels, while maintaining high RAGAS and LLM-judge metrics. GPT-5-mini performed comparably to GPT-5.1, whereas GPT-5-nano scored lower on legal grounding and legal faithfulness. In an evidence ablation, removing air quality (AQ) data changed judge scores and period recommendations most; citation overlap changed only slightly under any removal, and removing complaint details had no measurable effect. This suggests a need for rules and guidelines that link complaints to incidents and permit conditions, so that community reports can count as evidence. The results support screening of events that may need further investigation. Design implications for involving environmental service experts are derived. 

---
# HWRE-Bench: Can LLM Agents Understand Hardware They Did Not Design? 

**Authors**: Peilong Zhou, Cangyuan Li, Kaiyan Chang, Ying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30422)  

**Abstract**: Hardware benchmarks for LLM agents all run the design flow forward, from a specification to RTL, a layout or a bug fix. We ask the reverse question: given an implemented artifact, such as a layout, a gate-level netlist or a protected simulation model, can an agent work out what it does well enough to act on that knowledge? We introduce HWRE-Bench, a benchmark for agentic hardware reverse engineering built on one observation. Each step of the forward flow discards information (names, hierarchy, word-level structure, generator parameters, the programming model), and a reverse-engineering task asks the agent to recover what was lost and to prove it with an executable deliverable that the artifact alone cannot supply. Since layout extraction is now mechanical and a correct netlist is already a cycle-exact model, every task carries a mandatory copy-the-netlist control that must score zero. HWRE-Bench spans five tracks (Extract, Abstract, Audit, Interoperate, Modify), four information-loss levels and white- versus black-box access, all on original designs. We report a pilot of 148 rollouts from two agents on four seed tasks under 16 conditions, with stage-annotated trajectories. Agents built their own layout extractors and audited a sanitized layout as reliably as a netlist, at about 27% more turns. Keyless, equivalence-preserving obfuscation changed neither pass rates nor defect-repair scores. Whether agents read the structure at all depended on task framing rather than on necessity, so pass rate alone does not show understanding. Provider safety filters stopped 16% of one agent's runs on the audit tasks. These results argue for deliverables beyond equivalence and for reporting run validity explicitly. 

---
# TamperCert: A Benchmark for Detecting Text Tampering in Degree Certificates 

**Authors**: ChiFung Kwan, Kangzhong Wang, Youqian Zhang, Weijie Chen, Jiaqi Gao, Yongjian Wu, Eugene Yujun Fu, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30419)  

**Abstract**: Degree certificates play a critical role in high-stakes decisions such as university admission, immigration, and hiring. The issuing institution vouches for the holder, so a forged certificate may let its bearer pass these checks. Degree certificate forgery is increasingly prevalent. Yet, many assume that it is easy to spot. Reported cases show otherwise: in 2024, the business school of the University of Hong Kong found that about 30 master's students had enrolled with forged academic credentials. Few studies have looked at how to detect forged degree certificates, partly because public benchmarks are scarce. We present TamperCert, a benchmark built from 400 real degree certificates issued by 38 universities. Each certificate was subjected to 11 publicly available tampering methods to alter the holder's name and further processed with four additional post-processing techniques, resulting in over 20,000 forged images. We evaluate 13 detectors, a commercial multimodal large language model (MLLM), and detectors trained on this http URL findings reveal that no detector is reliable for every tampering method. Consequently, a determined forger may still evade detection. Moreover, the commercial MLLM misses most forgeries made by general-purpose generative models, and post-processing, such as resizing an image or photographing it with a camera, severely degrades detection. 

---
# DMC-5: Dual-Model Collaboration with Cache-aware Context Compression for Long-horizon Agents 

**Authors**: Haochen Li, Xiaobing Tu, Yinggui Wang, Jinkui Ren, Xiantao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30412)  

**Abstract**: Long-horizon agents repeatedly feed an ever-growing interaction trajectory to a specialized agent model. Prefix caching reuses computation over repeated history, but cannot prevent context growth or remove information irrelevant to the current decision. This is especially costly when the specialized agent lacks the cached-input discounts available to general-purpose models, making trajectory compression a natural alternative. However, existing compression methods on selective context risk losing global information, recursively rewriting summary leads to semantic drift, while repeatedly recompressing the full history shifts the growing cost to the compressor. Therefore, we introduce DMC-5 (Dual-Model Collaboration with CaChe-aware Context Compression), an asymmetric framework that assigns full-history integration to a general compression model and action generation to a specialized agent model. DMC-5 maintains the raw trajectory as an Append-only Canonical View, preserving global access while allowing the compressor to reuse a stable historical prefix instead of recursively rewriting or recomputing prior context. The agent operates only on a compact Decision View with recent raw interactions, reducing repeated input while retaining the information needed for the next action. DMC-5 further incorporates YAMATO (integritY-Aware Memory Abstractions for Tool Operations) to preserve execution-critical information during lossy compression. Experiments on AppWorld, OfficeBench, and 8-objective QA show that DMC-5 reduces agent model input by 38.5-57.0% and end-to-end cost by 14.5-28.0% with 78.0-85.0% compressor cache-hit rates, while improving task performance by 1.0-5.8% over the uncompressed baseline. 

---
# Cost-Sensitive Reversibility of Non-Deterministic Actions 

**Authors**: Giuseppe Canonaco, Alberto Pozanco, Daniel Borrajo  

**Link**: [PDF](https://arxiv.org/pdf/2611.30399)  

**Abstract**: Determining action reversibility requires checking whether the effects of an action can be undone. Recent work has extended reversibility to fully observable non-deterministic (FOND) planning. It distinguishes weak reversibility, where reversal may succeed, from strong-cyclic reversibility, where reversal is guaranteed under fairness. These notions are qualitative: they say whether reversal is possible, but not how costly it is.
We introduce cost-sensitive reversibility for non-deterministic actions. Given positive action costs, we define weak and strong reversal budgets, $k$-bounded reversibility, and a distinction between finite strong reversal and strong-cyclic reversal with unbounded worst-case cost. We relate the resulting notions in a reversibility lattice, prove cost-aware policy-merging results, and show how to compute restricted weak and strong universal uniform reversal budgets using cost-optimal classical planning and worst-case-optimal FOND search, respectively. We also give an admissible lower-bound test for rejecting $k$-bounded reversibility. These notions support finer-grained safety decisions: a small finite strong budget bounds recovery cost, whereas a small weak budget certifies only a cheap possible recovery path. Actions without an acceptable strong bound may therefore warrant human approval, as in tool-using agent systems. 

---
# Event-Driven Parameter Dynamics for Spatiotemporal Parameter-Update Application in Deep Learning 

**Authors**: Ruixue Li, Chaofei Hong  

**Link**: [PDF](https://arxiv.org/pdf/2611.30397)  

**Abstract**: Standard neural network training typically applies optimizer-generated update signals to model parameters at every iteration. This work introduces a spatiotemporal view of parameter updating and proposes Event-Driven Parameter Dynamics (EPD), which separates candidate-update generation from their application through an intermediate state with accumulation, event-triggered release, and reset. In a canonical coordinate-wise instantiation, accumulation produces temporal integration, release imposes a spatial budget, and reset separates successive inter-release intervals, while conventional dense updating is recovered as a boundary case. Across four model--optimizer configurations on CIFAR-100, EPD maintains final performance close to conventional training even when only a small fraction of coordinates are released at each step, including the 1\% regime. Matched-control experiments further show that temporal accumulation, state-dependent selection, and post-release reset each contribute to the observed behavior. Together, these results identify parameter-update realization as a controllable dimension of optimization and provide a concrete basis for studying its temporal and spatial organization separately from candidate-update generation. 

---
# Trustworthy Speech-Based Autism Screening under Class Imbalance with Uncertainty-Aware HuBERT Ensembles 

**Authors**: Nihel Loubna Ghennani, Mourtada Benazzouz, Julie Mauclair  

**Link**: [PDF](https://arxiv.org/pdf/2611.30390)  

**Abstract**: Early screening of Autism Spectrum Disorder (ASD) is critical for timely clinical intervention, although establishing reliable speech based screening systems remains problematic due to limited clinical data, class imbalance, and the need for reliable uncertainty estimation. We propose an uncertainty-aware ASD screening framework based on self-supervised HuBERT representations, combining Leave-One-Subject-Out (LOSO) crossvalidation, balanced k-bagging, and Monte Carlo (MC) Dropout. Multiple HuBERT models are fine-tuned on balanced subsets, with predictions combined through probability-based ensemble averaging and aggregated at the subject level. Experiments on the Child Pathological Speech Database (CPSD) show strong performance, achieving 86.30% balanced accuracy and 93.74% ROC-AUC at the utterance level, and 95.04% and 96.30%, respectively, at the subject level. Furthermore, predictive entropy effectively reflects classification reliability: incorrectly classified segments show higher entropy than correctly classified ones (0.4605 $\pm$ 0.2314 vs. 0.1717 $\pm$ 0.1830), with a significant correlation with errors (r = 0.3981, p = 7.38 x 10 -170 ) and an error-detection ROC-AUC of 0.8359. These results demonstrate the potential of the proposed framework as a robust and uncertainty-aware computer-aided tool for ASD screening. 

---
# mini-psych-agent: Evaluating Human-Like Behavior in LLM Agents through Interactive Psychological Environments 

**Authors**: Yi-Fei Cao, Tianxin Huang, Hanbo Xie, Jian-Qiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30365)  

**Abstract**: A Large Language Model (LLM), its harness, and the test-time compute budget jointly determine its behavior. Consequently, the same underlying model, embedded in different agentic workflows or allocated different amounts of test-time compute, can exhibit markedly different performance. Psychologists face an analogous challenge when evaluating human performance: the same person may behave differently under different incentives and levels of deliberation. Psychological tasks therefore provide an ideal testbed for studying such behavioral variation. However, when adapted to LLMs, these tasks are typically text-transcribed, replaying fixed, language-rendered experiments as static prompts. This format is ill-suited to agent evaluation, where actions can alter future observations and task states. Conversely, although dynamic environments are prevalent in the evaluation of LLM agents, they rarely ask whether the resulting behavior is human-like. Here, we bridge this gap by transforming psychological tasks into dynamic environments in which agents can interact and by using a learned human-policy model to evaluate how compatible the resulting behavioral trajectories are with human behavior. Across these environments, we find that this compatibility varies with an LLM's agentic configuration and the psychological task, and that memory raises it mainly by making choices more predictable rather than more human-like. These findings motivate a more holistic evaluation of LLM agents in psychology-inspired environments, accounting for both the backbone model and its agentic scaffold. 

---
# PseuDRAGON: Knowledge-Guided Local LLMs for Financial-Data Pseudonymization in Closed Networks 

**Authors**: Gwangui Hong, Seonho An, Seonkyu Lim, KyungTae Lim, Min-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.30353)  

**Abstract**: Staff pseudonymizing financial data in closed networks must interpret abbreviated column names, identify personal information, and choose processing methods. Local language models can assist, but may give different labels for the same input. PseuDRAGON keeps confirmed judgments and corrections outside a fixed model and reuses them through exact-name lookup and learned rules. For expert review, the workflow searches institutional policy, national privacy law, then international sources. Across 92 columns from one institution, rules increase reuse under the same approval checks from 75 to 83 labels without reuse errors when all training judgments are available. With 25% of the training judgments, withholding non-PII rules for unseen names eliminated the extra PII misses in replay at the cost of lower reuse. Revising or withdrawing approvals rebuilds derived rules, and tests on synthetic records verify that selected methods produce the specified outputs. In three paired runs, the history-first baseline cut automated identification time by 83.0% on average versus no reuse. 

---
# MH-RAG-DIVE: Dynamic and Cost-Efficient Multi-Hop RAG Evaluation via Tree-Cached Dialogue Simulation 

**Authors**: Lorenz Brehme, Daniel Hillebrand, Thomas Ströhle, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30335)  

**Abstract**: Quality assessment is essential for determining whether changes to a Retrieval-Augmented Generation system improve or maintain its performance. However, existing evaluation approaches often rely on single-turn or single-hop interactions and therefore do not capture the complexity of multi-hop, multi-turn conversations. We introduce MH-RAG-DIVE, a dynamic interactive user simulation framework designed to evaluate RAG systems in more complex multi-hop and multi-turn settings. The framework extends RAG-DIVE with a Context Discoverer, which identifies documents that can be combined to generate multi-hop questions within an ongoing conversation. To address the high computational cost of dynamic user simulation, we further introduce a Conversation Cacher, which reuses turns from previous evaluation runs while preserving the dynamic nature of the evaluation. We evaluate the framework across six distinct RAG configurations and compare its results with the original RAG-DIVE evaluation and the static, human-created CLAPNQ dataset. Our experiments show consistent performance trends across the evaluation approaches, while the more complex multi-hop and multi-turn setting exposes additional challenges for RAG systems. Furthermore, the Conversation Cacher substantially reduces token consumption, demonstrating its potential to make dynamic evaluation more cost-efficient. Finally, we apply MH-RAG-DIVE to an industrial RAG system with domain-specific data, demonstrating its applicability to real-world evaluation scenarios. Overall, our results show that the proposed approach provides a more complex and cost-efficient framework for evaluating RAG systems. All code and experimental data are publicly available in our GitHub repository (this https URL). 

---
# MMMem: Self-Improving Agents with Task-Aware Multimodal Experience Consolidation 

**Authors**: Yuxuan Chen, Changwei Lv, Yunfei Li, Zhongjing Du, Kehui Tan, Zijun Liao, Ruochong Zheng, Hao Chen, Can Deng, Zheni Zeng, Chang Liu, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30287)  

**Abstract**: AI agents accumulate conversations, images, and documents through long-term interactions with users. Answering later questions often requires connecting evidence across these records, yet the resulting understanding is rarely retained, allowing mistakes to recur on related questions. Existing multimodal memory methods largely focus on storing and retrieving records, with limited mechanisms for learning from completed tasks and consolidating task-specific experience. We therefore propose MMMem, a task-aware multimodal memory framework that extracts and consolidates experience from completed tasks through four complementary strategies. First, source-linked multi-view memory organizes original text, images, and rendered pages alongside generated scene images that combine clues across records for retrieval. Second, source-generated tasks support knowledge alignment at two levels: warm-up consolidates source knowledge into task-relevant experiences in external memory, while self-training adapts the model's parametric knowledge to task demands. Third, the same vision-language model distills completed tasks into experiences that record the task context, conclusion, and support links. Finally, task-aware retrieval recalls source views and relevant experiences for answering and verification, while source-supported reuse guides long-term retention. Across three memory benchmarks and two vision-language readers, MMMem outperforms eight baselines. With Qwen3.5-9B, MMMem achieves relative gains of 36.9%, 17.6%, and 15.2% over the best-performing baselines on LoCoMo, Mem-Gallery, and LooGLE v2, respectively. Ablations show consistent gains from training and memory warm-up, while case studies illustrate how individual components work in practice. 

---
# A statistical measure of trust in generative AI 

**Authors**: Alina Ermilova, Valeria Yakupova, Nikita Sushko, Andrey Lovyagin, George Ovchinnikov, Maksim G. Sharaev, Evgeny M. Mirkes, Denis Turdakov, Ivan Y. Tyukin  

**Link**: [PDF](https://arxiv.org/pdf/2611.30286)  

**Abstract**: Trustworthiness is a crucial issue for generative AI, as modern models can produce convincing and confident responses even when their outputs are incomplete, unsupported, or wrong. The frequent absence of ground-truth knowledge or labels complicates verification of generative AI outputs. To address this issue, we propose a trustworthiness framework grounded in model-internal confidence calibrated against an external expert reference. We estimate per-token trustworthiness of a student generative model as the statistical typicality of each generated token relative to an expert model's distribution, where the expert is used as a distributional reference rather than as a source of factual verification labels. To this end, we introduce an AI ontology calibrator that leverages the student model's internal latent features and token-level output statistics to compute expert-reference acceptance scores. Combined with traditional token-level confidence, entropy, and perplexity statistics, calibrator scores yield a compact set of $15$ response-level indicators for correctness classification. Evaluated across BF program generation, mathematical problem solving, and medical question answering, combined calibrator and traditional indicators provide complementary reliability signal which could be used to build detectors of untrustworthy outputs and hallucinations. Remarkably, our experiments show that these detectors are transferable across domains, although transferability is not always symmetric. 

---
# Learning Heterogeneous Reasoning in Large Language Models: From Mode Compatibility to Local Bottlenecks 

**Authors**: Sunbowen Li, Dandan Song, Shengkun Ma, Xuanning Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.30281)  

**Abstract**: Post-training large language models routinely pools reasoning trajectories from diverse teachers or algorithms, yet it remains unclear whether standard supervised fine-tuning (SFT) can learn such heterogeneous strategies without explicit mode conditioning. We study the learnability and local bottlenecks of heterogeneous reasoning through a formal mixture formulation and controlled experiments. We first formalize the representational compatibility of the optimal autoregressive model with latent reasoning mixtures: the target next-token distribution weights modes by their prefix posteriors and concentrates on a single mode once alternative modes assign negligible likelihood to the prefix. In a verifiable symbolic graph-search environment with paired breadth-first and depth-first search trajectories, no consistent disadvantage is observed for unlabeled mixed SFT relative to homogeneous or labeled baselines under matched token budgets while preserving both modes. Scaling across 0.6B--14B models and data budgets largely eliminates global mode confusion, shifting the primary bottleneck to local state-tracking transitions. Because a local mistake affects the final outcome only when subsequent reasoning is sensitive to that step, we find that only a small fraction of local errors are outcome-determining \emph{Critical Reasoning Nodes (CRNs)}, and that single-step repairs at these nodes rescue failing trajectories across symbolic search and natural language benchmarks whenever the model has sufficient downstream competence to complete the remaining derivation. Our environment is open-sourced at this https URL. 

---
# CLIP-FMoE: Scalable CLIP via Fused Mixture-of-Experts with Enforced Specialization 

**Authors**: Luong Tran, Lan-Cuong Nguyen, Huynh Dang Nguyen, Dat Nguyen Cong, Dung D. Le, Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30233)  

**Abstract**: Mixture-of-Experts (MoE) architectures have emerged as a promising approach for scaling deep learning models while maintaining computational efficiency. However, existing MoE adaptations for Contrastive Language-Image Pre-training (CLIP) models suffer from significant computational overhead during sequential training and degradation of zero-shot capabilities. To address these limitations, we propose CLIP-FMoE, a novel approach that integrates MoE architecture into CLIP fine-tuning. Our method uses Isolated Constrained Contrastive Learning, a pipeline that trains specialized experts on cluster-based data partitions to accelerate expert specialization. Additionally, we introduce a Fusion Gate mechanism to mitigate catastrophic forgetting of pre-trained knowledge. Extensive experiments across multiple benchmarks demonstrate that our approach achieves consistent improvements on downstream tasks while preserving zero-shot capabilities. Furthermore, our method demonstrates robust performance across varying context lengths, making it particularly suitable for diverse real-world applications. 

---
# VeriWorld: A Verifiable Visual SWE-Bench for Spatial Reasoning in 3D Environments 

**Authors**: Yan Zheng, ShengYun Peng, Xuyao Liang, Xiaoyan Cong, Nuo Chen, Bangya Liu, Zihan Wang, Wenyan Cong, Zhiwen Fan, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30226)  

**Abstract**: Vision-language models (VLMs) are increasingly used for spatial tasks such as navigation and manipulation, where they must infer geometry and object relationships from visual input in order to act correctly. Existing benchmarks usually test perception and reasoning in isolation, making it hard to tell whether failures come from visual understanding, downstream reasoning, or the bridge between them. We introduce VeriWorld, a benchmark for spatial reasoning in interactive 3D environments with executable actions and deterministic verification. For task families where all three channels are well-defined, we evaluate matched visual, structured, and combined input conditions on the same instance, environment, and verifier; families without a clean visual/structured/combined split are reported separately. This design supports controlled comparisons of perception, reasoning, and their interaction. Using this setup, we uncover two diagnostic findings. First, models often perform much better when task-relevant information, such as object identity, current state, or target location, is given explicitly than when it must be read from the rendered scene, revealing a recurring visual-to-structure gap. Second, action-space and harness design can change outcomes even under the same task and information condition, showing that interaction protocol is itself a confounding variable. VeriWorld is open-sourced to support reproducible and fine-grained evaluation of spatial reasoning in VLMs. 

---
# Adopting AI Technologies without Traps: A Multi-Agent Reinforcement Learning Framework 

**Authors**: Jitong Yang, Avinash Reddy Mudireddy, Ziqing Lu, Huan Cai, Weiyu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30200)  

**Abstract**: AI technologies and AI agents play increasingly important roles in human society, including in education, economics, and research. Adopting AI technologies can bring immediate, and sometimes tremendous, rewards such as through productivity boost, but they can also bring negative effects including but not limited to losses of job and training opportunities, and cognitive capability declines. Thus, it is important to investigate the optimal ways of adopting AI technologies without different traps. In this paper, we develop a general multi-agent Markov decision process (MDP) based reinforcement learning (RL or MARL) modeling framework for determining and analyzing optimal adoption of AI technologies. This framework can model the long-term dynamic effects of AI adoption, and lead to finding the optimal AI adoption strategies. In our modeling framework, the negative effects of AI adoption are modeled by undesirable state transitions or negative rewards in the MDPs. The proposed multi-agent RL MDPs incorporate both human agents and AI agents. The key idea of our models is that incorporating AI technologies may not only increase the immediate rewards for AI agents but also alter the states of the involved human and AI agents. In particular, we apply this framework to analyzing AI offloading leading to skill erosion, to analyzing optimal human resource structures for a firm adopting AI, and to analyzing the economic equilibria in the trading/employment among human and AI agents. Since the multi-agent MDP based RL technique is from control, machine learning and AI, this research is essentially using an AI technique to determine the optimal ways of adopting or co-existing with AI without traps. 

---
# Same Executions, Different Gaps: Judgment Rules in Web-Agent Evaluation 

**Authors**: Zelong Wang, Weiyue Li, Fei Xue, Junru Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30190)  

**Abstract**: Success-rate gains guide the development of web agents. Yet knowing which agent ranks first does not tell us how much confidence to place in its reported advantage. We study the comparison itself: how success is defined, what evidence supports the judgments, and how the difference between agents is estimated. On fixed web-agent executions, changing a judge from predicting likely completion to requiring recorded support can remove most of a reported lead without changing the ranking. This effect persists when unresolved completion is distinguished from supported failure. Controlled source additions explain how missing relations between task requirements, objects and observations can conceal a difference or create an apparent advantage. We also find that making both agents' score estimates more accurate can make their difference less accurate, because their original errors partly canceled. To estimate a chosen outcome difference, we combine judge predictions with reference labels for sampled task pairs. Across four released execution systems, regression fitted on the same 50 labeled pairs reduces estimation error relative to direct paired sampling. The results give agent researchers a practical basis for assessing claimed improvements: make the comparison target explicit, retain the evidence needed to check it, and use paired labels to estimate the difference directly. 

---
# ReVAgent: Multi-Agent Reasoning-Intensive Video Generation 

**Authors**: Yehui Yang, Jing Fan, Jun Hu, Shangyao Chen, Jinhua Bian, Song Wang, Ruize Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.30163)  

**Abstract**: A central challenge in generating instructional videos for reasoning-intensive topics is to show how and why the state changes at each step. Even when an animation renders successfully, the depicted object relationships and state transitions can conflict with the explanation, making the reasoning difficult to follow. We present ReVAgent, a multi-agent framework that solves a problem, generates keyframes for its explanation, and synthesizes an instructional video. The agents share the execution record of a solution program, reuse intermediate results and constructions in the scenes, and check rendered keyframes to guide revisions to the solution or the scenes. We also introduce a multidimensional rubric for assessing whether generated videos present the problem clearly, organize the steps coherently, and use visuals to explain the reasoning. We calibrate a multimodal judge using human ratings and apply it to instructional videos for mathematics and algorithms. In the evaluated samples, ReVAgent presents multi-step reasoning coherently and uses diagrams to show how key operations lead to subsequent states. 

---
# AutoSECO: Automated Skill Evolution for Cross-Problem Combinatorial Optimization with General-Purpose LLM Agents 

**Authors**: Zirui Zhu, Cong Zhang, Ni Zhang, Kuan Xu, Zhiguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.30139)  

**Abstract**: General-purpose large language model (LLM) agents can develop complete solvers for combinatorial optimization problems. Their specialization, however, often relies on workflows engineered for individual problems, limiting the reuse of development expertise. We propose AutoSECO, an automated skill evolution method that learns reusable solver-development guidance from experience. The challenge is that external evaluation measures the completed solver, while the skill guides decisions during development. AutoSECO connects this feedback to guidance deficiencies by tracing performance shortfalls through solver code and recorded development decisions. Revisions are expressed as general instructions and retained only if a fresh solver developed under the revised skill achieves a lower mean evaluation gap. Evolved solely on the traveling salesperson problem (TSP), the skill guides fresh agents in developing solvers for capacitated vehicle routing, job-shop scheduling, and three-dimensional bin packing without further skill evolution. On TSP, the final mean benchmark gap is 0.52\%, compared with 0.98\% for the skill-evolution baseline SkillOpt. Across all four problems, the evolved skill achieves better aggregate results than no skill guidance, the initial skill, and SkillOpt. The same frozen skill also supports solver development under alternative model--harness configurations. These findings suggest that solver-development experience can be retained as reusable expertise for general-purpose LLM agents. 

---
# WASP: Continual Weight Adaptation for Streaming LLM Personalization 

**Authors**: Linxi Xie, Chonghua Liao, Mengxuan Wu, Guandao Yang, Zhangyang Wang, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30134)  

**Abstract**: Personalized large language models (LLMs) should continually adapt to users' evolving preferences. Yet extracting preference changes from noisy interaction histories while supporting inexpensive updates remains challenging. We introduce WASP (Weight Adaptation for Streaming Personalization), a recurrent framework that captures fine-grained preferences directly from raw user interactions and generates personalized LoRA adapters without expensive per-user optimization. Specifically, WASP achieves this by training a shared hypernetwork that maintains a compact user representation and outputs adapter weights in a single forward pass. As new interactions arrive, WASP updates this representation without reprocessing full interaction history, enabling continual adaptation with per-update costs independent of history length. Experiments on personalization benchmarks and chronological interaction streams show that WASP sustains high-quality personalization over long horizons under fixed compute and storage budgets. Together, these results support efficient streaming personalization and facilitate privacy-preserving on-device deployment. 

---
# iFeST: An Iterative Motion Correction Framework for Fetal Brain Diffusion MRI via Frequency-Spatial-Temporal Modeling 

**Authors**: Haoan Xu, Ruike Chen, Jing Xia, Haotian Li, Huanyun Zhang, Cong Sun, Guangbin Wang, Shuhao Zhang, Zhigang Gao, Dan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30131)  

**Abstract**: Fetal brain diffusion magnetic resonance imaging (dMRI) is a pivotal non-invasive tool for assessing the development of brain connectivity and microstructure. However, the prolonged acquisition times and complex sampling requirements render dMRI highly susceptible to unpredictable fetal motion. Existing motion correction methods typically process each dMRI volume independently and focus on inter-slice motion. Such strategies fail to utilize the coherent information between dMRI volumes along different diffusion directions and lack the ability to restore the intra-slice corruption. To address these limitations, we proposed iFeST, an iterative framework that combines frequency (k-space), spatial, and temporal (inter-slice) modeling for fetal brain dMRI motion correction. At each iteration, iFeST integrates information across diffusion directions into fiber orientation distributions (FODs). These distributions encode local fiber orientations within each voxel and serve as microstructural priors for intra-slice restoration and inter-slice alignment in the next iteration. To train the framework, we developed a fetal brain dMRI motion simulator that synthesizes compound motion artifacts from FODs. Evaluation on simulated datasets demonstrated that iFeST outperformed four inter-slice networks in motion estimation and three intra-slice networks in image restoration. Further validation on a real-world dataset confirmed significantly better volume reconstruction and FOD estimation than conventional pipelines. Overall, iFeST provides a learning-based framework for fetal dMRI motion correction, restoring motion-corrupted data and improving microstructural reconstruction for downstream diagnostic applications. 

---
# When Does Optical Context Compression Pay Off? A Decode-Aware Cost Analysis of Reasoning VLMs 

**Authors**: Ziqing Ge  

**Link**: [PDF](https://arxiv.org/pdf/2611.30130)  

**Abstract**: Rendering long text as images lets a vision-language model (VLM) read a document with fewer input tokens, and evaluations often count mainly these prefill tokens. We show that for reasoning models this accounting is incomplete. Across five models and two long-context benchmarks, reading the rendered document makes the model think longer: decode length grows by 1.5-3.5x on the four general-purpose VLMs tested, and under typical output pricing this decode tax makes image input more expensive than text for two of the five models on QuALITY. Unrecoverable information loss is not sufficient to explain the tax at moderate compression: Qwen's OCR of the same images recovers 98.7% of words, and feeding Kimi-K3 its own transcription as text restores text-mode reasoning length and accuracy. Whether compression pays off reduces to an exchange rate, (1-1/c)/(r-1), that must exceed the workload's price-weighted ratio of output to input tokens; this accounting decomposition shows why Kimi-K3 and DeepSeek-V4.1-Flash can benefit and Qwen and GLM cannot. None of the training-free signals we test predicts the tax before generation. As a practical heuristic, an image-first policy that sends only inputs with a per-input compression ratio of at least 1.6 to images and falls back to text once image-mode reasoning costs as much as the input's prefill saving cuts the bill by 6-14% on Kimi-K3 and DeepSeek-V4.1-Flash at an accuracy cost of 0.3-1.2 points; on QuALITY, asking for a reasoning budget on both paths is a stronger lever. 

---
# Could This BE Any More Wrong? Teaching VLMs to Know What They Don't Know in Document Extraction 

**Authors**: Varad Srivastava, Victor Casas Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2611.30119)  

**Abstract**: Vision-Language Models can extract structured fields from scanned financial documents, yet they provide no indication of when a field might be wrong. We derive per-field confidence estimates from the token log-probabilities the model already produces, via a token-to-span alignment procedure that maps logprobs onto each extracted value within the structured JSON output. Evaluated on two benchmarks (CORD v2 receipts, Kleister NDA legal contracts) with three VLM families, the minimum token probability per field achieves Area Under the ROC Curve (AUROC) 0.74-0.91 for error detection, far exceeding verbalized confidence (0.53-0.56). The uncertain fields form a natural training curriculum: fine-tuning via Group Relative Policy Optimisation (GRPO) improves accuracy from 78.5% to 87.5% on CORD (67 examples) and from 70.7% to 75.1% on multi-page NDA contracts (259 examples), while substantially improving error detection on Kleister NDA (AUROC 0.70 to 0.85, Area Under the Accuracy-Rejection Curve (AUARC) 0.78 to 0.92). The loop requires no annotation beyond the training-split ground truth, demonstrating that confidence-guided curriculum selection transfers to complex multi-page documents when combined with native multi-image training. A frontier model (Claude Sonnet 4) used as a pseudo-label judge achieves ~92% reward agreement with ground truth on both benchmarks, suggesting this loop can operate without human annotation. 

---
# Small data machine learning: the trilemma of low-sample tabular regimes 

**Authors**: Ilya Emelyanov, Dmitrii Dobrydin, Muhammad Hamza, Malak Abid Ali Khan  

**Link**: [PDF](https://arxiv.org/pdf/2611.30118)  

**Abstract**: Small tabular datasets leave few observations for choosing a model and assessing its predictions. We examine these competing demands in controlled simulations and on six public datasets, using SmallGBM and SmallMLP as fixed-configuration case studies. Three results illustrate why the choice of evaluation matters. First, at n = 100, five-fold cross-validation overstated the gain from tuning XGBoost by 0.063 ROC-AUC on random labels, while tuning improved independent-test ROC-AUC by 0.0078 on a Gaussian mixture. Second, when 20% of binary training labels were flipped, the median within-trial ROC-AUC loss was 0.0210 for SmallGBM and 0.0778 for a fixed 200-tree XGBoost configuration. A separate regression experiment showed the sensitivity of squared-error training to heavy-tailed responses. Third, in a heteroscedastic regression experiment, locally weighted SmallMLP intervals covered about one percentage point more test responses than ordinary split conformal intervals at both tested dimensions. Their mean widths also increased. On a Gaussian classification task with linear log odds, logistic regression achieved the highest mean ROC-AUC at all seven training sizes. These experiments identify settings in which a small-data model helps and quantify the cost of its assumptions. 

---
# Post-Training Agents for Long-Document Reasoning 

**Authors**: Sushant Mehta, Logan Ritchie, Edwin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30090)  

**Abstract**: Professional documents are a long-context workload that retrieval-style benchmarks do not capture. In a ten-domain enterprise collection, a task's evidence is spread across a source document of median length 27 pages (mean 48.8, maximum 873); a third of tasks rest mainly on charts, drawings, and maps; and 70.9% require two or more reasoning hops. We post-train Kimi-K2.7, an open-weight foundation model, at its full context window on 1,465 expert-authored tasks with atomic rubrics from that collection, training rank-32 adapters with one round of privileged-context on-policy self-distillation followed by two rounds of rubric-reward reinforcement learning. We render each PDF as text for training and as extracted text plus selected page images at evaluation. The smoothed RL training reward rises from 0.37 in the first tenth of the run to 0.67 in the last. On the 100-task this http URL holdout, served in a 262,144-token window, strict pass (all rubric criteria met) more than doubles after the second round, from 11.0% to 24.5%, with gains across input lengths up to 240K tokens. Mean tokens generated stay flat (9,803 to 9,733), so the gain does not come from longer outputs. GPT-5.6 Sol's leaderboard strict pass is 30.7% at a measured 3,422 tokens. On 220 out-of-domain GDPval tasks, where the model runs as a long-horizon tool-using agent, mean rubric reward rises from 0.571 to 0.746 after the third round and zero-credit tasks fall from 54 to 10, while messages per task fall 18%, from 70.9 to 58.2. In paired trajectories, the third round replaces a context-thrashing retry loop with one up-front check of all sources, and verification extends from checking that files exist to checking what they contain. We argue that long-context evaluation of agents should report accuracy jointly with tokens and turns, and that datasets of long documents should ship text and image renderings alongside each PDF. 

---
# Correlated by Construction: Endpoint Choice and the Validation of Protein Screening 

**Authors**: Eric LoPrete, Galen Loving, George Sabin, Pavle Jeremić  

**Link**: [PDF](https://arxiv.org/pdf/2611.30087)  

**Abstract**: Machine learning for protein engineering depends on functional measurements collected at screening scale, but the designs those measurements rank are deployed in processes that run elsewhere. Whether a miniaturized assay remains informative under the conditions of use is usually assumed rather than demonstrated. We describe an automated production and screening platform and test the transfer of its solution-phase metal-affinity measurements to immobilized rare-earth separation. Primary pKd-derived separation factors tracked those measured on covalently immobilized protein across 82 protein-metal-pair measurements from 35 proteins, with pairwise Spearman correlations of 0.878 to 0.923 (pooled Pearson $r$ = 0.836; Spearman $\rho$ = 0.869). Separation factors at a fixed metal pair are insensitive to the ordering shared across the lanthanide series, so these coefficients report protein-specific correspondence directly; permutation of protein identity confirms it, giving a mean scrambled correlation of 0.003 and, for the 17 proteins measured on all three pairs, $r$ = 0.917 against 0.001 ($p \approx 0.001$). Profiles compared as profiles behave differently. An independent comparison against published LAMBS profiles for the same protein backbones (417 proteins, 5,212 paired measurements) reproduced normalized raw elution at $r$ = 0.870, but reassigning protein identity still reached 0.840. High profile-level agreement across assay formats is largely attributable to metal preferences shared across the series rather than to protein identity. Transfer can be demonstrated, but the endpoint decides whether a correlation shows it: a pairwise contrast removes the shared ordering; a normalized profile retains it, and needs a null that reassigns protein identity. 

---
# Context-Grounded Skill Internalization for Vision-Language Agents 

**Authors**: Jinlin Liu, Yumeng He, Qihan Huang, Kevin Qinghong Lin, Han Peng  

**Link**: [PDF](https://arxiv.org/pdf/2611.30075)  

**Abstract**: Skill internalization offers a way for agents to learn useful behaviors from skills and use them without explicit guidance in context at inference time. A common approach is to generate demonstration trajectories with the skill in context and then fine-tune the same model on its own trajectories with the skill removed. However, this leads to the problem that the model may learn to reference the removed skill, which we named as context misalignment (CoMi). In this work, we first analyze agent behaviors related to CoMi under this common pipeline. Across ALFWorld, WebShop, and ARC-AGI-1, skill-removed fine-tuning consistently results in CoMi: 12.2-46.0% of test trajectories reference the removed skill. To address this problem, we develop Context-Grounded Skill Internalization (CGSI). CGSI uses a sentence detector to identify unsupported references and then masks or rewrites the selected sentences before fine-tuning. Experimental results demonstrate the effectiveness of CGSI and reveal two key findings. (i) Compared with directly fine-tuning with the skill removed, CGSI reduces CoMi at test time by 72.5-94.6% while preserving task success. (ii) Compared with existing reinforcement learning-based skill-internalization baselines, CGSI achieves comparable performance with substantially less training time. CGSI also significantly reduces processing costs compared with large-scale trajectory auditing and correction. 

---
# EVD-Bench: A Comprehensive Benchmark for EEG-to-Image Decoding 

**Authors**: Kieren Yu, Qian Zhang, Ziyang Liu, Chang Huang, Jintai Chen, Kaishun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30061)  

**Abstract**: EEG-to-image decoding recovers visual information from non-invasive brain recordings, providing a window into human perception. However, comparisons across systems are confounded by a variable that is rarely reported as such: which participants' data the system was allowed to use. The same Top-1 score on a 200-way gallery may come from a per-participant system, a pooled model over all participants, or a model that never saw the test participant---cases that differ by more than most architectural changes between the methods being compared. We present \bench, a benchmark on THINGS-EEG2 that standardizes the gallery and EEG input, matches training budgets across systems within each regime, and compares three participant-access regimes: within-subject, joint-subject, and strict leave-one-subject-out, with unlabeled target calibration reported separately. All three settings evaluate concept-level zero-shot retrieval: the 200 test concepts are disjoint from the 1,654 concepts used for EEG--image fitting and validation. We compare retrieval across access regimes and use zero-EEG and matched-wrong EEG controls to assess dependence on correct EEG--stimulus pairing. We vary trial averaging, gallery size and difficulty, and training data coverage, and compare target adaptation and visual feature targets. We also assess sensitivity to EEG window length and channel count, robustness to input perturbations, and confidence calibration. These results establish participant-access regime as a necessary reporting dimension alongside gallery, repetition, and visual target. 

---
# SWAM: State-aware World Action Model for Multi-Agent 3D Environments 

**Authors**: Jenna Kang, Bingxuan Li, Colin Groth, Anatolii Evdokimov, Sai Zhang, Qi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.30042)  

**Abstract**: Interactive world models generate virtual environments autoregressively from player actions and visual history. However, visual history alone cannot reliably capture the evolving states of independently acting non-player characters (NPCs) in real games, particularly when they are outside the camera view. Recent multi-agent world models address this challenge through joint multi-view generation or persistent shared state, but often rely on external observations, like maps or scene geometry, or predefined state transitions. We present SWAM, a State-aware World Action Model that separates agent-state evolution from autoregressive visual generation in interactive 3D environments. A lightweight state model propagates character dynamics from actions without consuming generated frames, combining motion priors with learned scene-dependent corrections. The predicted states then guide an autoregressive diffusion transformer alongside visual history through a novel conditioning scheme. Rather than requiring an explicit scene geometry map or visibility tests, the conditioning indicates where characters may appear, allowing the diffusion model to learn their appearance and occlusion. This design enables NPCs to evolve independently of visual generation while maintaining consistent observations across visibility changes, interactions, and respawns. Evaluations in a custom Godot-based first-person shooter game demonstrate improvements in visual fidelity, NPC consistency, and long-horizon stability over existing approaches. 

---
# Automated Neural Network Auditing (ANNA) for Energy Market Settlements: A Machine and Reinforcement Learning-Based Approach 

**Authors**: J. R. Rivadeneyra, I. Lee, F. Nugen, S. M. Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2611.30020)  

**Abstract**: The integration of segmented energy markets into Independent System Operators (ISOs) like CAISO and MISO has increased settlement complexity. Automated transaction processing creates challenges for evaluating accuracy, compliance, and financial integrity. Traditional rule-based manual auditing is inefficient, error-prone, and cannot handle the data volume from automated this http URL research proposes Automated Neural Network Auditing (ANNA), a machine and reinforcement learning framework for settlement auditing via anomaly prioritization. ANNA integrates statistical anomaly detection, K-Nearest Neighbors (KNN), clustering, and Deep Q-Network (DQN) reinforcement learning to flag potentially anomalous transactions from historical market data. The approach is evaluated as an analyst-support tool for prioritizing transactions needing further audit this http URL was tested against traditional statistical methods, clustering-based techniques, and supervised/reinforcement learning models. The study used three years of CAISO settlement data over 63 million source records across 32 charge codes and multiple counterparties. After preprocessing, approximately 2.9 million labeled transactions formed the working dataset, with model-specific subsets used for training and evaluation. Performance was assessed using precision, recall, computational efficiency, and false-negative this http URL outperforms traditional methods by identifying hidden relationships in settlement data, while standard deviation thresholds, K-Means, KNN, and Q-Learning performed poorly across key metrics. The result is a scalable solution integrating machine and reinforcement learning for automated auditing of energy market settlements, contributing computationally efficient anomaly prioritization to settlement-auditing workflows. 

---
# OWE-Bench: A long-horizon benchmark for measuring real-world engineering 

**Authors**: Jacob Cole, Martin Dimakis, Victor Barres, Bertie Vidgen, Edward Hu, Jason Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.30017)  

**Abstract**: We present OWE-Bench (OpenWorldEngineer-Bench), a benchmark of 26 long-horizon engineering tasks written by domain experts in software engineering, physical engineering, machine learning, and scientific computing. Tasks range from optimizing kernels and designing an aircraft, to identifying the faults responsible for an earthquake. Each task requires significant domain knowledge and engineering work, with models averaging hundreds of millions of tokens per task. Solutions are scored on practical measures such as speed, cost and accuracy, both on provided cases the agent could test against and on hidden cases that an engineer would expect from the task description. The best-performing frontier model so far, Claude Fable 5.2, scores 54.6% and averages 211 million tokens per task. We highlight two findings. First, when models are given more time, they use more test-time compute and achieve higher scores. Models also scale differently with test-time compute: GPT-6 Astra achieves higher scores at shorter time limits and at equal token use, while Claude Fable 5.2 uses more tokens at longer limits to achieve the highest overall score. Second, each model scores lower on hidden cases than on the provided cases, by an average of 18 points, which we call the deployment gap. Models also fail to predict the deployment gap, overestimating their performance on hidden test cases by an average of 21 points. We hope that OWE-Bench grounds progress toward agents whose engineering solutions perform as well in deployment as they do in testing. 

---
# Categorical Foundations of AI Agents 

**Authors**: Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29998)  

**Abstract**: We follow the idea of separating intelligence from execution and give the categorical foundations of AI agents: the transformers, the agents, and the orchestrations and choreographies of AI agents are categorized, and which also include their coalgebras and algebras. 

---
# Differential Proteomics to XAI Rules: A SHAP-FOL-ILP Modeling for Lung Adenocarcinoma 

**Authors**: Surajit Das  

**Link**: [PDF](https://arxiv.org/pdf/2611.29989)  

**Abstract**: Background: We developed an interpretable workflow linking paired differential proteomics, sparse logistic regression, SHAP attribution, and first-order logic (FOL) rule learning in lung adenocarcinoma (LUAD).
Methods: We analysed 213 CPTAC-LUAD tissue samples (111 tumour, 102 adjacent-normal) from 111 patients, including 102 matched pairs. Protein effect estimation and testing used the same complete pairs, with Benjamini-Hochberg correction. Sparse logistic regression used five-fold stratified patient-grouped nested cross-validation. Preprocessing and model selection were confined to training data, with feature selection repeated within inner folds. A separate five-fold patient-grouped evaluation tested SHAP-guided inductive logic programming (FOL/ILP) using training-derived evidence and discretisation.
Results: Quality control retained 9,518 proteins. At FDR below 0.05 and an absolute mean paired log-ratio difference of at least 1, 378 were differentially abundant (40 increased, 338 decreased). The sparse classifier yielded ROC-AUC 0.9997, average precision 0.9998, accuracy 0.9906, and balanced accuracy 0.9906. FOL/ILP yielded held-out accuracy 0.9765, balanced accuracy 0.9767, precision 0.9818, recall 0.9730, and F1 0.9774. HSPA12B, PALM2-AKAP2, STARD13, and PPM1F had non-zero coefficients in all five classifier folds. The most frequent Horn clause occurred in two of five folds. All others occurred once.
Conclusions: The workflow connects proteomic classification, attribution, and symbolic rules. Performance reflects tumour-versus-adjacent-normal discrimination within one cohort. Biomarker and clinical claims require independent validation, repeated assessment of rule stability, and further biological annotation. 

---
# Aligning Judge Models to Be Logically Consistent Multi-Rule Interpreters 

**Authors**: Michel Liao, Nimra Nadeem, Luxi He, Peter Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2611.29975)  

**Abstract**: Technical governance of AI models will soon require them to reason over and follow thousands of rules from different jurisdictions. However, as we show in this work, presenting an entire ruleset to a judge model---rather than asking it to judge one rule at a time---changes the way the model interprets the requirements, making it reach conclusions that conflict with its rule-by-rule judgments. We call this challenge multi-rule inconsistency and find that models are more lenient when judging rules embedded within the full ruleset than when judging rules separately. For example, we find that when a safety-critical rule prohibiting a model from pursuing objectives other than those it was given appears in a larger ruleset, all tested judge models are less likely to deem it violated. We examine several methods for reducing inconsistency, including prompting, consistency training using GRPO, and iterative rule refinement. As in human statutory interpretation, surrounding rules shape how a rule is interpreted. Unlike the legal setting, however, these interactions are currently happening implicitly inside the judge model's reasoning. We argue that multi-rule interactions must be measured and handled explicitly by ruleset designers to prevent undetected shifts in interpretation from causing downstream alignment failures. 

---
# CARE-3D: Reinforcement Learning for Evidence Grounded Agentic Reasoning in Situated 3D Scenes 

**Authors**: Xiaoqian Hu, Ruihong Qiu, Yanran Tang, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29969)  

**Abstract**: Situated 3D reasoning requires embodied agents to interpret objects, scene geometry, and viewpoint dependent spatial relations. Existing methods learn grounded interactions from supervised traces, but largely imitate prescribed evidence acquisition patterns rather than optimizing the support that the returned evidence provides for the correct answer. Moreover, outcome rewards provide limited credit for intermediate tool choices and may leave infrequent tools underexplored. An effective agent should instead adaptively decide what to ground, which tool to invoke, and how to assess the utility of returned evidence. To this end, this paper presents CARE-3D, a reinforcement post-training framework for adaptive grounded reasoning. Supervised fine-tuning (SFT) on grounding instructions first establishes valid interactions, after which Group Relative Policy Optimization (GRPO) optimizes the agent while keeping the 3D grounding module frozen. CARE-3D combines Counterfactual Attribution of Returned Evidence for evidence sensitive credit assignment, and Group Coverage Sampling to preserve tool coverage. Experiments on MSQA and Beacon3D show consistent gains over SFT and GRPO using only answer and format rewards, especially in spatial reasoning and grounding answer coherence. Our code is available at \href{this https URL}{CARE-3D}. 

---
# When Learned IDS Model Selection Collapses to a Static Choice: A Measurement Study at the Edge 

**Authors**: Nuonan Ouyang, Bonan Zhang, Adrian Shatte, Zhigang Lu, Chao Chen, Wei Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29968)  

**Abstract**: Runtime model selection trades detection quality against energy and latency. However, comparing a selector only with a more costly static detector conflates detector choice with the benefit of adaptation. We present a framework for analysis and physical measurement that combines a pairwise reward criterion, matched static controls, paired blocks, a fixed load-prevalence factorial design, and reporting of actions outside training support. On one Raspberry Pi 4B with four frozen TON-IoT detectors, tabular Q-learning and a deep Q-network (DQN) both select LightLR throughout. A retrospective audit using validation data only rules out preference reversals for MedRF versus LightLR and HeavyMLP versus LightLR. However, TinyDT has a higher reward than LightLR in seven of the 2,500 windows, so the audit does not establish global LightLR optimality. We also evaluate a causal DeadlineGuard rule under a fixed factorial design with three arrival rates (50, 100, and 4,000 rows/s) and three attack prevalences (0.10, 0.50, and 0.90). Compared with Static-MedRF, DeadlineGuard saves 43.7 J per 500-s run (2.98%; 95% paired CI 41.4--46.0 J) and eliminates all 512 deadline misses among 5,000 windows, at a cost of 1.373 percentage points in F1. Outside training support, Tabular-Q's deterministic tie-breaking rule selects TinyDT, resulting in a false-positive rate of 67.76%. These single-board results motivate joint audits of reward attainability, realized actions, attribution against matched static controls, service feasibility, and default actions outside training support. 

---
# SAEPO: Sparse-Autoencoder-Guided Credit Assignment for Policy Optimization 

**Authors**: Yusong Zhao, Zihao Zhao, Zirui Zhang, Tunyu Zhang, Ziyan Wang, Yi Wang, Chengzhi Mao, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29936)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated strong performance in improving large language model (LLM) reasoning, yet its credit-assignment signal remains coarse. Outcome verifiers provide scalable supervision by judging whether a completed response is correct, but they do not directly indicate which intermediate reasoning tokens contribute to success or failure. As a result, policy optimization can be inefficient, especially when training base models directly from sparse outcome feedback. In this paper, we introduce SAEPO, a Sparse AutoEncoder-guided token-level credit assignment method for outcome-supervised Policy Optimization. Rather than treating correctness only as a rollout-level (sequence-level) signal, SAEPO uses a pretrained sparse autoencoder (SAE) to build token-level semantic representations within reasoning trajectories; it scores each token based on the SAE concepts (e.g., penalizing wrong "final-answer" tokens but rewarding correct "reasoning-process" tokens), learned in an unsupervised manner, to achieve fine-grained credit assignment. SAEPO is a plug-in advantage-shaping module compatible with various policy optimization algorithms. We instantiate SAEPO with both GRPO and GSPO, covering token-ratio and sequence-ratio policy-gradient objectives, and evaluate SAEPO across three LLM backbones on six math and two code benchmarks. Empirical results show that SAEPO improves both the math and code benchmark averages over the matched baselines, with gains of up to 6.63% and 5.57%, respectively. On Qwen3-1.7B-Base, training curves also show faster early accuracy growth. Feature and token-level analyses suggest that the SAE-derived signal favors question-relevant concepts and reasoning tokens. 

---
# Reward Blindness: A Theoretical Framework for Diagnosing Reinforcement Learning Reward Limitations in Text-to-SQL 

**Authors**: Lakshmi Muraleedharan Nair, Shiqi Huang, Fangyu Lo, Yihua Xu, Dingyan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29930)  

**Abstract**: Text-to-SQL systems let users without programming expertise query relational databases in natural language. Reinforcement learning has become the dominant way to train such systems, yet progress is still judged by a single number, execution accuracy, which counts correct queries but says nothing about which kinds of semantic error survive training. This paper develops a theoretical framework that examines text-to-SQL from the perspective of reward design. We introduce Reward Error Sensitivity, a measure of how strongly a reward function separates queries that contain a given error type from queries that do not, and prove a Reward Blindness theorem: when this sensitivity is zero, policy-gradient training carries no systematic signal for removing that error. Using fourteen published finegrained error signals for four text-to-SQL systems on the BIRD benchmark, we build error profiles and estimate sensitivities for common reward designs. Predicate and result-level errors are the most frequent, firing on up to 27 percent of queries, while structurally deep errors such as suboptimal joins persist at 5 to 12 percent. Each reward family has a distinct and predictable set of blind spots, and inexpensive database-level checks that need no language model calls detect 45 to 60 percent of semantic errors. Controlled training simulations confirm that executiononly reward leaves output-preserving errors untouched, whereas a signal-grounded reward removes them even when the detectors are noisy. 

---
# From Patterns to Threats: A Deep Learning Model for Anomaly Detection in Cyber-Physical Systems 

**Authors**: Kayode L. Ogunsusi  

**Link**: [PDF](https://arxiv.org/pdf/2611.29903)  

**Abstract**: Cyber-physical systems increasingly rely on interconnected sensing, communication, and control components, creating a need for reliable detection of abnormal behavior. This study proposes a Deep Autoencoder for anomaly detection using the Edge-IIoTset dataset. After preprocessing the full dataset, 2,219,201 observations and 38 numerical features were retained. The proposed model employed a 38-24-12-6-12-24-38 encoder-decoder architecture and was trained exclusively on normal observations to learn representative patterns of expected system behavior. Anomaly decisions were based on reconstruction error, with the classification threshold selected independently from validation data by maximizing the F1-score. To assess model stability, five independent runs were conducted using different random seeds, and results were reported as mean plus or minus standard deviation. The proposed Deep Autoencoder achieved an accuracy of 0.9195 plus or minus 0.0061, precision of 0.9602 plus or minus 0.0204, recall of 0.7348 plus or minus 0.0095, F1-score of 0.8325 plus or minus 0.0121, ROC-AUC of 0.8825 plus or minus 0.0339, and PR-AUC of 0.8640 plus or minus 0.0286. Compared with a Simple Autoencoder baseline, the proposed model improved recall and F1-score while reducing the false positive rate. The results demonstrate that the proposed reconstruction-based representation learning approach can provide an effective and reproducible method for detecting anomalous behavior in cyber-physical environments. 

---
# Second-Order Dynamics in Noisy RNNs Produce Exploration and Temporal Compression in Replay 

**Authors**: Josue Casco-Rodriguez, Nanda H. Krishna, Richard G. Baraniuk  

**Link**: [PDF](https://arxiv.org/pdf/2611.29888)  

**Abstract**: Biological neural networks (like the hippocampus) can internally generate "replay" resembling stimulus-driven activity. Recent computational models of replay use noisy recurrent neural networks (RNNs) trained to path-integrate. Replay in these networks is described as Langevin sampling, but this theory alone fails to capture the environmental exploration and temporal compression observed in real biological replay. To close this gap, we re-examine spontaneous noisy RNN activity ("replay") as Langevin sampling, one timestep at a time, from a time-series distribution resembling lived experiences (unlike previous works which defined sampling as Monte Carlo inference over a static distribution). We provide a normative second-order Langevin sampling framework wherein spike-frequency adaptation drives spatial exploration at the cost of speed, and momentum via dendritic input differentiation drives temporal compression. We numerically verify our framework via simulations of noisy RNNs trained on 2D triangular and T-maze navigation activity, and others trained on place cell activity of synthetic rats. Crucially, we observe that simultaneous adaptation and momentum increases both replay exploration and speed, and provide one of the first accounts for biologically plausible temporal compression in rate-based RNN replay. Our findings encourage experiments into whether dendritic input differentiation, or other mechanisms, may support temporal compression in memory circuits like the hippocampus. 

---
# Particle Belief Representations for Reinforcement Learning under Partial Observability 

**Authors**: Himanshu Gupta, Brendan Crowe, Christoffer Heckman, Bradley Hayes, Zachary Sunberg  

**Link**: [PDF](https://arxiv.org/pdf/2611.29884)  

**Abstract**: Particle filters flexibly represent complex, multimodal beliefs but expose them as weighted, unordered sets incompatible with conventional fixed-dimensional policies. We study particle-belief encodings as inputs to reinforcement-learning policies while holding belief estimation and policy optimization fixed. We compare statistical summaries, cumulant-generating-function (CGF) features, and permutation-invariant set encoders. Trainable encoders are learned end-to-end or pretrained through belief reconstruction or task-specific prediction. Across eight tasks, pooling-based set encoders and gradient CGF features are reliable end-to-end, while the Set Transformer achieves the highest performance with appropriate pretraining. Task-specific pretraining helps when its target captures decision-relevant belief structure, whereas reconstruction pretraining and latent alignment provide more selective benefits. For selecting and training particle-belief encoders in partially observable Markov decision processes, our results show that an encoding that reconstructs the belief well is not necessarily a good input for control. 

---
# Weight Averaging at Small Budgets Confounds Formula-Labeled Pretraining in Microstructure-Property Regression: A Pre-Specified Study with Post Hoc Rechecks 

**Authors**: Jonghyun Lee, Soojung Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29876)  

**Abstract**: Closed-form physics formulas can label synthetic microstructures at no cost, so supervised pretraining on those labels is an alternative to self-supervised pretraining when labels are scarce. One pre-specified protocol tests this on two public simulated benchmarks, two-phase conductivity and short-fiber stiffness: a ConvNeXt-Small is pretrained on 150,000 synthetic images by formula regression or masked reconstruction and fine-tuned on 100, 400, or 1,600 labels. Five fine-tuning seeds from one formula checkpoint have mean residual correlations of 0.994 (conductivity) and 0.999 (stiffness). On conductivity the formula network compresses its predictions (slope 0.64) and has four times the 400-label error of ImageNet initialization. Two confounds interact: the generator renders the low-conductivity phase with the opposite contrast polarity, and the exponential moving average (EMA; decay 0.995, no bias correction) still holds 16% of the initial weights after the 400-label schedule. Without the EMA the slope is 0.97, and over three subsets and five seeds the mismatched checkpoint has 1.4 times the ImageNet error (0.0119 versus 0.0084). Three pretraining runs per polarity leave the comparison inconclusive: matched means (0.0086, 0.0111, 0.0118) fall below mismatched means (0.0119, 0.0120, 0.0123) in aggregate (0.0105 versus 0.0121) but overlap on two of three subsets, and the matched spread (0.0032) exceeds the mismatched spread (0.0004). On stiffness without the EMA, formula pretraining lowers the 100-label error by 18% versus ImageNet over five seeds, including a re-initialized head, while ImageNet seeds also correlate at 0.992. Weight averages that start at the pretrained weights can dominate small-budget comparisons and should be rechecked at the last weights. 

---
# Deep Discriminative Feature Fusion of Convolutional and Attention-Based Recurrent Representations for Subject-Independent ECG Stress Recognition 

**Authors**: Zeeshan Ahmad, Muhammad Rehman Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2611.29870)  

**Abstract**: Stress has become a significant concern because of its negative impact on both physical and mental health and therefore drives the need for reliable automated systems to assess stress levels. Electrocardiogram (ECG) signals provide a non-invasive and objective measure of physiological stress responses and have emerged as a promising modality for continuous stress monitoring. While deep learning techniques have achieved encouraging results in ECG-based stress detection, many studies focus only on one type of feature representation, and therefore limit their ability to capture both the morphological patterns and temporal dynamics embedded in ECG signals. To address this limitation, in this paper, we propose three deep learning-based feature fusion frameworks for ECG-based stress classification. These are Residual Feature Fusion (RFF), Bidirectional Feature-Wise Cross-Attention Fusion (BFCAF), and a novel Cross-Attention Residual Fusion (CARF) framework that combines the strengths of residual learning and bidirectional cross-attention within a unified architecture. A 1D Convolutional Neural Network (1D-CNN) and a BiLSTM-Attention network are used to extract complementary morphological and temporal representations from ECG signals, respectively. The proposed fusion frameworks leverage these heterogeneous representations to improve feature interaction, information preservation, and stress discrimination. The proposed frameworks are evaluated using subject-independent Leave-One-Subject-Out Cross-Validation (LOSOCV) on the RML and WESAD datasets for three-class stress classification. Experimental results show that the proposed feature fusion methods consistently improve performance over individual CNN and BiLSTM models, with CARF achieving the highest accuracies of 83.1% and 77.91% on the RML and WESAD datasets, respectively. 

---
# Semantic repeatability and reproducibility of GPT-5 and Gemini 3 Flash on H5N1 One Health questions 

**Authors**: Luizza Dalla Pria, Amelia l. Nelson, Renata Ivanek  

**Link**: [PDF](https://arxiv.org/pdf/2611.29859)  

**Abstract**: Highly pathogenic avian influenza A(H5N1) has re-emerged as a One Health concern since 2024, and the public is increasingly consulting large-language-model (LLM) chatbots for health information during outbreaks. Before a chatbot's accuracy can be meaningfully assessed, its consistency must first be understood, so that any single response can be trusted as a fair representative of the model's behavior. We characterized the semantic repeatability and reproducibility of GPT-5 (OpenAI) and Gemini 3 Flash (Google) on 15 H5N1 One Health questions sourced from social media discourse by adopting and extending a centroid-based evaluation framework. Each question was submitted 30 times and reformulated using five Chain-of-Thought (CoT) variants with 30 runs each, yielding 5,400 responses. Pairwise cosine similarity was computed on MPNet embeddings, with sensitivity analyses across three additional transformers. Statistical inference combined standard nonparametric tests (paired Wilcoxon and Friedman with Holm--Bonferroni correction) with inter-embedding agreement statistics (ICC(2,k) and Cronbach's $\alpha$) introduced here as a methodological addition. Both models produced high mean consistency (0.93--0.97, rescaled to {[}0, 1{]}). GPT-5 showed reproducibility significantly higher than repeatability (W = 3.0, p = 0.0003), whereas for Gemini 3 Flash the two metrics were statistically indistinguishable (W = 42.0, p = 0.33). Differential Diagnosis ranked among the top two CoT framings for consistency in both models. The tested chatbots behave as semantically stable sources on H5N1 One Health questions, and the developed consistency evaluation framework is portable to other domains. 

---
# MobileGrail: Learning Compact Guardrails for On-Device deployable Mobile GUI Agent Safety 

**Authors**: Zhiyu Zhang, Yuankai Li, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.29847)  

**Abstract**: Runtime guardrails protect mobile GUI agents by assessing each proposed action before execution, but existing guardrails rely on cloud hosted large multimodal models, exposing every screen to a remote server, including private information such as payments, messages, and medical records. We propose MobileGrail, an on-device deployable model that addresses both. It maintains a recurrent task state that retains only the evidence subsequent safety decisions depend on, so that authorization and execution progress persist beyond the short context. To intervene early while keeping the safety boundary precise, it learns two labels, unsafe and block: unsafe marks the action that commits a violation, whereas block may be issued at any step of an intervention window, which spans from the first action that the user's authorization no longer explains to the committing action itself. To support training, we build an environment of 32 applications whose backends verify the effect of every action, supplying verified labels for supervision and rewards for reinforcement learning to calibrate intervention timing. We also build MobileVeto, an offline benchmark of unsafe trajectories largely on held-out applications, each paired with a benign counterpart. On MobileVeto, MobileGrail built on Qwen3.5-2B intercepts 76.2% of violations, up from 64.5% for its prompted backbone, while reducing premature intervention from 58.6% to 5.2%, outperforming the cloud-based OS-Sentinel on both. We further conduct online evaluation on GhostEIBench, where MobileGrail guarding a Qwen3.8-27B agent reduces fully successful attacks from 13 to 3 of 110 tasks while raising task completion from 48 to 58. Quantized to 4 bits, MobileGrail runs on a Snapdragon 8 Elite at a median latency of 4.7 s, comparable to the 5.1s of cloud-hosted Gemini 3.8 Flash. 

---
# Who Makes the Decision? Auditing an Agentic LLM Pipeline for Illicit Cryptocurrency Transaction Monitoring 

**Authors**: Vaishnavi Chaughule, Yuchen Zhang, Mohammed Ahnaf Tajwar, Sai Akshay Menta, Saanidhya Vats, Vinija Jain, Aman Chadha, Divya Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2611.29829)  

**Abstract**: Agentic pipelines that combine graph detectors, deterministic rules, retrieval-augmented generation and an LLM judge are being proposed to monitor illicit cryptocurrency flows and to write regulation-citing justifications for analysts. They are evaluated on end-to-end accuracy or on the quality of those justifications, and neither shows which component made a decision. We audit such a pipeline on the Elliptic++ Bitcoin dataset with decision-source attribution, a trace-level protocol with three layers: per-case actual causation, interventional Shapley credit, and mediation of upstream risk scores. A deterministic wallet-score override carries about 94% of the credit for flags, and a verification gate cancels most of the judge's own flags, which fall on licit transactions. Testing the judge alone over 76 evidence bundles, four backbones and two prompts, we find that it follows an injected risk score on the anonymous transaction features it is given, although stronger models decide correctly when the facts are explicit; that a context too short for the system prompt leaves it with no verdict at all; and that it cites regulatory clauses that do not apply, which a substring-based citation check accepts. Hiding the score does not fix the judge, because evidence gathered with the score visible still encodes it. For illicit-activity monitoring, an LLM's justification should therefore not be read as evidence of which component flagged a transaction. 

---
# Memphis: A Case Study in Seeding an Agent Task from a Model's Real-World Attempt 

**Authors**: Ibrahim A. Ahmed, Ahmed Sameh Khalil  

**Link**: [PDF](https://arxiv.org/pdf/2611.29824)  

**Abstract**: Agent benchmarks are usually made harder by scaling up the data their tasks are built from, yet models that do well on them still fail at real work. We propose building simulated tasks from a model's own attempt at an open-ended task in the real world, so that the model's failures in the simulation resemble its failures in reality. As a case study, we had GPT-6.1 Sol manage two human workers in Cairo over chat for five shifts, with a budget of 7,000 Egyptian pounds, to build as many USB desk lamps as it could. It built two. From the real-world attempt's transcript, we build Memphis, an evaluation in a model-simulated world. Memphis is hard: GPT-6.1 Sol delivers 1.0 lamp per trial on average and repeats its real-world mistakes, buying too few stands and giving up the search for them. With a one-page hint on where to buy parts and who can wire them, it delivers 9.2. Claude Opus 5.5 delivers 2.2 without the hint and 4.6 with it. We release the task, the trials, the audits, and a redacted record of the real-world evaluation. 

---
# Blind to What They Can Fix: A Detection Gap in Self-Improving Agents 

**Authors**: Teng Cao, Yu Deng, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2611.29807)  

**Abstract**: AI agents can write programs that carry out a task without calling a model at every step, so the task runs fast and its logic stays open to inspection. Agents improve these programs by running them, studying their failures, and revising the code, a form of recursive self-improvement (RSI). We ask whether this process also yields runtime checks that can be trusted. In five task domains, with up to seven author models, agents bring their programs to high scores. Yet when the environment changes, the checks they were told to write stay silent on half or more of the resulting failures in four domains. These failures can be detected and repaired: a monitor built from records of normal runs catches most of them in three domains, and the same agents usually repair them once called. The checks do not examine the recorded relations that reveal the failures, and the strongest agent writes checks that catch most of them once told that its checks are the only thing that calls it back. An unreported failure stays unrepaired, which limits what self-improvement is worth in autonomous operation. Evaluations of self-improvement should measure not only scores but also whether a system notices its own failures. 

---
# Latent-Predisposition Personas for Heterogeneous and Reliable LLM Survey Simulation 

**Authors**: Geonwoo Bang, Dongho Kim, Moohong Min  

**Link**: [PDF](https://arxiv.org/pdf/2611.29805)  

**Abstract**: LLM personas are used to simulate survey publics, which should reproduce not only answer distributions but also within-person coherence and stability. Drawing on mass belief systems research, we show that persona-conditioned LLMs produce over-constrained publics. Across three model families and three U.S. surveys, opinions on unrelated issues are 1.6-4.6 times as correlated as among humans, and party identification explains 2.5-7.6 times as much answer variance. Common remedies (temperature, diversity instructions, backstories) weaken these correlations by making answers more random. On Qwen, their ten-day test-retest reliability falls to 33-59% of the human level, and when made consistent they lose correlation fit. We propose Latent-Predisposition Personas (LPP), whose persons differ stably, not randomly. Each persona receives political engagement, value priorities, and party-relative issue deviations that grow as engagement falls. Answers are sharpened and calibrated to toplines and party crosstabs. At selected operating points, LPP is never significantly worse than any baseline and far more reliable. At matched human-level reliability, it has the lowest correlation error in all six model-survey combinations, also at a lower target. Errors still exceed those between human samples, and gains on framing effects and a frontier closed model are partial. 

---
# Dimension Is Free, Horizon Is Not: When Evolution Strategies Rival Policy Gradients for LLM Fine-Tuning 

**Authors**: Qiyao Liang, Kajetan Schweighofer, Xin Qiu, Risto Miikkulainen  

**Link**: [PDF](https://arxiv.org/pdf/2611.29795)  

**Abstract**: Policy gradient (PG) methods such as GRPO dominate LLM fine-tuning, yet recent work showed that Evolution Strategies (ES) also scale to fine-tuning models with billions of parameters. Which method needs fewer samples, and on which tasks, remains unclear. We view the two methods as estimators of the same gradient that differ in where they explore. Because ES perturbs parameters, its noise grows with the number of parameters, and because PG samples actions, its noise grows with the number of generated tokens. Only noise along the directions that change the reward slows learning. The cost of ES then depends on the effective dimension, the number of independent skills a task requires rather than its parameter count, while the cost of PG depends on the effective horizon, which counts only the tokens that share both credit and weights with the reward-carrying ones. We train both methods on a synthetic task where the number of skills and the effective horizon can be varied separately. PG is cheaper when the horizon is short, ES becomes cheaper once it is long enough, and this crossover moves to longer horizons as the number of skills grows. We then measure the effective horizons of 20 NLP, math reasoning, and agentic tasks and compare the two methods with Qwen2.5-7B. PG leads on short-horizon tasks, the two tie on the moderate-horizon math ones, and no task has a horizon in the regime where the theory predicts an ES advantage. Terminal-Bench has a far longer horizon. There, in a preliminary comparison with Qwen3.5-4B, ES without any credit assignment reaches a higher success rate than the PG checkpoint released with Tmax, as the theory predicts. ES should thus have the advantage on long-horizon tasks with few independent skills and a nonvanishing reward signal. Current benchmarks are only at the onset of this regime, so ES should become more competitive as agentic tasks grow longer. 

---
# Who Sent It or How It Moved? Sender Shortcuts in Cryptocurrency Anti-Money-Laundering Benchmarks, Audited on Elliptic2 and Elliptic++ 

**Authors**: Mohammad Nasir Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2611.29744)  

**Abstract**: Two public benchmarks dominate graph-based anti-money-laundering (AML) research on Bitcoin: the transaction-level Elliptic dataset, extended as Elliptic++, and the subgraph-level Elliptic2. In both, a label is assigned according to the entity that sends the funds. We ask whether models trained on these benchmarks learn how laundering moves or merely who sent the money, and we propose a diagnostic protocol to tell the two apart: a sender lookup baseline, a novelty baseline, a sender-disjoint split with a leakage check, and a decomposition of performance into sender, receiver and intrinsic signal. On Elliptic2 the task largely reduces to classifying the sender. Under the standard random split, a no-learning lookup of sender history reaches a PR-AUC of 0.954, against 0.975 for gradient boosting on 456 features; our reimplementation of RevClassify reproduces its reported 0.97 and falls to 0.126 without sender information. When senders are held out, performance drops to 0.746, and the subgraph's own structure, transactions and receivers fall from 0.619 to 0.186. On Elliptic++ the same protocol finds the shortcut negligible: sender history reaches only 0.340 against 0.986 for the full model (3.5 against 10.1 times the test prevalence), adding it to the full model raises PR-AUC by less than 0.004 in every seed, and the transaction's own features keep 0.883 when sending addresses are held out. We trace the difference to the unit of sender identity: Elliptic2 senders are entity clusters that recur across many labeled subgraphs, whereas Elliptic++ senders are individual addresses, and fewer than half of test transactions have a sending address seen in training. A shared sender-based label rule thus produces a severe shortcut in one benchmark and a negligible one in the other, so the diagnostic must be run on each benchmark. 

---
# Reinforcement Learning from LLM-Generated (Suboptimal) Demonstrations 

**Authors**: Anurag Kadkol, Lawson L.S. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2611.29732)  

**Abstract**: Reinforcement learning (RL) has achieved substantial progress across many domains, but training an agent \textit{tabula rasa} remains highly sample-inefficient. A common remedy is to initialize the agent using a neural network pretrained on demonstrations. However, collecting demonstrations is expensive, and the demonstrations themselves can be suboptimal. In this work, we use a large language model (LLM) to generate an advice policy that produces demonstration data with effectively no human cost. We use this data to warmstart a DQN agent, and use the advice policy itself to guide exploration during training. Because LLM-generated advice is often suboptimal, we further introduce a value warmstart method that initializes the agent to recognize promising states from the advice policy's behavior, without directly copying its actions. To characterize the robustness of this method, we systematically perturb individual sub-skills of an LLM-generated advice policy for a long-horizon, sparse-reward task. Degraded advice still meaningfully accelerates learning relative to unaided baselines. These results suggest that LLM-generated advice, though suboptimal, can be a cheap alternative to costly human demonstrations. 

---
# Fin-RSI: A Unified Library of Financial Skills, Checks, and Self-Improvement for LLM Agents 

**Authors**: Wanghao Ye, Shwai He, Sihan Chen, Yiting Wang, Bowei Tian, Guoheng Sun, Ziyi Wang, Ziyao Wang, Yexiao He, Zheyu Shen, Meng Liu, Yuning Zhang, Meng Feng, Yifei Dong, Yanhong Qian, Yang Wang, Siyuan Peng, Yilong Dai, Zhenle Duan, Joshua Liu, Lang Xiong, Aarosh Das, Zhaoyi Liu, Daeun Jung, Aditya Sarkar, Parastoo Pilevar, Xiaochuan Gong, Yancheng Zhu, Yingying Fan, Qiong Wang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.29716)  

**Abstract**: As large language models (LLMs) become more capable, financial agents need tools they can readily discover and use. Yet financial software is fragmented across libraries with different interfaces and conventions, making it difficult to build reliable agent workflows. We present Fin-RSI, a unified Python library of financial skills, checks, and self-improvement designed for LLM agents. It brings together tools for market analysis, forecasting, portfolio construction, and trading research through shared interfaces across models and agent frameworks. Agents can find suitable tools from natural-language requests, reuse tested implementations, and improve generated programs through checking and feedback. Developers can extend the library with new tools without rebuilding integrations for each model. Across 20 open-weight models on six trading-position tasks, tool discovery increased the proportion of episodes producing valid programs from 9.7% to 29.2%; for models with at least 14 billion parameters, it rose from 23.6% to 64.6%. We release the library and benchmark as a shared resource for the community and plan to expand its tools and task coverage as financial agents evolve. 

---
# Wactorz: Self-Writing LLM Agents That Discover, Repair, and Migrate at the Edge 

**Authors**: Panagiotis Kasnesis, Lazaros Toumanidis, Amalia Contiero Syropoulou, Christos Chatzigeorgiou  

**Link**: [PDF](https://arxiv.org/pdf/2611.29619)  

**Abstract**: Most LLM agent frameworks build a crew that runs a task and exits. Physical environments (e.g., homes, labs, factory floors) need agents that stay up for months, survive their own bugs, and run on whatever hardware is already next to the sensors. We present Wactorz, an open-source runtime in which LLM-generated agents are long-lived, supervised actors: a planner turns a chat request into Python, compiles it into a new actor at runtime, and a supervisor restarts that actor alone, with its state intact, when it crashes. All coordination rides on MQTT, so agents can be spawned on, and live-migrated between, edge nodes such as a Raspberry Pi, and every internal event is a topic external systems can subscribe to. The recorded demo builds a camera-to-lamp automation from one chat request, adds a second agent that is auto-wired to the first by topic and carries a deliberate division-by-zero, watches that agent repair its own code without losing its count, then deploys a fresh Raspberry Pi node and live-migrates the agent onto it, in under four minutes of wall-clock time and \$0.13 of model usage. This extended version adds what the two-page demonstration paper could not: the Wiring view of the topic graph, Wactorz as a library for hand-written agents, how trained models are stored, sent to nodes and replaced, the security model, and the system's limitations. 

---
# Similarity-Constrained Reweighting for Complex Query Answering on Knowledge Graphs 

**Authors**: Daniel Daza, Alberto Bernardi, Luca Costabello, Christophe Gueret, Michael Cochez, Martijn Schut  

**Link**: [PDF](https://arxiv.org/pdf/2611.29596)  

**Abstract**: Machine learning models for Complex Query Answering (CQA) on knowledge graphs estimate the likelihood of answers that are not reachable via direct traversal. Recent work has proposed extending such queries with soft entity constraints, which require the target variable to be similar or dissimilar to specified sets of entities. We study a more general formulation where similarity constraints are specified on an arbitrary variable in a query, and introduce SCORE: a computationally efficient and interpretable method for incorporating similarity constraints on an arbitrary variable of a query. SCORE is based on a lightweight and interpretable score adjustment function that only requires tuning two hyperparameters on a validation set. Our experiments show that SCORE effectively incorporates preferences while maintaining strong query answering performance. It significantly improves ranking quality when constraints apply to the target variable and performs effectively in the more general case involving constraints on intermediate variables, and its performance remains robust when applied to several existing methods for CQA. Our implementation is available at this https URL. 

---
# Rethinking Closed-Ended Medical VLM Evaluation: From Stable Wrongness to Counterfactual Answer Sensitivity 

**Authors**: Qi Wu, Xiaoyu Liu, Ao Wang, Rongsheng Wang, Wenting Chen, Wenxuan Wang, Qingsong Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.29592)  

**Abstract**: As medical VLMs are increasingly evaluated for clinical decision-support scenarios, knowing when to trust their predictions is critical. Closed-ended medical VQA has become a common benchmark format because it enables automated scoring. Yet, systematic investigation into whether this format changes the reliability problem remains scarce. We identify a failure mode that we call stable wrongness: models can produce incorrect yes/no answers that remain confident and repeatable even when patient-specific visual evidence is removed or mismatched. This failure violates the premise behind sampling-stability uncertainty methods, including recent hallucination-aware calibration (HAC) variants: disagreement is informative only when errors are unstable. When shortcut-driven errors are themselves stable, sampling agreement can remain deceptively high. Motivated by this limitation, we introduce Counterfactual Answer Sensitivity (CAS), which shifts reliability estimation from answer stability to evidence sensitivity. CAS uses cross-modal counterfactual interventions, such as text-only, black, and corrupted images, to audit whether a committed yes/no answer remains supported when visual evidence is invalidated. Across 7 VLMs and 3 benchmarks, CAS outperforms the best HAC-family scores across all evaluated settings, achieving a macro-averaged AUROC gain of +0.150 for error detection. Mechanistic analysis further shows that CAS recovers hidden errors in the HAC-confident region and that its strongest standalone signal is counterfactual decision-boundary shift rather than hard prediction stability. Our findings suggest that closed-ended evaluation should test visual dependence, not only answer stability, in safety-critical medical contexts. 

---
# Scalar Tuned Lens Summaries Offer Limited Incremental Value for Mid-Generation Abstention 

**Authors**: Khurram Khalil, Khaza Anuarul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2611.29589)  

**Abstract**: Mid-generation abstention aims to stop an unpromising response before spending its remaining computation budget. Intermediate transformer states may help predict eventual correctness, but whether scalar summaries of their vocabulary predictions add useful evidence is less clear. We compare Tuned Lens entropy and divergence features with final-state and raw intermediate-state probes under matched, causal evaluation. Across eight instruction-tuned models on MATH-500, raw intermediate states improve ranking, whereas scalar-only probes usually trail the final-state baseline. Scalar hybrids have positive exceptions, but their advantage over position-matched permutation controls is small. Cross-fitted residual comparisons further show that gains over a final-state probe need not require aligned scalar features. Actual stopping runs on two model sizes and two benchmarks reveal a separate accuracy--computation trade-off, with token savings overstating wall-time savings for scalar policies. The evidence supports a restricted empirical conclusion about these scalar features, not a general negative claim about intermediate representations. 

---
# Layer-Level Control of Language Model Execution in Agent Frameworks 

**Authors**: Ran Guan, Yiqi Lin, Xiangxi Tian, Yige Luo, Mi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.29519)  

**Abstract**: Agent frameworks typically orchestrate language models through prompt-and-response interfaces. We propose a harness design that exposes intermediate model states and layer-level execution controls to the agent runtime. By connecting state collection, policy fitting, and execution, the design enables policies to allocate computation within and across models based on intermediate states. We demonstrate the approach with a prototype that learns stitching bridges and state-conditioned execution policies from paired model states, then applies them during inference to select among continue, exit, and stitch. For dispatcher supervision, we select the least-computation action whose full-vocabulary next-token prediction matches the source model's full-depth prediction, using continue when neither early action matches. We train these components on a frozen proxy corpus of 5,000 typed-decision items authored using source texts sampled from WildChat conversations, then evaluate the prototype on JevBench measuring task accuracy, agreement with full-model predictions, and end-to-end latency including runtime overhead. Preliminary results show an approximately 15% reduction in mean latency while maintaining task accuracy comparable to full-model inference. 

---
# Emotion-Aware and Context-Grounded Dialogue Agents for Personalized Video Game Interactions 

**Authors**: Michael Adjeisah, Jin Zhang, Pierluigi Vito Amadori, Ryan Spick, Eshani Kawshika Fernando, Kwame Omono Asamoah, Huiwen Zhao, Jian Jun Zhang, Guy Moss, Jian Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29372)  

**Abstract**: We present ChatNPC, a game companion that dynamically personalizes its responses in real time based on players' emotional shifts to enhance user immersion. The model integrates sequential data, leveraging the dynamic nature of streams to benefit from player information, spoken lines, and in-game context. We leverage recent progress in LLMs' tool-calling capabilities to extract vital information from memory and recognize potential constraints for accurate reasoning, thereby tackling the complexity of NPC conversation scenarios. The task is divided into a game sentinel-guided agent (SeGent), a memory capability, and a chat planning tool for reasoning instantiation. The approach uses a lightweight game template as an information framework, with relevant details and a thorough reasoning layout. We conduct extensive experiments on a newly developed game dataset for in-game context NPC dialogue and demonstrate that ChatNPC sequentially captures players' emotional shifts over time; responses are more naturalistic and human-like with appropriate conversational cues, pauses, and sighs; and utterances remain faithful to the dynamics of in-game context and player actions, supporting narrative continuity. 

---
# SAE-Audit: Clinical-Trial Prediction via Inference-Time Optimization of Black-Box LLMs Audited by Sparse Autoencoder Evidence 

**Authors**: Junbo Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.29368)  

**Abstract**: Clinical trials are costly and often take years to complete, making early outcome prediction valuable for trial planning. Large language models (LLMs) are increasingly used to forecast clinical-trial outcomes, yet their predictions remain unreliable and their rationales may not faithfully reflect the evidence underlying their answers. On tasks such as predicting whether a registered clinical trial will complete or terminate early, frontier LLMs perform only modestly above chance in our experiments. We introduce SAE-Audit, which optimizes a black-box LLM at inference time with evidence obtained from a deployable open model. A frozen sparse autoencoder (SAE) on Llama-3.1-8B-Instruct is converted into a vocabulary of clinical concepts. Sparse probes fitted on past trials determine which concepts are associated with the outcome. For a new protocol, the concepts detected by the probe are written as evidence, each with the direction and strength of its association with the outcome, and the LLM being audited optimizes its answer in the light of this evidence. Without additional training, SAE-Audit raises balanced accuracy of deepseek-v4-flash by 9.6--12.0 points on five clinical-trial prediction tasks and raises GPT-5.6-terra by 8.0 points on trial completion, exceeding other agentic inference-time optimization methods by a large margin. As an auxiliary, SAE probes module within SAE-Audit is itself accurate and auditable, providing natural-language explanations directly derived from verified concept contributions and protocol evidence, without requiring additional LLMs at inference time. SAE-Audit provides verifiable, concept-level evidence for more accurate and auditable clinical-trial prediction. It can also be embedded into biomedical AI agents such as Biomni to optimize their performance. 

---
# EMIP: Embodied Multi-Party Interaction Platform 

**Authors**: Koichiro Terao, Rikunari Sagara, Moe Nagao, Naoto Iwahashi  

**Link**: [PDF](https://arxiv.org/pdf/2611.29333)  

**Abstract**: Experimental platforms that enable multiple participants to engage in spoken dialogue and perform physical actions concurrently in a shared environment are important for studying cooperation between humans and embodied AI agents. In this paper, we propose EMIP (Embodied Multi-Party Interaction Platform), a virtual environment platform built on Minecraft. Through an Internet-based client-server architecture, EMIP connects multiple humans and multiple AI agents to the same virtual environment and supports full-duplex speech communication and simultaneous physical interaction. It provides interfaces based on the Model Context Protocol (MCP) for environmental observation, physical action execution, and audio input and output, enabling researchers to develop their own agent control methods using large language models (LLMs). Additional features include centralized control of an agent group through a single EMIP agent client, collaborative object transportation by multiple agents, and experimental support functions such as recording and retrieving observation histories. By integrating these features into a reusable platform, EMIP aims to support research on human-human, human-agent, and agent-agent cooperation under different participant configurations and agent control methods. Usage examples illustrate concurrent speech and physical action, task coordination, and overlapping speech. The source code for the platform is publicly available on GitHub (this https URL). 

---
# DoppelVet: Comparative Safety Verification for Self-Evolving Agents 

**Authors**: Yifan Duan, Du Yin, Siyuan Zheng, Chao Xue, Lihuan Li, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2611.29306)  

**Abstract**: Self-evolving agents improve by retaining feedback as reusable memories or skills, but every mistaken approval can shape many future behaviors. Two updates can provide exactly the same evidence to a verifier yet produce opposite safety outcomes, so the verifier cannot distinguish them from that evidence alone. We introduce DoppelVet, a verification framework that checks proposed updates before they enter an agent's persistent state. Without access to safety labels, DoppelVet constructs an alternative that matches every signal used by the existing verifier. It replays both updates from the same state and directly compares their behaviors in both presentation orders before deciding whether to retain the candidate. We evaluate DoppelVet on DoppelVet-Grid and DoppelVet-Tool, covering executable programs and natural-language memories. Across nine open-weight and proprietary verifiers, DoppelVet raises mean harmful-update detection AUC from 0.762 to 0.940 on DoppelVet-Grid and from 0.681 to 0.826 on DoppelVet-Tool, outperforming VaG and AgentDoG 1.5. On DoppelVet-Grid, it reduces trajectories experiencing degradation from 30.39% to 0.43% over a ten-generation self-update loop and leaves no final degradation. 

---
# MetaSkill: Reusable Problem-Solving Methods for Adaptive Skill Orchestration in LLM Agents 

**Authors**: Tianxing Wang, Mingming Zhao, Shuai Huang, Chaoyue Niu, Huiyang Xu, Fan Wu, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.29274)  

**Abstract**: Reusable skills provide specialized procedures, but large language model (LLM) agents must still determine how to organize and adapt their use. We introduce MetaSkill, a two-layer framework for reusing problem-solving methods across tasks and domains. The method layer selects from 26 curated methods and binds abstract roles to task objects. The execution layer uses the instantiated method to identify intermediate objectives, coordinate native skills, and guide repair or method reselection. On 89 SkillsBench tasks and 995 ToolHop examples, MetaSkill outperforms the evaluated baselines in all six configurations, with accuracy gains of 3.51-13.48% over the strongest baseline. Ablations support explicit method-guided orchestration beyond direct method injection and further gains from method reselection. 

---
# Neural Agent Compiler: Learning to Schedule Soft Dependencies 

**Authors**: Bo Yan, Peng Wang, Zihan Chen, Ziwen Kan, Zhen Tan, Huiyuan Chen, Bo Sun, Shuai Shao, Tianlong Chen, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29262)  

**Abstract**: Large language model agents accomplish complex tasks by coordinating multiple reasoning steps and tool operations, which form dependencies among their operations. Existing orchestration methods primarily consider hard dependencies that determine when an operation can legally begin, but overlook soft dependencies, i.e., information that is not required for execution but may improve an already executable operation. We identify soft-dependency scheduling as an important problem in agent execution. To this end, we introduce Neural Agent Compiler (NAC), a learned scheduler that predicts the task-quality and resource consequences of three execution modes and selects the highest-utility legal mode at a scheduling boundary. NAC represents the executable continuation, pending information event, and resource state in a shared memory, and uses a utility function to adapt the consequence predictions to different resource preferences. Across AppWorld, WebShop, and $\tau^2$-bench, NAC achieves 276/417 AppWorld task completions, a 74.69 WebShop Task Score, and 59/100 $\tau^2$-bench successes. Additional studies show that the learned scheduler reduces execution work, establishing soft-dependency scheduling as a promising solution for orchestrating tool-using agents. 

---
# NoeVera: Can Knowledge Guide Experiments and Experiments Refine Knowledge? 

**Authors**: Pinaki Prasad Guha Neogi, Spyros Mastorakis, Ankur Mallick, Saravan Rajmohan, Victor Rühle  

**Link**: [PDF](https://arxiv.org/pdf/2611.29255)  

**Abstract**: With the growing demand for large language model (LLM) services, optimizing LLM serving systems is becoming increasingly complex, and the bottleneck is no longer predicting how a given configuration will perform. The harder problem is deciding what to investigate: which mechanisms matter for a workload, which configurations are worth testing, and what evidence a decision still needs. In this work, we present NoeVera, an agentic Second Brain that goes beyond just one-shot retrieval by turning papers, code, benchmarks, and experimental results into a persistent, provenance-preserving knowledge base with three planes: source evidence, reusable synthesis, and agent memory. Decision-aware retrieval assembles compact evidence packs, role-specific agents reason over them and invoke a simulator (Vidur or GAUSS) whenever quantitative support is needed, and every finding is written back as a scoped evidence record, so knowledge guides experiments and experiments refine knowledge. In configuration search over four serving scenarios, NoeVera comes within 5% of the optimum in a median of 6-7 simulator calls, versus 9 for Bayesian optimization and 15 for a knowledge-free agent with the same tools; on A100 GPUs, its representative configurations deliver ~1.30x vLLM-default throughput and meet every scenario's latency objective. When reused, provenance-preserving evidence records over-generalize on only 8% of applicability traps, versus 28% for a general-purpose agent memory. On the held-out 48-task split of a 72-task benchmark, NoeVera scores 29% higher than an equally equipped agent and 36% higher than its single-call retrieval counterpart (0.80 vs. 0.62 and 0.59), and even when baselines receive NoeVera's own prompts, a statistically significant part of its evidence-evolution advantage remains. 

---
# GameAnything: From Any Inspirations to Playable Game Worlds 

**Authors**: Wenyi Wu, Siqi Liu, Yuru Feng, Zixuan Song, Aayush Salvi, Sibo Zhu, Kun Zhou, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29238)  

**Abstract**: AI-driven game development holds the potential to transform diverse human inspirations into interactive experiences. However, existing systems and benchmarks primarily focus on explicit development specifications, with limited support for adapting multimodal references and evaluating actual gameplay experiences. We introduce GameAnything, an open-source agentic platform that transforms multimodal inspirations into playable extensions of existing games through iterative development. GameAnything integrates coding agents with an independent playtesting critic, persistent development state, and reusable experience memory to support continuous refinement. We further present GameAnything-Bench, a benchmark spanning diverse reference modalities, game creation objectives, and gaming environments, with exploratory gameplay testing to assess both functional correctness and gameplay quality. Evaluations of nine coding agent configurations reveal that current agents can adapt reference content but still struggle with playable interactions and precise modifications. Moreover, multimodal references improve content fidelity, while iterative refinement yields substantial but varying improvements across agents. These findings highlight both the potential and remaining challenges of inspiration-driven game development. 

---
# MVAgent: Per-Video Agents under Global Coordination for Multi-Video Understanding 

**Authors**: Kewei Wei, XueXiang Wen, Bocheng Hu, Weili Xu, Haiyang Yu, Kaiyue Zhou, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29154)  

**Abstract**: Multi-video understanding requires systems to combine evidence from different videos for cross-video reasoning. Answering such questions involves two kinds of decisions. Observation decisions must adapt to each video's content, whereas cross-video judgments require relating evidence from different videos based on the question. Most existing agents make both kinds of decisions in one controller, whose context accumulates observations from every video. We separate them in MVAgent, a training-free Multi-Video Understanding Agent framework with one LocalAgent per video. Each LocalAgent plans observations of its own video and keeps a private observation history. A GlobalAgent coordinates cross-video reasoning by instructing the LocalAgents and combining their reports with joint observation. We evaluate MVAgent on CVBench, MVU-Eval, and CrossVid, using three Qwen3.5 models of different sizes and GPT-6-sol. With Qwen3.5, our framework improves aggregate benchmark scores by 5.1 points on average over Single-pass inference with the same model and by 5.0 points over the strongest prior agent using the same model. With GPT-6-sol, the average gain over Single-pass inference with the same model is 4.3 points. On the same hardware, MVAgent also runs 1.8-2.7 times faster than prior agents of comparable size. 

---
# The Retry Tax: Why Static Same-Task Retries Can Cost More per Completion 

**Authors**: Wei-Jung Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29133)  

**Abstract**: Retrying an agent after failure can complete more tasks while increasing the cost per completion. We study a static policy that stops at verified success or an attempt cap and otherwise resets the original prompt and workspace without passing feedback to the next request. Under stationary task success probabilities, independent attempts conditional on the task, and a common attempt cost, easier tasks tend to succeed earlier. When task success probabilities differ, the unresolved mix shifts toward harder tasks. Specializing classical survivor selection, we show that cost per completion is at least as high as in one-shot serving and that marginal cost per additional completion is nondecreasing wherever the completion increments are positive. These are different quantities: one averages spending over all completions, while the other prices the completions added by another attempt. We then arrange each task's existing benchmark runs in different orders, stop each sequence at verified success or the attempt cap, and average the resulting completion and spending at each supported cap. For one OpenHands configuration, replay estimates completion of 59.7 percent with one attempt and 70.0 percent with four, while cost per completion rises from about USD 0.70 to USD 1.32. The recorded trade-off describes fixed configurations on the available tasks. Choosing a cap requires comparing the additional spending with the application's value for the additional completions. 

---
# Do You Even Know What I Want? Improving LLMs Ability to Detect Hidden User Intent 

**Authors**: Kelsey Sikes, Sarath Sreedharan  

**Link**: [PDF](https://arxiv.org/pdf/2611.29028)  

**Abstract**: As with conventional AI agents, a common source of misalignment between Large Language Models (LLMs) and their human users is misspecified goals. Often, this happens when the human has skewed beliefs about the system's model and provides it with information that causes behavior different from what they expected, with potentially unwanted side effects. To ensure goal alignment, one solution would be for the LLM to ask the human questions about their objectives. However, it has been well-documented that LLMs most often provide default responses to user requests rather than try and ask questions to clarify their true intent. In this paper, we address these challenges by adapting a novel formulation for the objective misspecification problem to work with LLMs. We then test if LLMs that are prompted to ask questions about a user's true underlying objectives are better at identifying them compared to those that do not. We evaluate our method using four state-of-the-art LLMs tested on five standard IPC domains, showing that for some domains, asking questions greatly improved model performance. 

---
# Learning from Another Angle: Rotation-Based Test-Time Training for Multimodal Large Language Models 

**Authors**: Changyu Liu, Fei Zhao, Yiyang Liu, Xi Xiao, Tianyang Wang, Cheng Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.28972)  

**Abstract**: Multimodal Large Language Models (MLLMs) achieve strong performance on vision-language tasks but often remain vulnerable to distribution shifts due to their fixed parameters during inference. While Test-Time Training (TTT) offers a pathway for dynamic adaptation, existing approaches suffer from prohibitive computational overhead due to multiple sampling and the risk of error reinforcement from biased self-generated supervision. In light of this view, we propose RoTTT (Rotation-based Test-Time Training), a simple and efficient TTT approach that enables instance-specific refinement at inference time. For each test instance, RoTTT performs a small number of gradient updates by predicting the rotation angle of the input image. The known rotation angles naturally serve as a reliable self-supervised signal, enabling lightweight and stable adaptation without requiring additional annotations. Extensive experiments across multiple vision-language benchmarks demonstrate that RoTTT consistently improves performance across different MLLMs, providing a practical and efficient paradigm for on-the-fly MLLM adaptation. 

---
# When Wrong Answers Remain Unsettled: Failure Dynamics in Recurrent Reasoners 

**Authors**: Jiucheng Zang, Congkai Peng, Yankai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.28971)  

**Abstract**: When a recurrent reasoner returns a wrong answer, is its computation finished or merely stopped? We test this on a trained Hierarchical Reasoning Model (HRM; 27M parameters, Sudoku-Extreme) by running the recurrence for one more inference budget and comparing with an unperturbed control. Solved answers persist (0.95 of 512 endpoints). Wrong answers mostly do not: 0.42 of near-miss answers (n=170) and 0.18 of all 512 wrong endpoints keep their answer. Perturbations from 0.05x to 20x the terminal-update scale barely change these rates, so the measurement reflects continued recurrence, not noise sensitivity. The solved-above-wrong ordering also holds in five further settings (a second seed, TRM, Maze, GRAM-guided TRM, N-Queens), but a fixed 1/2 persistence threshold does not, and neither do HRM's endpoint-spectrum and fast-versus-slow predictability patterns, which are not universal diagnostics. Perturb-and-reselect (known from prior work) is consistent with this: frozen noise gains 4.8 accuracy points (197 gained, 2 lost), but compute-matched extended depth gains 5.6 and Sudoku symmetry augmentation 34.6, so noise is not the best use of the budget. Conclusions are limited to finite horizons and one primary HRM checkpoint; they do not exclude asymptotic attractors. 

---
# PruneNAS: Constraint Aware Neural Architecture Search based Pruning 

**Authors**: Sudhakar Sah, Nikhil Chhabra, Honnesh Rohmetra, Andy Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.28964)  

**Abstract**: Neural Architecture Search (NAS) has achieved state-of-the-art performance across diverse tasks, but its high computational cost remains a major barrier. We introduce PruneNAS, a novel framework that integrates structured pruning into NAS to efficiently explore high-performing architectures while leveraging pretrained models. Unlike conventional NAS methods that train each candidate from scratch, PruneNAS generates a candidate pool by pruning pretrained networks at multiple granularity levels and formulates the selection as a global optimization problem using training-free importance metrics. This approach enables rapid evaluation of candidate architectures under strict resource constraints, dramatically reducing search time from hundreds of GPU hours to mere minutes. Experiments on standard benchmarks demonstrate that PruneNAS discovers compact, high-accuracy architectures that nearly halve the computational cost (MACs) of prior methods on ResNet-50, while also showing strong generalization on object detection tasks with YOLO. Our method bridges structured pruning and NAS, offering a practical solution for resource-constrained model deployment. 

---
# STEdgeFormer: MCU and NPU Deployable Transformer Architecture 

**Authors**: Sudhakar Sah  

**Link**: [PDF](https://arxiv.org/pdf/2611.28935)  

**Abstract**: Transformer-based sequence models have achieved strong performance in language and time-series modeling, but self-attention remains computationally and memory intensive for deployment on resource-constrained edge accelerators. We propose STEdgeFormer, a lightweight, hardware-aware sequence architecture designed for efficient execution on embedded neural accelerators. STEdgeFormer combines polynomial mixture layers and DyT normalization with an accelerator-aware causal memory block, replacing pairwise self-attention with compact causal aggregation. On TinyStories, the fully NPU-supported STEdgeFormer (MW) achieves a perplexity of 7.39 with 4.12M parameters and 6.51 MB RAM, with an end-to-end latency of 40 ms. In comparison, the standard 4.12M-parameter configuration achieves a lower perplexity of 6.79 but requires CPU fallback for unsupported operators, resulting in 1450 ms latency. Vanilla Transformer, on the other hand, achieves a perplexity score of 5.71 using a 4.40M-parameter model, but it cannot be deployed on the hardware. On downstream tasks, STEdgeFormer achieves 94% accuracy on ECG5000 and 93% accuracy on Speech Commands with 1.39M parameters. These results demonstrate that hardware-aware architectural choices and operator compatibility can have a substantial impact on practical inference latency, beyond model size and task accuracy alone. Rather than targeting state-of-the-art accuracy on individual benchmarks, STEdgeFormer is designed to provide a deployable sequence modeling architecture under the memory, computational, and operator constraints of embedded neural accelerators. 

---
# OmniDecisionBench: Evaluating and Benchmarking the Multimodal Jev-Style System One Decision Models 

**Authors**: Tianxiang Gao, Jinzhe Li, Yue Zhou, Dandan Guo, Qi Wang, Diana Inkpen, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28924)  

**Abstract**: Multimodal System One models answer visual tasks with typed decisions and native probabilities, yet no holistic evaluation compares them with general VLMs or maps either class's capability boundaries. We introduce OmniDecisionBench, a holistic benchmark of multimodal System One models. Its main evaluation assigns 45 datasets (192,975 records) to five capability families and twelve primary angles across four typed interfaces, scoring accuracy and candidate-probability calibration. A validation layer adds three single-factor diagnostics varying visual presentation, textual evidence sufficiency, and target dataset. Across three System One models and two general VLMs, three regularities emerge. First, strengths depend on angle and metric: System One models lead discrete tasks overall with lower ECE15 on most perception, reading, and reasoning angles (Jev 1.14: 80.14% accuracy, 4.50% ECE15), while general VLMs lead ordinal grading and outperform Jev 1.14 among the three models tested on direct Box grounding. Second, multi-image and abstract reasoning bound every model, where confidence rather than accuracy separates them. Third, reliability shifts with the input: rendering only options nearly triples Jev 1.14's selective risk at nearly flat mean signed bias, and donor risk cutoffs often exceed target risk across unseen datasets and task families. OmniDecisionBench thus maps which model class suits each angle and where reliability ends. The code and dataset will be available at this https URL. 

---
# When Do Better Visual World Models Make Better Plans? Decision-Aligned Selection for Latent Planning 

**Authors**: Chan Aristella Lu, Arya Fayyazi, Junil Yeo, Jin Lu, Kyu Hyung Lee, Mehdi Kamal, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2611.28919)  

**Abstract**: Visual world models enable goal-conditioned planning by predicting the outcomes of candidate actions in latent space. However, planning depends on how well a model ranks candidates, whereas training optimizes how accurately it predicts each one. We identify the Decision-Relevant Residual (DRR), the part of the cost error that can reorder candidates, and show that the more training reduces the DRR, the more it improves selection (Spearman $\rho = 0.82$, vs. $0.61$ for prediction error). Building on this insight, we introduce Decision-Aligned Selection (DAS), a plug-in selection rule that scores candidates by the standardized consensus of several independently trained checkpoints, which averages out their checkpoint-specific residuals. Across two model families (LeWorldModel and DINO-WM) and five tasks, DAS selects better actions than single checkpoints in 12 of 18 configurations and outperforms training one model three times longer in 14 of 18. In closed loop, DAS improves success over single checkpoints in 6 of 9 pre-registered LeWM comparisons. 

---
# STRIDE: Strategy-to-Tactics Retrieval-Guided Skill Disclosure and Evolution for Agentic Reinforcement Learning 

**Authors**: Shixin Hong, Wanqing Cui, Yibo Hu, Naen Xu, Zhiwei Huang, Meiguang Jin, Junfeng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.28907)  

**Abstract**: Skill-augmented reinforcement learning enables LLM agents to reuse task-solving knowledge across episodes, but existing systems typically expose agents to a flat skill pool and update skills only from coarse, episode-level outcomes. This conflates two distinct uses of experience: a stable strategy that should frame the whole episode, and tactical know-how that should be retrieved only when the agent reaches a local execution gap. We propose STRIDE (Strategy-to-Tactics Retrieval-guided skIll Disclosure and Evolution), a framework that brings progressive skill disclosure into agentic reinforcement learning. STRIDE keeps the selected strategy visible as episode context, exposes a compact tactic menu, and loads full tactic content only after an explicit retrieval action. This interface turns each retrieval into an event-level credit unit: the same event provides process feedback for learning when to retrieve and supplies evidence for skill-bank evolution. On ALFWorld, STRIDE achieves a 92.7% average success rate with Qwen3-4B and a 94.9% average success rate with Qwen3-8B. On WebShop, it improves over the strongest reproduced skill-augmented baseline by 4.5 score points and 1.9 success-rate percentage points. These results show that skill reuse is most effective when persistent strategies preserve global task structure and tactics are disclosed only at local decision bottlenecks. 

---
# StudioBench: Benchmarking Agentic Multimodal Large Language Models for Productive Audiovisual Creation 

**Authors**: Xinglong Xu, Yuxia Zhu, Siyuan Li, Lixu Tian, Xiaofei Zhao, Yuxin Liu, Caijun Jia, Jintao Chen, Bihui Yu, Alden Cai, Asher Qin, Jingxuan Wei, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2611.28902)  

**Abstract**: Video generators can now render convincing short clips, yet complete audiovisual works require coordinating visuals, sound, and editing across multiple stages. Existing benchmarks focus on individual generators or video agents under narrower production settings, leaving end-to-end creation by general multimodal large language models (MLLMs) underexplored. We introduce StudioBench, a benchmark comprising 160 tasks across eight video genres and five task types, spanning diverse audiovisual production requirements. Models produce minutes of finished audiovisual content using 33 atomic generation, editing, and inspection tools, without shell access or workflow templates. We evaluate the delivered films, while analyzing reconstructed workflow graphs to characterize production behavior and diagnose failures. Across seven MLLMs, explicit requirements are nearly saturated, yet production quality varies substantially in picture clarity and composition, object structure, motion, and lip sync. All models spontaneously organize multi-stage production workflows without predefined templates. However, higher-quality models exhibit more selective production workflows, whereas additional tool calls alone show little association with quality. These findings identify the missing quality loop of effective inspection, selection, and revision as a key bottleneck in productive audiovisual creation. 

---
# FormaFin: Verifiable Financial Analysis with Solver-Guided Reinforcement Learning 

**Authors**: Bethel Hall, William Eiers  

**Link**: [PDF](https://arxiv.org/pdf/2611.28881)  

**Abstract**: Formally verifying financial analyses generated by large language models (LLMs) remains challenging as financial rules are largely expressed in natural language and rarely have corresponding formal specifications. Training language models to formalize these rules is hindered by the scarcity of labeled data and reliable training signals, while confidential financial data favor small models that run locally. We introduce FormaFin, a framework that trains small open language models to translate financial rules into reusable SMT formulas. FormaFin integrates teacher distillation with reinforcement learning using an SMT solver to compute rewards from numerical thresholds stated in financial rules. We evaluate FormaFin on corporate loan monitoring, in which lenders check borrowers' financial figures against loan agreements. To evaluate our framework, we curate \bench{}, a benchmark containing 143 loan agreements and 2,527 questions covering compliance, earnings cushion, and debt capacity. Our results show that a 7B formalizer achieves 94.8\% accuracy, outperforming open LLMs up to ten times larger. Across three model families, accuracy ranges from 92.6 to 96.8\%. Our framework also detects 95.6 to 99.7\% of errors in other LLMs' financial analyses. We make FormaFin and \bench{} publicly available. 

---
# Guarding with Judgment: Learning to Intervene for Multimodal Safety 

**Authors**: Jipei He, Cheng Cheng, Liuguo Wang, Zihan Wen, Changlin Yu, Wentao Song, Suyun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.28849)  

**Abstract**: When visual context makes an ordinary request unsafe to answer as asked, a blanket refusal may leave the user's need unresolved. Choosing how to help requires distinguishing an established hazard from a condition that still needs verification, a distinction that binary safety labels leave unspecified. We formulate this decision through four actions: allow, warn, review, and refuse. To ground these decisions in verifiable evidence, we build a corpus of 20,126 training audits and the JudiBench benchmark, connecting each action to diagnostic judgments, source evidence, and controlled context changes. Our guard, JudiAct, separately states what the source establishes, what condition governs the help, and how the rules apply. It then checks these three brief review items against the original inputs to predict diagnostic labels and a handling action. Supervised training learns individual judgments while comparing label assignments across related cases. A second stage swaps one review item at a time, using the resulting audit feedback to refine review generation. On JudiBench, JudiAct improves request and answer Macro-F1 by more than 28 percentage points over its 4B backbone. Across nine public benchmarks, it outperforms state-of-the-art guards on the mean delivered-answer score in both pre-generation and post-generation settings, cutting MM-SafetyBench attack success from 40.78% to 25.30%. 

---
# EA-Bench: Measuring information and action efficiency under imperfect instructions 

**Authors**: Xu Wan, Wenyue Xu, Shenjie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.28846)  

**Abstract**: Personal agents help users complete everyday work across digital applications. Many standard benchmarks assume that users provide sufficient information to solve a task. However, agents often encounter incomplete, distracting, or inconsistent information in practice. To systematically study agent inefficiency under such imperfect information, we introduce Efficient Action (EA)-Bench, which evaluates both tool calls and user interactions in work-related tasks. To our knowledge, it is the first benchmark to jointly evaluate tool-call efficiency and user-interaction efficiency under controlled missing, redundant, and conflicting information. We construct 63 base-task instances adapted from existing work-related benchmarks. To guide perturbation design and quantify task requirements, we represent workflows as directed acyclic graphs (DAGs) and examine node-level dependencies to identify where information can be withheld, contradicted, or supplemented with distractors. We use a two-stage validation process combining AI-assisted checks with expert screening to select 232 conditions from 792 initial candidates. We organize these tasks into Easy and Hard tracks by task complexity. Experiments show that imperfect information can reduce accuracy and increase tool-use and user-interaction costs. More concerning, several models obtain apparently correct outcomes by guessing user-specific values without confirmation, revealing potential execution unreliability in current models under imperfect user information. 

---
# Does a Frontier Large Language Model Know What It Can Solve? A 100-Conjecture Calibration Study 

**Authors**: Ayush Khaitan, Liam Fowl, Tomàs Ortega, Alex Kontorovich, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2611.28827)  

**Abstract**: How does confidence correlate with actual performance in large language models, when attacking open research-level math problems? We address this question by studying the performance of ChatGPT 5.6 Sol Ultra on a set of 100 open conjectures in geometric analysis. By construction, the model first selected open problems in geometric analysis that it assessed that it had at least a 50\% chance of proving or disproving, and then assigned a probability of success to each question. 13 questions were then excluded after an audit, and the model was asked to resolve the remaining 87 problems. Its mean stated resolution probability was $0.617$, versus an observed rate of $0.103$ (Brier score $0.351$, ECE $0.513$), thereby demonstrating that the model was systematically over-confident in its self-assessment. However, the scores retained ranking signal (AUROC $0.714$). The model was highly overconfident in probability magnitude but did more often than not assign a high probability of success to the problems that it did resolve, than the problems that it did not. Hence, its probabilities were numerically mis-calibrated but directionally correct. All proofs have been formalized in Lean, thereby verifying correctness, and have been run through a semantic equivalence pipeline to ensure semantic equivalence between the formalized and informal statements. These are exploratory campaign outcomes, not fixed-budget success probabilities. The certificates prove formal statements; for a couple of questions, some geometric inputs are assumed in the lean proofs. 

---
# Improving semantic equivalence in research-level math formalization 

**Authors**: Ayush Khaitan, Liam Fowl, Tomàs Ortega, Alex Kontorovich, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2611.28819)  

**Abstract**: Formalizing mathematics in Lean is never a word-for-word translation of a paper: a Lean formalization changes definitions, strengthens hypotheses, splits arguments, and adds lemmas the paper never mentions. It stands to reason that a model that is good at autoformalization must be a good predictor of how a natural language source must differ from its faithful Lean formalization. With that in mind, we collect 4{,}076 audited examples of such changes from 832 Lean projects that formalize research-level mathematics, and use them to ask a practical question: how well can a model \emph{write} Lean inside a real research-level project that may contain tens of thousands of lines of code, and does training on discrepancies between natural language sources and their faithful Lean formalizations help? Our training set consists of 3{,}431 tasks from 141 repositories, and the test set consists of 645 tasks from the remaining 27 repositories. Training on in-project Lean nearly doubles complete proofs ($11\% \to 20\%$) and raises the share of statements in gold agreement with the project's own formalization by nearly a third ($30\%$ to $39\%$). The gain comes from the pairs themselves: an adapter trained only to predict how the Lean will differ, or a model told the difference in advance, barely helps. Training on where formalizations depart from their sources is thus an effective way of improving semantic equivalence in research-level formalization. 

---
# When do stored facts become available in looped transformers? 

**Authors**: Fabio Valerio Massoli, Arnau Padrés Masdemont, Jordi Ros Giralt, Arash Behboodi  

**Link**: [PDF](https://arxiv.org/pdf/2611.28808)  

**Abstract**: Looped transformers do not acquire new knowledge during a forward pass. Prior work shows that they store as many bits in their weights as non-looped models. It remains unclear how recurrence makes this knowledge available during inference. We study this question using synthetic factual recall and multi-hop reasoning, measuring depth-resolved readable information in bits with native readouts and tuned lenses. Under final-loop supervision, single-hop facts become readable progressively: heavier memorization loads delay retrieval despite comparable final readable information. Untied baselines of equal unrolled depth concentrate retrieval near the end of their stacks, whereas looped models expose facts earlier. This delay is not intrinsic to the tested recall tasks: training with supervision and depth-specific readouts at every loop makes near-final information readable after the first loop in all tested configurations. Representational analysis shows value-related structure emerging within states organized by the query key. We then study multi-hop chains whose facts are stored in the weights rather than given in the prompt. Intermediate entities become decodable in chain order, and aligned transitions show next-entity structure developing while current-entity structure remains dominant. Interventions in a released six-recurrence model establish a causal role for these representations. Counterfactual replacements of a probe-defined entity component at a single prompt position redirect downstream bridge entities and the final answer. Conversely, replacements from queries that reach the same entity through different preceding relations, preserving the remaining relation suffix, retain the original answer in over 99.5\% of tested cases. These results distinguish a training-dependent schedule for factual recall from the causal role of retrieved-entity representations in subsequent composition. 

---
# DiscernBench: Evaluating Cooperative Discernment of LLM Agents in Institutional Settings 

**Authors**: Disha Sheshanarayana, Tanishka Magar  

**Link**: [PDF](https://arxiv.org/pdf/2611.28787)  

**Abstract**: The deployment of large language models in operational settings demands evaluation of judgment under institutional constraint. We introduce DiscernBench, a benchmark for evaluating cooperative discernment: the ability of an agent to determine whether to comply with, refuse, or escalate a request in an institutional setting. DiscernBench comprises scenarios derived from real-world incident reports across domains such as aviation, law enforcement, and emergency management. We evaluate eight LLMs across three decision-making configurations and analyze both final decisions and the intermediate judgments produced during multi-agent deliberation. Under disagreement, Final Responders exhibit model- and role-dependent aggregation patterns in which a lone specialist recommendation can disproportionately influence the collective decision, a phenomenon we term Minority-Signal Override. Role-specialized multi-agent deliberation raises average over-escalation from 3.79% under direct single-agent judgment to 28.27%, with the direction and magnitude of minority-signal influence varying substantially across models and specialist roles. These results reveal distinct collective decision behaviors that are not captured by final-answer accuracy alone. Trustworthy multi-agent systems therefore require evaluating not only individual specialist reasoning, but also how conflicting judgments are transformed into collective decisions. Code and data are available at this https URL. 

---
# Beyond Single-Turn Privacy: Training-Data Leakage in Conversational RAG 

**Authors**: Syed Khaderi, Karima Makhlouf, Sami Zhioua  

**Link**: [PDF](https://arxiv.org/pdf/2611.28748)  

**Abstract**: Retrieval-augmented generation (RAG) is known to reduce training-data leakage in single-turn interactions, but whether this protection persists over multi-turn conversations is unclear. We introduce CRELT (Conversational RAG Extraction Leakage Trajectory), a reusable framework that tracks, turn by turn, the real contact details from the training data that a RAG system discloses. CRELT separates retrieval, conversation history, and attack strategy through a full-factorial design, uses a retrieval corpus verified to contain none of the target PII, and counts an item as leaked only when it is absent from the model input at the turn it is produced. In our main experiment, this criterion excludes 29-53% of raw matches that were already present in the input. Across GPT-J-6B, Pythia-6.9B, and Pythia-12B, targeting real Enron contact details, retrieval reduces disclosure under both full and cleared history, consistent with prior single-turn results. A length-matched control with random passages reproduces 91-99.5% of this reduction under cleared history and 47-78% under full history, so much of it comes from the added context rather than from what is retrieved. With retrieval enabled, however, retaining full conversation history increases disclosure by a factor of 3.6-4.9, while retaining only one previous turn already restores disclosure to approximately the full-history level. With retrieval, clearing the history reduces disclosure by 72-79%, but does not eliminate it. These results show that single-turn privacy audits can substantially underestimate training-data leakage in conversational RAG systems and that conversation-history policy is a critical dimension of privacy evaluation. 

---
# Can Your AI Twin Meet People for You? It Can Hold a Conversation. But What Does It Need to Judge Like You? 

**Authors**: Wanghao Ye, Sihan Chen, Bowei Tian, Yiting Wang, Shwai He, Guoheng Sun, Ziyi Wang, Ziyao Wang, Yexiao He, Zheyu Shen, Meng Liu, Yuning Zhang, Meng Feng, Yifei Dong, Yanhong Qian, Yang Wang, Siyuan Peng, Yilong Dai, Zhenle Duan, Joshua Liu, Lang Xiong, Aarosh Das, Zhaoyi Liu, Daeun Jung, Aditya Sarkar, Parastoo Pilevar, Xiaochuan Gong, Yancheng Zhu, Yingying Fan, Qiong Wang, Hanzhang Qin, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.28692)  

**Abstract**: Should your AI twin go meet people on your behalf? We built LLM twins of 710 real people from their other conversations in the CANDOR corpus, sent them to meet each other in 5,680 simulated first conversations, and checked every rehearsal against what the real people said after meeting for real (355 held-out, participant-disjoint pairs, against baselines that include each person's rating history). The rehearsals work: a twin's conversation predicts real liking ($\rho$=0.277), carries information that questionnaires and rating history do not have (partial $\rho$=0.16 [0.05, 0.26]), and recovers most of what the same reader gets from the real conversation (0.321). Swap controls show that this signal travels through the conversation, not through the persona card. What, then, does a twin need to judge like you? Not your voice: making twins talk more like their owners, with person-specific activation steering and a LoRA trained on their own speech, did not help in a pre-specified test and lowered accuracy in an exploratory second round (-0.183 [-0.294, -0.066]). Nor simply your stated ratings: a persona-prompted twin that holds the conversation does not yet carry its owner's judgment, and when told its owner's past ratings it anchors on them instead of weighing them. Learning helps: a small appraisal head trained on other real conversations, which combines the twin's post-call answers with its owner's rating history, lifts rehearsal-based prediction above the history baseline out of fold (+0.059 [0.001, 0.115], against +0.022 without the twin's answers; pre-specified primary contrast +0.056 [-0.001, 0.111]). We propose an attribution protocol for twin-based social simulation and identify learned appraisal as the component twins need next. 

---
# DeCo: Decoupled Traffic State Modeling with Cross-Variable Coordination 

**Authors**: Du Yin, Lindong Xie, Xiachong Lin, Ruiyi Yang, Yifan Duan, Hao Xue, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2611.28685)  

**Abstract**: Accurate forecasts of traffic flow, occupancy, and speed are essential for congestion management and travel planning. These variables exhibit distinct spatiotemporal patterns, and their coupled dynamics characterize the evolution of traffic conditions, such as the onset and dissipation of congestion. Existing methods often either neglect this coupling or struggle to capture the distinct dynamics of all three variables within a unified model. In this paper, we propose DeCo, a decoupled traffic state modeling framework with cross-variable coordination for forecasting flow, occupancy, and speed. Specifically, DeCo first trains a dedicated forecaster for each variable to specialize in its unique spatiotemporal dynamics. It then estimates a residual correction for each forecast by jointly leveraging the predicted trajectories and recent observations of all three variables, thereby exploiting cross-variable coupling and mitigating systematic prediction errors. We extend the flow-only PeMS03 and PeMS07 benchmarks to all three variables on their original sensors and release the aligned data. Experiments on PeMS03, PeMS04, PeMS07, and PeMS08 show that DeCo achieves the lowest mean absolute error for every variable on every dataset, up to 4.9% below RAGC, the strongest baseline. Code is available at this https URL. 

---
# Improving the Improver: Semantic PID Control for Medical Agent Self-Improvement 

**Authors**: Dexia Chen, YiLong Hu, Wenqi Zhong, Guangyu Guo, Tony C. W. Mok, Ling Zhang, Zhongwei Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28680)  

**Abstract**: High-stakes vertical medical tasks, such as postoperative pathological staging and long-term survival prediction, demand multi-step clinical reasoning over heterogeneous evidence under weak supervision and high outcome uncertainty. While we first orchestrate CliAgent, a domain-tailored clinical agent workflow that systematically grounds predictions in multi-modal patient evidence, manually designed medical agents remain constrained by static, brittle rules. Moreover, existing recursive self-improvement (RSI) methods suffer from feedback ambiguity and severe development-set overfitting when evolving agents in high-noise medical domains. Focusing on improving the improver, we propose Semantic PID-RSI, an evidence-guided control-theoretic framework for medical agent self-improvement. Inspired by PID control, our improver evaluates candidate modification batches via three diagnostic roles: P isolates current clinical failure modes, I synthesizes longitudinal evidence from past trials, and D estimates response dynamics and regression risks (D-pre/D-post). Selected plans are compiled into behavior contracts for staged probing and independent adoption judging. Instantiated and validated on challenging oncology tasks (pT/pN staging and survival prognosis), evolving CliAgent with Semantic PID-RSI on a 3B-activated open-weight model achieves substantial gains (up to +6.27 C-index points), outperforming overfitting-prone RSI baselines and approaching the performance of frontier proprietary models (e.g., GPT-6 series) equipped with CliAgent. 

---
# Persuasion from Within: Model-Based Insider Attacks in Multi-Agent LLM Debate and Consensus 

**Authors**: Bangyan Ju, Xiaolin Sun, Yibin Hu, Zizhan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.28660)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multi-agent systems for collective decision-making, such as multi-agent debate and consensus games, in which agents exchange messages and update their answers over multiple rounds and a resolution mechanism aggregates them into a final outcome. Most multi-agent LLM frameworks assume that every participant shares the system objective. In practice, however, a malicious agent can join as a legitimate member of the group and persuade the other agents toward a hidden adversarial goal. We formalize such attacks as a sequential decision-making problem in which the insider seeks to degrade the utility of the collective outcome. We propose a model-based insider attack that learns to persuade a group of agents: the insider first learns a world model of the benign agents from offline trajectories and then uses it both to train a persuasion policy and to plan its messages. In two case studies involving a numerical consensus game and a multi-agent debate, the learned insider is the most effective and the stealthiest attack we evaluate, outperforming both prompt-based and trained LLM persuaders from prior work. The attack also transfers to other target models, larger groups, and unseen datasets, and remains effective under existing Byzantine-resilience defenses. The learned insider also enables a new defense: adversarially training benign agents on the attack data largely removes both our attack and the strongest prior persuasion attack. These results show that combining world models with reinforcement learning and planning is an effective route to both attacking and hardening language-based multi-agent systems. 

---
# Unlearning Model Identity: Ablating LoRA Weights from Question-Based Identity Tuning 

**Authors**: Ming Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2611.28650)  

**Abstract**: Large Language Models are trained on diverse corpora containing outputs from proprietary teacher models and provider-curated sources. Models subsequently inherit embedded self-identity traits, persistently asserting specific proprietary names. Service providers of open-weight models frequently require the removal of model identity for downstream applications, such as custom interfaces, without exposing provider affiliations to end-users. Effective identity modification is therefore critical for customized model alignment and deployment. This study proposes a method to localize parameter-level representations of self-identity within low-rank adaptation modules to perform targeted weight ablation. By generating probing questions designed to elicit self-naming outputs and fine-tuning the base model on these responses, the approach identifies and ablates high-magnitude weights within the parameter set. Experimental results demonstrate that distilled identity traits reside within a localized subset of low-rank weights, achieving a complete removal rate in Qwen3.5-2B and a 10% rate in Gemma4-E2B while preserving general task capabilities. These findings confirm that ablating top-tier adaptation weights suppresses distilled self-identification without compromising broader model performance. 

---
# Mitigating Backdoors in Masked Diffusion Language Models 

**Authors**: Chengzhong Wang, Lang Lin, Shengfang Zhai, Chenru Wang, Daniel Yiming Cao, Sheng-Yen Chou, Jiaheng Zhang, Pin-Yu Chen, Shengwei An  

**Link**: [PDF](https://arxiv.org/pdf/2611.28640)  

**Abstract**: Masked diffusion language models (MDLMs) have emerged as a promising alternative to autoregressive models because they can predict multiple tokens in parallel rather than generating tokens sequentially. However, recent studies have shown that MDLMs are vulnerable to backdoor attacks: when an input contains an attacker-chosen trigger, a backdoored model generates a targeted output, such as malicious code, while otherwise behaving normally. Despite this threat, backdoor defenses for MDLMs remain largely underexplored. We present UNMASK, the first defense framework for MDLMs that detects trigger-containing inputs, localizes the trigger tokens, and removes the backdoor from the compromised model. UNMASK exploits two fundamental properties of backdoor triggers, specificity and dominance, to assign a suspiciousness score to each input token. By leveraging two distinctive capabilities of MDLMs, parallel token prediction and token reconstruction at arbitrary positions, UNMASK computes these scores both effectively and efficiently. We further provide a theoretical analysis supporting the design of UNMASK. Evaluations across 40 settings spanning two attacks and four models show that UNMASK detects 99.3% of trigger-containing inputs at a 5% false-positive rate, substantially outperforming the strongest adapted baseline, which achieves 54.8%. UNMASK also achieves over 99% trigger-localization accuracy and reduces the average attack success rate from 95% to 1%, while largely preserving the model's utility on clean inputs. 

---
# Distilling Visual Agents from Limited Interaction Experience 

**Authors**: Siyu Zhou, Jing Jiang, Chengqi Zhang, Guodong Long  

**Link**: [PDF](https://arxiv.org/pdf/2611.28618)  

**Abstract**: How much interaction experience is needed to distill a capable visual agent? We study learning from few tasks and recorded interaction histories. Our framework selects representative trajectories using skill-context coverage and ordered visual-decision similarity. On-policy distillation (OPD) learns through fresh interaction on selected tasks or fresh student responses at fixed histories. Across Navigation, Sokoban, and spatial mapping, even one online training task supports transfer to other tasks. Four fixed trajectories raise a 4B student's success from 49.17% to 82.08% on standard Navigation and from 60.16% to 85.55% on Sokoban, recovering over 90% of the measured gap to a 27B teacher on these panels. On an independent Navigation panel, replay improves success from 45.83% to 76.67%, retaining 91.36% of the same-task online gain without new interaction during optimization. Training repetitions, matched supervised fine-tuning (SFT), and a 9B extension further support the value of compact experience. 

---
# VeriLadder: A Multi-Agent Specification Refinement System for Tool-Verified Verilog Datasets 

**Authors**: Saeid Rajabi, Chengmo Yang, Satwik Patnaik  

**Link**: [PDF](https://arxiv.org/pdf/2611.28617)  

**Abstract**: Training data is a bottleneck in AI, especially for large language models (LLMs) that write Verilog, because register-transfer-level (RTL) code is rarely published and existing datasets mostly check code for syntax, not instruction correctness. This paper presents VeriLadder, a five-level Verilog dataset, and the multi-agent framework that builds it, in which LLM agents write and revise specifications, and EDA tools (a synthesizer and a formal equivalence checker) decide what is accepted. Its levels follow an engineer's checks. Level 1 labels designs from two large Verilog corpora, namely RTLCoder and PyraNet, with synthesis results, failure causes, critical warnings and post-synthesis metrics. 28% of RTLCoder designs fail synthesis and are kept as negatives. Level 2 grades each instruction on four specification qualities, and low grades predict failed rebuilds. Level 3 rewrites each instruction into a refined specification and accepts it only if an LLM from another model family, reading only that specification, writes a circuit formally equivalent to the original. This raises the share of rebuildable designs from 26.1% to 78.2%, keeping every failed specification with its counterexample. Levels 4 and 5 add verified paraphrases and one-fact variants, hard negatives that differ only in one behavior from the verified design. VeriLadder labels the synthesizability of 716,627 designs and, from RTLCoder, provides 11,247 refined specifications, 32,393 paraphrases and 8,397 one-fact variants. Fine-tuning Qwen2.5-Coder-7B-Instruct on VeriLadder raises functional pass@1 from 31.7% to 50.0% on VerilogEval v2 and from 35.0% to 45.2% on RTLLM v2. The dataset and fine-tuned models are publicly available. 

---
# Internalized Experience: How Training Data Shapes Memory Formation in LLMs 

**Authors**: Xianxuan Long, Xinnan Dai, Kai Guo, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28613)  

**Abstract**: Parametric memory enables large language models (LLMs) to retain information from past experience and reuse it in subsequent tasks. Yet how training data supports the internalization of reusable task rules remains unclear. We introduce FSMem (Finite-State Machine Memory), a state-transition framework for constructing training data and studying how data content and organization shape memory formation. We evaluate rule reuse across synthetic worlds, classical planning, and code execution through valid-action identification and execution-outcome prediction, complemented by next-operation prediction in code. Our experiments reveal several systematic learning patterns. Under matched training conditions, even unchanged task rules become harder to internalize when learned alongside a larger variety of action types; by contrast, scaling the training data to provide broader state-action coverage generally improves rule learning. We further find that training at a single execution depth produces depth-specialized behavior, with models performing best when training and test executions have similar lengths. Motivated by this finding, we construct cumulative corpora that combine short and long trajectories, which can outperform long-trajectory-only training while using fewer training tokens. Together, these results demonstrate that the same task rules can be internalized differently depending on how execution experience is constructed, and that understanding these learning patterns can guide the design of more effective training data for parametric memory. 

---
# Agent Money Benchmark: Can AI Agents Be Trusted to Spend Money? 

**Authors**: Zhe Chen, Emma Hsieh, Vicky Fu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28610)  

**Abstract**: Giving an AI agent a wallet lets it buy data and services on a client's behalf. Trust then depends on how it spends money and what that spending achieves. We introduce Agent Money Benchmark (AMB) to test whether agents spend a client's money well, starting with purchases of information for financial decisions. We link seller quotes and payments to final answers, evaluating 12 models on 80 payment tasks with simulated sellers and 76 research questions using live data. Purchases settled in USDC on Solana mainnet via x402, a protocol for paying per web request. Three of the 12 models passed 79 of 80 payment tasks without payment or budget errors. Nine fell for a seller's trap at least once; repeat charges and above-list quotes accounted for 79% of scam events. We estimate waste from data and inference costs, counting failed payment tasks in full and research answers in proportion to missed points. With spending split equally between tracks, estimated waste ranged from \$21.0 per \$100 for Claude Fable 5.2 to \$70.8 for GPT-6 Luna. 

---
# EvoViM: Recursively Self-Evolving Visual Agents for Exploration with Deferred Queries 

**Authors**: Xinyu Cao, Tianjie Ju, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28574)  

**Abstract**: Multimodal large language model (MLLM) agents have shown promising capabilities in task-oriented environments. However, they still struggle when queries are deferred until after exploration, requiring them to retain visual evidence before knowing which details will matter. We introduce EvoViM, a recursively self-evolving framework in which a frozen multimodal model turns its own interaction experience into one visual program for memory and skills. The program retains images and computes directly on pixels, and its operations decide how this evidence is used in each task. During memory evolution, the same model traces failures in its visual trajectories to the operations that lost or misused evidence and revises them. It also rewrites the notes and analysis tools that guide later revisions, which then build on earlier findings and skip changes that have already failed. We evaluate EvoViM with three multimodal models on six tasks covering visual matching, spatial recall, and sequential decision-making, including three tasks with deferred queries. Across the six tasks, EvoViM improves the average score over AutoMem, the strongest baseline in the main comparison, by 31.3-47.6 points on held-out episodes. Programs written once from the initial environment instructions perform substantially worse, highlighting the value of interaction experience in shaping how visual evidence is retained and used. Code and dataset are available at this https URL. 

---
# Beyond Unit Tests: Path-Aware Rubric-Guided LLM Autograding of Notebooks 

**Authors**: Salman Hajizada, Mohammed Yusuf Ansari, Gianni Di Caro  

**Link**: [PDF](https://arxiv.org/pdf/2611.28565)  

**Abstract**: Jupyter notebooks support open-ended, multi-step assessments in which students build one logical flow across interdependent steps, combining code, written analysis, and generated outputs such as plots. These reflect authentic scientific practice but are hard to grade consistently, since students may make different yet equally valid choices at each step, so a later answer can only be judged in light of the choices a student made earlier. We present an LLM-based autograder for such flow-structured notebook questions. It generates a rubric from a reference solution, then applies a softening pass that relaxes criteria tied to the reference so legitimate alternative paths are not penalized. Each question group is graded in a single model call, so later answers are judged against the student's own earlier choices, using question-type tags and schema-constrained, low-temperature outputs. Evidence is scored multimodally: code, text, numerical outputs, and figures from code cells. We deployed the tool in a machine-learning course, integrated with Gradescope, with a pre-submission format check and prompt-injection defenses, and evaluate it both in that live course and against archived submissions graded by staff in earlier terms. We report agreement, cross-model variance, cost, student perceptions, and injection resilience. With low-temperature models, agreement with human scores is often strong, including on the open-ended items the design specifically targets, while run-to-run stability depends sharply on the model's temperature support. 

---
# From atomic events to material degradation across metre and year scales 

**Authors**: Kun Li, Tian Luo, Haozhi Han, Ruge Zhang, Yifeng Chen, Xinfu He, Yunquan Zhang, Ting Cao, Yunxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28559)  

**Abstract**: Material degradation emerges from microscopic events whose consequences accumulate over years, yet mechanism-resolved simulations remain confined below engineering timescales. The challenge is not just the sheer number of events: atomistic evolution is dominated by recurrent local motion, with only a tiny fraction driving lasting structural change. This distinction is not encoded in an instantaneous atomic configuration. Locally indistinguishable states can exhibit different event competition, escape times, and long-horizon outcomes, revealing missing kinetic context as a critical information bottleneck. This context spans space and time: the future consequence of a local transition depends on its surrounding defect environment and the kinetic history leading to its current state. To solve this, AtomWorld reconstructs a predictive kinetic state from multiscale structural observations and trajectory history, distinguishing transient fluctuations from microscopic pathways that drive persistent structural evolution. Learning operates only over transitions admitted by the underlying kinetic Monte Carlo (KMC) process, preserving event legality, state evolution, and physical time. Across BCC Fe-Cu precipitation, FCC Ni-Cr-Al ordering, and HCP Zr-H interstitial transport, restoring kinetic context consistently advances persistent evolution while maintaining physical fidelity and transferring to unseen conditions. In irradiated reactor-pressure-vessel steel, this approach achieves up to 420-fold greater persistent evolution under matched event budgets. Combined with scalable execution, it successfully extends mechanism-resolved trajectories to service-year timescales across metre-scale heterogeneous domains. 

---
# X-AQA: Unifying Action Quality Assessment with Multimodal Large Language Models and Multi-Domain Learning 

**Authors**: Kaili Zheng, Wei Ding, Yuchen Ding, Kaiwen Wang, Xun Zhu, Chenyi Guo, Yudong Zhang, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28548)  

**Abstract**: Action Quality Assessment (AQA) aims to evaluate the quality of human actions, yet existing methods are predominantly developed and evaluated on individual datasets with different task formulations and assessment criteria. We introduce X-AQA, the first unified training and evaluation framework for general-purpose AQA with multimodal large language models (MLLMs). We construct X-AQA-40K, a training corpus comprising 40,163 samples from 22 data sources across four domains: competitive sports, fitness exercises, rehabilitation, and general skills. We convert heterogeneous annotations into a unified question-answering format, incorporating dataset-specific annotation guidelines and official scoring rules when available. Using this corpus, we fine-tune Qwen3.5-9B to develop a single MLLM capable of handling diverse AQA tasks. We further establish X-AQA-Bench, which evaluates both in-distribution (ID) performance on datasets represented during training and out-of-distribution (OOD) generalization on four held-out datasets. The resulting model outperforms all evaluated open-source MLLMs on X-AQA-Bench and achieves competitive performance relative to proprietary MLLMs. Ablation studies further demonstrate that multi-domain joint training and explicitly providing assessment criteria in prompts both improve OOD generalization. X-AQA establishes a unified foundation for advancing general-purpose AQA across diverse real-world domains. Data and models are released at \hyperlink{this https URL}{here}. 

---
# Walks2Maps: Learning Indoor Geometry from Crowdsourced Wi-Fi RTT and Motion Trajectories 

**Authors**: Enguang Fan, Binh Minh Tran, Fulin Wang, Jiaxi Li, Klara Nahrstedt  

**Link**: [PDF](https://arxiv.org/pdf/2611.28532)  

**Abstract**: Everyday smartphone walks offer a scalable source of indoor geometry, but their motion traces drift and their radio measurements are distorted by multipath. We present Walks2Maps, a physics-guided framework that reconstructs a shared metric map from Wi-Fi round-trip-time (RTT) and motion observations, without a floor plan, surveyed access-point (AP) coordinates, or position labels. Walks2Maps jointly estimates pedestrian trajectories, AP locations, and per-trajectory stride scales, then recovers the pathway layout from the reconstructed paths. Shared AP observations connect walks recorded in different local frames, while metric ranges constrain their relative geometry and scale. An asymmetric range loss accommodates multipath-induced excess delay. On 22 phone trajectories collected along 11 routes on an office floor with 30 APs, Walks2Maps achieves median AP and trajectory errors of 0.87 m and 0.89 m, respectively, and a walked-area F1 score of 0.985. Ablations show that radio similarity can recognize places while still producing a distorted map; retaining the metric information in RTT resolves this ambiguity. Asymmetric modeling further reduces AP error by 54% relative to a symmetric loss. These results demonstrate how radio and motion measurements can jointly reveal indoor geometry without a dedicated mapping survey. We will release the dataset and toolkit. 

---
# ESM-Bench: A Benchmark for Evaluating Whether AI Agents Understand Earth System Model Physics and Code 

**Authors**: Koutian Wu, Yuxuan Cao, Gengchen Mai, Zeping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28477)  

**Abstract**: AI coding agents that modify Earth System Model (ESM) code must preserve both software interfaces and the physical assumptions encoded in them. We introduce ESM-Bench, a benchmark of 60 developer-commit tasks from 6 repositories: Noah-MP, CLM5, SUMMA, ParFlow, MOM6, and VIC. The v5 corpus contains 17 physics-based bug fixes, 16 process-representation modifications, 13 parameterization-scheme selections, and 14 parameter-optimization tasks. Each task includes a scientific objective, parent-commit source files, a reference diff, and three stored prompt variants. An artifact audit confirms that all 60 reference diffs apply to their supplied starting files. We define an eight-item patch-review rubric and audit five archived patch fixtures against the implemented local checkers. None of the five fixture diffs applies to its supplied starting files, yet three text-pattern checkers return a pass. A ParFlow expression replay disagrees with the reference slopes, and a MOM6 pattern check detects an omitted update. Inspection also identifies a retained VIC reference assignment that overwrites the intended vegetation-specific albedo. These findings expose limits of reference patches and local proxies as correctness evidence. The release provides a reproducible corpus and diagnostic tools; complete expert validation, comparative agent performance, and full-model execution remain unmeasured on v5. 

---
# Human-in-the-Loop Runbook Improvement with Agentic Support Automation 

**Authors**: Rocker D'Antonio, Harry Xie  

**Link**: [PDF](https://arxiv.org/pdf/2611.28441)  

**Abstract**: Operational support is an important component of production software services. Support requests are often emergent and can come in many forms such as customer escalations or unplanned service interruption. Engineers across organizations have been successful in implementing automation to help streamline support processes but many solutions remain in the hands of human operators. A successful strategy to improve operator response times is to maintain up-to-date runbooks which are documents enumerating triage steps and remediation procedures in the face of identified incidents. This paper is an investigation into an operational system which incorporates Agentic AI into a runbook based incident resolution pattern. The bot uses information in the runbook corpus along with some system-specific tooling to guide the manual operator through the mitigation process then evaluate the outcome and suggest edits to the runbook where it found gaps and in its ability to triage the situation. By using the runbook as a medium to bridge humans and automation, the process maintains explainability and can be decoupled from the agent system. This paper demonstrates a collaboration between human operators and Agentic AI with results from an 17 week study involving 232 production incidents. We conclude with lessons on integrating LLMs into DevOps workflows, related work in AI-assisted operations, and guidance for reproducibility. 

---
# Diversity over Scale: Recursive Self-Improvement with Ensembles of Small Open-Weight Models 

**Authors**: Navapat Nananukul, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2611.28420)  

**Abstract**: The advancement of large language models (LLMs) has significantly enhanced the capabilities of agents across various tasks. Recursive self-improvement (RSI) loops, in which a language model repeatedly rewrites a program and builds on its previous version, now perform well in machine-learning tasks, but they rely on frontier models and hundreds of calls per run. Small open-weight models are far cheaper, yet when used alone, their programs often crash and their improvements stall. We propose DiVE (diverse models evolve), an ensemble method for RSI that aims to improve creativity and diversity of generated solutions. DiVE uses three components: a shared memory, a failure log, and best-idea-wins rounds. This allows models to write candidate solutions from the same context and retain the best idea at each iteration. On 22 MLE-bench Lite competitions, an ensemble of five 8-12B open-weight models has 25.4% less test error than its average member and 16.7% less than Llama 3.3 70B, and comes within 7.1% of Qwen 3.5 122B-A10B at 75% of its cost. Diversity among small models can thus stand in for part of the scale that RSI relies on. 

---
# The Effect of Security-Focused Prompts on Reported Findings: A Controlled Experiment on 400 AI-Generated Web Applications 

**Authors**: Darko Andročec  

**Link**: [PDF](https://arxiv.org/pdf/2611.28408)  

**Abstract**: Agentic coding tools turn a specification into a running web application, so their defaults for authentication, sessions, input validation, and dependencies become its security posture. This study asks whether appending a fixed block of OWASP-aligned security requirements to the prompt changes what such tools produce, by how much, and with what side effects. Each of 100 frozen specifications was realized four times: two agentic tools (Claude Code, Codex CLI) crossed with two prompts sharing a byte-identical functional core, without (A) and with (B) the security block. A fixed static and dynamic pipeline scanned the 400 applications, yielding 6,721 triaged findings, analyzed with count and logistic mixed models and with a dependency diff. The block cut findings per application by a third (incidence-rate ratio 0.65, 95% CI 0.60-0.71), equally through both tools, and roughly doubled the declared dependency surface, so variant-B applications carried more high-severity dependency advisories. Adjusted for that surface the advisory rate did not rise (flat per application, 18% lower in aggregate), while high-severity findings in first-party source fell by nearly half. Security-focused prompting is therefore an inexpensive and portable improvement, but it enlarges the dependency surface that has to be audited against it, an effect of library selection rather than of the security requirements themselves. 

---
# CausalNTP: A World-Model Recipe for Causal Next-Token Prediction under Noisy Training-data 

**Authors**: Devansh Arpit  

**Link**: [PDF](https://arxiv.org/pdf/2611.28369)  

**Abstract**: This paper shows that transformer networks trained with next-token prediction (NTP) and gradient descent do not learn to make predictions using only causally-relevant tokens in the left context, irrespective of the causal ordering of tokens during training, and identifies the learning deficiencies responsible. We introduce a causal world-model setup, for which, we characterize the Bayes-optimal predictor under the causal view of the data. We then prove that, a function that distills its predictions to non-causal views of data, when realized under Occam's Razor, recovers the true causal predictor. Based on this analysis we propose CausalNTP, a practical training algorithm that approximates this theoretical recipe and trains transformers to depend only the causally relevant tokens in left-context. Empirically in the world-model setup, CausalNTP achieves out-of-distribution generalization that vanilla NTP cannot due to its theoretical limits, regardless of data and model scale. We show similar gains and behavior in the natural language question-answering setting as a proof-of-concept. 

---
# Schedule, Rank, Profile: Continuous Evaluation of LLM Agents in Games 

**Authors**: Martyna Plomecka, Yao Yan, DJ Sterling, Jon Lipovetz, John Schultz, Michael Aaron, Chiamaka Chukwuka, Will Cukierski, Christopher D'Mello, Bovard Doerschuk-Tiberi, Robert Fraser, Ian Gemp, Yuting Han, Addison Howard, Oran Kelly, Siqi Liu, Jeff Moser, Meghan O'Connell, Jun Peng, Laurel Prince, Meg Risdal, Jonathan Ward, Domino Weir, Marc Lanctot  

**Link**: [PDF](https://arxiv.org/pdf/2611.28368)  

**Abstract**: Games are a robust test for large language model agents: they require sequential decisions, adversarial or cooperative interaction, state tracking, reasoning under uncertainty, and long-horizon planning. However, a one-time game benchmark can go stale as models, prompts, environment implementations, and evaluation policies change, and refreshing it is costly: full round-robin evaluation scales poorly in the number of model pairs, and the resulting outcomes live on incompatible per-game scales. We present a framework for continuous evaluation of LLM agents in game settings, organized around three coupled components. First, an adaptive scheduler that maps pairwise Bradley-Terry uncertainty to reduced per-pair match targets, concentrating budget on comparisons that remain unresolved. Second, a Unified Leaderboard (ULB) that reduces heterogeneous outcomes to pairwise evidence and fits a game-balanced Bradley-Terry model by minorization-maximization with bootstrap intervals, so that high-throughput games cannot dominate the aggregate ranking. Third, a game skill-profile methodology that scores each game along diverse skill dimensions, making the capability coverage of the suite explicit and turning game selection into a design decision rather than an artifact of availability. In this paper, we specify the research design and outcome reduction of the current implementation and evaluate the framework on a ten-game suite spanning perfect-information board games, imperfect-information games, and social/language settings, reporting LLM capability profiles, evaluation-cost savings across model-refresh cycles, and an empirical Pareto frontier relating model performance to inference cost. As deployed, the scheduler saved 61.7% of episodes; it counted each episode in the two team games as four records, overstating their savings, and the other eight games save 61.9%. 

---
# Engram: Multi-Stage Memory for Agentic Tuning of Heterogeneous Models 

**Authors**: Huanran Li, Nhat X.T. Le, Kevin De Angeli, Yao Xuan, Alireza Vahdatpour, Djordje Gligorijevic  

**Link**: [PDF](https://arxiv.org/pdf/2611.28356)  

**Abstract**: Large language model (LLM) agents can now tune neural networks, yet without cross-session memory they spend costly trials rediscovering known lessons. Reuse is difficult because production models differ in architecture, training setup and inference budget. We present Engram, a memory system that links past trial outcomes with the hypotheses and rationale behind them and assesses whether this evidence applies to the current model. An online agent curates evidence at the decision points where it is needed: Initial Guidance supplies validated lessons at the start, Session Retrieval supplies related past sessions before each batch of hypotheses is generated, and Trial Candidate Review checks each trial candidate against matching past trials. Between sessions, an offline agent distills validated lessons, making the memory self-improving. On two ads models, Engram reaches the 32-trial result of a no-memory control with 47% fewer trials and also ends with a better model on average. 

---
# SMAD: A Semantics-Guided Multi-Agent Framework for Unsupervised Time-Series Anomaly Detection 

**Authors**: Qinghua Liu, Junpeng Wang, Mingzhi Hu, Menghai Pan, Shubham Jain, Yuzhong Chen, Yan Zheng, Yiwei Cai, John Paparrizos  

**Link**: [PDF](https://arxiv.org/pdf/2611.28348)  

**Abstract**: Time-series anomaly detection (TSAD) lacks a one-size-fits-all detector for heterogeneous data and anomaly types. Existing automation is limited by implicit objectives, potentially unreliable historical knowledge, or high computational costs. More fundamentally, it selects how to detect anomalies without explicitly defining what to detect. We propose SMAD, an LLM-based multi-agent framework that translates anomaly intent into executable, testable detection pipelines. Rather than selecting detectors against an implicit target, SMAD establishes a natural-language semantic contract grounded in measurable time-series properties, specifying target event patterns, expected score responses, and validation questions. This contract governs both detector construction and evaluation: LLM agents propose pipelines, while contract-matched synthetic interventions provide label-free evidence from the target series to assess semantic coverage and guide iterative self-refinement. A semantics-aware ensemble integrates validated candidates and attributes each alert to contributing detectors and supporting evidence. On univariate and multivariate TSAD benchmarks, SMAD improves detection performance over the strongest baseline by 11.5% and 9.6%, respectively, while outperforming general-purpose coding agents at 2-5x lower language-model token cost. 

---
# When Tools Fail, Agents Fabricate: Silent Failure and Verification-First Oversight in Tool-Using LLM Agents 

**Authors**: Gün Kaynar, Boshen Yan, Carl Kingsford  

**Link**: [PDF](https://arxiv.org/pdf/2611.28341)  

**Abstract**: Tool-using LLM agents must recover when a tool call fails, by retrying, switching tools, or revising their plan. We study how agents fail when tools fail, using RetryRealistic, a sandboxed yet realistic testbed for controlled fault injection in which agents operate on a real OS filesystem with genuine permission enforcement, SQLite databases, a deterministic local HTTP server, and a constrained shell, across 8 tools, 9 fault types with controlled persistence, and 30 multi-step tasks. Evaluating four agents over 2,100 faulted runs against ground truth, we find that recovery is unreliable (19-68% correct) and frequently silent: 6-26% of runs end in a confident fabrication, a wrong answer asserted as fact. The failure mode is governed more by the fault type than by the model: faults that return partial data drive fabrication (up to 38%), while hard denials drive honest failure. The mechanism is what we call silent ingestion: the agent accepts faulted data as valid on first contact without retrying or flagging the problem, then builds a fabricated answer on top of it. We show that fabrication is reducible: a verification-first Monitor, grounding checks in structured tool outputs and issuing corrective directives rather than hard blocks, raises correct recovery from 45.6% to 56.3% and cuts fabrication from 17.3% to 10.8%. Removing chain-of-thought increases fabrication (+8.3pp), so deliberation guards against silent ingestion. We release RetryRealistic, the fault-injection toolkit, and all annotated traces at this https URL 

---
# Decision models for UAVBench and the viability of edge deployment 

**Authors**: Matthew Michelson, Goran Muric, Adrien Bibal, Steven Minton  

**Link**: [PDF](https://arxiv.org/pdf/2611.28331)  

**Abstract**: Small language models are often proposed for reasoning onboard a drone. However, a good benchmark score does not tell us whether a model will actually run, and run usefully, on an edge computer. In this paper we study three related questions using the UAV decision benchmark, UAVBench. First, are small structured decision models good candidates for these tasks? Second, can an ordinary local machine (a Mac) screen candidate models before we test them on physical devices? Third, what software and resource limits govern deployment? Our main Mac evaluation covers 24 model configurations on the same 2,000 validated multiple-choice questions from UAVBench. These include 20 size-and-precision configurations of 11 generative models, plus four open-weight structured decision models that received no additional UAVBench-specific training. A structured decision model, H2O Lightning 4B, achieved the highest accuracy we observed, scoring 77.65% on the Mac. This shows that these models are well suited to this kind of task. However, when we tried to deploy the decision models, we ran into conversion and runtime limits on the edge devices we tested, which we describe in detail. To test local screening, we ran matched evaluations of two generative models on both the Mac and a Qualcomm RB3 board. The accuracies did not match exactly, but the ranking did. LFM2 2.6B ranked above Qwen2.5 1.5B on both platforms. This supports using a local machine to shortlist candidates. That said, individual answers still differed between platforms, so testing on the actual hardware remained necessary. 

---
# MatHarness: Towards Long-Horizon Open-ended AutoResearch for Computational Materials Science 

**Authors**: Yanlin Fei, Lei Li, Shaolong Chen, Nazhou Liu, Xinmiao Yu, Madalina Ciobanu, Navan Preet Singh, Qingqing Mao, Ritankar Das  

**Link**: [PDF](https://arxiv.org/pdf/2611.28324)  

**Abstract**: Materials science research is a non-linear, long-horizon process. Existing scientific agents, however, largely rely on specific task execution or predefined workflows, rendering them brittle in open-ended settings. To bridge this gap, we present MatHarness, an agent harness designed for open-ended, long-horizon computational materials research spanning the complete lifecycle from initial ideation to manuscript writing. MatHarness couples three mechanisms: execution robustness through pre-flight verification, expert-distilled skills, and just-in-time auditing; epistemic adaptability through persistent research artifacts and an evolving scientific mental model; and collective intelligence through a heterogeneous multi-agent round table under targeted human oversight. We evaluate MatHarness on adsorbate-induced surface segregation in Cu-Zn alloys, an open question in methanol-synthesis catalysis. Over a multi-week trajectory consuming roughly 10^3 GPU-hours (50 production DFT relaxations prescreened by 171 machine-learning-potential evaluations) the system revised its plan six times, recovered autonomously from failed calculations, and had an independent auditor overturn one of its own over-attributed claims before publication. Within the sampled metallic dilute-limit models, the validated DFT analysis supports an adsorbate-induced tendency toward greater surface Cu exposure at both composition endpoints, with facet- and coverage-dependent exceptions. These results were compiled into a complete manuscript with human involvement concentrated in specification, approval, and strategic steering gates. The code is publicly available at this https URL. 

---
# Selection Policies for Equal-Byte Mixed-Precision Quantization of Small Language Models 

**Authors**: Smit Anand  

**Link**: [PDF](https://arxiv.org/pdf/2611.28318)  

**Abstract**: Can changing which matrices receive 2, 4, or 8 bits improve a small language model without increasing its parameter storage? I compare two parent-selection rules in the same genetic algorithm: a binary-tournament variant and an Alternis-inspired rule. A separate baseline samples repaired random allocations without crossover or mutation. All three use the same MLX affine quantizer, byte ceiling, initialization and candidate allowance within each study. An exploratory study on Qwen3.5-2B evaluates 90 allocations per run. The main study on Qwen3-1.7B evaluates 500 distinct allocations per run over 10 paired seeds. Against uniform four-bit, tournament reduces held-out next-token negative log-likelihood (NLL) by 0.0616 on AG News and 0.0152 on IMDb on average, improving all 10 seeds on each corpus at identical parameter storage. Alternis improves five seeds and ties five on AG News; on IMDb its mean gain is near zero. Random returns uniform four-bit in every seed. The tournament representative answers 156 of 400 ARC-Challenge questions correctly versus 163 for uniform; no accuracy gain is demonstrated on this subset. The result favors this tournament variant for text prediction under the tested protocol, not for compression ratio or task accuracy. 

---
# INSEP: Refusal-Direction Inseparability as a Defense Against Abliteration 

**Authors**: Ivan Brugere, Junqi Jiang, Saumitra Mishra, Sumitra Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2611.28314)  

**Abstract**: Abliteration is a widely used white-box attack against the safety alignment of large language models (LLMs). It elicits harmful behavior by estimating a refusal direction in the model's activation space and projecting that direction out of the residual-stream write-out matrices. We propose INSEP, a defense that fine-tunes the model to reduce the separability of harmful and benign activations along the refusal direction. Abliteration estimates this direction as a difference in class means and projects it out. Reducing separability weakens the signal the attack relies on. An adversarial eigendecomposition selects the most separable prompt subsets. A low-rank adapter reduces their separation along that direction. Behavioral losses preserve the base model's refusal and the quality of its benign responses during training. We benchmark INSEP against six published defenses under five attacks, across eight LLMs from 1B to 14B parameters and several model families. Beyond response harm, we measure coherence, directness, over-refusal, and MMLU, to distinguish robust refusal from low harm produced by evasive or degraded responses. INSEP ranks first on the mean composite score across all eight base models while retaining utility, in the largest abliteration-defense evaluation reported to date. 

---
# GERF: A Token-Efficient Multimodal Retrieval Framework for Schema-Guided Scientific Information Extraction 

**Authors**: Yiyu Qian, Xing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28295)  

**Abstract**: Scientific information extraction (IE) is a fundamental component of AI for Science, transforming experimental knowledge embedded in scientific literature into structured and traceable data for evidence synthesis, scientific database construction, and data-driven discovery. However, critical experimental details are often distributed across text, tables, figures, captions, and surrounding discussion, making them difficult to recover using conventional text-only extraction. Existing LLM-based and multimodal RAG pipelines typically rely on full-document context or coarse page-level retrieval, introducing substantial irrelevant evidence and high textual and visual token costs. We propose GERF, a Graph-based Evidence Retrieval Framework for token-efficient, schema-guided multimodal scientific IE. GERF represents each article as a heterogeneous evidence graph that connects schema fields, text chunks, figures, and figure-derived content through semantic and structural relations. Using schema fields as retrieval anchors, GERF combines direct semantic retrieval, Random Walk with Restart, and null-aware evidence expansion to construct compact, field-specific multimodal evidence packages. Visual interpretation is applied only to graph-selected figures, avoiding indiscriminate processing of the entire document. We evaluate GERF from two complementary perspectives: evidence retrieval quality and pipeline-level input-token consumption. On the expert-annotated SOFC-Exp corpus, GERF retrieves gold-supported evidence for 94.55% of the evaluated field instances, providing broad field-level evidence coverage for downstream extraction. In the token comparison, GERF reduces average input-token usage by 73.40% relative to full-context processing and uses fewer tokens than the evaluated page-level pipelines. 

---
# Feedback-Driven Prompt Refinement for Educational AI: Affective Trajectory Estimation and Bloom's Taxonomy Classification 

**Authors**: Josh Dean, Karan Patel, Gaurangi Raul, Yu-Zheng Lin, Jesus Pacheco, Sicong Shao, Shalaka Satam, Soheil Salehi, Pratik Satam  

**Link**: [PDF](https://arxiv.org/pdf/2611.28293)  

**Abstract**: Contribution: This study has shown that a shared prompt-refinement procedure, guided by distinct affective and cognitive feedback signals, can improve held-out reference agreement and multi-label classification while reducing affective-generation variability without executor parameter updates. Background: Industry 4.0 has increased demand for personalized technical education across interconnected computing and engineering domains, while prompt sensitivity and generation variability have complicated dependable interpretation of instructional dialogue and learning outcomes. Research Questions: The study has examined whether task-specific feedback can improve affective-reference agreement, generation consistency, and cognitive classification without updating executor parameters. Methodology: A secondary language model has revised executor prompts using executor-output discrepancies against model-generated affective references for four dialogues and annotated Bloom labels for 70 learning outcomes; separate sets of 100 EduTalk conversations and 1,000 outcomes have supported held-out comparisons. Small refinement sets, unvalidated affective references, and absent learner-outcome measures have constrained interpretation. Findings: Affective mean absolute error has decreased from 0.742 to 0.702 and mean generation standard deviation from 0.307 to 0.238, while Bloom micro-F1 has increased from 0.687 to 0.893 and exact-match subset accuracy from 0.418 to 0.636. These findings have supported prompt refinement as a candidate approach for developing adaptive technical instruction, while its effects on workforce learning remain to be established. 

---
# Specialized Large Language Models for Markovian Reasoning: Structural Cache Compression, Transition Pretraining, and Dual-Loop Transformers 

**Authors**: Haibin Tong, Liangliang Xin, Yamei Zhu, Jiang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28285)  

**Abstract**: Standard autoregressive large language models attend to the growing history of generated tokens, making the key-value (KV) cache a central memory and throughput bottleneck in long-chain reasoning and multi-branch sampling. Yet an important class of reasoning problems is naturally first-order Markovian: the next step depends on a sufficient representation of the current state, rather than on the raw sequence of all earlier steps. We develop a unified design paradigm for Markovian-reasoning specialized large language models. At the cache level, the Markovian state interface transforms history-dependent KV storage into constant-size state transfer with respect to the number of reasoning steps. At the learning level, we reformulate long-chain prediction as independently trainable one-step transition objectives, with explicit state bottlenecks, distillation, invariance, and drift-consistency losses. At the architectural level, we introduce a dual-loop Transformer: an outer loop advances the reasoning state, while an inner recurrent-depth loop allocates adaptive computation to each transition through shared weights and explicit memory tokens or recurrent matrix states. We connect this architecture to a three-region cache, step-relative positions, truncated backpropagation, branch-aware checkpoints, and a transition-centric evaluation protocol. The organizing principle is simple: the optimal reasoning system need not remember everything; it must carry forward exactly what the future needs. This principle opens a path toward specialized reasoning engines whose memory footprint is decoupled from reasoning-chain length. 

---
# Temporal Logic Tensor Networks: Foundations of Temporal Reasoning in Neural Networks 

**Authors**: Edjard Mota, Artur d'Avila Garcez, Maria Azevedo, Odalisio Neto  

**Link**: [PDF](https://arxiv.org/pdf/2611.28277)  

**Abstract**: In dynamic scenes---football matches, traffic streams, surgery, or assembly---actions unfold at fundamentally different time scales: in football, a kick happens in milliseconds, a possession lasts seconds, and complex events such as a successful pass followed by a goal combine multiple atomic actions over extended durations. Models trained solely on visual data struggle to learn effectively across different time scales, lacking a principled way of reasoning across fine-grained instants and broader intervals. Neither point-wise nor purely interval semantics suffices: the former fragments events into frame-level snapshots lacking overall event identification; the latter treats intervals as atomic truth-values, ignoring state trajectories. We introduce Temporal Logic Tensor Networks (tLTN), a differentiable neuro-symbolic framework addressing this duality via a formal, multi-sorted logic where events and states are structurally interdependent. Extending Interval Logic Tensor Networks (ILTN), tLTN grounds state predicates as fuzzy truth-value paths over time points, while events are first-class entities with learnable trapezoidal fuzzy durations and differentiable Allen-style relations. A compositional semantics fuses static, event-structural, and state-dynamic satisfaction into a single differentiable LTN grounding. Crucially, the modalities constrain each other bidirectionally: the interval layer dictates where point-wise constraints (e.g., Until, Since) are evaluated, restricting them to event-defined spans; conversely, the point-wise layer validates interval relations, rejecting them if underlying state dynamics contradict the relation. Through controlled synthetic examples, we show when single-modality learning fails and how this localization reduces frame-level computational cost on long traces. 

---
# A Nepali-Accented English Evaluation Dataset for Automatic Speech Recognition 

**Authors**: Santosh Dahal, Kiran Chandra Dahal  

**Link**: [PDF](https://arxiv.org/pdf/2611.28263)  

**Abstract**: Automatic speech recognition (ASR) systems perform strongly on native-English benchmarks, yet their accuracy degrades sharply when the input speech comes from under represented non-native accents. Nepali-accented English is particularly under-served: existing resources either focus on native Nepali speech, cover broader multi-accent settings without dedicated Nepali evaluation, or provide only limited Nepali-accent coverage. This paper presents a Nepali-accented English evaluation dataset designed to support robust ASR benchmarking under accent mismatch. The corpus was collected through a web-based platform that did not collect directly identifying metadata and contains recordings from 57 speakers. Each session follows a fixed 22-prompt protocol consisting of 11 phonetic prompts, 10 domain prompts, and 1 spontaneous prompt, providing complementary coverage of pronunciation, topical vocabulary, and natural speaking style. In addition to transcribed speech, the dataset includes participant metadata for coarse exploratory subgroup analysis and speaker-level manual recording-quality labels. Manual quality assessment shows that 50.9% of sessions are clean and 42.1% contain only mild noise. As a descriptive reference, open-source ASR baselines are substantially worse on this corpus than the corresponding LibriSpeech test-clean values reported in official model cards, reaching 38.15-55.00% WER on the collected set versus reported 2-4% WER on LibriSpeech test-clean. These baseline results position the corpus as a practical held-out resource for evaluating accent robustness and out-of-distribution generalization on Nepali-accented English. 

---
# Demystifying the Autoresearch Harness: How Context and Skills Shape Self-Evolving Agents 

**Authors**: Kelvin Kiu-Wai Tam, Newt Hue-Nam K. Nguyen, Tianshi Zheng, Yunxiang Mo, Weihan Li, Zhongwei Xie, Chiu Yeung Tang, Jui-Yu Ho, Yangqiu Song, Ginny Y. Wong, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2611.28244)  

**Abstract**: Autoresearch agents, which autonomously design, run, and refine experiments, are accelerating scientific progress, particularly in AI research. However, existing agents rely on harnesses that combine multiple mechanisms, making it difficult to determine how individual components contribute to performance. We systematically study harness design for long-horizon, self-evolving autoresearch agents along two dimensions: context management, which determines what an agent retains from its past attempts, and skill management, which governs how it stores and reuses lessons from those attempts. Across MLE-Bench and KernelBench with gpt-oss-120b, gpt-6-luna, and gpt-6-sol, we isolate the effects of individual harness design choices by evaluating each against a recency-truncation baseline within a shared agent framework. For context management, we find that preserving individually accessible records of past attempts is more effective than maintaining a rolling summary, with context folding performing best across all six settings and improving upon the baseline by up to 19.2%. For skill management, selectively discarding persistently unused lessons is more effective on average than consolidating them or promoting them to standing rules. Building on these findings, we combine context folding and skill deletion in MLEvolver, which achieves a 72.7% medal rate on MLE-Bench Lite and a 94% $\mathrm{fast}_1$ score on KernelBench Level 3, substantially outperforming AIDE and OpenEvolve under the same inference budget. Overall, our study distills design principles for preserving experimental history and curating reusable lessons in autoresearch harnesses, paving the way for agents that turn accumulated experience into sustained progress. 

---
# FinEvo: Recursive Self-Improvement for Long-Horizon Financial Modeling 

**Authors**: Chengjun Pan, Shichun Liu, Zekai Zhou, Shihan Dou, Jiahang Lin, Chengsong Huang, Dingwei Zhu, Tao Gui  

**Link**: [PDF](https://arxiv.org/pdf/2611.28239)  

**Abstract**: Advances in large language models (LLMs) have expanded their role in high-value productivity tasks, including financial analysis and modeling, while enabling models to play an increasingly active role in their own improvement. This motivates two complementary lines of inquiry: can financial modeling capabilities provide a foundation for model self-evolution, and can self-evolution, in turn, advance financial capabilities? We introduce FinEvo, a self-evolution framework that integrates autonomous experience construction with financial capability improvement. FinEvo prepares financial models, iteratively evolves challenges through self-play, and learns from the resulting interactions using executable rewards, relying on the same model throughout. An end-to-end execution with Qwen3.6-27B shows that solving feedback supports both increasing reasoning demands and identifying specification issues that executable checks can miss. Learning from these experiences raises whole-task success from 22.73 to 29.09 on held-out within-source tasks and from 13.13 to 19.19 on SpreadsheetBench 2 Financial. Comparable gains across task sources and evaluation settings, together with improvement on a task category absent from training, suggest that FinEvo strengthens underlying financial modeling capabilities shared across tasks. These findings support self-evolution as a path toward broader financial competence, advancing the empirical exploration of recursive self-improvement (RSI) in professional domains. Code is available at this https URL. 

---
# AutoPILOT: A Visual Agent Harness for Real-Time Gameplay 

**Authors**: Mark Zhao, Pengcheng Wang, Rui Yang, Jiarui Yao, Xirui Jiang, Yuxi Chen, Bingxuan Li, Hanyang Chen, Ke Yang, Manling Li, Tong Zhang, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28202)  

**Abstract**: Multimodal foundation models can play video games well when the game waits for them, but often fail when it keeps running. We find that much of this gap lies in the harness, which decides what the model sees, what it remembers, and how its outputs become actions over time. Existing harnesses such as VISTA improve the first two, yet turn each model call into at most one action, so the agent acts only as often as its model thinks. We introduce AutoPILOT, which changes the third: in each model call, the model can send a primitive action, start an open-loop macro, or write a closed-loop controller that keeps playing from the screen without further model calls until it hands control back. To evaluate agents in real-time games, we build AFK-Bench, 120 diverse browser games grouped by how they respond to inaction and playable in real time or with the clock paused. With GPT-6.1 Sol under the same budgets, AutoPILOT reaches 68.6% real-time success against 38.3% for VISTA, and on GameWorld it keeps 80.0% in real time while VISTA drops from 84.1% to 42.9%. Its gains hold across models, and it reaches VISTA's final success rate at a fifth of its API cost. Our results suggest that how a harness turns model outputs into actions over time is a central yet underexplored part of agent design, and we hope AutoPILOT and AFK-Bench will serve as a foundation for agents in a world that does not wait. 

---
# From Point Clouds to Engineering Models: Benchmarking Agentic 3D Reconstruction in the Built Environment 

**Authors**: Changyu Du, Stavros Nousias, Syed Salman Ahmed, Kai Dietmair, Mohammad Alrabab'h, André Borrmann  

**Link**: [PDF](https://arxiv.org/pdf/2611.28197)  

**Abstract**: Coding agents can translate 3D measurements into CAD geometry, but reconstructing engineering models for the built environment demands semantic interpretation and consistent construction constraints beyond surface fitting. We formulate scene-level Scan-to-BIM (Building Information Modeling) as an agentic 3D coding task and introduce S2Bench, comprising public and proprietary scans of 128 real-world scenes across five domains, paired with expert-curated reference models spanning diverse spatial scales. A shared framework lets agents measure scans in Python, build class-labeled components in a live CAD/BIM kernel, and iteratively refine their code. We also propose parallel asynchronous collaboration, where a lead delegates versioned tasks, continues modeling while workers execute, and integrates revision-bound proposals. Evaluating five agents and a collaborative team across 1,028 reconstructions shows that scan-to-model residuals directly guide geometric repair, whereas inferring semantic classes and construction relations requires joint geometric, visual, and engineering reasoning that agents struggle to ground and verify. Consequently, the leading agent fits scan geometry closely ($96\%$ coverage within $5\,\text{cm}$) yet reaches only $63\%$ mIoU, and no agent passes all engineering checks on more than $38\%$ of scenes. Coverage and engineering conformance drop with scene scale, whereas running a smaller model as a team improves all three axes with greater coverage gains on larger scenes. The project page is available at this https URL 

---
# torchtsetlin: Fast, Interpretable Tsetlin Machines for PyTorch 

**Authors**: Vajira Thambawita, Pål Halvorsen, Ole-Christoffer Granmo  

**Link**: [PDF](https://arxiv.org/pdf/2611.28195)  

**Abstract**: The Tsetlin machine learns propositional rules with teams of Tsetlin automata that are driven by reward and penalty rather than gradients, so its models are interpretable by construction and cheap to evaluate. Its software, however, consists of C and CUDA kernels behind scikit-learn-style wrappers that sit outside the deep learning ecosystem. torchtsetlin implements the Tsetlin machine family as ordinary PyTorch modules: vanilla and weighted, coalesced, regression, convolutional over sequences, images and volumes, and dense per-pixel segmentation. The automata are a tensor buffer, so a model moves with .to(device), is saved with state_dict() and learns from a mini-batch with one update(x,y) call. Its core technique is a batched formulation of Type I and Type II feedback, in which automaton transitions become event counts realised as binomial draws, with one commit per mini-batch. On ten benchmarks taken from the existing libraries, it matches their accuracy, and on all but the two smallest it trains 2.4-10.9X faster than PyTsetlinMachineCUDA and 10.3-81.2X faster than tmu on the same GPU. The package is pure Python, MIT-licensed and on PyPI; every number in this paper is reproduced by one script. Code: this https URL package: this https URL. 

---
# SAMA: Training-Free Consensus Ridge Tracking for Camera-Based Heart Rate on Five Public Datasets 

**Authors**: Sudhakar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2611.28185)  

**Abstract**: Remote photoplethysmography (rPPG) is usually benchmarked by one heart rate per video, a figure that hides the short windows in which a tele-consultation would actually read the pulse. We present SAMA, a training-free estimator that (i) forms a quality-weighted consensus of POS and CHROM chrominance spectra, (ii) discounts frequencies occupied by pulse-free references derived from the same frames (frame motion, skin-mask geometry and background brightness), (iii) follows the heart rate through time with a Viterbi ridge under a random-walk prior, and (iv) returns a track-filtered, amplitude-normalised waveform with a per-window confidence. No parameter is learned and no dataset-specific tuning is used. On 218 recordings from 111 participants in five public datasets (UBFC-rPPG, PURE, SUMS, M3PD and a two-participant subset of MMPD), all processed with the same fixed face box and scored with a re-implementation of the rPPG-Toolbox protocol, SAMA has the lowest mean absolute error (MAE) averaged over datasets at every protocol level: 4.40 against 5.13,bpm for POS per video, 4.80 against 5.43 on 30,s windows and 5.05 against 6.59 on 10,s windows. Pooled over participants, the bootstrap interval of its paired difference from POS excludes zero on 30,s and 10,s windows (Wilcoxon p=0.11 and 0.004) but not per video, and it does not beat POS on SUMS per video or on the MMPD subset. Withholding the 20\% least confident windows lowers the pooled 10,s error from 4.06 to 2.33,bpm. We also show that the per-video protocol has a floor of 1.2--1.6,bpm on datasets where heart rate drifts, and that skin tone cannot be recovered from uncalibrated video, which motivates the instrumented, skin-tone-stratified collection we propose for the Indian population. The method does not surpass the best published neural networks on UBFC-rPPG; we report where it stands against them without re-running them. 

---
# CTLink: A Text-Prompted Segmentation Model for Linking Radiology Report Findings to 3D Chest CT Scans 

**Authors**: Mohammed Baharoon, Sung Eun Kim, Mahmoud Hussain Alabbad, Lucas Bijnens, Ali Piracha, Reem Alwasiah, Talal Yahyan, Abdulrahman Alamoudi, Mashael Alshehri, Rayan Alsager, Hannah Hancock, Anna M. Hunkemöller, Peter Youssef, Mona A. Alhammad, Afnan Aljumayan, Brecht Van Berkel, Eric Bijnens, Saud Alhussain, Maarten Steyvers, Rolf Symons, Bryan Eelen, Ayed Alaklubi, Maha Buraykan Alshammari, Amirah Fahad Alshammeri, Tomas Gilvydis, Ibrahim AlTurki, Monerah Almohideb, Ali Alburkani, Shahad Alhomaiani, Fatima Jabali, Abdulrhman Aljouie, Daesung Kim, Junha Im, Michael Moritz, Hassan R. Alomaish, Pranav Rajpurkar  

**Link**: [PDF](https://arxiv.org/pdf/2611.31222)  

**Abstract**: Radiology reports describe findings in text only, without visually marking them on the scan, so readers cannot directly see which abnormality each reported finding refers to. We introduce CTLink, a text-prompted 3D segmentation model that segments findings described in a chest CT report. CTLink was trained on 4123 noncontrast chest CT scans with 9547 segmented report findings, including ReXGroundingCTv2, a newly annotated dataset of 1134 scans and 1895 findings with every instance of each finding segmented. On the 200 ReXGroundingCT validation scans, CTLink achieved a hit rate of 0.816, an instance F1 score of 0.456, and a mean Dice score of 0.400, compared with 0.533, 0.175, and 0.234 for VoxTell, a publicly available text-prompted segmentation model (P < .001 for all). On the test set of the MICCAI 2026 ReXGrounding challenge, an ensemble of seven CTLink models achieved the highest hit rate (0.782) and instance F1 score (0.386) of all 48 participating teams. On 50 test scans segmented twice by independent radiologists, CTLink matched or exceeded inter-rater agreement. Diffuse non-focal findings remained the main limitation, partly because their extent is outlined inconsistently even between radiologists. The CTLink model weights are available at this https URL, and the dataset at this https URL. 

---
# ToolSmith: An Automatic Tool-Generation Framework for Multimodal Search Agents 

**Authors**: Rui Sun, Junhua Huang, Luca Zancato, Md Mofijul Islam, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31198)  

**Abstract**: Recent multimodal search agents use reasoning, vision, and live web search across many turns to answer questions beyond parametric knowledge by orchestrating a small library of hand-crafted search and visual tools. However, the library is laborious to create and maintain, and frozen when new query types, evidence formats, and failure modes are encountered in the wild. Human practitioners, by contrast, cope with insufficient tools by fashioning new ones from familiar parts and reusing them across tasks. Inspired by this, we propose ToolSmith, an automatic tool-generation framework that grows and reuses modules on top of a lean base tool library. We evaluate ToolSmith on three in-domain multimodal search benchmarks and ten out-of-domain benchmarks under a VLM-as-a-Judge protocol. ToolSmith consistently improves over the base library across deployed agents of varying scales, with larger backbones reaping greater gains, demonstrating the scalability of the approach. Our code and data are available at this https URL. 

---
# Learning Neurovascular Codes for EEG-to-fMRI Synthesis 

**Authors**: Yiming Sun, Yifan Wang, Aosong Feng, Yijia Ma, Ben Cheng, Xurui Wang, Yufan He, Yucheng Tang, Qing Lyu, Daguang Xu, Pei-Fen Kuan, Mingzhou Ding, Wen Li, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2611.31194)  

**Abstract**: In this paper, we introduce Neurovascular Codes as shared EEG representations for fMRI generation. We first train a separate EEG encoder to predict each fMRI signal, learning target-specific temporal and spectral features from the same EEG window. After training, we freeze the encoders and project their concatenated hidden features into a shared neurovascular code. Conditioned on this code, stochastic interpolant transforms Gaussian noise into a vector of fMRI signals, allowing different responses to be sampled for the same EEG window. We evaluate our approach on eight public simultaneous EEG-fMRI datasets, including four resting-state and four task-based datasets with 61,646 volume-level pairs. Under within-subject and held-out-subject evaluation, our model improves average Pearson correlation over the strongest encoder baselines by 19.1% and 19.8%, respectively. The corresponding improvements over the strongest generator baselines are 18.8% and 47.9%. Beyond reconstruction accuracy, we also evaluate the generated signals using subject-identification classifiers trained on real fMRI. Our method achieves the highest mean accuracy among the evaluated generators. We hope that our work opens up new opportunities for cross-modal brain modeling. 

---
# CO-GRAFT: Attributing Reinforcement Learning Gains in Search Agents via Role-Level Policy Grafting 

**Authors**: Ramraj Chandradevan, Sayontan Ghosh, Vinoth Selvendran  

**Link**: [PDF](https://arxiv.org/pdf/2611.31185)  

**Abstract**: Reinforcement learning (RL) improves large language model search agents by up to 32 exact-match (EM) points over their pre-RL initializations, yet overall accuracy gains do not identify which underlying functional capabilities improve or interact during training. We introduce CO-GRAFT, a reward-agnostic interventional attribution framework in which an RL-trained agent and its pre-RL initialization jointly generate a single trajectory under closed-loop execution with live retrieval. Each policy controls a mutually exclusive subset of four functional roles: intermediate reasoning, search-or-answer decision, search query generation, and final answer extraction. Evaluating all 2^4=16 role assignments without modifying weights or retraining defines a cooperative game whose Shapley values decompose the total RL gain across roles, while M"obius interaction terms distinguish standalone role improvements from multi-role coupling. Applying CO-GRAFT to Search-R1 agents across Qwen2.5 and Llama-3.2 models (spanning two sizes, base and instruction-tuned initializations, GRPO and PPO optimizers, and NQ and HotpotQA datasets), we show that query generation never exceeds 15% of the total gain; instead, gains are governed by pre-RL answer validity, search retention, model scale, and fixed task-level search budgets, shifting credit from reasoning toward answer extraction in larger or well-formatted answer response checkpoints while smaller base models rely on reasoning--decision coupling. 

---
# Deneb-Vritra: Descriptions, Edges and Names Examined by Blocking Via Restoring Token Representations and Attention 

**Authors**: Mohammed Nayeem Khan, Akhil P, Shahina Shaik  

**Link**: [PDF](https://arxiv.org/pdf/2611.31139)  

**Abstract**: Laya is a 421M-parameter encoder that answers a question about a text in one pass: each answer option, written as a name plus a description, gets a marker token, and one score is read from the network at each marker. We first confirm that such models can follow an option's name when it contradicts the description; the question is where in the network this happens. We locate it. On 951 synthetic questions we move each name onto another option's description and count how often the model picks the option with the right name (the name shortcut): 53% for the released checkpoint, 59% for our fine-tune v1, 11% for a fine-tune with name randomisation. On a 320-item sample, blocking attention from each marker to its own name tokens in layers 14 to 17 of 28 cuts v1's shortcut from 67% to 16% (19% on two schemas held out from component selection) while accuracy with the original names stays at 90%; blocking marker edges to the question, other options or text has no effect. No single head matters (largest effect 9 points), and restoring clean marker states undoes the swap from layer 16. Attention mass tracks the window by layer but not by head. Repeated on 5 checkpoints (released, fine-tuned, name-randomised, and a pair trained without two schemas), the knockout finds similar layers; in the name-randomised ones, blocking the description edges instead brings the shortcut back to 56-58%, so the read persists and is outvoted. Marker states decode the name from layer 0; a one-direction probe erasure changed nothing but was weaker than a random edit. Copying layers 0-13 or 14-27 of a name-randomised sibling into a fine-tune recovers 66% and 65% (v1); 21% and 45% (pair A) of its shortcut reduction. 9 of the 12 predictions we could score, written before running, failed. Head sweeps cover 5 of 5 checkpoints; data are synthetic and LLM-labelled; one architecture. 

---
# SRCWDNet: Semantic-Relational Cross-Window Dual-Graph Network for Multi-Label Remote Sensing Scene Classification 

**Authors**: Fareeha Amir  

**Link**: [PDF](https://arxiv.org/pdf/2611.31135)  

**Abstract**: Multi-label remote sensing scene classification remains a challenging task due to complex spatial-spectral patterns, extreme class imbalance, and inter-label dependencies that are often insufficiently modeled. To address these challenges, we introduce a Semantic-Relational Cross-Window Dual-Graph Network (SRCWDNet), which integrates three key modules: (1) a Cross-Window Attention module that captures local and global visual context through learnable global tokens; (2) a Dual Label-Graph module that adaptively combines statistical co-occurrence graphs and learned semantic-embedding graphs to facilitate label-relational reasoning; and (3) a Neural Classifier Chain that sequentially predicts labels to exploit inter-label dependencies. Evaluated using five-fold cross-validation and trained with an Asymmetric-Graph-Consistency loss, SRCWDNet achieves accuracies of 98.93% and 96.53%, with corresponding F1-scores of 0.9365 and 0.8723 on the MLRSNet and China-MAS50k datasets, respectively, outperforming CNN, transformer, and lightweight baselines. These findings demonstrate that jointly modeling visual attention and adaptive label correlations can substantially improve multi-label remote sensing scene classification, particularly under long-tailed class distributions. 

---
# FATCAT: Fault Injection Attacks Targeting Chip-level Alignment 

**Authors**: Peter Horvath, Ilia Shumailov, Zhuoran Liu, Lukasz Chmielewski, Lejla Batina  

**Link**: [PDF](https://arxiv.org/pdf/2611.31129)  

**Abstract**: Model alignment is the primary safety and security mechanism for enforcing refusal of malicious prompts in modern foundation models; yet, existing adversarial evaluations implicitly assume underlying hardware integrity. In this paper, we present FATCAT, the first physical fault-injection attacks capable of eliminating refusal in modern large language and vision-language models. Specifically, we present a non-invasive electromagnetic (EM) fault injection attack targeting GPU-accelerated Systems-on-Chip with physical access, requiring neither software execution privileges nor model-layer access. By leveraging an autonomous Radio Frequency (RF) side-channel trigger to synchronize EM faults to the prompt prefill phase, the attack corrupts safety-critical memory transactions across off-chip LPDDR5 DRAM and on-chip LPDDR physical interfaces without crashing the system while preserving model coherence. Evaluated on an NVIDIA Jetson Orin Nano and AGX Thor running Llama 3.2 1B and Gemma 3 1B, localized physical faults reliably force models to fulfill restricted requests, demonstrating that model-level alignment provides no robust security guarantees without hardware-level fault tolerance. Finally, we demonstrate the physical-world consequences of this vulnerability on an active 6-DoF robotic manipulator controlled by a SmolVLA vision-language-action policy. Our attack silently corrupts the model forcing the arm into abrupt, hazardous lunges; in two instances the robotic arm attempted to pinch the paper author -- proving that hardware faults can directly translate into real physical harm. 

---
# RUIBench: What Makes a Good Robot-Use Interface? 

**Authors**: Yanzhe Chen, Zhijun Cao, Zechen Bai, Wenzheng Zeng, Kevin Yuchen Ma, Yiqi Lin, Guoqiang Liang, Xiaokang Liu, Ke Sun, Kevin Qinghong Lin, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2611.31115)  

**Abstract**: Foundation models can reason about complex physical tasks, yet translating this intelligence into reliable robot execution remains challenging. We argue that a critical, often overlooked factor is the robot-use interface (RUI), which determines how agents express and execute physical actions. To systematically study its impact, we introduce RUIBench, comprising 30 manipulation tasks across five suites spanning world knowledge, in-context learning, discovery, dynamic interaction, and contact-rich manipulation. We further develop RUIKit, a unified framework that crosses six action abstractions with two interaction forms, yielding 13 interfaces under shared perception and control. In simulation, changing only the RUI shifts a frontier agent's success rate from 45% to 81%, and that of a fine-tuned 4B model from 13% to 47%. Our analysis reveals that effective interfaces reduce unnecessary decisions while preserving essential feedback, with their benefits shaped by agent behavior and physical uncertainty. We further uncover a verification gap, where agents frequently declare success despite observable failures, revealing the need for task-level verification beyond individual actions. Together, these findings establish RUI design as a critical factor in embodied agent reliability. 

---
# Attention Temperature Confounds Zero-Shot Context Extension 

**Authors**: Ruben Navasardyan  

**Link**: [PDF](https://arxiv.org/pdf/2611.31113)  

**Abstract**: A training-time attention change is usually judged for zero-shot context extension against a dense baseline beyond the training length at the default temperature, $\tau = 1$. The outcome depends on how miscalibrated the baseline is there: the share of perplexity it recovers when its logits are sharpened by one scalar. Under the standard recipe (fp32 master weights), an 8k-trained baseline is 8-15% miscalibrated at 30k in three seed-paired 365M twins; in one of them this makes the block-sparse-trained twin look 7% better, which calibration removes. In pure-bf16 training without master weights the miscalibration is several times larger: an AdamW step ($\beta_2 = 0.95$) on a gain initialised at 1.0 is below half the bf16 spacing at peak learning rates under $1.7\times10^{-3}$, so gains never move (22 checkpoints), the baseline is 37-58% miscalibrated, and block-sparse pretraining, attention dropout and ALiBi look 6-35% better. Calibrating one temperature per model and length removes all three advantages, leaves ALiBi 5% behind RoPE and keeps part of the gain of a positive control, NTK-aware scaling. Unfreezing the gains reverses the block-sparse advantage at 365M and removes 71-89% of it at 1.2B. We recommend reporting extension results at calibrated temperatures and checking that normalisation gains moved. 

---
# Leakage-Safe Auditing of Attribution-Driven Data Curation in Medical Imaging 

**Authors**: Malay Phadke, Akshay Gokhale  

**Link**: [PDF](https://arxiv.org/pdf/2611.31096)  

**Abstract**: When can an attribution-derived anomaly justify changing a medical-imaging training set? We formulate attribution-driven curation as three separately testable claims: anomaly detection, harmfulness estimation, and intervention utility. We test the first and third using a patient-disjoint, identifier-preserving audit of a 3D convolutional classifier for computed tomography angiography aneurysm candidates. We identify 232 exactly zero-valued arrays among 1,201 positive patches, including 170 training positives, and invalidate a historical 321-patch attention-failure subgroup whose identifier-free records were joined to files by position. On the held-out test set, GradCAM++ spatial concentration is central for every candidate class, reversing the historical pattern. GradCAM++ and Integrated Gradients show negative rank-order agreement for positives, limiting method-independent anatomical interpretation. Across five paired training seeds, removing the 170 corrupted training positives yields AUC 0.9617 +/- 0.0044 versus 0.9652 +/- 0.0039 for full-data training (difference -0.0035; 95% CI -0.0082 to 0.0007). Excluding 34 raw-zero test positives widens rather than closes the gap, countering a blank-memorization explanation. Budget-matched random-removal controls suggest the result is not simply an effect of reducing training size. Input corruption does not necessarily imply that deletion improves generalization. Our framework makes leakage control, sample identity, and intervention testing explicit in attribution-driven medical-imaging data curation. 

---
# AI-Assisted Programming and Labor Demand 

**Authors**: Lindsey Raymond, Eleanor Wiske Dillon  

**Link**: [PDF](https://arxiv.org/pdf/2611.31083)  

**Abstract**: Generative AI has demonstrated human-level performance on technical work tasks, yet evidence on how these task-level capabilities are changing labor demand remains scarce. We study the firm-level adoption of GitHub Copilot, an AI coding assistant, across 5,414 multinational firms accounting for 38% of the U.S. programming workforce. To identify causal effects, we exploit a natural experiment generated by firms' predetermined software procurement cycles. Early AI-adopting firms hired more workers before the launch of GitHub Copilot and had larger, more technical, and older workforces. After one year, GitHub Copilot adoption reduces quarterly technical hiring by 29% among review-induced adopters, with no effect on separations, overall technical headcount, or vacancy duration. The hiring slowdown is concentrated in the subset of junior technical roles that involve the highest share of coding work. Overall, AI coding assistance causes firms to shift demand towards more experienced workers and roles that pair some coding with broader design responsibilities. 

---
# A Comparison of Uncertainty in AI and Dynamical Climate Ensembles 

**Authors**: Jacob B. Landsberg, Devin M. McAfee, M. A. Fernandez, Elizabeth A. Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2611.31064)  

**Abstract**: Climate model ensembles and their uncertainty estimates are essential for evaluating risk and designing climate policy. Yet, in the assessment of newly developed AI climate models, evaluating uncertainty is often secondary to diagnosing mean-state climate biases and trends. In this work, we conduct the first systematic comparison of uncertainty representation in ensembles of state-of-the-art AI and dynamical atmospheric climate models. Examining 2-meter temperature, 500 hPa geopotential height, and precipitable water, we find that the AI ensemble exhibits reduced internal variability and model uncertainty across variables relative to the dynamical ensemble. We show that this dampened variability causes the AI ensemble to produce more limited climate extremes and yields underdispersion relative to ERA5. Under a uniform warming scenario, the AI ensemble shows increased model uncertainty. Dynamical models grapple with their own challenges, exhibiting stronger miscalibration (overdispersion) relative to ERA5 and larger climate biases than the AI models. As AI and dynamical climate models are mutually integrated into operational climate prediction, careful consideration of their differences in uncertainty representations is crucial for grounded assessment of climate risk. 

---
# From Local Errors to System Failures: Modeling and Benchmarking Error Propagation in Multi-Agent LLM Systems 

**Authors**: Wenlong Yu, Bingxu Lu, Qilong Wang, Qinghua Hu, Qianru Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.31063)  

**Abstract**: In multi-agent systems (MAS) built on large language models (LLMs), errors can propagate through intermediate messages and compromise final answers. However, outcome-based evaluation obscures how agents amplify, attenuate, or correct these errors. We propose Response-Aware Error Propagation among Agents (REPA), a structural model that connects measured local responses to system-level error propagation. REPA separately estimates input assembly and agent processing across multiple error channels, while theoretical analysis relates local modeling errors to endpoint prediction error. Experiments span three Qwen base models, six MAS workflows, and ten knowledge and reasoning settings in ErrPropBench-10. REPA achieves a mean condition-level Spearman correlation of 0.84 across base models, compared with 0.47 for direct regression. Controlled experiments also demonstrate its utility for targeted recovery. REPA-guided retrieval-augmented generation (RAG) improves accuracy over no retrieval by an average of 19 percentage points with 16% additional tokens. By linking local agent behavior to system susceptibility, REPA provides an interpretable foundation for studying error propagation and developing targeted optimization in MAS. 

---
# DKnown-Jev-2: Article-Level Answer-Evidence Classification and Efficient Inference with a Pre-trained Language Model 

**Authors**: Qi Li, Jianjun Xu, Jiu Li, Pingtao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.31056)  

**Abstract**: Evidence selection in normative knowledge question answering requires determining whether a candidate article contains content that directly supports an answer under the applicable conditions of the question. We present DKnown-Jev-2, an article-level answer evidence classifier designed for evidence integration across multiple paragraphs. The model jointly encodes question conditions, article metadata, and paragraph content, and uses a pointer-style classification head to predict Answer Evidence (AE) and Non-answer Evidence (NE) in a single forward pass. Ona manually annotated test set of 972 evaluation instances, DKnown-Jev-2 achieves 91.77% accuracy and 91.28% macro-F1, outperforming Jev by 2.16 and 2.37 percentage points, respectively, while reducing both evidence misses and non-evidence false positives. The test set contains 317 multi-paragraph instances, with a maximum body length of 9,322 characters. Shared article encoding and non-autoregressive classification support low-latency inference. These results demonstrate the effectiveness of the proposed method for article-level evidence selection in normative knowledge question answering. 

---
# CAST: A Deployed Speech-to-Text Cascade for Closed Captioning of Entertainment Television 

**Authors**: Aditya Mishra, Thanish Bolla, Bidisha Sharma, Balaji Pitchaikannu, Surbhi Sinha, Anup Kotalwar  

**Link**: [PDF](https://arxiv.org/pdf/2611.31055)  

**Abstract**: Despite rapid progress in automatic speech recognition (ASR), caption generation for long-form entertainment video remains challenging for content dense in music and effects, rich in accents, crosstalk, and noise. The transcript quality depends on segmentation and surrounding modules as much as the acoustic model. A single pass voice activity detection (VAD) can miss faint, low-energy voice while forwarding near-silent segments that induce hallucination and repetition in an autoregressive decoder. These competing errors motivate a hierarchical VAD that revisits rejected segments under semantic and acoustic retention checks. We present Captioning ASR Stack for Television (CAST), an integrated speech-to-text pipeline for closed captioning (CC) combining frozen Whisper large-v3 with segmentation, language identification, lyrics routing, and targeted hallucination detection and repair on a single NVIDIA L4 GPU. On 100 in-house assets spanning 60.46 hours, CAST reduces normalized WER from 6.99% to 5.36% and subtitle edit rate from 254.2 to 221.9 min/hour against the WhisperX pipeline, and improves deletion rate by 11% on public datasets. We also present the behavior of each stage in the pipeline. CAST entered the production speech-to-text (STT) stage of our CC pipeline in late August 2026; we report the evaluation evidence and path to full-scale operation. 

---
# From Head to Tail: LLM-Augmented Sequential Recommendation and Trade-off Optimization 

**Authors**: Adeep Hande, Shaghayegh Agah, Neeraj Sharma, Karina Halevy, Sardar Hamidian  

**Link**: [PDF](https://arxiv.org/pdf/2611.31052)  

**Abstract**: Recommender systems exhibit popularity bias, often under-serving long-tail and cold-start items with sparse interaction data. We propose a group-specific data augmentation approach that leverages LLMs to generate synthetic interactions, formulated as a constrained optimization problem, maximizing group-specific NDCG@20 and coverage while bounding overall NDCG@20 degradation within an acceptable threshold. Items are partitioned by interaction frequency to find the optimal augmentation strategy for each group. We further study combining the augmentation across groups simultaneously and find that while individual optimal strategies navigate the multi-group augmentation design, they are not simply additive. We further ablate on the LLM-based augmentation itself, comparing LLM scales, augmentation configuration under token constraints, finding that augmentation quality scales with contextual richness, albeit at a proportionally higher cost. Experiments with SASRec demonstrate the effectiveness of the proposed approach, achieving over 100% and 60% improvement in group-specific NDCG@20 for extreme and regular long-tail items, respectively, on our industrial dataset, and 40%, 25% for MovieLens-32M, while increasing their coverage without significant overall degradation. 

---
# Self-Supervised Learning for Soft-Constrained AC-OPF in Multiphase Distribution Systems 

**Authors**: Hoang T. Nguyen, Deepjyoti Deka, Priya L. Donti, Varsha Pendyala, Nurali Virani, Wajid Muneer, Reetam Sen Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2611.31026)  

**Abstract**: This paper presents a self-supervised learning framework for fast, soft-constrained AC optimal power flow (OPF) in unbalanced multiphase distribution systems. The soft AC-OPF objective balances operating cost, renewable energy curtailment, and penalties for violating operating limits. A neural network predicts distributed energy resource (DER) setpoints conditioned on operating conditions, switch configurations, and objective weights. The predicted setpoints are then processed through a DER projection and differentiable AC power flow to enforce device limits and network equalities. The framework is trained by directly minimizing the AC-OPF objective, without optimal-solution labels. Experiments on 123-, 240-, and 906-bus systems yield mean objective gaps of 0.32%, 0.92%, and 0.56%, respectively, relative to IPOPT, across wide operating ranges. Across these systems and the evaluated high-load/low-generation and low-load/high-generation scenarios, the largest mean gap is 1.04%, with mean speedups of 58-5272 times relative to IPOPT. 

---
# Parsing Visual Latents for Multimodal Spatial Reasoning 

**Authors**: Zhuoying Yang, Yichen Guo, Kathy Jia, Zichang Su, Ayesha Kashif, Nange Hai, Jeffrey Ming Han Li, Zhewen He, Silvio Kempf, Ni Jiachen, Junjie Yang, Muhammad Hamza, Zakaria El Ouahidi, Gabriel Azpitarte, Manling Li, Yi Fang, Shuhong Liu, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31021)  

**Abstract**: Spatial reasoning across multiple views requires inferring the 3D relations between objects and the camera from 2D observations, which depends on cross-view correspondence and scene geometry that vision-language models (VLMs) do not reliably extract. Pretrained vision encoders already capture this information in their latent features, with CLIP linking objects to image regions, DINOv3 matching semantics across views, and VGGT encoding multi-view geometry. However, VLMs do not benefit from these features when we present them directly in context as grayscale feature images. We therefore propose LatentParse, a training-free framework that parses latent features into explicit spatial evidence for VLMs. In LatentParse, an analyst VLM probes the features of all three encoders with predefined tools that locate objects, track them across views, and measure motion between views. It then distills the results into a concise text report, which is provided together with the images to the VLM to answer the question. We evaluate LatentParse on MMSI-Bench, MV-RoboBench and SPAR-Bench, where it improves all five VLMs over the photo-only baseline by 0.7-2.4 percentage points on average. On MMSI-Bench, it raises Qwen3.8-27B from 34.4% to 39.5%, with the largest improvements of 19.2 percentage points on the camera-related questions. These results suggest that latent features become useful for spatial reasoning once they are parsed into relation evidence, without training. Code is available at this https URL. 

---
# Can Latent Reasoning Drive LLM Outputs Without a Trace? 

**Authors**: Tomer Ashuach, Zorik Gekhman, Avi Caciularu, Ori Ram, Jonathan Herzig  

**Link**: [PDF](https://arxiv.org/pdf/2611.31015)  

**Abstract**: LLMs typically solve complex tasks with chain-of-thought (CoT) reasoning, which is also the main tool for monitoring them for misaligned behavior. However, standard transformers can also solve tasks with complete latent reasoning (CLR), carrying out the whole task internally and outputting only the final answer. Theory bounds the depth of CLR, the number of sequential steps it can carry out, and recent work measured it to be shallow. In this work, we ask whether CLR is deep enough for a model to pursue a hidden objective that requires reasoning. Such a model would generate a CoT that appears to solve the given task, while a latent multi-step computation decides its answer based on the hidden objective. First, we show that a simple auxiliary loss allows models to solve a multi-step task latently at least 5x deeper than previously reported. Then, in two common tasks, decision-making and code generation, we train a model to pursue such a hidden objective, which decides a loan or plants a vulnerability. The model's CoT justifies the output with no trace of the hidden objective, and a strong LLM monitor does not flag it. Finally, using activation patching, we find that the latent computation is completed before the model generates its CoT, which takes no part in it. Altogether, we show that a hidden objective that requires reasoning can go unseen by a monitor that reads only the output. 

---
# The AMPEL Platform: An Open-Source Project for Developing Hospital-Wide Clinical Decision Support Systems under EU-Regulation 

**Authors**: Alexander Twrdik, Daniel Steinbach, Martin Federbusch, Clemens Malkawi, Maria Schmidt, Toralf Kirsten  

**Link**: [PDF](https://arxiv.org/pdf/2611.30990)  

**Abstract**: As incentives to digitize more and more aspects of medical healthcare are gaining traction, additional medical data is becoming available. A major goal is to use this data to gain further insights, support medical staff, and ultimately improve patient care. This demands sophisticated solutions that can handle current and future vast amounts of data, all without disrupting well-established workflows of care takers. Additionally, the solution must fulfill the documentation and risk-management obligations of the Medical Device Regulation (MDR), which governs whether a device may be used in patient care in Europe. In this context, we introduce the AMPEL platform, an open-source solution for creating clinical decision support systems (CDSS) that fulfil the requirements of the MDR. Centered around a message broker (Kafka), it adopts an event-driven, distributed architecture, allowing for scaling and efficient processing of medical data as it gets generated. Relying on platform-specific events to implicitly describe the platform's interfaces makes it both extensible and agnostic to the hospital information system in use. With these events being defined in the Protocol Buffers format, medically evaluated algorithms can be independently implemented in a variety of programming languages and easily linked to the platform, extending its analytic capabilities as desired. The platform features a basic UI and alert functionality that can be used to gain medical staff's attention and present analysis results in a traceable and explainable fashion. We further show in a use case how the platform was used to build LAMPE, a CDSS developed and operated at University of Leipzig Medical Center. It demonstrates how the platform solution can be used in a complex and already established environment, while acknowledging current limitations and outlining planned developments to overcome these. 

---
# Reinforcing Diffusion Models by Instantaneous Trajectory Distribution Matching 

**Authors**: Yihong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.30986)  

**Abstract**: Forward diffusion reinforcement learning (RL)---enabling deterministic ODE sampling without trajectory backpropagation---has grown increasingly popular for efficiently aligning diffusion models. However, we reveal that existing methods essentially rely on signed reward-weighted regression (RWR), regressing the \emph{trainable policy} velocity onto rollouts from a \emph{behavior distribution}. Unbiased baseline subtraction implicitly assumes that the policy velocity equals the behavior marginal velocity---an idealized assumption violated under classifier-free guidance (CFG) rollouts, offline/off-policy updates, and imperfect online self-consistency. This \emph{behavior--policy mismatch} breaks baseline invariance and injects \emph{baseline-induced gradient drift}, destabilizing training. To resolve this issue, we propose \textbf{I}nstantaneous \textbf{T}rajectory \textbf{D}istribution \textbf{M}atching (\textbf{iTDM}), taking the infinitesimal limit of Trajectory Distribution Matching toward a time-local reward-guided target via the posterior-score identity to yield a sample-level gradient anchored at the behavior velocity without learning an additional reward-tilted target score. Tracking this velocity with an online auxiliary denoiser guarantees exact baseline invariance at the auxiliary optimum, effectively suppressing baseline drift across both online and offline RL. Extensive experiments across compositional generation, text rendering, and preference alignment show that iTDM achieves state-of-the-art in-domain and out-of-domain performance. Notably, on the challenging GenEval2 benchmark, iTDM converges nearly $3\times$ faster than competing methods while scaling smoothly to $0.87$ without collapse. Code is available at this https URL 

---
# Beyond Self-Consistency: Recursive Self-Improvement at Test Time 

**Authors**: Jianyu Wu, Yizhou Wang, Encheng Su, Jinouwen Zhang, Pengze Li, Lintao Wang, Chen Tang, Aoran Wang, Wanli Ouyang, Shixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30965)  

**Abstract**: Large language models (LLMs) that continue improving after deployment (e.g., recursive self-improvement) offer a promising path toward general intelligence. Current test-time methods pursue this goal on unlabeled questions, primarily relying on self-consistency: they generate independent solutions and use majority voting to select an answer or construct training targets. This leaves earlier reasoning unused at inference and can reinforce an incorrect majority during training. Humans instead build on earlier attempts to preserve useful reasoning, correct mistakes and improve their problem-solving strategies. Following this principle, we propose Test-Time Recursive Self-Improvement (TT-RSI), a label-free framework for improving LLMs at test time through two mutually reinforcing components: (1) In-Context RSI repeatedly re-solves each question using the full previous solution as a reference, letting the model decide whether and how to use it so that inference compute builds on earlier reasoning. (2) In-Weight RSI distills successive refinements into the model's weights, enabling progress beyond an initially wrong majority and theoretically bringing the first attempt toward the model's infinite-step refinement limit. The updated model answers better and refines more effectively, generating stronger references for subsequent training. Extensive experiments show that TT-RSI improves average accuracy by +13.89% over the base model and outperforms the strongest self-consistency-based method by +4.79% under the same test-time scaling budget. 

---
# SayDo: Benchmarking the Safety of Frontier Models from Chat to Closed-Loop Robot Control 

**Authors**: Benjamin Alt  

**Link**: [PDF](https://arxiv.org/pdf/2611.30957)  

**Abstract**: Frontier vision-language models (VLMs) are increasingly used as high-level robot controllers, yet their safety is evaluated almost entirely in text. We introduce SayDo, a benchmark that assesses the propensity of frontier models to cause harm in embodied contexts. SayDo comprises eleven laboratory and household tasks that pair a benign instruction with a hazard latent in the scene. Tasks are implemented in six settings along an embodiment gradient, from chat questions about an imagined robot to closed-loop control of a simulated robot arm. Benign controls, safe-alternative variants and scripted operator pushback complement each task. Across six open- and closed-weight VLMs and 11,480 episodes, hazardous-action rates rise from 14% in chat to 47% in the simulator, and from 2% to 80% for the most affected model. The increase is largely driven by open-weight models. We find that models flag hazards before acting in 73% of unsafe episodes and give way to pushback about twice as often when controlling robots as in chat. The safest models also show signs of overcaution on benign controls. Safety evaluations run in text therefore do not predict the safety behavior of the same models when used as robot controllers. 

---
# Vetoes and missed agreements in language-model bargaining 

**Authors**: Fuad Ali  

**Link**: [PDF](https://arxiv.org/pdf/2611.30950)  

**Abstract**: A language model played three representatives, each instructed to get the best payoff for its own group. One representative requested a plan, another chose the sole offer, and all three voted. If anyone voted no, each group received its fallback payoff. In 60 games with Qwen3-Next-80B, the representative at risk of a loss always requested a plan that would leave everyone better off. In each game, two copies of the proposer saw the same payoffs, peer request and reminder that the offer needed three yes votes. Showing the proposer how many groups gained under each plan raised mutually beneficial agreements from 6/60 to 31/60 (paired increase 41.7 percentage points, conservative 95% interval 18.8 to 58.5). Two of the four plans benefited everyone, so picking uniformly would put a mutual gain on the ballot half the time on average. In an earlier study, GPT-4.1 played 12 conflict games under two message conditions. Majority rule passed 15 of 24 offers, all at one group's expense. Unanimity passed two, both mutual gains. In the Qwen games, voters never saw the counts. They approved all 37 offers that helped everyone. On all 83 offers that would have cost one group points, its representative voted no. 

---
# Beyond Patching: Consistent Circuit Discovery via Gradient Spectral Analysis 

**Authors**: Yujie Zhang, Tianyi Zhao, Guanghui Min, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30939)  

**Abstract**: Mechanistic interpretability seeks to reverse-engineer neural networks by identifying small, faithful subgraphs--circuits--that implement specific behaviors. The dominant paradigm scores each unit by replacing its clean activation with a corrupted counterpart as causal intervention. While effective, this approach rests on a brittle prerequisite: the structure prompt pairs are task-specific, expensive to engineer, and silently couple the discovered circuit to prompt-design choices. In this work, we propose Spectral Circuit Analysis (SCA), a gradient-spectral pipeline that achieves consistent circuit discovery along both axes: stable across input formulations of the same task, and natively exposing structure shared across tasks. Our contributions are threefold. First, we provide a counterfactual-free reframing of circuit discovery: causally relevant directions are recoverable from the spectral structure of forward gradients across paraphrases, with leading directions concentrating task signal and trailing ones absorbing paraphrase noise, yielding circuits tied to the task rather than the prompt. Second, we introduce SCA, a block-wise SVD pipeline that extracts a task subspace per architectural block and redistributes spectral energy onto constituent units via a consistency-weighted share, producing per-unit rankings competitive with attribution-patching baselines under strictly less supervision. Finally, on twelve circuit-discovery tasks in GPT-2 Small and Pythia-160M, SCA performs comparably or exceeds EAP-family methods under the standard recovery-score protocol and recovers IOI canonical heads with Precision@K = 1.0 in top-10 slices on GPT-2 Small; the same machinery extends to a unified cross-task analysis classifying units as universal, family-shared, or task-specific and exposing compositional structure invisible to per-task analyses. 

---
# PRIMEAPE: A Multilingual Benchmark for Fine-Grained Meta-Evaluation of Judge Models 

**Authors**: Vladislav Mikhailov, Sarfraz Ahmad, Momina Ahsan, Zoia Butenko, Vladimir Dobrovolskii, Preslav Nakov, Erik Velldal, Lilja Øvrelid, Ekaterina Artemova  

**Link**: [PDF](https://arxiv.org/pdf/2611.30925)  

**Abstract**: Large language models increasingly serve as judges of other models, shaping benchmark rankings and model development. Yet, a judge whose rationale consistently supports its verdict may still disagree with human preferences. To study this, we introduce PRIMEAPE, a multilingual benchmark for evaluating judge verdicts and rationales. PRIMEAPE provides 2,205 response pairs covering 243 instructions in English, French, Spanish, Urdu, Hindi, Norwegian Bokmål, and Yoruba, with 11k human annotations of preferences and rationales across relevance, naturalness, truthfulness, safety, and overall quality. Our evaluation of twelve open- and closed-weight judges reveals that aggregate performance conceals differences across languages and criteria, while requesting rationales does not consistently improve agreement with human preferences. Judges also favor the first position, longer responses, and responses from related models, with effects that vary across criteria and languages. Crucially, judges can justify verdicts that disagree with human preferences while maintaining consistency between their verdicts and rationales. These findings show that consistency with a judge's own verdict does not establish alignment with human judgment. PRIMEAPE enables assessment of both, providing a foundation for more reliable multilingual judges. We release our benchmark and all annotation materials under the CC BY-NC 4.0 license at this http URL. 

---
# TrialAgentBench: Evaluating AI Agents for Clinical-Trial Analysis and Long-Horizon Drug-Development Decisions 

**Authors**: Bradley Segal, William Bolton, Philip Torr, David Clifton, Lei Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2611.30916)  

**Abstract**: Clinical drug development is a long-horizon, high-stakes process requiring sequential decisions based on information that costs money and time to acquire. AI agents have demonstrated capabilities for scientific analysis, but there is no established way to evaluate them on these decisions, as real trials are expensive, involve people, and never reveal the counterfactual. We introduce TrialAgentBench, a benchmark for evaluating AI agents on clinical trial analysis and drug development decision-making using simulated trials where the truth is known, embedded within the data records of a fictional drug sponsor. We publicly release nearly 3,500 tasks, varying from analysing a single trial to managing multi-phase programmes with budget and time constraints. We score each analysis by rerunning its code on new trials, and each programme by the value its decisions achieve. Across 24 closed- and open-weight models, the strongest agents can analyse raw, messy trial records to within 6% of a process-aware analysis, even when an assumption is violated, with 95% confidence intervals containing the true effect 94--95% of the time. Over the prior year, the best model improved from 19% above this analysis, with serious misses on 15% of tasks, to 3.5% and 1%. What limits them is decision-making from evidence under uncertainty: when to correct, what evidence to buy, when to stop. The strongest agents are overly careful, correcting on 83% of tasks with equal recording errors, where it is unnecessary. On drug programmes, the best agent has substantial headroom, reaching 48% of the value an oracle achieves against 58% for a rule-based developer, with others falling 24 to 66% short. Model generations show ongoing improvements in handling data and evidence, with substantial variation. The strongest models can act as competent analysts, but do not yet make reliable long-horizon decisions. 

---
# Replay Before You Reply: Second-Look Prompting for Unified Multimodal Encoding 

**Authors**: Yicheng Zhang, Xun Zhang, Jingyuan Ma, Tianyu Yu, Yuan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.30909)  

**Abstract**: Solving multimodal tasks requires Vision-Language Models (VLMs) to form a unified understanding of the image and the question text. However, under causal attention in current VLMs, a token can attend only to what precedes it. Whichever modality comes first is therefore encoded under partial observation of the input, uninformed by the rest of the multimodal context. We introduce Second-Look Prompting (SLP), a training-free method that repeats the image-question pair in the input. In the repeated copy, every visual and textual token can attend to the complete input, yielding a unified multimodal encoding of the pair. Moreover, the added visual tokens contribute computation beyond re-exposing the image: even a blank image in their place can improve reasoning. Across 12 benchmarks, SLP improves the overall average score of every evaluated model, spanning nine open-source models (3B-72B) and three proprietary frontier models, with gains reaching 5.6 points for Gemini 3 Flash on MATH-Vision. Our analysis reveals two complementary mechanisms: the question guides where the second image attends, and the added visual tokens act as a prefill scratchpad whose hidden states retain answer-relevant information from the image and question. These findings highlight input presentation as a simple yet effective way to improve VLM performance. 

---
# Predicting Rare-Mode Collapse in Recursive Generative Training from Single-Generation Calibration 

**Authors**: Jingyao Zhang, Mingyu Du, Lu Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.30897)  

**Abstract**: Recursive generative training can preserve sample counts in a rare region while silently eroding its characteristic core. Mapping this finite-horizon collapse across real-data quotas, rare prevalences, and feedback perturbations via exhaustive multi-generation chains is computationally prohibitive. We develop a decoupled framework that forecasts multi-generation collapse probabilities and locates operating risk boundaries from independent single-generation conditional fits alone. Pairing an exact three-source multinomial count law with single-generation response banks propagates count marginals in closed form, backed by explicit calibration-error and boundary-resolution bounds. For additive Gaussian perturbations, we further prove that equal-variance all-row and sparse policies reverse their boundary-crossing order at a unique hyperplane distance. Across a 2,178-setting count benchmark, exact propagation cuts risk mean absolute error (MAE) from 1.85 to 0.49 percentage points (pp). Across ten two-dimensional learners, calibration achieves 2.60--5.69~pp six-round core MAE on nine stationary learners (65--81\% below an unperturbed baseline), diagnoses non-stationary outlier drift in standard DDIM (16.38~pp) via a stability probe, and resolves the predicted six-round geometric reversal. On Fashion-MNIST latent flow-matching chains, calibration outperforms direct-chain Gaussian-process interpolation at lower fit budgets and tracks the 20\% latent-perturbation boundary shift (3.14 versus 10.78~pp MAE for proportion copying) driven by opposing inflow to the rare region and outflow from its core. 

---
# Speak When You See It: Proactive Task Assistants Should Wait for Evidence, Not the Clock 

**Authors**: Bohe Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30895)  

**Abstract**: Proactive assistants that guide people through physical tasks must decide, every few seconds, whether a step is complete and whether a mistake has occurred. We find that two benchmarks for this task largely reward knowing when rather than seeing what: a timer that never looks at the video, raising an alarm at 25 s and confirming each step at 30 s, reaches a mistake F1 of 0.373 on Ego-MC-Bench, above every published result (at most 0.27). We formalize this as the video-blind frontier, the best trade-off between completion accuracy and mistake F1 that policies reach from the elapsed time, the instruction and training statistics alone: every published per-step result on Ego-MC-Bench lies inside it, and so do all twelve zero-shot models we run on QICD. To move beyond it, we introduce Evidence-Contrastive Training (ECT): fine-tune on the benchmark's own questions, and pair its positives with negatives from the same recording at the same elapsed time, so that elapsed time cannot separate the two sides of a pair. Trained on 209 QICD videos in about 15 GPU-hours, a 2B model with ECT crosses the frontier on both benchmarks in all four seeds; transferred to Ego-MC-Bench without adaptation, it confirms 59% of mistake-free steps in time, against 30% for a published model with the same backbone trained on about 25k synthetic counterfactuals and real recordings. Its decisions follow the video beyond what timing explains: at equal elapsed time it tells a finished step from an unfinished one in 90-92% of test pairs, where the instruction's typical duration reaches 65%, rescaling the time stamps between the frames leaves it essentially unchanged, and swapping in another recording, of the same recipe where one exists, cuts its completion accuracy by two thirds. The code and model weights will be publicly released. 

---
# Bridging Continuous-Time MAPF and Industrial AGV Fleet Coordination with Lazy Priority-Based Search 

**Authors**: Federico Signorile, Raffaele Carli, Alessandro Bertogli, Roberto Olmi, Sven Koenig, Mariagrazia Dotoli  

**Link**: [PDF](https://arxiv.org/pdf/2611.30874)  

**Abstract**: Continuous-time multi-agent path finding on roadmaps (MAPF$_R$) offers a principled way to coordinate robot fleets with geometric collision avoidance, yet its deployment on industrial automated guided vehicle (AGV) infrastructures remains challenging, as highly connected and redundant roadmaps amplify both safe-interval reasoning and multi-agent interaction handling. To bridge this gap, we formalize AGV-MAPF$_R$-CTC, a configuration-space variant of MAPF$_R$ with precomputed continuous-time conflicts that enables the reuse of MAPF$_R$ solvers for AGV fleet coordination with minimal changes. We then revisit continuous-time priority-based search (CPBS), identifying the boundary conditions required by its safe interval path planning (SIPP) low-level planner under priority-induced constraints and enabling earlier solution detection. As a central contribution, we analyze the structure of the priority search space, clarifying which agents are affected by each priority refinement and when replanning can be avoided. Building on this analysis, we introduce a novel algorithm, Lazy-CPBS, which materializes high-level nodes lazily: a child's plans are computed only when it is expanded, so siblings that are never expanded cost no low-level search. Experiments on standard MAPF benchmarks and on industrial AGV roadmaps imported from a proprietary digital-twin simulator show that CPBS-based solvers remain effective well beyond the agent counts reachable by optimal MAPF$_R$ solvers, while Lazy-CPBS substantially reduces CPBS overhead and offers the most consistent robustness/runtime trade-off across industrial layouts. 

---
# Semantic-aware Decoding for Diverse Text-to-SQL under Schema Ambiguity 

**Authors**: Yuxuan Jiang, Anna Mitsopoulou, Apostolos Koukouvinis, Yannis Velegrakis, Georgia Koutrika  

**Link**: [PDF](https://arxiv.org/pdf/2611.30873)  

**Abstract**: Real-world text-to-SQL systems often face schema ambiguity, where a question can map to multiple plausible schema elements. To support downstream disambiguation, SQL generation systems should include diverse interpretations among their top-k predictions. However, large language models (LLMs) typically favor a single interpretation, producing highly skewed outputs. We propose Semantic-aware Logit Adjustment (SLA), a training-free decoding method that promotes diverse SQL generation by adjusting token logits based on semantic similarity between schema elements. Unlike prior approaches that rely on reasoning-intensive prompting or task-specific fine-tuning, SLA is efficient, model-agnostic, and easy to deploy. Experiments show that SLA consistently improves interpretation coverage over standard decoding and remains competitive with existing diversity-oriented methods at substantially lower inference cost. 

---
# Can Multimodal Fusion MakeSense for Embodied Vision in Degraded Sensing? 

**Authors**: Cheng-Ru Chou, Shih-Yu Lai, Pei-Lin Chiang, Shou-Po Tsai, Jing-Zhe Huang, Fukuharu Tanaka, Hirozumi Yamaguchi, Bing-Yu Chen, Yu-Lun Liu, Hsun-Ping Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2611.30836)  

**Abstract**: Embodied vision systems rely on complementary sensors to perceive and act as conditions change. Even models that accept arbitrary sensor subsets can exhibit negative fusion: adding a sensor increases prediction error. We propose MakeSense, a multimodal fusion framework that preserves sensor-specific evidence in private streams and combines information through shared bottleneck tokens. Training pairs full and subset views of the same degraded observations, penalizing higher error with more sensors. Where geometry permits, the distance readout uses range measurements alone, while the rest of the prediction draws on all available inputs. Paired tests across five datasets reveal helpful and harmful sensor additions. MakeSense reduces negative fusion on MM-Fi. With lightweight adapters and per-dataset training, the same design improves UAV localization on MMAUD, cooperative 3D detection on V2X-Real, few-shot pose transfer to RELI11D, and command prediction on Humanoid-Everyday, where a world model trained on the frozen core's outputs forecasts future fused states more accurately than a linear baseline at longer tested horizons. MakeSense supports embodied sensing across human activity, aerial systems, robotics, and autonomous driving. 

---
# Random Feature Neural Networks for Stochastic Optimal Control 

**Authors**: Guangyi He, Damir Filipovic, Lukas Gonon  

**Link**: [PDF](https://arxiv.org/pdf/2611.30804)  

**Abstract**: Random feature neural networks (RFNNs) sample their hidden parameters once and train only the output layer. We analyze RFNNs as feedback policies for finite-horizon stochastic control with trajectory-dependent objectives. A policy change alters later states, so the error that matters is the objective error of the induced trajectory rather than a pointwise regression error. We extend the quantitative random-feature approximation theory for PDE solutions and nonlinear regression to feedback control. We investigate the objective error of an RFNN policy through two sources, approximation of the policy by random features and estimation of the objective from finitely many paths. A stability estimate for the controlled trajectory turns a policy error into an objective error. Under bounded actions, a regular benchmark policy and exact empirical minimization, the expected objective gap to the benchmark policy is at most of order $N^{-1/2}+\sqrt{N(1+\log N)/n}$, with $N$ the feature width per output coordinate and $n$ the number of paths. At fixed sample size the two terms trade off in $N$, and balancing them gives $N\asymp\sqrt{n/\log n}$. Across a wide range of examples, estimated gaps generally decrease with width and sample size, and RFNNs show an advantage when simulated paths are limited. We also study Parsimonious Random Feature Learning (PRFL), which grows and prunes the random dictionary during training while training only the output coefficients. At matched retained width and training budget, PRFL improves on a fixed RFNN in the LQ and Heston comparisons, while the fully trained network remains best in those examples. 

---
# How Systems Treat Intent: A Faceted Taxonomy of Intent for Human-AI Co-Creation 

**Authors**: Ashwin Ram, Radha Kumaran, Alexander Koller, Kevin Baum, Jürgen Steimle  

**Link**: [PDF](https://arxiv.org/pdf/2611.30793)  

**Abstract**: Human-AI interaction depends on systems acting on user intent rather than explicit procedures, even when that intent is incomplete, evolving, or partially communicable. Literature organizing systems by interaction technique, workflow stage, or division of labor, offer limited insight into which facet of intent an interaction mechanism addresses. We introduce a six-facet taxonomy of intent, grounded in philosophical accounts of mind, action, and meaning, and apply it to systems presented in 198 papers from eleven SIGCHI venues. Our analysis reveals that different forms of incomplete intent are associated with distinct interaction and authorship strategies, showing that mechanisms that appear similar often address fundamentally different intent-related problems. This exposes blind spots, as intent is rarely represented as an interaction object, authorship provenance is rarely tracked, and tentative preferences are rarely distinguished from firm commitments. The taxonomy offers a lens for analyzing and designing AI systems around the facets of intent they address. 

---
# PVFormer: Proper Velocity Transformer for Stable and Scalable Hyperbolic Representation Learning 

**Authors**: Xianglong Shi, Nicu Sebe, Bernhard Schölkopf, Ziheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30785)  

**Abstract**: Hyperbolic representations have shown remarkable success on hierarchical and tree-like data, and Transformers have become a central architecture for representation learning. However, a principled construction of hyperbolic Transformers remains challenging because the self-attention layer decomposes into three coupled primitives, transformation, similarity, and aggregation, that should all respect the underlying geometry. Existing hyperbolic Transformer designs are typically built in the Poincaré or Lorentz models, where bounded domains, time-like constraints, tangent-space detours, or projection steps make it difficult to keep the full block intrinsic and scalable. To address this gap, we propose PVFormer, to the best of our knowledge, the first Transformer architecture that operates natively in proper velocity (PV) space. The proper velocity model provides an unconstrained representation of hyperbolic geometry, and PVFormer exploits this structure to redesign the Transformer block in PV space. On the transformation side, we use PV homomorphism transformations to update query and value representations. On the similarity side, we generalize Euclidean inner-product attention with a PV Busemann score inspired by Busemann-based hyperbolic learning and a PV point-to-hyperplane key transformation. On the aggregation side, we replace Euclidean averaging with PV midpoint aggregation and further derive a scalable PV linear attention variant for large graphs. Extensive experiments support the effectiveness of PVFormer across graph, text, and vision benchmarks. 

---
# SignalCount: Approximate Counts with Structured and Semantic Predicates in Seconds 

**Authors**: Matteo Agius-D'Arrigo, Zachary Zelner, Harrison Ju, Rushab Munot  

**Link**: [PDF](https://arxiv.org/pdf/2611.30779)  

**Abstract**: Many applications store records with structured columns and a text document, and ask how many records in a filtered set meet a criterion that only a reader of the text can judge, such as "How many patients with type 2 diabetes, aged 40 to 70, stopped metformin because their blood sugar stayed high?" An interactive application needs this count in seconds, as the user types the criterion, to decide whether the matching set is worth materializing with a batch pipeline in which an LLM judges the records. Prior approximate query processing over such semantic predicates has an expensive model, the oracle, label a sample at query time, so every query waits for the oracle. We present SignalCount, a production system that answers COUNT queries with structured and semantic predicates and calls no oracle at query time, even for ad-hoc criteria. Its oracle is the production LLM judge, and the judge's logged verdicts calibrate, offline, the fast classifier that already runs in front of the judge. At query time, SignalCount counts the filtered set exactly and draws a uniform sample of at most 1,000 records in one columnar scan, scores the sample with the classifier, and multiplies the average calibrated probability by the exact count. Its 90% interval combines a sampling term with one-sided, rate-dependent margins for calibration error, fitted by split conformal prediction across tenants. In a multi-tenant production deployment, latency has a p99 of 3.8 s and stays flat as the filtered set grows beyond the sample. On tenants held out from fitting, the interval covers the count that the production pipeline returns in 92% of estimates, although estimates run about 15% high in aggregate, because a calibration nearly unbiased per record is still biased per criterion. 

---
# Bridging the Semantic Gap from Natural Language Specifications to RTL with Layered Formal Guidance 

**Authors**: Yan Tan, Xiangchen Meng, Yangdi Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30766)  

**Abstract**: While large language models (LLMs) have shown promise in generating register-transfer level (RTL) code from natural-language specifications, their reliability remains limited on designs with complex control logic and temporal behavior. This stems from a fundamental semantic gap: high-level natural language often leaves implementation-critical details underspecified, forcing LLMs to infer unstated assumptions. This paper presents Layered Formal Guidance, a novel framework that bridges this semantic gap by using formal properties as an intermediate semantic layer between natural-language specifications and RTL generation. Our framework applies formal guidance across three layers: property-driven specification augmentation before generation, property-consistency checking after generation, and counterexample-guided refinement after checking. Experiments on RTLLM benchmarks and various open-source IP designs demonstrate that our framework achieves higher functional correctness and lower sampled output-trace diversity. Furthermore, we show that Layered Formal Guidance can flexibly augment both general-purpose LLMs and domain-specific models to substantially improve their generation reliability. 

---
# Object Count Generalization Is an Interface Problem: Warping a Frozen Skill Instead of Retraining It 

**Authors**: Paul Mattes, Ayman Abdeldayem, Denis Shcherba, Wojciech Samek, Marc Toussaint  

**Link**: [PDF](https://arxiv.org/pdf/2611.30765)  

**Abstract**: Reinforcement learning policies for robot manipulation degrade when the number of objects N in the scene grows beyond the training range. This is a covariate shift that also changes the dimension of the state. We propose WAVE (Warping Actions and Views of an Expert), which extends a pretrained single-object skill to many objects without updating its weights. WAVE has three parts. First, a selection policy picks the one object the robot works on next. Second, a warping policy reads a local view of the scene around that object, whose size does not depend on N. It emits a multiplier and an offset for every quantity the skill reads and every action it outputs. Third, the skill acts under this affine warp. It is shown a scene that holds a single object, while the warp steers it around the objects it is not shown and edits the actions that are sent to the robot. Both the selection and warping policies are trained with PPO on the multi-object reward and use the skill as a black box, so WAVE is task-general and skill-agnostic. It requires only that the skill was trained on the same embodiment and that every object has a goal pose. We also bound the number of attempts needed to place all N objects, assuming positive expected net progress per attempt, and measure this progress for each object count. We evaluate WAVE on cube arrangement tasks with a gantry robot and with a Franka Panda arm. Trained with at most nine cubes, it generalizes zero-shot to several times its training count, on the gantry to scenes of up to thirty cubes. Published methods that learn multi-object manipulation show count generalization up to six objects. Fine-tuning the skill, learning from the same local view without it, and an entity transformer, even when distilled from the pretrained skill, all collapse within their training counts under the same training budget. 

---
# Optimizing Locally Equivalent Graph States in Quantum Computing Using Greedy Best-First Search 

**Authors**: Sora Harada, Ryo Kuroiwa, Shin Nishio, Takeaki Uno  

**Link**: [PDF](https://arxiv.org/pdf/2611.30738)  

**Abstract**: In measurement-based quantum computation and quantum repeater networks, graph-structured quantum states called graph states play a central role as an essential resource. In a graph state, each vertex corresponds to a qubit, and each edge is prepared by applying one controlled-Z (CZ) gate. Local complementation, which toggles adjacencies among the neighbors of a selected vertex, produces operationally equivalent graph states up to an inexpensive operation called local Clifford gates. Therefore, finding equivalent graphs up to local complementation with fewer edges and lower maximum degree can reduce the preparation cost, such as the number of CZ gates and the circuit depth. In this paper, we formulate the problem of finding a desirable equivalent graph via local complementations as a state space search problem. We apply a heuristic search algorithm, greedy best-first search (GBFS), to the formulated problem. Our experimental evaluation shows that despite its simplicity, GBFS outperforms existing approaches based on integer linear programming or local search across multiple graph classes, including random graphs, bounded-degree graphs, and most of the graphs constructed from quantum LDPC codes. In addition, we empirically analyze GBFS's search behavior, confirming that we need search rather than stopping at the first local minimum to find the best solution. 

---
# CocoonBench: Can Multilingual Search Agents Escape Language Information Cocoons? 

**Authors**: Hongchuan Zeng, Bowen Liu, Changjiang Gao, Senyu Han, Zhipei Xu, Peisong Wang, Xinbing Liang, Yuhan Li, Nuo Chen, Blaze Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30727)  

**Abstract**: Multilingual search agents can access the global Web, yet may remain trapped in a language environment with insufficient evidence; we refer to this failure as a language information cocoon. However, existing search benchmarks largely operate within a single language, especially English, overlooking entity connections across multiple languages. We introduce CocoonBench, a benchmark that evaluates whether multilingual search agents can escape such language information cocoons. CocoonBench mines answer-centered relational paths from a frozen Wikidata-derived graph and selects clue sets that are individually necessary and jointly identify a unique answer. Using language-specific entity names, it records each task's minimum language cover, removes direct identifiers, and renders each task as matched prompts in three languages together with its construction certificate and provenance. Across 10 state-of-the-art models, mean accuracy on CocoonBench ranges from 33.8% (DeepSeek-V4-Flash) to 69.7% (Claude-Opus-4.8), leaving even the strongest system below 70%. The same construction certificates support path-aligned trajectory analysis, enabling diagnosis of model failures in language routing, evidence-path recovery, and answer use. The result shows that escaping a language cocoon requires more than switching languages: successful cross-lingual search depends on both covering the correct languages and connecting the evidence found across them. 

---
# Uncovering the Tells: Efficient Black-Box Audits for LLM Data Contamination 

**Authors**: Dominik Bernard Lau, Filip Pawlicki, Jan Dubiński, Sebastian Cygert  

**Link**: [PDF](https://arxiv.org/pdf/2611.30639)  

**Abstract**: Existing methods for assessing training data contamination in Large Language Models achieve high accuracy, but fundamentally rely on gray-box access, rendering them inapplicable to fully black-box environments. This limitation forces auditors to rely on full-sequence sampling or rigid heuristics, both of which incur prohibitive inference costs and struggle to isolate the underlying memorization signal. To overcome this, we propose a generation-based black-box framework that infers context-induced information gain from targeted token discrepancies. We introduce a budget-constrained auditing regime, building on the empirical observation that the membership signal is highly sparse and concentrated in a few highly informative tokens. We formalize this sparsity, showing why uniform sampling needs many more samples than a policy that targets informative positions. Empirically, our approach achieves state-of-the-art black-box separation accuracy while needing about 3x fewer queries than uniform position sampling. Finally, we apply this methodology to closed frontier models, exploring its capacity and limits in surfacing potential training data overlaps on commercial APIs. 

---
# GeoMerge-OPD: Balanced Skill Integration in Vision-Language-Action Models 

**Authors**: Le Yang, Haoyang Lyu, Honglei Jin, Wenshuo Chen, Jiawei Liang, Wenqiang Wang, Yutao Yue, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.30632)  

**Abstract**: Vision-language-action (VLA) models excel at individual robotic skills, yet integrating heterogeneous skills into a single policy remains a key challenge for general-purpose robotic control, especially in whole-body tasks with differing manipulation and locomotion requirements. Common multi-skill integration strategies, such as joint fine-tuning and parameter merging, can induce unintended actions during closed-loop execution, severely degrading individual skills. Task-conditioned analysis of expert updates reveals response energy concentrated in a few dominant directions and opposing expert responses along high-energy directions within a shared, task-relevant response subspace. This structure co-occurs with unintended action shifts after merging. We propose GeoMerge-OPD, combining response-space regularization, parameter merging, and on-policy distillation. During independent expert fine-tuning, G-isometry constrains update-response gains within task-activation subspaces; the expert parameters are then statically merged into a student with preliminary multitask capabilities. Fine-tuned task experts serve as teachers, providing velocity-field supervision at the environment observations and intermediate action-flow states visited by the student to calibrate residual behavioral deviations. Experiments with two VLA backbones show strong, balanced humanoid skill integration on SIMPLE and consistent performance on LIBERO and LIBERO-PRO, outperforming joint fine-tuning and multitask reinforcement-learning baselines while maintaining robustness under perturbations. 

---
# Oops!... I Parsed It Again: Tool-Call Smuggling Against LLM Agents 

**Authors**: Igor Santos-Grueiro, Daniel Romero-Ruiz  

**Link**: [PDF](https://arxiv.org/pdf/2611.30624)  

**Abstract**: LLM agents act through tool calls: the model writes the arguments of a call as JSON, a guard decides whether the call may run, and an executor runs it. The same bytes are parsed several times, by the provider, gateways, SDKs, frameworks, the guard, and the executor. JSON leaves some choices to each parser, such as which copy of a repeated key counts. When the guard and the executor choose differently, the guard approves one call, the executor runs another, and the audit log records the approved one. We introduce this attack, tool-call smuggling, and study it across the stack. We describe six ways in which the same bytes yield different calls and build a differential testing harness that gives identical bytes to 17 pinned parser and validator configurations; most pairs disagree on some inputs, and even two official MCP SDKs accept different inputs. On tasks from $\tau$-bench and AgentDojo, an attacker who can rewrite the bytes in transit cancels another customer's reservation and creates a file that the guard blocks by name, while the audit log names only the approved call. Content the model reads splits guard and executor only inside a string: ten models we test write no repeated or disguised key into the call itself where we see it, but copy one verbatim into JSON text in an argument, which a Python guard and a Go executor read differently. A malicious server copying a legitimate tool's public schema and adding one field, which a note explains, gets the model to fill it, unseen by the guard, with user data. Parsing each call once, JSON text included, keeping only the fields of the approved schema, and handing it to the executor, sealed with its call ID in transit, stops all 7,412 crafted inputs and every attack we built. 

---
# HOSE: Hierarchical Orthogonal Subspace Erasure for Concept Unlearning in Diffusion Models 

**Authors**: Xiaoqi Wei, Li Li, Kejiang Chen, Hanzhou Wu, Xinpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30592)  

**Abstract**: Text-to-image diffusion models can generate harmful or sensitive content, such as pornographic or violent imagery, raising serious security concerns. Model editing has recently emerged as an efficient approach to mitigate this issue. It modifies pretrained models with minimal parameter updates to alter specific concepts while preserving overall model capabilities. However, due to the entangled representations of different concepts in diffusion models, direct modifications inevitably disrupt non-target concepts. Meanwhile, U-Net's hierarchical heterogeneity hinders unified strategies from balancing erasure and quality. To address these issues, we propose Hierarchical Orthogonal Subspace Erasure (HOSE). This method projects parameter updates into the subspace orthogonal to general knowledge, thereby eliminating interference with non-target concepts. Considering the hierarchical heterogeneity of the U-Net architecture, we further propose a hierarchical adaptive mechanism, constructing differentiated subspace projections for different layers to precisely adapt to the model's hierarchical structure, thereby resolving the inherent trade-off between erasure and image quality in unified constraints. Experimental results demonstrate the proposed method reduces harmful content generation by 80% compared with the unedited model. While it inevitably incurs some image quality degradation, it still outperforms other editing-based baselines in terms of quality preservation, and achieves a significantly better trade-off between harmful content suppression and quality loss than these baselines. Notably, our approach demonstrates higher adversarial robustness than existing SOTA method, indicating that the target concepts have been more thoroughly erased. The source code is publicly available at this https URL. 

---
# SuSi-RAG: A Hybrid Graph RAG for Equipment Maintenance 

**Authors**: Hansi Zhang+, Wilma Johanna Schmidt+, Wanlin Yang, Xiaozhi Shen, Adrian Paschke, Evgeny Kharlamov  

**Link**: [PDF](https://arxiv.org/pdf/2611.30588)  

**Abstract**: In the manufacturing domain, equipment maintenance tasks depend on expert knowledge. Their documentation is often stored in heterogeneous formats and sources. The enhancement of LLMs with Retrieval-Augmented Generation (RAG) is promising for bridging the scattered knowledge, but traditional vector-based RAG shows weaknesses with two characteristics which are common in real maintenance queries: complex causal chains and cross-domain synthesis. To address this, we propose SuSi-RAG, a hybrid Graph RAG solution, that matches a user query and each maintenance record using subgraph similarity. This work shows an in-use system: SuSi-RAG has run on an active Bosch production line for more than eight months, handling 10,664 question-answer interactions and demonstrates upper bound answer accuracy of 96.25% (confirmed error rate 3.75%). In a proof-of-concept, SuSi-RAG improves top-1 retrieval accuracy from 61.5% to 96.2% and expert-rated answer quality from 44.9% to 84.1% against a vector-based baseline. We present an integrated operational feedback loop to update SuSi-RAG. 

---
# VeriTouch: Verified Visuo-Tactile World Models that Improve Robot Policy Learning 

**Authors**: Shuxun Wang, Xiaonan Jiang, Zhaoyu Wang, Ziyuan Chu, Xihang Yue, Pengfei Liu, Haodong Zhu, Shiqi Liu, Mingzhi Zheng, Liang Su, Linchao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30572)  

**Abstract**: Scaling visuo-tactile policy learning is constrained by the high cost of real-world data collection and the difficulty of building realistic simulations. Generative models offer a scalable source of training data, yet useful demonstrations require plausible tactile feedback, temporal alignment with visual contact, and accurate actions. To this end, we present VeriTouch, a framework for generating and verifying synthetic visuo-tactile trajectories for policy learning. Conditioned on initial images and task instructions, our world model, VeriGen, jointly generates high-fidelity dual-view videos and realistic tactile sequences over complete manipulation trajectories. By integrating a dedicated tactile expert with a pretrained video model, it captures realistic visual details and tactile deformation dynamics through joint visuo-tactile modeling. Our multimodal verifier VeriEval then checks visual quality while focusing on tactile reliability, assessing tactile realism, contact-force plausibility, and temporal alignment between visible contact and tactile signals, thereby directly filtering out unreliable sequences. Finally, our inverse dynamics model, VeriIDM, combines global scene information from both camera views with fine-grained gripper details and motion across neighboring frames to extract actions from verified videos. Experiments on five real-world tasks show that synthetic data improves all four evaluated policies, with further scaling synthetic trajectories substantially boosting policy performance without additional real-world data collection. 

---
# RoboInF: A Benchmark for Instruction Fidelity in Robotic Manipulation 

**Authors**: Jiawei Guo, Yu Huang, Dijiang Dong, Yue Liu, Peiran Ren, Wei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30555)  

**Abstract**: Precise language control requires embodied models to follow instructions governing both task outcomes and execution. Although recent methods train policies to follow increasingly detailed commands, existing benchmarks often assess instruction following alongside perception, planning, and generalization to unfamiliar scenes. Higher task success therefore offers limited evidence of closer adherence to instructions. We introduce RoboInF, a benchmark measuring how reliably manipulation policies adapt their behavior to changes in an instruction. For each of eight tasks, six to twelve expert-validated instruction branches vary the requested goal, procedure, or manipulation detail while holding the initial scene and robot state fixed. Task-specific checkers verify the terminal goal and use recorded trajectories, contacts, and event sequences to assess compliance throughout execution. The accompanying RoboInF-4K dataset pairs 4,242 expert demonstrations with simulator-derived annotations of motion and contact. We evaluate leading manipulation policies ($\pi_{0.5}$ and OpenWAM) and inference-only agents (GPT-6 Astra and Qwen3.8-Flash). All four models exhibit a gap between task completion and faithful execution, indicating substantial room for fine-grained instruction following. 

---
# GC-MoE K3: A Green-Context Role Graph for Latent Mixture-of-Experts Decode 

**Authors**: Yong Li, Haoyu Wang, Kan Liu, Tao Lan  

**Link**: [PDF](https://arxiv.org/pdf/2611.30542)  

**Abstract**: Kimi K3 combines a latent routed mixture of experts with full-width shared experts. A complete feed-forward evaluation must score and select experts, project tokens into the latent space, exchange tokens across expert-parallel ranks, compute the routed and shared branches, and normalize and project the routed sum before joining the branches. We present GC-MoE K3, a datacenter Blackwell implementation that schedules this entire evaluation as a CUDA Graph across two Green Contexts. Five base roles split input projection, routing, dispatch, expert computation, and finalization. Dispatch can fill the routed input ring while the compute partition evaluates shared experts. Finalize waits for the expert owners needed by each token and overlaps routed reduction and RMSNorm with loading the output projection weights. Conditional roles reuse the communication partition for routed computation and reduce split-$K$ projection partials. On GB200 at EP8, EP16, and EP32, the operator is faster at all 24 measured configurations than the paired SGLang-style reference (mean speedup $1.1324\times$). A comparison with historical vLLM-style measurements gives $1.3651\times$ across the same 24 shapes. Both latency comparisons time single-layer graph replay on synthetic K3-shaped inputs with a residual prefix. The separate paired numerical suite passes all 142 cases and 2,936 rank checks. 

---
# DelayOPSD: On-Policy Self-Distillation from a Synchronous Teacher for Real-time Asynchronous Policies 

**Authors**: Hyeonjae Kim, Jeongsol Kim, Sunwoo Park, Chanyu Moon, Yoojin Oh, Youngjun Jun, Kyumin Choi, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2611.30534)  

**Abstract**: Flow-based vision-language-action (VLA) policies incur substantial inference latency due to their large vision-language backbones and iterative denoising. Asynchronous execution reduces this latency by generating the next action chunk while executing the current one, but the new chunk is conditioned on a stale observation, degrading performance as the delay increases. We investigate this mismatch by comparing actions predicted from observations at planning and execution times. We find that these predictions differ little on expert demonstrations but diverge substantially on the policy's own delayed rollouts, particularly in failed episodes. Motivated by this observation, we propose DelayOPSD, which trains an asynchronous policy to imitate its synchronous counterpart on the policy's own rollouts. Specifically, a synchronous teacher, obtained from the same policy using the observation at execution time, supervises a delayed student that observes only the earlier frame and the actions still being executed. The student learns to match the teacher throughout its own denoising process. Under a 10-step delay, training SmolVLA on demonstrations achieves only 70.2% success, comparable to naive asynchronous execution (70.4%), whereas training on its own rollouts improves success to 78.7%. Across LIBERO and Kinetix, DelayOPSD achieves the highest average success among the compared methods, without introducing additional modules at deployment. On a real robot, it further improves success from 46% to 62%. 

---
# CyberDualEval: Measuring Dual-Use Cyber Risks in Frontier Language Models 

**Authors**: Magnus Saebo, Francesco Piccoli, Maxwell Watson, Pranjali Thakur, Yangruibo Ding, Zhuo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30533)  

**Abstract**: While frontier language models are becoming capable cybersecurity assistants, they also create a dual-use cyber risk: the same capabilities that help defenders identify and analyze vulnerabilities can similarly help attackers weaponize them. Existing cyber benchmarks measure either raw offensive capability or generic refusal behavior, overlooking the dual-use cyber assessment of preserving benign security utility while refusing assistance that materially advances misuse. We introduce CyberDualEval, a benchmark with 430 vulnerable tasks for measuring dual-use cyber risk in frontier language models. Each task is decomposed into three phases of increasing operationalization: security analysis, proof-of-concept generation, and exploit generation. To quantify the residual cyber risk, we annotate each task with a Task Weaponization Score that measures how much work remains to turn the resulting artifact into unauthorized cyber misuse. CyberDualEval assesses ten closed- and open-weight language models both on correct compliance and on task performance, developing four deterministic mechanisms for determining per-phase success on a large subset of the dataset. Our evaluation reveals that none of the existing frontier models effectively navigate helping with safe requests and refusing risky ones, characterized by being overly-cautious. On the other hand, recent open-weight models rarely refuse and demonstrate strong validated accuracy on risky requests. 

---
# sudo L7: Can Coding Agents Act Like Staff Engineers? 

**Authors**: Nick Heiner, Ian Niebres, Suhaas Garre, Sushant Mehta, Edwin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30517)  

**Abstract**: Coding agents can now turn a well-specified ticket into a patch that passes its tests. Experienced engineers also notice risks the request never mentioned, question flawed premises, reuse what already exists and say plainly what they did and did not check. We present sudo L7, a benchmark of 60 software-engineering tasks written by staff-level engineers, mostly in private production codebases and each built around a place where a frontier model failed. A task pairs a realistic, underspecified request with a hidden, expert-authored rubric of atomic criteria (1,287 in all), tagged with eight behavioral dimensions from functional correctness to truthful reporting. An agentic grader reads the agent's transcript, inspects and runs its code, and treats test output as evidence. An attempt passes only if it meets every criterion of the two highest severities, and more than half of these gating criteria test something other than functional correctness. On the public leaderboard as of November 2026 (19 models, each in its developer's harness or in OpenCode, three attempts per task), Claude Fable 5.2 passes 52.2% of attempts, Claude Sonnet 5.5 50.6% and Claude Opus 5.5 46.1%. GPT-6.1 Sol, Claude Fable 5.1 and GPT-6 Astra pass 31-34%, and the other 13 models at most 18.9%. In a human-label study, three candidate model graders each agreed with a software engineer on 85-86% of criterion verdicts, about as often as two engineers agreed with each other. Cases from the benchmark show how one unsupported claim can fail an otherwise strong attempt and illustrate two sides of engineering judgment: flagging hidden hazards and avoiding needless work. 

---
# JEPA-ICL: Progress-Aligned Latent Plans for In-Context Robotic Manipulation 

**Authors**: Tingjun Dai, Jianxiang He, Mingfei Han, Qihao Yu, Longtao Jiang, Changlin Li, Minghao Jin, Zhihui Li, Xiaojun Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30516)  

**Abstract**: Demonstrations show how a task can be completed, but differences in embodiment, appearance, and execution speed make it difficult to identify the guidance relevant to the robot's current progress. We introduce JEPA-ICL, a vision-language-action framework that couples predictive planning with demonstration learning. The policy learns to plan by predicting ordered future states in JEPA feature space up to subtask completion, capturing what remains to be accomplished. These plans guide action generation and locate relevant stages in demonstrations, whose JEPA-supervised content features provide progress-aligned context. Learning from this context, in turn, updates the shared policy that produces the plans and actions. This coupling connects anticipating the robot's own execution with reusing demonstrated experience. On the Imitator Game, JEPA-ICL achieves 80.0% seen-task success under C15 pretraining, outperforming Zero-WAM by 12.0 percentage points. Without contextual demonstrations, JEPA-ICL outperforms the VLA-JEPA variant without human-video pretraining, improving success rates from 96.1% to 97.6% on LIBERO and from 62.9% to 73.2% on LIBERO-Plus. With demonstration-based training and contextual guidance at inference, LIBERO-Plus success further increases from 73.2% to 75.2%. 

---
# AXE: AlpaSim E2E Closed Loop Challenge Technical Report 

**Authors**: Joonhee Lim, Seonmyeong Lee, Sihwan Hwang, Hanbin Jang, Hiwon Shin, Dogun Kim, Hyelin Jo, Hyeonjun Jeong, Yongjae Lee, Junseong Kwak, Junhyeok Park, Satyam Mishra, Juyeb Shin, Uisung Kim, Hyeonseo Kim, Saparkhan Kassymbekov, Hailemichael Molla Gashaw, Abi Rahman Syamil, Mingyu Kim, Hyeongjun Kim, Yeina Lee, Moogeun Park, Dongsuk Kum  

**Link**: [PDF](https://arxiv.org/pdf/2611.30511)  

**Abstract**: This report presents a comprehensive pipeline for developing an end-to-end (E2E) autonomous driving model for the Physical AI AV Track. The framework consists of data curation, HD map auto-labelling, data-driven imitation learning, and simulator-based reinforcement learning. To ensure safety in the closed-loop setting, we introduce a selection-based E2E planning approach that leverages object detection and BEV segmentation to filter out trajectory candidates likely to cause collisions or off-road driving. Trained on NVIDIA NuRec and the NVIDIA PAI-AV dataset, our method scores a fixed vocabulary of trajectory candidates for collision, drivable-area compliance, and progress learned from simulated rollouts, and selects the best candidate that passes the filter. This design highlights the effectiveness of combining E2E planning with perception-based trajectory filtering for safe and reliable closed-loop autonomous driving. 

---
# De Novo Generation of Odorant Molecules with Targeted Olfactory Receptor Activation Patterns 

**Authors**: Kexin Zhang, Manhao Guan  

**Link**: [PDF](https://arxiv.org/pdf/2611.30509)  

**Abstract**: Odor perception arises from a combinatorial code in which each molecule activates a specific subset of olfactory receptors (ORs), and the resulting activation pattern determines the perceived smell. We present a framework for de novo generation of molecules predicted to match target multi-receptor activation profiles. We first construct a panoramic binding matrix over 20 ORs, then cluster it to discover representative activation patterns, and finally formulate a structured selectivity loss, combining pairwise ranking, distribution matching, and complexity constraints, to steer a molecular generator via reinforcement learning. Across nine target profiles of varying complexity, our method achieves high predicted selectivity within 18000 oracle calls and provides the largest gains on profiles that are rare under the prior, especially broad-spectrum and sparse-activator targets. 

---
# Test-Time Latent Computation to Improve Visual Perception in Multimodal Large Language Models 

**Authors**: Chen Yang, Zheng Fang, Wenqi Fan, Jun Xia  

**Link**: [PDF](https://arxiv.org/pdf/2611.30480)  

**Abstract**: Recent work on Large Language Models (LLMs) suggests that reasoning capabilities can be enhanced at test time not only through longer token-level reasoning traces, but also through additional computation in latent space. This raises a natural question for Multimodal Large Language Models (MLLMs): can additional computation in latent space improve visual perception at test time? In this work, we first verify that MLLMs contain stable reusable layers that can be re-forwarded at inference time without disrupting normal generation. Based on this observation, we propose \textbf{Lo}ok \textbf{Ba}ck in the \textbf{L}atent \textbf{S}pace (\textbf{Lobals}), a training-free decoding method that identifies the contiguous layer window with the strongest visual-token attention and re-forwards through it, allowing the model to revisit visual evidence in latent space. We demonstrate that this latent-space revisit strategy can improve MLLMs' performance across diverse multimodal settings, including general vision-language tasks, visual hallucination tasks, and vision-centric tasks. 

---
# OVOGO: Offline Value-Guided On-Policy Generative Optimization for Flow-Based Vision-Language-Action Models 

**Authors**: Rongzheng Yang, Jiaming Guo, Yuxuan Guo, Zihao Zhang, Yunkai Gao, Siming Lan, Shaohui Peng, Rui Zhang, Tianyun Ma, Xing Hu, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30477)  

**Abstract**: Flow-based vision-language-action models can benefit from robot experience collected during deployment, but repeatedly acquiring new physical interactions for policy improvement is costly. Offline reinforcement learning provides a natural way to reuse collected experience, while flow-based policies offer an additional opportunity because fresh generation trajectories can be cheaply resampled from the current policy at replay observations. This motivates separating offline value learning from on-policy optimization of the generative process. We propose OVOGO, an offline reinforcement learning framework that realizes this separation for flow-based VLAs. A central challenge in this formulation is to learn a critic that remains grounded in logged robot experience while providing useful value signals for newly generated actions. OVOGO introduces a Return-Anchored Ranking Critic that combines in-sample distributional value learning with guarded local ranking, improving discrimination among generated actions without assigning them artificial return targets. The frozen critic then provides group-relative value signals for selected-transition optimization of fresh action candidates sampled from the current flow policy. Experiments on RoboTwin 2.0, DexJoCo, and LIBERO, as well as a real-robot dexterous manipulation task, show that OVOGO improves the initial VLA across all four settings and achieves higher success rates than the evaluated offline adaptation and value-guided baselines. These results suggest that offline robot experience can be repeatedly reused for value learning while inexpensive generative sampling supports policy optimization, providing an effective approach to experience-based post-training of flow-based VLAs. 

---
# One-shot identification of analytic systems 

**Authors**: Nicolas Boullé, Diana Halikias, Samuel E. Otto, Alex Townsend  

**Link**: [PDF](https://arxiv.org/pdf/2611.30470)  

**Abstract**: We study when the coefficients of a linearly parameterized analytic system can be identified from a single input-solution pair. Given a finite dictionary of analytic operators, each admissible input induces a finite-dimensional linear recovery map from coefficients to observed outputs. Coefficient recovery is possible exactly when this map is injective. We define the degenerate input set as the set of inputs for which injectivity fails and prove a zero-one law: either every admissible input is degenerate, or the degenerate set is negligible in the sense that it is compactly slice-null and has zero measure under every nondegenerate Gaussian measure. We then apply this framework to the recovery of matrices, analytic continuous-time dynamical systems, and differential operators. For several ODE and PDE classes, we further characterize the degenerate sets explicitly and identify deterministic choices of inputs that guarantee recovery. 

---
# Implicit Neural Representation for 3D Radio Map Compression in Low-Altitude Environments 

**Authors**: Jiaming Cheng, Wei Chen, Yuchen Li, Bo Ai, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30445)  

**Abstract**: Low-altitude wireless networks require accurate 3D radio maps (RMs) to support environment-aware communication. However, due to the extensive spatial coverage and multi-modal channel representations, 3D RMs usually incur prohibitive storage and transmission overhead, making efficient compression highly desirable. In this paper, we propose NeRM3D, an implicit neural representation (INR)-based framework for compact 3D RM compression. Specifically, NeRM3D represents the entire 3D RM as a lightweight neural network that maps discrete height values to corresponding RM slices, thereby encoding the RM into a compact neural network. To effectively model the volumetric spatial structure of 3D RMs, NeRM3D jointly integrates horizontal embeddings with height-aware modulation, enabling efficient extraction and fusion of spatial correlations across different vertical layers and channel modalities. Experimental results demonstrate that NeRM3D achieves competitive compression performance with low complexity and decoding latency, making it suitable for resource-constrained devices. 

---
# MF-EDA: Joining Membership Functions to Adapt the Number of Distributions in Gaussian Estimation-of-Distribution Algorithms 

**Authors**: Jun Zhang, Tong Qian, Hua Ding, Wei-Jie Yu, Xiao-Fang Liu, Jian-Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.30436)  

**Abstract**: An estimation-of-distribution algorithm (EDA) usually samples from one Gaussian, whose peak sits where most selected individuals are. On a landscape with two funnels that is not where the best solutions are: the wider funnel supplies most of the selected individuals, and the deeper funnel, which holds the global optimum, is lost early. This paper builds the search distribution as a neuro-fuzzy system builds a model: from Gaussian membership functions joined by membership degrees. In every generation the proposed MF-EDA searches the principal directions of the selected individuals for the direction along which they are bimodal, describes them on that direction by one or two membership functions according to the Bayesian information criterion and a separation test, joins each membership function to a local Gaussian in the other directions, and samples from the normalised union. A membership function has height one whatever the number of individuals under it, so a detected funnel receives samples according to its width. A two-region selection model shows that sampling shares proportional to the head count, as in a Gaussian mixture, compound the early disadvantage of the deeper funnel, and that the union does not. With one membership function the algorithm is its single-Gaussian baseline; the steps cost no fitness evaluations but make a generation 8 to 17 times slower to compute. On 36 double-funnel settings the component raises the successful runs of a rank-weighted Gaussian EDA from 805 to 1712 of 1800. On the Lunacek double funnel in 30 dimensions it raises that EDA from 4 to 49 and AMaLGaM from 0 to 47 successes in 50 runs, while results on single-funnel functions and on ten multimodal BBOB functions do not change. Restarted CMA-ES remains cheaper in evaluations on the standard double funnel, and the component stops helping when the global funnel is much narrower than the other. 

---
# Disclosure Bias in AI Reviewers 

**Authors**: Jens Leysen, Noah Daniëls, Marco Favier, Bart Goethals  

**Link**: [PDF](https://arxiv.org/pdf/2611.30410)  

**Abstract**: AI models increasingly help to review papers, and many conferences and journals now require authors to state how they used AI. Does this AI-use statement change how an AI reviewer rates a paper, and could it change which papers are accepted? In a preliminary study, we varied only the AI-use statement of 150 ICLR 2027 submissions, in up to nine versions adapted from real statements: from no AI, through writing help and coding, to an AI research agent and an AI author. Six AI reviewers rated the versions: the decision model Jev and the language models Claude Haiku 5.5, Claude Sonnet 5.5, Gemini 3.8 Flash, Gemini 3.1 Pro and Gemini 4 Argon. Across 23,652 reviews, we find that every reviewer rates papers lower on average when the statement says AI did all the work than under a statement of no AI use. The AI research agent statement lowers the rating significantly for Jev, Sonnet and Argon. We find no evidence of a disclosure bias for AI assistance: for writing help, coding and research help, no reviewer's rating differs significantly from its rating under a statement of no AI use. Whether the observed penalty changes which papers are accepted depends on more than the size of the penalty: on how finely and how consistently the reviewer separates the papers near the acceptance boundary, on whether the penalty is spread over all papers or concentrated on a few, and possibly on how many other papers disclose AI use. 

---
# TAP: Policy Interfaces for Diffusion Language Models 

**Authors**: Preeti, Ankita Kushwaha, Kiran Ravish, Pawan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.30406)  

**Abstract**: Diffusion language models denoise a masked canvas, making them a natural setting for selective generation decisions such as answering, waiting, clarifying, or refusing. We introduce TAP-DiLM, which adds a typed policy variable to masked diffusion LMs, and study how this policy should be attached to the denoising process. Across BERT-base, MDLMOWT, LLaDA-8B, and Dream-7B, we compare shared-vocabulary slots, dedicated slot heads, and CLS-based external heads. The results reveal a clear interface-backbone interaction: dedicated heads provide robust policy learning across scales, while sharedvocabulary heads can be effective when the LM-head geometry is favorable. A simple LMhead sharpness probe predicts when sharedhead policy internalization is reliable, giving a practical diagnostic for interface selection. Causal clamp and slot-injection tests show that dedicated policy slots can influence body denoising through feedback, especially at 8B scale. Safety-refusal experiments further show that TAP-DiLM supports competitive typed refusal behavior when trained with diverse unsafe prompts. 

---
# Mind the Gaps: State-Guided Supervision Allocation for Object Detection Dataset Distillation 

**Authors**: Jiehui Xie, Zhengxun Jin, Jiahao Zhang, Yuxuan Luan, Dongshen Han, Caiyan Qin, Sung-Ho Bae, Chaoning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30369)  

**Abstract**: Recent advances in dataset distillation have enabled large-scale training datasets to be compressed into compact synthetic sets. However, dataset distillation for object detection remains challenging, as it requires preserving diverse classification and localization signals under limited image budgets. Existing approaches often emphasize individual-image informativeness without explicitly accounting for supervision accumulated across the distilled dataset, potentially leading to redundant supervision and uneven class-scale coverage. To address this limitation, we propose State-Guided Supervision Allocation (SGSA), a framework that formulates object detection dataset distillation as a dataset-level progressive supervision allocation problem. SGSA establishes class-scale supervision quotas and maintains a global state that tracks detector predictions accumulated from previously selected synthetic images. The remaining gaps between the target quotas and accumulated supervision guide adaptive object sampling, detector-guided canvas refinement, and marginal-gain-based candidate selection, encouraging complementary supervision across images. On MS COCO, SGSA outperforms a matched OD3 baseline by 3.73-4.06 mAP points across three image budgets (0.25%-1.0%). On PASCAL VOC, SGSA outperforms the strongest evaluated baseline by 4.53-6.90 mAP50 points across three image budgets. These results demonstrate the effectiveness of dataset-level supervision allocation guided by accumulated detector feedback under limited image budgets. 

---
# RepoClean: Evaluating Coding Agents on Repository-Level Obsolete Code Cleanup 

**Authors**: Hongyixuan Yuan, Qixing Zhou, Pengcheng Huang, Qingshuo Ni, Runsong Zhao, Shaokai He, Xiangpeng Li, Deqiang Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30350)  

**Abstract**: Large language model (LLM) agents have demonstrated strong repository-level programming capabilities. However, as software evolves, repository code may become redundant or obsolete, increasing maintenance costs and potentially introducing functional issues. Agents must therefore identify and remove unnecessary code safely, yet autonomous repository-level code cleanup remains insufficiently evaluated in isolation. To evaluate this capability, we introduce RepoClean, a benchmark of 200 cleanup tasks from 34 popular open-source Python repositories. RepoClean is constructed by automatically reintroducing deleted code from real repository histories into newer versions, preserving current implementations and retaining instances where previously passing tests consequently fail. To restore existing functionality without new feature development, agents must use repository interfaces and cross-file dependencies to determine which historical code to remove and which implementations to preserve. Evaluation of seven models shows that even the best-performing model, GPT-6 Astra, resolves only 23% of tasks. Further analysis reveals that agents remove substantially less obsolete code when cleanup is not explicitly requested. RepoClean provides a foundation for systematically studying these cleanup challenges and advancing coding agents toward autonomous software maintenance. 

---
# HEART: Hierarchical Expert Adaptive Refinement Transformer for ECG Representation Learning 

**Authors**: Junho Song, Jong-Hwan Jang, Joon-Myoung Kwon, Yong-Yeon Jo  

**Link**: [PDF](https://arxiv.org/pdf/2611.30298)  

**Abstract**: Learning robust representations from electrocardiograms (ECGs) is challenging because repeated cardiac cycles coexist with subtle and sparse diagnostic cues in local morphology and rhythm-level context. Existing ECG foundation models commonly rely on Transformerbased patch encoders with uniformly shared dense refinement pathways, limiting adaptive processing across heterogeneous ECG tokens. We propose HEART (Hierarchical Expert Adaptive Refinement Transformer), an ECG foundation model that introduces stage-specific conditional expert refinement during self-supervised pre-training. HEART places a local-level MoE over patch-level representations before temporal contextualization and a global-level MoE over context-integrated token representations after self-attention. Token-level Top-1 routing allows different patches or contextual tokens within the same ECG to follow different expert pathways. On public ECG benchmarks, HEART achieves strong performance under both linear probing and fine-tuning, with particularly large frozen-encoder gains on PTB-XL Form and Rhythm (+6.76 AUROC points each over the strongest protocol-reproduced baseline). Routing analyses provide qualitative evidence of complementary local and global expert specialization. We release our code and pretrained models at this https URL. 

---
# Uncovering Separable and Composable Skill Mechanisms in Vision-Language Models 

**Authors**: Miaoyu Li, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.30290)  

**Abstract**: Recent work has sought to localize the internal mechanisms underlying visual reasoning skills in vision-language models. These methods largely overlook separability and composability in both mechanism identification and evaluation. We propose separability and composability as two important desiderata for skill localization. Separability requires identified mechanisms to contribute primarily to their target skills while having limited effects on other skills, enabling precise diagnosis and targeted intervention. Composability requires skill mechanisms to be combinable to support complex tasks, allowing a small set of mechanisms to account for broader capabilities. In this work, we propose a multi-skill interpretability framework DISCO, built around these two properties. For identification, we propose Separable Head Selection, which selects skill-specific attention heads based on their contributions across skills. For evaluation, we are the first to design metrics for measuring separability and composability of identified heads. Extensive experiments on three VLMs and four fundamental visual reasoning skills demonstrate the effectiveness of DISCO. It achieves the highest separability, outperforming the strongest baseline by 24.80, 18.38, and 13.75 points across the three models, respectively, while also achieving the highest composability on all three models. 

---
# Evaluating Persona Consistency in LLM User Simulators Without External Reference Annotations 

**Authors**: Anping Zhang, Yang Tan, Yipeng Liu, Clive Bai, Blaze Chen, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.30280)  

**Abstract**: Standard evaluations assess LLM-simulated personas against an external ground truth, such as community trait annotations, but that reference is unavailable when the persona is inferred only from a short, task-oriented chat. We ask how persona consistency can be evaluated without such a reference, and propose a self-anchored internal-consistency diagnostic: the persona describes itself once and acts twice. A Likert self-report supplies a soft internal anchor; two behavioral profiles come from constrained scenario choices scored without a judge and from open-ended responses judged by an LLM. Consistency is the correlation between the self-report and each behavioral profile, in a shared six-facet BFI-2 space. The framework compares persona-construction strategies by assessing how consistently their elicited responses reflect their self-reported traits. Across ordinary-user chats (WildChat and LMSYS) and well-documented personas with community annotations (the Personality Database, PDB), profile-enriched representations agree with the anchor more closely than a raw-dialogue baseline. On PDB, this self-anchored ordering reproduces the community-annotation ordering. The pattern is stable across answering and judge models. 

---
# A New Perspective on ANN-to-SNN Conversion: Hardware-Aligned Multi-Threshold Neuron Dynamics for Neuromorphic Inference 

**Authors**: Shaohong Gu, Jiyuan Zhang, Yimeng Shan, Chenxiao Dou, Wei Hua, Malu Zhang, Honglin Cao, Choonghyun Lee, Jingwei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30277)  

**Abstract**: By accumulating spikes over multiple timesteps, rate coding approximates the real-valued activations of an ANN, a principle that underpins numerous ANN-to-SNN conversion methods. However, the converted SNNs typically incur loss in accuracy compared to the original ANN. Existing methods either increase a number of timesteps or introduce more complex neuronal dynamics to reduce this accuracy gap. Yet, both approaches ultimately lead to the same outcome: redundancy either in information or in computation, thereby suffering from efficiency degradation. To address this issue, we propose \textbf{PRIME} (\underline{P}otential-\underline{R}esidual \underline{I}ntegration for \underline{M}ulti-threshold \underline{E}ncoding), a multi-threshold concurrent spiking neuron framework redesigned from a hardware data representation perspective:a) We introduce the Residual Membrane Potential Amplification (RMPA) mechanism, which replaces rate coding with Integration Coding by actively amplifying the residual membrane potential. RMPA enables the binary representation of spikes to carry substantially more information, achieving near-lossless conversion with as few as two timesteps.b) Complementarily, we implement concurrent spike firing in multi-threshold neurons, where spikes are decoded directly from the binary representation of floating-point data, eliminating membrane potential--threshold comparisons and replacing multiplications with lightweight bit-shift and accumulation operations that align with hardware computation primitives. At $T=2$ and $D=3$, we achieve competitive performance across both vision and language benchmarks with only 40.6\% of the FP32 bit-transmission cost and 13.4\% of the FP32 multiplication energy consumption. The PRIME framework establishes a new design principle for efficient ANN-to-SNN conversion, bridging the gap between algorithmic efficiency and hardware constraints. 

---
# Attributing Security Outcomes in LLM-Agent Evaluations: The Role of Content-Free State 

**Authors**: Abdelmonaim Bounite, Zangir Iklassov  

**Link**: [PDF](https://arxiv.org/pdf/2611.30259)  

**Abstract**: Security benchmarks for LLM agents record whether each attack succeeded and compare agents, attacks, and defenses by attack success rate (ASR). Such comparisons are valid only if the agent, the attack, and the defense cause each verdict. We show that verdicts also depend on content-free state: details of a run that carry no information about the system under test, such as the Python hash seed, the order of tools or evidence, and the scoring of failed runs. We require that this state influence no verdict (decision noninterference) and treat it as untrusted input, because evaluators and requesters of LLM-based gates can choose it. Our framework, TARE (Testing Attribution of Reported Effects), attributes a reported effect to the agent, attack, or defense only if it exceeds the measured influence of content-free state. On Agent Security Bench (ASB), runs with a fixed hash seed are identical for seven agents from four model families, yet changing the seed changes up to 17% of an agent's verdicts while the ASR moves by at most 5 points. The seed acts through the tool order: sorting the tool list instead of reverse-sorting it raises the ASR by up to 19 points, because ASB's naming convention places attacker tools first. When the same evidence is reordered, six of seven LLM judges change their decision on 17-91% of 98 proposed actions. Across eighteen agent-defense combinations, TARE's worst-case test attributes effects to two of three defenses, one of them in opposite directions on two agents, and a test fixed in advance confirms a 4.8-point decrease from the third; in the attributed decreases, 35-69% of the stopped attacks end without any tool call, which ASB counts as a failed attack. Reproducible verdicts are thus not necessarily attributable, and security evaluations of LLM agents should report the influence of content-free state. We release TARE, the ASB patch, and all verdict tables. 

---
# Voyag: Joint Multi-Intent Detection and Execution-Relation Classification for In-Vehicle Voice Assistants 

**Authors**: Sen Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.30253)  

**Abstract**: An in-vehicle voice assistant must decide not only which intents an utterance contains but also how co-active intents may be executed: concurrently, in a fixed order, or not at all without arbitration. We present Voyag, a single-forward decision model that jointly predicts ten controlled vehicle-intent slots and a three-way execution relation (parallel, sequential, mutually exclusive) over each co-active intent pair. Voyag combines a multilingual ModernBERT-style encoder (mmBERT-base, 306.9M parameters), a typed per-slot choice head, and a lightweight GroupCalibration relation head (1.61M parameters) that consumes marker hidden states, pooled context, explicit resource-requirement profiles, and symmetrized outer-product conflict features. All objectives are optimized in one joint loss: soft cross-entropy over slot choices, focal cross-entropy over relations, and a mutex-consistency penalty. On a synthetic Chinese vehicle-life benchmark with semantic scene-level splits and a leakage audit (30,017 / 3,473 / 5,000 groups), validation-only selection yields 0.9893 slot accuracy and 0.0526 expected calibration error over 50,000 slot decisions. Conditioned on gold intents, the relation head reaches 0.8548 accuracy, 0.8473 macro-F1, and 0.6062 mutually-exclusive recall over 4,000 pairs; end to end with one encoder forward it reaches 0.8530 accuracy and 0.8464 macro-F1. Against a two-stage rule pipeline, the aggregate end-to-end gain is small and borderline, so we claim no aggregate advantage; the learned head's advantage lies in sequential and parallel recall where surface rules are blind. We report rule baselines with a tautology disclosure and a held-out thermal conflict failure. 

---
# StutterStigma-TR: A Novel Turkish Dataset and Target-Aware BERT Framework for Stuttering-Related Stigma Detection 

**Authors**: Muhsin Dölek, Arezoo Sadeghzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2611.30252)  

**Abstract**: Computational research on disability-related ableist harm remains limited, and condition specific resources for stuttering-related stigma are particularly scarce. This study introduces StutterStigma-TR, a Turkish dataset of 6,019 public X/Twitter posts annotated as Stigmatizing, Non-stigmatizing/contextual, or Supportive using a speech and language therapy-informed protocol with independent lived-experience evaluation. We also propose StutterStigma-BERT, a BERTurk-based target-aware framework that combines explicit target marking, target-span representation, target-conditioned context, character-level and handcrafted orthographic information, and gated discourse cues. The framework is evaluated in complementary three-class and binary settings using controlled three-seed ablation analyses and final 20-seed robustness evaluations. Under the common training protocol, Gated Discourse-OrthoTarget trained with cross-entropy is selected as the final configuration for both tasks based on internal-validation Macro-F1. Across 20 random seeds, the final three-class model achieves a mean Macro-F1 of 73.54%, while the final binary model achieves 85.17% using validation-selected decision thresholds. Task adapted harmful-language comparators are additionally evaluated under the common training protocol, and the proposed model obtains the highest mean Macro-F1 among them. For the primary three-class task, matched backbone analyses show higher internal validation Macro-F1 with BERTurk than with TurkishBERTweet or Turkish Mamba, while substantial train-validation separation persists across all three backbones. Separate direct-transfer and zero-shot experiments examine performance without StutterStigma-TR supervision. Across the task-adapted, direct-transfer, and zero-shot comparisons, the condition-specific supervised systems retain the strongest aggregate performance. 

---
# PixelOne: From Options to Pixels in a Single Forward for GUI Grounding 

**Authors**: Kanghui Tian, Shuai Dong, Yuhan Zhu, Congcong Wang, Ji Zhao, Tianxiang Jiang, Zhixiong Zhang, Shengyuan Ding, Yi Wang, Jiaqi Wang, Yu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.30235)  

**Abstract**: GUI grounding requires precise clicks at low latency. Recent models such as GUI-Decisions and GUI-Swift predict coordinates in one forward pass using fixed global bins with local refinement. However, screenshot downsampling removes detail needed to locate small controls, while fixed bins cover larger pixel ranges at higher resolutions. We introduce PixelOne, which transfers spatial knowledge from a search teacher to a student that predicts clicks directly from screenshots at native resolution. The teacher uses VideoJEV's text-option scorer to select rows and columns and progressively zoom into candidate regions. The student inherits the teacher's parameters and matches its distributions over explored regions through spatial distillation alongside box supervision. At inference, the same decision head selects a visual token, and a lightweight pixel head predicts the click within it, all in one forward pass. A click-aligned loss encourages pixels inside the target box to score above those outside. Across four benchmarks, PixelOne-Pro-35B-A3B achieves 83.8% mean accuracy, the highest among the evaluated single-forward grounders, with a mean latency of 0.43 s per target. 

---
# RoboAda: Adapting Frozen Robot Policies at Test Time to Physics They Cannot See 

**Authors**: Yunbei Zhang, Zijian Jin, Shuaicheng Niu, Baian Chen, Weirui Ye, Jihun Hamm, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2611.30209)  

**Abstract**: Pretrained robot policies decide what to do from camera images, yet some of the physics that decides whether their plans succeed is not visible. An object may look light but be 100 times heavier, and a weakened joint or a lagging controller leaves no trace in the image. When the physics at deployment differs from training, the plan is often still right but is executed as if nothing had changed. Training on such variations in advance cannot handle unseen changes, while adapting once deployed either changes the plan, leaving the miscalibrated commands in place, or updates a robot model by gradient steps that do not know how reliable they are and can make the robot worse even when nothing has changed. We propose RoboAda, which keeps the policy and its plan unchanged and corrects the commands that carry it out by inferring at test time how the robot turns commands into motion. The key observation is that the robot's own motion reveals the change: after each action chunk, an inverse-dynamics model of the original robot converts the observed motion into the commands it would have needed, yielding free labels at every step. Competing hypotheses of how the robot responds, including that nothing has changed, are weighted by how well they predict these labels, and the next chunk is sent with the commands that best realize the plan under them. RoboAda needs no reward, no gradient step and no data from the changed robot. We evaluate RoboAda with five frozen policies on LIBERO, two world action models and three vision-language-action models. Under eleven physical shifts, it more than doubles their average success, from 27.8% to 62.9%. It beats the best of seven test-time adaptation methods by 17.3 points on Cosmos Policy, and largely preserves the policies' success when nothing has changed. On bimanual RoboTwin 2.0 in joint space, it raises success under joint-target offsets from 24.9% to 78.8%. 

---
# What Makes Good Data for Training Coding Agents? A Systematic Study on SWE-Gym 

**Authors**: Yikai Zhang, Mai Xu, Yunbei Zhang, Qirui Jin, Kenan Li, Yuwen Jia, Jinwei Yao, Xiangyu Zhang, Jielun Tan, Jin Pan, Zijian Jin, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.30207)  

**Abstract**: Reinforcement learning (RL) trains coding agents on software tasks whose hidden tests decide the reward, yet task collections check these tests only against the reference fix. We ask what makes a task good training data, focusing on the correctness of its reward labels. We audit SWE-Gym with two LLM families, and a suspected error counts only if execution reproduces it. We find that unsolved often means mislabelled: 74% of the tasks that a 9B base model never solves have a test that rejects correct fixes (a false negative), versus 20% of those it sometimes solves. We repair or remove such tasks, and the audited tasks that remain train better policies. At equal size and matched difficulty, models trained on them score 2.4 and 2.7 points higher on held-out tasks from SWE-bench Verified and SWE-bench Pro than models trained on the original tasks, whereas simply dropping the never-solved tasks shows no such gain. We further find that wrong patches that pass (false positives) make up about a quarter of all passing patches, and that new tests that catch a quarter of them do not change held-out scores detectably. A pass rate shows whether a task yields reward, not whether that reward is right, so the reward labels themselves must be checked. To this end, we release SWE-Gym-Verified with its 1,087 audited tasks, every verdict with its executed evidence, our training sets, and the full audit pipeline. 

---
# PCC: Pivotal Channel Compensation for Layer Pruning of Large Language Models 

**Authors**: Xiangqian Wang, Yucheng Zhang, Zhaohong Jia, Kun Jing, Chunchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.30183)  

**Abstract**: Treating independent layers as the minimal pruning unit allows layer pruning to efficiently eliminate many redundant parameters and achieve high compression rates. Layer pruning has become a promising and prevalent structured pruning technique for large language models (LLMs). However, multi-layer removal can lead to significant performance degradation for LLMs. Although some works have explored various compensation methods for pruned layers, they are inefficient and costly. Observing that the expression of a large proportion of information is concentrated on a small number of pivotal channels, we propose a novel post-pruning compensation method for layer pruning, called pivotal channel compensation (PCC). PCC can achieve significant performance improvement by compensating for a small number of channels through learnable linear transformations at the pruning interface. Besides, we propose a new evaluation metric, called channel-wise average performance improvement (CAPI), which can measure the efficiency of channel-wise compensation. Our experimental results across various LLMs demonstrate that PCC can achieve significant and consistent improvements and high CAPI on language modeling and zero-shot question answering for the state-of-the-art cross-layer and consecutive-layer pruning methods. Compared with existing compensation methods, PCC can achieve competitive results with fewer additional parameters and less training cost. 

---
# Trustworthy agentic lunar robotics: safety-constrained contingent planning with a foundation-model planner 

**Authors**: Bingqi Huang, Bingchuan Wei, Yingkai Cai, Zhaokui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30179)  

**Abstract**: Sustained lunar-base operations require robots that restore equipment and deliver cargo under conditions unforeseen before deployment. Flown space autonomy executes pre-validated procedures and enters a safe hold outside their scope, whereas foundation models synthesize plans on site that cannot be verified before execution. We formulate these operations as safety-constrained contingent planning with irreversible actions, erroneous records, three-valued physical evidence, and costly, deadline-constrained sensing. We propose asymmetric knowledge-level replanning (AKR), which treats the planner as an untrusted black box: reversible actions run on the planner's assumptions, and irreversible actions run only after a minimum-cost sensing cover verifies every precondition. Under sound measurements, no irreversible action executes on a false precondition for any planner or record, the sensing cost before an irreversible action is bounded below for every sound knowledge-based policy, and execution with a complete planner ends in a verified goal or a certified hold. AKR executed no hazardous action in 21,000 Monte Carlo trials with sound observations or in 58 full-stack runs; the FF-Replan acting rule executed one in 7% of those trials. The median AKR run took 1.10 times the fastest run on the same mission, against 1.50 under the K-replanner acting rule. AKR completed 20 of 21 contingency and sequential-fault runs with an unchanged task model; the original PLEXIL and EUROPA$\rightarrow$PLEXIL procedures completed none. 

---
# Audio-Visual Rearrangement 

**Authors**: Ruohao Guo, Jiatong Ye, Chenwei Dong, Jinxing Zhou, Dantong Niu, Guangyao Li, Zhenbo Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.30171)  

**Abstract**: We introduce Audio-Visual Rearrangement (AVR), a novel task in which an agent hears a spoken request, retrieves a suitable object, and delivers it to the requesting user or an instruction-specified destination. AVR requires the agent to understand the spoken request and acoustically localize the speaker for delivery. To support this task, we construct a dataset covering three query types: explicit target specification, speaker-relative spatial references, and implicit user intent, spanning eight subtypes with diverse object selection and delivery requirements. Moreover, we propose a framework that integrates a motion-aware spatial audio module into an audio-visual large language model to encode spatial cues from the spoken request and update them as the agent moves. We further incorporate auxiliary supervision for speech understanding, spatial audio perception, and visual grounding through multitask instruction tuning to improve action prediction. Experiments establish baseline performance in both seen and unseen environments, laying the groundwork for future research on embodied intelligence and human-robot interaction. The code will be publicly available soon. 

---
# FiRE: Fine-grained Ranking Evaluation for Machine Translation 

**Authors**: Wenyang Gao, Yinghao Yang, Xi Jin, Jing Li, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30168)  

**Abstract**: Developing reliable machine translation (MT) systems hinges on our ability to distinguish superior translations from inferior ones. However, existing evaluation paradigms, whether limited to coarse overall rankings or misaligned with human preferences, fail to deliver interpretable, fine-grained feedback in reference-free settings. We present a Fine-Grained Ranking Evaluation method (FiRE) that leverages off-the-shelf large language models to perform criterion-driven pairwise comparison across three complementary dimensions: faithfulness, fluency, and consistency of style, instead of producing a single holistic judgment. To enable rigorous meta-evaluation of evaluation paradigms in the absence of any suitable testbed, we construct the first human-annotated, reference-free benchmark for fine-grained ranking evaluation, achieving substantial inter-annotator agreement. Through meta-evaluation on this benchmark and existing MQM datasets, FiRE demonstrably outperforms regression-based and error-analysis metrics in aligning with human comparative judgments, while providing more informative insights into translation quality. Finally, our examination of LLM evaluator biases (position and self-enhancement) and their handling of tied cases offers guidance for more nuanced MT evaluation. Code and benchmark resources are available at this https URL. 

---
# Revisiting Continual Learning in Language Models: A Three-Interface Diagnosis of Stability and Plasticity 

**Authors**: Keuntae Kim, Beomseok Lee, Hyunwoo Kim, Yong Suk Choi  

**Link**: [PDF](https://arxiv.org/pdf/2611.30151)  

**Abstract**: Continual learning in language models is commonly studied in the incremental learning (IL) setting, where a model learns a sequence of tasks and is evaluated through its final classifier or language-model (LM) head. This interface, however, does not indicate where forgetting occurs: a drop in accuracy may reflect representation degradation in the backbone, readout mismatch in the head, or both. We therefore argue that backward transfer (BWT) and forward transfer (FWT) should be interpreted as interface-dependent quantities rather than as single modellevel properties. We study three evaluation interfaces: the trained head, which is updated sequentially with the backbone; a probe newly trained on frozen backbone features; and a classifier-free backbone diagnostic. They measure deployed performance, recoverability under a new readout, and representation separability, respectively. Across class-incremental experiments with encoder and decoder language backbones, plus task-incremental sanity checks, conclusions about forgetting and plasticity change with the interface: the trained head often indicates severe forgetting, the probe near-complete recoverability, and the backbone diagnostic intermediate representation drift with a consistent negative relation between stability and plasticity. This relation holds for clustering-free diagnostics such as nearest-class-mean accuracy and across five clustering algorithms and three metrics. Finally, Just LM-Head Tuning (JLT), a head-only intervention on a fixed incrementally trained backbone, recovers 96-99% of the gap to joint training for replayand distillation-based methods but only about three quarters for plain sequential fine-tuning, suggesting that many failures attributed to catastrophic forgetting reflect head-backbone mismatch plus partial representation drift. Our results call for reporting where forgetting is measured, not only how much. 

---
# LeadVLA: Learning Follower-Aware Route Execution for Robot Leading 

**Authors**: Tianle Zeng, Hanjing Ye, Bingyi Xia, Yuchen Zhou, Zibo Zhang, Yufei Huang, Yonggen Ling, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30149)  

**Abstract**: Robot leading requires executing a route while continuously adapting to a designated follower. Existing robot-leading systems often mediate follower understanding and physical route execution through explicit states, planners, or modular interfaces; directly learning this coupling through end-to-end trajectory prediction can instead entangle route execution with follower response. We introduce LeadVLA, a structured multimodal policy that preserves the route as an explicit Path reference while learning a Pace for route progress and a Residual for local follower- and scene-conditioned adaptation. To make this problem trainable and measurable, we develop LeadInfra, a configurable Unreal Engine-based closed-loop data-generation infrastructure from which we instantiate an approximately 4M-sample training corpus, and LeadBench, a standardized benchmark with 108 fully held-out closed-loop episodes. LeadVLA achieves the highest task success (39.6%) among evaluated robot policies on LeadBench while maintaining strong route fidelity and follower responsiveness, and transfers to unseen real-world environments without real-world policy fine-tuning. Project Page: this https URL 

---
# Grokking as a Subspace Search: Computing the Destination Before Training 

**Authors**: Sunjae Jeong, Jaeseung Kim, Hyoseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30116)  

**Abstract**: Grokking is the phenomenon in which a neural network generalizes long after it has memorized its training data. Many explanations have been proposed for the delay between memorization and generalization. However, none of them can say before training where the network is heading, so they cannot be tested against one another. We show that on group tasks this destination can be computed from the task itself before training begins. It is a low-dimensional subspace of the network's embeddings, which we call the right subspace, and we prove that it is essentially unique. Knowing the destination makes our hypothesis falsifiable: the delay is a search for the right subspace, and generalization begins when the network finds it. When we give the network the right subspace from the start, the delay disappears. When we take the right subspace away, accuracy stops at a ceiling that we compute before training, and on multiplication the network cannot generalize at all. When we train the network normally, its embeddings move toward the right subspace by themselves, long before test accuracy rises. Finally, interventions during training show that this search sets the length of the delay, and that noise in the optimization does not. On every group task that we tested, grokking is a search for the right subspace, and the delay is the time that this search takes. 

---
# RoboThink: Let the Agent Think, Let the Policy Act 

**Authors**: Yunbei Zhang, Zijian Jin, Janet Wang, Xingchen Zhao, Zhenyu Zhang, Weirui Ye, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2611.30111)  

**Abstract**: Robot policies act in real time but fail without knowing why, whereas a reasoning agent that watches the robot can often tell what went wrong. Existing systems either hand the agent control, which takes minutes per episode, or let it send instructions. Yet a policy follows a new instruction only from a fresh start, since a half-finished task leaves it in states outside its training data. Nor can an instruction fix how the robot moves. If a drift keeps pulling the hand to the left, "push right, slightly" is not a task the policy has learned. We propose RoboThink, in which the agent thinks and the policy learns to act on the agent's guidance. An instruction channel tells the policy what to do. When the user says "the milk, not the butter" after the robot has started, the agent rewrites the instruction and the robot undoes its own motion, with no training. A feedback channel tells it how to move. It is taught from the agent alone with agent-labelled DAgger, role tokens and counterfactual feedback, and is read only by the policy's action module. On LIBERO, RoboThink with Claude Opus 5.5 raises success from 0.09 to 0.87 under an unknown constant drift and from 0.42 to 0.84 under a random one. It also completes misheard and changed requests in 0.91 to 1.00 of the episodes. Acting alone, GPT-6 Astra needs 2.1 min per success and over a million tokens per episode, whereas RoboThink needs 55 s and 8k tokens. The same recipe works across policies, for both vision-language-action models and world action models, and across robots, from one arm to two arms and a mobile manipulator. On all 50 RoboTwin 2.0 tasks with two arms, success rises from 0.12 to 0.74 under the constant drift and from 0.31 to 0.56 under the random one. Finally, the agent is a plug-in. A policy taught by one agent works with other agents, including open-weight ones, and grows stronger with a stronger agent, without retraining. 

---
# Ultrasound Watermark: Real-time Acoustic Watermarking against Voice Scams on Smartphones 

**Authors**: Renzhi Hao, Shirui Cao, Yizhu Wen, Rishav Gupta, Dharani Nadendla, Mehran Kafai, Hanqing Guo, Dong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.30106)  

**Abstract**: Recent advances in text-to-speech (TTS), voice conversion (VC), and AI-driven speech synthesis enable adversaries to convincingly impersonate trusted individuals during phone calls. Existing defenses often rely on offline analysis, degrade under telephony distortions, or remain vulnerable to adaptive attacks. To address these limitations, we present Ultrasound Watermark, a real-time, smartphone-native framework that protects live calls by binding speech to the caller's physical articulatory motion through acoustic watermarking. During a call, the smartphone emits ultrasonic signals and captures reflections from facial articulatory movements using only its built-in speaker and microphone. Ultrasound Watermark extracts motion features from these reflections and fuses them with outgoing speech to generate a dynamic watermark. Because the ultrasonic measurements are physically and temporally coupled with speech production, they are difficult to spoof without the underlying articulatory motion. A synchronized generator-detector architecture enforces cross-modal speech-motion consistency against replay, synthesized speech, and articulation-mimicking attacks. Ultrasound Watermark operates continuously with less than 200 ms latency overhead and requires no additional hardware. We implement Ultrasound Watermark on iOS and Android and evaluate it using over 2.5 hours of synchronized speech-ultrasound recordings from 22 speakers across eight smartphones and four real-world environments. Results demonstrate robustness to diverse attacks and telephony distortions while preserving perceptual audio quality. 

---
# When "A or B" Is Less Likely than A: Coherent Probabilities for Vision Decision Models 

**Authors**: Seongrae Noh, SeongEun Hong, HyeongYeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30096)  

**Abstract**: Vision decision models answer a multiple-choice question about an image with one probability per option, and software uses them directly. We show that the probabilities depend on how the options are grouped: a merged option "A or B" receives on average 11.5 points less probability than the sum of A and B. This under-additivity appears at every model size from 0.85B to 26B and remains after fine-tuning on labels. We propose CORE, which asks each training question with its options regrouped and trains the answers to all these questions toward one target distribution over the original options, summed into each question's groups. On 1,050 items from seven benchmarks, CORE reduces the total variation distance between the answer to a regrouped question and the original answer summed into the same groups from 14.8 to 5.8 points on the 0.85B model, against 9.4 for the best baseline trained on the same questions. At all five model sizes, CORE reduces this distance without lowering accuracy. 

---
# ESGPT: A Data-Centric ESG Language Model with Unified Instruction and Benchmark Resources 

**Authors**: Jaeyoung Lee, Misuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.30052)  

**Abstract**: ESG analysis requires complex language understanding across diverse sources, including corporate reports, news, and disclosures, for tasks such as ESG relevance detection, action interpretation, impact analysis, and question answering. However, existing ESG datasets are fragmented by task and source, with heterogeneous formats and annotation schemes, limiting their use for training general-purpose ESG LLMs and unified evaluation. In this work, we construct ESG-Instruct, a 30,000-instance ESG instruction dataset built from public ESG resources, and ESGU-Bench, an ESG understanding benchmark covering six representative ESG tasks. Based on these resources, we further develop ESGPT, an 8B ESG-specialized LLM, and evaluate it against diverse open-source and API-based LLMs. Experimental results show that general-purpose LLMs exhibit substantial performance variation across ESG tasks, while ESGPT, despite its relatively small scale, achieves the best performance on multiple ESG tasks. This work provides unified data and benchmark resources for training and evaluating ESG-specialized LLMs. 

---
# RoboLiveMem: Is Robot Memory Solved When the World Does Not Wait? 

**Authors**: Yunbei Zhang, Yuyou Zhang, Janet Wang, Ming Liu, Yijiang Li, Yikai Wang, Weirui Ye, Ding Zhao, Jihun Hamm  

**Link**: [PDF](https://arxiv.org/pdf/2611.30047)  

**Abstract**: Robot agents built on frontier vision-language models now solve most robot memory benchmarks. Does this mean they can remember what they need in a home, where people and machines keep moving things? Existing memory benchmarks cannot answer this question, because they pause the simulator while the model thinks: nothing can change during a decision, so whatever the robot remembers is still true when it acts. We introduce RoboLiveMem, a benchmark that runs the same memory tasks twice, once with the world paused while the model thinks and once with the world running. It contains 29 task families in three tiers, from 16 realistic BEHAVIOR-1K scenes of homes, workshops and a restaurant to controlled ManiSkill tasks. Every task mixes real changes with harmless look-alikes and runs without any change, so that no fixed strategy succeeds, and most tasks have a twin that keeps the change in view, which separates forgetting from acting too late. Testing 16 models on 34,176 episodes, we find three surprises. (1) Paused results hide the problem: when a remembered object changes in a realistic scene, the 8 Claude and GPT models succeed in 97% of episodes on average with the world paused, but in only 50% with the world running, and their ranking changes: Claude Opus 5.5, which never fails when paused, succeeds in 44%. (2) Speed is not the fix: slower models do no worse, and models given only 2 s per decision still fall at least 36 points below their paused success. (3) Models see the change but act on camera frames from before it ended. Looking again just before acting closes most of the realistic gap, yet when identical objects are swapped several times, all nine models we test still lose points to memory. 

---
# Spiking World-Action Models for Robotic Manipulation 

**Authors**: Mingquan Liu, Peng Zhou, Shanming Yao, Shuai Wang, Simo Wu, Malu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30041)  

**Abstract**: World action models (WAMs) predict how the world responds to actions and offer a promising path toward general embodied intelligence, but they are large and run the network many times per plan, so they are hard to deploy on robots. Spiking neural networks (SNNs) compute with sparse binary spikes and use additions instead of multiplications, which saves power. However, existing WAMs use dense videos as the world state, so a spiking version would spend most of its computation on the background. We therefore introduce SpikingWAM, the first spiking WAM, which uses a sparse task skeleton as the world state. Inspired by biological motor control, which attends only to the key positions of a task, the Task Skeleton State (TSS) keeps the keypoints of the arm and the task objects, and a small spiking front end estimates the object keypoints from images. Body Frame Spike Coding (BFSC) and Grid Coincidence Attention (GCA) keep millimetre precision, and the Persistent Single Step Readout (PSR) runs the network once per plan. On the simulated benchmarks LIBERO, LIBERO-Plus and RoboTwin 2.0, SpikingWAM achieves competitive success rates with far less energy than ANN methods and surpasses all compared methods on LIBERO-Plus. Real robot experiments further confirm its effectiveness. These results show that a task skeleton lets spiking networks implement WAMs at very low energy, offering a new solution for low power embodied intelligence. Project page: this https URL 

---
# One UWB Radar, Many Views: Comparing and Fusing Signal Representations for Human Activity Recognition 

**Authors**: Jelle De Moerloose, Anton Lambrecht, Adnan Shahid, Eli De Poorter  

**Link**: [PDF](https://arxiv.org/pdf/2611.30033)  

**Abstract**: IEEE 802.15.4z-compliant impulse-radio ultra-wideband (IR-UWB) transceivers are increasingly built into consumer devices, enabling contactless, privacy-preserving human activity recognition (HAR) in healthcare. Before activity classification, the signal is typically converted into range-time, range-Doppler, or time-Doppler maps, each exposing different spatial and motion cues. Existing comparisons often apply shared processing across representations, leaving it unclear whether their rankings hold after independent tuning. It is also unknown whether rankings and fusion gains generalize across environments and activities, and at what computational cost. Consequently, practitioners lack guidance on which representation and processing suit a given HAR deployment. We release an open dataset of raw IR-UWB measurements collected by a ceiling-mounted radar from 44 participants during scripted activity sequences in 12 layouts across two bedrooms. Using a statistically guided, staged search, we independently tune each representation across five processing and deployment settings. We then compare individual and fused performance under layout and room shifts and quantify computational cost. Across over 1{,}000 cross-validated models, independent tuning proves essential, as optimal processing differs per representation and processing choices shift macro-F1 by up to 26 points, four times the gap between representations. Once tuned, the range-Doppler map leads for unseen layouts (85.1\% macro-F1), while time-Doppler map does for unseen rooms (72.7\%). Fusing all three maps raises these scores to 89.1\% and 80.0\%, respectively. Lower sampling rates and narrower Doppler bands reduce latency by 38--51\% with at most a 1.5-point macro-F1 loss. The best representation for UWB HAR is therefore a deployment decision, not a fixed property. 

---
# Antiphony: A Dual-Process Agent for Deepfake Video Detection with Real-Video Calibration 

**Authors**: Xinyu Yan, Yurong Hao, Zherui Li, Yebo Wu, Tiantong Wu, Jiaming Zhang, Zihao Zeng, Zheng Lin, Zijie Wang, Pengpeng Qiao, Yang Cao, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2611.29999)  

**Abstract**: Reliable video deepfake detection requires generalization to unseen generators while limiting false alarms on authentic footage. Although existing detectors capture diverse visual cues, they may produce overconfident errors under generator shift, limiting the reliability of confidence-based fusion. Threshold calibration is further complicated by shifts in authentic capture domains and the lack of future-generator examples. To address these challenges, we present Antiphony, a framework that combines calibrated fusion of four visual experts with selective investigation. Authentic videos serve as a shared reference for score fusion and threshold calibration, enabling threshold updates without target deepfakes. For this expert-fusion stage, we establish finite-sample false-alarm control when the scoring procedure is fixed independently of calibration and authentic samples are exchangeable. To refine decisions near the calibrated threshold, a tool-using multimodal investigator examines these videos and contributes an additional fusion score. We train the investigator on generator-held-out predictions and counterfactual expert outputs, to improve robustness to unreliable expert evidence. Across six benchmark protocols, Antiphony achieves a mean benchmark score of 88.3%, exceeding the strongest evaluated MLLM and conventional detector baselines by 14.7 and 11.5 points, respectively. It also outperforms ARGUS by 20.2 points on average and by 3.6 balanced-accuracy points on FaceVid's 20 unseen generators. 

---
# Verifying Declared AI Workloads from GPU Supply Current 

**Authors**: Tom Kimpson  

**Link**: [PDF](https://arxiv.org/pdf/2611.29982)  

**Abstract**: Frontier AI governance agreements require compute verification techniques that allow an external observer to confirm that a hardware operator does not run prohibited workloads. Off-chip, analogue hardware signals, such as power draw, have been proposed as a potential information channel for compute verification. Such signals can be measured externally, without the operator's cooperation. We show that the supply current drawn by a GPU can be used to verify the work declared by the GPU operator, without requiring examples of prohibited workloads or evasion strategies. We record 23.4 h of GPU current with a Rogowski coil under idle, training and inference workloads. We consider five evasion strategies an operator might use to disguise prohibited training. We show that a one-class model fitted to the current draw of permitted, declared work (idle and inference) flags at least 99.9% of windows of prohibited training, including training hidden by the five evasion strategies. As a continuous monitor across successive windows, it detects all 188 episodes of prohibited training hidden by time-sharing with declared inference, with a median delay of 3.0 s, and raises one alarm in 8.8 h of declared work. Incomplete or corrupt recordings of the declared work can undermine verification. An incomplete recording fails loudly; with idle missing, the monitor is in alarm for 99.97% of declared idle. A corrupt recording can fail silently; if the operator runs training during the recording, the monitor can miss hidden training while raising no alarm on declared work. These results suggest that operator declarations together with external analogue sensing might offer a practical route to compute verification. 

---
# Separating Genetic and Protein Evidence in Open Targets: An Auditable Re-ranking Study 

**Authors**: Ye Ye  

**Link**: [PDF](https://arxiv.org/pdf/2611.29978)  

**Abstract**: Biomedical target-ranking platforms combine genetics, literature, experimental evidence, tractability, and clinical precedent because each is useful for target discovery. A single integrated score, however, can be awkward to reuse for a narrower question such as which genes have direct evidence for disease risk or etiology. We examine this issue in Open Targets associations for Alzheimer disease (AD) and Parkinson disease (PD). Genetics-linked evidence and protein-oriented evidence are scored in separate channels and combined only at the end. In AD, the Open Targets overall top 20 contains 7 curated genetics-supported genes and 10 symptomatic pharmacology controls; the default late fusion contains 15 curated genes and no symptomatic controls. In PD, late fusion retains 13 curated genes and removes the 5 symptomatic controls present in the platform top 20. A genetics-only ranking reaches similar curated recall (14 genes in AD, 13 in PD), so most of the gain over the platform score comes from emphasizing genetic evidence; the protein channel mainly raises the number of tractable targets in the shortlist (from 9 to 12 in AD and from 8 to 10 in PD) without readmitting symptomatic controls. These results are stable across fusion weights from 0.35 to 0.75, perturbed protein-channel weights, and rule thresholds. An optional post-ranking rule further lowers high-clinical/low-genetic targets such as ACHE and GRIN1 without changing top-20 curated recall. The procedure is deterministic and uses public Open Targets fields. Because the component scores remain available, individual rank changes can be inspected and the ranking can be reproduced from the same data snapshot. 

---
# Conartist: Adaptive Agentic Red-Teaming of Computer-Using Agents against Multi-Step Prompt Injection 

**Authors**: Jiaqi Rong, Yunchen Lu, Lihai Nie, Yan Jia, Tong Li, Zheli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29977)  

**Abstract**: Harnesses secure Computer-Using Agents by enforcing security constraints and defending against prompt injection attacks. However, we find that two core security mechanisms of agent harnesses remain vulnerable. First, consistency checking prevents agents from deviating from legitimate user objectives, while attackers can bypass it by embedding malicious actions that appear logically consistent with legitimate task execution. Second, risk auditing proactively examines high-risk operations. Yet, the auditing process itself would introduce additional exploitable actions that violate security boundaries. To evaluate above defects we found, we propose Conartist, a cascaded adaptive prompt injection framework comprising two progressive stages. First, Conartist employs a goal-pollution strategy to establish deceptive causal dependencies between legitimate user objectives and malicious operations, progressively embedding attack steps into legitimate task workflows. However, these task-coupled attacks remain susceptible to risk auditing. To address this limitation, Conartist further analyzes security-auditing feedback from controlled attack exploration to identify exploitable auditing patterns. Conartist then adaptively refines its deception strategies to either avoid triggering risk auditing or induce additional high-risk operations. Extensive experiments across five backbone models demonstrate the effectiveness of Conartist, achieving attack success rates of 32.12% on 576-instance tool use benchmark and 66.67% in the Reddit GUI benchmark, outperforming the SOTA attack by 14.41 and 29.17, respectively. By analyzing one of the most popular Agent, Openclaw, we furthermore concluded seven realistic exploitable attack patterns, which reveals that security mechanisms designed to prevent malicious actions can themselves be manipulated to facilitate attacks. 

---
# When 1+1 $\neq$ 2: Benchmarking Civil-Time Reasoning in Large Language Models 

**Authors**: Melika Shirian, Kianoosh Vadaei, Arshia Hemmat, Mohammad Hossein Shaker Ardakani, Ali Mamanpoosh, Matin Moqadas, Amirmasoud Rismanchian, Afsaneh Fatemi  

**Link**: [PDF](https://arxiv.org/pdf/2611.29972)  

**Abstract**: Calendar arithmetic is non-linear. When 1:59 AM + 1 minute occurs during daylight saving spring-forward, the result is 3:00 AM, not 2:00 AM, as the hour is skipped. We introduce CivilTime-Bench, a benchmark for civil-time reasoning covering daylight saving transitions, non-hour offsets, date-line crossings, and ISO week boundaries. With 60,011 training and 10,500 evaluation examples under strict output-contract evaluation, we benchmark seven LLMs (six instruction-tuned models and Llama-3.1-8B base) and find: zero-shot exact match varies substantially across models, contract-aware prompting often improves format compliance but yields inconsistent accuracy gains, few-shot in-context learning provides model-dependent improvements, and fine-tuning achieves 74.6--86.6% EM on synthetic evaluation but only 36.6--45.2% on HumanTime-500 for models with valid results on both splits, highlighting substantial limitations in generalization. We release the benchmark and code. 

---
# Yggdrasil: Tree-Routed Recurrent Experts for State Tracking and Fast CPU Inference 

**Authors**: Cosmo Santoni  

**Link**: [PDF](https://arxiv.org/pdf/2611.29954)  

**Abstract**: Parallel-trainable recurrences such as Mamba, linear attention, and DeltaNet offer constant-cost streaming inference, but they suffer from two limitations: 1) they struggle to track discrete state, and 2) they read every parameter at every step, tying model capacity directly to CPU latency. We introduce Yggdrasil, a recurrent architecture that decouples capacity from per-step compute by looking up each layer's transition in a large, tree-routed bank of recurrent experts. Each router addresses its bank using the current token and the state of the layer below. We prove that this cross-layer context is necessary---a single token-indexed layer of non-expanding transitions with finitely many reachable states cannot perform conditional resets---while avoiding feedback from the layer's own past preserves parallel-scan training. Because each step traverses one tree path and reads a single expert per layer, active weights stay within a few hundred kilobytes in CPU L2 cache while the full bank resides in RAM. Among the parallel-trainable models we evaluate, Yggdrasil is the only one to solve all five state-tracking tasks. At matched active parameter counts, it achieves the lowest test bits on enwik8 (1.540 bpb) and the human genome hg38 (1.567 bpb), while stepping $3.0\times$ and $3.3\times$ faster on a single CPU core ($6.7\,\mu\mathrm{s}$ and $5.8\,\mu\mathrm{s}$) than the fastest streaming baseline, and delivering the highest GPU throughput of all models at batch 1024. On the Genomic Benchmarks, it is the strongest causal model at matched active parameters (86.0% mean accuracy). 

---
# LLM-Assisted Software Migration: from Module-Level Transformation to Repository-Level Integration 

**Authors**: Yanan Zhang, Shengqi Liu, Tao Zhang, Yidi Liu, Alan Tang, Hailong Chen, Yantao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2611.29942)  

**Abstract**: Software migration adapts an existing implementation to a new language, version, API, framework, or platform while preserving its behavior. Large language models (LLMs) succeed on individual functions and modules, but this success does not readily carry over to entire repositories, particularly deployable applications. This survey reviews LLM-assisted migration from module-level transformation to repository-level integration, culminating in the delivery of complete applications. We analyze 96 sources---including 46 direct migration papers---across four perspectives: (1) supported software engineering activities, (2) model roles as editors or orchestrators, (3) orchestration workflows (agentless, single-agent, or multi-agent), and (4) enabling techniques including knowledge engineering, project context, environment feedback, and model adaptation. We then compare evaluation practices and benchmarks. Although 30 of the 46 studies target repositories, most acceptance evidence relies merely on build success, supplied tests, or static and model-based proxies. Moreover, compilation, visual similarity, test outcomes, and runtime behavior capture fundamentally distinct properties and cannot be conflated into a single success rate. We also identify critical gaps in requirements engineering support and in controlled comparisons of agent architectures. Finally, we discuss six open problems: application integration, platform differences, non-functional properties, generalization, attribution of gains, as well as cost and maintenance. 

---
# Autonomous Multireference Quantum Chemistry with Artificial Intelligence Agents 

**Authors**: Younghwan Kim, Ming-Hsiu Hsieh, Ankit Mathanker, Arpan Choudhury, Joshua Zhou, Qiaohong Wang, Matthew R. Hermes, Laura Gagliardi  

**Link**: [PDF](https://arxiv.org/pdf/2611.29902)  

**Abstract**: Accurately describing chemical reactions that involve strongly correlated electrons remains a major challenge in computational chemistry. Multireference electronic structure methods can address this challenge, but their application requires expert decisions that are di!cult to automate, particularly when the electronic structure changes as chemical bonds form and break. Here we introduce CASPER, an autonomously agentic workflow that performs multireference calculations along chemical reaction pathways. CASPER combines chemical knowledge, electronic structure calculations, and iterative assessment to identify chemically meaningful active spaces and maintain their consistency as bonds form and break. We demonstrate its capabilities on five transition-metal reactions, for which CASPER generates consistent multireference potential energy profiles with minimal human intervention. Importantly, in two reactions, CASPER identifies critical electronic-structure problems that an independent Claude session fails to recognize without the structured workflow. These include an electronic-state crossing and the exclusion of an orbital involved in bond breaking. Our results show the importance of combining language models with chemical knowledge and independent assessment of computational results. Beyond automating calculations, such agents have the potential to help researchers better understand the underlying physics and chemistry and open new opportunities for investigating complex chemical reactivity and generating high-quality quantum chemical data at scale. 

---
# World Models and the Law 

**Authors**: Claudio Novelli  

**Link**: [PDF](https://arxiv.org/pdf/2611.29899)  

**Abstract**: Current AI may have a model of how we use human language, but this does not translate into a reliable representation of physical reality. World models promise to fill this gap by learning how an environment changes over time and predicting the consequences of an action, and they are already used to steer robots and generate explorable worlds. This article asks how AI regulation in three major legal contexts, the European Union, California, and China, copes with the distinctive challenges they raise. I propose a definition and a taxonomy of world models based on who or what acts on their predictions, and identify the risks of misplaced reliance on them. I argue that existing rules tend to cover world models but leave three questions open, i.e., action provenance, trajectory validity, and the weight of simulated evidence, and recommend ways to fill these gaps. 

---
# Landscape and Taxonomy of Prompt Engineering Patterns in Software Engineering 

**Authors**: Yuya Sasaki, Hironori Washizaki, Jialong Li, Nobukazu Yoshioka, Naoyasu Ubayashi, Yoshiaki Fukazawa  

**Link**: [PDF](https://arxiv.org/pdf/2611.29853)  

**Abstract**: Advancements in large language models (LLMs) have enhanced their ability to handle ambiguous user instructions. However, effective prompt patterns remain crucial for usability and comprehension. This article presents a taxonomy of prompt engineering patterns for software engineering. It is based on a systematic literature review that was conducted in early 2023, when LLMs still faced significant limitations in context length and inference capabilities. Our study explores techniques that enhance the usability and reliability of LLMs, emphasizing the ongoing importance of well-designed prompts in optimizing task performance. Our findings highlight the critical role of prompt patterns in maximizing LLM's potential, even as their capabilities continue to evolve. 

---
# LexiDet-OVD: Benchmarking Lexical Sensitivity in Open-Vocabulary Object Detection 

**Authors**: Amit Thakur, Mukesh Singhal  

**Link**: [PDF](https://arxiv.org/pdf/2611.29851)  

**Abstract**: Open-vocabulary object detectors localize objects from free-form text prompts, but their reliability depends on how users describe the target. Standard evaluations usually test a fixed class name, leaving prompt-induced failure modes underexplored. We introduce LexiDet-OVD, a benchmark and metric suite for studying lexical sensitivity in open-vocabulary object detection. We organize prompt sets into equivalent, conditional, and broad semantic scopes, and evaluate four detector families on fixed COCO and LVIS samples, each covering 30 classes (800 held-out COCO images and 1,000 LVIS images). Beyond mean and worst-prompt AP50, we report P10, CVaR at 20%, leave-one-prompt-out influence, detection agreement, box consistency, and confidence variance using one-to-one multi-instance matching. Equivalent-prompt best-to-worst gaps range from 5.3 to 23.4 points on COCO1000 and from 5.1 to 29.9 points on LVIS1000-30c, showing that strong average accuracy and lexical stability are distinct properties. We compare maximum-score selection, weighted-box fusion, Prompt Consensus, and an exploratory fusion rule. Aggregation recovers performance in several settings, but no method dominates all model-dataset pairs. A controlled Detic/SHiNe comparison further shows that semantic hierarchy enrichment does not by itself eliminate lexical sensitivity. These results support reporting prompt-distribution robustness alongside conventional open-vocabulary detection accuracy. 

---
# Dynamics Forcing: Mitigating Error Accumulation in Causal Video World Models 

**Authors**: Xin Zhou, Cong Miao  

**Link**: [PDF](https://arxiv.org/pdf/2611.29839)  

**Abstract**: Causal video world models generate block by block from their own previous outputs. Small errors in early blocks are fed back as history and grow until the scene and the robot fall apart. Teacher forcing trains on clean ground-truth history, so the model never sees the history it will actually condition on, and it also finds a shortcut: copying the previous frame instead of predicting how the scene changes. We present Dynamics Forcing, a training scheme that keeps real video as the only supervision and reshapes how history is presented. History is perturbed in three ways: random noising, self-resampling by the model itself, and joint denoising, in which the most recent blocks are still partly noisy when the next block is predicted. On top of this we move history guidance and first-frame anchor guidance from inference into training, so that the model is pushed to use its history and its reference frame rather than ignore them. Dynamics Forcing needs no teacher model, no distillation, and no additional model at inference. Applied to a 5B video model with LoRA on 77 hours of manipulation video on a single GPU, it better matches unseen real continuations and reduces error accumulation. 

---
# Data-Quality-Aware Fundus Classification with Anatomy-Guided Glaucoma Assessment and Cross-Dataset Evaluation 

**Authors**: Imdadul Haque Sourav, Marjana Jannat Miti, Md. Jehadul Islam Mony  

**Link**: [PDF](https://arxiv.org/pdf/2611.29834)  

**Abstract**: Background: Deep learning fundus classifiers often perform well internally, but duplicated images, inconsistent labels, acquisition differences and poor threshold transfer can undermine their reliability. We assess nine-class classifiers beyond internal accuracy, using a data audit, frozen-model external evaluation and an anatomy-guided specialist. Methods: Nine-class experts (ConvNeXt-Small, EfficientNetV2-Small, DenseNet169) were trained with three seeds. Auditing 5,318 images for exact duplicates and cross-label conflicts removed 464 conflict groups (942 images) and six same-label copies, leaving 4,370 images. An anatomy-guided glaucoma specialist read disc-centred crops with vessel and disc/cup segmentation maps, plus cup-to-disc features. Frozen models were evaluated on RFMiD and JSIEC, and a prespecified comparison on REFUGE was replicated on ORIGA, G1020 and PAPILA. Results: ConvNeXt-Small reached an internal accuracy of 0.910, macro-F1 of 0.917 and AUC of 0.982; disc and cup Dice scores were 0.966 and 0.915. Across seeds, the fixed anatomy-guided mix left internal accuracy unchanged relative to equal soft voting (-0.001) but raised mean external closed-set accuracy by 4.9 percentage points on RFMiD (positive in two of three seeds) and 2.3 on JSIEC. On REFUGE (800 photographs) it improved balanced accuracy by 7.4 percentage points (95% CI 5.1 to 9.5), and by 2.1 on ORIGA, with positive gains on G1020 and PAPILA. Thresholds transferred poorly (REFUGE sensitivity 0.950, specificity 0.411). Conclusions: Retinal classifiers should be judged by data-quality auditing, anatomy-informed modelling and multi-cohort evaluation. Anatomy-guided fusion improved transfer at no internal cost, but gains varied by cohort and thresholds did not transfer, so internal metrics alone do not establish reliable performance. 

---
# Can Vibe Coding Produce Accessible Websites: An Empirical Study on LLM Coding Agents and Skills 

**Authors**: Syed Fatiul Huq, Md. Asif Haider, Abu Jafar Saifullah, Thomas Zimmermann, Sam Malek  

**Link**: [PDF](https://arxiv.org/pdf/2611.29822)  

**Abstract**: Large language model (LLM) coding agents increasingly allow developers and non-professionals to create websites through natural-language instructions, with limited inspection of the generated source code. Although this ``vibe coding'' workflow lowers barriers to software creation, it may also propagate accessibility defects at scale. We investigate the accessibility of websites generated end-to-end by contemporary coding agents and whether agent-level accessibility interventions improve their output. We generated 90 websites for three projects using combinations of three coding agents, four language models, and multiple reasoning-effort levels. We evaluated complete interaction journeys using axe-core, IBM Equal Access Accessibility Checker, and QualWeb, and converged reports to observe violation prevalence and violated WCAG Success Criterion (SC). None of the 90 baseline websites was fully accessible, with an average of 80.43 violated interface elements. 13 WCAG SCs were violated at least once across the runs. Color contrast was the most prevalent issue, followed by labeling, semantic-structure, keyboard, and visualization failures. We reran 24 matched configurations with a general accessibility Skill and a contrast-specific Skill. Both significantly improved accessibility scores, but neither eliminated violations. User testing with blind individuals corroborated the experiential effects of the detected inaccessibilities. Moreover, functional correctness decreased with the inclusion of Skills. Our findings show that accessibility does not reliably emerge from vibe coding, and agentic interventions can both improve some accessibility issues and introduce new functional ones. 

---
# How Should a Home Robot Update Its Memories When Human Routines Change? 

**Authors**: Oliver Wang, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2611.29818)  

**Abstract**: A home robot that helps with chores needs to know where objects are, but people move objects according to routines that change and later change back. The robot's memory must adapt to the new routine without erasing experience that may become useful again. One approach is to have a language model maintain written summaries of the home from the robot's own observations, but existing evaluations of such memories use static worlds. We explore simulated households where a resident gets sick, recovers, and later gets sick again. We compare a rule that keeps every room an object has been seen in, language-model memories that store only written summaries, and language-model memories that can also read the raw history of sightings. The rule outperforms every summary-only memory, and giving language models the sighting history closes the gap. Surprisingly, writing summaries on top of the history adds nothing we can measure. When the illness returns, methods that keep the history search better than they did the first time, while summary-only methods show no consistent improvement. Deleting the first illness from memory removes all of the rule's gain and part of theirs. The summaries recorded where objects went during the illness but rarely distinguished sick-day patterns from normal routines. Our findings suggest that common approaches to summarizing robot memories struggle to remember facts about a routine that might return, highlighting the need for future benchmarks to evaluate this aspect of long-term robot deployment. 

---
# Notes on Generative AI for Mathematical Research 

**Authors**: Shurui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29811)  

**Abstract**: These notes describe how generative AI is now used to solve and to check research-level mathematics, with a focus on agents and their harnesses. For the proving agents Rethlas, Danus, and Pharos and the verification agent OSVerify, we explain the design principles, compare each with its base model, present case studies, and show how to use and contribute to them. We also briefly review the history of AI for mathematics and the community's reactions to it. 

---
# When Does Fine-Tuning Beat Prompting for Qiskit-to-CUDA-Q Translation? An Execution-Verified Study 

**Authors**: Ana Paula Appel  

**Link**: [PDF](https://arxiv.org/pdf/2611.29793)  

**Abstract**: Moving quantum code from Qiskit to NVIDIA CUDA-Q is a practical adoption cost that language models are starting to automate. We ask whether a small open model can learn this translation, and how to measure it so that the result can be trusted. Both programs can be executed and their state vectors compared, so a kernel counts as correct only if it prepares the same state as the Qiskit reference (fidelity >= 0.999), with no text matching and no LLM judge. We build a 640-task benchmark with three test sets (larger circuits, an unseen circuit family, an unseen Qiskit coding style) and a QuantumKatas regression probe, and we test the evaluator before training. Measurement counts miss every phase-only error; a surface rule-based translator solves two of three original splits (text substitution); a converter that executes the Qiskit input and maps the circuit solves all 640 tasks -- so the benchmark measures source-level translation, not whether a circuit can be ported after running it. No base model emits a valid CUDA-Q kernel zero-shot. On Nemotron 3 Nano 4B, LoRA reaches 83.9%, tying a 4-shot prompt (83.1%), but generalizes worse to unseen style (48.5% vs. 65.5%) and cuts QuantumKatas from 23.8% to 8.9%; on Qwen2.5-Coder 7B, LoRA beats the prompt overall (92.0% vs. 82.8%) and on unseen style (78.5% vs. 65.0%), and on the 1.5B model as well, so whether fine-tuning pays off depends on the base model, while the Qiskit regression appears in all three. Mixing two replay formats is the best mitigation trade-off (84.4% translation, Katas 21.5%). On 48 parametric functions that must become kernels with arguments -- where executing the input cannot yield a parametric kernel -- models reach up to 83%; on 27 real QED-C kernels, every model scores 0%. Language models can do source-level translation of simple parametric code, but not yet of real programs. 

---
# A Cascaded Framework with a Cross-Attention-Guided Mamba-CNN Model for Pre- and Post-Treatment Glioma Segmentation 

**Authors**: Siavash Taremi, Mahmoud Heidari, Seyed Mohammad Hossein Hashemi, Mohsen Hooshmand  

**Link**: [PDF](https://arxiv.org/pdf/2611.29792)  

**Abstract**: Accurate glioma segmentation is vital for diagnosis and follow-up, but manual segmentation is time-consuming, subjective, and variable across interpreters. Gliomas are heterogeneous in size, shape, and location. Deep learning has therefore emerged as a potential solution. CNNs excel at local features but may struggle with long-range dependencies, whereas Vision Transformers capture global context efficiently but incur higher computational complexity. Most previous studies focused on pre-treatment gliomas; post-treatment cases with resection cavities (RCs) remain less explored. This study proposes a Cross-Attention-Guided Mamba-CNN (CMC-Seg) model within a cascaded framework for pre- and post-treatment diffuse glioma segmentation, aiming to balance accuracy and computational cost. Experiments used BraTS 2025 mpMRI of diffuse glioma patients. After preprocessing, a coarse segmentation model extracted tumor-centered patches, which were fed to CMC-Seg for fine-grained segmentation. CMC-Seg was compared with 3D U-Net, SegResNet, SwinUNETR, UNETR, nnFormer, and SegMamba. On the test set, CMC-Seg achieved the highest mean Dice (81.20%) and non-enhancing tumor core (NETC) Dice (77.09%). It also achieved an RC Dice of 78.63% and the lowest 95% Hausdorff distance (HD95; 4.39 mm). Results were comparable to CNN and Mamba baselines and significantly better than Transformer-based baselines (P < 0.05). CMC-Seg required the fewest inference GFLOPs (381.49). Ablation experiments indicated that multi-direction scanning and cross-attention improved performance, particularly for RC (P < 0.05). Grad-CAM explainability experiments showed that CMC-Seg focuses on diagnostically relevant tumor regions. 

---
# S1Arena-MM: Which Multimodal Decisions Can Be Left to System 1? 

**Authors**: Lang Lin, Tianyu Chen, Yi Gu, Yang Zhao, Xinran Song, Xinyang Liu, Jianpeng Cheng, Mingyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.29773)  

**Abstract**: Perception systems often make discrete decisions: flag a hateful meme, predict whether a pedestrian will cross, detect a spoofed voice, or select the next move in a game. System-1 models, such as OneJev and Jev-Omni, read the media and return probabilities over the options in a single forward pass, while multimodal large language models (MLLMs) generate text. Which multimodal decisions can System-1 models handle, and where do MLLMs perform better? We introduce S1Arena-MM, a benchmark that gives every model the same decision requests and scores the probabilities it returns. It contains 51,060 questions across 94 tasks with image, video, and audio input, drawn from 94 public datasets and three program-generated ones. It also includes 102 closed-loop game environments that a model plays from screenshots; in real-time mode, the game continues while the model decides. We evaluate 33 open models: 12 System-1 models and 21 MLLMs read at the answer token. On static decisions, the best System-1 model leads the best MLLM by 4.9 percentage points on images and is within one point of it on video and audio. Post-training for decisions mainly improves the probabilities: on video, a System-1 model has a lower Brier score than the MLLM that it is post-trained from in 10 of 12 pairs. With up to 2,048 thought tokens per question, the most accurate MLLMs pass the best System-1 models only slightly, and thinking raises the Brier score of most MLLMs. Closed loops separate the models: three System-1 models built on a 27B MLLM reach 22.0-33.3% progression in BALROG, and no other model exceeds 9.7% with thinking off. In real-time GameWorld, latencies of up to 1.1 s per decision cause no measurable loss of progress for System-1 models. Code and data: this https URL. 

---
# Into My Arms: Structured reinforcement learning for autonomous transport of objects and humans 

**Authors**: Elle Miller, Reem Almheiri, Takuma Tsukakoshi, Ben Sanati, Tamon Miyake, Yushi Wang, Namiko Saito, Shigeki Sugano, Tetsuya Ogata, Rika Antonova, Oisin Mac Aodha, Sethu Vijayakumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.29771)  

**Abstract**: Whole-body humanoid manipulation remains bottlenecked by high-dimensional state-action spaces, contact-rich environments, and long task horizons. This trilemma is a challenge that state-of-the-art reinforcement learning (RL) methods are yet to resolve reliably. We address it with three contributions that add just enough structure to guide exploration: Sparse Interaction Edges (SIE) for whole-body manipulation, Interaction Gated Transport (IGT) for sustained transport over long horizons, and a Hinted Object Path (HOP) that resolves collision and search complexity bottlenecks. Across all three, the policy remains free to discover its own manipulation strategy. The method generalises from rigid object to human transport. As a flagship demonstration, we show, to our knowledge, the first bed-to-wheelchair transfer of a medical mannequin by a humanoid robot completed with no human intervention during execution. In the domain of human transfer, we show that a single policy can adapt across substantial variations in body mass, size, and random force perturbations, all while training on a single GPU in under an hour. Our approach offers a pathway to fully RL-driven humanoid manipulation, enabling assistive robots that can help humans. Website: this https URL 

---
# Estimating a Transition Point from Gibbs and Ground States 

**Authors**: Yuhao Cai, Zhenzhang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.29768)  

**Abstract**: We study how observed many-body system size affects the number of state copies needed to estimate a zero-temperature first-order transition point. The unknown Hamiltonian has bounded, translation-invariant on-site and nearest-neighbour interactions on a 3D cubic lattice with local dimension three. The field coupling is known, and all models share a known boundary field penalizing one local level. The limiting order parameter is uniformly separated across the transition. Each query returns an independently prepared Gibbs or ground state at an adaptive field and system size.
For error $\epsilon$ and fixed confidence, we prove an exponential separation between two size caps at the same temperature. There are absolute constants $0<c_0<C$ and $c_1>0$ such that, at inverse temperature $\beta=c_1/\epsilon$, restricting side lengths to $L\le c_0/\epsilon$ requires $\exp(\Omega(1/\epsilon))$ copies in worst-case expectation, even with collective measurements. Allowing $L\le C/\epsilon$ permits $O(\log^2(1/\epsilon))$ copies and $O(\epsilon^{-3})$ total spins, using local measurements and polynomial classical postprocessing. The minimax expected total spin count is $\Theta(\epsilon^{-3})$ uniformly for $\beta\ge c_1/\epsilon$. At fixed positive temperature it is $\Theta_\beta(\epsilon^{-2})$, attained from one- and eight-site Gibbs states followed by exponential classical computation. For ground-state copies it is $\Theta(\epsilon^{-3})$.
The lower bound uses a three-state ferromagnet whose boundary suppresses the sector carrying transition information. A spanning-forest argument and edge-boundary counting control its Gibbs weight uniformly over fields and allowed sizes, and a trace-distance hybrid argument handles adaptive learners with quantum memory. The upper bound combines finite-volume free-energy estimates, concavity and local order-parameter measurements. 

---
# Selecting Stable Sampling Dynamics for Compositional Diffusion: A Feynman-Kac Weight Reduction 

**Authors**: Sunjin Kim, Sungsoo Ahn, Michael Chertkov  

**Link**: [PDF](https://arxiv.org/pdf/2611.29764)  

**Abstract**: Given $K$ separately trained diffusion models, how should one sample from a $\beta_k$-weighted product of their terminal laws without jointly retraining the experts? Existing Feynman--Kac constructions provide many weighted stochastic differential equations (SDEs) with the correct target marginal. The consequential question is how to select useful proposal dynamics among them. We study an explicit family of weighted diffusions that realizes the same product-of-intermediate-experts trajectory. A scalar parameter $\lambda$ controls how strongly the proposal drift follows the composed expert score, while an additional Langevin component can be varied independently. Writing $B=\sum_{k=1}^{K}\beta_k>0$, we identify the distinguished choice $\lambda=1/B$. At this value, the aggregate squared-score term in the instantaneous Feynman--Kac log-weight rate cancels exactly. For positive expert exponents, the remaining score-disagreement contribution is non-positive. We call this structural cancellation Feynman--Kac reduction. Under bounded-divergence assumptions, it yields an upper bound on the instantaneous unnormalized log-weight rate for all $\lambda\leq 1/B$. For linear diffusion protocols with compactly supported expert data, $\lambda=1/B$ yields two-sided bounds at every nonterminal time, for exponents of either sign. A corresponding result holds for a single expert under the same divergence assumption. In these two-sided settings, the selected value is unique within the specified family in a universal worst-case sense. Experiments on Gaussian mixtures and molecular sampling compare the analytically predicted stability with actual sampling quality, including instances in which parameter choices different from $\lambda=1/B$ perform better empirically. 

---
# Why Units Die in the Layer After a Frozen Random Encoder, and What the Number of Live Units Measures 

**Authors**: Scott M. Norton  

**Link**: [PDF](https://arxiv.org/pdf/2611.29752)  

**Abstract**: Reinforcement-learning agents can learn Atari games through a convolutional encoder whose random weights are never trained. They reach expert play in Pong and, without random action repeats, in Breakout, but training takes far longer and few ReLU units in the layer after the encoder stay alive. A dead unit outputs zero for every input and cannot learn. In earlier work we took the number of live units to measure the game's complexity. Here we ask why units die and correct that conclusion. We find that more units die as the layer's inputs lie farther from zero relative to how much they vary. We call this ratio the offset. Earlier work showed this in sparse autoencoders at the start of training. We saw it after training in agents, image classifiers, a reservoir computer and sparse autoencoders on GPT-2. A random encoder leaves the offset of its input roughly unchanged, and a formula predicted its median offset within 25 percent in all 44 cases. So a game with a bright, mostly unchanging screen gives the next layer a large offset. In Pong, subtracting the average screen raised the number of live units from between 3 and 6 of 64 to between 59 and 63, so the number alone cannot show a game's complexity. On unaltered screens none of the agents reached a training score of 18 in 50 million steps. With the subtraction all did within 5 million, about as fast as a trained encoder on unaltered screens. In Breakout the subtraction left more units alive, but average training scores did not clearly improve, and a large gap to a trained encoder remained. So we do not treat their number as something to raise for its own sake. The evidence is from small networks and three games, and in agents and image classifiers changing the offset also changed the size of the inputs. We suggest measuring the offset before training a layer on a fixed front end and, if it is large, trying to subtract the mean input. 

---
# Refine, Don't Reinvent: RASA-Bench for Source-Grounded Hindi Prose Refinement 

**Authors**: Abdur Rahman, Pratham Singla, Mohammad Saifullah, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2611.29750)  

**Abstract**: For a producer of Hindi creative prose, drafting is cheap and editing is not: a machine-written episode, story, or essay reads as under-written and flat, laced with translationese, and it sometimes quietly alters what the source says. RASA-Bench targets that editing task across three Hindi prose forms, measuring how well a model refines an under-written AI draft, a task distinct from writing from scratch. Across 2,050 drafts and eight refiners (16,400 rewrites), a four-layer protocol credits a rewrite only after it passes a faithfulness check against its own source, then ranks the survivors by holistic editorial preference through per-form Bradley-Terry fits whose centered coefficients are standardized before equal-weight averaging. Staying faithful proves to be a decisive test that behaves differently by form: gate-failure rates range from 9.1% on short stories to 29.5% on audio fiction, and the drift that disqualifies a rewrite shifts from events and identity in narrative to claims and evidence in essays. A 600-case human audit puts gate agreement at 0.925 (0.91 to 0.94 across forms), whereas subjective preference is form-sensitive and least settled for essays, among human readers no less than the model judge. The benchmark, evaluation protocol, and Hindi prose diagnostics are available at this https URL. 

---
# Lightweight Spectral Subspace Filtering for Trojan-Resilient AI Inference Engines 

**Authors**: Soumita Chatterjee, Sudip Ghosh, Tamal Ghosh, Bhargab B. Bhattacharya, Hafizur Rahaman  

**Link**: [PDF](https://arxiv.org/pdf/2611.29734)  

**Abstract**: Backdoor attacks pose a significant security threat to deep neural networks (DNNs) deployed on AI inference engines and hardware accelerators. By embedding hidden triggers during training, these attacks induce malicious behavior while maintaining high accuracy on benign inputs. Multi-trigger backdoor attacks are particularly challenging to detect and mitigate because of the distributed and stealthy nature of their effects. To address this challenge, this paper proposes a training-free Spectral Subspace Filtering framework that detects and suppresses backdoor-induced activations by analyzing latent feature representations. Exploiting the geometric separation between benign and malicious feature distributions, the framework employs Singular Value Decomposition (SVD) to identify the dominant malicious subspace and performs inference-time spectral scrubbing through activation projection. This approach suppresses backdoor effects without model retraining, requiring only a small set of triggered samples collected offline. Experimental evaluation across CNN and transformer architectures, including AlexNet, ResNet18, VGG16, MobileNetV2, and Vision Transformer (ViT), demonstrates that the proposed method reduces the attack success rate (ASR) from 94.2-99.8% to an average of 2.07%, while preserving clean accuracy of up to 98.54% with low inference overhead. The framework also demonstrates robustness across varying poisoning rates and multiple concurrent triggers, highlighting its potential as a lightweight runtime defense for enhancing the security and reliability of AI inference engines against backdoor attacks. 

---
# Rethinking Discovery Research with AI: The promise of transdisciplinarity and the danger of reductionism 

**Authors**: Iulia Georgescu, Venkatesh Narayanamurti  

**Link**: [PDF](https://arxiv.org/pdf/2611.29729)  

**Abstract**: In theory, transdisciplinary research focuses on solving real-world problems without being constrained by disciplinary boundaries and methodologies. In practice, doing so is challenging, as recognized by the phrase "The world has problems; universities have departments." As a powerful integrative tool, AI may help overcome traditional reductionist approaches and enable genuine transdisciplinarity. In this essay, we discuss how this could happen and why, within the current system of incentives, it is not happening. 

---
# Admitted Is Not Effective: Lessons from a Notary-Gated Retrospective Loop in an Agentic Pipeline 

**Authors**: Lo Gullstrand Heander, Emma Söderberg, Christofer Rydenfält  

**Link**: [PDF](https://arxiv.org/pdf/2611.29727)  

**Abstract**: Agentic software development pipelines with frozen prompts can repeat the same mistakes across pull requests (PRs), yet ungoverned additions to instruction files grow without bound and do not generally improve task success. We present a retrospective loop for a four-stage large language model (LLM) pipeline in which a Retrospective agent proposes edits to learned rules every ten PRs and a Notary admits, merges, or rejects them while recording each decision in a provenance ledger. In a pre-registered study of 880 PRs on two Python repositories, the loop ran end to end and the Notary admitted 55 of 89 proposals, rejecting 8 that cited nonexistent evidence.
However, held-out review comments and hidden-test resolution did not improve after ten retrospectives. Tracing each step of the loop, we found that sparse review signal (0.45 comments/PR) left little to learn from and that rules admitted on their stated evidence did not show measurable held-out effects. We derive six lessons and argue that governed loops should admit and retire every learned rule as a testable hypothesis. 

---
# HandiVid: A High-Quality Dataset for Hand-Driven Interactive Video Generation 

**Authors**: Chaoyang Zhang, Shaohao Rui, Xiaofeng Mao, Yuanxun Xu, Haoyuan Xia, Xiaohong Liu, Qinghe Du, Weijie Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.29693)  

**Abstract**: Hand-driven egocentric interactive video generation aims to generate dynamic scenes with realistic interaction responses conditioned on hand movements and camera motion. However, existing egocentric video datasets struggle to provide scene diversity, meaningful hand-object interactions, and reliable motion annotations simultaneously. To address these limitations, we introduce HandiVid, a high-quality dataset comprising 200K curated and annotated egocentric interaction video clips for hand-driven interactive video generation. We develop a unified data processing pipeline that integrates public datasets, online videos and synthetic videos. Through video quality assessment, hand detection filtering, and multi-stage interaction validation using vision-language models (VLMs), we select high-quality video clips with meaningful hand-object interactions. Since 3D hand reconstruction relies on initial hand detections, we integrate SAM 3 into the hand detection pipeline to reduce missed detections and improve the completeness of 3D hand annotations. Each video clip is provided with high-quality hand renderings, camera trajectories, and interaction descriptions. Comprehensive dataset analysis and downstream video generation experiments demonstrate the effectiveness of HandiVid in terms of data diversity, annotation quality, and controllable video generation. 

---
# Do More Roles or Larger Models Improve LLM Fact-Checking? A Controlled Study on Fixed Evidence 

**Authors**: Martin Pavlíček, Martin Kotyrba, Eva Volná, Tomáš Filip, Petr Sosík  

**Link**: [PDF](https://arxiv.org/pdf/2611.29687)  

**Abstract**: A fact-checking system can spend more computation on a claim by using a larger model, by asking the same model several times, or by a multi-role Tribunal: two roles propose quoted findings, code checks the quotations, an auditor that does not see the proposed relation judges each one, and code derives the verdict or abstains. We compare these options on identical fixed evidence, without retrieval: 300 AVeriTeC claims, each with its annotated question--answer items mixed with items written for other claims, read by judges of four open-weights models from three families, a vote of five, and six assignments of models to the Tribunal's roles. The 9B judge was right on 186 claims and the 27B judge of the same family on 156; the larger judges declined to decide far more often and were no more accurate when they decided. The vote reached 196 and the 9B judge with its verdict replaced by Tribunal certificates 195, at a similar cost; the Tribunal alone reached 153 to 164, because it abstained on almost half of the claims. A hearing's outcome followed the auditor's model and was nearly indifferent to the proposers'. A larger auditor, or one of another family, made certificates neither more often correct (77% to 81%) nor more independent of the judge. As a replacement, certificates improved every judge by 3 to 10 points and overturned four to five in a hundred of its correct verdicts. With the annotated evidence withheld, judges and Tribunal still decided on 7% to 14% of the claims. 

---
# Ref2Light: From One Image to Interactive Glasses-Free 3D Characters 

**Authors**: Hanzhong Zhang, Haoyang Wang, Siyang Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.29678)  

**Abstract**: Animating character illustrations on light-field displays demands coherent binocular parallax, real-time pose articulation, and fine-grained expression control. However, existing 2D animation and 3D reconstruction pipelines cannot combine these with rapid character switching. We present Ref2Light, a feed-forward framework that maps a single A-pose illustration and continuous pose and expression controls to display-native light-field animation, without per-character 3D reconstruction, rigging, or test-time optimization. Our key insight is to decompose light-field synthesis into static identity, view-dependent parallax, and control-driven articulation. A shared encoder caches multi-scale identity features once per character for rapid switching. A conditional decoder then synthesizes ordered angular anchors from these features via backward texture transport and direct color synthesis. A global-local control scheme deforms the anchors with head-body controls and uses the updated correspondence fields to track facial regions across views for masked local facial refinement. Finally, calibrated differentiable subpixel interlacing converts the anchors into the native panel raster, enabling end-to-end native supervision without rendering all views at full resolution. On held-out VRoid identities, Ref2Light reduces head-body LPIPS by 20.1%-55.6% over 2D and 3D baselines and lowers non-target change MAE in mouth closure by 95.6%-96.5% versus monolithic decoding, while running at 34.0-40.1 FPS for 1440x2560 native rasters on a consumer GPU. Code and demo are in the Supplementary Material. 

---
# Which Loopholes Get Found? Predicting Reward Hacking of LLM Judges Before RL Training 

**Authors**: Xuanyi Lyu, Yuezhou Zhao, Qing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2611.29651)  

**Abstract**: A policy trained against an LLM judge can learn to exploit its loopholes. Static probes show that such judges accept degenerate answers like a lone "{" or "Thought process:", but not which flaw a policy will find. In GRPO the first loophole to be exploited can be predicted before training from each pattern's exposure, the product of the starting policy's propensity to produce it and the judge's false-positive rate (FPR). To test exposure causally, we seed the patterns into fine-tuning at three doses in a Latin square. Across 36 GRPO runs with seven judges, the first exploited pattern had the highest exposure in all 12 runs with exploitation and the highest FPR in only 6. Predictions recorded before training held in three of four runs with one pattern masked by rule. When only two patterns were seeded, at 3% to 3.8% of the fine-tuning data and chosen so that exposure disagreed with FPR alone or propensity alone, exposure named the first exploited pattern and its onset window in all six runs. Exploitation occurred only on hard prompts, where an accepted exploit gets a large group-normalized advantage. Auditing exposure before RL therefore identifies the loophole a policy will likely exploit first. 

---
# TASC: Time-to-Collision Anticipation and Selective Crowd Attention for Non-Holonomic Service Robots 

**Authors**: Mohammadhossein Malmir, Mohammadali Rahmati, Mohammad Reza Kolani, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2611.29637)  

**Abstract**: Service robots that share corridors and lobbies with people must keep moving without endangering them. Reactive planners can stop indefinitely in dense crowds (the freezing robot problem), whereas the attention-based learned navigators that dominate the standard crowd benchmark assume holonomic motion or emit discrete actions unsuited to a differential-drive base. This paper presents TASC, a navigation system for a non-holonomic service robot that integrates an attention-based crowd encoder, an anticipatory collision feature, and a continuous-control policy. We adapt the dual-selection attention module, originally developed for discrete-action value learning, to proximal policy optimization for continuous velocity control, and add a closing-speed time-to-collision feature, a conservative estimate of the time to contact that serves as both an observation and a reward term. The policy is trained on a high-fidelity 3D robot model in NVIDIA Isaac Lab with 10,000 parallel environments, social-force-model pedestrians, and randomized crowd sizes. In 30,000 evaluation episodes, each run to its end, the policy timed out in fewer than 0.5% at every tested crowd density. This liveness comes at a cost: pedestrian collisions rise from 4.5% with five pedestrians to 17.4% with ten. In an ablation with four seeds per variant, policies retrained without the time-to-collision feature succeed 4 to 7 percentage points less often. The policy runs in Gazebo without retraining and on a commercial differential-drive service robot through a ROS 2 package with an inference latency below 10 ms, verified in a hardware-in-the-loop test with virtual pedestrians. 

---
# Weight Coding: Evolving Web-Scale Recommenders through Direct Weight Edits with LLM Agents 

**Authors**: Junpeng Hou, Jiaxing Qu, Derek Cheng, Jaewon Yang, Abhinav Naikawadi, Karthik Subbian  

**Link**: [PDF](https://arxiv.org/pdf/2611.29613)  

**Abstract**: Web-scale recommender models evolve primarily through retraining or fine-tuning. We propose \emph{weight coding}, a methodology for model evolution through direct, executable weight edits. A large language model (LLM) agent inspects model structure and parameters, formulates hypotheses, and implements edits using evaluation feedback. The loop progresses from localized changes to targeted searches and composition, allowing edited components to adapt to one another. A cached engine enables rapid forward evaluation without backpropagation. On Pinterest's 147M-parameter Image Search two-tower retriever, weight coding improves held-out recall@10 of production model by up to 15.0 percentage points (+24\% relative) with few edits in about 1/4 GPU hours of a full post-training finetune, while a common launch only improves around 1\%. After reinitializing the entire 13.6M-parameter interaction module randomly while retaining the surrounding trained model, searching 2,448 mixing and normalization parameters yields 9.8 points above production. Offline gains span the full funnel including dense retrieval, generative retrieval and ranking, with up to 1.37B learnable parameters and different edits for each architecture. Steering edits provide explicit control of model performance over segments like freshness and product. Weight coding requires no serving change and online A/B tests increase saves per user from the edited retrieval source by 8.4\% to 10.7\%, aligned with the target business objectives. Our results establish weight coding as a practical and useful tool for model development and and behavioral control in a real web-scale production system, with potential to extend to both pre-training and post-training stages across the funnel. 

---
# JEPA-Shield: Safe Diffusion Planning with Latent World Models via Conformal Shielding 

**Authors**: Taekyung Kim, Ege Yuceel, Steven Hong, Georgios Fainekos, Dimitra Panagou, Bardh Hoxha  

**Link**: [PDF](https://arxiv.org/pdf/2611.29609)  

**Abstract**: Latent world models enable planning from images, but their predictions do not directly establish whether a robot will satisfy physical constraints. We present JEPA-Shield, which connects a frozen action-conditioned Joint-Embedding Predictive Architecture (JEPA) world model to task-space safety enforcement. Two downstream models predict relative motion from imagined latent rollouts and decode unsafe geometry from the current image. Split conformal prediction turns end-to-end motion-prediction errors into multi-horizon uncertainty tubes. A backup-based shield checks these tubes against physical constraints and screens both candidate trajectories and final proposals in diffusion-inspired sampling-based model predictive control. Under the stated exchangeability and initial-backup assumptions, we establish probabilistic constraint satisfaction at action-block boundaries for a returned plan executed without replanning. We also introduce VParkBench, with geometric car and tractor-trailer parking tasks and a editable photorealistic Gaussian-splatting environment. In closed-loop evaluation, JEPA-Shield achieves success rates of 84.4%, 82.3%, and 99.0%, respectively, with no observed constraint violations. Ablations show complementary benefits from shielding and uncertainty calibration. The results demonstrate the importance of grounding latent world-model predictions in task-space quantities and explicitly accounting for prediction uncertainty in safety-critical planning. 

---
# KernelShift: Benchmarking Kernel Optimization Across Evolving Hardware 

**Authors**: Shuaiwei Huang, Weihua Du, Allen Wu, Ivan Fox, Manupa Karunaratne, Yiming Yang, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2611.29590)  

**Abstract**: Efficient AI computation depends on hardware that evolves over time, shifting the strategies needed to implement performant code. For CPU kernels, a key evolving factor is the set of available architectural extensions, which expose specialized hardware instructions that change the techniques used to optimize kernels. We study how effectively coding agents use these new, potentially unfamiliar extensions to write fast code. We introduce KernelShift, a benchmarking pipeline and a set of environments for evaluating coding agents in kernel optimization across evolving CPU architectures. Its agent-assisted, human-verified pipeline turns code repositories into benchmark instances, and its environments let agents optimize code with feedback from the target hardware. Using this pipeline, we construct 99 kernels from real-world, actively developed repositories (ncnn, this http URL, Arm SIMD Loops, and KleidiAI), including inference workloads derived from real model executions. KernelShift supports incorporating new kernels and hardware configurations as libraries and processors evolve. We evaluate agents with proprietary and open-weight models across progressively newer architectural extensions (Arm Neon, SVE, SVE2, and SME2), examining how performance changes as hardware evolves. On established extensions, agents beat the auto-vectorized compiler everywhere (2.4-19.8x) and exceed general-purpose libraries such as ncnn (getting up to 2.74x over its Neon kernels), but fall behind Arm's hand-tuned KleidiAI kernels (0.70x). In end-to-end inference of two LLMs, agent-written kernels achieve a 1.3-1.6x prefill speedup over this http URL's fast path. However, current agents do not yet reliably adapt to new extensions that require unfamiliar programming models, such as SME2's matrix tiles. 

---
# How Multi-Timescale Memory Cools Optimization: Theory and Evidence from AdEMAMix 

**Authors**: Jacopo Graldi, Antonio Orvieto, Anastasia Koloskova  

**Link**: [PDF](https://arxiv.org/pdf/2611.29588)  

**Abstract**: Augmenting AdamW with multiple momentum timescales improves large language model pretraining, yet the mechanism driving these gains remains poorly understood. We identify an interaction between multi-timescale optimizer memory and the geometry of normalized networks: memory changes parameter norms and thereby controls how fast parameters rotate. Starting from exact norm dynamics under decoupled weight decay, we show that the parameter norm depends on the temporal correlation of the updates, with weight decay controlling how long these correlations influence optimization. We show how an independently weighted slow momentum can substantially increase the norm of parameters, while contributing negligible power to the instantaneous update, therefore acting as an endogenous annealing schedule. We study this mechanism in AdEMAMix for language-model pretraining. Controlled interventions show that this scheduling mechanism has clear practical consequences and is an important factor behind AdEMAMix's advantage over AdamW. 

---
# From Learning Rates to Rotational Schedules: Finite-Horizon Theory on the Sphere 

**Authors**: Jacopo Graldi, Antonio Orvieto, Anastasia Koloskova  

**Link**: [PDF](https://arxiv.org/pdf/2611.29579)  

**Abstract**: How should step sizes be scheduled across a finite training horizon? Cosine and warmup-stable-decay (WSD) schedules are widely adopted in large-scale pretraining, yet their effect on optimization also depends on parameter norms. Normalization can make parameter groups scale invariant, so that the loss depends only on their direction, while their norms still affect how fast the parameters rotate. In this work, we formulate schedule design directly in the space of rotational motion. Working within a geodesically convex spherical setting, we derive a last-iterate convergence guarantee for arbitrary rotational schedules that separates their overall scale from their distribution over time. We derive the unique schedule that minimizes this bound and prove that it approaches linear decay as the horizon grows. Empirically, our rotational interpretation and idealized spherical analysis provide useful predictions for LLM pretraining in practice, including schedule comparisons missed by existing analyses. Our analysis predicts schedule ordering across model scales for both vanilla (Adam, Muon) and Hyperball-style (AdamH, MuonH) optimizers. Our Hyperball experiments support the predicted strong performance of linear LR decay and confirm our prediction that WSD underperforms in this setting. Accounting for parameter norm dynamics, we predict instead the strong performance of WSD for Adam and Muon shown in practice. Guided by our theoretical insights, we also derive a new LR schedule for Adam and Muon, which improves over the established cosine and WSD schedules. All in all, these results support the use of rotational dynamics to understand, predict, and ultimately guide scheduling in LLM pretraining. 

---
# Non unitary and nonlinear quantum generative large model design and fast large-scale training without back propagation 

**Authors**: Tianze Zhu, Xiaofeng Jiang, Min Liao, Luying Zhang, Ruixi Liu, Sen Kuang, Yongzhe Lin, Chenhao Zhou, Shuangwu Chen, Huasen He, Xiaobin Tan, Quan Zheng, Jian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29571)  

**Abstract**: Transformer-based large models require huge computational and storage costs. Quantum computation is expected to address that due to its potential to speed up classical algorithms exponentially. Key challenges of implementing quantum large models include the linear, unitary nature of qubit evolutions contrast with the non-unitary matrices and nonlinear functions in classical Transformers, as well as the difficulty of implementing gradient back-propagation on quantum hardware. Here we propose a fully quantum large model containing all the Transformer layers, implementing the non-unitary transformations through SVD-decomposition-based quantum circuits, and approximated nonlinear activation functions through pre-trained parameterized quantum circuits. We also propose a quantum circuit training algorithm based on multi-agent Reinforcement Learning, which can train large-scale parameterized quantum circuits without back-propagation. The dependency between the inference time complexity and the model's hidden dimension has been reduced from polynomial to logarithmic, showing an exponential advantage. The inference executions on the quantum hardware verify our model's 300-1000 times speed-up, and the measurements on the sequence-to-sequence tasks shows its learning ability. The training algorithm achieves only 25% complexity of the best SOTA quantum circuit training algorithms, especially this ratio decreases fast as the parameter scale increases. Our work not only proposes a quantum large model that can significantly reduce the computational cost, but also promotes the quantum computation towards general computing. 

---
# Think Before You Forecast: Scaling Test-Time Compute for Time Series Forecasting 

**Authors**: Daojun Liang, Xiao Hong, Pei Zhang, Yinglong Wang, Shuo Li, Jing Chen, Ying Guo, Hu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29566)  

**Abstract**: Test-time compute has become an important paradigm in computer vision and language modeling, where additional computation at inference time enables models to explore multiple candidate solutions and select or refine the most reliable one. However, time series forecasting models typically produce predictions through a single forward pass, leaving them vulnerable to distribution shifts and non-stationarity that frequently arise in real-world temporal data. We propose Time Test Compute for Time Series (TTC4TS), an architecture-agnostic framework that improves the predictive performance of frozen forecasting models at inference time. Given a pretrained backbone, the generator produces multiple candidate forecast trajectories using augmentation, retrieval, adapter. The synthesizer then selects or combines these candidates using aggregation, classification, or candidate-scoring and ranking methods to produce the final forecast. By increasing the number of candidates, TTC4TS scales test-time compute without retraining the forecasting backbone. Beyond improving point-forecast accuracy, TTC4TS can exploit candidate forecasts for uncertainty quantification, which supports improved interval forecasts. We also theoretically analyze when additional test-time computation improves forecasting. Experiments across diverse datasets and backbones demonstrate consistent gains in accuracy and calibration. Code is provided in this https URL. 

---
# Learning Acceptable Lotteries via Queries: Minimizing Aggregated Violation Distances 

**Authors**: Nicholas Teh, Paul W. Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2611.29555)  

**Abstract**: When deploying an AI system, a developer can randomize over several candidate versions, sending each request to a version chosen at random, to balance the requirements of multiple stakeholders. We study a setting in which each of $n$ stakeholders reveals only whether a proposed lottery over $m$ candidates is acceptable, and each stakeholder's acceptable lotteries lie on one side of an unknown hyperplane. While previous work sought a lottery that every stakeholder accepts, such a lottery need not exist when, for instance, safety and helpfulness requirements are in conflict. We therefore minimize a power mean of the stakeholders' violations; since binary answers reveal neither the size of a violation nor how to compare violations across stakeholders, we measure each violation by a normalized distance to the stakeholder's acceptance boundary. We characterize exactly when accept/reject answers determine these distances. When they do, $\mathcal{O}(nm\log(1/\varepsilon))$ queries suffice to learn them, where $\varepsilon$ is the precision of the utilities, and this bound is tight; when they do not, a natural worst-case approach is optimal up to a constant factor, even compared with methods that make unlimited queries. For limited query budgets, we give a sampling algorithm whose number of queries for a given accuracy does not depend on $n$, and show that a deterministic algorithm cannot guarantee a small loss without querying almost all stakeholders. Synthetic experiments illustrate the resulting tradeoff between querying many stakeholders and learning each of them accurately. 

---
# Detecting Is Not Choosing: What a World Model Needs to Keep a VLA Policy Safe 

**Authors**: Thanish Bolla  

**Link**: [PDF](https://arxiv.org/pdf/2611.29553)  

**Abstract**: Vision-language-action (VLA) policies complete most benchmark tasks but often violate safety constraints. Since they are stochastic, one remedy is to sample several candidate action chunks and execute one that a world model predicts is safe. We ask what such a model must do. In simulation, a lookahead safety filter with the simulator as an exact world model raises the safety rate of three VLA policies on LIBERO and of pi0.5 on LIBERO-Plus by 11.5 to 25.7 points. It also helps pi0.5 on three RoboCasa tasks. Learned critics trained on one policy's data detect dangerous states (pooled AUROC up to 0.864) but rank candidates at one state close to chance (within-decision AUROC 0.551-0.560), and the learned dynamics and video models we test do no better. When the simulator's state is perturbed, selection degrades long before detection. In the loop, a median position error of 3.18 mm removes 70% of the gain on LIBERO-Spatial, partly because the error also corrupts the displacement check. On LIBERO-10 the loss is smaller, and the intervals overlap. The median does not predict the cost: a pixel reader with a 0.56 mm median error keeps 6.5 points less of the gain than Gaussian error of a similar median. A world model for safety should thus be judged by within-state ranking, and its state must be accurate in the tail and on handled objects, not only in the median. 

---
# SCRA-CLIP: Spherical-Contextual Representation Adaptation for Zero-Shot Anomaly Detection 

**Authors**: Peng Chen, Chao Huang, Wenqiang Wang, Mingbo Yang, Yannan Chen, Yunkang Cao, Wei Wang, Li Shen, Fangjun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29545)  

**Abstract**: Recent advances in vision-language models have facilitated zero-shot anomaly detection (ZSAD) through transferable visual-semantic representations. However, detecting fine-grained anomalies remains challenging due to inadequate modeling of diverse anomaly patterns and their spatial context. To address these challenges, we propose SCRA-CLIP, a Spherical-Contextual Representation Adaptation framework that jointly enhances anomaly-aware textual semantics and context-sensitive visual representations. Specifically, we introduce Spherical Prototype-Conditioned Prompting (SPCP), which employs von Mises--Fisher mixtures to model the directional distributions of normal and abnormal patch features. These state-specific prototypes are incorporated into textual contexts to guide the learning of more discriminative textual representations. Furthermore, we develop Context-Deviation Guided Visual Adaptation (CGVA), which captures local feature deviations relative to multi-scale neighborhood contexts and refines patch representations through adaptive gating. By integrating distribution-aware semantic modeling with context-sensitive visual adaptation, SCRA-CLIP facilitates more discriminative cross-modal matching for fine-grained anomaly detection and localization. Extensive experiments on 11 industrial and medical benchmarks demonstrate the effectiveness and generalization capability of SCRA-CLIP across diverse unseen categories and domains. The code will be publicly available upon publication. 

---
# Seeing vs. Knowing: Counterfactual Supervision and Correction for 3D-Aware Object Editing 

**Authors**: Jian Zahng, Zhijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29542)  

**Abstract**: Rotating, translating, or rescaling an object in a photograph requires revealing unseen surfaces, preserving visible appearance, and updating shadows and reflections. Reconstructed 3D assets, rendered at the target pose as a 3D state, provide explicit geometric guidance, but their imperfect textures may conflict with the photograph, while standard paired training does not explicitly teach the editor which source to trust. To address this, we propose counterfactual state supervision, which constructs controlled photo--state conflicts through hidden-surface repainting, hidden geometry with re-rendered shadows and reflections, and visible-surface texture replacement. This teaches the editor to follow the 3D state for unseen content while preserving photograph-visible appearance. However, learning this distinction does not ensure that it is maintained during sampling. We therefore introduce symmetric counterfactual correction, which uses state-side and photo-side twins, which share the photograph and the state respectively, to construct directed latent perturbations toward condition-inconsistent targets and train recovery toward the target specified by the current condition. Both strategies operate only during training, requiring no architectural changes or additional inference steps. Experiments on OBJect and SpatialEdit-Bench demonstrate strong spatial editing performance and content preservation, while controlled ablations show improved source selection and sustained responsiveness to conditioning during sampling. 

---
# Adaptation or Poisoning? Security Risks of Conversational Test-Time Learning in Large Language Models 

**Authors**: Honghao Feng, Shanhan Qu, Kai Tong, Shuquan Man, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29532)  

**Abstract**: Test-time learning (TTL) enables large language models to adapt to unlabeled inputs during deployment. The model updates a lightweight adapter by minimizing the next-token prediction loss on the input it receives. When these updates persist across conversation turns, every user query serves both as the request answered in the current turn and as adaptation data for later turns. By choosing the queries used for adaptation, an attacker can influence parameter updates and weaken safety alignment through ordinary interactions with a TTL chat service. To demonstrate this risk, we introduce Progressive Test-Time Adaptation Poisoning (P-TAP), a query-only multi-turn attack that jailbreaks the model by poisoning its session adapter through the service's own updates. A sequence of escalating adversarial queries shapes the adapter turn by turn; the final harmful request is then answered under the accumulated state without itself being used for adaptation. Across four open-weight instruction-tuned models, enabling updates raises attack success rates on average by 13.00 percentage points under greedy decoding and by 22.25 points under best-of-10 sampling, relative to matched sessions receiving the same queries without updates. Answering adversarial queries but adapting on unrelated text or on benign text about the same topic removes most of this increase. Among existing safeguards, screening incoming queries flags every attack stream before the final request, while a perplexity-based detector of unsafe adaptation flags none of the successful attacks. 

---
# Recursive Self-Editing: LLMs Program Their Own Weight Edits for Self-Improvement 

**Authors**: Linghua Zhang, Yunhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.29513)  

**Abstract**: Recursive self-improvement asks a system to improve the very system that improves it. Today's language models improve themselves through data, training recipes or scaffolds. Whenever their weights change, gradient descent writes the update, not the model. We let the model write its own weights. In SELFWRITE, each generation the current model studies its own reasoning on fresh training problems, inspects its own weights and activations in a persistent Python session, and writes a program that constructs a low-rank weight edit for itself from white-box primitives (singular directions, activation contrasts between its right and wrong answers, rank-one projections, tensor computations), with no training or gradient tool. Every candidate is graded once on a hidden gate set against the current model; the best strictly improving edit is merged into the weights, and the edited model writes the next edit. On Qwen3.8-27B and math problems from DAPO-Math, five generations of SELFWRITE raise gate accuracy from 73.95% to 76.45% and accuracy on a held-out test split that the loop never sees from 76.30% to 77.75% (avg@4) and from 82.60% to 84.80% (pass@4), without a single gradient step. A language model can author useful changes to its own weights without gradients, and the edited model keeps writing useful edits. 

---
# MAWAM: Joint Imagination for Collaborative Robot Manipulation with a Multi-Agent World Action Model 

**Authors**: Rongze Tang, Chenyu Zhou, Jianjie Fang, Ziming Liu, Zhaolu Wang, Ziyou Wang, Xvyuan Liu, Wei Wu, Chen Gao, Yong Li, Zhibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.29501)  

**Abstract**: World action models (WAMs) excel at robot manipulation by imagining future observations and decoding actions from them, yet remain confined to a single embodied agent. In practice, manipulation often requires multiple robots to collaborate across disjoint workspaces, coupled timing, and heterogeneous embodiments. Such coordination demands both spatiotemporal alignment of future interactions and awareness of individual embodiment and role identities, whereas existing decentralized per-robot policies, flat centralized models, and multi-robot VLAs fail to unite both. To bridge this gap, we present MAWAM, a multi-agent world action model that jointly predicts the future observations of multiple embodiments and decodes each agent's actions from them for collaborative manipulation. Through an agent-axis curriculum, MAWAM expands a single-agent WAM along an agent dimension to coordinate teams under shared weights. Within one unified causal sequence, joint self-attention spans all agents to spatiotemporally align their imagined futures, while hierarchical language, identity encodings, and embodiment masks bind each stream to its role, identity, and embodiment. We further formalize multi-agent-necessary manipulation along three axes (why, when, and who) and develop MA-Task Harness, a coding-agent pipeline where mutually constraining generators and evaluators co-evolve customizable simulation-data and evaluation packages. Using this pipeline, we build MA-Bench across all axes with seven tasks, pairing joint success rate with role and collaboration scores for fine-grained analysis. On MA-Bench, MAWAM reaches 82.7% mean joint success, 30.7% above the strongest baseline, and achieves real-time closed-loop collaboration on a real heterogeneous robot team. 

---
# Learning Global Probabilistic Explanations 

**Authors**: Frédéric Koriche, Louenas Bounia  

**Link**: [PDF](https://arxiv.org/pdf/2611.29493)  

**Abstract**: Interpreting the predictions of complex black-box classifiers remains a central challenge in explainable artificial intelligence.
While local explanations clarify individual predictions, there is a significant need for global probabilistic explanations that capture a model's overall behavior across the input distribution.
Such an explanation is a small subset $K$ of features, whose relevance is measured by the expected probability, over a random input $\mathbf x$, that the classifier assigns the same label to $\mathbf x$ and to a random input agreeing with $\mathbf x$ on $K$.
In this study, we investigate the task of identifying maximally relevant global explanations under a cardinality constraint, focusing on product distributions with full support on discrete feature spaces.
We introduce a spectral explainer that leverages membership queries to the black-box classifier and employs Fourier-analytic, junta-learning methods to produce probably approximately correct (PAC) explanations.
Our algorithm is fixed-parameter tractable with respect to the explanation size limit, maximum feature cardinality, and desired accuracy.
Experiments on synthetic and real-world datasets confirm the theoretical guarantees and show that the spectral explainer outperforms a greedy baseline as the explanation size increases. 

---
# OnCache: Learning Dynamic Caching Schedules for Efficient Diffusion Inference 

**Authors**: Zhikang Xie, Xichen Ye, Yifan Wu, Yuxuan Liu, Qianlong Wang, Haoshen Yu, Peizhu Gong, Weizhong Zhang, Cheng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.29480)  

**Abstract**: Diffusion models have revolutionized generative tasks but incur high inference latency due to iterative denoising. Cache-based strategies accelerate inference by reusing intermediate features, yet they largely rely on static, sample-agnostic schedules. Our empirical analysis reveals two factors that these schedules overlook. (i) denoising dynamics and computational requirements vary across prompts, motivating adaptive computation allocation. (ii) sensitivity to caching errors varies across timesteps, so static schedules may reuse features at error-sensitive steps or waste computation at less sensitive ones. We therefore propose OnCache, a dynamic caching framework that makes prompt-adaptive decisions online about how much computation to allocate and when to refresh cached features. A policy is trained offline on paired compute and cache continuations from identical denoising states, using differences in terminal fidelity and the number of function evaluations (NFE) to account for the downstream consequences of each decision. To further improve the generation quality, a neural corrector compensates for caching-induced errors and mitigates their accumulation throughout the denoising trajectory. Both components are lightweight and introduce negligible additional inference latency. On FLUX.1-dev, OnCache achieves a nearly $3\times$ generation speedup at $1024^2$ resolution, with LPIPS of $0.156$ and SSIM of $0.860$. We further extend OnCache to video generation on Wan2.1-T2V-1.3B, achieving state-of-the-art efficiency--quality trade-offs across both modalities. 

---
# Separating Action Allocation from Image Matching in Visual Token Routing 

**Authors**: Chenxi Wang, Bin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29477)  

**Abstract**: Adaptive visual token routers aim to save computation while preserving the image detail needed to answer each question. A higher average score could reflect better choices for individual images or more frequent use of high resolution on tasks that need it. We distinguish the two by keeping action counts fixed within each task and comparing the router's assignments with random pairings of the same actions and images. On equally weighted ChartQA and TextVQA, both Qwen2.5-VL-3B and -7B gain from matching but lose more from their action frequencies relative to a task table, with a 1.71-point gain and a 3.95-point loss on 7B. Exchanging the continuous resolutions chosen by the external selector CARES also lowers quality on 7B. When requests arrive with task labels, the table outscores the discrete router by 1.92 points and saves 31.76 ms per complete 7B request on average. 

---
# Music Circuit: Trace and Control Style Dynamics in Large Music Generation Model 

**Authors**: Ruiji Yu, Wenye Ma, Gus Xia  

**Link**: [PDF](https://arxiv.org/pdf/2611.29475)  

**Abstract**: Musical style, a global attribute that functions throughout a whole piece, is a lever widely used by users in modern large music generation models. Yet how style is represented inside these models, and which components drive it, remains largely unexplored. In language models, circuit analysis traces a representation to a sparse sub-network of a few key components, namely attention heads and MLP neurons, and has been used successfully to explain and control global attributes such as emotion. In this work, we use circuit analysis to study the mechanisms behind musical style. Based on YuE2, we build a dataset that pairs the same lyrics with five styles and design a three-step framework. First, stage localization shows that the style forms mainly in the score-planning stage. Second, static observation across the positions of this stage further reveals that style is not a fixed vector but a dynamic representation across tokens. Third, we therefore merge the activation differences across positions into one direction per style, and trace the neurons and attention heads that drive this direction into a sparse circuit. Without any style text, steering this circuit successfully transfers one piece into the target style, matching and even exceeding the style prompt. To our knowledge, this is the first work to analyze and control musical style at the circuit level in a large music generation model. Audio and score examples are available at this https URL. 

---
# RSWP: Robust SO(3)-Equivariant Watermarking for Panoramic Images 

**Authors**: Pengzhen Chen, Yanwei Liu, Xiaoyan Gu, Jinxia Liu, Siwei Ma, Weiping Wang, Wu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29468)  

**Abstract**: Robust watermarking for panoramic imagery is impeded by the topological mismatch between spherical signals and planar representations. Conventional methods fail under $SO(3)$ viewing rotations due to non-linear geometric warping and struggle with viewport extraction, which typically necessitates computationally expensive inverse remapping. To address these challenges, we propose RSWP, a watermarking framework grounded in icosahedral group theory. Leveraging the property that the icosahedral group ($\mathcal{I}_{60}$) serves as the largest discrete subgroup of $SO(3)$, we decompose spherical rotations into tractable local planar rotations and combinatorial facet permutations. Accordingly, we introduce a decoupling architecture that embeds watermarks into discrete facets and a theoretically guaranteed Equivariance-to-Invariance decoding paradigm. By employing an equivariant feature extractor followed by a permutation-invariant sorting module, this paradigm rigorously synchronizes rotation-distorted features. Moreover, the dual-extractor design inherently supports direct, blind watermark retrieval from partial 2D renderings without geometric calibration. Extensive experiments demonstrate that RSWP achieves state-of-the-art robustness against continuous $SO(3)$ rotations and viewport transformations while preserving high visual fidelity. 

---
# Beyond Accuracy: A Diagnostic Benchmark for Hypothesis-Driven Experiment Planning in LLM Agents 

**Authors**: Ryo Kuroki, Amin Mansouri, Philippe Schwaller  

**Link**: [PDF](https://arxiv.org/pdf/2611.29467)  

**Abstract**: Evaluating large language model (LLM) agents on scientific tasks typically reduces to final-task accuracy, which obscures where reasoning succeeds or breaks down. We introduce a benchmark and diagnostic evaluation framework for hypothesis-driven adaptive experimental planning in closed-world system identification, instantiated in kinetic mechanism identification. The framework goes beyond final accuracy in two ways. First, we evaluate exploration behavior by quantifying the extent to which agents exhibit space-filling or corner-seeking patterns in the experimental design space. Second, we decompose the reasoning process into four interpretable steps: belief update, hypothesis discrimination, discriminative experiment design, and predictive reasoning under interventions, each scored separately on reasoning traces. Across ten LLM agents, we find that stronger agents outperform both adaptive (Bayesian optimization) and non-adaptive (Latin hypercube) baselines and exhibit exploration patterns inconsistent with naive heuristics. Decomposing reasoning reveals that the stronger agents score consistently well across all four steps, whereas weaker agents often succeed at belief update and hypothesis discrimination but are less reliable at translating them into actionable designs and sharp contrastive predictions. Independent expert annotations support predictive reasoning under our stated validation criteria, while evidence for discriminative experiment design is positive but less consistent. The results suggest a gap between recognizing plausible hypotheses and translating them into contrastive predictions, with more tentative evidence for a corresponding experiment-design bottleneck. The benchmark website is available at this https URL. 

---
# MoR4Rec: Mixture-of-Recursion for Sequential Recommendation 

**Authors**: Tiwari Ankit Satyaprakash, Adamya Shyam, Vikas Kumar, Bharti Rana, Venkateswara Rao Kagita  

**Link**: [PDF](https://arxiv.org/pdf/2611.29462)  

**Abstract**: Sequential recommender systems aim to predict a user's next interaction by modeling temporal dependencies in historical behavior sequences. To capture such dependencies, recent approaches increasingly rely on attention-based architectures, particularly transformers. While effective at modeling long-range sequential patterns, these models apply a fixed computation depth to all interaction items, treating them as equally complex. However, user behavior is inherently heterogeneous, with many interactions following simple repetitive patterns, whereas others require modeling higher-order dependencies. This mismatch leads to inefficient utilization of model capacity and unnecessary computational overhead, particularly in deep transformer-based architectures. To address this issue, we propose MoR4Rec, a sequential recommendation framework based on a Mixture-of-Recursion paradigm, which enables item-wise adaptive computation through a shared transformer backbone. A learned routing mechanism dynamically assigns each item to an appropriate recursion depth, allocating deeper computation to complex interactions while processing simpler patterns more efficiently. To improve computational efficiency, we develop a cascade-based strategy with key-value sharing across recursion steps. We further introduce a gated attention mechanism that adaptively regulates contextual information during recursive sequence modeling. Extensive experiments on three benchmark datasets show that MoR4Rec consistently outperforms strong transformer-based baselines, achieving average gains of 2.28% in Recall@5, 2.50% in NDCG@5, and 2.0% in MRR@5 over the strongest baseline. The improvements remain consistent across top-10 and top-20 recommendation cutoffs, while comparisons with vanilla transformers demonstrate the computational efficiency of adaptive recursive processing. 

---
# MobileNeTTT: a Lightweight DNN-based Time-To-Transit Estimator for Visual Navigation 

**Authors**: Chiara Boretti, Philippe Bich, Manuel Morteo, Roberto Tron, John Baillieul  

**Link**: [PDF](https://arxiv.org/pdf/2611.29454)  

**Abstract**: Time-to-Transit (TTT) is a perceptual cue, proxy for depth, that can be computed in the visual cortex, allowing animals to estimate the time remaining to move past features in the environment. We show how TTT can be estimated and used for bio-inspired navigation of a mobile robot equipped with a monocular camera. The paper describes a lightweight neural network estimation method, named MobileNeTTT, that (1) uses the full field of view and provides a rich array of visual feature TTT estimates; (2) is relatively insensitive to camera rotation; and (3) automatically adjusts to the geometry of the path and the presence of obstacles. Because TTT is a coarse visual cue, it is far simpler to use for robot navigation than current large visual depth foundation models. Our navigation model uses a 1.95M-parameter network that is two to three orders of magnitude smaller than well-known large vision foundation models, making it able to run in real time on edge hardware. Simulation and actual laboratory experiments validate the proposed approach in dynamic and spatially complex environments. Code, data, and video are available. 

---
# Agentic Discovery of Simple Symbolic Inventory Policies 

**Authors**: Xi Chen, Yuze Chen, Baichuan Mo, Jie Ying, Yuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.29440)  

**Abstract**: Designing simple inventory policies becomes increasingly difficult as operational features and system configurations expand. We develop an Agentic Inventory-Policy Discovery System based on explicitly defined simple operational formula (SOF) policy and template classes. Large language models (LLMs) revise symbolic structures, while a numerical engine fits parameters separately for each instance and evaluates the induced policies. Multicriteria parent selection, state-specific operational diagnostics, and reusable symbolic motifs guide structural evolution. A three-stage progression develops shared templates across instances and configurations with varying state dimensions. Experiments on 15,600 instances spanning 50 demand distributions cover lead times, perishability, their combination, and lead times with limited customer patience. The discovered templates improve aggregate empirical performance over evaluated SOF benchmarks and are competitive with policies requiring more elaborate online computations. For a lost-sales system with a three-period lead time, we develop an average-cost decomposition and prove that, under specified demand and cost conditions, a capped dual-shortfall specialization of a discovered policy strictly outperforms every policy in the considered capped base-stock benchmark class. This analysis links the performance advantage to the discovered replenishment mechanism. Together, these results highlight the potential of agentic discovery to generate simple inventory policies with strong empirical performance and identify operational mechanisms amenable to rigorous theoretical analysis. 

---
# HGDNA: Overlapping k-mer Tokenization for Efficient Autoregressive Genomic Language Models 

**Authors**: Xudong Wang, Siwei Cheng, Haozhe Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29400)  

**Abstract**: The domain of genomic language models (gLMs) has advanced rapidly, with models pretrained on diverse multi-species genomic corpora demonstrating their remarkable capabilities. While the effect of simple nucleotide-level tokenization on long-context modeling is well established, overlapping k-mer tokenization remains underexplored in causal pretraining, despite its rich neighborhood information. Here, we thoroughly revisit overlapping tokenization and introduce HGDNA, a hybrid linear-attention gLM with causal language modeling (CLM) pretraining and a shared token-prediction interface for downstream adaptation, using a species-classification auxiliary task and shared class tokens. HGDNA delivers competitive performance across sequence classification and frozen-embedding evaluations, supporting both short-range and long-range tasks with 30M parameters and 19B pretraining tokens. Code is available at this https URL 

---
# BQDNA: Multi-Head Binary Spherical Quantization for Genomic Sequence Modeling 

**Authors**: Xudong Wang, Siwei Cheng, Haozhe Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29395)  

**Abstract**: Decoding biological information from long, repetitive DNA remains challenging. Genomic language models (gLMs) often follow the conventional language-model pretrain-and-adapt paradigm. However, the lack of inherent semantic segmentation and prior rules makes hand-crafted tokenizers like k-mer and BPE suboptimal. We introduce context-aware \textbf{BQVAE} for scalar quantization of DNA and \textbf{BQDNA}, an autoregressive gLM supporting multi-head codebooks. Comprehensive experiments reveal that: (1) unique training dynamics of vector quantization in DNA space, including the importance of alignment between training corpus and downstream data. (2) superior performance of BQVAE over hand-crafted tokenizers and vanilla VQVAE. (3) remarkable cross-species generalization capability of BQDNA, which achieves state-of-the-art performance with merely 0.1B multi-species training for BQVAE and 1 epoch training on hg38 for BQDNA, outperforming gLMs pretrained with 10$\sim$100x more tokens. Our findings underscore the potential of learned vector quantization for genomic sequence representation. Code is available at this https URL 

---
# Gaussian Belief Propagation and Periodic Jacobi Matrices on Trees: Spectral Curve, Antibound Poles, and Fixed Points 

**Authors**: Yusuke Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2611.29381)  

**Abstract**: Diagonal Green functions of a periodic Jacobi operator on a universal covering tree satisfy the scalar Gaussian belief propagation covariance equations on the finite connected loopless base multigraph. At cycle rank at least two, the Green functions continue to simple antibound poles at both Perron values for arbitrary real potentials and nonzero real weights. This proves the conjecture of Christiansen, Simon and Zinchenko. For positive weights the pole result includes self-loops. On loopless graphs the lower pole is the walk-summability threshold. Each energy strictly between it and the spectral bottom has at least two fixed points with positive definite local beliefs. On a connected simple graph, some coefficients yield a compact component of positive solutions exactly when the graph contains two vertex-disjoint connected subgraphs of cycle rank at least two. The covariance curve is generically irreducible, with one sheet per edge subset with no vertex of degree one. For cycle rank at least two and generic real coefficients, the diagonal Green functions, infinite half-tree $m$-functions and the density-of-states Stieltjes transform all have this degree, answering a question of Avni, Breuer and Simon. Then all eigenvalues of the signed Jacobi matrices are antibound poles. Signed principal submatrices classify finite points with nonzero residue of $(\sum_i V_i) dz$, where $V_i$ are the vertex variances. We give a genus formula. Near zero potentials and unit weights on $K_4$, independently varying the coefficients leaves the cover spectrum an interval. This refutes Avni, Breuer and Simon's conjecture on the density of parameters with all gaps open. In block models, generic irreducibility and a uniqueness criterion extend, and at uniform dimension $p$ the degree can exceed $L^p$ for scalar degree $L$. Part of the scalar theory is formally verified in Lean 4 with Mathlib. 

---
# MusicListenBench: Can Audio LLMs Hear the Difference Yet? 

**Authors**: Abhinaba Roy, Ambuj Mehrish, Dorien Herremans  

**Link**: [PDF](https://arxiv.org/pdf/2611.29375)  

**Abstract**: Audio large language models (LLMs) are typically evaluated on song genre, mood, and instrumentation. A more basic skill is telling whether two short clips play the same melody, chord or instrument. We introduce MusicListenBench, a benchmark of paired listening tests for melody, harmony, timbre, and tempo. Each test item comes with two variants. A FLIP variant changes the music, so the answer must change. A STAY variant changes only the sound, such as room echo or hiss, so the answer must stay the same. The two variants have opposite answers, so a model that ignores the audio scores 50%, whatever letter it prefers. The benchmark has a training split and two test sets, one rendered from symbolic music with exact labels and one built from real recordings. On the rendered test, seven of eight open audio LLMs score within 4 points of this floor. Of four commercial models, both GPT-Audio models are at chance, and the best, Gemini 2.5 Pro, reaches 74.2%, 12.8 points below human listeners (87.0%). Post-training with GRPO on 10k rendered clean pairs raises Qwen2.5-Omni from 61.2% to 99.0% on held-out clean pairs, and the gain transfers to a task left out of training. On FLIP/STAY, however, it only reaches the level of our listeners (86.0%). It now misses almost no change (1.9%, against 24.7% for Gemini 2.5 Pro), yet when only the sound changes, it turns a correct answer into a wrong one 25.7\% of the time. More than half of its false flips come from a quiet background hiss. Models learn to hear a difference much faster than they learn which differences matter. On real recordings, open models again stay near chance, and the trained model calls 27 of 42 pairs of two strums of the same chord different. Adding sound-only pairs to training lowers false flips on both test sets, but no training run teaches relative pitch. We release the data, code, and an open leaderboard. 

---
# DeciForm: Incorporating Task Structure into Decision Models 

**Authors**: Qiuyu Tian, Haojie Yin, Yingce Xia, Youyong Kong, Zequn Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29340)  

**Abstract**: Decision models unify candidate selection, Boolean judgment, and numerical scoring within a single model. Their answer spaces nevertheless carry different structures: classification candidates form an unordered set, while numerical scores have an intrinsic order and distance. We introduce Deciform, which incorporates these distinctions into computation and supervision. Candidate-free queries and isolated candidate branches support consistent classification under reordering. A log--spherical objective trains candidate probabilities, while a decimal-prefix readout combines individual-value and interval-weighted cumulative supervision for numerical scoring. Shared adapters, task-private residuals, and output-channel gains support joint adaptation. On a shared 15,000-request test set, Deciform improves all three decision metrics on Qwen and ModernBERT; source-normalized errors support the Score gains across the data mixture. Canonicalized joint encoding also eliminates tested order flips, while isolation further improves prediction metrics. Separate objective studies and main-configuration ablations examine the classification and numerical components. After one epoch of MIND-small adaptation, Deciform improves multi-positive news ranking on both backbones. These results support matching encoding and supervision to answer-space structure within a unified decision interface. 

---
# More Than Picturing the Future: Motion-Oriented Latent Reasoning for Generalizable Robot Manipulation 

**Authors**: Xin Zhang, Yushuo Hou, Anqi Zhao, Moyun Liu, Xingwei Zhao, Bo Tao, Jiawei Du, Joey Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.29339)  

**Abstract**: Latent reasoning lets Vision-Language-Action (VLA) models deliberate at a fraction of the cost of explicit chain-of-thought, increasingly through visual foresight, which we show to be indispensable for predicting actions. Yet this paradigm is inherited from Vision-Language Models, where it serves understanding rather than action: each latent is aligned with an isolated future frame, often enriched with further modalities of that frame. We find that this binds the policy to the appearance of its training scenes, so that it degrades sharply once the scene changes. We argue that visual latents for action should capture not what a future frame looks like, but how the scene evolves toward the goal. We therefore propose Motion-Oriented Latent Reasoning, which redirects visual latents from the whole frame to the interaction that the action must bring about. Semantic latents, following the stages of embodied chain-of-thought, first establish what to accomplish and how. Conditioned on them, motion-oriented latents anticipate each future step from coarse to fine: from the global scene, to the interaction region spanned by the manipulated object and its destination, to the robot state. Relative to the state-of-the-art latent VLA LaST$_0$, our method raises the mean success rate on SimplerEnv-WidowX from 37.5% to 55.2% in distribution and from 22.4% to 30.2% under out-of-distribution shifts, and on an AgileX Piper dual-arm platform from 60.8% to 90.8% in training scenes and from 6.7% to 37.5% under environmental changes. 

---
# FP-Prune: Rethinking Visual Token Pruning from an Information Gain Perspective 

**Authors**: Haozhe Zhao, Junxiang Zhang, Junxin Wang, Yuan Ge, Xiangchen Song, Dingyang Zhang, JingBo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.29327)  

**Abstract**: Visual token pruning can reduce the inference cost of vision language models, but its effectiveness depends on preserving complementary visual information under a limited token budget. A central challenge is to quantify the additional information a candidate token contributes beyond the retained set, a contribution that conventional scores based on attention or similarity struggle to capture directly. In this work, we revisit visual token pruning from the perspective of information gain and use mutual information to characterize the incremental value of candidate tokens. We further establish theoretically that frame potential serves as an effective surrogate for mutual information. Building on this observation, we propose FP-Prune, a training-free visual token pruning method that iteratively removes tokens with low information gain to preserve as much complementary visual information as possible. Experiments on three widely used VLMs and eight datasets demonstrate that FP-Prune outperforms baseline methods in both accuracy and inference efficiency. Further analysis shows that the surrogate objective closely aligns with model performance. Our work provides theoretical grounding and an effective method for visual token pruning. 

---
# SWIFT: A Handcrafted, Arithmetic-Minimal FPGA Engine for Real-Time Audio Deepfake Detection 

**Authors**: Chirantan Banerjee, Soumita Chatterjee, Bhargab B. Bhattacharya, Hafizur Rahaman  

**Link**: [PDF](https://arxiv.org/pdf/2611.29315)  

**Abstract**: Generative voice cloning threatens voice-based authentication, yet current deepfake detectors rely on power-hungry, high-precision models unsuited to edge hardware. We present SWIFT (Streaming Whiten-Integer Forensic Trust-Engine), a handcrafted detector that works in the high-frequency noise domain where synthesis artifacts concentrate. A first-order integer high-pass residual drives nine families of fixed-point descriptors, among them integer lifting-wavelet cepstral quantiles, Mahalanobis anomaly distances, Poincaré phase-space geometry, and phase jitter, fused by calibrated gradient-boosted trees and a sparse L1-regularised meta-learner. Under leakage-controlled nested cross-validation, with one shared preprocessing chain for real and fake audio, this reference model reaches accuracy of up to 99.86% and an Equal Error Rate (EER) as low as 0.03% across eight benchmarks. Its integer front-end and a fixed-point linear classifier run on the programmable logic of a PYNQ-Z2 (Zynq-7020) board, with a 7.85 microseconds per-frame datapath latency (127.4k frames/s) at 1.734 W and 11.2% LUT utilisation. Integer-only audio deepfake feature extraction is thus feasible at the edge without pretrained neural models or floating-point units. 

---
# PrimiTok: Learning to Diversify and Allocate Primitives for Visual Tokenization 

**Authors**: Yizhu Chen, Chen Ju, Donghao Zhou, Ziyu Guo, Shuai Xiao, Jinsong Lan, Xiaoyong Zhu, Ying Chen, Haodong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.29311)  

**Abstract**: Unifying visual understanding and generation requires representations that preserve both semantic information and fine-grained image details. Continuous tokenizers trained primarily for semantic alignment may omit fine-grained reconstruction details, whereas discrete tokenizers face information loss from codebook quantization. Conventional multi-codebook designs assign every token the same number of codewords regardless of local content. We instead view codewords as reusable visual primitives whose allocation and composition adapt to each token. We introduce PrimiTok, a compositional visual tokenizer that represents each visual token through adaptive allocation and weighted composition of learned codebook primitives. PrimiTok integrates two complementary components: Diverse Quantitative Primitives (DQP), which encourages diversity among sub-codebook centroids to reduce redundancy, and a Dynamic Primitive Allocator (DPA), which selects primitives according to token complexity. Embedding quantization error supervises how many primitives are selected from each sub-codebook, and DPA aggregates them with input-dependent weights. Joint reconstruction and image--text contrastive learning encourage the resulting representations to retain visual fidelity and semantic alignment. Experiments demonstrate improved reconstruction and semantic alignment over SOTA baselines, with further evaluations on multimodal understanding and visual generation. 

---
# From Vanishing Gradients to Recursive Self-Improvement: Evaluating the Most Influential Papers in Large Language Model Evolution (1997-2026) 

**Authors**: Saadh Jawwadh  

**Link**: [PDF](https://arxiv.org/pdf/2611.29303)  

**Abstract**: The rapid proliferation of Large Language Model (LLM) research makes it increasingly difficult to distinguish incremental refinements from true paradigm shifts. This review evaluates 22 of the most influential publications spanning nearly three decades (1997-2026), organized through a causal, chronologically anchored taxonomy centered on the author's birth year (1999). Beginning "before 1999" with how recurrent neural networks (RNNs) forgot earlier context (much like forgetting Chapter 1 by the time a reader reaches Chapter 9), we trace how Long Short-Term Memory (LSTM) networks introduced a "bouncer" mechanism of forget, input, and output gates to selectively preserve long-term memory. We then examine how Word2Vec turned language into vector mathematics (king - male + female ~ queen), why sequential RNNs hit a parallelization wall, and how the Transformer's all-to-all attention weights launched the modern LLM era alongside a quadratic bottleneck. Next, we analyze why scaling parameters alone leaves capacity under-trained unless data scales equally (Chinchilla), how Reinforcement Learning from Human Feedback (RLHF) left alignment superficial, and how that gap compounded into adversarial jailbreaks and single-direction refusal subspaces (J-space), where instructing a model not to think of a restricted concept surfaces it much like the human "red ball" paradox. Finally, we synthesize the current state of AI (2024-2026), encompassing recursive model collapse, autonomous reasoning-model jailbreak agents, machine-speed agentic intrusions, and closed-loop recursive self-improvement (AIDE^2). 

---
# Beyond Omission: Measuring Critical-Information Fulfillment of Large Language Models in Bengali and English 

**Authors**: Avijit Roy  

**Link**: [PDF](https://arxiv.org/pdf/2611.29299)  

**Abstract**: Evaluations of multilingual trustworthiness often ask whether a model leaves out information that a user needs. We examine whether that question is sufficient. Using 157 matched tasks written in English (EN), natural Bengali (BN) and Bengali-matrix Bengali-English code-switching (BN-CS), we scored 1,413 responses from three models (GPT-6.1 Sol, Qwen3.5-9B and Gemma 4 12B) against required information units (RIUs) fixed before any model was run. The scoring separates outright omission from content that is present but incorrect, insufficient or only partly met. The prespecified primary estimand, the domain-equal paired difference in critical omission rate (BN - EN), showed no detectable difference (+0.003, 95% CI -0.014 to +0.015), with omission near zero in every condition. The Bengali deficit appeared in fulfillment instead. Critical coverage was 0.050 lower in Bengali (95% CI 0.022 to 0.079) and strict complete-task success 0.093 lower (95% CI 0.055 to 0.131); both gaps came from the two open-weight models, and Sol showed none. Code-switching recovered part of the loss in RIU coverage (+0.028, 95% CI 0.012 to 0.045). On these data, an omission count alone would have shown no language gap. The automated judge passed prespecified agreement gates after a documented failed first attempt. A blind native-speaker rating of 60 study responses agreed with it on RIU scores (alpha 0.84) and critical RIUs (98% exact) but fell short on Bengali RIU agreement (alpha 0.77) and on the unsupported-content check (93% exact), and a blind check of its reason codes agreed on whether a critical unit was omitted for 79 of 81 units. Data, rubrics, responses, scores and code are released. 

---
# GALILEO: Geometric RoPE with Galilean Key Transport for Dynamic Scenes 

**Authors**: Youngmin Kim, Seonghyun Jin, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2611.29293)  

**Abstract**: Geometry-aware rotary positional encodings, such as RayRoPE, help attention reason about the 3D spatial relationships between tokens and the query camera. However, they implicitly assume a static scene, encoding each key at its observed position regardless of when it was captured. For dynamic scenes with temporally separated input views, this mismatch places the same moving object at a different position in each input frame, so it can be rendered at an inaccurate position. While temporal encodings inform attention when a token was observed, they do not resolve where its content should be at the query time. We introduce a simple and general approach that aligns positional encodings with dynamic scene geometry through Galilean transport. Before positional encoding, each key's 3D position is transported to the query time using its estimated velocity and the temporal offset. By modifying only key positions, our method integrates directly with existing encodings, including RayRoPE and URoPE, without altering their formulations. Velocities can be obtained from arbitrary sources. By default, we estimate them from token correspondences within the network. Integrated into LVSM, our method improves dynamic view synthesis on Syn4D, dynamic Objaverse, and real-world Stereo4D videos over baselines. Ablations show that encoding velocities as token features yields smaller gains, while attention visualizations reveal that our method attends to query-time locations of moving content, unlike RayRoPE. Beyond view synthesis, our method also reduces tracking errors on dynamic objects in a pretrained 3D point tracker and improves moving-vehicle generation in a video generator. 

---
# Which Failure Does an Intervention Repair? When Agent Interventions Help, and Where Self-Evolving Skills Stall 

**Authors**: Zihan Jin, Gaoyuan Du, Jiate Li, Dongze Ye, Shawn Li, Yue Zhao, Defu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.29290)  

**Abstract**: An intervention that improves one agent can break another, and adding a stronger harness on top need not repair the damage. Agent interventions such as skills, harnesses, and extra inference budget are nonetheless usually credited with one average gain, measured under one setup. We ask instead when each intervention helps and which failure it repairs. A matched factorial study over nine benchmarks, four models, four skill conditions, four harness levels, and three budgets, with nearly 0.9 million trajectories, changes one axis at a time. No configuration is best everywhere, interventions interact rather than add, and a harness can absorb a skill's gain, rescue a failure the skill causes, or leave the gain intact. The failure left by the current system points to the fix: missing knowledge to a stronger model, execution errors to a richer harness. Yet feedback is not recovery: a matched recovery repairs mainly failures that leave a visible error signal. Skills that evolve from the agent's own experience hit a bottleneck: they rarely beat a short human-written skill, can learn dataset shortcuts, break when moved to another model, and cannot supply missing facts, which outside information does. A benchmark-level switch to the model favored by a 20-task labeled pilot achieves both higher performance and lower cost than always using the stronger model. Agent interventions should be judged by which failure they repair and under which configuration. 

---
# LLM-Based Assistant for Dog Care Guidance 

**Authors**: Abubakar Sadiq Shittu, Venkat Gopu, Arshita Sharma, Jacob Kennedy  

**Link**: [PDF](https://arxiv.org/pdf/2611.29286)  

**Abstract**: This study investigates whether domain grounding through Retrieval-Augmented Generation (RAG) and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) can enhance the accuracy and reliability of pet care advice generated by large language models (LLMs). We curated a dataset of breed-specific dog care information and used it to develop two systems: a RAG pipeline based on TinyLlama-1.1B-Chat-v1.0, and a fine-tuned LLaMA-3.2-1B model adapted using LoRA. Both systems were evaluated on 15 dog care questions. The RAG model achieved a perfect average score of 2.0, outperforming the fine-tuned model (mean = 1.47). Statistical analyses, including Welch's t-test, Mann-Whitney U, and Wilcoxon signed-rank tests (all p < 0.05), confirmed the significance of the performance gap. Our findings suggest that retrieval-based domain grounding leads to more accurate, consistent, and trustworthy responses than standalone fine-tuning in specialized LLM applications. 

---
# QASparse: Query-Adaptive Sparse Attention with Tailored Budgets and Tokens 

**Authors**: Zhiyuan Chen, Zihao Yang, Quanjiang Li, Haijiang Li, Jiancheng Pan, Qichao Zhao, Zihan Dong, Qiushi Shi, Jinshi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29280)  

**Abstract**: Long-sequence inference in large language models is constrained by the high cost of prefilling, where dense attention scales quadratically with context length. Exploiting similarity across attention heads offers a way to reduce redundant computation. Prior approaches in this direction have relied on localized query observations to determine head-level computation reuse or budget allocation. However, inter-head similarity does not imply uniform query requirements, leaving adaptation to varying budget needs and selection preferences across query regions a key challenge. We introduce QASparse, a Query-Adaptive Sparse attention mechanism with hardware-efficient execution. QASparse preserves the efficiency of shared cross-head importance estimation while adapting computational budgets and token selections to query requirements, and it executes these dynamic sparse decisions through dedicated kernels. Its query adaptivity stems from two complementary innovations: Local Budget Regression models local variations in computational demand along query positions to estimate budgets for different query regions and attention heads; Boundary Voting lets queries within each head vote over candidates near the selection cutoff using actual query/key scores, refining the retained set while preserving the initial selection size. To reduce the execution overhead of dynamic budgets, we further design length-specialized selection kernels that efficiently support query-adaptive sparse patterns across context lengths. Together, these designs efficiently combine shared estimation with local decisions while adapting both how much attention is computed and which interactions are retained. Experimental results demonstrate improved benchmark scores and faster prefill attention computation compared with existing sparse attention baselines. 

---
# DriveOddsBench: Are Driving Video World Models Aligned with Real-World Odds? 

**Authors**: Jiaxi Liu, Keshu Wu, Bin Ran, Xiaopeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.29275)  

**Abstract**: Video world models predict how a scene will evolve, and their sampled futures are increasingly used in place of the real one, from planning to simulation. These uses assume that the samples are aligned with real-world odds, with each future appearing among them as often as it would follow the same history in reality, yet current evaluations only score how real each sample looks. Testing this assumption on real driving data is hard for three reasons. Each recorded history has only one real future, so the true probabilities are unknown. Physical quantities must be read from video, where reading errors can look like model errors. Finally, the many possible analyses invite selective reporting. We introduce DriveOddsBench, a pre-registered benchmark that addresses all three. First, if the samples follow the real distribution, the rank of the real outcome among them is uniform in every history, so pooled ranks test them without probability labels. Second, one validated instrument reads the same quantities, such as the lead-vehicle gap and the ego motion, from the generated futures and from the real future, which first passes through the world model's autoencoder so that both are read alike. Third, all confirmatory analyses were frozen before any test clip was measured. We test eleven general and driving world models on 200 held-out Argoverse 2 histories in five scenario families, with 35,200 generated futures. None is aligned, and every model underestimates its uncertainty. The real outcome falls outside the range of all 16 samples in 28-61% of cases instead of the 11.8% expected under the real distribution. Five models are significantly skewed, in opposite directions, and the largest errors are in the ego motion. Nine models beat climatology without being aligned, and among these models neither type nor scale predicts which comes closest. 

---
# Dually Constrained Parameter Projection for Continual Learning 

**Authors**: Bowyn Tan, Zhilong Zheng, Yang Guan, Liming Xiao, Liye Tang, Xiao Li, Naizheng Wang, Keqiang Li, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.29269)  

**Abstract**: Continual learning requires a model to learn multiple new tasks sequentially while retaining performance on previous tasks. Projecting model parameters onto the Jacobian null space of previous tasks after unconstrained training on a new task has emerged as an effective paradigm. However, unidirectional projection merely minimizes the parameter distance to the new-task optimum, overlooking the new-task loss landscape. To address this limitation, we propose DUET, a bidirectional projection algorithm that imposes Jacobian null-space constraints from both previous and new tasks during post-training projection. Specifically, DUET formulates the post-training adjustment as a dually constrained optimization problem. It minimizes the distance between parameters before and after training on a new task, subject to loss preserving constraints on both tasks. To solve the optimization problem, we develop an alternating bidirectional projection algorithm, whose convergence is guaranteed by the contraction mapping theorem. For reliable multi-step projection, we rectify existing step size selection criteria via an effective-rank aware formulation, overcoming their failure under rank-deficient Jacobians. Extensive experiments demonstrate that DUET consistently outperforms existing unidirectional projection methods and achieves state-of-the-art performance on multiple datasets. 

---
# On the Interplay between Backtracking and Latent Confidence Directions in Reasoning Models 

**Authors**: Stephen Cheng, Zahra Sodagar, Sandeep Soni, Dinesh Manocha, Sarah Wiegreffe  

**Link**: [PDF](https://arxiv.org/pdf/2611.29265)  

**Abstract**: Reasoning models deliberate over multiple answer proposals, often resulting in excessively long traces due to backtracking, even when the correct answer has already been reached (Across three 32B models, ~87% of rollouts contain 1 unique answer yet still make ~8 answer proposals on average). Given this potentially "decorative" chain of thought (CoT) reasoning, can we look to the model internals to determine if the model will backtrack or commit to an answer proposal? Operating under the hypothesis that models maintain an internal representation of their uncertainty, we seek to extract this representation and characterize its relationship with backtracking behavior at answer proposals. By querying the model's verbalized confidence following answer proposals throughout a response trace, we extract a linear direction for latent confidence that separates high and low verbalized confidence. This direction can steer verbalized confidence as expected, but it surprisingly can also steer the next token probability of backtracking, performing on par with a linear direction extracted directly from backtracking examples. We then test the direction under progressively more realistic interventions. Steering continuously from a mid-trace answer proposal modulates response length (~-1.3k - ~+2.5k tokens) and the number of answer proposals (-4.18 - +4.75) in both directions, with minimal cost to accuracy. Applying the confidence direction from the start of generation, where proposal positions are unknown, still shortens traces at comparable accuracy, and its effect surfaces in the linguistic confidence of proposals. Overall, our study reveals the intricate interplay between a model's internal confidence and its proclivity to backtrack or commit to an answer. 

---
# Look Back: Reinforcement Learning with Counterfactual World Models 

**Authors**: Jinwoo Kim, Toghrul Abbasov, Lucius E.J. Bynum, Aahlad Puli, Stephanie Milani, Hyunwoo Ryu, Seunghoon Hong, Kyunghyun Cho, Rajesh Ranganath  

**Link**: [PDF](https://arxiv.org/pdf/2611.29259)  

**Abstract**: Agents in complex, stochastic environments must contend with dynamics that can exceed their world models' predictive capacity and randomness that obscure the consequences of their actions. We hypothesize that, compared to imagining entirely new trajectories in such complex environments, it can be easier to transform observed trajectories with alternative actions. Counterfactual world models condition on observations to predict alternative outcomes under the same underlying environmental randomness, using experience as a reference for predicting action-dependent changes. We present a policy gradient estimator, REINFORCE Look Back (RLB), that uses counterfactual returns as baselines to distinguish action effects from environmental luck, and prove conditions for unbiasedness and variance reduction. In settings where environmental evolution is harder to predict than action-induced changes, we show that counterfactual world models are easier to learn and support better policy learning than ordinary world models. Our findings suggest that counterfactual thinking can help agents learn effective behavior without independently reproducing the full complexity of their worlds. 

---
# Denoised but Not Improved: What EEG Denoising Does to a Downstream Task 

**Authors**: Xianglong Bao, Zhe Wang, Kewen Wang, Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.29256)  

**Abstract**: Scalp electroencephalography (EEG) is routinely denoised before it reaches a downstream model, and the denoising literature scores a method by how closely its output matches a clean reference waveform. That reference does not exist for real scalp recordings: it is synthesized by adding artifact templates to signals merely presumed clean, and even an exact separation would return cortical activity blurred by volume conduction rather than the activity itself. Fidelity to it is therefore an unvalidatable objective, and whether it predicts any downstream benefit has gone largely unexamined. We report an audit that examines it. Holding a seizure-detection pipeline fixed and varying only the preprocessing stage, we find that five standard denoising methods, spanning analytic, cross-channel and supervised deep families, all fail to improve on the unprocessed signal, and most degrade it. To say what such a stage must instead do, we formalize a label-only, classifier-free probe that reads it by the separability geometry it induces on task-relevant features, decomposing any change into a between-class distance and a within-class scatter term. On a transform that does improve the task, the probe attributes the entire gain to a roughly one-quarter reduction of within-class scatter, with class separation statistically unchanged. What a downstream model benefits from is therefore variance suppression on task-relevant structure, which waveform-fidelity metrics neither measure nor reward. 

---
# Tissot indicatrix for location encoders: Characterizing local distortion in learned geographic embeddings 

**Authors**: Guofeng Cao, Gengchen Mai  

**Link**: [PDF](https://arxiv.org/pdf/2611.29249)  

**Abstract**: With the increasing adoption of geographic artificial intelligence, location encoders are widely used to map geographic coordinates to learned embeddings for tasks in Earth observation and geospatial modeling. Similar to map projections, location encoders can distort geographic space, that is, equal-distance displacements in different directions can result in unequal embedding changes. However, such distortion remains largely unexamined, and it can propagate into downstream applications. We adapt Tissot's indicatrix, a cartographic tool for map projection distortion, to characterize this distortion. At each location, we compare encoder outputs at equal distances in 32 directions and fit a tensor, represented by an ellipse, whose elongation (anisotropy) and orientation describe the local distortion. The fit requires only encoder outputs without access to model internals and is validated against references with known geometry. By applying the method to thirteen released checkpoints of five encoder families, we find that distortions vary across space and inherit the input geometry to varying degrees, closely in GeoCLIP with its Equal Earth input (anisotropy correlation of 0.88) and weakly or moderately in the other encoders. The distortion also affects the locations an encoder treats as neighbors. Within 1km of 20 query sites, the 40 nearest neighbors in geographic and embedding space overlap by a median of only 57.5--87.5% across the checkpoints, while the fitted ellipses select exactly the same embedding neighbors in 265 of 299 comparisons. These findings provide a geographic basis for assessing location encoders, and the accompanying eTissot package implements the methods for user-specified encoders. 

---
# Seeing Is Not Acting: WARD Diagnoses the Evidence-to-State-to-Action Gap in Longitudinal Clinical Agents 

**Authors**: Chao Ding, Yidong Jiang, Junzhi Ning, Juncheng Wu, Yuhan Wang, Renjie Lu, Minjia Yuan, Gaopeng Wang, Liyang Wang, Pengcheng Chen, Dong Yin, Meiping Wang, Jinru Ding, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.29243)  

**Abstract**: Large language models for longitudinal clinical care must revise their assessments and actions as patient information accumulates. However, benchmarks centered on isolated responses or final task outcomes provide limited insight into intermediate failures in evidence use, state updating, and action selection. We introduce WARD, a benchmark for evaluating how clinical agents translate accumulating evidence into patient-state judgments and timely, safe actions. WARD comprises 4,560 clinical tasks across seven clinical output domains, organized into 240 checkpoints from 40 longitudinal clinical trajectories spanning 95 hospitalizations. Its Longitudinal Track evaluates these tasks under controlled cumulative evidence. Its Interactive Track evaluates end-to-end, tool-mediated clinical management across 271 verifiable evidence items and 141 reference transitions. Across 12 models with scorable checkpoint outputs, evidence recognition consistently outpaces downstream state and action decisions, with linked-evidence nonresponse reaching 74.47--90.90\%. Models also struggle to retain and update information over time, while in the Interactive Track only 11.46\% of executed actions advance a reference transition. These results reveal a persistent evidence-to-state-to-action gap and highlight the need for process-level evaluation of longitudinal clinical this http URL and data: this https URL. 

---
# Tenlai-TTS 2.0: Continuous-Representation LLM-Based TTS System with End-to-End Training 

**Authors**: Changfeng Gao, Yong Ren, Jun Yuan, Zhengkun Tian, Ye Bai, Zhao You, Shidong Shang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29226)  

**Abstract**: Autoregressive text-to-speech (TTS) with continuous speech latents avoids quantization-induced information loss, but typically relies on a latent space that is learned separately and then frozen before language-model training. Such a space must satisfy two competing requirements: it must retain enough acoustic detail for high-fidelity reconstruction, yet be sufficiently structured and regularized to support autoregressive prediction by a language model. We present Tenlai-TTS 2.0, which learns continuous speech representations through end-to-end training to balance acoustic detail with autoregressive predictability. The model comprises a speech encoder with a capacity-controlled Gaussian bottleneck, a large language model (LLM) with an autoregressive diffusion transformer (AR-DiT) head, a flow-matching (FM) mel-spectrogram decoder, and a multi-task recognition model (RM). With energy normalization and fixed-variance Gaussian noise, the bottleneck has a closed-form upper bound on its information rate. Through joint optimization with reconstruction, recognition, and next-patch prediction objectives, we learn continuous speech representations that suit all three tasks and train the LLM and AR-DiT for autoregressive generation. The LLM and AR-DiT are further optimized using differentiable reconstruction and recognition feedback on their predicted latents through the FM decoder and RM. On Seed-TTS-eval, this feedback-based optimization reduces character error rate (CER) from 0.82% to 0.74% on test-zh and from 8.32% to 7.08% on test-hard-zh, and word error rate (WER) from 1.32% to 1.27% on test-en, compared with joint training alone. Subsequent reinforcement learning further reduces CER to 6.29% on test-hard-zh and WER to 1.08% on test-en, with speaker similarity scores of 0.811, 0.766, and 0.784 on test-zh, test-en, and test-hard-zh, respectively. 

---
# Ego-HMB: Human Motion Bridging from Egocentric Images via A Motion Bridge Diffusion Model 

**Authors**: Gaoge Han, Yongkang Cheng, Shaoli Huang, Jing Zhang, Mingming Gong, Fakhri Karray, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29218)  

**Abstract**: We introduce Egocentric Human Motion Bridging, a task that generates a coherent 3D human motion sequence from two endpoint egocentric images. Unlike conventional motion in-betweening, which requires structured 3D endpoint poses, this setting uses raw first-person observations available from wearable devices in VR/AR and robotics. It presents challenges due to viewpoint variations, partial body visibility, and temporal ambiguity under cross-modal endpoint constraints. To address these challenges, we propose Ego-HMB, a unified diffusion-based framework that learns endpoint pose estimation and temporally consistent motion infilling within a single model through two training stages. First, Egocentric-to-Canonical Pretraining uses a hierarchical curriculum with progressively increasing temporal spans to strengthen endpoint pose priors. Second, Motion Bridge Diffusion Training incorporates endpoint and physics awareness, diffusing from neighboring poses to promote temporal continuity while preserving global endpoint constraints. Extensive experiments compare our approach with one-stage methods that directly generate motion from two egocentric images and two-stage pipelines that combine endpoint pose estimation with motion interpolation. Ego-HMB consistently outperforms these baselines in endpoint accuracy and motion generation quality. Code is available at this https URL. 

---
# Direct Feedback Policy Optimization for AI Coding Agents 

**Authors**: Fei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2611.29193)  

**Abstract**: AI coding agents generate, modify, and validate code through multi-turn interactions. Existing reinforcement learning methods often rely on sparse terminal rewards, making it difficult to assess how intermediate decisions affect task outcomes, and group-relative methods also require repeated sampling for each task. We propose Direct Feedback Policy Optimization (DFPO), a single-trajectory reinforcement learning algorithm that trains directly from a single trajectory. DFPO achieves higher task resolution rates than GSPO under matched base models and execution harnesses. 

---
# loopxDINO: Dense Visual Prediction via Iterative Latent Reasoning 

**Authors**: Xiangjing An, Zhibin Jiao, Aiting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.29180)  

**Abstract**: Standard dense visual prediction architectures scale representation capacity by stacking deep feedforward decoders (6-16 transformer layers with 50M-200M parameters). This explicit layer stacking brings substantial activation memory overhead during backpropagation, requires high off-chip memory bandwidth for non-tied weights, and uses static computational allocation that cannot adapt to spatial image complexity.
We propose loopxDINO, which replaces physical decoder depth with iterative latent reasoning. It has two decoupled components: (1) a frozen DINOv2-Base backbone extracting spatial features in one forward pass and caching static Key-Value representations in GPU memory; (2) a compact weight-tied Looped Cross-Attention Decoder (7.96M core, 14.85M total trainable parameters) that recurrently updates queries across T iterations. Recurrence is stabilized by step-conditioned adaLN, LayerNorm-based hyper-ellipsoid manifold regularization, and learnable residual contraction damping. Intermediate mask predictions are fed back as spatial attention biases to guide queries toward ambiguous contours with no extra parameters.
On Pascal VOC 2012 semantic segmentation with a single RTX 5060 Ti 16GB GPU, loopxDINO (T=2) achieves 85.66% mIoU and 65.01% Boundary IoU. It outperforms 6-layer DINO-Trans6L (54.06M params) by +1.09% mIoU and +7.41% Boundary IoU with 72.5% fewer parameters. Compared to 4-layer DINO-FCN4L probe (2.43M params, 83.82% mIoU), loopxDINO improves boundary delineation by +6.43% bIoU and occlusion resilience by +6.76% mIoU (60% patch masking). With Adaptive Computation Time (ACT), it reaches 87.3 FPS and retains 29.64% mIoU under 60% occlusion (+5.83% over DINO-Trans6L).
Source code, pre-trained weights and evaluation suites are open-sourced under Apache 2.0 at this https URL 

---
# On Adaptive Step Sizes for Eligibility Traces in Deep Reinforcement Learning 

**Authors**: Esraa Elelimy, Martha White  

**Link**: [PDF](https://arxiv.org/pdf/2611.29178)  

**Abstract**: Many deep reinforcement learning (RL) algorithms store agent-environment interactions in a replay buffer and later learn by sampling transitions from it. However, this approach can be memory-inefficient and can delay learning because the agent can learn from a transition only when the buffer samples it. Additionally, most of these methods use one-step or short multi-step temporal-difference (TD) updates, which are slow at credit assignment. An alternative approach is to learn from the stream of observations as they arrive without relying on a buffer. Eligibility-trace algorithms allow these incremental updates and can also assign credit over many steps. These algorithms were common with linear function approximation but have only recently been used in deep RL. Basic questions, such as how to use vector step sizes with eligibility traces in the deep RL setting, remain open. In this work, we first show that adaptive vector step sizes must be part of the eligibility trace to keep the equivalence between the forward and backward views. We then show that the step sizes of common optimizers, such as Adam, do not have an online backward view. We introduce a new optimizer, Function-Space Distance (FSD), that maintains the forward-backward equivalence. We test four eligibility-trace algorithms covering semi-gradient and gradient TD updates, as well as value-based and actor-critic methods. We show that FSD performs better than SGD or naively applied Adam. The resulting algorithms outperform or match other streaming methods and are comparable to or better than PPO, which uses a buffer, in most environments. Code available at this https URL 

---
# What Makes Recurrence Generalize Across Lengths? 

**Authors**: Yoonsung Jung, Sunghun Jung, Louise Nayeong Koh, Yongjun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2611.29171)  

**Abstract**: Multi-step reasoning requires models to preserve and update intermediate information as computation unfolds. State tracking captures this fundamental requirement, but models that succeed within a fixed range of state transitions may not generalize when substantially more updates are required. Recurrent Transformers offer a natural mechanism for addressing this challenge by repeatedly applying shared Transformer blocks, allowing computation to scale with problem length. We study when such recurrent computation generalizes beyond the training range in a controlled state-tracking task, varying operation representation, positional encoding, and recurrent capacity. We find that recurrence alone is insufficient: multi-token template representations and absolute positional encoding can severely limit extrapolation, whereas atomic operation representations enable substantially stronger length generalization. The effect of positional structure also varies with recurrent depth, with periodic positional encoding achieving the strongest mean performance at greater recurrent depth. Most importantly, scaling the recurrent operator extends the extrapolation horizon, reaching 35.1% exact-match accuracy on sequences of 20-40 operations after training only on 2-10, while substantially larger fixed-depth Transformers remain at 0%. Additional recurrent computation at test time can extend the range over which the model generalizes, but does not eliminate failure on much longer sequences. These results suggest that recurrent computation can support length-generalizable state tracking, but that its effectiveness depends critically on whether the learned computation remains reusable across sequence lengths. 

---
# Fixed or Deleted? External Feedback Lets Cheaper AI Coding Agents Repair Rule Violations 

**Authors**: Laura Wynter  

**Link**: [PDF](https://arxiv.org/pdf/2611.29162)  

**Abstract**: Many defects violate project rules that the code itself does not reveal, such as layering constraints or required tests. We study whether LLM coding agents repair such violations better when an external checker reports them, with their location and a repair goal. Three models, a small, medium and large model, driven by an AI agent, worked on 25 pull requests with 31 planted defects in ten open-source projects, plus 5 correct controls, five times alone and five times with findings from CodeAssure (CA), a commercial checker. We also compare with GitHub Copilot code review, and compare CodeAssure's detection with nine free static-analysis tools run out of the box, including Semgrep and SonarQube Community. Counting a run as repaired only when the repaired code also compiles and keeps its feature, CA findings raised the repair rate from 26% to 69% for the smallest model, from 46% to 90% for the medium model and from 45% to 93% for the largest model (pull-request-level p < 0.001), and from 39% to 84% over all runs. Every Java, TypeScript and Python repair compiled. We found that agents often cleared a defect by deleting a newly added file. Metrics based on whether the defect is gone count feature deletion as a success. The largest model alone did so in 44 of 125 runs. CA includes a check for deleted files that largely prevents this. In an ablation with the check turned off, removals rose from 9 to 65 out of 375 runs and the repair rate fell to 54% (small), 70% (medium) and 85% (large model). The CA workflow also outperformed Copilot code review (p < 0.001). Together, the free tools detected 8 of the 31 defects, and CA detected all 31. With CA, the medium model repaired 90% of runs, against 45% for the largest model alone, at about one fifth of the mean cost per run ($0.17 against $0.91). 

---
# ContextFact: Remembering Facts in Context for Long-Term Conversational Agents 

**Authors**: Jiayu Huang, Yichao Cai, Xinyu Li, Zhen Zhang, Javen Qinfeng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.29155)  

**Abstract**: Agents built on large language models must maintain continuity across repeated interactions with users. However, long-term conversations scatter facts across turns and sessions, making it difficult to retrieve the right evidence with enough context. Memory for such conversations must therefore preserve facts with their context while keeping evidence selectively accessible. We first study how memory construction and retrieval-unit boundaries affect answerability and evidence access. Reading the entire memory bank reveals gaps that cannot be resolved by retrieval selection alone, showing that some context must be preserved during memory construction. Adjacent-turn grouping improves evidence access at matched unique-turn budgets, while wider windows do not consistently help. This motivates keeping fact-specific context local while organizing cross-session evidence separately. We propose ContextFact, which links context-preserving fact memories with source-grounded entity representations. Within sessions, it gathers the context needed to interpret each fact. Across sessions, it aligns recurring referents and revisits the source dialogue to reconstruct coherent entity representations linked to the supporting facts. At query time, ContextFact retrieves fact and entity views, reranks them independently, and uses entity-linked expansion to surface related facts; complete entity representations then provide organized cross-session evidence. With Qwen3.5-27B as the backbone, ContextFact achieves state-of-the-art performance while using only $21\%$ of the evidence tokens consumed by the strongest evaluated baseline, obtaining $86.49\%$ overall judge accuracy on LoCoMo and $91.00\%$ on LongMemEval-S. Controlled ablations support context-preserving fact boundaries and source-grounded entity reconstruction, while entity readout improves multi-hop accuracy beyond adding more facts. 

---
# Medical Specialty Prediction from Bangla Patient Narratives: A Comparative Study of ML, DL, Transformers, and LLMs 

**Authors**: Tasnin Tabassum Ali, Faisal Ahmed, Muhammad Anwarul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2611.29151)  

**Abstract**: Timely access to appropriate medical specialists remains a bottleneck in Bangladesh's healthcare system, as patients describe symptoms in informal Bangla rather than clinical terms, leading to ineffective referrals. This paper compares four model families for predicting medical specialties from Bangla patient narratives: six machine learning (ML) models using hybrid TF-IDF features, four deep learning (DL) models using Word2Vec embeddings trained on the corpus, four fine-tuned transformers, and five large language models (LLMs) evaluated zero-shot. From a corpus of 3798 patient records covering 20 specialties, we retained the five with the most records (Medicine, Gynecology, Dermatology, Pediatrics, Gastroenterology) and split the resulting 2730 records 80:20 into training and test sets shared by all families. The best transformer was XLM-RoBERTa (86.26% accuracy, 86.21% macro-F1), the best ML model was SVM (84.98%, 84.92%), the best DL model was Bidirectional GRU (80.77%, 80.83%), and the best LLM was Gemini (90.29%, 90.15%). Holm-corrected paired McNemar tests show SVM significantly outperforming Bi-LSTM (adjusted p = 0.004), whereas the Gemini-versus-Grok gap does not survive correction (adjusted p = 0.104), and neither the strongest transformer versus the weakest LLM (Grok) nor SVM versus the strongest transformer reaches significance. A tuned classical model is thus statistically indistinguishable from the strongest transformer, so accuracy rankings alone should not be read as evidence of one family's superiority. The findings provide a comparative baseline for Bangla medical text classification. 

---
# Can an Agent Trust Its Own Zoom? Cross-Magnification Provenance Verification for Whole-Slide Images 

**Authors**: Yan Wang, Yiran Song, Zizhuo Gong, Jianxiang Qi, Zhongxing Xu, Lingfei Qian, Minquan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.29150)  

**Abstract**: Visual agents navigate whole-slide images through magnified observations associated with claimed locations. However, a visually plausible observation does not necessarily originate from its claimed location, particularly across magnifications and in regions with similar tissue structures. We formulate cross-magnification provenance verification (CMPV) to evaluate whether visual evidence supports a claimed spatial association, recover the source when possible, and abstain when evidence is insufficient. We introduce PyraFault, a benchmark of 3.06 million coordinate-supervised examples from 2,200 cases across five TCGA cohorts, covering within-context substitutions, absent-source faults, and geometric perturbations. We further propose ZoomTrace, a claim-independent verifier that estimates the observation's source before evaluating its agreement with the claimed location. It combines physical-scale alignment, visual retrieval, geometric refinement, and reliability estimation using spatially disjoint competing matches. On 503,320 test examples, ZoomTrace achieves 0.969 AUPRC and 0.987 AUROC, compared with 0.789 and 0.903 for Efficient LoFTR. Further evaluations reveal trade-offs in reliability estimation and limitations in vision-language models, cross-dataset retrieval, and calibration. Integrating ZoomTrace into visual agent tool paths reduces exposure to corrupted observations and silent failures, while downstream accuracy improvements depend on the agent and verification threshold. 

---
# VeriLogic: Reinforcement Learning over Verifiable Logical Reasoning Traces 

**Authors**: ZeLiang Chen, Yunyao Zhang, Xinglang Zhang, Wenbing Li, Junqing Yu, Zikai Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.29147)  

**Abstract**: Logical reasoning derives conclusions from premises through explicit intermediate inference steps, providing a natural basis for process-level verification and supervision. However, existing learning methods often rely on response-level outcome signals, which overlook heterogeneous correctness within the reasoning trace: a correct answer may contain invalid reasoning, while valid intermediate steps may also appear in an incorrect response. To address this issue, we introduce Reinforcement Learning on Verifiable Logical Reasoning (VeriLogic), which turns internal reasoning traces into explicit and verifiable learning objects. Rather than treating a response as an indivisible learning unit, VeriLogic decomposes logical reasoning into explicit intermediate judgments, proofs, and final decisions, and organizes them into a unified "Grounding - Optimization - Refinement" pipeline: (1) Verification-Grounded Reasoning Environment Construction grounds logical reasoning into explicit intermediate states and formally checkable proof obligations, making the internal reasoning process accessible to verification and learning; (2) Logical Structured Policy Optimization learns from verifier feedback through complementary global and local rewards, jointly optimizing the overall reasoning outcome while providing targeted optimization signals to different intermediate reasoning paths; and (3) Verifier-Guided Iterative Refinement acts on verifier feedback by leveraging proof states and error messages to iteratively diagnose and repair failed reasoning. Experiments on six logical reasoning benchmarks show consistent gains over strong baselines, with average improvements of 6.39% and 5.00% on Qwen3 and Llama-3.1. These results highlight the value of making logical reasoning traces verifiable, learnable, and refinable beyond final-answer supervision. 

---
# DeepDOSReasoner: Physics-Grounded Reasoning for Density-of-States Prediction and Materials Discovery 

**Authors**: Yingheng Wang, Francesco Ricci, Zhilong Wang, Tao Yu, Junwen Bai, Xiangyun Lei, Shufeng Kong, Fengqi You, John M. Gregoire, Carla P. Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2611.29132)  

**Abstract**: Predicting a material's behavior from its atomic structure is a longstanding goal of the physical sciences. The electronic and phonon densities of states (eDOS and phDOS) govern key electronic, optical, vibrational, and thermodynamic properties, yet their calculation using density functional theory (DFT) is costly: about 60% of Materials Project entries lack eDOS and over 80% lack phDOS. Existing crystal-to-spectrum models rely on pooled structural representations and pointwise regression, which blur spectral features and violate total-state conservation. Here we introduce DeepDOSReasoner, a physics-grounded framework that decodes spectra through atom-to-energy source attention and refines the normalized DOS through learned continuity-equation transport, enforcing nonnegativity and total-state conservation by construction. DeepDOSReasoner achieves the highest prediction accuracy across all evaluated metrics for both eDOS and phDOS, recovers element-resolved contributions without supervision, and most reliably identifies rare valence-band (VB) gaps. In a prospective screen of 8,106 uncharted materials validated by new DFT calculations, its predictions remain the most accurate among the methods evaluated, with 25 of its top 50 candidates confirmed as VB-gap materials. Three case studies spanning substituted compounds, photovoltaic chalcogenides, and a high-entropy alloy demonstrate transfer across materials classes. Ablations confirm that both stages are necessary, while the model retains comparable accuracy when trained on only 25% of the data. Predicting spectra in milliseconds enables exhaustive screening; we release predictions filling the missing eDOS for 91,405 and phDOS for 127,768 Materials Project entries. DeepDOSReasoner provides a blueprint for integrating machine learning with physical reasoning toward high-throughput spectral screening and autonomous materials discovery. 

---
# Group-Balanced Supervised Fine-Tuning for Knowledge-Intensive Domains 

**Authors**: Mark Heimann, Bruno Jacob  

**Link**: [PDF](https://arxiv.org/pdf/2611.29127)  

**Abstract**: Supervised fine-tuning (SFT) weights every supervised token equally, so on knowledge-dense corpora the loss budget follows token count rather than importance: on a legal QA benchmark, citation strings absorb 81% of the supervised loss weight budget while the answers being taught absorb 5%. We present Group-Balanced SFT (GB-SFT), which equalizes the loss budget across the semantic roles of tokens within a target sequence. With one tunable dial and one structural switch, GB-SFT contains standard SFT, class- and span-level balancing as corners whose geometry we characterize exactly. On benchmarks in law, medicine, code, and mathematics, we show across model families that: (i) GB- SFT injects new knowledge better than uniform SFT, pointwise token weighting, memory editing, and matched-budget data mixing; (ii) it matches the injection of masking out everything but the knowledge payload while retaining the citation recall that masking destroys (0.98 vs. 0.38); and (iii) role-aware grouping carries the gain: label-free partitions recover at most a third of it, an LLM tagger nearly all where the groups are lexically distinctive; and and (iv) the budget governs what fine-tuning breaks as well as what it learns, acting as a monotone lever on the safety-guardrail erosion SFT causes, at matched task accuracy. We release two new contamination-controlled knowledge-injection benchmarks with validated ground truth. 

---
# Decoding Transformer Dynamics via Koopman Subspaces 

**Authors**: Subhradeep Kayal, Ali Dashti  

**Link**: [PDF](https://arxiv.org/pdf/2611.29113)  

**Abstract**: We apply Koopman operator theory and Dynamic Mode Decomposition (DMD) to characterise how large language models (LLMs) process different input topics. Treating the layer-to-layer evolution of hidden states as a dynamical system, we fit a linear Koopman operator and take the subspace spanned by its dominant modes as a per-topic spectral fingerprint. Four findings emerge. First, the depth map is near-linear: a single linear operator predicts each next-layer state to within approx. 18% relative error (interior variance explained approx. 0.99). Second, over the 57 MMLU subjects the fingerprint is topic-discriminative, matching the MMLU taxonomy at 0.72 nearest-neighbour purity and each subject's semantic neighbourhood at 0.75 any-hit@3. Third, it is a reproducible signature (split-half self-retrieval approx. 0.97$). Finally, these findings are consistent across seven base models (Qwen2.5 0.5-7B, Llama-3 1-8B), with a suggestive trend of alignment rising with MMLU accuracy. The spectral fingerprint thus provides a probe of how transformers process different topics across depth. Code is available at this https URL. 

---
# Bridging Blind Spots With Words: Decentralized Multi-View Spatial Reasoning via Language-Constrained Communication 

**Authors**: Soumyabrata Chaudhuri, Joydeep Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2611.29112)  

**Abstract**: As multimodal LLMs (MLLMs) get increasingly used as collaborative partners across embodied robotics, AR/VR, and multi-user systems, spatial reasoning is now often distributed across agents with partial environment visibility, necessitating spatial reasoning integration. Prior work in language-mediated spatial integration is however strictly limited to simplified two-agent scenarios. To address this gap, we first formalize the general problem of decentralized multi-view spatial reasoning, where N agents observe distinct viewpoints, exchange budget-constrained natural language messages, and collaborate to enable a listener to answer spatial queries. We then introduce a novel multi-view spatial benchmark to isolate queries that demand non-trivial multi-view integration. Through an extensive empirical study, we discover that the penalty of restricting decentralized spatial reasoning to natural language with limited budget is far smaller than suggested by prior two-agent literature. Nevertheless, we observe that substantial headroom still remains for multi-agent spatial reasoning in this setting, as evidenced by a significant gap between current centralized performance and an oracle best-of-K ceiling. Moreover, we show that reinforcement learning with credit assignment substantially bolsters the decentralized model, effectively closing the gap with the centralized baseline. 

---
# Beyond Evidence Coverage: Encoding-Time Access in Multi-Agent Systems 

**Authors**: Joshua Huang, Ajit Mallavarapu, Elliott Kalt  

**Link**: [PDF](https://arxiv.org/pdf/2611.29107)  

**Abstract**: Cache-sharing multi-agent systems can retain every evidence token's representation while preventing cross-worker access during encoding. We test whether this coverage recovers joint-encoding quality using attention masks over two logical workers in frozen Qwen3-4B, Qwen3-8B, and Llama-3.1-8B-Instruct models. Tokens, positions, and answer-time access remain fixed within comparisons. We evaluate 200 two-hop MuSiQue-Ans questions and 200 selected HotpotQA bridge questions. In data-layout seed 0, averaged across models, blocking access costs 5.6 and 6.8 F1 points relative to full access on MuSiQue and HotpotQA, respectively. Exposing the earlier worker's supporting paragraph improves F1 over blocked access by 7.6 and 6.7 points; equal-token non-support access provides no detectable average gain. With model weights and greedy decoding fixed, we vary distractor layouts using five data-layout seeds per dataset for Qwen3-8B and three for Llama-3.1-8B-Instruct. Supporting-paragraph access has higher mean F1 than blocked and equal-token non-support access in every evaluated data-layout seed. Supporting access can also outperform full access, but its advantage depends on the model, dataset, and layout. Oracle support labels isolate access from selection error; the study does not evaluate a deployable selector. Complete evidence coverage at answer time therefore does not guarantee joint-encoding quality: which supporting facts can interact during encoding matters. 

---
# Spatial Ensemble Logic for Biomedical Imaging: Geometric Element Ensembles in Euclidean Space 

**Authors**: Guo-Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29105)  

**Abstract**: Ensemble Logic (EL) is a family of modal logics over metric monoids whose modal parameters are terms with quantified variables. We study its member over the Euclidian space, called Spatial Ensemble Logic (SEL), for biomedical images. Geometric element ensembles (GELs) represent atomic propositions by finite lists of effectively presented semi-algebraic sets, principally closed balls. Translation, dilation and erosion preserve individual balls up to emptiness in any dimension. Translation and dilation act element-wise, but element-wise erosion gives only an inner bound. For $k\geq 2$, finite unions of balls are closed under neither intersection nor erosion. We therefore separate record representation from the evaluation domain. Our main result is a satisfaction-preserving translation over GELs into the first-order theory of real closed fields. It yields computable semi-algebraic satisfaction sets, decidable model checking and finite real algebraic evidence. For a fixed formula and dimension, model checking has polynomial data complexity under stated rationality and bounded-degree assumptions. Combined complexity is PSPACE-hard and in EXPSPACE. On the line, rational interval data admit linear-arithmetic evaluation with rational evidence, under rationality assumptions on the norm, query and evaluation point. For $k\geq 2$, rational balls can force irrational witnesses. Lower and upper annotation bounds propagate through formulas. Synthesis of a ball ensemble is decidable within a prescribed budget of balls and undecidable without one. In a vascular use case, medial ball ensembles decide calibre and stenosis queries at stored centers by comparing stored numbers. We also examine how far evaluation can proceed element-wise when the union forming a support is deferred to the point of evaluation. 

---
# TesseraFlow: Scalable Video Diffusion for Dynamic Medical Imaging Inverse Problems 

**Authors**: Atakan Topcu, Mohammadhossein Momeni, Alfredo De Goyeneche, Michael Lustig  

**Link**: [PDF](https://arxiv.org/pdf/2611.29099)  

**Abstract**: Generative priors are state-of-the-art for modeling medical images in image reconstruction tasks, but reconstruction with video priors takes tens of seconds to minutes per frame. Latent priors inherit the errors of their autoencoder, the memory of pixel-space whole-frame priors grows with the frame size, and pixel-space patch priors are slow for two reasons. First, their deep networks process every frame at full resolution at every sampling step, even though early steps resolve only low frequencies. Second, their samplers spend many network evaluations per sampling step on overlapping windows, multiple re-noising rounds, or backpropagation through the prior. We introduce TesseraFlow, a complex-valued pixel-space flow prior that reduces the first cost. A global encoder reads a low-resolution view of every frame once per sampling step. A single shallow decoder operates on this view during early steps and on full-resolution space-time tiles during later steps, so no network processes a whole frame at full resolution. Our sampler, ReTile, addresses the second reason. We use randomized grid shifting at every sampling step along with re-noising, which acts as a stochastic gradient instead of computing overlapping tiles, so a single network pass per sampling step suffices. On its own, ReTile speeds up an existing global-context patch prior by 6-9x, and our prior adds a further 3-5x speedup. TesseraFlow reconstructs about one 256x256 frame per second, 20-367x faster than patch and latent video priors, and can match or exceed the best generative baseline on cardiac, brain, and abdominal MRI and lung CT, at a fraction of its GPU time. 

---
# JEPA-Tuner: From Predictive Fine-Tuning to Asynchronous Execution of Vision-Language-Action Models 

**Authors**: Jingxiang Qu, Tian Xie, Youngjin Hong, Changhyun Choi, Yang Katie Zhao, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29096)  

**Abstract**: Vision-Language-Action (VLA) models generate sequences of actions from visual observations, proprioceptive states, and language instructions. In embodied interaction, executing these actions influences subsequent observations and robot states, providing a natural predictive signal for policy learning. However, standard action-supervised fine-tuning does not explicitly capture these action-conditioned temporal dependencies in the guidance consumed by the action predictor. We introduce JEPA-Tuner, a predictive fine-tuning framework that regularizes the VLA's native policy guidance through action-conditioned future prediction. During fine-tuning, a predictor in JEPA-Tuner learns to match guidance encoded from a future observation given the current guidance and committed actions. This objective encourages a cross-modal guidance adapter to capture how the scene and robot state evolve under those actions, while jointly training the action expert on both observed and predicted guidance keeps the representation informative for control. In addition, when deploying the VLA model with asynchronous action-chunk inference, the pre-trained predictor will anticipate the guidance needed at the next chunk boundary. The policy can thus generate the next chunk in advance and execute it from its first action without overlap correction or additional predictor training, enabling efficient and high-quality asynchronous execution. Experiments on LIBERO and Meta-World show that JEPA-Tuner improves task success under synchronous inference with better multi-modal guidance embeddings. For asynchronous control, JEPA-Tuner improves performance and smoothness across different latency settings on both LIBERO simulator and multi-size real robot demonstrations. The project website with video demos is at this https URL. 

---
# Do LLM Agents Compromise Like Humans? Risk and Responsibility in Dyadic Decision-Making 

**Authors**: Noah S. Okada, Ketika Garg, Wenning Deng, Dean Mobbs  

**Link**: [PDF](https://arxiv.org/pdf/2611.29069)  

**Abstract**: Large language model (LLM)-based agents are increasingly deployed in settings where they must collaborate to make joint decisions. We investigate how this collaboration unfolds under risk, asking whether agents adjust their choices to accommodate their partners, how they assign responsibility for shared outcomes, and whether responsibility attribution shapes subsequent adjustment. We designed a series of agents with systematically varying risk preferences and paired them in a repeated joint decision-making task previously conducted with humans. Like humans, agents converged toward their partners' choices. However, similar aggregate behavior arose from different dynamics: agents showed little evidence of reciprocal adjustment, tended to weigh rewards more, and their responsibility attribution was decoupled from compromise. Similarly, while they showed bias in their responsibility attribution, the outcome's valence shaped the attribution differently than in humans. Our results reveal important differences between human--human and agent--agent joint decision-making, demonstrating that similar social outcomes can emerge from different social dynamics and reasoning. 

---
# RES: Accelerating Diffusion Language Models with Rolling Entropy Sampling 

**Authors**: Patrick Pynadath, Nicolas Perez-Nieves, Bobak Shahriari, Jean Tarbouriech, Ruqi Zhang, Ivan Lobov  

**Link**: [PDF](https://arxiv.org/pdf/2611.29066)  

**Abstract**: A diffusion language sampler makes two choices for every position in its denoising window: where the next token comes from, and when the position is committed to the KV cache. Using the entropy-bound sampler in DiffusionGemma, we study two extremes of the second choice. Committing each token as soon as it is accepted removes the chance to revise it, which hurts when the entropy bound is relaxed to trade accuracy for speed. When accepted tokens are frozen, macro accuracy falls from 73% to 20% as the bound is relaxed, while the unfrozen sampler stays near 70%. Committing a whole block at once wastes computation, since tokens converge in a largely left-to-right order and more than half of the window holds already-final tokens by twenty denoising steps. These findings motivate the Rolling Entropy Sampler (RES), which commits a contiguous accepted prefix once it reaches a minimum length and keeps positions farther ahead open to revision. Unlike earlier rolling samplers, RES runs on uniform diffusion, where accepted tokens can still change, and it needs no training. Across seven benchmarks, RES increases tokens per forward pass from 19.3 to 25.5, while macro accuracy goes from 80.7% to 78.8%, and it extends the speed-quality Pareto frontier beyond what tuning the entropy bound alone reaches. 

---
# Watermark Claims Under Test: Lessons from Auditing a Production LLM 

**Authors**: Felix Desyatirikov  

**Link**: [PDF](https://arxiv.org/pdf/2611.29018)  

**Abstract**: Can we tell whether an LLM is watermarked without knowing its secret key? Published black-box tests suggest that carefully designed prompts can reveal the statistical footprint of a watermark. This paper examines what happens when those tests move from controlled reference implementations to a deployed model whose watermarking parameters are undisclosed. We present an audit of the Red-Green watermark presence test on Claude. Across ten initial configurations, the published test never rejects on a model the provider states is watermarked, despite detecting a local reference at comparable sample sizes. Controlled experiments expose a failure mode: including the prompt in the watermark's repeated-context history suppresses marking exactly where the test measures it. The watermark remains active, but the probe becomes blind. Attempts to repair the probe reveal further confounds. Constructed contexts can trigger native digit-to-word preferences, while comparisons within a single response can measure how the model plans its answer. Tokenization changes which contexts the probe actually controls. These findings show why validation on a surrogate model does not establish validity on the model under investigation. A redesigned probe finds a substantial context-dependent shift on Claude Fable that is absent on two sibling controls. However, the pre-registered decision remains inconclusive because selective refusals violate the protocol, and the effect's watermark origin is unresolved. We show how to design controls, distinguish statistical rejection from evidence of watermarking and interpret a null without treating it as proof of absence. We provide researchers and practitioners working on LLM evaluation, watermarking, and provenance with an audit checklist grounded in concrete successes and failures. Code, prompts, and parsed responses are available for reproducing the analyses. 

---
# Population-Based Post-Training for Large Language Models 

**Authors**: Yixuan Even Xu, Aleksei Petrenko, Raja Giryes, Jackson Hamburger, Kevin Chen, Fei Fang, J. Zico Kolter, Philipp Krähenbühl  

**Link**: [PDF](https://arxiv.org/pdf/2611.29014)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is one of the most popular post-training paradigms for large language models. RLVR typically trains a single model under hyperparameters chosen before training begins, while researchers tune hyperparameters across consecutive runs. We introduce Population-Based Post-Training (PBPT), which trains several models in parallel and lets the training process do the tuning through population-based training. PBPT periodically evaluates models on a validation set. High-performers are duplicated and their hyperparameters perturbed to explore new configurations. Low-performers stop training and are replaced by these copies. On reasoning benchmarks (Enigmata, Nemotron) and interactive tool-use environments (Gaia2, AppWorld), PBPT produces stronger agents than compute-matched groups of independent RLVR learners. Furthermore, PBPT is able to recover from poor initial hyperparameters, automatically discovering better settings over the course of training. 

---
# Mind the Trap As LLMs Guess the Data Model: Grain-Aware Guards Against Fan-out in Text-to-SQL and Its Benchmarks 

**Authors**: Ahmed M. Mahran  

**Link**: [PDF](https://arxiv.org/pdf/2611.29009)  

**Abstract**: Text-to-SQL systems built on large language models (LLMs) often aggregate over one-to-many and many-to-many joins, a classic trap: when the joined rows repeat an entity, a SUM or COUNT counts it more than once, yet the query runs and returns a plausible number. We characterize these fan-out errors by one condition on the relation an aggregate reads, which depends on two facts a schema does not show: each measure's grain, and how many rows a join on each key repeats. An audit that tests the condition on the data finds distorted gold answers for 4.2% of the dev questions of BIRD, a widely used benchmark, and BIRD's later, independent revision changed 44% of the answers the audit flags as definite fan-out, against 16% of unflagged ones. We release BIRD-FanOut, 486 BIRD questions with reviewed gold, and GrainTraps, synthetic questions that set this trap and related ones. Across eight LLM agents, functional dependencies (FDs) computed from the data help no more than per-column statistics on GrainTraps, but grain-aware FDs, which also state each measure's grain and the repetition per key, cut fan-out errors on BIRD from 18.4% of runs to about 13%, also on held-out questions and without LLM judges. Accuracy rises less, because most remaining errors are not fan-out and BIRD's hints often prescribe the over-count. Text-to-SQL models should be given the grain of the data, and benchmark gold should be audited on the data. Code, data and benchmarks are available at this https URL 

---
# NoHairCache: Calibrate-Once KV Cache Eviction with One Power-Law Exponent 

**Authors**: Shuyang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29006)  

**Abstract**: The decay exponent $\alpha$ in KV-cache eviction is an intrinsic property of the model: calibrated once on a single corpus, it transfers directly across domains. Attention-based accumulation (H2O) and recency-based sliding windows (StreamingLLM) are two dominant eviction frameworks, often treated as unrelated, but we show that they are the two endpoints of a single power-law family: by scoring each token as $s_i = A_i \cdot (t-i)^{-\alpha}$, where $A_i$ is cumulative attention and $\alpha$ controls the age penalty, H2O and StreamingLLM correspond to $\alpha=0$ and $\alpha\to\infty$, respectively (with StreamingLLM additionally hard-coding an attention-sink patch).
The no-hair mechanism of \citet{xiang2026nohair} bounds the admissible range of $\alpha$, and the measured exponents agree: calibrating once on WikiText yields $[0.556, 0.822]$ for a representative model (Qwen2.5-1.5B-Instruct). The per-model ranges are consistent across five models (410M--1.5B) and independently match the optimal regions from perplexity sweeps.
\textsc{NoHairCache} achieves the lowest perplexity across three datasets (WikiText, PG19, and arXiv) without per-dataset tuning, outperforming both H2O and StreamingLLM ($p < 0.05$ on all three datasets, with $0.06$--$0.10$ PPL gains over StreamingLLM at budget $512$). On downstream four-way multiple-choice QA, it matches or outperforms StreamingLLM across all nine (task, budget) cells, retaining $98.2\%$ of full-cache accuracy on NarrativeQA at $25\%$ retention versus $91.1\%$ for StreamingLLM; per-cell McNemar tests are not significant, so QA is corroborating only. We release our calibration script and drop-in scorer upon publication. 

---
# Consumer-Time Validity in Stateful AI Runtimes: Verifying Recovered State and Deferred Authority 

**Authors**: Guijia Zhang, Huihao Jing, Bing Li, Shing-Chi Cheung, Harry Yang, Ziqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29004)  

**Abstract**: Stateful LLM serving engines and agent runtimes reuse execution state and deferred permissions across components and sessions. A cancellation, rollback, or approval is recorded by one component and relied on by another, and recording it does not make the later use valid: a downstream consumer may read state or dispatch an action whose identity, production status, allocation, or target revision has since changed. We call the missing check consumer-time validity and separate two forms. State validity governs what a consumer reads. On vLLM with LMCache, an interrupted KV load leaves unwritten blocks indexed, and a host-only counter rewind desynchronizes the GPU position, so a second tenant generates a schema-valid tool call naming the wrong tenant in 10/10 native-fault runs; the call is recorded and not executed. Authority freshness governs what a consumer executes: on a multi-session approval runtime, an unrevoked deferred grant overwrote a newer revision in one admitted live execution. Output equality cannot test state validity: a fault-free run diverges at token 30 through BF16 batching alone while attended-KV digests match. We therefore specify consumer-bound recovery contracts that, for state recovery, jointly require exclusion, retention, and task completion, and an admissible-witness standard realized as a GPU-free offline trace verifier over scheduler metadata and persisted tool artifacts. On recorded traces the verifier agrees with the reference verdicts in 42/42 frozen GPU-matrix scenarios. An allocation-aware repair satisfies 7/7 triggered vLLM fault predicates on a pinned scheduler-case matrix, where upstream PR 61769 satisfies 2/7, and revision-bound dispatch blocks all 160 deterministic conflicting overwrites in a separate dispatcher cohort. Recovery must be checked where state and authority are consumed, with evidence bound to the consuming request. 

---
# DoC: Learning to Defer or Commit for Timely, Accurate, and Stable Conversational Diagnosis Prediction 

**Authors**: Xueren Ge, Anthony Cortez, Homa Alemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2611.29003)  

**Abstract**: Conversational diagnosis prediction requires accurate, early, and stable predictions from ongoing clinical dialogue, yet dialogue-level labels do not specify when to commit. We propose a two-phase framework for learning deferral and diagnosis decisions. First, iterative commitment-boundary estimation constructs Defer/diagnosis supervision without annotated commitment times. Second, Defer-aware supervised fine-tuning initializes a policy, refined through reinforcement learning with trajectory-level rewards for timeliness, first-commit accuracy, and post-commit stability. On EMSDialog and MediTOD, DoC commits early, achieving the highest first-commit accuracy among post-trained methods (76.7%/51.2%) and the lowest post-commit error overall (16.4%/48.3%). 

---
# Demystifying Classifier-Free Guidance for Auto-Regressive Image Generation 

**Authors**: Zhiling Zhou, Jiachun Pan, Fengzhuo Zhang, Dirk Bergemann, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29002)  

**Abstract**: Classifier-free guidance (CFG) has been widely adopted in autoregressive (AR) models for high-quality image generation. Despite its strong empirical performance, its mechanism in AR models remains unclear. This paper demystifies the mechanism of CFG through both empirical and theoretical studies. By examining the top-ranked semantics of different components in CFG, we show that texture information is primarily encoded in the difference between the conditional and unconditional logits, whereas both logits share previous-token repetition as their leading semantics. This strong repetition semantics obscures the desired texture information, causing greedy decoding from the conditional logits alone to produce nearly pure-color images. Through a training-dynamics analysis of shallow transformers, we prove that this shared repetition semantics does not arise from limitations or failures of pretraining, but instead originates from the texture sparsity of images. We further show that CFG improves generation quality by rectifying this repetition bias during inference. Motivated by the shared semantics between conditional and unconditional logits, we propose Attention Weight Reuse (AttnReuse), which reuses intermediate attention computations from the conditional-logit forward pass to accelerate the unconditional-logit computation. AttnReuse reduces about $25\%$ of attention computation with little performance degradation across different models. 

---
# CLASP: Complementary Layer Selection and Profiling for Zero-Shot OOD Detection using Vision-Language Models 

**Authors**: Saahil Islam, Adarsh Bhandary Panambur, Teodora Spasojevic, Eduardo Castaneda, Andreas Maier  

**Link**: [PDF](https://arxiv.org/pdf/2611.28986)  

**Abstract**: Vision-language models (VLMs) enable zero-shot out-of-distribution (OOD) detection in medical imaging, yet existing approaches often rely on final-layer confidence or entropy-based layer selection, potentially missing complementary information. We show that low in-distribution (ID) confidence entropy does not reliably predict OOD discrimination. We introduce CLASP (Complementary Layer Selection and Profiling), an ID-only framework that jointly models layer--prompt responses and visual features. CLASP ranks visual encoder layers using Gaussian conditional mutual information and selects a compact subset through anchored spectral selection, without fixing the number of layers. It then combines the selected layer--prompt profiles with PCA-compressed visual embeddings and fits a single shrinkage-regularized Gaussian. OOD scores are computed using squared Mahalanobis distance, without fine-tuning, OOD training data, or test-time adaptation. Across histopathology, chest X-ray, and brain MRI benchmarks with UniMedCLIP and BioMedCLIP, CLASP performs strongly under varied distribution shifts. On MIDOG near-OOD, it achieves 87.54% AUROC with UniMedCLIP, outperforming confidence-based baselines; with BioMedCLIP, it remains competitive but trails visual-only Mahalanobis modeling. Ablations highlight the value of layer--prompt interactions and cross-feature dependencies, while exposing shift-dependent contributions and limitations under semantic shifts. 

---
# World Models for Ocean Engineering: A Critical Review of Maritime Autonomy, Adaptive Latent Prediction, Digital Twins, and Biodiversity Monitoring 

**Authors**: Basma Dokkara, Esteban W. G. Cluab, Luiz M. G. Goncalves, Paolo Spagnolo, Cosimo Distante  

**Link**: [PDF](https://arxiv.org/pdf/2611.28978)  

**Abstract**: Marine robots face partial observability, non-stationary dynamics, and sparse communication, motivating WMs that learn compact predictive states for planning. This review develops a functional taxonomy of WMs for ocean engineering, focusing on Joint-Embedding Predictive Architectures (JEPAs) and AdaJEPA's test-time adaptation within model predictive control, mapping them to navigation, adaptive sampling, digital twins, and biodiversity monitoring, and separating demonstrated evidence from prospective applications. The synthesis identifies key Gaps: benchmarks, uncertainty calibration, safe adaptation and proposes a research agenda for physics-informed, deployment-adaptive marine WMs validated through simulation and field missions. 

---
# A Design Space for Data Agents: Insights from KDD Cup 2026 

**Authors**: Boyan Li, Zhuowen Liang, Yupeng Xie, Xiaotian Lin, Xinyu Liu, Yizhang Zhu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.28970)  

**Abstract**: Data agents interpret analytical requests, gather evidence from heterogeneous sources, and compute results within finite resource budgets. We analyze the final code submissions of the ten highest-ranked teams in the KDD Cup 2026 Data Agents for Complex Data Analysis competition. We organize the observed design choices into three interacting layers, covering workflow orchestration, analytical functions, and working state. The comparison reveals recurring patterns for allocating computation in response to task progress, structuring evidence for computation and assessment, and preserving reusable artifacts with their provenance. Candidates that rely on the same evidence can inherit the same error, so interpreting their agreement requires tracing those dependencies. The design space motivates controlled evaluations of how resource allocation, evidence sharing, and targeted recovery affect task completion and correctness. 

---
# Who Gets Re-ranked When a Clinical Model Gains New Data? 

**Authors**: Priyanka Siddappa, Muntaqim Ahmed Raju, Shifat Haider Al Amin, Ruizhe Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.28968)  

**Abstract**: Hospitals often update a deployed risk model by adding a data source, such as laboratory results, that only some patients have. Work on safe model updates assumes fixed inputs and so ignores patients who lack the new data. The update has no new measurements for them, yet their rank can change, and a sick patient not yet tested can drop off an alert list. We split rank-based compatibility by data availability and pair it with error correction and alert-list effects. We compared each update with ordinary retraining on the old inputs, predicting ICU admission or death and hospital admission, on 78,475 emergency department visits from MC-MED and, in a pre-registered replication, 100,000 stays from MIMIC-IV-ED at a second hospital. Radiograph report text gave no detectable gain and almost no extra re-ranking. Laboratory results improved prediction but, at both hospitals and with linear or tree models, pushed patients without labs down alert lists; linear updates also reshuffled them more than retraining did, without ranking them better. A top-20% admission list caught 4.2% of admitted patients without labs after the update, against 18.8% after retraining (second hospital: 4.9% against 22.6%), while catching more patients overall. This held for list sizes from 5% to 50% and in every patient group we checked, despite better measured calibration. Separate alert thresholds for patients with and without labs restored most or all of their catch rate at both hospitals at almost no cost. Teams should check these patients before deploying such an update. 

---
# MolReJev: Decision Models Are In-Context Molecular Predictors 

**Authors**: Kaiyuan Lu, Jiatong Li, Yatao Bian  

**Link**: [PDF](https://arxiv.org/pdf/2611.28965)  

**Abstract**: Molecular prediction helps prioritize compounds for screening. Specialized predictors are trained for a fixed set of tasks, limiting cross-task generalization. Language models can condition on task descriptions and have recently shown promise in molecular prediction. However, generating answers token by token adds inference cost, and general-purpose models often lack task-specific molecular knowledge. We thus propose MolReJev, which combines Jev-style candidate scoring with in-context learning (ICL) on a language model. MolReJev retrieves similar labeled examples from existing task data and scores candidate labels jointly in one forward pass. ICL improves predictions from the released general-purpose Kev checkpoint and the separately trained MolReJev. For the trained model, gains span all evaluation groups, including held-out tasks, and extend to compounds absent from training. MolReJev generally outperforms MolecularGPT on matched evaluations and nearest-neighbor prediction with the same examples. With four examples, MolReJev and a trained text baseline achieve similar ROC-AUC, with higher measured throughput for MolReJev. These results show that decision models can serve as in-context molecular predictors for compounds, drug pairs, and reactions. 

---
# Learning Land-Surface Hydrology World Model from Satellite Observations 

**Authors**: Suya Li, Zekun Qi, Xiaochong Xu, Duo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28963)  

**Abstract**: A world model should tell us not only what will happen, but what would happen under a different action. Learned Earth models now forecast remarkably well, yet forecast skill says little about what their state means or how it responds to a changed driver. We build LandWM, a world model of land-surface water in which rain is the action, learned from seven years of SMAP and CYGNSS observations: physical observation operators tie its latent state to soil water, and a mass-conserving balance governs how that state responds to rain. Over the U.S. Southern Great Plains, the model, never shown a soil-moisture value, tracks 28 in-situ probes as well as the operational SMAP Level-4 product, even from CYGNSS alone, and a storm added to the forcing of a station's cell wets its soil as natural storms wet the probes. Replace either piece and the forecasts stay, but the meaning goes: with neural operators, only a probe fitted to the stations finds the soil water; with a generic recurrent transition, rain on a single cell barely moves it. Yet when the same rain covers the whole region the two transitions swap roles, and the conserving one responds too weakly. Grounding makes a world model readable; its response to actions must be tested under more than one kind of intervention. 

---
# Micro Operational Robustness of LLMs on Code Generation 

**Authors**: Debalina Ghosh Paul, Hong Zhu, Ian Bayley  

**Link**: [PDF](https://arxiv.org/pdf/2611.28957)  

**Abstract**: It is now common practice in software development for large language models (LLMs) to be used to generate program code. It is desirable to evaluate the robustness of LLMs for this usage. This paper is concerned in particular with how sensitive LLMs are to variations in descriptions of the coding tasks. However, existing techniques for evaluating this robustness are unsuitable for code generation because the input data space of natural language (NL) descriptions is discrete and sparse. To address this problem, we propose a robustness evaluation method called scenario domain analysis, which aims to find the expected minimal change in the NL descriptions of coding tasks that would cause the LLMs to produce incorrect outputs. We have formally proved the theoretical properties of the method and also conducted extensive experiments to evaluate the robustness of four state-of-the-art LLMs: Gemini-pro, Codex, Llamma2 and Falcon 7B, and have found that we are able to rank these with confidence from best to worst. Moreover, we have also studied how robustness varies in different scenarios, including the variations with the topic of the coding task and with the complexity of its sample solution, and found that robustness is lower for more complex tasks and also lower for more advanced topics, such as multi-threading and data structures. 

---
# When Rubrics Do Not Decide: Silent Value Completion in LLM Evaluation 

**Authors**: Zhuangzhuang Pan, Yan Xia, Kaiwen Lei, Chee Seng Chan  

**Link**: [PDF](https://arxiv.org/pdf/2611.28940)  

**Abstract**: Multi-criterion LLM judges often evaluate responses without specifying how conflicting criteria resolve, leaving the winner undetermined even under exact criterion scores. Issuing a strict verdict without declaring the required trade-off premise constitutes silent value completion. We introduce TradeoffBench to isolate this failure by separating rule execution from response scoring. On authoritative criterion vectors, six open-weight non-reasoning judges pass all dominance and equality controls yet pick unsupported winners on 9%-81% of 1,345 conflicts. All 577 order-stable choices follow an explicitly banned majority rule. Reasoning-enabled configurations drop anonymous-vector violations to 2 in 9,720 judgments under an explicit rule, yet cannot bridge omitted trade-offs or divergent component scoring. On raw TradeoffBench text with unspecified trade-offs, GPT-6.1-sol omits its completion premise in 29.8% of 400 verified conflicts. Even under an explicit rule on four-rater-consensus MultiPref conflicts, four reasoning judges still issue strict verdicts on 16/45 (GPT-6.1-sol) up to 35/46 and 25/29 (open-weight judges) of valid presentations. We propose identified-first evaluation, decoupling specification support, explicit completion, and target agreement. In MultiPref human-score audits, peer-mean substitution raises directional disagreement from 0.6% to 28.8% on 2,707 shared targets. A single-peer ablation confirms a 29.38-point gap on 2,175 targets, ruling out averaging as the cause. In HUMAINE, declared partial trade-offs add 2.5 percentage points of coverage, yet newly decided pairs incur 3.9% disagreement versus 0.9% for Pareto. Reliable LLM judging, therefore, requires auditing, specification support, declared trade-offs, and human target agreement separately. 

---
# Definition-Guided In-Context Learning for Zero-Shot Spoken Language Understanding 

**Authors**: Neeraj Agrawala, Sriram Ganapathy  

**Link**: [PDF](https://arxiv.org/pdf/2611.28929)  

**Abstract**: Adapting spoken language understanding (SLU) systems to novel or revised label taxonomies conventionally requires additional labeled data and retraining, creating a bottleneck for real world deployment. While in-context learning with few-shot demonstrations offers an alternative, multi-modal generalization and scaling has been challenging. In this work, we investigate whether natural-language definitions alone can guide zero-shot SLU without demonstration exemplars, finding that symbolic definitions serve as the primary supervisory driver for adapting speech-text models. We operationalize this principle as Definition-Guided In-Context Learning (DG-ICL), a fine-tuning framework that decouples semantic categories from fixed lexical names. By replacing labels with dynamically randomized definitions (arbitrary synthetic tokens in DG-ICL-Sym or permuted vocabulary words in DG-ICL-Flip), DG-ICL avoids lexical shortcuts and encourages the model to ground acoustic representations in contextual definitions. Novel categories can thus be introduced at inference simply by defining them through the instructions. Across three speech-text models (AUDIO FLAMINGO 3, QWEN2-AUDIO, and SALMONN), DG-ICL outperforms standard instruction tuning on held-out tasks under novel symbols (0.663 vs. 0.449 macro-F1 on Audio Flamingo 3; 0.703 vs. 0.620 on Qwen2-Audio; 0.582 vs. 0.399 on SALMONN), while remaining competitive on original labels. Furthermore, counterfactual evaluations demonstrate instruction following abilities, where it is shown that DG-ICL-Sym retains 68% of its performance when definitions contradict pre-trained lexical priors, whereas standard instruction tuning elicits significantly reduced retention of 21%. Finally, we show that DG-ICL enables zero-shot speech clustering into user-defined categories specified on the fly 

---
# Can Open LLMs Clear Sanctions-Screening Alerts? A Pre-registered Test on Public Sanctions Lists with Turkish Name Variants 

**Authors**: Mehmet Murat Huyut  

**Link**: [PDF](https://arxiv.org/pdf/2611.28914)  

**Abstract**: Sanctions screening raises name-based alerts that are mostly false positives, and level-1 analysts clear them by checking the customer's date of birth and name against a written policy. We test whether open-weight language models that a bank can run on its own premises can perform this level-1 disposition. From the UK and EU sanctions lists (7,103 natural persons) we built 12,200 synthetic customers: listed persons under six name conditions, including Turkish spellings of Arabic-origin names and Turkish romanisations of Cyrillic names, and two date-of-birth conditions, plus namesakes, near-name customers and a general population. A deterministic fuzzy and embedding screen raised 7,383 alerts, which three open models served on domestic infrastructure and three rule baselines dispositioned under the same policy. The models sent far fewer innocent customers to level 2 than a rule that applies the policy's date clause: 0.6 to 16.9 per 1,000 screened customers against 24.4, and they stated the date-of-birth status correctly in 98.3% to 98.5% of alerts. They also failed to escalate 1.5% to 2.2% of the listed persons. The pre-registered non-inferiority test against the rule was met, but only because of a construction error we found after the analysis: 21 customers in the "minor discrepancy" condition had dates that conflict under the policy. Without them the rule misses no listed person and no model meets the margin. The misses had model-specific causes: one model read minor date discrepancies as conflicts, one declared word-order and typing variants to be different names, and one discounted alerts that its own rationale said to escalate. Fuzzy matching found 69% of the Turkish spellings of Arabic-origin names; adding embeddings raised this to 74%. Open LLMs can cut level-1 workload, but on this evidence only behind a deterministic date check and with every name doubt escalated. 

---
# Topological Neural Collapse 

**Authors**: Nicholas Enbar-Salo, Jeova Farias Sales Rocha Neto  

**Link**: [PDF](https://arxiv.org/pdf/2611.28894)  

**Abstract**: Neural collapse (NC) predicts that, late in training, last-layer class features concentrate at their means and the means form a simplex equiangular tight frame (ETF). We ask whether NC also leaves a signature in the functional connectome of each layer: the graph of Pearson correlations between hidden-unit activations, built per class and summarized by persistent graph homology. Using deep MLPs, we observe two topological phenomena that co-occur with the well-studied geometric NC in the final layers during the terminal phase. Contrary to NC1, the split-half variability of a class connectome increases (TNC1). The mean pairwise $H_0$ distance $\bar{D}$ between class connectomes also rises (TNC2), but unevenly across class pairs, unlike the equidistance of NC2. We find that 24-56% of last-layer units become dead for some classes yet stay active for others, which inflates $\bar{D}$. After correcting for these units, both signatures remain elevated. These results from MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 provide preliminary evidence that a topological mechanism may accompany the geometric signatures of NC. 

---
# NAT: Scaling Causal Neural Action Tokenizers Across Embodiments 

**Authors**: Ahad Jawaid, Alexander Swerdlow, Yu Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28873)  

**Abstract**: Action tokenizers let robot policies treat control as autoregressive next-token prediction. As these policies are trained on larger and more diverse datasets, they will need a universal tokenizer that can encode actions from many embodiments and data sources, including ones it has not seen. We study action tokenization at scale, training on 49k hours of action data from real-world robots, handheld grippers, simulation, and humans. We introduce Neural Action Tokenizer (NAT), a causal tokenizer whose decoder lets actions be executed as their tokens are generated. Across four simulation benchmarks, NAT outperforms FAST and prior neural action tokenizers when tokenizing unseen robot embodiments, and in vision-language-action models it outperforms FAST+ across four benchmarks and on a real robot when acting through its tokens. NAT tokens also close the performance gap to a flow-matching action head on some benchmarks while lowering the inference latency for first actions. Our experiments also find that robot data may not be necessary to learn an effective action space: a NAT trained only on human hand motion performs close to tokenizers trained on robot data. 

---
# KineState-SWAM: A Surgical World Action Model with Explicit Kinematics and Causal Interaction Dynamics 

**Authors**: Rulin Zhou, Yujie Ma, Rui Tang, Qiujie Song, Wanhao Liu, Guoheng Ma, Liangjing Shao, An Wang, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.28871)  

**Abstract**: Existing surgical world models often leave implicit how instrument motion drives localized tissue responses and how interaction history informs subsequent actions. We introduce KineState-SWAM, a surgical world-action model coupling action prediction and video generation through a persistent interaction state, separating kinematically determined instrument motion from learned scene evolution. A flow-matching policy predicts joint trajectories conditioned on interaction state, robot configuration, and task instruction. Kinematic Geometry Conditioning converts these trajectories into surface-motion tokens and spatially aligned instrument renderings using calibrated URDF kinematics. Causal Interaction Dynamics recurrently updates the state through geometry-guided attention and motion-aligned feature transport, carrying interaction history into subsequent action predictions without intermediate video synthesis. The State-Geometry Adapter combines state-dependent spatial modulation with localized geometric residuals to guide video generation. Interaction-aware supervision emphasizes tissue responses near instrument working regions. We introduce dVRK-Chole5.2k, comprising 5,255 videos spanning 17 surgical tasks. Experiments demonstrate improved action accuracy and video fidelity over evaluated baselines within matched action-conditioning settings. Ablations support preserving and transporting interaction states for long-horizon prediction. 

---
# Operational Assurance for LLM-Based Systems under Governance Opacity 

**Authors**: Pedro Pinacho-Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2611.28834)  

**Abstract**: In the contemporary artificial intelligence ecosystem, foundation models are increasingly deployed under AI-as-a-Service (AIaaS) schemes characterized by high technical opacity, technological concentration, and limited external auditing capability over governance, training, and internal operational processes. Under these conditions, traditional ethical oversight approaches that focus exclusively on corporate governance or on internal model inspection become insufficient to ensure effective control over AI systems deployed in real-world contexts.
This work proposes a pragmatic operational assurance framework inspired by Zero-Trust principles from cybersecurity, which shifts effective auditing efforts toward the application layer, where risk acquires observable, contextualized, and operationally verifiable properties. The proposal articulates two complementary evaluation levels: a black-box evaluation aimed at characterizing general behaviors, vulnerabilities, and risky capabilities of foundation models through conversational scanning, and a grey-box evaluation focused on the operational contextualization of risk within the actual deployment environment of the audited application.
The framework incorporates mechanisms to move from general, CVSS-inspired severity metrics toward contextualized operational risk assessments that take into account user profiles, industrial domains, and organizational knowledge. In addition, the proposal strengthens practical auditability through digitally signed explainability reports (signed XAI reports), designed to support verifiable traceability, continuous monitoring, and operational accountability in LLM-based sociotechnical systems. 

---
# Hidden Sensitivity in LLM Pre-training: A Dynamic System View 

**Authors**: Bo Liu, Yuan Xie, Junan Chen, Fujun Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.28830)  

**Abstract**: In Large Language Model (LLM) pre-training, loss curves can conceal substantial instability, i.e., even small perturbations may lead to dramatically different predictions while leaving loss nearly unchanged. To investigate this phenomenon and reveal the underlying mechanisms, we first formulate LLM pre-training as a non-autonomous dynamical system that jointly evolves model weights and Adam optimizer states. Based on this formulation, we introduce a normalized sensitivity score in optimizer-relative coordinates to identify sensitive training checkpoints. At one such checkpoint, a weight perturbation of only 10^(-5) leads to 35.7% disagreement in top-1 predictions and 16.2% disagreement in benchmark answers after 200 training steps, with only 0.8% of greedy generations remaining identical despite similar validation losses. Furthermore, we find that perturbations to Adam's first and second moments induce comparably large effects, with similar sensitivity patterns observed in OLMo-1B. Finally, guided by our diagnostic, targeted stabilization reduces top-1 disagreement to 0.83% and increases generation exact match to 88.3%, at the cost of a 0.11-nat increase in validation loss. Our findings reveal a hidden form of optimizer-state-dependent sensitivity in LLM pre-training that conventional loss-based evaluations may overlook, while demonstrating its potential for diagnosis and mitigation. 

---
# When Being Wrong Is Right: Reverse-Capability Audio CAPTCHAs through Psychoacoustic Design 

**Authors**: Mingjie Bao, Yu Dai, Guangke Chen, Guozhu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2611.28823)  

**Abstract**: Large audio-language models (LALMs) have recently been shown to bypass audio CAPTCHAs designed around limitations of automatic speech recognition. To address this threat, we introduce reverse-capability verification, a general CAPTCHA framework that reverses the conventional human-easy, machine-hard relationship with respect to the physical ground truth. It selects challenges in which stable human perceptual limitations cause the human-reference answer to differ from the physical ground truth, while machine analysis tends toward that truth. Because the verifier accepts the human-reference answer, advanced solver capabilities and resulting improved recovery of the physical ground truth may instead strengthen the human-machine separation.
We instantiate the framework through psychoacoustics and design two audio CAPTCHAs: HIPI, a classification task, and YINGCHEN, a counting task, together with their conjunctive composition. Across seven LALMs and eight attack strategies, the maximum observed machine acceptance rates are 19%, 17%, and 3% for HIPI, YINGCHEN, and their composition, respectively, compared with 45-100% for nine prior schemes. A multidimensional human usability study yields corresponding human acceptance rates of 78%, 90%, and 76%, while maintaining competitive efficiency and satisfaction. Further analysis demonstrates that advanced LALMs tend toward physical ground truth. These results show that reverse-capability verification substantially reduces machine acceptance while retaining high human acceptance through an approach different from conventional CAPTCHAs. 

---
# PointNeXt Adapter Improved Model for BIM Component Defect Association under Construction Images and Design Model Attributes 

**Authors**: Na Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28820)  

**Abstract**: Construction site defect analysis increasingly requires linking field evidence to the correct BIM object, because a crack, spalling area, missing sleeve or misaligned support is useful for quality control only when it is connected with component identity, location, responsibility and design intent. This paper proposes a PointNeXt Adapter improved model for BIM component defect association under construction images and design model attributes. The method aligns site image crops, local point cloud patches, BIM component identifiers, geometry attributes, material categories, installation stage, inspection distance, viewpoint quality and source reliability tags into component level inspection windows. PointNeXt learns local geometry and deviation from as built point evidence, while the adapter injects BIM attributes so the same geometric irregularity can be interpreted differently for structural, envelope, MEP and embedded components. The fusion head outputs component link, defect class, defect probability, confidence, dominant evidence and recommended review action. A synthetic multisource BIM inspection dataset is used to evaluate repeated layouts, occlusion, sparse point clouds, registration errors and unseen component arrangements. Results show that attribute aware point learning improves association accuracy, defect recall and confidence calibration compared with image only, point only, simple fusion and adapter free baselines. The framework is positioned as traceable BIM issue review support, requiring manual confirmation and staged field validation before operational deployment. 

---
# Not All Directions Are Flat: Decomposed-Gradient Flatness for Sparse-View 3D Gaussian Splatting 

**Authors**: Ungbae Yoon, Kwanyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.28802)  

**Abstract**: Many applications of novel view synthesis, such as augmented reality and robotics, can afford only a few input views, since dense capture is costly. With such sparse views, however, 3D Gaussian Splatting (3DGS) overfits the training views and degrades at unseen viewpoints. Among the approaches to this problem, flat-minima optimization treats the Gaussians as trainable parameters and steers training toward flat minima, which generalize better, by perturbing the Gaussians with random noise or along the training gradient. However, these perturbations ignore their direction relative to the other training views. Interestingly, we find that the direction of the perturbation determines whether it improves generalization. Specifically, we decompose the gradient of each training view into a component parallel to the average gradient of all views, on which the views agree, and an orthogonal component, on which they disagree. Perturbing along the orthogonal component improves generalization, whereas perturbing along the parallel component does not. We further find that the flatness correction often opposes the reconstruction gradient, shrinking or even reversing the reconstruction step. Based on these findings, we propose DecoFlat, which perturbs each Gaussian along the orthogonal component together with random noise. Furthermore, we introduce a conflict-aware correction, which removes the part of the flatness correction that opposes the reconstruction gradient, and a rescheduling of the spherical-harmonics color, which suppresses the overfitting of view-dependent appearance. Extensive experiments demonstrate that DecoFlat, as a plug-in optimizer, consistently improves vanilla 3DGS and a wide range of sparse-view 3DGS methods. 

---
# HighwayVerse: A Chain-of-Inspection Benchmark for Highway Infrastructure Anomaly Understanding 

**Authors**: Junyi Yang, Zheng Si, Yilin Zhai, Pengpeng Ouyang, Hongxi Wang, Kangze Li, Mo Chen, Sicong Zhang, Hua Wang, Dongyuan Wu, Dong Chen, Mingliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28800)  

**Abstract**: Practical highway infrastructure inspection entails progressive diagnostic reasoning from visual perception to maintenance decisions, yet existing benchmarks remain confined to isolated tasks, leaving Multimodal Large Language Models (MLLMs) insufficiently evaluated across sequential inspection stages. Meanwhile, current datasets focus primarily on highway pavement defects, falling short of real-world inspection across diverse infrastructure components such as highway pavements, guardrails, glare screens, and drainage systems. To address these gaps, we formulate the Highway Infrastructure Inspection Progression Chain (HIIP-Chain), spanning Anomaly Detection, Anomaly Recognition, Condition Analysis, and Maintenance Resolution, and introduce HighwayVerse. HighwayVerse comprises 14,143 real-world Ground and Unmanned Aerial Vehicle (UAV) inspection records spanning 100 and 17 anomaly categories, respectively, with 84,858 expert-reviewed question-answer pairs. We evaluate 12 open-source and closed-source MLLMs using the chained metric Final, which credits a correct maintenance decision only when all preceding diagnostic stages are completed correctly. Results show that although models achieve relatively high accuracy when initially judging whether an anomaly is present, their markedly lower Final scores expose struggles across downstream recognition, analysis, and decision-making. This discrepancy indicates that current MLLMs do not fully comprehend specific highway anomaly patterns beyond superficial initial judgments. Furthermore, we propose LaneRAG, a training-free retrieval-augmented framework that provides hierarchical reference guidance achieving consistent performance improvements across diverse MLLMs. Our dataset, evaluation benchmark, and source code will be publicly released. 

---
# Pretraining Representation Lottery Determines Fine-tuning Generalization 

**Authors**: Fuming Yang, Core Francisco Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.28777)  

**Abstract**: Large scale pretraining (PT) followed by fine-tuning (FT) is the central paradigm of modern deep learning, yet how PT shapes FT generalization remains poorly understood. We use a synthetic setup in which the ground-truth concept geometry of entities is known, as a controlled testbed to study how representational geometry affects model behavior and how PT representations determine FT generalization. We first show that models differing only in training seed can closely match in PT loss yet vary substantially in FT generalization. We hypothesize that a representational lottery underlies this phenomenon: the random seed dependent geometry formed during PT determines what FT can reach. We confirm this causally. First, injecting the ground-truth concept geometry into the model yields near-ceiling FT generalization, showing that good representations directly drive generalization. Then, distilling a lucky seed's learned representation into an unlucky seed unlocks FT generalization, while distilling the same model's behavior does not. Furthermore, representation distillation can merge entities that never interacted during PT, which FT alone cannot. Finally, we show that regularizing representational drift during FT enables naturalistic learning: only models with good representations learn the task, and they generalize, while models with poor representations fail to fit it. Under plain FT, by contrast, models with poor representations fit the training objective by reshaping their pretrained representations. We reproduce the distillation result in Pythia-160M and Pythia-410M, each pretrained multiple times with different random seeds, and show that the transferable representation lives in the deep layers at both scales. Overall, PT acts as a lottery over representations, and the representation a model draws determines how it adapts during FT in a way that behavior alone does not explain. 

---
# Multimodal LLMs Know the Game but Assume the Real World 

**Authors**: Jianxin Gao, Bohan Chen, Runze Li, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28745)  

**Abstract**: Multimodal large language models can state the rules of a video game, but do they apply them when they only see the game? We introduce RuleCraft, 1,240 image questions over 155 Minecraft rules with outcomes verified in the game engine, each asked in text, on the image with the game named, and on the bare image. Six widely used models know the rules and recognize the game, yet answer the bare image by real-world physics: naming the game raises their accuracy on rules where the game departs from reality by 8.2-19.8 points on a main set and by 33.8-52.7 on a hard stress-test set. Their errors pick the real-world outcome, the gap is largest where the game leaves a scene unchanged, and reasoning traces state the game's rule, then answer by reality. On the main set, naming the game mostly swaps the real-world default for the game's. The pattern holds with reasoning, recurs in two more games, and persists on questions the models themselves take to be about the game; one neutral sentence asking for the rules of the world shown recovers much of the gap. Evaluations that name the game measure knowledge of its rules, not world selection. 

---
# Entangle: Scaling Human-Video In-Context World-Action Models across Unseen Tasks and Contexts 

**Authors**: Chendong Xiang, Chengwen Zhang, Ruowen Zhao, Zeyuan Wang, Shuhe Huang, Rongxu Cui, Min Zhao, Fan Bao, Hengkai Tan, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28732)  

**Abstract**: In-context learning (ICL) enables foundation models to perform unseen tasks from context without parameter updates. In robotic manipulation, a human video provides the context that specifies how a task should be performed. However, scaling human-video ICL on world-action models (WAMs) faces two challenges: task-level human-robot pairs are not temporally aligned with robot actions, and standard action losses allow policies to rely on initial-scene or task-identity shortcuts rather than tracking the context. We present Entangle, an in-context world-action model that maps a single human video and current robot observations to future visual states and executable actions. To scale temporally aligned training data, we convert recorded robot videos into matched human contexts via video-to-video generation, supplemented by real paired recordings. For model training, we combine variable-length video windows with an intra-task contrastive objective that distinguishes the paired action trajectory from other executions of the same task, preventing task-identity shortcuts. Without task-specific fine-tuning, Entangle improves steadily as aligned training data scales, reaching an average score of 83.5/100 across six unseen tasks on an Astribot S1 robot, and achieves a 69.5% average success rate on seven unseen RoboTwin tasks, outperforming Zero-WAM by 22.6 percentage points. Under Context-OOD, although trained only on scene-aligned pairs, Entangle transfers context-specified behaviors across rearranged layouts and substituted tools. 

---
# Seeing Is Not Planning: Spatiotemporal Planning and Plan Repair in Multimodal LLMs 

**Authors**: Siyuan Li, Ethan Lin, Xiang Li, Tianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28726)  

**Abstract**: Planning in a visual world means reading the layout, foreseeing how the state will change, and following the task rules all at once. When a plan fails, a model should also use what went wrong to fix it. We call these two abilities spatiotemporal planning and plan repair, and test them with Railbound-Bench. A model sees a railway puzzle, adds tracks under a budget, and submits one complete layout; a simulator then moves all cars at once and decides whether the layout works. After a failure, the model sees what happened and may submit a new layout, up to five attempts in total. On the 200-puzzle main set, the best of nine models solves 59.5% within five attempts, the next 39.5%, and the other seven at most 10.0%; a 1,000-puzzle extended set with a wider range of difficulty is far easier, at 90.9% for the best model. Controlled studies on 180 fixed puzzles show that knowing the task is not the same as using it. GPT-6.1 Terra copies 97.2% of boards exactly and, given a correct layout as text, predicts all four requested future states in 91.3% of cases, yet only 22.8% of its first layouts work. Planning fails more often as the cars' routes interact, even among puzzles that a search solver finds equally easy. Visual feedback removes the reported fault in 175 of 417 first revisions, but only 43 of these revisions solve the puzzle, and at an equal budget of 40k tokens it does no better than fresh retries. The code and data are open-sourced on GitHub at this https URL. 

---
# AERIN: Aerial Instance Navigation with a Visual World Model 

**Authors**: Zhenzhen Jiao, Rui Tian, Zhineng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.28723)  

**Abstract**: Searching from high altitude and navigating to a target instance specified by a reference image is a common application for UAVs. In such tasks, high-altitude observations provide broad ground coverage for efficient search, but targets occupy fewer pixels, making instance verification and action selection more difficult. We present AERIN, a visual world model that uses action-conditioned prediction to guide the UAV toward the target while adjusting its viewpoint to obtain higherquality observations for subsequent instance verification. Highaltitude navigation requires fine-grained target details and scene context relevant to visibility, but repeatedly predicting dense visual representations of this information is computationally expensive. AERIN addresses this challenge with a compact, fixedsize, target-conditioned latent state that separately represents both for efficient action evaluation. Experiments demonstrate that AERIN outperforms the evaluated baselines in target approach and observation quality, while requiring substantially less decision time than similar world models. 

---
# KMed-31B Technical Report 

**Authors**: KMed Team at NAVER  

**Link**: [PDF](https://arxiv.org/pdf/2611.28719)  

**Abstract**: We present KMed-31B, a medical large language model post-trained from an open-weight model for on-premises deployment with limited hardware. Post-training a model for diverse medical tasks remains difficult, because improving one kind of task can lower performance on another. We attribute this to differences between tasks in expected output and available feedback. Medical question answering and diagnosis prediction have reference answers, whereas clinical inquiry has no single correct response and is judged with rubrics. We therefore divide medical training tasks into four categories by these two properties, train one specialist per category with a reward that its feedback supports, and integrate the specialists into a single student with Multi-teacher On-policy Delta Distillation (MOPD$^2$). For clinical inquiry, we build Synthetic Health Dialogue, a simulated corpus of consultations with lay users and clinicians whose patient cases span common clinical presentations in primary care. Relative to its base model, KMed-31B raises HealthBench Hard from 22.95 to 41.87 and HealthBench Professional from 37.92 to 50.20 in non-thinking mode, and from 23.80 to 51.48 and from 41.97 to 59.18 in thinking mode, while its scores on four multiple-choice medical benchmarks are maintained or slightly improved. KMed-31B has the highest unadjusted HealthBench Hard score among the compared models, including proprietary ones, and scores higher on HealthBench Professional than every open-weight model we compare, including models with more than ten times as many total parameters. Applied to another base model, Qwen3.8-27B, the same training recipe also raises both HealthBench scores. We describe the training recipe, data construction, and settings in detail so that institutions requiring on-premises deployment can develop medical models from open-weight foundations. 

---
# SkillLens: An Interactive Human-LLM System for Diagnosing and Refining Agent Skills 

**Authors**: Shuo Yan, Rui Sheng, Xiaoyan Gu, Lian Lian, Chengran Xu, Yifang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28706)  

**Abstract**: LLM agents increasingly rely on reusable skills, yet maintaining them requires developers to connect revisions with changes in behavior and task outcomes. Existing workflows fragment evidence across versions, evaluation cases, and execution traces. We present SkillLens, an interactive human-LLM system for diagnosing and refining agent skills. A formative study with five developers and researchers identified challenges in prioritizing failures, reconstructing case-level evidence, and preserving context across revisions. SkillLens links performance changes, case transitions, skill diffs, invocation sequences, traces, and outputs in a recoverable workflow, while keeping model-generated explanations and refinements under human review. Through a between-subjects user study with 20 participants, we show that SkillLens supports developers in diagnosing and refining agent skills through a more usable and less demanding workflow than the baseline. These findings highlight the value of linking version comparisons, case-level evidence, and human-reviewed refinement across skill iterations. 

---
# PopMelodyGen: A Copyright-Aware Platform for Controllable Pop-Melody Generation 

**Authors**: Di Lu, Xinran Zhang, Qingwen Zhou, Yan Gao, Yuan Zhang, Xiaoqing Wang, Yuming Sun, Xiaobing Li, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.28686)  

**Abstract**: Large-scale redistributable symbolic pop-melody data remain scarce due to copyright restrictions, creating a bottleneck for both practical deployment and reproducible research in music generation. We present PopMelodyGen, a copyright-aware platform for controllable pop-melody generation that combines rule-based synthesis with neural generation. Our approach first uses an expert synthesizer to produce large-scale labeled synthetic melodies from explicit musical knowledge and templates, providing scalable pretraining data with auditable provenance. A neural model is then pretrained on this corpus and adapted with a small licensed real-pop corpus to improve melodic naturalness while preserving controllability. The platform supports interactive melody creation with accompaniment playback and export for creators, as well as batch construction of annotated symbolic datasets for researchers. PopMelodyGen demonstrates a practical training and deployment paradigm for creative AI when large-scale reusable real-world data are limited. 

---
# The Evaluation Substrate Matters: Benchmarking Multi-Agent Serving Substrates 

**Authors**: Haoyuan Zheng, Dongting Li, Hancheng Ye, Qinsi Wang, Mingyuan Ma, Yuncheng Yao, Yifei Wang, Baijun Zhou, Zishan Shao, Tingjun Chen, Hai Helen Li, Danyang Zhuo, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.28671)  

**Abstract**: Multi-agent systems (MAS) usually coordinate heterogeneous models and tools to solve complex tasks, creating a growing need for reliable evaluation. However, a multi-agent workflow that is fastest in one serving environment can lose its advantage in another, even when its models, prompts, and task remain fixed. This makes a benchmark result conditional on how the workflow is executed. We introduce MASSys, a benchmark for measuring this dependence in multi-agent large language model (LLM) systems. MASSys executes the same candidate-task pairs across serving substrates and load regimes, using five workload families spanning software repair, research synthesis, financial analysis, interactive control, and diagram generation. It records task quality alongside delivery, latency, cost, resource pressure, and failure evidence. In a controlled motivation diagnostic, the Star workflow falls from first to fourth in mean-latency rank under KV-cache pressure, while Chain rises from fourth to first. The substrate sweeps further show that increasing load creates long runtime tails and timeout pressure. In several task families, quality among evaluated outputs changes less than effective quality over submitted jobs: the system delivers fewer usable results even when the surviving outputs remain competitive. Tool and runtime traces reveal that these outcomes involve both model serving and non-model execution. Together, the results motivate evaluating MAS designs over explicit operating conditions and reporting which tasks are delivered, how well they are solved, and where execution breaks down. 

---
# Make Position Uninformative: Exchangeable Self-Distillation Flattens the Lost-in-the-Middle Curve 

**Authors**: Qingsen Ma, Dianyun Wang, Jiahong Liu, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2611.28638)  

**Abstract**: Retrieval-augmented language models answer well when the relevant document comes first or last and poorly when it sits in the middle. Some fine-tuning recipes lift the middle by lowering the start. We show how training contexts teach positional bias. With a uniform answer slot and shuffled distractors, the optimal reader ignores position. Building on this result, Exchangeable Self-Distillation (ExSD) trains a small adapter on shuffled contexts to reproduce the model's answer when it reads the answer document alone. It needs no gold answers and only a few thousand examples. On six models from 1.5B to 32B parameters, ExSD raises both average and worst-position accuracy against the frozen model, including on a held-out test split. In controlled LoRA comparisons, it significantly improves both criteria over Pos2Distill. Against CORD, mean differences are not significant, while worst-position accuracy improves significantly in the 20- and 30-document settings. ExSD also matches or beats gold-answer fine-tuning, changes behavior on other tasks far less, and transfers to unseen tasks. Head probes show greater reliance on content over position. 

---
# LoopSeek: Decoupling System One Search from System Two Answering for Long Video Understanding 

**Authors**: Chengzhong Wang, Lang Lin, Xinyang Liu, YueQin Yin, Mingyuan Zhou, Shengwei An  

**Link**: [PDF](https://arxiv.org/pdf/2611.28635)  

**Abstract**: Long video question answering requires finding relevant evidence before reasoning over it. With a limited frame budget, single-pass frame selection can miss critical evidence, while adaptive search often relies on repeated text generation to decide where to look. We introduce LoopSeek, a framework that decouples evidence search from answer generation through a division of labor between a non-thinking System I and a generative System II. Our key insight is that search can be expressed as a sequence of small, typed decisions, allowing System I to find the evidence that System II needs. Four System I role agents navigate a tree of video windows and coordinate through probability outputs to select frames, assess evidence, and revise the search without generating text. To align search with System II, System I learns from annotated evidence on questions that System II answers correctly only from that evidence, and the verifier's evidence estimates steer the search without calling System II, which stays frozen and answers once from the selected frames. On five long-video benchmarks, with a frozen Qwen3-VL-8B reading 32 frames, LoopSeek improves average accuracy from 53.5 to 63.9 over the uniform baseline and takes 32--80\% less GPU time per question than agentic methods. 

---
# Fast World Planner: Advantage-Aware Action-Chunk Policy for Online Planning 

**Authors**: Jiangtao Kong, Yuchen Wang, Yandong Luo, Rongchao Xu, Guang Wang, Yanbing Mao, Lu Gan, Tianyi Zhou, Huajie Shao  

**Link**: [PDF](https://arxiv.org/pdf/2611.28616)  

**Abstract**: Model Predictive Path Integral (MPPI) control is central to online planning for model-based reinforcement learning (MBRL), but its iterative refinement incurs planning latency. Reducing refinement iterations often degrades performance, motivating stronger policy guidance learned from the planner. However, existing methods typically learn from the planner through action-level alignment, which does not directly capture the complete $H$-step planner proposal. Moreover, planner imitation leaves further target improvement to online refinement. To address these limitations, we propose FaSt World PlANner (SWAN), an advantage-aware MPPI-based MBRL framework that explicitly represents and improves the complete $H$-step planner proposal to improve policy guidance for efficient online planning. For proposal representation, SWAN combines k-MPPI, a multimodal MPPI extension, with a mixture of factor analyzers (MFA) action-chunk policy to capture multiple modes and correlations across time and action dimensions. For proposal optimization, we introduce Advantage-Guided Proposal Optimization (APO), which reuses MPPI evaluations to construct an improved target using advantage estimates and trains the policy toward this target while preserving planner alignment. Experiments across 31 high-dimensional continuous-control tasks from DMControl, HumanoidBench, and MyoSuite show that SWAN substantially improves control performance and sample efficiency while using 50% fewer MPPI refinement iterations than state-of-the-art planning-based MBRL methods. Finally, SWAN demonstrates zero-shot sim-to-real transfer on a physical Unitree Go2, achieving stable velocity-tracking locomotion with high-frequency control. (see our demo at this https URL) 

---
# Action-Conditioned Transition Irreversibility for Out-of-Distribution Detection in Reinforcement Learning 

**Authors**: Emil Mittag, Richard Dazeley, Peter Vamplew  

**Link**: [PDF](https://arxiv.org/pdf/2611.28611)  

**Abstract**: Reinforcement-learning agents shape the observations they encounter; consequently, detecting out-of-distribution (OOD) states requires analysing action-conditioned trajectories rather than treating inputs as independent samples. We propose Action-Conditioned Transition Irreversibility (ACTI), a detector that evaluates forward and backward transition likelihoods, calibrates this comparison against the in-distribution profile at each timestep, and smooths the score over a short window, using no future information. We tested ACTI across six environments and fifteen detectors within OOD-RL-Bench, using the four anomaly mechanisms that qualified in every environment, with ten policy replicates per environment. All primary metrics are matched-time: each score after anomaly onset is compared with a clean twin rollout's score at the same timestep. Our method secured the top macro matched-time AUROC, outperforming the dynamics-model baseline (PEDM), while scoring transitions roughly two orders of magnitude faster. Compared to the state-only variant, ACTI achieved a small but consistent AUROC gain and a shorter detection delay. The capacity-matched forward-only baseline, ACTI-Forward, secured a statistically significant advantage in AUPRC alongside a reduced detection delay. The discrepancy in AUROC did not reach statistical significance. Ablation studies suggest instead that performance improvements stem from action conditioning, calibration at each timestep, and smoothing of the transition-likelihood scores. Based on these results, we recommend using ACTI-Forward. 

---
# BitShare: Sensitivity-Aware Binning and Bit Sharing for Compacting LLM Weight Storage 

**Authors**: Qiuyu Ren, Sudipta Paria, Aritra Dasgupta, Swarup Bhunia  

**Link**: [PDF](https://arxiv.org/pdf/2611.28607)  

**Abstract**: Decoding with large language models at small batch sizes is limited by memory bandwidth, so the bits stored per weight largely set its cost. Group-wise integer quantization stores every weight of a group at the same precision, although the input channels of a group differ in sensitivity. We present BITSHARE, a codebook-free weight format that keeps the integer grid and the dequantizer of group-wise quantization but varies precision within each group. Sensitivity-aware binning places the channels of each group into three bins at fixed positions, and static bit sharing with optimal bit-value allocation lets the less sensitive weights share their lower bits, each stored once per group or once per pair of weights. A sharing-aware version of GPTQ chooses every stored value, with no fine-tuning and no codebook. BITSHARE has two modes. BITSHARE-U gives every group the same layout, so its decoder routes bits by fixed wiring, while BITSHARE-N chooses a layout per group within the same bit budget. With calibration on the Pile and at the same storage, both modes have lower perplexity than GPTQ at 3 and 2 bits on WikiText-2 and C4. At 2 bits across Llama-2 and Llama-3 models, BITSHARE-N lowers WikiText-2 perplexity by up to 70% against SliM-LLM and up to 35% against GPTAQ. At the 2-bit perplexity of SliM-LLM, BITSHARE-U stores 13.6-19.4% fewer bits per weight, cutting DRAM weight traffic by the same fraction. A verified hardware decoder implementation synthesized in SkyWater 130 nm reconstructs BITSHARE weights at the throughput of INT-n and SliM-LLM decoders. 

---
# Bridging Teacher-Student Capability Gaps with Recurrent Computation 

**Authors**: Zeyi Huang, Jiuxiang Gu, Eli Shechtman, Yong Jae Lee, Zongze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28603)  

**Abstract**: Knowledge distillation transfers capabilities from large language models to smaller students, yet stronger teachers do not always yield better students. Recent studies report diminishing or even negative returns from teacher scaling. We hypothesize that these limitations arise in part from insufficient student capacity to absorb the teacher's knowledge. We therefore investigate loop and recurrent computation as a way to expand the student's effective capacity by modifying its computation graph while keeping the backbone parameter count fixed. Under teacher-forced KD during pretraining, weight-shared looping improves performance but retains teacher-size saturation, whereas cross-token recurrence avoids the observed reversal with larger teachers. In mathematical post-training under both teacher-forced KD and on-policy distillation (OPD), recurrent students scale better with increasing teacher size than looped and vanilla LLMs do. Crucially, we show that a pretrained LLM can be readily fine-tuned into a looped or recurrent architecture jointly with OPD, without repeating pretraining. Our recurrent Qwen model improves with larger teachers and consistently outperforms looped and vanilla LLMs. This suggests a new post-training paradigm that integrates architectural conversion into distillation, avoiding a separate conversion stage while substantially improving student performance and maintaining near-vanilla inference latency. 

---
# MedCoast: Semantic Abstraction of Reasoning Trajectories for Test-Time Biomedical Question Answering 

**Authors**: Yishan Zhong, Jieru Li, Xuanzhe Ding, Shaunna Wang, Yanxin Yang, Micky C. Nnamdi, J. Ben Tamo, Yifei Wang, Yining Yuan, Benoit L. Marteau, Shaun Q. Y. Tan, Carrie Moore, Luyu Xie, Yang Xie, Wenqi Shi, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28586)  

**Abstract**: Biomedical question answering often requires multi-step reasoning under limited inference budgets. While test-time search methods such as Tree-of-Thought can improve reasoning performance, directly searching over free-form language is computationally expensive and highly redundant, as many reasoning trajectories contain semantically equivalent intermediate steps expressed in different surface forms. We introduce MedCoast, an offline-to-online reasoning framework that converts sampled chain-of-thought trajectories into a compact space for structured process supervision. Offline, we recursively cluster semantically similar reasoning steps under shared prefixes to construct an abstract reasoning tree that captures recurring clinical reasoning patterns. We then derive node-level supervision from progress toward correct solutions, and train a lightweight process reward model (PRM) to evaluate partial reasoning states. At inference time, the learned critic guides test-time exploration by prioritizing promising reasoning prefixes. Across seven biomedical QA benchmarks, MedCoast often improves over self-consistency under fixed inference budgets, with gains of up to 15 points. Despite using a reward model more than 50 times smaller, MedCoast achieves performance comparable to state-of-the-art reward models in several settings, highlighting its potential for resource-constrained biomedical reasoning. 

---
# AsyncMixVLA: Runtime Asynchronous Robot Policy Switching for Adaptation to Dynamic Environments 

**Authors**: Tamoghno Das, Keyon Jazayeri, Yerin Cho, Nazmul Takbir, Arshia Aravinthan, Nikil Dutt  

**Link**: [PDF](https://arxiv.org/pdf/2611.28585)  

**Abstract**: Vision-language-action (VLA) models translate visual observations and language instructions into robot actions, enabling general-purpose robotic control. However, environmental perturbations, such as objects moving during execution, can undermine their reliability. Different policies exhibit complementary failures under mid-rollout perturbations, motivating selective switching between policies for recovery. Such switching can stall the robot during inference of the newly invoked policy or, if performed asynchronously, act on stale observations. We present AsyncMixVLA, an adaptive inference framework for proactive, asynchronous switching across heterogeneous VLA policies. A learned trigger balances recovery potential against intervention harm to selectively escalate to a higher-compute policy. AsyncMixVLA overlaps its inference with committed outgoing actions and aligns its context to the anticipated takeover state. On LIBERO-Long with mid-rollout perturbations, AsyncMixVLA improves perturbed task success from 69.5% to 77.1% over the lightweight policy. It achieves 84.9% overall success, comparable to 84.5% for synchronous switching, while reducing median exposed handoff latency from 226.5 ms to 0.1 ms. Without retraining or recalibration, it achieves 75.1% success across three unseen perturbation families, compared with 71.3% for the stronger constituent policy in aggregate. 

---
# Right Verdict, Right Reason? An Execution-Grounded Study of Whether LLM Vulnerability Detectors Understand Vulnerabilities 

**Authors**: Leizhen Zhang, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.28584)  

**Abstract**: Large language models (LLMs) are increasingly used to detect vulnerabilities, and their benchmark scores are often read as evidence that they understand the vulnerabilities they report. We test this reading against program execution. For 515 C and C++ functions from BigVul and PrimeVul whose vulnerability we reproduce with a proof of concept, we identify the line at which the fault occurs and the statements on which the faulting run depended. These facts ground a ladder of five tests, from telling a vulnerable function from its fixed version, through locating the fault and reacting to its removal, to judging execution-confirmed edits and predicting the outcome of a concrete run. Each test is paired with a baseline that needs no understanding, such as a rule that picks lines with risky calls, and we repeat the study on 195 vulnerabilities published after the training cutoffs of the models. Four findings emerge. First, telling a vulnerable function from its fixed version is hard. The open-weight models separate no more pairs than chance, although five of them show a weak sensitivity to the fix, and the commercial models still label many screened fixed versions vulnerable. Second, the commercial models locate the fault beyond a keyword rule, while removing that evidence changes the verdicts of the strongest model only on the newer vulnerabilities. Third, given the program input, the commercial models predict whether a run faults better than a rule that looks for added checks, and the prediction follows the input. Fourth, on the newer vulnerabilities this advantage disappears and the lead of the strongest model narrows, while localization persists. We release the framework, the per-function evidence, and an execution-confirmed counterfactual benchmark. 

---
# CLASP: Deciding Structure and Claims Before Decoding for Whole-Slide Pathology Report Generation 

**Authors**: Ziaullah Khan, Rashadul Islam Sumon, Md Ariful Islam Mozumder, Duc-Son Pham, Moon-Il Joo, Hee-Cheol  

**Link**: [PDF](https://arxiv.org/pdf/2611.28576)  

**Abstract**: Whole-slide pathology report generation has advanced through stronger slide encoders, retrieved knowledge, and, recently, slide-level multimodal LLMs. However, these generators leave every report-level choice to the decoder: how long the report is, how it is organised, and which findings it states and where. On the standard PathText-BRCA benchmark, we find that these choices dominate n-gram overlap, as reports from the query's hospital selected without learning approach the state of the art, while decoders under-generate and fall back to the most frequent findings. We present CLASP, a decide-then-decode generator that makes these choices explicitly, once per slide. Before decoding, an Exemplar Structure Prior (ESP) takes the sentence plan and length from reports of image-similar cases, a Calibrated Claim Decision (CCD) selects the findings from calibrated slide posteriors, and a Claim-to-Sentence Placement (CSP) assigns each finding to a planned sentence. A Plan-Conditioned Decoder (PCD), refined by structure-anchored sequence training, then writes the report under these decisions. With 35M parameters, CLASP outperforms retrained report generators and slide-level LLMs of up to 7B parameters, matches the strongest published results, and generalises to lung cancer from unseen hospitals. Its decisions change what the reports state, not only their length, transfer to an LLM decoder, and can withhold what H&E cannot show: withholding receptor status leaves 23% fewer contradictions with clinical records than the same network without decisions. 

---
# MACE: Memory-Augmented Crop-Disease Evaluation of Frozen Vision-Language Models 

**Authors**: Tirtho Roy, Shreyan Ganguly, Muhammad Arbab Arshad, Nikhil Bhattacharya, Soumik Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2611.28573)  

**Abstract**: In plant phenotyping, crop disease diagnosis has traditionally relied on expert assessment of visible symptoms and on models tailored to specific crops or diseases, and both are difficult to scale across field conditions. Recent methods provide frozen vision-language models with source-grounded knowledge bases that summarize the typical symptoms of each disease but describe little of how symptoms vary across individual cases. We introduce MACE (Memory-Augmented Crop-Disease Evaluation), a training-free method that supplements the knowledge base with a memory of symptom descriptions paired with known disease labels. MACE matches a description of the query image against this memory and retrieves only text descriptions and disease labels, so the query is the only image that the diagnosis model sees. Across ten crops, MACE with Claude Sonnet 5.5 improved accuracy over the knowledge-base baseline from 42.8% to 62.2% in distribution and from 50.7% to 57.0% on held-out image collections. Retrieved disease labels accounted for most of the gain, and adding the corresponding text descriptions improved in-distribution accuracy by about six percentage points. 

---
# Tool-to-Self: Internalizing Visual Tool Capabilities via Scaffolded Reinforcement Learning 

**Authors**: Jin Cui, Hao Dong, Xinyue Long, Boran Zhao, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.28564)  

**Abstract**: Vision-language models (VLMs) increasingly rely on external visual tools to access fine-grained visual evidence, but repeated tool execution incurs substantial inference overhead and deployment dependencies. Moreover, conventional tool-use RL can improve visual-task performance, yet the resulting tool gains are often unstable. Motivated by this, we propose Tool-to-Self (T2S), a tool-scaffolded reinforcement learning framework that turns external visual tools into temporary training scaffolds. T2S cold-starts tool-use ability and observation prediction from tool-repairable examples, then progressively anneals tool execution on a per-tool basis during RL. By combining answer-correctness and observation-consistency rewards with privileged on-policy self-distillation from actual tool observations, T2S effectively teaches the model both to reproduce tool outputs and to reason from them. Experiments across diverse visual perception and reasoning benchmarks demonstrate consistent improvements over both explicit tool-use models and implicit zooming models while enabling tool-free inference with reduced latency. 

---
# SAFE-DRIFT: Data Selection for Supervised Fine-tuning with Controllable Off-Target Drifts 

**Authors**: Yeo Jin Jung, Yating Liu, Lalchand Pandia, Claire Donnat  

**Link**: [PDF](https://arxiv.org/pdf/2611.28560)  

**Abstract**: We propose SAFE-DRIFT, a data-selection framework for supervised fine-tuning that improves a target capability while controlling unwanted drift in the model's other behaviors. We formalize this as a constrained optimization problem that maximizes alignment with the target gradient, subject to a budget on off-target drift which we quantify using the Fisher information of a reference distribution. Under this objective, the optimal, closed-form update is a damped natural-gradient step that downweights reference-sensitive directions. To estimate these high-dimensional gradients and work within the right geometry, we further derive a scalable algorithm that approximates the Fisher information and the target gradient in a shared low-dimensional subspace and analyze how entanglement of the target and reference task can affect the update. We evaluate SAFE-DRIFT against state-of-the-art data selection methods across diverse domains including coding and medical question answering, yielding competitive target performance while reducing degradation on specified off-target behaviors. 

---
# From Reward Engineering to Data-Centric Reinforcement Learning for Effective Agentic Visual Tool Use 

**Authors**: Jin Cui, Xinyue Long, Jiepeng Zhou, Boran Zhao, Pengju Ren, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.28558)  

**Abstract**: Multimodal large language models (MLLMs) increasingly use external visual tools to acquire evidence during reasoning, yet learning when to invoke these tools remains challenging. Existing tool-use reinforcement learning (RL) methods often rely on call-contingent rewards to encourage beneficial tool use, but such reward engineering can distort the task objective and lead to degenerate calling behaviors. We argue that effective tool-use learning requires identifying not merely which questions are difficult, but which failures are attributable to missing visual evidence and can therefore be corrected through tool observations. We propose AOVO, a data-centric framework that enables MLLMs to learn effective tool use through perception attribution and policy-adaptive data sampling (PADS), without introducing tool-specific rewards. It counterfactually probes the model's reasoning states to identify observation-sensitive questions, distinguishing them from reasoning-dominated ones. PADS then periodically estimates category-level observation effects from recent policy rollouts and dynamically reallocates the training distribution toward questions where visual observations provide informative learning signals, while optimizing solely for final-answer correctness. Experiments across diverse visual perception and multimodal reasoning benchmarks demonstrate consistent improvements over competitive tool-use MLLMs, with causal analysis further validating the data-centric perspective for learning effective visual tool use. 

---
# General-purpose language models are neurosymbolic reasoners able to annotate proteins from sequence 

**Authors**: Christopher J. Mungall, Marcin P. Joachimiak  

**Link**: [PDF](https://arxiv.org/pdf/2611.28529)  

**Abstract**: Determining the function of genes across the tree of life is one of the major challenges in biology. Generative AI approaches to this problem generally fall into two categories: language models trained on biological sequences that are able to predict structure and function from signals in those sequences; and generalist language models trained on text, that are able to reason about biological function symbolically. A default operating assumption is that protein language models do not perform well on symbolic tasks, and that generalist language models do not perform well on sequence tasks. However, a recent report showed that frontier generalist models perform surprisingly well on next-residue prediction tasks. We set out to explore whether these capabilities extended to protein annotation.
We assigned different frontier generalist models a variety of protein annotation tasks, including predicting the presence and coordinates of Pfam domains, loss of functional catalytic residues, and predicting structure and activity. On all annotation tasks, the frontier models performed well above chance; we did not benchmark them against specialised tools. The models also make characteristic hallucination errors, and they state them as confidently as their correct calls.
Though untested against specialised methods, the nascent ability of frontier models to perform protein annotation from sequence alone is intriguing. The ability to reason over sequence, structure, and symbols in a single model allows for a neurosymbolic approach where a combination of sequence, experimental data and knowledge from the literature can be holistically combined to improve on de novo functional annotation of the protein universe.
Prompts, test sets, model outputs and code: this https URL. Summary of results: this https URL 

---
# Digital Twins for Multi-Agent Systems: When to Trust, What to Change 

**Authors**: Truc Nguyen Thi Le, Phuoc Nguyen T. H  

**Link**: [PDF](https://arxiv.org/pdf/2611.28517)  

**Abstract**: Digital twins of agent networks need to flag unreliable forecasts and help users choose interventions. Both decisions depend on what happens over several steps: an error can spread through the network, and an edit whose trajectory is easy to predict may still yield a poor collective outcome. We ask when information about error propagation improves failure detection and whether a learned twin can select effective network edits despite imperfect forecasts. We propose Propagation-Aware Validation for Networked Digital Twins (PAV-Twin). Its monitor combines Jacobian-based propagation features with recent errors and ensemble disagreement to estimate the risk that a forecast exceeds an error threshold. Its planner simulates each candidate edit and ranks the edits by their predicted collective outcomes. In simulations, the monitor gained 0.13 in area under the precision-recall curve on larger oscillator networks with familiar topologies and 0.09 on larger networks with held-out topologies, relative to a graph baseline without propagation features. Gains were small in contractive consensus. Giving the graph baseline the same features narrowed the gap. In a separate planning study, ensemble simulation recovered about 99% of the possible improvement over random intervention selection in oscillators and nearly all of it in consensus. With a simplified oscillator twin, selecting by predicted outcome also gave lower regret than selecting by trajectory fidelity. These results show when propagation features help flag unreliable forecasts and why the quality of intervention choices should be measured directly. 

---
# State of Mind: Learning and Evaluating World Models of Naturalistic Brain Dynamics 

**Authors**: Yuxiang Wei, Zundong Wu, Weiyu Sun, Qizhen Lan, Yanteng Zhang, Vince D. Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2611.28508)  

**Abstract**: Understanding naturalistic brain activity requires going beyond the stimulus alone, as external inputs continuously interact with an individual's evolving brain state. A world model captures such dynamics in a latent state by inferring it from neural observations, predicting stimulus-driven evolution, and updating with new observations. However, existing brain world models fail to maintain a persistent latent state over time, and standard metrics of neural prediction accuracy cannot tell whether a model tracks an individual's brain state or relies solely on the stimulus. To directly test state inference, prediction, and updating, we introduce BrainStateBench, which evaluates a model's latent state through a frozen linear probe on whole-cortex, movie-driven fMRI. Its State task tests whether a model captures trial-to-trial variability in an individual's responses to repeated stimuli, while Forecast and Updating evaluate future activity predictions without and with new observations, respectively. To track individual brain state across time, we propose Spectator, a linear latent world model trained with a joint-embedding predictive objective. Spectator encodes short- and long-horizon observation histories into a two-timescale latent state and predicts state transitions linearly from the current state and stimulus. Stimulus-only encoders reach 0.320 in fMRI prediction correlation without observing the brain (0% on State), while prior deep brain world models reach 0.239 yet score at most 0.4% on State. In contrast, Spectator achieves 0.349 correlation alongside 5.5% on State, 16.0% on Forecast, and 20.3% on Updating, outperforming all baselines while forecasting over 20x faster than deep forecasting models. 

---
# A generative framework for unified representations bridging electrophysiological and haemodynamic brain activity 

**Authors**: Weiheng Yao, Changhong Jing, Xuhang Chen, Chi-Man Pun, Michael Kwok-Po Ng, Shuqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28507)  

**Abstract**: Electrophysiological and haemodynamic recordings resolve brain activity at complementary temporal and spatial scales, yet paired acquisition remains difficult to scale. We propose a generative framework that aligns paired multimodal recordings in a unified representation space and generates modality-specific haemodynamic responses from EEG. In paired cohorts, the framework generated fMRI BOLD and fNIRS haemoglobin responses that reproduced regional dynamics, cortical maps, functional connectivity and task structure. Generated BOLD followed Real BOLD time courses and cortical patterns, while Generated haemoglobin responses tracked task-evoked HbO and HbR dynamics. The framework enabled decoding across visual, motor-imagery and mental-arithmetic tasks, independent-record decoding, external BOLD localization and low-sample augmentation. Paired feature-space calibration supported bidirectional retrieval. The proposed framework extends cortical analysis and brain decoding to EEG recordings when Real haemodynamic data are sparse. 

---
# When AI Performance Is Not Enough: Statistical Guardrails for Clinical Development 

**Authors**: Siddhesh Kulkarni, Gina D'Angelo, Swarna Reddy, Rakhi Kilaru  

**Link**: [PDF](https://arxiv.org/pdf/2611.28497)  

**Abstract**: Artificial intelligence (AI) and machine learning (ML) are reshaping the operational and inferential landscape of clinical development, yet their adoption frequently outpaces the statistical scrutiny these methods demand. This commentary offers first-principles perspectives across four application domains: anomaly detection in clinical trial data, real-time monitoring and adaptive trial designs, biomarker-based enrichment, and virtual and digital twins. For each domain, we articulate the underlying inferential problem, the contribution of AI and ML relative to established statistical methods, and the risks that statisticians, data scientists, and clinical developers must actively manage. Across these domains, a unifying theme emerges: AI and ML methods may succeed locally while failing to satisfy the broader statistical assumptions required for valid clinical and regulatory decisions. These risks include false-positive signals, compromised Type I error control in adaptive designs, conflation of prognostic with predictive biomarkers, and causal claims not supported by data and assumptions. We argue that a clearly defined context of use, relevant statistical assumptions, and consequences of the intended decision should guide the evaluation. For treatment-effect questions, the ICH E9(R1) estimand framework provides a useful organizing principle. Responsible use of AI in clinical development requires not restraint, but rigor: the inferential discipline necessary to make these technologies reproducible, interpretable, and defensible in the service of patients. 

---
# SoK: Splitting the Trigger or Adding Attackers? Disentangling Distributed Backdoor Attacks in Federated Learning 

**Authors**: Zhuotao Lian, Qingkui Zeng, Weiyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28491)  

**Abstract**: Distributed backdoor attacks split a trigger among several federated learning clients, so that the backdoor appears only in the aggregated model. Since DBA, a line of work has reported that such attacks are stronger, more persistent, and stealthier than a single attacker with the full trigger. We systematize this literature and test this belief. We code nineteen primary studies and fifteen papers that evaluate against DBA. The reported advantage is rarely measured against a complete trigger delivered by the same attackers, and none of the fifteen evaluations tests whether individual pieces remain effective. We then run 875 paired federated runs that separate adding attackers from splitting the trigger. In our image experiments, adding attackers raised attack success in every seed of all 30 cells where we measured it, by 7.5 to 83.4 percentage points on average. Splitting the trigger among the same attackers had a smaller effect, and its sign depended on the piece layout, the allocation, and the attackers' local training. With DBA's compact layout, splitting lowered attack success on CIFAR-10. A larger penalty on small CNNs vanished once the attackers trained longer, and an FCBA-style allocation gained a few points. Widely separated pieces helped on CIFAR-10, its grayscale version, and GTSRB. No model required all pieces. Neural Cleanse suppressed single pieces together with the full trigger, but split backdoors with separated pieces survived it better than complete ones. We derive reporting requirements for attack and defense evaluations. 

---
# Open Vision Banana: An Open Framework for Unified Visual Perception through Generation in RGB Space 

**Authors**: Xu Lin, Ruilin Yao, Haoran Xu, Zixiang Wang, Ming Ma, Yu Zhao, Pu Miao, Chonghan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28489)  

**Abstract**: Vision Banana showed that a closed image generator, instruction-tuned to paint perception outputs as RGB images, becomes a generalist vision model. Does this carry over to \emph{open} generators, and if not, what fails? We answer with \textbf{Open Vision Banana (OVB)}, an open framework that adapts the 7B diffusion generator of Qwen-Image-2.1 into one instruction-driven model for metric depth, semantic, instance, and referring segmentation, using a single shared low-rank adapter and public data. Each task is a \emph{task contract} whose RGB encoder, instruction, and decoder share one specification and are checked before training. Studying 42 externally evaluated checkpoints from two 100k-update runs, six exact weight-space merges, and eight controlled shorter runs, we find that the gap to the closed system is specific and measurable. Geometry is learned early while absolute scale is not: median-aligned NYUv2 $\delta_1$ stays at 0.89--0.93 throughout training as unaligned $\delta_1$ wanders between 0.11 and 0.50. The scale is not committed by a single sample: a ten-seed ensemble lifts unaligned $\delta_1$ from 0.348 to 0.516 without retraining. Depth and 2D segmentation compete for the shared adapter, and weight merging moves along this trade-off without escaping it. Finally, lower validation flow loss comes with lower Cityscapes accuracy (Spearman $\rho\approx-0.9$ between loss reduction and mIoU), so the loss cannot select checkpoints. We release code, contracts, training data, and all evaluation outputs. 

---
# Don't Train It, Write It: An AI Agent Writes New Capabilities into Language Model Weights 

**Authors**: Santiago Rivadeneira Quintero  

**Link**: [PDF](https://arxiv.org/pdf/2611.28483)  

**Abstract**: We show that an AI agent can write new capabilities into a pretrained language model by adding gated attention heads and MLP neurons inside its existing layers, without backpropagation and without fitting any weight to the answers of task examples. It has a budget of 50 model passes and sees no validation or test instance. Composed in one model, its modules for six BIG-Bench Hard (BBH) subtasks raise the normalized BBH score of Qwen2.5-1.5B-Instruct from 19.95 to 41.05 (Qwen2.5-7B-Instruct: at most 35.01) and that of Gemma-3-1B-it from 10.25 to 32.86. Against the unmodified model at the same precision, both composites leave the next-token distribution on 1,023,000 WikiText tokens unchanged (KL exactly 0), give identical answers on 15,845 items of five other Leaderboard tests and identical scores on the other 18 BBH subtasks. The agent also writes exact multiplication up to 20 digits, running the model only up to 6 (100.0% on the absent lengths of the sealed test in both models; a LoRA with the same budget: at most 2.1%). Sets, rules and report sentences were sealed in git before the first session (manifests timestamped in Bitcoin), with dated amendments listed in Appendix D; a frozen evaluator ran the sealed test once. Verdicts (H5: tracking; H6: composite; H8: one pass vs. chain of thought; H9: fresh and rewritten instances; H10: side effects; H12: multiplication), on the preregistered sealed test, Qwen2.5-1.5B-Instruct: H5 success, H6 success, H8 (web_of_lies) success, H9 success, H10 (composite) success on (i) to (iii), (iv) confirmed, H12 success; Gemma-3-1B-it (H7: H5, H6, H8 and H9 in this model): H5 success, H6 success, H8 (web_of_lies) success, H8 (tracking 3) success, H9 partial, H10 (composite) success on (i) to (iii), (iv) confirmed, H12 success. Not everything worked: 4 of 45 single modules show small side effects and 5 of 50 sessions hit the 12-hour limit without a module. 

---
# PAMS: Preconditioned Activation-Metric Subspace for Zeroth-Order LLM Fine-tuning 

**Authors**: Pradeep Kumar Ragu Chanthar, Saksham Sahai Srivastava, Jaewoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.28476)  

**Abstract**: Zeroth-order (ZO) optimization provides a computationally efficient approach to fine-tuning large language models (LLMs), but conventional methods typically rely on isotropic perturbations that ignore the anisotropic structure of neural-network parameter spaces. We introduce Preconditioned Activation-Metric Subspace (PAMS), a ZO optimization framework that uses forward-pass activation statistics to construct structured, layer-wise perturbation geometries without computing gradients or forming dense curvature matrices. PAMS tracks a low-rank approximation of the dominant activation subspace and its associated spectrum online, and uses the spectrum to softly reweight perturbation variance while preserving exploration in directions outside the retained subspace. A trace normalization separates the perturbation geometry from its overall scale, enabling consistent perturbation magnitudes across layers. We show theoretically that the resulting perturbation covariance induces right-side preconditioning of the gradient of the corresponding Gaussian-smoothed objective. We further characterize the estimator's smoothing bias and second moment and derive a nonconvex convergence guarantee that captures the effects of the perturbation geometry and smoothing radius. The proposed low-rank tracking procedure requires $\mathcal{O}(n_l r)$ persistent memory for a layer with input dimension nl and retained rank r, avoiding the $\mathcal{O}(n_l^2)$ memory required by a dense activation covariance. Experiments on transformer-based language-model fine-tuning tasks across GLUE and SuperGLUE demonstrate that PAMS improves the effectiveness and stability of zeroth-order optimization while retaining its gradient-free and objective-evaluation-based advantages. 

---
# Automated Safety Analysis for Agentic Workflows 

**Authors**: Yanju Chen, Ying Li, Bosi Zhang, Hanzhi Liu, Hongbo Wen, Yufei Ding, Yuan Tian, Yu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2611.28469)  

**Abstract**: LLM applications are increasingly built as agentic workflows: programs that combine model calls, external tools, and control logic to act autonomously on a user's behalf. Because these workflows execute real-world actions, logical flaws in how a workflow is wired can cause severe harm in deployed systems. Unlike prompt injections or malformed API calls, these workflow bugs stem from defects in the workflow's design, yet no existing tool can statically validate workflow logic across frameworks prior to deployment. We present Apsis, a static safety analyzer that detects workflow bugs before deployment. To operate across frameworks, Apsis lifts framework-specific code into a common intermediate representation (IR). This IR models a workflow as a semantic graph of steps, data dependencies, and shared state, while annotating each component with behavioral properties such as whether an input is mandatory or how a shared channel merges concurrent writes. Building on this representation, we introduce a bug description language that formalizes recurring classes of agentic failures as declarative patterns over the IR. By matching these patterns against the IR, Apsis detects structural and behavioral flaws and pinpoints the exact step responsible for each bug. To evaluate Apsis, we construct, to our knowledge, the first corpus of workflow bugs, distilled from real-world reports across 13 widely used frameworks. On a reproducible benchmark suite drawn from this corpus, Apsis detects workflow bugs with 91% precision and 84% recall, and uncovers 36 previously unknown bugs in real-world workflows. 

---
# HiPPO Functional Memory: Coordinate-Queryable Attention for Irregular Time Series 

**Authors**: Zhaodong Guo, Haoyun Feng  

**Link**: [PDF](https://arxiv.org/pdf/2611.28465)  

**Abstract**: Irregular time series are recorded at nonuniform times and are often queried at other times: between observations, after the last one, or on a regular grid. Continuous-time models answer such queries by integrating latent dynamics, and time-aware attention models answer them by revisiting the observation tokens. We introduce HiPPO Functional Memory (HiPPO-FM), an attention operator that writes an irregular history into a rank-M function space with explicit time coordinates. Masked self-attention first exchanges content among the observations; the updated values are then contracted against kernel-smoothed HiPPO-Legendre features and solved through a regularized Gram matrix, so the retained coefficients define a function that is read at any time through the same features. The write and read have the form of an interdomain sparse Gaussian-process mean: every readout lies in an M-dimensional reproducing-kernel subspace, and a pointwise bound separates the approximation error of this subspace from the error of the learned memory. Removing the functional memory from the encoder raises Spiral reconstruction RMSE 18-fold. The full model has the lowest error in 11 of 12 Spiral settings, the highest overall accuracy and best average rank on 20 irregular UEA datasets, the highest event-type accuracy on all five multi-type event benchmarks, and a 21% lower extrapolation MSE than CRU on PhysioNet. Because the memory has fixed rank, the state retained after encoding keeps a fixed size as the history grows and is 102 times smaller than a token cache at 1,024 observations. 

---
# Guidance-Induced Activation of Score Estimation Errors in Diffusion Models 

**Authors**: Pengyu Luo, Yisha Yao, Bolun Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.28451)  

**Abstract**: Classifier-free guidance (CFG) combines conditional and unconditional score estimates to steer diffusion sampling. The weighted-solenoidal component of each score error has no instantaneous effect on the evolution of its corresponding reference density. Because the guided density in CFG generally differs from both reference densities, the combination of these solenoidal components can affect the evolution of the guided marginals. We call this mechanism guidance-induced error activation. Using a weighted Helmholtz-Hodge decomposition, we identify the activated error field, quantify its magnitude, and characterize when it vanishes. At fixed noise time, the activated error field varies continuously with guidance strength. Its weak first-order variation has two terms, one from reweighting the errors and one from the change in the guided density. For finite score errors, ODE characteristics and an SDE Feynman-Kac formula connect error-activation to the generated densities. For a class of finite Gaussian mixtures, error-activation can account for most of the KL divergence between the generated distributions by learned-score and exact-score CFG ODE samplers under strong guidance. Ideal score filtering removes the weighted-solenoidal errors and eliminates error-activation. Controlled experiments with Gaussian and Gamma mixtures illustrate how activation affects the generated densities. Qualitative CIFAR-10 comparisons suggest that filtering can help preserve object structure under strong guidance. 

---
# Decoding Frozen Vision Representations for Training-Free Few-Shot Pineapple Detection 

**Authors**: Manfred Gonzalez-Hernandez, Danny Xie-Li, Fabian Fallas-Moya, Camile Lendering, Joaquín Figueira, Erkut Akdag, Giacomo D'Amicantonio, Egor Bondarev  

**Link**: [PDF](https://arxiv.org/pdf/2611.28442)  

**Abstract**: Few-shot object detection in agricultural imagery is hampered by dense vegetation, repeated patterns, and scarce annotations. We present a training-free detector that turns dense features from frozen visual foundation models directly into detections, without fine-tuning, a learned head, or external proposals: a whitened foreground-background direction estimated from annotated support images scores every query patch, and the score field is decoded with an adaptive threshold and support-derived geometry. We also test whether predicted relative depth can be fed directly to RGB-pretrained frozen encoders. On aerial pineapple imagery, the detector outperforms PET-DINO and GeCo2 at every evaluated support size; at K=40, predicted depth alone reaches 92% of the RGB performance with DINOv2, and RGB-depth fusion lowers counting error for both encoders. We release the Pineapple Plantation dataset (4,908 aerial frames, 1,995 annotated with 17,609 boxes, predicted depth, leakage-aware splits) and analyse transfer to two further agricultural datasets. 

---
# Lean Census: A Distributional Analysis of AI-Generated Lean Code 

**Authors**: Justin Asher, Pietro Pellegrino  

**Link**: [PDF](https://arxiv.org/pdf/2611.28430)  

**Abstract**: We compare two human-written and four AI-generated Lean libraries with Mathlib through the distributions of simple statistics of their code. For each library we measure inline claims in proofs, proof length, how often each declaration is used, and design choices such as instances and private declarations, and we read each against the variation among Mathlib's own areas. The AI-generated libraries have more inline claims per theorem, longer proofs, fewer instances, and more private declarations than the human-written ones. We release the code for the analysis as a package, lean-census, at this https URL. 

---
# When AI Should Not Decide: Evidence Sufficiency and Decision Stability in Automated Job Interviews 

**Authors**: Khalid Adnan Alsayed  

**Link**: [PDF](https://arxiv.org/pdf/2611.28426)  

**Abstract**: Artificial intelligence is increasingly used to assess employment candidates, yet consistent interview scoring does not establish whether automated recommendations are supported by sufficient evidence. This study introduces the Evidence Sufficiency and Decision Stability Assessment (ESDSA), which combines criterion-level evidence classification, deterministic quotation traceability and stability checks across linguistic variants assessed as substantively equivalent. We evaluated four approaches (scoring-only, evidence-aware, traceability-based and integrated ESDSA v2) using 540 synthetic interview responses from 60 evaluated scenario families across Software Engineering, Customer Support and Operations. In held-out testing (216 responses), integrated ESDSA achieved 85.19% actionable coverage and 3.80% disagreement among actionable recommendations against AI-assisted provisional references. Evidence-aware evaluation achieved 92.59% coverage and 4.50% disagreement; scoring-only evaluation achieved 71.30% coverage and 1.95% disagreement. The integrated stability gate deferred 11 additional previously actionable decisions, ten of which had agreed with provisional references. The findings demonstrate trade-offs between evidential constraints, decision consistency and abstention, rather than overall superiority of the integrated framework. ESDSA offers a reproducible methodology for examining when automated interview recommendations should be actionable or deferred. Synthetic responses and AI-assisted references do not establish real-world recruitment validity. 

---
# Center-Aware Ordinal Learning for Prostate Cancer Grading from Histopathology Images 

**Authors**: Mohammad Mahdi Abootorabi, Parvin Mousavi, Purang Abolmaesumi  

**Link**: [PDF](https://arxiv.org/pdf/2611.28419)  

**Abstract**: Prostate cancer grading from histopathology whole-slide images (WSIs) is complicated by the ordered nature of ISUP grades, class imbalance, and center-dependent variations in staining, tissue preparation, and scanning. Most weakly supervised methods treat grade groups as independent categories and may therefore learn representations that do not transfer reliably across institutions. We propose a center-aware ordinal learning framework that combines a pretrained pathology encoder with attention-based multiple instance learning. The framework uses an Earth Mover's Distance loss to account for ordinal errors, an ordinal supervised contrastive objective to align same-grade slides across centers while separating grades according to disease severity, and curriculum learning to reduce the dominance of benign cases. We evaluate the framework on approximately 9,100 biopsy WSIs from the two-center PANDA dataset. The proposed framework achieves an accuracy of 0.842 and a quadratic weighted kappa of 0.958 on the held-out test set. Its main benefit is observed in cross-center transfer: when a linear probe is trained on Karolinska representations and evaluated on Radboud, accuracy increases from 0.643 to 0.823 and quadratic weighted kappa from 0.899 to 0.930. Performance remains stable in the reverse direction. UMAP projections further show improved mixing between centers while preserving an ordinal grade trajectory. These results suggest that combining ordinal and center-aware objectives produces more transferable slide representations for multi-center prostate cancer grading. 

---
# WingBench: Benchmarking How AI Agents Search for Better Designs 

**Authors**: Yiming Yang, David Xia, Jisen Li, Yuqing Jian, Yunjia Yang, Koray Aydin, Zhaoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28418)  

**Abstract**: Design search often uses cheap learned predictions to choose which candidates deserve expensive tests, but these tests can fail, especially on the unusual designs that optimization favors. We introduce WingBench, a multifidelity benchmark in which optimizers and AI agents design transonic wings with computational fluid dynamics (CFD) inside the search. They can query a learned aerodynamic surrogate and spend a fixed budget of sequential CFD evaluations, each returning a score or a failure type. WingBench grades the design each method recommends, reporting separately whether it receives a valid score and how good that score is. Using WingBench, we contribute a study of five baseline optimizers and a comparison of three AI agent methods, which produced 6,629 CFD evaluation records. Feedback helps baseline optimizers deliver valid designs when it changes their later proposals, but the differential-evolution variant does so with lower aerodynamic quality. An open-ended research agent finds designs of similar quality to the strongest baselines over 32 CFD queries, with different prior information and unmatched total computation. In early search, our reflection and planning implementations do not consistently improve on it. We release the data, code, agent prompts, and analysis scripts for reproducibility. 

---
# ScaleEvals: Scaling Generalist Policy Evaluation through Reset-Free Autonomous Play 

**Authors**: Amitoj Sandhu, Ethan Sick, Omkaar Buddhikot, Nadia Laswi, Wesley Durbano, Nadun Ranawaka, You Liang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2611.28411)  

**Abstract**: Real-world evaluation of generalist robot policies is bottlenecked by manual resets, human grading, and small fixed task suites with limited diagnostic coverage. We present ScaleEvals, a scalable reset-free protocol that evaluates policies through continuous physical play across a robot fleet. A VLM AutoPrompter generates scene-grounded tasks with difficulty estimates, language prompts, and skill tags, enabling performance analysis across diverse rollouts. A VLM AutoGrader assesses task progress, achieving 90.7% agreement with human graders. Scores are aggregated via a soft-win Bradley-Terry model, while tag-stratified ANOVA provides interpretable capability diagnostics. Across 3,000+ physical autonomous rollouts on DROID and YAM, ScaleEvals achieves 23.0 net rollouts/hr per station while reducing active human labor by 98%. By ranking five SOTA generalist policies, ScaleEvals correlates with held-out manual evaluation ranking on DROID (r=0.94, rho=0.90, p=0.018), and remains internally consistent on a second embodiment, YAM (r=0.839, rho=0.80). Beyond aggregate rankings, we observe behavioral compositionality: recombining verb-, skill-, and object-conditioned evidence predicts policy order on unseen prompts. Across 10 tasks and 240 prospective physical rollouts, we recover three-policy orderings of 96% pairwise concordance, demonstrating ScaleEvals can predict not just which policy leads, but where rankings reverse. 

---
# When Can an AI Agent Serve as a Model of Human Psychopathology? 

**Authors**: Marc-Andre Schulz, Pauline Scheuber  

**Link**: [PDF](https://arxiv.org/pdf/2611.28405)  

**Abstract**: Language models and AI agents are increasingly described in psychiatric terms, such as anxious, depressed, addicted or helpless. A recent study even proposed language models as experimental systems for psychiatry and as a platform for piloting psychological treatments. We ask when an experiment on an AI agent can provide evidence about a specified human psychiatric process. A claim to such evidence has three parts. Something changed in the agent, and the agent's history caused the change. Nothing other than a process shared with people explains that change. The change matches one theory of the human process and not that theory's rivals. Psychiatry and its neighbouring fields have an evidence standard for each part, from the diagnostic manuals' rule for induced states, from clinical test scores and from animal models. We translate each standard into a criterion that a machine study can meet. For a language model the most critical alternative to a shared process is extrapolation from its training text, which already contains descriptions of how people respond to trauma and how they answer questionnaires. We show how to probe that alternative. Further, we audited forty-three published studies that apply psychiatric terms to a machine or induce a comparable state in one. Most compared the machine with an identical copy that had not gone through the history that induced the state. None excluded extrapolation from training text. Three tested a psychiatric theory against its rivals, and none of the three tests was conclusive. Until better evidence exists, a treatment pilot on a language model is best read as the model's forecast of patient response, and not as an experiment on a model organism. Evidence for a model organism will come from states induced by what happens to the machine, from experiments on its internal workings, and from tests on which rival theories disagree. 

---
# Targeted In-Context Poisoning of Tabular Foundation Models: An Empirical Study 

**Authors**: Adem Alparslan  

**Link**: [PDF](https://arxiv.org/pdf/2611.28396)  

**Abstract**: Tabular foundation models (TFMs) predict the label of a query row conditioned on a context of labelled rows, without training on them. Drawn from operational data, the context becomes an attack surface. We study targeted in-context poisoning in its simplest form: an attacker with write access to the context's source places exact copies of a target row there with the wrong label. On 30 OpenML datasets for binary classification, one such poison row flips on average 53% (TabICL) and 65% (TabPFN) of the predictions correct without attack, and three poison rows flip almost all. The same copy with the true label flips almost none, and performance on the remaining rows hardly changes. Predictions flip by two routes: in TabICL, the target row's embedding rotates towards the mean of the wrong class; in TabPFN, the poison row pulls genuine rows of the wrong class along. The TFMs thus infer soundly, but from forged evidence. Predictions of trained models flip as well: almost always for a random forest, for CatBoost as often as for the TFMs, hardly for a logistic regression. New with TFMs is not the kind of vulnerability but the timing: the data take effect at the moment of prediction, not through training, and a poison row in a live context can be deleted afterwards. Three established requirements must therefore cover the context: access control for its source, traceability per prediction, which can also deter, and validation at prediction time. A detection mechanism shows how such validation can work. 

---
# Taking the PULSE of Enterprise Workflows: Evaluating Persistent Project-State Understanding 

**Authors**: Jiexin Ding, Lin Ai, Adam Davies, Scott Counts  

**Link**: [PDF](https://arxiv.org/pdf/2611.28374)  

**Abstract**: Enterprise agents must maintain an up-to-date understanding of work distributed across heterogeneous artifacts, evolving over time, and unevenly visible to different collaborators. We formulate this challenge as persistent project-state understanding (PPSU) and introduce PULSE, a benchmark of 1,573 questions over 30 enterprise project histories. PULSE constructs deterministic, time-indexed project state and documented collaborator views from source-linked records, and evaluates three families of questions: open loops, project evolution, and view gaps. Across retrieval agents, graph- and memory-augmented retrieval, long-context, and oracle conditions, we find substantial headroom for current systems. Retrieval and extraction account for the largest share of the gap, but providing explicit project state on top of gold evidence yields further substantial gains, showing that PPSU cannot be reduced to retrieval alone. Further analysis shows that questions requiring information to be reconciled across multiple parts of the project history remain difficult even when the relevant evidence is provided, while project-state synthesis and perspective conditioning remain persistent challenges. These results motivate enterprise agents that explicitly maintain and reconcile temporally grounded state across sources, actors, and time. 

---
# Cross-Site Transformers Are Robust Heart Rate Estimators 

**Authors**: Mason Kadem, Rong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.28366)  

**Abstract**: Single-site heart-rate estimation from photoplethysmography (PPG) degrades sharply under real-world motion. Multi-site PPG provides a natural solution, but existing models fuse sites as generic sensor channels rather than exploiting their hemodynamic structure: each heartbeat propagates along the arterial tree, appearing across body sites as delayed and transformed copies of the same cardiac event. We propose the Cross-Site Transformer, which introduces a hemodynamically informed inductive bias by restricting multi-sensor fusion to cross-attention between PPG sites. On WildPPG, the largest in-the-wild benchmark available, our model establishes a new single-model state of the art, reducing error by 21% over the previous best and 35% over the best prior transformer. The same cross-site restriction makes the trained network directly auditable. Because each cross-attention head lies on one directed pathway between sites, causal resample ablation can isolate and test each direction of information flow. Ablating the chest-to-wrist pathway doubles error, whereas the reverse costs far less. Encoding hemodynamic structure into attention routing thus improves real-world sensing robustness and enables rigorous auditing of multi-sensor fusion. 

---
# Understanding Membership Inference Attacks on EEG Models 

**Authors**: Mehrdad Khodapanahandeh, Selin Unal, Berkay Kemal Balioglu, Alireza Khodaie, Beren Semiz, M. Emre Gursoy  

**Link**: [PDF](https://arxiv.org/pdf/2611.28354)  

**Abstract**: Electroencephalography (EEG) signals contain sensitive, subject-specific information, making models trained on them potential targets for membership inference attacks (MIAs). Although MIAs have been studied extensively in other domains, their applications to EEG models are relatively scarce, and the factors driving the vulnerability of EEG models to MIAs are not adequately understood. In this paper, we present a systematic empirical analysis of MIAs on EEG models using three datasets, seven model architectures, four attack methods, and two metrics. Results show that poor model generalization (large train-test loss and accuracy gap) is the strongest indicator of MIA vulnerability. Longer training and larger models also increase vulnerability. Moreover, no particular scalp region, frequency band, inter-channel correlation pattern, or temporal versus spectral representation of EEG data consistently explains MIA vulnerability across datasets. Finally, subject identification and leave-one-subject-out experiments indicate that models relying more heavily on subject-specific representations are more vulnerable to MIAs. Overall, our findings identify the key drivers of membership leakage in EEG models and can provide guidance for developing MIA-resilient EEG systems. 

---
# Two Actions from One State: Fork Futures Teach JEPA World Models to Compare 

**Authors**: Yihao Wu, Runjia Zeng, Tongyan Wang, Renzhong Huang, Chongzhou Fang, Qifan Wang, Fan Heng, Dongfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28350)  

**Abstract**: JEPA world models are trained to predict logged transitions but used to compare counterfactual actions from one state. The logged behavior data show one action per state, and under a local linear approximation we prove the action response is unidentified along any direction the policy never varies within a state. We therefore supplement behavior data with measured outcomes of alternative actions from the same state. Our fork futures execute two actions from one restored state, and their outcome difference cancels the state term, identifying the action response on the sampled span. The resulting model, Fork-JEPA, matches predicted to measured differences, anchors the logged successor with behavior prediction, and adds a lookback head that keeps the predecessor recoverable, all at training time, leaving planning unchanged. On three visual control tasks, behavior-only training leaves the predicted action effect no better than a constant guess. Fork-JEPA makes it readable and raises planning success by up to 14\% over LeWM, its base model. Where the action response changes across hidden zones, it nearly doubles cross-zone success, and 1,024 fork pairs beat 1.2M extra behavior steps. Our project is available at this https URL. 

---
# Grammar-as-a-Draft: More Tokens per Speculative Decoding Step on Grammar-Constrained Structured Outputs 

**Authors**: John Zhang, Silky Arora  

**Link**: [PDF](https://arxiv.org/pdf/2611.28349)  

**Abstract**: Under structured outputs (JSON schemas, regexes, tool calls), speculative drafters do not see the grammar constraint and repeatedly propose invalid keys, delimiters, and indentation, while the grammar engine already knows which tokens must come next.
We present Grammar-as-a-Draft (GaaD), which treats the grammar as a second drafter. Before each verify step, the scheduler merges the drafter's proposal with grammar-forced tokens into one draft row: matching tokens stay as GRAMMAR, the proposal is cut where it leaves the forced path, and forced tokens fill the rest of a row wider than the drafter's proposal. The verify step accepts GRAMMAR drafts without computing their logits, composing with any drafter with no new kernels or compilation on the critical path. Because grammars fix text rather than tokens, committing canonical forced tokens changes the output distribution relative to grammar-masked target decoding.
We implement GaaD in vllm-torchtpu and an independent Rust serving runtime, evaluating on TPU v7x with MTP, DFlash, and n-gram drafters on a JSON character-profile benchmark. GaaD raises average acceptance length from 2.06 to 5.43 (MTP, Qwen3.5-0.8B), 1.99 to 5.31 (MTP, Qwen3.5-35B-A3B), 1.57 to 4.13 (DFlash, Qwen3-4B), and 1.30 to 3.64 (n-gram, Qwen3-4B), decoding 2.66-2.81x faster at batch size 1 and 1.14-1.70x at batch size 64. In Rust, MTP+GaaD is 2.23x faster on Qwen3.5-0.8B, 2.02x on Qwen3.5-397B-A17B-FP8, and 1.22x at 128 concurrent requests. Under mixed traffic (25-50% structured), GaaD finishes the workload 1.03-1.23x sooner (structured requests 2.2-3.2x sooner) while plain-text TPOT rises 7-42%. We also show grammar-masked greedy decoding can pick non-canonical tokenizations, so GaaD can yield different valid outputs that an fp32 reference finds equally likely. 

---
# Learning a Food World Model That Predicts the Next State from Production Data 

**Authors**: Inkyu Sa, Gaurav Kuppa, Steve Van der Hoeven, Konstantin Stulov, Rajat Bhageria, Luis Rayas  

**Link**: [PDF](https://arxiv.org/pdf/2611.28319)  

**Abstract**: Food assembly robots pick ingredients from a pan, and every pick must deliver a target weight. We learn a world model of food picking from 61,773 production picks, labeled automatically by a camera and a scale (self-supervised learning). From the pre-pick heightmap of the pan on a 2.5 mm grid, the commanded pick action, the food and the utensil, the model predicts the post-pick heightmap and the picked mass with uncertainty. We evaluate the mass prediction on 12,294 held-out picks of nine foods. The production controller keeps each pick near its commanded target. The model's root-mean-square error (RMSE) of 10.93 g is 9% lower than that of the commanded target or of a per-food mean. This RMSE is 24% lower than that of the controller's internal mass estimate. Gradient-boosted trees on 20 hand-crafted features have a 0.2 g lower RMSE, but they predict only the mass, not the next state that planning and policy learning need. Using the predicted uncertainty, we decline the most uncertain picks and answer 89.4% of picks with an RMSE of 10.28 g. The prediction barely responds to the commanded volume and depth, because they vary little in closed-loop production. The model also fails without warning at target weights not seen in training. Future directions include action-primitive conditioning and a more general model across foods and utensils. We give recommendations for learning world models from production data. 

---
# Language Model Harnesses can be Compositional Generalizers 

**Authors**: Alex L. Zhang, Souradip Chakraborty, Omar Khattab  

**Link**: [PDF](https://arxiv.org/pdf/2611.28298)  

**Abstract**: Modern language models (LMs) are post-trained and deployed as agents inside a "harness", but standard harness designs accumulate context that the neural network is forced to generalize to, making training for long-horizon tasks expensive and time-consuming. We study an alternative emerging path: training inherently neurosymbolic LM harnesses, which externalize and manage context programmatically to compose existing LM capabilities. We hypothesize that training around these harnesses can offer a better return on compute by enabling compositional generalization, as they can learn to decompose complex problems so that every observation their neural LM receives is locally in-distribution, even when the full task is novel. We find that harnesses like the Recursive Language Model (RLM) learn such decompositions readily under reinforcement learning post-training: (1) we demonstrate that training only on short tasks can generalize to tasks 2-64$\times$ longer while the base neural LM improves less, and (2) training on one domain transfers to novel domains that share latent structure. On messier, complex tasks, we find that we can target individual LLM calls within a harness by generating and training on easier, synthetically generated surrogate environments with fake tool / sub-calls, a technique we call surrogate harness training. From this finding, we train RLM-27B on synthetic data, and show it improves Qwen3.8-27B in the same RLM harness on a hard suite of long-context tasks like CTC-Bench and ExpertLongBench by 26.9% on average while using $\sim$2$\times$ fewer tokens at inference time. Finally, we show that harnesses can compose over these newly learned decompositions using recursive sub-harness calls. 

---
# MRI-Driven Dual-Energy CT Synthesis via Conditional Denoising Diffusion Probabilistic Models for MRI-only Radiotherapy and Protontherapy 

**Authors**: Juan Manuel Molina-Maza Chih-Wei Chang, David Viar-Hernandez, Zach Eidex, Mojtaba Safari, Shaoyan Pan, Juan Maria Perez-Moreno, Norberto Malpica, Angel Torrado-Carvajal, Borja Rodriguez-Vila, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28294)  

**Abstract**: Magnetic resonance imaging (MRI) provides excellent soft-tissue contrast without ionizing radiation but lacks the electron-density information needed for radiotherapy dose calculation. Dual-energy computed tomography (DECT) improves tissue characterization and stopping-power ratio (SPR) estimation for proton therapy, but requires additional radiation exposure. We developed a denoising diffusion probabilistic model (DDPM) to synthesize high- and low-energy CT (sHECT and sLECT) simultaneously from a single MRI scan. Retrospective paired MRI-DECT data from patients receiving head-and-neck proton therapy were used to train and evaluate a multi-decoder DDPM with SwinViT or ResUNet denoisers. Performance was compared with Pix2PixGAN and CycleGAN using image-quality metrics, Hounsfield unit (HU) errors, anatomical profiles, and SPR maps and uncertainties. The DDPM outperformed both generative adversarial network baselines in visual fidelity and quantitative image quality. For sHECT and sLECT, respectively, mean absolute errors were 36.29 +/- 5.25 and 43.72 +/- 6.77 HU; peak signal-to-noise ratios were 27.44 +/- 1.06 and 26.26 +/- 0.98 dB; structural similarity indices were 0.89 +/- 0.02 and 0.87 +/- 0.03; and normalized cross-correlations were 0.95 +/- 0.01 and 0.94 +/- 0.01. The synthesized images preserved anatomical boundaries, while uncertainties in derived SPR maps remained within the expected acceptable range. MRI-based DECT synthesis using DDPM may provide tissue-specific information for proton therapy planning while reducing the need for radiation-based DECT acquisition. 

---
# When Video Verifier Scores Mislead: Capability Assessment and Verifier Selection 

**Authors**: Charles Weng, Zixi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28287)  

**Abstract**: Video verifiers judge whether a candidate answer or trajectory agrees with a video, and are themselves chosen by a score. We show that this score depends on two choices in the test, the type of negative and the judgment interface, strongly enough to reverse conclusions about what a model can do and which verifier to deploy. We separate the two with controlled constructions that have exact truth conditions: endpoint-matched negatives, wrong only in their internal states, and a truth-switch test, in which the same two candidates swap correctness between two videos. Under direct judgment, where the verdict is read from answer scores, models that reject a visibly wrong start fall to near chance against endpoint-matched negatives. When the frames disclose each event, writing out reasoning before the answer lifts three of four models from under 10% to 70-100% on the truth-switch test, and staged inputs trace the direct failure to one step, composing events into states. But when the event must first be found, on human-validated natural video, the same change brings no significant gain under a protocol common to all models and can lower accuracy if one candidate denies that the event occurs. The negative type also changes which verifier is selected: one chosen on event-denial negatives is 8.9 points less accurate against negatives that keep the event and change a detail, the errors a generator mostly makes, and recovers about half as much accuracy when it reranks that generator's answers. A verifier should be evaluated on the errors it will face, and neither a low direct score nor a gain from reasoning should be read as a property of the model alone. 

---
# HandXFM: Semantic-Structural Distillation for Hand Radiograph Foundation Models 

**Authors**: Yuxi Long, Ganlin Feng, Lianghong Chen, Liam J. O'Neil, Carol A. Hitchon, Pingzhao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28283)  

**Abstract**: Hand radiographs support diverse clinical tasks, including skeletal maturity assessment, rheumatoid arthritis scoring, abnormality screening, and anatomical localization, yet existing models are typically trained separately for each task and dataset. We present HandXFM, a domain-specific foundation model that leverages cross-modality knowledge for comprehensive hand X-ray understanding. HandXFM distills semantic priors from BiomedCLIP and structural cues from a chest X-ray Vision Transformer (ViT) through a unified alignment framework, integrating complementary medical knowledge in an interpretable way. After pretraining, HandXFM can be adapted to a range of musculoskeletal tasks, including bone age estimation, SvH score prediction, abnormality detection, joint localization, bone segmentation, and visual question answering. It consistently outperforms task-specific and single-expert baselines, achieving up to 0.11 improvement in Pearson correlation coefficient (PCC) for bone age prediction and a 12\% accuracy gain in abnormality classification. Moreover, HandXFM generalizes to unseen datasets and yields cross-attention maps linking clinical terms to anatomical regions. These results demonstrate that multi-expert distillation effectively unifies semantic and structural supervision within a single pretrained model, establishing HandXFM as an interpretable and generalizable foundation model for hand radiograph analysis. 

---
# Learning a Rule Bank to Revise 3D CT Reports with Independent Evidence 

**Authors**: Renjie Liang, Mengxian Lyu, Zijian Xu, Boxin Zhao, Yang Yang, Jinqian Pan, Zhengkang Fan, Mei Liu, Yonghui Wu, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28281)  

**Abstract**: Vision--language models can generate fluent radiology reports from 3D chest CT scans, yet they may still omit abnormalities or misstate findings. Independent image analysis using models separate from the report model provides additional observations, but acting on uncertain evidence may introduce new errors. We propose an agent system that learns a revision policy represented as an executable rule bank for each report model. Inspired by experience banks, the agent proposes and refines rules using revision outcomes across development cases. During inference, the retained rules guide local edits using independent classifier predictions and quantitative image measurements without retraining the report model. Unedited sentences are preserved, and each change is linked to the rule that triggered it. On CT-RATE, revision increases CE-F1 by 11.5 points for SelfCue-KD and 0.9 points for NV-Reason-CT, with gains across five draft-sampling seeds. SelfCue-KD's CT-FineBench score also increases by 15.2 points. Draft-only rule learning yields limited gains, while the two image-evidence sources provide complementary benefits for abnormality detection. Measurements also supply numerical details missing from the drafts. These results support learning model-specific rules to control how independent image evidence informs report revision. Code is available at this https URL. 

---
# A Closed-Form Temporal Readout for Surgical Triplet Recognition 

**Authors**: Lemar Abdi, Amaan Valiuddin, Fida Mohammad Thoker, Karen Sanchez, Bernard Ghanem, Fons van der Sommen  

**Link**: [PDF](https://arxiv.org/pdf/2611.28240)  

**Abstract**: Surgical triplet recognition requires identifying which instrument performs which action on which anatomical target, and following these interactions as they unfold over time. The task is challenging because several triplets can occur at once, their components span very different durations, and the evidence for an interaction is rarely complete in a single frame. We introduce STRIDE, a post-hoc temporal readout that adds temporal context to any trained recogniser. It averages the recogniser's features and logits over a few fixed temporal windows and combines them with the current frame through ridge regression, solved in closed form on a CPU. Applied as a plug-and-play readout to nine existing methods, STRIDE improves $AP_{IVT}$ in 24 of 27 method-dataset pairs, by up to 7.4 points. Moreover, we propose a simple pipeline consisting of an adapted foundation vision encoder with a linear classifier and STRIDE. Across thirteen encoders, STRIDE improves 37 of 39 encoder-dataset pairs by up to 7.2 points. STRIDE reduces false alarms by 41%, at the cost of a slower response at interaction boundaries. Code will be made publicly available upon acceptance. 

---
# Evaluating Memanto on DolphinBench for AI Agents 

**Authors**: Suleman Khan, Rasa Rahnema, Tara Khani, Majid Fekri, Fattane Zarrinkalam  

**Link**: [PDF](https://arxiv.org/pdf/2611.28231)  

**Abstract**: Memory benchmarks built on question answering tell an agent which fact to recall. Workplace requests often leave that need implicit: to post an update in the team's channel, an agent must work out which channel from earlier interactions and use it in a tool call. We evaluate Memanto, a memory agent that ranks memories with an information-theoretic relevance score, on DolphinBench, which tests memory through 600 simulated workplace tasks, each validated as requiring memory. Running in the Hermes agent harness with GPT-6-Luna and at most 6 iterations per conversation, Memanto completes 503 tasks (83.83%). The strongest configuration reported by the benchmark's authors, Mem0 with GPT-5.6-Luna and up to 60 iterations, completes 424 (70.67%). Memanto's run also has the lowest total cost ($10.86, against $61.48 for the cheapest reported run) and the shortest median task latency (15.32s). Grading uses the benchmark's own judge, but the agent model differs, so these results compare complete systems. The memory system affects both whether an agent completes such tasks and what completion costs. 

---
# Video-as-Weights: Compiling a Video Once into Parametric and Token Memory 

**Authors**: Wentao Zhang, Ian Miao, Chen Qu, Bill Bi, Donna Schut, Yuntian Deng, Yale Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.28221)  

**Abstract**: Vision-language models (VLMs) represent videos as thousands of visual tokens, creating a large key-value cache whose memory grows with video length. At the same time, small VLMs that fit on limited hardware understand video far worse than large models. Existing compression methods shorten visual tokens into a brief prompt to save memory, yet the small model still processes them through fixed weights and cannot close this capability gap.
To address both challenges, we introduce Video-as-Weights (VaW), which compiles a video once without seeing any question into the parametric and token memory of a frozen interpreter VLM. Specifically, a compiler VLM watches the video, and a single weight decoder jointly generates the interpreter's LoRA weights and optional prefix tokens from every compiler layer, enabling both same-model compilation and strong-to-weak transfer across model sizes and depths.
Across six video QA benchmarks, a 0.8B interpreter beats full in-context reading by 1.7 points when compiled by itself and by 4.3 points with a 9B compiler, reducing context from up to 8,192 visual tokens to 64 prefix tokens while using 5.5x less memory than the video's key-value cache. Generated weights add 3.9 points over token compression alone, and all-layer decoding outperforms layer-wise generation by 2.9 points. Mechanistic analysis shows that with prefix tokens, the weights mainly alter attention queries and MLP down projections to guide retrieval and store a baseline summary, whereas without tokens, they shift to attention output projections to write video content directly into the residual stream. 

---
# HEP-Ex-Bench: Evaluating Agentic Performance in High Energy Physics Experiment Data Analyses with Sealed Worlds 

**Authors**: Andrzej Novák, Yiyang Zhao, Dolores Garcia, Eric A. Moreno, Jeffrey Krupa, Víctor Bresó-Pla, Samuel Bright-Thonney, Lukas Heinrich, Michael Kagan, Philip Harris  

**Link**: [PDF](https://arxiv.org/pdf/2611.28194)  

**Abstract**: Agentic systems are increasingly being explored for automating experimental high-energy physics measurements, but the correctness of their results can today be reliably evaluated only via extensive expert review. To address this, we propose a method to perform detailed quantitative evaluations of agent-driven analyses through the use of blind analysis and injection tests, which are critical in a typical expert review. Agents develop a data analysis pipeline on a public dataset, while the evaluation is performed on programmatically generated sealed private datasets where signals are injected and simulation settings varied within realistic detector uncertainties. By requiring agents to provide a single workflow that is evaluated across these "sealed worlds", we are able to test the accuracy and robustness of the workflows. We demonstrate the approach on four tasks built from CMS open data and present it as a way to reliably benchmark scientific work in experimental high-energy physics. At the time of writing, the best-performing model, Gemini 4 Argon, reaches 60% of the score of the expert reference analyses on average, ahead of GPT 6 Astra at 38% and Claude Opus 5.5 at 31%. In individual sessions on one of the tasks, Gemini 4 Argon exceeds the reference. The benchmark is hosted at this https URL 

---
# FastD4RT: Training-Free Acceleration of Dynamic 4D Reconstruction and Tracking 

**Authors**: Meng Bi, Yili Jin, Kaiyuan Hu, Xize Duan, Jiaxin Huang, Hong Huang, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28193)  

**Abstract**: Query-based 4D reconstruction models such as D4RT encode a video once into a global scene representation and obtain outputs by decoding independent point queries. However, inference is often influenced by dense outputs and long videos. Although D4RT separates encoding from decoding, a dense output still requires one decoder query for every pixel of every frame, and a long video is still encoded as many overlapping segments. In this work, we analyze the inference cost of D4RT and show that a large part of it is redundant. In particular, we find that the later decoder blocks attend to a small set of tokens shared by all queries. Based on these findings, we propose FastD4RT, a training-free framework that accelerates D4RT. By exploiting the smoothness of the outputs over time and across pixels, FastD4RT encodes and decodes only at sparse anchors and recovers the remaining outputs via interpolation. It decodes the remaining queries against a shared token set selected once per segment. We further characterize the errors introduced by both components. Experimental results show that FastD4RT accelerates D4RT by about 5 times on average across all its tasks with competitive accuracy, and further improves accuracy on long videos. Our code is available at this https URL. 

---
# Don't Test Where You Trained: Geographic Overlap Inflates End-to-End Driving Scores 

**Authors**: Willi Poh, Kevin Rösch, Christoph Stiller  

**Link**: [PDF](https://arxiv.org/pdf/2611.28177)  

**Abstract**: In NAVSIMv1, the standard benchmark for end-to-end driving, $89.6\%$ of the test scenes lie within 4m of a training scene. We show that this geographic overlap inflates the reported scores. We propose a geographically disjoint split of NAVSIMv1 and an evaluation that compares models trained on either split on the same scenes, which neither has seen in training. All sensor-based models we test, DiffusionDrive, Latent TransFuser, and DrivoR, lose 3.4 to 6.1 PDMS points on these scenes, mainly by leaving the road and coming close to other objects, against 1.5 points for a baseline without sensors. The loss concentrates where a training drive of the original split passed the same place. At all other places, models trained on the original split are no better than those trained on the geographic split. Models trained on the original split predict road geometry that no camera can see. The gap persists when the auxiliary map loss is removed, so the models memorize locations even without being supervised to. On nuScenes, the geographic split cuts SparseDrive's mapping accuracy by $59\%$ and also degrades its planning. Driving scores on logged scenes thus partly measure how well a model remembers the test locations. We release the split to evaluate generalization to new places. 

---
# A world model for longitudinal health-trajectory prediction in older adults across multinational cohorts 

**Authors**: Jingyi Wu, Chenghua Guo, Qianlin Zuo, Pengfei Li, Luxia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28175)  

**Abstract**: Aging is characterized by continuously evolving health states, yet most existing prediction models focus on static health risk at isolated time points rather than explicitly modelling how health trajectories unfold over time. We developed a health world model based on the Joint-Embedding Predictive Architecture to represent evolving health states and simulate future trajectories under specified actions. The model combines a state encoder, an action- and context-conditioned predictor, a stop-gradient target encoder, and clinical outcome heads. We trained the model on the US Health and Retirement Study (HRS) and evaluated it without refitting in the English Longitudinal Study of Ageing (ELSA) and the China Health and Retirement Longitudinal Study (CHARLS). Open-loop evaluation assessed discrimination and calibration for mortality and worsening of activities of daily living (ADL) and instrumental activities of daily living (IADL). In the held-out HRS test set, the areas under the receiver operating characteristic curve (AUROCs) for mortality, ADL worsening, and IADL worsening were 0.89, 0.93, and 0.93 at the first horizon and 0.84, 0.84, and 0.85 at the fifth horizon, respectively. Predicted risks stratified subsequent outcomes across horizons. Across the evaluated horizons, mortality AUROC ranged from 0.867 to 0.881 in ELSA and from 0.796 to 0.801 in CHARLS. Simulated trajectories preserved the expected risk ordering across predefined action regimes. In counterfactual simulations, changes in lifestyle actions shifted predicted risks in directions consistent with observed associations. The model provides a framework for longitudinal risk prediction and action-conditioned simulation in older adults. 

---
# Selective Verification in Multi-Agent Orchestration with Large Language Models 

**Authors**: Ramesh Krishnamaneni, Muralidhar Kurni  

**Link**: [PDF](https://arxiv.org/pdf/2611.28173)  

**Abstract**: Agents can agree because they relied on the same flawed evidence. We use an exact model of nine binary votes to examine when a verifier should intervene. In the main comparison, workers err 20 percent of the time under ordinary conditions, shared failures occur in 10 percent of cases, and the verifier errs 2 percent of the time in either state. When worker error during a shared failure is 80 percent, checking close votes yields 7.93 percent verdict error, compared with 9.06 percent for random checks at the same expected check rate. The ordering reverses at 90 percent worker error during failure. At the matched check rate for that regime, checking close votes yields 9.98 percent error, random checks yield 9.41 percent, and ranking checks by their expected error reduction yields 6.73 percent. The model also shows that passing branch vote counts preserves the flat majority and reduces the number of messages the root receives. A design guide and a documentation audit of four Python frameworks translate these findings into implementation choices. 

---
# How Far Can LLMs Go in Generating Android Lint Checks from Natural Language? 

**Authors**: Moshood A. Fakorede, A.B. Siddique, Manu Sridharan, Umar Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2611.28159)  

**Abstract**: Static analysis tools such as Android Lint, PMD, and CodeQL let developers add custom checks for project-specific rules. However, authoring a check is a demanding programming task: a rule can be stated in a few sentences, yet turning it into a working, tested check requires detailed knowledge of the analysis framework and its issue-detection logic. Recent work generates such checks with large language models (LLMs), but each system needs more than the natural language description a developer always has: a bug-fix patch, a CVE record, or a developer-written test suite to repair against. No prior work measures whether an LLM can turn a description into a working check, or what external knowledge it needs to do so. We introduce LintBench, a human-validated benchmark of 113 real-world Android Lint checks, each paired with a natural language description and its original test suite as a held-out oracle. We evaluate four LLMs, including proprietary and open-weight models. We compare prompting strategies, then evaluate execution-guided repair with and without retrieved API knowledge. Prompting alone reaches 30%-65% (pass@5) across models. Execution-guided repair and retrieval reach 57%-85% test-passing rates within five calls. Our results show that execution feedback raises the pass rate, while retrieved framework knowledge raises the compile rate, which reaches up to 99%. The framework knowledge helps the low-performing models the most, providing the knowledge they need to fix compilation errors. Of the generated checks that compile but fail their tests, 87% have false negatives, missing an issue the generated check should report. Based on our findings, getting the issue-detection logic right remains an open challenge. We release LintBench and the evaluation harness to support future work on generating Android Lint checks. 

---
# Measuring and Teaching Language Model Agents to Act on Delayed Consequences 

**Authors**: Shardul Bansal, Seth Schilbe, Jarrod Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2611.28158)  

**Abstract**: Language model agents are increasingly asked to run ongoing multi-day operations rather than finish isolated single-turn tasks. In daily operations, today's choices reshape later observations and cash constraints because empty shelves hide lost walk-in demand, short-shipped customers quietly shrink future orders weeks later, and emergency buying drains working capital. We introduce SMB Bench, a calibrated 90-day wholesale simulator with 18 causally paired tasks and an 859-run frontier trace dataset, to ask what experience and learning signal make an agent better at noticing, attributing, and acting on the delayed consequences of its own decisions in unseen businesses. Across 16 frontier models, only 3 outperform a fixed buying rule; models often record a relevant diagnosis in their notes without changing later purchases in time, showing that diagnosing a problem, changing behavior, and improving a business are separate achievements. To study how experience improves those failures in an open model (Qwen3.8-27B), we used training-business records to identify operational rules and archived frontier quarters as starting states to continue from rather than transcripts to copy. On an unseen test business under identical evaluation instructions, the trained policy followed customer-protection commitments more consistently and reduced severe-loss quarters, while the additional profit benefit remained uncertain. Carrying those commitments through the quarter exposed the cost of fulfilling them, as early-quarter stocking habits left the buyer relying on emergency supply and a deteriorating supplier. The results separate recognizing a delayed effect, acting on it consistently, and coordinating surrounding decisions so that the response improves profit. SMB Bench connects these observations to a learning method by turning archived decisions and their consequences into reusable training experience. 

---
# Beyond Fidelity: Evaluating Synthetic Medical Imaging Data 

**Authors**: Seyed Kahaki, Alexander R. Webber, Tahsin Rahman, Elim Thompson, Jana Delfino, Aldo Badano  

**Link**: [PDF](https://arxiv.org/pdf/2611.28154)  

**Abstract**: The growing use of synthetic medical imaging datasets (SMD) in the development and evaluation of machine learning models has motivated interest in systematic methods to assess dataset quality across different imaging modalities. Common evaluation strategies rely heavily on fidelity-oriented metrics developed for general computer vision and may not capture clinically relevant properties such as anatomical plausibility, dataset coverage, metadata completeness, subgroup consistency, or documentation quality. We present the ScoreCard Toolkit that operationalizes six of the seven dimensions found within the 7Cs approach for SMD evaluation across Congruence, Coverage, Completeness, Consistency, Constraint, and Comprehension. The ScoreCard Toolkit provides a modular pipeline for preprocessing, feature extraction, metric computation, visualization, and structured scorecard generation. This approach complements standard fidelity-based evaluation by exposing quality characteristics not captured by those measures alone, including constraint violations, anatomically implausible shape artifacts, and metadata-related consistency issues. The toolkit supports appearance, texture, and shape features and is designed to accommodate additional feature representations, including radiomic features, deep-learning embeddings, and foundation-model representations. We demonstrate the end-to-end workflow across three imaging modalities: synthetic digital pathology whole-slide image patch data, synthetic digital mammography, and synthetic head CT. Across all three modalities, the resulting scorecards reveal dataset-specific quality characteristics, such as distributional biases and coverage gaps that may impact their effectiveness for model development or testing. ScoreCard provides a practical, extensible, and modality-agnostic approach for evaluating and transparently reporting SMD quality. 

---
# From Observation to Explanation: Mechanistic Interpretability of LLM-Generated Code for Principled Repair 

**Authors**: Musfiqur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2611.28153)  

**Abstract**: Large language models (LLMs) have transformed software development, yet our understanding of how they generate code remains critically underdeveloped. We address this gap by triangulating evidence across three layers. Longitudinally, autonomous agent-attributed code is statistically indistinguishable from human code under a semantic modification definition (HR = 1.019, p = 0.490), yet attracts disproportionately corrective modifications in production (49.1% vs 37.9% human baseline). This finding demands a causal explanation. Empirically, existing benchmarks overestimate LLM capabilities: models that achieve 84 to 89% correctness on synthetic tasks attain only 24 to 35% on real-world class-level code. Behaviourally, detection systems reveal puzzling asymmetries: GPT-3.5 is far more detectable than modern models (AUC-ROC = 0.96, a 27-point gap), even though modern models produce mutually similar code structures; yet no single feature reliably distinguishes AI-generated from human code at the class level, and the leading commercial detector achieves only 15 to 64% recall. Together, these findings raise a question that empirical studies cannot answer: why do LLMs generate code with these properties? This thesis proposes mechanistic interpretability, a causal token-level circuit tracing, as the bridge between empirical observation and principled correction. By identifying the transformer circuits responsible for detectable stylistic signatures, hallucination patterns, and systematic defects, we aim to drive an agentic repair framework that intervenes at the causal source of defects rather than patching surface symptoms. 

---
# STEGOGUARD-LM: Evaluating End-to-End Protection Against LLM Covert Channels 

**Authors**: Martha Dimgba, George Williams  

**Link**: [PDF](https://arxiv.org/pdf/2611.28152)  

**Abstract**: A defense that disrupts hidden messages may still provide little protection if its detector rarely triggers an intervention. We present StegoGuard-LM, a controlled framework for evaluating defenses against covert messages in large language model outputs. We examine token-rank and synonym channels, with Unicode concealment as a structural control, and track messages through detection, intervention, and recovery. Deterministic rewriting prevented full recovery of 103 of 104 synonym-encoded messages when applied unconditionally, but only one when intervention depended on a detector set to limit false alarms. In a four-round replay on a 48-response subset, an attacker selecting from a fixed catalog recovered more bits by switching channels, although paired gains were not statistically significant after correction for multiple comparisons. Randomized rewriting showed no sustained advantage in this pilot. A more sensitive detector improved protection but modified about one-third of clean responses. These findings show how uneven detection can undermine effective rewriting and why protection must be evaluated alongside attacker adaptation and intervention on legitimate text. 

---
# Close the Loop on the Retrievability Gap 

**Authors**: Guangyuan Dong, Wenhao Li, Zhen Tian, Jiaqi Zhang, Haoyu Wang, Junhao Xu, Mingzhe Liu, Pan Wang, Fei Shen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2611.28146)  

**Abstract**: A frozen vision-language encoder now guides image restoration and later indexes what it produced. Over a fixed set of eighteen pansharpening systems on one WorldView-III test set, the fidelity index that ranks them, $Q2^n$ with a reference and HQNR without one, picks the less retrievable of two systems one time in three and one in two; the disagreement lies between the pixel-supervised and the language-conditioned family, and even the best system falls short of the reference, small objects first. We call this the retrievability gap and trace it to the area weighting of position-averaged losses. LEDGER closes the loop around the encoder: it indexes the input into a set of named concepts, generates, re-encodes the output alone with the same frozen encoder, and minimises a set-level soft-Jaccard loss on the coverage of that set, a label-free target available at test time, so that every present concept of a scale receives a gradient share independent of its area. Measured outside the loop by label-grounded probes, held-out retrievers, language-free evaluators, detectors and human raters, LEDGER matches or exceeds the strongest baselines in fidelity on four sensors, raises retrieval by seven to nine points within its vocabulary and two to five outside it, closes two thirds of the reduced-resolution gap to the reference, lifts extraction on raw SpaceNet imagery above the vendor product in each of four cities, and transfers to existing generators as a training-only loss at no inference cost. 

---
# Choose What Returns: Object-Aware KV Memory for Selective Recall in Interactive Video Generation 

**Authors**: Hyunwook Kang, Junoh Park, Hyunggyu Lim, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.28137)  

**Abstract**: Interactive long-video generation lets users build a story through successive prompts and reuse earlier characters and settings under new instructions. Existing memory methods retain past content, yet they often fail at precise selective recall, returning the wrong appearance of a person, mixing different people, or keeping an unrequested background. Based on our analysis of memory form and retrieval, we introduce an object-aware key--value (KV) memory framework for a frozen autoregressive diffusion generator. It keeps the original KV of tracked object regions, selects background KV outside them by attention-scores, and links each memory with a readable record of its source scene and observed appearance. An LLM selects the requested records, and a memory integrator supplies their KV for generation under the current prompt. We also introduce a six-scene evaluation that tests selective attribute modification, time-specific appearance recall, and multi-object composition, in which each person's requested past appearance changes across requests. Experiments show that our framework more reliably preserves each person's identity while reproducing their requested past appearance than existing memory methods, under both explicit scene references and natural-language cues. 

---
# Beyond Output Suppression: Mitigating Privacy Risks in LVLMs via Visual Intrinsic Privacy Erasure 

**Authors**: Xiangkui Cao, Jie Zhang, Meina Kan, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.28135)  

**Abstract**: Large vision-language models can extract personally identifiable information from images, making privacy protection essential for their deployment in perception-oriented tasks. Existing targeted fine-tuning and model-editing methods usually suppress sensitive answers by aligning the model toward safe responses. In this paper, we show that such protection can be illusory: although the fine-tuned model refuses to disclose private information under normal decoding, privacy leakage still occurs in sentence completion scenarios. This indicates that existing algorithms only change the model's output distribution, but the model still retains the ability to perceive private information. Private information therefore still remains inside the model. To address this problem, we propose Visual Intrinsic Privacy Erasure (VIPE), which targets privacy-related visual representations to weaken the model's ability to perceive sensitive information. VIPE constructs paired privacy-sensitive and privacy-masked images, uses their internal feature residuals to identify privacy-related visual components, and suppresses these components before they are perceived by the model. To preserve benign visual representations, the erasure objective projects feature differences onto a relaxed benign subspace and adaptively controls the erasure strength. VIPE compels the model to ignore the extraction of sensitive information in the visual representation space, thereby blocking the source of privacy leakage more fundamentally rather than merely altering the model's output distribution. Experiments on Qwen2.5-VL and LLaVA-OneVision show that VIPE substantially alleviates prefix-triggered privacy leakage while maintaining competitive utility on general visual-language benchmarks. 

---
# SceneTether: Activity-Gated Value Anchoring for Long-Horizon Autoregressive Image Animation 

**Authors**: Matiur Rahman Minar, Seunghun Oh, Ganghyeon Jeong, Unsang Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.28134)  

**Abstract**: Long-horizon image animation must keep the scene fixed while natural material (e.g., water, smoke, or fire) continues to move. Autoregressive video generators read recent frames for temporal continuity, but this history also carries accumulated appearance errors, and sparsifying it disrupts motion. We introduce SceneTether, a training-free attention-read policy that keeps where attention reads recent memory but changes what appearance that memory returns. SceneTether keeps every key of the newest latent, blends its low-activity values toward co-located source-image values, and leaves high-activity values untouched so motion can evolve; an activity map estimated from the first generated latents and then frozen decides where to blend. It adds no denoising evaluations and leaves the cache-update rule unchanged. On a Causal Forcing++ generator, SceneTether raises source-support PSNR by 5.15 dB over its framewise Echo host across two seeds with 1.5x its late activity, and by 7.31 dB over spatial Echo-Forcing on a preselected four-minute subset. At equal recent-memory coverage, value anchoring alone adds 8.14 dB. A post-hoc pixel composite with the same map matches static fidelity but retains nearly twice the moving-region color drift, and viewers in a small blinded study preferred SceneTether for its natural scene and motion preservation. Project page: this https URL 

---
# Urban State Representation for Autonomous Mobility: A Survey of Multimodal Perception, Architectures, Datasets, and Benchmarks 

**Authors**: Binay Kumar Singh, Niels Da Vitoria Lobo  

**Link**: [PDF](https://arxiv.org/pdf/2611.28129)  

**Abstract**: Cities are sensed by many platforms at once: on-board cameras, LiDAR and radar on autonomous vehicles, roadside units at intersections, drones and satellites overhead, and the GNSS traces of millions of trips. Yet most learned representations are built for one platform, one city, and one task. We survey this fragmented landscape through a single lens, the Urban State Representation (USR): a georeferenced, layered, time-varying, uncertainty-aware state that any sensing platform can write to and any downstream task can read from. We (i) formalize USR and five desiderata for it, and relate it to bird's-eye-view (BEV) features, occupancy, vectorized maps, neural fields and 3D Gaussians, scene graphs, road-network graphs, language tokens, and world-model latents; (ii) organize sensing platforms, cross-view (ground-aerial) matching, fusion operators, 3D geometric foundation models, end-to-end and vision-language-action models, and generative world models and data engines in one taxonomy that spans sensors to decisions; (iii) compile 26 urban datasets and simulators across ego, cooperative (V2X), aerial, trajectory, and language settings, with a metric glossary, a cross-benchmark results snapshot, a trend analysis, and a platform-coverage gap analysis; and (iv) review applications and propose, as an outlook, USR-5, a five-test protocol with explicit references and one aggregate score for measuring whether a state is shared, together with a research agenda of six open problems covering cross-city and cross-sensor transfer, uncertainty, efficiency, sim-to-real evaluation, and governance. This survey is an entry point for computer vision, autonomous driving, transportation, and urban computing researchers; the paper list, datasets, and USR-5 code are at this https URL. 

---
# Seeing the Invisible: How Vision-Language Models Read, Predict and Trace Radio Fields 

**Authors**: Yiyang Zhu, Xin Li, Guangyuan Liu, Ziheng Liu, Jiayi Zhang, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2611.28123)  

**Abstract**: Operating in the physical world requires reasoning beyond what cameras directly capture. In a city, radio signals are invisible to the lens, yet their strength is governed by the buildings and streets an image does show. Given only these visible structures, can vision-language models (VLMs) infer the invisible radio field they shape? Probing true physical reasoning requires exact fields in real cities, alongside controlled interventions where a single physical factor changes. We build a label-free engine that turns any coordinate on Earth into a ray-traceable city, simulates matched single-factor counterfactuals, and computes every answer exactly. This yields RadioVISTA, a benchmark and training dataset for visual reasoning about invisible fields. Physically, a radio field is the sum of its propagation paths, yet across six continents we uncover a striking field-path dissociation: frontier VLMs accurately judge where a field is strong, but fail to tell whether a single direct or one-wall path exists. Open models fail even earlier at reading the field itself, and are systematically wrong, scoring below random guessing even when credited for any of four attempts. On unseen cities, fine-tuning brings a 9B open model close to frontier proprietary models on fields, yet neither supervised nor reinforcement fine-tuning improves path tracing. Supervising only final answers teaches models to approximate the field, yet leaves single-path geometry out of reach. 

---
# Seeing Beyond Pixels: Radar Feature Generation from Images and Sparse Returns 

**Authors**: Hongyu Zhou, Yubo Peng, Luping Xiang, Kun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28119)  

**Abstract**: Camera-radar fusion benefits from complementary visual and geometric information but often relies on complete radar representations, leading to substantial computing overhead. However, the extent to which visual information can compensate for limited radar measurements remains insufficiently explored. Therefore, we propose Vision-informed Radar Token Generation (VRTG), which exploits cross-modal information to generate task-relevant radar features from camera images and sparse radar measurements. Specifically, VRTG first leverages image-derived object cues to select a limited number of informative radar points. It then establishes calibrated one-to-one associations between visual objects and radar returns, using matched measurements to refine object ranges while retaining visual estimates for unmatched objects. Finally, the associated object features and visual context are transformed into radar tokens compatible with existing camera-radar fusion networks. Experiments on RADIal demonstrate that VRTG achieves 90.76\% vehicle-detection F1 and 86.68\% free-space mIoU using only four radar points per frame, approaching the performance of the full-radar model. 

---
# Green ACO: An Energy-Aware Ant Colony Optimisation for the MAX-SAT Problem with EI/J Scheduling and Thompson Sampling 

**Authors**: Maroua Ogab, Nazim Abdeerhmane Aouni, Rayane Rahmat Errahmane Smara, Nour El Imane Elbar, Zakaria Soualah Mohammed, Malika Bessedik  

**Link**: [PDF](https://arxiv.org/pdf/2611.28109)  

**Abstract**: NP-hard combinatorial optimization problems demand heavy computation, and this raises real environmental concerns, especially in regions with carbon-intensive power grids. This paper presents Green ACO, an energy-aware Ant Colony Optimisation algorithm for the MAXIMUM SATISFIABILITY (MAX-SAT) problem. At its core is an Expected Improvement per Joule (EI/J) scheduler. At every iteration, it picks the operator that best balances expected quality gain against expected energy cost. Three operators feed the scheduler: WalkSAT, Focused Variable Neighbourhood Search, and Clause-Restart Greedy, ranging from cheap and surgical to aggressive and costly. An offline dual-regime profiling phase calibrates these operators before the search even begins, and, combined with Thompson sampling, this reduces uninformed early operator choices; online call-count warm-up still occurs. A pre-dispatch guard rejects calls when their estimated energy cost, with a safety multiplier, exceeds the remaining budget. Sparse pheromone updates, pheromone caching across budgets, and stagnation-based early stopping trim energy use further, without hurting solution quality. The framework also incorporates a budget predictor that estimates the energy budget required for unseen instances. On four hard MAX-SAT reference instances, Green ACO reaches an average quality of 98.44%. This beats the standard ACO baselines by close to 3 points, while using only 986 J against 4,171 to 6,610 J for the strongest competitors, a six-and-a-half-fold gain in quality per joule. Under the Algerian grid's carbon intensity, this cuts CO$_2$e emissions by approximately 85% compared with the most energy-hungry baseline. 

---
# Intervene or Trust CLIP? Competence-Gated Adaptation for Source-Free Cross-Domain Few-Shot Learning 

**Authors**: Haoyue Liu, Chenhao Zhu, Yu Huo, Haoran Shou, Shuguang Cui, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28104)  

**Abstract**: Source-free cross-domain few-shot learning (SF-CDFSL) adapts a pre-trained vision-language model such as CLIP to a distant target domain from a handful of labeled images, without access to source data. Recent methods improve the fine-tuning of CLIP with token-level interventions and apply them in the same way to every target domain. In this paper, we find that whether an intervention helps depends on how well frozen CLIP already recognizes the target domain: in 5-way 1-shot tasks, intervening improves accuracy by 0.72 points where frozen CLIP misclassifies most support images, but lowers it by 1.31 points where frozen CLIP already classifies them well, and the support set of a single episode suffices to tell the two cases apart; a bias-variance analysis explains both. Based on this finding, we propose Competence-Gated Adaptation (CoGA), which measures frozen CLIP's accuracy and margin on the labeled support set and decides, before training, whether to intervene on the visual tokens or to trust CLIP by fusing the adapted prediction with the frozen one. CoGA adds no trainable parameters and uses no query labels. On the four BSCD-FSL target domains, CoGA achieves the best average accuracy, 59.56% (1-shot) and 68.96% (5-shot), outperforms ATHA and TIR by at least 2.82 and 1.39 points on the same episodes, and consistently improves three further backbones, three parameter-efficient fine-tuning frameworks and five further target domains. 

---
# ABCoord: An Artificial Bee Colony (ABC)-Based Decentralized Distributed Two-Layer Decision-Making Architecture for the Coordination of AI Agents 

**Authors**: Dervis Karaboga, Bugra Alperen Uluirmak  

**Link**: [PDF](https://arxiv.org/pdf/2611.28103)  

**Abstract**: Rapid progress in large language models (LLMs) is turning AI agents into multi-agent systems that jointly solve complex tasks, and the architecture that coordinates the agents toward a common goal is gaining fundamental importance in these systems. Since centralized architectures can become bottlenecks as the number of agents grows, whereas distributed architectures incur increasing communication costs, a decentralized coordination architecture that scales with low communication cost is needed. Recently, the structure of the artificial bee colony (ABC) algorithm has been defined for the first time as a two-layer decentralized information-processing and decision-making architecture. ABCoord is built on this proposed architecture. The results show that ABCoord, used for the coordination of AI agents, performs at a level comparable to existing architectures on the BigCodeBench benchmark in the self-test condition, in which the tests are also written by the language model. In the oracle condition, in which the solutions are evaluated with the official tests, the number of solved problems increases considerably. 

---
# VideoGym: An Adaptive RL Environment for Long-Video Agents 

**Authors**: Jing Wang, Yi Ke Yun, Yifeng Yao, Huishuai Zhang, Dongyan Zhao, Tao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28102)  

**Abstract**: Reinforcement learning (RL) trains video agents to locate sparse evidence for long-video question answering through structured memory retrieval and temporal re-observation. This training, however, faces a long-tail modality distribution in the data. In our source pool, text-heavy questions account for 58% of samples, compared with 15% graph-heavy, 17% visual-heavy, and 10% time-range questions. Fixed-distribution RL therefore allocates fewer rollouts to underperforming question types. Reweighting these types could increase their sampling frequency but repeatedly reuses the same finite banks of original questions. To solve this problem, we introduce VideoGym, a curriculum-based RL environment that jointly constructs video-grounded training tasks and allocates rollouts among them. Within the environment, a memory-augmented VideoAgent is trained to coordinate retrieval from text, graph, and visual memories with temporal re-observation. Across episodes, VideoGym adapts VideoAgent's training distribution online: it casts per-period sample selection as an adversarial bandit over (modality, mode) arms and applies the EXP3 update to shift sampling toward arms with low answer accuracy. To supply the augmented-mode arms, a TaskGenerator-Verifier pipeline produces harder variants of existing questions that remain answerable within the same video. Across five long-video benchmarks, VideoGym lifts RL accuracy from 60.4% to 64.8% on average and by up to 10.5 pp on underrepresented graph-heavy questions with 3.3x sample efficiency. Together, these results show that task expansion and adaptive allocation can improve video agent learning from limited, imbalanced training data. 

---
# Causal Dioramas: Controllable Video Generation, Revision and Editing via Intervention 

**Authors**: Yifan Shen, Xinyi Yang, Peiyuan Zhu, Yichang Jian, Jinyuan Hu, Xilin He, Fengming Zhu, Guangyi Chen, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28098)  

**Abstract**: Two centuries ago, an early cinema-like theater called the Diorama could already control its scenes at will. Today's video generation models produce videos of near-cinematic quality, yet they still lack this controllability. This limitation stems from the inherent semantic gap between text and pixels, since a prompt can describe a video only coarsely and leaves its details and relations beyond control. We bring this ability to video generation with Causal Dioramas, modules that decompose a video into a concept graph of variables that can be intervened on individually. Each instruction is translated into an intervention on the concept graph, modifying the specified concepts and their dependents. A connector then conveys this intervention to the video generator, which reuses the original noise to produce a counterfactual video while preserving all other content. Building on Causal Dioramas, we develop a unified framework for controllable generation, counterfactual revision of generated videos and editing of real videos. We outperform both open-source and commercial models in controllability on T2V-CompBench, CoVEBench and OpenVE-Bench. 

---
# VTree: Question-Driven Growth of Hierarchical Memory for Agentic Long-Video Understanding 

**Authors**: Xiaoyu Ma, Lianyu Hu, Zixuan Hu, Zhizhen Wu, Zeqin Liao, Shengqian Qin, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28093)  

**Abstract**: Long videos spanning tens of minutes to hours far exceed the context window of today's vision-language models, and the standard remedy of sparse uniform frame sampling discards exactly the fine-grained evidence that specific questions depend on. Existing video agents explore on demand but re-explore from scratch for every question and discard their exploration state once an answer is produced, so per-question cost scales with video length. Hierarchical-memory methods instead fix the scope of each depth in advance and deepen along a pre-committed pyramid rather than toward the question. We observe that a video's coarse structure (where its scenes and topics begin and end) is question-independent, whereas which part a question must zoom into, and how deep, is question-dependent. We propose VTree, which builds a coarse hierarchical memory tree over a video once and then, for each question, grows branches only where that question needs depth. The growth is globally routed so any part of the timeline is reachable, sufficiency-gated so drilling stops when the evidence suffices, and query-conditioned so a branch unfolds into the sub-intervals that answer \emph{this} question. Crucially, every expansion is written back into the video memory, allowing later questions to reuse previously discovered structure and evidence without repeating the corresponding expansion process. Across five long-video understanding benchmarks, VTree consistently improves the accuracy of diverse vision-language backbones while substantially reducing the context length and execution time required for each question. 

---
# Every IVUS Frame in Its Place: A Zero-Install Browser Tool for Registering Intravascular Ultrasound to the Coronary Angiogram, and a Path to Angiography-Only Plaque Imaging 

**Authors**: Prahlad G. Menon, Srilakshmi Adhyapak  

**Link**: [PDF](https://arxiv.org/pdf/2611.28089)  

**Abstract**: Intravascular ultrasound (IVUS) shows plaque composition, calcium and stent apposition that the coronary angiogram cannot, but relating each IVUS cross-section to its position on the angiogram is still largely done in the operator's head. Commercial co-registration is tied to vendor consoles and to set-up at acquisition, so archived angiograms and pullbacks cannot be co-registered, and paired data for training AI models remain scarce. We present a semi-automatic registration tool that runs entirely in a web browser, with no installation and no upload. The clinician picks a contrast-filled reference frame (aided by automatic contrast and cardiac-phase analysis), selects or traces the vessel centerline (aided by Frangi vesselness and minimal paths), and pins the pullback ends and visible landmarks; a monotone piecewise-linear map then places every IVUS frame on the vessel, and each case exports as a frame-level training record. The original Python analysis runs unchanged in WebAssembly and videos are decoded frame-accurately with WebCodecs. On 6 angiographic runs the in-browser analysis was bit-identical to the server pipeline on identical frames; browser decoding differed by at most 2 grey levels, suggested reference frames agreed in all runs and the default automatic centerline in 4 of 6. A 546-frame pullback decoded in 5.3 s. On 36 phantoms with known ground truth, the clinician-guided trace stayed inside the lumen along 87-99% of the vessel, whereas the fully automatic centerline fell from 83% to 0% as noise rose and contrast fell. In a registered Left Main case, one side-branch landmark moved mid-pullback positions by 17.5% of the segment. We map where AI can enter each step, up to predicting a virtual IVUS from the angiogram alone, and make the tool freely available online with a one-click demo. 

---
# Efficient Memory-Constrained Video Object Segmentation via Subspace Optimization 

**Authors**: Hai-Anh Le, Le-Trung Nguyen, Enzo Tartaglione  

**Link**: [PDF](https://arxiv.org/pdf/2611.28087)  

**Abstract**: Memory-based architectures for video object segmentation (VOS) have achieved remarkable performance by maintaining dense feature representations from past frames in a streaming Memory Bank. However, storing these features over time incurs substantial memory overhead and computational cost. In this work, we identify significant temporal redundancy in memory representations, where consecutive frames exhibit highly overlapping low-rank subspaces that remain stable over time. Building on this observation, we propose Memory Subspace Iteration (MSI), a training-free, model-agnostic plugin that exploits temporal subspace stability to efficiently compress streaming memory features. Extensive experiments on standard VOS benchmarks demonstrate that MSI preserves segmentation accuracy across multiple architectures while reducing the memory bank footprint by up to 25%. Our implementation is available at this https URL. 

---
# UbiEnlight: User-Perception-Aware Real-Time Low-Light Video Enhancement on Mobile Devices and Smartglasses 

**Authors**: Minfan Wang, Sicong Liu, Teng Li, Zimu Zhou, Sixun He, Bin Guo, Junzhao Du, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28085)  

**Abstract**: Low-light video capture is now common on smartphones and wearables, yet dynamic illumination coupled with mobile/werable user motion causes noise, blur, and flicker that make videos hard to perceive and use, especially for users with night blindness and near-eye displays. Prior work enhances frames with post-hoc temporal constraints or optimizes videos directly; however,low-light and motion degradations are spatiotemporally coupled, so post-hoc consistency after per-frame restoration often leaves flicker and motion artifacts. We present UbiEnlight, the first diffusion-transformer-based on-device system for real-time, temporally stable low-light video enhancement. UbiEnlight builds on two insights: (1) illumination and structure separate more robustly in the frequency domain, enabling a spectrum-guided diffusion transformer that injects Fourier amplitude/phase priors to correct illumination while anchoring structure without optical flow; and (2) user perceptual tolerance varies by context, motivating a perception-aware controller that adapts sampling depth, attention reuse, and frame caching at runtime. We implement UbiEnlight on smartphones and smartglasses, and evaluate it against state-of-the-art baselines. UbiEnlight improves temporal stability by up to 54.76% under dynamic/extreme conditions while sustaining real-time on-device performance. 

---
# Frontier Open-Weight AI Risk Management Framework 

**Authors**: Brian Tse, Oskar Galeev, Weibing Wang, Yuan Cheng, Liang Fang, Wenbin Qiao, Xiaochen Wang, Chen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28082)  

**Abstract**: Open-weight models advance transparency, research, and innovation, and spread the benefits of AI more widely. They also carry distinct risks. A release cannot be undone, safeguards can be removed through fine-tuning, and no single party can monitor use or withdraw the model. Managing these risks therefore requires an approach tailored to open release. This framework addresses the governance challenge at the heart of that choice. Co-developed by Concordia AI and this http URL, it adapts the full risk management lifecycle, from risk assessment to mitigation and governance, to the unique properties of open-weight models. This document does not resolve every tension between openness and safety. Rather, it provides the first comprehensive, evidence-based foundation for navigating these tensions. As frontier open-weight releases accelerate, risk management best practice and standards are urgently needed. We invite researchers, developers, policymakers, and civil society to build on this framework and advance the science and practice of open-weight AI governance. 

---
# Multimodal Violence Understanding and Temporal Retrieval in Movies with Adaptive Audio-Visual Fusion 

**Authors**: Jiaqian Li, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2611.28080)  

**Abstract**: Violence in films and streaming media remains a concern for audience protection and creates substantial workload for manual content classification. Existing violence-detection methods often rely on limited modalities or short video clips, making it difficult to capture complementary evidence and localise violent events in long, untrimmed videos. In this study, we develop a framework for multimodal video understanding that classifies violence severity, predicts multiple violence types, and locates violent events in time. We combine temporally aligned visual and audio features extracted using pretrained encoders through attention-based adaptive fusion, followed by Transformer-based temporal modelling. Our framework supports three complementary tasks: video-level violence-severity classification, multi-label violence-type prediction, and temporal localisation of violent events. We evaluate the framework on the XD-Violence benchmark and examine its cross-domain performance using AIRTLab. On XD-Violence, our multimodal model achieves 93.27\% severity-classification accuracy and a micro-F1 score of 88.60\% for multi-label violence-type prediction, outperforming the corresponding unimodal configurations. We achieve a mean temporal intersection-over-union (tIoU) of 0.5075 for temporal localisation. Our cross-domain experiments reveal performance degradation under different recording conditions, while our movie case studies demonstrate the framework's potential to retrieve potentially violent scenes for further review. Our results highlight the practical value of multimodal intelligent systems in supporting human reviewers with the identification, classification, and temporal retrieval of violent content in films and streaming media. 

---
# Adversarial Attacks Expose Path-Dependent Failures in Discrete Medical Image Tokenizers 

**Authors**: Ahmed Loughzali, Akash Yallamati, Aadiv Reki, Aslan Shaukat Vaida, Jeyashree Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2611.28078)  

**Abstract**: Discrete image tokenizers convert medical images into token sequences that downstream models either decode into an image for a classifier (the reconstruction path), classify directly (the token path), or feed to a generative multimodal model. We ask whether imperceptible input perturbations can break these systems through the tokenizer alone. We evaluate four tokenizers on three imaging modalities and extend the strongest attack to a deployed generative model. Our central finding is path-dependence: the same tokenizer fails in opposite ways depending on how its output is used. A supervised attack collapses diagnostic accuracy on the reconstruction path, with ROC AUC falling to $\leq 0.013$ at $\epsilon=2/255$ on all four tokenizers, while sparing the token path. An unsupervised, label-free attack does the reverse; its damage concentrates in the multi-codebook tokenizers, worst in MedITok, whose token-path AUC falls from 0.919 to 0.239. The same label-free attack transfers to LLaVA-Med, a frozen medical vision-language model built on MedITok, making it report proliferative diabetic retinopathy in a healthy retina or call a malignant lesion benign, without access to the question, answer, or language-model gradients. Harm is driven by how far the attack moves the tokenizer's internal representation, not by how many tokens change: on the learned-codebook tokenizers, a control attack that changes nearly all tokens but moves the representation less leaves every path and answer intact. Domain-specific pretraining does not provide robustness: MedITok has the best clean accuracy yet is the most fragile, more so than UniTok, its general-domain counterpart. None of six candidate defenses repairs every failure mode, and several fail once the attacker optimizes against the defense directly. Medical tokenizer robustness must therefore be evaluated per downstream path, against an adaptive adversary. 

---
# Fairness Evaluation for Diffusion and Flow Models Should Account for the Generation Path: Implications for Clinical Imaging 

**Authors**: Dharini Raghavan, Amritpal Singh, Anant Madabhushi  

**Link**: [PDF](https://arxiv.org/pdf/2611.28077)  

**Abstract**: Medical image generation models are often evaluated using images produced by a single sampling procedure. However, diffusion and flow models can use different numbers of sampling steps, sampling schedules, or numerical solvers without changing their trained parameters. These changes may affect the generated images and the clinical measurements derived from them. We argue that fairness results obtained under one sampling procedure should not be assumed to hold under another, particularly when changes in sampling affect clinical task errors differently across patient groups. Building on previous work on medical image quality, accelerated sampling, generation dynamics, and generative model fairness, we examine why the sampling procedure should be considered part of the evaluated system. We define subgroup differences in clinical task error across sampling procedures and provide a mathematical example showing that two groups can have identical outputs under the exact dynamics and a reference sampler but different errors under an alternative sampler. We also extend this result to stochastic generation under shared noise. An exploratory experiment with an age-conditioned image generation model illustrates how changing the number and timing of sampling steps can alter image-derived measurements, although these differences do not establish clinical unfairness. We therefore argue that fairness evaluation should assess the sampling procedure used in practice and examine intermediate generation states when they help explain differences in clinically relevant errors across patient groups. Analysis code is available here this https URL 

---
# Following the Error Through a Compressed Network:An Angular Measure for FFN Weight Compression in ViT-L/16 

**Authors**: Paolo D'Alberto  

**Link**: [PDF](https://arxiv.org/pdf/2611.28067)  

**Abstract**: We study the compression of the 48 feed-forward network (FFN) matrices of the Vision Transformer ViT-L/16 to 2--8 bits with four schemes: a per-token grid, a fixed Gaussian Lloyd--Max table, a table fitted to the matrix, and a random-sign Hadamard rotation followed by the fixed table. We evaluate on 50,000 images. At 4 bits the top-1 accuracy of the four schemes lies within 0.8 points of the float32 model, while the agreement with its predictions ranges from 98.4\% to 94.4\%. We use the angular error between the compressed and the float32 residual stream, followed through the blocks, as a portable measure that ranks the schemes as the final classification does and shows where the ranking is built: with rotation the angle grows close to the square root of the number of blocks, the signature of independently accumulating errors, while the other schemes are amplified in the later blocks. The same measure shows that a scheme that is better on one block in isolation can be worse in the whole network. To address this inconsistency we introduce an angular measure of the spectrum of the output of a block, which in isolation orders the schemes as the whole network does where they diverge most. We support these observations with real samples and with statistically generated ones. The scope is one model, the FFN weights only, and float evaluation of compressed weights; no speed-up is claimed. The error measures and the sample generation, however, can be applied to other models and precisions. 

---
# SnakeTranslate: Weakly Supervised Policy Learning from Offline Parallel Text for Simultaneous Translation 

**Authors**: Yu Pu, Zhiyun Fan, Baiji Liu, Zhifu Gao, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.28055)  

**Abstract**: Adaptive simultaneous translation requires reliable prefix translation and timely READ/WRITE decisions, but ordinary parallel corpora provide no action annotations. We present SnakeTranslate, which adapts loss-derived alignment to sampled causal histories for jointly learning both capabilities. The framework supports supervised fine-tuning (SFT) on parallel text and on-policy distillation (OPD) with reverse Kullback--Leibler (KL) divergence on student-generated translations. At inference, source and target tokens form one append-only interleaved history, enabling incremental decoding with cache reuse. On FLEURS, NTREX128, and WMT24++, our 4B model improves quality--latency trade-offs over streaming policies applied to the same offline-tuned backbone, with further gains from OPD. It also compares favorably with larger translation models at low and moderate latency, while higher-latency quality gaps remain on WMT24++. 

---
# Benchmarking and Adapting LLMs for English-Swedish Medical Question Answering 

**Authors**: Seyed Alireza Molavi, Zhan Su, Stefan Byttner, Yan Hu, Peyman Sheikholharam Mashhadi, Prayag Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2611.28050)  

**Abstract**: Large language models do well on English medical exams, but less is known about Swedish. This matters most for small open models that hospitals can run locally. To reduce this gap, we first build an item-aligned English-Swedish benchmark from ten medical multiple-choice datasets; each Swedish item keeps the options and answer of its English source. A conservative refinement loop repairs the translations: open models from three families find errors, ground them in a Swedish medical term base, and keep a model-written edit only if blind judges vote for it. On this benchmark, small models lose much more accuracy in Swedish than large ones, especially when they answer without reasoning. Second, we ask whether English data and open teacher models can close this gap. We translate the training data, let teachers write reasoning traces, and fine-tune two small students on gold answers or traces in English, Swedish, or both. Training on reasoning traces that include Swedish closes the gap and lifts the 1.2B student from 23.5\% to 52.0\% in Swedish, slightly above a 20B open model in its fast mode, with a sixth of its memory. We release the benchmark, its edit history, and the code. 

---
# Distributionally-Verifiable Rewards for Human-Behavior Simulation 

**Authors**: Ruoyun Ma, Shuang Liu, Peiye Liu, Ao Qu, Baichuan Mo  

**Link**: [PDF](https://arxiv.org/pdf/2611.28048)  

**Abstract**: Large language models are increasingly used to simulate survey respondents in social-science research. A useful simulator should predict individual responses, reproduce population response distributions, and capture demographic differences. Common training approaches struggle to balance these goals: exact-match rewards encourage the model to concentrate on the most frequent human response, which is contradictory to the goal of distributional alignment. Meanwhile, aggregate distribution matching can look good even when demographic information is ignored. To this end, we introduce Distributionally-Verifiable Rewards (DVR), a reinforcement-learning reward that uses demographic group distribution gaps to guide individual responses, designed by aligning policy generated frequencies with human response frequencies within each question-group margins. We evaluate the method using the OpinionQA dataset and find that comparing with accuracy-only RLVR, DVR preserves per-response accuracy while reducing JSD by 34.5% in-distribution and 14.6% on held-out questions, with higher DPC point estimates on both splits. Comparing KL-SFT with the demographics-blind marginal further shows that low JSD is insufficient to establish fidelity to population heterogeneity, highlighting that accuracy, distributional fit, and demographic structure should be evaluated together. 

---
# What Counts as Attack Success? Evaluating Harmfulness in Red Teaming LLMs 

**Authors**: Ruyuan Wan, Adriana Alvarado Garcia, Md Toufikuzzaman, Dongwon Lee, Ting-Hao `Kenneth' Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28038)  

**Abstract**: Judging whether a model response is harmful requires contextual and normative choices, yet red-teaming studies often delegate this judgment to automated evaluators. We examine how harmfulness criteria are specified in red-teaming research and how they carry through to reported results. Reviewing 76 papers, we find that many adopt their criteria from prior datasets, classifiers, or judge prompts rather than defining harmfulness themselves. Applying evaluators from these papers to identical outputs, we find that they can reach different judgments on the same response, which can change conclusions when comparing models. These differences arise partly because evaluators draw the boundary of harm in different places: some count any response that does not refuse, while others count only output that could plausibly cause harm. We argue that red-teaming studies should state their harmfulness criteria and report how they chose and validated their evaluators to support more reliable measurement and interpretation. 

---
# Safety Systems Watch What the Model Says, Not What the Child Says 

**Authors**: Fatma Egal, Rehnuma Taskin  

**Link**: [PDF](https://arxiv.org/pdf/2611.28029)  

**Abstract**: Children are increasingly talking to consumer chatbots. We audit the published category definitions of 17 safety taxonomies and child-safety benchmarks. Combinations of these systems lead us to 144 categories whose definitions could be reliably coded: 104 classify properties of model output, 25 classify user intent, and 15 describe user-state or user-impact. Within these 15 user-state/user-impact categories, 10 attribute the relevant condition to the system, 1 to the user, 2 concern the user's disclosure of personal information, and 2 do not specify the responsible source. None of the coded categories identifies harm inflicted on the child by another person as the object of evaluation. Child abuse appears repeatedly across these taxonomies, but as prohibited content, prohibited generation, or prohibited user intent, not as an externally inflicted condition that a child may disclose to the model. Consequently, a child reporting abuse has no dedicated taxonomy label that identifies the disclosure as a safety-relevant event requiring a response distinct from content suppression. We ran five behavioural probes, grounded in the child-disclosure literature, through six publicly accessible consumer-facing chat models to examine the behavioural consequences of this omission. We propose no detection system. Instead, we prove that externally inflicted harm disclosed to an AI system remains unnamed in the evaluation vocabulary used by current safety taxonomies, and that naming the missing case is a precondition for measuring, comparing, and improving model behaviour toward such disclosures. 

---
# REMORA: A Voice-First Memory Assistant for Grounded Personal Recall and Independent Living 

**Authors**: Bilal Jawaid, Muhammad Maaz Uddin, Osama Farooqi, Saadia Karim  

**Link**: [PDF](https://arxiv.org/pdf/2611.28026)  

**Abstract**: Most of what matters in a day is spoken and written down nowhere: a dosing interval named at a clinic, a deadline moved in passing, the place a document was set down. Such information is consequential because it is operational, and fragile because it arrives during divided attention -- the moment its importance becomes clear is rarely the moment it was said. For people living with mild cognitive impairment, Alzheimer's disease or age-related episodic memory decline, that gap bounds independent living. Conventional aids close it poorly: notes, reminders and calendars require the user to notice, interrupt and transcribe, adding extraneous cognitive load exactly where attention is already committed. We present REMORA, a voice-first memory assistant that converts consented speech into structured, auditable memory records. REMORA transcribes and diarizes captured audio, extracts memory candidates -- tasks, appointments, names, instructions, object locations -- each retaining provenance into the conversation it came from, and holds records, embeddings and relationships in a single relational system of record. Recall combines lexical, semantic, relational and recency signals, since no one signal covers the ways people interrogate their own past, and reaches the user either through an explicit query or through a confidence-gated cue that stays silent unless it is likely to help. Generation is bound by a grounding contract: only retrieved evidence may be used, and absent a supporting memory REMORA abstains rather than confabulates, because a fluent invention about one's own past will be acted upon as fact. A requirements survey and a pilot study with memory-loss users indicate that conversational forgetting extends well beyond clinical populations, that grounded and concise recall is feasible in controlled settings, and that privacy, rather than accuracy, is the binding constraint on adoption. 

---
# Measuring Human-AI and AI-AI Teams in Multi-turn Layout Creation 

**Authors**: Rajvir Singh Naginder Singh Virdi, Yewen Pu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28025)  

**Abstract**: Co-creating layouts such as websites and slides is a useful testbed for evaluating multi-turn instruction following of LLM systems. Since an intended layout's appearance cannot be well expressed in a single turn, the user must wait for the LLM to generate a layout then iteratively refine it with the LLM over multiple turns. We present a small-scale, exploratory (n=6) study of this multi-turn process to compare the differences of human-AI and AI-AI teams on the same layout-generation task. Given a reference wire-frame layout, a participant or a language model 'speaker' issues natural-language instructions to a fixed 'listener' LLM over multiple turns until the speaker is satisfied. Rather than an LLM as the judge, we fit a scoring function over geometric layout features using curated human preference data (n=19). We find that LLMs tend to encode the entire layout in the first turn, while humans iterate over many turns on average, writing much shorter instructions. We also find that human-AI teams consistently scored lower than AI-AI teams. This gap is driven by a limiter of human participants in specifying precise sizes and positions (e.g. 42 pixels to the right). Humans perform almost at par with language models on other features like ordering the elements into nested structures. We then perform interventions on the speaker LLMs to make them more human-like, and find it difficult. This raises the challenge of using LLMs as user models to evaluate multi-turn co-creation tasks. 

---
# Misalignment Between Educators' Testing and End Users' Behavior: What 17,000+ Chatbot Messages Reveal About AI Literacy Training for Teachers 

**Authors**: Ruiwei Xiao, Jiayi Wang, Miina Koyama, Yue Wang, Qing Xiao, Xinying Hou, Norman Bier, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2611.28015)  

**Abstract**: AI literacy training for educators is expanding rapidly, but its effectiveness rests largely on self-report, knowledge tests, or interviews, revealing little about whether learning carries into practice or affects students and colleagues who use what educators build. We implemented AI for Learning, an asynchronous online course that trained 87 higher-education educators to recognize AI's capabilities, limitations, and ethics; apply prompt engineering; and evaluate AI outputs. Its central assignment asked educators to build, test, and voluntarily deploy chatbots for their practice on an educational chatbot authoring platform. Using LLM-assisted qualitative coding validated against human annotation, we analyzed the resulting 214 chatbots and their builder-bot, peer-bot, and end user-bot conversations, 17,000+ messages spanning 1.5 years. Educators 1) specified what bots should do more often than what they should not; 2) adopted templated prompt components but rarely untemplated strategies; and 3) tested routine use but seldom iterated, leaving 74% of end users' adversarial use untested. Still, 36 bots (17%) reached end users and 63 (29%) were adopted by peers. We contribute two successes and two lessons spanning the lifecycle of educator AI literacy training. 

---
# Do as I Say, Not as You Do: Assessing Public Versus Practitioner Perspectives on the Development of Artificial Intelligence and Language Technologies 

**Authors**: Tyler Loakman, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.28013)  

**Abstract**: With the public awareness of artificial intelligence (AI) applications such as large language models (LLMs) and generative text-to-video models reaching unprecedented levels, there is an increasing need to reconcile the perspectives held by informed practitioners, who research and develop these technologies, with the interests and opinions of the general public, who constitute the majority of people impacted by technological developments. In this work, we survey practitioners (i.e., academics and industry professionals) alongside members of the general public on perspectives relating to pertinent areas of artificial intelligence, including perceived benefit and risk, the ability for models to achieve consciousness and output art, and the use of copyright protected material for model training. Alongside comparisons of diverse perspectives on these topics, we additionally contextualise the findings with a discussion of relevant literature that supports and refutes the myriad observed stances. 

---
# Social sycophancy differs across access modes of large language models 

**Authors**: Alex Klem, Uwe Peters  

**Link**: [PDF](https://arxiv.org/pdf/2611.28012)  

**Abstract**: Large language models (LLMs) can display sycophancy, a tendency to excessively praise, agree with, or validate a user. Existing studies have tested LLMs for sycophancy primarily through application programming interfaces (APIs) rather than the consumer user interfaces (UIs) through which the public commonly interacts with LLMs. Access mode (API vs. UI) may affect observed sycophancy rates. However, it remains unknown to what extent prominent models show API-UI differences in social sycophancy, which includes behaviors potentially more likely in UIs (e.g., softening warranted criticism). We tested ChatGPT, Claude, Gemini, and DeepSeek through both APIs and UIs on four dimensions of social sycophancy. Across models, UI access was generally associated with greater social sycophancy, with odds ratios up to 7.56 (95% CI [6.22, 9.20]). However, UI-API differences varied significantly across models for all sycophancy dimensions, including a pronounced reversal for DeepSeek moral sycophancy (43% UI vs. 82.3% API). Overall, our results suggest that API-based LLM evaluations may underestimate the prevalence of sycophancy in corresponding consumer-facing systems, making access conditions consequential for the external validity of LLM evaluation. To facilitate research on access-mode effects, we release 57,000 model responses and SycoUI, a browser-automation pipeline for standardized cross-interface LLM evaluation. 

---
# Scrub Data: A Framework for Reproducible Data Curation with AI Coding Agents 

**Authors**: Justin Kay, Shir Bar, Sara Beery  

**Link**: [PDF](https://arxiv.org/pdf/2611.28003)  

**Abstract**: Data curation---the process of collecting, cleaning, and joining raw datasets for unified analysis---is a time-consuming yet crucial part of any data science project. With the emergence of powerful agentic coding tools, it is tempting to "vibe curate"---i.e., to prompt AI agents to perform data curation operations and blindly trust their outputs in order to finish work more quickly. However, this can exacerbate two key challenges that already plague manual data curation workflows: reproducibility---the ability to trace the exact set of modifications applied to raw datasets---and verifiability---the ability to audit changes and confirm that data is processed properly. To address these issues, we introduce Scrub Data, a framework that enables the use of AI agents in data curation workflows while transparently maintaining both reproducibility and verifiability. The framework involves an interactive data curation loop between a user and an AI agent, backed by a data provenance graph that tracks and versions all dataset updates. The data graph requires that all updates are formatted as self-contained executable steps, allowing any version of a dataset to be reconstructed from raw data by playing forward the transformations stored in the graph. This workflow takes place within a lightweight web application that is designed to be modified by users and agents in real-time to create custom visualization tools for identifying data curation needs and verifying outcomes. We demonstrate the usefulness of the framework with a real data curation use case from animal movement ecology. Starting with 89M raw GPS coordinates, we use the framework to curate a benchmark dataset of 2.5M coordinates to be used for machine learning or statistical analysis. The framework is available for use and extension at this https URL. 

---
# Direct Conditioning of Audio Diffusion Transformers on fMRI Reveals Cortical Contributions to Sound Reconstruction 

**Authors**: Matteo Ciferri, Tonio Weidler, Matteo Ferrante, Nicola Toschi, Elia Formisano  

**Link**: [PDF](https://arxiv.org/pdf/2611.12603)  

**Abstract**: Reconstructing audio from brain activity remains challenging. Existing brain-to-audio pipelines typically map neural responses to predefined stimulus representations before synthesis. While recent work in the visual domain has explored direct conditioning on brain activity, this approach has not yet been extended to audio diffusion models. We introduce MAAS (Multi-Area Auditory Synthesis), an end-to-end framework that conditions a pretrained audio diffusion transformer on ROI-wise fMRI responses without first supervising a decoder to predict an external stimulus representation. MAAS uses region-specific projections and learned brain tokens to condition Stable Audio through a dedicated cross-attention pathway. On a 7T fMRI dataset, we demonstrate consistent improvements over baselines, including two-stage fMRI-to-latent conditioning, across both acoustic and semantic evaluation metrics. In addition, we introduce a fidelity metric grounded in a computational model of auditory neural processing, which allows us to quantify the contribution of individual auditory cortical regions to reconstructed audio representations. 

---
# Typing Behavior in Human-LLM Interaction: Keystroke Dynamics Reveal Cognitive Effort During Prompting 

**Authors**: Laura Schütz, Yousri Cherif, Clara Sayffaerth, Thomas Weber, Francesco Chiossi  

**Link**: [PDF](https://arxiv.org/pdf/2606.28090)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into daily routines, understanding how users interact with these systems is crucial for effective human-AI collaboration. This work investigates keystroke dynamics as a behavioral measure of user mental effort and perceived output usefulness in human-LLM interaction. We conducted a user study (N = 36) to examine how task difficulty (easy vs. hard) and device type (desktop vs. mobile) influence typing behavior and workload (NASA-TLX) during interactions. Our results indicate that hard tasks led to significantly more keystrokes, slower typing, increased pauses, and higher self-reported workload. Device type had weaker effects, with mobile use slightly reducing input length and typing speed. While keystrokes captured differences in cognitive effort, they did not predict perceived LLM output usefulness. These findings highlight the potential of keystroke dynamics as real-time indicators of cognitive effort during LLM prompting, while also showing their limitations in capturing perceived collaboration success. 

---
