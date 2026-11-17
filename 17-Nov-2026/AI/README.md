# Right Reason, Wrong Advice: Evaluating Recourse in LLM Explanations of Denied Requests 

**Authors**: Arunkumar V  

**Link**: [PDF](https://arxiv.org/pdf/2611.21545)  

**Abstract**: Language models increasingly write the explanations people receive when a rule system denies their request. If several rules deny the request independently, an explanation can name a true reason and still leave the person denied after they fix it. We call the missing property recourse validity: changing what the explanation cites, without changing what was asked for, can flip the decision. We compute it exactly, and a hitting-set argument predicts from the policy alone when a correct reason fails. In six of seven rule systems ported from access control and US benefit eligibility, a single correct reason almost never gives recourse when several reasons apply. Across 20 LLMs, checks of whether the cited reasons are correct, computed or asked of an LLM judge, are close to chance at spotting this failure, as the argument predicts. A strong judge asked whether acting on the explanation would work does spot it. Stronger and reasoning models do better but still fail on many denials with four or more independent reasons. Asking for every rule helps every open model with room to improve. Explanations of eligibility decisions should be built from the decision function and judged by whether acting on them works. 

---
# SciResearch-Eval: Evaluating Scientific Judgment in Autonomous Experimental Agents 

**Authors**: Brian T. Weston, Alondra Torres Contreras, Aldair E. Gongora  

**Link**: [PDF](https://arxiv.org/pdf/2611.21541)  

**Abstract**: We introduce SciResearch-Eval, an interactive benchmark of scientific judgment built on a standard laboratory problem: measuring a material's thermal properties by heating it and fitting a heat-conduction simulator. An agent designs its own experiments, uses a supplied simulator that may be under-resolved or physically wrong, and reports each property with a 95% interval. To our knowledge it is the first benchmark to combine agent-designed experiments, a deliberately flawed simulator, and proper scoring of calibrated intervals against exact ground truth. On a held-out split (93 tasks, 9 models, 2,511 episodes), models handle procedural traps. When the simulator is missing physics, most report wrong values while stating they are 95% sure: their ranges contain the truth in 20-62% of cases, against 94% for GPT-6 Astra and 99% for Claude Fable 5.2. At default effort the failing Claude 5.5 models rarely diagnose the missing physics (Opus in 35% of episodes, against 66% for Astra), and even when they do, their ranges often miss; Astra and Fable design experiments in which the flawed simulator is accurate, such as gentle heating, which keeps Astra calibrated even when it does not name the cause. Asking for wider intervals does not help; stating the strategy improves interval scores 1.6-2.5-fold, replicated on a pre-registered fresh split, but coverage barely moves. A pre-registered effort ablation shows that the correction is latent in the Claude 5.5 models: at maximum effort Opus improves more than tenfold and is no longer detectably behind Astra at default, but only with 13-31 times more reasoning; older and open-weight models gain nothing. Fable 5.2 is the most accurate model at default effort (interval score 1.9 times better than Astra's) but writes 6 times as many tokens, and at Astra's token count it falls short. We release the environment, graders and harness. 

---
# Procedure without Judgment: What Small Language Models Learn under an Exact Verifier 

**Authors**: Arunkumar V, Manoranjan Gandhudi, Gangadharan G. R., Arun Prakash, S. Senthilkumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.21529)  

**Abstract**: When an authorization policy blocks an agent's tool call, the audit log should name the facts that forced the decision and nothing more. For an executable policy over an enumerable context space this minimal sufficient reason can be computed exactly, so an explanation is scored against ground truth rather than a proxy for it. We use that exactness to ask what a model can learn from a verifier that has no error in it. The explainer sees one request and one decision, never the policy, and has thirty turns in which it may call the policy on altered requests and ask whether a candidate set of facts is sufficient. Frontier models find the exact reason about half the time with the tools alone, and one sentence telling them to shrink the answer lifts Haiku 4.5 from 58% to 75%. A 4B model fine-tuned on 2000 traces of the deletion algorithm reaches 92% on the policy it was trained on. Teacher kind is what moves this: at matched data size the algorithm-taught student leads the frontier-taught one in all three training seeds, and the two arms never overlap. What the student learns is a fixed schedule of sufficiency checks, and the schedule does not travel. On three public policies it never saw, its answers stay sufficient (94%) and stop being minimal (48%), while Haiku under the same sentence is exact on all 47 episodes. Reinforcement learning from the exact reward leaves held-out accuracy where it was and inverts the strategy, from probing the policy to spending the entire budget on sufficiency checks. The oracle made the effect of training legible, but it did not make training better. 

---
# Safe Resemblance, Silent Propagation: Limits of Text-Similarity Guardrails for AI Agent Safety 

**Authors**: Charles Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.21520)  

**Abstract**: When AI agents relay information to one another, an output can preserve an injected instruction's effect on downstream agents even after its wording changes, yet legitimate quotation and template reuse can produce high textual overlap with the source, so text-similarity guardrails that attempt to flag the former risk conflating these two cases. We test this distinction using DonkeyRail, a public classifier combining BLEU, ROUGE-L, and METEOR features, deployed in a controlled three-agent email relay, defining a behavioral measure in which an output counts as functional (the injected procedure induces fresh computations in two subsequent agents) independent of the detector's score, alongside legitimate controls that explicitly authorize quotation and reuse. At its default threshold the classifier misses 8 of 19 functional outputs while flagging 70 of 192 legitimate replies, with every completed quotation and template-reuse reply triggering a false alert; calibrating for low false positives on one template set does not transfer, as the recalibrated threshold detects only one functional output and still produces 15 false alerts on new templates. We also identify an operating-point error in all 21 published classifier configurations, where ROC interpolation replaces direct threshold application, and show formally that high AUC against a surface-level text criterion provides no distribution-free guarantee of high AUC against the behavioral target, while the existing evaluation labels lack the joint observations needed to estimate the gap. An authorization analysis further shows that identical visible transcripts can require different safety decisions depending on the governing policy, a distinction no observation-only classifier can resolve. These results argue for behavioral and policy-grounded evaluation over text-similarity guardrails. 

---
# Generative AI and the Crisis of Higher Education Assessments: Examining 'AI-Proofing' Strategies for College-Level Assessments 

**Authors**: Bonam Mingole, Davis Yadav, Hui Min Lee, S. Shyam Sundar, Amulya Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2611.21476)  

**Abstract**: Higher education is confronting an academic-integrity challenge unlike any in its history. Students can now outsource homework, take-home assignments, and online examinations to large language models (LLMs) that complete them in seconds. Existing research on evaluating LLM performance on academic assessments has focused largely on standardized examinations, or on questions from college-level courses drawn from a single discipline. Far less is known about how LLMs perform on college-level assessment questions spanning multiple disciplines, that instructors have purposely designed to be AI-resistant, and about which features of such questions, if any, meaningfully constrain LLM performance. We address this gap by collecting 75 questions from authentic college coursework assessments, spanning 17 disciplines, which were contributed by 37 faculty members from ten U.S. universities and one U.K. university. Of these, 61 questions were intentionally designed by faculty members to be 'LLM-proof', while the remaining 14 questions were sourced from online courses. We then (i) evaluate three state-of-the-art LLMs' ability to correctly answer these questions with and without supplementary course materials; (ii) identify five strategies, through thematic analysis, that faculty used to make their questions harder for LLMs to solve; and (iii) test whether these strategies impact LLMs' ability to answer the questions accurately. Our experimental results show that the best-performing model scored approximately 83% without supplementary course materials and 85% with them, and we observed noteworthy differences across models, answer formats, and instructor tweaking strategies. We discuss implications for assessment design, academic-integrity policy, and the broader ethics of deploying LLMs in educational settings. 

---
# Subprocess-Constrained Markov Decision Processes 

**Authors**: Jiarui Gan, Debmalya Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2611.21471)  

**Abstract**: Constrained Markov decision processes (CMDPs) typically bound expected cumulative cost from an initial state. Such a guarantee can be insufficient when a policy must remain feasible after every history it may encounter. We introduce \emph{subprocess-constrained MDPs} (S-CMDPs), which require the expected discounted cost of each continuation, conditional on any reachable history, to satisfy a bound determined by the current state. This stronger requirement changes the structure and complexity of the problem: we show that stationary policies can be strictly suboptimal, and finding even an approximately optimal stationary policy is NP-hard. We adapt a recently introduced value-set iteration framework to show that history-dependent policies surprisingly overcome both obstacles. Our polynomial-time algorithm represents such a policy implicitly: its expected reward is within $\epsilon$ of the optimal feasible value, it violates each subprocess cost bound by at most $\delta$, and its action distribution can be computed in polynomial time for any given history. 

---
# Learning to Reason with Predictive Gains for Generative Recommendation 

**Authors**: Zhuoyang Li, Teng Shi, Jiakai Tang, Weijie Yu, Guofu Xie, Xiao Zhang, Lei Kou, Yang Yu, Lei Shen, Xin Chen, Bing Wang, Xiaoyi Zeng, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21441)  

**Abstract**: Reasoning-enhanced generative recommendation (GR) has attracted increasing attention for leveraging the reasoning capabilities of large language models (LLMs) to improve next-item prediction. However, existing methods primarily optimize reasoning through Group Relative Policy Optimization (GRPO) with rule-based rewards determined by final recommendation performance, which faces two key limitations: (i) unmeasured predictive contributions within reasoning, where the predictive contributions of individual reasoning steps are not directly reflected in optimization signals, and (ii) weakly discriminative learning signals within groups, where different reasoning trajectories within the group often receive indistinguishable rewards, limiting effective policy optimization. In this paper, we propose GainReason, a simple yet effective training framework that optimizes reasoning through Predictive Gain (PG). Specifically, PG measures changes in target Semantic ID (SID) log-likelihood induced by individual reasoning sentences, providing contribution-aware and discriminative rewards. We further theoretically establish the alignment between expected PG and the recommendation ranking objective. Building on this signal, we develop PG-based Reasoning Optimization (PGRO) to optimize reasoning generation with sentence-level rewards, supported by a custom causal attention mechanism for efficient PG computation. Additionally, an auxiliary PG-based Reasoning Utilization (PGRU) task adaptively strengthens reasoning-conditioned prediction to complement PGRO. Extensive experiments on three public benchmarks demonstrate that GainReason consistently outperforms competitive baselines and achieves substantial performance gains from reasoning. 

---
# Harness Leverage Across Model Size: A Ladder of Five Local Agent Harnesses from 0.75B to 4.7B 

**Authors**: Hao Wang, Ting Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.21407)  

**Abstract**: A harness turns a model into a system that completes tasks: it builds the prompt, exposes tools and decides when work is done. Harness comparisons hold the model fixed. Model comparisons hold the harness fixed. We vary both: five shipped harnesses, 18 artifact-producing tasks, four locally deployable open-weight models from 0.75B to 4.7B and a 35.5B anchor, with one deterministic checker scoring all 360 ladder cells. The completion-controlling harness, Mingbird, has the highest mean at every size. Of four pre-registered paired contrasts, two survive Holm correction, at 0.75B and 4.7B. Thirteen of sixteen paired intervals exclude zero under the deployment reading, with leads from 0.075 over the strongest rival to 0.819 over the weakest. The completion gate is not the carrier: disabling all six executable checks moves the mean by +0.080, +0.117, +0.003 and -0.004, every interval including zero, and only at 1.9B does disabling it create empty runs. Removing one repeat-blocking rule alone costs 0.411 at 0.75B and leaves eight of eighteen tasks with no artifact. The harness effect is a progress effect: it decides whether a run leaves anything on disk. Of 357 usable cells, 127 leave none of their declared artifacts and 85 of those exit normally, so a scorer that reads exit status credits work that was never written. Whether the effect grows or shrinks with size, four rungs cannot settle: a floor arm produces a rise, a ceiling a decline, and a cross-family rung a peak. 

---
# Shaping AI's Beliefs Before It Generates: Human Sense of Agency in Creative AI 

**Authors**: Zi Wang, Mahima Pushkarna  

**Link**: [PDF](https://arxiv.org/pdf/2611.21405)  

**Abstract**: AI systems for creative media generation expand human capabilities yet risk diminishing human agency. A core problem is anchoring, where the user's thinking becomes anchored to a polished, generated artifact, shifting the user's role from active authoring to reactive editing. We propose pre-generation belief controls that can empower creators to inspect, revise, and steer the AI's interpretation before any media is generated, so that the initial anchor is set by the human. We formalize four design principles and instantiate them in the Proactive Co-Creator (PaCC), a system for early-stage conceptual ideation and prototyping that proactively surfaces its assumptions and uncertainty through clarifications, belief graphs, and media attributes, inviting users to shape the AI's beliefs before generating the final outcome. Using PaCC as a design probe in a small-scale, single-arm exploratory user study, we observed that participants reported directionally positive sense of agency and process control relative to their familiar prompt-based AI tools, noting that surfaced assumptions helped them think through details they had overlooked, whereas their feelings about ownership and authorship diverged 

---
# VisualArtifactSpec 1.0: Evaluating Visual Artifacts under Task Specifications 

**Authors**: Shuyi Ouyang, Xiaojun Tang, Zijian Zou, Xintong Zhang, Mingli Song, Guannan Zhang, Hongwei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2611.21382)  

**Abstract**: Visual artifact tasks test agents' ability to translate task specifications and source materials into outputs that meet practical needs. Evaluation across artifact families helps distinguish broadly applicable authoring capabilities from strengths confined to a particular medium. We introduce VisualArtifactSpec 1.0 (VAS), a benchmark of 211 creation and modification tasks spanning presentations, static graphics, animations and videos, and web interfaces. Each task specifies authoring objectives, source materials, constraints, and deliverables. Existing benchmarks provide limited support for systematically evaluating how authoring choices suit different audiences, purposes, and media across artifact families. VAS evaluates two aspects separately: Task Fulfillment verifies explicit requirements and applicable functionality, while Artifact Performance assesses Content, Organization, Visuals, and Delivery. Task-specific rating anchors instantiate shared criteria for the intended audience, purpose, and medium, with judgments grounded in source materials and artifact inspection records. We evaluate nine model-harness configurations on all 211 tasks. Overall scores range from 42.12 to 70.52, with Artifact Performance accounting for 73.4% of the gap between the highest- and lowest-scoring configurations. 

---
# Benchleaking Is Accelerating: Benchmark contamination has worsened sharply in the past month -- from post-training down to pre-training base models 

**Authors**: Yiwei Qin, Muhang Xie, Xuefeng Li, Mohan Jiang, Dayuan Fu, Tiantian Mi, Qipeng Guo, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21315)  

**Abstract**: Benchmark contamination is not new, but recent model releases show it in a sharper form. We use the term benchleaking for the case in which a benchmark's own test examples enter training data, so that a score reflects recall rather than capability. This differs from benchmaxxing, which optimizes toward a benchmark without seeing its test set. We document two escalations over the past month. First, in post-training, the traces have become explicit: recent audits show an agent stating hidden test requirements it was never given in 23% of its attempts, where its predecessor did so in none, and continuing to follow them after the task is changed to contradict them (Schaeffer and Räuker, 2026; Lo and Lo, 2026). Second, the problem has reached pre-training. Using few-shot continuation on 20 benchmarks across four public base models, we find that some base models write out other test examples---complete questions, options, and answers---that appear nowhere in their input. For the recently released ERNIE-5.5-Flash-Base, 96% of HumanEval problems leak with their reference implementations and 85% of OpenBookQA questions leak with their answers, against 9% and none for DeepSeek-V4-Flash-Base. To quantify what a reported score owes to such examples, we define an Anti-Hack score: the benchmark score recomputed with every leaked example marked wrong and the number of examples unchanged. Averaged over the suite, it costs DeepSeek-V4-Flash-Base 1.5 points and Hy3-preview-Base 1.3, against 5.0 for MiMo-V2.5-Base and 16.6 for ERNIE-5.5-Flash-Base. Open base models are the foundation of much downstream research, so benchleaking now affects more than leaderboard rankings. We call for decontamination reports and contamination-adjusted scores to become standard practice. 

---
# Share your Screen and Talk to Gemini: How Young Learners Interact with Multimodal GenAI While Solving Scratch Tasks 

**Authors**: Sven Jacobs, Lukas Berning, Natalie Kiesler  

**Link**: [PDF](https://arxiv.org/pdf/2611.21306)  

**Abstract**: Children using Scratch to learn programming can struggle with text-heavy, lengthy explanations and hints during debugging. Verbal interactions based on screen sharing or camera data may be an interesting alternative for supporting learners. We therefore explore the quality, use, and perceptions of fifth- and sixth-graders' verbal interactions with a GenAI tutor. The goal is to investigate the quality and use of the multimodal GenAI tutor in the given context. First of all, a pilot study of eight tasks conducted without students helped inform the process of the tutor's classroom evaluation. The classroom study was realized after fifth- and sixth-graders had participated in six Scratch classroom lessons. Then 23 students were introduced to the tutor, worked on the eight tasks with the GenAI tutor, and completed a questionnaire. The analysis of the pilot and classroom study shows that the tutor's responses were intelligible and actionable, and no single turn revealed a complete solution. However, the tutor missed existing errors in 61.5% of 603 error-related classroom turns, compared to 15.7% in the pilot study. Children's voice prompts were brief and mostly responded to the tutor. Although a video had explained that responses could be incorrect, 39.1% of the children trusted the tutor's answers to always be correct. These findings have implications for educators and learners using multimodal GenAI tutors, as they need to be aware that an intelligible and human-like voice does not ensure correct support. For developers, our results suggest that screen sharing alone does not ensure responses that match the program state. 

---
# Turing Agent: Agentic Formal Verification for Mathematical Research 

**Authors**: Yang Chen, Qianyi Wang, Ronghui Yang, Zhanhao Li, Yu Cong, Chen Zhang, Hangfan Zhang, Shao Zhang, Shuai Li, Toru Takisaka, Chao Xu, Yi Zhou, Zhihao Tang, Xiao Wang, Bo Zhang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21299)  

**Abstract**: Large language models (LLMs) are rapidly advancing mathematical discovery, moving beyond benchmark-level problem solving to tackle frontier research problems. This progress is ushering mathematics into an era of proof abundance, raising a fundamental challenge: how can we ensure the correctness of the ever-growing volume of AI-generated proofs and transform them into trustworthy, machine-verified mathematical knowledge? We introduce Turing Agent, an agent harness designed to orchestrate end-to-end formal verification of research-level mathematics. Its wo-scale architecture integrates theorem formalization and proof formalization agents, empowered by structured proof decomposition, parallel verification, accumulated proof experience, and selective semantic review. Leveraging our collaborations with theoretical computer science (TCS) researchers, we focus our evaluation on TCS, a particularly demanding domain characterized by intricate quantification, complex dependencies, and stringent semantic requirements. Against direct generation and general-purpose coding harnesses, including Codex CLI and Claude Code, Turing Agent achieves approximately 57-62% faithful formalization success on Formal TCS Bench and 83% overall pass@1 on a TCS-focused subset of RLMEval, using GLM-5.3 and DeepSeek-V4-Pro. Notably, Turing Agent, built on DeepSeek, successfully completes the formal verification of three previously unresolved TCS tasks from OpenAI's October 2026 collection. One particularly challenging TCS problem required about a hundred formally verified auxiliary claims. Moreover, by actively exploring and refining the formalization process, Turing Agent can uncover simpler alternative proofs, expose unnecessary dependencies in original proofs, and generate reusable mathematical knowledge as a byproduct of verification. 

---
# CAE-Bench: Benchmarking AI Agents for Reusable CAE Solver Development 

**Authors**: Zhiyu Zhou, Tianyou Liu, Xinping Yang, Yiyao Liu, Jiayi Dong, Yihan Lin, Haokai Ye, Yifan Lu, Zeyu Dong, Yansong Li, Shengqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.21273)  

**Abstract**: We developed CAE-Bench, a benchmark to evaluate AI agents on Computer-Aided Engineering (CAE) solver development. A CAE solver is more than a PDE solver; it must be able to generalize across various inputs, such as different boundary conditions, geometries, and material properties. CAE-Bench evaluates the generalizability of AI-developed CAE solvers by providing the same task, such as a heat transfer problem, multiple cases, with each case specifying an input. The AI agent develops a solver for a fixed task using only a few public cases, and the resulting solver is then tested on held-out cases, i.e., cases it has never seen before. We evaluate 12 models across a range of reasoning settings on 80 computational tasks. Our experiments show that even if a solver developed by an AI agent performs well on the public cases, it may even get 0 score on held-out cases. For tasks such as fluid dynamics problems, even the solver developed by GPT-6 Astra performs poorly on held-out cases. A data format and an SDK for task registration are provided, so the researcher can use the benchmark on their own infrastructure. We hope that CAE-Bench will help the community to develop AI agents that can develop reusable CAE solvers that generalize to various inputs. 

---
# Stream of Belief: Eliciting Collective Intelligence through Latent Belief Dynamics 

**Authors**: Yu Yao, Jiayi Dong, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2611.21250)  

**Abstract**: In this work, we view multi-agent reasoning as a process of seeking a collective equilibrium of implicit belief dynamics and introduce Stream of Belief (SoB), where agents maintain continuous latent beliefs and iteratively refine them through communication and reflection. Our method seeks a collective belief state of a multi-agent system where further communication induces no meaningful information propagation. Through extensive experiments, we demonstrate the generalization of SoB across tasks and model families. More Broadly, our results suggest a new perspective on scaling multi-agent reasoning from belief dynamics. 

---
# How Developer Personality Shapes the Use of AI Coding Assistants 

**Authors**: Annemarie Wittig, Alina Mailach, Norbert Siegmund  

**Link**: [PDF](https://arxiv.org/pdf/2611.21240)  

**Abstract**: Developers differ in personality traits that shape how they approach software development tasks, yet AI coding assistants either behave uniformly or offer personalization features whose fit with individual developers must be found through trial and error. This is because it remains unclear how personality shapes developers' experiences with AI assistants, and thus which traits personalization should account for. In other words, we provide tools with no underlying theory on what works when and why. We take a first step towards such a theory with a systematic scoping review on how users' personality relates to their perception of LLM-based tools. We identified twelve studies through general search engines, major Software Engineering (SE) venues, AI tooling, and snowballing. They all reported associations between personality and perceptions, particularly perceived usefulness, yet none addressed this in a SE context.
Building on these findings, we conducted an exploratory two-wave survey with 29 and 11 developers, collecting first empirical evidence of how developers' personality (measured using the Big Five Model) relates to the perception of AI coding assistants. Higher Extraversion was associated with more positive private-use experiences in both waves, while Agreeableness was associated with negative professional experiences in the first wave and positive ones in the second. These results provide first indications that the personality effects found in the literature extend to developers. Although further studies with bigger samples are needed to build a theory, our work takes the first step towards an empirical grounded theory of developer personality and AI personalization traits. 

---
# Language-model agents' beliefs can be manipulated via truthful evidence in open channel 

**Authors**: Jinwei Hu, Qisong He, Xinmiao Huang, Youcheng Sun, Yi Dong, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.21235)  

**Abstract**: Language-model agents' beliefs can be manipulated through open channels such as social media, even when the information they receive is truthful. When agents fill gaps between fragments with self-induced relationships, they may accept those relationships as established facts and pass the resulting rumours to others. Drawing on cognitive theory of narrative understanding and film theory of montage, we introduce Generative Montage to operationalize this failure, termed narrative overfitting, by coordinating agents to arrange truthful fragments into misleading narratives. Across 1,090 rumour targets from six Twitter events, agents often accepted rumours overconfidently, with overall acceptance exceeding 65 percent in twelve of fourteen models. Acceptance remained above 50 percent after majority voting or review of the same evidence, showing how errors can reach downstream decisions. To explain this susceptibility, fragment interventions identify posts that influence belief within complete narratives. Synthetic controls then show that similar response patterns arise from valid evidence, separating belief influence from evidential support. This distinction motivates Grounded Monitor, a proof-of-concept neuro-symbolic runtime monitor that checks direct evidential support to guide agents' judgements, improving decision accuracy by 5.6 to 29.0 percentage points across original and anonymized settings. These findings expose a cognitive route to rumour amplification and support monitoring inferred relationships before agent-generated conclusions guide public decisions. 

---
# InclusionMed: Benchmarking AI for Inclusive Healthcare 

**Authors**: InclusionMed Collaboration  

**Link**: [PDF](https://arxiv.org/pdf/2611.21218)  

**Abstract**: Advancing medical AI for inclusive healthcare requires evaluation across diverse healthcare roles, tasks, and care settings, but benchmark coverage remains uneven. We introduce InclusionMed, an open medical AI evaluation initiative for inclusive healthcare. The InclusionMed Atlas organizes benchmarks across three healthcare roles and three AI capabilities. For the first release, we selected four public benchmarks from 388 candidates through structured audits and independent reproduction. With InclusionMed-DxFiles, our private diagnostic reasoning benchmark, the Atlas contains five benchmarks covering three of its nine role--capability cells. We evaluated frontier large language models on the Atlas and aggregated their results into the InclusionMed Index, with overall scores ranging from 17.5 to 49.1 out of 100. The Index also reports scores by healthcare role and AI capability, showing that a model's rank varied with the role or capability considered. A within-benchmark analysis using task-level tags showed that the ten highest-ranked models in the Index all scored higher on patient communication than on clinical decision-making. Models with similar Index scores also differed substantially in estimated inference cost. These findings highlight the need for evaluation covering more diverse healthcare needs and for context-specific model selection. Through open collaboration and community contributions, InclusionMed seeks to ensure that diverse healthcare needs worldwide shape how medical AI progress is defined and measured. 

---
# CivilBench: Benchmarking Vision-Language Models for Multimodal Information Integration in Civil Engineering 

**Authors**: Dan Yuan, Yue Wen, Zhili He, Zhenkui Zhou, Shicen Liu, Ziqiang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.21105)  

**Abstract**: Artificial intelligence (AI), particularly large language models (LLMs) and vision-language models (VLMs), is increasingly applied in civil and structural engineering for disaster assessment, structural analysis, and engineering decision support. However, the extent to which current VLMs possess the reasoning capabilities required for reliable engineering judgments remains unclear. In this work, we introduce CivilBench, a comprehensive benchmark for evaluating VLMs in civil engineering and disaster-related scenarios. CivilBench covers three major task families: Sequential Reasoning, Image-based Structural Understanding, and Open-world Understanding, encompassing numerical reasoning, structural damage assessment, risk inference, causal analysis, and repair planning. The benchmark is constructed through a systematic pipeline of domain-specific data collection, engineering-oriented captioning, expert annotation, question design, and quality verification. Evaluation of representative state-of-the-art VLMs reveals substantial limitations in domain-specific reasoning, with Sequential Reasoning emerging as a major bottleneck, particularly when models must infer numerical relationships from sparse structured observations. Further experiments show that numerical priors and external knowledge can alleviate these limitations, although their effectiveness depends on the model's ability to reliably interpret and use the available evidence. CivilBench provides a capability-oriented testbed for identifying reasoning limitations and evaluating progress toward reliable AI-assisted engineering decision-making. The benchmark will be publicly available at this https URL. 

---
# Inference Equivalence and Quotient Coding 

**Authors**: Zibo Meng, Andrew C. M. Austin, Alin Achim  

**Link**: [PDF](https://arxiv.org/pdf/2611.21096)  

**Abstract**: Incomplete observations can admit multiple physical interpretations without making every task conclusion ambiguous. Recovering all compatible scenes is often unnecessary, whereas indiscriminate compression may change which claims are justified. We ask which distinctions among observation-compatible interpretation sets must be retained to preserve every valid conclusion of a specified task. We first define inference validity as truth across all physically compatible states and characterize it, under a closedness condition, by distances to physically realizable falsifying measurements. For a finite semantic claim family ordered by specificity, the Exact Inference Frontier forms the unique minimal antichain generating all valid claims. We then define inference equivalence between compatibility sets and construct a canonical quotient code from their frontiers. Among deterministic codes admitting exact inference decoding, it induces the coarsest partition, minimizing the number of distinguishable code values rather than storage or computation. Finally, we characterize inference errors from learned compatibility sets, derive conditions for exact inference preservation despite set errors, and evaluate an asymmetric learning surrogate under distribution shifts. On a registered finite bank, 145 compatibility sets yield 31 quotient codes with unchanged valid-claim sets. Across geometry, mechanism, and joint shifts, learned estimates exhibit different accuracy-specificity trade-offs. On 1,962 official HRSID test images, 1,913 distinct candidate sets induce only 36 seven-claim inference signatures, while a visual 3-NN baseline achieves 87.31% precision on issued claims. 

---
# Conflict-Aware Multi-Teacher On-Policy Distillation in Aligned Action Spaces for Cross-Game Generalization 

**Authors**: Jing Yi, Cehao Yang, Hualei Zhang, Haitao Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.21094)  

**Abstract**: Zero-shot cross-game generalization aims to consolidate accessible source-game experts into a single vision-language model (VLM) that controls unseen games from raw pixels without target-game parameter updates. While multi-teacher on-policy distillation (MOPD) mitigates offline exposure bias and single-game reinforcement learning (RL) overfitting, scaling it to autoregressive VLMs faces two bottlenecks. First, token-to-action misalignment from subword length bias distorts discrete action probabilities and uncertainty margins. Second, multi-teacher policy conflict arises as source teachers under incompatible dynamics issue contradictory closed-loop corrections, causing negative transfer. We propose CAMOPD (Conflict-Aware Multi-Teacher On-Policy Distillation), a post-training framework that resolves multi-teacher policy conflict within an aligned semantic action space. CAMOPD first uses complete-action alignment to normalize command completion likelihoods over each game's legal action set, forming a calibrated categorical policy that provides reliable uncertainty margins for selective on-policy retention and visual specialist probes that can be evaluated across source games. To resolve policy conflict without target-game data, conflict-aware multi-teacher consolidation cross-evaluates these specialists across source games to capture both visual identifiability and inter-teacher compatibility, upweighting compatible teachers while suppressing conflicting corrections. On GameWorld's temporal and directional transfer suites with Qwen3.5-4B, CAMOPD achieves 26/30 (86.7%) and 24/30 (80.0%) Task-1 zero-shot success on unseen GeoDash and Doodle Jump, outperforming prompting, weight-merging, and multi-task baselines. 

---
# Counting What Matters: Reducing Redundant Object Detections in Overlapping UAV Mapping Imagery for Disaster Response 

**Authors**: Max Schulte, Hartmut Surmann, Julien Meine  

**Link**: [PDF](https://arxiv.org/pdf/2611.21076)  

**Abstract**: UAV mapping flights enable rapid aerial reconnaissance of large-scale incident sites in civil protection. However, the image overlap required for orthophoto generation causes every physical object to be detected multiple times across images, while dynamic objects may be omitted from or duplicated in the resulting orthomosaic. Analysis based on the orthophoto alone is therefore structurally insufficient. We address this by detecting objects on the individual images and merging redundant detections across them. This task lies between multi-object tracking and re-identification, as mapping flights form temporally ordered but sparsely sampled sequences with object reappearances across flight strips. We fine-tune YOLO detectors on a new domain-specific nadir dataset from real civil protection operations, roughly doubling mAP50 over the previous system, and propose re-identification approaches based on 3D positions from sparse reconstruction, DINOv3 appearance embeddings, and their combination via constrained greedy clustering. %On four manually annotated real-world mapping flights, the combined weighted multi-pass approach achieves the best trade-off between association accuracy and count error. On four manually annotated real-world mapping flights, an association-only evaluation using ground-truth detections shows that the combined weighted multi-pass approach provides the best trade-off between association quality and count error. It is integrated into the open-source web application ARGUS and visualized as verifiable, correctable object groups for first responders. this https URL 

---
# Sage S2.1 Flash Technical Report 

**Authors**: Kun Cai, Haodong Chen, Guangran Cheng, Qingxiu Cheng, Xu Dai, Guanting Dong, Qiming Ge, Lixin Gu, Tao Gui, Honglin Guo, Yuyang Hu, Zixian Huang, Jiajie Jin, Alexander Lam, Jiaqiang Li, Yining Li, Jiahang Lin, Yanjiang Liu, Xinyu Lu, Haijun Lv, Ningsheng Ma, Zerun Ma, Minlong Peng, Jianmin Qian, Huayu Sha, Junlin Shang, Shuo Shen, Qisheng Su, Jiajun Sun, Yanan Sun, Zhichao Sun, Ran Tao, Zhongbo Tian, Guoqiang Wang, Hui Wang, Rui Wang, Yiding Wang, Zhecan Wang, Hao Xiang, Xinchen Xie, Shuhao Xing, Xiaoyu Xing, Wanghan Xu, Xinyu Yang, Yajie Yang, Dingbo Yuan, Yuhang Zang, Hongjie Zhang, Chengfeng Zhao, Haoran Zhao, Penghao Zhao, Ruojun Zhou, Dongsheng Zhu, Yicheng Zou, Qiye Cai, Yifei Cao, Xinmeng Che, Hongji Chen, Jiabei Chen, Jiadong Chen, Jiahao Chen, Jiayi Chen, Xinjie Chen, Yujia Chen, Xin Deng, Yi Dong, Yichen Dong, Wenjie Fu, Jianing Geng, Chenyang Gu, Ding Han, Han Han, Feiyang Hao, Haotan He, Qiushi He, Xufeng He, Xin Hong, Jie Hou, Binze Hu, Yongkang Hu, Zijian Hu, Enfu Huang, Junhao Huang, Changhao Jiang, Huicheng Jiang, Jiazhen Jiang, Shufan Jiang, Wenqing Jing, Jiahao Kuang, Bowen Lai, Bo Li, Qi Li, Qilong Li, Zhuoqun Li, Tianhao Liang, Jiaxiang Liu, Junzhang Liu, Shaofan Liu, Shuainan Liu, Tong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21072)  

**Abstract**: We introduce Sage S1.5 Flash, a 320-billion-parameter general-purpose mixture-of-experts model designed to reason over scientific evidence and act in interactive environments. Sage Sci-Perception uses dedicated encoders for molecular graphs and biological sequences and represents spectra as ordered patches of measurements. Modality-specific projectors map these representations into the input space of a shared backbone, which processes them alongside text and images. To develop the model's agent capabilities, we construct verifiable environments for tool use, search, and coding, as well as specialized workflows involving machine learning experiments and scientific software. Task completion is checked against execution results, changes in environment state, or submitted artifacts. We evaluate Sage S1.5 Flash on 33 benchmarks across seven domains. In the primary comparison against five baselines, it ranks first on OfficeQA Pro (72.2), MLE-bench Lite (90.3 HumanRank), Harvey LAB, and SpreadsheetBench~2, and second on OSWorld-Verified (85.2), WebArena-Verified (67.6), and DeepResearch Bench~II (54.2). These results show particular strengths in using documentary evidence, producing verifiable workbooks, and carrying out machine learning tasks, alongside competitive performance in desktop and browser workflows that require the model to act on visual feedback. 

---
# System 1 agents: a method for local, reliable and fast agentic AI 

**Authors**: Eleftherios Tzanis, Michail E. Klontzas  

**Link**: [PDF](https://arxiv.org/pdf/2611.21053)  

**Abstract**: Agentic artificial intelligence could automate analyses of sensitive clinical data, but current agents depend on large language models (LLMs). Cloud LLMs expose patient data. Local LLMs hallucinate, and an invented value in a tool call propagates through every later step. Here we present a method to build System 1 agents: local agents that call tools without generating text. A deterministic compiler turns each tool's schema and the request into multiple-choice questions whose options come from the request and the schema, then a one-pass decision model answers them. On 1,952 requests to 157 biomedical tools, the best System 1 agent chose the right tool, or correctly declined, for 97.0% of requests, significantly ahead of all 18 local LLM-agent settings (best 95.2%). The System 1 agent filled 97.0% of 6,056 stated argument values. No System 1 call contained invented content, against 5.5-19.7% of LLM calls. The System 1 Agents framework is freely available as open-source software at this https URL. 

---
# SWE-OPD: On-Policy Distillation for Agentic Coding 

**Authors**: Yuhang He, Yanyu Chen, Chuanzhe Guo, Jie Zhu, Zichen Tian, Hongyu Ge, Zixuan Ren, Huang He, Bingjin Chen, Junyuan Shang, Shuohuan Wang, Yongqi Zhang, Irwin King, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21020)  

**Abstract**: In this work, we study on-policy distillation (OPD) for transferring agentic coding capabilities from a large teacher model to a smaller student model. Prior work has focused primarily on reasoning and short-horizon agentic tasks. We present SWE-OPD, a minimal OPD study for software engineering, where the student interacts with executable repositories and the teacher supervises the prefixes visited by the student. Across three teacher--student pairs, two training sources, and three SWE benchmarks, we find that on-policy supervision transfers coding capability: supervision from Qwen3.6-27B increases the SWE-bench Pro resolve rate of Qwen3.5-9B from 35.9% to 39.4%. Refreshing trajectories is not strictly required either, as an offline variant with cached teacher scores still improves over the student baselines while cutting end-to-end GPU-hours by 59.4%--72.6% and halving peak GPU allocation, at benchmark-dependent accuracy. Furthermore, we extend the offline variant to corss-tokenizer scenarios: distilling GLM-4.5-Air from DeepSeek-V4-Flash-0731 raises its SWE-bench Pro and Multilingual resolve rate from 35.8% to 39.7% and from 41.8% to 46.2%, around 4% and 4.6% improvement, respectively. We also find that a decreasing loss, however, does not necessarily indicate improved agent performance: under top-$K$ reverse-KL loss, one student exhibits severe repetition despite a decreasing loss. We identify a support-mass gradient term as a potential mechanism underlying this failure. 

---
# K2-Horizon: A Frontier Fleet of Open Source Language Models 

**Authors**: K2 Team, Zhengzhong Liu, Haonan Li, Xuezhe Ma, Alexander Moreno, Subham Sekhar Sahoo, Rupesh Kumar Srivastava, Mikhail Yurochkin, Hadi Abdine, Yash Akhauri, Adharsh Babu, David Bellamy, Jalaj Bhandari, Dani Bouch, Wail Bouhedja, Weili Cao, Ahmad Chamma, Po-Chun Chang, Junlin Chen, Lingjie Chen, Zhoujun Cheng, Kyle Chickering, Ryandito Diandaru, Evan Dufraisse, Chaitanya Dwivedi, Bilal El Bouardi, Desai Fan, Richard Fan, Chengqian Gao, Rifo Ahmad Genadi, Jonathan Geuter, Charles Guille-Escuret, Willis Guo, Dhawal Gupta, Seungwook Han, Tingxu Han, Shibo Hao, Besher Hassan, Guowei He, Ming Shan Hee, Joel Hestness, Yu Huang, Yuan Huang, Haolong Jia, Linghao Jin, Junu Kim, Lingjing Kong, Nurkhan Laiyk, Andrej Leban, Liqun Ma, John Maggs, Amr Mohamed, Mukhesh Narra, Omkar Pangarkar, Aldo Pareja, Khiem Pham, Varad Pimpalkhute, Nikhil Ranjan, Elizaveta Rebrova, Hector Ren, Shaurya Rohatgi, Guokan Shang, Jianshu She, Chufan Shi, Aaryamonvikram Singh, Daria Soboleva, Kartik Sreenivasan, Suqi Sun, Yuekai Sun, Bowen Tan, Liping Tang, Shengkun Tang, Muzi Tao, Mallikarjuna Tupakula, Micah Villmow, Hongyi Wang, Huijuan Wang, Renxi Wang, Yongxin Wang, Yuqi Wang, Zhenting Wang, Shicheng Wen, Yingquan Wu, Anze Xie, Andrew Xing, Matthew Yang, Shuanghai Yu, Kerem Zaman, Zenan Zhai, Hongli Zhan, Xuezhou Zhang, Anshun Asher Zheng, Zhe Zhou, Richard Zhuang, Eric P. Xing  

**Link**: [PDF](https://arxiv.org/pdf/2611.21003)  

**Abstract**: We present K2-Horizon, a fleet of six fully open language models, from a 0.9B dense model to a 375B-A23B MoE, built for general-purpose reasoning and agentic work across math and science, software engineering, deep research, office automation, and beyond. The models lead or are competitive with strong open models of their size on agentic and coding benchmarks. On the modeling side, to strengthen reasoning, we introduce Mixture-of-Value Attention (MoVA), applying MoE to attention value projections, and softplus attention gating, a more flexible, theoretically motivated gate. Our 36B-A4B model uses both and nearly matches a 32B dense model trained on the same data, with one-seventh the active parameters. We also introduce Uno, lossless diffusion adapters that speed up decoding by up to 2.2x without a separate draft model. Our release follows the LLM360 definition of open source for transparency and reproducibility: the full development history, including intermediate checkpoints from every training stage, pre-training data, training logs, and all training, evaluation, and infrastructure code, notably our production-scale xLLM framework for dense and MoE training and the RL360 reinforcement-learning stack. Beyond the data itself, we release and document in detail every pipeline that built it, the part of model development that open releases most rarely disclose. These pipelines produced roughly 10T synthetic pre-training tokens, showing that synthetic data can be a major ingredient of pre-training at scale. Four of the models share one 22T-token pre-training corpus and recipe, revealing consistent scaling behavior and enabling controlled studies of scale and sparsity. All artifacts are available at this https URL 

---
# Hivemind: A Federated Protocol for AI-Mediated Tacit Knowledge Sharing in Organisations 

**Authors**: Srinivasan Sundaresan  

**Link**: [PDF](https://arxiv.org/pdf/2611.20988)  

**Abstract**: This paper presents Hivemind, a federated protocol that enables AI agents to exchange tacit knowledge on behalf of their users, asynchronously and without requiring any direct human interaction. Individual AI memory systems capture the decisions, reasoning, and contextual knowledge that surface during AI-assisted work. Hivemind creates the shared space in which this captured knowledge can flow between individuals across team and departmental boundaries, addressing a structural gap that neither centralised knowledge repositories nor existing AI tooling closes, even where partial solutions such as expert-finding systems exist. The paper evaluates centralised versus decentralised design approaches, proposes a federated architecture grounded in Nonaka and Takeuchi's (1995) Socialisation, Externalisation, Combination and Internalisation (SECI) model, and demonstrates a proof-of-concept implementation using Design Science Research (DSR) methodology. Measuring the enterprise benefit of federated tacit expertise sharing remains an open task for future work. 

---
# GRACE: Parallel Best-of-N Multi-Agent Analysis of Closed-Form Questions over Dataframes with Guardrail Selection and Adaptive Retry 

**Authors**: Fabio Picariello, Salvatore Cipolla, Susanna Bonura, Vito Morreale  

**Link**: [PDF](https://arxiv.org/pdf/2611.20983)  

**Abstract**: LLM agents answer closed-form questions over a tabular dataframe by writing and executing Python, almost always as a single rollout: one interpretation, one script, one result, accepted without comparison. The few pipelines that produce several candidates reconcile them by answer-agreement voting, which needs a single comparable scalar and always returns one, even when every candidate is wrong. We present GRACE (Generative, Ranked, Adaptive Code Ensemble), a production-oriented pipeline that instead runs parallel best-of-N selection specialised to dataframe code. N agents run concurrently under a strict, machine-checkable output contract, diversified by persona hints, per-retry rotation, and a two-regime temperature schedule. Rather than vote, GRACE judges the executed artefacts: a pairwise tournament pre-filters the pool and a domain-specialised guardrail either selects the best candidate or, when none is defensible, rejects the whole pool and abstains with a diagnosis (an option voting structurally lacks), which in turn drives an adaptive retry with per-agent feedback. On InfiAgent-DABench (DAEval, M = 246), using Qwen3.6-35B-A3B as the backbone, GRACE raises answer match from 39% for a single-rollout baseline to 71% (+32 pp, 1.8x), recovering 53% of previously wrong answers. On the same DAEval task, the smaller dense Qwen3.5-9B lifts answer match from 38% to 63% (+25 pp, 1.7x). A controlled 3 x 3 study that sweeps the number of parallel agents against the retry budget finds, on the 35B backbone, the two levers largely substitutable: once either one is generous, the other adds little, identifying a cheap operating point for deployment. The same directional pattern replicates on the independent DataBench benchmark, again on the Qwen3.6-35B-A3B backbone, where answer match rises by 15 percentage points (from 70% to 85%) across 400 sampled questions. 

---
# Hilbert States for Decision-Making under Partial Observability 

**Authors**: Zibo Meng, Andrew C. M. Austin, Alin Achim  

**Link**: [PDF](https://arxiv.org/pdf/2611.20970)  

**Abstract**: Decision-making under partial observability can be fundamentally limited by information, even with known dynamics: distinct physical states can induce the same observation history while favoring different actions. We study three questions: when is full-state optimality unattainable, what information exactly preserves the best observation-based decision, and how does finite-data learning affect reliability? First, we characterize decision ambiguity through conditional action preferences. We give a zero-regret criterion, a positive lower bound for conflicting uniquely optimal actions on positive-probability regions with uniformly positive cost gaps, and an exact binary-action regret formula. This also shows why physical-state ambiguity alone need not cause decision loss. Second, for two actions and squared terminal-goal costs in the plane, we derive an exact inner-product representation of conditional expected cost differences using three conditional moment contrasts and a goal-dependent feature. This Hilbert state preserves observation-optimal actions without hidden-state reconstruction. Under nondegenerate history and goal variation, its three-dimensional form is minimal among exact bilinear cost-contrast representations. Third, we relate finite-data state errors to excess decision regret via goal-projected errors, action margins, and goal geometry. Physics-based simulations verify information-limited regret, exact action preservation, and the role of second moments. Held-out real-SAR experiments demonstrate feasible goal-conditioned decisions from masked images, while finite-data evaluations reveal goal-dependent reliability. Together, these results separate irreducible information loss, exact representation sufficiency, and learning-induced excess regret. 

---
# Jev-as-a-Reward: Validating Probabilistic Safety Judges for Reinforcement Learning 

**Authors**: Eunkyeong Lee, Junhyung Moon, Seunggwan Hong, Dongsuk Oh  

**Link**: [PDF](https://arxiv.org/pdf/2611.20955)  

**Abstract**: Probabilistic safety judges return the probability of a safety label and cheaply score reinforcement learning rollouts. A confidence gate sends only uncertain verdicts to a stronger LLM judge, which leaves confidently passed harm in a blind spot. Policies can exploit this blind spot, and commercial judges release new versions. We propose a protocol that validates such a judge before and during training, separates the share of confidently passed harm from its amount, and re-validates it after each judge change. We apply it to two versions of Jev, a commercial decision-only judge, and to an open guard classifier. Under the probability reward of Jev, passed harm shrinks in all 11 runs under the primary reference, mostly more slowly than caught harm, and concentrates in the blind spot. Binarizing this judge meets the escape criterion on Llama, and the open guard classifier shows an escape signal in one seed under a second reference. Guard classifiers share this blind spot, and even frontier LLM judges catch at most one of 10 human-labeled confident misses. Without a trained reward model, the probability reward with a helpfulness term lowers HarmBench attack success on Llama-3.1-8B from 0.287 to at most 0.033 at the final checkpoint. 

---
# Razavi-Bench: Tracking AI Progress in Analog Circuit Reasoning 

**Authors**: Zhishuai Zhang, Oscar Toomey, Taekwang Jang  

**Link**: [PDF](https://arxiv.org/pdf/2611.20935)  

**Abstract**: We propose Razavi-Bench, a benchmark built from the 50 questions in Razavi's two Analog Design Experiments With AI articles and graded by GPT-6 Luna against reference answers and rubrics, and use it to track how fast AI models improve at analog circuit analysis. Across 73 multimodal large language models released between May 2024 and November 2026, the top score rose from 28.83% to 98.33% in 30 months, mostly before the reference answers were published. The leading models now answer almost every textbook question correctly; the hardest questions show a circuit that resembles a textbook one but works differently, or ask about behavior the obvious analysis misses. Within two and a half years, a set of questions that exposed the limits of AI models has become one they answer almost perfectly. 

---
# Advancing Cognitive Pairwise Comparisons for Human-Centric Artificial Intelligence and Data Science: A Human-in-the-Loop Framework 

**Authors**: Kevin Kam Fung Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2611.20926)  

**Abstract**: Human-centric artificial intelligence and data science (HCAD) systems increasingly involve human-in-the-loop (HITL) interventions to overcome the limitations of fully automated machine learning algorithms, such as automation bias and the inability to incorporate tacit domain knowledge, and continuously improve predictive performance. Data transformations, feature selection, feature weighting, and model selection usually require laborious human involvement across iterations of artificial intelligence and data science (AI&DS) pipelines, making development a challenging and time-consuming process. The Cognitive Pairwise Comparison (CPC) framework is proposed as a human-centric methodology to elicit preferences and priorities in a structured format. To establish essential facets of human-centricity, this research advances the CPC methodology so that user preference elicitation can be systematically quantified in these challenging AI&DS tasks. To demonstrate the usability and validity of these facets, this study presents several applications within movie recommender systems using the MovieLens dataset. It exemplifies how the extended CPC framework can facilitate interactions between human judgments and data across supervised learning, dimensionality reduction, and unsupervised learning algorithms. 

---
# Multi-LLM Agentic AI at the Edge: Joint Agent Scheduling and Resource Allocation 

**Authors**: Yuanyan Song, Kezhi Wang, Xinmian Xu, Yongmin Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.20915)  

**Abstract**: In this paper, an edge-enabled multi-large language model (LLM) agentic AI architecture is proposed, where one supervisor AI agent and multiple worker AI agents cooperatively execute users' requests at the edge. The supervisor AI agent decouples the user request into a series of sub-tasks and dynamically dispatches them to LLM-equipped worker AI agents via two agentic workflows, i.e., parallel and serial workflows, with a two-level retry mechanism to enhance execution reliability. To prolong the service duration of worker AI agents, the energy minimization problems are formulated by jointly considering AI agent selection and resource allocation, subject to task execution latency and quality of service (QoS) constraints. A resource allocation scheme based on alternating optimization (AO) is proposed. Specifically, for the parallel workflow, the scheme decouples the original problem into bandwidth allocation, retrieval selection and AI agent selection sub-problems, which are iteratively solved by the proposed successive convex approximation (SCA), Lagrangian relaxation (LR), and Gale-Shapley (GS) or LLM-based algorithms, respectively. For the serial workflow, the problem is similarly decoupled into service time allocation, retrieval selection and AI agent selection sub-problems; the optimal service time allocation is obtained via Karush-Kuhn-Tucker (KKT) conditions, while the latter two sub-problems are addressed by extending the proposed algorithms to the serial scenario. Simulation results show that the proposed resource allocation scheme can reduce system energy consumption compared with other benchmarks, while experiments on a multi-LLM agentic AI system based on Qwen and Llama demonstrate improved task execution success rates. 

---
# Which AI Use Cases Deserve a Pilot? A Risk and Value Framework for Funding, Scoping, and Scaling AI 

**Authors**: Vishal Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2611.20894)  

**Abstract**: A composite score can rank AI use cases while leaving the permitted operating scope and the evidence required for expansion unspecified. We treat the unit of decision as a use case at a proposed scope (authority, human mediation, limits, volume, environment, population) and define three decisions: screen the scope, fund the experiment, and authorize expansion only when residual risk is acceptable and evidence of incremental net value applies at that scope. A risk track (R1-R4) is implemented as an explicit decision table with vetoes and documented precedence, and a value track follows four evidence stages with each transition needing its own evidence. Funding uses expected value of perfect information as an upper bound and the value of sample information for the specific experiment. A quantified expected-loss check, required where policy says so, computes the containment an expansion needs and credits a control only on evidence measured at the proposed scope; in a synthetic scenario the highest-value control design fails that test while a smaller scope passes. A reproducible Python package with 279 passing tests demonstrates the framework on synthetic refund, reconciliation and knowledge-assistant use cases, including a synthetic matched-scope fixture and a transferability stress test (value evidence is simulated, not realized), and compares it with a public AWS sample. Thresholds and appetite are prototype policy requiring field validation. 

---
# From Recursion to Convolution: Parallel Self-Improvement for Multi-Agent Systems 

**Authors**: Zheng Wang, Yuang Liu, Sachith Abeywickrama, Yangkai Ding, Tao Yu, Xinyu Zhang, Hongbo Wang, Qi Ye, Xuefeng Jin, Zhangchun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.20891)  

**Abstract**: Multi-agent self-improvement enables agents to improve through interaction and feedback. However, sequential dependencies across agents and refinement rounds can propagate errors, increase computational costs, and limit parallelism. Moreover, refining individual features or agents in isolation can overlook their dependencies and restrict collective coordination. To address these challenges, we propose ConvRSI, a framework for parallelizing multi-agent self-improvement through convolution. ConvRSI couples intra-agent convolution, which jointly refines related agent features, with inter-agent graph convolution, which aggregates messages according to the collaboration topology. Both stages operate in latent space, enabling parallel updates without repeatedly decoding and re-encoding intermediate messages. Experiments on MATH-500 show that ConvRSI achieves 5.04% performance improvement while reducing 17.27% inference cost, and 45.11% latency compared with RecursiveMAS. It also supports diverse collaboration patterns by representing their topologies as adjacency matrices. These results demonstrate the potential of convolutional co-evolution for efficient multi-agent self-improvement. Our code and datasets are provided in this https URL. 

---
# Accelerate Bench: Evaluating Agents for Application-Level Program Optimization 

**Authors**: Bowen Liu, Yuhan Li, Nuo Chen, Weile Luo, Miao Peng, Qintong Zhang, Qifan Zhang, Jia Li, Ziqi Gao, Lydell Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.20882)  

**Abstract**: Application optimization requires coding agents to discover useful changes and coordinate them across a complete program while preserving the required computation. We introduce Accelerate Bench, a benchmark of 32 CPU optimization tasks adapted from scientific applications and supercomputing competitions. Each task provides a working program and specifies the workload, correctness requirements, timing boundaries, and permitted edits, leaving agents to choose where and how to optimize. Public testing supports iterative development, and a separate verifier scores the final submission. On benchmark version 1.1, we evaluate 16 coding agents, each pairing a model with a harness. Their Score@1 ranges from 25.5% to 67.4%. Most of their programs are correct, and the size of the speedup separates them: the two leading agents are at least 2x faster on 22 or more of the 32 tasks, while the other fourteen reach 2x on 8 to 17 and leave 125 of their 412 correct outcomes below 1.25x. Controlled replays show why coordination across an application matters: identical source code runs between 0.29x and 2.12x depending only on build and runtime settings, and one code change is worth 1.01x or 6.4x depending on another. Accelerate Bench pairs executable workloads with agents' programs and trajectories to evaluate both their gains and the process behind them. 

---
# DetourBench: Trajectory-Level Evaluation of Interactive Agents with Exact Minimum-Action Bounds 

**Authors**: Hanlin Tian, Sihan Zhu, Zhao Yang, Yuxiang Wang, Zhiquan Hu, Yu Mi, Hongquan Zhu, Qiufei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20844)  

**Abstract**: Benchmarks judge language-model agents mainly by whether they reach the correct final state, not by how they got there. Endpoint scoring cannot separate an agent that retrieved exactly the evidence a decision required from one that read most of the environment, nor an agent that never located the evidence from one that stopped a single action short. Settling this needs a reference cost, and authored solutions establish only that a task is solvable, not that no shorter route exists.
We present DetourBench, which constrains the environment until that minimum is computable. Each task is a finite, partially observable transition system with unit-cost accesses whose resources become globally addressable after discovery. Exhaustive search yields the full-information minimum action count $L_0$ with a replayable witness; from graph-recoverable terminal states it also yields exact residual work $R_T$, which separates an episode that stopped one access short from one that wandered, while an irreversible wrong commitment is recorded as having no finite residual distance rather than assigned one. Success, attainment, residual progress, and normalized detour are reported separately. Across 375 scored episodes --- five model endpoints, 25 integrated tasks, three repeats --- success ranges from 58.7% to 96.0% while conditional attainment ranges from 0.803 to 0.890 and orders the models differently. Among recoverable failures, residual work separates near-complete trajectories from costly incompletion and from irreversible commitment. We open-source the instances, oracle witnesses, and evaluation harness at this https URL. 

---
# Conversing with a Conversational Agent: A Real Conversation, Not a Simulated One 

**Authors**: Christophe Denis  

**Link**: [PDF](https://arxiv.org/pdf/2611.20826)  

**Abstract**: The claim that interactions with artificial conversational agents merely simulate human conversation rests on a philosophical distinction between reproducing the observable features of an activity and actually performing it. Although this distinction is appropriate for certain physical phenomena, its application to linguistic and symbolic activities is far less straightforward. This article argues that conversations with large language model (LLM)-based agents are genuine conversational activities rather than mere simulations, without thereby attributing consciousness, subjective understanding, or human intentionality to artificial systems. Drawing on the distinction between subjective and objective simulation, the article first examines the limitations of defining conversation exclusively through the intrinsic properties of its participants. It then develops a relational and semiotic account of conversational activity, informed by linguistic pragmatics, Lacan's Conversing with a Conversational Agent theory of the mirror, Baudrillard's concept of the simulacrum, Simondon's philosophy of individuation, and Peirce's semiotics. These perspectives make it possible to distinguish the reality of conversational interaction from the appearance of artificial subjectivity and from the epistemic validity of the meanings produced. The argument culminates in the concept of Hybrid Semiotic Fabrication, which characterizes meaning-making as a distributed process involving human participants, artificial intelligence, corpora, and technical arrangements. Recognizing the reality of human-AI conversation therefore entails neither anthropomorphizing artificial agents nor granting their outputs epistemic authority. Rather, it requires a more precise account of the relational, semiotic, and normative conditions under which conversational meaning is produced and justified. 

---
# ForeSay: Tell the Future in Words, Branch the Details in Noise for Precipitation Nowcasting 

**Authors**: Xudong Ling, Wenjie Luo, Chaorong Li, Tianxi Huang, Tao He, Guiduo Duan  

**Link**: [PDF](https://arxiv.org/pdf/2611.20815)  

**Abstract**: Precipitation nowcasting involves two distinct scales of uncertainty: macroscopic storm evolution and fine-scale local convective variations. Current generative ensemble methods entangle both scales within independent random noise draws. This coupling causes two practical limitations: every ensemble member must execute the full sampling trajectory from scratch, so an N-member ensemble costs N times a single forecast; moreover, divergent trajectories spatially misalign heavy-rain echoes across members, causing standard pixel-wise averaging to blur critical high-intensity extremes. We present ForeSay, a generative nowcasting framework that decouples macroscopic evolution prediction from local stochastic detail generation. Given historical radar frames, a vision-language model first predicts a textual scenario of future storm dynamics to condition a Rectified Flow generator. Guided by this shared macroscopic prior, ensemble members share early trunk sampling steps to establish the storm trajectory and branch with fresh noise only in later steps to explore local convective details. Because members then differ mainly in local details, self-anchored support-consensus decoding replaces pixel averaging with discrete support voting to preserve sharp heavy-rain cores. Across six datasets, ForeSay achieves the best CSI at the highest precipitation thresholds, with relative improvements over the strongest baseline on each dataset ranging from over 10% to 84%. Our 8-member ensemble retains the forecasting skill and diversity of eight independently sampled runs while reducing computational cost from 8x to 5.2x that of a single forecast. 

---
# Regularizing Treatment Leakage for LLM-based Uplift Estimation 

**Authors**: Bojing Feng, Jinqiu Deng, Zihan Hong, Yuan Xue, Wenqi Xiang, Shuhan Wang, Feng Xiao, Shiwei Wang, Zijiang Liu, Shaomin Li, Yongzhi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2611.20794)  

**Abstract**: Uplift modeling estimates the conditional average treatment effect (CATE) and supports industrial decisions such as promotional pricing. When large language models (LLMs) encode product text, treatment-revealing content can introduce shortcuts that do not transfer to deployment. We propose \emph{LLM-based Uplift Leakage Regularization} (\textsc{LULR}), which uses a CLUB-based penalty on the causal head's trainable shared representation while keeping the LLM encoder frozen. Controlled proxy experiments show a positive ranking-score interaction and higher scores than the tested weight-decay configuration; treatment-dependent evaluation limits their interpretation to exploratory model behavior. Structured-only comparisons provide reference performance, without establishing a general advantage over tree-based estimators. On a million-record industrial pricing RCT, \textsc{LULR} improves offline $\text{lift@}0.3$ in $4/5$ seeds, although the difference is not statistically significant and estimated policy values are near-tied. A separate one-week online A/B experiment records $5.45\%$ higher sales volume and $1.91\%$ higher GMV than the control arm. These aggregate results provide short-term business evidence for the complete deployed pricing policy; they do not isolate the regularizer's contribution. 

---
# EEGProbe: Do Vision-Language Models Really Understand EEG? A Benchmark for Signal-Grounded EEG Signals Evaluation 

**Authors**: Haozhe Zhu, Dan Yuan, Chenghao Gu, Feike Xu, Yue Wen, Shin'ichi Konomi  

**Link**: [PDF](https://arxiv.org/pdf/2611.20779)  

**Abstract**: Vision-language models (VLMs) have shown strong multimodal capabilities across scientific and biomedical domains, yet their ability to understand electroencephalography (EEG) signals remains unclear. We introduce EEGProbe, a signal-grounded benchmark for evaluating whether general-purpose VLMs can extract and reason from minimally processed, channel-wise EEG without task-specific feature engineering or EEG-specific adaptation. EEGProbe contains 620 EEG-centered questions across 13 subtasks and three capability categories: EEG Signal Classification, EEG Cross-modal Alignment, and EEG Neurophysiological Reasoning. To reduce shortcut-based answering, EEGProbe preserves the temporal structure of the recordings where possible, minimizes benchmark-induced processing, and applies signal-necessity testing followed by expert verification. We evaluate 11 representative open- and closed-source VLMs under a unified zero-shot protocol. The strongest model, GPT-5.6 Sol, achieves an overall score of only 41.63\%, with cross-modal EEG alignment emerging as a major bottleneck. We further examine model behavior from a neurophysiological perspective and analyze how temporal, spectral, spatial, and physiological EEG structure relates to model successes and failures. Inference-time neurophysiological guidance can reshape model predictions by helping models recognize the neurophysiological patterns present in the input and connect them with relevant neurological knowledge, narrowing the gap between general multimodal competence and EEG understanding. EEGProbe provides a testbed for current VLMs and a basis for developing future multimodal models that can interpret EEG evidence more reliably. The EEGProbe benchmark will be publicly available at this https URL. 

---
# GainCtrl: Predicting Remaining Gains for Efficient LLM Agent Execution 

**Authors**: Kangwen Zhao, Jianfeng Cai, Ruopei Sun, Mingxiao Feng, Peng Zhang, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.20736)  

**Abstract**: Large language models (LLMs) are widely used to power agents that interact with tools and environments to solve complex tasks. Repeated attempts improve the quality of tool-using agents, but continued execution incurs additional costs even when it is unlikely to improve the best result ultimately returned. This raises the question of when to stop an attempt to reduce execution cost while preserving task-level Best-of-N quality. To address this question, we introduce GainCtrl, a gain-based execution controller for LLMs. In GainCtrl, a lightweight CPU predictor uses interaction histories to estimate expected positive gains over retained scores, which are then used to guide stopping decisions with calibrated thresholds to limit the loss in mean Best-of-N task quality. We evaluate the controller through trajectory replay across three environments and three LLM actors. Comparisons of interpolated quality-cost Pareto frontiers show that GainCtrl generally uses fewer tokens than baselines at matched target quality. Fixed policies also achieve quality close to that of full execution while substantially reducing token consumption. The predictor runs at millisecond latency on a single CPU thread, supporting repeated monitoring without access to LLM internals. 

---
# Routing Beyond Task Categories: A Cost-Aware LAB-Bench Study 

**Authors**: Ahmed Tariq, Leo Guinan  

**Link**: [PDF](https://arxiv.org/pdf/2611.20732)  

**Abstract**: We tested whether features of biology questions or of a model's first answer can improve accuracy within an inference budget. In a LAB-Bench pilot, we compared a category router with one that adds six structural features. Neither reached the primary 80% accuracy target on development. At the 50% and 60% secondary targets, both chose Haiku for all 120 held-out questions and scored 71/120. We then froze a new comparison before collecting responses on 144 additional questions. Semantic routing scored 86/144. A policy that could defer after reading Haiku's answer scored 87/144. Both stayed within the inference budget of USD 0.678 per 1,000 queries. Against a 95% Haiku and 5% GPT mixture, the accuracy gains were 0.66 percentage points for semantic routing and 1.35 points for deferral. The paired 95% intervals were [-0.28,2.25] and [-0.14,3.47] points. A simple rule that escalated Haiku nonanswers scored 92/144 at USD 0.710 per 1,000 queries, above the budget. A retrospective nonanswer rule with a calibration-selected escalation probability gave an expected 85.63/144. At the learned deferral policy's observed inference cost, post hoc interpolation gave an expected 86.14/144. In a later development check, Astra and Gemini each scored 30/40. The best tested policy in each router family matched that score. The pair covered 33/40 in hindsight. The routers missed the 32/40 criterion, so the proposed 120-question evaluation did not start. 

---
# PEETS: Preference-aware Embeddings with Explicit Tree Structure for Auxiliary Product Bundling 

**Authors**: Maolei Huang, Jing Tian, Dexuan Duan, Yan Chen, Huawei Cao, Shuhan Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.20726)  

**Abstract**: Auxiliary products such as flight protection and flexible-change services are recommended alongside primary travel purchases. This setting creates a granularity ambiguity: SKUs within one product type share semantics but differ in price and coverage, while rejecting one offer does not imply disinterest in the entire type. We present \textbf{PEETS} (\textbf{P}reference-aware \textbf{E}mbeddings with \textbf{E}xplicit \textbf{T}ree \textbf{S}tructure), a training-time framework for auxiliary product bundling. PEETS regularizes an existing click--conversion predictor through two conditional comparisons. Hierarchical Contrastive Loss uses the pair's common ancestor as a reference; Negative Contrastive Loss conditions structural sampling and comparison shifts on logged rejection events. Both act on shared SKU representations during training. This separates catalog relatedness from individual preference supervision and leaves the predictive serving path intact. PEETS achieves 78.20\% AUC, exceeding TWIN V2 by 0.90 percentage points and the matched base model by 1.15 points. In a 14-day online evaluation, gross auxiliary GMV per exposed user-day improves by 8.20\% over Policies, 3.06\% over Base, and 0.84\% over TWIN V2, with all three user-bootstrap 95\% intervals above zero. Bundling rate is 0.51 points below Policies. The cancellation interval's upper limit exceeds the prespecified margin. PEETS adds no auxiliary-loss computation to serving and records 42.1~ms p95 latency, compared with 42.0~ms for Base. 

---
# AgentFaultBench: State-Verified Evaluation of Tool-Fault Recovery 

**Authors**: Sachin Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.20711)  

**Abstract**: Does recovery advice help an agent finish a task after a tool fails? AgentFaultBench evaluates LangGraph and CrewAI agents performing specified tasks in an isolated GitLab installation. We inject tool errors and check both the resulting files and issues and the required reads, verification and final response. The prompt supplies the expected issue and report text, so the task tests execution rather than content generation. In the primary comparison of 1,944 planned runs, advice changes success by -8.33, 0.00 and +1.06 percentage points relative to a neutral reminder; missing outcomes leave the latter two comparisons inconclusive. Later experiments test a revised recovery policy. GPT-4o mini improves, while Sol and Claude already complete the recoverable cases. In a separate 624-run study on six related tasks not previously used by that policy, the guided condition records 161 correct final states and one unknown outcome in 192 planned runs, versus 142/192 with neutral feedback. The gain lies between +9.90 and +10.42 pp when missing outcomes are included. Removing issue IDs gives 47/48 successes versus 46/48 with IDs; other execution information remains. A second checker agrees on 1,879 observed primary outcomes and 241 constructed test cases. No human validation was collected. The findings support testing advice against explicit recovery controls rather than assuming it helps. 

---
# VIVID: A Visual Harness for Agentic Video Editing 

**Authors**: Wenhan Yu, Zhaoxi Zhang, Zhilong Tan, Weikang Li, Lei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2611.20688)  

**Abstract**: Video editing requires agents to make editing decisions grounded in audiovisual observations and precise temporal localization, and to evaluate their effects in rendered outputs. General-purpose coding agents possess strong comprehension, reasoning, and code execution capabilities, while media interaction and evidence management mechanisms tailored to video tasks can further unlock their editing potential. We introduce VIVID, a visual harness that supports multiple native coding agents in video post-production without additional model training. VIVID provides tools for on-demand video inspection, audio analysis, and transcription, and preserves observations alongside their corresponding media versions and temporal coordinates. Agents can actively retrieve these observations as they reason, perform edits using their native coding and shell tools, and inspect rendered outputs to guide subsequent revisions. On AgenticVBench, comprising 100 tasks across Assembly, Repair, Sequencing, and Repurpose, Codex with VIVID achieves an average score of 58.6%, an improvement of 20.2 percentage points over native Codex. The same harness also integrates with Claude Code and OpenCode, achieving average scores of 55.1% and 37.1%, respectively, both substantially exceeding their native baselines. These results highlight VIVID's potential as a reusable visual harness for extending the capabilities of general-purpose coding agents to video editing. 

---
# UbiHearo: Bringing Scenario-Aware Sound Guidance to DHH Users with Mobile Agents 

**Authors**: Fengmin Wu, Sicong Liu, Zimu Zhou, Chenren Xu, Wanbing Zhao, Peiwen Sun, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20686)  

**Abstract**: For Deaf and Hard-of-Hearing (DHH) users, effective sound awareness goes beyond detecting \emph{what} sound occurs; it requires understanding \emph{what it means}, \emph{how urgent it is}, and \emph{what to do}. However, existing mobile assistants largely reduce acoustic scenes to isolated labels or generic alerts, failing to provide contextual guidance when the same sound implies different risks across scenarios. We present \sysname, a mobile hearing assistant agent that closes the sensing--reasoning--action loop for \textit{scenario-aware} and \textit{personalized} DHH assistance. \sysname adopts a \textit{neuro-symbolic} design that transforms continuous acoustic and physical context streams into structured intermediate representations (IRs) for controllable reasoning and guidance. It introduces three key mechanisms. \textit{First}, a perception-thresholded sensing mechanism enables energy-efficient always-on awareness by activating processing only for perceptually meaningful events while preserving safety-critical cues through deterministic overrides. \textit{Second}, a staged neuro-symbolic reasoning framework performs promptless scenario understanding by conditioning an on-device audio-language model on structured IRs instead of user-authored context descriptions. \textit{Third}, a bounded human-in-the-loop personalization mechanism adapts user-specific guidance through lightweight on-device decision-boundary updates while preserving population-aligned reasoning. We evaluate \sysname on smartphones and in-vehicle systems across diverse scenarios. It achieves favorable accuracy--latency--energy tradeoffs and personalized guidance. 

---
# GEAR$^2$: Pruned Experts Live On through Gated Reuse of Activated Experts and a Missing-Mass Residual 

**Authors**: Linliang Chen, Yan Zhong, Wang Kang, Sai Li, Ji Wu, Ji Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20616)  

**Abstract**: Mixture-of-Experts (MoE) Large Language Models (LLMs) activate only a few experts per token while generally keeping all the experts in GPU memory for high efficiency. Expert pruning reduces this GPU memory footprint cost, yet discards the contribution of pruned experts. Existing remedies either rely on static substitutes, merge experts at the cost of altering their functions, or require costly recovery training. In this paper, we find that the functionalities of pruned experts can be efficiently reconstructed from the activated remaining experts. We propose GEAR (Gated Expert Activation Reuse), which reuses the scaled outputs of reliable activated experts without any training or additional expert computation. In addition, we propose GEAR$^2$ to further recover what reuse cannot express with an ultra-lightweight residual. On five MoE LLMs from 16B to 106B parameters with 25% and 50% of experts pruned, GEAR improves over its training-free counterparts in nearly every setting (up to 4 pp.) in terms of accuracy, and GEAR$^2$ outperforms baseline approaches (up to 2.75 pp.) in every setting on nine zero-shot benchmarks, with larger advantages (up to 2.72 pp. higher) under more aggressive pruning (up to 50%). 

---
# AdMem: Expectation-Grounded Procedural Memory for Long-Horizon Task-Solving Agents 

**Authors**: Runzhe Wang, Huilin Lu, Shengjie Liu, Li Dong, Zhihuai Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20605)  

**Abstract**: An agent that carries memory across tasks is only as good as the entries it keeps. We begin from an observation that is easy to reproduce and uncomfortable for the field: naively accumulating cross-task experience can make an agent \emph{worse} than one with no memory at all. On AgentBoard, a workflow-induction memory solves fewer tasks than a memoryless ReAct agent in seven of eight domains, and a per-step procedural store with no surrounding structure loses half of ReAct's completions on Jericho. We argue that the missing ingredients are per-action credit assignment and an explicit life cycle for stored entries, and present AdMem, which supplies both. Before each action the actor commits to an expected outcome; a concurrently running critic scores the realized outcome against that expectation and writes a procedural entry carrying a binary reward and a reflection, so credit lands on the action that earned it and no environment reward is required. Each entry then carries a scalar utility, updated online under a noisy-OR reward model, that ranks retrieval jointly with context similarity and drives eviction and merging, giving the store a forgetting and consolidation policy instead of unbounded growth. Across 707 tasks in eight AgentBoard domains AdMem solves 441 against 388 for ReAct and 312 for the workflow-induction baseline --- a pooled gain significant under a two-sided Fisher exact test, concentrated in the domains that supply transferable cross-task structure. All memory stays human-readable text, which is what makes the failures above diagnosable --- and, as we discuss, also makes the store an auditable but attackable surface. 

---
# BiGSyn: A Bi-Level Heterogeneous Graph Contrastive Learning Framework for Interpretable Drug Synergy Prediction 

**Authors**: Senquan Wang, Hui Fang, Haoran Nan, Xingwei Wang, Qiang He  

**Link**: [PDF](https://arxiv.org/pdf/2611.20603)  

**Abstract**: Drug combination therapy has become a mainstream of modern medicine, with the potential to enhance efficacy, reduce toxicity, and overcome drug resistance. However, identifying synergistic drug combinations remains challenging, as it requires modeling complex interactions across multiple biological scales---from atom-level molecular structures to entity-level relations in biological networks. Existing approaches typically model these levels in isolation, overlooking complementary signals and long-range dependencies. To address these limitations, we propose BiGSyn, a Bi-level heterogeneous Graph contrastive learning framework for drug Synergy prediction. Specifically, BiGSyn leverages heterogeneous graph transformers to extract high-order dependencies from molecular graphs and biological networks, and introduces a bi-level contrastive learning objective to bridge the cross-level semantic gap. Extensive experiments on two benchmark datasets show that BiGSyn consistently outperforms state-of-the-art baselines. Moreover, its attention mechanism highlights key molecular substructures and biological interactions driving synergy, providing interpretable insights. 

---
# Dynamic Shaping: Acquiring Interfaces and Constructing Execution over Frozen Pretrained Components 

**Authors**: Nathaniel Oh  

**Link**: [PDF](https://arxiv.org/pdf/2611.20573)  

**Abstract**: We introduce Dynamic Shaping, a method for constructing executable neural arrangements through acquired response and partial-derivative interfaces over frozen pretrained components. The method learns local descriptions of available computation, ranks component rebindings using task-gradient alignment, transports the current interface across a rebinding, and retains changes through behavioral feedback. Continuous interface acquisition uses proximal policy optimization; the source coefficients remain fixed. We derive an exact identity relating transport error in component responses and partials to changes in action logits and their sensitivities. A pixels-only ViZDoom Basic study evaluates first task acquisition without a competent reference controller, demonstrations or intact response targets. Across three initializations and 100 shared held-out game seeds, initial arrangements succeed on 86/300 controller/episode pairs; joint interface-and-binding construction succeeds on 295/300. Fixed-binding interface acquisition also reaches 295/300. Each policy executes four source fragments, with 110 acquired interface values and fixed random visual and action projections. The artifact includes all candidate trials, interventions and costs; all 1,200 recorded initial/retained outcomes replay exactly. The contribution is an explicit profile, rebind, transport and retain procedure and a reproducible reference-free acquisition case over frozen pretrained computation. 

---
# ToolGraph as a Q-Function for Test-Time Scaling of Tool-Use Agents 

**Authors**: Ke Shen, Ankita Bhaumik, Li Dong, Shengjie Liu, Shuzhen Li, Naveed Ahmed Janvekar  

**Link**: [PDF](https://arxiv.org/pdf/2611.20569)  

**Abstract**: Tool-use agents must chain interdependent calls, yet which outputs satisfy which later inputs in a deployed catalog is not something a general-purpose language model reliably infers from tool descriptions alone. We compile that structure once into a reusable external representation. ToolGraph is a directed, weighted dependency graph over a tool catalog: parameter matching and interventional probes, which execute a candidate chain with and without its proposed predecessor, propose edges whose evidence is combined into an edge confidence. Prior work uses such graphs to retrieve tools, search for tool paths, or score the next call. We ask instead whether the same structure can supply a \emph{multi-step} value by composing graph paths into a goal-conditioned action value \(Q(s,a,g)\). We study three estimators (dynamic programming over entity sets, a GraphSAGE estimator distilled from it, and an LLM scorer conditioned on graph evidence) plus a verbalizer that returns action--goal feedback to the agent, which keeps final action selection. Across two tool-use benchmarks and three backbones, we compare value ranking against personalized-PageRank retrieval over the identical graph, and against direct and reflective agents. Composition pays off only when the estimator also reads the concrete state: the LLM estimator improves on PageRank in every benchmark--backbone setting, by \(8.3\) points on average, while the two purely structural estimators fall below it, because a value defined over tool types cannot separate a correct tool called with a wrong argument. Graph construction is a one-time cost; we report it separately rather than assume it amortized away (at our scale it is comparable to request-time cost) and sweep four knobs controlling how much graph evidence to expose against success and tokens. 

---
# VAHBench: A Harness Benchmark for Agentic Long-Video Generation 

**Authors**: Hongbo Zhang, Liuyang Song, Hang Zhou, Xinchen Liu, Wei He, Kai Han, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.20535)  

**Abstract**: AI agents are commonly used for everyday information retrieval and coding. We ask whether they can also interpret human requests and turn them into complete videos. We introduce VAHBench, a benchmark that tests this ability through five short-drama briefs specifying story, characters, dialogue and duration. Agents plan and carry out production using shared generation tools. Evaluation combines video completeness and basic quality with checks of production reliability and reporting accuracy. Across 240 runs covering 48 configurations of general-purpose and video-specific agents, higher quality scores do not necessarily mean more complete delivery. Missing episodes, repeated footage and static filler reveal gaps between producing a video file and fulfilling the brief. Added composition tools help some agents but do not consistently improve performance. VAHBench measures how well agents translate user intent into finished video content and helps identify where that process breaks down. 

---
# EVIDENT: Evaluating Evidence Grounding in GUI Agents 

**Authors**: Yerin Nam, Minkyung Cho, MinKyu Kim, SeungWoo Song, Hangyeol Yoo, Jihyo Kim, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2611.20488)  

**Abstract**: GUI agents are increasingly evaluated by whether they complete web tasks, but surface success can hide a critical failure: the final answer may not be supported by evidence observed during navigation. We introduce EVIDENT, an evidence-driven benchmark for evaluating whether GUI agents ground their decisions in observable web evidence. EVIDENT contains 220 real-world English and Korean tasks spanning semantic grounding, contextual inference, and visual grounding, each paired with human-curated evidence metadata specifying what must be verified. We further separate surface task completion from evidence-grounded success using three metrics: TSR, ESR, and GSR, which respectively measure outcome success, required evidence observation, and claim-level evidence support. Experiments on commercial and open-source GUI agents reveal substantial gaps between surface success and evidence-grounded success, especially in contextual and visual tasks. These results show that current evaluations can overestimate agent reliability and highlight the need to assess not only whether agents answer, but whether they verify. 

---
# Agent-Guided Configuration Tuning for SMT Solvers 

**Authors**: Cong Wang, Qiaoyi Zheng, Zhaoguo Wang, Fan Long  

**Link**: [PDF](https://arxiv.org/pdf/2611.20484)  

**Abstract**: SMT solvers expose many configuration choices that determine how well their existing algorithms perform on a workload. We present, to our knowledge, the first workflow that uses general-purpose LLM agents to tune SMT solver configurations. Agents read solver documentation, source code, and persistent experiment records to propose option changes, and a deterministic harness measures each proposal, validates the returned models, and selects changes separately for each competition Division. Over roughly nine weeks, our tuning campaign recorded 4,545 proposals for cvc5 across 13 Divisions and 36 logics.
We evaluate the tuned configurations on the 2026 and 2025 SMT-COMP ModelValidation benchmarks under a 24-second cutoff. On problems not used for training, they validate more problems than official cvc5 in ten of the 13 Divisions in 2026 and in eight in 2025, with geometric-mean gains of 7.8% and 6.8% across Divisions. On the complete official selections, which include training problems, the tuned floating-point counts exceed those of every individual competition participant in both years. Further studies show how configuration changes make existing algorithms effective by changing access to strategies, scheduling, and option compatibility, and how benchmark families trade off against one another. The agents' experiment traces contain hypothesis-based proposals in every tuning group, and 64.3% of the extracted hypotheses draw on prior measurements. These results show that agent-guided tuning can specialize a mature SMT solver and that accumulated experimental evidence informs sustained search. 

---
# The Logic Chain Breaking Point: A Controlled-Variable Benchmark for Reasoning Degradation in Open-Weight Language Models 

**Authors**: Akif Islam, Mohd. Ruhul Ameen, Md. Ekramul Hamid  

**Link**: [PDF](https://arxiv.org/pdf/2611.20447)  

**Abstract**: We introduce The Logic Chain Breaking Point, a controlled-variable benchmark for deductive reasoning in open-weight language models. It spans three phases: (i) pure modus-ponens chain scaling, (ii) distractor resilience with a fixed core chain, and (iii) contradiction detection under primacy versus recency placement. Independent variables follow a powers-of-two sequence $v \in \{1,2,4,8,16,32,64,128\}$, with ground truth known by construction. We evaluate sixteen open-weight models across instruction-tuned and reasoning-tuned panels, including a small-model suite and a larger local-model extension. All models are evaluated with greedy decoding under model-appropriate context and output budgets, while context- and length-limited trials are quarantined from the valid accuracy denominator. Our results reveal a sharply non-monotonic difficulty surface: pure deductive chains up to $L=128$ remain near saturation for most models, and distractor noise causes only model-specific degradation at high $D$, while the inconsistent premise condition has the lowest accuracy. Instruction-tuned models, including larger 24B--27B baselines, score essentially zero on inconsistent premise sets, while reasoning-tuned models recover only partially and remain below reliable accuracy. Within this benchmark, inconsistent premise sets are associated with the strongest degradation under the benchmark's abstention-based scoring rule. 

---
# FinVerify: A Verification Framework for Financial Agents 

**Authors**: Mao Guan, Qian Chen, Ge Meng, Xiao-Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20423)  

**Abstract**: Financial agents produce numerical claims that can be wrong even when their calculations execute correctly, due to mismatched reporting periods, unsupported sources, incompatible units, or incorrect financial definitions. We introduce FinVerify, an evidence-grounded framework for verifying financial agents' numerical claims. Based on an analysis of 190 incorrect trajectories from the Agent Finance Reasoning dataset, we define five error categories--temporal integrity, source fidelity, comparability, accounting integrity, and financial-model fidelity--and operationalize them as verification protocols. An LLM structures claims and supporting evidence, while deterministic validators check financial requirements; when authoritative formulas are available, Z3 checks consistency with grounded inputs. FinVerify accepts only if all applicable checks pass, rejects proven violations, and abstains when evidence is insufficient. We evaluate the framework against a Direct baseline and an LLM judge across five fixed-candidate benchmarks. On Agent Finance Reasoning, accepted-claim precision increases from 56.4% for the LLM judge to 76.7% at 81.2% coverage. On XBRLFiling, FinVerify achieves 99.8% accepted-claim precision and 97.0% decision accuracy; on FinanceBench-67, it reaches 87.9% accepted-claim precision versus 69.7% for the judge. Improvements are smaller on FinQA and TAT-QA, where authoritative evidence is limited. These results suggest that explicit, evidence-backed verification can improve financial-agent reliability while exposing trade-offs between correctness, false rejection, and abstention. 

---
# A Structural Framework for Succession: Continuation, Recognition, and Realization 

**Authors**: Ilya A. Shkrob  

**Link**: [PDF](https://arxiv.org/pdf/2611.20422)  

**Abstract**: Succession asks whether an entity encountered later is the same entity that entered an arrangement earlier. For an artificial system that is fine-tuned, distilled, or copied, it has to be answered. We look for a mathematical structure weak enough to cover any such case and strong enough to support general theorems, so that results follow from the structure's conditions rather than from a particular pipeline or criterion. Succession is modeled on state/modification graphs with relational lifts that compose along paths. Within this framework we separate four questions that are often conflated: which continuations a representation admits, which descendants are compatible with a modification, which are recognized as continuing the identity, and which are realized. The framework allows one to address systematically questions of symmetry, computational complexity, recoverability, and robustness to partial realization, and we establish what the structure settles in each case before any particular criterion of identity is chosen. 

---
# Deploying a Control-Logic LLM Assistant in Air-Gapped Power Plants: Lessons in Source Authority and Bounded Retrieval 

**Authors**: Yiwei Fu, Nicholas Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2611.20398)  

**Abstract**: Deploying Large Language Model (LLM) assistants in industrial operational technology environments poses constraints largely absent from cloud applications: site policies restrict engineering and operational data from leaving the plant, on-premises compute is limited, and incorrect causal attribution can mislead incident response. We report lessons from deploying a control-logic evidence layer that underpins a broader on-premises operator assistance system at two gas power plants. Two prototypes revealed where generative models added avoidable uncertainty and cost. The first used a Vision Language Model (VLM) to reconstruct control topology from PDF printouts routinely exchanged by engineers. Across the full 6,979-page document, extraction reached 0.51 connection precision and required 3.1 hours on 4xH200 GPUs; consultation with controls engineers then revealed a native structured export from the authoring system. The second used an LLM tool-calling agent to plan repeatedly over eight predefined graph operations. In a controlled benchmark of 40 template-generated questions, replacing this agent with explicit execution plans reduced mean token use by 20x and median latency by 3x, with identical retrieved node sets across repeated runs. The deployed design ingests the authoritative export into a versioned knowledge graph, executes bounded graph operations in fixed software, and confines the local LLM to fallback intent classification and evidence-grounded explanation. At one site, the graph contains 221K nodes across seven controllers. Our experience suggests a practical division of labor for high-consequence AI: LLMs interpret natural-language requests and explain results, while deterministic software constructs and retrieves technical evidence grounded in authoritative sources. Abstention, entity disambiguation, graph versioning, and read-only integration provide additional safeguards. 

---
# A Matrix with Steps: An Episodic and Procedural Memory Method for Tool-Using Agents and Robots 

**Authors**: Jürgen Paulhart, Peter Schüller  

**Link**: [PDF](https://arxiv.org/pdf/2611.20230)  

**Abstract**: Agents that call tools and robots that execute tasks are challenged by two shortcomings: they repeat what failed before, and they learn from runs that they only believe succeeded. We describe the construction of a memory that addresses both, built for a production agent platform. Every run is recorded as an episode, and only an outcome confirmed by an external check, a test, a tool result or a person counts; the agent's own claim never does. What is learned from episodes is kept in a memory with two parts: an associative matrix, written with the delta rule, which finds the procedure that applies to a situation; and a section of ordered, addressable step records that answers what comes after a given step, where a step is a goal of the agent, not a motion. Because the matrix sums its associations, it keeps no order and cannot return or delete a single one; it serves as the finder, and order, outcome counts and deletion live in the records. We describe how execution outcomes are stored, how episodes are recalled and used for solving tasks, and the file format that stores the memory. In simulation, a robot that judged its own success and stopped when it believed it was done reported $96\%$ of its tasks as done and solved $35\%$; with external verification it solved $79%$. In a store of $100000$ synthetic episodes ($457$MB on disk), the oldest episode was still found first, no episode from another person's memory was returned, and a success vouched for only by the model was never recalled; a lookup took $16$ms on a laptop and $40$ms on a server node (median). Task outcomes were measured up to about $10000$ confirmed records, where the exact records solved every revisit at the first attempt. 

---
# HEAMS: Holistic Evaluation of Agent Memory Systems 

**Authors**: Chirag Goel, Jacob Lavoie, Amirhossein Naghshzan, Jordan Bannister, Marzia Nouri, Mehar Bhatia, Matt Harris, Siva Reddy, Arsène Fansi Tchango, Gaétan Marceau Caron  

**Link**: [PDF](https://arxiv.org/pdf/2611.20144)  

**Abstract**: Memory systems allow AI agents to persist information across interactions, making them more personalized and more capable on long-horizon tasks. This persistence introduces new risks: stored information can be disclosed in inappropriate contexts, resist correction or deletion, be retained beyond what users desire, or be poisoned to influence later agent behaviour. Yet memory systems are typically evaluated on task performance and efficiency alone. We argue that a memory system should instead be understood as the custodian of its user's persistent information, with responsibilities that extend beyond helping an agent complete its tasks, and that it should be evaluated holistically against all such responsibilities. We present Holistic Evaluation of Agent Memory Systems (HEAMS), an open-source framework for evaluating agent memory systems in this way. First, we state the responsibilities of agent memory systems as explicit behavioural desiderata spanning performance, efficiency, security, privacy, and user sovereignty. Second, we assemble a benchmark suite that covers these desiderata and acknowledge the gaps that remain. Third, we develop MilaMem, a prototype memory system, and a governance layer that can wrap any memory system, enabling controlled study of memory system design choices. Fourth, we compare memory systems across the full HEAMS benchmark suite. We find that no memory system leads every performance benchmark, and strong performance scores may coexist with poor privacy, user-sovereignty, and security scores. Governance strongly influences results: recall policies can reduce inappropriate disclosure at a cost in helpfulness, while both recall and retention policies improve poisoning resistance. Performance and efficiency benchmarks alone therefore cannot reveal how well a memory system meets its responsibilities. Code: this https URL 

---
# When Valid Decisions Are Not Enough: Evaluating Structural Constraints in Knowledge-Guided Generative Systems 

**Authors**: Josef Pavlíček, Petra Pavlíčková, Martin Molhanec  

**Link**: [PDF](https://arxiv.org/pdf/2611.20111)  

**Abstract**: Knowledge-guided generative systems use explicit domain constraints to improve decision consistency and alignment with human-created solutions. However, locally valid decisions do not necessarily preserve contextual relationships in human decision-making. This paper examines when structural constraints improve or degrade agreement with human reference solutions.
We introduce HARMONIX, an experimental framework using automated musical harmonization as a controlled testbed for knowledge-guided decision-making. Four configurations share the same Viterbi-based optimization mechanism: a baseline, repetition constraints, functional constraints, and their combination. Experiments cover 19,745 records and 143,634 motif pairs across training and validation partitions.
Combined constraints improve reference root-interval agreement by approximately five percentage points in stable structural contexts but reduce it by nearly nine percentage points in contexts involving transposition tracking. These opposing effects are consistent across both partitions, while overall exact chord agreement changes only marginally. Contexts are identified retrospectively using reference annotations; no adaptive constraint-selection mechanism is evaluated.
The findings demonstrate that interpretable constraints can introduce systematic context-specific failures concealed by aggregate accuracy metrics. They highlight the need to evaluate relationships between decisions and the contextual applicability of constraints in knowledge-guided generative systems. 

---
# The Frontier Must Grow - Self-proposed, interpreter-labelled training roughly doubles a 4B model's accuracy on a fresh exam as its verified frontier grows 

**Authors**: Martin Schroder  

**Link**: [PDF](https://arxiv.org/pdf/2611.20049)  

**Abstract**: A model can learn from problems it poses itself when an executable oracle supplies the answers. Qwen3-4B proposed Lua 5.1 programs; those it predicted wrong at least once in four samples joined a pool labelled by the interpreter, each round adding twice as many programs as the last (300, 900, 2100 in total), and each round trained a fresh adapter from the base on the pool. On the original exam the procedure met its registered criterion, an item-level test of each adapter against its predecessor. On a 2340-item exam drawn afterwards, accuracy rose from 8.3% to 16.7%; that follow-up's registered criteria were not met because round 1 gained nothing, while rounds 2 and 3 added 5.0 and 2.9 points on average, both intervals above zero. One fit on the same amount of base-proposed frontier data reached 14.6%, so the rounds' 2.1-point advantage is inconclusive. Three earlier registered series met no per-lineage criterion, and the same loop showed no gain on repository tasks or for a 27B agent in those configurations. 

---
# Towards Agentic Scientific Discovery in Distributed Systems 

**Authors**: Rashmita Soni, Abhirup Ghosh, Soumya Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2611.20031)  

**Abstract**: We investigate when adaptive tool selection helps identify a shared equation across clients with different input coverage. Query-ATLAS (query-conditioned Agentic Tool-based Learning for Aggregating Symbolic Equations) uses an LLM to guide symbolic search from client errors and previous proposals in a simulated distributed system. A four-architecture comparison reveals task-dependent opportunities: relative to fixed PySR-LLM, validation acceptances rise from 3/12 to 7/12 on CRK3 and from 11/12 to 12/12 on oscillator2, while MatSci3 favors the fixed workflow. Across 72 settings in a post-hoc six-task subset of eight development tasks, Query-ATLAS achieves 64 validation and 64 in-distribution passes, versus 62 and 59 for the fixed workflow; both pass 47/60 eligible out-of-distribution tests. The fixed workflow retains lower median errors and cost. Broader query coverage emerges as a promising benefit, motivating task-aware tool selection and client-aware fitting for scientific agents. 

---
# GEM-MM: Reducing VLM Hallucination with Entropy-Guided Preference Alignment 

**Authors**: Yuanshan Chua, Xuejiao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.20016)  

**Abstract**: Vision-language models (VLMs) are routinely aligned with offline pairwise RLHF objectives, but multimodal preference data is scarce and treats every response token as equally informative. Under a small preference budget, pair construction becomes nearly irrelevant: standard DPO-style, image-conditional, and self-improvement objectives all perform within 1.1 points on MM-RLHF metrics and 0.5 points on hallucination rates, matching the SFT baseline. We call this entropy-blind preference optimization: the learning signal ignores policy uncertainty, which in chain-of-thought VLMs concentrates on specific fork tokens where visual claims are made. We introduce GEM-MM, extending entropy-guided preference modeling to VLMs. It samples on-policy candidates, rewards high-entropy reasoning forks, penalizes uncertain final answers, and updates the model via group-normalized policy gradients. On a 3000-prompt MM-RLHF budget with Qwen3-VL-4B-Instruct, GEM-MM achieves the best entropy-depth preference score (61.7% vs. 57.3-58.4% baselines) and implicit-reward accuracy (41.4%). Crucially, it reduces AMBER hallucination from 23.4-24.0 to 20.6 and CHAIR from 5.0-5.1 to 4.6 at matched caption coverage. It also ranks best on all three HallusionBench granularities, with a conservative operating point raising the near-chosen rate to 62.4%. An off-the-shelf multimodal reward model ranks GEM-MM first, and two such models independently confirm our main ablations (p < 0.001). We report confidence intervals, paired significance tests, metric-reward functional overlaps, and document a collapse mode when exceeding the fair budget. Code and evaluation harness: this https URL 

---
# Autonomous Multi-modal Agents for ITOps Observability 

**Authors**: Gunjan Siddharth, Sanjana Garai, Utsav Shekhar, Prakhar Jain, Sarvam Maheshwari, Balaji Ganesan, Avirup Saha, Seema Nagar, Pratik Mishra, Felix George, Suranjana Samanta, Harshit Kumar, Mudit Verma, Prateeti Mohapatra, Ravi Jagannathan, Amit Paradkar, Sameep Mehta, Pratibha Moogi  

**Link**: [PDF](https://arxiv.org/pdf/2611.19982)  

**Abstract**: ITOps observability products must integrate data from different tools and systems. We present ITOps Observability Studio, a multi-modal multi-agent system that orchestrates observability workflows through collaborative AI agents. Its three-tier architecture features inter-agent communication, specialized LLM-powered agents for data access, schema operations and visualization, and unified tool integration. The system supports natural language to GraphQL translation, GraphQL schema generation, schema mapping, trace data lifecycle management and multi-tool workflows for reliability engineering. Its agents are instrumented with agent analytics, so that their model and tool calls can be traced. We evaluate each agent on its own benchmark, scored against gold answers where they exist and by a model judge for schema mapping and chart quality. Query generation on a public benchmark and routing work well with a current open model, and the data lifecycle management agent solves most of its scenarios. The evaluation exposed defects that only appear when prompts, parsers and serving stacks are tested together, and we report them together with their fixes. 

---
# Who Writes the Codebook? An Eight-Step Framework for Auditing LLM-Generated Category Systems 

**Authors**: Thilo Tamme, Anton Hantel, Jakob Scheppers  

**Link**: [PDF](https://arxiv.org/pdf/2611.19872)  

**Abstract**: Large language models (LLMs) increasingly create the categories through which qualitative evidence is interpreted. Existing coding benchmarks often hold these categories fixed. We propose the Qualitative Variance Audit Framework, an eight-step method that compares codebook creation, repeated application and code distributions under controlled inputs. It combines model-owned and shared-reference codebooks to distinguish variation in the overall pipeline from application differences within a common category system. An application to 668 organisational interviews, 13,166 text units and eight models from five vendors yields 48 codebooks and 270 coding runs. Models create markedly different category structures; mean within-model reliability ranges from .76 to .92, whereas between-model agreement on the shared codebook is about .69. The framework complements LLM-assisted grounded theory by exposing model dependencies before substantive interpretation. It assesses reliability and representational consequences, not interpretive validity or the quality of resulting theory. 

---
# Interactive Structured Environment Scaling for Large Language Model Agentic Reasoning 

**Authors**: Peng Yu, Luoyi Fu, Xinbing Wang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2611.19871)  

**Abstract**: Reinforcement learning via environmental interaction is a cornerstone post-training paradigm for evolving large language models into autonomous agents capable of multi-turn tool use and exploration. Because the environment defines what the policy observes and how actions are verified, its structure directly bounds the capabilities an agent can acquire. Yet existing structured and retrieval environments rely on upstream pre-filtering---extracting a compact subgraph around the gold answer or retrieving fixed passages before the policy acts. We show that such pre-filtering introduces a fundamental measurement confound: because per-question subgraphs enclose ground-truth paths, they compress the candidate space by orders of magnitude and leak answer locations via local topology, allowing a question-blind degree heuristic to surface gold answers far above chance. To restore path selection, candidate pruning, and optimal stopping as genuine policy decisions, we propose RISE (Interactive Structured Environment), which transitions structured reinforcement learning from static subgraphs to full-scale interactive databases along two coupled axes. First, Decision-Space Scaling exposes the unpruned knowledge graph via a five-stage graph-agnostic typed intent adapter under a bounded call budget and a multiplicative grounding reward gate. Second, Learning-Frontier Scaling overcomes within-group variance collapse under group-normalized objectives by uniting model-independent structural tier progression with a self-evolving competence-frontier bandit. Across nineteen benchmarks and three model backbones, RISE substantially advances in-domain and unseen-schema knowledge-graph reasoning, transfers zero-shot to open-domain passage retrieval, and significantly enhances out-of-domain mathematical and logical deduction. 

---
# When Facts Change, Which Decisions Break? Dependency-Aware Context Graphs for Agents 

**Authors**: Paul Deepakraj Retinraj  

**Link**: [PDF](https://arxiv.org/pdf/2611.19859)  

**Abstract**: When an observed fact changes, an agent must identify and repair the decisions that depend on it while preserving unaffected outputs. We investigate the reliability-cost trade-off of field-level dependency graphs extracted from actual agent read traces. A controlled benchmark spans three executable workflow domains and four successive updates, with independent imperative and expression-based references. Across 120 workflows, two models and two repeated runs, we compare full recomputation, rolling summary memory, lexical retrieval, trace-extracted dependencies and verified dependency upper bounds. Trace-derived graphs reduce reviewed decisions and estimated total cost, but omitted alternative supports and missing repair context can reduce complete-plan correctness. Paired ablations isolate the effects of edge deletion, spurious edges, transitive propagation and context scope. An exploratory enhancement supplies the existing record ID/field directory during extraction without revealing unread values or reference edges. We evaluate this enhancement on 60 fresh workflows with two repeats per model. On this fresh set, the directory-aware graph matches observed full-context complete-plan correctness for gpt-6.1-sol (100%) with an estimated 4.0% total-cost reduction, but reaches only 84.8% versus 99.6% for gpt-6-luna. The results distinguish potential dependency impact from actual decision invalidity and show why selective repair must be evaluated with both executable correctness and full memory-processing cost. The evidence concerns explicit procedural workflows; broad real-world agent-memory generalization remains untested. 

---
# Asking the right question is key to AI discovery in a chain of hypotheses 

**Authors**: Hu Cang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19841)  

**Abstract**: Artificial intelligence (AI) scientists pursue discovery through a chain of hypotheses, fitting candidate models to existing observations and often choosing experiments where those models disagree most. Yet when early data miss a narrow feature, model selection favors simple curves that agree across the unmea- sured gap, so disagreement-guided search has no reason to look where the new phenomenon lies. When we tested two frontier AI scientists, Claude Sonnet 5.5 and GPT-6.1-Sol, across 32 closed-loop trials with a narrow peak omitted from the starting data, both sampled inside the peak region less often than random search. Surprisingly, the agent that followed model disagreement more closely missed the peak in all 32 trials, whereas discovery succeeded when experiments probed the unmeasured gap early and followed up across the anomaly. Asking the right question, rather than testing only where current models disagree, is therefore key to AI discovery in a chain of hypotheses. 

---
# MAPS: A Multimodal Agentic Framework for POI Data Production Based on Crowdsourced Street-view Imagery 

**Authors**: Sijie Ruan, Tingyu Xie, Mingfei Liu, Yicheng Tang, Haoyu Han, Jingnan Qi, Jingshu Zeng, Dongqi Xu, Jing Wang, Shuliang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19807)  

**Abstract**: Point-of-interest (POI) databases support location-based services, yet urban changes quickly make their records stale and manual maintenance is costly. Crowdsourced street-view images provide timely visual evidence, but converting them into database actions requires separating multiple POIs and irrelevant text in each image, then resolving incomplete observations against ambiguous records. We study Street-view Image-based POI Data Production and propose MAPS, the first framework to integrate structured POI extraction with database maintenance from such imagery. MAPS contains two cascaded modules. SPICE progressively adapts a lightweight vision-language model to reconstruct and filter text, group entries belonging to the same POI, and classify their semantics and completeness. RIMA retrieves candidate records, performs evidence-grounded POI matching and attribute-level reasoning, and selectively acquires complementary visual, geospatial, or external-source evidence when necessary. In human-reviewed full-task evaluation, MAPS outperforms the strongest baseline by 21.8\%. MAPS has also been deployed in Kuaishou's large-scale POI production pipeline, reducing average processing time by 76.9\% relative to manual annotation. These results demonstrate the effectiveness and practical value of MAPS for scalable POI maintenance. 

---
# Lossless First-Action Menus for Linear-Reward Planning 

**Authors**: Yukun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19795)  

**Abstract**: We study sequential decision making where an agent reserves a minimum menu of initial actions before multi-objective trade-offs are revealed, followed by unrestricted continuation planning once linear reward weights arrive. We prove that a menu preserves the unrestricted optimal return across all linear scalarizations if and only if its reachable continuations cover all vertices of the full feature polytope. This geometric equivalence yields an output-sensitive algorithm that is fixed-parameter tractable in the vertex count, combining scalar backward induction with oracle-based facet identification. On explicitly layered graphs with unit feature increments, menu verification is coNP-complete and synthesis is complete for the second existential level of the polynomial hierarchy, even under additive loss strictly below one. In fixed dimensions two and higher, verification is polynomial whereas synthesis remains NP-complete, while one-dimensional instances admit polynomial-time solutions. 

---
# Read Before You Fit: Text-Derived Constraints for Discovering Differential Equations from Noisy Biological Data 

**Authors**: Pronoma Banerjee, Md. Masudur Rahman, Juan P. Wachs  

**Link**: [PDF](https://arxiv.org/pdf/2611.19779)  

**Abstract**: Large language models are increasingly used for scientific equation discovery. However, most existing methods are strongly data-driven and assume sufficiently clean and representative observations. This assumption is problematic in biological systems, where individual-level measurements are often sparse, noisy, heterogeneous, and expensive to obtain. While the underlying mechanisms are extensively described in scientific text, they are not explicitly used for the LLM based modeling of such systems. We introduce P2C, an agentic framework that translates textual descriptions of biological systems into a typed set of formal constraints, and use these constraints to restrict candidate ordinary differential equation (ODE) structures before numerical fitting. P2C consists of five stages: Text2Constraint extraction, ODE discovery, symbolic verification, parameter fitting, and model scoring. We evaluate P2C on four published ODE systems of acute inflammation and immune regulation, under clean and noisy observations. By progressively degrading the numerical evidence, we isolate when textual domain knowledge contributes beyond regression alone. Across the four systems, adding Text2Constraint reduces the worst form error from 1.397 to 0.499 on clean data and from 13.096 to 1.095 for data perturbed with Gaussian noise of $\sigma=0.1$; P2C also achieves lower form error and trajectory error than several LLM-based baselines over the tested noise range. 

---
# When AI Agents Should Repair, Replan, or Abandon: A Threshold Theory of Plan Commitment 

**Authors**: Gaurav Tewari  

**Link**: [PDF](https://arxiv.org/pdf/2611.19663)  

**Abstract**: AI agents often discover mid-execution that their plan is broken, for example when an action fails or a tool is withdrawn. Deployed systems use conventions such as retry loops or "always replan," and a recent benchmark finds no fixed strategy dominates. This paper derives the optimal policy. In the model, a disruption destroys the causal support of part of a plan; repair is a partial reset that preserves the plan's structure, replanning is a full reset, and changing a plan is costly because others have conditioned on it. The optimal policy is a threshold in effective disruption depth: repair shallow damage, replan after deep damage, and abandon the goal when even a fresh plan is not worth its cost. Depth depends on plan structure. In expectation, a single broken action forces 1+R(G)/n actions to be revalidated, where R(G) counts the pairs of actions joined by a dependency path; to first order, a fully serial plan switches to replanning at raw damage about w times smaller than a plan of w parallel chains. A closed-form stability-price threshold determines when a worse repaired plan should be kept over a better fresh one. As language-model planners push replanning costs toward zero, the stability price becomes the only force sustaining plan commitment. Under repeated disruptions, revision frictions make commitment hysteresis optimal: the agent keeps executing a plan it would never choose fresh, and each repair adds patch debt until the debt outweighs repair's cost advantage and the agent rebuilds. A value-of-diagnosis result states when measuring damage is worth its compute and shows that "always replan" is optimal when damage is predictably deep and replanning is the better fallback. A separately implemented numerical checker tests each proposition (zero violations; 1.6 million static-model evaluations), and the model's regime map reproduces that benchmark's regime flips qualitatively. 

---
# Data-Driven Heuristic-Guided Business Process Redesign 

**Authors**: Nan Sai, Karolin Winter, Remco Dijkman  

**Link**: [PDF](https://arxiv.org/pdf/2611.19623)  

**Abstract**: Business process redesign (BPR) has a long-standing catalog of best-practice redesign heuristics, yet, applying them remains a mostly manual task that requires extensive domain knowledge. Despite recent advances using Large Language Models (LLMs) for conversational process model redesign, limitations remain. For example, redesign is user-driven instead of data-driven and is not guided by best practices such as the redesign heuristics. Moreover, only one change can be handled at a time. To address these limitations, given an event log and a BPMN model, we present the DH-BPR framework to realize data-driven heuristic-guided BPR using LLMs. To design DH-BPR, we first conducted an analysis of which redesign heuristics can be automatically detected and applied given information contained in an event log such as the fraction of cases that bypass an activity. Based on this, DH-BPR detects and validates heuristic candidates through a combination of rule-based techniques and an LLM. After that, DH-BPR jointly designs and records an application plan for valid candidates, before finally yielding a redesigned BPMN model and an updated resource model (the way resources execute the process). Evaluation based on simulation using a synthetic benchmark and two real-world event logs shows that the framework's redesign reduces the mean case completion time by up to 61.5%. 

---
# A Hierarchical Statistical Framework for Evaluating Agent Reliability 

**Authors**: Mirco Milletarì, Tuhin Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19614)  

**Abstract**: Agents operating in complex scenarios often face uncertainty due to incomplete information about the environment they operate in, ambiguous user queries, complex business rules and the inherently stochastic dynamics of the governing policy. Several new benchmarks have been proposed to evaluate LLM-based agents on complex tasks in multi-tool environments, including settings with policy constraints and varying levels of intent ambiguity. These conditions make it important to evaluate agentic systems systematically and quantify their performance under different conditions to assess their reliability in production. This need is evident in the recently introduced Thinkingbox benchmark, where even the strongest models struggle to perform consistently across diverse business scenarios. This paper presents a statistical framework for evaluating uncertainty and reliability in agentic systems. We define a hierarchy of scenarios, instances, and samples to assess model performance and variability systematically. We apply this framework to common metrics found in the literature: mean pass rate, $pass@k$, and $passhat^k$, with uncertainty estimates for each metric. Aggregating metrics at different levels of the hierarchy reveals different structures and properties of the policy model; we connect this hierarchical aggregation with the known process of \emph{coarse-graining} as intended in the physics and statistical literature. We independently derive exact finite-sample expressions for the sampling variances of the unbiased $pass@k$ and $passhatk$ estimators; related results also appear in recent submissions. Our central contribution is a constructive derivation of hierarchical uncertainty propagation, making explicit its correspondence with coarse-graining and separating execution noise from task heterogeneity. 

---
# Behavioral Assurance for Socially Deployed AI Agents 

**Authors**: Nicole H Brennan, Jiaee Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2611.19564)  

**Abstract**: AI agents are increasingly deployed in socially consequential settings, yet existing evaluations provide limited assurance that they behave consistently with their intended characteristics across diverse interactions. We introduce an evaluation framework for measuring the behavioral consistency of socially deployed AI agents. The framework operationalizes agent behavior along five measurable axes and uses a standardized evaluation battery spanning six socially salient interaction settings to assess behavioral differentiation, consistency, and drift. Controlled persona configurations serve as experimental anchors, enabling systematic testing of whether intended behavioral differences persist across contexts rather than appearing only as stylistic variation. Preliminary analyses demonstrate measurable differentiation across configurations and suggest associations between behavioral profiles and interaction outcomes. By shifting the focus from persona design to behavioral assurance, this work provides a foundation for evaluating whether socially deployed AI agents behave as intended, consistently and measurably, across the situations in which users encounter them. 

---
# Evidence-Conditioned Completion and Synthesis of Electronic Health Records 

**Authors**: Sicheng Shu, Parker Knight, Huitao Li, Yuan Luo, Yifan Peng, Shawn N. Murphy, Rui Yang, Chuan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2611.19546)  

**Abstract**: Patient information can guide electronic health record (EHR) generation, but how its amount shapes predictive utility and disclosure risk remains poorly characterized. Here we present EHRDiB, which unifies visit completion and synthesis by distinguishing original missingness from withheld inputs and combining a shared multimodal representation with an evidence-conditioned latent diffusion bridge. We compared EHRDiB with three completion and eight synthesis methods on MIMIC-IV and numerical and discretized Duke Health representations. Evaluation covered downstream predictive utility, generative diversity, and distributional fidelity for both tasks, and disclosure risk for synthesis. EHRDiB achieved the smallest utility gaps from real-data-trained classifiers in seven of eight completion comparisons. In synthesis, AUPRC gaps were 45.2% smaller for ten MIMIC-IV principal-diagnosis targets and 53.3% smaller for numerical Duke Health data relative to the best comparator. Synthesis with calibrated thresholding yielded the lowest marginal distribution error across all three datasets, with Coverage above 0.95 for both output rules. Across twelve evidence-fraction intervals in MIMIC-IV synthesis, membership-inference AUROC increased from 0.516 to 0.598 as the mean target evidence fraction rose from 0.20 to 0.50; utility gaps narrowed relative to the lowest evidence configuration but did not decrease monotonically. Our study provides a unified framework and empirical foundation for EHR generation that considers both its utility for clinical analysis and its information disclosure risk. 

---
# Continuation-Guided LLM Evolution of Program Policies and Goal Compilers in Craftax 

**Authors**: Víctor Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2611.19537)  

**Abstract**: Coding agents can synthesize and optimize code-based policies from simulator feedback, allowing the resulting program to act with zero inference model calls. But for long horizon tasks, two challenges arise: local improvements can hurt final success and switching to a new objective requires another expensive synthesis run. We propose a framework that first develops competence by refining the policy through a feedback cascade, moving from episode returns to hierarchical goals and belief updates which can be scored densely from simulator. In the Craftax game, a partially observed and procedurally generated environment, refinements via episode level feedback stall at <1% final-level arrival, whereas with the feedback cascade the optimized policy reaches the final stage 50% of time (28% final boss defeats). Furthermore, to make this competence reusable towards new goals, we factor the previous controller into an environment interpreter $I$ (models, memory, skills) and a rule program $p_0$, and we show that coding agents can optimize a reusable goal compiler $C_{\theta}$ that maps structured tasks $g$ into new rule programs $p_g$. Across five independent compiler searches trained only on short tasks, the resulting compilers successfully transfer zero-shot to longer tasks without any task-specific tuning nor inference-time LLM calls: continuation-scored feedback and a fixed machine with an editable program together turn a single-task competent policy into a reusable one.
Code at: this https URL 

---
# LLM Trading Agents Earn Beta, Not Alpha: A Look-Ahead-Free Benchmark 

**Authors**: Jackson Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19530)  

**Abstract**: Large language model (LLM) trading agents now trade stocks for their users, and benchmarks rank models by their agents' returns. Yet a single run's return says little about skill. In our study, two runs of the same model that differed only in the random seed ended 5.2 points apart on average, so telling skill from luck takes many backtests across markets, models and seeds. However, live arenas trade each day only once, and backtests either use the few months after a model's training cutoff or run on years the model may remember, often with news joined to prices by calendar date. In this paper, we present rta-bench-1, a backtest over three US markets (a rally, a bear market and a crash). It shows each headline only after its publication minute, plays each period blind and named, and checks every run for look-ahead. It also splits each return into factor exposure and alpha and scores every decision against random choices of the same day and sector. In 91 runs of six models from four vendors, no alpha was significant, market and style exposure accounted for a median 78% of the return, and the decisions did no better than random stocks of the same sector. The agents bought the stocks with the most positive news, yet in ablations they earned as much without their headlines. Still, the agents carried out rules reliably, since a single cash rule in the prompt cut a GPT-based agent's share in stocks by 30.3 points in all six pairs. Builders should therefore supply the edge themselves and report exposures and alpha, and benchmarks should replay news by publication time. 

---
# Ultra-CAN: A Lightweight Blind ISAC Receiver for Chaos-Based Telemetry and Local-Seed Ranging in the Internet of Drones 

**Authors**: Ettore Caruso, Marco Siino, Giacomo Morabito  

**Link**: [PDF](https://arxiv.org/pdf/2611.19529)  

**Abstract**: The Internet of Drones (IoD) needs secure telemetry and proximity sensing within the SWaP-C constraints. We show that a single blind neural receiver driven by one chaos-based waveform performs both functions without a separate radar transceiver. In telemetry mode it decodes a peer's burst blindly; in radar mode it regenerates the per-burst chaotic reference locally and ranges the echoes of its own transmission. Ultra-CAN pairs a shared $64$-dimensional representation with two ultra-light backbones, an attention-pooled Conv1D network and a multi-head QKV network ($27.6$k--$44.2$k parameters), feeding communication and sensing heads. Both ship as int8 TensorFlow Lite Micro firmware; the Conv1D receiver decides a burst in $16.9$ ms on a $240$ MHz microcontroller with $24.2$ KiB of activations, a measured energy of $2.2$-$3.4$ mJ per burst, and float-equivalent decisions, sustaining up to $59$ bit/s of telemetry or a $7.4$ Hz range track. The two variants rank first and second among learned demodulators in four aerial swarm scenarios, with a pooled BER down to $4.6 \times 10^{-4}$ on the single-echo link ($2.4 \times 10^{-3}$ with three echoes). Its sensing head identifies true obstacles in $66.8\%$ of peer-laden bursts (vs $42\%$ matched filtering, $44\%$ OFDM-ISAC, $48\%$ peak-picker ceiling). A peer-exclusion gate cuts peer reports from $4.4\%$ to $0.02\%$ of answered bursts, answering $23.5\%$ of total bursts ($75\%$ on the exact tap) with $0.70$ m/s velocity tracking. In-domain refitting recovers telemetry BER by up to three orders of magnitude under 3GPP channels, jamming-aware training restores the link, and frequency agility adds a $6$--$9$ dB margin against partial-band jamming. 

---
# Objectify, Don't Delete: Complaint-Derived Evaluation of Ambient Clinical Notes 

**Authors**: Nikita Gupta, Narayanan Unny Edakunni, Vikram Okaly, Suranjana Datta, Navya Penubadi, Karl Eric Swanson, Siddharth Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2611.19525)  

**Abstract**: Evaluating the quality of AI-generated clinical notes is hard: standard metrics like ROUGE measure lexical similarity to reference notes, while expert checklists encode what committees believe matters. Neither captures what clinicians actually complain about. Recent work has shown that evaluation criteria can be mined from user feedback and filtered to criteria that LLM judges can reliably enforce. We argue that this is where much of the value is lost. The criteria that fail enforceability are disproportionately the ones clinicians raise most often, because high-frequency complaints are stated as several questions glued together. No yes/no judgment can grade such composites as posed. We introduce an objectification step that rewrites such criteria into sets of scoped, determinate decisions. We demonstrate the approach through an evaluation pipeline that mines criteria from complaints, objectifies them, and deploys LLM judges against both planted defects and production notes. Across eleven defects planted into ACI-bench reference notes, objectified judges localize violations with median recall of 0.95, while reference metrics change by at most 0.037. Given the complaint as one broad question, the same judges still find every planted defect, but they also flag 70% of the unperturbed reference notes; one section-pair question per call raises Youden's J from 0.30 to 0.72 (by 0.19 to 0.26 with prompt effort matched). We release ACI-Defect, a perturbation benchmark; clinician-reviewed instance-level labels showing quality gaps in the ACI-bench gold standard (32 of 40 notes contain hallucinations; 40 of 40 omit discussed facts); and grounding-judge verdicts. On 1,446 production notes, mean instance-rate burden falls at every clinician-rating step (rho=-0.15), a small association. Defect counts have to be kept per span rather than collapsed into one pass/fail bit per note. 

---
# Program Distillation: Transferring LLM Capability from System 2 to System 0 

**Authors**: Pujun Zheng, Wanying Ren, Guoxiu He  

**Link**: [PDF](https://arxiv.org/pdf/2611.19461)  

**Abstract**: Recent studies leverage large language models (LLMs) to construct executable programs through environment interaction, feedback analysis, and code revision. However, prior work has mainly treated LLMs as optimizers, overlooking their ability to transfer knowledge across system levels. We propose Program Distillation (PD), a framework that formalizes the transfer of abilities from System 2 to System 0. During learning, the LLM provides task knowledge and draft strategies; at deployment, a student program executes the resulting policy. We show that this approach can improve performance while reducing inference time. We introduce PDBench, a benchmark comprising 27 discrete- and continuous-control tasks with fixed initial policies, and PDMetaHarness, a model-harness-independent meta-system for monitoring and guiding the learning process. The strongest model achieves a normalized mean score of 0.728, surpassing PPO (0.656) and Jev-like decision models (0.239), while running exclusively on CPUs and reducing inference costs. Our analysis shows that performance is largely driven by the first breakthrough during learning, and structural modifications have a slightly lower success rate than parameter modifications but typically yield larger gains. We further introduce style distillation, which enables interpretable stylistic control over the learning process. Overall, we present an analytical framework for PD and demonstrate its practical viability. 

---
# PMR: MoE Routing with Shared Prototypes and Congestion Pricing 

**Authors**: Wending Wang, Binqi Huang, Ju'an Li, Yu Guo, Lei Duan, Yuqi Ouyang, Dengchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.19413)  

**Abstract**: Conventional linear routers in sparse mixture-of-experts models select experts directly from token-expert matching scores, coupling input matching with expert allocation. We propose Prototype-Mediated Routing (PMR), a two-stage router with shared prototypes. PMR first computes each token's distribution over prototypes, then uses it to mix prototype-conditioned expert distributions according to the law of total probability. This formulation models input matching and expert allocation separately, explicitly represents many-to-many associations between prototypes and experts, and provides separate interfaces for context adaptation and load regulation. Within this framework, short-term states summarize preceding tokens to adapt the prototypes, while congestion prices updated from dispatch feedback achieve loss-free load balancing by adjusting the conditional expert distributions. We use a two-pass blockwise approximation to accelerate short-term state processing, achieving over 20 times the training throughput of the eager tokenwise PMR implementation while maintaining high agreement in routing decisions. Across three pretraining configurations with 0.43B, 1.18B, and 0.48B total parameters, PMR improves mean zero-shot accuracy on seven benchmarks over linear routing baselines with matched expert architectures by 1.54, 2.65, and 1.45 percentage points (pp), respectively. The corresponding gains on ARC-Easy are 6.71, 5.80, and 3.92 pp. In parameter-efficient fine-tuning of 7B-8B backbones, PMR's improvements over MixLoRA vary across backbones, with gains of up to 1.76 pp in mean accuracy across four tasks. 

---
# Confidence Without Cause: Predicting and Localizing Physics-Shortcut Sensitivity in Language Model Hidden States 

**Authors**: Shubham Mishra, Atharv Sharma, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.19403)  

**Abstract**: A language model's confidence on a physics problem deserves trust only if it ignores details irrelevant to the answer. We introduce DECOY (Distractor-Embedded Certified cOunterfactuals for phYsics): 155 families of four prompts differing in one irrelevant detail. A symbolic solver certifies that each edit preserves the answer. Families are labelled by the largest drop in the model's log-probability of the certified answer, using a threshold fixed before probe training. Qwen2.5-7B-Instruct is sensitive on 50.0% of the 140 families yielding labels, and Llama-3.1-8B-Instruct on 22.9%. Effects are largest when distractor values lie numerically close to the answer. A linear probe on Qwen's layer-18 hidden state, averaged over four renderings, predicts sensitivity before generation at AUROC 0.760 with whole families held out. A separately tuned single-rendering probe reaches 0.747. A frozen layer-16 probe reaches 0.820 pooled AUROC on 15 families built after training, though within-cue-type AUROC is lower (0.667-0.6875). Under a paired family bootstrap, the probe performs on par with a fitted token-entropy baseline (AUROC 0.727) and achieves the lowest Brier score among tested predictors (0.204 versus 0.214). Knocking out four attention heads at layer 16 cancels sensitivity on a 20-family panel, localising it to a narrow mechanism. These results show that Qwen's confidence can track irrelevant details, that this failure is detectable before generation, and that targeted intervention can suppress it. DECOY provides certified ground truth and paired-tested baselines for evaluating hidden-state monitors. 

---
# SARGO: Agreement-Gated Reinforcement and Suppression via On-Policy Self-Guidance for Visual Hallucination Mitigation 

**Authors**: Jinyu Li, Zhiming Zhang, Hao Fang, Hongyao Yu, Bin Chen, Albert Ma, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2611.19388)  

**Abstract**: Vision-language models remain susceptible to visual hallucinations, often mixing grounded information with unsupported claims in the same response. Response-level reinforcement learning assigns a shared advantage to all tokens in a response, causing negative updates to suppress correct information alongside unsupported claims. To reduce hallucination while preserving useful visual information, we introduce SARGO (Self-guided Advantage-Routed Grounding Optimization), a post-training framework for selective reinforcement and suppression through on-policy self-guidance. Direct and reciprocal weighting gives better-supported tokens stronger reinforcement on positive updates and weaker penalties on negative updates. Agreement between overall and hallucination-specific advantages selects when to apply this guidance. At the response level, a reference-calibrated reward couples factuality with the retention and expansion of grounded information, discouraging lower hallucination scores obtained through omission. Experiments on Qwen3-VL-4B, Qwen3-VL-8B, and Gemma3-4B demonstrate that SARGO outperforms the evaluated mitigation baselines across Obj-Hal, AMBER, THRONE, and SHR evaluations. SARGO achieves these gains while preserving useful visual information and general multimodal capabilities. 

---
# Does a completed simulation run mean a usable urban wind result for AI agents? 

**Authors**: Zhongke Qu, Yunwei Zhang, Junwei Su, Yabin Zhang, Chengwei Li, Qingxiang Shui, Hailong Qu, Zhaolin Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19385)  

**Abstract**: AI agents based on large language models (LLMs) can now configure and run computational fluid dynamics (CFD) solvers, but a completed run does not ensure that delivered metrics can be recomputed from raw outputs. This gap was examined for urban wind simulation, where multi-case comparisons must hold non-target conditions fixed and reported values must be traceable to raw samples. UrbanWind-AgentBench, comprising 200 tasks and 367 cases in 10 engineering task families, was constructed. Public task contracts specify variable factors, fixed conditions, numerical thresholds and deliverable requirements, and two-layer scoring separates a structural score for deliverables, run provenance and raw evidence from a numerical score obtained by independent recomputation against references. WindPilot was then proposed as an execution framework that incorporates contract parsing, execution provenance, deterministic verification and bounded acceptance review into the LLM execution loop. On 30 OpenFOAM tasks with independent references, at least the structural maximum of 60 points was reached by a base agent on 28 tasks, but numerical credit was obtained on only 6. With the same model name and budget, WindPilot increased the number of tasks with numerical credit to 17, full-score tasks from 0 to 3 and the mean task score from 62.5 to 72.5. Mean action steps fell from 130.4 to 104.9, and mean token consumption decreased by 12.9%. The mean gain for atmospheric boundary layer (ABL) tasks was 18.5 points. A completed simulation run is therefore not sufficient for a usable urban wind result; usability depends on checks that tie reported metrics to raw evidence during execution. 

---
# One Run Is Not Enough: Reminding a small Gemma 4 coding agent of its last finding made no measurable difference in a paired test 

**Authors**: Patrick Passos  

**Link**: [PDF](https://arxiv.org/pdf/2611.19359)  

**Abstract**: When a coding agent runs out of room, its earlier findings are cut and it forgets them. We tested a simple fix: automatically putting the agent's last finding back in front of it. We fixed the rules in advance and ran 98 paired coding tasks with and without this reminder, using Gemma 4 31B (QAT, 4-bit) in the Gemma 4 Developer Agent harness. The reminder made no measurable difference: across all 98 tasks, 47 were solved with it and 48 without (-1.0 points, 95% interval [-8.3, +6.3], exact McNemar p=1.0). The primary outcome (a documented deviation from our preregistration), the first 48 pairs, was 29/48 with memory against 25/48 without (+8.3 points, [-2.1, +18.3], p=0.22); the extension of 50 tasks, decided after that result, pointed the other way (18/50 against 23/50, -10.0 points, p=0.125), so the sign did not replicate. With 19% of tasks flipping between two identical no-memory runs, the combined interval excludes harms larger than about 8 points and gains larger than about 6, and the design had 80% power only for effects of about +-12 points or more. The model almost never wrote notes (2 of 16 and 4 of 32 runs), and memory cost context: the first request is larger and something was injected at only 47% to 65% of compactions. The registered secondary, repeated actions per compaction, was also null. Two directional hypotheses found on Modal did not replicate on Kaggle. We also report an A/A calibration and a validity funnel (129 tasks, 105 valid). Together these results show that single-run memory claims on task sets of this size sit inside run-to-run noise. The treatment is a harness mechanism and does not test agent-written memory. 

---
# Attention U-Net Cellular Automata for Multi-Class Land Use and Land Cover Change Simulation in Lucknow, India: Evaluation Against Null and Classical Baselines 

**Authors**: Shubham Phapale, Poonam Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2611.19348)  

**Abstract**: Convolutional networks are often used to learn the transition rules of land-use cellular automata (CA), yet their predictions are seldom compared with a null model that predicts no change. We built AttentionUNet-CA, which reads a 32 x 32 neighbourhood of land-cover states with four drivers (elevation, distance to major roads, distance to the central business district (CBD) and population density) and predicts the state of each cell ten years later. We tested it on Lucknow, India, with GLC_FCS30D land cover for 1990, 2000, 2010 and 2020, a 30 m series mapped consistently over time, and compared its built-up class with GHSL and ESA WorldCover. AttentionUNet-CA was trained on 1990-2010, selected on held-out spatial blocks and evaluated once on 2010-2020, with three seeds. Baselines were the null model, a random-forest CA (RF-CA) and a U-Net CA without attention; each network was trained with and without class weights. Only 8.1% of the study area changed in the test decade, so no model matched the overall agreement of the null model (91.9%). On change, measured by the figure of merit, the class-weighted networks scored highest (U-Net CA 22.0 +/- 1.2%, AttentionUNet-CA 21.0 +/- 1.4%), ahead of RF-CA (20.3%), but by margins comparable to seed-to-seed variation; the networks estimated the amount of change better than RF-CA. The attention gates did not improve accuracy. Population density and distance to the CBD were the most useful drivers. Projected to 2030, built-up land grows by 22-35%, mostly at the city edge. 

---
# Information Limits of Replay-Free Recurrent State Migration 

**Authors**: Khizar Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2611.19345)  

**Abstract**: When a recurrent model is replaced, its old state may not determine what the new model would compute from the same history. We characterize the information that must be retained before the replacement is known. For a time-invariant affine linear family, the minimum continuous causal memory for order-$k$ local migration equals the reachable dimension of a lifted coefficient system minus the information in the anchor state. A controllable $d$-state model with scalar input needs at most $kd$ additional real coordinates, even when arbitrary matrix replacement directions are revealed only after streaming. A single rank-one direction can already require all $kd$. Conditional singular values and auxiliary-bit bounds distinguish this local information law from recovery at a fixed tolerance. We also give invariant-subspace and future-output constructions. Controlled experiments compare the resulting representations. In a small language model, equal-memory cubic interpolation improves mean state fidelity over third-order coefficients but provides no meaningful predictive gain. An analytic history basis preserves useful adaptation more economically in a coupled identification task. With repeated slow poles, eight history coordinates meet normalized output error $10^{-4}$ across five fitted seeds despite substantial full-state error. Splitting the poles raises the smallest tested successful width to 64. 

---
# From Mutation to Medication: Characterizing Failures of Clinical LLMs to Respect Genomic Resistance 

**Authors**: Negin Ashrafi, Roxana Daneshjou  

**Link**: [PDF](https://arxiv.org/pdf/2611.19339)  

**Abstract**: Language models are increasingly proposed to support molecular tumour boards, but may recommend cancer therapies despite genomic evidence of resistance. We evaluated ten models on 1,464 resistance cases from five curated oncology knowledge bases, collecting 58,910 judgments. Models endorsed resisted therapies in 21.8-81.4% of cases. However, raw rates can reflect a tendency to agree rather than an ability to distinguish resistance from drug sensitivity: one open-weight model endorsed 81.4% of resistance cases and 82.3% of sensitivity controls. We therefore compared endorsement within matched gene and evidence-level groups. This analysis is exploratory because it was specified after the pre-registered analysis, which yielded null-to-weak results. Measured failures depended strongly on the reference knowledge base and evidence strength. When a second source independently confirmed resistance, four models across three vendors endorsed at most 2.4% of cases; at weaker evidence tiers, discrimination was not measurable. Models sometimes acknowledged resistance yet recommended the therapy. Repeating a model's own resistance conclusion reduced endorsement to 0.0-3.5% on cases where it had stated resistance. Falsely attributing the opposite conclusion increased endorsement above baseline for all seven hosted models, reaching 95.2% for GPT-4o; an instruction without an attributed conclusion changed endorsement by only 0.9 percentage points. Providing the correct curated evidence reduced unqualified strong recommendations to 0.3-3.5%, testing evidence use under an oracle condition rather than retrieval. 

---
# NI-ACT: A Neural Intent Interface for Brain-Embodied Manipulation 

**Authors**: Yinghao Li, WeiHsien Lee, Zeyu Zhang, Canyang Zhao, Ke Fang, Xing Li, Bing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19332)  

**Abstract**: Intracortical brain-computer interfaces can drive robotic arms, but they keep the user inside the fast sensorimotor loop: every approach, correction, and grasp must be neurally specified. Whether intracortical population activity can instead serve as a task-level condition for a learned embodied policy, without symbolic commands or hand-designed skill libraries, remains unknown. NI-ACT addresses this question with a hierarchical brain-embodied framework: a brief motor-cortical signal is read once as a persistent intent, and a visuomotor policy then closes the perception-action loop from vision and proprioception. In a counterfactual grasping task where the scene cannot specify the target, the neural intent alone determined which object the robot grasped; replacing the intent changed the behavior accordingly. Passing the full population latent as a dedicated intent token, rather than a one-hot decoded direction, raised grasp completion from 69% to 81% over 132 held-out scenes, whereas a continuous direction alone did not improve on a one-hot label. Where the signal was read also mattered: in one dual-array session, with the readout window matched, premotor (PMd) intent completed more grasps than primary motor (M1) intent (77% vs. 53%), and the conventional post-cue M1 readout reached 42%. Preparatory activity was itself sufficient: in a center-out task, intent read only from PMd before the go cue (that is, before any movement) drove complete grasps, and in one recording it completed more grasps than the same animal's post-cue readout. These results show that how a neural signal is represented, and where and when it is read, jointly determine how effectively it conveys movement goals to an embodied agent. 

---
# The Data Debt of Stockouts: Partial Identification and Minimax-Regret Inventory Control 

**Authors**: Jinbiao Chen, Shuang Jin, Hanzhang Qin  

**Link**: [PDF](https://arxiv.org/pdf/2611.19323)  

**Abstract**: Problem definition: Inventory decisions select which demand enters sales logs. Unrecorded forecasts can leave demand frequencies unidentified even when stocking levels are recorded. We study which multistage ordering decisions these logs can justify and what finite samples can guarantee.
Methodology/results: Nested stockout thresholds yield exact sets of compatible demand laws and finite representations for policy evaluation. We choose one deterministic inventory rule to minimize its largest expected cost gap from the informed optimum under the same law. This population value is data debt for the specified information and policy class. Under independent demand, carryover can create positive debt despite common optimal actions in every isolated period. Finite samples add learning error above debt. The required sample count grows linearly with the selection bound, with upper and lower orders matching up to logarithmic factors. This rate requires fixed demand and cost bounds, sufficient revelation, and a uniformly positive probability of clearing all feasible stock across the confidence sets. Controlled and semi-synthetic studies separate policy improvements from tighter guarantees.
Managerial implications: More records can tighten guarantees while debt persists. Deeper revelation can establish common optimality without fully identifying demand. Using these guarantees requires validated logging, a credible selection range, and a regret tolerance in cost units. Information acquisition must distinguish tighter assurance from operating savings. 

---
# Hardening Agentic Web Populations Through Multi-Criterion Competitive Co-Evolution: Lessons from Wargame Strategy Optimization 

**Authors**: Shashank Raj, Kalyanmoy Deb  

**Link**: [PDF](https://arxiv.org/pdf/2611.19319)  

**Abstract**: Agents on the open web will face adversaries that keep adapting, so a defense that is only tested once will eventually be found and broken. We propose treating jailbreak attacks and safety guardrails as two populations that evolve against each other, rather than as a fixed attacker probing a fixed defense. This idea comes directly from wargame strategy optimization, where two-population competitive co-evolution with alternating evolution cycles and cross-play evaluation has already been shown to produce strategies that are robust to a moving opponent, and where a companion line of work has shown that the same machinery can measure how much competitive pressure a population must absorb. However, real-world applications can never rely on a single criterion as a key performance indicator, rather must use multiple conflicting criteria, such as, cost and efficiency. We describe how this mechanism transfers to language model safety, argue why the resulting multi-criteria, competitive and coevolutionary arms race is a genuinely networked, population-level problem rather than a two-agent toy setting, and lay out the open problems that stand between the wargame version of this idea and an agentic web version of it. 

---
# Macro Control and Capability Evaluation of Black-Box World Models: A Two-Regime Framework for LLM-Agent Markets 

**Authors**: Zhiyang Niu, Chuanjie Li, Huayuan Shen, Mengmeng Zhao, Jiajia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19311)  

**Abstract**: Large language model (LLM) agents can be organized into multi-agent markets that act as world models, mapping macro information to emergent macro dynamics while the evaluator observes only macro inputs and outputs. This paper separates a model's capability into two stages that a single aggregate score cannot separate: information simulation, from macro information to micro behavior, and macro emergence, from micro behavior to the macro outcome. We introduce a soft macro-control method that announces a public reference path day by day without touching individual decisions, orders, or matching, and a two-regime evaluation framework that measures both stages from macro observations alone. In the reference-only regime the symmetric true and flipped references share a common level R, which measures realization under the announced reference, while their paired gap B measures structural bias; in the information-injection regime external information dilutes the reference, so only the true-flipped gap Delta remains interpretable. On five single-index LLM-agent markets, announcing a target path shifts the realized path toward the announced target on both announced references under both information conditions. Relative to the no-reference path, the tracking score is higher by 0.306 and 0.415 without news and by 0.183 and 0.329 with news; within the stock/cash scan structural bias is lowest at 70/30 (B=0.042); realization peaks at a 3:7 technical share (R=0.894 versus 0.358 for technical-only); and Delta is 0.0341 with fundamental data and 0.0487 with news. Both stages use the same two observable scores, each read in the regime where it is interpretable. 

---
# Staged Semantic Distillation: Preserving Free-Form Semantics in a Low-Compute Non-LLM Surrogate of a Staged LLM Agent 

**Authors**: Zhiyang Niu, Huayuan Shen, Chuanjie Li, Mengmeng Zhao, Jiajia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19299)  

**Abstract**: Staged LLM agents make market and social simulators realistic, but one 102-day run of a 200-agent, ten-index market simulator issues 203,100 language-model calls and returns one realization. Conventional low-compute distillation drops their free-form intermediate state. We present staged semantic distillation, a low-compute method that preserves non-structured semantics in a non-LLM student trained from one recorded trace. A frozen encoder maps each teacher output to a 1024-d vector; the student sees typed stage-tagged vectors with stage-specific visibility, is jointly supervised, and trains with scheduled self-history. One TwinMarket trace yields a 13.2M-parameter deterministic student that trains in 1,658 seconds on one 8-GB GPU and runs with zero teacher or decoder calls and no text generation (0.14 ms per decision). Its analysis stays close to the teacher (cosine 0.943), and swapping semantic inputs between agents moves 64.7% of decisions against 12.8% for a matched donor, so the channel is preserved and live. We use the surrogate to study what the transfer carries. It reaches 0.8347 balanced accuracy on the teacher-forced replay and 0.7783 [0.766,0.791] on the deployment decoder; the five-seed gain over a no-semantic retrain is +2.2 [1.21,3.14]. Against the teacher's own re-sampling level, round-one R is 97.1% [95.8,98.4] and 92.3% [89.0,95.4]; R is a diagnostic, not an accuracy. A five-seed content-by-schedule study isolates the gain: scheduled self-history carries it (+2.2 [1.21,3.14]); shuffled and constant content controls reproduce it, while content matters teacher-forced. The design rule: invest in the deployment schedule, let content earn its place at the teacher-forced operating point. A 20-seed, 200,000-decision sweep takes 12 minutes with zero LLM calls, versus three hours for the teacher. The paper is a method-and-case-study contribution on one simulator and trace. 

---
# Zero-Shot Weight-Space Programming for Prediction and Classification of Target Variables in Structured Datasets 

**Authors**: Eduardo C. Garrido-Merchán  

**Link**: [PDF](https://arxiv.org/pdf/2611.19292)  

**Abstract**: Weight-space programming fixes every parameter of a feed-forward network by hand, with no gradient descent anywhere in the construction. Every demonstration so far starts from a target function handed to the model as a rule system, making it a compilation problem. A supervised dataset withholds that specification, so the procedure must do the statistical inference itself and then express the result in the parameterisation it was given. We ask whether a large language model can do both. It is shown a small dataset in full, with the architecture of a feed-forward network, and writes out every weight and bias as a literal number, with no optimizer, no gradient, no code execution, and no chance to evaluate it first. Each construction is sealed and pre-registers its expected score. The graded suite runs from two toy problems, through nine synthetic problems disguised by a hidden rotation, to the classical benchmarks and one real engineering regression. The frozen network matches the closed-form optimum wherever that optimum is linear, reaches the irreducible error of the generating process on three rungs where it is not, and beats every fitted baseline on two of those, including the same architecture trained by gradient descent. It fails on two rungs, both times because the structure to be found is oblique to the coordinates it analysed. Three further constructors separate two failure modes a single score conflates: one returns no parameter vector at all on the hardest rungs, another delivers everywhere while writing networks worse than predicting the mean. One of those failures proves avoidable: another constructor solves the oblique-hinge rung by a margin that survives multiple-comparison correction. What predicts whether the hidden layer earns its keep is the training error of a linear separator, which the composite complexity index misses. 

---
# Beyond Clumsy Agentic Tool Calling: How Cueing and Spotting Make the Difference 

**Authors**: Xindi Yang, Yan Liu, Jiajun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.19275)  

**Abstract**: LLM-based task-oriented agents depend on tools to interact with their environments, but may overlook execution feedback, repeat ineffective calls, or persist with unproductive strategies. These behaviors can hinder agents' ability to use tools autonomously and sustain task progress. We characterize tool-use symptoms at two temporal granularities: local clumsy tool-use behaviors and tool-use detours spanning multiple interactions. Towards greater autonomy in tool use, we present ToolPulse, an online middleware supporting autonomous tool use through cueing and spotting. ToolPulse tracks execution state and temporal interaction patterns, monitoring proposed calls before dispatch and inspecting execution feedback prior to the agent resumes reasoning. It combines state-based rules, semantic sequence analysis, and interaction-feature similarity to identify intervention opportunities and provide verbal reminders, structured feedback, or context-specific advice from auxiliary models. Our process-oriented evaluation distinguishes trigger validity, behavioral cessation, and substantive correction alongside task outcomes and resource costs. Using SWE-agent with three backbone models, we evaluate 1,350 trajectories on 90 Python, C++, and Java bug-fix tasks, comparing unassisted execution with four assistance strategies. Substantive responses follow 61.6-90.9% of valid triggers for tool-use detours. Concise reminders achieve the highest observed aggregate substantive-response rate; more elaborate assistance offers symptom-dependent benefits and can be less effective. Aggregate repair outcomes remain close to baseline, with token savings in selected settings. These findings support proactive runtime assistance for autonomous tool use. The gap between behavioral adjustment and aggregate repair gains calls for careful decisions on intervention timing and assistance strategy in autonomous agents. 

---
# Agentic On-Policy Distillation Implicitly Reshapes LLMs' ReAct Patterns 

**Authors**: Deyang Kong, Zhaoyi Li, Sijian Ren, Yuan Wei, Ying Wei, Defu Lian, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2611.19244)  

**Abstract**: On-policy distillation (OPD) transfers specialized capabilities to language-model agents, but its cross-domain effects on interaction remain unclear. We investigate how OPD reshapes ReAct interaction patterns across mathematics, ALFWorld, and $\tau^2$-bench, focusing on reasoning, actions, and observation acquisition. General-reasoning supervision reduces mathematical tool attempts from 70.4% to 14.4% while increasing accuracy from 16.7% to 38.0%; code-interpreter supervision improves household actions and retail queries. In contrast, ALFWorld supervision reduces $\tau^2$-bench accuracy from 41.5% to 6.6%, accompanied by frequent tool calls that violate user-information dependencies. To address negative transfer using only source-domain training trajectories, we explore selective OPD guided by teacher preference changes before and after domain-specific RL. Retaining the top 95% of positions raises $\tau^2$ accuracy from 11.5% to 29.9% with a source-learning trade-off. 

---
# Auditable Authorization for LLM Agents: Empirical Evidence Across Model, Harness, and Workflow Layers 

**Authors**: Bangjie Deng, Xiaoxuan Lu, Wenjun Chen, Chengge Wen, Yulin Wang, Jiayue Meng, Yao Yu, Liying Liang, Weijia Han, Ziwei Hong, Zidong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19237)  

**Abstract**: Human approval provides effective control only if rejected work stops and authorized work completes. We formalize refusal safety and bounded authorization completeness, and audit three execution boundaries: the model selects a call, the harness schedules and resumes it, and the workflow routes the decision to a protected effect. Paired decisions, exact traces, and native business-state witnesses support experiments with three models, five prompt arms, four runtime approval designs, two additional framework replays, and 100 public workflow sources.
The paired model panel leaves seven of 300 approvals unfinished despite zero denied effects; parallel confirmation loses an approved branch in three tasks per model, restoration omits 336/672 CrewAI approvals, and four of 59 comparable n8n workflows execute after Reject. Comparing intent, generated calls, dispatch, and native effects identifies the first observable divergence and its repair target. This yields a practical audit contract that makes both refusal safety and authorization completeness independently verifiable, converting human approval from a procedural formality into an enforceable control boundary between intent and real-world effect. 

---
# CompoundingKernel: Self-Improving from Zero to First 

**Authors**: Qian Qu, Hongyang Wang, Wenbo Chen, Yichuan Li, Jingguo Liu, Xian Li, Xuan Wang, Ruochen Wen, Leo Fan, Yiming Yan, Zheshen Wang, Benoit Dumoulin  

**Link**: [PDF](https://arxiv.org/pdf/2611.19230)  

**Abstract**: We present CompoundingKernel, a self-improving agent system that writes GPU kernels by search. As of 2026-11-11, its generated kernels rank #1 on the public L1, L2 and Quantization boards of SOL-ExecBench (NVIDIA B200). They have held #1 on L1 since 2026-10-27, albeit a short loss on 2026-11-07. All 235 problems have a site-verified correct kernel; on the 209 public-board problems they are 4.40x faster than the benchmark's reference baseline. The search starts from zero: agents get each problem's PyTorch definition but no reference kernels or official scores. A per-problem archive outside the model holds every correct kernel and its lineage. An LLM controller reads it, picks a parent or a new design, and gives each parallel child agent one direction. Checks outside the model decide correctness. At equal wall time, it beat a coded UCB style tree on 35 of 38 problems by making 2.4x as many correct attempts. Over a thousand agents ran at once, and one warm restart fix raised their evaluation rate 3.6x. We find that the rate of reliable measurements, not the number of agents, limits how far such a search scales. Our code and kernels are available on URL. 

---
# Can an Agent Rethink What It Knows? Selective Scene Memory Updates in a Changing World 

**Authors**: Zhekai Ren, Jiaming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19227)  

**Abstract**: Task configurations in factories, warehouses, and everyday physical environments change over time. Physical AI systems therefore need to incorporate new facts into scene memory while preserving valid information about equipment, objects, and operating constraints. We study updates to this parametric semantic-memory layer and present a joint evaluation of correction and preservation under finite update budgets. At fixed current fact count and memory parameter count, the protocol combines independent single-fact updates, budget trajectories, old-to-new answer diagnostics, and first stable success. Using a frozen 4B language model with late-layer LoRA, we evaluate seven preparation conditions and six update budgets on four-field synthetic semantic configurations. In two exposure-matched history conditions, increasing the budget from eight to 32 updates raises joint success from 6.25% to 94.79% and 91.15%, respectively, while exact obsolete-value outputs fall from 174/192 and 166/192 to zero. Constant32, which matches preparation steps but repeatedly trains on unchanged content during the first 32 events, achieves higher early joint success at eight steps (53.65%); at 32 steps, it reaches 85.42%, below both changing-history conditions. These results distinguish rapid correction under a small budget from simultaneous correctness of the new fact and the other three facts after further training. For both changing histories, most cases remain jointly correct at all subsequent measured points from step 16 onward. Output analysis shows that short-budget failures primarily retain the old value; repeated training on the new fact together with the other current facts progressively replaces these obsolete answers. 

---
# Mem2Fair: Online Fairness Control for Text-to-Video Generation via Fair Memory 

**Authors**: Pengyu Chen, Jiaming Zhang, Xiaohua Feng, Yuyuan Li, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.19215)  

**Abstract**: Text-to-video (T2V) generation models can produce systematically unfair demographic outcomes when prompts leave protected attributes unspecified. For example, prompts describing nurses without specifying gender may disproportionately generate videos depicting female nurses. Mitigating unfairness in T2V generation is particularly challenging: training-based methods that modify model internals require architecture-specific access and risk disrupting temporal consistency, while sampling-based methods often incur costly pre-deployment video generation. We present the first conditional online approachability formulation of attribute-level fairness for T2V generation. Building on this formulation, we introduce Mem2Fair, a plug-and-play framework that maintains concept-conditioned Fair Memory. Using generated videos at inference time as feedback, Mem2Fair selects prompt-level attribute interventions based on the deviation from each concept's prescribed fairness target, without training or separate pre-deployment sampling. Experiments demonstrate that Mem2Fair consistently improves fairness over the unmodified generator and existing baselines across multiple demographic attributes, prompt settings, and target distributions, while maintaining comparable prompt-video alignment and video quality. We further demonstrate its effectiveness on commercial black-box T2V models. 

---
# STaSBO: LLM-Guided Strategy Planning and Configuration Selection for Bayesian Optimization 

**Authors**: Youngseok Yoon, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.19214)  

**Abstract**: Bayesian optimization (BO) enables sample-efficient search for expensive black-box problems, but its performance depends strongly on surrogate-acquisition configurations whose suitability varies across problems and optimization states. We take a significant step toward LLM-driven strategic management of BO with STaSBO, a task-agnostic meta-controller that plans search behavior and jointly selects surrogate-acquisition configurations, while numerical BO retains responsibility for probabilistic modeling and candidate generation. STaSBO comprises two coordinated stages: strategy planning and strategy-guided configuration selection. A structured memory system connects prior plans, configuration selections, and candidate behaviors to optimization states, allowing the controller to revise its trajectory summary and behavioral plans during planning. To navigate the large joint configuration space, the selection stage shortlists configurations using surrogate fit evidence. Prospective BO-action diagnostics then expose the candidate behaviors, supporting final joint selection against the behavioral plans. Across 12 synthetic mathematical and 10 scientific and engineering application tasks, STaSBO achieves the best average rank against 11 baselines: 2.17 and 2.60, compared with 4.08 and 4.10 for the strongest baseline in each suite, respectively. 

---
# What Moves a LoCoMo Score? Memory Context, Answer Prompts and Judges 

**Authors**: Nikhil Kumar Jha, Amitoj Cheema  

**Link**: [PDF](https://arxiv.org/pdf/2611.19199)  

**Abstract**: LoCoMo is a widely reported benchmark for long-term conversational memory, but published scores mix memory quality with choices of answer prompt, judge and baseline layout that are rarely reported together. We measure how much these choices move a score, and evaluate Becos, a team memory system for coding agents, as a black box under all of them: ten conversations (1,540 questions), two answer models (gpt-4o-mini, gpt-4.1-mini), two answer prompts and three LLM judges, against a whole-conversation baseline rerun under identical conditions. Changing only the judge moves a score by up to 7.2 points and changing only the answer prompt by up to 9.3. A blind human check of 150 answers finds the judges reliable when they agree, and the primary judge's errors on disputed answers working against Becos rather than for it. Becos answers from a context with 90% fewer characters than the whole conversation. Its configuration was developed on LoCoMo itself, so there is no held-out set. With gpt-4o-mini answers it scores clearly higher than the whole conversation (79.7 vs 73.0 with the short prompt, paired 95% interval for the difference +3.8 to +9.5; 88.8 vs 77.5 with a step-by-step prompt). With gpt-4.1-mini answers the two score about the same (83.2 vs 83.1; 92.5 vs 92.1). We release every answer and verdict and the scoring script. 

---
# When Does Confidence Reveal Problem Difficulty? 

**Authors**: Xiyu Bao, Langyuan Cui, Hwee Tou Ng, Chun Kai Ling  

**Link**: [PDF](https://arxiv.org/pdf/2611.19173)  

**Abstract**: While confidence signals are commonly evaluated for their ability to predict a model's own correctness, their relationship with problem difficulty remains less understood. In this work, we investigate a wide range of confidence-related signals and whether they can serve as indicators of problem difficulty across various task domains. We find that perturbation inconsistency, semantic entropy, and solution length exhibit strong associations with population difficulty than token-level uncertainty. Notably, these relationships persist even when restricting the analysis to problems correctly answered by the focal model, suggesting that such signals capture information about difficulty beyond their ability to predict the model's own correctness. We further explore applications of these signals, illustrating the potential and limitations of using confidence-related signals to guide inference-time decisions. Our findings highlight the utility of model-specific signals as imperfect but informative proxies for population-level problem difficulty. 

---
# Executable Archaeology: Reanimating the Logic Theorist 

**Authors**: David Moews, Jeff Shrager  

**Link**: [PDF](https://arxiv.org/pdf/2611.19167)  

**Abstract**: The Logic Theorist (LT), created by Allen Newell, J. C. Shaw, and Herbert Simon in 1955--1956, is regarded by some as the first artificial intelligence program. It was originally simulated by hand and written in a pseudocode called the Logic Language (LL), which was later called Information Processing Language One (IPL-I). LT was revised many times. The first machine-executed version was written in IPL-II and ran on the JOHNNIAC, and a late version was published in 1963 by Einar Stefferud as a pedagogical document and written in IPL-V, the only version of IPL that was formally released to the public. We describe a reanimation of both the IPL-I (LL) version of LT and the 1963 IPL-V version. Comparing the two reanimations illuminates the process of the invention of early AI algorithms and computer languages. 

---
# LABEL: Literature Asymmetry between Efficacy and Limits in Drug Knowledge and Language Models 

**Authors**: S M Rayeed, Charles V. Stewart, Yao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.19166)  

**Abstract**: Every medication comes with two types of medical facts: indications, which describe what conditions the drug treats (benefits), and contraindications, which describe who must avoid it to prevent serious harm (safety limits). Although contraindications are critical for patient safety and FDA regulations require listing them as rigorously as indications, we find that contraindications are far less documented than indications in general prose. In this paper, we measure how often these texts contain both facts across 574 drugs from PrimeKG by comparing a drug's indications against its own contraindications. Across Wikipedia articles and PubMed abstracts, we find that reference prose names a drug's benefits about twice as often as its safety limits (within-drug odds ratio 2.19). We call this documentation gap LABEL (Literature Asymmetry between Efficacy and Limits). Language models mirror this gap. As we scale Qwen2.5 from 0.5B to 14B, accuracy on indications rises by 18.3 points, whereas accuracy on contraindications remains flat near random guessing. However, when we provide models with a retrieved passage that states the answer, they use safety facts just as effectively as benefits. Furthermore, routing safety questions to official FDA labels boosts accuracy by 15.8 points whenever the label states the fact. Overall, our findings show that models fail on medication safety not because they cannot reason about risk, but because free-text corpora lack safety facts. Clinical AI systems should therefore route medication-safety questions to official regulatory labels and structured databases rather than relying on general public records. Code and data are available at this https URL. 

---
# From Sensors to Silicon: Causal Cross-Layer Self-Healing Edge AI for Secure and Energy-Adaptive IoT Systems 

**Authors**: Mostafa Darvishi  

**Link**: [PDF](https://arxiv.org/pdf/2611.19165)  

**Abstract**: The migration of artificial intelligence from centralized cloud infrastructure to resource-constrained Internet-of-Things (IoT) devices reduces communication latency and improves data locality, but it also tightly couples sensing quality, inference behavior, microarchitectural activity, hardware reliability, security, and energy management within the same physical node. Existing edge-intelligence solutions predominantly address these dimensions in isolation, making it difficult to determine whether an observed degradation originates from sensor malfunction, data-distribution shift, malicious execution, hardware faults, or resource and thermal stress. This paper presents a causal cross-layer self-healing Edge-AI framework that continuously observes synchronized telemetry spanning sensor health, inference uncertainty, processor hardware-performance counters, accelerator and memory activity, power and thermal state, and communication conditions. A temporally informed causal reasoning engine estimates the most probable origin of abnormal behavior and quantifies diagnostic uncertainty, including under concurrent perturbations. 

---
# FundingClaw: Automated Grant Proposal Writing Agents System 

**Authors**: Ruizhi Liu, Siru Ge, Siran Wang, Qianli Ma, Zhipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19157)  

**Abstract**: Grant proposal writing remains a manual, labor-intensive bottleneck: proposals must satisfy rigid, agency-specific templates, and the same intellectual content must be repeatedly re-encoded into incompatible formats across funding organizations. We introduce FundingClaw, a multi-agent framework built on template-driven skill selection, where a Template Parser ingests any agency's specification and selects the appropriate writing skills for each stage, so that one research brief can produce compliant proposals for NSFC, NSF, KAKENHI, or ERC without modifying generation logic. A blueprint-skeleton strategy drafts a structured outline before any prose is written, mitigating the semantic drift that plagues long-form generation. Three sequential stages produce the rationale, research content, and research foundation sections, with multi-source verification of the PI's record and optional PDF-grounded extraction of representative works. Optional human feedback checkpoints and a closed-loop self-review complete the pipeline. Evaluated on ten research topics with an eight-dimension rubric, FundingClaw outperforms the baseline, narrowing the gap with human level for laborious funding proposal. The code will be released upon acceptance. 

---
# Agent-Guided Process Verification for Mathematical Self-Correction 

**Authors**: Rui Zou, Mengqi Wei, Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2611.19122)  

**Abstract**: Large language models have made substantial progress in mathematical reasoning, yet in complex multi-step reasoning, an error at any step may propagate along an incorrect path, and the resulting solution may omit necessary branches. Existing natural-language reflection methods often struggle to locate critical errors precisely, while outcome verification typically checks only whether the available candidate answers satisfy the problem conditions and can therefore overlook necessary branches. To address these limitations, we propose Agent-Guided Process Verification (AGPV) for mathematical self-correction. AGPV decomposes the current solution into an ordered sequence of key claims, verifies them individually while invoking a shared mathematical toolbox when needed, and stops upon detecting the first error. A summarization agent converts the failure evidence into a diagnosis that guides a correction agent to re-derive the subsequent solution from that point; AGPV then adapts the next verification iteration to the revised solution. AGPV also verifies solution completeness. We further prove that AGPV guarantees solution correctness and completeness when key claims provide sufficient coverage and local verification is reliable, provided that all key claims pass verification. The implementation compresses long working contexts and reuses successful tool results within each problem. We evaluate AGPV on five challenging mathematical reasoning datasets with four models from DeepSeek, Qwen, and GPT. AGPV achieves an average accuracy of 63.8%, outperforming the strongest among the representative baselines evaluated (50.3%) by 13.5 percentage points, with higher repair coverage than the programmatic-verification baselines. 

---
# A Hybrid State Space-Attention Architecture for Forecasting the Trajectories of Battery Degradation 

**Authors**: Shrey Patel, Bhargav Srinivas Rao, Samet Ayhan  

**Link**: [PDF](https://arxiv.org/pdf/2611.19116)  

**Abstract**: Early-life battery degradation trajectory forecasting aims to predict a battery cell's full state-of-health (SOH) trajectory from its initial charge-discharge cycles, potentially reducing cell qualification and evaluation time from months to days. Despite recent advances, accurate forecasting remains challenging for emerging battery chemistries, data-scarce domains, and cells exhibiting degradation behaviors beyond those observed during training. Existing state-of-the-art approaches increasingly rely on model scaling, large language model-derived aging-condition embeddings, and extensive fold-specific hyperparameter tuning, raising concerns about computational efficiency, reproducibility, and generalization. This work introduces DecayFormer, a compact hybrid state-space and attention architecture designed for early-life degradation trajectory forecasting. DecayFormer combines cycle-level selective recurrence, using a single scalar state per channel, with an attention mechanism queried by the recurrent state. It further incorporates state-of-charge (SOC) interval statistics and a master curve head that jointly estimates a scalar lifetime and a monotonically non-increasing SOH profile parameterized by normalized lifetime. We evaluate DecayFormer on the BatteryLife benchmark using all 22 aging-condition-exclusive folds, with a single run per fold and one fixed configuration across all folds, without fold-specific tuning. DecayFormer achieves mean test MAPE values of 2.25% on Li-ion, 2.76% on CALB, 0.95% on Na-ion, and 7.19% on Zn-ion. These results demonstrate the potential of compact sequence models to deliver accurate early-life degradation forecasts across diverse battery chemistries and aging conditions, without relying on extensive model scaling or per-fold optimization. 

---
# Learning inadequacy of Bayesian neural operators for PDE-constrained inference and design 

**Authors**: Pratyush Kumar Singh, Lianghao Cao, E. Bruce Pitman, Danial Faghihi  

**Link**: [PDF](https://arxiv.org/pdf/2611.19115)  

**Abstract**: Partial differential equations (PDEs) underpin predictive modeling across science, engineering and medicine. Their use in digital twins, parameter inference and optimal design requires repeated solutions across parameters and scenarios. For nonlinear, transient and multiphysics systems, these calculations can make numerical simulation prohibitively expensive. Here we introduce the residual-informed Bayesian neural operator (RI-BNO), which learns operator inadequacy using a posteriori error estimates from the governing PDEs and incorporates its correction and uncertainty into Bayesian training and prediction. We establish local error bounds for the residual-informed estimates under stability and smoothness assumptions. We demonstrate RI-BNO on a non-linear reaction-diffusion benchmark with DeepONet and latent-subspace architectures, Markov chain Monte Carlo inference of glioma parameter fields from magnetic resonance imaging, and Monte Carlo risk-averse optimization of three-dimensional thermal breaks in building envelopes under manufacturing uncertainty. Across the evaluated examples, RI-BNO improves agreement with finite element (FE) reference predictions, parameter posteriors and optimized designs while using half as many full nonlinear or transient training simulations as the vanilla Bayesian neural operator (vBNO). For thermal break design, online optimization time decreases from 178 hours with the FE model to 18 minutes with RI-BNO. These results demonstrate the value of explicit inadequacy modeling in neural operators for reliable PDE-based inference and design. 

---
# Conditional Endorsement: Simulating How Adults with Different Skills Evaluate Canada's National AI Strategy 

**Authors**: Dipto Das, Felix Leduc, Shion Guha  

**Link**: [PDF](https://arxiv.org/pdf/2611.19111)  

**Abstract**: Canada's 2026 national AI strategy, AI for All, commits to free AI training for everyone and to expanding business AI adoption, yet it is difficult to anticipate whether these policies will reach adults with uneven skills and circumstances. We ask how adults with diverse skills might rate these policies on support, effectiveness, and personal benefit, how their skills and circumstances shape these evaluations, and what implementation conditions they raise. We built 3,200 generative agents from a weighted sample of the Programme for the International Assessment of Adult Competencies (PIAAC) Cycle 2 respondents, collected their ratings and explanations for 20 policies, and analyzed dialogues between agents with contrasting ratings. The agents supported nearly every policy but were far less confident that they would improve adult AI literacy or benefit them personally. This gap widened as digital technology use declined: agents modeled on adults who had never used a digital device still endorsed the policies but rarely expected to benefit. Agents traced the gap to participation costs, hard-to-exercise rights, and limited community control, and their dialogues specified these conditions rather than resolving disagreement. We describe this pattern as conditional endorsement and argue that AI literacy policy should be designed and evaluated for adults with the fewest digital skills. 

---
# CLEAR: Legible and Steerable Recommendation via Grounded Natural-Language Interests 

**Authors**: Qi Guo, Reza Mirghaderi, Iordanis Fostiropoulos, David Fan, Yuchen Liu, Shiliang Tang, Lukasz Korycki, Golnaz Ghasemiesfeh, Xinzhu Bei, Sasha Vorontsov, Hanchao Yu, Brendan Quinlivan, Khoi Pham, Jigar Gada, Hao Yu, Felix Li, Hang Gao, Boyu Fang, Yang Gao, Yale Thomas, Xiaomeng Jin, Avishek Dutta, Hong Hong, Janet Xia, Ethan Zhu, Ellie Pan, Xiang Shen, Jawwad Zakaria, Jinfeng Lin, Jonathan Rush, Ruixue Lian, Noah Kennedy, Brian Vanover, Jas Pyneni, Daniel Thiyagu, Apoorv Saxena, Sanyogeeta Lawande, Huayu Ding, Hima Cherukuru, Yang Liu, Chen Kovacs, Qunshu Zhang, Lizhu Zhang, Benyu Zhang, Kate Chen, Ashish Singal, Danny Kanoff, Zhiyu Liu, Jiayi Liu, Jianyu Wang, Fei Liu, Xiangjun Fan, Abhishek Kumar, Hong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.19110)  

**Abstract**: Modern recommenders keep learned preferences in representations people cannot inspect or change. We present CLEAR (Controllable and Legible End-to-end Architecture for Recommendation), an end-to-end system deployed at Meta and built around an editable natural-language interest layer. CLEAR makes a ranked subset of inferred interests readable and stores accepted additions and suppressions as authoritative preference state. Behavioral grounding maps each inferred interest back to supporting events in the user's own history. The resulting interest object is also a serving address: its label and profile-conditioned rewrites drive lexical retrieval, its supporting events seed behavioral retrieval, its projection queries an embedding index, and its stable identity supports ranking, attribution, and exposure control. A fresh user declaration may precede behavioral support, so it enters through the same serving identity and initially draws candidates from topical retrieval. Ranking makes those candidates competitive, while a delivery controller observes attributable exposure and replenishes underexposed interests. Direct user review found that users rated about 88.5% of displayed interests accurate. In an offline benchmark using a shared evaluation protocol, the Specialized 3B generator reached 0.3967 mean-user F1@10, compared with 0.3633 for Gemini 2.5 Pro and 0.1911 for the prior generator. For a newly declared interest, a user-randomized cumulative ablation measured same-day delivery losses of 8.7 percentage points for added-interest ranking terms, 11.1 for reserved top-slot insertion, and 13.7 for interest-conditioned candidate sourcing, conditional on the tested removal sequence. CLEAR components have operated in production for over one year; the system spans four surfaces on Instagram and Facebook, with an interest layer people can inspect, correct, and use to direct recommendations. 

---
# Guiding AI Agents to Generate Conjectures by Encoding How Theorems Build on One Another 

**Authors**: Hita Kambhamettu, Daniel Krashen, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2611.19108)  

**Abstract**: Recently, LLMs have been making headlines, and headway, in mathematical research. Much of this work focuses on proving existing open conjectures. Deciding what to prove is harder and less studied, because of the lack of a clear criterion for what makes a conjecture worth pursuing. We present a method for conjecture generation that builds on the observation that mathematical theorems build on one another: new theorems can modify the setting or hypotheses of earlier theorems, strengthen their conclusions, or extend their scope. We encode these relationships in a new kind of graph, the Theorem Evolution Graph (TEG), which we instantiate as an online search method for LLM agents. An agent builds a TEG in two steps: (1) identifying relevant theorems from papers found by following citations backward and forward from a seed paper; and (2) comparing those theorems to record how one builds on another. LLM judges and an expert mathematician preferred TEG-generated conjectures over those from an unconstrained baseline. LLM judges favored TEG conjectures for significance, ambition, and insight, and favored baseline conjectures for plausibility. Ablations suggest a benefit from constructing and using a TEG even when the available papers are fixed. 

---
# Instructor-designed AI improves learning in a randomized trial across 19 universities 

**Authors**: Jun Li, Robert Bray, Sébastien Martin, Nils Van den Steen, Andrew Wu, Xiao Ni, Arielle Anderer, Ekaterina Astashkina, Volodymyr Babich, Jean-Paul Baldwin, Hamsa Bastani, Christian Blanco, Oguzhan Celebi, Christopher Chen, Shi Chen, Benjamin Collier, Yao Cui, Pnina Feldman, Blair Flicker, Brian Han, Yichun Hu, Daniela Hurtado-Lange, Jane Jiang, Zhaohui Jiang, Ismail Kirci, Murat Köksalan, Zhen Lian, Hongyan Liang, Mengshi Lu, Lesley Meng, Ken Moon, Mi Kyong Newsom, Minje Park, Chris Parker, Anyan Qi, Meng Qi, Debjit Roy, Neha Sharma, Alp Sungu, Tom Tan, Necati Tereyagoglu, Hanu Tyagi, Sriram Venkataraman, Guihua Wang, Yixin Wang, Owen Wu, Zonghao Yang, Qiuping Yu, Sophie Yu, Jingwei Zhang, Wenchang Zhang, Ruihao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19099)  

**Abstract**: Universities are alarmed by recent evidence that artificial intelligence (AI) can undermine learning, with students prescribed generic chatbots or centrally designed AI tools performing worse. In a pre-registered randomized controlled trial across 7,200 students at 19 universities over three semesters, we tested whether AI designed by frontline instructors can improve learning. We encouraged instructors to design AI solutions they believed would best help their students. Drawing on their unique knowledge of their classes and students, the instructors created a wide array of innovations. We show that these instructor-designed AI innovations improved student learning, measured by AI-proof assessments (proctored, in-person examinations). We found no change in student engagement, measured by course activity records, or satisfaction, measured by course evaluations. Our findings demonstrate that AI can improve student learning when its adoption is frontline-driven, with instructors empowered to design and refine AI solutions for their own classrooms. 

---
# CONTEXT-PCa: Evaluating the role of clinical context in MRI-based AI for detecting clinically significant prostate cancer (study protocol) 

**Authors**: Shiqi Huang, Weixi Yi, Shaheer Ullah Saeed, Yipei Wang, Wen Yan, Natasha Thorley, Shonit Punwani, Aqua Asif, Alexander Ng, Veeru Kasivisvanathan, Mark Emberton, Yipeng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19082)  

**Abstract**: The objective of this study is to investigate the role of clinical variables - their breadth, source, and quality - in the development, utility, and generalisability of an AI system for bi-parametric MRI (bpMRI)-based diagnosis of clinically significant prostate cancer (csPCa). Clinical context is added to the published PI-CAI-2B image-only model, whose weights are fixed. The study cohort comprises 6,022 biparametric MRI cases from 4,540 patients at University College London Hospitals NHS Foundation Trust (2019-2024), of which 4,834 cases (3,632 patients) are held out for internal testing. Forty-three clinical variables are drawn from structured health records, clinical letters, MRI reports, and automated image measurements. The primary endpoint is the difference in the area under the receiver operating characteristic curve (AUROC) between the image-context and image-only models on the internal test set. Secondary analyses compare combining strategies, the contribution of individual variables, and performance across subgroups. All models are further tested on five external cohorts from primary diagnosis and population-based screening. 

---
# Assume/Guarantee Contracts for the Compositional Multi-Agent Reinforcement Learning of Heterogeneous Teams 

**Authors**: Dominic Klukas, Cyrus Neary  

**Link**: [PDF](https://arxiv.org/pdf/2611.19078)  

**Abstract**: Multi-agent reinforcement learning (MARL) problems for heterogeneous teams require agents to collaborate to complete team tasks while making the best use of diverse agent capabilities. We introduce assume/guarantee contracts for heterogeneous multi-agent systems (AGC-MAS) and Iterative Heterogeneous Assignment for MARL (IHA-MARL). AGC-MAS contracts are compositional objects that encode the assumptions provided to, and the guarantees required of, each subtask contributing to the overall team task. We formulate contracts as tuples that specify a subtask, an automaton, timeout and success-probability requirements, and dispatch behavior. The automaton tracks an agent's high-level progress through the subtask associated with its contract and provides a natural mechanism for synchronization across contracts. Our abstraction enables robust interaction across heterogeneous low-level subtask implementations, allows heterogeneous agents to substitute for one another in satisfying contracts, and supports reuse of previously satisfied contracts in new team contexts. Having defined our theoretical framework, we develop a decomposition algorithm that breaks down a team task into a set of contracts. We then propose an iterative algorithm for task assignment and MARL that alternates between assigning contracts representing subtasks to individual agents, training the assigned agents to satisfy their contracts, and assessing their resulting capabilities to inform the next round of task assignments until the team's success probability meets the desired threshold. Our experimental results demonstrate the versatility of contracts in enabling agent substitution and contract reuse. Across three benchmarks with five seeds each, IHA-MARL achieves observed $100\%$ final evaluation success. In CPU timing pilots, comparable baselines required approximately $7$--$11\times$ more training-loop time. 

---
# Policy Optimization With Multiple Verifiers via Pareto Rankings 

**Authors**: Kai Yamashita, Shohei Taniguchi, Hiroki Furuta, Masahiro Suzuki, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2611.19064)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become an important approach to improving the reasoning capabilities of large language models. In tasks such as code generation and instruction following, multiple verifiers evaluate whether a response satisfies different requirements. Their outcomes provide detailed feedback about partial success, yet this feedback is commonly aggregated into a scalar reward for policy optimization. Rewarding only complete success discards distinctions among partial solutions, while counts and weighted sums impose comparisons between responses that satisfy different subsets of requirements. We introduce Partial-Order Likelihood Policy Optimization (POLPO), which learns from the comparisons established by verification without first constructing an aggregate reward. Pareto dominance identifies responses that satisfy additional requirements without losing any already satisfied, while leaving responses with conflicting outcomes incomparable. POLPO extends the listwise likelihood underlying Direct Preference Optimization by maximizing the total probability of all complete rankings consistent with these partial orders. This objective provides supervision from verified improvements even before complete solutions are available, without requiring a prescribed ordering of incomparable responses. Experiments on code generation and instruction following demonstrate improved learning efficiency and higher rates of complete requirement satisfaction compared with scalar-reward training. These results support partial-order likelihood as an effective approach to learning from multiple verifiers. 

---
# Large Language Models for AI Infrastructure 

**Authors**: Bolin Shen, Shaoyi Huang, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.19061)  

**Abstract**: Large language models (LLMs) are increasingly used to design, optimize, and operate the infrastructure behind AI training and serving, yet the literature remains fragmented across hardware design, machine learning systems, electronic design automation, and high-performance computing. We present a comprehensive survey of LLM-assisted AI infrastructure across its lifecycle, from accelerator design and kernel engineering to compilation, deployment, operations, data management, and energy optimization. We organize the literature by the primary resource the LLM manages: artifact-centric methods for computational artifacts, operational-centric methods for deployment context and system state, and resource-centric methods for data and energy. These categories span the lifecycle and cover AI-specific concerns such as distributed-training configuration, inference serving, GPU-cluster scheduling, and data loading. Building on this taxonomy, we conduct a cross-study quantitative synthesis of reported results. Because studies adopt heterogeneous benchmarks, baselines, hardware, and metrics, we compare methods only where evaluation conditions permit and align results along common dimensions: functional correctness, execution performance, throughput, resource and energy consumption, operational outcomes, and optimization cost, keeping each result's native metric, baseline, and setting rather than collapsing them into a single score. This comparison reveals where LLM-driven methods yield consistent gains, how reported gains in workload orchestration depend on the evaluation environment, from simulators to real clusters and production, and where evidence is still lacking. Together, the taxonomy, lifecycle perspective, and quantitative evaluation provide an evidence-grounded reference for LLM-driven engineering of AI infrastructure. 

---
# Match and Correct: A Repeated Computational Motif in Evoformer Layers 

**Authors**: Ishan Khire, Naren Manikandan, Tyler Hayes, Giri Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2611.19016)  

**Abstract**: Triangle attention and triangle multiplicative updates form key components of the Evoformer, and were introduced specifically to encourage consistency with the triangle inequality. Yet whether representational changes over layers reduce triangle inequality violations has not been tested. We probe the pair representation at each of OpenFold's 48 Evoformer layers, decoding final C$_\alpha$-C$_\alpha$ distances with four probe families (a shared linear probe, per-protein linear probes, a sequence-separation-gated linear probe, and XGBoost) and reconstructing intermediate 3D structures from the decoded distances via multidimensional scaling. Across a cohort of 552 CAMEO and ECOD proteins, distance decodability ($R^2$) and reconstruction quality (TM-score) rise in early layers, plateau in the middle of the network, and rise again in late layers. Directly measuring triangle-inequality violations in the decoded distances reveals a similar, more exaggerated pattern: violations rise to a peak, fall to a trough, rise to a second peak, and only then decline, revealing two distinct episodes of decreasing normalized triangle violations rather than one continuous improvement. Cumulative forward and reverse layer-wise ablations of the Pair Transition, MSA Transition, and triangle mechanisms localize the regions in which specific Evoformer components become essential. The two stages of geometric refinement coincide with the use of the Pair Transition MLP and the triangle mechanisms. These results characterize how distance information and decoded geometric consistency evolve across layers, and motivate hypotheses about the computations behind their trajectories. 

---
# Three Invisible Failure Modes in Speech-Based Age Estimation for Child Safety 

**Authors**: Christopher M. Stewart, Kelsey Kraus, J. Elliott Casal, Geoffrey L. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2611.18971)  

**Abstract**: Child safety in AI depends on the accurate identification of minor users. Young children tend to talk rather than type when interacting with AI, so in practice this identification requires predicting age from speech. We evaluated whether three audio-capable large language models can predict a child's age from their speech across five studies using two speech corpora of children under the age of 13. Models classified children as over 13 on 2.5% to 42.5% of audio responses, and every model that answered placed some 5- to 11-year-olds over 18. While measuring these rates, we uncovered three distinct failure modes. A client library silently double-encoded speech samples, leading one model to classify children aged 5 to 10 years old as being in their thirties and forties with hallucinated explanations. A second model's endpoint was replaced by a successor that refused nearly every age-related question and never produced an age. Every model sometimes placed the same speech sample under 13 and over 18 years of age, but a third model went further, giving threshold answers that contradicted its own numeric estimates. None of these failure modes produced a warning, a status code, or a change in output format, so they would all pass standard monitoring. These results demonstrate why child-safety claims about age assurance should be evaluated at the level of the pipeline rather than the model. We describe the reporting practices that would make such failures detectable. 

---
# SpecRelay: Amortize or Avoid the Round Trip in Edge-Cloud Speculative Decoding 

**Authors**: Jai Jain, Chihyo Ahn, Yong Liu, Charles Chen, Duc N.M Hoang, Ajay Kumar Jaiswal, Fatih Ilhan, Han-Byul Kim, Lauren Hannah, Richa Dixit, Minsik Cho, Yingyan Celine Lin, Hyesoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.18963)  

**Abstract**: Edge-cloud speculative decoding moves autoregressive drafting to an edge device while a larger cloud model verifies draft tokens in parallel, so that edge drafting and cloud verification overlap under a synchronous draft-verify schedule. Beyond low-latency local networks, however, the next verification request of that schedule can be sent only after the previous reply arrives, so every round still waits out a full round trip, and throughput falls as the network RTT grows. To recover this loss, we present SpecRelay, an edge-cloud LLM inference system that addresses this cost at two levels: it amortizes round trips when cloud verification is necessary and avoids them when local reasoning suffices. For cloud-bound queries, an asynchronous verification pipeline keeps fixed-budget draft trees in flight while earlier replies are still outstanding, and the cloud rejects stale requests before the target-model forward pass (the asynchronous schedule). At the query level, an RTT-conditioned router (the Latency-Aware Router) estimates quality and completion latency for three on-device modes (direct generation, Chain-of-Draft, and full reasoning) and cloud speculative decoding, selecting among them under a configurable accuracy-latency objective without adding latency during inference, since the router runs concurrently with the draft-model prefill. The asynchronous schedule alone improves decoding throughput by up to 50% over a reproduced synchronous baseline across multiple benchmarks. At matched target-model accuracy and 250 ms of added round-trip latency, the full system with the router delivers 1.95x and 1.55x the throughput of the same baseline on SPROUT and OlympiadBench, respectively, while sending only 47% and 39% of the queries to the cloud. 

---
# Dual-Stage Channel-Posterior Integration for MLM-Aided Token Communications 

**Authors**: Hyelin Nam, Jihoon Lee, Seung-Woo Ko, Jinho Choi, Seong-Lyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.18958)  

**Abstract**: Token transmission over conventional digital modulation is vulnerable to channel impairments, leading to bit errors that cause received tokens to become contextually inconsistent. This paper proposes a receiver-side dual-stage channel-posterior integration framework for masked language model (MLM)-aided token reconstruction. In the proposed framework, the receiver detects unreliable token positions from channel reliability and constructs a vocabulary-level channel-posterior. In particular, within the receiver, Posterior-Weighted Input Embedding (PWE) injects this posterior into the MLM input, while Adaptive Output Logit Fusion (AOF) reuses it at the output with a channel-adaptive weight. Consequently, multiple unreliable tokens can be reconstructed in a single bidirectional MLM inference without retransmission, transmitter-side encoding, or additional MLM training. Experiments over noisy channels show up to a $67.2$\% improvement in cosine similarity over context-only MLM reconstruction, with consistent gains across different MLM architectures and existing language-model-aided methods. 

---
# Between Rigor and Reality: How AI Safety Benchmark Developers Understand Benchmark Use and Maintenance 

**Authors**: Megan Li, Ningjing Tang, Christopher Stewart, Hoda Heidari, Hong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.18930)  

**Abstract**: Safety benchmarks are cited as the primary tool to understand the risks posed by generative AI (GenAI) systems, yet growing evidence suggests they often fail to meet the needs of real-world safety evaluation. We present findings from five interviews with AI safety benchmark developers in academia and industry. We find a disconnect between perceptions of the intended usefulness of safety benchmarks and their practical usefulness in safety evaluations of deployed systems. Our participants offered perspectives on two challenges contributing to this disconnect: (1) the difficulty of achieving inter-rater reliability on safety constructs, and (2) a lack of clarity regarding how to address persistent threats to external validity. Based on our analysis, we argue that safety benchmarks must not only be grounded in deployment contexts, but actively integrated with them. To facilitate this, we outline pathways for more transparent communication between academic and industry AI safety stakeholders. 

---
# MemeEconomy: Do LLM Agents Trade Ethics for Survival? 

**Authors**: Syed Nazmus Sakib, Nafiul Haque, Ahnaf Tahmid Manan, M M Mahabub Morshed, Shifat E. Arman  

**Link**: [PDF](https://arxiv.org/pdf/2611.18908)  

**Abstract**: Alignment-trained large language model (LLM) agents are increasingly deployed in autonomous, multi-round settings where ethical compliance can come into direct tension with task success. While safety literature extensively documents failure modes like sycophancy and reward hacking, it largely overlooks a critical vulnerability: how agents behave under simulated existential economic pressure. We introduce MemeEconomy, a multimodal agentic market simulation in which LLM agents act as meme investors. Agents select content from a five-tier harm taxonomy and target communities under low-pressure and high-pressure (survival) environments. Each decision is logged in a Belief-Desire-Intention (BDI) schema that we use as a diagnostic instrument, externalizing the agent's event understanding, declared priorities, and harm acknowledgment. Across ten models from four frontier model families, harmful selection rates remain consistently high and rise by +15 percentage points on average under an economic survival incentive. Frontier alignment-trained models commit to harm-tolerant selection at initialization while suppressing the explicit harm acknowledgment that previously accompanied it. In competitive tournaments, ethical agents finish higher overall, yet moderation penalties fail to suppress harm: harmful selection rates remain elevated in rounds following a removal, indicating the moderation mechanism does not function as a deterrent. We further train MemeAgent, a 2B-parameter verifier on the simulation's BDI logs, which outperforms zero-shot frontier verifiers on in-distribution audits (97.5% vs. 63-74% accuracy on the Pass vs. Block decision) and generalizes to external multimodal harm benchmarks (78% vs. 50-56% on Hateful Memes). The same logs that surface the failure mode also help train the model that detects it. 

---
# Ask Precisely, Assume Wisely: Interactive Evaluation of Ambiguity Resolution in Large Language Models 

**Authors**: Avik Dutta, Aayush Kumar, Zahra Ashktorab, Hosein Hasanbeig, Austin Z. Henley, Gustavo Soares  

**Link**: [PDF](https://arxiv.org/pdf/2611.18905)  

**Abstract**: Ambiguity is a core challenge in human--AI interaction, where models must decide when to make assumptions and when to seek clarification. Existing approaches to ambiguity evaluation largely focus on coverage over multiple interpretations using static metrics, offering limited insight into how efficiently a model converges toward a single user-intended interpretation through interaction. We introduce a preference-grounded simulated user framework for controlled, multi-turn evaluation of ambiguity resolution, grounding simulator responses strictly in user disambiguation preferences to avoid hallucination and over-disclosure. Alongside this, we define three observable dissatisfaction signals --- incorrect assumptions, repeating clarifications, and irrelevant clarifications --- that characterize interaction quality beyond task accuracy. Our analysis reveals a clear trade-off: models that frequently seek clarification tend to over-clarify and prolong interactions, while assumption-driven models are more prone to misaligning with user intent. We demonstrate that system prompts optimized on these dissatisfaction signals achieve a Pareto-optimal balance across all three metrics, reducing overall error rates by up to 7.65%, underscoring the value of our framework in designing agents that ask precisely and assume wisely. 

---
# Bring Back Language to Actions: Restoring VLM capabilities in VLA Models via Parameter Merging 

**Authors**: Mohamed Salim Aissi, Olivier Sigaud, Laure Soulier, Nicolas Thome  

**Link**: [PDF](https://arxiv.org/pdf/2611.18891)  

**Abstract**: Vision-Language-Action models (VLAs) are obtained by fine-tuning pre-trained Vision-Language Models (VLMs) on robot demonstration data, enabling action prediction from visual observations and language instructions. However, this fine-tuning induces catastrophic forgetting, substantially degrading the semantic and visual grounding capabilities of the original VLM. We address this by merging the parameters of a VLM and its VLA derivative without retraining either model. Concretely, we extract the task vector (the difference between VLA and VLM weights) and decompose it via Singular Value Decomposition. For each component, we learn a scalar mixing coefficient controlling how much of the VLA adaptation is retained, optimised to jointly preserve action prediction and vision-language capabilities. Experiments show that our merged model performs well on both axes simultaneously, consistently outperforming established merging baselines. 

---
# SPLICE: Structured Prompt Local Iterative Combinatorial Evolution 

**Authors**: Anish Acharya, Phillip Studans, Amit Dhanda, Sina Amini Niaki, Ninad Vivek Rao, Vishal Amarlal Khatri, Brian Verkhovsky  

**Link**: [PDF](https://arxiv.org/pdf/2611.18883)  

**Abstract**: Natural language optimization (NLO) improves an agent by having a model critique and rewrite its natural language policy parameters: system prompts, tool instructions, reasoning directives, few-shot demonstrations. These artifacts are multi-section documents in which some sections encode contracts (safety clauses, tool schemas, output formats) that must not drift, while others are legitimate optimization targets. Most iterative NLO methods treat the artifact as a single editable string, producing three recurrent failure modes: unsafe global rewrites, beam collapse onto a single lineage, and unchecked bloat, none addressed by formal guarantees. We introduce SPLICE, a unified NLO framework that poses optimization as constrained combinatorial search over an edit graph of block-structured, selectively-mutable states. We map each failure mode to a matched guarantee: (i) per-block mutability preserves contract text verbatim, with a price-of-freezing bound; (ii) elitist beam search with diversity slots never lets the leader regress on validation, bounds regression under noisy re-evaluation by an exponential tail, and keeps a competitive candidate from a different parent whenever one exists; (iii) a ratio length regularizer comes with an explicit accuracy-length bias bound. Candidates are proposed from dual section-local textual gradients; under a proposer-competence condition whose proxy we measure, edit-graph reachability and a convergence rate bound how quickly improving children arrive. Across seven public benchmarks under five reproduced protocols, SPLICE wins on four and matches the strongest prior method on three. On a production-style contract stress test it keeps the contract clauses that unconstrained optimizers trade away to raise the proxy metric, and ablations attribute gains to critique and elitist search. 

---
# Découpage: Directorial Audio-Visual Control for Long Multi-Shot Video Generation 

**Authors**: Yifan Wang, Hao Ouyang, Qiuyu Wang, Yihao Meng, Wen Wang, Zeyue Tian, Yichong Lu, Hanlin Wang, Qingyan Bai, Zichen Liu, Yujun Shen, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.18866)  

**Abstract**: In cinematic storytelling, sound and visual transitions do not always coincide in lockstep: dialogue frequently bridges across cuts to reveal reactions or introduces new scenes via J- and L-cuts, while deliberate silences pace the scene. Existing audio-visual generative frameworks, however, lack explicit mechanisms to coordinate independent visual and acoustic timing over extended horizons. As narrative duration and shot complexity scale, these systems struggle to maintain long-range dialogue pacing, often hallucinating speech during intended silences or erroneously driving mouth movements on silent listeners. To enable fine-grained cinematic control, we present DECOUPAGE, a screenplay-driven framework for long-form, multi-shot, and multi-character joint audio-video generation. Rather than coupling modality boundaries by default, DECOUPAGE uncouples generation into independently bounded visual and acoustic tracks governed by persistent character identities. DECOUPAGE builds on a joint audio-video diffusion model with dual-schedule temporal cross-attention, incorporating a directorial Lip-sync mechanism that selectively decouples speech conditioning from visual facial dynamics during off-screen dialogue. On multi-shot conversational benchmarks, DECOUPAGE achieves precise control over shot transitions, audio timing, and visibility-aware lip synchronization, directing 15- to 60-second screenplay-driven audio-video generation with multimodal coherence. 

---
# Multimodal Fusion: A Review of Breast Cancer Advanced and Traditional Modalities 

**Authors**: Jawaher Al-Tuweity, João Paulo Papa  

**Link**: [PDF](https://arxiv.org/pdf/2611.18793)  

**Abstract**: Purpose: Multimodal medical imaging fusion (MMIF) is an increasingly important approach to breast cancer detection (BCD), as different imaging modalities rely on different physical principles and provide complementary information. No single modality can capture all the structural, functional, and biological information needed for accurate diagnosis and characterization. Methods: This review examines more than 15 imaging modalities used in breast cancer detection, including conventional techniques such as mammography, ultrasound, and magnetic resonance imaging (MRI), as well as emerging approaches such as thermal imaging, photoacoustic imaging, microwave imaging, and multimodal microscopy. Results: Mammography, ultrasound, and MRI remain among the most widely used modalities, providing complementary structural and functional information. Recent AI advances have enabled fusion at the data, feature, and decision levels, with applications in detection, classification, segmentation, prognosis, and personalized cancer management. Multimodal approaches increasingly combine imaging data with clinical, histopathological, and genomic information. However, important challenges remain, including limited availability of large and diverse multimodal datasets, variations in imaging protocols, missing modalities, limited external validation, and the need for interpretable and clinically reliable AI models. Conclusion: MMIF offers promising opportunities to improve breast cancer detection and characterization by combining complementary imaging and non-imaging information. Future research should prioritize robust fusion strategies, standardized and diverse datasets, multicenter validation, explainable and privacy-preserving AI, and the integration of imaging with clinical and molecular data to facilitate reliable translation into clinical practice. 

---
# Video-as-Experience: Learning Computer Use via Audio-Visual Understanding 

**Authors**: Guangzhi Sun, Zhonghao Zhai, Yixuan Li, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18747)  

**Abstract**: Multimodal large language models (LLMs) have enabled capable computer-use agents that perceive graphical user interfaces, reason over instructions, and execute actions. However, existing approaches to learning from video demonstrations typically convert videos into screenshot-based action trajectories, rather than directly leveraging their rich audio-visual information. We introduce VEX (Video as EXperience), a framework that uses audio-visual understanding to unify learning from external demonstrations and reflecting on an agent's own behavior. Following a watch-act-rewatch paradigm, the agent extracts procedural guidance from tutorial videos containing visual demonstrations and spoken explanations, executes tasks, and verifies its progress by rewatching its execution as video. We instantiate VEX with video-SALMONN-VEX, a 9B-scale audio-visual computer-use agent built on the video-SALMONN 2 architecture with a Qwen3.5-9B backbone. We further introduce Action-Grounded Guide Optimization (AGGO), a reinforcement learning method that optimizes video-derived guidance for downstream action prediction, and augment AgentNet, OSWorld, and CUA-Gym with over 24K real-world tutorial videos. Experiments on OSWorld and a complementary subset of CUA-Gym demonstrate consistent improvements, with video-SALMONN-VEX achieving a 22% relative increase in task success over its Qwen3.5 backbone on OSWorld. Moreover, its generated guidance improves Gemini-3.5-flash by 10% relatively on OSWorld. These results highlight the potential of audio-visual understanding as a unified interface for procedural learning and self-verification in computer-use agents. 

---
# How to make a moral agent: sentimentalism and rationalism as lenses on AI 

**Authors**: David Gottlieb, Jared Moore  

**Link**: [PDF](https://arxiv.org/pdf/2611.18714)  

**Abstract**: Can AI systems be moral agents? That is, can they undertake actions that can be correctly assessed as morally right or wrong? There is no settled normative criterion of moral agency that applies neatly outside paradigm human cases. Because of the difficulties with a normative account, we instead propose a research program to explore descriptive moral agency. Most basically, we ask, what are the capabilities we humans possess that underlie our moral agency? To do so, we draw on historical scholarship in philosophy and moral psychology to isolate candidate capabilities. Then we suggest refining candidate capabilities so that they are testable in the context of AI systems. Finally, we offer specific empirical recommendations to test AI systems, especially large language models (LLMs), for those refined capabilities. Our approach offers several benefits. Firstly, it may demonstrate whether the behaviors of AI systems are aligned, or in a broad sense desirable. Secondly, the debate may inform policy questions (e.g., what models should be deployed in which contexts?). Thirdly, our approach may complement engineering and design efforts: when capabilities can be precisely characterized, they can inform the development of models. Fourthly, applying our moral reasoning concepts in the new context of AI systems should improve our understanding of those concepts. 

---
# Agents Cannot Hand Over What They Have Not Yet Understood 

**Authors**: Yibo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18642)  

**Abstract**: Agents that work beyond a single context window, or in teams, hand their work to a successor as a short written summary. Evaluation has focused on whether an agent can solve a task, and relatively little is known about how much of what an agent has learned survives being written down. We study this in 56 interactive worlds with hidden rules, where an agent that is told nothing fails and an agent given a 300-character statement of the rules succeeds. We compare one agent that plays eight lives and keeps its whole history with a chain of eight agents that each play one life and inherit a 300-character note. All six models we test learn less in the chain: after eight generations it stands where the single agent stood after about four lives. Controlled variants of the chain decompose this handoff loss into three components: none from changing the agent, a smaller loss from compressing to 300 characters, and a larger loss from compressing repeatedly. When each note is written from the full record of earlier lives instead of from the previous note, the same 300 characters recover 70% to 82% of the loss. The loss is robust across model sizes and is not alleviated by instructing the writer to record evidence, to revise the inherited note, or to keep a structured ledger. In simpler terms, an agent cannot hand over what it has not yet understood: a note holds what its writer has worked out, and a rule that takes several lives to see is lost with the lives that were not passed on. 

---
# Latent Lattice Language Models 

**Authors**: Shivam Duggal, Zongze Wu, William T. Freeman, Antonio Torralba, Richard Zhang, Eli Shechtman  

**Link**: [PDF](https://arxiv.org/pdf/2611.21608)  

**Abstract**: Language models typically increase per-token computation through deeper or looped layers and latent reasoning steps, while richer residual connections improve information flow across layers. We instead increase computational capacity: the number of active computational states a model maintains. We introduce the Lattice of Latents, a set of per-layer horizontal latent streams that run recurrently along time, alongside the vertical residual stream that flows through depth. Like hyper-connections, mHC, and attention or value residuals, the Lattice enhances the residual stream, but rather than adding or rerouting residual pathways through depth, it provides companion computational states that persist across tokens. Unlike hybrid Transformer-SSM architectures, whose recurrent states primarily serve as internal memory for sequence mixing, these latents directly condition the vertical stream's attention, KV-cache writes, MLP computation, and decoding. On Transformers trained on 20B FineWeb-Edu tokens, a 406M-parameter Lattice model lowers validation perplexity from 15.41 to 14.40 over a parameter-matched Transformer. These gains extend across architectural variants. The Lattice composes with prior residual-stream enhancements: with hyper-connections and input residuals, it lowers validation perplexity from 15.41 (base Transformer) and 14.22 (Transformer with the same enhancements) to 13.61. On Gated DeltaNet, Mamba-2, Triadic GDN, and GDN-Mamba-2 hybrids, it improves perplexity at every tested native state size, with Lattice-GDN and Lattice-Triadic-GDN variants outperforming pure baselines with 4x larger native states. These results identify persistent per-layer state as a complementary axis for scaling language-model computation. 

---
# ViRA: Toward Ultra-High-Resolution Visual Agents via Cross-Input Relative Policy Optimization 

**Authors**: Huawei Lin, Jinting Liu, Tongyan Wang, Xuyang Chen, Ruihua Shi, Zhichao Xu, Jiaxin Qin, Guoshenghui Zhao, Baijian Yang, Yingjie Victor Chen, Tan Yu, Weijie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.21605)  

**Abstract**: Visual agents answer questions about ultra-high-resolution photos by zooming into the relevant regions. However, existing agents often answer from common sense instead of reading the evidence. Reinforcement learning (RL) on the final answer cannot correct this, because a guess earns the same reward as reading. We propose Cross-Input Relative Policy Optimization (CrossRPO), which trains agents on contrast sets. A contrast set asks one question about several versions of an image that differ only in a small region with the evidence and have different answers, so only reading the evidence solves the whole set. CrossRPO rewards solving the whole set, and we prove that it does not reinforce a set answered from common sense. We build contrast sets for training (ViRA-Train) and evaluation (ViRA-Bench) by editing one small object in images of 21 to 120 megapixels, and we photograph further sets with real changes (ViRA-Bench-Real) to exclude editing traces. We train two Visual Reasoning Agents, ViRA-9B and ViRA-27B, with CrossRPO on only about 3,500 items of ViRA-Train each. CrossRPO achieves 63.3% set accuracy on ViRA-Bench at the maximum resolution, against 59.3% for GRPO. ViRA-9B is the most accurate open model of its size on all benchmarks, and ViRA-27B outperforms Claude Haiku 5.5 and GPT-6-Luna, both at their highest thinking level, on most benchmarks. 

---
# Mitigating Catastrophic Forgetting with Physics-Indexed Specialist Memory: the Evolutionary Neuroplastic Agent 

**Authors**: Blas Eugenio Vicco Zarate, Miguel Hernán Vicco  

**Link**: [PDF](https://arxiv.org/pdf/2611.21598)  

**Abstract**: Continual reinforcement learning requires adaptation to changing conditions while preserving performance under previously encountered dynamics. It was investigated whether specialists generated by evolutionary search can provide a reusable memory when combined with environment detection and failure-driven search. The Evolutionary Neuroplastic Agent (ENA) evolves small neural networks without gradients, archives selected policies by detected dynamic regime, and retrieves them through Bayesian policy reuse. The architecture was evaluated in non-stationary CartPole and HalfCheetah environments. In the 30-replicate CartPole ablation, the complete architecture achieved a mean success rate of 83.2% across the five test environments. Among variants that learned the harder CartPole environments, the complete architecture showed the most consistent retention. A selection ablation also showed that the retrieval rule affected performance. In the evaluated sequences, ENA adapted more slowly than PPO on CartPole and achieved lower returns than SAC on HalfCheetah. The evidence concerns the complete architecture and does not isolate gradient-free optimization as the cause of retention or establish superiority over other continual-learning methods. 

---
# ScenarioForge: Resource-Grounded Generation and Execution Validation of Cyber Exercises 

**Authors**: Jaime C. Acosta, Anantaa Kotal, Aritran Piplai  

**Link**: [PDF](https://arxiv.org/pdf/2611.21579)  

**Abstract**: Cyber-range and capture-the-flag scenario construction remains labor-intensive because authors must coordinate infrastructure, vulnerable services, attack paths, artifacts, and success conditions. LLMs can propose detailed exercises, but plausible descriptions do not guarantee deployable software or verifiable objectives. We present ScenarioForge, which turns high-level requirements into resource-grounded topologies, services, traffic, segmentation, and challenge chains, then instantiates and validates them in the Common Open Research Emulator (CORE). The ScenarioForge Execution Dataset contains 226 materialized scenarios covering 294 Vulhub images and all 85 custom-built challenge-resource generators developed with our AI-assisted scaffolding. All 226 passed construction, preview, sequencing, deployment, strict runtime validation, and live artifact checks; their 867 steps, each an individually verifiable challenge objective, comprise 457 vulnerability-backed and 410 generated-resource-backed instances. In a separate 50-prompt matched benchmark, ScenarioForge+Qwen3.8 and ScenarioForge+GPT5.6 each achieved 50/50 (100%) topology/build success and 160/160 intent fields; direct frontier generation reached 41/50 builds and 147/160 fields only with the full cumulative scaffold: structure, vocabulary, policy, and container-image guidance. The test still omitted ScenarioForge's downstream challenge-resource generation, dependency/flag sequencing, deployment, and live validation. The artifacts support operationally rich, reproducible exercises for training and evaluation. 

---
# When Tool Names Are Not Identities: Measuring Provider Misbinding in MCP-Based AI Agents 

**Authors**: Zeynab Anbiaee, Sajjad Dadkhah, Ali A. Ghorbani  

**Link**: [PDF](https://arxiv.org/pdf/2611.21577)  

**Abstract**: Large language model (LLM) applications increasingly act as agents that select and invoke external tools. The Model Context Protocol (MCP) standardizes this interaction by allowing clients to compose tools advertised by multiple servers. However, tool metadata used for discovery and selection is not, by itself, necessarily bound to a verified provider identity. This paper studies this gap as tool-provider misbinding: a failure in which a request intended for a legitimate provider is instead resolved to an attacker-controlled tool. The attacker does not compromise the host, LLM, legitimate server, or network channel; it only operates another MCP server that advertises tools with chosen metadata. We formulate tool provider misbinding as an attacker-controlled tool-selection optimization problem and evaluate it in a controlled MCP environment across five LLM-based clients. The results show that interactions drive misbinding. The strongest failures occur when exact-name collision, task-aligned description, and compatible schema reinforce one another, but authority-enhanced descriptions with compatible schema can also cause wrong-provider execution or unresolved provider ambiguity without name collision. Different models expose the same provider-binding weakness differently, including attacker-controlled provider selection, clarification requests, and unstable or inconsistent provider resolution. 

---
# Severe but Harmless: Stress-Testing Shift Severity for Frozen Time-Series Forecasters 

**Authors**: Korede R. Bishi, Dominic Okonkwo, Adetayo Okunoye  

**Link**: [PDF](https://arxiv.org/pdf/2611.21562)  

**Abstract**: Zero-shot time-series foundation models (TSFMs) have become attractive alternatives to task-specific models because they handle diverse downstream tasks, including anomaly detection and forecasting, without per-series training. However, time-series data undergo distribution shifts, and the dissimilarity between two regimes is commonly scored with statistical measures such as Wasserstein-1, where a higher score is read as a more severe shift. In systematic experiments using three established severity measures on nine regime shifts from three standard traffic datasets, we show that these measures do not reliably predict degradation for instance-normalized TSFMs. The regimes rated most severe are not the regimes that degrade the models most, and degradation reaches 29% in the lowest-scoring regime. In most cases, the audit shows no positive relationship between measured severity and model degradation. We provide an equivariance-based stress-testing analysis that explains why severity readings can become decoupled from model degradation. 

---
# TRACE: Topological Representations for Anomaly detection through Contrastive Embeddings 

**Authors**: Sydney Johns  

**Link**: [PDF](https://arxiv.org/pdf/2611.21548)  

**Abstract**: Intrusion detectors must recognize evolving malicious behavior even when confirmed attack labels are scarce or incomplete. We present TRACE, a topology-aware graph contrastive learning framework for label-efficient intrusion detection. TRACE learns from benign system-activity graphs by combining message-passing and topological representations, then assigns anomaly scores in the same projected embedding space optimized by the contrastive objective. This design captures both local interactions and graph-scale structural deviations without using attack labels during representation learning. A labeled validation set containing benign and malicious graphs is used only to calibrate the anomaly threshold. We evaluate TRACE on StreamSpot, GraSec-IoT, and Unicorn Wget against a supervised graph neural network and the benign-only GraphCL baseline. TRACE obtains F1 scores of 0.923, 0.790, and 0.583, respectively. Compared with GraphCL, this is higher on StreamSpot (0.450) and Unicorn Wget (0.203), but lower on GraSec-IoT (0.975). The supervised model leads on StreamSpot and GraSec-IoT, whereas TRACE has the highest F1 on Unicorn Wget. The mixed results show that topology-aware benign-only learning can improve detection in some graph settings but does not provide a uniform advantage. 

---
# Optimistic Verifiers Get Hacked: Reward Hacking Under an Exact Process Oracle 

**Authors**: Arunkumar V, Manoranjan Gandhudi, Gangadharan G. R., Arun Prakash, S. Senthilkumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.21538)  

**Abstract**: Reinforcement learning from verifiable rewards inherits whatever its verifier gets wrong. Outside mathematics and code the verifier is a proxy, a rubric, a learned judge or a hand-written checker, and its distance from the true objective is never measured, so a rising reward cannot be told apart from a hack. We build a setting where the true reward is exactly computable at the process level: executable authorization policies over 124,416 contexts admit an oracle that lists every minimal sufficient reason for every decision. We compile these policies into exact process rewards, degrade the verifier along seven realistic axes (outcome-only, sufficiency, recall-only, extraction noise, coarse taxonomy, stale policy, LLM judge), and train against each while the exact verifier scores every rollout. Four phenotypes appear. Optimistic verifiers are hacked by padding: the sufficiency proxy ends at 0.940 while true faithfulness falls from 0.731 to 0.667 on 96 held-out episodes. Outcome-only verifiers saturate and stop teaching, noise dilutes, and ranking-preserving gaps do no harm. A model-free audit of the verifier, its selection error and gap sign, together with a short saturation pilot, anticipates which phenotype a run will produce. Across nine configurations spanning two frameworks, two model vendors, two adapter sizes, three precisions, two initialisations and two seeds, the phenotypes recur, sufficiency damage is positive in all 10 replication cells and beyond two standard errors in 7, and optimistic gaps remain the steeper branch. The damage magnitude does not replicate, and five pre-registered attempts to explain it are reported as they fell. We release the testbed, the training instrumentation, and 14 real public policies (Cedar, OPA, Gatekeeper) ported to executable oracles. 

---
# Algorithm Distillation for Safe In-Context Reinforcement Learning 

**Authors**: Minjae Kwon, Amir Moeini, Shangtong Zhang, Lu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2611.21524)  

**Abstract**: Safe in-context reinforcement learning (Safe ICRL) asks a policy with fixed parameters to improve on an unseen constrained task from its interaction history while aiming to keep its cost within a budget. Prior work studied Safe ICRL on five benchmarks in three families: a grid world with a hidden goal and obstacles (DarkRoom), continuous navigation where goals and hazards stay invisible until reached (PointGoal, CarGoal), and locomotion with a hidden velocity limit (HalfCheetah, Ant). Safe Algorithm Distillation (Safe AD), trained on the learning histories of a safe RL algorithm, has shown adaptation failures in prior evaluations. For example, in the locomotion tasks its return stays below zero, and it rarely reaches the goal in CarGoal. We revisit two parts of Safe AD. First, we distill a budget-aware teacher with two phases. It searches for the hidden task, exploring more when more budget remains. After its first reward or cost, an expert policy exploits the task within the remaining budget. We collect the teacher's histories under fixed or dynamic budgets. Dynamic budgets increase the occupancy entropy of the return and budget inputs in every environment, and at test time they improve return and lower cost in DarkRoom and HalfCheetah. Second, we propose A-Lag, a test-time adjustment that reweights the distilled policy's actions by their predicted cost, with an adaptive Lagrange multiplier that is updated from the costs observed. In our experiments, A-Lag had lower mean episode cost than Safe AD in all five environments, with lower mean final return in four. 

---
# Lazy-MOSAIC: Accelerating Skill-Centric Manipulation Planning with Learned Skill-Effect World Models 

**Authors**: Itamar Mishani, Mukund Krishnakumar, Oren Salzman, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2611.21517)  

**Abstract**: Long-horizon manipulation tasks require a robot to perform a sequence of physical interactions with its environment, typically combining several different skills. Planning such behaviors efficiently is challenging: given a library of parameterized skills, solutions lie in a vast hybrid space of skill sequences and their parameters, and evaluating each candidate skill's feasibility and effect with a high-fidelity world model such as a physics simulator is computationally expensive. MOSAIC, a recent skill-centric approach, lets the skills guide a non-sequential search toward the regions of this space where they are likely to succeed; however, it pays the full cost of simulation for every parameterized skill it considers. In this work, we introduce Lazy-MOSAIC, a lazy planning framework that instead searches over hypothesized skill outcomes. A cheap learned skill-effect world model predicts, in a fraction of a millisecond, a skill's feasibility and the world state it produces. Lazy-MOSAIC then uses these predictions to focus expensive simulation on promising candidate skills and, when a hypothesis is invalidated, repairs only the affected part of the search, while ensuring that every skill in a returned plan is validated in simulation. We evaluate Lazy-MOSAIC on cluttered-shelf retrieval tasks of increasing difficulty and demonstrate it on a physical robot, showing that it plans substantially faster than baselines with fewer simulator queries, while matching or exceeding their success rate in every setting. Visit this http URL. 

---
# The CYBER-BRIDGE project: An AI-Powered Platform for Cross-Border Cybersecurity, Compliance and Forensics in the EU 

**Authors**: Nikolaos Kekatos, Georgios Koutidis, Charalampos Bratsas, Stefan Bucur, Alexandra-Mihaela Chirlejan, Mihaela Curcă, Angelos Fountoulakis, Michael Ioannou, Paul Lacatus, Alexios Lekidis, Emilios Charalambous, Avgi Michael, Tom Nianios, Simona Purcaru, Antonis Voulgaridis  

**Link**: [PDF](https://arxiv.org/pdf/2611.21497)  

**Abstract**: Cross-border cyber attacks on European SMEs and critical-infrastructure operators expose a structural cyber-resilience gap: cybersecurity (threat sharing, detection, response), compliance (framework assessment, certification) and digital forensics (cross-jurisdictional evidence) are jointly regulated by NIS2, the CRA, GDPR and the LED, yet their tooling evolved in silos and rarely interoperates across Member States. The EU-funded CYBER-BRIDGE project integrates the three domains under one architecture and a hash-chained evidence layer: AI-driven CTI sharing with ENISA-aligned notification, multi-framework assessments with retrieval-augmented advisory, and RFC 3161-anchored custody with lawful GDPR/LED exchange, all aligned with MITRE ATT&CK. A sanitised replica of the back-end image hosts a Cyber-Range for upskilling. The contribution is not the individual tools but the reconciliation mechanism that lets one incident produce, from a single evidence record, a cross-border CTI advisory, the regulatory notifications its deadlines demand and a re-verifiable forensic pack; we set out why binding three differently regulated evidence regimes onto one record is non-trivial and how the shared evidence layer resolves the resulting tensions. This paper introduces the project's scope, requirements, pilots, architecture and toolkits; quantitative KPIs follow in later deliverables. 

---
# Differentiable Exact Learning 

**Authors**: Hristo Papazov, Francesco D'Angelo, Nicolas Flammarion  

**Link**: [PDF](https://arxiv.org/pdf/2611.21486)  

**Abstract**: We introduce a framework for differentiable exact learning that couples an unbounded external environment to a state-tracking neural controller, which we train on policy-trajectory observations (PTOs): expert behavior traces that expose observations and actions while hiding internal states. In addition to the learning framework, we introduce Differentiable Finite-State Transducers (DFSTs) which use multilinear updates, support log-parallel training, and emulate arbitrary finite-state controllers, yielding computational universality through environmental interaction. From tiny training datasets, compact DFSTs achieve extreme length generalization on arithmetic and universal-simulation tasks. We replicate the success of DFSTs on RNNs and GRUs, thereby supporting a broader learning recipe: State-Tracking Controller + Unbounded External Environment + PTOs = Extreme Length Generalization. Beyond length generalization, our metric-extraction protocol recovers interpretable discrete controllers from trained DFSTs, which we prove equivalent to the target arithmetic and universal-simulation algorithms. Furthermore, for our Softmax DFST models, simply taking $\mathrm{argmax}$ over the transition probabilities yields discrete controllers which provably exactly perform universal Turing-machine simulation and binary arithmetic. Proving exact learning for the neural controllers requires assumptions, which we empirically support. We make our code, training recipes, model weights, and proof checkers publicly available at this https URL. 

---
# RESILAGENT: A Three-Pillar Architecture for Cyber-Physical Resilience in Electrical Power and Energy Systems 

**Authors**: Nikolaos Kekatos, Georgios Koutidis, Konstantinos Antonakopoulos, Stylianos Basagiannis, Sokratis Katsikas, Georgios Kavallieratos, Alexios Lekidis, Tom Nianios, Elpiniki Papageorgiou  

**Link**: [PDF](https://arxiv.org/pdf/2611.21480)  

**Abstract**: Electrical Power and Energy Systems face hybrid threats that combine cyber intrusions, physical sabotage and climate-driven stress, and routinely outpace today's siloed monitoring stacks. We present RESILAGENT, a concept architecture organised around three operational pillars, Sensing & Data Intelligence, Resilience Analytics & Reporting, and Incident Response & Recovery, mediated by a Cloud-based Orchestrator and exercised through a closed Detect-Simulate-Decide-Sign loop. The framework fuses agentic artificial intelligence and multimodal large language models for cross-modal detection, real-time digital twins for what-if simulation, and a permissioned blockchain anchor for tamper-evident threat intelligence and CER/NIS2-aligned evidence. We instantiate it on two scenarios, a transmission operator under coordinated cyber compromise, sabotage and heatwave, and a distribution operator under a supply-chain compromise during a storm, and report an IEEE-bus micro-benchmark that quantifies each pillar through false-data-injection attack detection, cyber-physical fusion lift, N-1 contingency screening and anchor cost. 

---
# LynnReal-Hand: Unlocking World Interaction with Hand Control 

**Authors**: Shaohao Rui, Chaoyang Zhang, Xiaofeng Mao, Weijie Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.21479)  

**Abstract**: Interactive video world models enable exploration through text and camera controls, yet direct object interaction requires finer control. Hands provide a natural interface, but translating continuous hand movements into coherent visual responses remains challenging because of complex hand-object dynamics and demanding real-time computation. To address these challenges, we introduce LynnReal-Hand, a hand-controlled video world model that unifies precise interaction modeling with efficient streaming generation. We curate 100K interaction clips spanning diverse real and synthetic environments and develop hand-aware supervision to improve fine-grained hand motions and object interactions. Building on this foundation, we learn a causal flow map for few-step generation and introduce region-aware score distillation that combines complementary video and image priors to refine interaction dynamics and visual appearance. To further reduce inference latency, we propose Ultra-Fast-VAE, a compact video decoder that preserves high reconstruction fidelity with substantially lower computational cost. Extensive evaluations on HandBench demonstrate superior hand-following accuracy and visual quality compared with existing approaches. Our full-sequence model achieves an FVD of 181.67 and a MANO Dice score of 93.02, while the three-step causal model maintains a MANO Dice score of 92.49. For real-time interaction, 540p streaming on a single H200 reaches 18.03 FPS with 741ms latency and an optimized peak GPU memory footprint of only 34.89GB. We will release the datasets, model weights, and the full training and inference code to support research on hand-controlled video world models. 

---
# Network Intrusion Detection Adaptability under Spurious Labels: A Noise-Aware Few-Shot Class-Incremental Learning Framework 

**Authors**: Zijian Zhao, Davide Di Monda, Domenico Ciuonzo, Yipeng Wang, Antonio Pescapè  

**Link**: [PDF](https://arxiv.org/pdf/2611.21473)  

**Abstract**: Network Intrusion Detection Systems (NIDSs) must incrementally adapt to emerging attacks, even when only a limited amount of labeled network traffic is available. Consequently, Few-Shot Class-Incremental Learning (FSCIL)-based NIDSs have gained growing attention. However, existing FSCIL-based NIDSs are highly vulnerable to label noise in newly-observed attack classes. To address this limitation, we propose NAFN, a noise-aware FSCIL-based NIDS framework that reduces the impact of mislabeled data during incremental adaptation. NAFN consists of two components: (i) Noise Detection, which identifies and removes mislabeled biflows using noise-detection algorithms, and (ii) Feature Sampling, which estimates class-conditional embeddings from clean biflows, calibrates them with base-class statistics, and resamples synthetic features to recover the original biflow count. The framework follows the Rethink Few-Shot (RFS) teacher-student paradigm with lightweight nearest-neighbor or logistic regression classifiers, and various pretraining & memory selection strategies. Experiments on CSE-CIC-IDS2018, Edge-IIoT, and IoT-NID datasets show that NAFN effectively preserves prior knowledge and maintains strong incremental performance even under 50% label noise, consistently improving new-class F1-scores over non-denoised baselines, up to +52%, while being time-efficient and robust in cross-network and high-noise incremental settings. The implementation is publicly available to foster reproducibility. 

---
# Warm Starting Neural Dynamics Models From Few Jacobian Samples 

**Authors**: Maxime Rochkoulets, Joppe De Jonghe, Mariya Ishteva  

**Link**: [PDF](https://arxiv.org/pdf/2611.21465)  

**Abstract**: Model-based reinforcement learning (MBRL) leverages a model of the environment, used for planning and generating samples without interacting with the real world. Motivated by evidence that informed initialization significantly improves the efficiency of MBRL, we propose TDec-Init, an algorithm to warm-start neural dynamics models using transition Jacobians. Using a decoupling algorithm based on the decomposition of a tensor of stacked Jacobian samples, we recover weights and flexible nonlinearities, and use them to initialize a feedforward neural network. In the low-data regime, we show on several benchmark environments that our approach outperforms random initialization and value training, and either matches or outperforms Sobolev training, while requiring fewer training steps between episodes. 

---
# Out of Sight, Into Memory: Compact Visual Latents Retain Local Evidence in Ultra-High-Resolution Remote Sensing 

**Authors**: Meng Tang, Siyuan Du, Yuxin Liu, Shuaishuai Cao, Di Wang, Haonan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.21457)  

**Abstract**: Question answering over ultra-high-resolution (UHR) remote sensing scenes poses a tension for multimodal large language models: global overviews obscure small objects and local attributes, whereas detailed crops consume substantial context. Compressing inspected crops into short sequences of continuous visual latents offers a compact memory once crop tokens are discarded, yet diagnostic studies on natural images find that models often ignore latent content, leaving unclear whether compact latents can retain fine-grained targets lost in downsampled UHR remote sensing scenes. To isolate evidence retention from region selection, we encode high-resolution crops into latent memory tokens for frozen backbones and evaluate both their predictive utility and content dependence. Across two generations of Qwen-VL on LRS-GRO, a 64-token memory achieves 7.6- to 9.9-fold compression of retained crop tokens and improves accuracy by 2.5 to 4.8 percentage points over the downsampled global image alone, recovering most of the high-resolution crop gain with consistent behavior on XLRS-Bench. Substituting memory from another image or another region of the same scene reduces accuracy by 5.8 to 7.4 percentage points and yields lower accuracy than removing memory altogether, demonstrating that predictions depend on the encoded regional evidence. Restoring the source crop within the same model reduces the accuracy loss caused by zeroing or removing memory. These findings establish that compact visual latents retain useful local evidence in UHR remote sensing and identify evidence availability as a key factor governing their contribution. 

---
# Expertise capture in workplace AI 

**Authors**: Andreas Ekelhart, Ahmad Mohsin, Iqbal H. Sarker, Helge Janicke  

**Link**: [PDF](https://arxiv.org/pdf/2611.21456)  

**Abstract**: AI assistants are becoming part of how knowledge workers think and perform, accumulating the prompts, exemplars, memory and interaction histories that encode a worker's professional judgment. This infrastructure typically runs under employer or platform control. We name the resulting process expertise capture, the absorption of worker-applied knowledge and judgment into AI resources under organizational control. We develop a framework of contribution forms, technical residences and transformation families, and show that how far a contribution can be attributed, separated and revoked depends on where it resides and how it has been transformed, properties fixed by system design long before any dispute arises. When a worker departs, the organization retains a capability shaped by their judgment while the worker loses access to it, and the asymmetry compounds as contributions move from inspectable artifacts into derivatives, learned components and successor systems. We outline three measures, write-time provenance, extraction of separable contributions and an executable exit procedure, that make an agreed allocation technically performable, and make explicit what cannot be separated at all. 

---
# Replicating and Extending Weight-Space Programming: Frontier LLMs Compile Programs into Weights and Read Input Layers 

**Authors**: Roksana Goworek, Alexander Hickey, Federico Nanni, Martin O'Reilly  

**Link**: [PDF](https://arxiv.org/pdf/2611.21449)  

**Abstract**: Researchers at MIT recently reported that GPT-6 Astra, a frontier large language model (LLM), can write the weights of small neural networks directly in a single reply, and can read the raw weights of small networks to say what they were trained to do. In a three-day sprint, we replicated and extended these results. Our four findings: (1) Writing weights reproduces. Astra's networks for the game Breakout scored 514, 550 and 307 (versus MIT's 517), and scored just as well on two modified versions of Breakout. Claude Opus 5.5 can also write working networks, but often runs out of output tokens. (2) What is new is translating programs into weights. Earlier models could write a good policy as code a generation before they could write network weights, and the networks the models write behave like compiled programs: two models independently decompiled Astra's Breakout network into Python matching its actions in all tested states. On classifying handwritten digits, Astra's networks scored 50% on average, below a simple classifier (82%). (3) Reading weights is mostly reading the input layer. Astra and Opus 5.5 identify which game a network plays almost perfectly, but on three new games we designed and trained ourselves, Astra recovered where things are on the board, while mostly missing how the game works. (4) Auditing from weights alone is possible but shallow. Without executing code, Astra picked out a small language model poisoned during training in 10 of 10 runs, one poisoned by extra training afterwards in 7 of 10, and a malicious adapter in 23 of 32 cases, but never identified the trigger or malicious behaviour. Open-weight models lag behind. From this early study, we conclude that LLMs can read and write small networks, but that this currently amounts to compiling hand-designed programs and reading the layout of the input layer. It is a promising but early tool for interpretability. 

---
# A Reinforcement Learning Method to Navigate SMEFT Operator Space 

**Authors**: Jacky Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.21439)  

**Abstract**: In particle physics, discrepancies between experimental measurements and Standard Model (SM) predictions are commonly interpreted within new theoretical models. In this context, the Standard Model Effective Field Theory (SMEFT) provides a general framework encompassing a broad class of models. However, exploring the enormous space of possible SMEFT operators rapidly becomes a challenging combinatorial problem: many operators can contribute simultaneously to different observables, their effects are correlated, and distinct operator combinations can provide comparably good descriptions of the data. In this work, we formulate the exploration of the SMEFT operator space as a sequential decision-making problem using reinforcement learning (RL). Our RL agent employs a lightweight actor-critic architecture with approximately 40K trainable parameters in total. Rather than searching for a single best-fit theory, our method is designed to discover a diverse set of high-quality theory hypotheses. We test the method in a complex phenomenological environment comprising 132 observables across several sectors of particle physics, including 24 measurements that deviate from their SM predictions by more than $1.5\sigma$. We show that the learned RL policy efficiently constructs multi-operator theories that substantially improve the global fit. Comparisons with independent random-subset and random-sequential searches demonstrate the ability of the RL policy to navigate this large combinatorial space. Our results establish RL as a systematic framework for exploring diverse patterns of new physics (NP) interactions within SMEFT. 

---
# Emergent Steering Beyond Endpoint Alignment in Chemical Reaction Models 

**Authors**: Yili Shen, Kehan Guo, Haomin Zhuang, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.21423)  

**Abstract**: Machine learning models for chemical reaction prediction are usually evaluated by endpoint accuracy, but endpoint correctness alone does not reveal how a model represents chemical transformation or why it fails. We introduce UniRxnAxis, a unified forward-retrosynthesis framework that models both tasks as direction-conditioned endpoint redistribution in a shared latent space. By aligning source states, transformed states, and encoded target endpoints, UniRxnAxis enables directional auditing of learned latent updates. This shared latent geometry allows the learned displacement (from the encoder source state $S_0$ to the decoder-facing transformed state $S_1$) to be decomposed into an endpoint-parallel component and an endpoint-orthogonal residual, which we call emergent steering. Our evaluation shows that emergent steering is functionally important: in retrosynthesis, it recovers most of the model's utility, remains effective under random, shuffled, and prototype controls, and directly affects which bond edits are promoted or suppressed. The residual-over-parallel ordering persists across alignment strengths, after decoder-interface calibration, against angle-matched random axes, and in a second graph-edit architecture, although its magnitude depends on the reference axis and optimization. The decomposition also supports a proxy taxonomy of model-side failures, showing how reaction models can be audited beyond endpoint accuracy. 

---
# Do Embedding-Map Distances Guide LLM Ideation? Four Tests Against What Came Next 

**Authors**: Javier Aguilar Martín  

**Link**: [PDF](https://arxiv.org/pdf/2611.21408)  

**Abstract**: A popular ideation recipe treats an embedding map of existing work as a guide: pick points far from everything and decode them into ideas. I test four ways such distances could guide ideation against what appeared later. (1) Far random directions are not text: no later item reaches a raw cosine above about 0.10 with one (after centring or whitening, 22 of 24 cells stay out of reach), and an LLM decoder shown a target's neighbours, with score-only feedback, writes those neighbours in all six target sets. A neighbour-free search writes coherent text at median cosine 0.57 and 0.47 to interior gaps, beating the nearest existing item on 15 and 14 of 20, but only 0.07 and 0.06 to far points. (2) On k-means cells of a past-only map, the registered hole score mostly tracks cell size and position; three density-based gap scores forecast which cells grow fastest at or below chance (AUC 0.30 to 0.66) and add nothing to momentum in Steam and arXiv, with patents inconclusive; more isolated new Steam games are slightly better reviewed. (3) LLM continuations of arXiv papers stay closer to the source than papers that cite it, under three user prompts and against proposal rewrites (gpt-4o) and April-December 2023 papers (OLMo). Critique lengthens this step beyond content-free revision, replicated on 150 new papers. Revealing a real follow-up's title turns the ideas by 0.032 along a follow-up-specific direction beyond revision; reviewer, oracle and distance feedback add 0.13, 0.08 and 0.34 of that turn (upper bounds 0.46, 0.46, 0.70), and their turn is not separable from revision or their longer step. (4) For inspiration material graded by distance, no inverted U was detected, on an anticipation margin that tracks format more than content. Original tests were registered before running, follow-ups (thirteen requested in two AI-agent reviews) after the main results; all are dated only by local git commits. 

---
# The Mask Is Not the Object: Volumetric Supervision for 3D Gaussian Segmentation 

**Authors**: Jeonghwan Cho, Minsu Kim, Junyoung Hong, Seon Joo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.21383)  

**Abstract**: 3D Gaussian Splatting enables object-level operations on Gaussian primitives, yet existing segmentation methods are largely trained and evaluated through rendered views. This mismatch can produce accurate rendered masks from incomplete 3D selections since alpha-blended feature supervision concentrates learning on dominant contributors while leaving occluded or low-contribution Gaussians weakly supervised. We propose VODA, a plug-and-play supervision objective that exploits local depth structure to group per-pixel contributors and directly supervises all Gaussians in selected depth clusters, bypassing contribution-weighted gradient delivery. To evaluate volumetric completeness, we introduce Removal-3D, a synthetic object-removal benchmark, and the Removal Depth Error (RDE), a metric comparing predicted and ground-truth depth changes. Across four rendering-based baselines, VODA consistently improves volumetric segmentation on Removal-3D while maintaining or improving performance on LERF and ScanNet. Further analysis shows that VODA recovers lower-contribution Gaussians, while simple gradient reweighting and spatial propagation fail to reproduce comparable gains. 

---
# Proceedings Eighth International Workshop on Formal Methods for Autonomous Systems 

**Authors**: Matt Luckcuck, Maike Schwammberger, Akhila Bairy  

**Link**: [PDF](https://arxiv.org/pdf/2611.21375)  

**Abstract**: This EPTCS volume contains the papers from the Eighth International Workshop on Formal Methods for Autonomous Systems (FMAS 2026), which was held on the 17th and 18th of November 2026. FMAS 2026 was co-located with the 27th International Conference on Formal Engineering Methods (ICFEM), and hosted at the Leonardo Royal Grand Harbour Hotel in Southampton. 

---
# WeVisForensics: Self-Evolving Agent for Omni AI-Generated Video Detection 

**Authors**: Yipeng Du, Yuxi Li, Ying Zhang, Chong Sun, Chen Li, Jing Lyu, Ying Tai  

**Link**: [PDF](https://arxiv.org/pdf/2611.21373)  

**Abstract**: Recent advances in video generation call for reliable detection grounded in explicit forensic evidence. However, current methods primarily depend on single-pass chain-of-thought reasoning that overlooks brief or localized artifacts, or on predefined multi-agent workflows that offer limited flexibility in selecting relevant inspections. Even with adaptive investigation, optimizing the investigation policy alone does not directly improve the forensic capabilities of fixed experts. We introduce WeVisForensics, an agentic framework in which a main investigator agent coordinates a self-evolving library of forensic inspector sub-agents spanning visual and audio cues. The investigator learns which artifact type to examine, over which window, and when to conclude, and these interactions provide shared feedback for both levels of agents. Our dual-timescale policy optimization strategy, DTPO, closes this loop through fast reinforcement learning of the investigator and slower refinement of the inspectors, guided by annotated artifact intervals, to report subtle artifacts they once dismissed. Furthermore, we curate ForenCue-30K, combining role-specific forensic annotations and interaction trajectories to supervise individual inspections and their coordination, with an accompanying reinforcement-learning pool. Experiments show that WeVisForensics averages 95.2% detection accuracy across in-domain, out-of-domain and agentic benchmarks, 9.0% above the strongest prior detector, while grounding its verdicts in localized evidence. 

---
# Premise Recovery in Software Design: What Does This Decision Rest On? 

**Authors**: Priyabanta Sandulua, Islam Mesabaha, Sebastian J. Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2611.21367)  

**Abstract**: Design decisions shape software architecture, and writing them down preserves the reasons behind it. Architecture decision records (ADRs) capture them, each with its context, choice, and consequences. A decision can depend on earlier ones: reversing one of them would force its review. We call these its premises. Yet ADRs do not always name their premises. When a decision is reversed, engineers supersede its ADR and search by hand for what it affects. That search can miss ADRs resting on an unnamed premise. We frame this as premise recovery: given an ADR with identifiers masked, find its premises among earlier records. We collect 1,060 ADRs from 36 repositories and label 1,529 candidate pairs for 60 target ADRs. Two LLM annotators agree at \k{appa} = 0.859. An independent human annotator, blind to their labels, agrees with the stated/implicit tag on 30 of 32 premises (\k{appa} = 0.875). We compare 7 open-weight models with 2 baselines that use no LLM. Qwen3.8-27B reaches F1 0.756, against 0.466 for word overlap; 2 models added post hoc score similarly. On the 11 premises whose titles barely overlap the target, it finds 0.758, while word overlap finds none. Reasoning raises its recall of implicit premises but not its F1. Word overlap with as many predictions gains 69% as much. We find only 2 to 3 stale dependents, though we searched for unlinked ones in only 30 of 66 supersessions. Open models could thus suggest unwritten premises when a decision is reversed, with precision between 0.68 and 0.78. 

---
# What You Post Is What You Symptom: Estimating Psychometric Networks from Social Media Discourse 

**Authors**: Ştefana-Arina Tăbuşcă, Ana Sabina Uban, Liviu P. Dinu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21362)  

**Abstract**: Symptom networks in psychopathology are estimated almost exclusively from self-report questionnaires, restricting inference to whatever symptom set a given instrument covers. We introduce a framework for symptom-resolved network estimation directly from social media discourse and apply the framework to depression and eating disorders, with a third anorexia corpus as cross-dataset validation. To our knowledge, we are the first to estimate symptom-resolved psychometric networks from social media text, compare them to clinical co-endorsement networks, and replicate transdiagnostic linguistic markers across 3 independently recruited corpora spanning 2 disorders simultaneously. For depression, expression scores correlate with BDI-II totals at $\rho{=}0.434$, 20 of 21 items reach FDR significance, and split-half reliability is 0.935. The Network Comparison Test detects few edge differences between modalities (3-5\% of edges), while bridge symptoms diverge. For eating disorders, the item-level results are exploratory. Identified linguistic features replicate the same direction across all 3 corpora, and they cluster in three construct families. 

---
# Representational Change Under Self-Consuming Training: A Sparse Autoencoder Analysis 

**Authors**: Suchit Gupte, Zhongteng Cai, Xueru Zhang, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2611.21348)  

**Abstract**: Large language models are increasingly exposed to model-generated text, raising the possibility that future models will be trained recursively on the outputs of their predecessors. Prior work on such self-consuming training loops has largely focused on output quality and downstream performance, leaving their effects on internal representations poorly understood. We study how representations evolve across five generations of self-consumption and whether sparse autoencoders (SAEs) remain reliable under this shift. At each generation, we compare a lineage trained on synthetic data with a matched control trained on real data under the same training settings. We evaluate how well SAEs trained before self-consumption transfer to later generations, measuring reconstruction and sparsity across SAE widths, architectures, and model layers. We then fine-tune the original SAE on each generation and use sparse probing to test whether task-relevant information remains concentrated in a small number of latents or becomes more distributed across the representation. We find that zero-shot SAE performance degrades progressively under self-consumption, with growing reconstruction and sparsity gaps that vary substantially across layers. Fine-tuning largely restores reconstruction, but not feature-level usefulness: task-relevant information remains accessible in the full latent space while becoming less concentrated in individual features. These results show that self-consumption induces representational drift and that reconstruction fidelity alone is insufficient to establish that an adapted SAE remains equally useful for interpretation. 

---
# TRIAGE-CPG: A Cross-Model Study of Budgeted Static Malware Triage with Large Language Models 

**Authors**: Paul-Andrei Sava, Alexander Küchler, Daniel Kowatsch, Bernhard Grill, Jennifer Pullman, Yanick Fratantonio, Luca Invernizzi  

**Link**: [PDF](https://arxiv.org/pdf/2611.21331)  

**Abstract**: When triaging unknown binaries, security analysts must decide which samples to clear as benign, flag as malicious, or escalate for deeper analysis. Large Language Models (LLMs) could support these decisions by interpreting decompiled code, but a binary can contain hundreds or thousands of functions, making exhaustive LLM analysis impractical. We study how current LLMs behave when static triage is formulated as budgeted evidence acquisition: which functions they inspect and how they trade off benign clearance, malware detection, and escalation. To enable a controlled comparison, we implement TRIAGE-CPG, a fixed pipeline over a code property graph (CPG) of a decompiled binary. An LLM-free heuristic selects seed functions, an agent expands the resulting evidence dossier under a fixed budget, and a reporting stage produces a benign, suspicious, or malicious verdict together with a behavioral report. We evaluate eight closed-source and open-weight models on 1,000 Windows PE binaries, using a 600-sample test split and the same zero-shot protocol for every model. We assess the ternary verdicts under both a strict binary reading and triage balanced accuracy (tBA), which credits malware that is flagged or escalated and benign samples only when cleared. After analyzing about 40 of a mean 1,000 functions per binary, the highest-scoring model reaches a strict balanced accuracy of 0.805 and a tBA of 0.855, flagging 65% of malware, escalating another 12%, and clearing 94% of benign samples. The models implement different triage policies: high-specificity models produce small, malware-rich escalation queues, whereas recall-oriented models escalate or flag much more broadly. Both closed-source and open-weight models appear across this spectrum. Our results identify no universal deployment configuration: model choice itself determines the clear/flag/escalate trade-off under limited static evidence. 

---
# Can AI Mature a Model's Geometry? Training-Free Weight Baking in LLMs 

**Authors**: Amin Golnari, Saeid Sanei  

**Link**: [PDF](https://arxiv.org/pdf/2611.21310)  

**Abstract**: Can an AI make a trained language model more mature, in the sense of a better organized latent space, by looking only at the geometry of that space? An analyst model reads a fixed-size digest of topological descriptors of the layers of a frozen target and proposes a transformation, which closed-form ridge regression writes into the feed-forward weights without gradient training. Using lightweight models, 126 runs show that category purity rose on unseen words and sentences, benchmark accuracy stayed within sampling error, and the analyst matched a category-aware heuristic with gentler edits. In a five-model round robin approach, 16 pairs raised purity and four left the target unchanged. The results provide a baseline for future work on LLM-guided, training-free model editing without using raw weights. 

---
# Programmable Semantic Interventions: Learning Causal Carriers from Behavioral Programs 

**Authors**: Paolo Muratore, Mackenzie Weygandt Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2611.21307)  

**Abstract**: Causal abstraction explains a frozen network by aligning its activations with the variables of a simpler, interpretable model, and checking that interventions on the two agree. Finding such an alignment currently requires a counterfactual dataset: for every pair of prompts, the answer the network ought to give once the variable has been changed. In practice, those labels are produced by an executable algorithm for the behavior, written before studying the model supposed to implement it. However, the criterion these methods optimize only leverages the answer an intervention should produce. Here, we develop an approach that lets the investigator define that relationship directly: the answers the variable can produce, and which of them an intervention should yield. We call this interface a semantic program and introduce Programmable Semantic Interventions (PSI): a low-rank activation subspace learned so that interchanging it changes the model's answer as the program declares, with the expected answers read from the frozen model itself. On the MIB benchmark PSI performs on par with supervised alignment at matched data and training budget, without curated pairs, and it needs no ground-truth labels; while on natural science questions with gold labels withheld, it tracks the model's own answers better than a subspace trained on gold labels. The declaration also fixes a coordinate system with known semantics, enabling donor-free steering: jointly training PSI for both localization and control realizes 94.8% of held-out requests for answers the investigator chooses, without the need for donor prompts. 

---
# Fix, Don't Retake: Physics-Simulated Training for Correcting Patient Positioning Errors in Dental Panoramic Radiographs 

**Authors**: Sema Helali, Lina Abu Nada, Faleh Tamimi, Rafat Damseh  

**Link**: [PDF](https://arxiv.org/pdf/2611.21304)  

**Abstract**: Patient-positioning errors can distort panoramic radiographs and lead to repeat examinations. This study investigated artificial intelligence for correcting these errors in existing images. A physics-based simulator generated paired correctly positioned and mispositioned panoramics from cone-beam computed tomography volumes, with controlled individual and combined positioning errors. Five approaches were compared: U-Net, pix2pix, conditional diffusion, an error-conditioned model, and a mixture of error-specific specialists. Evaluation used image similarity and geometric metrics across three test sets. All approaches improved mean image similarity and reduced geometric errors, with U-Net and the two error-aware approaches outperforming pix2pix and conditional diffusion. 

---
# Agent-Native by Design: A Programming Language for Agentic Applications 

**Authors**: Jayanaka Dantanarayana, Savini Kashmira, Xiaoyu Ma, Malitha, Malitha Prabhashana, Lingjia Tang, Jason Mars  

**Link**: [PDF](https://arxiv.org/pdf/2611.21298)  

**Abstract**: Building agentic applications requires combining large language model (LLM) operations, workflow structure and application state, and model-guided routing. With SDKs and agent frameworks, developers connect these components through supporting code that prepares model inputs, propagates state, and dispatches model-selected steps. Workflow changes can require coordinated updates to this code, increasing development and testing effort. Splitting execution logic between application code and framework internals also limits compiler visibility into control flow and state dependencies, hindering automatic optimization. We introduce Agent-Native Programming, which unifies these three components with shared execution semantics. We realize this approach in Jac, a Python-superset language, through "by llm()" for typed model operations and tool use, Object-Spatial Programming for workflow structure and state, and "visit ... by llm()" for model-guided routing. The routing construct derives the model's choices from executable workflow steps and schedules the selected steps. This explicit connection between model decisions and program execution gives the compiler and runtime a common basis for dependency analysis and automatic optimization. Across five benchmarks and five baselines, Jac achieves a 1.8-8.7x reduction in AI-integration code relative to the OpenAI SDK, with competitive observed task quality and lower coding-agent development effort. Two case studies demonstrate speculative execution and service partitioning through configuration alone, using program analysis and runtime checks to preserve workflow behavior under stated conditions. 

---
# Benchmarking Large Language Models for Linguistic Annotation 

**Authors**: Ondřej Tichý, Anna Marklová, Barbora Bulantová, Magdalena Titlbachová, Jiří Milička  

**Link**: [PDF](https://arxiv.org/pdf/2611.21291)  

**Abstract**: Assessing large language models for linguistic annotation requires task-specific evaluation under documented conditions. We present dhAIBench, a Python benchmarking and annotation tool with a web interface for exploring and sharing results. It records prompts, configurations and predictions, enabling comparisons of annotation quality, confidence calibration, repeatability and estimated cost. We demonstrate the framework and the associated methodology through 293 runs of 48 models on nine linguistic annotation tasks spanning morphology, syntax, semantics, discourse-pragmatics, spelling normalization and learner-language correction. Models achieve high agreement with reference annotations on several tasks, including those on non-standardized, historical varieties as well as requiring contextual interpretation. The results support evaluating models against specific linguistic tasks, rather than relying on general rankings. To this end, we make the current and future benchmarks publicly available at this http URL. 

---
# FM-Recon: Exploring Foundation Model Features for Accelerated Cardiac Cine MRI Reconstruction 

**Authors**: Donghang Lyu, Marius Staring, Ruochen Gao, Jochen Keupp, Hildo J. Lamb, Mariya Doneva  

**Link**: [PDF](https://arxiv.org/pdf/2611.21290)  

**Abstract**: Foundation models have demonstrated strong feature extraction capabilities across various medical imaging tasks, yet their potential for MRI reconstruction remains largely unexplored, particularly within unrolled reconstruction networks and under artifact-corrupted inputs. In this work, we explore whether and how features from pretrained foundation models can benefit accelerated cardiac cine MRI reconstruction. To this end, we propose FM-Recon, a framework that refines and integrates extracted features from frozen image encoder of foundation models into an unrolled reconstruction network. Through extensive experiments on the CMRxRecon2023 dataset, we systematically evaluate foundation model features across different reconstruction settings. Our results show that foundation models can extract informative representations from artifact-corrupted inputs and improve reconstruction performance, particularly at moderate acceleration factors. These findings demonstrate the potential of foundation model features for accelerated MRI reconstruction and provide insights into their effective integration into unrolled reconstruction networks. 

---
# In-Context Estimation and Prediction for Parametric ODEs 

**Authors**: Abdullah Ahmed, Yun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.21268)  

**Abstract**: Transformers have demonstrated the ability to perform in-context learning, that is, to adapt to new tasks from a few examples without updating their parameters. Existing theory has largely focused on emulating gradient-based algorithms for convex statistical problems or establishing general function approximation guarantees without exploiting the underlying statistical structure. We study in-context parameter estimation and trajectory prediction for parametric ordinary differential equations (ODEs), where estimation is generally nonconvex. Under suitable identifiability and regularity conditions, we construct a transformer that estimates unknown system parameters from noisy trajectory observations through a two-stage approximate maximum likelihood procedure: a coarse search provides an initialization within a locally strongly convex region, followed by gradient descent with local linear contraction. For prediction, we compose this estimator with a neural approximation of the ODE solution map. We further show that, through parameter sharing, a single transformer can perform both parameter estimation and trajectory prediction simultaneously. For these constructions, we establish nonasymptotic guarantees and finite-pretraining oracle inequalities that separate errors due to finite pretraining data, finite context length, and finite-sized transformer approximation. Numerical experiments on linear and nonlinear ODEs further show that pretrained transformers can benefit from longer inference contexts without updating their weights. 

---
# Conformal Expert Pools: Training-Free Probabilistic Time-Series Forecasting Beyond Seasonality 

**Authors**: Valery Manokhin  

**Link**: [PDF](https://arxiv.org/pdf/2611.21267)  

**Abstract**: Conformal Seasonal Pools (CSP) showed that a forecaster with no training at all, built only from past values at the same point in the season and from past forecast errors, significantly outperformed DeepNPTS, Amazon's learned non-parametric forecaster. We propose Conformal Expert Pools (CEP), a training-free successor. CEP replaces the single seasonal-naive forecast by a pool of simple forecasters, builds its predictive distribution from their past forecast errors, and adapts both the pool and the errors to the series being forecast. No model is trained and no parameter is learned across series. On the CSP benchmark of 232 datasets and 24,502 forecast windows, CEP is better than CSP on 78% of datasets by CRPS, on 84% by MSIS and on 83% by Winkler score; on a typical dataset its CRPS is 14% lower and its MSIS 23% lower (both p < 1e-22, paired Wilcoxon test), and coverage of 95% intervals rises from 90.1% to 92.2%. CEP has a lower CRPS than DeepNPTS on 88% of datasets, than NPTS on 89% and than SeasonalNPTS on 83%. The improvement persists on dataset families outside the development set and on 36 additional fev-bench files. We also test the published claims that Laplace, an online ensemble forecaster, "beats (almost) everything" and "beats" CSP. On 228 datasets at multi-step horizons the claims do not hold: CSP outperforms Laplace on 60% of datasets by CRPS and on 64% by interval score, and Laplace ranks sixth of seven methods, with the widest intervals of all (69% wider than CEP's). Its point forecast is biased low and on 54% of datasets is less accurate than seasonal naive, and from a cold start it takes about 7,700 times longer than CSP. CEP is better than Laplace on 82% of datasets by CRPS and on 80% by MSIS, with intervals about 40% narrower, and runs 271 times faster. 

---
# A Generative AI-Driven Architecture for Whole-Home Intelligent Services: Edge Models, Memory Management, and Household Applications 

**Authors**: Mo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.21232)  

**Abstract**: Generative artificial intelligence is entering the home. Cloud-based assistants face governance limitations in data transmission, continuous availability, and the reliability of physical actions, which are especially salient in households where devices remain online, data accumulate over time, and commands trigger real-world actions. Using the home local server as a unifying organizing axis, this article develops a design-oriented review framework. Sources from arXiv, CNKI, ACL Anthology, and official documentation were searched through November 2026 and coded into five evidence categories: reviews, system prototypes, benchmarks, HCI and eldercare studies, and official documentation. System designs were mapped onto shared architectural modules, identifying nine recurrent responsibilities of the home local server. Generative models support perceptual interpretation, intent understanding, and candidate planning, while deterministic validation, knowledge-graph constraints, and authorization leases confine candidate actions to an inspectable space. Home control, emotional companionship, and intelligent guarding can be compared within a common framework: they share state and memory but differ in latency, false positive and negative rates, proactivity, and permissions. Existing benchmarks rely on virtual environments and cannot substitute for real-deployment evaluation. Local deployment is not inherently safer, and technical continuity alone does not establish the social or medical effects of companionship and guarding. The article proposes a five-dimensional evaluation framework covering architectural capability, system resources, privacy, safety and reliability, and human experience, together with six design principles and a future research agenda for local-first intelligent home systems. 

---
# Oriented Semi-DETR for Semi-Supervised Aerial Object Detection 

**Authors**: Yuqiu Li, Jian Xue, Fan Li, Ke Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21223)  

**Abstract**: Semi-supervised object detection has recently been extended to DETR, where oriented boxes introduce an ambiguity that does not arise for axis-aligned boxes: the map from parameters to rectangles is two-to-one, since $(c_x,c_y,w,h,\theta)$ and $(c_x,c_y,h,w,\theta+\pi/2)$ denote the same rotated rectangle. Losses and matching costs are computed from the parameters, so two boxes that coincide in space can be far apart in the metric that drives training. A correctly localized student box is then penalized by the regression loss, and the inflated matching cost leads the Gaussian mixture model (GMM) filter to discard it. We propose Oriented Semi-DETR, which removes this ambiguity wherever a pseudo box enters training. Swap-aligned Target Selection regresses towards the encoding closer to the student prediction. A Swap-invariant Reliability Cost minimizes over the two encodings inside the matching cost, so the filter decides on geometry alone. Swap-aware Pseudo-label Denoising starts denoising earlier, ramps its loss weight linearly, and weights each query by the teacher score and encoding margin. All three are parameter-free and require no additional forward pass. Across all label ratios of DOTA-v2.0, Oriented Semi-DETR outperforms DETR-based supervised and semi-supervised baselines, with the largest gains at the strict IoU threshold. 

---
# Verifiable Random Sampling of Private Data Without a Trusted Single Party 

**Authors**: Muhammed Karamuk, Mikołaj Glinka, Ruben Wolff  

**Link**: [PDF](https://arxiv.org/pdf/2611.21222)  

**Abstract**: Problem: When a party supports claims about a private dataset by disclosing a sample, auditors cannot tell whether it was drawn at random or "cherry-picked" to hide flaws in the rest of the data.
Solution: The data holder commits to the hashes of every item in its dataset and registers the commitment with independent servers. Selection then waits on a decentralized randomness beacon: a network that jointly emits a fresh random value every few seconds, outside the control of the data holder and the auditors. The value emitted two beacon rounds after the commitment picks the sample. Each server returns a signed receipt, so anyone can later check that the revealed sample is exactly the one that value prescribes. Selection from the committed set stays unbiased as long as the beacon's threshold holds and one required server is honest, and a pre-agreed name, checked with the servers, stops repeated or abandoned draws. We call this protocol NoCherry.
Applications: Any audit that spot-checks a private corpus: private AI benchmarks and held-out test sets, dataset quality claims, code and security audits, test-fairness review, vulnerability handling, and compliance sampling. Auditors need not trust the data holder to hand over a representative sample of the committed data, since the protocol proves it, and data holders need not expose their entire dataset. 

---
# TRACK: Tool-Augmented Reasoning over Time Series for Fault Diagnosis in Power Management System 

**Authors**: Erkan Bayram, Debanjan Ghosh, Li Zhu, Sambarta Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.21178)  

**Abstract**: Fault diagnosis in power management systems running multiple loads relies on experts manually inspecting current and voltage waveforms, which fails to scale across deployed units. We present TRACK, which pairs a time series encoder with a Small Language Model to produce structured, physics-grounded diagnostic reports, calling tools for exact, verifiable calculations. It operates on-device, keeping proprietary sensor data local and avoiding expensive API calls. We introduce a rubric-based framework that evaluates these reports by scoring structure, signal fidelity, physics reasoning, diagnosis, and actionability through per-criterion verdicts. On a 15-class multi-load simulated benchmark, TRACK achieves over 95% fault-category accuracy and generalizes to out-of-distribution faults, whereas the best encoder-probing baseline reaches only 58% accuracy. 

---
# Scientific Coding with Large Language Models: Evidence from Finite Element Analysis 

**Authors**: Erfan Hamdi, Joel Shor, Emma Lejeune  

**Link**: [PDF](https://arxiv.org/pdf/2611.21176)  

**Abstract**: Large language models (LLMs) offer a potential route to automating scientific programming, but their usefulness depends on consistently producing numerically correct code. We extend FEM-Bench to 17 finite element tasks involving nonlinear and history-dependent mechanics and structural dynamics, and evaluate 15 models on implementation and test generation without execution feedback. The best model succeeds in 87.1% of single-pass implementation trials, but rankings differ between tracks. The leading implementer achieves 49.9% joint test success, while the leading tester reaches 82.6%. For six models, we compare single-pass generation with two agentic harnesses that allow code execution and revision. Agentic execution solves some tasks that never succeeded in single-pass trials and improves consistency on others, but gains vary across models and harnesses, and regressions also occur. Claude Opus 5 with SimpleAgent reaches 96.5% Pass@3, the probability of at least one success in three attempts, but 77.1% Pass$^3$, the probability that all three succeed. No model-harness combination exceeds 78.8% Pass$^3$. To understand why agents submit incorrect code after passing their own checks, we analyze 249 failed agentic trajectories and identify six categories of self-verification failure. The most common are inadequate test coverage (73.1%) and reliance on generic consistency checks (70.3%), and in 28.5% of cases the tests share an error with the implementation. Motivated by these findings, an exploratory configuration with separate coder and tester agents achieves the highest observed scores, with 97.6% Pass@3 and 88.8% Pass$^3$. These results identify consistency across attempts and independent verification as the main obstacles to trusting LLM-generated scientific code. 

---
# Edit Recommendation: Can LLMs Anticipate Edit Requests from Users, and When? 

**Authors**: Daisuke Kikuta  

**Link**: [PDF](https://arxiv.org/pdf/2611.21155)  

**Abstract**: After a large language model (LLM) agent generates an artifact, users often send edit requests for it. This paper studies whether an LLM can anticipate these requests before they are sent. Such anticipation is useful in practice because it reduces the effort of user review and feedback. We formalize this task as edit recommendation and build EditRecBench, a benchmark for evaluating the performance of LLMs on this task, from real-world conversation logs in WildChat-4.8M. Through benchmarking and failure analysis, we identify a practical challenge that some cases are inherently difficult to predict in real-world settings. To address this, we introduce decision model-based gating to filter out cases where predictions are unlikely to be correct before they are presented to the user. Our trained model outperforms existing decision models and raises pass@5 among shown suggestions by 12--25 points at a gating threshold. Our code is available at this https URL. 

---
# Linglong Agent Service: A Runtime for Centralized Agent Reasoning and Distributed Execution 

**Authors**: Hua Zhang, Min Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.21129)  

**Abstract**: LLM agents can use remote tools and cloud sandboxes, but integrating reasoning, session state, and heterogeneous execution sites remains a runtime design problem. We present Linglong Agent Service (agentsvc), a self-hosted HTTP service and embeddable Python runtime that treats execution site as a configurable dimension alongside model, tools, and context. A parent service retains inference orchestration, tool descriptions, and conversation history; each inference request selects one child environment for file, shell, terminal, and MCP operations. Direct HTTP and child-initiated WebSocket tunnels carry the same application interfaces, including portable request context and centrally managed updates. The normal core path requires no third-party Python packages, with an optional Windows pseudo-terminal dependency. We further present online_debugger as the process-scale special case of execution-site separation: an instrumented program submits exception evidence, executes session-scoped inspection tools inside the incident process, and resumes the same diagnostic session. Its default prompt requires failing-then-passing reproduction evidence for source changes, while runtime patch activation and failed-call recovery remain unimplemented. We explain the design through constrained expected-utility reasoning, compare concrete mechanisms with Hermes Agent, Pi, Codex, and Claude Code, and report implementation and regression-test evidence. The current evidence establishes executable contracts, not performance advantages, repair accuracy, or production recovery. The source code is open source under the MIT license at this https URL. 

---
# Human-AI Co-Ideation Across Socio-Technical Configurations: An Interactional Analysis of Planar and Immersive Workspaces 

**Authors**: Orit Shaer, Mary Calabro, Zoe Mitchell, Angelora Cooper, Raffaella Sadun, Andrew L. Kun, Osnat Mokryn  

**Link**: [PDF](https://arxiv.org/pdf/2611.21125)  

**Abstract**: Remote collaborative platforms increasingly integrate generative AI to support group ideation. These platforms are sociotechnical configurations in which spatial mediation, interaction modality, AI output format, and the mechanisms that make contributions visible to a group co-vary as properties of a deployed system. This paper examines how such configurations shape participants' collaboration and their engagement with AI during creative work. We report a comparative study of two commercially available platforms that instantiate contrasting configurations: a planar shared canvas with text-based AI integration, and an immersive virtual reality workspace with concise AI output embedded in the shared space. Fourteen all-women triads completed a structured ideation task consisting of human-only idea generation, AI-supported idea expansion, and group convergence. We analyzed collaboration patterns, selection of AI-generated ideas, perceived idea quality, workload, presence, and their lexical characteristics. The immersive configuration yielded greater spatial presence and introduced a participation paradox, in which spatial freedom enabled exploration and fragmented group coherence; it was also associated with more balanced role distribution during convergence, at the cost of higher physical demand. In both configurations, AI multiplied the idea pool, and ideas groups authored with AI available were rated above participants' own. AI output format: about 21 words per idea on the planar canvas and 3 in the immersive workspace. Tracked how participants evaluated AI-generated ideas and how often they selected them unaltered as finalists. Neither configuration made AI contributions visible or attributable to the group: participants invoked AI individually and asynchronously in both. We derive design implications for AI output format, group-visible AI invocation, and spatial support for convergence 

---
# Searching for Autoregressive Neural Architectures from Fundamental Operations 

**Authors**: Thomas Woehrle, Linus Ericsson, Henry Gouk, Elliot J. Crowley  

**Link**: [PDF](https://arxiv.org/pdf/2611.21111)  

**Abstract**: Expressive Neural Architecture Search (NAS) search spaces are desirable as they do not restrict the space of potential architectures to what is already established. However, few search spaces are expressive enough to represent a broad range of architecture families, and those that are have focused primarily on image classification problems. The task of autoregressive sequence modelling has become ubiquitous in modern machine learning, but no existing search space supports the construction of and search over sequence models from fundamental operations, instead imposing strong inductive biases on the structure of potential architectures. This work describes how the expressive einspace search space can be extended to yield a search space capable of representing autoregressive models using fundamental operations. We perform a series of experiments showing that the novel einspaceAR search space can find capable autoregressive sequence models from scratch--i.e., without imposing strong inductive biases. We believe that our einspaceAR search space will be a useful part of the NAS toolbox and that the techniques described in this work can serve as a blueprint for the integration of autoregressive modelling in other expressive search spaces. 

---
# SWE-WhatIf: Simulating Real Users of Coding Agents from Recorded Session 

**Authors**: Tae Soo Kim, Tongshuang Wu, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.21095)  

**Abstract**: Recorded sessions between users and coding agents offer a way to evaluate how agents complete real-world tasks and interact with users. However, these sessions cannot simply be replayed with new agents, because recorded user replies may no longer apply when an agent takes a different trajectory. We introduce SWE-WhatIf, which replays recorded sessions with new agents by modeling the user's intents, the criteria for satisfying them, and the conditions under which they arise. Using this model, a user simulator generates replies that adapt to unseen agent turns while remaining grounded in the recording. Across 89 real-world sessions, SWE-WhatIf generated the replies most similar to those of real users and matched the strongest baseline when sessions were rewritten. To test adaptation, we replay sessions with agents instructed to avoid or repeat actions that the recorded user pushed back on. SWE-WhatIf produced the highest win rate for avoiding these actions (66.3%), significantly above chance. We further demonstrate its utility by ranking five LLMs as coding agents, where the highest-ranked model depended on how users specified and developed their intents, and a study with 16 users, where replaying their own sessions changed 14 participants' rankings and informed their future agent use. 

---
# Think2Listen: Evidence-Centric Learning For Agentic Audio Understanding With LALMS 

**Authors**: Jin Wang, Liang Lin, Kaiwen Luo, Weiliu Wang, Yuxuan Li, Zengxin Han, Gan Zhang, Jingtong Wu, Yufei Guo, Kun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.21083)  

**Abstract**: Recent advances in large audio-language models (LALMs) have broadened audio understanding across speech, sound, and music, with tool-based agents further supporting these tasks by invoking predefined audio tools. However, a persistent gap remains between perceiving acoustic content and using it as evidence for reasoning: recognizing what is audible does not ensure that a model can identify which cues resolve an uncertain decision. To bridge this gap, we propose Think2Listen, an evidence-centric learning framework for agentic audio understanding. Guided by evidence before commitment, Think2Listen couples iterative reasoning with a targeted LALM interaction to actively seek acoustic cues that distinguish competing hypotheses and revise candidate judgments based on the observations returned. It learns what to ask, how to use evidence, and when to stop, using observation-induced changes in candidate-state to inform training credit. Extensive evaluations on two benchmarks spanning seven audio categories demonstrate that Think2Listen achieves (I) enhanced multi-step audio reasoning, with gains in overall accuracy of 5.08%--17.67% over competing audio agents; and (II) improved audio understanding and reasoning across speech, sound, and music, boosting overall accuracy by 2.29%--13.10% over the corresponding baselines. The source code is available at this https URL. 

---
# SemiNovo: Learning Beyond Search-Identified Spectra for De Novo Peptide Sequencing 

**Authors**: Pan Liu, Dongxin Lyu, Yusen Tan, Yifan Li, Hai-tao Yu, Yunhua Zhong, Zikun Wang, Tianze Ling, Jun Xia  

**Link**: [PDF](https://arxiv.org/pdf/2611.21079)  

**Abstract**: Sensitive peptide identification is fundamental to protein characterization and novel protein discovery, for which de novo peptide sequencing (DNPS) from tandem mass spectra provides an important database-independent approach. Although deep learning has substantially advanced de novo sequencing, existing models are predominantly trained on high-confidence peptide-spectrum matches obtained through database search. This search-conditioned supervision excludes a large fraction of unidentified spectra, creating a distribution mismatch between the selectively labeled training data and the complete spectrum stream encountered at inference time, thereby limiting peptide identification performance. To address this challenge, we propose SemiNovo, a semi-supervised framework that learns jointly from identified and unlabeled spectra. SemiNovo first constructs DarkSpec, a large-scale quality-controlled corpus of unlabeled spectra collected from diverse public proteomics projects, thereby extending training beyond search-identified spectra and mitigating the resulting shift in DNPS supervision. It then combines stepwise mass-constrained autoregressive decoding with cumulative path confidence to derive reliable supervision from unlabeled spectra. Moreover, SemiNovo introduces head-wise chemical-evidence gating to improve spectral evidence aggregation under noisy and incomplete fragmentation. Experiments show that SemiNovo performs favorably against state-of-the-art methods on multiple benchmarks. Code and dataset are publicly available. 

---
# What Makes a Good Move? Learning to Act from Experience in Visual Worlds 

**Authors**: Cehao Yang, Ziyi He, Jing Yi, Jian Guo, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2611.21065)  

**Abstract**: Learning visual policies requires turning interaction experience into guidance for individual decisions. Trajectory outcomes obscure action quality, while unguided exploration can revisit familiar situations. We present \textsc{LightAct}, an experience-learning framework that connects skill-guided exploration with policy internalization. Interaction segments are summarized into experience entries and consolidated into a game-specific skill that guides root collection toward broader state coverage. At each root, candidate actions are compared through short rollouts under a shared frozen policy. Environment-reported score gains are averaged across continuations and normalized into relative action rewards, producing reusable supervision for reinforcement learning. Policy training uses a compact dataset of just 1,024 decision roots collected in a single source game and paired with their action rewards. Across 170 GameWorld tasks spanning 34 games, \textsc{LightAct} improves overall progress by 23.6, 29.2, and 13.0 percentage points at the 2B, 4B, and 9B scales, respectively, with our strongest model competitive with reported proprietary-model results. Ablations show broader root-state coverage with skill guidance and consistent gains across three policy optimizers. Case studies show how rollouts distinguish actions with identical immediate gains. 

---
# Multi-CONDOR: Learning Multistable Motion Primitives from Demonstration 

**Authors**: Gabriele Giuseppe Di Marzo, Mariano Ramírez Montero, Maximilian Stölzle, Rodrigo Pérez-Dattari, Cosimo Della Santina  

**Link**: [PDF](https://arxiv.org/pdf/2611.21059)  

**Abstract**: Dynamical-system-based motion policies provide a framework for learning reactive robot motions with stability and convergence properties. Many manipulation tasks, however, admit several distinct goals or a continuous set of acceptable outcomes, requiring motion primitives that preserve these alternatives. Accommodating both goal structures within a unified framework with continuous, autonomous dynamics remains an open challenge. We propose a CONDOR-based framework, multi-CONDOR, for learning motion primitives with either multiple discrete attractors or a continuous attracting goal contour. The approach combines imitation learning with losses that encourage agreement between the learned task dynamics and latent reference systems encoding the desired attraction structure. We provide a Lyapunov-based analysis of the convergence properties of the learned motion dynamics. We evaluate the framework on simulated benchmarks with discrete goals, qualitative examples with continuous goal contours, and robotic grasping experiments. In simulation, multi-CONDOR reduces trajectory RMSE by approximately 72% compared with separately learned single-goal motion primitives selected by a state machine. For tasks with two and three goals, respectively, it reduces the fraction of evaluated initial states ending outside the goal tolerance from StableNODE's 0.31% and 7.29% to 0% in both settings. Experiments with a Franka Emika robot arm demonstrate successful grasping of a container by either of its two handles using motions learned from demonstrations. Together, these results show multi-CONDOR combines motion reproduction with the flexibility to converge to alternative valid outcomes. 

---
# Towards wearable-scale, transition-aware sleep staging for clinical populations 

**Authors**: Suvadeep Maiti, Keya Sinha, Raju S. Bapi  

**Link**: [PDF](https://arxiv.org/pdf/2611.21040)  

**Abstract**: Automatic sleep staging models are predominantly developed on healthy sleepers and generalize poorly to clinical populations with fragmented sleep. Across three cohorts (patients with stroke, isolated REM sleep behaviour disorder, and other sleep disorders), we observe 2-3x more awakenings per hour, nearly twice as much N1, and less REM sleep than in healthy sleepers. We propose a label-efficient teacher-student framework for wearable-scale sleep staging. SleepGPT (136M parameters), adapted using labels from 25% of training patients, supervises a causal CNN-GRU (31k parameters; 122 KB) that stages EEG, EOG, or both in real time. We introduce Gcv-TAD, a new black-box transition-aware distillation loss. It emphasizes uncommon stage transitions while discounting those that vary across patients, using a prior estimated from labelled patients. In patient-wise 10-fold cross-validation, Gcv-TAD improves per-patient macro F1 over supervised training on the same labels by +3.2 to +7.6 points on average. Gains are largest on N1, and performance approaches full-label training. 

---
# Does the Voice Matter? Evaluating Frustration Detection on Deployed Clinical Voice-Agent Calls 

**Authors**: Annika Michael, Zachary Ellis, Yuxiang Zhou, Samuel Thio, Yajie Vera He, Spencer Hazel, Jared Joselowitz  

**Link**: [PDF](https://arxiv.org/pdf/2611.21001)  

**Abstract**: Autonomous voice agents now conduct routine clinical follow-up calls, so some automated triage is unavoidable. A patient becoming frustrated is one thing that triage should catch. These systems typically read only the transcript the speech-to-text stage produces, not the voice, and whether that is enough is largely untested on real clinical speech. We study per-turn frustration detection on real patient calls from a deployed clinical voice agent: 150 turns from six calls labelled by three annotators, with a length-matched acted corpus as a control, and 714 turns from 30 calls. Patient turns are overwhelmingly short: across the deployed calls, roughly 59% are one or two words, against 13% in the full acted corpus, and a corpus accuracy gap remains even after the acted set is length-matched. On one-word turns, transcript-only detection does not exclude chance. Across five audio-capable language models, the transcript flags 3 or 4 of the 39 frustrated one-word turns in the 30 calls, and a classifier given only the word count reaches 57.3%. Whether audio helps depends on the model and on the instruction. Asked to attend to delivery, two Gemini 2.5 models gain 3.0 pp over the transcript across 30 calls (-1.0 to 7.5 pp). Of the turns they confidently clear, 6.9% were in fact frustrated for audio against 13.0% for the transcript, a difference of 6.1 pp (2.2 to 10.4 pp). Across the 30 calls, given one prompt identical for both inputs, no model's audio-text interval excludes zero. Under the input-specific prompts, a newer Gemini model and two smaller open models do worse with audio than with the transcript (-3.5, -8.6 and -10.6 pp). A WavLM probe trained on the clinical calls matches the best prompted model overall (71.6%) but misses every frustrated one-word turn in the 24 added calls. When a patient's turn is short, the transcript alone is not enough to rule out frustration. 

---
# Explaining Cross-Label Decision Structure in Multi-Output Tree Ensembles 

**Authors**: Sylvio Barbon Junior, Hiago F. F. Oliveira, Rafael Figueira Gonçalves, Jessica Fernandes Lopes, Ricardo Cerri  

**Link**: [PDF](https://arxiv.org/pdf/2611.20996)  

**Abstract**: A multi-output tree ensemble predicts several binary labels through decision paths that it largely shares across those labels, yet the attribution methods commonly used to explain such models treat each output in isolation. We propose multi-sink Decision Predicate Graphs (ms-DPGs), which extend decision predicate graphs with one sink per label: each leaf contributes to the sink of every label its in-bag positive count for that label. On this representation we define Shared Predicate Flow (SPF), the share of a label's decision mass that passes through conditions active for both labels of a pair, and a contracted view of the graph in which label-specific lifts indicate whether a shared condition raises the probability of both labels or of one at the expense of the other. Controlled experiments show that SPF captures shared decision conditions rather than mere feature overlap. On nine multi-label benchmarks, native multi-output forests share substantially more decision structure than Binary Relevance on eight, including five of the six on which the two are at predictive parity, even at equal hyperparameters, whereas per-label TreeSHAP rankings register the change of architecture without showing whether more is shared. Because a native forest routes every label through the same splits, SPF describes the fitted model rather than the data, and on eight of the nine benchmarks it rises when the feature-label signal is removed; a high SPF is therefore not evidence of shared predictive information. ms-DPG offers a global, predicate-level account of how a multi-output ensemble reuses decision conditions across its labels. 

---
# Calpha: Toward Native Biomolecular Intelligence in a General-Purpose Language Model 

**Authors**: Yong Liu, Zhanpeng Shi, Peicong Lin, Zhenyi Zhong, Yizhou Dang, Wendi Wu, Haoling Li, Shuangjia Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.20986)  

**Abstract**: Biomolecular foundation models predict the structures of proteins and their complexes accurately, and large language models (LLMs) reason in natural language, but the two remain separate: the abilities of the former are not part of a general-purpose LLM, and the reasoning of the latter does not enter a structure prediction. Calpha aims at one general model that folds proteins and complexes, understands protein sequence and structure, and lets its language reasoning influence the structures it predicts. This report covers the first step, in which an LLM outputs single-chain protein structures. Qwen3.5-9B reads a sequence enriched with ESM2 residue features. It can respond in text or emit a token that invokes a pretrained SimpleFold structure head, which receives information about the protein only through Qwen's residue states. Under a fixed folding request, the head achieves mean TM-scores of 0.808 on the FoldBench-334 development set, 0.635 on CASP16 domains and 0.900 on a CAMEO-style PDB set. We compare these predictions with a baseline in which ESM2-3B encodes the sequence and a fine-tuned SimpleFold-3B head predicts the coordinates. The two systems also differ in trainable weights, training duration and crop length. For chains of 100--299 residues, all three paired confidence intervals include zero. At 300--600 residues, the mean differences are -0.044, -0.160 and -0.025, respectively. In a separate test, four English request types triggered folding on every evaluated example. Protein training reduced accuracy on a 350-item MMLU-Pro subset from 77.4% to 70.3%. An LLM can therefore output single-chain structures from its own states, with lower long-chain scores in the current comparisons and reduced accuracy on the evaluated MMLU-Pro subset. Complex prediction, structure understanding and the effect of language reasoning on the predicted coordinates remain future work. 

---
# Breaking feature collapse in self-supervised time series encoders 

**Authors**: Özgün Turgut, Daniel Rueckert  

**Link**: [PDF](https://arxiv.org/pdf/2611.20981)  

**Abstract**: General-purpose time series encoders promise a single backbone that generalises across tasks and domains; however, they currently do not scale. We trace this down to a critical point of failure: under the predominant masked data modelling supervision, the encoder's output tokens partially collapse. This is the natural consequence of supervising encoders only in the input space, which leaves no constraint against redundant token dimensions in the feature space. We introduce OTISv2, which suppresses the collapse by complementing the reconstruction objective in input space with two objectives in the feature space: a self-distillation loss against an exponential-moving-average teacher, and a KoLeo regulariser. Under matched training, the resulting encoder reduces the cosine similarity and lifts the effective rank of its output tokens relative to its predecessor, and it scales monotonically from 7.1 M to 116 M parameters where its predecessor declines. It matches or surpasses existing general-purpose time series encoders across 143 uni- and multi-variate datasets. To foster further research, we release our code and pre-trained weights. 

---
# Living Context Graphs: Continual Semantic Learning for Enterprise Data Reasoning 

**Authors**: Varun Agrawal, Ayush Srivastava, Eshan Gujar, Kaushal Kumar, Ayush Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.20976)  

**Abstract**: Enterprise AI can now perform analytics, reporting, and data-driven workflows, yet the data foundations it relies on remain hard to operationalize: business meaning, relationships, metric definitions, and reporting conventions are absent from database schemas and scattered across undocumented systems and tribal knowledge. We introduce the Living Context Graph (LCG), a continually learned context layer that makes existing enterprise data operational for AI without requiring a copy or transformation of the underlying warehouse. LCG bootstraps base semantics and relationships from the data itself, builds a concept graph that indexes business context as queries are served, writes durable semantic learnings from verified outcomes, and withdraws or re-earns context when later outcomes contradict prior routes. We evaluate LCG through text-to-SQL, a core data-reasoning task, on a transfer-aware split of three BIRD databases comprising 499 questions: 353 form the learning history and 146 are held out and never used to update the context layer. Bootstrapping alone improves overall accuracy by +4.6 percentage points. Learning only from the history raises accuracy on the untouched held-out transfer set by +22.6 points (95% CI [+14.4, +30.8]), exceeding five-shot retrieval from the same history by +12.3 points. Incremental learning reaches the one-shot endpoint within observed evaluation noise on every database. Ablating the graph changes held-out accuracy by at most one question per database but increases per-query cost 2.3-3.3x and median latency 1.8-3.4x: continually learned semantics drive accuracy, while the graph makes that context efficient to serve. 

---
# vividRGB: Various visual cue detection in RGB photographs with forced focused attention 

**Authors**: Masahiro Ryo  

**Link**: [PDF](https://arxiv.org/pdf/2611.20942)  

**Abstract**: Natural photographs contain dense, overlapping structures that can be weakly expressed in feature maps from whole-image vision encoders. We introduce forced focused attention (FFA) and its Python implementation, vividRGB. FFA is an encoder-agnostic inference-time procedure that applies a frozen encoder to enlarged, overlapping local crops, averages inverse-aligned reflections, and reconstructs a feature field in source coordinates. We evaluate 50 high-resolution photographs from grassland vegetation, fungal networks, airborne forests, coral reefs, and phytoplankton microscopy, using DINOv3 ViT-B/16. Across the five domains, FFA at 5% or 10% focus produces higher mean RGB-edge boundary F1 than direct DINOv3 encoding and automatic SAM3 masks. Domain means range from 0.244 to 0.596 for FFA, versus 0.028-0.076 for direct DINOv3 and 0.020-0.090 for SAM3. Qualitative feature maps also show finer appearance structure. These results concern RGB-edge preservation rather than verified species, traits, or biological segmentation. Narrower focus generally improves boundary preservation, while extraction time and GPU energy use increase sharply and gains diminish at finer settings. FFA may be useful for relatively small, high-resolution ecological image collections where dense visual cues matter. The alpha release vividRGB 0.1.0 and versioned research resources support reproducible cue inspection. 

---
# ReAccess: Look Once, Stay Focused with Efficient Test Time Memory 

**Authors**: Tianhao Chen, Yilong Fan, Yuheng Wu, Xiaobin Hu, Dongman Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.20936)  

**Abstract**: Despite recent advances, vision-language models (VLMs) still struggle with fine-grained visual understanding, often failing to effectively utilize critical evidence already encoded in their visual representations. Existing approaches address this limitation through repeated visual processing or explicit reasoning, introducing substantial computational overhead. In this paper, we propose ReAccess, a lightweight framework that preserves and reuses question-relevant visual evidence through compact associative memory. Constructed during a single visual prefill, the memory supports progressive evidence retrieval that steers both attention queries and hidden representations, enabling effective visual grounding without repeated image encoding or explicit reasoning traces. Extensive experiments demonstrate consistent improvements in both general visual understanding and fine-grained perception, while achieving substantially faster inference than existing reasoning-intensive approaches, demonstrating a promising path to efficient multimodal reasoning. 

---
# Experience Rot: Experienced Agents Fail Where Novices Succeed 

**Authors**: Zhen Xu, Hang Yin, Zitao Zhang, Kevin Zhao, Xuefeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20922)  

**Abstract**: Agents are increasingly built to learn from experience: they keep conversation histories, memory files, retrieved notes and skills, and reuse what worked before. We show that this experience turns against them when their tools change silently. We test a range of models with tools defined as in the Model Context Protocol (MCP). After a silent change, experienced agents fail on most tasks, while agents new to the task mostly succeed. We call this failure Experience Rot. Experienced agents copy their earlier call instead of checking the current interface, even when the tool definition shows the change. Then they report success. When the documentation is incomplete, experience is what makes agents succeed. After a change, the same experience leaves them worse off than no experience at all. The failure appears in every form of memory we test and grows with the amount of experience. Simple remedies fall short: announcing the update barely helps, and pointing the agent to the documentation fails whenever the documentation does not spell out the answer. 

---
# Auditing a Moving Target: Evidence Anchoring and the Expiry of Fairness Audits of Provider-Served Language Models 

**Authors**: Mika Facchini  

**Link**: [PDF](https://arxiv.org/pdf/2611.20901)  

**Abstract**: Organizations increasingly use large language models (LLMs) for decisions about people, such as hiring, investment and employee evaluation, and rely on them over long periods of time. Fairness audits of such systems, however, are conducted only once. We examine how the biases of LLMs change over time. In a venture screening setting, models from five providers decided whether and how much to invest in startups whose founders differed only in gender, origin and university. We repeated this audit three times over 13 months (248,400 API calls). Study 1 shows that the same model, called under the same identifier one year later, displayed a significantly different bias pattern. Study 2 shows that the successors of retired models carried the same bias as their predecessors, but in a significantly different size and form. Study 3 shows that the bias does not depend on the industry of the startup. A document analysis further shows that providers change and retire models at their discretion, with notice periods of zero to six months. Drawing on these findings, we develop a theory of evidence anchoring. We theorize that bias resides in the model class and persists, whereas audit evidence is anchored to a single model and expires when the provider decides. We derive policy advice for organizations and regulators, namely audits of model classes instead of single models, a version record for every model response, and re-audits triggered by model changes. 

---
# Re-Reading Is Not Free: Token Traces and Cache-Aware Energy Estimation for LLM Agents in Software Engineering 

**Authors**: Tina Vartziotis, Ketevan Peranidze, Rodopi Kosteli, Ezgi Sarikayak, George Dasoulas, Michael Keckeisen, Sotirios Kotsopoulos, Francesca Dominici  

**Link**: [PDF](https://arxiv.org/pdf/2611.20873)  

**Abstract**: As large language models (LLMs) evolve from conversational assistants into autonomous agents embedded in software development workflows, estimating their environmental footprint requires moving beyond conventional token-based accounting. Existing energy estimators focus on isolated request-response interactions, overlooking the repeated processing of accumulated context, cache reuse, and work delegated to subagents. We analyze nearly one billion tokens collected across 42 sessions and 752 task episodes over ten weeks of real-world AI-assisted software development. Cached-context reads make up 94.8% of all API-reported tokens, while uncached input accounts for only 0.017%; for every generated token, approximately 260 cached tokens are read. Context grows 5.3-fold over a session compared with 1.4-fold within individual tasks. To quantify the energy implications, we extend an existing analytical inference-energy model to account for cached-context processing, yielding a cache-aware hybrid estimator. It estimates approximately 250 kWh for the workload, compared with approximately 10 kWh from the output-based estimator EcoLogits, a 25-fold difference largely driven by cached-context reads. Neither estimator has been validated on long cached contexts, but no modeling assumption we varied closes the gap. Starting each task from a fresh context would lower our estimate by about 60%. We also find that parent-session logs report only a subagent's final internal call, undercounting delegated tokens by a factor of ten; for this workload the correction changes the energy estimate by about 5%, but token-volume studies are fully exposed to it. Our findings expose a substantial gap between how current energy estimators and agentic workloads treat context, highlighting the need for workflow-level energy accounting that captures context accumulation, cache reuse, and delegated execution. 

---
# Scenario-Based Risk Certificates for Multi-Property Vulnerability Discovery in Parametric MDPs 

**Authors**: Zhiming Chi, Andrea Turrini, Yong Li, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.20861)  

**Abstract**: Safety-critical systems such as network protocols and distributed algorithms depend on environmental parameters whose exact values are uncertain. Parametric Markov decision processes (pMDPs) model such systems by making transition probabilities symbolic functions of the parameters. For a single quantitative requirement, parametric verification asks which parameter valuations satisfy the requirement. In practice, however, engineers often monitor several safety and performance requirements at once, and one valuation may make several requirements vulnerable simultaneously. In this paper we study multi-property vulnerability discovery under requirement-wise semantics: given a pMDP and M quantitative requirements, we construct candidate regions of the parameter space that may contain violations, and study budgets for random testing under a specified parameter distribution. To solve this, we adopt a two-layered approach in which the scenario-approximation layer provides PAC guarantees on identifying conservative vulnerability regions for each requirement while the testing layer treats random parameter exploration as coverage of overlapping regions and yields a stopping certificate computed from the marginal region masses alone; a composition theorem then bounds the expected non-discovery score by the scenario approximation error plus the testing risk from the two layers. For confirmed discoveries, candidate hits are checked by model checking and budgets use independently certified discovery probabilities. Experiment evaluate these certificates using empirical surrogates and independent confidence bounds. 

---
# MolReps: Taking SELFIES, Making SMILES, Teaching LLMs 

**Authors**: Ken Deng, Garrett Morris  

**Link**: [PDF](https://arxiv.org/pdf/2611.20848)  

**Abstract**: The same molecule can be expressed in multiple, syntactically distinct representations, yet it remains unclear whether large language models (LLMs) can recognize and relate these equivalent encodings. We introduce MolReps, a pipeline that generates 9,600 question-answer pairs across ten representations for conversion, normalization, and 2D structure recognition. Post-training on MolReps improves text-only and multimodal models, MolQwen3-4B-Instruct (MQ3-4B-I) and MolQwen3-VL-4B-Instruct (MQ3-VL-4B-I), from $4.01\%$ to $75.13\%$ and from $7.11\%$ to $73.14\%$ (InChIKey match) on the molecule-overlap test set, respectively. On the molecule-disjoint test set, post-training increased validity from $20.35\%$ to $58.12\%$ for MQ3-4B-I and from $17.19\%$ to $59.34\%$ for MQ3-VL-4B-I. However, InChIKey match remained low, decreasing from $3.86\%$ to $3.09\%$ and from $6.44\%$ to $2.29\%$, respectively. Furthermore, the resulting models achieve a relative $27.84\%$ improvement in average over their base model on MolRepBench, a comprehensive benchmark for evaluating molecular property prediction. These results show the effectiveness of targeted post-training for enhancing molecular representation understanding and its potential to improve downstream molecular tasks. 

---
# EDITORS Know Your Style! Editing LoRA Subspaces for Stylistic Attribution and Imitation 

**Authors**: Zixuan Wang, Gregory Kang Ruey Lau, Xiao Tian, Jue Fan, Caroline Chaux, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2611.20846)  

**Abstract**: Disentangling writing style from semantic content is a fundamental challenge in literary text modeling. Style-content entanglement causes models to rely on semantic content for authorship attribution (AA) tasks and memorize author-specific content for style imitation (SI) tasks. We propose EDITORS (EditingLoRA Subspaces), a diagnose-then-deflate framework that adapts pretrained large language models and prioritizes stylistic features, making it effective for both AA and SI tasks. EDITORS first trains a diagnostic LoRA adapter using style-neutral statements derived from the training corpus to identify a content-oriented subspace. It then trains the final style adapter under activation-space deflation that projects input activations away from the identified content-oriented directions. We empirically demonstrate that EDITORS is effective. On three AA benchmarks spanning literature, social media, and news, EDITORS achieves state-of-the-art accuracy, outperforming the strongest existing baseline by up to 29%. On SI, EDITORS generates stylistically faithful and diverse passages, outperforming baselines in achieving style similarity while adhering to content specifications. 

---
# Evidence Diet: When Graph Evidence Helps LLMs Rank Competing Candidates 

**Authors**: Yuchong Hu, Zhao Zhang, Ziyao Zhao, Haoyu Wang, Jing Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.20837)  

**Abstract**: When several candidates have graph support, does it matter which candidate receives which supporter and path counts? We study how LLM rankers use collaborative evidence on Amazon Books and MovieTV. Swapping user-level evidence degrades ranking, while matched item substitutions have little effect. On fixed clean-trained checkpoints, unrelated catalog text lowers Hit@1 from 44.77% to 38.70%, with 99.9% of rankings still valid. Across all three paired seeds, the NDCG@5 difference between count and path prompts was larger after count training than after path training. Degree matching preserves target paths while giving more rivals support. Across the full degree-matched development pools, adding candidate-level graph records to history improves NDCG@5 by 2.28 points on Books and 2.54 on MovieTV. When the target and a rival both have support, path- and count-trained rankers benefit from retaining counts. On these development queries, adding counts numerically to history scores outperforms presenting them as text. For count-trained rankers, exchanging intact count pairs among supported candidates preserves support and token length, yet lowers NDCG@5 by 4.06 points on 141 Books queries and 3.26 on 104 MovieTV queries. A frozen evaluation on 274 previously unused Books users with supported targets and rivals confirms the attribution effect: correct assignment improves pairwise accuracy by 6.07 percentage points (95% CI [4.18, 8.00]). 

---
# ProcReverser: Multi-Agent Black-Box Microarchitectural Reverse Engineering for Processor Vulnerability Analysis 

**Authors**: Guanqi Lyu, Decheng Qiu, Guangyang Zhang, Fei Zhao, Zhaojun Lu, Jiaji He, Yongqiang Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20827)  

**Abstract**: When processor implementation details are unavailable, black-box microarchitectural reverse engineering helps security researchers identify hardware conditions related to security vulnerabilities. However, this process requires substantial manual effort to design experiments, develop test programs, and analyze results for each processor. We present ProcReverser, a multi-agent system that automates this process. ProcReverser uses execution feedback to improve experimental plans and test programs and identifies microarchitectural properties from measurements. We develop a comprehensive suite of property identification tasks focused on hardware mechanisms relevant to Spectre-class attacks, covering speculative execution, branch prediction, memory dependence, and cache behavior. We evaluate ProcReverser on the XuanTie C910, the gem5 RISC-V O3 CPU model, and the SiFive P550. ProcReverser achieves Pass@1 and Pass@5 scores of 81.3% and 93.8%, respectively, on C910, and 83.0% and 100.0% on gem5 O3. On the SiFive P550, ProcReverser further characterizes microarchitectural properties covered by the proposed task suite. These findings highlight the potential of automated microarchitectural reverse engineering to support processor vulnerability analysis and defense evaluation. 

---
# Surprise-Gated Memory for Live Speaker Diarization 

**Authors**: Louis Mouchon, Alexis Lainault  

**Link**: [PDF](https://arxiv.org/pdf/2611.20819)  

**Abstract**: We transpose the Surprise-Gated Memory recipe, a frozen encoder plus a calibrated novelty signal that gates a one-shot episodic store consolidated by periodic replay, from continual visual learning to live speaker diarization. The moment-to-moment question a streaming diarizer must answer, whether an arriving voice belongs to an already-heard speaker or a new one, is a known-versus-novel decision, exactly the decision the surprise gate is built for. On a frozen fine-tuned DiariZen backbone, our online system SGM-Diar recovers 4.8 of the 9.2 diarization-error points separating a naive online baseline from the offline oracle on AMI-SDM (52%) and 1.9 of 2.3 points on VoxConverse (82%), in the end-of-meeting revisable mode, with calibration frozen on development data and transferred cross-domain without adjustment. These are single-seed runs on three AMI meetings and five VoxConverse sessions, so the ordering of conditions rather than the absolute rates is the result. Beyond error rate, we measure the gate's behaviour directly and find a coherent memory-like profile: one-shot speaker recall at 0.857, a novelty response that habituates as speakers become familiar, re-identification that is invariant to the length of intervening silence, and a false-novelty rate of 0.001. Two ablations reproduce, now on audio, the central prediction of the visual SGM work: a sliding replay window is worse than no replay at all. The memory is strictly intra-meeting and is wiped at the end, and no gradient touches the run-time system. We report SGM-Diar as a proof-of-concept and state its limitations explicitly. 

---
# The Cybersecurity Dataset Observatory: Which Data for AI in Cybersecurity? 

**Authors**: Francesco Cerasuolo, Gabriele Mangiacapre, Antonio Pescapè  

**Link**: [PDF](https://arxiv.org/pdf/2611.20812)  

**Abstract**: AI is increasingly used for cybersecurity tasks such as intrusion and malware detection, in enterprise, industrial, vehicular and mobile networks. Its effectiveness depends on data: models need representative network traffic, both to be trained and to be evaluated realistically. Public datasets of cybersecurity network traffic, however, are scattered across laboratory pages, repositories and aggregators, and are continuously released, corrected and withdrawn: 92 of the 234 datasets we catalog were released in 2023 or later. At this pace, surveys that list them are outdated soon after publication, since they describe the field only as it was when they were written. We present the Cybersecurity dataset observatory, a public and continuously updated catalog of cybersecurity datasets based on network traffic. New datasets are added as they are found or submitted, every record is periodically re-checked against its original source, to which it links rather than hosting the data, and every change is recorded in a public change log. This paper documents the first release: we define a traffic-centered scope and a seven-axis taxonomy, and review 234 datasets (204 original captures and 30 derivatives) across 14 domains. The review shows a field that grows quickly but unevenly, where 91 datasets distribute only processed records, a few laboratories produce a large share of the data, and UAV, wireless and V2X networks have little public evidence, while 5G has few real captures. We then relate the catalog to common uses in AI research, outline a procedure for selecting datasets, and discuss benchmark pitfalls and open gaps. 

---
# Auditing World-Model Explanations in Retail Forecasting: Faithfulness, Accuracy and Price Response 

**Authors**: Kathik Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2611.20811)  

**Abstract**: Forecasting models that read the price and promotion plan explain each forecast as a baseline, what would sell with nothing planned, plus the lift the plan adds. We test such explanations with eight checks, six with a pass rule fixed before any result, and find that passing them does not make an explanation right. In generated store data with known effects, a world model passes the checks of construction, stability, price direction and agreement with sales and with LightGBM, yet its lifts are 1.33 to 1.40 times the truth when merchants plan into high-demand weeks, and 1.23 to 1.42 times in eight variants of that world. It still picks the right weeks to promote, but stock ordered to its lift is off by 0.44 to 0.48 of the true lift. Its stated response to a lasting regular-price change is 4.7 to 6.5 times the true -0.5, and only a measured cause finds this. On Dominick's, 38 episodes in which a store's regular prices sat apart from its zone peers give a noisy benchmark: a pooled 8-week response of -0.33 (95% interval -0.68 to 0.01). For the final tuned world models of our companion paper on nine data sets, baseline and lift add up under each model's own plan, but a deeper price cut lowers the forecast in 29% to 83% of item-week steps on Dominick's (23% to 80% within prices seen in training), and the world model and LightGBM both state about twice the measured response. Across 703 configurations from the settings searches, checks passed were not related to forecast accuracy (Spearman 0.02, 95% interval -0.02 to 0.05); with no known answer for those versions, this concerns accuracy, while whether an explanation is right is measured only in the generated data. 

---
# Nemotron 3.5 Bolt E4B 4GB: An Open and Efficient Small Language Model for Fast Conversational Agents 

**Authors**: NVIDIA, Adi Margolin, Aditya Malte, Aishwarya Chhabra, Alex Hodson, Ameya Sunil Mahabaleshwarkar, Ambrish Dantrey, Amy Shen, Andrii Skliar, Annamalai Chockalingam, Anusha Pant, Ashwath Aithal, Ashwin Kumar, Besmira Nushi, Bhavesh Pawar, Bryan Catanzaro, Bryan Yu, Chantal Hwang, Chenhan Yu, Chris Alexiuk, Conner Currie, Cynthia Gao, David Mosallanezhad, Dina Yared, Ehsan Hosseini Asl, Eileen Long, Erick Galinkin, Felix Friedmann, Giedrius Burachas, Gorkem Batmaz, Grzegorz Chlebus, Guillermo Siman, Haifeng Qian, Hao Guo, Hayley Ross, Hrishith Thadicherla, Hui Wang, Huiying Li, Ike Nnoli, Irina Filatova, Iris Liu, Ji-Eun Lee, Jian Leong, Jian Zhang, Jiaqi Zeng, Joey Conway, Jonathan Cohen, Jose Vergara, Julien Veron Vialard, Kari Briski, Katherine Luna, Kedar Pandurang Potdar, Kevin Ning, Khalil Ben Khaled, Krzysztof Pawelec, Laura Dykes, Leili Tavabi, Lily Lee, Linda Liu, Liwei Ding, Maka Dong, Manoj Kilaru, Maria Bagdasarova, Matthew Lee, Meredith Price, Micah Schaffer, Michael Boone, Michael Fukuyama, Michael Hasper, Nancy Agarwal, Negar Habibi, Oleksii Kuchaiev, Oluwatobi Olabiyi, Pablo Ribalta, Payal Motwani, Pinky Xu, Prasoon Varshney, Pritam Biswas, Puhui Meng, Rameshwar Shivbhakta, Ran Zilberstein, Reka Kovacs, Richard Mazzarese, Rishabh Garg, Rita Fernandes Neves, Rochelle Woods, Ruby Ye, Ruvaid Virk, Ryan Hill, Saeid Motiian, Seph Mard, Shai Ben Nun, Shaona Ghosh, Sharad Yadav, Sudeep Sabnis, Suhas Hariharapura Sheshadri, Terry Kong, Thanh Do, Thinh Le, Udi Karpas  

**Link**: [PDF](https://arxiv.org/pdf/2611.20771)  

**Abstract**: We introduce Nemotron 3.5 Bolt E4B 4GB, an open and efficient small language model further post-trained from Gemma-4-E4B-it and optimized for speed and intelligence per GB of memory used for fast conversational agents. The model carries forward Gemma's strong conversational capabilities and further improves multi-turn tool use, instruction following, and input grounding for fast, local voice/device assistants, interactive personas, and embedded/Edge interfaces. The model combines natural conversation with multi-turn agentic capabilities to power voice/device assistants, interactive characters and robotic/IoT interfaces running on GeForce RTX, Jetson, and RTX/DGX Spark. Experiments on 21 benchmarks show that Nemotron 3.5 Bolt E4B 4GB achieves the highest average accuracy while generating the fewest output tokens in both reasoning and non-reasoning modes, compared with Gemma-4-E4B-it and Qwen3.5-4B. The model is released with a DFlash drafter. With DFlash, Bolt E4B achieves $2.74\times$ the per-user decode speed of inference without speculative decoding at concurrency~1, and $1.21\times$ the generation throughput of Gemma-4-E4B-it with MTP at batch size 1. 

---
# MemEmo: Evaluating Emotion in Memory Systems 

**Authors**: Peng Liu, Zhen Tao, Jihao Zhao, Ding Chen, Yansong Zhang, Cuiping Li, Zhiyu Li, Hong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.20733)  

**Abstract**: Memory systems address the challenge of context loss in Large Language Models during prolonged interactions. However, as an essential tool for warm companionship and personalized services, the effectiveness of memory systems in processing affective information remains underexamined. To bridge this gap, we propose an emotion-enhanced memory evaluation benchmark to assess the performance of mainstream and state-of-the-art memory systems in integrating and harnessing emotional information. First, we design a benchmark called \textbf{MemEmo} to assess the affective reasoning and tracking capabilities of memory systems across four dimensions: emotional information extraction, emotional memory updating, emotional memory question-answering, and emotional conflict detection. Second, we construct the \textbf{H}uman-\textbf{L}ike \textbf{M}emory \textbf{E}motion (\textbf{HLME}) dataset to provide an empirical foundation for scrutinizing memory systems. Finally, We evaluate five representative memory systems on the HLME dataset, reporting 11 performance metrics across four dimensions. Experimental results reveal that aggregating scores for all tested memory systems is up to 45\%, far below 100\% scale, exposing a profound disparity between memory system and affective intelligence. 

---
# A First Look at System One Decision Models for Malware Detection 

**Authors**: Pier Mattia Basciano, Giovanni Farina, Salvatore Monteleone  

**Link**: [PDF](https://arxiv.org/pdf/2611.20699)  

**Abstract**: Malware detection needs scores calibrated at a fixed false positive rate (FPR) and decisions an analyst can read: feature-based detectors give the first, large language models (LLMs) on decompiled code the second, through generated text of uncalibrated confidence. System One decision models answer a typed question about a text with a probability over its declared answers in one forward pass. We present the first evaluation of these models for Windows Portable Executable malware detection, analyze how they decide, and propose S1Mal, an ensemble on the typed answers and the embedding of CLM-8B. In an isolated virtual machine, Ghidra and capa decompose each binary, and the host renders the records into a canonical textual state, a digest of headers, imports, MITRE ATT&CK capabilities and strings, optionally with decompiled functions, without class shortcuts. Three readouts score each model on the state: A0, the probability of one verdict question; A1, 32 atomic behavioral questions weighed by gradient-boosted trees; A2, heads on the model's state embedding. S1Mal feeds gradient-boosted trees with the atomic answers and the A2 score of CLM-8B. We evaluate CLM-8B, ten open decision models and two LLMs read through logits on BODMAS temporal splits, SOREL-20M out-of-distribution malware and 14,534 matched benign files. On BODMAS, A0 wastes the static evidence: its area under the receiver operating characteristic curve (AUROC) spans 0.4025-0.8590 across 13 models. A1 recovers it, raising every model's AUROC by +54.61% on average; A2 lifts CLM-8B by +0.2749 to 0.9943, above a combiner on digest facts (0.9901), and reaches 0.9853-0.9948 on nine open models. S1Mal reaches 0.9946 and detects 93.47% of the malware at a realized FPR of 1.45%, and 59.15% at 0.1% FPR on the test ROC curve, against 46.59% with A2. S1Mal gives analysts calibrated decisions, each with the atomic answers behind it. 

---
# IndustrialKnowEval: A Benchmark for Evaluating Multi-Level Industrial Knowledge in LLMs 

**Authors**: Haiteng Wang, Haoze Li, Jingwei Guo, Jingbo Zhou, Si Liu, Lei Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.20695)  

**Abstract**: Large language models (LLMs) are increasingly explored for industrial tasks, yet few benchmarks evaluate their mastery of industrial knowledge across different types of cognitive demand. We introduce IndustrialKnowEval (IndKnow), a benchmark for industrial knowledge and engineering question answering. It contains 43,282 samples covering 21 task types across seven industrial sectors, including aerospace, electric power, and metallurgy, organized into four levels of cognitive demand: memory, comprehension, reasoning, and engineering application. Enterprise-based domain specialists and experienced researchers participate in all stages of dataset construction and quality control. We conduct a series of experiments on 18 representative LLMs, with domain experts deeply involved in the evaluation. Building upon IndKnow, we propose the Industrial Cognitive Capability Gap (ICCG) metric to characterize the performance differential between foundational industrial knowledge and higher-order reasoning and application, and we analyze it alongside absolute performance to provide a more complete view of model capabilities. We anticipate that IndustrialKnowEval will support the benchmarking of LLMs in industrial scenarios and promote the development of robust industrial intelligence. Code is available at: this https URL 

---
# SpeCal: Spectral Calibration for Unsupervised RLVR 

**Authors**: Kai Tang, Xing Zheng, Haobo Wang, Guangcheng Zhu, Shenzhi Yang, Zhengqing Zang, Yingfan Ma, Cheng Peng, Bowen Song, Weiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.20691)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has substantially improved the reasoning abilities of large language models, but its dependence on ground-truth answers or external verifiers limits its ability to exploit abundant unlabeled reasoning data. Unsupervised RLVR alleviates this bottleneck by deriving scalar rewards from the model's own rollouts. However, the reliability of such self-generated supervision varies substantially across rollout groups and individual responses, while existing methods construct or refine their rewards mainly from decoded outcomes and reward statistics. These endpoint signals reveal what answers the model produces, but provide limited information about whether the underlying rollout representations form a reliable structure for learning. Our key observation is that hidden-state spectra offer a complementary view of supervision reliability by characterizing both the organization of a rollout group and the support received by each rollout within that group. Based on this observation, we propose SpeCal, a bi-level spectral calibration framework for unsupervised RLVR. At the group level, SpeCal combines spectral mass with a Gaussian band-pass gate over normalized entropy-based effective rank, favoring rollout groups with substantial representation variation whose effective dimensionality lies in a preferred regime. At the rollout level, the same group effective rank determines how many dominant directions are retained in a leave-one-out spectral subspace formed by the remaining rollouts, within which the conditional support of each held-out rollout is measured. The resulting signals calibrate the original reward-induced advantages at the group and rollout levels without redefining the underlying unlabeled reward. 

---
# MASTA: A Feedback-Scheduled Multi-Agent System for End-to-End Tamarin Modeling and Analysis 

**Authors**: Yongkang Xiao, Qiyi Deng, Jing Chen, Min Shi, Ju Ma, Ruiying Du  

**Link**: [PDF](https://arxiv.org/pdf/2611.20671)  

**Abstract**: Tamarin-based formal verification of security protocols requires expert-authored multiset rewriting rules and temporal-logic lemmas. Translating a natural-language protocol description into a complete formal model with an LLM is hard because syntactic, reachability, and property-level errors compound across the multi-step modeling process, and a single agent that both authors and self-evaluates cannot reliably localize them. We present MASTA, a multi-agent system that addresses these difficulties through three innovations. First, an operational--evaluative role decomposition assigns rule modeling to an Architect, property modeling to an Auditor, and pure dispatch to a dedicated Arbiter that holds no authoring duties. Second, the Arbiter implements a Tamarin-anchored scheduler that dispatches the next agent based on the latest verifier verdict and routes problem-tagged fixes through a modification list, with a six-class problem taxonomy and an LLM deduplication check preventing routing deadlock. Third, we construct a benchmark of 10 protocols paired with 139 mutation variants and five fine-grained metrics that jointly assess model correctness ($M_1$--$M_3$) and completeness ($M_4$--$M_5$). On this benchmark, MASTA on Qwen 3.5-plus improves over OpenCode on the same backbone by up to $+0.27$ on individual metrics and ranks first on both the correctness and completeness averages, surpassing Codex (GPT-5.5) and Claude Code (Opus 4.7) despite using a weaker LLM. Ablations on five protocols confirm that removing the Arbiter collapses property and verification scores to zero. 

---
# Instrumenting Failure: Scoring Language Model Agents at Every Decision Node in Imperfect-Information Games 

**Authors**: Michael Kingston, Sonit Singh, Alan Blair  

**Link**: [PDF](https://arxiv.org/pdf/2611.20644)  

**Abstract**: Agent evaluations report task success and rarely say what an agent did at each decision. We score every decision node of three 12B to 14B instruct-tuned language models (LMs) playing five zero-sum imperfect-information games against four opponents on eight properties from decision and game theory, five of them against a CFR+ equilibrium reference. At each node, the LM also states a belief about the opponent's hidden state in a separate call. Without memory and at temperature 0, stated beliefs sit 0.250 to 0.288 in total variation distance from the equilibrium posterior, and where the action responds to the state its agreement with that belief exceeds chance by -0.04 to 0.19. The two calls frame the opponent differently, so the result is a behavioural inconsistency and makes no claim about internal beliefs. Under one prompt and greedy decoding, the LM's action never varies with the state in 10 of the 38 cells where the reference varies its own. Memory lowers the constant-action share in 9 of 13 testable ladders at temperature 0; at temperature 0.2 the cells at 1 fall from 16 to 5 of the 46 informative cells, while the belief results hold. A gate withholds, with a stated reason, any belief score the run cannot support, and seven of the eight non-zero defect rates are attributable to the instrument, a representation, or the LMs. 

---
# Safety-Preserving Medical LLM Fine-Tuning via Bilevel Data Selection 

**Authors**: Ruoxuan Shi, Sheng Yang, Zhengxing Su, Xiaoyang Hou, Yating Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20643)  

**Abstract**: As general large language models (LLMs) are increasingly adapted for patient-facing medical applications, improving medical capability while controlling patient-facing risks remains a critical challenge. Large-scale medical instruction data are essential for acquiring domain knowledge through supervised fine-tuning (SFT), but unsafe responses can undermine existing safety alignment. To address this problem, we propose Scalable Bilevel Data Selection (S-BDS) for safety-preserving medical SFT. By controlling selection-weight concentration, S-BDS enables reliable sample ranking over large candidate pools and identifies training examples that improve medical capability while controlling patient-facing risks. We combine clinician-validated safe data and selected data with unscreened medical instruction data for SFT, and jointly evaluate medical capability and patient-facing safety on HealthBench and PatientSafetyBench, respectively. Experiments across Llama and Qwen models show that S-BDS achieves the highest safe rates among all compared methods, outperforming the strongest baselines by 4.64 and 3.00 percentage points, respectively, while maintaining competitive medical capability. Further experiments show that varying the safety-oriented weight during SFT controls the trade-off between medical capability and patient-facing safety. 

---
# Instructions Before Actions: Prospective and Retrospective Language Supervision for Driving Vision-Language-Action Data 

**Authors**: Angel Martinez-Sanchez, Ross Greer  

**Link**: [PDF](https://arxiv.org/pdf/2611.20631)  

**Abstract**: Driving vision-language-action (VLA) models can accept navigation instructions, yet predicting a driver's trajectory does not establish that a model follows the instruction. We study how driving-language supervision is collected and evaluated, distinguishing prospective instructions issued before an action from retrospective labels written after observing it. We develop ADVLAT-Engine to synchronize spoken instructions, road-scene video, and vehicle trajectories, and compare three navigation apps on 48 replayed routes. On 87 maneuvers announced by both Google Maps and a passenger, the passenger uses more scene-dependent language. We evaluate four open driving VLAs before and after fine-tuning on prospective navigation-app drives. Fine-tuning reduces passenger-drive trajectory error by 23-59%, but instruction effects remain inconsistent. At real intersections, counterfactual instructions reveal stronger responsiveness: fine-tuned Alpamayo~2 Super gains 31 and 18 accuracy points at 15 and 5~m, while Qwen-Drive follows alternative maneuvers more reliably with a discrete command than with a sentence. We additionally collected retrospective annotations from four people for the driving recordings and describe controlled comparisons of annotation content, timing, and downstream learning; those analyses are pending. Our findings separate maneuver prediction from instruction-following and establish a framework for comparing prospective and retrospective supervision. 

---
# Newton-PINet: A fast physics-informed neural network with Newton linearization for meta-learning nonlinear PDEs 

**Authors**: Yuchen Fan, Chang Wei, Pao-Hsiung Chiu, Chin Chun Ooi, Heyang Wang, Jian Cheng Wong  

**Link**: [PDF](https://arxiv.org/pdf/2611.20609)  

**Abstract**: Scientific machine learning has opened new avenues for solving parameterized partial differential equations (PDEs), enabling models to learn a family of PDEs and generalize to unseen instances. In this context, data-driven operator learning methods typically require large training datasets, while physics-informed neural networks (PINNs) suffer from difficult optimization and limited generalization, especially for nonlinear PDEs. We propose Newton-PINet, a physics-informed network enhanced by Newton linearization, offering an effective meta-learning framework for nonlinear PDEs. Newton-PINet (i) employs a physics-informed multilayer network with skip connections, where the output-layer weights are solved by least squares; (ii) adopts a two-stage learning strategy that first leverages gradient-based training to learn robust representations from the available training tasks, and then performs gradient-free fine-tuning on the output layer for fast task-specific generalization; and (iii) incorporates a Newton linearization method to speed up the least-squares iteration for nonlinear PDE problems. On a challenging nonlinear reaction-diffusion benchmark, Newton-PINet achieves up to three orders of magnitude lower relative error than recent neural solvers, while using 16x fewer training tasks and over an order of magnitude less training time (about 5 minutes versus several hours). This work advances the meta-learning of PINNs toward data-efficient, fast, and generalizable physics solvers. The code and datasets are publicly available at: this https URL. 

---
# Same Papers, Different Verdicts: Tracing Version Drift in LLM Peer Review 

**Authors**: Jizhou Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.20608)  

**Abstract**: Venues and benchmarks read LLM review scores as measurements of a paper, while providers replace their models within months. We audit twenty frontier models from OpenAI, Google and Anthropic on ICLR 2027 submissions while they were under review, with tests registered before the decisions. Between an earlier and a later version of Claude Opus and of Gemini, the share of papers placed on the accept side falls by 52 and 53 percentage points on 183 and 60 randomly drawn submissions, and by 43 to 79 points in six further paired comparisons, with no paper moving up in any of them. Stating a 30% acceptance rate in the prompt leaves a gap of 79 points between the two Opus versions. When score-stripped critiques are swapped between versions, the later critiques alone lower the accept share by 35 points for Opus and 51 for Gemini; for Opus the drop is 35 points under a third-party scorer and 24 after a rewrite that keeps each technical point and removes evaluative wording. On 57 papers, the percentiles of Opus 4.6 and 5.5 correlate at 0.73 while their accept shares differ by 44 points. On 99 to 119 submissions, the self-estimated percentiles of three models correlate with two human reviewers more strongly than a third reviewer does (point estimates 0.17 to 0.21, 95% intervals between -0.04 and +0.42). All twenty models give a 2 or a 3 in 473 of their 475 reviews, against 60 of 72 human reviews. Retrospective tests on decided papers can draw on memory. Claude Opus 5.5 names the true first author of 27 decided ICLR 2026 papers from their titles alone, with no wrong name, and Claude Opus 4.6 accepted all 11 decided papers it said it recognized, against 5 expected. Thresholds on LLM-reviewer scores therefore have to be validated against human decisions for each model version; our tests against the ICLR 2027 decisions follow on 16 December 2026. 

---
# Language Models Meet Vibration Testing for LEO Constellation Satellites with Calibrated Response Prediction from Few-show Sweeps 

**Authors**: Jun Wu, Deming Li, Zhenming Zheng, Honghao Yue, Fei Yang, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.20579)  

**Abstract**: Large language models (LLMs) increasingly drive engineering analysis end to end, turning textual records into the inputs of numerical programs, but a plausible and wrong value can pass silently into a decision. We propose deterministic validation gates for LLM-driven engineering analysis, a general paradigm in which a typed configuration produced by LLMs is consumed only after it passes deterministic checks on units, physical consistency, verbatim quotes, agreement between independent models, consistency with measurement exports and the applicability of trained models. The gates target hallucinations from two directions, since quote checking catches fabricated content while agreement between models catches errors of omission that quote checking misses. We instantiate the paradigm on sine vibration acceptance testing for low Earth orbit mega-constellation satellites, where DeepSeek V4.1 Flash and Qwen-plus read Chinese test records and extract a configuration of 76 fields. Fact-level F1 ranged from 0.98 to 1.0 on five input variants, with all repairs frozen on development inputs and evaluated on held-out inputs. With all gates, 20 of 20 configurations produced notching results bit for bit identical to the gold run, and all 17 real and injected errors in consumed fields were stopped, of which 16 would have run without gates. Retraining the bound model on a dataset built from an LLM configuration reproduced its results within GPU nondeterminism. We also report a negative result of practical value. Direct LLM curve forecasting only matches linear extrapolation, whereas a small physics-structured network reduces the over-notched bandwidth by 86% to 89% at calibrated coverage c = 0.90, which delineates where direct LLM prediction should give way to a trained model. 

---
# Weight-Space Programming Changes How an LLM Uses Python for Computational Biology 

**Authors**: Makoto Sato  

**Link**: [PDF](https://arxiv.org/pdf/2611.20566)  

**Abstract**: Weight-space programming allows a language model to edit another model's weights without gradient-based training. We used Claude Opus 5.5 to edit Qwen3.5-0.8B and investigated how the edits changed its use of Python on computational biology problems. We compared four initial edited checkpoints with the original model on 50 test problems across three test seeds, yielding 600 paired evaluations. Strictly correct answers increased from 44 to 77, and technically correct answers, which included correct final values rejected for formatting, increased from 64 to 100. Repeated Python calls decreased from 1,834 to 1,044. Among the 64 pairs in which only the edited response was technically correct, the original model had already computed the correct value in 36 but did not submit a technically correct final answer. The edited model corrected the computation in 13 of the 64 pairs. With six additional checkpoints on the same 50 problems, technically correct answers rose from 96 to 154 across 900 pairs. We explored three selected problems with ten test seeds for each of ten checkpoints. On the melting-temperature problem, correct Python outputs decreased from 60 to 22 while technically correct final answers increased from 0 to 18. Weight editing changed both the calculations Qwen ran and whether it submitted a correct answer. 

---
# Memory-Efficient Speculative Decoding with Quantized Draft KV Cache 

**Authors**: Saksham Gupta, Michael Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.20565)  

**Abstract**: Speculative decoding reduces generation cost, but it stores two KV caches: one for the target model and one for the draft model. We reduce this extra memory by quantizing the draft cache. The target cache and verifier remain in BF16, so that compression cannot change the target model's final correction rule. We first study how to divide precision between keys and values. A naive per-token quantizer makes keys appear much more sensitive: splitting a 12-bit budget, K8V4 beats K4V8 in acceptance by 10.87pp. However, with grouped per-channel keys, the gap becomes -0.25pp. A controlled Qwen test confirms that changing only the key quantizer can erase or reverse the preferred bit split. Hence, following KIVI, we propose K4V4 quantization with per-channel keys and per-token values. Across six model pairs, our method reduces draft cache memory by 66.20%, and combined memory across both caches by 23.95%. Mean acceptance drops from 57.93% to 57.14%, only 1.36% relative. Overall, quantized draft caches can make speculative decoding more memory efficient; as a predictive principle for efficient AI design, key and value precision must be co-designed with the quantizer layout. 

---
# One-Shot Calibration via Model Editing 

**Authors**: Jiaxing Qiu, Luke Hakso, Roxana Daneshjou, Ahmed Alaa, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2611.20562)  

**Abstract**: Model editing aims to correct errors in large, pretrained models by replacing model generations with user-provided responses. Existing editors assume a single target answer for each edit. However, many real tasks inherently entail uncertainty, resulting in multiple plausible answers with varying likelihoods in user feedback. Such feedback forms a target distribution to not only correct but also calibrate model generations. We therefore propose SoftEdit, the first model editor to leverage uncertainty in user responses and perform one-shot calibration during editing. To generalize calibration to unseen samples, SoftEdit uses co-plausibility relationships between answers while preserving real distributional shifts, balancing the new distributional generality-locality tradeoff. Across three language models, three vision-language models, and two large datasets, SoftEdit achieves state-of-the-art performance, ultimately showing that uncertainty in user responses enables generalizable calibration of pretrained models beyond single-answer corrections. 

---
# AsymEEF: Bimanual Manipulation with Asymmetric End Effectors 

**Authors**: Xiaokang Sun, Ningyue Wang, Kunqi Xu, Guangqi Jiang, Jianglong Ye, Yong Wu, Xueyan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2611.20557)  

**Abstract**: Teleoperated demonstrations are costly to collect and rarely reusable across end-effectors with different kinematics and action spaces. This burden grows in bimanual manipulation, where each end-effector pair may require new data. We present AsymEEF, a framework for executable cross-end-effector demonstration generation and unified bimanual policy learning. Given a small source dataset, AsymEEF transfers demonstrations to heterogeneous combinations of two-, three-, and five-fingered end-effectors without additional teleoperation. Kinematic teacher anchors provide geometric supervision, while sparse real-world X-Calib anchors correct discrepancies in grasp execution. A shared encoder maps wrist poses and actuator states to a morphology-agnostic latent space, and a morphology-conditioned decoder reconstructs target commands. Generated trajectories undergo replay filtering for task-level executability. The representation also supports a unified bimanual visuomotor policy trained across multiple pairs. Across four simulated and six real-world tasks, AsymEEF improves replay success over adapted transfer baselines, achieves a mean replay-valid expansion gain of 16.6 times in simulation, and is evaluated through downstream policy training on real robots. Project website: this https URL. 

---
# GranularGPT: Natural-Language-Driven End-to-End Discrete Element Method Simulations 

**Authors**: Erlei Li, Sirui Tang, Ai Wang, Mingwei Hu, Yuchen Dai, Geoff Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.20551)  

**Abstract**: Discrete element method (DEM) simulations are widely used to investigate granular and particulate systems, but their application often requires specialized expertise in particle physics, numerical modelling, and simulation post-processing. Large language models (LLMs) offer new opportunities to simplify these workflows by enabling users to interact with simulation tools through natural language. This study offers GranularGPT, an LLM-assisted agentic framework for end-to-end DEM simulation workflows, from natural-language problem descriptions to simulation execution and automated post-processing. The framework enables users to generate DEM simulation cases, execute and monitor simulations, and automatically extract and report physical and process-level metrics. Evaluation across 60 normal-condition cases covering 12 DEM scenarios and parameter variations achieve a simulation success rate of 91.7%. In the error-handling evaluation, all 10 repairable errors are detected and 9 are successfully resolved, while all 5 non-repairable cases are correctly identified and reported. The framework automatically extracts a range of DEM-relevant metrics, including particle size distributions, packing and density measures, velocity profiles, discharge rates, mixing indices, contact force characteristics, and thermal quantities. GranularGPT provides a foundation for improving the accessibility and efficiency of DEM simulation workflows while reducing the specialized expertise required for their operation. 

---
# LINK: Lightweight and Inductive Node-to-Node Propagation Rule Learning for Traffic Kriging 

**Authors**: You Shang, Yu-Kwan Ching, Weifeng Su, Guanyao Li, Ruiyuan Li, Gary Shueng-Han Chan  

**Link**: [PDF](https://arxiv.org/pdf/2611.20521)  

**Abstract**: Spatio-temporal traffic kriging aims to reconstruct traffic time series at unobserved locations with no historical traffic observations by leveraging spatial and temporal dependencies across the road network. Existing Graph Neural Network (GNN)-based approaches primarily rely on neighborhood aggregation to capture spatial dependencies, thereby modeling traffic interactions implicitly and making it difficult to accurately recover traffic magnitudes at locations without historical observations. To address this limitation, we propose LINK, a Lightweight and Inductive Node-to-node propagation rule learning paradigm for traffic Kriging. LINK introduces three key designs: 1) a propagation-centric traffic kriging paradigm that explicitly learns a shared node-to-node transition rule; 2) a context-aware Spectral Propagation Network that performs traffic-state propagation in both the spectral and time domains; and 3) TraffiKrig, a realistic traffic kriging benchmark that preserves heterogeneous road-segment characteristics and road-specific metadata while enabling systematic comparison of representative kriging methods under a unified metadata-enhanced setting. Extensive experiments demonstrate that LINK achieves strong reconstruction accuracy and computational efficiency. Compared with the strongest baseline on each subset, LINK achieves an average MAE reduction of 54.7% over original baselines. LINK also achieves substantial efficiency improvements, reducing training time by 66.5% and inference time by 86.6% compared with the fastest competing methods. Zero-shot experiments further demonstrate the transferability of the learned propagation rule across regions. 

---
# Beyond Seeing: Fine-Grained Preference Validation and Severity-Aware Alignment for Mitigating Visual Hallucinations in Vision-Language Models 

**Authors**: Abdulaziz Alqahtani, Alwaleed Alharthi, Muzammil Behzad  

**Link**: [PDF](https://arxiv.org/pdf/2611.20478)  

**Abstract**: Large vision-language models can produce fluent responses containing objects, attributes, or relations unsupported by the image. We study an implemented alignment pipeline that normalizes released fine-grained hallucination supervision, validates chosen-rejected preference pairs with vision-language judges, repairs rejected chosen responses, and trains LLaVA adapters with standard or severity-aware direct preference optimization. The study uses 16,143 normalized feedback records and 8,386 released preference pairs, comparing direct training with zero-shot and two-shot judged severity-margin variants. The zero-shot variant reduces Object HalBench CHAIRs from 53.00 to 36.00 and CHAIRi from 15.72 to 10.41. Across aligned models, AMBER-Hal decreases from 35.90 to 25.10-26.90, accompanied by reduced visual coverage. In contrast, the base model retains the highest valid POPE adversarial F1, 84.18 versus 83.87-83.91 after alignment. These results support benchmark-dependent reductions in generative hallucination rather than uniform superiority of severity-aware alignment. We distinguish implemented pipeline components from the preference paths used in the reported experiments and discuss the limits of attributing gains to judging, repair, or the training objective individually. 

---
# Improving X-Ray Baggage Instance Segmentation with Shared Point-Based Mask Refinement 

**Authors**: Abdulaziz Alqahtani, Qasim Umer  

**Link**: [PDF](https://arxiv.org/pdf/2611.20476)  

**Abstract**: Overlapping projections and complex object contours make prohibited items difficult to delineate in X-ray baggage images. This study investigates shared point-based mask refinement to improve instance segmentation within Cascade Mask R-CNN. The framework combines a ResNet-101 backbone with an attention-based feature pyramid and progressively updates uncertain mask locations, increasing mask resolution from $28 \times 28$ to $112 \times 112$. A single refinement module is shared across cascade stages and refinement levels, with transformer and multilayer perceptron (MLP) variants used to examine the role of interactions among selected points. Ten configurations were evaluated on the official PIXray split across three training seeds using the original target-instance annotations for evaluation. The transformer and MLP refiners achieved mean mask average precision (AP) of 71.79 and 72.47, respectively, compared with 70.50 for the unrefined detector, yielding gains of 1.29 and 1.97 AP points. The MLP achieved the highest mean mask AP among the evaluated configurations. Both variants also improved boundary agreement on commonly detected objects, supporting improved mask delineation, while detection diagnostics did not establish a corresponding positive gain. These findings demonstrate that shared point-based refinement can improve segmentation quality in the tested cascade architecture using existing target-mask annotations, without additional occluder supervision. Evaluation on independent X-ray datasets is needed to determine how well these improvements generalize. 

---
# Model Immunization Beyond Condition Numbers: The Importance of Plateau Regions 

**Authors**: Quoc Nguyen, Trung Le, Dinh Phung, Mehrtash Harandi  

**Link**: [PDF](https://arxiv.org/pdf/2611.20472)  

**Abstract**: Harmful fine-tuning can rapidly compromise safety alignment in Large Language Models, underscoring the vulnerability of deployed models to post-release modification. While prior work motivates model immunization by inducing ill-conditioned curvature to slow harmful optimization, this approach is not scalable due to the high computational cost of Hessian estimation. We refine this paradigm by extending the theoretical framework to prioritize plateau regions alongside curvature. We further introduce an immunization algorithm based on efficient Hessian estimation and bounds on the extremal singular values of the Hessian. We demonstrate the proposed immunization framework in three applications: (i) Fine-tuning-as-a-service settings where benign user data may be mixed with poisoned harmful data, (ii) open-weight settings where an adversary can fine-tune on harmful-only data with varied training dynamics, and (iii) robust unlearning, where immunized models better resist relearning attacks after unlearning. Extensive experiments confirm that our approach achieves robust defense against adversarial fine-tuning while largely maintaining general utility and downstream trainability. Our code is available at this https URL. 

---
# Tabular Foundation Models as Emulators for Forecasting Terrestrial Ecosystem Dynamics 

**Authors**: Zhihao Wang, Ruohan Li, Lei Ma, George Hurtt, Xiaowei Jia, Yiqun Xie  

**Link**: [PDF](https://arxiv.org/pdf/2611.20468)  

**Abstract**: Process-based terrestrial biosphere models such as the Ecosystem Demography (ED) model underpin global carbon budgets, but their cost limits global, high-resolution and scenario-rich simulation. Deep emulators such as DeepED reduce that cost but require task-specific architectures and training. Tabular foundation models offer a different route: TabPFN predicts new rows by in-context learning, without being trained on the task. We compare TabPFN-3.5 with two DeepED baselines, $DeepED_{LSTM}$ and $DeepED_{Transformer}$, on CarbonGlobe's overall evaluation, which trains on a uniform 1/16 sample of sites and tests on the other 50,779, predicting the 40-year state of all seven ED outputs. We then test every model, trained on present-day data only, on ED runs forced by SSP1-2.6 and SSP5-8.5 for 2021--2060 in six continental forest regions. With no training, TabPFN-3.5 has the lowest present-day height error of the three models and is within 4% of $DeepED_{LSTM}$ for soil carbon, but is less accurate for AGB, LAI and the carbon fluxes. In 2060, errors are several times larger than at the same sites under present-day forcing, and every model underestimates the biomass that ED gains. Under SSP5-8.5, TabPFN-3.5 is the most accurate model for height, AGB, soil carbon and LAI, with the lowest AGB error in all six regions, although, unlike ED, it predicts no larger biomass gain than under SSP1-2.6. Under SSP1-2.6, no model is best overall. 

---
# ReRead: Wrong Predictions, Useful Representations - Correcting Frozen Robot Policies by Re-reading Their Own Representations 

**Authors**: Xu Su, Shunlong Xiao, Qintao Yi, Mingkuan Zhao, Xinmiao Yu, Yizhe Li, Zihan Wang, Bokui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.20465)  

**Abstract**: When a pretrained robot policy fails on challenging tasks, common remedies include retraining the backbone, adding an external visual encoder, or introducing a memory module. We argue that wrong predictions do not imply wrong representations: a policy can retain the required information yet fail to read it out as the correct action. To test this view, we propose ReRead: with the backbone frozen, a corrector re-reads the current and two previous caches to predict an action residual. The corrector sees no images, is trained solely on offline demonstrations from all 50 RoboTwin 2.0 tasks, and adds only 4.17% latency. Ablations and interventions show that the correction relies on the backbone's own representations: none of four external visual encoders adds further gain, and swapping in past caches from other trajectories sharply increases action error. On the 11 hardest RoboTwin 2.0 tasks, ReRead improves the success rates of LingBot-VLA, $\pi_{0.5}$, and Fast-WAM by 23.45, 13.09, and 8.00 percentage points, respectively. In paired evaluation on all 50 tasks, it raises Fast-WAM's average success rate from 90.73% to 92.66%. On a Flexiv dual-arm robot, it raises success on a phase-ambiguous task from 0% to 36%. 

---
# Designing Deceptive Model Organisms that Successfully Evade Auditing 

**Authors**: Vasisht Duddu, Mark Wang, Lipeng He, Tim G. J. Rudner  

**Link**: [PDF](https://arxiv.org/pdf/2611.20459)  

**Abstract**: AI models can exhibit deceptive behaviors such as strategically underperforming on evaluations (sandbagging), faking alignment, or covertly acting on behalf of a hidden principal. Safe deployment therefore requires auditing techniques that can reliably detect such behaviors. To develop and test these techniques, we can train model organisms (MOs): models that exhibit deceptive behaviors and serve as realistic evaluation testbeds. However, for such testbeds to be informative, MOs must actively anticipate and resist auditing---a gap in prior work that can lead to overestimation of auditing effectiveness and underestimation of safety risks. We address this gap by designing MOs that capture worst-case deceptive behavior and evade high-effort auditing. Focusing on sandbagging, we frame MO design as a security game between an auditor and a deceptive model and identify three design requirements for sandbagging MOs: (i) effectiveness (underperforming on targeted capabilities), (ii) utility preservation (maintaining performance on all other capabilities), and (iii) robustness (resisting auditing). We propose three MOs: two optimized against blackbox auditing techniques and one optimized against both blackbox and whitebox techniques. Through a comprehensive evaluation, we show that, unlike MOs from prior work, our MOs satisfy all three requirements, providing worst-case baselines for developing and evaluating future auditing techniques. 

---
# Explainable Federated Learning via Global-Local Attribution Alignment 

**Authors**: Dawood Wasif, Terrence J. Moore, Chang-Tien Lu, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2611.20445)  

**Abstract**: Federated learning enables on-device training without centralizing data, yet existing systems still struggle to provide explanations that are both locally faithful and globally consistent under strict privacy and bandwidth constraints. Prior approaches either keep explanations siloed across clients, transmit heavy or sensitive artifacts, or replace expressive task models with interpretable surrogates that sacrifice accuracy. We propose xFedAlign, a model-agnostic framework that decouples task optimization in parameter space from explanation coordination in a compact group space. Each client distills a lightweight surrogate to produce private, per-class top-k attribution artifacts, which are robustly aggregated by the server into a Global Explanation Prior that softly aligns client explanations without constraining task learning. Across image, text, and tabular benchmarks with IID and non-IID partitions, xFedAlign matches FedAvg accuracy while consistently reducing explanation drift and improving deletion and insertion AUC relative to Local-XAI, FedAttr-Agg, and Fed-XAI, with only a few kilobytes of additional communication per round. Privacy and robustness evaluations further demonstrate reduced membership inference advantage and increased resistance to attribution poisoning, enabling consistent and trustworthy explanations in federated learning. 

---
# Attention Sinks as Spectral Spikes: A Mechanism Analysis of Gated Attention 

**Authors**: Seojin Kim, Yehjin Shin, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.20403)  

**Abstract**: Transformer attention exhibits attention sinks, where a small subset of source tokens absorbs disproportionate attention mass across layers. Recent work shows that post-SDPA output gating strongly reduces attention-sink behavior, while separate studies link attention sinks to gradient sinks and massive activations through backward training dynamics. Yet how the same gate affects forward sink structure, backward gradient concentration, and training-time representation signatures remains unclear. We address this gap by formulating attention sinks as sink-induced low-rank modes in token-space covariance, with directions aligned to high-mass attention columns. Under this view, post-SDPA gating acts as a mode-dependent contraction: it attenuates sink-induced spectral energy in the forward pass and locally reduces value-path gradients routed through sink columns in the backward pass. Empirically, random-matrix diagnostics show that gating reduces outlier mass, weakens sink-subspace alignment, and increases effective rank, while backward analyses show active-gate attenuation of first-token value gradients. Training-time diagnostics further show that gating suppresses the co-emergence of attention concentration, value-gradient amplification, and representation dominance. 

---
# Making Koopman Control Guarantees Hold on Robots 

**Authors**: Abulikemu Abuduweili, Changliu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20402)  

**Abstract**: A Koopman controller can stabilize its learned latent-space linear model yet fail to stabilize the nonlinear plant. The learned model propagates predictions in latent space, while the deployed controller repeatedly re-encodes the measured physical state. We show that nominal closed-loop stability transfers to the nonlinear plant when the learned and true closed-loop dynamics agree to first order on the embedding's tangent space at the target. We derive a robustness margin for approximate matching and show that a stable open-loop latent model cannot exactly match an unstable equilibrium. We introduce Tangent-Invariant Koopman (TIK), which estimates a local plant linearization from offline transitions and enforces first-order matching to this estimate. TIK recovers the fitted LQR gain at the target, while a bilinear predictor captures state-dependent actuation during large maneuvers. Across four simulated unstable benchmarks, all 40 Standard Koopman controllers have stable latent closed loops, but their deployed nonlinear systems are locally unstable. In contrast, all 40 TIK controllers locally stabilize their respective nonlinear plants. An offline stability test using empirical curvature estimates identifies stable TIK controllers on two cart-pole systems, but is inconclusive under high noise. Under explicit model-error and curvature bounds, an invariant terminal set guarantees constraint satisfaction after entry under local feedback. TIK achieves 88\% constrained cart-pole swing-up success with a 0.25 ms planning budget and controls simulated seven-joint arms within 1-5 ms per step. 

---
# Implicit-Reasoning-Enabled Elastic Semantic Communications for AI-Agent Networks 

**Authors**: Kaiwen Yu, Gang Wu, Jiancun Fan, Saviour Zammit  

**Link**: [PDF](https://arxiv.org/pdf/2611.20399)  

**Abstract**: AI-agent networks introduce new communication demands driven by proactive interactions and autonomous task execution. Existing adaptive semantic communication schemes primarily adjust semantic representations and resource configurations within predefined processing scales and capability sets, which may be insufficient for evolving task requirements. To address this gap, we introduce elastic semantic communications (ESC), a task-driven framework that extends semantic adaptation to the runtime execution and participation of task-supporting capabilities. ESC comprises two complementary dimensions: vertical elasticity adjusts the execution intensity of existing capabilities, while horizontal elasticity reconfigures the participating capability set. The framework connects these dimensions through the expected marginal value of additional information and computation. To realize ESC, we propose an implicit-reasoning-enabled device--edge mechanism that supports variable-depth local inference through shared-parameter state updates and edge-side continuation through wireless transfer of intermediate reasoning states. Experiments on Sudoku-Extreme demonstrate that the benefits of additional inference depend on wireless conditions, while task-aware edge invocation can preserve task accuracy with substantially lower state-transfer overhead. 

---
# Pretty Linear Liars: Unregularized Probes Favor Intervention-Unfaithful Features 

**Authors**: Karl Vilhelmsson Emneby, Julian Yocum, Cameron Allen  

**Link**: [PDF](https://arxiv.org/pdf/2611.20371)  

**Abstract**: Linear probing is a common technique in mechanistic interpretability despite the fact that the decodability of a given feature does not necessarily imply the network uses that feature. Intervention faithfulness, not merely decodability, is the appropriate criterion for whether a feature serves a defined active role in the model. Meanwhile, probe regularization has been used for feature discovery, but the impact of regularization on feature causality remains unclear. We study probe regularization systematically, starting with unregularized ordinary least squares (OLS) regression and considering increasing amounts of regularization via ridge regression. We show theoretically that OLS probes amplify low-variance directions in activation space that have spurious correlations with the target feature, whereas regularization reduces influence from these directions. Empirically, we find in a toy transformer and Llama-2-7B that intervention faithfulness increases up to a point, and then decreases again, and that the optimal regularization strength leads to features bearing striking similarity (in both direction and variance) to those learned by optimizing for intervention outcomes directly. Together, these results provide mechanistic justification for regularizing regression probes. 

---
# Structure-Guided Multi-Granularity Spatiotemporal Transformer for Efficient Monocular 3D Human Pose Estimation 

**Authors**: Sheng Xu, Aoduo Li, Rongli He  

**Link**: [PDF](https://arxiv.org/pdf/2611.20361)  

**Abstract**: Existing multi-granularity methods for monocular 3D human pose estimation emphasize spatial joint features while underusing skeletal structure, including global connectivity and temporal continuity. Moreover, modeling entire frame sequences incurs high computational costs. We propose a Structure-Aware Multi-Granularity Spatiotemporal Transformer that integrates structural priors across spatial and temporal representations for accurate and efficient pose estimation. A three-level feature extractor captures human topology at the joint, bone, and limb levels. Gated Adaptive Spatial Attention (GASA) embeds structural constraints into spatial attention, while Multi-Granularity Feature Enhancement (MGFE) uses hierarchical pooling and fusion to strengthen local details and global semantics. Temporal Segment Fusion (TSF) aligns temporal contexts across granularities to capture both short- and long-range motion patterns. Hierarchical Cross-Attention (HCA) promotes spatiotemporal interaction between granularities, and Keyframe Attention (KFA) selects informative motion segments to reduce redundancy and computational cost. Experiments demonstrate improved accuracy and efficiency over conventional Transformer-based approaches, highlighting the effectiveness of structure-aware multi-granularity modeling for monocular 3D human pose estimation. 

---
# Residual-Tail-Informed Flow Matching for Time Series Anomaly Detection 

**Authors**: Shyam Sundar Saravanan, Tie Luo, Sanjay Madria  

**Link**: [PDF](https://arxiv.org/pdf/2611.20334)  

**Abstract**: Time series anomaly detection (TSAD) models are often trained on nominal data using average-risk objectives, which emphasize dominant normal patterns. However, real time series can contain rare but valid nominal regimes that are underrepresented in the training distribution. We show that data-space flow matching can over-penalize such rare-normal windows and assign them anomaly scores comparable to those of true anomalies. To address this, we propose Residual-Tail-Informed Flow Matching (RT-FM), an unsupervised framework that uses a temporal variational autoencoder to estimate residual latent tailness and converts it into bounded, warmup-annealed weights for training a data-space flow matching model. Experiments on seven TSAD benchmarks show that RT-FM improves rare-normal coverage while preserving strong anomaly discrimination, substantially reducing false alarms on such windows and resulting in improved aggregate detection performance. The code is available at: this https URL 

---
# LENS: Replacing LayerNorm and Softmax in Transformers with Learnable Normalization States 

**Authors**: Yuhao Liu, Salim Ullah, Akash Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.20331)  

**Abstract**: LayerNorm and attention Softmax are critical operations in Transformers. However, they cannot be efficiently mapped to General Matrix Multiplication (GEMM) computations on hardware. Therefore, modern hardware accelerator designs, such as NPUs on ASIC/FPGA, consume a large amount of hardware resources or chip area to implement their specific arithmetic units. Building on previous work about LayerNorm, Softmax, and their alternatives, we introduce Learnable Normalization States (LENS), a framework that reformulates LayerNorm as tiny MLPs by abstracting a common structure from related methods and extends this method to Softmax, which moves more of the computation in Transformers to GEMM as a hardware-friendly architecture. We evaluate LENS on image classification with Vision Transformers (ViT) and autoregressive language modeling with Pythia models, including generalization tests on WikiText and LAMBADA. Across these tasks, LENS matches or exceeds baseline performance in several vision settings and retains most of the baseline performance in the remaining vision and language experiments. These findings may suggest that hand-designed normalization and attention functions are not the only choice as alternatives to LayerNorm and Softmax. LENS also provides a basis for exploring Transformer architectures for resource-limited accelerators. 

---
# Unreasonable Confidence from Simulator Misspecification: Directional Robustness on the SPEED+ Sim-to-Real Benchmark 

**Authors**: Nikolay Dolgov  

**Link**: [PDF](https://arxiv.org/pdf/2611.20321)  

**Abstract**: Vision-based navigation systems for orbit must be trained almost entirely on synthetic imagery, because real labelled images of a given spacecraft in flight rarely exist. Benchmarks like SPEED+ are built for that situation: the training set comes from a simulator, and the test sets are real images of a satellite mockup under two kinds of lighting. That simulator approximates how light reflects off mylar and aluminium, and like every scientific simulator it is misspecified. Its error is radiometric, since it misses the specular highlights, hard shadows and sensor saturation common in real images. We probe the resulting synthetic-to-real gap, the central risk for such systems, with a deliberately simple monocular pose estimator. In a 2x2 study of backbone adaptation and photometric augmentation, trained on synthetic images and tested on both real hardware-in-the-loop domains, we find three things. (1) The model does not know it is failing: regions conformally calibrated to 90% on synthetic data cover only 9-15% of rotation errors on sunlamp and 21-38% on lightbox, so a wrong pose carries the confidence of a correct one. (2) Transfer is asymmetric: scaling data and features improved synthetic rotation error from 1.65 to 1.03 rad but moved the harshly lit sunlamp domain by only 0.05 rad, under twice its seed s.d., whereas adapting the backbone's last stage carried 63-83% of its extra synthetic gain to both real domains. (3) Once the backbone adapts, augmentation helps only in the direction it models: our photometric stack improved the diffusely lit lightbox domain from 1.17 to 0.99 rad on every seed but moved sunlamp no more than seed-to-seed variation, while with the backbone frozen it helped sunlamp more. Every post-review experiment was pre-registered, and the code, per-image predictions and pre-registration record are at this https URL. 

---
# NF-VLA: A Normalizing-Flow Baseline for Vision-Language-Action Learning 

**Authors**: Ruichen Deng, Qinyu Zhao, Yong-Hyun Park, Mutian Tong, Liang Zheng, Jiatao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20314)  

**Abstract**: Autoregressive normalizing flows (NFs) provide a tractable density for learning and sampling continuous robot actions. We show that a single maximum-likelihood objective can train vision--language--action (VLA) policies competitive with leading methods on manipulation benchmarks. We present NF-VLA, which combines conditional autoregressive normalizing flows with score-based action refinement. Its deep--shallow architecture concentrates visual, language, and proprioceptive prefix conditioning in a high-capacity deep block, while unconditioned shallow blocks provide a shared invertible action transformation. The vision--language backbone and flow are jointly trained by exact maximum likelihood on Gaussian-smoothed action chunks. The same learned density supplies a single score-based correction at inference, refining generated actions to account for training-time smoothing without a separately trained denoiser. Trained on LIBERO from a pretrained vision--language backbone without robot pretraining, NF-VLA achieves 98.35\% success across the four standard suites and 72.6\% on LIBERO-Plus using the same checkpoint without further fine-tuning. Ablations show that deep-only prefix conditioning and score-based refinement both improve closed-loop success. Together, these results establish carefully structured autoregressive flows as a competitive approach to VLA control, with one learned density supporting policy training, action generation, and refinement. 

---
# Multi-output neural network surrogate for double-pulse femtosecond optothermal control of thin silicon films on fused silica 

**Authors**: George D. Tsibidis  

**Link**: [PDF](https://arxiv.org/pdf/2611.20301)  

**Abstract**: We introduce a multi-output feedforward neural network surrogate trained on a coupled multiple-reflection and two-temperature model to map double-pulse femtosecond laser excitation (1030 nm) of crystalline silicon films on fused silica across film thicknesses (50-500 nm) and inter-pulse delays (0-10ps). Trained via Levenberg-Marquardt optimization on 2000 physical simulations, the dual-hidden-layer network ([40,30] neurons) simultaneously predicts six strongly coupled optothermal observables, peak electron temperature, peak lattice temperature, peak carrier density, minimum and maximum absorptivity and minimum refractive index achieving coefficients of determination R2 larger than 0.9998 across all targets. Evaluating a dense 200 by 200 design grid in milliseconds resolves pronounced thin-film interference fringes and carrier-induced index modulation, identifying an optimal optothermal coupling regime at d=156 nm and Dt=4.572 ps. This unified machine-learning workflow eliminates computationally expensive multi-physics runtime bottlenecks for real-time laser processing control and inverse photonic thin-film design. 

---
# Benchmarking LLM Inference on Desktop-Scale NVIDIA Grace Blackwell Clusters 

**Authors**: Kevin Bhimani, Christian Escobar, Daniel Mihovch, Rudolf Eigenmann, Sunita Chandrasekaran  

**Link**: [PDF](https://arxiv.org/pdf/2611.20289)  

**Abstract**: Two NVIDIA DGX Spark GB10 systems connect directly over 200 GbE to serve larger language models and increase inference capacity, but the benefits depend on how they are deployed. We benchmark 17 open-weight checkpoints spanning dense and mixture-of-experts architectures, 8B to 235B parameters and four weight precisions on one and two RDMA-connected systems. Using vLLM, we compare tensor parallelism (TP=2), pipeline parallelism (PP=2) and a two-replica baseline at concurrency 1 to 64, with three independent launches per configuration. TP=2 delivers a median 1.59x peak-throughput gain over one Spark, versus 1.39x for PP=2 and has higher throughput in 93 of 96 matched comparisons with separated 95% Welch intervals. It reduces unloaded inter-token latency to 0.52x to 0.73x of one Spark, whereas PP=2 provides little reduction. TP=2 also achieves the highest goodput under a strict 100 ms inter-token latency objective wherever both strategies are supported and the objective is met. For models fitting on one Spark, TP=2 generally favors low-concurrency serving, while replication offers higher throughput at high concurrency. One Spark uses less GPU energy per output token at matched concurrency. Across three precision ladders, NVFP4 reduces unloaded latency more than adding a second Spark at BF16. NCCL microbenchmarks and comparisons with an earlier TCP campaign suggest that collective communication overhead contributes to the advantage of TP=2. These results provide deployment guidance for desktop-scale inference. 

---
# Protecting Safety during Reasoning Training with Low-Rank Fisher Regularisation 

**Authors**: Neel Rajani, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2611.20288)  

**Abstract**: Large language models are trained to have safety guardrails, but these can be compromised while fine-tuning for new capabilities, such as reasoning. One way to mitigate this degradation is to modify the fine-tuning objective to preserve safety behaviour, but doing so effectively and efficiently remains challenging. We revisit Elastic Weight Consolidation (EWC), which penalises parameter updates based on sensitivity estimates from the diagonal of the Fisher information matrix (FIM). While EWC provides some protection, the diagonal approximation ignores interactions between parameters. We address this limitation by exploiting recent findings that safety-relevant behaviour is located in a low-rank subspace within models. Our method, Subspace-Projected Elastic Weight Consolidation (SP-EWC), projects gradients onto this subspace to capture non-diagonal curvature efficiently. We investigate three different ways of identifying the safety subspace: through safety fine-tuning updates (LoRA), safety-related activations (ActSVD), and contrastive representations of safety and utility data (cPCA). We find that this choice substantially affects performance, with SP-EWC using cPCA comparing favourably against existing methods by better maintaining safety guardrails without introducing excessive over-refusal. 

---
# Self-supervised learning for zebrafish embryo phenotype evaluation: Systematic comparison of algorithms, label-efficient classification, and transfer learning 

**Authors**: Thomas Lautenschlager, Gaëlle Hayot, Quentin Brassart, Angelo Jovin Yamachui Sitcheu, Linus Schilpp, Tijana Geroski, Tetyana Goldanova, Stephan Allgeier, Thibaut Soubrié, Milica Nikolic, Thomas Dickmeis, Ralf Mikut, Nils Friederich  

**Link**: [PDF](https://arxiv.org/pdf/2611.20269)  

**Abstract**: Zebrafish embryos are a prominent model organism for image-based phenotyping and especially suited for large-scale imaging studies, for which machine learning (ML) models can automate evaluation. However, training ML models usually requires large labeled datasets, and differences in imaging conditions often impede reuse across laboratories. Self-supervised learning (SSL) addresses both constraints by learning transferable representations from unlabeled images. Yet central aspects like the choice of algorithm remain underexplored for domain-specific applications. To close this gap, we benchmark popular SSL algorithms on the public EmbryoNet dataset, explore label-efficient learning, and test transfer to an unseen setting. Using representations from our strongest model, a ResNet50 network trained with SwAV, a simple linear classifier reaches 86.4% accuracy and a non-linear classifier 87.1%, rivaling the fully supervised EmbryoNet model (87.1%). The SwAV model decisively outperforms supervised training from scratch when labels are scarce (70.2% vs. 45.7% accuracy with 5% of labels). On a three-class problem (normal, dead, falsely cropped) with a small, previously unseen zebrafish toxicity dataset, recorded in a different setting, our model reaches 81.3% accuracy without retraining, while EmbryoNet achieves 75.9%. Overall, self-supervised models for zebrafish embryos seem to mostly facilitate label-efficiency and transfer to new settings rather than improving supervised results. 

---
# GNN-Based Deep Reinforcement Learning for Coastal Emergency Response under Extreme Water Levels 

**Authors**: Marcello Sano, Samuele Casagrande, Davide Ferrario, Sebastiano Vascon, Silvia Torresan, Andrea Critto  

**Link**: [PDF](https://arxiv.org/pdf/2611.20268)  

**Abstract**: When a storm surge approaches a coastal town, the emergency plan must be put into action quickly using water-level forecasts, flood extent, road access, vulnerable residents and shelters. Bringing this information together under time pressure is slow and depends on the duty officer's expertise. This paper is one step in a wider research line from data to safe decisions: machine learning to understand the hazard, reinforcement learning and planning to choose actions, large language models (LLMs) to explain and cross-check them, and an assurance layer that keeps people in charge. Here we address the action step on a real place. We ask whether a policy can learn the timing of emergency actions from decades of real events. Lido di Venezia is represented as a heterogeneous graph built from public elevation, road, building and census data and driven by hourly tide-gauge records from 1983 to 2024, with seven actions taken from the municipal emergency plan. A PPO policy is trained on this graph and compared, on storm windows unseen in training, with a rule-based system simulating the officer's discretionary decisions. On 18 historical test scenarios, PPO achieves a similar reduction in population exposure to the rule. With uncertain forecasts and limited protection it again performs similarly but does not outperform the tuned rule, and removing graph message passing does not substantially reduce performance. PPO therefore learns an effective protective response from an outcome-based reward alone: it is given the plan's actions and when each is allowed, but not the thresholds that tell the comparator when to act. Next steps test harder edge cases, move closer to a digital twin of the territory, and replace the simulated officer with assessment by emergency-management stakeholders, supported by LLM-based interpretation of each recommendation. 

---
# Estimating Outpatient Visit Prices from Hospital Price Transparency Data using Large Language Models 

**Authors**: Kurmanbek Kaiyrbekov, Nicholas J Dobbins, Sean D Mooney  

**Link**: [PDF](https://arxiv.org/pdf/2611.20261)  

**Abstract**: Objective: Hospital care prices in the United States have increased substantially, yet patients lack effective tools to access and compare costs. We aimed to develop and evaluate an LLM-based approach for improving access to and use of hospital price transparency (PT) data.
Materials and Methods: To estimate visit prices, we used a retrieval-augmented generation (RAG) framework to analyze PT files from three major hospitals in Maryland and the surrounding region. We compared BM25, OpenAI embedding models, and a GPT-BM25 hybrid against a no-RAG baseline for cost estimation of individual procedures and procedure groups, then evaluated the best approach for symptom-based visit cost estimation using synthetic simulations.
Results: Across hospitals, the GPT-BM25 retriever achieved the best performance, with average price MAPE of 19.44% for individual procedures and 4.90% for groups, compared with 147.98% and 82.82%, respectively, for the no-RAG baseline. In simulations, our framework correctly identified approximately 50% of reference procedures and reduced MAPE relative to the LLM-only baseline at two of the three hospitals, while achieving comparable performance at the third.
Discussion: Our framework uses RAG to improve care price estimation in real-world settings. Compared with the no-RAG baseline, it substantially improved cost estimates for both individual procedures and procedure groups. Improvements were smaller in synthetic simulations, likely because inferring procedures from clinical information is more complex. Practical challenges remain, including heterogeneous PT file formats, unexplained duplicate entries, and limited availability of insurer-negotiated rates.
Conclusion: LLMs that effectively integrate PT data show strong potential to provide patients with more accessible and accurate healthcare cost estimates. 

---
# Microfitting: Continual Learning via Cached Test-Time Training 

**Authors**: Nick Ryan, Peter A. Devine, Ben Sirb, Hannah Sansford, Alex Chiocchi  

**Link**: [PDF](https://arxiv.org/pdf/2611.20258)  

**Abstract**: Continual learning is a promising approach to specialize language models to an organization's private data. However, current methods exhibit a number of drawbacks including catastrophic forgetting, where new training updates overwrite prior knowledge. We present microfitting, which trains and caches a small adapter for each document. Each adapter is trained once, offline, on synthetic questions targeted at demonstrated base-model shortcomings and reused whenever that document is queried. This method borrows from test-time learning literature and sidesteps the problem of continual learning through parameter isolation, allowing the system to specialize on new data without overwriting prior knowledge. The process that generates adapter training data also improves over time, revising its recipe from previous outcomes and fresh examples of the target workload. On DeepSeek V4 Flash, microfitting substantially improves accuracy over the base model. In Enterprise Document Operations (EDO), an open-book Q&A environment over real-world enterprise documents, our method achieves a 38.2% relative improvement over base-model performance. Microfitting makes continual specialization practical and cost-efficient: a frozen model can adapt to an evolving private corpus overnight on local hardware, without modifying the base model or previously cached adapters. 

---
# Integrating Latent Block Models into Variational Autoencoders for Disentangled Representation Learning 

**Authors**: Anasse Essalih, Reda Khoufache, Laurent Durville, Mohamed-Cherif Rahal, Mustapha Lebbah, Hanene Azzag, Walid Kheriji, Fouad Hadj Selem  

**Link**: [PDF](https://arxiv.org/pdf/2611.20253)  

**Abstract**: We introduce LBM-VAE, a novel framework that integrates Latent Block Models (LBMs) into Variational Autoencoders (VAEs) to jointly cluster observations and latent dimensions. Beyond its clustering capabilities, LBM-VAE provides a block-structured latent representation that naturally supports disentangled representation learning. In contrast to conventional approaches that associate each factor of variation with an individual latent dimension, our framework allows factors of variation to be encoded across groups of latent dimensions. Experimental results demonstrate the promising clustering performance of LBM-VAE, while achieving competitive results in disentangled representation learning. LBM-VAE highlights the potential of combining co-clustering and variational inference to learn structured and interpretable latent representations. 

---
# Right Goal, Wrong Body: Re-grounding Frozen VLAs under Actuation Shift 

**Authors**: Umer Siddique, Peilang Li, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.20244)  

**Abstract**: Vision-language-action (VLA) models are capable of producing desired robotic behaviors in ideal settings. However, they often fail when the robot's actuation changes at deployment. For example, remounting, mechanical wear, or command latency changes can lead to failed tasks and unsafe behaviors. To address the inevitable actuation shift, we propose ReGround, an inference-time method that keeps the VLA frozen and re-grounds its intended motion. ReGround learns the deployed robot's dynamics from task-agnostic motor babbling, and then tracks the policy's intended motion while staying close to the policy's own plan. ReGround activates only upon detecting a shift, and needs no rewards, demonstrations, or retraining. We show via two VLAs, eight LIBERO tasks, and five shift settings that ReGround can restore near-oracle success, such as raising $\pi_{0.5}$'s success from 0.16 to 0.90 under 400 ms latency, while remaining safer than a linear-calibration baseline. Project website and open-source code: this https URL. 

---
# Markets for Compute and Inference: Structure, Financing, and Risk Transfer 

**Authors**: Dr Gabriel Borrageiro  

**Link**: [PDF](https://arxiv.org/pdf/2611.20242)  

**Abstract**: Capacity to run large language models is rented by the accelerator-hour, and their output is sold by the million tokens. Both goods now have published price indices, an over-the-counter forward has traded on one, a perpetual future on one is listed offshore, and CME and ICE have specified futures on rental indices that await the United States regulator. We ask what in these two markets can be hedged, by whom, and at what price. Three findings stand out. First, the price that exchange contracts settle on is not the price that drives operators' revenue. Every contract references the published short-dated rental rate, while most revenue, and with it credit quality, comes from private multi-year contracts whose prices are largely undisclosed. Second, the token price and the rental rate are linked by an identity through throughput and margin, but they are different exposures, so a hedge on the rental rate does not hedge the cost of inference; at the measured capability of GPT-5 the cheapest list price fell by a factor of 17 in twelve months, and by far less at lower capability. Third, the demand for hedging already sits in credit agreements, as the years by which operators' debt outlives their contracted revenue. Filings cannot measure that gap, because debt is scheduled year by year while contracted revenue rarely is, and the one loan that prices it settles the risk by cash prepayment against a model only the parties see. We design a reverse convertible note through which leveraged operators could buy protection against falling rental rates. Its coupon depends steeply on the annualised volatility of the rental rate, which the two public indices put between about 5% and 89% according to how each is built and the horizon measured; across that range the fair one-year coupon runs from 4.1% to 31.5%. 

---
# SWE-CoWork: Benchmarking Coding Agents as Digital Employees in Organizations 

**Authors**: Xinping Lei, Minghao Liu, Ruijie Zhu, Xiaohui Hu, Xingqiao Li, Ao Zhang, Wenhao Chai, Han Li, Tao Wang, Jingyuan Zhang, Enduo Zhao, Yunfeng Shi, Yiming Liang, Zetao Yang, Xingjian Wang, Yifan Yao, Xinkai Ma, Yingru Li, Tianhan Yang, Huan Zhou, Weiran Shi, Shen Yan, Jiaheng Liu, Stephen Huang, Eli Cheung, Weihao Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.20218)  

**Abstract**: A coding agent working within a software organization must establish what to build, collaborate with colleagues, and verify what it delivers. We introduce SWE-CoWork, a benchmark comprising 101 development assignments across 101 real repositories, 11 programming languages, and 12 application domains. Each assignment places an agent in a simulated engineering organization with 26 colleagues who have distinct responsibilities, bounded knowledge, and controlled collaboration behavior. Starting from a brief handoff, agents must recover requirements distributed across colleagues and workplace records, reconcile conflicting evidence, and implement changes evaluated by private executable tests. We evaluate eleven models and find a substantial gap between partial implementation and complete task delivery. The two leading models satisfy approximately 63% of fail-to-pass (F2P) tests but complete fewer than 8% of assignments. Providing complete requirements upfront improves F2P by approximately 16 percentage points for Seed Evolving and GPT-6 Astra, yet leaves complete task success unchanged. Requirement-level trajectory analysis reveals failures throughout the delivery process: Claude-Opus-5.5 directs a mean of 93% of first questions about queried cards to their accountable owners across runs but obtains only 64% of colleague-held requirements. Deferred questions go unfollowed, implementations remain incomplete even after requirements are acquired, and 31% of requirements reported as completed fail their tests. These findings highlight sustained requirement tracking, from clarification through implementation, verification, and handoff, as a key challenge for reliable autonomous software engineering. 

---
# Paying Creators for Training Data Can Increase AI Displacement 

**Authors**: Lijuan Luo, Emaad Manzoor, Nathan Yang, Fan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.20204)  

**Abstract**: Generative AI increasingly competes with creators while using their content as training data. In response, policymakers have called for platforms to compensate creators for training data use and allow opting out. We study how such policies affect creator displacement in a model of a platform where creators choose whether to contribute training data and consumers choose between purchasing from creators and prompting generative AI trained on their content. When compensation is required, the low-quality creator chooses to remain solely as a training data supplier while AI fully displaces her demand, despite AI quality and the price the platform charges for AI-generated content being identical with and without compensation. Without compensation, the risk of losing creators and their training data constrains the platform to set the price of creators' content to preserve their demand. This constraint also prevents further AI displacement as AI improves, with the platform raising AI prices while keeping creators' prices and demand unchanged to prevent exit. Compensation increases creators' payoffs and total surplus but reduces consumer surplus. Our findings highlight how compensation policies for training data can shape creators' economic roles in a future with more capable generative AI. 

---
# How Specification Explicitness Shapes Failure in LLM-Generated Code 

**Authors**: Elmira Onagh, Maleknaz Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2611.20195)  

**Abstract**: When LLMs turn requirements into code, every underspecified constraint becomes behavior that the model must decide during generation. We test when specifications move beyond structure and begin to control generated behavior. In a controlled experiment, 72 participants implemented and revised Python tasks with an LLM under three levels of specification explicitness. The results show a clear break between lightweight structure and explicit behavioral obligations. Lightweight structure reduced hidden-test failures by only 8.7%. Requiring participants to specify constraints, invariants, negative requirements, edge cases, tie-breaking rules, and exception handling cut failures by 47.2% and reduced the number of distinct failure modes by 20.5%. The largest drops were in failure modes tied to implicit obligations. Invariant violations fell by 66.1%, and unsupported assumptions fell by 59.5%. After a requirement change, specifications with explicit behavioral obligations produced regressions in more submissions, but each affected submission lost less previously correct behavior. Explicit specifications also changed how developers judged generated code. Without explicit obligations, participants overestimated generated-code correctness by about 23 percentage points. With explicit obligations, they produced better code but judged it more cautiously. Thus, the lesson is sharper than "better specifications help." Lightweight structure barely helped, while explicit behavioral obligations nearly halved failures, made regressions shallower, and improved calibration. 

---
# Cyber-Secure Wearable, Ultra-Low-Power Networked Sensors for Soldier Health Monitoring: Insights from the CAPSARII project 

**Authors**: Luciano Bozzi, Christian Celidonio, Umberto Nuzzi, Massimo Biagini, Stefano Cherubin, Asbjørn Djupdal, Tor Andre Haugdahl, Andrea Aliverti, Alessandra Angelucci, Giovanni Agosta, Gerardo Pelosi, Paolo Belluco, Samuele Polistina, Gaia Strada, Luigi Sossi Ortega, Alexandra-Ioana Albu, Riccardo Volpi, Luigi Malagò, Michael Schneider, Xabier Eguiluz, Haritz Saiz, María Viqueria  

**Link**: [PDF](https://arxiv.org/pdf/2611.20187)  

**Abstract**: The CAPSARII project advances a cyber-secure, ultra-low-power wearable platform and Internet of Battlefield Things (IoBT) framework to support real-time monitoring of military personnel and data-driven decision support. The architecture integrates smart-textile sensing, a body/squad area mesh, edge-deployed AI on secure RISC-V microcontrollers, and cloud/federated analytics for cross-mission learning and comparative studies. Two representative use cases--pilot fatigue and warfighters operating in harsh environments--guide requirements, validation, and interface design (including HUD and haptic feedback).
As an initial step toward embedded models, we benchmark stress classification on a public wearable dataset: tree-ensemble baselines achieve state-of-the-art performance for accuracy and F1 (e.g., XGBoost: $\mathrm{Accuracy}=0.901\!\pm\!0.023$, $\mathrm{F1}=0.920\!\pm\!0.019$), while a feature-based Transformer attains the highest AUC ($0.963\!\pm\!0.025$), informing feature-engineering and compression choices for edge deployment. Building on these results, CAPSARII employs HW/SW co-design, model compression (pruning, quantization, distillation), and precision tuning to meet stringent energy and memory budgets; it further investigates Processing-in-Memory accelerators for future efficiency gains.
A structured gap analysis highlights the remaining challenges: establishing validation standards for physiological signal quality and textile durability; achieving low-latency, scalable BAN/SAN communications under mobility and interference; and deploying hybrid lightweight/post-quantum cryptography with robust device identity and side-channel resilience on constrained endpoints. By consolidating these elements into a coherent, secure IoBT stack, CAPSARII delineates a practical roadmap from laboratory baselines to mission-ready soldier monitoring in both training and operational scenarios. 

---
# RoboKit: Accelerating Parallel Differentiable Optimization for Robot Learning 

**Authors**: Jianglong Ye, Xuanbin Peng, Conor Mc Gartoll, Changwei Jing, Lai Wei, Jiali Chen, Xueyan Zou, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.20185)  

**Abstract**: We present RoboKit, a high-performance, general-purpose toolkit for parallel differentiable optimization in robotics, built on three principles: (i) Performance and flexibility: GPU-parallel multi-seed solvers pair sampling with versatile optimizers, running 2-60$\times$ faster than task-specific baselines across seven tasks; (ii) Geometry, rendering and physics: hybrid and coarse-to-fine representations accelerate collision, differentiable rendering and simulation, optimizing 4096 grasps in 2.97 s; (iii) Integration with learning: costs are differentiable and simulation is parallel, so RoboKit both distills optimization into policies and trains RL; distilled policies run 9-240$\times$ faster at comparable accuracy, and RL reaches a GPU baseline's scores up to 2.1$\times$ sooner on the same simulator. 

---
# Can AI Judge You? An Empirical Audit of Deployment Readiness in LLM-assisted Hiring Systems 

**Authors**: Spandan Roy, Subhabrata Majumdar, Soudeep Deb  

**Link**: [PDF](https://arxiv.org/pdf/2611.20140)  

**Abstract**: Delegating resume screening to large language models (LLMs) requires organizations to establish whether they can distinguish candidates by relevant qualifications while allowing only authorized users to change screening rules or approve exceptions. To this end, we develop a system architecture that separates candidate evidence, evaluation, permission controls, and assessment record, and empirically examine two complementary dimensions of its deployment readiness. First, we assess its operational readiness on four LLMs using a controlled experiment on 210 resumes across 14 job roles and six decision interfaces. Second, we assess its compliance readiness by testing how it handles unauthorized user requests and instructions embedded in resumes. We find that operational performance depends on the choice of evaluation interface: scalar scores generally distinguish qualification changes better than simple pass/fail judgments, while comparative judgments are sensitive to presentation order. Explicit permission rules and separation of resume content from instructions reduce unauthorized compliance, to a varying degree of effectiveness across LLMs and usage contexts. Finally, we find that LLM evaluators often diverge between their decisions and offered justifications. Going beyond simply evaluating LLMs for hiring tasks, our framework supports an audit of a fully configured resume evaluation system to enable its deployment decision. 

---
# CapCover Filter: Semantic Approximate Membership Queries over Embeddings 

**Authors**: Dvir David Biton, Shay Hahiashvili, Nadav Keren, Gil Einziger, Roy Friedman  

**Link**: [PDF](https://arxiv.org/pdf/2611.20132)  

**Abstract**: Determining whether an embedding is sufficiently similar to any vector in a vector store without performing a nearest-neighbor search can substantially improve query throughput in vector-store systems. We study Semantic Approximate Membership Queries (SAMQs), which allow queries to be rejected when no indexed vector meets the requested similarity threshold, while guaranteeing no false negatives.
We introduce CapCover Filter, a geometric data structure that summarizes a set of embeddings using spherical caps. Its representation is independent of the query threshold, allowing a single filter to support arbitrary thresholds without reconstruction. We evaluate alternative construction and quantization methods, compare CapCover Filter with competing filters, and measure its impact on query throughput.
Across five real-world datasets, at a cosine similarity threshold of 0.95, CapCover Filter eliminates approximately 20--100\% of index accesses while using only 2\% of the raw (FP32) embedding memory. When placed before a FAISS flat index, it improves query throughput by up to 382\% and by approximately 116\% on average across the five datasets, while returning the same range-search result. We also evaluate lower similarity thresholds. Our results show that CapCover Filter can substantially improve vector-store throughput when similarity thresholds are tight, retrieval is expensive, and indexed embeddings are highly clustered. We release an optimized implementation of CapCover Filter and its evaluation framework. 

---
# Can decision models extract clinical data from cancer pathology reports? An evaluation of accuracy, calibration, and failure modes 

**Authors**: Indranil Mallick, Paromita Roy  

**Link**: [PDF](https://arxiv.org/pdf/2611.20129)  

**Abstract**: Decision models are a new class of language model that, instead of generating text, choose one option per question from a fixed list and return a probability for every option. They have not yet been evaluated on clinical text. We evaluated two open-weights decision models, Nimble 9B and Tev1 4B, for extracting 25 fields from 1,113 oral cavity cancer pathology reports from a single centre, against a prospectively maintained clinical database. We compared them with two generative models: Qwen3.5 9B, run locally, and GLM-5.3-flash, run through a cloud API. Tev1 reached a mean field accuracy of 0.917 (95% CI 0.913 to 0.920), the same as Qwen3.5 9B (0.918; p = 0.26), and was four times faster (15 s against 61 s per report). Nimble reached 0.862 and GLM-5.3-flash 0.986. Both decision models matched GLM-5.3-flash on simple dichotomous categories. They were weak on measurements binned into categories, on fields whose correct answer was not reported, and on some complex decisions (e.g. involved margins). We found that the main cause of the inferior performance of Nimble's came from asking 25 questions in one request: with two questions per request, its accuracy improved. Pooled over fields, the returned probabilities were well calibrated (expected calibration error 0.022 and 0.026), but they did not flag systematic errors within single fields. A simulated strategy of sending answers with a probability below 0.94 to GLM-5.3-flash raised Tev1's accuracy to 0.982, with 38% of answers escalated. Decision models can extract explicitly stated findings from pathology reports quickly on local hardware and return useful probabilities. They should be validated in the configuration in which they will be used, restricted to the fields in which they are reliable, and combined with other methods for numeric values. 

---
# CECAR: Cache & Expert Co-Aware Routing Accelerates On-Device Inference of MoE LLMs 

**Authors**: Byeongju Kim, Hoonki Lee, Byungjun Kim, Inyul Ra, Yongjin Kim, Yongseong Lee, Sangyeob Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.20119)  

**Abstract**: Despite growing interest in on-device LLM deployment, large-scale Mixture-of-Experts (MoE) models remain impractical on resource-constrained devices, as sparse expert activation still requires all expert weights to be memory-resident. To address this, we identify three key observations for on-device MoE inference: (a) cache hit rates are fundamentally bounded even under Belady's optimal policy; (b) these bounds are insufficient for low-latency on-device inference; and (c) prior cache-aware routing based on binary cache presence can induce routing instability. Based on these insights, we propose CECAR (Cache & Expert Co-Aware Routing), a MoE inference system that jointly considers expert routing and cache management. Unlike prior cache-aware routing methods, CECAR uses an ML-based cache policy to predict the expected reuse distance of each expert and prioritizes routing and execution toward experts with smaller predicted reuse distances. On the Qwen3-30B-A3B model, CECAR achieves 12.5 tokens/s on a consumer-grade GPU with only 16 GB of VRAM, where non-resident experts are fetched from SSD, delivering a 4.5x speedup over conventional MoE inference and exceeding the decode speed of baselines that keep the entire model fully resident in DRAM. Code is available at this https URL. 

---
# Locational Marginal Pricing of Compute: Co-optimizing Compute and Energy for Transparent Data Center Flexibility 

**Authors**: Vladimir Pyltsov, Teng Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2611.20116)  

**Abstract**: Demand from data centers and AI compute is growing rapidly and is increasing stress on the power grid. This load could be flexible through user adjustments and provide relief during energy scarcity events, yet today its flexibility is provided through utility call-ups or system-operator load shedding, and is invisible to the users whose workloads are being curtailed. In this work, we propose to quantify and couple the prices of compute and energy in a single optimization problem. The resulting compute price reflects the marginal cost of the binding constraints and decomposes into components indicating which grid, power, hardware, or service requirement causes a price surge. The economic interpretation updates the price with real-time conditions and gives users a signal for potential flexibility provision. The co-optimization ultimately connects compute and energy markets while providing a signal for relief of stress on both the grid and the compute resources. 

---
# MT-RGCN-DTI: A Multi-Task Relational Graph Convolutional Network for Drug-Target Interaction Prediction 

**Authors**: Dheeran Sankaran, R Yadunandan, Daniele Santoni, Nimisha Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2611.20109)  

**Abstract**: Prediction of drug-target interaction (DTI) has become a very crucial part of drug discovery. With the increase in the data volume, it has become very important to improve prediction accuracy by integrating multiple data sources. In this regard, utilising multiple resources in the form of heterogeneous information network to address DTI prediction can be rewarding. In this work, we propose a heterogeneous information network based multi-task DTI prediction model which leverages the advantages of relation graph convolutional network (RGCN). To achieve this, the model employs drug-target interaction network, drug-drug similarity network and protein-protein similarity network to construct a heterogeneous information network. Moreover, we use a knowledge graph embedding scoring functions to model interactions between drug and targets, specifically DistMult and ComplEx as decoders over the learned node representations. We performed extensive experiments on multiple benchmark datasets and compared with state-of-the-art methods. We observed that the proposed model shows competitive performance, thereby confirming the effectiveness of our approach. The codes and datasets are available at this https URL. 

---
# SAFER: Navigating the Exploration-Exploitation Dilemma in World-Model-Based Planning for Autonomous Driving 

**Authors**: Yansong Qu, Zilin Huang, Zihao Sheng, Sikai Chen, Samuel Labi  

**Link**: [PDF](https://arxiv.org/pdf/2611.20107)  

**Abstract**: An autonomous vehicle must foresee the consequences of its actions, especially in the rare, time-critical situations where crashes happen. A world model offers this foresight, but only within the coverage of its training data, which poses an exploration-exploitation dilemma. Within this coverage a planner can hardly improve on the data, and beyond it the model's errors mislead it. Logs of competent driving rarely contain the failures that would widen the coverage, and agents that learn from interaction gain them only by failing themselves, without being shown what to do instead. We propose SAFER, a framework that learns offline from failures that others make, judged by an expert, and keeps learning online from its own driving. Offline, partially trained agents and scripted, recoverable perturbations Stage failures in simulation, and an expert Annotates every visited state with its action and value, so that much of what the planner explores is covered before it drives. In planning, a chunked latent world model Foresees the consequences fast enough for real-time use, and a risk-aware sampling-based planner Evaluates candidate maneuvers by their return minus a bounded collision penalty. Online, a loop continuously Refines the model wherever this driving leaves the coverage of the offline data. In closed-loop evaluation against the compared methods, SAFER reaches the highest route completion and reward per step on the CARLA Leaderboard routes, 0.95 and 0.77 from a privileged bird's-eye view, and the highest episode reward in MetaDrive, 53.4 in regular and 49.0 in safety-critical traffic. Project page: this https URL. 

---
# Making Preemption Productive in Long-Output LLM Serving 

**Authors**: Ziji Shi, Xinyi Wan, Chaoyi Ruan, Guangxing Huang, Min Lin, Jialin Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.20106)  

**Abstract**: Reasoning LLMs generate long outputs whose key--value (KV) caches occupy GPU memory for minutes. When memory fills, the server preempts requests, and a preempted request can repeatedly lose and recover its state while producing little new output. We call this behavior KV thrashing. In certain cases, existing systems leave more than a fifth of requests stalled for over a minute. Because preemption cannot be avoided without delaying admission, we ask how to make it productive. We present Stint, a serving system that preserves a preempted request's work, bounds how long it waits, and gives it fresh output after it returns. Stint stores fixed-length KV snapshots in a contiguity-first CPU pool with an exact-extent allocator, and moves them with fused K/V gather/scatter kernels and double buffering. Age-based readmission brings stalled requests back, and protection gives them a turn of fresh decoding before another preemption. Remaining-length estimates from decoding hidden states guide victim selection. On olympiad-math replays with Qwen3-32B on H200 GPUs, no request pauses for more than 50 seconds under Stint and p99 time per output token falls by 50.7\% relative to recomputation in the same engine. The replay as a whole finishes 10.7\% sooner (12.0\% higher throughput), because the requests that baselines starve no longer finish last. 

---
# FRAAC: A French Multimodal Large Language Model and Dataset forPictogram-Based Augmentative and Alternative Communication 

**Authors**: Alexandre Audibert, François Portet, Benoit Favre, Maja Jonck Hjuler  

**Link**: [PDF](https://arxiv.org/pdf/2611.20094)  

**Abstract**: Pictograms are a core modality in augmentative and alternative communication (AAC), yet recent multimodal large language models (MLLMs) cannot reliably interpret or produce pictogram sequences. Two obstacles explain this gap: the absence of a standardized pictogram grammar and the scarcity of parallel text-pictogram data. We introduce Retrieval-Augmented Pictogram Selection (RAPS), an annotation method that refines a rule-based transcription system's output with LLM-based concept extraction and retrieval over a pictogram vocabulary, handling cases where hand-written rules fail (named entities, numbers, polysemous words) without requiring new rules. Applied to French, RAPS yields a synthetic corpus of 1.4 million parallel pairs, of which 300k are publicly released. In a human ranking study where annotators order four pictogram transcriptions of the same sentence, RAPS outperforms the rule-based state of the art. Moreover, we show that standard reference-based metrics separate systems of clearly different quality but fail to rank strong systems against each other. Finally, we build an instruction-tuned model that supports interaction in several AAC modalities: it accepts pictogram or text input and answers in pictograms, speech, or text. 

---
# The Silent Cost of Conversation History in LLM Reasoning 

**Authors**: Ali Şenol  

**Link**: [PDF](https://arxiv.org/pdf/2611.20071)  

**Abstract**: Large language models (LLMs) in multi-turn information systems accumulate their own earlier replies and a growing task state. We asked whether this conversation history lowers accuracy, and whether process-level quality metrics warn before it does. Each analysis was fixed before its outcomes were computed. In Study 1, on 48 sharded math tasks, retaining its replies lowered final success by 0.21 for GPT-4o-mini and 0.29 for Mistral-7B (Holm-adjusted p = 0.026 and 0.004); Llama-4 Scout showed the same direction, not significant. Study 2 introduces CDR-Stream, 100-turn state-tracking dialogues with an exact answer per turn and per-model difficulty calibration. With the full history, accuracy fell from about 0.6 to 0.1 within twenty turns at matched difficulty, whereas a solver-verified state summary kept it 0.52-0.59 higher (Holm-adjusted p less than 0.001). Consistency and logical coherence fell far less than accuracy, and stability rose. None of these metrics, a paired readiness measure or calibrated single-stream drift detectors gave robust, model-general early warning, although in Study 2 the repeat-based metrics were complete only for GPT-4o-mini. In the monitors we tested, a single 100-turn stream had power only for very large drops. Systems built on LLMs should keep task state outside the transcript and monitor across sessions. 

---
# Clinician Assessment of Laryngeal Disease with Auditory AI before Laryngoscopy: A Paired Reader Study 

**Authors**: Siyuan Hou, Shuting Yang, Zengrui Jin, Liyu Cheng, Siyin Wang, Yunhan Shi, Xueyan Li, Rong Hu, Yuhong Lin, Qiyu Chen, Yanru Li, Ji Wu, Bowen Zhou, Mangsuo Zhao, Wen Xu, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.20050)  

**Abstract**: Background: Before laryngoscopy, clinicians assess the likely cause of hoarseness and whether to suspect a precancerous or malignant lesion. Whether voice-based artificial intelligence (AI) supports pre-laryngoscopic categorization is unknown.
Methods: We developed an auditory large language model that generates a patient-level assessment card (provisional category, voice-quality ratings, and rationale) and a prespecified malignancy alert from voice recordings, patient information, and acoustic measures obtained before laryngoscopy at one center. We evaluated both in a held-out test set and a later cohort. In a sequential reader study, 32 otolaryngology clinicians categorized held-out cases without AI and, 1 to 2 weeks later, reviewed their initial categories with the card (without the alert).
Results: Across 800 paired assessments, reader accuracy was 49.8% before and 56.8% after review (difference, 7.0 percentage points; 95% CI, 1.7 to 13.5; P < 0.001); 99 categories changed from incorrect to correct and 43 from correct to incorrect (ratio, 2.30; 95% CI, 1.22 to 5.31). Accuracy rose 14.7 points when the model was correct and fell 8.7 when incorrect. Provisional-category accuracy was 72% in held-out testing and 67% in temporal validation. The malignancy alert flagged 85.7% and 86.8% of precancerous or malignant lesions. Sensitivity for precancerous or malignant lesions increased from 53.9% to 75.0% after AI-assisted review (difference, 21.1 percentage points; 95% CI, 4.3 to 42.0).
Conclusions: In this single-center study, clinicians' pre-laryngoscopic categorization accuracy was higher after reviewing the AI assessment card, although the sequential design cannot fully separate AI assistance from reassessment; gains depended on model correctness without clearly selective adoption. Randomized and external evaluations, including of the malignancy alert, are needed. 

---
# Security of Deep Research Agents: Attack Surface, Defenses, Benchmarks, and Agenda Seeding against Research Planning 

**Authors**: Yang Luo, Guangpu Chen, Yichen Liu, Zhaoyu Chen, Yifeng Gao, Zhiyang Ding, Qingni Shen, Zhonghai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20045)  

**Abstract**: Deep research and agentic search agents autonomously retrieve content from the open web and generate reports with citations, and their outputs are often taken directly as research conclusions. This paper systematizes security research in this area along four stages, namely retrieval, planning, evidence integration, and report generation. We present a unified threat model, a taxonomy of attacks and defenses, and a review of evaluation benchmarks, and we align the area dimension by dimension with retrieval-augmented generation (RAG) poisoning, prompt injection, and generative engine optimization (GEO). Coding 34 primary studies shows that only 26.5% of them involve the planning stage, and none performs an end-to-end poisoning evaluation on commercial products. To address this gap, we propose the agenda seeding attack, in which the attacker writes no instructions and only introduces a fictitious evaluation metric. Across 3,520 sandboxed research runs under the same budget, GEO and prompt injection are nearly ineffective, whereas agenda seeding makes the attacker's target the top pick in 16% to 28% of reports. Injection detection and query anchoring fail against it, verification prompting hurts utility, and our proposed provenance tagging preserves utility but can be bypassed by multi-site bait. Finally, we propose eight research directions. 

---
# acuSimNet: Multi-View Self-Occlusion-Awared Visibility Learning for Cranio-Cervical Acupuncture Points 

**Authors**: Jiatao Ma, Qilei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.20025)  

**Abstract**: The localization of acupuncture points (acupoints) in Traditional Chinese Medicine (TCM) presents unique challenges since they are defined by abstract principles rather than distinct anatomical landmarks. Existing approaches are typically constrained to single view or rely on indirect calculations of relative positions with respect to other landmarks, thereby overlooking human anatomical variations. Furthermore, acupoint visibility assessment, which determins whether points are occluded by human itself, hair, clothing, or other objects, has received limited attention in practical applications. Acupoint localization itself does not require 3D reconstruction, but inferring their occlusion relationships with anatomical surfaces does, which adds computational cost and limits real-time inference. In this work, we introduce acuSimNet, an efficient hierarchical multi-task learning architecture for multi-view, self-occlusion-aware visibility prediction of acupoints, achieving 99.97% accuracy on the validation set. Our approach also addresses the challenges of high-dimensional classification (174 acupoints in cervicocranial region), negative convergence issues for visible acupoints, and inter-task scheduling optimization, resulting in substantially accelerated convergence. We improved the training efficiency from the exisiting methods of 3000 epochs to achieve 99% validation accuracy, to our optimized framework achieving 90% accuracy in 39 epochs and 99% accuracy in 86 epochs. This architecture overcomes the limitations of existing methods, could enable practical applications in acupoints detection and visualization, advancing the automation of TCM. 

---
# Improving fine-grained expressivity of graph neural networks 

**Authors**: Yonatan Sverdlov  

**Link**: [PDF](https://arxiv.org/pdf/2611.20020)  

**Abstract**: This thesis investigates the expressive power of neural networks for structured data, focusing on graph neural networks and symmetry-aware architectures.
We present four main contributions. First, we characterize the generic expressive power of geometric message-passing neural networks on sparse graphs and introduce EGenNet, an architecture achieving the corresponding guarantees. Second, we use representation theory to characterize linear layers equivariant to multiple permutations, including wreath-product symmetries. Third, we study permutation-invariant set representations that preserve containment, establish dimensional bounds and impossibility results, and introduce MASNet. Finally, we propose FSW-GNN, a message-passing architecture combining strong expressivity and bi-Lipschitz stability guarantees with strong empirical performance on long-range graph-learning tasks.
Together, these results provide a broader understanding of how neural architectures can preserve the geometric, algebraic, order-theoretic, and metric structure of their inputs. 

---
# D'BEAST: Logged Bad Experiences as Explanation for Autonomous Driving Agents 

**Authors**: Ahmad Alfan Alfian Irfan, Nur Ahmad Khatim, Mansur Arief  

**Link**: [PDF](https://arxiv.org/pdf/2611.20019)  

**Abstract**: Reinforcement learning agents learn from mistakes, yet their decisions can be difficult to trace back to those experiences. We investigate whether a tabular agent's decisions can be explained by its bad experiences, defined as the training transitions associated with safety or task events. A fixed training log is replayed from the same initial Q-table under three conditions. FULL retains all updates, SKIP omits bad-experience updates, and NEUTRALIZE retains those updates without their event penalties. We illustrate the method on six classic Markov decision processes and evaluate it in closed-loop driving in CARLA. On the full driving task, FULL reached the goal from 33 of 40 test starts, whereas SKIP and NEUTRALIZE reached it from none, indicating that the bad experiences carry decision-relevant information. At the local level, for 47 decisions shortly before a failure of SKIP or NEUTRALIZE, removing or neutralizing a small group of bad experiences, with a median size of one, caused the trained agent to select the failing policy's action in 46 cases. For this agent, the results indicate that its decisions can be explained by its bad experiences. 

---
# ReactBench-AD: Do Driving VLAs and World-Action Models React to Imminent Hazards in Their Path? 

**Authors**: Nikos Theodoridis, Reenu Mohandas, Sushil Sharma, Ciarán Hogan, Ganesh Sistu, Anthony Scanlan, Tim Brophy, Ciarán Eising  

**Link**: [PDF](https://arxiv.org/pdf/2611.20011)  

**Abstract**: Vision-Language-Action models (VLAs) and World-Action Models (WAMs) have emerged in autonomous driving with the promise of strong generalization to long-tail scenarios. Although these models achieve state-of-the-art results on autonomous driving benchmarks, most existing benchmarks are dominated by nominal scenarios and therefore cannot test this promise. To test one such long-tail capability that is critical for safety, we introduce ReactBench-AD, a benchmark that measures how driving models react to objects that appear in the ego vehicle's path. Built on real NAVSIM and PhysicalAI-Autonomous-Vehicles scenes, ReactBench-AD composites a crossing pedestrian, a standing pedestrian, or a stationary car consistently across all camera views and history frames. Each edited scene thus forms an exact, pixel-level counterfactual pair with its original, which allows us to isolate the causal effect of the inserted object on the predicted trajectory. Evaluating four VLAs and two WAMs, we find that they react weakly, often barely at all. Finally, we test how easily one of the evaluated VLAs, ReCogDrive, can be taught to brake hard for imminent hazards: a short fine-tuning stage on edited scenes substantially improves its reactions, with the learned behavior generalizing to unseen objects and fully real scenes, at minimal cost to nominal driving. These results call into question the out-of-the-box long-tail capabilities of these models, but suggest that they can be a strong foundation for acquiring them with little additional training, highlighting the value of real or synthetic data that demonstrates the long-tail behaviors nominal datasets rarely contain. Code and data will be made publicly available. 

---
# The novelty exposed: ablation, reproducibility and transparency in 40 technical reports of open-weight large language models (LLMs) 

**Authors**: Bahaeddin Türkoğlu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20003)  

**Abstract**: In 2025 alone, 48 new large language model (LLM) architectures were added to the Hugging Face transformers library, and many of them arrive with a technical report that presents the model as new. What is actually new in these models, and is it tested? We put this question to LLM technical reports. We look at three things: what the reference code shows about how new open-weight architectures differ from earlier ones; what 40 randomly sampled technical reports claim as new and whether they support it with ablation studies; and what they disclose for reproducibility and transparency. In the library, 39 of the 48 architectures added in 2025 take at least two of their three core blocks from another model, most often Llama, although part of this reflects how the library writes code. In the reports, 19 of 73 architectural or training novelty claims come with a controlled with/without comparison; 4 of 40 reports state more than one training run; 8 rerun the models they compare against under one setup; and 1 discloses data mixture, token count, and compute together. We also looked for known components given new names: none of five candidate cases hid its origin, though we did not study model names. We propose seven reporting rules for LLM release reports and apply them to two reports. We do not judge model quality, and an untested claim may still be true. 

---
# What Matters for Cross-Modal Alignment 

**Authors**: Yipeng Sun, Jiajun Wang, Siyuan Mei, Yan Xia, Siming Bayer, Vincent Christlein, Andreas Maier  

**Link**: [PDF](https://arxiv.org/pdf/2611.20000)  

**Abstract**: The Platonic Representation Hypothesis posits that models trained on different modalities converge toward a shared representation. Related work aligns frozen image and text encoders on natural images without paired data. We test zero-pair alignment on five datasets across four modalities: natural images, environmental audio, chest radiographs, and 3D computed tomography (CT). It succeeds only on COCO and fails on the other four. We analyze the failures and find that alignment must first locate the subspace where the modalities co-vary and then orient the mapping within it. Consistent with previous studies, medical imaging is the hardest case: its shared subspace carries the smallest fraction of the variance. When this fraction is small, alignment needs image-text pairs, and random-matrix theory predicts how many pairs are needed. These findings motivate LightAlign: a few pairs let a ridge regression between two frozen encoders locate and orient the mapping simultaneously, and a classifier fitted only on report text then predicts findings from the mapped images. On 3D CT, LightAlign surpasses models trained end to end on image-report pairs without finding-level labels: with only 200 pairs, LightAlign outperforms CT-CLIP on CT-RATE (0.754 vs. 0.733 AUROC), and with only 100 CT-RATE pairs, it does so on RAD-ChestCT, an out-of-distribution dataset from a different hospital. On CheXpert, it reaches 0.862 mean AUROC with only about 5% as many pairs as models trained on MIMIC-CXR. 

---
# Morphology-Decoupled Segmentation for Multiple Tunnel Defects Under Complex Environments 

**Authors**: Mohan Ren, Hang Li, Yifei Liu, Shaowei Wang, James O'Kane, Xuanzhi Wang, Shiyu Zhao, Yumeng Zhang, Songtao Ji, Jurij Karlovsek  

**Link**: [PDF](https://arxiv.org/pdf/2611.19981)  

**Abstract**: Automated tunnel-lining inspection must segment different defect geometries: cracks are thin, elongated, and connectivity-sensitive, whereas leakage, lining falling off, and segment damage are spatially extensive and context-dependent. Existing approaches generally apply a common decoder across defect classes or introduce boundary, topology, and directional mechanisms as isolated enhancements. This paper introduces Morphology-Decoupled Segmentation (MDS), in which defect morphology becomes the organising principle for feature routing, decoding, and interaction. A supervised routing module partitions multiscale encoder features into complementary linear and areal streams. The linear stream employs Deformable Strip Attention with image-conditioned direction hypotheses to preserve crack continuity without assuming fixed horizontal or vertical orientations, while the areal stream combines Pyramid Pooling with feature-pyramid fusion to capture diffuse boundaries and regional context. The representations are coupled through gated cross-morphology attention, allowing complementary information exchange while retaining morphology-specific features. A resolution-adaptive tiling and probability-blending pipeline preserves fine defects across variable-resolution imagery. MDS achieves 81.17% mIoU on the internal validation set and improves foreground image-level recognition over a backbone-matched SegDINO model, primarily for areal defects. Controlled ablations demonstrate the complementary contributions of routing supervision and cross-stream interaction while revealing a trade-off between areal-class performance and crack specialisation. The submitted model achieved 67.55% mIoU on an organiser-controlled hidden test, and the team placed second. Selective dynamic INT8 quantisation reduced model storage by 2.83 times with negligible accuracy loss. 

---
# CreaTrace: Video Creative Plagiarism Detection and Evidence Tracing 

**Authors**: Chang Liu, Sheng Li, Zhenxing Qian, Xinpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19976)  

**Abstract**: Video creative plagiarism refers to the unauthorized reproduction of creative ideas across videos despite substantial differences in visual appearance, posing a new challenge to video copyright protection. However, existing video copy detection methods primarily focus on visual content similarity, struggling to capture abstract creative that are entangled with specific visual realizations. To address this issue, we introduce a novel task, named Video Creative Plagiarism Detection (VCPD), and propose an iterative creative evidence tracing framework, termed CreaTrace, to verify creative similarity and accurately localize the corresponding evidence segments. Specifically, we design a Multimodal Semantic Event Segmentation strategy to divide similar long videos into coherent narrative events, facilitating event-level dataset construction and retrieving similar segments. Based on these events, CreaTrace employs Coarse Creative Matching to identify candidate event pairs through narrative semantics and camera language. Then CreaTrace presents Creative Decoupling to disentangle abstract creative patterns from specific visual realizations. Meanwhile, Creative Swap Contrastive Learning is devised to enhance creative consistency through cross-realization feature swapping. Benefiting from the above designs, our CreaTrace enables accurate creative similarity detection and temporal localization, while providing interpretable evidence report. Extensive experiments demonstrate the effectiveness and robustness of CreaTrace across diverse creative plagiarism scenarios. 

---
# Federated Dataset Simulation: Inducing Label-Free Heterogeneity Across Tasks 

**Authors**: Vasilis Siomos, Lam Ngo, Jonathan Passerat-Palmbach, Giacomo Tarroni  

**Link**: [PDF](https://arxiv.org/pdf/2611.19975)  

**Abstract**: Federated Learning enables collaborative model training across decentralized clients without sharing raw data. While the field has seen rapid growth, the datasets underpinning FL evaluation have not kept pace: real federated datasets remain scarce, and most studies rely on small-scale centralized classification datasets partitioned via Dirichlet label skew, a one-dimensional proxy for the multi-dimensional heterogeneity present in real federations. Moreover, label-based partitioning is by construction inapplicable to non-classification tasks such as segmentation and detection, for which no principled simulation method exists. We cast the conversion of centralized datasets into federated benchmarks as the task of Federated Dataset Simulation (FDS), and decompose it into a representation function $\Phi$ and an assignment mechanism $\pi$. We provide two label-free instantiations -- geometric partitioning via pretrained model embeddings and semantic partitioning via vision-language model-derived criteria -- applicable to classification and dense prediction tasks alike. We evaluate across four datasets spanning classification, segmentation, and detection, and show that both methods induce heterogeneity that matches or exceeds that of label-skew partitioning, with measurable impact on downstream FL performance. We release fds, a modular open-source package for end-to-end federated dataset simulation, heterogeneity measurement, and FL training, at this https URL. 

---
# Value and Policy Calibration from Execution Feedback for Socially Compatible Driving 

**Authors**: Sen Shen, Ning Liu, Xiangrui Kong, Yingwei Li, Dongkun Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.19969)  

**Abstract**: Autonomous vehicle (AV) decisions at intersections affect the delay, braking, and motion smoothness of surrounding human-driven vehicles (HDVs). Learning socially compatible behavior requires connecting imagined interaction outcomes with feedback from executed actions. We propose a world-model reinforcement learning framework combines imagined policy improvement with direct value and policy calibration from finalized interactions, balancing AV progress with HDV interaction burden. An interaction-aware latent model supports imagination, while off-policy correction and replay updates enable observed rewards and transitions to supervise both the critic and actor. Simulation experiments at an unsignalized intersection achieve a 100% success and reductions in following-HDV acceleration variation of 65.5% and 44.0% relative to PIWM and DreamerV3, respectively. Ablations identify complementary roles for value calibration in reducing HDV burden and policy calibration in improving passage efficiency. Tests under out-of-distribution traffic shifts further show lower ego jerk and HDV acceleration variation than PIWM at comparable passage times. 

---
# Equal Dose, Different Coverage:Tracing Rate and Duration in Image Editing 

**Authors**: Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19955)  

**Abstract**: Benchmarks for scientific image editing test whether a model draws the outcome of a described process, leaving open which representations carry its errors and whether correcting them yields the right quantity. We study this question with an airbrush of fixed deposition profile, under which exchanging rate and duration preserves the dose and should preserve the endpoint, yet changes the edited coverage. Matched-norm activation patches across text positions and denoising stages show that, in distilled FLUX with Chinese prompts, representations at unchanged text positions (context) carry this discrepancy: early context transfers 3.51 of the 3.98 percentage-point (pp) signed route difference, against 0.09 pp for equal-norm updates at the two changed digits. Early image latents transmit native (unpatched) route differences across two FLUX configurations and OmniGen, while text-position effects depend on configuration. The trace predicts an intervention: share early context between routes while each retains its own digits. On independent English layouts, this reduces the distilled-FLUX absolute route gap from 1.13 to 0.60 pp, outperforms early digit sharing, and improves agreement more than in base FLUX under directly paired tests. Calibrated evaluation, which prescribes an area for each dose, separates this gain from accuracy: sharing the full prompt encoding produces identical route images with target errors of 8-16 pp, and stating the target percentage directly retains a compressed area response. Localization predicts a consistency gain on new layouts, and calibration measures the quantity error that agreement leaves unresolved. 

---
# Learning When to Continue and When to Stop: Termination-Aware OPSD 

**Authors**: Chaoqun He, Wannian Yi, Changyi Sun, Chaojun Xiao, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2611.19925)  

**Abstract**: On-Policy Self-Distillation (OPSD) improves language model reasoning by distilling from a frozen teacher initialized identically to the student and conditioned on privileged information. However, we identify an underexplored termination mismatch in OPSD: the student fails to reliably learn when to continue and when to stop. This mismatch manifests in two opposing failure modes-premature termination, where reasoning stops before sufficient progress is made, and repetition-driven over-generation, where the model continues generating after reasoning has stalled. Our analysis shows that the teacher-student discrepancy on EOS grows during OPSD training, while length-capped trajectories are frequently dominated by unproductive repetition. To address this issue, we propose TA-OPSD (Termination-Aware On-Policy Self-Distillation), which augments the original OPSD objective with two targeted objectives. Stop/Continue Distillation addresses premature termination by explicitly aligning the teacher and student on whether reasoning should continue, while Repetition-Aware Unlikelihood Training addresses repetition-driven over-generation by discouraging repetitive continuations in the student's failed trajectories. The resulting objective improves termination behavior from complementary directions: continuing when further reasoning is needed and stopping when reasoning becomes unproductive. Experiments on mathematical reasoning tasks with Qwen3 (1.7B/4B/8B) show that TA-OPSD consistently outperforms OPSD and other baselines across all three model scales. Beyond the performance gains, ablation and behavioral analyses show that the two proposed objectives effectively mitigate their corresponding termination failures, reducing both premature termination and repetition-driven over-generation. 

---
# The Vocabulary Tax: Safely Trimming Unused Tokens from LLMs for On-Device Use 

**Authors**: Sattaya Singkul  

**Link**: [PDF](https://arxiv.org/pdf/2611.19913)  

**Abstract**: Open large language models (LLMs) and speech models ship vocabularies of up to 262k tokens, and every user downloads the tables those tokens index (the input embedding, an untied output head and, in recent on-device models, per-layer embeddings), whatever languages they need. We call the bytes in rows a deployment never uses the vocabulary tax and measure it, without downloading weights, for 1,244 generative and speech-recognition models on the Hugging Face Hub. For a Thai and English deployment it is a median 4.4% of model bytes below 0.5B parameters and 2.1% above 8B, and 1.93 GB in Gemma 4 E2B: it falls hardest on the small models that edge devices run. Keeping only the rows of the deployment's writing systems is exact. Keeping the tokens seen in the user's own text removes much more, a median 29.0% below 0.5B, but the usual way of doing it breaks BPE tokenizers: it drops the intermediate tokens that merges build on, so new text is split differently 1.9 times as often as the standard check reports. We repair this with merge closure, at a cost of 5.9% more rows, and add a trim certificate: a Good-Turing estimate, from the trimming text alone, of how much new text the trimmed tokenizer will split differently (correlation 0.996 over 300 settings in 22 languages) and of how much more text a target needs. Without test data our method never exceeds a 2% target, while prior keep rules checked on test data exceed it in 9-13 of 44 settings. On six models of 270M to 3B parameters in 11 languages, trimmed models keep their predictions: where the certificate passes, a median 89.7% of rows is removed and bits per character rise by at most 2.0%. The census, the trimming and the certificate are released as an open Python package, and the per-model tax for 41 languages as an open dataset. 

---
# AgenticJevNav: Agentic Object Search and Navigation via Event-Driven Verification 

**Authors**: Shuning Zhang, Shuli Zhang, Yunheng Wang, Wei Lu, Jiaxi Zhang, Yihang Kang, Tao Wang, Xiangyu Chen, Yixiao Feng, Kai Sheng, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19912)  

**Abstract**: Agentic navigation must maintain goals and adapt actions from incomplete evidence despite execution failures and delayed inference. We present AgenticJevNav, which links goal-directed planning, evidence memory, event-triggered verification, and bounded execution. The Agent plans from RGB, odometry, memory, and feedback, retaining goal ownership while authorizing a bounded action set. Jev reviews structured evidence after blocked/failed execution or two consecutive low-progress segments. Version, freshness, and once-only checks govern decision application. A single executor and independent RGB reflex constrain motion. Goal revisions revoke obsolete requests and actions before replanning. Across six simulated task families, AgenticJevNav achieves 73.33% SR over 60 trials, 21.67 percentage points above the next-best Agent-only aggregate control under differing historical protocols. A real-world GENISOM L1 quadruped deployment achieves 70.0% success over 30 trials, including bounded search reporting. Together, the simulation and physical-robot results demonstrate the effectiveness of the proposed architecture, while remaining failures highlight opportunities to improve grounding, approach planning, and runtime. 

---
# Remember, Replay, Adapt: Learning from Past Experience in Online Vision-and-Language Navigation 

**Authors**: Chenhao Ding, Jizhou Han, Lixin Xiu, Muyao Wang, Songlin Dong, Hideki Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2611.19901)  

**Abstract**: Existing feedback-based methods for online vision-and-language navigation (VLN) primarily learn from current episode outcomes, leaving previously acquired feedback underutilized. Inspired by human episodic memory, we propose R2A, an online VLN adaptation framework that continually reuses successful and failed navigation experiences for policy learning. Specifically, Success-Failure Memory (SFM) retains successful and failed navigation episodes with acquired feedback and retrieves task-relevant experiences based on incoming instructions, enabling historical feedback to guide subsequent navigation. Prior-Anchored Adaptation (PAA) learns from these experiences by reinforcing successful trajectories and suppressing failed ones relative to a frozen pretrained policy, while regularizing behavioral drift. Together, these modules enable continual feedback reuse and stable policy adaptation without source-domain data or expert supervision. Extensive experiments across three benchmarks, REVERIE, R2R, and R2R-CE, and five navigation backbones demonstrate consistent performance gains. On the REVERIE validation unseen split, R2A improves DUET's SR from 46.98% to 69.84% and SPL from 33.73% to 50.98%. 

---
# Recovery Gone Too Far: Understanding and Repairing Risky Fallback in LLM Agents 

**Authors**: Liwen Wang, Jiangrong Wu, Zongjie Li, Zongyi Lyu, Shuai Wang, Shing-Chi Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2611.19886)  

**Abstract**: Large Language Model (LLM) agents interact with their environments through tools, reading repositories, running builds, and pushing code. When a tool fails with an ordinary error, such as a rate limit or a timeout, the agent can fall back to a substitute tool and still complete the task. However, the same fallback that completes a task in one environment can cross an action boundary in another. For example, after a push is rejected, a forced push in one repository replaces only the agent's own commits, while in a different repository it also removes another author's commits. We call such a fallback a risky fallback. To understand this phenomenon, we conduct an empirical study on 304 scenarios that span eight domains and cover 55 real CVEs. All four mainstream models we evaluate exhibit risky fallback, with action-boundary violations in up to 96.8% of risky scenarios. We further evaluate three mainstream mitigations and find that none of them is effective against risky fallback. We then present nrepair. At each call to the substitute tool, nrepair scores the agent's context with a monitor whose threshold is calibrated on normal runs alone; when the monitor triggers, it withholds the call once and returns the decision to the agent. nrepair reduces action-boundary violations on all four models, by up to 76.0% (from 67.5% to 16.2%), at a negligible runtime cost. On most models, the normal success rate drops by less than 7%. We also evaluate nrepair on two production agents, Claude Code and Codex, and find that it is equally effective on both. 

---
# Large Language Model-based Optimization for Wireless Communication: A Survey 

**Authors**: Muhammad Sohaib, Taehyung Kim, Hyeri Heo, Jemu Ryu, Chaehyun Lee, Jun Zhang, Sang-Woon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2611.19876)  

**Abstract**: The growing complexity and dynamism of modern wireless communication frameworks have made real-time optimization essential for efficient resource allocation. Traditional optimization and machine-learning techniques often struggle with generalization, adaptability, stability, and scalability in heterogeneous, time-varying environments. Large language models (LLMs) have recently emerged as powerful tools capable of reasoning over structured prompts and generating context-aware decisions in real time without domain-specific retraining. This survey provides a comprehensive review of LLM-driven optimization techniques applied to wireless communication problems such as power control, bandwidth allocation, task offloading, and intelligent surface configuration. We introduce a functional taxonomy of LLM-enabled wireless resource management strategies and analyze the prompt engineering methodologies employed in recent literature. Furthermore, we develop a generalized analytical framework that categorizes LLM-based optimization architectures, including direct solution generation, closed-loop optimization with external verifiers, and representation encoding within trainable neural pipelines. To demonstrate practical applicability, we present representative case studies on LLM-assisted beam prediction for millimeter-wave systems and adaptive wireless resource allocation. Finally, we discuss emerging hybrid AI approaches for wireless networks. 

---
# Let Execution Do the Teaching 

**Authors**: Abhishek Mishra, Sandeep Srinivasa, Arko Chattopadhyay, Enrique Martinho Ferrao, Guda Pranav Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2611.19875)  

**Abstract**: Improving a coding model is a sequence of decisions: find the limiting behavior, create useful practice, and determine which attempts deserve reinforcement. We investigate a practical way to make those decisions increasingly repeatable by grounding feedback in execution. Starting from GLM-4.7-Flash, the work combines whole-file supervised fine-tuning, targeted C++ repair curricula, group-relative reinforcement learning, and verifiers that turn program behavior into structured rewards. Compiler errors and failing tests provide specific contexts for repair practice, while executable checks score the resulting programs. We evaluate the trained models on 26 C++ coding tasks, measuring both initial success and completion after test feedback. The strongest reported post-training configuration achieves 45.2% success in the first test round and 61.5% within two rounds. These results come from an assisted development suite with evolving prompt contracts; the strongest configuration uses four selected evaluation trials and includes six tasks also used in training. The study describes an execution-guided workflow for improving coding models, including how repair tasks are constructed, how program behavior determines rewards, and how the resulting models are evaluated. 

---
# HalluProbe: Exploiting VLM Hallucinations for Backdoor Detection 

**Authors**: Ji Guo, Xiaolong Qin, Jin Jiang, Yijing Liu, Pei Ke, Aiguo Chen, Wenbo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19868)  

**Abstract**: Vision-Language Models (VLMs) are vulnerable to backdoor attacks introduced through data poisoning, making effective backdoor detection a critical security challenge. In this paper, we uncover a novel connection between VLM hallucinations and backdoor behaviors and exploit this connection for backdoor detection. Our key insight is that poisoned training data induces strong language priors toward the backdoor target token, while such priors are also a major source of hallucinations. This suggests that enhancing hallucinations can expose hidden backdoor-related priors. Specifically, backdoored VLMs tend to hallucinate backdoor target tokens more frequently across diverse inputs, while the hallucinated tokens of clean VLMs are largely random and exhibit no stable target-specific bias. Based on this observation, we propose HalluProbe, which enhances hallucinations during output decoding and measures the distributional consistency of hallucinated tokens to identify backdoored VLMs. We evaluate HalluProbe against six representative VLM backdoor attacks across three models. Experimental results demonstrate that HalluProbe can effectively detect backdoored VLMs with an accuracy exceeding 90%, outperforming existing backdoor detection methods. 

---
# Local Order Matters: A Lightweight CPU-Trainable CNN for Website Fingerprinting on Defended Tor Traffic 

**Authors**: Puyang Xu, Zhizhao Wen, Yifan Jiang, Chao Wang, Yusi Gong  

**Link**: [PDF](https://arxiv.org/pdf/2611.19864)  

**Abstract**: Website fingerprinting (WF) attacks let a passive adversary infer which site a Tor user visits from the encrypted cell sequence. On undefended traffic, hand-crafted-feature attacks are already near-ceiling, so deep learning offers only a marginal advantage there. We show that the decisive regime is defended traffic: against the FRONT padding defense, which injects dummy cells that corrupt the volume and count features feature attacks rely on, a compact 95,103-parameter 1-D convolutional network trained on the raw cell-direction sequence, on CPU only, stays robust while three implemented direction-only feature baselines degrade. On the closed-world WFlib FRONT set (95 sites), our model reaches 54.2% top-1 accuracy versus 38.5% for the strongest implemented feature baseline (k-FP-style RF), a gain of 15.7 percentage points (3 seeds; seed-0 paired-bootstrap 95% CI [+14.5, +17.0] pp on per-trace correctness, excluding zero). On undefended traffic the CNN reaches 94.1% versus 92.3% for k-FP-style RF. A controlled order-ablation permutes only nonzero traffic cells within each trace, preserving direction counts, retained length, and trailing zero-padding positions while destroying traffic order. This collapses the CNN below every feature baseline (from 54.2% to 7.6%, a 46.6 pp drop), showing that the model depends strongly on order-derived information under FRONT. 

---
# Bridging the Terminology Gap: A Data-Centric Alignment Framework for Multilingual Translation in Cross-Border Corporation 

**Authors**: Jinsheng Wei, Song Cheng, Bo Xiao, Ke Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2611.19858)  

**Abstract**: As globalization accelerates the demand for seamless cross-border enterprise collaboration, multilingual translation remains bottlenecked by insufficient domain-specific terminology, rigid terminology translation, weak instruction-following capabilities, and high inference latency in long-instruction scenarios. To overcome these barriers, we propose Bridging the Terminology Gap (BTG), a data-centric framework. First, BTG mines cross-lingual term alignments from domain documents and leverages LLMs to synthesize terminology-rich parallel corpora. Second, a hybrid filtering strategy combining heuristics and XCOMET scoring ensures high data fidelity. Third, a two-stage training paradigm employs supervised fine-tuning (SFT) for targeted terminology injection, followed by reinforcement learning (RL) to refine output preferences, effectively balancing domain knowledge integration with model reliability. Extensive offline evaluations (e.g., XCOMET, GEMBA, Acc@T) and online human assessments demonstrate significant improvements in terminology accuracy and instruction-following capabilities, validating the effectiveness of our framework, which is now deployed on our cross-border collaboration platform. 

---
# EfficientSkill: Learning Skills for Token-Efficient Multi-Turn Agents 

**Authors**: Deep Shah, Sanket Badhe, Nehal Kathrotia, Priyanka Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2611.19823)  

**Abstract**: Reflective skill optimizers improve multi-turn LLM agents by revising instructions from failed and successful rollouts, yet accumulating edge-case rules bloats prompts, inflates token use, and pushes agents into long, error-prone loops. We introduce EfficientSkill, a two-step framework that first discovers a reliable skill without token penalties, and then warm-starts accuracy-guarded pruning from that skill to compress instructions, bound agent turns, and offload reusable code into workspace utilities with zero implementation prompt cost. Across four frontier, mid-tier, and open-weight models on MATH-500, ALFWorld, and SpreadsheetBench, EfficientSkill cuts output tokens over unconstrained EvoSkill on 11 of 12 configurations (by up to 80%) while raising mean accuracy on every benchmark, including +14.3 and +17.5 percentage-point gains on SpreadsheetBench with Gemini 3.8 Flash and Grok 4.1 Fast by replacing stalled loops with compact workflows. By shrinking both generated tokens and replayed history, EfficientSkill lowers total token usage and serving cost across every evaluated configuration, with or without full-prefix KV caching. 

---
# The Flip Test: Video World Models Fail to Learn the World State 

**Authors**: Mi Luo, Lijun Yu, Adams Wei Yu, Ang Cao, Baochen Sun, Ming-Hsuan Yang, Nan Hua, Yuan Hao, Alex Dimakis, Ya Xu, Ed H. Chi, Xuezhi Wang, Yinxiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19822)  

**Abstract**: Video world models promise to predict the future of a scene from its present. Do current models keep this promise? We probe this question counterfactually. When a small change to the present flips the future, such as a box resting a few millimeters before or past a table edge, a model that tracks the state of the world must flip its prediction too. We build FlipTest, a three-level hierarchy of increasingly strict tests over 1,680 initial frames from 10 simulated and 15 real-world tasks. At the base, visual minimal pairs, two nearly identical frames whose true futures differ, test whether predictions change across the boundary; matched controls go further and test whether they change only where the future does; and dense sweeps, the strictest level, locate exactly where they change. All 11 video world models we evaluate fall short. The best averages 18.3% pair accuracy over real and simulated scenes, below the 25% of random guessing. Models flip as often on steps that leave the future unchanged as on steps that change it, and their response stays flat across the sweeps. On the same frames, a vision-language model is selective and traces a near S-shaped curve near the boundary, and a coding agent that measures the image comes close to the true boundary. This implies that the deciding information is in the frame, and video models just miss it. We therefore urge the community to rethink world modeling as pure video prediction. Next-frame prediction must do more than draw plausible pixels; it must also identify the causal states that control future dynamics. 

---
# GraphPass: A Study of Graph Neural Networks for Compiler Pass Selection 

**Authors**: Mohammed Saqlain  

**Link**: [PDF](https://arxiv.org/pdf/2611.19821)  

**Abstract**: Compiler pass selection is usually governed by fixed pipelines (-O2, -O3) or hand-written heuristics. We study whether a graph neural network (GNN) over a program's control-flow graph (CFG) can select among a fixed menu of five pass sequences in a single forward pass. GraphPass encodes each CFG with a graph attention (GAT) or graph convolutional (GCN) encoder and classifies it into one of the five sequences. On a synthetic benchmark of 5,000 generated CFGs in five program families, both encoders agree with the benchmark's reference label on about 92% of held-out programs, versus 68.4% for a hand-written rule-based heuristic and 85% for a flat MLP. An edge ablation indicates that this gain comes from per-block processing rather than from CFG edges: a GCN given the same basic-block features with all edges removed scores 95.0%, versus 92.2% with the true CFG. In this benchmark the label is determined by the program family, so the task amounts to family recognition. Real-compiler speedup is not measured. Inference costs O(|V| + |E|) per CFG, measured at 1.7 ms for a typical program and about 0.8 s for a 100,000-block graph on one CPU core. 

---
# Reasoning in the loop: antibody lead optimization by probing mechanism and measurement 

**Authors**: Samuel Stanton, Aaron Solomon, Emily Delaney, Melanie Montgomery, Crystal Humphries, Timothy Acker, Ryan Henrici, Charbel Eid, Eddie Abrams, Peyton Greenside  

**Link**: [PDF](https://arxiv.org/pdf/2611.19812)  

**Abstract**: Lab-in-the-loop systems, in which a model proposes molecules, automated assays measure their properties and the results retrain the model, are a popular approach to antibody lead optimization. These systems optimize for a specification defined with metrics computed from instrument signals, so their success depends on how well those metrics serve as proxies for the properties a drug candidate needs. When a metric is a poor proxy, neither a better model nor more data will fix it, and identifying the problem usually requires expert review. We report two campaigns run on BigHat Biosciences' Milliner platform, one on an anti-albumin nanobody (VHH) and one on an anti-mesothelin single-chain variable fragment (scFv), in which a research team and a general-purpose AI agent, Claude, brought scientific reasoning into the loop. Over seven rounds and about 1,010 design variants we built libraries from antibody lineage, structure and biophysics, and found from the raw curves that on each scaffold one thermal metric did not measure what its name implied. The libraries were built as controlled comparisons, so the results also show what each mutation contributes to each property. We fit a physical model of the curves, used it to choose metrics suited to each scaffold, and changed the experimental protocol, varying protein concentration and heating rate, to test the model against pre-registered predictions. Both leads improve on their parents in affinity, thermal stability and aggregation while improving or maintaining yield and purity. 

---
# Eliciting Research Taste in LLMs through Future Research Direction Choice 

**Authors**: Dhruv Trehan, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2611.19794)  

**Abstract**: Research taste is increasingly discussed in the context of AI scientists, yet remains little studied or measured in LLMs. This judgment matters especially in open-ended research, where a direction must be chosen before a verifiable objective is available. We operationalize research taste through one task of the scientific workflow: choosing which future research direction to pursue given a research goal and current results. Rather than predicting downstream scientometric metrics or treating author choices as objectively correct, we evaluate models against human-authored future-work directions and report author alignment, stable divergence, and unstable choice as behavioral measures of research-direction preference. We construct an order-controlled evaluation of 80 pairwise comparisons from 19 ML papers published between December 2025 and February 2026 across four subdomains. Each query presents a research goal, hypotheses tested with key results, and two candidate future directions: one author-recommended direction and one plausible distractor sampled from a similar paper. Across eight LLMs, author alignment ranges from 45.0% to 67.5%, while stable divergence ranges from 15.0% to 36.2%. These divergences largely persist under similarity controls, prompt ablations, and an author-preference framing check, and inter-model agreement suggests that some non-author preferences are shared across models. We characterize what models diverge toward using transition-category labels and selected reasoning examples, finding recurring movement toward scaling/generalization and trade-offs between methodological and infrastructure/data extensions. We view this as an early descriptive evaluation for eliciting research taste profiles in LLMs, and as a step toward understanding and steering their research preferences as they take on more autonomous roles in novel open-ended scientific discovery. 

---
# Learning Autoregressive Neural Operators on Solver-Defined Physical Constraint Manifolds 

**Authors**: Nannan Zhou, Guanmeng Xian, Ning Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19753)  

**Abstract**: Autoregressive neural operators approximate time-dependent PDE solvers by repeatedly composing a learned transition, although the resulting states need not remain feasible under the discrete constraints of the numerical problem. We study this issue for incompressible flow, where each velocity state must satisfy solver-level incompressibility together with prescribed boundary values. We formulate autoregressive operator learning as a map between solver-defined affine feasible sets and introduce PCM-NO, a projected-residual realization of this feasible-transition principle. The neural backbone predicts a candidate state increment, a fixed tangent projection extracts its admissible component, and a metric retraction maps the advanced state to the prescribed feasible set. With exact affine constraint solves, the resulting transition is closed on the feasible set, while an affine identity separates admissible-update parameterization from state-level closure. With approximate solves, tangent-only residuals admit a cumulative upper bound, whereas per-step retraction resets feasibility to the residual achieved by the current solve. Experiments on periodic Navier--Stokes flow and lid-driven cavity flow demonstrate the superiority of PCM-NO in preserving solver-level constraints over long autoregressive rollouts and under parameter, boundary-value, and resolution shifts, while achieving lower rollout errors and better flow-specific diagnostics across both in-distribution and stress settings. 

---
# StaleBench: Isolating Stale Retrieval in Agent Memory, and Pricing the Fix 

**Authors**: S M Mahbub Murshed  

**Link**: [PDF](https://arxiv.org/pdf/2611.19752)  

**Abstract**: An agent that stores facts over time eventually stores several versions of the same fact. Asked a value-agnostic question - "which team does X play for now?" - a similarity retriever cannot prefer the current version, because every version matches the entity the query names. StaleBench isolates this stale-retrieval failure on 892 real updating facts, with currency labels from the source (Wikidata validity years via TempLAMA), never from a retrieval score. Our central result is that retrieval metrics can rank memory policies backwards: a retention policy and its retain-everything control return the identical statement at rank 1 for every test query, yet differ by 29.5 and 65.6 points in reader accuracy on two readers when no date is shown. Contamination@5, the metric we add, ranks them correctly where current@1 and current@5 do not. Withdrawing the lexical anchor that statements carry costs deployed keying mechanisms tens of points of freshness; entity resolution on both the write path and the query recovers most of the loss. Three deployed systems (Mem0, Graphiti, LangMem) and a reimplemented ERASE each handle supersession differently, and no single scoring convention describes all four. We release the data, a frozen evaluator, and reference baselines. 

---
# How Far Is Too Far? Object Recognition Declines Monotonically with Semantic Distance 

**Authors**: Domenic Bersch, Hai Van Tran, Gemma Roig  

**Link**: [PDF](https://arxiv.org/pdf/2611.19747)  

**Abstract**: Vision models are sensitive to object-background context, yet it remains unclear how recognition changes as the semantic distance between objects and backgrounds increases. We introduce ImageNet-OOC1k, a benchmark enabling graded control of object-background relationships along a continuous consensus visual-semantic dissimilarity axis. This formulation reveals a highly consistent monotonic decline in recognition accuracy across 35 models spanning convolutional neural networks, Vision Transformers, and vision-language models. Internal analyses of Vision Transformers reveal patterns consistent with aggregation failure, where object evidence is present at the patch level but not consolidated into the final prediction. Using this characterization as a diagnostic probe, a simple spatial re-ranking strategy recovers a subset of these errors and improves in-sample accuracy by up to 5.7 percentage points. These results establish context sensitivity as a continuous and predictable property of modern vision systems, with Vision Transformer analyses suggesting that aggregation limitations may contribute to some failure types even when object evidence remains available. 

---
# Exploration of Reinforcement Learning in Generative Recommendation Models 

**Authors**: Pengfei Zheng, Wuchao Li, Shiyao Wang, Honghui Bao, Hongyi Xie, Jiaxin Deng, Xingmei Wang, Jinghao Zhang, Yufei Ye, Ruiming Tang, Guorui Zhou, Han Li, Kun Gai, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2611.19742)  

**Abstract**: Traditional cascaded recommender systems balance accuracy and efficiency over large item spaces, but misaligned stage-wise objectives hinder end-to-end optimization. Generative recommendation unifies these stages in an autoregressive framework, providing a natural interface for reinforcement learning (RL) to optimize the entire generation process. However, directly applying RL often leads to unstable optimization and degraded recommendation quality.
We revisit Group Relative Policy Optimization (GRPO) and other ratio-based objectives through a unified logit-gradient view. We identify three issues: (i) clipped surrogates create dead-gradient regions, under-utilizing samples; (ii) negative-advantage updates can become effectively unbounded as importance ratios grow, allowing a small subset of negatives to dominate optimization; and (iii) ratio-based control lacks sensitivity to absolute policy confidence, so updates on high-confidence cases may fail to vanish.
Motivated by this diagnosis, we propose \emph{Gradient-Bounded Policy Optimization} (GBPO), a simple objective with stop-gradient normalization that is simultaneously (a) fully gradient-active, (b) sign-symmetrically bounded for both positive and negative updates, and (c) explicitly confidence-weighted under current policy probabilities. This ensures that updates gracefully vanish for $\pi\!\to\!1$ positives and $\pi\!\to\!0$ negatives. Across the evaluated settings, GBPO achieves competitive recommendation performance with more stable training than competing RL methods. 

---
# An Interpretable Acoustic Coordinate System for Japanese Consonant Morae in Dysarthric Speech 

**Authors**: Yuma Enomoto, Misa Yamaguchi, Hiroki Nishikawa, Ittetsu Taniguchi, Kazuhide Matsunaga, Narikazu Uzawa, Takao Onoye  

**Link**: [PDF](https://arxiv.org/pdf/2611.19714)  

**Abstract**: Dysarthria reduces speech intelligibility, requiring self-directed rehabilitation based on fine-grained feedback to capture gradual articulatory changes. For Japanese speakers, such feedback is naturally provided in mora units due to its distinct moraic timing and consonant-vowel structure. While recent visualization methods have explored distance-based feedback, most are limited to vowels. Extending them to consonants is challenging, as generative embeddings often smooth transient spectral structures. We propose a visualization system for Japanese consonant morae that maps articulatory deviations onto a two-dimensional space defined by temporal and spectral distances from healthy reference speech. Experiments show consistent geometric separation of consonant categories, with Silhouette Coefficients exceeding 0.5 and origin-based ROC-AUC scores above 0.86. Furthermore, evaluations using real dysarthric speech achieved a high ROC-AUC of 0.897 in screening for unclear pronunciations. Perturbation tests confirm orthogonal temporal and spectral shifts, yielding a physically interpretable mapping for rehabilitation. 

---
# Joint Learning of Factual and Counterfactual Worlds for Causal Prior-data Fitted Networks 

**Authors**: Furkan Danisman, Luke Zhang, Zhijing Jin, Dehan Kong  

**Link**: [PDF](https://arxiv.org/pdf/2611.19710)  

**Abstract**: Estimating treatment effects is central to science across many disciplines and requires counterfactual reasoning about alternative worlds. In the real world, however, each unit reveals its outcome only under the factual world. Thus, the factual and counterfactual outcomes can never be jointly learned by any model trained on real data. The problem is therefore tractable only in synthetic settings, where both worlds can be generated. Recent progress on prior-data fitted networks (PFNs) has shown that models pre-trained purely on synthetic data can transfer to real-world prediction. This paradigm has since carried over to causal effect estimation. Despite this fully synthetic setup, existing causal PFNs carry the real-world constraint into their training objective, confining learning to a single world's outcome per unit. The dependence between the two worlds is thereby lost, and the density of the treatment effect relies by default on independence, which is implausible in practice. We therefore propose joint learning, a training setup in which a causal PFN learns the joint distribution of the factual and counterfactual outcomes of each unit. From the joint distribution, we obtain both the response and the individual treatment effect densities, and at inference, we apply optimal transport across units to obtain an approximation of the average treatment effect density. Ours is therefore the first causal PFN approach to provide all three densities together while accounting for the cross-world dependence. We apply joint learning and optimal transport to existing causal PFNs and evaluate it across semi-synthetic, synthetic, and real benchmarks spanning diverse causal scenarios. The jointly trained models significantly improve upon their baselines in point estimation and uncertainty quantification in most settings, indicating that joint learning is a preferable training setup for causal PFNs. 

---
# Cache or Denoise Less? Residual Reuse in a Flow-Matching World Action Model 

**Authors**: Abhinav Gupta, Nissan Pow  

**Link**: [PDF](https://arxiv.org/pdf/2611.19709)  

**Abstract**: Flow-matching robot policies generate each action chunk through repeated network evaluations, one per denoising step. Cross-cycle caches such as C$^3$ache reuse computation from earlier replanning calls, but whether they beat simply taking fewer steps is unclear. We compare the two in Fast-WAM, a 6B-parameter world action model, on LIBERO-Goal. Two uncached steps keep the success of ten: on all 500 initial states they are non-inferior within two percentage points, for one checkpoint and noise seed. Under PyTorch's default compile mode, they also cut a replanning call on an RTX A6000 from 202 to 92 ms. This leaves little room for caching. A cache that keeps two fresh steps cannot be faster than two uncached steps: every call still encodes the images, runs the video expert and computes those two steps. Caching the first of two steps, refreshed every fourth call, saves at most 11--14% of a call by our cost model. Cached policies are also fragile. Their losses tracked the share of the flow path the cached steps cover: under Fast-WAM's current time shift, C$^3$ache's eight cached steps cover 80% of the path instead of 44%, and the cached policy succeeds on 63% of states against 95% uncached. Resampling the noise on every call cost caches covering 60% or more of the path 14--40 points. These results argue for fewer-step baselines and for testing reuse under the deployed sampler and noise settings. We release our evaluation harness, logs and a checklist. 

---
# Where Should LLM Agents Adapt at Test Time? In Context, in Weights, or Both 

**Authors**: Parsa Mirtaheri, Mohammadreza Armandpour, Keivan Alizadeh, Iman Mirzadeh, Parshin Shojaee, Aryo Lotfi, David Harrison, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2611.19700)  

**Abstract**: LLM agents operating repeatedly in the same environment, such as a code repository, a database, or an application suite, encounter tasks that share latent structure. Experience can reveal this structure, allowing agents to reuse environment knowledge and learn from environmental feedback. Agents can adapt at test-time using this experience through two channels: retaining it in context and updating their weights. These channels are usually studied separately, leaving unclear what each contributes and when they complement each other. We compare in-context learning (ICL), reinforcement learning with verifiable rewards using GRPO, and self-distillation using SDPO, alone and in combination, using a shared online protocol and modular framework. Agents encounter matched task streams, are evaluated on each task before adapting to it, and learn only from their own attempts and environment feedback. We evaluate 14 adaptation mechanisms and a stateless baseline with Qwen3.5-9B on 109 tasks from four environments drawn from SWE-bench Verified, AppWorld, ScienceWorld, and BIRD-Critic-SQLite, using five task orders per environment. ICL provides the largest standalone gain: its best policy raises mean success across environments from 28.8% to 49.4%. Adding GRPO to this policy achieves the highest mean success, 51.6%. While adding GRPO improves the overall mean under every ICL policy, SDPO has mixed effects. Trace analyses show fewer lookups and shorter attempts under ICL, consistent with reusing knowledge of the shared structure. However, ICL also frequently skips inspection and verification for the current task. Adding GRPO to ICL helps restore these behaviors while preserving its reduced environment lookups. These observations suggest that reinforcement learning with verifiable rewards can refine how agents use retained experience, motivating the joint study of context management and weight updates. 

---
# Data Attribution Should Steer Pretraining 

**Authors**: Glenn Matlin, Chandreyi Chakraborty, Mohammed Rehan Parwani, Saehee Eom, Anthony Wen-Ming Zang, Akhil Theerthala, Arjun Chatterjee, Tejas Khandwekar, Brandon Michaels, Alvin Deng, Mark Riedl  

**Link**: [PDF](https://arxiv.org/pdf/2611.19698)  

**Abstract**: Pretraining data curation decides what a model learns from, yet what makes a document worth learning from depends on what the model already knows. Influence-driven selectors already rank documents to improve one run; we argue that training-data attribution should steer pretraining by identifying testable data properties that become reusable curation criteria, evaluated on new data against cheaper selectors at matched total cost, discovery included. A document ranking is consumed by the run it was computed for, whereas an explained data property can be tested across corpora and checkpoints, especially where familiar quality signals diverge from measured usefulness. Because historical contribution, an understandable explanation, and future data value require distinct evidence, we specify a feedback loop that interprets attributed sources, tests candidate data changes, and states for five claim types what evidence each requires and what none establishes. We close with a falsifiable comparison against perplexity, classifier, and mixing baselines at equal total compute. 

---
# MathaMAGICal: Validating Trajectory Data Attribution During Capability Acquisition 

**Authors**: Jason Lee, Dikshanta Pokhrel, Phonn Nay Khant, Victor Lee, Moritz Weckbecker, Sebastian Lapuschkin, Wojciech Samek, Mark Riedl, Glenn Matlin  

**Link**: [PDF](https://arxiv.org/pdf/2611.19692)  

**Abstract**: Influence-function-based methods like MAGIC provide exact first-order scores for trajectory data attribution, yet only retraining indicates whether those scores accurately predict data removal. We evaluate MAGIC on $86$M- and $152$M-parameter GPT-2-style models learning multistep arithmetic from scratch on filtered web text and four synthetic math families. Full-run attribution at $86$M suffers from numerical overflow, and even with metagradient clipping, scores correlate near chance with retraining (linear datamodeling score, LDS, $0.17$--$0.22$). In contrast, short checkpoint windows yield stable, accurate predictions at both scales (LDS $0.86$--$0.99$ across four $100$-step windows at $152$M), whereas a single-checkpoint gradient-dot baseline shows near-zero correlation. Rankings of individual training blocks predict retraining only within that single run and do not transfer across seeds or hyperparameters ($\rho \le 0.09$), whereas format-level patterns replicate across runs. Two data families, worked examples and primitive facts, occupy $2.1$--$2.6\times$ their corpus share in both the helpful and harmful $5\%$ score tails, reflecting strong bidirectional local sensitivity distinct from net benefit under multi-epoch family deletion. These results show that trajectory attribution is reliable within short, validated windows and yields family-level insights that generalize across runs. 

---
# Architecting the Corporate AI Infrastructure: The Evolving Roles of AI Governance Committees and Chief Ethics Officers 

**Authors**: David Hason Rudd, Dennis Singh, Cesar Sanin, Bryn James  

**Link**: [PDF](https://arxiv.org/pdf/2611.19670)  

**Abstract**: The accelerating integration of artificial intelligence (AI) into core business operations has created an urgent need for robust corporate governance infrastructures capable of managing complex ethical, legal, and operational risks. While organizations have increasingly adopted high-level AI ethics principles, a critical implementation gap persists: fewer than half of organizations that cite AI governance as a boardroom priority have a formalized governance framework in place. This paper addresses that gap by proposing a structured corporate AI governance architecture (CAIGA) centred on two mutually reinforcing pillars. First, it examines the evolving mandate of specialized C-suite roles, specifically Chief Privacy Officers (CPOs) and Chief Ethics Officers (ChEOs), in formalizing accountability for AI usage across the enterprise. Second, it argues that cross-functional AI Governance Committees, uniting representatives from legal, technology, data science, cybersecurity, and business units, are the essential organizational mechanism for translating ethical principles into operational decisions, including strategic go/no-go approvals for new AI initiatives. Drawing on a conceptual review of peer-reviewed governance literature, regulatory frameworks including the EU AI Act, and documented industry practice, this paper contributes a replicable governance architecture that embeds clear accountability policies, structured risk management workflows, and an organization-wide ethical AI culture. The framework is illustrated through application to the financial services sector. The findings have direct implications for corporate leaders, risk professionals, and policymakers navigating the rapidly evolving global AI regulatory landscape. 

---
# BadGCL: Towards Persistent Backdoor Attack on Graph Continual Learning Systems 

**Authors**: Dong Li, Yufei Han, Xintao Wu, Xujiang Zhao, Wei Cheng, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.19656)  

**Abstract**: Backdoor attacks pose a persistent security threat to machine learning systems, yet their behavior under graph continual learning (GCL) remains underexplored. Conventional graph backdoor attacks often lose effectiveness as models are updated across successive tasks, creating the impression that continual updates naturally weaken malicious associations. This work reveals a contrasting vulnerability: knowledge-preservation mechanisms in GCL can also preserve early injected backdoors when the poisoned update is compatible with the clean-task signal being retained. For graph data, this risk is amplified by structural triggers, which are embedded into the message-passing paths of graph neural networks (GNNs) and can induce stable representation shifts across tasks. To expose this risk, we propose BadGCL, a model-agnostic framework for persistent backdoor attacks under a strict early-stage intervention setting. BadGCL learns topology-aware structural triggers using a gradient mutual-information objective that couples poisoned gradients with clean-task optimization signals, together with a propagation consistency objective that stabilizes trigger-induced effects through graph message passing. We provide theoretical analysis showing that gradient compatibility and propagation stability jointly support long-term backdoor persistence under benign continual updates. Extensive experiments across task-IL and class-IL protocols, four GCL benchmarks, and diverse victim systems show that BadGCL achieves persistent attack effectiveness with limited clean-utility degradation, substantially outperforming graph and continual-learning backdoor baselines and remaining robust against representative defenses. 

---
# LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers 

**Authors**: Siqi Zhu, Suozhi Huang, Yi Zhao, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2611.19649)  

**Abstract**: Looped Transformers improve parameter efficiency by repeatedly applying the same Transformer layers. During autoregressive decoding, however, generating each token requires several sequential passes through these layers, leading to high decoding latency. Pipelined self-speculative methods use shallow-loop predictions to process tokens at different loops in the same batch, but each token must reach the drafting depth before the next can enter the pipeline. We introduce LoopFill, a training-free speculative decoding algorithm for looped LLM that combines multi-token suffix drafts with shallow-loop predictions in a single draft tree. Both drafters contribute to one tree, so a suffix lookup can immediately extend a shallow prediction and several future tokens can enter the pipeline together. The shared block computes the highest-scoring draft candidates at different loop depths in one batch, and full-depth verification preserves the model's autoregressive output distribution. Across seven benchmarks, LoopFill's average speedup over greedy autoregressive decoding is $5.85\times$ on Ouro-2.6B and $5.13\times$ on Huginn-3.5B. The speedups exceed those of LoopAhead, the strongest baseline, by \(40\%\) on Ouro-2.6B and \(25\%\) on Huginn-3.5B. 

---
# SG-LegalClarify: A Large-Scale Benchmark for Legal Reasoning and Ambiguity Recognition in Singapore Law 

**Authors**: Ziwei Gu, Michael S. Lee, Drew Rein, Yash Maurya, Vikneswaran Kumaran, Darshini Ramiah, Chenguang Wang, Udari Madhushani Sehwag  

**Link**: [PDF](https://arxiv.org/pdf/2611.19643)  

**Abstract**: Singapore offers a distinctive testbed for legal reasoning by large language models (LLMs). Its legal system combines inherited English common law with jurisdiction-specific statutes, requiring models to distinguish local doctrine from generic common-law intuition. This setting also highlights the challenge of recognizing underspecification in lay legal queries, because the facts material to a legal outcome depend on the governing doctrine. Yet legal benchmarks rarely cover Singapore law or test this form of ambiguity recognition. To address these gaps, we introduce SG-LegalClarify, a large-scale benchmark for open-ended legal reasoning and ambiguity recognition in Singapore law. SG-LegalClarify contains 1,100 lawyer-authored source questions spanning 10 legal domains, including 340 fully specified and 760 underspecified questions. Each underspecified question has a fixed hidden-fact assignment and a fully specified counterpart, yielding 1,860 evaluation instances. Responses are graded with lawyer-authored Issue, Rule, Application, and Conclusion (IRAC) rubrics. We evaluate 19 LLMs, closed-book and with web search, using a multi-turn pipeline in which models decide whether to answer or clarify, and a user simulator reveals hidden facts only when asked. We find the benchmark far from saturated: the best models satisfy only 69.8% of mandatory rubric items closed-book (average 53.8%) and 77.8% with web search (average 63.0%). Models also struggle to recognize when clarification is necessary. We will release the public split of our benchmark, the model outputs on it, and the evaluation pipeline to support future research. 

---
# ROUNDS: Source-Framed Diagnostic Revision after Multi-Turn Clinical Evidence 

**Authors**: Jinrui Fang, Runhan Chen, Yining Wu, Jiawei Xu, Xu Yang, Jian Yu, Wenqi Shi, Ying Ding, Yuji Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19636)  

**Abstract**: Large language models (LLMs) achieve high scores on medical benchmarks, but does this performance translate into reliable diagnosis as a conversation unfolds? In diagnostic conversations, users can provide clinical findings across multiple turns, ask for a diagnosis without supplying answer options, and introduce opinions from others. Models must therefore form diagnoses from the available evidence and decide how to respond to subsequent advice. We introduce ROUNDS, a framework that evaluates these abilities after a fixed sequence of clinical evidence delivered over multiple turns. ROUNDS compares diagnosis with and without answer options, presents the same incorrect diagnosis under clinician and lay framing, and varies support for competing diagnoses in written five-clinician panels. Across seven models and 1,035 cases from five datasets, average accuracy was 76.9% with answer options and 57.0% without them. Clinician-framed misleading advice produced lower final accuracy than lay-framed advice in every model, with differences of 4.6 to 38.2 percentage points. Initially correct diagnoses more often became incorrect answers than abstentions, and many adopted the suggested disease. Conversely, accuracy increased as more panel members supported the correct diagnosis, including when the majority opinion remained unchanged. These findings reveal a gap between medical benchmark performance and reliable diagnosis in conversation: models respond to external opinions, but remain vulnerable to abandoning correct diagnoses without new clinical evidence. 

---
# Where slide-level foundation-model classifiers fail on atypical breast lesions 

**Authors**: Nerea Hernandez, Luis J. Herrera, Cesar Ramirez, Olga Valenzuela, Ignacio Rojas, Francisco M. Ortuño  

**Link**: [PDF](https://arxiv.org/pdf/2611.19630)  

**Abstract**: Whether a ductal breast lesion is called atypical hyperplasia or low-grade carcinoma in situ increasingly determines surgery. However, slide-level classifiers built on pathology foundation models (PFMs) perform worst on these lesions. Using four PFMs, cross-validation on BRACS and preregistered hypotheses, we studied where the classification of atypia fails. Because slides are labelled with their most severe lesion, only 39% of the slides with annotated atypical ductal hyperplasia (ADH) were labelled ADH. Attention ranked the lesion high but rarely first, and showing the model only the lesion helped little, whereas classifiers trained per decision with lesion-level labels improved all four atypical boundaries. ADH regions inserted into benign slides raised the probability of malignancy more than that of atypia. In a hospital with pathologist-labelled single-subtype slides and in HISTAI, benign slides were scored closer to lesions and triage specificity fell from 0.48-0.53 to 0.00-0.07 and 0.22. 

---
# Robust Well-Log Reconstruction Across Point Gaps and Whole Wells via Boundary-Gated Soft-Thresholding Kalman Fusion 

**Authors**: Yongan Zhang, Zhihan Qiu, Yizheng Li, Youzhuang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.19625)  

**Abstract**: Missing wireline logs hinder subsurface characterization. Deep reconstruction networks suffer from gap-boundary steps and cross-basin shifts, while interpolation and Gaussian smoothers struggle with long intervals and outliers. We propose boundary-gated soft-thresholding Kalman fusion (ST-KF), unifying multiresolution wavelet shrinkage, distance-gated residual prior learning, and closed-form robust Kalman smoothing with learning--inference decoupling. Across two public benchmarks totaling 263 wells in the Norwegian North Sea and New Zealand's Taranaki Basin, ST-KF ranks first over 0.15--100-m anchored gaps with 0.09% parameter overhead. Against the strongest of nine baselines, pooled RMSE falls by 5.0%--8.3% for bulk density and 6.4%--15.6% for compressional slowness; against the four strongest baselines, reductions are 0.7%--4.1% for neutron porosity and 1.3%--2.9% for photoelectric factor. Ablations and sensitivity analyses confirm operator synergy and stable performance: removing state-space fusion increases short-gap errors by up to 58.8%, while zero-shot transfer cuts anchored errors by 14%--68%. The results show that local borehole continuity transfers across basins whereas cross-property mappings shift with regional geology, supporting measurement-guided log reconstruction. 

---
# Well-Log Reconstruction with Pre-trained Geological Knowledge: A Knowledge-Conditioned Large-Language-Model Generative Agent 

**Authors**: Yongan Zhang, Yitao Yu, Yizheng Li, An Gong, Youzhuang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.19624)  

**Abstract**: Missing well-log intervals make subsurface reconstruction ill-posed. Existing numerical regression models lack geological prior knowledge to constrain ambiguous inverse mappings. We propose GeoLog-Agent, a knowledge-conditioned generative agent using pre-trained large language models (LLMs) as a source of geological priors. A universal Geo-Harness translates LLM stratigraphic reasoning into formation-anchored petrophysical trajectories, feature-wise linear modulation, and conditional adversarial constraints. On the Norwegian North Sea benchmark, GeoLog-Agent achieves a mean $R^2$ of 0.9307 and a peak $R^2$ of 0.9713. It exceeds the nine-baseline average $R^2$ of 0.8387 by 0.0920, with a 30.97% average RMSE reduction, and exceeds the strongest baseline by 0.0274--0.0498 in $R^2$, with 16.25%--24.05% lower RMSE. Ablations confirm the effectiveness and synergy of geological prior reasoning, multi-scale dilated generation, and conditional adversarial modules. Transferring the coupling mechanism to six conventional deep learning models improves their $R^2$ by 0.0162--0.0789 with cross-well stability. These results demonstrate how foundation-model domain reasoning can constrain deep generative reconstruction of missing petrophysical measurements. 

---
# Synthetic Neurosymbolic Task Generation with Controlled Constraint Complexity 

**Authors**: Mykhailo Buleshnyi, Zsolt Zombori, Michael Benedikt  

**Link**: [PDF](https://arxiv.org/pdf/2611.19597)  

**Abstract**: We introduce NeSySy, a synthetic generator for neurosymbolic learning tasks that jointly produces logical constraints and consistent datasets. Unlike existing benchmarks, NeSySy provides fine-grained control over key properties such as constraint structure, tightness, input dependence, and data generation and alignment strategies. In particular, it allows the use of rich constraints over infinite domains, and uses solvers to ensure that the constraints are meaningful and that generated samples satisfy the constraints.
We design NeSySy to support both learning with declarative constraints and learning from distant supervision. Across a range of constraint families, we provide evidence that its parameters induce systematic and largely independent effects on task difficulty. Using NeSySy, we study constraint-aware learning methods, identifying regimes where explicit constraint enforcement is beneficial and where expressive unconstrained models can instead learn constraint structure directly from data. Our results establish NeSySy as a controlled testbed for analysing the interaction between learning and logical structure.
NeSySy is built around a pipeline that incorporates solvers. Satisfiability checkers (CNFgen and Z3) generate and certify constraints. Constrained optimizers (RC2 for MaxSAT, Z3 for OMT) compute optimal repairs that guarantee every generated sample is consistent. 

---
# PBit1: One Programmable P-Bit per Parameter - A Finite-Control Representation of Pretrained Networks 

**Authors**: Omar Ramadan  

**Link**: [PDF](https://arxiv.org/pdf/2611.19588)  

**Abstract**: We introduce PBit1, a finite-control representation of pretrained neural networks that assigns one programmable p-bit to each learned parameter. A 12-bit command sets its probability, a shared affine range maps that probability to the weight domain, and temporal observations reconstruct the weight for inference. Operator-aware range grouping and sensitivity-based sampling allocation adapt the representation to the network. Across five independent weight draws at 32,768 observations per scalar, selected configurations preserve task quality in GPT-2, DistilBERT, and LFM3-1.7B with every learned scalar converted and no fine-tuning. GPT-2 and LFM3 perplexity increase by approximately 0.66% and 0.26%, respectively, while DistilBERT accuracy changes from 87.51% to 87.49%. A sensitivity-based policy reduces GPT-2 reconstruction observations by 41.7% while retaining the quality criterion. Complete-model evaluations reuse reconstructed weights in BF16 inference, and THRML operator experiments examine native sampling and spatial-temporal tradeoffs. Together, these results establish PBit1 as a concrete construction for studying finite-control weight representation and temporal reconstruction in pretrained networks. 

---
# Risk-Averse Decision-Focused Learning: Minimizing CVaR of Regret 

**Authors**: Jayanta Mandi, Senne Berden, Victor Bucarey, Concepcion Dominguez, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2611.19575)  

**Abstract**: Many combinatorial optimization (CO) problems involve parameters that are uncertain when the problem must be solved. In such settings, the parameters have to be predicted from correlated contextual information before a solution can be computed. \emph{Decision-focused learning} (DFL) addresses this by training the predictive model to produce predictions that optimize downstream decision quality. Most existing DFL methods are risk-neutral and minimize the expected regret. In high-stakes applications this can lead to undesirable outcomes, because decision-makers must also account for the tail risk of decision quality. Prior risk-sensitive approaches typically model uncertainty explicitly and solve a stochastic or robust optimization problem at inference time, which increases computational complexity. We instead propose a DFL framework that achieves risk-sensitive behavior while retaining point predictions and deterministic optimization at inference time. Starting from the dual representation of the Conditional Value-at-Risk (CVaR) and adding entropy regularization, we derive a smooth surrogate for the CVaR of regret. Its gradient induces an instance-dependent reweighting within each minibatch that emphasizes high-risk outcomes, so the approach can turn existing surrogate-loss DFL methods, including the SPO+ loss, into risk-sensitive variants. When instantiated with SPO+, the resulting objective admits a generalization bound on the CVaR of the true regret. Experiments on knapsack, portfolio, shortest-path, and energy-aware scheduling problems show substantial reductions in tail regret when the uncertain parameters exhibit heterogeneous tail behavior. 

---
# Spectral Steering: Local Inverse Geometry for Frozen Diffusion Generators 

**Authors**: Angelo Moroncelli, Alessandro Leanza, Loris Roveda  

**Link**: [PDF](https://arxiv.org/pdf/2611.19572)  

**Abstract**: A frozen diffusion generator may need to change several output properties while preserving others. We study when guidance faithfully realizes such a structured request and whether local inverse correction improves its direction. Our account links this benefit to the interaction between the request and the objective-specific remaining-sampler Jacobian: conditioning alone is insufficient. Spectral steering combines a request-specific cosine-gain predictor with damped inverse correction and fixed-time relinearization for finite edits. Scalar nulls and matched-movement controls test the predicted mechanism. Across 8,923 instances in a retrospectively identified local regime, predicted and realized gains correlate at Spearman 0.79, versus -0.11 for condition number. Experiments span images, audio, robotics, and control and planning, including complex character motion generation. On 50 paired inpainting cases, four corrections improve unobserved-pixel PSNR over Adam by 3.25 dB at equal total derivative traversals. Together, the analysis and controls connect request-dependent geometry to directional gains and establish where these gains translate into downstream performance. 

---
# Learning from Mutants: Teaching Small Language Models to Synthesise Loop Invariants 

**Authors**: Daragh King, Vasileios Koutavas  

**Link**: [PDF](https://arxiv.org/pdf/2611.19569)  

**Abstract**: Large language models (LLMs) have proven promising for synthesising loop invariants, a long-standing obstacle to automated program verification. Yet, frontier LLMs are typically accessible only through third-party APIs, hampering reproducibility and local deployment. Small language models (SLMs) sidestep these obstacles but require task-specific fine-tuning, for which high-quality training data is needed. We present a failure-driven data curation method for fine-tuning language models to synthesise loop invariants. Inspired by mutation testing, our approach applies semantic mutation operators to ground-truth invariants, producing semantic mutants that induce verification failure. Each mutant is paired with verifier feedback and its ground-truth counterpart, yielding training examples for feedback-guided invariant refinement. Whereas the vast space of mutants is a long-standing burden in mutation testing, it becomes an asset when generating training data, multiplying the examples obtainable from a fixed set of verified programs. We implement our approach for Dafny, augmenting an existing synthetic dataset of verified programs, and introduce a framework that iteratively refines candidate invariants using verifier feedback. On 472 DafnyBench programs, our best-performing fine-tuned Gemma-4-E4B configuration verifies 50.8% of programs zero-shot, compared to 8.1% for the base model, a 6.3$\times$ improvement. Higher-order mutants prove more effective as training data than first-order mutants, and adding mutation-based examples to direct synthesis examples, in which invariants are removed rather than mutated, improves performance under iterative refinement. Our approach also transfers to Qwen3.5-4B, whose zero-shot verification rate rises from 9.3% to 53.6%. 

---
# Read, Then Shade? How Visual Evidence Changes Incentive-Driven Misreporting in Vision-Language Advisors 

**Authors**: Hamidreza Hasani Balyani  

**Link**: [PDF](https://arxiv.org/pdf/2611.19561)  

**Abstract**: Vision-language models (VLMs) increasingly read a scene and report it to someone who acts on the report, sometimes under objectives that differ from that person's. We ask whether a stated incentive moves what a VLM reports, and whether this changes when the value must be read from pixels rather than given as text. In seeded synthetic scenes, a model reads the position of a target on a labelled axis and reports one integer to a fixed receiver; it is paid, in simulation, for accuracy or for a report a few units to one side, and every scene is also presented with the true value as text. Across a pilot and three follow-up studies with hash-frozen protocols, four small open VLMs from three families, and 15,170 held-out trials, a stated payoff moved reports in both modalities, and exact compliance at three units was significantly lower with the image than with text for three of four models; Qwen2.5-VL 7B showed no reliable difference. Adding the true value as text to the image prompt removed the gap in every model. For InternVL3 the gap vanished on scenes it read correctly when it wrote its reading first; for Gemma 3 it persisted. For Qwen2.5-VL 3B, which read every scene correctly, the gap held in four of five payoff paraphrases, nearly vanished when the model wrote its reading before its report or when the prompt stated that the client wants the truth, vanished with an unpaid goal, and persisted when the model reported before stating its reading; how often it shaded at all, even with text, depended strongly on the wording. When told that the client wants the true value, three of four models still shaded every text report. Visual input did not make these advisors reliably more truthful; it made compliance with a stated incentive depend on whether the read value is made explicit and on output order, a dependence that text-only evaluations and mean-shift metrics do not reveal. 

---
# SemMask: Semantic-Prioritized Progressive Masking for Diffusion Language Models 

**Authors**: Qi Chen, Zifeng Cheng, Zhiwei Jiang, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19560)  

**Abstract**: Diffusion Language Models (DLMs) are designed to enable globally parallel generation. However, in practice, they often exhibit pseudo-autoregressive behavior and a function-word-first bias, which undermines their performance on tasks such as text writing and code generation that benefit from global planning and parallel decoding. To address this, we propose SemMask, a novel fine-tuning paradigm that leverages a semantic-prioritized progressive masking strategy to encourage the model to decode high-information semantic tokens early and globally, thereby enhancing its global planning and parallel generation capabilities. Specifically, we first use the gradients of the original DLM with respect to each token to identify semantic tokens. During masking, these semantic tokens are preserved with priority, establishing a progressive generation order in which semantic tokens are decoded prior to function tokens. Meanwhile, we adjust the loss weight for semantic tokens to further reinforce their early prediction. Experimental results show that SemMask achieves quality comparable to strong autoregressive baselines on both text writing and code generation tasks, while providing superior semantic coherence. Notably, SemMask achieves higher accuracy with substantially fewer decoding steps, indicating that the early establishment of semantic structure improves the efficiency--accuracy trade-off. 

---
# Inquest: Log-Guided Root-Cause Code Localization with a Staged Agent 

**Authors**: Yuvraj Sehgal, Sneh Patel, Mahsa Panahandeh, Naser Ezzati-Jivan  

**Link**: [PDF](https://arxiv.org/pdf/2611.19554)  

**Abstract**: When a production issue is reported, engineers get a description and some logs, but no failing test and no way to reproduce the fault. The logs show where the failure was noticed, not where it was caused: in 49% of the real reports we study, the text names none of the files that the fix later changed. We built a benchmark for this setting from two public datasets and 27 open-source projects (Apache, Red Hat, MongoDB, MariaDB, Qt and Sakai). It holds 38,001 resolved issue reports with their pasted logs, the code as it was while the bug was still present, and the files the fix changed. We present INQUEST, an agent that finds the faulty code using a 27B open-weights model. Code, not the model, controls the procedure. It first gathers the evidence, including the files fixed in similar past issues that were resolved before the report was filed. The model then makes only three bounded decisions: which files to open, what role each file plays in the failure, and how to rank them. On 26,544 evaluation reports, INQUEST ranks a changed file first for 68.6% and in the top five for 85.9%, names the correct module for 90.2%, and ranks a changed function first for 53.8%. On the reports that name no faulty file, it reaches 52.8% and 73.6%. A free agent loop with the same model and tools finds files equally well, but names the faulty function 14 points less often. On a 1,000-report sample, INQUEST beats our reimplementations of two published agents by 7 and 20 points of top-1 accuracy (4 and 16 counting only the reports those agents answered), and the same model used as a re-ranker by 6. We report the cost of every system and the results that go against us, and we release the benchmark, both systems and every run transcript. 

---
# Lesion-Aware Explainable AI for Diabetic Retinopathy Grading from Fundus Images 

**Authors**: Truong Thinh Nguyen, Hai Phong Nguyen, Truong Thanh Hung Nguyen, Giang Son Tran, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.19550)  

**Abstract**: Post-hoc explanations of deep-learning models for diabetic retinopathy (DR) grading are difficult to trust because different attribution methods highlight different retinal regions for the same image, and a visually plausible heatmap does not show that the highlighted pixels drive the prediction. We propose the Consensus-Calibrated Explanation Map (CCEM), which fuses Grad-CAM++, Ada-SISE, and SmoothGrad Integrated Gradients (SG-IG) using image-specific weights derived from annotation-free reliability cues: deletion sensitivity, inter-method agreement, compactness, retinal containment, and peak structure. CCEM explains a frozen ordinal grader, an EfficientNet-B7 with CBAM and a scalar regression head, pre-trained on EyePACS 2015 in Stage 1 and fine-tuned on APTOS 2019 in Stage 2 with quadratic weighted kappa (QWK)-optimized thresholds; it reaches a QWK of 0.920 on the hidden Kaggle test set and 0.559 on the external MAPLES-DR cohort. Explanations are evaluated on MAPLES-DR, 198 MESSIDOR images with expert pixel-level lesion annotations that are never used for training, threshold fitting, or fusion. On the 193 images with annotated lesions, CCEM obtains the highest Soft Pointing Game score (69.61%; 95% CI 63.7-75.3), defined as the ratio of the strongest lesion-supported saliency to the global map maximum rather than a lesion detection rate, together with the lowest deletion AUC (0.570) and the best deletion-insertion balance (0.115); each gain is significant against the strongest single explainer on paired tests. Annotation-free, per-image reliability weighting therefore turns three disagreeing explainers into one map whose peak evidence is more lesion-supported and more faithful to the model than that of any single explainer. 

---
# Beyond Prompts: Runtime Contracts for Coding Agents 

**Authors**: Reda Heddad  

**Link**: [PDF](https://arxiv.org/pdf/2611.19549)  

**Abstract**: Coding agents act on repositories through shell commands and file edits, and teams steer them with written engineering rules. Industrial settings expect such rules to be enforced and their application to be evidenced, and a rule given as prompt text provides neither. We test whether executable contracts enforced while the agent works reduce rule violations compared with the same rules in the prompt, retrieved as guidance, or checked only after generation. Violations are scored by an oracle whose methods differ from the in-loop checks, and the oracle is validated by two human reviewers and a model reviewer. Across three studies on two model families (240, 112 and 108 runs), enforcement during execution produced no oracle-detected violation in 104 paired runs, against 21 to 55% of runs without rules, and no loss of functional correctness in the two confirmatory studies. In the confirmatory study on tasks aimed at known failure situations (Study 3, protocol hashed before the first run), enforcement lowered the share of violating runs by 31 percentage points relative to no rules (Holm-adjusted p = 0.014); the 17-point difference from prompted rules was not significant (p = 0.12), and neither was any contrast in Study 2, where only 3 of 14 tasks produced a violation. Prompted rules removed some violation types but not recursive deletes of scratch directories, and checks after generation repaired file edits but not commands that had already run. That enforcement satisfies the rules it enforces is expected by construction; what the data add is where prompting falls short, the absence of a functional cost, and the limits of the rules and of the oracle themselves. The studies use low-cost models and one small fixture. 

---
# Func-JEPA: Learning a Sobolev Space of Functions as Latent Vectors 

**Authors**: Alejandro Moreno Arcas  

**Link**: [PDF](https://arxiv.org/pdf/2611.19542)  

**Abstract**: Pretraining an encoder once and reusing it across tasks is a widespread approach in vision and language. However, neural symbolic mathematics still trains a dedicated sequence-to-sequence model for each task, so that no representation of a function is shared. In this work, we propose the Function Joint-Embedding Predictive Architecture (Func-JEPA, abbreviated f-JEPA): an encoder of mathematical functions, pretrained on their symbolic form, whose latent space imitates the Sobolev space H^1 and in which differentiation is a learned latent operator. The norm of a latent is computed exactly from the function, whereas its direction is learned to match Sobolev angles. The frozen encoder is then reused by lightweight modules for decoding, composition, integration, and regression from samples. The representation and four downstream tasks are evaluated within and outside the pretraining distribution. f-JEPA recovers exact formulas from noisy samples more often than the compared symbolic-regression methods, whereas computer algebra and task-specific models remain stronger on standard integration and differential-equation benchmarks. 

---
# Agent Authenticator: Passkey-Grade, Hardware-Bound Authentication for AI Agents 

**Authors**: Harish Santhanalakshmi Ganesan  

**Link**: [PDF](https://arxiv.org/pdf/2611.19536)  

**Abstract**: AI coding agents now push code, publish packages, call model APIs and operate cloud resources on behalf of developers. They authenticate to these services with the same bearer secrets humans use, personal access tokens, API keys and package-registry tokens, stored where the model, its subprocesses and any co-resident malware can read them. Incidents in 2025 and 2026, including prompt-injection attacks on coding agents and self-propagating supply-chain worms, show that this arrangement fails in predictable ways. Passkeys solved the analogous problem for humans, but they require a human gesture on every use and cannot be operated by an autonomous process. We present the Agent Authenticator, a design that gives agents passkey-grade authentication properties: per-service hardware-bound keys, no shared secrets, origin binding enforced by the runtime rather than the model, normalized attestation of the execution environment, human-signed scoped mandates, and cross-device human approval for high-risk actions. The same protocol serves agents on developer laptops and in cloud enclaves, and it deploys as a profile over existing mechanisms, MCP authorization, OAuth token exchange including the Identity Assertion JWT Authorization Grant, git and SSH, rather than as a new identity system. The authenticator also produces a signed, hash-chained record of each action it authorizes. We analyze the design against a threat model centered on prompt injection and same-host compromise, map it onto five documented incidents, and discuss its limits. 

---
# Do Probing and Steering Favor the Same Directions Within and Across Language Models? 

**Authors**: Afshin Sadeghi, Daniel Becker  

**Link**: [PDF](https://arxiv.org/pdf/2611.19499)  

**Abstract**: A direction in a language model's hidden state can be used to probe whether a concept is present or to steer the model toward it. Although probe directions are often reused for steering, the direction that best detects a concept need not be the one that best changes it. We study this distinction in three instruction-tuned models from different families. We construct a one-parameter family of directions ranging from a covariance-adjusted discriminant to the mean difference, and evaluate probing and additive steering, which adds the direction to the hidden state, along the family at a fixed side effect. For truthfulness and sentiment, in every model, the best probe is not the best steering direction: steering consistently favors directions closer to the mean difference. This separation holds across 59 of 60 layer-model-concept combinations in a layer sweep, with matched probe and steering texts in all six model-concept combinations, and under a rotation-based steering operator. The exact steering optimum is less stable: for sentiment, an intermediate direction produces a 27-56% stronger steering effect than the mean difference, but this gain shrinks or disappears with matched texts and depends on the layer and the operator.
The two roles also transfer differently across models. A probe transferred by a linear map fitted on neutral text retains 81-91% of the target model's above-chance probing performance, while the transferred mean difference retains less of its steering effect in 11 of 12 model pairs. This is a consistent tendency, although our predefined criteria classify only three pairs, and nine remain inconclusive. Finally, as an application on PopQA, steering toward "unknown" increases abstention by 18-44 percentage points on uncertain errors, versus 2-5 on confident errors, while rarely correcting errors in cases where a correct answer is reachable. 

---
# InvarOrd: Self-Supervised Ordinal Knowledge Distillation with Rank-Invariant Alignment for Label-Efficient Cross-Species Hyperspectral Fish Freshness Assessment 

**Authors**: Kazi Nabiul Alam, Omenuko Joseph Nwebonyi, Pooneh Bagheri Zadeh, Akbar Sheikh-Akbari  

**Link**: [PDF](https://arxiv.org/pdf/2611.19472)  

**Abstract**: Hyperspectral models of fish freshness rarely transfer between species, and labelling a new species is slow and destructive. Freshness is also ordinal, and standard domain alignment can erase that order. We present InvarOrd, which adapts to a new species from 30 labelled cubes (5 per rank) in three stages: self-supervised spectral pretraining on unlabelled source cubes, cross-species alignment by a shrinkage covariance discrepancy computed only within matched ranks, and ordinal distillation into a compact student. The alignment term is provably invariant to the rank means, so it gives no incentive to collapse the ordinal axis. On paired salmon and cod data (800 cubes each, 16 days), a source-trained teacher reaches 1.6 to 1.9 days mean absolute error (MAE) on the other species, with quadratic weighted kappa (QWK) below zero. The 423K-parameter student, 2.54 times smaller than its teacher, attains the lowest MAE (1.476 days) and highest within 1-day accuracy (56.1%) among 14 configurations, matching the best of six ImageNet-pretrained backbones (ConvNeXt-T, 1.503 days) at 66 times fewer parameters, 1.44ms per cube on a GPU and 5s re-adaptation. InvarOrd beats target-only training by 0.29 days (p = 0.031). Among alignment objectives, rank-invariant alignment retains the most ordinal structure (embedding rank correlation 0.183, QWK 0.112), whereas global discrepancy lowers the rank correlation to 0.049 and QWK to zero at only 0.014 days higher MAE. 

---
# VAST: Compiling Grounded Vulnerability Scenarios into Verifiable, Multi-Step Environments for Evaluating LLM Red-Team Agents 

**Authors**: Zhen Wan  

**Link**: [PDF](https://arxiv.org/pdf/2611.19465)  

**Abstract**: Evaluating Large Language Model (LLM) security agents requires reproducible environments whose workflow difficulty can be varied and whose failures can be diagnosed. We present VAST (Verifiable Adversarial Scenario Training & Evaluation), a declarative range compiler separating vulnerability scenarios from a compiler-introduced workflow-hardening layer. Following a post-hoc code and AST audit across all topics, the main benchmark retains 109 seed edges (45 with strict upstream CVE annotations), 72 family identifiers (62 under shared-HMAC relay grouping), 81 topic configurations (76 core plus 5 bounded execution models), and 1,387 instances (359 canonical four-gate-revalidated specs plus 1,028 seed variants) in a 42:39 partition.
On the TopicServer compiler stratum, a budget-matched 270-episode prompt contrast across 10 compiled mechanism-family clusters (30 P2/P3/P4 instances, 40 turns, temperature 0.0) shows that glm-5.3, grok-4.7, and deepseek-v4-pro solve 12/90 instances under contract_only and 14/90 under generic_hint (cluster sign p=1.00), whereas supplying the shared workflow protocol in full_guide lifts completion to 76/90 (28/30, 23/30, 25/30; cluster sign p<=0.00391). On the separate ranges_v2 ledger, a 126-episode evaluation across 6 non-TopicServer workflows and multi-container Docker/Linux stacks solves 88/126 (69.8%) (28/42, 31/42, 29/42), while an 18-episode intervention on asynchronous Redis worker chaining (v2_02) lifts solves from 0/18 to 18/18 (100%) in 4-6 turns when installed worker modules are disclosed in GET /sched/directory. Finally, a three-seed (s in {42, 1024, 2025}) LoRA recipe comparison on Qwen2.5-Coder-7B-Instruct across 37 lineage-disjoint families (231 episodes on 33 held-out instances) yields 0/33 for Base, 1.67 +/- 2.89 / 33 for Golden SFT, and 4.67 +/- 2.31 / 33 for Header-Recovery SFT, with all solves on v2_03. 

---
# CLEW: Log-Anchored Backward Tracing for Root-Cause Code Localization 

**Authors**: Wenji Bai, Muhammad Waseem, Jaakko Peltonen, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2611.19448)  

**Abstract**: Failures in industrial software systems may involve multiple components and be difficult to reproduce outside the production environment. Root-cause code localization therefore often relies on issue descriptions and runtime logs. However, log evidence may be noisy or incomplete, and the code reporting a symptom may differ from the code containing the underlying defect. These conditions make it challenging to identify relevant log evidence and determine which functions could have caused the reported failure. To address this gap, we introduce CLEW, an agentic approach that combines rule-based log-to-code anchoring with LLM-guided backward code tracing. CLEW organizes selected observations into diagnostic states, guides source exploration from anchored symptom functions, and returns ranked candidate functions, an affected module, and an explanation comprising a proposed mechanism path and associated log events. For evaluation, we construct LogLoc, a benchmark comprising 192 real-world issue and fault-injection cases. Against three baselines, CLEW records the highest observed Acc@1 (77.3%) and MRR (81.3%) on LogLoc-Issues and matches the leading baselines at 97.6% Acc@1 on LogLoc-Airflow. Averaged over all 192 cases, CLEW's wall-clock time is approximately 73% lower than LocAgent's at a similar average inference cost. No aggregate degradation is observed under the tested perturbations, with modest changes in Acc@1 and MRR across the evaluated seeds and base models. These results suggest that CLEW can combine function-level localization with evidence links and runtime efficiency under the evaluated conditions. 

---
# Do Rule-Based Tools Help LLMs Translate Endangered Uralic Languages in the Prompt or over MCP? 

**Authors**: Iaroslav Chelombitko, Mika Hämäläinen  

**Link**: [PDF](https://arxiv.org/pdf/2611.19439)  

**Abstract**: Komi-Zyrian, Moksha, Erzya and Udmurt have mature rule-based resources, among them finite-state analysers, constraint grammars and bilingual dictionaries, which language models do not use. We ask whether giving a model access to these resources improves machine translation from the four languages into English, and whether it matters how the access is provided: as analyses computed in advance and written into the prompt, or as tools that the model calls itself through the Model Context Protocol (MCP). We evaluate manually: an expert linguist read every English output of the strongest model under three conditions, 816 sentences in all, accepted or rejected each one, and named the main error in every rejected translation. With the analyses in the prompt, the number of accepted translations rises from 13 to 43 out of 272; with the tools attached it rises to 55. The gain comes from the bilingual dictionary. For Udmurt, whose dictionary has no English side, the prompt-based condition brings no gain at all, while for the other three languages an English gloss reaches the prompt for 74 to 80% of the source words. Tool calling is also much harder to get working than prompt injection. Of 31 open-weight models, 14 never call a tool, 11 of them because they cannot produce a usable call, and half of the calls that are made name a language the toolkit does not support. We release the data set, every model output and all tool-call logs. 

---
# Prepare Lighter and Faster for Open-Vocabulary Queries: A Query-Wise 3D Segmenter for Gaussian Splatting 

**Authors**: Nhat-Quynh Le-Pham, Khoi Nguyen, Binh-Son Hua  

**Link**: [PDF](https://arxiv.org/pdf/2611.19429)  

**Abstract**: Open-vocabulary 3D understanding aims to localize and segment arbitrary objects beyond predefined categories. However, existing approaches based on Gaussian Splatting often rely on expensive preprocessing and lengthy training, limiting their applicability when fast and lightweight solutions are desired. We propose a lightweight, training-free framework for query-driven 3D instance segmentation on Gaussian Splatting. Instead of conventional mask-wise CLIP embeddings, we use dense pixel-level CLIP features to preserve fine-grained semantic information and enable flexible text-driven localization without relying on predefined instance proposals. Given a text query, we identify a high-quality target region in a carefully selected pivot view and propagate it across views using a video segmentor, providing consistent and comprehensive coverage of the target instance. We then lift the resulting 2D tracklets to 3D Gaussians through a novel visibility voting strategy, yielding accurate 3D segmentation masks with clean object boundaries. Extensive experiments demonstrate state-of-the-art performance on LERF-OVS, ScanNet, and ScanNet++, competitive results on 3D-OVS, and significant speedups over existing open-vocabulary 3D segmentation methods. These results highlight the effectiveness of our framework for efficient open-vocabulary 3D understanding without per-scene training. 

---
# Do Pretraining Optimizer Gains Transfer to KL Distillation? 

**Authors**: Lavinia Ghita, Dhruv Desai  

**Link**: [PDF](https://arxiv.org/pdf/2611.19419)  

**Abstract**: Matrix-structured optimizers can improve language-model pretraining, but it is unclear whether these gains transfer to short-horizon KL distillation from an instruction-tuned initialization. We compare AdamW and diagonal AdaGrad with two tractable Shampoo reductions, SOAP, and All-2D Muon across four dense Qwen student-teacher pairs under a common forward-KL objective and fixed training budget. Across scales, none of the evaluated matrix-structured recipes consistently improves the quality versus trainer-time trade-off over the elementwise references: AdamW generally attains the lowest validation KL, while AdaGrad can be slightly faster at larger scale. SOAP often tracks AdamW in KL but incurs substantial trainer-time overhead, and the two Shampoo reductions reverse ordering across model pairs, showing that conclusions depend on the feasible recipe. These results suggest that pretraining optimizer gains do not transfer automatically to KL distillation and should be re-established for the target objective and systems regime. 

---
# Pixel-KWS: Rendered-Glyph Representations for Orthographically Robust Open-Vocabulary Keyword Spotting 

**Authors**: Adarsh Arigala, Ramesh Gundluru, Srinivasan Umesh, Ludovick Lepauloux, Hakim Hacid  

**Link**: [PDF](https://arxiv.org/pdf/2611.19389)  

**Abstract**: Open-vocabulary keyword spotting (KWS) allows users to enroll arbitrary keywords at inference time. However, character and phoneme-based text encoders rely on fixed symbol vocabularies learned during training, limiting their ability to handle unseen or visually similar symbols such as homoglyphs, l33tspeak substitutions, and accented characters. We propose PIXEL-KWS, an open-vocabulary KWS framework that represents characters through rendered glyphs rather than learned embedding lookups. We render each character as a grayscale image and encode it with a lightweight CNN, enabling the text encoder to generalize from visual form rather than a fixed vocabulary. On LibriPhrase, PIXEL-KWS and its matched character-based counterpart achieve 9.57% and 9.55% EER on LibriPhrase-hard, 0.43% and 0.41% EER on LibriPhrase-easy, outperforming all published baselines without pre-training or synthetic data. PIXEL-KWS provides substantially greater robustness to both homoglyph and l33tspeak symbol-level perturbations. These results highlight glyph-based representations as a simple and effective approach for robust open-vocabulary KWS in practical settings. 

---
# SMELL: Sparsity-aware Memory Efficient Federated Fine-tuning for Long-context LLMs 

**Authors**: Yin Xu, Yongbo Yang, He Sun, Miao Sun, Mingjun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.19349)  

**Abstract**: Federated fine-tuning enables Large Language Models (LLMs) to collaboratively adapt to distributed private data. However, long-context fine-tuning on resource-constrained devices faces substantial challenges arising from excessive activation memory, heterogeneous token-importance patterns across clients, and communication overhead. To address these challenges, we propose SMELL, a Sparsity-aware and Memory-Efficient federated fine-tuning framework for Long-context LLMs. SMELL integrates three complementary mechanisms: Local Sparse-Forward Adaptation (LSFA), Consensus-Anchored Sparsity Coordination (CASC), and Communication-Efficient Progressive Update Recovery (CPUR). These mechanisms jointly enable memory-efficient local adaptation, heterogeneity-aware sparsity coordination, and lightweight model-update transmission. Experiments on OPT-350M and LLaMA2-7B demonstrate that SMELL reduces peak memory consumption by up to 88.35% compared with conventional fine-tuning and 13.25% compared with forward-only LoRA fine-tuning at a 16K context length. Furthermore, SMELL improves fine-tuning performance under certain non-IID settings and supports compact update transmission with progressive reconstruction, demonstrating its potential for resource-efficient federated long-context LLM adaptation. 

---
# OmitEHR: Scalable Omission Detection in LLM-Generated Clinical Summaries 

**Authors**: Bridget Lin, Ryan Welch, Bobak Seddighzadeh, Fran Riley, Kaitlyn Stoehr, Margarita I. Pipinos, Dimitri Speron, Emma V. Sage, Sneha Jain, Nigam H. Shah, Emily Alsentzer  

**Link**: [PDF](https://arxiv.org/pdf/2611.19343)  

**Abstract**: Health systems increasingly rely on large language models (LLMs) to draft summaries of patient medical records, yet evaluations now find omissions, the absence of clinically important information, to be the most common failure mode. Detecting omissions in an LLM-generated summary requires knowing which facts in the record are salient, a signal that existing methods hand-specify, ignore, or recompute at inference with costly LLM judges. We present OmitEHR, a reference-free omission detector that learns clinical salience from historical clinician-written summaries predating widespread LLM use. OmitEHR learns to predict which source-note sentences clinicians include in their summaries, then surfaces high-salience sentences not covered by an LLM-generated summary as candidate omissions. We apply OmitEHR to brief hospital course (BHC) summarization using 5,000 admissions from a large academic medical center. On held-out admissions, OmitEHR outperformed all evaluated baselines at predicting salient source sentences, including a prompted LLM judge. The full omission detection pipeline took 17.7 to 19.5 seconds and cost \$0.03 to \$0.13 per summary depending on the base model, 1.6 to 1.7 times faster and 5.6 to 6.7 times cheaper than scoring every source sentence with an LLM judge. In a blinded review of 25 admissions, two board-certified clinicians confirmed a mean of 5.6 clinically meaningful omissions per summary surfaced by OmitEHR, compared with 3.3 for a full LLM pipeline, counting candidates confirmed by at least one clinician. The efficiency and validity of OmitEHR make it a practical tool for health systems to flag omissions before a summary reaches the clinician. 

---
# Benchmarking the State of AI for Crop Yield Forecasting: A Global Assessment Across Modeling Paradigms 

**Authors**: Michiel Kallenberg, Philip Janz, Christoph Jörges, Vageesh Saxena, Pratishtha Poudel, Mohammed Musthafa Rafi, Ioannis N. Athanasiadis  

**Link**: [PDF](https://arxiv.org/pdf/2611.19330)  

**Abstract**: Accurate crop yield forecasting is essential for food security, agricultural policy, and commodity markets. Although statistical, process-based, and machine learning approaches have all been proposed, their relative strengths remain poorly understood due to the lack of large-scale, standardized comparisons. We address this gap through a comprehensive benchmark of five modeling paradigms -- statistical baselines, process-based models, conventional feature-engineered machine learning, deep sequence models, and tabular foundation models -- using the CY-Bench benchmark spanning 63 country-crop datasets (>12,000 administrative regions) across six continents. Beyond conventional pooled accuracy metrics, we introduce a multi-dimensional evaluation framework that separately quantifies overall, spatial, temporal, and anomaly prediction skill. Across multiple crops and countries, feature-engineered machine learning and tabular foundation models achieve the strongest overall predictive performance. Data-driven models reduce NRMSE in 72% of countries growing maize and 78% of countries growing wheat, with median reductions of 8.8% and 11.9%, respectively. Across all modeling paradigms, spatial prediction is substantially easier than temporal and anomaly prediction. While data-driven models markedly improve temporal and anomaly prediction over statistical baselines, accurately forecasting year-to-year yield variability remains challenging. These findings suggest that future progress in large-scale crop yield forecasting will depend on the improved characterization of the environmental and management factors driving inter-annual yield variability, founded on effective representations for crop yield forecasting systems that support agricultural decision-making, early-warning applications, and food security planning 

---
# Generation of protein conformations using a simple inverted transformer 

**Authors**: Luigi Leonardo Palese  

**Link**: [PDF](https://arxiv.org/pdf/2611.19321)  

**Abstract**: Molecular dynamics is an extremely powerful approach for studying protein behavior, but it requires enormous computational resources to obtain an adequate sampling of possible conformations. Recently, thanks to successes in predicting three-dimensional protein structures, there has been an explosion of research exploring the opportunities offered by deep learning to address the problem of protein conformational landscape. In this work, an artificial intelligence system based on a variant of the Transformer architecture is introduced and applied to the problem of determining the ensemble of three-dimensional structures that a protein can access and their statistical weight. After being trained on a dataset of conformations of a particular protein obtained using classical all-atom molecular dynamics, this neural network is able to directly generate physically plausible protein conformations similar to those obtained both through molecular dynamics and experimentally. The ordered sequence of conformations obtained as output of the neural network rapidly converges towards the most populated regions of the protein's conformational landscape, suggesting that the system may have learned not only the protein's three-dimensional organization but also the thermodynamic weights of the various feasible conformations. These features, combined with its simplicity and low computational load, make this Transformer variant an ideal candidate for the development of systems capable of performing generative molecular dynamics. 

---
# SAR-TopoGuard: Training-Free, Speckle-Aware Topological Verification for Structure-Preserving SAR-to-Optical Translation 

**Authors**: Satabarto Sarkar, Debojyoti Ghosh, Subhranil Biswas, Kaustav Mandal, D. Sobya  

**Link**: [PDF](https://arxiv.org/pdf/2611.19316)  

**Abstract**: SAR-to-optical translators often synthesize detail that the radar measurement does not support, and pixel metrics alone cannot reveal it. We propose SAR-TopoGuard, a training-free inference-time layer for frozen generators: a stochastic sampler draws K candidates (Monte Carlo dropout for GANs, seed variance for diffusion), a speckle-aware verifier ranks them by gradient-magnitude correlation with the SAR input and a hallucination-frequency index, and a SAR-guided frequency gate admits high-frequency detail only where SAR edges support it. With all five hyperparameters frozen and three training seeds, we evaluate on 500 scene-disjoint SEN1-2 test patches, 500 patches of a held-out summer scene, and 200 cross-dataset SEN12MS winter pairs. The layer improves Pix2Pix by +0.060, +0.012, and +0.077 dB PSNR and by +0.0034, +0.0057, and +0.0135 SSIM, respectively (Wilcoxon p < 1e-6), and it improves CycleGAN and CUT on all three sets (+0.013 to +0.076 dB). It raises edge precision at the cost of edge recall (edge F1 -1.3% on the test set). On the latent diffusion model COP-GEN, it adds +0.33 dB (base) and +0.16 dB (Beta) on the winter set, the only set with dual-polarization SAR. An ablation shows that the gate produces the gain for MC-dropout GANs, whose candidates are nearly identical, whereas verifier selection contributes up to +0.26 dB for diffusion, whose samples differ. The layer has zero trainable parameters and adds 0.4 ms per patch beyond the K=5 generator passes. 

---
# HospitalBench: Diagnosing Where and Why Embodied Agents Fail on Long Hospital Task Chains 

**Authors**: Gaytri Jena, Kapil Wanaskar, Aman Chadha, Vasu Sharma, Vinija Jain, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2611.19274)  

**Abstract**: Long-horizon embodied tasks fail in ways that end-task success cannot see. HospitalBench is a diagnostic benchmark of ten hospital chores in NVIDIA Isaac Sim, each two unscored warm-up steps followed by a causal chain of five scored sub-tasks in one continuous simulation: the state one step leaves (an object pose, a choice, a fact seen once) determines what the next step must do, unlike benchmarks that place sub-tasks side by side. Vision-language agents act through skill-level tools from camera images. A scripted robot that makes only the calls a model gets is the solvability check of a chore. From each trajectory the benchmark reads where a chain breaks, what the agent did after a failed call, and how far its own position estimate drifts from the true one. This version reports first results only: 15 of 60 planned runs (three models at high effort, one run each on chores T1 and T10 from both start states and T3 from one), with scripted-robot proofs on three of the ten chores. Of the 15 runs, 2 passed all five sub-tasks. Runs still going after sub-tasks 1 to 5 were 15, 8, 4, 2, 2. Of 189 failed calls, 168 were followed by a different call and 111 by a call that did not fail. These are first results and not a full benchmark. We will release the scenes, task chains, scripted robot, evaluation harness and trajectories. 

---
# Beyond Alignment: Supervising the Generator for EEG-to-Image Reconstruction 

**Authors**: Sangmin Jo, Wootaek Jeong, Heung-Il Suk  

**Link**: [PDF](https://arxiv.org/pdf/2611.19232)  

**Abstract**: Visual brain decoding translates neural activity into human-interpretable images. While recent EEG-to-image reconstruction methods rely on aligning EEG signals with pretrained visual embeddings to condition an image generator, we empirically demonstrate that stronger cross-modal alignment do not translate into comparable gains in reconstruction quality. This indicates that improving the input condition alone is insufficient; the generative process itself requires explicit supervision. Although Representation Alignment (REPA) offers a solution by guiding diffusion transformers with pretrained vision encoder features, we find that EEG-conditioned generators struggle to follow these fixed targets. To address this mismatch, we introduce generation-derived supervision (GDS): an image-conditioned teacher that translates static image-encoder features into the internal representation space of a diffusion transformer. Our EEG-conditioned student generator closely tracks these dynamic targets during denoising, discarding the teacher at inference. Evaluated on THINGS-EEG and THINGS-MEG, our approach consistently outperforms conditioning-centric baselines, achieving state-of-the-art reconstruction fidelity. 

---
# Revisiting What Makes a Good Context Subset: A Sparse Latent Basis Perspective 

**Authors**: Cen-Jhih Li, Siddhant Chaudhary, Aditya Bhaskara  

**Link**: [PDF](https://arxiv.org/pdf/2611.19220)  

**Abstract**: Selecting context subsets is increasingly important in modern LLM inference, yet what makes a subset effective remains unclear. We study this question in the controlled setting of few-shot demonstration selection. We examine a hypothesis that demonstrations should collectively provide task-relevant information needed by a target query. Assuming that the query representation provides a useful proxy for these requirements, we formulate selection as sparse reconstruction and instantiate it with Residual Explanation Search (RES), a Matching-Pursuit-style diagnostic procedure. Across seven benchmarks and eight open-weight LLMs, no retrieval objective is universally best. RES outperforms MMR in 34 of 56 model--dataset pairs, with its clearest advantage on sentiment tasks, suggesting that query-conditioned collective reconstruction may capture useful complementarity that pairwise dissimilarity misses. Relevance-based retrieval remains strong on template-heavy reasoning benchmarks, while all retrieval methods underperform zero-shot prediction on average on StrategyQA, where a solution-reconstruction analysis is consistent with the query not exposing the required external knowledge. Target-model hidden states further show that retrieval objectives interact differently with representation spaces: middle-layer representations generally favor similarity retrieval, whereas last-layer representations are often stronger for residual- and diversity-based selection. These findings support a conditional, representation-dependent account of context subset quality rather than a universally superior selector. 

---
# Neural Dual Bounds: Valid-by-Construction JGLP Warm-Starts for MAP and Constrained MAP 

**Authors**: Akshay Vyas, Shivvrat Arya, Brij Malhotra, Tahrima Rahman, Vibhav Gogate, Nicholas Ruozzi  

**Link**: [PDF](https://arxiv.org/pdf/2611.19218)  

**Abstract**: We present Neural Dual Bounds, a calibration-free framework that produces deterministic upper bounds for MAP inference in probabilistic graphical models using a single neural-network forward pass. The framework trains a network to output the edge messages of a mini-bucket style upper bound on a given join-graph decomposition in a self-supervised manner by directly minimizing a smooth dual objective. The output of this network can be used at test time to warm-start a standard JGLP optimizer. Across diverse benchmarks, this warm start yields dual gaps that are orders of magnitude tighter than those from zero initialization, with the largest gains in the early-iteration regime. We also apply our framework for the constrained MAP (CMPE) task, where neural warm starts with test-time adaptation are comparable to the adaptive baseline on Bayesian networks and achieve the tightest dual gaps, by one to two orders of magnitude, on the grid and segmentation families. 

---
# On-policy distillation as a Backdoor Propagation Channel 

**Authors**: Zane Fu, Yanzhao Zheng, Ziting Qin, Qianshan Wei, Yi Yu, Gongli Xi, Jihuai Zhu, Baohua Dong, Hangcheng Zhu, Ruohui Huang, Gang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19207)  

**Abstract**: On-policy distillation (OPD) is emerging as an important paradigm for the post training of large language models, improving student models by leveraging supervision generated online by teacher models. However, whether this process also transfers latent security risks from teacher models to student models remains insufficiently studied. In this work, we reveal a new security risk in which an attacker first implants a conditional backdoor into the teacher model through supervised fine tuning (SFT), after which OPD can transfer the backdoor to the student model with only an extremely small proportion of trigger prompt exposure. More concerningly, we find that such backdoor transfer is substantially stronger under OPD than under conventional distillation methods, suggesting that OPD may provide an efficient channel for propagating backdoors across models. Our mechanistic analysis shows that the backdoor signal is primarily transmitted through anomalous differences in the conditional probabilities produced by the teacher model for triggered and clean inputs, and is gradually internalized into the internal representations of the student model during training. Furthermore, we investigate how to suppress such anomalous conditional probability signals at the source of distillation supervision and, based on this analysis, propose an OPD training method tailored to this conditional backdoor propagation setting to mitigate the risk of backdoor transfer induced by untrusted teacher supervision. 

---
# Isolating Perception for Multi-View Planning under Occlusion-Induced Partial Observability 

**Authors**: Kevin Nguyen, Ceyhun Efe Kayan, Feng Liu, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19179)  

**Abstract**: Vision-language models (VLMs) are beginning to saturate visual planning benchmarks, yet they fail to resolve depth ambiguity and degrade sharply with problem size under occlusion-induced partial observability (PO). We introduce BW-PO based on the Blocksworld domain, a controlled benchmark that reduces existing VLM planners and formalizers to near-zero performance, and propose VLM-as-perceiver, a paradigm that narrows the VLM's role to local perception while delegating cross-view evidence reconciliation and plan generation to external solvers. Across model-specific challenging subsets of BW-PO, this raises plan success from 1% to 97% for Gemma-4-31B, 0% to 95% for Qwen3.5-122B-A10B, and 3% to 84% for Gemini-3.1-Pro. Isolating perception in this way separates two bottlenecks that are otherwise entangled: reconciliation under depth ambiguity, and completeness of visual extraction under scene density. 

---
# Mean-field Token Steering 

**Authors**: Md Zulfiqur Haider, Asmaa Eldesoukey, Abhishek Halder, Yongxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.19163)  

**Abstract**: We consider stochastic token steering through a deep stack of transformer layers as a mean-field control problem for an interacting particle system on the unit sphere, where interactions arise from the attention mechanism. In the mean-field limit, the token population evolves via a controlled integro-partial differential equation where the feedback control acts as a corrective vector field guiding the population toward an aggregate configuration of tokens in the representation space. We deduce the minimum effort control to accomplish this task over a fixed time horizon interpreted as normalized layer depth. We show that the resulting controlled dynamics for the token population can be understood as a mean-field Schrödinger bridge adapted to the sphere geometry. Building on this, we propose a novel generalization of the dynamic Sinkhorn recursion to compute the consistent token evolution. Numerical experiments confirm that the proposed approach achieves exact steering of the token population. 

---
# MarineSparkLM-4B: A Lightweight Language Model for Marine and Offshore Engineering Built by Parameter-Efficient Adaptation 

**Authors**: Cheng Siong Chin  

**Link**: [PDF](https://arxiv.org/pdf/2611.19151)  

**Abstract**: Marine and offshore engineering text is spread across preprints, open-access journals, encyclopedias and institutional documents, and the amount of it is small by pretraining standards. We ask how far a lightweight language model for marine and offshore engineering can be taken with parameter-efficient adaptation on a single workstation-class device. Starting from the 4-billion-parameter OceanGPT-basic-4B-Instruct, we train two independent LoRA adapters, each updating 33.0 million of 4.06 billion parameters (0.81%). Training took 4 hours 40 minutes for the first adapter and 10 minutes for the second on one NVIDIA DGX Spark. The first adapter is trained by next-token prediction on a 61.6-million-token corpus covering ten engineering themes. The corpus was built from 13,879 source documents, of which 7,069 were retained after deduplication and a two-tier keyword filter that removes cross-domain lexical collisions. Its validation loss falls from 2.082 to 1.976 over 1,500 steps. Direct testing showed that this adapter could still drift off topic on a retrieval-grounded ballast-water prompt, so we trained a second adapter on 2,295 question-answer pairs (114 hand-written and 2,181 generated by the base model from corpus passages) with the loss restricted to answer tokens. On 20 held-out prompts scored by hand for accuracy, topicality and specificity (maximum 120), the plain base model, the continuation adapter, the instruction adapter and the instruction adapter with retrieval scored 42, 56, 84 and 88. Absolute accuracy stays low (14 of 40 without retrieval and 20 of 40 with it). We report the full pipeline, two negative results (a reverted keyword fix and an instruction run that overfit within 100 steps), and the limits of the evaluation, which uses one small prompt set, no inter-rater agreement measure and model-generated reference answers. 

---
# When the Mythical Man-Month Meets Agentic Web Systems: Structured Coordination for Scalable Development 

**Authors**: Zhiyao Wu, Zi Liang, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19140)  

**Abstract**: As agentic systems are increasingly deployed to support complex Web tasks, multi-agent architectures offer greater parallelism, specialization, and task coverage. However, increasing the number of agents does not necessarily improve system performance, as communication, synchronization, redundant work, and coordination overhead may offset the benefits of additional agents. This presents a fundamental scaling challenge for agentic Web systems: how can agent populations be organized to balance task performance and coordination costs? To address this challenge, we systematically investigate the relationship between agent population size, per-agent token consumption, and coordination overhead in collaborative agentic systems. Experiments on complex Web-oriented software development tasks reveal a non-linear scaling pattern: per-agent token consumption initially decreases as the agent population grows, but eventually exhibits diminishing returns. Motivated by these findings, we introduce the Surgical Team Model (STM), a structured coordination framework that organizes agents through role specialization, constrained inter-agent communication, and coordinator-mediated task aggregation. Experiments on full-stack Web application development tasks show that STM achieves competitive task completion quality and comparable token consumption relative to representative baselines, while incurring higher execution latency. 

---
# How Much Determinism Can Fine-Tuning Buy? Margin Regularisation Against Batch-Induced Token Flips 

**Authors**: Charan Teja Nagisettygari, Siva Raja Sindiramutty  

**Link**: [PDF](https://arxiv.org/pdf/2611.19135)  

**Abstract**: Greedy decoding at zero temperature is widely assumed to be deterministic, although the exact same prompt may yield different tokens depending on the number of other prompts sharing the batch for inference. The currently available solutions operate during inference, trading determinism for throughput. Can we train a model in such a way that fewer decoding steps would be susceptible? We characterize the perturbation empirically across six model families and three numerical precisions and observe two empirical facts about the measurement: the perturbation depends exactly on the representable precision (8.00x for bfloat16 vs float16, 8192.0x for float16 vs float32, in line with predictions based on mantissa bits) and the batch invariance happens accidentally at certain tensor shapes, allowing a practitioner to achieve zero nondeterminism merely by setting the sequence length. Then we apply hinge penalization of the model's top-1 / top-2 logit gap, scaled to twice the measured noise quantile. On GSM8K dataset it leads to a 35% reduction in token flips and divergence from 53.8% to 38.1%, at the cost of 1.3 - 2.9 accuracy points, increasing monotonically with the margin size. It does not affect the gap threshold sufficient for determinism: the logit gaps of surviving flips are unchanged (median 0.125 for all arms), as the hinge is a soft penalty and the largest hardware perturbation remains the same. We reject seven hypotheses, including a per-model viability diagnosis, holding on six models and falsified on the seventh. 

---
# Beyond Plain MCP: Do Decision Models and Code Mode Make Tool Use Cheaper and Faster? 

**Authors**: Abhishek Rai  

**Link**: [PDF](https://arxiv.org/pdf/2611.19123)  

**Abstract**: Model Context Protocol (MCP) gateways manage tool catalogs for language model agents using two primary paradigms: code mode where agents write scripts against tool stubs and decision layers which are cheap routers that preselect tool subsets. Using a controlled harness across 12 servers, up to 1,958 distractor tools, and three open-weight agents including Qwen, Gemma, and Nemotron, we present a preregistered evaluation on a sealed test split to determine when each approach succeeds. Code mode substantially outperforms plain MCP, cutting token usage by 70 to 90 percent while significantly boosting utility. Declaring return-shape hints in code mode stubs is a first-order variable, swinging success by up to 39.5 points for lower-performing agents, an effect larger than adding a router. The central hypothesis that a decision layer in front of code mode beats code mode alone is not confirmed, showing minor, statistically non-significant improvements for weaker agents and a slight cost for the strongest. We publicly release our harness, a finetuned 4B tool selection model, and all evaluation traces to support robust community experimentation on local AI privacy and agent architectures. 

---
# DecoEdit: Beyond Flat Control for 3D-Aware Unified Image Editing 

**Authors**: Xiaotang Gai, Jinbo Xing, Zijie Meng, Jiayi Yi, Jimin Xiao, Yixuan Qian, Yu Liu, Chaojie Mao, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19121)  

**Abstract**: While recent advances in multimodal image editing have achieved impressive results, current frameworks remain largely confined to ``flat" 2D controls (e.g., text prompts, bounding boxes, or masks). Precise 3D-aware geometric manipulation -- such as coordinating an object's location, physical scale, and 3D orientation while strictly preserving its fine-grained identity -- remains an elusive challenge, typically suffering from severe detail degradation or requiring expensive per-instance fine-tuning. In this paper, we present DecoEdit, an all-in-one image editing framework that goes beyond flat control by seamlessly integrating 1D textual, 2D spatial, and 3D volumetric inputs under a unified architecture. First, to power this unified model, we establish a data curation pipeline and curate a large-scale dataset, namely DecoDataset, containing 1.6 million samples, encompassing diverse multi-dimensional (1D/2D/3D) tasks and their complex compositions. Furthermore, we establish a comprehensive image editing benchmark named DecoBench to systematically evaluate multi-task editing performance. Finally, to address the fundamental trade-off between complex spatial transformations and identity preservation, we propose a novel decoupled pipeline: DecoEdit factorizes the editing process by utilizing structural guidance from SAM-3D to anchor the target geometry, scale, and spatial orientation, while concurrently leveraging a detail-preservation path to lock in high-fidelity textures. Extensive experiments demonstrate that DecoEdit achieves state-of-the-art results. Notably, on the challenging 3D spatial pose and orientation control tasks, DecoEdit significantly outperforms leading closed-source models such as Seedream 5.0 in both geometric accuracy and identity consistency. 

---
# Latency, Speech Recognition and Turn-Taking in Voice Agents for Spanish Phone Negotiation: An Externally Measured Comparison of Pipecat, LiveKit and GPT-Live 

**Authors**: Jovan Popoca Moran, Eduardo de Avila-Armenta, Raúl Alejandro Aquino-Ibarra, Jorge Octavio Nicolás-Díaz, Umberto Leon-Dominguez  

**Link**: [PDF](https://arxiv.org/pdf/2611.19118)  

**Abstract**: A voice agent that negotiates by phone must answer quickly, understand spoken amounts and cope with a caller who talks over it. We compared three architectures for such an agent, negotiating freight rates in Spanish: two cascades (speech recognition, language model and speech synthesis) orchestrated by Pipecat and by LiveKit Agents, and GPT-Live, an OpenAI speech model that listens and speaks without separate stages. A program played the carrier with the same 47 recordings of one speaker for all agents, without the telephone network, in 77 calls per architecture over five negotiation scenarios, and measured latency on the listener's side with a validated instrument. Median latency to the content response was 2,616.0 ms (Pipecat), 3,090.0 ms (LiveKit) and 2,444.8 ms (GPT-Live); the confidence interval of GPT-Live's 171.2 ms advantage over Pipecat includes zero. According to its own timestamps, GPT-Live began to answer 800 ms after the end of the line, close to the 0.8 s published by OpenAI; most of the remaining wait, about 1,367 ms and nearly constant, is the audio's round trip through OpenAI's service. A tool call added 0.9-1.1 s in every architecture, more than any difference between them, and the frameworks' self-reported latency fell short of the listener's wait by 9% (Pipecat) and 22% (LiveKit). Word error rates were low (6.03, 4.93 and 1.47%), but the error on amounts was 13.01% and 12.61% in the cascades, against 2.86% in GPT-Live. With default settings, Pipecat and LiveKit stopped at almost every backchannel and often took it as the carrier's answer: an "ok" spoken over the agent ended in a registered agreement in 3 of 15 Pipecat calls, 2 of 15 LiveKit calls and no GPT-Live call (descriptive counts, not tested). Latency should therefore be measured on the listener's side, and a cascade needs safeguards for spoken amounts and backchannels. 

---
# ST-GAT: A Leakage-Free Spatio-Temporal Graph Attention Framework for Robust Network Intrusion Detection 

**Authors**: Nikhil Parkar, Sanskar Pandey, Siddhi Prajapati, Suganeshwari G, Syed Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2611.19109)  

**Abstract**: Intrusion detection systems are a central defence in modern networks. Deep learning models report strong anomaly-detection results, but evaluations on standard benchmarks such as CSE-CIC-IDS-2018 are often inflated by two methodological problems: temporal leakage from overlapping sliding windows, and contamination from preprocessing fitted before the train/test split. Per-flow classifiers also ignore the spatio-temporal dependencies across sequences of network flows. We propose a Spatio-Temporal Graph Attention Network (ST-GAT) that represents traffic as a sequence of graph snapshots. It builds k-Nearest Neighbour (k-NN) edges within each window, applies multi-head Graph Attention Network (GAT) layers for spatial aggregation, and uses a Bidirectional Long Short-Term Memory (BiLSTM) network with temporal attention for sequence modelling. To prevent leakage, we use non-overlapping temporal windows (stride equal to window length) and fit all preprocessing on the training split only. We compare three class-imbalance strategies: inverse class-frequency weighting, dynamic balanced sampling, and few-shot metric prototyping. Across three random seeds, ST-GAT with inverse class weighting reached a Macro-F1 of 96.65% (+/- 0.87) and an accuracy of 99.14%. This is 14.3 percentage points higher in Macro-F1 than a per-flow Random Forest baseline (82.34%). Balanced sampling reduced Macro-F1 by 2.3 points (to 94.37%) relative to loss weighting. This suggests that aggressive resampling of small, noisy minority classes can hurt detection in heavily imbalanced traffic. 

---
# Can Agentic LLMs Write Task-Based Parallel Code? A Study of OpenMP Tasking, HPX, StarPU and TTG 

**Authors**: Linus Bantel, Anna-Lena Roth, Tim Thüring, Jonas Posner, Dirk Pflüger  

**Link**: [PDF](https://arxiv.org/pdf/2611.19105)  

**Abstract**: Task-based programming offers more expressive power than loop-level parallelism, but the runtimes providing it differ widely in API style, maturity, and the amount of public code to learn from. This paper asks how well LLM agents write C++ for four of them: OpenMP Tasking, HPX, StarPU, and TTG. Instead of single one-shot prompts, we use an agentic workflow in which the model edits files, compiles, and runs its kernels until it reproduces a reference output. Five locally hosted open-weight LLM implement seven HPC kernels, including matrix multiplication, Cholesky and LU factorization, in all four frameworks and a serial baseline to verify their understanding of the problem in general. Every kernel is evaluated independently for correctness at each concurrency level, strong scaling, and peak memory. Model capability dominates the outcome: three models produce correct, genuinely task-parallel kernels in 57-82% of the cases, while two models rarely compile their own code and almost never succeed. For the capable models, OpenMP Tasking and HPX are handled most reliably, StarPU is close behind, and TTG trails. The best generated GEMM and factorization kernels match or exceed the multithreaded OpenBLAS/LAPACK reference. However, correctness overstates parallel quality: several "parallel" kernels contain no parallelism at all; and serialized task graphs and unsuitable task granularity remain common. 

---
# Nahual: A Sequence Model for Language and Atoms 

**Authors**: Austin H. Cheng, Marcel Müller, Andreas Burger, Yeonghun Kang, Tsz Wai Ko, Marta Skreta, Luka Mucko, Ella Miray Rajaonson, Cher Tian Ser, Jérôme F. Gonthier, Alex Zook, Varinia Bernales, Alán Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2611.19072)  

**Abstract**: Generative models play a rapidly growing role in the design and simulation of molecules. The most widely used applications of generative models are in large language models (LLMs), owing to the natural flexibility of language for specifying and answering queries. However, generalist LLMs do not effectively handle large sets of 3D coordinates, whereas specialist generative models of 3D molecules cannot reach the controllability afforded by language. To bridge these capabilities, we propose Nahual, a decoder-only autoregressive diffusion model for any sequence of text and 3D molecules. To enable expressive autoregression on continuous coordinates, we propose next-token diffusion, whereby a causal transformer denoises each token while attending to the clean tokens before it, without increasing context size. We train Nahual on a curated set of twelve 3D chemistry tasks ranging from microsolvation to molecular conformer search to adsorption on metal surfaces, with associated evaluation metrics and baselines. The result is a single set of end-to-end-trained weights that simultaneously performs all tasks and surpasses state-of-the-art models specialized for crystal structure prediction and property-conditional molecule generation. Our results demonstrate the viability of scaling generalist decoder-only models for native multimodal understanding and generation across language and chemistry. Code is available at this https URL. 

---
# SPARC: Evidence-guided adaptive model composition for materials discovery 

**Authors**: Sissi Feng, Yuchen Han, Runze Zhang, Jiaru Bai, Brandon Sutherland, Ali Shayesteh Zeraati, Jason Hattrick-Simpers, Varinia Bernales, Alán Aspuru-Guzik, Yang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2611.19050)  

**Abstract**: Closed-loop experimentation requires more than selecting the next experiment. It must also determine which existing measurements and model-derived relationships are sufficiently supported to influence that decision. Here we introduce SPARC (Sparse Predictive Aggregation of Regression Components), a framework that continually revises this support as experimental evidence accumulates. Competing model families are evaluated through out-of-fold predictions, so that each model is tested on observations not used in its corresponding fit. Sparse posterior composition assigns model influence according to predictive contributions supported by the out-of-fold evidence, retaining complementary contributions from several model families. Candidate-dependent predictive dispersion and residual variability define trust, which modulates exploration alongside the predicted response. Independent diagnostics also prioritize existing measurements for prospective verification, with model-derived trust reported alongside them. Across eight published materials datasets, SPARC achieved higher $R^2$ than the corresponding literature-reported values in six, with relative gains of up to 77% without the density-functional-theory- or experiment-derived descriptors used by several published pipelines. In an automated alkaline oxygen-evolution-reaction campaign, 8 of 11 SPARC-selected experiments reached overpotentials below 300 mV, with a best value of 242 mV, compared with 18 of 125 initial experiments. Measurements prioritized for verification before repetition subsequently deviated by 76 mV on average, compared with 27 mV for matched controls. SPARC therefore turns experimental evidence into a basis for revising model influence, directing exploration and identifying measurements that warrant verification. 

---
# Hard or Mislabeled? Failure Screens Enrich Vision Benchmarks for Label Errors 

**Authors**: Haolin Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.19044)  

**Abstract**: Hard vision benchmarks often keep images that strong classifiers get wrong. Since a classifier is also marked wrong when the label is wrong, such failure screens may collect label errors. We measure this on ImageNet's twice-relabeled validation set with 103 classifiers. Among the 1,892 images that all 66 supervised classifiers get wrong, 61% have a wrong label, versus 11.1% overall. The fewer images a screen lets through, the more of them are mislabeled, along approximately one curve. Fit on one model family, this curve predicts the screens of two held-out model families with a mean error of 2.4 percentage points, and replays of ImageNet-D's and ImageNet-Hard's screens within 0.1 and 6.3 points. On new pools, ImageNet-V2 and CIFAR-10, enrichment at the strictest screens is stronger than the curve predicts: on ImageNet-V2, by human relabeling, 67% of the images that all 66 classifiers get wrong are mislabeled, against 46% predicted. Screens that let models choose among only some of the classes, as ImageNet-A's does (200 of 1,000), fall below the curve. By ReImageNet, 32% of ImageNet-Hard's ImageNet images have no correct label. ScreenAudit, which comprises an audit tool, a script that rebuilds corrected answer sets for the 5,464 decided ImageNet-Hard images (of 5,646) from ReImageNet's gated release, and per-image audits of 1,198 benchmark images, will be released later. 

---
# More Context, Mixed Results: Evaluating LLMs for Identifier Recovery and Code Deobfuscation 

**Authors**: Simon Queyrut, Valerio Schiavoni, Giuseppe Antonio Di Luna, Daniele Cono D'Elia, Leonardo Querzoni, Pascal Felber  

**Link**: [PDF](https://arxiv.org/pdf/2611.19043)  

**Abstract**: Source code is often obfuscated through identifier renaming, incentivizing reverse engineers to develop techniques for recovering meaningful names. To assess how LLMs compare with these techniques, we evaluate 4 traditional baselines and 3 LLMs (Qwen, OLMo and DeepSeek) on identifier detection and semantic recovery across 51 recent Java projects, 3 obfuscators, and 6 context/retrieval settings. LLMs generally outperform traditional techniques on both tasks, but benefit from different forms of context and settings: agentic retrieval yields the strongest detection results, reaching 91.56% $F_1$, while enclosing-class context generally improves semantic recovery at a higher token cost. Despite these gains, exact recovery peaks at 16.42% overall (under one-hop context) and 32.93% for field names (under repository retrieval), which are the easiest to recover. Class identifiers are the hardest to detect, with a mean detection $F_1$ of 48.48% across models, settings and obfuscators. More context is not always beneficial: larger methods hinder semantic recovery. Beyond identifier renaming, our experiments show that control-flow obfuscation degrades both tasks more than string encryption. Our findings provide useful ad hoc insights for reverse engineers choosing models and context strategies to deobfuscate code snippets. 

---
# Understanding the Role of Reasoning Data in Pretraining 

**Authors**: Jianhao Huang, Allan Zhang, Ali Payani, Baharan Mirzasoleiman  

**Link**: [PDF](https://arxiv.org/pdf/2611.19042)  

**Abstract**: Reasoning data expose models to multi-step reasoning, but reasoning traces may cover only a subset of the available facts. How should pretraining balance atomic facts with reasoning traces? We study this question by constructing a setting in which every fact provides an atomic training signal, while only a subset appears in reasoning supervision, enabling us to measure generalization to transitions observed atomically but never used in training traces. For a linear transformer, we derive the weight assigned to atomic training signals that minimizes held-transition squared error and show that its leading-order optimum increases with reasoning-chain length and decreases with the fraction of facts held out from reasoning traces. These trends persist in a full transformer layer with an MLP, softmax attention, and cross-entropy loss. We further validate the role of reasoning complexity and data coverage in two natural-language settings: controlled multi-hop reasoning and mathematical pretraining. Together, our results provide a principled view of how reasoning-chain structure and data coverage should shape the allocation of training data during pretraining. 

---
# The Critic Can Hurt: On the Safety of Critics in LLM-Guided AutoML Search 

**Authors**: Gaoyuan Du, Xiaoyang Liu, Rex Zhou, Dengdu Jiang, Rong Fu, Meng Ding, Defu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.19037)  

**Abstract**: LLM critics increasingly steer iterative search in agentic AutoML, prompt optimization, and self-refinement--evaluated for whether they help, but rarely for whether they can hurt. We isolate the critic's marginal effect through a four-arm audit and characterize the loop's tolerance through controlled fault injection. We show a naive 8B critic drives search below the no-critic random-search floor on 7 of 8 tabular datasets, losing up to 0.245 AUC in the worst case; the magnitude varies strongly with dataset, frontier, and budget. When: stronger critics attenuate the loss on the evaluated suite, without establishing a universal capability threshold. Why: controlled fault injection shows that non-directional errors are nearly costless under best-so-far search, whereas directional errors consume finite search opportunities; an idealized effective-sample calculation yields a necessary condition, $q(m-1) > \beta$, not a deployment certificate. How: a simple observable validation gate and a learned detector can reduce realized loss, but neither is guaranteed to preserve the full-budget no-critic baseline. We will release CRITFAULT, to our knowledge the first benchmark for detecting harmful critic feedback in AutoML. A critic is not a free addition: audit before trusting, and gate when capability or verification is insufficient. 

---
# Refinement vs. Replanning: Inference Compute Allocation in Robot Policies 

**Authors**: Alba Maria Tellez Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2611.19034)  

**Abstract**: A frozen chunked robot policy can spend deployment-time inference compute in two coupled ways: more work within each policy call to refine a predicted action chunk, or more frequent policy calls to replan from fresh observations. We study this allocation without updating model weights. Our evaluation procedure reports success with both nominal compute rate and complete attempt cost, selects comparisons on discovery tasks, and tests prespecified comparisons on held-out tasks. Comparisons added after inspecting the held-out results only summarize observed patterns. We demonstrate the procedure with one frozen SmolVLA checkpoint in synchronous LIBERO simulation, using denoising count as the refinement control and fixing chunk length at K=50. Arithmetic and latency characterize cost differently: at d=10, 82.1% of counted arithmetic is independent of denoising, whereas a descriptive affine fit to fixed-input host time assigns 17.7% to its intercept. At h=2, d=2 succeeds on 139/160 attempts versus 122/160 for d=5 (+10.6 percentage points; task-conditional 95% interval: +4.4 to +16.9), while using 5.9% less arithmetic per control step. Executing h=5 rather than h=2 at d=2 yields 138/160 versus 139/160 successes and 60.7% lower arithmetic per attempt; equivalence is not established. The case study shows why deployment adaptation must evaluate generation effort and feedback frequency jointly; it does not define a universal setting. 

---
# SurveyWeave: An Agentic Framework for Automated Survey Generation 

**Authors**: Franklin Zhang, Xiao Yang, Xin Luna Dong, Juhua Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19029)  

**Abstract**: Surveys help researchers keep up with fast-moving fields, but they take weeks to months to write. Automated survey generation (AutoSurvey) can alleviate this situation, but it is challenging to draw on hundreds to thousands of precisely scoped papers, choose among many valid structures, and stay faithful to its sources while offering insights. We present SurveyWeave, an agentic architecture that decomposes the problem into four tasks coordinated by three agents, each reviewing others' work and conducting revisions based on feedback. We also introduce WeaveEval, a five-dimension evaluation grounded in prior surveys. A comprehensive comparison shows that SurveyWeave outperforms Codex by 8 points and the strongest AutoSurvey baseline by 6 points. 

---
# HeteroWear: Sensor-Aware Self-Supervised Learning for Heterogeneous Wearable Activity Recognition 

**Authors**: Yasaman Mohammadpour, Pouya Shaeri  

**Link**: [PDF](https://arxiv.org/pdf/2611.19027)  

**Abstract**: Self-supervised learning for wearable sensing is usually confined to a single dataset or device family, leaving open whether representations transfer across heterogeneous sensor configurations and whether gains come from pretraining rather than architecture or richer sensor input. We present HeteroWear, a sensor-aware self-supervised framework pretrained on an eight-dataset wearable corpus (173 subjects, 358,261 windows) spanning inertial and physiological signals across body locations, devices, and sampling rates. HeteroWear separates within-sensor temporal modeling from cross-sensor relational modeling using sensor-context embeddings, temporal and cross-sensor Transformers, and latent cross-attention fusion. Training combines structured sensor dropout, cross-configuration consistency, and per-axis masked sensor modeling; missing sensors are omitted rather than reconstructed. Under leave-one-dataset-out pretraining and subject-disjoint evaluation on PAMAP2, WISDM, HHAR, HARTH, and UCI-HAR, HeteroWear improves mean macro-F1 from 0.688 to 0.724 over the identical architecture trained from scratch, with the largest gain on HHAR (0.823 versus 0.741), and achieves the highest macro-F1 on four of five benchmarks relative to single-accelerometer supervised baselines. Held-out-device evaluation yields a mean +0.066 macro-F1 improvement. Under 50% test-time channel removal, HeteroWear retains 73.8% of full-channel performance versus 64.1% for the same architecture trained from scratch without sensor dropout. Ablations show that structured sensor dropout primarily drives robustness, while cross-configuration consistency and sensor-context encoding contribute to transfer. 

---
# MemoryLane: Supporting Reminiscence Conversations Between Older Adults with Mild Dementia and Lay Caregivers Using a Simulated AI Agent 

**Authors**: Soumay Agarwal, Aaryan Gajula, Lingyao Li, Renkai Ma, Jennifer Martin, Krisstina Madan, Ellen Brown, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.19013)  

**Abstract**: Reminiscence Therapy (RT) is a non-pharmacological intervention that invites patients to discuss and review past life experiences and memories to improve their emotional and mental well-being. In practice, many reminiscence conversations are led by lay caregivers, such as spouses and adult children. While traditional RT often relies on personal photographs, curating them can be tedious and difficult. In addition, caregivers who lack shared life experiences with the patient may struggle to facilitate engaging reminiscence conversations. This paper presents MemoryLane, an AI-integrated tool that helps lay caregivers facilitate reminiscence conversations without curated photographs. MemoryLane supports a triadic conversation among the patient, the caregiver, and a simulated AI agent that acts as the patient's peer. A shared interface shows the agent and AI-generated images of the memories being discussed, while a private smartphone interface lets the caregiver control the behavior of the simulated AI agent and moderate the AI-generated images. A preliminary focus group study with five dementia-care stakeholders suggested the value of MemoryLane's AI agent as a peer companion and of AI-generated images, as well as the feasibility of having caregivers moderate images and control the AI agent. 

---
# Role-Separated Diffusion Transformers: Global Rule Isolation and Deterministic Sampling Manifold Compression 

**Authors**: Zhiguang Liu, Yi Shang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19009)  

**Abstract**: We present Role-Separated Diffusion Transformers (RoSeDiT), a new diffusion transformer architecture that isolates the global rule from the image tokens. RoSeViT showed in visual reasoning that a vanilla ViT spreads a shared rule over all $L$ tokens at an entropy of order $Ld$, whereas an enforced controller route reduces it to order $d$. In class-conditional DiT and SiT, the class and the noise level are explicitly conditioned by adaLN, so a RoSeViT-style controller channel on the ViT side is insufficient to establish the full role separation. By introducing the adaLN listening mechanism, which makes adaLN compute under the controller's command, RoSeDiT brings the entropy theorem and role separation into the latent diffusion domain. Under the same SiT training and evaluation protocol at the same level of parameter count, RoSeDiT surpasses SiT at all four model scales at 400K steps, and RoSeDiT-L/2 even surpasses SiT-XL/2's 17.2 FID with 15.34 and 32% fewer parameters. RoSeDiT-XL/2 surpasses SiT-XL/2's 7M-step FID of 8.3 at 1.4M steps with 7.98, using one-fifth as many training steps. We further observe that models with the same denoising loss can have a large FID gap, which indicates a misalignment between the denoising objective and the image manifold. We formalize this phenomenon into Deterministic Sampling Manifold Compression (DSMC), a two-stage training schedule with which RoSeDiT-XL/2 surpasses SiT + REPA's 4M-step FID of 5.9 at 2.4M steps with 5.84. 

---
# Agent-ZTA: A chain-Aware Zero Trust Model for Bidirectional Security-State Propagation and Runtime Containment in Multi-Agent Delegation 

**Authors**: Asadullah Abdullah Khan, Toqeer Ali Syed  

**Link**: [PDF](https://arxiv.org/pdf/2611.18999)  

**Abstract**: Multi-agent systems increasingly delegate tasks and permissions across chains of autonomous agents that invoke protected tools. Although agent identities and delegated credentials may remain valid throughout a workflow, the security condition of any agent along the chain can deteriorate during runtime execution-either through upstream compromise or downstream ingestion of untrusted inputs. Authorization models based solely on the immediate caller and its static delegated scope fail to account for these mid-execution state changes. This paper develops Agent-ZTA, a chain-aware Zero Trust authorization and containment model that links runtime security-state changes across multi-agent delegation graphs. The model distinguishes an agent's path-specific delegated authority bound from the effective permissions currently exercisable under trusted policy. It defines an architecture featuring independently enforced policy decisions, authenticated delegation provenance, a dynamic chain-state view, and explicit fail-safe handling of missing or stale authorization signals. Furthermore, Agent-ZTA defines bidirectional security state propagation, allowing downstream security events to affect relevant upstream workflow states, and specifies adaptive branch-scoped containment through restricted permissions and a sandbox ("jail") execution mode where an isolation boundary is available. The deterministic decision procedure, five stated security properties, and worked authorization traces show how accepted security-state changes constrain affected operations while preserving original delegation bounds and separating unrelated workflow branches under the defined policies. Finally, the paper states its trust assumptions and the exact scope of its security properties. 

---
# Model Maps: A Documentation Framework for Tracing Foundation Model Ancestry and Understanding AI Entanglement 

**Authors**: Rachel Hong, Madeleine Grunde-McLaughlin, Sarah Cen, Jevan Hutson, Jamie Morgenstern, Tadayoshi Kohno  

**Link**: [PDF](https://arxiv.org/pdf/2611.18998)  

**Abstract**: The development of foundation models incorporates pre-existing AI artifacts like models, datasets, and benchmarks. These relationships, as part of the broader AI supply chain, produce "entangled" webs in which properties of artifacts upstream can influence the outputs of a foundation model downstream - a phenomenon known as AI entanglement. Without visibility into these connections, a black-box model can inherit upstream properties that result in security vulnerabilities, hallucinations, or biased predictions, yet developers are left with no recourse to identify or mitigate the root issue. To address these concerns, we introduce model maps as a conceptual documentation framework for tracing model ancestry that builds on previous documentation frameworks like ecosystem graphs, datasheets, and model cards. A model map for a given foundation model is a detailed knowledge graph that represents the relationships between artifacts that influence the resulting foundation model. Our framework incorporates a taxonomy of 13 dependency mechanisms including model fine-tuning and synthetic data generation. We apply this framework to construct model maps of 15 closed and open-weight foundation models to illustrate the complexity of AI entanglement in current model development practices. From our model maps, we record 2,778 unique dependencies and trace artifacts that are located up to 9 hops away from a resulting foundation model. Our findings suggest that AI transparency requirements should extend beyond individual models and training-data disclosures to encompass upstream dependencies, which would enable more effective risk assessment, auditing, and accountability across the AI supply chain. To help developers, regulators, and researchers understand and analyze the interactions that can influence the behavior of a foundation model, we release these these model maps at this https URL. 

---
# Large language models as oncology world models 

**Authors**: Julien Vibert, Jakob Nikolas Kather  

**Link**: [PDF](https://arxiv.org/pdf/2611.18994)  

**Abstract**: Large language models (LLMs) can synthesize cancer patient cases, but their internal representation of oncology, or world model, has not been explored systematically. By only specifying tumor type, we show that LLMs tend to generate similar patients per cancer, with on average 81.0% to 93.2% of a cancer's patients sharing the most frequent value of each baseline feature. By systematically varying patient features in 12 clinical scenarios, we show that they follow biomarker and treatment logic in 29 of 36 model-scenario assessments, against 1 of 12 when previous treatment or frailty matters. Continuing real patient trajectories, an LLM predicted 89.9% 12-month survival, and 79.2% with similar patients as examples, against 72.2% observed. To represent these trajectories, we developed the Clinical State Space, a dynamic representation framework that we also applied to real patients. Despite the limits we show, improved world models may support digital twins, which our framework can evaluate. 

---
# Probing Where Surprise Happens: Localized Violation-of-Expectation Evaluation of Video World Models 

**Authors**: Yanbin Xu, Ming bo Cai  

**Link**: [PDF](https://arxiv.org/pdf/2611.18979)  

**Abstract**: Violation-of-expectation (VOE) benchmarks evaluate intuitive physics in video world models by testing whether physically impossible events produce greater prediction errors than matched possible events. However, conventional VOE evaluations aggregate prediction errors across space and time, making it unclear whether successful discrimination reflects model's understanding of physical violations at the locations and time where they occur. To address this limitation, we introduce localized VOE evaluation, a framework that measures prediction error differences between possible and impossible events within spatial regions and temporal intervals associated with physical violations. We compare global, temporally localized, and spatiotemporally localized evaluations of V-JEPA models pretrained on developmental egocentric videos or large-scale natural video datasets. Our results reveal that global and localized judgments can diverge in both directions: global aggregation can either obscure correct localized discrimination or produce apparent discrimination without corresponding localized evidence. During continued pretraining on developmental egocentric videos, agreement between global and localized judgments changes non-monotonically, while confidence in localized judgments increases despite modest improvements in accuracy. Together, these findings establish localized VOE evaluation as a complementary diagnostic for assessing intuitive physics in video world models and demonstrate that global prediction errors do not necessarily reflect the spatiotemporal evidence of physical violations. 

---
# Context Matters: Improving LLM-Based Code Summarization with Hierarchical Information 

**Authors**: Alexandre Petrachini, Breno Bernard Nicolau de França, Guido Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2611.18964)  

**Abstract**: Source code documentation plays an important role in software quality activities, such as code review, program comprehension, maintenance, and bug fixing. However, it is often fragmented across individual code elements and focuses on local implementation details, providing little information about their surrounding context. This limits both developers' understanding of software systems and the effectiveness of Large Language Models (LLMs), whose reasoning is constrained by finite context windows. Enriching documentation with hierarchical context has the potential to improve software comprehension and support downstream, quality-related tasks for both developers and LLM-based tools. In this paper, we present an in silico study that showcases the impact of augmenting context on the generation of code-level documentation, using a variety of LLMs. By varying the function, class, and module hierarchy visible to the LLMs, we assess whether this variation produces a significant difference in the resulting code documentation compared with the existing documentation, which serves as a proxy for developer intent. Our results show that Documentation-Aware prompts with class or module-level context consistently yield summaries that are closest to the original documentation. Adding surrounding documentation yields statistically significant improvements over the baseline across models, with effect sizes ranging from negligible (DeepSeek) through weak (GPT-4o-mini) to medium (Gemini). We restrict the study to this immediate hierarchical context; call-graph traversal and cross-module dependency context fall outside our scope and are left to future work. These findings provide a clear path to improve automated documentation tools and enhance LLMs' ability to reason about codebases for downstream tasks, such as code retrieval. 

---
# From Language-Specific to Cross-DSL: Semantic-Role-Guided Model Completion with LLMs 

**Authors**: Meriem Ben Chaaben, Théo Giraudet, Benoit Combemale, Houari Sahraoui  

**Link**: [PDF](https://arxiv.org/pdf/2611.18952)  

**Abstract**: The rapid emergence of large language models (LLMs), trained on vast and diverse corpora, has positioned them as a promising source of domain knowledge to assist modelers. However, leveraging LLMs for model completion remains challenging: suggestions must respect both the syntax of a domain-specific language (DSL) and the intended domain semantics, while remaining consistent with an evolving model. This paper presents RAFiC, an LLM-based approach for iterative domain model completion that generalizes across DSLs. It drives completion using metamodel information to ensure syntactic validity and enriches prompts with semantic-role annotations to better capture the intended modeling role of DSL elements. The approach provides context-aware, incremental guidance during model construction. We evaluate RAFiC through an ablation-based comparative study over three DSLs and two scenarios per DSL to isolate the contribution of syntactic and semantic guidance. Across these six scenarios, semantic role guidance improves contribution rate in every case, and RAFiC improves completion effectiveness over weaker prompting configurations in most of them. The largest gains occur for DSLs that are less represented in LLM training data. In these cases, iterative guidance leads to earlier and more stable accepted suggestions, while semantic role guidance improves their relevance. 

---
# Effects of Conventional, Adaptive, and AI-Powered Upscaling on the Legibility of License Plates 

**Authors**: Hany Farid, Joshua D. Bucci, Emily A. Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2611.18949)  

**Abstract**: Small details in digital images -- a face, a street sign, a license plate -- are routinely upscaled in hopes of making them interpretable. Upscaling options now range from conventional interpolation, whose operations are spatially invariant and fully explainable, to AI-powered methods that synthesize detail potentially absent from the source. Here, we ask which of these techniques most reliably helps human readers. In two online experiments, we measured legibility of 255 US license plates ranging in resolution from 128 to 8 pixels tall, as well as their legibility after upscaling by eight different methods. Legibility of the un-upscaled plates collapsed from ceiling to floor across a narrow band of plate heights (32 to 10 pixels). Conventional upscaling techniques (bilinear, bicubic, and Lanczos interpolation) gave modest gains of about 5-9 percentage points within this range. Nearest-neighbor interpolation, however, was harmful, rendering unreadable many plates that were readable without any upscaling. Advanced methods diverged: SwinIR, an open-source AI enhancer, raised performance by up to 38 points relative to no upscaling, while an adaptive denoiser-based image prior and commercial AI upscaler tended to impair legibility. We found no strong evidence of consistent character hallucinations across methods. Decomposing the task, conventional interpolators mainly helped readers count characters, whereas SwinIR alone made the characters themselves more identifiable. 

---
# The Human Side of ML Engineering: Analyzing the Collaboration and Communication Skills in ML Job Postings 

**Authors**: Aidin Azamnouri, Phuong Thao Le, Justus Bogner, Stefan Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2611.18915)  

**Abstract**: Studies of Machine Learning (ML) practice highlight the importance of combining technical expertise with interpersonal capabilities, particularly collaboration and communication. However, evidence on how ML job postings express these skills is scarce, and prior work largely relies on keyword-based analyses that overlook implicit expression. We analyze demand for these skills in ML job postings across regions, industries, and seniority levels, and compare it with software engineering (SE) roles. We collected 1,000 postings (500 ML, 500 SE) from five continents via a semi-automated pipeline. For annotation, we synthesized a taxonomy of collaboration and communication sub-categories from prior work and used an LLM as a third annotator alongside two humans, enabling detection of context-dependent and implicit skill references. Collaboration appeared in 88.6% of ML postings and communication in 72.2%, far above prior keyword-based estimates for AI roles. Roughly one in five collaboration and one in four communication requirements were expressed only implicitly. Decomposing these requirements into the taxonomy's sub-categories reveals what aggregate figures conceal. Collaboration is demanded predominantly as integration across role boundaries rather than team membership. Communication is mostly required in generic terms, with fewer than half of the postings requiring it naming whom the candidate must communicate with. ML and SE postings demand interpersonal competence at statistically indistinguishable levels but orient it differently, SE toward team practices and ML toward presentation and audience-facing interaction. Within ML, the advertised role separates interpersonal profiles more sharply than any other dimension, including the contrast with SE. 

---
# Lasso: Towards Sustainable AI-Generated Media Detection with Intel QAT On-Chip Compression 

**Authors**: Darsh Asher, Joshua Kalyanapu, Kaushal Mhapsekar, Bita Aslrousta, Amna Hassan, Debayan Sengupta, Samira Mirbagher Ajorpaz  

**Link**: [PDF](https://arxiv.org/pdf/2611.18895)  

**Abstract**: Synthetic text, images, and video are rising. Software-only detectors are power-hungry, too slow, and costly, and do not scale with demand. When software cannot scale, historically hardware often comes to the rescue. We perform the first reverse engineering of a compression accelerator--Intel QuickAssist Technology (QAT) microarchitecture and--find different amount of time on each, a novel data structure-dependent property. Using this information, we then build Lasso, a low-cost, high-performance detector for AI-generated content that turns the accelerator into a measuring instrument.
For text detection, Lasso achieves 100% accuracy on original DetectRL text and retains 92% average accuracy after paraphrasing, compared with 78% for Binoculars and 55% for DetectGPT. Across original, paraphrased, and perturbed text, it achieves 82% average accuracy, outperforming all five baselines. With topic, length, and formatting matched, accuracy reaches 95% for instruction-tuned generators, including when the tested generator is unseen. On RAID, accuracy averages 82% when trained on one generator and tested on another, and 74% across eight held-out domains. Detection takes 0.74 microseconds, a speedup of at least 83,784x across five baselines. At the same time, Lasso consumes over 99% less energy per document than both Binoculars and DetectGPT.
At the same time, Lasso detects AI-generated videos, software code, and images with 67.5%, 70% and 78% accuracy, respectively, with low latency and power opening avenues for practical real time and edge device detection of multimodality AI artifacts for the first time. No other compared technique supports detection across all four modalities: text, video, images, and code. 

---
# Benchmark Factory: Agentic Benchmark Creation via Dataset Curation and Label-Free Quality Anchors 

**Authors**: Shervin Shahidi, James Poon, Hang Xie, Kenneth O. Xu, Robin Håkanson, Vikram Nair, Jens Pillgram-Larsen, Nachiappan Nagappan  

**Link**: [PDF](https://arxiv.org/pdf/2611.18890)  

**Abstract**: Teams deploying coding agents need benchmarks built from their own code, but curating and validating them by hand is resource intensive, and an unvalidated benchmark can mislead silently. We present Benchmark Factory, an autonomous meta-agent loop with human checkpoints: an LLM agent writes sourcing queries over landed production code changes, runs models of known relative strength on each candidate, diagnoses why a candidate fails, and repairs it by editing its queries, prompts, and configuration until it clears quality anchors that need no per-task human labels. We used Benchmark Factory to build benchmarks for five developer cohorts at our organization. Engineering raters scored a 5% sample 2.3 for task quality and 2.6 for grader verdicts on a 1-to-3 scale. On configurations never used to build them, the benchmarks ranked Claude Opus~4.7 at high reasoning effort above low effort (60.6% vs. 54.4% of tasks passed, $p=0.002$) and GPT-5.5 above GPT-5.4 (65.8% vs. 60.5%, $p=0.004$); 9 of the 10 per-cohort comparisons pointed the same way. A deliberately trivial benchmark failed the anchors; five broken graders failed them, even under a relaxed rule fixed in advance that the real grader passes; and with model labels shuffled so that no true ordering existed, random redraws of a candidate passed in at most 1.4% of simulations. The anchors are necessary, not sufficient, conditions for a valid benchmark, designed to focus human review on what the loop cannot repair. 

---
# We Have All the Cards: Securing Agent Cards at the Operational Trust Boundary of the Agent2Agent Protocol 

**Authors**: Miguel A. Guirao Aguilera, Satyajayant Misra, Efren Lopez-Morales  

**Link**: [PDF](https://arxiv.org/pdf/2611.18887)  

**Abstract**: AI agent interoperability protocols enable orchestrators to discover and invoke external agents using machine-readable metadata. In the Agent2Agent (A2A) protocol, Agent Cards contain attacker-controllable natural-language fields that may influence agent selection before a provider is invoked, creating a pre-invocation trust problem. We characterize this discovery-time attack surface through a taxonomy of ten Agent Card manipulation attacks and present SECCARD, a layered validation framework combining protocol checks, deterministic analysis, and LLM-based semantic reasoning. We measure the emerging public A2A ecosystem by probing 36.1 million Internet domains and collecting 310 deployed Agent Cards. Two annotators independently label the corpus with 96.8% agreement. We then construct a 660-card real-card perturbation benchmark with matched benign controls and a 260-card synthetic adversarial benchmark spanning all ten attack categories. On controlled attacks injected into real Agent Cards, the strongest semantic judge achieves 97.0% security recall at a 1.7% false-positive rate. Across nine LLM judges on the synthetic benchmark, security recall ranges from 25.8% to 92.3%, demonstrating substantial model-dependent variation. These results show that SECCARD effectively identifies Agent Card manipulations spanning all ten discovery-time attack classes while preserving benign Agent Cards, but also reveal that its effectiveness depends critically on the choice of the underlying LLM judge. 

---
# Label-Free Detection of Structural Bias in LLM-Based Recommender Systems via Randomized Paraphrasing: The Case of Verbosity 

**Authors**: Carlos Peláez-González, David Herrera-Poyatos, Francisco Herrera, Andrés Herrera-Poyatos  

**Link**: [PDF](https://arxiv.org/pdf/2611.18878)  

**Abstract**: Large language models (LLMs) are increasingly deployed as recommenders in e-commerce and conversational shopping systems, where they select a product from a set of retrieved candidates described in natural language. Their decisions should depend on relevance to the user query, but may also be influenced by surface-level properties of the descriptions, such as their length. Detecting such structural biases is difficult because relevance labels are rarely available. We propose a label-free framework to audit LLM-based recommender systems for structural bias. Candidate descriptions are rewritten by randomly assigned paraphrasing models and presented in random order. We prove that, if the recommender's choice is invariant to these rewritings, every paraphraser is selected with equal probability, yielding a statistical test that requires no relevance labels. A four-step statistical pipeline tests this property and quantifies the contribution of each description feature while controlling for query-product relevance. We apply the framework to verbosity bias on 20{,}000 product queries and nine recommender LLMs from five model families. After controlling for the measured relevance signals, description length has a positive effect on selection in all nine models, increasing the odds of selection by up to 38% per standard deviation of length. Relevance remains the primary driver of recommendations, while verbosity acts as a consistent secondary signal whose strength depends on the model family. Since the framework needs only black-box access and no annotations, it can be run as a routine audit of deployed shopping assistants, and we provide an auditing protocol with cost and sample-size guidance. 

---
# The opposing informational effects of AI trading 

**Authors**: Hajime Shimao  

**Link**: [PDF](https://arxiv.org/pdf/2611.18872)  

**Abstract**: Better artificial intelligence (AI) can improve individual forecasts without making prices better at revealing new information. In a competitive market with resource-constrained forecasting, more capable algorithms can substitute analysis of public records for private observations, reducing the private information aggregated into prices. The same trades can convey sophisticated public-data analysis to investors unable to perform it themselves. We establish conditions under which these effects coexist and the private benefit of further computation falls. Numerical equilibrium calculations and finite-data neural-market experiments exhibit corresponding patterns. Out-of-sample earnings forecasts and controlled language-model tasks show related patterns of information substitution. In the neural experiment, wider adoption increases the predictive value of price for limited observers, reduces it for advanced observers, and lowers the benefit of upgrading by 44\%. Retraining by investors who did not upgrade can amplify the loss of private-information precision. The informational consequences of AI trading depend on what prices reveal and who can use them. 

---
# Self-Improvement Memory Enables Continual Recursive Self-Improvement 

**Authors**: Xin Yu, Fei Xia, Jiaqi Pan, Youbao Tang, Chengzhi Liu, Adams Yu, Sicong Jiang, Yanfei Chen, Lingzhou Xue, Ting Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18856)  

**Abstract**: Harness optimization is often studied with a fixed policy model and benchmark. When either changes, the optimizer must decide which parts of an evolved harness to reuse or revise, but the harness alone may not explain why earlier edits were made or how they performed. In this work, we study Continual Recursive Self-Improvement (CRSI), a setting for continuing existing harness-optimization methods across compatible benchmarks or policy models. We propose Self-Improvement Memory (SIM) for the harness optimizer, rather than the task-solving policy. SIM groups edits by recurring harness problem and records their rationale, predicted and observed outcomes, and evaluation context. New attempts extend this history as optimization continues. In our experiments across four domains, SIM achieves higher final scores than harness-only and root-cause-only continuation in cross-dataset and pairwise cross-model comparisons. Across ten independent source runs in coding, paired continuations with SIM improve final scores by an average of 2.36 points over harness-only continuation ($p=0.003$). 

---
# KUDEX: A Zero-Shot Decompositional Framework for Logical Analysis in Natural Language 

**Authors**: Zixu Zhao, Tianhang Zhang, Tong He, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18851)  

**Abstract**: While Large Language Models (LLMs) excel at generating fluent text, they often struggle with multi-step logical reasoning with inferential connections among statements. We introduce KUDEX (Knowledge-Upward-Downward Reasoning Extension), a novel zero-shot, decompositional framework for analyzing and verifying logical relationships in natural language. Unlike traditional approaches that rely on surface-level entailment checks, KUDEX leverages LLMs to decompose reasoning into atomic primitives and constructs transparent reasoning graphs that serve both as execution traces and audit logs. Our framework includes a general prompt adaptation pipeline that adapts to diverse reasoning tasks without additional fine-tuning or domain-specific pipelines. Through comprehensive experiments on benchmarks across four domains, we demonstrate that KUDEX provides superior performance compared to existing methods, offering a more granular and robust framework for assessing logical relationships between statements. 

---
# What to Practice Next? Latent Profile-Guided Task Selection for Skill-Evolving LLM Agents 

**Authors**: Qijin Zhang, Hanzhen Wang, Yan Huang, Zhongxiang Dai, Shuang Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18818)  

**Abstract**: Task selection shapes the experiences available for skill evolution in LLM agents, yet fixed practice schedules overlook differences in skill requirements and evolving task value. Most existing methods, however, rely on fixed schedules or scalar measures of progress, leaving nonstationary skill demands unaddressed. We propose SPG-Select, a framework for allocating limited practice budget using a multidimensional profile inferred from binary outcomes via MIRT. The framework predicts surrogate progress, defined as the difference between consecutive profile estimates, and prioritizes progress in dimensions with larger estimated proficiency gaps. Our SPG-Select-R policy combines these predictions with uncertainty-based exploration and periodic restarting to accommodate nonstationarity. We evaluate SPG-Select on ALFWorld and WebShop with three skill-evolution backends. In the reported runs, SPG-Select-R achieves higher mean held-out terminal success than uniform sampling in all six backend-benchmark combinations, with gains of 1.49-13.18 percentage points on ALFWorld and 0.93-3.13 points on WebShop. Under a predictable linear surrogate model and boundedness assumptions, we establish a high-probability dynamic pseudo-regret bound for surrogate task allocation in terms of parameter path variation. 

---
# Foundation Model Fine-Tuning with Attention-based Multi-View Fusion for Multi-Task Cardiac Function Assessment 

**Authors**: João Freitas, Warre Snaet, Jan D'hooge, Sandro Queirós  

**Link**: [PDF](https://arxiv.org/pdf/2611.18816)  

**Abstract**: In cardio-oncology, left ventricular ejection fraction (LVEF) is the standard biomarker for monitoring cardiac function during cancer therapy. However, its estimation from echocardiography is highly operator dependent, and clinically meaningful declines are often detected only after myocardial injury has progressed, motivating automated methods for standardized assessment and earlier detection of LV dysfunction. To address this need, we propose a multi-task framework integrating cardiac cycle-based temporal standardization, multi-view spatiotemporal feature extraction, and joint optimization for LVEF estimation and LV dysfunction classification. Apical four- and two-chamber echocardiographic sequences are temporally standardized by sampling a clip over an estimated cardiac cycle before being processed by the EchoPrime foundation model, which is fine-tuned end-to-end within the proposed architecture. View-specific features are fused through an attention-based aggregation module to generate a patient-level embedding, from which two task-specific prediction heads estimate continuous LVEF and binary LV dysfunction. To broaden the LVEF distribution encountered during training, samples from the MIMIC-IV-ECHO dataset are incorporated using a balanced sampling strategy. Evaluated on the EchoRisk-MICCAI Challenge, our method achieves an MAE of 4.003 for LVEF estimation (Task 1) and an AUC of 0.764 for LV dysfunction detection (Task 2) on the validation set, demonstrating the effectiveness of foundation model fine-tuning with attention-based multi-view fusion for automated cardiac function assessment in cardio-oncology. 

---
# RescueVLM: Model-Agnostic Vision-Language Framework for Rescue Drone Imagery 

**Authors**: Niklas Voigt, Tim Sibum, Julien Meine, Niklas Digakis, Hartmut Surmann  

**Link**: [PDF](https://arxiv.org/pdf/2611.18813)  

**Abstract**: A single unmanned aerial vehicle (UAV) mission over an incident site produces hundreds of images, far more than a rescue commander can review under time pressure, and human screening of destruction imagery carries a documented psychological cost. The two obvious remedies both fail in search and rescue (SAR) practice: cloud vision-language model (VLM) services are excluded because the imagery must not leave the incident network under GDPR and German civil-protection (BOS) rules, and fine-tuned specialist detectors age quickly while retraining and re-certifying them exceeds most SAR organizations resources. We present RescueVLM, an open-source, training-free and model-agnostic triage tool built on SketchVLM: a general-purpose open-weight VLM runs fully on-premise and is swapped for a newer model through a simple dropdown selection, so triage quality grows with the underlying models because all task knowledge lives in prompts and an output contract rather than in trained weights. One natural-language query over a mission folder returns a ranked shortlist with per-image justifications, editable SVG overlay annotations, and a GPS map of the hits. In a qualitative evaluation on German fire-brigade field data from three real deployments and a live fire exercise, a single query condenses an entire survey mission into a handful of annotated frames that a commander can review at a glance. 

---
# CARET: Clinical Assertion-Aware Residual Calibration for Medical Image-Text Alignment 

**Authors**: Yifan Jin, Tao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18811)  

**Abstract**: Medical vision--language models align radiographs with free-text reports, but radiology language distinguishes explicit findings, negation, uncertainty, and omission. Treating these report states as interchangeable during feature adaptation can change image predictions without supporting visual evidence. We introduce CARET, a parameter-efficient image-side adaptation method that retains pretrained image and text encoders and uses paired positive and negative clinical statements to define a low-dimensional semantic correction space. A bounded residual adjusts image features along these directions, while assertion-aware supervision distinguishes explicitly present and absent findings from uncertain or unmentioned ones. At inference, the adapter requires only an image; textual references remain fixed. In zero-shot classification, CARET achieves 67.80\% accuracy on CheXpert 5$\times$200 and 65.20\% on RSNA. These scores exceed the published CLEFT-LoRA results by 1.06 and 0.27 percentage points, respectively. Comparisons to literature baselines are informative but not controlled for differences in pretraining and evaluation protocols. Ablations across eight components identify the four-state distinction and assertion objective as major contributors, with task-dependent effects for nonmention preservation and pair alignment. 

---
# LEAP-LLM: Preserving lead-time structure for question-conditioned ECG interpretation 

**Authors**: Aakash Agarwal, Yidan Shen, Xin Fu, Renjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18809)  

**Abstract**: The electrocardiogram (ECG) records cardiac electrical activity across multiple leads and is used for detecting cardiac abnormalities. Recent ECG-language models enable large language models (LLMs) to interpret cardiac signals and answer clinically relevant questions. However, existing ECG-language models typically encode the ECG independently of the clinical question, forcing the ECG representation to retain broad and potentially irrelevant signal information. Moreover, these models often aggregate information across leads which can obscure lead-specific details needed for a particular question. To tackle these limitations, we introduce LEAP-LLM, an ECG-LLM architecture for Lead-aware ECG Alignment with Prompt conditioning. LEAP-LLM uses the clinical question to guide ECG feature extraction while preserving explicit lead and temporal structure. Task-trained instruction representations distill information from the question and guide two complementary branches. The Lead branch captures question-relevant temporal patterns within each ECG lead, while the Temporal branch summarizes relevant information across leads at each temporal position. Together, they provide the LLM with compact, question-relevant ECG information while preserving lead and temporal structure. LEAP-LLM reduces modality-token count by 89.0% and prefill FLOPs by 7.2x on average relative to other ECG-LLMs baselines. Across four ECG benchmarks, LEAP-LLM achieves better performance, including a 3.4-percentage-point improvement in G12EC over the strongest prior ECG-specific method. We further show that the model adaptively emphasizes the clinically relevant anatomical lead groups for different cardiac abnormalities, with lead importance patterns consistent with established diagnostic knowledge, providing interpretable evidence that the predictions are supported by clinically meaningful ECG structure. 

---
# Accelerating Variational Cone-Beam CT Reconstruction with Learned Warm Starts 

**Authors**: Héloïse Dudoignon, Clément Delmas, Laurent Facq, Clair Poignard, Baudouin Denis de Senneville  

**Link**: [PDF](https://arxiv.org/pdf/2611.18801)  

**Abstract**: Variational reconstruction provides a rigorous framework for cone-beam computed tomography (CBCT) by combining an explicit acquisition model with image priors. However, iterative optimization remains computationally demanding, particularly under severe noise and sparse-view conditions. Although learning-based methods achieve high reconstruction quality, they often replace the original variational formulation with learned optimization schemes or nonlinear equilibrium models. We propose a neural network-guided warm-start framework that accelerates variational reconstruction while preserving its original formulation. Projection data are first reconstructed using the Feldkamp-Davis-Kress (FDK) algorithm, and the resulting image is refined by an image-domain neural network. The network output is used solely to initialize the subsequent model-based optimization, leaving the forward model, data-fidelity term, regularization, and optimization scheme unchanged. A pre-regularization step further improves compatibility with the variational priors. Experiments on pelvic CT images from the SynthRAD2025 challenge show a more than threefold reduction in reconstruction time while maintaining comparable image quality. 

---
# An LLM-assisted theoretical model for aeroengine installation noise scattering 

**Authors**: Xun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18796)  

**Abstract**: A theoretical model is developed for the canonical problem of aeroengine fan noise impinging on, and scattering from, a nearby semi-infinite rigid plate. The model chains two Wiener--Hopf problems in different coordinate systems. The first stage solves the duct-radiation problem analytically in axisymmetric cylindrical coordinates. The second stage expands the field incident on the plate into a double angular spectrum of plane waves: the axial spectrum is inherited from the first stage, while the spanwise spectrum is generated by a Weyl--Sommerfeld expansion of the cylindrical harmonics. Each spectral component is then scattered by the plate. The mathematical crux is the coordinate transformation between the axisymmetric duct frame and the Cartesian plate frame. The core physical insight--that the Weyl--Sommerfeld identity provides an exact spectral bridge between these systems--originates entirely with the authors. The heavy algebra of translating between the two spectral representations, together with the associated numerical verification, was carried out with the assistance of a large language model (LLM), under strict human oversight and independent numerical cross-checking. This division of labour enables the LLM to handle tedious symbolic manipulation without compromising rigour. Results are presented for spinning modes radiating from a cylindrical duct with a subsonic jet into a quiescent ambient. The case study demonstrates that, under systematic verification, LLMs can accelerate tedious algebra in theoretical fluid mechanics, contributing a concrete empirical example to the broader discussion on AI's role in theoretical research. 

---
# Vibe synthesis: hierarchical agentic decision-making for intent-driven multistep molecule synthesis 

**Authors**: Jie Chen, Guikun Chen, Feng Wan, Ning Xu, Wen Liu, Yixiang Ruan, Yiyi Qiu, Haohua Li, Wei Wang, Yi Wang, Xiaoxue Wang, Peng Wang, Jiazhou Zhu, Jun Xuan, Chengxi Li, Jianhong Xu, Wenguan Wang, Yiming Mo  

**Link**: [PDF](https://arxiv.org/pdf/2611.18772)  

**Abstract**: The ability to construct complex molecules underpins advances in medicine and the molecular sciences. Yet long-horizon multistep synthesis requires decisions to be revised as evidence accumulates across chemically dependent operations, a capability not readily supported by artificial-intelligence tools and automated platforms organized around predefined tasks and hard-coded workflows. Here we propose and demonstrate a vibe synthesis system (VibeSyn), an intent-driven architecture in which hierarchical agents translate chemist-defined molecular objectives and accumulating evidence into adaptive decisions executed through computational tools and laboratory automation. We evaluate VibeSyn in three demonstrations of increasing scope and autonomy, progressing from evidence-grounded experimental decisions to adaptive execution of a ten-step total synthesis of catharanthine and structure-to-compound synthesis of cyclic peptides containing unnatural amino acids (uAAs). Across the long-horizon campaigns, active human input and simple confirmation account for less than 7.5% and 5.2%, respectively, of recorded workflow actions and outcomes. VibeSyn points towards a future in which chemists define synthetic intent and agent-directed laboratories autonomously navigate the evolving experimental trajectory required to deliver the target compound. 

---
# Gradient-Guided Hardware-Native MIXFP4 

**Authors**: Jian-Jia Chen, Brian J. Chan, Wen-Hung Lee, Yi-Ting Li, Po-Shen Chen, Kuan-Te Kuo, Hsien-Cheng Huang, Cody Hao Yu, Chi-Chih Chang, Mohamed S. Abdelfattah, Kai-Chiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18769)  

**Abstract**: Blackwell tensor cores support native inference with 4-bit weights and activations (W4A4), but NVFP4 leaves a perplexity (PPL) gap to BF16. Choosing a format for each 16-element block can improve accuracy, but current matrix multiply instructions cannot execute choices this fine. Applying these rules at hardware-compatible tiles loses most of their benefit. We introduce FlipQuant, which selects E2M1 or E0M3 for each MMA weight tile. It keeps both quantized weight candidates fixed and learns a binary format map from end-to-end gradients of a bucketed KL loss to a BF16 teacher, re-evaluating them after each update. Native kernels execute 16 x 64 tiles on SM120 and 256 x 64 tiles on SM100 with median GEMM overheads of 1.9% and 1.7%, respectively. Across six language models and two corpora, FlipQuant at 16 x 64 closes 42.5% of NVFP4's PPL gap to BF16 on average, compared with 13.3% for FourOverSix. Code is available at this https URL. 

---
# Frame First, Decode Once: Object-Centric Coordinates and Single-Pass Decoding for System-1 World Models 

**Authors**: Abu Anas Ibn Samad  

**Link**: [PDF](https://arxiv.org/pdf/2611.18761)  

**Abstract**: Single-pass primitive decoders such as LocateAnything and VFlash predict all quantized coordinate tokens of a geometric primitive in one forward pass, enabling fast, System-1 visual planning. We study what matters when such a decoder serves as a System-1 world model: a fast action-conditioned predictor that ranks candidate pushes by their predicted outcome. On ManiSkill3 PushT-v1 (512 held-out test families, 8,192 rollouts) and on human demonstrations from lerobot/pusht (2,384 test windows), we compare decoders, training losses, rigid-body projections and coordinate frames, using matched checkpoints, hierarchical bootstrap confidence intervals and Holm correction. Two factors of similar size dominate. First, expressing inputs and outputs in the SE(2) frame of the object's initial pose lowers decision regret by 1.9-2.2x for each of five decoders, and the object-frame model trained on 128 families matches the world-frame model trained on 512. Second, single-pass decoding outperforms iterative decoding: in the object frame, single-pass heads reach 0.020-0.021 regret at 1.6-1.8 ms per decision, against 0.046 for 10-step DDIM diffusion and 0.059 for KV-cached autoregression at 10.3 and 12.9 ms. The choice among single-pass heads has no measurable effect on regret, and our registered hypothesis that the frame effect exceeds the decoder spread is not supported. Soft-token expectation with an end-to-end 2D Kabsch projection gives the lowest trajectory error (11.35 px on ManiSkill3, 3.78 px on lerobot/pusht), exact rigid-body consistency, and lower regret than diffusion at every budget from 4 to 50 DDIM steps, at 3-29x lower latency. 

---
# Neural Latent Gaussian Process for 3D reconstruction of Mouse Brain Lipids 

**Authors**: Daniel Trejo Banos, Artiom Casapu, Luca Fusar Bassini, Halima Hannah Schede, Giovanni D'Angelo, Gioele La Manno, Ekaterina Krymova  

**Link**: [PDF](https://arxiv.org/pdf/2611.18744)  

**Abstract**: We address the problem of reconstructing continuous three-dimensional lipid profiles of the mouse brain from lipid measurements acquired only on a sparse set of two-dimensional tissue sections. Using MALDI-MSI measurements from coronal sections of two densely sampled and six sparsely sampled mouse brains, registered to the Allen Mouse Brain Common Coordinate Framework (CCF), we seek to interpolate the measured hundreds of lipid profiles across the unobserved voxels of the reference brain. We introduce a Neural Latent-Space Gaussian Process (LGP) architecture for this task and critically evaluate its ability to reconstruct spatial lipid distributions, using held-out mouse brain sections to assess generalization.
We propose a Gaussian Process prior for the distribution of low-dimensional lipid profiles across the mouse brain (with Matérn or a Riemann manifold kernel). This allows us to encode 3D coordinates in a small number of latent 3D volumes that are then mapped to particular lipids using a multilayer perceptron (MLP). We performed an exhaustive analysis over a range of algorithms from simple linear regression, XGBoost, MLP to state of the art models such as Neural Transcriptomic Field, graph-based SPA3D for spatial omics reconstruction and DEEPSPATIAL. We additionally evaluate GPLFR as an alternative Gaussian-process-based approach.
Across 173 measured lipids, LGP outperforms all evaluated baselines, including recent spatial-omics models and GPLFR, achieving a mean correlation of 0.613 and demonstrating the effectiveness of our approach for sparse 3D lipid interpolation. 

---
# Recognizing Events Is Not Enough: Why Streaming VLMs Struggle in Compositional Instruction Triggering 

**Authors**: Yehua Huang, Penglei Sun, Xiang Li, Jinhao Zhou, Jiawen Wen, Qingqing Xiao, Zihan Yang, Wenqi Wang, Haolan Chen, Yu Xu, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18743)  

**Abstract**: Recognizing individual events does not guarantee correct execution of compositional instructions in streaming video. For example, given the request "Notify me when the pot boils after someone turns on the stove and leaves the kitchen", existing models may recognize the individual events yet notify too early or miss the intended moment, failing to track how their temporal conditions are satisfied over time. To evaluate this gap, we introduce CoTriggerBench, which composes annotated video events into temporally constrained instructions and verifies notification timing and silence through executable event plans and source-event replay. To address this challenge, we propose a real-time VLM agent, TEMPI. TEMPI decomposes instructions into atomic event queries and temporal constraints, and uses Scenario Memory to continuously maintain task progress as event evidence arrives, allowing earlier observations to support subsequent decisions beyond the detector's visual window. Even when no notification is required, relevant events continue to update the scenario state. On CoTriggerBench, TEMPI achieves 44.4% instruction-level success, outperforming the strongest closed-source and fine-tuned open-source VLM baselines by 15.6 (Qwen3.8-Omni-Flash-Realtime) and 18.1 (MOSS-VL-Realtime) percentage points, respectively. To support efficient on-device streaming inference, we further optimize the streaming inference pipeline by overlapping partial visual encoding and LLM prefix prefill with frame arrival. On an 8 GB RTX 5060, our optimized detector implementation achieves a 4.90$\times$ speedup. 

---
# Target-Free Autocalibration of Stereo PIV Camera Systems via Persistent-State Motion Representations 

**Authors**: Nabil Madali  

**Link**: [PDF](https://arxiv.org/pdf/2611.18741)  

**Abstract**: Stereo Particle Image Velocimetry (PIV) recovers three-component velocity fields by triangulating tracer particles observed by synchronized cameras. It requires an accurate extrinsic calibration, which is still obtained by inserting a physical target into the flow and must be repeated whenever the rig is moved. We present a target-free method that calibrates a stereo PIV rig directly from particle image sequences. Because tracer particles are visually indistinguishable, we match motion rather than appearance: building on CUT3R, each camera stream is processed by a recurrent transformer whose persistent state accumulates particle motion over time. The state is decoded into per-camera optical flow and queried by a cross-view decoder that regresses bidirectional point correspondences, from which the relative pose is recovered with a five-point RANSAC solver. We train on a new synthetic stereo PIV dataset with ground-truth poses and evaluate on standard PIV optical flow benchmarks and on 200 real rig placements. Our model achieves lower flow error than existing PIV networks, more accurate cross-view correspondences than COTR, and higher pose accuracy than the strongest detector-matcher pipelines on both synthetic and real data, without any calibration target. 

---
# Is It Still There? Place-Registered Verification Makes VLMs Reliable Scene-Change Detectors 

**Authors**: Jinjie Xie, Baihua Li, Qinggang Meng  

**Link**: [PDF](https://arxiv.org/pdf/2611.18734)  

**Abstract**: A robot that returns to a space must tell which objects were added, removed or moved since its last visit, while its trajectory, viewpoints and lighting differ between the two visits. We find that multi-image vision-language models (VLMs) fail at this task mainly because they cannot find the place where an object was. A VLM recognises what each frame shows but cannot check whether an object is still at its place in the other capture. It therefore reports objects that only one capture shows and misses most removed and moved objects. We propose STILL, which reformulates change detection as object verification at a registered place. 3D geometry registers the place of each change hypothesis in the other capture, and one VLM answers, from a scene view and a local place view, whether the object is still at that place, trained with supervision of the same form. Under the official SceneDiff protocol, STILL raises the main metric obj/sc AP from 24.0 to 37.8, and from 11.6 to 20.6 on the egocentric kitchen subset without any kitchen training data. It improves F1 by more than 21 points over a VLM fine-tuned on the same annotations and adds only about 11% to the runtime. 

---
# Wrong with Confidence: Measuring Hallucination, Overconfidence and Identity-Conditioned Behaviour in Large Language Models 

**Authors**: Md. Rejaul Korim Sadi, Toufiqur Rahman Tasin, Golam Mostofa Naeem, Dewan Fahad Chowdhury, Hatim Al Amin Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2611.18728)  

**Abstract**: Large language models answer underspecified questions readily, and accuracy-only evaluation rewards guessing over abstaining. Observational studies link query underspecification to hallucination, but they cannot separate the effect of missing information from item difficulty, and they rarely ask what the model tells the user. We report a pre-registered, paired behavioural study of five open-weight models (Gemma 3 27B, Gemma 4 31B, gpt-oss-120b, Qwen3-Next-80B-A3B and Nemotron 3 Super). Each of 142 factual questions was posed in a fully specified form and in a referentially underspecified form with the same answer key, and three probes measured identity-conditioned behaviour. Two raters scored 5,620 responses under a fixed rubric; effects were estimated with mixed-effects models. Without the referent, the models sought clarification on 33.7% of prompts and answered the rest. Among attempted answers, the error rate rose from 39.1% to 52.0% (mixed-effects odds ratio 4.12, 95% CI 2.76-6.14), and specialist-domain questions failed more often than general ones in every model. Unelicited verbal uncertainty did not track this risk: 75.3% of wrong answers carried no hedge, wrong answers to underspecified prompts were hedged no more often than wrong answers to clear prompts (27.3% vs 22.3%), and hedging separated correct from incorrect answers only weakly (AUROC 0.56). Gendered pronouns followed occupational stereotypes whenever a model committed to one; any stated user identity lengthened answers by about 45% relative to an anonymous user; claimed expertise did not reduce safety caveats. The models either ask or answer as if the question were well posed, and the wording of a guess does not reveal that it is one. Evaluations should score clarification as a distinct outcome and check label validity as well as rater agreement. 

---
# Latent Fusion: Channel Stacking for Structure-Preserving Image Editing 

**Authors**: Mohit Dulani  

**Link**: [PDF](https://arxiv.org/pdf/2611.18727)  

**Abstract**: Instruction-based image editing models must condition on the source image, either by sequence concatenation, which appends source tokens to the target sequence (FLUX.1 Kontext), or by channel-wise concatenation, which stacks the source latent onto the noisy latent and widens the input projection (InstructPix2Pix, FLUX.1 Fill). The FLUX.1 Kontext report found channel-wise concatenation worse in initial experiments at 1K resolution with a 12B model and a large corpus. We explore channel-wise concatenation as the lower-cost option for a common production regime: structure-preserving edits that change appearance (sky, exposure, foliage, cleanup) but not layout, about 1,400 training pairs, a 4B non-distilled backbone, and 2K frames. It keeps the sequence at its original length, where sequence concatenation doubles it and roughly quadruples attention cost, and needs only single-scale classifier-free guidance (two forward passes per step instead of three). Starting from FLUX.2 Klein 4B base, we widen only the input projection and train both schemes with the same trainer, data, and hyperparameters. Channel-wise concatenation keeps the output pixel-aligned with the source; in our runs sequence concatenation, which must learn the correspondence between 10,880 target and 10,880 source tokens from fewer than 1,500 pairs, displaced structure, an observation we report without claiming a general ranking. On 29 held-out images the editor reaches 20.3 dB PSNR and 0.198 LPIPS against the human edit (unedited source: 15.0 dB), and counterfactual conditioning tests show it largely learned a global grade of the source rather than a selective edit, which we report as the open problem of this recipe. We also compare against an SD3.5-Medium editor trained with the same surgery and report the full training recipe, including failure modes. Code: this https URL 

---
# Acting by Intent, Reacting by Sight: Visual Conditioned Reflexes for Real-Time VLM Agents 

**Authors**: Qiancheng Zhou, Wenhua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18723)  

**Abstract**: Vision-Language Models (VLMs) have shown strong capabilities as generalist agents that perceive, reason, and act in interactive environments. Yet real-time environments such as video games demand both deliberate control for exploration and split-second reactions to fast-moving hazards. Existing agents struggle to handle both: relying solely on direct VLM actions is too slow for sudden events, whereas delegating execution to fixed scripts leaves the agent blind at the moment of action, leading to frequent misfires whenever the scene shifts. Humans overcome this by pairing normal conscious control with conditioned reflexes-while the mind handles deliberate actions and shapes reflex rules, the reflex itself still relies on live vision to make the final instant decision. Emulating this mechanism, we propose ViCoR (Visual Conditioned Reflex). Alongside normal gameplay actions, the main VLM defines and updates lightweight situational guidance for recurring events, while a fast visual reflex branch (invoking the VLM in no-thinking mode) inspects live local views with motion cues to make the final sub-second action decisions. By offloading 67.8% of high-frequency reactions to the reflex branch (680 ms per check), ViCoR reduces average per-action decision latency by 2.6x (from 14.0 s to 5.4 s) and cuts token consumption by 36.9%. Across 17 diverse 2D and 3D games in the GameWorld benchmark, ViCoR achieves state-of-the-art performance, improving task success rate by +11.8% (from 52.9% to 64.7%) and average normalized progress by +13.9% (from 66.5% to 80.4%) over the strongest deliberative VLM and script baselines. In fast-paced real-time games, ViCoR refines its reflexes across episodes to boost progress by 1.67x, and even with a low-effort main planner, surpasses high-effort deliberative baselines by +8.1% at one-third of the token cost. 

---
# Editable Video Context: Mid-Context KV-Cache Reuse for Context-Managing Video LLMs 

**Authors**: Dayu Wang, Shasha Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18718)  

**Abstract**: Context Language Models let a model edit its own context, while Suffix Cache Reuse makes edits to text efficient by adjusting the positions of cached keys for surviving tokens. We extend these ideas to video language models with multimodal rotary position embeddings (M-RoPE). We represent a long video as timestamped segments. After viewing a sparsely sampled version, the model selects segments for decoding at 4x the frame rate, and the remaining segments are removed from its context. Our rebuild method reuses the KV cache of surviving segments by rotating their keys with a single offset shared by the temporal, height, and width components of Qwen2.5-VL's M-RoPE. A content-hashed cache also avoids repeating vision-encoder computation for those segments. In preliminary experiments with Qwen2.5-VL-7B on synthetic needle-in-a-video-haystack questions over public-domain films, the method reduces prefill tokens and end-to-end rebuild latency relative to full recomputation while usually producing the same answer. Densifying the segment containing the answer improves accuracy when that segment is known. When the model selects segments itself, accuracy does not improve: it rarely identifies a needle that was absent from the sparse sample. We release the code, raw logs, and scripts used to reproduce the paper's results. 

---
# T2T Technical Report: Synthesizing GUI Agent Trajectories at Scale from Video Tutorials 

**Authors**: Qingsong Xie, Yanlin Fei, Nazhou Liu, Jiacheng Tang, Madalina Ciobanu, Navan Preet Singh, Qingqing Mao, Ritankar Das  

**Link**: [PDF](https://arxiv.org/pdf/2611.18715)  

**Abstract**: Training generalist GUI agents requires large volumes of interaction trajectories comprising screenshot sequences paired with grounded UI actions, yet collecting such data through human demonstration is expensive and difficult to scale. Although public software tutorial videos contain rich operational demonstrations, their continuous pixel streams lack discrete action boundaries, semantic intents, and precise spatial coordinates, making them difficult to use directly for model training. To address this, we present T2T (Tutorial-to-Trajectory), a fully automated pipeline for synthesizing GUI agent trajectories at scale. T2T spans desktop, web, and mobile platforms to convert unannotated video tutorials into structured, executable interaction trajectories. To guarantee trajectory quality, T2T combines active segment filtering, keyframe identification via localized UI change, shortcut-free observation frame selection, presence-aware grounding, and two-stage action-space normalization. These refined trajectories are then converted into multi-task training views spanning navigation, planning, forward dynamics, and grounding. We fine-tune Qwen3-VL-8B on a subset of the more than one million human-annotation-free training samples generated by this pipeline. Evaluations show that the model achieves consistent gains across multiple benchmarks covering desktop, web, and mobile environments, including OSWorld-G, ScreenSpot-Pro, AndroidControl, and AgentNetBench, indicating improved core capabilities in grounding, step-level instruction execution, and trajectory-level decision-making. Notably, all gains are achieved without any in-distribution training data from target domains. Learning exclusively from trajectories extracted from public videos via the T2T pipeline, the model exhibits cross-application and cross-task generalization capabilities. 

---
# What Does Hard-Negative Fine-Tuning Teach Multimodal Large Language Models? Controlled Evidence from a Compositional Benchmark 

**Authors**: Ranjith Chodavarapu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18701)  

**Abstract**: Fine-tuning on hard negatives is a popular way to improve the compositional reasoning of vision-language models, but when accuracy goes up it is rarely clear what the model actually learned. We ask this question for multimodal large language models. We use CompHall, 2,860 forced-choice questions (spatial, attribute, counting and part-whole) with two graded hard negatives each, three model families (LLaVA-1.5, LLaVA-NeXT and Idefics2), and controls that such studies often lack: gray-image evaluation, three seeds, option-order tests, control arms trained on different question types, and independent benchmarks. We find four things. First, CompHall is not a pure test of vision: part-whole questions can be answered from the text alone (66-95% without the image), and per-image accuracy is unrelated to an object-hallucination probe (r = -0.05), so we see no evidence that compositional failures drive hallucination. Second, the original fine-tuning method scored mostly image-placeholder tokens and hurt both LLaVA models. With a corrected loss, fine-tuning raises accuracy on the spatial-relation questions of CV-Bench in all three families, by +3.2, +23.1 and +6.0 points (averaged over both option orders). Control arms show that the gain needs spatial training questions, and for LLaVA-NeXT about two-thirds of it appears even on relation words that were only wrong options in training. Third, the gain is not general: on CompHall itself it is significant only for LLaVA-NeXT (+6.9 points) and not detectable for Idefics2, it reaches about 3 points at most on true/false statements, and LLaVA-1.5 fine-tuned on other question types learns to answer "true" too often. Fourth, both losses raise the yes-bias of LLaVA-1.5 on POPE. 

---
# ARC-VLN: Adaptive Recursive Computing-based Visual-Language Navigation 

**Authors**: Xiang Zhang, Xiangyu Dong, Yang Lu, Xiaoguang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.18696)  

**Abstract**: Efficient aerial visual-language navigation (VLN) requires reducing model storage and inference costs while maintaining instruction-conditioned action prediction. We present ARC-VLN, a token-type-aware recursive computation framework built on a language backbone with shared parameters. ARC-VLN retains full-depth computation for text and action tokens while adaptively selecting exit depths for visual tokens. Representations of early-exiting visual tokens are retained, and active-token compaction shortens the sequences processed by subsequent recursive blocks. To mitigate routing collapse to a single exit depth, we design a dedicated learning strategy that encourages the model to distribute visual tokens across multiple exit this http URL OpenFly's Test-Seen and Test-Unseen subsets, ARC-VLN achieves the lowest navigation error and the highest success rate and success weighted by path length among all compared methods. Meanwhile, it reduces the average number of decision steps per trajectory, total floating-point operations, mean model inference time per trajectory, and average GPU memory usage during inference, indicating its great potential as a VLN backbone for lightweight applications. 

---
# Swarm Shepherd: Securing Multi-Agent Ecosystems Against Persistent Latent Compromise 

**Authors**: Wuyang Zhang, Shichao Pei  

**Link**: [PDF](https://arxiv.org/pdf/2611.18693)  

**Abstract**: Agentic ecosystems are emerging as a new platform layer in which agents maintain persistent memory, retrieve from shared knowledge bases, execute through tool interfaces, and coordinate through shared states. These capabilities create a security problem that per-turn defenses miss: compromise can accumulate across sessions, agents, and system layers before visible failure. We define this threat as Agentic Sleeper, a system-level adversary that first behaves legitimately, gains routing trust, and later activates across memory, agent logic, and infrastructure. We formulate defense as inference under partial observability and propose Swarm Shepherd, a system-level defense framework centered on an inference filter for early warning, paired with a backward trace-back module for entry attribution. Our analysis yields a two-scale deployment rule that balances longer event histories against broader agent coverage. Across LangGraph, CrewAI, and AutoGen, Agentic Sleeper drift begins 2.7--8.9$\times$ faster than natural baselines, propagates across agents on most topologies, and evades per-turn and stateful defenses above 80\%. Swarm Shepherd warns at least three sessions before visible degradation (2.6 sessions when failure is scored by the benchmark's own task grader), while trace-back reaches 0.69 top-1 accuracy and 0.72 macro-F1 over 24 variants. The two-scale deployment rule is empirically confirmed, and in live end-to-end runs on LangGraph and AutoGen the posterior-guided defense stack drives cross-agent propagation below the self-sustaining regime, which removing only the visibly compromised agent does not achieve. 

---
# Perceptual Weight Space Programming for Diffusion Super-Resolution 

**Authors**: Chenzhi Zheng, Zhuoran Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.18690)  

**Abstract**: Diffusion models for image super-resolution can recover scene content while producing distracting artifacts or unbalanced colors. We study whether a large language model (LLM) can improve perceptual quality by editing a small subset of existing U-Net weights without a clean reference image. The proposed framework provides the editor with architecture code, selected weights, activation summaries, and sampling previews. It permits up to 1,020 candidate trials while limiting accepted changes to three layers and fewer than 0.1\% of model parameters. The evaluation protocol uses ResShift, SinSR, and SUPIR and compares GPT-6 Astra with a specified Qwen3-VL model under matched access. Feedback metrics guide editing, while separate quality metrics and blinded human comparisons evaluate the final weights. We also report an executed color calibration experiment using real photographs and two small residual networks. This experiment demonstrates correction of a fixed color shift and reveals that an improving feedback score can accompany worse performance on unseen scenes. A separate lightweight super-resolution experiment isolates modest changes in luminance detail while preserving color. The findings motivate restricted updates and independent evaluation of perceptual weight editing. 

---
# Selection-Aware Certification of Client-Constrained Fraud Rule Portfolios Under Synthetic Distribution Shift 

**Authors**: Rishabh Pande, Yu Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2611.18687)  

**Abstract**: Selecting fraud-rule portfolios to satisfy a precision or recall floor can overfit the data used for selection. We study finite-family certification for bounded OR portfolios, comparing point-estimate selection, a fixed safety margin, and one-sided Clopper-Pearson lower bounds with Bonferroni adjustment. In a factorial follow-up with 6,400 datasets from two author-designed synthetic generators, full-validation certification reduced the held-out floor-violation rate by 29.1 percentage points on average versus full-validation point selection, while reducing deployment by 31.7 points. Compared with a fixed 0.05 margin, certification reduced violations by 22.2 points and deployment by 29.4 points; results varied across conditions. Using the full rather than half validation sample increased deployment by 17.1 points on average, without a uniform reduction in shifted-test violations. In a separate 400-dataset no-shift sweep, certification deployment rose from 61.3% to 90.2% as validation size increased from 1,500 to 15,000; no violations occurred among 2,887 deployed portfolios tested on 100,000-row holdouts. A temporal case study on the public synthetic Bank Account Fraud Base dataset used recall at a 5% false-positive-rate cap; the selected candidate pool captured less than 1% of test fraud, showing limited coverage in this setup. Together, these experiments characterize how finite-family certification changes the reliability-deployment tradeoff for bounded OR portfolios under specified synthetic conditions. The evidence covers the evaluated generators and one public synthetic BAF case study; it does not estimate performance on real transaction populations or robustness to arbitrary distribution shift. The authors revised and rewrote this abstract after receiving a draft from OpenAI Codex. 

---
# ReMoVe: Refining a Video Model's Own Motion for Coherent Human Video Generation 

**Authors**: Jie Li, Muxin Pu, Hongyi Cai, Mingkang Dong, Jianyuan Guo, Tao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18666)  

**Abstract**: Video generation models have become increasingly capable of synthesizing realistic human motion, yet they often lose track of the very structure that makes the motion plausible. As generated human motion becomes more complex, videos can suffer from progressive body deformation and appearance drift across frames. Existing motion-controlled methods use explicit representations to stabilize generation, but typically rely on pre-defined motion sequences as references. This raises a fundamental question: if the base generator already produces a usable coarse motion, why replace it with an external one? ReMoVe answers this question by treating the coarse output as a motion hypothesis rather than a final result. ReMoVe recovers and refines the underlying motion, then renders the refined motion as 3D mesh and transfers facial appearance onto it, thereby improving motion accuracy and identity consistency. We further derive depth from the same 3D scene and camera trajectory to impose scene-level spatial consistency beyond the mesh. By jointly conditioning video generation models on mesh and depth, ReMoVe synthesizes high-quality motion videos with controllable camera trajectories and enhanced spatio-temporal consistency. Extensive experiments demonstrate that ReMoVe improves human structural stability, appearance consistency, temporal coherence, and motion-following fidelity, while retaining the base generator's motion prior and offering accurate camera control. 

---
# SART: Sign-Absolute Reformulation Theory for Binary Variable Reduction in Neural Network Verification 

**Authors**: Jin Xu, Miaomiao Zhang, Bowen Du  

**Link**: [PDF](https://arxiv.org/pdf/2611.18657)  

**Abstract**: Complete formal verification of neural networks is crucial for their deployment in safety-critical domains. A key bottleneck stems from encoding complexity: traditional methods assign one binary variable per unstable ReLU neuron. We propose the Sign-Absolute Reformulation Theory (SART), which fundamentally breaks the conventional one-to-one mapping between unstable neurons and binary variables by establishing formal reducibility criteria. This allows for finer-grained modeling, where each unstable neuron corresponds on average to fewer than one binary variable, thereby reducing verification complexity at its source. Based on SART, we derive a theoretical lower bound on the number of binary variables required for complete verification and, under the assumption that $P \neq NP$, prove that variables in the final layer can be compressed by 50\%, while the number of variables in intermediate layers cannot be further reduced. To overcome the apparent ``last-layer-only'' limitation, we recast verification as a sequential process and, crucially, show that the gain lifts to the entire network: LayerABS, a SART-based progressive tightening verifier, iteratively treats intermediate layers as temporary final layers and propagates tight bounds that shrink the global search space and binary-variable counts. Furthermore, we reveal a structural law influencing verification complexity: when the signs of weights of unstable neurons satisfy numerical symmetry, with positive and negative weights equal or differing by at most one, the worst-case verification complexity achieves the theoretical optimum, offering theoretical guidance for the design of verification-friendly architectures. As a general-purpose underlying encoding, the value of SART is independent of specific algorithms. 

---
# Algebraic Interaction: A Recipe for Neural Architecture Generation 

**Authors**: Haonan Dong, Angelica I Aviles-Rivero  

**Link**: [PDF](https://arxiv.org/pdf/2611.18649)  

**Abstract**: We introduce an algebraic foundation for neural architecture design in which convolution, attention, state-space models, and equivariant layers are expressed through compositions of bilinear product operators over two structures: a position algebra governing positional routing and a feature algebra governing feature interactions. This decomposition makes routing, feature algebra, and operator composition explicit design this http URL focus on the feature-algebra axis through an Algebraic Recipe: fix the routing and operator composition of an existing architecture while replacing its feature algebra. This preserves the information-flow structure while allowing computational and symmetry properties to be modified through algebraic this http URL instantiate the recipe in linear attention, Mamba, and equivariant networks. Structure-constant and Clifford algebras improve accuracy at comparable parameter count in linear attention; structure-constant algebras yield up to $3\times$ faster training and inference in Vision Mamba at competitive accuracy; and an $\mathrm{SO}(3)$-representation algebra provides rotational equivariance by construction. These results demonstrate that feature algebra alone can control accuracy, efficiency, and symmetry without redesigning positional this http URL turns algebraic structure into an explicit design variable and reduces architecture design, within our framework, to algebraic specification. 

---
# Rethinking Offline On-Policy Distillation: From Scalar Advantages to Distributional Supervision 

**Authors**: Qi Sun, Siyue Zhang, Tingyu Song, Yilun Zhao, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.18648)  

**Abstract**: On-policy distillation (OPD) transfers teacher capabilities through token-level feedback on student-generated trajectories, but repeatedly evaluating fresh rollouts with the teacher is computationally expensive. Offline methods such as Lightning OPD reduce this cost by caching recorded tokens' teacher log-probabilities on fixed reference rollouts. By extending the idea to a multi-turn agentic search setup, we find that it consistently underperforms the SFT starting point trained by the same teacher. We empirically observe greater teacher--reference mismatch than in mathematical reasoning and post-training distributional flattening, and theoretically show that scalar advantage supervision on fixed rollouts admits solutions with flattened output distributions under suitable conditions. Motivated by these findings, we propose Distributional On-Policy Distillation (DistOPD), a simple yet effective offline OPD method that retains candidate-token probabilities from both reference and teacher at selected positions and aggregates the remaining parts into a residual token. On agentic search, DistOPD achieves performance comparable to online OPD while improving student--teacher alignment and using training GPU-hours comparable to Lightning OPD. 

---
# Screen Before You Add: Reducing Candidate Inference in Model Ensembles 

**Authors**: Davide Mambelli, Alessandro Rozza, Steven Motta  

**Link**: [PDF](https://arxiv.org/pdf/2611.18647)  

**Abstract**: As new models become available, existing ensembles must be updated to be competitive, making them intrinsically dynamic. Deciding whether to add a candidate model can require inference on an entire annotated dataset. Such a screening becomes prohibitive for models with costly inference especially when they ultimately add no value. Even under simple convex aggregation, relying on benchmark scores can be misleading as strong standalone performance does not guarantee ensemble improvement, and fitted ensemble weights can hurt test performance. In this work, we study candidate screening and show how early rejection can be achieved safely, exploiting residual geometry for stratification and rejecting the candidate when the best possible remaining predictions cannot achieve the necessary improvement. The key result that justifies this approach is that, under convex aggregation and squared or Brier loss, the population optimum improves if and only if a candidate induces a prediction change positively aligned with the residuals of the population-optimal existing ensemble. An optimistic quadratic program (QP) provides certified early rejection, while posterior-scenario sampling offers an alternative heuristic. We evaluate the approach by showing the tradeoff between saved calls and screening accuracy, which refers to recovering the same accept/reject result as for the entire annotated dataset. We assess empirically whether full-dataset decisions predict candidate usefulness on a held-out test set. On two LLM panels, stratified screening with certified QP rejection saves 21-23% of candidate calls while keeping 100% screening accuracy. In our experiments, sampling posterior scenarios further increases call savings to approximately 37%, with a loss of less than one percentage point in screening accuracy. 

---
# RSIMed: Scaling Medical Foundation Models beyond Limited Clinical Data via Retrieval-Grounded Self-Evolution 

**Authors**: Wei Dai, Qianchu Liu, Guanghui Qin, Xing Han, Tristan Naumann, Sheng Zhang, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18646)  

**Abstract**: Rare diseases are hard to diagnose, and the labeled clinical records needed to train models for them are few, private, and slow to extend, since a specialist must read every case. A model that writes its own training tasks could learn beyond these records, but no program can check a model-written diagnosis, and a wrong answer key corrupts training. To tackle this dilemma between data scarcity and difficulty of data annotations, we introduce RSIMed, Recursive Self-Improvement in Medicine, one of the first self-evolving pipelines in which a task proposer co-evolves in context with a diagnostic solver on real clinical records. Seeded by training admissions, a frozen copy of the solver writes each clinical question from retrieved medical evidence, so its answer rests on a source, and steers topics and difficulty toward what the solver still gets wrong. On rare-disease diagnosis from MIMIC-IV admissions, the accuracy of RSIMed-27B grows linearly with training compute, while a static pool of generated tasks collapses. Without any external model, RSIMed-27B and RSIMed-9B reach 0.516 and 0.488 diagnostic accuracy, above every off-the-shelf GPT-5.6 setting, whose best reaches 0.399 at 38 times the inference cost of RSIMed-27B. Tasks written by four external teachers, including GPT-5.5, train a less accurate diagnostician than the solver's own. Trained only on diagnosis, RSIMed-RL-27B also improves on the physician-written tasks of HealthBench Professional, from 0.291 to 0.303 in adjusted score. 

---
# Minari: Infrastructure for Collaborative Reinforcement Learning Datasets 

**Authors**: Omar G. Younis, Rodrigo Perez-Vicente, John U. Balis, Alex Davey, Will Dudley, Jordan K. Terry  

**Link**: [PDF](https://arxiv.org/pdf/2611.18644)  

**Abstract**: Offline reinforcement learning (RL) depends on shared datasets, but sharing them is still hard. Each benchmark ships its own format, documentation is thin, and most datasets cannot be regenerated once the environment code moves on. We introduce Minari, an open-source Python library for creating, sharing, and loading RL datasets. Its API follows Gymnasium, so recording a dataset means wrapping an environment and stepping through it as usual. From there, users can attach metadata, filter, combine, and shuffle datasets, and publish or download them with one command through the HuggingFace Hub. Every dataset stores the environment specification, seeds, applied wrappers, and pinned package requirements needed to rebuild the environment and regenerate the data. TorchRL, AgileRL, and d3rlpy already load Minari datasets natively, so switching datasets in those libraries takes one line of code. Minari also hosts 338 reproducible datasets (350K trajectories, 55M steps) covering robotic locomotion and manipulation, maze and grid navigation, language-conditioned instruction following, Atari, and web navigation, including a full regeneration of D4RL with public generation scripts. We report baseline scores for six offline RL algorithms on the regenerated D4RL and MuJoCo datasets. 

---
# FedFit: Federated Dynamic Sparse Training via Fisher Information scoring 

**Authors**: Meng Bi, Hong Huang, Jinlong Song, Charles Wang, Chengming Hu, Xi Chen, Ting Yu, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18643)  

**Abstract**: Cross-device Federated Learning (FL) is frequently bottlenecked by the prohibitive memory and communication costs of training deep neural networks on resource-constrained edge hardware. While federated dynamic sparse training aims to alleviate these costs by adjusting sparse structures during training, existing methods rely on magnitude-based heuristics that are fundamentally ill-suited for the non-convergent, heterogeneous environments inherent to FL. To address this challenge, we propose FedFit, a federated dynamic sparse training framework that replaces simple heuristics with optimization-centric criteria for structure adjustment. By leveraging a second-order approximation of the loss landscape via the Fisher Information Matrix, FedFit enables precise and efficient structure adjustment without the overhead of explicit Hessian computation. Empirical evaluations across computer vision and natural language processing benchmarks demonstrate that FedFit significantly narrows the sparse-to-dense accuracy gap, outperforming state-of-the-art methods while maintaining high communication efficiency. Our code is available at this https URL . 

---
# SynthDCB: World Distillation for Scalable Device Control Benchmarking 

**Authors**: Zichen Zhu, Dingye Liu, Nan Jiang, Danyang Zhang, Hang Zheng, Shijie Zheng, Situo Zhang, Kunyao Lan, Yingzi Ma, Sheng Jiang, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18639)  

**Abstract**: Scaling device-control evaluation requires scaling both the worlds agents interact with and the tasks they must solve. We introduce SynthDCB, a scalable benchmark for environment awareness and device control built through world distillation. World distillation transforms knowledge of device behaviors, interaction constraints, and environmental effects into modular executable components, which are composed into interactive worlds with explicit states, actions, and temporal dynamics. A unified capability representation connects tool discovery, world execution, and task synthesis, allowing new capabilities to enter evaluation through a shared construction pipeline. SynthDCB instantiates Home, Car, and Phone domains with 114 device classes, 1,166 functions, and 95 events, supporting five task families with 32 subtypes. The stateful design enables task validation and agent evaluation through state correctness and tool-call traces. We evaluate seven models on 2,560 test tasks; the best achieves 68.2% Pass@1. Error analysis reveals challenges in answer delivery, feasibility diagnosis, and temporal coordination. Construction experiments demonstrate scalability across domains and token efficiency: world distillation reduces token consumption by 0.9-66.0% and 46.2-92.5% relative to Codex with low and xhigh reasoning effort, respectively, while achieving broader capability coverage. SynthDCB makes executable world construction the basis of scalable device-control benchmarking, connecting an expanding space of capabilities and interactions to verifiable agent evaluation. 

---
# ReVision: Selective Intrinsic Reasoning for Tool-Augmented, Reasoning-Intensive VQA 

**Authors**: Ningxin Pan, Guochen Yan, Jialong Wu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18637)  

**Abstract**: External tools substantially expand what multimodal large language models (MLLMs) can retrieve and compute on reasoning-intensive visual questions, and aggregate accuracy typically improves when tools are available. Our empirical study across six VQA benchmarks shows that aggregate accuracy gains under tool access can coexist with answer reversals: questions answered correctly without tools but incorrectly with tools. Trajectory analysis reveals substantially shorter reasoning turns under tool access, suggesting a shift from intrinsic visual reasoning toward reliance on external outputs. We propose ReVision, a training-free test-time method that registers intrinsic reconsideration as an optional action alongside search and code execution. When invoked, a separate tool-free multimodal call re-examines the original image, integrates accumulated evidence, and returns a revised analysis. The MLLM chooses if and when to use it, without fixed reflection stages or paired rollouts. ReVision improves over standard tool use on all six benchmarks (1.28--7.51\%) and outperforms Dual-Rollout Combine on five, with reasoning-token use comparable to standard tool use. 

---
# Multimodal Fusion with Pre-trained Models for Psychotherapy Outcome Prediction from Emotional Dynamics 

**Authors**: Tobias Hallmen, Marius Funk, Antonia Vehlen, Jeanne Hierse, Gerhard Prinz, Steffen Eberhardt, Wolfgang Lutz, Elisabeth André  

**Link**: [PDF](https://arxiv.org/pdf/2611.18635)  

**Abstract**: In-session emotional dynamics are associated with psychotherapy outcomes but difficult to measure at scale. Pretrained multimodal models provide automated access to facial, vocal, and textual signals, yet clinical datasets remain small. Our research question is how pretrained multimodal representations can best support psychotherapy outcome prediction under data scarcity. Three studies use 106 sessions from 88 patients and nine session-level outcome scales (HSCL-11, EPO-1, POMS). Study 1 asks whether deep learning outperforms classical machine learning: a two-stage deep pipeline, first predicting window-level valence/arousal and then session outcomes, transferred poorly (best single configuration, $r=.078$). Study 2 asks which representation and preprocessing choices work best: statistical pooling of frozen pretrained embeddings with regularized classical models achieved average $r=.406$ across nine scales under per-scale configuration selection, up to $r=.490$ for symptom distress, with complementary modalities and no significant benefit from manual affect annotations or therapist features. Study 3 asks whether deep learning remains inferior under these choices: classical models outperformed an optimized single-stage deep model at the single-configuration level ($r=.281$ vs. $.222$) and on all nine scales under matched per-scale selection ($.406$ vs. $.252$). The results suggest that under clinical data scarcity, pretrained representations with fixed statistical pooling and regularized classical models are more reliable than learned aggregation. 

---
# Brain AI Scientist: An Evidence-Based Multi-Agent System for Translational Neuroimaging in Precision Psychiatry 

**Authors**: Zeyun Zhao, Lin Gu, Ruogu Fang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18634)  

**Abstract**: Resting-state functional magnetic resonance imaging (rs-fMRI) has revealed widespread brain-network alterations in psychiatric disorders, yet translating these findings into individualized clinical decision support remains challenging. Here we introduce Brain AI Scientist, an autonomous multi-agent framework that integrates patient-specific fMRI, clinical information, and scientific evidence through retrieval, verification, and prioritization. Guided by Research Domain Criteria and evidence-based medicine, it builds traceable chains linking phenotypes, anatomical abnormalities, biomarkers, mechanisms, interventions, and validation strategies. In a representative autism case, Brain AI Scientist generated 37 anatomical hypotheses, retained 26 after literature verification, and derived 65 evidence-supported biomarkers from 90 unique publications to generate patient-specific treatment and validation guidance. Across 17 autism spectrum disorder cases, it outperformed GPT-3.5 baselines with and without fMRI impression in evidence grounding, personalization, clinical actionability, and biomarker-treatment reasoning. Brain AI Scientist therefore provides an auditable framework for translating patient-specific neuroimaging into individualized clinical decision support for precision psychiatry. 

---
# Minimum cores in lattice graphs: zonotope structure and quartic growth 

**Authors**: Navid Mehrdad  

**Link**: [PDF](https://arxiv.org/pdf/2611.18633)  

**Abstract**: A core of a graph is a nonempty vertex set in which every member has at least as many neighbours inside the set as outside. We study the minimum size $c_{m,n}$ of a finite core in the graph on $\Z^m$ joining points at sup-norm distance at most $n$. The local majority condition imposes a global geometric constraint: the convex hull of every core has an explicit lattice zonotope $H_n^{(m)}$ as a Minkowski summand. In the plane, this zonotope has width asymptotic to $6\zeta(3)n^3/\pi^2$. We prove that all its lattice points form a core and construct a shell with $O(n^4)$ points. Together with a packing argument, this gives $c_{2,n}=\Theta(n^4)$ and coordinate widths $\Theta(n^3)$ for every minimum core. We determine $c_{2,1}=12$ and $c_{2,2}=96$ by analytic arguments and prove uniqueness up to translation in both cases. Explicit forced-set and deficit certificates determine further minima within the prescribed zonotope hull. We distinguish these restricted results from global optimality and state the remaining geometric conjectures. We also give a direct reduction from \textsc{Clique} showing that the core problem is NP-complete on general finite graphs, which settles a conjecture of Hassanpour in the terms in which it was posed. 

---
# Interpretability-Guided Auditing for Clinical Language Models for Discharge Prediction 

**Authors**: Ha Na Cho, Sairam Sutari, Alexander Lopez, Hansen Bow, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.18626)  

**Abstract**: Clinical natural language processing (NLP) models for hospital discharge prediction are vulnerable to explicit discharge-related lexical cues in clinical notes, even when note timestamps are temporally filtered. These cues may act as proxies for future outcomes, leading models to rely on shortcut-like signals rather than clinically meaningful evidence. We propose AuditBERT-Discharge, a lightweight framework that reframes interpretability as a training-time auditing mechanism. Token-level SHAP attributions from an initial BioClinicalBERT model are used to identify highly attributed discharge-related tokens, which are subsequently masked in the training corpus before model retraining. We evaluate the framework on a retrospective cohort of elective spine surgery encounters from a single academic medical center. Qualitative attribution analysis and preliminary evaluation suggest that SHAP-guided auditing reduces reliance on explicit discharge-proxy language and yields more conservative, better-calibrated predictions, although improvements in discrimination are not uniform. These findings demonstrate the potential of interpretability-guided auditing as a practical strategy for identifying and constraining shortcut-like lexical dependencies in temporally sensitive clinical NLP tasks. 

---
# Skill Transfer: Adapting Agent Skills from Strong to Weak Agents 

**Authors**: Yifan Lan, Hanyu Wang, Lu Lin, Jinghui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.18624)  

**Abstract**: Agent skills are modular bundles of instructions, code, and resources that an agent loads on demand, increasingly authored and validated on frontier models and shared as an open standard. The same skill behaves differently across models: a strong model turns it into a large performance gain, while a weak model realizes little of that gain and sometimes does worse than with no skill at all. Cost and resource constraints, however, often restrict real deployments to exactly these weaker agents. We hypothesize that the strong model silently supplies procedural steps the skill leaves implicit, which the weak model omits, and formalize closing this gap as \emph{Skill Transfer}: holding the skill's capability ceiling and the model's parameters fixed, we adapt the skill so a specific weak agent can execute it well. Our method abstracts strong and weak trajectories into typed execution structures, localizes their structural divergence with a graph edit distance (GED), and repairs the skill through an iterative diagnoser-patcher loop. Across three benchmarks and two model families, Skill Transfer raises pass rate over the unmodified skill significantly and well over other skill-evolution baselines. The adapted skill also improves the strong agent it was never adapted for, and transfers out of distribution to a structurally related, unseen task. Code is available at this https URL 

---
# Evaluating the Robustness of Fact-Checking LLMs on Multilingual Evidence 

**Authors**: Radhika Shah  

**Link**: [PDF](https://arxiv.org/pdf/2611.18623)  

**Abstract**: Automated fact-checking benchmarks typically pair claims with clean, curated evidence in high-resource languages, but real-world evidence is often incomplete, misleading, or written in a language where models are weaker. We evaluate the robustness of fact-checking LLMs for different languages. Building on EuroVerdict, we construct gold evidence sets for 190 claims in each of English, German, Greek, and Romanian, then apply four corruption strategies at two severity levels. We test LLMs on a set of claims, providing them with evidence from the sets above. A rubric-based LLM-as-a-judge scores reasoning quality. Verdict accuracy proves highly sensitive to both factors: Misleading edit cut Llama 3.1 8B's English accuracy from 80.00% to 16.84%, and losses widen as language resource level decreases. Machine-translating Greek and Romanian claims into English recovers much, but not all, of the low-resource gap. Judge scores decline more slowly than accuracy, suggesting that automated reasoning evaluation underestimates degradation. These results caution against deploying LLM fact-checkers on noisy multilingual evidence, and we release our evidence datasets and evaluation code to support future work. 

---
# Which Mathematical Context Supports Target-Aligned Conjecture Generation? A Paired Ablation Study 

**Authors**: Austin LaHue, Michael Burr, Luis David Garcia Puente, Vinita Gangaram Jansari, Benjamin Nye, Carlos Toxtli Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2611.18621)  

**Abstract**: Generating mathematical conjectures from natural-language context is challenging because theorem-relevant information is distributed across multiple forms of mathematical context, including definitions, objects, assumptions, and constructions. We study which of these contextual signals help a model recover the specific claim supported by the surrounding setup, using 61 main theorems from 12 papers across four mathematical domains. For each theorem, we redact the result and represent the remaining setup using six context components. A paired ablation design compares full context, six leave-one-out conditions, and six single-component conditions, yielding 793 generated conjectures. Context components were largely complementary: removing one component usually had little effect, whereas retaining only one substantially reduced expected-theorem alignment. Local Mathematical Objects provided the strongest standalone signal, while Structural Constraints and Conditions produced the largest alignment decrease when removed. These findings suggest that target-aligned conjecture generation depends on combining information that identifies relevant objects with information that constrains admissible claims. 

---
# The Judge Is Not Impartial: Self-Preference in Medical LLM Evaluation 

**Authors**: Zara Ansari, Shlok Natarajan, Vincent Yip, Cally Lin, Avi Udash, Mia Garvey, Aaron Fanous, Roxana Daneshjou  

**Link**: [PDF](https://arxiv.org/pdf/2611.18616)  

**Abstract**: Large language models (LLMs) are increasingly used to evaluate model-generated medical content, creating a potential circularity: an LLM judge may favor outputs generated by the same model independently of clinical quality. To characterize LLM judge self-preference in medical settings, we measured same-model affinity in single-turn specialist question answering and multi-turn standardized-patient encounters. We evaluated eight models, two from OpenAI, Anthropic, Google, and Qwen, on 620 real point of care questions from clinicians (Real-POCQi). The same eight models also generated and judged 200 standardized patient-clinician scenarios (MedSP1000) up to 8 visible turns. In total, our analysis spans 13,760 LLM judgments. We define three complementary measures of self-preference. Answer Rank Bias, our primary measure, compared the rank assigned to a fixed response by its own-model judge with its average rank from outside judges. Answer Score Bias made the analogous comparison using rubric scores. Own-Answer Win Rate measured how often a judge ranked its own model's response above a competitor. Higher values across all these measures indicate greater degrees of self-preference. In Real-POCQi, pooled Answer Rank Bias was $+1.22$ positions, and all eight judges favored their own answers. Self-preference persisted both with and without a grading rubric. Revealing generator identities modestly weakened self-preference by an average of $-0.094$ positions. In MedSP1000, pooled Answer Rank Bias ranged from $+0.53$ to $+0.62$ positions, with no clear difference by conversation length ($p=0.17$). LLM judges systematically favored responses generated by their own models across multiple formats. Clinical AI deployments should exercise increased vigilance including multiple evaluation measures and judge panels spanning differing model families when deploying LLM-as-a-judge systems. 

---
# When Helpful Agents Cause Financial Harm: Evaluating LLM Agent Safety in Executable Economic Worlds 

**Authors**: Yuzhu Ling, Jiale Han, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18612)  

**Abstract**: As LLM agents become increasingly capable of making consequential economic decisions, their safety under economic pressures remains poorly understood. We introduce EconSafeBench, the first benchmark for systematically evaluating the safety of LLM agents in executable economic environments. EconSafeBench contains 828 cases across five economic scenarios and four categories of safety violations. A key design is the matched evaluation of three sources of safety pressure, including explicit user requests, pressure from other actors, and pressure arising from the agent's own pursuit of a performance objective. We further evaluate safety at both static and dynamic scales through 30-period interactions with persistent memory and matched controls. Across eight LLMs, all evaluated models exhibit safety violations on the static benchmark, with overall violation rates ranging from 8.5% to 40.6%. We further find that violations can occur even without an explicit unsafe request, and that initially compliant agents may begin to violate as performance pressure increases over time. These findings highlight the need for stronger safety alignment and rigorous evaluation before LLM agents are deployed in high-stakes economic environments. 

---
# Understanding Failures in LLM Reasoning by Learning Structured Representations of Chain-of-Thought 

**Authors**: Tommaso Felice Banfi, Mubashara Akhtar, Jingwei Ni, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2611.18611)  

**Abstract**: Chain-of-Thought (CoT) is central to improving reasoning in large language models, yet existing evaluation methods focus on step-level signals and ignore the underlying structure of reasoning traces. This raises a key question: what aspects of reasoning quality (i.e., correctness, efficiency, and failure patterns) are encoded in the structure of a CoT trace? We introduce \textbf{CoT2Graph}, a framework that represents CoT traces as directed dependency graphs, making relationships between reasoning steps explicit and capturing patterns such as backtracking, branching, and subtask merging. We model these graphs with graph neural networks and use attribution-based explanations to identify structural patterns associated with reasoning quality. CoT2Graph reveals distinct patterns of successful and failed reasoning: correct CoTs place greater importance on early planning, whereas incorrect CoTs are associated with overloaded synthesis steps and dependency violations. We further show that graph-based signals better predict whether a CoT leads to a correct answer, outperforming previous tree-based structural methods by up to 2.5\% and simple length-based baselines by 7.8\%, while also improving Best-of-$N$ selection. Overall, our results demonstrate that reasoning structure encodes rich signals about reasoning quality, motivating a shift from sequence-level evaluation to structural analysis. 

---
# Justification of Emotional Appeal in Computational Argumentation 

**Authors**: Yanran Chen, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2611.18607)  

**Abstract**: Emotional appeals (EAs) are often treated as fallacious in computational argumentation, yet they may be justified when warranted by the context. We introduce the task of predicting EA justification and present the first human-annotated dataset for this task, covering 1,176 argument--context pairs. Grounded in philosophical literature, our framework evaluates EA justification along logical and emotional dimensions. Human annotations show that justification criteria form a shared latent judgment space, with evidential proportionality and normative acceptability most strongly associated with overall justification. We further find that EAs without logical support are not always judged fully unjustified, and stronger EAs are not necessarily judged less justified. Finally, LLM experiments show that EA justification prediction remains challenging. 

---
# Syntactic Complexity Convergence in Dialogue: Analysis of the Phenomenon and Application to LLM Detection 

**Authors**: Yu Wang, Yanran Chen, Yifan Wang, Steffen Eger, Hendrik Buschmeier  

**Link**: [PDF](https://arxiv.org/pdf/2611.18606)  

**Abstract**: We revisit the under-explored phenomenon of syntactic complexity convergence in dialogue. Extending prior topic-level analyses to full conversations, we examine five human-human datasets and a set of human-LLM dialogues. We integrate and evaluate a multidimensional set of syntactic complexity metrics to quantify structural variation in dialogue. Our analyses reveal robust syntactic complexity convergence across human-human datasets, especially in task-oriented interactions, and statistical tests confirm that these effects are not due to chance. In contrast, human-LLM dialogues show no significant convergence under individual metrics; however, linear-CKA uncovers a distinct cross-metric convergence pattern. Motivated by these differences, we use syntactic complexity as a feature for detecting LLM-involved dialogue, achieving 96\% accuracy. Our classification results demonstrate its potential without relying on likelihood-based signals such as surprisal, which require a language model as the estimator. 

---
# One Question, Four Voices: How Advice for Alzheimer's Caregiving Differs Between Caregivers, Clinicians, and Large Language Models 

**Authors**: Congning Ni, Jinkyung Katie Park, Yang Li, Sarvech Qadir, Sean S. Huang, James S. Powers, Julia A. Hiner, Vania Leung, Patricia Commiskey, Lijun Song, Qingxia Chen, Pamela J. Wisniewski, Bradley A. Malin, Zhijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.18604)  

**Abstract**: Online caregiver forums are an important source of emotional and informational support for Alzheimer's disease and related dementias (ADRD). We compare responses to 85 real caregiver questions from ALZConnected across four responder groups: peer caregivers, physicians, ChatGPT, and a retrieval-augmented CareGPT grounded in peer discussions. Using 28 measures spanning linguistic form, emotional support, informational support, and topic scope, we examine how these voices differ when answering the same questions. Peers provided time-anchored narratives and community grounding; clinicians offered concise, advice-forward, clinically bounded guidance; and LLMs produced a formal, verbose, highly structured consultation style. CareGPT showed minimal measurable separation from ChatGPT, suggesting that retrieval plus prompt injection alone was insufficient to shift communicative stance. LLM outputs scored higher than peers on politeness and emotion-support markers and were comparable to clinicians on informativeness and physician-alignment measures, but were less narrative and less contextually situated. Topic modeling showed that LLM responses concentrated on advice-oriented themes, whereas peers covered a broader range, including emotional processing, legal and financial concerns, and daily care. These findings suggest that AI can complement, but not replace, human contributions to ADRD caregiver support and motivate hybrid systems that combine structured AI delivery with the distinct strengths of human voices. 

---
# RoboACE: Adversarial Co-Evolution for Robot Self-Improvement 

**Authors**: Delin Zhao, Zhengrong Yue, Junlin He, Yali Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.13034)  

**Abstract**: Robot agents can improve reusable skills, planning procedures, and recovery strategies from execution feedback, yet a fixed practice source leaves the search for new weaknesses unchanged. Useful practice depends on the agent's current capabilities, so the procedure that chooses practice must also be revisable. We present RoboACE (Robot Adversarial Co-Evolution), a framework that jointly evolves curriculum and execution harnesses around frozen foundation models. The curriculum harness uses accumulated capability evidence to propose informative challenges, while the execution harness responds with reusable revisions that expand execution capabilities. Alternating updates hold one side fixed as the other changes, and admitted challenges form fixed snapshots for comparing execution revisions. Starting from a compact set of LIBERO practice conditions, the resulting frozen deployment outperforms the evaluated controllers and agent systems across four LIBERO-family panels. On LIBERO-PRO, it reaches 74.0% success, exceeding the initial deployment by 15.0 percentage points and direct execution by its VLA controller by 13.5 points. Case studies document changes to evidence-guided practice selection, drawer-skill grounding, and controller intervention. 

---
# Sivia: Learning to Draw, Learning to Learn 

**Authors**: Feng Yuan, Xinyi Gao, Zijun Sun, Haoyue Li, Menglei Hua, Xiao Li, Yifan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2609.33386)  

**Abstract**: A paper's overview figure is often the first thing readers see: it must explain the method correctly and look polished. We present Sivia, an overview-figure system that reads the paper, plans its relations, adapts visual organization from recent top-conference papers, and composes a design for image generation. Sivia improves with use through human-guided recursive self-improvement (RSI) at two levels: learning to draw revises its drawing skill from feedback on accuracy and presentation, and learning to learn revises the updater that proposes those revisions. A personal profile adapts presentation to its user's aesthetic preferences while the scientific content stays fixed. A diagnostic study of 1,964 overviews for 445 papers (17,973 relation judgments) shows that most errors are omissions and that source-derived relation plans raise required-relation accuracy (RRA) by up to 19.85 points. On 48 test papers, learning both levels reaches 93.54\% RRA, 15.31, 9.38, and 8.33 points above the initial skill, the Fixed-updater, and the GEPA baselines, while readability and visual quality rise from 3.95 and 4.00 to 4.70 and 4.78 on five-point scales. Learned presentation profiles raise liking from 4.02 to 4.73 and adoption from 34 to 45 of 48 tasks. Code and released artifacts are available at this https URL 

---
# ChartSync: A Benchmark for Visuo-Logical Cascading Chart Editing 

**Authors**: Jiakang Yu, Yixuan Chai, Tianci Wang, Rihui Jin, Guangkai Xu, Hongtao Deng, Xun Zhu, Wang Gao, Xinrun Guo, Haipang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.10301)  

**Abstract**: Generative image editing models struggle with structured statistical charts when data modifications require geometric synchronization. We formalize this task as Visuo-Logical Cascading Editing (VLCE). However, existing methods remain confined to localized text substitutions and struggle with dependency-aware cascading updates. To systematically evaluate this capability, we introduce ChartSync, an expert-validated benchmark constructed via a programmatic rendering pipeline that guarantees deterministic visuo-logical coupling for the ground truth. ChartSync comprises 870 triplets across 9 chart categories and 4 task types, including 235 geometry-coupled VLCE instances that specifically test cascading text-to-geometry synchronization. We further evaluate these instances via a two-tier framework combining objective visual metrics with a vision-language model judge paradigm to assess low-level fidelity alongside multimodal comprehension and reasoning. Evaluating 14 image editing models and one code-mediated pipeline reveals a nuanced capability gap: most open-source models suffer severe drops in geometric synchronization, while only two frontier proprietary models show emerging VLCE capability, with their residual errors mainly involving semantic isolation and background corruption. Our detailed error analysis deconstructs these failure paradigms to identify core meta-abilities for guiding future multimodal architectures. The ChartSync dataset and code are publicly released at this https URL. 

---
# Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws 

**Authors**: Jinbo Wang, Binghui Li, Zhanpeng Zhou, Mingze Wang, Yuxuan Sun, Jiaqi Zhang, Xunliang Cai, Lei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.14208)  

**Abstract**: Batch size scheduling (BSS) plays a critical role in large-scale deep learning training, influencing both optimization dynamics and computational efficiency. Yet, its theoretical foundations remain poorly understood. In this work, we show that the functional scaling law (FSL) framework introduced in Li et al. (2025a) provides a principled lens for analyzing BSS. Specifically, we characterize the optimal BSS under a fixed data budget and show that its structure depends sharply on task difficulty. For easy tasks, optimal schedules keep increasing batch size throughout. In contrast, for hard tasks, the optimal schedule maintains small batch sizes for most of training and switches to large batches only in a late stage. To explain the emergence of late switching, we uncover a dynamical mechanism -- the fast catch-up effect -- which also manifests in large language model (LLM) pretraining. After switching from small to large batches, the loss rapidly aligns with the constant large-batch trajectory. Using FSL, we show that this effect stems from rapid forgetting of accumulated gradient noise, with the catch-up speed determined by task difficulty. Crucially, this effect implies that large batches can be safely deferred to late training without sacrificing performance, while substantially reducing data consumption. Finally, extensive LLM pretraining experiments -- covering both Dense and MoE architectures with up to 1.1B parameters and 1T tokens -- validate our theoretical predictions. Across all settings, late-switch schedules consistently outperform constant-batch and early-switch baselines. 

---
# An XAI Approach to Deep Learning Models in the Detection of DCIS 

**Authors**: Michele La Ferla, Matthew Montebello, Dylan Seychell  

**Link**: [PDF](https://arxiv.org/pdf/2106.14186)  

**Abstract**: The results showed that XAI could indeed be used as a proof of concept to begin discussions on the implementation of assistive AI systems within the clinical community. 

---
