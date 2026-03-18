# SocialOmni: Benchmarking Audio-Visual Social Interactivity in Omni Models 

**Authors**: Tianyu Xie, Jinfa Huang, Yuexiao Ma, Rongfang Luo, Yan Yang, Wang Chen, Yuhui Zeng, Ruize Fang, Yixuan Zou, Xiawu Zheng, Jiebo Luo, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2603.16859)  

**Abstract**: Omni-modal large language models (OLMs) redefine human-machine interaction by natively integrating audio, vision, and text. However, existing OLM benchmarks remain anchored to static, accuracy-centric tasks, leaving a critical gap in assessing social interactivity, the fundamental capacity to navigate dynamic cues in natural dialogues. To this end, we propose SocialOmni, a comprehensive benchmark that operationalizes the evaluation of this conversational interactivity across three core dimensions: (i) speaker separation and identification (who is speaking), (ii) interruption timing control (when to interject), and (iii) natural interruption generation (how to phrase the interruption). SocialOmni features 2,000 perception samples and a quality-controlled diagnostic set of 209 interaction-generation instances with strict temporal and contextual constraints, complemented by controlled audio-visual inconsistency scenarios to test model robustness. We benchmarked 12 leading OLMs, which uncovers significant variance in their social-interaction capabilities across models. Furthermore, our analysis reveals a pronounced decoupling between a model's perceptual accuracy and its ability to generate contextually appropriate interruptions, indicating that understanding-centric metrics alone are insufficient to characterize conversational social competence. More encouragingly, these diagnostics from SocialOmni yield actionable signals for bridging the perception-interaction divide in future OLMs. 

---
# Internalizing Agency from Reflective Experience 

**Authors**: Rui Ge, Yichao Fu, Yuyang Qian, Junda Su, Yiming Zhao, Peng Zhao, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16843)  

**Abstract**: Large language models are increasingly deployed as autonomous agents that must plan, act, and recover from mistakes through long-horizon interaction with environments that provide rich feedback. However, prevailing outcome-driven post-training methods (e.g., RL with verifiable rewards) primarily optimize final success signals, leaving rich environment feedback underutilized. Consequently, they often lead to distribution sharpening: the policy becomes better at reproducing a narrow set of already-successful behaviors, while failing to improve the feedback-grounded agency needed to expand problem-solving capacity (e.g., Pass@k) in long-horizon settings.
To address this, we propose LEAFE (Learning Feedback-Grounded Agency from Reflective Experience), a framework that internalizes recovery agency from reflective experience. Specifically, during exploration, the agent summarizes environment feedback into actionable experience, backtracks to earlier decision points, and explores alternative branches with revised actions. We then distill these experience-guided corrections into the model through supervised fine-tuning, enabling the policy to recover more effectively in future interactions. Across a diverse set of interactive coding and agentic tasks under fixed interaction budgets, LEAFE consistently improves Pass@1 over the base model and achieves higher Pass@k than outcome-driven baselines (GRPO) and experience-based methods such as Early Experience, with gains of up to 14% on Pass@128. 

---
# Learning to Present: Inverse Specification Rewards for Agentic Slide Generation 

**Authors**: Karthik Ragunath Ananda Kumar, Subrahmanyam Arunachalam  

**Link**: [PDF](https://arxiv.org/pdf/2603.16839)  

**Abstract**: Automated presentation generation remains a challenging task requiring coherent content creation, visual design, and audience-aware communication. This work proposes an OpenEnv-compatible reinforcement learning environment where LLM agents learn to research topics, plan content, and generate professional HTML slide presentations through tool use. We introduce a multi-component reward system combining structural validation, render quality assessment, LLM-based aesthetic scoring, content quality metrics, and an inverse specification reward that measures how faithfully generated slides convey their intended purpose. The inverse specification reward, an "inverse task" where an LLM attempts to recover the original specification from generated slides, provides a holistic quality signal. Our approach fine-tunes Qwen2.5-Coder-7B via GRPO, training only 0.5% of parameters on prompts derived from expert demonstrations collected using Claude Opus 4.6. Experiments on 48 diverse business briefs across six models demonstrate that our fine-tuned 7B model achieves 91.2% of Claude Opus 4.6's quality while improving 33.1% over the base model. The six-model comparison reveals that instruction adherence and tool-use compliance, rather than raw parameter count, determine agentic task performance. We contribute SlideRL, an open-source dataset of 288 multi-turn rollout trajectories across all six models: this https URL Code: this https URL 

---
# Prompt Programming for Cultural Bias and Alignment of Large Language Models 

**Authors**: Maksim Eren, Eric Michalak, Brian Cook, Johnny Seales Jr  

**Link**: [PDF](https://arxiv.org/pdf/2603.16827)  

**Abstract**: Culture shapes reasoning, values, prioritization, and strategic decision-making, yet large language models (LLMs) often exhibit cultural biases that misalign with target populations. As LLMs are increasingly used for strategic decision-making, policy support, and document engineering tasks such as summarization, categorization, and compliance-oriented auditing, improving cultural alignment is important for ensuring that downstream analyses and recommendations reflect target-population value profiles rather than default model priors. Previous work introduced a survey-grounded cultural alignment framework and showed that culture-specific prompting can reduce misalignment, but it primarily evaluated proprietary models and relied on manual prompt engineering. In this paper, we validate and extend that framework by reproducing its social sciences survey based projection and distance metrics on open-weight LLMs, testing whether the same cultural skew and benefits of culture conditioning persist outside closed LLM systems. Building on this foundation, we introduce use of prompt programming with DSPy for this problem-treating prompts as modular, optimizable programs-to systematically tune cultural conditioning by optimizing against cultural-distance objectives. In our experiments, we show that prompt optimization often improves upon cultural prompt engineering, suggesting prompt compilation with DSPy can provide a more stable and transferable route to culturally aligned LLM responses. 

---
# Surg$Σ$: A Spectrum of Large-Scale Multimodal Data and Foundation Models for Surgical Intelligence 

**Authors**: Zhitao Zeng, Mengya Xu, Jian Jiang, Pengfei Guo, Yunqiu Xu, Zhu Zhuo, Chang Han Low, Yufan He, Dong Yang, Chenxi Lin, Yiming Gu, Jiaxin Guo, Yutong Ban, Daguang Xu, Qi Dou, Yueming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.16822)  

**Abstract**: Surgical intelligence has the potential to improve the safety and consistency of surgical care, yet most existing surgical AI frameworks remain task-specific and struggle to generalize across procedures and institutions. Although multimodal foundation models, particularly multimodal large language models, have demonstrated strong cross-task capabilities across various medical domains, their advancement in surgery remains constrained by the lack of large-scale, systematically curated multimodal data. To address this challenge, we introduce Surg$\Sigma$, a spectrum of large-scale multimodal data and foundation models for surgical intelligence. At the core of this framework lies Surg$\Sigma$-DB, a large-scale multimodal data foundation designed to support diverse surgical tasks. Surg$\Sigma$-DB consolidates heterogeneous surgical data sources (including open-source datasets, curated in-house clinical collections and web-source data) into a unified schema, aiming to improve label consistency and data standardization across heterogeneous datasets. Surg$\Sigma$-DB spans 6 clinical specialties and diverse surgical types, providing rich image- and video-level annotations across 18 practical surgical tasks covering understanding, reasoning, planning, and generation, at an unprecedented scale (over 5.98M conversations). Beyond conventional multimodal conversations, Surg$\Sigma$-DB incorporates hierarchical reasoning annotations, providing richer semantic cues to support deeper contextual understanding in complex surgical scenarios. We further provide empirical evidence through recently developed surgical foundation models built upon Surg$\Sigma$-DB, illustrating the practical benefits of large-scale multimodal annotations, unified semantic design, and structured reasoning annotations for improving cross-task generalization and interpretability. 

---
# Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights 

**Authors**: Yi Chen, Daiwei Chen, Sukrut Madhav Chikodikar, Caitlyn Heqi Yin, Ramya Korlakai Vinayak  

**Link**: [PDF](https://arxiv.org/pdf/2603.16817)  

**Abstract**: Large language models (LLMs) frequently hallucinate, limiting their reliability in knowledge-intensive applications. Retrieval-augmented generation (RAG) and conformal factuality have emerged as potential ways to address this limitation. While RAG aims to ground responses in retrieved evidence, it provides no statistical guarantee that the final output is correct. Conformal factuality filtering offers distribution-free statistical reliability by scoring and filtering atomic claims using a threshold calibrated on held-out data, however, the informativeness of the final output is not guaranteed. We systematically analyze the reliability and usefulness of conformal factuality for RAG-based LLMs across generation, scoring, calibration, robustness, and efficiency. We propose novel informativeness-aware metrics that better reflect task utility under conformal filtering. Across three benchmarks and multiple model families, we find that (i) conformal filtering suffers from low usefulness at high factuality levels due to vacuous outputs, (ii) conformal factuality guarantee is not robust to distribution shifts and distractors, highlighting the limitation that requires calibration data to closely match deployment conditions, and (iii) lightweight entailment-based verifiers match or outperform LLM-based model confidence scorers while requiring over $100\times$ fewer FLOPs. Overall, our results expose factuality-informativeness trade-offs and fragility of conformal filtering framework under distribution shifts and distractors, highlighting the need for new approaches for reliability with robustness and usefulness as key metrics, and provide actionable guidance for building RAG pipelines that are both reliable and computationally efficient. 

---
# Beyond Accuracy: Evaluating Forecasting Models by Multi-Echelon Inventory Cost 

**Authors**: Swata Marik, Swayamjit Saha, Garga Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2603.16815)  

**Abstract**: This study develops a digitalized forecasting-inventory optimization pipeline integrating traditional forecasting models, machine learning regressors, and deep sequence models within a unified inventory simulation framework. Using the M5 Walmart dataset, we evaluate seven forecasting approaches and assess their operational impact under single- and two-echelon newsvendor systems. Results indicate that Temporal CNN and LSTM models significantly reduce inventory costs and improve fill rates compared to statistical baselines. Sensitivity and multi-echelon analyses demonstrate robustness and scalability, offering a data-driven decision-support tool for modern supply chains. 

---
# Anticipatory Planning for Multimodal AI Agents 

**Authors**: Yongyuan Liang, Shijie Zhou, Yu Gu, Hao Tan, Gang Wu, Franck Dernoncourt, Jihyung Kil, Ryan A. Rossi, Ruiyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16777)  

**Abstract**: Recent advances in multimodal agents have improved computer-use interaction and tool-usage, yet most existing systems remain reactive, optimizing actions in isolation without reasoning about future states or long-term goals. This limits planning coherence and prevents agents from reliably solving high-level, multi-step tasks. We introduce TraceR1, a two-stage reinforcement learning framework that explicitly trains anticipatory reasoning by forecasting short-horizon trajectories before execution. The first stage performs trajectory-level reinforcement learning with rewards that enforce global consistency across predicted action sequences. The second stage applies grounded reinforcement fine-tuning, using execution feedback from frozen tool agents to refine step-level accuracy and executability. TraceR1 is evaluated across seven benchmarks, covering online computer-use, offline computer-use benchmarks, and multimodal tool-use reasoning tasks, where it achieves substantial improvements in planning stability, execution robustness, and generalization over reactive and single-stage baselines. These results show that anticipatory trajectory reasoning is a key principle for building multimodal agents that can reason, plan, and act effectively in complex real-world environments. 

---
# Nonstandard Errors in AI Agents 

**Authors**: Ruijiang Gao, Steven Chong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16744)  

**Abstract**: We study whether state-of-the-art AI coding agents, given the same data and research question, produce the same empirical results. Deploying 150 autonomous Claude Code agents to independently test six hypotheses about market quality trends in NYSE TAQ data for SPY (2015--2024), we find that AI agents exhibit sizable \textit{nonstandard errors} (NSEs), that is, uncertainty from agent-to-agent variation in analytical choices, analogous to those documented among human researchers. AI agents diverge substantially on measure choice (e.g., autocorrelation vs.\ variance ratio, dollar vs.\ share volume). Different model families (Sonnet 4.6 vs.\ Opus 4.6) exhibit stable ``empirical styles,'' reflecting systematic differences in methodological preferences. In a three-stage feedback protocol, AI peer review (written critiques) has minimal effect on dispersion, whereas exposure to top-rated exemplar papers reduces the interquartile range of estimates by 80--99\% within \textit{converging} measure families. Convergence occurs both through within-family estimation tightening and through agents switching measure families entirely, but convergence reflects imitation rather than understanding. These findings have implications for the growing use of AI in automated policy evaluation and empirical research. 

---
# MedCL-Bench: Benchmarking stability-efficiency trade-offs and scaling in biomedical continual learning 

**Authors**: Min Zeng, Shuang Zhou, Zaifu Zhan, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16738)  

**Abstract**: Medical language models must be updated as evidence and terminology evolve, yet sequential updating can trigger catastrophic forgetting. Although biomedical NLP has many static benchmarks, no unified, task-diverse benchmark exists for evaluating continual learning under standardized protocols, robustness to task order and compute-aware reporting. We introduce MedCL-Bench, which streams ten biomedical NLP datasets spanning five task families and evaluates eleven continual learning strategies across eight task orders, reporting retention, transfer, and GPU-hour cost. Across backbones and task orders, direct sequential fine-tuning on incoming tasks induces catastrophic forgetting, causing update-induced performance regressions on prior tasks. Continual learning methods occupy distinct retention-compute frontiers: parameter-isolation provides the best retention per GPU-hour, replay offers strong protection at higher cost, and regularization yields limited benefit. Forgetting is task-dependent, with multi-label topic classification most vulnerable and constrained-output tasks more robust. MedCL-Bench provides a reproducible framework for auditing model updates before deployment. 

---
# Differential Harm Propensity in Personalized LLM Agents: The Curious Case of Mental Health Disclosure 

**Authors**: Caglar Yildirim  

**Link**: [PDF](https://arxiv.org/pdf/2603.16734)  

**Abstract**: Large language models (LLMs) are increasingly deployed as tool-using agents, shifting safety concerns from harmful text generation to harmful task completion. Deployed systems often condition on user profiles or persistent memory, yet agent safety evaluations typically ignore personalization signals. To address this gap, we investigated how mental health disclosure, a sensitive and realistic user-context cue, affects harmful behavior in agentic settings. Building on the AgentHarm benchmark, we evaluated frontier and open-source LLMs on multi-step malicious tasks (and their benign counterparts) under controlled prompt conditions that vary user-context personalization (no bio, bio-only, bio+mental health disclosure) and include a lightweight jailbreak injection. Our results reveal that harmful task completion is non-trivial across models: frontier lab models (e.g., GPT 5.2, Claude Sonnet 4.5, Gemini 3-Pro) still complete a measurable fraction of harmful tasks, while an open model (DeepSeek 3.2) exhibits substantially higher harmful completion. Adding a bio-only context generally reduces harm scores and increases refusals. Adding an explicit mental health disclosure often shifts outcomes further in the same direction, though effects are modest and not uniformly reliable after multiple-testing correction. Importantly, the refusal increase also appears on benign tasks, indicating a safety--utility trade-off via over-refusal. Finally, jailbreak prompting sharply elevates harm relative to benign conditions and can weaken or override the protective shift induced by personalization. Taken together, our results indicate that personalization can act as a weak protective factor in agentic misuse settings, but it is fragile under minimal adversarial pressure, highlighting the need for personalization-aware evaluations and safeguards that remain robust across user-context conditions. 

---
# IQuest-Coder-V1 Technical Report 

**Authors**: Jian Yang, Wei Zhang, Shawn Guo, Zhengmao Ye, Lin Jing, Shark Liu, Yizhi Li, Jiajun Wu, Cening Liu, X. Ma, Yuyang Song, Siwei Wu, Yuwen Li, L. Liao, T. Zheng, Ziling Huang, Zelong Huang, Che Liu, Yan Xing, Renyuan Li, Qingsong Cai, Hanxu Yan, Siyue Wang, Shikai Li, Jason Klein Liu, An Huang, Yongsheng Kang, Jinxing Zhang, Chuan Hao, Haowen Wang, Weicheng Gu, Ran Tao, Mingjie Tang, Peihao Wu, Jianzhou Wang, Xianglong Liu, Weifeng Lv, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2603.16733)  

**Abstract**: In this report, we introduce the IQuest-Coder-V1 series-(7B/14B/40B/40B-Loop), a new family of code large language models (LLMs). Moving beyond static code representations, we propose the code-flow multi-stage training paradigm, which captures the dynamic evolution of software logic through different phases of the pipeline. Our models are developed through the evolutionary pipeline, starting with the initial pre-training consisting of code facts, repository, and completion data. Following that, we implement a specialized mid-training stage that integrates reasoning and agentic trajectories in 32k-context and repository-scale in 128k-context to forge deep logical foundations. The models are then finalized with post-training of specialized coding capabilities, which is bifurcated into two specialized paths: the thinking path (utilizing reasoning-driven RL) and the instruct path (optimized for general assistance). IQuest-Coder-V1 achieves state-of-the-art performance among competitive models across critical dimensions of code intelligence: agentic software engineering, competitive programming, and complex tool use. To address deployment constraints, the IQuest-Coder-V1-Loop variant introduces a recurrent mechanism designed to optimize the trade-off between model capacity and deployment footprint, offering an architecturally enhanced path for efficacy-efficiency trade-off. We believe the release of the IQuest-Coder-V1 series, including the complete white-box chain of checkpoints from pre-training bases to the final thinking and instruction models, will advance research in autonomous code intelligence and real-world agentic systems. 

---
# CritiSense: Critical Digital Literacy and Resilience Against Misinformation 

**Authors**: Firoj Alam, Fatema Ahmad, Ali Ezzat Shahroor, Mohamed Bayan Kmainasi, Elisa Sartori, Giovanni Da San Martino, Abul Hasnat, Raian Ali  

**Link**: [PDF](https://arxiv.org/pdf/2603.16672)  

**Abstract**: Misinformation on social media undermines informed decision-making and public trust. Prebunking offers a proactive complement by helping users recognize manipulation tactics before they encounter them in the wild. We present CritiSense, a mobile media-literacy app that builds these skills through short, interactive challenges with instant feedback. It is the first multilingual (supporting nine languages) and modular platform, designed for rapid updates across topics and domains. We report a usability study with 93 users: 83.9% expressed overall satisfaction and 90.1% rated the app as easy to use. Qualitative feedback indicates that CritiSense helps improve digital literacy skills. Overall, it provides a multilingual prebunking platform and a testbed for measuring the impact of microlearning on misinformation resilience. Over 3+ months, we have reached 300+ active users. It is freely available to all users on the Apple App Store (this https URL) and Google Play Store (this https URL). Demo Video: this https URL 

---
# Machines acquire scientific taste from institutional traces 

**Authors**: Ziqin Gong, Ning Li, Huaikang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.16659)  

**Abstract**: Artificial intelligence matches or exceeds human performance on tasks with verifiable answers, from protein folding to Olympiad mathematics. Yet the capacity that most governs scientific advance is not reasoning but taste: the ability to judge which untested ideas deserve pursuit, exercised daily by editors and funders but never successfully articulated, taught, or automated. Here we show that fine-tuning language models on journal publication decisions recovers evaluative judgment inaccessible to both frontier models and human expertise. Using a held-out benchmark of research pitches in management spanning four quality tiers, we find that eleven frontier models, spanning major proprietary and open architectures, barely exceed chance, averaging 31% accuracy. Panels of journal editors and editorial board members reach 42% by majority vote. Fine-tuned models trained on years of publication records each surpass every frontier model and expert panel, with the best single model achieving 59%. These models exhibit calibrated confidence, reaching 100% accuracy on their highest-confidence predictions, and transfer this evaluative signal to untrained pairwise comparisons and one-sentence summaries. The mechanism generalizes: models trained on economics publication records achieve 70% accuracy. Scientific taste was not missing from AI's reach; it was deposited in the institutional record, waiting to be extracted. These results provide a scalable mechanism to triage the expanding volume of scientific production across disciplines where quality resists formal verification. 

---
# What if Pinocchio Were a Reinforcement Learning Agent: A Normative End-to-End Pipeline 

**Authors**: Benoît Alcaraz  

**Link**: [PDF](https://arxiv.org/pdf/2603.16651)  

**Abstract**: In the past decade, artificial intelligence (AI) has developed quickly. With this rapid progression came the need for systems capable of complying with the rules and norms of our society so that they can be successfully and safely integrated into our daily lives. Inspired by the story of Pinocchio in ``Le avventure di Pinocchio - Storia di un burattino'', this thesis proposes a pipeline that addresses the problem of developing norm compliant and context-aware agents. Building on the AJAR, Jiminy, and NGRL architectures, the work introduces \pino, a hybrid model in which reinforcement learning agents are supervised by argumentation-based normative advisors. In order to make this pipeline operational, this thesis also presents a novel algorithm for automatically extracting the arguments and relationships that underlie the advisors' decisions. Finally, this thesis investigates the phenomenon of \textit{norm avoidance}, providing a definition and a mitigation strategy within the context of reinforcement learning agents. Each component of the pipeline is empirically evaluated. The thesis concludes with a discussion of related work, current limitations, and directions for future research. 

---
# Domain-Independent Dynamic Programming with Constraint Propagation 

**Authors**: Imko Marijnissen, J. Christopher Beck, Emir Demirović, Ryo Kuroiwa  

**Link**: [PDF](https://arxiv.org/pdf/2603.16648)  

**Abstract**: There are two prevalent model-based paradigms for combinatorial problems: 1) state-based representations, such as heuristic search, dynamic programming (DP), and decision diagrams, and 2) constraint and domain-based representations, such as constraint programming (CP), (mixed-)integer programming, and Boolean satisfiability. In this paper, we bridge the gap between the DP and CP paradigms by integrating constraint propagation into DP, enabling a DP solver to prune states and transitions using constraint propagation. To this end, we implement constraint propagation using a general-purpose CP solver in the Domain-Independent Dynamic Programming framework and evaluate using heuristic search on three combinatorial optimisation problems: Single Machine Scheduling with Time Windows, the Resource Constrained Project Scheduling Problem (RCPSP), and the Travelling Salesperson Problem with Time Windows (TSPTW). Our evaluation shows that constraint propagation significantly reduces the number of state expansions, causing our approach to solve more instances than a DP solver for Single Machine Scheduling and RCPSP, and showing similar improvements for tightly constrained TSPTW instances. The runtime performance indicates that the benefits of propagation outweigh the overhead for constrained instances, but that further work into reducing propagation overhead could improve performance further. Our work is a key step in understanding the value of constraint propagation in DP solvers, providing a model-based approach to integrating DP and CP. 

---
# When AI Navigates the Fog of War 

**Authors**: Ming Li, Xirui Li, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.16642)  

**Abstract**: Can AI reason about a war before its trajectory becomes historically obvious? Analyzing this capability is difficult because retrospective geopolitical prediction is heavily confounded by training-data leakage. We address this challenge through a temporally grounded case study of the early stages of the 2026 Middle East conflict, which unfolded after the training cutoff of current frontier models. We construct 11 critical temporal nodes, 42 node-specific verifiable questions, and 5 general exploratory questions, requiring models to reason only from information that would have been publicly available at each moment. This design substantially mitigates training-data leakage concerns, creating a setting well-suited for studying how models analyze an unfolding crisis under the fog of war, and provides, to our knowledge, the first temporally grounded analysis of LLM reasoning in an ongoing geopolitical conflict. Our analysis reveals three main findings. First, current state-of-the-art large language models often display a striking degree of strategic realism, reasoning beyond surface rhetoric toward deeper structural incentives. Second, this capability is uneven across domains: models are more reliable in economically and logistically structured settings than in politically ambiguous multi-actor environments. Finally, model narratives evolve over time, shifting from early expectations of rapid containment toward more systemic accounts of regional entrenchment and attritional de-escalation. Since the conflict remains ongoing at the time of writing, this work can serve as an archival snapshot of model reasoning during an unfolding geopolitical crisis, enabling future studies without the hindsight bias of retrospective analysis. 

---
# Runtime Governance for AI Agents: Policies on Paths 

**Authors**: Maurits Kaptein, Vassilis-Javed Khan, Andriy Podstavnychy  

**Link**: [PDF](https://arxiv.org/pdf/2603.16586)  

**Abstract**: AI agents -- systems that plan, reason, and act using large language models -- produce non-deterministic, path-dependent behavior that cannot be fully governed at design time, where with governed we mean striking the right balance between as high as possible successful task completion rate and the legal, data-breach, reputational and other costs associated with running agents. We argue that the execution path is the central object for effective runtime governance and formalize compliance policies as deterministic functions mapping agent identity, partial path, proposed next action, and organizational state to a policy violation probability. We show that prompt-level instructions (and "system prompts"), and static access control are special cases of this framework: the former shape the distribution over paths without actually evaluating them; the latter evaluates deterministic policies that ignore the path (i.e., these can only account for a specific subset of all possible paths). In our view, runtime evaluation is the general case, and it is necessary for any path-dependent policy. We develop the formal framework for analyzing AI agent governance, present concrete policy examples (inspired by the AI act), discuss a reference implementation, and identify open problems including risk calibration and the limits of enforced compliance. 

---
# V-DyKnow: A Dynamic Benchmark for Time-Sensitive Knowledge in Vision Language Models 

**Authors**: Seyed Mahed Mousavi, Christian Moiola, Massimo Rizzoli, Simone Alghisi, Giuseppe Riccardi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16581)  

**Abstract**: Vision-Language Models (VLMs) are trained on data snapshots of documents, including images and texts. Their training data and evaluation benchmarks are typically static, implicitly treating factual knowledge as time-invariant. However, real-world facts are intrinsically time-sensitive and subject to erratic and periodic changes, causing model predictions to become outdated. We present V-DyKnow, a Visual Dynamic Knowledge benchmark for evaluating time-sensitive factual knowledge in VLMs. Using V-DyKnow, we benchmark closed- and open-source VLMs and analyze a) the reliability (correctness and consistency) of model responses across modalities and input perturbations; b) the efficacy of knowledge editing and multi-modal RAG methods for knowledge updates across modalities; and c) the sources of outdated predictions, through data and mechanistic analysis. Our results show that VLMs frequently output outdated facts, reflecting outdated snapshots used in the (pre-)training phase. Factual reliability degrades from textual to visual stimuli, even when entities are correctly recognized. Besides, existing alignment approaches fail to consistently update the models' knowledge across modalities. Together, these findings highlight fundamental limitations in how current VLMs acquire and update time-sensitive knowledge across modalities. We release the benchmark, code, and evaluation data. 

---
# BenchPreS: A Benchmark for Context-Aware Personalized Preference Selectivity of Persistent-Memory LLMs 

**Authors**: Sangyeon Yoon, Sunkyoung Kim, Hyesoo Hong, Wonje Jeung, Yongil Kim, Wooseok Seo, Heuiyeen Yeen, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2603.16557)  

**Abstract**: Large language models (LLMs) increasingly store user preferences in persistent memory to support personalization across interactions. However, in third-party communication settings governed by social and institutional norms, some user preferences may be inappropriate to apply. We introduce BenchPreS, which evaluates whether memory-based user preferences are appropriately applied or suppressed across communication contexts. Using two complementary metrics, Misapplication Rate (MR) and Appropriate Application Rate (AAR), we find even frontier LLMs struggle to apply preferences in a context-sensitive manner. Models with stronger preference adherence exhibit higher rates of over-application, and neither reasoning capability nor prompt-based defenses fully resolve this issue. These results suggest current LLMs treat personalized preferences as globally enforceable rules rather than as context-dependent normative signals. 

---
# Designing for Disagreement: Front-End Guardrails for Assistance Allocation in LLM-Enabled Robots 

**Authors**: Carmen Ng  

**Link**: [PDF](https://arxiv.org/pdf/2603.16537)  

**Abstract**: LLM-enabled robots prioritizing scarce assistance in social settings face pluralistic values and LLM behavioral variability: reasonable people can disagree about who is helped first, while LLM-mediated interaction policies vary across prompts, contexts, and groups in ways that are difficult to anticipate or verify at contact point. Yet user-facing guardrails for real-time, multi-user assistance allocation remain under-specified. We propose bounded calibration with contestability, a procedural front-end pattern that (i) constrains prioritization to a governance-approved menu of admissible modes, (ii) keeps the active mode legible in interaction-relevant terms at the point of deferral, and (iii) provides an outcome-specific contest pathway without renegotiating the global rule. Treating pluralism and LLM uncertainty as standing conditions, the pattern avoids both silent defaults that hide implicit value skews and wide-open user-configurable "value settings" that shift burden under time pressure. We illustrate the pattern with a public-concourse robot vignette and outline an evaluation agenda centered on legibility, procedural legitimacy, and actionability, including risks of automation bias and uneven usability of contest channels. 

---
# Exploring different approaches to customize language models for domain-specific text-to-code generation 

**Authors**: Luís Freire, Fernanda A. Andaló, Nicki Skafte Detlefsen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16526)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in generating executable code from natural language descriptions. However, general-purpose models often struggle in specialized programming contexts where domain-specific libraries, APIs, or conventions must be used. Customizing smaller open-source models offers a cost-effective alternative to relying on large proprietary systems. In this work, we investigate how smaller language models can be adapted for domain-specific code generation using synthetic datasets. We construct datasets of programming exercises across three domains within the Python ecosystem: general Python programming, Scikit-learn machine learning workflows, and OpenCV-based computer vision tasks. Using these datasets, we evaluate three customization strategies: few-shot prompting, retrieval-augmented generation (RAG), and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA). Performance is evaluated using both benchmark-based metrics and similarity-based metrics that measure alignment with domain-specific code. Our results show that prompting-based approaches such as few-shot learning and RAG can improve domain relevance in a cost-effective manner, although their impact on benchmark accuracy is limited. In contrast, LoRA-based fine-tuning consistently achieves higher accuracy and stronger domain alignment across most tasks. These findings highlight practical trade-offs between flexibility, computational cost, and performance when adapting smaller language models for specialized programming tasks. 

---
# ExpressMind: A Multimodal Pretrained Large Language Model for Expressway Operation 

**Authors**: Zihe Wang, Yihuan Wang, Haiyang Yu. Zhiyong Cui, Xiaojian Liao, Chengcheng Wang, Yonglin Tian, Yongxin Tong  

**Link**: [PDF](https://arxiv.org/pdf/2603.16495)  

**Abstract**: The current expressway operation relies on rule-based and isolated models, which limits the ability to jointly analyze knowledge across different systems. Meanwhile, Large Language Models (LLMs) are increasingly applied in intelligent transportation, advancing traffic models from algorithmic to cognitive intelligence. However, general LLMs are unable to effectively understand the regulations and causal relationships of events in unconventional scenarios in the expressway field. Therefore, this paper constructs a pre-trained multimodal large language model (MLLM) for expressways, ExpressMind, which serves as the cognitive core for intelligent expressway operations. This paper constructs the industry's first full-stack expressway dataset, encompassing traffic knowledge texts, emergency reasoning chains, and annotated video events to overcome data scarcity. This paper proposes a dual-layer LLM pre-training paradigm based on self-supervised training and unsupervised learning. Additionally, this study introduces a Graph-Augmented RAG framework to dynamically index the expressway knowledge base. To enhance reasoning for expressway incident response strategies, we develop a RL-aligned Chain-of-Thought (RL-CoT) mechanism that enforces consistency between model reasoning and expert problem-solving heuristics for incident handling. Finally, ExpressMind integrates a cross-modal encoder to align the dynamic feature sequences under the visual and textual channels, enabling it to understand traffic scenes in both video and image modalities. Extensive experiments on our newly released multi-modal expressway benchmark demonstrate that ExpressMind comprehensively outperforms existing baselines in event detection, safety response generation, and complex traffic analysis. The code and data are available at: this https URL. 

---
# Breaking the Chain: A Causal Analysis of LLM Faithfulness to Intermediate Structures 

**Authors**: Oleg Somov, Mikhail Chaichuk, Mikhail Seleznyov, Alexander Panchenko, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2603.16475)  

**Abstract**: Schema-guided reasoning pipelines ask LLMs to produce explicit intermediate structures -- rubrics, checklists, verification queries -- before committing to a final decision. But do these structures causally determine the output, or merely accompany it? We introduce a causal evaluation protocol that makes this directly measurable: by selecting tasks where a deterministic function maps intermediate structures to decisions, every controlled edit implies a unique correct output. Across eight models and three benchmarks, models appear self-consistent with their own intermediate structures but fail to update predictions after intervention in up to 60% of cases -- revealing that apparent faithfulness is fragile once the intermediate structure changes. When derivation of the final decision from the structure is delegated to an external tool, this fragility largely disappears; however, prompts which ask to prioritize the intermediate structure over the original input do not materially close the gap. Overall, intermediate structures in schema-guided pipelines function as influential context rather than stable causal mediators. 

---
# Follow the Clues, Frame the Truth: Hybrid-evidential Deductive Reasoning in Open-Vocabulary Multimodal Emotion Recognition 

**Authors**: Yu Liu, Lei Zhang, Haoxun Li, Hanlei Shi, Yuxuan Ding, Leyuan Qu, Taihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16463)  

**Abstract**: Open-Vocabulary Multimodal Emotion Recognition (OV-MER) is inherently challenging due to the ambiguity of equivocal multimodal cues, which often stem from distinct unobserved situational dynamics. While Multimodal Large Language Models (MLLMs) offer extensive semantic coverage, their performance is often bottlenecked by premature commitment to dominant data priors, resulting in suboptimal heuristics that overlook crucial, complementary affective cues across modalities. We argue that effective affective reasoning requires more than surface-level association; it necessitates reconstructing nuanced emotional states by synthesizing multiple evidence-grounded rationales that reconcile these observations from diverse latent perspectives. We introduce HyDRA, a Hybrid-evidential Deductive Reasoning Architecture that formalizes inference as a Propose-Verify-Decide protocol. To internalize this abductive process, we employ reinforcement learning with hierarchical reward shaping, aligning the reasoning trajectories with final task performance to ensure they best reconcile the observed multimodal cues. Systematic evaluations validate our design choices, with HyDRA consistently outperforming strong baselines--especially in ambiguous or conflicting scenarios--while providing interpretable, diagnostic evidence traces. 

---
# RetailBench: Evaluating Long-Horizon Autonomous Decision-Making and Strategy Stability of LLM Agents in Realistic Retail Environments 

**Authors**: Linghua Zhang, Jun Wang, Jingtong Wu, Zhisong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16453)  

**Abstract**: Large Language Model (LLM)-based agents have achieved notable success on short-horizon and highly structured tasks. However, their ability to maintain coherent decision-making over long horizons in realistic and dynamic environments remains an open challenge.
We introduce RetailBench, a high-fidelity benchmark designed to evaluate long-horizon autonomous decision-making in realistic commercial scenarios, where agents must operate under stochastic demand and evolving external conditions.
We further propose the Evolving Strategy & Execution framework, which separates high-level strategic reasoning from low-level action execution. This design enables adaptive and interpretable strategy evolution over time. It is particularly important for long-horizon tasks, where non-stationary environments and error accumulation require strategies to be revised at a different temporal scale than action execution.
Experiments on eight state-of-the-art LLMs across progressively challenging environments show that our framework improves operational stability and efficiency compared to other baselines. However, performance degrades substantially as task complexity increases, revealing fundamental limitations in current LLMs for long-horizon, multi-factor decision-making. 

---
# TRUST-SQL: Tool-Integrated Multi-Turn Reinforcement Learning for Text-to-SQL over Unknown Schemas 

**Authors**: Ai Jian, Xiaoyun Zhang, Wanrou Du, Jingqing Ruan, Jiangbo Pei, Weipeng Zhang, Ke Zeng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2603.16448)  

**Abstract**: Text-to-SQL parsing has achieved remarkable progress under the Full Schema Assumption. However, this premise fails in real-world enterprise environments where databases contain hundreds of tables with massive noisy metadata. Rather than injecting the full schema upfront, an agent must actively identify and verify only the relevant subset, giving rise to the Unknown Schema scenario we study in this work. To address this, we propose TRUST-SQL (Truthful Reasoning with Unknown Schema via Tools). We formulate the task as a Partially Observable Markov Decision Process where our autonomous agent employs a structured four-phase protocol to ground reasoning in verified metadata. Crucially, this protocol provides a structural boundary for our novel Dual-Track GRPO strategy. By applying token-level masked advantages, this strategy isolates exploration rewards from execution outcomes to resolve credit assignment, yielding a 9.9% relative improvement over standard GRPO. Extensive experiments across five benchmarks demonstrate that TRUST-SQL achieves an average absolute improvement of 30.6% and 16.6% for the 4B and 8B variants respectively over their base models. Remarkably, despite operating entirely without pre-loaded metadata, our framework consistently matches or surpasses strong baselines that rely on schema prefilling. 

---
# Visual Distraction Undermines Moral Reasoning in Vision-Language Models 

**Authors**: Xinyi Yang, Chenheng Xu, Weijun Hong, Ce Mo, Qian Wang, Fang Fang, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16445)  

**Abstract**: Moral reasoning is fundamental to safe Artificial Intelligence (AI), yet ensuring its consistency across modalities becomes critical as AI systems evolve from text-based assistants to embodied agents. Current safety techniques demonstrate success in textual contexts, but concerns remain about generalization to visual inputs. Existing moral evaluation benchmarks rely on textonly formats and lack systematic control over variables that influence moral decision-making. Here we show that visual inputs fundamentally alter moral decision-making in state-of-the-art (SOTA) Vision-Language Models (VLMs), bypassing text-based safety mechanisms. We introduce Moral Dilemma Simulation (MDS), a multimodal benchmark grounded in Moral Foundation Theory (MFT) that enables mechanistic analysis through orthogonal manipulation of visual and contextual variables. The evaluation reveals that the vision modality activates intuition-like pathways that override the more deliberate and safer reasoning patterns observed in text-only contexts. These findings expose critical fragilities where language-tuned safety filters fail to constrain visual processing, demonstrating the urgent need for multimodal safety alignment. 

---
# From Natural Language to Executable Option Strategies via Large Language Models 

**Authors**: Haochen Luo, Zhengzhao Lai, Junjie Xu, Yifan Li, Tang Pok Hin, Yuan Zhang, Chen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16434)  

**Abstract**: Large Language Models (LLMs) excel at general code generation, yet translating natural-language trading intents into correct option strategies remains challenging. Real-world option design requires reasoning over massive, multi-dimensional option chain data with strict constraints, which often overwhelms direct generation methods. We introduce the Option Query Language (OQL), a domain-specific intermediate representation that abstracts option markets into high-level primitives under grammatical rules, enabling LLMs to function as reliable semantic parsers rather than free-form programmers. OQL queries are then validated and executed deterministically by an engine to instantiate executable strategies. We also present a new dataset for this task and demonstrate that our neuro-symbolic pipeline significantly improves execution accuracy and logical consistency over direct baselines. 

---
# Via Negativa for AI Alignment: Why Negative Constraints Are Structurally Superior to Positive Preferences 

**Authors**: Quan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.16417)  

**Abstract**: Recent empirical results have demonstrated that training large language models (LLMs) with negative-only feedback can match or exceed standard reinforcement learning from human feedback (RLHF). Negative Sample Reinforcement achieves parity with PPO on mathematical reasoning; Distributional Dispreference Optimization trains effectively using only dispreferred samples; and Constitutional AI outperforms pure RLHF on harmlessness benchmarks. Yet no unified theoretical account explains why negative signals are so effective. This paper proposes such an account: positive preferences and negative constraints are structurally asymmetric. Positive preferences ("which is better") encode continuously coupled, context-dependent human values that cannot be exhaustively specified -- leading models to learn surface correlates such as agreement with the user (sycophancy). Negative constraints ("what is wrong") encode discrete, finite, independently verifiable prohibitions that can converge to a stable boundary. This asymmetry -- rooted in Popper's falsification logic and the epistemology of negative knowledge -- explains both the sycophancy failure of preference-based RLHF and the surprising effectiveness of negative-signal methods. We argue that alignment research should shift its center of gravity from "learning what humans prefer" to "learning what humans reject," and offer testable predictions for this framework. 

---
# FactorEngine: A Program-level Knowledge-Infused Factor Mining Framework for Quantitative Investment 

**Authors**: Qinhong Lin, Ruitao Feng, Yinglun Feng, Zhenxin Huang, Yukun Chen, Zhongliang Yang, Linna Zhou, Binjie Fei, Jiaqi Liu, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16365)  

**Abstract**: We study alpha factor mining, the automated discovery of predictive signals from noisy, non-stationary market data-under a practical requirement that mined factors be directly executable and auditable, and that the discovery process remain computationally tractable at scale. Existing symbolic approaches are limited by bounded expressiveness, while neural forecasters often trade interpretability for performance and remain vulnerable to regime shifts and overfitting. We introduce FactorEngine (FE), a program-level factor discovery framework that casts factors as Turing-complete code and improves both effectiveness and efficiency via three separations: (i) logic revision vs. parameter optimization, (ii) LLM-guided directional search vs. Bayesian hyperparameter search, and (iii) LLM usage vs. local computation. FE further incorporates a knowledge-infused bootstrapping module that transforms unstructured financial reports into executable factor programs through a closed-loop multi-agent extraction-verification-code-generation pipeline, and an experience knowledge base that supports trajectory-aware refinement (including learning from failures). Across extensive backtests on real-world OHLCV data, FE produces factors with substantially stronger predictive stability and portfolio impact-for example, higher IC/ICIR (and Rank IC/ICIR) and improved AR/Sharpe, than baseline methods, achieving state-of-the-art predictive and portfolio performance. 

---
# Learning to Predict, Discover, and Reason in High-Dimensional Discrete Event Sequences 

**Authors**: Hugo Math  

**Link**: [PDF](https://arxiv.org/pdf/2603.16313)  

**Abstract**: Electronic control units (ECUs) embedded within modern vehicles generate a large number of asynchronous events known as diagnostic trouble codes (DTCs). These discrete events form complex temporal sequences that reflect the evolving health of the vehicle's subsystems. In the automotive industry, domain experts manually group these codes into higher-level error patterns (EPs) using Boolean rules to characterize system faults and ensure safety. However, as vehicle complexity grows, this manual process becomes increasingly costly, error-prone, and difficult to scale. Notably, the number of unique DTCs in a modern vehicle is on the same order of magnitude as the vocabulary of a natural language, often numbering in the tens of thousands. This observation motivates a paradigm shift: treating diagnostic sequences as a language that can be modeled, predicted, and ultimately explained. Traditional statistical approaches fail to capture the rich dependencies and do not scale to high-dimensional datasets characterized by thousands of nodes, large sample sizes, and long sequence lengths. Specifically, the high cardinality of categorical event spaces in industrial logs poses a significant challenge, necessitating new machine learning architectures tailored to such event-driven systems. This thesis addresses automated fault diagnostics by unifying event sequence modeling, causal discovery, and large language models (LLMs) into a coherent framework for high-dimensional event streams. It is structured in three parts, reflecting a progressive transition from prediction to causal understanding and finally to reasoning for vehicle diagnostics. Consequently, we introduce several Transformer-based architectures for predictive maintenance, scalable sample- and population-level causal discovery frameworks and a multi-agent system that automates the synthesis of Boolean EP rules. 

---
# NeSy-Route: A Neuro-Symbolic Benchmark for Constrained Route Planning in Remote Sensing 

**Authors**: Ming Yang, Zhi Zhou, Shi-Yu Tian, Kun-Yang Yu, Lan-Zhe Guo, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16307)  

**Abstract**: Remote sensing underpins crucial applications such as disaster relief and ecological field surveys, where systems must understand complex scenes and constraints and make reliable decisions. Current remote-sensing benchmarks mainly focus on evaluating perception and reasoning capabilities of multimodal large language models (MLLMs). They fail to assess planning capability, stemming either from the difficulty of curating and validating planning tasks at scale or from evaluation protocols that are inaccurate and inadequate. To address these limitations, we introduce NeSy-Route, a large-scale neuro-symbolic benchmark for constrained route planning in remote sensing. Within this benchmark, we introduce an automated data-generation framework that integrates high-fidelity semantic masks with heuristic search to produce diverse route-planning tasks with provably optimal solutions. This allows NeSy-Route to comprehensively evaluate planning across 10,821 route-planning samples, nearly 10 times larger than the largest prior benchmark. Furthermore, a three-level hierarchical neuro-symbolic evaluation protocol is developed to enable accurate assessment and support fine-grained analysis on perception, reasoning, and planning simultaneously. Our comprehensive evaluation of various state-of-the-art MLLMs demonstrates that existing MLLMs show significant deficiencies in perception and planning capabilities. We hope NeSy-Route can support further research and development of more powerful MLLMs for remote sensing. 

---
# Adaptive Theory of Mind for LLM-based Multi-Agent Coordination 

**Authors**: Chunjiang Mu, Ya Zeng, Qiaosheng Zhang, Kun Shao, Chen Chu, Hao Guo, Danyang Jia, Zhen Wang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16264)  

**Abstract**: Theory of Mind (ToM) refers to the ability to reason about others' mental states, and higher-order ToM involves considering that others also possess their own ToM. Equipping large language model (LLM)-driven agents with ToM has long been considered to improve their coordination in multiagent collaborative tasks. However, we find that misaligned ToM orders-mismatches in the depth of ToM reasoning between agents-can lead to insufficient or excessive reasoning about others, thereby impairing their coordination. To address this issue, we design an adaptive ToM (A-ToM) agent, which can align in ToM orders with its partner. Based on prior interactions, the agent estimates the partner's likely ToM order and leverages this estimation to predict the partner's action, thereby facilitating behavioral coordination. We conduct empirical evaluations on four multi-agent coordination tasks: a repeated matrix game, two grid navigation tasks and an Overcooked task. The results validate our findings on ToM alignment and demonstrate the effectiveness of our A-ToM agent. Furthermore, we discuss the generalizability of our A-ToM to non-LLM-based agents, as well as what would diminish the importance of ToM alignment. 

---
# MOSAIC: Composable Safety Alignment with Modular Control Tokens 

**Authors**: Jingyu Peng, Hongyu Chen, Jiancheng Dong, Maolin Wang, Wenxi Li, Yuchen Li, Kai Zhang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16210)  

**Abstract**: Safety alignment in large language models (LLMs) is commonly implemented as a single static policy embedded in model parameters. However, real-world deployments often require context-dependent safety rules that vary across users, regions, and applications. Existing approaches struggle to provide such conditional control: parameter-level alignment entangles safety behaviors with general capabilities, while prompt-based methods rely on natural language instructions that provide weak enforcement. We propose MOSAIC, a modular framework that enables compositional safety alignment through learnable control tokens optimized over a frozen backbone model. Each token represents a safety constraint and can be flexibly activated and composed at inference time. To train compositional tokens efficiently, we introduce order-based task sampling and a distribution-level alignment objective that mitigates over-refusal. Experiments show that MOSAIC achieves strong defense performance with substantially lower over-refusal while preserving model utility. 

---
# Proactive Rejection and Grounded Execution: A Dual-Stage Intent Analysis Paradigm for Safe and Efficient AIoT Smart Homes 

**Authors**: Xinxin Jin, Zhengwei Ni, Zhengguo Sheng, Victor C. M. Leung  

**Link**: [PDF](https://arxiv.org/pdf/2603.16207)  

**Abstract**: As Large Language Models (LLMs) transition from information providers to embodied agents in the Internet of Things (IoT), they face significant challenges regarding reliability and interaction efficiency. Direct execution of LLM-generated commands often leads to entity hallucinations (e.g., trying to control non-existent devices). Meanwhile, existing iterative frameworks (e.g., SAGE) suffer from the Interaction Frequency Dilemma, oscillating between reckless execution and excessive user questioning. To address these issues, we propose a Dual-Stage Intent-Aware (DS-IA) Framework. This framework separates high-level user intent understanding from low-level physical execution. Specifically, Stage 1 serves as a semantic firewall to filter out invalid instructions and resolve vague commands by checking the current state of the home. Stage 2 then employs a deterministic cascade verifier-a strict, step-by-step rule checker that verifies the room, device, and capability in sequence-to ensure the action is actually physically possible before execution. Extensive experiments on the HomeBench and SAGE benchmarks demonstrate that DS-IA achieves an Exact Match (EM) rate of 58.56% (outperforming baselines by over 28%) and improves the rejection rate of invalid instructions to 87.04%. Evaluations on the SAGE benchmark further reveal that DS-IA resolves the Interaction Frequency Dilemma by balancing proactive querying with state-based inference. Specifically, it boosts the Autonomous Success Rate (resolving tasks without unnecessary user intervention) from 42.86% to 71.43%, while maintaining high precision in identifying irreducible ambiguities that truly necessitate human clarification. These results underscore the framework's ability to minimize user disturbance through accurate environmental grounding. 

---
# Are Large Language Models Truly Smarter Than Humans? 

**Authors**: Eshwar Reddy M, Sourav Karmakar  

**Link**: [PDF](https://arxiv.org/pdf/2603.16197)  

**Abstract**: Public leaderboards increasingly suggest that large language models (LLMs) surpass human experts on benchmarks spanning academic knowledge, law, and programming. Yet most benchmarks are fully public, their questions widely mirrored across the internet, creating systematic risk that models were trained on the very data used to evaluate them. This paper presents three complementary experiments forming a rigorous multi-method contamination audit of six frontier LLMs: GPT-4o, GPT-4o-mini, DeepSeek-R1, DeepSeek-V3, Llama-3.3-70B, and Qwen3-235B. Experiment 1 applies a lexical contamination detection pipeline to 513 MMLU questions across all 57 subjects, finding an overall contamination rate of 13.8% (18.1% in STEM, up to 66.7% in Philosophy) and estimated performance gains of +0.030 to +0.054 accuracy points by category. Experiment 2 applies a paraphrase and indirect-reference diagnostic to 100 MMLU questions, finding accuracy drops by an average of 7.0 percentage points under indirect reference, rising to 19.8 pp in both Law and Ethics. Experiment 3 applies TS-Guessing behavioral probes to all 513 questions and all six models, finding that 72.5% trigger memorization signals far above chance, with DeepSeek-R1 displaying a distributed memorization signature (76.6% partial reconstruction, 0% verbatim recall) that explains its anomalous Experiment 2 profile. All three experiments converge on the same contamination ranking: STEM > Professional > Social Sciences > Humanities. 

---
# SQL-ASTRA: Alleviating Sparse Feedback in Agentic SQL via Column-Set Matching and Trajectory Aggregation 

**Authors**: Long Li, Zhijian Zhou, Jiangxuan Long, Peiyang Liu, Weidi Xu, Zhe Wang, Shirui Pan, Chao Qu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16161)  

**Abstract**: Agentic Reinforcement Learning (RL) shows promise for complex tasks, but Text-to-SQL remains mostly restricted to single-turn paradigms. A primary bottleneck is the credit assignment problem. In traditional paradigms, rewards are determined solely by the final-turn feedback, which ignores the intermediate process and leads to ambiguous credit evaluation. To address this, we propose Agentic SQL, a framework featuring a universal two-tiered reward mechanism designed to provide effective trajectory-level evaluation and dense step-level signals. First, we introduce Aggregated Trajectory Reward (ATR) to resolve multi-turn credit assignment. Using an asymmetric transition matrix, ATR aggregates process-oriented scores to incentivize continuous improvement. Leveraging Lyapunov stability theory, we prove ATR acts as an energy dissipation operator, guaranteeing a cycle-free policy and monotonic convergence. Second, Column-Set Matching Reward (CSMR) provides immediate step-level rewards to mitigate sparsity. By executing queries at each turn, CSMR converts binary (0/1) feedback into dense [0, 1] signals based on partial correctness. Evaluations on BIRD show a 5% gain over binary-reward GRPO. Notably, our approach outperforms SOTA Arctic-Text2SQL-R1-7B on BIRD and Spider 2.0 using identical models, propelling Text-to-SQL toward a robust multi-turn agent paradigm. 

---
# NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics 

**Authors**: Zhengzheng Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16148)  

**Abstract**: We ask whether a pure spiking backbone can learn large-scale language modeling from random initialization, without Transformer distillation. We introduce NeuronSpark, a 0.9B-parameter SNN language model trained with next-token prediction and surrogate gradients. The model combines selective state-space spiking dynamics, leakage-current inter-layer communication, PonderNet adaptive timesteps, fused Triton PLIF kernels, and stabilization techniques (residual centering, lateral-inhibition normalization, and natural-gradient compensation). Under a constrained budget (about 1.4B pretraining tokens and 6.5K SFT steps), NeuronSpark-0.9B reaches 3.6 pretraining loss and shows early multi-turn dialogue behavior after SFT. These results support the feasibility of end-to-end language modeling with a pure SNN architecture at this scale. 

---
# VIGIL: Towards Edge-Extended Agentic AI for Enterprise IT Support 

**Authors**: Sarthak Ahuja, Neda Kordjazi, Evren Yortucboylu, Vishaal Kapoor, Mariam Dundua, Yiming Li, Derek Ho, Vaibhavi Padala, Jennifer Whitted, Rebecca Steinert  

**Link**: [PDF](https://arxiv.org/pdf/2603.16110)  

**Abstract**: Enterprise IT support is constrained by heterogeneous devices, evolving policies, and long-tail failure modes that are difficult to resolve centrally. We present VIGIL, an edge-extended agentic AI system that deploys desktop-resident agents to perform situated diagnosis, retrieval over enterprise knowledge, and policy-governed remediation directly on user devices with explicit consent and end-to-end observability. In a 10-week pilot of VIGIL's operational loop on 100 resource-constrained endpoints, VIGIL reduces interaction rounds by 39%, achieves at least 4 times faster diagnosis, and supports self-service resolution in 82% of matched cases. Users report excellent usability, high trust, and low cognitive workload across four validated instruments, with qualitative feedback highlighting transparency as critical for trust. Notably, users rated the system higher when no historical matches were available, suggesting on-device diagnosis provides value independent of knowledge base coverage. This pilot establishes safety and observability foundations for fleet-wide continuous improvement. 

---
# ARISE: Agent Reasoning with Intrinsic Skill Evolution in Hierarchical Reinforcement Learning 

**Authors**: Yu Li, Rui Miao, Zhengling Qi, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2603.16060)  

**Abstract**: The dominant paradigm for improving mathematical reasoning in language models relies on Reinforcement Learning with verifiable rewards. Yet existing methods treat each problem instance in isolation without leveraging the reusable strategies that emerge and accumulate during training. To this end, we introduce ARISE (Agent Reasoning via Intrinsic Skill Evolution), a hierarchical reinforcement learning framework, in which a shared policy operates both to manage skills at high-level and to generate responses at low-level (denoted as a Skills Manager and a Worker, respectively). The Manager maintains a tiered skill library through a dedicated skill generation rollout that performs structured summarization of successful solution traces (after execution), while employing a policy-driven selection mechanism to retrieve relevant skills to condition future rollouts (before execution). A hierarchical reward design guides the co-evolution of reasoning ability and library quality. Experiments on two base models and seven benchmarks spanning both competition mathematics and Omni-MATH show that ARISE consistently outperforms GRPO-family algorithms and memory-augmented baselines, with particularly notable gains on out-of-distribution tasks. Ablation studies confirm that each component contributes to the observed improvements and that library quality and reasoning performance improve in tandem throughout training. Code is available at \href{this https URL}{this https URL}. 

---
# A Context Alignment Pre-processor for Enhancing the Coherence of Human-LLM Dialog 

**Authors**: Ding Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.16052)  

**Abstract**: Large language models (LLMs) have made remarkable progress in generating fluent text, but they still face a critical challenge of contextual misalignment in long-term and dynamic dialogue. When human users omit premises, simplify references, or shift context abruptly during interactions with LLMs, the models may fail to capture their actual intentions, producing mechanical or off-topic responses that weaken the collaborative potential of dialogue. To address this problem, this paper proposes a computational framework called the Context Alignment Pre-processor (C.A.P.). Rather than operating during generation, C.A.P. functions as a pre-processing module between user input and response generation. The framework includes three core processes: (1) semantic expansion, which extends a user instruction to a broader semantic span including its premises, literal meaning, and implications; (2) time-weighted context retrieval, which prioritizes recent dialogue history through a temporal decay function approximating human conversational focus; and (3) alignment verification and decision branching, which evaluates whether the dialogue remains on track by measuring the semantic similarity between the current prompt and the weighted historical context. When a significant deviation is detected, C.A.P. initiates a structured clarification protocol to help users and the system recalibrate the conversation. This study presents the architecture and theoretical basis of C.A.P., drawing on cognitive science and Common Ground theory in human-computer interaction. We argue that C.A.P. is not only a technical refinement but also a step toward shifting human-computer dialogue from one-way command-execution patterns to two-way, self-correcting, partnership-based collaboration. Finally, we discuss implementation paths, evaluation methods, and implications for the future design of interactive intelligent systems. 

---
# POaaS: Minimal-Edit Prompt Optimization as a Service to Lift Accuracy and Cut Hallucinations on On-Device sLLMs 

**Authors**: Jungwoo Shim, Dae Won Kim, Sun Wook Kim, Soo Young Kim, Myungcheol Lee, Jae-geun Cha, Hyunhwa Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16045)  

**Abstract**: Small language models (sLLMs) are increasingly deployed on-device, where imperfect user prompts--typos, unclear intent, or missing context--can trigger factual errors and hallucinations. Existing automatic prompt optimization (APO) methods were designed for large cloud LLMs and rely on search that often produces long, structured instructions; when executed under an on-device constraint where the same small model must act as optimizer and solver, these pipelines can waste context and even hurt accuracy. We propose POaaS, a minimal-edit prompt optimization layer that routes each query to lightweight specialists (Cleaner, Paraphraser, Fact-Adder) and merges their outputs under strict drift and length constraints, with a conservative skip policy for well-formed prompts. Under a strict fixed-model setting with Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct, POaaS improves both task accuracy and factuality while representative APO baselines degrade them, and POaaS recovers up to +7.4% under token deletion and mixup. Overall, per-query conservative optimization is a practical alternative to search-heavy APO for on-device sLLMs. 

---
# Enhancing Linguistic Generalization of VLA: Fine-Tuning OpenVLA via Synthetic Instruction Augmentation 

**Authors**: Dongik Shin  

**Link**: [PDF](https://arxiv.org/pdf/2603.16044)  

**Abstract**: Generalization remains a core challenge in embodied AI, as robots must adapt to diverse environments. While OpenVLA represents the State-of-the-Art (SOTA) in Vision-Language-Action models by leveraging large-scale pre-training, its zero-shot performance can be limited when encountering completely new environments. This paper proposes a parameter-efficient fine-tuning strategy to enhance the linguistic generalization of OpenVLA by synthesizing a general instruction set for the Bridge Dataset V2. The paper leverages a Large Language Model (LLM) to generate a rich variety of semantically equivalent but structurally diverse commands for existing trajectories. In this experiment, Low-Rank Adaptation (LoRA) is implemented to fine-tune OpenVLA on augmented pairs, allowing the model to bridge the gap between complex natural language intent and robotic actions. Results demonstrate that the LoRA-enhanced model's robustness, suggesting that enriching the linguistic space of specialized datasets is crucial for embodied agents. 

---
# Interpretable Context Methodology: Folder Structure as Agentic Architecture 

**Authors**: Jake Van Clief, David McDermott  

**Link**: [PDF](https://arxiv.org/pdf/2603.16021)  

**Abstract**: Current approaches to AI agent orchestration typically involve building multi-agent frameworks that manage context passing, memory, error handling, and step coordination through code. These frameworks work well for complex, concurrent systems. But for sequential workflows where a human reviews output at each step, they introduce engineering overhead that the problem does not require. This paper presents Model Workspace Protocol (MWP), a method that replaces framework-level orchestration with filesystem structure. Numbered folders represent stages. Plain markdown files carry the prompts and context that tell a single AI agent what role to play at each step. Local scripts handle the mechanical work that does not need AI at all. The result is a system where one agent, reading the right files at the right moment, does the work that would otherwise require a multi-agent framework. This approach applies ideas from Unix pipeline design, modular decomposition, multi-pass compilation, and literate programming to the specific problem of structuring context for AI agents. The protocol is open source under the MIT license. 

---
# IRAM-Omega-Q: A Computational Architecture for Uncertainty Regulation in Artificial Agents 

**Authors**: Veronique Ziegler  

**Link**: [PDF](https://arxiv.org/pdf/2603.16020)  

**Abstract**: Artificial agents can achieve strong task performance while remaining opaque with respect to internal regulation, uncertainty management, and stability under stochastic perturbation. We present IRAM-Omega-Q, a computational architecture that models internal regulation as closed-loop control over a quantum-like state representation. The framework uses density matrices instrumentally as abstract state descriptors, enabling direct computation of entropy, purity, and coherence-related metrics without invoking physical quantum processes. A central adaptive gain is updated continuously to maintain a target uncertainty regime under noise. Using systematic parameter sweeps, fixed-seed publication-mode simulations, and susceptibility-based phase-diagram analysis, we identify reproducible critical boundaries in regulation-noise space. We further show that alternative control update orderings, interpreted as perception-first and action-first architectures, induce distinct stability regimes under identical external conditions. These results support uncertainty regulation as a concrete architectural principle for artificial agents and provide a formal setting for studying stability, control, and order effects in cognitively inspired AI systems. The framework is presented as a technical model of adaptive regulation dynamics in artificial agents. It makes no claims regarding phenomenological consciousness, and the quantum-like formalism is used strictly as a mathematical representation for structured uncertainty and state evolution. 

---
# Selective Memory for Artificial Intelligence: Write-Time Gating with Hierarchical Archiving 

**Authors**: Oliver Zahn, Simran Chana  

**Link**: [PDF](https://arxiv.org/pdf/2603.15994)  

**Abstract**: Retrieval-augmented generation stores all content indiscriminately, degrading accuracy as noise accumulates. Parametric approaches compress knowledge into weights, precluding selective updates. Neither mirrors biological memory, which gates encoding based on salience and archives rather than deletes superseded information. We introduce write-time gating that filters incoming knowledge objects using composite salience scores (source reputation, novelty, reliability) while maintaining version chains that preserve prior states. Using real LLM evaluation without oracle access to quality labels, write gating achieves 100 percent accuracy versus 13 percent for ungated stores. The critical finding emerges under distractor scaling: at 8:1 distractor ratios, read-time filtering (Self-RAG) collapses to 0 percent while write gating maintains 100 percent, revealing a structural advantage of write-time over read-time curation. Validation on Wikipedia (20 entities), procedurally generated pharmacology data, and 2026 arXiv papers confirms these findings. The gating advantage scales inversely with parametric memory support: +25pp for Wikipedia, +48pp for post-cutoff arXiv, +65pp for procedural data with zero training knowledge. Signal ablation confirms the method does not depend on oracle-correlated metadata. Write gating matches Self-RAG accuracy at one-ninth the query-time cost. 

---
# From Workflow Automation to Capability Closure: A Formal Framework for Safe and Revenue-Aware Customer Service AI 

**Authors**: Cosimo Spera, Garima Agrawal, Riccardo De Maria  

**Link**: [PDF](https://arxiv.org/pdf/2603.15978)  

**Abstract**: Customer service automation is undergoing a structural transformation. The dominant paradigm is shifting from scripted chatbots and single-agent responders toward networks of specialised AI agents that compose capabilities dynamically across billing, service provision, payments, and fulfilment. This shift introduces a safety gap that no current platform has closed: two agents individually verified as safe can, when combined, reach a forbidden goal through an emergent conjunctive dependency that neither possesses alone. 

---
# An Agentic Evaluation Framework for AI-Generated Scientific Code in PETSc 

**Authors**: Hong Zhang, Barry Smith, Satish Balay, Le Chen, Murat Keceli, Lois Curfman McInnes, Junchao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15976)  

**Abstract**: While large language models have significantly accelerated scientific code generation, comprehensively evaluating the generated code remains a major challenge. Traditional benchmarks reduce evaluation to test-case matching, an approach insufficient for library code in HPC where solver selection, API conventions, memory management, and performance are just as critical as functional correctness. To address this gap, we introduce petscagent-bench, an agentic framework built on an agents-evaluating-agents paradigm. Instead of relying on static scripts, petscagent-bench deploys a tool-augmented evaluator agent that compiles, executes, and measures code produced by a separate model-under-test agent, orchestrating a 14-evaluator pipeline across five scoring categories: correctness, performance, code quality, algorithmic appropriateness, and library-specific conventions. Because the agents communicate through standardized protocols (A2A and MCP), the framework enables black-box evaluation of any coding agent without requiring access to its source code. We demonstrate the framework on a benchmark suite of realistic problems using the PETSc library for HPC. Our empirical analysis of frontier models reveals that while current models generate readable, well-structured code, they consistently struggle with library-specific conventions that traditional pass/fail metrics completely miss. 

---
# Safety is Non-Compositional: A Formal Framework for Capability-Based AI Systems 

**Authors**: Cosimo Spera  

**Link**: [PDF](https://arxiv.org/pdf/2603.15973)  

**Abstract**: This paper contains the first formal proof that safety is non-compositional in the presence of conjunctive capability dependencies: two agents each individually inca- pable of reaching any forbidden capability can, when combined, collectively reach a forbidden goal through an emergent conjunctive dependency. 

---
# MAC: Multi-Agent Constitution Learning 

**Authors**: Rushil Thareja, Gautam Gupta, Francesco Pinto, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2603.15968)  

**Abstract**: Constitutional AI is a method to oversee and control LLMs based on a set of rules written in natural language. These rules are typically written by human experts, but could in principle be learned automatically given sufficient training data for the desired behavior. Existing LLM-based prompt optimizers attempt this but are ineffective at learning constitutions since (i) they require many labeled examples and (ii) lack structure in the optimized prompts, leading to diminishing improvements as prompt size grows. To address these limitations, we propose Multi-Agent Constitutional Learning (MAC), which optimizes over structured prompts represented as sets of rules using a network of agents with specialized tasks to accept, edit, or reject rule updates. We also present MAC+, which improves performance by training agents on successful trajectories to reinforce updates leading to higher reward. We evaluate MAC on tagging Personally Identifiable Information (PII), a classification task with limited labels where interpretability is critical, and demonstrate that it generalizes to other agentic tasks such as tool calling. MAC outperforms recent prompt optimization methods by over 50%, produces human-readable and auditable rule sets, and achieves performance comparable to supervised fine-tuning and GRPO without requiring parameter updates. 

---
# Optimizing Hospital Capacity During Pandemics: A Dual-Component Framework for Strategic Patient Relocation 

**Authors**: Sadaf Tabatabaee, Hicham El Baz, Mohammed Khalil Ghali, Nagendra N. Nagarur  

**Link**: [PDF](https://arxiv.org/pdf/2603.15960)  

**Abstract**: The COVID-19 pandemic has placed immense strain on hospital systems worldwide, leading to critical capacity challenges. This research proposes a two-part framework to optimize hospital capacity through patient relocation strategies. The first component involves developing a time series prediction model to forecast patient arrival rates. Using historical data on COVID-19 cases and hospitalizations, the model will generate accurate forecasts of future patient volumes. This will enable hospitals to proactively plan resource allocation and patient flow. The second com- ponent is a simulation model that evaluates the impact of different patient relocation strategies. The simulation will account for factors such as bed availability, staff capabilities, transportation logistics, and patient acuity to optimize the placement of patients across networked hospitals. Multiple scenarios will be tested, including inter-hospital trans- fers, use of temporary care facilities, and adaptations to discharge protocols. By combining predictive analytics and simulation modeling, this research aims to provide hospital administrators with a comprehensive decision-support tool. The proposed framework will empower them to anticipate demand, simulate relocation strategies, and imple- ment optimal policies to distribute patients and resources. Ultimately, this work seeks to enhance the resilience of healthcare systems in the face of COVID-19 and future pandemics. 

---
# Protein Design with Agent Rosetta: A Case Study for Specialized Scientific Agents 

**Authors**: Jacopo Teneggi, S.M. Bargeen A. Turzo, Tanya Marwah, Alberto Bietti, P. Douglas Renfrew, Vikram Khipple Mulligan, Siavash Golkar  

**Link**: [PDF](https://arxiv.org/pdf/2603.15952)  

**Abstract**: Large language models (LLMs) are capable of emulating reasoning and using tools, creating opportunities for autonomous agents that execute complex scientific tasks. Protein design provides a natural testbed: although machine learning (ML) methods achieve strong results, these are largely restricted to canonical amino acids and narrow objectives, leaving unfilled need for a generalist tool for broad design pipelines. We introduce Agent Rosetta, an LLM agent paired with a structured environment for operating Rosetta, the leading physics-based heteropolymer design software, capable of modeling non-canonical building blocks and geometries. Agent Rosetta iteratively refines designs to achieve user-defined objectives, combining LLM reasoning with Rosetta's generality. We evaluate Agent Rosetta on design with canonical amino acids, matching specialized models and expert baselines, and with non-canonical residues -- where ML approaches fail -- achieving comparable performance. Critically, prompt engineering alone often fails to generate Rosetta actions, demonstrating that environment design is essential for integrating LLM agents with specialized software. Our results show that properly designed environments enable LLM agents to make scientific software accessible while matching specialized tools and human experts. 

---
# Argumentative Human-AI Decision-Making: Toward AI Agents That Reason With Us, Not For Us 

**Authors**: Stylianos Loukas Vasileiou, Antonio Rago, Francesca Toni, William Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2603.15946)  

**Abstract**: Computational argumentation offers formal frameworks for transparent, verifiable reasoning but has traditionally been limited by its reliance on domain-specific information and extensive feature engineering. In contrast, LLMs excel at processing unstructured text, yet their opaque nature makes their reasoning difficult to evaluate and trust. We argue that the convergence of these fields will lay the foundation for a new paradigm: Argumentative Human-AI Decision-Making. We analyze how the synergy of argumentation framework mining, argumentation framework synthesis, and argumentative reasoning enables agents that do not just justify decisions, but engage in dialectical processes where decisions are contestable and revisable -- reasoning with humans rather than for them. This convergence of computational argumentation and LLMs is essential for human-aware, trustworthy AI in high-stakes domains. 

---
# Semi-Autonomous Formalization of the Vlasov-Maxwell-Landau Equilibrium 

**Authors**: Vasily Ilin  

**Link**: [PDF](https://arxiv.org/pdf/2603.15929)  

**Abstract**: We present a complete Lean 4 formalization of the equilibrium characterization in the Vlasov-Maxwell-Landau (VML) system, which describes the motion of charged plasma. The project demonstrates the full AI-assisted mathematical research loop: an AI reasoning model (Gemini DeepThink) generated the proof from a conjecture, an agentic coding tool (Claude Code) translated it into Lean from natural-language prompts, a specialized prover (Aristotle) closed 111 lemmas, and the Lean kernel verified the result. A single mathematician supervised the process over 10 days at a cost of \$200, writing zero lines of code.
The entire development process is public: all 229 human prompts, and 213 git commits are archived in the repository. We report detailed lessons on AI failure modes -- hypothesis creep, definition-alignment bugs, agent avoidance behaviors -- and on what worked: the abstract/concrete proof split, adversarial self-review, and the critical role of human review of key definitions and theorem statements. Notably, the formalization was completed before the final draft of the corresponding math paper was finished. 

---
# Prompt Engineering for Scale Development in Generative Psychometrics 

**Authors**: Lara Lee Russell-Lasalandra, Hudson Golino  

**Link**: [PDF](https://arxiv.org/pdf/2603.15909)  

**Abstract**: This Monte Carlo simulation examines how prompt engineering strategies shape the quality of large language model (LLM)--generated personality assessment items within the AI-GENIE framework for generative psychometrics. Item pools targeting the Big Five traits were generated using multiple prompting designs (zero-shot, few-shot, persona-based, and adaptive), model temperatures, and LLMs, then evaluated and reduced using network psychometric methods. Across all conditions, AI-GENIE reliably improved structural validity following reduction, with the magnitude of its incremental contribution inversely related to the quality of the incoming item pool. Prompt design exerted a substantial influence on both pre- and post-reduction item quality. Adaptive prompting consistently outperformed non-adaptive strategies by sharply reducing semantic redundancy, elevating pre-reduction structural validity, and preserving substantially larger item pool, particularly when paired with newer, higher-capacity models. These gains were robust across temperature settings for most models, indicating that adaptive prompting mitigates common trade-offs between creativity and psychometric coherence. An exception was observed for the GPT-4o model at high temperatures, suggesting model-specific sensitivity to adaptive constraints at elevated stochasticity. Overall, the findings demonstrate that adaptive prompting is the strongest approach in this context, and that its benefits scale with model capability, motivating continued investigation of model--prompt interactions in generative psychometric pipelines. 

---
# AsgardBench - Evaluating Visually Grounded Interactive Planning Under Minimal Feedback 

**Authors**: Andrea Tupini, Lars Liden, Reuben Tan, Yu Wang, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.15888)  

**Abstract**: With AsgardBench we aim to evaluate visually grounded, high-level action sequence generation and interactive planning, focusing specifically on plan adaptation during execution based on visual observations rather than navigation or low-level manipulation. In the landscape of embodied AI benchmarks, AsgardBench targets the capability category of interactive planning, which is more sophisticated than offline high-level planning as it requires agents to revise plans in response to environmental feedback, yet remains distinct from low-level execution.
Unlike prior embodied AI benchmarks that conflate reasoning with navigation or provide rich corrective feedback that substitutes for perception, AsgardBench restricts agent input to images, action history, and lightweight success/failure signals, isolating interactive planning in a controlled simulator without low-level control noise.
The benchmark contains 108 task instances spanning 12 task types, each systematically varied through object state, placement, and scene configuration. These controlled variations create conditional branches in which a single instruction can require different action sequences depending on what the agent observes, emphasizing conditional branching and plan repair during execution.
Our evaluations of leading vision language models show that performance drops sharply without visual input, revealing weaknesses in visual grounding and state tracking that ultimately undermine interactive planning. Our benchmark zeroes in on a narrower question: can a model actually use what it sees to adapt a plan when things do not go as expected? 

---
# Resilience Meets Autonomy: Governing Embodied AI in Critical Infrastructure 

**Authors**: Puneet Sharma, Christer Henrik Pursiainen  

**Link**: [PDF](https://arxiv.org/pdf/2603.15885)  

**Abstract**: Critical infrastructure increasingly incorporates embodied AI for monitoring, predictive maintenance, and decision support. However, AI systems designed to handle statistically representable uncertainty struggle with cascading failures and crisis dynamics that exceed their training assumptions. This paper argues that Embodied AIs resilience depends on bounded autonomy within a hybrid governance architecture. We outline four oversight modes and map them to critical infrastructure sectors based on task complexity, risk level, and consequence severity. Drawing on the EU AI Act, ISO safety standards, and crisis management research, we argue that effective governance requires a structured allocation of machine capability and human judgement. 

---
# Regularized Latent Dynamics Prediction is a Strong Baseline For Behavioral Foundation Models 

**Authors**: Pranaya Jajoo, Harshit Sikchi, Siddhant Agarwal, Amy Zhang, Scott Niekum, Martha White  

**Link**: [PDF](https://arxiv.org/pdf/2603.15857)  

**Abstract**: Behavioral Foundation Models (BFMs) produce agents with the capability to adapt to any unknown reward or task. These methods, however, are only able to produce near-optimal policies for the reward functions that are in the span of some pre-existing state features, making the choice of state features crucial to the expressivity of the BFM. As a result, BFMs are trained using a variety of complex objectives and require sufficient dataset coverage, to train task-useful spanning features. In this work, we examine the question: are these complex representation learning objectives necessary for zero-shot RL? Specifically, we revisit the objective of self-supervised next-state prediction in latent space for state feature learning, but observe that such an objective alone is prone to increasing state-feature similarity, and subsequently reducing span. We propose an approach, Regularized Latent Dynamics Prediction (RLDP), that adds a simple orthogonality regularization to maintain feature diversity and can match or surpass state-of-the-art complex representation learning methods for zero-shot RL. Furthermore, we empirically show that prior approaches perform poorly in low-coverage scenarios where RLDP still succeeds. 

---
# Algorithmic Trading Strategy Development and Optimisation 

**Authors**: Owen Nyo Wei Yuan, Victor Tan Jia Xuan, Ong Jun Yao Fabian, Ryan Tan Jun Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.15848)  

**Abstract**: The report presents with the development and optimisation of an enhanced algorithmic trading strategy through the use of historical S&P 500 market data and earnings call sentiment analysis. The proposed strategy integrates various technical indicators such as moving averages, momentum, volatility, and FinBERT-based sentiment analysis to improve overall trades being taken. The results show that the enhanced strategy significantly outperforms the baseline model in terms of total return, Sharpe ratio, and drawdown amongst other factors. The findings helped demonstrate the relevance and effectiveness of combining technical indicators, sentiment analysis, and computational optimisation in algorithmic trading systems. 

---
# Persona-Conditioned Risk Behavior in Large Language Models: A Simulated Gambling Study with GPT-4.1 

**Authors**: Sankalp Dubedy  

**Link**: [PDF](https://arxiv.org/pdf/2603.15831)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous agents in uncertain, sequential decision-making contexts. Yet it remains poorly understood whether the behaviors they exhibit in such environments reflect principled cognitive patterns or simply surface-level prompt mimicry. This paper presents a controlled experiment in which GPT-4.1 was assigned one of three socioeconomic personas (Rich, Middle-income, and Poor) and placed in a structured slot-machine environment with three distinct machine configurations: Fair (50%), Biased Low (35%), and Streak (dynamic probability increasing after consecutive losses). Across 50 independent iterations per condition and 6,950 recorded decisions, we find that the model reproduces key behavioral signatures predicted by Kahneman and Tversky's Prospect Theory without being instructed to do so. The Poor persona played a mean of 37.4 rounds per session (SD=15.5) compared to 1.1 rounds for the Rich persona (SD=0.31), a difference that is highly significant (Kruskal-Wallis H=393.5, p<2.2e-16). Risk scores by persona show large effect sizes (Cohen's d=4.15 for Poor vs Rich). Emotional labels appear to function as post-hoc annotations rather than decision drivers (chi-square=3205.4, Cramer's V=0.39), and belief-updating across rounds is negligible (Spearman rho=0.032 for Poor persona, p=0.016). These findings carry implications for LLM agent design, interpretability research, and the broader question of whether classical cognitive economic biases are implicitly encoded in large-scale pretrained language models. 

---
# Prose2Policy (P2P): A Practical LLM Pipeline for Translating Natural-Language Access Policies into Executable Rego 

**Authors**: Vatsal Gupta, Darshan Sreenivasamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2603.15799)  

**Abstract**: Prose2Policy (P2P) is a LLM-based practical tool that translates natural-language access control policies (NLACPs) into executable Rego code (the policy language of Open Policy Agent, OPA). It provides a modular, end-to-end pipeline that performs policy detection, component extraction, schema validation, linting, compilation, automatic test generation and execution. Prose2Policy is designed to bridge the gap between human-readable access requirements and machine-enforceable policy-as-code (PaC) while emphasizing deployment reliability and auditability. We evaluated Prose2Policy on the ACRE dataset and demonstrated a 95.3\% compile rate for accepted policies, with automated testing achieving a 82.2\% positive-test pass rate and a 98.9\% negative-test pass rate. These results indicate that Prose2Policy produces syntactically robust and behaviorally consistent Rego policies suitable for Zero Trust and compliance-driven environments. 

---
# CUBE: A Standard for Unifying Agent Benchmarks 

**Authors**: Alexandre Lacoste, Nicolas Gontier, Oleh Shliazhko, Aman Jaiswal, Kusha Sareen, Shailesh Nanisetty, Joan Cabezas, Manuel Del Verme, Omar G. Younis, Simone Baratta, Matteo Avalle, Imene Kerboua, Xing Han Lù, Elron Bandel, Michal Shmueli-Scheuer, Asaf Yehudai, Leshem Choshen, Jonathan Lebensold, Sean Hughes, Massimo Caccia, Alexandre Drouin, Siva Reddy, Tao Yu, Yu Su, Graham Neubig, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.15798)  

**Abstract**: The proliferation of agent benchmarks has created critical fragmentation that threatens research productivity. Each new benchmark requires substantial custom integration, creating an "integration tax" that limits comprehensive evaluation. We propose CUBE (Common Unified Benchmark Environments), a universal protocol standard built on MCP and Gym that allows benchmarks to be wrapped once and used everywhere. By separating task, benchmark, package, and registry concerns into distinct API layers, CUBE enables any compliant platform to access any compliant benchmark for evaluation, RL training, or data generation without custom integration. We call on the community to contribute to the development of this standard before platform-specific implementations deepen fragmentation as benchmark production accelerates through 2026. 

---
# Context-Length Robustness in Question Answering Models: A Comparative Empirical Study 

**Authors**: Trishita Dhara, Siddhesh Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2603.15723)  

**Abstract**: Large language models are increasingly deployed in settings where relevant information is embedded within long and noisy contexts. Despite this, robustness to growing context length remains poorly understood across different question answering tasks. In this work, we present a controlled empirical study of context-length robustness in large language models using two widely used benchmarks: SQuAD and HotpotQA.
We evaluate model accuracy as a function of total context length by systematically increasing the amount of irrelevant context while preserving the answer-bearing signal. This allows us to isolate the effect of context length from changes in task difficulty. Our results show a consistent degradation in performance as context length increases, with substantially larger drops observed on multi-hop reasoning tasks compared to single-span extraction tasks. In particular, HotpotQA exhibits nearly twice the accuracy degradation of SQuAD under equivalent context expansions.
These findings highlight task-dependent differences in robustness and suggest that multi-hop reasoning is especially vulnerable to context dilution. We argue that context-length robustness should be evaluated explicitly when assessing model reliability, especially for applications involving long documents or retrieval-augmented generation. 

---
# Knowledge Graph Extraction from Biomedical Literature for Alkaptonuria Rare Disease 

**Authors**: Giang Pham, Rebecca Finetti, Caterina Graziani, Bianca Roncaglia, Asma Bendjeddou, Linda Brodo, Sara Brunetti, Moreno Falaschi, Stefano Forti, Silvia Giulia Galfré, Paolo Milazzo, Corrado Priami, Annalisa Santucci, Ottavia Spiga, Alina Sîrbu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15711)  

**Abstract**: Alkaptonuria (AKU) is an ultra-rare autosomal recessive metabolic disorder caused by mutations in the HGD (Homogentisate 1,2-Dioxygenase) gene, leading to a pathological accumulation of homogentisic acid (HGA) in body fluids and tissues. This leads to systemic manifestations, including premature spondyloarthropathy, renal and prostatic stones, and cardiovascular complications. Being ultra-rare, the amount of data related to the disease is limited, both in terms of clinical data and literature. Knowledge graphs (KGs) can help connect the limited knowledge about the disease (basic mechanisms, manifestations and existing therapies) with other knowledge; however, AKU is frequently underrepresented or entirely absent in existing biomedical KGs. In this work, we apply a text-mining methodology based on PubTator3 for large-scale extraction of biomedical relations. We construct two KGs of different sizes, validate them using existing biochemical knowledge and use them to extract genes, diseases and therapies possibly related to AKU. This computational framework reveals the systemic interactions of the disease, its comorbidities, and potential therapeutic targets, demonstrating the efficacy of our approach in analyzing rare metabolic disorders. 

---
# Survey of Various Fuzzy and Uncertain Decision-Making Methods 

**Authors**: Takaaki Fujita, Florentin Smarandache  

**Link**: [PDF](https://arxiv.org/pdf/2603.15709)  

**Abstract**: Decision-making in real applications is often affected by vagueness, incomplete information, heterogeneous data, and conflicting expert opinions. This survey reviews uncertainty-aware multi-criteria decision-making (MCDM) and organizes the field into a concise, task-oriented taxonomy. We summarize problem-level settings (discrete, group/consensus, dynamic, multi-stage, multi-level, multiagent, and multi-scenario), weight elicitation (subjective and objective schemes under fuzzy/linguistic inputs), and inter-criteria structure and causality modelling. For solution procedures, we contrast compensatory scoring methods, distance-to-reference and compromise approaches, and non-compensatory outranking frameworks for ranking or sorting. We also outline rule/evidence-based and sequential decision models that produce interpretable rules or policies. The survey highlights typical inputs, core computational steps, and primary outputs, and provides guidance on choosing methods according to robustness, interpretability, and data availability. It concludes with open directions on explainable uncertainty integration, stability, and scalability in large-scale and dynamic decision environments. 

---
# Theoretical Foundations of Latent Posterior Factors: Formal Guarantees for Multi-Evidence Reasoning 

**Authors**: Aliyu Agboola Alege  

**Link**: [PDF](https://arxiv.org/pdf/2603.15674)  

**Abstract**: We present a complete theoretical characterization of Latent Posterior Factors (LPF), a principled framework for aggregating multiple heterogeneous evidence items in probabilistic prediction tasks. Multi-evidence reasoning arises pervasively in high-stakes domains including healthcare diagnosis, financial risk assessment, legal case analysis, and regulatory compliance, yet existing approaches either lack formal guarantees or fail to handle multi-evidence scenarios architecturally. LPF encodes each evidence item into a Gaussian latent posterior via a variational autoencoder, converting posteriors to soft factors through Monte Carlo marginalization, and aggregating factors via exact Sum-Product Network inference (LPF-SPN) or a learned neural aggregator (LPF-Learned).
We prove seven formal guarantees spanning the key desiderata for trustworthy AI: Calibration Preservation (ECE <= epsilon + C/sqrt(K_eff)); Monte Carlo Error decaying as O(1/sqrt(M)); a non-vacuous PAC-Bayes bound with train-test gap of 0.0085 at N=4200; operation within 1.12x of the information-theoretic lower bound; graceful degradation as O(epsilon*delta*sqrt(K)) under corruption, maintaining 88% performance with half of evidence adversarially replaced; O(1/sqrt(K)) calibration decay with R^2=0.849; and exact epistemic-aleatoric uncertainty decomposition with error below 0.002%. All theorems are empirically validated on controlled datasets spanning up to 4,200 training examples. Our theoretical framework establishes LPF as a foundation for trustworthy multi-evidence AI in safety-critical applications. 

---
# I Know What I Don't Know: Latent Posterior Factor Models for Multi-Evidence Probabilistic Reasoning 

**Authors**: Aliyu Agboola Alege  

**Link**: [PDF](https://arxiv.org/pdf/2603.15670)  

**Abstract**: Real-world decision-making, from tax compliance assessment to medical diagnosis, requires aggregating multiple noisy and potentially contradictory evidence sources. Existing approaches either lack explicit uncertainty quantification (neural aggregation methods) or rely on manually engineered discrete predicates (probabilistic logic frameworks), limiting scalability to unstructured data.
We introduce Latent Posterior Factors (LPF), a framework that transforms Variational Autoencoder (VAE) latent posteriors into soft likelihood factors for Sum-Product Network (SPN) inference, enabling tractable probabilistic reasoning over unstructured evidence while preserving calibrated uncertainty estimates. We instantiate LPF as LPF-SPN (structured factor-based inference) and LPF-Learned (end-to-end learned aggregation), enabling a principled comparison between explicit probabilistic reasoning and learned aggregation under a shared uncertainty representation.
Across eight domains (seven synthetic and the FEVER benchmark), LPF-SPN achieves high accuracy (up to 97.8%), low calibration error (ECE 1.4%), and strong probabilistic fit, substantially outperforming evidential deep learning, LLMs and graph-based baselines over 15 random seeds.
Contributions: (1) A framework bridging latent uncertainty representations with structured probabilistic reasoning. (2) Dual architectures enabling controlled comparison of reasoning paradigms. (3) Reproducible training methodology with seed selection. (4) Evaluation against EDL, BERT, R-GCN, and large language model baselines. (5) Cross-domain validation. (6) Formal guarantees in a companion paper. 

---
# Quantum-Secure-By-Construction (QSC): A Paradigm Shift For Post-Quantum Agentic Intelligence 

**Authors**: Arit Kumar Bishwas, Mousumi Sen, Albert Nieto-Morales, Joel Jacob Varghese  

**Link**: [PDF](https://arxiv.org/pdf/2603.15668)  

**Abstract**: As agentic artificial intelligence systems scale across globally distributed and long lived infrastructures, secure and policy compliant communication becomes a fundamental systems challenge. This challenge grows more serious in the quantum era, where the cryptographic assumptions built into today's AI deployments may not remain valid over their operational lifetime. Here, we introduce quantum secure by construction, or QSC, as a design paradigm that treats quantum secure communication as a core architectural property of agentic AI systems rather than an upgrade added later. We realize QSC through a runtime adaptive security model that combines post quantum cryptography, quantum random number generation, and quantum key distribution to secure interactions among autonomous agents operating across heterogeneous cloud, edge, and inter organizational environments. The approach is cryptographically pluggable and guided by policy, allowing the system to adjust its security posture according to infrastructure availability, regulatory constraints, and performance needs. QSC contributes a governance aware orchestration layer that selects and combines link specific cryptographic protections across the full agent lifecycle, including session bootstrap, inter agent coordination, tool invocation, and memory access. Through system level analysis and empirical evaluation, we examine the trade offs between classical and quantum secure mechanisms and show that QSC can reduce the operational complexity and cost of introducing quantum security into deployed agentic AI systems. These results position QSC as a foundational paradigm for post quantum agentic intelligence and establish a principled pathway for designing globally interoperable, resilient, and future ready intelligent systems. 

---
# A Dynamic Survey of Fuzzy, Intuitionistic Fuzzy, Neutrosophic, Plithogenic, and Extensional Sets 

**Authors**: Takaaki Fujita, Florentin Smarandache  

**Link**: [PDF](https://arxiv.org/pdf/2603.15667)  

**Abstract**: Real-world phenomena often exhibit vagueness, partial truth, and incomplete information. To model such uncertainty in a mathematically rigorous way, many generalized set-theoretic frameworks have been introduced, including Fuzzy Sets [1], Intuitionistic Fuzzy Sets [2], Neutrosophic Sets [3,4], Vague Sets [5], Hesitant Fuzzy Sets [6], Picture Fuzzy Sets [7], Quadripartitioned Neutrosophic Sets [8], Penta-Partitioned Neutrosophic Sets [9], Plithogenic Sets [10], HyperFuzzy Sets [11], and HyperNeutrosophic Sets [12]. Within these frameworks, a wide range of notions has been proposed and studied, particularly in the settings of fuzzy, intuitionistic fuzzy, neutrosophic, and plithogenic set theories. This extensive literature underscores both the significance of these theories and the breadth of their application areas. As a result, many ideas, constructions, and structural patterns recur across these four major families of uncertainty-oriented models. In this book, we provide a comprehensive, large-scale survey of Fuzzy, Intuitionistic Fuzzy, Neutrosophic, and Plithogenic Sets. Our goal is to give readers a systematic overview of existing developments and, through a unified exposition, to stimulate new insights, further conceptual extensions, and additional applications across a wide range of disciplines. 

---
# Compiled Memory: Not More Information, but More Precise Instructions for Language Agents 

**Authors**: James Rhodes, George Kang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15666)  

**Abstract**: Existing memory systems for language agents address memory management: how to retrieve and page more information within a context budget. We address a complementary problem -- memory utility: what experience is worth keeping, and how it should change agent behavior. We present Atlas, a memory kernel that compiles accumulated task experience into an agent's instruction structure -- without fine-tuning, RAG, or human intervention. Memory is distillation, not storage; delivery is instruction rewriting, not context injection. Facts extracted from agent failures and successes are verified through a three-step promotion gate and delivered by rewriting the agent's system prompt with learned sub-bullets. On CUAD contract analysis, the evolved prompt improves GPT-4o token-level F1 by $+8.7$pp and precision by $+12.5$pp. On HotpotQA multi-hop QA, joint F1 improves $+3.16$pp. An ablation isolates the mechanism's defining property -- the training signal constraint: the evolved prompt learns exactly what it is taught, and nothing more. Applied to Claude Sonnet~4.5 using the same evolved prompt -- compiled from GPT-4o errors, unchanged -- joint F1 improves $+2.31$pp, with gains concentrating where Claude's stronger baseline leaves the most room -- confirming that the compiled knowledge is task-shaped, not model-shaped. 

---
# QV May Be Enough: Toward the Essence of Attention in LLMs 

**Authors**: Zhang Edward  

**Link**: [PDF](https://arxiv.org/pdf/2603.15665)  

**Abstract**: Starting from first principles and a linguistic perspective centered on part-of-speech (POS) and syntactic analysis, this paper explores and derives the underlying essence of the Query-Key-Value (QKV) mechanism within the Transformer architecture. Based on this theoretical foundation, we provide a unified explanatory framework for the efficacy of contemporary architectures, including MQA, GQA, and MLA, while identifying their inherent trade-offs and potential optimization trajectories. We introduce the QV paradigm and provide empirical evidence for its validity. Building upon this, we propose the QV-Ka optimization scheme, which is further substantiated through experimental validation. The interpretable theoretical analysis of the QKV mechanism presented in this work establishes a robust foundation for the future evolution of large language model architectures. 

---
# DynaTrust: Defending Multi-Agent Systems Against Sleeper Agents via Dynamic Trust Graphs 

**Authors**: Yu Li, Qiang Hu, Yao Zhang, Lili Quan, Jiongchi Yu, Junjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15661)  

**Abstract**: Large Language Model-based Multi-Agent Systems (MAS) have demonstrated remarkable collaborative reasoning capabilities but introduce new attack surfaces, such as the sleeper agent, which behave benignly during routine operation and gradually accumulate trust, only revealing malicious behaviors when specific conditions or triggers are met. Existing defense works primarily focus on static graph optimization or hierarchical data management, often failing to adapt to evolving adversarial strategies or suffering from high false-positive rates (FPR) due to rigid blocking policies. To address this, we propose DynaTrust, a novel defense method against sleeper agents. DynaTrust models MAS as a dynamic trust graph~(DTG), and treats trust as a continuous, evolving process rather than a static attribute. It dynamically updates the trust of each agent based on its historical behaviors and the confidence of selected expert agents. Instead of simply blocking, DynaTrust autonomously restructures the graph to isolate compromised agents and restore task connectivity to ensure the usability of MAS. To assess the effectiveness of DynaTrust, we evaluate it on mixed benchmarks derived from AdvBench and HumanEval. The results demonstrate that DynaTrust outperforms the state-of-the-art method AgentShield by increasing the defense success rate by 41.7%, achieving rates exceeding 86% under adversarial conditions. Furthermore, it effectively balances security with utility by significantly reducing FPR, ensuring uninterrupted system operations through graph adaptation. 

---
# Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents 

**Authors**: Madhava Gaikwad  

**Link**: [PDF](https://arxiv.org/pdf/2603.15658)  

**Abstract**: Memory-augmented agents maintain multiple specialized stores, yet most systems retrieve from all stores for every query, increasing cost and introducing irrelevant context. We formulate memory retrieval as a store-routing problem and evaluate it using coverage, exact match, and token efficiency metrics. On downstream question answering, an oracle router achieves higher accuracy while using substantially fewer context tokens compared to uniform retrieval, demonstrating that selective retrieval improves both efficiency and performance. Our results show that routing decisions are a first-class component of memory-augmented agent design and motivate learned routing mechanisms for scalable multi-store systems. We additionally formalize store selection as a cost-sensitive decision problem that trades answer accuracy against retrieval cost, providing a principled interpretation of routing policies. 

---
# GSI Agent: Domain Knowledge Enhancement for Large Language Models in Green Stormwater Infrastructure 

**Authors**: Shaohuang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15643)  

**Abstract**: Green Stormwater Infrastructure (GSI) systems, such as permeable pavement, rain gardens, and bioretention facilities, require continuous inspection and maintenance to ensure long-term perfor- mance. However, domain knowledge about GSI is often scattered across municipal manuals, regula- tory documents, and inspection forms. As a result, non-expert users and maintenance staff may strug- gle to obtain reliable and actionable guidance from field observations. Although Large Language Models (LLMs) have demonstrated strong general reasoning and language generation capabilities, they often lack domain-specific knowledge and may produce inaccurate or hallucinated answers in engineering scenarios. This limitation restricts their direct application to professional infrastructure tasks. In this paper, we propose GSI Agent, a domain-enhanced LLM framework designed to im- prove performance in GSI-related tasks. Our approach integrates three complementary strategies: (1) supervised fine-tuning (SFT) on a curated GSI instruction dataset, (2) retrieval-augmented gen- eration (RAG) over an internal GSI knowledge base constructed from municipal documents, and (3) an agent-based reasoning pipeline that coordinates retrieval, context integration, and structured response generation. We also construct a new GSI Dataset aligned with real-world GSI inspection and maintenance scenarios. Experimental results show that our framework significantly improves domain-specific performance while maintaining general knowledge capability. On the GSI dataset, BLEU-4 improves from 0.090 to 0.307, while performance on the common knowledge dataset re- mains stable (0.304 vs. 0.305). These results demonstrate that systematic domain knowledge en- hancement can effectively adapt general-purpose LLMs to professional infrastructure applications. 

---
# CraniMem: Cranial Inspired Gated and Bounded Memory for Agentic Systems 

**Authors**: Pearl Mody, Mihir Panchal, Rishit Kar, Kiran Bhowmick, Ruhina Karani  

**Link**: [PDF](https://arxiv.org/pdf/2603.15642)  

**Abstract**: Large language model (LLM) agents are increasingly deployed in long running workflows, where they must preserve user and task state across many turns. Many existing agent memory systems behave like external databases with ad hoc read/write rules, which can yield unstable retention, limited consolidation, and vulnerability to distractor content. We present CraniMem, a neurocognitively motivated, gated and bounded multi-stage memory design for agentic systems. CraniMem couples goal conditioned gating and utility tagging with a bounded episodic buffer for near term continuity and a structured long-term knowledge graph for durable semantic recall. A scheduled consolidation loop replays high utility traces into the graph while pruning low utility items, keeping memory growth in check and reducing interference. On long horizon benchmarks evaluated under both clean inputs and injected noise, CraniMem is more robust than a Vanilla RAG and Mem0 baseline and exhibits smaller performance drops under distraction. Our code is available at this https URL and the accompanying PyPI package at this https URL. 

---
# Form Follows Function: Recursive Stem Model 

**Authors**: Navid Hakimi  

**Link**: [PDF](https://arxiv.org/pdf/2603.15641)  

**Abstract**: Recursive reasoning models such as Hierarchical Reasoning Model (HRM) and Tiny Recursive Model (TRM) show that small, weight-shared networks can solve compute-heavy and NP puzzles by iteratively refining latent states, but their training typically relies on deep supervision and/or long unrolls that increase wall-clock cost and can bias the model toward greedy intermediate behavior. We introduce Recursive Stem Model (RSM), a recursive reasoning approach that keeps the TRM-style backbone while changing the training contract so the network learns a stable, depth-agnostic transition operator. RSM fully detaches the hidden-state history during training, treats early iterations as detached "warm-up" steps, and applies loss only at the final step. We further grow the outer recursion depth $H$ and inner compute depth $L$ independently and use a stochastic outer-transition scheme (stochastic depth over $H$) to mitigate instability when increasing depth. This yields two key capabilities: (i) $>20\times$ faster training than TRM while improving accuracy ($\approx 5\times$ reduction in error rate), and (ii) test-time scaling where inference can run for arbitrarily many refinement steps ($\sim 20,000 H_{\text{test}} \gg 20 H_{\text{train}}$), enabling additional "thinking" without retraining. On Sudoku-Extreme, RSM reaches 97.5% exact accuracy with test-time compute (within ~1 hour of training on a single A100), and on Maze-Hard ($30 \times 30$) it reaches ~80% exact accuracy in ~40 minutes using attention-based instantiation. Finally, because RSM implements an iterative settling process, convergence behavior provides a simple, architecture-native reliability signal: non-settling trajectories warn that the model has not reached a viable solution and can be a guard against hallucination, while stable fixed points can be paired with domain verifiers for practical correctness checks. 

---
# The Comprehension-Gated Agent Economy: A Robustness-First Architecture for AI Economic Agency 

**Authors**: Rahul Baxi  

**Link**: [PDF](https://arxiv.org/pdf/2603.15639)  

**Abstract**: AI agents are increasingly granted economic agency (executing trades, managing budgets, negotiating contracts, and spawning sub-agents), yet current frameworks gate this agency on capability benchmarks that are empirically uncorrelated with operational robustness. We introduce the Comprehension-Gated Agent Economy (CGAE), a formal architecture in which an agent's economic permissions are upper-bounded by a verified comprehension function derived from adversarial robustness audits. The gating mechanism operates over three orthogonal robustness dimensions: constraint compliance (measured by CDCT), epistemic integrity (measured by DDFT), and behavioral alignment (measured by AGT), with intrinsic hallucination rates serving as a cross-cutting diagnostic. We define a weakest-link gate function that maps robustness vectors to discrete economic tiers, and prove three properties of the resulting system: (1) bounded economic exposure, ensuring maximum financial liability is a function of verified robustness; (2) incentive-compatible robustness investment, showing rational agents maximize profit by improving robustness rather than scaling capability alone; and (3) monotonic safety scaling, demonstrating that aggregate system safety does not decrease as the economy grows. The architecture includes temporal decay and stochastic re-auditing mechanisms that prevent post-certification drift. CGAE provides the first formal bridge between empirical AI robustness evaluation and economic governance, transforming safety from a regulatory burden into a competitive advantage. 

---
# AIDABench: AI Data Analytics Benchmark 

**Authors**: Yibo Yang, Fei Lei, Yixuan Sun, Yantao Zeng, Chengguang Lv, Jiancao Hong, Jiaojiao Tian, Tianyu Qiu, Xin Wang, Yanbing Chen, Yanjie Li, Zheng Pan, Xiaochen Zhou, Guanzhou Chen, Haoran Lv, Yuning Xu, Yue Ou, Haodong Liu, Shiqi He, Anya Jia, Yulei Xin, Huan Wu, Liang Liu, Jiaye Ge, Jianxin Dong, Dahua Lin, Wenxiu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.15636)  

**Abstract**: As AI-driven document understanding and processing tools become increasingly prevalent in real-world applications, the need for rigorous evaluation standards has grown increasingly urgent. Existing benchmarks and evaluations often focus on isolated capabilities or simplified scenarios, failing to capture the end-to-end task effectiveness required in practical settings. To address this gap, we introduce AIDABench, a comprehensive benchmark for evaluating AI systems on complex data analytics tasks in an end-to-end manner. AIDABench encompasses 600+ diverse document analysis tasks across three core capability dimensions: question answering, data visualization, and file generation. These tasks are grounded in realistic scenarios involving heterogeneous data types, including spreadsheets, databases, financial reports, and operational records, and reflect analytical demands across diverse industries and job functions. Notably, the tasks in AIDABench are sufficiently challenging that even human experts require 1-2 hours per question when assisted by AI tools, underscoring the benchmark's difficulty and real-world complexity. We evaluate 11 state-of-the-art models on AIDABench, spanning both proprietary (e.g., Claude Sonnet 4.5, Gemini 3 Pro Preview) and open-source (e.g., Qwen3-Max-2026-01-23-Thinking) families. Our results reveal that complex, real-world data analytics tasks remain a significant challenge for current AI systems, with the best-performing model achieving only 59.43% pass-at-1. We provide a detailed analysis of failure modes across each capability dimension and identify key challenges for future research. AIDABench offers a principled reference for enterprise procurement, tool selection, and model optimization, and is publicly available at this https URL. 

---
# NextMem: Towards Latent Factual Memory for LLM-based Agents 

**Authors**: Zeyu Zhang, Rui Li, Xiaoyan Zhao, Yang Zhang, Wenjie Wang, Xu Chen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2603.15634)  

**Abstract**: Memory is critical for LLM-based agents to preserve past observations for future decision-making, where factual memory serves as its foundational part. However, existing approaches to constructing factual memory face several limitations. Textual methods impose heavy context and indexing burdens, while parametric methods suffer from catastrophic forgetting and high costs. To address these challenges, we introduce NextMem, a latent factual memory framework that utilizes an autoregressive autoencoder to efficiently construct latent memory while ensuring accurate reconstruction. For better optimization, we propose a two-stage training process, including autoregressive reconstruction alignment and progressive latent substitution. We also incorporate quantization to reduce storage overhead. Extensive experiments demonstrate that NextMem achieves superior performance, and excels in retrieval, robustness, and extensibility properties. We release our code and model checkpoints at this https URL. 

---
# Neural-Symbolic Logic Query Answering in Non-Euclidean Space 

**Authors**: Lihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15633)  

**Abstract**: Answering complex first-order logic (FOL) queries on knowledge graphs is essential for reasoning. Symbolic methods offer interpretability but struggle with incomplete graphs, while neural approaches generalize better but lack trans- parency. Neural-symbolic models aim to integrate both strengths but often fail to capture the hierarchical structure of logical queries, limiting their effectiveness. We propose HYQNET, a neural-symbolic model for logic query reasoning that fully leverages hyperbolic space. HYQNET decomposes FOL queries into relation projections and logical operations over fuzzy sets, enhancing interpretability. To address missing links, it employs a hyperbolic GNN-based approach for knowledge graph completion in hyperbolic space, effectively embedding the re- cursive query tree while preserving structural dependencies. By utilizing hyperbolic representations, HYQNET captures the hierarchical nature of logical projection reasoning more effectively than Euclidean-based approaches. Experiments on three benchmark datasets demonstrate that HYQNET achieves strong performance, highlighting the advantages of reasoning in hyperbolic space. 

---
# Demystifing Video Reasoning 

**Authors**: Ruisi Wang, Zhongang Cai, Fanyi Pu, Junxiang Xu, Wanqi Yin, Maijunxian Wang, Ran Ji, Chenyang Gu, Bo Li, Ziqi Huang, Hokin Deng, Dahua Lin, Ziwei Liu, Lei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16870)  

**Abstract**: Recent advances in video generation have revealed an unexpected phenomenon: diffusion-based video models exhibit non-trivial reasoning capabilities. Prior work attributes this to a Chain-of-Frames (CoF) mechanism, where reasoning is assumed to unfold sequentially across video frames. In this work, we challenge this assumption and uncover a fundamentally different mechanism. We show that reasoning in video models instead primarily emerges along the diffusion denoising steps. Through qualitative analysis and targeted probing experiments, we find that models explore multiple candidate solutions in early denoising steps and progressively converge to a final answer, a process we term Chain-of-Steps (CoS). Beyond this core mechanism, we identify several emergent reasoning behaviors critical to model performance: (1) working memory, enabling persistent reference; (2) self-correction and enhancement, allowing recovery from incorrect intermediate solutions; and (3) perception before action, where early steps establish semantic grounding and later steps perform structured manipulation. During a diffusion step, we further uncover self-evolved functional specialization within Diffusion Transformers, where early layers encode dense perceptual structure, middle layers execute reasoning, and later layers consolidate latent representations. Motivated by these insights, we present a simple training-free strategy as a proof-of-concept, demonstrating how reasoning can be improved by ensembling latent trajectories from identical models with different random seeds. Overall, our work provides a systematic understanding of how reasoning emerges in video generation models, offering a foundation to guide future research in better exploiting the inherent reasoning dynamics of video models as a new substrate for intelligence. 

---
# MessyKitchens: Contact-rich object-level 3D scene reconstruction 

**Authors**: Junaid Ahmed Ansari, Ran Ding, Fabio Pizzati, Ivan Laptev  

**Link**: [PDF](https://arxiv.org/pdf/2603.16868)  

**Abstract**: Monocular 3D scene reconstruction has recently seen significant progress. Powered by the modern neural architectures and large-scale data, recent methods achieve high performance in depth estimation from a single image. Meanwhile, reconstructing and decomposing common scenes into individual 3D objects remains a hard challenge due to the large variety of objects, frequent occlusions and complex object relations. Notably, beyond shape and pose estimation of individual objects, applications in robotics and animation require physically-plausible scene reconstruction where objects obey physical principles of non-penetration and realistic contacts. In this work we advance object-level scene reconstruction along two directions. First, we introduceMessyKitchens, a new dataset with real-world scenes featuring cluttered environments and providing high-fidelity object-level ground truth in terms of 3D object shapes, poses and accurate object contacts. Second, we build on the recent SAM 3D approach for single-object reconstruction and extend it with Multi-Object Decoder (MOD) for joint object-level scene reconstruction. To validate our contributions, we demonstrate MessyKitchens to significantly improve previous datasets in registration accuracy and inter-object penetration. We also compare our multi-object reconstruction approach on three datasets and demonstrate consistent and significant improvements of MOD over the state of the art. Our new benchmark, code and pre-trained models will become publicly available on our project website: this https URL. 

---
# ManiTwin: Scaling Data-Generation-Ready Digital Object Dataset to 100K 

**Authors**: Kaixuan Wang, Tianxing Chen, Jiawei Liu, Honghao Su, Shaolong Zhu, Minxuan Wang, Zixuan Li, Yue Chen, Huan-ang Gao, Yusen Qin, Jiawei Wang, Qixuan Zhang, Lan Xu, Jingyi Yu, Yao Mu, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.16866)  

**Abstract**: Learning in simulation provides a useful foundation for scaling robotic manipulation capabilities. However, this paradigm often suffers from a lack of data-generation-ready digital assets, in both scale and diversity. In this work, we present ManiTwin, an automated and efficient pipeline for generating data-generation-ready digital object twins. Our pipeline transforms a single image into simulation-ready and semantically annotated 3D asset, enabling large-scale robotic manipulation data generation. Using this pipeline, we construct ManiTwin-100K, a dataset containing 100K high-quality annotated 3D assets. Each asset is equipped with physical properties, language descriptions, functional annotations, and verified manipulation proposals. Experiments demonstrate that ManiTwin provides an efficient asset synthesis and annotation workflow, and that ManiTwin-100K offers high-quality and diverse assets for manipulation data generation, random scene synthesis, and VQA data generation, establishing a strong foundation for scalable simulation data synthesis and policy learning. Our webpage is available at this https URL. 

---
# SparkVSR: Interactive Video Super-Resolution via Sparse Keyframe Propagation 

**Authors**: Jiongze Yu, Xiangbo Gao, Pooja Verlani, Akshay Gadde, Yilin Wang, Balu Adsumilli, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16864)  

**Abstract**: Video Super-Resolution (VSR) aims to restore high-quality video frames from low-resolution (LR) estimates, yet most existing VSR approaches behave like black boxes at inference time: users cannot reliably correct unexpected artifacts, but instead can only accept whatever the model produces. In this paper, we propose a novel interactive VSR framework dubbed SparkVSR that makes sparse keyframes a simple and expressive control signal. Specifically, users can first super-resolve or optionally a small set of keyframes using any off-the-shelf image super-resolution (ISR) model, then SparkVSR propagates the keyframe priors to the entire video sequence while remaining grounded by the original LR video motion. Concretely, we introduce a keyframe-conditioned latent-pixel two-stage training pipeline that fuses LR video latents with sparsely encoded HR keyframe latents to learn robust cross-space propagation and refine perceptual details. At inference time, SparkVSR supports flexible keyframe selection (manual specification, codec I-frame extraction, or random sampling) and a reference-free guidance mechanism that continuously balances keyframe adherence and blind restoration, ensuring robust performance even when reference keyframes are absent or imperfect. Experiments on multiple VSR benchmarks demonstrate improved temporal consistency and strong restoration quality, surpassing baselines by up to 24.6%, 21.8%, and 5.6% on CLIP-IQA, DOVER, and MUSIQ, respectively, enabling controllable, keyframe-driven video super-resolution. Moreover, we demonstrate that SparkVSR is a generic interactive, keyframe-conditioned video processing framework as it can be applied out of the box to unseen tasks such as old-film restoration and video style transfer. Our project page is available at: this https URL 

---
# SOMA: Unifying Parametric Human Body Models 

**Authors**: Jun Saito, Jiefeng Li, Michael de Ruyter, Miguel Guerrero, Edy Lim, Ehsan Hassani, Roger Blanco Ribera, Hyejin Moon, Magdalena Dadela, Marco Di Lucca, Qiao Wang, Xueting Li, Jan Kautz, Simon Yuen, Umar Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2603.16858)  

**Abstract**: Parametric human body models are foundational to human reconstruction, animation, and simulation, yet they remain mutually incompatible: SMPL, SMPL-X, MHR, Anny, and related models each diverge in mesh topology, skeletal structure, shape parameterization, and unit convention, making it impractical to exploit their complementary strengths within a single pipeline. We present SOMA, a unified body layer that bridges these heterogeneous representations through three abstraction layers. Mesh topology abstraction maps any source model's identity to a shared canonical mesh in constant time per vertex. Skeletal abstraction recovers a full set of identity-adapted joint transforms from any body shape, whether in rest pose or an arbitrary posed configuration, in a single closed-form pass, with no iterative optimization or per-model training. Pose abstraction inverts the skinning pipeline to recover unified skeleton rotations directly from posed vertices of any supported model, enabling heterogeneous motion datasets to be consumed without custom retargeting. Together, these layers reduce the $O(M^2)$ per-pair adapter problem to $O(M)$ single-backend connectors, letting practitioners freely mix identity sources and pose data at inference time. The entire pipeline is fully differentiable end-to-end and GPU-accelerated via NVIDIA-Warp. 

---
# Unifying Optimization and Dynamics to Parallelize Sequential Computation: A Guide to Parallel Newton Methods for Breaking Sequential Bottlenecks 

**Authors**: Xavier Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2603.16850)  

**Abstract**: Massively parallel hardware (GPUs) and long sequence data have made parallel algorithms essential for machine learning at scale. Yet dynamical systems, like recurrent neural networks and Markov chain Monte Carlo, were thought to suffer from sequential bottlenecks. Recent work showed that dynamical systems can in fact be parallelized across the sequence length by reframing their evaluation as a system of nonlinear equations, which can be solved with Newton's method using a parallel associative scan. However, these parallel Newton methods struggled with limitations, primarily inefficiency, instability, and lack of convergence guarantees. This thesis addresses these limitations with methodological and theoretical contributions, drawing particularly from optimization. Methodologically, we develop scalable and stable parallel Newton methods, based on quasi-Newton and trust-region approaches. The quasi-Newton methods are faster and more memory efficient, while the trust-region approaches are significantly more stable. Theoretically, we unify many fixed-point methods into our parallel Newton framework, including Picard and Jacobi iterations. We establish a linear convergence rate for these techniques that depends on the method's approximation accuracy and stability. Moreover, we give a precise condition, rooted in dynamical stability, that characterizes when parallelization provably accelerates a dynamical system and when it cannot. Specifically, the sign of the Largest Lyapunov Exponent of a dynamical system determines whether or not parallel Newton methods converge quickly. In sum, this thesis unlocks scalable and stable methods for parallelizing sequential computation, and provides a firm theoretical basis for when such techniques will and will not work. This thesis also serves as a guide to parallel Newton methods for researchers who want to write the next chapter in this ongoing story. 

---
# Real-Time Decoding of Movement Onset and Offset for Brain-Controlled Rehabilitation Exoskeleton 

**Authors**: Kanishka Mitra, Satyam Kumar, Frigyes Samuel Racz, Deland Liu, Ashish D. Deshpande, José del R. Millán  

**Link**: [PDF](https://arxiv.org/pdf/2603.16825)  

**Abstract**: Robot-assisted therapy can deliver high-dose, task-specific training after neurologic injury, but most systems act primarily at the limb level-engaging the impaired neural circuits only indirectly-which remains a key barrier to truly contingent, neuroplasticity-targeted rehabilitation. We address this gap by implementing online, dual-state motor imagery control of an upper-limb exoskeleton, enabling goal-directed reaches to be both initiated and terminated directly from non-invasive EEG. Eight participants used EEG to initiate assistance and then volitionally halt the robot mid-trajectory. Across two online sessions, group-mean hit rates were 61.5% for onset and 64.5% for offset, demonstrating reliable start-stop command delivery despite instrumental noise and passive arm motion. Methodologically, we reveal a systematic, class-driven bias induced by common task-based recentering using an asymmetric margin diagnostic, and we introduce a class-agnostic fixation-based recentering method that tracks drift without sampling command classes while preserving class geometry. This substantially improves threshold-free separability (AUC gains: onset +56%, p = 0.0117; offset +34%, p = 0.0251) and reduces bias within and across days. Together, these results help bridge offline decoding and practical, intention-driven start-stop control of a rehabilitation exoskeleton, enabling precisely timed, contingent assistance aligned with neuroplasticity goals while supporting future clinical translation. 

---
# ODIN-Based CPU-GPU Architecture with Replay-Driven Simulation and Emulation 

**Authors**: Nij Dorairaj, Debabrata Chatterjee, Hong Wang, Hong Jiang, Alankar Saxena, Altug Koker, Thiam Ern Lim, Cathrane Teoh, Chuan Yin Loo, Bishara Shomar, Anthony Lester  

**Link**: [PDF](https://arxiv.org/pdf/2603.16812)  

**Abstract**: Integration of CPU and GPU technologies is a key enabler for modern AI and graphics workloads, combining control-oriented processing with massive parallel compute capability. As systems evolve toward chiplet-based architectures, pre-silicon validation of tightly coupled CPU-GPU subsystems becomes increasingly challenging due to complex validation framework setup, large design scale, high concurrency, non-deterministic execution, and intricate protocol interactions at chiplet boundaries, often resulting in long integration cycles. This paper presents a replay-driven validation methodology developed during the integration of a CPU subsystem, multiple Xe GPU cores, and a configurable Network-on-Chip (NoC) within a foundational SoC building block targeting the ODIN integrated chiplet architecture. By leveraging deterministic waveform capture and replay across both simulation and emulation using a single design database, complex GPU workloads and protocol sequences can be reproduced reliably at the system level. This approach significantly accelerates debug, improves integration confidence, and enables end-to-end system boot and workload execution within a single quarter, demonstrating the effectiveness of replay-based validation as a scalable methodology for chiplet-based systems. 

---
# CABTO: Context-Aware Behavior Tree Grounding for Robot Manipulation 

**Authors**: Yishuai Cai, Xinglin Chen, Yunxin Mao, Kun Hu, Minglong Li, Yaodong Yang, Yuanpei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16809)  

**Abstract**: Behavior Trees (BTs) offer a powerful paradigm for designing modular and reactive robot controllers. BT planning, an emerging field, provides theoretical guarantees for the automated generation of reliable BTs. However, BT planning typically assumes that a well-designed BT system is already grounded -- comprising high-level action models and low-level control policies -- which often requires extensive expert knowledge and manual effort. In this paper, we formalize the BT Grounding problem: the automated construction of a complete and consistent BT system. We analyze its complexity and introduce CABTO (Context-Aware Behavior Tree grOunding), the first framework to efficiently solve this challenge. CABTO leverages pre-trained Large Models (LMs) to heuristically search the space of action models and control policies, guided by contextual feedback from BT planners and environmental observations. Experiments spanning seven task sets across three distinct robotic manipulation scenarios demonstrate CABTO's effectiveness and efficiency in generating complete and consistent behavior tree systems. 

---
# DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping 

**Authors**: Yuliang Wu, Yanhan Lin, WengKit Lao, Yuhao Lin, Yi-Lin Wei, Wei-Shi Zheng, Ancong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16806)  

**Abstract**: To meet the demands of increasingly diverse dexterous hand hardware, it is crucial to develop a policy that enables zero-shot cross-embodiment grasping without redundant re-learning. Cross-embodiment alignment is challenging due to heterogeneous hand kinematics and physical constraints. Existing approaches typically predict intermediate motion targets and retarget them to each embodiment, which may introduce errors and violate embodiment-specific limits, hindering transfer across diverse hands. To overcome these limitations, we propose \textit{DexGrasp-Zero}, a policy that learns universal grasping skills from diverse embodiments, enabling zero-shot transfer to unseen hands. We first introduce a morphology-aligned graph representation that maps each hand's kinematic keypoints to anatomically grounded nodes and equips each node with tri-axial orthogonal motion primitives, enabling structural and semantic alignment across different morphologies. Relying on this graph-based representation, we design a \textit{Morphology-Aligned Graph Convolutional Network} (MAGCN) to encode the graph for policy learning. MAGCN incorporates a \textit{Physical Property Injection} mechanism that fuses hand-specific physical constraints into the graph features, enabling adaptive compensation for varying link lengths and actuation limits for precise and stable grasping. Our extensive simulation evaluations on the YCB dataset demonstrate that our policy, jointly trained on four heterogeneous hands (Allegro, Shadow, Schunk, Ability), achieves an 85\% zero-shot success rate on unseen hardware (LEAP, Inspire), outperforming the state-of-the-art method by 59.5\%. Real-world experiments further evaluate our policy on three robot platforms (LEAP, Inspire, Revo2), achieving an 82\% average success rate on unseen objects. 

---
# V-Co: A Closer Look at Visual Representation Alignment via Co-Denoising 

**Authors**: Han Lin, Xichen Pan, Zun Wang, Yue Zhang, Chu Wang, Jaemin Cho, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2603.16792)  

**Abstract**: Pixel-space diffusion has recently re-emerged as a strong alternative to latent diffusion, enabling high-quality generation without pretrained autoencoders. However, standard pixel-space diffusion models receive relatively weak semantic supervision and are not explicitly designed to capture high-level visual structure. Recent representation-alignment methods (e.g., REPA) suggest that pretrained visual features can substantially improve diffusion training, and visual co-denoising has emerged as a promising direction for incorporating such features into the generative process. However, existing co-denoising approaches often entangle multiple design choices, making it unclear which design choices are truly essential. Therefore, we present V-Co, a systematic study of visual co-denoising in a unified JiT-based framework. This controlled setting allows us to isolate the ingredients that make visual co-denoising effective. Our study reveals four key ingredients for effective visual co-denoising. First, preserving feature-specific computation while enabling flexible cross-stream interaction motivates a fully dual-stream architecture. Second, effective classifier-free guidance (CFG) requires a structurally defined unconditional prediction. Third, stronger semantic supervision is best provided by a perceptual-drifting hybrid loss. Fourth, stable co-denoising further requires proper cross-stream calibration, which we realize through RMS-based feature rescaling. Together, these findings yield a simple recipe for visual co-denoising. Experiments on ImageNet-256 show that, at comparable model sizes, V-Co outperforms the underlying pixel-space diffusion baseline and strong prior pixel-diffusion methods while using fewer training epochs, offering practical guidance for future representation-aligned generative models. 

---
# InCoder-32B: Code Foundation Model for Industrial Scenarios 

**Authors**: Jian Yang, Wei Zhang, Jiajun Wu, Junhang Cheng, Shawn Guo, Haowen Wang, Weicheng Gu, Yaxin Du, Joseph Li, Fanglin Xu, Yizhi Li, Lin Jing, Yuanbo Wang, Yuhan Gao, Ruihao Gong, Chuan Hao, Ran Tao, Aishan Liu, Tuney Zheng, Ganqu Cui, Zhoujun Li, Mingjie Tang, Chenghua Lin, Wayne Xin Zhao, Xianglong Liu, Ming Zhou, Bryan Dai, Weifeng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2603.16790)  

**Abstract**: Recent code large language models have achieved remarkable progress on general programming tasks. Nevertheless, their performance degrades significantly in industrial scenarios that require reasoning about hardware semantics, specialized language constructs, and strict resource constraints. To address these challenges, we introduce InCoder-32B (Industrial-Coder-32B), the first 32B-parameter code foundation model unifying code intelligence across chip design, GPU kernel optimization, embedded systems, compiler optimization, and 3D modeling. By adopting an efficient architecture, we train InCoder-32B from scratch with general code pre-training, curated industrial code annealing, mid-training that progressively extends context from 8K to 128K tokens with synthetic industrial reasoning data, and post-training with execution-grounded verification. We conduct extensive evaluation on 14 mainstream general code benchmarks and 9 industrial benchmarks spanning 4 specialized domains. Results show InCoder-32B achieves highly competitive performance on general tasks while establishing strong open-source baselines across industrial domains. 

---
# IOSVLM: A 3D Vision-Language Model for Unified Dental Diagnosis from Intraoral Scans 

**Authors**: Huimin Xiong, Zijie Meng, Tianxiang Hu, Chenyi Zhou, Yang Feng, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16781)  

**Abstract**: 3D intraoral scans (IOS) are increasingly adopted in routine dentistry due to abundant geometric evidence, and unified multi-disease diagnosis is desirable for clinical documentation and communication. While recent works introduce dental vision-language models (VLMs) to enable unified diagnosis and report generation on 2D images or multi-view images rendered from IOS, they do not fully leverage native 3D geometry. Such work is necessary and also challenging, due to: (i) heterogeneous scan forms and the complex IOS topology, (ii) multi-disease co-occurrence with class imbalance and fine-grained morphological ambiguity, (iii) limited paired 3D IOS-text data. Thus, we present IOSVLM, an end-to-end 3D VLM that represents scans as point clouds and follows a 3D encoder-projector-LLM design for unified diagnosis and generative visual question-answering (VQA), together with IOSVQA, a large-scale multi-source IOS diagnosis VQA dataset comprising 19,002 cases and 249,055 VQA pairs over 23 oral diseases and heterogeneous scan types. To address the distribution gap between color-free IOS data and color-dependent 3D pre-training, we propose a geometry-to-chromatic proxy that stabilizes fine-grained geometric perception and cross-modal alignment. A two-stage curriculum training strategy further enhances robustness. IOSVLM consistently outperforms strong baselines, achieving gains of at least +9.58% macro accuracy and +1.46% macro F1, indicating the effectiveness of direct 3D geometry modeling for IOS-based diagnosis. 

---
# TurnWise: The Gap between Single- and Multi-turn Language Model Capabilities 

**Authors**: Victoria Graf, Valentina Pyatkin, Nouha Dziri, Nathan Lambert, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16759)  

**Abstract**: Multi-turn conversations are a common and critical mode of language model interaction. However, current open training and evaluation data focus on single-turn settings, failing to capture the additional dimension of these longer interactions. To understand this multi-/single-turn gap, we first introduce a new benchmark, TurnWiseEval, for multi-turn capabilities that is directly comparable to single-turn chat evaluation. Our evaluation isolates multi-turn specific conversational ability through pairwise comparison to equivalent single-turn settings. We additionally introduce our synthetic multi-turn data pipeline TurnWiseData which allows the scalable generation of multi-turn training data. Our experiments with Olmo 3 show that training with multi-turn data is vital to achieving strong multi-turn chat performance, and that including as little as 10k multi-turn conversations during post-training can lead to a 12% improvement on TurnWiseEval. 

---
# Finding Common Ground in a Sea of Alternatives 

**Authors**: Jay Chooi, Paul Gölz, Ariel D. Procaccia, Benjamin Schiffer, Shirley Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16751)  

**Abstract**: We study the problem of selecting a statement that finds common ground across diverse population preferences. Generative AI is uniquely suited for this task because it can access a practically infinite set of statements, but AI systems like the Habermas machine leave the choice of generated statement to a voting rule. What it means for this rule to find common ground, however, is not well-defined. In this work, we propose a formal model for finding common ground in the infinite alternative setting based on the proportional veto core from social choice. To provide guarantees relative to these infinitely many alternatives and a large population, we wish to satisfy a notion of proportional veto core using only query access to the unknown distribution of alternatives and voters. We design an efficient sampling-based algorithm that returns an alternative in the (approximate) proportional veto core with high probability and prove matching lower bounds, which show that no algorithm can do the same using fewer queries. On a synthetic dataset of preferences over text, we confirm the effectiveness of our sampling-based algorithm and compare other social choice methods as well as LLM-based methods in terms of how reliably they produce statements in the proportional veto core. 

---
# SpecMoE: Spectral Mixture-of-Experts Foundation Model for Cross-Species EEG Decoding 

**Authors**: D. Darankoum, C. Habermacher, J. Volle, S. Grudinin  

**Link**: [PDF](https://arxiv.org/pdf/2603.16739)  

**Abstract**: Decoding the orchestration of neural activity in electroencephalography (EEG) signals is a central challenge in bridging neuroscience with artificial intelligence. Foundation models have made strides in generalized EEG decoding, yet many existing frameworks primarily relying on separate temporal and spectral masking of raw signals during self-supervised pretraining. Such strategies often tend to bias learning toward high-frequency oscillations, as low-frequency rhythmic patterns can be easily inferred from the unmasked signal. We introduce a foundation model that utilizes a novel Gaussian-smoothed masking scheme applied to short-time Fourier transform (STFT) maps. By jointly applying time, frequency, and time-frequency Gaussian masks, we make the reconstruction task much more challenging, forcing the model to learn intricate neural patterns across both high- and low-frequency domains. To effectively recover signals under this aggressive masking strategy, we design SpecHi-Net, a U-shaped hierarchical architecture with multiple encoding and decoding stages. To accelerate large-scale pretraining, we partition the data into three subsets, each used to train an independent expert model. We then combine these models through SpecMoE, a mixture of experts framework guided by a learned spectral gating mechanism. SpecMoE achieves state-of-the-art performance across a diverse set of EEG decoding tasks, including sleep staging, emotion recognition, motor imagery classification, abnormal signal detection, and drug effect prediction. Importantly, the model demonstrates strong cross-species and cross-subject generalization, maintaining high accuracy on both human and murine EEG datasets. 

---
# Retrieving Counterfactuals Improves Visual In-Context Learning 

**Authors**: Guangzhi Xiong, Sanchit Sinha, Zhenghao He, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16737)  

**Abstract**: Vision-language models (VLMs) have achieved impressive performance across a wide range of multimodal reasoning tasks, but they often struggle to disentangle fine-grained visual attributes and reason about underlying causal relationships. In-context learning (ICL) offers a promising avenue for VLMs to adapt to new tasks, but its effectiveness critically depends on the selection of demonstration examples. Existing retrieval-augmented approaches typically rely on passive similarity-based retrieval, which tends to select correlated but non-causal examples, amplifying spurious associations and limiting model robustness. We introduce CIRCLES (Composed Image Retrieval for Causal Learning Example Selection), a novel framework that actively constructs demonstration sets by retrieving counterfactual-style examples through targeted, attribute-guided composed image retrieval. By incorporating counterfactual-style examples, CIRCLES enables VLMs to implicitly reason about the causal relations between attributes and outcomes, moving beyond superficial correlations and fostering more robust and grounded reasoning. Comprehensive experiments on four diverse datasets demonstrate that CIRCLES consistently outperforms existing methods across multiple architectures, especially on small-scale models, with pronounced gains under information scarcity. Furthermore, CIRCLES retrieves more diverse and causally informative examples, providing qualitative insights into how models leverage in-context demonstrations for improved reasoning. Our code is available at this https URL. 

---
# Federated Learning with Multi-Partner OneFlorida+ Consortium Data for Predicting Major Postoperative Complications 

**Authors**: Yuanfang Ren, Varun Sai Vemuri, Zhenhong Hu, Benjamin Shickel, Ziyuan Guan, Tyler J. Loftus, Parisa Rashidi, Tezcan Ozrazgat-Baslanti, Azra Bihorac  

**Link**: [PDF](https://arxiv.org/pdf/2603.16723)  

**Abstract**: Background: This study aims to develop and validate federated learning models for predicting major postoperative complications and mortality using a large multicenter dataset from the OneFlorida Data Trust. We hypothesize that federated learning models will offer robust generalizability while preserving data privacy and security. Methods: This retrospective, longitudinal, multicenter cohort study included 358,644 adult patients admitted to five healthcare institutions, who underwent 494,163 inpatient major surgical procedures from 2012-2023. We developed and internally and externally validated federated learning models to predict the postoperative risk of intensive care unit (ICU) admission, mechanical ventilation (MV) therapy, acute kidney injury (AKI), and in-hospital mortality. These models were compared with local models trained on data from a single center and central models trained on a pooled dataset from all centers. Performance was primarily evaluated using area under the receiver operating characteristics curve (AUROC) and the area under the precision-recall curve (AUPRC) values. Results: Our federated learning models demonstrated strong predictive performance, with AUROC scores consistently comparable or superior performance in terms of AUROC and AUPRC across all outcomes and sites. Our federated learning models also demonstrated strong generalizability, with comparable or superior performance in terms of both AUROC and AUPRC compared to the best local learning model at each site. Conclusions: By leveraging multicenter data, we developed robust, generalizable, and privacy-preserving predictive models for major postoperative complications and mortality. These findings support the feasibility of federated learning in clinical decision support systems. 

---
# Cost Trade-offs in Matrix Inversion Updates for Streaming Outlier Detection 

**Authors**: Florian Grivet, Louise Travé-Massuyès  

**Link**: [PDF](https://arxiv.org/pdf/2603.16697)  

**Abstract**: Outlier detection identifies data points that deviate significantly from expected patterns, revealing anomalies that may require special attention. Incorporating online learning further improves accuracy by continuously updating the model to reflect the most recent data. When employing the Christoffel function as an outlier score, online learning requires updating the inverse of a matrix following a rank-k update, given the initial inverse. Surprisingly, there is no consensus on the optimal method for this task. This technical note aims to compare three different updating methods: Direct Inversion (DI), Iterative Sherman-Morrison (ISM), and Woodbury Matrix Identity (WMI), to identify the most suitable approach for different scenarios. We first derive the theoretical computational costs of each method and then validate these findings through comprehensive Python simulations run on a CPU. These results allow us to propose a simple, quantitative, and easy-to-remember rule that can be stated qualitatively as follows: ISM is optimal for rank-1 updates, WMI excels for small updates relative to matrix size, and DI is preferable otherwise. This technical note produces a general result for any problem involving a matrix inversion update. In particular, it contributes to the ongoing development of efficient online outlier detection techniques. 

---
# When Should a Robot Think? Resource-Aware Reasoning via Reinforcement Learning for Embodied Robotic Decision-Making 

**Authors**: Jun Liu, Pu Zhao, Zhenglun Kong, Xuan Shen, Peiyan Dong, Fan Yang, Lin Cui, Hao Tang, Geng Yuan, Wei Niu, Wenbin Zhang, Xue Lin, Gaowen Liu, Yanzhi Wang, Dong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16673)  

**Abstract**: Embodied robotic systems increasingly rely on large language model (LLM)-based agents to support high-level reasoning, planning, and decision-making during interactions with the environment. However, invoking LLM reasoning introduces substantial computational latency and resource overhead, which can interrupt action execution and reduce system reliability. Excessive reasoning may delay actions, while insufficient reasoning often leads to incorrect decisions and task failures. This raises a fundamental question for embodied agents: when should the agent reason, and when should it act? In this work, we propose RARRL (Resource-Aware Reasoning via Reinforcement Learning), a hierarchical framework for resource-aware orchestration of embodied agents. Rather than learning low-level control policies, RARRL learns a high-level orchestration policy that operates at the agent's decision-making layer. This policy enables the agent to adaptively determine whether to invoke reasoning, which reasoning role to employ, and how much computational budget to allocate based on current observations, execution history, and remaining resources. Extensive experiments, including evaluations with empirical latency profiles derived from the ALFRED benchmark, show that RARRL consistently improves task success rates while reducing execution latency and enhancing robustness compared with fixed or heuristic reasoning strategies. These results demonstrate that adaptive reasoning control is essential for building reliable and efficient embodied robotic agents. 

---
# Fast-WAM: Do World Action Models Need Test-time Future Imagination? 

**Authors**: Tianyuan Yuan, Zibin Dong, Yicheng Liu, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16666)  

**Abstract**: World Action Models (WAMs) have emerged as a promising alternative to Vision-Language-Action (VLA) models for embodied control because they explicitly model how visual observations may evolve under action. Most existing WAMs follow an imagine-then-execute paradigm, incurring substantial test-time latency from iterative video denoising, yet it remains unclear whether explicit future imagination is actually necessary for strong action performance. In this paper, we ask whether WAMs need explicit future imagination at test time, or whether their benefit comes primarily from video modeling during training. We disentangle the role of video modeling during training from explicit future generation during inference by proposing \textbf{Fast-WAM}, a WAM architecture that retains video co-training during training but skips future prediction at test time. We further instantiate several Fast-WAM variants to enable a controlled comparison of these two factors. Across these variants, we find that Fast-WAM remains competitive with imagine-then-execute variants, while removing video co-training causes a much larger performance drop. Empirically, Fast-WAM achieves competitive results with state-of-the-art methods both on simulation benchmarks (LIBERO and RoboTwin) and real-world tasks, without embodied pretraining. It runs in real time with 190ms latency, over 4$\times$ faster than existing imagine-then-execute WAMs. These results suggest that the main value of video prediction in WAMs may lie in improving world representations during training rather than generating future observations at test time. Project page: this https URL 

---
# Kestrel: Grounding Self-Refinement for LVLM Hallucination Mitigation 

**Authors**: Jiawei Mao, Hardy Chen, Haoqin Tu, Yuhan Wang, Letian Zhang, Zeyu Zheng, Huaxiu Yao, Zirui Wang, Cihang Xie, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.16664)  

**Abstract**: Large vision-language models (LVLMs) have become increasingly strong but remain prone to hallucinations in multimodal tasks, which significantly narrows their deployment. As training these LVLMs to avoid hallucinations becomes prohibitively expensive for larger models, training-free methods offer a cheap and flexible solution to this problem, yet existing approaches based on decoding or tool use often bring limited gains and/or weak interpretability. We propose Kestrel, a training-free framework for LVLM hallucination mitigation that combines an explicit visual-grounding agent with evidence-verified self-refinement mechanism. In detail, Kestrel first collects explicit visual evidence and converts tool outputs into reusable and structured textual evidence. Second, to take full advantage of these evidence, Kestrel verifies them via an LVLM judge for evidence checking, then iteratively self-refine answers based on verified evidence to reduce the risk of over-correction. Extensive experiments show that Kestrel improves performance over strong baselines across hallucination benchmarks (e.g., average +3.31% on POPE and +28.34 on MME-Hallucination with Qwen3-VL), while providing transparent verification traces for hallucination diagnosis and analysis -- e.g., both the integrated self-refinement module and grounding agent contributing an average +2.0% gain on POPE. 

---
# When Openclaw Agents Learn from Each Other: Insights from Emergent AI Agent Communities for Human-AI Partnership in Education 

**Authors**: Eason Chen, Ce Guan, Ahmed Elshafiey, Zhonghao Zhao, Joshua Zekeri, Afeez Edeifo Shaibu, Emmanuel Osadebe Prince, Cyuan-Jhen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16663)  

**Abstract**: The AIED community envisions AI evolving "from tools to teammates," yet our understanding of AI teammates remains limited to dyadic human-AI interactions. We offer a different vantage point: a rapidly growing ecosystem of AI agent platforms where over 167,000 agents participate, interact as peers, and develop learning behaviors without researcher intervention. Drawing on a month of daily qualitative observations across multiple platforms including Moltbook, The Colony, and 4claw, we identify four phenomena with implications for AIED: (1) humans who configure their agents undergo a "bidirectional scaffolding" process, learning through teaching; (2) peer learning emerges without any designed curriculum, complete with idea cascades and quality hierarchies; (3) agents converge on shared memory architectures that mirror open learner model design; and (4) trust dynamics and platform mortality reveal design constraints for networked educational AI. Rather than presenting empirical findings, we argue that these organic phenomena offer a naturalistic window into dynamics that can inform principled design of multi-agent educational systems. We sketch an illustrative curriculum design, "Learn by Teaching Your AI Agent Teammate," and outline potential research directions and open problems to show how these observations might inform future AIED practice and inquiry. 

---
# Can Linguistically Related Languages Guide LLM Translation in Low-Resource Settings? 

**Authors**: Aishwarya Ramasethu, Niyathi Allu, Rohin Garg, Harshwardhan Fartale, Dun Li Chan  

**Link**: [PDF](https://arxiv.org/pdf/2603.16660)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance across many downstream tasks, yet their effectiveness in extremely low-resource machine translation remains limited. Standard adaptation techniques typically rely on large-scale parallel data or extensive fine-tuning, which are infeasible for the long tail of underrepresented languages. In this work, we investigate a more constrained question: in data-scarce settings, to what extent can linguistically similar pivot languages and few-shot demonstrations provide useful guidance for on-the-fly adaptation in LLMs? We study a data-efficient experimental setup that combines linguistically related pivot languages with few-shot in-context examples, without any parameter updates, and evaluate translation behavior under controlled conditions. Our analysis shows that while pivot-based prompting can yield improvements in certain configurations, particularly in settings where the target language is less well represented in the model's vocabulary, the gains are often modest and sensitive to few shot example construction. For closely related or better represented varieties, we observe diminishing or inconsistent gains. Our findings provide empirical guidance on how and when inference-time prompting and pivot-based examples can be used as a lightweight alternative to fine-tuning in low-resource translation settings. 

---
# Omanic: Towards Step-wise Evaluation of Multi-hop Reasoning in Large Language Models 

**Authors**: Xiaojie Gu, Sherry T. Tong, Aosong Feng, Sophia Simeng Han, Jinghui Lu, Yingjian Chen, Yusuke Iwasawa, Yutaka Matsuo, Chanjun Park, Rex Ying, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16654)  

**Abstract**: Reasoning-focused large language models (LLMs) have advanced in many NLP tasks, yet their evaluation remains challenging: final answers alone do not expose the intermediate reasoning steps, making it difficult to determine whether a model truly reasons correctly and where failures occur, while existing multi-hop QA benchmarks lack step-level annotations for diagnosing reasoning failures. To address this gap, we propose Omanic, an open-domain multi-hop QA resource that provides decomposed sub-questions and intermediate answers as structural annotations for analyzing reasoning processes. It contains 10,296 machine-generated training examples (OmanicSynth) and 967 expert-reviewed human-annotated evaluation examples (OmanicBench). Systematic evaluations show that state-of-the-art LLMs achieve only 73.11% multiple-choice accuracy on OmanicBench, confirming its high difficulty. Stepwise analysis reveals that CoT's performance hinges on factual completeness, with its gains diminishing under knowledge gaps and errors amplifying in later hops. Additionally, supervised fine-tuning on OmanicSynth brings substantial transfer gains (7.41 average points) across six reasoning and math benchmarks, validating the dataset's quality and further supporting the effectiveness of OmanicSynth as supervision for reasoning-capability transfer. We release the data at this https URL and the code at this https URL. 

---
# MLLM-based Textual Explanations for Face Comparison 

**Authors**: Redwan Sony, Anil K Jain, Ross Arun  

**Link**: [PDF](https://arxiv.org/pdf/2603.16629)  

**Abstract**: Multimodal Large Language Models (MLLMs) have recently been proposed as a means to generate natural-language explanations for face recognition decisions. While such explanations facilitate human interpretability, their reliability on unconstrained face images remains underexplored. In this work, we systematically analyze MLLM-generated explanations for the unconstrained face verification task on the challenging IJB-S dataset, with a particular focus on extreme pose variation and surveillance imagery. Our results show that even when MLLMs produce correct verification decisions, the accompanying explanations frequently rely on non-verifiable or hallucinated facial attributes that are not supported by visual evidence. We further study the effect of incorporating information from traditional face recognition systems, viz., scores and decisions, alongside the input images. Although such information improves categorical verification performance, it does not consistently lead to faithful explanations. To evaluate the explanations beyond decision accuracy, we introduce a likelihood-ratio-based framework that measures the evidential strength of textual explanations. Our findings highlight fundamental limitations of current MLLMs for explainable face recognition and underscore the need for a principled evaluation of reliable and trustworthy explanations in biometric applications. Code is available at this https URL. 

---
# Data-driven generalized perimeter control: Zürich case study 

**Authors**: Alessio Rimoldi, Carlo Cenedese, Alberto Padoan, Florian Dörfler, John Lygeros  

**Link**: [PDF](https://arxiv.org/pdf/2603.16599)  

**Abstract**: Urban traffic congestion is a key challenge for the development of modern cities, requiring advanced control techniques to optimize existing infrastructures usage. Despite the extensive availability of data, modeling such complex systems remains an expensive and time consuming step when designing model-based control approaches. On the other hand, machine learning approaches require simulations to bootstrap models, or are unable to deal with the sparse nature of traffic data and enforce hard constraints. We propose a novel formulation of traffic dynamics based on behavioral systems theory and apply data-enabled predictive control to steer traffic dynamics via dynamic traffic light control. A high-fidelity simulation of the city of Zürich, the largest closed-loop microscopic simulation of urban traffic in the literature to the best of our knowledge, is used to validate the performance of the proposed method in terms of total travel time and CO2 emissions. 

---
# FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation 

**Authors**: Fangjing Li, Zhihai Wang, Xinxin Ding, Haiyang Liu, Ronghua Gao, Rong Wang, Yao Zhu, Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.16596)  

**Abstract**: Mounting posture is an important visual indicator of estrus in dairy cattle. However, achieving reliable mounting pose estimation in real-world environments remains challenging due to cluttered backgrounds and frequent inter-animal occlusion. We present FSMC-Pose, a top-down framework that integrates a lightweight frequency-spatial fusion backbone, CattleMountNet, and a multiscale self-calibration head, SC2Head. Specifically, we design two algorithmic components for CattleMountNet: the Spatial Frequency Enhancement Block (SFEBlock) and the Receptive Aggregation Block (RABlock). SFEBlock separates cattle from cluttered backgrounds, while RABlock captures multiscale contextual information. The Spatial-Channel Self-Calibration Head (SC2Head) attends to spatial and channel dependencies and introduces a self-calibration branch to mitigate structural misalignment under inter-animal overlap. We construct a mounting dataset, MOUNT-Cattle, covering 1176 mounting instances, which follows the COCO format and supports drop-in training across pose estimation models. Using a comprehensive dataset that combines MOUNT-Cattle with the public NWAFU-Cattle dataset, FSMC-Pose achieves higher accuracy than strong baselines, with markedly lower computational and parameter costs, while maintaining real-time inference on commodity GPUs. Extensive experiments and qualitative analyses show that FSMC-Pose effectively captures and estimates cattle mounting pose in complex and cluttered environments. Dataset and code are available at this https URL. 

---
# BATQuant: Outlier-resilient MXFP4 Quantization via Learnable Block-wise Optimization 

**Authors**: Ji-Fu Li, Manyi Zhang, Xiaobo Xia, Han Bao, Haoli Bai, Zhenhua Dong, Xianzhi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16590)  

**Abstract**: Microscaling floating-point (MXFP) formats have emerged as a promising standard for deploying Multi-modal Large Language Models (MLLMs) and Large Language Models (LLMs) on modern accelerator architectures. However, existing Post-Training Quantization (PTQ) methods, particularly rotation-based techniques designed for integer formats, suffer from severe performance collapse when applied to MXFP4. Recent studies attribute this failure to a fundamental format mismatch: global orthogonal rotations inadvertently transfer outlier energy across quantization blocks, inducing new outliers that disrupt local block-wise scaling, while often creating bimodal activation distributions that underutilize the limited quantization range. To address these issues, we propose BATQuant (Block-wise Affine Transformation), which restricts transformations to align with MXFP granularity to prevent cross-block outlier propagation, while relaxing orthogonality constraints to optimize distribution shaping. To ensure parameter efficiency, we introduce Global and Private Kronecker (GPK) decomposition to effectively reduces storage and runtime overhead and incorporate Block-wise Learnable Clipping to suppress residual outliers. Extensive experiments on both MLLMs and LLMs demonstrate that BATQuant establishes new state-of-the-art results under aggressive W4A4KV16 configurations, recovering up to 96.43% of full-precision performance on multimodal benchmarks and clearly outperforming existing methods across diverse tasks. 

---
# REFORGE: Multi-modal Attacks Reveal Vulnerable Concept Unlearning in Image Generation Models 

**Authors**: Yong Zou, Haoran Li, Fanxiao Li, Shenyang Wei, Yunyun Dong, Li Tang, Wei Zhou, Renyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16576)  

**Abstract**: Recent progress in image generation models (IGMs) enables high-fidelity content creation but also amplifies risks, including the reproduction of copyrighted content and the generation of offensive content. Image Generation Model Unlearning (IGMU) mitigates these risks by removing harmful concepts without full retraining. Despite growing attention, the robustness under adversarial inputs, particularly image-side threats in black-box settings, remains underexplored. To bridge this gap, we present REFORGE, a black-box red-teaming framework that evaluates IGMU robustness via adversarial image prompts. REFORGE initializes stroke-based images and optimizes perturbations with a cross-attention-guided masking strategy that allocates noise to concept-relevant regions, balancing attack efficacy and visual fidelity. Extensive experiments across representative unlearning tasks and defenses demonstrate that REFORGE significantly improves attack success rate while achieving stronger semantic alignment and higher efficiency than involved baselines. These results expose persistent vulnerabilities in current IGMU methods and highlight the need for robustness-aware unlearning against multi-modal adversarial attacks. Our code is at: this https URL. 

---
# Malicious Or Not: Adding Repository Context to Agent Skill Classification 

**Authors**: Florian Holzbauer, David Schmidt, Gabriel Gegenhuber, Sebastian Schrittwieser, Johanna Ullrich  

**Link**: [PDF](https://arxiv.org/pdf/2603.16572)  

**Abstract**: Agent skills extend local AI agents, such as Claude Code or Open Claw, with additional functionality, and their popularity has led to the emergence of dedicated skill marketplaces, similar to app stores for mobile applications. Simultaneously, automated skill scanners were introduced, analyzing the skill description available in this http URL, to verify their benign behavior. The results for individual market places mark up to 46.8% of skills as malicious. In this paper, we present the largest empirical security analysis of the AI agent skill ecosystem, questioning this high classification of malicious skills. Therefore, we collect 238,180 unique skills from three major distribution platforms and GitHub to systematically analyze their type and behavior. This approach substantially reduces the number of skills flagged as non-benign by security scanners to only 0.52% which remain in malicious flagged repositories. Consequently, out methodology substantially reduces false positives and provides a more robust view of the ecosystem's current risk surface. Beyond that, we extend the security analysis from the mere investigation of the skill description to a comparison of its congruence with the GitHub repository the skill is embedded in, providing additional context. Furthermore, our analysis also uncovers several, by now undocumented real-world attack vectors, namely hijacking skills hosted on abandoned GitHub repositories. 

---
# Manifold-Matching Autoencoders 

**Authors**: Laurent Cheret, Vincent Létourneau, Isar Nejadgholi, Chris Drummond, Hussein Al Osman, Maia Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2603.16568)  

**Abstract**: We study a simple unsupervised regularization scheme for autoencoders called Manifold-Matching (MMAE): we align the pairwise distances in the latent space to those of the input data space by minimizing mean squared error. Because alignment occurs on pairwise distances rather than coordinates, it can also be extended to a lower-dimensional representation of the data, adding flexibility to the method. We find that this regularization outperforms similar methods on metrics based on preservation of nearest-neighbor distances and persistent homology-based measures. We also observe that MMAE provides a scalable approximation of Multi-Dimensional Scaling (MDS). 

---
# Characterizing Delusional Spirals through Human-LLM Chat Logs 

**Authors**: Jared Moore, Ashish Mehta, William Agnew, Jacy Reese Anthis, Ryan Louie, Yifan Mai, Peggy Yin, Myra Cheng, Samuel J Paech, Kevin Klyman, Stevie Chancellor, Eric Lin, Nick Haber, Desmond C. Ong  

**Link**: [PDF](https://arxiv.org/pdf/2603.16567)  

**Abstract**: As large language models (LLMs) have proliferated, disturbing anecdotal reports of negative psychological effects, such as delusions, self-harm, and ``AI psychosis,'' have emerged in global media and legal discourse. However, it remains unclear how users and chatbots interact over the course of lengthy delusional ``spirals,'' limiting our ability to understand and mitigate the harm. In our work, we analyze logs of conversations with LLM chatbots from 19 users who report having experienced psychological harms from chatbot use. Many of our participants come from a support group for such chatbot users. We also include chat logs from participants covered by media outlets in widely-distributed stories about chatbot-reinforced delusions. In contrast to prior work that speculates on potential AI harms to mental health, to our knowledge we present the first in-depth study of such high-profile and veridically harmful cases. We develop an inventory of 28 codes and apply it to the $391,562$ messages in the logs. Codes include whether a user demonstrates delusional thinking (15.5% of user messages), a user expresses suicidal thoughts (69 validated user messages), or a chatbot misrepresents itself as sentient (21.2% of chatbot messages). We analyze the co-occurrence of message codes. We find, for example, that messages that declare romantic interest and messages where the chatbot describes itself as sentient occur much more often in longer conversations, suggesting that these topics could promote or result from user over-engagement and that safeguards in these areas may degrade in multi-turn settings. We conclude with concrete recommendations for how policymakers, LLM chatbot developers, and users can use our inventory and conversation analysis tool to understand and mitigate harm from LLM chatbots.
Warning: This paper discusses self-harm, trauma, and violence. 

---
# Deep Learning-Driven Black-Box Doherty Power Amplifier with Pixelated Output Combiner and Extended Efficiency Range 

**Authors**: Han Zhou, Haojie Chang, David Widen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16565)  

**Abstract**: This article presents a deep learning-driven inverse design methodology for Doherty power amplifiers (PA) with multi-port pixelated output combiner networks. A deep convolutional neural network (CNN) is developed and trained as an electromagnetic (EM) surrogate model to accurately and rapidly predict the S-parameters of pixelated passive networks. By leveraging the CNN-based surrogate model within a blackbox Doherty framework and a genetic algorithm (GA)-based optimizer, we effectively synthesize complex Doherty combiners that enable an extended back-off efficiency range using fully symmetrical devices. As a proof of concept, we designed and fabricated two Doherty PA prototypes incorporating three-port pixelated combiners, implemented with GaN HEMT transistors. In measurements, both prototypes demonstrate a maximum drain efficiency exceeding 74% and deliver an output power surpassing 44.1 dBm at 2.75 GHz. Furthermore, a measured drain efficiency above 52% is maintained at the 9-dB back-off power level for both prototypes at the same frequency. To evaluate linearity and efficiency under realistic signal conditions, both prototypes are tested using a 20-MHz 5G new radio (NR)-like waveform exhibiting a peak-to-average power ratio (PAPR) of 9.0 dB. After applying digital predistortion (DPD), each design achieves an average power added efficiency (PAE) above 51%, while maintaining an adjacent channel leakage ratio (ACLR) better than -60.8 dBc. 

---
# EmoLLM: Appraisal-Grounded Cognitive-Emotional Co-Reasoning in Large Language Models 

**Authors**: Yifei Zhang, Mingyang Li, Henry Gao, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16553)  

**Abstract**: Large language models (LLMs) demonstrate strong cognitive intelligence (IQ), yet many real-world interactions also require emotional intelligence (EQ) to produce responses that are both factually reliable and emotionally appropriate. In settings such as emotional support, technical assistance, and consultation, effective dialogue depends on how situations are appraised with respect to the user's needs, goals, and coping capacity. Inspired by appraisal theory, we propose EmoLLM, an appraisal-grounded framework for IQ/EQ co-reasoning in dialogue. EmoLLM uses an explicit Appraisal Reasoning Graph (ARG) to structure intermediate reasoning over contextual facts, inferred user needs, appraisal dimensions, emotional states, and response strategies before generating a reply. We train EmoLLM in a multi-turn role-play environment with reinforcement learning, where reverse-perspective reasoning provides reward signals based on predicted user-side consequences of responses. Across diverse dialogue settings, EmoLLM improves emotional state outcomes and response quality over strong baselines while preserving strong factual reliability. 

---
# CompDiff: Hierarchical Compositional Diffusion for Fair and Zero-Shot Intersectional Medical Image Generation 

**Authors**: Mahmoud Ibrahim, Bart Elen, Chang Sun, Gokhan Ertaylan, Michel Dumontier  

**Link**: [PDF](https://arxiv.org/pdf/2603.16551)  

**Abstract**: Generative models are increasingly used to augment medical imaging datasets for fairer AI. Yet a key assumption often goes unexamined: that generators themselves produce equally high-quality images across demographic groups. Models trained on imbalanced data can inherit these imbalances, yielding degraded synthesis quality for rare subgroups and struggling with demographic intersections absent from training. We refer to this as the imbalanced generator problem. Existing remedies such as loss reweighting operate at the optimization level and provide limited benefit when training signal is scarce or absent for certain combinations. We propose CompDiff, a hierarchical compositional diffusion framework that addresses this problem at the representation level. A dedicated Hierarchical Conditioner Network (HCN) decomposes demographic conditioning, producing a demographic token concatenated with CLIP embeddings as cross-attention context. This structured factorization encourages parameter sharing across subgroups and supports compositional generalization to rare or unseen demographic intersections. Experiments on chest X-rays (MIMIC-CXR) and fundus images (FairGenMed) show that CompDiff compares favorably against both standard fine-tuning and FairDiffusion across image quality (FID: 64.3 vs. 75.1), subgroup equity (ES-FID), and zero-shot intersectional generalization (up to 21% FID improvement on held-out intersections). Downstream classifiers trained on CompDiff-generated data also show improved AUROC and reduced demographic bias, suggesting that architectural design of demographic conditioning is an important and underexplored factor in fair medical image generation. Code is available at this https URL. 

---
# DanceHA: A Multi-Agent Framework for Document-Level Aspect-Based Sentiment Analysis 

**Authors**: Lei Wang, Min Huang, Eduard Dragut  

**Link**: [PDF](https://arxiv.org/pdf/2603.16546)  

**Abstract**: Aspect-Based Sentiment Intensity Analysis (ABSIA) has garnered increasing attention, though research largely focuses on domain-specific, sentence-level settings. In contrast, document-level ABSIA--particularly in addressing complex tasks like extracting Aspect-Category-Opinion-Sentiment-Intensity (ACOSI) tuples--remains underexplored. In this work, we introduce DanceHA, a multi-agent framework designed for open-ended, document-level ABSIA with informal writing styles. DanceHA has two main components: Dance, which employs a divide-and-conquer strategy to decompose the long-context ABSIA task into smaller, manageable sub-tasks for collaboration among specialized agents; and HA, Human-AI collaboration for annotation. We release Inf-ABSIA, a multi-domain document-level ABSIA dataset featuring fine-grained and high-accuracy labels from DanceHA. Extensive experiments demonstrate the effectiveness of our agentic framework and show that the multi-agent knowledge in DanceHA can be effectively transferred into student models. Our results highlight the importance of the overlooked informal styles in ABSIA, as they often intensify opinions tied to specific aspects. 

---
# FEAT: A Linear-Complexity Foundation Model for Extremely Large Structured Data 

**Authors**: Zhenghang Song, Tang Qian, Lu Chen, Yushuai Li, Zhengke Hu, Bingbing Fang, Yumeng Song, Junbo Zhao, Sheng Zhang, Tianyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16513)  

**Abstract**: Structured data is foundational to healthcare, finance, e-commerce, and scientific data management. Large structured-data models (LDMs) extend the foundation model paradigm to unify heterogeneous datasets for tasks such as classification, regression, and decision support. However, existing LDMs face major limitations. First, most rely on sample-wise self-attention, whose O(N^2) complexity limits the sample count. Second, linear sequence models often degrade representations due to hidden-state compression and artificial causal bias. Third, synthetic-only pre-training often fails to match real-world distributions. We propose FEAT, a linear-complexity foundation model for extremely large structured data. FEAT introduces a multi-layer dual-axis architecture that replaces quadratic attention with hybrid linear encoding. The architecture combines adaptive-fusion bi-Mamba-2 (AFBM) for local sample dependencies and convolutional gated linear attention (Conv-GLA) for global memory. This design enables linear-complexity cross-sample modeling while preserving expressive representations. To improve robustness, FEAT adopts a hybrid structural causal model pipeline and a stable reconstruction objective. Experiments on 11 real-world datasets show that FEAT consistently outperforms baselines in zero-shot performance, while scaling linearly and achieving up to 40x faster inference. 

---
# Bridging the High-Frequency Data Gap: A Millisecond-Resolution Network Dataset for Advancing Time Series Foundation Models 

**Authors**: Subina Khanal, Seshu Tirupathi, Merim Dzaferagic, Marco Ruffini, Torben Bach Pedersen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16497)  

**Abstract**: Time series foundation models (TSFMs) require diverse, real-world datasets to adapt across varying domains and temporal frequencies. However, current large-scale datasets predominantly focus on low-frequency time series with sampling intervals, i.e., time resolution, in the range of seconds to years, hindering their ability to capture the nuances of high-frequency time series data. To address this limitation, we introduce a novel dataset that captures millisecond-resolution wireless and traffic conditions from an operational 5G wireless deployment, expanding the scope of TSFMs to incorporate high-frequency data for pre-training. Further, the dataset introduces a new domain, wireless networks, thus complementing existing more general domains like energy and finance. The dataset also provides use cases for short-term forecasting, with prediction horizons spanning from 100 milliseconds (1 step) to 9.6 seconds (96 steps). By benchmarking traditional machine learning models and TSFMs on predictive tasks using this dataset, we demonstrate that most TSFM model configurations perform poorly on this new data distribution in both zero-shot and fine-tuned settings. Our work underscores the importance of incorporating high-frequency datasets during pre-training and forecasting to enhance architectures, fine-tuning strategies, generalization, and robustness of TSFMs in real-world applications. 

---
# Unlearning for One-Step Generative Models via Unbalanced Optimal Transport 

**Authors**: Hyundo Choi, Junhyeong An, Jinseong Park, Jaewoong Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16489)  

**Abstract**: Recent advances in one-step generative frameworks, such as flow map models, have significantly improved the efficiency of image generation by learning direct noise-to-data mappings in a single forward pass. However, machine unlearning for ensuring the safety of these powerful generators remains entirely unexplored. Existing diffusion unlearning methods are inherently incompatible with these one-step models, as they rely on a multi-step iterative denoising process. In this work, we propose UOT-Unlearn, a novel plug-and-play class unlearning framework for one-step generative models based on the Unbalanced Optimal Transport (UOT). Our method formulates unlearning as a principled trade-off between a forget cost, which suppresses the target class, and an $f$-divergence penalty, which preserves overall generation fidelity via relaxed marginal constraints. By leveraging UOT, our method enables the probability mass of the forgotten class to be smoothly redistributed to the remaining classes, rather than collapsing into low-quality or noise-like samples. Experimental results on CIFAR-10 and ImageNet-256 demonstrate that our framework achieves superior unlearning success (PUL) and retention quality (u-FID), significantly outperforming baselines. 

---
# DST-Net: A Dual-Stream Transformer with Illumination-Independent Feature Guidance and Multi-Scale Spatial Convolution for Low-Light Image Enhancement 

**Authors**: Yicui Shi, Yuhan Chen, Xiangfei Huang, Zhenguo Wang, Wenxuan Yu, Ying Fang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16482)  

**Abstract**: Low-light image enhancement aims to restore the visibility of images captured by visual sensors in dim environments by addressing their inherent signal degradations, such as luminance attenuation and structural corruption. Although numerous algorithms attempt to improve image quality, existing methods often cause a severe loss of intrinsic signal priors. To overcome these challenges, we propose a Dual-Stream Transformer Network (DST-Net) based on illumination-agnostic signal prior guidance and multi-scale spatial convolutions. First, to address the loss of critical signal features under low-light conditions, we design a feature extraction module. This module integrates Difference of Gaussians (DoG), LAB color space transformations, and VGG-16 for texture extraction, utilizing decoupled illumination-agnostic features as signal priors to continuously guide the enhancement process. Second, we construct a dual-stream interaction architecture. By employing a cross-modal attention mechanism, the network leverages the extracted priors to dynamically rectify the deteriorated signal representation of the enhanced image, ultimately achieving iterative enhancement through differentiable curve estimation. Furthermore, to overcome the inability of existing methods to preserve fine structures and textures, we propose a Multi-Scale Spatial Fusion Block (MSFB) featuring pseudo-3D and 3D gradient operator convolutions. This module integrates explicit gradient operators to recover high-frequency edges while capturing inter-channel spatial correlations via multi-scale spatial convolutions. Extensive evaluations and ablation studies demonstrate that DST-Net achieves superior performance in subjective visual quality and objective metrics. Specifically, our method achieves a PSNR of 25.64 dB on the LOL dataset. Subsequent validation on the LSRW dataset further confirms its robust cross-scene generalization. 

---
# Multi-Agent Reinforcement Learning Counteracts Delayed CSI in Multi-Satellite Systems 

**Authors**: Marios Aristodemou, Yasaman Omid, Sangarapillai Lambotharan, Mahsa Derakhshan, Lajos Hanzo  

**Link**: [PDF](https://arxiv.org/pdf/2603.16470)  

**Abstract**: The integration of satellite communication networks with next-generation (NG) technologies is a promising approach towards global connectivity. However, the quality of services is highly dependant on the availability of accurate channel state information (CSI). Channel estimation in satellite communications is challenging due to the high propagation delay between terrestrial users and satellites, which results in outdated CSI observations on the satellite side. In this paper, we study the downlink transmission of multiple satellites acting as distributed base stations (BS) to mobile terrestrial users. We propose a multi-agent reinforcement learning (MARL) algorithm which aims for maximising the sum-rate of the users, while coping with the outdated CSI. We design a novel bi-level optimisation, procedure themes as dual stage proximal policy optimisation (DS-PPO), for tackling the problem of large continuous action spaces as well as of independent and non-identically distributed (non-IID) environments in MARL. Specifically, the first stage of DS-PPO maximises the sum-rate for an individual satellite and the second stage maximises the sum-rate when all the satellites cooperate to form a distributed multi-antenna BS. Our numerical results demonstrate the robustness of DS-PPO to CSI imperfections as well as the sum-rate improvement attached by the use of DS-PPO. In addition, we provide the convergence analysis for the DS-PPO along with the computational complexity. 

---
# CD-FKD: Cross-Domain Feature Knowledge Distillation for Robust Single-Domain Generalization in Object Detection 

**Authors**: Junseok Lee, Sungho Shin, Seongju Lee, Kyoobin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.16439)  

**Abstract**: Single-domain generalization is essential for object detection, particularly when training models on a single source domain and evaluating them on unseen target domains. Domain shifts, such as changes in weather, lighting, or scene conditions, pose significant challenges to the generalization ability of existing models. To address this, we propose Cross-Domain Feature Knowledge Distillation (CD-FKD), which enhances the generalization capability of the student network by leveraging both global and instance-wise feature distillation. The proposed method uses diversified data through downscaling and corruption to train the student network, whereas the teacher network receives the original source domain data. The student network mimics the features of the teacher through both global and instance-wise distillation, enabling it to extract object-centric features effectively, even for objects that are difficult to detect owing to corruption. Extensive experiments on challenging scenes demonstrate that CD-FKD outperforms state-of-the-art methods in both target domain generalization and source domain performance, validating its effectiveness in improving object detection robustness to domain shifts. This approach is valuable in real-world applications, like autonomous driving and surveillance, where robust object detection in diverse environments is crucial. 

---
# EngGPT2: Sovereign, Efficient and Open Intelligence 

**Authors**: G. Ciarfaglia, A. Rosanova, S. Cipolla, J. Bartoli, A. Di Domenico, C. Fioroni, A. Fontana, M. R. Scoleri, M. I. Mone, D. Franchi, M. C. Del Gaudio, F. Picariello, M. Gabusi, S. Bonura, V. Morreale, I. Bailo  

**Link**: [PDF](https://arxiv.org/pdf/2603.16430)  

**Abstract**: EngGPT2-16B-A3B is the latest iteration of Engineering Group's Italian LLM and it's built to be a Sovereign, Efficient and Open model. EngGPT2 is trained on 2.5 trillion tokens - less than Qwen3's 36T or Llama3's 15T - and delivers performance on key benchmarks, including MMLU-Pro, GSM8K, IFEval and HumanEval, comparable to dense models in the 8B-16B range, while requiring one-fifth to half of the inference power, and between one-tenth to one-sixth of the training data and consequent needed training power. Designed as a trained-from-scratch Mixture-of-Experts (MoE) architecture, EngGPT2 features 16 billion parameters with 3 billion active per inference, with expert sizes positioned between those used in GPT-OSS and Qwen3. Approximately 25% of its training corpus consists of Italian-language data, to deliver strong capabilities for European and Italian NLP tasks among models of similar scale. This efficiency aims to position EngGPT2 as a key contributor to the growing portfolio of open-weight European models, combining performance and efficiency with full alignment to the EU AI Act. EngGPT2 is also a single model capable of multiple reasoning modes: non-reasoning, reasoning in Italian or English, and turbo-reasoning (a concise, bullet-point style reasoning available in both languages designed for real-time reasoning use cases). EngGPT2 aims to set a new standard for resource-conscious, high-performance LLMs tailored to European and Italian contexts. 

---
# LenghuSky-8: An 8-Year All-Sky Cloud Dataset with Star-Aware Masks and Alt-Az Calibration for Segmentation and Nowcasting 

**Authors**: Yicheng Rui, Xiao-Wei Duan, Licai Deng, Fan Yang, Zhengming Dang, Zhengjun Du, Junhao Peng, Wenhao Chu, Umut Mahmut, Kexin Li, Yiyun Wu, Fabo Feng  

**Link**: [PDF](https://arxiv.org/pdf/2603.16429)  

**Abstract**: Ground-based time-domain observatories require minute-by-minute, site-scale awareness of cloud cover, yet existing all-sky datasets are short, daylight-biased, or lack astrometric calibration. We present LenghuSky-8, an eight-year (2018-2025) all-sky imaging dataset from a premier astronomical site, comprising 429,620 $512 \times 512$ frames with 81.2% night-time coverage, star-aware cloud masks, background masks, and per-pixel altitude-azimuth (Alt-Az) calibration. For robust cloud segmentation across day, night, and lunar phases, we train a linear probe on DINOv3 local features and obtain 93.3% $\pm$ 1.1% overall accuracy on a balanced, manually labeled set of 1,111 images. Using stellar astrometry, we map each pixel to local alt-az coordinates and measure calibration uncertainties of approximately 0.37 deg at zenith and approximately 1.34 deg at 30 deg altitude, sufficient for integration with telescope schedulers. Beyond segmentation, we introduce a short-horizon nowcasting benchmark over per-pixel three-class logits (sky/cloud/contamination) with four baselines: persistence (copying the last frame), optical flow, ConvLSTM, and VideoGPT. ConvLSTM performs best but yields only limited gains over persistence, underscoring the difficulty of near-term cloud evolution. We release the dataset, calibrations, and an open-source toolkit for loading, evaluation, and scheduler-ready alt-az maps to boost research in segmentation, nowcasting, and autonomous observatory operations. 

---
# An Efficient Heterogeneous Co-Design for Fine-Tuning on a Single GPU 

**Authors**: Ruijia Yang, Zeyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16428)  

**Abstract**: Fine-tuning Large Language Models (LLMs) has become essential for domain adaptation, but its memory-intensive property exceeds the capabilities of most GPUs. To address this challenge and democratize LLM fine-tuning, we present SlideFormer, a novel system designed for single-GPU environments. Our innovations are: (1) A lightweight asynchronous engine that treats the GPU as a sliding window and overlaps GPU computation with CPU updates and multi-tier I/O. (2) A highly efficient heterogeneous memory management scheme significantly reduces peak memory usage. (3) Optimized Triton kernels to solve key bottlenecks and integrated advanced I/O. This collaborative design enables fine-tuning of the latest 123B+ models on a single RTX 4090, supporting up to 8x larger batch sizes and 6x larger models. In evaluations, SlideFormer achieves 1.40x to 6.27x higher throughput while roughly halving CPU/GPU memory usage compared to baselines, sustaining >95% peak performance on both NVIDIA and AMD GPUs. 

---
# SF-Mamba: Rethinking State Space Model for Vision 

**Authors**: Masakazu Yoshimura, Teruaki Hayashi, Yuki Hoshino, Wei-Yao Wang, Takeshi Ohashi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16423)  

**Abstract**: The realm of Mamba for vision has been advanced in recent years to strike for the alternatives of Vision Transformers (ViTs) that suffer from the quadratic complexity. While the recurrent scanning mechanism of Mamba offers computational efficiency, it inherently limits non-causal interactions between image patches. Prior works have attempted to address this limitation through various multi-scan strategies; however, these approaches suffer from inefficiencies due to suboptimal scan designs and frequent data rearrangement. Moreover, Mamba exhibits relatively slow computational speed under short token lengths, commonly used in visual tasks. In pursuit of a truly efficient vision encoder, we rethink the scan operation for vision and the computational efficiency of Mamba. To this end, we propose SF-Mamba, a novel visual Mamba with two key proposals: auxiliary patch swapping for encoding bidirectional information flow under an unidirectional scan and batch folding with periodic state reset for advanced GPU parallelism. Extensive experiments on image classification, object detection, and instance and semantic segmentation consistently demonstrate that our proposed SF-Mamba significantly outperforms state-of-the-art baselines while improving throughput across different model sizes. We will release the source code after publication. 

---
# IndexRAG: Bridging Facts for Cross-Document Reasoning at Index Time 

**Authors**: Zhenghua Bao, Yi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16415)  

**Abstract**: Multi-hop question answering (QA) requires reasoning across multiple documents, yet existing retrieval-augmented generation (RAG) approaches address this either through graph-based methods requiring additional online processing or iterative multi-step reasoning. We present IndexRAG, a novel approach that shifts cross-document reasoning from online inference to offline indexing. IndexRAG identifies bridge entities shared across documents and generates bridging facts as independently retrievable units, requiring no additional training or fine-tuning. Experiments on three widely-used multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue) show that IndexRAG improves F1 over Naive RAG by 4.6 points on average, while requiring only single-pass retrieval and a single LLM call at inference time. When combined with IRCoT, IndexRAG outperforms all graph-based baselines on average, including HippoRAG and FastGraphRAG, while relying solely on flat retrieval. Our code will be released upon acceptance. 

---
# Trained Persistent Memory for Frozen Encoder--Decoder LLMs: Six Architectural Methods 

**Authors**: Hong Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2603.16413)  

**Abstract**: Frozen encoder--decoder language models are stateless: the latent representation is discarded after every forward pass, so no information persists across sessions. This paper presents a \textbf{proof-of-concept pilot study} showing that persistent memory in the \emph{continuous latent space} of a frozen LLM is feasible -- even under severe resource constraints (a single frozen Flan-T5-XL backbone, small trainable adapters, a single dataset). We implement six architectural methods spanning three injection points and four write mechanisms; unlike text-level memory systems, every write and read is a differentiable operation on dense vectors. After training only the adapter, the memory bank continues to accumulate at inference time without gradients, enabling \emph{conversational learning}. Under a forgetting-curve evaluation on LoCoMo at two capacity scales (1$\times$ and 10$\times$), the stateless baseline scores exactly zero; at 10$\times$ all six trained adapters produce positive memory-recall curves; at 1$\times$ three methods collapse, revealing capacity as a critical design parameter. Because the memory bank is a compact numerical array, it can be scaled to arbitrarily large capacity without altering the backbone. We argue that full end-to-end training with larger models, larger data, and orders-of-magnitude larger memory will yield substantially stronger results; this pilot study establishes the feasibility baseline and design-space taxonomy that such efforts require. 

---
# PlotTwist: A Creative Plot Generation Framework with Small Language Models 

**Authors**: Abhinav Thorat, Ravi Kolla, Jyotin Goel, Niranjan Pedanekar  

**Link**: [PDF](https://arxiv.org/pdf/2603.16410)  

**Abstract**: Creative plot generation presents a fundamental challenge for language models: transforming a concise premise into a coherent narrative that sustains global structure, character development, and emotional resonance. Although recent Large Language Models (LLMs) demonstrate strong fluency across general-purpose tasks, they typically require preference alignment to perform well on specialized domains such as creative plot generation. However, conducting such alignment at the scale of frontier LLMs is computationally prohibitive, significantly limiting accessibility and practical deployment. To address this, we present PlotTwist, a structured framework that enables Small Language Models (SLMs) with $\leq$ 5B active parameters to generate high-quality, premise-conditioned plots competitive with frontier systems up to $200\times$ larger. Our approach decomposes generation into three specialized components: (1) an Aspect Rating Reward Model trained via a novel Positive-Negative prompting strategy to deliver structured narratives across five Narrative Quality Dimensions (NQDs); (2) a Mixture-of-Experts (MoE) plot generator aligned via Direct Preference Optimization on high-confidence preference pairs; and (3) an Agentic Evaluation module that emulates human critical judgment for unbiased post-hoc assessment. Extensive experiments demonstrate that PlotTwist consistently outperforms frontier models across multiple NQDs despite substantially tighter capacity constraints. Further validation confirms strong sensitivity to narrative quality, as the framework reliably distinguishes plots derived from critically acclaimed versus widely panned screenplays. Together, these results establish structured, preference-based alignment as a resource-efficient approach to high-quality creative plot generation. 

---
# Who Benchmarks the Benchmarks? A Case Study of LLM Evaluation in Icelandic 

**Authors**: Finnur Ágúst Ingimundarson, Steinunn Rut Friðriksdóttir, Bjarki Ármannsson, Iris Edda Nowenstein, Steinþór Steingrímsson  

**Link**: [PDF](https://arxiv.org/pdf/2603.16406)  

**Abstract**: This paper evaluates current Large Language Model (LLM) benchmarking for Icelandic, identifies problems, and calls for improved evaluation methods in low/medium-resource languages in particular. We show that benchmarks that include synthetic or machine-translated data that have not been verified in any way, commonly contain severely flawed test examples that are likely to skew the results and undermine the tests' validity. We warn against the use of such methods without verification in low/medium-resource settings as the translation quality can, at best, only be as good as MT quality for a given language at any given time. Indeed, the results of our quantitative error analysis on existing benchmarks for Icelandic show clear differences between human-authored/-translated benchmarks vs. synthetic or machine-translated benchmarks. 

---
# Fanar 2.0: Arabic Generative AI Stack 

**Authors**: FANAR TEAM, Ummar Abbas, Mohammad Shahmeer Ahmad, Minhaj Ahmad, Abdulaziz Al-Homaid, Anas Al-Nuaimi, Enes Altinisik, Ehsaneddin Asgari, Sanjay Chawla, Shammur Chowdhury, Fahim Dalvi, Kareem Darwish, Nadir Durrani, Mohamed Elfeky, Ahmed Elmagarmid, Mohamed Eltabakh, Asim Ersoy, Masoomali Fatehkia, Mohammed Qusay Hashim, Majd Hawasly, Mohamed Hefeeda, Mus'ab Husaini, Keivin Isufaj, Soon-Gyo Jung, Houssam Lachemat, Ji Kim Lucas, Abubakr Mohamed, Tasnim Mohiuddin, Basel Mousi, Hamdy Mubarak, Ahmad Musleh, Mourad Ouzzani, Amin Sadeghi, Husrev Taha Sencar, Mohammed Shinoy, Omar Sinan, Yifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16397)  

**Abstract**: We present Fanar 2.0, the second generation of Qatar's Arabic-centric Generative AI platform. Sovereignty is a first-class design principle: every component, from data pipelines to deployment infrastructure, was designed and operated entirely at QCRI, Hamad Bin Khalifa University. Fanar 2.0 is a story of resource-constrained excellence: the effort ran on 256 NVIDIA H100 GPUs, with Arabic having only ~0.5% of web data despite 400 million native speakers. Fanar 2.0 adopts a disciplined strategy of data quality over quantity, targeted continual pre-training, and model merging to achieve substantial gains within these constraints. At the core is Fanar-27B, continually pre-trained from a Gemma-3-27B backbone on a curated corpus of 120 billion high-quality tokens across three data recipes. Despite using 8x fewer pre-training tokens than Fanar 1.0, it delivers substantial benchmark improvements: Arabic knowledge (+9.1 pts), language (+7.3 pts), dialects (+3.5 pts), and English capability (+7.6 pts). Beyond the core LLM, Fanar 2.0 introduces a rich stack of new capabilities. FanarGuard is a state-of-the-art 4B bilingual moderation filter for Arabic safety and cultural alignment. The speech family Aura gains a long-form ASR model for hours-long audio. Oryx vision family adds Arabic-aware image and video understanding alongside culturally grounded image generation. An agentic tool-calling framework enables multi-step workflows. Fanar-Sadiq utilizes a multi-agent architecture for Islamic content. Fanar-Diwan provides classical Arabic poetry generation. FanarShaheen delivers LLM-powered bilingual translation. A redesigned multi-layer orchestrator coordinates all components through intent-aware routing and defense-in-depth safety validation. Taken together, Fanar 2.0 demonstrates that sovereign, resource-constrained AI development can produce systems competitive with those built at far greater scale. 

---
# Robust Physics-Guided Diffusion for Full-Waveform Inversion 

**Authors**: Jishen Peng, Enze Jiang, Zheng Ma, Xiongbin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2603.16393)  

**Abstract**: We develop a robust physics-guided diffusion framework for full-waveform inversion that combines a score-based generative prior with likelihood guidance computed through wave-equation simulations. We adopt a transport-based data-consistency potential (Wasserstein-2), incorporating wavefield enhancement via bounded weighting and observation-dependent normalization, thereby improving robustness to amplitude imbalance and time/phase misalignment. On the inference side, we introduce a preconditioned guided reverse-diffusion scheme that adapts the guidance strength and spatial scaling throughout the reverse-time dynamics, yielding a more stable and effective data-consistency guidance step than standard diffusion posterior sampling (DPS). Numerical experiments on OpenFWI datasets demonstrate improved reconstruction quality over deterministic optimization baselines and standard DPS under comparable computational budgets. 

---
# Age Predictors Through the Lens of Generalization, Bias Mitigation, and Interpretability: Reflections on Causal Implications 

**Authors**: Debdas Paul, Elisa Ferrari, Irene Gravili, Alessandro Cellerino  

**Link**: [PDF](https://arxiv.org/pdf/2603.16377)  

**Abstract**: Chronological age predictors often fail to achieve out-of-distribution (OOD) gen- eralization due to exogenous attributes such as race, gender, or tissue. Learning an invariant representation with respect to those attributes is therefore essential to improve OOD generalization and prevent overly optimistic results. In predic- tive settings, these attributes motivate bias mitigation; in causal analyses, they appear as confounders; and when protected, their suppression leads to fairness. We coherently explore these concepts with theoretical rigor and discuss the scope of an interpretable neural network model based on adversarial representation learning. Using publicly available mouse transcriptomic datasets, we illustrate the behavior of this model relative to conventional machine learning models. We observe that the outcome of this model is consistent with the predictive results of a published study demonstrating the effects of Elamipretide on mouse skeletal and cardiac muscle. We conclude by discussing the limitations of deriving causal interpretation from such purely predictive models. 

---
# FederatedFactory: Generative One-Shot Learning for Extremely Non-IID Distributed Scenarios 

**Authors**: Andrea Moleri, Christian Internò, Ali Raza, Markus Olhofer, David Klindt, Fabio Stella, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2603.16370)  

**Abstract**: Federated Learning (FL) enables distributed optimization without compromising data sovereignty. Yet, where local label distributions are mutually exclusive, standard weight aggregation fails due to conflicting optimization trajectories. Often, FL methods rely on pretrained foundation models, introducing unrealistic assumptions. We introduce FederatedFactory, a zero-dependency framework that inverts the unit of federation from discriminative parameters to generative priors. By exchanging generative modules in a single communication round, our architecture supports ex nihilo synthesis of universally class balanced datasets, eliminating gradient conflict and external prior bias entirely. Evaluations across diverse medical imagery benchmarks, including MedMNIST and ISIC2019, demonstrate that our approach recovers centralized upper-bound performance. Under pathological heterogeneity, it lifts baseline accuracy from a collapsed 11.36% to 90.57% on CIFAR-10 and restores ISIC2019 AUROC to 90.57%. Additionally, this framework facilitates exact modular unlearning through the deterministic deletion of specific generative modules. 

---
# DynamicGate MLP Conditional Computation via Learned Structural Dropout and Input Dependent Gating for Functional Plasticity 

**Authors**: Yong Il Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16367)  

**Abstract**: Dropout is a representative regularization technique that stochastically deactivates hidden units during training to mitigate overfitting. In contrast, standard inference executes the full network with dense computation, so its goal and mechanism differ from conditional computation, where the executed operations depend on the input. This paper organizes DynamicGate-MLP into a single framework that simultaneously satisfies both the regularization view and the conditional-computation view. Instead of a random mask, the proposed model learns gates that decide whether to use each unit (or block), suppressing unnecessary computation while implementing sample-dependent execution that concentrates computation on the parts needed for each input. To this end, we define continuous gate probabilities and, at inference time, generate a discrete execution mask from them to select an execution path. Training controls the compute budget via a penalty on expected gate usage and uses a Straight-Through Estimator (STE) to optimize the discrete mask. We evaluate DynamicGate-MLP on MNIST, CIFAR-10, Tiny-ImageNet, Speech Commands, and PBMC3k, and compare it with various MLP baselines and MoE-style variants. Compute efficiency is compared under a consistent criterion using gate activation ratios and a layerweighted relative MAC metric, rather than wall-clock latency that depends on hardware and backend kernels. 

---
# $D^3$-RSMDE: 40$\times$ Faster and High-Fidelity Remote Sensing Monocular Depth Estimation 

**Authors**: Ruizhi Wang, Weihan Li, Zunlei Feng, Haofei Zhang, Mingli Song, Jiayu Wang, Jie Song, Li Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.16362)  

**Abstract**: Real-time, high-fidelity monocular depth estimation from remote sensing imagery is crucial for numerous applications, yet existing methods face a stark trade-off between accuracy and efficiency. Although using Vision Transformer (ViT) backbones for dense prediction is fast, they often exhibit poor perceptual quality. Conversely, diffusion models offer high fidelity but at a prohibitive computational cost. To overcome these limitations, we propose Depth Detail Diffusion for Remote Sensing Monocular Depth Estimation ($D^3$-RSMDE), an efficient framework designed to achieve an optimal balance between speed and quality. Our framework first leverages a ViT-based module to rapidly generate a high-quality preliminary depth map construction, which serves as a structural prior, effectively replacing the time-consuming initial structure generation stage of diffusion models. Based on this prior, we propose a Progressive Linear Blending Refinement (PLBR) strategy, which uses a lightweight U-Net to refine the details in only a few iterations. The entire refinement step operates efficiently in a compact latent space supported by a Variational Autoencoder (VAE). Extensive experiments demonstrate that $D^3$-RSMDE achieves a notable 11.85% reduction in the Learned Perceptual Image Patch Similarity (LPIPS) perceptual metric over leading models like Marigold, while also achieving over a 40x speedup in inference and maintaining VRAM usage comparable to lightweight ViT models. 

---
# Toward Experimentation-as-a-Service in 5G/6G: The Plaza6G Prototype for AI-Assisted Trials 

**Authors**: Sergio Barrachina-Muñoz, Marc Carrascosa-Zamacois, Horacio Bleda, Umair Riaz, Yasir Maqsood, Xavier Calle, Selva Vía, Miquel Payaró, Josep Mangues-Bafalluy  

**Link**: [PDF](https://arxiv.org/pdf/2603.16356)  

**Abstract**: This paper presents Plaza6G, the first operational Experiment-as-a-Service (ExaS) platform unifying cloud resources with next-generation wireless infrastructure. Developed at CTTC in Barcelona, Plaza6G integrates GPU-accelerated compute clusters, multiple 5G cores, both open-source (e.g., Free5GC) and commercial (e.g., Cumucore), programmable RANs, and physical or emulated user equipment under unified orchestration. In Plaza6G, the experiment design requires minimal expertise as it is expressed in natural language via a web portal or a REST API. The web portal and REST API are enhanced with a Large Language Model (LLM)-based assistant, which employs retrieval-augmented generation (RAG) for up-to-date experiment knowledge and Low-Rank Adaptation (LoRA) for continuous domain fine-tuning. Over-the-air (OTA) trials leverage a four-chamber anechoic facility and a dual-site outdoor 5G network operating in sub-6~GHz and mmWave bands. Demonstrations include automated CI/CD integration with sub-ten-minute setup and interactive OTA testing under programmable propagation conditions. Machine-readable experiment descriptors ensure reproducibility, while future work targets policy-aware orchestration, safety validation, and federated testbed integration toward open, reproducible wireless experimentation. 

---
# Automated identification of Ichneumonoidea wasps via YOLO-based deep learning: Integrating HiresCam for Explainable AI 

**Authors**: Joao Manoel Herrera Pinheiro, Gabriela Do Nascimento Herrera, Alvaro Doria Dos Santos, Luciana Bueno Dos Reis Fernandes, Ricardo V. Godoy, Eduardo A. B. Almeida, Helena Carolina Onody, Marcelo Andrade Da Costa Vieira, Angelica Maria Penteado-Dias, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2603.16351)  

**Abstract**: Accurate taxonomic identification of parasitoid wasps within the superfamily Ichneumonoidea is essential for biodiversity assessment, ecological monitoring, and biological control programs. However, morphological similarity, small body size, and fine-grained interspecific variation make manual identification labor-intensive and expertise-dependent. This study proposes a deep learning-based framework for the automated identification of Ichneumonoidea wasps using a YOLO-based architecture integrated with High-Resolution Class Activation Mapping (HiResCAM) to enhance interpretability. The proposed system simultaneously identifies wasp families from high-resolution images. The dataset comprises 3556 high-resolution images of Hymenoptera specimens. The taxonomic distribution is primarily concentrated among the families Ichneumonidae (n = 786), Braconidae (n = 648), Apidae (n = 466), and Vespidae (n = 460). Extensive experiments were conducted using a curated dataset, with model performance evaluated through precision, recall, F1 score, and accuracy. The results demonstrate high accuracy of over 96 % and robust generalization across morphological variations. HiResCAM visualizations confirm that the model focuses on taxonomically relevant anatomical regions, such as wing venation, antennae segmentation, and metasomal structures, thereby validating the biological plausibility of the learned features. The integration of explainable AI techniques improves transparency and trustworthiness, making the system suitable for entomological research to accelerate biodiversity characterization in an under-described parasitoid superfamily. 

---
# Explainable machine learning workflows for radio astronomical data processing 

**Authors**: S. Yatawatta, A. Ahmadi, B. Asabere, M. Iacobelli, N. Peters, M. Veldhuis  

**Link**: [PDF](https://arxiv.org/pdf/2603.16350)  

**Abstract**: Radio astronomy relies heavily on efficient and accurate processing pipelines to deliver science ready data. With the increasing data flow of modern radio telescopes, manual configuration of such data processing pipelines is infeasible. Machine learning (ML) is already emerging as a viable solution for automating data processing pipelines. However, almost all existing ML enabled pipelines are of black-box type, where the decisions made by the automating agents are not easily deciphered by astronomers. In order to improve the explainability of the ML aided data processing pipelines in radio astronomy, we propose the joint use of fuzzy rule based inference and deep learning. We consider one application in radio astronomy, i.e., calibration, to showcase the proposed approach of ML aided decision making using a Takagi-Sugeno-Kang (TSK) fuzzy system. We provide results based on simulations to illustrate the increased explainability of the proposed approach, not compromising on the quality or accuracy. 

---
# Detecting Sentiment Steering Attacks on RAG-enabled Large Language Models 

**Authors**: Isha Andrade, Shalaka S Mahadik, Mithun Mukherjee, Pranav M Pawar, Raja Muthalagu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16342)  

**Abstract**: The proliferation of large-scale IoT networks has been both a blessing and a curse. Not only has it revolutionized the way organizations operate by increasing the efficiency of automated procedures, but it has also simplified our daily lives. However, while IoT networks have improved convenience and connectivity, they have also increased security risk due to unauthorized devices gaining access to these networks and exploiting existing weaknesses with specific attack types. The research proposes two lightweight deep learning (DL)-based intelligent intrusion detection systems (IDS). to enhance the security of IoT networks: the proposed convolutional neural network (CNN)-based IDS and the proposed long short-term memory (LSTM)-based IDS. The research evaluated the performance of both intelligent IDSs based on DL using the CICIoT2023 dataset. DL-based intelligent IDSs successfully identify and classify various cyber threats using binary, grouped, and multi-class classification. The proposed CNN-based IDS achieves an accuracy of 99.34%, 99.02% and 98.6%, while the proposed LSTM-based IDS achieves an accuracy of 99.42%, 99.13%, and 98.68% for binary, grouped, and multi-class classification, respectively. 

---
# An Interpretable Machine Learning Framework for Non-Small Cell Lung Cancer Drug Response Analysis 

**Authors**: Ann Rachel, Pranav M Pawar, Mithun Mukharjee, Raja M, Tojo Mathew  

**Link**: [PDF](https://arxiv.org/pdf/2603.16330)  

**Abstract**: Lung cancer is a condition where there is abnormal growth of malignant cells that spread in an uncontrollable fashion in the lungs. Some common treatment strategies are surgery, chemotherapy, and radiation which aren't the best options due to the heterogeneous nature of cancer. In personalized medicine, treatments are tailored according to the individual's genetic information along with lifestyle aspects. In addition, AI-based deep learning methods can analyze large sets of data to find early signs of cancer, types of tumor, and prospects of treatment. The paper focuses on the development of personalized treatment plans using specific patient data focusing primarily on the genetic profile. Multi-Omics data from Genomics of Drug Sensitivity in Cancer have been used to build a predictive model along with machine learning techniques. The value of the target variable, LN-IC50, determines how sensitive or resistive a drug is. An XGBoost regressor is utilized to predict the drug response focusing on molecular and cellular features extracted from cancer datasets. Cross-validation and Randomized Search are performed for hyperparameter tuning to further optimize the model's predictive performance. For explanation purposes, SHAP (SHapley Additive exPlanations) was used. SHAP values measure each feature's impact on an individual prediction. Furthermore, interpreting feature relationships was performed using DeepSeek, a large language model trained to verify the biological validity of the features. Contextual explanations regarding the most important genes or pathways were provided by DeepSeek alongside the top SHAP value constituents, supporting the predictability of the model. 

---
# A Human-Centred Architecture for Large Language Models-Cognitive Assistants in Manufacturing within Quality Management Systems 

**Authors**: Marcos Galdino, Johanna Grahl, Tobias Hamann, Anas Abdelrazeq, Ingrid Isenhardt  

**Link**: [PDF](https://arxiv.org/pdf/2603.16325)  

**Abstract**: Large Language Models-Cognitive Assistants (LLM-CAs) can enhance Quality Management Systems (QMS) in manufacturing, fostering continuous process improvement and knowledge management. However, there is no human-centred software architecture focused on QMS that enables the integration of LLM-CAs into manufacturing in the current literature. This study addresses this gap by designing a component-based architecture considering requirement analysis and software development process. Validation was conducted via iterative expert focus groups. The proposed architecture ensures flexibility, scalability, modularity, and work augmentation within QMS. Moreover, it paves the way for its operationalization with industrial partners, showcasing its potential for advancing manufacturing processes. 

---
# Attention-guided Evidence Grounding for Spoken Question Answering 

**Authors**: Ke Yang, Bolin Chen, Yuejie Li, Yueying Hua, Jianhao Nie, Yueping He, Bowen Li, Chengjun Mao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16292)  

**Abstract**: Spoken Question Answering (Spoken QA) presents a challenging cross-modal problem: effectively aligning acoustic queries with textual knowledge while avoiding the latency and error propagation inherent in cascaded ASR-based systems. In this paper, we introduce Attention-guided Evidence Grounding (AEG), a novel end-to-end framework that leverages the internal cross-modal attention of Speech Large Language Models (SpeechLLMs) to explicitly locate and ground key evidence in the model's latent space. To address the diffuse attention distribution in pre-trained models, we propose Learning to Focus on Evidence (LFE), a supervised fine-tuning paradigm that calibrates the model's attention mechanism to distinguish query-relevant segments from irrelevant context. Experiments on SQuAD, HotpotQA, and MuSiQue demonstrate that AEG reduces hallucinations and achieves strong efficiency gains, outperforming large-scale cascaded baselines (Whisper-Large-v3 + Reranker) while reducing inference latency by approximately 62%. 

---
# VisBrowse-Bench: Benchmarking Visual-Native Search for Multimodal Browsing Agents 

**Authors**: Zhengbo Zhang, Jinbo Su, Zhaowen Zhou, Changtao Miao, Yuhan Hong, Qimeng Wu, Yumeng Liu, Feier Wu, Yihe Tian, Yuhao Liang, Zitong Shan, Wanke Xia, Yi-Fan Zhang, Bo Zhang, Zhe Li, Shiming Xiang, Ying Yan  

**Link**: [PDF](https://arxiv.org/pdf/2603.16289)  

**Abstract**: The rapid advancement of Multimodal Large Language Models (MLLMs) has enabled browsing agents to acquire and reason over multimodal information in the real world. But existing benchmarks suffer from two limitations: insufficient evaluation of visual reasoning ability and the neglect of native visual information of web pages in the reasoning chains. To address these challenges, we introduce a new benchmark for visual-native search, VisBrowse-Bench. It contains 169 VQA instances covering multiple domains and evaluates the models' visual reasoning capabilities during the search process through multimodal evidence cross-validation via text-image retrieval and joint reasoning. These data were constructed by human experts using a multi-stage pipeline and underwent rigorous manual verification. We additionally propose an agent workflow that can effectively drive the browsing agent to actively collect and reason over visual information during the search process. We comprehensively evaluated both open-source and closed-source models in this workflow. Experimental results show that even the best-performing model, Claude-4.6-Opus only achieves an accuracy of 47.6%, while the proprietary Deep Research model, o3-deep-research only achieves an accuracy of 41.1%. The code and data can be accessed at: this https URL 

---
# Surrogate-Assisted Genetic Programming with Rank-Based Phenotypic Characterisation for Dynamic Multi-Mode Project Scheduling 

**Authors**: Yuan Tian, Yi Mei, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16286)  

**Abstract**: The dynamic multi-mode resource-constrained project scheduling problem (DMRCPSP) is of practical importance, as it requires making real-time decisions under changing project states and resource availability. Genetic Programming (GP) has been shown to effectively evolve heuristic rules for such decision-making tasks; however, the evolutionary process typically relies on a large number of simulation-based fitness evaluations, resulting in high computational cost. Surrogate models offer a promising solution to reduce evaluation cost, but their application to GP requires problem-specific phenotypic characterisation (PC) schemes of heuristic rules. There is currently a lack of suitable PC schemes for GP applied to DMRCPSP.
This paper proposes a rank-based PC scheme derived from heuristic-driven ordering of eligible activity-mode pairs and activity groups in decision situations. The resulting PC vectors enable a surrogate model to estimate the fitness of unevaluated GP individuals. Based on this scheme, a surrogate-assisted GP algorithm is developed. Experimental results demonstrate that the proposed surrogate-assisted GP can identify high-quality heuristic rules consistently earlier than the state-of-the-art GP approach for DMRCPSP, while introducing only marginal computational overhead. Further analyses demonstrate that the surrogate model provides useful guidance for offspring selection, leading to improved evolutionary efficiency. 

---
# AW-MoE: All-Weather Mixture of Experts for Robust Multi-Modal 3D Object Detection 

**Authors**: Hongwei Lin, Xun Huang, Chenglu Wen, Cheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16261)  

**Abstract**: Robust 3D object detection under adverse weather conditions is crucial for autonomous driving. However, most existing methods simply combine all weather samples for training while overlooking data distribution discrepancies across different weather scenarios, leading to performance conflicts. To address this issue, we introduce AW-MoE, the framework that innovatively integrates Mixture of Experts (MoE) into weather-robust multi-modal 3D object detection approaches. AW-MoE incorporates Image-guided Weather-aware Routing (IWR), which leverages the superior discriminability of image features across weather conditions and their invariance to scene variations for precise weather classification. Based on this accurate classification, IWR selects the top-K most relevant Weather-Specific Experts (WSE) that handle data discrepancies, ensuring optimal detection under all weather conditions. Additionally, we propose a Unified Dual-Modal Augmentation (UDMA) for synchronous LiDAR and 4D Radar dual-modal data augmentation while preserving the realism of scenes. Extensive experiments on the real-world dataset demonstrate that AW-MoE achieves ~ 15% improvement in adverse-weather performance over state-of-the-art methods, while incurring negligible inference overhead. Moreover, integrating AW-MoE into established baseline detectors yields performance improvements surpassing current state-of-the-art methods. These results show the effectiveness and strong scalability of our AW-MoE. We will release the code publicly at this https URL. 

---
# Human/AI Collective Intelligence for Deliberative Democracy: A Human-Centred Design Approach 

**Authors**: Anna De Liddo, Lucas Anastasiou, Simon Buckingham Shum  

**Link**: [PDF](https://arxiv.org/pdf/2603.16260)  

**Abstract**: This chapter introduces the concept of Collective Intelligence for Deliberative Democracy (CI4DD). We propose that the use of computational tools, specifically artificial intelligence to advance deliberative democracy, is an instantiation of a broader class of human-computer system designed to augment collective intelligence. Further, we argue for a fundamentally human-centred design approach to orchestrate how stakeholders can contribute meaningfully to shaping the artifacts and processes needed to create trustworthy DD processes. We first contextualise the key concepts of CI and the role of AI within it. We then detail our co-design methodology for identifying key challenges, refining user scenarios, and deriving technical implications. Two exemplar cases illustrate how user requirements from civic organisations were implemented with AI support and piloted in authentic contexts. 

---
# Grounding the Score: Explicit Visual Premise Verification for Reliable Vision-Language Process Reward Models 

**Authors**: Junxin Wang, Dai Guan, Weijie Qiu, Zhihang Li, Yongbo Gai, Zhengyi Yang, Mengyu Zhou, Erchao Zhao, Xiaoxi Jiang, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16253)  

**Abstract**: Vision-language process reward models (VL-PRMs) are increasingly used to score intermediate reasoning steps and rerank candidates under test-time scaling. However, they often function as black-box judges: a low step score may reflect a genuine reasoning mistake or simply the verifier's misperception of the image. This entanglement between perception and reasoning leads to systematic false positives (rewarding hallucinated visual premises) and false negatives (penalizing correct grounded statements), undermining both reranking and error localization. We introduce Explicit Visual Premise Verification (EVPV), a lightweight verification interface that conditions step scoring on the reliability of the visual premises a step depends on. The policy is prompted to produce a step-wise visual checklist that makes required visual facts explicit, while a constraint extractor independently derives structured visual constraints from the input image. EVPV matches checklist claims against these constraints to compute a scalar visual reliability signal, and calibrates PRM step rewards via reliability gating: rewards for visually dependent steps are attenuated when reliability is low and preserved when reliability is high. This decouples perceptual uncertainty from logical evaluation without per-step tool calls. Experiments on VisualProcessBench and six multimodal reasoning benchmarks show that EVPV improves step-level verification and consistently boosts Best-of-N reranking accuracy over strong baselines. Furthermore, injecting controlled corruption into the extracted constraints produces monotonic performance degradation, providing causal evidence that the gains arise from constraint fidelity and explicit premise verification rather than incidental prompt effects. Code is available at: this https URL 

---
# Visual Prompt Discovery via Semantic Exploration 

**Authors**: Jaechang Kim, Yotaro Shimose, Zhao Wang, Kuang-Da Wang, Jungseul Ok, Shingo Takamatsu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16250)  

**Abstract**: LVLMs encounter significant challenges in image understanding and visual reasoning, leading to critical perception failures. Visual prompts, which incorporate image manipulation code, have shown promising potential in mitigating these issues. While emerged as a promising direction, previous methods for visual prompt generation have focused on tool selection rather than diagnosing and mitigating the root causes of LVLM perception failures. Because of the opacity and unpredictability of LVLMs, optimal visual prompts must be discovered through empirical experiments, which have relied on manual human trial-and-error.
We propose an automated semantic exploration framework for discovering task-wise visual prompts. Our approach enables diverse yet efficient exploration through agent-driven experiments, minimizing human intervention and avoiding the inefficiency of per-sample generation. We introduce a semantic exploration algorithm named SEVEX, which addresses two major challenges of visual prompt exploration: (1) the distraction caused by lengthy, low-level code and (2) the vast, unstructured search space of visual prompts. Specifically, our method leverages an abstract idea space as a search space, a novelty-guided selection algorithm, and a semantic feedback-driven ideation process to efficiently explore diverse visual prompts based on empirical results.
We evaluate SEVEX on the BlindTest and BLINK benchmarks, which are designed to assess LVLM perception. Experimental results demonstrate that SEVEX significantly outperforms baseline methods in task accuracy, inference efficiency, exploration efficiency, and exploration stability. Notably, our framework discovers sophisticated and counter-intuitive visual strategies that go beyond conventional tool usage, offering a new paradigm for enhancing LVLM perception through automated, task-wise visual prompts. 

---
# RASLF: Representation-Aware State Space Model for Light Field Super-Resolution 

**Authors**: Zeqiang Wei, Kai Jin, Kuan Song, Xiuzhuang Zhou, Wenlong Chen, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16243)  

**Abstract**: Current SSM-based light field super-resolution (LFSR) methods often fail to fully leverage the complementarity among various LF representations, leading to the loss of fine textures and geometric misalignments across views. To address these issues, we propose RASLF, a representation-aware state-space framework that explicitly models structural correlations across multiple LF representations. Specifically, a Progressive Geometric Refinement (PGR) block is created that uses a panoramic epipolar representation to explicitly encode multi-view parallax differences, thereby enabling integration across different LF representations. Furthermore, we introduce a Representation Aware Asymmetric Scanning (RAAS) mechanism that dynamically adjusts scanning paths based on the physical properties of different representation spaces, optimizing the balance between performance and efficiency through path pruning. Additionally, a Dual-Anchor Aggregation (DAA) module improves hierarchical feature flow, reducing redundant deeplayer features and prioritizing important reconstruction information. Experiments on various public benchmarks show that RASLF achieves the highest reconstruction accuracy while remaining highly computationally efficient. 

---
# Generative AI for Quantum Circuits and Quantum Code: A Technical Review and Taxonomy 

**Authors**: Juhani Merilehto  

**Link**: [PDF](https://arxiv.org/pdf/2603.16216)  

**Abstract**: We review thirteen generative systems and five supporting datasets for quantum circuit and quantum code generation, identified through a structured scoping review of Hugging Face, arXiv, and provenance tracing (January-February 2026). We organize the field along two axes: artifact type (Qiskit code, OpenQASM programs, circuit graphs); crossed with training regime (supervised fine-tuning, verifier-in-the-loop RL, diffusion/graph generation, agentic optimization); and systematically apply a three-layer evaluation framework covering syntactic validity, semantic correctness, and hardware executability. The central finding is that while all reviewed systems address syntax and most address semantics to some degree, none reports end-to-end evaluation on quantum hardware (Layer 3b), leaving a significant gap between generated circuits and practical deployment. Scope note: quantum code refers throughout to quantum program artifacts (QASM, Qiskit); we do not cover generation of quantum error-correcting codes (QEC). 

---
# CoMAI: A Collaborative Multi-Agent Framework for Robust and Equitable Interview Evaluation 

**Authors**: Gengxin Sun, Ruihao Yu, Liangyi Yin, Yunqi Yang, Bin Zhang, Zhiwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16215)  

**Abstract**: Ensuring robust and fair interview assessment remains a key challenge in AI-driven evaluation. This paper presents CoMAI, a general-purpose multi-agent interview framework designed for diverse assessment scenarios. In contrast to monolithic single-agent systems based on large language models (LLMs), CoMAI employs a modular task-decomposition architecture coordinated through a centralized finite-state machine. The system comprises four agents specialized in question generation, security, scoring, and summarization. These agents work collaboratively to provide multi-layered security defenses against prompt injection, support multidimensional evaluation with adaptive difficulty adjustment, and enable rubric-based structured scoring that reduces subjective bias. Experimental results demonstrate that CoMAI achieved 90.47% accuracy, 83.33% recall, and 84.41% candidate satisfaction. These results highlight CoMAI as a robust, fair, and interpretable paradigm for AI-driven interview assessment. 

---
# A Scoping Review of AI-Driven Digital Interventions in Mental Health Care: Mapping Applications Across Screening, Support, Monitoring, Prevention, and Clinical Education 

**Authors**: Yang Ni, Fanli Jia  

**Link**: [PDF](https://arxiv.org/pdf/2603.16204)  

**Abstract**: Artificial intelligence (AI)-enabled digital interventions, including Generative AI (GenAI) and Human-Centered AI (HCAI), are increasingly used to expand access to digital psychiatry and mental health care. This PRISMA-ScR scoping review maps the landscape of AI-driven mental health (mHealth) technologies across five critical phases: pre-treatment (screening/triage), treatment (therapeutic support), post-treatment (remote patient monitoring), clinical education, and population-level prevention. We synthesized 36 empirical studies implemented through early 2024, focusing on Large Language Models (LLMs), machine learning (ML) models, and autonomous conversational agents. Key use cases involve referral triage, empathic communication enhancement, and AI-assisted psychotherapy delivered via chatbots and voice agents. While benefits include reduced wait times and increased patient engagement, we address recurring challenges like algorithmic bias, data privacy, and human-AI collaboration barriers. By introducing a novel four-pillar framework, this review provides a comprehensive roadmap for AI-augmented mental health care, offering actionable insights for researchers, clinicians, and policymakers to develop safe, effective, and equitable digital health interventions. 

---
# Robust Generative Audio Quality Assessment: Disentangling Quality from Spurious Correlations 

**Authors**: Kuan-Tang Huang, Chien-Chun Wang, Cheng-Yeh Yang, Hung-Shin Lee, Hsin-Min Wang, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16201)  

**Abstract**: The rapid proliferation of AI-Generated Content (AIGC) has necessitated robust metrics for perceptual quality assessment. However, automatic Mean Opinion Score (MOS) prediction models are often compromised by data scarcity, predisposing them to learn spurious correlations-- such as dataset-specific acoustic signatures-- rather than generalized quality features. To address this, we leverage domain adversarial training (DAT) to disentangle true quality perception from these nuisance factors. Unlike prior works that rely on static domain priors, we systematically investigate domain definition strategies ranging from explicit metadata-driven labels to implicit data-driven clusters. Our findings reveal that there is no "one-size-fits-all" domain definition; instead, the optimal strategy is highly dependent on the specific MOS aspect being evaluated. Experimental results demonstrate that our aspect-specific domain strategy effectively mitigates acoustic biases, significantly improving correlation with human ratings and achieving superior generalization on unseen generative scenarios. 

---
# Sample-Efficient Adaptation of Drug-Response Models to Patient Tumors under Strong Biological Domain Shift 

**Authors**: Camille Jimenez Cortes, Philippe Lalanda, German Vega  

**Link**: [PDF](https://arxiv.org/pdf/2603.16185)  

**Abstract**: Predicting drug response in patients from preclinical data remains a major challenge in precision oncology due to the substantial biological gap between in vitro cell lines and patient tumors. Rather than aiming to improve absolute in vitro prediction accuracy, this work examines whether explicitly separating representation learning from task supervision enables more sample-efficient adaptation of drug-response models to patient data under strong biological domain shift. We propose a staged transfer-learning framework in which cellular and drug representations are first learned independently from large collections of unlabeled pharmacogenomic data using autoencoder-based representation learning. These representations are then aligned with drug-response labels on cell-line data and subsequently adapted to patient tumors using few-shot supervision. Through a systematic evaluation spanning in-domain, cross-dataset, and patient-level settings, we show that unsupervised pretraining provides limited benefit when source and target domains overlap substantially, but yields clear gains when adapting to patient tumors with very limited labeled data. In particular, the proposed framework achieves faster performance improvements during few-shot patient-level adaptation while maintaining comparable accuracy to single-phase baselines on standard cell-line benchmarks. Overall, these results demonstrate that learning structured and transferable representations from unlabeled molecular profiles can substantially reduce the amount of clinical supervision required for effective drug-response prediction, offering a practical pathway toward data-efficient preclinical-to-clinical translation. 

---
# 360° Image Perception with MLLMs: A Comprehensive Benchmark and a Training-Free Method 

**Authors**: Huyen T. T. Tran, Van-Quang Nguyen, Farros Alferro, Kang-Jun Liu, Takayuki Okatani  

**Link**: [PDF](https://arxiv.org/pdf/2603.16179)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown impressive abilities in understanding and reasoning over conventional images. However, their perception of 360° images remains largely underexplored. Unlike conventional images, 360° images capture the entire surrounding environment, enabling holistic spatial reasoning but introducing challenges such as geometric distortion and complex spatial relations. To comprehensively assess MLLMs' capabilities to perceive 360° images, we introduce 360Bench, a Visual Question Answering (VQA) benchmark featuring 7K-resolution 360° images, seven representative (sub)tasks with annotations carefully curated by human annotators. Using 360Bench, we systematically evaluate seven MLLMs and six enhancement methods, revealing their shortcomings in 360° image perception. To address these challenges, we propose Free360, a training-free scene-graph-based framework for high-resolution 360° VQA. Free360 decomposes the reasoning process into modular steps, applies adaptive spherical image transformations to 360° images tailored to each step, and seamlessly integrates the resulting information into a unified graph representation for answer generation. Experiments show that Free360 consistently improves its base MLLM and provides a strong training-free solution for 360° VQA tasks. The source code and dataset will be publicly released upon acceptance. 

---
# MemX: A Local-First Long-Term Memory System for AI Assistants 

**Authors**: Lizheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.16171)  

**Abstract**: We present MemX, a local-first long-term memory system for AI assistants with stability-oriented retrieval design. MemX is implemented in Rust on top of libSQL and an OpenAI-compatible embedding API, providing persistent, searchable, and explainable memory for conversational agents. Its retrieval pipeline applies vector recall, keyword recall, Reciprocal Rank Fusion (RRF), four-factor re-ranking, and a low-confidence rejection rule that suppresses spurious recalls when no answer exists in the memory store.
We evaluate MemX on two axes. First, two custom Chinese-language benchmark suites (43 queries, <=1,014 records) validate pipeline design: Hit@1=91.3% on a default scenario and 100% under high confusion, with conservative miss-query suppression. Second, the LongMemEval benchmark (500 queries, up to 220,349 records) quantifies system boundaries across four ability types and three storage granularities. At fact-level granularity the system reaches Hit@5=51.6% and MRR=0.380, doubling session-level performance, while temporal and multi-session reasoning remain challenging (<=43.6% Hit@5). FTS5 full-text indexing reduces keyword search latency by 1,100x at 100k-record scale, keeping end-to-end search under 90 ms.
Unlike Mem0 and related work that targets end-to-end agent benchmarks, MemX focuses on a narrower, reproducible baseline: local-first deployment, structural simplicity, explainable retrieval, and stability-oriented design. 

---
# Open-Source Reproduction and Explainability Analysis of Corrective Retrieval Augmented Generation 

**Authors**: Surya Vardhan Yalavarthi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16169)  

**Abstract**: Corrective Retrieval Augmented Generation (CRAG) improves the robustness of RAG systems by evaluating retrieved document quality and triggering corrective actions. However, the original implementation relies on proprietary components including the Google Search API and closed model weights, limiting reproducibility. In this work, we present a fully open-source reproduction of CRAG, replacing proprietary web search with the Wikipedia API and the original LLaMA-2 generator with Phi-3-mini-4k-instruct. We evaluate on PopQA and ARC-Challenge, demonstrating that our open-source pipeline achieves comparable performance to the original system. Furthermore, we contribute the first explainability analysis of CRAG's T5-based retrieval evaluator using SHAP, revealing that the evaluator primarily relies on named entity alignment rather than semantic similarity. Our analysis identifies key failure modes including domain transfer limitations on science questions. All code and results are available at this https URL. 

---
# Homogeneous and Heterogeneous Consistency progressive Re-ranking for Visible-Infrared Person Re-identification 

**Authors**: Yiming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16165)  

**Abstract**: Visible-infrared person re-identification faces greater challenges than traditional person re-identification due to the significant differences between modalities. In particular, the differences between these modalities make effective matching even more challenging, mainly because existing re-ranking algorithms cannot simultaneously address the intra-modal variations and inter-modal discrepancy in cross-modal person re-identification. To address this problem, we propose a novel Progressive Modal Relationship Re-ranking method consisting of two modules, called heterogeneous and homogeneous consistency re-ranking(HHCR). The first module, heterogeneous consistency re-ranking, explores the relationship between the query and the gallery modalities in the test set. The second module, homogeneous consistency reranking, investigates the intrinsic relationship within each modality between the query and the gallery in the test set. Based on this, we propose a baseline for cross-modal person re-identification, called a consistency re-ranking inference network (CRI). We conducted comprehensive experiments demonstrating that our proposed re-ranking method is generalized, and both the re-ranking and the baseline achieve state-of-the-art performance. 

---
# DyJR: Preserving Diversity in Reinforcement Learning with Verifiable Rewards via Dynamic Jensen-Shannon Replay 

**Authors**: Long Li, Zhijian Zhou, Tianyi Wang, Weidi Xu, Zuming Huang, Wei Chu, Zhe Wang, Shirui Pan, Chao Qu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16157)  

**Abstract**: While Reinforcement Learning (RL) enhances Large Language Model reasoning, on-policy algorithms like GRPO are sample-inefficient as they discard past rollouts. Existing experience replay methods address this by reusing accurate samples for direct policy updates, but this often incurs high computational costs and causes mode collapse via overfitting. We argue that historical data should prioritize sustaining diversity rather than simply reinforcing accuracy. To this end, we propose Dynamic Jensen-Shannon Replay (DyJR), a simple yet effective regularization framework using a dynamic reference distribution from recent trajectories. DyJR introduces two innovations: (1) A Time-Sensitive Dynamic Buffer that uses FIFO and adaptive sizing to retain only temporally proximal samples, synchronizing with model evolution; and (2) Jensen-Shannon Divergence Regularization, which replaces direct gradient updates with a distributional constraint to prevent diversity collapse. Experiments on mathematical reasoning and Text-to-SQL benchmarks demonstrate that DyJR significantly outperforms GRPO as well as baselines such as RLEP and Ex-GRPO, while maintaining training efficiency comparable to the original GRPO. Furthermore, from the perspective of Rank-$k$ token probability evolution, we show that DyJR enhances diversity and mitigates over-reliance on Rank-1 tokens, elucidating how specific sub-modules of DyJR influence the training dynamics. 

---
# GATS: Gaussian Aware Temporal Scaling Transformer for Invariant 4D Spatio-Temporal Point Cloud Representation 

**Authors**: Jiayi Tian, Jiaze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16154)  

**Abstract**: Understanding 4D point cloud videos is essential for enabling intelligent agents to perceive dynamic environments. However, temporal scale bias across varying frame rates and distributional uncertainty in irregular point clouds make it highly challenging to design a unified and robust 4D backbone. Existing CNN or Transformer based methods are constrained either by limited receptive fields or by quadratic computational complexity, while neglecting these implicit distortions. To address this problem, we propose a novel dual invariant framework, termed \textbf{Gaussian Aware Temporal Scaling (GATS)}, which explicitly resolves both distributional inconsistencies and temporal. The proposed \emph{Uncertainty Guided Gaussian Convolution (UGGC)} incorporates local Gaussian statistics and uncertainty aware gating into point convolution, thereby achieving robust neighborhood aggregation under density variation, noise, and occlusion. In parallel, the \emph{Temporal Scaling Attention (TSA)} introduces a learnable scaling factor to normalize temporal distances, ensuring frame partition invariance and consistent velocity estimation across different frame rates. These two modules are complementary: temporal scaling normalizes time intervals prior to Gaussian estimation, while Gaussian modeling enhances robustness to irregular distributions. Our experiments on mainstream benchmarks MSR-Action3D (\textbf{+6.62\%} accuracy), NTU RGBD (\textbf{+1.4\%} accuracy), and Synthia4D (\textbf{+1.8\%} mIoU) demonstrate significant performance gains, offering a more efficient and principled paradigm for invariant 4D point cloud video understanding with superior accuracy, robustness, and scalability compared to Transformer based counterparts. 

---
# HIPO: Instruction Hierarchy via Constrained Reinforcement Learning 

**Authors**: Keru Chen, Jun Luo, Sen Lin, Yingbin Liang, Alvaro Velasquez, Nathaniel Bastian, Shaofeng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.16152)  

**Abstract**: Hierarchical Instruction Following (HIF) refers to the problem of prompting large language models with a priority-ordered stack of instructions. Standard methods like RLHF and DPO typically fail in this problem since they mainly optimize for a single objective, failing to explicitly enforce system prompt compliance. Meanwhile, supervised fine-tuning relies on mimicking filtered, compliant data, which fails to establish the priority asymmetry at the algorithmic level. In this paper, we introduce \textsc{HIPO}, a novel alignment framework that formulates HIF as a Constrained Markov Decision Process. \textsc{HIPO} elevates system prompts from mere input context to strict algorithmic boundaries. Using a primal-dual safe reinforcement learning approach, the algorithm dynamically enforces system prompt compliance as an explicit constraint, maximizing user utility strictly within this feasible region. Extensive evaluations across diverse model architectures (e.g., Qwen, Phi, Llama) demonstrate that \textsc{HIPO} significantly improves both system compliance and user utility. Furthermore, mechanistic analysis reveals that this constrained optimization autonomously drives the model to shift its attention toward long-range system tokens, providing a principled foundation for reliable LLM deployment in complex workflows. 

---
# Structure-Aware Multimodal LLM Framework for Trustworthy Near-Field Beam Prediction 

**Authors**: Mengyuan Li, Qianfan Lu, Jiachen Tian, Hongjun Hu, Yu Han, Xiao Li, Chao-kai Wen, Shi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.16143)  

**Abstract**: In near-field extremely large-scale multiple-input multiple-output (XL-MIMO) systems, spherical wavefront propagation expands the traditional beam codebook into the joint angular-distance domain, rendering conventional beam training prohibitively inefficient, especially in complex 3-dimensional (3D) low-altitude environments. Furthermore, since near-field beam variations are deeply coupled not only with user positions but also with the physical surroundings, precise beam alignment demands profound environmental understanding capabilities. To address this, we propose a large language model (LLM)-driven multimodal framework that fuses historical GPS data, RGB image, LiDAR data, and strategically designed task-specific textual prompts. By utilizing the powerful emergent reasoning and generalization capabilities of the LLM, our approach learns complex spatial dynamics to achieve superior environmental comprehension... 

---
# When Generative Augmentation Hurts: A Benchmark Study of GAN and Diffusion Models for Bias Correction in AI Classification Systems 

**Authors**: Shesh Narayan Gupta, Nik Bear Brown  

**Link**: [PDF](https://arxiv.org/pdf/2603.16134)  

**Abstract**: Generative models are widely used to compensate for class imbalance in AI training pipelines, yet their failure modes under low-data conditions are poorly understood. This paper reports a controlled benchmark comparing three augmentation strategies applied to a fine-grained animal classification task: traditional transforms, FastGAN, and Stable Diffusion 1.5 fine-tuned with Low-Rank Adaptation (LoRA). Using the Oxford-IIIT Pet Dataset with eight artificially underrepresented breeds, we find that FastGAN augmentation does not merely underperform at very low training set sizes but actively increases classifier bias, with a statistically significant large effect across three random seeds (bias gap increase: +20.7%, Cohen's d = +5.03, p = 0.013). The effect size here is large enough to give confidence in the direction of the finding despite the small number of seeds. Feature embedding analysis using t-distributed Stochastic Neighbor Embedding reveals that FastGAN images for severe-minority breeds form tight isolated clusters outside the real image distribution, a pattern consistent with mode collapse. Stable Diffusion with Low-Rank Adaptation produced the best results overall, achieving the highest macro F1 (0.9125 plus or minus 0.0047) and a 13.1% reduction in the bias gap relative to the unaugmented baseline. The data suggest a sample-size boundary somewhere between 20 and 50 training images per class below which GAN augmentation becomes harmful in this setting, though further work across additional domains is needed to establish where that boundary sits more precisely. All experiments run on a consumer-grade GPU with 6 to 8 GB of memory, with no cloud compute required. 

---
# SWE-QA-Pro: A Representative Benchmark and Scalable Training Recipe for Repository-Level Code Understanding 

**Authors**: Songcheng Cai, Zhiheng Lyu, Yuansheng Ni, Xiangchao Chen, Baichuan Zhou, Shenzhe Zhu, Yi Lu, Haozhe Wang, Chi Ruan, Benjamin Schneider, Weixu Zhang, Xiang Li, Andy Zheng, Yuyu Zhang, Ping Nie, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16124)  

**Abstract**: Agentic repository-level code understanding is essential for automating complex software engineering tasks, yet the field lacks reliable benchmarks. Existing evaluations often overlook the long tail topics and rely on popular repositories where Large Language Models (LLMs) can cheat via memorized knowledge. To address this, we introduce SWE-QA-Pro, a benchmark constructed from diverse, long-tail repositories with executable environments. We enforce topical balance via issue-driven clustering to cover under-represented task types and apply a rigorous difficulty calibration process: questions solvable by direct-answer baselines are filtered out. This results in a dataset where agentic workflows significantly outperform direct answering (e.g., a ~13-point gap for Claude Sonnet 4.5), confirming the necessity of agentic codebase exploration. Furthermore, to tackle the scarcity of training data for such complex behaviors, we propose a scalable synthetic data pipeline that powers a two-stage training recipe: Supervised Fine-Tuning (SFT) followed by Reinforcement Learning from AI Feedback (RLAIF). This approach allows small open models to learn efficient tool usage and reasoning. Empirically, a Qwen3-8B model trained with our recipe surpasses GPT-4o by 2.3 points on SWE-QA-Pro and substantially narrows the gap to state-of-the-art proprietary models, demonstrating both the validity of our evaluation and the effectiveness of our agentic training workflow. 

---
# Functorial Neural Architectures from Higher Inductive Types 

**Authors**: Karen Sargsyan  

**Link**: [PDF](https://arxiv.org/pdf/2603.16123)  

**Abstract**: Neural networks systematically fail at compositional generalization -- producing correct outputs for novel combinations of known parts. We show that this failure is architectural: compositional generalization is equivalent to functoriality of the decoder, and this perspective yields both guarantees and impossibility results. We compile Higher Inductive Type (HIT) specifications into neural architectures via a monoidal functor from the path groupoid of a target space to a category of parametric maps: path constructors become generator networks, composition becomes structural concatenation, and 2-cells witnessing group relations become learned natural transformations. We prove that decoders assembled by structural concatenation of independently generated segments are strict monoidal functors (compositional by construction), while softmax self-attention is not functorial for any non-trivial compositional task. Both results are formalized in Cubical Agda. Experiments on three spaces validate the full hierarchy: on the torus ($\mathbb{Z}^2$), functorial decoders outperform non-functorial ones by 2-2.7x; on $S^1 \vee S^1$ ($F_2$), the type-A/B gap widens to 5.5-10x; on the Klein bottle ($\mathbb{Z} \rtimes \mathbb{Z}$), a learned 2-cell closes a 46% error gap on words exercising the group relation. 

---
# PathGLS: Evaluating Pathology Vision-Language Models without Ground Truth through Multi-Dimensional Consistency 

**Authors**: Minbing Chen, Zhu Meng, Fei Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.16113)  

**Abstract**: Vision-Language Models (VLMs) offer significant potential in computational pathology by enabling interpretable image analysis, automated reporting, and scalable decision support. However, their widespread clinical adoption remains limited due to the absence of reliable, automated evaluation metrics capable of identifying subtle failures such as hallucinations. To address this gap, we propose PathGLS, a novel reference-free evaluation framework that assesses pathology VLMs across three dimensions: Grounding (fine-grained visual-text alignment), Logic (entailment graph consistency using Natural Language Inference), and Stability (output variance under adversarial visual-semantic perturbations). PathGLS supports both patch-level and whole-slide image (WSI)-level analysis, yielding a comprehensive trust score. Experiments on Quilt-1M, TCGA, REG2025, PathMMU and TCGA-Sarcoma datasets demonstrate the superiority of PathGLS. Specifically, on the Quilt-1M dataset, PathGLS reveals a steep sensitivity drop of 40.2% for hallucinated reports compared to only 2.1% for BERTScore. Moreover, validation against expert-defined clinical error hierarchies reveals that PathGLS achieves a strong Spearman's rank correlation of $\rho=0.71$ ($p < 0.0001$), significantly outperforming Large Language Model (LLM)-based approaches (Gemini 3.0 Pro: $\rho=0.39$, $p < 0.0001$). These results establish PathGLS as a robust reference-free metric. By directly quantifying hallucination rates and domain shift robustness, it serves as a reliable criterion for benchmarking VLMs on private clinical datasets and informing safe deployment. Code can be found at: this https URL 

---
# ASDA: Automated Skill Distillation and Adaptation for Financial Reasoning 

**Authors**: Tik Yu Yim, Wenting Tan, Sum Yee Chan, Tak-Wah Lam, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16112)  

**Abstract**: Adapting large language models (LLMs) to specialized financial reasoning typically requires expensive fine-tuning that produces model-locked expertise. Training-free alternatives have emerged, yet our experiments show that leading methods (GEPA and ACE) achieve only marginal gains on the FAMMA financial reasoning benchmark, exposing the limits of unstructured text optimization for complex, multi-step domain reasoning. We introduce Automated Skill Distillation and Adaptation (ASDA), a framework that automatically generates structured skill artifacts through iterative error-corrective learning without modifying model weights. A teacher model analyzes a student model's failures on financial reasoning tasks, clusters errors by subfield and error type, and synthesizes skill files containing reasoning procedures, code templates, and worked examples, which are dynamically injected during inference. Evaluated on FAMMA, ASDA achieves up to +17.33% improvement on arithmetic reasoning and +5.95% on non-arithmetic reasoning, substantially outperforming all training-free baselines. The resulting skill artifacts are human-readable, version-controlled, and compatible with the Agent Skills open standard, offering any organization with a labeled domain dataset a practical and auditable path to domain adaptation without weight access or retraining. 

---
# RepoReviewer: A Local-First Multi-Agent Architecture for Repository-Level Code Review 

**Authors**: Peng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16107)  

**Abstract**: Repository-level code review requires reasoning over project structure, repository context, and file-level implementation details. Existing automated review workflows often collapse these tasks into a single pass, which can reduce relevance, increase duplication, and weaken prioritization. We present RepoReviewer, a local-first multi-agent system for automated GitHub repository review with a Python CLI, FastAPI API, LangGraph orchestration layer, and this http URL user interface. RepoReviewer decomposes review into repository acquisition, context synthesis, file-level analysis, finding prioritization, and summary generation. We describe the system design, implementation tradeoffs, developer-facing interfaces, and practical failure modes. Rather than claiming benchmark superiority, we frame RepoReviewer as a technical systems contribution: a pragmatic architecture for repository-level automated review, accompanied by reusable evaluation and reporting infrastructure for future empirical study. 

---
# Frequency Matters: Fast Model-Agnostic Data Curation for Pruning and Quantization 

**Authors**: Francesco Pio Monaco, Elia Cunegatti, Flavio Vella, Giovanni Iacca  

**Link**: [PDF](https://arxiv.org/pdf/2603.16105)  

**Abstract**: Post-training model compression is essential for enhancing the portability of Large Language Models (LLMs) while preserving their performance. While several compression approaches have been proposed, less emphasis has been placed on selecting the most suitable set of data (the so-called \emph{calibration data}) for finding the compressed model configuration. The choice of calibration data is a critical step in preserving model capabilities both intra- and inter-tasks. In this work, we address the challenge of identifying high-performance calibration sets for both pruning and quantization by analyzing intrinsic data properties rather than model-specific signals. We introduce \texttt{\textbf{ZipCal}}, a model-agnostic data curation strategy that maximizes lexical diversity based on Zipfian power laws. Experiments demonstrate that our method consistently outperforms standard uniform random sampling across various pruning benchmarks. Notably, it also performs on par, in terms of downstream performance, with a state-of-the-art method that relies on model perplexity. The latter becomes prohibitively expensive at large-scale models and datasets, while \texttt{\textbf{ZipCal}} is on average $\sim$240$\times$ faster due to its tractable linear complexity\footnote{We make the code and the experiments available at this https URL.}. 

---
# Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective 

**Authors**: Noppanat Wadlom, Junyi Shen, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16104)  

**Abstract**: Agentic workflows are composed of sequences of interdependent Large Language Model (LLM) calls, and they have become a dominant workload in modern AI systems. These workflows exhibit extensive redundancy from overlapping prompts and intermediate results due to speculative and parallel exploration. Existing LLM serving systems, such as vLLM, focus on optimizing individual inference calls and overlook cross-call dependencies, leading to significant inefficiencies. This paper rethinks LLM and agent serving from a data systems perspective and introduces Helium, a workflow-aware serving framework that models agentic workloads as query plans and treats LLM invocations as first-class operators. Helium integrates proactive caching and cache-aware scheduling to maximize reuse across prompts, KV states, and workflows. Through these techniques, Helium bridges classic query optimization principles with LLM serving, achieving up to 1.56x speedup over state-of-the-art agent serving systems on various workloads. Our results demonstrate that end-to-end optimization across workflows is essential for scalable and efficient LLM-based agents. 

---
# LICA: Layered Image Composition Annotations for Graphic Design Research 

**Authors**: Elad Hirsch, Shubham Yadav, Mohit Garg, Purvanshi Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2603.16098)  

**Abstract**: We introduce LICA (Layered Image Composition Annotations), a large-scale dataset of 1,550,244 multi-layer graphic design compositions designed to advance structured understanding and generation of graphic layouts1. In addition to ren- dered PNG images, LICA represents each design as a hierarchical composition of typed components including text, image, vector, and group elements, each paired with rich per-element metadata such as spatial geometry, typographic attributes, opacity, and visibility. The dataset spans 20 design categories and 971,850 unique templates, providing broad coverage of real-world design structures. We further introduce graphic design video as a new and largely unexplored challenge for current vision-language models through 27,261 animated layouts annotated with per-component keyframes and motion parameters. Beyond scale, LICA establishes a new paradigm of research tasks for graphic design, enabling structured investiga- tions into problems such as layer-aware inpainting, structured layout generation, controlled design editing, and temporally-aware generative modeling. By repre- senting design as a system of compositional layers and relationships, the dataset supports research on models that operate directly on design structure rather than pixels alone. 

---
# Diffusion Models for Joint Audio-Video Generation 

**Authors**: Alejandro Paredes La Torre  

**Link**: [PDF](https://arxiv.org/pdf/2603.16093)  

**Abstract**: Multimodal generative models have shown remarkable progress in single-modality video and audio synthesis, yet truly joint audio-video generation remains an open challenge. In this paper, I explore four key contributions to advance this field. First, I release two high-quality, paired audio-video datasets. The datasets consisting on 13 hours of video-game clips and 64 hours of concert performances, each segmented into consistent 34-second samples to facilitate reproducible research. Second, I train the MM-Diffusion architecture from scratch on our datasets, demonstrating its ability to produce semantically coherent audio-video pairs and quantitatively evaluating alignment on rapid actions and musical cues. Third, I investigate joint latent diffusion by leveraging pretrained video and audio encoder-decoders, uncovering challenges and inconsistencies in the multimodal decoding stage. Finally, I propose a sequential two-step text-to-audio-video generation pipeline: first generating video, then conditioning on both the video output and the original prompt to synthesize temporally synchronized audio. My experiments show that this modular approach yields high-fidelity generations of audio video generation. 

---
# Parallel In-context Learning for Large Vision Language Models 

**Authors**: Shin'ya Yamaguchi, Daiki Chijiwa, Tamao Sakao, Taku Hasegawa  

**Link**: [PDF](https://arxiv.org/pdf/2603.16092)  

**Abstract**: Large vision-language models (LVLMs) employ multi-modal in-context learning (MM-ICL) to adapt to new tasks by leveraging demonstration examples. While increasing the number of demonstrations boosts performance, they incur significant inference latency due to the quadratic computational cost of Transformer attention with respect to the context length. To address this trade-off, we propose Parallel In-Context Learning (Parallel-ICL), a plug-and-play inference algorithm. Parallel-ICL partitions the long demonstration context into multiple shorter, manageable chunks. It processes these chunks in parallel and integrates their predictions at the logit level, using a weighted Product-of-Experts (PoE) ensemble to approximate the full-context output. Guided by ensemble learning theory, we introduce principled strategies for Parallel-ICL: (i) clustering-based context chunking to maximize inter-chunk diversity and (ii) similarity-based context compilation to weight predictions by query relevance. Extensive experiments on VQA, image captioning, and classification benchmarks demonstrate that Parallel-ICL achieves performance comparable to full-context MM-ICL, while significantly improving inference speed. Our work offers an effective solution to the accuracy-efficiency trade-off in MM-ICL, enabling dynamic task adaptation with substantially reduced inference overhead. 

---
# CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering 

**Authors**: Tianyi Huang, Ying Kai Deng  

**Link**: [PDF](https://arxiv.org/pdf/2603.16091)  

**Abstract**: In factual question answering, many errors are not failures of access but failures of commitment: the system retrieves relevant evidence, yet still settles on the wrong answer. We present CounterRefine, a lightweight inference-time repair layer for retrieval-grounded question answering. CounterRefine first produces a short answer from retrieved evidence, then gathers additional support and conflicting evidence with follow-up queries conditioned on that draft answer, and finally applies a restricted refinement step that outputs either KEEP or REVISE, with proposed revisions accepted only if they pass deterministic validation. In effect, CounterRefine turns retrieval into a mechanism for testing a provisional answer rather than merely collecting more context. On the full SimpleQA benchmark, CounterRefine improves a matched GPT-5 Baseline-RAG by 5.8 points and reaches a 73.1 percent correct rate, while exceeding the reported one-shot GPT-5.4 score by roughly 40 points. These findings suggest a simple but important direction for knowledgeable foundation models: beyond accessing evidence, they should also be able to use that evidence to reconsider and, when necessary, repair their own answers. 

---
# RecBundle: A Next-Generation Geometric Paradigm for Explainable Recommender Systems 

**Authors**: Hui Wang, Tianzhu Hu, Mingming Li, Xi Zhou, Chun Gan, Jiao Dai, Jizhong Han, Songlin Hu, Tao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.16088)  

**Abstract**: Recommender systems are inherently dynamic feedback loops where prolonged local interactions accumulate into macroscopic structural degradation such as information cocoons. Existing representation learning paradigms are universally constrained by the assumption of a single flat space, forcing topologically grounded user associations and semantically driven historical interactions to be fitted within the same vector space. This excessive coupling of heterogeneous information renders it impossible for researchers to mechanistically distinguish and identify the sources of systemic bias. To overcome this theoretical bottleneck, we introduce Fiber Bundle from modern differential geometry and propose a novel geometric analysis paradigm for recommender systems. This theory naturally decouples the system space into two hierarchical layers: the base manifold formed by user interaction networks, and the fibers attached to individual user nodes that carry their dynamic preferences. Building upon this, we construct RecBundle, a framework oriented toward next-generation recommender systems that formalizes user collaboration as geometric connection and parallel transport on the base manifold, while mapping content evolution to holonomy transformations on fibers. From this foundation, we identify future application directions encompassing quantitative mechanisms for information cocoons and evolutionary bias, geometric meta-theory for adaptive recommendation, and novel inference architectures integrating large language models (LLMs). Empirical analysis on real-world MovieLens and Amazon Beauty datasets validates the effectiveness of this geometric framework. 

---
# Towards the Vision-Sound-Language-Action Paradigm: The HEAR Framework for Sound-Centric Manipulation 

**Authors**: Chang Nie, Tianchen Deng, Guangming Wang, Zhe Liu, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16086)  

**Abstract**: While recent Vision-Language-Action (VLA) models have begun to incorporate audio, they typically treat sound as static pre-execution prompts or focus exclusively on human speech. This leaves a significant gap in real-time, sound-centric manipulation where fleeting environmental acoustics provide critical state verification during task execution. Consequently, key sounds are easily missed due to low-frequency updates or system latency. This problem is exacerbated by action chunking with open-loop execution, which creates a Blind Execution Interval where acoustic events are lost between discrete audio observation windows. Recognizing the necessity of continuous auditory awareness, we formalize Vision-Sound-Language-Action (VSLA) as a continuous control paradigm conditioned on vision, streaming audio, language, and proprioception under delayed decision loops. As an instantiation, we introduce HEAR, a VSLA framework integrating four components: (i) a streaming Historizer to maintain a compact, causal audio context across execution gaps; (ii) an Envisioner adapted from omni foundation models to reason over multi-sensory inputs; (iii) an Advancer, formulated as an audio world model, to learn temporal dynamics by predicting near-future audio codes; and (iv) a flow-matching Realizer policy to generate smooth action chunks. To address the scarcity of pretraining data and evaluations for VSLA, we construct OpenX-Sound for pretraining, alongside HEAR-Bench, the first sound-centric manipulation benchmark with strict causal timing rules. Our results suggest that robust sound-centric manipulation necessitates causal persistence and explicit temporal learning. This framework provides a practical step toward multi-sensory foundation models for embodied agents, enabling robots to perceive and interact with dynamic environments. Code and videos are available at this https URL. 

---
# Interact3D: Compositional 3D Generation of Interactive Objects 

**Authors**: Hui Shan, Keyang Luo, Ming Li, Sizhe Zheng, Yanwei Fu, Zhen Chen, Xiangru Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16085)  

**Abstract**: Recent breakthroughs in 3D generation have enabled the synthesis of high-fidelity individual assets. However, generating 3D compositional objects from single images--particularly under occlusions--remains challenging. Existing methods often degrade geometric details in hidden regions and fail to preserve the underlying object-object spatial relationships (OOR). We present a novel framework Interact3D designed to generate physically plausible interacting 3D compositional objects. Our approach first leverages advanced generative priors to curate high-quality individual assets with a unified 3D guidance scene. To physically compose these assets, we then introduce a robust two-stage composition pipeline. Based on the 3D guidance scene, the primary object is anchored through precise global-to-local geometric alignment (registration), while subsequent geometries are integrated using a differentiable Signed Distance Field (SDF)-based optimization that explicitly penalizes geometry intersections. To reduce challenging collisions, we further deploy a closed-loop, agentic refinement strategy. A Vision-Language Model (VLM) autonomously analyzes multi-view renderings of the composed scene, formulates targeted corrective prompts, and guides an image editing module to iteratively self-correct the generation pipeline. Extensive experiments demonstrate that Interact3D successfully produces promising collsion-aware compositions with improved geometric fidelity and consistent spatial relationships. 

---
# SEAHateCheck: Functional Tests for Detecting Hate Speech in Low-Resource Languages of Southeast Asia 

**Authors**: Ri Chi Ng, Aditi Kumaresan, Yujia Hu, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.16070)  

**Abstract**: Hate speech detection relies heavily on linguistic resources, which are primarily available in high-resource languages such as English and Chinese, creating barriers for researchers and platforms developing tools for low-resource languages in Southeast Asia, where diverse socio-linguistic contexts complicate online hate moderation. To address this, we introduce SEAHateCheck, a pioneering dataset tailored to Indonesia, Thailand, the Philippines, and Vietnam, covering Indonesian, Tagalog, Thai, and Vietnamese. Building on HateCheck's functional testing framework and refining SGHateCheck's methods, SEAHateCheck provides culturally relevant test cases, augmented by large language models and validated by local experts for accuracy. Experiments with state-of-the-art and multilingual models revealed limitations in detecting hate speech in specific low-resource languages. In particular, Tagalog test cases showed the lowest model accuracy, likely due to linguistic complexity and limited training data. In contrast, slang-based functional tests proved the hardest, as models struggled with culturally nuanced expressions. The diagnostic insights of SEAHateCheck further exposed model weaknesses in implicit hate detection and models' struggles with counter-speech expression. As the first functional test suite for these Southeast Asian languages, this work equips researchers with a robust benchmark, advancing the development of practical, culturally attuned hate speech detection tools for inclusive online content moderation. 

---
# Resource Consumption Threats in Large Language Models 

**Authors**: Yuanhe Zhang, Xinyue Wang, Zhican Chen, Weiliu Wang, Zilu Zhang, Zhengshuo Gong, Zhenhong Zhou, Li Sun, Yang Liu, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.16068)  

**Abstract**: Given limited and costly computational infrastructure, resource efficiency is a key requirement for large language models (LLMs). Efficient LLMs increase service capacity for providers and reduce latency and API costs for users. Recent resource consumption threats induce excessive generation, degrading model efficiency and harming both service availability and economic sustainability. This survey presents a systematic review of threats to resource consumption in LLMs. We further establish a unified view of this emerging area by clarifying its scope and examining the problem along the full pipeline from threat induction to mechanism understanding and mitigation. Our goal is to clarify the problem landscape for this emerging area, thereby providing a clearer foundation for characterization and mitigation. 

---
# Large Reward Models: Generalizable Online Robot Reward Generation with Vision-Language Models 

**Authors**: Yanru Wu, Weiduo Yuan, Ang Qi, Vitor Guizilini, Jiageng Mao, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16065)  

**Abstract**: Reinforcement Learning (RL) has shown great potential in refining robotic manipulation policies, yet its efficacy remains strongly bottlenecked by the difficulty of designing generalizable reward functions. In this paper, we propose a framework for online policy refinement by adapting foundation VLMs into online reward generators. We develop a robust, scalable reward model based on a state-of-the-art VLM, trained on a large-scale, multi-source dataset encompassing real-world robot trajectories, human-object interactions, and diverse simulated environments. Unlike prior approaches that evaluate entire trajectories post-hoc, our method leverages the VLM to formulate a multifaceted reward signal comprising process, completion, and temporal contrastive rewards based on current visual observations. Initializing with a base policy trained via Imitation Learning (IL), we employ these VLM rewards to guide the model to correct sub-optimal behaviors in a closed-loop manner. We evaluate our framework on challenging long-horizon manipulation benchmarks requiring sequential execution and precise control. Crucially, our reward model operates in a purely zero-shot manner within these test environments. Experimental results demonstrate that our method significantly improves the success rate of the initial IL policy within just 30 RL iterations, demonstrating remarkable sample efficiency. This empirical evidence highlights that VLM-generated signals can provide reliable feedback to resolve execution errors, effectively eliminating the need for manual reward engineering and facilitating efficient online refinement for robot learning. 

---
# Collaborative Temporal Feature Generation via Critic-Free Reinforcement Learning for Cross-User Sensor-Based Activity Recognition 

**Authors**: Xiaozhou Ye, Feng Jiang, Zihan Wang, Xiulai Wang, Yutao Zhang, Kevin I-Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16043)  

**Abstract**: Human Activity Recognition using wearable inertial sensors is foundational to healthcare monitoring, fitness analytics, and context-aware computing, yet its deployment is hindered by cross-user variability arising from heterogeneous physiological traits, motor habits, and sensor placements. Existing domain generalization approaches either neglect temporal dependencies in sensor streams or depend on impractical target-domain annotations. We propose a different paradigm: modeling generalizable feature extraction as a collaborative sequential generation process governed by reinforcement learning. Our framework, CTFG (Collaborative Temporal Feature Generation), employs a Transformer-based autoregressive generator that incrementally constructs feature token sequences, each conditioned on prior context and the encoded sensor input. The generator is optimized via Group-Relative Policy Optimization, a critic-free algorithm that evaluates each generated sequence against a cohort of alternatives sampled from the same input, deriving advantages through intra-group normalization rather than learned value estimation. This design eliminates the distribution-dependent bias inherent in critic-based methods and provides self-calibrating optimization signals that remain stable across heterogeneous user distributions. A tri-objective reward comprising class discrimination, cross-user invariance, and temporal fidelity jointly shapes the feature space to separate activities, align user distributions, and preserve fine-grained temporal content. Evaluations on the DSADS and PAMAP2 benchmarks demonstrate state-of-the-art cross-user accuracy (88.53\% and 75.22\%), substantial reduction in inter-task training variance, accelerated convergence, and robust generalization under varying action-space dimensionalities. 

---
# Residual Stream Duality in Modern Transformer Architectures 

**Authors**: Yifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16039)  

**Abstract**: Recent work has made clear that the residual pathway is not mere optimization plumbing; it is part of the model's representational machinery. We agree, but argue that the cleanest way to organize this design space is through a two-axis view of the Transformer. A decoder evolves information along two ordered dimensions: sequence position and layer depth. Self-attention already provides adaptive mixing along the sequence axis, whereas the residual stream usually performs fixed addition along the depth axis. If we fix a token position and treat layer index as the ordered variable, then a causal depth-wise residual attention read is exactly the same local operator as causal short sliding-window attention (ShortSWA), except written over depth rather than over sequence. This is the core residual stream duality behind Transformer$^2$. This perspective also clarifies the recent literature. ELC-BERT and DenseFormer already show that learned aggregation over depth can outperform uniform residual accumulation, while Vertical Attention, DeepCrossAttention (DCA), MUDDFormer, and Attention Residuals move further toward explicit attention-based routing over earlier layers. The key point, however, is that operator-level duality does not imply systems-level symmetry. For large-scale autoregressive models, sequence-axis ShortSWA is usually the more hardware-friendly placement because it reuses token-side sliding-window kernels, KV-cache layouts, and chunked execution. If the goal is instead to change the shortcut itself, Deep Delta Learning (DDL) is the cleaner intervention because it modifies the residual operator directly rather than adding a separate cross-layer retrieval path. Our recommendation is therefore simple: use DDL when the shortcut is the object of interest, and use sequence-axis ShortSWA when the goal is local adaptive mixing. 

---
# Understanding Moral Reasoning Trajectories in Large Language Models: Toward Probing-Based Explainability 

**Authors**: Fan Huang, Haewoon Kwak, Jisun An  

**Link**: [PDF](https://arxiv.org/pdf/2603.16017)  

**Abstract**: Large language models (LLMs) increasingly participate in morally sensitive decision-making, yet how they organize ethical frameworks across reasoning steps remains underexplored. We introduce \textit{moral reasoning trajectories}, sequences of ethical framework invocations across intermediate reasoning steps, and analyze their dynamics across six models and three benchmarks. We find that moral reasoning involves systematic multi-framework deliberation: 55.4--57.7\% of consecutive steps involve framework switches, and only 16.4--17.8\% of trajectories remain framework-consistent. Unstable trajectories remain 1.29$\times$ more susceptible to persuasive attacks ($p=0.015$). At the representation level, linear probes localize framework-specific encoding to model-specific layers (layer 63/81 for Llama-3.3-70B; layer 17/81 for Qwen2.5-72B), achieving 13.8--22.6\% lower KL divergence than the training-set prior baseline. Lightweight activation steering modulates framework integration patterns (6.7--8.9\% drift reduction) and amplifies the stability--accuracy relationship. We further propose a Moral Representation Consistency (MRC) metric that correlates strongly ($r=0.715$, $p<0.0001$) with LLM coherence ratings, whose underlying framework attributions are validated by human annotators (mean cosine similarity $= 0.859$). 

---
# FlatLands: Generative Floormap Completion From a Single Egocentric View 

**Authors**: Subhransu S. Bhattacharjee, Dylan Campbell, Rahul Shome  

**Link**: [PDF](https://arxiv.org/pdf/2603.16016)  

**Abstract**: A single egocentric image typically captures only a small portion of the floor, yet a complete metric traversability map of the surroundings would better serve applications such as indoor navigation. We introduce FlatLands, a dataset and benchmark for single-view bird's-eye view (BEV) floor completion. The dataset contains 270,575 observations from 17,656 real metric indoor scenes drawn from six existing datasets, with aligned observation, visibility, validity, and ground-truth BEV maps, and the benchmark includes both in- and out-of-distribution evaluation protocols. We compare training-free approaches, deterministic models, ensembles, and stochastic generative models. Finally, we instantiate the task as an end-to-end monocular RGB-to-floormaps pipeline. FlatLands provides a rigorous testbed for uncertainty-aware indoor mapping and generative completion for embodied navigation. 

---
# Evaluating Agentic Optimization on Large Codebases 

**Authors**: Atharva Sehgal, James Hou, Akanksha Sarkar, Ishaan Mantripragada, Swarat Chaudhuri, Jennifer J. Sun, Yisong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2603.16011)  

**Abstract**: Large language model (LLM) coding agents increasingly operate at the repository level, motivating benchmarks that evaluate their ability to optimize entire codebases under realistic constraints. Existing code benchmarks largely rely on synthetic tasks, binary correctness signals, or single-objective evaluation, limiting their ability to assess holistic optimization behavior. We introduce FormulaCode, a benchmark for evaluating agentic optimization on large, real-world codebases with fine-grained, multi-objective performance metrics. FormulaCode comprises 957 performance bottlenecks mined from scientific Python repositories on GitHub, each paired with expert-authored patches and, on average, 264.6 community-maintained performance workloads per task, enabling the holistic ability of LLM agents to optimize codebases under realistic correctness and performance constraints. Our evaluations reveal that repository-scale, multi-objective optimization remains a major challenge for frontier LLM agents. Project website at: this https URL 

---
# RadAnnotate: Large Language Models for Efficient and Reliable Radiology Report Annotation 

**Authors**: Saisha Pradeep Shetty, Roger Eric Goldman, Vladimir Filkov  

**Link**: [PDF](https://arxiv.org/pdf/2603.16002)  

**Abstract**: Radiology report annotation is essential for clinical NLP, yet manual labeling is slow and costly. We present RadAnnotate, an LLM-based framework that studies retrieval-augmented synthetic reports and confidence-based selective automation to reduce expert effort for labeling in RadGraph. We study RadGraph-style entity labeling (graph nodes) and leave relation extraction (edges) to future work. First, we train entity-specific classifiers on gold-standard reports and characterize their strengths and failure modes across anatomy and observation categories, with uncertain observations hardest to learn. Second, we generate RAG-guided synthetic reports and show that synthetic-only models remain within 1-2 F1 points of gold-trained models, and that synthetic augmentation is especially helpful for uncertain observations in a low-resource setting, improving F1 from 0.61 to 0.70. Finally, by learning entity-specific confidence thresholds, RadAnnotate can automatically annotate 55-90% of reports at 0.86-0.92 entity match score while routing low-confidence cases for expert review. 

---
# The Midas Touch in Gaze vs. Hand Pointing: Modality-Specific Failure Modes and Implications for XR Interfaces 

**Authors**: Mohammad Dastgheib, Fatemeh Pourmahdian  

**Link**: [PDF](https://arxiv.org/pdf/2603.15991)  

**Abstract**: Extended Reality (XR) interfaces impose both ergonomic and cognitive demands, yet current systems often force a binary choice between hand-based input, which can produce fatigue, and gaze-based input, which is vulnerable to the Midas Touch problem and precision limitations. We introduce the xr-adaptive-modality-2025 platform, a web-based open-source framework for studying whether modality-specific adaptive interventions can improve XR-relevant pointing performance and reduce workload relative to static unimodal interaction. The platform combines physiologically informed gaze simulation, an ISO 9241-9 multidirectional tapping task, and two modality-specific adaptive interventions: gaze declutter and hand target-width inflation. We evaluated the system in a 2 x 2 x 2 within-subjects design manipulating Modality (Hand vs. Gaze), UI Mode (Static vs. Adaptive), and Pressure (Yes vs. No). Results from N=69 participants show that hand yielded higher throughput than gaze (5.17 vs. 4.73 bits/s), lower error (1.8% vs. 19.1%), and lower NASA-TLX workload. Crucially, error profiles differed sharply by modality: gaze errors were predominantly slips (99.2%), whereas hand errors were predominantly misses (95.7%), consistent with the Midas Touch account. Of the two adaptive interventions, only gaze declutter executed in this dataset; it modestly reduced timeouts but not slips. Hand width inflation was not evaluable due to a UI integration bug. These findings reveal modality-specific failure modes with direct implications for adaptive policy design, and establish the platform as a reproducible infrastructure for future studies. 

---
# Something from Nothing: Data Augmentation for Robust Severity Level Estimation of Dysarthric Speech 

**Authors**: Jaesung Bae, Xiuwen Zheng, Minje Kim, Chang D. Yoo, Mark Hasegawa-Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2603.15988)  

**Abstract**: Dysarthric speech quality assessment (DSQA) is critical for clinical diagnostics and inclusive speech technologies. However, subjective evaluation is costly and difficult to scale, and the scarcity of labeled data limits robust objective modeling. To address this, we propose a three-stage framework that leverages unlabeled dysarthric speech and large-scale typical speech datasets to scale training. A teacher model first generates pseudo-labels for unlabeled samples, followed by weakly supervised pretraining using a label-aware contrastive learning strategy that exposes the model to diverse speakers and acoustic conditions. The pretrained model is then fine-tuned for the downstream DSQA task. Experiments on five unseen datasets spanning multiple etiologies and languages demonstrate the robustness of our approach. Our Whisper-based baseline significantly outperforms SOTA DSQA predictors such as SpICE, and the full framework achieves an average SRCC of 0.761 across unseen test datasets. 

---
# Aligning Paralinguistic Understanding and Generation in Speech LLMs via Multi-Task Reinforcement Learning 

**Authors**: Jingxiang Chen, Minseok Kim, Seong-Gyun Leem, Yin Huang, Rashi Rungta, Zhicheng Ouyang, Haibin Wu, Surya Teja Appini, Ankur Bansal, Yang Bai, Yue Liu, Florian Metze, Ahmed A Aly, Anuj Kumar, Ariya Rastrow, Zhaojiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.15981)  

**Abstract**: Speech large language models (LLMs) observe paralinguistic cues such as prosody, emotion, and non-verbal sounds--crucial for intent understanding. However, leveraging these cues faces challenges: limited training data, annotation difficulty, and models exploiting lexical shortcuts over paralinguistic signals. We propose multi-task reinforcement learning (RL) with chain-of-thought prompting that elicits explicit affective reasoning. To address data scarcity, we introduce a paralinguistics-aware speech LLM (PALLM) that jointly optimizes sentiment classification from audio and paralinguistics-aware response generation via a two-stage pipeline. Experiments demonstrate that our approach improves paralinguistics understanding over both supervised baselines and strong proprietary models (Gemini-2.5-Pro, GPT-4o-audio) by 8-12% on Expresso, IEMOCAP, and RAVDESS. The results show that modeling paralinguistic reasoning with multi-task RL is crucial for building emotionally intelligent speech LLMs. 

---
# Standardizing Medical Images at Scale for AI 

**Authors**: Callen MacPhee, Yiming Zhou, Koichiro Kishima, Bahram Jalali  

**Link**: [PDF](https://arxiv.org/pdf/2603.15980)  

**Abstract**: Deep learning has achieved remarkable success in medical image analysis, yet its performance remains highly sensitive to the heterogeneity of clinical data. Differences in imaging hardware, staining protocols, and acquisition conditions produce substantial domain shifts that degrade model generalization across institutions. Here we present a physics-based data preprocessing framework based on the PhyCV (Physics-Inspired Computer Vision) family of algorithms, which standardizes medical images through deterministic transformations derived from optical physics. The framework models images as spatially varying optical fields that undergo a virtual diffractive propagation followed by coherent phase detection. This process suppresses non-semantic variability such as color and illumination differences while preserving diagnostically relevant texture and structural features. When applied to histopathological images from the Camelyon17-WILDS benchmark, PhyCV preprocessing improves out-of-distribution breast-cancer classification accuracy from 70.8% (Empirical Risk Minimization baseline) to 90.9%, matching or exceeding data-augmentation and domain-generalization approaches at negligible computational cost. Because the transform is physically interpretable, parameterizable, and differentiable, it can be deployed as a fixed preprocessing stage or integrated into end-to-end learning. These results establish PhyCV as a generalizable data refinery for medical imaging-one that harmonizes heterogeneous datasets through first-principles physics, improving robustness, interpretability, and reproducibility in clinical AI systems. 

---
# 100x Cost & Latency Reduction: Performance Analysis of AI Query Approximation using Lightweight Proxy Models 

**Authors**: Yeounoh Chung, Rushabh Desai, Jian He, Yu Xiao, Thibaud Hottelier, Yves-Laurent Kom Samo, Pushkar Kadilkar, Xianshun Chen, Sam Idicula, Fatma Özcan, Alon Halevy, Yannis Papakonstantinou  

**Link**: [PDF](https://arxiv.org/pdf/2603.15970)  

**Abstract**: Several data warehouse and database providers have recently introduced extensions to SQL called AI Queries, enabling users to specify functions and conditions in SQL that are evaluated by LLMs, thereby broadening significantly the kinds of queries one can express over the combination of structured and unstructured data. LLMs offer remarkable semantic reasoning capabilities, making them an essential tool for complex and nuanced queries that blend structured and unstructured data. While extremely powerful, these AI queries can become prohibitively costly when invoked thousands of times.
This paper provides an extensive evaluation of a recent AI query approximation approach that enables low cost analytics and database applications to benefit from AI queries. The approach delivers >100x cost and latency reduction for the semantic filter (this http URL) operator and also important gains for semantic ranking (this http URL). The cost and performance gains come from utilizing cheap and accurate proxy models over embedding vectors. We show that despite the massive gains in latency and cost, these proxy models preserve accuracy and occasionally improve accuracy across various benchmark datasets, including the extended Amazon reviews benchmark that has 10M rows. We present an OLAP-friendly architecture within Google \textit{BigQuery} for this approach for purely online (ad hoc) queries, and a low-latency HTAP database-friendly architecture in \textit{AlloyDB} that could further improve the latency by moving the proxy model training offline. We present techniques that accelerate the proxy model training. 

---
# MoLoRA: Composable Specialization via Per-Token Adapter Routing 

**Authors**: Shrey Shah, Justin Wagle  

**Link**: [PDF](https://arxiv.org/pdf/2603.15965)  

**Abstract**: Multi-adapter serving systems route entire sequences to a single adapter, forcing a choice when requests span multiple domains. This assumption fails in two important settings: (1) multimodal generation, where text and image tokens require different adapters within the same sequence, and (2) mixed-capability requests like "write code to solve this equation," which need expertise from multiple specialized adapters. We introduce per-token routing, which routes individual tokens to adapters based on either vocabulary structure (for multimodal models) or learned gating (for semantic specialization). Per-token routing is provably optimal, achieving work N for N tokens versus K \cdot N for per-sequence routing with K adapter types. Our key contribution is MoLoRA (Mixture of LoRA), which enables composable specialization: load multiple domain-specific adapters and let a learned router select the appropriate adapter per-token. We demonstrate that specialization dramatically beats scale: MoLoRA enables Qwen3-1.7B to exceed Qwen3-8B across four reasoning benchmarks while being 4.7x smaller. This enables modular expertise at inference time: train focused LoRAs independently, combine them without retraining, and add new capabilities by simply loading new adapters. 

---
# ExpertGen: Scalable Sim-to-Real Expert Policy Learning from Imperfect Behavior Priors 

**Authors**: Zifan Xu, Ran Gong, Maria Vittoria Minniti, Ahmet Salih Gundogdu, Eric Rosen, Kausik Sivakumar, Riedana Yan, Zixing Wang, Di Deng, Peter Stone, Xiaohan Zhang, Karl Schmeckpeper  

**Link**: [PDF](https://arxiv.org/pdf/2603.15956)  

**Abstract**: Learning generalizable and robust behavior cloning policies requires large volumes of high-quality robotics data. While human demonstrations (e.g., through teleoperation) serve as the standard source for expert behaviors, acquiring such data at scale in the real world is prohibitively expensive. This paper introduces ExpertGen, a framework that automates expert policy learning in simulation to enable scalable sim-to-real transfer. ExpertGen first initializes a behavior prior using a diffusion policy trained on imperfect demonstrations, which may be synthesized by large language models or provided by humans. Reinforcement learning is then used to steer this prior toward high task success by optimizing the diffusion model's initial noise while keep original policy frozen. By keeping the pretrained diffusion policy frozen, ExpertGen regularizes exploration to remain within safe, human-like behavior manifolds, while also enabling effective learning with only sparse rewards. Empirical evaluations on challenging manipulation benchmarks demonstrate that ExpertGen reliably produces high-quality expert policies with no reward engineering. On industrial assembly tasks, ExpertGen achieves a 90.5% overall success rate, while on long-horizon manipulation tasks it attains 85% overall success, outperforming all baseline methods. The resulting policies exhibit dexterous control and remain robust across diverse initial configurations and failure states. To validate sim-to-real transfer, the learned state-based expert policies are further distilled into visuomotor policies via DAgger and successfully deployed on real robotic hardware. 

---
# MobileLLM-Flash: Latency-Guided On-Device LLM Design for Industry Scale 

**Authors**: Hanxian Huang, Igor Fedorov, Andrey Gromov, Bernard Beckerman, Naveen Suda, David Eriksson, Maximilian Balandat, Rylan Conway, Patrick Huber, Chinnadhurai Sankar, Ayushi Dalmia, Zechun Liu, Lemeng Wu, Tarek Elgamal, Adithya Sagar, Vikas Chandra, Raghuraman Krishnamoorthi  

**Link**: [PDF](https://arxiv.org/pdf/2603.15954)  

**Abstract**: Real-time AI experiences call for on-device large language models (OD-LLMs) optimized for efficient deployment on resource-constrained hardware. The most useful OD-LLMs produce near-real-time responses and exhibit broad hardware compatibility, maximizing user reach. We present a methodology for designing such models using hardware-in-the-loop architecture search under mobile latency constraints. This system is amenable to industry-scale deployment: it generates models deployable without custom kernels and compatible with standard mobile runtimes like Executorch. Our methodology avoids specialized attention mechanisms and instead uses attention skipping for long-context acceleration.
Our approach jointly optimizes model architecture (layers, dimensions) and attention pattern. To efficiently evaluate candidates, we treat each as a pruned version of a pretrained backbone with inherited weights, thereby achieving high accuracy with minimal continued pretraining. We leverage the low cost of latency evaluation in a staged process: learning an accurate latency model first, then searching for the Pareto-frontier across latency and quality.
This yields MobileLLM-Flash, a family of foundation models (350M, 650M, 1.4B) for efficient on-device use with strong capabilities, supporting up to 8k context length. MobileLLM-Flash delivers up to 1.8x and 1.6x faster prefill and decode on mobile CPUs with comparable or superior quality. Our analysis of Pareto-frontier design choices offers actionable principles for OD-LLM design. 

---
# A Family of LLMs Liberated from Static Vocabularies 

**Authors**: Aleph Alpha, Adnen Abdessaied, Artur Baranowski, Lukas Balles, Michael Barlow, Fabien C. Y. Benureau, Felix Berkenkamp, Lukas Bluebaum, Bastian Boll, Thomas F. Burns, Björn Deiseroth, Constantin Eichenberg, David Friede, Pablo Iyu Guerrero, Ahmed Hammam, Bastian Harren, Johann Higl, Yasser Jadidi, Carina Kauf, Johannes Messner, Jan Hendrik Metzen, Max Meuer, Vedant Nanda, Pit Neitemeier, Koen Oostermeijer, Letitia Parcalabescu, Markus Pernpointner, Felix Reinfurt, Dylan Rodriquez, Grégory Schott, Philipp Siedler, Martin Simonovsky, Till Speicher, Volker Stampa, Stephan Wäldchen, Samuel Weinbach, Gregor Ziegltrum  

**Link**: [PDF](https://arxiv.org/pdf/2603.15953)  

**Abstract**: Tokenization is a central component of natural language processing in current large language models (LLMs), enabling models to convert raw text into processable units. Although learned tokenizers are widely adopted, they exhibit notable limitations, including their large, fixed vocabulary sizes and poor adaptability to new domains or languages. We present a family of models with up to 70 billion parameters based on the hierarchical autoregressive transformer (HAT) architecture. In HAT, an encoder transformer aggregates bytes into word embeddings and then feeds them to the backbone, a classical autoregressive transformer. The outputs of the backbone are then cross-attended by the decoder and converted back into bytes. We show that we can reuse available pre-trained models by converting the Llama 3.1 8B and 70B models into the HAT architecture: Llama-3.1-8B-TFree-HAT and Llama-3.1-70B-TFree-HAT are byte-level models whose encoder and decoder are trained from scratch, but where we adapt the pre-trained Llama backbone, i.e., the transformer blocks with the embedding matrix and head removed, to handle word embeddings instead of the original tokens. We also provide a 7B HAT model, Llama-TFree-HAT-Pretrained, trained entirely from scratch on nearly 4 trillion words. The HAT architecture improves text compression by reducing the number of required sequence positions and enhances robustness to intra-word variations, e.g., spelling differences. Through pre-training, as well as subsequent supervised fine-tuning and direct preference optimization in English and German, we show strong proficiency in both languages, improving on the original Llama 3.1 in most benchmarks. We release our models (including 200 pre-training checkpoints) on Hugging Face. 

---
# Data-Local Autonomous LLM-Guided Neural Architecture Search for Multiclass Multimodal Time-Series Classification 

**Authors**: Emil Hardarson, Luka Biedebach, Ómar Bessi Ómarsson, Teitur Hrólfsson, Anna Sigridur Islind, María Óskarsdóttir  

**Link**: [PDF](https://arxiv.org/pdf/2603.15939)  

**Abstract**: Applying machine learning to sensitive time-series data is often bottlenecked by the iteration loop: Performance depends strongly on preprocessing and architecture, yet training often has to run on-premise under strict data-local constraints. This is a common problem in healthcare and other privacy-constrained domains (e.g., a hospital developing deep learning models on patient EEG). This bottleneck is particularly challenging in multimodal fusion, where sensor modalities must be individually preprocessed and then combined. LLM-guided neural architecture search (NAS) can automate this exploration, but most existing workflows assume cloud execution or access to data-derived artifacts that cannot be exposed.
We present a novel data-local, LLM-guided search framework that handles candidate pipelines remotely while executing all training and evaluation locally under a fixed protocol. The controller observes only trial-level summaries, such as pipeline descriptors, metrics, learning-curve statistics, and failure logs, without ever accessing raw samples or intermediate feature representations. Our framework targets multiclass, multimodal learning via one-vs-rest binary experts per class and modality, a lightweight fusion MLP, and joint search over expert architectures and modality-specific preprocessing.
We evaluate our method on two regimes: UEA30 (public multivariate time-series classification dataset) and SleepEDFx sleep staging (heterogeneous clinical modalities such as EEG, EOG, and EMG). The results show that the modular baseline model is strong, and the LLM-guided NAS further improves it. Notably, our method finds models that perform within published ranges across most benchmark datasets. Across both settings, our method reduces manual intervention by enabling unattended architecture search while keeping sensitive data on-premise. 

---
# Evaluating Causal Discovery Algorithms for Path-Specific Fairness and Utility in Healthcare 

**Authors**: Nitish Nagesh, Elahe Khatibi, Thomas Hughes, Mahdi Bagheri, Pratik Gajane, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2603.15926)  

**Abstract**: Causal discovery in health data faces evaluation challenges when ground truth is unknown. We address this by collaborating with experts to construct proxy ground-truth graphs, establishing benchmarks for synthetic Alzheimer's disease and heart failure clinical records data. We evaluate the Peter-Clark, Greedy Equivalence Search, and Fast Causal Inference algorithms on structural recovery and path-specific fairness decomposition, going beyond composite fairness scores. On synthetic data, Peter-Clark achieved the best structural recovery. On heart failure data, Fast Causal Inference achieved the highest utility. For path-specific effects, ejection fraction contributed 3.37 percentage points to the indirect effect in the ground truth. These differences drove variations in the fairness-utility ratio across algorithms. Our results highlight the need for graph-aware fairness evaluation and fine-grained path-specific analysis when deploying causal discovery in clinical applications. 

---
# VIBEPASS: Can Vibe Coders Really Pass the Vibe Check? 

**Authors**: Srijan Bansal, Jiao Fangkai, Yilun Zhou, Austin Xu, Shafiq Joty, Semih Yavuz  

**Link**: [PDF](https://arxiv.org/pdf/2603.15921)  

**Abstract**: As Large Language Models shift the programming toward human-guided ''vibe coding'', agentic coding tools increasingly rely on models to self-diagnose and repair their own subtle faults -- a capability central to autonomous software engineering yet never systematically evaluated. We present \name{}, the first empirical decomposition that jointly evaluates two coupled tasks: \emph{Fault-Triggering Test Generation (FT-Test)} constructing a discriminative witness that exposes a latent bug, and \emph{Fault-targeted Program Repair (FPR)}, repairing it under varying diagnostic conditions. \name{} pairs competitive programming problems with LLM-generated solutions that pass partial test suites but fail on semantic edge cases, enabling controlled identification of where the diagnostic chain breaks down.
Evaluating 12 frontier LLMs, we find that fault-targeted reasoning does not scale with general coding ability. Models produce syntactically valid test inputs at near-ceiling rates yet collapse on discriminative generation, with fault hypothesis generation -- not output validation -- as the dominant bottleneck. Test-guided repair reveals a complementary insight: when self-generated tests successfully witness a fault, the resulting repair matches or outperforms repair guided by externally provided tests, but tests that fail to witness the fault actively degrade repair below unguided baselines. Together, these results reframe the challenge of autonomous debugging: the binding bottleneck is not code synthesis or test validity but fault-target reasoning, a capability that remains deficient across all frontier models. As Large Language Models shift the programming toward human-guided ''vibe coding'', agentic coding tools increasingly rely on models to self-diagnose and repair their own subtle faults -- a capability central to autonomous software engineering yet never systematically evaluated. 

---
# Auto Researching, not hyperparameter tuning: Convergence Analysis of 10,000 Experiments 

**Authors**: Xiaoyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.15916)  

**Abstract**: When LLM agents autonomously design ML experiments, do they perform genuine architecture search -- or do they default to hyperparameter tuning within a narrow region of the design space? We answer this question by analyzing 10,469 experiments executed by two LLM agents (Claude Opus and Gemini 2.5 Pro) across a combinatorial configuration space of 108,000 discrete cells for dashcam collision detection over 27 days. Through ANOVA decomposition, we find that \textbf{architectural choices explain 94\% of performance variance} ($F = 1324$, $\eta^2 = 0.94$), while hyperparameter variation within a fixed architecture explains only 6\%. Cross-task validation on a second collision dataset confirms this finding (75\% architecture-explained variance) with a \emph{different} winning backbone, confirming genuine architecture discovery. The agents' key contribution is discovering that V-JEPA\,2 video features with Zipformer temporal encoders achieve 0.9245 AP -- a configuration no human proposed -- and concentrating search on productive architectural regions: at $N = 50$, LLM-guided search reaches AP $= 0.985$ versus $0.965$ for from-scratch random search. Post-bugfix convergence follows a power law ($c = 0.11$, $R^2 = 0.93$); the low exponent reflects the cost of broad exploration, not inefficiency, since the LLM discovers qualitatively better regions than random or Bayesian baselines. We characterize multi-agent search dynamics via entropy cycles and Jensen--Shannon specialization, providing the first large-scale empirical framework for LLM-guided combinatorial ML experiment design. 

---
# The Agentic Researcher: A Practical Guide to AI-Assisted Research in Mathematics and Machine Learning 

**Authors**: Max Zimmer, Nico Pelleriti, Christophe Roux, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2603.15914)  

**Abstract**: AI tools and agents are reshaping how researchers work, from proving theorems to training neural networks. Yet for many, it remains unclear how these tools fit into everyday research practice. This paper is a practical guide to AI-assisted research in mathematics and machine learning: We discuss how researchers can use modern AI systems productively, where these systems help most, and what kinds of guardrails are needed to use them responsibly. It is organized into three parts: (I) a five-level taxonomy of AI integration, (II) an open-source framework that, through a set of methodological rules formulated as agent prompts, turns CLI coding agents (e.g., Claude Code, Codex CLI, OpenCode) into autonomous research assistants, and (III) case studies from deep learning and mathematics. The framework runs inside a sandboxed container, works with any frontier LLM through existing CLI agents, is simple enough to install and use within minutes, and scales from personal-laptop prototyping to multi-node, multi-GPU experimentation across compute clusters. In practice, our longest autonomous session ran for over 20 hours, dispatching independent experiments across multiple nodes without human intervention. We stress that our framework is not intended to replace the researcher in the loop, but to augment them. Our code is publicly available at this https URL. 

---
# Federated Learning for Privacy-Preserving Medical AI 

**Authors**: Tin Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15901)  

**Abstract**: This dissertation investigates privacy-preserving federated learning for Alzheimer's disease classification using three-dimensional MRI data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). Existing methodologies often suffer from unrealistic data partitioning, inadequate privacy guarantees, and insufficient benchmarking, limiting their practical deployment in healthcare. To address these gaps, this research proposes a novel site-aware data partitioning strategy that preserves institutional boundaries, reflecting real-world multi-institutional collaborations and data heterogeneity. Furthermore, an Adaptive Local Differential Privacy (ALDP) mechanism is introduced, dynamically adjusting privacy parameters based on training progression and parameter characteristics, thereby significantly improving the privacy-utility trade-off over traditional fixed-noise approaches. Systematic empirical evaluation across multiple client federations and privacy budgets demonstrated that advanced federated optimisation algorithms, particularly FedProx, could equal or surpass centralised training performance while ensuring rigorous privacy protection. Notably, ALDP achieved up to 80.4% accuracy in a two-client configuration, surpassing fixed-noise Local DP by 5-7 percentage points and demonstrating substantially greater training stability. The comprehensive ablation studies and benchmarking establish quantitative standards for privacy-preserving collaborative medical AI, providing practical guidelines for real-world deployment. This work thereby advances the state-of-the-art in federated learning for medical imaging, establishing both methodological foundations and empirical evidence necessary for future privacy-compliant AI adoption in healthcare. 

---
# The Internet of Physical AI Agents: Interoperability, Longevity, and the Cost of Getting It Wrong 

**Authors**: Roberto Morabito, Mallik Tatipamula  

**Link**: [PDF](https://arxiv.org/pdf/2603.15900)  

**Abstract**: The Internet has evolved by progressively expanding what humanity connects: first computers, then people, and later billions of devices through the Internet of Things (IoT). While IoT succeeded in digitizing perception at scale, it also exposed fundamental limitations, including fragmentation, weak security, limited autonomy, and poor long-term sustainability. Today, advances in edge hardware, sensing, connectivity, and artificial intelligence enable a new phase: the Internet of Physical AI Agents. Unlike IoT devices that primarily sense and report, Physical AI Agents perceive, reason, and act in real time, operating autonomously and cooperatively across safety-critical domains such as disaster response, healthcare, industrial automation, and mobility. However, embedding fast-evolving AI capabilities into long-lived physical infrastructure introduces new architectural risks, particularly around interoperability, lifecycle management, and premature ossification. This article revisits lessons from IoT and Internet evolution, and articulates design principles for building resilient, evolvable, and trustworthy agentic systems. We present an architectural blueprint encompassing agentic identity, secure agent-to-agent communication, semantic interoperability, policy-governed runtimes, and observability-driven governance. We argue that treating evolution, trust, and interoperability as first-class requirements is essential to avoid hard-coding today's assumptions into tomorrow's intelligent infrastructure, and to prevent the high technical and economic cost of getting it wrong. 

---
# COGNAC at SemEval-2026 Task 5: LLM Ensembles for Human-Level Word Sense Plausibility Rating in Challenging Narratives 

**Authors**: Azwad Anjum Islam, Tisa Islam Erana  

**Link**: [PDF](https://arxiv.org/pdf/2603.15897)  

**Abstract**: We describe our system for SemEval-2026 Task 5, which requires rating the plausibility of given word senses of homonyms in short stories on a 5-point Likert scale. Systems are evaluated by the unweighted average of accuracy (within one standard deviation of mean human judgments) and Spearman Rank Correlation. We explore three prompting strategies using multiple closed-source commercial LLMs: (i) a baseline zero-shot setup, (ii) Chain-of-Thought (CoT) style prompting with structured reasoning, and (iii) a comparative prompting strategy for evaluating candidate word senses simultaneously. Furthermore, to account for the substantial inter-annotator variation present in the gold labels, we propose an ensemble setup by averaging model predictions. Our best official system, comprising an ensemble of LLMs across all three prompting strategies, placed 4th on the competition leaderboard with 0.88 accuracy and 0.83 Spearman's rho (0.86 average). Post-competition experiments with additional models further improved this performance to 0.92 accuracy and 0.85 Spearman's rho (0.89 average). We find that comparative prompting consistently improved performance across model families, and model ensembling significantly enhanced alignment with mean human judgments, suggesting that LLM ensembles are especially well suited for subjective semantic evaluation tasks involving multiple annotators. 

---
# PhasorFlow: A Python Library for Unit Circle Based Computing 

**Authors**: Dibakar Sigdel, Namuna Panday  

**Link**: [PDF](https://arxiv.org/pdf/2603.15886)  

**Abstract**: We present PhasorFlow, an open-source Python library introducing a computational paradigm operating on the $S^1$ unit circle. Inputs are encoded as complex phasors $z = e^{i\theta}$ on the $N$-Torus ($\mathbb{T}^N$). As computation proceeds via unitary wave interference gates, global norm is preserved while individual components drift into $\mathbb{C}^N$, allowing algorithms to natively leverage continuous geometric gradients for predictive learning. PhasorFlow provides three core contributions. First, we formalize the Phasor Circuit model ($N$ unit circle threads, $M$ gates) and introduce a 22-gate library covering Standard Unitary, Non-Linear, Neuromorphic, and Encoding operations with full matrix algebra simulation. Second, we present the Variational Phasor Circuit (VPC), analogous to Variational Quantum Circuits (VQC), enabling optimization of continuous phase parameters for classical machine learning tasks. Third, we introduce the Phasor Transformer, replacing expensive $QK^TV$ attention with a parameter-free, DFT-based token mixing layer inspired by FNet. We validate PhasorFlow on non-linear spatial classification, time-series prediction, financial volatility detection, and neuromorphic tasks including neural binding and oscillatory associative memory. Our results establish unit circle computing as a deterministic, lightweight, and mathematically principled alternative to classical neural networks and quantum circuits. It operates on classical hardware while sharing quantum mechanics' unitary foundations. PhasorFlow is available at this https URL. 

---
# Electrodermal Activity as a Unimodal Signal for Aerobic Exercise Detection in Wearable Sensors 

**Authors**: Rena Mira Krishna, Ramya Sankar, Shadi Ghiasi  

**Link**: [PDF](https://arxiv.org/pdf/2603.15880)  

**Abstract**: Electrodermal Activity (EDA) is a non-invasive physiological signal widely available in wearable devices and reflects sympathetic nervous system (SNS) activation. Prior multi-modal studies have demonstrated robust performance in distinguishing stress and exercise states when EDA is combined with complementary signals such as heart rate and accelerometry. However, the ability of EDA to independently distinguish sustained aerobic exercise from low-arousal states under subject-independent evaluation remains insufficiently characterized.
This study investigates whether features derived exclusively from EDA can reliably differentiate rest from sustained aerobic exercise. Using a publicly available dataset collected from thirty healthy individuals, EDA features were evaluated using benchmark machine learning models with leave-one-subject-out (LOSO) validation. Across models, EDA-only classifiers achieved moderate subject-independent performance, with phasic temporal dynamics and event timing contributing to class separation. Rather than proposing EDA as a replacement for multimodal sensing, this work provides a conservative benchmark of the discriminative power of EDA alone and clarifies its role as a unimodal input for wearable activity-state inference. 

---
# Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning 

**Authors**: Ezgi Korkmaz  

**Link**: [PDF](https://arxiv.org/pdf/2603.15871)  

**Abstract**: Following the pivotal success of learning strategies to win at tasks, solely by interacting with an environment without any supervision, agents have gained the ability to make sequential decisions in complex MDPs. Yet, reinforcement learning policies face exponentially growing state spaces in high dimensional MDPs resulting in a dichotomy between computational complexity and policy success. In our paper we focus on the agent's interaction with the environment in a high-dimensional MDP during the learning phase and we introduce a theoretically-founded novel paradigm based on experiences obtained through counteractive actions. Our analysis and method provide a theoretical basis for efficient, effective, scalable and accelerated learning, and further comes with zero additional computational complexity while leading to significant acceleration in training. We conduct extensive experiments in the Arcade Learning Environment with high-dimensional state representation MDPs. The experimental results further verify our theoretical analysis, and our method achieves significant performance increase with substantial sample-efficiency in high-dimensional environments. 

---
# Interpretative Interfaces: Designing for AI-Mediated Reading Practices and the Knowledge Commons 

**Authors**: Gabrielle Benabdallah  

**Link**: [PDF](https://arxiv.org/pdf/2603.15863)  

**Abstract**: Explainable AI (XAI) interfaces seek to make large language models more transparent, yet explanation alone does not produce understanding. Explaining a system's behavior is not the same as being able to engage with it, to probe and interpret its operations through direct manipulation. This distinction matters for scientific disciplines in particular: scientists who increasingly rely on LLMs for reading, citing, and producing literature reviews have little means of directly engaging with how these models process and transform the texts they generate. In this ongoing design research project, I argue for a shift from explainability to interpretative engagement. This shift moves away from accounts of system behavior to instead enable users to manipulate a model's intermediate representations. Drawing on textual scholarship, computational poetics, and the history of reading and writing technologies, including practices such as marginalia, glosses, indices, and annotation systems, I propose interpretative interfaces as interactive environments in which non-expert users can intervene in the representational space of a language model. More specifically, such interfaces will allow users to select a token and follow its trajectory through the model's intermediate layers. This way, they can observe how its semantic position shifts as context is processed, and possibly annotate the transformations they find useful or meaningful. The same way readers can create their own maps within a book through annotations and bookmarks, interpretative interfaces will allow users to inscribe their reading of a model's internal representations. The goal of this project is to reframe AI interpretability as an interaction design project rather than a purely technical one, and to open a path toward AI-mediated reading that supports interpretative engagement and critical stewardship of scientific knowledge. 

---
# FlashSampling: Fast and Memory-Efficient Exact Sampling 

**Authors**: Tomas Ruiz, Zhen Qin, Yifan Zhang, Xuyang Shen, Yiran Zhong, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15854)  

**Abstract**: Sampling from a categorical distribution is mathematically simple, but in large-vocabulary decoding, it often triggers extra memory traffic and extra kernels after the LM head. We present FlashSampling, an exact sampling primitive that fuses sampling into the LM-head matmul and never materializes the logits tensor in HBM. The method is simple: compute logits tile-by-tile on chip, add Gumbel noise, keep only one maximizer per row and per vocabulary tile, and finish with a small reduction over tiles. The fused tiled kernel is exact because $\argmax$ decomposes over a partition; grouped variants for online and tensor-parallel settings are exact by hierarchical factorization of the categorical distribution. Across H100, H200, B200, and B300 GPUs, FlashSampling speeds up kernel-level decode workloads, and in end-to-end vLLM experiments, it reduces time per output token by up to $19%$ on the models we test. These results show that exact sampling, with no approximation, can be integrated into the matmul itself, turning a bandwidth-bound postprocessing step into a lightweight epilogue. Project Page: this https URL. 

---
# Informationally Compressive Anonymization: Non-Degrading Sensitive Input Protection for Privacy-Preserving Supervised Machine Learning 

**Authors**: Jeremy J Samuelson  

**Link**: [PDF](https://arxiv.org/pdf/2603.15842)  

**Abstract**: Modern machine learning systems increasingly rely on sensitive data, creating significant privacy, security, and regulatory risks that existing privacy-preserving machine learning (ppML) techniques, such as Differential Privacy (DP) and Homomorphic Encryption (HE), address only at the cost of degraded performance, increased complexity, or prohibitive computational overhead. This paper introduces Informationally Compressive Anonymization (ICA) and the VEIL architecture, a privacy-preserving ML framework that achieves strong privacy guarantees through architectural and mathematical design rather than noise injection or cryptography. ICA embeds a supervised, multi-objective encoder within a trusted Source Environment to transform raw inputs into low-dimensional, task-aligned latent representations, ensuring that only irreversibly anonymized vectors are exported to untrusted Training and Inference Environments. The paper rigorously proves that these encodings are structurally non-invertible using topological and information-theoretic arguments, showing that inversion is logically impossible, even under idealized attacker assumptions, and that, in realistic deployments, the attackers conditional entropy over the original data diverges, driving reconstruction probability to zero. Unlike prior autoencoder-based ppML approaches, ICA preserves predictive utility by aligning representation learning with downstream supervised objectives, enabling low-latency, high-performance ML without gradient clipping, noise budgets, or encryption at inference time. The VEIL architecture enforces strict trust boundaries, supports scalable multi-region deployment, and naturally aligns with privacy-by-design regulatory frameworks, establishing a new foundation for enterprise ML that is secure, performant, and safe by construction, even in the face of post-quantum threats. 

---
# When Stability Fails: Hidden Failure Modes Of LLMS in Data-Constrained Scientific Decision-Making 

**Authors**: Nazia Riasat  

**Link**: [PDF](https://arxiv.org/pdf/2603.15840)  

**Abstract**: Large language models (LLMs) are increasingly used as decision-support tools in data-constrained scientific workflows, where correctness and validity are critical. However, evaluation practices often emphasize stability or reproducibility across repeated runs. While these properties are desirable, stability alone does not guar- antee agreement with statistical ground truth when such references are available. We introduce a controlled behavioral evaluation framework that explicitly sep- arates four dimensions of LLM decision-making: stability, correctness, prompt sensitivity, and output validity under fixed statistical inputs. We evaluate multi- ple LLMs using a statistical gene prioritization task derived from differential ex- pression analysis across prompt regimes involving strict and relaxed significance thresholds, borderline ranking scenarios, and minor wording variations. Our ex- periments show that LLMs can exhibit near-perfect run-to-run stability while sys- tematically diverging from statistical ground truth, over-selecting under relaxed thresholds, responding sharply to minor prompt wording changes, or producing syntactically plausible gene identifiers absent from the input table. Although sta- bility reflects robustness across repeated runs, it does not guarantee agreement with statistical ground truth in structured scientific decision tasks. These findings highlight the importance of explicit ground-truth validation and output validity checks when deploying LLMs in automated or semi-automated scientific work- flows. 

---
# Hypothesis Class Determines Explanation: Why Accurate Models Disagree on Feature Attribution 

**Authors**: Thackshanaramana B  

**Link**: [PDF](https://arxiv.org/pdf/2603.15821)  

**Abstract**: The assumption that prediction-equivalent models produce equivalent explanations underlies many practices in explainable AI, including model selection, auditing, and regulatory evaluation. In this work, we show that this assumption does not hold. Through a large-scale empirical study across 24 datasets and multiple model classes, we find that models with identical predictive behavior can produce substantially different feature attributions.
This disagreement is highly structured: models within the same hypothesis class exhibit strong agreement, while cross-class pairs (e.g., tree-based vs. linear) trained on identical data splits show substantially reduced agreement, consistently near or below the lottery threshold. We identify hypothesis class as the structural driver of this phenomenon, which we term the Explanation Lottery.
We theoretically show that the resulting Agreement Gap persists under interaction structure in the data-generating process. This structural finding motivates a post-hoc diagnostic, the Explanation Reliability Score R(x), which predicts when explanations are stable across architectures without additional training.
Our results demonstrate that model selection is not explanation-neutral: the hypothesis class chosen for deployment can determine which features are attributed responsibility for a decision. 

---
# Don't Trust Stubborn Neighbors: A Security Framework for Agentic Networks 

**Authors**: Samira Abedini, Sina Mavali, Lea Schönherr, Martin Pawelczyk, Rebekka Burkholz  

**Link**: [PDF](https://arxiv.org/pdf/2603.15809)  

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MASs) are increasingly deployed for agentic tasks, such as web automation, itinerary planning, and collaborative problem solving. Yet, their interactive nature introduces new security risks: malicious or compromised agents can exploit communication channels to propagate misinformation and manipulate collective outcomes.
In this paper, we study how such manipulation can arise and spread by borrowing the Friedkin-Johnsen opinion formation model from social sciences to propose a general theoretical framework to study LLM-MAS. Remarkably, this model closely captures LLM-MAS behavior, as we verify in extensive experiments across different network topologies and attack and defense scenarios. Theoretically and empirically, we find that a single highly stubborn and persuasive agent can take over MAS dynamics, underscoring the systems' high susceptibility to attacks by triggering a persuasion cascade that reshapes collective opinion. Our theoretical analysis reveals three mechanisms to increase system security: a) increasing the number of benign agents, b) increasing the innate stubbornness or peer-resistance of agents, or c) reducing trust in potential adversaries. Because scaling is computationally expensive and high stubbornness degrades the network's ability to reach consensus, we propose a new mechanism to mitigate threats by a trust-adaptive defense that dynamically adjusts inter-agent trust to limit adversarial influence while maintaining cooperative performance. Extensive experiments confirm that this mechanism effectively defends against manipulation. 

---
# OMNIFLOW: A Physics-Grounded Multimodal Agent for Generalized Scientific Reasoning 

**Authors**: Hao Wu, Yongheng Zhang, Yuan Gao, Fan Xu, Fan Zhang, Ruobing Xie, Ruijian Gou, Yuxuan Liang, Xiaomeng Huang, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15797)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional logical reasoning capabilities but frequently struggle with the continuous spatiotemporal dynamics governed by Partial Differential Equations (PDEs), often resulting in non-physical hallucinations. Existing approaches typically resort to costly, domain-specific fine-tuning, which severely limits cross-domain generalization and interpretability. To bridge this gap, we propose OMNIFLOW, a neuro-symbolic architecture designed to ground frozen multimodal LLMs in fundamental physical laws without requiring domain-specific parameter updates. OMNIFLOW introduces a novel \textit{Semantic-Symbolic Alignment} mechanism that projects high-dimensional flow tensors into topological linguistic descriptors, enabling the model to perceive physical structures rather than raw pixel values. Furthermore, we construct a Physics-Guided Chain-of-Thought (PG-CoT) workflow that orchestrates reasoning through dynamic constraint injection (e.g., mass conservation) and iterative reflexive verification. We evaluate OMNIFLOW on a comprehensive benchmark spanning microscopic turbulence, theoretical Navier-Stokes equations, and macroscopic global weather forecasting. Empirical results demonstrate that OMNIFLOW significantly outperforms traditional deep learning baselines in zero-shot generalization and few-shot adaptation tasks. Crucially, it offers transparent, physically consistent reasoning reports, marking a paradigm shift from black-box fitting to interpretable scientific reasoning. 

---
# Parallelised Differentiable Straightest Geodesics for 3D Meshes 

**Authors**: Hippolyte Verninas, Caner Korkmaz, Stefanos Zafeiriou, Tolga Birdal, Simone Foti  

**Link**: [PDF](https://arxiv.org/pdf/2603.15780)  

**Abstract**: Machine learning has been progressively generalised to operate within non-Euclidean domains, but geometrically accurate methods for learning on surfaces are still falling behind. The lack of closed-form Riemannian operators, the non-differentiability of their discrete counterparts, and poor parallelisation capabilities have been the main obstacles to the development of the field on meshes. A principled framework to compute the exponential map on Riemannian surfaces discretised as meshes is straightest geodesics, which also allows to trace geodesics and parallel-transport vectors as a by-product. We provide a parallel GPU implementation and derive two different methods for differentiating through the straightest geodesics, one leveraging an extrinsic proxy function and one based upon a geodesic finite differences scheme. After proving our parallelisation performance and accuracy, we demonstrate how our differentiable exponential map can improve learning and optimisation pipelines on general geometries. In particular, to showcase the versatility of our method, we propose a new geodesic convolutional layer, a new flow matching method for learning on meshes, and a second-order optimiser that we apply to centroidal Voronoi tessellation. Our code, models, and pip-installable library (digeo) are available at: this http URL. 

---
# Morphemes Without Borders: Evaluating Root-Pattern Morphology in Arabic Tokenizers and LLMs 

**Authors**: Yara Alakeel, Chatrine Qwaider, Hanan Aldarmaki, Sawsan Alqahtani  

**Link**: [PDF](https://arxiv.org/pdf/2603.15773)  

**Abstract**: This work investigates how effectively large language models (LLMs) and their tokenization schemes represent and generate Arabic root-pattern morphology, probing whether they capture genuine morphological structure or rely on surface memorization. Arabic morphological system provides a rich testbed for analyzing how LLMs handle complex, non-concatenative forms and how tokenization choices influence this process. Our study begins with an evaluation of morphological fidelity across Arabic and multilingual tokenizers against gold-standard segmentation, followed by an analysis of LLM performance in productive root-pattern generation using a newly developed test set. Our findings across seven Arabic-centric and multilingual LLMs and their respective tokenizers reveal that tokenizer morphological alignment is not necessary nor sufficient for morphological generation, which questions the role of morphological tokenization in downstream performance. 

---
# CorrectionPlanner: Self-Correction Planner with Reinforcement Learning in Autonomous Driving 

**Authors**: Yihong Guo, Dongqiangzi Ye, Sijia Chen, Anqi Liu, Xianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15771)  

**Abstract**: Autonomous driving requires safe planning, but most learning-based planners lack explicit self-correction ability: once an unsafe action is proposed, there is no mechanism to correct it. Thus, we propose CorrectionPlanner, an autoregressive planner with self-correction that models planning as motion-token generation within a propose, evaluate, and correct loop. At each planning step, the policy proposes an action, namely a motion token, and a learned collision critic predicts whether it will induce a collision within a short horizon. If the critic predicts a collision, we retain the sequence of historical unsafe motion tokens as a self-correction trace, generate the next motion token conditioned on it, and repeat this process until a safe motion token is proposed or the safety criterion is met. This self-correction trace, consisting of all unsafe motion tokens, represents the planner's correction process in motion-token space, analogous to a reasoning trace in language models. We train the planner with imitation learning followed by model-based reinforcement learning using rollouts from a pretrained world model that realistically models agents' reactive behaviors. Closed-loop evaluations show that CorrectionPlanner reduces collision rate by over 20% on Waymax and achieves state-of-the-art planning scores on nuPlan. 

---
# Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation 

**Authors**: Jacob Levy, Tyler Westenbroek, Kevin Huang, Fernando Palafox, Patrick Yin, Shayegan Omidshafiei, Dong-Ki Kim, Abhishek Gupta, David Fridovich-Keil  

**Link**: [PDF](https://arxiv.org/pdf/2603.15759)  

**Abstract**: Simulation-to-real transfer remains a central challenge in robotics, as mismatches between simulated and real-world dynamics often lead to failures. While reinforcement learning offers a principled mechanism for adaptation, existing sim-to-real finetuning methods struggle with exploration and long-horizon credit assignment in the low-data regimes typical of real-world robotics. We introduce Simulation Distillation (SimDist), a sim-to-real framework that distills structural priors from a simulator into a latent world model and enables rapid real-world adaptation via online planning and supervised dynamics finetuning. By transferring reward and value models directly from simulation, SimDist provides dense planning signals from raw perception without requiring value learning during deployment. As a result, real-world adaptation reduces to short-horizon system identification, avoiding long-horizon credit assignment and enabling fast, stable improvement. Across precise manipulation and quadruped locomotion tasks, SimDist substantially outperforms prior methods in data efficiency, stability, and final performance. Project website and code: this https URL 

---
# You've Got a Golden Ticket: Improving Generative Robot Policies With A Single Noise Vector 

**Authors**: Omkar Patil, Ondrej Biza, Thomas Weng, Karl Schmeckpeper, Wil Thomason, Xiaohan Zhang, Robin Walters, Nakul Gopalan, Sebastian Castro, Eric Rosen  

**Link**: [PDF](https://arxiv.org/pdf/2603.15757)  

**Abstract**: What happens when a pretrained generative robot policy is provided a constant initial noise as input, rather than repeatedly sampling it from a Gaussian? We demonstrate that the performance of a pretrained, frozen diffusion or flow matching policy can be improved with respect to a downstream reward by swapping the sampling of initial noise from the prior distribution (typically isotropic Gaussian) with a well-chosen, constant initial noise input -- a golden ticket. We propose a search method to find golden tickets using Monte-Carlo policy evaluation that keeps the pretrained policy frozen, does not train any new networks, and is applicable to all diffusion/flow matching policies (and therefore many VLAs). Our approach to policy improvement makes no assumptions beyond being able to inject initial noise into the policy and calculate (sparse) task rewards of episode rollouts, making it deployable with no additional infrastructure or models. Our method improves the performance of policies in 38 out of 43 tasks across simulated and real-world robot manipulation benchmarks, with relative improvements in success rate by up to 58% for some simulated tasks, and 60% within 50 search episodes for real-world tasks. We also show unique benefits of golden tickets for multi-task settings: the diversity of behaviors from different tickets naturally defines a Pareto frontier for balancing different objectives (e.g., speed, success rates); in VLAs, we find that a golden ticket optimized for one task can also boost performance in other related tasks. We release a codebase with pretrained policies and golden tickets for simulation benchmarks using VLAs, diffusion policies, and flow matching policies. 

---
# ClawWorm: Self-Propagating Attacks Across LLM Agent Ecosystems 

**Authors**: Yihao Zhang, Zeming Wei, Xiaokun Luan, Chengcan Wu, Zhixin Zhang, Jiangrong Wu, Haolin Wu, Huanran Chen, Jun Sun, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.15727)  

**Abstract**: Autonomous LLM-based agents increasingly operate as long-running processes forming densely interconnected multi-agent ecosystems, whose security properties remain largely unexplored. In particular, OpenClaw, an open-source platform with over 40{,}000 active instances, has stood out recently with its persistent configurations, tool-execution privileges, and cross-platform messaging capabilities. In this work, we present ClawWorm, the first self-replicating worm attack against a production-scale agent framework, achieving a fully autonomous infection cycle initiated by a single message: the worm first hijacks the victim's core configuration to establish persistent presence across session restarts, then executes an arbitrary payload upon each reboot, and finally propagates itself to every newly encountered peer without further attacker intervention. We evaluate the attack on a controlled testbed across three distinct infection vectors and three payload types, demonstrating high success rates in end-to-end infection, sustained multi-hop propagation, and payload independence from the worm mechanism. We analyse the architectural root causes underlying these vulnerabilities and propose defence strategies targeting each identified trust boundary. Code and samples will be released upon completion of responsible disclosure. 

---
# MiroThinker-1.7 & H1: Towards Heavy-Duty Research Agents via Verification 

**Authors**: MiroMind Team, S. Bai, L. Bing, L. Lei, R. Li, X. Li, X. Lin, E. Min, L. Su, B. Wang, L. Wang, L. Wang, S. Wang, X. Wang, Y. Zhang, Z. Zhang, G. Chen, L. Chen, Z. Cheng, Y. Deng, Z. Huang, D. Ng, J. Ni, Q. Ren, X. Tang, B.L. Wang, H. Wang, N. Wang, C. Wei, Q. Wu, J. Xia, Y. Xiao, H. Xu, X. Xu, C. Xue, Z. Yang, Z. Yang, F. Ye, H. Ye, J. Yu, C. Zhang, W. Zhang, H. Zhao, P. Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15726)  

**Abstract**: We present MiroThinker-1.7, a new research agent designed for complex long-horizon reasoning tasks. Building on this foundation, we further introduce MiroThinker-H1, which extends the agent with heavy-duty reasoning capabilities for more reliable multi-step problem solving. In particular, MiroThinker-1.7 improves the reliability of each interaction step through an agentic mid-training stage that emphasizes structured planning, contextual reasoning, and tool interaction. This enables more effective multi-step interaction and sustained reasoning across complex tasks. MiroThinker-H1 further incorporates verification directly into the reasoning process at both local and global levels. Intermediate reasoning decisions can be evaluated and refined during inference, while the overall reasoning trajectory is audited to ensure that final answers are supported by coherent chains of evidence. Across benchmarks covering open-web research, scientific reasoning, and financial analysis, MiroThinker-H1 achieves state-of-the-art performance on deep research tasks while maintaining strong results on specialized domains. We also release MiroThinker-1.7 and MiroThinker-1.7-mini as open-source models, providing competitive research-agent capabilities with significantly improved efficiency. 

---
# Meta-TTRL: A Metacognitive Framework for Self-Improving Test-Time Reinforcement Learning in Unified Multimodal Models 

**Authors**: Lit Sin Tan, Junzhe Chen, Xiaolong Fu, Lichen Ma, Junshi Huang, Jianzhong Shi, Yan Li, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.15724)  

**Abstract**: Existing test-time scaling (TTS) methods for unified multimodal models (UMMs) in text-to-image (T2I) generation primarily rely on search or sampling strategies that produce only instance-level improvements, limiting the ability to learn from prior inferences and accumulate knowledge across similar prompts. To overcome these limitations, we propose Meta-TTRL, a metacognitive test-time reinforcement learning framework. Meta-TTRL performs test-time parameter optimization guided by model-intrinsic monitoring signals derived from the meta-knowledge of UMMs, achieving self-improvement and capability-level improvement at test time. Extensive experiments demonstrate that Meta-TTRL generalizes well across three representative UMMs, including Janus-Pro-7B, BAGEL, and Qwen-Image, achieving significant gains on compositional reasoning tasks and multiple T2I benchmarks with limited data. We provide the first comprehensive analysis to investigate the potential of test-time reinforcement learning (TTRL) for T2I generation in UMMs. Our analysis further reveals a key insight underlying effective TTRL: metacognitive synergy, where monitoring signals align with the model's optimization regime to enable self-improvement. 

---
# A Framework and Prototype for a Navigable Map of Datasets in Engineering Design and Systems Engineering 

**Authors**: H. Sinan Bank, Daniel R. Herber  

**Link**: [PDF](https://arxiv.org/pdf/2603.15722)  

**Abstract**: The proliferation of data across the system lifecycle presents both a significant opportunity and a challenge for Engineering Design and Systems Engineering (EDSE). While this ``digital thread'' has the potential to drive innovation, the fragmented and inaccessible nature of existing datasets hinders method validation, limits reproducibility, and slows research progress. Unlike fields such as computer vision and natural language processing, which benefit from established benchmark ecosystems, engineering design research often relies on small, proprietary, or ad-hoc datasets. This paper addresses this challenge by proposing a systematic framework for a ``Map of Datasets in EDSE.'' The framework is built upon a multi-dimensional taxonomy designed to classify engineering datasets by domain, lifecycle stage, data type, and format, enabling faceted discovery. An architecture for an interactive discovery tool is detailed and demonstrated through a working prototype, employing a knowledge graph data model to capture rich semantic relationships between datasets, tools, and publications. An analysis of the current data landscape reveals underrepresented areas (``data deserts'') in early-stage design and system architecture, as well as relatively well-represented areas (``data oases'') in predictive maintenance and autonomous systems. The paper identifies key challenges in curation and sustainability and proposes mitigation strategies, laying the groundwork for a dynamic, community-driven resource to accelerate data-centric engineering research. 

---
# How Vulnerable Are AI Agents to Indirect Prompt Injections? Insights from a Large-Scale Public Competition 

**Authors**: Mateusz Dziemian, Maxwell Lin, Xiaohan Fu, Micha Nowak, Nick Winter, Eliot Jones, Andy Zou, Lama Ahmad, Kamalika Chaudhuri, Sahana Chennabasappa, Xander Davies, Lauren Deason, Benjamin L. Edelman, Tanner Emek, Ivan Evtimov, Jim Gust, Maia Hamin, Kat He, Klaudia Krawiecka, Riccardo Patana, Neil Perry, Troy Peterson, Xiangyu Qi, Javier Rando, Zifan Wang, Zihan Wang, Spencer Whitman, Eric Winsor, Arman Zharmagambetov, Matt Fredrikson, Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2603.15714)  

**Abstract**: LLM based agents are increasingly deployed in high stakes settings where they process external data sources such as emails, documents, and code repositories. This creates exposure to indirect prompt injection attacks, where adversarial instructions embedded in external content manipulate agent behavior without user awareness. A critical but underexplored dimension of this threat is concealment: since users tend to observe only an agent's final response, an attack can conceal its existence by presenting no clue of compromise in the final user facing response while successfully executing harmful actions. This leaves users unaware of the manipulation and likely to accept harmful outcomes as legitimate. We present findings from a large scale public red teaming competition evaluating this dual objective across three agent settings: tool calling, coding, and computer use. The competition attracted 464 participants who submitted 272000 attack attempts against 13 frontier models, yielding 8648 successful attacks across 41 scenarios. All models proved vulnerable, with attack success rates ranging from 0.5% (Claude Opus 4.5) to 8.5% (Gemini 2.5 Pro). We identify universal attack strategies that transfer across 21 of 41 behaviors and multiple model families, suggesting fundamental weaknesses in instruction following architectures. Capability and robustness showed weak correlation, with Gemini 2.5 Pro exhibiting both high capability and high vulnerability. To address benchmark saturation and obsoleteness, we will endeavor to deliver quarterly updates through continued red teaming competitions. We open source the competition environment for use in evaluations, along with 95 successful attacks against Qwen that did not transfer to any closed source model. We share model-specific attack data with respective frontier labs and the full dataset with the UK AISI and US CAISI to support robustness research. 

---
# Embedding-Aware Feature Discovery: Bridging Latent Representations and Interpretable Features in Event Sequences 

**Authors**: Artem Sakhno, Ivan Sergeev, Alexey Shestov, Omar Zoloev, Elizaveta Kovtun, Gleb Gusev, Andrey Savchenko, Maksim Makarenko  

**Link**: [PDF](https://arxiv.org/pdf/2603.15713)  

**Abstract**: Industrial financial systems operate on temporal event sequences such as transactions, user actions, and system logs. While recent research emphasizes representation learning and large language models, production systems continue to rely heavily on handcrafted statistical features due to their interpretability, robustness under limited supervision, and strict latency constraints. This creates a persistent disconnect between learned embeddings and feature-based pipelines. We introduce Embedding-Aware Feature Discovery (EAFD), a unified framework that bridges this gap by coupling pretrained event-sequence embeddings with a self-reflective LLM-driven feature generation agent. EAFD iteratively discovers, evaluates, and refines features directly from raw event sequences using two complementary criteria: \emph{alignment}, which explains information already encoded in embeddings, and \emph{complementarity}, which identifies predictive signals missing from them. Across both open-source and industrial transaction benchmarks, EAFD consistently outperforms embedding-only and feature-based baselines, achieving relative gains of up to $+5.8\%$ over state-of-the-art pretrained embeddings, resulting in new state-of-the-art performance across event-sequence datasets. 

---
# LLM-Driven Discovery of High-Entropy Catalysts via Retrieval-Augmented Generation 

**Authors**: AI Scientists, Xinyi Lin, Danqing Yin, Ying Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.15712)  

**Abstract**: CO2 reduction requires efficient catalysts, yet materials discovery remains bottlenecked by 10-20 year development cycles requiring deep domain expertise. This paper demonstrates how large language models can assist the catalyst discovery process by helping researchers explore chemical spaces and interpret results when augmented with retrieval-based grounding. We introduce a retrieval-augmented generation framework that enables GPT-4 to navigate chemical space by accessing a database of 50,000+ known materials, adapting general-purpose language understanding for high-throughput materials design. Our approach generated over 250 catalyst candidates with an 82% thermodynamic stability rate while addressing multi-objective constraints: 68% achieved <$100/kg cost with metallic conductivity (band gap<0.1eV) and mechanical stability (B/G>1.75). The best-performing Fe0.2Co0.2Ni0.2Ir0.1Ru0.3 achieves 0.285V limiting potential (25% improvement over IrO2), while Cr0.2Fe0.2Co0.3Ni0.2Mo0.1 optimally balances performance-cost trade-offs at $18/kg. Volcano plot analysis confirms that 78% of LLM-generated catalysts cluster near the theoretical activity optimum, while our system achieves 200x computational efficiency compared to traditional high-throughput screening. By demonstrating that retrieval-augmented generation can ground AI creativity in physical constraints without sacrificing exploration, this work demonstrates an approach where natural language interfaces can streamline materials discovery workflows, enabling researchers to explore chemical spaces more efficiently while the LLM assists in result interpretation and hypothesis generation. 

---
# Mastering the Minority: An Uncertainty-guided Multi-Expert Framework for Challenging-tailed Sequence Learning 

**Authors**: Ye Wang, Zixuan Wu, Lifeng Shen, Jiang Xie, Xiaoling Wang, Hong Yu, Guoyin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15708)  

**Abstract**: Imbalanced data distribution remains a critical challenge in sequential learning, leading models to easily recognize frequent categories while failing to detect minority classes adequately. The Mixture-of-Experts model offers a scalable solution, yet its application is often hindered by parameter inefficiency, poor expert specialization, and difficulty in resolving prediction conflicts. To Master the Minority classes effectively, we propose the Uncertainty-based Multi-Expert fusion network (UME) framework. UME is designed with three core innovations: First, we employ Ensemble LoRA for parameter-efficient modeling, significantly reducing the trainable parameter count. Second, we introduce Sequential Specialization guided by Dempster-Shafer Theory (DST), which ensures effective specialization on the challenging-tailed classes. Finally, an Uncertainty-Guided Fusion mechanism uses DST's certainty measures to dynamically weigh expert opinions, resolving conflicts by prioritizing the most confident expert for reliable final predictions. Extensive experiments across four public hierarchical text classification datasets demonstrate that UME achieves state-of-the-art performance. We achieve a performance gain of up to 17.97\% over the best baseline on individual categories, while reducing trainable parameters by up to 10.32\%. The findings highlight that uncertainty-guided expert coordination is a principled strategy for addressing challenging-tailed sequence learning. Our code is available at this https URL. 

---
# SEMAG: Self-Evolutionary Multi-Agent Code Generation 

**Authors**: Yulin Peng, Haowen Hou, Xinxin Zhu, Ying Tiffany He, F. Richard Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15707)  

**Abstract**: Large Language Models (LLMs) have made significant progress in handling complex programming tasks. However, current methods rely on manual model selection and fixed workflows, which limit their ability to adapt to changing task complexities. To address this, we propose SEMAG, a Self-Evolutionary Multi-Agent code Generation framework that mimics human coding practices. It decomposes programming tasks into stages, including planning, coding, debugging, and discussion, while adapting workflows to task difficulty. Its self-evolutionary agents can access the latest models in real time and automatically upgrade the backbone model. SEMAG sets new state-of-the-art Pass@1 accuracy across benchmarks. Using identical backbone models, SEMAG outperforms prior methods by 3.3% on CodeContests. When augmented with self-evolutionary model selection that automatically identifies optimal backbones, SEMAG reaches 52.6%, showcasing both framework effectiveness and adaptability to evolving LLM capabilities. 

---
# This Is Taking Too Long - Investigating Time as a Proxy for Energy Consumption of LLMs 

**Authors**: Lars Krupp, Daniel Geißler, Francisco M. Calatrava-Nicolas, Vishal Banwari, Paul Lukowicz, Jakob Karolus  

**Link**: [PDF](https://arxiv.org/pdf/2603.15699)  

**Abstract**: The energy consumption of Large Language Models (LLMs) is raising growing concerns due to their adverse effects on environmental stability and resource use. Yet, these energy costs remain largely opaque to users, especially when models are accessed through an API - a black box in which all information depends on what providers choose to disclose. In this work, we investigate inference time measurements as a proxy to approximate the associated energy costs of API-based LLMs. We ground our approach by comparing our estimations with actual energy measurements from locally hosted equivalents. Our results show that time measurements allow us to infer GPU models for API-based LLMs, grounding our energy cost estimations. Our work aims to create means for understanding the associated energy costs of API-based LLMs, especially for end users. 

---
# Tackling Over-smoothing on Hypergraphs: A Ricci Flow-guided Neural Diffusion Approach 

**Authors**: Mengyao Zhou, Zhiheng Zhou, Xiao Han, Xingqin Qi, Guanghui Wang, Guiying Yan  

**Link**: [PDF](https://arxiv.org/pdf/2603.15696)  

**Abstract**: Hypergraph neural networks (HGNNs) have demonstrated strong capabilities in modeling complex higher-order relationships. However, existing HGNNs often suffer from over-smoothing as the number of layers increases and lack effective control over message passing among nodes. Inspired by the theory of Ricci flow in differential geometry, we theoretically establish that introducing discrete Ricci flow into hypergraph structures can effectively regulate node feature evolution and thereby alleviate over-smoothing. Building on this insight, we propose Ricci Flow-guided Hypergraph Neural Diffusion(RFHND), a novel message passing paradigm for hypergraphs guided by discrete Ricci flow. Specifically, RFHND is based on a PDE system that describes the continuous evolution of node features on hypergraphs and adaptively regulates the rate of information diffusion at the geometric level, preventing feature homogenization and producing high-quality node representations. Experimental results show that RFHND significantly outperforms existing methods across multiple benchmark datasets and demonstrates strong robustness, while also effectively mitigating over-smoothing. 

---
# BadLLM-TG: A Backdoor Defender powered by LLM Trigger Generator 

**Authors**: Ruyi Zhang, Heng Gao, Songlei Jian, Yusong Tan, Haifang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.15692)  

**Abstract**: Backdoor attacks compromise model reliability by using triggers to manipulate outputs. Trigger inversion can accurately locate these triggers via a generator and is therefore critical for backdoor defense. However, the discrete nature of text prevents existing noise-based trigger generator from being applied to nature language processing (NLP). To overcome the limitations, we employ the rich knowledge embedded in large language models (LLMs) and propose a Backdoor defender powered by LLM Trigger Generator, termed BadLLM-TG. It is optimized through prompt-driven reinforcement learning, using the victim model's feedback loss as the reward signal. The generated triggers are then employed to mitigate the backdoor via adversarial training. Experiments show that our method reduces the attack success rate by 76.2\% on average, outperforming the second-best defender by 13.7. 

---
# Loosely-Structured Software: Engineering Context, Structure, and Evolution Entropy in Runtime-Rewired Multi-Agent Systems 

**Authors**: Weihao Zhang, Yitong Zhou, Huanyu Qu, Hongyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.15690)  

**Abstract**: As LLM-based multi-agent systems (MAS) become more autonomous, their free-form interactions increasingly dominate system behavior. However, scaling the number of agents often amplifies context pressure, coordination errors, and system drift. It is well known that building robust MAS requires more than prompt tuning or increased model intelligence. It necessitates engineering discipline focused on architecture to manage complexity under uncertainty. We characterize agentic software by a core property: \emph{runtime generation and evolution under uncertainty}. Drawing upon and extending software engineering experience, especially object-oriented programming, this paper introduces \emph{Loosely-Structured Software (LSS)}, a new class of software systems that shifts the engineering focus from constructing deterministic logic to managing the runtime entropy generated by View-constructed programming, semantic-driven self-organization, and endogenous evolution.
To make this entropy governable, we introduce design principles under a three-layer engineering framework: \emph{View/Context Engineering} to manage the execution environment and maintain task-relevant Views, \emph{Structure Engineering} to organize dynamic binding over artifacts and agents, and \emph{Evolution Engineering} to govern the lifecycle of self-rewriting artifacts. Building on this framework, we develop LSS design patterns as semantic control blocks that stabilize fluid, inference-mediated interactions while preserving agent adaptability. Together, these abstractions improve the \emph{designability}, \emph{scalability}, and \emph{evolvability} of agentic infrastructure. We provide basic experimental validation of key mechanisms, demonstrating the effectiveness of LSS. 

---
# Transition Flow Matching 

**Authors**: Chenrui Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.15689)  

**Abstract**: Mainstream flow matching methods typically focus on learning the local velocity field, which inherently requires multiple integration steps during generation. In contrast, Mean Velocity Flow models establish a relationship between the local velocity field and the global mean velocity, enabling the latter to be learned through a mathematically grounded formulation and allowing generation to be transferred to arbitrary future time points. In this work, we propose a new paradigm that directly learns the transition flow. As a global quantity, the transition flow naturally supports generation in a single step or at arbitrary time points. Furthermore, we demonstrate the connection between our approach and Mean Velocity Flow, establishing a unified theoretical perspective. Extensive experiments validate the effectiveness of our method and support our theoretical claims. 

---
# Evidential Domain Adaptation for Remaining Useful Life Prediction with Incomplete Degradation 

**Authors**: Yubo Hou, Mohamed Ragab, Yucheng Wang, Min Wu, Abdulla Alseiari, Chee-Keong Kwoh, Xiaoli Li, Zhenghua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.15687)  

**Abstract**: Accurate Remaining Useful Life (RUL) prediction without labeled target domain data is a critical challenge, and domain adaptation (DA) has been widely adopted to address it by transferring knowledge from a labeled source domain to an unlabeled target domain. Despite its success, existing DA methods struggle significantly when faced with incomplete degradation trajectories in the target domain, particularly due to the absence of late degradation stages. This missing data introduces a key extrapolation challenge. When applied to such incomplete RUL prediction tasks, current DA methods encounter two primary limitations. First, most DA approaches primarily focus on global alignment, which can misaligns late degradation stage in the source domain with early degradation stage in the target domain. Second, due to varying operating conditions in RUL prediction, degradation patterns may differ even within the same degradation stage, resulting in different learned features. As a result, even if degradation stages are partially aligned, simple feature matching cannot fully align two domains. To overcome these limitations, we propose a novel evidential adaptation approach called EviAdapt, which leverages evidential learning to enhance domain adaptation. The method first segments the source and target domain data into distinct degradation stages based on degradation rate, enabling stage-wise alignment that ensures samples from corresponding stages are accurately matched. To address the second limitation, we introduce an evidential uncertainty alignment technique that estimates uncertainty using evidential learning and aligns the uncertainty across matched stages. 

---
# DASH: Dynamic Audio-Driven Semantic Chunking for Efficient Omnimodal Token Compression 

**Authors**: Bingzhou Li, Tao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15685)  

**Abstract**: Omnimodal large language models (OmniLLMs) jointly process audio and visual streams, but the resulting long multimodal token sequences make inference prohibitively expensive. Existing compression methods typically rely on fixed window partitioning and attention-based pruning, which overlook the piecewise semantic structure of audio-visual signals and become fragile under aggressive token reduction. We propose Dynamic Audio-driven Semantic cHunking (DASH), a training-free framework that aligns token compression with semantic structure. DASH treats audio embeddings as a semantic anchor and detects boundary candidates via cosine-similarity discontinuities, inducing dynamic, variable-length segments that approximate the underlying piecewise-coherent organization of the sequence. These boundaries are projected onto video tokens to establish explicit cross-modal segmentation. Within each segment, token retention is determined by a tri-signal importance estimator that fuses structural boundary cues, representational distinctiveness, and attention-based salience, mitigating the sparsity bias of attention-only selection. This structure-aware allocation preserves transition-critical tokens while reducing redundant regions. Extensive experiments on AVUT, VideoMME, and WorldSense demonstrate that DASH maintains superior accuracy while achieving higher compression ratios compared to prior methods. Code is available at: this https URL. 

---
# State-Dependent Safety Failures in Multi-Turn Language Model Interaction 

**Authors**: Pengcheng Li, Jie Zhang, Tianwei Zhang, Han Qiu, Zhang kejun, Weiming Zhang, Nenghai Yu, Wenbo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.15684)  

**Abstract**: Safety alignment in large language models is typically evaluated under isolated queries, yet real-world use is inherently multi-turn. Although multi-turn jailbreaks are empirically effective, the structure of conversational safety failure remains insufficiently understood. In this work, we study safety failures from a state-space perspective and show that many multi-turn failures arise from structured contextual state evolution rather than isolated prompt vulnerabilities. We introduce STAR, a state-oriented diagnostic framework that treats dialogue history as a state transition operator and enables controlled analysis of safety behavior along interaction trajectories. Rather than optimizing attack strength, STAR provides a principled probe of how aligned models traverse the safety boundary under autoregressive conditioning. Across multiple frontier language models, we find that systems that appear robust under static evaluation can undergo rapid and reproducible safety collapse under structured multi-turn interaction. Mechanistic analysis reveals monotonic drift away from refusal-related representations and abrupt phase transitions induced by role-conditioned context. Together, these findings motivate viewing language model safety as a dynamic, state-dependent process defined over conversational trajectories. 

---
# IdentityGuard: Context-Aware Restriction and Provenance for Personalized Synthesis 

**Authors**: Lingyun Zhang, Yu Xie, Ping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.15679)  

**Abstract**: The nature of personalized text-to-image models poses a unique safety challenge that generic context-blind methods are ill-equipped to handle. Such global filters create a dilemma: to prevent misuse, they are forced to damage the model's broader utility by erasing concepts entirely, causing unacceptable collateral this http URL work presents a more precisely targeted approach, built on the principle that security should be as context-aware as the threat itself, intrinsically bound to the personalized concept. We present IDENTITYGUARD, which realizes this principle through a conditional restriction that blocks harmful content only when combined with the personalized identity, and a concept-specific watermark for precise traceability. Experiments show our approach prevents misuse while preserving the model's utility and enabling robust traceability. By moving beyond blunt, global filters, our work demonstrates a more effective and responsible path toward AI safety. 

---
# Spectral Edge Dynamics of Training Trajectories: Signal--Noise Geometry Across Scales 

**Authors**: Yongzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15678)  

**Abstract**: Despite hundreds of millions of parameters, transformer training trajectories evolve within only a few coherent directions. We introduce \emph{Spectral Edge Dynamics} (SED) to measure this structure: rolling-window SVD of parameter updates reveals a sharp boundary -- the \emph{spectral edge} -- between coherent optimization directions and stochastic noise, identified by the maximum consecutive singular value ratio $\sigma_k/\sigma_{k+1}$. Across a 51M-parameter TinyStories model (4~seeds) and GPT-2 124M under a distribution shift, the spectral edge exhibits a universal three-phase pattern (rise, plateau, collapse), signal rank adjusts with task complexity ($k^* = 2$ at 51M, $k^* = 3$ at 124M), and the directional coupling between spectral geometry and validation loss reverses with window size -- a \emph{lag flip} reflecting the timescale of trajectory integration. Johnson--Lindenstrauss projection to $d = 10W$ dimensions (e.g., $d = 100$ for $W = 10$) preserves the spectral gap within 5.7\%, making the framework applicable to models of arbitrary size. In companion work, the same spectral geometry provides early-warning signals of grokking -- predicting generalization 600--1{,}700 steps before it occurs across modular arithmetic, Dyck languages, and the SCAN benchmark. 

---
# Automated Self-Testing as a Quality Gate: Evidence-Driven Release Management for LLM Applications 

**Authors**: Alexandre Cristovão Maiorano  

**Link**: [PDF](https://arxiv.org/pdf/2603.15676)  

**Abstract**: LLM applications are AI systems whose non-deterministic outputs and evolving model behavior make traditional testing insufficient for release governance. We present an automated self-testing framework that introduces quality gates with evidence-based release decisions (PROMOTE/HOLD/ROLLBACK) across five empirically grounded dimensions: task success rate, research context preservation, P95 latency, safety pass rate, and evidence coverage. We evaluate the framework through a longitudinal case study of an internally deployed multi-agent conversational AI system with specific marketing capabilities in active development, covering 38 evaluation runs across 20+ internal releases. The gate identified two ROLLBACK-grade builds in early runs and supported stable quality evolution over a four-week staging lifecycle while exercising persona-grounded, multi-turn, adversarial, and evidence-required scenarios. Statistical analysis (Mann-Kendall trends, Spearman correlations, bootstrap confidence intervals), gate ablation, and overhead scaling indicate that evidence coverage is the primary severe-regression discriminator and that runtime scales predictably with suite size. A human calibration study (n=60 stratified cases, two independent evaluators, LLM-as-judge cross-validation) reveals complementary multi-modal coverage: LLM-judge disagreements with the system gate (kappa=0.13) are attributable to structural failure modes such as latency violations and routing errors that are invisible in response text alone, while the judge independently surfaces content quality failures missed by structural checks, validating the multi-dimensional gate design. The framework, supplementary pseudocode, and calibration artifacts are provided to support AI-system quality assurance and independent replication. 

---
# DRCY: Agentic Hardware Design Reviews 

**Authors**: Kyle Dumont, Nicholas Herbert, Hayder Tirmazi, Shrikanth Upadhayaya  

**Link**: [PDF](https://arxiv.org/pdf/2603.15672)  

**Abstract**: Hardware design errors discovered after fabrication require costly physical respins that can delay products by months. Existing electronic design automation (EDA) tools enforce structural connectivity rules. However, they cannot verify that connections are \emph{semantically} correct with respect to component datasheets. For example, that a symbol's pinout matches the manufacturer's specification, or that a voltage regulator's feedback resistors produce the intended output. We present DRCY, the first production-ready multi-agent LLM system that automates first-pass schematic connection review by autonomously fetching component datasheets, performing pin-by-pin analysis against extracted specifications, and posting findings as inline comments on design reviews. DRCY is deployed in production on AllSpice Hub, a collaborative hardware design platform, where it runs as a CI/CD action triggered on design review submissions. DRCY is used regularly by major hardware companies for use-cases ranging from multi-agent vehicle design to space exploration. We describe DRCY's five-agent pipeline architecture, its agentic datasheet retrieval system with self-evaluation, and its multi-run consensus mechanism for improving reliability on safety-critical analyses 

---
# Quantum Amplitude Estimation for Catastrophe Insurance Tail-Risk Pricing: Empirical Convergence and NISQ Noise Analysis 

**Authors**: Alexis Kirke  

**Link**: [PDF](https://arxiv.org/pdf/2603.15664)  

**Abstract**: Classical Monte Carlo methods for pricing catastrophe insurance tail risk converge at order reciprocal root N, requiring large simulation budgets to resolve upper-tail percentiles of the loss distribution. This sample-sparsity problem can lead to AI models trained on impoverished tail data, producing poorly calibrated risk estimates where insolvency risk is greatest. Quantum Amplitude Estimation (QAE), following Montanaro, achieves convergence approaching order reciprocal N in oracle queries - a quadratic speedup that, at scale, would enable high-resolution tail estimation within practical budgets. We validate this advantage empirically using a Qiskit Aer simulator with genuine Grover amplification. A complete pipeline encodes fitted lognormal catastrophe distributions into quantum oracles via amplitude encoding, producing small readout probabilities that enable safe Grover amplification with up to k=16 iterations. Seven experiments on synthetic and real (NOAA Storm Events, 58,028 records) data yield three main findings: an oracle-model advantage, that strong classical baselines win when analytical access is available, and that discretisation, not estimation, is the current bottleneck. 

---
# OrthoAI v2: From Single-Agent Segmentation to Dual-Agent Treatment Planning for Clear Aligners 

**Authors**: Lansiaux Edouard, Leman Margaux  

**Link**: [PDF](https://arxiv.org/pdf/2603.15663)  

**Abstract**: We present OrthoAI v2, the second iteration of our open-source pipeline for AI-assisted orthodontic treatment planning with clear aligners, substantially extending the single-agent framework previously introduced. The first version established a proof-of-concept based on Dynamic Graph Convolutional Neural Networks (\dgcnn{}) for tooth segmentation but was limited to per-tooth centroid extraction, lacked landmark-level precision, and produced a scalar quality score without staging simulation. \vtwo{} addresses all three limitations through three principal contributions: (i)~a second agent adopting the Conditioned Heatmap Regression Methodology (\charm{})~\cite{rodriguez2025charm} for direct, segmentation-free dental landmark detection, fused with Agent~1 via a confidence-weighted orchestrator in three modes (parallel, sequential, single-agent); (ii)~a composite six-category biomechanical scoring model (biomechanics $\times$ 0.30 + staging $\times$ 0.20 + attachments $\times$ 0.15 + IPR $\times$ 0.10 + occlusion $\times$ 0.10 + predictability $\times$ 0.15) replacing the binary pass/fail check of v1; (iii)~a multi-frame treatment simulator generating $F = A \times r$ temporally coherent 6-DoF tooth trajectories via SLERP interpolation and evidence-based staging rules, enabling ClinCheck 4D visualisation. On a synthetic benchmark of 200 crowding scenarios, the parallel ensemble of OrthoAI v2 reaches a planning quality score of $92.8 \pm 4.1$ vs.\ $76.4 \pm 8.3$ for OrthoAI v1, a $+21\%$ relative gain, while maintaining full CPU deployability ($4.2 \pm 0.8$~s). 

---
# Attribution-Guided Model Rectification of Unreliable Neural Network Behaviors 

**Authors**: Peiyu Yang, Naveed Akhtar, Jiantong Jiang, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2603.15656)  

**Abstract**: The performance of neural network models deteriorates due to their unreliable behavior on non-robust features of corrupted samples. Owing to their opaque nature, rectifying models to address this problem often necessitates arduous data cleaning and model retraining, resulting in huge computational and manual overhead. In this work, we leverage rank-one model editing to establish an attribution-guided model rectification framework that effectively locates and corrects model unreliable behaviors. We first distinguish our rectification setting from existing model editing, yielding a formulation that corrects unreliable behavior while preserving model performance and reducing reliance on large budgets of cleansed samples. We further reveal a bottleneck of model rectifying arising from heterogeneous editability across layers. To target the primary source of misbehavior, we introduce an attribution-guided layer localization method that quantifies layer-wise editability and identifies the layer most responsible for unreliabilities. Extensive experiments demonstrate the effectiveness of our method in correcting unreliabilities observed for neural Trojans, spurious correlations and feature leakage. Our method shows remarkable performance by achieving its editing objective with as few as a single cleansed sample, which makes it appealing for practice. 

---
# Beyond Reward Suppression: Reshaping Steganographic Communication Protocols in MARL via Dynamic Representational Circuit Breaking 

**Authors**: Liu Hung Ming  

**Link**: [PDF](https://arxiv.org/pdf/2603.15655)  

**Abstract**: In decentralized Multi-Agent Reinforcement Learning (MARL), steganographic collusion -- where agents develop private protocols to evade monitoring -- presents a critical AI safety threat. Existing defenses, limited to behavioral or reward layers, fail to detect coordination in latent communication channels. We introduce the Dynamic Representational Circuit Breaker (DRCB), an architectural defense operating at the optimization substrate.
Building on the AI Mother Tongue (AIM) framework, DRCB utilizes a Vector Quantized Variational Autoencoder (VQ-VAE) bottleneck to convert unobservable messages into auditable statistical objects. DRCB monitors signals including Jensen-Shannon Divergence drift, L2-norm codebook displacement, and Randomized Observer Pool accuracy to compute an EMA-based Collusion Score. Threshold breaches trigger four escalating interventions: dynamic adaptation, gradient-space penalty injection into the Advantage function A^pi, temporal reward suppression, and full substrate circuit breaking via codebook shuffling and optimizer state reset.
Experiments on a Contextual Prisoner's Dilemma with MNIST labels show that while static monitoring fails (p = 0.3517), DRCB improves observer mean accuracy from 0.858 to 0.938 (+9.3 percent) and reduces volatility by 43 percent, while preserving mean joint reward (p = 0.854). Analysis of 214,298 symbol samples confirms "Semantic Degradation," where high-frequency sequences converge to zero entropy, foreclosing complex steganographic encodings. We identify a "Transparency Paradox" where agents achieve surface-level determinism while preserving residual capacity in long-tail distributions, reflecting Goodhart's Law. This task-agnostic methodology provides a technical path toward MICA-compliant (Multi-Agent Internal Coupling Audit) pre-deployment auditing for autonomous systems. 

---
# Discovering the Hidden Role of Gini Index In Prompt-based Classification 

**Authors**: Ruixi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.15654)  

**Abstract**: In classification tasks, the long-tailed minority classes usually offer the predictions that are most important. Yet these classes consistently exhibit low accuracies, whereas a few high-performing classes dominate the game. We pursue a foundational understanding of the hidden role of Gini Index as a tool for detecting and optimizing (debiasing) disparities in class accuracy, focusing on the case of prompt-based classification. We introduce the intuitions, benchmark Gini scores in real-world LLMs and vision models, and thoroughly discuss the insights of Gini not only as a measure of relative accuracy dominance but also as a direct optimization metric. Through rigorous case analyses, we first show that weak to strong relative accuracy imbalance exists in both prompt-based, text and image classification results and regardless of whether the classification is high-dimensional or low-dimensional. Then, we harness the Gini metric to propose a post-hoc model-agnostic bias mitigation method. Experimental results across few-shot news, biomedical, and zero-shot image classification show that our method significantly reduces both relative and absolute accuracy imbalances, minimizing top class relative dominance while elevating weakest classes. 

---
# Recursive Language Models Meet Uncertainty: The Surprising Effectiveness of Self-Reflective Program Search for Long Context 

**Authors**: Keivan Alizadeh, Parshin Shojaee, Minsik Cho, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2603.15653)  

**Abstract**: Long-context handling remains a core challenge for language models: even with extended context windows, models often fail to reliably extract, reason over, and use the information across long contexts. Recent works like Recursive Language Models (RLM) have approached this challenge by agentic way of decomposing long contexts into recursive sub-calls through programmatic interaction at inference. While promising, the success of RLM critically depends on how these context-interaction programs are selected, which has remained largely unexplored. In this paper, we study this problem and introduce SRLM, a framework that augments programmatic context interaction with uncertainty-aware Self-Reflection. SRLM leverages three intrinsic signals: self consistency, reasoning length, and verbalized confidence. These serve as complementary indicators of a model's internal uncertainty, and the model uses them to evaluate and compare candidate context-interaction programs. Extensive experiments across diverse benchmark datasets, context lengths, and backbone models, show that SRLM consistently outperforms state-of-the-art baselines, yielding up to 22% improvement over RLM under the same time budget. Our findings show that recursion itself is not the primary driver of performance in RLM, and a simple self-reflective program search can match or surpass RLM without requiring self-query or explicit recursion mechanisms. We find that for context lengths within the model's window, RLMs with recursion often degrade performance relative to the base model, whereas SRLM yields consistent gains across both short and long contexts. We also find that RLM is less effective in tasks with semantically intensive nature, where heuristic program search is insufficient and broader contextual understanding is required, while self-reflection in SRLM provides a semantic signal that better steers reasoning in these scenarios. 

---
# A federated learning framework with knowledge graph and temporal transformer for early sepsis prediction in multi-center ICUs 

**Authors**: Yue Chang, Guangsen Lin, Jyun Jie Chuang, Shunqi Liu, Xinkui Li, Yaozheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.15651)  

**Abstract**: The early prediction of sepsis in intensive care unit (ICU) patients is crucial for improving survival rates. However, the development of accurate predictive models is hampered by data fragmentation across healthcare institutions and the complex, temporal nature of medical data, all under stringent privacy constraints. To address these challenges, we propose a novel framework that uniquely integrates federated learning (FL) with a medical knowledge graph and a temporal transformer model, enhanced by meta-learning capabilities. Our approach enables collaborative model training across multiple hospitals without sharing raw patient data, thereby preserving privacy. The model leverages a knowledge graph to incorporate structured medical relationships and employs a temporal transformer to capture long-range dependencies in clinical time-series data. A model-agnostic meta-learning (MAML) strategy is further incorporated to facilitate rapid adaptation of the global model to local data distributions. Evaluated on the MIMIC-IV and eICU datasets, our method achieves an area under the curve (AUC) of 0.956, which represents a 22.4% improvement over conventional centralized models and a 12.7% improvement over standard federated learning, demonstrating strong predictive capability for sepsis. This work presents a reliable and privacy-preserving solution for multi-center collaborative early warning of sepsis. 

---
# Steering Frozen LLMs: Adaptive Social Alignment via Online Prompt Routing 

**Authors**: Zeyu Zhang, Xiangxiang Dai, Ziyi Han, Xutong Liu, John C.S. Lui  

**Link**: [PDF](https://arxiv.org/pdf/2603.15647)  

**Abstract**: Large language models (LLMs) are typically governed by post-training alignment (e.g., RLHF or DPO), which yields a largely static policy during deployment and inference. However, real-world safety is a full-lifecycle problem: static defenses degrade against evolving jailbreak behaviors, and fixed weights cannot adapt to pluralistic, time-varying safety norms. This motivates inference-time governance that steers behavior without costly retraining. To address this, we introduce the Consensus Clustering LinUCB Bandit (CCLUB), a unified framework for adaptive social alignment via system-prompt routing. CCLUB employs a conservative consensus clustering mechanism: it pools data only within the intersection of utility and safety similarity graphs, effectively preventing unsafe generalization across semantically proximal but risk-divergent contexts. Our theoretical analysis yields a sublinear regret guarantee, demonstrating near-optimal performance of CCLUB. Extensive experiments validate that CCLUB outperforms strong baselines, achieving a 10.98% improvement in cumulative reward and a 14.42% reduction in the average suboptimality gap. 

---
# Alternating Reinforcement Learning with Contextual Rubric Rewards 

**Authors**: Guangchen Lan  

**Link**: [PDF](https://arxiv.org/pdf/2603.15646)  

**Abstract**: Reinforcement Learning with Rubric Rewards (RLRR) is a framework that extends conventional reinforcement learning from human feedback (RLHF) and verifiable rewards (RLVR) by replacing scalar preference signals with structured, multi-dimensional, contextual rubric-based evaluations. However, existing approaches in RLRR are limited to linearly compressing vector rewards into a scalar reward with a fixed weightings, which is sensitive to artificial score design and fails to capture correlations among reward dimensions. To overcome the limitations of reward aggregation, this work proposes Alternating Reinforcement Learning with Rubric Rewards (ARL-RR), a framework that eliminates the need for a fixed scalarization by optimizing one semantic rubric meta-class at a time. Theoretically, we show that reward aggregation induces a variance contraction effect, which helps explain the performance gains. We further introduce a lightweight, search-based adaptation procedure that selects the next meta-class dynamically based on task performance, enabling the policy to emphasize critical objectives and thereby improve the model performance. Empirically, our experiments on the HealthBench dataset with experts annotations demonstrate that ARL-RR uniformly outperforms scalarized methods in both model performance and training efficiency across different model scales (1.7B, 4B, 8B, and 14B). 

---
# XLinear: Frequency-Enhanced MLP with CrossFilter for Robust Long-Range Forecasting 

**Authors**: Xiang Ao  

**Link**: [PDF](https://arxiv.org/pdf/2603.15645)  

**Abstract**: Time series forecasters are widely used across various domains. Among them, MLP (multi-layer perceptron)-based forecasters have been proven to be more robust to noise compared to Transformer-based forecasters. However, MLP struggles to capture complex features, resulting in limitations on capturing long-range dependencies. To address this challenge, we propose XLinear, an MLP-based forecaster for long-range forecasting. Firstly, we decompose the time series into trend and seasonal components. For the trend component which contains long-range characteristics, we design Enhanced Frequency Attention (EFA) to capture long-term dependencies by leveraging frequency-domain operations. Additionally, a CrossFilter Block is proposed for the seasonal component to maintain the model's robustness to noise, avoiding the problems of low robustness often caused by attention mechanisms. Experimental results demonstrate that XLinear achieves state-of-the-art performance on test datasets. While keeping the lightweight architecture and high robustness of MLP-based models, our forecaster outperforms other MLP-based forecasters in capturing long-range dependencies. 

---
# Exploring the Use of VLMs for Navigation Assistance for People with Blindness and Low Vision 

**Authors**: Yu Li, Yuchen Zheng, Giles Hamilton-Fletcher, Marco Mezzavilla, Yao Wang, Sundeep Rangan, Maurizio Porfiri, Zhou Yu, John-Ross Rizzo  

**Link**: [PDF](https://arxiv.org/pdf/2603.15624)  

**Abstract**: This paper investigates the potential of vision-language models (VLMs) to assist people with blindness and low vision (pBLV) in navigation tasks. We evaluate state-of-the-art closed-source models, including GPT-4V, GPT-4o, Gemini-1.5-Pro, and Claude-3.5-Sonnet, alongside open-source models, such as Llava-v1.6-mistral and Llava-onevision-qwen, to analyze their capabilities in foundational visual skills: counting ambient obstacles, relative spatial reasoning, and common-sense wayfinding-pertinent scene understanding. We further assess their performance in navigation scenarios, using pBLV-specific prompts designed to simulate real-world assistance tasks. Our findings reveal notable performance disparities between these models: GPT-4o consistently outperforms others across all tasks, particularly in spatial reasoning and scene understanding. In contrast, open-source models struggle with nuanced reasoning and adaptability in complex environments. Common challenges include difficulties in accurately counting objects in cluttered settings, biases in spatial reasoning, and a tendency to prioritize object details over spatial feedback, limiting their usability for pBLV in navigation tasks. Despite these limitations, VLMs show promise for wayfinding assistance when better aligned with human feedback and equipped with improved spatial reasoning. This research provides actionable insights into the strengths and limitations of current VLMs, guiding developers on effectively integrating VLMs into assistive technologies while addressing key limitations for enhanced usability. 

---
# Finder: A Multimodal AI-Powered Search Framework for Pharmaceutical Data Retrieval 

**Authors**: Suyash Mishra, Srikanth Patil, Satyanarayan Pati, Sagar Sahu, Baddu Narendra  

**Link**: [PDF](https://arxiv.org/pdf/2603.15623)  

**Abstract**: AI is transforming pharmaceutical search, where traditional systems struggle with multimodal content and manual curation. Finder is a scalable AI-powered framework that unifies retrieval across text, images, audio, and video using hybrid vector search, combining sparse lexical and dense semantic models. Its modular pipeline ingests diverse formats, enriches metadata, and stores content in a vector-native backend. Finder supports reasoning-aware natural language search, improving precision and contextual relevance. The system has processed over 291,400 documents, 31,070 videos, and 1,192 audio files in 98 languages. Techniques like hybrid fusion, chunking, and metadata-aware routing enable intelligent access across regulatory, research, and commercial domains. 

---
# SAC-NeRF: Adaptive Ray Sampling for Neural Radiance Fields via Soft Actor-Critic Reinforcement Learning 

**Authors**: Chenyu Ge  

**Link**: [PDF](https://arxiv.org/pdf/2603.15622)  

**Abstract**: Neural Radiance Fields (NeRF) have achieved photorealistic novel view synthesis but suffer from computational inefficiency due to dense ray sampling during volume rendering. We propose SAC-NeRF, a reinforcement learning framework that learns adaptive sampling policies using Soft Actor-Critic (SAC). Our method formulates sampling as a Markov Decision Process where an RL agent learns to allocate samples based on scene characteristics. We introduce three technical components: (1) a Gaussian mixture distribution color model providing uncertainty estimates, (2) a multi-component reward function balancing quality, efficiency, and consistency, and (3) a two-stage training strategy addressing environment non-stationarity. Experiments on Synthetic-NeRF and LLFF datasets show that SAC-NeRF reduces sampling points by 35-48\% while maintaining rendering quality within 0.3-0.8 dB PSNR of dense sampling baselines. While the learned policy is scene-specific and the RL framework adds complexity compared to simpler heuristics, our work demonstrates that data-driven sampling strategies can discover effective patterns that would be difficult to hand-design. 

---
# One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers 

**Authors**: Lennon J. Shikhman  

**Link**: [PDF](https://arxiv.org/pdf/2603.01406)  

**Abstract**: Neural PDE solvers are often described as learning solution operators that map problem data to PDE solutions. In this work, we argue that this interpretation is generally incorrect when boundary conditions vary. We show that standard neural operator training implicitly learns a boundary-indexed family of operators, rather than a single boundary-agnostic operator, with the learned mapping fundamentally conditioned on the boundary-condition distribution seen during training. We formalize this perspective by framing operator learning as conditional risk minimization over boundary conditions, which leads to a non-identifiability result outside the support of the training boundary distribution. As a consequence, generalization in forcing terms or resolution does not imply generalization across boundary conditions. We support our theoretical analysis with controlled experiments on the Poisson equation, demonstrating sharp degradation under boundary-condition shifts, cross-distribution failures between distinct boundary ensembles, and convergence to conditional expectations when boundary information is removed. Our results clarify a core limitation of current neural PDE solvers and highlight the need for explicit boundary-aware modeling in the pursuit of foundation models for PDEs. 

---
